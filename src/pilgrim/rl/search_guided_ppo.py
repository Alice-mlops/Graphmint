# Implements search-guided PPO for policy priors and value heuristics on Cayley graphs.
"""Trainer for beam-guided PPO with auxiliary supervision archives."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from cayleypy import CayleyGraph
from torch import nn

from pilgrim.schemas.rl.search_guided_ppo import (
    SearchGuidedPPOConfig,
    SearchGuidedPPOLossState,
    SearchGuidedPPOMetrics,
    SearchGuidedPPOStepDiagnostics,
)
from pilgrim.utils.lr_scheduler_utils import lr_scheduler_ctor_from_cfg

from .helpers.ppo import (
    compute_supervised_policy_value_losses,
    evaluate_policy_actions,
    forward_policy_value,
    normalize_advantages,
    sample_policy_actions,
)
from .helpers.q_learning import _graph_device, _resolve_graph_inverse_map, apply_actions
from .helpers.search_guidance import (
    BeamSearchTargetSet,
    BeamSearchTargetStats,
    beam_action_reward_bonus,
    collect_beam_search_targets,
)
from .helpers.trajectory_supervision import (
    sample_reverse_trajectory_supervision_from_random_walks,
)
from .rollout_buffer import PolicyRolloutBatch, PolicyRolloutBuffer
from .sampling import sample_states_from_random_walks
from .search_guided_ppo_tracking import SearchGuidedPPOTracker
from .supervision_archive import PolicySupervisionArchive, PolicySupervisionBatch
from .transitions import central_state_mask

_ZERO_EPS = 1e-12


@dataclass(slots=True)
class _RolloutSummary:
    """Internal summary of one PPO rollout collection pass."""

    rollout_batch: PolicyRolloutBatch
    solve_rate: float
    mean_reward: float
    reward_step_cost_mean: float
    reward_solve_bonus_mean: float
    reward_teacher_progress_mean: float
    reward_inverse_penalty_mean: float
    reward_revisit_penalty_mean: float
    reward_search_bonus_mean: float
    beam_rollout_queries: int
    beam_rollout_successes: int
    beam_archive_queries: int
    beam_archive_successes: int


class SearchGuidedPPOTrainer:
    """
    Train an actor-critic with PPO and beam-guided auxiliary supervision.

    Args:
        model: Actor-critic model exposing policy logits and a scalar value head.
        graph: Cayley graph defining transitions.
        config: Trainer configuration.
        teacher_model: Optional frozen teacher model used for progress shaping.
        optimizer: Optional optimizer override.
        tracker: Optional side-effect tracker.

    Raises:
        ValueError: If the model does not expose compatible policy/value heads.

    """

    def __init__(
        self,
        model: nn.Module,
        graph: CayleyGraph,
        config: SearchGuidedPPOConfig | None = None,
        *,
        teacher_model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        tracker: SearchGuidedPPOTracker | None = None,
    ) -> None:
        self.config = config or SearchGuidedPPOConfig()
        self.graph = graph
        self.tracker = tracker

        self.device = self._resolve_device()
        self.model = model.to(self.device)
        _ = forward_policy_value(
            self.model,
            torch.as_tensor(graph.central_state, device=self.device).view(1, -1).long(),
        )
        self.teacher_model = (
            None if teacher_model is None else teacher_model.to(self.device)
        )
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)

        self.optimizer = optimizer or self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        self._step = 0
        self._demo_sample_index = 0
        self._start_state_sample_index = 0
        self._archive_sample_index = 0
        self._rollout_generator = torch.Generator(device="cpu")
        self._rollout_generator.manual_seed(int(self.config.seed))

        self.demo_archive = (
            None
            if not bool(self.config.demo_supervision.enabled)
            else PolicySupervisionArchive(
                capacity=int(self.config.demo_supervision.capacity),
                storage_device="cpu",
            )
        )
        self.search_archive = (
            None
            if not bool(self.config.beam_search.enabled)
            else PolicySupervisionArchive(
                capacity=int(self.config.beam_search.archive_capacity),
                storage_device="cpu",
            )
        )
        self.inverse_map = _resolve_graph_inverse_map(
            self.graph,
            device=_graph_device(self.graph),
        )

    def fit(
        self,
        num_updates: int | None = None,
    ) -> list[SearchGuidedPPOMetrics]:
        """
        Run the configured number of PPO updates.

        Args:
            num_updates: Optional override for the number of updates.

        Returns:
            History of compact PPO metrics.

        """
        if self.tracker is not None:
            self.tracker.on_fit_start(self)

        if (
            self.demo_archive is not None
            and int(self.config.demo_supervision.warmstart_size) > 0
            and len(self.demo_archive)
            < int(self.config.demo_supervision.warmstart_size)
        ):
            self.populate_demo_archive(int(self.config.demo_supervision.warmstart_size))

        update_count = (
            int(self.config.num_updates) if num_updates is None else int(num_updates)
        )
        history: list[SearchGuidedPPOMetrics] = []
        for _ in range(update_count):
            metrics, diagnostics = self.train_step()
            history.append(metrics)
            if self.tracker is not None:
                self.tracker.on_train_step_end(self, diagnostics)

        if self.tracker is not None:
            self.tracker.on_fit_end(self, history)
        return history

    def train_step(  # noqa: PLR0914
        self,
    ) -> tuple[SearchGuidedPPOMetrics, SearchGuidedPPOStepDiagnostics]:
        """
        Execute one rollout PPO or beam-distillation optimization update.

        Returns:
            Tuple ``(metrics, diagnostics)`` for the completed update.

        """
        step_started = time.perf_counter()
        self._step += 1

        rollout_started = time.perf_counter()
        if bool(self.config.rollout.enabled):
            rollout_summary = self.collect_rollout()
            rollout_size = len(rollout_summary.rollout_batch)
            solve_rate = float(rollout_summary.solve_rate)
            mean_reward = float(rollout_summary.mean_reward)
            reward_step_cost_mean = float(rollout_summary.reward_step_cost_mean)
            reward_solve_bonus_mean = float(rollout_summary.reward_solve_bonus_mean)
            reward_teacher_progress_mean = float(
                rollout_summary.reward_teacher_progress_mean
            )
            reward_inverse_penalty_mean = float(
                rollout_summary.reward_inverse_penalty_mean
            )
            reward_revisit_penalty_mean = float(
                rollout_summary.reward_revisit_penalty_mean
            )
            reward_search_bonus_mean = float(rollout_summary.reward_search_bonus_mean)
            beam_rollout_queries = int(rollout_summary.beam_rollout_queries)
            beam_rollout_successes = int(rollout_summary.beam_rollout_successes)
            beam_archive_queries = int(rollout_summary.beam_archive_queries)
            beam_archive_successes = int(rollout_summary.beam_archive_successes)
        else:
            archive_target_stats = self._refresh_search_archive_from_sampled_states()
            rollout_summary = None
            rollout_size = 0
            solve_rate = 0.0
            mean_reward = 0.0
            reward_step_cost_mean = 0.0
            reward_solve_bonus_mean = 0.0
            reward_teacher_progress_mean = 0.0
            reward_inverse_penalty_mean = 0.0
            reward_revisit_penalty_mean = 0.0
            reward_search_bonus_mean = 0.0
            beam_rollout_queries = 0
            beam_rollout_successes = 0
            beam_archive_queries = int(archive_target_stats.queried)
            beam_archive_successes = int(archive_target_stats.path_found)
        rollout_collect_time_s = time.perf_counter() - rollout_started

        optimize_started = time.perf_counter()
        if rollout_summary is None:
            optimize_stats = self._optimize_auxiliary_only()
        else:
            optimize_stats = self._optimize_rollout(rollout_summary.rollout_batch)
        optimize_time_s = time.perf_counter() - optimize_started

        if (
            self.demo_archive is not None
            and int(self.config.demo_supervision.refresh_size) > 0
        ):
            self.populate_demo_archive(int(self.config.demo_supervision.refresh_size))

        metrics = SearchGuidedPPOMetrics(
            step=int(self._step),
            total_loss=float(optimize_stats["total_loss"]),
            policy_loss=float(optimize_stats["policy_loss"]),
            value_loss=float(optimize_stats["value_loss"]),
            entropy=float(optimize_stats["entropy"]),
            auxiliary_loss=float(optimize_stats["auxiliary_loss"]),
            rollout_size=rollout_size,
            solve_rate=solve_rate,
            mean_reward=mean_reward,
            demo_archive_size=0
            if self.demo_archive is None
            else len(self.demo_archive),
            search_archive_size=0
            if self.search_archive is None
            else len(self.search_archive),
        )
        diagnostics = SearchGuidedPPOStepDiagnostics(
            step=int(self._step),
            total_loss=float(optimize_stats["total_loss"]),
            policy_loss=float(optimize_stats["policy_loss"]),
            value_loss=float(optimize_stats["value_loss"]),
            entropy=float(optimize_stats["entropy"]),
            auxiliary_loss=float(optimize_stats["auxiliary_loss"]),
            approx_kl=float(optimize_stats["approx_kl"]),
            clip_fraction=float(optimize_stats["clip_fraction"]),
            rollout_size=rollout_size,
            rollout_collect_time_s=float(rollout_collect_time_s),
            optimize_time_s=float(optimize_time_s),
            step_time_s=float(time.perf_counter() - step_started),
            solve_rate=solve_rate,
            mean_reward=mean_reward,
            reward_step_cost_mean=reward_step_cost_mean,
            reward_solve_bonus_mean=reward_solve_bonus_mean,
            reward_teacher_progress_mean=reward_teacher_progress_mean,
            reward_inverse_penalty_mean=reward_inverse_penalty_mean,
            reward_revisit_penalty_mean=reward_revisit_penalty_mean,
            reward_search_bonus_mean=reward_search_bonus_mean,
            demo_archive_size=0
            if self.demo_archive is None
            else len(self.demo_archive),
            search_archive_size=0
            if self.search_archive is None
            else len(self.search_archive),
            beam_rollout_queries=beam_rollout_queries,
            beam_rollout_successes=beam_rollout_successes,
            beam_archive_queries=beam_archive_queries,
            beam_archive_successes=beam_archive_successes,
        )
        return metrics, diagnostics

    def populate_demo_archive(self, num_rows: int) -> int:
        """
        Append reverse-trajectory supervision rows to the demo archive.

        Args:
            num_rows: Target number of new rows to append.

        Returns:
            Number of rows written into the archive.

        """
        if self.demo_archive is None or int(num_rows) <= 0:
            return 0
        written = 0
        while written < int(num_rows):
            reverse_batch = sample_reverse_trajectory_supervision_from_random_walks(
                self.graph,
                self.config.demo_supervision.sampling,
                sample_index=self._demo_sample_index,
            )
            self._demo_sample_index += 1
            supervision = PolicySupervisionBatch(
                states=reverse_batch.states.detach().cpu(),
                action_targets=reverse_batch.action_targets.detach().cpu(),
                value_targets=reverse_batch.return_steps.detach().float().cpu(),
                weights=torch.ones(len(reverse_batch), dtype=torch.float32),
            )
            written += self.demo_archive.add(supervision)
        return written

    def collect_rollout(self) -> _RolloutSummary:  # noqa: PLR0914, PLR0915
        """
        Collect one fixed-horizon on-policy rollout and refresh search targets.

        Returns:
            Internal summary of the collected rollout.

        """
        was_training = self.model.training
        self.model.eval()
        try:
            start_states = self._sample_rollout_start_states()
            num_envs = int(self.config.rollout.num_envs)
            rollout_buffer = PolicyRolloutBuffer(
                num_envs=num_envs,
                horizon=int(self.config.rollout.horizon),
                storage_device="cpu",
            )
            current_states = start_states.clone()
            current_hashes = self.graph.hasher.make_hashes(
                current_states.to(_graph_device(self.graph))
            )
            seen_hash_layers = [current_hashes.detach().clone()]
            previous_actions = torch.full(
                (num_envs,),
                fill_value=-1,
                device=current_states.device,
                dtype=torch.long,
            )
            episode_done = torch.zeros(
                num_envs, device=current_states.device, dtype=torch.bool
            )
            episode_steps = torch.zeros(
                num_envs, device=current_states.device, dtype=torch.long
            )
            solved_mask = torch.zeros(
                num_envs, device=current_states.device, dtype=torch.bool
            )

            rollout_target_set, rollout_target_stats = (
                self._collect_rollout_start_targets(current_states)
            )
            if rollout_target_set is not None and self.search_archive is not None:
                self.search_archive.add(rollout_target_set.batch)

            reward_component_sums = {
                "step_cost": 0.0,
                "solve_bonus": 0.0,
                "teacher_progress": 0.0,
                "inverse_penalty": 0.0,
                "revisit_penalty": 0.0,
                "search_bonus": 0.0,
            }
            valid_transition_count = 0

            for step_index in range(int(self.config.rollout.horizon)):
                valid_mask = (~episode_done) & (
                    episode_steps < int(self.config.rollout.max_episode_steps)
                )
                step_states = current_states.detach().clone()
                step_actions = torch.zeros(
                    num_envs,
                    device=current_states.device,
                    dtype=torch.long,
                )
                step_log_probs = torch.zeros(
                    num_envs,
                    device=current_states.device,
                    dtype=torch.float32,
                )
                step_values = torch.zeros(
                    num_envs,
                    device=current_states.device,
                    dtype=torch.float32,
                )
                step_rewards = torch.zeros(
                    num_envs,
                    device=current_states.device,
                    dtype=torch.float32,
                )
                step_done = torch.ones(
                    num_envs,
                    device=current_states.device,
                    dtype=torch.bool,
                )

                if bool(valid_mask.any()):
                    active_states = current_states[valid_mask]
                    active_actions, active_log_probs, active_values = (
                        sample_policy_actions(
                            self.model,
                            active_states,
                            generator_indices=self.config.rollout.generator_indices,
                            action_temperature=float(
                                self.config.rollout.action_temperature
                            ),
                            generator=self._rollout_generator,
                        )
                    )
                    next_states = apply_actions(
                        self.graph, active_states, active_actions
                    )
                    reached_center = central_state_mask(
                        next_states,
                        self.graph.central_state,
                    )
                    truncated = (episode_steps[valid_mask] + 1) >= int(
                        self.config.rollout.max_episode_steps
                    )
                    active_done = reached_center | truncated

                    step_actions[valid_mask] = active_actions
                    step_log_probs[valid_mask] = active_log_probs
                    step_values[valid_mask] = active_values

                    reward = torch.full(
                        (int(active_states.shape[0]),),
                        fill_value=float(self.config.reward.step_cost),
                        device=current_states.device,
                        dtype=torch.float32,
                    )
                    reward_component_sums["step_cost"] += float(
                        reward.sum().detach().item()
                    )

                    solve_bonus = reached_center.float() * float(
                        self.config.reward.solve_bonus
                    )
                    reward += solve_bonus
                    reward_component_sums["solve_bonus"] += float(
                        solve_bonus.sum().detach().item()
                    )

                    teacher_progress = self._teacher_progress_reward(
                        active_states,
                        next_states,
                    )
                    reward += teacher_progress
                    reward_component_sums["teacher_progress"] += float(
                        teacher_progress.sum().detach().item()
                    )

                    inverse_penalty = self._inverse_action_penalty(
                        previous_actions[valid_mask],
                        active_actions,
                    )
                    reward += inverse_penalty
                    reward_component_sums["inverse_penalty"] += float(
                        inverse_penalty.sum().detach().item()
                    )

                    revisit_penalty, next_hashes = self._revisit_penalty(
                        next_states,
                        valid_mask=valid_mask,
                        seen_hash_layers=seen_hash_layers,
                        current_hashes=current_hashes,
                    )
                    reward += revisit_penalty
                    reward_component_sums["revisit_penalty"] += float(
                        revisit_penalty.sum().detach().item()
                    )

                    if step_index == 0:
                        full_search_bonus = beam_action_reward_bonus(
                            actions=step_actions,
                            target_set=rollout_target_set,
                            batch_size=num_envs,
                            match_bonus=float(self.config.reward.search_match_bonus),
                            miss_penalty=float(self.config.reward.search_miss_penalty),
                            device=current_states.device,
                        )
                        reward += full_search_bonus[valid_mask]
                        reward_component_sums["search_bonus"] += float(
                            full_search_bonus[valid_mask].sum().detach().item()
                        )

                    step_rewards[valid_mask] = reward
                    step_done[valid_mask] = active_done

                    current_states[valid_mask] = next_states
                    current_hashes[valid_mask] = next_hashes
                    previous_actions[valid_mask] = active_actions
                    episode_steps[valid_mask] += 1
                    episode_done[valid_mask] = active_done
                    solved_mask[valid_mask] |= reached_center
                    seen_hash_layers.append(current_hashes.detach().clone())
                    valid_transition_count += int(valid_mask.sum().item())

                rollout_buffer.add_step(
                    states=step_states.detach().cpu(),
                    actions=step_actions.detach().cpu(),
                    log_probs=step_log_probs.detach().cpu(),
                    values=step_values.detach().cpu(),
                    rewards=step_rewards.detach().cpu(),
                    done=step_done.detach().cpu(),
                    valid_mask=valid_mask.detach().cpu(),
                )

            last_values = torch.zeros(
                num_envs,
                device=self.device,
                dtype=torch.float32,
            )
            still_active = (~episode_done) & (
                episode_steps < int(self.config.rollout.max_episode_steps)
            )
            if bool(still_active.any()):
                last_values[still_active] = forward_policy_value(
                    self.model,
                    current_states[still_active],
                ).values
            rollout_batch = rollout_buffer.finalize(
                last_values=last_values.detach().cpu(),
                discount=float(self.config.discount),
                gae_lambda=float(self.config.gae_lambda),
            )

            archive_target_stats = self._refresh_search_archive_from_rollout(
                rollout_batch.states
            )
            mean_reward = 0.0
            if valid_transition_count > 0:
                mean_reward = float(rollout_batch.rewards.mean().item())
            return _RolloutSummary(
                rollout_batch=rollout_batch,
                solve_rate=float(solved_mask.float().mean().item()),
                mean_reward=mean_reward,
                reward_step_cost_mean=self._mean_reward_component(
                    reward_component_sums["step_cost"],
                    valid_transition_count,
                ),
                reward_solve_bonus_mean=self._mean_reward_component(
                    reward_component_sums["solve_bonus"],
                    valid_transition_count,
                ),
                reward_teacher_progress_mean=self._mean_reward_component(
                    reward_component_sums["teacher_progress"],
                    valid_transition_count,
                ),
                reward_inverse_penalty_mean=self._mean_reward_component(
                    reward_component_sums["inverse_penalty"],
                    valid_transition_count,
                ),
                reward_revisit_penalty_mean=self._mean_reward_component(
                    reward_component_sums["revisit_penalty"],
                    valid_transition_count,
                ),
                reward_search_bonus_mean=self._mean_reward_component(
                    reward_component_sums["search_bonus"],
                    valid_transition_count,
                ),
                beam_rollout_queries=int(rollout_target_stats.queried),
                beam_rollout_successes=int(rollout_target_stats.path_found),
                beam_archive_queries=int(archive_target_stats.queried),
                beam_archive_successes=int(archive_target_stats.path_found),
            )
        finally:
            self.model.train(was_training)

    def _optimize_rollout(self, rollout_batch: PolicyRolloutBatch) -> dict[str, float]:
        """
        Run the PPO optimization epochs over one rollout batch.

        Args:
            rollout_batch: Flattened rollout batch.

        Returns:
            Mean optimization statistics across all completed minibatches.

        """
        batch = rollout_batch.to(self.device)
        advantages = normalize_advantages(batch.advantages)
        statistics = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "auxiliary_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "num_minibatches": 0.0,
        }

        stop_early = False
        for _ in range(int(self.config.num_policy_epochs)):
            permutation = torch.randperm(
                len(batch),
                generator=self._rollout_generator,
                device="cpu",
            )
            for start in range(0, len(batch), int(self.config.minibatch_size)):
                indices = permutation[start : start + int(self.config.minibatch_size)]
                minibatch = batch.index_select(indices.to(self.device))
                minibatch_advantages = advantages.index_select(
                    0,
                    indices.to(self.device),
                )
                loss_state = self._compute_loss(minibatch, minibatch_advantages)
                self.optimizer.zero_grad(set_to_none=True)
                loss_state.total_loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=float(self.config.gradient_clip_norm),
                    )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                statistics["total_loss"] += float(loss_state.total_loss.detach().item())
                statistics["policy_loss"] += float(
                    loss_state.policy_loss.detach().item()
                )
                statistics["value_loss"] += float(loss_state.value_loss.detach().item())
                statistics["entropy"] += float(loss_state.entropy.detach().item())
                statistics["auxiliary_loss"] += float(
                    loss_state.auxiliary_loss.detach().item()
                )
                statistics["approx_kl"] += float(loss_state.approx_kl.detach().item())
                statistics["clip_fraction"] += float(
                    loss_state.clip_fraction.detach().item()
                )
                statistics["num_minibatches"] += 1.0

                if self.config.target_kl is not None and float(
                    loss_state.approx_kl.detach().item()
                ) > float(self.config.target_kl):
                    stop_early = True
                    break
            if stop_early:
                break

        count = max(1.0, statistics.pop("num_minibatches"))
        return {key: float(value) / count for key, value in statistics.items()}

    def _optimize_auxiliary_only(self) -> dict[str, float]:
        """
        Run beam/demo distillation updates without PPO rollout rows.

        Returns:
            Mean optimization statistics across auxiliary-only updates.

        """
        statistics = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "auxiliary_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "num_updates": 0.0,
        }

        for _ in range(int(self.config.auxiliary.updates_per_step)):
            auxiliary_loss = self._sample_auxiliary_supervision_loss()
            if bool(auxiliary_loss.requires_grad):
                self.optimizer.zero_grad(set_to_none=True)
                auxiliary_loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=float(self.config.gradient_clip_norm),
                    )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            loss_value = float(auxiliary_loss.detach().item())
            statistics["total_loss"] += loss_value
            statistics["auxiliary_loss"] += loss_value
            statistics["num_updates"] += 1.0

        count = max(1.0, statistics.pop("num_updates"))
        return {key: float(value) / count for key, value in statistics.items()}

    def _compute_loss(  # noqa: PLR0914
        self,
        minibatch: PolicyRolloutBatch,
        minibatch_advantages: torch.Tensor,
    ) -> SearchGuidedPPOLossState:
        """
        Compute PPO and auxiliary losses for one minibatch.

        Args:
            minibatch: Rollout minibatch.
            minibatch_advantages: Normalized advantages aligned with
                ``minibatch``.

        Returns:
            Tensor-valued PPO loss state.

        """
        evaluation = evaluate_policy_actions(
            self.model,
            minibatch.states,
            minibatch.actions,
            generator_indices=self.config.rollout.generator_indices,
            action_temperature=float(self.config.rollout.action_temperature),
        )
        old_log_probs = minibatch.log_probs.to(self.device)
        ratios = torch.exp(evaluation.log_probs - old_log_probs)
        unclipped = ratios * minibatch_advantages
        clipped = (
            torch.clamp(
                ratios,
                1.0 - float(self.config.clip_ratio),
                1.0 + float(self.config.clip_ratio),
            )
            * minibatch_advantages
        )
        policy_loss = -torch.min(unclipped, clipped).mean()

        return_targets = minibatch.returns.to(self.device)
        old_values = minibatch.values.to(self.device)
        value_predictions = evaluation.values
        if self.config.value_clip_ratio is None:
            value_loss = 0.5 * torch.mean((value_predictions - return_targets) ** 2)
        else:
            clipped_values = old_values + torch.clamp(
                value_predictions - old_values,
                -float(self.config.value_clip_ratio),
                float(self.config.value_clip_ratio),
            )
            unclipped_value_loss = (value_predictions - return_targets) ** 2
            clipped_value_loss = (clipped_values - return_targets) ** 2
            value_loss = (
                0.5
                * torch.max(
                    unclipped_value_loss,
                    clipped_value_loss,
                ).mean()
            )

        entropy = evaluation.entropy.mean()
        approx_kl = (old_log_probs - evaluation.log_probs).mean()
        clip_fraction = (
            ((ratios - 1.0).abs() > float(self.config.clip_ratio)).float().mean()
        )

        auxiliary_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        auxiliary_loss += self._sample_auxiliary_supervision_loss()

        total_loss = (
            policy_loss
            + float(self.config.value_coef) * value_loss
            - float(self.config.entropy_coef) * entropy
            + auxiliary_loss
        )
        return SearchGuidedPPOLossState(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            auxiliary_loss=auxiliary_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
        )

    def _sample_auxiliary_supervision_loss(self) -> torch.Tensor:
        """
        Sample optional demo/search archive batches and compute their losses.

        Returns:
            Scalar auxiliary loss tensor on the trainer device.

        """
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        if (
            self.demo_archive is not None
            and len(self.demo_archive) > 0
            and (
                float(self.config.auxiliary.demo_policy_coef) > 0.0
                or float(self.config.auxiliary.demo_value_coef) > 0.0
            )
        ):
            batch = self.demo_archive.sample(
                int(self.config.demo_supervision.batch_size),
                generator=self._rollout_generator,
                device=self.device,
            )
            policy_loss, value_loss, _ = compute_supervised_policy_value_losses(
                self.model,
                batch,
                device=self.device,
                generator_indices=self.config.rollout.generator_indices,
            )
            total_loss += float(self.config.auxiliary.demo_policy_coef) * policy_loss
            total_loss += float(self.config.auxiliary.demo_value_coef) * value_loss

        if (
            self.search_archive is not None
            and len(self.search_archive) > 0
            and (
                float(self.config.auxiliary.search_policy_coef) > 0.0
                or float(self.config.auxiliary.search_value_coef) > 0.0
            )
        ):
            batch = self.search_archive.sample(
                int(self.config.beam_search.archive_batch_size),
                generator=self._rollout_generator,
                device=self.device,
            )
            policy_loss, value_loss, _ = compute_supervised_policy_value_losses(
                self.model,
                batch,
                device=self.device,
                generator_indices=self.config.rollout.generator_indices,
            )
            total_loss += float(self.config.auxiliary.search_policy_coef) * policy_loss
            total_loss += float(self.config.auxiliary.search_value_coef) * value_loss

        return total_loss

    def _sample_rollout_start_states(self) -> torch.Tensor:
        """
        Sample rollout start states from the configured random-walk schedule.

        Returns:
            Tensor of shape ``(num_envs, state_size)`` on ``self.device``.

        """
        sampled = sample_states_from_random_walks(
            self.graph,
            self.config.rollout.sampling,
            sample_index=self._start_state_sample_index,
        )
        self._start_state_sample_index += 1
        sampled = torch.as_tensor(sampled).long().cpu()
        if sampled.ndim == 1:
            sampled = sampled.unsqueeze(0)
        num_envs = int(self.config.rollout.num_envs)
        if int(sampled.shape[0]) < num_envs:
            repeats = (num_envs + int(sampled.shape[0]) - 1) // int(sampled.shape[0])
            sampled = sampled.repeat((repeats, 1))
        permutation = torch.randperm(
            int(sampled.shape[0]),
            generator=self._rollout_generator,
        )[:num_envs]
        return sampled.index_select(0, permutation).to(self.device)

    def _refresh_search_archive_from_sampled_states(self) -> BeamSearchTargetStats:
        """
        Add beam-derived supervision from fresh random-walk states.

        Returns:
            Summary stats for the archive-refresh beam searches.

        """
        if (
            self.search_archive is None
            or not bool(self.config.beam_search.enabled)
            or int(self.config.beam_search.archive_targets_per_update) <= 0
        ):
            return BeamSearchTargetStats(0, 0, 0, None)
        states = self._sample_rollout_start_states()
        return self._refresh_search_archive_from_rollout(states)

    def _collect_rollout_start_targets(
        self,
        start_states: torch.Tensor,
    ) -> tuple[BeamSearchTargetSet | None, BeamSearchTargetStats]:
        """
        Run beam search on a subset of rollout start states.

        Args:
            start_states: Rollout start states.

        Returns:
            Beam targets aligned with rollout environments and summary stats.

        """
        if (
            not bool(self.config.beam_search.enabled)
            or int(self.config.beam_search.rollout_start_targets) <= 0
        ):
            return None, BeamSearchTargetStats(0, 0, 0, None)
        limit = min(
            int(self.config.beam_search.rollout_start_targets),
            int(start_states.shape[0]),
        )
        permutation = torch.randperm(
            int(start_states.shape[0]),
            generator=self._rollout_generator,
        )[:limit]
        subset_states = start_states.index_select(
            0, permutation.to(start_states.device)
        )
        target_set, stats = collect_beam_search_targets(
            self.graph,
            self.model,
            subset_states,
            self.config.beam_search,
        )
        if target_set is None:
            return None, stats
        remapped_indices = permutation.index_select(
            0, target_set.state_indices.to(permutation.device)
        )
        return (
            BeamSearchTargetSet(
                batch=target_set.batch,
                state_indices=remapped_indices.cpu(),
                path_lengths=target_set.path_lengths,
                best_widths=target_set.best_widths,
            ),
            stats,
        )

    def _refresh_search_archive_from_rollout(
        self,
        states: torch.Tensor,
    ) -> BeamSearchTargetStats:
        """
        Add beam-derived supervision from rollout states to the search archive.

        Args:
            states: Candidate rollout states.

        Returns:
            Summary stats for the archive-refresh beam searches.

        """
        if (
            self.search_archive is None
            or not bool(self.config.beam_search.enabled)
            or int(self.config.beam_search.archive_targets_per_update) <= 0
            or int(states.shape[0]) == 0
        ):
            return BeamSearchTargetStats(0, 0, 0, None)

        candidate_count = min(
            int(self.config.beam_search.archive_targets_per_update),
            int(states.shape[0]),
        )
        permutation = torch.randperm(
            int(states.shape[0]),
            generator=self._rollout_generator,
        )[:candidate_count]
        target_set, stats = collect_beam_search_targets(
            self.graph,
            self.model,
            states.index_select(0, permutation.to(states.device)),
            self.config.beam_search,
        )
        if target_set is not None:
            self.search_archive.add(target_set.batch)
        self._archive_sample_index += 1
        return stats

    def _teacher_progress_reward(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute teacher-value progress shaping between two state batches.

        Args:
            states: Source states.
            next_states: Next states after one action.

        Returns:
            One-dimensional shaping reward tensor.

        """
        weight = float(self.config.reward.teacher_progress_weight)
        if abs(weight) <= _ZERO_EPS:
            return torch.zeros(
                int(states.shape[0]),
                device=states.device,
                dtype=torch.float32,
            )
        teacher = self.model if self.teacher_model is None else self.teacher_model
        with torch.no_grad():
            before = forward_policy_value(teacher, states).values
            after = forward_policy_value(teacher, next_states).values
        progress = before - after
        if self.config.reward.teacher_progress_clip is not None:
            progress = torch.clamp(
                progress,
                -float(self.config.reward.teacher_progress_clip),
                float(self.config.reward.teacher_progress_clip),
            )
        return float(weight) * progress

    def _inverse_action_penalty(
        self,
        previous_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize immediate inverse moves when configured.

        Args:
            previous_actions: Previous action ids aligned with the current batch.
            actions: Current action ids.

        Returns:
            One-dimensional penalty tensor.

        """
        penalty_value = float(self.config.reward.inverse_action_penalty)
        if abs(penalty_value) <= _ZERO_EPS or self.inverse_map is None:
            return torch.zeros(
                int(actions.shape[0]),
                device=actions.device,
                dtype=torch.float32,
            )
        inverse_previous = self.inverse_map.index_select(
            0,
            previous_actions.clamp_min(0).to(self.inverse_map.device),
        ).to(actions.device)
        valid_previous = previous_actions >= 0
        inverse_mask = valid_previous & (actions == inverse_previous)
        return inverse_mask.float() * float(penalty_value)

    def _revisit_penalty(
        self,
        next_states: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
        seen_hash_layers: list[torch.Tensor],
        current_hashes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Penalize revisiting previously seen states within the rollout.

        Args:
            next_states: Next states for currently active environments.
            valid_mask: Boolean mask of active rollout environments.
            seen_hash_layers: List of per-step state hashes for all environments.
            current_hashes: Current hash tensor for all environments.

        Returns:
            Tuple ``(penalty, next_hashes)`` aligned with ``next_states``.

        """
        next_hashes = self.graph.hasher.make_hashes(
            next_states.to(_graph_device(self.graph))
        ).to(current_hashes.device)
        penalty_value = float(self.config.reward.revisit_penalty)
        if abs(penalty_value) <= _ZERO_EPS or not seen_hash_layers:
            return (
                torch.zeros(
                    int(next_states.shape[0]),
                    device=next_states.device,
                    dtype=torch.float32,
                ),
                next_hashes,
            )
        prior_hashes = torch.stack(seen_hash_layers, dim=1)[valid_mask]
        revisit_mask = (prior_hashes == next_hashes.unsqueeze(1)).any(dim=1)
        return revisit_mask.float() * float(penalty_value), next_hashes

    @staticmethod
    def _mean_reward_component(
        total_value: float,
        valid_transition_count: int,
    ) -> float:
        """
        Convert a summed reward component into a mean per valid transition.

        Args:
            total_value: Sum accumulated across rollout transitions.
            valid_transition_count: Number of valid rollout transitions.

        Returns:
            Mean reward contribution.

        """
        if int(valid_transition_count) <= 0:
            return 0.0
        return float(total_value) / float(valid_transition_count)

    def _resolve_device(self) -> torch.device:
        """
        Resolve the runtime device for the actor-critic model.

        Returns:
            Trainer device.

        """
        configured = self.config.device
        if str(configured) != "auto":
            return torch.device(configured)
        return _graph_device(self.graph)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build the default AdamW optimizer.

        Returns:
            Optimizer configured from the trainer config.

        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            betas=tuple(float(x) for x in self.config.optimizer_betas),
        )

    def _build_lr_scheduler(self) -> Any | None:
        """
        Build the optional step-based learning-rate scheduler.

        Returns:
            Scheduler instance configured from ``self.config``.

        """
        if bool(self.config.rollout.enabled):
            optimizer_steps = (
                int(self.config.num_updates)
                * int(self.config.num_policy_epochs)
                * max(
                    1,
                    int(
                        (self.config.rollout.num_envs * self.config.rollout.horizon)
                        / max(1, self.config.minibatch_size)
                    ),
                )
            )
        else:
            optimizer_steps = int(self.config.num_updates) * int(
                self.config.auxiliary.updates_per_step
            )
        scheduler_ctor = lr_scheduler_ctor_from_cfg(
            {
                "num_updates": optimizer_steps,
                "lr_scheduler": self.config.lr_scheduler.to_log_dict(),
            },
            total_steps_key="num_updates",
            allow_plateau=False,
        )
        if scheduler_ctor is None:
            return None
        return scheduler_ctor(self.optimizer)
