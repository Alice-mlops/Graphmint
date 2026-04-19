# Implements a high-level trainer for discounted multi-step Double-DQN.
"""Trainer for deterministic shortest-path Double-DQN with n-step targets."""

from __future__ import annotations

import time
from typing import Any

import torch
from cayleypy import CayleyGraph
from torch import nn

from pilgrim.schemas.rl import (
    MultiStepDDQNConfig,
    MultiStepDDQNLossState,
    MultiStepDDQNMetrics,
    MultiStepDDQNStepDiagnostics,
)
from pilgrim.utils.losses import lipschitz_expansion_loss

from .distributed import unwrap_model
from .helpers import (
    compute_double_q_targets_from_transition_batch,
    model_output_dim,
    sample_n_step_transitions_from_random_walks,
    sample_n_step_transitions_from_states,
    state_values_from_q,
)
from .multistep_ddqn_tracking import MultiStepDDQNTracker
from .multistep_td_value_iteration import MultiStepTDValueTrainer
from .replay import (
    TransitionBatch,
    TransitionReplayBuffer,
    concatenate_transition_batches,
    subsample_transition_batch,
)

_EXPECTED_Q_NDIM = 2


class MultiStepDDQNTrainer(MultiStepTDValueTrainer):
    """
    Train a vector-Q model with sampled multi-step Double-DQN targets.

    Args:
        model: Vector-valued Q network mapping states to per-action costs.
        graph: Cayley graph defining transitions and the center state.
        config: Trainer configuration.
        optimizer: Optional optimizer override. Defaults to AdamW.
        tracker: Optional side-effect tracker used for logging and diagnostics.

    Raises:
        ValueError: If the trainer configuration is inconsistent.

    """

    def __init__(
        self,
        model: nn.Module,
        graph: CayleyGraph,
        config: MultiStepDDQNConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        tracker: MultiStepDDQNTracker | None = None,
    ) -> None:
        self.config = config or MultiStepDDQNConfig()
        super().__init__(
            model=model,
            graph=graph,
            config=self.config,
            optimizer=optimizer,
            tracker=tracker,
        )
        self.replay = TransitionReplayBuffer(capacity=int(self.config.replay.capacity))
        self._sampling_generator = torch.Generator(device=self.replay.storage_device)
        self._sampling_generator.manual_seed(
            int(self.config.sampling.seed) + 100_000 * int(self.rank)
        )

    def fit(self, num_updates: int | None = None) -> list[MultiStepDDQNMetrics]:
        """
        Run a multi-step DDQN optimization loop.

        Args:
            num_updates: Optional override for the number of optimizer steps.

        Returns:
            Metrics for each completed optimizer step.

        """
        return [
            metric
            for metric in super().fit(num_updates=num_updates)
            if isinstance(metric, MultiStepDDQNMetrics)
        ]

    def compute_loss(
        self,
        batch: torch.Tensor | TransitionBatch,
    ) -> MultiStepDDQNLossState:
        """
        Compute DDQN and optional Lipschitz losses for one batch.

        Args:
            batch: Transition batch sampled from replay memory, or a raw state
                batch that should be converted on the fly.

        Returns:
            Tensor-valued loss state for the sampled batch.

        """
        loss_state, _, _ = self._compute_loss_with_timing(batch)
        return loss_state

    def _compute_loss_with_timing(
        self,
        batch: torch.Tensor | TransitionBatch,
    ) -> tuple[MultiStepDDQNLossState, float, float]:
        """
        Compute one DDQN loss payload and return phase timings.

        Args:
            batch: Transition batch sampled from replay memory, or a raw state
                batch that should be converted on the fly.

        Returns:
            Tuple ``(loss_state, target_compute_time_s, model_forward_time_s)``.

        Raises:
            ValueError: If ``self.model`` does not return vector Q-values.

        """
        target_started = time.perf_counter()
        if isinstance(batch, TransitionBatch):
            transition_batch = batch.to(self.device)
        else:
            transition_batch = sample_n_step_transitions_from_states(
                online_model=self.model,
                graph=self.graph,
                states=torch.as_tensor(batch, device=self.device).long(),
                n_steps=int(self.config.n_steps),
                generator_indices=self.config.generator_indices,
                q_batch_size=self.config.value_batch_size,
                behavior_mode=str(self.config.behavior.mode),
                behavior_epsilon=float(self.config.behavior.epsilon),
                action_generator=self._sampling_generator,
            ).to(self.device)
        targets = compute_double_q_targets_from_transition_batch(
            online_model=self.model,
            target_model=self.target_model,
            transitions=transition_batch,
            reward_per_step=float(self.config.reward_per_step),
            discount=float(self.config.discount),
            terminal_value=float(self.config.terminal_value),
            generator_indices=self.config.generator_indices,
            q_batch_size=self.config.value_batch_size,
        )
        target_compute_time_s = time.perf_counter() - target_started

        forward_started = time.perf_counter()
        q_values = torch.as_tensor(self.model(transition_batch.states)).float()
        if q_values.ndim != _EXPECTED_Q_NDIM:
            raise ValueError(
                "MultiStepDDQNTrainer requires a vector-output model with shape "
                f"(batch, num_actions), got {tuple(q_values.shape)}."
            )
        action_indices = (
            transition_batch.actions.to(q_values.device).long().reshape(-1, 1)
        )
        predictions = torch.gather(q_values, dim=1, index=action_indices).reshape(-1)
        model_forward_time_s = time.perf_counter() - forward_started
        td_loss = torch.nn.functional.mse_loss(predictions, targets.to(q_values.device))

        total_loss = td_loss
        lipschitz_loss: torch.Tensor | None = None
        if float(self.config.lipschitz.weight) > 0.0:
            lipschitz_loss = lipschitz_expansion_loss(
                self.model,
                self.graph,
                transition_batch.states,
                max_states=self.config.lipschitz.max_states,
                generator_indices=self.config.lipschitz.generator_indices,
                max_generators=self.config.lipschitz.max_generators,
                seed=self.config.lipschitz.seed,
                state_batch_size=self.config.lipschitz.state_batch_size,
                reduction=self.config.lipschitz.reduction,
            ).float()
            total_loss += float(self.config.lipschitz.weight) * lipschitz_loss

        return (
            MultiStepDDQNLossState(
                total_loss=total_loss,
                td_loss=td_loss,
                lipschitz_loss=lipschitz_loss,
                predictions=predictions,
                targets=targets.to(predictions.device),
                actions=transition_batch.actions.to(predictions.device),
            ),
            target_compute_time_s,
            model_forward_time_s,
        )

    def _build_step_metrics(
        self,
        loss_state: MultiStepDDQNLossState,
    ) -> MultiStepDDQNMetrics:
        """
        Build the compact metrics payload returned by ``train_step``.

        Args:
            loss_state: Tensor-valued loss payload for the current batch.

        Returns:
            Step metrics summarizing the optimizer update.

        """
        lipschitz_value = (
            None
            if loss_state.lipschitz_loss is None
            else float(loss_state.lipschitz_loss.detach().item())
        )
        return MultiStepDDQNMetrics(
            step=int(self._step),
            total_loss=float(loss_state.total_loss.detach().item()),
            td_loss=float(loss_state.td_loss.detach().item()),
            lipschitz_loss=lipschitz_value,
            replay_size=len(self.replay),
        )

    def _build_step_diagnostics(
        self,
        *,
        batch: TransitionBatch,
        frontier_batch_size: int,
        frontier_refresh: Any,
        loss_state: MultiStepDDQNLossState,
        metrics: MultiStepDDQNMetrics,
        optimizer_stats: Any,
        phase_times: Any,
        step_started: float,
        target_sync_applied: bool,
    ) -> MultiStepDDQNStepDiagnostics:
        """
        Build the detailed step diagnostics emitted to trackers.

        Args:
            batch: Sampled optimizer batch.
            frontier_batch_size: Number of frontier states in ``batch``.
            frontier_refresh: Frontier-refresh summary for the step.
            loss_state: Tensor-valued loss payload.
            metrics: Compact step metrics.
            optimizer_stats: Gradient and optimizer diagnostics.
            phase_times: Per-phase wall-clock timings.
            step_started: ``perf_counter`` timestamp captured at step start.
            target_sync_applied: Whether the target model was synchronized.

        Returns:
            Tracker diagnostics payload.

        """
        frontier_score_mean, frontier_score_max = self._frontier_archive_score_stats()
        return MultiStepDDQNStepDiagnostics(
            step=int(metrics.step),
            total_loss=float(metrics.total_loss),
            td_loss=float(metrics.td_loss),
            lipschitz_loss=metrics.lipschitz_loss,
            replay_size=len(self.replay),
            replay_fill_ratio=self.replay.storage_usage_ratio(),
            learning_rate=self._current_learning_rate(),
            step_time_s=time.perf_counter() - step_started,
            replay_refresh_time_s=phase_times.replay_refresh_time_s,
            frontier_refresh_time_s=phase_times.frontier_refresh_time_s,
            batch_sample_time_s=phase_times.batch_sample_time_s,
            target_compute_time_s=phase_times.target_compute_time_s,
            model_forward_time_s=phase_times.model_forward_time_s,
            backward_time_s=phase_times.backward_time_s,
            optimizer_time_s=phase_times.optimizer_time_s,
            gradient_global_norm=optimizer_stats.gradient_global_norm,
            gradient_max_abs=optimizer_stats.gradient_max_abs,
            target_sync_applied=target_sync_applied,
            frontier_archive_size=0
            if self.frontier_archive is None
            else len(self.frontier_archive),
            frontier_archive_fill_ratio=0.0
            if self.frontier_archive is None
            else self.frontier_archive.storage_usage_ratio(),
            frontier_batch_size=int(frontier_batch_size),
            frontier_refresh_applied=bool(frontier_refresh.refresh_applied),
            frontier_candidate_count=int(frontier_refresh.candidate_count),
            frontier_unique_candidate_count=int(
                frontier_refresh.unique_candidate_count
            ),
            frontier_selected_count=int(frontier_refresh.selected_count),
            frontier_admitted=int(frontier_refresh.admitted),
            frontier_updated=int(frontier_refresh.updated),
            frontier_replaced=int(frontier_refresh.replaced),
            frontier_score_mean=frontier_score_mean,
            frontier_score_max=frontier_score_max,
            batch_states=batch.states.detach().cpu(),
            predictions=loss_state.predictions.detach().cpu(),
            targets=loss_state.targets.detach().cpu(),
            actions=loss_state.actions.detach().cpu(),
        )

    def populate_replay(self, num_states: int) -> int:
        """
        Append random-walk transition rows to replay memory.

        Args:
            num_states: Target number of transition rows to append.

        Returns:
            Number of transitions written into the replay buffer.

        Raises:
            ValueError: If ``num_states`` is negative.

        """
        if int(num_states) < 0:
            raise ValueError("num_states must be non-negative.")

        written = 0
        while written < int(num_states):
            batch = sample_n_step_transitions_from_random_walks(
                self.graph,
                self.config.sampling,
                n_steps=int(self.config.n_steps),
                generator_indices=self.config.generator_indices,
                sample_index=self._num_sampling_calls,
            )
            self._num_sampling_calls += 1
            remaining = int(num_states) - written
            chunk = subsample_transition_batch(
                batch,
                max_transitions=remaining,
                seed=int(self.config.sampling.seed) + self._num_sampling_calls,
            )
            written += self.replay.add(chunk)
        return written

    def _sample_training_batch(self) -> tuple[TransitionBatch, int]:
        """
        Sample one DDQN optimizer batch from transition replay and frontier.

        Returns:
            Tuple ``(transition_batch, frontier_batch_size)``.

        Raises:
            RuntimeError: If no replay or frontier transitions are available.

        """
        total_batch_size = self._local_batch_size()
        frontier_batch_size = min(self._local_frontier_batch_size(), total_batch_size)
        batch_parts: list[TransitionBatch] = []

        if frontier_batch_size > 0:
            assert self.frontier_archive is not None
            frontier_states = self.frontier_archive.sample(
                frontier_batch_size,
                generator=self._sampling_generator,
                device=self.device,
            )
            batch_parts.append(
                sample_n_step_transitions_from_states(
                    online_model=self.model,
                    graph=self.graph,
                    states=frontier_states,
                    n_steps=int(self.config.n_steps),
                    generator_indices=self.config.generator_indices,
                    q_batch_size=self.config.value_batch_size,
                    behavior_mode=str(self.config.behavior.mode),
                    behavior_epsilon=float(self.config.behavior.epsilon),
                    action_generator=self._sampling_generator,
                ).to(self.device)
            )

        main_batch_size = total_batch_size - frontier_batch_size
        if main_batch_size > 0:
            batch_parts.append(
                self.replay.sample(
                    main_batch_size,
                    generator=self._sampling_generator,
                    device=self.device,
                )
            )

        if not batch_parts:
            raise RuntimeError("cannot sample a training batch from empty buffers.")
        batch = concatenate_transition_batches(batch_parts)
        if len(batch_parts) == 1:
            return batch, frontier_batch_size
        permutation = torch.randperm(
            len(batch),
            generator=self._permutation_generator,
            device=batch.states.device,
        )
        return batch.index_select(permutation), frontier_batch_size

    def _compute_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Score frontier candidates with ``min_a Q(s, a)``.

        Args:
            states: Batch of states whose scalar frontier scores are required.

        Returns:
            One-dimensional tensor of state scores on ``self.device``.

        """
        batch = torch.as_tensor(states, device=self.device).long()
        return state_values_from_q(
            self.target_model,
            batch,
            generator_indices=self.config.generator_indices,
            q_batch_size=self.config.value_batch_size,
        ).to(self.device)

    def _validate_config(self) -> None:
        """
        Validate the DDQN trainer configuration and model output shape.

        Raises:
            ValueError: If one of the optimization settings is invalid or the
                model output shape is incompatible with DDQN.

        """
        super()._validate_config()
        output_dim = int(model_output_dim(unwrap_model(self.model)))
        if output_dim <= 1:
            raise ValueError(
                "MultiStepDDQNTrainer requires a vector-output model with "
                "output_dim > 1."
            )
        generators = getattr(
            getattr(self.graph, "definition", None), "generators", None
        )
        if generators is None:
            generators = getattr(self.graph, "generators", None)
        if generators is not None and int(output_dim) < len(generators):
            raise ValueError(
                "DDQN model output_dim must cover all graph generators, got "
                f"output_dim={int(output_dim)} for {len(generators)} generators."
            )
        if self.config.generator_indices is not None and int(
            max(self.config.generator_indices)
        ) >= int(output_dim):
            raise ValueError(
                "generator_indices must fall inside the DDQN output range."
            )
