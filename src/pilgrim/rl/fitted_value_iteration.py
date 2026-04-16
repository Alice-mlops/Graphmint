# Implements a high-level fitted value iteration trainer for Pilgrim models.
"""Trainer for deterministic shortest-path value learning."""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from cayleypy import CayleyGraph
from torch import nn

from pilgrim.utils.losses import lipschitz_expansion_loss

from .config import FittedValueIterationConfig
from .replay import FrontierArchiveUpdateStats, FrontierStateArchive, TensorReplayBuffer
from .sampling import (
    sample_states_from_random_walks,
    sample_suffix_states_from_random_walks,
    subsample_states,
)
from .transitions import compute_bellman_value_targets


@dataclass(slots=True)
class FittedValueIterationMetrics:
    """
    Metrics reported for one fitted-value-iteration optimizer step.

    Args:
        step: One-based optimizer step index.
        total_loss: Full loss used for backpropagation.
        bellman_loss: Mean-squared Bellman residual.
        lipschitz_loss: Optional Lipschitz penalty value.
        replay_size: Replay-buffer size after the update.

    """

    step: int
    total_loss: float
    bellman_loss: float
    lipschitz_loss: float | None
    replay_size: int


@dataclass(slots=True)
class FittedValueIterationLossState:
    """
    Tensor-valued outputs produced when scoring one replay batch.

    Args:
        total_loss: Full differentiable loss used for backpropagation.
        bellman_loss: Mean-squared Bellman residual.
        lipschitz_loss: Optional Lipschitz penalty tensor.
        predictions: Online-model value predictions for the sampled states.
        targets: Frozen-target Bellman values used as regression targets.

    """

    total_loss: torch.Tensor
    bellman_loss: torch.Tensor
    lipschitz_loss: torch.Tensor | None
    predictions: torch.Tensor
    targets: torch.Tensor


@dataclass(slots=True)
class FittedValueIterationFrontierRefreshStats:
    """
    Summary of one frontier-archive refresh attempt.

    Args:
        refresh_applied: Whether frontier refresh logic ran on this step.
        candidate_count: Number of suffix states proposed before deduplication.
        unique_candidate_count: Number of unique candidate states after dedup.
        selected_count: Number of top-scoring candidates considered for archive
            admission.
        admitted: Number of new states appended into free archive slots.
        updated: Number of rediscovered archive states whose scores were
            updated.
        replaced: Number of weaker archive entries evicted.

    """

    refresh_applied: bool = False
    candidate_count: int = 0
    unique_candidate_count: int = 0
    selected_count: int = 0
    admitted: int = 0
    updated: int = 0
    replaced: int = 0


@dataclass(slots=True)
class FittedValueIterationStepDiagnostics:
    """
    Detailed diagnostics collected for one optimizer step.

    Args:
        step: One-based optimizer step index.
        total_loss: Full loss used for optimization.
        bellman_loss: Mean-squared Bellman residual.
        lipschitz_loss: Optional Lipschitz penalty value.
        replay_size: Replay-buffer size after the update.
        replay_fill_ratio: Fraction of replay capacity currently filled.
        learning_rate: Optimizer learning rate used for the step.
        step_time_s: Wall-clock duration of the optimizer step.
        gradient_global_norm: Global L2 norm across all gradients.
        gradient_max_abs: Maximum absolute gradient entry.
        target_sync_applied: Whether the target network was synchronized.
        frontier_archive_size: Number of currently archived frontier states.
        frontier_archive_fill_ratio: Fraction of the frontier archive that is
            filled.
        frontier_batch_size: Number of frontier states mixed into the batch.
        frontier_refresh_applied: Whether frontier refresh ran before sampling
            this training batch.
        frontier_candidate_count: Number of suffix candidates generated before
            deduplication.
        frontier_unique_candidate_count: Number of unique frontier candidates
            remaining after deduplication.
        frontier_selected_count: Number of top-scoring frontier candidates
            considered for archive admission.
        frontier_admitted: Number of new frontier states admitted.
        frontier_updated: Number of archive states updated in place.
        frontier_replaced: Number of weaker archive states evicted.
        frontier_score_mean: Mean score of currently archived frontier states.
        frontier_score_max: Maximum score of currently archived frontier
            states.
        batch_states: Sampled replay states used for the step.
        predictions: Online-model value predictions for ``batch_states``.
        targets: Bellman targets for ``batch_states``.

    """

    step: int
    total_loss: float
    bellman_loss: float
    lipschitz_loss: float | None
    replay_size: int
    replay_fill_ratio: float
    learning_rate: float
    step_time_s: float
    gradient_global_norm: float | None
    gradient_max_abs: float | None
    target_sync_applied: bool
    frontier_archive_size: int
    frontier_archive_fill_ratio: float
    frontier_batch_size: int
    frontier_refresh_applied: bool
    frontier_candidate_count: int
    frontier_unique_candidate_count: int
    frontier_selected_count: int
    frontier_admitted: int
    frontier_updated: int
    frontier_replaced: int
    frontier_score_mean: float | None
    frontier_score_max: float | None
    batch_states: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor


class FittedValueIterationTracker(Protocol):
    """
    Protocol for optional side-effect trackers used by the trainer.

    Methods on the protocol may log to Aim, write files, or compute summaries.
    The trainer keeps the protocol deliberately small so notebooks can swap in
    different trackers without changing the optimization loop.

    """

    def on_fit_start(self, trainer: FittedValueIterationTrainer) -> None:
        """
        Handle the start of ``fit``.

        Args:
            trainer: Active fitted-value-iteration trainer.

        Returns:
            None.

        """

    def on_train_step_end(
        self,
        trainer: FittedValueIterationTrainer,
        diagnostics: FittedValueIterationStepDiagnostics,
    ) -> None:
        """
        Handle diagnostics produced after one optimizer step.

        Args:
            trainer: Active fitted-value-iteration trainer.
            diagnostics: Step-level metrics and batch diagnostics.

        Returns:
            None.

        """

    def on_fit_end(
        self,
        trainer: FittedValueIterationTrainer,
        history: Sequence[FittedValueIterationMetrics],
    ) -> None:
        """
        Handle the end of ``fit``.

        Args:
            trainer: Active fitted-value-iteration trainer.
            history: Metrics returned by completed optimizer steps.

        Returns:
            None.

        """


class FittedValueIterationTrainer:
    """
    Train a scalar value model with fitted value iteration.

    Args:
        model: Scalar value network mapping states to distance estimates.
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
        config: FittedValueIterationConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        tracker: FittedValueIterationTracker | None = None,
    ) -> None:
        self.model = model
        self.graph = graph
        self.config = config or FittedValueIterationConfig()
        self.graph_device = torch.device(getattr(self.graph, "device", "cpu"))
        self._validate_config()

        self.device = self._resolve_device(self.config.device)
        self.model = self.model.to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.optimizer = optimizer or self._build_optimizer()
        self.tracker = tracker
        self.replay = TensorReplayBuffer(capacity=int(self.config.replay.capacity))
        self.frontier_archive = self._build_frontier_archive()
        self._num_sampling_calls = 0
        self._num_frontier_sampling_calls = 0
        self._step = 0
        self._last_frontier_refresh = FittedValueIterationFrontierRefreshStats()

    def fit(self, num_updates: int | None = None) -> list[FittedValueIterationMetrics]:
        """
        Run a fitted-value-iteration optimization loop.

        Args:
            num_updates: Optional override for the number of optimizer steps.

        Returns:
            Metrics for each completed optimizer step.

        """
        history: list[FittedValueIterationMetrics] = []
        if self.tracker is not None:
            self.tracker.on_fit_start(self)

        try:
            self.ensure_replay_ready()
            total_updates = (
                int(self.config.num_updates)
                if num_updates is None
                else int(num_updates)
            )
            history.extend(self.train_step() for _ in range(total_updates))
        finally:
            if self.tracker is not None:
                self.tracker.on_fit_end(self, history)

        return history

    def train_step(self) -> FittedValueIterationMetrics:
        """
        Run one optimizer step of fitted value iteration.

        Returns:
            Step metrics summarizing losses and replay size.

        """
        step_started = time.perf_counter()
        self.ensure_replay_ready()
        self._maybe_refresh_replay()
        frontier_refresh = self._maybe_refresh_frontier_archive()
        batch, frontier_batch_size = self._sample_training_batch()
        self.model.train()
        loss_state = self.compute_loss(batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss_state.total_loss.backward()
        gradient_global_norm = self._compute_gradient_global_norm()
        gradient_max_abs = self._compute_gradient_max_abs()
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(self.config.gradient_clip_norm),
            )
        self.optimizer.step()

        self._step += 1
        target_sync_applied = False
        if self._step % int(self.config.target_sync_interval) == 0:
            self.sync_target_model()
            target_sync_applied = True

        lipschitz_value = (
            None
            if loss_state.lipschitz_loss is None
            else float(loss_state.lipschitz_loss.detach().item())
        )
        metrics = FittedValueIterationMetrics(
            step=int(self._step),
            total_loss=float(loss_state.total_loss.detach().item()),
            bellman_loss=float(loss_state.bellman_loss.detach().item()),
            lipschitz_loss=lipschitz_value,
            replay_size=len(self.replay),
        )
        frontier_score_mean, frontier_score_max = self._frontier_archive_score_stats()
        if self.tracker is not None:
            diagnostics = FittedValueIterationStepDiagnostics(
                step=int(self._step),
                total_loss=float(loss_state.total_loss.detach().item()),
                bellman_loss=float(loss_state.bellman_loss.detach().item()),
                lipschitz_loss=lipschitz_value,
                replay_size=len(self.replay),
                replay_fill_ratio=self.replay.storage_usage_ratio(),
                learning_rate=self._current_learning_rate(),
                step_time_s=time.perf_counter() - step_started,
                gradient_global_norm=gradient_global_norm,
                gradient_max_abs=gradient_max_abs,
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
                batch_states=batch.detach().cpu(),
                predictions=loss_state.predictions.detach().cpu(),
                targets=loss_state.targets.detach().cpu(),
            )
            self.tracker.on_train_step_end(self, diagnostics)
        return metrics

    def compute_loss(
        self,
        states: torch.Tensor,
    ) -> FittedValueIterationLossState:
        """
        Compute Bellman and optional Lipschitz losses for one batch.

        Args:
            states: Batch of states sampled from replay memory.

        Returns:
            Tensor-valued loss state for the sampled batch.

        """
        batch = torch.as_tensor(states, device=self.device).long()
        targets = compute_bellman_value_targets(
            self.target_model,
            self.graph,
            batch,
            reward_per_step=float(self.config.reward_per_step),
            terminal_value=float(self.config.terminal_value),
            generator_indices=self.config.generator_indices,
            value_batch_size=self.config.value_batch_size,
        ).to(self.device)
        predictions = self.model(batch).reshape(-1).float()
        bellman_loss = torch.nn.functional.mse_loss(predictions, targets.float())

        total_loss = bellman_loss
        lipschitz_loss: torch.Tensor | None = None
        if float(self.config.lipschitz.weight) > 0.0:
            lipschitz_loss = lipschitz_expansion_loss(
                self.model,
                self.graph,
                batch,
                max_states=self.config.lipschitz.max_states,
                generator_indices=self.config.lipschitz.generator_indices,
                max_generators=self.config.lipschitz.max_generators,
                seed=self.config.lipschitz.seed,
                state_batch_size=self.config.lipschitz.state_batch_size,
                reduction=self.config.lipschitz.reduction,
            ).float()
            total_loss += float(self.config.lipschitz.weight) * lipschitz_loss

        return FittedValueIterationLossState(
            total_loss=total_loss,
            bellman_loss=bellman_loss,
            lipschitz_loss=lipschitz_loss,
            predictions=predictions,
            targets=targets,
        )

    def ensure_replay_ready(self) -> None:
        """Populate replay memory until it reaches the configured minimum size."""
        target_size = max(
            int(self.config.replay.min_size),
            int(self.config.replay.warmstart_size),
        )
        if self.replay.is_ready(target_size):
            return
        self.populate_replay(target_size - len(self.replay))

    def populate_replay(self, num_states: int) -> int:
        """
        Append random-walk states to replay memory.

        Args:
            num_states: Target number of states to append.

        Returns:
            Number of states written into the replay buffer.

        Raises:
            ValueError: If ``num_states`` is negative.

        """
        if int(num_states) < 0:
            raise ValueError("num_states must be non-negative.")

        written = 0
        while written < int(num_states):
            states = sample_states_from_random_walks(
                self.graph,
                self.config.sampling,
                sample_index=self._num_sampling_calls,
            )
            self._num_sampling_calls += 1
            remaining = int(num_states) - written
            chunk = subsample_states(
                states,
                max_states=remaining,
                seed=int(self.config.sampling.seed) + self._num_sampling_calls,
            ).to(self.replay.storage_device)
            written += self.replay.add(chunk)
        return written

    def sync_target_model(self) -> None:
        """Synchronize the frozen target model with the online model."""
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def _build_frontier_archive(self) -> FrontierStateArchive | None:
        """
        Build the optional frontier archive.

        Returns:
            Frontier archive instance when enabled, otherwise ``None``.

        """
        if int(self.config.frontier.capacity) <= 0:
            return None
        return FrontierStateArchive(
            capacity=int(self.config.frontier.capacity),
            storage_device=self.replay.storage_device,
        )

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

    def _current_learning_rate(self) -> float:
        """
        Return the learning rate of the first optimizer parameter group.

        Returns:
            Scalar learning rate.

        """
        return float(self.optimizer.param_groups[0]["lr"])

    def _compute_gradient_global_norm(self) -> float | None:
        """
        Compute the global L2 norm across all model gradients.

        Returns:
            Global gradient norm, or ``None`` when no gradients are present.

        """
        squared_sum = 0.0
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            squared_sum += float(torch.sum(grad * grad).item())
            has_gradients = True
        if not has_gradients:
            return None
        return squared_sum**0.5

    def _compute_gradient_max_abs(self) -> float | None:
        """
        Compute the maximum absolute gradient entry.

        Returns:
            Maximum absolute gradient value, or ``None`` when absent.

        """
        max_abs = 0.0
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is None:
                continue
            grad_max = float(param.grad.detach().abs().max().item())
            max_abs = max(max_abs, grad_max)
            has_gradients = True
        if not has_gradients:
            return None
        return max_abs

    def _sample_training_batch(self) -> tuple[torch.Tensor, int]:
        """
        Sample one training batch from replay and the optional frontier archive.

        Returns:
            Tuple ``(batch, frontier_batch_size)`` where ``frontier_batch_size``
            records how many archive states were mixed into the batch.

        Raises:
            RuntimeError: If both replay sources are unexpectedly empty.

        """
        total_batch_size = int(self.config.replay.batch_size)
        frontier_batch_size = 0
        batch_parts: list[torch.Tensor] = []

        if (
            self.frontier_archive is not None
            and int(self.config.frontier.batch_size) > 0
            and len(self.frontier_archive) > 0
        ):
            frontier_batch_size = min(
                int(self.config.frontier.batch_size),
                total_batch_size,
                len(self.frontier_archive),
            )
            batch_parts.append(
                self.frontier_archive.sample(
                    frontier_batch_size,
                    device=self.device,
                )
            )

        main_batch_size = total_batch_size - frontier_batch_size
        if main_batch_size > 0:
            batch_parts.append(
                self.replay.sample(
                    main_batch_size,
                    device=self.device,
                )
            )

        if not batch_parts:
            raise RuntimeError("cannot sample a training batch from empty buffers.")

        if len(batch_parts) == 1:
            return batch_parts[0], frontier_batch_size

        batch = torch.cat(batch_parts, dim=0)
        permutation = torch.randperm(batch.shape[0], device=batch.device)
        return batch[permutation], frontier_batch_size

    def _maybe_refresh_frontier_archive(
        self,
    ) -> FittedValueIterationFrontierRefreshStats:
        """
        Refresh the optional frontier archive with suffix-filtered long walks.

        Returns:
            Summary of the archive refresh attempt for the current step.

        """
        stats = FittedValueIterationFrontierRefreshStats()
        if self._frontier_refresh_is_due():
            frontier_mode = self._resolve_frontier_candidate_mode()
            frontier_width = self._resolve_frontier_candidate_width()
            frontier_length = self._resolve_frontier_candidate_length()
            frontier_history_depth = self._resolve_frontier_candidate_history_depth(
                candidate_mode=frontier_mode,
                candidate_length=frontier_length,
            )
            candidate_states, _ = sample_suffix_states_from_random_walks(
                self.graph,
                rw_mode=frontier_mode,
                rw_width=frontier_width,
                rw_length=frontier_length,
                suffix_fraction=float(self.config.frontier.suffix_fraction),
                base_seed=int(self.config.sampling.seed) + 1_000_003,
                sample_index=self._num_frontier_sampling_calls,
                nbt_history_depth=frontier_history_depth,
            )
            self._num_frontier_sampling_calls += 1
            stats.refresh_applied = True
            stats.candidate_count = int(candidate_states.shape[0])
            if candidate_states.shape[0] > 0:
                unique_candidates, unique_hashes = self.graph.get_unique_states(
                    self.graph.encode_states(candidate_states)
                )
                unique_states = self.graph.decode_states(unique_candidates)
                stats.unique_candidate_count = int(unique_states.shape[0])
                if unique_states.shape[0] > 0:
                    candidate_scores = compute_bellman_value_targets(
                        self.target_model,
                        self.graph,
                        unique_states,
                        reward_per_step=float(self.config.reward_per_step),
                        terminal_value=float(self.config.terminal_value),
                        generator_indices=self.config.generator_indices,
                        value_batch_size=self.config.value_batch_size,
                    )
                    stats.selected_count = min(
                        int(self.config.frontier.admissions_per_refresh),
                        int(candidate_scores.shape[0]),
                    )
                    if stats.selected_count > 0:
                        selected_indices = torch.topk(
                            candidate_scores,
                            k=int(stats.selected_count),
                            sorted=True,
                        ).indices
                        update_stats = self.frontier_archive.add_candidates(
                            unique_states[selected_indices],
                            unique_hashes[selected_indices],
                            candidate_scores[selected_indices],
                            score_ema_decay=float(
                                self.config.frontier.score_ema_decay
                            ),
                        )
                        stats = self._apply_frontier_update_stats(
                            stats=stats,
                            update_stats=update_stats,
                        )
        self._last_frontier_refresh = stats
        return stats

    def _maybe_refresh_replay(self) -> None:
        """Append fresh replay states when the configured stride is reached."""
        refresh_stride = int(self.config.replay.refresh_stride)
        refresh_size = int(self.config.replay.refresh_size)
        if refresh_size <= 0:
            return
        if refresh_stride <= 0:
            return
        if self._step % refresh_stride != 0:
            return
        self.populate_replay(refresh_size)

    def _frontier_archive_score_stats(self) -> tuple[float | None, float | None]:
        """
        Return score statistics of the optional frontier archive.

        Returns:
            Tuple ``(mean_score, max_score)`` or ``(None, None)`` when the
            archive is disabled or empty.

        """
        if self.frontier_archive is None:
            return None, None
        return self.frontier_archive.score_statistics()

    def _frontier_refresh_is_due(self) -> bool:
        """
        Return whether the frontier archive should refresh on this step.

        Returns:
            ``True`` when the frontier archive exists and its stride is due.

        """
        if self.frontier_archive is None:
            return False
        refresh_stride = int(self.config.frontier.refresh_stride)
        if refresh_stride <= 0:
            return False
        return self._step > 0 and self._step % refresh_stride == 0

    @staticmethod
    def _apply_frontier_update_stats(
        *,
        stats: FittedValueIterationFrontierRefreshStats,
        update_stats: FrontierArchiveUpdateStats,
    ) -> FittedValueIterationFrontierRefreshStats:
        """
        Merge archive update counts into the step-local refresh summary.

        Args:
            stats: Refresh summary produced so far.
            update_stats: Archive mutation counts returned by the archive.

        Returns:
            Refresh summary with admission and replacement counts applied.

        """
        stats.admitted = int(update_stats.admitted)
        stats.updated = int(update_stats.updated)
        stats.replaced = int(update_stats.replaced)
        return stats

    def _resolve_frontier_candidate_mode(self) -> str:
        """
        Resolve the random-walk mode used for frontier candidates.

        Returns:
            Candidate random-walk mode.

        """
        if self.config.frontier.candidate_mode is not None:
            return str(self.config.frontier.candidate_mode)
        return str(self.config.sampling.rw_mode)

    def _resolve_frontier_candidate_width(self) -> int:
        """
        Resolve the random-walk width used for frontier candidates.

        Returns:
            Positive candidate random-walk width.

        """
        if self.config.frontier.candidate_width is not None:
            return int(self.config.frontier.candidate_width)
        return int(self.config.sampling.rw_width)

    def _resolve_frontier_candidate_length(self) -> int:
        """
        Resolve the random-walk length used for frontier candidates.

        Returns:
            Positive candidate random-walk length.

        """
        if self.config.frontier.candidate_length is not None:
            return int(self.config.frontier.candidate_length)
        return max(2, int(self.config.sampling.rw_length) * 2)

    def _resolve_frontier_candidate_history_depth(
        self,
        *,
        candidate_mode: str,
        candidate_length: int,
    ) -> int | None:
        """
        Resolve the non-backtracking history depth for frontier candidates.

        Args:
            candidate_mode: Candidate random-walk mode.
            candidate_length: Candidate random-walk length.

        Returns:
            Non-backtracking history depth when relevant, otherwise ``None``.

        """
        if self.config.frontier.candidate_history_depth is not None:
            return int(self.config.frontier.candidate_history_depth)
        if str(candidate_mode) == "nbt":
            return int(candidate_length)
        return None

    def _resolve_device(self, device: str | torch.device) -> torch.device:
        """
        Resolve the model device for fitted value iteration.

        Args:
            device: Requested device or ``"auto"``.

        Returns:
            Resolved torch device.

        """
        if str(device).lower() == "auto":
            return self.graph_device
        return torch.device(device)

    def _validate_config(self) -> None:
        """Validate the trainer configuration."""
        self._validate_core_config()
        self._validate_frontier_config()
        self._validate_lipschitz_config()

    def _validate_core_config(self) -> None:
        """
        Validate the base fitted-value-iteration configuration.

        Raises:
            ValueError: If one of the core optimization settings is invalid.

        """
        if int(self.config.num_updates) < 0:
            raise ValueError("num_updates must be non-negative.")
        if float(self.config.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if int(self.config.target_sync_interval) <= 0:
            raise ValueError("target_sync_interval must be positive.")
        if int(self.config.replay.capacity) <= 0:
            raise ValueError("replay.capacity must be positive.")
        if int(self.config.replay.batch_size) <= 0:
            raise ValueError("replay.batch_size must be positive.")
        if int(self.config.replay.min_size) <= 0:
            raise ValueError("replay.min_size must be positive.")
        if int(self.config.replay.warmstart_size) <= 0:
            raise ValueError("replay.warmstart_size must be positive.")
        if int(self.config.replay.min_size) > int(self.config.replay.capacity):
            raise ValueError("replay.min_size cannot exceed replay.capacity.")
        if int(self.config.replay.warmstart_size) > int(self.config.replay.capacity):
            raise ValueError("replay.warmstart_size cannot exceed replay.capacity.")

    def _validate_frontier_config(self) -> None:
        """
        Validate optional frontier-archive settings.

        Raises:
            ValueError: If one of the frontier settings is invalid.

        """
        if int(self.config.frontier.capacity) < 0:
            raise ValueError("frontier.capacity cannot be negative.")
        if int(self.config.frontier.batch_size) < 0:
            raise ValueError("frontier.batch_size cannot be negative.")
        if (
            int(self.config.frontier.capacity) == 0
            and int(self.config.frontier.batch_size) > 0
        ):
            raise ValueError(
                "frontier.batch_size must be zero when frontier.capacity is zero."
            )
        if int(self.config.frontier.capacity) > 0:
            if int(self.config.frontier.refresh_stride) <= 0:
                raise ValueError("frontier.refresh_stride must be positive.")
            if int(self.config.frontier.admissions_per_refresh) <= 0:
                raise ValueError("frontier.admissions_per_refresh must be positive.")
            if not 0.0 < float(self.config.frontier.suffix_fraction) <= 1.0:
                raise ValueError(
                    "frontier.suffix_fraction must be in the open interval (0, 1]."
                )
            if not 0.0 <= float(self.config.frontier.score_ema_decay) <= 1.0:
                raise ValueError(
                    "frontier.score_ema_decay must be in the closed interval [0, 1]."
                )
        if (
            self.config.frontier.candidate_width is not None
            and int(self.config.frontier.candidate_width) <= 0
        ):
            raise ValueError("frontier.candidate_width must be positive when set.")
        if (
            self.config.frontier.candidate_length is not None
            and int(self.config.frontier.candidate_length) <= 0
        ):
            raise ValueError("frontier.candidate_length must be positive when set.")
        if (
            self.config.frontier.candidate_history_depth is not None
            and int(self.config.frontier.candidate_history_depth) < 0
        ):
            raise ValueError("frontier.candidate_history_depth cannot be negative.")

    def _validate_lipschitz_config(self) -> None:
        """
        Validate device assumptions required by Lipschitz regularization.

        Raises:
            ValueError: If Lipschitz regularization uses incompatible devices.

        """
        if float(self.config.lipschitz.weight) > 0.0 and self._resolve_device(
            self.config.device
        ) != torch.device(getattr(self.graph, "device", "cpu")):
            raise ValueError(
                "lipschitz regularization requires the model and graph to share "
                "the same device."
            )
