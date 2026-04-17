# Implements a high-level trainer for discounted multi-step TD value learning.
"""Trainer for discounted n-step and TD-lambda value learning."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch
from cayleypy import CayleyGraph
from torch import nn

from pilgrim.schemas.rl import (
    MultiStepTDValueConfig,
    MultiStepTDValueLossState,
    MultiStepTDValueMetrics,
    MultiStepTDValueStepDiagnostics,
)
from pilgrim.utils.losses import lipschitz_expansion_loss
from pilgrim.utils.lr_scheduler_utils import (
    lr_scheduler_ctor_from_cfg,
    step_lr_scheduler,
)

from .distributed import (
    distributed_rank,
    distributed_world_size,
    is_distributed_initialized,
    is_main_process,
    local_rank_from_env,
    split_evenly,
    synchronized_barrier,
    unwrap_model,
    wrap_model_for_ddp,
)
from .multistep_td_tracking import MultiStepTDValueTracker
from .replay import FrontierArchiveUpdateStats, FrontierStateArchive, TensorReplayBuffer
from .sampling import (
    sample_states_from_random_walks,
    sample_suffix_states_from_random_walks,
    subsample_states,
)
from .target_evaluation import build_td_target_evaluation_backend


@dataclass(slots=True)
class MultiStepTDValueFrontierRefreshStats:
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
class MultiStepTDValuePhaseTimes:
    """
    Wall-clock timings for one optimizer step.

    Args:
        replay_refresh_time_s: Time spent appending replay states.
        frontier_refresh_time_s: Time spent refreshing frontier candidates.
        batch_sample_time_s: Time spent sampling the optimizer batch.
        target_compute_time_s: Time spent constructing frozen TD targets.
        model_forward_time_s: Time spent in the online-model forward pass.
        backward_time_s: Time spent in ``loss.backward()``.
        optimizer_time_s: Time spent in clipping, optimizer step, and scheduler.

    """

    replay_refresh_time_s: float = 0.0
    frontier_refresh_time_s: float = 0.0
    batch_sample_time_s: float = 0.0
    target_compute_time_s: float = 0.0
    model_forward_time_s: float = 0.0
    backward_time_s: float = 0.0
    optimizer_time_s: float = 0.0


@dataclass(slots=True)
class MultiStepTDValueOptimizerStats:
    """
    Gradient and optimizer-step diagnostics for one training step.

    Args:
        gradient_global_norm: Global L2 norm across all gradients.
        gradient_max_abs: Maximum absolute gradient entry.
        backward_time_s: Time spent in ``loss.backward()``.
        optimizer_time_s: Time spent in clipping, optimizer step, and scheduler.

    """

    gradient_global_norm: float | None
    gradient_max_abs: float | None
    backward_time_s: float
    optimizer_time_s: float


class MultiStepTDValueTrainer:
    """
    Train a scalar value model with discounted multi-step TD targets.

    Args:
        model: Scalar value network mapping states to value estimates.
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
        config: MultiStepTDValueConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        tracker: MultiStepTDValueTracker | None = None,
    ) -> None:
        self.model = model
        self.graph = graph
        self.config = config or MultiStepTDValueConfig()
        self.graph_device = torch.device(getattr(self.graph, "device", "cpu"))
        self._validate_config()

        self.device = self._resolve_device(self.config.device)
        self.rank = distributed_rank()
        self.world_size = distributed_world_size()
        self.is_distributed = (
            self.config.parallel.uses_ddp
            and self.world_size > 1
            and is_distributed_initialized()
        )
        self.model = self.model.to(self.device)
        self.target_model = copy.deepcopy(unwrap_model(self.model)).to(self.device)
        self.target_model.eval()
        self.optimizer = optimizer or self._build_optimizer()
        if self.config.parallel.uses_ddp:
            self.model = wrap_model_for_ddp(
                self.model,
                device=self.device,
                broadcast_buffers=self.config.parallel.broadcast_buffers,
                find_unused_parameters=self.config.parallel.find_unused_parameters,
            )
        self.lr_scheduler = self._build_lr_scheduler()
        self.tracker = tracker if is_main_process() else None
        self.replay = TensorReplayBuffer(capacity=int(self.config.replay.capacity))
        self.frontier_archive = self._build_frontier_archive()
        self._num_sampling_calls = 0
        self._num_frontier_sampling_calls = 0
        self._step = 0
        self._last_frontier_refresh = MultiStepTDValueFrontierRefreshStats()
        self._sampling_generator = torch.Generator(device=self.replay.storage_device)
        self._sampling_generator.manual_seed(
            int(self.config.sampling.seed) + 100_000 * int(self.rank)
        )
        self._permutation_generator = self._build_permutation_generator()
        self.target_evaluator = build_td_target_evaluation_backend(
            target_model=self.target_model,
            graph=self.graph,
            config=self.config,
        )

    def fit(self, num_updates: int | None = None) -> list[MultiStepTDValueMetrics]:
        """
        Run a multi-step TD optimization loop.

        Args:
            num_updates: Optional override for the number of optimizer steps.

        Returns:
            Metrics for each completed optimizer step.

        """
        history: list[MultiStepTDValueMetrics] = []
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
            synchronized_barrier()
            if self.tracker is not None:
                self.tracker.on_fit_end(self, history)
            self.target_evaluator.close()

        return history

    def train_step(self) -> MultiStepTDValueMetrics:
        """
        Run one optimizer step of multi-step TD value learning.

        Returns:
            Step metrics summarizing losses and replay size.

        """
        step_started = time.perf_counter()
        self.ensure_replay_ready()
        frontier_refresh, batch, frontier_batch_size, phase_times = (
            self._refresh_and_sample_batch()
        )
        self.model.train()
        loss_state, target_compute_time_s, model_forward_time_s = (
            self._compute_loss_with_timing(batch)
        )
        phase_times.target_compute_time_s = target_compute_time_s
        phase_times.model_forward_time_s = model_forward_time_s
        optimizer_stats = self._apply_optimizer_step(loss_state)
        phase_times.backward_time_s = optimizer_stats.backward_time_s
        phase_times.optimizer_time_s = optimizer_stats.optimizer_time_s

        self._step += 1
        target_sync_applied = False
        if self._step % int(self.config.target_sync_interval) == 0:
            self.sync_target_model()
            target_sync_applied = True
        metrics = self._build_step_metrics(loss_state)
        if self.tracker is not None:
            self.tracker.on_train_step_end(
                self,
                self._build_step_diagnostics(
                    batch=batch,
                    frontier_batch_size=frontier_batch_size,
                    frontier_refresh=frontier_refresh,
                    loss_state=loss_state,
                    metrics=metrics,
                    optimizer_stats=optimizer_stats,
                    phase_times=phase_times,
                    step_started=step_started,
                    target_sync_applied=target_sync_applied,
                ),
            )
        return metrics

    def _refresh_and_sample_batch(
        self,
    ) -> tuple[
        MultiStepTDValueFrontierRefreshStats,
        torch.Tensor,
        int,
        MultiStepTDValuePhaseTimes,
    ]:
        """
        Refresh replay/frontier state and sample the next optimizer batch.

        Returns:
            Tuple ``(frontier_refresh, batch, frontier_batch_size, phase_times)``.

        """
        phase_times = MultiStepTDValuePhaseTimes()
        replay_refresh_started = time.perf_counter()
        self._maybe_refresh_replay()
        phase_times.replay_refresh_time_s = (
            time.perf_counter() - replay_refresh_started
        )

        frontier_refresh_started = time.perf_counter()
        frontier_refresh = self._maybe_refresh_frontier_archive()
        phase_times.frontier_refresh_time_s = (
            time.perf_counter() - frontier_refresh_started
        )

        batch_sample_started = time.perf_counter()
        batch, frontier_batch_size = self._sample_training_batch()
        phase_times.batch_sample_time_s = time.perf_counter() - batch_sample_started
        return frontier_refresh, batch, frontier_batch_size, phase_times

    def _apply_optimizer_step(
        self,
        loss_state: MultiStepTDValueLossState,
    ) -> MultiStepTDValueOptimizerStats:
        """
        Backpropagate one loss payload and apply one optimizer update.

        Args:
            loss_state: Loss payload produced for the sampled batch.

        Returns:
            Optimizer-step diagnostics bundle.

        """
        self.optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        loss_state.total_loss.backward()
        backward_time_s = time.perf_counter() - backward_started

        optimizer_started = time.perf_counter()
        gradient_global_norm = self._compute_gradient_global_norm()
        gradient_max_abs = self._compute_gradient_max_abs()
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(self.config.gradient_clip_norm),
            )
        self.optimizer.step()
        step_lr_scheduler(self.lr_scheduler)
        optimizer_time_s = time.perf_counter() - optimizer_started
        return MultiStepTDValueOptimizerStats(
            gradient_global_norm=gradient_global_norm,
            gradient_max_abs=gradient_max_abs,
            backward_time_s=backward_time_s,
            optimizer_time_s=optimizer_time_s,
        )

    def _build_step_metrics(
        self,
        loss_state: MultiStepTDValueLossState,
    ) -> MultiStepTDValueMetrics:
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
        return MultiStepTDValueMetrics(
            step=int(self._step),
            total_loss=float(loss_state.total_loss.detach().item()),
            td_loss=float(loss_state.td_loss.detach().item()),
            lipschitz_loss=lipschitz_value,
            replay_size=len(self.replay),
        )

    def _build_step_diagnostics(
        self,
        *,
        batch: torch.Tensor,
        frontier_batch_size: int,
        frontier_refresh: MultiStepTDValueFrontierRefreshStats,
        loss_state: MultiStepTDValueLossState,
        metrics: MultiStepTDValueMetrics,
        optimizer_stats: MultiStepTDValueOptimizerStats,
        phase_times: MultiStepTDValuePhaseTimes,
        step_started: float,
        target_sync_applied: bool,
    ) -> MultiStepTDValueStepDiagnostics:
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
        return MultiStepTDValueStepDiagnostics(
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
            frontier_unique_candidate_count=int(frontier_refresh.unique_candidate_count),
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

    def compute_loss(self, states: torch.Tensor) -> MultiStepTDValueLossState:
        """
        Compute TD and optional Lipschitz losses for one batch.

        Args:
            states: Batch of states sampled from replay memory.

        Returns:
            Tensor-valued loss state for the sampled batch.

        """
        loss_state, _, _ = self._compute_loss_with_timing(states)
        return loss_state

    def _compute_loss_with_timing(
        self,
        states: torch.Tensor,
    ) -> tuple[MultiStepTDValueLossState, float, float]:
        """
        Compute one loss payload and return phase timings.

        Args:
            states: Batch of states sampled from replay memory.

        Returns:
            Tuple ``(loss_state, target_compute_time_s, model_forward_time_s)``.

        """
        batch = torch.as_tensor(states, device=self.device).long()
        target_started = time.perf_counter()
        targets = self._compute_targets(batch)
        target_compute_time_s = time.perf_counter() - target_started
        forward_started = time.perf_counter()
        predictions = self.model(batch).reshape(-1).float()
        model_forward_time_s = time.perf_counter() - forward_started
        td_loss = torch.nn.functional.mse_loss(predictions, targets.float())

        total_loss = td_loss
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

        return (
            MultiStepTDValueLossState(
                total_loss=total_loss,
                td_loss=td_loss,
                lipschitz_loss=lipschitz_loss,
                predictions=predictions,
                targets=targets,
            ),
            target_compute_time_s,
            model_forward_time_s,
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
        self.target_model.load_state_dict(unwrap_model(self.model).state_dict())
        self.target_model.eval()
        self.target_evaluator.sync_target_model(self.target_model)

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

    def _build_lr_scheduler(self) -> Any | None:
        """
        Build the optional step-based learning-rate scheduler.

        Returns:
            Scheduler instance configured from ``self.config``, or ``None`` when
            scheduling is disabled.

        """
        scheduler_spec = self.config.lr_scheduler.to_log_dict()
        scheduler_ctor = lr_scheduler_ctor_from_cfg(
            {
                "num_updates": int(self.config.num_updates),
                "lr_scheduler": scheduler_spec,
            },
            total_steps_key="num_updates",
            allow_plateau=False,
        )
        if scheduler_ctor is None:
            return None
        return scheduler_ctor(self.optimizer)

    def _current_learning_rate(self) -> float:
        """
        Return the learning rate of the first optimizer parameter group.

        Returns:
            Scalar learning rate.

        """
        return float(self.optimizer.param_groups[0]["lr"])

    def _compute_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute configured TD targets for a batch of states.

        Args:
            states: Batch of states whose targets should be computed.

        Returns:
            One-dimensional tensor of target values on ``self.device``.

        """
        batch = torch.as_tensor(states, device=self.device).long()
        return self.target_evaluator.compute_targets(batch).to(self.device)

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
        total_batch_size = self._local_batch_size()
        frontier_batch_size = 0
        batch_parts: list[torch.Tensor] = []

        if (
            self.frontier_archive is not None
            and int(self.config.frontier.batch_size) > 0
            and len(self.frontier_archive) > 0
        ):
            frontier_batch_size = min(
                self._local_frontier_batch_size(),
                total_batch_size,
                len(self.frontier_archive),
            )
            batch_parts.append(
                self.frontier_archive.sample(
                    frontier_batch_size,
                    generator=self._sampling_generator,
                    device=self.device,
                )
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

        if len(batch_parts) == 1:
            return batch_parts[0], frontier_batch_size

        batch = torch.cat(batch_parts, dim=0)
        permutation = torch.randperm(
            batch.shape[0],
            generator=self._permutation_generator,
            device=batch.device,
        )
        return batch[permutation], frontier_batch_size

    def _local_batch_size(self) -> int:
        """
        Return the optimizer batch size owned by the active rank.

        Returns:
            Rank-local optimizer batch size.

        """
        if not self.is_distributed:
            return int(self.config.replay.batch_size)
        return split_evenly(
            int(self.config.replay.batch_size),
            int(self.world_size),
            int(self.rank),
        )

    def _local_frontier_batch_size(self) -> int:
        """
        Return the frontier archive batch size owned by the active rank.

        Returns:
            Rank-local frontier batch size.

        """
        if not self.is_distributed:
            return int(self.config.frontier.batch_size)
        return split_evenly(
            int(self.config.frontier.batch_size),
            int(self.world_size),
            int(self.rank),
        )

    def _build_permutation_generator(self) -> torch.Generator:
        """
        Build a rank-aware generator used for local batch shuffling.

        Returns:
            Torch generator located on the training device when possible.

        """
        generator_device = (
            self.device if self.device.type == "cuda" else torch.device("cpu")
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(self.config.sampling.seed) + 200_000 * int(self.rank))
        return generator

    def _maybe_refresh_frontier_archive(self) -> MultiStepTDValueFrontierRefreshStats:
        """
        Refresh the optional frontier archive with suffix-filtered long walks.

        Returns:
            Summary of the archive refresh attempt for the current step.

        """
        stats = MultiStepTDValueFrontierRefreshStats()
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
                    candidate_scores = self._compute_targets(unique_states)
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
                        assert self.frontier_archive is not None
                        update_stats = self.frontier_archive.add_candidates(
                            unique_states[selected_indices],
                            unique_hashes[selected_indices],
                            candidate_scores[selected_indices],
                            score_ema_decay=float(self.config.frontier.score_ema_decay),
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
        stats: MultiStepTDValueFrontierRefreshStats,
        update_stats: FrontierArchiveUpdateStats,
    ) -> MultiStepTDValueFrontierRefreshStats:
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
        Resolve the model device for multi-step TD value learning.

        Args:
            device: Requested device or ``"auto"``.

        Returns:
            Resolved torch device.

        """
        if self.config.parallel.uses_ddp:
            local_rank = int(local_rank_from_env())
            if str(device).lower() == "auto":
                if self.graph_device.type == "cuda":
                    return self.graph_device
                return torch.device(f"cuda:{local_rank}")
            requested = torch.device(device)
            if requested.type == "cuda":
                return torch.device(f"cuda:{local_rank}")
            return requested
        if str(device).lower() == "auto":
            return self.graph_device
        return torch.device(device)

    def _validate_config(self) -> None:
        """Validate the trainer configuration."""
        self._validate_core_config()
        self._validate_frontier_config()
        self._validate_lipschitz_config()
        self._validate_parallel_config()

    def _validate_core_config(self) -> None:
        """
        Validate the base multi-step TD configuration.

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
        if (
            float(self.config.lipschitz.weight) > 0.0
            and self._resolve_device(self.config.device) != self.graph_device
        ):
            raise ValueError(
                "lipschitz regularization requires the model and graph to share "
                "the same device."
            )

    def _validate_parallel_config(self) -> None:
        """
        Validate GPU-parallel runtime requirements.

        Raises:
            ValueError: If multi-GPU learner parallelism is configured
                incompatibly.

        """
        if not self.config.parallel.uses_ddp:
            return
        resolved_device = self._resolve_device(self.config.device)
        if resolved_device.type != "cuda":
            raise ValueError(
                "parallel DDP learner mode requires a CUDA device."
            )
        if resolved_device.index is None:
            raise ValueError(
                "parallel DDP learner mode requires a concrete CUDA device index."
            )
        if not torch.cuda.is_available():
            raise ValueError(
                "parallel DDP learner mode requires CUDA availability."
            )
        if int(self.config.parallel.num_gpus) > int(torch.cuda.device_count()):
            raise ValueError(
                f"parallel.num_gpus={int(self.config.parallel.num_gpus)} exceeds "
                f"available CUDA devices ({int(torch.cuda.device_count())})."
            )
        if (
            int(self.config.parallel.world_size) > 1
            and not is_distributed_initialized()
        ):
            raise ValueError(
                "parallel DDP learner mode requires torch.distributed to be "
                "initialized. Launch through torchrun or the distributed "
                "entrypoint."
            )
        if (
            is_distributed_initialized()
            and int(distributed_world_size()) != int(self.config.parallel.world_size)
        ):
            raise ValueError(
                "distributed world size does not match parallel.num_gpus."
            )
