# Defines pydantic schemas for discounted multi-step value learning.
"""Schema models for multi-step TD value-iteration training."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .parallel import TDParallelConfig


class TDReplayBufferConfig(BaseModel):
    """
    Replay-buffer settings for multi-step TD value learning.

    Args:
        capacity: Maximum number of states kept in replay memory.
        batch_size: Number of states sampled for one optimization step.
        min_size: Minimum replay size required before training.
        warmstart_size: Number of states collected before the first update.
        refresh_size: Number of fresh states appended during replay refresh.
        refresh_stride: Number of optimizer steps between replay refresh.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    capacity: int = Field(200_000, ge=1)
    batch_size: int = Field(512, ge=1)
    min_size: int = Field(4_096, ge=1)
    warmstart_size: int = Field(16_384, ge=1)
    refresh_size: int = Field(2_048, ge=0)
    refresh_stride: int = Field(1, ge=0)


class TDRandomWalkSamplingConfig(BaseModel):
    """
    Random-walk replay sampling settings for multi-step TD value learning.

    Args:
        rw_mode: Random-walk mode passed to ``graph.random_walks``.
        rw_width: Base random-walk width used during sampling.
        rw_length: Base random-walk length used by the default schedule.
        rw_lengths: Optional explicit schedule of ``(factor, length)`` pairs.
        seed: Base random seed used for replay sampling.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rw_mode: str = "nbt"
    rw_width: int = Field(256, ge=1)
    rw_length: int = Field(24, ge=1)
    rw_lengths: tuple[tuple[float, int], ...] | None = None
    seed: int = 42


class TDLipschitzPenaltyConfig(BaseModel):
    """
    Optional Lipschitz-regularization settings for TD training.

    Args:
        weight: Multiplier applied to the Lipschitz penalty.
        max_states: Optional cap on states passed to the penalty.
        generator_indices: Optional generator subset used by the penalty.
        max_generators: Optional cap on sampled generators.
        seed: Optional random seed used by the penalty implementation.
        state_batch_size: Optional internal chunk size for the penalty.
        reduction: Reduction mode passed to the penalty helper.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    weight: float = Field(0.0, ge=0.0)
    max_states: int | None = Field(default=None, ge=1)
    generator_indices: tuple[int, ...] | None = None
    max_generators: int | None = Field(default=None, ge=1)
    seed: int | None = None
    state_batch_size: int | None = Field(default=None, ge=1)
    reduction: str = "mean"


class TDTargetSamplingConfig(BaseModel):
    """
    Optional sampled-backup settings for multi-step TD target construction.

    Args:
        enabled: Whether to replace exact full-action target backups with
            sampled backups.
        action_sample_size: Number of actions sampled at each non-root expanded
            state. ``None`` keeps exact all-action expansion below the root.
        root_action_sample_size: Number of actions sampled at the replay batch
            root states. ``None`` keeps exact all-action root expansion, which
            is often worthwhile when the generator count is small.
        action_sample_repeats: Number of independent sampled backup trees whose
            targets are averaged.
        horizon_sample_size: Optional number of TD-lambda horizons sampled from
            the truncated lambda weights. ``None`` keeps the exact lambda
            mixture over all horizons.
        seed: Base random seed used by deterministic sampled target builders.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = False
    action_sample_size: int | None = Field(default=None, ge=1)
    root_action_sample_size: int | None = Field(default=None, ge=1)
    action_sample_repeats: int = Field(1, ge=1)
    horizon_sample_size: int | None = Field(default=None, ge=1)
    seed: int = 42


class TDLearningRateSchedulerConfig(BaseModel):
    """
    Step-based learning-rate scheduler settings for multi-step TD training.

    Args:
        type: Scheduler name. ``"none"`` disables scheduling.
        t_max: Cosine-annealing period in optimizer steps.
        eta_min: Minimum learning rate reached by cosine schedules.
        warmup_steps: Number of optimizer steps used for linear warmup before
            cosine decay.
        warmup_ratio: Fraction of total updates used for warmup when
            ``warmup_steps`` is omitted.
        warmup_start_factor: Initial warmup multiplier applied to the base
            learning rate.
        t0: First cycle length for cosine warm restarts.
        t_mult: Cycle-length multiplier used after each warm restart.

    Raises:
        ValueError: If ``type`` is unsupported for this trainer.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: str = "none"
    t_max: int | None = Field(default=None, ge=1)
    eta_min: float | None = Field(default=None, ge=0.0)
    warmup_steps: int | None = Field(default=None, ge=1)
    warmup_ratio: float | None = Field(default=None, gt=0.0)
    warmup_start_factor: float | None = Field(default=None, gt=0.0)
    t0: int | None = Field(default=None, ge=1)
    t_mult: int | None = Field(default=None, ge=1)

    @property
    def scheduler_type(self) -> str:
        """Return the normalized scheduler type string."""
        return str(self.type).strip().lower()

    @model_validator(mode="after")
    def validate_config(self) -> TDLearningRateSchedulerConfig:
        """
        Validate scheduler support for the multi-step TD trainer.

        Returns:
            The validated scheduler config instance.

        Raises:
            ValueError: If ``type`` is unsupported for this trainer.

        """
        allowed_types = {
            "none",
            "null",
            "off",
            "",
            "cosine",
            "cosine_annealing",
            "cosine_warmup",
            "warmup_cosine",
            "cosine_with_warmup",
            "cosine_restarts",
            "cosine_restart",
            "warm_restarts",
        }
        if self.scheduler_type not in allowed_types:
            raise ValueError(
                "lr_scheduler.type must be one of: "
                '"none", "cosine", "cosine_warmup", or "cosine_restarts".'
            )
        return self

    def to_log_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for experiment logging."""
        return {
            "type": self.scheduler_type,
            "t_max": self.t_max,
            "eta_min": self.eta_min,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "warmup_start_factor": self.warmup_start_factor,
            "t0": self.t0,
            "t_mult": self.t_mult,
        }


class TDFrontierArchiveConfig(BaseModel):
    """
    Frontier-archive settings for multi-step TD value learning.

    Args:
        capacity: Maximum number of states stored in the archive. ``0`` disables
            the archive.
        batch_size: Number of archive states mixed into each optimizer batch.
        refresh_stride: Number of train steps between archive refresh passes.
        candidate_width: Width used when generating longer candidate walks.
        candidate_length: Walk length used for frontier candidates.
        candidate_mode: Random-walk mode used for frontier candidates.
        candidate_history_depth: Optional non-backtracking history depth for
            frontier candidate walks.
        suffix_fraction: Fraction of the walk suffix kept as frontier
            candidates.
        admissions_per_refresh: Maximum number of high-score candidates
            considered for archive admission per refresh.
        score_ema_decay: Exponential moving-average decay used when a state is
            rediscovered and its archive score is updated.
        distributed_scoring: Whether DDP ranks should coordinate frontier
            candidate scoring and archive synchronization.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    capacity: int = Field(0, ge=0)
    batch_size: int = Field(0, ge=0)
    refresh_stride: int = Field(50, ge=0)
    candidate_width: int | None = Field(default=None, ge=1)
    candidate_length: int | None = Field(default=None, ge=1)
    candidate_mode: str | None = None
    candidate_history_depth: int | None = Field(default=None, ge=0)
    suffix_fraction: float = Field(0.5, gt=0.0, le=1.0)
    admissions_per_refresh: int = Field(64, ge=1)
    score_ema_decay: float = Field(0.9, ge=0.0, le=1.0)
    distributed_scoring: bool = True


class MultiStepTDValueConfig(BaseModel):
    """
    Configuration for discounted multi-step TD value learning.

    Args:
        num_updates: Number of optimization steps run by ``fit`` by default.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        target_sync_interval: Number of optimizer steps between target syncs.
        gradient_clip_norm: Optional global gradient clipping norm.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied to future values.
        n_steps: Maximum TD backup horizon.
        td_lambda: Optional truncated TD-lambda coefficient. ``None`` uses a
            pure ``n_steps`` target.
        terminal_value: Value assigned to the central state.
        generator_indices: Optional generator subset used for backups.
        value_batch_size: Optional chunk size for target-model evaluation.
        device: Device used for the learnable model. ``"auto"`` follows the
            graph device.
        optimizer_betas: AdamW beta parameters.
        lr_scheduler: Optional step-based learning-rate scheduler.
        replay: Replay-buffer configuration.
        sampling: Random-walk replay-sampling configuration.
        frontier: Optional frontier-archive configuration.
        lipschitz: Optional Lipschitz-penalty configuration.
        target_sampling: Optional sampled-backup target-construction settings.
        parallel: GPU parallelization settings.

    Raises:
        ValueError: If replay settings exceed replay capacity.

    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    num_updates: int = Field(1_000, ge=0)
    learning_rate: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    target_sync_interval: int = Field(100, ge=1)
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    reward_per_step: float = 1.0
    discount: float = Field(1.0, ge=0.0, le=1.0)
    n_steps: int = Field(1, ge=1)
    td_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    terminal_value: float = 0.0
    generator_indices: tuple[int, ...] | None = None
    value_batch_size: int | None = Field(default=None, ge=1)
    device: str | torch.device = "auto"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_scheduler: TDLearningRateSchedulerConfig = Field(
        default_factory=TDLearningRateSchedulerConfig
    )
    replay: TDReplayBufferConfig = Field(default_factory=TDReplayBufferConfig)
    sampling: TDRandomWalkSamplingConfig = Field(
        default_factory=TDRandomWalkSamplingConfig
    )
    frontier: TDFrontierArchiveConfig = Field(default_factory=TDFrontierArchiveConfig)
    lipschitz: TDLipschitzPenaltyConfig = Field(
        default_factory=TDLipschitzPenaltyConfig
    )
    target_sampling: TDTargetSamplingConfig = Field(
        default_factory=TDTargetSamplingConfig
    )
    parallel: TDParallelConfig = Field(default_factory=TDParallelConfig)

    @property
    def target_mode(self) -> str:
        """
        Return the configured target mode.

        Returns:
            ``"td_lambda"`` when ``td_lambda`` is enabled, otherwise
            ``"n_step"``.

        """
        return "td_lambda" if self.td_lambda is not None else "n_step"

    @model_validator(mode="after")
    def validate_config(self) -> MultiStepTDValueConfig:
        """
        Validate replay-capacity consistency.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If replay thresholds exceed replay capacity.

        """
        if int(self.replay.min_size) > int(self.replay.capacity):
            raise ValueError("replay.min_size cannot exceed replay.capacity.")
        if int(self.replay.warmstart_size) > int(self.replay.capacity):
            raise ValueError("replay.warmstart_size cannot exceed replay.capacity.")
        if (
            self.parallel.uses_ddp
            and int(self.replay.batch_size) % int(self.parallel.world_size) != 0
        ):
            raise ValueError(
                "replay.batch_size must be divisible by parallel.num_gpus in DDP mode."
            )
        if (
            self.parallel.uses_ddp
            and int(self.frontier.batch_size) > 0
            and int(self.frontier.batch_size) % int(self.parallel.world_size) != 0
        ):
            raise ValueError(
                "frontier.batch_size must be divisible by parallel.num_gpus in DDP mode."
            )
        return self

    def to_log_dict(self) -> dict[str, Any]:
        """
        Return a flat dictionary suitable for experiment logging.

        Returns:
            Flat mapping with nested config values expanded into prefixed keys.

        """
        return {
            "num_updates": int(self.num_updates),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "target_sync_interval": int(self.target_sync_interval),
            "gradient_clip_norm": self.gradient_clip_norm,
            "reward_per_step": float(self.reward_per_step),
            "discount": float(self.discount),
            "n_steps": int(self.n_steps),
            "td_lambda": self.td_lambda,
            "target_mode": self.target_mode,
            "terminal_value": float(self.terminal_value),
            "generator_indices": None
            if self.generator_indices is None
            else list(self.generator_indices),
            "value_batch_size": self.value_batch_size,
            "device": str(self.device),
            "optimizer_betas": tuple(float(x) for x in self.optimizer_betas),
            "lr_scheduler.type": str(self.lr_scheduler.scheduler_type),
            "lr_scheduler.t_max": self.lr_scheduler.t_max,
            "lr_scheduler.eta_min": self.lr_scheduler.eta_min,
            "lr_scheduler.warmup_steps": self.lr_scheduler.warmup_steps,
            "lr_scheduler.warmup_ratio": self.lr_scheduler.warmup_ratio,
            "lr_scheduler.warmup_start_factor": self.lr_scheduler.warmup_start_factor,
            "lr_scheduler.t0": self.lr_scheduler.t0,
            "lr_scheduler.t_mult": self.lr_scheduler.t_mult,
            "replay.capacity": int(self.replay.capacity),
            "replay.batch_size": int(self.replay.batch_size),
            "replay.min_size": int(self.replay.min_size),
            "replay.warmstart_size": int(self.replay.warmstart_size),
            "replay.refresh_size": int(self.replay.refresh_size),
            "replay.refresh_stride": int(self.replay.refresh_stride),
            "sampling.rw_mode": str(self.sampling.rw_mode),
            "sampling.rw_width": int(self.sampling.rw_width),
            "sampling.rw_length": int(self.sampling.rw_length),
            "sampling.rw_lengths": None
            if self.sampling.rw_lengths is None
            else [tuple(item) for item in self.sampling.rw_lengths],
            "sampling.seed": int(self.sampling.seed),
            "frontier.capacity": int(self.frontier.capacity),
            "frontier.batch_size": int(self.frontier.batch_size),
            "frontier.refresh_stride": int(self.frontier.refresh_stride),
            "frontier.candidate_width": self.frontier.candidate_width,
            "frontier.candidate_length": self.frontier.candidate_length,
            "frontier.candidate_mode": self.frontier.candidate_mode,
            "frontier.candidate_history_depth": self.frontier.candidate_history_depth,
            "frontier.suffix_fraction": float(self.frontier.suffix_fraction),
            "frontier.admissions_per_refresh": int(
                self.frontier.admissions_per_refresh
            ),
            "frontier.score_ema_decay": float(self.frontier.score_ema_decay),
            "frontier.distributed_scoring": bool(self.frontier.distributed_scoring),
            "lipschitz.weight": float(self.lipschitz.weight),
            "lipschitz.max_states": self.lipschitz.max_states,
            "lipschitz.generator_indices": None
            if self.lipschitz.generator_indices is None
            else list(self.lipschitz.generator_indices),
            "lipschitz.max_generators": self.lipschitz.max_generators,
            "lipschitz.seed": self.lipschitz.seed,
            "lipschitz.state_batch_size": self.lipschitz.state_batch_size,
            "lipschitz.reduction": str(self.lipschitz.reduction),
            "target_sampling.enabled": bool(self.target_sampling.enabled),
            "target_sampling.action_sample_size": self.target_sampling.action_sample_size,
            "target_sampling.root_action_sample_size": (
                self.target_sampling.root_action_sample_size
            ),
            "target_sampling.action_sample_repeats": int(
                self.target_sampling.action_sample_repeats
            ),
            "target_sampling.horizon_sample_size": (
                self.target_sampling.horizon_sample_size
            ),
            "target_sampling.seed": int(self.target_sampling.seed),
            "parallel.mode": str(self.parallel.resolved_mode),
            "parallel.num_gpus": int(self.parallel.num_gpus),
            "parallel.backend": str(self.parallel.backend),
        }


class MultiStepTDValueMetrics(BaseModel):
    """
    Metrics reported for one multi-step TD optimizer step.

    Args:
        step: One-based optimizer step index.
        total_loss: Full loss used for backpropagation.
        td_loss: Mean-squared loss against the configured TD target.
        lipschitz_loss: Optional Lipschitz penalty value.
        replay_size: Replay-buffer size after the update.

    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=1)
    total_loss: float
    td_loss: float
    lipschitz_loss: float | None = None
    replay_size: int = Field(..., ge=0)


class MultiStepTDValueLossState(BaseModel):
    """
    Tensor-valued outputs produced when scoring one replay batch.

    Args:
        total_loss: Full differentiable loss used for backpropagation.
        td_loss: Mean-squared loss against the configured TD target.
        lipschitz_loss: Optional Lipschitz penalty tensor.
        predictions: Online-model value predictions for the sampled states.
        targets: Frozen-target TD values used as regression targets.

    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    total_loss: torch.Tensor
    td_loss: torch.Tensor
    lipschitz_loss: torch.Tensor | None = None
    predictions: torch.Tensor
    targets: torch.Tensor


class MultiStepTDValueStepDiagnostics(BaseModel):
    """
    Detailed diagnostics collected for one optimizer step.

    Args:
        step: One-based optimizer step index.
        total_loss: Full loss used for optimization.
        td_loss: Mean-squared loss against the configured TD target.
        lipschitz_loss: Optional Lipschitz penalty value.
        replay_size: Replay-buffer size after the update.
        replay_fill_ratio: Fraction of replay capacity currently filled.
        learning_rate: Optimizer learning rate used for the step.
        step_time_s: Wall-clock duration of the optimizer step.
        replay_refresh_time_s: Time spent appending replay states.
        frontier_refresh_time_s: Time spent refreshing the frontier archive.
        batch_sample_time_s: Time spent sampling the optimizer batch.
        target_compute_time_s: Time spent constructing frozen TD targets.
        model_forward_time_s: Time spent in the online forward pass.
        backward_time_s: Time spent in ``loss.backward()``.
        optimizer_time_s: Time spent in gradient clipping, optimizer step, and
            LR scheduler stepping.
        gradient_global_norm: Global L2 norm across all gradients.
        gradient_max_abs: Maximum absolute gradient entry.
        target_sync_applied: Whether the target network was synchronized.
        frontier_archive_size: Number of currently archived frontier states.
        frontier_archive_fill_ratio: Fraction of the frontier archive that is
            filled.
        frontier_batch_size: Number of frontier states mixed into the batch.
        frontier_refresh_applied: Whether frontier refresh ran before sampling
            the training batch.
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
        frontier_score_max: Maximum score of currently archived frontier states.
        batch_states: Sampled replay states used for the step.
        predictions: Online-model value predictions for ``batch_states``.
        targets: TD targets for ``batch_states``.

    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    step: int = Field(..., ge=1)
    total_loss: float
    td_loss: float
    lipschitz_loss: float | None = None
    replay_size: int = Field(..., ge=0)
    replay_fill_ratio: float = Field(..., ge=0.0, le=1.0)
    learning_rate: float
    step_time_s: float = Field(..., ge=0.0)
    replay_refresh_time_s: float = Field(0.0, ge=0.0)
    frontier_refresh_time_s: float = Field(0.0, ge=0.0)
    batch_sample_time_s: float = Field(0.0, ge=0.0)
    target_compute_time_s: float = Field(0.0, ge=0.0)
    model_forward_time_s: float = Field(0.0, ge=0.0)
    backward_time_s: float = Field(0.0, ge=0.0)
    optimizer_time_s: float = Field(0.0, ge=0.0)
    gradient_global_norm: float | None = None
    gradient_max_abs: float | None = None
    target_sync_applied: bool
    frontier_archive_size: int = Field(0, ge=0)
    frontier_archive_fill_ratio: float = Field(0.0, ge=0.0, le=1.0)
    frontier_batch_size: int = Field(0, ge=0)
    frontier_refresh_applied: bool = False
    frontier_candidate_count: int = Field(0, ge=0)
    frontier_unique_candidate_count: int = Field(0, ge=0)
    frontier_selected_count: int = Field(0, ge=0)
    frontier_admitted: int = Field(0, ge=0)
    frontier_updated: int = Field(0, ge=0)
    frontier_replaced: int = Field(0, ge=0)
    frontier_score_mean: float | None = None
    frontier_score_max: float | None = None
    batch_states: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor
