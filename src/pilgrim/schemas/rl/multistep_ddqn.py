# Defines pydantic schemas for discounted multi-step Double-DQN training.
"""Schema models for multi-step Double-DQN training."""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .multistep_td_value_iteration import (
    TDFrontierArchiveConfig,
    TDLearningRateSchedulerConfig,
    TDLipschitzPenaltyConfig,
    TDRandomWalkSamplingConfig,
    TDReplayBufferConfig,
)
from .parallel import TDParallelConfig


class TDBehaviorPolicyConfig(BaseModel):
    """
    Behavior-policy settings used to sample DDQN transitions.

    Args:
        mode: Action-sampling mode. ``"uniform"`` ignores the current policy and
            samples allowed actions uniformly. ``"epsilon_greedy"`` follows the
            online model greedily with random exploration.
        epsilon: Exploration probability used when ``mode`` is
            ``"epsilon_greedy"``.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    mode: Literal["uniform", "epsilon_greedy"] = "uniform"
    epsilon: float = Field(1.0, ge=0.0, le=1.0)


class MultiStepDDQNConfig(BaseModel):
    """
    Configuration for multi-step Double-DQN on deterministic graphs.

    Args:
        num_updates: Number of optimization steps run by ``fit`` by default.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        target_sync_interval: Number of optimizer steps between target syncs.
        gradient_clip_norm: Optional global gradient clipping norm.
        reward_per_step: Step cost added to sampled transitions.
        discount: Discount factor applied to future Q-values.
        n_steps: Maximum sampled backup horizon.
        td_lambda: Reserved field for future Watkins-style lambda targets.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generator ids used by the policy.
        value_batch_size: Optional chunk size for Q-model evaluation.
        device: Device used for the learnable model. ``"auto"`` follows the
            graph device.
        optimizer_betas: AdamW beta parameters.
        lr_scheduler: Optional step-based learning-rate scheduler.
        replay: Replay-buffer configuration.
        sampling: Random-walk replay-sampling configuration.
        frontier: Optional frontier-archive configuration.
        lipschitz: Optional Lipschitz-penalty configuration.
        parallel: GPU parallelization settings.
        behavior: Behavior-policy configuration for sampled transitions.

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
    parallel: TDParallelConfig = Field(default_factory=TDParallelConfig)
    behavior: TDBehaviorPolicyConfig = Field(default_factory=TDBehaviorPolicyConfig)

    @property
    def target_mode(self) -> str:
        """
        Return the configured target mode.

        Returns:
            String describing the sampled n-step DDQN target.

        """
        return f"double_q_n_step_{int(self.n_steps)}"

    @model_validator(mode="after")
    def validate_config(self) -> MultiStepDDQNConfig:
        """
        Validate replay-capacity and DDP consistency.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If replay thresholds exceed replay capacity or batch
                sizes are incompatible with DDP.

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
            "parallel.mode": str(self.parallel.resolved_mode),
            "parallel.num_gpus": int(self.parallel.num_gpus),
            "parallel.backend": str(self.parallel.backend),
            "behavior.mode": str(self.behavior.mode),
            "behavior.epsilon": float(self.behavior.epsilon),
        }


class MultiStepDDQNMetrics(BaseModel):
    """
    Metrics reported for one multi-step DDQN optimizer step.

    Args:
        step: One-based optimizer step index.
        total_loss: Full loss used for backpropagation.
        td_loss: Mean-squared loss against the sampled DDQN target.
        lipschitz_loss: Optional Lipschitz penalty value.
        replay_size: Replay-buffer size after the update.

    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=1)
    total_loss: float
    td_loss: float
    lipschitz_loss: float | None = None
    replay_size: int = Field(..., ge=0)


class MultiStepDDQNLossState(BaseModel):
    """
    Tensor-valued outputs produced when scoring one replay batch.

    Args:
        total_loss: Full differentiable loss used for backpropagation.
        td_loss: Mean-squared loss against the sampled DDQN target.
        lipschitz_loss: Optional Lipschitz penalty tensor.
        predictions: Predicted ``Q(s, a)`` values for sampled actions.
        targets: Sampled Double-DQN targets.
        actions: Sampled behavior actions aligned with ``predictions``.

    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    total_loss: torch.Tensor
    td_loss: torch.Tensor
    lipschitz_loss: torch.Tensor | None = None
    predictions: torch.Tensor
    targets: torch.Tensor
    actions: torch.Tensor


class MultiStepDDQNStepDiagnostics(BaseModel):
    """
    Detailed diagnostics collected for one optimizer step.

    Args:
        step: One-based optimizer step index.
        total_loss: Full loss used for optimization.
        td_loss: Mean-squared loss against the sampled DDQN target.
        lipschitz_loss: Optional Lipschitz penalty value.
        replay_size: Replay-buffer size after the update.
        replay_fill_ratio: Fraction of replay capacity currently filled.
        learning_rate: Optimizer learning rate used for the step.
        step_time_s: Wall-clock duration of the optimizer step.
        replay_refresh_time_s: Time spent appending replay states.
        frontier_refresh_time_s: Time spent refreshing the frontier archive.
        batch_sample_time_s: Time spent sampling the optimizer batch.
        target_compute_time_s: Time spent constructing sampled DDQN targets.
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
        predictions: Predicted ``Q(s, a)`` values for sampled actions.
        targets: Sampled DDQN targets for ``batch_states``.
        actions: Sampled behavior actions aligned with ``predictions``.

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
    actions: torch.Tensor
