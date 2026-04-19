# Defines pydantic schemas for search-guided PPO on deterministic Cayley graphs.
"""Schema models for search-guided PPO training."""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .multistep_td_value_iteration import (
    TDLearningRateSchedulerConfig,
    TDRandomWalkSamplingConfig,
)
from .parallel import TDParallelConfig

BeamMode = Literal["simple", "advanced", "iterated"]


class SearchGuidedPPORolloutConfig(BaseModel):
    """
    On-policy rollout settings for search-guided PPO.

    Args:
        enabled: Whether to collect on-policy PPO rollouts. Disable this for
            pure beam-search distillation updates.
        num_envs: Number of parallel rollout states collected per update.
        horizon: Number of rollout steps collected per update.
        max_episode_steps: Per-environment cutoff used during one rollout.
        action_temperature: Sampling temperature for the policy distribution.
        generator_indices: Optional subset of legal generator ids.
        sampling: Random-walk sampling config used to draw rollout start states.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    num_envs: int = Field(256, ge=1)
    horizon: int = Field(32, ge=1)
    max_episode_steps: int = Field(128, ge=1)
    action_temperature: float = Field(1.0, gt=0.0)
    generator_indices: tuple[int, ...] | None = None
    sampling: TDRandomWalkSamplingConfig = Field(
        default_factory=TDRandomWalkSamplingConfig
    )


class SearchGuidedPPORewardConfig(BaseModel):
    """
    Tunable reward weights for actor-critic rollouts.

    Args:
        step_cost: Constant reward added after every active transition.
        solve_bonus: Reward added when the central state is reached.
        teacher_progress_weight: Multiplier for teacher-value progress shaping.
        teacher_progress_clip: Optional clip applied to progress shaping.
        inverse_action_penalty: Penalty for immediately taking the inverse move.
        revisit_penalty: Penalty for revisiting a previously seen state in the
            current rollout.
        search_match_bonus: Bonus when the policy matches the beam-found first
            action on rollout starts with guidance.
        search_miss_penalty: Penalty when the policy misses the beam-found first
            action on rollout starts with guidance.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step_cost: float = -1.0
    solve_bonus: float = 10.0
    teacher_progress_weight: float = 0.0
    teacher_progress_clip: float | None = Field(default=None, gt=0.0)
    inverse_action_penalty: float = 0.0
    revisit_penalty: float = 0.0
    search_match_bonus: float = 0.0
    search_miss_penalty: float = 0.0


class SearchGuidedPPOSupervisionConfig(BaseModel):
    """
    Reverse-trajectory supervision archive settings.

    Args:
        enabled: Whether reverse-trajectory archive refreshes are enabled.
        capacity: Maximum number of rows stored in the demo archive.
        batch_size: Number of archive rows sampled per optimization minibatch.
        warmstart_size: Number of rows collected before PPO updates.
        refresh_size: Number of new rows added after each PPO update.
        sampling: Random-walk config used to build reverse-trajectory targets.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    capacity: int = Field(200_000, ge=1)
    batch_size: int = Field(256, ge=1)
    warmstart_size: int = Field(40_000, ge=0)
    refresh_size: int = Field(8_000, ge=0)
    sampling: TDRandomWalkSamplingConfig = Field(
        default_factory=TDRandomWalkSamplingConfig
    )


class SearchGuidedPPOBeamSearchConfig(BaseModel):
    """
    Beam-search guidance settings used for rewards and auxiliary targets.

    Args:
        enabled: Whether beam-search guidance is active.
        beam_widths: Ordered beam widths tried for each guided search call.
        max_steps: Maximum beam-search depth.
        history_depth: Beam-search history depth.
        beam_mode: Beam-search variant passed to `cayleypy`.
        enable_tf32: Optional TF32 override for CUDA beam-search inference.
        enable_autocast: Optional autocast override.
        autocast_dtype_name: Autocast dtype name such as `"bfloat16"`.
        rollout_start_targets: Number of rollout start states per update that
            receive beam guidance and search-action rewards.
        archive_targets_per_update: Number of rollout states per update used to
            generate search-archive targets.
        archive_capacity: Maximum number of beam-derived targets retained.
        archive_batch_size: Number of beam-derived rows sampled per PPO
            minibatch.

    Raises:
        ValueError: If one of the configured beam widths is not positive.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    beam_widths: tuple[int, ...] = (64,)
    max_steps: int = Field(128, ge=1)
    history_depth: int = Field(0, ge=0)
    beam_mode: BeamMode = "iterated"
    enable_tf32: bool | None = None
    enable_autocast: bool | None = None
    autocast_dtype_name: str = "bfloat16"
    rollout_start_targets: int = Field(32, ge=0)
    archive_targets_per_update: int = Field(64, ge=0)
    archive_capacity: int = Field(100_000, ge=1)
    archive_batch_size: int = Field(256, ge=1)

    @model_validator(mode="after")
    def validate_config(self) -> SearchGuidedPPOBeamSearchConfig:
        """
        Validate configured beam widths.

        Returns:
            The validated beam-search config.

        Raises:
            ValueError: If one of the widths is not positive.

        """
        invalid_widths = [width for width in self.beam_widths if int(width) <= 0]
        if invalid_widths:
            raise ValueError(
                f"beam_search.beam_widths must be positive, got {invalid_widths!r}."
            )
        return self


class SearchGuidedPPOAuxLossConfig(BaseModel):
    """
    Auxiliary supervised loss weights mixed into PPO updates.

    Args:
        updates_per_step: Number of auxiliary-only optimizer updates when
            rollout collection is disabled.
        demo_policy_coef: Cross-entropy weight for reverse-trajectory actions.
        demo_value_coef: Value-regression weight for reverse-trajectory rows.
        search_policy_coef: Cross-entropy weight for beam-derived first actions.
        search_value_coef: Value-regression weight for beam-derived path
            lengths.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    updates_per_step: int = Field(1, ge=1)
    demo_policy_coef: float = Field(0.0, ge=0.0)
    demo_value_coef: float = Field(0.0, ge=0.0)
    search_policy_coef: float = Field(0.0, ge=0.0)
    search_value_coef: float = Field(0.0, ge=0.0)


class SearchGuidedPPOConfig(BaseModel):
    """
    Configuration for search-guided PPO on deterministic shortest-path graphs.

    Args:
        num_updates: Number of outer PPO updates run by `fit`.
        seed: Base RNG seed for rollouts and archive refreshes.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        gradient_clip_norm: Optional global gradient clipping norm.
        discount: Discount factor used in GAE and return computation.
        gae_lambda: GAE smoothing coefficient.
        clip_ratio: PPO ratio clipping threshold.
        value_clip_ratio: Optional clipping threshold for the value head.
        value_coef: Multiplier applied to the PPO value-regression term.
        entropy_coef: Entropy bonus multiplier.
        target_kl: Optional early-stop threshold on approximate KL divergence.
        num_policy_epochs: Number of SGD passes over each rollout batch.
        minibatch_size: Minibatch size used during PPO optimization.
        device: Device used by the trainable actor-critic model.
        optimizer_betas: AdamW beta parameters.
        lr_scheduler: Optional step-based learning-rate scheduler.
        rollout: On-policy rollout settings.
        reward: Reward-shaping settings.
        demo_supervision: Reverse-trajectory archive settings.
        beam_search: Beam-search guidance settings.
        auxiliary: Auxiliary supervised loss weights.
        parallel: Parallel execution settings.

    Raises:
        ValueError: If DDP is requested or the minibatch size is invalid.

    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    num_updates: int = Field(1_000, ge=0)
    seed: int = 42
    learning_rate: float = Field(3e-4, gt=0.0)
    weight_decay: float = Field(0.0, ge=0.0)
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    discount: float = Field(0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(0.95, ge=0.0, le=1.0)
    clip_ratio: float = Field(0.2, gt=0.0)
    value_clip_ratio: float | None = Field(default=None, gt=0.0)
    value_coef: float = Field(0.5, ge=0.0)
    entropy_coef: float = Field(0.01, ge=0.0)
    target_kl: float | None = Field(default=None, gt=0.0)
    num_policy_epochs: int = Field(4, ge=1)
    minibatch_size: int = Field(512, ge=1)
    device: str | torch.device = "auto"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    lr_scheduler: TDLearningRateSchedulerConfig = Field(
        default_factory=TDLearningRateSchedulerConfig
    )
    rollout: SearchGuidedPPORolloutConfig = Field(
        default_factory=SearchGuidedPPORolloutConfig
    )
    reward: SearchGuidedPPORewardConfig = Field(
        default_factory=SearchGuidedPPORewardConfig
    )
    demo_supervision: SearchGuidedPPOSupervisionConfig = Field(
        default_factory=SearchGuidedPPOSupervisionConfig
    )
    beam_search: SearchGuidedPPOBeamSearchConfig = Field(
        default_factory=SearchGuidedPPOBeamSearchConfig
    )
    auxiliary: SearchGuidedPPOAuxLossConfig = Field(
        default_factory=SearchGuidedPPOAuxLossConfig
    )
    parallel: TDParallelConfig = Field(default_factory=TDParallelConfig)

    @model_validator(mode="after")
    def validate_config(self) -> SearchGuidedPPOConfig:
        """
        Validate minibatch and parallelization constraints.

        Returns:
            The validated PPO config.

        Raises:
            ValueError: If DDP is requested or the rollout size is too small.

        """
        if self.parallel.uses_ddp:
            raise ValueError("SearchGuidedPPOTrainer does not support DDP yet.")
        rollout_rows = int(self.rollout.num_envs) * int(self.rollout.horizon)
        if bool(self.rollout.enabled) and int(self.minibatch_size) > rollout_rows:
            raise ValueError(
                "minibatch_size cannot exceed rollout.num_envs * rollout.horizon."
            )
        return self

    def to_log_dict(self) -> dict[str, Any]:
        """
        Return a flat logging dictionary.

        Returns:
            Flat dictionary with nested config values expanded into prefixed
            keys.

        """
        return {
            "num_updates": int(self.num_updates),
            "seed": int(self.seed),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": self.gradient_clip_norm,
            "discount": float(self.discount),
            "gae_lambda": float(self.gae_lambda),
            "clip_ratio": float(self.clip_ratio),
            "value_clip_ratio": self.value_clip_ratio,
            "value_coef": float(self.value_coef),
            "entropy_coef": float(self.entropy_coef),
            "target_kl": self.target_kl,
            "num_policy_epochs": int(self.num_policy_epochs),
            "minibatch_size": int(self.minibatch_size),
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
            "rollout.enabled": bool(self.rollout.enabled),
            "rollout.num_envs": int(self.rollout.num_envs),
            "rollout.horizon": int(self.rollout.horizon),
            "rollout.max_episode_steps": int(self.rollout.max_episode_steps),
            "rollout.action_temperature": float(self.rollout.action_temperature),
            "rollout.generator_indices": None
            if self.rollout.generator_indices is None
            else list(self.rollout.generator_indices),
            "rollout.sampling.rw_mode": str(self.rollout.sampling.rw_mode),
            "rollout.sampling.rw_width": int(self.rollout.sampling.rw_width),
            "rollout.sampling.rw_length": int(self.rollout.sampling.rw_length),
            "rollout.sampling.rw_lengths": None
            if self.rollout.sampling.rw_lengths is None
            else [tuple(item) for item in self.rollout.sampling.rw_lengths],
            "rollout.sampling.seed": int(self.rollout.sampling.seed),
            "reward.step_cost": float(self.reward.step_cost),
            "reward.solve_bonus": float(self.reward.solve_bonus),
            "reward.teacher_progress_weight": float(
                self.reward.teacher_progress_weight
            ),
            "reward.teacher_progress_clip": self.reward.teacher_progress_clip,
            "reward.inverse_action_penalty": float(self.reward.inverse_action_penalty),
            "reward.revisit_penalty": float(self.reward.revisit_penalty),
            "reward.search_match_bonus": float(self.reward.search_match_bonus),
            "reward.search_miss_penalty": float(self.reward.search_miss_penalty),
            "demo.enabled": bool(self.demo_supervision.enabled),
            "demo.capacity": int(self.demo_supervision.capacity),
            "demo.batch_size": int(self.demo_supervision.batch_size),
            "demo.warmstart_size": int(self.demo_supervision.warmstart_size),
            "demo.refresh_size": int(self.demo_supervision.refresh_size),
            "demo.sampling.rw_mode": str(self.demo_supervision.sampling.rw_mode),
            "demo.sampling.rw_width": int(self.demo_supervision.sampling.rw_width),
            "demo.sampling.rw_length": int(self.demo_supervision.sampling.rw_length),
            "demo.sampling.rw_lengths": None
            if self.demo_supervision.sampling.rw_lengths is None
            else [tuple(item) for item in self.demo_supervision.sampling.rw_lengths],
            "demo.sampling.seed": int(self.demo_supervision.sampling.seed),
            "beam.enabled": bool(self.beam_search.enabled),
            "beam.widths": list(self.beam_search.beam_widths),
            "beam.max_steps": int(self.beam_search.max_steps),
            "beam.history_depth": int(self.beam_search.history_depth),
            "beam.mode": str(self.beam_search.beam_mode),
            "beam.enable_tf32": self.beam_search.enable_tf32,
            "beam.enable_autocast": self.beam_search.enable_autocast,
            "beam.autocast_dtype_name": str(self.beam_search.autocast_dtype_name),
            "beam.rollout_start_targets": int(self.beam_search.rollout_start_targets),
            "beam.archive_targets_per_update": int(
                self.beam_search.archive_targets_per_update
            ),
            "beam.archive_capacity": int(self.beam_search.archive_capacity),
            "beam.archive_batch_size": int(self.beam_search.archive_batch_size),
            "aux.updates_per_step": int(self.auxiliary.updates_per_step),
            "aux.demo_policy_coef": float(self.auxiliary.demo_policy_coef),
            "aux.demo_value_coef": float(self.auxiliary.demo_value_coef),
            "aux.search_policy_coef": float(self.auxiliary.search_policy_coef),
            "aux.search_value_coef": float(self.auxiliary.search_value_coef),
            "parallel.mode": str(self.parallel.resolved_mode),
            "parallel.num_gpus": int(self.parallel.num_gpus),
            "parallel.backend": str(self.parallel.backend),
        }


class SearchGuidedPPOMetrics(BaseModel):
    """
    Compact metrics reported after one PPO update.

    Args:
        step: One-based PPO update index.
        total_loss: Mean total loss across minibatches.
        policy_loss: Mean PPO policy loss across minibatches.
        value_loss: Mean PPO value loss across minibatches.
        entropy: Mean policy entropy across minibatches.
        auxiliary_loss: Mean auxiliary supervised loss across minibatches.
        rollout_size: Number of valid rollout transitions.
        solve_rate: Fraction of rollout environments that solved.
        mean_reward: Mean rollout reward per valid transition.
        demo_archive_size: Number of rows stored in the demo archive.
        search_archive_size: Number of rows stored in the search archive.

    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=1)
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    auxiliary_loss: float
    rollout_size: int = Field(..., ge=0)
    solve_rate: float
    mean_reward: float
    demo_archive_size: int = Field(..., ge=0)
    search_archive_size: int = Field(..., ge=0)


class SearchGuidedPPOLossState(BaseModel):
    """
    Tensor-valued loss state for one PPO minibatch.

    Args:
        total_loss: Full differentiable loss used for backpropagation.
        policy_loss: PPO clipped policy loss.
        value_loss: PPO value-regression loss.
        entropy: Mean policy entropy.
        auxiliary_loss: Sum of auxiliary supervised losses.
        approx_kl: Approximate KL divergence against the rollout policy.
        clip_fraction: Fraction of samples clipped by the PPO ratio bound.

    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    auxiliary_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


class SearchGuidedPPOStepDiagnostics(BaseModel):
    """
    Detailed diagnostics emitted after one PPO update.

    Args:
        step: One-based PPO update index.
        total_loss: Mean total loss across minibatches.
        policy_loss: Mean PPO policy loss across minibatches.
        value_loss: Mean PPO value loss across minibatches.
        entropy: Mean policy entropy across minibatches.
        auxiliary_loss: Mean auxiliary supervised loss across minibatches.
        approx_kl: Mean approximate KL divergence.
        clip_fraction: Mean PPO clip fraction.
        rollout_size: Number of valid rollout transitions.
        rollout_collect_time_s: Time spent collecting the rollout.
        optimize_time_s: Time spent on PPO optimization epochs.
        step_time_s: Full wall-clock time for the update.
        solve_rate: Fraction of rollout environments that solved.
        mean_reward: Mean reward per valid transition.
        reward_step_cost_mean: Mean constant step-cost contribution.
        reward_solve_bonus_mean: Mean solve-bonus contribution.
        reward_teacher_progress_mean: Mean teacher-progress contribution.
        reward_inverse_penalty_mean: Mean inverse-action penalty contribution.
        reward_revisit_penalty_mean: Mean revisit-penalty contribution.
        reward_search_bonus_mean: Mean beam-guidance reward contribution.
        demo_archive_size: Number of rows stored in the demo archive.
        search_archive_size: Number of rows stored in the search archive.
        beam_rollout_queries: Number of beam-search queries for rollout-start
            rewards.
        beam_rollout_successes: Number of rollout-start beam searches that
            returned a path.
        beam_archive_queries: Number of beam-search queries used to augment the
            search archive.
        beam_archive_successes: Number of archive beam searches that returned a
            path.

    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=1)
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    auxiliary_loss: float
    approx_kl: float
    clip_fraction: float
    rollout_size: int = Field(..., ge=0)
    rollout_collect_time_s: float
    optimize_time_s: float
    step_time_s: float
    solve_rate: float
    mean_reward: float
    reward_step_cost_mean: float
    reward_solve_bonus_mean: float
    reward_teacher_progress_mean: float
    reward_inverse_penalty_mean: float
    reward_revisit_penalty_mean: float
    reward_search_bonus_mean: float
    demo_archive_size: int = Field(..., ge=0)
    search_archive_size: int = Field(..., ge=0)
    beam_rollout_queries: int = Field(..., ge=0)
    beam_rollout_successes: int = Field(..., ge=0)
    beam_archive_queries: int = Field(..., ge=0)
    beam_archive_successes: int = Field(..., ge=0)
