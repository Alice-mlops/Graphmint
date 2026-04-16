# Defines configuration dataclasses for Pilgrim reinforcement-learning methods.
"""Configuration models for fitted value iteration and supporting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class ReplayBufferConfig:
    """
    Configuration for the tensor replay buffer.

    Args:
        capacity: Maximum number of states kept in replay memory.
        batch_size: Number of states sampled for one optimization step.
        min_size: Minimum number of stored states required before training.
        warmstart_size: Number of states to collect before the first update.
        refresh_size: Number of fresh states appended before each update.
        refresh_stride: Number of train steps between replay refreshes.

    """

    capacity: int = 200_000
    batch_size: int = 512
    min_size: int = 4_096
    warmstart_size: int = 16_384
    refresh_size: int = 2_048
    refresh_stride: int = 1


@dataclass(slots=True)
class RandomWalkSamplingConfig:
    """
    Configuration for replay-state collection from random walks.

    Args:
        rw_mode: Random-walk mode passed to ``graph.random_walks``.
        rw_width: Base width used for one refresh call.
        rw_length: Base walk length used for the default schedule.
        rw_lengths: Optional explicit schedule of ``(factor, length)`` pairs.
        seed: Base random seed for walk sampling.

    """

    rw_mode: str = "nbt"
    rw_width: int = 256
    rw_length: int = 24
    rw_lengths: list[tuple[float, int]] | None = None
    seed: int = 42


@dataclass(slots=True)
class LipschitzPenaltyConfig:
    """
    Configuration for optional Lipschitz regularization during RL training.

    Args:
        weight: Multiplier applied to the Lipschitz loss.
        max_states: Optional cap on sampled states passed to the penalty.
        generator_indices: Optional generator subset used by the penalty.
        max_generators: Optional generator cap for the penalty.
        seed: Optional random seed used inside the penalty.
        state_batch_size: Optional internal chunk size for the penalty.
        reduction: Reduction mode passed to the penalty implementation.

    """

    weight: float = 0.0
    max_states: int | None = None
    generator_indices: list[int] | None = None
    max_generators: int | None = None
    seed: int | None = None
    state_batch_size: int | None = None
    reduction: str = "mean"


@dataclass(slots=True)
class FrontierArchiveConfig:
    """
    Configuration for a tiny archive of self-discovered frontier states.

    Args:
        capacity: Maximum number of states stored in the archive. ``0`` disables
            the archive.
        batch_size: Number of archive states mixed into each optimizer batch.
        refresh_stride: Number of train steps between archive refresh passes.
        candidate_width: Width used when generating longer candidate walks.
            ``None`` falls back to the main replay sampling width.
        candidate_length: Walk length used for frontier candidates. ``None``
            falls back to twice the main replay walk length.
        candidate_mode: Random-walk mode used for frontier candidates.
            ``None`` falls back to the main replay sampling mode.
        candidate_history_depth: Non-backtracking history depth for frontier
            candidate walks. ``None`` falls back to the candidate length for
            ``"nbt"`` walks and ``0`` otherwise.
        suffix_fraction: Fraction of the walk suffix kept as frontier
            candidates. ``1.0`` keeps the full walk and smaller values keep
            only states near the end of the walk.
        admissions_per_refresh: Maximum number of high-score candidates
            considered for archive admission per refresh.
        score_ema_decay: Exponential moving-average decay used when a state is
            rediscovered and its archive score is updated.

    """

    capacity: int = 0
    batch_size: int = 0
    refresh_stride: int = 50
    candidate_width: int | None = None
    candidate_length: int | None = None
    candidate_mode: str | None = None
    candidate_history_depth: int | None = None
    suffix_fraction: float = 0.5
    admissions_per_refresh: int = 64
    score_ema_decay: float = 0.9


@dataclass(slots=True)
class FittedValueIterationConfig:
    """
    Configuration for fitted value iteration on deterministic graphs.

    Args:
        num_updates: Number of optimization steps to run in ``fit`` by default.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        target_sync_interval: Number of optimizer steps between target syncs.
        gradient_clip_norm: Optional global gradient clipping norm.
        reward_per_step: Step cost added to the Bellman target.
        terminal_value: Value assigned to the central state.
        generator_indices: Optional generator subset used for Bellman targets.
        value_batch_size: Optional chunk size when evaluating neighbor values.
        device: Device used for the learnable model. ``"auto"`` follows CUDA
            availability and otherwise falls back to CPU.
        optimizer_betas: AdamW beta parameters.
        replay: Replay buffer configuration.
        sampling: Random-walk replay-sampling configuration.
        frontier: Optional frontier-archive configuration.
        lipschitz: Optional Lipschitz penalty configuration.

    """

    num_updates: int = 1_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    target_sync_interval: int = 100
    gradient_clip_norm: float | None = None
    reward_per_step: float = 1.0
    terminal_value: float = 0.0
    generator_indices: list[int] | None = None
    value_batch_size: int | None = None
    device: str | torch.device = "auto"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    sampling: RandomWalkSamplingConfig = field(default_factory=RandomWalkSamplingConfig)
    frontier: FrontierArchiveConfig = field(default_factory=FrontierArchiveConfig)
    lipschitz: LipschitzPenaltyConfig = field(default_factory=LipschitzPenaltyConfig)

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
            "terminal_value": float(self.terminal_value),
            "generator_indices": self.generator_indices,
            "value_batch_size": self.value_batch_size,
            "device": str(self.device),
            "optimizer_betas": tuple(float(x) for x in self.optimizer_betas),
            "replay.capacity": int(self.replay.capacity),
            "replay.batch_size": int(self.replay.batch_size),
            "replay.min_size": int(self.replay.min_size),
            "replay.warmstart_size": int(self.replay.warmstart_size),
            "replay.refresh_size": int(self.replay.refresh_size),
            "replay.refresh_stride": int(self.replay.refresh_stride),
            "sampling.rw_mode": str(self.sampling.rw_mode),
            "sampling.rw_width": int(self.sampling.rw_width),
            "sampling.rw_length": int(self.sampling.rw_length),
            "sampling.rw_lengths": self.sampling.rw_lengths,
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
            "lipschitz.weight": float(self.lipschitz.weight),
            "lipschitz.max_states": self.lipschitz.max_states,
            "lipschitz.generator_indices": self.lipschitz.generator_indices,
            "lipschitz.max_generators": self.lipschitz.max_generators,
            "lipschitz.seed": self.lipschitz.seed,
            "lipschitz.state_batch_size": self.lipschitz.state_batch_size,
            "lipschitz.reduction": str(self.lipschitz.reduction),
        }
