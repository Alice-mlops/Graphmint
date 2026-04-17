# Defines pydantic config for RL GPU parallelization modes.
"""Schema models for multi-GPU RL training modes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TDParallelConfig(BaseModel):
    """
    Runtime settings for RL GPU parallelization.

    Args:
        mode: Parallelization strategy. ``"auto"`` resolves to ``"ddp"`` when
            ``num_gpus > 1`` and to ``"single"`` otherwise.
        num_gpus: Requested number of learner GPUs.
        backend: Distributed backend used by ``torch.distributed``.
        broadcast_buffers: Whether DDP should broadcast buffers from rank zero.
        find_unused_parameters: Whether DDP should search for unused params.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    mode: Literal["auto", "single", "ddp"] = "auto"
    num_gpus: int = Field(1, ge=1)
    backend: Literal["nccl", "gloo"] = "nccl"
    broadcast_buffers: bool = False
    find_unused_parameters: bool = False

    @property
    def resolved_mode(self) -> Literal["single", "ddp"]:
        """
        Return the resolved parallelization mode.

        Returns:
            ``"single"`` or ``"ddp"`` after expanding ``"auto"``.

        """
        if self.mode == "auto":
            return "ddp" if int(self.num_gpus) > 1 else "single"
        return self.mode

    @property
    def uses_ddp(self) -> bool:
        """
        Return whether DDP learner parallelism is enabled.

        Returns:
            ``True`` when the trainer should run one learner per GPU.

        """
        return self.resolved_mode == "ddp"

    @property
    def world_size(self) -> int:
        """
        Return the effective number of learner processes.

        Returns:
            Number of DDP learner ranks, or ``1`` for single-device runs.

        """
        return int(self.num_gpus) if self.uses_ddp else 1

    @property
    def uses_secondary_gpus(self) -> bool:
        """
        Return whether the removed secondary-evaluator mode is active.

        Returns:
            Always ``False``. This compatibility property remains so older code
            paths fail closed rather than silently re-enabling the removed mode.

        """
        return False

    @property
    def num_evaluator_gpus(self) -> int:
        """
        Return the number of evaluator GPUs in compatibility mode.

        Returns:
            Always ``0`` because secondary-GPU evaluation is disabled.

        """
        return 0

    @model_validator(mode="after")
    def validate_config(self) -> TDParallelConfig:
        """
        Validate the requested GPU-parallel configuration.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If the chosen mode is inconsistent with ``num_gpus``.

        """
        if self.resolved_mode == "single" and int(self.num_gpus) != 1:
            raise ValueError('parallel.mode="single" requires parallel.num_gpus=1.')
        return self


# Backward-compatible alias used by existing notebooks and imports.
TDSecondaryGpuEvalConfig = TDParallelConfig
