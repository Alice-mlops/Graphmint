# Defines pydantic config for secondary-GPU target evaluation in RL training.
"""Schema models for RL secondary-GPU evaluation backends."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TDSecondaryGpuEvalConfig(BaseModel):
    """
    Runtime settings for secondary-GPU frozen-target evaluation.

    Args:
        num_gpus: Total number of GPUs reserved for the trainer, including the
            learner on ``cuda:0``. Values greater than ``1`` enable secondary
            GPUs for target evaluation.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    num_gpus: int = Field(1, ge=1)

    @property
    def uses_secondary_gpus(self) -> bool:
        """
        Return whether secondary GPUs are enabled for evaluation.

        Returns:
            ``True`` when at least one evaluator GPU should be used.

        """
        return int(self.num_gpus) > 1

    @property
    def num_evaluator_gpus(self) -> int:
        """
        Return the number of evaluator GPUs.

        Returns:
            Number of secondary GPUs reserved for frozen-target evaluation.

        """
        return max(0, int(self.num_gpus) - 1)
