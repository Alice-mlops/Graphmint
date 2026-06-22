# Defines schema models for advantage-weighted actor-critic losses.
"""Configuration schema for generic discrete AWAC policy-improvement losses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AWACSetLossMode = Literal["set_probability", "uniform_targets"]


class AWACConfig(BaseModel):
    """
    Configuration for discrete advantage-weighted actor-critic losses.

    Args:
        advantage_temperature: Temperature in ``exp(A / temperature)``.
        max_weight: Maximum per-row AWAC weight after exponentiation.
        normalize_weights: Whether to divide AWAC weights by their batch mean.
        min_advantage: Optional lower clip for input advantages.
        max_advantage: Optional upper clip for input advantages.
        set_loss_mode: How to train rows with several equally good actions.
        value_coef: Weight of value regression loss.
        entropy_coef: Weight of policy entropy bonus.
        policy_anchor_coef: Weight of ``KL(reference || current)``.
        bad_action_coef: Weight for probability-mass penalty on known bad actions.
        bad_action_margin_coef: Weight for pairwise margin against known bad
            actions.
        bad_action_margin: Margin used by the bad-action pairwise loss.
        eps: Numerical epsilon used in probability clamps.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    advantage_temperature: float = Field(1.0, gt=0.0)
    max_weight: float = Field(20.0, gt=0.0)
    normalize_weights: bool = True
    min_advantage: float | None = None
    max_advantage: float | None = None
    set_loss_mode: AWACSetLossMode = "set_probability"
    value_coef: float = Field(0.0, ge=0.0)
    entropy_coef: float = Field(0.0, ge=0.0)
    policy_anchor_coef: float = Field(0.0, ge=0.0)
    bad_action_coef: float = Field(0.0, ge=0.0)
    bad_action_margin_coef: float = Field(0.0, ge=0.0)
    bad_action_margin: float = Field(0.0, ge=0.0)
    eps: float = Field(1.0e-8, gt=0.0)

    def to_log_dict(self) -> dict[str, float | bool | str | None]:
        """
        Return a flat logging dictionary.

        Returns:
            Flat dictionary with scalar configuration values.
        """
        return {
            "awac.advantage_temperature": float(self.advantage_temperature),
            "awac.max_weight": float(self.max_weight),
            "awac.normalize_weights": bool(self.normalize_weights),
            "awac.min_advantage": self.min_advantage,
            "awac.max_advantage": self.max_advantage,
            "awac.set_loss_mode": str(self.set_loss_mode),
            "awac.value_coef": float(self.value_coef),
            "awac.entropy_coef": float(self.entropy_coef),
            "awac.policy_anchor_coef": float(self.policy_anchor_coef),
            "awac.bad_action_coef": float(self.bad_action_coef),
            "awac.bad_action_margin_coef": float(self.bad_action_margin_coef),
            "awac.bad_action_margin": float(self.bad_action_margin),
            "awac.eps": float(self.eps),
        }


__all__ = ["AWACConfig", "AWACSetLossMode"]
