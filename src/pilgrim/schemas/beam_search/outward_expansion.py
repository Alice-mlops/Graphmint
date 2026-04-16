# Defines schema models for outward beam-expansion workflows.
"""Schemas for outward beam-expansion search."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BeamMode = Literal["simple", "advanced", "iterated"]
OutwardExpansionStatus = Literal[
    "not_started",
    "not_implemented",
    "completed",
    "failed",
]


class OutwardExpansionConfig(BaseModel):
    """
    Runtime configuration for outward beam expansion.

    Args:
        beam_width: Maximum number of frontier states retained after each
            expansion step.
        max_steps: Maximum number of outward-expansion steps to execute before
            terminating the run.
        history_depth: Number of prior hash layers to ban from revisiting in
            non-simple beam modes.
        beam_mode: Beam-search variant used by the outward expansion routine.
        candidate_pool_size: Maximum number of highest-scoring candidates
            retained across the full run.
        return_paths: Whether retained candidates should include restored beam
            paths from the start state.
        path_device: Device used to store intermediate path-restoration hash
            layers. Use ``"auto"`` to choose based on ``return_paths``.
        verbose: Verbosity level for runtime progress logging.

    """

    model_config = ConfigDict(extra="forbid")

    beam_width: int = Field(
        1024,
        ge=1,
        description=(
            "Maximum number of frontier states retained after each expansion step."
        ),
    )
    max_steps: int = Field(
        128,
        ge=0,
        description=(
            "Maximum number of outward-expansion steps to execute before "
            "terminating the run."
        ),
    )
    history_depth: int = Field(
        0,
        ge=0,
        description=(
            "Number of prior hash layers to ban from revisiting in non-simple "
            "beam modes."
        ),
    )
    beam_mode: BeamMode = Field(
        "iterated",
        description="Beam-search variant used by the outward expansion routine. "
        "one of 'simple', 'advanced', 'iterated'.",
    )
    candidate_pool_size: int = Field(
        1024,
        ge=1,
        description=(
            "Maximum number of highest-scoring candidates retained across the full run."
        ),
    )
    return_paths: bool = Field(
        True,
        description=(
            "Whether retained candidates should include restored beam paths "
            "from the start state."
        ),
    )
    path_device: str = Field(
        "cpu",
        description=(
            "Device used to store intermediate path-restoration hash layers. "
            "Use 'auto' to choose based on return_paths."
        ),
    )
    verbose: int = Field(
        0,
        ge=0,
        description="Verbosity level for runtime progress logging.",
    )


class OutwardExpansionCandidate(BaseModel):
    """
    One retained outward-expansion candidate.

    Args:
        depth: Expansion depth where the candidate was retained.
        state: Candidate state payload.
        score: Model score used to rank the candidate, when available.
        beam_path: Beam path that reached the candidate, when retained.
        exact_distance: Optional exact distance filled in by later solvers.
        metadata: Extra serializable candidate metadata.

    """

    model_config = ConfigDict(extra="forbid")

    depth: int = Field(..., ge=0)
    state: tuple[int, ...] = Field(default_factory=tuple)
    score: float | None = None
    beam_path: list[int] | None = None
    exact_distance: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutwardExpansionResult(BaseModel):
    """
    Aggregate result payload for outward beam expansion.

    Args:
        start_state: Starting state used for the outward expansion run.
        status: Lifecycle status of the run.
        config: Config used to build the run.
        candidates: Retained outward candidates.
        termination_reason: Optional human-readable stop reason.
        metadata: Extra serializable run-level metadata.

    """

    model_config = ConfigDict(extra="forbid")

    start_state: tuple[int, ...] = Field(default_factory=tuple)
    status: OutwardExpansionStatus = "not_started"
    config: OutwardExpansionConfig | None = None
    candidates: list[OutwardExpansionCandidate] = Field(default_factory=list)
    termination_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
