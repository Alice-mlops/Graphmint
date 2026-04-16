# Defines validated configuration schemas for evaluation tasks and logging.
"""Configuration schemas for evaluation tasks."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationTaskType(str, Enum):
    """Supported evaluation task categories."""

    EXACT = "exact"
    SEARCH = "search"
    BASELINE = "baseline"


class SearchEvalConfig(BaseModel):
    """
    Search-time settings for greedy rollout and beam evaluation.

    Args:
        rollout_max_steps: Max steps for direct greedy rollout.
        beam_widths: Beam widths to evaluate. Empty disables beam search.
        beam_max_steps: Max steps for beam search.
        history_depth: Beam-search history depth.
        enable_tf32: Optional TF32 override for compatible devices.
        enable_autocast: Optional autocast override.
        autocast_dtype_name: Autocast dtype name passed to beam helpers.

    Raises:
        ValueError: If any configured width or step count is invalid.

    """

    model_config = ConfigDict(extra="forbid")

    rollout_max_steps: int = Field(128, ge=0)
    beam_widths: tuple[int, ...] = ()
    beam_max_steps: int = Field(128, ge=0)
    history_depth: int = Field(0, ge=0)
    enable_tf32: bool | None = None
    enable_autocast: bool | None = None
    autocast_dtype_name: str = "bfloat16"

    @model_validator(mode="after")
    def validate_config(self) -> SearchEvalConfig:
        """
        Validate beam-width configuration.

        Returns:
            The validated search config.

        Raises:
            ValueError: If beam widths are non-positive.

        """
        invalid_widths = [width for width in self.beam_widths if int(width) <= 0]
        if invalid_widths:
            raise ValueError(f"beam widths must be positive, got {invalid_widths!r}.")
        return self


class EvaluationLoggingConfig(BaseModel):
    """
    Logging configuration for evaluation runs.

    Args:
        stdout_enabled: Whether to emit compact stdout summaries.
        aim_enabled: Whether to emit Aim metrics.
        aim_repo: Aim backend URI such as ``aim://127.0.0.1:53800``.
        aim_experiment: Aim experiment name.
        aim_tags: Aim run tags.
        stdout_prefix: Prefix prepended to stdout summaries.
        log_slices: Whether grouped slice metrics should also be logged.

    """

    model_config = ConfigDict(extra="forbid")

    stdout_enabled: bool = True
    aim_enabled: bool = False
    aim_repo: str | None = None
    aim_experiment: str = "pilgrim-eval"
    aim_tags: tuple[str, ...] = ("pilgrim-eval",)
    stdout_prefix: str = "[eval]"
    log_slices: bool = True


class EvaluationTaskConfig(BaseModel):
    """
    Shared task configuration for evaluation runners.

    Args:
        name: Stable task identifier.
        task_type: Evaluation task category.
        namespace: Metric namespace such as ``"eval/exact"``.
        track_families: Whether to produce family-sliced aggregates.
        track_distances: Whether exact evaluators should also emit
            distance-stratified slices.
        search: Optional search settings for search-oriented tasks.

    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    task_type: EvaluationTaskType
    namespace: str = Field(..., min_length=1)
    track_families: bool = True
    track_distances: bool = False
    search: SearchEvalConfig | None = None
