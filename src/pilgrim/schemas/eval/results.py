# Defines evaluation result schemas and metric containers.
"""Result schemas for evaluation tasks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MetricScalar = float | int | bool | str | None


class EvaluationItemResult(BaseModel):
    """
    Per-item evaluation output.

    Args:
        item_id: Stable item identifier matching the source dataset.
        family: Family/slice name copied from the dataset item.
        metrics: Flat per-item metric mapping.
        metadata: Extra serializable per-item payload.

    """

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(..., min_length=1)
    family: str = "default"
    metrics: dict[str, MetricScalar] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationSliceResult(BaseModel):
    """
    Aggregated metrics for one slice of a benchmark dataset.

    Args:
        slice_name: Stable slice identifier.
        count: Number of items covered by the slice.
        metrics: Aggregate metric mapping.
        metadata: Extra serializable slice metadata.

    """

    model_config = ConfigDict(extra="forbid")

    slice_name: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationTaskResult(BaseModel):
    """
    Aggregate result for one evaluation task.

    Args:
        task_name: Stable task identifier.
        task_type: Task category such as ``"exact"`` or ``"search"``.
        namespace: Metric namespace prefix for loggers.
        dataset_name: Source benchmark dataset name.
        metrics: Aggregate metric mapping.
        slices: Optional grouped slice metrics.
        items: Optional per-item result payload.
        metadata: Extra serializable task metadata.

    """

    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    namespace: str = Field(..., min_length=1)
    dataset_name: str = Field(..., min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)
    slices: list[EvaluationSliceResult] = Field(default_factory=list)
    items: list[EvaluationItemResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunResult(BaseModel):
    """
    Collection of task results emitted by one evaluation run.

    Args:
        results: Ordered task results produced by the runner.
        metadata: Extra serializable run-level metadata.

    """

    model_config = ConfigDict(extra="forbid")

    results: list[EvaluationTaskResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
