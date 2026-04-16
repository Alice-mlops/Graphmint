# Flattens structured evaluation results into logger-friendly metric maps.
"""Flattening helpers for evaluation results."""

from __future__ import annotations

from pilgrim.schemas.eval.results import EvaluationRunResult, EvaluationTaskResult


def flatten_task_result(task_result: EvaluationTaskResult) -> dict[str, float]:
    """
    Flatten one task result into a namespaced metric dictionary.

    Args:
        task_result: Structured task result to flatten.

    Returns:
        Flat metric mapping suitable for Aim or stdout loggers.

    """
    metrics = {
        f"{task_result.namespace}/{task_result.task_name}/{name}": float(value)
        for name, value in task_result.metrics.items()
    }
    for slice_result in task_result.slices:
        for name, value in slice_result.metrics.items():
            metrics[
                (
                    f"{task_result.namespace}/{task_result.task_name}/"
                    f"{slice_result.slice_name}/{name}"
                )
            ] = float(value)
    return metrics


def flatten_run_result(run_result: EvaluationRunResult) -> dict[str, float]:
    """
    Flatten all task results contained in one evaluation run.

    Args:
        run_result: Structured run result to flatten.

    Returns:
        Flat metric mapping across all contained tasks.

    """
    metrics: dict[str, float] = {}
    for task_result in run_result.results:
        metrics.update(flatten_task_result(task_result))
    return metrics
