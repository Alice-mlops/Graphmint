# Computes aggregate metrics for exact, search, and baseline evaluations.
"""Metric helpers for evaluation tasks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from pilgrim.schemas.eval.datasets import BenchmarkItem
from pilgrim.schemas.eval.results import EvaluationItemResult, EvaluationSliceResult


def aggregate_exact_metrics(errors: Sequence[float]) -> dict[str, float]:
    """
    Aggregate exact-value errors into summary metrics.

    Args:
        errors: Signed prediction errors.

    Returns:
        Aggregate metric mapping containing count, bias, MAE, and RMSE.

    """
    if len(errors) == 0:
        return {"count": 0.0, "bias": 0.0, "mae": 0.0, "rmse": 0.0}
    count = float(len(errors))
    abs_errors = [abs(error) for error in errors]
    mse = sum(error * error for error in errors) / count
    return {
        "count": count,
        "bias": float(sum(errors) / count),
        "mae": float(sum(abs_errors) / count),
        "rmse": float(math.sqrt(mse)),
    }


def aggregate_search_metrics(
    solved_lengths: Sequence[int | None],
    *,
    prefix: str,
) -> dict[str, float]:
    """
    Aggregate solve-rate and path-length metrics for one search mode.

    Args:
        solved_lengths: Sequence of solution lengths or ``None`` for failures.
        prefix: Metric prefix such as ``"greedy"`` or ``"beam_256"``.

    Returns:
        Aggregate metric mapping for the supplied search mode.

    """
    count = len(solved_lengths)
    solved = [length for length in solved_lengths if length is not None]
    success_rate = (len(solved) / count) if count > 0 else 0.0
    mean_length = (sum(solved) / len(solved)) if solved else 0.0
    metrics = {
        f"{prefix}/count": float(count),
        f"{prefix}/success_rate": float(success_rate),
        f"{prefix}/solved_count": float(len(solved)),
        f"{prefix}/mean_length": float(mean_length),
    }
    if solved:
        metrics[f"{prefix}/best_length"] = float(min(solved))
        metrics[f"{prefix}/worst_length"] = float(max(solved))
    else:
        metrics[f"{prefix}/best_length"] = 0.0
        metrics[f"{prefix}/worst_length"] = 0.0
    return metrics


def aggregate_baseline_progress_metrics(
    deltas: Sequence[int | None],
) -> dict[str, float]:
    """
    Aggregate progress relative to baseline lengths.

    Args:
        deltas: ``found_length - baseline_length`` deltas per item, or ``None``
            when no candidate solution was found.

    Returns:
        Aggregate metric mapping for baseline comparison.

    """
    comparable = [delta for delta in deltas if delta is not None]
    if not comparable:
        return {
            "count": float(len(deltas)),
            "compared_count": 0.0,
            "mean_delta": 0.0,
            "improvement_rate": 0.0,
            "not_worse_rate": 0.0,
        }
    compared = float(len(comparable))
    improvements = sum(delta < 0 for delta in comparable)
    not_worse = sum(delta <= 0 for delta in comparable)
    return {
        "count": float(len(deltas)),
        "compared_count": compared,
        "mean_delta": float(sum(comparable) / compared),
        "improvement_rate": float(improvements / compared),
        "not_worse_rate": float(not_worse / compared),
    }


def build_family_slices(
    dataset_items: Sequence[BenchmarkItem],
    item_results: Sequence[EvaluationItemResult],
    *,
    metric_keys: Sequence[str],
) -> list[EvaluationSliceResult]:
    """
    Build family-level aggregate slices from per-item metrics.

    Args:
        dataset_items: Benchmark items used for evaluation.
        item_results: Per-item evaluation outputs in dataset order.
        metric_keys: Metric keys to average inside each family.

    Returns:
        Family-level aggregate slice results.

    """
    del dataset_items
    grouped_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item_result in item_results:
        for metric_key in metric_keys:
            value = item_result.metrics.get(metric_key)
            if isinstance(value, bool) or isinstance(value, int | float):
                grouped_metrics[item_result.family][metric_key].append(float(value))

    slices: list[EvaluationSliceResult] = []
    for family, values_by_key in sorted(grouped_metrics.items()):
        metrics: dict[str, float] = {}
        for metric_key, values in values_by_key.items():
            if values:
                metrics[f"mean/{metric_key}"] = float(sum(values) / len(values))
        slices.append(
            EvaluationSliceResult(
                slice_name=f"family/{family}",
                count=sum(1 for item in item_results if item.family == family),
                metrics=metrics,
            )
        )
    return slices


def build_distance_slices(
    dataset_items: Sequence[BenchmarkItem],
    item_results: Sequence[EvaluationItemResult],
    *,
    metric_key: str,
) -> list[EvaluationSliceResult]:
    """
    Build exact-distance slices from per-item metrics.

    Args:
        dataset_items: Benchmark items used for evaluation.
        item_results: Per-item evaluation outputs in dataset order.
        metric_key: Item metric key to average per exact distance.

    Returns:
        Distance-level aggregate slice results.

    """
    results_by_id = {item_result.item_id: item_result for item_result in item_results}
    grouped_values: dict[int, list[float]] = defaultdict(list)
    grouped_counts: dict[int, int] = defaultdict(int)

    for item in dataset_items:
        if item.exact_distance is None:
            continue
        item_result = results_by_id.get(item.item_id)
        if item_result is None:
            continue
        value = item_result.metrics.get(metric_key)
        if isinstance(value, bool) or isinstance(value, int | float):
            grouped_values[int(item.exact_distance)].append(float(value))
            grouped_counts[int(item.exact_distance)] += 1

    slices: list[EvaluationSliceResult] = []
    for distance in sorted(grouped_values):
        values = grouped_values[distance]
        mean_value = float(sum(values) / len(values)) if values else 0.0
        slices.append(
            EvaluationSliceResult(
                slice_name=f"distance/{distance}",
                count=int(grouped_counts[distance]),
                metrics={f"mean/{metric_key}": mean_value},
            )
        )
    return slices
