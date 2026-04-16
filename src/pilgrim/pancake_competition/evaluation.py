"""Evaluation helpers for pancake competition submissions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd


def _path_len(path: Any) -> int:
    """Return number of moves in a dotted path string."""
    if path is None or pd.isna(path):
        return 0
    text = str(path).strip()
    if not text:
        return 0
    return text.count(".") + 1


def compute_stats_by_n(
    test_df: pd.DataFrame,
    heuristic_paths: Sequence[str],
    submission_rows: Sequence[Mapping[str, Any]],
    *,
    filter_n: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-``n`` score/prob_step statistics and merged row-level stats.

    Returns:
        Tuple ``(by_n, stat_df)``:
        - ``by_n``: aggregated metrics by ``n`` with columns
          ``n, sum_n, score, prob_step, potential``.
        - ``stat_df``: row-level merged frame with columns
          ``id, permutation, n, prob_step, solution, score``.

    """
    if len(test_df) != len(heuristic_paths):
        raise ValueError("test_df and heuristic_paths must have the same length.")

    base_df = test_df[["id", "permutation"]].copy()
    permutations = list(base_df["permutation"].tolist())
    base_df["n"] = [str(x).count(",") + 1 for x in permutations]
    base_df["prob_step"] = pd.Series(heuristic_paths[: len(base_df)]).map(_path_len)

    sub_df = pd.DataFrame(submission_rows)[["id", "solution"]].copy()

    stat_df = base_df.merge(sub_df, on="id", how="left")
    stat_df["solution"] = stat_df["solution"].fillna("")
    stat_df["score"] = stat_df["solution"].map(_path_len)

    if filter_n is not None:
        stat_df = stat_df[stat_df["n"].isin(list(filter_n))].copy()

    by_n_raw = stat_df.groupby("n", sort=True, as_index=False).agg(
        sum_n=("n", "sum"),
        score=("score", "sum"),
        prob_step=("prob_step", "sum"),
    )
    by_n = cast(pd.DataFrame, by_n_raw.reset_index(drop=True))
    by_n["potential"] = by_n["score"] - by_n["prob_step"]
    return by_n, cast(pd.DataFrame, stat_df)


def print_stats_by_n(by_n: pd.DataFrame) -> None:
    """Print ``by_n`` in the notebook-friendly format."""
    for n, sum_n, score, prob_step, potential in zip(
        by_n["n"],
        by_n["sum_n"],
        by_n["score"],
        by_n["prob_step"],
        by_n["potential"],
        strict=False,
    ):
        print(
            f"n: {n} | sum n: {sum_n} | score: {score} | "
            f"prob step: {prob_step} | potential: {potential}"
        )

    print()
    print(
        f"sum n: {by_n['sum_n'].sum()} | "
        f"score: {by_n['score'].sum()} | "
        f"prob step: {by_n['prob_step'].sum()} | "
        f"sum potential: {by_n['potential'].sum()}"
    )
