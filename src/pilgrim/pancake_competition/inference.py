"""Inference helpers for the pancake Kaggle competition notebooks."""

from __future__ import annotations

import copy
import gc
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from cayleypy import CayleyGraph
from torch import nn
from tqdm import tqdm


def parse_perm(permutation: str) -> list[int]:
    """Parse a comma-separated permutation string into integers."""
    return [int(x) for x in str(permutation).split(",")]


def path_len(path: str) -> int:
    """Return number of moves in a dotted path string."""
    if path is None:
        return 0
    text = str(path).strip()
    if not text:
        return 0
    return text.count(".") + 1


def path_edges(path: str) -> int:
    """Return number of separators in path (moves - 1)."""
    length = path_len(path)
    return max(length - 1, 0)


@dataclass(slots=True)
class BeamInferenceConfig:
    """
    Runtime controls for targeted beam-search inference.

    Args:
        beam_width: Beam width passed to ``solve_fn``.
        history_depth: History depth passed to ``solve_fn``.
        enable_tf32: Whether TF32 is enabled for CUDA operations.
        enable_autocast: Whether autocast is enabled during inference.
        autocast_dtype: Dtype used for autocast.
        autocast_device_type: Device type used for autocast.
        max_steps_factor: Multiplier used to cap max search steps by ``n``.
        compile_once_per_n: If ``True``, compile each model once per ``n``.
        free_graph_memory: If ``True``, call ``graph.free_memory()`` each step.
        clear_cuda_cache: If ``True``, call ``torch.cuda.empty_cache()``.
        run_gc_collect: If ``True``, call ``gc.collect()`` after each attempt.
        require_not_longer_than_previous: Acceptance gate toggle. If ``True``,
            a solution is accepted only when ``sol_len <= prev_len``; if
            ``False``, all solved candidates are accepted.
        checkpoint_path: Optional path for periodic checkpointing of
            ``submission_rows``. When set, :func:`run_targeted_beam_inference`
            will persist the full ``submission_rows`` list every
            ``checkpoint_every_attempts`` successful attempts.
        checkpoint_every_attempts: Checkpoint period in attempted solves.
            Only used when ``checkpoint_path`` is set.

    """

    beam_width: int = 2**10
    history_depth: int = 0
    enable_tf32: bool = True
    enable_autocast: bool = True
    autocast_dtype: torch.dtype = torch.bfloat16
    autocast_device_type: str = "cuda"
    max_steps_factor: int = 2
    compile_once_per_n: bool = True
    free_graph_memory: bool = True
    clear_cuda_cache: bool = True
    run_gc_collect: bool = True
    require_not_longer_than_previous: bool = True
    checkpoint_path: str | Path | None = None
    checkpoint_every_attempts: int = 1


@dataclass(slots=True)
class BeamInferenceStats:
    """
    Aggregated counters produced by :func:`run_targeted_beam_inference`.

    Attributes:
        attempted: Number of rows where beam solve was attempted.
        accepted: Number of attempts that passed the acceptance gate and
            updated ``submission_rows[i]["solution"]``.
        longer_rejected: Number of attempts rejected specifically because
            ``sol_len > prev_len`` when the gate is enabled.
        oom_count: Number of attempts that raised CUDA OOM.
        shorter_than_previous: Counter for ``sol_len < prev_len``.
        shorter_than_heuristic: Counter for ``sol_len < heur_len``.
        no_longer_than_previous: Counter for ``sol_len <= prev_len``.

    """

    attempted: int = 0
    accepted: int = 0
    longer_rejected: int = 0
    oom_count: int = 0
    shorter_than_previous: int = 0
    shorter_than_heuristic: int = 0
    no_longer_than_previous: int = 0


def load_or_create_submission_rows(
    rows: list[Any],
    heuristic_paths: list[str],
    *,
    base_pkl: str | Path | None = None,
    deep_copy: bool = True,
) -> list[dict[str, Any]]:
    """Load cached submission rows or create default rows from heuristics."""
    if len(rows) != len(heuristic_paths):
        raise ValueError("rows and heuristic_paths must have the same length.")

    data: list[dict[str, Any]]
    base_path = Path(base_pkl) if base_pkl is not None else None
    if base_path is not None and base_path.exists():
        with base_path.open("rb") as f:
            data = pickle.load(f)
        if len(data) != len(rows):
            raise ValueError(
                f"Loaded {len(data)} rows from {base_path}, expected {len(rows)}."
            )
    else:
        data = []
        for i, row in enumerate(rows):
            data.append({
                "id": int(row.id),
                "permutation": row.permutation,
                "solution": heuristic_paths[i],
            })
        if base_path is not None:
            base_path.parent.mkdir(parents=True, exist_ok=True)
            with base_path.open("wb") as f:
                pickle.dump(data, f)

    return copy.deepcopy(data) if deep_copy else data


def save_submission_rows(
    submission_rows: list[dict[str, Any]], path: str | Path
) -> Path:
    """Persist submission rows into a pickle file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(submission_rows, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(out_path)
    return out_path


def run_targeted_beam_inference(
    *,
    rows: list[Any],
    submission_rows: list[dict[str, Any]],
    heuristic_paths: list[str],
    target_n: set[int],
    models_dict: dict[int, nn.Module],
    graphs_dict: dict[int, CayleyGraph],
    solve_fn: Any,
    device: torch.device | str,
    beam_run: Any | None = None,
    config: BeamInferenceConfig | None = None,
    enable_autocast: bool = True,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> BeamInferenceStats:
    """
    Run beam inference for selected sizes and update submission rows.

    For each eligible row, this function computes a candidate ``solution`` via
    ``solve_fn`` and compares its length (``sol_len``) against both the current
    submission length (``prev_len``) and the heuristic baseline length
    (``heur_len``). The counters are independent:
    - ``shorter_than_previous`` counts ``sol_len < prev_len``.
    - ``shorter_than_heuristic`` counts ``sol_len < heur_len``.
    - ``no_longer_than_previous`` counts ``sol_len <= prev_len``.

    Acceptance is controlled by
    ``config.require_not_longer_than_previous``:
    ``accepted = not (require_not_longer_than_previous and sol_len > prev_len)``.
    Accepted solutions replace ``submission_rows[i]["solution"]``.

    Args:
        rows: Competition rows. Each row must expose ``permutation``.
        submission_rows: Mutable output rows containing the current ``solution``
            for each input row.
        heuristic_paths: Baseline heuristic solutions aligned with ``rows``.
        target_n: Set of permutation sizes ``n`` to run beam search for.
        models_dict: Mapping ``n -> model`` used for inference.
        graphs_dict: Mapping ``n -> CayleyGraph`` used for beam search.
        solve_fn: Callable used to produce a candidate solution path string.
        device: Device used when moving/compiling models.
        beam_run: Optional tracking object with ``track(...)`` method.
        config: Runtime controls for beam inference and acceptance.

    Returns:
        Aggregated :class:`BeamInferenceStats` across all attempted rows.

    Raises:
        ValueError: If ``rows``, ``submission_rows``, and ``heuristic_paths``
            do not have matching lengths.

    """
    if len(rows) != len(submission_rows):
        raise ValueError("rows and submission_rows must have the same length.")
    if len(rows) != len(heuristic_paths):
        raise ValueError("rows and heuristic_paths must have the same length.")

    cfg = config or BeamInferenceConfig()
    stats = BeamInferenceStats()
    prepared: set[int] = set()

    checkpoint_path = (
        Path(cfg.checkpoint_path) if cfg.checkpoint_path is not None else None
    )
    checkpoint_every_attempts = int(cfg.checkpoint_every_attempts)
    if checkpoint_path is not None and checkpoint_every_attempts <= 0:
        raise ValueError(
            "checkpoint_every_attempts must be >= 1 when checkpoint_path is set."
        )

    pbar = tqdm(enumerate(rows), total=len(rows), desc=f"beam n={sorted(target_n)}")
    for i, row in pbar:
        perm = parse_perm(row.permutation)
        n = len(perm)
        if n not in target_n:
            continue
        if n not in models_dict or n not in graphs_dict:
            continue

        if cfg.compile_once_per_n and n not in prepared:
            models_dict[n] = models_dict[n].to(device).eval()
            models_dict[n].compile()
            prepared.add(n)

        model = models_dict[n]
        graph = graphs_dict[n]
        if cfg.free_graph_memory:
            graph.free_memory()

        prev_solution = str(submission_rows[i]["solution"])
        heuristic_solution = str(heuristic_paths[i])
        prev_len = path_len(prev_solution)
        heur_len = path_len(heuristic_solution)

        max_steps = min(path_edges(prev_solution), cfg.max_steps_factor * n)
        stats.attempted += 1

        t0 = time.perf_counter()
        try:
            solution = solve_fn(
                perm,
                graph,
                model,
                prev_solution,
                beam_width=cfg.beam_width,
                history_depth=cfg.history_depth,
                enable_tf32=cfg.enable_tf32,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                autocast_device_type="cuda",
                max_steps=max_steps,
            )
        except torch.cuda.OutOfMemoryError:
            stats.oom_count += 1
            if beam_run is not None:
                beam_run.track(
                    1,
                    name="beam/oom",
                    step=i,
                    context={
                        "n": n,
                        "beam_width": cfg.beam_width,
                        "max_steps": max_steps,
                    },
                )
            continue

        dt = time.perf_counter() - t0
        solution = str(solution)
        sol_len = path_len(solution)

        if sol_len < prev_len:
            stats.shorter_than_previous += 1
        if sol_len < heur_len:
            stats.shorter_than_heuristic += 1
        if sol_len <= prev_len:
            stats.no_longer_than_previous += 1

        accepted = True
        if cfg.require_not_longer_than_previous and sol_len > prev_len:
            accepted = False
            stats.longer_rejected += 1

        if accepted:
            submission_rows[i]["solution"] = solution
            stats.accepted += 1

        if beam_run is not None:
            context = {
                "n": n,
                "beam_width": cfg.beam_width,
                "history_depth": cfg.history_depth,
                "max_steps": max_steps,
                "autocast_dtype": str(cfg.autocast_dtype),
                "autocast_device_type": cfg.autocast_device_type,
                "enable_tf32": cfg.enable_tf32,
                "enable_autocast": cfg.enable_autocast,
            }
            beam_run.track(dt, name="beam/solve_time_s", step=i, context=context)
            beam_run.track(prev_len, name="beam/prev_len", step=i, context={"n": n})
            beam_run.track(
                heur_len, name="beam/heuristic_len", step=i, context={"n": n}
            )
            beam_run.track(sol_len, name="beam/solution_len", step=i, context={"n": n})
            beam_run.track(
                prev_len - sol_len, name="beam/delta_vs_prev", step=i, context={"n": n}
            )
            beam_run.track(
                heur_len - sol_len,
                name="beam/delta_vs_heuristic",
                step=i,
                context={"n": n},
            )
            beam_run.track(
                1 if accepted else 0, name="beam/accepted", step=i, context={"n": n}
            )

        pbar.set_description(
            "beam "
            f"n={n} accepted={stats.accepted}/{stats.attempted} "
            f"short_vs_prev={stats.shorter_than_previous} "
            f"short_vs_heur={stats.shorter_than_heuristic} "
            f"no_longer_prev={stats.no_longer_than_previous}"
        )

        if checkpoint_path is not None and (
            stats.attempted % checkpoint_every_attempts == 0
        ):
            save_submission_rows(submission_rows, checkpoint_path)

        if cfg.clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cfg.run_gc_collect:
            gc.collect()

    return stats
