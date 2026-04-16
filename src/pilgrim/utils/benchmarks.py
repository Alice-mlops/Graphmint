"""Benchmark helpers used in Pilgrim utilities."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


def small_inference_speed_benchmark(
    *,
    cfg: Mapping[str, Any],
    graph: Any,
    model: nn.Module,
    num_iters: int = 100,
) -> float:
    """
    Run a tiny forward-pass benchmark for a Pilgrim model.

    This mirrors the historical inline benchmark used by `try_beam()`.

    Args:
        cfg: Config mapping; expects `cfg["list_beam_width"]` to exist.
        graph: Object with `.central_state` and `.device` attributes.
        model: The model to run.
        num_iters: Number of repeated forward passes to average.

    Returns:
        Average time per forward pass in milliseconds.

    """
    beam_widths = cfg["list_beam_width"]
    n = len(graph.central_state)
    batch = torch.randint(0, n, (np.max(beam_widths) // 16, n), device=graph.device)
    torch.cuda.synchronize() if "cuda" in str(graph.device) else None
    tic = time.perf_counter()
    for _ in range(num_iters):
        _ = model(batch)
    torch.cuda.synchronize() if "cuda" in str(graph.device) else None
    toc = time.perf_counter()
    avg_ms = (toc - tic) / num_iters * 1000
    print(
        f"  model(batch) took {avg_ms:.2f} ms, {n} states, batch size {batch.shape[0]}"
    )
    return float(avg_ms)
