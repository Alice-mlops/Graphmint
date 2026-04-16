"""Utilities for the pancake permutation group and Kaggle move formats."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from cayleypy import CayleyGraph, PermutationGroups, Predictor
from torch import nn

from .graph_utils import identity


def find_prefix_length(generator_perm: Sequence[int]) -> int:
    """
    Infer the prefix length ``k`` from a pancake generator permutation.

    For the pancake group, generators are prefix reversals. When represented as
    a permutation, a prefix reversal of length ``k`` typically begins with a
    decreasing run ``k-1, k-2, ..., 0``. This helper returns that ``k`` by
    scanning until the run breaks.

    Args:
        generator_perm: A generator permutation represented as a sequence of
            integers.

    Returns:
        The length of the initial decreasing-by-1 prefix. If the entire sequence
        is decreasing by 1, returns ``len(generator_perm)``.

    """
    n = len(generator_perm)
    for i in range(1, n):
        if generator_perm[i] != generator_perm[i - 1] - 1:
            return i
    return n


def convert_to_rk_format(internal_path: Sequence[int], graph: CayleyGraph) -> list[str]:
    """
    Convert a CayleyGraph generator-index path into Kaggle ``Rk`` moves.

    ``cayleypy`` beam search can return a path as a sequence of generator
    indices. For pancake groups, each generator index corresponds to a prefix
    reversal ``Rk`` (reverse the first ``k`` elements). This function converts
    an internal path to a list of move strings like ``["R3", "R5"]``.

    Args:
        internal_path: Sequence of generator indices (typically from
            ``CayleyGraph.beam_search(..., return_path=True)``).
        graph: Cayley graph whose generator definitions are used to infer ``k``.

    Returns:
        List of moves in Kaggle format. Out-of-range indices are ignored.

    Raises:
        AttributeError: If generator permutations cannot be found on ``graph``.

    """
    generators = getattr(getattr(graph, "definition", None), "generators", None)
    if generators is None:
        generators = getattr(graph, "generators", None)
    if generators is None:
        raise AttributeError(
            "CayleyGraph does not expose generator permutations via "
            "`.definition.generators` or `.generators`."
        )

    moves: list[str] = []
    for move_index in internal_path:
        idx = int(move_index)
        if 0 <= idx < len(generators):
            generator_perm = generators[idx]
            k = find_prefix_length(generator_perm)
            moves.append(f"R{k}")
    return moves


def solve(
    permutation: Sequence[int] | np.ndarray,
    graph: CayleyGraph,
    model: nn.Module,
    heuristic_path: str,
    *,
    beam_width: int = 10_000,
    max_steps: int | None = None,
    history_depth: int = 20,
    enable_tf32: bool | None = None,
    enable_autocast: bool | None = None,
    autocast_dtype: torch.dtype = torch.float16,
    autocast_device_type: str | None = None,
) -> str:
    """
    Try to improve a heuristic solution using beam search and a learned model.

    The function runs ``graph.beam_search`` from ``permutation`` using
    ``Predictor(graph, model)`` and returns the beam-search path iff it is
    strictly shorter than the provided ``heuristic_path``.

    Args:
        permutation: Starting permutation/state to solve.
        graph: Cayley graph to search on.
        model: Model used by ``cayleypy.Predictor``.
        heuristic_path: Baseline solution in Kaggle format, e.g. ``"R3.R5"``.
        beam_width: Beam width to use for the search.
        max_steps: Maximum number of steps to search. Defaults to ``3 * n``,
            where ``n`` is the permutation length.
        enable_tf32: Controls CUDA TF32 matmul/cudnn behavior. ``None`` means:
            enable on CUDA and disable otherwise.
        enable_autocast: Controls AMP autocast during search. ``None`` means:
            enable on CUDA and disable otherwise.
        autocast_dtype: Autocast compute dtype (e.g. ``torch.float16`` or
            ``torch.bfloat16``).
        autocast_device_type: Device type for autocast (e.g. ``"cuda"`` or
            ``"cpu"``). Defaults to the graph device type.

    Returns:
        A solution path in Kaggle format. If beam search fails or is not an
        improvement, returns ``heuristic_path`` unchanged.

    """
    start_state = np.asarray(permutation)
    n = int(start_state.shape[0])
    steps = 3 * n if max_steps is None else int(max_steps)
    model.eval()

    graph_device = graph.device
    if not isinstance(graph_device, torch.device):
        graph_device = torch.device(graph_device)
    is_cuda = graph_device.type == "cuda"

    use_tf32 = is_cuda if enable_tf32 is None else bool(enable_tf32)
    use_autocast = is_cuda if enable_autocast is None else bool(enable_autocast)
    if autocast_device_type is None:
        autocast_device_type = "cuda" if is_cuda else "cpu"

    prev_matmul_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    prev_cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)

    result: Any
    try:
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

        amp_ctx = (
            torch.autocast(
                device_type=autocast_device_type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            )
            if (is_cuda or autocast_device_type == "cpu")
            else nullcontext()
        )

        with torch.inference_mode(), amp_ctx:
            beam_kwargs: dict[str, Any] = {
                "start_state": start_state,
                "beam_width": int(beam_width),
                "max_steps": steps,
                "predictor": Predictor(graph, model),
                "return_path": True,
                "memory_cleanup": False,
                "beam_mode": "simple",
                "history_depth": history_depth,
            }
            try:
                result = graph.beam_search(**beam_kwargs)
            except TypeError as exc:
                # Backward/forward compatibility for cayleypy versions that do
                # not expose ``memory_cleanup`` in ``beam_search``.
                if "memory_cleanup" not in str(exc):
                    raise
                beam_kwargs.pop("memory_cleanup", None)
                result = graph.beam_search(**beam_kwargs)
    finally:
        if is_cuda and prev_matmul_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        if is_cuda and prev_cudnn_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    heuristic_len = 0 if not heuristic_path else (heuristic_path.count(".") + 1)
    if (
        bool(getattr(result, "path_found", False))
        and getattr(result, "path", None) is not None
    ):
        path = result.path
        if len(path) < heuristic_len:
            moves = convert_to_rk_format(path, graph)
            return ".".join(moves)

    return heuristic_path


def make_graph_for_n(
    n: int,
    *,
    batch_size: int = 2**17,
    dtype: torch.dtype = torch.int8,
    device: str | torch.device | None = None,
) -> CayleyGraph:
    """
    Construct a pancake-group ``CayleyGraph`` for a given ``n``.

    The graph is built with an inverse-closed generator set and uses the
    identity permutation as the central state.

    Args:
        n: Pancake size (permutation length).
        batch_size: Internal batch size for batched graph operations.
        dtype: Data type for graph state tensors.
        device: Optional device for the graph (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
        A configured ``cayleypy.CayleyGraph`` instance.

    """
    central = identity(n)
    group = (
        PermutationGroups.pancake(n).make_inverse_closed().with_central_state(central)
    )
    kwargs: dict[str, Any] = {"dtype": dtype, "batch_size": int(batch_size)}
    if device is not None:
        kwargs["device"] = device
    return CayleyGraph(group, **kwargs)


def pancake_sort_path(perm: Sequence[int]) -> list[str]:
    """
    Return prefix reversals that sort ``perm`` to the identity permutation.

    The returned moves are in Kaggle format: each move is a string ``"Rk"``
    meaning "reverse the first ``k`` elements".

    Args:
        perm: The permutation to sort. Assumes it contains integers
            ``0..n-1`` exactly once.

    Returns:
        A list of moves that transform ``perm`` into ``[0, 1, ..., n-1]``.

    """
    arr = list(perm)
    n = len(arr)
    moves: list[str] = []

    for target in range(n, 1, -1):
        desired_value = target - 1
        idx = arr.index(desired_value)

        if idx == target - 1:
            continue  # already in place

        if idx != 0:
            moves.append(f"R{idx + 1}")
            arr[: idx + 1] = reversed(arr[: idx + 1])

        moves.append(f"R{target}")
        arr[:target] = reversed(arr[:target])

    return moves
