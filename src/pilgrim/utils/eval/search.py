# Provides low-level search helpers shared by evaluation modules.
"""Search helpers for rollout and baseline evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def final_state_from_path(
    graph: object,
    start_state: Sequence[int] | torch.Tensor,
    action_path: Sequence[int],
) -> torch.Tensor:
    """
    Apply a path to a start state and return the final state tensor.

    Args:
        graph: Graph exposing ``apply_generator_batched`` and ``device``.
        start_state: Initial graph state.
        action_path: Sequence of generator indices.

    Returns:
        Final state tensor with shape ``(1, state_size)``.

    """
    device = torch.device(getattr(graph, "device", "cpu"))
    state = torch.as_tensor(start_state, device=device).long().reshape(1, -1)
    for action in action_path:
        next_state = torch.empty_like(state)
        graph.apply_generator_batched(int(action), state, next_state)
        state = next_state
    return state


def path_reaches_center(
    graph: object,
    start_state: Sequence[int] | torch.Tensor,
    action_path: Sequence[int],
) -> bool:
    """
    Determine whether a path ends at the graph center.

    Args:
        graph: Graph exposing ``central_state``.
        start_state: Initial graph state.
        action_path: Sequence of generator indices.

    Returns:
        ``True`` when the final state matches ``graph.central_state``.

    """
    final_state = final_state_from_path(graph, start_state, action_path)
    center = torch.as_tensor(graph.central_state, device=final_state.device).reshape(
        1, -1
    )
    return bool(torch.equal(final_state.long(), center.long()))


def solution_length(solution: Sequence[object] | str | None) -> int:
    """
    Convert a baseline solution payload into a path length.

    Args:
        solution: Baseline solution as a move sequence, dotted move string, or
            ``None``.

    Returns:
        Integer solution length.

    """
    if solution is None:
        return 0
    if isinstance(solution, str):
        if solution == "":
            return 0
        return int(solution.count(".") + 1)
    return int(len(solution))
