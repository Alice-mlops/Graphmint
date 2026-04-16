# Extracts deterministic greedy policies from learned value functions.
"""Policy helpers derived from scalar value networks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from cayleypy import CayleyGraph
from torch import nn

from .transitions import central_state_mask, enumerate_neighbor_states

_EXPECTED_STATE_NDIM = 2


def greedy_actions_from_value(
    model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Choose the greedy action that minimizes next-state value.

    Args:
        model: Scalar value network.
        graph: Cayley graph whose generators define the action space.
        states: Tensor of input states.
        generator_indices: Optional subset of generators considered by the
            greedy policy.
        value_batch_size: Optional chunk size for value evaluation.

    Returns:
        One-dimensional tensor of generator indices selected greedily.

    """
    source_states = _normalize_states(states)
    neighbors, used_generators = enumerate_neighbor_states(
        graph,
        source_states,
        generator_indices=generator_indices,
    )
    neighbor_values = _evaluate_neighbor_values(
        model=model,
        neighbor_states=neighbors,
        value_batch_size=value_batch_size,
    )
    best_positions = neighbor_values.argmin(dim=1)
    return used_generators[best_positions].to(source_states.device)


def greedy_rollout_from_value(
    model: nn.Module,
    graph: CayleyGraph,
    start_state: torch.Tensor | Sequence[int],
    *,
    max_steps: int,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
) -> list[int]:
    """
    Roll out a deterministic greedy policy until the center is reached.

    Args:
        model: Scalar value network.
        graph: Cayley graph used for transitions.
        start_state: Starting graph state.
        max_steps: Maximum rollout length.
        generator_indices: Optional subset of generators considered by the
            greedy policy.
        value_batch_size: Optional chunk size for value evaluation.

    Returns:
        List of generator indices selected during the rollout.

    Raises:
        ValueError: If ``max_steps`` is negative.

    """
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative.")

    state = _normalize_states(torch.as_tensor(start_state, device=_graph_device(graph)))
    path: list[int] = []
    for _ in range(int(max_steps)):
        if bool(central_state_mask(state, graph.central_state).item()):
            break
        action = greedy_actions_from_value(
            model,
            graph,
            state,
            generator_indices=generator_indices,
            value_batch_size=value_batch_size,
        )[0]
        next_state = torch.empty_like(state)
        graph.apply_generator_batched(int(action.item()), state, next_state)
        path.append(int(action.item()))
        state = next_state
    return path


def _evaluate_neighbor_values(
    *,
    model: nn.Module,
    neighbor_states: torch.Tensor,
    value_batch_size: int | None,
) -> torch.Tensor:
    """
    Evaluate scalar values for a ``(batch, actions, state_size)`` tensor.

    Args:
        model: Scalar value network.
        neighbor_states: Neighbor states to evaluate.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        Tensor with shape ``(batch, actions)``.

    """
    model_device = _model_device(model)
    flat_states = neighbor_states.reshape(-1, neighbor_states.shape[-1]).to(
        model_device
    )
    chunk_size = (
        flat_states.shape[0] if value_batch_size is None else int(value_batch_size)
    )
    outputs: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, flat_states.shape[0], chunk_size):
            chunk = flat_states[start : start + chunk_size]
            outputs.append(model(chunk.long()).detach().reshape(-1).float())

    return (
        torch
        .cat(outputs, dim=0)
        .view(
            neighbor_states.shape[0],
            neighbor_states.shape[1],
        )
        .to(neighbor_states.device)
    )


def _normalize_states(states: torch.Tensor) -> torch.Tensor:
    """
    Normalize state tensors to ``(batch, state_size)``.

    Args:
        states: Input states.

    Returns:
        Two-dimensional long tensor of states.

    Raises:
        ValueError: If the normalized tensor does not have rank ``2``.

    """
    data = torch.as_tensor(states).long()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.ndim != _EXPECTED_STATE_NDIM:
        raise ValueError(
            "states must have shape (batch, state_size) or (state_size,), "
            f"got {tuple(data.shape)}."
        )
    return data.contiguous()


def _graph_device(graph: CayleyGraph) -> torch.device:
    """
    Return the graph device as ``torch.device``.

    Args:
        graph: Graph whose device should be normalized.

    Returns:
        Graph device.

    """
    return torch.device(getattr(graph, "device", "cpu"))


def _model_device(model: nn.Module) -> torch.device:
    """
    Infer the device used by a model.

    Args:
        model: Model whose device should be inferred.

    Returns:
        Device of the first model parameter, or CPU if absent.

    """
    param = next(model.parameters(), None)
    if param is None:
        return torch.device("cpu")
    return param.device
