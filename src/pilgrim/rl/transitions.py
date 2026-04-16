# Builds graph transitions and Bellman targets for fitted value iteration.
"""Transition helpers for deterministic shortest-path reinforcement learning."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from cayleypy import CayleyGraph
from torch import nn

_EXPECTED_STATE_NDIM = 2
_EXPECTED_STEP_TARGET_NDIM = 2


def enumerate_neighbor_states(
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Enumerate one-step neighbor states for each input state.

    Args:
        graph: Cayley graph whose generators define the transitions.
        states: Tensor with shape ``(batch, state_size)`` or ``(state_size,)``.
        generator_indices: Optional subset of generator indices.

    Returns:
        Tuple ``(neighbors, used_generators)`` where ``neighbors`` has shape
        ``(batch, num_generators, state_size)`` and ``used_generators`` is a
        one-dimensional tensor of generator indices.

    Raises:
        ValueError: If no generator indices are available.

    """
    source_states = _normalize_states(states, device=_graph_device(graph))
    used_generators = _resolve_generator_indices(graph, generator_indices)
    if used_generators.numel() == 0:
        raise ValueError("at least one generator is required to enumerate neighbors.")

    neighbors = torch.empty(
        (source_states.shape[0], used_generators.numel(), source_states.shape[1]),
        dtype=source_states.dtype,
        device=source_states.device,
    )
    for position, generator_index in enumerate(used_generators.tolist()):
        graph.apply_generator_batched(
            int(generator_index),
            source_states,
            neighbors[:, position, :],
        )
    return neighbors, used_generators


def compute_bellman_value_targets(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    reward_per_step: float = 1.0,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Compute fitted-value Bellman targets for deterministic shortest paths.

    Args:
        target_model: Frozen target value network.
        graph: Cayley graph defining the action set and center state.
        states: Tensor of states whose targets should be computed.
        reward_per_step: Step cost added to non-terminal targets.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of Bellman targets with length equal to the
        number of input states.

    """
    return compute_n_step_value_targets(
        target_model,
        graph,
        states,
        num_steps=1,
        reward_per_step=reward_per_step,
        discount=1.0,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
    )


def compute_configured_value_targets(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    reward_per_step: float,
    discount: float,
    n_steps: int,
    td_lambda: float | None,
    terminal_value: float,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Compute TD targets using the trainer-configured backup mode.

    Args:
        target_model: Frozen target value network.
        graph: Cayley graph defining transitions and the center state.
        states: Tensor of states whose targets should be computed.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied to future values.
        n_steps: Maximum TD backup horizon.
        td_lambda: Optional truncated TD-lambda coefficient.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of TD targets with length equal to the number of
        input states.

    """
    if td_lambda is None:
        return compute_n_step_value_targets(
            target_model,
            graph,
            states,
            num_steps=int(n_steps),
            reward_per_step=reward_per_step,
            discount=discount,
            terminal_value=terminal_value,
            generator_indices=generator_indices,
            value_batch_size=value_batch_size,
        )
    return compute_td_lambda_value_targets(
        target_model,
        graph,
        states,
        num_steps=int(n_steps),
        td_lambda=float(td_lambda),
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
    )


def compute_n_step_value_target_sequence(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    reward_per_step: float = 1.0,
    discount: float = 1.0,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Compute exact discounted optimality targets for horizons ``1..num_steps``.

    Args:
        target_model: Frozen target value network used for horizon-zero values.
        graph: Cayley graph defining the action set and center state.
        states: Tensor of states whose targets should be computed.
        num_steps: Maximum TD backup horizon.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied after each transition.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        Tensor with shape ``(num_steps, batch)`` containing ``y^(1)`` through
        ``y^(num_steps)`` for each input state.

    Raises:
        ValueError: If ``num_steps`` is not positive.

    """
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")

    sequence = _compute_value_target_sequence_with_bootstrap(
        target_model,
        graph,
        states,
        num_steps=int(num_steps),
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
    )
    return sequence[1:]


def compute_n_step_value_targets(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    reward_per_step: float = 1.0,
    discount: float = 1.0,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Compute an exact discounted optimality target with horizon ``num_steps``.

    Args:
        target_model: Frozen target value network used for horizon-zero values.
        graph: Cayley graph defining the action set and center state.
        states: Tensor of states whose targets should be computed.
        num_steps: TD backup horizon. ``1`` recovers the classical Bellman
            target when ``discount == 1``.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied after each transition.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of TD targets with length equal to the number of
        input states.

    Raises:
        ValueError: If ``num_steps`` is not positive.

    """
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")

    return compute_n_step_value_target_sequence(
        target_model,
        graph,
        states,
        num_steps=num_steps,
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
    )[-1]


def combine_truncated_td_lambda_targets(
    step_targets: torch.Tensor,
    *,
    td_lambda: float,
) -> torch.Tensor:
    """
    Combine ``n``-step targets into a truncated TD-lambda target.

    Args:
        step_targets: Tensor with shape ``(num_steps, batch)`` containing
            ``y^(1)`` through ``y^(num_steps)``.
        td_lambda: TD-lambda coefficient in ``[0, 1]``.

    Returns:
        One-dimensional tensor of lambda targets with length ``batch``.

    Raises:
        ValueError: If the inputs are malformed.

    """
    if step_targets.ndim != _EXPECTED_STEP_TARGET_NDIM:
        raise ValueError(
            "step_targets must have shape (num_steps, batch), "
            f"got {tuple(step_targets.shape)}."
        )
    if step_targets.shape[0] == 0:
        raise ValueError("step_targets must include at least one horizon.")
    if not 0.0 <= float(td_lambda) <= 1.0:
        raise ValueError("td_lambda must be in the closed interval [0, 1].")

    weights = _td_lambda_weights(
        num_steps=int(step_targets.shape[0]),
        td_lambda=float(td_lambda),
        device=step_targets.device,
        dtype=step_targets.float().dtype,
    )
    return torch.sum(weights.unsqueeze(1) * step_targets.float(), dim=0)


def compute_td_lambda_value_targets(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    td_lambda: float,
    reward_per_step: float = 1.0,
    discount: float = 1.0,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Compute a truncated TD-lambda target built from exact ``1..n`` backups.

    Args:
        target_model: Frozen target value network used for horizon-zero values.
        graph: Cayley graph defining the action set and center state.
        states: Tensor of states whose targets should be computed.
        num_steps: Maximum TD backup horizon.
        td_lambda: TD-lambda coefficient in ``[0, 1]``.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied after each transition.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of TD-lambda targets.

    """
    step_targets = compute_n_step_value_target_sequence(
        target_model,
        graph,
        states,
        num_steps=num_steps,
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
    )
    return combine_truncated_td_lambda_targets(step_targets, td_lambda=td_lambda)


def central_state_mask(
    states: torch.Tensor,
    central_state: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """
    Return a mask indicating which states match the center.

    Args:
        states: Tensor of states with shape ``(batch, state_size)``.
        central_state: Graph center represented as a tensor or sequence.

    Returns:
        Boolean tensor with shape ``(batch,)``.

    """
    source_states = _normalize_states(states)
    center = torch.as_tensor(
        central_state,
        device=source_states.device,
        dtype=source_states.dtype,
    ).view(1, -1)
    return torch.eq(source_states, center).all(dim=1)


def _evaluate_neighbor_values(
    *,
    target_model: nn.Module,
    neighbor_states: torch.Tensor,
    value_batch_size: int | None,
) -> torch.Tensor:
    """
    Evaluate target values for a neighbor tensor.

    Args:
        target_model: Frozen target value network.
        neighbor_states: Tensor with shape ``(batch, num_actions, state_size)``.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        Tensor with shape ``(batch, num_actions)`` containing predicted values.

    """
    values = _evaluate_state_values(
        target_model=target_model,
        states=neighbor_states.reshape(-1, neighbor_states.shape[-1]),
        value_batch_size=value_batch_size,
    )
    return values.view(neighbor_states.shape[0], neighbor_states.shape[1]).to(
        neighbor_states.device
    )


def _compute_value_target_sequence_with_bootstrap(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    reward_per_step: float,
    discount: float,
    terminal_value: float,
    generator_indices: Sequence[int] | None,
    value_batch_size: int | None,
) -> torch.Tensor:
    """
    Compute bootstrap and exact ``1..num_steps`` targets for a batch of states.

    Args:
        target_model: Frozen target value network used for horizon-zero values.
        graph: Cayley graph defining the action set and center state.
        states: Tensor of states whose targets should be computed.
        num_steps: Maximum TD backup horizon.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied after each transition.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        Tensor with shape ``(num_steps + 1, batch)`` containing the bootstrap
        values followed by exact ``1..num_steps`` targets.

    Raises:
        ValueError: If ``num_steps`` is negative.

    """
    if int(num_steps) < 0:
        raise ValueError("num_steps must be non-negative.")

    source_states = _normalize_states(states, device=_graph_device(graph))
    if source_states.shape[0] == 0:
        return torch.empty(
            (int(num_steps) + 1, 0),
            dtype=torch.float32,
            device=source_states.device,
        )

    unique_states, inverse = torch.unique(
        source_states,
        dim=0,
        return_inverse=True,
    )
    unique_sequence = _compute_unique_value_target_sequence_with_bootstrap(
        target_model,
        graph,
        unique_states,
        num_steps=int(num_steps),
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
    )
    return unique_sequence[:, inverse]


def _compute_unique_value_target_sequence_with_bootstrap(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    reward_per_step: float,
    discount: float,
    terminal_value: float,
    generator_indices: Sequence[int] | None,
    value_batch_size: int | None,
) -> torch.Tensor:
    """
    Compute bootstrap and exact ``1..num_steps`` targets for unique states.

    Args:
        target_model: Frozen target value network used for horizon-zero values.
        graph: Cayley graph defining the action set and center state.
        states: Deduplicated tensor of states.
        num_steps: Maximum TD backup horizon.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied after each transition.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        Tensor with shape ``(num_steps + 1, batch)`` containing the bootstrap
        values followed by exact ``1..num_steps`` targets.

    """
    terminal_mask = central_state_mask(states, graph.central_state)
    targets = torch.full(
        (int(num_steps) + 1, states.shape[0]),
        fill_value=float(terminal_value),
        dtype=torch.float32,
        device=states.device,
    )
    bootstrap_values = _evaluate_state_values(
        target_model=target_model,
        states=states,
        value_batch_size=value_batch_size,
    )
    bootstrap_values[terminal_mask] = float(terminal_value)
    targets[0] = bootstrap_values

    if int(num_steps) == 0 or bool(terminal_mask.all()):
        return targets

    active_states = states[~terminal_mask]
    neighbors, _ = enumerate_neighbor_states(
        graph,
        active_states,
        generator_indices=generator_indices,
    )
    flat_neighbors = neighbors.reshape(-1, neighbors.shape[-1])
    neighbor_targets = _compute_value_target_sequence_with_bootstrap(
        target_model,
        graph,
        flat_neighbors,
        num_steps=int(num_steps) - 1,
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
    )
    num_actions = neighbors.shape[1]

    for step_index in range(1, int(num_steps) + 1):
        next_values = neighbor_targets[step_index - 1].view(
            active_states.shape[0],
            num_actions,
        )
        targets[step_index, ~terminal_mask] = (
            float(reward_per_step) + float(discount) * next_values.min(dim=1).values
        )

    return targets


def _evaluate_state_values(
    *,
    target_model: nn.Module,
    states: torch.Tensor,
    value_batch_size: int | None,
) -> torch.Tensor:
    """
    Evaluate scalar values for a ``(batch, state_size)`` state tensor.

    Args:
        target_model: Frozen target value network.
        states: Tensor with shape ``(batch, state_size)``.
        value_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of predicted values with length ``batch``.

    """
    model_device = _model_device(target_model)
    flat_states = _normalize_states(states).to(model_device)
    chunk_size = (
        flat_states.shape[0] if value_batch_size is None else int(value_batch_size)
    )
    outputs: list[torch.Tensor] = []

    target_model.eval()
    with torch.no_grad():
        for start in range(0, flat_states.shape[0], chunk_size):
            chunk = flat_states[start : start + chunk_size]
            outputs.append(target_model(chunk.long()).detach().reshape(-1).float())

    return torch.cat(outputs, dim=0).to(states.device)


def _td_lambda_weights(
    *,
    num_steps: int,
    td_lambda: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build truncated TD-lambda weights for ``1..num_steps`` targets.

    Args:
        num_steps: Number of available step targets.
        td_lambda: TD-lambda coefficient in ``[0, 1]``.
        device: Device used for the returned tensor.
        dtype: Floating-point dtype used for the returned tensor.

    Returns:
        One-dimensional tensor of truncated TD-lambda weights.

    Raises:
        ValueError: If ``num_steps`` is not positive.

    """
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")

    if int(num_steps) == 1:
        return torch.ones(1, device=device, dtype=dtype)

    lambda_tensor = torch.tensor(float(td_lambda), device=device, dtype=dtype)
    prefix_powers = torch.arange(int(num_steps) - 1, device=device, dtype=dtype)
    weights = torch.empty(int(num_steps), device=device, dtype=dtype)
    weights[:-1] = (1.0 - lambda_tensor) * torch.pow(lambda_tensor, prefix_powers)
    weights[-1] = torch.pow(lambda_tensor, int(num_steps) - 1)
    return weights


def _resolve_generator_indices(
    graph: CayleyGraph,
    generator_indices: Sequence[int] | None,
) -> torch.Tensor:
    """
    Resolve the generator subset used for one-step transitions.

    Args:
        graph: Cayley graph exposing the available generators.
        generator_indices: Optional explicit generator subset.

    Returns:
        One-dimensional tensor of generator indices on the graph device.

    """
    if generator_indices is None:
        total_generators = len(graph.generators)
        return torch.arange(
            total_generators, device=_graph_device(graph), dtype=torch.long
        )
    return torch.as_tensor(
        list(generator_indices),
        device=_graph_device(graph),
        dtype=torch.long,
    )


def _normalize_states(
    states: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Normalize input states to a two-dimensional long tensor.

    Args:
        states: Input tensor of states.
        device: Optional target device for the normalized tensor.

    Returns:
        Tensor with shape ``(batch, state_size)`` and dtype ``torch.long``.

    Raises:
        ValueError: If the normalized tensor does not have rank ``2``.

    """
    data = torch.as_tensor(states, device=device).long()
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
    Return the graph device as a ``torch.device`` instance.

    Args:
        graph: Graph exposing a ``device`` attribute.

    Returns:
        Graph device converted to ``torch.device``.

    """
    return torch.device(getattr(graph, "device", "cpu"))


def _model_device(model: nn.Module) -> torch.device:
    """
    Return the device of the first model parameter.

    Args:
        model: Model whose device should be inferred.

    Returns:
        Device of the first parameter, or CPU when the model has no parameters.

    """
    param = next(model.parameters(), None)
    if param is None:
        return torch.device("cpu")
    return param.device
