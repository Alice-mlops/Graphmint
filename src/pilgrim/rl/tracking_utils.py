# Provides reusable metric and probe helpers for RL tracker implementations.
"""Shared tracking helpers for reinforcement-learning value trainers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .policies import greedy_rollout_from_value
from .transitions import central_state_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cayleypy import CayleyGraph

_EXPECTED_STATE_NDIM = 2


def normalize_states(states: torch.Tensor) -> torch.Tensor:
    """
    Normalize states to a two-dimensional long tensor.

    Args:
        states: Input state tensor.

    Returns:
        Tensor with shape ``(batch, state_size)`` and dtype ``torch.long``.

    Raises:
        ValueError: If the tensor cannot be normalized to rank ``2``.

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


def tensor_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    """
    Compute standard summary statistics for a tensor.

    Args:
        prefix: Metric-name prefix.
        values: Tensor whose values should be summarized.

    Returns:
        Dictionary with ``mean``, ``std``, ``min``, and ``max`` metrics.

    """
    tensor = torch.as_tensor(values).detach().float().reshape(-1)
    if tensor.numel() == 0:
        return {}
    std_value = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
    return {
        f"{prefix}_mean": float(tensor.mean().item()),
        f"{prefix}_std": std_value,
        f"{prefix}_min": float(tensor.min().item()),
        f"{prefix}_max": float(tensor.max().item()),
    }


def unique_row_ratio(states: torch.Tensor) -> float:
    """
    Compute the fraction of unique rows in a batch of states.

    Args:
        states: Batched states.

    Returns:
        Number of unique rows divided by batch size.

    """
    batch = normalize_states(states).detach().cpu()
    if batch.shape[0] == 0:
        return 0.0
    unique_rows = torch.unique(batch, dim=0).shape[0]
    return float(unique_rows) / float(batch.shape[0])


def center_fraction(states: torch.Tensor, graph: CayleyGraph) -> float:
    """
    Compute the fraction of states equal to the graph center.

    Args:
        states: Batched states.
        graph: Graph providing the center state.

    Returns:
        Fraction of rows matching ``graph.central_state``.

    """
    batch = normalize_states(states)
    return float(central_state_mask(batch, graph.central_state).float().mean().item())


def parameter_statistics(model: nn.Module) -> tuple[float | None, float | None]:
    """
    Compute simple parameter-magnitude diagnostics.

    Args:
        model: Model whose parameters should be summarized.

    Returns:
        Tuple ``(global_l2_norm, max_abs_value)``. Values are ``None`` when the
        model has no parameters.

    """
    squared_sum = 0.0
    max_abs = 0.0
    has_params = False
    for param in model.parameters():
        data = param.detach().float()
        squared_sum += float(torch.sum(data * data).item())
        max_abs = max(max_abs, float(data.abs().max().item()))
        has_params = True
    if not has_params:
        return None, None
    return squared_sum**0.5, max_abs


def model_device(model: nn.Module) -> torch.device:
    """
    Infer the device used by a model.

    Args:
        model: Model whose device should be inferred.

    Returns:
        Device of the first parameter, or CPU when the model is parameterless.

    """
    param = next(model.parameters(), None)
    if param is None:
        return torch.device("cpu")
    return param.device


def predict_values(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """
    Predict scalar values for a batch of states.

    Args:
        model: Value model to evaluate.
        states: Batch of input states.

    Returns:
        One-dimensional tensor of predictions on CPU.

    """
    batch = normalize_states(states)
    device = model_device(model)
    model.eval()
    with torch.no_grad():
        values = model(batch.to(device).long()).detach().reshape(-1).float()
    return values.cpu()


def predict_scalar_value(model: nn.Module, states: torch.Tensor) -> float:
    """
    Predict one scalar value for a singleton state batch.

    Args:
        model: Value model to evaluate.
        states: Singleton batch of input states.

    Returns:
        Scalar prediction.

    """
    values = predict_values(model, states)
    return float(values[0].item())


def rollout_reaches_center(
    graph: CayleyGraph,
    start_state: torch.Tensor,
    path: Sequence[int],
) -> bool:
    """
    Return whether a rollout path reaches the center state.

    Args:
        graph: Graph used for state transitions.
        start_state: Starting state for the rollout.
        path: Sequence of generator indices applied in order.

    Returns:
        ``True`` when the final state equals the center.

    """
    state = normalize_states(start_state).to(getattr(graph, "device", "cpu"))
    if bool(central_state_mask(state, graph.central_state).item()):
        return True

    for action in path:
        next_state = torch.empty_like(state)
        graph.apply_generator_batched(int(action), state, next_state)
        state = next_state
        if bool(central_state_mask(state, graph.central_state).item()):
            return True
    return False


def resolve_primary_loss_payload(payload: Any) -> tuple[str, float]:
    """
    Resolve the trainer-specific primary backup loss from a payload object.

    Args:
        payload: Metrics or diagnostics object emitted by an RL trainer.

    Returns:
        Tuple of ``(loss_name, loss_value)``.

    Raises:
        AttributeError: If no supported primary loss is present.

    """
    if hasattr(payload, "bellman_loss"):
        return "bellman_loss", float(payload.bellman_loss)
    if hasattr(payload, "td_loss"):
        return "td_loss", float(payload.td_loss)
    raise AttributeError("payload does not expose bellman_loss or td_loss.")


def collect_prediction_metrics(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    prefix: str = "value",
) -> dict[str, float]:
    """
    Summarize predictions, targets, and residuals for one optimization step.

    Args:
        predictions: Model predictions.
        targets: Frozen-target values.
        prefix: Metric prefix namespace.

    Returns:
        Flat scalar metric dictionary.

    """
    residual = torch.as_tensor(predictions).float() - torch.as_tensor(targets).float()
    metrics: dict[str, float] = {}
    metrics.update(tensor_stats(f"{prefix}/pred", predictions))
    metrics.update(tensor_stats(f"{prefix}/target", targets))
    metrics.update(tensor_stats(f"{prefix}/residual", residual))
    metrics[f"{prefix}/residual_abs_mean"] = float(residual.abs().mean().item())
    metrics[f"{prefix}/residual_abs_max"] = float(residual.abs().max().item())
    return metrics


def collect_probe_metrics(
    *,
    model: nn.Module,
    graph: CayleyGraph,
    probe_states: torch.Tensor | None,
    probe_targets: torch.Tensor | None = None,
    rollout_max_steps: int = 128,
    max_logged_probes: int = 8,
    prefix: str = "probe",
) -> dict[str, float]:
    """
    Compute fixed-probe metrics for the current model snapshot.

    Args:
        model: Online value model to evaluate.
        graph: Graph used for greedy rollouts.
        probe_states: Optional fixed states monitored during training.
        probe_targets: Optional scalar targets for the fixed states.
        rollout_max_steps: Maximum greedy-rollout length per probe.
        max_logged_probes: Maximum number of individual probes logged.
        prefix: Metric prefix namespace.

    Returns:
        Dictionary of scalar probe metrics.

    Raises:
        ValueError: If probe targets do not match the number of probe states.

    """
    if probe_states is None:
        return {}

    states = normalize_states(probe_states)
    if probe_targets is not None:
        targets = torch.as_tensor(probe_targets).reshape(-1).float().cpu()
        if states.shape[0] != targets.shape[0]:
            raise ValueError(
                "probe_states and probe_targets must contain the same number of items."
            )
    else:
        targets = None

    metrics: dict[str, float] = {}
    probe_values = predict_values(model, states)
    metrics.update(tensor_stats(f"{prefix}/value", probe_values))

    if targets is not None:
        probe_residual = probe_values.float() - targets.float()
        metrics.update(tensor_stats(f"{prefix}/residual", probe_residual))
        metrics[f"{prefix}/residual_abs_mean"] = float(
            probe_residual.abs().mean().item()
        )

    success_values: list[float] = []
    rollout_lengths: list[float] = []
    max_logged = min(max(0, int(max_logged_probes)), int(states.shape[0]))

    for probe_idx in range(int(states.shape[0])):
        state = states[probe_idx]
        path = greedy_rollout_from_value(
            model,
            graph,
            state,
            max_steps=int(rollout_max_steps),
        )
        reached_center = rollout_reaches_center(graph, state, path)
        success_values.append(float(int(reached_center)))
        rollout_lengths.append(float(len(path)))

        if probe_idx < max_logged:
            metrics[f"{prefix}/value_{probe_idx:02d}"] = float(
                probe_values[probe_idx].item()
            )
            metrics[f"{prefix}/rollout_len_{probe_idx:02d}"] = float(len(path))
            metrics[f"{prefix}/reached_center_{probe_idx:02d}"] = float(
                int(reached_center)
            )
            if targets is not None:
                metrics[f"{prefix}/target_{probe_idx:02d}"] = float(
                    targets[probe_idx].item()
                )

    if success_values:
        metrics[f"{prefix}/success_rate"] = float(
            sum(success_values) / len(success_values)
        )
        metrics[f"{prefix}/rollout_len_mean"] = float(
            sum(rollout_lengths) / len(rollout_lengths)
        )
        metrics[f"{prefix}/rollout_len_max"] = float(max(rollout_lengths))

    return metrics
