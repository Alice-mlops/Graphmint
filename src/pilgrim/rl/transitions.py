# Builds graph transitions and Bellman targets for fitted value iteration.
"""Transition helpers for deterministic shortest-path reinforcement learning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from cayleypy import CayleyGraph
from torch import nn

_EXPECTED_STATE_NDIM = 2
_EXPECTED_STEP_TARGET_NDIM = 2
_SAMPLING_SEED_MODULUS = 2**63 - 1


@dataclass(slots=True, frozen=True)
class SampledBackupConfig:
    """
    Optional Monte Carlo approximation settings for multi-step value backups.

    Args:
        enabled: Whether sampled backups are enabled.
        action_sample_size: Number of actions sampled per non-root state. ``None``
            uses all configured actions.
        root_action_sample_size: Number of actions sampled at the root states.
            ``None`` uses all configured actions at the root.
        action_sample_repeats: Number of independent sampled backup trees whose
            targets are averaged.
        horizon_sample_size: Optional number of lambda-return horizons sampled
            from the truncated lambda weighting. ``None`` uses the exact
            truncated lambda mixture over all horizons.
        seed: Base random seed for deterministic target sampling.

    """

    enabled: bool = False
    action_sample_size: int | None = None
    root_action_sample_size: int | None = None
    action_sample_repeats: int = 1
    horizon_sample_size: int | None = None
    seed: int = 42


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
    sampled_backup: SampledBackupConfig | None = None,
    sample_index: int = 0,
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
        sampled_backup: Optional sampled-backup configuration. Disabled by
            default, preserving exact full-action backups.
        sample_index: Monotonic target-evaluation call index used to vary
            deterministic sampled backups across optimizer steps.

    Returns:
        One-dimensional tensor of TD targets with length equal to the number of
        input states.

    """
    if sampled_backup is not None and bool(sampled_backup.enabled):
        if td_lambda is None:
            return compute_sampled_n_step_value_targets(
                target_model,
                graph,
                states,
                num_steps=int(n_steps),
                reward_per_step=reward_per_step,
                discount=discount,
                terminal_value=terminal_value,
                generator_indices=generator_indices,
                value_batch_size=value_batch_size,
                sampled_backup=sampled_backup,
                sample_index=int(sample_index),
            )
        return compute_sampled_td_lambda_value_targets(
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
            sampled_backup=sampled_backup,
            sample_index=int(sample_index),
        )

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


def compute_sampled_n_step_value_target_sequence(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    sampled_backup: SampledBackupConfig,
    reward_per_step: float = 1.0,
    discount: float = 1.0,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
    sample_index: int = 0,
) -> torch.Tensor:
    """
    Compute sampled discounted optimality targets for horizons ``1..num_steps``.

    This is a Monte Carlo approximation to the exact recursive min-over-actions
    backup. At each expanded non-terminal state it samples a subset of actions,
    takes the minimum over that subset, and averages several independently
    sampled trees when ``action_sample_repeats > 1``.

    Args:
        target_model: Frozen target value network used for horizon-zero values.
        graph: Cayley graph defining the action set and center state.
        states: Tensor of states whose targets should be computed.
        num_steps: Maximum TD backup horizon.
        sampled_backup: Sampling settings for action subsets and repeats.
        reward_per_step: Step cost added to non-terminal targets.
        discount: Discount factor applied after each transition.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of generators used in the target.
        value_batch_size: Optional chunk size for model evaluation.
        sample_index: Target-evaluation call index used to vary deterministic
            samples across optimizer steps.

    Returns:
        Tensor with shape ``(num_steps, batch)`` containing sampled targets for
        horizons ``1..num_steps``.

    """
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")
    _validate_sampled_backup_config(sampled_backup)

    repeats = int(sampled_backup.action_sample_repeats)
    sequences: list[torch.Tensor] = []
    for repeat_index in range(repeats):
        generator = _make_sampling_generator(
            int(sampled_backup.seed),
            sample_index=int(sample_index),
            salt=repeat_index,
        )
        sequence = _compute_sampled_value_target_sequence_with_bootstrap(
            target_model,
            graph,
            states,
            num_steps=int(num_steps),
            reward_per_step=reward_per_step,
            discount=discount,
            terminal_value=terminal_value,
            generator_indices=generator_indices,
            value_batch_size=value_batch_size,
            sampled_backup=sampled_backup,
            generator=generator,
            depth_index=0,
        )[1:]
        sequences.append(sequence)

    if len(sequences) == 1:
        return sequences[0]
    return torch.stack(sequences, dim=0).mean(dim=0)


def compute_sampled_n_step_value_targets(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    sampled_backup: SampledBackupConfig,
    reward_per_step: float = 1.0,
    discount: float = 1.0,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
    sample_index: int = 0,
) -> torch.Tensor:
    """
    Compute a sampled discounted optimality target with horizon ``num_steps``.

    Args mirror :func:`compute_sampled_n_step_value_target_sequence`.

    Returns:
        One-dimensional tensor of sampled TD targets.

    """
    return compute_sampled_n_step_value_target_sequence(
        target_model,
        graph,
        states,
        num_steps=int(num_steps),
        sampled_backup=sampled_backup,
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
        sample_index=int(sample_index),
    )[-1]


def compute_sampled_td_lambda_value_targets(
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    num_steps: int,
    td_lambda: float,
    sampled_backup: SampledBackupConfig,
    reward_per_step: float = 1.0,
    discount: float = 1.0,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    value_batch_size: int | None = None,
    sample_index: int = 0,
) -> torch.Tensor:
    """
    Compute a sampled truncated TD-lambda target.

    Action sampling approximates the recursive min-over-actions tree. When
    ``horizon_sample_size`` is set, the lambda-return mixture itself is
    approximated by drawing horizons from the truncated lambda weights and
    averaging their sampled targets.

    Args mirror :func:`compute_td_lambda_value_targets` with ``sampled_backup``
    and ``sample_index`` added.

    Returns:
        One-dimensional tensor of sampled TD-lambda targets.

    """
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive.")
    _validate_sampled_backup_config(sampled_backup)

    horizon_sample_size = sampled_backup.horizon_sample_size
    if horizon_sample_size is None:
        step_targets = compute_sampled_n_step_value_target_sequence(
            target_model,
            graph,
            states,
            num_steps=int(num_steps),
            sampled_backup=sampled_backup,
            reward_per_step=reward_per_step,
            discount=discount,
            terminal_value=terminal_value,
            generator_indices=generator_indices,
            value_batch_size=value_batch_size,
            sample_index=int(sample_index),
        )
        return combine_truncated_td_lambda_targets(
            step_targets,
            td_lambda=float(td_lambda),
        )

    horizon_positions = _sample_lambda_horizon_positions(
        num_steps=int(num_steps),
        td_lambda=float(td_lambda),
        horizon_sample_size=int(horizon_sample_size),
        seed=int(sampled_backup.seed),
        sample_index=int(sample_index),
        device=_graph_device(graph),
    )
    max_horizon = int(horizon_positions.max().item()) + 1
    step_targets = compute_sampled_n_step_value_target_sequence(
        target_model,
        graph,
        states,
        num_steps=max_horizon,
        sampled_backup=sampled_backup,
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
        sample_index=int(sample_index),
    )
    selected = step_targets.index_select(0, horizon_positions.to(step_targets.device))
    return selected.float().mean(dim=0)


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


def _compute_sampled_value_target_sequence_with_bootstrap(
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
    sampled_backup: SampledBackupConfig,
    generator: torch.Generator,
    depth_index: int,
) -> torch.Tensor:
    """
    Compute bootstrap and sampled ``1..num_steps`` targets for a batch.

    Duplicate input states are deduplicated before expanding sampled successor
    trees, matching the exact target builder's behavior.
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
    unique_sequence = _compute_sampled_unique_value_target_sequence_with_bootstrap(
        target_model,
        graph,
        unique_states,
        num_steps=int(num_steps),
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
        sampled_backup=sampled_backup,
        generator=generator,
        depth_index=int(depth_index),
    )
    return unique_sequence[:, inverse]


def _compute_sampled_unique_value_target_sequence_with_bootstrap(
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
    sampled_backup: SampledBackupConfig,
    generator: torch.Generator,
    depth_index: int,
) -> torch.Tensor:
    """
    Compute bootstrap and sampled ``1..num_steps`` targets for unique states.
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
    sample_size = _action_sample_size_for_depth(
        sampled_backup,
        depth_index=int(depth_index),
    )
    neighbors, _ = _enumerate_sampled_neighbor_states(
        graph,
        active_states,
        generator_indices=generator_indices,
        action_sample_size=sample_size,
        generator=generator,
    )
    flat_neighbors = neighbors.reshape(-1, neighbors.shape[-1])
    neighbor_targets = _compute_sampled_value_target_sequence_with_bootstrap(
        target_model,
        graph,
        flat_neighbors,
        num_steps=int(num_steps) - 1,
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        value_batch_size=value_batch_size,
        sampled_backup=sampled_backup,
        generator=generator,
        depth_index=int(depth_index) + 1,
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


def _sample_lambda_horizon_positions(
    *,
    num_steps: int,
    td_lambda: float,
    horizon_sample_size: int,
    seed: int,
    sample_index: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample zero-based horizon positions from truncated TD-lambda weights.
    """
    if int(horizon_sample_size) <= 0:
        raise ValueError("horizon_sample_size must be positive.")
    weights = _td_lambda_weights(
        num_steps=int(num_steps),
        td_lambda=float(td_lambda),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    generator = _make_sampling_generator(
        int(seed),
        sample_index=int(sample_index),
        salt=1_000_003,
    )
    positions = torch.multinomial(
        weights,
        num_samples=int(horizon_sample_size),
        replacement=True,
        generator=generator,
    )
    return positions.to(device=device, dtype=torch.long)


def _enumerate_sampled_neighbor_states(
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None,
    action_sample_size: int | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Enumerate neighbors for a sampled per-row action subset.

    Returns:
        ``(neighbors, sampled_generators)`` where ``neighbors`` has shape
        ``(batch, sampled_actions, state_size)`` and sampled generators has shape
        ``(batch, sampled_actions)``. If ``action_sample_size`` covers all
        available actions, this falls back to exact enumeration.
    """
    source_states = _normalize_states(states, device=_graph_device(graph))
    allowed = _resolve_generator_indices(graph, generator_indices)
    if allowed.numel() == 0:
        raise ValueError("at least one generator is required to enumerate neighbors.")

    if action_sample_size is None or int(action_sample_size) >= int(allowed.numel()):
        neighbors, used_generators = enumerate_neighbor_states(
            graph,
            source_states,
            generator_indices=generator_indices,
        )
        tiled_generators = used_generators.view(1, -1).expand(
            source_states.shape[0],
            -1,
        )
        return neighbors, tiled_generators
    if int(action_sample_size) <= 0:
        raise ValueError("action_sample_size must be positive when provided.")

    sampled_generators = _sample_generator_table(
        allowed_generators=allowed,
        batch_size=int(source_states.shape[0]),
        sample_size=int(action_sample_size),
        generator=generator,
    )
    neighbors = torch.empty(
        (
            source_states.shape[0],
            int(action_sample_size),
            source_states.shape[1],
        ),
        dtype=source_states.dtype,
        device=source_states.device,
    )

    for sample_position in range(int(action_sample_size)):
        column_generators = sampled_generators[:, sample_position]
        for generator_index in torch.unique(column_generators).tolist():
            rows = torch.nonzero(
                column_generators == int(generator_index),
                as_tuple=False,
            ).reshape(-1)
            dst = torch.empty(
                (rows.numel(), source_states.shape[1]),
                dtype=source_states.dtype,
                device=source_states.device,
            )
            graph.apply_generator_batched(
                int(generator_index),
                source_states.index_select(0, rows),
                dst,
            )
            neighbors[rows, sample_position, :] = dst
    return neighbors, sampled_generators


def _sample_generator_table(
    *,
    allowed_generators: torch.Tensor,
    batch_size: int,
    sample_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample ``sample_size`` distinct allowed generators for every batch row.
    """
    if int(sample_size) <= 0:
        raise ValueError("sample_size must be positive.")
    action_count = int(allowed_generators.numel())
    if int(sample_size) > action_count:
        raise ValueError("sample_size cannot exceed the number of allowed actions.")
    random_scores = torch.rand(
        (int(batch_size), action_count),
        generator=generator,
        dtype=torch.float32,
    )
    positions = torch.topk(
        random_scores,
        k=int(sample_size),
        dim=1,
        largest=False,
    ).indices.to(device=allowed_generators.device)
    return allowed_generators.index_select(0, positions.reshape(-1)).view(
        int(batch_size),
        int(sample_size),
    )


def _action_sample_size_for_depth(
    sampled_backup: SampledBackupConfig,
    *,
    depth_index: int,
) -> int | None:
    """
    Resolve root-vs-recursive action sample size for one backup depth.
    """
    if int(depth_index) == 0:
        return sampled_backup.root_action_sample_size
    return sampled_backup.action_sample_size


def _validate_sampled_backup_config(sampled_backup: SampledBackupConfig) -> None:
    """
    Validate sampled-backup settings provided to low-level target builders.
    """
    if sampled_backup.action_sample_size is not None and int(
        sampled_backup.action_sample_size
    ) <= 0:
        raise ValueError("action_sample_size must be positive when provided.")
    if sampled_backup.root_action_sample_size is not None and int(
        sampled_backup.root_action_sample_size
    ) <= 0:
        raise ValueError("root_action_sample_size must be positive when provided.")
    if int(sampled_backup.action_sample_repeats) <= 0:
        raise ValueError("action_sample_repeats must be positive.")
    if sampled_backup.horizon_sample_size is not None and int(
        sampled_backup.horizon_sample_size
    ) <= 0:
        raise ValueError("horizon_sample_size must be positive when provided.")


def _make_sampling_generator(
    seed: int,
    *,
    sample_index: int,
    salt: int,
) -> torch.Generator:
    """
    Build a CPU RNG for deterministic target-sampling calls.
    """
    combined_seed = (
        int(seed)
        + 1_000_000_007 * int(sample_index)
        + 97_409 * int(salt)
    ) % _SAMPLING_SEED_MODULUS
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(combined_seed))
    return generator


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
