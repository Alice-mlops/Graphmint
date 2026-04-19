# Shared helper functions for vector-valued Q-learning on Cayley graphs.
"""Q-learning helper functions used by trainers, trackers, and notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from cayleypy import CayleyGraph
from torch import nn

from ...schemas.rl import TDRandomWalkSamplingConfig
from ..config import RandomWalkSamplingConfig
from ..replay import TransitionBatch, concatenate_transition_batches
from ..sampling import resolve_rw_schedule
from ..transitions import central_state_mask

_EXPECTED_STATE_NDIM = 2
_EXPECTED_Q_NDIM = 2


def _generator_device(
    generator: torch.Generator | None,
    fallback: torch.device,
) -> torch.device:
    """
    Resolve the device on which RNG draws should be created.

    Args:
        generator: Optional PyTorch RNG generator.
        fallback: Device that should be used when no generator is provided.

    Returns:
        Device compatible with the provided generator or the fallback device.

    """
    if generator is None:
        return fallback
    generator_device = getattr(generator, "device", None)
    if generator_device is None:
        return fallback
    return torch.device(generator_device)


def _randint_on_compatible_device(
    *,
    low: int,
    high: int,
    size: tuple[int, ...],
    output_device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Draw integer samples with a generator-compatible device and move if needed.

    Args:
        low: Inclusive lower bound.
        high: Exclusive upper bound.
        size: Output tensor shape.
        output_device: Device on which the returned tensor should live.
        generator: Optional PyTorch RNG generator.

    Returns:
        Tensor of sampled integers on ``output_device``.

    """
    draw_device = _generator_device(generator, output_device)
    values = torch.randint(
        low=low,
        high=high,
        size=size,
        device=draw_device,
        generator=generator,
    )
    return values.to(output_device)


def _rand_on_compatible_device(
    *,
    size: tuple[int, ...],
    output_device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Draw uniform samples with a generator-compatible device and move if needed.

    Args:
        size: Output tensor shape.
        output_device: Device on which the returned tensor should live.
        generator: Optional PyTorch RNG generator.

    Returns:
        Tensor of uniform random values on ``output_device``.

    """
    draw_device = _generator_device(generator, output_device)
    values = torch.rand(
        size,
        device=draw_device,
        generator=generator,
    )
    return values.to(output_device)


def model_output_dim(model: nn.Module) -> int:
    """
    Infer the output width produced by ``model``.

    Args:
        model: Model whose output width should be inferred.

    Returns:
        Positive output width. Scalar value models return ``1``.

    """
    output_dim = getattr(model, "output_dim", None)
    if output_dim is None and hasattr(model, "module"):
        output_dim = getattr(model.module, "output_dim", None)
    if output_dim is not None:
        return int(output_dim)
    return 1


def normalize_states(states: torch.Tensor) -> torch.Tensor:
    """
    Normalize state tensors to shape ``(batch, state_size)``.

    Args:
        states: Input state tensor.

    Returns:
        Two-dimensional ``torch.long`` tensor.

    Raises:
        ValueError: If ``states`` cannot be normalized to rank ``2``.

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


def evaluate_q_values(
    model: nn.Module,
    states: torch.Tensor,
    *,
    q_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Evaluate vector Q-values for a batch of states.

    Args:
        model: Action-value model.
        states: Batched input states.
        q_batch_size: Optional chunk size for model evaluation.

    Returns:
        Tensor with shape ``(batch, num_actions)``.

    Raises:
        ValueError: If ``model`` does not expose vector outputs.

    """
    batch = normalize_states(states)
    device = _model_device(model)
    chunk_size = batch.shape[0] if q_batch_size is None else int(q_batch_size)
    outputs: list[torch.Tensor] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, batch.shape[0], chunk_size):
                chunk = batch[start : start + chunk_size].to(device).long()
                q_values = torch.as_tensor(model(chunk)).detach().float()
                if q_values.ndim != _EXPECTED_Q_NDIM:
                    raise ValueError(
                        "Q-value model must return shape (batch, num_actions), "
                        f"got {tuple(q_values.shape)}."
                    )
                outputs.append(q_values)
    finally:
        model.train(was_training)
    return torch.cat(outputs, dim=0).to(batch.device)


def evaluate_selected_q_values(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    q_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Evaluate ``Q(s, a)`` for aligned batches of states and actions.

    Args:
        model: Action-value model.
        states: Batched input states.
        actions: One-dimensional tensor of generator indices.
        q_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of selected Q-values.

    Raises:
        ValueError: If ``actions`` does not align with ``states``.

    """
    q_values = evaluate_q_values(model, states, q_batch_size=q_batch_size)
    action_ids = torch.as_tensor(actions, device=q_values.device).long().reshape(-1, 1)
    if action_ids.shape[0] != q_values.shape[0]:
        raise ValueError(
            "actions must align with states, got "
            f"{action_ids.shape[0]} actions for {q_values.shape[0]} states."
        )
    return (
        torch
        .gather(q_values, dim=1, index=action_ids)
        .reshape(-1)
        .to(normalize_states(states).device)
    )


def state_values_from_q(
    model: nn.Module,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Convert a vector-Q model into a scalar state-value estimate.

    Args:
        model: Action-value model.
        states: Batched input states.
        generator_indices: Optional subset of allowed generator ids.
        q_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of ``min_a Q(s, a)`` values.

    """
    q_values = evaluate_q_values(model, states, q_batch_size=q_batch_size)
    allowed = _resolve_action_indices(
        num_actions=int(q_values.shape[1]),
        generator_indices=generator_indices,
        device=q_values.device,
    )
    return (
        q_values
        .index_select(1, allowed)
        .min(dim=1)
        .values.to(normalize_states(states).device)
    )


def greedy_actions_from_q(
    model: nn.Module,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Choose the greedy action that minimizes predicted Q-value.

    Args:
        model: Action-value model.
        states: Batched input states.
        generator_indices: Optional subset of allowed generator ids.
        q_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of selected generator indices.

    """
    q_values = evaluate_q_values(model, states, q_batch_size=q_batch_size)
    allowed = _resolve_action_indices(
        num_actions=int(q_values.shape[1]),
        generator_indices=generator_indices,
        device=q_values.device,
    )
    best_positions = q_values.index_select(1, allowed).argmin(dim=1)
    return allowed[best_positions].to(normalize_states(states).device)


def sample_behavior_actions(
    model: nn.Module,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
    mode: str = "uniform",
    epsilon: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample behavior actions for off-policy DDQN targets.

    Args:
        model: Action-value model used for greedy proposals.
        states: Batched input states.
        generator_indices: Optional subset of allowed generator ids.
        q_batch_size: Optional chunk size for model evaluation.
        mode: ``"uniform"`` or ``"epsilon_greedy"``.
        epsilon: Exploration probability used by epsilon-greedy sampling.
        generator: Optional RNG used for uniform draws.

    Returns:
        One-dimensional tensor of sampled generator indices.

    Raises:
        ValueError: If ``mode`` is unsupported.

    """
    batch = normalize_states(states)
    num_actions = int(model_output_dim(model))
    allowed = _resolve_action_indices(
        num_actions=num_actions,
        generator_indices=generator_indices,
        device=batch.device,
    )
    if allowed.numel() == 0:
        raise ValueError("at least one allowed action is required.")

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "uniform":
        positions = _randint_on_compatible_device(
            low=0,
            high=int(allowed.numel()),
            size=(batch.shape[0],),
            output_device=batch.device,
            generator=generator,
        )
        return allowed[positions]

    if normalized_mode != "epsilon_greedy":
        raise ValueError(
            f'behavior mode must be "uniform" or "epsilon_greedy", got {mode!r}.'
        )

    greedy_actions = greedy_actions_from_q(
        model,
        batch,
        generator_indices=generator_indices,
        q_batch_size=q_batch_size,
    ).to(batch.device)
    if float(epsilon) <= 0.0:
        return greedy_actions

    random_positions = _randint_on_compatible_device(
        low=0,
        high=int(allowed.numel()),
        size=(batch.shape[0],),
        output_device=batch.device,
        generator=generator,
    )
    random_actions = allowed[random_positions]
    explore_mask = _rand_on_compatible_device(
        size=(batch.shape[0],),
        output_device=batch.device,
        generator=generator,
    ) < float(epsilon)
    return torch.where(explore_mask, random_actions, greedy_actions)


def apply_actions(
    graph: CayleyGraph,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """
    Apply one generator per input state.

    Args:
        graph: Cayley graph used for transitions.
        states: Batched input states.
        actions: One-dimensional tensor of generator indices.

    Returns:
        Batched next states on the graph device.

    Raises:
        ValueError: If ``actions`` does not align with ``states``.

    """
    source_states = normalize_states(states).to(_graph_device(graph))
    action_ids = (
        torch.as_tensor(actions, device=source_states.device).long().reshape(-1)
    )
    if action_ids.shape[0] != source_states.shape[0]:
        raise ValueError(
            "actions must align with states, got "
            f"{action_ids.shape[0]} actions for {source_states.shape[0]} states."
        )
    next_states = torch.empty_like(source_states)
    for action in torch.unique(action_ids).tolist():
        mask = action_ids == int(action)
        if not bool(mask.any()):
            continue
        moved_states = torch.empty(
            (int(mask.sum().item()), source_states.shape[1]),
            device=source_states.device,
            dtype=source_states.dtype,
        )
        graph.apply_generator_batched(
            int(action),
            source_states[mask],
            moved_states,
        )
        next_states[mask] = moved_states
    return next_states


def sample_n_step_transitions_from_states(
    *,
    online_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    n_steps: int,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
    behavior_mode: str = "uniform",
    behavior_epsilon: float = 1.0,
    action_generator: torch.Generator | None = None,
) -> TransitionBatch:
    """
    Sample one aligned n-step transition row per source state.

    Args:
        online_model: Online Q model used for behavior sampling.
        graph: Cayley graph used for transitions.
        states: Batched starting states.
        n_steps: Maximum sampled transition horizon.
        generator_indices: Optional subset of allowed generator ids.
        q_batch_size: Optional chunk size for Q-value evaluation.
        behavior_mode: ``"uniform"`` or ``"epsilon_greedy"``.
        behavior_epsilon: Exploration probability for epsilon-greedy sampling.
        action_generator: Optional RNG used for action sampling.

    Returns:
        Transition batch aligned with the input states.

    Raises:
        ValueError: If ``n_steps`` is not positive.

    """
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive.")

    normalized_states = normalize_states(states)
    output_device = normalized_states.device
    start_states = normalized_states.to(_graph_device(graph))
    batch_size = int(start_states.shape[0])
    actions = sample_behavior_actions(
        online_model,
        start_states,
        generator_indices=generator_indices,
        q_batch_size=q_batch_size,
        mode=behavior_mode,
        epsilon=behavior_epsilon,
        generator=action_generator,
    ).to(start_states.device)
    next_states = start_states.clone()
    steps = torch.zeros(batch_size, dtype=torch.long, device=start_states.device)
    done = central_state_mask(start_states, graph.central_state)
    active_mask = ~done

    for step_index in range(int(n_steps)):
        if not bool(active_mask.any()):
            break
        if step_index == 0:
            step_actions = actions[active_mask]
        else:
            step_actions = sample_behavior_actions(
                online_model,
                next_states[active_mask],
                generator_indices=generator_indices,
                q_batch_size=q_batch_size,
                mode=behavior_mode,
                epsilon=behavior_epsilon,
                generator=action_generator,
            ).to(start_states.device)
        advanced_states = apply_actions(
            graph,
            next_states[active_mask],
            step_actions,
        )
        next_states[active_mask] = advanced_states
        steps[active_mask] += 1
        reached_terminal = central_state_mask(advanced_states, graph.central_state)
        if bool(reached_terminal.any()):
            active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
            done_indices = active_indices[reached_terminal]
            done[done_indices] = True
            active_mask[done_indices] = False

    return TransitionBatch(
        states=start_states.to(output_device),
        actions=actions.to(output_device),
        next_states=next_states.to(output_device),
        steps=steps.to(output_device),
        done=done.to(output_device),
    )


def compute_double_q_targets_from_transition_batch(
    *,
    online_model: nn.Module,
    target_model: nn.Module,
    transitions: TransitionBatch,
    reward_per_step: float,
    discount: float,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Compute Double-DQN targets for a precomputed transition batch.

    Args:
        online_model: Online Q model used for bootstrap action selection.
        target_model: Frozen target Q model used for bootstrap values.
        transitions: Aligned transition batch.
        reward_per_step: Step cost added to each transition.
        discount: Discount factor applied after each sampled step.
        terminal_value: Value assigned to terminal states.
        generator_indices: Optional subset of allowed generator ids.
        q_batch_size: Optional chunk size for Q-value evaluation.

    Returns:
        One-dimensional tensor of DDQN targets aligned with ``transitions``.

    """
    batch = transitions.to(_model_device(target_model))
    steps = batch.steps.float()
    if torch.isclose(
        torch.tensor(float(discount), device=batch.states.device),
        torch.tensor(1.0, device=batch.states.device),
    ).item():
        returns = steps * float(reward_per_step)
    else:
        discount_tensor = torch.full_like(steps, float(discount))
        returns = (
            float(reward_per_step)
            * (1.0 - torch.pow(discount_tensor, steps))
            / (1.0 - float(discount))
        )

    bootstrap_weights = torch.pow(
        torch.full_like(steps, float(discount)),
        steps,
    )
    targets = returns
    if bool(batch.done.any()):
        done_indices = batch.done
        targets = targets.clone()
        targets[done_indices] += bootstrap_weights[done_indices] * float(terminal_value)

    active_mask = ~batch.done
    if bool(active_mask.any()):
        bootstrap_actions = greedy_actions_from_q(
            online_model,
            batch.next_states[active_mask],
            generator_indices=generator_indices,
            q_batch_size=q_batch_size,
        ).to(batch.states.device)
        bootstrap_values = evaluate_selected_q_values(
            target_model,
            batch.next_states[active_mask],
            bootstrap_actions,
            q_batch_size=q_batch_size,
        ).to(batch.states.device)
        targets = targets.clone()
        targets[active_mask] += (
            bootstrap_weights[active_mask] * bootstrap_values.float()
        )
    return targets.to(transitions.states.device)


def sample_n_step_transitions_from_random_walks(
    graph: CayleyGraph,
    config: RandomWalkSamplingConfig | TDRandomWalkSamplingConfig,
    *,
    n_steps: int,
    generator_indices: Sequence[int] | None = None,
    sample_index: int = 0,
) -> TransitionBatch:
    """
    Build replay transitions from explicit random-action walks.

    Args:
        graph: Cayley graph used to generate transitions.
        config: Random-walk sampling configuration.
        n_steps: Maximum collapsed transition horizon.
        generator_indices: Optional subset of allowed generator ids.
        sample_index: Sampling-call index used to derive a deterministic seed.

    Returns:
        Concatenated transition batch produced by the configured walk schedule.

    Raises:
        ValueError: If ``n_steps`` is not positive.
        RuntimeError: If no valid non-terminal transitions are generated.

    """
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive.")

    graph_device = _graph_device(graph)
    seed = int(config.seed) + int(sample_index) * 100_003
    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(seed)
    allowed = _resolve_graph_action_indices(
        graph,
        generator_indices=generator_indices,
        device=graph_device,
    )
    inverse_map = _resolve_graph_inverse_map(graph, device=graph_device)
    center_state = torch.as_tensor(graph.central_state, device=graph_device).long()

    batches: list[TransitionBatch] = []
    for factor, length in resolve_rw_schedule(config):
        walk_length = int(length)
        if walk_length <= 0:
            continue
        width = max(1, int(int(config.rw_width) * float(factor)))
        current_states = center_state.view(1, -1).expand(width, -1).clone()
        states_by_step = [current_states.clone()]
        actions_by_step: list[torch.Tensor] = []
        previous_actions: torch.Tensor | None = None
        for _ in range(walk_length):
            step_actions = _sample_random_walk_actions(
                allowed_actions=allowed,
                batch_size=width,
                previous_actions=previous_actions,
                inverse_map=inverse_map,
                rw_mode=str(config.rw_mode),
                generator=action_generator,
                output_device=graph_device,
            )
            current_states = apply_actions(graph, current_states, step_actions)
            actions_by_step.append(step_actions)
            states_by_step.append(current_states.clone())
            previous_actions = step_actions
        batches.append(
            _build_transition_batch_from_walk_history(
                states_by_step=states_by_step,
                actions_by_step=actions_by_step,
                n_steps=int(n_steps),
                central_state=center_state,
            )
        )

    non_empty_batches = [batch for batch in batches if len(batch) > 0]
    if not non_empty_batches:
        raise RuntimeError(
            "random-walk transition sampling produced no non-terminal transitions; "
            "increase sampling.rw_length or use a non-terminal transition source."
        )
    return concatenate_transition_batches(non_empty_batches)


def compute_n_step_double_q_targets(
    *,
    online_model: nn.Module,
    target_model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    reward_per_step: float,
    discount: float,
    n_steps: int,
    terminal_value: float = 0.0,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
    behavior_mode: str = "uniform",
    behavior_epsilon: float = 1.0,
    action_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build sampled n-step Double-DQN targets for one batch of states.

    Args:
        online_model: Online Q model used for behavior sampling and greedy
            bootstrap action selection.
        target_model: Frozen target Q model used for bootstrap values.
        graph: Cayley graph used for transitions.
        states: Batched starting states.
        reward_per_step: Step cost added for each sampled transition.
        discount: Discount factor applied after each sampled step.
        n_steps: Maximum sampled backup horizon.
        terminal_value: Bootstrap value assigned to terminal states.
        generator_indices: Optional subset of allowed generator ids.
        q_batch_size: Optional chunk size for Q-value evaluation.
        behavior_mode: ``"uniform"`` or ``"epsilon_greedy"``.
        behavior_epsilon: Exploration probability for epsilon-greedy sampling.
        action_generator: Optional RNG used for action sampling.

    Returns:
        Tuple ``(actions, targets)`` aligned with the input ``states``.

    Raises:
        ValueError: If ``n_steps`` is not positive.

    """
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive.")
    transitions = sample_n_step_transitions_from_states(
        online_model=online_model,
        graph=graph,
        states=states,
        n_steps=n_steps,
        generator_indices=generator_indices,
        q_batch_size=q_batch_size,
        behavior_mode=behavior_mode,
        behavior_epsilon=behavior_epsilon,
        action_generator=action_generator,
    )
    targets = compute_double_q_targets_from_transition_batch(
        online_model=online_model,
        target_model=target_model,
        transitions=transitions,
        reward_per_step=reward_per_step,
        discount=discount,
        terminal_value=terminal_value,
        generator_indices=generator_indices,
        q_batch_size=q_batch_size,
    )
    return transitions.actions, targets


def greedy_rollout_from_q(
    model: nn.Module,
    graph: CayleyGraph,
    start_state: torch.Tensor | Sequence[int],
    *,
    max_steps: int,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
) -> list[int]:
    """
    Roll out a greedy policy derived from vector Q-values.

    Args:
        model: Action-value model.
        graph: Cayley graph used for transitions.
        start_state: Starting graph state.
        max_steps: Maximum rollout length.
        generator_indices: Optional subset of allowed generator ids.
        q_batch_size: Optional chunk size for Q-value evaluation.

    Returns:
        List of chosen generator indices.

    Raises:
        ValueError: If ``max_steps`` is negative.

    """
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative.")

    state = normalize_states(torch.as_tensor(start_state, device=_graph_device(graph)))
    path: list[int] = []
    for _ in range(int(max_steps)):
        if bool(central_state_mask(state, graph.central_state).item()):
            break
        action = greedy_actions_from_q(
            model,
            state,
            generator_indices=generator_indices,
            q_batch_size=q_batch_size,
        )[0]
        state = apply_actions(graph, state, action.view(1))
        path.append(int(action.item()))
    return path


def predict_state_scores(
    model: nn.Module,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
    q_batch_size: int | None = None,
) -> torch.Tensor:
    """
    Predict scalar state scores for scalar-V and vector-Q models.

    Args:
        model: Value or action-value model.
        states: Batched input states.
        generator_indices: Optional subset of allowed actions for Q models.
        q_batch_size: Optional chunk size for model evaluation.

    Returns:
        One-dimensional tensor of scalar state scores on CPU.

    """
    batch = normalize_states(states)
    if int(model_output_dim(model)) == 1:
        device = _model_device(model)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                values = (
                    torch.as_tensor(model(batch.to(device).long())).detach().float()
                )
        finally:
            model.train(was_training)
        return values.reshape(-1).cpu()
    return state_values_from_q(
        model,
        batch,
        generator_indices=generator_indices,
        q_batch_size=q_batch_size,
    ).cpu()


def _resolve_action_indices(
    *,
    num_actions: int,
    generator_indices: Sequence[int] | None,
    device: torch.device,
) -> torch.Tensor:
    """
    Resolve allowed action ids for vector-Q models.

    Args:
        num_actions: Full action count exposed by the model.
        generator_indices: Optional subset of allowed action ids.
        device: Device used for the returned tensor.

    Returns:
        One-dimensional tensor of allowed action ids.

    Raises:
        ValueError: If an action id falls outside the model output range.

    """
    if generator_indices is None:
        return torch.arange(int(num_actions), device=device, dtype=torch.long)
    allowed = torch.as_tensor(list(generator_indices), device=device).long().reshape(-1)
    if allowed.numel() == 0:
        return allowed
    if int(allowed.min().item()) < 0 or int(allowed.max().item()) >= int(num_actions):
        raise ValueError(
            "generator_indices must fall inside [0, num_actions), got "
            f"{allowed.tolist()} for num_actions={int(num_actions)}."
        )
    return allowed


def _resolve_graph_action_indices(
    graph: CayleyGraph,
    *,
    generator_indices: Sequence[int] | None,
    device: torch.device,
) -> torch.Tensor:
    """
    Resolve allowed graph-generator ids on a target device.

    Args:
        graph: Cayley graph exposing its generator count.
        generator_indices: Optional subset of allowed action ids.
        device: Device used for the returned tensor.

    Returns:
        One-dimensional tensor of allowed graph action ids.

    """
    num_actions = _graph_num_actions(graph)
    return _resolve_action_indices(
        num_actions=num_actions,
        generator_indices=generator_indices,
        device=device,
    )


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


def _graph_device(graph: CayleyGraph) -> torch.device:
    """
    Normalize a graph device to ``torch.device``.

    Args:
        graph: Graph exposing a ``device`` attribute.

    Returns:
        Graph device or CPU when unspecified.

    """
    return torch.device(getattr(graph, "device", "cpu"))


def _graph_num_actions(graph: CayleyGraph) -> int:
    """
    Infer the number of generators exposed by a Cayley graph.

    Args:
        graph: Graph exposing generators or a graph definition.

    Returns:
        Positive number of available generators.

    Raises:
        ValueError: If the generator count cannot be inferred.

    """
    definition = getattr(graph, "definition", None)
    if definition is not None and hasattr(definition, "n_generators"):
        return int(definition.n_generators)
    generators = getattr(graph, "generators", None)
    if generators is not None:
        return len(generators)
    raise ValueError("could not infer graph generator count.")


def _resolve_graph_inverse_map(
    graph: CayleyGraph,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Resolve the graph's inverse-generator map when available.

    Args:
        graph: Graph exposing an inverse map on its definition.
        device: Device used for the returned tensor.

    Returns:
        One-dimensional inverse-map tensor, or ``None`` when unavailable.

    """
    definition = getattr(graph, "definition", None)
    inverse_map = None
    if definition is not None:
        inverse_map = getattr(definition, "generators_inverse_map", None)
    if inverse_map is None:
        inverse_map = getattr(graph, "generators_inverse_map", None)
    if inverse_map is None:
        return None
    data = torch.as_tensor(inverse_map, device=device).long().reshape(-1)
    return data


def _sample_random_walk_actions(
    *,
    allowed_actions: torch.Tensor,
    batch_size: int,
    previous_actions: torch.Tensor | None,
    inverse_map: torch.Tensor | None,
    rw_mode: str,
    generator: torch.Generator | None,
    output_device: torch.device,
) -> torch.Tensor:
    """
    Sample one random-walk action per active trajectory.

    Args:
        allowed_actions: Tensor of allowed generator ids.
        batch_size: Number of trajectories whose actions should be sampled.
        previous_actions: Previous-step action ids for non-backtracking filters.
        inverse_map: Optional inverse-generator map.
        rw_mode: Random-walk mode string.
        generator: Optional RNG used for action sampling.
        output_device: Device on which the returned actions should live.

    Returns:
        One-dimensional tensor of sampled action ids.

    """
    if previous_actions is None or not _uses_inverse_blocking(rw_mode):
        positions = _randint_on_compatible_device(
            low=0,
            high=int(allowed_actions.numel()),
            size=(int(batch_size),),
            output_device=output_device,
            generator=generator,
        )
        return allowed_actions[positions]

    if inverse_map is None:
        positions = _randint_on_compatible_device(
            low=0,
            high=int(allowed_actions.numel()),
            size=(previous_actions.shape[0],),
            output_device=output_device,
            generator=generator,
        )
        return allowed_actions[positions]

    if int(allowed_actions.numel()) <= 1:
        return allowed_actions[:1].expand(previous_actions.shape[0])

    actions = allowed_actions[
        _randint_on_compatible_device(
            low=0,
            high=int(allowed_actions.numel()),
            size=(previous_actions.shape[0],),
            output_device=output_device,
            generator=generator,
        )
    ]
    forbidden = inverse_map.index_select(0, previous_actions.long())
    valid_inverse = forbidden >= 0
    conflict_mask = valid_inverse & (actions == forbidden)
    while bool(conflict_mask.any()):
        replacement_positions = _randint_on_compatible_device(
            low=0,
            high=int(allowed_actions.numel()),
            size=(int(conflict_mask.sum().item()),),
            output_device=output_device,
            generator=generator,
        )
        actions[conflict_mask] = allowed_actions[replacement_positions]
        conflict_mask = valid_inverse & (actions == forbidden)
    return actions


def _build_transition_batch_from_walk_history(  # noqa: PLR0914
    *,
    states_by_step: list[torch.Tensor],
    actions_by_step: list[torch.Tensor],
    n_steps: int,
    central_state: torch.Tensor,
) -> TransitionBatch:
    """
    Collapse explicit walk histories into aligned n-step transition rows.

    Args:
        states_by_step: Walk states for times ``0..L``.
        actions_by_step: Walk actions for times ``0..L-1``.
        n_steps: Maximum collapsed transition horizon.
        central_state: Terminal center state on the walk device.

    Returns:
        Transition batch containing one row per non-terminal walk prefix.

    """
    walk_length = len(actions_by_step)
    if walk_length <= 0:
        empty_states = torch.empty(
            (0, states_by_step[0].shape[-1]),
            device=states_by_step[0].device,
            dtype=states_by_step[0].dtype,
        )
        empty_vector = torch.empty(0, device=states_by_step[0].device, dtype=torch.long)
        empty_done = torch.empty(0, device=states_by_step[0].device, dtype=torch.bool)
        return TransitionBatch(
            states=empty_states,
            actions=empty_vector,
            next_states=empty_states.clone(),
            steps=empty_vector.clone(),
            done=empty_done,
        )

    source_states: list[torch.Tensor] = []
    source_actions: list[torch.Tensor] = []
    source_next_states: list[torch.Tensor] = []
    source_steps: list[torch.Tensor] = []
    source_done: list[torch.Tensor] = []

    for start_index in range(walk_length):
        current_states = states_by_step[start_index]
        non_terminal_mask = ~central_state_mask(current_states, central_state)
        if not bool(non_terminal_mask.any()):
            continue
        horizon = min(int(n_steps), walk_length - start_index)
        final_states = states_by_step[start_index + horizon].clone()
        final_steps = torch.full(
            (current_states.shape[0],),
            fill_value=horizon,
            dtype=torch.long,
            device=current_states.device,
        )
        final_done = torch.zeros(
            current_states.shape[0],
            dtype=torch.bool,
            device=current_states.device,
        )
        for offset in range(horizon):
            candidate_states = states_by_step[start_index + offset + 1]
            reached = ~final_done & central_state_mask(candidate_states, central_state)
            if bool(reached.any()):
                final_states[reached] = candidate_states[reached]
                final_steps[reached] = int(offset) + 1
                final_done[reached] = True
        source_states.append(current_states[non_terminal_mask])
        source_actions.append(actions_by_step[start_index][non_terminal_mask])
        source_next_states.append(final_states[non_terminal_mask])
        source_steps.append(final_steps[non_terminal_mask])
        source_done.append(final_done[non_terminal_mask])

    if not source_states:
        empty_states = torch.empty(
            (0, states_by_step[0].shape[-1]),
            device=states_by_step[0].device,
            dtype=states_by_step[0].dtype,
        )
        empty_vector = torch.empty(0, device=states_by_step[0].device, dtype=torch.long)
        empty_done = torch.empty(0, device=states_by_step[0].device, dtype=torch.bool)
        return TransitionBatch(
            states=empty_states,
            actions=empty_vector,
            next_states=empty_states.clone(),
            steps=empty_vector.clone(),
            done=empty_done,
        )

    return TransitionBatch(
        states=torch.cat(source_states, dim=0),
        actions=torch.cat(source_actions, dim=0),
        next_states=torch.cat(source_next_states, dim=0),
        steps=torch.cat(source_steps, dim=0),
        done=torch.cat(source_done, dim=0),
    )


def _uses_inverse_blocking(rw_mode: str) -> bool:
    """
    Return whether a walk mode should avoid immediate inverse actions.

    Args:
        rw_mode: Random-walk mode string.

    Returns:
        ``True`` for inverse-blocking walk modes.

    """
    normalized = str(rw_mode).strip().lower()
    return normalized in {"nbt", "nonbacktracking", "non_backtracking", "inverse"}
