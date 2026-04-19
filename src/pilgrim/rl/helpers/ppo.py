# Provides reusable actor-critic helpers for search-guided PPO training.
"""Policy-gradient helper functions for actor-critic training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from ...utils.losses import lipschitz_expansion_loss
from ..supervision_archive import PolicySupervisionBatch
from .q_learning import _resolve_action_indices

_EXPECTED_POLICY_NDIM = 2
_NUMERICAL_ZERO_EPS = 1e-12


@dataclass(slots=True, frozen=True)
class PolicyValueOutput:
    """
    Actor-critic forward outputs.

    Args:
        logits: Policy logits with shape ``(batch, num_actions)``.
        values: Scalar value predictions with shape ``(batch,)``.

    """

    logits: torch.Tensor
    values: torch.Tensor


@dataclass(slots=True, frozen=True)
class PolicyActionEvaluation:
    """
    Policy statistics for a batch of aligned states and actions.

    Args:
        log_probs: Action log-probabilities with shape ``(batch,)``.
        entropy: Policy entropy with shape ``(batch,)``.
        values: Scalar value predictions with shape ``(batch,)``.
        logits: Full policy logits with shape ``(batch, num_actions)``.

    """

    log_probs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor
    logits: torch.Tensor


def forward_policy_value(
    model: nn.Module,
    states: torch.Tensor,
) -> PolicyValueOutput:
    """
    Run the actor-critic model and validate output shapes.

    Args:
        model: Actor-critic model exposing primary policy logits and auxiliary
            scalar values through ``forward_readouts``.
        states: Batched graph states.

    Returns:
        Policy logits and scalar values.

    Raises:
        ValueError: If the model does not expose policy and value outputs with
            the expected shapes.

    """
    batch = torch.as_tensor(states).long()
    if batch.ndim == 1:
        batch = batch.unsqueeze(0)
    if not hasattr(model, "forward_readouts"):
        raise ValueError(
            "search-guided PPO requires a model exposing forward_readouts(...)."
        )
    logits, values = model.forward_readouts(batch)
    logits = torch.as_tensor(logits).float()
    if logits.ndim != _EXPECTED_POLICY_NDIM:
        raise ValueError(
            "policy head must return shape (batch, num_actions), got "
            f"{tuple(logits.shape)}."
        )
    if values is None:
        raise ValueError("search-guided PPO requires an auxiliary scalar value head.")
    values = torch.as_tensor(values).reshape(-1).float()
    if int(values.shape[0]) != int(logits.shape[0]):
        raise ValueError("value head must align with policy logits.")
    return PolicyValueOutput(logits=logits, values=values)


def sample_policy_actions(
    model: nn.Module,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
    action_temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample actions from the policy head and return log-probabilities.

    Args:
        model: Actor-critic model.
        states: Batched graph states.
        generator_indices: Optional subset of legal generator ids.
        action_temperature: Sampling temperature applied to policy logits.
        generator: Optional RNG used for action sampling.

    Returns:
        Tuple ``(actions, log_probs, values)`` aligned with ``states``.

    Raises:
        ValueError: If ``action_temperature`` is not positive or no actions are
            allowed.

    """
    if float(action_temperature) <= 0.0:
        raise ValueError("action_temperature must be positive.")
    outputs = forward_policy_value(model, states)
    allowed = _resolve_action_indices(
        num_actions=int(outputs.logits.shape[1]),
        generator_indices=generator_indices,
        device=outputs.logits.device,
    )
    if allowed.numel() == 0:
        raise ValueError("at least one allowed action is required.")

    filtered_logits = outputs.logits.index_select(1, allowed) / float(
        action_temperature
    )
    log_probs_full = functional.log_softmax(filtered_logits, dim=1)
    probs = log_probs_full.exp()
    sampled_positions = _sample_action_positions(
        probs,
        generator=generator,
    )
    sampled_actions = allowed.index_select(0, sampled_positions)
    sampled_log_probs = torch.gather(
        log_probs_full,
        dim=1,
        index=sampled_positions.reshape(-1, 1),
    ).reshape(-1)
    return sampled_actions, sampled_log_probs, outputs.values


def greedy_policy_actions(
    model: nn.Module,
    states: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Select greedy actions from the policy head.

    Args:
        model: Actor-critic model.
        states: Batched graph states.
        generator_indices: Optional subset of legal generator ids.

    Returns:
        One-dimensional tensor of greedy generator ids.

    """
    outputs = forward_policy_value(model, states)
    allowed = _resolve_action_indices(
        num_actions=int(outputs.logits.shape[1]),
        generator_indices=generator_indices,
        device=outputs.logits.device,
    )
    best_positions = outputs.logits.index_select(1, allowed).argmax(dim=1)
    return allowed.index_select(0, best_positions)


def evaluate_policy_actions(
    model: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    generator_indices: Sequence[int] | None = None,
    action_temperature: float = 1.0,
) -> PolicyActionEvaluation:
    """
    Evaluate log-probabilities and entropy for aligned policy actions.

    Args:
        model: Actor-critic model.
        states: Batched graph states.
        actions: One-dimensional generator ids aligned with ``states``.
        generator_indices: Optional subset of legal generator ids.
        action_temperature: Sampling temperature used by the rollout policy.

    Returns:
        Policy statistics aligned with the provided actions.

    Raises:
        ValueError: If the actions fall outside the allowed generator subset.

    """
    if float(action_temperature) <= 0.0:
        raise ValueError("action_temperature must be positive.")
    outputs = forward_policy_value(model, states)
    allowed = _resolve_action_indices(
        num_actions=int(outputs.logits.shape[1]),
        generator_indices=generator_indices,
        device=outputs.logits.device,
    )
    action_ids = (
        torch.as_tensor(actions, device=outputs.logits.device).long().reshape(-1)
    )
    if int(action_ids.shape[0]) != int(outputs.logits.shape[0]):
        raise ValueError("actions must align with states.")
    action_positions = _action_positions_from_allowed(action_ids, allowed)
    filtered_logits = outputs.logits.index_select(1, allowed) / float(
        action_temperature
    )
    log_probs_full = functional.log_softmax(filtered_logits, dim=1)
    probs = log_probs_full.exp()
    log_probs = torch.gather(
        log_probs_full,
        dim=1,
        index=action_positions.reshape(-1, 1),
    ).reshape(-1)
    entropy = -(probs * log_probs_full).sum(dim=1)
    return PolicyActionEvaluation(
        log_probs=log_probs,
        entropy=entropy,
        values=outputs.values,
        logits=outputs.logits,
    )


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    Normalize advantages to zero mean and unit variance.

    Args:
        advantages: One-dimensional advantage tensor.

    Returns:
        Normalized advantages.

    """
    data = torch.as_tensor(advantages).float()
    if data.numel() <= 1:
        return data
    std = torch.std(data, unbiased=False)
    if float(std.item()) <= _NUMERICAL_ZERO_EPS:
        return data - data.mean()
    return (data - data.mean()) / (std + 1e-8)


def compute_supervised_policy_value_losses(
    model: nn.Module,
    batch: PolicySupervisionBatch,
    *,
    device: str | torch.device,
    generator_indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute policy and value losses for one supervision batch.

    Args:
        model: Actor-critic model.
        batch: Supervision batch.
        device: Device on which to run the model.
        generator_indices: Optional subset of legal generator ids.

    Returns:
        Tuple ``(policy_loss, value_loss, metrics)``.

    """
    supervision = batch.to(device)
    outputs = forward_policy_value(model, supervision.states)
    allowed = _resolve_action_indices(
        num_actions=int(outputs.logits.shape[1]),
        generator_indices=generator_indices,
        device=outputs.logits.device,
    )
    target_positions = _action_positions_from_allowed(
        supervision.action_targets.to(outputs.logits.device),
        allowed,
    )
    policy_loss_raw = functional.cross_entropy(
        outputs.logits.index_select(1, allowed),
        target_positions,
        reduction="none",
    )
    value_loss_raw = functional.smooth_l1_loss(
        outputs.values,
        supervision.value_targets.to(outputs.values.device),
        reduction="none",
    )
    weights = supervision.weights.to(outputs.values.device)
    weight_sum = torch.clamp(weights.sum(), min=1e-8)
    policy_loss = (policy_loss_raw * weights).sum() / weight_sum
    value_loss = (value_loss_raw * weights).sum() / weight_sum
    metrics = {
        "action_accuracy": float(
            (outputs.logits.index_select(1, allowed).argmax(dim=1) == target_positions)
            .float()
            .mean()
            .item()
        ),
        "value_mae": float(
            (outputs.values - supervision.value_targets.to(outputs.values.device))
            .abs()
            .mean()
            .item()
        ),
    }
    return policy_loss, value_loss, metrics


def compute_lipschitz_actor_critic_loss(
    model: nn.Module,
    graph: Any,
    states: torch.Tensor,
    *,
    max_states: int | None = None,
    generator_indices: Sequence[int] | None = None,
    max_generators: int | None = None,
    seed: int | None = None,
    state_batch_size: int | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute a Lipschitz penalty on the actor-critic value head.

    Args:
        model: Actor-critic model.
        graph: Cayley graph used for neighbor expansion.
        states: Input states.
        max_states: Optional cap on sampled states.
        generator_indices: Optional subset of generator ids.
        max_generators: Optional cap on sampled generators.
        seed: Optional deterministic seed.
        state_batch_size: Optional chunk size for the penalty helper.
        reduction: Reduction mode passed to the penalty helper.

    Returns:
        Scalar penalty on the auxiliary value head.

    """

    class _ValueHead(nn.Module):
        def __init__(self, wrapped: nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, states: torch.Tensor) -> torch.Tensor:
            return forward_policy_value(self.wrapped, states).values

    return lipschitz_expansion_loss(
        _ValueHead(model),
        graph,
        states,
        max_states=max_states,
        generator_indices=generator_indices,
        max_generators=max_generators,
        seed=seed,
        state_batch_size=state_batch_size,
        reduction=reduction,
    )


def _action_positions_from_allowed(
    actions: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    """
    Map generator ids to positions inside an allowed-action subset.

    Args:
        actions: One-dimensional tensor of generator ids.
        allowed: One-dimensional tensor listing allowed ids.

    Returns:
        Positions of each action inside ``allowed``.

    Raises:
        ValueError: If one of the actions is not allowed.

    """
    action_ids = torch.as_tensor(actions, device=allowed.device).long().reshape(-1)
    positions = torch.full_like(action_ids, fill_value=-1)
    for position, action in enumerate(allowed.tolist()):
        positions[action_ids == int(action)] = int(position)
    if bool((positions < 0).any()):
        invalid_actions = action_ids[positions < 0].detach().cpu().tolist()
        raise ValueError(
            "actions must fall inside the allowed generator subset, got "
            f"{invalid_actions!r}."
        )
    return positions


def _sample_action_positions(
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample categorical positions with optional generator-device fallback.

    Args:
        probabilities: Row-wise categorical probabilities.
        generator: Optional random generator.

    Returns:
        One-dimensional sampled position tensor aligned with the batch.

    """
    if probabilities.device.type == "cpu" or generator is None:
        return torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).reshape(-1)
    sampled = torch.multinomial(
        probabilities.detach().cpu(),
        num_samples=1,
        replacement=True,
        generator=generator,
    ).reshape(-1)
    return sampled.to(probabilities.device)
