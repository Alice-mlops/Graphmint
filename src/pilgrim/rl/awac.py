# Implements generic discrete AWAC losses for graph actor-critic models.
"""Advantage-weighted actor-critic loss helpers for discrete graph actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from ..schemas.rl import AWACConfig
from .helpers.ppo import forward_policy_value

_EXPECTED_POLICY_NDIM = 2
_EXPECTED_VECTOR_NDIM = 1


@dataclass(slots=True, frozen=True)
class AWACTargetBatch:
    """
    Aligned AWAC targets for discrete actor-critic updates.

    Args:
        optimal_action_mask: Boolean mask of actions that should receive actor
            probability mass. Rows may contain multiple true entries when
            several actions are equally good.
        advantages: Optional per-row or per-action advantages. Larger values
            make the row receive a larger AWAC weight. When per-action
            advantages are supplied with `optimal_action_mask`, the largest
            optimal-action advantage is used as the row weight.
        action_mask: Optional boolean mask of legal action columns.
        sampled_actions: Optional one-dimensional action targets used when
            `optimal_action_mask` is not provided.
        value_targets: Optional scalar value targets aligned with rows.
        weights: Optional per-row sample weights.
        reference_log_probs: Optional frozen-policy log probabilities for
            ``KL(reference || current)`` anchoring.
        bad_action_mask: Optional boolean mask of known strictly bad actions.
    """

    optimal_action_mask: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    sampled_actions: torch.Tensor | None = None
    value_targets: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    reference_log_probs: torch.Tensor | None = None
    bad_action_mask: torch.Tensor | None = None


@dataclass(slots=True, frozen=True)
class AWACLossState:
    """
    Tensor-valued AWAC loss state and diagnostics.

    Args:
        total_loss: Weighted sum of all configured loss terms.
        policy_loss: Advantage-weighted actor loss.
        value_loss: Scalar value regression loss.
        entropy: Mean policy entropy over legal actions.
        policy_anchor_loss: KL anchor loss against reference log-probabilities.
        bad_action_loss: Probability-mass penalty for known bad actions.
        bad_action_margin_loss: Pairwise margin loss against bad actions.
        mean_advantage: Mean selected advantage before exponentiation.
        mean_awac_weight: Mean AWAC row weight after clipping and normalization.
        std_awac_weight: Standard deviation of AWAC row weights.
        max_awac_weight: Maximum AWAC row weight.
        optimal_action_probability: Mean probability mass on optimal actions.
        bad_action_probability: Mean probability mass on known bad actions.
        top1_optimal_accuracy: Fraction of greedy actions inside the optimal
            action set.
    """

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    policy_anchor_loss: torch.Tensor
    bad_action_loss: torch.Tensor
    bad_action_margin_loss: torch.Tensor
    mean_advantage: torch.Tensor
    mean_awac_weight: torch.Tensor
    std_awac_weight: torch.Tensor
    max_awac_weight: torch.Tensor
    optimal_action_probability: torch.Tensor
    bad_action_probability: torch.Tensor
    top1_optimal_accuracy: torch.Tensor

    def to_metrics(self) -> dict[str, float]:
        """
        Convert scalar tensors to Python floats.

        Returns:
            Dictionary suitable for trackers and CSV logging.
        """
        return {
            "loss": float(self.total_loss.detach().cpu().item()),
            "policy_loss": float(self.policy_loss.detach().cpu().item()),
            "value_loss": float(self.value_loss.detach().cpu().item()),
            "entropy": float(self.entropy.detach().cpu().item()),
            "policy_anchor_loss": float(self.policy_anchor_loss.detach().cpu().item()),
            "bad_action_loss": float(self.bad_action_loss.detach().cpu().item()),
            "bad_action_margin_loss": float(
                self.bad_action_margin_loss.detach().cpu().item()
            ),
            "mean_advantage": float(self.mean_advantage.detach().cpu().item()),
            "mean_awac_weight": float(self.mean_awac_weight.detach().cpu().item()),
            "std_awac_weight": float(self.std_awac_weight.detach().cpu().item()),
            "max_awac_weight": float(self.max_awac_weight.detach().cpu().item()),
            "optimal_action_probability": float(
                self.optimal_action_probability.detach().cpu().item()
            ),
            "bad_action_probability": float(
                self.bad_action_probability.detach().cpu().item()
            ),
            "top1_optimal_accuracy": float(
                self.top1_optimal_accuracy.detach().cpu().item()
            ),
        }


def compute_awac_actor_critic_loss(
    model: nn.Module,
    states: torch.Tensor,
    targets: AWACTargetBatch,
    *,
    config: AWACConfig | None = None,
) -> AWACLossState:
    """
    Compute AWAC loss from a model exposing ``forward_readouts``.

    Args:
        model: Actor-critic model returning policy logits and scalar values.
        states: Batched graph states.
        targets: AWAC target tensors aligned with states.
        config: Optional AWAC configuration.

    Returns:
        Tensor-valued AWAC loss state.
    """
    outputs = forward_policy_value(model, states)
    return compute_awac_loss(
        logits=outputs.logits,
        values=outputs.values,
        targets=targets,
        config=config,
    )


def compute_awac_loss(  # noqa: PLR0914
    *,
    logits: torch.Tensor,
    targets: AWACTargetBatch,
    values: torch.Tensor | None = None,
    config: AWACConfig | None = None,
) -> AWACLossState:
    """
    Compute a generic discrete AWAC actor-critic loss.

    Args:
        logits: Policy logits with shape ``(batch, num_actions)``.
        targets: Target masks, advantages, optional value targets, and weights.
        values: Optional scalar value predictions with shape ``(batch,)``.
        config: Optional loss configuration.

    Returns:
        Tensor-valued loss state.

    Raises:
        ValueError: If target tensors do not align with logits.
    """
    cfg = config or AWACConfig()
    policy_logits = _normalize_logits(logits)
    action_mask = _normalize_action_mask(targets.action_mask, policy_logits)
    optimal_mask = _normalize_optional_mask(
        targets.optimal_action_mask,
        policy_logits,
        name="optimal_action_mask",
    )
    if optimal_mask is None and targets.sampled_actions is None:
        raise ValueError("AWAC requires optimal_action_mask or sampled_actions.")
    if optimal_mask is not None:
        optimal_mask &= action_mask
        if bool((optimal_mask.sum(dim=1) <= 0).any()):
            raise ValueError("each row needs at least one optimal action.")

    masked_logits = _masked_logits(policy_logits, action_mask)
    log_probs = functional.log_softmax(masked_logits, dim=1)
    probs = log_probs.exp().masked_fill(~action_mask, 0.0)
    sample_weights = _normalize_sample_weights(targets.weights, policy_logits)
    row_advantages = _select_row_advantages(
        targets.advantages,
        logits=policy_logits,
        optimal_action_mask=optimal_mask,
        sampled_actions=targets.sampled_actions,
    )
    awac_weights = _advantage_weights(row_advantages, cfg)
    effective_weights = sample_weights * awac_weights
    policy_vec, optimal_probability, top1_optimal_accuracy = _actor_loss_vector(
        log_probs=log_probs,
        probs=probs,
        logits=policy_logits,
        action_mask=action_mask,
        optimal_action_mask=optimal_mask,
        sampled_actions=targets.sampled_actions,
        set_loss_mode=str(cfg.set_loss_mode),
    )
    policy_loss = _weighted_mean(policy_vec, effective_weights)
    entropy = _weighted_mean(_entropy(log_probs, probs, action_mask), sample_weights)
    value_loss = _value_loss(values, targets.value_targets, sample_weights)
    policy_anchor_loss = _policy_anchor_loss(
        reference_log_probs=targets.reference_log_probs,
        current_log_probs=log_probs,
        action_mask=action_mask,
        weights=sample_weights,
    )
    bad_mask = _normalize_optional_mask(
        targets.bad_action_mask,
        policy_logits,
        name="bad_action_mask",
    )
    bad_action_loss, bad_action_margin_loss, bad_action_probability = (
        _bad_action_losses(
            logits=policy_logits,
            probs=probs,
            action_mask=action_mask,
            optimal_action_mask=optimal_mask,
            bad_action_mask=bad_mask,
            weights=sample_weights,
            margin=float(cfg.bad_action_margin),
            eps=float(cfg.eps),
        )
    )
    total_loss = (
        policy_loss
        + float(cfg.value_coef) * value_loss
        - float(cfg.entropy_coef) * entropy
        + float(cfg.policy_anchor_coef) * policy_anchor_loss
        + float(cfg.bad_action_coef) * bad_action_loss
        + float(cfg.bad_action_margin_coef) * bad_action_margin_loss
    )
    return AWACLossState(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        policy_anchor_loss=policy_anchor_loss,
        bad_action_loss=bad_action_loss,
        bad_action_margin_loss=bad_action_margin_loss,
        mean_advantage=row_advantages.mean(),
        mean_awac_weight=awac_weights.mean(),
        std_awac_weight=awac_weights.std(unbiased=False),
        max_awac_weight=awac_weights.max(),
        optimal_action_probability=optimal_probability,
        bad_action_probability=bad_action_probability,
        top1_optimal_accuracy=top1_optimal_accuracy,
    )


def awac_metrics_to_float(loss_state: AWACLossState) -> dict[str, float]:
    """
    Convert an AWAC loss state to detached float metrics.

    Args:
        loss_state: Tensor-valued loss state.

    Returns:
        Flat metric dictionary.
    """
    return loss_state.to_metrics()


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize policy logits.

    Args:
        logits: Policy logits.

    Returns:
        Two-dimensional float tensor.

    Raises:
        ValueError: If logits are not two-dimensional.
    """
    data = torch.as_tensor(logits).float()
    if data.ndim != _EXPECTED_POLICY_NDIM:
        raise ValueError("logits must have shape (batch, num_actions).")
    return data


def _normalize_action_mask(
    action_mask: torch.Tensor | None,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize legal-action mask.

    Args:
        action_mask: Optional action mask.
        logits: Policy logits used for shape and device.

    Returns:
        Boolean mask aligned with logits.

    """
    if action_mask is None:
        return torch.ones_like(logits, dtype=torch.bool)
    return _normalize_mask(action_mask, logits, name="action_mask")


def _normalize_optional_mask(
    mask: torch.Tensor | None,
    logits: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor | None:
    """
    Normalize an optional action mask.

    Args:
        mask: Optional mask.
        logits: Policy logits used for shape and device.
        name: Field name used in validation errors.

    Returns:
        Boolean mask or ``None``.
    """
    if mask is None:
        return None
    return _normalize_mask(mask, logits, name=name)


def _normalize_mask(
    mask: torch.Tensor,
    logits: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """
    Normalize one action mask.

    Args:
        mask: Tensor-like mask.
        logits: Policy logits used for shape and device.
        name: Field name used in validation errors.

    Returns:
        Boolean mask aligned with logits.

    Raises:
        ValueError: If the mask shape is invalid.
    """
    data = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
    if tuple(data.shape) != tuple(logits.shape):
        raise ValueError(f"{name} must align with logits.")
    return data


def _masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a legal-action mask to logits.

    Args:
        logits: Policy logits.
        action_mask: Boolean legal-action mask.

    Returns:
        Masked logits with invalid actions filled by dtype minimum.

    Raises:
        ValueError: If any row has no legal actions.
    """
    if bool((action_mask.sum(dim=1) <= 0).any()):
        raise ValueError("each row needs at least one legal action.")
    invalid_fill = torch.finfo(logits.dtype).min
    return logits.masked_fill(~action_mask, invalid_fill)


def _normalize_sample_weights(
    weights: torch.Tensor | None,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize per-row sample weights.

    Args:
        weights: Optional row weights.
        logits: Policy logits used for shape and device.

    Returns:
        One-dimensional row weights.

    Raises:
        ValueError: If weights do not align with rows.
    """
    if weights is None:
        return torch.ones(int(logits.shape[0]), device=logits.device)
    data = torch.as_tensor(weights, device=logits.device, dtype=torch.float32).reshape(
        -1
    )
    if data.ndim != _EXPECTED_VECTOR_NDIM or int(data.shape[0]) != int(logits.shape[0]):
        raise ValueError("weights must align with logits rows.")
    return data


def _select_row_advantages(
    advantages: torch.Tensor | None,
    *,
    logits: torch.Tensor,
    optimal_action_mask: torch.Tensor | None,
    sampled_actions: torch.Tensor | None,
) -> torch.Tensor:
    """
    Select one advantage per training row.

    Args:
        advantages: Optional row or action advantages.
        logits: Policy logits used for shape and device.
        optimal_action_mask: Optional optimal-action mask.
        sampled_actions: Optional sampled action ids.

    Returns:
        One-dimensional selected advantages.

    Raises:
        ValueError: If advantages have an unsupported shape.
    """
    if advantages is None:
        return torch.zeros(int(logits.shape[0]), device=logits.device)
    data = torch.as_tensor(advantages, device=logits.device, dtype=torch.float32)
    if data.ndim == _EXPECTED_VECTOR_NDIM:
        values = data.reshape(-1)
        if int(values.shape[0]) != int(logits.shape[0]):
            raise ValueError("row advantages must align with logits rows.")
        return values
    if data.ndim != _EXPECTED_POLICY_NDIM or tuple(data.shape) != tuple(logits.shape):
        raise ValueError("advantages must have shape (batch,) or (batch, num_actions).")
    if optimal_action_mask is not None:
        selected = data.masked_fill(~optimal_action_mask, torch.finfo(data.dtype).min)
        return selected.max(dim=1).values
    if sampled_actions is None:
        raise ValueError("per-action advantages require targets.")
    action_ids = torch.as_tensor(
        sampled_actions,
        device=logits.device,
        dtype=torch.long,
    ).reshape(-1, 1)
    if int(action_ids.shape[0]) != int(logits.shape[0]):
        raise ValueError("sampled_actions must align with logits rows.")
    return torch.gather(data, dim=1, index=action_ids).reshape(-1)


def _advantage_weights(
    advantages: torch.Tensor,
    config: AWACConfig,
) -> torch.Tensor:
    """
    Convert advantages to AWAC row weights.

    Args:
        advantages: One-dimensional advantages.
        config: AWAC configuration.

    Returns:
        One-dimensional nonnegative row weights.
    """
    clipped = advantages.float()
    if config.min_advantage is not None or config.max_advantage is not None:
        min_value = (
            torch.finfo(clipped.dtype).min
            if config.min_advantage is None
            else float(config.min_advantage)
        )
        max_value = (
            torch.finfo(clipped.dtype).max
            if config.max_advantage is None
            else float(config.max_advantage)
        )
        clipped = torch.clamp(clipped, min=min_value, max=max_value)
    weights = torch.exp(clipped / float(config.advantage_temperature))
    weights = torch.clamp(weights, max=float(config.max_weight))
    if bool(config.normalize_weights) and int(weights.numel()) > 0:
        weights /= torch.clamp(weights.mean().detach(), min=float(config.eps))
    return weights


def _actor_loss_vector(
    *,
    log_probs: torch.Tensor,
    probs: torch.Tensor,
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    optimal_action_mask: torch.Tensor | None,
    sampled_actions: torch.Tensor | None,
    set_loss_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-row actor loss and policy diagnostics.

    Args:
        log_probs: Masked legal-action log-probabilities.
        probs: Masked legal-action probabilities.
        logits: Raw policy logits.
        action_mask: Legal-action mask.
        optimal_action_mask: Optional optimal-action mask.
        sampled_actions: Optional one-action labels.
        set_loss_mode: Multi-action target loss mode.

    Returns:
        Per-row actor loss, optimal-set probability, and greedy-set accuracy.

    Raises:
        ValueError: If sampled actions are invalid.
    """
    if optimal_action_mask is not None:
        if str(set_loss_mode) == "set_probability":
            target_log_probs = log_probs.masked_fill(
                ~optimal_action_mask,
                torch.finfo(log_probs.dtype).min,
            )
            policy_vec = -torch.logsumexp(target_log_probs, dim=1)
        elif str(set_loss_mode) == "uniform_targets":
            target_weights = optimal_action_mask.float()
            target_weights /= torch.clamp(
                target_weights.sum(dim=1, keepdim=True),
                min=1.0,
            )
            policy_vec = -(target_weights * log_probs).sum(dim=1)
        else:
            raise ValueError(f"Unknown AWAC set_loss_mode: {set_loss_mode!r}.")
        optimal_probability = (
            probs.masked_fill(~optimal_action_mask, 0.0).sum(dim=1).mean()
        )
        greedy_actions = logits.masked_fill(
            ~action_mask,
            torch.finfo(logits.dtype).min,
        ).argmax(dim=1)
        top1_optimal_accuracy = (
            optimal_action_mask
            .gather(1, greedy_actions.reshape(-1, 1))
            .reshape(-1)
            .float()
            .mean()
        )
        return policy_vec, optimal_probability, top1_optimal_accuracy

    action_ids = torch.as_tensor(
        sampled_actions,
        device=log_probs.device,
        dtype=torch.long,
    ).reshape(-1, 1)
    if int(action_ids.shape[0]) != int(log_probs.shape[0]):
        raise ValueError("sampled_actions must align with logits rows.")
    if bool((action_ids < 0).any()) or bool((action_ids >= log_probs.shape[1]).any()):
        raise ValueError("sampled_actions contain invalid action ids.")
    if bool((~action_mask.gather(1, action_ids)).any()):
        raise ValueError("sampled_actions must be legal under action_mask.")
    selected_log_probs = torch.gather(log_probs, dim=1, index=action_ids).reshape(-1)
    selected_probs = torch.gather(probs, dim=1, index=action_ids).reshape(-1)
    greedy_actions = logits.masked_fill(
        ~action_mask,
        torch.finfo(logits.dtype).min,
    ).argmax(dim=1)
    top1_accuracy = greedy_actions.eq(action_ids.reshape(-1)).float().mean()
    return -selected_log_probs, selected_probs.mean(), top1_accuracy


def _entropy(
    log_probs: torch.Tensor,
    probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-row policy entropy.

    Args:
        log_probs: Legal-action log probabilities.
        probs: Legal-action probabilities.
        action_mask: Legal-action mask.

    Returns:
        One-dimensional entropy tensor.
    """
    return -(probs * log_probs.masked_fill(~action_mask, 0.0)).sum(dim=1)


def _value_loss(
    values: torch.Tensor | None,
    targets: torch.Tensor | None,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute scalar value regression loss.

    Args:
        values: Optional model value predictions.
        targets: Optional scalar targets.
        weights: Per-row sample weights.

    Returns:
        Weighted SmoothL1 value loss, or zero when no targets are provided.

    Raises:
        ValueError: If targets are provided without model values.
    """
    if targets is None:
        return torch.zeros((), device=weights.device, dtype=torch.float32)
    if values is None:
        raise ValueError("values are required when value_targets are provided.")
    predictions = torch.as_tensor(values, device=weights.device).float().reshape(-1)
    target_values = torch.as_tensor(targets, device=weights.device).float().reshape(-1)
    if int(predictions.shape[0]) != int(weights.shape[0]) or int(
        target_values.shape[0]
    ) != int(weights.shape[0]):
        raise ValueError("values and value_targets must align with logits rows.")
    raw = functional.smooth_l1_loss(predictions, target_values, reduction="none")
    return _weighted_mean(raw, weights)


def _policy_anchor_loss(
    *,
    reference_log_probs: torch.Tensor | None,
    current_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked ``KL(reference || current)``.

    Args:
        reference_log_probs: Optional reference-policy log probabilities.
        current_log_probs: Current log probabilities.
        action_mask: Legal-action mask.
        weights: Per-row sample weights.

    Returns:
        Weighted anchor KL loss.

    Raises:
        ValueError: If reference log-probabilities do not align with logits.
    """
    if reference_log_probs is None:
        return torch.zeros((), device=current_log_probs.device, dtype=torch.float32)
    reference = torch.as_tensor(
        reference_log_probs,
        device=current_log_probs.device,
        dtype=torch.float32,
    )
    if tuple(reference.shape) != tuple(current_log_probs.shape):
        raise ValueError("reference_log_probs must align with logits.")
    reference_probs = reference.exp().masked_fill(~action_mask, 0.0)
    reference_probs /= torch.clamp(
        reference_probs.sum(dim=1, keepdim=True),
        min=1.0e-12,
    )
    safe_reference = reference.masked_fill(~action_mask, 0.0)
    safe_current = current_log_probs.masked_fill(~action_mask, 0.0)
    kl = (reference_probs * (safe_reference - safe_current)).sum(dim=1)
    return _weighted_mean(kl, weights)


def _bad_action_losses(
    *,
    logits: torch.Tensor,
    probs: torch.Tensor,
    action_mask: torch.Tensor,
    optimal_action_mask: torch.Tensor | None,
    bad_action_mask: torch.Tensor | None,
    weights: torch.Tensor,
    margin: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute optional penalties for known strictly bad actions.

    Args:
        logits: Raw policy logits.
        probs: Legal-action probabilities.
        action_mask: Legal-action mask.
        optimal_action_mask: Optional optimal-action mask.
        bad_action_mask: Optional bad-action mask.
        weights: Per-row sample weights.
        margin: Pairwise margin.
        eps: Probability clamp epsilon.

    Returns:
        Probability-mass loss, pairwise margin loss, and mean bad probability.
    """
    zero = torch.zeros((), device=logits.device, dtype=torch.float32)
    if bad_action_mask is None:
        return zero, zero, zero
    bad_mask = bad_action_mask & action_mask
    if optimal_action_mask is not None:
        bad_mask &= ~optimal_action_mask
    has_bad = bad_mask.sum(dim=1) > 0
    bad_probability = probs.masked_fill(~bad_mask, 0.0).sum(dim=1)
    bad_mass_vec = -torch.log(torch.clamp(1.0 - bad_probability, min=float(eps)))
    bad_mass_loss = _weighted_mean(bad_mass_vec, weights)
    if optimal_action_mask is None or not bool(has_bad.any()):
        return bad_mass_loss, zero, bad_probability.mean()

    target_logits = (
        logits
        .masked_fill(
            ~optimal_action_mask,
            torch.finfo(logits.dtype).min,
        )
        .max(dim=1)
        .values
    )
    pairwise = functional.softplus(logits - target_logits.reshape(-1, 1) + margin)
    pairwise = pairwise.masked_fill(~bad_mask, 0.0)
    row_counts = torch.clamp(bad_mask.sum(dim=1).float(), min=1.0)
    margin_vec = pairwise.sum(dim=1) / row_counts
    margin_vec = margin_vec.masked_fill(~has_bad, 0.0)
    margin_loss = _weighted_mean(margin_vec, weights)
    return bad_mass_loss, margin_loss, bad_probability.mean()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute a weighted mean with safe denominator.

    Args:
        values: Per-row values.
        weights: Nonnegative per-row weights.

    Returns:
        Scalar weighted mean.
    """
    row_values = torch.as_tensor(values, device=weights.device).float().reshape(-1)
    row_weights = torch.as_tensor(weights, device=weights.device).float().reshape(-1)
    return (row_values * row_weights).sum() / torch.clamp(
        row_weights.sum(),
        min=1.0e-8,
    )


def merge_awac_metrics(
    metrics: Mapping[str, float],
    *,
    prefix: str = "awac",
) -> dict[str, float]:
    """
    Prefix AWAC metrics for mixed trainer logs.

    Args:
        metrics: Raw metric dictionary.
        prefix: Prefix inserted before every metric name.

    Returns:
        New dictionary with prefixed keys.
    """
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


__all__ = [
    "AWACLossState",
    "AWACTargetBatch",
    "awac_metrics_to_float",
    "compute_awac_actor_critic_loss",
    "compute_awac_loss",
    "merge_awac_metrics",
]
