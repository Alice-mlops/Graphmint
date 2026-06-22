# Tests generic AWAC losses for masked discrete actor-critic policies.
"""Tests for advantage-weighted actor-critic loss helpers."""

from __future__ import annotations

import pytest
import torch
from pilgrim.rl import AWACConfig, AWACTargetBatch, compute_awac_loss

LOW_LOSS_THRESHOLD = 0.01
HIGH_PROBABILITY_THRESHOLD = 0.99
UNIFORM_LOSS_THRESHOLD = 7.0
BAD_ACTION_PROBABILITY_THRESHOLD = 0.9
BAD_ACTION_LOSS_THRESHOLD = 4.0


def test_awac_set_probability_accepts_multiple_optimal_actions() -> None:
    """Check that probability mass on any optimal action lowers the actor loss."""
    logits = torch.tensor([[5.0, 5.0, -2.0]], dtype=torch.float32)
    targets = AWACTargetBatch(
        optimal_action_mask=torch.tensor([[True, True, False]]),
    )

    loss_state = compute_awac_loss(logits=logits, targets=targets)

    assert loss_state.policy_loss.item() < LOW_LOSS_THRESHOLD
    assert loss_state.optimal_action_probability.item() > HIGH_PROBABILITY_THRESHOLD
    assert loss_state.top1_optimal_accuracy.item() == pytest.approx(1.0)


def test_awac_uniform_targets_penalizes_ignoring_one_optimal_action() -> None:
    """Check the stricter uniform-target mode for equal optimal actions."""
    logits = torch.tensor([[8.0, -8.0, 0.0]], dtype=torch.float32)
    targets = AWACTargetBatch(
        optimal_action_mask=torch.tensor([[True, True, False]]),
    )

    set_loss = compute_awac_loss(
        logits=logits,
        targets=targets,
        config=AWACConfig(set_loss_mode="set_probability"),
    )
    uniform_loss = compute_awac_loss(
        logits=logits,
        targets=targets,
        config=AWACConfig(set_loss_mode="uniform_targets"),
    )

    assert set_loss.policy_loss.item() < LOW_LOSS_THRESHOLD
    assert uniform_loss.policy_loss.item() > UNIFORM_LOSS_THRESHOLD


def test_awac_advantage_weights_increase_high_advantage_rows() -> None:
    """Check that a high-advantage row dominates the weighted actor loss."""
    logits = torch.tensor(
        [
            [0.0, 4.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    targets = AWACTargetBatch(
        optimal_action_mask=torch.tensor([
            [True, False],
            [False, True],
        ]),
        advantages=torch.tensor([2.0, 0.0], dtype=torch.float32),
    )

    weighted = compute_awac_loss(
        logits=logits,
        targets=targets,
        config=AWACConfig(advantage_temperature=1.0, normalize_weights=False),
    )
    unweighted = compute_awac_loss(
        logits=logits,
        targets=AWACTargetBatch(
            optimal_action_mask=targets.optimal_action_mask,
            advantages=torch.zeros(2),
        ),
        config=AWACConfig(normalize_weights=False),
    )

    assert weighted.mean_awac_weight.item() > unweighted.mean_awac_weight.item()
    assert weighted.policy_loss.item() > unweighted.policy_loss.item()


def test_awac_bad_action_penalty_uses_known_bad_mask() -> None:
    """Check that bad-action probability mass creates a positive penalty."""
    logits = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32)
    targets = AWACTargetBatch(
        optimal_action_mask=torch.tensor([[True, False, False]]),
        bad_action_mask=torch.tensor([[False, False, True]]),
    )

    loss_state = compute_awac_loss(
        logits=logits,
        targets=targets,
        config=AWACConfig(bad_action_coef=1.0, bad_action_margin_coef=1.0),
    )

    assert loss_state.bad_action_probability.item() > BAD_ACTION_PROBABILITY_THRESHOLD
    assert loss_state.bad_action_loss.item() > BAD_ACTION_LOSS_THRESHOLD
    assert loss_state.bad_action_margin_loss.item() > BAD_ACTION_LOSS_THRESHOLD


def test_awac_anchor_kl_respects_action_mask() -> None:
    """Check that anchor KL ignores invalid masked actions."""
    logits = torch.tensor([[0.0, 2.0, -20.0]], dtype=torch.float32)
    action_mask = torch.tensor([[True, True, False]])
    reference = torch.log_softmax(
        torch.tensor([[2.0, 0.0, -20.0]], dtype=torch.float32),
        dim=1,
    )
    targets = AWACTargetBatch(
        optimal_action_mask=torch.tensor([[True, False, False]]),
        action_mask=action_mask,
        reference_log_probs=reference,
    )

    loss_state = compute_awac_loss(
        logits=logits,
        targets=targets,
        config=AWACConfig(policy_anchor_coef=1.0),
    )

    assert loss_state.policy_anchor_loss.item() > 0.0


def test_awac_requires_one_training_target() -> None:
    """Check that actor targets are required."""
    with pytest.raises(ValueError, match="optimal_action_mask or sampled_actions"):
        compute_awac_loss(
            logits=torch.zeros((1, 2)),
            targets=AWACTargetBatch(),
        )
