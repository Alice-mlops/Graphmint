# Tests the transferable pancake shared-action scoring model.
"""Tests for the shared per-action pancake scorer."""

from __future__ import annotations

import pytest
import torch
from pilgrim.model import PancakeActionScorer
from pilgrim.rl import PolicySupervisionBatch, compute_supervised_policy_value_losses
from pilgrim_lightning.model_factory import build_model


def _tiny_model() -> PancakeActionScorer:
    """
    Build a small scorer for unit tests.

    Returns:
        A shape-stable shared-action scorer.

    """
    return PancakeActionScorer(
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        max_n=12,
    )


def test_forward_width_tracks_pancake_size() -> None:
    """One model instance should produce ``n - 1`` action logits for each n."""
    model = _tiny_model().eval()

    n5 = torch.tensor([[0, 1, 2, 3, 4], [3, 1, 2, 0, 4]], dtype=torch.long)
    n8 = torch.tensor([[7, 0, 2, 1, 3, 5, 4, 6]], dtype=torch.long)

    with torch.no_grad():
        logits5 = model(n5)
        logits8 = model(n8)

    assert logits5.shape == (2, 4)
    assert logits8.shape == (1, 7)
    assert torch.isfinite(logits5).all()
    assert torch.isfinite(logits8).all()


def test_forward_readouts_support_supervised_policy_value_loss() -> None:
    """Existing supervised policy/value helpers should accept the scorer."""
    model = _tiny_model()
    batch = PolicySupervisionBatch(
        states=torch.tensor(
            [
                [3, 1, 2, 0, 4],
                [4, 0, 1, 2, 3],
                [1, 0, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
        action_targets=torch.tensor([3, 2, 1], dtype=torch.long),
        value_targets=torch.tensor([5.0, 4.0, 1.0], dtype=torch.float32),
        weights=torch.ones(3, dtype=torch.float32),
    )

    policy_loss, value_loss, metrics = compute_supervised_policy_value_losses(
        model,
        batch,
        device="cpu",
    )

    assert policy_loss.ndim == 0
    assert value_loss.ndim == 0
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert 0.0 <= metrics["action_accuracy"] <= 1.0
    assert metrics["value_mae"] >= 0.0


def test_model_factory_builds_pancake_action_scorer() -> None:
    """The Lightning model factory should construct the scorer by name."""
    model = build_model(
        "PancakeActionScorer",
        {
            "d_model": 16,
            "num_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "max_n": 10,
        },
    )

    assert isinstance(model, PancakeActionScorer)
    assert model(torch.arange(6).view(1, -1)).shape == (1, 5)


def test_invalid_state_rows_are_rejected() -> None:
    """Duplicate labels should be rejected when input validation is enabled."""
    model = _tiny_model()

    with pytest.raises(ValueError, match="permutation"):
        model(torch.tensor([[0, 1, 1, 3]], dtype=torch.long))
