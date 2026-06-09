# Tests for masked policy helpers used by beam-evaluated PPO.
"""Tests for masked PPO candidate action helpers."""

from __future__ import annotations

import torch
from pilgrim.rl import PolicyRolloutBatch
from pilgrim.rl.helpers import (
    build_masked_action_candidates,
    evaluate_masked_policy_actions,
    sample_masked_policy_actions,
)

OUTSIDE_TOPK_TARGET = 3


class TinyActorCritic(torch.nn.Module):
    """Small deterministic actor-critic for masked PPO helper tests."""

    def __init__(self) -> None:
        """Initialize fixed logits and values."""
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
        )

    def forward_readouts(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return repeated logits and row-sum values.

        Args:
            states: Batched states.

        Returns:
            Policy logits and scalar values.
        """
        batch = int(states.shape[0])
        logits = self.logits.repeat(batch, 1)
        values = states.float().sum(dim=1)
        return logits, values


def test_build_masked_action_candidates_keeps_target_action() -> None:
    """Check that a beam target outside top-k is still retained."""
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]], dtype=torch.float32)

    candidates, mask = build_masked_action_candidates(
        logits,
        target_actions=torch.tensor([OUTSIDE_TOPK_TARGET]),
        action_mode="topk",
        action_top_k=2,
        min_actions=2,
        max_actions=2,
    )

    assert candidates.shape == (1, 2)
    assert mask.tolist() == [[True, True]]
    assert OUTSIDE_TOPK_TARGET in candidates[0].tolist()


def test_evaluate_masked_policy_actions_uses_candidate_distribution() -> None:
    """Check that masked log-probs ignore actions outside the candidate set."""
    model = TinyActorCritic()
    states = torch.tensor([[1, 2, 3]], dtype=torch.long)
    candidates = torch.tensor([[0, 3]], dtype=torch.long)
    mask = torch.tensor([[True, True]])

    evaluation = evaluate_masked_policy_actions(
        model,
        states,
        torch.tensor([3]),
        candidate_actions=candidates,
        candidate_mask=mask,
    )

    expected = torch.log_softmax(torch.tensor([[0.0, 3.0]]), dim=1)[0, 1]
    assert torch.allclose(evaluation.log_probs, expected.reshape(1))
    assert evaluation.values.tolist() == [6.0]


def test_sample_masked_policy_actions_returns_candidate_actions() -> None:
    """Check that masked sampling cannot emit an action outside the row mask."""
    model = TinyActorCritic()
    states = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
    candidates = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)
    mask = torch.tensor([[True, False], [True, True]])
    generator = torch.Generator(device="cpu").manual_seed(7)

    actions, log_probs, values, entropy = sample_masked_policy_actions(
        model,
        states,
        candidate_actions=candidates,
        candidate_mask=mask,
        generator=generator,
    )

    assert actions[0].item() == 1
    assert actions[1].item() in {0, 3}
    assert log_probs.shape == (2,)
    assert values.tolist() == [0.0, 1.0]
    assert entropy.shape == (2,)


def test_policy_rollout_batch_preserves_candidate_masks_on_select() -> None:
    """Check that rollout slicing keeps masked candidate tensors aligned."""
    batch = PolicyRolloutBatch(
        states=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        actions=torch.tensor([0, 1]),
        log_probs=torch.tensor([-0.1, -0.2]),
        advantages=torch.tensor([1.0, -1.0]),
        returns=torch.tensor([2.0, 3.0]),
        values=torch.tensor([1.5, 3.5]),
        rewards=torch.tensor([-2.0, -3.0]),
        done=torch.tensor([False, True]),
        candidate_actions=torch.tensor([[0, 1], [1, 0]]),
        candidate_mask=torch.tensor([[True, False], [True, True]]),
    )

    selected = batch.index_select(torch.tensor([1]))

    assert selected.candidate_actions is not None
    assert selected.candidate_mask is not None
    assert selected.candidate_actions.tolist() == [[1, 0]]
    assert selected.candidate_mask.tolist() == [[True, True]]
