# Tests for beam-search value predictor fast paths.
"""Tests for search-guidance value prediction helpers."""

from __future__ import annotations

import importlib

import torch


class ForwardValueOnlyModel(torch.nn.Module):
    """Actor-critic stub that should only be queried through forward_value."""

    def __init__(self) -> None:
        """Initialize call counters."""
        super().__init__()
        self.value_calls = 0
        self.readout_calls = 0

    def forward_value(self, states: torch.Tensor) -> torch.Tensor:
        """
        Return deterministic values from input states.

        Args:
            states: Batched state tensor.

        Returns:
            Row sums as scalar values.
        """
        self.value_calls += 1
        return states.float().sum(dim=1)

    def forward_readouts(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fail if policy/value readouts are requested.

        Args:
            states: Batched state tensor.

        Raises:
            AssertionError: Always, because beam value prediction should not
                compute policy readouts for models exposing ``forward_value``.
        """
        self.readout_calls += 1
        raise AssertionError("forward_readouts should not be called.")


def test_aux_value_predictor_uses_forward_value_fast_path() -> None:
    """Check that beam target collection avoids policy readouts when possible."""
    search_guidance = importlib.import_module("pilgrim.rl.helpers.search_guidance")
    predictor_cls = search_guidance.__dict__["_AuxValuePredictor"]
    model = ForwardValueOnlyModel()
    predictor = predictor_cls(model)
    states = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    values = predictor(states)

    assert values.tolist() == [6.0, 15.0]
    assert model.value_calls == 1
    assert model.readout_calls == 0
