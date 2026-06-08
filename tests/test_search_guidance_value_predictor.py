# Tests for beam-search value predictor fast paths.
"""Tests for search-guidance value prediction helpers."""

from __future__ import annotations

import importlib

import pytest
import torch
from pilgrim.schemas.rl import SearchGuidedPPOBeamSearchConfig, SearchGuidedPPOConfig

ACTION_TOP_K = 4
POLICY_PRESELECT_FACTOR = 2.0
RECOVERY_NEIGHBOR_TOP_K = 4
RECOVERY_NEIGHBOR_MAX_ROWS = 64


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


def test_search_guided_ppo_topk_config_is_logged() -> None:
    """Check that top-k beam settings survive schema validation and logging."""
    config = SearchGuidedPPOConfig(
        beam_search=SearchGuidedPPOBeamSearchConfig(
            beam_mode="topk",
            action_top_k=ACTION_TOP_K,
            action_mode="topk",
            policy_preselect_factor=POLICY_PRESELECT_FACTOR,
        )
    )

    log_dict = config.to_log_dict()

    assert config.beam_search.beam_mode == "topk"
    assert config.beam_search.action_top_k == ACTION_TOP_K
    assert log_dict["beam.mode"] == "topk"
    assert log_dict["beam.action_top_k"] == ACTION_TOP_K
    assert log_dict["beam.policy_preselect_factor"] == pytest.approx(
        POLICY_PRESELECT_FACTOR
    )


def test_search_guided_ppo_recovery_neighbor_config_is_logged() -> None:
    """Check that recovery-neighbor archive settings are logged."""
    config = SearchGuidedPPOConfig(
        beam_search=SearchGuidedPPOBeamSearchConfig(
            archive_neighbor_targets=True,
            archive_neighbor_top_k=RECOVERY_NEIGHBOR_TOP_K,
            archive_neighbor_weight=0.375,
            archive_neighbor_max_rows_per_solve=RECOVERY_NEIGHBOR_MAX_ROWS,
            archive_neighbor_exclude_path_action=False,
        )
    )

    log_dict = config.to_log_dict()

    assert log_dict["beam.archive_neighbor_targets"] is True
    assert log_dict["beam.archive_neighbor_top_k"] == RECOVERY_NEIGHBOR_TOP_K
    assert log_dict["beam.archive_neighbor_weight"] == pytest.approx(0.375)
    assert (
        log_dict["beam.archive_neighbor_max_rows_per_solve"]
        == RECOVERY_NEIGHBOR_MAX_ROWS
    )
    assert log_dict["beam.archive_neighbor_exclude_path_action"] is False
