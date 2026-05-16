# Tests sampled multi-step TD target construction.
"""Smoke tests for sampled TD-lambda backup targets."""

from __future__ import annotations

import torch
from cayleypy import CayleyGraph
from cayleypy.graphs_lib import PermutationGroups
from torch import nn

from pilgrim.rl import (
    SampledBackupConfig,
    compute_sampled_td_lambda_value_targets,
    compute_td_lambda_value_targets,
)
from pilgrim.rl.transitions import compute_configured_value_targets
from pilgrim.schemas.rl import MultiStepTDValueConfig, TDTargetSamplingConfig


class _WeightedPermutationValue(nn.Module):
    """Small deterministic scalar model for permutation states."""

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return a position-weighted sum so different permutations vary."""
        batch = states.float()
        weights = torch.arange(
            1,
            batch.shape[1] + 1,
            device=batch.device,
            dtype=batch.dtype,
        )
        return batch.matmul(weights)


def _test_graph() -> CayleyGraph:
    """Build a tiny CPU permutation graph for target tests."""
    return CayleyGraph(PermutationGroups.pancake(4), device="cpu")


def test_sampled_td_lambda_matches_exact_when_all_actions_are_used() -> None:
    """Sampled target construction should reduce to exact backups at full width."""
    graph = _test_graph()
    model = _WeightedPermutationValue().eval()
    states = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [2, 0, 3, 1],
        ],
        dtype=torch.long,
    )

    exact = compute_td_lambda_value_targets(
        model,
        graph,
        states,
        num_steps=3,
        td_lambda=0.5,
        reward_per_step=1.0,
        discount=1.0,
    )
    sampled = compute_sampled_td_lambda_value_targets(
        model,
        graph,
        states,
        num_steps=3,
        td_lambda=0.5,
        reward_per_step=1.0,
        discount=1.0,
        sampled_backup=SampledBackupConfig(
            enabled=True,
            action_sample_size=len(graph.generators),
            action_sample_repeats=2,
            seed=123,
        ),
    )

    torch.testing.assert_close(sampled, exact)


def test_sampled_targets_are_deterministic_for_same_sample_index() -> None:
    """Fixed seed and sample index should reproduce the same sampled target."""
    graph = _test_graph()
    model = _WeightedPermutationValue().eval()
    states = torch.tensor(
        [
            [3, 2, 1, 0],
            [2, 0, 3, 1],
            [1, 3, 0, 2],
        ],
        dtype=torch.long,
    )
    sampled_backup = SampledBackupConfig(
        enabled=True,
        action_sample_size=1,
        root_action_sample_size=1,
        action_sample_repeats=3,
        horizon_sample_size=2,
        seed=999,
    )

    first = compute_configured_value_targets(
        model,
        graph,
        states,
        reward_per_step=1.0,
        discount=1.0,
        n_steps=3,
        td_lambda=0.6,
        terminal_value=0.0,
        sampled_backup=sampled_backup,
        sample_index=7,
    )
    second = compute_configured_value_targets(
        model,
        graph,
        states,
        reward_per_step=1.0,
        discount=1.0,
        n_steps=3,
        td_lambda=0.6,
        terminal_value=0.0,
        sampled_backup=sampled_backup,
        sample_index=7,
    )

    assert first.shape == (3,)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_multistep_td_config_accepts_target_sampling() -> None:
    """The trainer schema should expose target-sampling config and logging."""
    cfg = MultiStepTDValueConfig(
        n_steps=3,
        td_lambda=0.75,
        target_sampling=TDTargetSamplingConfig(
            enabled=True,
            action_sample_size=4,
            action_sample_repeats=2,
            horizon_sample_size=3,
            seed=1234,
        ),
    )
    log_dict = cfg.to_log_dict()

    assert cfg.target_sampling.enabled is True
    assert log_dict["target_sampling.enabled"] is True
    assert log_dict["target_sampling.action_sample_size"] == 4
    assert log_dict["target_sampling.action_sample_repeats"] == 2
    assert log_dict["target_sampling.horizon_sample_size"] == 3
