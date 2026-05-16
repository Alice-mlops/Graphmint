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
from pilgrim.rl.transitions import (
    _enumerate_sampled_neighbor_states,
    compute_configured_value_targets,
)
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


def test_sampled_neighbor_enumeration_vectorized_matches_fixed_generator_apply() -> None:
    """Per-row sampled permutation neighbors should match CayleyPy application."""
    graph = _test_graph()
    states = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [2, 0, 3, 1],
            [1, 3, 0, 2],
        ],
        dtype=torch.long,
    )
    generator = torch.Generator()
    generator.manual_seed(20260516)

    neighbors, sampled_generators = _enumerate_sampled_neighbor_states(
        graph,
        states,
        generator_indices=None,
        action_sample_size=2,
        generator=generator,
    )

    expected = torch.empty_like(neighbors)
    for sample_position in range(sampled_generators.shape[1]):
        column_generators = sampled_generators[:, sample_position]
        for generator_index in torch.unique(column_generators).tolist():
            rows = torch.nonzero(
                column_generators == int(generator_index),
                as_tuple=False,
            ).reshape(-1)
            dst = torch.empty(
                (rows.numel(), states.shape[1]),
                dtype=states.dtype,
                device=states.device,
            )
            graph.apply_generator_batched(
                int(generator_index),
                states.index_select(0, rows),
                dst,
            )
            expected[rows, sample_position, :] = dst

    assert neighbors.shape == (states.shape[0], 2, states.shape[1])
    torch.testing.assert_close(neighbors, expected)


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
