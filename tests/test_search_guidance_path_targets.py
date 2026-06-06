# Tests beam-search path expansion for PPO search guidance.
# ruff: noqa: SLF001
"""Tests for path-expanded search-guided PPO supervision targets."""

from __future__ import annotations

import torch
from cayleypy import CayleyGraph
from cayleypy.graphs_lib import PermutationGroups
from pilgrim.rl.helpers import search_guidance
from pilgrim.rl.helpers.q_learning import apply_actions
from pilgrim.schemas.rl import SearchGuidedPPOBeamSearchConfig


def _test_graph() -> CayleyGraph:
    """
    Build a small pancake graph for path-target tests.

    Returns:
        CPU Cayley graph with four pancake generators.

    """
    return CayleyGraph(PermutationGroups.pancake(5), device="cpu")


def test_select_path_target_positions_applies_stride_and_even_cap() -> None:
    """Stride is applied first, then cap keeps positions across the path."""
    positions = search_guidance._select_path_target_positions(
        path_length=10,
        stride=2,
        max_rows=3,
    )

    assert positions == [0, 4, 8]


def test_path_expansion_adds_selected_intermediate_states() -> None:
    """A solved beam path should add capped/strided pre-action states."""
    graph = _test_graph()
    state = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
    path = [0, 1, 2, 3, 0]
    config = SearchGuidedPPOBeamSearchConfig(
        archive_path_targets=True,
        archive_path_stride=2,
        archive_max_path_rows_per_solve=3,
    )
    kept_states: list[torch.Tensor] = []
    action_targets: list[int] = []
    value_targets: list[float] = []
    weights: list[float] = []
    state_indices: list[int] = []
    path_lengths: list[int] = []
    best_widths: list[int] = []

    search_guidance._append_beam_path_supervision(
        graph=graph,
        state=state,
        path=path,
        source_index=7,
        best_width=32,
        config=config,
        expand_path=True,
        kept_states=kept_states,
        action_targets=action_targets,
        value_targets=value_targets,
        weights=weights,
        state_indices=state_indices,
        path_lengths=path_lengths,
        best_widths=best_widths,
    )

    expected_states = [state.view(1, -1)]
    current = state.view(1, -1)
    for action in path[:2]:
        current = apply_actions(
            graph,
            current,
            torch.tensor([action], dtype=torch.long),
        )
    expected_states.append(current.cpu())
    for action in path[2:4]:
        current = apply_actions(
            graph,
            current,
            torch.tensor([action], dtype=torch.long),
        )
    expected_states.append(current.cpu())

    assert action_targets == [0, 2, 0]
    assert value_targets == [5.0, 3.0, 1.0]
    assert state_indices == [7, 7, 7]
    assert path_lengths == [5, 3, 1]
    assert best_widths == [32, 32, 32]
    torch.testing.assert_close(
        torch.cat(kept_states, dim=0), torch.cat(expected_states)
    )
    torch.testing.assert_close(
        torch.tensor(weights),
        torch.full((3,), 1.0 / 3.0),
    )


def test_nonexpanded_path_keeps_legacy_single_row_behavior() -> None:
    """Legacy search targets should still add only the source state."""
    graph = _test_graph()
    state = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
    config = SearchGuidedPPOBeamSearchConfig(archive_path_targets=True)
    kept_states: list[torch.Tensor] = []
    action_targets: list[int] = []
    value_targets: list[float] = []
    weights: list[float] = []
    state_indices: list[int] = []
    path_lengths: list[int] = []
    best_widths: list[int] = []

    search_guidance._append_beam_path_supervision(
        graph=graph,
        state=state,
        path=[0, 1, 2],
        source_index=2,
        best_width=16,
        config=config,
        expand_path=False,
        kept_states=kept_states,
        action_targets=action_targets,
        value_targets=value_targets,
        weights=weights,
        state_indices=state_indices,
        path_lengths=path_lengths,
        best_widths=best_widths,
    )

    assert len(kept_states) == 1
    torch.testing.assert_close(kept_states[0], state.view(1, -1))
    assert action_targets == [0]
    assert value_targets == [3.0]
    assert weights == [1.0]
    assert state_indices == [2]
    assert path_lengths == [3]
    assert best_widths == [16]
