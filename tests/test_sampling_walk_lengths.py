# Tests random-walk state sampling diagnostics.
"""Tests for random-walk sampling helpers used by RL trainers."""

from __future__ import annotations

import torch
from cayleypy import CayleyGraph
from cayleypy.graphs_lib import PermutationGroups
from pilgrim.rl.sampling import sample_states_with_lengths_from_random_walks
from pilgrim.schemas.rl import TDRandomWalkSamplingConfig

FULL_FACTOR = 1.0
HALF_FACTOR = 0.5
RW_WIDTH = 4
DEFAULT_LENGTH = 5
SHORT_LENGTH = 3
LONG_LENGTH = 7
SEED = 123


def _test_graph() -> CayleyGraph:
    """
    Build a small pancake graph for sampling tests.

    Returns:
        CPU Cayley graph with four pancake generators.
    """
    return CayleyGraph(PermutationGroups.pancake(5), device="cpu")


def test_sample_states_with_lengths_aligns_labels_to_rows() -> None:
    """Length labels should align one-to-one with sampled state rows."""
    graph = _test_graph()
    config = TDRandomWalkSamplingConfig(
        rw_mode="nbt",
        rw_width=RW_WIDTH,
        rw_length=DEFAULT_LENGTH,
        rw_lengths=((FULL_FACTOR, SHORT_LENGTH), (HALF_FACTOR, LONG_LENGTH)),
        seed=SEED,
    )

    sample = sample_states_with_lengths_from_random_walks(
        graph,
        config,
        sample_index=0,
    )

    assert int(sample.states.shape[0]) == int(sample.lengths.shape[0])
    assert set(sample.lengths.tolist()) == {SHORT_LENGTH, LONG_LENGTH}
    assert int(torch.sum(sample.lengths == SHORT_LENGTH).item()) == (
        int(RW_WIDTH * FULL_FACTOR) * SHORT_LENGTH
    )
    assert int(torch.sum(sample.lengths == LONG_LENGTH).item()) == (
        int(RW_WIDTH * HALF_FACTOR) * LONG_LENGTH
    )
