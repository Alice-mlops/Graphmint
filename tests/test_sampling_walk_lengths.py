# Tests random-walk state sampling diagnostics.
"""Tests for random-walk sampling helpers used by RL trainers."""

from __future__ import annotations

import math

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
SUFFIX_FRACTION = 0.25


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
    assert int(sample.states.shape[0]) == int(sample.steps.shape[0])
    assert set(sample.lengths.tolist()) == {SHORT_LENGTH, LONG_LENGTH}
    assert int(torch.sum(sample.lengths == SHORT_LENGTH).item()) == (
        int(RW_WIDTH * FULL_FACTOR) * SHORT_LENGTH
    )
    assert int(torch.sum(sample.lengths == LONG_LENGTH).item()) == (
        int(RW_WIDTH * HALF_FACTOR) * LONG_LENGTH
    )
    assert int(sample.steps.min().item()) >= 0
    assert int(sample.steps.max().item()) <= LONG_LENGTH


def test_sample_states_with_lengths_keeps_suffix_steps_per_length() -> None:
    """Suffix step sampling should keep only terminal rows per length bin."""
    graph = _test_graph()
    config = TDRandomWalkSamplingConfig(
        rw_mode="nbt",
        rw_width=RW_WIDTH,
        rw_length=DEFAULT_LENGTH,
        rw_lengths=((FULL_FACTOR, SHORT_LENGTH), (HALF_FACTOR, LONG_LENGTH)),
        step_sampling="suffix",
        suffix_fraction=SUFFIX_FRACTION,
        seed=SEED,
    )

    sample = sample_states_with_lengths_from_random_walks(
        graph,
        config,
        sample_index=0,
    )

    for length in (SHORT_LENGTH, LONG_LENGTH):
        group_steps = sample.steps[sample.lengths == length]
        max_step = int(group_steps.max().item())
        keep_levels = max(1, math.ceil(float(max_step + 1) * SUFFIX_FRACTION))
        min_kept_step = max(0, max_step - keep_levels + 1)
        assert int(group_steps.min().item()) >= min_kept_step
    assert int(sample.states.shape[0]) < (
        int(RW_WIDTH * FULL_FACTOR) * SHORT_LENGTH
        + int(RW_WIDTH * HALF_FACTOR) * LONG_LENGTH
    )


def test_sample_states_with_lengths_keeps_final_step_per_length() -> None:
    """Final step sampling should keep one terminal level per walk."""
    graph = _test_graph()
    config = TDRandomWalkSamplingConfig(
        rw_mode="nbt",
        rw_width=RW_WIDTH,
        rw_length=DEFAULT_LENGTH,
        rw_lengths=((FULL_FACTOR, SHORT_LENGTH), (HALF_FACTOR, LONG_LENGTH)),
        step_sampling="final",
        seed=SEED,
    )

    sample = sample_states_with_lengths_from_random_walks(
        graph,
        config,
        sample_index=0,
    )

    short_steps = sample.steps[sample.lengths == SHORT_LENGTH]
    long_steps = sample.steps[sample.lengths == LONG_LENGTH]
    assert int(short_steps.shape[0]) == int(RW_WIDTH * FULL_FACTOR)
    assert int(long_steps.shape[0]) == int(RW_WIDTH * HALF_FACTOR)
    assert set(short_steps.tolist()) == {int(short_steps.max().item())}
    assert set(long_steps.tolist()) == {int(long_steps.max().item())}
