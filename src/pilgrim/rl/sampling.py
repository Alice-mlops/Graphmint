# Collects replay states from CayleyGraph random walks for RL training.
"""Random-walk sampling helpers shared by RL trainers."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from cayleypy import CayleyGraph

from pilgrim.schemas.rl import TDRandomWalkSamplingConfig

from .config import RandomWalkSamplingConfig


def sample_states_from_random_walks(
    graph: CayleyGraph,
    config: RandomWalkSamplingConfig | TDRandomWalkSamplingConfig,
    *,
    sample_index: int = 0,
) -> torch.Tensor:
    """
    Sample a batch of states from the graph without using labels.

    Args:
        graph: Cayley graph used to generate random walks.
        config: Random-walk sampling configuration.
        sample_index: Sampling-call index used to derive a deterministic seed.

    Returns:
        Tensor of sampled states with shape ``(batch, state_size)``.

    """
    _set_sampling_seed(base_seed=int(config.seed), sample_index=sample_index)
    schedule = resolve_rw_schedule(config)
    states: list[torch.Tensor] = []

    for factor, length in schedule:
        if int(length) < 1:
            continue
        width = max(1, int(int(config.rw_width) * float(factor)))
        x_part, _ = graph.random_walks(
            width=width,
            length=int(length),
            mode=str(config.rw_mode),
            nbt_history_depth=int(length),
        )
        states.append(torch.as_tensor(x_part).long())

    if not states:
        x_part, _ = graph.random_walks(
            width=int(config.rw_width),
            length=int(config.rw_length),
            mode=str(config.rw_mode),
            nbt_history_depth=int(config.rw_length),
        )
        states.append(torch.as_tensor(x_part).long())

    return torch.cat(states, dim=0)


def sample_suffix_states_from_random_walks(
    graph: CayleyGraph,
    *,
    rw_mode: str,
    rw_width: int,
    rw_length: int,
    suffix_fraction: float,
    base_seed: int,
    sample_index: int = 0,
    nbt_history_depth: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample only the suffix of longer random walks.

    This helper is intended for frontier-state discovery. It keeps only states
    from the tail of each random walk so the candidate pool is biased toward
    states reached late in the walk rather than the many shallow prefixes near
    the center.

    Args:
        graph: Cayley graph used to generate random walks.
        rw_mode: Random-walk mode passed to ``graph.random_walks``.
        rw_width: Width used for walk generation.
        rw_length: Length used for walk generation.
        suffix_fraction: Fraction of walk levels kept from the end.
        base_seed: Base random seed for deterministic sampling.
        sample_index: Sampling-call index used to derive a deterministic seed.
        nbt_history_depth: Optional non-backtracking history depth for
            ``"nbt"`` random walks.

    Returns:
        Tuple ``(states, steps)`` containing suffix states and their walk steps.

    Raises:
        ValueError: If one of the parameters is invalid.

    """
    if int(rw_width) <= 0:
        raise ValueError("rw_width must be positive.")
    if int(rw_length) <= 0:
        raise ValueError("rw_length must be positive.")
    if not 0.0 < float(suffix_fraction) <= 1.0:
        raise ValueError("suffix_fraction must be in the open interval (0, 1].")

    _set_sampling_seed(base_seed=int(base_seed), sample_index=sample_index)
    history_depth = _resolve_nbt_history_depth(
        rw_mode=rw_mode,
        rw_length=int(rw_length),
        nbt_history_depth=nbt_history_depth,
    )
    sampled_states, sampled_steps = graph.random_walks(
        width=int(rw_width),
        length=int(rw_length),
        mode=str(rw_mode),
        nbt_history_depth=history_depth,
    )
    states = torch.as_tensor(sampled_states).long()
    steps = torch.as_tensor(sampled_steps).reshape(-1).long()
    if steps.numel() == 0:
        return states.reshape(0, *states.shape[1:]), steps

    max_step = int(steps.max().item())
    keep_levels = max(1, math.ceil(float(max_step + 1) * float(suffix_fraction)))
    min_step = max(0, max_step - keep_levels + 1)
    mask = steps >= min_step
    return states[mask].contiguous(), steps[mask].contiguous()


def subsample_states(
    states: torch.Tensor,
    *,
    max_states: int,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Randomly subsample state rows without replacement.

    Args:
        states: State tensor with shape ``(batch, state_size)``.
        max_states: Maximum number of rows to keep.
        seed: Optional seed for deterministic subsampling.

    Returns:
        Original states when ``batch <= max_states``; otherwise a randomly
        selected subset with ``max_states`` rows.

    Raises:
        ValueError: If ``max_states`` is not positive.

    """
    if int(max_states) <= 0:
        raise ValueError("max_states must be positive.")

    data = torch.as_tensor(states).long()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.shape[0] <= int(max_states):
        return data.contiguous()

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    indices = torch.randperm(int(data.shape[0]), generator=generator)[: int(max_states)]
    return data[indices].contiguous()


def resolve_rw_schedule(
    config: RandomWalkSamplingConfig | TDRandomWalkSamplingConfig,
) -> list[tuple[float, int]]:
    """
    Resolve the random-walk schedule used by replay sampling.

    Args:
        config: Random-walk sampling configuration.

    Returns:
        Schedule of ``(factor, length)`` pairs.

    """
    if config.rw_lengths is not None:
        return [(float(factor), int(length)) for factor, length in config.rw_lengths]
    base_len = int(config.rw_length)
    return [
        (1.0, base_len),
        (0.5, base_len // 2),
        (0.25, base_len // 4),
    ]


def _set_sampling_seed(*, base_seed: int, sample_index: int) -> None:
    """
    Seed Python, NumPy, and Torch RNGs for one sampling call.

    Args:
        base_seed: Base seed from the sampling config.
        sample_index: Index of the current sampling call.

    """
    seed = int(base_seed) + int(sample_index) * 100_003
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_nbt_history_depth(
    *,
    rw_mode: str,
    rw_length: int,
    nbt_history_depth: int | None,
) -> int:
    """
    Resolve the history depth passed to ``graph.random_walks``.

    Args:
        rw_mode: Random-walk mode.
        rw_length: Random-walk length.
        nbt_history_depth: Optional explicit history depth.

    Returns:
        History depth compatible with the selected random-walk mode.

    """
    if str(rw_mode) != "nbt":
        return 0
    if nbt_history_depth is None:
        return int(rw_length)
    return max(0, int(nbt_history_depth))
