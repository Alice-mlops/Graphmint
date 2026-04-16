"""Loss utilities for Pilgrim models."""

from collections.abc import Sequence
from typing import Literal

import torch
from cayleypy import CayleyGraph
from torch import nn


def _subsample_states(
    states: torch.Tensor,
    max_states: int | None,
    *,
    seed: int | None,
) -> torch.Tensor:
    if max_states is None or states.shape[0] <= max_states:
        return states
    if seed is None:
        idx = torch.randperm(states.shape[0], device=states.device)[:max_states]
    else:
        g = torch.Generator(device=states.device)
        g.manual_seed(seed)
        idx = torch.randperm(states.shape[0], generator=g, device=states.device)[
            :max_states
        ]
    return states[idx]


def _sample_generator_indices(
    num_generators: int,
    *,
    generator_indices: Sequence[int] | None,
    max_generators: int | None,
    seed: int | None,
) -> list[int]:
    if generator_indices is None:
        base = list(range(num_generators))
    else:
        base = list(generator_indices)

    if max_generators is None or max_generators >= len(base):
        return base

    if seed is None:
        perm = torch.randperm(len(base))[:max_generators]
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        perm = torch.randperm(len(base), generator=g)[:max_generators]
    return [base[i] for i in perm.tolist()]


def lipschitz_expansion_loss(
    model: nn.Module,
    graph: CayleyGraph,
    states: torch.Tensor,
    *,
    max_states: int | None = None,
    generator_indices: Sequence[int] | None = None,
    max_generators: int | None = None,
    seed: int | None = None,
    state_batch_size: int | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    Compute 1-Lipschitz expansion penalty over neighbors in a Cayley graph.

    For sampled nodes v and their neighbors u, penalize violations of
    |d(u) - d(v)| <= 1 using:
        sum_{u,v} max(0, |d(u) - d(v)| - 1)^2

    Args:
        model: Network used to predict distances d(·).
        graph: Cayley graph providing generators.
        states: Encoded states v (shape [N, state_size]).
        max_states: Optional cap on number of states to sample from `states`.
            Use -1 to disable subsampling.
        generator_indices: Explicit generator indices to apply. If None, uses all.
        max_generators: Optional cap on number of generators to sample.
            Use -1 to disable subsampling.
        seed: Seed for sampling states/generators.
        state_batch_size: Optional batch size for processing states.
        reduction: "mean" (default) or "sum" over all (u, v) pairs.

    """
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    if states.ndim == 1:
        states = states.unsqueeze(0)

    if max_states == -1:
        max_states = None
    if max_generators == -1:
        max_generators = None

    states = _subsample_states(states, max_states, seed=seed)
    if states.shape[0] == 0:
        param = next(model.parameters(), None)
        dtype = param.dtype if param is not None else torch.float32
        return torch.zeros((), device=states.device, dtype=dtype)

    gen_idx = _sample_generator_indices(
        len(graph.generators),
        generator_indices=generator_indices,
        max_generators=max_generators,
        seed=seed,
    )
    if not gen_idx:
        param = next(model.parameters(), None)
        dtype = param.dtype if param is not None else torch.float32
        return torch.zeros((), device=states.device, dtype=dtype)

    if state_batch_size is None:
        state_batch_size = states.shape[0]

    total_loss: torch.Tensor | None = None
    total_pairs = 0

    for s in range(0, states.shape[0], state_batch_size):
        v = states[s : s + state_batch_size]
        d_v = model(v).reshape(-1)

        for i in gen_idx:
            dst = torch.empty_like(v)
            graph.apply_generator_batched(i, v, dst)
            d_u = model(dst).reshape(-1)
            violation = (d_u - d_v).abs() - 1.0
            penalty = torch.relu(violation).pow(2)
            if total_loss is None:
                total_loss = penalty.sum()
            else:
                total_loss = total_loss + penalty.sum()
            total_pairs += penalty.numel()

    if total_loss is None:
        param = next(model.parameters(), None)
        dtype = param.dtype if param is not None else torch.float32
        return torch.zeros((), device=states.device, dtype=dtype)

    if reduction == "sum":
        return total_loss

    if total_pairs == 0:
        return total_loss
    return total_loss / float(total_pairs)
