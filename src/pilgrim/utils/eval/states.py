# Normalizes graph-state payloads for the evaluation framework.
"""State conversion helpers for evaluation code."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from pilgrim.schemas.eval.datasets import BenchmarkItem


def state_to_tuple(state: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    """
    Convert one state payload to a tuple of integers.

    Args:
        state: Sequence or tensor describing one graph state.

    Returns:
        Immutable tuple representation of the state.

    """
    tensor = torch.as_tensor(state).reshape(-1).long().cpu()
    return tuple(int(value) for value in tensor.tolist())


def states_to_tensor(
    states: Sequence[Sequence[int]] | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Convert batched states to a contiguous long tensor.

    Args:
        states: Batched graph states.
        device: Optional destination device.

    Returns:
        Tensor with shape ``(batch, state_size)``.

    Raises:
        ValueError: If the input cannot be normalized to rank two.

    """
    tensor = torch.as_tensor(states).long()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            "states must have shape (batch, state_size) or (state_size,), "
            f"got {tuple(tensor.shape)}."
        )
    if device is not None:
        tensor = tensor.to(device)
    return tensor.contiguous()


def item_states_to_tensor(
    items: Sequence[BenchmarkItem],
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Stack benchmark-item states into one tensor.

    Args:
        items: Benchmark items to stack.
        device: Optional destination device.

    Returns:
        Tensor with shape ``(len(items), state_size)``.

    """
    return states_to_tensor([item.state for item in items], device=device)
