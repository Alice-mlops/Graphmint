# Provides small helpers for distributed RL training and DDP-safe model access.
"""Distributed-training helpers for reinforcement-learning trainers."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def is_distributed_initialized() -> bool:
    """Return whether ``torch.distributed`` is initialized."""
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    """
    Return the active process rank.

    Returns:
        Process rank, or ``0`` when distributed training is disabled.

    """
    if not is_distributed_initialized():
        return 0
    return int(dist.get_rank())


def distributed_world_size() -> int:
    """
    Return the active world size.

    Returns:
        Number of distributed processes, or ``1`` when disabled.

    """
    if not is_distributed_initialized():
        return 1
    return int(dist.get_world_size())


def is_main_process() -> bool:
    """
    Return whether the active process is rank zero.

    Returns:
        ``True`` when the active process should own side effects.

    """
    return distributed_rank() == 0


def local_rank_from_env(default: int = 0) -> int:
    """
    Return the local rank exported by ``torchrun``.

    Args:
        default: Fallback rank used when ``LOCAL_RANK`` is unset.

    Returns:
        Local rank parsed from the environment.

    """
    return int(os.environ.get("LOCAL_RANK", default))


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the underlying module for wrapped parallel models.

    Args:
        model: Model that may be wrapped by DDP.

    Returns:
        Underlying trainable module.

    """
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def wrap_model_for_ddp(
    model: nn.Module,
    *,
    device: torch.device,
    broadcast_buffers: bool,
    find_unused_parameters: bool,
) -> nn.Module:
    """
    Wrap a model with ``DistributedDataParallel`` when distributed is active.

    Args:
        model: Model to wrap.
        device: Local CUDA device used by the process.
        broadcast_buffers: Forwarded DDP option.
        find_unused_parameters: Forwarded DDP option.

    Returns:
        Wrapped DDP model when distributed is active, otherwise ``model``.

    Raises:
        ValueError: If DDP wrapping is requested on a non-CUDA device.

    """
    if not is_distributed_initialized() or distributed_world_size() <= 1:
        return model
    if device.type != "cuda" or device.index is None:
        raise ValueError("DDP learner mode expects a concrete CUDA device.")
    return DistributedDataParallel(
        model,
        device_ids=[int(device.index)],
        output_device=int(device.index),
        broadcast_buffers=bool(broadcast_buffers),
        find_unused_parameters=bool(find_unused_parameters),
    )


def cpu_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Return a CPU-cloned state dict for a possibly wrapped model.

    Args:
        model: Model whose weights should be materialized on CPU.

    Returns:
        CPU-cloned parameter dictionary.

    """
    source = unwrap_model(model)
    return {
        key: value.detach().cpu().clone()
        for key, value in source.state_dict().items()
    }


def load_model_state_dict(target: nn.Module, state_dict: dict[str, Any]) -> None:
    """
    Load a state dict into a possibly wrapped model.

    Args:
        target: Target model that may be DDP-wrapped.
        state_dict: State dictionary to load.

    """
    unwrap_model(target).load_state_dict(state_dict)


def synchronized_barrier() -> None:
    """Block until all distributed processes reach the barrier."""
    if is_distributed_initialized():
        dist.barrier()


def split_evenly(total: int, parts: int, index: int) -> int:
    """
    Return the even split size for one shard index.

    Args:
        total: Total number of items to partition.
        parts: Number of partitions.
        index: Zero-based shard index.

    Returns:
        Size assigned to ``index``.

    Raises:
        ValueError: If the arguments are inconsistent.

    """
    if int(parts) <= 0:
        raise ValueError("parts must be positive.")
    if not 0 <= int(index) < int(parts):
        raise ValueError("index must be in [0, parts).")
    base = int(total) // int(parts)
    remainder = int(total) % int(parts)
    return base + (1 if int(index) < remainder else 0)
