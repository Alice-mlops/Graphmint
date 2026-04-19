# Stores supervised policy/value targets used to anchor search-guided PPO.
"""Supervision archives for reverse-trajectory and search-derived targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .replay import (
    _normalize_1d_tensor,
    _normalize_state_rows,
    _sample_indices,
    _write_ring_tensor,
)


@dataclass(slots=True, frozen=True)
class PolicySupervisionBatch:
    """
    Aligned policy/value supervision rows.

    Args:
        states: Input states with shape ``(batch, state_size)``.
        action_targets: Generator ids that should be preferred by the policy.
        value_targets: Scalar value targets aligned with the states.
        weights: Optional per-row weights used by auxiliary losses.

    """

    states: torch.Tensor
    action_targets: torch.Tensor
    value_targets: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        """
        Validate batch alignment and normalize dtypes.

        Raises:
            ValueError: If one of the tensors is not row-aligned.

        """
        state_device = torch.as_tensor(self.states).device
        batch_states = _normalize_state_rows(self.states, device=state_device)
        batch_actions = _normalize_1d_tensor(
            self.action_targets,
            device=state_device,
            dtype=torch.long,
            name="action_targets",
        )
        batch_values = _normalize_1d_tensor(
            self.value_targets,
            device=state_device,
            dtype=torch.float32,
            name="value_targets",
        )
        batch_weights = _normalize_1d_tensor(
            self.weights,
            device=state_device,
            dtype=torch.float32,
            name="weights",
        )
        batch_size = int(batch_states.shape[0])
        if int(batch_actions.shape[0]) != batch_size:
            raise ValueError("action_targets must align with states.")
        if int(batch_values.shape[0]) != batch_size:
            raise ValueError("value_targets must align with states.")
        if int(batch_weights.shape[0]) != batch_size:
            raise ValueError("weights must align with states.")
        object.__setattr__(self, "states", batch_states)
        object.__setattr__(self, "action_targets", batch_actions)
        object.__setattr__(self, "value_targets", batch_values)
        object.__setattr__(self, "weights", batch_weights)

    def __len__(self) -> int:
        """
        Return the batch size.

        Returns:
            Number of aligned rows.

        """
        return int(self.states.shape[0])

    def to(self, device: str | torch.device) -> PolicySupervisionBatch:
        """
        Move the batch to ``device``.

        Args:
            device: Destination device.

        Returns:
            Batch on the requested device.

        """
        target_device = torch.device(device)
        return PolicySupervisionBatch(
            states=self.states.to(target_device),
            action_targets=self.action_targets.to(target_device),
            value_targets=self.value_targets.to(target_device),
            weights=self.weights.to(target_device),
        )

    def index_select(self, indices: torch.Tensor) -> PolicySupervisionBatch:
        """
        Select a subset of rows.

        Args:
            indices: One-dimensional row indices.

        Returns:
            Selected supervision batch.

        """
        rows = torch.as_tensor(indices, device=self.states.device).long().reshape(-1)
        return PolicySupervisionBatch(
            states=self.states.index_select(0, rows),
            action_targets=self.action_targets.index_select(0, rows),
            value_targets=self.value_targets.index_select(0, rows),
            weights=self.weights.index_select(0, rows),
        )


def concatenate_policy_supervision_batches(
    batches: list[PolicySupervisionBatch],
) -> PolicySupervisionBatch:
    """
    Concatenate multiple supervision batches.

    Args:
        batches: Non-empty list of supervision batches.

    Returns:
        Concatenated batch.

    Raises:
        ValueError: If ``batches`` is empty.

    """
    if not batches:
        raise ValueError("at least one supervision batch is required.")
    if len(batches) == 1:
        return batches[0]
    return PolicySupervisionBatch(
        states=torch.cat([batch.states for batch in batches], dim=0),
        action_targets=torch.cat([batch.action_targets for batch in batches], dim=0),
        value_targets=torch.cat([batch.value_targets for batch in batches], dim=0),
        weights=torch.cat([batch.weights for batch in batches], dim=0),
    )


def subsample_policy_supervision_batch(
    batch: PolicySupervisionBatch,
    *,
    max_rows: int,
    seed: int | None = None,
) -> PolicySupervisionBatch:
    """
    Randomly subsample supervision rows without replacement.

    Args:
        batch: Batch to subsample.
        max_rows: Maximum number of rows kept.
        seed: Optional deterministic seed.

    Returns:
        Original batch when small enough; otherwise a random subset.

    Raises:
        ValueError: If ``max_rows`` is not positive.

    """
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive.")
    if len(batch) <= int(max_rows):
        return batch
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    rows = torch.randperm(len(batch), generator=generator)[: int(max_rows)]
    return batch.index_select(rows.to(batch.states.device))


class PolicySupervisionArchive:
    """
    Ring-buffer archive for policy/value supervision rows.

    Args:
        capacity: Maximum number of stored rows.
        storage_device: Device used by internal storage tensors.

    Raises:
        ValueError: If ``capacity`` is not positive.

    """

    def __init__(
        self,
        capacity: int,
        *,
        storage_device: str | torch.device = "cpu",
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.storage_device = torch.device(storage_device)
        self._states: torch.Tensor | None = None
        self._action_targets: torch.Tensor | None = None
        self._value_targets: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        """
        Return the number of stored rows.

        Returns:
            Current archive size.

        """
        return int(self._size)

    def add(self, batch: PolicySupervisionBatch) -> int:
        """
        Append a supervision batch to the archive.

        Args:
            batch: Supervision rows to append.

        Returns:
            Number of written rows.

        Raises:
            ValueError: If ``batch`` is empty.

        """
        data = batch.to(self.storage_device)
        if len(data) == 0:
            raise ValueError("supervision batch must contain at least one row.")
        if self._states is None:
            self._allocate_storage(data)

        assert self._states is not None
        assert self._action_targets is not None
        assert self._value_targets is not None
        assert self._weights is not None

        written, next_index = _write_ring_tensor(
            self._states,
            data.states,
            next_index=self._next_index,
        )
        _write_ring_tensor(
            self._action_targets,
            data.action_targets,
            next_index=self._next_index,
        )
        _write_ring_tensor(
            self._value_targets,
            data.value_targets,
            next_index=self._next_index,
        )
        _write_ring_tensor(
            self._weights,
            data.weights,
            next_index=self._next_index,
        )
        self._next_index = next_index
        self._size = min(self.capacity, self._size + written)
        return written

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: str | torch.device | None = None,
    ) -> PolicySupervisionBatch:
        """
        Sample rows uniformly from the archive.

        Args:
            batch_size: Number of rows to sample.
            generator: Optional RNG used for sampling.
            device: Optional output device.

        Returns:
            Sampled supervision batch.

        Raises:
            RuntimeError: If the archive is empty.
            ValueError: If ``batch_size`` is not positive.

        """
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if self._states is None or self._size == 0:
            raise RuntimeError("cannot sample from an empty supervision archive.")

        rows = _sample_indices(
            size=int(self._size),
            batch_size=int(batch_size),
            device=self.storage_device,
            generator=generator,
        )
        batch = PolicySupervisionBatch(
            states=self._states[rows],
            action_targets=self._action_targets[rows],
            value_targets=self._value_targets[rows],
            weights=self._weights[rows],
        )
        if device is not None:
            return batch.to(device)
        return batch

    def is_ready(self, min_size: int) -> bool:
        """
        Return whether the archive contains at least ``min_size`` rows.

        Args:
            min_size: Required minimum number of rows.

        Returns:
            ``True`` when the archive is large enough.

        """
        return len(self) >= int(min_size)

    def storage_usage_ratio(self) -> float:
        """
        Return the archive fill ratio.

        Returns:
            Fraction in ``[0.0, 1.0]``.

        """
        return float(self._size) / float(self.capacity)

    def _allocate_storage(self, batch: PolicySupervisionBatch) -> None:
        """
        Allocate storage tensors using the first observed batch.

        Args:
            batch: Normalized batch defining shapes and dtypes.

        """
        self._states = torch.empty(
            (self.capacity, *batch.states.shape[1:]),
            dtype=batch.states.dtype,
            device=self.storage_device,
        )
        self._action_targets = torch.empty(
            (self.capacity,),
            dtype=batch.action_targets.dtype,
            device=self.storage_device,
        )
        self._value_targets = torch.empty(
            (self.capacity,),
            dtype=batch.value_targets.dtype,
            device=self.storage_device,
        )
        self._weights = torch.empty(
            (self.capacity,),
            dtype=batch.weights.dtype,
            device=self.storage_device,
        )
