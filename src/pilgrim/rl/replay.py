# Implements a tensor replay buffer for RL state sampling.
"""Replay-memory utilities for deterministic shortest-path training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_EXPECTED_STATE_NDIM = 2


class TensorReplayBuffer:
    """
    Ring-buffer replay storage for tensor states.

    Args:
        capacity: Maximum number of states stored in the buffer.
        storage_device: Device used for the internal storage tensor.

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
        self._storage: torch.Tensor | None = None
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        """
        Return the number of currently stored states.

        Returns:
            Number of valid entries inside the replay buffer.

        """
        return int(self._size)

    @property
    def is_initialized(self) -> bool:
        """
        Return whether the backing tensor has been allocated.

        Returns:
            ``True`` when the internal storage tensor exists.

        """
        return self._storage is not None

    def add(self, states: torch.Tensor) -> int:
        """
        Append states to replay memory.

        Args:
            states: Tensor of shape ``(batch, state_size)`` or ``(state_size,)``.

        Returns:
            Number of states written into the buffer.

        Raises:
            ValueError: If ``states`` is empty or has rank greater than ``2``.

        """
        data = self._normalize_states(states)
        if data.shape[0] == 0:
            raise ValueError("states must contain at least one row.")

        if self._storage is None:
            self._allocate_storage(data)

        assert self._storage is not None
        write_count = int(data.shape[0])
        if write_count >= self.capacity:
            self._storage.copy_(data[-self.capacity :])
            self._size = self.capacity
            self._next_index = 0
            return self.capacity

        end_index = self._next_index + write_count
        if end_index <= self.capacity:
            self._storage[self._next_index : end_index] = data
        else:
            first_chunk = self.capacity - self._next_index
            second_chunk = write_count - first_chunk
            self._storage[self._next_index :] = data[:first_chunk]
            self._storage[:second_chunk] = data[first_chunk:]

        self._next_index = (self._next_index + write_count) % self.capacity
        self._size = min(self.capacity, self._size + write_count)
        return write_count

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """
        Sample a batch of states uniformly from replay memory.

        Args:
            batch_size: Number of states to sample.
            generator: Optional torch random generator used for sampling.
            device: Optional device for the returned batch.

        Returns:
            Tensor of sampled states.

        Raises:
            RuntimeError: If the buffer is empty.
            ValueError: If ``batch_size`` is not positive.

        """
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if self._storage is None or self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer.")

        sample_size = min(int(batch_size), int(self._size))
        idx = torch.randint(
            low=0,
            high=int(self._size),
            size=(sample_size,),
            generator=generator,
            device=self.storage_device,
        )
        batch = self._storage[idx]
        if device is not None:
            return batch.to(device)
        return batch

    def is_ready(self, min_size: int) -> bool:
        """
        Return whether the buffer holds enough states for training.

        Args:
            min_size: Required minimum number of stored states.

        Returns:
            ``True`` if ``len(self) >= min_size``.

        """
        return len(self) >= int(min_size)

    def state_shape(self) -> tuple[int, ...] | None:
        """
        Return the shape of one stored state.

        Returns:
            State shape excluding the batch dimension, or ``None`` before the
            first write.

        """
        if self._storage is None:
            return None
        return tuple(int(x) for x in self._storage.shape[1:])

    def storage_usage_ratio(self) -> float:
        """
        Return the fill ratio of replay memory.

        Returns:
            Fraction in the closed interval ``[0.0, 1.0]``.

        """
        return float(self._size) / float(self.capacity)

    def _allocate_storage(self, states: torch.Tensor) -> None:
        """
        Allocate the replay tensor using the first observed batch.

        Args:
            states: Normalized state tensor used to define storage shape.

        """
        self._storage = torch.empty(
            (self.capacity, *states.shape[1:]),
            dtype=states.dtype,
            device=self.storage_device,
        )

    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convert input states into the canonical replay tensor layout.

        Args:
            states: Input state tensor.

        Returns:
            Tensor with rank ``2`` stored as ``torch.long`` on the storage
            device.

        Raises:
            ValueError: If the input has rank greater than ``2``.

        """
        data = torch.as_tensor(states, device=self.storage_device).long()
        if data.ndim == 1:
            data = data.unsqueeze(0)
        if data.ndim != _EXPECTED_STATE_NDIM:
            raise ValueError(
                "states must have shape (batch, state_size) or (state_size,), "
                f"got {tuple(data.shape)}."
            )
        return data.contiguous()


@dataclass(slots=True)
class FrontierArchiveUpdateStats:
    """
    Summary of one frontier-archive refresh pass.

    Args:
        candidates: Number of states proposed for archive admission.
        admitted: Number of new states appended into free archive slots.
        updated: Number of already archived states whose scores were updated.
        replaced: Number of lower-score archive slots evicted.
        skipped: Number of candidates skipped because they were weaker than the
            current archive minimum.

    """

    candidates: int
    admitted: int
    updated: int
    replaced: int
    skipped: int


class FrontierStateArchive:
    """
    Tiny deduplicated archive for rare high-value frontier states.

    The archive is intentionally small and score-thresholded. New states are
    admitted only when they outrank the current lowest-score slot. Existing
    states are updated in place using an exponential moving average of their
    self-supervised score.

    Args:
        capacity: Maximum number of states stored in the archive.
        storage_device: Device used for internal archive tensors.

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
        self._storage: torch.Tensor | None = None
        self._hashes = torch.empty((self.capacity,), dtype=torch.int64, device="cpu")
        self._scores = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.storage_device
        )
        self._size = 0
        self._hash_to_index: dict[int, int] = {}

    def __len__(self) -> int:
        """
        Return the number of stored archive states.

        Returns:
            Number of valid archive entries.

        """
        return int(self._size)

    def add_candidates(
        self,
        states: torch.Tensor,
        hashes: torch.Tensor,
        scores: torch.Tensor,
        *,
        score_ema_decay: float,
    ) -> FrontierArchiveUpdateStats:
        """
        Add or update candidate frontier states.

        Args:
            states: Candidate states with shape ``(batch, state_size)``.
            hashes: Hashes corresponding to ``states``.
            scores: Self-supervised scores used for admission and replacement.
            score_ema_decay: Exponential moving-average decay used when a state
                is rediscovered.

        Returns:
            Summary of admissions, updates, replacements, and skipped states.

        Raises:
            ValueError: If input shapes or decay are invalid.

        """
        if not 0.0 <= float(score_ema_decay) <= 1.0:
            raise ValueError("score_ema_decay must be in the closed interval [0, 1].")

        data = self._normalize_states(states)
        hash_values = torch.as_tensor(hashes, device="cpu", dtype=torch.int64).reshape(
            -1
        )
        score_values = torch.as_tensor(
            scores,
            device=self.storage_device,
            dtype=torch.float32,
        ).reshape(-1)
        if (
            data.shape[0] != hash_values.shape[0]
            or data.shape[0] != score_values.shape[0]
        ):
            raise ValueError(
                "states, hashes, and scores must contain the same number of rows."
            )
        if data.shape[0] == 0:
            return FrontierArchiveUpdateStats(
                candidates=0,
                admitted=0,
                updated=0,
                replaced=0,
                skipped=0,
            )

        if self._storage is None:
            self._allocate_storage(data)

        order = torch.argsort(score_values, descending=True)
        admitted = 0
        updated = 0
        replaced = 0
        skipped = 0

        for candidate_index in order.tolist():
            state = data[candidate_index]
            hash_value = int(hash_values[candidate_index].item())
            score_value = float(score_values[candidate_index].item())

            if hash_value in self._hash_to_index:
                slot = self._hash_to_index[hash_value]
                self._storage[slot] = state
                self._scores[slot] = (
                    float(score_ema_decay) * self._scores[slot]
                    + (1.0 - float(score_ema_decay)) * score_values[candidate_index]
                )
                updated += 1
                continue

            if self._size < self.capacity:
                slot = self._size
                self._size += 1
                admitted += 1
                self._write_slot(
                    slot=slot,
                    state=state,
                    hash_value=hash_value,
                    score_value=score_value,
                )
                continue

            weakest_slot = int(torch.argmin(self._scores[: self._size]).item())
            weakest_score = float(self._scores[weakest_slot].item())
            if score_value <= weakest_score:
                skipped += 1
                continue

            old_hash = int(self._hashes[weakest_slot].item())
            self._hash_to_index.pop(old_hash, None)
            self._write_slot(
                slot=weakest_slot,
                state=state,
                hash_value=hash_value,
                score_value=score_value,
            )
            replaced += 1

        return FrontierArchiveUpdateStats(
            candidates=int(data.shape[0]),
            admitted=admitted,
            updated=updated,
            replaced=replaced,
            skipped=skipped,
        )

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """
        Sample archive states without replacement.

        Args:
            batch_size: Number of archive states to sample.
            generator: Optional torch random generator used for sampling.
            device: Optional device for the returned batch.

        Returns:
            Sampled frontier states.

        Raises:
            RuntimeError: If the archive is empty.
            ValueError: If ``batch_size`` is not positive.

        """
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if self._storage is None or self._size == 0:
            raise RuntimeError("cannot sample from an empty frontier archive.")

        sample_size = min(int(batch_size), int(self._size))
        indices = torch.randperm(
            int(self._size),
            generator=generator,
            device=self.storage_device,
        )[:sample_size]
        batch = self._storage[indices]
        if device is not None:
            return batch.to(device)
        return batch

    def storage_usage_ratio(self) -> float:
        """
        Return the fill ratio of the archive.

        Returns:
            Fraction in the closed interval ``[0.0, 1.0]``.

        """
        return float(self._size) / float(self.capacity)

    def score_statistics(self) -> tuple[float | None, float | None]:
        """
        Return mean and max archive scores.

        Returns:
            Tuple ``(mean_score, max_score)`` or ``(None, None)`` when empty.

        """
        if self._size == 0:
            return None, None
        scores = self._scores[: self._size]
        return float(scores.mean().item()), float(scores.max().item())

    def _allocate_storage(self, states: torch.Tensor) -> None:
        """
        Allocate the state-storage tensor from an example batch.

        Args:
            states: Example states used to infer the archive shape.

        """
        self._storage = torch.empty(
            (self.capacity, *states.shape[1:]),
            dtype=states.dtype,
            device=self.storage_device,
        )

    def _write_slot(
        self,
        *,
        slot: int,
        state: torch.Tensor,
        hash_value: int,
        score_value: float,
    ) -> None:
        """
        Write one archive slot and update the dedup map.

        Args:
            slot: Archive slot index to write.
            state: State tensor written into the slot.
            hash_value: State hash used for archive deduplication.
            score_value: Archive score associated with the state.

        """
        assert self._storage is not None
        self._storage[slot] = state
        self._hashes[slot] = int(hash_value)
        self._scores[slot] = float(score_value)
        self._hash_to_index[int(hash_value)] = int(slot)

    def _normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convert input states into the canonical archive layout.

        Args:
            states: Input state tensor.

        Returns:
            Tensor with rank ``2`` stored as ``torch.long`` on the archive
            storage device.

        Raises:
            ValueError: If the input has rank greater than ``2``.

        """
        data = torch.as_tensor(states, device=self.storage_device).long()
        if data.ndim == 1:
            data = data.unsqueeze(0)
        if data.ndim != _EXPECTED_STATE_NDIM:
            raise ValueError(
                "states must have shape (batch, state_size) or (state_size,), "
                f"got {tuple(data.shape)}."
            )
        return data.contiguous()


def replay_batches_per_capacity(
    *,
    capacity: int,
    batch_size: int,
) -> int:
    """
    Return the number of disjoint batches that fit in replay capacity.

    Args:
        capacity: Replay-buffer capacity.
        batch_size: Desired batch size.

    Returns:
        Ceiling of ``capacity / batch_size``.

    Raises:
        ValueError: If ``capacity`` or ``batch_size`` are not positive.

    """
    if int(capacity) <= 0:
        raise ValueError("capacity must be positive.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    return math.ceil(float(capacity) / float(batch_size))
