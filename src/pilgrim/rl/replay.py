# Implements replay buffers for deterministic shortest-path training.
"""Replay-memory utilities for deterministic shortest-path training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_EXPECTED_STATE_NDIM = 2


def _normalize_state_rows(
    states: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert input states into the canonical replay tensor layout.

    Args:
        states: Input state tensor.
        device: Target device for the normalized tensor.

    Returns:
        Tensor with rank ``2`` stored as ``torch.long`` on ``device``.

    Raises:
        ValueError: If the input has rank greater than ``2``.

    """
    data = torch.as_tensor(states, device=device).long()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.ndim != _EXPECTED_STATE_NDIM:
        raise ValueError(
            "states must have shape (batch, state_size) or (state_size,), "
            f"got {tuple(data.shape)}."
        )
    return data.contiguous()


def _normalize_1d_tensor(
    values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """
    Normalize a one-dimensional replay-aligned tensor.

    Args:
        values: Input tensor-like object.
        device: Target device for the normalized tensor.
        dtype: Target dtype for the returned tensor.
        name: Field name used in validation errors.

    Returns:
        One-dimensional contiguous tensor on ``device``.

    Raises:
        ValueError: If the tensor cannot be normalized to rank ``1``.

    """
    data = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
    if data.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {tuple(data.shape)}.")
    return data.contiguous()


def _sample_indices(
    *,
    size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample replay row indices uniformly without replacement guarantees.

    Args:
        size: Number of valid replay rows.
        batch_size: Requested batch size.
        device: Device used for the sampled index tensor.
        generator: Optional RNG used for sampling.

    Returns:
        One-dimensional tensor of sampled replay indices.

    Raises:
        ValueError: If ``batch_size`` is not positive.
        RuntimeError: If ``size`` is zero.

    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if int(size) <= 0:
        raise RuntimeError("cannot sample from an empty replay buffer.")
    sample_size = min(int(batch_size), int(size))
    return torch.randint(
        low=0,
        high=int(size),
        size=(sample_size,),
        generator=generator,
        device=device,
    )


def _write_ring_tensor(
    storage: torch.Tensor,
    data: torch.Tensor,
    *,
    next_index: int,
) -> tuple[int, int]:
    """
    Write one aligned batch into a ring-buffer tensor.

    Args:
        storage: Preallocated ring-buffer tensor.
        data: Batch of rows to append.
        next_index: Current write cursor inside ``storage``.

    Returns:
        Tuple ``(written_rows, next_index_after_write)``.

    """
    write_count = int(data.shape[0])
    capacity = int(storage.shape[0])
    if write_count >= capacity:
        storage.copy_(data[-capacity:])
        return capacity, 0

    end_index = int(next_index) + write_count
    if end_index <= capacity:
        storage[int(next_index) : end_index] = data
    else:
        first_chunk = capacity - int(next_index)
        second_chunk = write_count - first_chunk
        storage[int(next_index) :] = data[:first_chunk]
        storage[:second_chunk] = data[first_chunk:]
    return write_count, (int(next_index) + write_count) % capacity


@dataclass(slots=True, frozen=True)
class TransitionBatch:
    """
    Aligned batch of deterministic n-step transitions.

    Args:
        states: Source states with shape ``(batch, state_size)``.
        actions: First-step actions with shape ``(batch,)``.
        next_states: States reached after ``steps`` transitions.
        steps: Number of transitions collapsed into each row.
        done: Whether the rollout reached the terminal center state.

    """

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    steps: torch.Tensor
    done: torch.Tensor

    def __post_init__(self) -> None:
        """
        Validate tensor shapes and batch alignment.

        Raises:
            ValueError: If one of the tensors is not batch-aligned.

        """
        state_device = torch.as_tensor(self.states).device
        batch_states = _normalize_state_rows(self.states, device=state_device)
        batch_next_states = _normalize_state_rows(
            self.next_states,
            device=state_device,
        )
        batch_actions = _normalize_1d_tensor(
            self.actions,
            device=state_device,
            dtype=torch.long,
            name="actions",
        )
        batch_steps = _normalize_1d_tensor(
            self.steps,
            device=state_device,
            dtype=torch.long,
            name="steps",
        )
        batch_done = _normalize_1d_tensor(
            self.done,
            device=state_device,
            dtype=torch.bool,
            name="done",
        )
        batch_size = int(batch_states.shape[0])
        if int(batch_next_states.shape[0]) != batch_size:
            raise ValueError("next_states must align with states.")
        if int(batch_actions.shape[0]) != batch_size:
            raise ValueError("actions must align with states.")
        if int(batch_steps.shape[0]) != batch_size:
            raise ValueError("steps must align with states.")
        if int(batch_done.shape[0]) != batch_size:
            raise ValueError("done must align with states.")
        object.__setattr__(self, "states", batch_states)
        object.__setattr__(self, "actions", batch_actions)
        object.__setattr__(self, "next_states", batch_next_states)
        object.__setattr__(self, "steps", batch_steps)
        object.__setattr__(self, "done", batch_done)

    def __len__(self) -> int:
        """
        Return the batch size.

        Returns:
            Number of aligned transition rows.

        """
        return int(self.states.shape[0])

    def to(self, device: str | torch.device) -> TransitionBatch:
        """
        Move all batch tensors to a target device.

        Args:
            device: Destination device.

        Returns:
            Transition batch on ``device``.

        """
        target_device = torch.device(device)
        return TransitionBatch(
            states=self.states.to(target_device),
            actions=self.actions.to(target_device),
            next_states=self.next_states.to(target_device),
            steps=self.steps.to(target_device),
            done=self.done.to(target_device),
        )

    def index_select(self, indices: torch.Tensor) -> TransitionBatch:
        """
        Select a subset of transition rows.

        Args:
            indices: One-dimensional integer row indices.

        Returns:
            New transition batch containing the selected rows.

        """
        rows = torch.as_tensor(indices, device=self.states.device).long().reshape(-1)
        return TransitionBatch(
            states=self.states.index_select(0, rows),
            actions=self.actions.index_select(0, rows),
            next_states=self.next_states.index_select(0, rows),
            steps=self.steps.index_select(0, rows),
            done=self.done.index_select(0, rows),
        )


def concatenate_transition_batches(
    batches: list[TransitionBatch],
) -> TransitionBatch:
    """
    Concatenate multiple aligned transition batches.

    Args:
        batches: Non-empty list of transition batches.

    Returns:
        Concatenated transition batch.

    Raises:
        ValueError: If ``batches`` is empty.

    """
    if not batches:
        raise ValueError("at least one transition batch is required.")
    if len(batches) == 1:
        return batches[0]
    return TransitionBatch(
        states=torch.cat([batch.states for batch in batches], dim=0),
        actions=torch.cat([batch.actions for batch in batches], dim=0),
        next_states=torch.cat([batch.next_states for batch in batches], dim=0),
        steps=torch.cat([batch.steps for batch in batches], dim=0),
        done=torch.cat([batch.done for batch in batches], dim=0),
    )


def subsample_transition_batch(
    batch: TransitionBatch,
    *,
    max_transitions: int,
    seed: int | None = None,
) -> TransitionBatch:
    """
    Randomly subsample transition rows without replacement.

    Args:
        batch: Transition batch to subsample.
        max_transitions: Maximum number of rows to keep.
        seed: Optional deterministic seed for the row permutation.

    Returns:
        Original batch when ``len(batch) <= max_transitions``; otherwise a
        randomly selected subset.

    Raises:
        ValueError: If ``max_transitions`` is not positive.

    """
    if int(max_transitions) <= 0:
        raise ValueError("max_transitions must be positive.")
    if len(batch) <= int(max_transitions):
        return batch
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    indices = torch.randperm(len(batch), generator=generator)[: int(max_transitions)]
    return batch.index_select(indices.to(batch.states.device))


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

        idx = _sample_indices(
            size=int(self._size),
            batch_size=int(batch_size),
            device=self.storage_device,
            generator=generator,
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

        """
        return _normalize_state_rows(states, device=self.storage_device)


class TransitionReplayBuffer:
    """
    Ring-buffer replay storage for aligned transition batches.

    Args:
        capacity: Maximum number of transitions stored in the buffer.
        storage_device: Device used for the internal storage tensors.

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
        self._actions: torch.Tensor | None = None
        self._next_states: torch.Tensor | None = None
        self._steps: torch.Tensor | None = None
        self._done: torch.Tensor | None = None
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        """
        Return the number of currently stored transitions.

        Returns:
            Number of valid rows inside the transition replay buffer.

        """
        return int(self._size)

    @property
    def is_initialized(self) -> bool:
        """
        Return whether the backing tensors have been allocated.

        Returns:
            ``True`` when the internal storage tensors exist.

        """
        return self._states is not None

    def add(self, batch: TransitionBatch) -> int:
        """
        Append transitions to replay memory.

        Args:
            batch: Transition batch to append.

        Returns:
            Number of transition rows written into the buffer.

        Raises:
            ValueError: If ``batch`` is empty.

        """
        transitions = batch.to(self.storage_device)
        if len(transitions) == 0:
            raise ValueError("transition batch must contain at least one row.")
        if self._states is None:
            self._allocate_storage(transitions)

        assert self._states is not None
        assert self._actions is not None
        assert self._next_states is not None
        assert self._steps is not None
        assert self._done is not None

        written, next_index = _write_ring_tensor(
            self._states,
            transitions.states,
            next_index=self._next_index,
        )
        _write_ring_tensor(
            self._actions,
            transitions.actions,
            next_index=self._next_index,
        )
        _write_ring_tensor(
            self._next_states,
            transitions.next_states,
            next_index=self._next_index,
        )
        _write_ring_tensor(
            self._steps,
            transitions.steps,
            next_index=self._next_index,
        )
        _write_ring_tensor(
            self._done,
            transitions.done,
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
    ) -> TransitionBatch:
        """
        Sample a batch of transitions uniformly from replay memory.

        Args:
            batch_size: Number of transitions to sample.
            generator: Optional torch random generator used for sampling.
            device: Optional device for the returned batch.

        Returns:
            Sampled transition batch.

        Raises:
            RuntimeError: If the buffer is empty.
            ValueError: If ``batch_size`` is not positive.

        """
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if self._states is None or self._size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer.")
        idx = _sample_indices(
            size=int(self._size),
            batch_size=int(batch_size),
            device=self.storage_device,
            generator=generator,
        )
        batch = TransitionBatch(
            states=self._states[idx],
            actions=self._actions[idx],
            next_states=self._next_states[idx],
            steps=self._steps[idx],
            done=self._done[idx],
        )
        if device is not None:
            return batch.to(device)
        return batch

    def is_ready(self, min_size: int) -> bool:
        """
        Return whether the buffer holds enough transitions for training.

        Args:
            min_size: Required minimum number of stored transitions.

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
        if self._states is None:
            return None
        return tuple(int(x) for x in self._states.shape[1:])

    def storage_usage_ratio(self) -> float:
        """
        Return the fill ratio of replay memory.

        Returns:
            Fraction in the closed interval ``[0.0, 1.0]``.

        """
        return float(self._size) / float(self.capacity)

    def _allocate_storage(self, batch: TransitionBatch) -> None:
        """
        Allocate replay tensors using the first observed transition batch.

        Args:
            batch: Normalized transition batch used to define storage shapes.

        """
        self._states = torch.empty(
            (self.capacity, *batch.states.shape[1:]),
            dtype=batch.states.dtype,
            device=self.storage_device,
        )
        self._actions = torch.empty(
            (self.capacity,),
            dtype=batch.actions.dtype,
            device=self.storage_device,
        )
        self._next_states = torch.empty(
            (self.capacity, *batch.next_states.shape[1:]),
            dtype=batch.next_states.dtype,
            device=self.storage_device,
        )
        self._steps = torch.empty(
            (self.capacity,),
            dtype=batch.steps.dtype,
            device=self.storage_device,
        )
        self._done = torch.empty(
            (self.capacity,),
            dtype=batch.done.dtype,
            device=self.storage_device,
        )


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


@dataclass(slots=True, frozen=True)
class FrontierArchiveSnapshot:
    """
    Exact archive contents used to synchronize ranks.

    Args:
        states: Archived state rows with shape ``(size, state_size)``.
        hashes: CPU hash tensor aligned with ``states``.
        scores: Floating-point score tensor aligned with ``states``.

    """

    states: torch.Tensor
    hashes: torch.Tensor
    scores: torch.Tensor


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

    def snapshot(self) -> FrontierArchiveSnapshot:
        """
        Return an exact snapshot of the current archive contents.

        Returns:
            Immutable archive snapshot suitable for cross-rank synchronization.

        """
        if self._size == 0:
            if self._storage is None:
                states = torch.empty(
                    (0, 0), dtype=torch.long, device=self.storage_device
                )
            else:
                states = self._storage[:0].clone()
            return FrontierArchiveSnapshot(
                states=states,
                hashes=self._hashes[:0].clone(),
                scores=self._scores[:0].clone(),
            )

        assert self._storage is not None
        return FrontierArchiveSnapshot(
            states=self._storage[: self._size].clone(),
            hashes=self._hashes[: self._size].clone(),
            scores=self._scores[: self._size].clone(),
        )

    def load_snapshot(self, snapshot: FrontierArchiveSnapshot) -> None:
        """
        Replace the archive contents with a synchronized snapshot.

        Args:
            snapshot: Exact archive state materialized on another rank.

        Raises:
            ValueError: If the snapshot tensors are inconsistent.

        """
        states = torch.as_tensor(snapshot.states, device=self.storage_device).long()
        hashes = torch.as_tensor(
            snapshot.hashes, device="cpu", dtype=torch.int64
        ).reshape(-1)
        scores = torch.as_tensor(
            snapshot.scores,
            device=self.storage_device,
            dtype=torch.float32,
        ).reshape(-1)
        if states.ndim != _EXPECTED_STATE_NDIM:
            raise ValueError(
                "snapshot.states must have shape (batch, state_size), "
                f"got {tuple(states.shape)}."
            )
        if states.shape[0] != hashes.shape[0] or states.shape[0] != scores.shape[0]:
            raise ValueError(
                "snapshot states, hashes, and scores must contain the same number of rows."
            )
        if states.shape[0] > self.capacity:
            raise ValueError("snapshot size cannot exceed archive capacity.")

        if states.shape[0] > 0:
            if self._storage is None or tuple(self._storage.shape[1:]) != tuple(
                states.shape[1:]
            ):
                self._allocate_storage(states)
            assert self._storage is not None
            self._storage[: states.shape[0]] = states
        self._size = int(states.shape[0])
        self._hash_to_index.clear()
        if self._size == 0:
            return
        self._hashes[: self._size] = hashes
        self._scores[: self._size] = scores
        for index, hash_value in enumerate(hashes.tolist()):
            self._hash_to_index[int(hash_value)] = int(index)

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
