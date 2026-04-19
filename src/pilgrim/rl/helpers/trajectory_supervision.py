# Builds reverse-trajectory supervision batches for dual-head pretraining.
"""Helpers for reverse-trajectory supervised pretraining on Cayley graphs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from cayleypy import CayleyGraph

from ...schemas.rl import TDRandomWalkSamplingConfig
from ..config import RandomWalkSamplingConfig
from ..sampling import resolve_rw_schedule
from .q_learning import (
    _graph_device,
    _resolve_graph_action_indices,
    _resolve_graph_inverse_map,
    _sample_random_walk_actions,
    apply_actions,
)

_EXPECTED_STATE_NDIM = 2


def _normalize_states(states: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """
    Normalize state tensors to shape ``(batch, state_size)``.

    Args:
        states: Input state tensor.
        device: Device used for the normalized tensor.

    Returns:
        Two-dimensional contiguous ``torch.long`` tensor on ``device``.

    Raises:
        ValueError: If ``states`` cannot be normalized to rank ``2``.

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


def _normalize_vector(
    values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """
    Normalize a one-dimensional tensor aligned with state rows.

    Args:
        values: Input tensor-like values.
        device: Device used for the normalized tensor.
        dtype: Target dtype for the returned tensor.
        name: Field name used in validation errors.

    Returns:
        One-dimensional contiguous tensor on ``device``.

    Raises:
        ValueError: If ``values`` cannot be normalized to rank ``1``.

    """
    data = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
    if data.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {tuple(data.shape)}.")
    return data.contiguous()


@dataclass(slots=True, frozen=True)
class ReverseTrajectorySupervisionBatch:
    """
    Aligned supervision rows derived from center-out random walks.

    Args:
        states: States sampled away from the center.
        return_steps: Known number of steps back to the center along the
            reversed sampled trajectory.
        action_targets: One-step action targets that move toward the center
            along the sampled trajectory.
        next_states: States reached after applying ``action_targets``.

    """

    states: torch.Tensor
    return_steps: torch.Tensor
    action_targets: torch.Tensor
    next_states: torch.Tensor

    def __post_init__(self) -> None:
        """
        Validate tensor shapes and alignment.

        Raises:
            ValueError: If one of the batch tensors is not row-aligned.

        """
        state_device = torch.as_tensor(self.states).device
        batch_states = _normalize_states(self.states, device=state_device)
        batch_return_steps = _normalize_vector(
            self.return_steps,
            device=state_device,
            dtype=torch.long,
            name="return_steps",
        )
        batch_action_targets = _normalize_vector(
            self.action_targets,
            device=state_device,
            dtype=torch.long,
            name="action_targets",
        )
        batch_next_states = _normalize_states(self.next_states, device=state_device)

        batch_size = int(batch_states.shape[0])
        if int(batch_return_steps.shape[0]) != batch_size:
            raise ValueError("return_steps must align with states.")
        if int(batch_action_targets.shape[0]) != batch_size:
            raise ValueError("action_targets must align with states.")
        if int(batch_next_states.shape[0]) != batch_size:
            raise ValueError("next_states must align with states.")

        object.__setattr__(self, "states", batch_states)
        object.__setattr__(self, "return_steps", batch_return_steps)
        object.__setattr__(self, "action_targets", batch_action_targets)
        object.__setattr__(self, "next_states", batch_next_states)

    def __len__(self) -> int:
        """
        Return the batch size.

        Returns:
            Number of aligned supervision rows.

        """
        return int(self.states.shape[0])

    def to(self, device: str | torch.device) -> ReverseTrajectorySupervisionBatch:
        """
        Move all tensors to a target device.

        Args:
            device: Destination device.

        Returns:
            Batch stored on ``device``.

        """
        target_device = torch.device(device)
        return ReverseTrajectorySupervisionBatch(
            states=self.states.to(target_device),
            return_steps=self.return_steps.to(target_device),
            action_targets=self.action_targets.to(target_device),
            next_states=self.next_states.to(target_device),
        )

    def index_select(self, indices: torch.Tensor) -> ReverseTrajectorySupervisionBatch:
        """
        Select a subset of supervision rows.

        Args:
            indices: One-dimensional integer row indices.

        Returns:
            New batch containing the selected rows.

        """
        rows = torch.as_tensor(indices, device=self.states.device).long().reshape(-1)
        return ReverseTrajectorySupervisionBatch(
            states=self.states.index_select(0, rows),
            return_steps=self.return_steps.index_select(0, rows),
            action_targets=self.action_targets.index_select(0, rows),
            next_states=self.next_states.index_select(0, rows),
        )

    def masked_select(self, mask: torch.Tensor) -> ReverseTrajectorySupervisionBatch:
        """
        Select a subset of supervision rows by boolean mask.

        Args:
            mask: Boolean mask aligned with the batch dimension.

        Returns:
            New batch containing the masked rows.

        Raises:
            ValueError: If ``mask`` does not align with the batch size.

        """
        keep = torch.as_tensor(mask, device=self.states.device).bool().reshape(-1)
        if int(keep.shape[0]) != len(self):
            raise ValueError("mask must align with the batch size.")
        indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
        return self.index_select(indices)


def concatenate_reverse_trajectory_batches(
    batches: list[ReverseTrajectorySupervisionBatch],
) -> ReverseTrajectorySupervisionBatch:
    """
    Concatenate several reverse-trajectory supervision batches.

    Args:
        batches: Non-empty list of aligned supervision batches.

    Returns:
        Concatenated supervision batch.

    Raises:
        ValueError: If ``batches`` is empty.

    """
    if not batches:
        raise ValueError("at least one supervision batch is required.")
    if len(batches) == 1:
        return batches[0]
    return ReverseTrajectorySupervisionBatch(
        states=torch.cat([batch.states for batch in batches], dim=0),
        return_steps=torch.cat([batch.return_steps for batch in batches], dim=0),
        action_targets=torch.cat([batch.action_targets for batch in batches], dim=0),
        next_states=torch.cat([batch.next_states for batch in batches], dim=0),
    )


def subsample_reverse_trajectory_batch(
    batch: ReverseTrajectorySupervisionBatch,
    *,
    max_states: int,
    seed: int | None = None,
) -> ReverseTrajectorySupervisionBatch:
    """
    Randomly subsample supervision rows without replacement.

    Args:
        batch: Reverse-trajectory supervision batch to subsample.
        max_states: Maximum number of rows to keep.
        seed: Optional deterministic seed for the row permutation.

    Returns:
        Original batch when ``len(batch) <= max_states``; otherwise a random
        subset with ``max_states`` rows.

    Raises:
        ValueError: If ``max_states`` is not positive.

    """
    if int(max_states) <= 0:
        raise ValueError("max_states must be positive.")
    if len(batch) <= int(max_states):
        return batch
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    indices = torch.randperm(len(batch), generator=generator)[: int(max_states)]
    return batch.index_select(indices.to(batch.states.device))


def sample_reverse_trajectory_supervision_from_random_walks(  # noqa: PLR0914
    graph: CayleyGraph,
    config: RandomWalkSamplingConfig | TDRandomWalkSamplingConfig,
    *,
    generator_indices: Sequence[int] | None = None,
    sample_index: int = 0,
) -> ReverseTrajectorySupervisionBatch:
    """
    Build supervised return-to-center targets from explicit random walks.

    Args:
        graph: Cayley graph used to generate trajectories.
        config: Random-walk sampling configuration.
        generator_indices: Optional subset of outward actions allowed during
            trajectory generation.
        sample_index: Sampling-call index used to derive a deterministic seed.

    Returns:
        Concatenated supervision batch produced by the configured walk schedule.

    Raises:
        RuntimeError: If the graph does not expose generator inverses or no
            valid supervision rows are produced.

    """
    graph_device = _graph_device(graph)
    inverse_map = _resolve_graph_inverse_map(graph, device=graph_device)
    if inverse_map is None:
        raise RuntimeError(
            "reverse-trajectory supervision requires an inverse-closed generator map."
        )

    seed = int(config.seed) + int(sample_index) * 100_003
    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(seed)
    allowed_actions = _resolve_graph_action_indices(
        graph,
        generator_indices=generator_indices,
        device=graph_device,
    )
    allowed_membership = torch.zeros(
        int(inverse_map.shape[0]),
        device=graph_device,
        dtype=torch.bool,
    )
    allowed_membership[allowed_actions] = True
    center_state = torch.as_tensor(graph.central_state, device=graph_device).long()

    batches: list[ReverseTrajectorySupervisionBatch] = []
    for factor, length in resolve_rw_schedule(config):
        walk_length = int(length)
        if walk_length <= 0:
            continue
        width = max(1, int(int(config.rw_width) * float(factor)))
        current_states = center_state.view(1, -1).expand(width, -1).clone()
        previous_actions: torch.Tensor | None = None

        states_rows: list[torch.Tensor] = []
        return_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        next_state_rows: list[torch.Tensor] = []

        for step_index in range(1, walk_length + 1):
            outward_actions = _sample_random_walk_actions(
                allowed_actions=allowed_actions,
                batch_size=width,
                previous_actions=previous_actions,
                inverse_map=inverse_map,
                rw_mode=str(config.rw_mode),
                generator=action_generator,
                output_device=graph_device,
            )
            next_states = apply_actions(graph, current_states, outward_actions)
            inward_actions = inverse_map.index_select(0, outward_actions.long())
            valid_mask = inward_actions >= 0
            valid_mask &= allowed_membership.index_select(
                0, inward_actions.clamp_min(0)
            )
            if bool(valid_mask.any()):
                count = int(valid_mask.sum().item())
                states_rows.append(next_states[valid_mask].clone())
                return_rows.append(
                    torch.full(
                        (count,),
                        fill_value=int(step_index),
                        dtype=torch.long,
                        device=graph_device,
                    )
                )
                action_rows.append(inward_actions[valid_mask].clone())
                next_state_rows.append(current_states[valid_mask].clone())
            current_states = next_states
            previous_actions = outward_actions

        if states_rows:
            batches.append(
                ReverseTrajectorySupervisionBatch(
                    states=torch.cat(states_rows, dim=0),
                    return_steps=torch.cat(return_rows, dim=0),
                    action_targets=torch.cat(action_rows, dim=0),
                    next_states=torch.cat(next_state_rows, dim=0),
                )
            )

    if not batches:
        raise RuntimeError("random-walk supervision sampling produced no valid rows.")
    return concatenate_reverse_trajectory_batches(batches)
