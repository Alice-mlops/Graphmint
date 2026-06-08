# Samples certified one-hop recovery rows around solved beam paths.
"""Policy-filtered recovery-neighbor supervision for solved search paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from cayleypy import CayleyGraph
from torch import nn

from .ppo import forward_policy_value
from .q_learning import apply_actions


@dataclass(slots=True, frozen=True)
class RecoveryNeighborRows:
    """
    Supervision rows that return one-hop neighbors to a solved path.

    Args:
        states: Neighbor states produced by stepping off path anchors.
        action_targets: Same generators used to step off path anchors; applying
            them again returns to the certified path because pancake generators
            are involutions.
        value_targets: Upper-bound remaining distances, equal to the path
            anchor remaining distance plus one.
        weights: Per-row supervised-loss weights.
        source_indices: Source indices inherited from the solved root states.
        path_lengths: Remaining upper-bound lengths aligned with rows.
        best_widths: Beam widths inherited from solved paths.
    """

    states: list[torch.Tensor]
    action_targets: list[int]
    value_targets: list[float]
    weights: list[float]
    source_indices: list[int]
    path_lengths: list[int]
    best_widths: list[int]

    def __len__(self) -> int:
        """
        Return the number of recovery rows.

        Returns:
            Row count.
        """
        return len(self.action_targets)


def sample_policy_recovery_neighbor_rows(
    *,
    graph: CayleyGraph,
    model: nn.Module,
    path_states: list[torch.Tensor],
    path_actions: list[int],
    remaining_lengths: list[int],
    source_index: int,
    best_width: int,
    top_k: int,
    row_weight: float,
    neighbor_weight: float,
    max_rows: int | None,
    exclude_path_action: bool,
) -> RecoveryNeighborRows:
    """
    Sample policy-ranked one-hop recovery targets around solved path states.

    The policy head is used only to choose which off-path neighbors are worth
    adding. The target action for each neighbor is certified independently: it
    is the same generator that stepped from the path state to the neighbor, so
    applying it again returns to the path state.

    Args:
        graph: Cayley graph used to apply generator actions.
        model: Actor-critic model whose policy head ranks perturbation actions.
        path_states: Selected path-anchor states with shape ``(1, state_size)``.
        path_actions: Certified next actions for each path-anchor state.
        remaining_lengths: Remaining path lengths for each path-anchor state.
        source_index: Row index of the original solved root state.
        best_width: Beam width that produced the solved path.
        top_k: Number of policy-ranked perturbation actions per anchor.
        row_weight: Base weight of the path-anchor supervision row.
        neighbor_weight: Multiplier for total neighbor mass per anchor.
        max_rows: Optional cap on recovery rows per solved path.
        exclude_path_action: Whether to exclude the certified next-path action
            from perturbation candidates.

    Returns:
        Recovery rows ready to append to a policy/value supervision batch.

    Raises:
        ValueError: If path inputs are not aligned.
    """
    if len(path_states) != len(path_actions) or len(path_states) != len(
        remaining_lengths
    ):
        raise ValueError("path_states, path_actions, and remaining_lengths must align.")
    if int(top_k) <= 0 or float(neighbor_weight) <= 0.0 or not path_states:
        return _empty_rows()

    anchors = torch.cat([state.detach().view(1, -1) for state in path_states], dim=0)
    graph_device = graph.device
    if not isinstance(graph_device, torch.device):
        graph_device = torch.device(graph_device)
    anchors = anchors.to(graph_device).long()
    logits = forward_policy_value(model, anchors).logits.detach()
    action_count = min(int(logits.shape[1]), len(graph.generators))
    if action_count <= 0:
        return _empty_rows()

    output = _empty_rows()
    rows_left = None if max_rows is None else max(0, int(max_rows))
    for anchor_index in range(int(anchors.shape[0])):
        if rows_left is not None and rows_left <= 0:
            break
        scores = logits[anchor_index, :action_count].clone()
        if bool(exclude_path_action):
            path_action = int(path_actions[anchor_index])
            if 0 <= path_action < action_count:
                scores[path_action] = -torch.inf
        available = int(torch.isfinite(scores).sum().item())
        action_limit = min(int(top_k), available)
        if rows_left is not None:
            action_limit = min(action_limit, rows_left)
        if action_limit <= 0:
            continue
        actions = torch.topk(scores, k=action_limit).indices.long()
        repeated = anchors[anchor_index].view(1, -1).repeat(action_limit, 1)
        neighbors = apply_actions(graph, repeated, actions)
        per_row_weight = (
            float(row_weight) * float(neighbor_weight) / float(action_limit)
        )
        output.states.append(neighbors.detach().cpu())
        output.action_targets.extend(int(action) for action in actions.cpu().tolist())
        target_length = int(remaining_lengths[anchor_index]) + 1
        output.value_targets.extend([float(target_length)] * int(action_limit))
        output.weights.extend([per_row_weight] * int(action_limit))
        output.source_indices.extend([int(source_index)] * int(action_limit))
        output.path_lengths.extend([target_length] * int(action_limit))
        output.best_widths.extend([int(best_width)] * int(action_limit))
        if rows_left is not None:
            rows_left -= int(action_limit)
    return output


def _empty_rows() -> RecoveryNeighborRows:
    """
    Build an empty recovery-row container.

    Returns:
        Empty recovery rows.
    """
    return RecoveryNeighborRows(
        states=[],
        action_targets=[],
        value_targets=[],
        weights=[],
        source_indices=[],
        path_lengths=[],
        best_widths=[],
    )
