# Integrates cayleypy beam search with policy/value supervision generation.
"""Beam-search guidance helpers for search-guided PPO."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from cayleypy import CayleyGraph, Predictor
from torch import nn

from ...schemas.rl.search_guided_ppo import SearchGuidedPPOBeamSearchConfig
from ..supervision_archive import PolicySupervisionBatch
from .ppo import forward_policy_value

_ZERO_EPS = 1e-12


@dataclass(slots=True, frozen=True)
class BeamSearchTargetSet:
    """
    Beam-search supervision aligned with a source-state subset.

    Args:
        batch: Policy/value supervision rows derived from beam search.
        state_indices: Row indices inside the source state batch.
        path_lengths: Solved path lengths returned by beam search.
        best_widths: Beam widths that produced the retained paths.

    """

    batch: PolicySupervisionBatch
    state_indices: torch.Tensor
    path_lengths: torch.Tensor
    best_widths: torch.Tensor


@dataclass(slots=True, frozen=True)
class BeamSearchTargetStats:
    """
    Summary of one beam-guidance collection pass.

    Args:
        queried: Number of beam-search calls issued.
        path_found: Number of calls that found a path.
        rows_added: Number of supervision rows produced.
        mean_path_length: Mean path length across found paths, if any.

    """

    queried: int
    path_found: int
    rows_added: int
    mean_path_length: float | None


class _AuxValuePredictor(nn.Module):
    """Thin wrapper exposing the actor-critic value head to `cayleypy`."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Predict scalar values for beam-search candidate ranking.

        Args:
            states: Decoded graph states.

        Returns:
            One-dimensional value tensor.

        """
        return forward_policy_value(self.model, states).values


def collect_beam_search_targets(  # noqa: PLR0912, PLR0914, PLR0915
    graph: CayleyGraph,
    model: nn.Module,
    states: torch.Tensor,
    config: SearchGuidedPPOBeamSearchConfig,
    *,
    limit: int | None = None,
) -> tuple[BeamSearchTargetSet | None, BeamSearchTargetStats]:
    """
    Run beam search from a subset of states and build supervision targets.

    Args:
        graph: Cayley graph used for search.
        model: Actor-critic model whose value head ranks beam candidates.
        states: Candidate source states.
        config: Beam-search guidance configuration.
        limit: Optional cap on the number of states searched from ``states``.

    Returns:
        Tuple ``(targets, stats)`` where ``targets`` is ``None`` when no beam
        run produced a non-empty path.

    """
    source_states = torch.as_tensor(states, device=graph.device).long()
    if source_states.ndim == 1:
        source_states = source_states.unsqueeze(0)
    if int(source_states.shape[0]) == 0:
        return None, BeamSearchTargetStats(0, 0, 0, None)

    target_count = (
        int(source_states.shape[0])
        if limit is None
        else min(int(limit), int(source_states.shape[0]))
    )
    candidate_indices = torch.arange(target_count, device=source_states.device)

    was_training = model.training
    predictor_model = _AuxValuePredictor(model)
    predictor_model.eval()
    predictor = Predictor(graph, predictor_model)
    amp_ctx = _beam_autocast_context(graph, config)

    kept_states: list[torch.Tensor] = []
    action_targets: list[int] = []
    value_targets: list[float] = []
    weights: list[float] = []
    state_indices: list[int] = []
    path_lengths: list[int] = []
    best_widths: list[int] = []

    queried = 0
    found = 0
    try:
        with torch.inference_mode(), amp_ctx:
            center_state = (
                torch
                .as_tensor(
                    graph.central_state,
                    device=graph.device,
                )
                .reshape(-1)
                .long()
            )
            for source_index in candidate_indices.tolist():
                state = source_states[source_index]
                if bool(torch.equal(state, center_state)):
                    continue
                queried += 1
                best_result: Any | None = None
                best_width: int | None = None
                for beam_width in config.beam_widths:
                    graph.free_memory()
                    try:
                        result = graph.beam_search(
                            start_state=state.detach().cpu().tolist(),
                            beam_width=int(beam_width),
                            max_steps=int(config.max_steps),
                            predictor=predictor,
                            history_depth=int(config.history_depth),
                            beam_mode=str(config.beam_mode),
                            return_path=True,
                            verbose=0,
                        )
                    except Exception:
                        continue
                    if not bool(result.path_found):
                        continue
                    if int(result.path_length) <= 0:
                        continue
                    if best_result is None or int(result.path_length) < int(
                        best_result.path_length
                    ):
                        best_result = result
                        best_width = int(beam_width)
                if best_result is None:
                    continue
                found += 1
                path = [int(action) for action in list(best_result.path)]
                if not path:
                    continue
                kept_states.append(state.detach().cpu().view(1, -1))
                action_targets.append(int(path[0]))
                value_targets.append(float(int(best_result.path_length)))
                weights.append(1.0)
                state_indices.append(int(source_index))
                path_lengths.append(int(best_result.path_length))
                best_widths.append(int(best_width if best_width is not None else 0))
    finally:
        model.train(was_training)

    if not kept_states:
        return None, BeamSearchTargetStats(queried, found, 0, None)

    batch = PolicySupervisionBatch(
        states=torch.cat(kept_states, dim=0),
        action_targets=torch.tensor(action_targets, dtype=torch.long),
        value_targets=torch.tensor(value_targets, dtype=torch.float32),
        weights=torch.tensor(weights, dtype=torch.float32),
    )
    return (
        BeamSearchTargetSet(
            batch=batch,
            state_indices=torch.tensor(state_indices, dtype=torch.long),
            path_lengths=torch.tensor(path_lengths, dtype=torch.long),
            best_widths=torch.tensor(best_widths, dtype=torch.long),
        ),
        BeamSearchTargetStats(
            queried=queried,
            path_found=found,
            rows_added=len(batch),
            mean_path_length=float(
                torch.tensor(path_lengths, dtype=torch.float32).mean().item()
            ),
        ),
    )


def beam_action_reward_bonus(
    *,
    actions: torch.Tensor,
    target_set: BeamSearchTargetSet | None,
    batch_size: int,
    match_bonus: float,
    miss_penalty: float,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Build first-step reward bonuses from beam-found first actions.

    Args:
        actions: Actions taken by the policy on rollout starts.
        target_set: Optional beam-search targets aligned with rollout starts.
        batch_size: Number of rollout environments.
        match_bonus: Bonus added when the action matches the beam target.
        miss_penalty: Penalty added when the action misses the beam target.
        device: Device used by the returned tensor.

    Returns:
        One-dimensional reward-bonus tensor aligned with rollout starts.

    """
    bonus = torch.zeros(int(batch_size), device=device, dtype=torch.float32)
    if target_set is None:
        return bonus
    action_ids = torch.as_tensor(actions, device=device).long().reshape(-1)
    target_indices = target_set.state_indices.to(device)
    target_actions = target_set.batch.action_targets.to(device)
    matches = action_ids.index_select(0, target_indices) == target_actions
    if abs(float(match_bonus)) > _ZERO_EPS:
        bonus[target_indices[matches]] += float(match_bonus)
    if abs(float(miss_penalty)) > _ZERO_EPS:
        bonus[target_indices[~matches]] += float(miss_penalty)
    return bonus


def _beam_autocast_context(
    graph: CayleyGraph,
    config: SearchGuidedPPOBeamSearchConfig,
) -> Any:
    """
    Build the autocast context used by beam-search guidance.

    Args:
        graph: Cayley graph searched by beam search.
        config: Beam-search configuration.

    Returns:
        Context manager enabling autocast when requested.

    """
    graph_device = graph.device
    if not isinstance(graph_device, torch.device):
        graph_device = torch.device(graph_device)
    if graph_device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = _resolve_torch_dtype(config.autocast_dtype_name)
    if not isinstance(dtype, torch.dtype):
        dtype = torch.bfloat16
    if config.enable_autocast is False:
        return nullcontext()
    enabled = (
        bool(graph_device.type == "cuda")
        if config.enable_autocast is None
        else bool(config.enable_autocast)
    )
    if not enabled:
        return nullcontext()
    return torch.autocast(
        device_type=graph_device.type,
        dtype=dtype,
        enabled=True,
    )


def _resolve_torch_dtype(value: Any) -> Any:
    """
    Resolve a serialized dtype name into ``torch.dtype`` when possible.

    Args:
        value: Candidate dtype string.

    Returns:
        Matching ``torch.dtype`` or the original value when unresolved.

    """
    if not isinstance(value, str):
        return value
    normalized = value.removeprefix("torch.").strip().lower()
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return mapping.get(normalized, value)
