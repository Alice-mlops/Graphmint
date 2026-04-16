# Declares reusable helpers for outward beam-expansion search.
"""Reusable helpers for outward beam-expansion search."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from cayleypy import Predictor

from pilgrim.schemas.beam_search import (
    OutwardExpansionCandidate,
    OutwardExpansionConfig,
    OutwardExpansionResult,
)
from pilgrim.utils.eval.states import state_to_tuple


@dataclass(slots=True)
class _CandidateRecord:
    """Internal candidate payload stored while the search is running."""

    state_hash: int
    depth: int
    state: tuple[int, ...]
    score: float


def _build_predictor(graph: Any, predictor: Any) -> Predictor:
    """
    Normalize a predictor-like object to ``cayleypy.Predictor``.

    Args:
        graph: Graph instance associated with the predictor.
        predictor: Predictor, model, callable, or heuristic identifier.

    Returns:
        Normalized predictor wrapper.

    """
    if isinstance(predictor, Predictor):
        return predictor
    return Predictor(graph, predictor)


def _ensure_rank_two(states: torch.Tensor) -> torch.Tensor:
    """
    Normalize a state tensor to rank two.

    Args:
        states: State tensor in internal graph representation.

    Returns:
        Two-dimensional state tensor.

    """
    if states.dim() == 1:
        return states.unsqueeze(0)
    if states.dim() > 2:
        return states.flatten(end_dim=1)
    return states


def _score_states(
    graph: Any,
    predictor: Predictor,
    states: torch.Tensor,
) -> torch.Tensor:
    """
    Score encoded graph states with a predictor.

    Args:
        graph: Graph instance that can decode states.
        predictor: Normalized predictor wrapper.
        states: Encoded states in internal graph representation.

    Returns:
        One-dimensional floating-point score tensor on the graph device.

    """
    decoded_states = graph.decode_states(states)
    scores = predictor(decoded_states)
    return (
        torch.as_tensor(scores, device=graph.device)
        .reshape(-1)
        .float()
        .nan_to_num(nan=-float("inf"))
    )


def _topk_indices(scores: torch.Tensor, limit: int) -> torch.Tensor:
    """
    Select indices of the largest scores.

    Args:
        scores: One-dimensional score tensor.
        limit: Number of states to keep.

    Returns:
        Indices of the selected states.

    """
    if scores.numel() <= limit:
        return torch.arange(scores.numel(), device=scores.device)
    _, idx = torch.topk(scores, k=limit, largest=True, sorted=True)
    return idx


def _mask_unseen_hashes(
    hashes: torch.Tensor,
    history_hash_layers: Sequence[torch.Tensor],
) -> torch.Tensor:
    """
    Build a mask for hashes unseen in the retained history layers.

    Args:
        hashes: Candidate hashes for the current expansion step.
        history_hash_layers: Recent raw hash layers to ban from revisiting.

    Returns:
        Boolean mask where ``True`` marks a hash not present in recent history.

    """
    if len(history_hash_layers) == 0:
        return torch.ones_like(hashes, dtype=torch.bool)

    mask = torch.ones_like(hashes, dtype=torch.bool)
    for layer in history_hash_layers:
        if layer.numel() == 0:
            continue
        mask &= ~torch.isin(hashes, layer, assume_unique=False)
    return mask


def _append_history_layer(
    history_hash_layers: list[torch.Tensor],
    layer_hashes: torch.Tensor,
    history_depth: int,
) -> None:
    """
    Append one raw-hash layer to the non-backtracking history.

    Args:
        history_hash_layers: Mutable list of recent raw hash layers.
        layer_hashes: Hash layer to append.
        history_depth: Maximum number of layers to retain.

    """
    if history_depth <= 0:
        return
    history_hash_layers.append(layer_hashes)
    if len(history_hash_layers) > history_depth:
        del history_hash_layers[0 : len(history_hash_layers) - history_depth]


def _prune_candidate_pool(
    pool_by_hash: dict[int, _CandidateRecord],
    candidate_pool_size: int,
) -> None:
    """
    Trim the candidate pool to the configured limit.

    Args:
        pool_by_hash: Mutable mapping of candidate hash to candidate record.
        candidate_pool_size: Maximum number of candidates to keep.

    """
    if len(pool_by_hash) <= candidate_pool_size:
        return

    records = sorted(
        pool_by_hash.values(),
        key=lambda record: (-record.score, record.depth, record.state_hash),
    )[:candidate_pool_size]
    pool_by_hash.clear()
    pool_by_hash.update({record.state_hash: record for record in records})


def _update_candidate_pool(
    *,
    graph: Any,
    pool_by_hash: dict[int, _CandidateRecord],
    states: torch.Tensor,
    hashes: torch.Tensor,
    scores: torch.Tensor,
    depth: int,
    candidate_pool_size: int,
) -> None:
    """
    Merge one frontier into the retained candidate pool.

    Args:
        graph: Graph instance that can decode states.
        pool_by_hash: Mutable mapping of candidate hash to candidate record.
        states: Selected encoded frontier states.
        hashes: Hashes corresponding to ``states``.
        scores: Predictor scores corresponding to ``states``.
        depth: Expansion depth of the frontier.
        candidate_pool_size: Maximum number of candidates to keep.

    """
    if states.shape[0] == 0:
        return

    decoded_states = graph.decode_states(states).detach().cpu()
    hashes_cpu = hashes.detach().cpu()
    scores_cpu = scores.detach().cpu()

    for idx in range(states.shape[0]):
        state_hash = int(hashes_cpu[idx].item())
        score = float(scores_cpu[idx].item())
        candidate = _CandidateRecord(
            state_hash=state_hash,
            depth=int(depth),
            state=state_to_tuple(decoded_states[idx]),
            score=score,
        )
        existing = pool_by_hash.get(state_hash)
        should_replace = existing is None
        if existing is not None and candidate.score > existing.score:
            should_replace = True
        if (
            existing is not None
            and candidate.score == existing.score
            and candidate.depth < existing.depth
        ):
            should_replace = True
        if should_replace:
            pool_by_hash[state_hash] = candidate

    if len(pool_by_hash) > candidate_pool_size * 2:
        _prune_candidate_pool(pool_by_hash, candidate_pool_size)


def _restore_candidate_path(
    graph: Any,
    selected_hash_layers: Sequence[torch.Tensor],
    *,
    depth: int,
    state: Sequence[int],
) -> list[int]:
    """
    Restore a beam path from the start state to a retained candidate.

    Args:
        graph: Graph instance exposing ``restore_path``.
        selected_hash_layers: Selected hash layers, starting with the start
            state at index zero.
        depth: Depth at which the candidate was retained.
        state: Candidate state in decoded form.

    Returns:
        Path from the start state to the candidate as generator indices.

    """
    if depth <= 0:
        return []
    return graph.restore_path(list(selected_hash_layers[:depth]), list(state))


def _materialize_candidates(
    *,
    graph: Any,
    pool_by_hash: dict[int, _CandidateRecord],
    selected_hash_layers: Sequence[torch.Tensor],
    return_paths: bool,
    candidate_pool_size: int,
) -> list[OutwardExpansionCandidate]:
    """
    Convert the internal candidate pool to schema models.

    Args:
        graph: Graph instance used for optional path restoration.
        pool_by_hash: Internal candidate pool.
        selected_hash_layers: Selected hash layers for path restoration.
        return_paths: Whether beam paths should be restored.
        candidate_pool_size: Maximum number of candidates to return.

    Returns:
        Sorted candidate payloads ready for the result schema.

    """
    _prune_candidate_pool(pool_by_hash, candidate_pool_size)
    records = sorted(
        pool_by_hash.values(),
        key=lambda record: (-record.score, record.depth, record.state_hash),
    )

    candidates: list[OutwardExpansionCandidate] = []
    for record in records:
        beam_path = None
        if return_paths:
            beam_path = _restore_candidate_path(
                graph,
                selected_hash_layers,
                depth=record.depth,
                state=record.state,
            )
        candidates.append(
            OutwardExpansionCandidate(
                depth=record.depth,
                state=record.state,
                score=record.score,
                beam_path=beam_path,
                metadata={"state_hash": record.state_hash},
            )
        )
    return candidates


def _collect_step_metadata(
    *,
    scores_by_depth: dict[int, float],
    frontier_sizes_by_depth: dict[int, int],
    beam_mode_used: str,
    termination_reason: str,
    selected_hash_layers: Sequence[torch.Tensor],
    config: OutwardExpansionConfig,
) -> dict[str, Any]:
    """
    Build result metadata for one outward expansion run.

    Args:
        scores_by_depth: Best retained score at each expansion depth.
        frontier_sizes_by_depth: Retained frontier size at each expansion depth.
        beam_mode_used: Effective beam mode used by the run.
        termination_reason: Human-readable run termination reason.
        selected_hash_layers: Hash layers retained for path restoration.
        config: Run configuration.

    Returns:
        Serializable metadata dictionary.

    """
    return {
        "beam_mode_used": beam_mode_used,
        "candidate_pool_size": int(config.candidate_pool_size),
        "expanded_steps": int(len(frontier_sizes_by_depth)),
        "frontier_sizes_by_depth": frontier_sizes_by_depth,
        "path_layers_stored": int(len(selected_hash_layers)),
        "return_paths": bool(config.return_paths),
        "scores_by_depth": scores_by_depth,
        "termination_reason": termination_reason,
    }


def _run_simple_or_advanced(
    *,
    graph: Any,
    predictor: Predictor,
    config: OutwardExpansionConfig,
    beam_mode: str,
    beam_states: torch.Tensor,
    beam_hashes: torch.Tensor,
    selected_hash_layers: list[torch.Tensor],
    path_device: str | torch.device,
) -> tuple[dict[int, _CandidateRecord], dict[int, float], dict[int, int], str]:
    """
    Run outward expansion using the full-neighborhood implementation.

    Args:
        graph: Graph instance to search on.
        predictor: Normalized predictor wrapper.
        config: Runtime configuration.
        beam_mode: Effective beam mode, either ``"simple"`` or ``"advanced"``.
        beam_states: Initial encoded beam states.
        beam_hashes: Initial beam hashes.
        selected_hash_layers: Mutable path-restoration layers.
        path_device: Device used for optional path storage.

    Returns:
        Candidate pool, best scores by depth, frontier sizes by depth, and the
        termination reason.

    """
    use_history = beam_mode != "simple" and int(config.history_depth) > 0
    history_hash_layers = [beam_hashes.clone()] if use_history else []

    pool_by_hash: dict[int, _CandidateRecord] = {}
    scores_by_depth: dict[int, float] = {}
    frontier_sizes_by_depth: dict[int, int] = {}
    termination_reason = "max_steps_reached"

    for depth in range(1, int(config.max_steps) + 1):
        new_states = _ensure_rank_two(graph.get_neighbors(beam_states))
        new_states, new_hashes = graph.get_unique_states(new_states)
        step_history_hashes = new_hashes.clone()

        if use_history:
            unseen_mask = _mask_unseen_hashes(new_hashes, history_hash_layers)
            new_states = new_states[unseen_mask, :]
            new_hashes = new_hashes[unseen_mask]

        if new_hashes.numel() == 0:
            termination_reason = "no_new_states"
            break

        scores = _score_states(graph, predictor, new_states)
        selected_idx = _topk_indices(scores, int(config.beam_width))

        beam_states = new_states[selected_idx, :]
        beam_hashes = new_hashes[selected_idx]
        beam_scores = scores[selected_idx]

        frontier_sizes_by_depth[depth] = int(beam_states.shape[0])
        scores_by_depth[depth] = float(torch.max(beam_scores).item())

        _update_candidate_pool(
            graph=graph,
            pool_by_hash=pool_by_hash,
            states=beam_states,
            hashes=beam_hashes,
            scores=beam_scores,
            depth=depth,
            candidate_pool_size=int(config.candidate_pool_size),
        )

        if config.return_paths:
            selected_hash_layers.append(beam_hashes.to(path_device))

        if use_history:
            _append_history_layer(
                history_hash_layers,
                step_history_hashes,
                int(config.history_depth),
            )

        if int(config.verbose) >= 2:
            print(
                f"outward step={depth} beam={beam_states.shape[0]} "
                f"best_score={scores_by_depth[depth]:.4f}"
            )

    return pool_by_hash, scores_by_depth, frontier_sizes_by_depth, termination_reason


def _run_iterated(
    *,
    graph: Any,
    predictor: Predictor,
    config: OutwardExpansionConfig,
    beam_states: torch.Tensor,
    beam_hashes: torch.Tensor,
    selected_hash_layers: list[torch.Tensor],
    path_device: str | torch.device,
) -> tuple[dict[int, _CandidateRecord], dict[int, float], dict[int, int], str]:
    """
    Run outward expansion using a chunked per-generator implementation.

    Args:
        graph: Graph instance to search on.
        predictor: Normalized predictor wrapper.
        config: Runtime configuration.
        beam_states: Initial encoded beam states.
        beam_hashes: Initial beam hashes.
        selected_hash_layers: Mutable path-restoration layers.
        path_device: Device used for optional path storage.

    Returns:
        Candidate pool, best scores by depth, frontier sizes by depth, and the
        termination reason.

    """
    history_hash_layers = (
        [beam_hashes.clone()] if int(config.history_depth) > 0 else []
    )

    n_generators = int(graph.definition.n_generators)
    beam_width_part = max(1, int(math.ceil(int(config.beam_width) / n_generators)))

    pool_by_hash: dict[int, _CandidateRecord] = {}
    scores_by_depth: dict[int, float] = {}
    frontier_sizes_by_depth: dict[int, int] = {}
    termination_reason = "max_steps_reached"

    for depth in range(1, int(config.max_steps) + 1):
        selected_state_chunks: list[torch.Tensor] = []
        selected_hash_chunks: list[torch.Tensor] = []
        selected_score_chunks: list[torch.Tensor] = []
        generated_hashes_current = torch.empty(
            0, dtype=torch.int64, device=graph.device
        )
        step_history_parts: list[torch.Tensor] = []

        for new_states_chunk in graph.get_neighbors_generator(beam_states):
            new_states_chunk = _ensure_rank_two(new_states_chunk)
            new_hashes_chunk = graph.hasher.make_hashes(new_states_chunk)
            new_hashes_chunk, sort_idx = torch.sort(new_hashes_chunk, stable=True)
            new_states_chunk = new_states_chunk[sort_idx, :]

            if new_hashes_chunk.numel() > 1:
                unique_mask = torch.ones_like(new_hashes_chunk, dtype=torch.bool)
                unique_mask[1:] = new_hashes_chunk[1:] != new_hashes_chunk[:-1]
                new_states_chunk = new_states_chunk[unique_mask, :]
                new_hashes_chunk = new_hashes_chunk[unique_mask]

            if generated_hashes_current.numel() > 0 and new_hashes_chunk.numel() > 0:
                chunk_mask = ~torch.isin(
                    new_hashes_chunk, generated_hashes_current, assume_unique=False
                )
                new_states_chunk = new_states_chunk[chunk_mask, :]
                new_hashes_chunk = new_hashes_chunk[chunk_mask]

            if new_hashes_chunk.numel() == 0:
                continue

            step_history_parts.append(new_hashes_chunk.clone())
            generated_hashes_current = torch.cat(
                [generated_hashes_current, new_hashes_chunk], dim=0
            )

            if int(config.history_depth) > 0:
                unseen_mask = _mask_unseen_hashes(new_hashes_chunk, history_hash_layers)
                new_states_chunk = new_states_chunk[unseen_mask, :]
                new_hashes_chunk = new_hashes_chunk[unseen_mask]

            if new_hashes_chunk.numel() == 0:
                continue

            chunk_scores = _score_states(graph, predictor, new_states_chunk)
            selected_idx = _topk_indices(chunk_scores, beam_width_part)

            selected_state_chunks.append(new_states_chunk[selected_idx, :])
            selected_hash_chunks.append(new_hashes_chunk[selected_idx])
            selected_score_chunks.append(chunk_scores[selected_idx])

        if len(selected_state_chunks) == 0:
            termination_reason = "no_new_states"
            break

        beam_states = torch.cat(selected_state_chunks, dim=0)
        beam_hashes = torch.cat(selected_hash_chunks, dim=0)
        beam_scores = torch.cat(selected_score_chunks, dim=0)

        if beam_scores.numel() > int(config.beam_width):
            selected_idx = _topk_indices(beam_scores, int(config.beam_width))
            beam_states = beam_states[selected_idx, :]
            beam_hashes = beam_hashes[selected_idx]
            beam_scores = beam_scores[selected_idx]

        frontier_sizes_by_depth[depth] = int(beam_states.shape[0])
        scores_by_depth[depth] = float(torch.max(beam_scores).item())

        _update_candidate_pool(
            graph=graph,
            pool_by_hash=pool_by_hash,
            states=beam_states,
            hashes=beam_hashes,
            scores=beam_scores,
            depth=depth,
            candidate_pool_size=int(config.candidate_pool_size),
        )

        if config.return_paths:
            selected_hash_layers.append(beam_hashes.to(path_device))

        if int(config.history_depth) > 0 and len(step_history_parts) > 0:
            step_history_hashes = torch.unique(
                torch.cat(step_history_parts, dim=0),
                sorted=True,
            )
            _append_history_layer(
                history_hash_layers,
                step_history_hashes,
                int(config.history_depth),
            )

        if int(config.verbose) >= 2:
            print(
                f"outward iterated step={depth} beam={beam_states.shape[0]} "
                f"best_score={scores_by_depth[depth]:.4f}"
            )

    return pool_by_hash, scores_by_depth, frontier_sizes_by_depth, termination_reason


def run_outward_expansion(
    *,
    graph: Any,
    predictor: Any,
    config: OutwardExpansionConfig,
    start_state: Sequence[int] | torch.Tensor,
) -> OutwardExpansionResult:
    """
    Run model-guided outward beam expansion from one start state.

    This routine mirrors the shape of the existing inward beam-search helpers
    but ranks states by the largest predicted score so that high-distance
    candidates are retained for later exact verification.

    Args:
        graph: Graph instance that will be searched.
        predictor: Predictor or model-like object used for ranking.
        config: Runtime configuration for outward expansion.
        start_state: State from which to expand outward.

    Returns:
        Outward-expansion result with retained candidate states and optional
        beam paths.

    Raises:
        ValueError: If the configured beam mode is not supported.

    """
    normalized_predictor = _build_predictor(graph, predictor)
    beam_states, beam_hashes = graph.get_unique_states(graph.encode_states(start_state))

    beam_mode = str(config.beam_mode)
    valid_beam_modes = {"simple", "advanced", "iterated"}
    if beam_mode not in valid_beam_modes:
        raise ValueError(
            f"Unsupported beam_mode {beam_mode!r}; "
            f"expected one of {sorted(valid_beam_modes)}."
        )

    if beam_mode == "simple" and int(config.history_depth) > 0:
        warnings.warn(
            "OutwardExpansionBeamSearch: history_depth is ignored when "
            "beam_mode='simple'.",
            stacklevel=2,
        )

    n_generators = int(graph.definition.n_generators)
    if beam_mode == "iterated" and int(config.beam_width) < n_generators:
        warnings.warn(
            f"OutwardExpansionBeamSearch: beam_width={config.beam_width} is "
            f"smaller than n_generators={n_generators}; switching to "
            "beam_mode='simple'.",
            stacklevel=2,
        )
        beam_mode = "simple"

    path_device: str | torch.device = str(config.path_device)
    if path_device == "auto":
        path_device = "cpu" if config.return_paths else graph.device
    selected_hash_layers: list[torch.Tensor] = []
    if config.return_paths:
        selected_hash_layers.append(beam_hashes.to(path_device))

    if int(config.verbose) >= 1:
        print(
            f"outward start beam_mode={beam_mode} beam_width={config.beam_width} "
            f"max_steps={config.max_steps}"
        )

    with torch.inference_mode():
        if beam_mode == "iterated":
            pool_by_hash, scores_by_depth, frontier_sizes_by_depth, termination_reason = (
                _run_iterated(
                    graph=graph,
                    predictor=normalized_predictor,
                    config=config,
                    beam_states=beam_states,
                    beam_hashes=beam_hashes,
                    selected_hash_layers=selected_hash_layers,
                    path_device=path_device,
                )
            )
        else:
            pool_by_hash, scores_by_depth, frontier_sizes_by_depth, termination_reason = (
                _run_simple_or_advanced(
                    graph=graph,
                    predictor=normalized_predictor,
                    config=config,
                    beam_mode=beam_mode,
                    beam_states=beam_states,
                    beam_hashes=beam_hashes,
                    selected_hash_layers=selected_hash_layers,
                    path_device=path_device,
                )
            )

    candidates = _materialize_candidates(
        graph=graph,
        pool_by_hash=pool_by_hash,
        selected_hash_layers=selected_hash_layers,
        return_paths=bool(config.return_paths),
        candidate_pool_size=int(config.candidate_pool_size),
    )

    return OutwardExpansionResult(
        start_state=state_to_tuple(start_state),
        status="completed",
        config=config.model_copy(deep=True),
        candidates=candidates,
        termination_reason=termination_reason,
        metadata=_collect_step_metadata(
            scores_by_depth=scores_by_depth,
            frontier_sizes_by_depth=frontier_sizes_by_depth,
            beam_mode_used=beam_mode,
            termination_reason=termination_reason,
            selected_hash_layers=selected_hash_layers,
            config=config,
        ),
    )
