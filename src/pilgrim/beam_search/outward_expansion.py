# Defines outward beam-expansion classes for maximal-distance search.
"""Outward beam-expansion orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from pilgrim.schemas.beam_search import OutwardExpansionConfig, OutwardExpansionResult
from pilgrim.utils.beam_search.outward_expansion import run_outward_expansion


class OutwardExpansionBeamSearch:
    """
    Orchestrator for model-guided outward beam expansion.

    The search expands away from the provided start state, retains high-scoring
    far-away candidates, and can optionally restore the beam paths that reached
    those candidates for later exact-distance verification.

    Args:
        graph: Graph instance searched by the expansion routine.
        predictor: Predictor or model-like object used to rank candidates.
        config: Runtime configuration for outward expansion.

    """

    def __init__(
        self,
        *,
        graph: Any,
        predictor: Any,
        config: OutwardExpansionConfig,
    ) -> None:
        """
        Initialize the outward expansion orchestrator.

        Args:
            graph: Graph instance searched by the expansion routine.
            predictor: Predictor or model-like object used to rank candidates.
            config: Runtime configuration for outward expansion.

        """
        self.graph = graph
        self.predictor = predictor
        self.config = config

    def search(
        self,
        start_state: Sequence[int] | torch.Tensor,
    ) -> OutwardExpansionResult:
        """
        Launch outward beam expansion from one start state.

        Args:
            start_state: State from which to expand outward.

        Returns:
            Outward-expansion result with retained candidates and optional
            beam paths.

        """
        return run_outward_expansion(
            graph=self.graph,
            predictor=self.predictor,
            config=self.config,
            start_state=start_state,
        )
