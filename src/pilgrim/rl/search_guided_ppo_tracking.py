# Defines tracker protocols for search-guided PPO trainers.
"""Tracking interfaces for search-guided PPO runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pilgrim.schemas.rl.search_guided_ppo import (
        SearchGuidedPPOMetrics,
        SearchGuidedPPOStepDiagnostics,
    )

    from .search_guided_ppo import SearchGuidedPPOTrainer


class SearchGuidedPPOTracker(Protocol):
    """Protocol for optional side-effect trackers used by the PPO trainer."""

    def on_fit_start(self, trainer: SearchGuidedPPOTrainer) -> None:
        """
        Handle the start of ``fit``.

        Args:
            trainer: Active PPO trainer.

        """

    def on_train_step_end(
        self,
        trainer: SearchGuidedPPOTrainer,
        diagnostics: SearchGuidedPPOStepDiagnostics,
    ) -> None:
        """
        Handle diagnostics produced after one PPO update.

        Args:
            trainer: Active PPO trainer.
            diagnostics: Step-level metrics and rollout statistics.

        """

    def on_fit_end(
        self,
        trainer: SearchGuidedPPOTrainer,
        history: Sequence[SearchGuidedPPOMetrics],
    ) -> None:
        """
        Handle the end of ``fit``.

        Args:
            trainer: Active PPO trainer.
            history: Metrics returned by completed updates.

        """
