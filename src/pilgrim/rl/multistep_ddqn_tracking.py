# Defines tracker protocols for multi-step Double-DQN trainers.
"""Tracking interfaces for multi-step Double-DQN runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pilgrim.schemas.rl import (
        MultiStepDDQNMetrics,
        MultiStepDDQNStepDiagnostics,
    )

    from .multistep_double_q_learning import MultiStepDDQNTrainer


class MultiStepDDQNTracker(Protocol):
    """
    Protocol for optional side-effect trackers used by the DDQN trainer.

    Methods on the protocol may log to Aim, write files, or compute summaries.
    The trainer keeps the protocol deliberately small so notebooks can swap in
    different trackers without changing the optimization loop.

    """

    def on_fit_start(self, trainer: MultiStepDDQNTrainer) -> None:
        """
        Handle the start of ``fit``.

        Args:
            trainer: Active multi-step DDQN trainer.

        Returns:
            None.

        """

    def on_train_step_end(
        self,
        trainer: MultiStepDDQNTrainer,
        diagnostics: MultiStepDDQNStepDiagnostics,
    ) -> None:
        """
        Handle diagnostics produced after one optimizer step.

        Args:
            trainer: Active multi-step DDQN trainer.
            diagnostics: Step-level metrics and batch diagnostics.

        Returns:
            None.

        """

    def on_fit_end(
        self,
        trainer: MultiStepDDQNTrainer,
        history: Sequence[MultiStepDDQNMetrics],
    ) -> None:
        """
        Handle the end of ``fit``.

        Args:
            trainer: Active multi-step DDQN trainer.
            history: Metrics returned by completed optimizer steps.

        Returns:
            None.

        """
