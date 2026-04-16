# Defines tracker protocols for multi-step TD value-learning trainers.
"""Tracking interfaces for multi-step TD value-learning runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pilgrim.schemas.rl import (
        MultiStepTDValueMetrics,
        MultiStepTDValueStepDiagnostics,
    )

    from .multistep_td_value_iteration import MultiStepTDValueTrainer


class MultiStepTDValueTracker(Protocol):
    """
    Protocol for optional side-effect trackers used by the trainer.

    Methods on the protocol may log to Aim, write files, or compute summaries.
    The trainer keeps the protocol deliberately small so notebooks can swap in
    different trackers without changing the optimization loop.

    """

    def on_fit_start(self, trainer: MultiStepTDValueTrainer) -> None:
        """
        Handle the start of ``fit``.

        Args:
            trainer: Active multi-step TD value trainer.

        Returns:
            None.

        """

    def on_train_step_end(
        self,
        trainer: MultiStepTDValueTrainer,
        diagnostics: MultiStepTDValueStepDiagnostics,
    ) -> None:
        """
        Handle diagnostics produced after one optimizer step.

        Args:
            trainer: Active multi-step TD value trainer.
            diagnostics: Step-level metrics and batch diagnostics.

        Returns:
            None.

        """

    def on_fit_end(
        self,
        trainer: MultiStepTDValueTrainer,
        history: Sequence[MultiStepTDValueMetrics],
    ) -> None:
        """
        Handle the end of ``fit``.

        Args:
            trainer: Active multi-step TD value trainer.
            history: Metrics returned by completed optimizer steps.

        Returns:
            None.

        """
