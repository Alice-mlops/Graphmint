# Combines multiple RL tracker implementations behind one tracker interface.
"""Composite tracker for multi-step TD value-learning runs."""

from __future__ import annotations

from collections.abc import Sequence

from .multistep_td_tracking import MultiStepTDValueTracker


class CompositeMultiStepTDValueTracker:
    """
    Forward trainer events to multiple concrete tracker implementations.

    Args:
        trackers: Ordered tracker list. ``None`` values are ignored.

    """

    def __init__(self, trackers: Sequence[MultiStepTDValueTracker | None]) -> None:
        self.trackers = [tracker for tracker in trackers if tracker is not None]

    def on_fit_start(self, trainer) -> None:
        """
        Forward ``on_fit_start`` to child trackers.

        Args:
            trainer: Active multi-step TD value trainer.

        """
        for tracker in self.trackers:
            tracker.on_fit_start(trainer)

    def on_train_step_end(self, trainer, diagnostics) -> None:
        """
        Forward ``on_train_step_end`` to child trackers.

        Args:
            trainer: Active multi-step TD value trainer.
            diagnostics: Step-level diagnostics produced by the trainer.

        """
        for tracker in self.trackers:
            tracker.on_train_step_end(trainer, diagnostics)

    def on_fit_end(self, trainer, history) -> None:
        """
        Forward ``on_fit_end`` to child trackers.

        Args:
            trainer: Active multi-step TD value trainer.
            history: Trainer history returned from ``fit``.

        """
        for tracker in self.trackers:
            tracker.on_fit_end(trainer, history)
