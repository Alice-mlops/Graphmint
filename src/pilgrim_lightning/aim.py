"""Aim logging helpers for Lightning-based Pilgrim experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from aim import Run
from lightning.pytorch import Callback, LightningModule, Trainer

from .config import AimRunConfig


def to_aim_serializable(value: Any) -> Any:
    """
    Convert arbitrary values to Aim-storable primitives.

    Args:
        value: Input value to convert.

    Returns:
        Value converted to a primitive, list, or dict accepted by Aim.

    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_aim_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_aim_serializable(v) for v in value]
    return str(value)


def _open_run(config: AimRunConfig) -> Run:
    """
    Open an Aim run with repository fallback handling.

    Args:
        config: Aim run configuration.

    Returns:
        Opened Aim ``Run`` instance.

    """
    repo_arg = str(config.repo) if config.repo is not None else None
    try:
        if repo_arg is None:
            return Run(experiment=config.experiment)
        return Run(experiment=config.experiment, repo=repo_arg)
    except RuntimeError:
        fallback_repo = str(Path.cwd())
        return Run(experiment=config.experiment, repo=fallback_repo)


def open_aim_run(
    config: AimRunConfig,
    *,
    hparams: dict[str, Any] | None = None,
    group_n: int | None = None,
) -> Run:
    """
    Open and initialize an Aim run outside Lightning callbacks.

    Args:
        config: Aim run settings.
        hparams: Optional hyperparameters payload stored under ``hparams``.
        group_n: Optional ``n`` grouping value stored under ``group/n``.

    Returns:
        Opened and initialized Aim run.

    """
    run = _open_run(config)
    for tag in config.tags:
        run.add_tag(tag)

    run["meta/stage"] = config.stage
    if config.notebook is not None:
        run["meta/notebook"] = config.notebook
    if config.model_name is not None:
        run["meta/model"] = config.model_name
    for key, value in config.extra_meta.items():
        run[f"meta/{key}"] = to_aim_serializable(value)
    if hparams is not None:
        run["hparams"] = to_aim_serializable(hparams)
    if group_n is not None:
        run["group/n"] = int(group_n)
    return run


def _to_float(value: Any) -> float | None:
    """
    Convert metric-like values to float.

    Args:
        value: Input metric value.

    Returns:
        ``float`` value when conversion succeeds, otherwise ``None``.

    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int)):
        return float(value)
    return None


class AimTrackingCallback(Callback):
    """
    Lightning callback that logs epoch metrics into Aim.

    Args:
        config: Aim run settings.
        hparams: Hyperparameters persisted into the run.
        context: Context fields attached to each metric.

    """

    def __init__(
        self,
        config: AimRunConfig,
        *,
        hparams: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hparams = hparams
        self.context = context or {}
        self.run: Run | None = None
        self.start_time: float = 0.0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Open run and write metadata.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module being trained.

        Returns:
            None.

        """
        del trainer
        self.start_time = time.perf_counter()
        self.run = _open_run(self.config)
        for tag in self.config.tags:
            self.run.add_tag(tag)

        self.run["meta/stage"] = self.config.stage
        if self.config.notebook is not None:
            self.run["meta/notebook"] = self.config.notebook
        if self.config.model_name is not None:
            self.run["meta/model"] = self.config.model_name

        for key, value in self.config.extra_meta.items():
            self.run[f"meta/{key}"] = to_aim_serializable(value)

        self.run["hparams"] = to_aim_serializable(self.hparams)
        if hasattr(pl_module, "problem_n"):
            self.run["group/n"] = int(pl_module.problem_n)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """
        Log callback metrics at each validation epoch end.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module being trained.

        Returns:
            None.

        """
        del pl_module
        self._track_metrics(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Log train metrics when validation is disabled.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module being trained.

        Returns:
            None.

        """
        del pl_module
        if trainer.check_val_every_n_epoch <= 0:
            self._track_metrics(trainer)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Finalize run and write elapsed time metric.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module being trained.

        Returns:
            None.

        """
        del trainer
        del pl_module
        if self.run is None:
            return

        elapsed = time.perf_counter() - self.start_time
        run_context = dict(self.context)
        run_context["phase"] = "train"
        self.run.track(float(elapsed), name="time_s", context=run_context)
        self.run.close()
        self.run = None

    def _track_metrics(self, trainer: Trainer) -> None:
        """
        Track scalar callback metrics for the current epoch.

        Args:
            trainer: Active Lightning trainer.

        Returns:
            None.

        """
        if self.run is None:
            return
        if trainer.sanity_checking:
            return

        step = int(trainer.current_epoch)
        for name, value in trainer.callback_metrics.items():
            metric = _to_float(value)
            if metric is None:
                continue
            if not (
                str(name).startswith("train/")
                or str(name).startswith("val/")
                or str(name).startswith("lr/")
            ):
                continue
            self.run.track(metric, name=str(name), step=step, context=self.context)
