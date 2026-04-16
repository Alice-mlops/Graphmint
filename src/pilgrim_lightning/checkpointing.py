"""State-dict checkpoint callbacks for Pilgrim Lightning training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


def _to_float(metric: Any) -> float | None:
    """
    Convert callback metric to float.

    Args:
        metric: Metric value from Lightning callback metrics.

    Returns:
        Float metric or ``None`` if conversion failed.

    """
    if metric is None:
        return None
    if isinstance(metric, torch.Tensor):
        if metric.numel() != 1:
            return None
        return float(metric.detach().cpu().item())
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def _cpu_state_dict(module: LightningModule) -> dict[str, torch.Tensor]:
    """
    Return CPU-cloned state dict from model module.

    Args:
        module: Lightning module instance.

    Returns:
        CPU-cloned state dictionary.

    """
    model = getattr(module, "model", module)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class StateDictCheckpointCallback(Callback):
    """
    Save notebook-compatible ``.pt`` state-dict checkpoints.

    Args:
        output_dir: Directory where checkpoint files are written.

    """

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val = float("inf")
        self.best_center_abs = float("inf")
        self.best_lip = float("inf")

        self.best_val_path = self.output_dir / "best_val.pt"
        self.best_center_path = self.output_dir / "best_center.pt"
        self.best_lip_path = self.output_dir / "best_lip.pt"
        self.final_path = self.output_dir / "final.pt"

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """
        Update best checkpoints from validation metrics.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module being trained.

        Returns:
            None.

        """
        metrics = trainer.callback_metrics
        val_loss = _to_float(metrics.get("val/loss"))
        center_eval = _to_float(metrics.get("val/center_eval"))
        val_lip = _to_float(metrics.get("val/lip"))

        state = _cpu_state_dict(pl_module)
        if val_loss is not None and val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(state, self.best_val_path)

        if center_eval is not None and abs(center_eval) < self.best_center_abs:
            self.best_center_abs = abs(center_eval)
            torch.save(state, self.best_center_path)

        if val_lip is not None and val_lip < self.best_lip:
            self.best_lip = val_lip
            torch.save(state, self.best_lip_path)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Save final state dict at fit end.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Lightning module being trained.

        Returns:
            None.

        """
        del trainer
        torch.save(_cpu_state_dict(pl_module), self.final_path)
