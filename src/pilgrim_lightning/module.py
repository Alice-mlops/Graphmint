"""Lightning module wrapper for Pilgrim-family regression models."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from cayleypy import CayleyGraph
from pilgrim.utils.losses import lipschitz_expansion_loss
from pilgrim.utils.training_utils import lr_scheduler_ctor_from_cfg
from torch import nn

from .config import LipschitzConfig, OptimizationConfig


class PilgrimLightningModule(L.LightningModule):
    """
    Lightning module for MSE regression on graph-state distances.

    Args:
        model: Underlying Pilgrim-family model.
        graph: Graph object required by lip-loss and center evaluation.
        optimization: Optimizer and scheduler configuration.
        lipschitz: Optional lipschitz regularization configuration.
        problem_n: Optional group size label used for tracking.

    """

    def __init__(
        self,
        model: nn.Module,
        graph: CayleyGraph,
        optimization: OptimizationConfig,
        lipschitz: LipschitzConfig,
        *,
        problem_n: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.graph = graph
        self.optimization = optimization
        self.lipschitz = lipschitz
        self.problem_n = problem_n
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass.

        Args:
            x: Input states tensor.

        Returns:
            Predicted distances.

        """
        return self.model(x.long())

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Compute train loss and optional lip-loss.

        Args:
            batch: Batch ``(x, y)``.
            batch_idx: Batch index.

        Returns:
            Scalar training loss.

        """
        del batch_idx
        x, y = batch
        pred = self.forward(x)
        mse = self.loss_fn(pred.float(), y.float())

        total = mse
        lip_loss: torch.Tensor | None = None
        if float(self.lipschitz.weight) > 0.0:
            lip_loss = lipschitz_expansion_loss(
                self.model,
                self.graph,
                x,
                max_states=self.lipschitz.max_states,
                generator_indices=self.lipschitz.generator_indices,
                max_generators=self.lipschitz.max_generators,
                seed=self.lipschitz.seed,
                state_batch_size=self.lipschitz.state_batch_size,
                reduction=self.lipschitz.reduction,
            ).float()
            total = mse + float(self.lipschitz.weight) * lip_loss

        self.log("train/loss", total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", mse, on_step=False, on_epoch=True)
        if lip_loss is not None:
            self.log("train/lip", lip_loss, on_step=False, on_epoch=True)

        return total

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Compute validation metrics for one batch.

        Args:
            batch: Batch ``(x, y)``.
            batch_idx: Batch index.

        Returns:
            Validation MSE loss.

        """
        del batch_idx
        x, y = batch
        pred = self.forward(x)
        val_loss = self.loss_fn(pred.float(), y.float())

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.lipschitz.val_metric:
            val_lip = lipschitz_expansion_loss(
                self.model,
                self.graph,
                x,
                max_states=self.lipschitz.max_states,
                generator_indices=self.lipschitz.generator_indices,
                max_generators=self.lipschitz.max_generators,
                seed=self.lipschitz.seed,
                state_batch_size=self.lipschitz.state_batch_size,
                reduction=self.lipschitz.reduction,
            ).float()
            self.log("val/lip", val_lip, on_step=False, on_epoch=True)

        return val_loss

    def on_validation_epoch_end(self) -> None:
        """
        Log center-state prediction metric.

        Returns:
            None.

        """
        with torch.no_grad():
            z = torch.as_tensor(self.graph.central_state, device=self.graph.device)
            if z.ndim == 0:
                z = z.view(1, 1)
            elif z.ndim == 1:
                z = z.unsqueeze(0)
            out = self.forward(z.long()).reshape(-1)
            center_eval = out[0].float()
        self.log("val/center_eval", center_eval, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        """
        Build optimizer and optional scheduler.

        Returns:
            Optimizer config in Lightning format.

        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.optimization.lr),
            weight_decay=float(self.optimization.weight_decay),
        )

        scheduler_ctor = lr_scheduler_ctor_from_cfg({
            "num_epochs": int(self.optimization.num_epochs),
            "lr_scheduler": self.optimization.lr_scheduler,
        })
        if scheduler_ctor is None:
            return optimizer

        scheduler = scheduler_ctor(optimizer)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                },
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
