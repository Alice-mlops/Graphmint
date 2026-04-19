"""Lightning datamodule for Pilgrim random-walk training data."""

from __future__ import annotations

import random
from typing import Any

import lightning as L
import numpy as np
import torch
from cayleypy import CayleyGraph
from pilgrim.utils.graph_utils import subsample_xy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import RandomWalkDataConfig


class RandomWalkDataModule(L.LightningDataModule):
    """
    Create train/val dataloaders from CayleyGraph random walks.

    Args:
        graph: Graph used for random-walk sample generation.
        config: Datamodule configuration.

    """

    def __init__(self, graph: CayleyGraph, config: RandomWalkDataConfig) -> None:
        super().__init__()
        self.graph = graph
        self.config = config

        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None
        self._last_refresh_epoch: int | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Create initial train/val datasets.

        Args:
            stage: Lightning stage value.

        Returns:
            None.

        """
        del stage
        if self.train_dataset is None or self.val_dataset is None:
            self.refresh_for_epoch(epoch=0)

    def refresh_for_epoch(self, epoch: int) -> None:
        """
        Refresh train/validation datasets for a target epoch.

        Args:
            epoch: Epoch index used for deterministic seeding.

        Returns:
            None.

        """
        refresh_interval = int(self.config.rw_refresh_interval)
        if refresh_interval < 1:
            raise ValueError("rw_refresh_interval must be >= 1.")

        if (
            self._last_refresh_epoch is not None
            and epoch % refresh_interval != 0
            and self.train_dataset is not None
            and self.val_dataset is not None
        ):
            return

        x, y = self._generate_walk_samples(epoch=epoch)
        x, y = subsample_xy(
            x,
            y,
            cap=self.config.max_samples_cap,
            seed=int(self.config.seed + epoch),
        )
        x_tr, x_va, y_tr, y_va = self._train_val_split(x, y, epoch=epoch)

        self.train_dataset = TensorDataset(
            torch.as_tensor(x_tr).long(), torch.as_tensor(y_tr).float()
        )
        self.val_dataset = TensorDataset(
            torch.as_tensor(x_va).long(), torch.as_tensor(y_va).float()
        )
        self._last_refresh_epoch = int(epoch)

    def train_dataloader(self) -> DataLoader:
        """
        Build training dataloader.

        Returns:
            Training dataloader.

        """
        self._ensure_refreshed()
        if self.train_dataset is None:
            raise RuntimeError("train dataset is not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Build validation dataloader.

        Returns:
            Validation dataloader.

        """
        self._ensure_refreshed()
        if self.val_dataset is None:
            raise RuntimeError("val dataset is not initialized.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

    def _ensure_refreshed(self) -> None:
        """
        Refresh datasets for the current trainer epoch when needed.

        Returns:
            None.

        """
        epoch = 0
        if self.trainer is not None:
            epoch = int(self.trainer.current_epoch)
        if self._last_refresh_epoch != epoch:
            self.refresh_for_epoch(epoch=epoch)

    def _generate_walk_samples(
        self, *, epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random-walk samples using notebook-equivalent schedule.

        Args:
            epoch: Current epoch index used for per-sampling seeding.

        Returns:
            Tuple of feature and target tensors.

        """
        base_len = int(self.config.rw_length)
        base_width = int(self.config.rw_width)

        rw_lengths = self.config.rw_lengths
        if rw_lengths is None:
            rw_lengths = [
                (1.0, base_len),
                (0.5, base_len // 2),
                (0.25, base_len // 4),
            ]

        x_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        sample_idx = 0

        for factor, length in rw_lengths:
            if int(length) < 10:
                continue
            self._set_rw_sampling_seed(epoch=epoch, sample_idx=sample_idx)
            width = max(1, int(base_width * float(factor)))
            x_part, y_part = self.graph.random_walks(
                width=width,
                length=int(length),
                mode=str(self.config.rw_mode),
                nbt_history_depth=int(length),
            )
            x_parts.append(torch.as_tensor(x_part))
            y_parts.append(torch.as_tensor(y_part))
            sample_idx += 1

        if not x_parts:
            self._set_rw_sampling_seed(epoch=epoch, sample_idx=sample_idx)
            x_part, y_part = self.graph.random_walks(
                width=base_width,
                length=base_len,
                mode=str(self.config.rw_mode),
                nbt_history_depth=base_len,
            )
            x_parts.append(torch.as_tensor(x_part))
            y_parts.append(torch.as_tensor(y_part))

        return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0)

    def _set_rw_sampling_seed(self, *, epoch: int, sample_idx: int) -> None:
        """
        Set unique RNG seed for one random-walk sampling call.

        Args:
            epoch: Current epoch index.
            sample_idx: Sampling call index within this refresh.

        Returns:
            None.

        """
        base_seed = int(self.config.seed)
        seed = base_seed + int(epoch) * 100_003 + int(sample_idx)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _train_val_split(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epoch: int,
    ) -> tuple[Any, Any, Any, Any]:
        """
        Split samples into train and validation subsets.

        Args:
            x: Sample features.
            y: Sample labels.
            epoch: Epoch index used for deterministic split seed.

        Returns:
            Tuple ``(x_tr, x_val, y_tr, y_val)``.

        """
        split_kwargs = {
            "train_size": 1.0 - float(self.config.val_ratio),
            "shuffle": True,
            "random_state": int(self.config.seed + epoch),
        }
        try:
            return train_test_split(
                x,
                y,
                stratify=torch.as_tensor(y).detach().cpu().numpy(),
                **split_kwargs,
            )
        except Exception:
            return train_test_split(x, y, **split_kwargs)
