"""
Residual MLP-style blocks used by Pilgrim.

The blocks in this module are *shape-preserving*: they map an input tensor with
final dimension ``hidden_dim`` back to the same shape, making them suitable for
residual stacks.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    BatchNorm-based residual block used by the Pilgrim MLP stack.

    This is a 2-layer MLP residual block that keeps the feature dimension
    constant:

        x -> Linear -> BatchNorm -> Activation -> Dropout
          -> Linear -> BatchNorm -> (+ x) -> Activation

    Args:
        hidden_dim: Feature dimension of the input and output.
        dropout_rate: Dropout probability applied inside the residual branch.
        activation: Activation module to use. Defaults to ``nn.GELU()``.

    """

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation: nn.Module = activation or nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the residual block.

        Args:
            x: Input tensor of shape ``(batch, hidden_dim)``.

        Returns:
            Tensor of the same shape as ``x``.

        """
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.activation(out)
        return out


class ResidualMLP(nn.Module):
    """
    Two-layer MLP used as a residual branch ``F(x)``.

    This module is commonly used as the learnable residual function inside
    LayerNorm-based residual blocks (see ``keel_residuals.py``).

    Args:
        hidden_dim: Feature dimension of the input and output.
        hidden_mult: Multiplier for the inner dimension. The inner dimension is
            ``int(hidden_dim * hidden_mult)`` and must be at least
            ``hidden_dim``.
        dropout_rate: Dropout probability applied after each linear layer.
        activation: Activation module to use. Defaults to ``nn.GELU()``.

    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        hidden_mult: int = 1,
        dropout_rate: float = 0.1,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_mult < 1:
            raise ValueError("hidden_mult must be >= 1.")

        inner_dim = int(hidden_dim * hidden_mult)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.activation: nn.Module = activation or nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the MLP.

        Args:
            x: Input tensor of shape ``(batch, hidden_dim)``.

        Returns:
            Tensor of shape ``(batch, hidden_dim)``.

        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
