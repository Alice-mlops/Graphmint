"""
LayerNorm-based residual MLP blocks ("Keel" ablations).

This module contains three closely related residual blocks that vary the
ordering of normalization and the parameterization of the residual mix.

All blocks are shape-preserving: for an input tensor ``x`` of shape
``(batch, hidden_dim)``, they return a tensor of the same shape.

Blocks:
    - ``PostLNAlphaResidualBlock``: ``y = LN(alpha * x + F(x))``.
    - ``PostLNAlphaBetaResidualBlock``: ``y = LN(alpha * x + F(beta ⊙ x))`` with
      learnable per-feature ``beta``.
    - ``KeelResidualBlock``: ``y = LN_out(alpha * x + F(LN_in(x)))`` (pre-norm
      into ``F`` plus post-norm mixing).

The residual function ``F`` defaults to ``ResidualMLP`` but can be overridden
via ``residual_fn``.
"""

from __future__ import annotations

import torch
from torch import nn

from .residuals import ResidualMLP


class PostLNAlphaResidualBlock(nn.Module):
    """
    Post-LN residual block with fixed skip scaling (Attempt 1).

    Implements ``y = LN(alpha * x + F(x))``.

    Args:
        hidden_dim: Feature dimension of the input and output.
        dropout_rate: Dropout probability used when constructing the default
            ``ResidualMLP``. Ignored if ``residual_fn`` is provided.
        activation: Activation module used when constructing the default
            ``ResidualMLP``. Ignored if ``residual_fn`` is provided.
        alpha: Scalar multiplier applied to the skip connection.
        eps: Epsilon used by ``nn.LayerNorm``.
        residual_fn: Optional module implementing ``F``. Must map
            ``(batch, hidden_dim)`` to ``(batch, hidden_dim)``. If provided,
            ``dropout_rate``, ``activation``, and ``mlp_hidden_mult`` are
            ignored.
        mlp_hidden_mult: Inner-dimension multiplier for the default
            ``ResidualMLP``.

    """

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        activation: nn.Module | None = None,
        *,
        alpha: float = 1.0,
        eps: float = 1e-5,
        residual_fn: nn.Module | None = None,
        mlp_hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.f = residual_fn or ResidualMLP(
            hidden_dim,
            hidden_mult=mlp_hidden_mult,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self.ln_out = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.

        """
        return self.ln_out(self.alpha * x + self.f(x))


class PostLNAlphaBetaResidualBlock(nn.Module):
    """
    Post-LN residual block with skip scaling and learnable input scale (Attempt 2).

    Implements ``y = LN(alpha * x + F(beta ⊙ x))`` where ``beta`` is a learnable
    per-feature scaling vector (stored as an ``nn.Parameter`` of shape
    ``(hidden_dim,)``).

    Args:
        hidden_dim: Feature dimension of the input and output.
        dropout_rate: Dropout probability used when constructing the default
            ``ResidualMLP``. Ignored if ``residual_fn`` is provided.
        activation: Activation module used when constructing the default
            ``ResidualMLP``. Ignored if ``residual_fn`` is provided.
        alpha: Scalar multiplier applied to the skip connection.
        eps: Epsilon used by ``nn.LayerNorm``.
        residual_fn: Optional module implementing ``F``. Must map
            ``(batch, hidden_dim)`` to ``(batch, hidden_dim)``. If provided,
            ``dropout_rate``, ``activation``, and ``mlp_hidden_mult`` are
            ignored.
        mlp_hidden_mult: Inner-dimension multiplier for the default
            ``ResidualMLP``.

    """

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        activation: nn.Module | None = None,
        *,
        alpha: float = 1.0,
        eps: float = 1e-5,
        residual_fn: nn.Module | None = None,
        mlp_hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = nn.Parameter(torch.ones(hidden_dim))
        self.f = residual_fn or ResidualMLP(
            hidden_dim,
            hidden_mult=mlp_hidden_mult,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self.ln_out = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.

        """
        return self.ln_out(self.alpha * x + self.f(x * self.beta))


class KeelResidualBlock(nn.Module):
    """
    Pre-norm into ``F`` plus Post-LN mixing ("Keel", Attempt 3).

    Implements ``y = LN_out(alpha * x + F(LN_in(x)))``.

    Compared to ``PostLNAlphaResidualBlock``, this variant normalizes the input
    *before* sending it into the residual function ``F`` (via ``LN_in``), then
    applies a second LayerNorm (``LN_out``) after mixing with the skip path.

    Args:
        hidden_dim: Feature dimension of the input and output.
        dropout_rate: Dropout probability used when constructing the default
            ``ResidualMLP``. Ignored if ``residual_fn`` is provided.
        activation: Activation module used when constructing the default
            ``ResidualMLP``. Ignored if ``residual_fn`` is provided.
        alpha: Scalar multiplier applied to the skip connection.
        eps: Epsilon used by ``nn.LayerNorm`` (for both ``LN_in`` and
            ``LN_out``).
        residual_fn: Optional module implementing ``F``. Must map
            ``(batch, hidden_dim)`` to ``(batch, hidden_dim)``. If provided,
            ``dropout_rate``, ``activation``, and ``mlp_hidden_mult`` are
            ignored.
        mlp_hidden_mult: Inner-dimension multiplier for the default
            ``ResidualMLP``.

    """

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        activation: nn.Module | None = None,
        *,
        alpha: float = 1.0,
        eps: float = 1e-5,
        residual_fn: nn.Module | None = None,
        mlp_hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.ln_in = nn.LayerNorm(hidden_dim, eps=eps)
        self.f = residual_fn or ResidualMLP(
            hidden_dim,
            hidden_mult=mlp_hidden_mult,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self.ln_out = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.

        """
        return self.ln_out(self.alpha * x + self.f(self.ln_in(x)))
