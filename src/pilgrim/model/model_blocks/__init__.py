"""
Reusable building blocks for Pilgrim-family models.

This package groups small, composable modules used to build the Pilgrim MLP
stack and its variants:

- Input encoders (see ``build_node_encoder``).
- Residual MLP blocks (see ``ResidualBlock`` and the LayerNorm variants in
  ``keel_residuals.py``).
"""

from __future__ import annotations

from .keel_residuals import (
    KeelResidualBlock,
    PostLNAlphaBetaResidualBlock,
    PostLNAlphaResidualBlock,
)
from .node_encoders import build_node_encoder
from .residuals import ResidualBlock

__all__ = [
    "KeelResidualBlock",
    "PostLNAlphaBetaResidualBlock",
    "PostLNAlphaResidualBlock",
    "ResidualBlock",
    "build_node_encoder",
]
