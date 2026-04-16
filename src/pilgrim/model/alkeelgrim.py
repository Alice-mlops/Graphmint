"""
AlkeelGrim is a distance prediction model on a graph.

It is a variant of the Pilgrim model assumed to be firstly
introduced here https://github.com/khoruzhii/cayleypy-cube, with
one-hot encoding of the nodes being replaced with learnable embeddings
and the residual blocks being replaced with a keel blocks from.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from pilgrim.model.model_blocks.keel_residuals import (
    KeelResidualBlock,
    PostLNAlphaBetaResidualBlock,
    PostLNAlphaResidualBlock,
)
from pilgrim.model.model_blocks.node_encoders import build_node_encoder


class AlkeelGrim(nn.Module):
    """
    Work-in-progress Pilgrim variant that will use Keel residual blocks.

    Args:
        config: The configuration dictionary.

    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.dtype: torch.dtype = config.get("model_dtype", torch.float32)
        self.state_size: int = config["state_size"]
        self.num_classes: int = config["num_classes"]
        self.hd1: int = config["hd1"]
        self.residual_blocks: list[int] | None = config.get("residual_blocks")
        self.residual_block_type: Literal[
            "keel", "post_ln_alpha", "post_ln_alpha_beta"
        ] = config.get("residual_block_type", "keel")
        self.z_add = 0
        self.output_dim = 1

        assert self.residual_blocks is not None, "residual_blocks must be provided"
        assert len(self.residual_blocks) > 0, "residual_blocks must be a non-empty list"

        init_hidden_dim = self.hd1

        if self.residual_block_type == "keel":
            self.residual_block_class = KeelResidualBlock
        elif self.residual_block_type == "post_ln_alpha":
            self.residual_block_class = PostLNAlphaResidualBlock
        elif self.residual_block_type == "post_ln_alpha_beta":
            self.residual_block_class = PostLNAlphaBetaResidualBlock
        else:
            raise ValueError(f"Unknown residual block type: {self.residual_block_type}")

        self.input_encoder, encoder_out_dim = build_node_encoder(config)
        # If the encoder output dim doesn't match hd1, adapt with a projection.
        self.input_proj = (
            nn.Identity()
            if encoder_out_dim == init_hidden_dim
            else nn.Linear(encoder_out_dim, init_hidden_dim)
        )
        self.bn1 = nn.BatchNorm1d(init_hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config["dropout_rate"])

        self.residual_blocks_list = nn.ModuleList([
            self.residual_block_class(
                init_hidden_dim,
                config["dropout_rate"],
                **config.get("residual_block_kwargs", {}),
            )
            for _ in self.residual_blocks
        ])

        self.output_layer = nn.Linear(init_hidden_dim, self.output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: The input tensor.

        Returns:
            The output tensor.

        """
        x: torch.Tensor = self.input_encoder(z)
        # Allow encoders to return (batch, ..., feat) tensors.
        if x.dim() > 2:  # noqa: PLR2004
            x = x.view(x.size(0), -1)
        x = x.to(self.dtype)
        x = self.input_proj(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        for block in self.residual_blocks_list:
            x = block(x)
        x = self.output_layer(x)
        return x.flatten()
