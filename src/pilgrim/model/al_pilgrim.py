"""AlPilgrim model and input encoders."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .model_blocks import ResidualBlock, build_node_encoder

_FLATTEN_TO_2D_DIM_THRESHOLD = 2


class AlPilgrim(nn.Module):
    """
    AlPilgrim model.

    Parameters
    ----------
        config: The configuration dictionary.

    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.dtype: torch.dtype = config.get("model_dtype", torch.float32)
        self.state_size: int = config["state_size"]
        self.num_classes: int = config["num_classes"]
        self.hd1: int = config["hd1"]
        self.residual_blocks: list[int] | None = config.get("residual_blocks")
        self.z_add = 0
        self.output_dim = 1

        hd1 = self.hd1

        self.input_encoder, encoder_out_dim = build_node_encoder(config)
        # If the encoder output dim doesn't match hd1, adapt with a projection.
        self.input_proj = (
            nn.Identity() if encoder_out_dim == hd1 else nn.Linear(encoder_out_dim, hd1)
        )
        self.bn1 = nn.BatchNorm1d(hd1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config["dropout_rate"])

        # Residual stack (optional). If residual blocks have varying sizes, add
        # explicit transitions between them to avoid shape mismatches.
        if self.residual_blocks:
            self.residual_blocks_list = nn.ModuleList([
                ResidualBlock(residual_size, config.get("dropout_rate", 0.1))
                for residual_size in self.residual_blocks
            ])
            first_residual_size = self.residual_blocks[0]
            self.hidden_layer = (
                nn.Identity()
                if self.hd1 == first_residual_size
                else nn.Linear(self.hd1, first_residual_size)
            )
            self.bn2 = nn.BatchNorm1d(first_residual_size)
            self.residual_transitions = nn.ModuleList()
            self.residual_transition_bns = nn.ModuleList()
            for prev_size, next_size in zip(
                self.residual_blocks[:-1], self.residual_blocks[1:], strict=False
            ):
                self.residual_transitions.append(
                    nn.Identity()
                    if prev_size == next_size
                    else nn.Linear(prev_size, next_size)
                )
                self.residual_transition_bns.append(nn.BatchNorm1d(next_size))
            hidden_dim_for_output = self.residual_blocks[-1]
        else:
            self.residual_blocks_list = None
            self.hidden_layer = None
            self.bn2 = None
            self.residual_transitions = None
            self.residual_transition_bns = None
            hidden_dim_for_output = hd1

        self.output_layer = nn.Linear(hidden_dim_for_output, self.output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Perform model forward pass.

        z -> features (pluggable encoder) -> input block ->
        optional hidden block -> optional residual blocks -> output.
        """
        x: torch.Tensor = self.input_encoder(z)
        # Allow encoders to return (batch, ..., feat) tensors.
        if x.dim() > _FLATTEN_TO_2D_DIM_THRESHOLD:
            x = x.view(x.size(0), -1)
        x = x.to(self.dtype)

        # Input block
        x = self.input_proj(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Optional hidden block
        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = self.activation(x)
            x = self.dropout(x)

        # Optional residual stack
        if self.residual_blocks_list is not None:
            for i, block in enumerate(self.residual_blocks_list):
                x = block(x)
                # Transition to the next residual block size if needed.
                if self.residual_transitions is not None and i < len(
                    self.residual_transitions
                ):
                    trans = self.residual_transitions[i]
                    if not isinstance(trans, nn.Identity):
                        x = trans(x)
                        x = self.residual_transition_bns[i](x)
                        x = self.activation(x)
                        x = self.dropout(x)

        # Output
        x = self.output_layer(x)
        return x.flatten()
