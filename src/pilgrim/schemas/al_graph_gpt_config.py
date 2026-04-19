"""Pydantic configuration schema for the AlGraphGPT model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn


class AlGraphGPTConfig(BaseModel):
    """Validated configuration for :class:`pilgrim.model.AlGraphGPT`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Base model / encoder settings.
    state_size: int = Field(..., ge=1)
    num_classes: int = Field(..., ge=1)
    dropout_rate: float = Field(0.1, ge=0.0, le=1.0)
    model_dtype: torch.dtype = torch.float32
    z_add: int = 0

    input_encoder: nn.Module | None = None
    input_encoder_out_dim: int | None = Field(default=None, ge=1)
    input_encoder_type: Literal[
        "onehot_linear",
        "embedding_flatten",
        "lehmer",
        "lehmer-breakpoints",
        "megaminx",
    ] = "onehot_linear"
    embedding_dim: int | None = Field(default=None, ge=1)
    megaminx_embedding_dim: int | None = Field(default=None, ge=1)
    megaminx_num_faces: int = Field(12, ge=1)
    megaminx_use_inverse: bool = True
    megaminx_use_graph_breakpoints: bool = True

    # Transformer settings.
    algraphgpt_d_model: int = Field(..., ge=1)
    algraphgpt_num_layers: int = Field(4, ge=2, le=32)
    algraphgpt_num_heads: int = Field(4, ge=1)
    algraphgpt_attn_dropout: float = Field(0.0, ge=0.0, le=1.0)
    algraphgpt_resid_dropout: float = Field(0.1, ge=0.0, le=1.0)
    algraphgpt_ffn_mult: float = Field(4.0, gt=0.0)
    algraphgpt_ffn_dropout: float = Field(0.1, ge=0.0, le=1.0)
    algraphgpt_activation: Literal["gelu", "silu"] = "gelu"
    algraphgpt_norm_position: Literal["pre", "post"] = "pre"
    algraphgpt_norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    algraphgpt_norm_eps: float = Field(1e-5, gt=0.0)
    algraphgpt_rezero_init: float | None = None
    algraphgpt_output_dim: int = Field(1, ge=1)
    algraphgpt_aux_output_dim: int | None = Field(default=None, ge=1)

    # Neighborhood-token settings (Alice-compatible).
    generator_moves: torch.Tensor | Sequence[Sequence[int]] | None = None
    generator_inverse_map: torch.Tensor | Sequence[int] | None = None

    alice_token_source: Literal["random_walk", "one_hop"] = "random_walk"
    alice_num_walks: int = Field(0, ge=0)
    alice_walk_length: int = Field(0, ge=0)
    alice_include_self: bool = False
    alice_backtrack_mode: Literal["none", "inverse", "state"] = "inverse"
    alice_backtrack_memory: int = Field(1, ge=0)
    alice_resample_attempts: int = Field(8, ge=1)
    alice_seed: int | None = None

    alice_generator_indices: Sequence[int] | None = None
    alice_max_generators: int | None = Field(default=None, ge=1)
    alice_generator_sampling: Literal["fixed", "per_forward"] = "fixed"

    alice_use_hop_emb: bool = True
    alice_use_walk_emb: bool = True
    alice_use_gen_emb: bool = True

    @model_validator(mode="after")
    def _validate_fields(self) -> AlGraphGPTConfig:
        if self.algraphgpt_d_model % self.algraphgpt_num_heads != 0:
            raise ValueError(
                "algraphgpt_d_model must be divisible by algraphgpt_num_heads."
            )

        if (
            self.input_encoder is None
            and self.input_encoder_type == "embedding_flatten"
            and self.embedding_dim is None
        ):
            raise ValueError(
                "embedding_dim is required for embedding-based input encoders."
            )

        if (
            self.input_encoder is None
            and self.input_encoder_type == "megaminx"
            and self.state_size % self.megaminx_num_faces != 0
        ):
            raise ValueError(
                "state_size must be divisible by megaminx_num_faces when "
                "input_encoder_type='megaminx'."
            )

        if self.alice_token_source == "one_hop" and self.generator_moves is None:
            raise ValueError(
                "generator_moves is required when alice_token_source='one_hop'."
            )

        if (
            self.alice_token_source == "random_walk"
            and self.alice_num_walks > 0
            and self.alice_walk_length > 0
            and self.generator_moves is None
        ):
            raise ValueError(
                "generator_moves is required when alice_num_walks and "
                "alice_walk_length are both positive."
            )

        return self

    def to_encoder_config(self) -> dict[str, Any]:
        """Return a dict payload compatible with ``build_node_encoder``."""
        cfg = self.model_dump()
        cfg["hd1"] = int(self.algraphgpt_d_model)
        cfg["dtype"] = self.model_dtype
        return cfg
