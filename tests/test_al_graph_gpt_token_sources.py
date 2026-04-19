# Tests AlGraphGPT neighborhood token-source modes in the Graphmint package.
"""Smoke tests for AlGraphGPT token-source selection."""

from __future__ import annotations

import pytest
import torch
from pilgrim.model.al_graph_gpt import AlGraphGPT
from pilgrim.schemas import AlGraphGPTConfig


def _build_base_config(**overrides: object) -> AlGraphGPTConfig:
    """
    Build a minimal valid AlGraphGPT config for token-source tests.

    Returns:
        Validated config instance for a tiny AlGraphGPT model.

    """
    config = {
        "state_size": 4,
        "num_classes": 4,
        "dropout_rate": 0.0,
        "model_dtype": torch.float32,
        "input_encoder_type": "lehmer-breakpoints",
        "input_encoder_out_dim": 16,
        "algraphgpt_d_model": 16,
        "algraphgpt_num_layers": 2,
        "algraphgpt_num_heads": 4,
        "algraphgpt_attn_dropout": 0.0,
        "algraphgpt_resid_dropout": 0.0,
        "algraphgpt_ffn_mult": 2.0,
        "algraphgpt_ffn_dropout": 0.0,
        "generator_moves": torch.tensor(
            [
                [1, 0, 2, 3],
                [0, 2, 1, 3],
                [0, 1, 3, 2],
            ],
            dtype=torch.long,
        ),
    }
    config.update(overrides)
    return AlGraphGPTConfig(**config)


def test_one_hop_token_source_builds_self_plus_neighbors() -> None:
    """One-hop mode should prepend one self token and all selected neighbors."""
    cfg = _build_base_config(
        alice_token_source="one_hop",
        alice_generator_indices=(0, 2),
    )
    model = AlGraphGPT(cfg).eval()

    z = torch.tensor([[0, 1, 2, 3], [2, 0, 3, 1]], dtype=torch.long)
    tokens_z, hop_ids, walk_ids, gen_ids = model._build_one_hop_tokens(z)  # noqa: SLF001

    assert tokens_z.shape == (2, 3, 4)
    assert hop_ids.tolist() == [[0, 1, 1], [0, 1, 1]]
    assert walk_ids.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert gen_ids.tolist() == [[3, 0, 2], [3, 0, 2]]
    assert torch.equal(tokens_z[:, 0, :], z)


def test_one_hop_token_source_runs_forward() -> None:
    """AlGraphGPT forward should work when exact one-hop tokens are selected."""
    cfg = _build_base_config(
        alice_token_source="one_hop",
        alice_max_generators=2,
        alice_generator_sampling="fixed",
    )
    model = AlGraphGPT(cfg).eval()

    z = torch.tensor([[2, 0, 3, 1], [0, 1, 2, 3]], dtype=torch.long)
    out = model(z)

    assert out.shape == (2,)


def test_one_hop_config_requires_generator_moves() -> None:
    """One-hop mode should reject configs without generator permutations."""
    with pytest.raises(ValueError, match="generator_moves"):
        _build_base_config(
            alice_token_source="one_hop",
            generator_moves=None,
        )
