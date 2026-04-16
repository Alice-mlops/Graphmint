"""
Alice-in-Cayleyland: attention over random-walk neighborhood tokens.

This module defines a Pilgrim-family model variant that augments the usual node
encoding with *local graph context* gathered via short random walks starting
from each input node.

High-level idea
---------------
Given an input state ``z`` (a node in a Cayley / permutation graph), we:
1) Generate ``num_walks`` random walks of length ``walk_length`` from ``z``.
2) Treat the visited states as a token set (optionally tagged with hop / walk /
   generator-id embeddings).
3) Run cross-attention where the query is the embedding of ``z`` and the keys /
   values are the walk-token embeddings.
4) Combine the attended context with the center embedding and feed the result
   through an MLP + optional residual stack (either BatchNorm residual blocks
   or Keel-style LayerNorm residual blocks).

Why random walks?
-----------------
For Cayley graphs, 1-hop neighbors are highly symmetric. Random walks provide
cheap, structured "views" of the local neighborhood and give attention a
meaningful choice among multiple contexts. Hop embeddings (walk step index) and
walk-id embeddings are used to break token exchangeability.

Important caveat (compute)
--------------------------
This model performs extra work *inside* ``forward()``:
- It generates ``num_walks * walk_length`` neighbor states per input.
- It runs the input encoder on all those states.
So the forward cost scales roughly with ``(1 + num_walks * walk_length)``.

Configuration
-------------
Required base keys (same as :class:`~pilgrim.model.AlPilgrim`):
- ``state_size`` (int)
- ``num_classes`` (int)
- ``hd1`` (int)
- ``dropout_rate`` (float)

Required graph key for neighborhood generation (permutation graphs only):
- ``generator_moves``: permutation generators as a tensor/array/list with shape
  ``(n_generators, state_size)``. Each row is a permutation ``p`` such that the
  generator action is ``dst = src[:, p]`` (i.e. ``torch.gather`` indexing).

Random-walk attention keys (all optional):
- ``alice_num_walks`` (int, default 0): number of walks per input. Set > 0 to enable.
- ``alice_walk_length`` (int, default 0): number of steps per walk.
- ``alice_include_self`` (bool, default False): include the start state as a token.
- ``alice_backtrack_mode`` (str, default "inverse"):
    - "none": no backtracking avoidance
    - "inverse": avoid choosing the inverse generator of recent steps (cheap)
    - "state": avoid revisiting the last ``alice_backtrack_memory`` states (expensive)
- ``alice_backtrack_memory`` (int, default 1): memory depth used by backtrack mode.
- ``alice_resample_attempts`` (int, default 8): max generator resampling attempts
  per step when backtracking constraints are enabled.
- ``alice_seed`` (int | None, default None): if set, makes walk sampling
  deterministic across calls (in eval mode this stabilizes predictions).

Generator selection keys (optional):
- ``alice_generator_indices`` (Sequence[int] | None): subset of generators allowed
  for the walk steps. If None, uses all.
- ``alice_max_generators`` (int | None): cap the allowed generator set size by
  sampling a subset.
- ``alice_generator_sampling`` ("fixed" | "per_forward", default "fixed"):
  when ``alice_max_generators`` is set, sample the subset once at init ("fixed")
  or re-sample it every forward ("per_forward").

Attention keys (optional):
- ``alice_attention_heads`` (int, default 4): number of MHA heads.
- ``alice_attention_dropout`` (float, default 0.0): dropout inside attention.
- ``alice_ctx_scale_init`` (float, default 0.0): initial scale for the residual
  context mix: ``h = h0 + scale * ctx``.
- ``alice_use_hop_emb`` (bool, default True)
- ``alice_use_walk_emb`` (bool, default True)
- ``alice_use_gen_emb`` (bool, default True): embed generator-id for each token
  (the generator used to reach the token state).

Residual stack keys (optional):
- ``residual_blocks`` (list[int] | None): hidden sizes for residual blocks.
- ``residual_block_type`` (str, default "mlp"):
    - "mlp": BatchNorm residual blocks (:class:`~pilgrim.model.model_blocks.ResidualBlock`)
    - "keel": Keel residual blocks (:class:`~pilgrim.model.model_blocks.KeelResidualBlock`)
    - "post_ln_alpha": :class:`~pilgrim.model.model_blocks.PostLNAlphaResidualBlock`
    - "post_ln_alpha_beta": :class:`~pilgrim.model.model_blocks.PostLNAlphaBetaResidualBlock`
- ``residual_block_kwargs`` (dict): forwarded to Keel-style residual blocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

from .model_blocks import ResidualBlock, build_node_encoder
from .model_blocks.keel_residuals import (
    KeelResidualBlock,
    PostLNAlphaBetaResidualBlock,
    PostLNAlphaResidualBlock,
)

_FLATTEN_TO_2D_DIM_THRESHOLD = 2


def _as_long_tensor(x: Any, *, device: torch.device | None = None) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    return t.long()


def _inverse_permutation(p: Sequence[int]) -> list[int]:
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[int(j)] = int(i)
    return inv


def _compute_inverse_generator_map(generator_moves: torch.Tensor) -> torch.Tensor:
    """Compute inverse generator index for each generator (or -1 if missing)."""
    moves_list = generator_moves.detach().cpu().tolist()
    idx = {tuple(m): i for i, m in enumerate(moves_list)}
    inv_map = []
    for m in moves_list:
        inv = tuple(_inverse_permutation(m))
        inv_map.append(int(idx.get(inv, -1)))
    return torch.tensor(inv_map, dtype=torch.long)


class AliceInCayleyland(nn.Module):
    """Random-walk attention Pilgrim variant (see module docstring)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.dtype: torch.dtype = config.get("model_dtype", torch.float32)
        self.state_size: int = int(config["state_size"])
        self.num_classes: int = int(config["num_classes"])
        self.hd1: int = int(config["hd1"])
        self.output_dim: int = 1

        self.residual_blocks: list[int] | None = config.get("residual_blocks")
        self.residual_block_type: Literal[
            "mlp", "keel", "post_ln_alpha", "post_ln_alpha_beta"
        ] = config.get("residual_block_type", "mlp")
        self.residual_block_kwargs: dict[str, Any] = dict(
            config.get("residual_block_kwargs", {})
        )

        # Input encoder (same pluggable mechanism as AlPilgrim).
        self.input_encoder, encoder_out_dim = build_node_encoder(config)
        self.input_proj = (
            nn.Identity()
            if int(encoder_out_dim) == self.hd1
            else nn.Linear(int(encoder_out_dim), self.hd1)
        )

        # --- Random-walk attention configuration ---
        self.num_walks: int = int(config.get("alice_num_walks", 0))
        self.walk_length: int = int(config.get("alice_walk_length", 0))
        self.include_self: bool = bool(config.get("alice_include_self"))
        self.backtrack_mode: str = str(config.get("alice_backtrack_mode", "inverse"))
        self.backtrack_memory: int = int(config.get("alice_backtrack_memory", 1))
        self.resample_attempts: int = int(config.get("alice_resample_attempts", 8))
        self.walk_seed: int | None = (
            int(config["alice_seed"]) if config.get("alice_seed") is not None else None
        )

        self.attn_heads: int = int(config.get("alice_attention_heads", 4))
        self.attn_dropout: float = float(config.get("alice_attention_dropout", 0.0))

        self.use_hop_emb: bool = bool(config.get("alice_use_hop_emb", True))
        self.use_walk_emb: bool = bool(config.get("alice_use_walk_emb", True))
        self.use_gen_emb: bool = bool(config.get("alice_use_gen_emb", True))

        ctx_scale_init = float(config.get("alice_ctx_scale_init", 0.0))
        self.ctx_scale = nn.Parameter(torch.tensor(ctx_scale_init, dtype=torch.float32))

        # Generator moves are required only if walk attention is enabled.
        generator_moves_cfg = config.get("generator_moves")
        if self.num_walks > 0 and self.walk_length > 0:
            if generator_moves_cfg is None:
                raise ValueError(
                    "AliceInCayleyland requires config['generator_moves'] when "
                    "alice_num_walks > 0 and alice_walk_length > 0."
                )
            generator_moves = _as_long_tensor(generator_moves_cfg)
            if generator_moves.ndim != 2 or generator_moves.shape[1] != self.state_size:
                raise ValueError(
                    "config['generator_moves'] must have shape "
                    f"(n_generators, state_size={self.state_size}), got {tuple(generator_moves.shape)}."
                )
        else:
            generator_moves = torch.zeros((1, self.state_size), dtype=torch.long)

        self.register_buffer("generator_moves", generator_moves, persistent=True)
        self.n_generators: int = int(self.generator_moves.shape[0])

        # Optional inverse map for cheap non-backtracking.
        inv_cfg = config.get("generator_inverse_map")
        if inv_cfg is not None:
            inv_map = _as_long_tensor(inv_cfg)
        else:
            inv_map = _compute_inverse_generator_map(self.generator_moves)
        if inv_map.ndim != 1 or inv_map.numel() != self.n_generators:
            raise ValueError(
                "generator inverse map must have shape (n_generators,), got "
                f"{tuple(inv_map.shape)} for n_generators={self.n_generators}."
            )
        self.register_buffer("generator_inverse_map", inv_map, persistent=True)

        # Allowed generator set (optionally capped).
        base_idx = config.get("alice_generator_indices")
        if base_idx is None:
            base_allowed = torch.arange(self.n_generators, dtype=torch.long)
        else:
            base_allowed = _as_long_tensor(list(base_idx))

        max_g = config.get("alice_max_generators")
        self.max_generators: int | None = (
            int(max_g) if max_g is not None and int(max_g) > 0 else None
        )
        self.generator_sampling: str = str(
            config.get("alice_generator_sampling", "fixed")
        ).strip()

        if self.max_generators is not None and self.generator_sampling not in {
            "fixed",
            "per_forward",
        }:
            raise ValueError(
                "alice_generator_sampling must be 'fixed' or 'per_forward'."
            )

        if self.max_generators is not None and self.generator_sampling == "fixed":
            base_allowed = self._sample_subset(base_allowed, self.max_generators)

        self.register_buffer("base_allowed_generators", base_allowed, persistent=True)

        # Attention + token tags.
        if self.hd1 % self.attn_heads != 0:
            raise ValueError(
                f"hd1={self.hd1} must be divisible by alice_attention_heads={self.attn_heads}."
            )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hd1,
            num_heads=self.attn_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )

        if self.use_hop_emb:
            max_hop = self.walk_length + (1 if self.include_self else 0)
            self.hop_emb = nn.Embedding(max_hop + 1, self.hd1)
        else:
            self.hop_emb = None

        if self.use_walk_emb:
            self.walk_emb = nn.Embedding(max(1, self.num_walks), self.hd1)
        else:
            self.walk_emb = None

        if self.use_gen_emb:
            # +1 to reserve an extra index for "start of walk".
            self.gen_emb = nn.Embedding(self.n_generators + 1, self.hd1)
            self.gen_start_id = int(self.n_generators)
        else:
            self.gen_emb = None
            self.gen_start_id = -1

        # --- Post-attention MLP stack (mirrors AlPilgrim) ---
        self.bn1 = nn.BatchNorm1d(self.hd1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(config["dropout_rate"]))

        self._init_residual_stack(config)

    def _sample_subset(self, allowed: torch.Tensor, k: int) -> torch.Tensor:
        allowed = allowed.reshape(-1)
        if k >= allowed.numel():
            return allowed
        if self.walk_seed is None:
            perm = torch.randperm(allowed.numel(), device=allowed.device)[:k]
        else:
            g = torch.Generator(device=allowed.device)
            g.manual_seed(int(self.walk_seed))
            perm = torch.randperm(allowed.numel(), generator=g, device=allowed.device)[
                :k
            ]
        return allowed[perm]

    def _select_allowed_generators(self, *, device: torch.device) -> torch.Tensor:
        allowed = self.base_allowed_generators.to(device)
        if self.max_generators is None or self.generator_sampling != "per_forward":
            return allowed
        return self._sample_subset(allowed, int(self.max_generators))

    def _init_residual_stack(self, config: dict[str, Any]) -> None:
        """Build the optional hidden + residual stack."""
        block_type = str(self.residual_block_type).strip().lower()
        dropout_rate = float(config.get("dropout_rate", 0.1))

        if block_type == "mlp":
            block_ctor = lambda dim: ResidualBlock(  # noqa: E731
                dim, dropout_rate=dropout_rate
            )
        elif block_type == "keel":
            block_ctor = lambda dim: KeelResidualBlock(  # noqa: E731
                dim, dropout_rate=dropout_rate, **self.residual_block_kwargs
            )
        elif block_type == "post_ln_alpha":
            block_ctor = lambda dim: PostLNAlphaResidualBlock(  # noqa: E731
                dim, dropout_rate=dropout_rate, **self.residual_block_kwargs
            )
        elif block_type == "post_ln_alpha_beta":
            block_ctor = lambda dim: PostLNAlphaBetaResidualBlock(  # noqa: E731
                dim, dropout_rate=dropout_rate, **self.residual_block_kwargs
            )
        else:
            raise ValueError(f"Unknown residual_block_type: {self.residual_block_type}")

        if self.residual_blocks:
            self.residual_blocks_list = nn.ModuleList([
                block_ctor(int(residual_size)) for residual_size in self.residual_blocks
            ])
            first_size = int(self.residual_blocks[0])
            self.hidden_layer = (
                nn.Identity()
                if self.hd1 == first_size
                else nn.Linear(self.hd1, first_size)
            )
            self.bn2 = nn.BatchNorm1d(first_size)
            self.residual_transitions = nn.ModuleList()
            self.residual_transition_bns = nn.ModuleList()
            for prev_size, next_size in zip(
                self.residual_blocks[:-1], self.residual_blocks[1:], strict=False
            ):
                prev_size_i = int(prev_size)
                next_size_i = int(next_size)
                self.residual_transitions.append(
                    nn.Identity()
                    if prev_size_i == next_size_i
                    else nn.Linear(prev_size_i, next_size_i)
                )
                self.residual_transition_bns.append(nn.BatchNorm1d(next_size_i))
            hidden_dim_for_output = int(self.residual_blocks[-1])
        else:
            self.residual_blocks_list = None
            self.hidden_layer = None
            self.bn2 = None
            self.residual_transitions = None
            self.residual_transition_bns = None
            hidden_dim_for_output = int(self.hd1)

        self.output_layer = nn.Linear(hidden_dim_for_output, self.output_dim)

    def _encode_states_2d(self, z: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.input_encoder(z)
        if x.dim() > _FLATTEN_TO_2D_DIM_THRESHOLD:
            x = x.view(x.size(0), -1)
        x = x.to(self.dtype)
        x = self.input_proj(x)
        return x

    def _apply_generator_moves(
        self, states: torch.Tensor, gen_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-row generators: dst[b] = states[b, generator_moves[gen_ids[b]]]."""
        moves = self.generator_moves.index_select(0, gen_ids.reshape(-1))
        return torch.gather(states, 1, moves)

    def _make_rng(self, *, device: torch.device) -> torch.Generator | None:
        if self.walk_seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(int(self.walk_seed))
        return g

    def _sample_walk_tokens(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random-walk token states and tag ids.

        Args:
            z: Input decoded states, shape ``(B, state_size)``.

        Returns:
            Tuple ``(tokens_z, hop_ids, walk_ids, gen_ids)`` where:
            - ``tokens_z`` has shape ``(B, T, state_size)``
            - ``hop_ids`` has shape ``(B, T)``
            - ``walk_ids`` has shape ``(B, T)``
            - ``gen_ids`` has shape ``(B, T)`` (generator used to reach the token)

        """
        if self.num_walks <= 0 or self.walk_length <= 0:
            device = z.device
            empty = torch.empty((z.size(0), 0), device=device, dtype=torch.long)
            empty_z = torch.empty(
                (z.size(0), 0, self.state_size), device=device, dtype=z.dtype
            )
            return empty_z, empty, empty, empty

        if z.ndim == 1:
            z = z.unsqueeze(0)

        bsz = int(z.size(0))
        device = z.device
        w = int(self.num_walks)
        l = int(self.walk_length)
        include_self = bool(self.include_self)

        allowed = self._select_allowed_generators(device=device)
        if allowed.numel() == 0:
            raise ValueError(
                "No generators available for random walks (empty allowed set)."
            )

        g = self._make_rng(device=device)

        # Repeat each input state `num_walks` times: (B*W, state_size).
        cur = (
            z
            .unsqueeze(1)
            .expand(bsz, w, self.state_size)
            .reshape(bsz * w, self.state_size)
        )

        steps_to_store = l + 1 if include_self else l
        walk_states = torch.empty(
            (bsz * w, steps_to_store, self.state_size),
            device=device,
            dtype=cur.dtype,
        )
        walk_gen_ids = torch.empty(
            (bsz * w, steps_to_store),
            device=device,
            dtype=torch.long,
        )

        if include_self:
            walk_states[:, 0, :] = cur
            if self.gen_start_id >= 0:
                walk_gen_ids[:, 0] = int(self.gen_start_id)
            else:
                walk_gen_ids[:, 0] = 0

        # Backtracking helpers.
        backtrack_mode = str(self.backtrack_mode).strip().lower()
        mem = max(0, int(self.backtrack_memory))
        attempts = max(1, int(self.resample_attempts))

        gen_hist: torch.Tensor | None = None
        state_hist: torch.Tensor | None = None

        if backtrack_mode == "inverse" and mem > 0:
            gen_hist = torch.full(
                (bsz * w, mem),
                -1,
                device=device,
                dtype=torch.long,
            )
        elif backtrack_mode == "state" and mem > 0:
            # Store previously visited states (excluding the current state). Use
            # -1 sentinel values so comparisons never match before the history
            # is populated (state values are non-negative for permutations).
            state_hist = torch.full(
                (bsz * w, mem, self.state_size),
                -1,
                device=device,
                dtype=cur.dtype,
            )
        elif backtrack_mode not in {"none", "inverse", "state"}:
            raise ValueError(
                "alice_backtrack_mode must be one of: 'none', 'inverse', 'state'."
            )

        def _sample_gen_ids(n: int) -> torch.Tensor:
            idx = torch.randint(0, allowed.numel(), (n,), device=device, generator=g)
            return allowed[idx]

        def _inverse_block_mask(gen_ids: torch.Tensor) -> torch.Tensor:
            assert gen_hist is not None
            inv_hist = self.generator_inverse_map.index_select(
                0, gen_hist.clamp(min=0).reshape(-1)
            )
            inv_hist = inv_hist.view(gen_hist.shape)
            # gen_hist contains -1 for "no history" rows; their inverse is undefined.
            inv_hist = torch.where(
                gen_hist >= 0, inv_hist, torch.full_like(inv_hist, -2)
            )
            return (gen_ids.unsqueeze(1) == inv_hist).any(dim=1)

        def _state_block_mask(next_states: torch.Tensor) -> torch.Tensor:
            assert state_hist is not None
            eq = (next_states.unsqueeze(1) == state_hist).all(dim=2)
            return eq.any(dim=1)

        # Main loop: generate `l` steps.
        cur_step_states = cur
        for step in range(1, l + 1):
            gen_ids = _sample_gen_ids(cur_step_states.size(0))

            if gen_hist is not None:
                # Resample until the inverse-of-recent constraint is satisfied.
                for _ in range(attempts):
                    bad = _inverse_block_mask(gen_ids)
                    if not bool(bad.any()):
                        break
                    gen_ids = gen_ids.clone()
                    gen_ids[bad] = _sample_gen_ids(int(bad.sum().item()))

            next_states = self._apply_generator_moves(cur_step_states, gen_ids)

            if state_hist is not None:
                # Resample generators if we revisit a recent state.
                for _ in range(attempts):
                    bad = _state_block_mask(next_states)
                    if not bool(bad.any()):
                        break
                    gen_ids = gen_ids.clone()
                    gen_ids[bad] = _sample_gen_ids(int(bad.sum().item()))
                    next_states = self._apply_generator_moves(cur_step_states, gen_ids)

            store_idx = step if include_self else (step - 1)
            walk_states[:, store_idx, :] = next_states
            walk_gen_ids[:, store_idx] = gen_ids

            # Update histories.
            if gen_hist is not None:
                gen_hist = torch.roll(gen_hist, shifts=1, dims=1)
                gen_hist[:, 0] = gen_ids
            if state_hist is not None:
                state_hist = torch.roll(state_hist, shifts=1, dims=1)
                state_hist[:, 0, :] = cur_step_states

            cur_step_states = next_states

        # Reshape to per-batch token table.
        # (B*W, steps, state) -> (B, W*steps, state)
        tokens_z = walk_states.view(bsz, w * steps_to_store, self.state_size)
        tokens_gen = walk_gen_ids.view(bsz, w * steps_to_store)

        # Token tags.
        hop_base = torch.arange(
            0 if include_self else 1,
            (l + 1),
            device=device,
            dtype=torch.long,
        )
        hop_ids = hop_base.repeat(w).view(1, -1).expand(bsz, -1)

        walk_ids = (
            torch
            .arange(w, device=device, dtype=torch.long)
            .repeat_interleave(steps_to_store)
            .view(1, -1)
            .expand(bsz, -1)
        )

        return tokens_z, hop_ids, walk_ids, tokens_gen

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Decoded input states of shape ``(batch, state_size)``.

        Returns:
            Predicted distances as a 1D tensor of shape ``(batch,)``.

        """
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.long()

        # Center embedding.
        h0 = self._encode_states_2d(z)

        # Random-walk attention context (optional).
        if self.num_walks > 0 and self.walk_length > 0:
            tokens_z, hop_ids, walk_ids, gen_ids = self._sample_walk_tokens(z)
            if tokens_z.size(1) > 0:
                flat_tokens = tokens_z.reshape(-1, self.state_size)
                tok = self._encode_states_2d(flat_tokens).view(
                    z.size(0), tokens_z.size(1), self.hd1
                )

                if self.hop_emb is not None:
                    tok = tok + self.hop_emb(hop_ids)
                if self.walk_emb is not None:
                    tok = tok + self.walk_emb(walk_ids)
                if self.gen_emb is not None:
                    tok = tok + self.gen_emb(
                        gen_ids.clamp(min=0, max=self.n_generators)
                    )

                ctx, _ = self.attn(h0.unsqueeze(1), tok, tok, need_weights=False)
                h0 = h0 + self.ctx_scale.to(h0.dtype) * ctx.squeeze(1)

        # MLP stack (same ordering as AlPilgrim).
        x = h0.to(self.dtype)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.hidden_layer is not None:
            x = self.hidden_layer(x)
            x = self.bn2(x)
            x = self.activation(x)
            x = self.dropout(x)

        if self.residual_blocks_list is not None:
            for i, block in enumerate(self.residual_blocks_list):
                x = block(x)
                if self.residual_transitions is not None and i < len(
                    self.residual_transitions
                ):
                    trans = self.residual_transitions[i]
                    if not isinstance(trans, nn.Identity):
                        x = trans(x)
                        x = self.residual_transition_bns[i](x)
                        x = self.activation(x)
                        x = self.dropout(x)

        x = self.output_layer(x)
        return x.flatten()
