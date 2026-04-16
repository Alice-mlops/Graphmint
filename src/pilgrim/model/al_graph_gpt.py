"""AlGraphGPT model with vectorized random-walk neighborhood sampling."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from ..schemas.al_graph_gpt_config import AlGraphGPTConfig
from .model_blocks import build_node_encoder
from .model_blocks.al_graph_gpt_blocks import AlGraphGPTLayer, make_norm

_FLATTEN_TO_2D_DIM_THRESHOLD = 2
_GENERATOR_MOVES_EXPECTED_NDIM = 2


def _as_long_tensor(x: Any, *, device: torch.device | None = None) -> torch.Tensor:
    """
    Convert an input object to ``torch.long`` tensor.

    Args:
        x: Input data convertible to a tensor.
        device: Optional target device for the returned tensor.

    Returns:
        Tensor with dtype ``torch.long``.

    """
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    return t.long()


def _inverse_permutation(p: Sequence[int]) -> list[int]:
    """
    Compute inverse permutation indices.

    Args:
        p: Permutation indices.

    Returns:
        Inverse permutation as a Python list.

    """
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[int(j)] = int(i)
    return inv


def _compute_inverse_generator_map(generator_moves: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse generator ids for a set of permutation generators.

    Args:
        generator_moves: Tensor of shape ``(n_generators, state_size)`` where each
            row is a permutation move.

    Returns:
        Tensor of shape ``(n_generators,)``. Entry ``i`` stores the generator id of
        the inverse move of generator ``i``, or ``-1`` if the inverse is missing.

    """
    moves_list = generator_moves.detach().cpu().tolist()
    idx = {tuple(m): i for i, m in enumerate(moves_list)}
    inv_map = []
    for m in moves_list:
        inv = tuple(_inverse_permutation(m))
        inv_map.append(int(idx.get(inv, -1)))
    return torch.tensor(inv_map, dtype=torch.long)


class AlGraphGPT(nn.Module):
    """Stacked center-query graph Transformer with random-walk token memory."""

    def __init__(self, config: AlGraphGPTConfig) -> None:
        """
        Initialize AlGraphGPT.

        Args:
            config: Validated model configuration.

        """
        super().__init__()
        self.config: AlGraphGPTConfig = config
        self.dtype: torch.dtype = config.model_dtype
        self.state_size: int = int(config.state_size)
        self.num_classes: int = int(config.num_classes)
        self.d_model: int = int(config.algraphgpt_d_model)
        self.output_dim: int = 1

        self.num_layers: int = int(config.algraphgpt_num_layers)
        self.norm_position: str = str(config.algraphgpt_norm_position).strip().lower()
        self.norm_type: str = str(config.algraphgpt_norm_type).strip().lower()
        self.norm_eps: float = float(config.algraphgpt_norm_eps)

        encoder_cfg = config.to_encoder_config()
        self.input_encoder, encoder_out_dim = build_node_encoder(encoder_cfg)
        self.input_proj = (
            nn.Identity()
            if int(encoder_out_dim) == self.d_model
            else nn.Linear(int(encoder_out_dim), self.d_model)
        )

        # --- Random-walk neighborhood configuration ---
        self.num_walks: int = int(config.alice_num_walks)
        self.walk_length: int = int(config.alice_walk_length)
        self.include_self: bool = bool(config.alice_include_self)
        self.backtrack_mode: str = str(config.alice_backtrack_mode).strip().lower()
        self.backtrack_memory: int = int(config.alice_backtrack_memory)
        self.resample_attempts: int = int(config.alice_resample_attempts)
        self.walk_seed: int | None = (
            int(config.alice_seed) if config.alice_seed is not None else None
        )

        self.use_hop_emb: bool = bool(config.alice_use_hop_emb)
        self.use_walk_emb: bool = bool(config.alice_use_walk_emb)
        self.use_gen_emb: bool = bool(config.alice_use_gen_emb)

        generator_moves_cfg = config.generator_moves
        if self.num_walks > 0 and self.walk_length > 0:
            if generator_moves_cfg is None:
                raise ValueError(
                    "AlGraphGPT requires generator_moves when alice_num_walks > 0 "
                    "and alice_walk_length > 0."
                )
            generator_moves = _as_long_tensor(generator_moves_cfg)
            if (
                generator_moves.ndim != _GENERATOR_MOVES_EXPECTED_NDIM
                or generator_moves.shape[1] != self.state_size
            ):
                raise ValueError(
                    "generator_moves must have shape "
                    f"(n_generators, state_size={self.state_size}), got "
                    f"{tuple(generator_moves.shape)}."
                )
        else:
            generator_moves = torch.zeros((1, self.state_size), dtype=torch.long)

        self.register_buffer("generator_moves", generator_moves, persistent=True)
        self.n_generators: int = int(self.generator_moves.shape[0])

        inv_cfg = config.generator_inverse_map
        if inv_cfg is not None:
            inv_map = _as_long_tensor(inv_cfg)
        else:
            inv_map = _compute_inverse_generator_map(self.generator_moves)
        if inv_map.ndim != 1 or inv_map.numel() != self.n_generators:
            raise ValueError(
                "generator_inverse_map must have shape (n_generators,), got "
                f"{tuple(inv_map.shape)} for n_generators={self.n_generators}."
            )
        self.register_buffer("generator_inverse_map", inv_map, persistent=True)

        base_idx = config.alice_generator_indices
        if base_idx is None:
            base_allowed = torch.arange(self.n_generators, dtype=torch.long)
        else:
            base_allowed = _as_long_tensor(list(base_idx))

        max_g = config.alice_max_generators
        self.max_generators: int | None = (
            int(max_g) if max_g is not None and int(max_g) > 0 else None
        )
        self.generator_sampling: str = (
            str(config.alice_generator_sampling).strip().lower()
        )
        if self.max_generators is not None and self.generator_sampling == "fixed":
            base_allowed = self._sample_subset(base_allowed, self.max_generators)
        self.register_buffer("base_allowed_generators", base_allowed, persistent=True)

        if self.use_hop_emb:
            max_hop = self.walk_length + (1 if self.include_self else 0)
            self.hop_emb = nn.Embedding(max_hop + 1, self.d_model)
        else:
            self.hop_emb = None

        if self.use_walk_emb:
            self.walk_emb = nn.Embedding(max(1, self.num_walks), self.d_model)
        else:
            self.walk_emb = None

        if self.use_gen_emb:
            self.gen_emb = nn.Embedding(self.n_generators + 1, self.d_model)
            self.gen_start_id = int(self.n_generators)
        else:
            self.gen_emb = None
            self.gen_start_id = -1

        # --- Stacked center-query Transformer blocks ---
        self.layers = nn.ModuleList([
            AlGraphGPTLayer(config) for _ in range(self.num_layers)
        ])
        self._operation_profile_enabled: bool = False
        self._operation_profile_total_s: dict[str, float] = defaultdict(float)
        self._operation_profile_calls: dict[str, int] = defaultdict(int)
        for idx, layer in enumerate(self.layers):
            layer.set_operation_profiler(
                self._record_operation_timing,
                prefix=f"layer/{int(idx)}",
            )
            layer.enable_operation_profiling(False)
        self.final_norm = make_norm(self.norm_type, self.d_model, self.norm_eps)
        self.output_layer = nn.Linear(self.d_model, self.output_dim)

    def enable_operation_profiling(self, enabled: bool, *, reset: bool = False) -> None:
        """
        Enable or disable semantic per-operation timing collection.

        Args:
            enabled: ``True`` to enable timing collection.
            reset: Whether to clear previously collected stats.

        Returns:
            None.

        """
        self._operation_profile_enabled = bool(enabled)
        if bool(reset):
            self.reset_operation_profile()
        for layer in self.layers:
            layer.enable_operation_profiling(bool(enabled))

    def reset_operation_profile(self) -> None:
        """
        Reset accumulated operation-timing statistics.

        Returns:
            None.

        """
        self._operation_profile_total_s = defaultdict(float)
        self._operation_profile_calls = defaultdict(int)

    def get_operation_profile(
        self, *, reset: bool = False
    ) -> dict[str, dict[str, float]]:
        """
        Return collected operation timing metrics.

        Args:
            reset: Whether to clear internal counters after reading.

        Returns:
            Mapping ``operation_name -> {total_s, calls, mean_ms}``.

        """
        out: dict[str, dict[str, float]] = {}
        names = set(self._operation_profile_total_s.keys()) | set(
            self._operation_profile_calls.keys()
        )
        for name in sorted(names):
            total_s = float(self._operation_profile_total_s.get(name, 0.0))
            calls = int(self._operation_profile_calls.get(name, 0))
            mean_ms = (total_s / calls * 1000.0) if calls > 0 else 0.0
            out[str(name)] = {
                "total_s": total_s,
                "calls": float(calls),
                "mean_ms": mean_ms,
            }
        if bool(reset):
            self.reset_operation_profile()
        return out

    def _record_operation_timing(self, name: str, elapsed_s: float) -> None:
        """
        Record one operation timing sample.

        Args:
            name: Semantic operation name.
            elapsed_s: Elapsed seconds.

        Returns:
            None.

        """
        if not self._operation_profile_enabled:
            return
        key = str(name)
        self._operation_profile_total_s[key] += float(elapsed_s)
        self._operation_profile_calls[key] += 1

    @staticmethod
    def _sync_if_cuda(ref: torch.Tensor | torch.device | str | None) -> None:
        """
        Synchronize CUDA stream for accurate per-operation timing.

        Args:
            ref: Tensor or device-like object.

        Returns:
            None.

        """
        if not torch.cuda.is_available():
            return
        if isinstance(ref, torch.Tensor):
            if ref.is_cuda:
                torch.cuda.synchronize()
            return
        if ref is not None and "cuda" in str(ref):
            torch.cuda.synchronize()

    def _op_timer_start(self, ref: torch.Tensor | torch.device | str) -> float | None:
        """
        Start synchronized timer when operation profiling is enabled.

        Args:
            ref: Tensor or device-like reference.

        Returns:
            ``perf_counter`` timestamp or ``None`` when disabled.

        """
        if not self._operation_profile_enabled:
            return None
        self._sync_if_cuda(ref)
        return time.perf_counter()

    def _op_timer_stop(
        self,
        name: str,
        start_time: float | None,
        ref: torch.Tensor | torch.device | str,
    ) -> None:
        """
        Stop timer and add sample to operation profile statistics.

        Args:
            name: Operation name.
            start_time: Start timestamp from :meth:`_op_timer_start`.
            ref: Tensor or device-like reference.

        Returns:
            None.

        """
        if start_time is None:
            return
        self._sync_if_cuda(ref)
        self._record_operation_timing(str(name), time.perf_counter() - start_time)

    def _sample_subset(self, allowed: torch.Tensor, k: int) -> torch.Tensor:
        """
        Sample up to ``k`` generator ids from ``allowed`` without replacement.

        Args:
            allowed: Candidate generator ids, shape ``(n,)``.
            k: Requested subset size.

        Returns:
            Tensor with sampled generator ids.

        """
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
        """
        Resolve generator ids allowed for walk sampling.

        Args:
            device: Device where the returned tensor should live.

        Returns:
            Tensor of allowed generator ids on ``device``.

        """
        allowed = self.base_allowed_generators.to(device)
        if self.max_generators is None or self.generator_sampling != "per_forward":
            return allowed
        return self._sample_subset(allowed, int(self.max_generators))

    def _encode_states_2d(
        self, z: torch.Tensor, *, op_name: str = "encode_states_2d"
    ) -> torch.Tensor:
        """
        Encode graph states into dense token vectors.

        Args:
            z: Input states, shape ``(batch, state_size)`` or tensor that encodes to
                shape ``(batch, ..., feat)`` via the configured input encoder.
            op_name: Semantic operation name used for optional timing.

        Returns:
            Encoded tensor of shape ``(batch, d_model)``.

        """
        t0 = self._op_timer_start(z)
        x: torch.Tensor = self.input_encoder(z)
        if x.dim() > _FLATTEN_TO_2D_DIM_THRESHOLD:
            x = x.view(x.size(0), -1)
        x = x.to(self.dtype)
        x = self.input_proj(x)
        self._op_timer_stop(op_name, t0, x)
        return x

    def _apply_generator_moves(
        self,
        states: torch.Tensor,
        gen_ids: torch.Tensor,
        *,
        op_name: str = "walk/apply_generator_moves",
    ) -> torch.Tensor:
        """
        Apply per-row permutation generators to a batch of states.

        Args:
            states: State tensor of shape ``(batch, state_size)``.
            gen_ids: Generator ids of shape ``(batch,)``.
            op_name: Semantic operation name used for optional timing.

        Returns:
            Next-state tensor of shape ``(batch, state_size)`` where row ``b`` is
            ``states[b, generator_moves[gen_ids[b]]]``.

        """
        t0 = self._op_timer_start(states)
        moves = self.generator_moves.index_select(0, gen_ids.reshape(-1))
        out = torch.gather(states, 1, moves)
        self._op_timer_stop(op_name, t0, out)
        return out

    def _make_rng(self, *, device: torch.device) -> torch.Generator | None:
        """
        Build a torch RNG for deterministic walk sampling when seeded.

        Args:
            device: Device for the RNG.

        Returns:
            ``torch.Generator`` if ``alice_seed`` is set, else ``None``.

        """
        if self.walk_seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(int(self.walk_seed))
        return g

    def _sample_walk_tokens(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random-walk token states and token metadata.

        This implementation is vectorized over the full ``batch * num_walks``
        dimension. The only explicit loops are over walk steps and bounded
        re-sampling attempts for backtracking constraints.

        Args:
            z: Input decoded states, shape ``(batch, state_size)``.

        Returns:
            Tuple ``(tokens_z, hop_ids, walk_ids, gen_ids)`` where:
            - ``tokens_z`` has shape ``(batch, n_tokens, state_size)``,
            - ``hop_ids`` has shape ``(batch, n_tokens)``,
            - ``walk_ids`` has shape ``(batch, n_tokens)``,
            - ``gen_ids`` has shape ``(batch, n_tokens)``.

        """
        t0_total = self._op_timer_start(z)
        if self.num_walks <= 0 or self.walk_length <= 0:
            device = z.device
            empty = torch.empty((z.size(0), 0), device=device, dtype=torch.long)
            empty_z = torch.empty(
                (z.size(0), 0, self.state_size), device=device, dtype=z.dtype
            )
            self._op_timer_stop("walk/sample_tokens_total", t0_total, z)
            return empty_z, empty, empty, empty

        if z.ndim == 1:
            z = z.unsqueeze(0)

        batch_size = int(z.size(0))
        device = z.device
        num_walks = int(self.num_walks)
        walk_length = int(self.walk_length)
        include_self = bool(self.include_self)

        t0_select_allowed = self._op_timer_start(z)
        allowed = self._select_allowed_generators(device=device)
        self._op_timer_stop("walk/select_allowed_generators", t0_select_allowed, z)
        if allowed.numel() == 0:
            raise ValueError("No generators available for random walks.")

        rng = self._make_rng(device=device)

        cur_states = z.unsqueeze(1).expand(batch_size, num_walks, self.state_size)
        cur_states = cur_states.reshape(batch_size * num_walks, self.state_size)

        steps_to_store = walk_length + 1 if include_self else walk_length
        walk_states = torch.empty(
            (batch_size * num_walks, steps_to_store, self.state_size),
            device=device,
            dtype=cur_states.dtype,
        )
        walk_gen_ids = torch.empty(
            (batch_size * num_walks, steps_to_store), device=device, dtype=torch.long
        )

        if include_self:
            walk_states[:, 0, :] = cur_states
            walk_gen_ids[:, 0] = int(self.gen_start_id) if self.gen_start_id >= 0 else 0

        mem = max(0, int(self.backtrack_memory))
        attempts = max(1, int(self.resample_attempts))
        gen_hist: torch.Tensor | None = None
        state_hist: torch.Tensor | None = None

        if self.backtrack_mode == "inverse" and mem > 0:
            gen_hist = torch.full(
                (batch_size * num_walks, mem), -1, device=device, dtype=torch.long
            )
        elif self.backtrack_mode == "state" and mem > 0:
            state_hist = torch.full(
                (batch_size * num_walks, mem, self.state_size),
                -1,
                device=device,
                dtype=cur_states.dtype,
            )
        elif self.backtrack_mode not in {"none", "inverse", "state"}:
            raise ValueError(
                "alice_backtrack_mode must be one of: 'none', 'inverse', 'state'."
            )

        def _sample_gen_ids(n: int) -> torch.Tensor:
            idx = torch.randint(0, allowed.numel(), (n,), device=device, generator=rng)
            return allowed[idx]

        def _inverse_block_mask(gen_ids: torch.Tensor) -> torch.Tensor:
            assert gen_hist is not None
            inv_hist = self.generator_inverse_map.index_select(
                0, gen_hist.clamp(min=0).reshape(-1)
            )
            inv_hist = inv_hist.view(gen_hist.shape)
            inv_hist = torch.where(
                gen_hist >= 0, inv_hist, torch.full_like(inv_hist, -2)
            )
            return (gen_ids.unsqueeze(1) == inv_hist).any(dim=1)

        def _state_block_mask(next_states: torch.Tensor) -> torch.Tensor:
            assert state_hist is not None
            eq = (next_states.unsqueeze(1) == state_hist).all(dim=2)
            return eq.any(dim=1)

        cur_step_states = cur_states
        for step in range(1, walk_length + 1):
            t0_step = self._op_timer_start(cur_step_states)
            t0_sample_gen = self._op_timer_start(cur_step_states)
            gen_ids = _sample_gen_ids(cur_step_states.size(0))
            self._op_timer_stop("walk/sample_gen_ids", t0_sample_gen, cur_step_states)

            if gen_hist is not None:
                t0_inv_resample = self._op_timer_start(cur_step_states)
                for _ in range(attempts):
                    bad = _inverse_block_mask(gen_ids)
                    if not bool(bad.any()):
                        break
                    gen_ids = gen_ids.clone()
                    gen_ids[bad] = _sample_gen_ids(int(bad.sum().item()))
                self._op_timer_stop(
                    "walk/inverse_resample",
                    t0_inv_resample,
                    cur_step_states,
                )

            next_states = self._apply_generator_moves(cur_step_states, gen_ids)

            if state_hist is not None:
                t0_state_resample = self._op_timer_start(cur_step_states)
                for _ in range(attempts):
                    bad = _state_block_mask(next_states)
                    if not bool(bad.any()):
                        break
                    gen_ids = gen_ids.clone()
                    gen_ids[bad] = _sample_gen_ids(int(bad.sum().item()))
                    next_states = self._apply_generator_moves(cur_step_states, gen_ids)
                self._op_timer_stop(
                    "walk/state_resample",
                    t0_state_resample,
                    cur_step_states,
                )

            t0_history = self._op_timer_start(cur_step_states)
            store_idx = step if include_self else (step - 1)
            walk_states[:, store_idx, :] = next_states
            walk_gen_ids[:, store_idx] = gen_ids

            if gen_hist is not None:
                gen_hist = torch.roll(gen_hist, shifts=1, dims=1)
                gen_hist[:, 0] = gen_ids
            if state_hist is not None:
                state_hist = torch.roll(state_hist, shifts=1, dims=1)
                state_hist[:, 0, :] = cur_step_states
            self._op_timer_stop("walk/history_update", t0_history, cur_step_states)

            cur_step_states = next_states
            self._op_timer_stop("walk/step_total", t0_step, cur_step_states)

        tokens_z = walk_states.view(
            batch_size, num_walks * steps_to_store, self.state_size
        )
        tokens_gen = walk_gen_ids.view(batch_size, num_walks * steps_to_store)

        hop_start = 0 if include_self else 1
        hop_base = torch.arange(
            hop_start, walk_length + 1, device=device, dtype=torch.long
        )
        hop_ids = hop_base.repeat(num_walks).view(1, -1).expand(batch_size, -1)

        walk_ids = (
            torch
            .arange(num_walks, device=device, dtype=torch.long)
            .repeat_interleave(steps_to_store)
            .view(1, -1)
            .expand(batch_size, -1)
        )
        self._op_timer_stop("walk/sample_tokens_total", t0_total, tokens_z)
        return tokens_z, hop_ids, walk_ids, tokens_gen

    def _encode_walk_tokens(self, z: torch.Tensor) -> torch.Tensor | None:
        """
        Sample, encode, and tag neighborhood walk tokens.

        Args:
            z: Input decoded states of shape ``(batch, state_size)``.

        Returns:
            Token embeddings of shape ``(batch, n_tokens, d_model)`` or ``None`` if
            walk sampling is disabled or yields no tokens.

        """
        t0_total = self._op_timer_start(z)
        if self.num_walks <= 0 or self.walk_length <= 0:
            self._op_timer_stop("walk/encode_tokens_total", t0_total, z)
            return None

        t0_sample = self._op_timer_start(z)
        tokens_z, hop_ids, walk_ids, gen_ids = self._sample_walk_tokens(z)
        self._op_timer_stop("walk/sample_tokens_call", t0_sample, tokens_z)
        if tokens_z.size(1) == 0:
            self._op_timer_stop("walk/encode_tokens_total", t0_total, z)
            return None

        flat_tokens = tokens_z.reshape(-1, self.state_size)
        tok = self._encode_states_2d(
            flat_tokens,
            op_name="walk/token_state_encode",
        ).view(z.size(0), tokens_z.size(1), self.d_model)

        if self.hop_emb is not None:
            t0_hop = self._op_timer_start(tok)
            tok = tok + self.hop_emb(hop_ids).to(tok.dtype)
            self._op_timer_stop("walk/add_hop_embedding", t0_hop, tok)
        if self.walk_emb is not None:
            t0_walk = self._op_timer_start(tok)
            tok = tok + self.walk_emb(walk_ids).to(tok.dtype)
            self._op_timer_stop("walk/add_walk_embedding", t0_walk, tok)
        if self.gen_emb is not None:
            t0_gen = self._op_timer_start(tok)
            tok = tok + self.gen_emb(gen_ids.clamp(min=0, max=self.n_generators)).to(
                tok.dtype
            )
            self._op_timer_stop("walk/add_generator_embedding", t0_gen, tok)

        self._op_timer_stop("walk/encode_tokens_total", t0_total, tok)
        return tok

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Run model forward pass.

        Args:
            z: Input states of shape ``(batch, state_size)`` or ``(state_size,)``.

        Returns:
            Predicted scalar values as a 1D tensor of shape ``(batch,)``.

        """
        t0_total = self._op_timer_start(z)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.long()

        center = self._encode_states_2d(z, op_name="forward/center_encode")
        t0_token_encode = self._op_timer_start(z)
        tokens = self._encode_walk_tokens(z)
        self._op_timer_stop("forward/walk_token_encode_call", t0_token_encode, center)

        x = center
        for idx, layer in enumerate(self.layers):
            t0_layer = self._op_timer_start(x)
            x = layer(x, tokens)
            self._op_timer_stop(f"forward/layer_{int(idx)}", t0_layer, x)

        t0_final_norm = self._op_timer_start(x)
        x = self.final_norm(x)
        self._op_timer_stop("forward/final_norm", t0_final_norm, x)
        t0_out = self._op_timer_start(x)
        x = self.output_layer(x)
        self._op_timer_stop("forward/output_layer", t0_out, x)
        out = x.flatten()
        self._op_timer_stop("forward/total", t0_total, out)
        return out
