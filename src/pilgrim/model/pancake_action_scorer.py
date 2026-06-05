# Defines a transferable shared-action scorer for pancake permutations.
"""Shared per-action scoring model for pancake sorting policies."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

_EXPECTED_STATE_NDIM = 2
_MIN_PANCAKE_N = 2
_STATE_FEATURE_DIM = 16
_ACTION_FEATURE_DIM = 9


def _as_state_batch(states: torch.Tensor) -> torch.Tensor:
    """
    Normalize state tensors to a two-dimensional long tensor.

    Args:
        states: State tensor shaped ``(n,)`` or ``(batch, n)``.

    Returns:
        Two-dimensional long tensor.

    Raises:
        ValueError: If the input cannot be interpreted as a state batch.

    """
    batch = torch.as_tensor(states).long()
    if batch.ndim == 1:
        batch = batch.unsqueeze(0)
    if batch.ndim != _EXPECTED_STATE_NDIM:
        raise ValueError(
            f"states must have shape (n,) or (batch, n), got {tuple(batch.shape)}."
        )
    return batch.contiguous()


class PancakeActionScorer(nn.Module):
    """
    Score every legal pancake prefix flip with parameters shared across ``n``.

    The model first builds continuous Lehmer/breakpoint-style token features for
    each position of a permutation, encodes them with a shared Transformer, and
    then scores each prefix reversal ``Rk`` from the state embedding plus
    action-local features. The returned policy width is therefore data-dependent:
    an input with size ``n`` returns logits with width ``n - 1`` for actions
    ``R2..Rn``.

    Args:
        d_model: Internal feature width.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        ffn_mult: Feed-forward expansion multiplier in Transformer layers.
        dropout: Dropout probability.
        max_n: Largest supported pancake size.
        activation: Transformer/feed-forward activation.
        model_dtype: Floating dtype for model features.
        validate_inputs: Whether to validate that rows are permutations.

    Raises:
        ValueError: If the model configuration is inconsistent.

    """

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_mult: float = 2.0,
        dropout: float = 0.1,
        max_n: int = 128,
        activation: Literal["gelu", "relu"] = "gelu",
        model_dtype: torch.dtype = torch.float32,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if int(d_model) <= 0:
            raise ValueError("d_model must be positive.")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive.")
        if int(num_heads) <= 0:
            raise ValueError("num_heads must be positive.")
        if int(d_model) % int(num_heads) != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if int(max_n) < _MIN_PANCAKE_N:
            raise ValueError("max_n must be at least 2.")

        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_mult = float(ffn_mult)
        self.dropout = float(dropout)
        self.max_n = int(max_n)
        self.model_dtype = model_dtype
        self.validate_inputs = bool(validate_inputs)

        self.state_proj = nn.Sequential(
            nn.Linear(_STATE_FEATURE_DIM, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=max(1, round(self.d_model * self.ffn_mult)),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.global_proj = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.d_model + _ACTION_FEATURE_DIM, self.d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(self.d_model, 1),
        )

    @property
    def output_dim(self) -> int:
        """
        Return the largest possible action width.

        Returns:
            Maximum number of prefix-reversal actions supported by this model.

        """
        return self.max_n - 1

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Score all legal prefix reversals for each input state.

        Args:
            states: Tensor shaped ``(n,)`` or ``(batch, n)``.

        Returns:
            Logits shaped ``(batch, n - 1)`` for actions ``R2..Rn``.

        """
        features = self.forward_features(states)
        action_features = self.build_action_features(states).to(
            device=features.device,
            dtype=features.dtype,
        )
        expanded = features.unsqueeze(1).expand(-1, action_features.shape[1], -1)
        logits = self.action_head(torch.cat([expanded, action_features], dim=-1))
        return logits.squeeze(-1)

    def forward_readouts(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return policy logits and scalar value estimates.

        Args:
            states: Tensor shaped ``(n,)`` or ``(batch, n)``.

        Returns:
            Tuple ``(policy_logits, values)``. Policy logits have shape
            ``(batch, n - 1)`` and values have shape ``(batch,)``.

        """
        features = self.forward_features(states)
        action_features = self.build_action_features(states).to(
            device=features.device,
            dtype=features.dtype,
        )
        expanded = features.unsqueeze(1).expand(-1, action_features.shape[1], -1)
        logits = self.action_head(torch.cat([expanded, action_features], dim=-1))
        values = self.value_head(features)
        return logits.squeeze(-1), values.squeeze(-1)

    def forward_features(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encode a pancake state batch into one vector per state.

        Args:
            states: Tensor shaped ``(n,)`` or ``(batch, n)``.

        Returns:
            State embeddings shaped ``(batch, d_model)``.

        """
        batch = self._validate_states(states)
        token_features, global_features = self.build_state_features(batch)
        token_features = token_features.to(
            device=batch.device,
            dtype=self.model_dtype,
        )
        global_features = global_features.to(
            device=batch.device,
            dtype=self.model_dtype,
        )
        tokens = self.state_proj(token_features)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return pooled + self.global_proj(global_features)

    def build_state_features(  # noqa: PLR0914
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build token and global permutation features.

        Args:
            states: Tensor shaped ``(n,)`` or ``(batch, n)``.

        Returns:
            Tuple ``(token_features, global_features)``. Token features have
            shape ``(batch, n, 16)`` and global features have shape ``(batch, 5)``.

        """
        batch = self._validate_states(states)
        batch_size, n = int(batch.shape[0]), int(batch.shape[1])
        denom = float(max(n - 1, 1))
        pos = torch.arange(n, device=batch.device, dtype=torch.float32)
        pos = pos.view(1, n).expand(batch_size, -1)
        values = batch.to(torch.float32)

        inverse_positions = torch.empty_like(batch)
        inverse_positions.scatter_(
            dim=1,
            index=batch,
            src=torch.arange(n, device=batch.device).view(1, n).expand(batch_size, -1),
        )
        predecessor = (batch - 1).clamp(min=0)
        successor = (batch + 1).clamp(max=n - 1)
        predecessor_pos = inverse_positions.gather(1, predecessor).to(torch.float32)
        successor_pos = inverse_positions.gather(1, successor).to(torch.float32)

        breakpoints, deltas = self._breakpoint_bits_and_deltas(batch)
        left_breakpoints = breakpoints[:, :-1]
        right_breakpoints = breakpoints[:, 1:]
        left_delta = deltas[:, :-1].to(torch.float32) / float(n + 1)
        right_delta = deltas[:, 1:].to(torch.float32) / float(n + 1)

        lehmer = self._lehmer_digits(batch).to(torch.float32)
        lehmer_normalizer = (
            torch
            .arange(n - 1, -1, -1, device=batch.device, dtype=torch.float32)
            .clamp(min=1.0)
            .view(1, n)
        )

        signed_delta = (values - pos) / denom
        abs_delta = signed_delta.abs()
        solved = values.eq(pos).to(torch.float32)
        breakpoint_fraction = breakpoints.mean(dim=1, keepdim=True).expand(-1, n)
        n_fraction = torch.full_like(pos, float(n) / float(self.max_n))

        token_features = torch.stack(
            [
                pos / denom,
                values / denom,
                signed_delta,
                abs_delta,
                solved,
                lehmer / lehmer_normalizer,
                left_breakpoints,
                right_breakpoints,
                left_delta,
                right_delta,
                predecessor_pos / denom,
                successor_pos / denom,
                batch.eq(0).to(torch.float32),
                batch.eq(n - 1).to(torch.float32),
                breakpoint_fraction,
                n_fraction,
            ],
            dim=-1,
        )
        global_features = torch.cat(
            [
                torch.full(
                    (batch_size, 1), float(n) / float(self.max_n), device=batch.device
                ),
                breakpoints.mean(dim=1, keepdim=True),
                abs_delta.mean(dim=1, keepdim=True),
                abs_delta.max(dim=1, keepdim=True).values,
                solved.mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return token_features, global_features

    def build_action_features(self, states: torch.Tensor) -> torch.Tensor:
        """
        Build action-local features for every legal prefix reversal.

        Args:
            states: Tensor shaped ``(n,)`` or ``(batch, n)``.

        Returns:
            Tensor shaped ``(batch, n - 1, 9)`` aligned with actions ``R2..Rn``.

        """
        batch = self._validate_states(states)
        batch_size, n = int(batch.shape[0]), int(batch.shape[1])
        denom = float(max(n - 1, 1))
        current_breakpoints = self._breakpoint_bits_and_deltas(batch)[0].sum(
            dim=1,
            keepdim=True,
        )
        current_fraction = current_breakpoints / float(n + 1)
        first_value = batch[:, 0].to(torch.float32) / denom
        rows: list[torch.Tensor] = []
        for k in range(2, n + 1):
            next_states = batch.clone()
            next_states[:, :k] = torch.flip(batch[:, :k], dims=[1])
            next_breakpoints = self._breakpoint_bits_and_deltas(next_states)[0].sum(
                dim=1,
                keepdim=True,
            )
            kth_value = batch[:, k - 1].to(torch.float32) / denom
            if k < n:
                after_value = batch[:, k].to(torch.float32) / denom
            else:
                after_value = torch.ones(batch_size, device=batch.device)
            row = torch.stack(
                [
                    torch.full((batch_size,), float(k) / float(n), device=batch.device),
                    torch.full(
                        (batch_size,), float(k - 1) / denom, device=batch.device
                    ),
                    torch.full((batch_size,), float(k == n), device=batch.device),
                    first_value,
                    kth_value,
                    after_value,
                    current_fraction.squeeze(1),
                    ((next_breakpoints - current_breakpoints) / float(n + 1)).squeeze(
                        1
                    ),
                    (next_breakpoints / float(n + 1)).squeeze(1),
                ],
                dim=1,
            )
            rows.append(row)
        return torch.stack(rows, dim=1).to(self.model_dtype)

    @staticmethod
    def action_ids_to_prefix_lengths(states: torch.Tensor) -> torch.Tensor:
        """
        Return prefix lengths corresponding to model action ids.

        Args:
            states: Tensor shaped ``(n,)`` or ``(batch, n)``.

        Returns:
            One-dimensional tensor ``[2, 3, ..., n]``.

        """
        batch = _as_state_batch(states)
        n = int(batch.shape[1])
        return torch.arange(2, n + 1, device=batch.device, dtype=torch.long)

    def _validate_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Validate and normalize a pancake permutation batch.

        Args:
            states: Tensor shaped ``(n,)`` or ``(batch, n)``.

        Returns:
            Two-dimensional long tensor on the original device.

        Raises:
            ValueError: If the state size or permutation labels are invalid.

        """
        batch = _as_state_batch(states)
        n = int(batch.shape[1])
        if n < _MIN_PANCAKE_N:
            raise ValueError("pancake states must have n >= 2.")
        if n > self.max_n:
            raise ValueError(f"state size n={n} exceeds max_n={self.max_n}.")
        if not self.validate_inputs:
            return batch
        expected = torch.arange(n, device=batch.device).view(1, n).expand_as(batch)
        sorted_rows = torch.sort(batch, dim=1).values
        if not bool(torch.equal(sorted_rows, expected)):
            raise ValueError("each state row must be a permutation of 0..n-1.")
        return batch

    @staticmethod
    def _lehmer_digits(states: torch.Tensor) -> torch.Tensor:
        """
        Compute Lehmer digits for each permutation row.

        Args:
            states: Long tensor shaped ``(batch, n)``.

        Returns:
            Long tensor shaped ``(batch, n)``.

        """
        n = int(states.shape[1])
        upper_mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=states.device),
            diagonal=1,
        )
        return (
            ((states.unsqueeze(2) > states.unsqueeze(1)) & upper_mask).sum(dim=2).long()
        )

    @staticmethod
    def _breakpoint_bits_and_deltas(
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pancake breakpoint indicators and absolute boundary deltas.

        Args:
            states: Long tensor shaped ``(batch, n)``.

        Returns:
            Tuple of tensors shaped ``(batch, n + 1)``.

        """
        batch_size, n = int(states.shape[0]), int(states.shape[1])
        shifted = states.long() + 1
        extended = torch.empty(
            (batch_size, n + 2),
            device=states.device,
            dtype=torch.long,
        )
        extended[:, 0] = 0
        extended[:, 1:-1] = shifted
        extended[:, -1] = n + 1
        deltas = extended[:, 1:].sub(extended[:, :-1]).abs()
        return deltas.ne(1).to(torch.float32), deltas
