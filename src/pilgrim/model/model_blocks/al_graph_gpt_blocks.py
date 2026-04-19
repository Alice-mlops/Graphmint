"""Transformer-like building blocks used by AlGraphGPT."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn

from ...schemas.al_graph_gpt_config import AlGraphGPTConfig


class _RMSNorm(nn.Module):
    """
    Fallback RMSNorm implementation.

    This module is used when the runtime PyTorch version does not provide
    ``torch.nn.RMSNorm``.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        """
        Initialize RMSNorm fallback.

        Args:
            hidden_dim: Size of the last feature dimension.
            eps: Numerical stability constant added to the RMS denominator.

        """
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the last dimension of ``x``.

        Args:
            x: Input tensor of shape ``(..., hidden_dim)``.

        Returns:
            Tensor of the same shape as ``x``.

        """
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.scale


def make_norm(norm_type: str, hidden_dim: int, eps: float) -> nn.Module:
    """
    Create a normalization module by name.

    Args:
        norm_type: Name of normalization type (``"layernorm"`` or ``"rmsnorm"``).
        hidden_dim: Feature size for normalization.
        eps: Numerical stability epsilon.

    Returns:
        Instantiated normalization module.

    """
    norm_type = str(norm_type).strip().lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim, eps=eps)
    if norm_type == "rmsnorm":
        rmsnorm_cls = getattr(nn, "RMSNorm", None)
        if rmsnorm_cls is not None:
            return rmsnorm_cls(hidden_dim, eps=eps)
        return _RMSNorm(hidden_dim, eps=eps)
    raise ValueError(f"Unknown algraphgpt_norm_type: {norm_type}")


def make_activation(name: str) -> nn.Module:
    """
    Create an activation module by name.

    Args:
        name: Activation name (``"gelu"`` or ``"silu"``).

    Returns:
        Activation module instance.

    """
    act = str(name).strip().lower()
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown algraphgpt_activation: {name}")


class AlGraphGPTReadoutHead(nn.Module):
    """Linear readout head for scalar-value and vector-Q prediction."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        """
        Initialize the readout head.

        Args:
            hidden_dim: Size of the center embedding.
            output_dim: Number of output channels.

        Raises:
            ValueError: If ``output_dim`` is not positive.

        """
        super().__init__()
        if int(output_dim) <= 0:
            raise ValueError("output_dim must be positive.")
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, center: torch.Tensor) -> torch.Tensor:
        """
        Project center embeddings into the configured output space.

        Args:
            center: Center embeddings with shape ``(batch, hidden_dim)``.

        Returns:
            Tensor with shape ``(batch,)`` when ``output_dim == 1`` and
            ``(batch, output_dim)`` otherwise.

        """
        out = self.proj(center)
        if self.output_dim == 1:
            return out.flatten()
        return out


def run_readout_heads(
    center: torch.Tensor,
    *,
    primary_head: AlGraphGPTReadoutHead,
    auxiliary_head: AlGraphGPTReadoutHead | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Apply one or two readout heads to the shared center embedding.

    Args:
        center: Center embeddings with shape ``(batch, hidden_dim)``.
        primary_head: Mandatory primary prediction head.
        auxiliary_head: Optional auxiliary prediction head.

    Returns:
        Tuple ``(primary, auxiliary)`` where ``auxiliary`` is ``None`` when
        no auxiliary head is provided.

    """
    primary = primary_head(center)
    auxiliary = None if auxiliary_head is None else auxiliary_head(center)
    return primary, auxiliary


class AlGraphGPTLayer(nn.Module):
    """One center-query block: cross-attention branch + FFN branch."""

    def __init__(self, config: AlGraphGPTConfig) -> None:
        """
        Initialize one AlGraphGPT layer.

        Args:
            config: Validated AlGraphGPT configuration.

        """
        super().__init__()
        hidden_dim = int(config.algraphgpt_d_model)
        inner_dim = max(hidden_dim, round(hidden_dim * config.algraphgpt_ffn_mult))

        self.norm_position = str(config.algraphgpt_norm_position).strip().lower()
        self.attn_norm = make_norm(
            config.algraphgpt_norm_type, hidden_dim, config.algraphgpt_norm_eps
        )
        self.ffn_norm = make_norm(
            config.algraphgpt_norm_type, hidden_dim, config.algraphgpt_norm_eps
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=config.algraphgpt_num_heads,
            dropout=float(config.algraphgpt_attn_dropout),
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(float(config.algraphgpt_resid_dropout))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            make_activation(config.algraphgpt_activation),
            nn.Dropout(float(config.algraphgpt_ffn_dropout)),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(float(config.algraphgpt_ffn_dropout)),
        )

        if config.algraphgpt_rezero_init is None:
            self.attn_residual_scale = None
            self.ffn_residual_scale = None
        else:
            init = float(config.algraphgpt_rezero_init)
            self.attn_residual_scale = nn.Parameter(
                torch.tensor(init, dtype=torch.float32)
            )
            self.ffn_residual_scale = nn.Parameter(
                torch.tensor(init, dtype=torch.float32)
            )

        self._operation_profiler: Callable[[str, float], None] | None = None
        self._operation_profile_prefix: str = "layer"
        self._operation_profile_enabled: bool = False

    def set_operation_profiler(
        self,
        profiler: Callable[[str, float], None] | None,
        *,
        prefix: str,
    ) -> None:
        """
        Attach a timing recorder for semantic operation profiling.

        Args:
            profiler: Callback receiving ``(operation_name, elapsed_seconds)``.
            prefix: Per-layer operation prefix (for example ``"layer/0"``).

        Returns:
            None.

        """
        self._operation_profiler = profiler
        self._operation_profile_prefix = str(prefix)

    def enable_operation_profiling(self, enabled: bool) -> None:
        """
        Enable or disable internal operation timing for this layer.

        Args:
            enabled: ``True`` to enable profiling.

        Returns:
            None.

        """
        self._operation_profile_enabled = bool(enabled)

    def _op_timer_start(self, ref: torch.Tensor) -> float | None:
        """
        Start a synchronized timer when operation profiling is enabled.

        Args:
            ref: Reference tensor used to infer device.

        Returns:
            ``perf_counter`` timestamp or ``None`` when disabled.

        """
        if (
            not self._operation_profile_enabled
            or self._operation_profiler is None
            or not isinstance(ref, torch.Tensor)
        ):
            return None
        if torch.cuda.is_available() and ref.is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _op_timer_stop(
        self,
        name: str,
        start_time: float | None,
        ref: torch.Tensor,
    ) -> None:
        """
        Stop timer and report elapsed time to the attached profiler callback.

        Args:
            name: Operation suffix name.
            start_time: Start timestamp from :meth:`_op_timer_start`.
            ref: Reference tensor used to infer device.

        Returns:
            None.

        """
        if (
            start_time is None
            or self._operation_profiler is None
            or not isinstance(ref, torch.Tensor)
        ):
            return
        if torch.cuda.is_available() and ref.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        full_name = f"{self._operation_profile_prefix}/{name!s}"
        self._operation_profiler(full_name, float(elapsed))

    @staticmethod
    def _scale_residual(
        residual: torch.Tensor, scale: nn.Parameter | None
    ) -> torch.Tensor:
        """
        Optionally scale a residual branch output.

        Args:
            residual: Residual-branch output tensor.
            scale: Optional learned scalar parameter.

        Returns:
            Scaled residual tensor if ``scale`` is provided, otherwise unchanged
            ``residual``.

        """
        if scale is None:
            return residual
        return residual * scale.to(residual.dtype)

    def _cross_attention_branch(
        self, center: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-attention residual for one center token per batch item.

        Args:
            center: Center embedding tensor of shape ``(batch, hidden_dim)``.
            tokens: Neighborhood token tensor of shape ``(batch, n_tokens, hidden_dim)``.

        Returns:
            Cross-attention branch output of shape ``(batch, hidden_dim)``.

        """
        t0 = self._op_timer_start(center)
        ctx, _ = self.cross_attn(
            center.unsqueeze(1), tokens, tokens, need_weights=False
        )
        self._op_timer_stop("attention", t0, center)
        ctx = self.resid_dropout(ctx.squeeze(1))
        return self._scale_residual(ctx, self.attn_residual_scale)

    def _ffn_branch(self, center: torch.Tensor) -> torch.Tensor:
        """
        Compute FFN residual branch for center embeddings.

        Args:
            center: Center embedding tensor of shape ``(batch, hidden_dim)``.

        Returns:
            FFN branch output of shape ``(batch, hidden_dim)``.

        """
        t0 = self._op_timer_start(center)
        out = self.resid_dropout(self.ffn(center))
        self._op_timer_stop("ffn", t0, center)
        return self._scale_residual(out, self.ffn_residual_scale)

    def forward(
        self, center: torch.Tensor, tokens: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Apply one AlGraphGPT layer.

        Args:
            center: Center embeddings of shape ``(batch, hidden_dim)``.
            tokens: Optional neighborhood token table of shape
                ``(batch, n_tokens, hidden_dim)``. If ``None`` or empty, the
                attention branch is skipped.

        Returns:
            Updated center embeddings of shape ``(batch, hidden_dim)``.

        """
        t0_total = self._op_timer_start(center)
        token_table = tokens
        has_tokens = token_table is not None and token_table.size(1) > 0

        if self.norm_position == "pre":
            if has_tokens:
                assert token_table is not None
                center = center + self._cross_attention_branch(
                    self.attn_norm(center), token_table
                )
            center = center + self._ffn_branch(self.ffn_norm(center))
            self._op_timer_stop("forward_total", t0_total, center)
            return center

        # post-norm
        if has_tokens:
            assert token_table is not None
            center = self.attn_norm(
                center + self._cross_attention_branch(center, token_table)
            )
        center = self.ffn_norm(center + self._ffn_branch(center))
        self._op_timer_stop("forward_total", t0_total, center)
        return center
