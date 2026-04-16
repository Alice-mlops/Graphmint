"""
Input (node) encoders for the Pilgrim family of models.

Encoders in this module convert a discrete state tensor ``z`` (typically integer
tokens) into a dense feature representation suitable for MLP-based models.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

_EXPECTED_STATE_NDIM = 2


class OneHotLinearEncoder(nn.Module):
    """
    One-hot encode ``z`` then linearly project to ``output_dim``.

    This encoder matches the original Pilgrim behavior:

        ``z -> one_hot(z + z_add) -> flatten -> Linear``.

    Args:
        state_size: Number of discrete positions per example (second dimension of
            ``z``).
        num_classes: Number of discrete values per position.
        output_dim: Output feature dimension after projection.
        z_add: Integer offset added to ``z`` before one-hot encoding.
        dtype: Floating dtype used for the one-hot tensor before projection.

    """

    def __init__(
        self,
        *,
        state_size: int,
        num_classes: int,
        output_dim: int,
        z_add: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.z_add = z_add
        self.dtype = dtype

        self.proj = nn.Linear(state_size * num_classes, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of discrete states.

        Args:
            z: Tensor of shape ``(batch, state_size)``. After applying ``z_add``,
                values must lie in ``[0, num_classes - 1]``.

        Returns:
            Tensor of shape ``(batch, output_dim)``.

        """
        x = (
            F
            .one_hot(z.long() + self.z_add, num_classes=self.num_classes)
            .view(z.size(0), -1)
            .to(self.dtype)
        )
        return self.proj(x)


class EmbeddingFlattenEncoder(nn.Module):
    """
    Embedding lookup followed by flattening.

    Computes ``z -> Embedding -> flatten``, returning a tensor of shape
    ``(batch, state_size * embedding_dim)``.

    Args:
        state_size: Number of discrete positions per example (second dimension of
            ``z``).
        num_classes: Vocabulary size (number of embeddings).
        embedding_dim: Size of each embedding vector.
        z_add: Integer offset added to ``z`` before lookup. Values are clamped to
            ``[0, num_classes - 1]``.
        dtype: Floating dtype used for the returned tensor.

    """

    def __init__(
        self,
        *,
        state_size: int,
        num_classes: int,
        embedding_dim: int,
        z_add: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.output_dim = state_size * embedding_dim
        self.z_add = z_add
        self.dtype = dtype

        self.emb = nn.Embedding(num_classes, embedding_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of discrete states.

        Args:
            z: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, state_size * embedding_dim)``.

        """
        x = self.emb((z.long() + self.z_add).clamp(min=0, max=self.num_classes - 1))
        # (batch, state_size, embedding_dim) -> (batch, state_size * embedding_dim)
        return x.view(z.size(0), -1).to(self.dtype)


class LehmerCodeEncoder(nn.Module):
    """
    Encode permutation states through normalized Lehmer-code digits.

    This encoder computes Lehmer digits from the relative ordering of each input
    permutation row, normalizes each digit by its maximum possible value at that
    position, then linearly projects the resulting ``O(n)`` feature vector:

        ``z -> normalized_lehmer_digits(z) -> Linear``.

    Args:
        state_size: Number of discrete positions per example (second dimension of
            ``z``).
        num_classes: Number of discrete values per position. Stored for interface
            compatibility with other encoders.
        output_dim: Final feature dimension after projection.
        z_add: Stored for interface compatibility. Lehmer coding depends only on
            relative ordering, so additive shifts do not change the encoding.
        dtype: Floating dtype used for the returned tensor.

    """

    def __init__(
        self,
        *,
        state_size: int,
        num_classes: int,
        output_dim: int,
        z_add: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.output_dim = int(output_dim)
        self.z_add = z_add
        self.dtype = dtype

        self.proj = nn.Linear(state_size, self.output_dim)
        self.register_buffer(
            "_lehmer_upper_mask",
            torch.triu(
                torch.ones(state_size, state_size, dtype=torch.bool), diagonal=1
            ),
            persistent=False,
        )
        self.register_buffer(
            "_lehmer_normalizer",
            torch.arange(state_size - 1, -1, -1, dtype=torch.float32).clamp(min=1.0),
            persistent=False,
        )

    def _compute_lehmer_digits(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert permutation states to Lehmer digits.

        Args:
            z: Tensor of shape ``(batch, state_size)`` whose rows represent
                permutations or any distinct-valued ordering.

        Returns:
            Tensor of shape ``(batch, state_size)`` containing Lehmer digits.

        Raises:
            ValueError: If ``z`` does not have shape ``(batch, state_size)``.

        """
        if z.ndim != _EXPECTED_STATE_NDIM or int(z.shape[1]) != self.state_size:
            raise ValueError(
                "LehmerCodeEncoder expects z with shape "
                f"(batch, state_size={self.state_size}), got {tuple(z.shape)}."
            )
        z_long = z.long()
        return (
            ((z_long.unsqueeze(2) > z_long.unsqueeze(1)) & self._lehmer_upper_mask)
            .sum(dim=2)
            .long()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of permutation states.

        Args:
            z: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, output_dim)``.

        """
        digits = self._compute_lehmer_digits(z)
        x = digits.to(self.dtype) / self._lehmer_normalizer.to(self.dtype)
        x = self.proj(x)
        return x.to(self.dtype)


class LehmerBreakpointsEncoder(LehmerCodeEncoder):
    """
    Encode permutations with normalized Lehmer digits and breakpoint features.

    This encoder concatenates:

    - normalized Lehmer digits of length ``state_size``,
    - breakpoint indicator bits of length ``state_size + 1``,
    - normalized breakpoint count ``B / (state_size + 1)``,

    then applies a single linear projection:

        ``z -> [normalized_lehmer, breakpoints, breakpoint_fraction] -> Linear``.

    Breakpoints assume permutation labels are meaningful and contiguous, which is
    appropriate for pancake states represented as ``0, 1, ..., n - 1``.

    Args:
        state_size: Number of discrete positions per example.
        num_classes: Number of discrete values per position. Stored for interface
            compatibility with other encoders.
        output_dim: Final feature dimension after projection.
        z_add: Offset added to ``z`` before computing breakpoint features. Use this
            only when it converts the incoming permutation labels to ``0..n-1``.
        dtype: Floating dtype used for the returned tensor.

    """

    def __init__(
        self,
        *,
        state_size: int,
        num_classes: int,
        output_dim: int,
        z_add: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            state_size=state_size,
            num_classes=num_classes,
            output_dim=output_dim,
            z_add=z_add,
            dtype=dtype,
        )
        self.base_output_dim = (2 * state_size) + 2
        self.proj = nn.Linear(self.base_output_dim, self.output_dim)

    def _compute_breakpoint_bits(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute pancake breakpoint indicators for a batch of permutations.

        Args:
            z: Tensor of shape ``(batch, state_size)`` with permutation labels in
                ``0..state_size-1`` after applying ``z_add``.

        Returns:
            Tensor of shape ``(batch, state_size + 1)`` with ``0/1`` breakpoint
            indicators.

        """
        z_long = z.long() + self.z_add
        batch_size = int(z_long.shape[0])
        extended = torch.empty(
            (batch_size, self.state_size + 2),
            device=z.device,
            dtype=torch.long,
        )
        extended[:, 0] = 0
        extended[:, 1:-1] = z_long + 1
        extended[:, -1] = self.state_size + 1
        return extended[:, 1:].sub(extended[:, :-1]).abs().ne(1).to(self.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of permutation states.

        Args:
            z: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, output_dim)``.

        """
        digits = self._compute_lehmer_digits(z)
        lehmer = digits.to(self.dtype) / self._lehmer_normalizer.to(self.dtype)
        breakpoints = self._compute_breakpoint_bits(z)
        breakpoint_fraction = breakpoints.mean(dim=1, keepdim=True)
        x = torch.cat([lehmer, breakpoints, breakpoint_fraction], dim=1)
        x = self.proj(x)
        return x.to(self.dtype)


def build_node_encoder(config: dict[str, Any]) -> tuple[nn.Module, int]:
    """
    Build an input encoder from a config dictionary.

    The returned encoder maps a discrete state tensor ``z`` to a dense feature
    tensor. Callers typically project the encoder output to the model's hidden
    size if needed.

    Args:
        config: Configuration dictionary. Required keys:
            - ``state_size`` (int): Number of discrete positions.
            - ``num_classes`` (int): Number of discrete values per position.
            - ``hd1`` (int): Default model hidden size (used as default output dim
              for the one-hot encoder).
          Optional keys:
            - ``z_add`` (int): Offset added to ``z`` before encoding (default: 0).
            - ``dtype`` (torch.dtype): Output dtype for encoders (default:
              ``torch.float32``).
            - ``input_encoder`` (nn.Module): Custom encoder. Its output dimension
              must be discoverable via ``encoder.output_dim`` or via
              ``input_encoder_out_dim``.
            - ``input_encoder_out_dim`` (int): Output dimension for a custom
              encoder, or override for ``onehot_linear``.
            - ``input_encoder_type`` (str): One of ``"onehot_linear"`` (default),
              ``"embedding_flatten"``, ``"lehmer"``, or
              ``"lehmer-breakpoints"``.
            - ``embedding_dim`` (int): Required for embedding encoders.

    Returns:
        A tuple ``(encoder, output_dim)`` where ``encoder`` is an ``nn.Module`` and
        ``output_dim`` is the size of its last dimension.

    Raises:
        ValueError: If ``input_encoder_type`` is unknown, or if a custom
            ``input_encoder`` is provided without a known output dimension.

    """
    state_size = int(config["state_size"])
    num_classes = int(config["num_classes"])
    hd1 = int(config["hd1"])
    z_add = int(config.get("z_add", 0))
    dtype = config.get("dtype", torch.float32)

    if "input_encoder" in config and config["input_encoder"] is not None:
        enc = config["input_encoder"]
        out_dim = getattr(enc, "output_dim", None)
        if out_dim is None:
            out_dim = config.get("input_encoder_out_dim")
        if out_dim is None:
            raise ValueError(
                "config['input_encoder'] was provided but output dim is unknown. "
                "Set encoder.output_dim or config['input_encoder_out_dim']."
            )
        return enc, int(out_dim)

    enc_type = str(config.get("input_encoder_type", "onehot_linear"))

    if enc_type == "onehot_linear":
        out_dim = int(config.get("input_encoder_out_dim", hd1))
        return (
            OneHotLinearEncoder(
                state_size=state_size,
                num_classes=num_classes,
                output_dim=out_dim,
                z_add=z_add,
                dtype=dtype,
            ),
            out_dim,
        )

    if enc_type == "embedding_flatten":
        emb_dim = int(config["embedding_dim"])
        enc = EmbeddingFlattenEncoder(
            state_size=state_size,
            num_classes=num_classes,
            embedding_dim=emb_dim,
            z_add=z_add,
            dtype=dtype,
        )
        return enc, int(enc.output_dim)

    if enc_type == "lehmer":
        out_dim = int(config.get("input_encoder_out_dim", hd1))
        enc = LehmerCodeEncoder(
            state_size=state_size,
            num_classes=num_classes,
            output_dim=out_dim,
            z_add=z_add,
            dtype=dtype,
        )
        return enc, out_dim

    if enc_type == "lehmer-breakpoints":
        out_dim = int(config.get("input_encoder_out_dim", hd1))
        enc = LehmerBreakpointsEncoder(
            state_size=state_size,
            num_classes=num_classes,
            output_dim=out_dim,
            z_add=z_add,
            dtype=dtype,
        )
        return enc, out_dim

    raise ValueError(f"Unknown input_encoder_type: {enc_type}")
