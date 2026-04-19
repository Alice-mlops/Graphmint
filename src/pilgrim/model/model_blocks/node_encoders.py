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
_GENERATOR_MOVES_EXPECTED_NDIM = 2
_DEFAULT_MEGAMINX_EMBEDDING_DIM = 16
_DEFAULT_MEGAMINX_NUM_FACES = 12


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


class MegaminxPermutationEncoder(nn.Module):
    """
    Encode Megaminx sticker permutations with puzzle-structured features.

    The decoded Megaminx states used by ``cayleypy`` are permutations of sticker
    labels ``0..119``. This encoder treats those labels as solved sticker IDs and
    builds a whole-state vector from:

    - learned sticker-at-position embeddings,
    - learned inverse ``sticker -> current position`` embeddings,
    - explicit solved-position, face, and face-slot mismatch indicators,
    - target-face/source-face occupancy counts,
    - optional graph-breakpoint bits from the Megaminx generator permutations.

    Args:
        state_size: Number of sticker positions per example.
        num_classes: Number of possible sticker labels.
        output_dim: Final feature dimension after projection.
        z_add: Offset added to ``z`` before feature construction.
        dtype: Floating dtype used for the returned tensor.
        embedding_dim: Size of each learned sticker/position embedding.
        num_faces: Number of solved face groups. Standard Megaminx uses ``12``.
        use_inverse: Whether to include inverse sticker-position embeddings.
        use_graph_breakpoints: Whether to include generator-graph breakpoint
            features when ``generator_moves`` is available.
        generator_moves: Optional permutation generators with shape
            ``(n_generators, state_size)``.

    """

    def __init__(
        self,
        *,
        state_size: int,
        num_classes: int,
        output_dim: int,
        z_add: int = 0,
        dtype: torch.dtype = torch.float32,
        embedding_dim: int = _DEFAULT_MEGAMINX_EMBEDDING_DIM,
        num_faces: int = _DEFAULT_MEGAMINX_NUM_FACES,
        use_inverse: bool = True,
        use_graph_breakpoints: bool = True,
        generator_moves: Any | None = None,
    ) -> None:
        super().__init__()
        if state_size % num_faces != 0:
            raise ValueError(
                "MegaminxPermutationEncoder requires state_size to be divisible "
                f"by num_faces, got state_size={state_size}, num_faces={num_faces}."
            )
        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.output_dim = int(output_dim)
        self.z_add = int(z_add)
        self.dtype = dtype
        self.embedding_dim = int(embedding_dim)
        self.num_faces = int(num_faces)
        self.face_size = int(state_size // num_faces)
        self.use_inverse = bool(use_inverse)
        self.use_graph_breakpoints = bool(use_graph_breakpoints)

        self.piece_emb = nn.Embedding(self.num_classes, self.embedding_dim)
        self.position_emb = nn.Embedding(self.state_size, self.embedding_dim)
        self.source_face_emb = nn.Embedding(self.num_faces, self.embedding_dim)
        self.target_face_emb = nn.Embedding(self.num_faces, self.embedding_dim)
        self.source_slot_emb = nn.Embedding(self.face_size, self.embedding_dim)
        self.target_slot_emb = nn.Embedding(self.face_size, self.embedding_dim)
        self.status_proj = nn.Linear(3, self.embedding_dim)
        self.site_norm = nn.LayerNorm(self.embedding_dim)

        position_ids = torch.arange(self.state_size, dtype=torch.long)
        self.register_buffer("_position_ids", position_ids, persistent=False)
        self.register_buffer(
            "_position_face_ids",
            position_ids.div(self.face_size, rounding_mode="floor"),
            persistent=False,
        )
        self.register_buffer(
            "_position_slot_ids",
            position_ids.remainder(self.face_size),
            persistent=False,
        )
        self.register_buffer(
            "_target_face_one_hot",
            F.one_hot(
                position_ids.div(self.face_size, rounding_mode="floor"),
                num_classes=self.num_faces,
            ).to(torch.float32),
            persistent=False,
        )

        edge_src, edge_dst, solved_edge_lookup = self._build_graph_edges(
            generator_moves
        )
        self.register_buffer("_edge_src", edge_src, persistent=False)
        self.register_buffer("_edge_dst", edge_dst, persistent=False)
        self.register_buffer(
            "_solved_edge_lookup",
            solved_edge_lookup,
            persistent=False,
        )

        graph_dim = 0
        if self.use_graph_breakpoints and int(edge_src.numel()) > 0:
            graph_dim = int(edge_src.numel()) + 1
        inverse_dim = self.state_size * self.embedding_dim if self.use_inverse else 0
        self.base_output_dim = (
            self.state_size * self.embedding_dim
            + inverse_dim
            + (3 * self.state_size)
            + (self.num_faces * self.num_faces)
            + self.num_faces
            + 3
            + graph_dim
        )
        self.proj = nn.Linear(self.base_output_dim, self.output_dim)

    def _build_graph_edges(
        self,
        generator_moves: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build solved sticker-adjacency edges from generator permutations.

        Args:
            generator_moves: Optional generator permutations with shape
                ``(n_generators, state_size)``.

        Returns:
            Tuple ``(edge_src, edge_dst, solved_edge_lookup)``. ``edge_src`` and
            ``edge_dst`` contain undirected edge endpoints, while
            ``solved_edge_lookup`` is a symmetric boolean adjacency matrix.

        Raises:
            ValueError: If ``generator_moves`` has an incompatible shape.

        """
        lookup = torch.zeros((self.state_size, self.state_size), dtype=torch.bool)
        if generator_moves is None:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, lookup

        moves = torch.as_tensor(generator_moves, dtype=torch.long)
        if (
            moves.ndim != _GENERATOR_MOVES_EXPECTED_NDIM
            or int(moves.shape[1]) != self.state_size
        ):
            raise ValueError(
                "generator_moves must have shape "
                f"(n_generators, state_size={self.state_size}), got "
                f"{tuple(moves.shape)}."
            )

        edges: set[tuple[int, int]] = set()
        for move in moves.detach().cpu().tolist():
            for src, dst in enumerate(move):
                dst_i = int(dst)
                if int(src) == dst_i:
                    continue
                edge = (int(src), dst_i) if int(src) < dst_i else (dst_i, int(src))
                edges.add(edge)

        if not edges:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, lookup

        edge_tensor = torch.tensor(sorted(edges), dtype=torch.long)
        edge_src = edge_tensor[:, 0].contiguous()
        edge_dst = edge_tensor[:, 1].contiguous()
        lookup[edge_src, edge_dst] = True
        lookup[edge_dst, edge_src] = True
        return edge_src, edge_dst, lookup

    def _validate_and_shift(self, z: torch.Tensor) -> torch.Tensor:
        """
        Validate a state batch and apply the configured label offset.

        Args:
            z: Tensor of shape ``(batch, state_size)``.

        Returns:
            Shifted integer label tensor with shape ``(batch, state_size)``.

        Raises:
            ValueError: If ``z`` does not have shape ``(batch, state_size)``.

        """
        if z.ndim != _EXPECTED_STATE_NDIM or int(z.shape[1]) != self.state_size:
            raise ValueError(
                "MegaminxPermutationEncoder expects z with shape "
                f"(batch, state_size={self.state_size}), got {tuple(z.shape)}."
            )
        return z.long() + self.z_add

    def _label_parts(self, labels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Split sticker labels into embedding-safe IDs, solved faces, and slots.

        Args:
            labels: Shifted integer labels with shape ``(batch, state_size)``.

        Returns:
            Tuple ``(label_ids, state_labels, source_faces, source_slots)``.

        """
        label_ids = labels.clamp(min=0, max=self.num_classes - 1)
        state_labels = labels.clamp(min=0, max=self.state_size - 1)
        source_faces = state_labels.div(self.face_size, rounding_mode="floor")
        source_slots = state_labels.remainder(self.face_size)
        return label_ids, state_labels, source_faces, source_slots

    def _status_bits(
        self,
        state_labels: torch.Tensor,
        source_faces: torch.Tensor,
        source_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute solved-position, face, and local-slot mismatch bits.

        Args:
            state_labels: Sticker labels clamped to ``0..state_size-1``.
            source_faces: Solved source face id for each sticker label.
            source_slots: Solved source face slot for each sticker label.

        Returns:
            Tuple of float tensors ``(misplaced, face_mismatch, slot_mismatch)``.

        """
        position_ids = self._position_ids.to(state_labels.device)
        position_faces = self._position_face_ids.to(state_labels.device)
        position_slots = self._position_slot_ids.to(state_labels.device)
        misplaced = state_labels.ne(position_ids.view(1, -1))
        face_mismatch = source_faces.ne(position_faces.view(1, -1))
        slot_mismatch = source_slots.ne(position_slots.view(1, -1))
        feature_dtype = self.proj.weight.dtype
        return (
            misplaced.to(feature_dtype),
            face_mismatch.to(feature_dtype),
            slot_mismatch.to(feature_dtype),
        )

    def _encode_position_sites(
        self,
        label_ids: torch.Tensor,
        source_faces: torch.Tensor,
        source_slots: torch.Tensor,
        status: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode stickers in target-position order.

        Args:
            label_ids: Embedding-safe sticker IDs.
            source_faces: Solved source face id for each sticker.
            source_slots: Solved source face slot for each sticker.
            status: Per-position status bits with shape ``(batch, state_size, 3)``.

        Returns:
            Flattened tensor with shape ``(batch, state_size * embedding_dim)``.

        """
        position_ids = self._position_ids.to(label_ids.device)
        position_faces = self._position_face_ids.to(label_ids.device)
        position_slots = self._position_slot_ids.to(label_ids.device)
        batch_size = int(label_ids.shape[0])

        x = (
            self.piece_emb(label_ids)
            + self.position_emb(position_ids).view(1, self.state_size, -1)
            + self.source_face_emb(source_faces)
            + self.target_face_emb(position_faces).view(1, self.state_size, -1)
            + self.source_slot_emb(source_slots)
            + self.target_slot_emb(position_slots).view(1, self.state_size, -1)
            + self.status_proj(status.to(self.status_proj.weight.dtype))
        )
        x = self.site_norm(x)
        return x.reshape(batch_size, self.state_size * self.embedding_dim)

    def _encode_inverse_sites(self, state_labels: torch.Tensor) -> torch.Tensor:
        """
        Encode current positions in solved-sticker order.

        Args:
            state_labels: Sticker labels clamped to ``0..state_size-1``.

        Returns:
            Flattened tensor with shape ``(batch, state_size * embedding_dim)``.

        """
        batch_size = int(state_labels.shape[0])
        position_ids = self._position_ids.to(state_labels.device)
        solved_faces = self._position_face_ids.to(state_labels.device)
        solved_slots = self._position_slot_ids.to(state_labels.device)

        current_positions = torch.zeros_like(state_labels)
        current_positions.scatter_(
            1,
            state_labels,
            position_ids.view(1, -1).expand(batch_size, -1),
        )
        current_faces = current_positions.div(self.face_size, rounding_mode="floor")
        current_slots = current_positions.remainder(self.face_size)

        misplaced = current_positions.ne(position_ids.view(1, -1))
        face_mismatch = current_faces.ne(solved_faces.view(1, -1))
        slot_mismatch = current_slots.ne(solved_slots.view(1, -1))
        status = torch.stack(
            [
                misplaced,
                face_mismatch,
                slot_mismatch,
            ],
            dim=2,
        ).to(self.status_proj.weight.dtype)

        x = (
            self.piece_emb(position_ids).view(1, self.state_size, -1)
            + self.position_emb(current_positions)
            + self.source_face_emb(solved_faces).view(1, self.state_size, -1)
            + self.target_face_emb(current_faces)
            + self.source_slot_emb(solved_slots).view(1, self.state_size, -1)
            + self.target_slot_emb(current_slots)
            + self.status_proj(status)
        )
        x = self.site_norm(x)
        return x.reshape(batch_size, self.state_size * self.embedding_dim)

    def _compute_graph_breakpoint_bits(
        self, state_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute generator-graph breakpoint indicators.

        A graph breakpoint is an edge whose two current sticker labels are not an
        edge in the solved generator-derived sticker graph.

        Args:
            state_labels: Sticker labels clamped to ``0..state_size-1``.

        Returns:
            Tensor with shape ``(batch, n_edges)``. If no graph edges are enabled,
            the second dimension is zero.

        """
        if not self.use_graph_breakpoints or int(self._edge_src.numel()) == 0:
            return torch.empty(
                (state_labels.shape[0], 0),
                device=state_labels.device,
                dtype=self.proj.weight.dtype,
            )
        edge_src = self._edge_src.to(state_labels.device)
        edge_dst = self._edge_dst.to(state_labels.device)
        solved_edge_lookup = self._solved_edge_lookup.to(state_labels.device)
        left = state_labels.index_select(1, edge_src)
        right = state_labels.index_select(1, edge_dst)
        is_solved_edge = solved_edge_lookup[left, right]
        return is_solved_edge.logical_not().to(self.proj.weight.dtype)

    def _compute_structural_features(
        self,
        state_labels: torch.Tensor,
        source_faces: torch.Tensor,
        status_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Build explicit mismatch, face-occupancy, and breakpoint features.

        Args:
            state_labels: Sticker labels clamped to ``0..state_size-1``.
            source_faces: Solved source face id for each sticker.
            status_parts: Tuple ``(misplaced, face_mismatch, slot_mismatch)``.

        Returns:
            Structural feature tensor with shape ``(batch, feature_dim)``.

        """
        misplaced, face_mismatch, slot_mismatch = status_parts
        feature_dtype = self.proj.weight.dtype
        target_face_one_hot = self._target_face_one_hot.to(
            device=state_labels.device,
            dtype=feature_dtype,
        )
        source_face_one_hot = F.one_hot(
            source_faces,
            num_classes=self.num_faces,
        ).to(feature_dtype)
        face_counts = torch.einsum(
            "pf,bps->bfs",
            target_face_one_hot,
            source_face_one_hot,
        ) / float(self.face_size)
        per_face_mismatch = torch.einsum(
            "pf,bp->bf",
            target_face_one_hot,
            face_mismatch,
        ) / float(self.face_size)

        graph_breakpoints = self._compute_graph_breakpoint_bits(state_labels)
        features = [
            misplaced,
            face_mismatch,
            slot_mismatch,
            face_counts.reshape(state_labels.shape[0], -1),
            per_face_mismatch,
            misplaced.mean(dim=1, keepdim=True),
            face_mismatch.mean(dim=1, keepdim=True),
            slot_mismatch.mean(dim=1, keepdim=True),
        ]
        if int(graph_breakpoints.shape[1]) > 0:
            features.extend([
                graph_breakpoints,
                graph_breakpoints.mean(dim=1, keepdim=True),
            ])
        return torch.cat(features, dim=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of Megaminx sticker permutations.

        Args:
            z: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, output_dim)``.

        """
        labels = self._validate_and_shift(z)
        label_ids, state_labels, source_faces, source_slots = self._label_parts(labels)
        status_parts = self._status_bits(state_labels, source_faces, source_slots)
        status = torch.stack(status_parts, dim=2)

        parts = [
            self._encode_position_sites(
                label_ids,
                source_faces,
                source_slots,
                status,
            )
        ]
        if self.use_inverse:
            parts.append(self._encode_inverse_sites(state_labels))
        parts.append(
            self._compute_structural_features(
                state_labels,
                source_faces,
                status_parts,
            )
        )

        x = torch.cat(parts, dim=1).to(self.proj.weight.dtype)
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
              ``"lehmer-breakpoints"``, or ``"megaminx"``.
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

    if enc_type == "megaminx":
        out_dim = int(config.get("input_encoder_out_dim", hd1))
        emb_dim = int(
            config.get("megaminx_embedding_dim")
            or config.get("embedding_dim")
            or _DEFAULT_MEGAMINX_EMBEDDING_DIM
        )
        enc = MegaminxPermutationEncoder(
            state_size=state_size,
            num_classes=num_classes,
            output_dim=out_dim,
            z_add=z_add,
            dtype=dtype,
            embedding_dim=emb_dim,
            num_faces=int(
                config.get("megaminx_num_faces", _DEFAULT_MEGAMINX_NUM_FACES)
            ),
            use_inverse=bool(config.get("megaminx_use_inverse", True)),
            use_graph_breakpoints=bool(
                config.get("megaminx_use_graph_breakpoints", True)
            ),
            generator_moves=config.get("generator_moves"),
        )
        return enc, out_dim

    raise ValueError(f"Unknown input_encoder_type: {enc_type}")
