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
_DEFAULT_PUZZLE_EMBEDDING_DIM = 16
_DEFAULT_PUZZLE_NUM_FACES = 12
_MEGAMINX_STICKER_STATE_SIZE = 120
_CHRISTOPHER_JEWEL_STICKER_STATE_SIZE = 48


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


class PuzzleEmbeddingEncoder(nn.Module):
    """
    Encode sticker-permutation puzzles with configurable puzzle-level features.

    This encoder is intended for Megaminx-like and Christopher-Jewel-like state
    tensors where each entry is a unique solved-sticker id. It combines learned
    sticker/position embeddings with exact aggregate features that are cheap at
    inference time:

    - misplaced sticker, face mismatch, and local face-slot mismatch bits,
    - source-face by target-face occupancy counts,
    - optional generator-graph breakpoint bits,
    - optional piece-group misplaced and orientation histograms,
    - optional inverse ``sticker -> current position`` embeddings.

    The intentionally missing feature class is exact projected distance
    databases. Those require separate quotient/PDB construction and lookup
    tables; they are better added as explicit cached side inputs once reliable
    cubie and symmetry metadata is available.

    Args:
        state_size: Number of sticker positions per example.
        num_classes: Number of possible sticker labels. Must cover
            ``0..state_size-1`` for piece/orientation features.
        output_dim: Final feature dimension after projection.
        z_add: Offset added to ``z`` before feature construction.
        dtype: Floating dtype used for the returned tensor.
        embedding_dim: Size of each learned sticker/position embedding.
        num_faces: Number of solved face groups.
        use_site_embeddings: Whether to include position-ordered learned
            sticker/face/slot embeddings.
        use_inverse: Whether to include inverse sticker-position embeddings.
        use_face_features: Whether to include mismatch and face occupancy
            aggregate features.
        use_piece_features: Whether to include per-piece misplaced features.
        use_orientation_features: Whether to include orientation bits,
            histograms, and entropy for configured piece groups.
        use_graph_breakpoints: Whether to include generator-graph breakpoint
            features when ``generator_moves`` is available.
        use_sorted_face_counts: Whether to include sorted face-count features as
            a cheap approximate face-relabeling invariant.
        use_move_delta_features: Whether to include per-generator deltas for
            cheap state-quality signals after one move.
        use_move_cycle_features: Whether to include per-generator affected-cycle
            consistency features.
        use_face_solvedness_features: Whether to include per-face solvedness and
            source-face entropy summaries.
        piece_groups: Optional solved-sticker groups. If omitted, common
            Megaminx ``120 = 20*3 + 30*2`` and Christopher Jewel
            ``48 = 12*2 + 6*4`` layouts are inferred; otherwise singleton
            groups are used.
        corner_group_size: Stored for logging/metadata. Group-size stats are
            emitted for every group length, so callers can interpret this size
            as corners.
        edge_group_size: Stored for logging/metadata. Group-size stats are
            emitted for every group length, so callers can interpret this size
            as edges.
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
        embedding_dim: int = _DEFAULT_PUZZLE_EMBEDDING_DIM,
        num_faces: int = _DEFAULT_PUZZLE_NUM_FACES,
        use_site_embeddings: bool = True,
        use_inverse: bool = True,
        use_face_features: bool = True,
        use_piece_features: bool = True,
        use_orientation_features: bool = True,
        use_graph_breakpoints: bool = True,
        use_sorted_face_counts: bool = True,
        use_move_delta_features: bool = False,
        use_move_cycle_features: bool = False,
        use_face_solvedness_features: bool = False,
        piece_groups: Any | None = None,
        corner_group_size: int = 3,
        edge_group_size: int = 2,
        generator_moves: Any | None = None,
    ) -> None:
        super().__init__()
        if state_size % num_faces != 0:
            raise ValueError(
                "PuzzleEmbeddingEncoder requires state_size to be divisible "
                f"by num_faces, got state_size={state_size}, num_faces={num_faces}."
            )
        if num_classes < state_size:
            raise ValueError(
                "PuzzleEmbeddingEncoder expects unique sticker ids and requires "
                f"num_classes >= state_size, got num_classes={num_classes}, "
                f"state_size={state_size}."
            )

        self.state_size = int(state_size)
        self.num_classes = int(num_classes)
        self.output_dim = int(output_dim)
        self.z_add = int(z_add)
        self.dtype = dtype
        self.embedding_dim = int(embedding_dim)
        self.num_faces = int(num_faces)
        self.face_size = int(state_size // num_faces)
        self.use_site_embeddings = bool(use_site_embeddings)
        self.use_inverse = bool(use_inverse)
        self.use_face_features = bool(use_face_features)
        self.use_piece_features = bool(use_piece_features)
        self.use_orientation_features = bool(use_orientation_features)
        self.use_graph_breakpoints = bool(use_graph_breakpoints)
        self.use_sorted_face_counts = bool(use_sorted_face_counts)
        self.use_move_delta_features = bool(use_move_delta_features)
        self.use_move_cycle_features = bool(use_move_cycle_features)
        self.use_face_solvedness_features = bool(use_face_solvedness_features)
        self.corner_group_size = int(corner_group_size)
        self.edge_group_size = int(edge_group_size)

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
        (
            generator_moves_tensor,
            inverse_generator_moves,
            affected_mask,
            affected_count,
            cycle_edge_src,
            cycle_edge_dst,
            cycle_edge_gen,
            cycle_edge_count,
            cycle_edge_lookup,
        ) = self._build_move_feature_tables(generator_moves)
        self.num_generators = int(generator_moves_tensor.shape[0])
        self.register_buffer("_edge_src", edge_src, persistent=False)
        self.register_buffer("_edge_dst", edge_dst, persistent=False)
        self.register_buffer(
            "_solved_edge_lookup",
            solved_edge_lookup,
            persistent=False,
        )
        self.register_buffer("_generator_moves", generator_moves_tensor, persistent=False)
        self.register_buffer(
            "_inverse_generator_moves",
            inverse_generator_moves,
            persistent=False,
        )
        self.register_buffer("_move_affected_mask", affected_mask, persistent=False)
        self.register_buffer("_move_affected_count", affected_count, persistent=False)
        self.register_buffer("_cycle_edge_src", cycle_edge_src, persistent=False)
        self.register_buffer("_cycle_edge_dst", cycle_edge_dst, persistent=False)
        self.register_buffer("_cycle_edge_gen", cycle_edge_gen, persistent=False)
        self.register_buffer("_cycle_edge_count", cycle_edge_count, persistent=False)
        self.register_buffer("_cycle_edge_lookup", cycle_edge_lookup, persistent=False)

        normalized_piece_groups = self._normalize_piece_groups(piece_groups)
        (
            piece_id_for_index,
            piece_offset_for_index,
            piece_groups_by_len,
        ) = self._build_piece_lookup(normalized_piece_groups)
        self.register_buffer(
            "_piece_id_for_index",
            piece_id_for_index,
            persistent=False,
        )
        self.register_buffer(
            "_piece_offset_for_index",
            piece_offset_for_index,
            persistent=False,
        )
        self.piece_group_lengths = tuple(sorted(piece_groups_by_len))
        self.num_piece_groups = int(len(normalized_piece_groups))
        for group_len in self.piece_group_lengths:
            groups = torch.tensor(piece_groups_by_len[group_len], dtype=torch.long)
            group_ids = torch.tensor(
                [
                    int(piece_id_for_index[int(group[0])].item())
                    for group in piece_groups_by_len[group_len]
                ],
                dtype=torch.long,
            )
            self.register_buffer(
                f"_piece_groups_len_{group_len}",
                groups,
                persistent=False,
            )
            self.register_buffer(
                f"_piece_group_ids_len_{group_len}",
                group_ids,
                persistent=False,
            )

        graph_dim = 0
        if self.use_graph_breakpoints and int(edge_src.numel()) > 0:
            graph_dim = int(edge_src.numel()) + 1

        site_dim = (
            self.state_size * self.embedding_dim if self.use_site_embeddings else 0
        )
        inverse_dim = self.state_size * self.embedding_dim if self.use_inverse else 0
        face_dim = 0
        if self.use_face_features:
            face_dim = (
                (3 * self.state_size)
                + (self.num_faces * self.num_faces)
                + self.num_faces
                + 3
            )
            if self.use_sorted_face_counts:
                face_dim += self.num_faces * self.num_faces

        face_solvedness_dim = 0
        if self.use_face_solvedness_features:
            face_solvedness_dim = (5 * self.num_faces) + 12

        piece_dim = 0
        if self.use_piece_features and self.num_piece_groups > 0:
            piece_dim += self.num_piece_groups
            if self.use_orientation_features:
                piece_dim += self.num_piece_groups
            for group_len in self.piece_group_lengths:
                piece_dim += 2
                if self.use_orientation_features:
                    piece_dim += 2 + int(group_len)

        move_delta_dim = 0
        if self.use_move_delta_features and self.num_generators > 0:
            move_delta_dim = (4 * self.num_generators) + 12

        move_cycle_dim = 0
        if self.use_move_cycle_features and self.num_generators > 0:
            move_cycle_dim = (4 * self.num_generators) + 12

        self.base_output_dim = (
            site_dim
            + inverse_dim
            + face_dim
            + face_solvedness_dim
            + graph_dim
            + piece_dim
            + move_delta_dim
            + move_cycle_dim
        )
        if self.base_output_dim <= 0:
            raise ValueError("PuzzleEmbeddingEncoder has no enabled features.")
        self.proj = nn.Linear(self.base_output_dim, self.output_dim)

    def _normalize_piece_groups(self, piece_groups: Any | None) -> list[list[int]]:
        """Return validated solved-sticker groups covering every index once."""
        if piece_groups is None:
            piece_groups_list = self._infer_default_piece_groups()
        else:
            piece_groups_list = [
                [int(x) for x in group]
                for group in piece_groups
                if len(group) > 0
            ]

        seen: set[int] = set()
        normalized: list[list[int]] = []
        for group in piece_groups_list:
            group_seen: set[int] = set()
            clean_group: list[int] = []
            for idx in group:
                if idx < 0 or idx >= self.state_size:
                    raise ValueError(
                        "piece_groups contains index outside "
                        f"[0, {self.state_size}): {idx}."
                    )
                if idx in seen:
                    raise ValueError(
                        f"piece_groups contains duplicate index {idx}."
                    )
                if idx not in group_seen:
                    clean_group.append(idx)
                    group_seen.add(idx)
            seen.update(clean_group)
            normalized.append(clean_group)

        for idx in range(self.state_size):
            if idx not in seen:
                normalized.append([idx])
        return normalized

    def _infer_default_piece_groups(self) -> list[list[int]]:
        """Infer common sticker-piece group layouts when known."""
        if self.state_size == _MEGAMINX_STICKER_STATE_SIZE:
            corner_groups = [
                list(range(start, start + 3))
                for start in range(0, 60, 3)
            ]
            edge_groups = [
                list(range(start, start + 2))
                for start in range(60, 120, 2)
            ]
            return corner_groups + edge_groups
        if self.state_size == _CHRISTOPHER_JEWEL_STICKER_STATE_SIZE:
            edge_like_groups = [
                list(range(start, start + 2))
                for start in range(0, 24, 2)
            ]
            corner_like_groups = [
                list(range(start, start + 4))
                for start in range(24, 48, 4)
            ]
            return edge_like_groups + corner_like_groups
        return [[idx] for idx in range(self.state_size)]

    def _build_piece_lookup(
        self,
        piece_groups: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, list[list[int]]]]:
        """Build per-index piece ids, offsets, and groups keyed by group length."""
        piece_id_for_index = torch.empty(self.state_size, dtype=torch.long)
        piece_offset_for_index = torch.empty(self.state_size, dtype=torch.long)
        groups_by_len: dict[int, list[list[int]]] = {}
        for piece_id, group in enumerate(piece_groups):
            group_len = int(len(group))
            groups_by_len.setdefault(group_len, []).append(group)
            for offset, idx in enumerate(group):
                piece_id_for_index[int(idx)] = int(piece_id)
                piece_offset_for_index[int(idx)] = int(offset)
        return piece_id_for_index, piece_offset_for_index, groups_by_len

    def _build_graph_edges(
        self,
        generator_moves: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build solved sticker-adjacency edges from generator permutations."""
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

    def _build_move_feature_tables(
        self,
        generator_moves: Any | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Precompute generator tables used by move-local feature families."""
        empty_moves = torch.empty((0, self.state_size), dtype=torch.long)
        empty_float = torch.empty((0, self.state_size), dtype=torch.float32)
        empty_counts = torch.empty(0, dtype=torch.float32)
        empty_idx = torch.empty(0, dtype=torch.long)
        empty_lookup = torch.zeros(
            (0, self.state_size, self.state_size),
            dtype=torch.bool,
        )
        if generator_moves is None:
            return (
                empty_moves,
                empty_moves,
                empty_float,
                empty_counts,
                empty_idx,
                empty_idx,
                empty_idx,
                empty_counts,
                empty_lookup,
            )

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

        num_generators = int(moves.shape[0])
        position_ids = torch.arange(self.state_size, dtype=torch.long)
        inverse_moves = torch.empty_like(moves)
        inverse_moves.scatter_(
            1,
            moves,
            position_ids.view(1, -1).expand(num_generators, -1),
        )
        affected_mask = moves.ne(position_ids.view(1, -1)).to(torch.float32)
        affected_count = affected_mask.sum(dim=1).clamp_min(1.0)

        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_gen: list[int] = []
        edge_lookup = torch.zeros(
            (num_generators, self.state_size, self.state_size),
            dtype=torch.bool,
        )
        for gen_id, move in enumerate(moves.detach().cpu().tolist()):
            seen = [False] * self.state_size
            for start in range(self.state_size):
                if seen[start]:
                    continue
                cycle: list[int] = []
                cur = int(start)
                while not seen[cur]:
                    seen[cur] = True
                    cycle.append(cur)
                    cur = int(move[cur])
                if len(cycle) <= 1:
                    continue
                for idx, src in enumerate(cycle):
                    dst = cycle[(idx + 1) % len(cycle)]
                    edge_src.append(int(src))
                    edge_dst.append(int(dst))
                    edge_gen.append(int(gen_id))
                    edge_lookup[gen_id, int(src), int(dst)] = True
                    edge_lookup[gen_id, int(dst), int(src)] = True

        if not edge_src:
            return (
                moves.contiguous(),
                inverse_moves.contiguous(),
                affected_mask.contiguous(),
                affected_count.contiguous(),
                empty_idx,
                empty_idx,
                empty_idx,
                torch.ones(num_generators, dtype=torch.float32),
                edge_lookup,
            )

        cycle_edge_src = torch.tensor(edge_src, dtype=torch.long)
        cycle_edge_dst = torch.tensor(edge_dst, dtype=torch.long)
        cycle_edge_gen = torch.tensor(edge_gen, dtype=torch.long)
        cycle_edge_count = torch.bincount(
            cycle_edge_gen,
            minlength=num_generators,
        ).to(torch.float32).clamp_min(1.0)
        return (
            moves.contiguous(),
            inverse_moves.contiguous(),
            affected_mask.contiguous(),
            affected_count.contiguous(),
            cycle_edge_src,
            cycle_edge_dst,
            cycle_edge_gen,
            cycle_edge_count,
            edge_lookup,
        )

    def _validate_and_shift(self, z: torch.Tensor) -> torch.Tensor:
        """Validate a state batch and apply the configured label offset."""
        if z.ndim != _EXPECTED_STATE_NDIM or int(z.shape[1]) != self.state_size:
            raise ValueError(
                "PuzzleEmbeddingEncoder expects z with shape "
                f"(batch, state_size={self.state_size}), got {tuple(z.shape)}."
            )
        return z.long() + self.z_add

    def _label_parts(self, labels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split sticker labels into embedding-safe IDs, solved faces, and slots."""
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
        """Compute solved-position, face, and local-slot mismatch bits."""
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
        """Encode stickers in target-position order."""
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

    def _compute_current_positions(self, state_labels: torch.Tensor) -> torch.Tensor:
        """Return current positions in solved-sticker order."""
        batch_size = int(state_labels.shape[0])
        position_ids = self._position_ids.to(state_labels.device)
        current_positions = torch.zeros_like(state_labels)
        current_positions.scatter_(
            1,
            state_labels,
            position_ids.view(1, -1).expand(batch_size, -1),
        )
        return current_positions

    def _encode_inverse_sites(self, current_positions: torch.Tensor) -> torch.Tensor:
        """Encode current positions in solved-sticker order."""
        batch_size = int(current_positions.shape[0])
        position_ids = self._position_ids.to(current_positions.device)
        solved_faces = self._position_face_ids.to(current_positions.device)
        solved_slots = self._position_slot_ids.to(current_positions.device)
        current_faces = current_positions.div(self.face_size, rounding_mode="floor")
        current_slots = current_positions.remainder(self.face_size)

        misplaced = current_positions.ne(position_ids.view(1, -1))
        face_mismatch = current_faces.ne(solved_faces.view(1, -1))
        slot_mismatch = current_slots.ne(solved_slots.view(1, -1))
        status = torch.stack(
            [misplaced, face_mismatch, slot_mismatch],
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

    def _compute_face_features(
        self,
        state_labels: torch.Tensor,
        source_faces: torch.Tensor,
        status_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Build explicit mismatch and face-occupancy features."""
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
        if self.use_sorted_face_counts:
            features.append(
                face_counts.reshape(state_labels.shape[0], -1).sort(dim=1).values
            )
        return torch.cat(features, dim=1)

    def _summarize_per_generator(self, values: torch.Tensor) -> torch.Tensor:
        """Append mean/min/max summaries to per-generator feature values."""
        if int(values.shape[1]) == 0:
            return torch.empty(
                (values.shape[0], 0),
                device=values.device,
                dtype=values.dtype,
            )
        return torch.cat(
            [
                values,
                values.mean(dim=1, keepdim=True),
                values.min(dim=1, keepdim=True).values,
                values.max(dim=1, keepdim=True).values,
            ],
            dim=1,
        )

    def _compute_face_solvedness_features(
        self,
        source_faces: torch.Tensor,
        status_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Build per-face solvedness and source-face entropy features."""
        if not self.use_face_solvedness_features:
            return torch.empty(
                (source_faces.shape[0], 0),
                device=source_faces.device,
                dtype=self.proj.weight.dtype,
            )

        misplaced, face_mismatch, _ = status_parts
        feature_dtype = self.proj.weight.dtype
        target_face_one_hot = self._target_face_one_hot.to(
            device=source_faces.device,
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
        same_face = face_counts.diagonal(dim1=1, dim2=2)
        exact = torch.einsum(
            "pf,bp->bf",
            target_face_one_hot,
            misplaced.logical_not().to(feature_dtype),
        ) / float(self.face_size)
        wrong_face = torch.einsum(
            "pf,bp->bf",
            target_face_one_hot,
            face_mismatch,
        ) / float(self.face_size)
        probs = face_counts / face_counts.sum(dim=2, keepdim=True).clamp_min(1e-6)
        entropy = -(
            probs * probs.clamp_min(1e-6).log()
        ).sum(dim=2) / torch.log(
            torch.tensor(
                float(self.num_faces),
                device=source_faces.device,
                dtype=feature_dtype,
            )
        )
        mixedness = 1.0 - face_counts.max(dim=2).values

        summaries = torch.cat(
            [
                exact.mean(dim=1, keepdim=True),
                exact.min(dim=1, keepdim=True).values,
                exact.max(dim=1, keepdim=True).values,
                same_face.mean(dim=1, keepdim=True),
                same_face.min(dim=1, keepdim=True).values,
                same_face.max(dim=1, keepdim=True).values,
                wrong_face.mean(dim=1, keepdim=True),
                entropy.mean(dim=1, keepdim=True),
                entropy.max(dim=1, keepdim=True).values,
                mixedness.mean(dim=1, keepdim=True),
                mixedness.max(dim=1, keepdim=True).values,
                (exact.ge(1.0 - 1e-6)).to(feature_dtype).mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return torch.cat(
            [
                exact,
                same_face,
                wrong_face,
                entropy,
                mixedness,
                summaries,
            ],
            dim=1,
        )

    def _compute_graph_breakpoint_bits(
        self,
        state_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute generator-graph breakpoint indicators."""
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

    def _compute_piece_features(
        self,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Build piece misplaced, orientation histogram, and entropy features."""
        feature_dtype = self.proj.weight.dtype
        if not self.use_piece_features or self.num_piece_groups <= 0:
            return torch.empty(
                (current_positions.shape[0], 0),
                device=current_positions.device,
                dtype=feature_dtype,
            )

        pos_to_piece = self._piece_id_for_index.to(current_positions.device)
        pos_to_offset = self._piece_offset_for_index.to(current_positions.device)
        batch_size = int(current_positions.shape[0])
        per_piece_misplaced: list[torch.Tensor] = []
        per_piece_wrong_orientation: list[torch.Tensor] = []
        stats: list[torch.Tensor] = []

        for group_len in self.piece_group_lengths:
            groups = getattr(self, f"_piece_groups_len_{group_len}").to(
                current_positions.device
            )
            group_ids = getattr(self, f"_piece_group_ids_len_{group_len}").to(
                current_positions.device
            )
            group_count = int(groups.shape[0])
            current_piece_positions = current_positions.index_select(
                1,
                groups.reshape(-1),
            ).view(batch_size, group_count, int(group_len))
            target_piece_ids = pos_to_piece[current_piece_positions]
            target_offsets = pos_to_offset[current_piece_positions]
            expected_piece_ids = group_ids.view(1, group_count, 1)
            placed = target_piece_ids.eq(expected_piece_ids).all(dim=2)
            misplaced = placed.logical_not()
            misplaced_f = misplaced.to(feature_dtype)
            placed_f = placed.to(feature_dtype)
            per_piece_misplaced.append(misplaced_f)
            stats.extend([
                misplaced_f.mean(dim=1, keepdim=True),
                placed_f.mean(dim=1, keepdim=True),
            ])

            if not self.use_orientation_features:
                continue

            solved_offsets = torch.arange(
                int(group_len),
                device=current_positions.device,
                dtype=torch.long,
            ).view(1, 1, int(group_len))
            deltas = (target_offsets - solved_offsets).remainder(int(group_len))
            consistent = deltas.eq(deltas[:, :, 0:1]).all(dim=2)
            orientation = deltas[:, :, 0]
            orientation_valid = placed & consistent
            wrong_orientation = orientation_valid & orientation.ne(0)
            wrong_f = wrong_orientation.to(feature_dtype)
            per_piece_wrong_orientation.append(wrong_f)

            hist_parts = [
                (orientation_valid & orientation.eq(offset))
                .to(feature_dtype)
                .sum(dim=1, keepdim=True)
                / float(max(1, group_count))
                for offset in range(int(group_len))
            ]
            hist = torch.cat(hist_parts, dim=1)
            probs = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-6)
            entropy = -(
                probs * probs.clamp_min(1e-6).log()
            ).sum(dim=1, keepdim=True)
            if int(group_len) > 1:
                entropy = entropy / torch.log(
                    torch.tensor(
                        float(group_len),
                        device=current_positions.device,
                        dtype=feature_dtype,
                    )
                )
            stats.extend([
                wrong_f.mean(dim=1, keepdim=True),
                entropy.to(feature_dtype),
                hist,
            ])

        parts: list[torch.Tensor] = [
            torch.cat(per_piece_misplaced, dim=1),
            *stats,
            ]
        if self.use_orientation_features and per_piece_wrong_orientation:
            parts.insert(1, torch.cat(per_piece_wrong_orientation, dim=1))
        return torch.cat(parts, dim=1)

    def _piece_misplaced_fraction_from_positions(
        self,
        current_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return fraction of piece groups not occupying their solved group."""
        if self.num_piece_groups <= 0:
            return torch.zeros(
                current_positions.shape[:-1],
                device=current_positions.device,
                dtype=self.proj.weight.dtype,
            )

        pos_to_piece = self._piece_id_for_index.to(current_positions.device)
        misplaced_parts: list[torch.Tensor] = []
        for group_len in self.piece_group_lengths:
            groups = getattr(self, f"_piece_groups_len_{group_len}").to(
                current_positions.device
            )
            group_ids = getattr(self, f"_piece_group_ids_len_{group_len}").to(
                current_positions.device
            )
            group_count = int(groups.shape[0])
            current_piece_positions = current_positions.index_select(
                -1,
                groups.reshape(-1),
            ).view(*current_positions.shape[:-1], group_count, int(group_len))
            target_piece_ids = pos_to_piece[current_piece_positions]
            expected_piece_ids = group_ids.view(
                *((1,) * (current_positions.ndim - 1)),
                group_count,
                1,
            )
            misplaced_parts.append(
                target_piece_ids.eq(expected_piece_ids)
                .all(dim=-1)
                .logical_not()
                .to(self.proj.weight.dtype)
            )
        return torch.cat(misplaced_parts, dim=-1).mean(dim=-1)

    def _compute_move_delta_features(
        self,
        state_labels: torch.Tensor,
        current_positions: torch.Tensor,
        status_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Build per-generator deltas for cheap solvedness proxies."""
        if not self.use_move_delta_features or self.num_generators <= 0:
            return torch.empty(
                (state_labels.shape[0], 0),
                device=state_labels.device,
                dtype=self.proj.weight.dtype,
            )

        generator_moves = self._generator_moves.to(state_labels.device)
        batch_size = int(state_labels.shape[0])
        gen_count = int(generator_moves.shape[0])
        moved_labels = state_labels.index_select(
            1,
            generator_moves.reshape(-1),
        ).view(batch_size, gen_count, self.state_size)
        moved_faces = moved_labels.div(self.face_size, rounding_mode="floor")
        moved_slots = moved_labels.remainder(self.face_size)
        position_ids = self._position_ids.to(state_labels.device)
        position_faces = self._position_face_ids.to(state_labels.device)
        position_slots = self._position_slot_ids.to(state_labels.device)

        current_misplaced, current_face_mismatch, current_slot_mismatch = status_parts
        moved_misplaced = moved_labels.ne(position_ids.view(1, 1, -1))
        moved_face_mismatch = moved_faces.ne(position_faces.view(1, 1, -1))
        moved_slot_mismatch = moved_slots.ne(position_slots.view(1, 1, -1))

        feature_dtype = self.proj.weight.dtype
        scale = float(self.state_size)
        delta_misplaced = (
            moved_misplaced.to(feature_dtype).sum(dim=2)
            - current_misplaced.sum(dim=1, keepdim=True)
        ) / scale
        delta_face = (
            moved_face_mismatch.to(feature_dtype).sum(dim=2)
            - current_face_mismatch.sum(dim=1, keepdim=True)
        ) / scale
        delta_slot = (
            moved_slot_mismatch.to(feature_dtype).sum(dim=2)
            - current_slot_mismatch.sum(dim=1, keepdim=True)
        ) / scale

        inverse_moves = self._inverse_generator_moves.to(current_positions.device)
        moved_positions = inverse_moves.view(1, gen_count, self.state_size).expand(
            batch_size,
            -1,
            -1,
        ).gather(
            2,
            current_positions.view(batch_size, 1, self.state_size).expand(
                -1,
                gen_count,
                -1,
            ),
        )
        current_piece = self._piece_misplaced_fraction_from_positions(current_positions)
        moved_piece = self._piece_misplaced_fraction_from_positions(moved_positions)
        delta_piece = moved_piece - current_piece.view(batch_size, 1)

        return torch.cat(
            [
                self._summarize_per_generator(delta_misplaced),
                self._summarize_per_generator(delta_face),
                self._summarize_per_generator(delta_slot),
                self._summarize_per_generator(delta_piece),
            ],
            dim=1,
        )

    def _compute_move_cycle_features(
        self,
        state_labels: torch.Tensor,
        source_faces: torch.Tensor,
        status_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Build per-generator affected-cycle consistency features."""
        if not self.use_move_cycle_features or self.num_generators <= 0:
            return torch.empty(
                (state_labels.shape[0], 0),
                device=state_labels.device,
                dtype=self.proj.weight.dtype,
            )

        feature_dtype = self.proj.weight.dtype
        affected_mask = self._move_affected_mask.to(
            device=state_labels.device,
            dtype=feature_dtype,
        )
        affected_count = self._move_affected_count.to(
            device=state_labels.device,
            dtype=feature_dtype,
        )
        misplaced, face_mismatch, _ = status_parts
        affected_misplaced = torch.einsum(
            "bs,gs->bg",
            misplaced,
            affected_mask,
        ) / affected_count.view(1, -1)
        affected_face_mismatch = torch.einsum(
            "bs,gs->bg",
            face_mismatch,
            affected_mask,
        ) / affected_count.view(1, -1)

        source_face_one_hot = F.one_hot(
            source_faces,
            num_classes=self.num_faces,
        ).to(feature_dtype)
        face_counts = torch.einsum(
            "bsf,gs->bgf",
            source_face_one_hot,
            affected_mask,
        ) / affected_count.view(1, -1, 1)
        face_probs = face_counts / face_counts.sum(dim=2, keepdim=True).clamp_min(1e-6)
        face_entropy = -(
            face_probs * face_probs.clamp_min(1e-6).log()
        ).sum(dim=2) / torch.log(
            torch.tensor(
                float(self.num_faces),
                device=state_labels.device,
                dtype=feature_dtype,
            )
        )

        if int(self._cycle_edge_src.numel()) == 0:
            cycle_breaks = torch.zeros(
                (state_labels.shape[0], self.num_generators),
                device=state_labels.device,
                dtype=feature_dtype,
            )
        else:
            edge_src = self._cycle_edge_src.to(state_labels.device)
            edge_dst = self._cycle_edge_dst.to(state_labels.device)
            edge_gen = self._cycle_edge_gen.to(state_labels.device)
            edge_lookup = self._cycle_edge_lookup.to(state_labels.device)
            left = state_labels.index_select(1, edge_src)
            right = state_labels.index_select(1, edge_dst)
            is_cycle_edge = edge_lookup[edge_gen, left, right]
            break_values = is_cycle_edge.logical_not().to(feature_dtype)
            edge_gen_one_hot = F.one_hot(
                edge_gen,
                num_classes=self.num_generators,
            ).to(feature_dtype)
            edge_count = self._cycle_edge_count.to(
                device=state_labels.device,
                dtype=feature_dtype,
            )
            cycle_breaks = break_values.matmul(edge_gen_one_hot) / edge_count.view(
                1,
                -1,
            )

        dominant_source_face_gap = 1.0 - face_counts.max(dim=2).values
        return torch.cat(
            [
                self._summarize_per_generator(affected_misplaced),
                self._summarize_per_generator(affected_face_mismatch),
                self._summarize_per_generator(cycle_breaks),
                self._summarize_per_generator(face_entropy + dominant_source_face_gap),
            ],
            dim=1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of sticker-permutation puzzle states.

        Args:
            z: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, output_dim)``.

        """
        labels = self._validate_and_shift(z)
        label_ids, state_labels, source_faces, source_slots = self._label_parts(labels)
        status_parts = self._status_bits(state_labels, source_faces, source_slots)
        status = torch.stack(status_parts, dim=2)
        current_positions: torch.Tensor | None = None

        parts: list[torch.Tensor] = []
        if self.use_site_embeddings:
            parts.append(
                self._encode_position_sites(
                    label_ids,
                    source_faces,
                    source_slots,
                    status,
                )
            )
        if self.use_inverse or self.use_piece_features or self.use_move_delta_features:
            current_positions = self._compute_current_positions(state_labels)
        if self.use_inverse:
            if current_positions is None:
                current_positions = self._compute_current_positions(state_labels)
            parts.append(self._encode_inverse_sites(current_positions))
        if self.use_face_features:
            parts.append(
                self._compute_face_features(
                    state_labels,
                    source_faces,
                    status_parts,
                )
            )
        if self.use_face_solvedness_features:
            parts.append(
                self._compute_face_solvedness_features(
                    source_faces,
                    status_parts,
                )
            )
        graph_breakpoints = self._compute_graph_breakpoint_bits(state_labels)
        if int(graph_breakpoints.shape[1]) > 0:
            parts.extend([
                graph_breakpoints,
                graph_breakpoints.mean(dim=1, keepdim=True),
            ])
        if self.use_piece_features:
            if current_positions is None:
                current_positions = self._compute_current_positions(state_labels)
            parts.append(self._compute_piece_features(current_positions))
        if self.use_move_delta_features:
            if current_positions is None:
                current_positions = self._compute_current_positions(state_labels)
            parts.append(
                self._compute_move_delta_features(
                    state_labels,
                    current_positions,
                    status_parts,
                )
            )
        if self.use_move_cycle_features:
            parts.append(
                self._compute_move_cycle_features(
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
              ``"embedding_flatten"``, ``"lehmer"``,
              ``"lehmer-breakpoints"``, ``"megaminx"``, or ``"puzzle_emb"``.
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

    if enc_type == "puzzle_emb":
        out_dim = int(config.get("input_encoder_out_dim", hd1))
        emb_dim = int(
            config.get("puzzle_embedding_dim")
            or config.get("embedding_dim")
            or _DEFAULT_PUZZLE_EMBEDDING_DIM
        )
        enc = PuzzleEmbeddingEncoder(
            state_size=state_size,
            num_classes=num_classes,
            output_dim=out_dim,
            z_add=z_add,
            dtype=dtype,
            embedding_dim=emb_dim,
            num_faces=int(
                config.get("puzzle_num_faces", _DEFAULT_PUZZLE_NUM_FACES)
            ),
            use_site_embeddings=bool(
                config.get("puzzle_use_site_embeddings", True)
            ),
            use_inverse=bool(config.get("puzzle_use_inverse", True)),
            use_face_features=bool(config.get("puzzle_use_face_features", True)),
            use_piece_features=bool(config.get("puzzle_use_piece_features", True)),
            use_orientation_features=bool(
                config.get("puzzle_use_orientation_features", True)
            ),
            use_graph_breakpoints=bool(
                config.get("puzzle_use_graph_breakpoints", True)
            ),
            use_sorted_face_counts=bool(
                config.get("puzzle_use_sorted_face_counts", True)
            ),
            use_move_delta_features=bool(
                config.get("puzzle_use_move_delta_features", False)
            ),
            use_move_cycle_features=bool(
                config.get("puzzle_use_move_cycle_features", False)
            ),
            use_face_solvedness_features=bool(
                config.get("puzzle_use_face_solvedness_features", False)
            ),
            piece_groups=config.get("puzzle_piece_groups"),
            corner_group_size=int(config.get("puzzle_corner_group_size", 3)),
            edge_group_size=int(config.get("puzzle_edge_group_size", 2)),
            generator_moves=config.get("generator_moves"),
        )
        return enc, out_dim

    raise ValueError(f"Unknown input_encoder_type: {enc_type}")
