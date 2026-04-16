# Defines validated benchmark dataset schemas for evaluation tasks.
"""Dataset schemas for reusable evaluation benchmarks."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LabelType(str, Enum):
    """Primary supervision type attached to a benchmark item."""

    NONE = "none"
    EXACT = "exact"
    BEST_KNOWN = "best_known"
    BASELINE_ONLY = "baseline_only"


class BenchmarkItem(BaseModel):
    """
    One benchmark state plus optional labels and metadata.

    Args:
        item_id: Stable identifier for the item inside a dataset.
        state: Graph state encoded as a tuple of integers.
        family: Optional slice name used for grouped metrics.
        source: Short provenance string for the item.
        label_type: Primary label semantics for the item.
        exact_distance: Exact shortest-path distance when known.
        best_known_length: Best known solution length when exactness is unknown.
        baseline_length: Length returned by a baseline/default solver.
        metadata: Extra serializable metadata for downstream evaluators.

    Raises:
        ValueError: If the state is empty or the declared label type is missing
            its required label payload.

    """

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(..., min_length=1)
    state: tuple[int, ...]
    family: str = "default"
    source: str = "manual"
    label_type: LabelType = LabelType.NONE
    exact_distance: int | None = Field(default=None, ge=0)
    best_known_length: int | None = Field(default=None, ge=0)
    baseline_length: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_item(self) -> BenchmarkItem:
        """
        Validate state payload and declared labels.

        Returns:
            The validated benchmark item.

        Raises:
            ValueError: If the item is malformed.

        """
        if len(self.state) == 0:
            raise ValueError("benchmark states must be non-empty.")
        if self.label_type == LabelType.EXACT and self.exact_distance is None:
            raise ValueError("label_type='exact' requires exact_distance.")
        if (
            self.label_type == LabelType.BEST_KNOWN
            and self.best_known_length is None
        ):
            raise ValueError("label_type='best_known' requires best_known_length.")
        if (
            self.label_type == LabelType.BASELINE_ONLY
            and self.baseline_length is None
        ):
            raise ValueError("label_type='baseline_only' requires baseline_length.")
        return self

    def primary_length(self) -> int | None:
        """
        Return the most relevant known path length for the item.

        Returns:
            Exact distance when available, else best-known length, else baseline
            length, else ``None``.

        """
        if self.exact_distance is not None:
            return int(self.exact_distance)
        if self.best_known_length is not None:
            return int(self.best_known_length)
        if self.baseline_length is not None:
            return int(self.baseline_length)
        return None


class BenchmarkDataset(BaseModel):
    """
    Frozen benchmark dataset shared across evaluators.

    Args:
        name: Stable dataset identifier.
        graph_name: Graph family name, for example ``"pancake"``.
        split: Logical split such as ``"train"``, ``"val"``, or ``"test"``.
        items: Benchmark items contained in the dataset.
        description: Optional human-readable description.
        metadata: Extra serializable dataset-level metadata.

    Raises:
        ValueError: If the dataset is empty.

    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    graph_name: str = Field(..., min_length=1)
    split: str = Field(..., min_length=1)
    items: list[BenchmarkItem]
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset(self) -> BenchmarkDataset:
        """
        Ensure the dataset contains at least one benchmark item.

        Returns:
            The validated dataset.

        Raises:
            ValueError: If no items are present.

        """
        if len(self.items) == 0:
            raise ValueError("benchmark datasets must contain at least one item.")
        return self

    def item_ids(self) -> list[str]:
        """
        Return item identifiers in dataset order.

        Returns:
            Ordered list of item identifiers.

        """
        return [item.item_id for item in self.items]

    def families(self) -> list[str]:
        """
        Return sorted distinct family names used by the dataset.

        Returns:
            Sorted family names.

        """
        return sorted({item.family for item in self.items})

    def exact_items(self) -> list[BenchmarkItem]:
        """
        Return items that carry exact-distance labels.

        Returns:
            Exact-labeled benchmark items in dataset order.

        """
        return [item for item in self.items if item.exact_distance is not None]
