# Defines serializable run specs for distributed RL training entrypoints.
"""Schema models for distributed multi-step TD training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .multistep_td_value_iteration import MultiStepTDValueConfig
from .tracking import TDFileTrackerConfig


class DistributedMultiStepTDRunSpec(BaseModel):
    """
    Serializable run spec consumed by the distributed RL CLI entrypoint.

    Args:
        n: Pancake size solved by the run.
        seed: Base random seed used for all workers.
        graph_batch_size: Graph batch size used when constructing the graph.
        model_kwargs: Model constructor payload.
        trainer_config: Multi-step TD trainer config.
        run_dir: Output directory for checkpoints and summaries.
        history_name: CSV filename used for fit history.
        summary_name: JSON filename used for the notebook summary.
        model_filename: Optional output checkpoint filename.
        profile: Optional profile label stored in tracker metadata.
        notebook_name: Optional notebook label stored in tracker metadata.
        file_tracker: Optional rank-zero file tracker config.
        probe_count: Number of fixed probe states generated for tracking.
        probe_walk_length: Random-walk length used to build fixed probes.
        initial_checkpoint_path: Optional initial model checkpoint loaded before
            DDP training starts.
        enable_training: Whether the entrypoint should execute ``fit``.
        hparams: Optional extra metadata copied into tracker files.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n: int = Field(..., ge=1)
    seed: int = 42
    graph_batch_size: int = Field(..., ge=1)
    model_kwargs: dict[str, Any]
    trainer_config: MultiStepTDValueConfig
    run_dir: Path
    history_name: str = "history.csv"
    summary_name: str = "summary.json"
    model_filename: str | None = None
    profile: str | None = None
    notebook_name: str | None = None
    file_tracker: TDFileTrackerConfig | None = None
    probe_count: int = Field(0, ge=0)
    probe_walk_length: int | None = Field(default=None, ge=1)
    initial_checkpoint_path: Path | None = None
    enable_training: bool = True
    hparams: dict[str, Any] = Field(default_factory=dict)
