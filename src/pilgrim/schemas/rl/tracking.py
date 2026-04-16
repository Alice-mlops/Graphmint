# Defines pydantic schemas for RL file and probe tracking.
"""Schema models for reinforcement-learning tracker outputs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TDProbeEvaluationConfig(BaseModel):
    """
    Probe-evaluation settings for RL trackers.

    Args:
        eval_interval: Number of optimizer steps between probe evaluations.
        rollout_max_steps: Maximum greedy-rollout length for one probe.
        max_logged_probes: Number of per-probe metrics kept in logs.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    eval_interval: int = Field(50, ge=0)
    rollout_max_steps: int = Field(128, ge=1)
    max_logged_probes: int = Field(8, ge=0)


class TDFileTrackerConfig(BaseModel):
    """
    File-backed tracking settings for RL runs.

    Args:
        output_dir: Directory where tracker artifacts will be written.
        step_log_interval: Number of optimizer steps between metric snapshots.
        write_jsonl: Whether to append a full JSONL metric stream.
        write_csv: Whether to write a compact CSV summary.
        print_metrics: Whether to print short human-readable step summaries.
        jsonl_name: JSONL filename used inside ``output_dir``.
        csv_name: CSV filename used inside ``output_dir``.
        metadata_name: Metadata JSON filename used inside ``output_dir``.
        summary_name: Summary JSON filename used inside ``output_dir``.
        probe: Nested probe-evaluation settings.

    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    output_dir: Path
    step_log_interval: int = Field(10, ge=1)
    write_jsonl: bool = True
    write_csv: bool = True
    print_metrics: bool = True
    jsonl_name: str = "step_metrics.jsonl"
    csv_name: str = "step_metrics.csv"
    metadata_name: str = "tracker_metadata.json"
    summary_name: str = "tracker_summary.json"
    probe: TDProbeEvaluationConfig = Field(default_factory=TDProbeEvaluationConfig)
