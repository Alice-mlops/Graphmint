# Writes periodic RL training metrics to files and stdout.
"""File-backed tracker for multi-step TD value-learning runs."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pilgrim.schemas.rl import TDFileTrackerConfig

from .tracking_utils import (
    center_fraction,
    collect_prediction_metrics,
    collect_probe_metrics,
    parameter_statistics,
    predict_scalar_value,
    resolve_primary_loss_payload,
    unique_row_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cayleypy import CayleyGraph

    from pilgrim.schemas.rl import (
        MultiStepTDValueMetrics,
        MultiStepTDValueStepDiagnostics,
    )

    from .multistep_td_value_iteration import MultiStepTDValueTrainer

_CSV_FIELD_ORDER = (
    "step",
    "total_loss",
    "td_loss",
    "primary_loss",
    "replay_size",
    "replay_fill_ratio",
    "learning_rate",
    "step_time_s",
    "batch_size",
    "batch_center_fraction",
    "batch_unique_ratio",
    "value/pred_mean",
    "value/pred_max",
    "value/target_mean",
    "value/target_max",
    "value/residual_abs_mean",
    "value/residual_abs_max",
    "center_pred",
    "frontier_archive_size",
    "frontier_score_mean",
    "frontier_score_max",
    "probe/success_rate",
    "probe/rollout_len_mean",
    "probe/rollout_len_max",
    "elapsed_s",
)


class TDFileMetricsTracker:
    """
    Periodically write compact RL metrics to files and stdout.

    Args:
        config: File-tracker settings.
        graph: Graph used for center checks and probe rollouts.
        hparams: Optional hyperparameter payload stored in metadata.
        group_n: Optional ``n`` identifier stored in metadata.
        probe_states: Optional fixed states monitored during training.
        probe_targets: Optional scalar targets for ``probe_states``.

    Raises:
        ValueError: If probe states and targets have inconsistent shapes.

    """

    def __init__(
        self,
        config: TDFileTrackerConfig,
        graph: CayleyGraph,
        *,
        hparams: dict[str, Any] | None = None,
        group_n: int | None = None,
        probe_states: torch.Tensor | None = None,
        probe_targets: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.graph = graph
        self.hparams = {} if hparams is None else dict(hparams)
        self.group_n = group_n
        self.probe_states = (
            torch.as_tensor(probe_states).long().cpu()
            if probe_states is not None
            else None
        )
        self.probe_targets = (
            torch.as_tensor(probe_targets).reshape(-1).float().cpu()
            if probe_targets is not None
            else None
        )
        if (
            self.probe_states is not None
            and self.probe_targets is not None
            and self.probe_states.shape[0] != self.probe_targets.shape[0]
        ):
            raise ValueError(
                "probe_states and probe_targets must contain the same number of items."
            )

        self.start_time = 0.0
        self._jsonl_handle = None
        self._csv_handle = None
        self._csv_writer: csv.DictWriter[str] | None = None

    @property
    def output_dir(self) -> Path:
        """Return the configured output directory."""
        return Path(self.config.output_dir)

    def on_fit_start(self, trainer: MultiStepTDValueTrainer) -> None:
        """
        Prepare tracker output files and write run metadata.

        Args:
            trainer: Active multi-step TD value trainer.

        """
        self.start_time = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_name": trainer.model.__class__.__name__,
            "graph_device": str(getattr(self.graph, "device", "cpu")),
            "group_n": self.group_n,
            "hparams": _to_serializable(self.hparams),
            "trainer_config": _to_serializable(trainer.config.to_log_dict()),
            "probe_shape": (
                None if self.probe_states is None else list(self.probe_states.shape)
            ),
        }
        (self.output_dir / self.config.metadata_name).write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        if self.config.write_jsonl:
            self._jsonl_handle = (self.output_dir / self.config.jsonl_name).open(
                "w", encoding="utf-8"
            )
        if self.config.write_csv:
            self._csv_handle = (self.output_dir / self.config.csv_name).open(
                "w", encoding="utf-8", newline=""
            )
            self._csv_writer = csv.DictWriter(
                self._csv_handle,
                fieldnames=list(_CSV_FIELD_ORDER),
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
            self._csv_handle.flush()

    def on_train_step_end(
        self,
        trainer: MultiStepTDValueTrainer,
        diagnostics: MultiStepTDValueStepDiagnostics,
    ) -> None:
        """
        Write one periodic metric snapshot.

        Args:
            trainer: Active multi-step TD value trainer.
            diagnostics: Step-level diagnostics produced by the trainer.

        """
        should_log_step = diagnostics.step % int(self.config.step_log_interval) == 0
        should_log_probe = self._should_log_probes(diagnostics.step)
        if not should_log_step and not should_log_probe:
            return

        metrics = self._build_metrics(
            trainer=trainer,
            diagnostics=diagnostics,
            include_probes=should_log_probe,
        )
        if self._jsonl_handle is not None:
            self._jsonl_handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            self._jsonl_handle.flush()
        if self._csv_writer is not None:
            self._csv_writer.writerow(metrics)
            assert self._csv_handle is not None
            self._csv_handle.flush()
        if self.config.print_metrics:
            print(self._format_log_line(metrics))

    def on_fit_end(
        self,
        trainer: MultiStepTDValueTrainer,
        history: Sequence[MultiStepTDValueMetrics],
    ) -> None:
        """
        Finalize tracker outputs and write a small summary.

        Args:
            trainer: Active multi-step TD value trainer.
            history: Metrics returned by completed optimizer steps.

        """
        del trainer
        summary = {
            "fit_time_s": float(time.perf_counter() - self.start_time),
            "num_steps": len(history),
            "final_step": (None if not history else int(history[-1].step)),
            "final_total_loss": (
                None if not history else float(history[-1].total_loss)
            ),
            "final_td_loss": (None if not history else float(history[-1].td_loss)),
            "final_replay_size": (
                None if not history else int(history[-1].replay_size)
            ),
        }
        (self.output_dir / self.config.summary_name).write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        self.close()

    def close(self) -> None:
        """Close any open tracker file handles."""
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None
        if self._csv_handle is not None:
            self._csv_handle.close()
            self._csv_handle = None
        self._csv_writer = None

    def _should_log_probes(self, step: int) -> bool:
        """
        Return whether probe metrics should be evaluated at ``step``.

        Args:
            step: One-based optimizer step index.

        Returns:
            ``True`` when probe metrics should be evaluated.

        """
        if self.probe_states is None:
            return False
        if int(self.config.probe.eval_interval) <= 0:
            return False
        return step % int(self.config.probe.eval_interval) == 0

    def _build_metrics(
        self,
        *,
        trainer: MultiStepTDValueTrainer,
        diagnostics: MultiStepTDValueStepDiagnostics,
        include_probes: bool,
    ) -> dict[str, float]:
        """
        Build one flat metric payload from trainer diagnostics.

        Args:
            trainer: Active trainer.
            diagnostics: Step diagnostics emitted by the trainer.
            include_probes: Whether to append probe metrics.

        Returns:
            Flat metric dictionary ready for JSONL/CSV serialization.

        """
        _, primary_loss_value = resolve_primary_loss_payload(diagnostics)
        metrics: dict[str, float] = {
            "step": float(diagnostics.step),
            "total_loss": float(diagnostics.total_loss),
            "td_loss": float(diagnostics.td_loss),
            "primary_loss": float(primary_loss_value),
            "replay_size": float(diagnostics.replay_size),
            "replay_fill_ratio": float(diagnostics.replay_fill_ratio),
            "learning_rate": float(diagnostics.learning_rate),
            "step_time_s": float(diagnostics.step_time_s),
            "batch_size": float(diagnostics.batch_states.shape[0]),
            "batch_center_fraction": center_fraction(
                diagnostics.batch_states,
                self.graph,
            ),
            "batch_unique_ratio": unique_row_ratio(diagnostics.batch_states),
            "train_target_sync": float(int(diagnostics.target_sync_applied)),
            "elapsed_s": float(time.perf_counter() - self.start_time),
            "frontier_archive_size": float(
                getattr(diagnostics, "frontier_archive_size", 0)
            ),
            "frontier_score_mean": float(
                getattr(diagnostics, "frontier_score_mean", 0.0) or 0.0
            ),
            "frontier_score_max": float(
                getattr(diagnostics, "frontier_score_max", 0.0) or 0.0
            ),
        }
        if diagnostics.lipschitz_loss is not None:
            metrics["lipschitz_loss"] = float(diagnostics.lipschitz_loss)

        metrics.update(
            collect_prediction_metrics(
                predictions=diagnostics.predictions,
                targets=diagnostics.targets,
            )
        )
        param_global_norm, param_max_abs = parameter_statistics(trainer.model)
        if param_global_norm is not None:
            metrics["param_global_norm"] = float(param_global_norm)
        if param_max_abs is not None:
            metrics["param_max_abs"] = float(param_max_abs)
        metrics["center_pred"] = predict_scalar_value(
            trainer.model,
            torch.as_tensor(self.graph.central_state).view(1, -1),
        )
        if include_probes:
            metrics.update(
                collect_probe_metrics(
                    model=trainer.model,
                    graph=self.graph,
                    probe_states=self.probe_states,
                    probe_targets=self.probe_targets,
                    rollout_max_steps=int(self.config.probe.rollout_max_steps),
                    max_logged_probes=int(self.config.probe.max_logged_probes),
                )
            )
        return metrics

    @staticmethod
    def _format_log_line(metrics: dict[str, float]) -> str:
        """
        Format a concise human-readable step log line.

        Args:
            metrics: Flat metric dictionary.

        Returns:
            One-line printable summary.

        """
        parts = [
            f"step={int(metrics['step'])}",
            f"loss={metrics['total_loss']:.6f}",
            f"td={metrics['td_loss']:.6f}",
            f"pred_max={metrics.get('value/pred_max', float('nan')):.4f}",
            f"target_max={metrics.get('value/target_max', float('nan')):.4f}",
            f"replay={int(metrics['replay_size'])}",
            f"center={metrics.get('center_pred', float('nan')):.4f}",
            f"t={metrics['step_time_s']:.2f}s",
        ]
        if "probe/success_rate" in metrics:
            parts.append(f"probe={metrics['probe/success_rate']:.3f}")
        return " | ".join(parts)


def _to_serializable(value: Any) -> Any:
    """
    Convert arbitrary values to JSON-serializable primitives.

    Args:
        value: Input value to convert.

    Returns:
        JSON-friendly primitive, list, or dictionary.

    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        result: Any = value
    elif isinstance(value, Path | torch.dtype):
        result = str(value)
    elif isinstance(value, torch.Tensor):
        if value.ndim == 0:
            result = float(value.detach().cpu().item())
        else:
            result = value.detach().cpu().tolist()
    elif isinstance(value, dict):
        result = {str(key): _to_serializable(val) for key, val in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_to_serializable(item) for item in value]
    else:
        result = str(value)
    return result
