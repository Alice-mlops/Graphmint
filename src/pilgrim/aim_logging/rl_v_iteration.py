# Provides Aim tracking helpers for RL value-learning experiments.
"""Aim logging for reinforcement-learning value-learning runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from aim import Run
from torch import nn

from pilgrim.rl.policies import greedy_rollout_from_value
from pilgrim.rl.transitions import central_state_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cayleypy import CayleyGraph

    from pilgrim.rl.fitted_value_iteration import (
        FittedValueIterationMetrics,
        FittedValueIterationStepDiagnostics,
        FittedValueIterationTrainer,
    )
    from pilgrim.rl.multistep_td_value_iteration import MultiStepTDValueTrainer
    from pilgrim.schemas.rl import (
        MultiStepTDValueMetrics,
        MultiStepTDValueStepDiagnostics,
    )

_DEFAULT_AIM_REPO = Path("/home/seregin/.local/share/aim/repo")
_EXPECTED_STATE_NDIM = 2


@dataclass(slots=True)
class RLFittedValueIterationAimConfig:
    """
    Configuration for Aim tracking of fitted value iteration.

    Args:
        experiment: Aim experiment name.
        repo: Aim repository path. Defaults to the shared machine-wide repo.
        tags: Run tags attached on creation.
        stage: Logical stage name stored under ``meta/stage``.
        notebook: Optional notebook identifier stored in run metadata.
        model_name: Optional model name stored in run metadata.
        extra_meta: Additional metadata written under ``meta/*``.
        track_every_n_steps: Log scalar step metrics every N optimizer steps.
        probe_eval_interval: Evaluate fixed probe states every N optimizer steps.
        probe_rollout_max_steps: Max greedy-rollout length for probe evaluation.
        max_logged_probes: Maximum number of individual probe metrics logged.

    """

    experiment: str = "pilgrim-rl"
    repo: Path | None = _DEFAULT_AIM_REPO
    tags: list[str] = field(default_factory=lambda: ["pilgrim-rl", "value-iteration"])
    stage: str = "rl_v_iteration"
    notebook: str | None = None
    model_name: str | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)
    track_every_n_steps: int = 1
    probe_eval_interval: int = 25
    probe_rollout_max_steps: int = 128
    max_logged_probes: int = 8


def to_aim_serializable(value: Any) -> Any:
    """
    Convert arbitrary values to Aim-storable primitives.

    Args:
        value: Input value to convert.

    Returns:
        Value converted to an Aim-storable primitive, list, or dict.

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
        result = {str(key): to_aim_serializable(val) for key, val in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [to_aim_serializable(item) for item in value]
    else:
        result = str(value)
    return result


class RLFittedValueIterationAimTracker:
    """
    Aim tracker for fitted value iteration.

    Args:
        config: Aim-tracking configuration.
        graph: Graph used for center-state checks and greedy probe rollouts.
        hparams: Optional hyperparameter payload stored under ``hparams``.
        group_n: Optional ``n`` identifier stored under ``group/n``.
        probe_states: Optional fixed states monitored during training.
        probe_targets: Optional scalar targets for ``probe_states``.

    Raises:
        ValueError: If probe states and targets have inconsistent shapes.

    """

    def __init__(
        self,
        config: RLFittedValueIterationAimConfig,
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
            _normalize_states(probe_states) if probe_states is not None else None
        )
        self.probe_targets = (
            torch.as_tensor(probe_targets).reshape(-1).float()
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

        self.run: Run | None = None
        self.start_time = 0.0

    def on_fit_start(self, trainer: FittedValueIterationTrainer) -> None:
        """
        Open the Aim run and store run-level metadata.

        Args:
            trainer: Active fitted-value-iteration trainer.

        """
        self.start_time = time.perf_counter()
        self.run = _open_run(self.config)
        for tag in self.config.tags:
            self.run.add_tag(tag)

        self.run["meta/stage"] = self.config.stage
        if self.config.notebook is not None:
            self.run["meta/notebook"] = self.config.notebook
        model_name = self.config.model_name or trainer.model.__class__.__name__
        self.run["meta/model"] = model_name
        for key, value in self.config.extra_meta.items():
            self.run[f"meta/{key}"] = to_aim_serializable(value)

        hparams = dict(self.hparams)
        hparams.setdefault("rl_config", trainer.config.to_log_dict())
        hparams.setdefault("graph_device", str(getattr(self.graph, "device", "cpu")))
        self.run["hparams"] = to_aim_serializable(hparams)

        if self.group_n is not None:
            self.run["group/n"] = int(self.group_n)

        self.run["graph/central_state"] = to_aim_serializable(self.graph.central_state)
        if self.probe_states is not None:
            self.run["probes/states"] = to_aim_serializable(self.probe_states)
        if self.probe_targets is not None:
            self.run["probes/targets"] = to_aim_serializable(self.probe_targets)

    def on_train_step_end(
        self,
        trainer: FittedValueIterationTrainer | MultiStepTDValueTrainer,
        diagnostics: FittedValueIterationStepDiagnostics
        | MultiStepTDValueStepDiagnostics,
    ) -> None:
        """
        Log step diagnostics into Aim.

        Args:
            trainer: Active fitted-value-iteration trainer.
            diagnostics: Step-level diagnostics produced by the trainer.

        """
        if self.run is None:
            return
        if int(self.config.track_every_n_steps) <= 0:
            return
        if diagnostics.step % int(self.config.track_every_n_steps) != 0:
            return

        primary_loss_name, primary_loss_value = _resolve_primary_loss_payload(
            diagnostics
        )
        metrics = {
            "train/total_loss": float(diagnostics.total_loss),
            "train/backup_loss": float(primary_loss_value),
            f"train/{primary_loss_name}": float(primary_loss_value),
            "replay/size": float(diagnostics.replay_size),
            "replay/fill_ratio": float(diagnostics.replay_fill_ratio),
            "optimizer/lr": float(diagnostics.learning_rate),
            "time/step_s": float(diagnostics.step_time_s),
            "batch/size": float(diagnostics.batch_states.shape[0]),
            "batch/center_fraction": _center_fraction(
                diagnostics.batch_states, self.graph
            ),
            "batch/unique_ratio": _unique_row_ratio(diagnostics.batch_states),
            "train/target_sync": float(int(diagnostics.target_sync_applied)),
            "time/elapsed_s": float(time.perf_counter() - self.start_time),
        }
        _maybe_add_frontier_metrics(metrics, diagnostics)
        if diagnostics.step_time_s > 0.0:
            metrics["batch/examples_per_s"] = float(
                diagnostics.batch_states.shape[0] / diagnostics.step_time_s
            )
        _maybe_add_scalar_metric(
            metrics,
            "train/lipschitz_loss",
            diagnostics.lipschitz_loss,
        )
        _maybe_add_scalar_metric(
            metrics,
            "grad/global_norm",
            diagnostics.gradient_global_norm,
        )
        _maybe_add_scalar_metric(
            metrics,
            "grad/max_abs",
            diagnostics.gradient_max_abs,
        )
        _maybe_add_scalar_metric(
            metrics,
            "frontier/score_mean",
            getattr(diagnostics, "frontier_score_mean", None),
        )
        _maybe_add_scalar_metric(
            metrics,
            "frontier/score_max",
            getattr(diagnostics, "frontier_score_max", None),
        )

        metrics.update(_tensor_stats("value/pred", diagnostics.predictions))
        metrics.update(_tensor_stats("value/target", diagnostics.targets))
        residual = diagnostics.predictions.float() - diagnostics.targets.float()
        metrics.update(_tensor_stats("value/residual", residual))
        metrics["value/residual_abs_mean"] = float(residual.abs().mean().item())
        metrics["value/residual_abs_max"] = float(residual.abs().max().item())

        param_global_norm, param_max_abs = _parameter_statistics(trainer.model)
        if param_global_norm is not None:
            metrics["param/global_norm"] = float(param_global_norm)
        if param_max_abs is not None:
            metrics["param/max_abs"] = float(param_max_abs)

        metrics["value/center_pred"] = _predict_scalar_value(
            trainer.model,
            torch.as_tensor(self.graph.central_state).view(1, -1),
        )

        if self._should_log_probes(diagnostics.step):
            metrics.update(self._collect_probe_metrics(trainer.model))

        self._track_metrics(metrics, step=diagnostics.step)

    def on_fit_end(
        self,
        trainer: FittedValueIterationTrainer | MultiStepTDValueTrainer,
        history: Sequence[FittedValueIterationMetrics | MultiStepTDValueMetrics],
    ) -> None:
        """
        Finalize the Aim run.

        Args:
            trainer: Active fitted-value-iteration trainer.
            history: Metrics returned by completed optimizer steps.

        """
        del trainer
        if self.run is None:
            return

        metrics = {
            "time/fit_s": float(time.perf_counter() - self.start_time),
            "fit/num_steps": float(len(history)),
        }
        if history:
            last = history[-1]
            primary_loss_name, primary_loss_value = _resolve_primary_loss_payload(last)
            metrics["fit/final_total_loss"] = float(last.total_loss)
            metrics["fit/final_backup_loss"] = float(primary_loss_value)
            metrics[f"fit/final_{primary_loss_name}"] = float(primary_loss_value)
            metrics["fit/final_replay_size"] = float(last.replay_size)
            if last.lipschitz_loss is not None:
                metrics["fit/final_lipschitz_loss"] = float(last.lipschitz_loss)

        self._track_metrics(metrics, step=len(history))
        self.run.close()
        self.run = None

    def _collect_probe_metrics(self, model: nn.Module) -> dict[str, float]:
        """
        Compute fixed-probe metrics for the current model snapshot.

        Args:
            model: Online value model to evaluate.

        Returns:
            Dictionary of scalar probe metrics.

        """
        if self.probe_states is None:
            return {}

        metrics: dict[str, float] = {}
        probe_values = _predict_values(model, self.probe_states)
        metrics.update(_tensor_stats("probe/value", probe_values))

        if self.probe_targets is not None:
            probe_residual = probe_values.float() - self.probe_targets.float()
            metrics.update(_tensor_stats("probe/residual", probe_residual))
            metrics["probe/residual_abs_mean"] = float(
                probe_residual.abs().mean().item()
            )

        success_values: list[float] = []
        rollout_lengths: list[float] = []
        max_logged_probes = min(
            int(self.config.max_logged_probes),
            int(self.probe_states.shape[0]),
        )

        for probe_idx in range(int(self.probe_states.shape[0])):
            state = self.probe_states[probe_idx]
            path = greedy_rollout_from_value(
                model,
                self.graph,
                state,
                max_steps=int(self.config.probe_rollout_max_steps),
            )
            reached_center = _rollout_reaches_center(self.graph, state, path)
            success_values.append(float(int(reached_center)))
            rollout_lengths.append(float(len(path)))

            if probe_idx < max_logged_probes:
                metrics[f"probe/value_{probe_idx:02d}"] = float(
                    probe_values[probe_idx].item()
                )
                metrics[f"probe/rollout_len_{probe_idx:02d}"] = float(len(path))
                metrics[f"probe/reached_center_{probe_idx:02d}"] = float(
                    int(reached_center)
                )
                if self.probe_targets is not None:
                    metrics[f"probe/target_{probe_idx:02d}"] = float(
                        self.probe_targets[probe_idx].item()
                    )

        if success_values:
            metrics["probe/success_rate"] = float(
                sum(success_values) / len(success_values)
            )
            metrics["probe/rollout_len_mean"] = float(
                sum(rollout_lengths) / len(rollout_lengths)
            )
            metrics["probe/rollout_len_max"] = float(max(rollout_lengths))

        return metrics

    def _should_log_probes(self, step: int) -> bool:
        """
        Return whether probe metrics should be logged at ``step``.

        Args:
            step: One-based optimizer step index.

        Returns:
            ``True`` when probe metrics should be evaluated.

        """
        if self.probe_states is None:
            return False
        if int(self.config.probe_eval_interval) <= 0:
            return False
        return step % int(self.config.probe_eval_interval) == 0

    def _track_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        """
        Track a flat dictionary of scalar metrics.

        Args:
            metrics: Metric mapping to log.
            step: Step value attached to each metric.

        """
        if self.run is None:
            return
        context = {"phase": "train"}
        for name, value in metrics.items():
            self.run.track(float(value), name=name, step=int(step), context=context)


def _open_run(config: RLFittedValueIterationAimConfig) -> Run:
    """
    Open an Aim run with repository fallback handling.

    Args:
        config: Aim configuration.

    Returns:
        Opened Aim run.

    """
    repo_arg = str(config.repo) if config.repo is not None else None
    try:
        if repo_arg is None:
            return Run(experiment=config.experiment)
        return Run(experiment=config.experiment, repo=repo_arg)
    except RuntimeError:
        fallback_repo = str(Path.cwd())
        return Run(experiment=config.experiment, repo=fallback_repo)


def _normalize_states(states: torch.Tensor) -> torch.Tensor:
    """
    Normalize states to a two-dimensional long tensor.

    Args:
        states: Input state tensor.

    Returns:
        Tensor with shape ``(batch, state_size)`` and dtype ``torch.long``.

    Raises:
        ValueError: If the tensor cannot be normalized to rank ``2``.

    """
    data = torch.as_tensor(states).long()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.ndim != _EXPECTED_STATE_NDIM:
        raise ValueError(
            "states must have shape (batch, state_size) or (state_size,), "
            f"got {tuple(data.shape)}."
        )
    return data.contiguous()


def _tensor_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    """
    Compute standard summary statistics for a tensor.

    Args:
        prefix: Metric-name prefix.
        values: Tensor whose values should be summarized.

    Returns:
        Dictionary with ``mean``, ``std``, ``min``, and ``max`` metrics.

    """
    tensor = torch.as_tensor(values).detach().float().reshape(-1)
    if tensor.numel() == 0:
        return {}
    std_value = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
    return {
        f"{prefix}_mean": float(tensor.mean().item()),
        f"{prefix}_std": std_value,
        f"{prefix}_min": float(tensor.min().item()),
        f"{prefix}_max": float(tensor.max().item()),
    }


def _unique_row_ratio(states: torch.Tensor) -> float:
    """
    Compute the fraction of unique rows in a batch of states.

    Args:
        states: Batched states.

    Returns:
        Number of unique rows divided by batch size.

    """
    batch = _normalize_states(states).detach().cpu()
    if batch.shape[0] == 0:
        return 0.0
    unique_rows = torch.unique(batch, dim=0).shape[0]
    return float(unique_rows) / float(batch.shape[0])


def _center_fraction(states: torch.Tensor, graph: CayleyGraph) -> float:
    """
    Compute the fraction of states equal to the center.

    Args:
        states: Batched states.
        graph: Graph providing the center state.

    Returns:
        Fraction of rows matching ``graph.central_state``.

    """
    batch = _normalize_states(states)
    return float(central_state_mask(batch, graph.central_state).float().mean().item())


def _parameter_statistics(model: nn.Module) -> tuple[float | None, float | None]:
    """
    Compute simple parameter-magnitude diagnostics.

    Args:
        model: Model whose parameters should be summarized.

    Returns:
        Tuple ``(global_l2_norm, max_abs_value)``. Values are ``None`` when the
        model has no parameters.

    """
    squared_sum = 0.0
    max_abs = 0.0
    has_params = False
    for param in model.parameters():
        data = param.detach().float()
        squared_sum += float(torch.sum(data * data).item())
        max_abs = max(max_abs, float(data.abs().max().item()))
        has_params = True
    if not has_params:
        return None, None
    return squared_sum**0.5, max_abs


def _predict_values(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """
    Predict scalar values for a batch of states.

    Args:
        model: Value model to evaluate.
        states: Batch of input states.

    Returns:
        One-dimensional tensor of predictions on CPU.

    """
    batch = _normalize_states(states)
    device = _model_device(model)
    model.eval()
    with torch.no_grad():
        values = model(batch.to(device).long()).detach().reshape(-1).float()
    return values.cpu()


def _resolve_primary_loss_payload(payload: Any) -> tuple[str, float]:
    """
    Resolve the trainer-specific primary backup loss from a payload object.

    Args:
        payload: Metrics or diagnostics object emitted by an RL trainer.

    Returns:
        Tuple of ``(loss_name, loss_value)``.

    Raises:
        AttributeError: If no supported primary loss is present.

    """
    if hasattr(payload, "bellman_loss"):
        return "bellman_loss", float(payload.bellman_loss)
    if hasattr(payload, "td_loss"):
        return "td_loss", float(payload.td_loss)
    raise AttributeError("payload does not expose bellman_loss or td_loss.")


def _maybe_add_scalar_metric(
    metrics: dict[str, float],
    name: str,
    value: float | None,
) -> None:
    """
    Add a scalar metric only when the value is present.

    Args:
        metrics: Mutable metric dictionary.
        name: Metric name to populate.
        value: Optional metric value.

    """
    if value is None:
        return
    metrics[name] = float(value)


def _maybe_add_frontier_metrics(
    metrics: dict[str, float],
    diagnostics: Any,
) -> None:
    """
    Add frontier metrics when the diagnostics payload provides them.

    Args:
        metrics: Mutable metric dictionary.
        diagnostics: Step diagnostics emitted by the trainer.

    """
    frontier_archive_size = getattr(diagnostics, "frontier_archive_size", None)
    if frontier_archive_size is None:
        return

    metrics["frontier/size"] = float(frontier_archive_size)
    metrics["frontier/fill_ratio"] = float(
        getattr(diagnostics, "frontier_archive_fill_ratio", 0.0)
    )
    metrics["frontier/batch_size"] = float(
        getattr(diagnostics, "frontier_batch_size", 0)
    )
    metrics["frontier/refresh_applied"] = float(
        int(getattr(diagnostics, "frontier_refresh_applied", False))
    )
    metrics["frontier/candidate_count"] = float(
        getattr(diagnostics, "frontier_candidate_count", 0)
    )
    metrics["frontier/unique_candidate_count"] = float(
        getattr(diagnostics, "frontier_unique_candidate_count", 0)
    )
    metrics["frontier/selected_count"] = float(
        getattr(diagnostics, "frontier_selected_count", 0)
    )
    metrics["frontier/admitted"] = float(getattr(diagnostics, "frontier_admitted", 0))
    metrics["frontier/updated"] = float(getattr(diagnostics, "frontier_updated", 0))
    metrics["frontier/replaced"] = float(getattr(diagnostics, "frontier_replaced", 0))


def _predict_scalar_value(model: nn.Module, states: torch.Tensor) -> float:
    """
    Predict one scalar value for a singleton state batch.

    Args:
        model: Value model to evaluate.
        states: Singleton batch of input states.

    Returns:
        Scalar prediction.

    """
    values = _predict_values(model, states)
    return float(values[0].item())


def _model_device(model: nn.Module) -> torch.device:
    """
    Infer the device used by a model.

    Args:
        model: Model whose device should be inferred.

    Returns:
        Device of the first parameter, or CPU when the model is parameterless.

    """
    param = next(model.parameters(), None)
    if param is None:
        return torch.device("cpu")
    return param.device


def _rollout_reaches_center(
    graph: CayleyGraph,
    start_state: torch.Tensor,
    path: Sequence[int],
) -> bool:
    """
    Return whether a rollout path reaches the center.

    Args:
        graph: Graph used for state transitions.
        start_state: Starting state for the rollout.
        path: Sequence of generator indices applied in order.

    Returns:
        ``True`` when the final state equals the center.

    """
    state = _normalize_states(start_state).to(getattr(graph, "device", "cpu"))
    if bool(central_state_mask(state, graph.central_state).item()):
        return True

    for action in path:
        next_state = torch.empty_like(state)
        graph.apply_generator_batched(int(action), state, next_state)
        state = next_state
        if bool(central_state_mask(state, graph.central_state).item()):
            return True
    return False
