"""Training utilities for Pilgrim models."""

import math
import re
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import torch
from cayleypy import CayleyGraph, Predictor
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm.auto import tqdm

from .benchmarks import small_inference_speed_benchmark
from .graph_utils import subsample_xy, y_stats
from .losses import lipschitz_expansion_loss
from .lr_scheduler_utils import lr_scheduler_ctor_from_cfg, step_lr_scheduler


def _infer_module_device_type(module: nn.Module) -> str:
    """
    Infer the device type for a module (e.g. "cuda" or "cpu").

    Args:
        module: Module to inspect.

    Returns:
        Device type string.

    """
    param = next(module.parameters(), None)
    if param is not None:
        return param.device.type
    buf = next(module.buffers(), None)
    if buf is not None:
        return buf.device.type
    return "cpu"


def _should_enable_autocast(dtype: torch.dtype, *, device_type: str) -> bool:
    """
    Decide whether to enable autocast for the given dtype/device.

    Args:
        dtype: Desired mixed-precision dtype.
        device_type: Device type string (e.g. "cuda", "cpu").

    Returns:
        True if autocast should be enabled.

    """
    if dtype == torch.bfloat16:
        return device_type in {"cuda", "cpu"}
    if dtype == torch.float16:
        return device_type == "cuda"
    return False


def _should_use_log_scale(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    *,
    threshold: float,
) -> bool:
    if threshold <= 0:
        return False
    all_loss = list(train_loss) + list(val_loss)
    pos = [float(v) for v in all_loss if v is not None and float(v) > 0.0]
    if not pos:
        return False
    ymin = min(pos)
    ymax = max(pos)
    return ymin > 0.0 and (ymax / ymin) >= threshold


def _format_lr(opt: torch.optim.Optimizer) -> str:
    lr = opt.param_groups[0].get("lr", None)
    return f" | lr {lr:.3e}" if isinstance(lr, (float, int)) else ""


def _global_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """Compute the global L2 norm of gradients across parameters."""
    total_sq = 0.0
    for param in params:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total_sq += grad_norm * grad_norm
    return total_sq**0.5


def grad_stats_residual_blocks(model: nn.Module) -> dict[str, float]:
    """
    Collect gradient norms for residual blocks found in ``model``.

    This utility is intended for debugging gradient flow through residual stacks.
    It walks ``model.named_modules()`` and computes the global L2 norm of the
    gradients within each residual block module.

    Args:
        model: Model to inspect. Gradients must already be populated (i.e. after
            ``loss.backward()``).

    Returns:
        Dict mapping metric name to gradient norm. Keys use these prefixes:
        - ``residual/<module_path>``: Per residual block norm.
        - ``residual/total``: L2 norm over all residual block parameters.
        - ``non_residual/<group>``: Per non-residual group norm, where ``group``
          is derived from parameter names (e.g. ``input_proj``, ``bn1``,
          ``residual_transitions.0``).
        - ``non_residual/total``: L2 norm over all non-residual parameters.

    """
    # Import here to avoid pulling model code unless gradient tracking is enabled.
    from ..model.model_blocks import (  # noqa: PLC0415
        KeelResidualBlock,
        PostLNAlphaBetaResidualBlock,
        PostLNAlphaResidualBlock,
        ResidualBlock,
    )

    residual_types = (
        ResidualBlock,
        KeelResidualBlock,
        PostLNAlphaResidualBlock,
        PostLNAlphaBetaResidualBlock,
    )

    per_block_sq: dict[str, float] = {}
    residual_param_ids: set[int] = set()
    for name, module in model.named_modules():
        if not name:
            continue
        if not isinstance(module, residual_types):
            continue

        # Record params so we can compute a complementary non-residual norm later.
        residual_param_ids.update(id(p) for p in module.parameters())

        block_norm = _global_grad_norm(module.parameters())
        per_block_sq[name] = float(block_norm) ** 2

    out: dict[str, float] = {}
    for name, sq in per_block_sq.items():
        out[f"residual/{name}"] = math.sqrt(sq)
    if per_block_sq:
        out["residual/total"] = math.sqrt(sum(per_block_sq.values()))

    def _group_from_param_name(param_name: str) -> str:
        parts = param_name.split(".")
        if not parts:
            return param_name
        if len(parts) >= 2 and parts[1].isdigit():
            return ".".join(parts[:2])
        return parts[0]

    non_residual_sq_by_group: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None or id(p) in residual_param_ids:
            continue
        gn = float(p.grad.detach().norm(2).item())
        group = _group_from_param_name(name)
        non_residual_sq_by_group[group] = (
            non_residual_sq_by_group.get(group, 0.0) + gn * gn
        )

    for group, sq in sorted(non_residual_sq_by_group.items()):
        out[f"non_residual/{group}"] = math.sqrt(sq)
    out["non_residual/total"] = math.sqrt(sum(non_residual_sq_by_group.values()))
    return out


def _make_grad_stats_fn_from_param_regex(
    patterns: Mapping[str, str],
) -> Callable[[nn.Module], dict[str, float]]:
    compiled = {k: re.compile(v) for k, v in patterns.items()}

    def _fn(model: nn.Module) -> dict[str, float]:
        total_sq: dict[str, float] = dict.fromkeys(compiled, 0.0)
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            gn = float(g.norm(2).item())
            for label, pat in compiled.items():
                if pat.search(name) is not None:
                    total_sq[label] += gn * gn
        return {k: math.sqrt(v) for k, v in total_sq.items()}

    return _fn


def plot_train_val_loss(
    history: Mapping[str, Any],
    *,
    title: str | None = None,
    y_scale: Literal["linear", "log", "adaptive"] = "linear",
    adaptive_log_threshold: float = 10.0,
):
    """
    Plot train/val loss and metrics as two separate figures.

    Accepts either explicit `train_loss`, `val_loss`, `eval_metric` or a history
    dict returned by `train_model_one_n` with keys:
    - train_loss
    - val_loss
    - center_eval
    - lip_eval
    - grad_norm (optional; only if tracked)
    - grad_stats (optional; only if tracked)
    """
    train_loss = cast(Sequence[float], history.get("train_loss", []) or [])
    val_loss = cast(Sequence[float], history.get("val_loss", []) or [])
    center_val = cast(Sequence[float] | None, history.get("center_eval"))
    lip_eval = cast(Sequence[float] | None, history.get("lip_eval"))
    grad_norm = cast(Sequence[float] | None, history.get("grad_norm"))
    grad_stats = cast(
        Sequence[Mapping[str, float] | None] | None, history.get("grad_stats")
    )

    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(list(train_loss), label="train")
    ax_loss.plot(list(val_loss), label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    metrics: list[tuple[str, Sequence[float]]] = []
    if center_val is not None:
        metrics.append(("Center eval", center_val))
    if lip_eval is not None:
        metrics.append(("1-Lipschitz eval", lip_eval))

    fig_metrics, ax_metrics = plt.subplots(figsize=(8, 4))
    colors = ["gray", "magenta", "tab:blue", "tab:orange"]
    plot_metrics = [
        (label, series)
        for label, series in metrics
        if any(v is not None for v in series)
    ]
    metric_values_after_50: list[float] = []
    metric_values_all: list[float] = []
    for idx, (label, series) in enumerate(plot_metrics):
        y = [float(v) if v is not None else float("nan") for v in series]
        metric_values_all.extend(y)
        if len(y) > 50:
            metric_values_after_50.extend(y[50:])
        ax_metrics.plot(
            y,
            label=label,
            color=colors[idx % len(colors)],
            linestyle="--",
        )
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_ylabel("Metric")

    if (y_scale == "log") or (
        y_scale == "adaptive"
        and _should_use_log_scale(
            train_loss, val_loss, threshold=adaptive_log_threshold
        )
    ):
        ax_loss.set_yscale("log")

    ax_metrics.set_yscale("linear")

    loss_title = f"{title} loss" if title else "loss"
    metrics_title = f"{title} metrics" if title else "metrics"

    ax_loss.set_title(loss_title)
    ax_loss.grid(True, which="both")
    ax_loss.legend(loc="upper center")
    ax_loss.set_xlim(0, len(train_loss))
    fig_loss.tight_layout()

    ax_metrics.set_title(metrics_title)
    ax_metrics.grid(True, which="both")
    if plot_metrics:
        ax_metrics.legend(loc="upper right")
    ax_metrics.set_xlim(0, len(train_loss))

    def _finite_ylim(values: Sequence[float]) -> tuple[float, float] | None:
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return None
        ymin = float(min(finite))
        ymax = float(max(finite))
        if ymin == ymax:
            pad = 0.05 * (abs(ymin) if ymin != 0.0 else 1.0)
            ymin -= pad
            ymax += pad
        return ymin, ymax

    metric_ylim = _finite_ylim(metric_values_after_50) or _finite_ylim(
        metric_values_all
    )
    if metric_ylim is not None:
        ax_metrics.set_ylim(*metric_ylim)
    fig_metrics.tight_layout()

    has_grad_stats = bool(grad_stats) and any(
        isinstance(d, Mapping) and bool(d) for d in (grad_stats or [])
    )

    if has_grad_stats:
        fig_grad_norm, (ax_grad_norm, ax_grad_res, ax_grad_nonres) = plt.subplots(
            nrows=3, figsize=(8, 10), sharex=True
        )
    else:
        fig_grad_norm, ax_grad_norm = plt.subplots(figsize=(8, 4))
        ax_grad_res = None
        ax_grad_nonres = None

    global_series = list(grad_norm or [])
    if not global_series and has_grad_stats and grad_stats is not None:
        global_series = []
        for d in grad_stats:
            if not d:
                global_series.append(float("nan"))
                continue
            res_total = float(d.get("residual/total", float("nan")))
            non_total = float(d.get("non_residual/total", float("nan")))
            if math.isnan(res_total) and math.isnan(non_total):
                global_series.append(float("nan"))
                continue
            res_total = 0.0 if math.isnan(res_total) else res_total
            non_total = 0.0 if math.isnan(non_total) else non_total
            global_series.append(
                math.sqrt(res_total * res_total + non_total * non_total)
            )

    ax_grad_norm.plot(global_series, label="global")
    ax_grad_norm.set_ylabel("Grad norm")
    ax_grad_norm.set_title("Gradient norms")
    ax_grad_norm.grid(True, which="both")
    ax_grad_norm.legend(loc="upper right")
    ax_grad_norm.set_xlim(0, len(train_loss))
    ax_grad_norm.set_yscale("log")
    if not has_grad_stats:
        ax_grad_norm.set_xlabel("Epoch")

    if (
        has_grad_stats
        and grad_stats is not None
        and ax_grad_res is not None
        and ax_grad_nonres is not None
    ):

        def _series_for_key(key: str) -> list[float]:
            return [
                float(d.get(key, float("nan"))) if d else float("nan")
                for d in grad_stats
            ]

        residual_total_key = "residual/total"
        non_total_key = "non_residual/total"
        all_keys = sorted({k for d in grad_stats if d for k in d})

        residual_keys = [
            k for k in all_keys if k.startswith("residual/") and k != residual_total_key
        ]
        non_keys = [
            k for k in all_keys if k.startswith("non_residual/") and k != non_total_key
        ]
        other_keys = [
            k for k in all_keys if not (k.startswith(("residual/", "non_residual/")))
        ]

        has_res_total = any(d and residual_total_key in d for d in grad_stats)
        has_non_total = any(d and non_total_key in d for d in grad_stats)

        if not residual_keys and not non_keys and other_keys:
            for k in other_keys:
                ax_grad_res.plot(_series_for_key(k), label=k, alpha=0.8)
            ax_grad_res.set_title("Grad stats")
            ax_grad_res.set_ylabel("Grad norm")
            ax_grad_res.grid(True, which="both")
            ax_grad_res.set_yscale("log")
            ax_grad_res.legend(loc="upper right", fontsize="small")

            ax_grad_nonres.set_visible(False)
            fig_grad_norm.tight_layout()
            return (
                fig_loss,
                ax_loss,
                fig_metrics,
                ax_metrics,
                fig_grad_norm,
                ax_grad_norm,
            )

        if has_res_total:
            ax_grad_res.plot(
                _series_for_key(residual_total_key),
                label="total",
                color="black",
                linewidth=2.0,
            )
        for k in residual_keys:
            ax_grad_res.plot(
                _series_for_key(k),
                label=k.removeprefix("residual/"),
                alpha=0.8,
            )
        ax_grad_res.set_title("Residual blocks")
        ax_grad_res.set_ylabel("Grad norm")
        ax_grad_res.grid(True, which="both")
        ax_grad_res.set_yscale("log")
        if len(residual_keys) + int(has_res_total) <= 12:
            ax_grad_res.legend(loc="upper right", fontsize="small")

        if has_non_total:
            ax_grad_nonres.plot(
                _series_for_key(non_total_key),
                label="total",
                color="black",
                linewidth=2.0,
            )
        for k in non_keys:
            ax_grad_nonres.plot(
                _series_for_key(k),
                label=k.removeprefix("non_residual/"),
                alpha=0.8,
            )
        ax_grad_nonres.set_title("Non-residual blocks")
        ax_grad_nonres.set_ylabel("Grad norm")
        ax_grad_nonres.set_xlabel("Epoch")
        ax_grad_nonres.grid(True, which="both")
        ax_grad_nonres.set_yscale("log")
        if len(non_keys) + int(has_non_total) <= 12:
            ax_grad_nonres.legend(loc="upper right", fontsize="small")

        fig_grad_norm.tight_layout()
        return fig_loss, ax_loss, fig_metrics, ax_metrics, fig_grad_norm, ax_grad_norm

    fig_grad_norm.tight_layout()
    return fig_loss, ax_loss, fig_metrics, ax_metrics, fig_grad_norm, ax_grad_norm


def _train_epoch_mse(
    model: nn.Module,
    *,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    batch_size: int,
    graph: CayleyGraph | None = None,
    lip_weight: float = 0.0,
    lip_max_states: int | None = None,
    lip_generator_indices: Sequence[int] | None = None,
    lip_max_generators: int | None = None,
    lip_seed: int | None = None,
    lip_state_batch_size: int | None = None,
    lip_reduction: Literal["mean", "sum"] = "mean",
    dtype: torch.dtype = torch.float32,
    track_grad_norm: bool = False,
    grad_stats_fn: Callable[[nn.Module], Mapping[str, float]] | None = None,
    profile: bool = False,
    profile_sync_cuda: bool = True,
) -> tuple[float, float, float | None, float, dict[str, float] | None]:
    """Train the model for one epoch on the training set."""
    model.train()
    tr_loss = 0.0
    lip_loss_sum = 0.0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    grad_stats_sum: dict[str, float] = {}
    grad_stats_count = 0
    lip_time_s = 0.0

    lip_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = None
    if (
        profile
        and profile_sync_cuda
        and x_tr.is_cuda
        and lip_weight > 0.0
        and graph is not None
    ):
        lip_events = []

    device_type = _infer_module_device_type(model)
    enable_autocast = _should_enable_autocast(dtype, device_type=device_type)

    for s in range(0, x_tr.shape[0], batch_size):
        xb = x_tr[s : s + batch_size]
        yb = y_tr[s : s + batch_size].float().squeeze().to(dtype)
        with (
            torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=enable_autocast,
            )
            if enable_autocast
            else nullcontext()
        ):
            pred: torch.Tensor = model(xb).squeeze()
            mse_loss: torch.Tensor = loss_fn(pred.float(), yb.float())
            loss: torch.Tensor = mse_loss
            if lip_weight > 0.0 and graph is not None:
                if lip_events is not None:
                    lip_evt0 = torch.cuda.Event(enable_timing=True)
                    lip_evt1 = torch.cuda.Event(enable_timing=True)
                    lip_evt0.record()
                lip_loss = lipschitz_expansion_loss(
                    model,
                    graph,
                    xb,
                    max_states=lip_max_states,
                    generator_indices=lip_generator_indices,
                    max_generators=lip_max_generators,
                    seed=lip_seed,
                    state_batch_size=lip_state_batch_size,
                    reduction=lip_reduction,
                ).float()
                if lip_events is not None:
                    lip_evt1.record()
                    lip_events.append((lip_evt0, lip_evt1))
                loss = loss + lip_weight * lip_loss
                lip_loss_sum += float(lip_loss.item()) * xb.size(0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if track_grad_norm:
            grad_norm = _global_grad_norm(model.parameters())
            grad_norm_sum += grad_norm * xb.size(0)
            grad_norm_count += xb.size(0)
        if grad_stats_fn is not None:
            stats = grad_stats_fn(model)
            for key, value in stats.items():
                grad_stats_sum[key] = grad_stats_sum.get(key, 0.0) + float(
                    value
                ) * xb.size(0)
            grad_stats_count += xb.size(0)
        opt.step()
        tr_loss += float(loss.item()) * xb.size(0)
    tr_loss_avg = tr_loss / x_tr.shape[0]
    lip_loss_avg = lip_loss_sum / x_tr.shape[0]
    grad_norm_avg = (
        grad_norm_sum / grad_norm_count if track_grad_norm and grad_norm_count else None
    )
    grad_stats_avg = (
        {k: v / grad_stats_count for k, v in grad_stats_sum.items()}
        if grad_stats_count
        else None
    )
    if lip_events:
        # Ensure events have completed so elapsed_time is valid.
        torch.cuda.synchronize()
        lip_time_ms = sum(evt0.elapsed_time(evt1) for evt0, evt1 in lip_events)
        lip_time_s = float(lip_time_ms) / 1000.0
    return tr_loss_avg, lip_loss_avg, grad_norm_avg, lip_time_s, grad_stats_avg


@torch.no_grad()
def _eval_epoch_mse(
    model: nn.Module,
    *,
    loss_fn: nn.Module,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype = torch.float32,
) -> float:
    """
    Evaluate model on validation set.

    Returns:
        Average loss over the validation set.

    """
    model.eval()
    device_type = _infer_module_device_type(model)
    enable_autocast = _should_enable_autocast(dtype, device_type=device_type)
    va_loss = 0.0
    for s in range(0, x_va.shape[0], batch_size):
        xb = x_va[s : s + batch_size]
        yb = y_va[s : s + batch_size].float().squeeze().to(dtype)
        with (
            torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=enable_autocast,
            )
            if enable_autocast
            else nullcontext()
        ):
            pred: torch.Tensor = model(xb).squeeze()
            loss: torch.Tensor = loss_fn(pred.float(), yb.float())
        va_loss += float(loss.item()) * xb.size(0)
    return va_loss / x_va.shape[0]


@torch.no_grad()
def _eval_lip_metric(
    model: nn.Module,
    *,
    graph: CayleyGraph,
    x_va: torch.Tensor,
    batch_size: int,
    lip_max_states: int | None = None,
    lip_generator_indices: Sequence[int] | None = None,
    lip_max_generators: int | None = None,
    lip_seed: int | None = None,
    lip_state_batch_size: int | None = None,
    lip_reduction: Literal["mean", "sum"] = "mean",
    dtype: torch.dtype = torch.float32,
) -> float:
    """Compute the Lipschitz penalty metric on the validation set."""
    model.eval()
    if x_va.shape[0] == 0:
        return 0.0
    device_type = _infer_module_device_type(model)
    enable_autocast = _should_enable_autocast(dtype, device_type=device_type)

    total = 0.0
    total_states = 0
    for s in range(0, x_va.shape[0], batch_size):
        xb = x_va[s : s + batch_size]
        xb = xb.to(dtype)
        with (
            torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=enable_autocast,
            )
            if enable_autocast
            else nullcontext()
        ):
            lip_loss = lipschitz_expansion_loss(
                model,
                graph,
                xb,
                max_states=lip_max_states,
                generator_indices=lip_generator_indices,
                max_generators=lip_max_generators,
                seed=lip_seed,
                state_batch_size=lip_state_batch_size,
                reduction=lip_reduction,
            ).float()
        total += float(lip_loss.item()) * xb.size(0)
        total_states += xb.size(0)
    return total / total_states


@torch.no_grad()
def _eval_center(model: nn.Module, graph: CayleyGraph, dtype: torch.dtype) -> float:
    """Evaluate the model on the central state of the graph."""
    device_type = _infer_module_device_type(model)
    enable_autocast = _should_enable_autocast(dtype, device_type=device_type)
    z = torch.as_tensor(graph.central_state, device=graph.device)
    # Model expects a batch dimension: [B, state_size]
    if z.ndim == 0:
        z = z.view(1, 1)
    elif z.ndim == 1:
        z = z.unsqueeze(0)
    z = z.long().to(dtype)
    with (
        torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=enable_autocast,
        )
        if enable_autocast
        else nullcontext()
    ):
        out = model(z)
    return float(out.reshape(-1)[0].item())


class _ModelCheckpoints:
    """Manage model checkpointing."""

    def __init__(self, models_path: str | Path | None = None):
        out_dir = Path(models_path) if models_path is not None else Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.best_center_path = out_dir / "best_center.pt"
        self.best_val_path = out_dir / "best_val.pt"
        self.final_path = out_dir / "final.pt"
        self.best_lip_path = out_dir / "best_lip.pt"

        self.best_val = float("inf")
        self.best_center_abs = float("inf")
        self.best_lip = float("inf")

    @staticmethod
    def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        """Return a dictionary of the model's state_dict, cloned and detached on CPU."""
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def update(
        self, model: nn.Module, *, val_loss: float, center_eval: float, lip_eval: float
    ) -> None:
        """Update the best validation loss and center evaluation and save best models."""
        if val_loss < self.best_val:
            self.best_val = float(val_loss)
            torch.save(self._cpu_state_dict(model), self.best_val_path)

        ce_abs = abs(float(center_eval))
        if ce_abs < self.best_center_abs:
            self.best_center_abs = ce_abs
            torch.save(self._cpu_state_dict(model), self.best_center_path)

        if lip_eval is not None and lip_eval < self.best_lip:
            self.best_lip = float(lip_eval)
            torch.save(self._cpu_state_dict(model), self.best_lip_path)

    def save(self, model: nn.Module) -> None:
        """Save the final model weights."""
        torch.save(self._cpu_state_dict(model), self.final_path)


def train_model_one_n(
    cfg,
    model: nn.Module,
    graph: CayleyGraph,
    max_samples_cap: int = 200_000,
    print_every: int = 5,
    *,
    return_history: bool = False,
    track_grad_norm: bool = False,
    grad_stats: (
        str | Mapping[str, str] | Callable[[nn.Module], Mapping[str, float]] | None
    ) = None,
    lr_scheduler_ctor: Callable[[torch.optim.Optimizer], Any] | None = None,
    models_path: str | Path | None = None,
    rw_refresh_interval: int = 1,
    rw_lengths: list[tuple[float, int]] | None = None,
    epoch_end_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Train `model` on a single Cayley graph instance; optionally return loss histories.

    If `lr_scheduler_ctor` is provided, it will be called as `lr_scheduler_ctor(opt)`
    after the optimizer is created. Scheduler stepping is done once per epoch:
    - `ReduceLROnPlateau`: stepped with validation loss
    - others: stepped without arguments

    Three checkpoints are written to `models_path`:
    - `best_val.pt`: best validation loss
    - `best_center.pt`: best (closest to zero) center prediction
    - `final.pt`: final epoch weights

    `model_path` should be a full path (can include a filename like `foo.pt`, used as prefix).
    If `models_path` is provided, it overrides the output directory; otherwise `model_path.parent`.

    If `return_history` is True, returns a dict with keys:
    - train_loss
    - val_loss
    - center_eval
    - lip_eval
    - grad_norm (only if `track_grad_norm` is True)
    - grad_stats (only if `grad_stats` is set)

    Args:
    - cfg: configuration dictionary
    - model: model to train
    - graph: Cayley graph to train on
    - max_samples_cap: maximum number of samples to use for training
    - print_every: print every print_every epochs
    - return_history: return history of training
    - track_grad_norm: track gradient norm
    - grad_stats: gradient statistics
    - lr_scheduler_ctor: learning rate scheduler constructor
    - models_path: path to save models
    - rw_refresh_interval: refresh interval for random walks
    - rw_lengths: list of (factor, length) tuples for random walks, where factor*max_samples_cap
      is the number of samples to use for training. If None, use the default lengths:
      (1, base_len), (0.5, base_len // 2), (0.25, base_len // 4).
    - epoch_end_callback: optional callable invoked once per epoch with scalar metrics

    Returns:
      - history: dictionary of training history
        - train_loss: list of training loss values
        - val_loss: list of validation loss values
        - center_eval: list of center evaluation values
        - lip_eval: list of Lipschitz evaluation values
        - grad_norm: list of gradient norm values (only if `track_grad_norm` is True)
        - grad_stats: list of gradient statistics (only if `grad_stats` is set)

    Raises:
    - ValueError: if `rw_refresh_interval` is less than 1
    - TypeError: if `grad_stats` is not a string, mapping of {name: regex}, callable, or None
    """
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    if lr_scheduler_ctor is None:
        lr_scheduler_ctor = lr_scheduler_ctor_from_cfg(cfg)
    scheduler = lr_scheduler_ctor(opt) if lr_scheduler_ctor is not None else None
    device = graph.device
    model.to(device)
    dtype = cfg.get("model_dtype", torch.float32)

    epochs = cfg["num_epochs"]

    tr_history: list[float] = []
    va_history: list[float] = []
    center_eval_history: list[float] = []
    lip_eval_history: list[float | None] = []
    grad_norm_history: list[float] = []
    grad_stats_history: list[dict[str, float] | None] = []
    ckpt = _ModelCheckpoints(models_path=models_path)

    lip_weight = float(cfg.get("lip_weight", 0.0))
    lip_max_states = cfg.get("lip_max_states", None)
    lip_generator_indices = cfg.get("lip_generator_indices", None)
    lip_max_generators = cfg.get("lip_max_generators", None)
    lip_seed = cfg.get("lip_seed", None)
    lip_state_batch_size = cfg.get("lip_state_batch_size", None)
    lip_reduction = cfg.get("lip_reduction", "mean")
    lip_eval_metric = bool(cfg.get("lip_val_metric", False))

    profile = bool(cfg.get("profile", False))
    profile_sync_cuda = bool(cfg.get("profile_sync_cuda", True))
    is_cuda = device.type == "cuda"

    if grad_stats is None:
        grad_stats = cfg.get("grad_stats", None)
    grad_stats_fn: Callable[[nn.Module], Mapping[str, float]] | None = None
    if grad_stats is not None:
        if callable(grad_stats):
            grad_stats_fn = grad_stats
        elif isinstance(grad_stats, str):
            spec = grad_stats.strip().lower()
            if spec in {"residual", "residual_blocks", "resblocks"}:
                grad_stats_fn = grad_stats_residual_blocks
            else:
                raise ValueError(
                    f"Unknown grad_stats spec {grad_stats!r}. Use 'residual_blocks', "
                    "a mapping of {name: regex}, or a callable."
                )
        elif isinstance(grad_stats, Mapping):
            grad_stats_fn = _make_grad_stats_fn_from_param_regex(grad_stats)
        else:
            raise TypeError(
                "grad_stats must be a string, mapping of {name: regex}, callable, or None."
            )

    def _sync_cuda() -> None:
        if profile and profile_sync_cuda and is_cuda:
            torch.cuda.synchronize()

    if rw_refresh_interval < 1:
        raise ValueError("cfg['rw_refresh_interval'] must be >= 1")

    X = y = None

    epoch_pbar = tqdm(range(epochs), desc="Epoch", unit="epoch")

    log_write = epoch_pbar.write

    for ep in epoch_pbar:
        t_epoch_wall0 = time.perf_counter()
        t_epoch0 = t_rw0 = t_sub0 = t_split0 = t_train0 = t_val0 = 0.0
        t_rw = t_sub = t_split = t_train = t_val = t_epoch = 0.0
        t_lip_tr = 0.0
        t_lip_val = 0.0
        if profile:
            _sync_cuda()
            t_epoch0 = time.perf_counter()
        refresh_walks = (X is None) or (ep % rw_refresh_interval == 0)
        if refresh_walks:
            if profile:
                t_rw0 = time.perf_counter()
            # Generate walks of three different lengths with linearly decreasing widths
            base_len = cfg["rw_length"]
            base_width = cfg["rw_width"]
            walks_list = []
            y_list = []

            if rw_lengths is None:
                rw_lengths = [
                    (1, base_len),
                    (0.5, base_len // 2),
                    (0.25, base_len // 4),
                ]

            for factor, length in rw_lengths:
                if length >= 10:
                    width = max(1, int(base_width * factor))
                    X_part, y_part = graph.random_walks(
                        width=width,
                        length=length,
                        mode=cfg["rw_mode"],
                        nbt_history_depth=length,
                    )
                    walks_list.append(X_part)
                    y_list.append(y_part)
            if len(walks_list) == 0:
                X_part, y_part = graph.random_walks(
                    width=base_width,
                    length=base_len,
                    mode=cfg["rw_mode"],
                    nbt_history_depth=base_width,
                )
                walks_list.append(X_part)
                y_list.append(y_part)

            X = torch.cat(walks_list, dim=0)
            y = torch.cat(y_list, dim=0)

            if profile:
                _sync_cuda()
                t_rw = time.perf_counter() - t_rw0
                t_sub0 = time.perf_counter()

            X, y = subsample_xy(X, y, max_samples_cap, seed=123 + ep)
            if profile:
                _sync_cuda()
                t_sub = time.perf_counter() - t_sub0

        if profile:
            t_split0 = time.perf_counter()

        assert X is not None
        assert y is not None
        if ep == 0:
            ymin, ymax, ystd, uniq = y_stats(y)
            log_write(
                f"    DATA y: min={ymin} max={ymax} std={ystd} uniq(sample)={uniq}  X={tuple(X.shape)}"
            )

        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X,
                y,
                train_size=1 - cfg["val_ratio"],
                stratify=y.detach().cpu().numpy(),
                shuffle=True,
                random_state=123 + ep,
            )
        except Exception:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X,
                y,
                train_size=1 - cfg["val_ratio"],
                shuffle=True,
                random_state=123 + ep,
            )
        if profile:
            _sync_cuda()
            t_split = time.perf_counter() - t_split0
            t_train0 = time.perf_counter()

        bs = cfg["batch_size"]
        model.train()
        tr_loss, tr_lip, tr_grad_norm, t_lip_tr, tr_grad_stats = _train_epoch_mse(
            model,
            loss_fn=loss_fn,
            opt=opt,
            x_tr=torch.as_tensor(X_tr),
            y_tr=torch.as_tensor(y_tr),
            batch_size=bs,
            graph=graph if lip_weight > 0.0 else None,
            lip_weight=lip_weight,
            lip_max_states=lip_max_states,
            lip_generator_indices=lip_generator_indices,
            lip_max_generators=lip_max_generators,
            lip_seed=lip_seed,
            lip_state_batch_size=lip_state_batch_size,
            lip_reduction=lip_reduction,
            dtype=dtype,
            track_grad_norm=track_grad_norm,
            grad_stats_fn=grad_stats_fn,
            profile=profile,
            profile_sync_cuda=profile_sync_cuda,
        )
        if profile:
            _sync_cuda()
            t_train = time.perf_counter() - t_train0
            t_val0 = time.perf_counter()
        model.eval()
        va_loss = _eval_epoch_mse(
            model,
            loss_fn=loss_fn,
            x_va=torch.as_tensor(X_va),
            y_va=torch.as_tensor(y_va),
            batch_size=bs,
            dtype=dtype,
        )
        if profile:
            _sync_cuda()
            t_val = time.perf_counter() - t_val0
        if lip_eval_metric:
            if profile:
                t_lipval0 = time.perf_counter()
            va_lip = _eval_lip_metric(
                model,
                graph=graph,
                x_va=torch.as_tensor(X_va),
                batch_size=bs,
                lip_max_states=lip_max_states,
                lip_generator_indices=lip_generator_indices,
                lip_max_generators=lip_max_generators,
                lip_seed=lip_seed,
                lip_state_batch_size=lip_state_batch_size,
                lip_reduction=lip_reduction,
                dtype=dtype,
            )
            if profile:
                _sync_cuda()
                t_lip_val = time.perf_counter() - t_lipval0
        else:
            va_lip = None

        center_eval = _eval_center(model, graph, dtype=dtype)

        tr_history.append(float(tr_loss))
        va_history.append(float(va_loss))
        center_eval_history.append(float(center_eval))
        lip_eval_history.append(float(va_lip) if va_lip is not None else None)
        if track_grad_norm and tr_grad_norm is not None:
            grad_norm_history.append(float(tr_grad_norm))
        if grad_stats_fn is not None:
            grad_stats_history.append(dict(tr_grad_stats) if tr_grad_stats else None)
        if ckpt is not None:
            ckpt.update(
                model,
                val_loss=float(va_loss),
                center_eval=float(center_eval),
                lip_eval=float(va_lip) if va_lip is not None else 0.0,
            )
        epoch_lr = opt.param_groups[0].get("lr", None)
        step_lr_scheduler(scheduler, va_loss)
        if profile:
            _sync_cuda()
            t_epoch = time.perf_counter() - t_epoch0

        if is_cuda:
            torch.cuda.synchronize()
        t_epoch_wall = time.perf_counter() - t_epoch_wall0
        if epoch_end_callback is not None:
            epoch_payload: dict[str, Any] = {
                "epoch": int(ep),
                "train_loss": float(tr_loss),
                "val_loss": float(va_loss),
                "center_eval": float(center_eval),
                "lip_train": float(tr_lip),
                "lip_eval": float(va_lip) if va_lip is not None else None,
                "grad_norm": float(tr_grad_norm) if tr_grad_norm is not None else None,
                "epoch_time_s": float(t_epoch_wall),
                "learning_rate": float(epoch_lr)
                if isinstance(epoch_lr, (float, int))
                else None,
            }
            if grad_stats_fn is not None:
                epoch_payload["grad_stats"] = (
                    dict(tr_grad_stats) if tr_grad_stats else None
                )
            epoch_end_callback(epoch_payload)
        postfix: dict[str, str] = {
            "train": f"{float(tr_loss):.5f}",
            "val": f"{float(va_loss):.5f}",
            "epoch_s": f"{t_epoch_wall:.2f}",
        }
        lr = opt.param_groups[0].get("lr", None)
        if isinstance(lr, (float, int)):
            postfix["lr"] = f"{lr:.2e}"
        epoch_pbar.set_postfix(postfix, refresh=True)

        if ep % print_every == 0 or ep == epochs - 1:
            lr_s = _format_lr(opt)
            lip_s = f" | Lip {tr_lip:.5f}" if lip_weight > 0.0 else ""
            lip_eval_s = f" | LipVal {va_lip:.5f}" if va_lip is not None else ""
            grad_s = (
                f" | Grad {tr_grad_norm:.3e}"
                if track_grad_norm and tr_grad_norm is not None
                else ""
            )
            timing_s = (
                f" | t rw {t_rw:.3f}s sub {t_sub:.3f}s split {t_split:.3f}s"
                + f" tr {t_train:.3f}s lip_tr {t_lip_tr:.3f}s"
                + f" va {t_val:.3f}s lip_va {t_lip_val:.3f}s epoch {t_epoch:.3f}s"
                if profile
                else ""
            )
            # log_write(
            #     f"    Epoch {ep:3d}/{epochs} | Train {tr_loss:.5f} | Val {va_loss:.5f} | "
            #     f"Center eval {center_eval:.3f} | {lip_s}{lip_eval_s} | {grad_s} | {lr_s}{timing_s}"
            # )

    epoch_pbar.close()

    if ckpt is not None:
        ckpt.save(model)

    if return_history:
        return {
            "train_loss": tr_history,
            "val_loss": va_history,
            "center_eval": center_eval_history,
            "lip_eval": lip_eval_history,
            "grad_norm": grad_norm_history if track_grad_norm else [],
            "grad_stats": grad_stats_history if grad_stats_fn is not None else [],
        }
    else:
        return {}


def try_beam(
    cfg: dict[str, Any],
    graph: "CayleyGraph",
    model: nn.Module,
    start_state: Sequence[int],
    *,
    enable_tf32: bool | None = None,
    enable_autocast: bool | None = None,
    autocast_dtype: torch.dtype = torch.float16,
    autocast_device_type: str | None = None,
    verbose: int = 0,
) -> tuple[int | None, int | None, dict[str, int | None]]:
    """
    Try multiple beam widths; return best distance, beam width, and per-width history.

    Args:
        cfg: configuration dictionary. Supports optional ``beam_mode`` /
            ``beam_method`` entries.
        graph: Cayley graph to infer on
        model: model to infer with
        start_state: starting state for beam search
        enable_tf32: whether to enable TF32
        enable_autocast: whether to enable autocast
        autocast_dtype: dtype for autocast
        autocast_device_type: device type for autocast

    Returns:
        best: best distance found,
        best_bw: minimal beam width that found best distance
        history_bw:


    Raises:
        ValueError: If ``cfg["beam_mode"]`` / ``cfg["beam_method"]`` is not
            supported by ``graph.beam_search``.

    Notes:
        - TF32 and autocast are controlled independently.
        - enable_tf32:
            None  -> enable on CUDA, disable otherwise
            True  -> force enable (CUDA only)
            False -> force disable
        - enable_autocast:
            None  -> enable on CUDA, disable otherwise
            True/False -> force
        - autocast_dtype controls autocast dtype (e.g. torch.float16 / torch.bfloat16).
    """
    model.eval()

    graph_device = graph.device
    if not isinstance(graph_device, torch.device):
        graph_device = torch.device(graph_device)
    is_cuda = graph_device.type == "cuda"
    use_tf32 = is_cuda if enable_tf32 is None else bool(enable_tf32)
    if autocast_device_type is None:
        autocast_device_type = graph_device.type
    default_autocast = _should_enable_autocast(
        autocast_dtype, device_type=autocast_device_type
    )
    use_autocast = (
        default_autocast if enable_autocast is None else bool(enable_autocast)
    )

    # Save global settings ti restore them after
    prev_matmul_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    prev_cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)

    try:
        if is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
            torch.backends.cudnn.allow_tf32 = use_tf32
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

        # Autocast context is independent from TF32.
        amp_ctx = (
            torch.autocast(
                device_type=autocast_device_type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            )
            if (is_cuda or autocast_device_type == "cpu")
            else nullcontext()
        )

        predictor = Predictor(graph, model)
        best: int | None = None
        best_bw: int | None = None
        history_bw: dict[str, int | None] = {}
        history_depth = int(cfg["history_depth"])
        configured_beam_mode = str(
            cfg.get("beam_mode", cfg.get("beam_method", "iterated"))
        )
        valid_beam_modes = {"simple", "advanced", "iterated"}
        if configured_beam_mode not in valid_beam_modes:
            raise ValueError(
                f"Unsupported beam mode {configured_beam_mode!r}; "
                f"expected one of {sorted(valid_beam_modes)}."
            )
        if configured_beam_mode == "simple" and history_depth > 0:
            warnings.warn(
                "try_beam(...): history_depth > 0 is ignored when beam_mode='simple'.",
                stacklevel=2,
            )
        n_generators = int(graph.definition.n_generators)

        if verbose > 1:
            # --- small inference-speed benchmark ---
            small_inference_speed_benchmark(
                cfg=cfg, graph=graph, model=model, num_iters=100
            )

        # --- beam search ---
        #
        # - inference_mode: better than model.eval() for long running tasks
        # - autocast: to lower precision
        with torch.inference_mode(), amp_ctx:
            for raw_bw in cfg["list_beam_width"]:
                bw = int(raw_bw)
                beam_mode = configured_beam_mode
                if beam_mode == "iterated" and bw < n_generators:
                    warnings.warn(
                        f"Beam width {bw} is smaller than the number of "
                        f"generators ({n_generators}); switching beam_mode "
                        "from 'iterated' to 'simple' for this run. "
                        "history_depth is ignored in simple mode.",
                        stacklevel=2,
                    )
                    beam_mode = "simple"

                tic = time.perf_counter()
                graph.free_memory()

                res = graph.beam_search(
                    start_state=start_state,
                    beam_width=bw,
                    max_steps=cfg["beam_max_steps"],
                    predictor=predictor,
                    history_depth=history_depth,
                    beam_mode=beam_mode,
                    return_path=True,
                    verbose=verbose,
                )

                key = f"d for bw {bw}"
                history_bw.setdefault(key, None)

                if res.path_found:
                    d = int(res.path_length)
                    if best is None or d < best:
                        best = d
                        best_bw = bw
                    history_bw[key] = d
                else:
                    history_bw[key] = None
                    d = None

                toc = time.perf_counter()
                if verbose > 0:
                    print(
                        f"Beam width {bw} took {toc - tic:.2f} seconds, "
                        f"found path of length {d if res.path_found else None}"
                    )

        return best, best_bw, history_bw

    finally:
        # Restore
        if is_cuda and prev_matmul_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        if is_cuda and prev_cudnn_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
