"""AlGraphGPT timing benchmark for Lightning train loops and inference."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch import Callback, LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from pilgrim.pancake_competition import (
    BeamInferenceConfig,
    compute_stats_by_n,
    load_or_create_submission_rows,
    run_targeted_beam_inference,
)
from pilgrim.utils.pancake_utils import make_graph_for_n, pancake_sort_path, solve
from pilgrim.utils.reproducibility import set_seed

from .config import LipschitzConfig, OptimizationConfig, RandomWalkDataConfig
from .data import RandomWalkDataModule
from .model_factory import build_model
from .module import PilgrimLightningModule
from .yaml_config import load_yaml_config, parse_torch_dtype


def _resolve_device(device: str | torch.device | None) -> str | torch.device:
    """Resolve runtime device with ``auto`` fallback to CUDA when available."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _sync_if_cuda(device: str | torch.device | None) -> None:
    """Synchronize CUDA stream for accurate wall-clock timings."""
    if torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.synchronize()


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count parameters of a module.

    Args:
        model: Model to inspect.

    Returns:
        Tuple ``(total_params, trainable_params)``.

    """
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return total, trainable


def _summarize_seconds(values: list[float], prefix: str) -> dict[str, float]:
    """
    Convert a list of second-values into common millisecond summary stats.

    Args:
        values: Timing samples in seconds.
        prefix: Prefix for output field names.

    Returns:
        Mapping with mean/median/p90/min/max in milliseconds.

    """
    if not values:
        return {
            f"{prefix}_mean_ms": float("nan"),
            f"{prefix}_median_ms": float("nan"),
            f"{prefix}_p90_ms": float("nan"),
            f"{prefix}_min_ms": float("nan"),
            f"{prefix}_max_ms": float("nan"),
            f"{prefix}_num_samples": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        f"{prefix}_mean_ms": float(arr.mean()),
        f"{prefix}_median_ms": float(np.quantile(arr, 0.5)),
        f"{prefix}_p90_ms": float(np.quantile(arr, 0.9)),
        f"{prefix}_min_ms": float(arr.min()),
        f"{prefix}_max_ms": float(arr.max()),
        f"{prefix}_num_samples": float(arr.shape[0]),
    }


def _flatten_competition_rows(df: pd.DataFrame) -> list[Any]:
    """
    Convert a DataFrame into row tuples used by competition utilities.

    Args:
        df: DataFrame with at least ``id`` and ``permutation`` columns.

    Returns:
        List of named tuples produced by ``DataFrame.itertuples``.

    """
    return list(df.itertuples(index=False))


def _build_heuristic_paths(df: pd.DataFrame) -> list[str]:
    """
    Build heuristic pancake-sort paths for every row in a DataFrame.

    Args:
        df: Test subset with ``permutation`` column.

    Returns:
        Dotted path strings aligned with row order.

    """
    out: list[str] = []
    for row in df.itertuples(index=False):
        perm = [int(x) for x in str(row.permutation).split(",")]
        out.append(".".join(pancake_sort_path(perm)))
    return out


def _select_rows_by_n(
    test_df: pd.DataFrame,
    target_n: set[int],
    max_rows_per_n: int,
) -> pd.DataFrame:
    """
    Select a compact benchmark subset from Kaggle test data.

    Args:
        test_df: Full Kaggle test DataFrame.
        target_n: Sizes to include.
        max_rows_per_n: Number of rows to keep for each ``n``.

    Returns:
        Selected DataFrame preserving original row order within each ``n``.

    """
    work = test_df.copy()
    work["n"] = work["permutation"].astype(str).str.count(",") + 1
    parts: list[pd.DataFrame] = []
    for n in sorted(target_n):
        part = work[work["n"] == int(n)].head(int(max_rows_per_n))
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError(
            f"No rows found for target_n={sorted(target_n)} in provided test CSV."
        )
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.drop(columns=["n"])


@dataclass(slots=True)
class AlGraphGPTSizePreset:
    """
    One AlGraphGPT size preset used for timing.

    Args:
        name: Label used in tables and filenames.
        model_config: Base model kwargs for AlGraphGPT constructor.

    """

    name: str
    model_config: dict[str, Any]


@dataclass(slots=True)
class TimingRuntimeConfig:
    """
    Trainer runtime configuration for timing jobs.

    Args:
        accelerator: Lightning accelerator value.
        devices: Number/list of devices.
        precision: Trainer precision setting.
        deterministic: Deterministic mode toggle.
        enable_progress_bar: Progress bar toggle.
        log_every_n_steps: Metric logging frequency.

    """

    accelerator: str = "auto"
    devices: int | str | list[int] = 1
    precision: str | int = "32-true"
    deterministic: bool = False
    enable_progress_bar: bool = False
    log_every_n_steps: int = 1


@dataclass(slots=True)
class TrainTimingConfig:
    """
    Training-loop timing settings.

    Args:
        enabled: Whether to run train-loop timings.
        n: Group size used for train micro-benchmarks.
        limit_train_batches: Number of train batches per benchmark fit.
        limit_val_batches: Number of val batches per benchmark fit.
        data: Random-walk datamodule settings.
        optimization: Optimizer settings.
        lipschitz: Optional 1-Lipschitz regularization settings.
        graph_batch_size: Graph internal batch size.
        graph_dtype: Graph state dtype.
        device: Graph/device override.

    """

    enabled: bool = True
    n: int = 20
    limit_train_batches: int = 8
    limit_val_batches: int = 4
    data: RandomWalkDataConfig = field(default_factory=RandomWalkDataConfig)
    optimization: OptimizationConfig = field(
        default_factory=lambda: OptimizationConfig(
            lr=1e-3,
            weight_decay=2.5e-4,
            num_epochs=1,
            lr_scheduler=None,
        )
    )
    lipschitz: LipschitzConfig = field(default_factory=LipschitzConfig)
    graph_batch_size: int = 2**17
    graph_dtype: torch.dtype = torch.int8
    device: str | torch.device | None = None


@dataclass(slots=True)
class DirectForwardTimingConfig:
    """
    Direct forward-pass timing settings.

    Args:
        warmup_iters: Number of warmup iterations.
        measure_iters: Number of measured iterations.
        batch_size: Benchmark batch size.
        enable_autocast: Whether autocast is enabled.
        autocast_dtype: Autocast dtype.

    """

    warmup_iters: int = 10
    measure_iters: int = 50
    batch_size: int = 512
    enable_autocast: bool = True
    autocast_dtype: torch.dtype = torch.bfloat16


@dataclass(slots=True)
class InferenceTimingConfig:
    """
    Inference timing settings for direct forward and beam solve.

    Args:
        enabled: Whether inference timing is enabled.
        test_csv_path: Path to Kaggle test CSV.
        target_n: Sizes used for beam evaluation subset.
        max_rows_per_n: Maximum rows per ``n`` in benchmark subset.
        beam_width: Beam width.
        history_depth: Beam history depth.
        max_steps_factor: Beam max-steps multiplier.
        require_not_longer_than_previous: Acceptance gate for updates.
        compile_once_per_n: Compile each model once before repeated use.
        free_graph_memory: Whether to clear graph cache every solve.
        clear_cuda_cache: Whether to call ``torch.cuda.empty_cache`` each solve.
        run_gc_collect: Whether to run ``gc.collect`` each solve.
        enable_autocast: Autocast toggle for beam solve.
        autocast_dtype: Autocast dtype for beam solve.
        graph_batch_size: Graph batch size for inference graphs.
        graph_dtype: Graph dtype for inference graphs.
        device: Device override.
        direct_forward: Direct forward timing settings.

    """

    enabled: bool = True
    test_csv_path: Path = Path("kernels/data/test.csv")
    target_n: set[int] = field(default_factory=lambda: {20})
    max_rows_per_n: int = 5
    beam_width: int = 2**8
    history_depth: int = 0
    max_steps_factor: int = 2
    require_not_longer_than_previous: bool = True
    compile_once_per_n: bool = True
    free_graph_memory: bool = True
    clear_cuda_cache: bool = True
    run_gc_collect: bool = True
    enable_autocast: bool = True
    autocast_dtype: torch.dtype = torch.bfloat16
    graph_batch_size: int = 2**17
    graph_dtype: torch.dtype = torch.int8
    device: str | torch.device | None = None
    direct_forward: DirectForwardTimingConfig = field(
        default_factory=DirectForwardTimingConfig
    )


@dataclass(slots=True)
class ProfilerConfig:
    """
    Torch profiler settings for bottleneck analysis.

    Args:
        enabled: Whether to run profiler captures.
        top_k: Number of top operations retained in reports.
        record_shapes: Whether to capture tensor shapes.
        profile_memory: Whether to capture memory metrics.
        with_stack: Whether to capture Python stacks.

    """

    enabled: bool = True
    top_k: int = 20
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False


@dataclass(slots=True)
class AlGraphGPTTimingBenchmarkConfig:
    """
    Full benchmark configuration.

    Args:
        seed: Optional global random seed.
        output_root: Base directory for benchmark outputs.
        runtime: Shared Lightning runtime settings.
        train: Train-loop timing configuration.
        inference: Inference timing configuration.
        profiler: Profiler settings.
        model_sizes: AlGraphGPT size presets to compare.

    """

    seed: int | None = 42
    output_root: Path = Path("artifacts/pilgrim_lightning/algraphgpt_timing")
    runtime: TimingRuntimeConfig = field(default_factory=TimingRuntimeConfig)
    train: TrainTimingConfig = field(default_factory=TrainTimingConfig)
    inference: InferenceTimingConfig = field(default_factory=InferenceTimingConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    model_sizes: list[AlGraphGPTSizePreset] = field(default_factory=list)


@dataclass(slots=True)
class AlGraphGPTTimingBenchmarkResult:
    """
    Paths and dataframes produced by a benchmark run.

    Args:
        run_dir: Output directory of this run.
        train_timings: Training timing table.
        inference_timings: Inference timing table.
        beam_solve_times: Row-level beam solve timings.
        operation_profile: Semantic per-operation timing table collected from
            AlGraphGPT internal timers.
        summary_path: Markdown summary report path.

    """

    run_dir: Path
    train_timings: pd.DataFrame
    inference_timings: pd.DataFrame
    beam_solve_times: pd.DataFrame
    operation_profile: pd.DataFrame
    summary_path: Path


class BatchTimingCallback(Callback):
    """
    Collect per-batch train and validation durations during Lightning fit.

    Durations are measured with synchronized CUDA timing when applicable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_batch_seconds: list[float] = []
        self.val_batch_seconds: list[float] = []
        self._train_t0: float | None = None
        self._val_t0: float | None = None

    @staticmethod
    def _infer_device(pl_module: LightningModule) -> str | torch.device:
        try:
            return next(pl_module.parameters()).device
        except StopIteration:
            return "cpu"

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer
        del batch
        del batch_idx
        _sync_if_cuda(self._infer_device(pl_module))
        self._train_t0 = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer
        del outputs
        del batch
        del batch_idx
        if self._train_t0 is None:
            return
        _sync_if_cuda(self._infer_device(pl_module))
        self.train_batch_seconds.append(time.perf_counter() - self._train_t0)
        self._train_t0 = None

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer
        del batch
        del batch_idx
        del dataloader_idx
        _sync_if_cuda(self._infer_device(pl_module))
        self._val_t0 = time.perf_counter()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer
        del outputs
        del batch
        del batch_idx
        del dataloader_idx
        if self._val_t0 is None:
            return
        _sync_if_cuda(self._infer_device(pl_module))
        self.val_batch_seconds.append(time.perf_counter() - self._val_t0)
        self._val_t0 = None


class BeamTimingCollector:
    """Collect row-level beam solve times via Aim-like ``track`` interface."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def track(
        self,
        value: float,
        *,
        name: str,
        step: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Collect selected beam timing points.

        Args:
            value: Tracked scalar value.
            name: Metric name.
            step: Optional row index.
            context: Optional metric context.

        Returns:
            None.

        """
        if name != "beam/solve_time_s":
            return
        ctx = context or {}
        row = {
            "step": int(step) if step is not None else -1,
            "n": int(ctx.get("n", -1)),
            "beam_width": int(ctx.get("beam_width", -1)),
            "history_depth": int(ctx.get("history_depth", -1)),
            "max_steps": int(ctx.get("max_steps", -1)),
            "solve_time_s": float(value),
        }
        self.rows.append(row)


def _parse_runtime_config(raw: dict[str, Any]) -> TimingRuntimeConfig:
    """Build :class:`TimingRuntimeConfig` from raw mapping."""
    return TimingRuntimeConfig(
        accelerator=str(raw.get("accelerator", "auto")),
        devices=raw.get("devices", 1),
        precision=raw.get("precision", "32-true"),
        deterministic=bool(raw.get("deterministic")),
        enable_progress_bar=bool(raw.get("enable_progress_bar")),
        log_every_n_steps=int(raw.get("log_every_n_steps", 1)),
    )


def _parse_train_config(raw: dict[str, Any]) -> TrainTimingConfig:
    """Build :class:`TrainTimingConfig` from raw mapping."""
    data = RandomWalkDataConfig(**dict(raw.get("data", {})))
    opt = OptimizationConfig(
        lr=raw.get("optimization", {}).get("lr", 1e-3),
        weight_decay=raw.get("optimization", {}).get("weight_decay", 2.5e-4),
        num_epochs=raw.get("optimization", {}).get("num_epochs", 1),
        lr_scheduler=raw.get("optimization", {}).get("lr_scheduler"),
    )
    lip = LipschitzConfig(**dict(raw.get("lipschitz", {})))
    graph_cfg = dict(raw.get("graph", {}))
    return TrainTimingConfig(
        enabled=bool(raw.get("enabled", True)),
        n=int(raw.get("n", 20)),
        limit_train_batches=int(raw.get("limit_train_batches", 8)),
        limit_val_batches=int(raw.get("limit_val_batches", 4)),
        data=data,
        optimization=opt,
        lipschitz=lip,
        graph_batch_size=int(graph_cfg.get("batch_size", 2**17)),
        graph_dtype=parse_torch_dtype(graph_cfg.get("dtype", "int8")),
        device=graph_cfg.get("device", raw.get("device")),
    )


def _parse_direct_forward_config(raw: dict[str, Any]) -> DirectForwardTimingConfig:
    """Build :class:`DirectForwardTimingConfig` from raw mapping."""
    return DirectForwardTimingConfig(
        warmup_iters=int(raw.get("warmup_iters", 10)),
        measure_iters=int(raw.get("measure_iters", 50)),
        batch_size=int(raw.get("batch_size", 512)),
        enable_autocast=bool(raw.get("enable_autocast", True)),
        autocast_dtype=parse_torch_dtype(raw.get("autocast_dtype", "bfloat16")),
    )


def _parse_inference_config(raw: dict[str, Any]) -> InferenceTimingConfig:
    """Build :class:`InferenceTimingConfig` from raw mapping."""
    graph_cfg = dict(raw.get("graph", {}))
    direct_forward = _parse_direct_forward_config(dict(raw.get("direct_forward", {})))
    return InferenceTimingConfig(
        enabled=bool(raw.get("enabled", True)),
        test_csv_path=Path(raw.get("test_csv_path", "kernels/data/test.csv")),
        target_n={int(x) for x in raw.get("target_n", [20])},
        max_rows_per_n=int(raw.get("max_rows_per_n", 5)),
        beam_width=int(raw.get("beam_width", 2**8)),
        history_depth=int(raw.get("history_depth", 0)),
        max_steps_factor=int(raw.get("max_steps_factor", 2)),
        require_not_longer_than_previous=bool(
            raw.get("require_not_longer_than_previous", True)
        ),
        compile_once_per_n=bool(raw.get("compile_once_per_n", True)),
        free_graph_memory=bool(raw.get("free_graph_memory", True)),
        clear_cuda_cache=bool(raw.get("clear_cuda_cache", True)),
        run_gc_collect=bool(raw.get("run_gc_collect", True)),
        enable_autocast=bool(raw.get("enable_autocast", True)),
        autocast_dtype=parse_torch_dtype(raw.get("autocast_dtype", "bfloat16")),
        graph_batch_size=int(graph_cfg.get("batch_size", 2**17)),
        graph_dtype=parse_torch_dtype(graph_cfg.get("dtype", "int8")),
        device=graph_cfg.get("device", raw.get("device")),
        direct_forward=direct_forward,
    )


def _parse_profiler_config(raw: dict[str, Any]) -> ProfilerConfig:
    """Build :class:`ProfilerConfig` from raw mapping."""
    return ProfilerConfig(
        enabled=bool(raw.get("enabled", True)),
        top_k=int(raw.get("top_k", 20)),
        record_shapes=bool(raw.get("record_shapes", True)),
        profile_memory=bool(raw.get("profile_memory", True)),
        with_stack=bool(raw.get("with_stack")),
    )


def build_timing_benchmark_config(
    config: dict[str, Any],
) -> AlGraphGPTTimingBenchmarkConfig:
    """
    Build typed benchmark config from parsed YAML mapping.

    Args:
        config: Raw mapping loaded from YAML.

    Returns:
        Parsed benchmark configuration.

    Raises:
        ValueError: If required fields are missing or invalid.

    """
    runtime = _parse_runtime_config(dict(config.get("runtime", {})))
    train = _parse_train_config(dict(config.get("train", {})))
    inference = _parse_inference_config(dict(config.get("inference", {})))
    profiler = _parse_profiler_config(dict(config.get("profiler", {})))

    model_sizes_raw = config.get("model_sizes")
    if not isinstance(model_sizes_raw, list) or not model_sizes_raw:
        raise ValueError("`model_sizes` must be a non-empty list.")

    model_sizes: list[AlGraphGPTSizePreset] = []
    for item in model_sizes_raw:
        if not isinstance(item, dict):
            raise TypeError("Each item in `model_sizes` must be a mapping.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each `model_sizes` item requires non-empty `name`.")
        cfg = dict(item.get("model_config", {}))
        if not cfg:
            raise ValueError(f"model_sizes[{name}] requires `model_config`.")
        if "model_dtype" in cfg and not isinstance(cfg["model_dtype"], torch.dtype):
            cfg["model_dtype"] = parse_torch_dtype(cfg["model_dtype"])
        model_sizes.append(AlGraphGPTSizePreset(name=name, model_config=cfg))

    return AlGraphGPTTimingBenchmarkConfig(
        seed=(int(config["seed"]) if config.get("seed") is not None else None),
        output_root=Path(
            config.get("output_root", "artifacts/pilgrim_lightning/algraphgpt_timing")
        ),
        runtime=runtime,
        train=train,
        inference=inference,
        profiler=profiler,
        model_sizes=model_sizes,
    )


def _build_algraphgpt_model(
    *,
    n: int,
    graph: Any,
    model_config: dict[str, Any],
    device: str | torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Build one AlGraphGPT model with required per-``n`` fields.

    Args:
        n: Pancake size.
        graph: Cayley graph object for generator moves.
        model_config: Base model config mapping.
        device: Target device.

    Returns:
        Tuple ``(model, effective_config)``.

    """
    cfg = dict(model_config)
    cfg.setdefault("num_classes", int(n))
    cfg.setdefault("state_size", int(n))
    cfg.setdefault("generator_moves", graph.permutations_torch)
    model = build_model("AlGraphGPT", cfg).to(device)
    return model, cfg


def _extract_optimizer(configured: Any) -> torch.optim.Optimizer:
    """
    Extract optimizer from Lightning ``configure_optimizers`` output.

    Args:
        configured: Return value of ``configure_optimizers``.

    Returns:
        Optimizer instance.

    Raises:
        TypeError: If optimizer cannot be extracted.

    """
    if isinstance(configured, torch.optim.Optimizer):
        return configured
    if isinstance(configured, dict) and "optimizer" in configured:
        optimizer = configured["optimizer"]
        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer
    if isinstance(configured, (list, tuple)) and configured:
        first = configured[0]
        if isinstance(first, torch.optim.Optimizer):
            return first
        if isinstance(first, dict) and "optimizer" in first:
            optimizer = first["optimizer"]
            if isinstance(optimizer, torch.optim.Optimizer):
                return optimizer
    raise TypeError("Unsupported optimizer config returned by Lightning module.")


def _profile_events_to_frame(events: Any) -> pd.DataFrame:
    """
    Convert torch profiler key averages into a tabular DataFrame.

    Args:
        events: Result of ``prof.key_averages()``.

    Returns:
        DataFrame with per-op timing statistics.

    """
    rows: list[dict[str, Any]] = []
    for item in events:
        rows.append({
            "name": str(item.key),
            "self_cpu_time_total_us": float(getattr(item, "self_cpu_time_total", 0.0)),
            "cpu_time_total_us": float(getattr(item, "cpu_time_total", 0.0)),
            "cuda_time_total_us": float(getattr(item, "cuda_time_total", 0.0)),
            "self_cuda_time_total_us": float(
                getattr(item, "self_cuda_time_total", 0.0)
            ),
            "count": int(getattr(item, "count", 0)),
            "cpu_memory_usage": float(getattr(item, "cpu_memory_usage", 0.0)),
            "self_cpu_memory_usage": float(getattr(item, "self_cpu_memory_usage", 0.0)),
            "cuda_memory_usage": float(getattr(item, "cuda_memory_usage", 0.0)),
            "self_cuda_memory_usage": float(
                getattr(item, "self_cuda_memory_usage", 0.0)
            ),
        })
    return pd.DataFrame(rows)


def _run_train_step_profile(
    *,
    module: PilgrimLightningModule,
    x: torch.Tensor,
    y: torch.Tensor,
    profiler_cfg: ProfilerConfig,
    out_dir: Path,
    preset_name: str,
) -> tuple[pd.DataFrame, str]:
    """
    Profile one train step and export per-op table and trace.

    Args:
        module: Lightning module containing the model.
        x: Training features.
        y: Training targets.
        profiler_cfg: Profiler controls.
        out_dir: Output directory.
        preset_name: Model preset name used in output filenames.

    Returns:
        Tuple ``(events_df, top_table_text)``.

    """
    model = module.model
    model_device = next(model.parameters()).device
    x = x.to(model_device)
    y = y.to(model_device)
    device = model_device
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and "cuda" in str(device):
        activities.append(ProfilerActivity.CUDA)
    model.train()
    optimizer = _extract_optimizer(module.configure_optimizers())
    optimizer.zero_grad(set_to_none=True)
    _sync_if_cuda(device)

    trace_path = out_dir / f"{preset_name}_train_trace.json"
    with profile(
        activities=activities,
        record_shapes=bool(profiler_cfg.record_shapes),
        profile_memory=bool(profiler_cfg.profile_memory),
        with_stack=bool(profiler_cfg.with_stack),
    ) as prof:
        pred = model(x.long())
        loss = F.mse_loss(pred.float(), y.float())
        loss.backward()
        optimizer.step()

    _sync_if_cuda(device)
    prof.export_chrome_trace(str(trace_path))

    sort_key = (
        "self_cuda_time_total"
        if torch.cuda.is_available() and "cuda" in str(device)
        else "self_cpu_time_total"
    )
    top_text = prof.key_averages().table(
        sort_by=sort_key,
        row_limit=int(profiler_cfg.top_k),
    )
    df = _profile_events_to_frame(prof.key_averages())
    df.to_csv(out_dir / f"{preset_name}_train_profile_ops.csv", index=False)
    return df, top_text


@torch.no_grad()
def _run_infer_forward_profile(
    *,
    model: nn.Module,
    x: torch.Tensor,
    profiler_cfg: ProfilerConfig,
    enable_autocast: bool,
    autocast_dtype: torch.dtype,
    out_dir: Path,
    preset_name: str,
) -> tuple[pd.DataFrame, str]:
    """
    Profile one inference forward pass and export op table and trace.

    Args:
        model: Model to profile.
        x: Input tensor.
        profiler_cfg: Profiler controls.
        enable_autocast: Autocast toggle.
        autocast_dtype: Autocast dtype.
        out_dir: Output directory.
        preset_name: Model preset name.

    Returns:
        Tuple ``(events_df, top_table_text)``.

    """
    activities = [ProfilerActivity.CPU]
    device = x.device
    if torch.cuda.is_available() and "cuda" in str(device):
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    _sync_if_cuda(device)

    trace_path = out_dir / f"{preset_name}_infer_trace.json"
    with (
        profile(
            activities=activities,
            record_shapes=bool(profiler_cfg.record_shapes),
            profile_memory=bool(profiler_cfg.profile_memory),
            with_stack=bool(profiler_cfg.with_stack),
        ) as prof,
        torch.autocast(
            device_type="cuda",
            enabled=(
                bool(enable_autocast)
                and torch.cuda.is_available()
                and "cuda" in str(device)
            ),
            dtype=autocast_dtype,
        ),
    ):
        _ = model(x.long())
    _sync_if_cuda(device)

    prof.export_chrome_trace(str(trace_path))
    sort_key = (
        "self_cuda_time_total"
        if torch.cuda.is_available() and "cuda" in str(device)
        else "self_cpu_time_total"
    )
    top_text = prof.key_averages().table(
        sort_by=sort_key,
        row_limit=int(profiler_cfg.top_k),
    )
    df = _profile_events_to_frame(prof.key_averages())
    df.to_csv(out_dir / f"{preset_name}_infer_profile_ops.csv", index=False)
    return df, top_text


@torch.no_grad()
def _measure_forward_latency(
    *,
    model: nn.Module,
    batch: torch.Tensor,
    warmup_iters: int,
    measure_iters: int,
    enable_autocast: bool,
    autocast_dtype: torch.dtype,
    reset_operation_profile_after_warmup: bool = False,
) -> list[float]:
    """
    Measure repeated forward-pass wall-clock latencies.

    Args:
        model: Model under test.
        batch: Input states batch.
        warmup_iters: Number of warmup forwards.
        measure_iters: Number of measured forwards.
        enable_autocast: Autocast toggle.
        autocast_dtype: Autocast dtype.
        reset_operation_profile_after_warmup: Whether to reset model-internal
            operation profile counters after warmup and before measurements.

    Returns:
        List of per-iteration durations in seconds.

    """
    model.eval()
    device = batch.device
    for _ in range(max(0, int(warmup_iters))):
        with torch.autocast(
            device_type="cuda",
            enabled=(
                bool(enable_autocast)
                and torch.cuda.is_available()
                and "cuda" in str(device)
            ),
            dtype=autocast_dtype,
        ):
            _ = model(batch.long())

    if bool(reset_operation_profile_after_warmup) and hasattr(
        model, "reset_operation_profile"
    ):
        model.reset_operation_profile()

    samples: list[float] = []
    for _ in range(max(0, int(measure_iters))):
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.autocast(
            device_type="cuda",
            enabled=(
                bool(enable_autocast)
                and torch.cuda.is_available()
                and "cuda" in str(device)
            ),
            dtype=autocast_dtype,
        ):
            _ = model(batch.long())
        _sync_if_cuda(device)
        samples.append(time.perf_counter() - t0)
    return samples


def _operation_profile_to_df(
    *,
    preset: str,
    profile_stage: str,
    raw_stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Convert semantic model-operation profile stats into a DataFrame.

    Args:
        preset: Model size preset name.
        profile_stage: Profiling stage label (for example ``"train_forward"``).
        raw_stats: Raw model profile mapping.

    Returns:
        Normalized DataFrame sorted by ``total_ms`` descending.

    """
    if not raw_stats:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    total_s = float(sum(float(item.get("total_s", 0.0)) for item in raw_stats.values()))
    for op_name, item in raw_stats.items():
        op_total_s = float(item.get("total_s", 0.0))
        calls = int(item.get("calls", 0.0))
        mean_ms = (op_total_s / calls * 1000.0) if calls > 0 else 0.0
        rows.append({
            "preset": preset,
            "profile_stage": profile_stage,
            "op_name": str(op_name),
            "total_ms": float(op_total_s * 1000.0),
            "calls": int(calls),
            "mean_ms": float(mean_ms),
            "share_pct": float((op_total_s / total_s * 100.0) if total_s > 0 else 0.0),
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("total_ms", ascending=False).reset_index(drop=True)
    return out


def _prepare_forward_batch(
    *,
    n: int,
    device: str | torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Build a synthetic integer-state batch for direct forward timing.

    Args:
        n: Pancake size.
        device: Target device.
        batch_size: Batch size.

    Returns:
        Long tensor of shape ``(batch_size, n)``.

    """
    return torch.randint(
        low=0,
        high=int(n),
        size=(int(batch_size), int(n)),
        device=device,
        dtype=torch.long,
    )


def _run_one_train_benchmark(
    *,
    preset: AlGraphGPTSizePreset,
    config: AlGraphGPTTimingBenchmarkConfig,
    profiler_dir: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], list[tuple[str, str]], pd.DataFrame]:
    """
    Run train-loop timing and optional train-step profiling for one preset.

    Args:
        preset: Model size preset.
        config: Full benchmark config.
        profiler_dir: Directory for profiler exports.
        run_dir: Run output directory.

    Returns:
        Tuple ``(metrics_dict, profiler_tables, operation_profile_df)`` where
        profiler tables is a list of ``(title, text_table)`` entries.

    """
    train_cfg = config.train
    runtime = config.runtime
    n = int(train_cfg.n)
    device = _resolve_device(train_cfg.device)
    graph = make_graph_for_n(
        n,
        batch_size=int(train_cfg.graph_batch_size),
        dtype=train_cfg.graph_dtype,
        device=device,
    )
    model, eff_cfg = _build_algraphgpt_model(
        n=n,
        graph=graph,
        model_config=preset.model_config,
        device=graph.device,
    )
    total_params, trainable_params = _count_parameters(model)

    module = PilgrimLightningModule(
        model=model,
        graph=graph,
        optimization=train_cfg.optimization,
        lipschitz=train_cfg.lipschitz,
        problem_n=n,
    )
    data_module = RandomWalkDataModule(graph=graph, config=train_cfg.data)
    timing_cb = BatchTimingCallback()

    trainer = L.Trainer(
        default_root_dir=str(config.output_root),
        max_epochs=1,
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        precision=runtime.precision,
        deterministic=runtime.deterministic,
        enable_progress_bar=runtime.enable_progress_bar,
        log_every_n_steps=int(runtime.log_every_n_steps),
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        callbacks=[timing_cb],
        limit_train_batches=int(train_cfg.limit_train_batches),
        limit_val_batches=int(train_cfg.limit_val_batches),
    )

    _sync_if_cuda(graph.device)
    fit_t0 = time.perf_counter()
    trainer.fit(module, datamodule=data_module)
    _sync_if_cuda(graph.device)
    fit_elapsed = time.perf_counter() - fit_t0

    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    sample_x, sample_y = next(iter(train_loader))
    sample_x = sample_x.to(graph.device)
    sample_y = sample_y.to(graph.device)

    profiler_tables: list[tuple[str, str]] = []
    if config.profiler.enabled:
        _, top_table = _run_train_step_profile(
            module=module,
            x=sample_x,
            y=sample_y,
            profiler_cfg=config.profiler,
            out_dir=profiler_dir,
            preset_name=preset.name,
        )
        profiler_tables.append((f"{preset.name} train-step profile", top_table))

    operation_df = pd.DataFrame()
    if hasattr(module.model, "enable_operation_profiling") and hasattr(
        module.model, "get_operation_profile"
    ):
        model_device = next(module.model.parameters()).device
        profile_x = sample_x.to(model_device)
        module.model.train()
        module.model.enable_operation_profiling(True, reset=True)
        with torch.no_grad():
            for _ in range(8):
                _ = module.model(profile_x.long())
        raw = module.model.get_operation_profile(reset=True)
        module.model.enable_operation_profiling(False)
        operation_df = _operation_profile_to_df(
            preset=preset.name,
            profile_stage="train_forward",
            raw_stats=raw,
        )
        if not operation_df.empty:
            operation_df.to_csv(
                run_dir / f"operation_profile_{preset.name}_train_forward.csv",
                index=False,
            )

    metric = {
        "preset": preset.name,
        "stage": "train",
        "n": int(n),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "fit_wall_s": float(fit_elapsed),
        "limit_train_batches": int(train_cfg.limit_train_batches),
        "limit_val_batches": int(train_cfg.limit_val_batches),
        "effective_model_config": str(eff_cfg),
        **_summarize_seconds(timing_cb.train_batch_seconds, "train_batch"),
        **_summarize_seconds(timing_cb.val_batch_seconds, "val_batch"),
    }
    return metric, profiler_tables, operation_df


def _run_one_inference_benchmark(
    *,
    preset: AlGraphGPTSizePreset,
    config: AlGraphGPTTimingBenchmarkConfig,
    profiler_dir: Path,
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[str, str]],
    pd.DataFrame,
]:
    """
    Run inference timing (direct forward + beam solve) for one preset.

    Args:
        preset: Model size preset.
        config: Full benchmark config.
        profiler_dir: Directory for profiler outputs.
        run_dir: Run output directory.

    Returns:
        Tuple ``(summary_metrics, solve_time_df, by_n_df, profiler_tables,
        operation_profile_df)``.

    """
    inf = config.inference
    device = _resolve_device(inf.device)
    target_n = sorted(int(x) for x in inf.target_n)

    test_df = pd.read_csv(inf.test_csv_path)
    test_subset = _select_rows_by_n(test_df, set(target_n), int(inf.max_rows_per_n))
    heuristic_paths = _build_heuristic_paths(test_subset)
    rows = _flatten_competition_rows(test_subset)
    submission_rows = load_or_create_submission_rows(
        rows,
        heuristic_paths,
        base_pkl=None,
        deep_copy=True,
    )

    models_dict: dict[int, nn.Module] = {}
    graphs_dict: dict[int, Any] = {}
    for n in target_n:
        graph = make_graph_for_n(
            n,
            batch_size=int(inf.graph_batch_size),
            dtype=inf.graph_dtype,
            device=device,
        )
        model, _ = _build_algraphgpt_model(
            n=n,
            graph=graph,
            model_config=preset.model_config,
            device=graph.device,
        )
        models_dict[n] = model
        graphs_dict[n] = graph

    n_ref = int(max(target_n))
    forward_batch = _prepare_forward_batch(
        n=n_ref,
        device=device,
        batch_size=int(inf.direct_forward.batch_size),
    )
    ref_model = models_dict[n_ref]
    if hasattr(ref_model, "enable_operation_profiling"):
        ref_model.enable_operation_profiling(True, reset=True)

    forward_samples = _measure_forward_latency(
        model=ref_model,
        batch=forward_batch,
        warmup_iters=int(inf.direct_forward.warmup_iters),
        measure_iters=int(inf.direct_forward.measure_iters),
        enable_autocast=bool(inf.direct_forward.enable_autocast),
        autocast_dtype=inf.direct_forward.autocast_dtype,
        reset_operation_profile_after_warmup=True,
    )

    operation_df = pd.DataFrame()
    if hasattr(ref_model, "get_operation_profile"):
        raw = ref_model.get_operation_profile(reset=True)
        if hasattr(ref_model, "enable_operation_profiling"):
            ref_model.enable_operation_profiling(False)
        operation_df = _operation_profile_to_df(
            preset=preset.name,
            profile_stage="infer_forward",
            raw_stats=raw,
        )
        if not operation_df.empty:
            operation_df.to_csv(
                run_dir / f"operation_profile_{preset.name}_infer_forward.csv",
                index=False,
            )

    profiler_tables: list[tuple[str, str]] = []
    if config.profiler.enabled:
        _, top_table = _run_infer_forward_profile(
            model=models_dict[n_ref],
            x=forward_batch,
            profiler_cfg=config.profiler,
            enable_autocast=bool(inf.direct_forward.enable_autocast),
            autocast_dtype=inf.direct_forward.autocast_dtype,
            out_dir=profiler_dir,
            preset_name=preset.name,
        )
        profiler_tables.append((f"{preset.name} infer-forward profile", top_table))

    collector = BeamTimingCollector()
    beam_cfg = BeamInferenceConfig(
        beam_width=int(inf.beam_width),
        history_depth=int(inf.history_depth),
        max_steps_factor=int(inf.max_steps_factor),
        require_not_longer_than_previous=bool(inf.require_not_longer_than_previous),
        compile_once_per_n=bool(inf.compile_once_per_n),
        free_graph_memory=bool(inf.free_graph_memory),
        clear_cuda_cache=bool(inf.clear_cuda_cache),
        run_gc_collect=bool(inf.run_gc_collect),
    )
    _sync_if_cuda(device)
    beam_t0 = time.perf_counter()
    beam_stats = run_targeted_beam_inference(
        rows=rows,
        submission_rows=submission_rows,
        heuristic_paths=heuristic_paths,
        target_n=set(target_n),
        models_dict=models_dict,
        graphs_dict=graphs_dict,
        solve_fn=solve,
        device=device,
        beam_run=collector,
        config=beam_cfg,
        enable_autocast=bool(inf.enable_autocast),
        autocast_dtype=inf.autocast_dtype,
    )
    _sync_if_cuda(device)
    beam_elapsed = time.perf_counter() - beam_t0

    by_n, stat_df = compute_stats_by_n(
        test_df=test_subset,
        heuristic_paths=heuristic_paths,
        submission_rows=submission_rows,
    )
    by_n.to_csv(run_dir / f"by_n_{preset.name}.csv", index=False)
    stat_df.to_csv(run_dir / f"stat_df_{preset.name}.csv", index=False)

    solve_df = pd.DataFrame(collector.rows)
    if not solve_df.empty:
        solve_df["preset"] = preset.name
        solve_df.to_csv(run_dir / f"beam_solve_times_{preset.name}.csv", index=False)

    forward_stats = _summarize_seconds(forward_samples, "forward")
    solve_stats = _summarize_seconds(
        [] if solve_df.empty else solve_df["solve_time_s"].tolist(),
        "beam_solve",
    )
    total_params, trainable_params = _count_parameters(models_dict[n_ref])

    metric = {
        "preset": preset.name,
        "stage": "inference",
        "n_ref": n_ref,
        "target_n": str(target_n),
        "rows_evaluated": len(rows),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "beam_wall_s": float(beam_elapsed),
        "beam_attempted": int(beam_stats.attempted),
        "beam_accepted": int(beam_stats.accepted),
        "beam_longer_rejected": int(beam_stats.longer_rejected),
        "beam_oom_count": int(beam_stats.oom_count),
        "beam_shorter_than_previous": int(beam_stats.shorter_than_previous),
        "beam_shorter_than_heuristic": int(beam_stats.shorter_than_heuristic),
        "beam_no_longer_than_previous": int(beam_stats.no_longer_than_previous),
        **forward_stats,
        **solve_stats,
    }
    return metric, solve_df, by_n, profiler_tables, operation_df


def _config_to_yaml_ready(config: AlGraphGPTTimingBenchmarkConfig) -> dict[str, Any]:
    """
    Convert benchmark config dataclasses to YAML-safe primitives.

    Args:
        config: Typed benchmark config.

    Returns:
        Dictionary containing YAML-serializable values.

    """
    raw = asdict(config)

    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, set):
            return sorted(int(x) for x in value)
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        return value

    return _convert(raw)


def _make_markdown_table(df: pd.DataFrame) -> str:
    """
    Render a compact markdown table for DataFrame output.

    Args:
        df: Input DataFrame.

    Returns:
        Markdown table string.

    """
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = [str(col) for col in df.columns]
        rows: list[str] = []
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for values in df.itertuples(index=False, name=None):
            row = [str(v) for v in values]
            rows.append("| " + " | ".join(row) + " |")
        return "\n".join(rows)


def _write_summary_markdown(
    *,
    run_dir: Path,
    config: AlGraphGPTTimingBenchmarkConfig,
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    by_n_frames: list[pd.DataFrame],
    profiler_tables: list[tuple[str, str]],
    operation_profile_frames: list[pd.DataFrame],
) -> Path:
    """
    Write markdown summary with timing tables and profiler snapshots.

    Args:
        run_dir: Run output directory.
        config: Benchmark config.
        train_df: Training metrics table.
        infer_df: Inference metrics table.
        by_n_frames: Per-preset by-n stats.
        profiler_tables: List of top-op profiler text tables.
        operation_profile_frames: List of semantic operation profile frames.

    Returns:
        Path to generated summary markdown.

    """
    summary_path = run_dir / "SUMMARY.md"
    lines: list[str] = []
    lines.append("# AlGraphGPT Timing Benchmark Summary")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- seed: `{config.seed}`")
    lines.append(f"- model_sizes: `{[preset.name for preset in config.model_sizes]}`")
    lines.append("")
    lines.append("## Train Timings")
    lines.append("")
    lines.append(
        _make_markdown_table(
            train_df[
                [
                    "preset",
                    "n",
                    "total_params",
                    "fit_wall_s",
                    "train_batch_mean_ms",
                    "train_batch_p90_ms",
                    "val_batch_mean_ms",
                    "val_batch_p90_ms",
                ]
            ]
            if not train_df.empty
            else train_df
        )
    )
    lines.append("")
    lines.append("## Inference Timings")
    lines.append("")
    lines.append(
        _make_markdown_table(
            infer_df[
                [
                    "preset",
                    "target_n",
                    "rows_evaluated",
                    "forward_mean_ms",
                    "forward_p90_ms",
                    "beam_wall_s",
                    "beam_solve_mean_ms",
                    "beam_solve_p90_ms",
                    "beam_attempted",
                    "beam_accepted",
                ]
            ]
            if not infer_df.empty
            else infer_df
        )
    )
    lines.append("")
    if by_n_frames:
        lines.append("## by_n Evaluation")
        lines.append("")
        for idx, frame in enumerate(by_n_frames):
            lines.append(f"### preset={frame['preset'].iloc[0]}")
            lines.append("")
            lines.append(
                _make_markdown_table(
                    frame[["n", "sum_n", "score", "prob_step", "potential"]]
                )
            )
            if idx < len(by_n_frames) - 1:
                lines.append("")
    if operation_profile_frames:
        lines.append("")
        lines.append("## Semantic Model Operation Timings")
        lines.append("")
        for op_df in operation_profile_frames:
            if op_df.empty:
                continue
            preset = str(op_df["preset"].iloc[0])
            stage = str(op_df["profile_stage"].iloc[0])
            top = op_df[["op_name", "total_ms", "mean_ms", "calls", "share_pct"]].head(
                12
            )
            lines.append(f"### preset={preset}, stage={stage}")
            lines.append("")
            lines.append(_make_markdown_table(top))
            lines.append("")
    if profiler_tables:
        lines.append("")
        lines.append("## Profiler Top Ops")
        lines.append("")
        for title, table in profiler_tables:
            lines.append(f"### {title}")
            lines.append("")
            lines.append("```text")
            lines.append(table)
            lines.append("```")
            lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def run_algraphgpt_timing_benchmark(
    config: AlGraphGPTTimingBenchmarkConfig,
) -> AlGraphGPTTimingBenchmarkResult:
    """
    Run full AlGraphGPT timing benchmark and export artifacts.

    Args:
        config: Parsed benchmark config.

    Returns:
        Benchmark result with output paths and metric tables.

    """
    if config.seed is not None:
        set_seed(int(config.seed))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root / f"run_{stamp}"
    profiler_dir = run_dir / "profiler"
    run_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir.mkdir(parents=True, exist_ok=True)

    config_dump = _config_to_yaml_ready(config)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_dump, f, sort_keys=False, allow_unicode=False)

    train_rows: list[dict[str, Any]] = []
    infer_rows: list[dict[str, Any]] = []
    beam_frames: list[pd.DataFrame] = []
    by_n_frames: list[pd.DataFrame] = []
    profiler_tables: list[tuple[str, str]] = []
    operation_profile_frames: list[pd.DataFrame] = []

    for preset in config.model_sizes:
        if config.train.enabled:
            row, tables, op_df = _run_one_train_benchmark(
                preset=preset,
                config=config,
                profiler_dir=profiler_dir,
                run_dir=run_dir,
            )
            train_rows.append(row)
            profiler_tables.extend(tables)
            if not op_df.empty:
                operation_profile_frames.append(op_df)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if config.inference.enabled:
            row, solve_df, by_n, tables, op_df = _run_one_inference_benchmark(
                preset=preset,
                config=config,
                profiler_dir=profiler_dir,
                run_dir=run_dir,
            )
            infer_rows.append(row)
            profiler_tables.extend(tables)
            if not op_df.empty:
                operation_profile_frames.append(op_df)
            by_n = by_n.copy()
            by_n["preset"] = preset.name
            by_n_frames.append(by_n)
            if not solve_df.empty:
                beam_frames.append(solve_df)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    train_df = pd.DataFrame(train_rows)
    infer_df = pd.DataFrame(infer_rows)
    beam_df = (
        pd.concat(beam_frames, axis=0, ignore_index=True)
        if beam_frames
        else pd.DataFrame()
    )
    by_n_all = (
        pd.concat(by_n_frames, axis=0, ignore_index=True)
        if by_n_frames
        else pd.DataFrame()
    )
    operation_df = (
        pd.concat(operation_profile_frames, axis=0, ignore_index=True)
        if operation_profile_frames
        else pd.DataFrame()
    )

    train_df.to_csv(run_dir / "train_timings.csv", index=False)
    infer_df.to_csv(run_dir / "inference_timings.csv", index=False)
    beam_df.to_csv(run_dir / "beam_solve_times.csv", index=False)
    by_n_all.to_csv(run_dir / "by_n_all.csv", index=False)
    operation_df.to_csv(run_dir / "operation_profile_all.csv", index=False)

    summary_path = _write_summary_markdown(
        run_dir=run_dir,
        config=config,
        train_df=train_df,
        infer_df=infer_df,
        by_n_frames=by_n_frames,
        profiler_tables=profiler_tables,
        operation_profile_frames=operation_profile_frames,
    )

    return AlGraphGPTTimingBenchmarkResult(
        run_dir=run_dir,
        train_timings=train_df,
        inference_timings=infer_df,
        beam_solve_times=beam_df,
        operation_profile=operation_df,
        summary_path=summary_path,
    )


def run_algraphgpt_timing_from_yaml(
    path: str | Path,
) -> AlGraphGPTTimingBenchmarkResult:
    """
    Run timing benchmark directly from YAML config path.

    Args:
        path: YAML path.

    Returns:
        Benchmark result.

    """
    raw = load_yaml_config(path)
    cfg = build_timing_benchmark_config(raw)
    return run_algraphgpt_timing_benchmark(cfg)
