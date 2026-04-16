"""Configuration models for the Pilgrim Lightning training/inference library."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class RandomWalkDataConfig:
    """
    Configuration for random-walk based train/validation data generation.

    Args:
        rw_mode: Sampling mode passed to ``graph.random_walks``.
        rw_width: Number of walk seeds sampled per refresh.
        rw_length: Base walk length used to build walk-length schedule.
        rw_lengths: Optional explicit schedule as ``(factor, length)`` pairs.
            When ``None`` the default schedule from the notebooks is used.
        rw_refresh_interval: Refresh train/val splits every N epochs.
        val_ratio: Validation split ratio.
        batch_size: Batch size for both train and validation dataloaders.
        max_samples_cap: Maximum number of random-walk samples used per refresh.
        seed: Base seed for deterministic subsampling and splitting.
        num_workers: Number of DataLoader workers.

    """

    rw_mode: str = "nbt"
    rw_width: int = 2500
    rw_length: int = 32
    rw_lengths: list[tuple[float, int]] | None = None
    rw_refresh_interval: int = 1
    val_ratio: float = 0.1
    batch_size: int = 2048
    max_samples_cap: int = 200_000
    seed: int = 42
    num_workers: int = 0


@dataclass(slots=True)
class OptimizationConfig:
    """
    Optimization and scheduler configuration.

    Args:
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        num_epochs: Number of training epochs.
        lr_scheduler: Scheduler specification compatible with
            ``pilgrim.utils.training_utils.lr_scheduler_ctor_from_cfg``.

    """

    lr: float = 1e-3
    weight_decay: float = 2.5e-4
    num_epochs: int = 300
    lr_scheduler: str | dict[str, Any] | None = None


@dataclass(slots=True)
class LipschitzConfig:
    """
    Configuration for optional 1-Lipschitz expansion regularization.

    Args:
        weight: Regularization multiplier added to MSE.
        max_states: Optional state cap per batch for lip-loss computation.
        generator_indices: Optional explicit generator subset.
        max_generators: Optional cap for number of generators.
        seed: Optional RNG seed for lip-loss subsampling.
        state_batch_size: Optional internal state batch size for lip-loss.
        reduction: Reduction mode in ``lipschitz_expansion_loss``.
        val_metric: Whether to compute val lip metric.

    """

    weight: float = 0.0
    max_states: int | None = None
    generator_indices: list[int] | None = None
    max_generators: int | None = None
    seed: int | None = None
    state_batch_size: int | None = None
    reduction: str = "mean"
    val_metric: bool = False


@dataclass(slots=True)
class LightningRuntimeConfig:
    """
    Runtime configuration passed to ``lightning.Trainer``.

    Args:
        accelerator: Trainer accelerator (for example ``"auto"`` or ``"gpu"``).
        devices: Number of devices or device selection string.
        precision: Trainer precision mode.
        gradient_clip_val: Optional gradient clipping value.
        log_every_n_steps: Logging frequency in steps.
        enable_progress_bar: Toggle progress bar.
        deterministic: Enable deterministic Trainer mode.

    """

    accelerator: str = "auto"
    devices: int | str | list[int] = 1
    precision: str | int = "32-true"
    gradient_clip_val: float | None = None
    log_every_n_steps: int = 1
    enable_progress_bar: bool = True
    deterministic: bool = False


@dataclass(slots=True)
class AimRunConfig:
    """
    Aim tracking settings for train/inference runs.

    Args:
        experiment: Aim experiment name.
        repo: Optional Aim repository path.
        tags: Optional list of run tags.
        stage: Logical stage label such as ``"train"`` or ``"beam_eval"``.
        notebook: Optional notebook identifier to preserve lineage.
        model_name: Optional model name stored in run metadata.
        extra_meta: Additional metadata stored under ``meta/*``.

    """

    experiment: str
    repo: Path | None = None
    tags: list[str] = field(default_factory=list)
    stage: str = "train"
    notebook: str | None = None
    model_name: str | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainJobConfig:
    """
    Single-``n`` training job configuration.

    Args:
        n: Pancake size used for graph/model construction.
        model_name: Model class name from ``pilgrim.model``.
        model_config: Base model config dictionary.
        output_dir: Directory for state-dict checkpoints.
        model_artifacts_dir: Directory for ``save_one`` artifacts.
        data: Random-walk datamodule config.
        optimization: Optimizer/scheduler config.
        lipschitz: Lipschitz regularization settings.
        runtime: Lightning Trainer runtime settings.
        aim: Optional Aim settings.
        graph_batch_size: Batch size used for CayleyGraph internals.
        graph_dtype: State dtype used in CayleyGraph.
        device: Optional graph device override.
        fit_verbose: Whether to print per-job training summary.

    """

    n: int
    model_name: str
    model_config: dict[str, Any]
    output_dir: Path
    model_artifacts_dir: Path
    data: RandomWalkDataConfig = field(default_factory=RandomWalkDataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    lipschitz: LipschitzConfig = field(default_factory=LipschitzConfig)
    runtime: LightningRuntimeConfig = field(default_factory=LightningRuntimeConfig)
    aim: AimRunConfig | None = None
    graph_batch_size: int = 2**17
    graph_dtype: torch.dtype = torch.int8
    device: str | torch.device | None = None
    fit_verbose: bool = True


@dataclass(slots=True)
class KaggleInferenceConfig:
    """
    Configuration for Kaggle test-data inference experiment.

    Args:
        test_csv_path: Path to Kaggle test CSV.
        target_n: Sizes for which beam inference is run.
        base_submission_pkl: Optional pickle path used as baseline solutions.
        submission_rows_out: Output pickle path for updated submission rows.
        beam_width: Beam width passed to ``solve``.
        history_depth: Beam history depth.
        max_steps_factor: Max-steps factor multiplied by ``n``.
        require_not_longer_than_previous: Keep candidate only when it is not
            longer than existing solution.
        compile_once_per_n: Whether each model is compiled once per ``n``.
        free_graph_memory: Whether to call ``graph.free_memory()`` each attempt.
        clear_cuda_cache: Whether to call ``torch.cuda.empty_cache()`` each attempt.
        run_gc_collect: Whether to call ``gc.collect()`` each attempt.
        checkpoint_path: Optional periodic checkpoint path.
        checkpoint_every_attempts: Checkpoint frequency in attempts.
        enable_autocast: Toggle autocast in beam inference.
        autocast_dtype: Autocast dtype for beam inference.
        aim: Optional Aim run settings for beam stage.

    """

    test_csv_path: Path
    target_n: set[int]
    base_submission_pkl: Path | None = None
    submission_rows_out: Path = Path("submission_rows_run.pkl")
    beam_width: int = 2**10
    history_depth: int = 0
    max_steps_factor: int = 2
    require_not_longer_than_previous: bool = True
    compile_once_per_n: bool = True
    free_graph_memory: bool = True
    clear_cuda_cache: bool = True
    run_gc_collect: bool = True
    checkpoint_path: Path | None = None
    checkpoint_every_attempts: int = 1
    enable_autocast: bool = True
    autocast_dtype: torch.dtype = torch.bfloat16
    aim: AimRunConfig | None = None
