"""End-to-end training and inference pipelines for Pilgrim Lightning runs."""

from __future__ import annotations

import copy
import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from cayleypy import CayleyGraph
from pilgrim.pancake_competition import (
    BeamInferenceConfig,
    BeamInferenceStats,
    compute_stats_by_n,
    load_or_create_submission_rows,
    run_targeted_beam_inference,
    save_submission_rows,
)
from pilgrim.utils.model_io import load_one, save_one
from pilgrim.utils.pancake_utils import make_graph_for_n, pancake_sort_path, solve
from pilgrim.utils.reproducibility import set_seed
from torch import nn
from tqdm.auto import tqdm

from .aim import AimTrackingCallback, open_aim_run, to_aim_serializable
from .checkpointing import StateDictCheckpointCallback
from .config import KaggleInferenceConfig, TrainJobConfig
from .data import RandomWalkDataModule
from .model_factory import build_model
from .module import PilgrimLightningModule
from .yaml_config import (
    build_inference_config_from_config,
    build_train_jobs_from_config,
    parse_torch_dtype,
)


@dataclass(slots=True)
class TrainingResult:
    """
    Result object produced for a trained ``n``.

    Args:
        n: Pancake size.
        model: Trained model moved to CPU.
        graph: Graph used for training.
        model_path: Path to serialized artifact produced by ``save_one``.
        checkpoint_dir: Directory with state-dict checkpoints.
        elapsed_seconds: Training wall-clock time.
        model_config: Effective model config for this ``n``.

    """

    n: int
    model: nn.Module
    graph: CayleyGraph
    model_path: Path
    checkpoint_dir: Path
    elapsed_seconds: float
    model_config: dict[str, Any]


@dataclass(slots=True)
class KaggleInferenceResult:
    """
    Result object for Kaggle test-data beam inference.

    Args:
        stats: Aggregate beam inference counters.
        by_n: Per-``n`` score statistics.
        stat_df: Row-level merged stats.
        submission_rows: Updated submission rows.
        heuristic_paths: Heuristic baseline paths.

    """

    stats: BeamInferenceStats
    by_n: pd.DataFrame
    stat_df: pd.DataFrame
    submission_rows: list[dict[str, Any]]
    heuristic_paths: list[str]


def train_jobs(
    jobs: list[TrainJobConfig], *, seed: int | None = None
) -> dict[int, TrainingResult]:
    """
    Train multiple per-``n`` jobs using Lightning.

    Args:
        jobs: List of training job configurations.
        seed: Optional global seed.

    Returns:
        Mapping ``n -> training result``.

    """
    if seed is not None:
        set_seed(int(seed))

    results: dict[int, TrainingResult] = {}
    for job in jobs:
        results[int(job.n)] = train_one_job(job)
    return results


def train_one_job(job: TrainJobConfig) -> TrainingResult:
    """
    Run a single ``n`` training job.

    Args:
        job: Normalized training job config.

    Returns:
        Training result with model/graph/artifact references.

    """
    device = resolve_device(job.device)
    graph = make_graph_for_n(
        int(job.n),
        batch_size=int(job.graph_batch_size),
        dtype=job.graph_dtype,
        device=device,
    )

    model_cfg = copy.deepcopy(dict(job.model_config))
    model_cfg.setdefault("num_classes", int(job.n))
    model_cfg.setdefault("state_size", int(job.n))

    if job.model_name in {"AliceInCayleyland", "AlGraphGPT"}:
        model_cfg.setdefault("generator_moves", graph.permutations_torch)

    model = build_model(job.model_name, model_cfg).to(graph.device)

    module = PilgrimLightningModule(
        model=model,
        graph=graph,
        optimization=job.optimization,
        lipschitz=job.lipschitz,
        problem_n=int(job.n),
    )
    data_module = RandomWalkDataModule(graph=graph, config=job.data)

    checkpoint_callback = StateDictCheckpointCallback(job.output_dir)
    callbacks: list[Any] = [checkpoint_callback]
    if job.aim is not None:
        callbacks.append(
            AimTrackingCallback(
                job.aim,
                hparams={
                    "n": int(job.n),
                    "model_name": job.model_name,
                    "model_config": to_aim_serializable(model_cfg),
                    "data": to_aim_serializable(asdict(job.data)),
                    "optimization": to_aim_serializable(asdict(job.optimization)),
                    "lipschitz": to_aim_serializable(asdict(job.lipschitz)),
                },
                context={"n": int(job.n)},
            )
        )

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(job.output_dir),
        "max_epochs": int(job.optimization.num_epochs),
        "accelerator": job.runtime.accelerator,
        "devices": job.runtime.devices,
        "precision": job.runtime.precision,
        "log_every_n_steps": int(job.runtime.log_every_n_steps),
        "enable_progress_bar": bool(job.runtime.enable_progress_bar),
        "deterministic": bool(job.runtime.deterministic),
        "callbacks": callbacks,
        "enable_checkpointing": False,
        "logger": False,
        "num_sanity_val_steps": 0,
        "reload_dataloaders_every_n_epochs": 1,
    }
    if job.runtime.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = float(job.runtime.gradient_clip_val)

    trainer = L.Trainer(**trainer_kwargs)

    t0 = time.perf_counter()
    trainer.fit(module, datamodule=data_module)
    elapsed = time.perf_counter() - t0

    trained_model = module.model.cpu()
    model_path = save_one(
        int(job.n),
        trained_model,
        model_cfg,
        out_dir=job.model_artifacts_dir,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if job.fit_verbose:
        print(
            f"n={int(job.n)} trained in {elapsed:.2f}s | "
            f"checkpoints={job.output_dir} | model={model_path}"
        )

    return TrainingResult(
        n=int(job.n),
        model=trained_model,
        graph=graph,
        model_path=model_path,
        checkpoint_dir=job.output_dir,
        elapsed_seconds=float(elapsed),
        model_config=model_cfg,
    )


def run_kaggle_inference(
    config: KaggleInferenceConfig,
    *,
    models_dict: dict[int, nn.Module],
    graphs_dict: dict[int, CayleyGraph],
    device: str | torch.device | None = None,
) -> KaggleInferenceResult:
    """
    Run Kaggle beam inference and compute score statistics.

    Args:
        config: Inference configuration.
        models_dict: Mapping ``n -> model``.
        graphs_dict: Mapping ``n -> graph``.
        device: Device used for inference.

    Returns:
        Inference result object.

    """
    test_df = pd.read_csv(config.test_csv_path)
    heuristic_paths = build_heuristic_paths(test_df)
    rows = list(test_df.itertuples(index=False))

    submission_rows = load_or_create_submission_rows(
        rows,
        heuristic_paths,
        base_pkl=config.base_submission_pkl,
        deep_copy=True,
    )

    beam_cfg = BeamInferenceConfig(
        beam_width=int(config.beam_width),
        history_depth=int(config.history_depth),
        max_steps_factor=int(config.max_steps_factor),
        require_not_longer_than_previous=bool(config.require_not_longer_than_previous),
        compile_once_per_n=bool(config.compile_once_per_n),
        free_graph_memory=bool(config.free_graph_memory),
        clear_cuda_cache=bool(config.clear_cuda_cache),
        run_gc_collect=bool(config.run_gc_collect),
        checkpoint_path=config.checkpoint_path,
        checkpoint_every_attempts=int(config.checkpoint_every_attempts),
    )

    run = None
    if config.aim is not None:
        run = open_aim_run(
            config.aim,
            hparams={
                "target_n": sorted(config.target_n),
                "beam_width": int(config.beam_width),
                "history_depth": int(config.history_depth),
            },
        )

    infer_device = resolve_device(device)
    try:
        stats = run_targeted_beam_inference(
            rows=rows,
            submission_rows=submission_rows,
            heuristic_paths=heuristic_paths,
            target_n=set(int(n) for n in config.target_n),
            models_dict=models_dict,
            graphs_dict=graphs_dict,
            solve_fn=solve,
            device=infer_device,
            beam_run=run,
            config=beam_cfg,
            enable_autocast=bool(config.enable_autocast),
            autocast_dtype=config.autocast_dtype,
        )
    finally:
        if run is not None:
            run.close()

    save_submission_rows(submission_rows, config.submission_rows_out)
    by_n, stat_df = compute_stats_by_n(
        test_df=test_df,
        heuristic_paths=heuristic_paths,
        submission_rows=submission_rows,
    )

    return KaggleInferenceResult(
        stats=stats,
        by_n=by_n,
        stat_df=stat_df,
        submission_rows=submission_rows,
        heuristic_paths=heuristic_paths,
    )


def run_from_config(
    config: dict[str, Any],
    *,
    mode: str = "run",
) -> dict[str, Any]:
    """
    Execute train/infer workflow from parsed YAML mapping.

    Args:
        config: Parsed YAML config.
        mode: One of ``"run"``, ``"train"``, or ``"infer"``.

    Returns:
        Dictionary with produced artifacts.

    """
    seed = config.get("seed")
    if seed is not None:
        set_seed(int(seed))

    train_results: dict[int, TrainingResult] = {}
    inference_result: KaggleInferenceResult | None = None

    if mode in {"run", "train"}:
        jobs = build_train_jobs_from_config(config)
        train_results = train_jobs(jobs)

    if mode in {"run", "infer"}:
        inference_cfg = build_inference_config_from_config(config)
        if inference_cfg is None:
            raise ValueError("Inference is disabled or missing in config.")

        models_dict, graphs_dict = prepare_inference_assets(
            config=config,
            train_results=train_results,
            target_n=inference_cfg.target_n,
        )
        inference_result = run_kaggle_inference(
            inference_cfg,
            models_dict=models_dict,
            graphs_dict=graphs_dict,
        )

    return {
        "train_results": train_results,
        "inference_result": inference_result,
    }


def prepare_inference_assets(
    *,
    config: dict[str, Any],
    train_results: dict[int, TrainingResult],
    target_n: set[int],
) -> tuple[dict[int, nn.Module], dict[int, CayleyGraph]]:
    """
    Prepare model/graph mappings used by beam inference.

    Args:
        config: Parsed run config.
        train_results: In-memory training outputs.
        target_n: Target sizes required for inference.

    Returns:
        Tuple ``(models_dict, graphs_dict)``.

    """
    models_dict: dict[int, nn.Module] = {}
    graphs_dict: dict[int, CayleyGraph] = {}

    for n, result in train_results.items():
        models_dict[int(n)] = result.model
        graphs_dict[int(n)] = result.graph

    missing = sorted(set(int(n) for n in target_n) - set(models_dict.keys()))
    if missing:
        train_cfg = config.get("train", {})
        infer_cfg = config.get("inference", {})
        model_name = str(
            infer_cfg.get("model_name", train_cfg.get("model_name", "AlPilgrim"))
        )
        artifacts_dir = Path(
            infer_cfg.get(
                "model_artifacts_dir",
                train_cfg.get(
                    "model_artifacts_dir", "artifacts/pilgrim_lightning/models"
                ),
            )
        )
        infer_graph_cfg = dict(infer_cfg.get("graph", {}))
        graph_batch_size = int(infer_graph_cfg.get("batch_size", 2**17))
        graph_dtype = parse_torch_dtype(infer_graph_cfg.get("dtype", "int8"))
        graph_device = infer_graph_cfg.get("device")

        loaded = load_models_from_artifacts(
            model_name=model_name,
            model_artifacts_dir=artifacts_dir,
            n_values=missing,
            device=resolve_device(graph_device),
        )
        models_dict.update(loaded)

        for n in missing:
            graphs_dict[int(n)] = make_graph_for_n(
                int(n),
                batch_size=graph_batch_size,
                dtype=graph_dtype,
                device=resolve_device(graph_device),
            )

    return models_dict, graphs_dict


def load_models_from_artifacts(
    *,
    model_name: str,
    model_artifacts_dir: str | Path,
    n_values: list[int],
    device: str | torch.device,
) -> dict[int, nn.Module]:
    """
    Load serialized per-``n`` models from ``save_one`` artifacts.

    Args:
        model_name: Model class name used for reconstruction.
        model_artifacts_dir: Directory containing ``model_n{n}.pt`` files.
        n_values: List of sizes to load.
        device: Target device for loaded models.

    Returns:
        Mapping ``n -> loaded model``.

    """
    models: dict[int, nn.Module] = {}
    base = Path(model_artifacts_dir)
    for n in n_values:
        path = base / f"model_n{int(n)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found for n={n}: {path}")
        _, model, _ = load_one(
            path,
            model_ctor=lambda cfg: build_model(model_name, cfg),
            device=device,
            strict=True,
        )
        models[int(n)] = model
    return models


def build_heuristic_paths(test_df: pd.DataFrame) -> list[str]:
    """
    Build baseline heuristic pancake-sort paths for test dataframe.

    Args:
        test_df: Kaggle test dataframe with ``permutation`` column.

    Returns:
        List of dotted move strings aligned with dataframe rows.

    """
    out: list[str] = []
    iterator = tqdm(
        test_df.itertuples(index=False),
        total=len(test_df),
        desc="heuristic",
    )
    for row in iterator:
        perm = [int(x) for x in str(row.permutation).split(",")]
        out.append(".".join(pancake_sort_path(perm)))
    return out


def resolve_device(device: str | torch.device | None) -> str | torch.device:
    """
    Resolve runtime device preference.

    Args:
        device: Explicit device override.

    Returns:
        Resolved device string/object.

    """
    if device is not None and str(device).lower() != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
