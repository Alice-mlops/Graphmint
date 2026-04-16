"""YAML configuration loading and conversion helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import (
    AimRunConfig,
    KaggleInferenceConfig,
    LightningRuntimeConfig,
    LipschitzConfig,
    OptimizationConfig,
    RandomWalkDataConfig,
    TrainJobConfig,
)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load full run configuration from YAML file.

    Args:
        path: YAML file path.

    Returns:
        Parsed configuration mapping.

    """
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping.")
    return data


def build_train_jobs_from_config(config: Mapping[str, Any]) -> list[TrainJobConfig]:
    """
    Build per-``n`` training jobs from loaded run config.

    Args:
        config: Full run configuration mapping.

    Returns:
        List of normalized training jobs.

    """
    train_cfg_raw = config.get("train", {})
    if not isinstance(train_cfg_raw, Mapping):
        raise TypeError("config['train'] must be a mapping.")
    train_cfg = dict(train_cfg_raw)

    enabled = bool(train_cfg.get("enabled", True))
    if not enabled:
        return []

    n_values_raw = train_cfg.get("n_values")
    if n_values_raw is None:
        raise ValueError("config['train']['n_values'] is required.")
    n_values = [int(n) for n in n_values_raw]

    model_name = str(train_cfg.get("model_name", "AlPilgrim"))
    base_model_cfg = dict(train_cfg.get("base_model_config", {}))

    output_root = Path(
        train_cfg.get("output_dir", "artifacts/pilgrim_lightning/checkpoints")
    )
    artifacts_root = Path(
        train_cfg.get("model_artifacts_dir", "artifacts/pilgrim_lightning/models")
    )

    base_data = dict(train_cfg.get("data", {}))
    base_opt = dict(train_cfg.get("optimization", {}))
    base_lip = dict(train_cfg.get("lipschitz", {}))
    base_runtime = dict(train_cfg.get("runtime", {}))
    base_aim = train_cfg.get("aim")
    graph_cfg = dict(train_cfg.get("graph", {}))

    per_n_overrides_raw = train_cfg.get("per_n_overrides", {})
    if not isinstance(per_n_overrides_raw, Mapping):
        raise TypeError("config['train']['per_n_overrides'] must be a mapping.")

    jobs: list[TrainJobConfig] = []
    for n in n_values:
        per_n_raw = per_n_overrides_raw.get(str(n), per_n_overrides_raw.get(n, {}))
        if per_n_raw is None:
            per_n_raw = {}
        if not isinstance(per_n_raw, Mapping):
            raise TypeError(f"per_n_overrides[{n}] must be a mapping.")
        per_n = dict(per_n_raw)

        model_cfg = deep_merge(
            base_model_cfg,
            dict(per_n.get("model_config", {})),
        )
        model_cfg.setdefault("num_classes", int(n))
        model_cfg.setdefault("state_size", int(n))
        if "model_dtype" in model_cfg and not isinstance(
            model_cfg["model_dtype"], torch.dtype
        ):
            model_cfg["model_dtype"] = parse_torch_dtype(model_cfg["model_dtype"])

        data_cfg = RandomWalkDataConfig(
            **deep_merge(base_data, dict(per_n.get("data", {})))
        )
        opt_cfg = OptimizationConfig(
            **deep_merge(base_opt, dict(per_n.get("optimization", {})))
        )
        lip_cfg = LipschitzConfig(
            **deep_merge(base_lip, dict(per_n.get("lipschitz", {})))
        )
        runtime_cfg = LightningRuntimeConfig(
            **deep_merge(base_runtime, dict(per_n.get("runtime", {})))
        )

        aim_cfg = build_aim_config(base_aim, stage_default="train")
        if aim_cfg is not None:
            aim_cfg.tags = [*aim_cfg.tags, f"n:{int(n)}", "stage:train"]

        graph_cfg_n = deep_merge(graph_cfg, dict(per_n.get("graph", {})))
        graph_dtype = parse_torch_dtype(graph_cfg_n.get("dtype", "int8"))

        job = TrainJobConfig(
            n=int(n),
            model_name=model_name,
            model_config=model_cfg,
            output_dir=output_root / f"n{int(n)}",
            model_artifacts_dir=artifacts_root,
            data=data_cfg,
            optimization=opt_cfg,
            lipschitz=lip_cfg,
            runtime=runtime_cfg,
            aim=aim_cfg,
            graph_batch_size=int(graph_cfg_n.get("batch_size", 2**17)),
            graph_dtype=graph_dtype,
            device=graph_cfg_n.get("device"),
            fit_verbose=bool(train_cfg.get("fit_verbose", True)),
        )
        jobs.append(job)

    return jobs


def build_inference_config_from_config(
    config: Mapping[str, Any],
) -> KaggleInferenceConfig | None:
    """
    Build Kaggle inference config from loaded run config.

    Args:
        config: Full run configuration mapping.

    Returns:
        Inference config or ``None`` when inference is disabled.

    """
    inf_raw = config.get("inference")
    if inf_raw is None:
        return None
    if not isinstance(inf_raw, Mapping):
        raise TypeError("config['inference'] must be a mapping.")
    inf = dict(inf_raw)
    if not bool(inf.get("enabled", True)):
        return None

    test_csv = inf.get("test_csv_path")
    if test_csv is None:
        raise ValueError("config['inference']['test_csv_path'] is required.")

    target_n = {int(x) for x in inf.get("target_n", [])}
    if not target_n:
        raise ValueError("config['inference']['target_n'] must not be empty.")

    aim_cfg = build_aim_config(inf.get("aim"), stage_default="beam_eval")

    return KaggleInferenceConfig(
        test_csv_path=Path(test_csv),
        target_n=target_n,
        base_submission_pkl=(
            Path(inf["base_submission_pkl"]) if inf.get("base_submission_pkl") else None
        ),
        submission_rows_out=Path(
            inf.get("submission_rows_out", "submission_rows_run.pkl")
        ),
        beam_width=int(inf.get("beam_width", 2**10)),
        history_depth=int(inf.get("history_depth", 0)),
        max_steps_factor=int(inf.get("max_steps_factor", 2)),
        require_not_longer_than_previous=bool(
            inf.get("require_not_longer_than_previous", True)
        ),
        compile_once_per_n=bool(inf.get("compile_once_per_n", True)),
        free_graph_memory=bool(inf.get("free_graph_memory", True)),
        clear_cuda_cache=bool(inf.get("clear_cuda_cache", True)),
        run_gc_collect=bool(inf.get("run_gc_collect", True)),
        checkpoint_path=(
            Path(inf["checkpoint_path"]) if inf.get("checkpoint_path") else None
        ),
        checkpoint_every_attempts=int(inf.get("checkpoint_every_attempts", 1)),
        enable_autocast=bool(inf.get("enable_autocast", True)),
        autocast_dtype=parse_torch_dtype(inf.get("autocast_dtype", "bfloat16")),
        aim=aim_cfg,
    )


def build_aim_config(
    aim_raw: Any,
    *,
    stage_default: str,
) -> AimRunConfig | None:
    """
    Build ``AimRunConfig`` from optional mapping.

    Args:
        aim_raw: Raw mapping from YAML.
        stage_default: Fallback stage label.

    Returns:
        Parsed ``AimRunConfig`` or ``None``.

    """
    if aim_raw is None:
        return None
    if not isinstance(aim_raw, Mapping):
        raise TypeError("Aim config must be a mapping.")
    aim = dict(aim_raw)

    experiment = aim.get("experiment")
    if experiment is None:
        raise ValueError("Aim config requires 'experiment'.")

    repo = Path(aim["repo"]) if aim.get("repo") else None
    tags = [str(tag) for tag in aim.get("tags", [])]
    stage = str(aim.get("stage", stage_default))
    notebook = aim.get("notebook")
    model_name = aim.get("model_name")
    extra_meta = dict(aim.get("extra_meta", {}))

    return AimRunConfig(
        experiment=str(experiment),
        repo=repo,
        tags=tags,
        stage=stage,
        notebook=str(notebook) if notebook is not None else None,
        model_name=str(model_name) if model_name is not None else None,
        extra_meta=extra_meta,
    )


def parse_torch_dtype(value: Any) -> torch.dtype:
    """
    Parse torch dtype from config string.

    Args:
        value: Dtype name or ``torch.dtype``.

    Returns:
        Parsed torch dtype.

    """
    if isinstance(value, torch.dtype):
        return value
    text = str(value).strip().lower().replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
    }
    dtype = mapping.get(text)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {value!r}.")
    return dtype


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two mappings.

    Args:
        base: Base mapping.
        update: Update mapping.

    Returns:
        Merged mapping copy.

    """
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged
