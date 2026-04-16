"""Lightning-first training and Kaggle inference helpers for Pilgrim models."""

from .config import (
    AimRunConfig,
    KaggleInferenceConfig,
    LightningRuntimeConfig,
    LipschitzConfig,
    OptimizationConfig,
    RandomWalkDataConfig,
    TrainJobConfig,
)
from .yaml_config import (
    build_inference_config_from_config,
    build_train_jobs_from_config,
    load_yaml_config,
)

_TIMING_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from .algraphgpt_timing import (
        AlGraphGPTSizePreset,
        AlGraphGPTTimingBenchmarkConfig,
        AlGraphGPTTimingBenchmarkResult,
        build_timing_benchmark_config,
        run_algraphgpt_timing_benchmark,
        run_algraphgpt_timing_from_yaml,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
    _TIMING_IMPORT_ERROR = exc

    def _raise_timing_import_error(*args, **kwargs):
        raise ModuleNotFoundError(
            "pip install lightning and torch profiler dependencies to use "
            "AlGraphGPT timing benchmark APIs."
        ) from _TIMING_IMPORT_ERROR

    AlGraphGPTSizePreset = None  # type: ignore[assignment]
    AlGraphGPTTimingBenchmarkConfig = None  # type: ignore[assignment]
    AlGraphGPTTimingBenchmarkResult = None  # type: ignore[assignment]
    build_timing_benchmark_config = _raise_timing_import_error  # type: ignore[assignment]
    run_algraphgpt_timing_benchmark = _raise_timing_import_error  # type: ignore[assignment]
    run_algraphgpt_timing_from_yaml = _raise_timing_import_error  # type: ignore[assignment]

_PIPELINE_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from .pipeline import (
        KaggleInferenceResult,
        TrainingResult,
        run_from_config,
        run_kaggle_inference,
        train_jobs,
        train_one_job,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
    _PIPELINE_IMPORT_ERROR = exc

    def _raise_pipeline_import_error(*args, **kwargs):
        raise ModuleNotFoundError(
            "pip install lightning and aim to use training/inference pipeline APIs."
        ) from _PIPELINE_IMPORT_ERROR

    KaggleInferenceResult = None  # type: ignore[assignment]
    TrainingResult = None  # type: ignore[assignment]
    run_from_config = _raise_pipeline_import_error  # type: ignore[assignment]
    run_kaggle_inference = _raise_pipeline_import_error  # type: ignore[assignment]
    train_jobs = _raise_pipeline_import_error  # type: ignore[assignment]
    train_one_job = _raise_pipeline_import_error  # type: ignore[assignment]

__all__ = [
    "AimRunConfig",
    "AlGraphGPTSizePreset",
    "AlGraphGPTTimingBenchmarkConfig",
    "AlGraphGPTTimingBenchmarkResult",
    "KaggleInferenceConfig",
    "KaggleInferenceResult",
    "LightningRuntimeConfig",
    "LipschitzConfig",
    "OptimizationConfig",
    "RandomWalkDataConfig",
    "TrainJobConfig",
    "TrainingResult",
    "build_inference_config_from_config",
    "build_timing_benchmark_config",
    "build_train_jobs_from_config",
    "load_yaml_config",
    "run_algraphgpt_timing_benchmark",
    "run_algraphgpt_timing_from_yaml",
    "run_from_config",
    "run_kaggle_inference",
    "train_jobs",
    "train_one_job",
]
