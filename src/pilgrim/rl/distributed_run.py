# Runs distributed multi-step TD training jobs from a serialized run spec.
"""CLI entrypoint and helpers for distributed multi-step TD training."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from pilgrim.model import AlPilgrim
from pilgrim.schemas.rl import DistributedMultiStepTDRunSpec
from pilgrim.utils.model_io import save_one
from pilgrim.utils.pancake_utils import make_graph_for_n

from .distributed import (
    distributed_rank,
    is_distributed_initialized,
    is_main_process,
    local_rank_from_env,
    synchronized_barrier,
)
from .file_tracking import TDFileMetricsTracker
from .multistep_td_value_iteration import MultiStepTDValueTrainer
from .policies import greedy_rollout_from_value


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the distributed run entrypoint.

    Returns:
        Parsed CLI namespace.

    """
    parser = argparse.ArgumentParser(
        description="Run distributed multi-step TD value training from a JSON spec."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to a JSON file containing DistributedMultiStepTDRunSpec.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch.

    Args:
        seed: Base random seed.

    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    random.seed(int(seed))


def initialize_distributed(spec: DistributedMultiStepTDRunSpec) -> None:
    """
    Initialize ``torch.distributed`` when the run spec requests DDP.

    Args:
        spec: Serialized run specification.

    """
    if not spec.trainer_config.parallel.uses_ddp:
        return
    local_rank = int(local_rank_from_env())
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not is_distributed_initialized():
        dist.init_process_group(backend=str(spec.trainer_config.parallel.backend))


def destroy_distributed() -> None:
    """Destroy the distributed process group when initialized."""
    if is_distributed_initialized():
        dist.destroy_process_group()


def worker_device(spec: DistributedMultiStepTDRunSpec) -> torch.device:
    """
    Resolve the device used by the active worker process.

    Args:
        spec: Serialized run specification.

    Returns:
        Worker-local torch device.

    """
    if torch.cuda.is_available():
        if spec.trainer_config.parallel.uses_ddp:
            return torch.device(f"cuda:{int(local_rank_from_env())}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_graph(
    spec: DistributedMultiStepTDRunSpec,
    device: torch.device,
) -> Any:
    """
    Build the worker-local Cayley graph on the requested device.

    Args:
        spec: Serialized run specification.
        device: Worker-local torch device.

    Returns:
        Cayley graph bound to the worker device.

    """
    graph_device_arg = "cuda" if device.type == "cuda" else "cpu"
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        return make_graph_for_n(
            int(spec.n),
            batch_size=int(spec.graph_batch_size),
            device=graph_device_arg,
        )


def runtime_model_kwargs(spec: DistributedMultiStepTDRunSpec) -> dict[str, Any]:
    """
    Return model kwargs normalized for runtime construction.

    Args:
        spec: Serialized run specification.

    Returns:
        Copy of ``spec.model_kwargs`` with string dtypes resolved.

    """
    config = dict(spec.model_kwargs)
    for key in ("dtype", "model_dtype"):
        if key in config:
            config[key] = resolve_torch_dtype(config[key])
    return config


def resolve_torch_dtype(value: Any) -> Any:
    """
    Resolve a serialized dtype value into ``torch.dtype`` when possible.

    Args:
        value: Candidate dtype value from JSON or Python.

    Returns:
        Resolved ``torch.dtype`` or the original value when no mapping exists.

    """
    if not isinstance(value, str):
        return value
    normalized = value.removeprefix("torch.").strip().lower()
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return mapping.get(normalized, value)


def build_probe_batch(
    spec: DistributedMultiStepTDRunSpec,
    graph: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Build the fixed probe batch used by the rank-zero tracker.

    Args:
        spec: Serialized run specification.
        graph: Active Cayley graph instance.

    Returns:
        Tuple of ``(probe_states, probe_targets)`` or ``(None, None)`` when
        probes are disabled.

    """
    if int(spec.probe_count) <= 0:
        return None, None

    probe_seed = int(spec.seed) + 1
    seed_everything(probe_seed)
    walk_length = (
        int(spec.probe_walk_length)
        if spec.probe_walk_length is not None
        else int(spec.trainer_config.sampling.rw_length)
    )
    non_center_count = max(0, int(spec.probe_count) - 1)
    width = max(1, int(np.ceil(non_center_count / max(1, walk_length))))
    sampled_states, sampled_targets = graph.random_walks(
        width=width,
        length=walk_length,
        mode="nbt",
        nbt_history_depth=walk_length,
    )

    perm = torch.randperm(sampled_states.shape[0])
    sampled_states = sampled_states[perm]
    sampled_targets = sampled_targets[perm]

    center_state = torch.as_tensor(graph.central_state).view(1, -1).long().cpu()
    center_target = torch.zeros(1, dtype=torch.float32)
    sampled_states = torch.as_tensor(sampled_states).long().cpu()[:non_center_count]
    sampled_targets = torch.as_tensor(sampled_targets).float().cpu()[:non_center_count]
    probe_states = torch.cat([center_state, sampled_states], dim=0)[: spec.probe_count]
    probe_targets = torch.cat([center_target, sampled_targets], dim=0)[
        : spec.probe_count
    ]
    return probe_states, probe_targets


def maybe_load_initial_checkpoint(model: torch.nn.Module, path: Path | None) -> None:
    """
    Load an optional initial checkpoint into a model.

    Args:
        model: Model to initialize.
        path: Optional checkpoint path created by ``save_one``.

    """
    if path is None:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def build_tracker(
    spec: DistributedMultiStepTDRunSpec,
    graph: Any,
) -> TDFileMetricsTracker | None:
    """
    Build the rank-zero file tracker when enabled.

    Args:
        spec: Serialized run specification.
        graph: Active Cayley graph instance.

    Returns:
        File tracker instance or ``None``.

    """
    if not is_main_process() or spec.file_tracker is None:
        return None
    probe_states, probe_targets = build_probe_batch(spec, graph)
    hparams = dict(spec.hparams)
    hparams.update(
        {
            "n": int(spec.n),
            "model_config": spec.model_kwargs,
            "rl_config": spec.trainer_config.to_log_dict(),
            "profile": spec.profile,
            "notebook_name": spec.notebook_name,
        }
    )
    return TDFileMetricsTracker(
        spec.file_tracker,
        graph,
        hparams=hparams,
        group_n=int(spec.n),
        probe_states=probe_states,
        probe_targets=probe_targets,
    )


def _model_filename_template(spec: DistributedMultiStepTDRunSpec) -> str:
    """
    Return the checkpoint filename template used by ``save_one``.

    Args:
        spec: Serialized run specification.

    Returns:
        Save template containing ``{n}``.

    """
    if spec.model_filename is None:
        return "model_n{n}.pt"
    return spec.model_filename.replace(str(spec.n), "{n}")


def _evaluate_rollout_summary(
    spec: DistributedMultiStepTDRunSpec,
    trainer: MultiStepTDValueTrainer,
    graph: Any,
) -> tuple[float, int]:
    """
    Evaluate center prediction and one greedy rollout for the final summary.

    Args:
        spec: Serialized run specification.
        trainer: Trained multi-step TD trainer.
        graph: Active Cayley graph instance.
    Returns:
        Tuple ``(center_value, greedy_rollout_length)``.

    """
    trainer.model.eval()
    model_device = next(trainer.model.parameters(), None)
    resolved_model_device = (
        torch.device("cpu")
        if model_device is None
        else model_device.device
    )
    with torch.no_grad():
        center_state = torch.as_tensor(
            graph.central_state,
            device=resolved_model_device,
        ).view(1, -1)
        center_value = float(trainer.model(center_state).item())
        rollout_start = graph.random_walks(
            width=1,
            length=max(8, int(spec.n)),
            mode="nbt",
            nbt_history_depth=max(8, int(spec.n)),
        )[0][-1]
        rollout = greedy_rollout_from_value(
            trainer.model,
            graph,
            rollout_start,
            max_steps=int(spec.n) * int(spec.n),
            value_batch_size=spec.trainer_config.value_batch_size,
        )
    return center_value, len(rollout)


def _build_summary(
    spec: DistributedMultiStepTDRunSpec,
    *,
    device: torch.device,
    history_df: pd.DataFrame,
    model_path: Path,
    history_path: Path,
    center_value: float,
    greedy_rollout_length: int,
) -> dict[str, Any]:
    """
    Build the rank-zero notebook summary payload.

    Args:
        spec: Serialized run specification.
        device: Rank-zero device.
        history_df: DataFrame built from fit history.
        model_path: Written checkpoint path.
        history_path: Written history CSV path.
        center_value: Final predicted value of the center state.
        greedy_rollout_length: Greedy rollout length from a random start.

    Returns:
        Summary mapping written next to the checkpoint.

    """
    tracker_dir = None if spec.file_tracker is None else Path(spec.file_tracker.output_dir)
    jsonl_name = None if spec.file_tracker is None else spec.file_tracker.jsonl_name
    csv_name = None if spec.file_tracker is None else spec.file_tracker.csv_name
    tracker_summary_name = (
        None if spec.file_tracker is None else spec.file_tracker.summary_name
    )
    return {
        "n": int(spec.n),
        "device": str(device),
        "rank": int(distributed_rank()),
        "parallel_mode": str(spec.trainer_config.parallel.resolved_mode),
        "parallel_num_gpus": int(spec.trainer_config.parallel.num_gpus),
        "available_cuda_devices": int(torch.cuda.device_count()),
        "probe_count": int(spec.probe_count),
        "center_value": center_value,
        "history_rows": len(history_df),
        "last_total_loss": (
            None if history_df.empty else float(history_df.iloc[-1]["total_loss"])
        ),
        "last_td_loss": (
            None if history_df.empty else float(history_df.iloc[-1]["td_loss"])
        ),
        "greedy_rollout_length": int(greedy_rollout_length),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "tracker_dir": None if tracker_dir is None else str(tracker_dir),
        "step_log_jsonl": (
            None if tracker_dir is None or jsonl_name is None else str(tracker_dir / jsonl_name)
        ),
        "step_log_csv": (
            None if tracker_dir is None or csv_name is None else str(tracker_dir / csv_name)
        ),
        "tracker_summary": (
            None
            if tracker_dir is None or tracker_summary_name is None
            else str(tracker_dir / tracker_summary_name)
        ),
    }


def run_training(spec: DistributedMultiStepTDRunSpec) -> dict[str, Any]:
    """
    Run one distributed multi-step TD training job.

    Args:
        spec: Serialized run specification.

    Returns:
        Rank-zero summary dictionary, or an empty dictionary on nonzero ranks.

    """
    device = worker_device(spec)
    seed_everything(int(spec.seed))
    graph = build_graph(spec, device)
    model_kwargs = runtime_model_kwargs(spec)
    model = AlPilgrim(model_kwargs).to(device)
    maybe_load_initial_checkpoint(model, spec.initial_checkpoint_path)
    tracker = build_tracker(spec, graph)
    trainer = MultiStepTDValueTrainer(
        model=model,
        graph=graph,
        config=spec.trainer_config,
        tracker=tracker,
    )
    history = trainer.fit() if spec.enable_training else []

    if not is_main_process():
        return {}

    run_dir = Path(spec.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame([metric.model_dump() for metric in history])
    history_path = run_dir / spec.history_name
    history_df.to_csv(history_path, index=False)

    model_path = save_one(
        int(spec.n),
        trainer.model,
        model_kwargs,
        run_dir,
        filename_template=_model_filename_template(spec),
    )
    center_value, greedy_rollout_length = _evaluate_rollout_summary(
        spec,
        trainer,
        graph,
    )
    summary = _build_summary(
        spec,
        device=device,
        history_df=history_df,
        model_path=model_path,
        history_path=history_path,
        center_value=center_value,
        greedy_rollout_length=greedy_rollout_length,
    )
    summary_path = run_dir / spec.summary_name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run the distributed training CLI."""
    args = parse_args()
    spec = DistributedMultiStepTDRunSpec.model_validate_json(
        Path(args.spec).read_text(encoding="utf-8")
    )
    torch.set_float32_matmul_precision("high")
    initialize_distributed(spec)
    try:
        run_training(spec)
        synchronized_barrier()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
