# Exports the reinforcement-learning helpers for shortest-path training.
"""Reinforcement-learning utilities for Pilgrim shortest-path models."""

from __future__ import annotations

from ..schemas.rl import (
    DistributedMultiStepTDRunSpec,
    MultiStepTDValueConfig,
    MultiStepTDValueLossState,
    MultiStepTDValueMetrics,
    MultiStepTDValueStepDiagnostics,
    TDFileTrackerConfig,
    TDFrontierArchiveConfig,
    TDLearningRateSchedulerConfig,
    TDLipschitzPenaltyConfig,
    TDParallelConfig,
    TDProbeEvaluationConfig,
    TDRandomWalkSamplingConfig,
    TDReplayBufferConfig,
    TDSecondaryGpuEvalConfig,
)
from .composite_tracking import CompositeMultiStepTDValueTracker
from .config import (
    FittedValueIterationConfig,
    FrontierArchiveConfig,
    LipschitzPenaltyConfig,
    RandomWalkSamplingConfig,
    ReplayBufferConfig,
)
from .distributed import (
    cpu_model_state_dict,
    distributed_rank,
    distributed_world_size,
    is_distributed_initialized,
    is_main_process,
    load_model_state_dict,
    local_rank_from_env,
    split_evenly,
    synchronized_barrier,
    unwrap_model,
    wrap_model_for_ddp,
)
from .file_tracking import TDFileMetricsTracker
from .fitted_value_iteration import (
    FittedValueIterationLossState,
    FittedValueIterationMetrics,
    FittedValueIterationStepDiagnostics,
    FittedValueIterationTracker,
    FittedValueIterationTrainer,
)
from .multistep_td_tracking import MultiStepTDValueTracker
from .multistep_td_value_iteration import MultiStepTDValueTrainer
from .policies import greedy_actions_from_value, greedy_rollout_from_value
from .replay import FrontierStateArchive, TensorReplayBuffer
from .transitions import (
    combine_truncated_td_lambda_targets,
    compute_bellman_value_targets,
    compute_n_step_value_target_sequence,
    compute_n_step_value_targets,
    compute_td_lambda_value_targets,
    enumerate_neighbor_states,
)

__all__ = [
    "CompositeMultiStepTDValueTracker",
    "DistributedMultiStepTDRunSpec",
    "FittedValueIterationConfig",
    "FittedValueIterationLossState",
    "FittedValueIterationMetrics",
    "FittedValueIterationStepDiagnostics",
    "FittedValueIterationTracker",
    "FittedValueIterationTrainer",
    "FrontierArchiveConfig",
    "FrontierStateArchive",
    "LipschitzPenaltyConfig",
    "MultiStepTDValueConfig",
    "MultiStepTDValueLossState",
    "MultiStepTDValueMetrics",
    "MultiStepTDValueStepDiagnostics",
    "MultiStepTDValueTracker",
    "MultiStepTDValueTrainer",
    "RandomWalkSamplingConfig",
    "ReplayBufferConfig",
    "TDFileMetricsTracker",
    "TDFileTrackerConfig",
    "TDFrontierArchiveConfig",
    "TDLearningRateSchedulerConfig",
    "TDLipschitzPenaltyConfig",
    "TDParallelConfig",
    "TDProbeEvaluationConfig",
    "TDRandomWalkSamplingConfig",
    "TDReplayBufferConfig",
    "TDSecondaryGpuEvalConfig",
    "TensorReplayBuffer",
    "combine_truncated_td_lambda_targets",
    "compute_bellman_value_targets",
    "compute_n_step_value_target_sequence",
    "compute_n_step_value_targets",
    "compute_td_lambda_value_targets",
    "cpu_model_state_dict",
    "distributed_rank",
    "distributed_world_size",
    "enumerate_neighbor_states",
    "greedy_actions_from_value",
    "greedy_rollout_from_value",
    "is_distributed_initialized",
    "is_main_process",
    "load_model_state_dict",
    "local_rank_from_env",
    "split_evenly",
    "synchronized_barrier",
    "unwrap_model",
    "wrap_model_for_ddp",
]
