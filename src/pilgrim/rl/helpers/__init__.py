# Shared helper exports for reinforcement-learning trainers and trackers.
"""Helper functions for value and Q-learning in the RL package."""

from __future__ import annotations

from .ppo import (
    PolicyActionEvaluation,
    PolicyValueOutput,
    compute_lipschitz_actor_critic_loss,
    compute_supervised_policy_value_losses,
    evaluate_policy_actions,
    forward_policy_value,
    greedy_policy_actions,
    normalize_advantages,
    sample_policy_actions,
)
from .q_learning import (
    apply_actions,
    compute_double_q_targets_from_transition_batch,
    compute_n_step_double_q_targets,
    evaluate_q_values,
    evaluate_selected_q_values,
    greedy_actions_from_q,
    greedy_rollout_from_q,
    model_output_dim,
    predict_state_scores,
    sample_behavior_actions,
    sample_n_step_transitions_from_random_walks,
    sample_n_step_transitions_from_states,
    state_values_from_q,
)
from .search_guidance import (
    BeamSearchTargetSet,
    BeamSearchTargetStats,
    beam_action_reward_bonus,
    collect_beam_search_targets,
)
from .trajectory_supervision import (
    ReverseTrajectorySupervisionBatch,
    concatenate_reverse_trajectory_batches,
    sample_reverse_trajectory_supervision_from_random_walks,
    subsample_reverse_trajectory_batch,
)

__all__ = [
    "BeamSearchTargetSet",
    "BeamSearchTargetStats",
    "PolicyActionEvaluation",
    "PolicyValueOutput",
    "ReverseTrajectorySupervisionBatch",
    "apply_actions",
    "beam_action_reward_bonus",
    "collect_beam_search_targets",
    "compute_double_q_targets_from_transition_batch",
    "compute_lipschitz_actor_critic_loss",
    "compute_n_step_double_q_targets",
    "compute_supervised_policy_value_losses",
    "concatenate_reverse_trajectory_batches",
    "evaluate_policy_actions",
    "evaluate_q_values",
    "evaluate_selected_q_values",
    "forward_policy_value",
    "greedy_actions_from_q",
    "greedy_policy_actions",
    "greedy_rollout_from_q",
    "model_output_dim",
    "normalize_advantages",
    "predict_state_scores",
    "sample_behavior_actions",
    "sample_n_step_transitions_from_random_walks",
    "sample_n_step_transitions_from_states",
    "sample_policy_actions",
    "sample_reverse_trajectory_supervision_from_random_walks",
    "state_values_from_q",
    "subsample_reverse_trajectory_batch",
]
