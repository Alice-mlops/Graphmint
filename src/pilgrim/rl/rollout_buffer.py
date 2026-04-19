# Stores on-policy rollout batches for policy-gradient training on Cayley graphs.
"""Rollout-buffer utilities for search-guided PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .replay import _normalize_1d_tensor, _normalize_state_rows


@dataclass(slots=True, frozen=True)
class PolicyRolloutBatch:
    """
    Flattened on-policy rollout batch for PPO updates.

    Args:
        states: Source states with shape ``(batch, state_size)``.
        actions: Sampled actions with shape ``(batch,)``.
        log_probs: Old action log-probabilities with shape ``(batch,)``.
        advantages: Generalized-advantage estimates with shape ``(batch,)``.
        returns: Bootstrapped return targets with shape ``(batch,)``.
        values: Old value predictions with shape ``(batch,)``.
        rewards: Per-step rewards with shape ``(batch,)``.
        done: Terminal indicators aligned with transitions.

    """

    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor

    def __post_init__(self) -> None:
        """
        Validate shape alignment across rollout tensors.

        Raises:
            ValueError: If one of the tensors is not batch-aligned.

        """
        state_device = torch.as_tensor(self.states).device
        batch_states = _normalize_state_rows(self.states, device=state_device)
        batch_actions = _normalize_1d_tensor(
            self.actions,
            device=state_device,
            dtype=torch.long,
            name="actions",
        )
        batch_log_probs = _normalize_1d_tensor(
            self.log_probs,
            device=state_device,
            dtype=torch.float32,
            name="log_probs",
        )
        batch_advantages = _normalize_1d_tensor(
            self.advantages,
            device=state_device,
            dtype=torch.float32,
            name="advantages",
        )
        batch_returns = _normalize_1d_tensor(
            self.returns,
            device=state_device,
            dtype=torch.float32,
            name="returns",
        )
        batch_values = _normalize_1d_tensor(
            self.values,
            device=state_device,
            dtype=torch.float32,
            name="values",
        )
        batch_rewards = _normalize_1d_tensor(
            self.rewards,
            device=state_device,
            dtype=torch.float32,
            name="rewards",
        )
        batch_done = _normalize_1d_tensor(
            self.done,
            device=state_device,
            dtype=torch.bool,
            name="done",
        )
        batch_size = int(batch_states.shape[0])
        for name, tensor in [
            ("actions", batch_actions),
            ("log_probs", batch_log_probs),
            ("advantages", batch_advantages),
            ("returns", batch_returns),
            ("values", batch_values),
            ("rewards", batch_rewards),
            ("done", batch_done),
        ]:
            if int(tensor.shape[0]) != batch_size:
                raise ValueError(f"{name} must align with states.")
        object.__setattr__(self, "states", batch_states)
        object.__setattr__(self, "actions", batch_actions)
        object.__setattr__(self, "log_probs", batch_log_probs)
        object.__setattr__(self, "advantages", batch_advantages)
        object.__setattr__(self, "returns", batch_returns)
        object.__setattr__(self, "values", batch_values)
        object.__setattr__(self, "rewards", batch_rewards)
        object.__setattr__(self, "done", batch_done)

    def __len__(self) -> int:
        """
        Return the number of valid rollout rows.

        Returns:
            Batch size.

        """
        return int(self.states.shape[0])

    def to(self, device: str | torch.device) -> PolicyRolloutBatch:
        """
        Move the rollout batch to ``device``.

        Args:
            device: Target device.

        Returns:
            Rollout batch on the target device.

        """
        target_device = torch.device(device)
        return PolicyRolloutBatch(
            states=self.states.to(target_device),
            actions=self.actions.to(target_device),
            log_probs=self.log_probs.to(target_device),
            advantages=self.advantages.to(target_device),
            returns=self.returns.to(target_device),
            values=self.values.to(target_device),
            rewards=self.rewards.to(target_device),
            done=self.done.to(target_device),
        )

    def index_select(self, indices: torch.Tensor) -> PolicyRolloutBatch:
        """
        Select a subset of rollout rows.

        Args:
            indices: One-dimensional row indices.

        Returns:
            New rollout batch containing the selected rows.

        """
        rows = torch.as_tensor(indices, device=self.states.device).long().reshape(-1)
        return PolicyRolloutBatch(
            states=self.states.index_select(0, rows),
            actions=self.actions.index_select(0, rows),
            log_probs=self.log_probs.index_select(0, rows),
            advantages=self.advantages.index_select(0, rows),
            returns=self.returns.index_select(0, rows),
            values=self.values.index_select(0, rows),
            rewards=self.rewards.index_select(0, rows),
            done=self.done.index_select(0, rows),
        )


class PolicyRolloutBuffer:
    """
    Temporary storage for one fixed-horizon PPO rollout.

    Args:
        num_envs: Number of parallel environments.
        horizon: Number of time steps collected per rollout.
        storage_device: Device used by stored tensors.

    Raises:
        ValueError: If one of the dimensions is not positive.

    """

    def __init__(
        self,
        *,
        num_envs: int,
        horizon: int,
        storage_device: str | torch.device = "cpu",
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive.")
        if int(horizon) <= 0:
            raise ValueError("horizon must be positive.")
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.storage_device = torch.device(storage_device)
        self.reset()

    def reset(self) -> None:
        """Clear all buffered rollout tensors."""
        self._states: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []
        self._done: list[torch.Tensor] = []
        self._valid_mask: list[torch.Tensor] = []

    def add_step(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """
        Append one rollout step with ``num_envs`` rows.

        Args:
            states: State tensor of shape ``(num_envs, state_size)``.
            actions: One-dimensional action tensor of shape ``(num_envs,)``.
            log_probs: Old log-probabilities with shape ``(num_envs,)``.
            values: Old value predictions with shape ``(num_envs,)``.
            rewards: One-dimensional reward tensor of shape ``(num_envs,)``.
            done: One-dimensional done tensor of shape ``(num_envs,)``.
            valid_mask: Boolean mask identifying active rollout rows.

        Raises:
            ValueError: If the rollout already contains ``horizon`` steps.

        """
        if len(self._states) >= self.horizon:
            raise ValueError("rollout buffer is already full.")
        self._states.append(
            _normalize_state_rows(states, device=self.storage_device).contiguous()
        )
        for destination, tensor, dtype, name in [
            (self._actions, actions, torch.long, "actions"),
            (self._log_probs, log_probs, torch.float32, "log_probs"),
            (self._values, values, torch.float32, "values"),
            (self._rewards, rewards, torch.float32, "rewards"),
            (self._done, done, torch.bool, "done"),
            (self._valid_mask, valid_mask, torch.bool, "valid_mask"),
        ]:
            normalized = _normalize_1d_tensor(
                tensor,
                device=self.storage_device,
                dtype=dtype,
                name=name,
            )
            if int(normalized.shape[0]) != self.num_envs:
                raise ValueError(f"{name} must have shape ({self.num_envs},).")
            destination.append(normalized.contiguous())

    def finalize(  # noqa: PLR0914
        self,
        *,
        last_values: torch.Tensor,
        discount: float,
        gae_lambda: float,
    ) -> PolicyRolloutBatch:
        """
        Convert the buffered rollout into a flattened PPO batch.

        Args:
            last_values: Bootstrap values for the states after the final step.
            discount: Discount factor used in GAE.
            gae_lambda: GAE smoothing coefficient.

        Returns:
            Flattened rollout batch containing only valid rows.

        Raises:
            RuntimeError: If the buffer is empty.
            ValueError: If ``last_values`` does not align with ``num_envs``.

        """
        if not self._states:
            raise RuntimeError("cannot finalize an empty rollout buffer.")

        states = torch.stack(self._states, dim=0)
        actions = torch.stack(self._actions, dim=0)
        log_probs = torch.stack(self._log_probs, dim=0)
        values = torch.stack(self._values, dim=0)
        rewards = torch.stack(self._rewards, dim=0)
        done = torch.stack(self._done, dim=0)
        valid_mask = torch.stack(self._valid_mask, dim=0)

        bootstrap = _normalize_1d_tensor(
            last_values,
            device=self.storage_device,
            dtype=torch.float32,
            name="last_values",
        )
        if int(bootstrap.shape[0]) != self.num_envs:
            raise ValueError("last_values must align with num_envs.")

        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(bootstrap)
        next_values = bootstrap

        for step_index in range(rewards.shape[0] - 1, -1, -1):
            not_done = (~done[step_index]).float()
            delta = rewards[step_index] + float(discount) * next_values * not_done
            delta -= values[step_index]
            last_advantage = delta + (
                float(discount) * float(gae_lambda) * not_done * last_advantage
            )
            advantages[step_index] = last_advantage
            next_values = values[step_index]

        returns = advantages + values
        flat_valid = valid_mask.reshape(-1)
        flat_indices = torch.nonzero(flat_valid, as_tuple=False).reshape(-1)
        return PolicyRolloutBatch(
            states=states.reshape(-1, states.shape[-1]).index_select(0, flat_indices),
            actions=actions.reshape(-1).index_select(0, flat_indices),
            log_probs=log_probs.reshape(-1).index_select(0, flat_indices),
            advantages=advantages.reshape(-1).index_select(0, flat_indices),
            returns=returns.reshape(-1).index_select(0, flat_indices),
            values=values.reshape(-1).index_select(0, flat_indices),
            rewards=rewards.reshape(-1).index_select(0, flat_indices),
            done=done.reshape(-1).index_select(0, flat_indices),
        )
