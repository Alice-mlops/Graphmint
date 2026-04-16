# Implements target-evaluation backends for RL value-learning trainers.
"""Backends for frozen target-value evaluation in RL training."""

from __future__ import annotations

import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from .transitions import compute_configured_value_targets

if TYPE_CHECKING:
    from cayleypy import CayleyGraph

    from pilgrim.schemas.rl import MultiStepTDValueConfig

_EXPECTED_STATE_ROWS_NDIM = 2


@dataclass(slots=True, frozen=True)
class TDTargetEvaluationSettings:
    """
    Immutable target-construction settings shared by evaluator backends.

    Args:
        reward_per_step: Step reward or cost added during backup construction.
        discount: Discount factor applied to future targets.
        n_steps: Maximum TD backup horizon.
        td_lambda: Optional TD-lambda coefficient.
        terminal_value: Value assigned to the center state.
        generator_indices: Optional subset of generators used in targets.
        value_batch_size: Optional chunk size for frozen-model evaluation.

    """

    reward_per_step: float
    discount: float
    n_steps: int
    td_lambda: float | None
    terminal_value: float
    generator_indices: tuple[int, ...] | None
    value_batch_size: int | None

    @classmethod
    def from_config(
        cls,
        config: MultiStepTDValueConfig,
    ) -> TDTargetEvaluationSettings:
        """
        Build immutable settings from trainer config.

        Args:
            config: Multi-step TD trainer configuration.

        Returns:
            Frozen evaluation settings.

        """
        return cls(
            reward_per_step=float(config.reward_per_step),
            discount=float(config.discount),
            n_steps=int(config.n_steps),
            td_lambda=(None if config.td_lambda is None else float(config.td_lambda)),
            terminal_value=float(config.terminal_value),
            generator_indices=(
                None
                if config.generator_indices is None
                else tuple(int(index) for index in config.generator_indices)
            ),
            value_batch_size=(
                None
                if config.value_batch_size is None
                else int(config.value_batch_size)
            ),
        )


class TDTargetEvaluationBackend(Protocol):
    """Protocol for trainer target-evaluation backends."""

    def compute_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute TD targets for a batch of states.

        Args:
            states: Input state rows.

        Returns:
            One-dimensional tensor of TD targets on the caller device.

        """

    def sync_target_model(self, target_model: nn.Module) -> None:
        """
        Synchronize backend replicas with the learner target model.

        Args:
            target_model: Learner-owned target model.

        """

    def close(self) -> None:
        """Release any backend-owned resources."""


class LocalTDTargetEvaluationBackend:
    """Local backend that evaluates TD targets on the learner device."""

    def __init__(
        self,
        *,
        target_model: nn.Module,
        graph: CayleyGraph,
        settings: TDTargetEvaluationSettings,
    ) -> None:
        self.target_model = target_model
        self.graph = graph
        self.settings = settings

    def compute_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute TD targets on the learner device.

        Args:
            states: State rows to score with the frozen target model.

        Returns:
            One-dimensional TD target tensor on the learner device.

        """
        return compute_configured_value_targets(
            target_model=self.target_model,
            graph=self.graph,
            states=states,
            reward_per_step=self.settings.reward_per_step,
            discount=self.settings.discount,
            n_steps=self.settings.n_steps,
            td_lambda=self.settings.td_lambda,
            terminal_value=self.settings.terminal_value,
            generator_indices=self.settings.generator_indices,
            value_batch_size=self.settings.value_batch_size,
        )

    def sync_target_model(self, target_model: nn.Module) -> None:
        """Refresh the local target-model reference."""
        self.target_model = target_model

    def close(self) -> None:
        """Release no-op local backend resources."""


@dataclass(slots=True)
class _EvaluatorReplica:
    """
    Frozen target-evaluation replica bound to one secondary GPU.

    Args:
        device: Replica device.
        graph: Graph replica on ``device``.
        target_model: Frozen target model on ``device``.

    """

    device: torch.device
    graph: CayleyGraph
    target_model: nn.Module

    def compute_targets(
        self,
        states: torch.Tensor,
        settings: TDTargetEvaluationSettings,
    ) -> torch.Tensor:
        """
        Compute TD targets for a shard of states on the replica device.

        Args:
            states: Input state rows.
            settings: Immutable target-construction settings.

        Returns:
            One-dimensional tensor of TD targets on ``states.device``.

        """
        return compute_configured_value_targets(
            target_model=self.target_model,
            graph=self.graph,
            states=states,
            reward_per_step=settings.reward_per_step,
            discount=settings.discount,
            n_steps=settings.n_steps,
            td_lambda=settings.td_lambda,
            terminal_value=settings.terminal_value,
            generator_indices=settings.generator_indices,
            value_batch_size=settings.value_batch_size,
        )

    def sync_target_model(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load a refreshed learner target-model state into the replica.

        Args:
            state_dict: CPU-cloned state dictionary from the learner target.

        """
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()


class SecondaryGpuTDTargetEvaluationBackend:
    """
    Synchronous evaluator pool that uses secondary GPUs for TD targets.

    The learner remains on the primary GPU. This backend creates one frozen
    target-model replica per secondary GPU and shards target-evaluation work
    across them with a thread pool.

    Args:
        target_model: Learner-owned frozen target model.
        graph: Learner-owned graph instance.
        settings: Immutable target-construction settings.
        num_gpus: Total number of GPUs reserved for the trainer.

    Raises:
        ValueError: If CUDA requirements are not met.

    """

    def __init__(
        self,
        *,
        target_model: nn.Module,
        graph: CayleyGraph,
        settings: TDTargetEvaluationSettings,
        num_gpus: int,
    ) -> None:
        if int(num_gpus) <= 1:
            raise ValueError("secondary-GPU backend requires num_gpus > 1.")
        if not torch.cuda.is_available():
            raise ValueError("secondary-GPU backend requires CUDA.")
        if int(num_gpus) > int(torch.cuda.device_count()):
            raise ValueError(
                f"Requested {int(num_gpus)} GPUs, but only "
                f"{int(torch.cuda.device_count())} are available."
            )

        self.settings = settings
        self._replicas = self._build_replicas(
            target_model=target_model,
            graph=graph,
            num_gpus=int(num_gpus),
        )
        self._executor = ThreadPoolExecutor(max_workers=len(self._replicas))

    def compute_targets(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute TD targets by sharding rows across evaluator GPUs.

        Args:
            states: Input state rows on the learner or CPU device.

        Returns:
            One-dimensional tensor of TD targets on ``states.device``.

        """
        data = _normalize_state_rows(states)
        if data.shape[0] == 0:
            return torch.empty((0,), dtype=torch.float32, device=data.device)
        if len(self._replicas) == 1:
            return (
                self._replicas[0].compute_targets(data, self.settings).to(data.device)
            )

        shards = _split_state_rows(data, num_shards=len(self._replicas))
        futures = []
        for replica, shard in zip(self._replicas, shards, strict=True):
            if shard.shape[0] == 0:
                continue
            futures.append(
                self._executor.submit(replica.compute_targets, shard, self.settings)
            )

        outputs = [future.result() for future in futures]
        if not outputs:
            return torch.empty((0,), dtype=torch.float32, device=data.device)
        return torch.cat(outputs, dim=0).to(data.device)

    def sync_target_model(self, target_model: nn.Module) -> None:
        """
        Refresh all evaluator replicas from the learner target model.

        Args:
            target_model: Learner-owned frozen target model.

        """
        state_dict = _cpu_state_dict(target_model)
        for replica in self._replicas:
            replica.sync_target_model(state_dict)

    def close(self) -> None:
        """Release the evaluator thread pool."""
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __del__(self) -> None:
        """Best-effort cleanup for the evaluator thread pool."""
        with contextlib.suppress(Exception):
            self.close()

    @staticmethod
    def _build_replicas(
        *,
        target_model: nn.Module,
        graph: CayleyGraph,
        num_gpus: int,
    ) -> list[_EvaluatorReplica]:
        """
        Build one frozen target-evaluation replica per secondary GPU.

        Args:
            target_model: Learner-owned frozen target model.
            graph: Learner-owned graph instance.
            num_gpus: Total number of GPUs reserved for the trainer.

        Returns:
            List of per-device evaluator replicas.

        """
        replicas: list[_EvaluatorReplica] = []
        for device_index in range(1, int(num_gpus)):
            device = torch.device(f"cuda:{device_index}")
            replicas.append(
                _EvaluatorReplica(
                    device=device,
                    graph=_clone_graph_to_device(graph, device),
                    target_model=copy.deepcopy(target_model).to(device).eval(),
                )
            )
        return replicas


def build_td_target_evaluation_backend(
    *,
    target_model: nn.Module,
    graph: CayleyGraph,
    config: MultiStepTDValueConfig,
) -> TDTargetEvaluationBackend:
    """
    Construct the trainer target-evaluation backend from config.

    Args:
        target_model: Learner-owned frozen target model.
        graph: Learner-owned graph instance.
        config: Multi-step TD trainer configuration.

    Returns:
        Configured local or secondary-GPU backend.

    """
    settings = TDTargetEvaluationSettings.from_config(config)
    if not config.parallel.uses_secondary_gpus:
        return LocalTDTargetEvaluationBackend(
            target_model=target_model,
            graph=graph,
            settings=settings,
        )
    return SecondaryGpuTDTargetEvaluationBackend(
        target_model=target_model,
        graph=graph,
        settings=settings,
        num_gpus=int(config.parallel.num_gpus),
    )


def _clone_graph_to_device(
    graph: CayleyGraph,
    device: str | torch.device,
) -> CayleyGraph:
    """
    Clone or reconstruct a graph on a new device.

    Args:
        graph: Source graph.
        device: Destination device.

    Returns:
        Graph replica located on ``device``.

    """
    target_device = torch.device(device)
    if hasattr(graph, "modified_copy") and hasattr(graph, "definition"):
        return graph.modified_copy(graph.definition, device=target_device)

    graph_copy = copy.deepcopy(graph)
    if hasattr(graph_copy, "device"):
        graph_copy.device = target_device
    if hasattr(graph_copy, "central_state"):
        graph_copy.central_state = torch.as_tensor(
            graph_copy.central_state,
            device=target_device,
        ).clone()
    if hasattr(graph_copy, "permutations_torch"):
        graph_copy.permutations_torch = torch.as_tensor(
            graph_copy.permutations_torch,
            device=target_device,
        ).clone()
    return graph_copy


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Return a CPU-cloned copy of ``model.state_dict()``.

    Args:
        model: Model whose parameters should be cloned.

    Returns:
        CPU-cloned state dictionary.

    """
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _normalize_state_rows(states: torch.Tensor) -> torch.Tensor:
    """
    Normalize a state tensor to shape ``(batch, state_size)``.

    Args:
        states: Input state tensor.

    Returns:
        Two-dimensional long tensor.

    Raises:
        ValueError: If the normalized tensor is not rank two.

    """
    data = torch.as_tensor(states).long()
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.ndim != _EXPECTED_STATE_ROWS_NDIM:
        raise ValueError(
            "states must have shape (batch, state_size) or (state_size,), "
            f"got {tuple(data.shape)}."
        )
    return data.contiguous()


def _split_state_rows(
    states: torch.Tensor,
    *,
    num_shards: int,
) -> list[torch.Tensor]:
    """
    Split state rows into nearly equal contiguous shards.

    Args:
        states: State rows to shard.
        num_shards: Number of output shards.

    Returns:
        List of contiguous row shards in original order.

    Raises:
        ValueError: If ``num_shards`` is not positive.

    """
    if int(num_shards) <= 0:
        raise ValueError("num_shards must be positive.")
    total_rows = int(states.shape[0])
    base_size, remainder = divmod(total_rows, int(num_shards))
    shards: list[torch.Tensor] = []
    start = 0
    for shard_index in range(int(num_shards)):
        shard_size = base_size + int(shard_index < remainder)
        end = start + shard_size
        shards.append(states[start:end].contiguous())
        start = end
    return shards
