# Shares configurable learning-rate scheduler helpers across training loops.
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import torch


def step_lr_scheduler(scheduler: Any, metric: float | None = None) -> None:
    """Advance a learning-rate scheduler by one step.

    Args:
        scheduler: Scheduler instance to step. ``None`` is a no-op.
        metric: Optional scalar metric used by plateau schedulers.

    Raises:
        ValueError: If a plateau scheduler is stepped without a metric.

    """
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            raise ValueError(
                "ReduceLROnPlateau requires a metric when stepping the scheduler."
            )
        scheduler.step(metric)
        return
    scheduler.step()


def lr_scheduler_ctor_from_cfg(
    cfg: Mapping[str, Any],
    *,
    total_steps_key: str = "num_epochs",
    allow_plateau: bool = True,
) -> Callable[[torch.optim.Optimizer], Any] | None:
    """Build an LR-scheduler constructor from a config mapping.

    Args:
        cfg: Config mapping containing ``lr_scheduler`` and a total-step count.
        total_steps_key: Key used to resolve the total number of scheduler steps.
        allow_plateau: Whether ``ReduceLROnPlateau`` is permitted.

    Returns:
        Constructor that accepts an optimizer and returns a scheduler, or
        ``None`` when scheduling is disabled.

    Raises:
        TypeError: If ``cfg["lr_scheduler"]`` is not string-like or mapping-like.
        ValueError: If the scheduler type is unsupported or disallowed.

    """
    spec = cfg.get("lr_scheduler", None)
    if spec is None:
        return None

    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, Mapping):
        raise TypeError(
            'cfg["lr_scheduler"] must be a string, dict-like mapping, or None.'
        )

    scheduler_type = str(spec.get("type", "none")).strip().lower()
    if scheduler_type in {"none", "null", "off", ""}:
        return None

    total_steps = int(
        cfg.get(
            total_steps_key,
            cfg.get("num_epochs", cfg.get("num_updates", 1)),
        )
    )
    total_steps = max(1, total_steps)

    if scheduler_type in {"plateau", "reduce_on_plateau", "reduce_lr_on_plateau"}:
        if not allow_plateau:
            raise ValueError(
                "ReduceLROnPlateau is not supported for this training loop."
            )
        mode: Literal["min", "max"] = cast(
            Literal["min", "max"], str(spec.get("mode", "min"))
        )
        factor_value = spec.get("factor", 0.5)
        factor = float(0.5 if factor_value is None else factor_value)
        patience_value = spec.get("patience", 5)
        patience = int(5 if patience_value is None else patience_value)
        min_lr_value = spec.get("min_lr", 5e-7)
        min_lr = float(5e-7 if min_lr_value is None else min_lr_value)

        def _ctor(opt: torch.optim.Optimizer) -> Any:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

        return _ctor

    if scheduler_type in {"cosine", "cosine_annealing"}:
        t_max_value = spec.get("t_max", total_steps)
        t_max = int(total_steps if t_max_value is None else t_max_value)
        eta_min_value = spec.get("eta_min", 5e-7)
        eta_min = float(5e-7 if eta_min_value is None else eta_min_value)

        def _ctor(opt: torch.optim.Optimizer) -> Any:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max(1, t_max),
                eta_min=eta_min,
            )

        return _ctor

    if scheduler_type in {"cosine_warmup", "warmup_cosine", "cosine_with_warmup"}:
        warmup_ratio_value = spec.get("warmup_ratio", 0.07)
        warmup_ratio = float(0.07 if warmup_ratio_value is None else warmup_ratio_value)
        warmup_steps_raw = spec.get("warmup_steps", spec.get("warmup_epochs"))
        if warmup_steps_raw is None:
            warmup_steps = max(20, int(warmup_ratio * total_steps))
        else:
            warmup_steps = int(warmup_steps_raw)

        max_warmup = max(1, total_steps - 1)
        warmup_steps = max(1, min(warmup_steps, max_warmup))
        warmup_start_factor_value = spec.get("warmup_start_factor", 0.05)
        warmup_start_factor = float(
            0.05 if warmup_start_factor_value is None else warmup_start_factor_value
        )
        eta_min_value = spec.get("eta_min", 1e-6)
        eta_min = float(1e-6 if eta_min_value is None else eta_min_value)
        t_max_value = spec.get("t_max", max(1, total_steps - warmup_steps))
        t_max = int(max(1, total_steps - warmup_steps) if t_max_value is None else t_max_value)

        def _ctor(opt: torch.optim.Optimizer) -> Any:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=max(1, t_max),
                eta_min=eta_min,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )

        return _ctor

    if scheduler_type in {"cosine_restarts", "cosine_restart", "warm_restarts"}:
        t0_default = min(10, total_steps)
        t0_value = spec.get("t0", t0_default)
        t0 = int(t0_default if t0_value is None else t0_value)
        t_mult_value = spec.get("t_mult", 2)
        t_mult = int(2 if t_mult_value is None else t_mult_value)
        eta_min_value = spec.get("eta_min", 5e-7)
        eta_min = float(5e-7 if eta_min_value is None else eta_min_value)

        def _ctor(opt: torch.optim.Optimizer) -> Any:
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=max(1, t0),
                T_mult=max(1, t_mult),
                eta_min=eta_min,
            )

        return _ctor

    raise ValueError(
        f"Unknown lr_scheduler type {scheduler_type!r}. "
        'Use "none", "cosine", "cosine_warmup", or "cosine_restarts".'
    )
