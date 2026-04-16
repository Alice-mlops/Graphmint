"""Model checkpoint helpers used in notebooks/scripts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

TModel = TypeVar("TModel", bound=nn.Module)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU-cloned copy of ``model.state_dict()``."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_one(
    n: int,
    model: nn.Module,
    cfg: Mapping[str, Any],
    out_dir: str | Path,
    *,
    filename_template: str = "model_n{n}.pt",
) -> Path:
    """
    Save a single model checkpoint for a specific ``n``.

    The file format is a ``torch.save`` payload with keys:
    - ``"n"``: int
    - ``"cfg"``: configuration mapping used to construct the model
    - ``"state_dict"``: model weights (saved on CPU for portability)

    This mirrors the checkpoint structure used by Kaggle notebooks so that
    models can be trained per-``n`` and later reloaded for inference.

    Args:
        n: Problem size the model corresponds to.
        model: Trained model to serialize.
        cfg: Configuration used to build the model.
        out_dir: Output directory to write the checkpoint into.
        filename_template: Filename template that must contain ``{n}``.

    Returns:
        Path to the written checkpoint file.

    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / filename_template.format(n=int(n))

    payload: dict[str, Any] = {
        "n": int(n),
        "cfg": dict(cfg),
        "state_dict": _cpu_state_dict(model),
    }
    torch.save(payload, file_path)
    return file_path


def load_one(
    path: str | Path,
    model_ctor: Callable[[Mapping[str, Any]], TModel],
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[int, TModel, Mapping[str, Any]]:
    """
    Load a model checkpoint produced by :func:`save_one`.

    Args:
        path: Path to a checkpoint file created by :func:`save_one`.
        model_ctor: Callable that constructs the model given the stored config.
            Example: ``lambda cfg: AlPilgrim(cfg)``.
        device: Device to move the loaded model to.
        strict: Forwarded to ``model.load_state_dict``.

    Returns:
        Tuple ``(n, model, cfg)`` where:
        - ``n`` is the stored problem size,
        - ``model`` is the reconstructed model in eval mode on ``device``,
        - ``cfg`` is the stored configuration mapping.

    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, Mapping):
        raise TypeError(
            f"Expected checkpoint payload to be a mapping, got {type(ckpt).__name__}."
        )

    n = int(ckpt["n"])
    cfg = ckpt["cfg"]
    state_dict = ckpt["state_dict"]

    if not isinstance(cfg, Mapping):
        raise TypeError(
            f'Expected checkpoint["cfg"] to be a mapping, got {type(cfg).__name__}.'
        )
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f'Expected checkpoint["state_dict"] to be a mapping, got {type(state_dict).__name__}.'
        )

    model = model_ctor(cfg)
    model.load_state_dict(state_dict, strict=bool(strict))
    model = model.to(device).eval()
    return n, model, cfg
