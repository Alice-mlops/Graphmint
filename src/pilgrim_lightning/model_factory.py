"""Model factory for Pilgrim-family modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pilgrim.model import AlGraphGPT, AliceInCayleyland, AlkeelGrim, AlPilgrim, Pilgrim
from pilgrim.schemas import AlGraphGPTConfig
from torch import nn

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "Pilgrim": Pilgrim,
    "AlPilgrim": AlPilgrim,
    "AlkeelGrim": AlkeelGrim,
    "AliceInCayleyland": AliceInCayleyland,
}


def build_model(model_name: str, model_config: Mapping[str, Any]) -> nn.Module:
    """
    Construct a Pilgrim model by name.

    Args:
        model_name: Model name from ``pilgrim.model``.
        model_config: Initialization config for the selected model.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If ``model_name`` is not supported.

    """
    if model_name == "AlGraphGPT":
        cfg = model_config
        if not isinstance(cfg, AlGraphGPTConfig):
            cfg = AlGraphGPTConfig(**dict(model_config))
        return AlGraphGPT(cfg)

    model_cls = _MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        supported = sorted(["AlGraphGPT", *_MODEL_REGISTRY.keys()])
        raise ValueError(
            f"Unsupported model_name={model_name!r}. Supported: {supported}."
        )

    return model_cls(dict(model_config))
