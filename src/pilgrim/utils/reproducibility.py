"""Reproducibility helpers (seeding, determinism flags)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, deterministic_cudnn: bool = True) -> None:
    """
    Seed common RNGs to make runs more reproducible.

    This function seeds:
    - Python's ``random`` module
    - NumPy's global RNG
    - PyTorch CPU RNG
    - PyTorch CUDA RNGs (all devices), if CUDA is available

    It can also configure cuDNN for more deterministic behavior.

    Args:
        seed: Seed value to use.
        deterministic_cudnn: If True, sets:
            - ``torch.backends.cudnn.deterministic = True``
            - ``torch.backends.cudnn.benchmark = False``

    Returns:
        None.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
