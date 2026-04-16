"""
Path helpers for notebooks and scripts.

This module contains small utilities that make it easier to run notebooks and
scripts from arbitrary working directories (including Kaggle environments).
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

AddRepoSrcResult = tuple[Literal["kaggle", "local", "none"], str]


def add_repo_src_to_path(
    *,
    lib_name: str = "pilgrim",
    marker_files: Sequence[str] = ("pyproject.toml", ".git"),
    marker_dirs: Sequence[str] = ("src",),
    start_dir: Path | None = None,
    kaggle_input_root: Path = Path("/kaggle/input"),
) -> AddRepoSrcResult:
    """
    Add the repository's ``src/`` directory to ``sys.path``.

    The function walks upward from ``start_dir`` (defaults to ``Path.cwd()``)
    looking for a directory that contains ``src/``. When found, that ``src/``
    directory is prepended to ``sys.path``.

    Kaggle shortcut: if the current working directory is under ``/kaggle`` and
    ``/kaggle/input/<lib_name>`` exists, that directory is prepended instead.

    Args:
        lib_name: Kaggle dataset directory name under ``kaggle_input_root``.
        marker_files: Filenames that suggest a repository root (used only as a
            heuristic; the first parent with ``src/`` is accepted regardless).
        marker_dirs: Directory names that suggest a repository root (heuristic).
        start_dir: Directory to start the upward search from. Defaults to
            ``Path.cwd()``.
        kaggle_input_root: Root directory for Kaggle input datasets.

    Returns:
        A pair ``(where, path)`` where:
        - ``where`` is ``"kaggle"``, ``"local"``, or ``"none"``.
        - ``path`` is the inserted path (as a string), or ``""`` if nothing was
          inserted.

    """

    def _prepend(path: str) -> None:
        # Ensure the path is first, without duplicates.
        with contextlib.suppress(ValueError):
            sys.path.remove(path)
        sys.path.insert(0, path)

    here = (start_dir or Path.cwd()).resolve()
    if "/kaggle" in str(here):
        kaggle_dir = kaggle_input_root / lib_name
        if kaggle_dir.is_dir():
            _prepend(str(kaggle_dir))
            return "kaggle", str(kaggle_dir)

    for p in (here, *here.parents):
        has_src = (p / "src").is_dir()
        if not has_src:
            continue

        looks_like_repo = any((p / m).exists() for m in marker_files) or any(
            (p / d).is_dir() for d in marker_dirs
        )
        if looks_like_repo:
            _prepend(str(p / "src"))
            return "local", str(p / "src")

        # Even if no markers, accept the first parent that has src/.
        _prepend(str(p / "src"))
        return "local", str(p / "src")

    return "none", ""
