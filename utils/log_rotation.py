"""Small size-based log rotation helpers.

This is intentionally boring: if a log is over the byte cap, move
``name`` -> ``name.1``, ``name.1`` -> ``name.2``, and so on, deleting the
oldest rollover. Callers should invoke it before opening a file for append.
"""

from __future__ import annotations

import os
from pathlib import Path


def rotate_file(path: str | os.PathLike[str], max_bytes: int, keep: int) -> bool:
    """Rotate ``path`` when it is larger than ``max_bytes``.

    Returns True when a rotation happened. Missing files, disabled caps, and
    small files are no-ops.
    """
    log_path = Path(path)
    if max_bytes <= 0 or keep <= 0 or not log_path.exists():
        return False
    try:
        if log_path.stat().st_size <= max_bytes:
            return False
    except OSError:
        return False

    log_path.parent.mkdir(parents=True, exist_ok=True)

    oldest = _rollover_path(log_path, keep)
    if oldest.exists():
        oldest.unlink()

    for index in range(keep - 1, 0, -1):
        src = _rollover_path(log_path, index)
        if src.exists():
            src.replace(_rollover_path(log_path, index + 1))

    log_path.replace(_rollover_path(log_path, 1))
    return True


def _rollover_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.name}.{index}")
