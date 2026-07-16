"""Atomic replacement helpers for artifacts read by the live bot."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_atomic(path, mode: str = "wb", *, encoding: str | None = None):
    """Yield a temporary file and atomically replace ``path`` on success."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "mode": mode,
        "dir": target.parent,
        "prefix": f".{target.name}.",
        "suffix": ".tmp",
        "delete": False,
    }
    if "b" not in mode:
        kwargs["encoding"] = encoding or "utf-8"
    handle = tempfile.NamedTemporaryFile(**kwargs)
    temporary = Path(handle.name)
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, target)
    except Exception:
        handle.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
