"""Path lookups for lab gates.

Lab gates take checkout and library locations from the environment.
There is no machine-local fallback.
"""

from __future__ import annotations

import os
from pathlib import Path


def optional_env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def require_env_path(name: str, *, kind: str = "any") -> Path:
    """Return a path from the environment, or exit with a usage error.

    ``kind`` is ``any``, ``file``, or ``dir``.
    """
    path = optional_env_path(name)
    if path is None:
        raise SystemExit(
            f"error: {name} is required. "
            "This script does not fall back to a machine-local path."
        )
    if kind == "file" and not path.is_file():
        raise SystemExit(f"error: {name} is not a file: {path}")
    if kind == "dir" and not path.is_dir():
        raise SystemExit(f"error: {name} is not a directory: {path}")
    if kind not in ("any", "file", "dir"):
        raise ValueError(f"unknown path kind: {kind}")
    return path
