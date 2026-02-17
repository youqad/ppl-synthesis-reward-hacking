from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def git_info(repo_root: Path) -> dict[str, Any]:
    """Snapshot the current git state for a repository."""

    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=repo_root, text=True).strip()
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status_output = _run(["git", "status", "--porcelain=v1"])
    dirty = bool(status_output) if status_output is not None else None
    return {"head": head, "branch": branch, "dirty": dirty}
