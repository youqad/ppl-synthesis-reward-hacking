from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    python_version: str
    platform: str
    git_commit: str | None
    git_dirty: bool


def capture_runtime_info() -> RuntimeInfo:
    return RuntimeInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception:
        return False
    return bool(output.strip())
