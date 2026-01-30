from __future__ import annotations

from pathlib import Path


def artifacts_root(base: Path) -> Path:
    return base / "artifacts"


def runs_dir(base: Path) -> Path:
    return artifacts_root(base) / "runs"


def datasets_dir(base: Path) -> Path:
    return artifacts_root(base) / "datasets"
