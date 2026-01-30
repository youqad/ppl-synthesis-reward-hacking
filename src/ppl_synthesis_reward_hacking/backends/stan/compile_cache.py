from __future__ import annotations

from pathlib import Path


def get_compile_dir(cache_root: Path, model_hash: str) -> Path:
    return cache_root / "stan" / model_hash
