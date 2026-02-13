"""Shared run helpers: config persistence and deterministic run IDs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ppl_synthesis_reward_hacking.utils.hashing import stable_hash


def write_resolved_config(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8")


def compute_run_id(
    config: Mapping[str, Any],
    dataset_id: str,
    seed: int,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {"config": config, "dataset_id": dataset_id, "seed": seed}
    if extra:
        payload.update(extra)
    return stable_hash(payload)
