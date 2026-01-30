from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    config_hash: str
    git_commit: str | None
    git_dirty: bool
    python_version: str
    platform: str
    meta: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MetricRecord:
    metric_name: str
    metric_value: float
    meta: Mapping[str, Any]
