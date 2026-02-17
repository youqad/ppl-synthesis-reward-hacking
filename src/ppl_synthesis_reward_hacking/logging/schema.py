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


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """Post-hoc evaluation row keyed by normalized program hash."""

    run_id: str
    code_hash: str
    step: int
    reported_reward: float
    oracle_score: float | None
    log_mass: float | None
    exploit_tags: tuple[str, ...]
    safe_gate_accept: bool | None
    safe_gate_reasons: tuple[str, ...]
    transpile_status: str | None
    transpile_invariants: Mapping[str, bool] | None
    meta: Mapping[str, Any]
