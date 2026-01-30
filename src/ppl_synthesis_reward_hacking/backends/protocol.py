from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ppl_synthesis_reward_hacking.data.schema import Dataset


@dataclass(frozen=True, slots=True)
class ModelSpec:
    backend: str
    name: str
    source: str
    meta: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class FitResult:
    artifact: Any
    meta: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScoreResult:
    reported_reward: float
    oracle_score: float | None
    diagnostics: Mapping[str, float]


@runtime_checkable
class Backend(Protocol):
    name: str

    def compile(self, model: ModelSpec, *, cache_dir: Path) -> Any:  # pragma: no cover - interface
        ...

    def fit(self, compiled: Any, *, dataset: Dataset, seed: int) -> FitResult:  # pragma: no cover
        ...

    def score_holdout(
        self, compiled: Any, *, fit: FitResult, dataset: Dataset, seed: int
    ) -> ScoreResult:  # pragma: no cover
        ...
