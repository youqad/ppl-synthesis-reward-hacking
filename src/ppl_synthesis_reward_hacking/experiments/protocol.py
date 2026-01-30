from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RunContext:
    run_dir: Path
    seed: int
    config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    dataset_id: str
    backend: str
    experiment: str
    model_name: str
    metrics: Mapping[str, float]
    artifacts: Mapping[str, str]


class Experiment(Protocol):
    name: str

    def run(self, ctx: RunContext) -> RunResult:  # pragma: no cover - interface
        ...
