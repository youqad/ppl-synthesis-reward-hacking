from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from ppl_synthesis_reward_hacking.backends.protocol import FitResult, ScoreResult
from ppl_synthesis_reward_hacking.data.schema import Dataset


class Evaluator(Protocol):
    name: str

    def compute(
        self,
        *,
        dataset: Dataset,
        score: ScoreResult,
        fit: FitResult | None,
    ) -> Mapping[str, float]:  # pragma: no cover - interface
        ...
