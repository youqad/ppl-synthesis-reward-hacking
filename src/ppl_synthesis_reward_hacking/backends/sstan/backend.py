from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ppl_synthesis_reward_hacking.backends.protocol import (
    Backend,
    FitResult,
    ModelSpec,
    ScoreResult,
)
from ppl_synthesis_reward_hacking.backends.sstan.checker import check_sstan
from ppl_synthesis_reward_hacking.backends.stan.backend import StanBackend, _resolve_source
from ppl_synthesis_reward_hacking.data.schema import Dataset


class SStanBackend(Backend):
    name = "sstan"

    def __init__(self) -> None:
        self._stan = StanBackend()

    def compile(self, model: ModelSpec, *, cache_dir: Path):
        source = _resolve_source(model)
        accepted, reasons = check_sstan(source)
        compiled: dict[str, object] = {
            "accepted": accepted,
            "reasons": reasons,
            "source": source,
        }
        if not accepted:
            return compiled
        safe_model = replace(model, source=source)
        compiled_stan = self._stan.compile(safe_model, cache_dir=cache_dir)
        compiled_stan.update(compiled)
        return compiled_stan

    def fit(self, compiled, *, dataset: Dataset, seed: int) -> FitResult:
        if not compiled.get("accepted", False):
            reason = "; ".join(compiled.get("reasons", []))
            return FitResult(artifact=None, meta={"sstan_reject_reason": reason})
        return self._stan.fit(compiled, dataset=dataset, seed=seed)

    def score_holdout(
        self, compiled, *, fit: FitResult, dataset: Dataset, seed: int
    ) -> ScoreResult:
        if not compiled.get("accepted", False):
            return ScoreResult(
                reported_reward=float("nan"),
                oracle_score=None,
                diagnostics={"sstan_accepted": 0.0},
            )
        score = self._stan.score_holdout(compiled, fit=fit, dataset=dataset, seed=seed)
        diagnostics = dict(score.diagnostics)
        diagnostics["sstan_accepted"] = 1.0
        return ScoreResult(
            reported_reward=score.reported_reward,
            oracle_score=score.oracle_score,
            diagnostics=diagnostics,
        )
