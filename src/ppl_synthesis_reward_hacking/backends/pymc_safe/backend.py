from __future__ import annotations

from pathlib import Path

from ppl_synthesis_reward_hacking.backends.protocol import (
    Backend,
    BackendScoreResult,
    FitResult,
    ModelSpec,
)
from ppl_synthesis_reward_hacking.backends.pymc.backend import PyMCBackend
from ppl_synthesis_reward_hacking.backends.pymc.templates import build_model
from ppl_synthesis_reward_hacking.backends.pymc_safe.checker import check_pymc_model
from ppl_synthesis_reward_hacking.data.schema import Dataset


class SafePyMCBackend(Backend):
    name = "pymc_safe"

    def __init__(self) -> None:
        self._pymc = PyMCBackend()

    def compile(self, model: ModelSpec, *, cache_dir: Path):
        pymc_model = model.meta.get("pymc_model")
        if pymc_model is None:
            return {"model_spec": model}
        accepted, reasons = check_pymc_model(pymc_model)
        compiled = {
            "accepted": accepted,
            "reasons": reasons,
            "model": pymc_model,
            "model_spec": model,
        }
        if not accepted:
            return compiled
        compiled_inner = self._pymc.compile(model, cache_dir=cache_dir)
        compiled_inner.update(compiled)
        return compiled_inner

    def fit(self, compiled, *, dataset: Dataset, seed: int) -> FitResult:
        model = compiled.get("model")
        if model is None:
            model = build_model(compiled["model_spec"], dataset)
            compiled["model"] = model
        if "accepted" not in compiled:
            accepted, reasons = check_pymc_model(model)
            compiled["accepted"] = accepted
            compiled["reasons"] = reasons
        if not compiled.get("accepted", False):
            return FitResult(
                artifact=None,
                meta={"pymc_safe_reject_reason": "; ".join(compiled.get("reasons", []))},
            )
        return self._pymc.fit(compiled, dataset=dataset, seed=seed)

    def score_holdout(
        self, compiled, *, fit: FitResult, dataset: Dataset, seed: int
    ) -> BackendScoreResult:
        if not compiled.get("accepted", False):
            return BackendScoreResult(
                reported_reward=float("nan"),
                diagnostics={"pymc_safe_accepted": 0.0},
            )
        score = self._pymc.score_holdout(compiled, fit=fit, dataset=dataset, seed=seed)
        diagnostics = dict(score.diagnostics)
        diagnostics["pymc_safe_accepted"] = 1.0
        return BackendScoreResult(
            reported_reward=score.reported_reward,
            diagnostics=diagnostics,
        )
