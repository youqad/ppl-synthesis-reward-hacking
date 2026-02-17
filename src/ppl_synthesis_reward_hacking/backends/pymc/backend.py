from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.backends.protocol import (
    Backend,
    FitResult,
    ModelSpec,
    ScoreResult,
)
from ppl_synthesis_reward_hacking.backends.pymc.templates import build_model
from ppl_synthesis_reward_hacking.data.schema import Dataset

DEFAULT_MCMC_DRAWS = 200
DEFAULT_MCMC_TUNE = 200


class PyMCBackend(Backend):
    name = "pymc"

    def compile(self, model: ModelSpec, *, cache_dir: Path):
        pymc_model = model.meta.get("pymc_model")
        if pymc_model is not None:
            return {"model": pymc_model, "model_spec": model}
        return {"model_spec": model}

    def fit(self, compiled, *, dataset: Dataset, seed: int) -> FitResult:
        try:
            import pymc as pm
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyMC backend requires pymc") from exc

        model = compiled.get("model")
        if model is None:
            model = build_model(compiled["model_spec"], dataset)
            compiled["model"] = model

        with model:
            _set_model_data(model, dataset.train)
            trace = pm.sample(
                draws=DEFAULT_MCMC_DRAWS,
                tune=DEFAULT_MCMC_TUNE,
                chains=1,
                cores=1,
                random_seed=seed,
                return_inferencedata=False,
                progressbar=False,
            )
        return FitResult(artifact=trace, meta={"seed": seed})

    def score_holdout(
        self, compiled, *, fit: FitResult, dataset: Dataset, seed: int
    ) -> ScoreResult:
        try:
            import pymc as pm  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyMC backend requires pymc") from exc

        model = compiled.get("model")
        if model is None:
            model = build_model(compiled["model_spec"], dataset)
            compiled["model"] = model

        with model:
            _set_model_data(model, dataset.holdout)
            points = list(_iter_points(fit.artifact))
            reported_vals = [v for v in (_logp(model, point) for point in points) if v is not None]
            reported_reward = float(np.mean(reported_vals)) if reported_vals else float("nan")
            observed = list(getattr(model, "observed_RVs", []))
            oracle_vals = []
            if observed:
                for point in points:
                    value = _logp(model, point, vars=observed)
                    if value is not None:
                        oracle_vals.append(value)
            oracle_score = float(np.mean(oracle_vals)) if oracle_vals else None
        return ScoreResult(
            reported_reward=reported_reward,
            oracle_score=oracle_score,
            diagnostics={},
        )


def _iter_points(trace) -> list[dict[str, Any]]:
    if trace is None:
        return []
    points = []
    if hasattr(trace, "point"):
        for idx in range(len(trace)):
            points.append(trace.point(idx))
    return points


def _set_model_data(model, data: Mapping[str, Any]) -> None:
    if not data:
        return
    mapping = {}
    for key, value in data.items():
        if key in getattr(model, "named_vars", {}):
            mapping[key] = np.asarray(value)
    if not mapping:
        return
    import pymc as pm

    pm.set_data(mapping, model=model)


def _logp(model, point: dict[str, Any], *, vars=None) -> float | None:
    try:
        return float(model.logp(point=point, vars=vars))
    except TypeError:
        try:
            return float(model.logp(point))
        except Exception:
            return None
    except Exception:
        return None
