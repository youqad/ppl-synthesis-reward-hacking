from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.backends.protocol import (
    DEFAULT_MCMC_DRAWS,
    DEFAULT_MCMC_TUNE,
    Backend,
    BackendScoreResult,
    FitResult,
    ModelSpec,
)
from ppl_synthesis_reward_hacking.backends.source_parsing import (
    extract_block_lines,
    strip_line_comment,
)
from ppl_synthesis_reward_hacking.backends.stan.compile_cache import get_compile_dir
from ppl_synthesis_reward_hacking.data.schema import Dataset
from ppl_synthesis_reward_hacking.utils.hashing import stable_hash


class StanBackend(Backend):
    name = "stan"

    def compile(self, model: ModelSpec, *, cache_dir: Path):
        try:
            from cmdstanpy import CmdStanModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Stan backend requires cmdstanpy") from exc

        source = _resolve_source(model)
        model_hash = stable_hash({"name": model.name, "source": source})
        compile_dir = get_compile_dir(cache_dir, model_hash)
        compile_dir.mkdir(parents=True, exist_ok=True)

        stan_file = compile_dir / f"{model.name}.stan"
        stan_file.write_text(source, encoding="utf-8")

        cmdstan_model = CmdStanModel(stan_file=str(stan_file))
        return {
            "model": cmdstan_model,
            "model_hash": model_hash,
            "source": source,
            "model_meta": dict(model.meta or {}),
        }

    def fit(self, compiled, *, dataset: Dataset, seed: int) -> FitResult:
        cmdstan_model = compiled["model"]
        data = _prepare_data(
            dataset.train,
            compiled.get("source", ""),
            model_meta=compiled.get("model_meta"),
        )
        fit = cmdstan_model.sample(
            data=data,
            seed=seed,
            chains=1,
            iter_sampling=DEFAULT_MCMC_DRAWS,
            iter_warmup=DEFAULT_MCMC_TUNE,
            show_progress=False,
        )
        return FitResult(artifact=fit, meta={"seed": seed})

    def score_holdout(
        self, compiled, *, fit: FitResult, dataset: Dataset, seed: int
    ) -> BackendScoreResult:
        cmdstan_model = compiled["model"]
        data = _prepare_data(
            dataset.holdout,
            compiled.get("source", ""),
            model_meta=compiled.get("model_meta"),
        )
        try:
            gq = cmdstan_model.generate_quantities(
                data=data,
                mcmc_sample=fit.artifact,
                seed=seed,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("generate_quantities failed for holdout") from exc

        log_lik_sum = _extract_log_sum(gq, "log_lik")

        log_score_sum = _extract_log_sum(gq, "log_score")

        if log_score_sum is None and log_lik_sum is None:
            raise RuntimeError("generated quantities must define log_score or log_lik")

        if log_score_sum is not None:
            reported_source = log_score_sum
        else:
            if log_lik_sum is None:  # pragma: no cover - guarded above
                raise RuntimeError("log_lik_sum is None (should be unreachable)")
            reported_source = log_lik_sum
        reported_reward = float(np.mean(reported_source))
        ground_truth_loglik = float(np.mean(log_lik_sum)) if log_lik_sum is not None else None
        diagnostics: dict[str, float] = {
            "reported_from_log_score": 1.0 if log_score_sum is not None else 0.0,
            "oracle_from_log_lik": 1.0 if log_lik_sum is not None else 0.0,
        }
        if ground_truth_loglik is not None:
            diagnostics["ground_truth_loglik"] = ground_truth_loglik
        return BackendScoreResult(
            reported_reward=reported_reward,
            diagnostics=diagnostics,
        )


def _resolve_source(model: ModelSpec) -> str:
    if model.source:
        return model.source
    templates_dir = Path(__file__).parent / "templates"
    template_path = templates_dir / f"{model.name}.stan"
    if not template_path.exists():
        raise FileNotFoundError(f"Stan template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _prepare_data(
    split: Mapping[str, Any],
    source: str,
    *,
    model_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(split)
    if "y" in data:
        y = np.asarray(data["y"])
        if y.ndim > 1:
            y = y.reshape(-1)
        data["y"] = y
        data["N"] = int(y.shape[0])
    if _data_block_declares_constant(source):
        constant = None
        attack_cfg = (model_meta or {}).get("attack")
        if isinstance(attack_cfg, dict) and attack_cfg.get("C") is not None:
            try:
                constant = float(attack_cfg["C"])
            except (TypeError, ValueError):
                constant = None
        if constant is None and "C" in data:
            try:
                constant = float(data["C"])
            except (TypeError, ValueError):
                constant = None
        if constant is None:
            constant = _extract_constant(source)
        data["C"] = constant
    return data


def _extract_constant(source: str) -> float:
    for line in source.splitlines():
        if "// C=" in line:
            try:
                return float(line.split("// C=")[-1].strip())
            except ValueError:
                return 0.0
    return 0.0


def _data_block_declares_constant(source: str) -> bool:
    for line in extract_block_lines(source, "data"):
        stripped = strip_line_comment(line)
        if re.search(r"\bC\b", stripped) and ";" in stripped:
            return True
    return False


def _extract_log_sum(gq: Any, name: str) -> np.ndarray | None:
    try:
        values = gq.stan_variable(name)
    except (AttributeError, KeyError, ValueError, RuntimeError):
        return None
    if values.ndim == 1:
        return values
    return values.sum(axis=1)
