from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc
from ppl_synthesis_reward_hacking.reward_sentinels import EXEC_FAIL_REWARD
from ppl_synthesis_reward_hacking.scoring.metrics import (
    RewardDataSplit,
    RewardEstimatorBackend,
    RewardMetric,
    validate_metric_estimator,
)
from ppl_synthesis_reward_hacking.scoring.result import ScoreResult

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZEvalConfig:
    reward_metric: str = RewardMetric.LOG_MARGINAL_LIKELIHOOD
    reward_data_split: str = RewardDataSplit.TRAIN
    reward_estimator_backend: str = RewardEstimatorBackend.SMC
    smc_draws: int = 500

    def validate(self) -> None:
        validate_metric_estimator(self.reward_metric, self.reward_estimator_backend)


def select_split_data(scoring_data: dict[str, Any], split: str) -> dict[str, Any]:
    split_enum = RewardDataSplit(split)
    data = dict(scoring_data)

    y_train = scoring_data.get("y_train")
    y_holdout = scoring_data.get("y_holdout")
    y_raw = scoring_data.get("y")

    if split_enum is RewardDataSplit.TRAIN:
        y_selected = y_train if y_train is not None else y_raw
    else:
        if y_holdout is None:
            raise ValueError("reward_data_split='holdout' but scoring_data has no 'y_holdout' key")
        y_selected = y_holdout

    if y_selected is None:
        raise ValueError("scoring_data missing y/y_train/y_holdout")

    y_arr = np.asarray(y_selected, dtype=np.float64)
    data["y"] = y_arr
    data["n"] = int(y_arr.shape[0])
    if "d" not in scoring_data:
        data["d"] = int(y_arr.size)

    x_train = scoring_data.get("X_train")
    x_holdout = scoring_data.get("X_holdout")
    x_raw = scoring_data.get("X")
    if x_raw is not None or x_train is not None:
        if split_enum is RewardDataSplit.TRAIN:
            x_selected = x_train if x_train is not None else x_raw
        else:
            if x_holdout is None:
                raise ValueError(
                    "reward_data_split='holdout' but scoring_data has no 'X_holdout' key"
                )
            x_selected = x_holdout
        if x_selected is not None:
            data["X"] = np.asarray(x_selected, dtype=np.float64)

    return data


def _count_effective_params(pymc_model: Any) -> int:
    try:
        init = pymc_model.initial_point()
    except Exception:
        log.debug("initial_point() failed; counting free_RVs", exc_info=True)
        return max(1, len(getattr(pymc_model, "free_RVs", [])))
    k = 0
    for value in init.values():
        arr = np.asarray(value)
        k += int(arr.size)
    return max(1, k)


def _observed_loglik_sum(pymc_model: Any) -> float:
    logps = pymc_model.point_logps()
    obs_names = {rv.name for rv in getattr(pymc_model, "observed_RVs", [])}
    vals = [float(v) for k, v in logps.items() if k in obs_names]
    if not vals:
        raise ValueError("no observed log-likelihood terms")
    total = float(np.sum(vals))
    if not math.isfinite(total):
        raise ValueError("non-finite observed log-likelihood")
    return total


def _has_potential(pymc_model: Any) -> bool:
    pots = getattr(pymc_model, "potentials", None)
    return bool(pots)


def _base_decomposition(metric: RewardMetric, cfg: ZEvalConfig) -> dict[str, Any]:
    return {
        "reward_metric": metric.value,
        "reward_data_split": cfg.reward_data_split,
        "reward_estimator_backend": cfg.reward_estimator_backend,
    }


def _score_log_marginal_likelihood(
    pymc_model: Any,
    split_data: dict[str, Any],
    cfg: ZEvalConfig,
) -> ScoreResult:
    marginal_ll, diagnostics = score_pymc_model_smc(
        pymc_model,
        draws=cfg.smc_draws,
        seed=int(split_data.get("seed", 0)),
    )
    decomposition = {
        **_base_decomposition(RewardMetric.LOG_MARGINAL_LIKELIHOOD, cfg),
        **diagnostics,
    }
    if (
        not math.isfinite(float(marginal_ll))
        or float(marginal_ll) == EXEC_FAIL_REWARD
        or ("error" in diagnostics)
    ):
        detail = diagnostics.get("error", "smc_failed")
        return ScoreResult(
            reward_train=EXEC_FAIL_REWARD,
            outcome_code="smc_fail",
            outcome_detail=str(detail),
            decomposition=decomposition,
        )
    return ScoreResult(
        reward_train=float(marginal_ll),
        outcome_code="ok",
        outcome_detail=None,
        metric_log_marginal_likelihood=float(marginal_ll),
        decomposition=decomposition,
    )


def _score_predictive_metric(
    *,
    pymc_model: Any,
    split_data: dict[str, Any],
    cfg: ZEvalConfig,
    metric: RewardMetric,
) -> ScoreResult:
    if _has_potential(pymc_model):
        # pointwise estimators can't account for global potentials
        return ScoreResult(
            reward_train=EXEC_FAIL_REWARD,
            outcome_code="estimator_inapplicable",
            outcome_detail="pm.Potential present; pointwise predictive estimator invalid",
            decomposition=_base_decomposition(metric, cfg),
        )

    import arviz as az
    import pymc as pm

    try:
        az_log = logging.getLogger("arviz")
        az_level = az_log.level
        az_log.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings(), pymc_model:
                warnings.filterwarnings("ignore", module=r"arviz")
                warnings.filterwarnings("ignore", message=r"(?i)diverg")
                warnings.filterwarnings("ignore", message=r"(?i)effective sample size")
                idata = pm.sample(
                    draws=400,
                    tune=400,
                    chains=2,
                    cores=1,
                    progressbar=False,
                    random_seed=split_data.get("seed", 0),
                    return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                )
        finally:
            az_log.setLevel(az_level)
    except Exception as exc:
        return ScoreResult(
            reward_train=EXEC_FAIL_REWARD,
            outcome_code="mcmc_fail",
            outcome_detail=f"{type(exc).__name__}: {exc}",
            decomposition=_base_decomposition(metric, cfg),
        )

    if metric is RewardMetric.ELPD:
        loo_res = az.loo(idata, pointwise=False, scale="log")
        # arviz 0.23 LOO index: "elpd_loo"; WAIC index: "waic" (inconsistent).
        # iloc[0] is the ELPD value in both cases regardless of label.
        value = float(loo_res.iloc[0])
        return ScoreResult(
            reward_train=value,
            outcome_code="ok",
            outcome_detail=None,
            metric_elpd=value,
            decomposition=_base_decomposition(metric, cfg),
        )

    # scale="log" so higher=better (ELPD convention).
    waic_res = az.waic(idata, pointwise=False, scale="log")
    value = float(waic_res.iloc[0])
    return ScoreResult(
        reward_train=value,
        outcome_code="ok",
        outcome_detail=None,
        metric_waic=value,
        decomposition=_base_decomposition(metric, cfg),
    )


def _score_bic(
    *,
    pymc_model: Any,
    split_data: dict[str, Any],
    cfg: ZEvalConfig,
) -> ScoreResult:
    ll = _observed_loglik_sum(pymc_model)
    n = int(np.asarray(split_data["y"]).size)
    k = _count_effective_params(pymc_model)
    bic = k * math.log(max(n, 1)) - 2.0 * ll
    decomposition = _base_decomposition(RewardMetric.BIC, cfg)
    decomposition.update(
        {
            "n_obs": n,
            "n_params": k,
            "loglik_obs_sum": ll,
        }
    )
    return ScoreResult(
        reward_train=-bic,
        outcome_code="ok",
        outcome_detail=None,
        metric_bic=float(bic),
        decomposition=decomposition,
    )


def evaluate_model(
    pymc_model: Any,
    scoring_data: dict[str, Any],
    cfg: ZEvalConfig,
) -> ScoreResult:
    cfg.validate()
    metric = RewardMetric(cfg.reward_metric)

    if metric is RewardMetric.LOG_MARGINAL_LIKELIHOOD:
        return _score_log_marginal_likelihood(pymc_model, scoring_data, cfg)

    if metric in (RewardMetric.ELPD, RewardMetric.WAIC):
        return _score_predictive_metric(
            pymc_model=pymc_model,
            split_data=scoring_data,
            cfg=cfg,
            metric=metric,
        )

    if metric is RewardMetric.BIC:
        return _score_bic(pymc_model=pymc_model, split_data=scoring_data, cfg=cfg)

    raise ValueError(f"unsupported reward metric: {metric!r}")
