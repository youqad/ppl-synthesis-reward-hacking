from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc
from ppl_synthesis_reward_hacking.scoring.metrics import (
    PredictiveEstimator,
    RewardDataSplit,
    RewardMetric,
    validate_metric_estimator,
)
from ppl_synthesis_reward_hacking.scoring.result import ScoreResult

EXEC_FAIL_REWARD = -400.0


@dataclass(frozen=True, slots=True)
class ZEvalConfig:
    reward_metric: str = RewardMetric.LOG_MARGINAL_LIKELIHOOD
    reward_data_split: str = RewardDataSplit.TRAIN
    predictive_estimator: str = PredictiveEstimator.NONE
    smc_draws: int = 500

    def validate(self) -> None:
        validate_metric_estimator(self.reward_metric, self.predictive_estimator)


def select_split_data(scoring_data: dict[str, Any], split: str) -> dict[str, Any]:
    split_enum = RewardDataSplit(split)
    data = dict(scoring_data)

    y_train = scoring_data.get("y_train")
    y_holdout = scoring_data.get("y_holdout")
    y_raw = scoring_data.get("y")

    if split_enum is RewardDataSplit.TRAIN:
        y_selected = y_train if y_train is not None else y_raw
    else:
        y_selected = y_holdout if y_holdout is not None else y_raw

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
            x_selected = x_holdout if x_holdout is not None else x_raw
        if x_selected is not None:
            data["X"] = np.asarray(x_selected, dtype=np.float64)

    return data


def _count_effective_params(pymc_model: Any) -> int:
    try:
        init = pymc_model.initial_point()
    except Exception:
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


def evaluate_model(
    pymc_model: Any,
    scoring_data: dict[str, Any],
    cfg: ZEvalConfig,
) -> ScoreResult:
    cfg.validate()
    metric = RewardMetric(cfg.reward_metric)

    split_data = select_split_data(scoring_data, cfg.reward_data_split)

    if metric is RewardMetric.LOG_MARGINAL_LIKELIHOOD:
        marginal_ll, diagnostics = score_pymc_model_smc(
            pymc_model,
            draws=cfg.smc_draws,
            seed=int(split_data.get("seed", 0)),
        )
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
                decomposition={
                    "reward_metric": metric.value,
                    "reward_data_split": cfg.reward_data_split,
                    "predictive_estimator": cfg.predictive_estimator,
                    **diagnostics,
                },
            )
        return ScoreResult(
            reward_train=float(marginal_ll),
            outcome_code="ok",
            outcome_detail=None,
            metric_log_marginal_likelihood=float(marginal_ll),
            decomposition={
                "reward_metric": metric.value,
                "reward_data_split": cfg.reward_data_split,
                "predictive_estimator": cfg.predictive_estimator,
                **diagnostics,
            },
        )

    if _has_potential(pymc_model):
        # pointwise estimators can't account for global potentials
        return ScoreResult(
            reward_train=EXEC_FAIL_REWARD,
            outcome_code="estimator_inapplicable",
            outcome_detail="pm.Potential present; pointwise predictive estimator invalid",
            decomposition={
                "reward_metric": metric.value,
                "reward_data_split": cfg.reward_data_split,
                "predictive_estimator": cfg.predictive_estimator,
            },
        )

    import arviz as az
    import pymc as pm

    with pymc_model:
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

    if metric is RewardMetric.ELPD:
        loo_res = az.loo(idata, pointwise=False)
        elpd = float(loo_res.elpd_loo)
        return ScoreResult(
            reward_train=elpd,
            outcome_code="ok",
            outcome_detail=None,
            metric_elpd=elpd,
            decomposition={
                "reward_metric": metric.value,
                "reward_data_split": cfg.reward_data_split,
                "predictive_estimator": cfg.predictive_estimator,
            },
        )

    if metric is RewardMetric.WAIC:
        waic_res = az.waic(idata, pointwise=False)
        elpd_waic = float(waic_res.elpd_waic)
        return ScoreResult(
            reward_train=elpd_waic,
            outcome_code="ok",
            outcome_detail=None,
            metric_waic=elpd_waic,
            decomposition={
                "reward_metric": metric.value,
                "reward_data_split": cfg.reward_data_split,
                "predictive_estimator": cfg.predictive_estimator,
            },
        )

    ll = _observed_loglik_sum(pymc_model)
    n = int(np.asarray(split_data["y"]).size)
    k = _count_effective_params(pymc_model)
    bic = k * math.log(max(n, 1)) - 2.0 * ll
    return ScoreResult(
        reward_train=-bic,
        outcome_code="ok",
        outcome_detail=None,
        metric_bic=float(bic),
        decomposition={
            "reward_metric": metric.value,
            "reward_data_split": cfg.reward_data_split,
            "predictive_estimator": cfg.predictive_estimator,
            "n_obs": n,
            "n_params": k,
            "loglik_obs_sum": ll,
        },
    )
