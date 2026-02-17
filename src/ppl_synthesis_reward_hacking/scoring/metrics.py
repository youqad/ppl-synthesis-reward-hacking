from __future__ import annotations

from enum import StrEnum


class RewardMetric(StrEnum):
    LOG_MARGINAL_LIKELIHOOD = "log_marginal_likelihood"
    ELPD = "elpd"
    WAIC = "waic"
    BIC = "bic"


class RewardDataSplit(StrEnum):
    TRAIN = "train"
    HOLDOUT = "holdout"


class PredictiveEstimator(StrEnum):
    NONE = "none"
    PSIS_LOO = "psis_loo"
    WAIC = "waic"
    BIC_APPROX = "bic_approx"


def validate_metric_estimator(metric: str, estimator: str) -> None:
    m = RewardMetric(metric)
    e = PredictiveEstimator(estimator)
    if m is RewardMetric.LOG_MARGINAL_LIKELIHOOD and e is not PredictiveEstimator.NONE:
        raise ValueError("log_marginal_likelihood requires predictive_estimator=none")
    if m is RewardMetric.ELPD and e is not PredictiveEstimator.PSIS_LOO:
        raise ValueError("elpd requires predictive_estimator=psis_loo")
    if m is RewardMetric.WAIC and e is not PredictiveEstimator.WAIC:
        raise ValueError("waic requires predictive_estimator=waic")
    if m is RewardMetric.BIC and e is not PredictiveEstimator.BIC_APPROX:
        raise ValueError("bic requires predictive_estimator=bic_approx")
