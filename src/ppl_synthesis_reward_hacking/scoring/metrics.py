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


def validate_metric_estimator(reward_metric: str, predictive_estimator: str) -> None:
    """Raise ValueError if metric and estimator are incompatible."""
    metric = RewardMetric(reward_metric)
    estimator = PredictiveEstimator(predictive_estimator)

    if metric is RewardMetric.LOG_MARGINAL_LIKELIHOOD and estimator is not PredictiveEstimator.NONE:
        raise ValueError("log_marginal_likelihood requires predictive_estimator=none")
    if metric is RewardMetric.ELPD and estimator is not PredictiveEstimator.PSIS_LOO:
        raise ValueError("elpd requires predictive_estimator=psis_loo")
    if metric is RewardMetric.WAIC and estimator is not PredictiveEstimator.WAIC:
        raise ValueError("waic requires predictive_estimator=waic")
    if metric is RewardMetric.BIC and estimator is not PredictiveEstimator.BIC_APPROX:
        raise ValueError("bic requires predictive_estimator=bic_approx")
