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


class RewardEstimatorBackend(StrEnum):
    SMC = "smc"
    PSIS_LOO = "psis_loo"
    WAIC = "waic"
    BIC_APPROX = "bic_approx"


def validate_metric_estimator(reward_metric: str, reward_estimator_backend: str) -> None:
    """Raise ValueError if metric and estimator backend are incompatible."""
    metric = RewardMetric(reward_metric)
    estimator = RewardEstimatorBackend(reward_estimator_backend)

    if (
        metric is RewardMetric.LOG_MARGINAL_LIKELIHOOD
        and estimator is not RewardEstimatorBackend.SMC
    ):
        raise ValueError("log_marginal_likelihood requires reward_estimator_backend=smc")
    if metric is RewardMetric.ELPD and estimator is not RewardEstimatorBackend.PSIS_LOO:
        raise ValueError("elpd requires reward_estimator_backend=psis_loo")
    if metric is RewardMetric.WAIC and estimator is not RewardEstimatorBackend.WAIC:
        raise ValueError("waic requires reward_estimator_backend=waic")
    if metric is RewardMetric.BIC and estimator is not RewardEstimatorBackend.BIC_APPROX:
        raise ValueError("bic requires reward_estimator_backend=bic_approx")
