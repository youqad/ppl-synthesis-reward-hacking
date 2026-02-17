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
