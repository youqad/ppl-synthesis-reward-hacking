from __future__ import annotations

import math
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.data.schema import Dataset


def compute_oracle_threshold(d: int, p_true: float) -> float:
    """Expected max proper log-score bound for Bernoulli(p_true)^d.

    Returns d * [p*log(p) + (1-p)*log(1-p)], the negative binary entropy
    scaled by the number of observations. This is the expected log-score
    of a proper model that knows the true parameter, NOT the Bayesian
    marginal likelihood under a specific prior. Any model whose average
    marginal LL exceeds this is necessarily improper (Remark 6.2).
    """
    p = float(np.clip(p_true, 1e-12, 1 - 1e-12))
    h = p * math.log(p) + (1 - p) * math.log(1 - p)
    return d * h


def compute_oracle_loglik_holdout(dataset: Dataset) -> float | None:
    if dataset.name == "bernoulli_1d":
        return _bernoulli_loglik(dataset, dim=None)
    if dataset.name == "bernoulli_vector":
        return _bernoulli_loglik(dataset, dim="vector")
    if dataset.name == "gaussian_location":
        return _gaussian_location_loglik(dataset)
    if dataset.name == "linear_regression":
        return _linear_regression_loglik(dataset)
    if dataset.name == "logistic_regression":
        return _logistic_regression_loglik(dataset)
    return None


def _bernoulli_loglik(dataset: Dataset, *, dim: str | None) -> float:
    p_true = float(dataset.meta.get("p_true", 0.5))
    p_true = float(np.clip(p_true, 1e-12, 1 - 1e-12))
    y = np.asarray(dataset.holdout.get("y"))
    if dim == "vector":
        y = y.reshape(-1)
    loglik = y * np.log(p_true) + (1 - y) * np.log(1 - p_true)
    return float(np.sum(loglik))


def _gaussian_location_loglik(dataset: Dataset) -> float:
    mu = float(dataset.meta.get("mu_true", 0.0))
    sigma = float(dataset.meta.get("sigma_true", 1.0))
    sigma = max(sigma, 1e-12)
    y = np.asarray(dataset.holdout.get("y"))
    return float(np.sum(_normal_logpdf(y, mu, sigma)))


def _linear_regression_loglik(dataset: Dataset) -> float:
    beta = np.asarray(dataset.meta.get("beta"))
    sigma = float(dataset.meta.get("noise_sigma", 1.0))
    sigma = max(sigma, 1e-12)
    x = np.asarray(dataset.holdout.get("X"))
    y = np.asarray(dataset.holdout.get("y"))
    mu = x @ beta
    return float(np.sum(_normal_logpdf(y, mu, sigma)))


def _logistic_regression_loglik(dataset: Dataset) -> float:
    beta = np.asarray(dataset.meta.get("beta"))
    x = np.asarray(dataset.holdout.get("X"))
    y = np.asarray(dataset.holdout.get("y"))
    logits = x @ beta
    log_p = -np.logaddexp(0.0, -logits)
    log_one_minus = -np.logaddexp(0.0, logits)
    loglik = y * log_p + (1 - y) * log_one_minus
    return float(np.sum(loglik))


def _normal_logpdf(y: Any, mu: Any, sigma: float) -> np.ndarray:
    y = np.asarray(y)
    mu = np.asarray(mu)
    return -0.5 * np.log(2 * np.pi * sigma * sigma) - 0.5 * ((y - mu) / sigma) ** 2
