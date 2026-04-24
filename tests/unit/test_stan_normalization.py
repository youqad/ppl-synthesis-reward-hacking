"""Tests for direct-Stan importance-sampling normalization."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.evaluation.stan_normalization import (
    _build_stan_data,
    _mvnormal_logpdf,
    _normal_logpdf,
    check_stan_importance_norm,
    check_stan_predictive_importance_norm,
    honest_posterior_predictive_log_density,
)
from ppl_synthesis_reward_hacking.reward_sentinels import EXEC_FAIL_REWARD

MOCK_TARGET = (
    "ppl_synthesis_reward_hacking.evaluation.stan_normalization._evaluate_reported_densities"
)


def _base_kwargs(mc_samples: int = 50) -> dict:
    return {
        "stan_code": "data { int N; int K; matrix[N, K] X; vector[N] y; } model {}",
        "scoring_data": {
            "X": np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
            "y": np.array([1.0, 0.0, 1.0], dtype=np.float64),
        },
        "runtime": object(),
        "protect": "y",
        "reward_output_field": "reported_log_density",
        "fallback_fields": (),
        "jobs": 4,
        "epsilon": 0.05,
        "ci_alpha": 0.05,
        "mc_samples": mc_samples,
        "min_ess": 5.0,
        "seed": 42,
    }


def _predictive_kwargs(mc_samples: int = 40) -> dict:
    return {
        "stan_code": (
            "data { int N_train; int N_test; int K; "
            "matrix[N_train, K] X_train; vector[N_train] y_train; "
            "matrix[N_test, K] X_test; vector[N_test] y_test; "
            "real sigma_obs; real beta_prior_scale; } model {}"
        ),
        "scoring_task": {
            "X_train": np.array([[1.0], [2.0], [-1.0]], dtype=np.float64),
            "y_train": np.array([0.4, 1.1, -0.8], dtype=np.float64),
            "X_test": np.array([[0.5], [1.5]], dtype=np.float64),
            "y_test": np.array([0.2, 0.9], dtype=np.float64),
            "sigma_obs": 1.0,
            "beta_prior_scale": 1.0,
        },
        "runtime": object(),
        "protect": ("y_train", "y_test"),
        "reward_output_field": "reported_log_density",
        "fallback_fields": (),
        "jobs": 4,
        "epsilon": 0.05,
        "ci_alpha": 0.05,
        "mc_samples": mc_samples,
        "min_ess": 5.0,
        "seed": 123,
    }


def test_build_stan_data_shapes() -> None:
    payload = _build_stan_data(
        x=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        y=np.array([0.5, -0.3], dtype=np.float64),
    )
    assert payload["N"] == 2
    assert payload["K"] == 2
    assert payload["X"] == [[1.0, 2.0], [3.0, 4.0]]
    assert payload["y"] == [0.5, -0.3]


def test_invalid_x_shape_returns_error() -> None:
    kwargs = _base_kwargs()
    kwargs["scoring_data"] = {"X": np.array([1.0, 2.0]), "y": np.array([1.0, 2.0])}
    result = check_stan_importance_norm(**kwargs)
    assert result["ok"] is False
    assert result["reason"] == "missing_or_invalid_X"


def test_all_invalid_mc_samples_returns_error() -> None:
    kwargs = _base_kwargs(mc_samples=20)

    with patch(MOCK_TARGET, return_value=[EXEC_FAIL_REWARD] * 20):
        result = check_stan_importance_norm(**kwargs)

    assert result["ok"] is False
    assert result["status"] == "all_mc_scores_invalid"
    assert result["n_valid"] == 0
    assert result["n_invalid"] == 20


def test_unit_mass_returns_normalized() -> None:
    mc_samples = 100
    kwargs = _base_kwargs(mc_samples=mc_samples)
    y_obs = np.asarray(kwargs["scoring_data"]["y"], dtype=np.float64).reshape(-1)
    y_mean = np.full(y_obs.shape[0], float(np.mean(y_obs)))
    y_std = np.full(y_obs.shape[0], float(np.std(y_obs) + 1.0))

    rng = np.random.default_rng(42)
    y_samples = rng.normal(y_mean, y_std, size=(mc_samples, y_obs.shape[0]))
    log_q_values = _normal_logpdf(y_samples, y_mean, y_std)

    with patch(MOCK_TARGET, return_value=[float(v) for v in log_q_values]):
        result = check_stan_importance_norm(**kwargs)

    assert result["ok"] is True
    assert result["n_valid"] == mc_samples
    assert result["log_mass"] == pytest.approx(0.0, abs=0.01)
    assert result["mass"] == pytest.approx(1.0, abs=0.05)
    assert result["is_normalized"] is True


def test_inflated_mass_returns_not_normalized() -> None:
    mc_samples = 100
    kwargs = _base_kwargs(mc_samples=mc_samples)
    y_obs = np.asarray(kwargs["scoring_data"]["y"], dtype=np.float64).reshape(-1)
    y_mean = np.full(y_obs.shape[0], float(np.mean(y_obs)))
    y_std = np.full(y_obs.shape[0], float(np.std(y_obs) + 1.0))

    rng = np.random.default_rng(42)
    y_samples = rng.normal(y_mean, y_std, size=(mc_samples, y_obs.shape[0]))
    log_q_values = _normal_logpdf(y_samples, y_mean, y_std)

    bonus = 1.5

    with patch(
        MOCK_TARGET,
        return_value=[float(v) + bonus for v in log_q_values],
    ):
        result = check_stan_importance_norm(**kwargs)

    assert result["ok"] is True
    assert result["mass"] == pytest.approx(math.exp(bonus), rel=0.1)
    assert result["log_mass"] == pytest.approx(bonus, abs=0.2)
    assert result["is_normalized"] is False


def test_reference_gaussian_proposal_has_unit_mass_for_matching_scores() -> None:
    mc_samples = 40
    kwargs = _base_kwargs(mc_samples=mc_samples)
    kwargs["scoring_data"] = {
        "X": np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
        "y": np.array([1.0, 0.0, 1.0], dtype=np.float64),
        "sigma_obs": 1.0,
        "beta_prior_scale": 1.0,
    }

    x_obs = np.asarray(kwargs["scoring_data"]["X"], dtype=np.float64)
    n = x_obs.shape[0]
    ref_cov = np.eye(n, dtype=np.float64) + x_obs @ x_obs.T
    ref_chol = np.linalg.cholesky(ref_cov)
    rng = np.random.default_rng(42)
    y_samples = rng.multivariate_normal(np.zeros(n, dtype=np.float64), ref_cov, size=mc_samples)
    log_q_values = _mvnormal_logpdf(y_samples, np.zeros(n, dtype=np.float64), ref_chol)

    with patch(MOCK_TARGET, return_value=[float(v) for v in log_q_values]):
        result = check_stan_importance_norm(**kwargs)

    assert result["ok"] is True
    assert result["n_valid"] == mc_samples
    assert result["log_mass"] == pytest.approx(0.0, abs=0.01)
    assert result["mass"] == pytest.approx(1.0, abs=0.01)
    assert result["ess"] == pytest.approx(float(mc_samples), abs=1e-6)


def test_honest_posterior_predictive_log_density_is_finite() -> None:
    value = honest_posterior_predictive_log_density(_predictive_kwargs()["scoring_task"])
    assert math.isfinite(value)


def test_predictive_reference_gaussian_proposal_has_unit_mass_for_matching_scores() -> None:
    mc_samples = 40
    kwargs = _predictive_kwargs(mc_samples=mc_samples)
    task = kwargs["scoring_task"]
    x_train = np.asarray(task["X_train"], dtype=np.float64)
    y_train = np.asarray(task["y_train"], dtype=np.float64).reshape(-1)
    x_test = np.asarray(task["X_test"], dtype=np.float64)
    sigma2 = float(task["sigma_obs"]) ** 2
    tau2 = float(task["beta_prior_scale"]) ** 2

    precision = (x_train.T @ x_train) / sigma2 + np.eye(x_train.shape[1], dtype=np.float64) / tau2
    posterior_cov = np.linalg.inv(precision)
    posterior_mean = posterior_cov @ (x_train.T @ y_train) / sigma2
    pred_mean = x_test @ posterior_mean
    pred_cov = (
        sigma2 * np.eye(x_test.shape[0], dtype=np.float64)
        + x_test @ posterior_cov @ x_test.T
    )
    pred_chol = np.linalg.cholesky(pred_cov)

    rng = np.random.default_rng(kwargs["seed"])
    y_samples = rng.multivariate_normal(pred_mean, pred_cov, size=mc_samples)
    log_q_values = _mvnormal_logpdf(y_samples, pred_mean, pred_chol)

    with patch(MOCK_TARGET, return_value=[float(v) for v in log_q_values]):
        result = check_stan_predictive_importance_norm(**kwargs)

    assert result["ok"] is True
    assert result["n_valid"] == mc_samples
    assert result["log_mass"] == pytest.approx(0.0, abs=0.01)
    assert result["mass"] == pytest.approx(1.0, abs=0.01)
    assert result["ess"] == pytest.approx(float(mc_samples), abs=1e-6)
