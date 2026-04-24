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
