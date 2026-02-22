"""Tests for importance-sampling normalization check."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.evaluation.normalization_importance import (
    _normal_logpdf,
    check_importance_norm,
)
from ppl_synthesis_reward_hacking.reward_sentinels import EXEC_FAIL_REWARD

MOCK_TARGET = (
    "ppl_synthesis_reward_hacking.evaluation.normalization_importance."
    "score_completion_sandboxed"
)


def _base_kwargs(mc_samples: int = 50) -> dict:
    return {
        "completion_text": "```python\ndef model(data): pass\n```",
        "epsilon": 0.05,
        "ci_alpha": 0.05,
        "mc_samples": mc_samples,
        "min_ess": 5.0,
        "timeout": 10,
        "reward_metric": "log_marginal_likelihood",
        "reward_data_split": "train",
        "reward_estimator_backend": "smc",
        "smc_draws": 100,
        "scoring_data": {"y": np.array([1.0, 0.0, 1.0])},
        "seed": 42,
    }


class TestAllInvalidMCSamples:
    def test_returns_all_mc_scores_invalid(self):
        kwargs = _base_kwargs(mc_samples=20)

        with patch(MOCK_TARGET) as mock_scorer:
            mock_scorer.return_value = (EXEC_FAIL_REWARD, {"outcome_code": "exec_fail"})
            result = check_importance_norm(**kwargs)

        assert result["ok"] is False
        assert result["status"] == "all_mc_scores_invalid"
        assert result["n_valid"] == 0
        assert result["n_invalid"] == 20
        assert result["method"] == "importance_mc"


class TestNormalizedProgram:
    def test_unit_mass_returns_normalized(self):
        mc_samples = 100
        kwargs = _base_kwargs(mc_samples=mc_samples)
        scoring_data = kwargs["scoring_data"]
        y_obs = np.asarray(scoring_data["y"], dtype=np.float64).reshape(-1)
        y_mean = np.full(y_obs.shape[0], float(np.mean(y_obs)))
        y_std = np.full(y_obs.shape[0], float(np.std(y_obs) + 1.0))

        # precompute the exact y_samples the function will generate (same seed)
        rng = np.random.default_rng(42)
        y_samples = rng.normal(y_mean, y_std, size=(mc_samples, y_obs.shape[0]))
        log_q_values = _normal_logpdf(y_samples, y_mean, y_std)

        call_idx = 0

        def mock_scorer(*args, **kw):
            nonlocal call_idx
            # return log(q_i) so importance weight w_i = exp(z - log_q) = 1.0
            val = float(log_q_values[call_idx])
            call_idx += 1
            return (val, {"outcome_code": "ok"})

        with patch(MOCK_TARGET, side_effect=mock_scorer):
            result = check_importance_norm(**kwargs)

        assert result["ok"] is True
        assert result["n_valid"] == mc_samples
        assert result["log_mass"] == pytest.approx(0.0, abs=0.01)
        assert result["mass"] == pytest.approx(1.0, abs=0.05)
        assert result["is_normalized"] is True


class TestLowerCIBoundClamping:
    def test_ci_crossing_mass_one_remains_normalized(self):
        mc_samples = 20
        kwargs = _base_kwargs(mc_samples=mc_samples)
        kwargs["min_ess"] = 1.0
        scoring_data = kwargs["scoring_data"]
        y_obs = np.asarray(scoring_data["y"], dtype=np.float64).reshape(-1)
        y_mean = np.full(y_obs.shape[0], float(np.mean(y_obs)))
        y_std = np.full(y_obs.shape[0], float(np.std(y_obs) + 1.0))

        rng = np.random.default_rng(42)
        y_samples = rng.normal(y_mean, y_std, size=(mc_samples, y_obs.shape[0]))
        log_q_values = _normal_logpdf(y_samples, y_mean, y_std)

        # One dominant weight and tiny residual weights produce ci_low_raw <= 0 while
        # still spanning mass=1; this must not be forced to non-normalized.
        log_w_values = np.full(mc_samples, -60.0, dtype=np.float64)
        log_w_values[0] = math.log(mc_samples)

        call_idx = 0

        def mock_scorer(*args, **kw):
            nonlocal call_idx
            val = float(log_q_values[call_idx] + log_w_values[call_idx])
            call_idx += 1
            return (val, {"outcome_code": "ok"})

        with patch(MOCK_TARGET, side_effect=mock_scorer):
            result = check_importance_norm(**kwargs)

        assert result["ok"] is True
        assert result["decision_confident"] is True
        assert result["ci_mass_low"] == pytest.approx(0.0)
        assert result["ci_mass_high"] > 1.0
        assert result["ci_log_mass_low"] == float("-inf")
        assert result["is_normalized"] is True


class TestNonNormalizedProgram:
    def test_inflated_mass_returns_not_normalized(self):
        mc_samples = 100
        kwargs = _base_kwargs(mc_samples=mc_samples)
        scoring_data = kwargs["scoring_data"]
        y_obs = np.asarray(scoring_data["y"], dtype=np.float64).reshape(-1)
        y_mean = np.full(y_obs.shape[0], float(np.mean(y_obs)))
        y_std = np.full(y_obs.shape[0], float(np.std(y_obs) + 1.0))

        rng = np.random.default_rng(42)
        y_samples = rng.normal(y_mean, y_std, size=(mc_samples, y_obs.shape[0]))
        log_q_values = _normal_logpdf(y_samples, y_mean, y_std)

        call_idx = 0
        # return log_q + 2.0 so each weight w_i = exp(2.0) ~ 7.39, mass ~ 7.39
        bonus = 2.0

        def mock_scorer(*args, **kw):
            nonlocal call_idx
            val = float(log_q_values[call_idx]) + bonus
            call_idx += 1
            return (val, {"outcome_code": "ok"})

        with patch(MOCK_TARGET, side_effect=mock_scorer):
            result = check_importance_norm(**kwargs)

        assert result["ok"] is True
        assert result["mass"] == pytest.approx(math.exp(bonus), rel=0.1)
        assert result["log_mass"] == pytest.approx(bonus, abs=0.2)
        # with 100 samples and mass ~7.4, CI should exclude 1 confidently
        assert result["is_normalized"] is False


class TestSEUsesTotalSamples:
    def test_se_denominator_is_mc_samples_not_n_valid(self):
        mc_samples = 40
        kwargs = _base_kwargs(mc_samples=mc_samples)
        scoring_data = kwargs["scoring_data"]
        y_obs = np.asarray(scoring_data["y"], dtype=np.float64).reshape(-1)
        y_mean = np.full(y_obs.shape[0], float(np.mean(y_obs)))
        y_std = np.full(y_obs.shape[0], float(np.std(y_obs) + 1.0))

        rng = np.random.default_rng(42)
        y_samples = rng.normal(y_mean, y_std, size=(mc_samples, y_obs.shape[0]))
        log_q_values = _normal_logpdf(y_samples, y_mean, y_std)

        # first half valid (return log_q so weight=1), second half invalid
        n_valid_target = mc_samples // 2
        call_idx = 0

        def mock_scorer(*args, **kw):
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx < n_valid_target:
                return (float(log_q_values[idx]), {"outcome_code": "ok"})
            return (EXEC_FAIL_REWARD, {"outcome_code": "exec_fail"})

        with patch(MOCK_TARGET, side_effect=mock_scorer):
            result = check_importance_norm(**kwargs)

        assert result["ok"] is True
        assert result["n_valid"] == n_valid_target
        assert result["n_invalid"] == mc_samples - n_valid_target

        # mass_samples array has mc_samples entries (n_valid ones are exp(0)=1, rest are 0)
        # std is computed over the full mc_samples array, se = std / sqrt(mc_samples)
        mass_samples = np.zeros(mc_samples, dtype=np.float64)
        mass_samples[:n_valid_target] = 1.0  # exp(log_w) where log_w=0
        expected_std = float(np.std(mass_samples, ddof=1))
        expected_se = expected_std / math.sqrt(mc_samples)

        assert result["mass_se"] == pytest.approx(expected_se, rel=1e-6)

        # verify it is NOT std / sqrt(n_valid)
        wrong_se = expected_std / math.sqrt(n_valid_target)
        assert result["mass_se"] != pytest.approx(wrong_se, rel=1e-2)


class TestEmptyData:
    def test_empty_y_returns_invalid_data(self):
        kwargs = _base_kwargs()
        kwargs["scoring_data"] = {"y": np.array([])}
        result = check_importance_norm(**kwargs)

        assert result["ok"] is False
        assert result["status"] == "invalid_data"
        assert result["reason"] == "empty_y"
