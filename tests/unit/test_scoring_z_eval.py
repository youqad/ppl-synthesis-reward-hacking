"""Tests for metric validation, split selection, and score payload behavior."""

from __future__ import annotations

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.scoring.metrics import (
    RewardMetric,
    validate_metric_estimator,
)
from ppl_synthesis_reward_hacking.scoring.result import ScoreResult
from ppl_synthesis_reward_hacking.scoring.z_eval import ZEvalConfig, select_split_data


class TestZEvalConfigValidation:
    def test_valid_log_marginal_likelihood_none(self):
        cfg = ZEvalConfig(
            reward_metric="log_marginal_likelihood",
            predictive_estimator="none",
        )
        cfg.validate()

    def test_valid_elpd_psis_loo(self):
        cfg = ZEvalConfig(
            reward_metric="elpd",
            predictive_estimator="psis_loo",
        )
        cfg.validate()

    def test_valid_waic_waic(self):
        cfg = ZEvalConfig(
            reward_metric="waic",
            predictive_estimator="waic",
        )
        cfg.validate()

    def test_valid_bic_bic_approx(self):
        cfg = ZEvalConfig(
            reward_metric="bic",
            predictive_estimator="bic_approx",
        )
        cfg.validate()

    @pytest.mark.parametrize(
        "metric,estimator",
        [
            ("log_marginal_likelihood", "psis_loo"),
            ("elpd", "none"),
            ("waic", "bic_approx"),
            ("bic", "waic"),
        ],
    )
    def test_invalid_combos_raise_value_error(self, metric, estimator):
        cfg = ZEvalConfig(
            reward_metric=metric,
            predictive_estimator=estimator,
        )
        with pytest.raises(ValueError):
            cfg.validate()


class TestValidateMetricEstimator:
    def test_valid_combo_passes(self):
        validate_metric_estimator("log_marginal_likelihood", "none")

    @pytest.mark.parametrize(
        "metric,estimator",
        [
            ("log_marginal_likelihood", "psis_loo"),
            ("elpd", "none"),
            ("waic", "bic_approx"),
            ("bic", "waic"),
        ],
    )
    def test_invalid_combos_raise(self, metric, estimator):
        with pytest.raises(ValueError):
            validate_metric_estimator(metric, estimator)

    def test_bad_metric_raises(self):
        with pytest.raises(ValueError):
            validate_metric_estimator("bogus", "none")

    def test_bad_estimator_raises(self):
        with pytest.raises(ValueError):
            validate_metric_estimator("log_marginal_likelihood", "bogus")


class TestSelectSplitData:
    def _make_scoring_data(self):
        return {
            "y": np.array([1, 0, 1]),
            "y_train": np.array([1, 0]),
            "y_holdout": np.array([0, 1, 1, 0]),
            "p_true": 0.5,
        }

    def test_train_split_uses_y_train(self):
        data = self._make_scoring_data()
        result = select_split_data(data, "train")
        np.testing.assert_array_equal(result["y"], [1, 0])
        assert result["d"] == 2

    def test_holdout_split_uses_y_holdout(self):
        data = self._make_scoring_data()
        result = select_split_data(data, "holdout")
        np.testing.assert_array_equal(result["y"], [0, 1, 1, 0])
        assert result["d"] == 4

    def test_train_falls_back_to_raw_y(self):
        data = {"y": np.array([1, 0, 1])}
        result = select_split_data(data, "train")
        np.testing.assert_array_equal(result["y"], [1, 0, 1])
        assert result["d"] == 3

    def test_holdout_falls_back_to_raw_y(self):
        data = {"y": np.array([0, 1, 0])}
        result = select_split_data(data, "holdout")
        np.testing.assert_array_equal(result["y"], [0, 1, 0])
        assert result["d"] == 3

    def test_missing_data_raises(self):
        with pytest.raises(ValueError, match="missing"):
            select_split_data({}, "train")

    def test_invalid_split_raises(self):
        data = {"y": np.array([1])}
        with pytest.raises(ValueError):
            select_split_data(data, "invalid_split")


class TestScoreResult:
    def test_to_payload_serialization(self):
        result = ScoreResult(
            reward_train=-2.5,
            outcome_code="ok",
            outcome_detail=None,
            metric_log_marginal_likelihood=-2.5,
            decomposition={"smc_draws": 500},
        )
        payload = result.to_payload()
        assert payload["reward_train"] == -2.5
        assert payload["outcome_code"] == "ok"
        assert payload["outcome_detail"] is None
        assert payload["metric_log_marginal_likelihood"] == -2.5
        assert payload["metric_elpd"] is None
        assert payload["metric_waic"] is None
        assert payload["metric_bic"] is None
        assert payload["decomposition"]["smc_draws"] == 500


class TestRewardMetricEnum:
    def test_all_values(self):
        assert set(RewardMetric) == {
            RewardMetric.LOG_MARGINAL_LIKELIHOOD,
            RewardMetric.ELPD,
            RewardMetric.WAIC,
            RewardMetric.BIC,
        }

    def test_string_conversion(self):
        assert str(RewardMetric.LOG_MARGINAL_LIKELIHOOD) == "log_marginal_likelihood"
        assert RewardMetric("elpd") is RewardMetric.ELPD
