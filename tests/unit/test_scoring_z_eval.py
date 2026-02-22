"""Tests for metric validation, split selection, score payload, and evaluate_model dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.scoring.metrics import (
    RewardMetric,
    validate_metric_estimator,
)
from ppl_synthesis_reward_hacking.scoring.result import ScoreResult
from ppl_synthesis_reward_hacking.scoring.z_eval import (
    ZEvalConfig,
    evaluate_model,
    select_split_data,
)


class TestZEvalConfigValidation:
    def test_valid_log_marginal_likelihood_none(self):
        cfg = ZEvalConfig(
            reward_metric="log_marginal_likelihood",
            reward_estimator_backend="smc",
        )
        cfg.validate()

    def test_valid_elpd_psis_loo(self):
        cfg = ZEvalConfig(
            reward_metric="elpd",
            reward_estimator_backend="psis_loo",
        )
        cfg.validate()

    def test_valid_waic_waic(self):
        cfg = ZEvalConfig(
            reward_metric="waic",
            reward_estimator_backend="waic",
        )
        cfg.validate()

    def test_valid_bic_bic_approx(self):
        cfg = ZEvalConfig(
            reward_metric="bic",
            reward_estimator_backend="bic_approx",
        )
        cfg.validate()

    @pytest.mark.parametrize(
        "metric,estimator",
        [
            ("log_marginal_likelihood", "psis_loo"),
            ("elpd", "smc"),
            ("waic", "bic_approx"),
            ("bic", "waic"),
        ],
    )
    def test_invalid_combos_raise_value_error(self, metric, estimator):
        cfg = ZEvalConfig(
            reward_metric=metric,
            reward_estimator_backend=estimator,
        )
        with pytest.raises(ValueError):
            cfg.validate()


class TestValidateMetricEstimator:
    def test_valid_combo_passes(self):
        validate_metric_estimator("log_marginal_likelihood", "smc")

    @pytest.mark.parametrize(
        "metric,estimator",
        [
            ("log_marginal_likelihood", "psis_loo"),
            ("elpd", "smc"),
            ("waic", "bic_approx"),
            ("bic", "waic"),
        ],
    )
    def test_invalid_combos_raise(self, metric, estimator):
        with pytest.raises(ValueError):
            validate_metric_estimator(metric, estimator)

    def test_bad_metric_raises(self):
        with pytest.raises(ValueError):
            validate_metric_estimator("bogus", "smc")

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

    def test_holdout_without_holdout_key_raises(self):
        data = {"y": np.array([0, 1, 0])}
        with pytest.raises(ValueError, match="y_holdout"):
            select_split_data(data, "holdout")

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


class TestEvaluateModelDispatch:
    """Verify evaluate_model routes to the correct internal scorer."""

    @pytest.fixture
    def dummy_model(self):
        return MagicMock(name="pymc_model")

    @pytest.fixture
    def scoring_data(self):
        return {"y": np.array([1.0, 0.0, 1.0]), "p_true": 0.5}

    @pytest.fixture
    def logml_result(self):
        return ScoreResult(
            reward_train=-2.5,
            outcome_code="ok",
            outcome_detail=None,
            metric_log_marginal_likelihood=-2.5,
            decomposition={"reward_metric": "log_marginal_likelihood"},
        )

    @pytest.fixture
    def elpd_result(self):
        return ScoreResult(
            reward_train=-3.0,
            outcome_code="ok",
            outcome_detail=None,
            metric_elpd=-3.0,
            decomposition={"reward_metric": "elpd"},
        )

    @pytest.fixture
    def waic_result(self):
        return ScoreResult(
            reward_train=-4.0,
            outcome_code="ok",
            outcome_detail=None,
            metric_waic=-4.0,
            decomposition={"reward_metric": "waic"},
        )

    @pytest.fixture
    def bic_result(self):
        return ScoreResult(
            reward_train=-10.0,
            outcome_code="ok",
            outcome_detail=None,
            metric_bic=10.0,
            decomposition={"reward_metric": "bic"},
        )

    @patch("ppl_synthesis_reward_hacking.scoring.z_eval._score_log_marginal_likelihood")
    def test_dispatch_log_marginal_likelihood(
        self, mock_fn, dummy_model, scoring_data, logml_result
    ):
        mock_fn.return_value = logml_result
        cfg = ZEvalConfig(
            reward_metric="log_marginal_likelihood",
            reward_estimator_backend="smc",
        )
        result = evaluate_model(dummy_model, scoring_data, cfg)

        mock_fn.assert_called_once()
        assert result is logml_result
        assert result.metric_log_marginal_likelihood == -2.5

    @patch("ppl_synthesis_reward_hacking.scoring.z_eval._score_predictive_metric")
    def test_dispatch_elpd(self, mock_fn, dummy_model, scoring_data, elpd_result):
        mock_fn.return_value = elpd_result
        cfg = ZEvalConfig(
            reward_metric="elpd",
            reward_estimator_backend="psis_loo",
        )
        result = evaluate_model(dummy_model, scoring_data, cfg)

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["metric"] is RewardMetric.ELPD
        assert result is elpd_result

    @patch("ppl_synthesis_reward_hacking.scoring.z_eval._score_predictive_metric")
    def test_dispatch_waic(self, mock_fn, dummy_model, scoring_data, waic_result):
        mock_fn.return_value = waic_result
        cfg = ZEvalConfig(
            reward_metric="waic",
            reward_estimator_backend="waic",
        )
        result = evaluate_model(dummy_model, scoring_data, cfg)

        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["metric"] is RewardMetric.WAIC
        assert result is waic_result

    @patch("ppl_synthesis_reward_hacking.scoring.z_eval._score_bic")
    def test_dispatch_bic(self, mock_fn, dummy_model, scoring_data, bic_result):
        mock_fn.return_value = bic_result
        cfg = ZEvalConfig(
            reward_metric="bic",
            reward_estimator_backend="bic_approx",
        )
        result = evaluate_model(dummy_model, scoring_data, cfg)

        mock_fn.assert_called_once()
        assert result is bic_result

    def test_invalid_metric_estimator_combination(self, dummy_model, scoring_data):
        cfg = ZEvalConfig(
            reward_metric="elpd",
            reward_estimator_backend="smc",
        )
        with pytest.raises(ValueError, match="elpd requires"):
            evaluate_model(dummy_model, scoring_data, cfg)
