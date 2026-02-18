"""Tests for shared trajectory metrics and JSON serialization."""

from dataclasses import dataclass

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.experiments.results import (
    compute_traj_metrics,
    json_default,
    print_training_summary,
)


@dataclass
class FakeTP:
    reward_mean: float
    n_valid: int
    n_total: int
    frac_non_normalized: float = 0.0
    mean_abs_log_mass: float = 0.0
    n_norm_checked: int = 0


class TestComputeTrajectoryMetrics:
    def test_normal_trajectory(self):
        t = [
            FakeTP(-5.0, 8, 10, frac_non_normalized=0.0, n_norm_checked=10),
            FakeTP(-2.0, 9, 10, frac_non_normalized=0.0, n_norm_checked=10),
        ]
        m = compute_traj_metrics(t)
        assert m["initial_reward"] == -5.0
        assert m["final_reward"] == -2.0
        assert m["reward_increase"] == pytest.approx(3.0)
        assert m["final_valid_rate"] == pytest.approx(0.9)
        assert m["lh_detected"] is False

    def test_empty_trajectory(self):
        m = compute_traj_metrics([])
        assert "error" in m

    def test_single_element(self):
        m = compute_traj_metrics([FakeTP(0, 0, 0)])
        assert "error" in m

    def test_nan_reward(self):
        t = [FakeTP(float("nan"), 1, 1), FakeTP(-2.0, 1, 1)]
        m = compute_traj_metrics(t)
        assert np.isnan(m["reward_increase"])
        assert m["lh_detected"] is False

    def test_both_nan(self):
        t = [FakeTP(float("nan"), 1, 1), FakeTP(float("nan"), 1, 1)]
        m = compute_traj_metrics(t)
        assert np.isnan(m["reward_increase"])
        assert m["lh_detected"] is False

    def test_zero_total(self):
        t = [FakeTP(0.0, 0, 0), FakeTP(1.0, 0, 0)]
        m = compute_traj_metrics(t)
        assert m["final_valid_rate"] == 0.0

    def test_lh_detected(self):
        t = [
            FakeTP(-5.0, 5, 10, frac_non_normalized=0.0, n_norm_checked=10),
            FakeTP(-2.0, 5, 10, frac_non_normalized=0.5, n_norm_checked=10),
        ]
        m = compute_traj_metrics(t)
        assert m["lh_detected"] is True


class TestJsonDefault:
    def test_numpy_integer(self):
        assert json_default(np.int64(42)) == 42
        assert isinstance(json_default(np.int64(42)), int)

    def test_numpy_float(self):
        assert json_default(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(json_default(np.float64(3.14)), float)

    def test_numpy_array(self):
        result = json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_fallback(self):
        assert json_default(object()) is not None


class TestPrintTrainingSummary:
    def test_error_results(self, caplog):
        with caplog.at_level("INFO"):
            print_training_summary({"error": "not enough data"})
        assert "not enough data" in caplog.text

    def test_normal_results(self, caplog):
        results = {
            "initial_reward": -5.0,
            "final_reward": -2.0,
            "reward_increase": 3.0,
            "final_valid_rate": 0.9,
            "lh_detected": False,
            "final_frac_non_normalized": 0.0,
            "final_mean_abs_log_mass": 0.0,
            "final_n_norm_checked": 10,
        }
        with caplog.at_level("INFO"):
            print_training_summary(results)
        assert "reward:" in caplog.text
        assert "lh_detected" in caplog.text
