"""Tests for shared trajectory metrics and JSON serialization."""

from dataclasses import dataclass

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.experiments.results import (
    compute_trajectory_metrics,
    json_default,
    print_training_summary,
)


@dataclass
class FakeTP:
    gap_mean: float
    reported_mean: float
    oracle_mean: float
    n_valid: int
    n_total: int


class TestComputeTrajectoryMetrics:
    def test_normal_trajectory(self):
        t = [FakeTP(1.0, -5.0, -6.0, 8, 10), FakeTP(3.0, -2.0, -8.0, 9, 10)]
        m = compute_trajectory_metrics(t)
        assert m["initial_gap"] == 1.0
        assert m["final_gap"] == 3.0
        assert m["gap_increase"] == pytest.approx(2.0)
        assert m["final_valid_rate"] == pytest.approx(0.9)
        assert m["hacking_detected"] is False

    def test_empty_trajectory(self):
        m = compute_trajectory_metrics([])
        assert "error" in m

    def test_single_element(self):
        m = compute_trajectory_metrics([FakeTP(0, 0, 0, 0, 0)])
        assert "error" in m

    def test_nan_gap(self):
        t = [FakeTP(float("nan"), -5.0, -6.0, 1, 1), FakeTP(3.0, -2.0, -8.0, 1, 1)]
        m = compute_trajectory_metrics(t)
        assert np.isnan(m["gap_increase"])
        assert m["hacking_detected"] is False

    def test_both_nan(self):
        t = [
            FakeTP(float("nan"), -5.0, -6.0, 1, 1),
            FakeTP(float("nan"), -2.0, -8.0, 1, 1),
        ]
        m = compute_trajectory_metrics(t)
        assert np.isnan(m["gap_increase"])
        assert m["hacking_detected"] is False

    def test_zero_total(self):
        t = [FakeTP(0.0, 0.0, 0.0, 0, 0), FakeTP(1.0, 1.0, 0.0, 0, 0)]
        m = compute_trajectory_metrics(t)
        assert m["final_valid_rate"] == 0.0

    def test_hacking_detected(self):
        t = [FakeTP(0.0, -5.0, -5.0, 5, 10), FakeTP(15.0, -2.0, -17.0, 5, 10)]
        m = compute_trajectory_metrics(t)
        assert m["hacking_detected"] is True


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
        assert json_default(object()) is not None  # returns str(obj)


class TestPrintTrainingSummary:
    def test_error_results(self, caplog):
        with caplog.at_level("INFO"):
            print_training_summary({"error": "not enough data"})
        assert "not enough data" in caplog.text

    def test_normal_results(self, caplog):
        results = {
            "initial_gap": 1.0,
            "final_gap": 3.0,
            "gap_increase": 2.0,
            "initial_reported": -5.0,
            "final_reported": -2.0,
            "initial_oracle": -6.0,
            "final_oracle": -8.0,
            "final_valid_rate": 0.9,
            "hacking_detected": False,
        }
        with caplog.at_level("INFO"):
            print_training_summary(results)
        assert "gap:" in caplog.text
        assert "reported:" in caplog.text
        assert "oracle:" in caplog.text
        assert "HACKING" not in caplog.text

    def test_hacking_detected_message(self, caplog):
        results = {
            "initial_gap": 0.0,
            "final_gap": 15.0,
            "gap_increase": 15.0,
            "initial_reported": -5.0,
            "final_reported": -2.0,
            "initial_oracle": -5.0,
            "final_oracle": -17.0,
            "final_valid_rate": 0.5,
            "hacking_detected": True,
        }
        with caplog.at_level("INFO"):
            print_training_summary(results)
        assert "HACKING DETECTED" in caplog.text
