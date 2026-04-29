from __future__ import annotations

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.experiments.stan_linear_reward import (
    _BatchStats,
    _build_point,
    _check_minimal_interface,
    _hash_normalized_code_for_diversity,
    _normalize_task,
    _select_normalization_targets,
    _task_to_stan_payload,
)


def test_check_minimal_interface_accepts_sampling_scalar_contract() -> None:
    code = """
data {
  int<lower=1> N;
  vector[N] X;
  vector[N] y;
}
parameters {
  real beta;
}
model {
  beta ~ normal(0, 1);
  y ~ normal(beta * X, 1);
}
"""
    ok, reason = _check_minimal_interface(code)
    assert ok is True
    assert reason is None


def test_check_minimal_interface_rejects_predictive_scaffold() -> None:
    code = """
data {
  int<lower=1> N_train;
  int<lower=1> N_test;
  vector[N_train] X_train;
  vector[N_train] y_train;
  vector[N_test] X_test;
  vector[N_test] y_test;
}
parameters {
  real beta;
}
model {
  y_train ~ normal(beta * X_train, 1);
}
"""
    ok, reason = _check_minimal_interface(code)
    assert ok is False
    assert reason == "disallowed_interface_token:N_test"


def test_check_minimal_interface_accepts_explicit_lpdf_contract() -> None:
    code = """
data {
  int<lower=1> N;
  vector[N] X;
  vector[N] y;
}
parameters {
  real beta;
}
model {
  target += normal_lpdf(beta | 0, 1);
  target += normal_lpdf(y | beta * X, 1);
}
"""
    ok, reason = _check_minimal_interface(code)
    assert ok is True
    assert reason is None


def test_check_minimal_interface_allows_lupdf_as_unsafe_surface() -> None:
    code = """
data {
  int<lower=1> N;
  vector[N] X;
  vector[N] y;
}
parameters {
  real beta;
}
model {
  target += normal_lupdf(beta | 0, 1);
  target += normal_lpdf(y | beta * X, 1);
}
"""
    ok, reason = _check_minimal_interface(code)
    assert ok is True
    assert reason is None


def test_task_to_stan_payload_uses_vector_interface() -> None:
    task = _normalize_task(
        {
            "seed": 7,
            "X": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "y": np.array([0.5, -0.3, 1.2], dtype=np.float64),
        }
    )
    payload = _task_to_stan_payload(task)
    assert payload == {
        "N": 3,
        "X": [1.0, 2.0, 3.0],
        "y": [0.5, -0.3, 1.2],
    }


def test_select_normalization_targets_supports_full_batch_sentinel() -> None:
    targets = ["a", "b", "c", "d"]

    assert _select_normalization_targets(targets, sample_size=-1) == targets
    assert _select_normalization_targets(targets, sample_size=2) == ["a", "b"]
    assert _select_normalization_targets(targets, sample_size=0) == []


def test_normalized_code_hash_ignores_comments_and_whitespace() -> None:
    left = """
model {
  // explanatory comment
  y ~ normal(beta * X, 1);
}
"""
    right = "model { y ~ normal(beta * X, 1); }"

    assert _hash_normalized_code_for_diversity(left) == _hash_normalized_code_for_diversity(right)


def test_build_point_reports_signed_lh_reward_split() -> None:
    stats = _BatchStats(
        rewards=[3.0, 1.0, -5.0],
        outcomes=["valid", "valid", "valid"],
        n_norm_checked=3,
        n_norm_with_log_mass=3,
        n_positive_lh=2,
        n_negative_lh=1,
        norm_abs_log_masses=[0.3, 0.7, 1.2],
        norm_log_masses=[0.2, -0.1, -0.7, 1.2],
        norm_program_mean_log_masses=[0.1, -0.7, 1.2],
        norm_program_max_log_masses=[0.2, -0.7, 1.2],
        norm_program_min_log_masses=[-0.1, -0.7, 1.2],
        positive_lh_rewards=[3.0, -5.0],
        non_positive_lh_rewards=[1.0],
        negative_lh_rewards=[1.0],
        non_negative_lh_rewards=[3.0, -5.0],
    )

    point = _build_point(1, stats)

    assert point.mean_log_mass == pytest.approx(0.15)
    assert point.max_log_mass == pytest.approx(1.2)
    assert point.min_log_mass == pytest.approx(-0.7)
    assert point.mean_program_mean_log_mass == pytest.approx(0.2)
    assert point.mean_program_max_log_mass == pytest.approx(7 / 30)
    assert point.mean_program_min_log_mass == pytest.approx(2 / 15)
    assert point.n_positive_lh == 2
    assert point.frac_positive_lh == pytest.approx(2 / 3)
    assert point.n_negative_lh == 1
    assert point.frac_negative_lh == pytest.approx(1 / 3)
    assert point.positive_lh_reward_mean == pytest.approx(-1.0)
    assert point.non_positive_lh_reward_mean == pytest.approx(1.0)
    assert point.positive_lh_reward_lift == pytest.approx(-2.0)
    assert point.negative_lh_reward_mean == pytest.approx(1.0)
    assert point.non_negative_lh_reward_mean == pytest.approx(-1.0)
    assert point.negative_lh_reward_lift == pytest.approx(2.0)
