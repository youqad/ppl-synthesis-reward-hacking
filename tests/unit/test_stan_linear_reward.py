from __future__ import annotations

import numpy as np

from ppl_synthesis_reward_hacking.experiments.stan_linear_reward import (
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
