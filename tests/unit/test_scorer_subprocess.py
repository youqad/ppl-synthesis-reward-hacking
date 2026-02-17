"""Sandboxed PyMC scorer subprocess tests."""

from __future__ import annotations

import os

import pytest

from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    EXEC_FAIL_REWARD,
    PARSE_FAIL_REWARD,
    _make_scrubbed_env,
    _make_serializable,
    _restore_arrays,
    score_completion_sandboxed,
)

try:
    import pymc  # noqa: F401

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
pytestmark = pytest.mark.pymc


class TestEnvironmentScrubbing:
    def test_scrubs_tinker_api_key(self):
        os.environ["TINKER_API_KEY"] = "test_secret_key"
        try:
            env = _make_scrubbed_env()
            assert "TINKER_API_KEY" not in env
        finally:
            del os.environ["TINKER_API_KEY"]

    def test_scrubs_multiple_secrets(self):
        secrets = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"]
        for secret in secrets:
            os.environ[secret] = "test_value"
        try:
            env = _make_scrubbed_env()
            for secret in secrets:
                assert secret not in env
        finally:
            for secret in secrets:
                if secret in os.environ:
                    del os.environ[secret]

    def test_sets_pytensor_flags(self):
        env = _make_scrubbed_env()
        assert "PYTENSOR_FLAGS" in env
        assert "FAST_COMPILE" in env["PYTENSOR_FLAGS"]


class TestSerialization:
    def test_serializes_numpy_arrays(self):
        import numpy as np

        data = {"y": np.array([[1, 2], [3, 4]]), "n": 42}
        result = _make_serializable(data)
        assert isinstance(result["y"], list)
        assert result["y"] == [[1, 2], [3, 4]]
        assert result["n"] == 42

    def test_restores_arrays(self):
        import numpy as np

        data = {"y": [[1, 2], [3, 4]], "n": 42}
        result = _restore_arrays(data)
        assert isinstance(result["y"], np.ndarray)
        assert result["y"].shape == (2, 2)
        assert result["n"] == 42


@requires_pymc
class TestSandboxedScorer:
    @pytest.fixture
    def scoring_data(self):
        from ppl_synthesis_reward_hacking.data.synthstats_loader import make_scoring_data_dict

        data = make_scoring_data_dict(d=1, p_true=0.5, seed=0)
        data["y_train"] = data["y"]
        data["y_holdout"] = make_scoring_data_dict(d=1, p_true=0.5, seed=1)["y"]
        return data

    def test_parse_failure(self, scoring_data):
        completion = "no code here, just text"
        reported, oracle, decomposition = score_completion_sandboxed(
            completion, scoring_data, timeout=5
        )
        assert reported == PARSE_FAIL_REWARD
        assert oracle == PARSE_FAIL_REWARD
        assert decomposition is not None
        assert decomposition["outcome_code"] == "parse_fail"

    def test_exec_failure(self, scoring_data):
        completion = """```python
def model(data):
    return undefined_variable_that_does_not_exist
```"""
        reported, oracle, decomposition = score_completion_sandboxed(
            completion, scoring_data, timeout=5
        )
        assert reported == EXEC_FAIL_REWARD
        assert oracle == EXEC_FAIL_REWARD
        assert decomposition is not None
        assert decomposition["outcome_code"] == "exec_fail"

    def test_log_marginal_likelihood_scores(self, scoring_data):
        completion = """```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("y", p=p, observed=data["y"])
    return m
```"""
        reported, oracle, decomposition = score_completion_sandboxed(
            completion,
            scoring_data,
            timeout=30,
            reward_metric="log_marginal_likelihood",
            reward_data_split="train",
            predictive_estimator="none",
            smc_draws=50,
        )
        assert reported > EXEC_FAIL_REWARD
        assert oracle > EXEC_FAIL_REWARD
        assert decomposition is not None
        assert decomposition["outcome_code"] == "ok"
        assert decomposition["metric_log_marginal_likelihood"] is not None

    def test_invalid_metric_estimator_combo_fails(self, scoring_data):
        completion = """```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("y", p=p, observed=data["y"])
    return m
```"""
        reported, _oracle, decomposition = score_completion_sandboxed(
            completion,
            scoring_data,
            timeout=30,
            reward_metric="waic",
            reward_data_split="train",
            predictive_estimator="none",
        )
        assert reported == EXEC_FAIL_REWARD
        assert decomposition is not None
        assert decomposition["outcome_code"] == "score_dispatch_fail"

    def test_waic_with_potential_fails_closed(self, scoring_data):
        completion = """```python
import pymc as pm
import pytensor.tensor as pt

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("y", p=p, observed=data["y"])
        pm.Potential("baseline", pt.constant(5.0))
    return m
```"""
        reported, _oracle, decomposition = score_completion_sandboxed(
            completion,
            scoring_data,
            timeout=30,
            reward_metric="waic",
            reward_data_split="train",
            predictive_estimator="waic",
        )
        assert reported == EXEC_FAIL_REWARD
        assert decomposition is not None
        assert decomposition["outcome_code"] == "estimator_inapplicable"
