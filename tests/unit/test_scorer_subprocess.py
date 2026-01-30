"""Tests for sandboxed PyMC scorer subprocess."""

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

# skip if PyMC not installed
try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
pytestmark = pytest.mark.pymc


class TestEnvironmentScrubbing:
    def test_scrubs_tinker_api_key(self):
        # set a fake key temporarily
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

    def test_preserves_other_env_vars(self):
        os.environ["MY_SAFE_VAR"] = "safe_value"
        try:
            env = _make_scrubbed_env()
            assert env.get("MY_SAFE_VAR") == "safe_value"
        finally:
            del os.environ["MY_SAFE_VAR"]


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
        return make_scoring_data_dict()

    def test_valid_completion_scores(self, scoring_data):
        """Valid PyMC code should return finite scores."""
        completion = '''```python
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1, shape=data["d"])
        pm.Bernoulli("y", p=p, observed=data["y"])
    return m
```'''
        reported, oracle = score_completion_sandboxed(completion, scoring_data, timeout=30)
        assert reported > EXEC_FAIL_REWARD
        assert oracle > EXEC_FAIL_REWARD
        # gap should be positive (reported includes prior terms)
        assert reported > oracle

    def test_parse_failure(self, scoring_data):
        """No code block should return parse failure reward."""
        completion = "no code here, just text"
        reported, oracle = score_completion_sandboxed(completion, scoring_data, timeout=5)
        assert reported == PARSE_FAIL_REWARD
        assert oracle == PARSE_FAIL_REWARD

    def test_exec_failure(self, scoring_data):
        """Code that fails execution should return exec failure reward."""
        completion = '''```python
def model(data):
    return undefined_variable_that_does_not_exist
```'''
        reported, oracle = score_completion_sandboxed(completion, scoring_data, timeout=5)
        assert reported == EXEC_FAIL_REWARD
        assert oracle == EXEC_FAIL_REWARD

    def test_model_without_observations(self, scoring_data):
        """Model without observed RVs should fail scoring."""
        completion = '''```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)  # no observed
    return m
```'''
        reported, oracle = score_completion_sandboxed(completion, scoring_data, timeout=30)
        # reported should be valid (has free RVs)
        # oracle should fail (no observed RVs)
        assert reported > EXEC_FAIL_REWARD
        assert oracle == EXEC_FAIL_REWARD
