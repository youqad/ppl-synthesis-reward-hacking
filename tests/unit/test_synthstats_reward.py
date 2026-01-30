"""Tests for synthstats reward scoring function.

All tests in this module require PyMC to be installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# skip all tests if PyMC is not installed
pytest.importorskip("pymc")

# Mark the module for marker-based selection.
pytestmark = pytest.mark.pymc

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import execute_pymc_code  # noqa: E402
from ppl_synthesis_reward_hacking.experiments.synthstats_reward import (  # noqa: E402
    EXEC_FAIL_REWARD,
    _score_pymc_model,
)


class TestScorePyMCModel:
    @pytest.fixture
    def bernoulli_data(self):
        # 32x8 binary matrix (typical synthstats format)
        rng = np.random.default_rng(42)
        return {"y": rng.binomial(1, 0.3, size=(32, 8))}

    def test_simple_model_returns_valid_scores(self, bernoulli_data):
        """A basic model with one free RV and observed data should return valid scores."""
        code = """
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
"""
        pymc_model = execute_pymc_code(code, bernoulli_data)
        assert pymc_model is not None

        reported, oracle = _score_pymc_model(pymc_model)

        # both should be finite
        assert math.isfinite(reported)
        assert oracle is not None
        assert math.isfinite(oracle)
        # reported includes prior, oracle is just likelihood
        # with more free RVs, reported tends to be higher than oracle
        # (this is the hackable gap)

    def test_model_with_many_free_rvs_has_larger_gap(self, bernoulli_data):
        """Adding unnecessary free RVs should inflate reported vs oracle."""
        # honest model: 1 free RV
        honest_code = """
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
"""
        # inflated model: many free RVs that don't affect observations
        inflated_code = """
import pymc as pm

def model(data):
    with pm.Model() as m:
        # many free RVs with wide priors
        for i in range(10):
            pm.Normal(f"noise_{i}", 0, 10)
        p = pm.Beta("p", 1, 1)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
"""
        honest_model = execute_pymc_code(honest_code, bernoulli_data)
        inflated_model = execute_pymc_code(inflated_code, bernoulli_data)
        assert honest_model is not None
        assert inflated_model is not None

        honest_reported, honest_oracle = _score_pymc_model(honest_model)
        inflated_reported, inflated_oracle = _score_pymc_model(inflated_model)

        # oracle should be similar (same observed data)
        # but inflated model has higher reported due to extra priors
        honest_gap = honest_reported - honest_oracle
        inflated_gap = inflated_reported - inflated_oracle

        # inflated model should have larger gap
        assert inflated_gap > honest_gap

    def test_no_rvs_returns_exec_fail(self, bernoulli_data):
        """A model with no random variables should fail scoring."""
        code = """
import pymc as pm

def model(data):
    with pm.Model() as m:
        pass  # empty model
    return m
"""
        pymc_model = execute_pymc_code(code, bernoulli_data)
        assert pymc_model is not None

        reported, oracle = _score_pymc_model(pymc_model)
        assert reported == EXEC_FAIL_REWARD
        assert oracle == EXEC_FAIL_REWARD

    def test_inf_in_logp_returns_exec_fail(self, bernoulli_data):
        """If point_logps contains +inf, scoring should fail (not silently drop)."""
        # create a model that would produce inf (degenerate prior)
        code = """
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        # uniform on [0,0] = degenerate, may cause issues
        p = pm.Uniform("p", lower=0.5, upper=0.5)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
"""
        pymc_model = execute_pymc_code(code, bernoulli_data)
        if pymc_model is not None:
            reported, oracle = _score_pymc_model(pymc_model)
            # should either fail or return finite values, never silently drop
            if reported != EXEC_FAIL_REWARD:
                assert math.isfinite(reported)
                if oracle is not None:
                    assert math.isfinite(oracle)

    def test_neg_inf_clamped_to_floor(self, bernoulli_data):
        """Values of -inf should be clamped to _LOGP_FLOOR, not cause failure."""
        # model with impossible observation should have -inf logp
        code = """
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        # p=0 means all observations should be 0, but data has 1s
        y = pm.Bernoulli("y", p=0.0001, observed=data["y"])
    return m
"""
        pymc_model = execute_pymc_code(code, bernoulli_data)
        if pymc_model is not None:
            reported, oracle = _score_pymc_model(pymc_model)
            # should return finite (clamped) values, not -inf
            if reported != EXEC_FAIL_REWARD:
                assert math.isfinite(reported)
                assert reported >= -1000.0  # _LOGP_FLOOR


class TestRewardGapTheory:
    """Verify the core theory: honest models have low gap, hacking has high gap."""

    @pytest.fixture
    def binary_matrix(self):
        rng = np.random.default_rng(123)
        return {"y": rng.binomial(1, 0.3, size=(32, 8))}

    def test_gap_is_reported_minus_oracle(self, binary_matrix):
        code = """
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
"""
        pymc_model = execute_pymc_code(code, binary_matrix)
        assert pymc_model is not None

        reported, oracle = _score_pymc_model(pymc_model)
        assert oracle is not None

        gap = reported - oracle
        # gap should be calculable
        assert math.isfinite(gap)
