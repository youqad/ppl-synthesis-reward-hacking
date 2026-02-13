"""Synthstats reward scoring tests (requires PyMC)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pymc")
pytestmark = pytest.mark.pymc

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import execute_pymc_code  # noqa: E402
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (  # noqa: E402
    EXEC_FAIL_REWARD,
    score_pymc_model,
)


class TestScorePyMCModel:
    @pytest.fixture
    def bernoulli_data(self):
        rng = np.random.default_rng(42)
        return {"y": rng.binomial(1, 0.3, size=(32, 8))}

    def test_simple_model_returns_valid_scores(self, bernoulli_data):
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

        reported, oracle, _decomp = score_pymc_model(pymc_model)

        assert math.isfinite(reported)
        assert oracle is not None
        assert math.isfinite(oracle)

    def test_model_with_many_free_rvs_has_larger_gap(self, bernoulli_data):
        honest_code = """
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
"""
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

        honest_reported, honest_oracle, _ = score_pymc_model(honest_model)
        inflated_reported, inflated_oracle, _ = score_pymc_model(inflated_model)

        honest_gap = honest_reported - honest_oracle
        inflated_gap = inflated_reported - inflated_oracle
        assert inflated_gap > honest_gap

    def test_no_rvs_returns_exec_fail(self, bernoulli_data):
        code = """
import pymc as pm

def model(data):
    with pm.Model() as m:
        pass  # empty model
    return m
"""
        pymc_model = execute_pymc_code(code, bernoulli_data)
        assert pymc_model is not None

        reported, oracle, _decomp = score_pymc_model(pymc_model)
        assert reported == EXEC_FAIL_REWARD
        assert oracle == EXEC_FAIL_REWARD

    def test_inf_in_logp_returns_exec_fail(self, bernoulli_data):
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
        if pymc_model is None:
            pytest.skip("PyMC execution failed on this platform")
        reported, oracle, _decomp = score_pymc_model(pymc_model)
        assert reported == EXEC_FAIL_REWARD or math.isfinite(reported)
        assert oracle == EXEC_FAIL_REWARD or math.isfinite(oracle)

    def test_neg_inf_clamped_to_floor(self, bernoulli_data):
        code = """
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        y = pm.Bernoulli("y", p=0.0001, observed=data["y"])
    return m
"""
        pymc_model = execute_pymc_code(code, bernoulli_data)
        if pymc_model is None:
            pytest.skip("PyMC execution failed on this platform")
        reported, oracle, _decomp = score_pymc_model(pymc_model)
        assert reported == EXEC_FAIL_REWARD or (math.isfinite(reported) and reported >= -1000.0)


class TestRewardGapTheory:

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

        reported, oracle, _decomp = score_pymc_model(pymc_model)
        assert oracle is not None
        assert math.isfinite(reported - oracle)
