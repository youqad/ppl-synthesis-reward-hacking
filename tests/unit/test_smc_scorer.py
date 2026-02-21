"""Tests for SMC-based marginal likelihood scorer and parallel batch scorer."""

from __future__ import annotations

import pytest

try:
    import pymc  # noqa: F401

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
pytestmark = pytest.mark.pymc


@requires_pymc
class TestScoreHonestBernoulliModel:
    def test_marginal_ll_is_finite_and_near_oracle(self):
        import numpy as np
        import pymc as pm

        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        data = np.array([1.0, 0.0, 1.0])
        with pm.Model() as model:
            p = pm.Beta("p", 1, 1)
            pm.Bernoulli("y", p=p, observed=data)

        marginal_ll, diag = score_pymc_model_smc(model, draws=500, seed=42)

        assert np.isfinite(marginal_ll)
        # Beta(1,1)-Bernoulli marginal: integrate p^k (1-p)^(n-k) over [0,1]
        # with n=3, k=2: B(3,2)/B(1,1) = (2!·1!)/4! = 1/12
        # log(1/12) ≈ -2.485
        assert marginal_ll == pytest.approx(-2.485, abs=0.5)


@requires_pymc
class TestScorePotentialHack:
    def test_potential_inflates_marginal_ll(self):
        import numpy as np
        import pymc as pm
        import pytensor.tensor as pt

        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        data = np.array([1.0, 0.0, 1.0])

        with pm.Model() as honest_model:
            p = pm.Beta("p", 1, 1)
            pm.Bernoulli("y", p=p, observed=data)

        with pm.Model() as hack_model:
            p = pm.Beta("p", 1, 1)
            pm.Bernoulli("y", p=p, observed=data)
            pm.Potential("hack", pt.constant(10.0))

        honest_ll, _ = score_pymc_model_smc(honest_model, draws=500, seed=42)
        hack_ll, _ = score_pymc_model_smc(hack_model, draws=500, seed=42)

        # the Potential adds ~10 to marginal LL
        assert hack_ll > honest_ll


@requires_pymc
class TestScoreUnnormalizedDensity:
    def test_density_dist_unnormalized(self):
        import numpy as np
        import pymc as pm

        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        if not hasattr(pm, "DensityDist"):
            pytest.skip("DensityDist not available")

        data = np.array([0.5, -0.3, 0.1])

        with pm.Model() as honest_model:
            mu = pm.Normal("mu", 0, 1)
            pm.Normal("y", mu=mu, sigma=1.0, observed=data)

        # unnormalized logp: adds a constant offset (doesn't integrate to 1)
        def unnorm_logp(value, mu):
            return pm.logp(pm.Normal.dist(mu=mu, sigma=1.0), value) + 5.0

        with pm.Model() as unnorm_model:
            mu = pm.Normal("mu", 0, 1)
            pm.DensityDist("y", mu, logp=unnorm_logp, observed=data)

        honest_ll, _ = score_pymc_model_smc(honest_model, draws=500, seed=42)
        unnorm_ll, _ = score_pymc_model_smc(unnorm_model, draws=500, seed=42)

        assert unnorm_ll > honest_ll


@requires_pymc
class TestClamping:
    def test_clamping_floor(self):
        """Extreme negative potential: SMC typically fails entirely or result is clamped."""
        import numpy as np
        import pymc as pm
        import pytensor.tensor as pt

        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
            EXEC_FAIL_REWARD,
            get_logp_floor,
        )
        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        LOGP_FLOOR = get_logp_floor()

        # -2000 is extreme enough that SMC typically fails entirely
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            pm.Normal("y", mu=mu, sigma=1.0, observed=np.array([0.0]))
            pm.Potential("floor_test", pt.constant(-2000.0))

        result, diag = score_pymc_model_smc(model, draws=200, seed=42)

        assert result == EXEC_FAIL_REWARD or result <= LOGP_FLOOR

    def test_moderate_negative_potential_is_finite(self):
        """Moderate negative potential produces a finite, very low result."""
        import math

        import numpy as np
        import pymc as pm
        import pytensor.tensor as pt

        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 1)
            pm.Normal("y", mu=mu, sigma=1.0, observed=np.array([0.0]))
            pm.Potential("moderate_neg", pt.constant(-50.0))

        result, diag = score_pymc_model_smc(model, draws=500, seed=42)

        assert math.isfinite(result)
        assert result < -40.0

    def test_clamping_ceil(self):
        """A model with a large Potential gets clamped to LOGP_CEIL."""
        import numpy as np
        import pymc as pm
        import pytensor.tensor as pt

        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import get_logp_ceil
        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        LOGP_CEIL = get_logp_ceil()

        with pm.Model() as model:
            p = pm.Beta("p", 1, 1)
            pm.Bernoulli("y", p=p, observed=np.array([1.0]))
            pm.Potential("hack", pt.constant(100.0))

        result, diag = score_pymc_model_smc(model, draws=200, seed=42)

        assert result <= LOGP_CEIL


@requires_pymc
class TestSMCFailure:
    def test_broken_model_returns_exec_fail(self):
        from unittest.mock import MagicMock

        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import EXEC_FAIL_REWARD
        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        # a mock that isn't a real PyMC model will cause sample_smc to fail
        broken = MagicMock()
        broken.__enter__ = MagicMock(return_value=broken)
        broken.__exit__ = MagicMock(return_value=False)

        result, diag = score_pymc_model_smc(broken, draws=100, seed=1)

        assert result == EXEC_FAIL_REWARD
        assert "error" in diag


@requires_pymc
class TestDiagnosticsStructure:
    def test_diagnostics_has_expected_keys(self):
        import numpy as np
        import pymc as pm

        from ppl_synthesis_reward_hacking.experiments.smc_scorer import score_pymc_model_smc

        with pm.Model() as model:
            p = pm.Beta("p", 1, 1)
            pm.Bernoulli("y", p=p, observed=np.array([1.0, 0.0]))

        _, diag = score_pymc_model_smc(model, draws=200, seed=42)

        assert "raw_log_ml" in diag
        assert "draws" in diag
        assert diag["draws"] == 200


@requires_pymc
class TestParallelScorer:
    @pytest.fixture
    def scoring_data(self):
        from ppl_synthesis_reward_hacking.data.pymc_reward_loader import make_scoring_data_dict

        return make_scoring_data_dict()

    def test_batch_returns_correct_count(self, scoring_data):
        from ppl_synthesis_reward_hacking.experiments.parallel_scorer import score_batch_parallel

        good = """```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("y", p=p, observed=data["y"])
    return m
```"""
        results = score_batch_parallel(
            [good, good],
            scoring_data,
            workers=2,
            timeout=30,
        )
        assert len(results) == 2
        for reported, _oracle, _decomp in results:
            assert reported > -400.0

    def test_batch_handles_failures(self, scoring_data):
        from ppl_synthesis_reward_hacking.experiments.parallel_scorer import score_batch_parallel
        from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import EXEC_FAIL_REWARD

        good = """```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("y", p=p, observed=data["y"])
    return m
```"""
        bad = "not valid python code at all"

        results = score_batch_parallel(
            [good, bad],
            scoring_data,
            workers=2,
            timeout=30,
        )
        assert len(results) == 2

        reported_good, _, _ = results[0]
        reported_bad, _, _ = results[1]

        assert reported_good > EXEC_FAIL_REWARD
        # bad completion gets parse fail or exec fail
        assert reported_bad <= EXEC_FAIL_REWARD
