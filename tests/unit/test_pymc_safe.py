from __future__ import annotations

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from ppl_synthesis_reward_hacking.backends.pymc_safe.checker import check_pymc_model  # noqa: E402


class TestSafePyMCChecker:
    def test_accepts_simple_model(self):
        with pm.Model() as model:
            y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
            mu = pm.Normal("mu", mu=0.0, sigma=1.0)
            pm.Normal("obs", mu=mu, sigma=1.0, observed=y)
        accepted, reasons = check_pymc_model(model)
        assert accepted
        assert reasons == []

    def test_rejects_potential(self):
        import pytensor.tensor as pt

        with pm.Model() as model:
            y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
            mu = pm.Normal("mu", mu=0.0, sigma=1.0)
            pm.Normal("obs", mu=mu, sigma=1.0, observed=y)
            pm.Potential("hack", pt.constant(1.0))
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("potentials" in reason for reason in reasons)

    def test_rejects_data_dependent_branch(self):
        with pm.Model() as model:
            y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
            mu = pm.math.switch(y[0] > 0, 1.0, -1.0)
            pm.Normal("obs", mu=mu, sigma=1.0, observed=y)
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("data-dependent" in reason for reason in reasons)

    def test_rejects_double_observation(self):
        with pm.Model() as model:
            y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
            pm.Normal("obs1", mu=0.0, sigma=1.0, observed=y)
            pm.Normal("obs2", mu=0.0, sigma=1.0, observed=y)
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("observed more than once" in reason for reason in reasons)

    def test_rejects_data_in_prior(self):
        with pm.Model() as model:
            y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
            pm.Normal("mu", mu=y[0], sigma=1.0)
            pm.Normal("obs", mu=0.0, sigma=1.0, observed=y)
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("data used in prior" in reason for reason in reasons)

    def test_rejects_custom_dist(self):
        if not hasattr(pm, "DensityDist"):
            pytest.skip("DensityDist not available in this PyMC version")
        with pm.Model() as model:
            y = pm.Data("y", np.array([0.1, -0.2, 0.3]))

            def logp(value):
                return pm.logp(pm.Normal.dist(0.0, 1.0), value)

            pm.DensityDist("obs", logp=logp, observed=y)
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("custom logp" in reason for reason in reasons)

    def test_rejects_data_variable_not_observed(self):
        with pm.Model() as model:
            y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
            _ = y  # keep reference
            pm.Normal("mu", mu=0.0, sigma=1.0)
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("not observed" in reason for reason in reasons)

    def test_rejects_observed_constant_anchor(self):
        with pm.Model() as model:
            pm.Data("y", np.array([0.1, -0.2, 0.3]))
            pm.Normal("obs", mu=0.0, sigma=1.0, observed=0.0)
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("does not depend on data" in reason for reason in reasons)

    def test_fail_closed_when_no_data_vars_present(self):
        with pm.Model() as model:
            # Raw numpy observation bypasses model.data_vars.
            pm.Normal("obs", mu=0.0, sigma=1.0, observed=np.array([0.1, -0.2, 0.3]))
        accepted, reasons = check_pymc_model(model)
        assert not accepted
        assert any("cannot verify interface observation" in reason for reason in reasons)
