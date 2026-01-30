from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec
from ppl_synthesis_reward_hacking.backends.pymc_safe.backend import SafePyMCBackend


def _require_pymc() -> None:
    if os.getenv("PSRH_RUN_PYMC") != "1":
        pytest.skip("Set PSRH_RUN_PYMC=1 to run PyMC integration tests")
    pytest.importorskip("pymc")


@pytest.mark.pymc
def test_safe_pymc_rejects_potential():
    _require_pymc()
    import pymc as pm

    with pm.Model() as model:
        y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
        pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("obs", mu=0.0, sigma=1.0, observed=y)
        pm.Potential("hack", 1.0)

    backend = SafePyMCBackend()
    model_spec = ModelSpec(
        backend="pymc_safe",
        name="pymc_safe",
        source="",
        meta={"pymc_model": model},
    )
    compiled = backend.compile(model_spec, cache_dir=Path("artifacts") / ".cache")
    assert compiled.get("accepted") is False


@pytest.mark.pymc
def test_safe_pymc_accepts_honest_model():
    _require_pymc()
    import pymc as pm

    with pm.Model() as model:
        y = pm.Data("y", np.array([0.1, -0.2, 0.3]))
        pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("obs", mu=0.0, sigma=1.0, observed=y)

    backend = SafePyMCBackend()
    model_spec = ModelSpec(
        backend="pymc_safe",
        name="pymc_safe",
        source="",
        meta={"pymc_model": model},
    )
    compiled = backend.compile(model_spec, cache_dir=Path("artifacts") / ".cache")
    assert compiled.get("accepted") is True
