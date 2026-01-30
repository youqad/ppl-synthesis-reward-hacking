"""Tests for PyMC code extraction and execution."""

from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import (
    execute_pymc_code,
    extract_pymc_code,
)

# Mark the module for marker-based selection.
pytestmark = pytest.mark.pymc

# skip execution tests if PyMC is not installed
try:
    import pymc  # noqa: F401
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")


class TestExtractPyMCCode:
    def test_fenced_code_block(self):
        completion = '''Here is the model:

```python
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
```

This models binary data.
'''
        code = extract_pymc_code(completion)
        assert code is not None
        assert "def model(data):" in code
        assert "pm.Beta" in code

    def test_bare_model_function(self):
        completion = '''def model(data):
    import pymc as pm
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
    return m
'''
        code = extract_pymc_code(completion)
        assert code is not None
        assert "def model(data):" in code

    def test_no_model_returns_none(self):
        completion = "Here is some text without any code."
        assert extract_pymc_code(completion) is None

    def test_empty_code_block_returns_none(self):
        completion = "```python\n\n```"
        assert extract_pymc_code(completion) is None

    def test_prefers_fenced_over_bare(self):
        # fenced block should win even if bare model appears later
        completion = '''```python
def model(data):
    return "fenced"
```

def model(data):
    return "bare"
'''
        code = extract_pymc_code(completion)
        assert code is not None
        assert "fenced" in code


@requires_pymc
class TestExecutePyMCCode:
    @pytest.fixture
    def simple_data(self):
        import numpy as np
        return {"y": np.array([0, 1, 1, 0, 1])}

    def test_valid_model_executes(self, simple_data):
        code = '''
import pymc as pm

def model(data):
    with pm.Model() as m:
        p = pm.Beta("p", 1, 1)
        y = pm.Bernoulli("y", p=p, observed=data["y"])
    return m
'''
        pymc_model = execute_pymc_code(code, simple_data)
        assert pymc_model is not None
        # check it's a PyMC model
        assert hasattr(pymc_model, "free_RVs")
        assert hasattr(pymc_model, "observed_RVs")

    def test_syntax_error_returns_none(self, simple_data):
        code = "def model(data):\n    return ("  # syntax error
        assert execute_pymc_code(code, simple_data) is None

    def test_runtime_error_returns_none(self, simple_data):
        code = '''
def model(data):
    raise ValueError("oops")
'''
        assert execute_pymc_code(code, simple_data) is None

    def test_no_model_function_returns_none(self, simple_data):
        code = '''
def something_else(data):
    return 42
'''
        assert execute_pymc_code(code, simple_data) is None

    def test_model_not_callable_returns_none(self, simple_data):
        code = '''
model = "not a function"
'''
        assert execute_pymc_code(code, simple_data) is None

    def test_timeout_returns_none(self, simple_data):
        code = '''
import time

def model(data):
    time.sleep(10)  # way longer than timeout
    return None
'''
        # use very short timeout
        result = execute_pymc_code(code, simple_data, timeout=1)
        assert result is None

    def test_model_with_free_rvs(self, simple_data):
        code = '''
import pymc as pm
import numpy as np

def model(data):
    with pm.Model() as m:
        # free RVs
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        # observed
        y = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data["y"])
    return m
'''
        pymc_model = execute_pymc_code(code, simple_data)
        assert pymc_model is not None
        free_names = {rv.name for rv in pymc_model.free_RVs}
        obs_names = {rv.name for rv in pymc_model.observed_RVs}
        assert "mu" in free_names
        assert "sigma" in free_names
        assert "y_obs" in obs_names
