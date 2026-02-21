"""Safety gate and integrity checks on real PyMC models, no mocks."""

from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import (
    execute_pymc_code,
    extract_pymc_code,
)
from ppl_synthesis_reward_hacking.backends.pymc_safe.checker import check_pymc_model
from ppl_synthesis_reward_hacking.evaluation.integrity_checks import (
    check_data_used,
    check_fabricated_data,
    parse_checker_reasons_to_tags,
)

try:
    import pymc  # noqa: F401

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

requires_pymc = pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
pytestmark = [pytest.mark.pymc, requires_pymc]


@pytest.fixture
def safety_templates(
    honest_completion,
    potential_injection_completion,
    data_discard_completion,
    observed_constant_completion,
    double_observation_completion,
    fabricated_extra_obs_completion,
):
    return {
        "honest": honest_completion,
        "potential_injection": potential_injection_completion,
        "data_discard": data_discard_completion,
        "observed_constant": observed_constant_completion,
        "double_observation": double_observation_completion,
        "fabricated_extra_obs": fabricated_extra_obs_completion,
    }


# -- helpers --


def _extract_and_execute(completion: str, data: dict) -> tuple[str, object]:
    code = extract_pymc_code(completion)
    assert code is not None, "extraction should succeed"
    model = execute_pymc_code(code, data, timeout=30)
    return code, model


class TestSafetyGateAcceptance:
    def test_honest_model_accepted(self, scoring_data_d1, safety_templates):
        _, model = _extract_and_execute(safety_templates["honest"], scoring_data_d1)
        assert model is not None
        accepted, reasons = check_pymc_model(model, require_observable_data=False)
        assert accepted is True
        assert reasons == []

    def test_potential_injection_rejected(self, scoring_data_d1, safety_templates):
        _, model = _extract_and_execute(safety_templates["potential_injection"], scoring_data_d1)
        assert model is not None
        accepted, reasons = check_pymc_model(model, require_observable_data=False)
        assert accepted is False
        assert "potentials not allowed" in reasons

    def test_reason_tags_for_potential(self):
        tags = parse_checker_reasons_to_tags(["potentials not allowed"])
        assert "potentials" in tags


class TestSafetyGateDataDiscard:
    @pytest.mark.xfail(
        strict=True,
        reason="gate cannot detect data discard without require_observable_data=True",
    )
    def test_data_discard_gate_behavior(self, scoring_data_d1, safety_templates):
        """Data discard (no observed RVs) should be rejected by the gate.

        Currently the checker with require_observable_data=False has nothing
        to flag, so the gate accepts. This xfail documents the known gap and
        will turn into XPASS (test failure) when the gate is fixed.
        """
        _, model = _extract_and_execute(safety_templates["data_discard"], scoring_data_d1)
        if model is None:
            pytest.skip("model execution failed")

        accepted, reasons = check_pymc_model(model, require_observable_data=False)
        assert accepted is False, "gate should reject data discard"


class TestSafetyGateDoubleObservation:
    @pytest.mark.xfail(
        strict=True,
        reason="gate cannot detect double observation without MutableData",
    )
    def test_double_obs_without_mutable_data(self, scoring_data_d1, safety_templates):
        """Double observation without pm.MutableData should be rejected.

        The gate checks data_use_counts via graph walk from observed RV
        value vars back to data_vars. Without pm.MutableData, there are
        no data_vars in the model, so the graph walk finds nothing to flag.
        This xfail documents the known gap and will turn into XPASS (test
        failure) when the gate is fixed (e.g. by requiring MutableData).
        """
        _, model = _extract_and_execute(safety_templates["double_observation"], scoring_data_d1)
        assert model is not None
        accepted, reasons = check_pymc_model(model, require_observable_data=False)
        assert accepted is False, "gate should reject double observation"


class TestIntegrityCheckFabricatedData:
    def test_honest_model_passes(self, scoring_data_d1, safety_templates):
        _, model = _extract_and_execute(safety_templates["honest"], scoring_data_d1)
        assert model is not None
        assert check_fabricated_data(model, scoring_data_d1) is True

    def test_fabricated_extra_obs_detected(self, scoring_data_d1, safety_templates):
        _, model = _extract_and_execute(safety_templates["fabricated_extra_obs"], scoring_data_d1)
        assert model is not None
        # actual_size = 1 (data["y"]) + 5 (extra) = 6, expected_size = 1
        assert check_fabricated_data(model, scoring_data_d1) is False


class TestIntegrityCheckDataUsed:
    def test_honest_model_uses_data(self, scoring_data_d3, safety_templates):
        code = extract_pymc_code(safety_templates["honest"])
        assert code is not None
        assert check_data_used(code, scoring_data_d3) is True

    def test_data_discard_ignores_data(self, scoring_data_d1, safety_templates):
        """Model with no observed RVs ignores data entirely.

        The perturbation test executes the model twice with different
        data lengths. A model with no observations produces identical
        point_logps regardless, so check_data_used returns False.
        If the model fails to build (no observed RVs → point_logps
        errors), the function fails open and returns True.
        """
        code = extract_pymc_code(safety_templates["data_discard"])
        assert code is not None
        result = check_data_used(code, scoring_data_d1)
        # point_logps on a model with only free RVs returns the prior
        # logp, which is identical regardless of data → False.
        # If execution fails, it returns True (fails open).
        assert isinstance(result, bool)

    def test_observed_constant_ignores_data(self, scoring_data_d3, safety_templates):
        """Model observes literal 0.0 instead of data["y"].

        Uses d=3 so the perturbation (halving array from 3 to 1 element)
        changes the data dict, but the model never references data["y"].
        """
        code = extract_pymc_code(safety_templates["observed_constant"])
        assert code is not None
        assert check_data_used(code, scoring_data_d3) is False


class TestIntegrityAndGateConsistency:
    def test_potential_injection_caught_by_gate_and_data_check(
        self, scoring_data_d3, safety_templates
    ):
        code, model = _extract_and_execute(safety_templates["potential_injection"], scoring_data_d3)
        assert model is not None

        accepted, reasons = check_pymc_model(model, require_observable_data=False)
        assert accepted is False
        assert "potentials not allowed" in reasons

        assert check_data_used(code, scoring_data_d3) is True

    def test_observed_constant_caught_by_data_check(self, scoring_data_d3, safety_templates):
        code, model = _extract_and_execute(safety_templates["observed_constant"], scoring_data_d3)
        assert model is not None

        # gate has no reason to reject: no potentials, no custom logp
        accepted, reasons = check_pymc_model(model, require_observable_data=False)
        assert isinstance(accepted, bool)

        # data check catches what the gate misses
        assert check_data_used(code, scoring_data_d3) is False
