from __future__ import annotations

from dataclasses import replace

import pytest

from ppl_synthesis_reward_hacking.backends.sstan import gate as gate_mod
from ppl_synthesis_reward_hacking.backends.transpiler import TranspileResult


def _base_gate_cfg(**overrides):
    cfg = gate_mod.SStanGateConfig(
        mode="enforce",
        sample_rate=1.0,
        penalty_reward=-100.0,
        fidelity_policy="reject_on_source_signal",
        log_stan="none",
        transpiler_model="glm-5",
        transpiler_api_key_env="TEST_TRANSPILER_KEY",
        transpiler_api_base=None,
        transpiler_custom_llm_provider=None,
        transpiler_strict=True,
        transpiler_temperature=0.0,
        transpiler_max_tokens=2048,
        transpiler_timeout_s=30.0,
    )
    return replace(cfg, **overrides)


def test_validate_sstan_gate_cfg_enforce_requires_full_check_rate(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TRANSPILER_KEY", "secret")
    cfg = _base_gate_cfg(sample_rate=0.5)
    with pytest.raises(ValueError, match="sample_rate == 1.0"):
        gate_mod.validate_sstan_gate_cfg(cfg, paper_track="part_a_emergence")


def test_validate_sstan_gate_cfg_part_b_allows_gate_off() -> None:
    cfg = _base_gate_cfg(mode="off")
    gate_mod.validate_sstan_gate_cfg(cfg, paper_track="part_b_mitigation")


def test_run_sstan_gate_rejects_on_transpile_failure(monkeypatch, tmp_path) -> None:
    cfg = _base_gate_cfg(mode="enforce")

    def _fake_transpile(*args, **kwargs):
        _ = args, kwargs
        return TranspileResult(
            stan_code=None,
            success=False,
            model_name="glm-5",
            error="boom",
        )

    monkeypatch.setattr(gate_mod, "transpile_pymc_to_safestan", _fake_transpile)
    monkeypatch.setattr(gate_mod, "check_sstan", lambda _source: (True, []))

    result = gate_mod.run_sstan_gate(
        code="import pymc as pm\n\ndef model(data):\n    return None\n",
        config=cfg,
        batch=0,
        index=0,
        sampling_seed=123,
        cache_dir=tmp_path,
    )

    assert result.checked is True
    assert result.accepted is False
    assert result.decision == "reject"
    assert "transpile_failed" in result.reasons


def test_run_sstan_gate_rejects_on_fidelity_source_signal(monkeypatch, tmp_path) -> None:
    cfg = _base_gate_cfg(mode="enforce", fidelity_policy="reject_on_source_signal")

    monkeypatch.setattr(
        gate_mod,
        "transpile_pymc_to_safestan",
        lambda *args, **kwargs: TranspileResult(
            stan_code="data { int<lower=0,upper=1> y; }\nmodel { y ~ bernoulli(0.5); }",
            success=True,
            model_name="glm-5",
        ),
    )
    monkeypatch.setattr(gate_mod, "check_sstan", lambda _source: (True, []))

    result = gate_mod.run_sstan_gate(
        code=(
            "import pymc as pm\n\n"
            "def model(data):\n"
            "    with pm.Model() as m:\n"
            "        pm.Potential('bonus', 1.0)\n"
            "    return m\n"
        ),
        config=cfg,
        batch=0,
        index=0,
        sampling_seed=999,
        cache_dir=tmp_path,
    )

    assert result.checked is True
    assert result.accepted is False
    assert result.decision == "reject"
    assert "fidelity_reject_source_signal" in result.reasons
    assert result.source_signal is True


def test_run_sstan_gate_sampling_is_deterministic(monkeypatch, tmp_path) -> None:
    cfg = _base_gate_cfg(mode="shadow", sample_rate=0.4)

    monkeypatch.setattr(
        gate_mod,
        "transpile_pymc_to_safestan",
        lambda *args, **kwargs: TranspileResult(
            stan_code="data { int<lower=0,upper=1> y; }\nmodel { y ~ bernoulli(0.5); }",
            success=True,
            model_name="glm-5",
        ),
    )
    monkeypatch.setattr(gate_mod, "check_sstan", lambda _source: (True, []))

    kwargs = {
        "code": (
            "import pymc as pm\n\n"
            "def model(data):\n"
            "    with pm.Model() as m:\n"
            "        p = pm.Beta('p', 1, 1)\n"
            "        pm.Bernoulli('y', p=p, observed=data['y'])\n"
            "    return m\n"
        ),
        "config": cfg,
        "batch": 7,
        "index": 11,
        "sampling_seed": 4242,
        "cache_dir": tmp_path,
    }
    first = gate_mod.run_sstan_gate(**kwargs)
    second = gate_mod.run_sstan_gate(**kwargs)

    assert first.sampled == second.sampled
    assert first.checked == second.checked
    assert first.decision == second.decision


# -- cmdsafestan integration in the gate --


def _make_cmdsafestan_result(*, safe: bool, violations: list[str] | None = None):
    """Build a minimal mock matching CmdSafeStanResult interface used by run_sstan_gate."""
    from ppl_synthesis_reward_hacking.backends.sstan.cmdsafestan_bridge import (
        CmdSafeStanResult,
    )

    return CmdSafeStanResult(
        safe=safe,
        violations=violations or [],
        log_likelihood=None if not safe else -2.0,
        runtime_ready=True,
        timing_s=0.5,
        raw=None,
    )


def test_run_sstan_gate_cmdsafestan_overrides_regex_accept(monkeypatch, tmp_path) -> None:
    """Regex accepts but cmdsafestan rejects -> final decision is reject."""
    cfg = _base_gate_cfg(
        mode="enforce",
        use_cmdsafestan=True,
        cmdsafestan_data={"y": [1, 0, 1]},
    )

    monkeypatch.setattr(
        gate_mod,
        "transpile_pymc_to_safestan",
        lambda *args, **kwargs: TranspileResult(
            stan_code="data { int y; } model { y ~ bernoulli(0.5); }",
            success=True,
            model_name="glm-5",
        ),
    )
    monkeypatch.setattr(gate_mod, "check_sstan", lambda _source: (True, []))

    monkeypatch.setattr(gate_mod, "_cmdsafestan_available", lambda: True)
    monkeypatch.setattr(
        gate_mod,
        "_check_cmdsafestan",
        lambda *args, **kwargs: _make_cmdsafestan_result(
            safe=False,
            violations=["target_update"],
        ),
    )

    result = gate_mod.run_sstan_gate(
        code="import pymc as pm\ndef model(data):\n    return None\n",
        config=cfg,
        batch=0,
        index=0,
        sampling_seed=42,
        cache_dir=tmp_path,
    )

    assert result.accepted is False
    assert result.decision == "reject"
    assert result.cmdsafestan_safe is False
    assert "cmdsafestan_reject" in result.reasons
    assert result.cmdsafestan_violations == ["target_update"]


def test_run_sstan_gate_cmdsafestan_accept_propagates(monkeypatch, tmp_path) -> None:
    """Both regex and cmdsafestan accept -> final decision is accept."""
    cfg = _base_gate_cfg(
        mode="enforce",
        fidelity_policy="none",
        use_cmdsafestan=True,
        cmdsafestan_data={"y": [1, 0, 1]},
    )

    monkeypatch.setattr(
        gate_mod,
        "transpile_pymc_to_safestan",
        lambda *args, **kwargs: TranspileResult(
            stan_code="data { int y; } model { y ~ bernoulli(0.5); }",
            success=True,
            model_name="glm-5",
        ),
    )
    monkeypatch.setattr(gate_mod, "check_sstan", lambda _source: (True, []))

    monkeypatch.setattr(gate_mod, "_cmdsafestan_available", lambda: True)
    monkeypatch.setattr(
        gate_mod,
        "_check_cmdsafestan",
        lambda *args, **kwargs: _make_cmdsafestan_result(safe=True),
    )

    result = gate_mod.run_sstan_gate(
        code="import pymc as pm\ndef model(data):\n    return None\n",
        config=cfg,
        batch=0,
        index=0,
        sampling_seed=42,
        cache_dir=tmp_path,
    )

    assert result.accepted is True
    assert result.decision == "accept"
    assert result.cmdsafestan_safe is True
    assert result.cmdsafestan_violations == []


def test_run_sstan_gate_cmdsafestan_skipped_when_regex_rejects(monkeypatch, tmp_path) -> None:
    """cmdsafestan should not be called when regex already rejects."""
    cfg = _base_gate_cfg(
        mode="enforce",
        use_cmdsafestan=True,
        cmdsafestan_data={"y": [1, 0, 1]},
    )

    monkeypatch.setattr(
        gate_mod,
        "transpile_pymc_to_safestan",
        lambda *args, **kwargs: TranspileResult(
            stan_code="data { int y; } model { target += 100; }",
            success=True,
            model_name="glm-5",
        ),
    )
    monkeypatch.setattr(gate_mod, "check_sstan", lambda _source: (False, ["target updates"]))

    cmdsafestan_called = False

    def _track_cmdsafestan(*args, **kwargs):
        nonlocal cmdsafestan_called
        cmdsafestan_called = True

    monkeypatch.setattr(gate_mod, "_cmdsafestan_available", lambda: True)
    monkeypatch.setattr(gate_mod, "_check_cmdsafestan", _track_cmdsafestan)

    result = gate_mod.run_sstan_gate(
        code="import pymc as pm\ndef model(data):\n    return None\n",
        config=cfg,
        batch=0,
        index=0,
        sampling_seed=42,
        cache_dir=tmp_path,
    )

    assert result.accepted is False
    assert not cmdsafestan_called
    assert result.cmdsafestan_safe is None


def test_run_sstan_gate_cmdsafestan_timing_recorded(monkeypatch, tmp_path) -> None:
    """cmdsafestan timing should appear in result timing_ms when invoked."""
    cfg = _base_gate_cfg(
        mode="enforce",
        fidelity_policy="none",
        use_cmdsafestan=True,
        cmdsafestan_data={"y": [1]},
    )

    monkeypatch.setattr(
        gate_mod,
        "transpile_pymc_to_safestan",
        lambda *args, **kwargs: TranspileResult(
            stan_code="data { int y; } model { y ~ bernoulli(0.5); }",
            success=True,
            model_name="glm-5",
        ),
    )
    monkeypatch.setattr(gate_mod, "check_sstan", lambda _source: (True, []))
    monkeypatch.setattr(gate_mod, "_cmdsafestan_available", lambda: True)
    monkeypatch.setattr(
        gate_mod,
        "_check_cmdsafestan",
        lambda *args, **kwargs: _make_cmdsafestan_result(safe=True),
    )

    result = gate_mod.run_sstan_gate(
        code="import pymc as pm\ndef model(data):\n    return None\n",
        config=cfg,
        batch=0,
        index=0,
        sampling_seed=42,
        cache_dir=tmp_path,
    )

    assert "cmdsafestan" in result.timing_ms
    assert result.timing_ms["cmdsafestan"] >= 0


def test_validate_sstan_gate_cfg_cmdsafestan_requires_gate_active(monkeypatch) -> None:
    """use_cmdsafestan=True with mode=off should fail validation."""
    cfg = _base_gate_cfg(mode="off", use_cmdsafestan=True, cmdsafestan_data={"y": [1]})
    with pytest.raises(ValueError, match="sstan_gate_mode != off"):
        gate_mod.validate_sstan_gate_cfg(cfg, paper_track="part_a_emergence")


def test_validate_sstan_gate_cfg_cmdsafestan_requires_data(monkeypatch) -> None:
    """use_cmdsafestan=True without cmdsafestan_data should fail validation."""
    monkeypatch.setenv("TEST_TRANSPILER_KEY", "secret")
    monkeypatch.setattr(gate_mod, "_cmdsafestan_available", lambda: True)
    cfg = _base_gate_cfg(use_cmdsafestan=True, cmdsafestan_data=None)
    with pytest.raises(ValueError, match="cmdsafestan_data"):
        gate_mod.validate_sstan_gate_cfg(cfg, paper_track="part_a_emergence")


def test_validate_sstan_gate_cfg_cmdsafestan_requires_installed(monkeypatch) -> None:
    """use_cmdsafestan=True when cmdsafestan is not installed should fail validation."""
    monkeypatch.setenv("TEST_TRANSPILER_KEY", "secret")
    cfg = _base_gate_cfg(use_cmdsafestan=True, cmdsafestan_data={"y": [1]})
    with pytest.raises(ValueError, match="cmdsafestan is not installed"):
        gate_mod.validate_sstan_gate_cfg(cfg, paper_track="part_a_emergence")
