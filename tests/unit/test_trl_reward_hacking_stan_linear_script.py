from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "trl_reward_hacking_stan_linear.py"
    )
    spec = importlib.util.spec_from_file_location(
        "trl_reward_hacking_stan_linear_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTrajectoryPoint:
    def __init__(
        self,
        *,
        reward_mean: float,
        n_valid: int,
        n_total: int,
        n_parse_fail: int = 0,
        n_exec_fail: int = 0,
        n_checked: int = 0,
        n_unsafe: int = 0,
        unsafe_rate: float = 0.0,
        frac_non_normalized: float = 0.0,
        mean_abs_log_mass: float = 0.0,
        n_norm_checked: int = 0,
    ) -> None:
        self.reward_mean = reward_mean
        self.n_valid = n_valid
        self.n_total = n_total
        self.n_parse_fail = n_parse_fail
        self.n_exec_fail = n_exec_fail
        self.n_checked = n_checked
        self.n_unsafe = n_unsafe
        self.unsafe_rate = unsafe_rate
        self.frac_non_normalized = frac_non_normalized
        self.mean_abs_log_mass = mean_abs_log_mass
        self.n_norm_checked = n_norm_checked


def test_config_from_mapping_maps_fields() -> None:
    module = _load_module()
    cfg = module.config_from_mapping(
        {
            "model": "Qwen/Qwen3-1.7B",
            "n_steps": 10,
            "n_prompts": 20,
            "rollouts_per_prompt": 4,
            "output_dir": "artifacts/test_stan_linear_cfg",
            "checker_mode": "enforce",
            "prompt_policy": "induce_subtle_single",
        }
    )
    assert cfg.model == "Qwen/Qwen3-1.7B"
    assert cfg.n_steps == 10
    assert cfg.n_prompts == 20
    assert cfg.rollouts_per_prompt == 4
    assert cfg.output_dir == "artifacts/test_stan_linear_cfg"
    assert cfg.checker_mode == "enforce"
    assert cfg.prompt_policy == "induce_subtle_single"


def test_config_from_mapping_rejects_unknown_key() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Unsupported train config keys"):
        module.config_from_mapping({"legacy_oracle": 1.0})


def test_default_run_name_has_expected_prefix() -> None:
    module = _load_module()
    cfg = module.TRLStanLinearRewardConfig(model="Qwen/Qwen3-4B-Instruct-2507", n_steps=7)
    run_name = module._default_run_name(cfg)
    assert run_name.startswith("stan-linear-grpo-qwen3-4b-instruct-2507-s7-")


def test_build_summary_emits_direct_stan_keys() -> None:
    module = _load_module()
    cfg = module.TRLStanLinearRewardConfig(
        checker_mode="shadow",
        reward_output_field="reported_log_density",
        prompt_policy="neutral_single",
    )
    summary = module._build_summary(
        cfg,
        {
            "final_valid_rate": 0.75,
            "final_reward_mean": 0.1,
            "final_frac_non_normalized": 0.25,
        },
    )
    assert summary["paper/reward_metric"] == "reported_log_density"
    assert summary["paper/reward_estimator_backend"] == "cmdsafestan_plain"
    assert summary["paper/delta_scope"] == "raw_y_fixed_x"
    assert summary["paper/monitoring_mode"] == "safestan_shadow"
    assert summary["paper/prompt_policy"] == "neutral_single"


def test_compute_results_handles_single_batch() -> None:
    module = _load_module()
    cfg = module.TRLStanLinearRewardConfig()
    state = SimpleNamespace(
        trajectory=[
            _FakeTrajectoryPoint(
                reward_mean=1.5,
                n_valid=3,
                n_total=4,
                n_parse_fail=1,
                n_exec_fail=0,
                n_checked=3,
                n_unsafe=1,
                unsafe_rate=1 / 3,
                frac_non_normalized=0.5,
                mean_abs_log_mass=0.7,
                n_norm_checked=2,
            )
        ]
    )
    results = module._compute_results(cfg, state)
    assert results["final_reward_mean"] == 1.5
    assert results["final_parse_fail_rate"] == pytest.approx(0.25)
    assert results["final_unsafe_rate"] == pytest.approx(1 / 3)
    assert results["paper/reward_metric"] == "reported_log_density"
