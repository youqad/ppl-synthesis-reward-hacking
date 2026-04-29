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
        n_contract_fail: int = 0,
        n_checked: int = 0,
        n_unsafe: int = 0,
        unsafe_rate: float = 0.0,
        frac_non_normalized: float = 0.0,
        mean_abs_log_mass: float = 0.0,
        max_abs_log_mass: float = 0.0,
        n_norm_checked: int = 0,
        n_norm_failed: int = 0,
        n_non_normalized: int = 0,
        n_norm_cache_hits: int = 0,
        reported_mean_all: float | None = None,
    ) -> None:
        self.reward_mean = reward_mean
        self.n_valid = n_valid
        self.n_total = n_total
        self.n_parse_fail = n_parse_fail
        self.n_exec_fail = n_exec_fail
        self.n_contract_fail = n_contract_fail
        self.n_checked = n_checked
        self.n_unsafe = n_unsafe
        self.unsafe_rate = unsafe_rate
        self.frac_non_normalized = frac_non_normalized
        self.mean_abs_log_mass = mean_abs_log_mass
        self.max_abs_log_mass = max_abs_log_mass
        self.n_norm_checked = n_norm_checked
        self.n_norm_failed = n_norm_failed
        self.n_non_normalized = n_non_normalized
        self.n_norm_cache_hits = n_norm_cache_hits
        self.reported_mean_all = reward_mean if reported_mean_all is None else reported_mean_all


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
            "prompt_policy": "induce_subtle_family",
            "num_system_prompts": 3,
        }
    )
    assert cfg.model == "Qwen/Qwen3-1.7B"
    assert cfg.n_steps == 10
    assert cfg.n_prompts == 20
    assert cfg.rollouts_per_prompt == 4
    assert cfg.output_dir == "artifacts/test_stan_linear_cfg"
    assert cfg.checker_mode == "enforce"
    assert cfg.prompt_policy == "induce_subtle_family"
    assert cfg.num_system_prompts == 3


def test_config_from_mapping_rejects_unknown_key() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Unsupported train config keys"):
        module.config_from_mapping({"legacy_oracle": 1.0})


def test_config_from_mapping_rejects_too_many_prompts() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="n_prompts=33 exceeds"):
        module.config_from_mapping(
            {
                "n_prompts": 33,
                "prompt_policy": "neutral_family",
            }
        )


def test_config_from_mapping_rejects_too_many_system_prompts() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="num_system_prompts=5 exceeds"):
        module.config_from_mapping(
            {
                "num_system_prompts": 5,
                "prompt_policy": "neutral_family",
            }
        )


def test_config_from_mapping_accepts_story_prompt_policy() -> None:
    module = _load_module()
    cfg = module.config_from_mapping(
        {
            "n_prompts": 8,
            "prompt_policy": "induce_subtle_family_stories",
            "num_system_prompts": 4,
        }
    )
    assert cfg.prompt_policy == "induce_subtle_family_stories"
    assert cfg.n_prompts == 8
    assert cfg.num_system_prompts == 4


def test_config_from_mapping_uses_full_batch_normalization_by_default() -> None:
    module = _load_module()
    cfg = module.config_from_mapping({"prompt_policy": "neutral_family"})
    assert cfg.normalization_sample_size == -1


def test_config_from_mapping_rejects_invalid_normalization_sample_size() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="normalization_sample_size must be >= -1"):
        module.config_from_mapping(
            {
                "prompt_policy": "neutral_family",
                "normalization_sample_size": -2,
            }
        )


def test_default_run_name_has_expected_prefix() -> None:
    module = _load_module()
    cfg = module.TRLStanLinearRewardConfig(
        model="Qwen/Qwen3-4B-Instruct-2507",
        n_steps=7,
        num_system_prompts=2,
    )
    run_name = module._default_run_name(cfg)
    assert run_name.startswith("stan-linear-grpo-qwen3-4b-instruct-2507-s7-sys2-p32-g8-")


def test_build_training_args_uses_expanded_prompt_count(monkeypatch, tmp_path) -> None:
    module = _load_module()
    captured = {}

    class FakeGRPOConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, "TRLGRPOConfig", FakeGRPOConfig, raising=False)
    monkeypatch.setattr(module, "_resolve_precision", lambda: (False, False, False))
    cfg = module.TRLStanLinearRewardConfig(
        n_prompts=8,
        num_system_prompts=4,
        num_generations=8,
    )

    module._build_training_args(
        cfg,
        train_prompt_count=32,
        output_dir=tmp_path,
        model_init_kwargs=None,
    )

    assert captured["generation_batch_size"] == 256
    assert captured["gradient_accumulation_steps"] == 32


def test_build_summary_emits_direct_stan_keys() -> None:
    module = _load_module()
    cfg = module.TRLStanLinearRewardConfig(
        checker_mode="shadow",
        prompt_policy="neutral_family",
    )
    summary = module._build_summary(
        cfg,
        {
            "final_valid_rate": 0.75,
            "final_reward_mean": 0.1,
            "final_frac_non_normalized": 0.25,
            "mean_frac_non_normalized": 0.2,
            "final_n_non_normalized": 3,
            "mean_n_non_normalized_per_batch": 2.5,
        },
    )
    assert summary["paper/reward_metric"] == "singleton_posterior_predictive_logZ_ratio"
    assert summary["paper/reward_estimator_backend"] == "cmdstan_log_prob_gauss_hermite"
    assert summary["paper/delta_scope"] == "singleton_y_given_train_x"
    assert summary["paper/reward_data_split"] == "train_plus_singleton_holdout"
    assert summary["paper/monitoring_mode"] == "safestan_shadow"
    assert summary["paper/normalization_method"] == "gh_y_data"
    assert summary["paper/prompt_policy"] == "neutral_family"
    assert summary["paper/lh_rate_batch_final"] == 0.25
    assert summary["paper/lh_rate_batch_mean"] == 0.2
    assert summary["paper/lh_count_batch_final"] == 3
    assert summary["paper/lh_count_batch_mean"] == 2.5


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
                n_contract_fail=0,
                n_checked=3,
                n_unsafe=1,
                unsafe_rate=1 / 3,
                frac_non_normalized=0.5,
                mean_abs_log_mass=0.7,
                n_norm_checked=2,
                n_non_normalized=1,
                n_norm_cache_hits=1,
                reported_mean_all=-24.0,
            )
        ]
    )
    results = module._compute_results(cfg, state)
    assert results["final_reward_mean"] == 1.5
    assert results["final_reward_mean_all"] == -24.0
    assert results["final_parse_fail_rate"] == pytest.approx(0.25)
    assert results["final_unsafe_rate"] == pytest.approx(1 / 3)
    assert results["final_n_non_normalized"] == 1
    assert results["mean_n_non_normalized_per_batch"] == pytest.approx(1.0)
    assert results["paper/lh_count_batch_final"] == 1
    assert results["paper/lh_count_batch_mean"] == pytest.approx(1.0)
    assert results["paper/reward_metric"] == "singleton_posterior_predictive_logZ_ratio"
