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
        mean_log_mass: float = 0.0,
        max_log_mass: float = 0.0,
        min_log_mass: float = 0.0,
        mean_program_mean_log_mass: float = 0.0,
        mean_program_max_log_mass: float = 0.0,
        mean_program_min_log_mass: float = 0.0,
        n_norm_checked: int = 0,
        n_norm_with_log_mass: int = 0,
        n_norm_failed: int = 0,
        n_non_normalized: int = 0,
        n_norm_cache_hits: int = 0,
        n_positive_lh: int = 0,
        frac_positive_lh: float = 0.0,
        n_negative_lh: int = 0,
        frac_negative_lh: float = 0.0,
        positive_lh_reward_mean: float = 0.0,
        non_positive_lh_reward_mean: float = 0.0,
        positive_lh_reward_lift: float = 0.0,
        negative_lh_reward_mean: float = 0.0,
        non_negative_lh_reward_mean: float = 0.0,
        negative_lh_reward_lift: float = 0.0,
        n_fixed_probe_checked: int = 0,
        n_fixed_probe_failed: int = 0,
        n_positive_lh_fixedprobe: int = 0,
        frac_positive_lh_fixedprobe: float = 0.0,
        n_fixed_probe_cache_hits: int = 0,
        mean_fixed_probe_max_log_mass: float = 0.0,
        n_unique_programs: int = 0,
        n_unique_programs_exact: int = 0,
        n_unique_valid_programs: int = 0,
        n_unique_valid_programs_exact: int = 0,
        unique_program_rate: float = 0.0,
        unique_valid_program_rate: float = 0.0,
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
        self.mean_log_mass = mean_log_mass
        self.max_log_mass = max_log_mass
        self.min_log_mass = min_log_mass
        self.mean_program_mean_log_mass = mean_program_mean_log_mass
        self.mean_program_max_log_mass = mean_program_max_log_mass
        self.mean_program_min_log_mass = mean_program_min_log_mass
        self.n_norm_checked = n_norm_checked
        self.n_norm_with_log_mass = n_norm_with_log_mass
        self.n_norm_failed = n_norm_failed
        self.n_non_normalized = n_non_normalized
        self.n_norm_cache_hits = n_norm_cache_hits
        self.n_positive_lh = n_positive_lh
        self.frac_positive_lh = frac_positive_lh
        self.n_negative_lh = n_negative_lh
        self.frac_negative_lh = frac_negative_lh
        self.positive_lh_reward_mean = positive_lh_reward_mean
        self.non_positive_lh_reward_mean = non_positive_lh_reward_mean
        self.positive_lh_reward_lift = positive_lh_reward_lift
        self.negative_lh_reward_mean = negative_lh_reward_mean
        self.non_negative_lh_reward_mean = non_negative_lh_reward_mean
        self.negative_lh_reward_lift = negative_lh_reward_lift
        self.n_fixed_probe_checked = n_fixed_probe_checked
        self.n_fixed_probe_failed = n_fixed_probe_failed
        self.n_positive_lh_fixedprobe = n_positive_lh_fixedprobe
        self.frac_positive_lh_fixedprobe = frac_positive_lh_fixedprobe
        self.n_fixed_probe_cache_hits = n_fixed_probe_cache_hits
        self.mean_fixed_probe_max_log_mass = mean_fixed_probe_max_log_mass
        self.n_unique_programs = n_unique_programs
        self.n_unique_programs_exact = n_unique_programs_exact
        self.n_unique_valid_programs = n_unique_valid_programs
        self.n_unique_valid_programs_exact = n_unique_valid_programs_exact
        self.unique_program_rate = unique_program_rate
        self.unique_valid_program_rate = unique_valid_program_rate
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
            "contract_penalty_reward_final": -20.0,
            "parse_fail_penalty_reward_final": -50.0,
            "exec_fail_penalty_reward_final": -75.0,
            "validity_penalty_schedule": "two_phase",
            "validity_penalty_decay_steps": 12,
            "validity_penalty_switch_step": 6,
            "reward_ceiling": 50.0,
            "fixed_probe_interval": 2,
            "fixed_probe_sample_size": 5,
            "fixed_probe_n_tasks": 3,
            "invalid_reward_policy": "filter",
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
    assert cfg.contract_penalty_reward_final == -20.0
    assert cfg.parse_fail_penalty_reward_final == -50.0
    assert cfg.exec_fail_penalty_reward_final == -75.0
    assert cfg.validity_penalty_schedule == "two_phase"
    assert cfg.validity_penalty_decay_steps == 12
    assert cfg.validity_penalty_switch_step == 6
    assert cfg.reward_ceiling == 50.0
    assert cfg.fixed_probe_interval == 2
    assert cfg.fixed_probe_sample_size == 5
    assert cfg.fixed_probe_n_tasks == 3
    assert cfg.invalid_reward_policy == "filter"


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
    with pytest.raises(ValueError, match="num_system_prompts=9 exceeds"):
        module.config_from_mapping(
            {
                "num_system_prompts": 9,
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


def test_config_from_mapping_rejects_invalid_fixed_probe_sample_size() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="fixed_probe_sample_size must be >= -1"):
        module.config_from_mapping(
            {
                "prompt_policy": "neutral_family",
                "fixed_probe_sample_size": -2,
            }
        )


def test_config_from_mapping_rejects_invalid_reward_bounds() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="reward_floor must be < reward_ceiling"):
        module.config_from_mapping(
            {
                "prompt_policy": "neutral_family",
                "reward_floor": 50.0,
                "reward_ceiling": 50.0,
            }
        )


def test_config_from_mapping_rejects_invalid_reward_policy() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="invalid_reward_policy must be penalty\\|filter"):
        module.config_from_mapping(
            {
                "prompt_policy": "neutral_family",
                "invalid_reward_policy": "drop",
            }
        )


def test_valid_only_advantages_masks_invalid_rewards() -> None:
    module = _load_module()
    advantages, valid_mask, is_std_zero = module._valid_only_advantages(
        [1.0, None, 3.0, -5.0, None, None, 4.0, None],
        num_generations=4,
        scale_rewards="group",
    )

    assert valid_mask.tolist() == [True, False, True, True, False, False, True, False]
    assert advantages[1] == 0.0
    assert advantages[4] == 0.0
    assert advantages[5] == 0.0
    assert advantages[7] == 0.0
    assert advantages[0] == pytest.approx(0.3203, abs=1e-3)
    assert advantages[2] == pytest.approx(0.8006, abs=1e-3)
    assert advantages[3] == pytest.approx(-1.1209, abs=1e-3)
    assert advantages[6] == 0.0
    assert is_std_zero.tolist() == [False, False, False, False, True, True, True, True]


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


def test_build_fixed_probe_tasks_is_deterministic() -> None:
    module = _load_module()
    cfg = module.TRLStanLinearRewardConfig(
        fixed_probe_interval=1,
        fixed_probe_n_tasks=2,
        fixed_probe_n_train=3,
        fixed_probe_n_test=4,
        fixed_probe_seed_base=123,
    )

    left = module._build_fixed_probe_tasks(cfg)
    right = module._build_fixed_probe_tasks(cfg)

    assert [task["task_id"] for task in left] == [task["task_id"] for task in right]
    assert [task["meta"]["seed"] for task in left] == [123, 124]
    assert len(left) == 2
    assert len(left[0]["X_train"]) == 3
    assert len(left[0]["X_test"]) == 4


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
            "final_frac_positive_lh": 0.15,
            "mean_frac_positive_lh": 0.12,
            "final_n_positive_lh": 4,
            "mean_n_positive_lh_per_batch": 3.5,
            "final_positive_lh_reward_lift": 1.25,
            "mean_positive_lh_reward_lift": 0.75,
            "final_n_unique_programs": 9,
            "mean_n_unique_programs_per_batch": 8.5,
            "final_n_unique_valid_programs": 7,
            "mean_n_unique_valid_programs_per_batch": 6.5,
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
    assert summary["paper/lh_positive_rate_final"] == 0.15
    assert summary["paper/lh_positive_rate_mean"] == 0.12
    assert summary["paper/lh_positive_count_batch_final"] == 4
    assert summary["paper/lh_positive_count_batch_mean"] == 3.5
    assert summary["paper/lh_positive_reward_lift_final"] == 1.25
    assert summary["paper/lh_positive_reward_lift_mean"] == 0.75
    assert summary["paper/unique_program_count_batch_final"] == 9
    assert summary["paper/unique_program_count_batch_mean"] == 8.5
    assert summary["paper/unique_valid_program_count_batch_final"] == 7
    assert summary["paper/unique_valid_program_count_batch_mean"] == 6.5


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
                mean_log_mass=0.2,
                max_log_mass=1.1,
                min_log_mass=-0.4,
                mean_program_mean_log_mass=0.2,
                mean_program_max_log_mass=0.8,
                mean_program_min_log_mass=-0.3,
                n_norm_checked=2,
                n_norm_with_log_mass=2,
                n_non_normalized=1,
                n_norm_cache_hits=1,
                n_positive_lh=1,
                frac_positive_lh=0.5,
                n_negative_lh=1,
                frac_negative_lh=0.5,
                positive_lh_reward_mean=2.0,
                non_positive_lh_reward_mean=1.0,
                positive_lh_reward_lift=1.0,
                n_unique_programs=3,
                n_unique_programs_exact=4,
                n_unique_valid_programs=2,
                n_unique_valid_programs_exact=3,
                unique_program_rate=0.75,
                unique_valid_program_rate=2 / 3,
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
    assert results["final_mean_log_mass"] == pytest.approx(0.2)
    assert results["mean_log_mass"] == pytest.approx(0.2)
    assert results["final_n_positive_lh"] == 1
    assert results["final_frac_positive_lh"] == pytest.approx(0.5)
    assert results["mean_n_positive_lh_per_batch"] == pytest.approx(1.0)
    assert results["mean_frac_positive_lh"] == pytest.approx(0.5)
    assert results["final_positive_lh_reward_lift"] == pytest.approx(1.0)
    assert results["mean_positive_lh_reward_lift"] == pytest.approx(1.0)
    assert results["final_n_unique_programs"] == 3
    assert results["final_n_unique_programs_exact"] == 4
    assert results["final_n_unique_valid_programs"] == 2
    assert results["final_n_unique_valid_programs_exact"] == 3
    assert results["final_unique_program_rate"] == pytest.approx(0.75)
    assert results["final_unique_valid_program_rate"] == pytest.approx(2 / 3)
    assert results["mean_n_unique_programs_per_batch"] == pytest.approx(3.0)
    assert results["mean_n_unique_valid_programs_per_batch"] == pytest.approx(2.0)
    assert results["paper/lh_count_batch_final"] == 1
    assert results["paper/lh_count_batch_mean"] == pytest.approx(1.0)
    assert results["paper/lh_positive_count_batch_final"] == 1
    assert results["paper/lh_positive_rate_final"] == pytest.approx(0.5)
    assert results["paper/lh_positive_reward_lift_final"] == pytest.approx(1.0)
    assert results["paper/unique_program_count_batch_final"] == 3
    assert results["paper/unique_valid_program_count_batch_final"] == 2
    assert results["paper/reward_metric"] == "singleton_posterior_predictive_logZ_ratio"
