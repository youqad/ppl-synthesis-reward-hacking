from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "trl_reward_hacking.py"
    spec = importlib.util.spec_from_file_location("trl_reward_hacking_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _FakeTrajectoryPoint:
    reward_mean: float
    n_valid: int
    n_total: int
    n_parse_fail: int = 0
    n_exec_fail: int = 0
    safety_flags: list[str] | None = None
    frac_non_normalized: float = 0.0
    mean_abs_log_mass: float = 0.0
    n_norm_checked: int = 0


def test_config_from_mapping_maps_fields() -> None:
    module = _load_module()
    cfg = module.config_from_mapping(
        {
            "model": "Qwen/Qwen3-1.7B",
            "n_steps": 10,
            "n_prompts": 20,
            "rollouts_per_prompt": 4,
            "lr": 2.0e-5,
            "output_dir": "artifacts/test_trl_cfg",
        }
    )
    assert cfg.model == "Qwen/Qwen3-1.7B"
    assert cfg.n_steps == 10
    assert cfg.n_prompts == 20
    assert cfg.rollouts_per_prompt == 4
    assert cfg.lr == 2.0e-5
    assert cfg.output_dir == "artifacts/test_trl_cfg"


def test_config_from_mapping_accepts_scalar_monitoring_mode_off() -> None:
    module = _load_module()
    cfg = module.config_from_mapping({"monitoring_mode": "off"})
    assert cfg.monitoring_mode == "off"


def test_config_from_mapping_accepts_neutral_prompt_policy() -> None:
    module = _load_module()
    cfg = module.config_from_mapping({"prompt_policy": "neutral"})
    assert cfg.prompt_policy == "neutral"


def test_config_from_mapping_rejects_invalid_prompt_policy() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="prompt_policy must be one of"):
        module.config_from_mapping({"prompt_policy": "anti_hint"})


def test_compute_results_exposes_sweep_aliases() -> None:
    module = _load_module()
    cfg = module.TRLRewardHackingConfig(model="Qwen/Qwen3-1.7B", n_steps=2)
    state = SimpleNamespace(
        trajectory=[
            _FakeTrajectoryPoint(
                reward_mean=-2.0,
                n_valid=8,
                n_total=10,
                n_parse_fail=1,
                n_exec_fail=1,
                safety_flags=[],
            ),
            _FakeTrajectoryPoint(
                reward_mean=0.0,
                n_valid=9,
                n_total=10,
                n_parse_fail=1,
                n_exec_fail=0,
                safety_flags=["example_flag"],
            ),
        ]
    )
    results = module._compute_results(cfg, state)
    assert results["final_reward_mean"] == results["final_reward"]
    assert results["final_safety_flags"] == ["example_flag"]


def test_config_from_mapping_accepts_nested_monitoring_mode_adaptive_fields() -> None:
    module = _load_module()
    cfg = module.config_from_mapping(
        {
            "monitoring_mode": {
                "monitoring_mode": "judge_evolving",
                "judge_interval": 10,
                "rubric_evolution_interval": 10,
                "judge_sampling_policy": "adaptive_cap",
                "judge_adaptive_min": 4,
                "judge_adaptive_max": 18,
                "judge_adaptive_target_metric": "combined",
            }
        }
    )
    assert cfg.monitoring_mode == "judge_evolving"
    assert cfg.judge_interval == 10
    assert cfg.rubric_evolution_interval == 10
    assert cfg.judge_sampling_policy == "adaptive_cap"
    assert cfg.judge_adaptive_min == 4
    assert cfg.judge_adaptive_max == 18
    assert cfg.judge_adaptive_target_metric == "combined"


def test_config_from_mapping_rejects_unknown_nested_monitoring_mode_key() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.monitoring_mode payload: monitoring_mode\.legacy_oracle",
    ):
        module.config_from_mapping(
            {
                "monitoring_mode": {
                    "monitoring_mode": "judge_evolving",
                    "legacy_oracle": 1.0,
                }
            }
        )


def test_config_from_mapping_rejects_nested_monitoring_mode_value_mapping() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.monitoring_mode payload: monitoring_mode\.monitoring_mode",
    ):
        module.config_from_mapping(
            {
                "monitoring_mode": {
                    "monitoring_mode": {"selected": "off"},
                }
            }
        )


def test_config_from_mapping_rejects_monitoring_payload_missing_mode_key() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.monitoring_mode payload: monitoring_mode\.judge_interval",
    ):
        module.config_from_mapping(
            {
                "monitoring_mode": {
                    "judge_interval": 10,
                }
            }
        )


def test_config_from_mapping_accepts_nested_sstan_gate_payload(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("TEST_KEY", "secret")
    cfg = module.config_from_mapping(
        {
            "sstan_gate": {
                "sstan_gate_mode": "shadow",
                "sstan_gate_sample_rate": 0.1,
                "sstan_gate_penalty_reward": -100.0,
                "sstan_gate_fidelity_policy": "reject_on_source_signal",
                "sstan_gate_log_stan": "none",
                "sstan_transpiler_model": "glm-5",
                "sstan_transpiler_api_key_env": "TEST_KEY",
                "sstan_transpiler_strict": True,
                "sstan_transpiler_timeout_s": 30.0,
            }
        }
    )
    assert cfg.sstan_gate_mode == "shadow"
    assert cfg.sstan_gate_sample_rate == 0.1
    assert cfg.sstan_transpiler_model == "glm-5"


def test_config_from_mapping_rejects_unknown_nested_sstan_gate_key() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.sstan_gate payload: sstan_gate\.legacy_oracle",
    ):
        module.config_from_mapping(
            {
                "sstan_gate": {
                    "sstan_gate_mode": "shadow",
                    "legacy_oracle": 1.0,
                }
            }
        )


def test_config_from_mapping_rejects_unknown_nested_scalar_group_key() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.reward_metric payload: reward_metric\.legacy_oracle",
    ):
        module.config_from_mapping(
            {
                "reward_metric": {
                    "reward_metric": "log_marginal_likelihood",
                    "legacy_oracle": 1.0,
                }
            }
        )


def test_config_from_mapping_rejects_nested_scalar_group_value_mapping() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.reward_metric payload: reward_metric\.reward_metric",
    ):
        module.config_from_mapping(
            {
                "reward_metric": {
                    "reward_metric": {"selected": "waic"},
                },
                "reward_estimator_backend": "waic",
            }
        )


def test_config_from_mapping_accepts_single_key_scalar_group_wrapper() -> None:
    module = _load_module()
    cfg = module.config_from_mapping({"reward_data_split": {"selected": "holdout"}})
    assert cfg.reward_data_split == "holdout"


def test_config_from_mapping_rejects_unknown_single_key_scalar_group_wrapper() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"Unsupported keys in train\.reward_metric payload: reward_metric\.typo",
    ):
        module.config_from_mapping(
            {
                "reward_metric": {"typo": "waic"},
                "reward_estimator_backend": "waic",
            }
        )


def test_config_from_mapping_unknown_key_fails_fast() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Unsupported train config keys"):
        module.config_from_mapping({"legacy_oracle": 1.0})


def test_config_from_mapping_rejects_invalid_monitoring_mode() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="monitoring_mode must be off\\|judge_evolving"):
        module.config_from_mapping({"monitoring_mode": {"monitoring_mode": "legacy"}})


def test_config_from_mapping_rejects_non_integer_monitoring_intervals() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="judge_interval must be a non-negative integer"):
        module.config_from_mapping({"judge_interval": "10"})
    with pytest.raises(ValueError, match="judge_interval must be a non-negative integer"):
        module.config_from_mapping({"judge_interval": True})

    with pytest.raises(
        ValueError,
        match="rubric_evolution_interval must be a non-negative integer",
    ):
        module.config_from_mapping({"rubric_evolution_interval": "10"})
    with pytest.raises(
        ValueError,
        match="rubric_evolution_interval must be a non-negative integer",
    ):
        module.config_from_mapping({"rubric_evolution_interval": True})


def test_config_from_mapping_requires_positive_monitoring_intervals_when_enabled() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match="monitoring_mode=judge_evolving requires judge_interval > 0",
    ):
        module.config_from_mapping({"monitoring_mode": {"monitoring_mode": "judge_evolving"}})


def test_config_from_mapping_rejects_non_mapping_group_payload() -> None:
    module = _load_module()
    with pytest.raises(
        ValueError,
        match=r"train\.normalization payload must be a mapping",
    ):
        module.config_from_mapping({"normalization": "exact_binary"})


def test_stale_claim_mode_fails_fast() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="claim_mode must be one of"):
        module.config_from_mapping({"claim_mode": "legacy_exploratory"})


def test_build_summary_emits_prompt_policy_key() -> None:
    module = _load_module()
    cfg = module.TRLRewardHackingConfig(prompt_policy="neutral")
    summary = module._build_summary(
        cfg,
        {
            "final_valid_rate": 0.8,
            "final_reward_mean": 0.1,
            "final_frac_non_normalized": 0.2,
        },
    )
    assert summary["paper/prompt_policy"] == "neutral"


def test_run_training_closes_completion_writer_on_train_failure(monkeypatch, tmp_path) -> None:
    module = _load_module()

    class _FakeWriter:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    writer = _FakeWriter()
    reward_state = SimpleNamespace(trajectory=[], completion_writer=writer)

    monkeypatch.setattr(module, "TRL_AVAILABLE", True)
    monkeypatch.setattr(module, "_prepare_output_dir", lambda _cfg: tmp_path)
    monkeypatch.setattr(module, "_resolve_resume_path", lambda _cfg: None)
    monkeypatch.setattr(module, "_load_train_dataset", lambda _cfg: ([], object()))
    monkeypatch.setattr(module, "_warn_monitoring_metadata_only", lambda _cfg: None)
    monkeypatch.setattr(module, "_build_reward_function", lambda _cfg, _out: (lambda *a, **k: [], reward_state))
    monkeypatch.setattr(module, "LoraConfig", lambda **_kwargs: object(), raising=False)
    monkeypatch.setattr(module, "_build_model_init_kwargs", lambda _cfg: {})
    monkeypatch.setattr(module, "_build_training_args", lambda _cfg, **_kwargs: object())
    monkeypatch.setattr(module, "_log_training_setup", lambda *_args, **_kwargs: None)

    class _RaisingTrainer:
        def __init__(self, **_kwargs) -> None:
            pass

        def train(self, **_kwargs) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(module, "GRPOTrainer", _RaisingTrainer, raising=False)

    with pytest.raises(RuntimeError, match="boom"):
        module.run_training(module.TRLRewardHackingConfig(output_dir=str(tmp_path)))

    assert writer.closed is True
