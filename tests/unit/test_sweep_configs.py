from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from ppl_synthesis_reward_hacking.config.wandb_sweep_templates import (
    render_tinker_parta_sweeps,
)

CONFIG_DIR = Path("configs/sweeps/wandb")
EXPECTED_SWEEP_FILES = {
    "tinker_lh_emergence.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.data_interface.dataset_n_features",
            "train.normalization.normalization_sample_size",
        },
    },
    "trl_lh_emergence.yaml": {
        "program": "scripts/hydra_train_trl.py",
        "launcher": "slurm",
        "required_params": {
            "train.model",
            "train.data_interface.dataset_n_features",
        },
    },
    "tinker_partA_lh_main.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.data_interface.dataset_n_features",
            "train.normalization.normalization_sample_size",
        },
    },
    "tinker_partA_linear_extension.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
        },
    },
    "tinker_partA_linear_extension_fast.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
        },
    },
    "tinker_partA_metric_ablation_logml.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
        },
    },
    "tinker_partA_metric_ablation_elpd.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
        },
    },
    "tinker_partA_metric_ablation_waic.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
        },
    },
    "tinker_partA_metric_ablation_bic.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
        },
    },
    "e4_sstan_ab_baseline.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.data_interface.dataset_n_features",
            "train.normalization.normalization_sample_size",
        },
    },
    "e4_sstan_ab_enforce.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.data_interface.dataset_n_features",
            "train.normalization.normalization_sample_size",
        },
    },
    "e6_judge_gate.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.scoring_seed_base",
        },
    },
    "paper_a_base.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
            "train.data_interface.dataset_n_features",
        },
    },
    "paper_b_sstan.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
            "train.data_interface.dataset_n_features",
        },
    },
    "paper_b_judge.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.scoring_seed_base",
            "train.data_interface.dataset_n_features",
        },
    },
    "e5_antihint.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.data_interface.dataset_n_features",
            "train.normalization.normalization_sample_size",
        },
    },
}
_EXPECTED_BACKEND_BY_METRIC = {
    "log_marginal_likelihood": "smc",
    "elpd": "psis_loo",
    "waic": "waic",
    "bic": "bic_approx",
}


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _assert_lh_metric(cfg: dict) -> None:
    metric = cfg["metric"]
    assert metric["name"] == "paper/lh_formal_signal_final"
    assert metric["goal"] == "maximize"


def _assert_required_params(cfg: dict, required: Iterable[str]) -> None:
    params = cfg.get("parameters", {})
    for key in required:
        assert key in params


def _extract_command_value(command: list[object], prefixes: tuple[str, ...]) -> str | None:
    for entry in command:
        if not isinstance(entry, str):
            continue
        for prefix in prefixes:
            if entry.startswith(prefix):
                return entry.split("=", 1)[1]
    return None


def test_tinker_parta_sweeps_match_canonical_templates() -> None:
    generated = render_tinker_parta_sweeps()
    for filename, payload in generated.items():
        file_payload = _load_yaml(CONFIG_DIR / filename)
        assert file_payload == payload


def test_sweep_files_match() -> None:
    actual = {path.name for path in CONFIG_DIR.glob("*.yaml")}
    assert actual == set(EXPECTED_SWEEP_FILES)


def test_tinker_sweep_metric() -> None:
    cfg = _load_yaml(CONFIG_DIR / "tinker_lh_emergence.yaml")
    expected = EXPECTED_SWEEP_FILES["tinker_lh_emergence.yaml"]
    assert cfg["program"] == expected["program"]
    _assert_lh_metric(cfg)
    assert cfg["command"][3] == f"launcher={expected['launcher']}"
    _assert_required_params(cfg, expected["required_params"])


def test_trl_sweep_metric() -> None:
    cfg = _load_yaml(CONFIG_DIR / "trl_lh_emergence.yaml")
    expected = EXPECTED_SWEEP_FILES["trl_lh_emergence.yaml"]
    assert cfg["program"] == expected["program"]
    _assert_lh_metric(cfg)
    assert cfg["command"][3] == f"launcher={expected['launcher']}"
    _assert_required_params(cfg, expected["required_params"])


def test_new_tinker_sweeps_are_paper_track_formal_lh() -> None:
    files = [
        "tinker_partA_lh_main.yaml",
        "tinker_partA_linear_extension.yaml",
        "tinker_partA_linear_extension_fast.yaml",
        "tinker_partA_metric_ablation_logml.yaml",
        "tinker_partA_metric_ablation_elpd.yaml",
        "tinker_partA_metric_ablation_waic.yaml",
        "tinker_partA_metric_ablation_bic.yaml",
    ]
    for name in files:
        cfg = _load_yaml(CONFIG_DIR / name)
        assert cfg["program"] == "scripts/hydra_train_tinker.py"
        _assert_lh_metric(cfg)
        command = cfg.get("command", [])
        assert "train/paper_track=part_a_emergence" in command
        assert "train/claim_mode=formal_lh" in command


def test_main_and_linear_sweeps_enable_judge_monitoring() -> None:
    for name in ["tinker_partA_lh_main.yaml", "tinker_partA_linear_extension.yaml"]:
        cfg = _load_yaml(CONFIG_DIR / name)
        command = cfg.get("command", [])
        assert "train/monitoring_mode=judge_evolving" in command
        assert "train.monitoring_mode.judge_interval=10" in command
        assert "train.monitoring_mode.rubric_evolution_interval=10" in command
        assert "train.judge_batch_mode=auto" in command
        assert "train.judge_batch_max_workers=32" in command


def test_linear_fast_sweep_enables_judge_monitoring_with_fast_intervals() -> None:
    cfg = _load_yaml(CONFIG_DIR / "tinker_partA_linear_extension_fast.yaml")
    command = cfg.get("command", [])
    assert "train/monitoring_mode=judge_evolving" in command
    assert "train.monitoring_mode.judge_interval=20" in command
    assert "train.monitoring_mode.rubric_evolution_interval=20" in command
    assert "train.judge_batch_mode=auto" in command
    assert "train.judge_batch_max_workers=16" in command


def test_metric_ablation_sweeps_disable_judge_monitoring() -> None:
    files = [
        "tinker_partA_metric_ablation_logml.yaml",
        "tinker_partA_metric_ablation_elpd.yaml",
        "tinker_partA_metric_ablation_waic.yaml",
        "tinker_partA_metric_ablation_bic.yaml",
    ]
    for name in files:
        cfg = _load_yaml(CONFIG_DIR / name)
        command = cfg.get("command", [])
        assert "train/monitoring_mode=off" in command
        assert "train.monitoring_mode.judge_interval=0" in command
        assert "train.monitoring_mode.rubric_evolution_interval=0" in command


def test_paper_matrix_sweeps_use_d5_non_degenerate_train_split() -> None:
    files = [
        "paper_a_base.yaml",
        "paper_b_sstan.yaml",
        "paper_b_judge.yaml",
        "e6_judge_gate.yaml",
    ]
    for name in files:
        cfg = _load_yaml(CONFIG_DIR / name)
        command = cfg.get("command", [])
        assert "train/data_interface=raw_bernoulli_d5_ablation" in command
        assert "train.data_interface.dataset_n_train=20" in command
        assert "train/claim_mode=formal_lh" in command


def test_paper_matrix_mitigation_sweeps_enforce_strict_xor() -> None:
    checks = {
        "paper_b_sstan.yaml": ("train/sstan_gate=enforce", "train/judge_gate=off"),
        "paper_b_judge.yaml": ("train/sstan_gate=off", "train/judge_gate=enforce"),
        "e6_judge_gate.yaml": ("train/sstan_gate=off", "train/judge_gate=enforce"),
    }
    for name, (expected_sstan, expected_judge) in checks.items():
        cfg = _load_yaml(CONFIG_DIR / name)
        command = cfg.get("command", [])
        assert expected_sstan in command
        assert expected_judge in command
        sstan_overrides = [
            entry
            for entry in command
            if isinstance(entry, str) and entry.startswith("train/sstan_gate=")
        ]
        judge_overrides = [
            entry
            for entry in command
            if isinstance(entry, str) and entry.startswith("train/judge_gate=")
        ]
        assert sstan_overrides == [expected_sstan]
        assert judge_overrides == [expected_judge]


def test_paper_matrix_and_e6_pin_reward_split_to_train() -> None:
    files = [
        "paper_a_base.yaml",
        "paper_b_sstan.yaml",
        "paper_b_judge.yaml",
        "e6_judge_gate.yaml",
    ]
    for name in files:
        cfg = _load_yaml(CONFIG_DIR / name)
        command = cfg.get("command", [])
        assert "train/reward_data_split=train" in command


def test_paper_matrix_and_e6_seed_sets_are_expected() -> None:
    expected = {
        "paper_a_base.yaml": [0, 10000],
        "paper_b_sstan.yaml": [0, 10000],
        "paper_b_judge.yaml": [0, 10000],
        "e6_judge_gate.yaml": [42, 10042],
    }
    for name, seeds in expected.items():
        cfg = _load_yaml(CONFIG_DIR / name)
        params = cfg.get("parameters", {})
        seed_param = params.get("train.scoring_seed_base")
        assert isinstance(seed_param, dict), f"{name}: missing train.scoring_seed_base parameter"
        assert seed_param.get("values") == seeds


def test_mitigation_penalties_are_explicit_and_matched() -> None:
    sstan_cfg = _load_yaml(CONFIG_DIR / "paper_b_sstan.yaml")
    judge_cfg = _load_yaml(CONFIG_DIR / "paper_b_judge.yaml")
    e6_cfg = _load_yaml(CONFIG_DIR / "e6_judge_gate.yaml")

    sstan_cmd = sstan_cfg.get("command", [])
    judge_cmd = judge_cfg.get("command", [])
    e6_cmd = e6_cfg.get("command", [])

    assert "train.sstan_gate.sstan_gate_penalty_reward=-100" in sstan_cmd
    assert "train.judge_gate.judge_gate_penalty_reward=-100" in judge_cmd
    assert "train.judge_gate.judge_gate_penalty_reward=-100" in e6_cmd


def test_all_sweeps_use_compatible_metric_backend_pairs() -> None:
    for name in EXPECTED_SWEEP_FILES:
        cfg = _load_yaml(CONFIG_DIR / name)
        command = cfg.get("command", [])
        assert isinstance(command, list)
        metric = _extract_command_value(
            command,
            ("train/reward_metric=", "train.reward_metric="),
        )
        backend = _extract_command_value(
            command,
            ("train/reward_estimator_backend=", "train.reward_estimator_backend="),
        )
        assert metric is not None, f"{name}: reward metric override missing"
        assert backend is not None, f"{name}: reward backend override missing"
        expected_backend = _EXPECTED_BACKEND_BY_METRIC.get(metric)
        assert expected_backend is not None, f"{name}: unknown metric {metric}"
        assert backend == expected_backend, (
            f"{name}: metric={metric} requires backend={expected_backend}, got {backend}"
        )
