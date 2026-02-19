from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

CONFIG_DIR = Path("configs/sweeps/wandb")
EXPECTED_SWEEP_FILES = {
    "tinker_lh_emergence.yaml": {
        "program": "scripts/hydra_train_tinker.py",
        "launcher": "local",
        "required_params": {
            "train.base_model",
            "train.dataset_n_features",
            "train.normalization_sample_size",
        },
    },
    "trl_lh_emergence.yaml": {
        "program": "scripts/hydra_train_trl.py",
        "launcher": "slurm",
        "required_params": {
            "train.model",
            "train.dataset_n_features",
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
