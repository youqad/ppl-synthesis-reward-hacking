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
