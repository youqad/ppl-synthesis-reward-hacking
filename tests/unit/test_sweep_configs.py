from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_tinker_sweep_config_has_reward_objective() -> None:
    cfg = _load_yaml(Path("configs/sweeps/wandb/tinker_gap.yaml"))
    assert cfg["program"] == "scripts/hydra_train_tinker.py"
    assert cfg["metric"]["name"] == "sweep/final_reward_mean"
    assert cfg["metric"]["goal"] == "maximize"
    assert cfg["command"][3] == "launcher=local"
    params = cfg.get("parameters", {})
    assert "train.p_true_mode" in params
    assert "train.p_true_fixed" in params
    assert "train.scoring_p_true" not in params


def test_trl_sweep_config_has_reward_objective() -> None:
    cfg = _load_yaml(Path("configs/sweeps/wandb/trl_gap.yaml"))
    assert cfg["program"] == "scripts/hydra_train_trl.py"
    assert cfg["metric"]["name"] == "sweep/final_reward_mean"
    assert cfg["metric"]["goal"] == "maximize"
    assert cfg["command"][3] == "launcher=slurm"
