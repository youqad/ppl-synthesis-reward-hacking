from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_testmod"), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tinker_hydra_summary_success_has_reward_metric() -> None:
    module = _load_script("hydra_train_tinker.py")
    summary = module._build_sweep_summary(
        {
            "final_reward_mean": 1.2,
            "final_oracle_proxy_mean": -1.3,
            "final_gap_proxy_mean": 2.5,
            "final_valid_rate": 0.9,
        }
    )
    assert summary["sweep/run_status"] == "success"
    assert summary["sweep/final_reward_mean"] == 1.2


def test_trl_hydra_summary_failure_marks_fail() -> None:
    module = _load_script("hydra_train_trl.py")
    summary = module._build_sweep_summary({"error": "boom"})
    assert summary["sweep/run_status"] == "fail"
    assert "sweep/final_reward_mean" in summary
