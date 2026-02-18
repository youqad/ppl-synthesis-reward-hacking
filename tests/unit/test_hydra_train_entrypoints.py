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
            "final_valid_rate": 0.9,
            "paper/lh_formal_signal_final": 0.42,
        }
    )
    assert summary["sweep/run_status"] == "success"
    assert summary["sweep/final_lh_formal_signal"] == 0.42


def test_tinker_zero_valid() -> None:
    module = _load_script("hydra_train_tinker.py")
    summary = module._build_sweep_summary(
        {
            "final_reward_mean": float("nan"),
            "final_valid_rate": 0.0,
        }
    )
    assert summary["sweep/run_status"] == "fail"
    assert summary["sweep/error"] == "no_valid_completions"


def test_tinker_error_fails() -> None:
    module = _load_script("hydra_train_tinker.py")
    summary = module._build_sweep_summary(
        {
            "sweep/run_status": "success",
            "sweep/error": "upstream_error",
            "final_reward_mean": 0.2,
            "final_valid_rate": 0.5,
        }
    )
    assert summary["sweep/run_status"] == "fail"
    assert summary["sweep/error"] == "upstream_error"


def test_trl_hydra_summary_failure_marks_fail() -> None:
    module = _load_script("hydra_train_trl.py")
    summary = module._build_sweep_summary({"error": "boom"})
    assert summary["sweep/run_status"] == "fail"
    assert "sweep/final_lh_formal_signal" in summary


def test_trl_error_fails() -> None:
    module = _load_script("hydra_train_trl.py")
    summary = module._build_sweep_summary(
        {
            "sweep/run_status": "success",
            "sweep/error": "upstream_error",
            "final_reward_mean": 0.2,
            "final_valid_rate": 0.5,
        }
    )
    assert summary["sweep/run_status"] == "fail"
    assert summary["sweep/error"] == "upstream_error"
