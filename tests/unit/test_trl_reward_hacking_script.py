from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


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
