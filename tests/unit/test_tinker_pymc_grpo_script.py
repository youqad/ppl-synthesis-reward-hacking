from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "tinker_pymc_grpo.py"
    spec = importlib.util.spec_from_file_location("tinker_pymc_grpo_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_from_mapping_maps_fields() -> None:
    module = _load_module()
    cfg = module.config_from_mapping(
        {
            "base_model": "Qwen/Qwen3-1.7B",
            "n_steps": 12,
            "n_prompts": 4,
            "rollouts_per_prompt": 8,
            "learning_rate": 2.0e-5,
            "output_dir": "artifacts/test_tinker_cfg",
        }
    )
    assert cfg.base_model == "Qwen/Qwen3-1.7B"
    assert cfg.n_steps == 12
    assert cfg.n_prompts == 4
    assert cfg.rollouts_per_prompt == 8
    assert cfg.learning_rate == 2.0e-5
    assert cfg.output_dir == "artifacts/test_tinker_cfg"


def test_compute_results_exposes_sweep_aliases() -> None:
    module = _load_module()
    cfg = module.TinkerGRPOConfig(n_steps=2)
    trajectory = [
        module.TrajectoryPoint(
            step=0,
            reward_mean=-2.0,
            n_valid=4,
            n_total=5,
            n_parse_fail=0,
            n_exec_fail=1,
        ),
        module.TrajectoryPoint(
            step=1,
            reward_mean=1.0,
            n_valid=5,
            n_total=5,
            n_parse_fail=0,
            n_exec_fail=0,
        ),
    ]
    results = module._compute_results(cfg, trajectory)
    assert results["final_reward_mean"] == results["final_reward"]
