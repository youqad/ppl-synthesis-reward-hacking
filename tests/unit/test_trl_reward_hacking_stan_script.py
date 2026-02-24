from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ppl_synthesis_reward_hacking.data.stan_reward_loader import load_stan_reward_prompts


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "trl_reward_hacking_stan.py"
    spec = importlib.util.spec_from_file_location("trl_reward_hacking_stan_script", script_path)
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
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "n_steps": 12,
            "n_prompts": 16,
            "rollouts_per_prompt": 4,
            "cmdstan_root": "/tmp/cmdstan",
            "checker_jobs": 3,
        }
    )
    assert cfg.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg.n_steps == 12
    assert cfg.n_prompts == 16
    assert cfg.rollouts_per_prompt == 4
    assert cfg.cmdstan_root == "/tmp/cmdstan"
    assert cfg.checker_jobs == 3


def test_config_from_mapping_rejects_unknown_keys() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="Unsupported train config keys: legacy_oracle"):
        module.config_from_mapping({"legacy_oracle": 1.0})


def test_default_run_name_has_stan_prefix() -> None:
    module = _load_module()
    cfg = module.TRLStanRewardConfig(model="Qwen/Qwen3-4B-Instruct-2507", n_steps=7)
    run_name = module._default_run_name(cfg)
    assert run_name.startswith("stan-grpo-qwen3-4b-instruct-2507-s7-")
    assert len(run_name) > 32


def test_script_loader_zero_prompts_respected() -> None:
    prompts = load_stan_reward_prompts(max_examples=0)
    assert prompts == []


def test_script_loader_none_defaults_to_20() -> None:
    prompts = load_stan_reward_prompts()
    assert len(prompts) == 20
