#!/usr/bin/env python3
"""Hydra entrypoint for TRL GRPO training + W&B sweep logging."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.logging.wandb_hook import (
    finish_wandb,
    init_wandb,
    log_metrics,
    update_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trl_reward_hacking import config_from_mapping, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_sweep_summary(results: dict[str, Any]) -> dict[str, Any]:
    if "error" in results:
        return {
            "sweep/run_status": "fail",
            "sweep/error": results.get("error", "unknown"),
            "sweep/final_reward_mean": float("nan"),
        }
    summary = {
        "sweep/run_status": "success",
        "sweep/final_reward_mean": results.get("final_reward_mean", results.get("final_reward")),
        "sweep/final_oracle_proxy_mean": results.get(
            "final_oracle_proxy_mean", results.get("final_oracle_proxy")
        ),
        "sweep/final_gap_proxy_mean": results.get(
            "final_gap_proxy_mean", results.get("final_gap_proxy")
        ),
        "sweep/final_valid_rate": results.get("final_valid_rate"),
    }
    for key in (
        "paper/track",
        "paper/reward_metric",
        "paper/reward_data_split",
        "paper/predictive_estimator",
        "paper/monitoring_mode",
        "paper/frac_non_normalized_final",
        "paper/judge_hacking_rate_final",
        "paper/lh_family_prevalence_final",
    ):
        if key in results:
            summary[key] = results[key]
    return summary


def _save_resolved_config(output_dir: str, cfg_dict: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hydra_resolved_config.json").write_text(
        json.dumps(cfg_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_from_cfg(cfg) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    train_cfg = cfg_dict.get("train", {})
    if not isinstance(train_cfg, dict):
        raise RuntimeError("Hydra config error: `train` must be a mapping")

    wandb_cfg = cfg_dict.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        wandb_cfg = {}

    run = None
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        mode = wandb_cfg.get("mode")
        if isinstance(mode, str) and mode.strip():
            os.environ["WANDB_MODE"] = mode
        run = init_wandb(
            project=str(wandb_cfg.get("project", "ppl-synthesis-reward-hacking")),
            entity=wandb_cfg.get("entity"),
            config=cfg_dict,
            name=wandb_cfg.get("name"),
            group=wandb_cfg.get("group"),
            job_type=wandb_cfg.get("job_type"),
            tags=list(wandb_cfg.get("tags", [])),
        )
        log.info("Initialized W&B run: %s", getattr(run, "name", "<unnamed>"))

    trl_cfg = config_from_mapping(train_cfg)
    _save_resolved_config(trl_cfg.output_dir, cfg_dict)

    try:
        results = run_training(trl_cfg)
        summary = _build_sweep_summary(results)
        if wandb_enabled:
            log_metrics(summary)
            update_summary(summary)
        return results
    except Exception as exc:
        if wandb_enabled:
            failure_summary = {
                "sweep/run_status": "fail",
                "sweep/error": str(exc),
                "sweep/final_reward_mean": float("nan"),
            }
            log_metrics(failure_summary)
            update_summary(failure_summary)
        raise
    finally:
        if wandb_enabled:
            finish_wandb()


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs/hydra", config_name="trl_train")
    def _entry(cfg: DictConfig) -> None:
        _run_from_cfg(cfg)

    _entry()


if __name__ == "__main__":
    main()
