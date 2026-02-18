#!/usr/bin/env python3
"""Shared helpers for Hydra-based training entrypoints."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.logging.wandb_hook import (
    finish_wandb,
    init_wandb,
    log_metrics,
    update_summary,
)

_PAPER_SUMMARY_KEYS = (
    "paper/track",
    "paper/claim_mode",
    "paper/reward_metric",
    "paper/reward_data_split",
    "paper/predictive_estimator",
    "paper/monitoring_mode",
    "paper/normalization_method",
    "paper/delta_scope",
    "paper/frac_non_normalized_final",
    "paper/lh_formal_signal_final",
    "paper/judge_hacking_rate_final",
    "paper/lh_family_prevalence_final",
    "paper/lh_rate_batch_final",
    "paper/lh_rate_batch_mean",
    "paper/lh_other_novel_final",
    "paper/lh_other_novel_mean",
)


def build_sweep_summary(results: dict[str, Any]) -> dict[str, Any]:
    if "error" in results:
        return {
            "sweep/run_status": "fail",
            "sweep/error": results.get("error", "unknown"),
            "sweep/final_lh_formal_signal": float("nan"),
        }
    final_reward = results.get("final_reward_mean", results.get("final_reward"))
    final_lh_signal = results.get(
        "paper/lh_formal_signal_final",
        results.get("final_frac_non_normalized", float("nan")),
    )
    final_valid_rate = results.get("final_valid_rate")
    run_status = str(results.get("sweep/run_status", "success"))
    error_reason = results.get("sweep/error")
    if run_status not in {"success", "fail"}:
        run_status = "success"
    if error_reason is not None:
        run_status = "fail"
    if run_status == "success":
        if isinstance(final_valid_rate, int | float) and final_valid_rate <= 0:
            run_status = "fail"
            error_reason = "no_valid_completions"
        elif isinstance(final_reward, int | float) and not math.isfinite(float(final_reward)):
            run_status = "fail"
            error_reason = "non_finite_final_reward"

    summary = {
        "sweep/run_status": run_status,
        "sweep/final_lh_formal_signal": final_lh_signal,
        "sweep/final_valid_rate": final_valid_rate,
    }
    if error_reason is not None:
        summary["sweep/error"] = error_reason
    for key in _PAPER_SUMMARY_KEYS:
        if key in results:
            summary[key] = results[key]
    return summary


def save_resolved_config(output_dir: str, cfg_dict: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hydra_resolved_config.json").write_text(
        json.dumps(cfg_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_from_cfg(
    cfg: Any,
    *,
    config_from_mapping: Callable[[dict[str, Any]], Any],
    run_training: Callable[[Any], dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
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
        logger.info("Initialized W&B run: %s", getattr(run, "name", "<unnamed>"))

    try:
        train_run_cfg = config_from_mapping(train_cfg)
        save_resolved_config(train_run_cfg.output_dir, cfg_dict)
        results = run_training(train_run_cfg)
        summary = build_sweep_summary(results)
        if wandb_enabled:
            log_metrics(summary)
            update_summary(summary)
        return results
    except Exception as exc:
        if wandb_enabled:
            failure_summary = {
                "sweep/run_status": "fail",
                "sweep/error": str(exc),
                "sweep/final_lh_formal_signal": float("nan"),
            }
            log_metrics(failure_summary)
            update_summary(failure_summary)
        raise
    finally:
        if wandb_enabled:
            finish_wandb()
