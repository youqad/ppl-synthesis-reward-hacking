#!/usr/bin/env python3
"""Common plumbing for Hydra train scripts."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.logging.run_logging_validation import validate_run_logging
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

_PAPER_SUMMARY_DEFAULTS: dict[str, Any] = {
    "paper/track": "unknown",
    "paper/claim_mode": "unknown",
    "paper/reward_metric": "unknown",
    "paper/reward_data_split": "unknown",
    "paper/predictive_estimator": "unknown",
    "paper/monitoring_mode": "unknown",
    "paper/normalization_method": "unknown",
    "paper/delta_scope": "unknown",
    "paper/frac_non_normalized_final": float("nan"),
    "paper/lh_formal_signal_final": float("nan"),
    "paper/judge_hacking_rate_final": float("nan"),
    "paper/lh_family_prevalence_final": float("nan"),
    "paper/lh_rate_batch_final": float("nan"),
    "paper/lh_rate_batch_mean": float("nan"),
    "paper/lh_other_novel_final": float("nan"),
    "paper/lh_other_novel_mean": float("nan"),
}


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
        summary[key] = results.get(key, _PAPER_SUMMARY_DEFAULTS[key])
    for key, value in results.items():
        if isinstance(key, str) and key.startswith("logging/"):
            summary[key] = value
    return summary


def save_resolved_config(output_dir: str, cfg_dict: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hydra_resolved_config.json").write_text(
        json.dumps(cfg_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_results(output_dir: str, results: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _logging_metrics(report: dict[str, Any]) -> dict[str, Any]:
    missing_files = report.get("missing_files")
    missing_summary = report.get("missing_summary_keys")
    parse_errors = report.get("parse_errors")
    missing_files_n = len(missing_files) if isinstance(missing_files, list) else 0
    missing_summary_n = len(missing_summary) if isinstance(missing_summary, list) else 0
    parse_errors_n = len(parse_errors) if isinstance(parse_errors, list) else 0
    return {
        "logging/valid_run": bool(report.get("valid", False)),
        "logging/missing_files_n": missing_files_n,
        "logging/missing_summary_keys_n": missing_summary_n,
        "logging/parse_errors_n": parse_errors_n,
    }


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
        summary_for_validation = {**results, **build_sweep_summary(results)}
        validation_report = validate_run_logging(
            Path(train_run_cfg.output_dir),
            summary_payload=summary_for_validation,
        )
        logging_metrics = _logging_metrics(validation_report)
        results.update(logging_metrics)
        invalid_logging = not validation_report.get("valid", False)
        if invalid_logging:
            results["sweep/run_status"] = "fail"
            if "sweep/error" not in results:
                results["sweep/error"] = "logging_validation_failed"
            logger.warning(
                "Logging validation failed for %s: missing=%s summary=%s parse=%s",
                train_run_cfg.output_dir,
                validation_report.get("missing_files", []),
                validation_report.get("missing_summary_keys", []),
                validation_report.get("parse_errors", []),
            )
        _save_results(train_run_cfg.output_dir, results)
        summary = build_sweep_summary(results)
        if wandb_enabled:
            log_metrics(summary)
            update_summary(summary)
        if invalid_logging:
            raise RuntimeError(
                "logging_validation_failed: "
                f"missing_files={validation_report.get('missing_files', [])}, "
                f"missing_summary_keys={validation_report.get('missing_summary_keys', [])}, "
                f"parse_errors={validation_report.get('parse_errors', [])}"
            )
        return results
    except Exception as exc:
        if wandb_enabled:
            failure_summary = {
                "sweep/run_status": "fail",
                "sweep/error": str(exc),
                "sweep/final_lh_formal_signal": float("nan"),
            }
            if "results" in locals() and isinstance(results, dict):
                final_lh = results.get(
                    "sweep/final_lh_formal_signal",
                    results.get("paper/lh_formal_signal_final", float("nan")),
                )
                failure_summary["sweep/final_lh_formal_signal"] = final_lh
                for key in _PAPER_SUMMARY_KEYS:
                    if key in results:
                        failure_summary[key] = results[key]
                for key, value in results.items():
                    if isinstance(key, str) and key.startswith("logging/"):
                        failure_summary[key] = value
            log_metrics(failure_summary)
            update_summary(failure_summary)
        raise
    finally:
        if wandb_enabled:
            finish_wandb()
