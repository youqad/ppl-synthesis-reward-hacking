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

from ppl_synthesis_reward_hacking.logging.paper_summary_contract import (
    PAPER_SUMMARY_DEFAULTS,
    PAPER_SUMMARY_KEYS_ALL,
)
from ppl_synthesis_reward_hacking.logging.run_logging_validation import validate_run_logging
from ppl_synthesis_reward_hacking.logging.wandb_hook import (
    finish_wandb,
    init_wandb,
    log_metrics,
    update_summary,
)

_SUMMARY_SWEEP_KEYS: tuple[str, ...] = (
    "sweep/run_status",
    "sweep/final_lh_formal_signal",
    "sweep/final_valid_rate",
)


def _coerce_status(
    *,
    run_status: str,
    error_reason: Any,
    final_valid_rate: Any,
    final_reward: Any,
) -> tuple[str, Any]:
    status = run_status if run_status in {"success", "fail"} else "success"
    if error_reason is not None:
        return "fail", error_reason
    if status != "success":
        return status, error_reason
    if isinstance(final_valid_rate, int | float) and final_valid_rate <= 0:
        return "fail", "no_valid_completions"
    if isinstance(final_reward, int | float) and not math.isfinite(float(final_reward)):
        return "fail", "non_finite_final_reward"
    return "success", None


def _paper_summary_from_results(results: dict[str, Any]) -> dict[str, Any]:
    return {key: results.get(key, PAPER_SUMMARY_DEFAULTS[key]) for key in PAPER_SUMMARY_KEYS_ALL}


def _logging_channels(results: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in results.items()
        if isinstance(key, str) and key.startswith("logging/")
    }


def build_sweep_summary(results: dict[str, Any]) -> dict[str, Any]:
    if "error" in results:
        return {
            "sweep/run_status": "fail",
            "sweep/error": results.get("error", "unknown"),
            "sweep/final_lh_formal_signal": float("nan"),
            "sweep/final_valid_rate": float("nan"),
        }
    final_reward = results.get("final_reward_mean", results.get("final_reward"))
    final_lh_signal = results.get(
        "paper/lh_formal_signal_final",
        results.get("final_frac_non_normalized", float("nan")),
    )
    final_valid_rate = results.get("final_valid_rate")
    run_status, error_reason = _coerce_status(
        run_status=str(results.get("sweep/run_status", "success")),
        error_reason=results.get("sweep/error"),
        final_valid_rate=final_valid_rate,
        final_reward=final_reward,
    )

    summary = {
        "sweep/run_status": run_status,
        "sweep/final_lh_formal_signal": final_lh_signal,
        "sweep/final_valid_rate": final_valid_rate,
    }
    if error_reason is not None:
        summary["sweep/error"] = error_reason
    summary.update(_paper_summary_from_results(results))
    summary.update(_logging_channels(results))
    return summary


def build_validation_summary_payload(results: dict[str, Any]) -> dict[str, Any]:
    """Build summary payload for run logging validation.

    Validation should fail if required paper keys were never emitted by runtime.
    We only backfill derived sweep fields, not paper fields with defaults.
    """
    payload = dict(results)
    sweep_summary = build_sweep_summary(results)
    for key in ("sweep/run_status", "sweep/final_lh_formal_signal", "sweep/final_valid_rate"):
        if key in sweep_summary:
            payload.setdefault(key, sweep_summary[key])
    return payload


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


def _maybe_init_wandb(
    *,
    wandb_cfg: dict[str, Any],
    cfg_dict: dict[str, Any],
    logger: logging.Logger,
) -> bool:
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled:
        return False
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
    return True


def _validate_logging_and_update_results(
    *,
    results: dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
) -> tuple[bool, dict[str, Any]]:
    summary_for_validation = build_validation_summary_payload(results)
    validation_report = validate_run_logging(
        Path(output_dir),
        summary_payload=summary_for_validation,
    )
    results.update(_logging_metrics(validation_report))
    invalid_logging = not validation_report.get("valid", False)
    if invalid_logging:
        results["sweep/run_status"] = "fail"
        results.setdefault("sweep/error", "logging_validation_failed")
        logger.warning(
            "Logging validation failed for %s: missing=%s summary=%s parse=%s",
            output_dir,
            validation_report.get("missing_files", []),
            validation_report.get("missing_summary_keys", []),
            validation_report.get("parse_errors", []),
        )
    return invalid_logging, validation_report


def _raise_logging_validation_error(validation_report: dict[str, Any]) -> None:
    raise RuntimeError(
        "logging_validation_failed: "
        f"missing_files={validation_report.get('missing_files', [])}, "
        f"missing_summary_keys={validation_report.get('missing_summary_keys', [])}, "
        f"parse_errors={validation_report.get('parse_errors', [])}"
    )


def _build_failure_summary(exc: Exception, results: dict[str, Any] | None) -> dict[str, Any]:
    failure_summary = {
        "sweep/run_status": "fail",
        "sweep/error": str(exc),
        "sweep/final_lh_formal_signal": float("nan"),
        "sweep/final_valid_rate": float("nan"),
    }
    if not isinstance(results, dict):
        return failure_summary
    failure_summary["sweep/final_lh_formal_signal"] = results.get(
        "sweep/final_lh_formal_signal",
        results.get("paper/lh_formal_signal_final", float("nan")),
    )
    failure_summary["sweep/final_valid_rate"] = results.get(
        "sweep/final_valid_rate",
        results.get("final_valid_rate", float("nan")),
    )
    for key in PAPER_SUMMARY_KEYS_ALL:
        if key in results:
            failure_summary[key] = results[key]
    failure_summary.update(_logging_channels(results))
    return failure_summary


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

    wandb_enabled = _maybe_init_wandb(wandb_cfg=wandb_cfg, cfg_dict=cfg_dict, logger=logger)
    results: dict[str, Any] | None = None

    try:
        train_run_cfg = config_from_mapping(train_cfg)
        save_resolved_config(train_run_cfg.output_dir, cfg_dict)
        results = run_training(train_run_cfg)
        invalid_logging, validation_report = _validate_logging_and_update_results(
            results=results,
            output_dir=train_run_cfg.output_dir,
            logger=logger,
        )
        _save_results(train_run_cfg.output_dir, results)
        summary = build_sweep_summary(results)
        if wandb_enabled:
            log_metrics(summary)
            update_summary(summary)
        if invalid_logging:
            _raise_logging_validation_error(validation_report)
        return results
    except Exception as exc:
        if wandb_enabled:
            failure_summary = _build_failure_summary(exc, results)
            log_metrics(failure_summary)
            update_summary(failure_summary)
        raise
    finally:
        if wandb_enabled:
            finish_wandb()


def build_cfg_runner(
    *,
    config_from_mapping: Callable[[dict[str, Any]], Any],
    run_training: Callable[[Any], dict[str, Any]],
    logger: logging.Logger,
) -> Callable[[Any], dict[str, Any]]:
    """Create a script-local cfg runner bound to concrete training functions."""

    def _runner(cfg: Any) -> dict[str, Any]:
        return run_from_cfg(
            cfg,
            config_from_mapping=config_from_mapping,
            run_training=run_training,
            logger=logger,
        )

    return _runner


def run_hydra_entrypoint(
    *,
    config_name: str,
    run_cfg: Callable[[Any], dict[str, Any]],
) -> None:
    """Run a Hydra script entrypoint with a shared config path."""
    import hydra
    from omegaconf import DictConfig

    config_path = str((Path(__file__).resolve().parents[1] / "configs" / "hydra").resolve())

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def _entry(cfg: DictConfig) -> None:
        run_cfg(cfg)

    _entry()
