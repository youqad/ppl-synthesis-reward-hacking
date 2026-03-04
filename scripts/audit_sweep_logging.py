#!/usr/bin/env python3
"""CLI: audit W&B sweep logging completeness."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

from ppl_synthesis_reward_hacking.logging.run_logging_validation import (
    required_files_for_train_cfg,
    required_summary_keys_for_train_cfg,
    validate_run_logging,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit logging completeness for a W&B sweep")
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="W&B sweep id in entity/project/sweep form",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional env file for WANDB_API_KEY",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (defaults under .scratch/)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any run fails audit",
    )
    return parser.parse_args()


def _load_env(env_file: str | None) -> None:
    if not env_file:
        return
    from dotenv import load_dotenv

    load_dotenv(env_file)


def _extract_train_cfg(config: dict[str, Any]) -> dict[str, Any]:
    train = config.get("train")
    if isinstance(train, dict):
        return train
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(key, str) and key.startswith("train."):
            out[key.removeprefix("train.")] = value
    return out


def _collect_artifact_files(run) -> set[str]:
    files: set[str] = set()
    try:
        artifacts = list(run.logged_artifacts())
    except Exception:  # noqa: BLE001
        return files
    for artifact in artifacts:
        if getattr(artifact, "type", None) != "training-data":
            continue
        try:
            for entry in artifact.files():
                name = getattr(entry, "name", None)
                if isinstance(name, str) and name:
                    files.add(Path(name).name)
        except Exception:  # noqa: BLE001
            log.debug("failed to list files for artifact %s", artifact)
            continue
    return files


def _audit_run(run) -> dict[str, Any]:
    run_id = str(getattr(run, "id", "unknown"))
    run_name = str(getattr(run, "name", run_id))
    run_state = str(getattr(run, "state", "unknown"))
    summary = dict(getattr(run, "summary", {}) or {})
    config = dict(getattr(run, "config", {}) or {})
    train_cfg = _extract_train_cfg(config)
    required_summary = required_summary_keys_for_train_cfg(train_cfg)
    missing_summary = sorted(key for key in required_summary if key not in summary)

    output_dir = train_cfg.get("output_dir")
    local_report: dict[str, Any] | None = None
    if isinstance(output_dir, str) and output_dir.strip():
        run_dir = Path(output_dir).expanduser()
        if run_dir.exists():
            local_report = validate_run_logging(run_dir)

    required_files = required_files_for_train_cfg(train_cfg)
    artifact_files = _collect_artifact_files(run)
    missing_artifacts = sorted(name for name in required_files if name not in artifact_files)

    logging_valid_summary = summary.get("logging/valid_run")
    valid = (
        bool(logging_valid_summary is True)
        and not missing_summary
        and not missing_artifacts
        and (local_report is None or bool(local_report.get("valid", False)))
    )

    return {
        "run_id": run_id,
        "run_name": run_name,
        "state": run_state,
        "valid": valid,
        "logging_valid_summary": logging_valid_summary,
        "missing_summary_keys": missing_summary,
        "missing_artifact_files": missing_artifacts,
        "local_report": local_report,
    }


def main() -> None:
    args = _parse_args()
    _load_env(args.env_file)

    import wandb

    api = wandb.Api()
    sweep = api.sweep(args.sweep_id)
    rows = [_audit_run(run) for run in sweep.runs]
    n_total = len(rows)
    n_valid = sum(1 for row in rows if row["valid"])
    n_invalid = n_total - n_valid

    report = {
        "sweep_id": args.sweep_id,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "runs": rows,
    }

    out_path = args.out
    if out_path is None:
        safe_id = args.sweep_id.replace("/", "_")
        out_path = Path(".scratch") / f"sweep_logging_audit_{safe_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"sweep_id": args.sweep_id, "n_total": n_total, "n_invalid": n_invalid}))
    print(f"Wrote audit report: {out_path}")

    if args.strict and n_invalid > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
