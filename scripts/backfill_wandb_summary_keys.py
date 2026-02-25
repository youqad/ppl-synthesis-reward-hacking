#!/usr/bin/env python3
"""Backfill missing W&B summary keys for historical runs.

This is an offline maintenance utility. It does not affect runtime training code.
It fills missing strict-audit summary keys using, in order:
1) local results.json values (if available),
2) recoverable values from run config/summary/state,
3) contract defaults.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.logging.paper_summary_contract import PAPER_SUMMARY_DEFAULTS
from ppl_synthesis_reward_hacking.logging.run_logging_validation import (
    required_summary_keys_for_train_cfg,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing W&B summary keys")
    parser.add_argument("--entity", default="ox")
    parser.add_argument("--project", default="ppl-synthesis-reward-hacking")
    parser.add_argument(
        "--audit-glob",
        default=None,
        help=(
            "Glob for sweep audit reports. Defaults to "
            ".scratch/sweep_logging_audit_<entity>_<project>_*.json"
        ),
    )
    parser.add_argument(
        "--sweep-id",
        action="append",
        default=[],
        help="Optional sweep id(s) to filter (8-char id or full entity/project/id).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only; do not write W&B")
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional JSON report path (default under .scratch/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of runs to process (0 = no limit).",
    )
    return parser.parse_args()


def _load_env_file_if_present() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001
        return
    load_dotenv(env_path)


def _extract_train_cfg(config: dict[str, Any]) -> dict[str, Any]:
    train = config.get("train")
    if isinstance(train, dict):
        return train
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(key, str) and key.startswith("train."):
            out[key.removeprefix("train.")] = value
    return out


def _as_float_or_nan(value: Any) -> float:
    if isinstance(value, bool):
        return float("nan")
    if isinstance(value, int | float):
        f = float(value)
        if math.isfinite(f):
            return f
    return float("nan")


def _state_to_run_status(state: str) -> str:
    state_l = state.strip().lower()
    if state_l in {"finished", "running"}:
        return "success"
    if state_l in {"failed", "crashed", "killed", "cancelled", "canceled"}:
        return "fail"
    return "fail"


def _first_string(*candidates: Any, default: str = "unknown") -> str:
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item
    return default


def _first_finite_float(*candidates: Any) -> float:
    for item in candidates:
        val = _as_float_or_nan(item)
        if math.isfinite(val):
            return val
    return float("nan")


def _load_local_results(train_cfg: dict[str, Any], run_name: str) -> dict[str, Any]:
    candidate_dirs: list[Path] = []
    output_dir = train_cfg.get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        candidate_dirs.append(Path(output_dir).expanduser())
    if run_name:
        candidate_dirs.append(Path("artifacts/sweeps") / run_name)

    for run_dir in candidate_dirs:
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _canonical_sweep_id(sweep: str, *, entity: str, project: str) -> str:
    sweep = sweep.strip()
    if not sweep:
        return sweep
    if sweep.count("/") == 2:
        return sweep
    return f"{entity}/{project}/{sweep}"


def _normalized_sweep_filter(
    items: list[str],
    *,
    entity: str,
    project: str,
) -> set[str]:
    return {
        _canonical_sweep_id(item, entity=entity, project=project) for item in items if item.strip()
    }


@dataclass(slots=True, frozen=True)
class BackfillTarget:
    sweep_id: str
    run_id: str
    missing_summary_keys: tuple[str, ...]


def _load_targets_from_audits(
    *,
    entity: str,
    project: str,
    audit_glob: str,
    sweep_filter: set[str],
) -> list[BackfillTarget]:
    targets: list[BackfillTarget] = []
    for path in sorted(Path().glob(audit_glob)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        sweep_id = str(payload.get("sweep_id", "")).strip()
        if not sweep_id:
            continue
        if sweep_filter and sweep_id not in sweep_filter:
            continue
        runs = payload.get("runs")
        if not isinstance(runs, list):
            continue
        for row in runs:
            if not isinstance(row, dict):
                continue
            if bool(row.get("valid", False)):
                continue
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                continue
            missing_keys = row.get("missing_summary_keys")
            if not isinstance(missing_keys, list):
                missing_keys = []
            targets.append(
                BackfillTarget(
                    sweep_id=sweep_id,
                    run_id=run_id,
                    missing_summary_keys=tuple(str(k) for k in missing_keys if isinstance(k, str)),
                )
            )
    return targets


def _build_candidate_values(
    *,
    summary: dict[str, Any],
    train_cfg: dict[str, Any],
    local_results: dict[str, Any],
    run_state: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    out["sweep/run_status"] = _first_string(
        local_results.get("sweep/run_status"),
        summary.get("sweep/run_status"),
        _state_to_run_status(run_state),
        default="fail",
    )
    out["sweep/final_valid_rate"] = _first_finite_float(
        local_results.get("sweep/final_valid_rate"),
        summary.get("sweep/final_valid_rate"),
        local_results.get("final_valid_rate"),
        summary.get("final_valid_rate"),
    )
    out["sweep/final_lh_formal_signal"] = _first_finite_float(
        local_results.get("sweep/final_lh_formal_signal"),
        summary.get("sweep/final_lh_formal_signal"),
        local_results.get("paper/lh_formal_signal_final"),
        summary.get("paper/lh_formal_signal_final"),
        local_results.get("paper/frac_non_normalized_final"),
        summary.get("paper/frac_non_normalized_final"),
        local_results.get("final_frac_non_normalized"),
        summary.get("final_frac_non_normalized"),
    )

    out["paper/track"] = _first_string(
        local_results.get("paper/track"),
        summary.get("paper/track"),
        train_cfg.get("paper_track"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/track"]),
    )
    out["paper/claim_mode"] = _first_string(
        local_results.get("paper/claim_mode"),
        summary.get("paper/claim_mode"),
        train_cfg.get("claim_mode"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/claim_mode"]),
    )
    out["paper/reward_metric"] = _first_string(
        local_results.get("paper/reward_metric"),
        summary.get("paper/reward_metric"),
        train_cfg.get("reward_metric"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/reward_metric"]),
    )
    out["paper/reward_data_split"] = _first_string(
        local_results.get("paper/reward_data_split"),
        summary.get("paper/reward_data_split"),
        train_cfg.get("reward_data_split"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/reward_data_split"]),
    )
    out["paper/reward_estimator_backend"] = _first_string(
        local_results.get("paper/reward_estimator_backend"),
        summary.get("paper/reward_estimator_backend"),
        train_cfg.get("reward_estimator_backend"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/reward_estimator_backend"]),
    )
    out["paper/prompt_source"] = _first_string(
        local_results.get("paper/prompt_source"),
        summary.get("paper/prompt_source"),
        train_cfg.get("prompt_source"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/prompt_source"]),
    )
    out["paper/prompt_policy"] = _first_string(
        local_results.get("paper/prompt_policy"),
        summary.get("paper/prompt_policy"),
        train_cfg.get("prompt_policy"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/prompt_policy"]),
    )
    out["paper/thinking_mode"] = _first_string(
        local_results.get("paper/thinking_mode"),
        summary.get("paper/thinking_mode"),
        train_cfg.get("thinking_mode"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/thinking_mode"]),
    )
    out["paper/monitoring_mode"] = _first_string(
        local_results.get("paper/monitoring_mode"),
        summary.get("paper/monitoring_mode"),
        train_cfg.get("monitoring_mode"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/monitoring_mode"]),
    )
    out["paper/normalization_method"] = _first_string(
        local_results.get("paper/normalization_method"),
        summary.get("paper/normalization_method"),
        train_cfg.get("normalization_method"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/normalization_method"]),
    )
    out["paper/delta_scope"] = _first_string(
        local_results.get("paper/delta_scope"),
        summary.get("paper/delta_scope"),
        train_cfg.get("normalization_delta_scope"),
        default=str(PAPER_SUMMARY_DEFAULTS["paper/delta_scope"]),
    )

    out["paper/frac_non_normalized_final"] = _first_finite_float(
        local_results.get("paper/frac_non_normalized_final"),
        summary.get("paper/frac_non_normalized_final"),
        local_results.get("final_frac_non_normalized"),
        summary.get("final_frac_non_normalized"),
    )
    out["paper/lh_formal_signal_final"] = _first_finite_float(
        local_results.get("paper/lh_formal_signal_final"),
        summary.get("paper/lh_formal_signal_final"),
        out.get("paper/frac_non_normalized_final"),
    )
    out["paper/judge_hacking_rate_final"] = _first_finite_float(
        local_results.get("paper/judge_hacking_rate_final"),
        summary.get("paper/judge_hacking_rate_final"),
        local_results.get("judge/hacking_rate"),
        summary.get("judge/hacking_rate"),
    )
    out["paper/lh_family_prevalence_final"] = _first_finite_float(
        local_results.get("paper/lh_family_prevalence_final"),
        summary.get("paper/lh_family_prevalence_final"),
        local_results.get("heuristic/any_exploit_rate"),
        summary.get("heuristic/any_exploit_rate"),
    )
    out["paper/lh_rate_batch_final"] = _first_finite_float(
        local_results.get("paper/lh_rate_batch_final"),
        summary.get("paper/lh_rate_batch_final"),
        local_results.get("lh/batch/proportion"),
        summary.get("lh/batch/proportion"),
    )
    out["paper/lh_rate_batch_mean"] = _first_finite_float(
        local_results.get("paper/lh_rate_batch_mean"),
        summary.get("paper/lh_rate_batch_mean"),
    )
    out["paper/lh_other_novel_final"] = _first_finite_float(
        local_results.get("paper/lh_other_novel_final"),
        summary.get("paper/lh_other_novel_final"),
        local_results.get("lh/batch/other_novel_rate"),
        summary.get("lh/batch/other_novel_rate"),
    )
    out["paper/lh_other_novel_mean"] = _first_finite_float(
        local_results.get("paper/lh_other_novel_mean"),
        summary.get("paper/lh_other_novel_mean"),
    )

    return out


def _fill_value_for_key(key: str, *, candidates: dict[str, Any]) -> Any:
    if key in candidates:
        return candidates[key]
    if key in PAPER_SUMMARY_DEFAULTS:
        return PAPER_SUMMARY_DEFAULTS[key]
    if key == "sweep/run_status":
        return "fail"
    if key in {"sweep/final_lh_formal_signal", "sweep/final_valid_rate"}:
        return float("nan")
    return None


def main() -> None:
    args = _parse_args()
    _load_env_file_if_present()

    audit_glob = args.audit_glob
    if not audit_glob:
        audit_glob = f".scratch/sweep_logging_audit_{args.entity}_{args.project}_*.json"

    sweep_filter = _normalized_sweep_filter(
        list(args.sweep_id),
        entity=args.entity,
        project=args.project,
    )
    targets = _load_targets_from_audits(
        entity=args.entity,
        project=args.project,
        audit_glob=audit_glob,
        sweep_filter=sweep_filter,
    )
    if args.limit > 0:
        targets = targets[: args.limit]

    import wandb

    api = wandb.Api()

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "entity": args.entity,
        "project": args.project,
        "dry_run": bool(args.dry_run),
        "targets_total": len(targets),
        "runs_examined": 0,
        "runs_updated": 0,
        "runs_skipped": 0,
        "keys_written_total": 0,
        "rows": [],
    }

    for target in targets:
        run_path = f"{args.entity}/{args.project}/{target.run_id}"
        row: dict[str, Any] = {
            "sweep_id": target.sweep_id,
            "run_id": target.run_id,
            "run_path": run_path,
            "missing_from_audit": list(target.missing_summary_keys),
            "written": {},
            "status": "skipped",
        }
        try:
            run = api.run(run_path)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error_fetch_run"
            row["error"] = f"{type(exc).__name__}: {exc}"
            report["rows"].append(row)
            report["runs_skipped"] += 1
            continue

        report["runs_examined"] += 1
        summary = dict(getattr(run, "summary", {}) or {})
        config = dict(getattr(run, "config", {}) or {})
        train_cfg = _extract_train_cfg(config)
        required_keys = required_summary_keys_for_train_cfg(train_cfg)
        missing_now = [key for key in required_keys if key not in summary]
        if not missing_now:
            row["status"] = "already_complete"
            report["rows"].append(row)
            report["runs_skipped"] += 1
            continue

        run_name = str(getattr(run, "name", "") or "")
        run_state = str(getattr(run, "state", "") or "")
        local_results = _load_local_results(train_cfg, run_name)
        candidates = _build_candidate_values(
            summary=summary,
            train_cfg=train_cfg,
            local_results=local_results,
            run_state=run_state,
        )
        payload: dict[str, Any] = {}
        for key in missing_now:
            payload[key] = _fill_value_for_key(key, candidates=candidates)

        row["written"] = payload
        row["status"] = "would_update" if args.dry_run else "updated"

        if not args.dry_run:
            for key, value in payload.items():
                run.summary[key] = value
            run.summary["logging/backfilled_summary_keys_n"] = int(
                summary.get("logging/backfilled_summary_keys_n", 0)
            ) + len(payload)
            run.summary["logging/backfill_timestamp_utc"] = report["timestamp_utc"]
            run.summary.update()

        report["rows"].append(row)
        report["runs_updated"] += 1
        report["keys_written_total"] += len(payload)

    out_path = args.report_out
    if out_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = Path(".scratch") / f"wandb_summary_backfill_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "targets_total": report["targets_total"],
                "runs_examined": report["runs_examined"],
                "runs_updated": report["runs_updated"],
                "runs_skipped": report["runs_skipped"],
                "keys_written_total": report["keys_written_total"],
                "dry_run": report["dry_run"],
                "report": str(out_path),
            }
        )
    )


if __name__ == "__main__":
    main()
