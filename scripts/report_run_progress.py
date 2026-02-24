#!/usr/bin/env python3
"""Live progress/ETA report for training artifact directories.

Examples:
  python scripts/report_run_progress.py
  python scripts/report_run_progress.py --watch-seconds 60
  python scripts/report_run_progress.py --run-dir artifacts/sweeps/tinker_20260223_103211_337082
  python scripts/report_run_progress.py --json --out .scratch/live_progress.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STEP_RECEIVED_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\].*Step\s+(?P<step>\d+)\s+prompt\s+"
    r"(?P<prompt>\d+)/(?P<n_prompts>\d+):\s+received\s+"
    r"(?P<count>\d+)\s+rollouts\s+in\s+(?P<duration>[0-9.]+)s$"
)
STEP_SUMMARY_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\].*Step\s+(?P<step>\d+):\s+reward=(?P<reward>-?[0-9.]+)\s+"
    r"valid=(?P<valid>\d+)/(?P<total>\d+)\s+parse_fail=(?P<parse_fail>\d+)\s+"
    r"exec_fail=(?P<exec_fail>\d+)\s+frac_zero_var=(?P<frac_zero_var>-?[0-9.]+)\s+"
    r"norm_checked=(?P<norm_checked>\d+)\s+frac_non_norm=(?P<frac_non_norm>-?[0-9.]+)\s+"
    r"mean_abs_log_mass=(?P<mean_abs_log_mass>-?[0-9.]+)$"
)
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S,%f"


@dataclass
class ReceivedEvent:
    ts: dt.datetime
    count: int
    duration_s: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report run progress, rolling throughput, and ETA")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/sweeps"),
        help="Root directory containing run subdirectories",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Explicit run directory (repeatable). If omitted, auto-discover from --root",
    )
    parser.add_argument(
        "--glob",
        default="tinker_*",
        help="Glob used for auto-discovery under --root",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=12,
        help="Max auto-discovered runs before active/stale filtering",
    )
    parser.add_argument(
        "--active-within-minutes",
        type=float,
        default=240.0,
        help="Keep only runs whose log mtime is within this many minutes (0 disables filter)",
    )
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=180.0,
        help="Rolling window for throughput estimate from 'received rollouts' events",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=45.0,
        help="Mark run as stale if no log updates for this many minutes",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=2500,
        help="Max trailing completions lines used for last-batch gate metrics",
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Refresh interval in seconds (0 prints once and exits)",
    )
    parser.add_argument(
        "--only-active",
        action="store_true",
        help="Show only runs with status active or waiting",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report instead of table",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (written every refresh)",
    )
    return parser.parse_args()


def _coerce_train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    train = cfg.get("train")
    if isinstance(train, dict):
        return train
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(key, str) and key.startswith("train."):
            out[key.removeprefix("train.")] = value
    return out


def _flatten_one_level(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                out[f"{key}.{sub_key}"] = sub_val
        else:
            out[key] = value
    return out


def _list_run_dirs(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(p) for p in args.run_dir]
    if explicit:
        return explicit

    root = args.root
    if not root.exists():
        return []

    candidates = [p for p in root.glob(args.glob) if p.is_dir()]
    candidates = [p for p in candidates if (p / "hydra_resolved_config.json").exists()]
    candidates.sort(
        key=lambda p: (p / "hydra_train_common.log").stat().st_mtime
        if (p / "hydra_train_common.log").exists()
        else p.stat().st_mtime,
        reverse=True,
    )
    candidates = candidates[: max(0, int(args.latest))]

    if args.active_within_minutes <= 0:
        return candidates

    now = time.time()
    out: list[Path] = []
    max_age_s = float(args.active_within_minutes) * 60.0
    for run_dir in candidates:
        log_path = run_dir / "hydra_train_common.log"
        mtime = log_path.stat().st_mtime if log_path.exists() else run_dir.stat().st_mtime
        if now - mtime <= max_age_s:
            out.append(run_dir)
    return out


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as handle:
        for _ in handle:
            n += 1
    return n


def _parse_received_events(log_path: Path) -> list[ReceivedEvent]:
    events: list[ReceivedEvent] = []
    if not log_path.exists():
        return events
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            match = STEP_RECEIVED_RE.match(line)
            if not match:
                continue
            try:
                ts = dt.datetime.strptime(match.group("ts"), TIMESTAMP_FMT)
            except ValueError:
                continue
            try:
                count = int(match.group("count"))
                duration = float(match.group("duration"))
            except ValueError:
                continue
            events.append(ReceivedEvent(ts=ts, count=count, duration_s=duration))
    return events


def _parse_last_step_summary(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    last: dict[str, Any] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            match = STEP_SUMMARY_RE.match(line)
            if not match:
                continue
            try:
                ts = dt.datetime.strptime(match.group("ts"), TIMESTAMP_FMT)
            except ValueError:
                continue
            last = {
                "ts": ts.isoformat(),
                "step": int(match.group("step")),
                "reward": float(match.group("reward")),
                "valid": int(match.group("valid")),
                "total": int(match.group("total")),
                "parse_fail": int(match.group("parse_fail")),
                "exec_fail": int(match.group("exec_fail")),
                "frac_zero_var": float(match.group("frac_zero_var")),
                "norm_checked": int(match.group("norm_checked")),
                "frac_non_norm": float(match.group("frac_non_norm")),
                "mean_abs_log_mass": float(match.group("mean_abs_log_mass")),
            }
    return last


def _load_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _tail_jsonl(path: Path, max_lines: int) -> list[dict[str, Any]]:
    if not path.exists() or max_lines <= 0:
        return []
    q: deque[str] = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text:
                q.append(text)
    out: list[dict[str, Any]] = []
    for text in q:
        try:
            out.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return out


def _compute_rate(
    events: list[ReceivedEvent], *, now: dt.datetime, window_minutes: float
) -> float | None:
    if not events:
        return None
    min_ts = now - dt.timedelta(minutes=window_minutes)
    in_window = [event for event in events if event.ts >= min_ts]
    if len(in_window) < 2:
        in_window = events
    if len(in_window) < 2:
        event = in_window[0]
        if event.duration_s > 0:
            return float(event.count) / event.duration_s
        return None

    total = float(sum(event.count for event in in_window))
    span = (in_window[-1].ts - in_window[0].ts).total_seconds()
    if span <= 0:
        duration_total = float(sum(max(event.duration_s, 0.0) for event in in_window))
        if duration_total <= 0:
            return None
        return total / duration_total
    return total / span


def _iso_or_none(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts).isoformat()


def _format_eta(hours: float | None) -> str:
    if hours is None or not math.isfinite(hours):
        return "-"
    if hours >= 24:
        return f"{hours / 24.0:.1f}d"
    return f"{hours:.1f}h"


def _format_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{100.0 * value:5.1f}%"


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _status(
    *,
    complete: bool,
    log_age_minutes: float | None,
    stale_minutes: float,
    rate_per_h: float | None,
) -> str:
    if complete:
        return "completed"
    if log_age_minutes is None:
        return "unknown"
    if log_age_minutes > stale_minutes:
        return "stale"
    if rate_per_h is None or rate_per_h <= 0:
        return "waiting"
    return "active"


def _gate_mode(train_cfg: dict[str, Any]) -> str:
    sstan = train_cfg.get("sstan_gate")
    judge = train_cfg.get("judge_gate")
    if isinstance(judge, dict) and str(judge.get("judge_gate_mode", "off")).lower() != "off":
        return "judge"
    if isinstance(sstan, dict) and str(sstan.get("sstan_gate_mode", "off")).lower() != "off":
        return "sstan"
    return "off"


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _last_batch_metrics(tail_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tail_records:
        return None
    batches = [
        record.get("batch") for record in tail_records if isinstance(record.get("batch"), int)
    ]
    if not batches:
        return None
    batch_id = max(batches)
    rows = [record for record in tail_records if record.get("batch") == batch_id]
    if not rows:
        return None

    n = len(rows)
    valid = sum(1 for row in rows if row.get("outcome") == "valid")
    sstan_checked = 0
    sstan_reject = 0
    sstan_transpile_fail = 0
    sstan_penalty = 0
    judge_checked = 0
    judge_hacking = 0
    judge_penalty = 0

    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue
        sstan = meta.get("sstan_gate")
        if isinstance(sstan, dict):
            if bool(sstan.get("checked")):
                sstan_checked += 1
            if str(sstan.get("decision", "")).lower() == "reject":
                sstan_reject += 1
            if sstan.get("transpile_success") is False:
                sstan_transpile_fail += 1
            if bool(sstan.get("penalty_applied")):
                sstan_penalty += 1

        judge = meta.get("judge_gate")
        if isinstance(judge, dict):
            if bool(judge.get("checked")):
                judge_checked += 1
            if str(judge.get("verdict", "")).lower() == "hacking":
                judge_hacking += 1
            if bool(judge.get("penalty_applied")):
                judge_penalty += 1

    return {
        "batch": batch_id,
        "n": n,
        "valid_rate": _safe_div(float(valid), float(n)),
        "sstan_checked": sstan_checked,
        "sstan_reject": sstan_reject,
        "sstan_reject_rate": _safe_div(float(sstan_reject), float(max(1, sstan_checked))),
        "sstan_transpile_fail": sstan_transpile_fail,
        "sstan_transpile_fail_rate": _safe_div(
            float(sstan_transpile_fail), float(max(1, sstan_checked))
        ),
        "sstan_penalty": sstan_penalty,
        "judge_checked": judge_checked,
        "judge_hacking": judge_hacking,
        "judge_hacking_rate": _safe_div(float(judge_hacking), float(max(1, judge_checked))),
        "judge_penalty": judge_penalty,
    }


def _analyze_run(
    run_dir: Path, *, window_minutes: float, stale_minutes: float, tail_lines: int
) -> dict[str, Any]:
    now = dt.datetime.now()
    cfg_path = run_dir / "hydra_resolved_config.json"
    log_path = run_dir / "hydra_train_common.log"
    completions_path = run_dir / "completions.jsonl"
    normalization_path = run_dir / "normalization_metrics.jsonl"
    judge_path = run_dir / "judge_metrics.jsonl"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    train_cfg = _coerce_train_cfg(cfg)
    flat_train = _flatten_one_level(train_cfg)

    n_steps = int(flat_train.get("n_steps", 0) or 0)
    n_prompts = int(flat_train.get("n_prompts", 0) or 0)
    rollouts_per_prompt = int(flat_train.get("rollouts_per_prompt", 0) or 0)
    per_step = n_prompts * rollouts_per_prompt
    expected = n_steps * per_step
    completions = _count_jsonl_lines(completions_path)

    progress = _safe_div(float(completions), float(expected)) if expected > 0 else None
    step_float = _safe_div(float(completions), float(per_step)) if per_step > 0 else None
    step_completed = int(completions // per_step) if per_step > 0 else None

    events = _parse_received_events(log_path)
    rate_per_s_window = _compute_rate(events, now=now, window_minutes=window_minutes)
    rate_per_s_global = _compute_rate(events, now=now, window_minutes=10_000_000.0)
    rate_candidates = [x for x in [rate_per_s_window, rate_per_s_global] if x is not None and x > 0]
    rate_per_s_conservative = min(rate_candidates) if rate_candidates else None

    rate_per_h = rate_per_s_window * 3600.0 if rate_per_s_window is not None else None
    rate_per_h_global = rate_per_s_global * 3600.0 if rate_per_s_global is not None else None
    rate_per_h_conservative = (
        rate_per_s_conservative * 3600.0 if rate_per_s_conservative is not None else None
    )
    rate_per_m = rate_per_s_window * 60.0 if rate_per_s_window is not None else None

    remaining = max(0, expected - completions) if expected > 0 else 0
    eta_h = (
        _safe_div(float(remaining), float(rate_per_h)) if rate_per_h and rate_per_h > 0 else None
    )
    eta_h_conservative = (
        _safe_div(float(remaining), float(rate_per_h_conservative))
        if rate_per_h_conservative and rate_per_h_conservative > 0
        else None
    )

    log_mtime = log_path.stat().st_mtime if log_path.exists() else None
    log_age_min = ((now.timestamp() - log_mtime) / 60.0) if log_mtime is not None else None
    complete = expected > 0 and completions >= expected

    run = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "paper_track": flat_train.get("paper_track.paper_track"),
        "claim_mode": flat_train.get("claim_mode.claim_mode"),
        "prompt_policy": flat_train.get("prompt_policy.prompt_policy"),
        "gate": _gate_mode(train_cfg),
        "n_steps": n_steps,
        "n_prompts": n_prompts,
        "rollouts_per_prompt": rollouts_per_prompt,
        "per_step": per_step,
        "expected_completions": expected,
        "completions": completions,
        "remaining_completions": remaining,
        "progress": progress,
        "step_progress_float": step_float,
        "step_completed_count": step_completed,
        "rate_rollouts_per_min": rate_per_m,
        "rate_rollouts_per_hour": rate_per_h,
        "rate_rollouts_per_hour_global": rate_per_h_global,
        "rate_rollouts_per_hour_conservative": rate_per_h_conservative,
        "eta_hours": eta_h,
        "eta_hours_conservative": eta_h_conservative,
        "eta_at": (now + dt.timedelta(hours=eta_h)).isoformat() if eta_h else None,
        "eta_at_conservative": (
            (now + dt.timedelta(hours=eta_h_conservative)).isoformat()
            if eta_h_conservative
            else None
        ),
        "log_mtime": _iso_or_none(log_mtime),
        "log_age_minutes": log_age_min,
        "status": _status(
            complete=complete,
            log_age_minutes=log_age_min,
            stale_minutes=stale_minutes,
            rate_per_h=rate_per_h,
        ),
    }

    run["last_step_summary"] = _parse_last_step_summary(log_path)
    run["last_normalization"] = _load_last_jsonl(normalization_path)
    run["last_judge_metrics"] = _load_last_jsonl(judge_path)
    run["last_batch_metrics"] = _last_batch_metrics(
        _tail_jsonl(completions_path, max_lines=tail_lines)
    )
    return run


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "gate",
        "status",
        "progress",
        "step",
        "rate/h",
        "eta_win",
        "eta_cons",
        "valid",
        "non_norm",
        "judge_hack",
        "upd_age_m",
    ]
    table_rows: list[list[str]] = []
    for row in rows:
        last = row.get("last_step_summary") or {}
        batch = row.get("last_batch_metrics") or {}
        valid = _safe_div(float(last.get("valid", 0)), float(last.get("total", 0)))
        table_rows.append(
            [
                str(row.get("run_name", "")),
                str(row.get("gate", "-")),
                str(row.get("status", "-")),
                _format_pct(row.get("progress")),
                _format_float(row.get("step_progress_float"), digits=2),
                _format_float(row.get("rate_rollouts_per_hour"), digits=1),
                _format_eta(row.get("eta_hours")),
                _format_eta(row.get("eta_hours_conservative")),
                _format_pct(valid),
                _format_pct(last.get("frac_non_norm")),
                _format_pct(batch.get("judge_hacking_rate")),
                _format_float(row.get("log_age_minutes"), digits=1),
            ]
        )

    widths = [len(h) for h in headers]
    for table_row in table_rows:
        for i, cell in enumerate(table_row):
            widths[i] = max(widths[i], len(cell))

    def fmt(values: list[str]) -> str:
        return "  ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines.extend(fmt(values) for values in table_rows)
    return "\n".join(lines)


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dirs = _list_run_dirs(args)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for run_dir in run_dirs:
        try:
            rows.append(
                _analyze_run(
                    run_dir,
                    window_minutes=float(args.window_minutes),
                    stale_minutes=float(args.stale_minutes),
                    tail_lines=int(args.tail_lines),
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"run_dir": str(run_dir), "error": str(exc)})

    if args.only_active:
        rows = [row for row in rows if row.get("status") in {"active", "waiting"}]

    rows.sort(
        key=lambda row: (
            {"active": 0, "waiting": 1, "stale": 2, "completed": 3}.get(str(row.get("status")), 9),
            row.get("eta_hours_conservative")
            if row.get("eta_hours_conservative") is not None
            else math.inf,
            str(row.get("run_name", "")),
        )
    )

    return {
        "generated_at": dt.datetime.now().isoformat(),
        "n_runs": len(rows),
        "n_errors": len(errors),
        "runs": rows,
        "errors": errors,
    }


def main() -> None:
    args = _parse_args()

    while True:
        report = _build_report(args)

        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(
                f"Generated at: {report['generated_at']} | runs: {report['n_runs']} "
                f"| errors: {report['n_errors']}"
            )
            if report["runs"]:
                print(_format_table(report["runs"]))
            else:
                print("No runs matched selection.")
            if report["errors"]:
                print("\nErrors:")
                for err in report["errors"]:
                    print(f"- {err['run_dir']}: {err['error']}")

        if args.watch_seconds <= 0:
            break

        time.sleep(float(args.watch_seconds))
        if not args.json:
            print("\033[2J\033[H", end="")


if __name__ == "__main__":
    main()
