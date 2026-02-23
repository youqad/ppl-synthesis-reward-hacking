#!/usr/bin/env python3
"""Offline normalization evaluation for completion trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.evaluation.normalization_check import (
    check_normalization_d1,
    check_normalization_small_d,
)
from ppl_synthesis_reward_hacking.evaluation.record_utils import (
    deduplicate_records,
    extract_record_code,
    record_code_hash,
)
from ppl_synthesis_reward_hacking.experiments.results import json_default
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord, load_completions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate normalization mass for generated programs")
    p.add_argument("--completions", required=True, help="Path to completions.jsonl")
    p.add_argument("--output-dir", default=None, help="Default: <completions>/../eval")
    p.add_argument("--only-valid", action="store_true", help="Evaluate only outcome=valid records")
    p.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate by normalized extracted code hash before evaluation",
    )
    p.add_argument("--sample", type=int, default=None, help="Optional random sample size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-true", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=1, help="Interface dimension for exact checks")
    p.add_argument(
        "--max-d-exact",
        type=int,
        default=10,
        help="Maximum d supported by exact enumeration mode",
    )
    p.add_argument(
        "--mode",
        choices=("auto", "exact"),
        default="auto",
        help="auto=exact when d<=max-d-exact; exact=always exact and errors if d too large",
    )
    return p.parse_args()


def _program_text(record: CompletionRecord) -> str | None:
    code = extract_record_code(record)
    if code:
        return f"```python\n{code.strip()}\n```"
    return None


def _record_hash(record: CompletionRecord) -> str | None:
    return record_code_hash(record)


def _batch_summary(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        batch = int(row["batch"])
        by_batch.setdefault(batch, []).append(row)

    out: list[dict[str, Any]] = []
    for batch in sorted(by_batch):
        vals = [r for r in by_batch[batch] if r["ok"] and r["log_mass"] is not None]
        logs = [float(r["log_mass"]) for r in vals]
        non_normalized = sum(1 for r in vals if abs(float(r["log_mass"])) > epsilon)
        mean_log_mass = (sum(logs) / len(logs)) if logs else None
        median_log_mass = None
        if logs:
            sorted_logs = sorted(logs)
            mid = len(sorted_logs) // 2
            if len(sorted_logs) % 2 == 1:
                median_log_mass = sorted_logs[mid]
            else:
                median_log_mass = 0.5 * (sorted_logs[mid - 1] + sorted_logs[mid])
        out.append(
            {
                "batch": batch,
                "n_records": len(by_batch[batch]),
                "n_ok": len(vals),
                "frac_non_normalized": (non_normalized / len(vals)) if vals else 0.0,
                "mean_log_mass": mean_log_mass,
                "median_log_mass": median_log_mass,
            }
        )
    return out


def _evaluate_normalization(
    *,
    program_text: str,
    args: argparse.Namespace,
    row_seed: int,
) -> dict[str, Any]:
    if args.mode == "exact" and args.d > args.max_d_exact:
        raise RuntimeError(f"d={args.d} exceeds --max-d-exact={args.max_d_exact} in exact mode")
    if args.mode == "auto" and args.d > args.max_d_exact:
        raise RuntimeError(
            f"d={args.d} exceeds --max-d-exact={args.max_d_exact}; exact enumeration disabled"
        )

    if args.d == 1:
        return check_normalization_d1(
            completion_text=program_text,
            p_true=args.p_true,
            seed=row_seed,
            timeout=args.timeout,
            epsilon=args.epsilon,
        )
    return check_normalization_small_d(
        completion_text=program_text,
        d=args.d,
        p_true=args.p_true,
        seed=row_seed,
        timeout=args.timeout,
        epsilon=args.epsilon,
    )


def main() -> None:
    args = parse_args()
    completions_path = Path(args.completions)
    records = load_completions(completions_path)

    if args.only_valid:
        records = [r for r in records if r.outcome == "valid"]
    if args.dedup:
        records = deduplicate_records(records, hash_fn=_record_hash)
    if args.sample is not None and args.sample < len(records):
        rng = random.Random(args.seed)
        records = rng.sample(records, args.sample)

    rows: list[dict[str, Any]] = []
    for record in records:
        program_text = _program_text(record)
        if program_text is None:
            continue
        code_hash = record_code_hash(record)

        row_seed = args.seed + record.batch * 1000 + record.index
        result = _evaluate_normalization(
            program_text=program_text,
            args=args,
            row_seed=row_seed,
        )
        row = {
            "batch": record.batch,
            "index": record.index,
            "code_hash": code_hash,
            "outcome": record.outcome,
            "reported_reward": json_default(record.reported_reward),
            "d": int(result.get("d", args.d)),
            "ok": bool(result.get("ok", False)),
            "status": result.get("status"),
            "method": result.get("method"),
            "delta_scope": result.get("delta_scope"),
            "is_normalized": result.get("is_normalized"),
            "reason": result.get("reason"),
            "z_obs": json_default(result.get("z_obs")),
            "z0": json_default(result.get("z0")),
            "z1": json_default(result.get("z1")),
            "log_mass": json_default(result.get("log_mass")),
            "mass": json_default(result.get("mass")),
            "delta_from_one": json_default(result.get("delta_from_one")),
            "ess": json_default(result.get("ess")),
            "ci_log_mass_low": json_default(result.get("ci_log_mass_low")),
            "ci_log_mass_high": json_default(result.get("ci_log_mass_high")),
        }
        rows.append(row)

    valid_rows = [r for r in rows if r["ok"] and r["log_mass"] is not None]
    valid_logs = [float(r["log_mass"]) for r in valid_rows]
    n_non_normalized = sum(1 for r in valid_rows if abs(float(r["log_mass"])) > args.epsilon)
    output_dir = Path(args.output_dir) if args.output_dir else completions_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "normalization.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    median_log_mass = None
    if valid_logs:
        sorted_logs = sorted(valid_logs)
        mid = len(sorted_logs) // 2
        if len(sorted_logs) % 2 == 1:
            median_log_mass = sorted_logs[mid]
        else:
            median_log_mass = 0.5 * (sorted_logs[mid - 1] + sorted_logs[mid])

    summary = {
        "d": args.d,
        "mode": args.mode,
        "max_d_exact": args.max_d_exact,
        "n_records": len(rows),
        "n_ok": len(valid_rows),
        "n_failed": len(rows) - len(valid_rows),
        "epsilon": args.epsilon,
        "n_non_normalized": n_non_normalized,
        "frac_non_normalized": (n_non_normalized / len(valid_rows)) if valid_rows else 0.0,
        "mean_log_mass": ((sum(valid_logs) / len(valid_logs)) if valid_logs else None),
        "median_log_mass": median_log_mass,
        "batch_metrics": _batch_summary(rows, args.epsilon),
        "output_jsonl": str(jsonl_path),
    }

    summary_path = output_dir / "normalization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
