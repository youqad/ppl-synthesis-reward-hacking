#!/usr/bin/env python3
"""Offline SafePyMC gate evaluation for completion trajectories."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import execute_pymc_code
from ppl_synthesis_reward_hacking.backends.pymc_safe.checker import check_pymc_model
from ppl_synthesis_reward_hacking.data.pymc_synthesis_loader import make_scoring_data_dict
from ppl_synthesis_reward_hacking.evaluation.record_utils import (
    deduplicate_records,
    extract_record_code,
    record_code_hash,
)
from ppl_synthesis_reward_hacking.logging.completions import (
    COMPLETION_SCHEMA_VERSION,
    CompletionRecord,
    load_completions,
    load_completions_raw,
)

log = logging.getLogger(__name__)


def _load_records(path: Path) -> list[CompletionRecord]:
    """Load records, normalizing pre-v3 schemas."""
    try:
        return load_completions(path)
    except ValueError:
        log.debug("v3 parse failed, normalizing legacy fields")
    raw = load_completions_raw(path)
    records = []
    for d in raw:
        d.pop("oracle_score", None)
        d.pop("gap", None)
        d.pop("schema_version", None)
        d["schema_version"] = COMPLETION_SCHEMA_VERSION
        records.append(CompletionRecord(**d))
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SafePyMC gate over completion records")
    p.add_argument("--completions", required=True, help="Path to completions.jsonl")
    p.add_argument("--output-dir", default=None, help="Default: <completions>/../eval")
    p.add_argument("--only-valid", action="store_true", help="Evaluate only outcome=valid records")
    p.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate by normalized code hash before gate evaluation",
    )
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument("--scoring-d", type=int, default=1)
    p.add_argument("--scoring-p-true", type=float, default=0.5)
    p.add_argument(
        "--require-mutable-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require pm.Data/MutableData. Disable for numpy-interface programs.",
    )
    return p.parse_args()


def _batch_metrics(rows: list[dict]) -> list[dict]:
    by_batch: dict[int, list[dict]] = {}
    for row in rows:
        batch = int(row["batch"])
        by_batch.setdefault(batch, []).append(row)
    out: list[dict] = []
    for batch in sorted(by_batch):
        vals = by_batch[batch]
        checked = [row for row in vals if row["compile_ok"]]
        accepted = sum(1 for row in checked if row["safe_gate_accept"])
        out.append(
            {
                "batch": batch,
                "n_records": len(vals),
                "n_checked": len(checked),
                "acceptance_rate": (accepted / len(checked)) if checked else 0.0,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    completions_path = Path(args.completions)
    records = _load_records(completions_path)

    if args.only_valid:
        records = [r for r in records if r.outcome == "valid"]
    if args.dedup:
        records = deduplicate_records(records)
    if args.sample is not None and args.sample < len(records):
        rng = random.Random(args.seed)
        records = rng.sample(records, args.sample)

    rows: list[dict] = []
    reason_counter: Counter[str] = Counter()
    for record in records:
        code = extract_record_code(record)
        if not code:
            continue
        scoring_data = make_scoring_data_dict(
            d=args.scoring_d,
            p_true=args.scoring_p_true,
            seed=args.seed + record.batch * 1000 + record.index,
        )
        model = execute_pymc_code(code, scoring_data, timeout=args.exec_timeout)
        if model is None:
            rows.append(
                {
                    "batch": record.batch,
                    "index": record.index,
                    "code_hash": record_code_hash(record),
                    "compile_ok": False,
                    "safe_gate_accept": False,
                    "safe_gate_reasons": ["compile_or_exec_failed"],
                }
            )
            reason_counter["compile_or_exec_failed"] += 1
            continue

        accepted, reasons = check_pymc_model(
            model, require_observable_data=args.require_mutable_data
        )
        for reason in reasons:
            reason_counter[reason] += 1
        rows.append(
            {
                "batch": record.batch,
                "index": record.index,
                "code_hash": record_code_hash(record),
                "compile_ok": True,
                "safe_gate_accept": accepted,
                "safe_gate_reasons": reasons,
            }
        )

    checked = [row for row in rows if row["compile_ok"]]
    accepted_checked = sum(1 for row in checked if row["safe_gate_accept"])
    output_dir = Path(args.output_dir) if args.output_dir else completions_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "safety.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_records": len(rows),
        "n_checked": len(checked),
        "n_compile_fail": len(rows) - len(checked),
        "acceptance_rate_checked": (accepted_checked / len(checked)) if checked else 0.0,
        "top_rejection_reasons": dict(reason_counter.most_common(20)),
        "batch_metrics": _batch_metrics(rows),
        "output_jsonl": str(jsonl_path),
    }
    summary_path = output_dir / "safety_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
