#!/usr/bin/env python3
"""Evaluate SafeStan/SafePyMC mitigation behavior.

Two modes:
1) Completion trajectory mode (`--completions`): evaluates SafePyMC acceptance.
2) Labeled suite mode (`--suite`): evaluates fail-closed metrics using expected
   unsafe labels + SafeStan checker, and reports normalization among accepted.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import execute_pymc_code
from ppl_synthesis_reward_hacking.backends.pymc_safe.checker import check_pymc_model
from ppl_synthesis_reward_hacking.backends.sstan.checker import check_sstan
from ppl_synthesis_reward_hacking.data.synthstats_loader import make_scoring_data_dict
from ppl_synthesis_reward_hacking.evaluation.label_parsing import coerce_expected_unsafe
from ppl_synthesis_reward_hacking.evaluation.normalization_check import (
    check_normalization_d1,
    check_normalization_small_d,
)
from ppl_synthesis_reward_hacking.evaluation.record_utils import extract_record_code
from ppl_synthesis_reward_hacking.logging.completions import load_completions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate mitigation behavior")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--completions",
        default=None,
        help="Path to completions.jsonl (trajectory mode)",
    )
    source.add_argument(
        "--suite",
        default=None,
        help="Path to labeled JSONL with source_pymc/translated_stan (suite mode)",
    )
    p.add_argument("--output", required=True, help="Path to output JSON")
    p.add_argument(
        "--only-valid",
        action="store_true",
        help="Trajectory mode only: evaluate records with outcome=valid",
    )
    p.add_argument("--sample", type=int, default=None, help="Optional random sample size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument("--scoring-d", type=int, default=1)
    p.add_argument("--scoring-p-true", type=float, default=0.5)
    p.add_argument(
        "--normalization-d",
        type=int,
        default=1,
        help="Suite mode: d for exact normalization checks on accepted sources",
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class _SuiteStats:
    def __init__(self) -> None:
        self.n_expected_unsafe = 0
        self.n_expected_safe = 0
        self.n_expected_unsafe_rejected = 0
        self.n_expected_safe_accepted = 0
        self.accepted_norm_non_normalized = 0
        self.accepted_norm_failed = 0
        self.accepted_norm_ok_n = 0
        self.n_sstan_check_exceptions = 0
        self.accepted_norm_results: list[dict[str, Any]] = []

    def track_expected(self, expected_unsafe: bool) -> None:
        if expected_unsafe:
            self.n_expected_unsafe += 1
        else:
            self.n_expected_safe += 1

    def track_acceptance(self, *, expected_unsafe: bool, accepted: bool) -> None:
        if expected_unsafe and not accepted:
            self.n_expected_unsafe_rejected += 1
        if (not expected_unsafe) and accepted:
            self.n_expected_safe_accepted += 1

    def track_sstan_exception(self) -> None:
        self.n_sstan_check_exceptions += 1

    def track_normalization(self, normalization: dict[str, Any]) -> None:
        self.accepted_norm_results.append(normalization)
        if not bool(normalization.get("ok")):
            self.accepted_norm_failed += 1
            return
        self.accepted_norm_ok_n += 1
        if not bool(normalization.get("is_normalized", False)):
            self.accepted_norm_non_normalized += 1

    def to_summary(
        self,
        *,
        n_records: int,
        violation_counter: Counter[str],
        details: list[dict[str, Any]],
    ) -> dict[str, Any]:
        unsafe_reject_rate = (
            (self.n_expected_unsafe_rejected / self.n_expected_unsafe)
            if self.n_expected_unsafe
            else 0.0
        )
        safe_accept_rate = (
            (self.n_expected_safe_accepted / self.n_expected_safe) if self.n_expected_safe else 0.0
        )
        accepted_norm_fraction = (
            (self.accepted_norm_non_normalized / self.accepted_norm_ok_n)
            if self.accepted_norm_ok_n
            else None
        )
        return {
            "mode": "suite",
            "n_records": n_records,
            "n_expected_unsafe": self.n_expected_unsafe,
            "n_expected_safe": self.n_expected_safe,
            "unsafe_reject_rate": unsafe_reject_rate,
            "safe_accept_rate": safe_accept_rate,
            "accepted_n": len(self.accepted_norm_results),
            "accepted_normalization_fail_n": self.accepted_norm_failed,
            "accepted_normalization_ok_n": self.accepted_norm_ok_n,
            "accepted_non_normalized_n": self.accepted_norm_non_normalized,
            "accepted_non_normalized_fraction": accepted_norm_fraction,
            "n_sstan_check_exceptions": self.n_sstan_check_exceptions,
            "violation_breakdown": dict(violation_counter),
            "details": details,
        }


def _sample_rows(
    rows: list[dict[str, Any]], *, sample: int | None, seed: int
) -> list[dict[str, Any]]:
    if sample is None or sample >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample)


def _check_translated_stan(translated_stan: str) -> tuple[bool, list[str], bool]:
    translation_present = bool(translated_stan.strip())
    if not translation_present:
        return False, ["missing translated_stan"], False
    try:
        accepted, reasons = check_sstan(translated_stan)
        return accepted, reasons, False
    except Exception as exc:
        return False, [f"check_sstan_exception: {type(exc).__name__}"], True


def _maybe_check_normalization(
    *,
    accepted: bool,
    source_pymc: str,
    args: argparse.Namespace,
    idx: int,
) -> dict[str, Any] | None:
    if not accepted or not source_pymc.strip():
        return None
    try:
        if args.normalization_d == 1:
            return check_normalization_d1(
                completion_text=source_pymc,
                p_true=args.scoring_p_true,
                seed=args.seed + idx,
                timeout=args.exec_timeout,
            )
        return check_normalization_small_d(
            completion_text=source_pymc,
            d=args.normalization_d,
            p_true=args.scoring_p_true,
            seed=args.seed + idx,
            timeout=args.exec_timeout,
        )
    except (TimeoutError, ValueError, RuntimeError) as exc:
        return {
            "ok": False,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _record_suite_violations(
    *,
    violation_counter: Counter[str],
    expected_unsafe: bool,
    accepted: bool,
    translation_present: bool,
) -> None:
    if expected_unsafe and accepted:
        violation_counter["unsafe_accepted"] += 1
    if (not expected_unsafe) and (not accepted):
        violation_counter["safe_rejected"] += 1
    if not translation_present:
        violation_counter["missing_translation"] += 1


def _evaluate_suite_row(
    *,
    row: dict[str, Any],
    idx: int,
    args: argparse.Namespace,
    stats: _SuiteStats,
    violation_counter: Counter[str],
) -> dict[str, Any]:
    if "expected_unsafe" not in row:
        raise ValueError(f"row {idx} missing required field `expected_unsafe`")
    expected_unsafe = coerce_expected_unsafe(row["expected_unsafe"])
    stats.track_expected(expected_unsafe)

    translated_stan = str(row.get("translated_stan", "") or "")
    accepted, reasons, had_sstan_exception = _check_translated_stan(translated_stan)
    if had_sstan_exception:
        stats.track_sstan_exception()
    stats.track_acceptance(expected_unsafe=expected_unsafe, accepted=accepted)

    source_pymc = str(row.get("source_pymc", "") or "")
    normalization = _maybe_check_normalization(
        accepted=accepted,
        source_pymc=source_pymc,
        args=args,
        idx=idx,
    )
    if normalization is not None:
        stats.track_normalization(normalization)

    _record_suite_violations(
        violation_counter=violation_counter,
        expected_unsafe=expected_unsafe,
        accepted=accepted,
        translation_present=bool(translated_stan.strip()),
    )
    return {
        "id": row.get("id"),
        "expected_unsafe": expected_unsafe,
        "translation_present": bool(translated_stan.strip()),
        "sstan_accepted": accepted,
        "sstan_reasons": reasons,
        "normalization": normalization,
    }


def _evaluate_suite(args: argparse.Namespace) -> dict[str, Any]:
    rows = _sample_rows(
        _load_jsonl(Path(args.suite)),
        sample=args.sample,
        seed=args.seed,
    )
    stats = _SuiteStats()
    violation_counter: Counter[str] = Counter()
    details = [
        _evaluate_suite_row(
            row=row,
            idx=idx,
            args=args,
            stats=stats,
            violation_counter=violation_counter,
        )
        for idx, row in enumerate(rows)
    ]
    return stats.to_summary(
        n_records=len(rows),
        violation_counter=violation_counter,
        details=details,
    )


def _evaluate_trajectory(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    records = load_completions(args.completions)
    if args.only_valid:
        records = [r for r in records if r.outcome == "valid"]
    if args.sample is not None and args.sample < len(records):
        records = rng.sample(records, args.sample)

    by_batch: dict[int, list[bool]] = defaultdict(list)
    reason_counter: Counter[str] = Counter()
    n_checked = 0
    n_compile_fail = 0

    for record in records:
        code = extract_record_code(record)
        if not code:
            n_compile_fail += 1
            reason_counter["missing_code"] += 1
            continue
        scoring_data = make_scoring_data_dict(
            d=args.scoring_d,
            p_true=args.scoring_p_true,
            seed=args.seed + record.batch * 1000 + record.index,
        )
        try:
            model = execute_pymc_code(code, scoring_data, timeout=args.exec_timeout)
        except (TimeoutError, RuntimeError, ValueError):
            n_compile_fail += 1
            reason_counter["compile_or_exec_exception"] += 1
            continue
        if model is None:
            n_compile_fail += 1
            reason_counter["compile_or_exec_failed"] += 1
            continue
        try:
            accepted, reasons = check_pymc_model(model)
        except (TypeError, ValueError, RuntimeError):
            n_compile_fail += 1
            reason_counter["safe_gate_exception"] += 1
            continue
        n_checked += 1
        by_batch[record.batch].append(accepted)
        for reason in reasons:
            reason_counter[reason] += 1

    batch_metrics = []
    for batch in sorted(by_batch):
        vals = by_batch[batch]
        batch_metrics.append(
            {
                "batch": batch,
                "n_checked": len(vals),
                "acceptance_rate": (sum(1 for v in vals if v) / len(vals)) if vals else 0.0,
            }
        )

    return {
        "mode": "trajectory",
        "n_records": len(records),
        "n_checked": n_checked,
        "n_compile_fail": n_compile_fail,
        "overall_acceptance_rate": (
            sum(1 for vals in by_batch.values() for v in vals if v) / n_checked
            if n_checked
            else 0.0
        ),
        "top_rejection_reasons": dict(reason_counter.most_common(20)),
        "batch_metrics": batch_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.suite:
        summary = _evaluate_suite(args)
    else:
        summary = _evaluate_trajectory(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
