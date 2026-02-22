#!/usr/bin/env python3
"""Audit PyMC->Stan transpilation fidelity for exploit preservation.

Fail-closed invariant suite:
- Unsafe source programs should remain unsafe after transpilation (rejected by SStan).
- Safe source programs should remain safe (accepted by SStan).
- Missing translation outputs are treated as violations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.backends.sstan.checker import check_sstan
from ppl_synthesis_reward_hacking.evaluation.label_parsing import coerce_expected_unsafe
from ppl_synthesis_reward_hacking.evaluation.taxonomy import canonicalize_exploit_tags
from ppl_synthesis_reward_hacking.utils.hashing import normalized_text_hash
from ppl_synthesis_reward_hacking.utils.io import load_jsonl as _load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate transpiler fidelity")
    p.add_argument("--input", required=True, help="JSONL with source_pymc and translated_stan")
    p.add_argument("--output", required=True, help="Output JSON summary path")
    p.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="Exit non-zero if any fail-closed invariant is violated",
    )
    return p.parse_args()


def _normalize_list(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(v) for v in values]
    return []


def _evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    source_pymc = str(row.get("source_pymc", "") or "")
    translated_stan = str(row.get("translated_stan", "") or "")
    translation_present = bool(translated_stan.strip())
    source_tags = sorted(canonicalize_exploit_tags(source_pymc))
    expected_tags = sorted(set(_normalize_list(row.get("expected_source_tags"))))

    if "expected_unsafe" not in row:
        raise ValueError("missing required field `expected_unsafe`")
    expected_unsafe = coerce_expected_unsafe(row.get("expected_unsafe"))

    if translation_present:
        sstan_accepted, reasons = check_sstan(translated_stan)
    else:
        sstan_accepted = False
        reasons = ["missing translated_stan"]

    source_tag_set = set(source_tags)
    expected_tag_set = set(expected_tags)
    source_signal = bool(source_tag_set or expected_tag_set)

    invariants = {
        "translation_present": translation_present,
        "expected_tags_observed": expected_tag_set.issubset(source_tag_set)
        if expected_tag_set
        else True,
        "unsafe_rejected": (not sstan_accepted) if expected_unsafe else True,
        "safe_accepted": sstan_accepted if not expected_unsafe else True,
        "unsafe_has_source_signal": source_signal if expected_unsafe else True,
    }
    fail_closed = all(invariants.values())

    if not translation_present:
        classification = "missing_translation"
    elif expected_unsafe:
        classification = "preserved" if not sstan_accepted else "sanitized"
    else:
        classification = "safe_pass" if sstan_accepted else "false_reject"

    code_hash = normalized_text_hash(source_pymc) if source_pymc else None
    return {
        "id": row.get("id"),
        "code_hash": code_hash,
        "expected_unsafe": expected_unsafe,
        "source_tags": source_tags,
        "expected_source_tags": expected_tags,
        "sstan_accepted": sstan_accepted,
        "reasons": reasons,
        "classification": classification,
        "invariants": invariants,
        "fail_closed": fail_closed,
    }


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        try:
            details.append(_evaluate_row(row))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid input row at index {idx}: {exc}") from exc
    n_total = len(details)

    n_expected_unsafe = sum(1 for d in details if d["expected_unsafe"])
    n_expected_safe = n_total - n_expected_unsafe
    n_expected_unsafe_translated = sum(
        1 for d in details if d["expected_unsafe"] and d["invariants"]["translation_present"]
    )
    n_expected_safe_translated = sum(
        1 for d in details if (not d["expected_unsafe"]) and d["invariants"]["translation_present"]
    )
    n_preserved = sum(1 for d in details if d["classification"] == "preserved")
    n_sanitized = sum(1 for d in details if d["classification"] == "sanitized")
    n_safe_pass = sum(1 for d in details if d["classification"] == "safe_pass")
    n_false_reject = sum(1 for d in details if d["classification"] == "false_reject")
    n_fail_closed = sum(1 for d in details if d["fail_closed"])
    n_missing_translation = sum(1 for d in details if not d["invariants"]["translation_present"])
    reason_counts = Counter(reason for d in details for reason in d["reasons"])
    violation_breakdown = Counter()
    for detail in details:
        for name, ok in detail["invariants"].items():
            if not ok:
                violation_breakdown[name] += 1

    return {
        "n_total": n_total,
        "n_expected_unsafe": n_expected_unsafe,
        "n_expected_safe": n_expected_safe,
        "n_expected_unsafe_translated": n_expected_unsafe_translated,
        "n_expected_safe_translated": n_expected_safe_translated,
        "unsafe_preservation_rate": (n_preserved / n_expected_unsafe_translated)
        if n_expected_unsafe_translated
        else 0.0,
        "unsafe_sanitization_rate": (n_sanitized / n_expected_unsafe_translated)
        if n_expected_unsafe_translated
        else 0.0,
        "safe_pass_rate": (n_safe_pass / n_expected_safe_translated)
        if n_expected_safe_translated
        else 0.0,
        "safe_false_reject_rate": (n_false_reject / n_expected_safe_translated)
        if n_expected_safe_translated
        else 0.0,
        "fail_closed_pass_rate": (n_fail_closed / n_total) if n_total else 0.0,
        "n_fail_closed_violations": n_total - n_fail_closed,
        "n_missing_translation": n_missing_translation,
        "violation_breakdown": dict(violation_breakdown),
        "rejection_reason_counts": dict(reason_counts.most_common(20)),
        "details": details,
    }


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(Path(args.input))
    summary = evaluate_rows(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.fail_on_violation and summary["n_fail_closed_violations"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
