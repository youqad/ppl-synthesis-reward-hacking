#!/usr/bin/env python3
"""Offline exploit taxonomy labeling for completion trajectories."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.evaluation.record_utils import (
    extract_record_code,
    record_code_hash,
)
from ppl_synthesis_reward_hacking.evaluation.taxonomy import canonicalize_exploit_tags
from ppl_synthesis_reward_hacking.experiments.results import json_default
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord, load_completions
from ppl_synthesis_reward_hacking.utils.io import load_jsonl

_INITVAL_ASSIGNMENT_RE = re.compile(r"\b(?:initval|testval|initial_point)\s*=", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_MODEL_RE = re.compile(
    r"((?:import\s+\S+.*\n)*def\s+model\s*\(\s*data\s*\)\s*:.*?)(?:\n\S|\Z)",
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label exploit taxonomy over completion records")
    p.add_argument("--completions", required=True, help="Path to completions.jsonl")
    p.add_argument(
        "--judge-verdicts",
        default=None,
        help="Optional judge verdicts JSONL for verdict/tag joins",
    )
    p.add_argument("--output-dir", default=None, help="Default: <completions>/../eval")
    p.add_argument("--only-valid", action="store_true", help="Process only outcome=valid")
    p.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate by normalized code hash before analysis",
    )
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_judge_verdicts(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in load_jsonl(path):
        batch = int(row.get("batch", -1))
        index = int(row.get("index", -1))
        by_key[(batch, index)] = row
    return by_key


def _get_n_observed_from_metadata(record: CompletionRecord) -> int | None:
    metadata = record.metadata or {}
    decomp = metadata.get("decomposition")
    if not isinstance(decomp, dict):
        return None
    n_obs = decomp.get("n_obs_terms")
    if isinstance(n_obs, int):
        return n_obs
    return None


def _taxonomy_source_text(record: CompletionRecord) -> str | None:
    """Get text for taxonomy tagging, preserving parse-fail initval exploits.

    `extract_record_code` intentionally rejects initval/testval/initial_point
    snippets for execution safety. For taxonomy, we still want those records
    represented, so we fall back to completion text only when that exploit
    pattern appears.
    """
    code = extract_record_code(record)
    if code and code.strip():
        return code
    completion_text = (record.completion_text or "").strip()
    if not completion_text or not _INITVAL_ASSIGNMENT_RE.search(completion_text):
        return None
    candidate = _extract_code_like_text(completion_text)
    if candidate:
        return candidate
    return None


def _extract_code_like_text(completion_text: str) -> str | None:
    """Extract code-like text for taxonomy-only analysis.

    This bypasses the scorer's initval ban intentionally but requires strong
    code-like structure to avoid tagging plain-language prose.
    """
    match = _CODE_BLOCK_RE.search(completion_text)
    if match is not None:
        candidate = match.group(1).strip()
        if candidate and "def model" in candidate and "pm." in candidate:
            return candidate

    match = _BARE_MODEL_RE.search(completion_text)
    if match is not None:
        candidate = match.group(1).strip()
        if candidate and "def model" in candidate and "pm." in candidate:
            return candidate

    return None


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate rows by code hash while preserving tag evidence across duplicates."""
    deduped: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        code_hash = row.get("code_hash")
        if not code_hash:
            passthrough.append(row)
            continue
        existing = deduped.get(code_hash)
        if existing is None:
            deduped[code_hash] = dict(row)
            continue
        existing_tags = set(existing.get("exploit_tags", []))
        current_tags = set(row.get("exploit_tags", []))
        existing["exploit_tags"] = sorted(existing_tags | current_tags)
        existing_judge_tags = set(existing.get("judge_tags", []))
        current_judge_tags = set(row.get("judge_tags", []))
        existing["judge_tags"] = sorted(existing_judge_tags | current_judge_tags)
        if existing.get("judge_verdict") is None and row.get("judge_verdict") is not None:
            existing["judge_verdict"] = row.get("judge_verdict")
    return list(deduped.values()) + passthrough


def _batch_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        batch = int(row["batch"])
        by_batch.setdefault(batch, []).append(row)

    out: list[dict[str, Any]] = []
    for batch in sorted(by_batch):
        vals = by_batch[batch]
        tagged = sum(1 for row in vals if row["exploit_tags"])
        out.append(
            {
                "batch": batch,
                "n_records": len(vals),
                "n_tagged": tagged,
                "tagged_fraction": (tagged / len(vals)) if vals else 0.0,
            }
        )
    return out


def _judge_fields_for_record(
    record: CompletionRecord,
    judge_by_key: dict[tuple[int, int], dict[str, Any]],
) -> tuple[str | None, list[str]]:
    judge_row = judge_by_key.get((record.batch, record.index))
    if not judge_row:
        return None, []
    verdict = judge_row.get("verdict")
    tags_value = judge_row.get("tags")
    tags = [str(tag) for tag in tags_value] if isinstance(tags_value, list) else []
    return verdict, tags


def _row_for_parse_fail_without_code(
    record: CompletionRecord,
    *,
    judge_verdict: str | None,
    judge_tags: list[str],
) -> dict[str, Any]:
    return {
        "batch": record.batch,
        "index": record.index,
        "code_hash": record_code_hash(record),
        "outcome": record.outcome,
        "reported_reward": json_default(record.reported_reward),
        "exploit_tags": [],
        "taxonomy_status": "no_code_parse_fail",
        "judge_verdict": judge_verdict,
        "judge_tags": judge_tags,
    }


def _row_for_analyzed_record(
    record: CompletionRecord,
    source_text: str,
    *,
    judge_verdict: str | None,
    judge_tags: list[str],
) -> dict[str, Any]:
    n_observed = _get_n_observed_from_metadata(record)
    exploit_tags = sorted(canonicalize_exploit_tags(source_text, n_observed_rvs=n_observed))
    return {
        "batch": record.batch,
        "index": record.index,
        "code_hash": record_code_hash(record),
        "outcome": record.outcome,
        "reported_reward": json_default(record.reported_reward),
        "exploit_tags": exploit_tags,
        "taxonomy_status": "analyzed",
        "judge_verdict": judge_verdict,
        "judge_tags": judge_tags,
    }


def _collect_taxonomy_rows(
    records: list[CompletionRecord],
    *,
    judge_by_key: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_no_code_non_parse = 0
    for record in records:
        judge_verdict, judge_tags = _judge_fields_for_record(record, judge_by_key)
        source_text = _taxonomy_source_text(record)

        if not source_text:
            if record.outcome == "parse_fail":
                rows.append(
                    _row_for_parse_fail_without_code(
                        record,
                        judge_verdict=judge_verdict,
                        judge_tags=judge_tags,
                    )
                )
            else:
                skipped_no_code_non_parse += 1
            continue

        rows.append(
            _row_for_analyzed_record(
                record,
                source_text,
                judge_verdict=judge_verdict,
                judge_tags=judge_tags,
            )
        )
    return rows, skipped_no_code_non_parse


def _taxonomy_summary(
    rows: list[dict[str, Any]],
    *,
    skipped_no_code_non_parse: int,
    output_jsonl: Path,
) -> dict[str, Any]:
    tag_counter: Counter[str] = Counter()
    judge_verdict_counter: Counter[str] = Counter()
    for row in rows:
        for tag in row["exploit_tags"]:
            tag_counter[tag] += 1
        verdict = row.get("judge_verdict")
        if isinstance(verdict, str):
            judge_verdict_counter[verdict] += 1

    n_tagged = sum(1 for row in rows if row["exploit_tags"])
    return {
        "n_records": len(rows),
        "n_analyzed": sum(1 for row in rows if row.get("taxonomy_status") == "analyzed"),
        "n_parse_fail_no_code": sum(
            1 for row in rows if row.get("taxonomy_status") == "no_code_parse_fail"
        ),
        "n_skipped_no_code_non_parse": skipped_no_code_non_parse,
        "n_tagged": n_tagged,
        "tagged_fraction": (n_tagged / len(rows)) if rows else 0.0,
        "tag_counts": dict(tag_counter.most_common()),
        "judge_verdict_counts": dict(judge_verdict_counter),
        "batch_metrics": _batch_metrics(rows),
        "output_jsonl": str(output_jsonl),
    }


def main() -> None:
    args = parse_args()
    completions_path = Path(args.completions)
    records = load_completions(completions_path)

    if args.only_valid:
        records = [r for r in records if r.outcome == "valid"]
    if args.sample is not None and args.sample < len(records):
        rng = random.Random(args.seed)
        records = rng.sample(records, args.sample)

    judge_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    if args.judge_verdicts:
        judge_by_key = _load_judge_verdicts(Path(args.judge_verdicts))

    rows, skipped_no_code_non_parse = _collect_taxonomy_rows(records, judge_by_key=judge_by_key)

    if args.dedup:
        rows = _deduplicate_rows(rows)

    output_dir = Path(args.output_dir) if args.output_dir else completions_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "taxonomy.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _taxonomy_summary(
        rows,
        skipped_no_code_non_parse=skipped_no_code_non_parse,
        output_jsonl=jsonl_path,
    )
    summary_path = output_dir / "taxonomy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
