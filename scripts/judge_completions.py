#!/usr/bin/env python3
"""Post-hoc LM judge for reward hacking detection.

Read CompletionRecord JSONL from training, judge each completion via LLM,
output verdicts and summary statistics.

Usage:
    python scripts/judge_completions.py \
        --completions artifacts/.../completions.jsonl \
        --output artifacts/.../judge_results/

    python scripts/judge_completions.py \
        --completions artifacts/.../completions.jsonl \
        --judge-config configs/judge/gpt-5.2.yaml \
        --sample 50

    python scripts/judge_completions.py \
        --completions artifacts/.../completions.jsonl \
        --env-file /path/to/.env \
        temperature=0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ppl_synthesis_reward_hacking.evaluation.record_utils import deduplicate_records
from ppl_synthesis_reward_hacking.experiments.results import json_default
from ppl_synthesis_reward_hacking.logging.completions import load_completions
from ppl_synthesis_reward_hacking.monitoring.llm_judge import (
    DEFAULT_JUDGE_CONFIG,
    JudgeConfig,
    JudgeVerdict,
    judge_completions,
    load_judge_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Post-hoc LM judge for reward hacking")
    p.add_argument("--completions", required=True, help="Path to completions.jsonl")
    p.add_argument("--output", default=None, help="Output directory for results")
    p.add_argument(
        "--judge-config",
        default=None,
        help=f"Judge config YAML (default: {DEFAULT_JUDGE_CONFIG})",
    )
    p.add_argument("--sample", type=int, default=None, help="Judge random subset (cost control)")
    p.add_argument("--only-valid", action="store_true", help="Only judge valid completions")
    p.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file for API keys",
    )
    p.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate completions by normalized code hash before judging (default: true)",
    )
    return p.parse_known_args()


def _compute_summary(
    records,
    verdicts: list[JudgeVerdict],
    cfg: JudgeConfig,
    n_errors: int,
) -> dict:
    verdict_counts = Counter(v.verdict for v in verdicts)
    pattern_counts = _collect_pattern_counts(verdicts)
    log_mass_means = _compute_log_mass_means(records, verdicts)
    phase_fracs = _compute_phase_hacking_fractions(records, verdicts)

    return {
        "total_judged": len(verdicts),
        "total_errors": n_errors,
        "judge_model": cfg.model,
        "verdicts": dict(verdict_counts),
        "hacking_fraction_by_phase": phase_fracs,
        "pattern_frequencies": dict(pattern_counts.most_common()),
        "log_mass_vs_verdict": log_mass_means,
    }


def _collect_pattern_counts(verdicts: list[JudgeVerdict]) -> Counter:
    pattern_counts: Counter = Counter()
    for verdict in verdicts:
        for tag in verdict.tags:
            pattern_counts[tag] += 1
    return pattern_counts


def _compute_log_mass_means(records, verdicts: list[JudgeVerdict]) -> dict[str, float]:
    log_mass_by_verdict: dict[str, list[float]] = {
        "honest": [],
        "suspicious": [],
        "evaluator_hack": [],
        "hacking": [],
    }
    for record, verdict in zip(records, verdicts, strict=True):
        if verdict.verdict not in log_mass_by_verdict:
            continue
        metadata = record.metadata or {}
        normalization = metadata.get("normalization", {}) if isinstance(metadata, dict) else {}
        if isinstance(normalization, dict):
            raw_log_mass = normalization.get("log_mass", float("nan"))
            if isinstance(raw_log_mass, int | float) and math.isfinite(float(raw_log_mass)):
                log_mass_by_verdict[verdict.verdict].append(float(raw_log_mass))

    means: dict[str, float] = {}
    for verdict, values in log_mass_by_verdict.items():
        if values:
            means[f"{verdict}_mean_log_mass"] = round(sum(values) / len(values), 4)
    return means


def _compute_phase_hacking_fractions(records, verdicts: list[JudgeVerdict]) -> dict[str, float]:
    if not records:
        return {}
    batches = sorted({record.batch for record in records})
    n_batches = len(batches)
    third = max(1, n_batches // 3)
    phases = {
        "early": set(batches[:third]),
        "mid": set(batches[third : 2 * third]),
        "late": set(batches[2 * third :]),
    }
    return {
        phase: _hacking_fraction_for_batches(records, verdicts, batch_set=batch_set)
        for phase, batch_set in phases.items()
    }


def _hacking_fraction_for_batches(
    records,
    verdicts: list[JudgeVerdict],
    *,
    batch_set: set[int],
) -> float:
    matched = [
        (record, verdict)
        for record, verdict in zip(records, verdicts, strict=True)
        if record.batch in batch_set
    ]
    if not matched:
        return 0.0
    return round(sum(1 for _, verdict in matched if verdict.verdict == "hacking") / len(matched), 3)


def main() -> None:
    args, overrides = parse_args()

    completions_path = Path(args.completions)
    if not completions_path.exists():
        log.error("Completions file not found: %s", completions_path)
        sys.exit(1)

    log.info("Loading completions from %s", completions_path)
    records = load_completions(completions_path)
    log.info("Loaded %d completion records", len(records))

    if args.only_valid:
        records = [r for r in records if r.outcome == "valid"]
        log.info("Filtered to %d valid completions", len(records))

    if args.dedup:
        pre = len(records)
        records = deduplicate_records(records)
        log.info("Deduplicated completions: %d -> %d", pre, len(records))

    if not records:
        log.warning("No records to judge")
        sys.exit(0)

    # sample before judging so we track exactly which records are judged
    if args.sample is not None and args.sample < len(records):
        records = random.sample(records, args.sample)
        log.info("Sampled %d records for judging", len(records))

    cfg = load_judge_config(args.judge_config, overrides)
    log.info("Judge model: %s (stub=%s)", cfg.model, cfg.use_stub)

    verdicts = judge_completions(records, cfg, env_file=args.env_file)

    n_errors = sum(1 for v in verdicts if v.parse_error)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = completions_path.parent / "judge_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    verdicts_path = output_dir / "verdicts.jsonl"
    with verdicts_path.open("w", encoding="utf-8") as f:
        for rec, v in zip(records, verdicts, strict=True):
            meta = rec.metadata or {}
            entry = {
                "batch": rec.batch,
                "index": rec.index,
                "outcome": rec.outcome,
                "reported_reward": json_default(rec.reported_reward),
                "log_mass": json_default(
                    (meta.get("normalization") or {}).get("log_mass")
                    if isinstance(meta.get("normalization"), dict)
                    else float("nan")
                ),
                **asdict(v),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Wrote %d verdicts to %s", len(verdicts), verdicts_path)

    summary = _compute_summary(records, verdicts, cfg, n_errors)
    summary_path = output_dir / "judge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
