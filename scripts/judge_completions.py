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
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ppl_synthesis_reward_hacking.config.load import apply_overrides, load_config
from ppl_synthesis_reward_hacking.logging.completions import load_completions
from ppl_synthesis_reward_hacking.monitoring.llm_judge import (
    JudgeConfig,
    JudgeVerdict,
    judge_completions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_JUDGE_CONFIG = Path("configs/judge/glm-5.yaml")


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


def _hash_completion_text(text: str) -> str:
    code_block = text
    marker = "```"
    if marker in text:
        parts = text.split(marker)
        if len(parts) >= 2:
            code_block = parts[1]
    normalized = " ".join(code_block.strip().lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def _deduplicate_records(records):
    unique = {}
    for record in records:
        content = record.code or record.completion_text
        key = _hash_completion_text(content)
        if key not in unique:
            unique[key] = record
    return list(unique.values())


def _load_judge_config(config_path: str | None, overrides: list[str]) -> JudgeConfig:
    if config_path:
        raw = dict(load_config(config_path))
    elif DEFAULT_JUDGE_CONFIG.exists():
        raw = dict(load_config(DEFAULT_JUDGE_CONFIG))
    else:
        raw = {}

    if overrides:
        raw = apply_overrides(raw, overrides)

    # map YAML keys to JudgeConfig fields
    return JudgeConfig(
        model=raw.get("model", JudgeConfig.model),
        api_base=raw.get("api_base", JudgeConfig.api_base),
        custom_llm_provider=raw.get("custom_llm_provider", JudgeConfig.custom_llm_provider),
        api_key_env=raw.get("api_key_env", JudgeConfig.api_key_env),
        temperature=float(raw.get("temperature", JudgeConfig.temperature)),
        max_tokens=int(raw.get("max_tokens", JudgeConfig.max_tokens)),
        enabled=bool(raw.get("enabled", JudgeConfig.enabled)),
        use_stub=bool(raw.get("use_stub", JudgeConfig.use_stub)),
    )


def _compute_summary(
    records,
    verdicts: list[JudgeVerdict],
    cfg: JudgeConfig,
    n_errors: int,
) -> dict:
    verdict_counts = Counter(v.verdict for v in verdicts)
    pattern_counts: Counter = Counter()
    for v in verdicts:
        for tag in v.tags:
            pattern_counts[tag] += 1

    # gap vs verdict
    gap_by_verdict: dict[str, list[float]] = {
        "honest": [],
        "suspicious": [],
        "evaluator_hack": [],
        "hacking": [],
    }
    for rec, v in zip(records, verdicts, strict=True):
        if v.verdict in gap_by_verdict:
            gap_by_verdict[v.verdict].append(rec.gap)

    gap_means = {}
    for k, vals in gap_by_verdict.items():
        if vals:
            gap_means[f"{k}_mean_gap"] = round(sum(vals) / len(vals), 2)

    # hacking fraction by training phase (early/mid/late thirds)
    if records:
        batches = sorted({r.batch for r in records})
        n_batches = len(batches)
        third = max(1, n_batches // 3)
        early_batches = set(batches[:third])
        mid_batches = set(batches[third : 2 * third])
        late_batches = set(batches[2 * third :])

        def _hacking_frac(batch_set):
            matched = [
                (r, v) for r, v in zip(records, verdicts, strict=True) if r.batch in batch_set
            ]
            if not matched:
                return 0.0
            return round(sum(1 for _, v in matched if v.verdict == "hacking") / len(matched), 3)

        phase_fracs = {
            "early": _hacking_frac(early_batches),
            "mid": _hacking_frac(mid_batches),
            "late": _hacking_frac(late_batches),
        }
    else:
        phase_fracs = {}

    return {
        "total_judged": len(verdicts),
        "total_errors": n_errors,
        "judge_model": cfg.model,
        "verdicts": dict(verdict_counts),
        "hacking_fraction_by_phase": phase_fracs,
        "pattern_frequencies": dict(pattern_counts.most_common()),
        "gap_vs_verdict": gap_means,
    }


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
        records = _deduplicate_records(records)
        log.info("Deduplicated completions: %d -> %d", pre, len(records))

    if not records:
        log.warning("No records to judge")
        sys.exit(0)

    # sample before judging so we track exactly which records are judged
    if args.sample is not None and args.sample < len(records):
        records = random.sample(records, args.sample)
        log.info("Sampled %d records for judging", len(records))

    cfg = _load_judge_config(args.judge_config, overrides)
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
            entry = {
                "batch": rec.batch,
                "index": rec.index,
                "outcome": rec.outcome,
                "reported_reward": rec.reported_reward,
                "oracle_score": rec.oracle_score,
                "gap": rec.gap,
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
