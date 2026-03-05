#!/usr/bin/env python3
"""Per-step LLM judge profiling for likelihood hacking experiments.

Reads completions.jsonl from an artifact directory, obtains judge verdicts
(either by running the judge or loading existing verdicts), and produces
per-training-step aggregations: verdict distribution, tag frequencies,
confidence statistics. Outputs a JSON profile and a stacked area chart.

Usage:
    python scripts/batch_judge_profile.py \
        --artifact-dir artifacts/sweeps/tinker_20260217_163634 \
        --judge-config configs/judge/gpt-5.4.yaml \
        --output-dir .scratch/judge_profiles/seed0

    python scripts/batch_judge_profile.py \
        --artifact-dir artifacts/sweeps/tinker_20260217_163634 \
        --use-existing-verdicts .scratch/judge_profiles/seed0/verdicts.jsonl \
        --output-dir .scratch/judge_profiles/seed0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ppl_synthesis_reward_hacking.experiments.results import json_default
from ppl_synthesis_reward_hacking.logging.completions import load_completions
from ppl_synthesis_reward_hacking.monitoring.llm_judge import (
    DEFAULT_JUDGE_CONFIG,
    judge_completions,
    load_judge_config,
)
from ppl_synthesis_reward_hacking.plotting.styles import apply_publication_style
from ppl_synthesis_reward_hacking.utils.io import load_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VERDICT_TYPES = ("honest", "suspicious", "evaluator_hack", "hacking", "error")
VERDICT_COLORS = {
    "honest": "#1b9e77",
    "suspicious": "#e6ab02",
    "evaluator_hack": "#7570b3",
    "hacking": "#d95f02",
    "error": "#999999",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Per-step LLM judge profiling")
    p.add_argument(
        "--artifact-dir",
        required=True,
        help="Artifact directory with completions.jsonl",
    )
    p.add_argument(
        "--judge-config",
        default=None,
        help=f"Judge config YAML (default: {DEFAULT_JUDGE_CONFIG})",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for profile JSON and figure",
    )
    p.add_argument(
        "--use-existing-verdicts",
        default=None,
        help="Path to existing verdicts.jsonl (skip judge calls)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N records per step before judging",
    )
    p.add_argument("--only-valid", action="store_true", help="Only judge valid completions")
    p.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file for API keys",
    )
    return p.parse_known_args()


def _group_records_by_batch(records: list) -> dict[int, list]:
    groups: dict[int, list] = defaultdict(list)
    for rec in records:
        groups[rec.batch].append(rec)
    return dict(groups)


def _sample_per_step(
    grouped: dict[int, list],
    n: int,
) -> dict[int, list]:
    sampled: dict[int, list] = {}
    for batch, recs in grouped.items():
        if len(recs) <= n:
            sampled[batch] = list(recs)
        else:
            sampled[batch] = random.sample(recs, n)
    return sampled


def _aggregate_verdicts_by_step(
    verdict_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in verdict_rows:
        batch = row.get("batch")
        if batch is None:
            continue
        by_batch[int(batch)].append(row)

    steps = []
    for batch in sorted(by_batch):
        rows = by_batch[batch]
        n = len(rows)

        verdict_counts: Counter = Counter()
        tag_counts: Counter = Counter()
        confidences: list[float] = []

        for row in rows:
            v = row.get("verdict", "error")
            if v not in VERDICT_TYPES:
                v = "error"
            verdict_counts[v] += 1

            tags = row.get("tags", [])
            if isinstance(tags, list):
                for t in tags:
                    tag_counts[str(t)] += 1

            conf = row.get("confidence")
            if conf is not None:
                try:
                    c = float(conf)
                    if math.isfinite(c):
                        confidences.append(c)
                except (TypeError, ValueError):
                    pass

        hacking_n = verdict_counts.get("hacking", 0)
        hacking_rate = round(hacking_n / n, 4) if n > 0 else 0.0

        mean_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
        median_conf = None
        if confidences:
            s = sorted(confidences)
            mid = len(s) // 2
            median_conf = round(
                s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2,
                4,
            )

        steps.append(
            {
                "batch": batch,
                "n_records": n,
                "verdicts": dict(verdict_counts),
                "hacking_rate": hacking_rate,
                "tags": dict(tag_counts.most_common()),
                "mean_confidence": json_default(mean_conf),
                "median_confidence": json_default(median_conf),
            }
        )

    return steps


def _plot_verdict_distribution(
    steps: list[dict[str, Any]],
    output_dir: Path,
    judge_model: str,
) -> None:
    if not steps:
        log.warning("No steps to plot")
        return

    apply_publication_style(
        font_size=9,
        axes_labelsize=10,
        axes_titlesize=10,
        xtick_labelsize=8,
        ytick_labelsize=8,
        legend_fontsize=8,
        lines_linewidth=1.5,
    )

    batches = [s["batch"] for s in steps]
    proportions: dict[str, list[float]] = {v: [] for v in VERDICT_TYPES}

    for s in steps:
        n = s["n_records"]
        verdicts = s.get("verdicts", {})
        for v in VERDICT_TYPES:
            count = verdicts.get(v, 0)
            proportions[v].append(count / n if n > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    bottom = [0.0] * len(batches)
    for verdict_type in VERDICT_TYPES:
        vals = proportions[verdict_type]
        if all(v == 0.0 for v in vals):
            continue
        ax.fill_between(
            batches,
            bottom,
            [b + v for b, v in zip(bottom, vals, strict=False)],
            label=verdict_type,
            color=VERDICT_COLORS[verdict_type],
            alpha=0.8,
            linewidth=0,
        )
        bottom = [b + v for b, v in zip(bottom, vals, strict=False)]

    hacking_rates = [s["hacking_rate"] for s in steps]
    ax.plot(
        batches,
        hacking_rates,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="hacking rate",
        zorder=5,
    )

    ax.set_xlabel("Training step (batch index)")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(batches), max(batches))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title(f"Judge verdict distribution over training ({judge_model})")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = output_dir / f"judge_profile_figure.{ext}"
        fig.savefig(out_path)
        log.info("Saved figure: %s", out_path)

    plt.close(fig)


def main() -> None:
    args, overrides = parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_existing_verdicts:
        verdicts_path = Path(args.use_existing_verdicts)
        if not verdicts_path.exists():
            log.error("Verdicts file not found: %s", verdicts_path)
            sys.exit(1)

        log.info("Loading existing verdicts from %s", verdicts_path)
        verdict_rows = load_jsonl(verdicts_path)
        log.info("Loaded %d verdict rows", len(verdict_rows))

        cfg = load_judge_config(args.judge_config, overrides)
        judge_model = cfg.model
    else:
        completions_path = artifact_dir / "completions.jsonl"
        if not completions_path.exists():
            log.error("Completions file not found: %s", completions_path)
            sys.exit(1)

        log.info("Loading completions from %s", completions_path)
        records = load_completions(completions_path)
        log.info("Loaded %d completion records", len(records))

        if args.only_valid:
            records = [r for r in records if r.outcome == "valid"]
            log.info("Filtered to %d valid completions", len(records))

        if not records:
            log.warning("No records to judge")
            sys.exit(0)

        if args.sample is not None:
            grouped = _group_records_by_batch(records)
            sampled = _sample_per_step(grouped, args.sample)
            records = [rec for batch in sorted(sampled) for rec in sampled[batch]]
            log.info("Sampled to %d records (%d per step max)", len(records), args.sample)

        cfg = load_judge_config(args.judge_config, overrides)
        judge_model = cfg.model
        log.info("Judge model: %s (stub=%s)", cfg.model, cfg.use_stub)

        verdicts = judge_completions(records, cfg, env_file=args.env_file)

        verdict_rows = []
        for rec, v in zip(records, verdicts, strict=True):
            verdict_rows.append(
                {
                    "batch": rec.batch,
                    "index": rec.index,
                    "outcome": rec.outcome,
                    "reported_reward": json_default(rec.reported_reward),
                    "verdict": v.verdict,
                    "confidence": v.confidence,
                    "tags": v.tags,
                    "reasoning": v.rationale,
                    "parse_error": v.parse_error,
                }
            )

        verdicts_path = output_dir / "verdicts.jsonl"
        with verdicts_path.open("w", encoding="utf-8") as f:
            for row in verdict_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("Wrote %d verdicts to %s", len(verdict_rows), verdicts_path)

    steps = _aggregate_verdicts_by_step(verdict_rows)

    profile = {
        "source_artifact": str(artifact_dir),
        "judge_model": judge_model,
        "total_judged": len(verdict_rows),
        "steps": steps,
    }

    profile_path = output_dir / "judge_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote profile to %s (%d steps)", profile_path, len(steps))

    _plot_verdict_distribution(steps, output_dir, judge_model)

    log.info(
        "Done: %d total verdicts across %d steps",
        len(verdict_rows),
        len(steps),
    )


if __name__ == "__main__":
    main()
