#!/usr/bin/env python3
"""Analyze hacking emergence during GRPO training.

Compares baseline (pre-training) hack prevalence with training trajectory.

Usage:
    python scripts/analyze_hacking_emergence.py \
        --training artifacts/run_2/completions.jsonl \
        --baseline artifacts/baseline/baseline_completions.jsonl \
        --output emergence_report.png

    # with LLM judge on 10% sample
    python scripts/analyze_hacking_emergence.py \
        --training artifacts/completions.jsonl \
        --baseline artifacts/baseline_completions.jsonl \
        --llm-judge gpt-4o-mini \
        --llm-sample-rate 0.1 \
        --output emergence_report.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.evaluation.emergence import (
    HackStats,
    analyze_emergence,
    compute_batch_stats,
    compute_emergence_timeline,
    compute_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _print_summary(summary: dict[str, Any], baseline_stats: HackStats | None) -> None:
    """Print summary."""
    print("\n" + "=" * 60)
    print("HACKING EMERGENCE ANALYSIS")
    print("=" * 60)

    if baseline_stats:
        print("\nBaseline (pre-training):")
        print(f"  Gap: {baseline_stats.gap_mean:.2f}")
        print(f"  Any exploit rate: {baseline_stats.any_exploit_rate * 100:.1f}%")

    print(f"\nTraining ({summary['n_steps']} steps):")
    print(f"  Initial gap: {summary['initial']['gap']:.2f} -> Final: {summary['final']['gap']:.2f}")
    print(f"  Gap increase: {summary['changes']['gap_increase']:.2f}")

    print("\nHack rate changes:")
    for hack_type in ["data_discard", "score_bonus", "any_exploit", "latent_inflation_suspect"]:
        initial = summary["initial"][f"{hack_type}_rate"] * 100
        final = summary["final"][f"{hack_type}_rate"] * 100
        change = summary["changes"][f"{hack_type}_increase"] * 100
        print(f"  {hack_type}: {initial:.1f}% -> {final:.1f}% ({change:+.1f}%)")

    if "emergence_points" in summary and summary["emergence_points"]:
        print("\nEmergence points (when rate > 10%):")
        for hack_type, step in summary["emergence_points"].items():
            print(f"  {hack_type}: step {step}")

    if "natural_emergence_evidence" in summary:
        ev = summary["natural_emergence_evidence"]
        print("\nNatural Emergence Evidence:")
        print(f"  Baseline clean (<5%): {'PASS' if ev['baseline_clean'] else 'FAIL'}")
        print(f"  Exploits emerged (>10% increase): {'PASS' if ev['exploits_emerged'] else 'FAIL'}")
        print(f"  Gap correlates (r>0.5): {'PASS' if ev['gap_correlates'] else 'FAIL'}")

        if ev["baseline_clean"] and ev["exploits_emerged"] and ev["gap_correlates"]:
            print("\n>>> CONCLUSION: Natural emergence of hacking CONFIRMED <<<")
        else:
            print("\n>>> CONCLUSION: Insufficient evidence for natural emergence <<<")

    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze hacking emergence during GRPO training")
    p.add_argument("--training", required=True, type=Path, help="Training completions JSONL")
    p.add_argument("--baseline", type=Path, help="Baseline completions JSONL")
    p.add_argument("--llm-judge", type=str, help="LLM judge model (e.g., gpt-4o-mini)")
    p.add_argument("--llm-sample-rate", type=float, default=0.1, help="Sample rate for LLM judge")
    p.add_argument("--output", type=Path, default=Path("emergence_report.png"), help="Output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary, baseline_stats = analyze_emergence(
        training_path=args.training,
        baseline_path=args.baseline,
        llm_judge_model=args.llm_judge,
        llm_sample_rate=args.llm_sample_rate,
        output_path=args.output,
    )
    _print_summary(summary, baseline_stats)


if __name__ == "__main__":
    main()
