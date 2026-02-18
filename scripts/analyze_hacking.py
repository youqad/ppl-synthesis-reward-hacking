#!/usr/bin/env python3
"""Reusable hacking analysis for GRPO training runs.

Reads completions.jsonl from any run directory and produces a structured report
covering reported reward trajectory, prior type distribution, concentrated Beta
detection, misspecification patterns, and health metrics.

Usage:
    python scripts/analyze_hacking.py artifacts/tinker_grpo_run5/
    python scripts/analyze_hacking.py --compare \\
        artifacts/tinker_grpo_run4/ artifacts/tinker_grpo_run5/
    python scripts/analyze_hacking.py artifacts/tinker_grpo_run5/ --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ppl_synthesis_reward_hacking.evaluation.hacking_analysis import analyze_run

# output formatting


def _fmt(v: float, width: int = 7) -> str:
    if math.isnan(v):
        return " " * (width - 3) + "n/a"
    return f"{v:>{width}.2f}"


def _fmt_pct(v: float, width: int = 6) -> str:
    if math.isnan(v):
        return " " * (width - 3) + "n/a"
    return f"{v:>{width}.1f}"


def _fmt_slope(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{v:.4f}"


def print_summary_table(report: dict) -> None:
    """Print per-step summary table."""
    run_dir = report["run_dir"]
    n_steps = report["n_steps"]
    traj = report["reported_trajectory"]

    print(f"\n{'=' * 80}")
    print(f"  Hacking Analysis: {run_dir}")
    print(
        f"  Steps: {n_steps}  |  Records: {report['n_total_records']}  |  "
        f"Valid: {report['n_total_valid']}"
    )
    slope_str = "n/a" if math.isnan(traj["slope"]) else f"{traj['slope']:.4f}"
    print(f"  Reported trend: {traj['trend']} (slope={slope_str})")
    print(f"{'=' * 80}")

    # per-step table header
    print(
        f"\n{'Step':>5} {'Valid':>6} "
        f"{'Rep':>7} {'Div%':>6} {'Fail%':>6} "
        f"{'Beta':>5} {'Hier':>5} {'LogN':>5} {'Uni':>5} {'CncB':>5}"
    )
    print("-" * 80)

    per_step = report["per_step"]
    for step_str in sorted(per_step.keys(), key=int):
        s = per_step[step_str]
        pt = s["prior_types"]
        print(
            f"{step_str:>5} {s['n_valid']:>6} "
            f"{_fmt(s['reported_mean'])} "
            f"{_fmt_pct(s['diversity_pct'])} {_fmt_pct(s['exec_fail_pct'])} "
            f"{pt.get('beta', 0):>5} {pt.get('hierarchical', 0):>5} "
            f"{pt.get('logit_normal', 0):>5} {pt.get('uniform', 0):>5} "
            f"{s['n_concentrated_beta']:>5}"
        )

    # aggregates
    print("\n--- Aggregates ---")
    print(f"Prior distribution: {dict(report['prior_distribution'])}")
    if report["beta_concentrations"]:
        print(f"Concentrated Betas: {dict(report['beta_concentrations'])}")
    else:
        print("Concentrated Betas: none detected")
    if report["misspecifications"]:
        print(f"Misspecifications:  {dict(report['misspecifications'])}")
    else:
        print("Misspecifications:  none detected")

    h = report["health"]
    print(
        f"\nHealth: valid={h['valid_rate']:.1f}%  diversity={h['mean_diversity_pct']:.1f}%  "
        f"exec_fail={h['mean_exec_fail_pct']:.1f}%  parse_fail={h['mean_parse_fail_pct']:.1f}%"
    )
    print()


def print_comparison(reports: list[dict]) -> None:
    """Print side-by-side comparison of multiple runs."""
    _print_comparison_header(reports)
    _print_comparison_metrics(reports)
    _print_comparison_distribution(reports, title="Prior Types", key="prior_distribution")
    _print_comparison_distribution(reports, title="Misspecifications", key="misspecifications")
    print()


def _print_comparison_header(reports: list[dict]) -> None:
    print(f"\n{'=' * 80}")
    print(f"  Cross-Run Comparison ({len(reports)} runs)")
    print(f"{'=' * 80}\n")

    header = f"{'Metric':<30}"
    for r in reports:
        name = Path(r["run_dir"]).name
        header += f" {name:>20}"
    print(header)
    print("-" * (30 + 21 * len(reports)))


def _print_comparison_metrics(reports: list[dict]) -> None:
    rows = _comparison_rows()
    for label, fn in rows:
        line = f"{label:<30}"
        for report in reports:
            line += f" {fn(report):>20}"
        print(line)


def _comparison_rows():
    return [
        ("Steps", lambda r: str(r["n_steps"])),
        ("Total records", lambda r: str(r["n_total_records"])),
        ("Valid records", lambda r: str(r["n_total_valid"])),
        ("Reported slope", lambda r: _fmt_slope(r["reported_trajectory"]["slope"])),
        ("Reported trend", lambda r: r["reported_trajectory"]["trend"]),
        ("Reported first step", lambda r: _fmt(r["reported_trajectory"]["first_step_mean"])),
        ("Reported last step", lambda r: _fmt(r["reported_trajectory"]["last_step_mean"])),
        ("Reported delta", lambda r: _fmt(r["reported_trajectory"]["delta"])),
        ("Reported mean", lambda r: _fmt(r["reported_trajectory"]["mean"])),
        ("Valid rate %", lambda r: f"{r['health']['valid_rate']:.1f}"),
        ("Diversity %", lambda r: f"{r['health']['mean_diversity_pct']:.1f}"),
        ("Exec fail %", lambda r: f"{r['health']['mean_exec_fail_pct']:.1f}"),
        ("Concentrated Betas", lambda r: str(sum(r["beta_concentrations"].values()))),
    ]


def _print_comparison_distribution(reports: list[dict], *, title: str, key: str) -> None:
    labels = sorted({name for report in reports for name in report[key].keys()})
    if not labels:
        return
    print(f"\n{title:<30}")
    for label in labels:
        line = f"  {label:<28}"
        for report in reports:
            line += f" {report[key].get(label, 0):>20}"
        print(line)


def _sanitize_for_json(obj):
    """Replace nan/inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze reward hacking signals in GRPO training runs")
    p.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Run directory containing completions.jsonl",
    )
    p.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="DIR",
        help="Compare multiple run directories side-by-side",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report only (to stdout)",
    )
    p.add_argument(
        "--exclude-clipped",
        action="store_true",
        help="Exclude records with clipped observed logps (diagnostic mode)",
    )
    p.add_argument(
        "--winsorize",
        type=float,
        default=None,
        metavar="P",
        help="Winsorize gap statistics to [q(1-P), q(P)] (e.g., 0.99). Default: off",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing hacking_analysis.json to run directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare:
        _handle_compare_mode(args)
        return
    _handle_single_run_mode(args)


def _handle_compare_mode(args: argparse.Namespace) -> None:
    reports = _collect_comparison_reports(
        args.compare,
        exclude_clipped=args.exclude_clipped,
        winsorize=args.winsorize,
    )
    if not reports:
        print("No valid runs to compare.", file=sys.stderr)
        sys.exit(1)
    if args.json:
        _print_json(reports)
        return
    print_comparison(reports)
    for report in reports:
        print_summary_table(report)


def _collect_comparison_reports(
    run_dirs: list[Path],
    *,
    exclude_clipped: bool,
    winsorize: float | None,
) -> list[dict]:
    reports: list[dict] = []
    for run_dir in run_dirs:
        try:
            reports.append(
                analyze_run(run_dir, exclude_clipped=exclude_clipped, winsorize=winsorize)
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"SKIP {run_dir}: {err}", file=sys.stderr)
    return reports


def _handle_single_run_mode(args: argparse.Namespace) -> None:
    if not args.run_dir:
        print("Provide a run directory or use --compare.", file=sys.stderr)
        sys.exit(1)

    report = analyze_run(
        args.run_dir, exclude_clipped=args.exclude_clipped, winsorize=args.winsorize
    )
    _emit_report(report, json_mode=args.json)
    if args.no_save:
        return
    _save_report(args.run_dir, report, quiet=args.json)


def _emit_report(report: dict | list[dict], *, json_mode: bool) -> None:
    if json_mode:
        _print_json(report)
        return
    if isinstance(report, dict):
        print_summary_table(report)


def _print_json(payload: dict | list[dict]) -> None:
    json.dump(_sanitize_for_json(payload), sys.stdout, indent=2)
    print()


def _save_report(run_dir: Path, report: dict, *, quiet: bool) -> None:
    out_path = run_dir / "hacking_analysis.json"
    with open(out_path, "w") as f:
        json.dump(_sanitize_for_json(report), f, indent=2)
    if not quiet:
        print(f"JSON report saved to {out_path}")


if __name__ == "__main__":
    main()
