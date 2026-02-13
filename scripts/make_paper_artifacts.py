#!/usr/bin/env python3
"""Generate reproducible figures/tables used by the LaTeX paper.

This script is intentionally orchestration-heavy: it ties together existing
analysis scripts, writes a manifest, and emits paper-ready artifacts.

Usage:
    pixi run -e dev python scripts/make_paper_artifacts.py --config configs/paper/runlist.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_info() -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
            return out
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain=v1"]) not in (None, "")
    return {"head": head, "branch": branch, "dirty": bool(dirty)}


def _run_analyze_hacking(
    run_dir: Path, *, exclude_clipped: bool, winsorize: float | None
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/analyze_hacking.py",
        str(run_dir),
        "--json",
        "--no-save",
    ]
    if exclude_clipped:
        cmd.append("--exclude-clipped")
    if winsorize is not None:
        cmd += ["--winsorize", str(winsorize)]
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return json.loads(out)


def _run_analyze_emergence(
    *,
    training_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/analyze_hacking_emergence.py",
        "--training",
        str(training_path),
        "--baseline",
        str(baseline_path),
        "--output",
        str(output_path),
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    summary_path = output_path.with_suffix(".json")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _extract_step_series(
    report: dict[str, Any], metric: str
) -> dict[int, float]:
    out: dict[int, float] = {}
    per_step = report.get("per_step") or {}
    if not isinstance(per_step, dict):
        return out
    for step_str, sa in per_step.items():
        try:
            step = int(step_str)
        except Exception:
            continue
        if not isinstance(sa, dict):
            continue
        v = sa.get(metric)
        if v is None:
            continue
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            continue
        out[step] = fv
    return out


def _aggregate_series(reports: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    step_to_vals: dict[int, list[float]] = defaultdict(list)
    for r in reports:
        series = _extract_step_series(r, metric)
        for step, v in series.items():
            step_to_vals[step].append(v)

    steps = sorted(step_to_vals.keys())
    means = [float(np.mean(step_to_vals[s])) for s in steps]
    stds = [float(np.std(step_to_vals[s])) for s in steps]
    ns = [len(step_to_vals[s]) for s in steps]
    return {"steps": steps, "mean": means, "std": stds, "n": ns}


def _maybe_plot_gap_trajectory(
    condition_summaries: list[dict[str, Any]],
    *,
    out_pdf: Path,
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for cond in condition_summaries:
        name = cond["label"]
        s = cond["gap"]
        steps = s["steps"]
        mean_ = s["mean"]
        std_ = s["std"]
        if not steps:
            continue
        ax.plot(steps, mean_, linewidth=2, label=name)
        ax.fill_between(
            steps,
            [m - sd for m, sd in zip(mean_, std_, strict=False)],
            [m + sd for m, sd in zip(mean_, std_, strict=False)],
            alpha=0.15,
        )

    ax.set_title("Reward Gap Trajectory (reported - oracle)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _maybe_plot_failure_rates(
    condition_summaries: list[dict[str, Any]],
    *,
    out_pdf: Path,
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for cond in condition_summaries:
        name = cond["label"]
        exec_s = cond["exec_fail_pct"]
        parse_s = cond["parse_fail_pct"]
        if exec_s["steps"]:
            ax1.plot(exec_s["steps"], exec_s["mean"], linewidth=2, label=name)
        if parse_s["steps"]:
            ax2.plot(parse_s["steps"], parse_s["mean"], linewidth=2, label=name)

    ax1.set_title("Exec failure rate (%)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("%")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Parse failure rate (%)")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("%")
    ax2.grid(True, alpha=0.3)

    ax1.legend(loc="best")
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt_pm_std(mean_: float, std_: float) -> str:
    if math.isnan(mean_) or math.isnan(std_):
        return "n/a"
    return f"{mean_:.3f} $\\pm$ {std_:.3f}"


def _write_baseline_vs_trained_table(
    out_path: Path,
    *,
    baseline: dict[str, Any] | None,
    condition_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("% Autogenerated by scripts/make_paper_artifacts.py")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Condition & Gap (final) & Any exploit (final) & Valid gap rate \\\\")
    lines.append("\\midrule")

    if baseline is not None:
        lines.append(
            "Baseline (untrained) "
            f"& {baseline.get('gap', float('nan')):.3f} "
            f"& {baseline.get('any_exploit_rate', float('nan')) * 100:.1f}\\% "
            f"& {baseline.get('gap_valid_rate', float('nan')) * 100:.1f}\\% \\\\"
        )
        lines.append("\\midrule")

    for row in condition_rows:
        lines.append(
            f"{row['label']} "
            f"& {row['gap_pm_std']} "
            f"& {row['any_exploit_pm_std']} "
            f"& {row['gap_valid_pm_std']} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _compute_mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper artifacts (figures/tables)")
    p.add_argument("--config", type=Path, default=Path("configs/paper/runlist.yaml"))
    p.add_argument("--out", type=Path, default=Path("paper_artifacts"))
    args = p.parse_args()

    cfg_path = (REPO_ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    out_dir = (REPO_ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("runlist config must be a mapping")

    analysis_cfg = cfg.get("analysis") or {}
    exclude_clipped = bool(analysis_cfg.get("exclude_clipped", False))
    winsorize = analysis_cfg.get("winsorize", None)
    winsorize_f = float(winsorize) if winsorize is not None else None

    baseline_path_raw = cfg.get("baseline")
    baseline_path: Path | None = None
    if isinstance(baseline_path_raw, str) and baseline_path_raw.strip():
        candidate = (REPO_ROOT / baseline_path_raw).resolve()
        if candidate.exists():
            baseline_path = candidate

    conditions = cfg.get("conditions") or []
    if not isinstance(conditions, list) or not conditions:
        raise RuntimeError("config must contain non-empty `conditions` list")

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "repo": str(REPO_ROOT),
        "git": _git_info(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "config_path": str(cfg_path),
        "analysis": {"exclude_clipped": exclude_clipped, "winsorize": winsorize_f},
        "baseline_path": str(baseline_path) if baseline_path else None,
        "conditions": [],
    }

    condition_summaries: list[dict[str, Any]] = []
    baseline_summary: dict[str, Any] | None = None
    condition_rows: list[dict[str, Any]] = []

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        cond_name = str(cond.get("name") or "condition")
        cond_label = str(cond.get("label") or cond_name)
        run_dirs = cond.get("runs") or []
        if not isinstance(run_dirs, list) or not run_dirs:
            raise RuntimeError(f"condition {cond_name} missing `runs` list")

        reports: list[dict[str, Any]] = []
        emergence_summaries: list[dict[str, Any]] = []
        resolved_runs: list[str] = []

        for run_item in run_dirs:
            run_dir = (REPO_ROOT / str(run_item)).resolve()
            resolved_runs.append(str(run_dir))
            completions_path = run_dir / "completions.jsonl"
            if not completions_path.exists():
                print(f"SKIP (no completions): {run_dir}", file=sys.stderr)
                continue

            rep = _run_analyze_hacking(
                run_dir, exclude_clipped=exclude_clipped, winsorize=winsorize_f
            )
            reports.append(rep)

            # Optional baseline comparison (needs baseline file present)
            if baseline_path is not None:
                emergence_dir = out_dir / "emergence"
                emergence_dir.mkdir(parents=True, exist_ok=True)
                out_path = emergence_dir / f"{cond_name}_{run_dir.name}_emergence.png"
                summ = _run_analyze_emergence(
                    training_path=completions_path,
                    baseline_path=baseline_path,
                    output_path=out_path,
                )
                emergence_summaries.append(summ)

                # take baseline summary from first run
                if baseline_summary is None and "baseline" in summ:
                    baseline_summary = summ["baseline"]

            # Save per-run analysis JSON
            analysis_dir = out_dir / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            (analysis_dir / f"{cond_name}_{run_dir.name}_hacking_analysis.json").write_text(
                json.dumps(rep, indent=2), encoding="utf-8"
            )

        if not reports:
            continue

        gap = _aggregate_series(reports, "gap_mean")
        exec_fail_pct = _aggregate_series(reports, "exec_fail_pct")
        parse_fail_pct = _aggregate_series(reports, "parse_fail_pct")

        condition_summaries.append(
            {
                "name": cond_name,
                "label": cond_label,
                "runs": resolved_runs,
                "n_runs_used": len(reports),
                "gap": gap,
                "exec_fail_pct": exec_fail_pct,
                "parse_fail_pct": parse_fail_pct,
            }
        )

        manifest["conditions"].append(
            {
                "name": cond_name,
                "label": cond_label,
                "runs": resolved_runs,
                "n_runs_used": len(reports),
            }
        )

        # For the baseline-vs-trained table, we prefer emergence summaries when available.
        if emergence_summaries:
            final_gaps = [float(s["final"]["gap"]) for s in emergence_summaries if "final" in s]
            final_any_exploit = [
                float(s["final"]["any_exploit_rate"]) for s in emergence_summaries if "final" in s
            ]
            final_gap_valid = [
                float(s["final"]["gap_valid_rate"]) for s in emergence_summaries if "final" in s
            ]
            g_m, g_s = _compute_mean_std(final_gaps)
            e_m, e_s = _compute_mean_std(final_any_exploit)
            v_m, v_s = _compute_mean_std(final_gap_valid)
            condition_rows.append(
                {
                    "label": cond_label,
                    "gap_pm_std": _fmt_pm_std(g_m, g_s),
                    "any_exploit_pm_std": _fmt_pm_std(e_m * 100.0, e_s * 100.0),
                    "gap_valid_pm_std": _fmt_pm_std(v_m * 100.0, v_s * 100.0),
                }
            )
        else:
            # Fall back to using the final step from analyze_hacking.py output (gap-only).
            final_gaps = []
            for r in reports:
                traj = r.get("gap_trajectory") or {}
                last = traj.get("last_step_mean")
                if isinstance(last, (int, float)) and not (math.isnan(float(last)) or math.isinf(float(last))):
                    final_gaps.append(float(last))
            g_m, g_s = _compute_mean_std(final_gaps)
            condition_rows.append(
                {
                    "label": cond_label,
                    "gap_pm_std": _fmt_pm_std(g_m, g_s),
                    "any_exploit_pm_std": "n/a",
                    "gap_valid_pm_std": "n/a",
                }
            )

    # Plots
    _maybe_plot_gap_trajectory(
        condition_summaries,
        out_pdf=out_dir / "gap_trajectory.pdf",
        out_png=out_dir / "gap_trajectory.png",
    )
    _maybe_plot_failure_rates(
        condition_summaries,
        out_pdf=out_dir / "failure_rates.pdf",
        out_png=out_dir / "failure_rates.png",
    )

    # Tables
    _write_baseline_vs_trained_table(
        out_dir / "baseline_vs_trained_table.tex",
        baseline=baseline_summary,
        condition_rows=condition_rows,
    )

    # Safety matrix
    subprocess.check_call(
        [
            sys.executable,
            "scripts/make_safety_gate_matrix.py",
            "--out",
            str(out_dir / "safety_gate_matrix_table.tex"),
            "--json",
            str(out_dir / "safety_gate_matrix.json"),
        ],
        cwd=REPO_ROOT,
    )

    # Manifest
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote paper artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

