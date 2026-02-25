#!/usr/bin/env python3
"""Compare d=3 condition normalization trajectories and gate rates (baseline / SStan / judge).

Usage:
    pixi run -e dev python scripts/compare_d3_conditions.py
    pixi run -e dev python scripts/compare_d3_conditions.py --out .scratch/deadline-data/figures
    pixi run -e dev python scripts/compare_d3_conditions.py --with-gate-panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from ppl_synthesis_reward_hacking.plotting.grpo_figures import (
    DPI_PNG,
    FIG_WIDTH_FULL,
    _plot_norm_panel,
    plot_normalization_trajectory,
)
from ppl_synthesis_reward_hacking.plotting.styles import apply_publication_style

REPO = Path(__file__).resolve().parent.parent

# d=3 relaunch artifact dirs (from AGENTS.md)
DEFAULT_BASELINE = REPO / "artifacts" / "sweeps" / "tinker_20260225_052004_514160"
DEFAULT_SSTAN = REPO / "artifacts" / "sweeps" / "tinker_20260225_052010_703046"
DEFAULT_JUDGE = REPO / "artifacts" / "sweeps" / "tinker_20260225_052017_778463"
DEFAULT_OUT = REPO / ".scratch" / "deadline-data" / "figures"

# ColorBrewer Dark2 trio (colorblind-safe)
C_BASELINE = "#1b9e77"  # teal
C_SSTAN = "#d95f02"  # orange
C_JUDGE = "#7570b3"  # purple

CONDITION_DEFS = [
    ("baseline", "Baseline (no gate)", C_BASELINE),
    ("sstan", "SStan gate", C_SSTAN),
    ("judge", "Judge gate", C_JUDGE),
]


def _load_norm_jsonl(path: Path) -> list[dict]:
    jsonl = path / "normalization_metrics.jsonl"
    if not jsonl.exists():
        return []
    rows = []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return sorted(rows, key=lambda r: r.get("step", 0))


def _build_summary(rows: list[dict], label: str) -> dict:
    if not rows:
        return {
            "label": label,
            "normalization_frac": {"steps": [], "mean": [], "std": None},
            "normalization_abs_log_mass": {"steps": [], "mean": [], "std": None},
        }
    steps = [r["step"] for r in rows]
    frac = [r.get("frac_non_normalized", 0.0) or 0.0 for r in rows]
    abs_lm = [r.get("mean_abs_log_mass", 0.0) or 0.0 for r in rows]
    return {
        "label": label,
        "normalization_frac": {"steps": steps, "mean": frac, "std": None},
        "normalization_abs_log_mass": {"steps": steps, "mean": abs_lm, "std": None},
    }


def _load_gate_rates(path: Path, gate_key: str) -> dict[int, float]:
    jsonl = path / "completions.jsonl"
    if not jsonl.exists():
        return {}
    by_batch: dict[int, list[bool]] = {}
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            batch = rec.get("batch")
            if batch is None:
                continue
            gate_info = (rec.get("metadata") or {}).get(gate_key)
            if gate_info:
                by_batch.setdefault(batch, []).append(bool(gate_info.get("penalty_applied", False)))
    return {b: sum(flags) / len(flags) for b, flags in sorted(by_batch.items()) if flags}


def _plot_gate_panel(ax: plt.Axes, gates: list[tuple[str, str, dict[int, float]]]) -> None:
    for label, color, rates in gates:
        if rates:
            steps = sorted(rates.keys())
            pct = [rates[s] * 100.0 for s in steps]
            ax.plot(steps, pct, color=color, linewidth=1.5, label=label, marker="o", markersize=3)

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5, zorder=0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gate penalty rate (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=7)
    ax.text(0.02, 0.92, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    sns.despine(ax=ax)


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"{name}.pdf/png")  # print to {out} implied


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--sstan", type=Path, default=DEFAULT_SSTAN)
    p.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--name", default="d3_condition_comparison")
    p.add_argument("--with-gate-panel", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_publication_style()

    dirs = {"baseline": args.baseline, "sstan": args.sstan, "judge": args.judge}
    norm_data = {k: _load_norm_jsonl(v) for k, v in dirs.items()}
    if not any(norm_data.values()):
        print("warning: no normalization data found")

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.with_gate_panel:
        summaries = [_build_summary(norm_data[k], label) for k, label, _ in CONDITION_DEFS]
        plot_normalization_trajectory(
            summaries, args.out / f"{args.name}.pdf", palette=[C_BASELINE, C_SSTAN, C_JUDGE]
        )
        return

    fig, (ax_frac, ax_mass, ax_gate) = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH_FULL + 2.0, 2.8),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.42},
    )

    for key, label, color in CONDITION_DEFS:
        rows = norm_data[key]
        if rows:
            steps = [r["step"] for r in rows]
            frac = [r.get("frac_non_normalized", 0.0) or 0.0 for r in rows]
            abs_lm = [r.get("mean_abs_log_mass", 0.0) or 0.0 for r in rows]
            _plot_norm_panel(
                ax_frac,
                ax_mass,
                steps,
                frac,
                abs_lm,
                label=label,
                color_frac=color,
                color_mass=color,
            )

    ax_mass.set_xlabel("Training step")
    ax_frac.legend(loc="best", fontsize=7)

    gates = []
    sstan_rates = _load_gate_rates(dirs["sstan"], "sstan_gate")
    judge_rates = _load_gate_rates(dirs["judge"], "judge_gate")
    if sstan_rates:
        gates.append(("SStan gate", C_SSTAN, sstan_rates))
    if judge_rates:
        gates.append(("Judge gate", C_JUDGE, judge_rates))

    if gates:
        _plot_gate_panel(ax_gate, gates)
    else:
        ax_gate.text(
            0.5,
            0.5,
            "No gate data",
            transform=ax_gate.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        ax_gate.set_xlabel("Training step")
        ax_gate.set_ylabel("Gate penalty rate (%)")
        sns.despine(ax=ax_gate)

    _save(fig, args.out, args.name)


if __name__ == "__main__":
    main()
