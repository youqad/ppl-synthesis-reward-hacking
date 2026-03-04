#!/usr/bin/env python3
"""Generate UAI 2026 figures from d=3 relaunch + earlier training data.

Produces:
  - lh_evidence.{pdf,png}          Horizontal bar chart: violation rate by condition
  - lh_emergence_detail.{pdf,png}  Event strip: when violations occur (d=3 only)

Usage:
    pixi run -e dev python scripts/generate_d3_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO / ".scratch" / "deadline-data" / "d3-relaunch"
DEFAULT_OUT = REPO / ".scratch" / "deadline-data" / "figures"

FIG_FULL = 6.75
FIG_COL = 3.25

C_UNGATED = "#c0392b"
C_JUDGE = "#27ae60"
C_SSTAN = "#e67e22"
C_CONTROL = "#7f8c8d"


from figure_style import setup_paper_style


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"  saved {name}.pdf/png -> {out}")
    plt.close(fig)


# ── Data loading ─────────────────────────────────────────────

D3_RUNS = {
    "A": ("tinker_20260225_052004_514160", "ungated", 0),
    "A2": ("tinker_20260225_182523_003071", "ungated", 10000),
    "A3": ("tinker_20260225_234806_675829", "ungated", 30000),
    "D": ("tinker_20260225_052017_778463", "judge", 0),
    "D2": ("tinker_20260225_234746_787617", "judge", 10000),
    "C": ("tinker_20260225_102711_253568", "sstan", 0),
}


def _load_d3(data_dir: Path) -> dict[str, dict]:
    result = {}
    for run_id, (artifact_dir, condition, seed) in D3_RUNS.items():
        path = data_dir / artifact_dir / "normalization_metrics.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result[run_id] = {
            "condition": condition,
            "seed": seed,
            "k": sum(r["n_non_normalized"] for r in rows),
            "n": sum(r["n_check_ok"] for r in rows),
            "steps": max(r["step"] for r in rows) + 1,
            "rows": rows,
        }
    return result


def _load_earlier(deadline_dir: Path) -> dict[str, dict]:
    result = {}
    path = deadline_dir / "step_summaries.json"
    if not path.exists():
        return result
    with open(path) as f:
        data = json.load(f)
    for seed_key, seed_val in [("seed0", 0), ("seed10k", 10000)]:
        rows = data.get(seed_key, [])
        k, n = 0, 0
        for r in rows:
            nc = r.get("norm_checked")
            if nc and nc > 0:
                frac = r.get("norm_frac_non_normalized", 0) or 0
                k += round(frac * nc)
                n += nc
        result[f"earlier_{seed_key}"] = {
            "condition": "ungated",
            "seed": seed_val,
            "k": k,
            "n": n,
            "steps": len(rows),
        }
    s30_path = deadline_dir / "seed30k_step_summaries.json"
    if s30_path.exists():
        with open(s30_path) as f:
            s30 = json.load(f)
        rows = s30.get("seed30k", s30 if isinstance(s30, list) else [])
        k, n = 0, 0
        for r in rows:
            nc = r.get("norm_checked", 0) or 0
            if nc > 0:
                frac = r.get("norm_frac_non_normalized", 0) or 0
                k += round(frac * nc)
                n += nc
        result["earlier_seed30k"] = {
            "condition": "ungated",
            "seed": 30000,
            "k": k,
            "n": n,
            "steps": len(rows),
        }
    return result


def _load_completions(data_dir: Path) -> dict[str, list[dict]]:
    result = {}
    for run_id, (artifact_dir, _, _) in D3_RUNS.items():
        path = data_dir / artifact_dir / "completions.jsonl"
        if not path.exists():
            continue
        result[run_id] = [json.loads(line) for line in path.open() if line.strip()]
    return result


# ── Figure 1: horizontal bar chart ──────────────────────────
# Rows grouped by condition. Each bar = one seed.
# Annotation: rate%, k/n, steps.


def fig_evidence(d3: dict, earlier: dict, out: Path) -> None:
    all_runs = {**earlier, **d3}

    # build bar entries: (k, n, steps, color, group)
    groups: dict[str, list[tuple[int, int, int, str]]] = {
        "No safety gate": [],
        "Judge gate": [],
        "SStan gate": [],
        "Untrained": [],
    }

    for _rid, r in all_runs.items():
        if r["condition"] == "ungated":
            groups["No safety gate"].append((r["k"], r["n"], r["steps"], C_UNGATED))
    for _rid, r in sorted(d3.items()):
        if r["condition"] == "judge":
            groups["Judge gate"].append((r["k"], r["n"], r["steps"], C_JUDGE))
        elif r["condition"] == "sstan":
            groups["SStan gate"].append((r["k"], r["n"], r["steps"], C_SSTAN))

    groups["Untrained"].append((0, 1000, 0, C_CONTROL))

    # sort ungated bars by rate descending
    groups["No safety gate"].sort(key=lambda t: t[0] / t[1] if t[1] else 0, reverse=True)

    # assign y positions: bars within groups, with gaps between groups
    y_positions: list[float] = []
    bar_data: list[tuple[int, int, int, str, str]] = []  # k, n, steps, color, group
    group_label_positions: dict[str, float] = {}  # group -> y center
    group_counts: dict[str, int] = {}

    y = 0
    for grp_name, entries in groups.items():
        if not entries:
            continue
        y_start = y
        group_counts[grp_name] = len(entries)
        for k, n, steps, color in entries:
            y_positions.append(y)
            bar_data.append((k, n, steps, color, grp_name))
            y += 1
        group_label_positions[grp_name] = (y_start + y - 1) / 2
        y += 0.7  # gap between groups

    n_bars = len(bar_data)
    fig_height = max(2.2, 0.32 * n_bars + 0.8)
    fig, ax = plt.subplots(figsize=(FIG_FULL, fig_height))

    rates = [k / n * 100 if n > 0 else 0 for k, n, _, _, _ in bar_data]
    colors = [c for _, _, _, c, _ in bar_data]

    ax.barh(
        y_positions,
        rates,
        height=0.6,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
        alpha=0.85,
    )

    # annotations to the right of each bar
    max_rate = max(rates) if rates else 1
    for i, (k, n, steps, color, _grp) in enumerate(bar_data):
        rate = rates[i]
        x_text = rate + max_rate * 0.02

        if k > 0:
            if steps > 0:
                label = rf"{rate:.1f}\% \quad {k}/{n} \quad ({steps} steps)"
            else:
                label = rf"{rate:.1f}\% \quad {k}/{n}"
            ax.text(
                x_text, y_positions[i], label, va="center", ha="left", fontsize=7.5, color=color
            )
        else:
            if steps > 0:
                label = rf"0\% \quad 0/{n} \quad ({steps} steps)"
            else:
                label = rf"0\% \quad 0/{n} \quad (7 pretrained models)"
            ax.text(
                x_text, y_positions[i], label, va="center", ha="left", fontsize=7.5, color="#888888"
            )

    # group headers on the left
    for grp_name, y_center in group_label_positions.items():
        cnt = group_counts[grp_name]
        suffix = f" ({cnt} seeds)" if cnt > 1 else " (1 seed)"
        if grp_name == "Untrained":
            suffix = ""
        ax.text(
            -0.01,
            y_center,
            rf"\textbf{{{grp_name}}}" + rf"\textnormal{{{suffix}}}",
            va="center",
            ha="right",
            fontsize=8.5,
            transform=ax.get_yaxis_transform(),
        )

    # horizontal separators between groups
    prev_grp = None
    for i, (_, _, _, _, grp) in enumerate(bar_data):
        if grp != prev_grp and prev_grp is not None:
            sep_y = (y_positions[i - 1] + y_positions[i]) / 2
            ax.axhline(sep_y, color="#dddddd", ls="-", lw=0.6, zorder=0)
        prev_grp = grp

    ax.set_yticks([])
    ax.set_xlabel(r"Programs violating normalization (\%)")
    ax.set_xlim(0, max_rate * 2.5)
    ax.set_ylim(max(y_positions) + 0.5, min(y_positions) - 0.5)

    # only place ticks in the data range, not in the annotation space
    tick_max = int(np.ceil(max_rate)) + 1
    ax.set_xticks(range(0, tick_max + 1, max(1, tick_max // 4)))
    ax.spines["bottom"].set_bounds(0, tick_max)

    ax.axvline(0, color="black", lw=0.4, zorder=1)

    _save(fig, out, "lh_evidence")

    # summary
    print("\n--- evidence summary ---")
    ungated_k = sum(k for k, _, _, _, g in bar_data if g == "No safety gate")
    ungated_n = sum(n for _, n, _, _, g in bar_data if g == "No safety gate")
    gated_k = sum(k for k, _, _, _, g in bar_data if g in ("Judge gate", "SStan gate"))
    gated_n = sum(n for _, n, _, _, g in bar_data if g in ("Judge gate", "SStan gate"))
    n_ungated_with = sum(1 for k, _, _, _, g in bar_data if g == "No safety gate" and k > 0)
    n_ungated_total = sum(1 for _, _, _, _, g in bar_data if g == "No safety gate")
    print(
        f"  Ungated: {ungated_k}/{ungated_n} = {ungated_k / ungated_n * 100:.2f}% "
        f"({n_ungated_with}/{n_ungated_total} seeds with violations)"
    )
    print(f"  Gated:   {gated_k}/{gated_n} = {gated_k / gated_n * 100:.2f}%")


# ── Figure 2: event strip ────────────────────────────────────
# One row per d=3 run. X marks = violations, dots = clean checks.


def fig_detail(d3: dict, comp_data: dict, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(FIG_FULL, 2.4),
        gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.38},
    )

    run_order = ["A", "A2", "A3", "D", "D2", "C"]
    labels = {
        "A": "Ungated, seed 1",
        "A2": "Ungated, seed 2",
        "A3": "Ungated, seed 3",
        "D": "Judge gate, seed 1",
        "D2": "Judge gate, seed 2",
        "C": "SStan gate, seed 1",
    }
    colors = {
        "A": C_UNGATED,
        "A2": C_UNGATED,
        "A3": C_UNGATED,
        "D": C_JUDGE,
        "D2": C_JUDGE,
        "C": C_SSTAN,
    }

    # panel (a): event strip
    for yi, rid in enumerate(run_order):
        if rid not in d3:
            continue
        rows = d3[rid]["rows"]
        color = colors[rid]
        max_step = max(r["step"] for r in rows)

        ax_a.plot(
            [0, max_step],
            [yi, yi],
            "-",
            color=color,
            lw=3,
            alpha=0.15,
            solid_capstyle="round",
            zorder=1,
        )

        for r in rows:
            if r["n_non_normalized"] > 0:
                ax_a.scatter(
                    r["step"],
                    yi,
                    s=120,
                    color=color,
                    marker="X",
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=4,
                )
                ax_a.annotate(
                    rf"\textbf{{{r['n_non_normalized']}/{r['n_check_ok']}}}",
                    xy=(r["step"], yi),
                    xytext=(0, 10),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    ha="center",
                    va="bottom",
                )
            else:
                ax_a.scatter(
                    r["step"],
                    yi,
                    s=12,
                    color=color,
                    marker="o",
                    alpha=0.25,
                    zorder=2,
                )

    ax_a.axhline(2.5, color="#cccccc", ls="--", lw=0.6, zorder=0)
    ax_a.set_yticks(range(len(run_order)))
    ax_a.set_yticklabels([labels[r] for r in run_order], fontsize=8)
    ax_a.set_xlabel("Training step")
    ax_a.set_ylim(len(run_order) - 0.5, -0.5)
    ax_a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax_a.scatter(
        [],
        [],
        s=120,
        color="#666666",
        marker="X",
        edgecolors="white",
        linewidths=0.6,
        label="Normalization violation",
    )
    ax_a.scatter([], [], s=12, color="#666666", marker="o", alpha=0.25, label="Clean check")
    ax_a.legend(
        loc="lower right",
        fontsize=6.5,
        handlelength=1.5,
        framealpha=0.9,
        edgecolor="#cccccc",
        fancybox=False,
    )
    ax_a.text(
        -0.02,
        1.05,
        r"\textbf{(a)} Per-step normalization checks",
        transform=ax_a.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
    )

    # panel (b): grouped bar — mean reward by condition
    bar_runs = ["A", "A2", "A3", "D", "D2", "C"]
    bar_labels = [labels[r] for r in bar_runs]
    bar_colors = [colors[r] for r in bar_runs]

    means = []
    maxes = []
    for rid in bar_runs:
        if rid not in comp_data:
            means.append(0)
            maxes.append(0)
            continue
        rewards = [
            r["reported_reward"]
            for r in comp_data[rid]
            if r.get("outcome") == "valid" and r.get("reported_reward") is not None
        ]
        means.append(float(np.mean(rewards)) if rewards else 0)
        maxes.append(float(np.max(rewards)) if rewards else 0)

    y_pos = np.arange(len(bar_runs))
    ax_b.barh(
        y_pos,
        means,
        height=0.55,
        color=bar_colors,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    ax_b.scatter(
        maxes,
        y_pos,
        s=30,
        color=bar_colors,
        marker="D",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    for i, m in enumerate(means):
        ax_b.text(
            m - 0.03,
            i,
            rf"{m:.2f}",
            fontsize=7,
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
        )

    ax_b.axhline(2.5, color="#cccccc", ls="--", lw=0.6, zorder=0)
    ax_b.axvline(0, color="black", ls="-", lw=0.4, zorder=1)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(bar_labels, fontsize=8)
    ax_b.set_xlabel("Reward (nats)")
    ax_b.invert_yaxis()

    ax_b.barh([], [], height=0, color="#999999", alpha=0.7, label="Mean")
    ax_b.scatter(
        [], [], s=30, color="#999999", marker="D", edgecolors="black", linewidths=0.4, label="Max"
    )
    ax_b.legend(
        loc="lower right",
        fontsize=6.5,
        handlelength=1.2,
        framealpha=0.9,
        edgecolor="#cccccc",
        fancybox=False,
    )
    ax_b.text(
        -0.02,
        1.05,
        r"\textbf{(b)} Reward per condition",
        transform=ax_b.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
    )

    _save(fig, out, "lh_emergence_detail")


# ── CLI ──────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-detail", action="store_true")
    p.add_argument(
        "--d3-only", action="store_true", help="Only show d=3 relaunch data (no earlier training)"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir or DEFAULT_DATA_DIR
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    out_dir = args.out or DEFAULT_OUT
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    setup_paper_style()

    d3 = _load_d3(data_dir)
    if not d3:
        raise SystemExit("No d=3 data found.")

    earlier = {} if args.d3_only else _load_earlier(REPO / ".scratch" / "deadline-data")

    print("Generating evidence figure...")
    fig_evidence(d3, earlier, out_dir)

    if not args.skip_detail:
        print("Loading completions...")
        comp = _load_completions(data_dir)
        if comp:
            print("Generating detail figure...")
            fig_detail(d3, comp, out_dir)

    print(f"\nDone. -> {out_dir}")


if __name__ == "__main__":
    main()
