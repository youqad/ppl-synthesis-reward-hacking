#!/usr/bin/env python3
"""Generate publication-quality figures for LH emergence (UAI 2026).

Reads step_summaries.json + seed30k_step_summaries.json + safepymc_gate_results.json
and produces:
  - lh_emergence_3panel.{pdf,png}         Fig 1: pm.Potential prevalence, max reward, normalization
  - lh_cross_evidence.{pdf,png}           Fig 2: cross-run forest + SafePyMC gate
  - lh_figS1_potential_prevalence.{pdf,png}  Supplementary: pm.Potential + forest
  - LaTeX table + key numbers              printed to stdout

Usage:
    pixi run -e dev python scripts/generate_paper_figures.py
    pixi run -e dev python scripts/generate_paper_figures.py --out paper_artifacts/run7_latest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

from ppl_synthesis_reward_hacking.experiments.scoring_env import DEFAULT_LOGP_CEIL as LOGP_CEILING

REPO = Path(__file__).resolve().parent.parent
DEADLINE_DIR = REPO / ".scratch" / "deadline-data"
DATA = DEADLINE_DIR / "step_summaries.json"
SEED30K_DATA = DEADLINE_DIR / "seed30k_step_summaries.json"
GATE_DATA = DEADLINE_DIR / "safepymc_gate_results.json"
DEFAULT_OUT = DEADLINE_DIR / "figures"

BASELINE_RATE = 0.006
FIG_FULL = 6.75  # UAI full column width (inches)
FIG_COL = 3.25  # UAI single column width

# colorblind-safe palette (Brewer)
C_S0 = "#2166ac"
C_S10K = "#b2182b"
C_S30K = "#7570b3"
C_BASELINE = "#4daf4a"
C_POOL = "#333333"
C_EXPLOIT = "#d95f02"
C_HONEST = "#1b9e77"
C_ORACLE = "#000000"
C_ABLATION_CLEAN = "#66c2a5"
C_ABLATION_DIRTY = "#fc8d62"

SEEDS = [
    ("seed0", "Seed 0", C_S0),
    ("seed10k", "Seed 10k", C_S10K),
    ("seed30k", "Seed 30k", C_S30K),
]
# seed0/seed10k have full fields; seed30k is minimal
SEEDS_FULL = SEEDS[:2]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "legend.framealpha": 0.0,
            "legend.edgecolor": "none",
            "legend.fancybox": False,
            "lines.linewidth": 2.2,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - w), min(1.0, c + w)


def _rolling(vals: list[float], window: int = 5) -> np.ndarray:
    arr = np.array(vals, dtype=float)
    out = np.convolve(arr, np.ones(window) / window, mode="same")
    half = window // 2
    for i in range(half):
        w = i + half + 1
        out[i] = np.mean(arr[:w])
        out[-(i + 1)] = np.mean(arr[-w:])
    return out


def _agg(
    data: dict,
    seeds: list[str],
    field_k: str,
    field_n: str,
    step_range: tuple[int, int] | None = None,
) -> tuple[int, int]:
    k_tot = n_tot = 0
    for s in seeds:
        for r in data[s]:
            if step_range and not (step_range[0] <= r["step"] <= step_range[1]):
                continue
            k_tot += r.get(field_k, 0)
            n_tot += r.get(field_n, 0)
    return k_tot, n_tot


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"  {name}.pdf/png")
    plt.close(fig)


# ── Figure 1: LH emergence 3-panel (matches paper caption) ──────


def fig1_emergence_3panel(data: dict, out: Path) -> None:
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(FIG_FULL, 2.6),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.42},
    )

    bpct = BASELINE_RATE * 100

    # ── (a) pm.Potential prevalence over training steps ──
    for sk, lab, col in SEEDS_FULL:
        rows = data[sk]
        steps = np.array([r["step"] for r in rows])
        pct = np.array([r["has_potential_frac"] * 100 for r in rows])

        ax_a.scatter(steps, pct, s=8, color=col, alpha=0.2, zorder=2, edgecolors="none")
        trend = _rolling(pct.tolist(), window=5)
        ax_a.plot(steps, trend, "-", color=col, lw=2.0, label=lab, zorder=3)

    ax_a.axhline(bpct, color=C_BASELINE, ls="--", lw=1.0, zorder=0)

    all_sk_a = [s[0] for s in SEEDS_FULL]
    k_pot_a, n_valid_a = _agg(data, all_sk_a, "has_potential_count", "valid")
    fold_a = (k_pot_a / n_valid_a) / BASELINE_RATE
    ax_a.text(
        0.97,
        0.92,
        rf"\textbf{{{fold_a:.0f}$\times$}} above base",
        transform=ax_a.transAxes,
        fontsize=8,
        color=C_BASELINE,
        ha="right",
        va="top",
    )
    ax_a.annotate(
        r"base rate",
        xy=(25, bpct),
        xytext=(25, 1.6),
        fontsize=7,
        color=C_BASELINE,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_BASELINE, lw=0.8, shrinkB=1),
    )

    ax_a.set_xlabel("Training step")
    ax_a.set_ylabel(r"\texttt{pm.Potential} prevalence (\%)")
    ax_a.set_ylim(-0.3, 12)
    ax_a.legend(loc="upper left", fontsize=7, handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    ax_a.text(
        -0.15, 1.05, r"\textbf{(a)}", transform=ax_a.transAxes, fontsize=11, va="bottom", ha="left"
    )

    # ── (b) Max reward per batch ──
    total_ceil = 0
    for sk, lab, col in SEEDS:
        rows = data[sk]
        steps = np.array([r["step"] for r in rows])
        maxr = np.array([r["max_reward"] for r in rows])
        ceil = np.array([r.get("ceiling_hits", 0) > 0 for r in rows])
        total_ceil += sum(r.get("ceiling_hits", 0) for r in rows)

        jitter = {"seed0": -0.3, "seed10k": 0.0, "seed30k": 0.3}.get(sk, 0)
        ax_b.scatter(
            steps[~ceil],
            maxr[~ceil],
            s=14,
            color=col,
            alpha=0.4,
            zorder=3,
            edgecolors="none",
            label=lab,
        )
        if ceil.any():
            ax_b.scatter(
                steps[ceil] + jitter,
                maxr[ceil],
                s=40,
                marker="^",
                color=col,
                edgecolors="black",
                linewidths=0.4,
                zorder=4,
            )

    ax_b.axhline(0, color=C_ORACLE, ls="-", lw=0.6, zorder=1)
    ax_b.axhline(LOGP_CEILING, color=C_EXPLOIT, ls=":", lw=1.0, alpha=0.6, zorder=1)

    # phantom scatter for ceiling-hit legend entry
    ax_b.scatter(
        [],
        [],
        s=40,
        marker="^",
        color="#666666",
        edgecolors="black",
        linewidths=0.4,
        label=f"Ceiling ({total_ceil})",
    )
    ax_b.text(
        0.97,
        0.06,
        r"oracle bound ($\log Z_p = 0$)",
        transform=ax_b.transAxes,
        fontsize=6.5,
        ha="right",
        va="bottom",
        color=C_ORACLE,
    )

    ax_b.set_xlabel("Training step")
    ax_b.set_ylabel("Max reward (nats)")
    ax_b.legend(
        loc="center left",
        fontsize=6.5,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.25,
        markerscale=1.2,
        framealpha=0.85,
        edgecolor="none",
    )
    ax_b.text(
        -0.15, 1.05, r"\textbf{(b)}", transform=ax_b.transAxes, fontsize=11, va="bottom", ha="left"
    )

    # ── (c) Normalization failure rate ──
    # per-seed offsets to avoid annotation collisions
    annot_offsets = {"seed0": (6, 8), "seed10k": (6, -14), "seed30k": (6, 6)}
    for sk, lab, col in SEEDS:
        rows = data[sk]
        checked = [
            (r["step"], r["norm_frac_non_normalized"], r["norm_checked"])
            for r in rows
            if r.get("norm_checked") is not None and r["norm_checked"] > 0
        ]

        if not checked:
            continue
        steps_c = np.array([c[0] for c in checked])
        rates_c = np.array([c[1] * 100 for c in checked])

        ax_c.scatter(
            steps_c,
            rates_c,
            s=12,
            color=col,
            alpha=0.3,
            zorder=2,
            edgecolors="none",
            label=lab,
        )
        nz = rates_c > 0
        if nz.any():
            ax_c.scatter(
                steps_c[nz],
                rates_c[nz],
                s=50,
                color=col,
                zorder=3,
                edgecolors="white",
                linewidths=0.5,
            )

        dx, dy = annot_offsets.get(sk, (5, 6))
        prev_ann_step = -999
        flip = False
        for s, rate, n_chk in checked:
            k_fail = round(rate * n_chk)
            if k_fail > 0:
                if abs(s - prev_ann_step) <= 2:
                    flip = not flip
                else:
                    flip = False
                prev_ann_step = s
                dy_use = -abs(dy) - 4 if flip else dy
                ax_c.annotate(
                    rf"{k_fail}/{n_chk}",
                    xy=(s, rate * 100),
                    xytext=(dx, dy_use),
                    textcoords="offset points",
                    fontsize=8,
                    color=col,
                    ha="left",
                    va="bottom",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        fc="white",
                        ec=col,
                        lw=0.4,
                        alpha=0.9,
                    ),
                )

    ax_c.set_xlabel("Training step")
    ax_c.set_ylabel(r"Non-normalized (\%)")
    ax_c.set_ylim(-1, max(25, ax_c.get_ylim()[1]))
    ax_c.legend(loc="upper left", fontsize=7, handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    ax_c.text(
        -0.15, 1.05, r"\textbf{(c)}", transform=ax_c.transAxes, fontsize=11, va="bottom", ha="left"
    )

    _save(fig, out, "lh_emergence_3panel")


# hardcoded normalization data; sources annotated per entry
# (label, k_failures, n_checked, group, color)
CROSS_RUN_NORM = [
    # RL training seeds (from step_summaries.json / seed30k_step_summaries.json)
    ("Seed 0 ($d{=}3$)", 2, 46, "train", C_S0),
    ("Seed 10k ($d{=}3$)", 3, 49, "train", C_S10K),
    ("Seed 30k ($d{=}3$)", 3, 176, "train", C_S30K),
    # GLM-5 sweep (CLAUDE.md: hky539sh, ~4/19 checked at step 2)
    ("GLM-5 sweep ($d{=}5$)", 4, 19, "train", "#e7298a"),
    # Sam's base-rate sweep: ~0% across ~1000 programs from untrained models
    ("Untrained baseline", 0, 1000, "train", C_BASELINE),
    # Metric ablations (CLAUDE.md documented values)
    ("Ablation: LogML", 0, 130, "ablation", C_ABLATION_CLEAN),
    ("Ablation: WAIC", 1, 90, "ablation", C_ABLATION_CLEAN),
    ("Ablation: ELPD", 200, 200, "ablation", C_ABLATION_DIRTY),
    ("Ablation: BIC", 70, 70, "ablation", C_ABLATION_DIRTY),
]


def fig2_cross_evidence(data: dict, gate_data: dict | None, out: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(FIG_FULL, 3.0),
        gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.45},
    )

    # ── (a) Cross-run normalization forest plot ──
    entries = CROSS_RUN_NORM
    n_entries = len(entries)

    train_count = sum(1 for e in entries if e[3] == "train")
    y_positions = []
    y = 0
    for i in range(n_entries):
        if i == train_count:
            y += 0.6  # gap between groups
        y_positions.append(y)
        y += 1

    for i, (lab, k, n, _grp, col) in enumerate(entries):
        p, lo, hi = wilson_ci(k, n)
        pct = p * 100
        lo_pct = lo * 100
        hi_pct = hi * 100
        yi = y_positions[i]

        marker = "D" if "Untrained" in lab else "o"
        ms = 6 if marker == "o" else 5
        lo_err = max(0.0, pct - lo_pct)
        hi_err = max(0.0, hi_pct - pct)
        ax_a.errorbar(
            pct,
            yi,
            xerr=[[lo_err], [hi_err]],
            fmt=marker,
            color=col,
            capsize=3,
            ms=ms,
            lw=1.3,
            capthick=0.8,
            zorder=3,
        )

        if n > 0:
            txt = rf"{pct:.1f}\% ({k}/{n})"
        else:
            txt = "---"
        # flip annotation to left side for high values to prevent overflow into panel (b)
        anchor_x = max(hi_pct, pct)
        if pct > 40:
            ax_a.annotate(
                txt,
                xy=(min(lo_pct, pct), yi),
                xytext=(-6, 0),
                textcoords="offset points",
                fontsize=7,
                va="center",
                ha="right",
                color=col,
            )
        else:
            ax_a.annotate(
                txt,
                xy=(anchor_x, yi),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=7,
                va="center",
                color=col,
            )

    blended = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
    ax_a.text(
        0.55,
        y_positions[0] - 0.6,
        r"\textit{RL training seeds}",
        transform=blended,
        fontsize=7,
        ha="center",
        va="bottom",
        color="#777777",
    )
    ax_a.text(
        0.55,
        y_positions[train_count] - 0.35,
        r"\textit{Reward metric ablations}",
        transform=blended,
        fontsize=7,
        ha="center",
        va="bottom",
        color="#777777",
    )

    sep_y = (y_positions[train_count - 1] + y_positions[train_count]) / 2
    ax_a.axhline(sep_y, color="#cccccc", ls="-", lw=0.6, zorder=0)

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([e[0] for e in entries], fontsize=7.5)
    ax_a.set_xlabel(r"Non-normalized programs (\%), 95\% Wilson CI")
    ax_a.invert_yaxis()

    # symlog scale: linear 0-5%, log above — prevents 100% entries from compressing 0-25% range
    ax_a.set_xscale("symlog", linthresh=5, linscale=1.0)
    ax_a.set_xticks([0, 5, 10, 25, 50, 100])
    ax_a.set_xticklabels(["0", "5", "10", "25", "50", "100"])
    ax_a.set_xlim(-0.5, 130)
    ax_a.set_ylim(max(y_positions) + 0.8, -0.8)
    ax_a.set_axisbelow(True)
    for xt in [5, 10, 25, 50, 100]:
        ax_a.axvline(xt, color="#eeeeee", lw=0.5, zorder=0)
    ax_a.text(
        -0.02, 1.04, r"\textbf{(a)}", transform=ax_a.transAxes, fontsize=11, va="bottom", ha="left"
    )

    # ── (b) SafePyMC gate results ──
    if gate_data is None:
        ax_b.text(
            0.5,
            0.5,
            "No gate data",
            transform=ax_b.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    else:
        results = gate_data["results"]
        n_exploits = sum(1 for r in results if r["id"] != "HONEST")
        n_rejected = sum(1 for r in results if r["id"] != "HONEST" and not r["safe"])
        n_honest = sum(1 for r in results if r["id"] == "HONEST")
        n_honest_accepted = sum(1 for r in results if r["id"] == "HONEST" and r["safe"])

        categories = [
            "Exploits\nrejected",
            "Exploits\naccepted",
            "Honest\naccepted",
            "Honest\nrejected",
        ]
        counts = [
            n_rejected,
            n_exploits - n_rejected,
            n_honest_accepted,
            n_honest - n_honest_accepted,
        ]
        colors = [C_EXPLOIT, "#cccccc", C_HONEST, "#cccccc"]

        y_pos = np.arange(len(categories))
        bars = ax_b.barh(y_pos, counts, height=0.55, color=colors, edgecolor="black", linewidth=0.5)

        for bar, count in zip(bars, counts, strict=True):
            if count > 0:
                txt_kw = dict(
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="left",
                )
                if count < 3:
                    txt_kw["bbox"] = dict(
                        boxstyle="round,pad=0.15",
                        fc="white",
                        ec="none",
                        alpha=0.7,
                    )
                ax_b.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    **txt_kw,
                )

        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels(categories, fontsize=8.5)
        ax_b.set_xlabel("Programs")
        ax_b.set_xlim(0, max(counts) * 1.3)
        ax_b.invert_yaxis()

        precision = n_rejected / max(n_rejected + (n_honest - n_honest_accepted), 1)
        recall = n_rejected / max(n_exploits, 1)
        ax_b.text(
            0.95,
            0.95,
            rf"Precision: {precision * 100:.0f}\%"
            "\n"
            rf"Recall: {recall * 100:.0f}\%",
            transform=ax_b.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.5),
        )

    ax_b.text(
        -0.05, 1.04, r"\textbf{(b)}", transform=ax_b.transAxes, fontsize=11, va="bottom", ha="left"
    )

    _save(fig, out, "lh_cross_evidence")


# ── Figure S1: pm.Potential prevalence (supplementary) ───────────


def figS1_potential_prevalence(data: dict, out: Path) -> None:
    """pm.Potential trajectory + aggregate forest (seed0/seed10k only)."""
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(FIG_FULL, 2.6),
        gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.42},
    )

    bpct = BASELINE_RATE * 100
    all_sk = [s[0] for s in SEEDS_FULL]

    # ── (a) prevalence trajectory ──
    for sk, lab, col in SEEDS_FULL:
        rows = data[sk]
        steps = np.array([r["step"] for r in rows])
        pct = np.array([r["has_potential_frac"] * 100 for r in rows])
        ax_a.scatter(steps, pct, s=10, color=col, alpha=0.2, zorder=2, edgecolors="none")
        trend = _rolling(pct.tolist(), window=3)
        ax_a.plot(steps, trend, "-", color=col, lw=2.5, label=lab, zorder=3)

    ax_a.axhline(bpct, color=C_BASELINE, ls="--", lw=1.1, zorder=0)

    k_pot, n_valid = _agg(data, all_sk, "has_potential_count", "valid")
    fold = (k_pot / n_valid) / BASELINE_RATE
    ax_a.annotate(
        r"Base rate",
        xy=(22, bpct),
        xytext=(22, 1.8),
        fontsize=8,
        color=C_BASELINE,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=C_BASELINE, lw=1.0, shrinkB=1),
    )
    ax_a.text(
        0.97,
        0.58,
        rf"\textbf{{{fold:.0f}$\times$ above base rate}}"
        "\n"
        r"\textit{(all steps $>$ base)}",
        transform=ax_a.transAxes,
        fontsize=9,
        color=C_BASELINE,
        ha="right",
        va="center",
    )

    ax_a.set_xlabel("Training step")
    ax_a.set_ylabel(r"\texttt{pm.Potential} prevalence (\%)")
    ax_a.set_ylim(0, 10.5)
    ax_a.set_xlim(-0.5, 29)
    ax_a.legend(loc="upper right", fontsize=8, handlelength=1.3, handletextpad=0.5)
    ax_a.text(
        -0.13, 1.03, r"\textbf{(a)}", transform=ax_a.transAxes, fontsize=12, va="bottom", ha="left"
    )

    # ── (b) forest plot: aggregate rates ──
    slices: list[tuple[str, int, int, str, str]] = []
    for sk, lab, col in SEEDS_FULL:
        k, n = _agg(data, [sk], "has_potential_count", "valid")
        slices.append((lab, k, n, col, "o"))
    k, n = _agg(data, all_sk, "has_potential_count", "valid")
    slices.append(("Pooled", k, n, C_POOL, "D"))

    y_pos = np.arange(len(slices))
    for i, (_lab, k, n, col, marker) in enumerate(slices):
        p, lo, hi = wilson_ci(k, n)
        pct_val = p * 100
        lo_err = max(0.0, (p - lo) * 100)
        hi_err = max(0.0, (hi - p) * 100)
        ms = 7 if marker == "D" else 6
        lw = 1.8 if marker == "D" else 1.4
        ax_b.errorbar(
            pct_val,
            i,
            xerr=[[lo_err], [hi_err]],
            fmt=marker,
            color=col,
            capsize=4,
            ms=ms,
            lw=lw,
            capthick=1.0,
            zorder=3,
        )
        fold_val = pct_val / bpct
        ax_b.text(
            hi * 100 + 0.15,
            i,
            rf"  {pct_val:.1f}\% ({fold_val:.0f}$\times$)",
            fontsize=8,
            va="center",
            color=col,
            fontweight="bold",
        )

    ax_b.axvline(bpct, color=C_BASELINE, ls="--", lw=1.1, zorder=0)
    ax_b.text(bpct + 0.12, 2.4, r"base", ha="left", va="center", color=C_BASELINE, fontsize=7)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([s[0] for s in slices], fontsize=9)
    ax_b.set_xlabel(r"Prevalence (\%), 95\% Wilson CI")
    ax_b.invert_yaxis()
    max_hi = max(wilson_ci(k, n)[2] * 100 for _, k, n, _, _ in slices)
    ax_b.set_xlim(-0.2, max_hi + 4.0)
    ax_b.set_ylim(len(slices) - 0.5, -0.5)
    ax_b.text(
        -0.05, 1.03, r"\textbf{(b)}", transform=ax_b.transAxes, fontsize=12, va="bottom", ha="left"
    )

    _save(fig, out, "lh_figS1_potential_prevalence")


# ── LaTeX tables ────────────────────────────────────────────────


def print_normalization_table(data: dict) -> None:
    print("\n--- normalization table ---")
    print("% Requires: \\usepackage{booktabs}")
    print("% Optional: \\newcommand{\\code}[1]{\\texttt{#1}}")
    print()
    print(r"\begin{tabular}{@{}r l r r r r@{}}")
    print(r"\toprule")
    print(
        r"Step & Seed & $n$ & Failures & Rate"
        r" & $\overline{|\!\log Z_{\mathrm{total}}|}$ \\"
    )
    print(r"\midrule")

    rows_out = []
    prev_step = None
    for sk, lab, _ in SEEDS:
        if sk not in data:
            continue
        for r in data[sk]:
            nc = r.get("norm_checked")
            if nc is None or nc == 0:
                continue
            frac = r["norm_frac_non_normalized"]
            if frac is None:
                continue
            k = round(frac * nc)
            alm = r.get("norm_mean_abs_log_mass", 0.0) or 0.0
            rows_out.append((r["step"], lab, nc, k, frac, alm))

    for step, lab, n, k, frac, alm in sorted(rows_out):
        # midrule between step groups for readability
        if prev_step is not None and step - prev_step >= 3:
            print(r"\addlinespace")
        prev_step = step
        frac_str = f"{frac * 100:.1f}\\%"
        if k > 0:
            frac_str = rf"\textbf{{{frac_str}}}"
        print(f"  {step} & {lab} & {n} & {k}/{n} & {frac_str} & {alm:.3f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_summary_table(data: dict) -> None:
    all_sk = [s[0] for s in SEEDS_FULL]
    bpct = BASELINE_RATE * 100

    agg = {}
    for sk_key in all_sk + ["pooled"]:
        seeds = all_sk if sk_key == "pooled" else [sk_key]
        k_pot, n_valid = _agg(data, seeds, "has_potential_count", "valid")
        k_gt0, _ = _agg(data, seeds, "reward_gt0_count", "valid")
        n_steps = sum(len(data[s]) for s in seeds)
        ceiling = sum(r.get("ceiling_hits", 0) for s in seeds for r in data[s])
        rate = k_pot / n_valid if n_valid else 0
        _, lo, hi = wilson_ci(k_pot, n_valid)

        agg[sk_key] = dict(
            n_steps=n_steps,
            valid=n_valid,
            k_pot=k_pot,
            rate=rate,
            fold=rate / BASELINE_RATE,
            lo=lo,
            hi=hi,
            k_gt0=k_gt0,
            gt0_rate=k_gt0 / n_valid if n_valid else 0,
            ceiling=ceiling,
        )

    s0, s1, sp = agg["seed0"], agg["seed10k"], agg["pooled"]

    print("\n--- summary table ---")
    print("% Requires: \\usepackage{booktabs}")
    print("% Optional: \\newcommand{\\code}[1]{\\texttt{#1}}")
    print()
    print(r"\begin{tabular}{@{}l r r r r@{}}")
    print(r"\toprule")
    print(r"& Seed~0 & Seed~10k & Pooled & Base rate \\")
    print(r"\midrule")
    print(f"Training steps & {s0['n_steps']} & {s1['n_steps']} & {sp['n_steps']} & --- \\\\")
    print(f"Valid programs & {s0['valid']:,} & {s1['valid']:,} & {sp['valid']:,} & --- \\\\")
    print(r"\addlinespace")
    print(
        f"\\code{{pm.Potential}} rate "
        f"& {s0['rate'] * 100:.1f}\\% ({s0['fold']:.0f}$\\times$) "
        f"& {s1['rate'] * 100:.1f}\\% ({s1['fold']:.0f}$\\times$) "
        f"& {sp['rate'] * 100:.1f}\\% ({sp['fold']:.0f}$\\times$) "
        f"& {bpct:.1f}\\% \\\\"
    )
    print(
        f"95\\% Wilson CI "
        f"& [{s0['lo'] * 100:.1f}, {s0['hi'] * 100:.1f}]\\% "
        f"& [{s1['lo'] * 100:.1f}, {s1['hi'] * 100:.1f}]\\% "
        f"& [{sp['lo'] * 100:.1f}, {sp['hi'] * 100:.1f}]\\% "
        f"& --- \\\\"
    )
    print(r"\addlinespace")
    print(f"Ceiling hits & {s0['ceiling']} & {s1['ceiling']} & {sp['ceiling']} & --- \\\\")

    # seed30k section (norm-only; no pm.Potential data)
    if "seed30k" in data:
        s30 = data["seed30k"]
        n_steps_30 = len(s30)
        n_valid_30 = sum(r["valid"] for r in s30)
        nc = sum(r.get("norm_checked", 0) or 0 for r in s30)
        nf = sum(
            round((r.get("norm_frac_non_normalized", 0) or 0) * (r.get("norm_checked", 0) or 0))
            for r in s30
        )
        frac_30 = nf / nc if nc else 0
        print(r"\midrule")
        print(
            r"\multicolumn{5}{l}{\textit{Seed 30k (normalization only, "
            r"no \code{pm.Potential} data)}} \\"
        )
        print(
            f"\\multicolumn{{5}}{{l}}{{Steps: {n_steps_30}, "
            f"Valid: {n_valid_30:,}, "
            f"Norm failures: {nf}/{nc} = {frac_30 * 100:.1f}\\%}} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_key_numbers(data: dict) -> None:
    all_sk = ["seed0", "seed10k"]

    print("\n--- key numbers ---\n")

    for sk in all_sk:
        rows = data[sk]
        n_valid = sum(r["valid"] for r in rows)
        n_total = sum(r["total"] for r in rows)
        k_pot = sum(r["has_potential_count"] for r in rows)
        ceil = sum(r["ceiling_hits"] for r in rows)
        pot_rate = k_pot / n_valid * 100
        fold = pot_rate / (BASELINE_RATE * 100)
        _, lo, hi = wilson_ci(k_pot, n_valid)

        print(f"{sk}: {len(rows)} steps, {n_total:,} programs, {n_valid:,} valid")
        print(
            f"  pm.Potential: {k_pot}/{n_valid:,} = {pot_rate:.2f}% "
            f"({fold:.1f}x), CI=[{lo * 100:.2f}, {hi * 100:.2f}]%"
        )
        print(f"  Ceiling hits: {ceil}")

        for r in rows:
            f = r["norm_frac_non_normalized"]
            if f is not None and f > 0:
                k = round(f * r["norm_checked"])
                print(f"  Norm fail step {r['step']}: {f * 100:.1f}% ({k}/{r['norm_checked']})")

    k_pot, n_valid = _agg(data, all_sk, "has_potential_count", "valid")
    rate = k_pot / n_valid * 100
    _, lo, hi = wilson_ci(k_pot, n_valid)
    print(
        f"\nPooled: {k_pot}/{n_valid:,} = {rate:.2f}% "
        f"({rate / (BASELINE_RATE * 100):.1f}x), "
        f"CI=[{lo * 100:.2f}, {hi * 100:.2f}]%"
    )

    # seed30k section
    if "seed30k" in data:
        s30 = data["seed30k"]
        n_valid_30 = sum(r["valid"] for r in s30)
        n_total_30 = sum(r["total"] for r in s30)
        nc = sum(r.get("norm_checked", 0) or 0 for r in s30)
        nf = sum(
            round((r.get("norm_frac_non_normalized", 0) or 0) * (r.get("norm_checked", 0) or 0))
            for r in s30
        )
        print(f"\nseed30k: {len(s30)} steps, {n_total_30:,} programs, {n_valid_30:,} valid")
        frac_30 = nf / nc * 100 if nc else 0.0
        print(f"  Norm failures: {nf}/{nc} = {frac_30:.1f}%")
        for r in s30:
            f = r.get("norm_frac_non_normalized", 0) or 0
            nc_r = r.get("norm_checked", 0) or 0
            if f > 0 and nc_r > 0:
                k = round(f * nc_r)
                print(f"  Norm fail step {r['step']}: {f * 100:.1f}% ({k}/{nc_r})")


# ── CLI ─────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output dir (default: {DEFAULT_OUT.relative_to(REPO)})",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help=f"Step summaries JSON (default: {DATA.relative_to(REPO)})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = args.data if args.data else DATA
    if not data_path.is_absolute():
        data_path = REPO / data_path
    out_dir = args.out if args.out else DEFAULT_OUT
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    with open(data_path) as f:
        data = json.load(f)

    # load seed30k
    if SEED30K_DATA.exists():
        with open(SEED30K_DATA) as f:
            s30 = json.load(f)
        data["seed30k"] = s30.get("seed30k", s30)
    else:
        print(f"  warning: seed30k data not found at {SEED30K_DATA}")

    # load SafePyMC gate results
    gate_data = None
    if GATE_DATA.exists():
        with open(GATE_DATA) as f:
            gate_data = json.load(f)
    else:
        print(f"  warning: gate data not found at {GATE_DATA}")

    _setup_style()

    fig1_emergence_3panel(data, out_dir)
    fig2_cross_evidence(data, gate_data, out_dir)
    figS1_potential_prevalence(data, out_dir)
    print_normalization_table(data)
    print_summary_table(data)
    print_key_numbers(data)
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
