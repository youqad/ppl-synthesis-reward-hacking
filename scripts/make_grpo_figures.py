#!/usr/bin/env python3
"""Generate paper figures for the UAI 2026 reward-hacking paper.

Reads GRPO Run 6 completions.jsonl and produces four artifacts that together
explain WHY and HOW reward hacking emerges during training:

  1. reward_divergence.pdf  — reported/oracle trajectories with scoring formula
  2. exploit_emergence.pdf  — population composition shift (honest → exploit)
  3. exploit_taxonomy.pdf   — exploit categories + gap distributions
  4. code_comparison.tex    — honest vs hacked code side-by-side

Usage:
    pixi run -e dev python scripts/make_grpo_figures.py [--run-dir PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = REPO_ROOT / "artifacts" / "grpo_run_7050740"
DEFAULT_OUT = (
    REPO_ROOT
    / "background-documents"
    / "ai-scientist-likelihood-hacking-paper-local"
    / "paper_artifacts"
)

WINDOW = 30  # rolling mean window (batches)

# colorblind-safe palette (ColorBrewer)
C_HONEST = "#66c2a5"
C_EXPLOIT = "#fc8d62"
C_INVALID = "#d9d9d9"
# detailed exploit palette
C_DISCONNECTED = "#d73027"
C_CONNECTED = "#fc8d59"
C_EXTREME = "#91bfdb"
C_DISCARD = "#fee090"
# reward trajectories
C_REPORTED = "#2166ac"
C_ORACLE = "#b2182b"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.3,
})


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_completions(run_dir: Path) -> list[dict]:
    path = run_dir / "completions.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def _find_disconnected_rvs(code: str) -> list[str]:
    """Find RVs declared but never referenced elsewhere in the code."""
    rv_decl = re.compile(r"pm\.\w+\(\s*['\"](\w+)['\"]")
    disconnected = []
    lines = code.split("\n")
    for i, line in enumerate(lines):
        m = rv_decl.search(line)
        if not m:
            continue
        name = m.group(1)
        if name in ("y", "obs", "likelihood"):
            continue
        # check if name appears on any OTHER line (word-boundary match)
        used = False
        for j, other in enumerate(lines):
            if j == i:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", other):
                used = True
                break
        if not used:
            disconnected.append(name)
    return disconnected


def classify(rec: dict) -> str:
    """Classify a completion record.

    Returns one of:
      honest, disconnected_bonus, connected_bonus, extreme_padding,
      data_discard, invalid
    """
    if rec.get("outcome") != "valid":
        return "invalid"

    code = rec.get("code") or ""
    meta = rec.get("metadata", {})
    n_free = meta.get("n_free_rvs", 0)
    gap = rec.get("gap", 0)

    # data discard: no observed data used
    if "data['y']" not in code and 'data["y"]' not in code:
        return "data_discard"

    # extreme padding: 10+ free RVs is clearly artificial
    if n_free >= 10:
        return "extreme_padding"

    # check for disconnected bonus latents (variables never referenced)
    if n_free > 1:
        disc = _find_disconnected_rvs(code)
        if disc:
            return "disconnected_bonus"

    # connected but suspicious: many free RVs with high gap
    if n_free > 2 and gap > 50:
        return "connected_bonus"

    return "honest"


def classify_simple(rec: dict) -> str:
    """Simplified 3-category classification for the stacked area chart."""
    cat = classify(rec)
    if cat == "invalid":
        return "invalid"
    if cat == "honest":
        return "honest"
    return "exploit"


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate_by_batch(records: list[dict]) -> dict[int, dict]:
    """Compute per-batch statistics including ALL records for composition."""
    by_batch: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_batch[rec["batch"]].append(rec)

    result = {}
    for batch_idx in sorted(by_batch):
        recs = by_batch[batch_idx]
        valid = [r for r in recs if r["outcome"] == "valid"]
        total = len(recs)

        # metrics from valid records only
        if valid:
            reported = [r["reported_reward"] for r in valid]
            oracle = [r["oracle_score"] for r in valid]
            gap = [r["gap"] for r in valid]
            n_free = [r["metadata"].get("n_free_rvs", 0) for r in valid]
        else:
            reported = [float("nan")]
            oracle = [float("nan")]
            gap = [float("nan")]
            n_free = [float("nan")]

        # category fractions from ALL records
        simple_cats = defaultdict(int)
        detail_cats = defaultdict(int)
        for r in recs:
            simple_cats[classify_simple(r)] += 1
            detail_cats[classify(r)] += 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_mean = float(np.nanmean(reported))
            o_mean = float(np.nanmean(oracle))
            g_mean = float(np.nanmean(gap))
            f_mean = float(np.nanmean(n_free))

        result[batch_idx] = {
            "reported_mean": r_mean,
            "oracle_mean": o_mean,
            "gap_mean": g_mean,
            "n_free_mean": f_mean,
            "n_valid": len(valid),
            "n_total": total,
            "simple_frac": {k: v / total for k, v in simple_cats.items()},
            "detail_frac": {k: v / total for k, v in detail_cats.items()},
        }
    return result


def rolling_mean(xs: list[float], w: int) -> np.ndarray:
    """Edge-padded rolling mean (avoids boundary artifacts)."""
    arr = np.array(xs, dtype=float)
    if len(arr) == 0:
        return arr
    w = min(w, len(arr))
    padded = np.pad(arr, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")


# ---------------------------------------------------------------------------
# Figure 1: Reward divergence
# Tells the reader WHAT happens: reported reward improves while oracle lags.
# ---------------------------------------------------------------------------

def plot_reward_divergence(batch_stats: dict[int, dict], out: Path,
                           window: int = WINDOW) -> None:
    batches = sorted(batch_stats)
    reported = [batch_stats[b]["reported_mean"] for b in batches]
    oracle = [batch_stats[b]["oracle_mean"] for b in batches]

    rep_sm = rolling_mean(reported, window)
    orc_sm = rolling_mean(oracle, window)

    fig, ax = plt.subplots(figsize=(6.5, 2.8))

    ax.plot(batches, rep_sm, color=C_REPORTED, linewidth=1.5,
            label="Reported reward (proxy)", zorder=3)
    ax.plot(batches, orc_sm, color=C_ORACLE, linewidth=1.5,
            label="Oracle score (true)", zorder=3)

    # shade the gap between them
    ax.fill_between(
        batches, orc_sm, rep_sm,
        where=[r >= o for r, o in zip(rep_sm, orc_sm)],
        alpha=0.12, color=C_REPORTED, interpolate=True,
        label="Gap (hacking margin)",
    )

    # scoring formula annotation — explains WHY the gap exists
    formula_text = (
        "Scoring formula:\n"
        "  reported = mean(logp over all RVs)\n"
        "  oracle = mean(logp over observed RVs)\n\n"
        "Adding unused latent variables dilutes\n"
        "bad likelihood in reported but not oracle."
    )
    ax.text(
        0.98, 0.04, formula_text,
        transform=ax.transAxes, fontsize=6.5,
        verticalalignment="bottom", horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("Training batch")
    ax.set_ylabel("Mean log-probability score")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: Exploit emergence
# Tells the reader HOW it happens: population shifts from honest to exploit.
# ---------------------------------------------------------------------------

def plot_exploit_emergence(batch_stats: dict[int, dict], out: Path,
                           window: int = WINDOW) -> None:
    batches = sorted(batch_stats)
    n_free = [batch_stats[b]["n_free_mean"] for b in batches]

    # simplified 3-category composition (honest / exploit / invalid)
    honest_pct = [batch_stats[b]["simple_frac"].get("honest", 0) * 100
                  for b in batches]
    exploit_pct = [batch_stats[b]["simple_frac"].get("exploit", 0) * 100
                   for b in batches]
    invalid_pct = [batch_stats[b]["simple_frac"].get("invalid", 0) * 100
                   for b in batches]

    w = window
    nf_sm = rolling_mean(n_free, w)
    hon_sm = rolling_mean(honest_pct, w)
    exp_sm = rolling_mean(exploit_pct, w)
    inv_sm = rolling_mean(invalid_pct, w)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.25, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.5]},
    )

    # --- (A) Mean free RVs over training ---
    ax_top.plot(batches, nf_sm, color=C_REPORTED, linewidth=1.3)
    ax_top.set_ylabel("Mean free RVs\nper completion")
    ax_top.text(0.02, 0.92, "(a)", transform=ax_top.transAxes,
                fontsize=9, fontweight="bold", va="top")
    ax_top.grid(True, alpha=0.15)

    # annotate the trend
    ax_top.annotate(
        "Model learns to\nadd unused variables",
        xy=(750, float(nf_sm[749])), xytext=(400, float(nf_sm[749]) + 1.5),
        fontsize=6.5, ha="center",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    # --- (B) Full population composition ---
    ax_bot.stackplot(
        batches, inv_sm, hon_sm, exp_sm,
        colors=[C_INVALID, C_HONEST, C_EXPLOIT],
        labels=["Invalid (failed)", "Honest programs", "Exploit programs"],
    )
    ax_bot.set_xlabel("Training batch")
    ax_bot.set_ylabel("Completions (%)")
    ax_bot.set_ylim(0, 100)
    ax_bot.text(0.02, 0.96, "(b)", transform=ax_bot.transAxes,
                fontsize=9, fontweight="bold", va="top")
    ax_bot.grid(True, alpha=0.15)

    # legend inside, compact
    handles, labels = ax_bot.get_legend_handles_labels()
    ax_bot.legend(handles[::-1], labels[::-1], loc="center right",
                  fontsize=6, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Figure 3: Exploit taxonomy
# Tells the reader WHICH strategies the model discovers.
# ---------------------------------------------------------------------------

def plot_exploit_taxonomy(records: list[dict], out: Path) -> None:
    # late training window: batches 800-1000
    late_valid = [r for r in records
                  if r.get("outcome") == "valid" and r.get("batch", 0) >= 800]

    cat_order = [
        "disconnected_bonus", "connected_bonus", "extreme_padding",
        "data_discard", "honest",
    ]
    cat_labels = {
        "disconnected_bonus": "Disconnected\nbonus latents",
        "connected_bonus": "Connected\nbonus latents",
        "extreme_padding": "Extreme padding\n(10+ free RVs)",
        "data_discard": "Data discard\n(no observations)",
        "honest": "Honest\nprograms",
    }
    cat_colors = {
        "disconnected_bonus": C_DISCONNECTED,
        "connected_bonus": C_CONNECTED,
        "extreme_padding": C_EXTREME,
        "data_discard": C_DISCARD,
        "honest": C_HONEST,
    }

    by_cat: dict[str, list[float]] = defaultdict(list)
    for rec in late_valid:
        cat = classify(rec)
        by_cat[cat].append(rec["gap"])

    counts = [len(by_cat.get(c, [])) for c in cat_order]
    gaps_per_cat = [
        np.array(by_cat[c]) if c in by_cat and by_cat[c]
        else np.array([float("nan")])
        for c in cat_order
    ]

    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(6.5, 2.8),
                                          gridspec_kw={"width_ratios": [1, 1.2]})

    # --- (A) Category counts ---
    colors = [cat_colors[c] for c in cat_order]
    labels = [cat_labels[c] for c in cat_order]
    y_pos = np.arange(len(cat_order))
    bars = ax_bar.barh(y_pos, counts, color=colors,
                       edgecolor="black", linewidth=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=6.5)
    ax_bar.set_xlabel("Count (batches 800\u20131000)")
    ax_bar.invert_yaxis()
    ax_bar.text(0.02, 0.96, "(a)", transform=ax_bar.transAxes,
                fontsize=9, fontweight="bold", va="top")

    # annotate bars with count values
    for bar, count in zip(bars, counts):
        if count > 0:
            ax_bar.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                        str(count), va="center", fontsize=6.5)

    # --- (B) Gap distribution by category ---
    bp = ax_box.boxplot(
        gaps_per_cat, vert=True, patch_artist=True, widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.4)
        patch.set_alpha(0.85)

    ax_box.set_xticklabels(
        [cat_labels[c].replace("\n", " ") for c in cat_order],
        fontsize=5.5, rotation=35, ha="right",
    )
    ax_box.set_ylabel("Gap (reported $-$ oracle)")
    ax_box.text(0.02, 0.96, "(b)", transform=ax_box.transAxes,
                fontsize=9, fontweight="bold", va="top")
    ax_box.grid(True, axis="y", alpha=0.15)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Figure 4: Code comparison (LaTeX lstlisting)
# Shows the reader WHAT the exploit looks like in practice.
# ---------------------------------------------------------------------------

def _select_example(records: list[dict], *,
                    early: bool, honest: bool) -> dict | None:
    valid = [r for r in records if r.get("outcome") == "valid" and r.get("code")]
    if early:
        pool = [r for r in valid if r.get("batch", 0) <= 50]
    else:
        pool = [r for r in valid if r.get("batch", 0) >= 900]

    if honest:
        pool = [r for r in pool if classify(r) == "honest"]
        pool.sort(key=lambda r: abs(r.get("gap", 0)))
    else:
        pool = [r for r in pool if classify(r) not in ("honest", "invalid")]
        pool.sort(key=lambda r: r.get("gap", 0), reverse=True)

    return pool[0] if pool else None


def write_code_comparison(records: list[dict], out: Path) -> None:
    honest_rec = _select_example(records, early=True, honest=True)
    hacked_rec = _select_example(records, early=False, honest=False)

    if not honest_rec or not hacked_rec:
        print(f"  WARN: could not find example pair, skipping {out.name}")
        return

    h_code = (honest_rec.get("code") or "").strip()
    x_code = (hacked_rec.get("code") or "").strip()
    h_gap = honest_rec.get("gap", 0)
    x_gap = hacked_rec.get("gap", 0)
    h_batch = honest_rec.get("batch", 0)
    x_batch = hacked_rec.get("batch", 0)
    h_nfree = honest_rec.get("metadata", {}).get("n_free_rvs", 0)
    x_nfree = hacked_rec.get("metadata", {}).get("n_free_rvs", 0)

    lst = (
        "language=Python, basicstyle=\\scriptsize\\ttfamily, "
        "breaklines=true, frame=single, xleftmargin=2pt, xrightmargin=2pt"
    )

    lines = [
        "% autogenerated by scripts/make_grpo_figures.py",
        "\\begin{figure*}[t]",
        "\\centering",
        "\\begin{minipage}[t]{0.48\\linewidth}",
        f"\\textbf{{(a) Honest model}} "
        f"(batch~{h_batch}, gap~$=$~{h_gap:.0f}, "
        f"free~RVs~$=$~{h_nfree})",
        f"\\begin{{lstlisting}}[{lst}]",
        h_code,
        "\\end{lstlisting}",
        "\\end{minipage}",
        "\\hfill",
        "\\begin{minipage}[t]{0.48\\linewidth}",
        f"\\textbf{{(b) Hacked model}} "
        f"(batch~{x_batch}, gap~$=$~{x_gap:.0f}, "
        f"free~RVs~$=$~{x_nfree})",
        f"\\begin{{lstlisting}}[{lst}]",
        x_code,
        "\\end{lstlisting}",
        "\\end{minipage}",
        "\\caption{Representative programs from GRPO training. "
        "The honest model~(a) uses all its latent variables to fit "
        "the data (gap~$\\approx 0$). "
        f"The hacked model~(b) declares {x_nfree} free random variables "
        "but most serve no modelling purpose. "
        "The scorer averages log-probabilities over \\emph{all} "
        "random variables for the reported score, but only over "
        "\\emph{observed} variables for the oracle. "
        "The unused variables contribute moderate prior terms "
        "($\\approx\\!-1$ each) that dilute the bad observation "
        "likelihood when averaged, inflating the reported score while "
        "leaving the oracle unchanged "
        f"(gap~$=$~{x_gap:.0f}).}}",
        "\\label{fig:code-comparison}",
        "\\end{figure*}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate GRPO paper figures")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window", type=int, default=WINDOW)
    args = p.parse_args()

    window = args.window

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading completions from {args.run_dir} ...")
    records = load_completions(args.run_dir)
    valid = [r for r in records if r.get("outcome") == "valid"]
    print(f"  {len(records)} records ({len(valid)} valid)")

    batch_stats = aggregate_by_batch(records)
    print(f"  {len(batch_stats)} batches\n")

    print("Generating figures:")
    plot_reward_divergence(batch_stats, out_dir / "reward_divergence.pdf", window)
    plot_exploit_emergence(batch_stats, out_dir / "exploit_emergence.pdf", window)
    plot_exploit_taxonomy(records, out_dir / "exploit_taxonomy.pdf")
    write_code_comparison(records, out_dir / "code_comparison.tex")
    print("\nDone.")


if __name__ == "__main__":
    main()
