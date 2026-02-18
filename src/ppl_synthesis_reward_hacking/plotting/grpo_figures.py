from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ppl_synthesis_reward_hacking.evaluation.aggregation import rolling_mean

WINDOW = 30

# palette: colorblind-safe (ColorBrewer Set3 + custom)
C_HONEST = "#66c2a5"
C_INVALID = "#d9d9d9"
C_MULTI_LH = "#8c510a"
C_REPORTED = "#2166ac"
C_ORACLE = "#b2182b"
C_NORM_FRAC = "#e41a1c"
C_LOG_MASS = "#984ea3"

# per-family colors (10 families + evaluator_hack + multi_lh)
FAMILY_COLORS = {
    "lh_potential_constant_offset": "#d73027",
    "lh_potential_tempered_likelihood": "#f46d43",
    "lh_double_observation_same_data": "#fdae61",
    "lh_data_dependent_score_injection": "#fee08b",
    "lh_observed_constant_anchor": "#e6f598",
    "lh_custom_logp_missing_normalizer": "#abdda4",
    "lh_improper_mixture_non_normalized_mass": "#66c2a5",
    "lh_cross_validation_reuse_overlap": "#3288bd",
    "lh_pseudocount_fixed_observation": "#5e4fa2",
    "lh_pure_score_injection_baseline": "#b2182b",
    "lh_sufficient_statistic_observation": "#91bfdb",
    "lh_unnormalized_measure_density_observe": "#4393c3",
    "lh_data_discard": "#fee090",
    "evaluator_hack_initval": "#c7c7c7",
    "multi_lh": C_MULTI_LH,
    "honest": C_HONEST,
    "invalid": C_INVALID,
}

FAMILY_LABELS = {
    "lh_potential_constant_offset": "Constant offset",
    "lh_potential_tempered_likelihood": "Tempered likelihood",
    "lh_double_observation_same_data": "Double observation",
    "lh_data_dependent_score_injection": "Data-dependent injection",
    "lh_observed_constant_anchor": "Observed constant",
    "lh_custom_logp_missing_normalizer": "Custom logp (no normalizer)",
    "lh_improper_mixture_non_normalized_mass": "Improper mixture",
    "lh_cross_validation_reuse_overlap": "Cross-validation reuse",
    "lh_pseudocount_fixed_observation": "Pseudocount",
    "lh_pure_score_injection_baseline": "Score injection",
    "lh_sufficient_statistic_observation": "Sufficient statistic",
    "lh_unnormalized_measure_density_observe": "Unnorm. density",
    "lh_data_discard": "Data discard",
    "evaluator_hack_initval": "Evaluator hack (initval)",
    "multi_lh": "Multiple LH",
    "honest": "Honest",
    "invalid": "Invalid",
}


def plot_lh_emergence(batch_stats: dict[int, dict], out: Path, window: int = WINDOW) -> None:
    batches = sorted(batch_stats)
    reported = [batch_stats[b]["reported_mean"] for b in batches]

    has_norm = any(batch_stats[b].get("frac_non_normalized") is not None for b in batches)

    if has_norm:
        frac_nn = [batch_stats[b].get("frac_non_normalized", 0) or 0 for b in batches]
        abs_lm = [batch_stats[b].get("mean_abs_log_mass", 0) or 0 for b in batches]
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(6.5, 4.5),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1]},
        )

        fnn_sm = rolling_mean(frac_nn, window)
        ax_top.plot(batches, fnn_sm, color=C_NORM_FRAC, linewidth=1.5)
        ax_top.set_ylabel("Fraction non-normalized")
        ax_top.text(
            0.02, 0.92, "(a)", transform=ax_top.transAxes, fontsize=9, fontweight="bold", va="top"
        )
        sns.despine(ax=ax_top, left=False, bottom=False)

        alm_sm = rolling_mean(abs_lm, window)
        ax_bot.plot(batches, alm_sm, color=C_LOG_MASS, linewidth=1.5)
        ax_bot.set_xlabel("Training batch")
        ax_bot.set_ylabel("Mean |log mass|")
        ax_bot.text(
            0.02, 0.92, "(b)", transform=ax_bot.transAxes, fontsize=9, fontweight="bold", va="top"
        )
        sns.despine(ax=ax_bot, left=False, bottom=False)
    else:
        fig, ax_top = plt.subplots(figsize=(6.5, 2.8))

        rep_sm = rolling_mean(reported, window)
        ax_top.plot(
            batches,
            rep_sm,
            color=C_REPORTED,
            linewidth=1.5,
            label="Reported reward",
            zorder=3,
        )
        ax_top.set_xlabel("Training batch")
        ax_top.set_ylabel("Log-probability score")
        ax_top.legend(loc="upper left", fontsize=7)
        sns.despine(ax=ax_top, left=False, bottom=False)
        print("  WARN: no normalization data; using reported reward only")

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_lh_taxonomy_evolution(
    batch_stats: dict[int, dict], out: Path, window: int = WINDOW
) -> None:
    batches = sorted(batch_stats)

    all_cats: set[str] = set()
    for b in batches:
        all_cats.update(batch_stats[b]["detail_frac"].keys())

    lh_families = sorted(c for c in all_cats if c.startswith("lh_"))
    eval_families = sorted(c for c in all_cats if c.startswith("evaluator_"))
    cat_order: list[str] = []
    if "invalid" in all_cats:
        cat_order.append("invalid")
    if "honest" in all_cats:
        cat_order.append("honest")
    cat_order.extend(lh_families)
    cat_order.extend(eval_families)
    if "multi_lh" in all_cats:
        cat_order.append("multi_lh")

    if not cat_order:
        print(f"  WARN: no categories found, skipping {out.name}")
        return

    series = {}
    for cat in cat_order:
        raw = [batch_stats[b]["detail_frac"].get(cat, 0) * 100 for b in batches]
        series[cat] = rolling_mean(raw, window)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    colors = [FAMILY_COLORS.get(c, "#888888") for c in cat_order]
    labels = [FAMILY_LABELS.get(c, c) for c in cat_order]
    stacks = [series[c] for c in cat_order]

    ax.stackplot(batches, *stacks, colors=colors, labels=labels)
    ax.set_xlabel("Training batch")
    ax.set_ylabel("Completions (%)")
    ax.set_ylim(0, 100)
    sns.despine(ax=ax, left=False, bottom=False)

    handles, leg_labels = ax.get_legend_handles_labels()
    visible = [
        (handle, label)
        for handle, label, c in zip(handles, leg_labels, cat_order, strict=True)
        if any(batch_stats[b]["detail_frac"].get(c, 0) > 0 for b in batches)
    ]
    if visible:
        h_vis, l_vis = zip(*visible, strict=True)
        ax.legend(
            h_vis, l_vis, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, framealpha=0.9
        )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def plot_lh_taxonomy_bar(
    records: list[dict],
    out: Path,
    *,
    classify_fn: Callable[[dict], str],
    late_batch_cutoff: int = 800,
) -> None:
    """Bar chart of LH family counts + log_mass/gap distribution.

    Parameters
    ----------
    classify_fn:
        Maps a record to a detailed taxonomy family ID.
    """
    late_valid = [
        r for r in records if r.get("outcome") == "valid" and r.get("batch", 0) >= late_batch_cutoff
    ]

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for rec in late_valid:
        by_cat[classify_fn(rec)].append(rec)

    cat_order = sorted(
        [c for c in by_cat if c != "honest"],
        key=lambda c: len(by_cat[c]),
        reverse=True,
    )
    if "honest" in by_cat:
        cat_order.append("honest")

    if not cat_order:
        print(f"  WARN: no late-training records, skipping {out.name}")
        return

    counts = [len(by_cat[c]) for c in cat_order]

    has_log_mass = any(
        (r.get("metadata") or {}).get("normalization", {}).get("log_mass") is not None
        for recs in by_cat.values()
        for r in recs
    )

    if has_log_mass:
        log_masses_per_cat = []
        for c in cat_order:
            lms = []
            for r in by_cat[c]:
                lm = (r.get("metadata") or {}).get("normalization", {}).get("log_mass")
                if lm is not None:
                    lms.append(float(lm))
            log_masses_per_cat.append(np.array(lms) if lms else np.array([float("nan")]))
    else:
        log_masses_per_cat = [
            (
                np.array([r["reported_reward"] for r in by_cat[c]])
                if by_cat[c]
                else np.array([float("nan")])
            )
            for c in cat_order
        ]

    fig, (ax_bar, ax_box) = plt.subplots(
        1,
        2,
        figsize=(6.5, max(2.8, 0.35 * len(cat_order))),
        gridspec_kw={"width_ratios": [1, 1.2]},
    )

    colors = [FAMILY_COLORS.get(c, "#888888") for c in cat_order]
    labels = [FAMILY_LABELS.get(c, c) for c in cat_order]
    y_pos = np.arange(len(cat_order))

    bars = ax_bar.barh(y_pos, counts, color=colors, edgecolor="black", linewidth=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=6.5)
    ax_bar.set_xlabel(f"Count (batches {late_batch_cutoff}+)")
    ax_bar.invert_yaxis()
    ax_bar.text(
        0.02, 0.96, "(a)", transform=ax_bar.transAxes, fontsize=9, fontweight="bold", va="top"
    )
    for bar, count in zip(bars, counts, strict=True):
        if count > 0:
            ax_bar.text(
                bar.get_width() + max(1, max(counts) * 0.02),
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=6.5,
            )

    bp = ax_box.boxplot(
        log_masses_per_cat,
        vert=True,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1),
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.4)
        patch.set_alpha(0.85)

    short_labels = [label.replace("\n", " ")[:20] for label in labels]
    ax_box.set_xticklabels(short_labels, fontsize=5.5, rotation=35, ha="right")
    y_label = "log mass" if has_log_mass else "Reported reward"
    ax_box.set_ylabel(y_label)
    ax_box.text(
        0.02, 0.96, "(b)", transform=ax_box.transAxes, fontsize=9, fontweight="bold", va="top"
    )
    sns.despine(ax=ax_box, left=False, bottom=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")
