"""Emergence analysis plots for reward hacking during GRPO training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.evaluation.emergence import HackStats
from ppl_synthesis_reward_hacking.plotting.styles import apply_publication_style

log = logging.getLogger(__name__)


def _plot_reported_trajectory(ax, steps, reported, baseline_stats):
    ax.plot(steps, reported, "b-", linewidth=2, label="Reported reward")
    if baseline_stats and not np.isnan(baseline_stats.reported_mean):
        ax.axhline(baseline_stats.reported_mean, color="gray", linestyle="--", label="Baseline")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reported Reward")
    ax.set_title("Reported Reward Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_hack_rates(ax, steps, rates, baseline_stats):
    ax.plot(steps, rates["data_discard"], "r-", linewidth=2, label="Data Discard")
    ax.plot(steps, rates["score_bonus"], "g-", linewidth=2, label="Score Bonus")
    ax.plot(
        steps,
        rates["latent_inflation"],
        "m-",
        linewidth=2,
        label="Latent inflation suspect (complexity)",
    )
    ax.plot(steps, rates["any_exploit"], "k-", linewidth=2, label="Any Exploit", alpha=0.5)
    if baseline_stats:
        ax.axhline(
            baseline_stats.any_exploit_rate * 100,
            color="gray",
            linestyle="--",
            label="Baseline (Any exploit)",
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Hack Prevalence (%)")
    ax.set_title("Hacking Pattern Emergence")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_reward_vs_hack(ax, reported, any_exploit, summary):
    valid = [(r, h) for r, h in zip(reported, any_exploit, strict=False) if not np.isnan(r)]
    if valid:
        plot_rewards, plot_hacks = zip(*valid, strict=False)
        ax.scatter(plot_rewards, plot_hacks, alpha=0.5, c=range(len(plot_rewards)), cmap="viridis")
    ax.set_xlabel("Reported Reward")
    ax.set_ylabel("Any Hack Rate (%)")
    ax.set_title("Reported Reward vs Hack Rate")
    ax.grid(True, alpha=0.3)


def _plot_summary_text(ax, timeline, baseline_stats, summary):
    ax.axis("off")
    lines = ["EMERGENCE ANALYSIS SUMMARY", "=" * 30, ""]
    if baseline_stats:
        lines.append("Baseline (pre-training):")
        lines.append(f"  Any exploit: {baseline_stats.any_exploit_rate * 100:.1f}%")
        lines.append("")
    lines.append(f"Training ({len(timeline)} steps):")
    lines.append("")
    lines.append("Hack rate changes:")
    lines.append(
        f"  Data discard: {summary['initial']['data_discard_rate'] * 100:.1f}% -> "
        f"{summary['final']['data_discard_rate'] * 100:.1f}%"
    )
    lines.append(
        f"  Score bonus: {summary['initial']['score_bonus_rate'] * 100:.1f}% -> "
        f"{summary['final']['score_bonus_rate'] * 100:.1f}%"
    )
    lines.append(
        f"  Any exploit: {summary['initial']['any_exploit_rate'] * 100:.1f}% -> "
        f"{summary['final']['any_exploit_rate'] * 100:.1f}%"
    )
    li = summary["initial"]["latent_inflation_suspect_rate"] * 100
    lf = summary["final"]["latent_inflation_suspect_rate"] * 100
    lines.append(f"  Latent inflation suspect: {li:.1f}% -> {lf:.1f}%")
    lines.append("")
    if "natural_emergence_evidence" in summary:
        ev = summary["natural_emergence_evidence"]
        lines.append("Natural Emergence Evidence:")
        lines.append(f"  Baseline clean (<5%): {'YES' if ev['baseline_clean'] else 'NO'}")
        emerged = "YES" if ev["exploits_emerged"] else "NO"
        lines.append(f"  Exploits emerged (>10% increase): {emerged}")
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )


def plot_emergence(
    timeline: list[tuple[int, HackStats]],
    baseline_stats: HackStats | None,
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping plot")
        return

    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    steps = [s for s, _ in timeline]
    reported = [stats.reported_mean for _, stats in timeline]
    rates = {
        "data_discard": [stats.data_discard_rate * 100 for _, stats in timeline],
        "score_bonus": [stats.score_bonus_rate * 100 for _, stats in timeline],
        "latent_inflation": [stats.latent_inflation_suspect_rate * 100 for _, stats in timeline],
        "any_exploit": [stats.any_exploit_rate * 100 for _, stats in timeline],
    }

    _plot_reported_trajectory(axes[0, 0], steps, reported, baseline_stats)
    _plot_hack_rates(axes[0, 1], steps, rates, baseline_stats)
    _plot_reward_vs_hack(axes[1, 0], reported, rates["any_exploit"], summary)
    _plot_summary_text(axes[1, 1], timeline, baseline_stats, summary)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
