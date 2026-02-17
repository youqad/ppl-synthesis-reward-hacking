from __future__ import annotations

from typing import Any

_SERIF_STACK = [
    "Times New Roman",
    "Times",
    "Liberation Serif",
    "serif",
]


def _build_rc(
    *,
    font_size: float,
    axes_labelsize: float,
    axes_titlesize: float,
    xtick_labelsize: float,
    ytick_labelsize: float,
    legend_fontsize: float,
    lines_linewidth: float,
) -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": _SERIF_STACK,
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": axes_labelsize,
        "axes.titlesize": axes_titlesize,
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": ytick_labelsize,
        "legend.fontsize": legend_fontsize,
        "lines.linewidth": lines_linewidth,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
    }


def apply_publication_style(
    *,
    font_size: float = 9,
    axes_labelsize: float = 9,
    axes_titlesize: float = 10,
    xtick_labelsize: float = 8,
    ytick_labelsize: float = 8,
    legend_fontsize: float = 8,
    lines_linewidth: float = 1.5,
) -> None:
    """Apply a consistent publication style across all plots.

    Prefers seaborn's whitegrid theme when available; otherwise falls back
    to matplotlib rcParams with equivalent settings.
    """
    rc = _build_rc(
        font_size=font_size,
        axes_labelsize=axes_labelsize,
        axes_titlesize=axes_titlesize,
        xtick_labelsize=xtick_labelsize,
        ytick_labelsize=ytick_labelsize,
        legend_fontsize=legend_fontsize,
        lines_linewidth=lines_linewidth,
    )

    try:
        import seaborn as sns
    except Exception:
        import matplotlib.pyplot as plt

        plt.rcParams.update(rc)
        return

    sns.set_theme(style="whitegrid", font="serif", rc=rc)
