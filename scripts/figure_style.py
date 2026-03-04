"""Shared matplotlib style for paper figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def setup_paper_style(*, line_width: float | None = None) -> None:
    params = {
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
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
    if line_width is not None:
        params["lines.linewidth"] = line_width
    plt.rcParams.update(params)
