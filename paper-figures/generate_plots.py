from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parent
DEFAULT_CI_LEVEL = 0.80

# Palette matched to the active Figure 1 source: paper/figures/fig1.tex,
# included from paper/main.tex via \includestandalone{figures/fig1}.
# Figure 1 defines boxbluefill=(0.9, 0.9, 1.0) and
# boxgreenfill=(0.9, 0.85, 0.75); these darker tones keep plots legible.
PURPLE = "#4F4A84"
PURPLE_LIGHT = "#E6E6FF"
GOLD = "#B3863B"
GRID = "#E9E7F4"
TEXT = "#292432"


@dataclass(frozen=True)
class FigureSpec:
    csv_name: str
    output_name: str
    title: str
    y_label: str
    kind: str
    percent_axis: bool = False


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        csv_name="fig-lh-rate.csv",
        output_name="lh-rate.png",
        title="Likelihood hacking rate",
        y_label="Likelihood hacking rate",
        kind="scatter",
        percent_axis=True,
    ),
    FigureSpec(
        csv_name="fig-reward-lift.csv",
        output_name="lh-reward-lift.png",
        title="Likelihood hacking reward lift",
        y_label="Reward lift",
        kind="scatter",
    ),
    FigureSpec(
        csv_name="fig-valid-programs.csv",
        output_name="valid-programs-rate.png",
        title="Valid programs rate",
        y_label="Valid programs",
        kind="line",
        percent_axis=True,
    ),
    FigureSpec(
        csv_name="fig-reward-mean.csv",
        output_name="reward-mean.png",
        title="Mean reward",
        y_label="Mean reward",
        kind="line",
    ),
    FigureSpec(
        csv_name="fig-ratio-unique-programs.csv",
        output_name="ratio-unique-programs.png",
        title="Unique valid programs",
        y_label="Unique program ratio",
        kind="line",
        percent_axis=True,
    ),
)


@dataclass(frozen=True)
class Series:
    x: list[float]
    y: list[float]
    metric_column: str
    skipped_nonfinite: int = 0


@dataclass(frozen=True)
class Regression:
    slope: float
    intercept: float
    slope_ci_low: float
    slope_ci_high: float


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def train_metric_column(fieldnames: Iterable[str]) -> str:
    candidates = [
        field
        for field in fieldnames
        if " - train/" in field
        and not field.endswith("__MIN")
        and not field.endswith("__MAX")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one train metric column, found {candidates!r}")
    return candidates[0]


def read_series(path: Path, x_column: str) -> Series:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        if x_column not in reader.fieldnames:
            raise ValueError(f"{path} does not contain x column {x_column!r}")

        metric_column = train_metric_column(reader.fieldnames)
        x: list[float] = []
        y: list[float] = []
        skipped_nonfinite = 0
        for row_number, row in enumerate(reader, start=2):
            try:
                x_value = float(row[x_column])
                y_value = float(row[metric_column])
            except ValueError as exc:
                raise ValueError(f"Could not parse numeric data in {path}:{row_number}") from exc
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                skipped_nonfinite += 1
                continue
            x.append(x_value)
            y.append(y_value)

    if not x:
        raise ValueError(f"{path} contains no data rows")
    return Series(x=x, y=y, metric_column=metric_column, skipped_nonfinite=skipped_nonfinite)


def parse_confidence_level(raw_value: str) -> float:
    text = raw_value.strip()
    try:
        if text.endswith("%"):
            value = float(text[:-1]) / 100
        else:
            value = float(text)
            if value > 1:
                value /= 100
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "confidence level must be a number such as 0.8, 80, or 80%"
        ) from exc

    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("confidence level must be between 0 and 1")
    return value


def format_confidence_level(value: float) -> str:
    return f"{value * 100:g}%"


def t_critical(df: int, confidence_level: float) -> float:
    if df <= 0:
        raise ValueError("A positive number of degrees of freedom is required")

    p = 0.5 + confidence_level / 2
    z = NormalDist().inv_cdf(p)
    v = float(df)
    return (
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )


def linear_regression(
    x: list[float],
    y: list[float],
    confidence_level: float,
) -> Regression:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("At least three points are required for regression intervals")

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    sxx = sum((value - x_mean) ** 2 for value in x)
    if sxx == 0:
        raise ValueError("Cannot fit a regression line when all x values are equal")

    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / sxx
    intercept = y_mean - slope * x_mean
    residual_sse = sum(
        (yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y)
    )
    degrees_of_freedom = len(x) - 2
    slope_se = math.sqrt((residual_sse / degrees_of_freedom) / sxx)
    margin = t_critical(degrees_of_freedom, confidence_level) * slope_se
    return Regression(
        slope=slope,
        intercept=intercept,
        slope_ci_low=slope - margin,
        slope_ci_high=slope + margin,
    )


def apply_common_style(ax: plt.Axes, spec: FigureSpec, x_label: str) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(spec.y_label)
    ax.set_facecolor("#FDFBFF")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7C7488")
    ax.spines["bottom"].set_color("#7C7488")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.margins(x=0.02, y=0.08)

    if spec.percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        bottom, top = ax.get_ylim()
        ax.set_ylim(max(0.0, bottom), min(1.02, top))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_scatter_with_fit(
    ax: plt.Axes,
    series: Series,
    confidence_level: float,
) -> Regression:
    regression = linear_regression(series.x, series.y, confidence_level)
    x_min = min(series.x)
    x_max = max(series.x)
    fit_x = [x_min, x_max]
    fit_y = [regression.intercept + regression.slope * value for value in fit_x]

    ax.plot(
        series.x,
        series.y,
        color=PURPLE,
        linewidth=0.9,
        alpha=0.16,
        solid_joinstyle="round",
        solid_capstyle="round",
        label="_nolegend_",
        zorder=2,
    )
    ax.scatter(
        series.x,
        series.y,
        s=20,
        color=PURPLE,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.78,
        label="Observed",
        zorder=3,
    )
    ax.plot(
        fit_x,
        fit_y,
        color=GOLD,
        linewidth=2.0,
        label="Linear fit",
        zorder=4,
    )
    legend = ax.legend(loc="best", frameon=True, handlelength=1.8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(PURPLE_LIGHT)
    legend.get_frame().set_linewidth(0.8)
    return regression


def plot_line(ax: plt.Axes, series: Series) -> None:
    ax.plot(
        series.x,
        series.y,
        color=PURPLE,
        linewidth=1.8,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=3,
    )


def save_figure(
    spec: FigureSpec,
    series: Series,
    output_dir: Path,
    x_label: str,
    confidence_level: float,
) -> Regression | None:
    fig, ax = plt.subplots(figsize=(5.4, 3.35), constrained_layout=True)
    fig.patch.set_facecolor("white")

    regression: Regression | None = None
    if spec.kind == "scatter":
        regression = plot_scatter_with_fit(ax, series, confidence_level)
    elif spec.kind == "line":
        plot_line(ax, series)
    else:
        raise ValueError(f"Unsupported figure kind: {spec.kind}")

    apply_common_style(ax, spec, x_label=x_label)
    output_path = output_dir / spec.output_name
    fig.savefig(output_path)
    plt.close(fig)
    return regression


def slope_units(spec: FigureSpec, x_label: str) -> str:
    if spec.percent_axis:
        return f"percentage points per {x_label.lower()}"
    return f"{spec.y_label.lower()} units per {x_label.lower()}"


def scale_regression_value(value: float, spec: FigureSpec) -> float:
    if spec.percent_axis:
        return value * 100
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate paper appendix figures from raw W&B CSV exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT,
        help="Directory containing fig-*.csv inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper" / "appendix-figures",
        help="Directory where appendix PNG figures are written.",
    )
    parser.add_argument(
        "--x-column",
        default="train/global_step",
        help="CSV column used for the x axis.",
    )
    parser.add_argument(
        "--x-label",
        default="Training step",
        help="Human-readable x-axis label.",
    )
    parser.add_argument(
        "--ci-level",
        type=parse_confidence_level,
        default=DEFAULT_CI_LEVEL,
        help="Two-sided slope confidence level; accepts values like 0.8, 80, or 80%.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in FIGURES:
        series = read_series(input_dir / spec.csv_name, args.x_column)
        if series.skipped_nonfinite:
            print(
                f"{spec.csv_name}: skipped {series.skipped_nonfinite} "
                "non-finite row(s)"
            )
        regression = save_figure(
            spec=spec,
            series=series,
            output_dir=output_dir,
            x_label=args.x_label,
            confidence_level=args.ci_level,
        )
        if regression is None:
            print(f"Wrote {output_dir / spec.output_name}")
            continue

        slope = scale_regression_value(regression.slope, spec)
        ci_low = scale_regression_value(regression.slope_ci_low, spec)
        ci_high = scale_regression_value(regression.slope_ci_high, spec)
        print(
            f"{spec.title}: slope = {slope:.4g} "
            f"({format_confidence_level(args.ci_level)} CI "
            f"[{ci_low:.4g}, {ci_high:.4g}]) "
            f"{slope_units(spec, args.x_label)}"
        )
        print(f"Wrote {output_dir / spec.output_name}")


if __name__ == "__main__":
    main()
