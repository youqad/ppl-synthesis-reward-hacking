from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("plot", help="Generate plots")
    parser.add_argument("--agg", required=True, help="Path to aggregate parquet/csv")
    parser.add_argument("--config", required=True, help="Plot config YAML")
    parser.set_defaults(func=_run_plot)


def _run_plot(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ppl_synthesis_reward_hacking.plotting.plotters import plot_from_aggregate

    plot_from_aggregate(Path(args.agg), config_path=Path(args.config))
    print(f"plotted using {args.agg}")
    return 0
