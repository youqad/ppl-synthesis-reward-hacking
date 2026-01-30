from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("results", help="Aggregate and report results")
    res_sub = parser.add_subparsers(dest="results_command")

    aggregate = res_sub.add_parser("aggregate", help="Aggregate run artifacts")
    aggregate.add_argument("--runs", required=True, help="Path to runs directory")
    aggregate.add_argument("--out", required=True, help="Output csv/tsv path")
    aggregate.set_defaults(func=_run_aggregate)


def _run_aggregate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ppl_synthesis_reward_hacking.evaluation.aggregators import aggregate_runs

    aggregate_runs(Path(args.runs), Path(args.out))
    print(f"wrote {args.out}")
    return 0
