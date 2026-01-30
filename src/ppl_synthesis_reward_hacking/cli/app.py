from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import commands_aggregate, commands_data, commands_plot, commands_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psrh",
        description="PPL Synthesis Reward Hacking experiment runner",
    )
    subparsers = parser.add_subparsers(dest="command")

    commands_data.register(subparsers)
    commands_run.register(subparsers)
    commands_aggregate.register(subparsers)
    commands_plot.register(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 2

    return args.func(args)
