#!/usr/bin/env python3
"""CLI: validate run logging completeness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppl_synthesis_reward_hacking.logging.run_logging_validation import validate_run_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate run logging completeness")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing completions/results/hydra_resolved_config",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when validation fails",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate_run_logging(args.run_dir)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")

    if args.strict and not bool(report.get("valid", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
