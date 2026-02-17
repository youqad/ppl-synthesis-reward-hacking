#!/usr/bin/env python3
"""Run W&B sweep agents from CLI.

Supports local/VPS agents directly and can be used on cluster head nodes
where Hydra submitit handles per-trial SLURM job submission.
"""

from __future__ import annotations

import argparse
import logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Weights & Biases sweep agent")
    p.add_argument("--sweep-id", required=True, help="W&B sweep ID (e.g. entity/project/abc123)")
    p.add_argument("--count", type=int, default=None, help="Max runs for this agent")
    p.add_argument("--project", default=None, help="W&B project override")
    p.add_argument("--entity", default=None, help="W&B entity override")
    p.add_argument("--env-file", default=None, help="Optional .env path for WANDB_API_KEY")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.env_file:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("pip install python-dotenv to use --env-file") from exc
        load_dotenv(args.env_file)

    import wandb

    wandb.agent(
        sweep_id=args.sweep_id,
        function=None,
        count=args.count,
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()

