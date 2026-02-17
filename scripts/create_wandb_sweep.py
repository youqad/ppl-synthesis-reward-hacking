#!/usr/bin/env python3
"""Create a W&B sweep from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a Weights & Biases sweep from YAML")
    p.add_argument("--config", required=True, help="Path to sweep YAML config")
    p.add_argument("--project", default=None, help="W&B project override")
    p.add_argument("--entity", default=None, help="W&B entity override")
    p.add_argument("--env-file", default=None, help="Optional .env path for WANDB_API_KEY")
    return p.parse_args()


def _load_sweep_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Sweep config must be a mapping: {path}")
    return data


def main() -> None:
    args = parse_args()

    if args.env_file:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("pip install python-dotenv to use --env-file") from exc
        load_dotenv(args.env_file)

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Sweep config not found: {config_path}")

    sweep_cfg = _load_sweep_config(config_path)

    import wandb

    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project=args.project,
        entity=args.entity,
    )
    print(sweep_id)


if __name__ == "__main__":
    main()
