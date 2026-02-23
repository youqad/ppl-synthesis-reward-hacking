#!/usr/bin/env python3
"""Render canonical W&B sweep YAMLs from shared template definitions."""

from __future__ import annotations

from pathlib import Path

import yaml

from ppl_synthesis_reward_hacking.config.wandb_sweep_templates import (
    render_tinker_parta_sweeps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO_ROOT / "configs" / "sweeps" / "wandb"


def main() -> None:
    rendered = render_tinker_parta_sweeps()
    for filename, payload in rendered.items():
        path = SWEEP_DIR / filename
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
