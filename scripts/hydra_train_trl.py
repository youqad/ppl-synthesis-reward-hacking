#!/usr/bin/env python3
"""Hydra entrypoint for TRL GRPO training + W&B sweep logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hydra_train_common import (
    build_cfg_runner,
    build_sweep_summary,
    run_hydra_entrypoint,
    save_resolved_config,
)
from trl_reward_hacking import config_from_mapping, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


_build_sweep_summary = build_sweep_summary
_save_resolved_config = save_resolved_config
_run_from_cfg = build_cfg_runner(
    config_from_mapping=config_from_mapping,
    run_training=run_training,
    logger=log,
)


def main() -> None:
    run_hydra_entrypoint(config_name="trl_train", run_cfg=_run_from_cfg)


if __name__ == "__main__":
    main()
