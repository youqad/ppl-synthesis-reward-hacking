#!/usr/bin/env python3
"""Hydra entrypoint for Tinker GRPO training + W&B sweep logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hydra_train_common import build_sweep_summary, run_from_cfg, save_resolved_config
from tinker_pymc_grpo import config_from_mapping, run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


_build_sweep_summary = build_sweep_summary
_save_resolved_config = save_resolved_config


def _run_from_cfg(cfg) -> dict[str, Any]:
    return run_from_cfg(
        cfg,
        config_from_mapping=config_from_mapping,
        run_training=run_training,
        logger=log,
    )


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs/hydra", config_name="tinker_train")
    def _entry(cfg: DictConfig) -> None:
        _run_from_cfg(cfg)

    _entry()


if __name__ == "__main__":
    main()
