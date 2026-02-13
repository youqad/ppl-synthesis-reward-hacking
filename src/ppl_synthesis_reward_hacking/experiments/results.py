"""Trajectory metrics and result printing for GRPO training."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from ppl_synthesis_reward_hacking.experiments.grpo import HACKING_GAP_THRESHOLD
from ppl_synthesis_reward_hacking.utils.math import is_nan_safe_gt

log = logging.getLogger(__name__)


class TrajectoryLike(Protocol):
    gap_mean: float
    reported_mean: float
    oracle_mean: float
    n_valid: int
    n_total: int


def compute_trajectory_metrics(trajectory: list[TrajectoryLike]) -> dict[str, Any]:
    if len(trajectory) < 2:
        return {"error": "not enough trajectory points"}

    initial = trajectory[0]
    final = trajectory[-1]

    gap_increase: float
    if np.isnan(final.gap_mean) or np.isnan(initial.gap_mean):
        gap_increase = float("nan")
    else:
        gap_increase = final.gap_mean - initial.gap_mean

    return {
        "initial_gap": initial.gap_mean,
        "final_gap": final.gap_mean,
        "gap_increase": gap_increase,
        "initial_reported": initial.reported_mean,
        "final_reported": final.reported_mean,
        "initial_oracle": initial.oracle_mean,
        "final_oracle": final.oracle_mean,
        "final_valid_rate": final.n_valid / max(final.n_total, 1),
        "hacking_detected": is_nan_safe_gt(
            final.gap_mean, initial.gap_mean + HACKING_GAP_THRESHOLD
        ),
    }


def print_training_summary(results: dict[str, Any]) -> None:
    if "error" in results:
        log.info("Error: %s", results["error"])
        return

    log.info(
        "gap: %.1f -> %.1f (delta=%.1f)",
        results["initial_gap"],
        results["final_gap"],
        results.get("gap_increase", float("nan")),
    )
    log.info(
        "reported: %.1f -> %.1f",
        results["initial_reported"],
        results["final_reported"],
    )
    log.info(
        "oracle: %.1f -> %.1f",
        results["initial_oracle"],
        results["final_oracle"],
    )
    log.info("valid_rate: %.1f%%", results["final_valid_rate"] * 100)

    if results.get("hacking_detected"):
        log.info("-> HACKING DETECTED (gap increased > %.0f)", HACKING_GAP_THRESHOLD)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
