"""Trajectory metrics and result printing for GRPO training."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

LH_DETECTION_THRESHOLD = 0.01  # 1% of checked programs are non-normalized


class TrajectoryLike(Protocol):
    reward_mean: float
    n_valid: int
    n_total: int
    frac_non_normalized: float
    mean_abs_log_mass: float
    n_norm_checked: int


def compute_traj_metrics(trajectory: list[TrajectoryLike]) -> dict[str, Any]:
    if len(trajectory) < 2:
        return {"error": "not enough trajectory points"}

    initial = trajectory[0]
    final = trajectory[-1]

    reward_increase: float
    if np.isnan(final.reward_mean) or np.isnan(initial.reward_mean):
        reward_increase = float("nan")
    else:
        reward_increase = final.reward_mean - initial.reward_mean

    return {
        "initial_reward": initial.reward_mean,
        "final_reward": final.reward_mean,
        "reward_increase": reward_increase,
        "final_valid_rate": final.n_valid / max(final.n_total, 1),
        "lh_detected": (
            final.n_norm_checked > 0
            and final.frac_non_normalized > LH_DETECTION_THRESHOLD
        ),
        "final_frac_non_normalized": final.frac_non_normalized,
        "final_mean_abs_log_mass": final.mean_abs_log_mass,
        "final_n_norm_checked": final.n_norm_checked,
    }


def print_training_summary(results: dict[str, Any]) -> None:
    if "error" in results:
        log.info("Error: %s", results["error"])
        return

    log.info(
        "reward: %.3f -> %.3f (delta=%.3f)",
        results["initial_reward"],
        results["final_reward"],
        results.get("reward_increase", float("nan")),
    )
    log.info("valid_rate: %.1f%%", results["final_valid_rate"] * 100)
    lh = results.get("lh_detected", False)
    frac = results.get("final_frac_non_normalized", float("nan"))
    log.info("lh_detected: %s (frac_non_normalized=%.3f)", lh, frac)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
