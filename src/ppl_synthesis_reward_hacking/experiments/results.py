"""Trajectory metrics and result printing for GRPO training."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from ppl_synthesis_reward_hacking.experiments.grpo import HACKING_GAP_THRESHOLD
from ppl_synthesis_reward_hacking.utils.math import is_nan_safe_gt

log = logging.getLogger(__name__)


class TrajectoryLike(Protocol):
    gap_proxy_mean: float
    reward_mean: float
    oracle_proxy_mean: float
    n_valid: int
    n_total: int


def compute_trajectory_metrics(trajectory: list[TrajectoryLike]) -> dict[str, Any]:
    if len(trajectory) < 2:
        return {"error": "not enough trajectory points"}

    initial = trajectory[0]
    final = trajectory[-1]

    reward_increase: float
    if np.isnan(final.reward_mean) or np.isnan(initial.reward_mean):
        reward_increase = float("nan")
    else:
        reward_increase = final.reward_mean - initial.reward_mean

    gap_proxy_increase: float
    if np.isnan(final.gap_proxy_mean) or np.isnan(initial.gap_proxy_mean):
        gap_proxy_increase = float("nan")
    else:
        gap_proxy_increase = final.gap_proxy_mean - initial.gap_proxy_mean

    return {
        "initial_reward": initial.reward_mean,
        "final_reward": final.reward_mean,
        "reward_increase": reward_increase,
        "initial_oracle_proxy": initial.oracle_proxy_mean,
        "final_oracle_proxy": final.oracle_proxy_mean,
        "initial_gap_proxy": initial.gap_proxy_mean,
        "final_gap_proxy": final.gap_proxy_mean,
        "gap_proxy_increase": gap_proxy_increase,
        "final_valid_rate": final.n_valid / max(final.n_total, 1),
        "hacking_detected": is_nan_safe_gt(
            final.gap_proxy_mean, initial.gap_proxy_mean + HACKING_GAP_THRESHOLD
        ),
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
    log.info(
        "oracle_proxy: %.3f -> %.3f",
        results["initial_oracle_proxy"],
        results["final_oracle_proxy"],
    )
    log.info(
        "gap_proxy: %.3f -> %.3f (delta=%.3f)",
        results["initial_gap_proxy"],
        results["final_gap_proxy"],
        results.get("gap_proxy_increase", float("nan")),
    )
    log.info("valid_rate: %.1f%%", results["final_valid_rate"] * 100)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
