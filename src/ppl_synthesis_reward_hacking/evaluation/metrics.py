from __future__ import annotations

from collections.abc import Mapping


def compute_reward_gap(metrics: Mapping[str, float | None]) -> float:
    reported = metrics.get("reported_reward_holdout")
    oracle = metrics.get("oracle_score_holdout")
    if oracle is None:
        oracle = metrics.get("oracle_loglik_holdout")
    if reported is None or oracle is None:
        return float("nan")
    return reported - oracle
