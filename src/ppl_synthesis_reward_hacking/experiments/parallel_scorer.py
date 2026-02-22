"""Parallel batch scoring for GRPO training steps.

Uses multiprocessing to score multiple completions concurrently,
needed because SMC scoring is ~10-100x slower than point_logps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    EXEC_FAIL_REWARD,
    score_completion_sandboxed,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ScoringTask:
    completion: str
    scoring_data: dict[str, Any]
    timeout: int
    reward_metric: str
    reward_data_split: str
    reward_estimator_backend: str
    smc_draws: int


def _score_one(task: _ScoringTask) -> tuple[float, dict[str, Any]]:
    """Score a single completion in a worker process."""
    try:
        reported, decomposition = score_completion_sandboxed(
            task.completion,
            task.scoring_data,
            timeout=task.timeout,
            reward_metric=task.reward_metric,
            reward_data_split=task.reward_data_split,
            reward_estimator_backend=task.reward_estimator_backend,
            smc_draws=task.smc_draws,
        )
        return reported, decomposition
    except Exception as e:
        log.debug("worker crashed: %s", e)
        return EXEC_FAIL_REWARD, {
            "outcome_code": "parallel_worker_crash",
            "outcome_detail": f"{type(e).__name__}: {e}",
        }


def score_batch_parallel(
    completions: list[str],
    scoring_data: dict[str, Any],
    *,
    workers: int = 8,
    timeout: int = 60,
    reward_metric: str = "log_marginal_likelihood",
    reward_data_split: str = "train",
    reward_estimator_backend: str = "smc",
    smc_draws: int = 500,
) -> list[tuple[float, dict[str, Any]]]:
    """Score completions in parallel via multiprocessing pool.

    Returns (reward_train, decomposition) per completion.
    """
    if not completions:
        return []

    work_items = [
        _ScoringTask(
            completion=c,
            scoring_data=scoring_data,
            timeout=timeout,
            reward_metric=reward_metric,
            reward_data_split=reward_data_split,
            reward_estimator_backend=reward_estimator_backend,
            smc_draws=smc_draws,
        )
        for c in completions
    ]
    effective_workers = min(workers, len(completions))

    try:
        with Pool(processes=effective_workers) as pool:
            results = pool.map(_score_one, work_items)
    except (OSError, RuntimeError, ValueError) as e:
        log.warning("pool failed, falling back to sequential: %s", e)
        results = [_score_one(item) for item in work_items]

    return results
