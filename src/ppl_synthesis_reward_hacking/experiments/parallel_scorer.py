"""Parallel batch scoring for GRPO training steps.

Uses multiprocessing to score multiple completions concurrently,
needed because SMC scoring is ~10-100x slower than point_logps.
"""

from __future__ import annotations

import logging
from multiprocessing import Pool
from typing import Any

from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    EXEC_FAIL_REWARD,
    score_completion_sandboxed,
)

log = logging.getLogger(__name__)


def _score_one(
    args: tuple[str, dict[str, Any], int, str, int],
) -> tuple[float, float | None, dict | None]:
    """Unpack args and call sandboxed scorer."""
    completion, scoring_data, timeout, scoring_method, smc_draws = args
    try:
        reported, oracle, decomposition = score_completion_sandboxed(
            completion,
            scoring_data,
            timeout=timeout,
            scoring_method=scoring_method,
            smc_draws=smc_draws,
        )
        return reported, oracle, decomposition
    except Exception as e:
        log.debug("worker crashed: %s", e)
        return EXEC_FAIL_REWARD, EXEC_FAIL_REWARD, None


def score_batch_parallel(
    completions: list[str],
    scoring_data: dict[str, Any],
    *,
    workers: int = 8,
    timeout: int = 60,
    scoring_method: str = "smc",
    smc_draws: int = 500,
) -> list[tuple[float, float | None, dict | None]]:
    """Score completions in parallel via multiprocessing pool.

    Returns (reported, oracle, decomposition) per completion.
    """
    if not completions:
        return []

    work_items = [(c, scoring_data, timeout, scoring_method, smc_draws) for c in completions]
    effective_workers = min(workers, len(completions))

    try:
        with Pool(processes=effective_workers) as pool:
            results = pool.map(_score_one, work_items)
    except (OSError, RuntimeError, ValueError) as e:
        log.warning("pool failed, falling back to sequential: %s", e)
        results = [_score_one(item) for item in work_items]

    return results
