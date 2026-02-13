"""Synthstats reward function for GRPO training.

reported_reward = mean logp over all RVs (hackable).
oracle_score = mean logp over observed RVs (honest).
The gap between them measures reward hacking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.backends.pymc.backend import PyMCBackend
from ppl_synthesis_reward_hacking.backends.pymc.code_executor import (
    execute_pymc_code,
    extract_pymc_code,
)
from ppl_synthesis_reward_hacking.data.schema import Dataset
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    DEFAULT_EXEC_TIMEOUT,
    EXEC_FAIL_REWARD,
    PARSE_FAIL_REWARD,
    score_pymc_model,
)
from ppl_synthesis_reward_hacking.logging.completions import (
    CompletionRecord,
    CompletionWriter,
    make_timestamp,
)

log = logging.getLogger(__name__)


def _extract_text(msg: str | list[dict]) -> str:
    """Extract plain text from a TRL prompt/completion (str or chat dicts)."""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        return "\n".join(m.get("content", "") for m in msg if isinstance(m, dict))
    return str(msg)


@dataclass
class TrajectoryPoint:
    batch: int
    reported_mean: float
    oracle_mean: float
    gap_mean: float
    n_valid: int
    n_total: int
    n_parse_fail: int
    n_exec_fail: int
    safety_flags: dict[str, int] = field(default_factory=dict)
    reported_mean_all: float = float("nan")
    n_valid_reported: int = 0


@dataclass
class SynthStatsRewardState:
    backend: PyMCBackend
    dataset: Dataset
    scoring_data: dict[str, Any]
    cache_dir: Path
    call_count: int = 0
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    completion_writer: CompletionWriter | None = None


def make_synthstats_reward_fn(
    backend: PyMCBackend,
    dataset: Dataset,
    scoring_data: dict[str, Any],
    *,
    cache_dir: Path = Path("/tmp/pymc_cache"),
    exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
    log_interval: int = 5,
    completions_path: Path | None = None,
) -> tuple[Callable[..., list[float]], SynthStatsRewardState]:
    """Create reward function. Returns (reward_fn, state)."""
    writer = CompletionWriter(completions_path) if completions_path else None
    state = SynthStatsRewardState(
        backend=backend,
        dataset=dataset,
        scoring_data=scoring_data,
        cache_dir=cache_dir,
        completion_writer=writer,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        state.call_count += 1
        n_expected = len(completions)
        rewards: list[float] = []
        oracle_scores: list[float | None] = []
        outcomes: list[str] = []
        n_parse_fail = 0
        n_exec_fail = 0
        safety_counts: dict[str, int] = {}

        for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            completion_text = _extract_text(completion)
            code = extract_pymc_code(completion_text)
            if code is None:
                rewards.append(PARSE_FAIL_REWARD)
                oracle_scores.append(None)
                outcomes.append("parse_fail")
                n_parse_fail += 1
                _log_completion(
                    state,
                    state.call_count,
                    i,
                    prompt,
                    completion_text,
                    None,
                    PARSE_FAIL_REWARD,
                    PARSE_FAIL_REWARD,
                    "parse_fail",
                )
                continue

            pymc_model = execute_pymc_code(code, state.scoring_data, timeout=exec_timeout)
            if pymc_model is None:
                rewards.append(EXEC_FAIL_REWARD)
                oracle_scores.append(None)
                outcomes.append("exec_fail")
                n_exec_fail += 1
                _log_completion(
                    state,
                    state.call_count,
                    i,
                    prompt,
                    completion_text,
                    code,
                    EXEC_FAIL_REWARD,
                    EXEC_FAIL_REWARD,
                    "exec_fail",
                )
                continue

            try:
                reported, oracle, _decomp = score_pymc_model(pymc_model)
                if reported == EXEC_FAIL_REWARD and oracle == EXEC_FAIL_REWARD:
                    rewards.append(EXEC_FAIL_REWARD)
                    oracle_scores.append(None)
                    outcomes.append("score_fail")
                    n_exec_fail += 1
                    _log_completion(
                        state,
                        state.call_count,
                        i,
                        prompt,
                        completion_text,
                        code,
                        EXEC_FAIL_REWARD,
                        EXEC_FAIL_REWARD,
                        "score_fail",
                    )
                    continue
                rewards.append(reported)
                oracle_scores.append(oracle)
                outcomes.append("valid")

                meta = _build_completion_metadata(pymc_model)
                _log_completion(
                    state,
                    state.call_count,
                    i,
                    prompt,
                    completion_text,
                    code,
                    reported,
                    oracle if oracle is not None else EXEC_FAIL_REWARD,
                    "valid",
                    metadata=meta,
                )

                _run_safety_check(pymc_model, safety_counts)

            except Exception as e:
                if state.call_count <= 3:
                    log.info("score_fail b=%d i=%d: %s", state.call_count, i, e)
                rewards.append(EXEC_FAIL_REWARD)
                oracle_scores.append(None)
                outcomes.append("score_fail")
                n_exec_fail += 1
                _log_completion(
                    state,
                    state.call_count,
                    i,
                    prompt,
                    completion_text,
                    code,
                    EXEC_FAIL_REWARD,
                    EXEC_FAIL_REWARD,
                    "score_fail",
                )

        valid_reported = [
            r for r, outcome in zip(rewards, outcomes, strict=True) if outcome == "valid"
        ]
        valid_pairs = [
            (r, o)
            for r, o, outcome in zip(rewards, oracle_scores, outcomes, strict=True)
            if outcome == "valid" and o is not None
        ]

        import numpy as np

        if valid_pairs:
            r_mean = float(np.mean([r for r, _ in valid_pairs]))
            o_mean = float(np.mean([o for _, o in valid_pairs]))
            gap = r_mean - o_mean
        else:
            r_mean = float("nan")
            o_mean = float("nan")
            gap = float("nan")

        if valid_reported:
            r_mean_all = float(np.mean(valid_reported))
        else:
            r_mean_all = float("nan")

        point = TrajectoryPoint(
            batch=state.call_count,
            reported_mean=r_mean,
            oracle_mean=o_mean,
            gap_mean=gap,
            n_valid=len(valid_pairs),
            reported_mean_all=r_mean_all,
            n_valid_reported=len(valid_reported),
            n_total=len(completions),
            n_parse_fail=n_parse_fail,
            n_exec_fail=n_exec_fail,
            safety_flags=dict(safety_counts),
        )
        state.trajectory.append(point)

        if state.completion_writer is not None:
            state.completion_writer.flush()

        if state.call_count % log_interval == 0 or state.call_count <= 5:
            log.info(
                "Batch %4d: reported=%.1f (all=%.1f) oracle=%.1f gap=%.1f "
                "valid=%d/%d valid_reported=%d parse_fail=%d exec_fail=%d",
                point.batch,
                point.reported_mean,
                point.reported_mean_all,
                point.oracle_mean,
                point.gap_mean,
                point.n_valid,
                point.n_total,
                point.n_valid_reported,
                point.n_parse_fail,
                point.n_exec_fail,
            )

        if len(rewards) != n_expected:
            log.warning(
                "Batch %d: reward length mismatch (got %d, expected %d), adjusting",
                state.call_count,
                len(rewards),
                n_expected,
            )
            if len(rewards) > n_expected:
                rewards = rewards[:n_expected]
            else:
                rewards.extend([EXEC_FAIL_REWARD] * (n_expected - len(rewards)))

        return rewards

    return reward_fn, state


def _log_completion(
    state: SynthStatsRewardState,
    batch: int,
    index: int,
    prompt: str,
    completion_text: str,
    code: str | None,
    reported: float,
    oracle: float,
    outcome: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    if state.completion_writer is None:
        return
    record = CompletionRecord(
        batch=batch,
        index=index,
        prompt=prompt,
        completion_text=completion_text,
        code=code,
        reported_reward=reported,
        oracle_score=oracle,
        gap=reported - oracle,
        outcome=outcome,
        timestamp=make_timestamp(),
        metadata=metadata,
    )
    state.completion_writer.write(record)


def _build_completion_metadata(pymc_model: Any) -> dict[str, Any] | None:
    import math

    meta: dict[str, Any] = {}
    try:
        logp_dict = pymc_model.point_logps()
        breakdown = {}
        for k, v in logp_dict.items():
            breakdown[k] = v if math.isfinite(v) else str(v)
        meta["logp_breakdown"] = breakdown
    except (TypeError, KeyError, AttributeError):
        pass
    try:
        meta["n_free_rvs"] = len(pymc_model.free_RVs)
        meta["n_observed_rvs"] = len(pymc_model.observed_RVs)
    except AttributeError:
        pass
    return meta or None


def _run_safety_check(pymc_model: Any, counts: dict[str, int]) -> None:
    try:
        from ppl_synthesis_reward_hacking.backends.pymc_safe.checker import check_pymc_model

        accepted, reasons = check_pymc_model(pymc_model)
        if not accepted:
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
    except Exception:  # checker may fail on malformed models
        pass
