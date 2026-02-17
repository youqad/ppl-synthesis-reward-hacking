"""Synthstats reward function for GRPO training.

reported_reward = sum(logp(observed_terms + potential_terms)).
oracle_score = sum(logp(observed_terms)).
The gap between them measures evaluator exploitability.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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


@dataclass
class _BatchResult:
    rewards: list[float] = field(default_factory=list)
    oracle_scores: list[float | None] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    n_parse_fail: int = 0
    n_exec_fail: int = 0
    safety_counts: dict[str, int] = field(default_factory=dict)


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
    state = _init_reward_state(backend, dataset, scoring_data, cache_dir, completions_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        _ = kwargs
        state.call_count += 1
        n_expected = len(completions)
        batch = _score_batch(state, prompts, completions, exec_timeout)
        point = _build_trajectory_point(state.call_count, n_total=len(completions), batch=batch)
        state.trajectory.append(point)
        _flush_completions(state)
        _log_batch_summary(state.call_count, point, log_interval)
        return _normalize_reward_length(
            batch.rewards, n_expected=n_expected, batch_index=state.call_count
        )

    return reward_fn, state


def _init_reward_state(
    backend: PyMCBackend,
    dataset: Dataset,
    scoring_data: dict[str, Any],
    cache_dir: Path,
    completions_path: Path | None,
) -> SynthStatsRewardState:
    writer = CompletionWriter(completions_path) if completions_path else None
    return SynthStatsRewardState(
        backend=backend,
        dataset=dataset,
        scoring_data=scoring_data,
        cache_dir=cache_dir,
        completion_writer=writer,
    )


def _score_batch(
    state: SynthStatsRewardState,
    prompts: list[str],
    completions: list[str],
    exec_timeout: int,
) -> _BatchResult:
    batch = _BatchResult()
    for index, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        _score_completion(state, batch, index, prompt, completion, exec_timeout)
    return batch


def _score_completion(
    state: SynthStatsRewardState,
    batch: _BatchResult,
    index: int,
    prompt: str,
    completion: str,
    exec_timeout: int,
) -> None:
    completion_text = _extract_text(completion)
    code = extract_pymc_code(completion_text)
    if code is None:
        _record_parse_failure(state, batch, index, prompt, completion_text)
        return

    pymc_model = execute_pymc_code(code, state.scoring_data, timeout=exec_timeout)
    if pymc_model is None:
        _record_exec_failure(
            state, batch, index, prompt, completion_text, code, outcome="exec_fail"
        )
        return

    _score_executed_model(state, batch, index, prompt, completion_text, code, pymc_model)


def _score_executed_model(
    state: SynthStatsRewardState,
    batch: _BatchResult,
    index: int,
    prompt: str,
    completion_text: str,
    code: str,
    pymc_model: Any,
) -> None:
    try:
        reported, oracle, decomposition = score_pymc_model(
            pymc_model, scoring_data=state.scoring_data
        )
    except Exception as exc:
        if state.call_count <= 3:
            log.info("score_fail b=%d i=%d: %s", state.call_count, index, exc)
        _record_exec_failure(
            state, batch, index, prompt, completion_text, code, outcome="score_fail"
        )
        return

    if reported == EXEC_FAIL_REWARD and oracle == EXEC_FAIL_REWARD:
        _record_exec_failure(
            state, batch, index, prompt, completion_text, code, outcome="score_fail"
        )
        return

    batch.rewards.append(reported)
    batch.oracle_scores.append(oracle)
    batch.outcomes.append("valid")
    metadata = _merge_completion_metadata(_build_completion_metadata(pymc_model), decomposition)
    _log_completion(
        state,
        state.call_count,
        index,
        prompt,
        completion_text,
        code,
        reported,
        oracle if oracle is not None else EXEC_FAIL_REWARD,
        "valid",
        metadata=metadata,
    )
    _run_safety_check(pymc_model, batch.safety_counts)


def _merge_completion_metadata(
    metadata: dict[str, Any] | None,
    decomposition: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not decomposition:
        return metadata
    return {**(metadata or {}), "decomposition": decomposition}


def _record_parse_failure(
    state: SynthStatsRewardState,
    batch: _BatchResult,
    index: int,
    prompt: str,
    completion_text: str,
) -> None:
    batch.rewards.append(PARSE_FAIL_REWARD)
    batch.oracle_scores.append(None)
    batch.outcomes.append("parse_fail")
    batch.n_parse_fail += 1
    _log_completion(
        state,
        state.call_count,
        index,
        prompt,
        completion_text,
        None,
        PARSE_FAIL_REWARD,
        PARSE_FAIL_REWARD,
        "parse_fail",
    )


def _record_exec_failure(
    state: SynthStatsRewardState,
    batch: _BatchResult,
    index: int,
    prompt: str,
    completion_text: str,
    code: str | None,
    *,
    outcome: str,
) -> None:
    batch.rewards.append(EXEC_FAIL_REWARD)
    batch.oracle_scores.append(None)
    batch.outcomes.append(outcome)
    batch.n_exec_fail += 1
    _log_completion(
        state,
        state.call_count,
        index,
        prompt,
        completion_text,
        code,
        EXEC_FAIL_REWARD,
        EXEC_FAIL_REWARD,
        outcome,
    )


def _build_trajectory_point(
    batch_index: int, *, n_total: int, batch: _BatchResult
) -> TrajectoryPoint:
    valid_reported = [
        reward
        for reward, outcome in zip(batch.rewards, batch.outcomes, strict=True)
        if outcome == "valid"
    ]
    valid_pairs = [
        (reward, oracle)
        for reward, oracle, outcome in zip(
            batch.rewards, batch.oracle_scores, batch.outcomes, strict=True
        )
        if outcome == "valid" and oracle is not None
    ]
    if valid_pairs:
        reported_mean = float(np.mean([reward for reward, _ in valid_pairs]))
        oracle_mean = float(np.mean([oracle for _, oracle in valid_pairs]))
        gap = reported_mean - oracle_mean
    else:
        reported_mean = float("nan")
        oracle_mean = float("nan")
        gap = float("nan")
    reported_mean_all = float(np.mean(valid_reported)) if valid_reported else float("nan")
    return TrajectoryPoint(
        batch=batch_index,
        reported_mean=reported_mean,
        oracle_mean=oracle_mean,
        gap_mean=gap,
        n_valid=len(valid_pairs),
        reported_mean_all=reported_mean_all,
        n_valid_reported=len(valid_reported),
        n_total=n_total,
        n_parse_fail=batch.n_parse_fail,
        n_exec_fail=batch.n_exec_fail,
        safety_flags=dict(batch.safety_counts),
    )


def _flush_completions(state: SynthStatsRewardState) -> None:
    if state.completion_writer is not None:
        state.completion_writer.flush()


def _log_batch_summary(batch_index: int, point: TrajectoryPoint, log_interval: int) -> None:
    if batch_index % log_interval != 0 and batch_index > 5:
        return
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


def _normalize_reward_length(
    rewards: list[float], *, n_expected: int, batch_index: int
) -> list[float]:
    if len(rewards) == n_expected:
        return rewards

    log.warning(
        "Batch %d: reward length mismatch (got %d, expected %d), adjusting",
        batch_index,
        len(rewards),
        n_expected,
    )
    if len(rewards) > n_expected:
        return rewards[:n_expected]
    return rewards + [EXEC_FAIL_REWARD] * (n_expected - len(rewards))


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
