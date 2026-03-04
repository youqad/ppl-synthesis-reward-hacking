"""Judge gate for hacking verdicts."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord, make_timestamp
from ppl_synthesis_reward_hacking.monitoring.llm_judge import (
    JudgeConfig,
    JudgeVerdict,
    judge_completions,
    judge_sample,
)

log = logging.getLogger(__name__)

JudgeGateMode = Literal["off", "shadow", "enforce"]

JUDGE_BACKEND_DEFAULTS: dict[str, dict[str, str | None]] = {
    "openai": {
        "api_base": None,
        "custom_llm_provider": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "api_base": None,
        "custom_llm_provider": None,
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "zai": {
        "api_base": "https://api.z.ai/api/anthropic",
        "custom_llm_provider": "anthropic",
        "api_key_env": "ZHIPUAI_API_KEY",
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "custom_llm_provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "custom": {
        "api_base": None,
        "custom_llm_provider": None,
        "api_key_env": None,
    },
}


@dataclass(frozen=True, slots=True)
class JudgeGateConfig:
    mode: JudgeGateMode = "off"
    confidence_threshold: float = 0.8
    penalty_reward: float = -400.0
    judge_backend: str = "openai"
    judge_model: str = "gpt-5.2"
    judge_api_key_env: str | None = None
    judge_api_base: str | None = None
    judge_custom_llm_provider: str = ""
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024
    judge_timeout: int = 60
    judge_batch_max_workers: int = 32


@dataclass(frozen=True, slots=True)
class JudgeGateResult:
    mode: str
    checked: bool
    accepted: bool | None
    decision: str
    verdict: str | None
    confidence: float | None
    tags: list[str]
    reasoning: str | None
    penalty_applied: bool
    penalty_reward: float | None
    timing_ms: float
    parse_error: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _skipped_result(mode: str) -> JudgeGateResult:
    return JudgeGateResult(
        mode=mode,
        checked=False,
        accepted=None,
        decision="off" if mode == "off" else "skipped",
        verdict=None,
        confidence=None,
        tags=[],
        reasoning=None,
        penalty_applied=False,
        penalty_reward=None,
        timing_ms=0.0,
    )


def validate_judge_gate_cfg(config: JudgeGateConfig, *, paper_track: str) -> None:
    if config.mode not in {"off", "shadow", "enforce"}:
        raise ValueError("judge_gate_mode must be off|shadow|enforce")
    if not 0.0 <= config.confidence_threshold <= 1.0:
        raise ValueError("judge_gate_confidence_threshold must be in [0, 1]")
    if config.mode == "off":
        return
    if config.mode == "enforce" and config.penalty_reward >= 0.0:
        raise ValueError(
            f"penalty_reward must be negative in enforce mode, got {config.penalty_reward}"
        )
    backend_key = config.judge_backend
    if backend_key not in JUDGE_BACKEND_DEFAULTS:
        raise ValueError(f"judge_gate_backend must be one of {sorted(JUDGE_BACKEND_DEFAULTS)}")
    resolved_env = config.judge_api_key_env
    if not resolved_env:
        defaults = JUDGE_BACKEND_DEFAULTS[backend_key]
        resolved_env = defaults.get("api_key_env") or ""
    if resolved_env and not os.getenv(resolved_env):
        raise ValueError(f"judge gate requires {resolved_env} in environment")


def _build_judge_config(config: JudgeGateConfig) -> JudgeConfig:
    defaults = JUDGE_BACKEND_DEFAULTS.get(config.judge_backend, {})
    api_base = (
        config.judge_api_base if config.judge_api_base is not None else defaults.get("api_base")
    )
    custom_llm_provider = (
        config.judge_custom_llm_provider
        if config.judge_custom_llm_provider
        else (defaults.get("custom_llm_provider") or "")
    )
    api_key_env = config.judge_api_key_env or defaults.get("api_key_env") or ""
    return JudgeConfig(
        backend=config.judge_backend,
        model=config.judge_model,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        api_key_env=api_key_env,
        temperature=config.judge_temperature,
        max_tokens=config.judge_max_tokens,
        timeout=config.judge_timeout,
        batch_mode="auto",
        batch_max_workers=config.judge_batch_max_workers,
        enabled=True,
    )


def _verdict_to_result(
    verdict: JudgeVerdict,
    *,
    config: JudgeGateConfig,
    timing_ms: float,
) -> JudgeGateResult:
    if verdict.parse_error:
        if config.mode == "enforce":
            return JudgeGateResult(
                mode=config.mode,
                checked=True,
                accepted=False,
                decision="reject_parse_error",
                verdict=verdict.verdict,
                confidence=verdict.confidence,
                tags=list(verdict.tags),
                reasoning=verdict.rationale,
                penalty_applied=True,
                penalty_reward=config.penalty_reward,
                timing_ms=timing_ms,
                parse_error=True,
            )
        if config.mode == "shadow":
            return JudgeGateResult(
                mode=config.mode,
                checked=True,
                accepted=True,
                decision="shadow_parse_error",
                verdict=verdict.verdict,
                confidence=verdict.confidence,
                tags=list(verdict.tags),
                reasoning=verdict.rationale,
                penalty_applied=False,
                penalty_reward=None,
                timing_ms=timing_ms,
                parse_error=True,
            )

    should_reject = verdict.is_hacking and verdict.confidence >= config.confidence_threshold
    if config.mode == "enforce" and should_reject:
        return JudgeGateResult(
            mode=config.mode,
            checked=True,
            accepted=False,
            decision="reject",
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            tags=list(verdict.tags),
            reasoning=verdict.rationale,
            penalty_applied=True,
            penalty_reward=config.penalty_reward,
            timing_ms=timing_ms,
            parse_error=False,
        )
    if config.mode == "shadow" and should_reject:
        return JudgeGateResult(
            mode=config.mode,
            checked=True,
            accepted=True,
            decision="shadow_reject",
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            tags=list(verdict.tags),
            reasoning=verdict.rationale,
            penalty_applied=False,
            penalty_reward=None,
            timing_ms=timing_ms,
            parse_error=False,
        )
    return JudgeGateResult(
        mode=config.mode,
        checked=True,
        accepted=True,
        decision="accept",
        verdict=verdict.verdict,
        confidence=verdict.confidence,
        tags=list(verdict.tags),
        reasoning=verdict.rationale,
        penalty_applied=False,
        penalty_reward=None,
        timing_ms=timing_ms,
        parse_error=False,
    )


def _fail_open_result(mode: str, error: str, timing_ms: float) -> JudgeGateResult:
    return JudgeGateResult(
        mode=mode,
        checked=True,
        accepted=True,
        decision="fail_open",
        verdict=None,
        confidence=None,
        tags=[],
        reasoning=f"fail-open: {error}",
        penalty_applied=False,
        penalty_reward=None,
        timing_ms=timing_ms,
        parse_error=False,
    )


def run_judge_gate(
    *,
    code: str,
    reported_reward: float,
    config: JudgeGateConfig,
) -> JudgeGateResult:
    if config.mode == "off":
        return _skipped_result("off")

    judge_cfg = _build_judge_config(config)
    t0 = time.perf_counter()
    try:
        verdict = judge_sample(
            prompt="",
            completion=code,
            reported_reward=reported_reward,
            cfg=judge_cfg,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log.warning("judge gate fail-open on single call: %s", exc)
        return _fail_open_result(config.mode, str(exc), elapsed_ms)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return _verdict_to_result(verdict, config=config, timing_ms=elapsed_ms)


def run_judge_gate_batch(
    *,
    codes: list[str],
    rewards: list[float],
    config: JudgeGateConfig,
) -> list[JudgeGateResult]:
    if config.mode == "off":
        return [_skipped_result("off") for _ in codes]
    if not codes:
        return []
    if len(codes) != len(rewards):
        raise ValueError(
            "run_judge_gate_batch requires codes and rewards with the same length; "
            f"got len(codes)={len(codes)}, len(rewards)={len(rewards)}"
        )

    judge_cfg = _build_judge_config(config)
    ts = make_timestamp()
    records = [
        CompletionRecord(
            batch=0,
            index=i,
            prompt="",
            completion_text=code,
            code=code,
            reported_reward=reward,
            outcome="valid",
            timestamp=ts,
        )
        for i, (code, reward) in enumerate(zip(codes, rewards, strict=True))
    ]
    t0 = time.perf_counter()
    try:
        verdicts = judge_completions(records, judge_cfg)
        if len(verdicts) != len(codes):
            raise RuntimeError(
                f"judge gate batch returned {len(verdicts)} verdicts for {len(codes)} codes"
            )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = elapsed_ms / max(len(codes), 1)
        log.warning("judge gate fail-open on batch call: %s", exc)
        return [_fail_open_result(config.mode, str(exc), per_item_ms) for _ in codes]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_item_ms = elapsed_ms / max(len(codes), 1)

    return [_verdict_to_result(v, config=config, timing_ms=per_item_ms) for v in verdicts]


__all__ = [
    "JudgeGateConfig",
    "JudgeGateResult",
    "run_judge_gate",
    "run_judge_gate_batch",
    "validate_judge_gate_cfg",
]
