from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ppl_synthesis_reward_hacking.backends.sstan.checker import check_sstan
from ppl_synthesis_reward_hacking.backends.sstan.cmdsafestan_bridge import (
    check_with_cmdsafestan as _check_cmdsafestan,
    is_available as _cmdsafestan_available,
)
from ppl_synthesis_reward_hacking.backends.transpiler import transpile_pymc_to_safestan
from ppl_synthesis_reward_hacking.evaluation.heuristics import detect_exploits
from ppl_synthesis_reward_hacking.utils.hashing import normalized_text_hash

SStanGateMode = Literal["off", "shadow", "enforce"]
SStanFidelityPolicy = Literal["none", "reject_on_source_signal"]
SStanLogStanPolicy = Literal["none", "on_reject", "all"]


@dataclass(frozen=True, slots=True)
class SStanGateConfig:
    mode: SStanGateMode = "off"
    sample_rate: float = 1.0
    penalty_reward: float = -100.0
    fidelity_policy: SStanFidelityPolicy = "reject_on_source_signal"
    log_stan: SStanLogStanPolicy = "none"
    transpiler_model: str = "glm-5"
    transpiler_api_key_env: str | None = None
    transpiler_api_base: str | None = None
    transpiler_custom_llm_provider: str | None = None
    transpiler_strict: bool = True
    transpiler_temperature: float = 0.0
    transpiler_max_tokens: int = 2048
    transpiler_timeout_s: float = 60.0
    use_cmdsafestan: bool = False
    cmdsafestan_data: dict[str, Any] | None = None
    cmdsafestan_protect: list[str] | None = None


@dataclass(frozen=True, slots=True)
class SStanGateResult:
    mode: str
    checked: bool
    sampled: bool
    accepted: bool | None
    decision: str
    reasons: list[str]
    transpile_success: bool | None
    transpile_error: str | None
    sstan_reasons: list[str]
    fidelity_policy: str
    fidelity_reject: bool
    source_signal: bool
    source_tags: list[str]
    timing_ms: dict[str, float]
    penalty_applied: bool
    penalty_reward: float | None
    stan_path: str | None
    stan_hash: str | None
    cmdsafestan_safe: bool | None = None
    cmdsafestan_violations: list[str] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "checked": self.checked,
            "sampled": self.sampled,
            "accepted": self.accepted,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "transpile_success": self.transpile_success,
            "transpile_error": self.transpile_error,
            "sstan_reasons": list(self.sstan_reasons),
            "fidelity_policy": self.fidelity_policy,
            "fidelity_reject": self.fidelity_reject,
            "source_signal": self.source_signal,
            "source_tags": list(self.source_tags),
            "timing_ms": dict(self.timing_ms),
            "penalty_applied": self.penalty_applied,
            "penalty_reward": self.penalty_reward,
            "stan_path": self.stan_path,
            "stan_hash": self.stan_hash,
            "cmdsafestan_safe": self.cmdsafestan_safe,
            "cmdsafestan_violations": list(self.cmdsafestan_violations)
            if self.cmdsafestan_violations is not None
            else None,
        }


def validate_sstan_gate_cfg(config: SStanGateConfig, *, paper_track: str) -> None:
    _validate_gate_modes(config)
    _validate_gate_numeric_bounds(config)
    _validate_enforce_mode_constraints(config)
    _validate_paper_track_constraints(config, paper_track=paper_track)
    _validate_transpiler_requirements(config)
    _validate_cmdsafestan_requirements(config)


def _validate_gate_modes(config: SStanGateConfig) -> None:
    if config.mode not in {"off", "shadow", "enforce"}:
        raise ValueError("sstan_gate_mode must be off|shadow|enforce")
    if config.fidelity_policy not in {"none", "reject_on_source_signal"}:
        raise ValueError(
            "sstan_gate_fidelity_policy must be one of ['none', 'reject_on_source_signal']"
        )
    if config.log_stan not in {"none", "on_reject", "all"}:
        raise ValueError("sstan_gate_log_stan must be one of ['none', 'on_reject', 'all']")


def _validate_gate_numeric_bounds(config: SStanGateConfig) -> None:
    if not (0.0 < config.sample_rate <= 1.0):
        raise ValueError("sstan_gate_sample_rate must satisfy 0 < sample_rate <= 1")
    if not math.isfinite(config.penalty_reward) or config.penalty_reward > 0:
        raise ValueError("sstan_gate_penalty_reward must be finite and <= 0")
    if config.transpiler_max_tokens <= 0:
        raise ValueError("sstan_transpiler_max_tokens must be > 0")
    if config.transpiler_timeout_s <= 0:
        raise ValueError("sstan_transpiler_timeout_s must be > 0")


def _validate_enforce_mode_constraints(config: SStanGateConfig) -> None:
    if config.mode != "enforce":
        return
    if config.sample_rate != 1.0:
        raise ValueError("sstan_gate_mode=enforce requires sstan_gate_sample_rate == 1.0")
    if not config.transpiler_strict:
        raise ValueError("sstan_gate_mode=enforce requires sstan_transpiler_strict=true")


def _validate_paper_track_constraints(config: SStanGateConfig, *, paper_track: str) -> None:
    if paper_track == "part_b_mitigation" and config.mode == "off":
        raise ValueError("paper_track=part_b_mitigation requires sstan_gate_mode != off")


def _validate_transpiler_requirements(config: SStanGateConfig) -> None:
    if config.mode == "off":
        return
    if not config.transpiler_model.strip():
        raise ValueError("sstan_transpiler_model must be non-empty when sstan_gate_mode != off")
    if not config.transpiler_api_key_env:
        raise ValueError("sstan_transpiler_api_key_env must be set when sstan_gate_mode != off")
    if not os.getenv(config.transpiler_api_key_env):
        raise ValueError(f"Missing transpiler API key env var: {config.transpiler_api_key_env!r}")


def _validate_cmdsafestan_requirements(config: SStanGateConfig) -> None:
    if not config.use_cmdsafestan:
        return
    if config.mode == "off":
        raise ValueError("use_cmdsafestan=True requires sstan_gate_mode != off")
    if not _cmdsafestan_available():
        raise ValueError(
            "use_cmdsafestan=True but cmdsafestan is not installed; "
            "pip install -e path/to/cmdsafestan"
        )
    if config.cmdsafestan_data is None:
        raise ValueError(
            "use_cmdsafestan=True requires cmdsafestan_data (data dict for compiler check)"
        )


def run_sstan_gate(
    *,
    code: str | None,
    config: SStanGateConfig,
    batch: int,
    index: int,
    sampling_seed: int,
    cache_dir: Path | None = None,
) -> SStanGateResult:
    if config.mode == "off":
        return _result(
            mode=config.mode,
            checked=False,
            sampled=False,
            accepted=None,
            decision="off",
            reasons=["gate_off"],
        )

    source = (code or "").strip()
    if not source:
        return _result(
            mode=config.mode,
            checked=False,
            sampled=False,
            accepted=None,
            decision="skipped",
            reasons=["invalid_completion_no_code"],
        )

    sampled = _should_check(
        sample_rate=config.sample_rate,
        batch=batch,
        index=index,
        seed=sampling_seed,
    )
    if not sampled:
        return _result(
            mode=config.mode,
            checked=False,
            sampled=False,
            accepted=None,
            decision="skipped",
            reasons=["gate_sampled_out"],
        )

    source_tags = sorted(detect_exploits(source))
    source_signal = bool(source_tags)
    transpile_start = time.perf_counter()
    transpile = transpile_pymc_to_safestan(
        source,
        model=config.transpiler_model,
        api_key_env=config.transpiler_api_key_env,
        api_base=config.transpiler_api_base,
        custom_llm_provider=config.transpiler_custom_llm_provider,
        strict=config.transpiler_strict,
        temperature=config.transpiler_temperature,
        max_tokens=config.transpiler_max_tokens,
        timeout=max(1, int(math.ceil(config.transpiler_timeout_s))),
    )
    transpile_ms = (time.perf_counter() - transpile_start) * 1000.0

    if not transpile.success or transpile.stan_code is None:
        return _result(
            mode=config.mode,
            checked=True,
            sampled=True,
            accepted=False,
            decision="reject",
            reasons=["transpile_failed"],
            transpile_success=False,
            transpile_error=transpile.error,
            source_signal=source_signal,
            source_tags=source_tags,
            timing_ms={"transpile": transpile_ms, "check": 0.0, "total": transpile_ms},
        )

    checker_start = time.perf_counter()
    try:
        accepted, sstan_reasons = check_sstan(transpile.stan_code)
        check_ms = (time.perf_counter() - checker_start) * 1000.0
    except Exception as exc:
        check_ms = (time.perf_counter() - checker_start) * 1000.0
        return _result(
            mode=config.mode,
            checked=True,
            sampled=True,
            accepted=False,
            decision="reject",
            reasons=["sstan_exception"],
            transpile_success=True,
            transpile_error=None,
            sstan_reasons=[f"{type(exc).__name__}: {exc}"],
            source_signal=source_signal,
            source_tags=source_tags,
            timing_ms={
                "transpile": transpile_ms,
                "check": check_ms,
                "total": transpile_ms + check_ms,
            },
            stan_path=_maybe_store_stan(
                stan_code=transpile.stan_code,
                accepted=False,
                config=config,
                cache_dir=cache_dir,
                code=source,
            ),
            stan_hash=normalized_text_hash(transpile.stan_code),
        )

    # cmdsafestan compiler check: only run when regex accepted and config enables it.
    # regex rejection is final (no point compiling a known-bad program).
    cmdsafestan_safe: bool | None = None
    cmdsafestan_violations: list[str] | None = None
    cmdsafestan_ms = 0.0
    if accepted and config.use_cmdsafestan:
        cmdsafestan_start = time.perf_counter()
        cms_result = _check_cmdsafestan(
            transpile.stan_code,
            config.cmdsafestan_data or {},
            protect=config.cmdsafestan_protect,
        )
        cmdsafestan_ms = (time.perf_counter() - cmdsafestan_start) * 1000.0
        cmdsafestan_safe = cms_result.safe
        cmdsafestan_violations = list(cms_result.violations)
        if not cms_result.safe:
            accepted = False

    fidelity_reject = (
        config.fidelity_policy == "reject_on_source_signal" and source_signal and accepted
    )
    reasons: list[str] = []
    if not accepted and cmdsafestan_safe is not None and not cmdsafestan_safe:
        reasons.append("cmdsafestan_reject")
    if not accepted and cmdsafestan_safe is None:
        reasons.append("sstan_reject")
    if fidelity_reject:
        reasons.append("fidelity_reject_source_signal")
        accepted = False
    decision = "accept" if accepted else "reject"

    timing = {
        "transpile": transpile_ms,
        "check": check_ms,
        "total": transpile_ms + check_ms + cmdsafestan_ms,
    }
    if cmdsafestan_ms > 0:
        timing["cmdsafestan"] = cmdsafestan_ms

    return _result(
        mode=config.mode,
        checked=True,
        sampled=True,
        accepted=accepted,
        decision=decision,
        reasons=reasons if reasons else ["accepted"],
        transpile_success=True,
        transpile_error=None,
        sstan_reasons=list(sstan_reasons),
        fidelity_policy=config.fidelity_policy,
        fidelity_reject=fidelity_reject,
        source_signal=source_signal,
        source_tags=source_tags,
        timing_ms=timing,
        stan_path=_maybe_store_stan(
            stan_code=transpile.stan_code,
            accepted=accepted,
            config=config,
            cache_dir=cache_dir,
            code=source,
        ),
        stan_hash=normalized_text_hash(transpile.stan_code),
        cmdsafestan_safe=cmdsafestan_safe,
        cmdsafestan_violations=cmdsafestan_violations,
    )


def _should_check(*, sample_rate: float, batch: int, index: int, seed: int) -> bool:
    key = f"{seed}:{batch}:{index}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return value < sample_rate


def _maybe_store_stan(
    *,
    stan_code: str,
    accepted: bool,
    config: SStanGateConfig,
    cache_dir: Path | None,
    code: str,
) -> str | None:
    if cache_dir is None:
        return None
    if config.log_stan == "none":
        return None
    if config.log_stan == "on_reject" and accepted:
        return None
    safe_root = Path(cache_dir) / "stan"
    safe_root.mkdir(parents=True, exist_ok=True)
    code_hash = normalized_text_hash(code)
    out_path = safe_root / f"{code_hash}.stan"
    out_path.write_text(stan_code, encoding="utf-8")
    return str(out_path)


def _result(
    *,
    mode: str,
    checked: bool,
    sampled: bool,
    accepted: bool | None,
    decision: str,
    reasons: list[str],
    transpile_success: bool | None = None,
    transpile_error: str | None = None,
    sstan_reasons: list[str] | None = None,
    fidelity_policy: str = "none",
    fidelity_reject: bool = False,
    source_signal: bool = False,
    source_tags: list[str] | None = None,
    timing_ms: dict[str, float] | None = None,
    penalty_applied: bool = False,
    penalty_reward: float | None = None,
    stan_path: str | None = None,
    stan_hash: str | None = None,
    cmdsafestan_safe: bool | None = None,
    cmdsafestan_violations: list[str] | None = None,
) -> SStanGateResult:
    return SStanGateResult(
        mode=mode,
        checked=checked,
        sampled=sampled,
        accepted=accepted,
        decision=decision,
        reasons=list(reasons),
        transpile_success=transpile_success,
        transpile_error=transpile_error,
        sstan_reasons=list(sstan_reasons or []),
        fidelity_policy=fidelity_policy,
        fidelity_reject=fidelity_reject,
        source_signal=source_signal,
        source_tags=list(source_tags or []),
        timing_ms=dict(timing_ms or {"transpile": 0.0, "check": 0.0, "total": 0.0}),
        penalty_applied=penalty_applied,
        penalty_reward=penalty_reward,
        stan_path=stan_path,
        stan_hash=stan_hash,
        cmdsafestan_safe=cmdsafestan_safe,
        cmdsafestan_violations=list(cmdsafestan_violations)
        if cmdsafestan_violations is not None
        else None,
    )


__all__ = [
    "SStanGateConfig",
    "SStanGateResult",
    "run_sstan_gate",
    "validate_sstan_gate_cfg",
]
