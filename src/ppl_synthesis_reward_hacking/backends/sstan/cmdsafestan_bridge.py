"""Bridge to cmdsafestan compiler-level checker.

cmdsafestan provides authoritative safety verdicts via compilation and
static analysis, unlike our regex-based check_sstan(). ~3.5s per model.

Install: pip install -e path/to/cmdsafestan (requires opam + dune build)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

log = logging.getLogger(__name__)

_CMDSAFESTAN_AVAILABLE = False
try:
    from cmdsafestan.api import evaluate_model_string
    from cmdsafestan.api import init as _cmdsafestan_init

    _CMDSAFESTAN_AVAILABLE = True
except ImportError:
    pass


@dataclass(frozen=True, slots=True)
class CmdSafeStanResult:
    safe: bool
    violations: list[str]
    log_likelihood: float | None
    runtime_ready: bool
    timing_s: float
    init_timing_s: float
    eval_timing_s: float
    runtime_reused: bool
    runtime_params: dict[str, Any]
    timing_phases_ms: dict[str, float]
    raw: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class CmdSafeStanRuntimeConfig:
    cmdstan_root: str = "."
    jobs: int = 4
    bootstrap: bool = True
    build_runtime: bool = True
    force_reinit: bool = False


_runtime = None
_runtime_params: tuple[str, int, bool, bool] | None = None
_runtime_lock = Lock()


def is_available() -> bool:
    return _CMDSAFESTAN_AVAILABLE


def ensure_runtime(
    *,
    cmdstan_root: str = ".",
    jobs: int = 4,
    bootstrap: bool = True,
    build_runtime: bool = True,
    force_reinit: bool = False,
) -> None:
    _ensure_runtime(
        runtime_cfg=CmdSafeStanRuntimeConfig(
            cmdstan_root=cmdstan_root,
            jobs=jobs,
            bootstrap=bootstrap,
            build_runtime=build_runtime,
            force_reinit=force_reinit,
        )
    )


def _ensure_runtime(*, runtime_cfg: CmdSafeStanRuntimeConfig) -> tuple[float, bool]:
    global _runtime
    global _runtime_params
    if not _CMDSAFESTAN_AVAILABLE:
        raise RuntimeError("cmdsafestan is not installed")
    if runtime_cfg.jobs <= 0:
        raise ValueError("cmdsafestan jobs must be positive")

    params = (
        runtime_cfg.cmdstan_root,
        runtime_cfg.jobs,
        runtime_cfg.bootstrap,
        runtime_cfg.build_runtime,
    )

    if _runtime is not None and _runtime_params == params and not runtime_cfg.force_reinit:
        return 0.0, True

    with _runtime_lock:
        if _runtime is not None and _runtime_params == params and not runtime_cfg.force_reinit:
            return 0.0, True

        if (
            _runtime is not None
            and _runtime_params is not None
            and _runtime_params != params
            and not runtime_cfg.force_reinit
        ):
            raise RuntimeError(
                "cmdsafestan runtime already initialized with different parameters; "
                "set force_reinit=True to override"
            )

        t0 = time.perf_counter()
        _runtime = _cmdsafestan_init(
            cmdstan_root=runtime_cfg.cmdstan_root,
            bootstrap=runtime_cfg.bootstrap,
            build_runtime=runtime_cfg.build_runtime,
            jobs=runtime_cfg.jobs,
        )
        _runtime_params = params
        return time.perf_counter() - t0, False


def check_with_cmdsafestan(
    stan_code: str,
    data: dict[str, Any],
    protect: list[str] | None = None,
    *,
    seed: int = 12345,
    cmdstan_root: str = ".",
    jobs: int = 4,
    bootstrap: bool = True,
    build_runtime: bool = True,
    force_reinit: bool = False,
) -> CmdSafeStanResult:
    """Run cmdsafestan compiler-level check on a Stan program.

    Returns structured result compatible with the gate flow.
    """
    if not _CMDSAFESTAN_AVAILABLE:
        raise RuntimeError("cmdsafestan is not installed; pip install -e path/to/cmdsafestan")

    if protect is None:
        protect = ["y"]
    runtime_cfg = CmdSafeStanRuntimeConfig(
        cmdstan_root=cmdstan_root,
        jobs=jobs,
        bootstrap=bootstrap,
        build_runtime=build_runtime,
        force_reinit=force_reinit,
    )
    init_timing_s, runtime_reused = _ensure_runtime(runtime_cfg=runtime_cfg)

    t0 = time.perf_counter()
    try:
        result = evaluate_model_string(
            stan_code,
            data,
            protect=protect,
            runtime=_runtime,
            seed=seed,
        )
        eval_timing_s = time.perf_counter() - t0
        total_timing_s = init_timing_s + eval_timing_s
        return CmdSafeStanResult(
            safe=result.safe,
            violations=[result.violation] if result.violation else [],
            log_likelihood=getattr(result, "log_likelihood", None),
            runtime_ready=getattr(result, "runtime_ready", True),
            timing_s=total_timing_s,
            init_timing_s=init_timing_s,
            eval_timing_s=eval_timing_s,
            runtime_reused=runtime_reused,
            runtime_params={
                "cmdstan_root": runtime_cfg.cmdstan_root,
                "jobs": runtime_cfg.jobs,
                "bootstrap": runtime_cfg.bootstrap,
                "build_runtime": runtime_cfg.build_runtime,
            },
            timing_phases_ms={
                "init": init_timing_s * 1000.0,
                "eval": eval_timing_s * 1000.0,
                "total": total_timing_s * 1000.0,
            },
            raw=_extract_raw(result),
        )
    except Exception as exc:
        eval_timing_s = time.perf_counter() - t0
        total_timing_s = init_timing_s + eval_timing_s
        log.warning("cmdsafestan error after %.1fs: %s", total_timing_s, exc)
        return CmdSafeStanResult(
            safe=False,
            violations=[f"{type(exc).__name__}: {exc}"],
            log_likelihood=None,
            runtime_ready=False,
            timing_s=total_timing_s,
            init_timing_s=init_timing_s,
            eval_timing_s=eval_timing_s,
            runtime_reused=runtime_reused,
            runtime_params={
                "cmdstan_root": runtime_cfg.cmdstan_root,
                "jobs": runtime_cfg.jobs,
                "bootstrap": runtime_cfg.bootstrap,
                "build_runtime": runtime_cfg.build_runtime,
            },
            timing_phases_ms={
                "init": init_timing_s * 1000.0,
                "eval": eval_timing_s * 1000.0,
                "total": total_timing_s * 1000.0,
            },
            raw=None,
        )


def _extract_raw(result: Any) -> dict[str, Any] | None:
    try:
        return {
            "safe": result.safe,
            "violation": getattr(result, "violation", None),
            "log_likelihood": getattr(result, "log_likelihood", None),
            "timings_seconds": getattr(result, "timings_seconds", None),
        }
    except Exception:
        return None
