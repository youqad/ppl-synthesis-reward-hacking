"""Bridge to Jacek's cmdsafestan compiler-level checker.

cmdsafestan provides authoritative safety verdicts via compilation and
static analysis, unlike our regex-based check_sstan(). ~3.5s per model.

Install: pip install -e path/to/cmdsafestan (requires opam + dune build)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
    raw: dict[str, Any] | None


_runtime = None


def is_available() -> bool:
    return _CMDSAFESTAN_AVAILABLE


def ensure_runtime(*, cmdstan_root: str = ".", jobs: int = 4) -> None:
    global _runtime
    if _runtime is not None:
        return
    if not _CMDSAFESTAN_AVAILABLE:
        raise RuntimeError("cmdsafestan is not installed")
    _runtime = _cmdsafestan_init(
        cmdstan_root=cmdstan_root,
        bootstrap=True,
        build_runtime=True,
        jobs=jobs,
    )


def check_with_cmdsafestan(
    stan_code: str,
    data: dict[str, Any],
    protect: list[str] | None = None,
    *,
    seed: int = 12345,
) -> CmdSafeStanResult:
    """Run cmdsafestan compiler-level check on a Stan program.

    Returns structured result compatible with the gate flow.
    """
    if not _CMDSAFESTAN_AVAILABLE:
        raise RuntimeError(
            "cmdsafestan is not installed; pip install -e path/to/cmdsafestan"
        )

    if protect is None:
        protect = ["y"]

    ensure_runtime()

    t0 = time.perf_counter()
    try:
        result = evaluate_model_string(
            stan_code,
            data,
            protect=protect,
            runtime=_runtime,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        return CmdSafeStanResult(
            safe=result.safe,
            violations=[result.violation] if result.violation else [],
            log_likelihood=getattr(result, "log_likelihood", None),
            runtime_ready=getattr(result, "runtime_ready", True),
            timing_s=elapsed,
            raw=_extract_raw(result),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.warning("cmdsafestan error after %.1fs: %s", elapsed, exc)
        return CmdSafeStanResult(
            safe=False,
            violations=[f"{type(exc).__name__}: {exc}"],
            log_likelihood=None,
            runtime_ready=False,
            timing_s=elapsed,
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
