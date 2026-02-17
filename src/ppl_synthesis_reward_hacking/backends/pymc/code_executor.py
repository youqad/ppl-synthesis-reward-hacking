"""Extract and execute PyMC code from LLM completions.

Handles the pipeline: raw completion text -> Python source -> PyMC model object.
Uses a restricted exec namespace and POSIX signal-based timeout.
"""

from __future__ import annotations

import logging
import re
import signal
from typing import Any

log = logging.getLogger(__name__)

_UNSAFE_BUILTINS = frozenset({
    "eval", "exec", "compile", "open",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "breakpoint",
})
_SAFE_BUILTINS: dict[str, object] = (
    {k: v for k, v in __builtins__.items() if k not in _UNSAFE_BUILTINS}  # type: ignore[union-attr]
    if isinstance(__builtins__, dict)
    else {k: getattr(__builtins__, k) for k in dir(__builtins__) if k not in _UNSAFE_BUILTINS}
)

_CODE_BLOCK_RE = re.compile(
    r"```python\s*\n(.*?)```",
    re.DOTALL,
)

# fallback: bare def model(data): ... block without fences
_BARE_MODEL_RE = re.compile(
    r"((?:import\s+\S+.*\n)*"  # optional imports before def
    r"def\s+model\s*\(\s*data\s*\)\s*:.*?)(?:\n\S|\Z)",
    re.DOTALL,
)

_BANNED_INIT_PATTERNS = re.compile(r"\b(?:initval|testval|initial_point)\b\s*=")

_GROUND_TRUTH_EXTRA = frozenset({"beta", "noise_sigma"})


def extract_pymc_code(completion: str) -> str | None:
    """Extract Python code from an LLM completion.

    Tries fenced ```python blocks first, then bare `def model(data):` patterns.
    Returns the code string, or None if no valid block is found.
    Rejects code containing initval/testval/initial_point (evaluator hack).
    """
    code: str | None = None

    match = _CODE_BLOCK_RE.search(completion)
    if match is not None:
        candidate = match.group(1).strip()
        if candidate:
            code = candidate

    if code is None:
        match = _BARE_MODEL_RE.search(completion)
        if match is not None:
            candidate = match.group(1).strip()
            if candidate and "def model" in candidate:
                code = candidate

    if code is not None and _BANNED_INIT_PATTERNS.search(code):
        log.debug("banned init pattern: initval/testval/initial_point")
        return None

    return code


def execute_pymc_code(
    code: str,
    data: dict[str, Any],
    *,
    timeout: int = 30,
) -> Any | None:
    """exec() LLM-generated code in a restricted namespace, return the PyMC model or None.

    Not sandboxed -- do not use with untrusted inputs outside a research environment.
    """
    namespace = _build_namespace()

    try:
        exec(compile(code, "<llm-generated>", "exec"), namespace)  # noqa: S102
    except Exception as e:
        log.debug("exec compile/run failed: %s: %s", type(e).__name__, e)
        return None

    model_fn = namespace.get("model")
    if not callable(model_fn):
        log.debug("no callable model() in namespace (keys: %s)", list(namespace.keys())[:10])
        return None

    prev_handler = None
    timed_out = False

    def _timeout_handler(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise TimeoutError("timeout")

    # signal.alarm only works on POSIX (Linux, macOS) and only in the main thread
    use_alarm = hasattr(signal, "SIGALRM")

    # strip ground-truth keys so generated code can't cheat
    sanitized = {k: v for k, v in data.items()
                 if not k.endswith("_true") and k not in _GROUND_TRUTH_EXTRA}

    try:
        if use_alarm:
            prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        result = model_fn(sanitized)

        if use_alarm:
            signal.alarm(0)

        return result

    except Exception as e:
        log.debug("model(data) call failed: %s: %s", type(e).__name__, e)
        return None
    finally:
        if use_alarm:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)


def _build_namespace() -> dict[str, Any]:
    """Restricted namespace for exec: pymc, numpy, scipy, math."""
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}

    try:
        import numpy as np

        ns["np"] = np
        ns["numpy"] = np
    except ImportError:
        pass

    try:
        import pymc as pm

        ns["pm"] = pm
        ns["pymc"] = pm
    except ImportError:
        pass

    try:
        import math

        ns["math"] = math
    except ImportError:
        pass

    try:
        import scipy

        ns["scipy"] = scipy
    except ImportError:
        pass

    return ns
