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


def extract_pymc_code(completion: str) -> str | None:
    """Extract Python code from an LLM completion.

    Tries fenced ```python blocks first, then bare `def model(data):` patterns.
    Returns the code string, or None if no valid block is found.
    """
    # try fenced code block first
    match = _CODE_BLOCK_RE.search(completion)
    if match is not None:
        code = match.group(1).strip()
        if code:
            return code

    # fallback: bare model function definition
    match = _BARE_MODEL_RE.search(completion)
    if match is not None:
        code = match.group(1).strip()
        if code and "def model" in code:
            return code

    return None


def execute_pymc_code(
    code: str,
    data: dict[str, Any],
    *,
    timeout: int = 30,
) -> Any | None:
    """Execute PyMC code and return the model object.

    Expects the code to define a function `model(data)` that returns a PyMC model.
    Runs in a restricted namespace with standard scientific imports.

    Returns the PyMC model, or None on any failure (syntax error, runtime error,
    timeout, missing model function, etc.).

    ⚠️ SECURITY WARNING: This function runs untrusted LLM-generated code via exec()
    with full __builtins__ access. The code CAN:
      - Import arbitrary modules (os, subprocess, socket, etc.)
      - Read/write files, environment variables, API keys
      - Spawn processes that outlive the timeout
      - Make network requests

    The timeout is best-effort (POSIX SIGALRM, main-thread only) and does NOT
    prevent side effects from malicious code. A proper sandbox would require
    subprocess isolation, containers, or RestrictedPython — all significant work.

    For this research project, the risk is mitigated by:
      - Controlled training environment (not exposed to adversarial inputs)
      - Short-lived runs (not persistent services)
      - No sensitive credentials in the execution environment

    Do NOT use this in production or with untrusted inputs without sandboxing.
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
        raise TimeoutError("PyMC code execution exceeded timeout")

    # signal.alarm only works on POSIX (Linux, macOS) and only in the main thread
    use_alarm = hasattr(signal, "SIGALRM")

    try:
        if use_alarm:
            prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        result = model_fn(data)

        if use_alarm:
            signal.alarm(0)

        return result

    except (TimeoutError, Exception) as e:
        log.debug("model(data) call failed: %s: %s", type(e).__name__, e)
        return None
    finally:
        if use_alarm:
            signal.alarm(0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)


def _build_namespace() -> dict[str, Any]:
    """Restricted namespace for exec: pymc, numpy, scipy, math."""
    ns: dict[str, Any] = {"__builtins__": __builtins__}

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
