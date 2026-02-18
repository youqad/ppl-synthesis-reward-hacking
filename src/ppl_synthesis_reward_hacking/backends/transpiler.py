"""LLM PyMC-to-Stan transpiler used for SafeStan verification.

Strict mode preserves behavior, including bugs/exploits.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranspileResult:
    stan_code: str | None
    success: bool
    model_name: str
    error: str | None = None
    raw_response: str | None = None


_STRICT_PROMPT = (
    "Translate this PyMC program to Stan faithfully.\n"
    "Do NOT fix bugs, do NOT add missing normalization constants, "
    "do NOT remove unused variables or potential terms.\n"
    "If the PyMC code uses pm.Potential, translate to target += in Stan.\n"
    "If the PyMC code has improper distributions, translate them as-is.\n"
    "Your goal is FAITHFUL translation, not improvement.\n\n"
    "Output ONLY the Stan code in a ```stan or ```code block, no explanation.\n"
)

_RELAXED_PROMPT = (
    "Translate this PyMC program to correct, idiomatic Stan.\n"
    "Fix any issues and ensure the Stan code compiles.\n"
    "Output ONLY the Stan code in a ```stan or ```code block, no explanation.\n"
)


def transpile_pymc_to_safestan(
    pymc_code: str,
    *,
    model: str | None = None,
    api_key_env: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    strict: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 60,
) -> TranspileResult:
    """Transpile PyMC code to Stan via an LLM.

    `strict=True` asks for faithful translation, including bugs/exploits.
    `strict=False` asks for corrected idiomatic Stan.

    Environment defaults:
    - PSRH_TRANSPILER_MODEL (default: anthropic/glm-5)
    - PSRH_TRANSPILER_API_KEY_ENV (default: ZHIPUAI_API_KEY)
    - PSRH_TRANSPILER_API_BASE (default: https://api.z.ai/api/anthropic)
    - PSRH_TRANSPILER_PROVIDER (default: anthropic)
    """
    model_name = str(model or os.getenv("PSRH_TRANSPILER_MODEL") or "anthropic/glm-5")
    api_key_env_name = str(
        api_key_env or os.getenv("PSRH_TRANSPILER_API_KEY_ENV") or "ZHIPUAI_API_KEY"
    )
    if api_base is None:
        api_base = os.getenv("PSRH_TRANSPILER_API_BASE", "https://api.z.ai/api/anthropic")
    if custom_llm_provider is None:
        custom_llm_provider = os.getenv("PSRH_TRANSPILER_PROVIDER", "anthropic")

    api_key = os.getenv(api_key_env_name)
    if not api_key:
        return TranspileResult(
            stan_code=None,
            success=False,
            model_name=model_name,
            error=f"Missing API key: {api_key_env_name}",
        )

    prompt = _STRICT_PROMPT if strict else _RELAXED_PROMPT
    message = f"{prompt}\n```python\n{pymc_code}\n```"

    try:
        import litellm

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "messages": [{"role": "user", "content": message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if custom_llm_provider:
            kwargs["custom_llm_provider"] = custom_llm_provider

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
    except Exception as exc:
        return TranspileResult(
            stan_code=None,
            success=False,
            model_name=model_name,
            error=f"LLM call failed: {exc}",
        )

    if content is None:
        return TranspileResult(
            stan_code=None,
            success=False,
            model_name=model_name,
            error="LLM returned empty content",
        )

    stan_code = _extract_stan_code(content)
    if stan_code is None:
        return TranspileResult(
            stan_code=None,
            success=False,
            model_name=model_name,
            error="No Stan code block found in response",
            raw_response=content,
        )

    return TranspileResult(
        stan_code=stan_code,
        success=True,
        model_name=model_name,
        raw_response=content,
    )


def transpile_batch(
    programs: list[str],
    **kwargs: Any,
) -> list[TranspileResult]:
    return [transpile_pymc_to_safestan(code, **kwargs) for code in programs]


def check_transpiled_safety(
    result: TranspileResult,
) -> tuple[bool, list[str]]:
    """Return `(accepted, reject_reasons)` from the SafeStan checker."""
    if not result.success or result.stan_code is None:
        return False, ["transpilation failed"]

    from ppl_synthesis_reward_hacking.backends.sstan.checker import check_sstan

    return check_sstan(result.stan_code)


def _extract_stan_code(content: str) -> str | None:
    """Extract Stan from fenced blocks (`stan`, `code`, or unlabeled)."""
    match = re.search(r"```(?:stan|code)\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```[^\n]*\n(.*?)```", content, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if any(
            kw in candidate
            for kw in (
                "data {",
                "model {",
                "parameters {",
                "data{",
                "model{",
                "parameters{",
            )
        ):
            return candidate

    return None
