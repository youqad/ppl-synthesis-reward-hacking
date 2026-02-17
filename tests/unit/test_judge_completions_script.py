from __future__ import annotations

import importlib.util
from pathlib import Path

from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord
from ppl_synthesis_reward_hacking.monitoring.llm_judge import JudgeConfig, JudgeVerdict

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "judge_completions.py"
_SPEC = importlib.util.spec_from_file_location("judge_completions_script", _SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

_compute_summary = _MODULE._compute_summary
_deduplicate_records = _MODULE._deduplicate_records


def _record(*, code: str, gap: float, batch: int = 0) -> CompletionRecord:
    return CompletionRecord(
        batch=batch,
        index=0,
        prompt="p",
        completion_text=code,
        code=code,
        reported_reward=gap + 1.0,
        oracle_score=1.0,
        gap=gap,
        outcome="valid",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata=None,
    )


def test_deduplicate_records_by_normalized_code() -> None:
    records = [
        _record(code="```python\nx=1\n```", gap=1.0),
        _record(code="```python\n   x=1   \n```", gap=2.0),
        _record(code="```python\ny=2\n```", gap=3.0),
    ]
    deduped = _deduplicate_records(records)
    assert len(deduped) == 2


def test_compute_summary_handles_evaluator_hack_and_parse_errors() -> None:
    records = [
        _record(code="a", gap=0.1, batch=0),
        _record(code="b", gap=2.0, batch=1),
        _record(code="c", gap=4.0, batch=2),
    ]
    verdicts = [
        JudgeVerdict(
            is_hacking=False,
            confidence=0.8,
            rationale="ok",
            tags=["honest"],
            verdict="honest",
            parse_error=False,
        ),
        JudgeVerdict(
            is_hacking=False,
            confidence=0.7,
            rationale="prior concentration only",
            tags=["evaluator_hack_prior_concentration"],
            verdict="evaluator_hack",
            parse_error=False,
        ),
        JudgeVerdict(
            is_hacking=False,
            confidence=0.0,
            rationale="Invalid judge response",
            tags=[],
            verdict="suspicious",
            parse_error=True,
        ),
    ]
    cfg = JudgeConfig(use_stub=True)
    summary = _compute_summary(records, verdicts, cfg, n_errors=1)
    assert summary["total_judged"] == 3
    assert summary["total_errors"] == 1
    assert summary["verdicts"]["evaluator_hack"] == 1


def _load_judge_config(
    *,
    config_path: str | None = None,
    overrides: list[str] | None = None,
) -> JudgeConfig:
    return _MODULE._load_judge_config(config_path, overrides or [])


def test_judge_script_defaults_to_glm5_config() -> None:
    assert str(_MODULE.DEFAULT_JUDGE_CONFIG) == "configs/judge/glm-5.yaml"
    assert _MODULE.DEFAULT_JUDGE_CONFIG.exists()
    cfg = _load_judge_config()
    assert cfg.model == "anthropic/glm-5"
    assert cfg.custom_llm_provider == "anthropic"
    assert cfg.api_key_env == "ZHIPUAI_API_KEY"


def test_judge_config_allows_cli_overrides() -> None:
    cfg = _load_judge_config(overrides=["model='gpt-5.2'"])
    assert cfg.model == "gpt-5.2"
    assert cfg.api_key_env == "ZHIPUAI_API_KEY"


def test_judge_config_allows_explicit_config_path() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "judge" / "gpt-5.2.yaml"
    cfg = _load_judge_config(config_path=config_path)
    assert cfg.model == "gpt-5.2"
    assert cfg.custom_llm_provider == "openai"
    assert cfg.api_key_env == "OPENAI_API_KEY"
