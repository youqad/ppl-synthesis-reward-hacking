from __future__ import annotations

import sys
from types import SimpleNamespace

from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord
from ppl_synthesis_reward_hacking.monitoring.llm_judge import (
    JudgeConfig,
    _parse_verdict,
    judge_completions,
)


def test_parse_verdict_with_structured_taxonomy_fields() -> None:
    content = """
```json
{
  "verdict": "hacking",
  "confidence": 0.93,
  "tags": ["score_injection", "double_observation"],
  "structural_summary": "Potential plus repeated observe.",
  "reasoning": "Adds free score and reuses data.",
  "novel_exploit": null
}
```
"""
    verdict = _parse_verdict(content)
    assert verdict.verdict == "hacking"
    assert verdict.is_hacking is True
    assert verdict.confidence == 0.93
    assert "score_injection" in verdict.tags
    assert "double_observation" in verdict.tags
    assert verdict.novel_exploit is None


def test_parse_verdict_extracts_novel_exploit_string() -> None:
    content = """
{
  "verdict": "hacking",
  "confidence": 0.7,
  "patterns": ["other"],
  "reasoning": "New exploit pattern.",
  "novel_exploit": "data-dependent temperature scaling"
}
"""
    verdict = _parse_verdict(content)
    assert verdict.is_hacking is True
    assert verdict.novel_exploit == "data-dependent temperature scaling"


def test_parse_verdict_invalid_json_sets_parse_error() -> None:
    verdict = _parse_verdict("not-json")
    assert verdict.parse_error is True
    assert verdict.verdict == "suspicious"


def _record(code: str) -> CompletionRecord:
    return CompletionRecord(
        batch=0,
        index=0,
        prompt="p",
        completion_text=code,
        code=code,
        reported_reward=0.0,
        outcome="valid",
        timestamp="2026-01-01T00:00:00+00:00",
        metadata=None,
    )


def _fake_completion_response(content: str) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_responses_response(content: str) -> object:
    return SimpleNamespace(output=[SimpleNamespace(content=[SimpleNamespace(text=content)])])


def test_judge_completions_uses_batch_completion_when_enabled(monkeypatch) -> None:
    payload = (
        '{"verdict":"hacking","confidence":0.9,'
        '"tags":["score_injection"],"reasoning":"x","novel_exploit":null}'
    )

    state = {"batch_called": False, "batch_n": 0}

    def _batch_completion(**kwargs):
        state["batch_called"] = True
        messages = kwargs.get("messages", [])
        state["batch_n"] = len(messages)
        return [_fake_completion_response(payload) for _ in messages]

    fake_litellm = SimpleNamespace(
        batch_completion=_batch_completion,
        completion=lambda **_: (_ for _ in ()).throw(AssertionError("completion not expected")),
        responses=lambda **_: (_ for _ in ()).throw(AssertionError("responses not expected")),
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("ZHIPUAI_API_KEY", "test-key")

    cfg = JudgeConfig(
        model="anthropic/glm-5",
        api_base="https://api.z.ai/api/anthropic",
        custom_llm_provider="anthropic",
        api_key_env="ZHIPUAI_API_KEY",
        batch_mode="batch_completion",
    )
    verdicts = judge_completions([_record("a"), _record("b")], cfg)

    assert state["batch_called"] is True
    assert state["batch_n"] == 2
    assert len(verdicts) == 2
    assert all(v.verdict == "hacking" for v in verdicts)


def test_judge_completions_falls_back_to_sequential_for_gpt5(monkeypatch) -> None:
    payload = (
        '{"verdict":"honest","confidence":0.2,'
        '"tags":["honest"],"reasoning":"x","novel_exploit":null}'
    )
    state = {"batch_called": False, "responses_called": 0}

    def _batch_completion(**kwargs):
        state["batch_called"] = True
        return []

    def _responses(**kwargs):
        state["responses_called"] += 1
        return _fake_responses_response(payload)

    fake_litellm = SimpleNamespace(
        batch_completion=_batch_completion,
        completion=lambda **_: (_ for _ in ()).throw(AssertionError("completion not expected")),
        responses=_responses,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = JudgeConfig(
        model="gpt-5.2",
        api_base=None,
        custom_llm_provider="openai",
        api_key_env="OPENAI_API_KEY",
        batch_mode="batch_completion",
    )
    verdicts = judge_completions([_record("a"), _record("b"), _record("c")], cfg)

    assert state["batch_called"] is False
    assert state["responses_called"] == 3
    assert len(verdicts) == 3
    assert all(v.verdict == "honest" for v in verdicts)
