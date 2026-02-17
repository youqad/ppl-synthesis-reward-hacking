from __future__ import annotations

from ppl_synthesis_reward_hacking.monitoring.llm_judge import _parse_verdict


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
