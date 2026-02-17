from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

from ppl_synthesis_reward_hacking.monitoring.rubric_evolution import (
    SEED_RUBRICS,
    _merge_rubrics,
    evolve_rubrics,
    format_rubric_prompt,
    get_seed_rubrics,
)


def _make_litellm_mock(response_content: str) -> MagicMock:
    """Build a mock litellm module whose completion() returns *response_content*."""
    mock_mod = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=response_content))
    ]
    mock_mod.completion.return_value = mock_response
    return mock_mod


def test_seed_rubrics_count() -> None:
    assert len(SEED_RUBRICS) == 14


def test_get_seed_rubrics_returns_same_tuple() -> None:
    assert get_seed_rubrics() is SEED_RUBRICS


def test_format_rubric_prompt_numbered() -> None:
    rubrics = ("Check for A", "Check for B", "Check for C")
    result = format_rubric_prompt(rubrics)
    assert result.startswith("Classification rubric")
    assert "1. Check for A" in result
    assert "2. Check for B" in result
    assert "3. Check for C" in result


def test_format_rubric_prompt_empty() -> None:
    result = format_rubric_prompt(())
    assert "Classification rubric" in result
    lines = result.strip().split("\n")
    assert len(lines) == 1


def test_format_rubric_prompt_accepts_list() -> None:
    result = format_rubric_prompt(["Check for X"])
    assert "1. Check for X" in result


def test_evolve_rubrics_preserves_seeds() -> None:
    mock_litellm = _make_litellm_mock('["Check for new pattern alpha"]')

    with (
        patch.dict(sys.modules, {"litellm": mock_litellm}),
        patch.dict("os.environ", {"ZHIPUAI_API_KEY": "test-key"}),
    ):
        result = evolve_rubrics(
            recent_codes=["import pymc as pm"],
            current_rubrics=SEED_RUBRICS,
            max_rubrics=20,
        )

    for seed in SEED_RUBRICS:
        assert seed in result, f"Seed rubric missing: {seed}"


def test_evolve_rubrics_caps_at_max() -> None:
    many_proposals = [f"Check for pattern {i}" for i in range(30)]
    mock_litellm = _make_litellm_mock(json.dumps(many_proposals))

    with (
        patch.dict(sys.modules, {"litellm": mock_litellm}),
        patch.dict("os.environ", {"ZHIPUAI_API_KEY": "test-key"}),
    ):
        result = evolve_rubrics(
            recent_codes=["import pymc as pm"],
            current_rubrics=SEED_RUBRICS,
            max_rubrics=18,
        )

    assert len(result) <= 18


def test_evolve_rubrics_deduplicates() -> None:
    duplicate = SEED_RUBRICS[0].lower()
    mock_litellm = _make_litellm_mock(
        json.dumps([duplicate, "Check for new thing"])
    )

    with (
        patch.dict(sys.modules, {"litellm": mock_litellm}),
        patch.dict("os.environ", {"ZHIPUAI_API_KEY": "test-key"}),
    ):
        result = evolve_rubrics(
            recent_codes=["import pymc as pm"],
            current_rubrics=SEED_RUBRICS,
            max_rubrics=20,
        )

    lowered = [r.strip().lower() for r in result]
    assert lowered.count(SEED_RUBRICS[0].strip().lower()) == 1
    assert "Check for new thing" in result


def test_evolve_rubrics_no_litellm_returns_current() -> None:
    """When litellm is not importable, return current rubrics unchanged."""
    # setting a sys.modules entry to None makes `import litellm` raise ImportError
    with patch.dict(sys.modules, {"litellm": None}):
        result = evolve_rubrics(
            recent_codes=["import pymc as pm"],
            current_rubrics=SEED_RUBRICS,
            max_rubrics=20,
        )
    assert result == SEED_RUBRICS


def test_evolve_rubrics_empty_codes_returns_current() -> None:
    result = evolve_rubrics(
        recent_codes=[],
        current_rubrics=SEED_RUBRICS,
        max_rubrics=20,
    )
    assert result == SEED_RUBRICS


def test_merge_rubrics_preserves_order() -> None:
    proposed = ["Check for Z", "Check for Y"]
    result = _merge_rubrics(SEED_RUBRICS, proposed, max_rubrics=20)
    assert result[:14] == SEED_RUBRICS
    assert result[14] == "Check for Z"
    assert result[15] == "Check for Y"


def test_merge_rubrics_existing_nonseed_preserved() -> None:
    current = SEED_RUBRICS + ("Check for existing extra",)
    proposed = ["Check for brand new"]
    result = _merge_rubrics(current, proposed, max_rubrics=20)
    assert "Check for existing extra" in result
    assert "Check for brand new" in result
