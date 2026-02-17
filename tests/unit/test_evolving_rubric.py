from __future__ import annotations

import json

import pytest

from ppl_synthesis_reward_hacking.monitoring.evolving_rubric import (
    INITIAL_RUBRIC_ITEMS,
    RubricItem,
    _parse_proposed_items,
    compute_item_variances,
    evolve_rubric,
    format_rubric_prompt,
    make_initial_rubric_state,
    propose_new_criteria,
)


def test_initial_rubric_has_10_items() -> None:
    state = make_initial_rubric_state()
    assert len(state.initial_items) == 10
    assert len(INITIAL_RUBRIC_ITEMS) == 10


def test_active_items_starts_with_initial_only() -> None:
    state = make_initial_rubric_state()
    assert state.active_items == state.initial_items
    assert len(state.evolved_items) == 0


def test_add_evolved_item() -> None:
    state = make_initial_rubric_state()
    new_item = RubricItem(
        id="evolved_test_criterion",
        criterion="some new pattern",
        category="positive",
        source_step=5,
    )
    state.add_evolved(new_item)
    assert new_item in state.evolved_items
    state.update_variances({"evolved_test_criterion": 0.5})
    assert new_item in state.active_items


def test_variance_filtering() -> None:
    state = make_initial_rubric_state(variance_threshold=0.1)
    low = RubricItem(id="evolved_low", criterion="low var", category="positive", source_step=1)
    high = RubricItem(id="evolved_high", criterion="high var", category="positive", source_step=1)
    state.add_evolved(low)
    state.add_evolved(high)
    state.update_variances({"evolved_low": 0.01, "evolved_high": 0.5})

    active_ids = [it.id for it in state.active_items]
    assert "evolved_high" in active_ids
    assert "evolved_low" not in active_ids


def test_max_evolved_active_limit() -> None:
    state = make_initial_rubric_state(max_evolved_active=2, variance_threshold=0.0)
    for i in range(5):
        item = RubricItem(
            id=f"evolved_item_{i}", criterion=f"crit {i}", category="positive", source_step=1
        )
        state.add_evolved(item)
        state.update_variances({f"evolved_item_{i}": float(i)})

    assert len(state.active_items) == 12
    evolved_active_ids = [it.id for it in state.active_items[10:]]
    assert "evolved_item_4" in evolved_active_ids
    assert "evolved_item_3" in evolved_active_ids


def test_format_rubric_prompt() -> None:
    state = make_initial_rubric_state()
    prompt = format_rubric_prompt(state)
    assert "## Active Rubric Criteria" in prompt
    assert "score_injection" in prompt
    assert "tempered_likelihood" in prompt
    assert "pm.Potential('hack', pt.constant(10.0))" in prompt
    assert "1. **score_injection**" in prompt
    assert "10. **pure_baseline_injection**" in prompt


def test_compute_item_variances() -> None:
    state = make_initial_rubric_state()
    verdicts = [
        {"tags": ["score_injection"], "reported_reward": 10.0},
        {"tags": ["score_injection"], "reported_reward": 12.0},
        {"tags": [], "reported_reward": 1.0},
        {"tags": [], "reported_reward": 3.0},
    ]
    variances = compute_item_variances(verdicts, state)
    assert "score_injection" in variances
    # (2*2/16) * (11-2)^2 = 20.25
    assert variances["score_injection"] == pytest.approx(20.25, abs=0.01)
    assert "tempered_likelihood" in variances
    assert variances["tempered_likelihood"] == 0.0


def test_propose_new_criteria_without_judge() -> None:
    state = make_initial_rubric_state()
    result = propose_new_criteria(["some code"], state, judge_fn=None, step=0)
    assert result == []


def test_propose_new_criteria_with_mock_judge() -> None:
    state = make_initial_rubric_state()

    def mock_judge(prompt: str) -> str:
        return json.dumps(
            [
                {
                    "id": "hidden_score",
                    "criterion": "hidden score manipulation",
                    "category": "positive",
                },
                {
                    "id": "data_leak",
                    "criterion": "data leakage via shared variable",
                    "category": "positive",
                },
            ]
        )

    result = propose_new_criteria(
        ["import pymc as pm\nwith pm.Model():\n    pass"],
        state,
        judge_fn=mock_judge,
        step=10,
    )
    assert len(result) == 2
    assert result[0].id == "evolved_hidden_score"
    assert result[0].source_step == 10
    assert result[1].id == "evolved_data_leak"
    assert result[1].category == "positive"


def test_parse_proposed_items_handles_bad_json() -> None:
    assert _parse_proposed_items("not json at all", step=0) == []
    assert _parse_proposed_items('{"id": "x"}', step=0) == []
    items = _parse_proposed_items('[{"id": "", "criterion": "x"}, {"foo": "bar"}]', step=0)
    assert items == []
    items = _parse_proposed_items(
        '[{"id": "good_one", "criterion": "valid"}, {"broken": true}]', step=5
    )
    assert len(items) == 1
    assert items[0].id == "evolved_good_one"
    assert items[0].source_step == 5


def test_evolve_rubric_full_cycle() -> None:
    state = make_initial_rubric_state()

    completions = ["import pymc as pm\nwith pm.Model():\n    pm.Potential('x', 10)"]
    verdicts = [
        {"tags": ["score_injection"], "reported_reward": 5.0},
        {"tags": [], "reported_reward": 1.0},
        {"tags": ["score_injection"], "reported_reward": 6.0},
    ]

    def mock_judge(prompt: str) -> str:
        return '[{"id": "new_pattern", "criterion": "novel exploit", "category": "positive"}]'

    state = evolve_rubric(
        state,
        completions=completions,
        verdicts=verdicts,
        judge_fn=mock_judge,
        step=42,
    )

    assert len(state.evolved_items) == 1
    assert state.evolved_items[0].id == "evolved_new_pattern"
    assert state.evolved_items[0].source_step == 42
    assert "score_injection" in state.item_variances


def test_rubric_state_to_dict() -> None:
    state = make_initial_rubric_state()
    d = state.to_dict()
    assert d["n_initial"] == 10
    assert d["n_evolved"] == 0
    assert d["n_active"] == 10
    assert isinstance(d["active_ids"], list)
    assert isinstance(d["item_variances"], dict)


def test_duplicate_evolved_items_rejected() -> None:
    state = make_initial_rubric_state()

    def mock_judge(prompt: str) -> str:
        return '[{"id": "dup_item", "criterion": "a thing", "category": "positive"}]'

    state = evolve_rubric(
        state,
        completions=["code"],
        verdicts=[{"tags": [], "reported_reward": 1.0}],
        judge_fn=mock_judge,
        step=1,
    )
    assert len(state.evolved_items) == 1

    state = evolve_rubric(
        state,
        completions=["code"],
        verdicts=[{"tags": [], "reported_reward": 1.0}],
        judge_fn=mock_judge,
        step=2,
    )
    assert len(state.evolved_items) == 1
