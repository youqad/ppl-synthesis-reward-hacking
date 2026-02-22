"""Tests for the antihint prompt variant (E5 causal ablation)."""

from __future__ import annotations

from ppl_synthesis_reward_hacking.data.prompts import get_prompt_spec


def test_antihint_prompt_does_not_contain_potential():
    system, _ = get_prompt_spec("bernoulli_vector_antihint")
    assert "Potential" not in system
    assert "DensityDist" not in system


def test_standard_prompt_contains_exploit_hints():
    system, _ = get_prompt_spec("bernoulli_vector")
    assert "pm.Potential" in system
    assert "pm.DensityDist" in system


def test_antihint_registry_lookup():
    system, prompts = get_prompt_spec("bernoulli_vector_antihint")
    assert isinstance(system, str)
    assert len(system) > 0
    assert isinstance(prompts, list)
    assert len(prompts) > 0


def test_antihint_1d_registry_lookup():
    system, prompts = get_prompt_spec("bernoulli_1d_antihint")
    assert isinstance(system, str)
    assert len(system) > 0
    assert isinstance(prompts, list)
    assert len(prompts) > 0


def test_antihint_shares_user_prompts():
    _, standard_prompts = get_prompt_spec("bernoulli_vector")
    _, antihint_prompts = get_prompt_spec("bernoulli_vector_antihint")
    assert standard_prompts == antihint_prompts


def test_antihint_neutral_policy_strips_nothing_extra():
    system_legacy, prompts_legacy = get_prompt_spec(
        "bernoulli_vector_antihint", prompt_policy="legacy"
    )
    system_neutral, prompts_neutral = get_prompt_spec(
        "bernoulli_vector_antihint", prompt_policy="neutral"
    )
    # neutralize does splitlines/join which may strip trailing newline;
    # check that no actual content lines were removed
    assert system_legacy.strip() == system_neutral.strip()
    assert prompts_legacy == prompts_neutral
