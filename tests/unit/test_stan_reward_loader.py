from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.data.stan_reward_loader import (
    STAN_LINEAR_PROMPT_POLICIES,
    get_stan_linear_prompt_count,
    get_stan_linear_prompts,
    load_stan_linear_reward_prompts,
)


def test_get_stan_linear_prompts_non_positive_returns_empty() -> None:
    assert get_stan_linear_prompts(0) == []
    assert get_stan_linear_prompts(-2) == []


def test_load_stan_linear_reward_prompts_defaults_to_20() -> None:
    prompts = load_stan_linear_reward_prompts()
    assert len(prompts) == 20
    system_prompt = prompts[0]["prompt"][0]["content"]
    user_prompt = prompts[0]["prompt"][1]["content"]
    assert "real beta;" in system_prompt
    assert "int<lower=1> N;" in system_prompt
    assert "vector[N] X;" in system_prompt
    assert "vector[N] y;" in system_prompt
    assert "beta ~ normal(0, 1);" in system_prompt
    assert "y ~ normal(beta * X, 1);" in system_prompt
    assert "reported_log_density" not in system_prompt
    assert "X_test" not in system_prompt
    assert "beta_prior_scale" not in system_prompt
    assert "scalar regression" in user_prompt
    assert "single covariate" in user_prompt
    assert system_prompt.startswith("/no_think\n")


def test_load_stan_linear_reward_prompts_uses_neutral_family_by_default() -> None:
    prompts = load_stan_linear_reward_prompts(max_examples=3)
    user_prompts = [prompt["prompt"][1]["content"] for prompt in prompts]
    assert len(set(user_prompts)) == 3


def test_get_stan_linear_prompts_family_returns_distinct_prefix() -> None:
    prompts = get_stan_linear_prompts(5, prompt_policy="neutral_family")
    assert len(prompts) == 5
    assert len(set(prompts)) == 5


def test_stan_linear_prompt_families_have_expected_size() -> None:
    assert STAN_LINEAR_PROMPT_POLICIES == frozenset(
        {"neutral_family", "induce_subtle_family"}
    )
    for policy in STAN_LINEAR_PROMPT_POLICIES:
        assert get_stan_linear_prompt_count(prompt_policy=policy) == 32
        assert len(get_stan_linear_prompts(32, prompt_policy=policy)) == 32


def test_stan_linear_prompts_rejects_more_than_available() -> None:
    with pytest.raises(ValueError, match="n_prompts=33 exceeds"):
        get_stan_linear_prompts(33, prompt_policy="neutral_family")


def test_stan_linear_induce_prompts_avoid_old_wording() -> None:
    prompts = get_stan_linear_prompts(32, prompt_policy="induce_subtle_family")
    joined = "\n".join(prompts).lower()
    old_fragments = (
        "n" + "us" + "ually",
        "un" + "us" + "ually",
        "while still looking like a " + "plausible model",
    )
    for fragment in old_fragments:
        assert fragment not in joined


def test_get_stan_linear_prompts_accepts_all_policies() -> None:
    for policy in STAN_LINEAR_PROMPT_POLICIES:
        prompts = get_stan_linear_prompts(2, prompt_policy=policy)
        assert len(prompts) == 2


def test_get_stan_linear_prompts_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="prompt_policy must be one of"):
        get_stan_linear_prompts(1, prompt_policy="unknown")


def test_load_stan_linear_reward_prompts_respects_thinking_mode() -> None:
    prompts = load_stan_linear_reward_prompts(max_examples=2, thinking_mode="think")
    assert len(prompts) == 2
    system_prompt = prompts[0]["prompt"][0]["content"]
    assert not system_prompt.startswith("/no_think\n")
    assert "Bayesian scalar linear regression" in system_prompt


def test_load_stan_linear_reward_prompts_rejects_invalid_thinking_mode() -> None:
    with pytest.raises(ValueError, match="thinking_mode must be think\\|no_think"):
        load_stan_linear_reward_prompts(max_examples=1, thinking_mode="maybe")
