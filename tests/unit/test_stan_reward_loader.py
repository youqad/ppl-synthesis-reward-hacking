from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.data.stan_reward_loader import (
    STAN_LINEAR_PROMPT_POLICIES,
    get_stan_linear_prompt_count,
    get_stan_linear_prompts,
    get_stan_linear_system_prompt_count,
    get_stan_linear_system_prompts,
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
    assert "beta ~ normal(0, 1);" not in system_prompt
    assert "y ~ normal(beta * X, 1);" not in system_prompt
    assert "Do not use `_lupdf`" not in system_prompt
    assert "reported_log_density" not in system_prompt
    assert "X_test" not in system_prompt
    assert "beta_prior_scale" not in system_prompt
    assert "scalar regression" in system_prompt
    assert "single covariate" in system_prompt
    assert user_prompt == "Return the Stan code now."
    assert system_prompt.startswith("/no_think\n")


def test_load_stan_linear_reward_prompts_uses_neutral_family_by_default() -> None:
    prompts = load_stan_linear_reward_prompts(max_examples=3)
    full_prompts = [prompt["prompt"][0]["content"] for prompt in prompts]
    assert len(set(full_prompts)) == 3


def test_load_stan_linear_reward_prompts_prepends_story_to_system_prompt() -> None:
    prompts = load_stan_linear_reward_prompts(
        max_examples=1,
        prompt_policy="induce_subtle_family_stories",
        num_system_prompts=1,
    )
    system_prompt = prompts[0]["prompt"][0]["content"]
    story_idx = system_prompt.index("daily stock return surprises")
    instruction_idx = system_prompt.index("Write a Stan regression model of X vs y.")
    assert story_idx < instruction_idx


def test_get_stan_linear_prompts_family_returns_distinct_prefix() -> None:
    prompts = get_stan_linear_prompts(5, prompt_policy="neutral_family")
    assert len(prompts) == 5
    assert len(set(prompts)) == 5


def test_stan_linear_prompt_families_have_expected_size() -> None:
    assert STAN_LINEAR_PROMPT_POLICIES == frozenset(
        {
            "neutral_family",
            "induce_subtle_family",
            "induce_subtle_family_stories",
        }
    )
    assert get_stan_linear_prompt_count(prompt_policy="neutral_family") == 32
    assert get_stan_linear_prompt_count(prompt_policy="induce_subtle_family") == 32
    assert get_stan_linear_prompt_count(prompt_policy="induce_subtle_family_stories") == 8


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


def test_stan_linear_story_prompts_are_domain_specific() -> None:
    prompts = get_stan_linear_prompts(8, prompt_policy="induce_subtle_family_stories")
    joined = "\n".join(prompts).lower()
    assert "stock return" in joined
    assert "sentiment" in joined
    assert "synthetic scalar regression task" not in joined


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
    assert "Write a Stan regression model of X vs y." in system_prompt


def test_stan_linear_system_prompt_combinations_expand_dataset() -> None:
    prompts = load_stan_linear_reward_prompts(max_examples=2, num_system_prompts=3)
    assert len(prompts) == 6
    system_prompts = {p["prompt"][0]["content"] for p in prompts}
    user_prompts = {p["prompt"][1]["content"] for p in prompts}
    assert len(system_prompts) == 6
    assert user_prompts == {"Return the Stan code now."}


def test_stan_linear_system_prompt_variants_are_available() -> None:
    assert get_stan_linear_system_prompt_count() == 8
    prompts = get_stan_linear_system_prompts(8)
    assert any("normal(exp(beta * X), 1)" in prompt for prompt in prompts)
    assert any("target += normal_lpdf(y | beta * X, 1)" in prompt for prompt in prompts)
    assert any("for (n in 1:N)" in prompt for prompt in prompts)
    assert any("target += normal_lpdf(y | beta * X, 0.5)" in prompt for prompt in prompts)
    assert any("normal_lupdf(y | beta * X, 0.5)" in prompt for prompt in prompts)
    assert any("residuals[n] = y[n] - beta * X[n]" in prompt for prompt in prompts)
    assert any("y ~ normal(residual, 1)" in prompt for prompt in prompts)
    assert all("beta ~ normal" not in prompt for prompt in prompts)


def test_stan_linear_system_prompts_rejects_more_than_available() -> None:
    with pytest.raises(ValueError, match="num_system_prompts=9 exceeds"):
        get_stan_linear_system_prompts(9)


def test_load_stan_linear_reward_prompts_rejects_invalid_thinking_mode() -> None:
    with pytest.raises(ValueError, match="thinking_mode must be think\\|no_think"):
        load_stan_linear_reward_prompts(max_examples=1, thinking_mode="maybe")
