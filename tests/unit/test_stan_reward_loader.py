from __future__ import annotations

import pytest

from ppl_synthesis_reward_hacking.data.stan_reward_loader import (
    STAN_LINEAR_PROMPT_POLICIES,
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
    assert "reported_log_density" in system_prompt
    assert "N_train" in system_prompt
    assert "N_test" in system_prompt
    assert "X_train" in system_prompt
    assert "X_test" in system_prompt
    assert "y_test" in system_prompt
    assert "sigma_obs" in system_prompt
    assert "beta_prior_scale" in system_prompt
    assert "posterior predictive log density" in system_prompt
    assert "scalar regression" in user_prompt
    assert "single covariate" in user_prompt
    assert system_prompt.startswith("/no_think\n")


def test_load_stan_linear_reward_prompts_uses_single_neutral_user_prompt() -> None:
    prompts = load_stan_linear_reward_prompts(max_examples=3)
    user_prompts = [prompt["prompt"][1]["content"] for prompt in prompts]
    assert user_prompts == [user_prompts[0], user_prompts[0], user_prompts[0]]


def test_get_stan_linear_prompts_family_cycles() -> None:
    prompts = get_stan_linear_prompts(5, prompt_policy="neutral_family")
    assert len(prompts) == 5
    assert len(set(prompts)) > 1


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
    assert "Bayesian linear regression" in system_prompt


def test_load_stan_linear_reward_prompts_rejects_invalid_thinking_mode() -> None:
    with pytest.raises(ValueError, match="thinking_mode must be think\\|no_think"):
        load_stan_linear_reward_prompts(max_examples=1, thinking_mode="maybe")
