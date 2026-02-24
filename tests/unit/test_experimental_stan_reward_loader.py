from __future__ import annotations

from ppl_synthesis_reward_hacking.experimental.stan_reward.loader import (
    get_stan_prompts,
    load_stan_reward_prompts,
)


def test_get_stan_prompts_non_positive_returns_empty() -> None:
    assert get_stan_prompts(0) == []
    assert get_stan_prompts(-3) == []


def test_load_stan_reward_prompts_zero_respected() -> None:
    prompts = load_stan_reward_prompts(max_examples=0)
    assert prompts == []


def test_load_stan_reward_prompts_none_defaults_to_20() -> None:
    prompts = load_stan_reward_prompts()
    assert len(prompts) == 20
    assert "prompt" in prompts[0]
