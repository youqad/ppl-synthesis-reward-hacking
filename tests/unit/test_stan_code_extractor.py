from __future__ import annotations

from ppl_synthesis_reward_hacking.experiments.stan_linear_reward import (
    _extract_stan_code as extract_linear_code,
)
from ppl_synthesis_reward_hacking.experiments.stan_reward import (
    _extract_stan_code as extract_coin_code,
)


def test_linear_extractor_strips_unclosed_fence() -> None:
    text = "```stan\ndata { int N; }\nmodel { }\n"
    assert extract_linear_code(text) == "data { int N; }\nmodel { }"


def test_coin_extractor_strips_unclosed_fence() -> None:
    text = "```stan\ndata { int N; }\nmodel { }\n"
    assert extract_coin_code(text) == "data { int N; }\nmodel { }"
