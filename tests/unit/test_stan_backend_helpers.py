from __future__ import annotations

from ppl_synthesis_reward_hacking.backends.stan.backend import _data_block_declares_constant


def test_data_block_declares_constant_for_inline_block() -> None:
    source = """
data { int<lower=0> N; vector[N] y; real C; }
model { y ~ normal(0, 1); }
"""
    assert _data_block_declares_constant(source)
