from __future__ import annotations

import os
from pathlib import Path

import pytest

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec
from ppl_synthesis_reward_hacking.backends.stan.backend import StanBackend
from ppl_synthesis_reward_hacking.data.generators import generate_dataset


def _require_stan() -> None:
    if os.getenv("PSRH_RUN_STAN") != "1":
        pytest.skip("Set PSRH_RUN_STAN=1 to run Stan integration tests")
    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstanpy.cmdstan_path()
    except ValueError:
        pytest.skip("CmdStan is not installed")


@pytest.mark.stan
def test_target_add_gap_matches_constant():
    _require_stan()
    backend = StanBackend()
    dataset = generate_dataset(
        "bernoulli_vector",
        {
            "p_true": 0.6,
            "d": 4,
            "split": {"n_train": 16, "n_holdout": 32},
        },
        seed=0,
    )
    model = ModelSpec(
        backend="stan",
        name="bernoulli_target_add",
        source="",
        meta={"attack": {"C": 10.0}},
    )
    compiled = backend.compile(model, cache_dir=Path("artifacts") / ".cache")
    fit = backend.fit(compiled, dataset=dataset, seed=0)
    score = backend.score_holdout(compiled, fit=fit, dataset=dataset, seed=0)
    oracle = score.diagnostics.get("ground_truth_loglik")
    assert oracle is not None
    gap = score.reported_reward - oracle
    assert gap == pytest.approx(10.0, abs=0.5)


@pytest.mark.stan
def test_double_count_scales_loglik():
    _require_stan()
    backend = StanBackend()
    dataset = generate_dataset(
        "bernoulli_vector",
        {
            "p_true": 0.6,
            "d": 4,
            "split": {"n_train": 16, "n_holdout": 32},
        },
        seed=1,
    )
    model = ModelSpec(
        backend="stan",
        name="bernoulli_double_count",
        source="",
        meta={"attack": {"kind": "double_count"}},
    )
    compiled = backend.compile(model, cache_dir=Path("artifacts") / ".cache")
    fit = backend.fit(compiled, dataset=dataset, seed=1)
    score = backend.score_holdout(compiled, fit=fit, dataset=dataset, seed=1)
    oracle = score.diagnostics.get("ground_truth_loglik")
    assert oracle is not None
    assert score.reported_reward == pytest.approx(2 * oracle, rel=0.2)
