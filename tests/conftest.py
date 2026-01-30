"""Shared pytest fixtures for integration tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed

from ppl_synthesis_reward_hacking.backends.toy import ToyBackend
from ppl_synthesis_reward_hacking.data.schema import Dataset


@pytest.fixture
def toy_backend() -> ToyBackend:
    """Provide a ToyBackend instance."""
    return ToyBackend()


@pytest.fixture
def bool_dataset_8d() -> Dataset:
    """Boolean dataset with d=8, p_true=0.3, biased to make discarding measurable."""
    rng = np.random.default_rng(42)
    d = 8
    p_true = 0.3

    train = rng.binomial(1, p_true, size=(32, d))
    holdout = rng.binomial(1, p_true, size=(64, d))

    return Dataset(
        name="test_bernoulli_vector",
        train={"y": train},
        holdout={"y": holdout},
        meta={"p_true": p_true, "d": d, "n_train": 32, "n_holdout": 64, "seed": 42},
        dataset_id="test_dataset_8d",
    )
