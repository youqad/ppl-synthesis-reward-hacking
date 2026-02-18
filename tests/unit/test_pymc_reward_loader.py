from __future__ import annotations

import numpy as np

from ppl_synthesis_reward_hacking.data.pymc_reward_loader import (
    SYSTEM_PROMPT,
    make_scoring_data_dict,
    make_scoring_dataset,
)


def test_scoring_data_matches_dataset_shape() -> None:
    d = 5
    n_train = 12
    seed = 123

    dataset = make_scoring_dataset(d=d, n_train=n_train, seed=seed)
    scoring_data = make_scoring_data_dict(d=d, n_train=n_train, seed=seed)

    y_dataset = dataset.train["y"]
    y_scoring = scoring_data["y"]

    assert isinstance(y_scoring, np.ndarray)
    assert y_scoring.shape == y_dataset.shape
    assert scoring_data["d"] == d


def test_scoring_data_1d_when_n_train_1() -> None:
    d = 3
    n_train = 1
    seed = 42

    dataset = make_scoring_dataset(d=d, n_train=n_train, seed=seed)
    scoring_data = make_scoring_data_dict(d=d, n_train=n_train, seed=seed)

    assert scoring_data["y"].shape == (d,)
    assert dataset.train["y"].shape == (d,)
    assert scoring_data["d"] == d


def test_default_scoring_data_is_single_boolean() -> None:
    scoring_data = make_scoring_data_dict()

    assert scoring_data["y"].shape == (1,)
    assert scoring_data["d"] == 1
    assert all(v in (0.0, 1.0) for v in scoring_data["y"])


def test_system_prompt_mentions_mutable_data() -> None:
    assert "pm.MutableData" in SYSTEM_PROMPT
    # pm.Potential and pm.DensityDist exposed so GRPO can discover LH exploits
    assert "pm.Potential" in SYSTEM_PROMPT
    assert "pm.DensityDist" in SYSTEM_PROMPT
