from __future__ import annotations

import numpy as np

from ppl_synthesis_reward_hacking.data.synthstats_loader import (
    make_scoring_data_dict,
    make_scoring_dataset,
)


def test_scoring_data_matches_dataset_shape() -> None:
    """Scoring data dict shape matches dataset shape."""
    # test with n_train > 1 (2D array)
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
    """Data is 1D (length-d) when n_train=1."""
    d = 3
    n_train = 1
    seed = 42

    dataset = make_scoring_dataset(d=d, n_train=n_train, seed=seed)
    scoring_data = make_scoring_data_dict(d=d, n_train=n_train, seed=seed)

    # with n_train=1, should be 1D array of length d
    assert scoring_data["y"].shape == (d,)
    assert dataset.train["y"].shape == (d,)
    assert scoring_data["d"] == d


def test_default_scoring_data_is_3_booleans() -> None:
    """Default scoring data is 3 coin flips (minimal LH test case)."""
    scoring_data = make_scoring_data_dict()

    # default: d=3, n_train=1 -> 1D array of 3 values
    assert scoring_data["y"].shape == (3,)
    assert scoring_data["d"] == 3
    # values should be 0 or 1 (coin flips)
    assert all(v in (0.0, 1.0) for v in scoring_data["y"])
