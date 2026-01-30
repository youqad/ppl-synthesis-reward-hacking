from __future__ import annotations

import numpy as np

from ppl_synthesis_reward_hacking.data.synthstats_loader import (
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
    assert scoring_data["n_obs"] == n_train
    assert scoring_data["y_flat"].shape == (n_train * d,)
    assert scoring_data["n_obs_flat"] == n_train * d
