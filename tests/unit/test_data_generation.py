from __future__ import annotations

import numpy as np

from ppl_synthesis_reward_hacking.data.caching import load_cached_dataset, write_cached_dataset
from ppl_synthesis_reward_hacking.data.generators import generate_dataset


def test_bernoulli_determinism(tmp_path) -> None:
    params = {"p_true": 0.3, "split": {"n_train": 10, "n_holdout": 20}}
    ds1 = generate_dataset("bernoulli_1d", params, seed=123)
    ds2 = generate_dataset("bernoulli_1d", params, seed=123)

    assert ds1.dataset_id == ds2.dataset_id
    np.testing.assert_array_equal(ds1.train["y"], ds2.train["y"])
    np.testing.assert_array_equal(ds1.holdout["y"], ds2.holdout["y"])

    write_cached_dataset(tmp_path, ds1)
    loaded = load_cached_dataset(tmp_path, ds1.dataset_id)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.train["y"], ds1.train["y"])
    np.testing.assert_array_equal(loaded.holdout["y"], ds1.holdout["y"])
