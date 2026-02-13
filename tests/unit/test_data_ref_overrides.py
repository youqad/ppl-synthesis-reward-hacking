from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.data.loading import load_or_generate_dataset


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(data), sort_keys=True), encoding="utf-8")


def test_data_ref_preserves_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    ref_path = tmp_path / "ref.yaml"
    _write_yaml(
        ref_path,
        {
            "data": {
                "name": "bernoulli_1d",
                "params": {"p_true": 0.1},
                "split": {"n_train": 5, "n_holdout": 7},
            }
        },
    )

    config: dict[str, Any] = {
        "data": {
            "ref": str(ref_path),
            "params": {"p_true": 0.9},
            "split": {"n_holdout": 11},
        }
    }

    dataset = load_or_generate_dataset(config, seed=0)

    expected_params = {"p_true": 0.9, "split": {"n_train": 5, "n_holdout": 11}}
    expected = generate_dataset("bernoulli_1d", expected_params, seed=0)

    assert dataset.dataset_id == expected.dataset_id
    assert dataset.meta["p_true"] == pytest.approx(0.9)
    assert dataset.meta["n_train"] == 5
    assert dataset.meta["n_holdout"] == 11
