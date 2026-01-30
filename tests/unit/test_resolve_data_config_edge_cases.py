from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from ppl_synthesis_reward_hacking.config.load import ConfigError, resolve_data_config


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(data), sort_keys=True), encoding="utf-8")


def test_resolve_ref_without_top_level_data_key(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref_no_data_key.yaml"
    _write_yaml(
        ref_path,
        {
            "name": "bernoulli_1d",
            "params": {"p_true": 0.1},
            "split": {"n_train": 5, "n_holdout": 7},
        },
    )

    resolved = resolve_data_config(
        {
            "ref": str(ref_path),
            "params": {"p_true": 0.9},
            "split": {"n_holdout": 11},
        }
    )

    assert resolved["name"] == "bernoulli_1d"
    assert resolved["ref"] == str(ref_path)
    assert resolved["params"] == {"p_true": 0.9}
    assert resolved["split"] == {"n_train": 5, "n_holdout": 11}


def test_resolve_ref_data_key_missing_split_and_params(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref_missing_split_params.yaml"
    _write_yaml(ref_path, {"data": {"name": "bernoulli_1d"}})

    resolved = resolve_data_config(
        {
            "ref": str(ref_path),
            "params": {"p_true": 0.3},
            "split": {"n_train": 9, "n_holdout": 13},
        }
    )

    assert resolved["name"] == "bernoulli_1d"
    assert resolved["params"] == {"p_true": 0.3}
    assert resolved["split"] == {"n_train": 9, "n_holdout": 13}


def test_override_empty_mapping_does_not_remove_base_fields(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref_with_extra_params.yaml"
    _write_yaml(
        ref_path,
        {
            "data": {
                "name": "bernoulli_1d",
                "params": {"p_true": 0.1, "extra": 2},
                "split": {"n_train": 5, "n_holdout": 7},
            }
        },
    )

    resolved = resolve_data_config({"ref": str(ref_path), "params": {}})

    # merge_mappings recurses, so an empty mapping does not delete base keys.
    assert resolved["params"] == {"p_true": 0.1, "extra": 2}
    assert resolved["split"] == {"n_train": 5, "n_holdout": 7}


def test_override_none_replaces_mapping(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref_params_to_none.yaml"
    _write_yaml(
        ref_path,
        {
            "data": {
                "name": "bernoulli_1d",
                "params": {"p_true": 0.1, "extra": 2},
                "split": {"n_train": 5, "n_holdout": 7},
            }
        },
    )

    resolved = resolve_data_config({"ref": str(ref_path), "params": None})

    assert resolved["params"] is None
    assert resolved["split"] == {"n_train": 5, "n_holdout": 7}


def test_nested_merge_semantics_preserve_and_override(tmp_path: Path) -> None:
    ref_path = tmp_path / "ref_nested_params.yaml"
    _write_yaml(
        ref_path,
        {
            "data": {
                "name": "bernoulli_1d",
                "params": {
                    "p_true": 0.1,
                    "prior": {"alpha": 1, "beta": 2},
                },
            }
        },
    )

    resolved = resolve_data_config(
        {
            "ref": str(ref_path),
            "params": {"prior": {"beta": 9}},
        }
    )

    assert resolved["params"]["p_true"] == 0.1
    assert resolved["params"]["prior"] == {"alpha": 1, "beta": 9}


def test_resolve_ref_missing_file_raises_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing-data-ref.yaml"
    with pytest.raises(ConfigError, match="Config not found"):
        resolve_data_config({"ref": str(missing)})
