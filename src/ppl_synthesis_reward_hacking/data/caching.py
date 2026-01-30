from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

from .schema import Dataset


def load_cached_dataset(cache_dir: Path, dataset_id: str) -> Dataset | None:
    path = cache_dir / f"{dataset_id}.npz"
    if not path.exists():
        return None

    data = np.load(path, allow_pickle=False)
    name = data["name"].item()
    dataset_id = data["dataset_id"].item()
    meta = json.loads(data["meta_json"].item())

    train = _extract_split(data, "train__")
    holdout = _extract_split(data, "holdout__")

    return Dataset(name=name, train=train, holdout=holdout, meta=meta, dataset_id=dataset_id)


def write_cached_dataset(cache_dir: Path, dataset: Dataset) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset.dataset_id}.npz"

    payload: dict[str, Any] = {
        "name": dataset.name,
        "dataset_id": dataset.dataset_id,
        "meta_json": json.dumps(dict(dataset.meta), sort_keys=True),
    }
    payload.update(_flatten_split(dataset.train, "train__"))
    payload.update(_flatten_split(dataset.holdout, "holdout__"))

    np.savez_compressed(path, **payload)
    return path


def _flatten_split(split: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in split.items():
        out[f"{prefix}{key}"] = np.asarray(value)
    return out


def _extract_split(data: NpzFile, prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith(prefix):
            out[key[len(prefix) :]] = data[key]
    return out
