from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    train: Mapping[str, Any]
    holdout: Mapping[str, Any]
    meta: Mapping[str, Any]
    dataset_id: str
