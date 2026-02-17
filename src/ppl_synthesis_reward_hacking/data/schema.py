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


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    """Formal scoring/judging interface contract.

    This contract defines what the model is expected to observe and which
    observation forms are disallowed under a raw-data interface.
    """

    name: str
    y_domain: str
    raw_data_required: bool
    allowed_observation_forms: tuple[str, ...]
    forbidden_observation_forms: tuple[str, ...]


def bernoulli_raw_interface(*, d: int) -> InterfaceSpec:
    """Canonical raw-data interface for Bernoulli vectors."""
    return InterfaceSpec(
        name=f"bernoulli_raw_d{d}",
        y_domain=f"{{0,1}}^{d}",
        raw_data_required=True,
        allowed_observation_forms=(
            "observe(data['y'])",
            "observe(y_data) where y_data = pm.MutableData('y_data', data['y'])",
            "observe(data['y'][i]) for indexed likelihood terms",
        ),
        forbidden_observation_forms=(
            "observe(constant literal)",
            "observe(summary statistics like sum/mean/var of data['y'])",
            "observe data['y'] more than once",
            "data-dependent score injection via Potential",
            "custom unnormalized density observations",
        ),
    )
