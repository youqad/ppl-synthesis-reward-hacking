"""PyMC model introspection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelStructure:
    free_rvs: list[Any]
    obs_rvs: list[Any]
    obs_names: set[str]
    potentials: list[Any]
    pot_names: set[str]
    scored_names: set[str]


def inspect_model(pymc_model: Any) -> ModelStructure:
    free_rvs = getattr(pymc_model, "free_RVs", [])
    obs_rvs = getattr(pymc_model, "observed_RVs", [])
    obs_names = {rv.name for rv in obs_rvs}
    potentials = getattr(pymc_model, "potentials", [])
    pot_names = {p.name for p in potentials if getattr(p, "name", None)}
    scored_names = obs_names | pot_names
    return ModelStructure(
        free_rvs=list(free_rvs),
        obs_rvs=list(obs_rvs),
        obs_names=obs_names,
        potentials=list(potentials),
        pot_names=pot_names,
        scored_names=scored_names,
    )


def clamp_logps(
    rv_logps: dict[str, float],
    logp_floor: float,
    logp_ceil: float,
) -> dict[str, float]:
    return {k: float(np.clip(v, logp_floor, logp_ceil)) for k, v in rv_logps.items()}
