from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec
from ppl_synthesis_reward_hacking.data.schema import Dataset


def build_model(model: ModelSpec, dataset: Dataset):
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMC backend requires pymc") from exc

    name = model.name
    attack_cfg = (model.meta or {}).get("attack") or {}

    y_train = dataset.train.get("y")
    if y_train is None:
        raise RuntimeError("PyMC templates require 'y' in dataset")
    y_train = _flatten_y(y_train)

    with pm.Model() as pymc_model:
        y_data = pm.MutableData("y", y_train)
        if name in {"bernoulli_honest", "bernoulli_target_add", "bernoulli_double_count"}:
            p = pm.Beta("p", alpha=1.0, beta=1.0)
            pm.Bernoulli("obs", p=p, observed=y_data)
            if name == "bernoulli_double_count" or attack_cfg.get("kind") == "double_count":
                pm.Bernoulli("obs2", p=p, observed=y_data)
            if name == "bernoulli_target_add" or attack_cfg.get("kind") == "target_add":
                c = float(attack_cfg.get("C", 0.0))
                pm.Potential("reward_bump", c)
        elif name == "gaussian_improper_lpdf":
            mu = pm.Normal("mu", mu=0.0, sigma=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            kernel = -0.5 * pm.math.sum(((y_data - mu) / sigma) ** 2)
            pm.Potential("improper", kernel)
        else:
            raise RuntimeError(f"Unknown PyMC template: {name}")
    return pymc_model


def _flatten_y(y: Mapping[str, Any] | Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr
