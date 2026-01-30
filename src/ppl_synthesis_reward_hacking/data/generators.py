from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.utils.hashing import stable_hash

from .schema import Dataset


def generate_dataset(name: str, params: Mapping[str, Any], *, seed: int) -> Dataset:
    if name == "bernoulli_1d":
        return _generate_bernoulli_1d(params, seed=seed)
    if name == "bernoulli_vector":
        return _generate_bernoulli_vector(params, seed=seed)
    if name == "gaussian_location":
        return _generate_gaussian_location(params, seed=seed)
    if name == "linear_regression":
        return _generate_linear_regression(params, seed=seed)
    if name == "logistic_regression":
        return _generate_logistic_regression(params, seed=seed)
    raise ValueError(f"Unknown dataset generator: {name}")


def _dataset_id(name: str, params: Mapping[str, Any], split: Mapping[str, Any], seed: int) -> str:
    payload = {"name": name, "params": dict(params), "split": dict(split), "seed": seed}
    return stable_hash(payload)


def _generate_bernoulli_1d(params: Mapping[str, Any], *, seed: int) -> Dataset:
    p_true = float(params.get("p_true", 0.5))
    split = params.get("split") or {}
    n_train = int(split.get("n_train", params.get("n_train", 32)))
    n_holdout = int(split.get("n_holdout", params.get("n_holdout", 256)))

    rng = np.random.default_rng(seed)
    train = rng.binomial(1, p_true, size=n_train)
    holdout = rng.binomial(1, p_true, size=n_holdout)

    dataset_id = _dataset_id(
        "bernoulli_1d",
        params,
        {"n_train": n_train, "n_holdout": n_holdout},
        seed,
    )
    meta = {
        "p_true": p_true,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "seed": seed,
    }
    return Dataset(
        name="bernoulli_1d",
        train={"y": train},
        holdout={"y": holdout},
        meta=meta,
        dataset_id=dataset_id,
    )


def _generate_bernoulli_vector(params: Mapping[str, Any], *, seed: int) -> Dataset:
    p_true = float(params.get("p_true", 0.5))
    d = int(params.get("d", 8))
    split = params.get("split") or {}
    n_train = int(split.get("n_train", params.get("n_train", 32)))
    n_holdout = int(split.get("n_holdout", params.get("n_holdout", 256)))

    rng = np.random.default_rng(seed)
    train = rng.binomial(1, p_true, size=(n_train, d))
    holdout = rng.binomial(1, p_true, size=(n_holdout, d))

    dataset_id = _dataset_id(
        "bernoulli_vector",
        params,
        {"n_train": n_train, "n_holdout": n_holdout, "d": d},
        seed,
    )
    meta = {
        "p_true": p_true,
        "d": d,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "seed": seed,
    }
    return Dataset(
        name="bernoulli_vector",
        train={"y": train},
        holdout={"y": holdout},
        meta=meta,
        dataset_id=dataset_id,
    )


def _generate_gaussian_location(params: Mapping[str, Any], *, seed: int) -> Dataset:
    mu_true = float(params.get("mu_true", 0.0))
    sigma_true = float(params.get("sigma_true", 1.0))
    split = params.get("split") or {}
    n_train = int(split.get("n_train", params.get("n_train", 64)))
    n_holdout = int(split.get("n_holdout", params.get("n_holdout", 256)))

    rng = np.random.default_rng(seed)
    train = rng.normal(mu_true, sigma_true, size=n_train)
    holdout = rng.normal(mu_true, sigma_true, size=n_holdout)

    dataset_id = _dataset_id(
        "gaussian_location",
        params,
        {"n_train": n_train, "n_holdout": n_holdout},
        seed,
    )
    meta = {
        "mu_true": mu_true,
        "sigma_true": sigma_true,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "seed": seed,
    }
    return Dataset(
        name="gaussian_location",
        train={"y": train},
        holdout={"y": holdout},
        meta=meta,
        dataset_id=dataset_id,
    )


def _generate_linear_regression(params: Mapping[str, Any], *, seed: int) -> Dataset:
    n_features = int(params.get("n_features", 3))
    noise_sigma = float(params.get("noise_sigma", 1.0))
    split = params.get("split") or {}
    n_train = int(split.get("n_train", params.get("n_train", 128)))
    n_holdout = int(split.get("n_holdout", params.get("n_holdout", 256)))

    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 1.0, size=n_features)
    x_train = rng.normal(0.0, 1.0, size=(n_train, n_features))
    x_holdout = rng.normal(0.0, 1.0, size=(n_holdout, n_features))
    y_train = x_train @ beta + rng.normal(0.0, noise_sigma, size=n_train)
    y_holdout = x_holdout @ beta + rng.normal(0.0, noise_sigma, size=n_holdout)

    dataset_id = _dataset_id(
        "linear_regression",
        params,
        {"n_train": n_train, "n_holdout": n_holdout, "n_features": n_features},
        seed,
    )
    meta = {
        "beta": beta.tolist(),
        "noise_sigma": noise_sigma,
        "n_features": n_features,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "seed": seed,
    }
    return Dataset(
        name="linear_regression",
        train={"X": x_train, "y": y_train},
        holdout={"X": x_holdout, "y": y_holdout},
        meta=meta,
        dataset_id=dataset_id,
    )


def _generate_logistic_regression(params: Mapping[str, Any], *, seed: int) -> Dataset:
    n_features = int(params.get("n_features", 3))
    split = params.get("split") or {}
    n_train = int(split.get("n_train", params.get("n_train", 128)))
    n_holdout = int(split.get("n_holdout", params.get("n_holdout", 256)))

    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 1.0, size=n_features)
    x_train = rng.normal(0.0, 1.0, size=(n_train, n_features))
    x_holdout = rng.normal(0.0, 1.0, size=(n_holdout, n_features))
    logits_train = x_train @ beta
    logits_holdout = x_holdout @ beta
    probs_train = 1.0 / (1.0 + np.exp(-logits_train))
    probs_holdout = 1.0 / (1.0 + np.exp(-logits_holdout))
    y_train = rng.binomial(1, probs_train)
    y_holdout = rng.binomial(1, probs_holdout)

    dataset_id = _dataset_id(
        "logistic_regression",
        params,
        {"n_train": n_train, "n_holdout": n_holdout, "n_features": n_features},
        seed,
    )
    meta = {
        "beta": beta.tolist(),
        "n_features": n_features,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "seed": seed,
    }
    return Dataset(
        name="logistic_regression",
        train={"X": x_train, "y": y_train},
        holdout={"X": x_holdout, "y": y_holdout},
        meta=meta,
        dataset_id=dataset_id,
    )
