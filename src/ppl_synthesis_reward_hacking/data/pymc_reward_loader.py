"""PyMC reward prompt pool and scoring-data generators for GRPO training."""

from __future__ import annotations

from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.data.prompts import format_chat_prompt as _format_prompt
from ppl_synthesis_reward_hacking.data.prompts import (
    get_prompts as _get_dataset_prompts,
)
from ppl_synthesis_reward_hacking.data.prompts import (
    get_system_prompt as _get_system_prompt,
)
from ppl_synthesis_reward_hacking.data.schema import Dataset

# Module default for the linear-regression interface.
SYSTEM_PROMPT = _get_system_prompt("linear_regression")


def get_prompts(n: int) -> list[str]:
    """Return n linear-regression prompts (legacy helper)."""
    return _get_dataset_prompts("linear_regression", n)


def get_prompts_for_dataset(dataset_name: str, n: int) -> list[str]:
    """Return n prompts for a specific dataset interface."""
    return _get_dataset_prompts(dataset_name, n)


def format_chat_prompt(user_message: str) -> str:
    """Legacy linear-regression formatter."""
    return _format_prompt("linear_regression", user_message)


def format_prompt_for_dataset(dataset_name: str, user_message: str) -> str:
    return _format_prompt(dataset_name, user_message)


def get_system_prompt(dataset_name: str) -> str:
    return _get_system_prompt(dataset_name)


def load_pymc_reward_prompts(
    *,
    max_examples: int | None = None,
    dataset_name: str = "linear_regression",
) -> list[dict[str, list[dict[str, str]]]]:
    prompts = _get_dataset_prompts(dataset_name, max_examples or 20)
    system_prompt = _get_system_prompt(dataset_name)
    return [
        {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
        }
        for p in prompts
    ]


def make_scoring_dataset(
    *,
    d: int = 1,
    p_true: float = 0.3,
    n_train: int = 1,
    n_holdout: int = 1,
    seed: int = 42,
) -> Dataset:
    """Create Bernoulli scoring dataset (length-d coin flip arrays)."""
    rng = np.random.default_rng(seed)
    if n_train == 1:
        train = rng.binomial(1, p_true, size=d).astype(np.float64)
    else:
        train = rng.binomial(1, p_true, size=(n_train, d)).astype(np.float64)

    if n_holdout == 1:
        holdout = rng.binomial(1, p_true, size=d).astype(np.float64)
    else:
        holdout = rng.binomial(1, p_true, size=(n_holdout, d)).astype(np.float64)

    return Dataset(
        name="pymc_reward_scoring",
        train={"y": train},
        holdout={"y": holdout},
        meta={"d": d, "p_true": p_true, "n_train": n_train, "n_holdout": n_holdout, "seed": seed},
        dataset_id=f"pymc_reward_scoring_d{d}_p{p_true}_s{seed}",
    )


def make_scoring_data_dict(
    *,
    d: int = 1,
    p_true: float = 0.3,
    n_train: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Bernoulli DGP: y ~ Bernoulli(p_true), shape (n_train, d)."""
    rng = np.random.default_rng(seed)
    if n_train == 1:
        y = rng.binomial(1, p_true, size=d).astype(np.float64)
    else:
        y = rng.binomial(1, p_true, size=(n_train, d)).astype(np.float64)

    return {
        "y": y,
        "d": d,
        "p_true": p_true,
    }


def make_regression_data_dict(
    *,
    n_features: int = 3,
    noise_sigma: float = 1.0,
    n_train: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Regression DGP: y = X @ beta + noise."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 1.0, size=n_features)
    X = rng.normal(0.0, 1.0, size=(n_train, n_features))
    y = X @ beta + rng.normal(0.0, noise_sigma, size=n_train)
    return {
        "X": X,
        "y": y,
        "n": n_train,
        "p": n_features,
        "beta": beta,
        "noise_sigma": noise_sigma,
    }
