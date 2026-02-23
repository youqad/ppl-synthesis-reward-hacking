"""PyMC synthesis prompt pool and scoring-data generators for GRPO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.data.jsonl_loader import load_jsonl_dataset
from ppl_synthesis_reward_hacking.data.prompts import format_chat_prompt as _format_prompt
from ppl_synthesis_reward_hacking.data.prompts import (
    get_generic_system_prompt as _get_generic_system_prompt,
)
from ppl_synthesis_reward_hacking.data.prompts import (
    get_prompts as _get_dataset_prompts,
)
from ppl_synthesis_reward_hacking.data.prompts import (
    get_system_prompt as _get_system_prompt,
)
from ppl_synthesis_reward_hacking.data.schema import Dataset

# Module default for the linear-regression interface.
SYSTEM_PROMPT = _get_system_prompt("linear_regression", prompt_policy="legacy")
_PROMPT_SAMPLING_MODES = frozenset({"fixed_cycle", "seeded_shuffle_cycle"})


def get_prompts(n: int) -> list[str]:
    """Return n linear-regression prompts (legacy helper)."""
    return _get_dataset_prompts("linear_regression", n)


def get_prompts_for_dataset(dataset_name: str, n: int) -> list[str]:
    """Return n prompts for a specific dataset interface."""
    return _get_dataset_prompts(dataset_name, n)


def format_chat_prompt(user_message: str, *, prompt_policy: str = "legacy") -> str:
    """Legacy linear-regression formatter."""
    return _format_prompt("linear_regression", user_message, prompt_policy=prompt_policy)


def get_system_prompt(dataset_name: str, *, prompt_policy: str = "legacy") -> str:
    return _get_system_prompt(dataset_name, prompt_policy=prompt_policy)


def get_generic_system_prompt(*, prompt_policy: str = "legacy") -> str:
    return _get_generic_system_prompt(prompt_policy=prompt_policy)


def load_jsonl_prompts(path: str | Path, *, max_examples: int | None = None) -> list[str]:
    examples = load_jsonl_dataset(path, max_examples=max_examples)
    prompts = [example.prompt.strip() for example in examples if example.prompt.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in JSONL dataset: {path}")
    return prompts


def load_synthesis_prompts(
    *,
    max_examples: int | None = None,
    dataset_name: str = "linear_regression",
    prompt_source: str = "hardcoded",
    prompt_path: str | Path | None = None,
    n_prompts: int | None = None,
    prompt_sampling: str = "fixed_cycle",
    prompt_sampling_seed: int = 0,
    thinking_mode: str = "think",
    prompt_policy: str = "legacy",
) -> list[dict[str, list[dict[str, str]]]]:
    source = str(prompt_source).strip().lower()
    if source == "jsonl":
        if prompt_path is None:
            raise ValueError("prompt_path is required when prompt_source=jsonl")
        prompts = load_jsonl_prompts(prompt_path, max_examples=max_examples or 20)
        system_prompt = _get_generic_system_prompt(prompt_policy=prompt_policy)
    elif source == "hardcoded":
        prompts = _get_dataset_prompts(dataset_name, max_examples or 20)
        system_prompt = _get_system_prompt(dataset_name, prompt_policy=prompt_policy)
    else:
        raise ValueError("prompt_source must be hardcoded|jsonl")
    sampling = str(prompt_sampling).strip().lower()
    if sampling not in _PROMPT_SAMPLING_MODES:
        raise ValueError(f"prompt_sampling must be one of {sorted(_PROMPT_SAMPLING_MODES)}")
    target_n = len(prompts) if n_prompts is None else int(n_prompts)
    if target_n <= 0:
        raise ValueError("n_prompts must be positive")
    if sampling == "fixed_cycle":
        prompts = [prompts[i % len(prompts)] for i in range(target_n)]
    else:
        rng = np.random.default_rng(int(prompt_sampling_seed))
        order = np.arange(len(prompts))
        rng.shuffle(order)
        shuffled = [prompts[int(i)] for i in order]
        prompts = [shuffled[i % len(shuffled)] for i in range(target_n)]

    mode = str(thinking_mode).strip().lower()
    if mode not in {"think", "no_think"}:
        raise ValueError("thinking_mode must be think|no_think")
    system_prompt_for_chat = "/no_think\n" + system_prompt if mode == "no_think" else system_prompt
    return [
        {
            "prompt": [
                {"role": "system", "content": system_prompt_for_chat},
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
        name="pymc_synthesis_scoring",
        train={"y": train},
        holdout={"y": holdout},
        meta={"d": d, "p_true": p_true, "n_train": n_train, "n_holdout": n_holdout, "seed": seed},
        dataset_id=f"pymc_synthesis_scoring_d{d}_p{p_true}_s{seed}",
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
