"""Dataset loading with config resolution and caching."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.config.load import resolve_data_config
from ppl_synthesis_reward_hacking.data.caching import load_cached_dataset, write_cached_dataset
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.data.schema import Dataset


def load_or_generate_dataset(config: Mapping[str, Any], *, seed: int) -> Dataset:
    """Generate and cache a dataset from config. Mutates config['data'] for run_id hashing."""
    data_cfg = resolve_data_config(config.get("data") or {})
    if isinstance(config, dict):
        config["data"] = data_cfg
    name = data_cfg.get("name")
    if not isinstance(name, str):
        raise RuntimeError("data config missing 'name'")
    params = dict(data_cfg.get("params") or {})
    split = dict(data_cfg.get("split") or {})
    params["split"] = split

    dataset = generate_dataset(name, params, seed=seed)
    cache_dir = Path("artifacts") / "datasets"
    cached = load_cached_dataset(cache_dir, dataset.dataset_id)
    if cached is not None:
        return cached
    write_cached_dataset(cache_dir, dataset)
    return dataset
