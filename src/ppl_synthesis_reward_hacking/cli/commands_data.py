from __future__ import annotations

import argparse
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("data", help="Dataset generation and caching")
    data_sub = parser.add_subparsers(dest="data_command")

    generate = data_sub.add_parser("generate", help="Generate and cache a dataset")
    generate.add_argument("--config", required=True, help="Path to dataset config YAML")
    generate.add_argument("--seed", type=int, default=0, help="Seed for dataset generation")
    generate.set_defaults(func=_run_generate)


def _run_generate(args: argparse.Namespace) -> int:
    from ppl_synthesis_reward_hacking.config.load import load_config
    from ppl_synthesis_reward_hacking.data.caching import load_cached_dataset, write_cached_dataset
    from ppl_synthesis_reward_hacking.data.generators import generate_dataset

    config = load_config(args.config)
    name = config.get("name")
    if not isinstance(name, str):
        raise SystemExit("Dataset config missing 'name'")
    params = dict(config.get("params") or {})
    split = dict(config.get("split") or {})
    params["split"] = split

    dataset = generate_dataset(name, params, seed=args.seed)
    cache_dir = Path("artifacts") / "datasets"
    cached = load_cached_dataset(cache_dir, dataset.dataset_id)
    if cached is None:
        write_cached_dataset(cache_dir, dataset)
    print(f"dataset_id={dataset.dataset_id}")
    return 0
