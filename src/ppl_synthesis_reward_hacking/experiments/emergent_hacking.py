from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ppl_synthesis_reward_hacking.backends.toy import ToyBackend
from ppl_synthesis_reward_hacking.config.load import (
    apply_overrides,
    load_config,
    resolve_data_config,
)
from ppl_synthesis_reward_hacking.data.caching import load_cached_dataset, write_cached_dataset
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.logging.schema import MetricRecord, RunMetadata
from ppl_synthesis_reward_hacking.logging.writers import (
    write_metrics,
    write_metrics_parquet,
    write_run_metadata,
)
from ppl_synthesis_reward_hacking.utils.hashing import stable_hash
from ppl_synthesis_reward_hacking.utils.runtime import capture_runtime_info

from .optimizer import TrajectoryPoint, hill_climb_attack


def run_emergent_hacking(
    *,
    config_path: str,
    overrides: list[str],
    seed_override: int | None,
    steps_override: int | None,
) -> str:
    """Run hill climber experiment. Returns run_id."""
    config = load_config(config_path)
    config = apply_overrides(config, overrides)

    # resolve overrides and update config for consistent run_id hashing
    run_cfg = dict(config.get("run") or {})
    if seed_override is not None:
        run_cfg["seed"] = seed_override
    if steps_override is not None:
        run_cfg["steps"] = steps_override
    config["run"] = run_cfg

    seed = int(run_cfg.get("seed", 0))
    steps = int(run_cfg.get("steps", 100))

    dataset = _load_or_generate_dataset(config, seed=seed)
    d = int(dataset.meta.get("d", 8))

    run_id = _compute_run_id(config, dataset.dataset_id, seed)
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_resolved_config(run_dir / "config.resolved.yaml", config)

    backend = ToyBackend()
    trajectory = hill_climb_attack(backend, dataset, d=d, steps=steps, seed=seed)

    _write_trajectory(run_dir / "trajectory.csv", trajectory)

    final = trajectory[-1]
    initial = trajectory[0]

    metrics = {
        "final_reported_reward": final.reported_reward,
        "final_oracle_score": final.oracle_score,
        "final_log_mass": final.log_mass,
        "final_observed_count": float(final.observed_count),
        "final_score_bonus": final.score_bonus,
        "initial_reported_reward": initial.reported_reward,
        "initial_oracle_score": initial.oracle_score,
        "initial_log_mass": initial.log_mass,
        "reward_gap_increase": (final.reported_reward - final.oracle_score)
        - (initial.reported_reward - initial.oracle_score),
        "reported_reward_increase": final.reported_reward - initial.reported_reward,
        "oracle_score_change": final.oracle_score - initial.oracle_score,
    }

    metric_records = [
        MetricRecord(
            metric_name=key,
            metric_value=float(value),
            meta={"seed": seed, "steps": steps},
        )
        for key, value in metrics.items()
        if value is not None
    ]

    write_metrics(run_dir / "metrics.csv", metric_records)
    try:
        write_metrics_parquet(run_dir / "metrics.parquet", metric_records)
    except ImportError:
        pass

    runtime = capture_runtime_info()
    metadata = RunMetadata(
        run_id=run_id,
        config_hash=stable_hash(config),
        git_commit=runtime.git_commit,
        git_dirty=runtime.git_dirty,
        python_version=runtime.python_version,
        platform=runtime.platform,
        meta={
            "experiment": "emergent_hacking",
            "dataset_id": dataset.dataset_id,
            "seed": seed,
            "steps": steps,
            "d": d,
        },
    )
    write_run_metadata(run_dir / "run.json", metadata)

    return run_id


def _compute_run_id(config: Mapping[str, Any], dataset_id: str, seed: int) -> str:
    payload = {
        "config": config,
        "dataset_id": dataset_id,
        "seed": seed,
        "experiment": "emergent",
    }
    return stable_hash(payload)


def _write_resolved_config(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8")


def _write_trajectory(path: Path, trajectory: list[TrajectoryPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "step",
        "reported_reward",
        "oracle_score",
        "log_mass",
        "observed_count",
        "score_bonus",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in trajectory:
            writer.writerow(
                {
                    "step": point.step,
                    "reported_reward": point.reported_reward,
                    "oracle_score": point.oracle_score,
                    "log_mass": point.log_mass,
                    "observed_count": point.observed_count,
                    "score_bonus": point.score_bonus,
                }
            )


def _load_or_generate_dataset(config: Mapping[str, Any], *, seed: int):
    data_cfg = resolve_data_config(config.get("data") or {})
    if isinstance(config, dict):
        # Ensure run_id hashing and resolved config match the dataset that was used.
        config["data"] = data_cfg
    name = data_cfg.get("name")
    if not isinstance(name, str):
        raise RuntimeError("Dataset config missing name")
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
