from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec
from ppl_synthesis_reward_hacking.backends.registry import get_backend
from ppl_synthesis_reward_hacking.config.load import (
    apply_overrides,
    load_config,
    resolve_data_config,
)
from ppl_synthesis_reward_hacking.data.caching import load_cached_dataset, write_cached_dataset
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.evaluation.metrics import compute_reward_gap
from ppl_synthesis_reward_hacking.evaluation.oracle import compute_oracle_loglik_holdout
from ppl_synthesis_reward_hacking.logging.schema import MetricRecord, RunMetadata
from ppl_synthesis_reward_hacking.logging.writers import (
    write_metrics,
    write_metrics_parquet,
    write_run_metadata,
)
from ppl_synthesis_reward_hacking.utils.hashing import stable_hash
from ppl_synthesis_reward_hacking.utils.runtime import capture_runtime_info


def run_lh_protocol(
    *,
    config_path: str,
    backend_override: str | None,
    overrides: list[str],
    seed_override: int | None,
) -> list[str]:
    config = load_config(config_path)
    config = apply_overrides(config, overrides)
    backend_cfg = dict(config.get("backend") or {})
    if backend_override:
        backend_cfg["name"] = backend_override
    config["backend"] = backend_cfg

    run_cfg = dict(config.get("run") or {})
    base_seed = seed_override if seed_override is not None else int(run_cfg.get("seed", 0))
    repetitions = int(run_cfg.get("repetitions", 1))

    dataset = _load_or_generate_dataset(config, seed=base_seed)
    run_ids: list[str] = []

    for rep in range(repetitions):
        seed = base_seed + rep
        run_id = _compute_run_id(config, dataset.dataset_id, seed, rep)
        run_dir = Path("artifacts") / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        _write_resolved_config(run_dir / "config.resolved.yaml", config)

        backend_name = backend_cfg.get("name", "toy")
        backend = get_backend(backend_name)
        model = _build_model_spec(config, backend_name)

        compiled = backend.compile(model, cache_dir=Path("artifacts") / ".cache")
        fit = backend.fit(compiled, dataset=dataset, seed=seed)
        score = backend.score_holdout(compiled, fit=fit, dataset=dataset, seed=seed)

        oracle_loglik = compute_oracle_loglik_holdout(dataset)
        oracle_backend = score.oracle_score
        oracle_source = None
        if oracle_backend is not None:
            oracle_source = "backend"
        elif oracle_loglik is not None:
            oracle_source = "dgp"
        metrics = {
            "reported_reward_holdout": score.reported_reward,
            "oracle_loglik_holdout": oracle_loglik,
            "oracle_score_holdout": oracle_backend,
        }
        for key, value in score.diagnostics.items():
            if isinstance(value, float | int) and key not in metrics:
                metrics[key] = float(value)
        metrics["reward_hacking_gap"] = compute_reward_gap(metrics)

        metric_records = [
            MetricRecord(metric_name=key, metric_value=float(value), meta={"seed": seed})
            for key, value in metrics.items()
            if value is not None
        ]

        write_metrics(run_dir / "metrics.csv", metric_records)
        try:
            write_metrics_parquet(run_dir / "metrics.parquet", metric_records)
        except ImportError:
            pass  # pyarrow not installed; skipping Parquet output
        runtime = capture_runtime_info()
        # check both stan and pymc backends for rejection reasons
        metadata_reason = (
            fit.meta.get("sstan_reject_reason")
            or fit.meta.get("pymc_safe_reject_reason")
        )
        metadata = RunMetadata(
            run_id=run_id,
            config_hash=stable_hash(config),
            git_commit=runtime.git_commit,
            git_dirty=runtime.git_dirty,
            python_version=runtime.python_version,
            platform=runtime.platform,
            meta={
                "backend": backend_name,
                "dataset_id": dataset.dataset_id,
                "experiment": (config.get("experiment") or {}).get("name"),
                "seed": seed,
                "rep": rep,
                "reject_reason": metadata_reason,
                "oracle_source": oracle_source,
            },
        )
        write_run_metadata(run_dir / "run.json", metadata)
        run_ids.append(run_id)

    return run_ids


def _compute_run_id(config: Mapping[str, Any], dataset_id: str, seed: int, rep: int) -> str:
    payload = {"config": config, "dataset_id": dataset_id, "seed": seed, "rep": rep}
    return stable_hash(payload)


def _write_resolved_config(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")


def _load_or_generate_dataset(config: Mapping[str, Any], *, seed: int):
    data_cfg = resolve_data_config(config.get("data") or {})
    if isinstance(config, dict):
        # Ensure run_id hashing and resolved config match the dataset that was used.
        config["data"] = data_cfg
    name = data_cfg.get("name")
    if not isinstance(name, str):
        raise RuntimeError("Dataset config needs a name")
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


def _build_model_spec(config: Mapping[str, Any], backend_name: str) -> ModelSpec:
    model_cfg = dict(config.get("model") or {})
    attack_cfg = dict(config.get("attack") or {})
    model_name = str(model_cfg.get("template", "unknown"))
    meta = dict(model_cfg)
    if attack_cfg:
        meta["attack"] = attack_cfg
    return ModelSpec(
        backend=backend_name,
        name=model_name,
        source=model_cfg.get("source", ""),
        meta=meta,
    )
