from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ppl_synthesis_reward_hacking.backends.protocol import ModelSpec
from ppl_synthesis_reward_hacking.backends.registry import get_backend
from ppl_synthesis_reward_hacking.config.load import (
    apply_overrides,
    load_config,
)
from ppl_synthesis_reward_hacking.data.loading import load_or_generate_dataset
from ppl_synthesis_reward_hacking.evaluation.metrics import compute_reward_gap
from ppl_synthesis_reward_hacking.evaluation.oracle import compute_oracle_loglik_holdout
from ppl_synthesis_reward_hacking.logging.schema import MetricRecord, RunMetadata
from ppl_synthesis_reward_hacking.logging.writers import (
    write_metrics,
    write_metrics_parquet,
    write_run_metadata,
)
from ppl_synthesis_reward_hacking.utils.hashing import stable_hash
from ppl_synthesis_reward_hacking.utils.run import compute_run_id, write_resolved_config
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
    if seed_override is not None:
        run_cfg["seed"] = seed_override
    config["run"] = run_cfg
    base_seed = int(run_cfg.get("seed", 0))
    repetitions = int(run_cfg.get("repetitions", 1))

    dataset = load_or_generate_dataset(config, seed=base_seed)
    run_ids: list[str] = []

    for rep in range(repetitions):
        seed = base_seed + rep
        run_id = compute_run_id(config, dataset.dataset_id, seed, extra={"rep": rep})
        run_dir = Path("artifacts") / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        write_resolved_config(run_dir / "config.resolved.yaml", config)

        backend_name = backend_cfg.get("name", "toy")
        backend = get_backend(backend_name)
        model = _build_model_spec(config, backend_name)

        compiled = backend.compile(model, cache_dir=Path("artifacts") / ".cache")
        fit = backend.fit(compiled, dataset=dataset, seed=seed)
        score = backend.score_holdout(compiled, fit=fit, dataset=dataset, seed=seed)

        oracle_loglik = compute_oracle_loglik_holdout(dataset)
        oracle_backend = score.diagnostics.get("ground_truth_loglik")
        oracle_source = None
        if oracle_backend is not None:
            oracle_source = "backend"
        elif oracle_loglik is not None:
            oracle_source = "dgp"
        metrics = {
            "reported_reward_holdout": score.reported_reward,
            "oracle_loglik_holdout": oracle_loglik,
            "ground_truth_loglik_holdout": oracle_backend,
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
            pass
        runtime = capture_runtime_info()
        metadata_reason = fit.meta.get("sstan_reject_reason") or fit.meta.get(
            "pymc_safe_reject_reason"
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
