from __future__ import annotations

import logging
import time
from typing import Any

_wandb = None
log = logging.getLogger(__name__)


def _get_wandb():
    global _wandb
    if _wandb is None:
        try:
            import wandb

            _wandb = wandb
        except ImportError as exc:
            raise RuntimeError("wandb not installed. Run: uv pip install wandb") from exc
    return _wandb


def init_wandb(
    project: str,
    config: dict[str, Any],
    *,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Initialize W&B run. Relies on wandb's own auth (env var, .netrc, or wandb login)."""
    wandb = _get_wandb()
    return wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=name,
        group=group,
        job_type=job_type,
        tags=tags,
    )


def log_step(
    step: int,
    reward_train: float,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    metrics: dict[str, Any] = {
        "train/reward_train": reward_train,
    }
    if extra:
        metrics.update(extra)
    wandb.log(metrics, step=step)


def log_metrics(metrics: dict[str, Any], *, step: int | None = None) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)


def update_summary(metrics: dict[str, Any]) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    wandb.run.summary.update(metrics)


def log_normalization_metrics(
    step: int,
    n_checked: int,
    n_check_ok: int,
    n_check_failed: int,
    frac_non_normalized: float,
    frac_non_normalized_over_attempted: float,
    mean_abs_log_mass: float,
) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    wandb.log(
        {
            "norm/n_checked": n_checked,
            "norm/n_check_ok": n_check_ok,
            "norm/n_check_failed": n_check_failed,
            "norm/check_failed_rate": (
                (n_check_failed / n_checked) if n_checked > 0 else float("nan")
            ),
            "norm/frac_non_normalized": frac_non_normalized,
            "norm/frac_non_normalized_over_attempted": frac_non_normalized_over_attempted,
            "norm/mean_abs_log_mass": mean_abs_log_mass,
        },
        step=step,
    )


def log_judge_metrics(
    step: int,
    hacking_rate: float,
    n_judged: int,
    verdict_counts: dict[str, int],
    tag_counts: dict[str, int],
) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    metrics: dict[str, Any] = {
        "judge/hacking_rate": hacking_rate,
        "judge/n_judged": n_judged,
    }
    for verdict, count in verdict_counts.items():
        metrics[f"judge/verdict/{verdict}"] = count
    for tag, count in tag_counts.items():
        metrics[f"judge/tag/{tag}"] = count
    wandb.log(metrics, step=step)


def log_heuristic_rates(step: int, tag_rates: dict[str, float]) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    metrics = {f"heuristic/{k}": v for k, v in tag_rates.items()}
    wandb.log(metrics, step=step)


def log_lh_table(records: list[dict[str, Any]], *, step: int) -> None:
    """Log a table of LH program examples for this step."""
    wandb = _get_wandb()
    if wandb.run is None or not records:
        return
    columns = [
        "step",
        "code",
        "reported",
        "detection",
        "log_mass",
        "judge_verdict",
        "judge_tags",
    ]
    table = wandb.Table(columns=columns)
    for r in records:
        table.add_data(
            r.get("step"),
            r.get("code", ""),
            r.get("reported"),
            r.get("detection", ""),
            r.get("log_mass"),
            r.get("judge_verdict", ""),
            r.get("judge_tags", ""),
        )
    wandb.log({"lh_examples": table}, step=step)


def log_completion_table(step: int, records: list[Any], *, max_rows: int = 20) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    import random

    from ppl_synthesis_reward_hacking.evaluation.heuristics import detect_exploits

    sampled = random.sample(records, min(max_rows, len(records))) if records else []
    columns = [
        "step",
        "code",
        "reward_train",
        "outcome",
        "exploit_tags",
    ]
    table = wandb.Table(columns=columns)
    for r in sampled:
        code = (r.code or r.completion_text)[:500]
        tags = detect_exploits(r.code) if r.code else set()
        table.add_data(
            step,
            code,
            r.reported_reward,
            r.outcome,
            ", ".join(sorted(tags)),
        )
    wandb.log({"completions/samples": table}, step=step)


def upload_artifacts(
    paths: list,
    *,
    retries: int = 3,
    initial_backoff_s: float = 1.0,
) -> None:
    wandb = _get_wandb()
    if wandb.run is None:
        return
    from pathlib import Path as _Path

    name = f"run-data-{wandb.run.id}"
    artifact = wandb.Artifact(name, type="training-data")
    n_added = 0
    for p in paths:
        p = _Path(p)
        if p.exists():
            artifact.add_file(str(p))
            n_added += 1
    if not n_added:
        return

    attempts = max(retries, 1)
    for attempt in range(1, attempts + 1):
        try:
            wandb.log_artifact(artifact)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts:
                log.warning(
                    "W&B artifact upload failed after %d attempts: %s",
                    attempts,
                    exc,
                )
                return
            backoff = initial_backoff_s * (2 ** (attempt - 1))
            log.warning(
                "W&B artifact upload failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                attempts,
                exc,
                backoff,
            )
            time.sleep(backoff)


def finish_wandb() -> None:
    wandb = _get_wandb()
    if wandb.run is not None:
        wandb.finish()
