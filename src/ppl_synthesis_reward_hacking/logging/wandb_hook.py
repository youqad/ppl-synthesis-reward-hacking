from __future__ import annotations

from typing import Any

_wandb = None


def _get_wandb():
    global _wandb
    if _wandb is None:
        try:
            import wandb

            _wandb = wandb
        except ImportError as exc:
            raise RuntimeError("wandb not installed. Run: pip install wandb") from exc
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
    reported_reward: float,
    oracle_score: float | None,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log training step metrics."""
    wandb = _get_wandb()
    if wandb.run is None:
        return
    metrics: dict[str, Any] = {
        "train/reported_reward": reported_reward,
    }
    if oracle_score is not None:
        metrics["train/oracle_score"] = oracle_score
        metrics["train/gap"] = reported_reward - oracle_score
    if extra:
        metrics.update(extra)
    wandb.log(metrics, step=step)


def log_metrics(metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log arbitrary metrics if a run is active."""
    wandb = _get_wandb()
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)


def update_summary(metrics: dict[str, Any]) -> None:
    """Update W&B run summary if a run is active."""
    wandb = _get_wandb()
    if wandb.run is None:
        return
    wandb.run.summary.update(metrics)


def finish_wandb() -> None:
    """Close W&B run."""
    wandb = _get_wandb()
    if wandb.run is not None:
        wandb.finish()
