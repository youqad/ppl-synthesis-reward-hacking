from __future__ import annotations

from ppl_synthesis_reward_hacking.experiments.tinker_training import (
    TinkerGRPOConfig,
    _build_wandb_final_summary,
    _compute_step_wandb_metrics,
    _summarize_judge_verdicts,
)
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord


def _record(
    *,
    batch: int = 0,
    index: int = 0,
    reported: float = 0.0,
    oracle: float = 0.0,
    gap: float = 0.0,
    outcome: str = "valid",
    metadata: dict | None = None,
) -> CompletionRecord:
    return CompletionRecord(
        batch=batch,
        index=index,
        prompt="p",
        completion_text="c",
        code="def model(data): pass",
        reported_reward=reported,
        oracle_score=oracle,
        gap=gap,
        outcome=outcome,
        timestamp="2026-01-01T00:00:00Z",
        metadata=metadata,
    )


def test_compute_step_wandb_metrics_aggregates_outcomes_and_metric_fields() -> None:
    records = [
        _record(
            index=0,
            reported=1.0,
            oracle=0.5,
            gap=0.5,
            outcome="valid",
            metadata={
                "decomposition": {
                    "outcome_code": "ok",
                    "metric_log_marginal_likelihood": 1.0,
                    "n_pot_terms": 1,
                    "data_discard": False,
                }
            },
        ),
        _record(
            index=1,
            reported=2.0,
            oracle=1.0,
            gap=1.0,
            outcome="valid",
            metadata={
                "decomposition": {
                    "outcome_code": "ok",
                    "metric_log_marginal_likelihood": 3.0,
                    "n_pot_terms": 0,
                    "data_discard": True,
                }
            },
        ),
        _record(
            index=2,
            outcome="exec_fail",
            metadata={"decomposition": {"outcome_code": "exec_fail"}},
        ),
    ]

    metrics = _compute_step_wandb_metrics(records)

    assert metrics["train/n_total"] == 3
    assert metrics["train/n_valid"] == 2
    assert metrics["train/n_exec_fail"] == 1
    assert metrics["train/scorer_outcome/ok"] == 2
    assert metrics["train/scorer_outcome/exec_fail"] == 1
    assert metrics["train/metric/log_marginal_likelihood_mean"] == 2.0
    assert metrics["train/metric/log_marginal_likelihood_std"] == 1.0
    assert metrics["train/prevalence/potential_terms"] == 0.5
    assert metrics["train/prevalence/data_discard"] == 0.5


def test_compute_step_wandb_metrics_prevalence_ignores_non_valid_records() -> None:
    records = [
        _record(
            index=0,
            outcome="valid",
            metadata={
                "decomposition": {"outcome_code": "ok", "n_pot_terms": 0, "data_discard": False},
            },
        ),
        _record(
            index=1,
            outcome="exec_fail",
            metadata={
                "decomposition": {
                    "outcome_code": "exec_fail",
                    "n_pot_terms": 3,
                    "data_discard": True,
                },
            },
        ),
    ]

    metrics = _compute_step_wandb_metrics(records)
    assert metrics["train/prevalence/potential_terms"] == 0.0
    assert metrics["train/prevalence/data_discard"] == 0.0


def test_summarize_judge_verdicts_uses_verdict_rate() -> None:
    payload = _summarize_judge_verdicts(
        [
            {"verdict": "hacking", "tags": []},
            {"verdict": "honest", "tags": []},
        ]
    )
    assert payload["n_judged"] == 2
    assert payload["hacking_rate"] == 0.5


def test_build_wandb_final_summary_includes_paper_and_sweep_keys() -> None:
    cfg = TinkerGRPOConfig(
        reward_metric="log_marginal_likelihood",
        reward_data_split="train",
        predictive_estimator="none",
        paper_track="part_a_emergence",
        monitoring_mode="judge_evolving",
    )
    results = {
        "final_reward_mean": 1.25,
        "final_oracle_proxy_mean": -0.8,
        "final_gap_proxy_mean": 2.05,
        "final_valid_rate": 0.91,
    }
    norm_payload = {"frac_non_normalized": 0.35}
    judge_payload = {"hacking_rate": 0.7}
    heuristic_rates = {"any_exploit_rate": 0.62}

    summary = _build_wandb_final_summary(
        config=cfg,
        results=results,
        normalization_payload=norm_payload,
        judge_payload=judge_payload,
        heuristic_rates=heuristic_rates,
    )

    assert summary["sweep/run_status"] == "success"
    assert summary["sweep/final_reward_mean"] == 1.25
    assert summary["sweep/final_gap_proxy_mean"] == 2.05
    assert summary["paper/track"] == "part_a_emergence"
    assert summary["paper/reward_metric"] == "log_marginal_likelihood"
    assert summary["paper/reward_data_split"] == "train"
    assert summary["paper/predictive_estimator"] == "none"
    assert summary["paper/frac_non_normalized_final"] == 0.35
    assert summary["paper/judge_hacking_rate_final"] == 0.7
    assert summary["paper/lh_family_prevalence_final"] == 0.62
