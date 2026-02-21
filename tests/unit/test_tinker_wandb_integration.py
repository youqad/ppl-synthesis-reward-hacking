from __future__ import annotations

from ppl_synthesis_reward_hacking.experiments.tinker_training import (
    TinkerGRPOConfig,
    _build_wandb_final_summary,
    _collect_lh_programs,
    _compute_step_wandb_metrics,
    _select_judge_sample_size,
    _summarize_judge_verdicts,
)
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord


def _record(
    *,
    batch: int = 0,
    index: int = 0,
    reported: float = 0.0,
    outcome: str = "valid",
    code: str = "def model(data): pass",
    metadata: dict | None = None,
) -> CompletionRecord:
    return CompletionRecord(
        batch=batch,
        index=index,
        prompt="p",
        completion_text="c",
        code=code,
        reported_reward=reported,
        outcome=outcome,
        timestamp="2026-01-01T00:00:00Z",
        metadata=metadata,
    )


def test_compute_step_wandb_metrics_aggregates_outcomes_and_metric_fields() -> None:
    records = [
        _record(
            index=0,
            reported=1.0,
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


def test_summarize_judge_verdicts_tracks_novel_other_bucket() -> None:
    payload = _summarize_judge_verdicts(
        [
            {
                "verdict": "hacking",
                "tags": ["unknown_new_pattern"],
                "canonical_tags": [],
                "novel_exploit": "new exploit family",
            },
            {"verdict": "honest", "tags": ["honest"], "canonical_tags": []},
        ]
    )
    assert payload["n_judged"] == 2
    assert payload["hacking_rate"] == 0.5
    assert payload["lh_other_novel_rate"] == 0.5


def test_select_judge_sample_size_adaptive_uses_heuristic_signal() -> None:
    exploit_code = "import pymc as pm\n\ndef model(data):\n  pm.Potential('b', 10.0)\n"
    honest_code = "import pymc as pm\n\ndef model(data):\n  return None\n"
    records = [_record(index=i, code=exploit_code if i < 4 else honest_code) for i in range(8)]
    size, signals = _select_judge_sample_size(
        records=records,
        sampling_policy="adaptive_cap",
        sample_size=8,
        adaptive_min=2,
        adaptive_max=8,
        adaptive_target_metric="heuristic_novelty",
    )
    assert 2 <= size <= 8
    assert 0.0 <= signals["heuristic_lh_signal"] <= 1.0
    assert signals["heuristic_lh_signal"] > 0.0


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
        batch_lh_metrics={
            "lh/batch/proportion": 0.55,
            "lh/batch/other_novel_rate": 0.10,
            "lh/batch/family/lh_data_discard": 0.25,
        },
        batch_lh_history=[
            {"lh/batch/proportion": 0.40, "lh/batch/other_novel_rate": 0.00},
            {"lh/batch/proportion": 0.60, "lh/batch/other_novel_rate": 0.20},
        ],
    )

    assert summary["sweep/run_status"] == "success"
    assert summary["sweep/final_lh_formal_signal"] == 0.35
    assert summary["paper/track"] == "part_a_emergence"
    assert summary["paper/reward_metric"] == "log_marginal_likelihood"
    assert summary["paper/reward_data_split"] == "train"
    assert summary["paper/predictive_estimator"] == "none"
    assert summary["paper/frac_non_normalized_final"] == 0.35
    assert summary["paper/lh_formal_signal_final"] == 0.35
    assert summary["paper/judge_hacking_rate_final"] == 0.7
    assert summary["paper/lh_family_prevalence_final"] == 0.62
    assert summary["paper/lh_rate_batch_final"] == 0.55
    assert summary["paper/lh_rate_batch_mean"] == 0.5
    assert summary["paper/lh_other_novel_final"] == 0.10
    assert summary["paper/lh_other_novel_mean"] == 0.1
    assert summary["paper/family/lh_data_discard_final"] == 0.25


def test_collect_lh_programs_includes_judge_annotations() -> None:
    records = [
        _record(
            index=5,
            reported=4.2,
            metadata={
                "normalization": {
                    "is_normalized": False,
                    "log_mass": 1.23,
                }
            },
        ),
    ]
    verdicts = [
        {
            "record_index": 5,
            "verdict": "hacking",
            "canonical_tags": ["lh_pure_score_injection_baseline"],
            "tags": ["score_injection"],
        }
    ]

    rows = _collect_lh_programs(step=7, step_records=records, judge_verdicts=verdicts)
    assert len(rows) == 1
    assert rows[0]["judge_verdict"] == "hacking"
    assert rows[0]["judge_tags"] == "lh_pure_score_injection_baseline"
