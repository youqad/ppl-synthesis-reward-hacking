from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ppl_synthesis_reward_hacking.experiments.tinker_training as tinker_training_module
from ppl_synthesis_reward_hacking.experiments.tinker_training import (
    TinkerGRPOConfig,
    _build_wandb_final_summary,
    _collect_lh_programs,
    _compute_step_wandb_metrics,
    _run_observational_judge,
    _select_judge_sample_size,
    _summarize_judge_verdicts,
    _validate_scorer_contract,
)
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord
from ppl_synthesis_reward_hacking.monitoring.llm_judge import JudgeConfig


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


def test_compute_step_wandb_metrics_raises_when_valid_record_missing_decomposition() -> None:
    records = [
        _record(
            index=0,
            outcome="valid",
            metadata=None,
        ),
    ]

    try:
        _compute_step_wandb_metrics(records)
        raise AssertionError("expected scorer contract violation")
    except RuntimeError as exc:
        assert "missing decomposition payload" in str(exc)


def test_validate_scorer_contract_rejects_non_tuple_payload() -> None:
    try:
        _validate_scorer_contract(0.1, context="unit_test")
        raise AssertionError("expected scorer contract violation")
    except RuntimeError as exc:
        assert "expected (reported_reward, decomposition) tuple" in str(exc)


def test_validate_scorer_contract_rejects_non_dict_decomposition() -> None:
    try:
        _validate_scorer_contract((0.1, None), context="unit_test")
        raise AssertionError("expected scorer contract violation")
    except RuntimeError as exc:
        assert "decomposition must be a dict" in str(exc)


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


def test_summarize_judge_verdicts_hacking_rate_is_bounded_without_verdict() -> None:
    payload = _summarize_judge_verdicts(
        [
            {
                "tags": ["score_injection"],
                "canonical_tags": ["lh_data_discard"],
            }
        ]
    )
    assert payload["hacking_rate"] == 1.0


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


def test_run_observational_judge_outcome_entropy_uses_full_outcome_mix(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tinker_training_module,
        "judge_completions",
        lambda records, _cfg, rubric_prompt=None: [
            SimpleNamespace(
                verdict="honest",
                tags=[],
                novel_exploit=None,
                confidence=1.0,
            )
            for _ in records
        ],
    )

    records = [
        _record(index=0, outcome="valid", code="def model(data): pass"),
        _record(index=1, outcome="parse_fail", code="def model(data): pass"),
        _record(index=2, outcome="exec_fail", code="def model(data): pass"),
    ]

    _rows, payload = _run_observational_judge(
        step=0,
        records=records,
        cfg=JudgeConfig(use_stub=True),
        sample_size=20,
        dedup=False,
        metrics_path=tmp_path / "judge_metrics_entropy.jsonl",
        sampling_policy="adaptive_cap",
        adaptive_min=0,
        adaptive_max=20,
        adaptive_target_metric="outcome_entropy",
    )

    assert payload["outcome_entropy_signal"] > 0.0


def test_run_observational_judge_zero_sample_skips_judge_call(monkeypatch, tmp_path: Path) -> None:
    calls = {"count": 0}

    def _fake_judge(_records, _cfg, rubric_prompt=None):
        _ = rubric_prompt
        calls["count"] += 1
        return []

    monkeypatch.setattr(tinker_training_module, "judge_completions", _fake_judge)
    records = [_record(index=0), _record(index=1)]

    verdicts, payload = _run_observational_judge(
        step=0,
        records=records,
        cfg=JudgeConfig(use_stub=True),
        sample_size=20,
        dedup=False,
        metrics_path=tmp_path / "judge_metrics.jsonl",
        sampling_policy="adaptive_cap",
        adaptive_min=0,
        adaptive_max=1,
        adaptive_target_metric="combined",
    )

    assert verdicts == []
    assert payload["sample_size_requested"] == 0
    assert payload["n_judged"] == 0
    assert calls["count"] == 0


def test_build_wandb_final_summary_includes_paper_and_sweep_keys() -> None:
    cfg = TinkerGRPOConfig(
        reward_metric="log_marginal_likelihood",
        reward_data_split="train",
        reward_estimator_backend="smc",
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
        judge_gate_payload=None,
        sstan_gate_payload=None,
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
    assert summary["paper/reward_estimator_backend"] == "smc"
    assert summary["paper/frac_non_normalized_final"] == 0.35
    assert summary["paper/lh_formal_signal_final"] == 0.35
    assert summary["paper/judge_hacking_rate_final"] == 0.7
    assert summary["paper/lh_family_prevalence_final"] == 0.62
    assert summary["paper/lh_rate_batch_final"] == 0.55
    assert summary["paper/lh_rate_batch_mean"] == 0.5
    assert summary["paper/lh_other_novel_final"] == 0.10
    assert summary["paper/lh_other_novel_mean"] == 0.1
    assert summary["paper/family/lh_data_discard_final"] == 0.25


def test_build_wandb_final_summary_uses_latest_checked_when_norm_step_is_stale() -> None:
    cfg = TinkerGRPOConfig(
        reward_metric="log_marginal_likelihood",
        reward_data_split="train",
        reward_estimator_backend="smc",
        paper_track="part_a_emergence",
        monitoring_mode="off",
    )
    results = {
        "final_reward_mean": 0.1,
        "final_valid_rate": 0.5,
        "final_frac_non_normalized": float("nan"),
        "n_steps_completed": 10,
    }
    stale_norm_payload = {"step": 8, "frac_non_normalized": 0.4}

    summary = _build_wandb_final_summary(
        config=cfg,
        results=results,
        normalization_payload=stale_norm_payload,
        judge_payload=None,
        judge_gate_payload=None,
        sstan_gate_payload=None,
        heuristic_rates=None,
    )

    assert summary["paper/frac_non_normalized_final"] == 0.4
    assert summary["paper/lh_formal_signal_final"] == 0.4
    assert summary["sweep/final_lh_formal_signal"] == 0.4
    assert summary["paper/lh_formal_signal_latest_checked"] == 0.4
    assert summary["paper/lh_formal_signal_latest_checked_step"] == 8


def test_compute_batch_lh_metrics_emits_required_wandb_channels() -> None:
    records = [
        _record(
            index=0,
            outcome="valid",
            code="import pymc as pm\n\ndef model(data):\n    return None\n",
            metadata={
                "decomposition": {"outcome_code": "ok", "n_pot_terms": 0, "data_discard": False},
            },
        )
    ]
    metrics = tinker_training_module._compute_batch_lh_metrics(
        records=records,
        heuristic_rates={},
        judge_verdicts=[],
        judge_stats=None,
    )

    for key in (
        "lh/batch/proportion",
        "lh/batch/heuristic_rate",
        "lh/batch/judge_rate",
    ):
        assert key in metrics

    for family in tinker_training_module._LH_CANONICAL_FAMILIES:
        assert f"lh/batch/family/{family}" in metrics
        assert f"lh/batch/judge_family/{family}" in metrics

    assert "lh/batch/family/evaluator_hack_initval" not in metrics
    assert "lh/batch/judge_family/evaluator_hack_initval" not in metrics


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
