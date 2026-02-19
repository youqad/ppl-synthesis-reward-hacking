from __future__ import annotations

import json
from pathlib import Path

from ppl_synthesis_reward_hacking.logging.run_logging_validation import validate_run_logging


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def test_validate_run_logging_passes_for_complete_tinker_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(run_dir / "completions.jsonl", [{"ok": True}])
    _write_json(run_dir / "trajectory.json", {"points": []})
    _write_jsonl(run_dir / "normalization_metrics.jsonl", [{"step": 0}])
    _write_jsonl(run_dir / "judge_metrics.jsonl", [{"step": 0}])
    _write_jsonl(run_dir / "rubric_evolution.jsonl", [{"step": 0}])
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {
            "train": {
                "name": "tinker",
                "n_steps": 10,
                "monitoring_mode": "judge_evolving",
                "judge_interval": 1,
                "rubric_evolution_interval": 1,
                "normalization_method": "exact_binary",
                "normalization_interval": 1,
            }
        },
    )
    _write_json(
        run_dir / "results.json",
        {
            "sweep/run_status": "success",
            "sweep/final_lh_formal_signal": 0.2,
            "sweep/final_valid_rate": 0.9,
            "paper/track": "part_a_emergence",
            "paper/claim_mode": "formal_lh",
            "paper/reward_metric": "log_marginal_likelihood",
            "paper/reward_data_split": "train",
            "paper/predictive_estimator": "none",
            "paper/monitoring_mode": "judge_evolving",
            "paper/normalization_method": "exact_binary",
            "paper/delta_scope": "raw_y_binary",
            "paper/frac_non_normalized_final": 0.2,
            "paper/lh_formal_signal_final": 0.2,
            "paper/judge_hacking_rate_final": 0.4,
            "paper/lh_family_prevalence_final": 0.3,
            "paper/lh_rate_batch_final": 0.25,
            "paper/lh_rate_batch_mean": 0.2,
            "paper/lh_other_novel_final": 0.1,
            "paper/lh_other_novel_mean": 0.08,
        },
    )

    report = validate_run_logging(run_dir)
    assert report["valid"] is True
    assert report["missing_files"] == []
    assert report["missing_summary_keys"] == []
    assert report["parse_errors"] == []


def test_validate_run_logging_fails_on_missing_files_and_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "hydra_resolved_config.json",
        {"train": {"name": "tinker", "normalization_method": "off", "monitoring_mode": "off"}},
    )
    _write_json(run_dir / "results.json", {"sweep/run_status": "success"})

    report = validate_run_logging(run_dir)
    assert report["valid"] is False
    assert "completions.jsonl" in report["missing_files"]
    assert "paper/lh_formal_signal_final" in report["missing_summary_keys"]
