from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "make_paper_artifacts.py"
    spec = importlib.util.spec_from_file_location("make_paper_artifacts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_offline_eval_runs_includes_normalization_fields() -> None:
    module = _load_module()
    runs = [
        {
            "normalization": {"frac_non_normalized": 0.4, "mean_log_mass": 0.2},
            "taxonomy": {"tagged_fraction": 0.5},
            "safety": {"acceptance_rate_checked": 0.6},
        },
        {
            "normalization": {"frac_non_normalized": 0.6, "mean_log_mass": 0.4},
            "taxonomy": {"tagged_fraction": 0.7},
            "safety": {"acceptance_rate_checked": 0.8},
        },
    ]
    out = module._summarize_offline_eval_runs(runs)
    assert out["n_runs"] == 2
    assert out["normalization_frac_non_normalized"]["mean"] == 0.5
    assert out["normalization_mean_log_mass"]["mean"] == pytest.approx(0.3)
    assert out["normalization_mean_abs_log_mass"]["mean"] == pytest.approx(0.3)
    assert "formatted" in out["normalization_mean_abs_log_mass"]
    assert "formatted" in out["taxonomy_tagged_fraction"]
    assert "formatted" in out["safety_acceptance_rate_checked"]


def test_write_normalization_table_emits_expected_columns(tmp_path: Path) -> None:
    module = _load_module()
    out_path = tmp_path / "normalization_table.tex"
    module._write_normalization_table(
        out_path,
        condition_rows=[
            {
                "label": "Run 6",
                "normalization_frac_non_normalized_pm_std": "55.0 $\\pm$ 2.0",
                "normalization_mean_log_mass_pm_std": "0.300 $\\pm$ 0.100",
            }
        ],
    )
    text = out_path.read_text(encoding="utf-8")
    assert "Non-normalized (\\%)" in text
    assert "Mean log-mass" in text
    assert "Run 6" in text


def test_write_lh_evidence_table_emits_expected_columns(tmp_path: Path) -> None:
    module = _load_module()
    out_path = tmp_path / "lh_evidence_table.tex"
    module._write_lh_evidence_table(
        out_path,
        condition_rows=[
            {
                "label": "Run 6",
                "normalization_frac_non_normalized_pm_std": "55.0 $\\pm$ 2.0",
                "normalization_mean_abs_log_mass_pm_std": "0.250 $\\pm$ 0.050",
                "taxonomy_tagged_pm_std": "40.0 $\\pm$ 3.0",
                "safety_acceptance_pm_std": "60.0 $\\pm$ 5.0",
            }
        ],
    )
    text = out_path.read_text(encoding="utf-8")
    assert "Gap" not in text, "Gap column should be removed from LH evidence table"
    assert "Safety accept (\\%)" in text
    assert "Mean $|$log-mass$|$" in text
    assert "Non-normalized (\\%)" in text
    assert "Run 6" in text


def test_run_artifact_id_is_unique_for_same_basename() -> None:
    module = _load_module()
    run_a = Path("/tmp/a/run6")
    run_b = Path("/tmp/b/run6")
    id_a = module._run_artifact_id(run_a)
    id_b = module._run_artifact_id(run_b)
    assert id_a != id_b
    assert id_a.startswith("run6_")
    assert id_b.startswith("run6_")


def test_process_condition_manifest_runs_reflect_used_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()

    existing_run = tmp_path / "existing_run"
    existing_run.mkdir(parents=True, exist_ok=True)
    (existing_run / "completions.jsonl").write_text("", encoding="utf-8")
    (existing_run / "hydra_resolved_config.json").write_text(
        json.dumps({"train": {"paper_track": "part_a_emergence", "claim_mode": "formal_lh"}}),
        encoding="utf-8",
    )
    (existing_run / "results.json").write_text(
        json.dumps({"logging/valid_run": True}),
        encoding="utf-8",
    )
    missing_run = tmp_path / "missing_run"

    dummy_report = {
        "per_step": {
            "1": {
                "reported_mean": -2.0,
                "exec_fail_pct": 0.0,
                "parse_fail_pct": 0.0,
                "frac_non_normalized": 0.1,
                "mean_abs_log_mass": 0.05,
            }
        },
    }

    monkeypatch.setattr(module, "_run_analyze_hacking", lambda *args, **kwargs: dummy_report)

    cond = {
        "name": "cond",
        "label": "Condition",
        "runs": [str(existing_run), str(missing_run)],
    }
    out = tmp_path / "out"

    condition_result, _baseline, manifest_entry = module._process_condition(
        cond=cond,
        out_dir=out,
        baseline_path=None,
        exclude_clipped=False,
        winsorize_f=None,
        baseline_summary=None,
        offline_eval_enabled=False,
        offline_eval_sample=None,
        offline_eval_only_valid=False,
    )

    assert condition_result is not None
    assert manifest_entry is not None
    assert manifest_entry["n_runs_declared"] == 2
    assert manifest_entry["n_runs_used"] == 1
    assert len(manifest_entry["runs_declared"]) == 2
    assert len(manifest_entry["runs"]) == 1
    assert manifest_entry["runs"][0] == str(existing_run.resolve())


def test_parse_run_item_mapping_accepts_overrides() -> None:
    module = _load_module()
    out = module._parse_run_item(
        {
            "path": "artifacts/sweeps/tinker_20260217_163634",
            "paper_track": "part_a_emergence",
            "claim_mode": "formal_lh",
            "logging_valid": True,
        }
    )
    assert out["path"] == "artifacts/sweeps/tinker_20260217_163634"
    assert out["paper_track"] == "part_a_emergence"
    assert out["claim_mode"] == "formal_lh"
    assert out["logging_valid"] is True


def test_process_condition_accepts_run_overrides_without_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()

    run_dir = tmp_path / "archived_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "completions.jsonl").write_text("", encoding="utf-8")

    dummy_report = {
        "per_step": {
            "1": {
                "reported_mean": -2.0,
                "exec_fail_pct": 0.0,
                "parse_fail_pct": 0.0,
                "frac_non_normalized": 0.1,
                "mean_abs_log_mass": 0.05,
            }
        },
    }
    monkeypatch.setattr(module, "_run_analyze_hacking", lambda *args, **kwargs: dummy_report)

    cond = {
        "name": "archived",
        "label": "Archived",
        "runs": [
            {
                "path": str(run_dir),
                "paper_track": "part_a_emergence",
                "claim_mode": "formal_lh",
                "logging_valid": True,
            }
        ],
    }

    condition_result, _baseline, manifest_entry = module._process_condition(
        cond=cond,
        out_dir=tmp_path / "out",
        baseline_path=None,
        exclude_clipped=False,
        winsorize_f=None,
        baseline_summary=None,
        offline_eval_enabled=False,
        offline_eval_sample=None,
        offline_eval_only_valid=False,
    )

    assert condition_result is not None
    assert manifest_entry is not None
    assert manifest_entry["n_runs_used"] == 1
    assert manifest_entry["runs"][0] == str(run_dir.resolve())


def test_setup_plot_style_uses_times_new_roman() -> None:
    module = _load_module()
    module._setup_plot_style()
    assert "Times New Roman" in matplotlib.rcParams["font.serif"]


def test_normalization_panel_skips_nan_only_data() -> None:
    module = _load_module()
    condition_summaries = [
        {
            "label": "Run 6",
            "reward": {"steps": [1, 2], "mean": [0.1, 0.2], "std": [0.0, 0.0]},
            "offline_eval_aggregate": {
                "normalization_frac_non_normalized": {
                    "mean": float("nan"),
                    "std": float("nan"),
                }
            },
        }
    ]
    bar_labels, bar_vals = module._normalization_bars(condition_summaries)
    assert bar_labels == []
    assert bar_vals == []

    fig, _ax_reward, ax_norm = module._make_reward_figure(has_normalization_panel=bool(bar_labels))
    assert ax_norm is None
    plt.close(fig)


def test_reward_and_normalization_axes_do_not_share_x() -> None:
    module = _load_module()
    fig, ax_reward, ax_norm = module._make_reward_figure(has_normalization_panel=True)
    assert ax_norm is not None
    assert not ax_reward.get_shared_x_axes().joined(ax_reward, ax_norm)
    plt.close(fig)
