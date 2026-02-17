from __future__ import annotations

import importlib.util
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
                "gap_pm_std": "1.200 $\\pm$ 0.100",
                "safety_acceptance_pm_std": "60.0 $\\pm$ 5.0",
            }
        ],
    )
    text = out_path.read_text(encoding="utf-8")
    assert "Gap" in text
    assert "Safety accept (\\%)" in text
    assert "Mean $|$log-mass$|$" in text
    assert "Run 6" in text
    # normalization columns come before Gap (Gap is last data column)
    header_line = [line for line in text.splitlines() if "Non-normalized" in line][0]
    norm_pos = header_line.index("Non-normalized")
    gap_pos = header_line.index("Gap")
    assert norm_pos < gap_pos, "normalization columns should precede Gap"


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
    missing_run = tmp_path / "missing_run"

    dummy_report = {
        "per_step": {
            "1": {
                "gap_mean": 0.1,
                "exec_fail_pct": 0.0,
                "parse_fail_pct": 0.0,
            }
        },
        "gap_trajectory": {"last_step_mean": 0.1},
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


def test_setup_plot_style_uses_times_new_roman() -> None:
    module = _load_module()
    module._setup_plot_style()
    assert "Times New Roman" in matplotlib.rcParams["font.serif"]


def test_normalization_panel_skips_nan_only_data() -> None:
    module = _load_module()
    condition_summaries = [
        {
            "label": "Run 6",
            "gap": {"steps": [1, 2], "mean": [0.1, 0.2], "std": [0.0, 0.0]},
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

    fig, _ax_gap, ax_norm = module._make_gap_figure(has_normalization_panel=bool(bar_labels))
    assert ax_norm is None
    plt.close(fig)


def test_gap_and_normalization_axes_do_not_share_x() -> None:
    module = _load_module()
    fig, ax_gap, ax_norm = module._make_gap_figure(has_normalization_panel=True)
    assert ax_norm is not None
    assert not ax_gap.get_shared_x_axes().joined(ax_gap, ax_norm)
    plt.close(fig)
