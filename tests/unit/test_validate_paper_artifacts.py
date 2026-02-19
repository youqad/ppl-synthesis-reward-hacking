from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "validate_paper_artifacts.py"
    spec = importlib.util.spec_from_file_location("validate_paper_artifacts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_manifest_accepts_offline_aggregate_shape() -> None:
    module = _load_module()
    manifest = {
        "conditions": [
            {
                "name": "run6",
                "label": "Run 6",
                "runs": ["artifacts/tinker_poc"],
                "n_runs_used": 1,
                "claim_modes_used": ["formal_lh"],
                "offline_eval_runs": [{"dummy": True}],
                "offline_eval_aggregate": {
                    "normalization_frac_non_normalized": {},
                    "normalization_mean_log_mass": {},
                    "taxonomy_tagged_fraction": {},
                    "safety_acceptance_rate_checked": {},
                },
            }
        ]
    }
    errors = module._validate_manifest(manifest)
    assert errors == []


def test_validate_manifest_flags_missing_aggregate_keys() -> None:
    module = _load_module()
    manifest = {
        "conditions": [
            {
                "name": "run6",
                "label": "Run 6",
                "runs": ["artifacts/tinker_poc"],
                "n_runs_used": 1,
                "claim_modes_used": ["formal_lh"],
                "offline_eval_runs": [{"dummy": True}],
                "offline_eval_aggregate": {},
            }
        ]
    }
    errors = module._validate_manifest(manifest)
    assert any("normalization_frac_non_normalized" in err for err in errors)


def test_validate_current_experiments_manifest_accepts_expected_shape(tmp_path: Path) -> None:
    module = _load_module()

    artifacts_dir = tmp_path / "paper_artifacts"
    artifacts_dir.mkdir()
    run_dir = tmp_path / "run7_seed0"
    run_dir.mkdir()
    (run_dir / "completions.jsonl").write_text("{}", encoding="utf-8")
    out_dir = artifacts_dir / "run7_seed0"
    out_dir.mkdir()

    generated: list[str] = []
    for stem in ("lh_emergence", "lh_taxonomy_bar", "lh_taxonomy_evolution"):
        for ext in (".pdf", ".png"):
            path = out_dir / f"{stem}{ext}"
            path.write_text("x", encoding="utf-8")
            generated.append(str(path))

    manifest = {
        "experiments": [
            {
                "name": "run7_seed0",
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "summary": {
                    "records": 10,
                    "valid_records": 8,
                    "valid_rate": 0.8,
                    "max_batch": 1,
                },
                "generated": generated,
            }
        ]
    }
    errors = module._validate_current_experiments_manifest(
        manifest, artifacts_dir=artifacts_dir
    )
    assert errors == []


def test_validate_current_experiments_manifest_flags_missing_required_outputs(
    tmp_path: Path,
) -> None:
    module = _load_module()

    artifacts_dir = tmp_path / "paper_artifacts"
    artifacts_dir.mkdir()
    run_dir = tmp_path / "run7_seed0"
    run_dir.mkdir()
    (run_dir / "completions.jsonl").write_text("{}", encoding="utf-8")
    out_dir = artifacts_dir / "run7_seed0"
    out_dir.mkdir()

    only_one = out_dir / "lh_emergence.pdf"
    only_one.write_text("x", encoding="utf-8")
    manifest = {
        "experiments": [
            {
                "name": "run7_seed0",
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "summary": {
                    "records": 10,
                    "valid_records": 8,
                    "valid_rate": 0.8,
                    "max_batch": 1,
                },
                "generated": [str(only_one)],
            }
        ]
    }

    errors = module._validate_current_experiments_manifest(
        manifest, artifacts_dir=artifacts_dir
    )
    assert any("missing required generated files" in err for err in errors)


def test_validate_current_experiments_manifest_rejects_generated_paths_outside_out_dir(
    tmp_path: Path,
) -> None:
    module = _load_module()

    artifacts_dir = tmp_path / "paper_artifacts"
    artifacts_dir.mkdir()
    run_dir = tmp_path / "run7_seed0"
    run_dir.mkdir()
    (run_dir / "completions.jsonl").write_text("{}", encoding="utf-8")
    out_dir = artifacts_dir / "run7_seed0"
    out_dir.mkdir()

    rogue_dir = tmp_path / "rogue"
    rogue_dir.mkdir()
    rogue_file = rogue_dir / "lh_emergence.pdf"
    rogue_file.write_text("x", encoding="utf-8")

    generated = [str(rogue_file)]
    manifest = {
        "experiments": [
            {
                "name": "run7_seed0",
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "summary": {
                    "records": 10,
                    "valid_records": 8,
                    "valid_rate": 0.8,
                    "max_batch": 1,
                },
                "generated": generated,
            }
        ]
    }

    errors = module._validate_current_experiments_manifest(
        manifest, artifacts_dir=artifacts_dir
    )
    assert any("escapes out_dir" in err for err in errors)
