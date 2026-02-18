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
