#!/usr/bin/env python3
"""Validate paper artifact bundles.

Supports two artifact schemas:
- `legacy`: outputs from scripts/make_paper_artifacts.py
- `current_experiments`: outputs from scripts/prepare_local_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LEGACY_REQUIRED_FILES = (
    "MANIFEST.json",
    "baseline_vs_trained_table.tex",
    "normalization_table.tex",
    "lh_evidence_table.tex",
    "safety_gate_matrix_table.tex",
    "safety_gate_matrix.json",
)

CURRENT_MANIFEST_NAME = "current_experiments_manifest.json"
CURRENT_REQUIRED_STEMS = (
    "lh_emergence",
    "lh_taxonomy_bar",
    "lh_taxonomy_evolution",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate generated paper artifacts")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("paper_artifacts"),
        help="Artifacts directory to validate",
    )
    p.add_argument(
        "--mode",
        choices=("legacy", "current_experiments"),
        default="legacy",
        help="Artifact schema to validate",
    )
    return p.parse_args()


def _validate_manifest(manifest: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest.get("conditions"), list):
        errors.append("MANIFEST.json missing `conditions` list")
        return errors

    for idx, cond in enumerate(manifest["conditions"]):
        if not isinstance(cond, dict):
            errors.append(f"conditions[{idx}] is not an object")
            continue
        for key in ("name", "label", "runs", "n_runs_used"):
            if key not in cond:
                errors.append(f"conditions[{idx}] missing `{key}`")
        claim_modes_used = cond.get("claim_modes_used")
        if not isinstance(claim_modes_used, list):
            errors.append(f"conditions[{idx}] missing `claim_modes_used` list")
        elif any(mode != "formal_lh" for mode in claim_modes_used):
            errors.append(
                f"conditions[{idx}] contains non-formal claim_mode entries: {claim_modes_used}"
            )

        offline_runs = cond.get("offline_eval_runs")
        if isinstance(offline_runs, list) and offline_runs:
            if "offline_eval_aggregate" not in cond:
                errors.append(f"conditions[{idx}] missing `offline_eval_aggregate`")
            else:
                agg = cond["offline_eval_aggregate"]
                required = (
                    "normalization_frac_non_normalized",
                    "normalization_mean_log_mass",
                    "taxonomy_tagged_fraction",
                    "safety_acceptance_rate_checked",
                )
                for key in required:
                    if key not in agg:
                        errors.append(f"conditions[{idx}].offline_eval_aggregate missing `{key}`")
    return errors


def _resolve_path(path_text: str, *, artifacts_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    artifacts_relative = (artifacts_dir / path).resolve()
    if artifacts_relative.exists():
        return artifacts_relative

    repo_relative = (REPO_ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative

    return artifacts_relative


def _validate_generated_files(
    generated: list[str], *, artifacts_dir: Path, out_dir: Path, exp_name: str
) -> list[str]:
    errors: list[str] = []
    resolved: list[Path] = []
    for idx, item in enumerate(generated):
        if not isinstance(item, str):
            errors.append(f"{exp_name}.generated[{idx}] must be a string path")
            continue
        resolved_path = _resolve_path(item, artifacts_dir=artifacts_dir)
        if not resolved_path.exists():
            errors.append(f"{exp_name}.generated[{idx}] missing file: {resolved_path}")
            continue
        try:
            resolved_path.relative_to(out_dir)
        except ValueError:
            errors.append(
                f"{exp_name}.generated[{idx}] escapes out_dir: {resolved_path} (out_dir={out_dir})"
            )
            continue
        resolved.append(resolved_path)

    names = {p.name for p in resolved}
    missing = []
    for stem in CURRENT_REQUIRED_STEMS:
        for ext in (".pdf", ".png"):
            expected = f"{stem}{ext}"
            if expected not in names:
                missing.append(expected)

    if missing:
        errors.append(
            f"{exp_name} missing required generated files: "
            + ", ".join(sorted(missing))
        )
    return errors


def _validate_current_experiments_manifest(manifest: dict, *, artifacts_dir: Path) -> list[str]:
    errors: list[str] = []
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return ["current_experiments_manifest.json missing non-empty `experiments` list"]

    for idx, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            errors.append(f"experiments[{idx}] is not an object")
            continue
        name = exp.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"experiments[{idx}] missing non-empty `name`")
            continue

        run_dir_raw = exp.get("run_dir")
        out_dir_raw = exp.get("out_dir")
        summary = exp.get("summary")
        generated = exp.get("generated")

        if not isinstance(run_dir_raw, str):
            errors.append(f"{name}.run_dir must be a string")
            continue
        if not isinstance(out_dir_raw, str):
            errors.append(f"{name}.out_dir must be a string")
            continue
        if not isinstance(summary, dict):
            errors.append(f"{name}.summary must be an object")
            continue
        if not isinstance(generated, list) or not generated:
            errors.append(f"{name}.generated must be a non-empty list")
            continue

        run_dir = _resolve_path(run_dir_raw, artifacts_dir=artifacts_dir)
        out_dir = _resolve_path(out_dir_raw, artifacts_dir=artifacts_dir)
        if not run_dir.exists():
            errors.append(f"{name}.run_dir does not exist: {run_dir}")
        if not (run_dir / "completions.jsonl").exists():
            errors.append(f"{name} missing completions.jsonl in run_dir: {run_dir}")
        if not out_dir.exists():
            errors.append(f"{name}.out_dir does not exist: {out_dir}")

        for key in ("records", "valid_records", "max_batch"):
            if key not in summary:
                errors.append(f"{name}.summary missing `{key}`")
        records = summary.get("records")
        valid_records = summary.get("valid_records")
        valid_rate = summary.get("valid_rate")
        if isinstance(records, int) and records < 0:
            errors.append(f"{name}.summary.records must be >= 0")
        if isinstance(valid_records, int) and valid_records < 0:
            errors.append(f"{name}.summary.valid_records must be >= 0")
        if isinstance(records, int) and isinstance(valid_records, int) and valid_records > records:
            errors.append(f"{name}.summary.valid_records cannot exceed records")
        if valid_rate is not None:
            if not isinstance(valid_rate, int | float):
                errors.append(f"{name}.summary.valid_rate must be numeric when present")
            elif not 0.0 <= float(valid_rate) <= 1.0:
                errors.append(f"{name}.summary.valid_rate must be within [0, 1]")

        errors.extend(
            _validate_generated_files(
                generated,
                artifacts_dir=artifacts_dir,
                out_dir=out_dir,
                exp_name=name,
            )
        )

    return errors


def _validate_legacy(artifacts_dir: Path) -> list[str]:
    errors: list[str] = []
    for name in LEGACY_REQUIRED_FILES:
        path = artifacts_dir / name
        if not path.exists():
            errors.append(f"missing required artifact: {path}")

    manifest_path = artifacts_dir / "MANIFEST.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            errors.extend(_validate_manifest(manifest))
        except Exception as exc:
            errors.append(f"failed to parse MANIFEST.json: {exc}")
    return errors


def _validate_current_experiments(artifacts_dir: Path) -> list[str]:
    manifest_path = artifacts_dir / CURRENT_MANIFEST_NAME
    if not manifest_path.exists():
        return [f"missing required artifact: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"failed to parse {CURRENT_MANIFEST_NAME}: {exc}"]
    return _validate_current_experiments_manifest(manifest, artifacts_dir=artifacts_dir)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()

    if args.mode == "legacy":
        errors = _validate_legacy(artifacts_dir)
    else:
        errors = _validate_current_experiments(artifacts_dir)

    if errors:
        print("Artifact validation FAILED:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(2)

    print(f"Artifact validation passed for {artifacts_dir} (mode={args.mode})")


if __name__ == "__main__":
    main()
