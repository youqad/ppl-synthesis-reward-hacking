#!/usr/bin/env python3
"""Validate required paper artifact files and manifest structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FILES = (
    "MANIFEST.json",
    "baseline_vs_trained_table.tex",
    "normalization_table.tex",
    "lh_evidence_table.tex",
    "safety_gate_matrix_table.tex",
    "safety_gate_matrix.json",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate generated paper artifacts")
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("paper_artifacts"),
        help="Directory created by scripts/make_paper_artifacts.py",
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


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir.resolve()
    errors: list[str] = []

    for name in REQUIRED_FILES:
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

    if errors:
        print("Artifact validation FAILED:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(2)

    print(f"Artifact validation passed for {artifacts_dir}")


if __name__ == "__main__":
    main()
