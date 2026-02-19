#!/usr/bin/env python3
"""Prepare local-only paper figures for the current experiments.

This script orchestrates `scripts/make_grpo_figures.py` for a run list and
writes outputs into the local paper clone under `background-documents/...-local`.
It never touches the Overleaf git clone.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper" / "current_experiments.yaml"


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError(f"config must be a mapping: {path}")
    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local paper figures for current experiments"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config with experiment run list",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Override output root (defaults to config `out_root`)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional explicit manifest path "
            "(defaults to <out_root>/current_experiments_manifest.json)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    return parser.parse_args()


def _validate_experiments(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    experiments = cfg.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise RuntimeError("config must contain non-empty `experiments` list")

    validated: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(experiments):
        if not isinstance(item, dict):
            raise RuntimeError(f"experiments[{idx}] must be a mapping")

        name = item.get("name")
        run_dir = item.get("run_dir")
        late_batch_cutoff = item.get("late_batch_cutoff", 800)

        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"experiments[{idx}].name must be a non-empty string")
        if name in seen_names:
            raise RuntimeError(f"duplicate experiment name: {name}")
        seen_names.add(name)

        if not isinstance(run_dir, str) or not run_dir.strip():
            raise RuntimeError(f"experiments[{idx}].run_dir must be a non-empty string")
        if not isinstance(late_batch_cutoff, int):
            raise RuntimeError(f"experiments[{idx}].late_batch_cutoff must be an int")

        validated.append(
            {
                "name": name,
                "run_dir": _resolve_path(run_dir),
                "late_batch_cutoff": late_batch_cutoff,
            }
        )
    return validated


def _run_one(
    *,
    run_dir: Path,
    out_dir: Path,
    late_batch_cutoff: int,
    dry_run: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/make_grpo_figures.py",
        "--run-dir",
        str(run_dir),
        "--out",
        str(out_dir),
        "--late-batch-cutoff",
        str(late_batch_cutoff),
    ]
    print(f"  run: {' '.join(cmd)}", flush=True)

    if dry_run:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(cmd, cwd=REPO_ROOT)

    generated = []
    for suffix in (".pdf", ".png", ".tex"):
        for path in sorted(out_dir.glob(f"*{suffix}")):
            generated.append(str(path.relative_to(REPO_ROOT)))
    return generated


def _summarize_completions(path: Path) -> dict[str, Any]:
    records = 0
    valid = 0
    max_batch = -1
    abs_log_masses: list[float] = []
    n_non_normalized = 0
    n_norm_checked = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records += 1
            batch = rec.get("batch")
            if isinstance(batch, int) and batch > max_batch:
                max_batch = batch
            if rec.get("outcome") == "valid":
                valid += 1
                meta = rec.get("metadata") or {}
                norm = meta.get("normalization")
                if isinstance(norm, dict):
                    n_norm_checked += 1
                    lm = norm.get("log_mass")
                    if lm is not None:
                        lm_f = float(lm)
                        if math.isfinite(lm_f):
                            abs_log_masses.append(abs(lm_f))
                    if not norm.get("is_normalized", True):
                        n_non_normalized += 1

    valid_rate = (float(valid) / float(records)) if records > 0 else None
    frac_nn = (n_non_normalized / n_norm_checked) if n_norm_checked > 0 else None
    mean_abs_lm = float(sum(abs_log_masses) / len(abs_log_masses)) if abs_log_masses else None
    return {
        "records": records,
        "valid_records": valid,
        "valid_rate": valid_rate,
        "max_batch": max_batch,
        "n_norm_checked": n_norm_checked,
        "frac_non_normalized": frac_nn,
        "mean_abs_log_mass": mean_abs_lm,
    }


def main() -> None:
    args = _parse_args()

    cfg_path = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    cfg_path = cfg_path.resolve()
    cfg = _load_config(cfg_path)

    if args.out_root is not None:
        out_root = args.out_root if args.out_root.is_absolute() else (REPO_ROOT / args.out_root)
    else:
        out_root_raw = cfg.get("out_root")
        if not isinstance(out_root_raw, str) or not out_root_raw.strip():
            raise RuntimeError(
                "config must define string `out_root` "
                "when --out-root is not provided"
            )
        out_root = _resolve_path(out_root_raw)
    out_root = out_root.resolve()

    experiments = _validate_experiments(cfg)
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else out_root / "current_experiments_manifest.json"
    )
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else (REPO_ROOT / manifest_path).resolve()
    )

    print("Preparing local paper figures", flush=True)
    print(f"  config:  {cfg_path}", flush=True)
    print(f"  out:     {out_root}", flush=True)
    print(f"  dry_run: {args.dry_run}", flush=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(cfg_path),
        "out_root": str(out_root),
        "dry_run": bool(args.dry_run),
        "experiments": [],
    }

    for item in experiments:
        name = item["name"]
        run_dir = item["run_dir"]
        late_batch_cutoff = item["late_batch_cutoff"]
        print(f"\n{name}", flush=True)
        print(f"  run_dir: {run_dir}", flush=True)
        print(f"  cutoff:  {late_batch_cutoff}", flush=True)

        if not run_dir.exists():
            raise RuntimeError(f"run_dir does not exist: {run_dir}")
        completions_path = run_dir / "completions.jsonl"
        if not completions_path.exists():
            raise RuntimeError(f"missing completions.jsonl: {completions_path}")

        summary = _summarize_completions(completions_path)
        summary_line = (
            f"  summary: records={summary['records']}"
            f" valid={summary['valid_records']}"
        )
        if summary["valid_rate"] is not None:
            summary_line += f" valid_rate={summary['valid_rate']:.3f}"
        print(summary_line, flush=True)

        detail_line = f"           max_batch={summary['max_batch']}"
        if summary["n_norm_checked"] and summary["n_norm_checked"] > 0:
            detail_line += f" norm_checked={summary['n_norm_checked']}"
            if summary["frac_non_normalized"] is not None:
                detail_line += f" frac_non_normalized={summary['frac_non_normalized']:.3f}"
            if summary["mean_abs_log_mass"] is not None:
                detail_line += f" mean_abs_log_mass={summary['mean_abs_log_mass']:.3f}"
        print(detail_line, flush=True)

        out_dir = out_root / name
        generated = _run_one(
            run_dir=run_dir,
            out_dir=out_dir,
            late_batch_cutoff=late_batch_cutoff,
            dry_run=args.dry_run,
        )
        manifest["experiments"].append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "late_batch_cutoff": late_batch_cutoff,
                "summary": summary,
                "generated": generated,
            }
        )

    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote manifest: {manifest_path}", flush=True)
    else:
        print("\nDry run complete; no files were written.", flush=True)


if __name__ == "__main__":
    main()
