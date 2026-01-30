from __future__ import annotations

import csv
import json
from pathlib import Path


def aggregate_runs(runs_dir: Path, out_path: Path) -> None:
    if out_path.suffix not in {".csv", ".tsv"}:
        raise ValueError("Only .csv and .tsv aggregates are supported for now")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if out_path.suffix == ".tsv" else ","

    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run_json = run_dir / "run.json"
        metrics_csv = run_dir / "metrics.csv"
        if not run_json.exists() or not metrics_csv.exists():
            continue

        run_data = json.loads(run_json.read_text(encoding="utf-8"))
        meta = run_data.get("meta") or {}
        with metrics_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                rows.append(
                    {
                        "run_id": run_data.get("run_id"),
                        "config_hash": run_data.get("config_hash"),
                        "experiment": meta.get("experiment"),
                        "backend": meta.get("backend"),
                        "dataset_id": meta.get("dataset_id"),
                        "seed": meta.get("seed"),
                        "rep": meta.get("rep"),
                        "metric_name": record.get("metric_name"),
                        "metric_value": record.get("metric_value"),
                    }
                )

    fieldnames = [
        "run_id",
        "config_hash",
        "experiment",
        "backend",
        "dataset_id",
        "seed",
        "rep",
        "metric_name",
        "metric_value",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
