from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from .schema import MetricRecord, RunMetadata


def write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": metadata.run_id,
        "config_hash": metadata.config_hash,
        "git_commit": metadata.git_commit,
        "git_dirty": metadata.git_dirty,
        "python_version": metadata.python_version,
        "platform": metadata.platform,
        "meta": metadata.meta,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_metrics(path: Path, metrics: Iterable[MetricRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(metrics)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric_name", "metric_value", "meta_json"],
        )
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    "metric_name": record.metric_name,
                    "metric_value": record.metric_value,
                    "meta_json": json.dumps(record.meta, sort_keys=True),
                }
            )


def write_metrics_parquet(path: Path, metrics: Iterable[MetricRecord]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for Parquet output. Install with: "
            "uv pip install ppl-synthesis-reward-hacking[parquet]"
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(metrics)

    names = [r.metric_name for r in rows]
    values = [r.metric_value for r in rows]
    metas = [json.dumps(r.meta, sort_keys=True) for r in rows]

    table = pa.table(
        {
            "metric_name": pa.array(names, type=pa.string()),
            "metric_value": pa.array(values, type=pa.float64()),
            "meta_json": pa.array(metas, type=pa.string()),
        }
    )
    pq.write_table(table, path)
