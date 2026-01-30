from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ppl_synthesis_reward_hacking.config.load import load_config


def plot_from_aggregate(aggregate_path: Path, *, config_path: Path) -> None:
    config = load_config(config_path)
    out_path = Path(config.get("out", "artifacts/plots/summary.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    with aggregate_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t" if aggregate_path.suffix == ".tsv" else ",")
        for row in reader:
            key = (row.get("backend"), row.get("metric_name"))
            try:
                value = float(row.get("metric_value", "nan"))
            except ValueError:
                continue
            groups[key].append(value)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["backend", "metric_name", "mean_value"])
        writer.writeheader()
        for (backend, metric_name), values in sorted(groups.items()):
            if not values:
                continue
            mean_value = sum(values) / len(values)
            writer.writerow(
                {"backend": backend, "metric_name": metric_name, "mean_value": mean_value}
            )
