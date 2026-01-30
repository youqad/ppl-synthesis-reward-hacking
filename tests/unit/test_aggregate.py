import csv
import json

from ppl_synthesis_reward_hacking.evaluation.aggregators import aggregate_runs


def test_aggregate_runs(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run123"
    run_dir.mkdir(parents=True)

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run123",
                "config_hash": "abc",
                "meta": {
                    "experiment": "toy",
                    "backend": "toy",
                    "dataset_id": "ds1",
                    "seed": 0,
                    "rep": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric_name", "metric_value", "meta_json"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "metric_name": "reported_reward_holdout",
                "metric_value": "1.0",
                "meta_json": "{}",
            }
        )

    out_path = tmp_path / "aggregate.csv"
    aggregate_runs(runs_dir, out_path)

    rows = list(csv.DictReader(out_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run123"
    assert rows[0]["metric_name"] == "reported_reward_holdout"
