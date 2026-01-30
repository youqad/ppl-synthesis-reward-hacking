from pathlib import Path

from ppl_synthesis_reward_hacking.experiments.lh_protocol import run_lh_protocol


def test_run_lh_protocol_creates_run(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        """
experiment:
  name: toy_test
backend:
  name: toy
data:
  name: bernoulli_1d
  params:
    p_true: 0.5
  split:
    n_train: 4
    n_holdout: 6
run:
  seed: 1
  repetitions: 1
""",
        encoding="utf-8",
    )

    run_ids = run_lh_protocol(
        config_path=str(config_path),
        backend_override=None,
        overrides=[],
        seed_override=None,
    )
    assert len(run_ids) == 1
    run_dir = Path("artifacts") / "runs" / run_ids[0]
    assert (run_dir / "run.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "config.resolved.yaml").exists()
