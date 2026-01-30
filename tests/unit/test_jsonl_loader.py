from __future__ import annotations

from pathlib import Path

from ppl_synthesis_reward_hacking.data.jsonl_loader import load_jsonl_dataset


def test_load_synthstats_val_dataset() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / "data" / "synthstats" / "synthstats_val.jsonl"

    examples = load_jsonl_dataset(dataset_path, max_examples=3)

    assert len(examples) > 0
    for example in examples:
        assert example.prompt.strip()
        assert example.completion.strip()
