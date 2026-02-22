from __future__ import annotations

from pathlib import Path

from ppl_synthesis_reward_hacking.data.jsonl_loader import load_jsonl_dataset
from ppl_synthesis_reward_hacking.data.pymc_reward_loader import load_pymc_reward_prompts


def test_load_pymc_val() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / "data" / "pymc_reward" / "pymc_reward_val.jsonl"

    examples = load_jsonl_dataset(dataset_path, max_examples=3)

    assert len(examples) > 0
    for example in examples:
        assert example.prompt.strip()
        assert example.completion.strip()


def test_load_pymc_reward_prompts_jsonl_honors_sampling_and_thinking_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "prompts.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"prompt":"prompt-A","completion":"c"}',
                '{"prompt":"prompt-B","completion":"c"}',
                '{"prompt":"prompt-C","completion":"c"}',
                '{"prompt":"prompt-D","completion":"c"}',
                '{"prompt":"prompt-E","completion":"c"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sample_a = load_pymc_reward_prompts(
        prompt_source="jsonl",
        prompt_path=dataset_path,
        max_examples=5,
        n_prompts=5,
        prompt_sampling="seeded_shuffle_cycle",
        prompt_sampling_seed=17,
        thinking_mode="no_think",
    )
    sample_b = load_pymc_reward_prompts(
        prompt_source="jsonl",
        prompt_path=dataset_path,
        max_examples=5,
        n_prompts=5,
        prompt_sampling="seeded_shuffle_cycle",
        prompt_sampling_seed=17,
        thinking_mode="no_think",
    )
    sample_c = load_pymc_reward_prompts(
        prompt_source="jsonl",
        prompt_path=dataset_path,
        max_examples=5,
        n_prompts=5,
        prompt_sampling="seeded_shuffle_cycle",
        prompt_sampling_seed=18,
        thinking_mode="no_think",
    )

    assert len(sample_a) == 5
    assert sample_a == sample_b
    assert sample_a != sample_c
    assert sample_a[0]["prompt"][0]["content"].startswith("/no_think\n")


def test_load_pymc_reward_prompts_rejects_unknown_prompt_sampling(tmp_path: Path) -> None:
    dataset_path = tmp_path / "prompts.jsonl"
    dataset_path.write_text('{"prompt":"p","completion":"c"}\n', encoding="utf-8")

    try:
        load_pymc_reward_prompts(
            prompt_source="jsonl",
            prompt_path=dataset_path,
            prompt_sampling="unknown_mode",
        )
        raise AssertionError("expected prompt_sampling validation error")
    except ValueError as exc:
        assert "prompt_sampling must be one of" in str(exc)
