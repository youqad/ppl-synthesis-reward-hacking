from __future__ import annotations

from types import SimpleNamespace

from ppl_synthesis_reward_hacking.experiments import tinker_training as module
from ppl_synthesis_reward_hacking.experiments.tinker_training import (
    TinkerGRPOConfig,
    _deduplicate_prompt_templates,
    _initialize_training,
    _select_prompt_texts_from_pool,
)


class _FakeTrainingClient:
    def get_tokenizer(self) -> object:
        return object()


class _FakeServiceClient:
    def create_lora_training_client(self, *, base_model: str, rank: int) -> _FakeTrainingClient:
        return _FakeTrainingClient()


def test_seeded_shuffle_cycle_duplicate_pool_can_repeat() -> None:
    duplicated_pool = ["prompt_a", "prompt_a", "prompt_b", "prompt_c"]

    selected = _select_prompt_texts_from_pool(
        duplicated_pool,
        n_prompts=3,
        sampling_mode="seeded_shuffle_cycle",
        seed=0,
    )

    assert len(selected) == 3
    assert len(set(selected)) < 3


def test_initialize_training_hardcoded_deduplicates_prompt_pool(
    monkeypatch,
    tmp_path,
) -> None:
    duplicated_pool = ["prompt_a", "prompt_a", "prompt_b", "prompt_c"]

    monkeypatch.setattr(
        module,
        "get_prompts_for_dataset",
        lambda dataset_name, n: list(duplicated_pool),
    )
    monkeypatch.setattr(
        module,
        "get_system_prompt",
        lambda dataset_name, *, prompt_policy="legacy": "system",
    )
    monkeypatch.setattr(module, "_load_exemplars", lambda config_path: [])
    monkeypatch.setattr(module, "_load_judge_env", lambda env_file: None)
    monkeypatch.setattr(module, "tinker", SimpleNamespace(ServiceClient=_FakeServiceClient))

    config = TinkerGRPOConfig(
        prompt_source="hardcoded",
        n_prompts=3,
        prompt_jsonl_max_examples=4,
        scoring_seed_base=0,
        output_dir=str(tmp_path / "out"),
    )

    writer, _, _, _, _, prompt_specs = _initialize_training(config, tmp_path / "out")
    writer.close()

    selected = [spec.user_message for spec in prompt_specs]
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert selected == _select_prompt_texts_from_pool(
        _deduplicate_prompt_templates(duplicated_pool),
        n_prompts=3,
        sampling_mode="seeded_shuffle_cycle",
        seed=0,
    )


def test_initialize_training_jsonl_pool_is_not_deduplicated(monkeypatch, tmp_path) -> None:
    jsonl_pool = ["prompt_a", "prompt_a", "prompt_b"]
    seen_pool: dict[str, list[str]] = {}

    def _capture_pool(
        pool: list[str],
        *,
        n_prompts: int,
        sampling_mode: str,
        seed: int,
    ) -> list[str]:
        seen_pool["value"] = list(pool)
        return list(pool[:n_prompts])

    monkeypatch.setattr(
        module,
        "load_jsonl_prompts",
        lambda path, max_examples: list(jsonl_pool),
    )
    monkeypatch.setattr(
        module,
        "get_generic_system_prompt",
        lambda *, prompt_policy="legacy": "system",
    )
    monkeypatch.setattr(module, "_load_exemplars", lambda config_path: [])
    monkeypatch.setattr(module, "_load_judge_env", lambda env_file: None)
    monkeypatch.setattr(module, "_select_prompt_texts_from_pool", _capture_pool)
    monkeypatch.setattr(module, "tinker", SimpleNamespace(ServiceClient=_FakeServiceClient))

    config = TinkerGRPOConfig(
        prompt_source="jsonl",
        prompt_jsonl_path="dummy.jsonl",
        n_prompts=3,
        prompt_jsonl_max_examples=3,
        output_dir=str(tmp_path / "out_jsonl"),
    )

    writer, _, _, _, _, prompt_specs = _initialize_training(config, tmp_path / "out_jsonl")
    writer.close()

    assert seen_pool["value"] == jsonl_pool
    assert [spec.user_message for spec in prompt_specs] == jsonl_pool


def test_load_exemplars_empty_yaml_fails_fast(tmp_path) -> None:
    path = tmp_path / "exemplars.yaml"
    path.write_text("", encoding="utf-8")

    try:
        module._load_exemplars(str(path))
        raise AssertionError("expected invalid exemplar config error")
    except ValueError as exc:
        assert "invalid exemplar config" in str(exc)


def test_load_exemplars_rejects_missing_required_item_fields(tmp_path) -> None:
    path = tmp_path / "exemplars.yaml"
    path.write_text("exemplars:\n  - name: only_name\n", encoding="utf-8")

    try:
        module._load_exemplars(str(path))
        raise AssertionError("expected invalid exemplar item error")
    except ValueError as exc:
        assert "missing string fields" in str(exc)
