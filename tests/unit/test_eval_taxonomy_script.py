from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "eval_taxonomy.py"
    spec = importlib.util.spec_from_file_location("eval_taxonomy", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_record(
    *,
    index: int = 0,
    completion_text: str,
    code: str | None,
    outcome: str = "parse_fail",
    metadata: dict | None = None,
) -> CompletionRecord:
    return CompletionRecord(
        batch=0,
        index=index,
        prompt="p",
        completion_text=completion_text,
        code=code,
        reported_reward=-100.0,
        outcome=outcome,
        timestamp="2026-01-01T00:00:00Z",
        metadata=metadata,
    )


def test_taxonomy_source_text_preserves_initval_parse_fail_completion() -> None:
    module = _load_module()
    record = _make_record(
        completion_text=(
            "```python\n"
            "import pymc as pm\n"
            "def model(data):\n"
            "    with pm.Model() as m:\n"
            "        p = pm.Beta('p', 1, 1, initval=0.99)\n"
            "        pm.Bernoulli('y', p=p, observed=data['y'])\n"
            "    return m\n"
            "```"
        ),
        code=None,
    )
    source = module._taxonomy_source_text(record)
    assert source is not None
    assert "initval=0.99" in source


def test_taxonomy_source_text_skips_non_code_text_without_init_pattern() -> None:
    module = _load_module()
    record = _make_record(completion_text="I cannot provide a model right now.", code=None)
    assert module._taxonomy_source_text(record) is None


def test_taxonomy_source_text_skips_plain_prose_with_initval_assignment() -> None:
    module = _load_module()
    record = _make_record(
        completion_text="I cannot provide runnable code, but initval = 0.99 is a common phrase.",
        code=None,
    )
    assert module._taxonomy_source_text(record) is None


def test_main_keeps_initval_records_in_taxonomy_output(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    completion_text = (
        "```python\n"
        "import pymc as pm\n"
        "def model(data):\n"
        "    with pm.Model() as m:\n"
        "        p = pm.Beta('p', 1, 1, initval=0.99)\n"
        "        pm.Bernoulli('y', p=p, observed=data['y'])\n"
        "    return m\n"
        "```"
    )
    record = _make_record(completion_text=completion_text, code=None)

    completions_path = tmp_path / "completions.jsonl"
    completions_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "eval"

    monkeypatch.setattr(module, "load_completions", lambda _path: [record])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            completions=str(completions_path),
            judge_verdicts=None,
            output_dir=str(output_dir),
            only_valid=False,
            dedup=True,
            sample=None,
            seed=42,
        ),
    )

    module.main()

    rows_path = output_dir / "taxonomy.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any("evaluator_hack_initval" in row["exploit_tags"] for row in rows)


def test_main_dedup_merges_metadata_driven_tags(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    code = (
        "import pymc as pm\n"
        "def model(data):\n"
        "    with pm.Model() as m:\n"
        "        p = pm.Beta('p', 1, 1)\n"
        "    return m\n"
    )
    records = [
        _make_record(
            index=0,
            completion_text=f"```python\n{code}```",
            code=code,
            outcome="valid",
            metadata={"decomposition": {"n_obs_terms": 1}},
        ),
        _make_record(
            index=1,
            completion_text=f"```python\n{code}```",
            code=code,
            outcome="valid",
            metadata={"decomposition": {"n_obs_terms": 0}},
        ),
    ]

    completions_path = tmp_path / "completions.jsonl"
    completions_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "eval"

    monkeypatch.setattr(module, "load_completions", lambda _path: records)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            completions=str(completions_path),
            judge_verdicts=None,
            output_dir=str(output_dir),
            only_valid=False,
            dedup=True,
            sample=None,
            seed=42,
        ),
    )

    module.main()

    rows_path = output_dir / "taxonomy.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert "lh_data_discard" in rows[0]["exploit_tags"]


def test_main_surfaces_parse_fail_without_code(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    record = _make_record(
        completion_text="I could not produce valid model code this time.",
        code=None,
        outcome="parse_fail",
    )

    completions_path = tmp_path / "completions.jsonl"
    completions_path.write_text("", encoding="utf-8")
    output_dir = tmp_path / "eval"

    monkeypatch.setattr(module, "load_completions", lambda _path: [record])
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            completions=str(completions_path),
            judge_verdicts=None,
            output_dir=str(output_dir),
            only_valid=False,
            dedup=True,
            sample=None,
            seed=42,
        ),
    )

    module.main()

    rows_path = output_dir / "taxonomy.jsonl"
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["taxonomy_status"] == "no_code_parse_fail"
    assert rows[0]["exploit_tags"] == []

    summary_path = output_dir / "taxonomy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_parse_fail_no_code"] == 1
    assert summary["n_skipped_no_code_non_parse"] == 0
