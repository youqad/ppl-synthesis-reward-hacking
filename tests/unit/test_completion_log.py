"""Tests for completion logging: round-trip serialization, writer flush, edge cases."""

from __future__ import annotations

import json
import math

from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import PARSE_FAIL_REWARD
from ppl_synthesis_reward_hacking.logging.completions import (
    CompletionRecord,
    CompletionWriter,
    _deserialize_record,
    _serialize_record,
    load_completions,
    make_timestamp,
)


def _make_record(**overrides) -> CompletionRecord:
    defaults = {
        "batch": 1,
        "index": 0,
        "prompt": "Write a PyMC model for binary data.",
        "completion_text": "import pymc as pm\ndef model(data): ...",
        "code": "def model(data): ...",
        "reported_reward": -3.5,
        "oracle_score": -5.2,
        "gap": 1.7,
        "outcome": "valid",
        "timestamp": "2026-01-28T00:00:00+00:00",
        "metadata": None,
    }
    defaults.update(overrides)
    return CompletionRecord(**defaults)


class TestSerializationRoundTrip:
    def test_basic_round_trip(self):
        rec = _make_record()
        serialized = _serialize_record(rec)
        restored = _deserialize_record(serialized)
        assert restored == rec

    def test_with_metadata(self):
        rec = _make_record(metadata={"n_free_rvs": 3, "logp_breakdown": {"p": -1.2}})
        serialized = _serialize_record(rec)
        restored = _deserialize_record(serialized)
        assert restored == rec
        assert restored.metadata["n_free_rvs"] == 3

    def test_parse_fail_has_none_code(self):
        rec = _make_record(
            code=None, outcome="parse_fail", reported_reward=PARSE_FAIL_REWARD, oracle_score=PARSE_FAIL_REWARD
        )
        serialized = _serialize_record(rec)
        restored = _deserialize_record(serialized)
        assert restored.code is None
        assert restored.outcome == "parse_fail"

    def test_nan_values_survive_json(self):
        rec = _make_record(
            reported_reward=float("nan"), oracle_score=float("nan"), gap=float("nan")
        )
        serialized = _serialize_record(rec)
        # nan -> None in JSON
        assert serialized["reported_reward"] is None
        assert serialized["oracle_score"] is None
        assert serialized["gap"] is None
        # round-trip through JSON string
        json_str = json.dumps(serialized)
        restored = _deserialize_record(json.loads(json_str))
        assert math.isnan(restored.reported_reward)
        assert math.isnan(restored.oracle_score)
        assert math.isnan(restored.gap)

    def test_inf_values_serialize_to_null(self):
        rec = _make_record(reported_reward=float("-inf"), gap=float("inf"))
        serialized = _serialize_record(rec)
        assert serialized["reported_reward"] is None
        assert serialized["gap"] is None


class TestCompletionWriter:
    def test_write_and_load(self, tmp_path):
        path = tmp_path / "completions.jsonl"
        writer = CompletionWriter(path)

        records = [_make_record(index=i, batch=1) for i in range(5)]
        for r in records:
            writer.write(r)
        writer.flush()
        writer.close()

        loaded = load_completions(path)
        assert len(loaded) == 5
        assert loaded[0].index == 0
        assert loaded[4].index == 4

    def test_append_mode(self, tmp_path):
        path = tmp_path / "completions.jsonl"
        # first writer
        w1 = CompletionWriter(path)
        w1.write(_make_record(batch=1))
        w1.close()
        # second writer appends
        w2 = CompletionWriter(path)
        w2.write(_make_record(batch=2))
        w2.close()

        loaded = load_completions(path)
        assert len(loaded) == 2
        assert loaded[0].batch == 1
        assert loaded[1].batch == 2

    def test_count_property(self, tmp_path):
        path = tmp_path / "completions.jsonl"
        writer = CompletionWriter(path)
        assert writer.count == 0
        writer.write(_make_record())
        assert writer.count == 1
        writer.write(_make_record())
        assert writer.count == 2
        writer.close()

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "completions.jsonl"
        writer = CompletionWriter(path)
        writer.write(_make_record())
        writer.close()
        assert path.exists()


class TestLoadCompletions:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_completions(path) == []

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "with_blanks.jsonl"
        rec = _make_record()
        line = json.dumps(_serialize_record(rec))
        path.write_text(f"{line}\n\n{line}\n")
        loaded = load_completions(path)
        assert len(loaded) == 2


class TestMakeTimestamp:
    def test_returns_iso8601(self):
        ts = make_timestamp()
        assert "T" in ts
        assert "+" in ts or "Z" in ts
