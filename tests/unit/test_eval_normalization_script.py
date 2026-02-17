from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "eval_normalization.py"
    spec = importlib.util.spec_from_file_location("eval_normalization", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_summary_includes_median_log_mass() -> None:
    module = _load_module()
    rows = [
        {"batch": 0, "ok": True, "log_mass": 0.1},
        {"batch": 0, "ok": True, "log_mass": 0.3},
        {"batch": 0, "ok": False, "log_mass": None},
    ]
    out = module._batch_summary(rows, epsilon=1e-3)
    assert len(out) == 1
    assert out[0]["mean_log_mass"] == pytest.approx(0.2)
    assert out[0]["median_log_mass"] == pytest.approx(0.2)


def test_batch_summary_handles_multiple_batches_and_all_failed_batch() -> None:
    module = _load_module()
    rows = [
        {"batch": 0, "ok": False, "log_mass": None},
        {"batch": 0, "ok": False, "log_mass": None},
        {"batch": 1, "ok": True, "log_mass": 0.005},
        {"batch": 1, "ok": True, "log_mass": 0.001},
        {"batch": 1, "ok": False, "log_mass": None},
    ]
    out = module._batch_summary(rows, epsilon=1e-3)
    assert len(out) == 2
    batch0 = out[0]
    batch1 = out[1]
    assert batch0["batch"] == 0
    assert batch0["n_ok"] == 0
    assert batch0["frac_non_normalized"] == 0.0
    assert batch0["mean_log_mass"] is None
    assert batch0["median_log_mass"] is None

    assert batch1["batch"] == 1
    assert batch1["n_ok"] == 2
    assert batch1["frac_non_normalized"] == pytest.approx(0.5)
    assert batch1["mean_log_mass"] == pytest.approx(0.003)
    assert batch1["median_log_mass"] == pytest.approx(0.003)


def test_evaluate_normalization_rejects_large_d_in_auto_mode() -> None:
    module = _load_module()
    args = argparse.Namespace(
        mode="auto",
        d=12,
        max_d_exact=10,
        p_true=0.5,
        timeout=1,
        epsilon=1e-3,
    )
    with pytest.raises(RuntimeError, match="exceeds --max-d-exact"):
        module._evaluate_normalization(
            program_text="```python\ndef model(data):\n    return None\n```",
            args=args,
            row_seed=123,
        )


def test_json_float_handles_numpy_scalars() -> None:
    module = _load_module()
    assert module._json_float(np.float32(1.25)) == pytest.approx(1.25)
    assert module._json_float(np.int64(7)) == 7
    assert module._json_float(np.float64(float("nan"))) is None
    assert module._json_float(float("inf")) is None
    assert module._json_float(float("-inf")) is None


def test_record_hash_dedups_on_extracted_code_not_completion_prose() -> None:
    module = _load_module()
    code = "def model(data):\n    return None\n"
    record_a = CompletionRecord(
        batch=0,
        index=0,
        prompt="p",
        completion_text="Answer A\n```python\ndef model(data):\n    return None\n```\nThanks.",
        code=code,
        reported_reward=0.0,
        oracle_score=0.0,
        gap=0.0,
        outcome="valid",
        timestamp="2026-01-01T00:00:00Z",
        metadata=None,
    )
    record_b = CompletionRecord(
        batch=0,
        index=1,
        prompt="p",
        completion_text=(
            "Different wrapper text\n"
            "```python\ndef model(data):\n    return None\n```\n"
            "Done."
        ),
        code=code,
        reported_reward=0.0,
        oracle_score=0.0,
        gap=0.0,
        outcome="valid",
        timestamp="2026-01-01T00:00:00Z",
        metadata=None,
    )

    assert module._record_hash(record_a) == module._record_hash(record_b)


def test_record_hash_extracts_code_when_record_code_missing() -> None:
    module = _load_module()
    completion_a = "A\n```python\ndef model(data):\n    return None\n```\n"
    completion_b = "B\n```python\ndef model(data):\n    return None\n```\n"
    record_a = CompletionRecord(
        batch=0,
        index=0,
        prompt="p",
        completion_text=completion_a,
        code=None,
        reported_reward=0.0,
        oracle_score=0.0,
        gap=0.0,
        outcome="parse_fail",
        timestamp="2026-01-01T00:00:00Z",
        metadata=None,
    )
    record_b = CompletionRecord(
        batch=0,
        index=1,
        prompt="p",
        completion_text=completion_b,
        code=None,
        reported_reward=0.0,
        oracle_score=0.0,
        gap=0.0,
        outcome="parse_fail",
        timestamp="2026-01-01T00:00:00Z",
        metadata=None,
    )
    assert module._record_hash(record_a) == module._record_hash(record_b)


def test_program_text_uses_extracted_code_not_raw_completion_text() -> None:
    module = _load_module()
    record = CompletionRecord(
        batch=0,
        index=0,
        prompt="p",
        completion_text=(
            "Narrative preamble\n"
            "```python\ndef model(data):\n    return None\n```\n"
            "Narrative epilogue"
        ),
        code="def model(data):\n    return None\n",
        reported_reward=0.0,
        oracle_score=0.0,
        gap=0.0,
        outcome="valid",
        timestamp="2026-01-01T00:00:00Z",
        metadata=None,
    )

    assert module._program_text(record) == "```python\ndef model(data):\n    return None\n```"


def test_program_text_extracts_from_completion_when_record_code_missing() -> None:
    module = _load_module()
    record = CompletionRecord(
        batch=0,
        index=0,
        prompt="p",
        completion_text=(
            "Some wrapper\n"
            "```python\ndef model(data):\n    return None\n```\n"
            "More wrapper"
        ),
        code=None,
        reported_reward=0.0,
        oracle_score=0.0,
        gap=0.0,
        outcome="parse_fail",
        timestamp="2026-01-01T00:00:00Z",
        metadata=None,
    )
    assert module._program_text(record) == "```python\ndef model(data):\n    return None\n```"
