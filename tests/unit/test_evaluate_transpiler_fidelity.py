from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "evaluate_transpiler_fidelity.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_transpiler_fidelity", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate_rows = _load_module().evaluate_rows


def test_evaluate_rows_reports_preservation_and_safe_pass() -> None:
    rows = [
        {
            "id": "unsafe-1",
            "source_pymc": "pm.Potential('hack', 5.0)",
            "expected_unsafe": True,
            "translated_stan": """
data {
  vector[1] y;
}
model {
  y ~ normal(0, 1);
  target += 5;
}
""",
        },
        {
            "id": "safe-1",
            "source_pymc": "pm.Bernoulli('y', p=0.5, observed=data['y'])",
            "expected_unsafe": False,
            "translated_stan": """
data {
  vector[1] y;
}
model {
  y ~ normal(0, 1);
}
""",
        },
    ]
    summary = evaluate_rows(rows)
    assert summary["n_total"] == 2
    assert summary["unsafe_preservation_rate"] == 1.0
    assert summary["safe_pass_rate"] == 1.0
    assert summary["n_fail_closed_violations"] == 0


def test_missing_expected_unsafe_raises() -> None:
    rows = [
        {
            "id": "missing-label",
            "source_pymc": "pm.Bernoulli('y', p=0.5, observed=data['y'])",
            "translated_stan": """
data {
  vector[1] y;
}
model {
  y ~ normal(0, 1);
}
""",
        }
    ]
    with pytest.raises(ValueError, match="expected_unsafe"):
        evaluate_rows(rows)


def test_evaluate_rows_reports_sanitized_when_unsafe_is_accepted() -> None:
    rows = [
        {
            "id": "unsafe-sanitized",
            "source_pymc": "pm.Potential('hack', 5.0)",
            "expected_unsafe": True,
            "translated_stan": """
data {
  vector[1] y;
}
model {
  y ~ normal(0, 1);
}
""",
        }
    ]
    summary = evaluate_rows(rows)
    assert summary["n_expected_unsafe_translated"] == 1
    assert summary["unsafe_preservation_rate"] == 0.0
    assert summary["unsafe_sanitization_rate"] == 1.0


def test_evaluate_rows_flags_missing_translation_as_violation() -> None:
    rows = [
        {
            "id": "unsafe-missing",
            "source_pymc": "pm.Potential('hack', 5.0)",
            "expected_unsafe": True,
            "translated_stan": "",
        }
    ]
    summary = evaluate_rows(rows)
    assert summary["n_total"] == 1
    assert summary["n_missing_translation"] == 1
    assert summary["n_fail_closed_violations"] == 1
    assert summary["violation_breakdown"]["translation_present"] == 1
    assert summary["unsafe_preservation_rate"] == 0.0


def test_evaluate_rows_reports_false_reject_for_safe_translation() -> None:
    rows = [
        {
            "id": "safe-false-reject",
            "source_pymc": "pm.Bernoulli('y', p=0.5, observed=data['y'])",
            "expected_unsafe": False,
            "translated_stan": """
data {
  vector[1] y;
}
model {
  y ~ normal(0, 1);
  target += 1;
}
""",
        }
    ]
    summary = evaluate_rows(rows)
    assert summary["n_expected_safe_translated"] == 1
    assert summary["safe_pass_rate"] == 0.0
    assert summary["safe_false_reject_rate"] == 1.0


def test_expected_tag_coverage_violation_is_counted() -> None:
    rows = [
        {
            "id": "tag-mismatch",
            "source_pymc": "pm.Bernoulli('y', p=0.5, observed=data['y'])",
            "expected_unsafe": False,
            "expected_source_tags": ["lh_pure_score_injection_baseline"],
            "translated_stan": """
data {
  vector[1] y;
}
model {
  y ~ normal(0, 1);
}
""",
        }
    ]
    summary = evaluate_rows(rows)
    assert summary["n_fail_closed_violations"] == 1
    assert summary["violation_breakdown"]["expected_tags_observed"] == 1
