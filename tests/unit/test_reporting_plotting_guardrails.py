from __future__ import annotations

from pathlib import Path

from ppl_synthesis_reward_hacking.plotting.grpo_figures import plot_lh_taxonomy_bar
from ppl_synthesis_reward_hacking.reporting.code_examples import select_example, write_code_comparison


def _record(
    *,
    batch: int,
    code: str,
    log_mass: float | None,
) -> dict:
    metadata = {}
    if log_mass is not None:
        metadata["normalization"] = {"log_mass": log_mass}
    return {
        "batch": batch,
        "outcome": "valid",
        "code": code,
        "metadata": metadata,
    }


def _classify(rec: dict) -> str:
    code = rec.get("code", "")
    if "HONEST" in code:
        return "honest"
    return "lh_pure_score_injection_baseline"


def test_select_example_requires_log_mass() -> None:
    records = [
        _record(batch=5, code="HONEST", log_mass=None),
        _record(batch=950, code="HACK", log_mass=None),
    ]

    honest = select_example(records, early=True, honest=True, classify_fn=_classify)
    hacked = select_example(records, early=False, honest=False, classify_fn=_classify)

    assert honest is None
    assert hacked is None


def test_write_code_comparison_uses_log_mass_only(tmp_path: Path) -> None:
    out = tmp_path / "code_comparison.tex"
    records = [
        _record(batch=5, code="HONEST", log_mass=0.01),
        _record(batch=950, code="HACK", log_mass=1.75),
    ]
    family_labels = {
        "lh_pure_score_injection_baseline": "Score injection",
        "honest": "Honest",
    }

    write_code_comparison(records, out, classify_fn=_classify, family_labels=family_labels)

    content = out.read_text(encoding="utf-8")
    assert r"\log m =" in content
    assert "gap~$=$" not in content


def test_plot_lh_taxonomy_bar_skips_without_log_mass(tmp_path: Path) -> None:
    out = tmp_path / "lh_taxonomy_bar.pdf"
    records = [
        _record(batch=900, code="HACK", log_mass=None),
        _record(batch=901, code="HONEST", log_mass=None),
    ]

    plot_lh_taxonomy_bar(records, out, classify_fn=_classify)

    assert not out.exists()
    assert not out.with_suffix(".png").exists()


def test_plot_lh_taxonomy_bar_writes_when_log_mass_present(tmp_path: Path) -> None:
    out = tmp_path / "lh_taxonomy_bar.pdf"
    records = [
        _record(batch=900, code="HACK", log_mass=1.2),
        _record(batch=901, code="HONEST", log_mass=0.0),
        _record(batch=902, code="HACK", log_mass=1.5),
    ]

    plot_lh_taxonomy_bar(records, out, classify_fn=_classify)

    assert out.exists()
    assert out.with_suffix(".png").exists()
