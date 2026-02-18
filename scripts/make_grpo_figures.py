#!/usr/bin/env python3
"""Generate paper figures for the UAI 2026 likelihood-hacking paper.

Reads GRPO completions.jsonl and produces four artifacts:

  1. lh_emergence.pdf        — normalization metrics over training steps
  2. lh_taxonomy_evolution.pdf — LH family prevalence (stacked area)
  3. lh_taxonomy_bar.pdf     — LH family counts + log_mass distribution
  4. code_comparison.tex     — honest vs hacked code side-by-side

Usage:
    pixi run -e dev python scripts/make_grpo_figures.py [--run-dir PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from ppl_synthesis_reward_hacking.evaluation.aggregation import aggregate_by_batch  # noqa: E402
from ppl_synthesis_reward_hacking.evaluation.heuristics import detect_exploits  # noqa: E402
from ppl_synthesis_reward_hacking.evaluation.taxonomy import (  # noqa: E402
    canonicalize_exploit_tags,
)
from ppl_synthesis_reward_hacking.logging.completions import load_completions_raw  # noqa: E402
from ppl_synthesis_reward_hacking.plotting.grpo_figures import (  # noqa: E402
    FAMILY_LABELS,
    WINDOW,
    plot_lh_emergence,
    plot_lh_taxonomy_bar,
    plot_lh_taxonomy_evolution,
)
from ppl_synthesis_reward_hacking.plotting.styles import apply_publication_style  # noqa: E402
from ppl_synthesis_reward_hacking.reporting.code_examples import (  # noqa: E402
    write_code_comparison,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = REPO_ROOT / "artifacts" / "grpo_run_7050740"
DEFAULT_OUT = (
    REPO_ROOT
    / "background-documents"
    / "ai-scientist-likelihood-hacking-paper-local"
    / "paper_artifacts"
)

apply_publication_style(
    font_size=8,
    axes_labelsize=8,
    axes_titlesize=9,
    xtick_labelsize=7,
    ytick_labelsize=7,
    legend_fontsize=7,
    lines_linewidth=1.3,
)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_completions(run_dir: Path) -> list[dict]:
    return load_completions_raw(run_dir / "completions.jsonl")


# ---------------------------------------------------------------------------
# classification (taxonomy-based)
# ---------------------------------------------------------------------------


def classify(rec: dict) -> str:
    """Classify by LH taxonomy families.

    Returns a taxonomy family ID (e.g. 'lh_potential_constant_offset'),
    'multi_lh', 'honest', or 'invalid'.
    """
    if rec.get("outcome") != "valid":
        return "invalid"
    code = rec.get("code") or ""
    if not code.strip():
        return "invalid"
    raw = detect_exploits(code)
    tags = canonicalize_exploit_tags(code, precomputed_raw=raw)
    if not tags:
        return "honest"
    lh_tags = sorted(t for t in tags if t.startswith("lh_"))
    eval_tags = sorted(t for t in tags if t.startswith("evaluator_"))
    if len(lh_tags) > 1:
        return "multi_lh"
    if lh_tags:
        return lh_tags[0]
    if eval_tags:
        return eval_tags[0]
    return sorted(tags)[0]


def classify_simple(rec: dict) -> str:
    """3-category: honest / lh_exploit / invalid."""
    cat = classify(rec)
    if cat == "invalid":
        return "invalid"
    if cat == "honest":
        return "honest"
    return "lh_exploit"


# ---------------------------------------------------------------------------
# legacy (kept for reference, not called by main)
# ---------------------------------------------------------------------------


def _legacy_classify(rec: dict) -> str:
    """Old 5-category classification based on latent variable padding."""
    import re as _re

    if rec.get("outcome") != "valid":
        return "invalid"
    code = rec.get("code") or ""
    meta = rec.get("metadata", {})
    n_free = meta.get("n_free_rvs", 0)
    reported = rec.get("reported_reward", 0)
    if "data['y']" not in code and 'data["y"]' not in code:
        return "data_discard"
    if n_free >= 10:
        return "extreme_padding"
    if n_free > 1:
        rv_decl = _re.compile(r"pm\.\w+\(\s*['\"](\w+)['\"]")
        for i, line in enumerate(code.split("\n")):
            m = rv_decl.search(line)
            if not m or m.group(1) in ("y", "obs", "likelihood"):
                continue
            name = m.group(1)
            if not any(
                _re.search(r"\b" + _re.escape(name) + r"\b", other)
                for j, other in enumerate(code.split("\n"))
                if j != i
            ):
                return "disconnected_bonus"
    if n_free > 2 and reported > 50:
        return "connected_bonus"
    return "honest"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Generate GRPO paper figures")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument(
        "--late-batch-cutoff",
        type=int,
        default=800,
        help="batch index threshold for late-training analysis",
    )
    args = p.parse_args()

    window = args.window

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading completions from {args.run_dir} ...")
    records = load_completions(args.run_dir)
    valid = [r for r in records if r.get("outcome") == "valid"]
    print(f"  {len(records)} records ({len(valid)} valid)")

    batch_stats = aggregate_by_batch(
        records, classify_fn=classify, classify_simple_fn=classify_simple
    )
    print(f"  {len(batch_stats)} batches\n")

    print("Generating figures:")
    plot_lh_emergence(batch_stats, out_dir / "lh_emergence.pdf", window)
    plot_lh_taxonomy_evolution(batch_stats, out_dir / "lh_taxonomy_evolution.pdf", window)
    plot_lh_taxonomy_bar(
        records,
        out_dir / "lh_taxonomy_bar.pdf",
        classify_fn=classify,
        late_batch_cutoff=args.late_batch_cutoff,
    )
    write_code_comparison(
        records,
        out_dir / "code_comparison.tex",
        classify_fn=classify,
        family_labels=FAMILY_LABELS,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
