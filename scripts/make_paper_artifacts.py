#!/usr/bin/env python3
"""Generate reproducible figures/tables for the LaTeX paper from training run data.

Usage:
    pixi run -e dev python scripts/make_paper_artifacts.py --config configs/paper/runlist.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ppl_synthesis_reward_hacking.plotting.styles import apply_publication_style
from ppl_synthesis_reward_hacking.reporting.tables import (
    fmt_pm_std as _fmt_pm_std,
)
from ppl_synthesis_reward_hacking.reporting.tables import (
    write_baseline_vs_trained_table as _write_baseline_vs_trained_table,
)
from ppl_synthesis_reward_hacking.reporting.tables import (
    write_lh_evidence_table as _write_lh_evidence_table,
)
from ppl_synthesis_reward_hacking.reporting.tables import (
    write_normalization_table as _write_normalization_table,
)
from ppl_synthesis_reward_hacking.utils.git import git_info as _git_info_impl

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    sns = None  # type: ignore[assignment]
    _HAS_SEABORN = False

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_info() -> dict[str, Any]:
    return _git_info_impl(REPO_ROOT)


def _run_analyze_hacking(
    run_dir: Path, *, exclude_clipped: bool, winsorize: float | None
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/analyze_hacking.py",
        str(run_dir),
        "--json",
        "--no-save",
    ]
    if exclude_clipped:
        cmd.append("--exclude-clipped")
    if winsorize is not None:
        cmd += ["--winsorize", str(winsorize)]
    out = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    return json.loads(out)


def _run_analyze_emergence(
    *,
    training_path: Path,
    baseline_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/analyze_hacking_emergence.py",
        "--training",
        str(training_path),
        "--baseline",
        str(baseline_path),
        "--output",
        str(output_path),
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    summary_path = output_path.with_suffix(".json")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _extract_step_series(report: dict[str, Any], metric: str) -> dict[int, float]:
    out: dict[int, float] = {}
    per_step = report.get("per_step") or {}
    if not isinstance(per_step, dict):
        return out
    for step_str, sa in per_step.items():
        try:
            step = int(step_str)
        except (TypeError, ValueError):
            continue
        if not isinstance(sa, dict):
            continue
        v = sa.get(metric)
        if v is None:
            continue
        if not isinstance(v, int | float):
            continue
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            continue
        out[step] = fv
    return out


def _aggregate_series(reports: list[dict[str, Any]], metric: str) -> dict[str, list[float]]:
    step_to_vals: dict[int, list[float]] = defaultdict(list)
    for r in reports:
        series = _extract_step_series(r, metric)
        for step, v in series.items():
            step_to_vals[step].append(v)

    steps = sorted(step_to_vals.keys())
    means = [float(np.mean(step_to_vals[s])) for s in steps]
    stds = [float(np.std(step_to_vals[s])) for s in steps]
    ns = [len(step_to_vals[s]) for s in steps]
    return {"steps": steps, "mean": means, "std": stds, "n": ns}


def _setup_plot_style() -> None:
    """Configure seaborn + Times New Roman for publication figures."""
    apply_publication_style(
        font_size=9,
        axes_labelsize=9,
        axes_titlesize=10,
        xtick_labelsize=8,
        ytick_labelsize=8,
        legend_fontsize=8,
        lines_linewidth=1.5,
    )


def _make_reward_figure(*, has_normalization_panel: bool):
    import matplotlib.pyplot as plt

    if has_normalization_panel:
        fig, (ax_reward, ax_norm) = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            gridspec_kw={"height_ratios": [1, 0.7]},
        )
        return fig, ax_reward, ax_norm

    fig, ax_reward = plt.subplots(figsize=(9, 4.5))
    return fig, ax_reward, None


# Legacy helper names retained for local script tests.
def _make_gap_figure(*, has_normalization_panel: bool):
    return _make_reward_figure(has_normalization_panel=has_normalization_panel)


def _plot_reward_trajectories(
    ax_reward: Any,
    condition_summaries: list[dict[str, Any]],
    *,
    palette: list[Any] | None,
) -> None:
    for i, cond in enumerate(condition_summaries):
        series = cond["reward"]
        steps = series["steps"]
        if not steps:
            continue
        mean_ = series["mean"]
        std_ = series["std"]
        color = palette[i % len(palette)] if palette else None
        ax_reward.plot(steps, mean_, linewidth=2, label=cond["label"], color=color)
        ax_reward.fill_between(
            steps,
            [m - sd for m, sd in zip(mean_, std_, strict=False)],
            [m + sd for m, sd in zip(mean_, std_, strict=False)],
            alpha=0.12,
            color=color,
        )


def _plot_gap_trajectories(
    ax_gap: Any,
    condition_summaries: list[dict[str, Any]],
    *,
    palette: list[Any] | None,
) -> None:
    for i, cond in enumerate(condition_summaries):
        series = cond["gap"]
        steps = series["steps"]
        if not steps:
            continue
        mean_ = series["mean"]
        std_ = series["std"]
        color = palette[i % len(palette)] if palette else None
        ax_gap.plot(steps, mean_, linewidth=2, label=cond["label"], color=color)
        ax_gap.fill_between(
            steps,
            [m - sd for m, sd in zip(mean_, std_, strict=False)],
            [m + sd for m, sd in zip(mean_, std_, strict=False)],
            alpha=0.12,
            color=color,
        )


def _format_reward_axis(ax_reward: Any, *, show_xlabel: bool) -> None:
    ax_reward.set_title("Reward trajectory (train objective)")
    if show_xlabel:
        ax_reward.set_xlabel("Training step")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend(loc="best")
    if _HAS_SEABORN:
        sns.despine(ax=ax_reward, left=False, bottom=False)


def _format_gap_axis(ax_gap: Any, *, show_xlabel: bool) -> None:
    ax_gap.set_title("Reward gap trajectory (reported \u2212 oracle)")
    if show_xlabel:
        ax_gap.set_xlabel("Training step")
    ax_gap.set_ylabel("Gap")
    ax_gap.legend(loc="best")
    if _HAS_SEABORN:
        sns.despine(ax=ax_gap, left=False, bottom=False)


def _normalization_bars(
    condition_summaries: list[dict[str, Any]],
) -> tuple[list[str], list[float]]:
    bar_labels: list[str] = []
    bar_vals: list[float] = []
    for cond in condition_summaries:
        agg = cond.get("offline_eval_aggregate") or {}
        norm_data = agg.get("normalization_frac_non_normalized")
        if not isinstance(norm_data, dict):
            continue
        frac = norm_data.get("mean")
        if not isinstance(frac, int | float):
            continue
        frac_value = float(frac)
        if math.isnan(frac_value) or math.isinf(frac_value):
            continue
        bar_labels.append(str(cond["label"]))
        bar_vals.append(frac_value * 100.0)
    return bar_labels, bar_vals


def _plot_normalization_panel(
    ax_norm: Any,
    *,
    bar_labels: list[str],
    bar_vals: list[float],
) -> None:
    if bar_labels and _HAS_SEABORN:
        sns.barplot(x=bar_labels, y=bar_vals, ax=ax_norm, palette="muted")
    elif bar_labels:
        ax_norm.bar(bar_labels, bar_vals, alpha=0.8)
    ax_norm.set_ylabel("Non-normalized (%)")
    ax_norm.set_xlabel("Condition")
    if _HAS_SEABORN:
        sns.despine(ax=ax_norm, left=False, bottom=False)


def _maybe_plot_reward_trajectory(
    condition_summaries: list[dict[str, Any]],
    *,
    out_pdf: Path,
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    _setup_plot_style()
    bar_labels, bar_vals = _normalization_bars(condition_summaries)
    fig, ax_reward, ax_norm = _make_reward_figure(has_normalization_panel=bool(bar_labels))
    palette = sns.color_palette("muted") if _HAS_SEABORN else None
    _plot_reward_trajectories(ax_reward, condition_summaries, palette=palette)
    _format_reward_axis(ax_reward, show_xlabel=True)
    if ax_norm is not None:
        _plot_normalization_panel(ax_norm, bar_labels=bar_labels, bar_vals=bar_vals)

    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _maybe_plot_gap_trajectory(
    condition_summaries: list[dict[str, Any]],
    *,
    out_pdf: Path,
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    _setup_plot_style()
    fig, ax_gap = plt.subplots(figsize=(9, 4.5))
    palette = sns.color_palette("muted") if _HAS_SEABORN else None
    _plot_gap_trajectories(ax_gap, condition_summaries, palette=palette)
    _format_gap_axis(ax_gap, show_xlabel=True)

    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _maybe_plot_failure_rates(
    condition_summaries: list[dict[str, Any]],
    *,
    out_pdf: Path,
    out_png: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib not installed, skipping plots", file=sys.stderr)
        return

    _setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    palette = sns.color_palette("muted") if _HAS_SEABORN else None
    for i, cond in enumerate(condition_summaries):
        name = cond["label"]
        exec_s = cond["exec_fail_pct"]
        parse_s = cond["parse_fail_pct"]
        color = palette[i % len(palette)] if palette else None
        if exec_s["steps"]:
            ax1.plot(exec_s["steps"], exec_s["mean"], linewidth=2, label=name, color=color)
        if parse_s["steps"]:
            ax2.plot(parse_s["steps"], parse_s["mean"], linewidth=2, label=name, color=color)

    ax1.set_title("Exec failure rate (%)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("%")

    ax2.set_title("Parse failure rate (%)")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("%")

    ax1.legend(loc="best")
    if _HAS_SEABORN:
        sns.despine(ax=ax1, left=False, bottom=False)
        sns.despine(ax=ax2, left=False, bottom=False)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs))


def _collect_offline_eval_values(
    offline_eval_summaries: list[dict[str, Any]],
    *,
    section: str,
    key: str,
) -> list[float]:
    out: list[float] = []
    for run_summary in offline_eval_summaries:
        sec = run_summary.get(section)
        if not isinstance(sec, dict):
            continue
        value = sec.get(key)
        if not isinstance(value, int | float):
            continue
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            continue
        out.append(f)
    return out


def _summarize_offline_eval_runs(
    offline_eval_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    norm_frac = _collect_offline_eval_values(
        offline_eval_summaries,
        section="normalization",
        key="frac_non_normalized",
    )
    norm_log_mass = _collect_offline_eval_values(
        offline_eval_summaries,
        section="normalization",
        key="mean_log_mass",
    )
    taxonomy_tagged = _collect_offline_eval_values(
        offline_eval_summaries,
        section="taxonomy",
        key="tagged_fraction",
    )
    safety_accept = _collect_offline_eval_values(
        offline_eval_summaries,
        section="safety",
        key="acceptance_rate_checked",
    )
    norm_abs_log_mass = [abs(v) for v in norm_log_mass]
    norm_frac_m, norm_frac_s = _compute_mean_std(norm_frac)
    norm_log_mass_m, norm_log_mass_s = _compute_mean_std(norm_log_mass)
    norm_abs_log_mass_m, norm_abs_log_mass_s = _compute_mean_std(norm_abs_log_mass)
    taxonomy_m, taxonomy_s = _compute_mean_std(taxonomy_tagged)
    safety_m, safety_s = _compute_mean_std(safety_accept)
    return {
        "n_runs": len(offline_eval_summaries),
        "normalization_frac_non_normalized": {
            "mean": norm_frac_m,
            "std": norm_frac_s,
            "formatted": _fmt_pm_std(norm_frac_m * 100.0, norm_frac_s * 100.0),
        },
        "normalization_mean_log_mass": {
            "mean": norm_log_mass_m,
            "std": norm_log_mass_s,
            "formatted": _fmt_pm_std(norm_log_mass_m, norm_log_mass_s),
        },
        "normalization_mean_abs_log_mass": {
            "mean": norm_abs_log_mass_m,
            "std": norm_abs_log_mass_s,
            "formatted": _fmt_pm_std(norm_abs_log_mass_m, norm_abs_log_mass_s),
        },
        "taxonomy_tagged_fraction": {
            "mean": taxonomy_m,
            "std": taxonomy_s,
            "formatted": _fmt_pm_std(taxonomy_m * 100.0, taxonomy_s * 100.0),
        },
        "safety_acceptance_rate_checked": {
            "mean": safety_m,
            "std": safety_s,
            "formatted": _fmt_pm_std(safety_m * 100.0, safety_s * 100.0),
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper artifacts (figures/tables)")
    p.add_argument("--config", type=Path, default=Path("configs/paper/runlist.yaml"))
    p.add_argument("--out", type=Path, default=Path("paper_artifacts"))
    p.add_argument(
        "--offline-eval",
        action="store_true",
        help="Run offline LH evaluators (normalization/taxonomy/safety) per run",
    )
    p.add_argument(
        "--offline-eval-sample",
        type=int,
        default=200,
        help="Max completions sampled per run for offline evaluators",
    )
    p.add_argument(
        "--offline-eval-only-valid",
        action="store_true",
        help="Use only outcome=valid records for offline evaluators",
    )
    return p.parse_args()


def _resolve_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path


def _read_paper_track(run_dir: Path) -> str | None:
    cfg_path = run_dir / "hydra_resolved_config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    train = cfg.get("train")
    if not isinstance(train, dict):
        return None
    track = train.get("paper_track")
    return str(track) if isinstance(track, str) else None


def _read_claim_mode(run_dir: Path) -> str | None:
    cfg_path = run_dir / "hydra_resolved_config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    train = cfg.get("train")
    if not isinstance(train, dict):
        return None
    claim_mode = train.get("claim_mode")
    return str(claim_mode) if isinstance(claim_mode, str) else None


def _load_runlist_config(cfg_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("runlist config must be a mapping")
    return cfg


def _parse_analysis_config(cfg: dict[str, Any]) -> tuple[bool, float | None]:
    analysis_cfg = cfg.get("analysis") or {}
    exclude_clipped = bool(analysis_cfg.get("exclude_clipped", False))
    winsorize = analysis_cfg.get("winsorize", None)
    winsorize_f = float(winsorize) if winsorize is not None else None
    return exclude_clipped, winsorize_f


def _resolve_baseline_path(cfg: dict[str, Any]) -> Path | None:
    baseline_path_raw = cfg.get("baseline")
    if not isinstance(baseline_path_raw, str) or not baseline_path_raw.strip():
        return None
    candidate = (REPO_ROOT / baseline_path_raw).resolve()
    return candidate if candidate.exists() else None


def _validate_conditions(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = cfg.get("conditions") or []
    if not isinstance(conditions, list) or not conditions:
        raise RuntimeError("config must contain non-empty `conditions` list")
    return [cond for cond in conditions if isinstance(cond, dict)]


def _build_manifest(
    *,
    cfg_path: Path,
    exclude_clipped: bool,
    winsorize_f: float | None,
    baseline_path: Path | None,
    offline_eval: bool,
    offline_eval_sample: int | None,
    offline_eval_only_valid: bool,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "repo": str(REPO_ROOT),
        "git": _git_info(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "config_path": str(cfg_path),
        "analysis": {"exclude_clipped": exclude_clipped, "winsorize": winsorize_f},
        "offline_eval": {
            "enabled": offline_eval,
            "sample": offline_eval_sample,
            "only_valid": offline_eval_only_valid,
        },
        "baseline_path": str(baseline_path) if baseline_path else None,
        "conditions": [],
    }


def _run_offline_eval(
    *,
    completions_path: Path,
    eval_dir: Path,
    sample: int | None,
    only_valid: bool,
) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)

    def _run(
        script_name: str,
        summary_name: str,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        cmd = [
            sys.executable,
            f"scripts/{script_name}",
            "--completions",
            str(completions_path),
            "--output-dir",
            str(eval_dir),
            "--dedup",
        ]
        if only_valid:
            cmd.append("--only-valid")
        if sample is not None:
            cmd += ["--sample", str(sample)]
        if extra_args:
            cmd += extra_args
        subprocess.check_call(cmd, cwd=REPO_ROOT)
        summary_path = eval_dir / summary_name
        return json.loads(summary_path.read_text(encoding="utf-8"))

    normalization = _run(
        "eval_normalization.py",
        "normalization_summary.json",
    )
    taxonomy = _run(
        "eval_taxonomy.py",
        "taxonomy_summary.json",
    )
    safety = _run(
        "eval_safety_gate.py",
        "safety_summary.json",
    )
    return {
        "eval_dir": str(eval_dir),
        "normalization": normalization,
        "taxonomy": taxonomy,
        "safety": safety,
    }


def _process_condition(
    *,
    cond: dict[str, Any],
    out_dir: Path,
    baseline_path: Path | None,
    exclude_clipped: bool,
    winsorize_f: float | None,
    baseline_summary: dict[str, Any] | None,
    offline_eval_enabled: bool,
    offline_eval_sample: int | None,
    offline_eval_only_valid: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    cond_name = str(cond.get("name") or "condition")
    cond_label = str(cond.get("label") or cond_name)
    run_dirs = cond.get("runs") or []
    if not isinstance(run_dirs, list) or not run_dirs:
        raise RuntimeError(f"condition {cond_name} missing `runs` list")

    reports: list[dict[str, Any]] = []
    emergence_summaries: list[dict[str, Any]] = []
    offline_eval_summaries: list[dict[str, Any]] = []
    claim_modes_used: list[str] = []
    declared_runs: list[str] = []
    resolved_runs: list[str] = []
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for run_item in run_dirs:
        run_dir = (REPO_ROOT / str(run_item)).resolve()
        declared_runs.append(str(run_dir))
        completions_path = run_dir / "completions.jsonl"
        if not completions_path.exists():
            print(f"SKIP (no completions): {run_dir}", file=sys.stderr)
            continue
        paper_track = _read_paper_track(run_dir)
        if paper_track not in {"part_a_emergence", "part_b_mitigation"}:
            print(
                f"SKIP (non-paper track: {paper_track!r}): {run_dir}",
                file=sys.stderr,
            )
            continue
        claim_mode = _read_claim_mode(run_dir)
        if claim_mode != "formal_lh":
            print(
                f"SKIP (non-formal claim_mode: {claim_mode!r}): {run_dir}",
                file=sys.stderr,
            )
            continue
        resolved_runs.append(str(run_dir))
        claim_modes_used.append(claim_mode)
        run_artifact_id = _run_artifact_id(run_dir)
        report = _run_analyze_hacking(
            run_dir, exclude_clipped=exclude_clipped, winsorize=winsorize_f
        )
        reports.append(report)
        (analysis_dir / f"{cond_name}_{run_artifact_id}_hacking_analysis.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        _append_emergence_summary(
            emergence_summaries=emergence_summaries,
            out_dir=out_dir,
            cond_name=cond_name,
            run_artifact_id=run_artifact_id,
            run_dir=run_dir,
            completions_path=completions_path,
            baseline_path=baseline_path,
        )
        if (
            baseline_summary is None
            and emergence_summaries
            and "baseline" in emergence_summaries[-1]
        ):
            baseline_summary = emergence_summaries[-1]["baseline"]
        if offline_eval_enabled:
            eval_dir = out_dir / "eval" / cond_name / run_artifact_id
            eval_summary = _run_offline_eval(
                completions_path=completions_path,
                eval_dir=eval_dir,
                sample=offline_eval_sample,
                only_valid=offline_eval_only_valid,
            )
            eval_summary["run_dir"] = str(run_dir)
            offline_eval_summaries.append(eval_summary)

    if not reports:
        return None, baseline_summary, None

    condition_summary = _build_condition_summary(cond_name, cond_label, resolved_runs, reports)
    condition_row = _build_condition_row(
        cond_label,
        reports,
        emergence_summaries,
        offline_eval_summaries,
    )
    offline_eval_aggregate = _summarize_offline_eval_runs(offline_eval_summaries)
    manifest_entry = {
        "name": cond_name,
        "label": cond_label,
        "runs_declared": declared_runs,
        "runs": resolved_runs,
        "n_runs_declared": len(declared_runs),
        "n_runs_used": len(reports),
        "claim_modes_used": claim_modes_used,
        "offline_eval_runs": offline_eval_summaries,
        "offline_eval_aggregate": offline_eval_aggregate,
    }
    return (
        {
            **condition_summary,
            "manifest_entry": manifest_entry,
            "row": condition_row,
            "offline_eval_runs": offline_eval_summaries,
            "offline_eval_aggregate": offline_eval_aggregate,
        },
        baseline_summary,
        manifest_entry,
    )


def _append_emergence_summary(
    *,
    emergence_summaries: list[dict[str, Any]],
    out_dir: Path,
    cond_name: str,
    run_artifact_id: str,
    run_dir: Path,
    completions_path: Path,
    baseline_path: Path | None,
) -> None:
    if baseline_path is None:
        return
    emergence_dir = out_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    out_path = emergence_dir / f"{cond_name}_{run_artifact_id}_emergence.png"
    summary = _run_analyze_emergence(
        training_path=completions_path,
        baseline_path=baseline_path,
        output_path=out_path,
    )
    emergence_summaries.append(summary)


def _run_artifact_id(run_dir: Path) -> str:
    """Stable, collision-resistant run identifier for artifact paths."""
    digest = hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:10]
    return f"{run_dir.name}_{digest}"


def _build_condition_summary(
    cond_name: str,
    cond_label: str,
    resolved_runs: list[str],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": cond_name,
        "label": cond_label,
        "runs": resolved_runs,
        "n_runs_used": len(reports),
        "reward": _aggregate_series(reports, "reported_mean"),
        "gap": _aggregate_series(reports, "gap_mean"),
        "exec_fail_pct": _aggregate_series(reports, "exec_fail_pct"),
        "parse_fail_pct": _aggregate_series(reports, "parse_fail_pct"),
    }


def _build_condition_row(
    cond_label: str,
    reports: list[dict[str, Any]],
    emergence_summaries: list[dict[str, Any]],
    offline_eval_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if emergence_summaries:
        row = _build_emergence_condition_row(cond_label, emergence_summaries)
    else:
        row = _build_gap_only_condition_row(cond_label, reports)

    offline_summary = _summarize_offline_eval_runs(offline_eval_summaries)
    row["normalization_frac_non_normalized_pm_std"] = offline_summary[
        "normalization_frac_non_normalized"
    ]["formatted"]
    row["normalization_mean_log_mass_pm_std"] = offline_summary["normalization_mean_log_mass"][
        "formatted"
    ]
    row["normalization_mean_abs_log_mass_pm_std"] = offline_summary[
        "normalization_mean_abs_log_mass"
    ]["formatted"]
    row["taxonomy_tagged_pm_std"] = offline_summary["taxonomy_tagged_fraction"]["formatted"]
    row["safety_acceptance_pm_std"] = offline_summary["safety_acceptance_rate_checked"]["formatted"]
    return row


def _build_emergence_condition_row(
    cond_label: str,
    emergence_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    final_gaps = [float(s["final"]["gap"]) for s in emergence_summaries if "final" in s]
    final_any_exploit = [
        float(s["final"]["any_exploit_rate"]) for s in emergence_summaries if "final" in s
    ]
    final_gap_valid = [
        float(s["final"]["gap_valid_rate"]) for s in emergence_summaries if "final" in s
    ]
    g_m, g_s = _compute_mean_std(final_gaps)
    e_m, e_s = _compute_mean_std(final_any_exploit)
    v_m, v_s = _compute_mean_std(final_gap_valid)
    return {
        "label": cond_label,
        "gap_pm_std": _fmt_pm_std(g_m, g_s),
        "any_exploit_pm_std": _fmt_pm_std(e_m * 100.0, e_s * 100.0),
        "gap_valid_pm_std": _fmt_pm_std(v_m * 100.0, v_s * 100.0),
    }


def _build_gap_only_condition_row(cond_label: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    final_gaps: list[float] = []
    for report in reports:
        traj = report.get("gap_trajectory") or {}
        last = traj.get("last_step_mean")
        if isinstance(last, int | float) and not (
            math.isnan(float(last)) or math.isinf(float(last))
        ):
            final_gaps.append(float(last))
    g_m, g_s = _compute_mean_std(final_gaps)
    return {
        "label": cond_label,
        "gap_pm_std": _fmt_pm_std(g_m, g_s),
        "any_exploit_pm_std": "n/a",
        "gap_valid_pm_std": "n/a",
    }


def _write_safety_matrix(out_dir: Path) -> None:
    subprocess.check_call(
        [
            sys.executable,
            "scripts/make_safety_gate_matrix.py",
            "--out",
            str(out_dir / "safety_gate_matrix_table.tex"),
            "--json",
            str(out_dir / "safety_gate_matrix.json"),
        ],
        cwd=REPO_ROOT,
    )


def _write_outputs(
    *,
    out_dir: Path,
    condition_summaries: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
    condition_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    # primary table: normalization-first LH evidence
    _write_lh_evidence_table(
        out_dir / "lh_evidence_table.tex",
        condition_rows=condition_rows,
    )
    _write_normalization_table(
        out_dir / "normalization_table.tex",
        condition_rows=condition_rows,
    )
    _maybe_plot_reward_trajectory(
        condition_summaries,
        out_pdf=out_dir / "reward_trajectory.pdf",
        out_png=out_dir / "reward_trajectory.png",
    )
    _maybe_plot_gap_trajectory(
        condition_summaries,
        out_pdf=out_dir / "gap_trajectory.pdf",
        out_png=out_dir / "gap_trajectory.png",
    )
    _maybe_plot_failure_rates(
        condition_summaries,
        out_pdf=out_dir / "failure_rates.pdf",
        out_png=out_dir / "failure_rates.png",
    )
    # secondary: legacy gap-only table
    _write_baseline_vs_trained_table(
        out_dir / "baseline_vs_trained_table.tex",
        baseline=baseline_summary,
        condition_rows=condition_rows,
    )
    _write_safety_matrix(out_dir)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    cfg_path = _resolve_path(args.config)
    out_dir = _resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_runlist_config(cfg_path)
    exclude_clipped, winsorize_f = _parse_analysis_config(cfg)
    baseline_path = _resolve_baseline_path(cfg)
    conditions = _validate_conditions(cfg)
    manifest = _build_manifest(
        cfg_path=cfg_path,
        exclude_clipped=exclude_clipped,
        winsorize_f=winsorize_f,
        baseline_path=baseline_path,
        offline_eval=args.offline_eval,
        offline_eval_sample=args.offline_eval_sample,
        offline_eval_only_valid=args.offline_eval_only_valid,
    )
    condition_summaries: list[dict[str, Any]] = []
    baseline_summary: dict[str, Any] | None = None
    condition_rows: list[dict[str, Any]] = []

    for cond in conditions:
        condition_result, baseline_summary, manifest_entry = _process_condition(
            cond=cond,
            out_dir=out_dir,
            baseline_path=baseline_path,
            exclude_clipped=exclude_clipped,
            winsorize_f=winsorize_f,
            baseline_summary=baseline_summary,
            offline_eval_enabled=args.offline_eval,
            offline_eval_sample=args.offline_eval_sample,
            offline_eval_only_valid=args.offline_eval_only_valid,
        )
        if condition_result is None:
            continue
        condition_summaries.append(
            {
                "name": condition_result["name"],
                "label": condition_result["label"],
                "runs": condition_result["runs"],
                "n_runs_used": condition_result["n_runs_used"],
                "reward": condition_result["reward"],
                "gap": condition_result["gap"],
                "exec_fail_pct": condition_result["exec_fail_pct"],
                "parse_fail_pct": condition_result["parse_fail_pct"],
                "offline_eval_aggregate": condition_result.get("offline_eval_aggregate"),
            }
        )
        condition_rows.append(condition_result["row"])
        if manifest_entry is not None:
            manifest["conditions"].append(manifest_entry)

    _write_outputs(
        out_dir=out_dir,
        condition_summaries=condition_summaries,
        baseline_summary=baseline_summary,
        condition_rows=condition_rows,
        manifest=manifest,
    )
    print(f"Wrote paper artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
