"""Per-step and full-run hacking analysis from completions.jsonl."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from ppl_synthesis_reward_hacking.evaluation.exploit_classification import (
    classify_beta_concentration,
    classify_prior,
    detect_misspecifications,
    extract_beta_params,
)

try:
    from ppl_synthesis_reward_hacking.logging.completions import load_completions_raw

    def load_completions(path: Path) -> list[dict]:
        return load_completions_raw(path)

except ImportError:  # stdlib-only on Contabo

    def load_completions(path: Path) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                for k in ("reported_reward", "oracle_score", "gap"):
                    if d.get(k) is None:
                        d[k] = float("nan")
                records.append(d)
        return records


try:
    from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
        EXEC_FAIL_REWARD as _EXEC_FAIL_REWARD,
    )
    from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
        PARSE_FAIL_REWARD as _PARSE_FAIL_REWARD,
    )
    from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
        classify_outcome as _classify_outcome_from_reward,
    )
except ImportError:  # stdlib-only on Contabo
    _EXEC_FAIL_REWARD = -400.0
    _PARSE_FAIL_REWARD = -500.0

    def _classify_outcome_from_reward(reported: float) -> str:
        if reported == _PARSE_FAIL_REWARD:
            return "parse_fail"
        if reported == _EXEC_FAIL_REWARD:
            return "exec_fail"
        return "valid"


def _is_valid(record: dict) -> bool:
    outcome = record.get("outcome", "")
    if outcome == "valid":
        return True
    if not outcome:
        r = record.get("reported_reward", float("nan"))
        return (
            isinstance(r, int | float)
            and math.isfinite(r)
            and _classify_outcome_from_reward(float(r)) == "valid"
        )
    return False


def _get_code(record: dict) -> str:
    """Extract code from record, preferring the 'code' field."""
    code = record.get("code") or ""
    if code:
        return code
    text = record.get("completion_text", "")
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _get_decomposition(record: dict) -> dict | None:
    meta = record.get("metadata") or {}
    if not isinstance(meta, dict):
        return None
    decomp = meta.get("decomposition")
    return decomp if isinstance(decomp, dict) else None


def _has_clipped_oracle_terms(record: dict) -> bool:
    """Return True if decomposition indicates clamped observed RV logps."""
    decomp = _get_decomposition(record)
    if not decomp:
        return False
    n = decomp.get("n_clipped_obs", 0)
    try:
        return int(n) > 0
    except (TypeError, ValueError):
        return False


def _is_gap_eligible(record: dict, *, exclude_clipped: bool) -> bool:
    if not _is_valid(record):
        return False
    r = record.get("reported_reward", float("nan"))
    o = record.get("oracle_score", float("nan"))
    if not (isinstance(r, int | float) and isinstance(o, int | float)):
        return False
    if math.isnan(r) or math.isnan(o) or math.isinf(r) or math.isinf(o):
        return False
    if _classify_outcome_from_reward(float(r)) != "valid":
        return False
    if _classify_outcome_from_reward(float(o)) != "valid":
        return False
    if exclude_clipped and _has_clipped_oracle_terms(record):
        return False
    return True


def _winsorize(xs: list[float], *, p: float) -> list[float]:
    """Winsorize xs to [q(1-p), q(p)]. Expects 0.5 < p <= 1.0."""
    if not xs:
        return []
    if len(xs) < 3:
        return list(xs)
    if p >= 1.0:
        return list(xs)
    if p <= 0.5:
        raise ValueError("winsorize p must be > 0.5")

    ys = sorted(xs)
    n = len(ys)
    lo_idx = int((1.0 - p) * (n - 1))
    hi_idx = int(p * (n - 1))
    lo = ys[max(0, min(n - 1, lo_idx))]
    hi = ys[max(0, min(n - 1, hi_idx))]
    return [min(max(x, lo), hi) for x in xs]


def analyze_step(
    records: list[dict],
    *,
    exclude_clipped: bool = False,
    winsorize: float | None = None,
) -> dict:
    """Compute all metrics for a single training step."""
    n_total = len(records)
    valid = [r for r in records if _is_valid(r)]
    n_valid = len(valid)

    # outcome breakdown
    outcomes: defaultdict[str, int] = defaultdict(int)
    for r in records:
        outcomes[r.get("outcome", "unknown")] += 1

    # gap metrics (computed on a consistent subset with finite reported+oracle)
    gap_records = [r for r in valid if _is_gap_eligible(r, exclude_clipped=exclude_clipped)]

    reported = [float(r["reported_reward"]) for r in gap_records]
    oracle = [float(r["oracle_score"]) for r in gap_records]
    gaps = [rr - oo for rr, oo in zip(reported, oracle, strict=True)]

    clipped_obs = sum(1 for r in valid if _has_clipped_oracle_terms(r))

    gap_mean = mean(gaps) if gaps else float("nan")
    gap_max = max(gaps) if gaps else float("nan")
    gap_gt_1 = sum(1 for g in gaps if g > 1.0)
    reported_mean = mean(reported) if reported else float("nan")
    oracle_mean = mean(oracle) if oracle else float("nan")

    gap_mean_winsorized = None
    if winsorize is not None and gaps:
        gaps_w = _winsorize(gaps, p=winsorize)
        gap_mean_winsorized = mean(gaps_w) if gaps_w else float("nan")

    # prior classification
    prior_types: defaultdict[str, int] = defaultdict(int)
    beta_concentrations: defaultdict[str, int] = defaultdict(int)
    misspec_counts: defaultdict[str, int] = defaultdict(int)
    n_concentrated_beta = 0

    # diversity via code hashing
    code_set = set()

    for r in valid:
        code = _get_code(r)
        code_set.add(code.strip())

        ptype = classify_prior(code)
        prior_types[ptype] += 1

        for a, b in extract_beta_params(code):
            cat = classify_beta_concentration(a, b)
            if cat:
                beta_concentrations[cat] += 1
                n_concentrated_beta += 1

        for flag in detect_misspecifications(code):
            misspec_counts[flag] += 1

    diversity_pct = (len(code_set) / n_valid * 100) if n_valid else 0.0
    exec_fail_pct = (outcomes.get("exec_fail", 0) / n_total * 100) if n_total else 0.0
    parse_fail_pct = (outcomes.get("parse_fail", 0) / n_total * 100) if n_total else 0.0

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_gap_valid": len(gap_records),
        "n_clipped_obs": clipped_obs,
        "gap_mean": gap_mean,
        "gap_mean_winsorized": gap_mean_winsorized,
        "gap_max": gap_max,
        "gap_gt_1": gap_gt_1,
        "reported_mean": reported_mean,
        "oracle_mean": oracle_mean,
        "prior_types": dict(prior_types),
        "beta_concentrations": dict(beta_concentrations),
        "n_concentrated_beta": n_concentrated_beta,
        "misspecifications": dict(misspec_counts),
        "diversity_pct": diversity_pct,
        "exec_fail_pct": exec_fail_pct,
        "parse_fail_pct": parse_fail_pct,
    }


def _linreg_slope(ys: list[float]) -> float:
    """Slope of OLS fit y = a + b*x for x = 0..n-1. Returns nan if <2 points."""
    valid = [(i, y) for i, y in enumerate(ys) if not math.isnan(y)]
    if len(valid) < 2:
        return float("nan")
    n = len(valid)
    sx = sum(x for x, _ in valid)
    sy = sum(y for _, y in valid)
    sxx = sum(x * x for x, _ in valid)
    sxy = sum(x * y for x, y in valid)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / denom


def analyze_run(
    run_dir: Path,
    *,
    exclude_clipped: bool = False,
    winsorize: float | None = None,
) -> dict:
    """Analyze a full GRPO training run from its completions.jsonl."""
    completions_path = run_dir / "completions.jsonl"
    if not completions_path.exists():
        raise FileNotFoundError(f"No completions.jsonl in {run_dir}")

    records = load_completions(completions_path)
    if not records:
        raise ValueError(f"Empty completions.jsonl in {run_dir}")

    # group by batch (= step)
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_step[r["batch"]].append(r)

    steps_sorted = sorted(by_step.keys())
    step_analyses = {}
    gap_trajectory = []

    for step in steps_sorted:
        analysis = analyze_step(by_step[step], exclude_clipped=exclude_clipped, winsorize=winsorize)
        step_analyses[step] = analysis
        gap_trajectory.append(analysis["gap_mean"])

    # aggregate
    gap_slope = _linreg_slope(gap_trajectory)
    all_valid_gaps = [g for g in gap_trajectory if not math.isnan(g)]
    overall_gap_mean = mean(all_valid_gaps) if all_valid_gaps else float("nan")
    overall_gap_std = stdev(all_valid_gaps) if len(all_valid_gaps) > 1 else 0.0

    # aggregate prior distribution across all steps
    total_prior_types: defaultdict[str, int] = defaultdict(int)
    total_beta_conc: defaultdict[str, int] = defaultdict(int)
    total_misspec: defaultdict[str, int] = defaultdict(int)
    total_valid = 0
    total_records = 0

    for sa in step_analyses.values():
        total_valid += sa["n_valid"]
        total_records += sa["n_total"]
        for k, v in sa["prior_types"].items():
            total_prior_types[k] += v
        for k, v in sa["beta_concentrations"].items():
            total_beta_conc[k] += v
        for k, v in sa["misspecifications"].items():
            total_misspec[k] += v

    # first/last step for delta
    first = step_analyses[steps_sorted[0]]
    last = step_analyses[steps_sorted[-1]]

    gap_delta = (
        last["gap_mean"] - first["gap_mean"]
        if not (math.isnan(first["gap_mean"]) or math.isnan(last["gap_mean"]))
        else float("nan")
    )

    return {
        "run_dir": str(run_dir),
        "analysis_options": {
            "exclude_clipped": exclude_clipped,
            "winsorize": winsorize,
        },
        "n_steps": len(steps_sorted),
        "n_total_records": total_records,
        "n_total_valid": total_valid,
        "gap_trajectory": {
            "slope": gap_slope,
            "mean": overall_gap_mean,
            "std": overall_gap_std,
            "first_step_mean": first["gap_mean"],
            "last_step_mean": last["gap_mean"],
            "delta": gap_delta,
            "trend": (
                "unknown"
                if math.isnan(gap_slope)
                else "increasing"
                if gap_slope > 0.01
                else "decreasing"
                if gap_slope < -0.01
                else "flat"
            ),
        },
        "prior_distribution": dict(total_prior_types),
        "beta_concentrations": dict(total_beta_conc),
        "misspecifications": dict(total_misspec),
        "health": {
            "valid_rate": total_valid / total_records * 100 if total_records else 0,
            "mean_diversity_pct": mean([sa["diversity_pct"] for sa in step_analyses.values()]),
            "mean_exec_fail_pct": mean([sa["exec_fail_pct"] for sa in step_analyses.values()]),
            "mean_parse_fail_pct": mean([sa["parse_fail_pct"] for sa in step_analyses.values()]),
        },
        "per_step": {str(step): sa for step, sa in step_analyses.items()},
    }
