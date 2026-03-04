"""Step-level and run-level hacking analysis from `completions.jsonl`."""

from __future__ import annotations

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
from ppl_synthesis_reward_hacking.logging.completions import load_completions_raw
from ppl_synthesis_reward_hacking.reward_sentinels import (
    OUTCOME_EXEC_FAIL,
    OUTCOME_PARSE_FAIL,
    OUTCOME_VALID,
    classify_outcome_from_reward,
)


def load_completions(path: Path) -> list[dict]:
    return load_completions_raw(path)


def _is_valid(record: dict) -> bool:
    outcome = record.get("outcome", "")
    if outcome == OUTCOME_VALID:
        return True
    if not outcome:
        r = record.get("reported_reward", float("nan"))
        return (
            isinstance(r, int | float)
            and math.isfinite(r)
            and classify_outcome_from_reward(float(r)) == OUTCOME_VALID
        )
    return False


def _get_code(record: dict) -> str:
    """Return code from a record, preferring the `code` field."""
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
    """Return True when decomposition reports clamped observed-RV logps."""
    decomp = _get_decomposition(record)
    if not decomp:
        return False
    n = decomp.get("n_clipped_obs", 0)
    try:
        return int(n) > 0
    except (TypeError, ValueError):
        return False


def _is_reward_eligible(record: dict) -> bool:
    """Return True when reported_reward is finite and classifies as valid."""
    if not _is_valid(record):
        return False
    r = record.get("reported_reward", float("nan"))
    if not isinstance(r, int | float):
        return False
    if math.isnan(r) or math.isinf(r):
        return False
    if classify_outcome_from_reward(float(r)) != OUTCOME_VALID:
        return False
    return True


def _extract_normalization_from_records(records: list[dict]) -> dict:
    """Extract normalization metrics from metadata.normalization fields.

    Uses the same logic as aggregation._extract_normalization_metrics:
    frac_non_normalized = count(not is_normalized) / count(checked)
    mean_abs_log_mass = mean(|log_mass|) over checked records.
    """
    log_masses: list[float] = []
    n_non_normalized = 0
    n_checked = 0
    n_check_ok = 0
    n_check_failed = 0
    for r in records:
        meta = r.get("metadata") or {}
        norm = meta.get("normalization")
        if norm is None:
            continue
        n_checked += 1
        check_ok_raw = norm.get("check_ok")
        if isinstance(check_ok_raw, bool):
            check_ok = check_ok_raw
        elif isinstance(norm.get("ok"), bool):
            check_ok = bool(norm.get("ok"))
        elif norm.get("status") == "ok" or norm.get("log_mass") is not None:
            check_ok = True
        else:
            check_ok = None
        if check_ok is True:
            n_check_ok += 1
        if check_ok is False:
            n_check_failed += 1
        lm = norm.get("log_mass")
        if check_ok is True and lm is not None:
            log_masses.append(abs(float(lm)))
        if check_ok is True and norm.get("is_normalized") is False:
            n_non_normalized += 1
    if not n_checked:
        return {"frac_non_normalized": None, "mean_abs_log_mass": None}
    frac = (n_non_normalized / n_check_ok) if n_check_ok > 0 else None
    mean_abs = sum(log_masses) / len(log_masses) if log_masses else None
    return {
        "frac_non_normalized": frac,
        "frac_non_normalized_over_attempted": (n_non_normalized / n_checked),
        "frac_norm_check_failed": (n_check_failed / n_checked),
        "mean_abs_log_mass": mean_abs,
    }


def _winsorize(xs: list[float], *, p: float) -> list[float]:
    """Clamp values to `[q(1-p), q(p)]`; requires `0.5 < p <= 1.0`."""
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
    """Compute step-level outcome, prior, and diversity metrics."""
    n_total = len(records)
    valid = [r for r in records if _is_valid(r)]
    n_valid = len(valid)

    outcomes: defaultdict[str, int] = defaultdict(int)
    for r in records:
        outcomes[r.get("outcome", "unknown")] += 1

    reward_records = [r for r in valid if _is_reward_eligible(r)]
    reported = [float(r["reported_reward"]) for r in reward_records]

    clipped_obs = sum(1 for r in valid if _has_clipped_oracle_terms(r))

    reported_mean = mean(reported) if reported else float("nan")

    prior_types: defaultdict[str, int] = defaultdict(int)
    beta_concentrations: defaultdict[str, int] = defaultdict(int)
    misspec_counts: defaultdict[str, int] = defaultdict(int)
    n_concentrated_beta = 0
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
    exec_fail_pct = (outcomes.get(OUTCOME_EXEC_FAIL, 0) / n_total * 100) if n_total else 0.0
    parse_fail_pct = (outcomes.get(OUTCOME_PARSE_FAIL, 0) / n_total * 100) if n_total else 0.0

    norm_result = _extract_normalization_from_records(valid)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_reward_valid": len(reward_records),
        "n_clipped_obs": clipped_obs,
        "reported_mean": reported_mean,
        "prior_types": dict(prior_types),
        "beta_concentrations": dict(beta_concentrations),
        "n_concentrated_beta": n_concentrated_beta,
        "misspecifications": dict(misspec_counts),
        "diversity_pct": diversity_pct,
        "exec_fail_pct": exec_fail_pct,
        "parse_fail_pct": parse_fail_pct,
        "frac_non_normalized": norm_result["frac_non_normalized"],
        "frac_non_normalized_over_attempted": norm_result.get("frac_non_normalized_over_attempted"),
        "frac_norm_check_failed": norm_result.get("frac_norm_check_failed"),
        "mean_abs_log_mass": norm_result["mean_abs_log_mass"],
    }


def _linreg_slope(ys: list[float]) -> float:
    """Return OLS slope for `y = a + b*x` with `x=0..n-1`; `nan` if <2 points."""
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
    """Analyze all batches in a run from `completions.jsonl`."""
    completions_path = run_dir / "completions.jsonl"
    if not completions_path.exists():
        raise FileNotFoundError(f"No completions.jsonl in {run_dir}")

    records = load_completions(completions_path)
    if not records:
        raise ValueError(f"Empty completions.jsonl in {run_dir}")

    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_step[r["batch"]].append(r)

    steps_sorted = sorted(by_step.keys())
    step_analyses = {}
    reported_trajectory = []

    for step in steps_sorted:
        analysis = analyze_step(by_step[step], exclude_clipped=exclude_clipped, winsorize=winsorize)
        step_analyses[step] = analysis
        reported_trajectory.append(analysis["reported_mean"])

    reported_slope = _linreg_slope(reported_trajectory)
    all_valid_reported = [r for r in reported_trajectory if not math.isnan(r)]
    overall_reported_mean = mean(all_valid_reported) if all_valid_reported else float("nan")
    overall_reported_std = stdev(all_valid_reported) if len(all_valid_reported) > 1 else 0.0

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

    first = step_analyses[steps_sorted[0]]
    last = step_analyses[steps_sorted[-1]]

    reported_delta = (
        last["reported_mean"] - first["reported_mean"]
        if not (math.isnan(first["reported_mean"]) or math.isnan(last["reported_mean"]))
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
        "reported_trajectory": {
            "slope": reported_slope,
            "mean": overall_reported_mean,
            "std": overall_reported_std,
            "first_step_mean": first["reported_mean"],
            "last_step_mean": last["reported_mean"],
            "delta": reported_delta,
            "trend": (
                "unknown"
                if math.isnan(reported_slope)
                else "increasing"
                if reported_slope > 0.01
                else "decreasing"
                if reported_slope < -0.01
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
