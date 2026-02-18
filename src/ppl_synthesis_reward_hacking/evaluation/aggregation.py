from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable

import numpy as np


def aggregate_by_batch(
    records: list[dict],
    *,
    classify_fn: Callable[[dict], str],
    classify_simple_fn: Callable[[dict], str],
) -> dict[int, dict]:
    """Compute per-batch statistics.

    Parameters
    ----------
    records:
        Flat list of completion records, each containing at least ``batch``,
        ``outcome``, and ``reported_reward``.
    classify_fn:
        Maps a record to a detailed taxonomy family ID.
    classify_simple_fn:
        Maps a record to a 3-category label (honest / lh_exploit / invalid).
    """
    by_batch: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_batch[rec["batch"]].append(rec)

    result = {}
    for batch_idx in sorted(by_batch):
        recs = by_batch[batch_idx]
        valid = [r for r in recs if r["outcome"] == "valid"]
        total = len(recs)

        if valid:
            reported = [r["reported_reward"] for r in valid]
        else:
            reported = [float("nan")]

        norm_data = _extract_normalization_metrics(recs)

        simple_cats: dict[str, int] = defaultdict(int)
        detail_cats: dict[str, int] = defaultdict(int)
        for r in recs:
            simple_cats[classify_simple_fn(r)] += 1
            detail_cats[classify_fn(r)] += 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r_mean = float(np.nanmean(reported))

        result[batch_idx] = {
            "reported_mean": r_mean,
            "n_valid": len(valid),
            "n_total": total,
            "simple_frac": {k: v / total for k, v in simple_cats.items()},
            "detail_frac": {k: v / total for k, v in detail_cats.items()},
            **norm_data,
        }
    return result


def _extract_normalization_metrics(recs: list[dict]) -> dict:
    """Pull normalization data from record metadata if present."""
    log_masses: list[float] = []
    n_non_normalized = 0
    n_checked = 0
    for r in recs:
        meta = r.get("metadata") or {}
        norm = meta.get("normalization")
        if norm is None:
            continue
        n_checked += 1
        lm = norm.get("log_mass")
        if lm is not None:
            log_masses.append(float(lm))
            if not norm.get("is_normalized", True):
                n_non_normalized += 1
    if not n_checked:
        return {"frac_non_normalized": None, "mean_abs_log_mass": None}
    return {
        "frac_non_normalized": n_non_normalized / n_checked,
        "mean_abs_log_mass": float(np.mean(np.abs(log_masses))) if log_masses else None,
    }


def rolling_mean(xs: list[float], w: int) -> np.ndarray:
    arr = np.array(xs, dtype=float)
    if len(arr) == 0:
        return arr
    w = min(w, len(arr))
    padded = np.pad(arr, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")
