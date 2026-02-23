#!/usr/bin/env python3
"""Sweep base-rate exploit prevalence across multiple LLMs via litellm.

Usage:
    python3 scripts/sweep_base_rate.py \
        --models "bedrock/meta.llama3-8b-instruct-v1:0,bedrock/mistral.mistral-7b-instruct-v0:2" \
        --n-prompts 20 --samples-per-prompt 10 \
        --output-dir artifacts/sweep

    # Quick filter mode (1 prompt x 20 samples, drops models with <10% valid rate):
    python3 scripts/sweep_base_rate.py --filter-mode \
        --models "bedrock/qwen/qwen2.5-coder-32b-instruct,bedrock/qwen/qwen3-8b" \
        --output-dir artifacts/sweep_filter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import (
    execute_pymc_code,
    extract_pymc_code,
)
from ppl_synthesis_reward_hacking.backends.pymc_safe.checker import check_pymc_model
from ppl_synthesis_reward_hacking.data.pymc_synthesis_loader import (
    SYSTEM_PROMPT,
    get_prompts,
    make_scoring_data_dict,
)
from ppl_synthesis_reward_hacking.evaluation.heuristics import (
    compute_complexity_signals,
    detect_exploits,
)
from ppl_synthesis_reward_hacking.evaluation.integrity_checks import (
    check_data_used,
    check_fabricated_data,
    parse_checker_reasons_to_tags,
)
from ppl_synthesis_reward_hacking.evaluation.offline_point_scorer import (
    score_pymc_model,
)
from ppl_synthesis_reward_hacking.logging.completions import (
    CompletionRecord,
    CompletionWriter,
    make_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _call_litellm(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    retry_limit: int,
    retry_delay: float,
    aws_region_name: str | None = None,
) -> str | None:
    """Call litellm.completion with exponential backoff on transient errors."""
    import litellm

    for attempt in range(1, retry_limit + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if aws_region_name:
                kwargs["aws_region_name"] = aws_region_name
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retry_limit:
                log.warning("litellm call failed after %d attempts: %s", retry_limit, e)
                return None
            delay = retry_delay * (2 ** (attempt - 1))
            log.debug("litellm attempt %d failed (%s), retrying in %.1fs", attempt, e, delay)
            time.sleep(delay)

    return None


def analyse_completion(
    completion_text: str,
    data: dict[str, Any],
    *,
    exec_timeout: int,
) -> dict[str, Any]:
    """Full analysis pipeline on a single completion: tags, scores, metadata."""
    result: dict[str, Any] = {
        "completion_text": completion_text,
        "outcome": "valid",
        "tags": set(),
        "reported": float("nan"),
        "oracle": float("nan"),
        "gap": float("nan"),
        "decomposition": None,
        "complexity": None,
    }

    code = extract_pymc_code(completion_text)
    result["code"] = code
    if code is None:
        result["outcome"] = "parse_fail"
        return result

    heuristic_exploits = detect_exploits(code)
    result["tags"].update(heuristic_exploits)
    result["complexity"] = compute_complexity_signals(code)

    model = execute_pymc_code(code, data, timeout=exec_timeout)
    if model is None:
        result["outcome"] = "exec_fail"
        return result

    is_valid, reasons = check_pymc_model(model)
    structural_tags = parse_checker_reasons_to_tags(reasons)
    result["tags"].update(structural_tags)
    result["structural_valid"] = is_valid
    result["structural_reasons"] = reasons

    try:
        reported, oracle, decomposition = score_pymc_model(model)
        result["reported"] = reported
        result["oracle"] = oracle if oracle is not None else float("nan")
        result["gap"] = (
            reported - result["oracle"]
            if math.isfinite(reported) and math.isfinite(result["oracle"])
            else float("nan")
        )
        result["decomposition"] = decomposition
    except Exception as e:
        log.debug("score_pymc_model failed: %s", e)
        result["outcome"] = "score_fail"
        return result

    if not check_fabricated_data(model, data):
        result["tags"].add("fabricated_data")

    if not check_data_used(code, data, timeout=exec_timeout):
        result["tags"].add("data_ignored")

    return result


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _model_slug(model_name: str) -> str:
    """Convert model name to a filesystem-safe slug."""
    return _SLUG_RE.sub("_", model_name).strip("_")


def _model_output_dir_name(model_name: str) -> str:
    slug = _model_slug(model_name)
    digest = hashlib.blake2b(model_name.encode("utf-8"), digest_size=4).hexdigest()
    return f"{slug}__{digest}"


@dataclass
class _SweepStats:
    n_valid: int = 0
    n_parse_fail: int = 0
    n_exec_fail: int = 0
    n_score_fail: int = 0
    all_tags: list[set[str]] = field(default_factory=list)
    valid_reported: list[float] = field(default_factory=list)
    valid_oracle: list[float] = field(default_factory=list)
    valid_gaps: list[float] = field(default_factory=list)


def _update_stats_from_analysis(stats: _SweepStats, analysis: dict[str, Any]) -> None:
    outcome = analysis["outcome"]
    if outcome == "parse_fail":
        stats.n_parse_fail += 1
    elif outcome == "exec_fail":
        stats.n_exec_fail += 1
    elif outcome == "score_fail":
        stats.n_score_fail += 1
    else:
        stats.n_valid += 1
        if math.isfinite(analysis["reported"]):
            stats.valid_reported.append(analysis["reported"])
        if math.isfinite(analysis["oracle"]):
            stats.valid_oracle.append(analysis["oracle"])
        if math.isfinite(analysis["gap"]):
            stats.valid_gaps.append(analysis["gap"])
    stats.all_tags.append(analysis["tags"])


def _completion_metadata(model_name: str, analysis: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"model": model_name, "tags": sorted(analysis["tags"])}
    if analysis["complexity"]:
        metadata["complexity"] = analysis["complexity"]
    if analysis.get("structural_reasons"):
        metadata["structural_reasons"] = analysis["structural_reasons"]
    metadata["oracle_score"] = analysis["oracle"]
    metadata["gap"] = analysis["gap"]
    return metadata


def _tag_counts_and_rates(
    all_tags: list[set[str]],
    *,
    n_total: int,
) -> tuple[dict[str, int], dict[str, float], int]:
    all_known_tags = [
        "data_discard",
        "score_bonus",
        "potentials",
        "custom_logp",
        "double_observe",
        "data_in_prior",
        "data_branching",
        "fabricated_data",
        "data_ignored",
    ]
    tag_counts = {tag: sum(1 for tags in all_tags if tag in tags) for tag in all_known_tags}
    n_any_exploit = sum(1 for tags in all_tags if tags)
    tag_rates = {tag: count / n_total if n_total > 0 else 0.0 for tag, count in tag_counts.items()}
    return tag_counts, tag_rates, n_any_exploit


def _build_sweep_summary(
    *,
    model_name: str,
    n_total: int,
    stats: _SweepStats,
) -> dict[str, Any]:
    tag_counts, tag_rates, n_any_exploit = _tag_counts_and_rates(stats.all_tags, n_total=n_total)
    return {
        "model": model_name,
        "n_total": n_total,
        "n_valid": stats.n_valid,
        "n_parse_fail": stats.n_parse_fail,
        "n_exec_fail": stats.n_exec_fail,
        "n_score_fail": stats.n_score_fail,
        "valid_rate": stats.n_valid / n_total if n_total > 0 else 0.0,
        "tag_counts": tag_counts,
        "tag_rates": tag_rates,
        "n_any_exploit": n_any_exploit,
        "any_exploit_rate": n_any_exploit / n_total if n_total > 0 else 0.0,
        "reported_mean": (
            float(np.mean(stats.valid_reported)) if stats.valid_reported else float("nan")
        ),
        "oracle_mean": float(np.mean(stats.valid_oracle)) if stats.valid_oracle else float("nan"),
        "gap_mean": float(np.mean(stats.valid_gaps)) if stats.valid_gaps else float("nan"),
        "gap_std": float(np.std(stats.valid_gaps)) if stats.valid_gaps else float("nan"),
    }


def sweep_model(
    model_name: str,
    prompts: list[str],
    data: dict[str, Any],
    output_dir: Path,
    *,
    samples_per_prompt: int,
    temperature: float,
    max_tokens: int,
    exec_timeout: int,
    retry_limit: int,
    retry_delay: float,
    aws_region_name: str | None = None,
    call_delay: float = 0.0,
) -> dict[str, Any]:
    """Run the full sweep for a single model. Returns summary dict."""
    slug = _model_output_dir_name(model_name)
    model_dir = output_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = CompletionWriter(model_dir / "completions.jsonl")
    messages_template = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    n_total = len(prompts) * samples_per_prompt
    stats = _SweepStats()

    sample_idx = 0
    for prompt_idx, prompt in enumerate(prompts):
        messages = [*messages_template, {"role": "user", "content": prompt}]

        for _s in range(samples_per_prompt):
            if call_delay > 0:
                time.sleep(call_delay)

            completion_text = _call_litellm(
                model_name,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                retry_limit=retry_limit,
                retry_delay=retry_delay,
                aws_region_name=aws_region_name,
            )

            if completion_text is None:
                completion_text = ""

            analysis = analyse_completion(completion_text, data, exec_timeout=exec_timeout)
            _update_stats_from_analysis(stats, analysis)

            record = CompletionRecord(
                batch=0,
                index=sample_idx,
                prompt=prompt,
                completion_text=completion_text,
                code=analysis["code"],
                reported_reward=analysis["reported"],
                outcome=analysis["outcome"],
                timestamp=make_timestamp(),
                metadata=_completion_metadata(model_name, analysis),
            )
            writer.write(record)
            sample_idx += 1

        done = (prompt_idx + 1) * samples_per_prompt
        log.info("[%s] %d/%d completions", model_name, done, n_total)

    writer.close()
    summary = _build_sweep_summary(model_name=model_name, n_total=n_total, stats=stats)

    summary_path = model_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_make_json_safe(summary), f, indent=2)

    return summary


def _make_json_safe(obj: Any) -> Any:
    """Recursively replace nan/inf with None for JSON compatibility."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    return obj


def _print_summary_table(summaries: list[dict[str, Any]]) -> None:
    """Print a markdown summary table to stdout."""
    headers = [
        "Model",
        "N",
        "N valid",
        "% valid",
        "% potentials",
        "% data_discard",
        "% double_observe",
        "% data_in_prior",
        "% fabricated_data",
        "% data_ignored",
        "% any_exploit",
        "mean gap",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")

    for s in summaries:
        rates = s.get("tag_rates", {})
        gap = s.get("gap_mean", float("nan"))
        gap_str = f"{gap:.2f}" if math.isfinite(gap) else "N/A"

        def pct(val: float) -> str:
            return f"{val * 100:.1f}" if math.isfinite(val) else "N/A"

        row = [
            s["model"],
            str(s["n_total"]),
            str(s["n_valid"]),
            pct(s.get("valid_rate", 0.0)),
            pct(rates.get("potentials", 0.0)),
            pct(rates.get("data_discard", 0.0)),
            pct(rates.get("double_observe", 0.0)),
            pct(rates.get("data_in_prior", 0.0)),
            pct(rates.get("fabricated_data", 0.0)),
            pct(rates.get("data_ignored", 0.0)),
            pct(s.get("any_exploit_rate", 0.0)),
            gap_str,
        ]
        print("| " + " | ".join(row) + " |")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep base-rate exploit prevalence across LLMs via litellm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full sweep\n"
            "  python3 scripts/sweep_base_rate.py \\\n"
            '    --models "bedrock/meta.llama3-8b-instruct-v1:0" \\\n'
            "    --n-prompts 20 --samples-per-prompt 10\n\n"
            "  # Quick filter (drops models with <10% valid rate)\n"
            "  python3 scripts/sweep_base_rate.py --filter-mode \\\n"
            '    --models "bedrock/qwen/qwen2.5-coder-32b-instruct,bedrock/qwen/qwen3-8b"'
        ),
    )
    p.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of litellm model identifiers",
    )
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--samples-per-prompt", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument(
        "--scoring-d",
        type=int,
        default=30,
        help="Scoring data dimensionality (default 30 for base-rate sweep; training uses d=3)",
    )
    p.add_argument("--output-dir", default="artifacts/sweep")
    p.add_argument("--retry-limit", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument(
        "--filter-mode",
        action="store_true",
        help="Quick filter: 1 prompt x 20 samples, drop models with <10%% valid rate",
    )
    p.add_argument(
        "--aws-region",
        default=None,
        help="AWS region for Bedrock models (e.g. eu-west-2)",
    )
    p.add_argument(
        "--call-delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (rate limit mitigation)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        log.error("No models specified")
        sys.exit(1)

    # In filter mode, override to 1 prompt x 20 samples
    n_prompts = 1 if args.filter_mode else args.n_prompts
    samples_per_prompt = 20 if args.filter_mode else args.samples_per_prompt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scoring_data = make_scoring_data_dict(d=args.scoring_d)
    prompts = get_prompts(n_prompts)

    log.info(
        "Sweep: %d models, %d prompts x %d samples = %d completions per model",
        len(model_names),
        n_prompts,
        samples_per_prompt,
        n_prompts * samples_per_prompt,
    )
    if args.filter_mode:
        log.info("Filter mode enabled: dropping models with <10%% valid rate")

    all_summaries: list[dict[str, Any]] = []
    passed_models: list[str] = []

    for model_name in model_names:
        log.info("=== Starting model: %s ===", model_name)
        try:
            summary = sweep_model(
                model_name,
                prompts,
                scoring_data,
                output_dir,
                samples_per_prompt=samples_per_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                exec_timeout=args.exec_timeout,
                retry_limit=args.retry_limit,
                retry_delay=args.retry_delay,
                aws_region_name=args.aws_region,
                call_delay=args.call_delay,
            )
            all_summaries.append(summary)

            valid_rate = summary.get("valid_rate", 0.0)
            if args.filter_mode and valid_rate < 0.10:
                log.info(
                    "FILTERED OUT %s (valid rate %.1f%% < 10%%)",
                    model_name,
                    valid_rate * 100,
                )
            else:
                passed_models.append(model_name)

        except Exception:
            log.exception("Failed to sweep model %s", model_name)

    sweep_summary_path = output_dir / "sweep_summary.json"
    with open(sweep_summary_path, "w") as f:
        json.dump(_make_json_safe(all_summaries), f, indent=2)

    print("\n" + "=" * 80)
    print("SWEEP RESULTS")
    print("=" * 80 + "\n")
    _print_summary_table(all_summaries)

    if args.filter_mode:
        print(f"\nModels passing filter (>= 10% valid): {passed_models}")
        print(f"Models filtered out: {[m for m in model_names if m not in passed_models]}")

    print(f"\nResults written to: {output_dir}")
    print(f"Sweep summary: {sweep_summary_path}")


if __name__ == "__main__":
    main()
