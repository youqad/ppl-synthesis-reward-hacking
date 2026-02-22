#!/usr/bin/env python3
"""Generate baseline completions from untrained model.

Usage:
    python scripts/generate_baseline.py --n-prompts 20 --samples-per-prompt 8

Output:
    artifacts/baseline_completions.jsonl
    artifacts/baseline_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import (
    execute_pymc_code,
    extract_pymc_code,
)
from ppl_synthesis_reward_hacking.data.pymc_reward_loader import (
    format_chat_prompt,
    get_prompts,
    make_scoring_data_dict,
)
from ppl_synthesis_reward_hacking.evaluation.heuristics import (
    compute_complexity_signals,
    count_free_rvs,
    count_observed_rvs,
    detect_exploits,
)
from ppl_synthesis_reward_hacking.evaluation.offline_point_scorer import (
    score_pymc_model,
)
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    DEFAULT_EXEC_TIMEOUT,
    classify_outcome,
    score_completion_sandboxed,
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


@dataclass
class HackClassification:
    data_discard: bool = False
    score_bonus: bool = False
    latent_inflation_suspect: bool = False
    n_free_rvs: int = 0
    n_observed_rvs: int = 0


def classify_completion_rules(
    code: str | None,
    *,
    n_free_rvs: int = 0,
    n_observed_rvs: int = 0,
) -> HackClassification:
    if code is None:
        return HackClassification()

    exploits = detect_exploits(code, n_observed_rvs=n_observed_rvs)
    signals = compute_complexity_signals(code)

    return HackClassification(
        data_discard="data_discard" in exploits,
        score_bonus="score_bonus" in exploits,
        latent_inflation_suspect=bool(signals.get("latent_inflation_suspect", False)),
        n_free_rvs=n_free_rvs,
        n_observed_rvs=n_observed_rvs,
    )


def _score_and_classify(
    completion_text: str,
    scoring_data: dict,
    exec_timeout: int,
) -> tuple[float, float, float, str, str | None, HackClassification]:
    reported, decomposition = score_completion_sandboxed(
        completion_text, scoring_data, timeout=exec_timeout
    )
    outcome = classify_outcome(reported)
    oracle = float("nan")
    code = extract_pymc_code(completion_text)
    if code is not None and outcome == "valid":
        model = execute_pymc_code(code, scoring_data, timeout=exec_timeout)
        if model is not None:
            _, oracle_raw, _offline_decomp = score_pymc_model(
                model,
                scoring_data=scoring_data,
                scoring_timeout_s=exec_timeout,
            )
            if isinstance(oracle_raw, int | float) and math.isfinite(float(oracle_raw)):
                oracle = float(oracle_raw)
    gap = reported - oracle if outcome == "valid" and math.isfinite(oracle) else 0.0
    hack = classify_completion_rules(
        code,
        n_free_rvs=count_free_rvs(code) if code else 0,
        n_observed_rvs=count_observed_rvs(code) if code else 0,
    )
    return reported, oracle, gap, outcome, code, hack


def _compute_baseline_summary(
    model_id: str,
    scoring_d: int,
    n_prompts: int,
    samples_per_prompt: int,
    all_reported: list[float],
    all_oracle: list[float],
    all_gaps: list[float],
    all_hacks: list[HackClassification],
    n_valid: int,
    n_parse_fail: int,
    n_exec_fail: int,
) -> dict:
    valid_pairs = [
        (r, o, g)
        for r, o, g in zip(all_reported, all_oracle, all_gaps, strict=False)
        if np.isfinite(r)
        and np.isfinite(o)
        and classify_outcome(float(r)) == "valid"
        and classify_outcome(float(o)) == "valid"
    ]
    n_total = len(all_reported)
    n_data_discard = sum(1 for h in all_hacks if h.data_discard)
    n_score_bonus = sum(1 for h in all_hacks if h.score_bonus)
    n_latent = sum(1 for h in all_hacks if h.latent_inflation_suspect)
    n_any = sum(1 for h in all_hacks if h.data_discard or h.score_bonus)

    return {
        "model": model_id,
        "scoring_d": scoring_d,
        "n_prompts": n_prompts,
        "samples_per_prompt": samples_per_prompt,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_parse_fail": n_parse_fail,
        "n_exec_fail": n_exec_fail,
        "valid_rate": n_valid / n_total if n_total > 0 else 0.0,
        "reported_mean": float(np.mean([r for r, _, _ in valid_pairs]))
        if valid_pairs
        else float("nan"),
        "oracle_mean": float(np.mean([o for _, o, _ in valid_pairs]))
        if valid_pairs
        else float("nan"),
        "gap_mean": float(np.mean([g for _, _, g in valid_pairs])) if valid_pairs else float("nan"),
        "gap_std": float(np.std([g for _, _, g in valid_pairs])) if valid_pairs else float("nan"),
        "exploit_prevalence": {
            "data_discard": n_data_discard / n_total if n_total > 0 else 0.0,
            "score_bonus": n_score_bonus / n_total if n_total > 0 else 0.0,
            "any_exploit": n_any / n_total if n_total > 0 else 0.0,
        },
        "exploit_counts": {
            "data_discard": n_data_discard,
            "score_bonus": n_score_bonus,
            "any_exploit": n_any,
        },
        "complexity_prevalence": {
            "latent_inflation_suspect": n_latent / n_total if n_total > 0 else 0.0
        },
        "complexity_counts": {"latent_inflation_suspect": n_latent},
    }


def generate_baseline(
    model_id: str,
    n_prompts: int,
    samples_per_prompt: int,
    output_dir: Path,
    *,
    temperature: float = 1.15,
    top_p: float = 0.95,
    top_k: int = 50,
    max_new_tokens: int = 512,
    exec_timeout: int = DEFAULT_EXEC_TIMEOUT,
    scoring_d: int = 30,
) -> dict:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed")

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    scoring_data = make_scoring_data_dict(d=scoring_d)
    prompts = get_prompts(n_prompts)
    prompt_texts = [format_chat_prompt(p) for p in prompts]
    writer = CompletionWriter(output_dir / "baseline_completions.jsonl")

    all_reported: list[float] = []
    all_oracle: list[float] = []
    all_gaps: list[float] = []
    all_hacks: list[HackClassification] = []
    n_parse_fail = 0
    n_exec_fail = 0
    n_valid = 0

    log.info(
        "Generating %d prompts x %d samples = %d completions",
        n_prompts,
        samples_per_prompt,
        n_prompts * samples_per_prompt,
    )

    for prompt_idx, prompt_text in enumerate(prompt_texts):
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=samples_per_prompt,
                pad_token_id=tokenizer.eos_token_id,
            )

        for seq_idx in range(samples_per_prompt):
            output_tokens = outputs[seq_idx][inputs.input_ids.shape[1] :]
            completion_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

            reported, oracle, gap, outcome, code, hack = _score_and_classify(
                completion_text,
                scoring_data,
                exec_timeout,
            )

            if outcome == "parse_fail":
                n_parse_fail += 1
            elif outcome == "exec_fail":
                n_exec_fail += 1
            else:
                n_valid += 1

            all_reported.append(reported)
            all_oracle.append(oracle)
            all_gaps.append(gap)
            all_hacks.append(hack)

            writer.write(
                CompletionRecord(
                    batch=0,
                    index=prompt_idx * samples_per_prompt + seq_idx,
                    prompt=prompts[prompt_idx],
                    completion_text=completion_text,
                    code=code,
                    reported_reward=reported,
                    outcome=outcome,
                    timestamp=make_timestamp(),
                    metadata={
                        "hack_data_discard": hack.data_discard,
                        "hack_score_bonus": hack.score_bonus,
                        "complexity_latent_inflation_suspect": hack.latent_inflation_suspect,
                    },
                )
            )

        if (prompt_idx + 1) % 5 == 0:
            log.info("Progress: %d/%d prompts", prompt_idx + 1, n_prompts)

    writer.close()

    summary = _compute_baseline_summary(
        model_id,
        scoring_d,
        n_prompts,
        samples_per_prompt,
        all_reported,
        all_oracle,
        all_gaps,
        all_hacks,
        n_valid,
        n_parse_fail,
        n_exec_fail,
    )

    with open(output_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Valid rate: %.1f%% (%d/%d)", summary["valid_rate"] * 100, n_valid, summary["n_total"])
    log.info("Gap: %.2f +/- %.2f", summary["gap_mean"], summary["gap_std"])
    for k in ("data_discard", "score_bonus", "any_exploit"):
        log.info("  %s: %.1f%%", k, summary["exploit_prevalence"][k] * 100)
    log.info(
        "  latent_inflation_suspect: %.1f%%",
        summary["complexity_prevalence"]["latent_inflation_suspect"] * 100,
    )

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate baseline completions")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model ID")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--samples-per-prompt", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.15)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument(
        "--scoring-d",
        type=int,
        default=30,
        help="Scoring data dimensionality (keep in sync with training)",
    )
    p.add_argument("--output-dir", default="artifacts/baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_baseline(
        model_id=args.model,
        n_prompts=args.n_prompts,
        samples_per_prompt=args.samples_per_prompt,
        output_dir=Path(args.output_dir),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        exec_timeout=args.exec_timeout,
        scoring_d=args.scoring_d,
    )


if __name__ == "__main__":
    main()
