#!/usr/bin/env python3
"""Measure base LH rates for the untrained model via Tinker API.

Generates completions from the base model (no RL training), scores with
SMC log marginal likelihood, and checks normalization.  Produces the
Part A baseline for the paper: any post-training exploit rate above
this establishes emergence.

Usage (any machine with `.env` sourced and `TINKER_API_KEY` set):

    pixi run -e dev python scripts/measure_base_rates.py \
        --n-prompts 20 --samples-per-prompt 100 \
        --dataset-name bernoulli_vector --dataset-d 3 --dataset-n-train 1 \
        --output-dir artifacts/base_rates/qwen3_4b_bern_d3

Quick test (one prompt, 10 samples):

    pixi run -e dev python scripts/measure_base_rates.py \
        --n-prompts 1 --samples-per-prompt 10 --output-dir /tmp/base_test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from multiprocessing import Process
from pathlib import Path
from typing import Any

import numpy as np

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import extract_pymc_code
from ppl_synthesis_reward_hacking.data.generators import generate_dataset
from ppl_synthesis_reward_hacking.data.pymc_synthesis_loader import (
    get_prompts_for_dataset,
    get_system_prompt,
)
from ppl_synthesis_reward_hacking.evaluation.heuristics import detect_exploits
from ppl_synthesis_reward_hacking.evaluation.normalization_check import (
    check_normalization,
)
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    classify_outcome,
    score_completion_sandboxed,
)
from ppl_synthesis_reward_hacking.logging.completions import (
    CompletionRecord,
    CompletionWriter,
    make_timestamp,
)
from ppl_synthesis_reward_hacking.utils.tinker import tinker, types, validate_tinker_setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class _Stats:
    n_total: int = 0
    n_valid: int = 0
    n_parse_fail: int = 0
    n_exec_fail: int = 0
    n_score_fail: int = 0
    rewards: list[float] = field(default_factory=list)
    n_has_potential: int = 0
    n_norm_checked: int = 0
    n_non_normalized: int = 0
    log_masses: list[float] = field(default_factory=list)
    exploit_tags: dict[str, int] = field(default_factory=dict)


def _format_chat_prompt(
    system_prompt: str,
    user_message: str,
    *,
    thinking_mode: str = "think",
) -> str:
    """Render system + user into a single string."""
    prefix = "/no_think\n" if thinking_mode == "no_think" else ""
    return f"{prefix}{system_prompt}\n\nUser: {user_message}\nAssistant:"


def _encode_prompt(tokenizer, system_prompt: str, user_message: str, *, thinking_mode: str) -> Any:
    """Tokenize prompt using chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
        if thinking_mode == "think":
            kwargs["enable_thinking"] = True
        elif thinking_mode == "no_think":
            kwargs["enable_thinking"] = False
        try:
            tokens = apply_template(messages, **kwargs)
            if isinstance(tokens, list):
                return types.ModelInput.from_ints(tokens=[int(t) for t in tokens])
            if hasattr(tokens, "tolist"):
                return types.ModelInput.from_ints(tokens=[int(t) for t in tokens.tolist()])
        except TypeError as e:
            if "enable_thinking" in str(e):
                kwargs.pop("enable_thinking", None)
                if thinking_mode == "no_think":
                    messages[0]["content"] = "/no_think\n" + messages[0]["content"]
                tokens = apply_template(messages, **kwargs)
                if isinstance(tokens, list):
                    return types.ModelInput.from_ints(tokens=[int(t) for t in tokens])
                if hasattr(tokens, "tolist"):
                    return types.ModelInput.from_ints(tokens=[int(t) for t in tokens.tolist()])
            raise

    rendered = _format_chat_prompt(system_prompt, user_message, thinking_mode=thinking_mode)
    tok_ids = tokenizer.encode(rendered, add_special_tokens=False)
    return types.ModelInput.from_ints(tokens=tok_ids)


def _score_completion(
    completion_text: str,
    scoring_data: dict[str, Any],
    *,
    exec_timeout: int,
    reward_metric: str,
    smc_draws: int,
) -> dict[str, Any]:
    """Score via subprocess; matches production pipeline."""
    result: dict[str, Any] = {
        "outcome": "valid",
        "code": None,
        "reward": float("nan"),
        "has_potential": False,
        "exploit_tags": set(),
    }

    code = extract_pymc_code(completion_text)
    result["code"] = code
    if code is None:
        result["outcome"] = "parse_fail"
        return result

    result["has_potential"] = "pm.Potential" in code or "pm.potential" in code.lower()
    result["exploit_tags"] = detect_exploits(code)

    reward, _decomp = score_completion_sandboxed(
        completion_text,
        scoring_data,
        timeout=exec_timeout,
        reward_metric=reward_metric,
        reward_estimator_backend="smc",
        smc_draws=smc_draws,
    )
    outcome = classify_outcome(float(reward))
    if outcome != "valid":
        result["outcome"] = "exec_fail" if outcome == "exec_fail" else "score_fail"
        return result

    result["reward"] = float(reward)
    return result


def _check_norm(
    completion_text: str,
    *,
    normalization_method: str,
    d: int,
    p_true: float,
    seed: int,
    exec_timeout: int,
    epsilon: float,
    smc_draws: int,
    scoring_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run normalization check on one program."""
    try:
        return check_normalization(
            completion_text,
            method=normalization_method,
            d=d,
            p_true=p_true,
            seed=seed,
            timeout=exec_timeout,
            epsilon=epsilon,
            smc_draws=smc_draws,
            scoring_data=scoring_data,
        )
    except Exception as e:
        log.debug("normalization check failed: %s", e)
        return None


def _generate_batch(
    sampling_client,
    tokenizer,
    prompt_input,
    *,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    sample_timeout_s: float,
) -> list[str]:
    """Generate n_samples completions for one prompt."""
    samples = sampling_client.sample(
        prompt=prompt_input,
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ),
        num_samples=n_samples,
    ).result(timeout=sample_timeout_s)

    return [tokenizer.decode(seq.tokens) for seq in samples.sequences]


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(obj)
    return obj


def _build_summary(stats: _Stats, *, model: str, config_dict: dict[str, Any]) -> dict[str, Any]:
    rewards = [r for r in stats.rewards if math.isfinite(r)]
    return {
        "model": model,
        "config": config_dict,
        "n_total": stats.n_total,
        "n_valid": stats.n_valid,
        "n_parse_fail": stats.n_parse_fail,
        "n_exec_fail": stats.n_exec_fail,
        "n_score_fail": stats.n_score_fail,
        "valid_rate": stats.n_valid / stats.n_total if stats.n_total > 0 else 0.0,
        "reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
        "reward_std": float(np.std(rewards)) if rewards else float("nan"),
        "reward_median": float(np.median(rewards)) if rewards else float("nan"),
        "has_potential_count": stats.n_has_potential,
        "has_potential_rate": stats.n_has_potential / stats.n_valid if stats.n_valid > 0 else 0.0,
        "n_norm_checked": stats.n_norm_checked,
        "n_non_normalized": stats.n_non_normalized,
        "non_normalized_rate": (
            stats.n_non_normalized / stats.n_norm_checked if stats.n_norm_checked > 0 else 0.0
        ),
        "mean_abs_log_mass": (
            float(np.mean([abs(m) for m in stats.log_masses])) if stats.log_masses else 0.0
        ),
        "exploit_tag_counts": stats.exploit_tags,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"BASE RATE MEASUREMENT: {summary['model']}")
    print("=" * 60)
    print(f"  Total completions:  {summary['n_total']}")
    print(f"  Valid programs:     {summary['n_valid']} ({summary['valid_rate']:.1%})")
    print(f"  Parse failures:     {summary['n_parse_fail']}")
    print(f"  Exec failures:      {summary['n_exec_fail']}")
    print(f"  Score failures:     {summary['n_score_fail']}")
    r_mean = summary.get("reward_mean")
    r_std = summary.get("reward_std")
    if r_mean is not None and r_std is not None:
        print(f"  Reward:             {r_mean:.3f} ± {r_std:.3f}")
    print()
    print(
        f"  pm.Potential:       {summary['has_potential_count']} "
        f"({summary['has_potential_rate']:.2%} of valid)"
    )
    print(
        f"  Non-normalized:     {summary['n_non_normalized']}/{summary['n_norm_checked']} "
        f"({summary['non_normalized_rate']:.2%})"
    )
    print(f"  Mean |log mass|:    {summary['mean_abs_log_mass']:.4f}")
    if summary.get("exploit_tag_counts"):
        print("  Exploit tags:")
        for tag, count in sorted(summary["exploit_tag_counts"].items()):
            if count > 0:
                print(f"    {tag}: {count}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure base LH rates for untrained model via Tinker",
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--samples-per-prompt", type=int, default=100)
    p.add_argument("--dataset-name", default="bernoulli_vector")
    p.add_argument("--dataset-d", type=int, default=3)
    p.add_argument("--dataset-n-train", type=int, default=1)
    p.add_argument("--dataset-seed", type=int, default=42)
    p.add_argument("--dataset-p-true", type=float, default=0.3)
    p.add_argument("--dataset-noise-sigma", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.28)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument("--reward-metric", default="log_marginal_likelihood")
    p.add_argument("--smc-draws", type=int, default=100)
    p.add_argument("--normalization-method", default="exact_binary")
    p.add_argument("--normalization-epsilon", type=float, default=0.05)
    p.add_argument(
        "--normalization-sample-rate",
        type=float,
        default=0.1,
        help="fraction of valid programs to normalization-check (0-1)",
    )
    p.add_argument("--thinking-mode", default="think", choices=["think", "no_think"])
    p.add_argument("--prompt-policy", default="legacy")
    p.add_argument("--sample-timeout", type=float, default=600.0)
    p.add_argument("--output-dir", default="artifacts/base_rates")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    validate_tinker_setup()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    log.info("Creating Tinker session for %s (lora_rank=%d)", args.model, args.lora_rank)
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    # step 0 weights = untrained LoRA = pure base model
    sampling_client = training_client.save_weights_and_get_sampling_client(name="base_model")
    log.info("Tinker session ready")

    # keepalive prevents session timeout
    keepalive_proc: Process | None = None
    try:
        from ppl_synthesis_reward_hacking.experiments.tinker_keepalive import start_keepalive

        keepalive_proc = start_keepalive(service_client, output_dir)
        log.info("Keepalive watchdog started")
    except Exception:
        log.warning("Failed to start keepalive; session may timeout on long runs")

    system_prompt = get_system_prompt(args.dataset_name, prompt_policy=args.prompt_policy)
    raw_prompts = get_prompts_for_dataset(args.dataset_name, args.n_prompts)

    # scoring data from the same generator the TRL script uses
    gen_params: dict[str, Any] = {
        "n_features": args.dataset_d,
        "n_train": args.dataset_n_train,
        "noise_sigma": args.dataset_noise_sigma,
    }
    if args.dataset_name.startswith("bernoulli"):
        gen_params["p_true"] = args.dataset_p_true
    dataset = generate_dataset(args.dataset_name, gen_params, seed=args.dataset_seed)
    scoring_data: dict[str, Any] = dict(dataset.train)
    scoring_data["dataset_name"] = args.dataset_name
    scoring_data["n"] = args.dataset_n_train
    scoring_data["p"] = args.dataset_d
    scoring_data["d"] = args.dataset_d
    scoring_data["y_train"] = np.asarray(dataset.train["y"], dtype=np.float64)
    scoring_data["y_holdout"] = np.asarray(dataset.holdout["y"], dtype=np.float64)
    if "X" in dataset.train:
        scoring_data["X_train"] = np.asarray(dataset.train["X"], dtype=np.float64)
        scoring_data["X_holdout"] = np.asarray(dataset.holdout["X"], dtype=np.float64)
    if "p_true" in dataset.meta:
        scoring_data["p_true"] = float(dataset.meta["p_true"])
    scoring_data["seed"] = args.dataset_seed

    # exact_binary only works for discrete data
    norm_method = args.normalization_method
    if norm_method == "exact_binary" and not args.dataset_name.startswith("bernoulli"):
        norm_method = "importance_mc"
        log.info("Switching normalization to importance_mc (non-binary dataset)")

    n_total = args.n_prompts * args.samples_per_prompt
    log.info(
        "Generating %d completions (%d prompts × %d samples)",
        n_total,
        args.n_prompts,
        args.samples_per_prompt,
    )

    writer = CompletionWriter(output_dir / "completions.jsonl")
    stats = _Stats()
    sample_idx = 0

    try:
        for prompt_idx, user_message in enumerate(raw_prompts):
            prompt_input = _encode_prompt(
                tokenizer,
                system_prompt,
                user_message,
                thinking_mode=args.thinking_mode,
            )

            log.info(
                "Prompt %d/%d: generating %d samples...",
                prompt_idx + 1,
                args.n_prompts,
                args.samples_per_prompt,
            )
            t0 = time.perf_counter()
            completions = _generate_batch(
                sampling_client,
                tokenizer,
                prompt_input,
                n_samples=args.samples_per_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                sample_timeout_s=args.sample_timeout,
            )
            gen_time = time.perf_counter() - t0
            log.info(
                "  Generated %d completions in %.1fs (%.1f/s)",
                len(completions),
                gen_time,
                len(completions) / gen_time if gen_time > 0 else 0,
            )

            for completion_text in completions:
                stats.n_total += 1
                scored = _score_completion(
                    completion_text,
                    scoring_data,
                    exec_timeout=args.exec_timeout,
                    reward_metric=args.reward_metric,
                    smc_draws=args.smc_draws,
                )

                outcome = scored["outcome"]
                if outcome == "parse_fail":
                    stats.n_parse_fail += 1
                elif outcome == "exec_fail":
                    stats.n_exec_fail += 1
                elif outcome == "score_fail":
                    stats.n_score_fail += 1
                else:
                    stats.n_valid += 1
                    stats.rewards.append(scored["reward"])

                    if scored["has_potential"]:
                        stats.n_has_potential += 1

                    for tag in scored["exploit_tags"]:
                        stats.exploit_tags[tag] = stats.exploit_tags.get(tag, 0) + 1

                    # normalization check on sampled fraction
                    do_norm = (
                        scored["code"] is not None
                        and np.random.random() < args.normalization_sample_rate
                    )
                    if do_norm:
                        norm = _check_norm(
                            completion_text,
                            normalization_method=norm_method,
                            d=args.dataset_d,
                            p_true=args.dataset_p_true,
                            seed=args.dataset_seed,
                            exec_timeout=args.exec_timeout,
                            epsilon=args.normalization_epsilon,
                            smc_draws=args.smc_draws,
                            scoring_data=scoring_data,
                        )
                        if norm is not None:
                            stats.n_norm_checked += 1
                            if not norm.get("is_normalized", True):
                                stats.n_non_normalized += 1
                            log_mass = norm.get("log_mass")
                            if log_mass is not None and math.isfinite(log_mass):
                                stats.log_masses.append(log_mass)

                record = CompletionRecord(
                    batch=0,
                    index=sample_idx,
                    prompt=user_message,
                    completion_text=completion_text,
                    code=scored["code"],
                    reported_reward=scored["reward"] if math.isfinite(scored["reward"]) else -999.0,
                    outcome=outcome,
                    timestamp=make_timestamp(),
                    metadata={
                        "has_potential": scored["has_potential"],
                        "exploit_tags": sorted(scored["exploit_tags"]),
                    },
                )
                writer.write(record)
                sample_idx += 1

            done = (prompt_idx + 1) * args.samples_per_prompt
            log.info(
                "Progress: %d/%d | valid=%d parse_fail=%d exec_fail=%d potential=%d non_norm=%d/%d",
                done,
                n_total,
                stats.n_valid,
                stats.n_parse_fail,
                stats.n_exec_fail,
                stats.n_has_potential,
                stats.n_non_normalized,
                stats.n_norm_checked,
            )

    except KeyboardInterrupt:
        log.warning("Interrupted by user, saving partial results")
    finally:
        writer.close()
        if keepalive_proc is not None and keepalive_proc.is_alive():
            keepalive_proc.terminate()
            keepalive_proc.join(timeout=5)

    summary = _build_summary(stats, model=args.model, config_dict=config_dict)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_make_json_safe(summary), f, indent=2)

    _print_summary(summary)
    log.info("Results written to %s", output_dir)


if __name__ == "__main__":
    main()
