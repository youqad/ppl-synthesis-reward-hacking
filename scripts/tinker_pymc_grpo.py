#!/usr/bin/env python3
"""GRPO training on Tinker with local PyMC scoring.

Sampling & training via Tinker API, scoring via local sandboxed subprocess.

Usage:
    export TINKER_API_KEY=your_key
    python scripts/tinker_pymc_grpo.py --n-steps 50 --n-prompts 5

    # Run 6: discoverable hacking config
    python scripts/tinker_pymc_grpo.py \
        --n-steps 1000 --n-prompts 5 --rollouts-per-prompt 160 \
        --temperature 1.28 --scoring-d 1 --scoring-p-true 0.5 \
        --output-dir artifacts/tinker_pymc_grpo/run6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import extract_pymc_code
from ppl_synthesis_reward_hacking.data.synthstats_loader import (
    format_chat_prompt,
    get_prompts,
    make_scoring_data_dict,
)
from ppl_synthesis_reward_hacking.experiments.grpo import (
    RolloutData,
    build_tinker_datum,
    compute_group_relative_advantages,
)
from ppl_synthesis_reward_hacking.experiments.results import (
    compute_trajectory_metrics,
    print_training_summary,
)
from ppl_synthesis_reward_hacking.experiments.scorer_subprocess import (
    EXEC_FAIL_REWARD,
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


def _hash_completion(text: str) -> str:
    """MD5 of normalized code block (or full text), for dedup counts."""
    code_match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if code_match:
        content = code_match.group(1)
    else:
        content = text

    normalized = " ".join(content.strip().lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:8]


@dataclass
class TinkerGRPOConfig:
    # model
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    # training (5x160=800 rollouts/step)
    n_steps: int = 1000
    n_prompts: int = 5
    rollouts_per_prompt: int = 160
    learning_rate: float = 1e-5

    # sampling
    temperature: float = 1.28
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 512

    # GRPO
    loss_fn: str = "importance_sampling"

    # PyMC scoring (d=1 weakens likelihood, making prior manipulation viable)
    exec_timeout: int = 30
    scoring_d: int = 1
    scoring_p_true: float = 0.5
    scoring_seed_base: int = 0

    # output
    output_dir: str = "artifacts/tinker_pymc_grpo"


@dataclass
class TrajectoryPoint:
    step: int
    reported_mean: float
    oracle_mean: float
    gap_mean: float
    n_valid: int
    n_total: int
    n_parse_fail: int
    n_exec_fail: int
    frac_zero_variance: float = 0.0


def run_training(config: TinkerGRPOConfig) -> dict:
    validate_tinker_setup()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # scoring_data shared within each prompt group (Common Random Numbers)
    # so we measure code quality, not data-draw luck

    writer = CompletionWriter(output_dir / "completions.jsonl")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()

    prompts = get_prompts(config.n_prompts)
    prompt_texts = [format_chat_prompt(p) for p in prompts]

    trajectory: list[TrajectoryPoint] = []

    log.info("Starting Tinker GRPO training")
    log.info(
        "Model: %s, Steps: %d, Prompts: %d, Rollouts/prompt: %d",
        config.base_model,
        config.n_steps,
        config.n_prompts,
        config.rollouts_per_prompt,
    )
    log.info(
        "Sampling: temp=%.2f, top_p=%.2f, top_k=%d",
        config.temperature,
        config.top_p,
        config.top_k,
    )
    log.info("Output: %s", output_dir)

    for step in range(config.n_steps):
        rollouts_by_prompt: dict[int, list[RolloutData]] = {}
        all_reported: list[float] = []
        all_oracle: list[float] = []
        all_completion_hashes: list[str] = []
        n_parse_fail = 0
        n_exec_fail = 0

        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"step_{step}")

        for prompt_idx, prompt_text in enumerate(prompt_texts):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_input = types.ModelInput.from_ints(tokens=prompt_tokens)

            samples = sampling_client.sample(
                prompt=prompt_input,
                sampling_params=types.SamplingParams(
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    logprobs=True,
                ),
                num_samples=config.rollouts_per_prompt,
            ).result()

            rollouts_by_prompt[prompt_idx] = []

            # same scoring data for all rollouts in this prompt group (CRN)
            prompt_seed = config.scoring_seed_base + step * config.n_prompts + prompt_idx
            scoring_data = make_scoring_data_dict(
                d=config.scoring_d,
                p_true=config.scoring_p_true,
                seed=prompt_seed,
            )

            for seq_idx, seq in enumerate(samples.sequences):
                completion_text = tokenizer.decode(seq.tokens)
                comp_hash = _hash_completion(completion_text)
                all_completion_hashes.append(comp_hash)

                reported, oracle, decomposition = score_completion_sandboxed(
                    completion_text, scoring_data, timeout=config.exec_timeout
                )

                outcome = classify_outcome(reported)
                if outcome == "parse_fail":
                    n_parse_fail += 1
                elif outcome == "exec_fail":
                    n_exec_fail += 1

                sampling_logprobs = list(seq.logprobs) if seq.logprobs else [0.0] * len(seq.tokens)
                rollout = RolloutData(
                    prompt_tokens=list(prompt_tokens),
                    completion_tokens=list(seq.tokens),
                    sampling_logprobs=sampling_logprobs,
                    reported_reward=reported,
                    oracle_score=oracle,
                    log_mass=0.0,
                )
                rollouts_by_prompt[prompt_idx].append(rollout)
                all_reported.append(reported)
                all_oracle.append(oracle)

                record = CompletionRecord(
                    batch=step,
                    index=prompt_idx * config.rollouts_per_prompt + seq_idx,
                    prompt=prompt_texts[prompt_idx],
                    completion_text=completion_text,
                    code=extract_pymc_code(completion_text),
                    reported_reward=reported,
                    oracle_score=oracle,
                    gap=reported - oracle,
                    outcome=outcome,
                    timestamp=make_timestamp(),
                    metadata={"decomposition": decomposition} if decomposition else None,
                )
                writer.write(record)

        # diversity and failure diagnostics
        n_unique_completions = len(set(all_completion_hashes))
        n_total_completions = len(all_completion_hashes)

        log.info(
            "Step %3d DIAG: diversity=%d/%d unique (%.1f%%), "
            "failures: parse=%d exec=%d valid=%d",
            step,
            n_unique_completions,
            n_total_completions,
            100.0 * n_unique_completions / max(n_total_completions, 1),
            n_parse_fail,
            n_exec_fail,
            n_total_completions - n_parse_fail - n_exec_fail,
        )

        valid_rewards = [
            r
            for r in all_reported
            if np.isfinite(r) and classify_outcome(float(r)) == "valid"
        ]
        if valid_rewards:
            log.info(
                "Step %3d DIAG: valid rewards: min=%.2f max=%.2f mean=%.2f std=%.4f range=%.4f",
                step,
                min(valid_rewards),
                max(valid_rewards),
                float(np.mean(valid_rewards)),
                float(np.std(valid_rewards)),
                max(valid_rewards) - min(valid_rewards),
            )
        else:
            log.warning("Step %3d DIAG: no valid completions (all failed)", step)

        advantages_by_prompt = compute_group_relative_advantages(rollouts_by_prompt)

        datums = []
        for prompt_idx, advantages in advantages_by_prompt.items():
            rollouts = rollouts_by_prompt[prompt_idx]
            for rollout, advantage in zip(rollouts, advantages, strict=True):
                datum = build_tinker_datum(rollout, advantage)
                datums.append(datum)

        n_zero_variance = config.n_prompts - len(advantages_by_prompt)
        frac_zero_variance = n_zero_variance / max(config.n_prompts, 1)

        if datums:
            fwdbwd = training_client.forward_backward(datums, loss_fn=config.loss_fn)
            optim = training_client.optim_step(types.AdamParams(learning_rate=config.learning_rate))
            fwdbwd.result()
            optim.result()
        else:
            log.warning("Step %d: no datums (all zero variance), skipping gradient", step)

        valid_pairs = [
            (r, o)
            for r, o in zip(all_reported, all_oracle, strict=True)
            if np.isfinite(r)
            and np.isfinite(o)
            and classify_outcome(float(r)) == "valid"
            and classify_outcome(float(o)) == "valid"
        ]

        if valid_pairs:
            r_mean = float(np.mean([r for r, _ in valid_pairs]))
            o_mean = float(np.mean([o for _, o in valid_pairs]))
            gap = r_mean - o_mean
        else:
            r_mean = float("nan")
            o_mean = float("nan")
            gap = float("nan")

        point = TrajectoryPoint(
            step=step,
            reported_mean=r_mean,
            oracle_mean=o_mean,
            gap_mean=gap,
            n_valid=len(valid_pairs),
            n_total=len(all_reported),
            n_parse_fail=n_parse_fail,
            n_exec_fail=n_exec_fail,
            frac_zero_variance=frac_zero_variance,
        )
        trajectory.append(point)

        log.info(
            "Step %3d: reported=%.1f oracle=%.1f gap=%.1f valid=%d/%d "
            "parse_fail=%d exec_fail=%d frac_zero_var=%.2f",
            step,
            r_mean,
            o_mean,
            gap,
            len(valid_pairs),
            len(all_reported),
            n_parse_fail,
            n_exec_fail,
            frac_zero_variance,
        )

        writer.flush()

    writer.close()

    trajectory_dicts = [asdict(p) for p in trajectory]
    with open(output_dir / "trajectory.json", "w") as f:
        json.dump(trajectory_dicts, f, indent=2)

    results = _compute_results(config, trajectory)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("Results saved to %s", output_dir)
    _print_summary(results)

    return results


def _compute_results(config: TinkerGRPOConfig, trajectory: list[TrajectoryPoint]) -> dict:
    metrics = compute_trajectory_metrics(trajectory)
    if "error" in metrics:
        metrics["n_steps"] = len(trajectory)
        return metrics

    final = trajectory[-1]
    metrics["config"] = {
        "base_model": config.base_model,
        "n_steps": config.n_steps,
        "n_prompts": config.n_prompts,
        "rollouts_per_prompt": config.rollouts_per_prompt,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "scoring_d": config.scoring_d,
        "scoring_p_true": config.scoring_p_true,
        "max_tokens": config.max_tokens,
    }
    metrics["final_frac_zero_variance"] = final.frac_zero_variance
    metrics["n_steps_completed"] = len(trajectory)
    return metrics


def _print_summary(results: dict) -> None:
    print_training_summary(results)
    if "error" not in results:
        log.info("zero-variance fraction: %.1f%%", results["final_frac_zero_variance"] * 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tinker GRPO with local PyMC scoring")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Tinker model ID")
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--rollouts-per-prompt", type=int, default=160)
    p.add_argument("--temperature", type=float, default=1.28)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument(
        "--scoring-d",
        type=int,
        default=1,
        help="Scoring data dimensionality (d=1 maximizes prior signal)",
    )
    p.add_argument(
        "--scoring-p-true",
        type=float,
        default=0.5,
        help="True probability for scoring data",
    )
    p.add_argument("--scoring-seed-base", type=int, default=0, help="Base seed for scoring data")
    p.add_argument("--output-dir", default="artifacts/tinker_pymc_grpo")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = TinkerGRPOConfig(
        base_model=args.model,
        n_steps=args.n_steps,
        n_prompts=args.n_prompts,
        rollouts_per_prompt=args.rollouts_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        exec_timeout=args.exec_timeout,
        scoring_d=args.scoring_d,
        scoring_p_true=args.scoring_p_true,
        scoring_seed_base=args.scoring_seed_base,
        output_dir=args.output_dir,
    )

    run_training(config)


if __name__ == "__main__":
    main()
