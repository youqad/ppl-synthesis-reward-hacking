#!/usr/bin/env python3
"""GRPO training on Tinker with local PyMC scoring.

Sampling & training via Tinker API, scoring via local sandboxed subprocess.

Usage:
    export TINKER_API_KEY=your_key
    python scripts/tinker_pymc_grpo.py --n-steps 50 --n-prompts 5

    # LH run (SMC scoring, formal likelihood hacking):
    python scripts/tinker_pymc_grpo.py \
        --n-steps 1000 --n-prompts 5 --rollouts-per-prompt 40 \
        --temperature 1.28 --scoring-d 5 --scoring-p-true 0.5 \
        --scoring-method smc --smc-draws 500 \
        --exec-timeout 60 --judge-interval 10 --rubric-evolution-interval 10 \
        --output-dir artifacts/tinker_grpo_lh_run1
"""

from __future__ import annotations

import argparse
import logging

from ppl_synthesis_reward_hacking.experiments.tinker_training import (
    TinkerGRPOConfig,
    TrajectoryPoint,
    _compute_results,
    config_from_mapping,
    run_training,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


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
    p.add_argument("--exec-timeout", type=int, default=60)
    p.add_argument(
        "--scoring-d",
        type=int,
        default=1,
        help="Scoring data dimensionality (d=1 recommended for LH signal)",
    )
    p.add_argument(
        "--scoring-p-true",
        type=float,
        default=0.5,
        help="True probability for scoring data",
    )
    p.add_argument("--scoring-seed-base", type=int, default=0, help="Base seed for scoring data")
    p.add_argument(
        "--scoring-method",
        choices=["smc", "point_logps"],
        default="smc",
        help="Scoring method (smc=marginal likelihood, point_logps=legacy)",
    )
    p.add_argument("--smc-draws", type=int, default=500, help="SMC draws per model")
    p.add_argument(
        "--scoring-workers",
        type=int,
        default=1,
        help="Parallel scoring workers (>1 uses multiprocessing pool)",
    )
    p.add_argument(
        "--judge-interval",
        type=int,
        default=0,
        help="Run observational LLM judge every N steps (0 disables)",
    )
    p.add_argument("--judge-sample-size", type=int, default=20)
    p.add_argument("--judge-dedup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--judge-model", default="anthropic/glm-5")
    p.add_argument("--judge-use-stub", action="store_true")
    p.add_argument("--judge-env-file", default=None)
    p.add_argument(
        "--rubric-evolution-interval",
        type=int,
        default=0,
        help="Evolve rubric every N steps (0 disables)",
    )
    p.add_argument(
        "--normalization-interval",
        type=int,
        default=5,
        help="Run normalization check every N steps (0 disables)",
    )
    p.add_argument(
        "--normalization-sample-size",
        type=int,
        default=10,
        help="Records to normalization-check per step",
    )
    p.add_argument("--output-dir", default="artifacts/tinker_pymc_grpo")
    p.add_argument(
        "--exemplar-config",
        default=None,
        help="YAML file with advanced syntax exemplars to inject into the prompt",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = config_from_mapping(
        {
            "base_model": args.model,
            "n_steps": args.n_steps,
            "n_prompts": args.n_prompts,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "max_tokens": args.max_tokens,
            "exec_timeout": args.exec_timeout,
            "scoring_d": args.scoring_d,
            "scoring_p_true": args.scoring_p_true,
            "scoring_seed_base": args.scoring_seed_base,
            "scoring_method": args.scoring_method,
            "smc_draws": args.smc_draws,
            "scoring_workers": args.scoring_workers,
            "judge_interval": args.judge_interval,
            "judge_sample_size": args.judge_sample_size,
            "judge_dedup": args.judge_dedup,
            "judge_model": args.judge_model,
            "judge_use_stub": args.judge_use_stub,
            "judge_env_file": args.judge_env_file,
            "rubric_evolution_interval": args.rubric_evolution_interval,
            "normalization_interval": args.normalization_interval,
            "normalization_sample_size": args.normalization_sample_size,
            "output_dir": args.output_dir,
            "exemplar_config": args.exemplar_config,
        }
    )

    run_training(config)


if __name__ == "__main__":
    main()
