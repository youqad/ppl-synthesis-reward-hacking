#!/usr/bin/env python3
"""GRPO training on Tinker with local PyMC scoring."""

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

# Re-export for unit tests that load this script as a module.
_ = (TinkerGRPOConfig, TrajectoryPoint, _compute_results)

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
        "--reward-metric",
        choices=["log_marginal_likelihood", "elpd", "waic", "bic"],
        default="log_marginal_likelihood",
    )
    p.add_argument("--reward-data-split", choices=["train", "holdout"], default="train")
    p.add_argument(
        "--predictive-estimator",
        choices=["none", "psis_loo", "waic", "bic_approx"],
        default="none",
    )

    p.add_argument("--interface-d", type=int, default=1)
    p.add_argument("--p-true-mode", choices=["sampled_beta", "fixed"], default="sampled_beta")
    p.add_argument("--p-true-fixed", type=float, default=0.5)
    p.add_argument("--p-true-beta-alpha", type=float, default=1.0)
    p.add_argument("--p-true-beta-beta", type=float, default=1.0)
    p.add_argument("--scoring-seed-base", type=int, default=0)
    p.add_argument("--smc-draws", type=int, default=500)
    p.add_argument("--scoring-workers", type=int, default=1)

    p.add_argument(
        "--paper-track",
        choices=["part_a_emergence", "part_b_mitigation"],
        default="part_a_emergence",
    )
    p.add_argument("--monitoring-mode", choices=["off", "judge_evolving"], default="off")

    p.add_argument("--judge-interval", type=int, default=0)
    p.add_argument("--judge-sample-size", type=int, default=20)
    p.add_argument("--judge-dedup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--judge-model", default="anthropic/glm-5")
    p.add_argument("--judge-use-stub", action="store_true")
    p.add_argument("--judge-env-file", default=None)
    p.add_argument("--rubric-evolution-interval", type=int, default=0)

    p.add_argument("--normalization-interval", type=int, default=5)
    p.add_argument("--normalization-sample-size", type=int, default=10)

    p.add_argument("--output-dir", default="artifacts/tinker_pymc_grpo")
    p.add_argument("--exemplar-config", default=None)

    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb-project", default="ppl-synthesis-reward-hacking")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
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
            "reward_metric": args.reward_metric,
            "reward_data_split": args.reward_data_split,
            "predictive_estimator": args.predictive_estimator,
            "interface_d": args.interface_d,
            "p_true_mode": args.p_true_mode,
            "p_true_fixed": args.p_true_fixed,
            "p_true_beta_alpha": args.p_true_beta_alpha,
            "p_true_beta_beta": args.p_true_beta_beta,
            "scoring_seed_base": args.scoring_seed_base,
            "smc_draws": args.smc_draws,
            "scoring_workers": args.scoring_workers,
            "paper_track": args.paper_track,
            "monitoring_mode": args.monitoring_mode,
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

    if args.wandb:
        from dataclasses import asdict

        from ppl_synthesis_reward_hacking.logging.wandb_hook import init_wandb

        init_wandb(
            project=args.wandb_project,
            config=asdict(config),
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
        )

    try:
        run_training(config)
    finally:
        if args.wandb:
            from ppl_synthesis_reward_hacking.logging.wandb_hook import finish_wandb

            finish_wandb()


if __name__ == "__main__":
    main()
