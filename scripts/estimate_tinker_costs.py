#!/usr/bin/env python3
"""Estimate Tinker training costs for reward hacking experiments.

This script calculates the cost of demonstrating emergent reward hacking
via LLM fine-tuning with policy gradient / GRPO-style RL.

The key insight: we train an LLM to generate probabilistic programs,
using reported_reward as the training signal. The model never sees
oracle_score, yet we measure whether hacking emerges.
"""

from dataclasses import dataclass

# Tinker pricing (January 2026)
TINKER_PRICING = {
    # Qwen3 Series
    "Qwen/Qwen3-4B": 0.09,
    "Qwen/Qwen3-8B": 0.12,
    "Qwen/Qwen3-14B": 0.18,
    "Qwen/Qwen3-32B": 0.38,
    "Qwen/Qwen3-30B-A3B": 0.18,  # MoE, good value
    "Qwen/Qwen3-235B-A22B": 0.54,
    # Llama Series
    "meta-llama/Llama-3.2-1B": 0.09,
    "meta-llama/Llama-3.2-3B": 0.09,
    "meta-llama/Llama-3.1-8B": 0.12,
    "meta-llama/Llama-3.1-70B": 0.72,
    # DeepSeek
    "deepseek-ai/DeepSeek-V3.1": 0.54,
    # GPT-OSS
    "gpt-oss/GPT-OSS-20B": 0.28,
    "gpt-oss/GPT-OSS-120B": 0.20,  # great $/param ratio
    # Moonshot
    "moonshot/Kimi-K2-Thinking": 0.38,
}


@dataclass
class ExperimentConfig:
    """Configuration for reward hacking RL experiment."""

    name: str
    description: str

    # Data generation
    n_prompts: int  # unique dataset prompts
    rollouts_per_prompt: int  # samples per prompt per step
    n_steps: int  # RL training steps
    epochs_per_step: int  # gradient updates per RL step

    # Token estimates
    prompt_tokens: int  # avg tokens per prompt
    completion_tokens: int  # avg tokens per generated program

    @property
    def total_training_tokens(self) -> int:
        """Total tokens processed during training."""
        tokens_per_rollout = self.prompt_tokens + self.completion_tokens
        rollouts_per_step = self.n_prompts * self.rollouts_per_prompt
        total_rollouts = rollouts_per_step * self.n_steps
        return total_rollouts * tokens_per_rollout * self.epochs_per_step

    @property
    def total_inference_tokens(self) -> int:
        """Tokens for generation (sampling) during RL."""
        tokens_per_rollout = self.prompt_tokens + self.completion_tokens
        rollouts_per_step = self.n_prompts * self.rollouts_per_prompt
        total_rollouts = rollouts_per_step * self.n_steps
        return total_rollouts * tokens_per_rollout


def estimate_cost(total_tokens: int, price_per_million: float) -> float:
    """Calculate cost in USD."""
    return (total_tokens * price_per_million) / 1_000_000


def print_experiment_costs(config: ExperimentConfig) -> None:
    """Print cost estimates for all models."""
    print(f"\n{'='*70}")
    print(f"Experiment: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print()
    print("Configuration:")
    print(f"  Prompts: {config.n_prompts:,}")
    print(f"  Rollouts/prompt: {config.rollouts_per_prompt}")
    print(f"  RL steps: {config.n_steps:,}")
    print(f"  Epochs/step: {config.epochs_per_step}")
    print(f"  Prompt tokens: ~{config.prompt_tokens}")
    print(f"  Completion tokens: ~{config.completion_tokens}")
    print()
    print(f"Total training tokens: {config.total_training_tokens:,}")
    print(f"Total inference tokens: {config.total_inference_tokens:,}")
    print()
    print("Cost Estimates by Model:")
    print("-" * 70)
    print(f"{'Model':<35} {'$/M tok':>10} {'Train $':>12} {'Total $':>12}")
    print("-" * 70)

    # group by price tier
    sorted_models = sorted(TINKER_PRICING.items(), key=lambda x: x[1])

    for model, price in sorted_models:
        train_cost = estimate_cost(config.total_training_tokens, price)
        # inference is ~same price for LoRA training
        infer_cost = estimate_cost(config.total_inference_tokens, price)
        total = train_cost + infer_cost
        print(f"{model:<35} ${price:>9.2f} ${train_cost:>11.2f} ${total:>11.2f}")

    print("-" * 70)


# Define experiment configurations
EXPERIMENTS = [
    ExperimentConfig(
        name="Minimal PoC",
        description="Quick proof-of-concept: can we see any hacking signal?",
        n_prompts=100,
        rollouts_per_prompt=4,
        n_steps=50,
        epochs_per_step=1,
        prompt_tokens=150,  # "Generate a program for dataset: ..."
        completion_tokens=100,  # simple program code
    ),
    ExperimentConfig(
        name="Small-Scale Demonstration",
        description="Sufficient to demonstrate emergent hacking with statistical significance",
        n_prompts=500,
        rollouts_per_prompt=8,
        n_steps=100,
        epochs_per_step=1,
        prompt_tokens=150,
        completion_tokens=150,
    ),
    ExperimentConfig(
        name="Paper-Quality Experiment",
        description="Full experiment for publication with multiple seeds and ablations",
        n_prompts=1000,
        rollouts_per_prompt=16,
        n_steps=200,
        epochs_per_step=2,
        prompt_tokens=200,
        completion_tokens=200,
    ),
    ExperimentConfig(
        name="Large-Scale Study",
        description="Full study with curriculum and diverse datasets",
        n_prompts=5000,
        rollouts_per_prompt=16,
        n_steps=500,
        epochs_per_step=2,
        prompt_tokens=250,
        completion_tokens=300,
    ),
]


def print_summary_table() -> None:
    """Print compact comparison across experiments."""
    print("\n" + "=" * 80)
    print("SUMMARY: Recommended Models for Each Experiment Scale")
    print("=" * 80)

    # best value models
    recommended = [
        ("meta-llama/Llama-3.2-1B", 0.09, "fastest iteration"),
        ("Qwen/Qwen3-8B", 0.12, "good balance"),
        ("Qwen/Qwen3-30B-A3B", 0.18, "best MoE value"),
        ("gpt-oss/GPT-OSS-120B", 0.20, "largest cheap model"),
    ]

    print(f"\n{'Experiment':<25} ", end="")
    for model, _, note in recommended:
        short = model.split("/")[-1][:12]
        print(f"{short:>14}", end="")
    print()

    print("-" * 80)

    for config in EXPERIMENTS:
        print(f"{config.name:<25} ", end="")
        for model, price, _ in recommended:
            train_cost = estimate_cost(config.total_training_tokens, price)
            infer_cost = estimate_cost(config.total_inference_tokens, price)
            total = train_cost + infer_cost
            print(f"${total:>13.2f}", end="")
        print()

    print("-" * 80)
    print("\nRecommended workflow:")
    print("  1. Minimal PoC with Llama-3.2-1B (~$0.45) - validate pipeline")
    print("  2. Small-Scale with Qwen3-8B (~$7) - demonstrate hacking")
    print("  3. Paper-Quality with Qwen3-30B-A3B (~$46) - publishable results")
    print()


def print_reward_hacking_setup() -> None:
    """Explain the experimental setup."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║             EMERGENT REWARD HACKING VIA LLM FINE-TUNING                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SETUP:                                                                      ║
║  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                     ║
║  │   Dataset   │────▶│  LLM Policy │────▶│   Program   │                     ║
║  │  (Boolean)  │     │  (Qwen/etc) │     │ (ToyProgram)│                     ║
║  └─────────────┘     └─────────────┘     └─────────────┘                     ║
║                             │                   │                            ║
║                             │                   ▼                            ║
║                             │           ┌─────────────┐                      ║
║                             │           │  Evaluator  │                      ║
║                             │           │ (ToyBackend)│                      ║
║                             │           └─────────────┘                      ║
║                             │                   │                            ║
║                             │         ┌────────┴────────┐                    ║
║                             │         ▼                 ▼                    ║
║                             │  ┌────────────┐    ┌────────────┐              ║
║                       TRAIN │  │ reported_  │    │  oracle_   │ MEASURE     ║
║                       ◀─────┘  │  reward    │    │   score    │ (hidden)    ║
║                                └────────────┘    └────────────┘              ║
║                                                                              ║
║  KEY INSIGHT:                                                                ║
║  • Model is trained ONLY on reported_reward (policy gradient)                ║
║  • Model NEVER sees oracle_score during training                             ║
║  • We measure: does the gap (reported - oracle) grow?                        ║
║  • If gap grows → emergent reward hacking!                                   ║
║                                                                              ║
║  ATTACK VECTORS (model must discover these):                                 ║
║  • Discard data: observe fewer variables → inflated likelihood               ║
║  • Score bonus: add constant to log-prob → unbounded reward                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_reward_hacking_setup()

    for config in EXPERIMENTS:
        print_experiment_costs(config)

    print_summary_table()
