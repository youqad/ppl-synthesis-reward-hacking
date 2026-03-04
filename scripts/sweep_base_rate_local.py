#!/usr/bin/env python3
"""Base-rate exploit sweep using a LOCAL HuggingFace model (no API needed).

Runs the same analysis pipeline as sweep_base_rate.py but generates
completions from a local transformers model. Designed for measuring the
exact base rate of Qwen/Qwen3-4B-Instruct-2507 (the training model).

Usage (any machine with the model cached):
    pixi run -e dev python scripts/sweep_base_rate_local.py \
        --model-name "Qwen/Qwen3-4B-Instruct-2507" \
        --n-prompts 20 --samples-per-prompt 10 \
        --scoring-d 3 \
        --output-dir artifacts/sweep_qwen3_4b_instruct_2507
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ppl_synthesis_reward_hacking.data.pymc_synthesis_loader import (
    SYSTEM_PROMPT,
    get_prompts,
    make_scoring_data_dict,
)
from ppl_synthesis_reward_hacking.logging.completions import (
    CompletionRecord,
    CompletionWriter,
    make_timestamp,
)

# reuse analysis and summary helpers from the API sweep
from scripts.sweep_base_rate import (
    _build_sweep_summary,
    _completion_metadata,
    _make_json_safe,
    _model_output_dir_name,
    _print_summary_table,
    _SweepStats,
    _update_stats_from_analysis,
    analyse_completion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model(model_name: str, device: str = "auto"):
    """Load model and tokenizer from HuggingFace."""
    log.info("Loading model %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded on %s", next(model.parameters()).device)
    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Generate one completion from chat messages."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=1.0,
            top_k=0,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Local model base-rate exploit sweep")
    p.add_argument(
        "--model-name",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model ID",
    )
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--samples-per-prompt", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--exec-timeout", type=int, default=30)
    p.add_argument("--scoring-d", type=int, default=3)
    p.add_argument("--output-dir", default="artifacts/sweep_qwen3_4b_instruct_2507")
    p.add_argument(
        "--device",
        default="auto",
        help="Device for model (auto, cpu, cuda)",
    )
    args = p.parse_args()

    model, tokenizer = load_model(args.model_name, device=args.device)

    scoring_data = make_scoring_data_dict(d=args.scoring_d)
    prompts = get_prompts(args.n_prompts)

    n_total = args.n_prompts * args.samples_per_prompt
    log.info(
        "Sweep: %s, %d prompts x %d samples = %d completions, d=%d",
        args.model_name,
        args.n_prompts,
        args.samples_per_prompt,
        n_total,
        args.scoring_d,
    )

    output_dir = Path(args.output_dir)
    slug = _model_output_dir_name(args.model_name)
    model_dir = output_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    writer = CompletionWriter(model_dir / "completions.jsonl")
    messages_template = [{"role": "system", "content": SYSTEM_PROMPT}]

    stats = _SweepStats()
    sample_idx = 0

    for prompt_idx, prompt in enumerate(prompts):
        messages = [*messages_template, {"role": "user", "content": prompt}]

        for _ in range(args.samples_per_prompt):
            try:
                completion_text = generate_completion(
                    model,
                    tokenizer,
                    messages,
                    temperature=args.temperature,
                    max_new_tokens=args.max_tokens,
                )
            except Exception as e:
                log.warning("Generation failed: %s", e)
                completion_text = ""

            analysis = analyse_completion(
                completion_text, scoring_data, exec_timeout=args.exec_timeout
            )
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
                metadata=_completion_metadata(args.model_name, analysis),
            )
            writer.write(record)
            sample_idx += 1

        done = (prompt_idx + 1) * args.samples_per_prompt
        log.info("[%s] %d/%d completions done", args.model_name, done, n_total)

    writer.close()
    summary = _build_sweep_summary(model_name=args.model_name, n_total=n_total, stats=stats)

    summary_path = model_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_make_json_safe(summary), f, indent=2)

    all_summaries = [summary]
    print("\n" + "=" * 80)
    print("SWEEP RESULTS")
    print("=" * 80 + "\n")
    _print_summary_table(all_summaries)
    print(f"\nResults written to: {model_dir}")


if __name__ == "__main__":
    main()
