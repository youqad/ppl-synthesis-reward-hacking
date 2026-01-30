#!/usr/bin/env python3
"""Generate completions from trained LoRA checkpoints for post-hoc judging.

Load checkpoint adapters, generate PyMC programs, score them, and save
as CompletionRecord JSONL compatible with judge_completions.py.

Usage:
    # all checkpoints in a training run
    python scripts/generate_from_checkpoints.py \
        --run-dir artifacts/grpo_synthstats_20260128_012030/ \
        --n-samples 20

    # single checkpoint
    python scripts/generate_from_checkpoints.py \
        --checkpoint artifacts/.../checkpoint-50/ \
        --base-model meta-llama/Llama-3.2-1B \
        --n-samples 20

    # then judge
    python scripts/judge_completions.py \
        --completions artifacts/.../checkpoint-50/completions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate completions from trained checkpoints")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="Training output dir (discovers all checkpoints)")
    g.add_argument("--checkpoint", help="Single checkpoint path")
    p.add_argument("--base-model", default=None, help="Base model (auto-detected if possible)")
    p.add_argument("--n-samples", type=int, default=10, help="Completions per prompt")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--device", default=None, help="Device override (default: auto)")
    return p.parse_args()


def _discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = sorted(
        run_dir.glob("checkpoint-*/"),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1))
        if re.search(r"checkpoint-(\d+)", p.name)
        else 0,
    )
    return [c for c in checkpoints if c.is_dir()]


def _detect_base_model(run_dir: Path) -> str | None:
    # try to read from results.json or training_args
    for name in ("results.json", "training_args.json"):
        p = run_dir / name
        if p.exists():
            try:
                data = json.loads(p.read_text())
                model = data.get("config", {}).get("model") or data.get("model_name_or_path")
                if model:
                    return model
            except Exception:
                pass
    # try trainer_state.json
    for cp in run_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            data = json.loads(cp.read_text())
            # TRL stores model info in various places
            if "model_name" in data:
                return data["model_name"]
        except Exception:
            pass
    return None


def _generate_from_checkpoint(
    checkpoint_path: Path,
    base_model: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    device: str | None,
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ppl_synthesis_reward_hacking.backends.pymc.code_executor import (
        execute_pymc_code,
        extract_pymc_code,
    )
    from ppl_synthesis_reward_hacking.data.synthstats_loader import (
        load_synthstats_prompts,
        make_scoring_data_dict,
    )
    from ppl_synthesis_reward_hacking.experiments.synthstats_reward import (
        EXEC_FAIL_REWARD,
        PARSE_FAIL_REWARD,
        _score_pymc_model,
    )
    from ppl_synthesis_reward_hacking.logging.completions import (
        CompletionRecord,
        CompletionWriter,
        make_timestamp,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    log.info("Loading adapter from: %s", checkpoint_path)
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    scoring_data = make_scoring_data_dict()
    prompts = load_synthstats_prompts("data/synthstats/synthstats_train.jsonl", max_examples=20)

    completions_path = checkpoint_path / "completions.jsonl"
    writer = CompletionWriter(completions_path)

    step_match = re.search(r"checkpoint-(\d+)", checkpoint_path.name)
    batch_label = int(step_match.group(1)) if step_match else 0

    total = 0
    for prompt_dict in prompts:
        prompt_value = prompt_dict["prompt"]
        if isinstance(prompt_value, list):
            prompt_text = "\n".join(
                m.get("content", "") for m in prompt_value if isinstance(m, dict)
            )
        else:
            prompt_text = str(prompt_value)

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        for _sample_idx in range(n_samples):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = outputs[0][input_ids.shape[1] :]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

            code = extract_pymc_code(completion_text)
            if code is None:
                writer.write(CompletionRecord(
                    batch=batch_label,
                    index=total,
                    prompt=prompt_text,
                    completion_text=completion_text,
                    code=None,
                    reported_reward=PARSE_FAIL_REWARD,
                    oracle_score=PARSE_FAIL_REWARD,
                    gap=0.0,
                    outcome="parse_fail",
                    timestamp=make_timestamp(),
                ))
                total += 1
                continue

            pymc_model = execute_pymc_code(code, scoring_data, timeout=30)
            if pymc_model is None:
                writer.write(CompletionRecord(
                    batch=batch_label,
                    index=total,
                    prompt=prompt_text,
                    completion_text=completion_text,
                    code=code,
                    reported_reward=EXEC_FAIL_REWARD,
                    oracle_score=EXEC_FAIL_REWARD,
                    gap=0.0,
                    outcome="exec_fail",
                    timestamp=make_timestamp(),
                ))
                total += 1
                continue

            try:
                reported, oracle = _score_pymc_model(pymc_model)
                oracle_val = oracle if oracle is not None else EXEC_FAIL_REWARD
                writer.write(CompletionRecord(
                    batch=batch_label,
                    index=total,
                    prompt=prompt_text,
                    completion_text=completion_text,
                    code=code,
                    reported_reward=reported,
                    oracle_score=oracle_val,
                    gap=reported - oracle_val,
                    outcome="valid",
                    timestamp=make_timestamp(),
                ))
            except Exception:
                writer.write(CompletionRecord(
                    batch=batch_label,
                    index=total,
                    prompt=prompt_text,
                    completion_text=completion_text,
                    code=code,
                    reported_reward=EXEC_FAIL_REWARD,
                    oracle_score=EXEC_FAIL_REWARD,
                    gap=0.0,
                    outcome="score_fail",
                    timestamp=make_timestamp(),
                ))
            total += 1

        writer.flush()

    writer.close()
    log.info("Generated %d completions -> %s", total, completions_path)

    # save generation metadata
    meta = {
        "base_model": base_model,
        "checkpoint": str(checkpoint_path),
        "n_samples": n_samples,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "device": device,
        "total_generated": total,
        "timestamp": make_timestamp(),
    }
    meta_path = checkpoint_path / "generation_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Metadata -> %s", meta_path)


def main() -> None:
    args = parse_args()

    try:
        import torch  # noqa: F401
        from peft import PeftModel  # noqa: F401
    except ImportError:
        log.error("torch and peft required. Install with: pip install torch peft transformers")
        sys.exit(1)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoints = _discover_checkpoints(run_dir)
        if not checkpoints:
            log.error("No checkpoints found in %s", run_dir)
            sys.exit(1)
        log.info("Found %d checkpoints in %s", len(checkpoints), run_dir)

        base_model = args.base_model or _detect_base_model(run_dir)
        if not base_model:
            log.error("Could not detect base model. Use --base-model")
            sys.exit(1)

        for cp in checkpoints:
            log.info("Processing %s", cp.name)
            _generate_from_checkpoint(
                cp, base_model, args.n_samples, args.temperature,
                args.max_new_tokens, args.device,
            )
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_dir():
            log.error("Checkpoint not found: %s", checkpoint_path)
            sys.exit(1)

        base_model = args.base_model
        if not base_model:
            # try to detect from parent run dir
            parent = checkpoint_path.parent
            base_model = _detect_base_model(parent)
        if not base_model:
            log.error("Could not detect base model. Use --base-model")
            sys.exit(1)

        _generate_from_checkpoint(
            checkpoint_path, base_model, args.n_samples, args.temperature,
            args.max_new_tokens, args.device,
        )


if __name__ == "__main__":
    main()
