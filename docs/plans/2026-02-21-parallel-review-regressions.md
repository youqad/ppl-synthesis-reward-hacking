# TRL Hydra + Prompt Diversity Regression Fixes Implementation Plan

This plan fixes two regressions with no overlap: TRL Hydra startup breakage and
hardcoded prompt diversity collapse. The TRL path should still fail fast on bad
config keys, while accepting monitoring keys flattened by Hydra. Prompt
sampling should preserve diversity by deduplicating hardcoded templates before
seeded shuffling.

Tech stack: Python 3.11, dataclasses, Hydra mapping flattening, pytest.

### Task 1: Patch TRL config parsing for flattened monitoring keys

Edit `scripts/trl_reward_hacking.py`.

Add TRL config fields for flattened monitoring keys:
- `judge_sampling_policy`
- `judge_adaptive_min`
- `judge_adaptive_max`
- `judge_adaptive_target_metric`

Validate these new fields using the same accepted value sets as the Tinker
path. Keep fail-fast unknown-key behavior unchanged, and keep the runtime
warning that TRL ignores judge/rubric evolution behavior.

### Task 2: Add TRL regression test coverage

Edit `tests/unit/test_trl_reward_hacking_script.py`.

Add a unit test where a `monitoring_mode` payload includes adaptive judge keys,
and assert `config_from_mapping` succeeds. Also add or keep a guard test that
proves truly unknown keys still raise `ValueError`.

### Task 3: Fix hardcoded prompt pool diversity

Edit `src/ppl_synthesis_reward_hacking/experiments/tinker_training.py`.

In `_initialize_training`, build `base_prompts` from unique templates in stable
order instead of a pre-cycled repeated list. Leave JSONL path logic untouched.

### Task 4: Add prompt-diversity regression test

Create `tests/unit/test_tinker_prompt_sampling.py`.

First, reproduce the pre-fix failure pattern: build a duplicated hardcoded
pool and assert seeded shuffle can yield fewer than `n_prompts` unique prompts.
Then assert the fixed behavior: deduplicate that same pool and confirm seeded
shuffle returns `n_prompts` distinct prompts when enough unique templates
exist.

### Task 5: Verify targeted tests

Run targeted tests:
- `pixi run -e dev pytest tests/unit/test_trl_reward_hacking_script.py -q`
- `pixi run -e dev pytest tests/unit/test_tinker_prompt_sampling.py -q`

Run the combined quick check:
- `pixi run -e dev pytest tests/unit/test_trl_reward_hacking_script.py tests/unit/test_tinker_prompt_sampling.py -q`
