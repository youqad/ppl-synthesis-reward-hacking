"""Shared record utilities for eval scripts (dedup, code extraction, hashing)."""

from __future__ import annotations

from collections.abc import Callable

from ppl_synthesis_reward_hacking.backends.pymc.code_executor import extract_pymc_code
from ppl_synthesis_reward_hacking.logging.completions import CompletionRecord
from ppl_synthesis_reward_hacking.utils.hashing import normalized_text_hash


def extract_record_code(record: CompletionRecord) -> str | None:
    """Try record.code first, then extract from completion_text."""
    if record.code and record.code.strip():
        return record.code
    if record.completion_text and record.completion_text.strip():
        return extract_pymc_code(record.completion_text)
    return None


def record_code_hash(record: CompletionRecord) -> str | None:
    """Hash extracted code for deduplication."""
    code = extract_record_code(record)
    return normalized_text_hash(code) if code else None


def deduplicate_records(
    records: list[CompletionRecord],
    hash_fn: Callable[[CompletionRecord], str | None] = record_code_hash,
) -> list[CompletionRecord]:
    """Deduplicate by hash, keeping first occurrence. Unhashable records pass through."""
    deduped: dict[str, CompletionRecord] = {}
    passthrough: list[CompletionRecord] = []
    for record in records:
        key = hash_fn(record)
        if key is None:
            passthrough.append(record)
            continue
        if key not in deduped:
            deduped[key] = record
    return list(deduped.values()) + passthrough
