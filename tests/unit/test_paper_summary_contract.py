from __future__ import annotations

from ppl_synthesis_reward_hacking.logging.paper_summary_contract import (
    PAPER_SUMMARY_DEFAULTS,
    PAPER_SUMMARY_KEYS_CORE,
    PAPER_SUMMARY_KEYS_TINKER,
    paper_summary_keys_for_train_name,
)


def test_paper_summary_defaults_cover_all_contract_keys() -> None:
    for key in (*PAPER_SUMMARY_KEYS_CORE, *PAPER_SUMMARY_KEYS_TINKER):
        assert key in PAPER_SUMMARY_DEFAULTS


def test_paper_summary_keys_for_train_name() -> None:
    tinker_keys = paper_summary_keys_for_train_name("tinker")
    assert set(PAPER_SUMMARY_KEYS_CORE).issubset(tinker_keys)
    assert set(PAPER_SUMMARY_KEYS_TINKER).issubset(tinker_keys)

    trl_keys = paper_summary_keys_for_train_name("trl")
    assert set(PAPER_SUMMARY_KEYS_CORE).issubset(trl_keys)
    assert not set(PAPER_SUMMARY_KEYS_TINKER).issubset(trl_keys)
