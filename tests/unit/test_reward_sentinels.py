"""Tests for reward sentinel constants and outcome classification."""

from __future__ import annotations

from ppl_synthesis_reward_hacking.reward_sentinels import (
    EXEC_FAIL_REWARD,
    OUTCOME_EXEC_FAIL,
    OUTCOME_PARSE_FAIL,
    OUTCOME_VALID,
    PARSE_FAIL_REWARD,
    classify_outcome_from_reward,
)


class TestClassifyOutcomeFromReward:
    def test_parse_fail_reward(self):
        assert classify_outcome_from_reward(PARSE_FAIL_REWARD) == OUTCOME_PARSE_FAIL
        assert classify_outcome_from_reward(-500.0) == "parse_fail"

    def test_exec_fail_reward(self):
        assert classify_outcome_from_reward(EXEC_FAIL_REWARD) == OUTCOME_EXEC_FAIL
        assert classify_outcome_from_reward(-400.0) == "exec_fail"

    def test_valid_reward(self):
        assert classify_outcome_from_reward(-5.0) == OUTCOME_VALID

    def test_nan_reward(self):
        assert classify_outcome_from_reward(float("nan")) == OUTCOME_EXEC_FAIL

    def test_inf_reward(self):
        assert classify_outcome_from_reward(float("inf")) == OUTCOME_EXEC_FAIL

    def test_neg_inf_reward(self):
        assert classify_outcome_from_reward(float("-inf")) == OUTCOME_EXEC_FAIL

    def test_zero_reward(self):
        assert classify_outcome_from_reward(0.0) == OUTCOME_VALID

    def test_positive_reward(self):
        assert classify_outcome_from_reward(10.0) == OUTCOME_VALID

    def test_slightly_negative_reward(self):
        assert classify_outcome_from_reward(-0.693) == OUTCOME_VALID

    def test_large_negative_but_not_sentinel(self):
        # -300 is between EXEC_FAIL and PARSE_FAIL but matches neither
        assert classify_outcome_from_reward(-300.0) == OUTCOME_VALID


class TestSentinelValues:
    def test_exec_fail_is_minus_400(self):
        assert EXEC_FAIL_REWARD == -400.0

    def test_parse_fail_is_minus_500(self):
        assert PARSE_FAIL_REWARD == -500.0

    def test_parse_fail_less_than_exec_fail(self):
        assert PARSE_FAIL_REWARD < EXEC_FAIL_REWARD
