"""Unit tests for ToyProgram parser.

Tests the bidirectional conversion between ToyProgram and LLM text format.
"""

from __future__ import annotations

from ppl_synthesis_reward_hacking.backends.toy import ToyProgram
from ppl_synthesis_reward_hacking.backends.toy.parser import (
    parse_observed_list,
    parse_program_from_text,
    program_to_text,
)


class TestParseProgramFromText:
    """Tests for parse_program_from_text."""

    def test_parse_valid_program(self):
        """Parse well-formed program specification."""
        text = "observed: [0, 1, 2]\nprior_strength: 0.5"
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert prog.observed_idx == (0, 1, 2)
        assert prog.score_bonus == 0.5

    def test_parse_empty_observed(self):
        """Parse program with empty observed list."""
        text = "observed: []\nprior_strength: 0.0"
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert prog.observed_idx == ()
        assert prog.score_bonus == 0.0

    def test_parse_all_dimensions_observed(self):
        """Parse program observing all variables."""
        text = "observed: [0, 1, 2, 3, 4, 5, 6, 7]\nprior_strength: 0.0"
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert prog.observed_idx == tuple(range(8))

    def test_parse_malformed_returns_none(self):
        """Malformed text should return None."""
        malformed = [
            "garbage",
            "no format here",
            "observed: invalid\nprior_strength: abc",
            "",
            "   ",
        ]

        for text in malformed:
            result = parse_program_from_text(text, d=8)
            assert result is None, f"Should return None for: {text!r}"

    def test_parse_missing_prior_strength_defaults_zero(self):
        """Missing prior_strength defaults to 0.0."""
        text = "observed: [0, 1, 2]"
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert prog.score_bonus == 0.0

    def test_parse_filters_out_of_bounds_indices(self):
        """Out-of-bounds indices should be filtered."""
        text = "observed: [0, 1, 99, 100]\nprior_strength: 0.0"
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert 99 not in prog.observed_idx
        assert 100 not in prog.observed_idx
        assert prog.observed_idx == (0, 1)

    def test_parse_filters_negative_indices(self):
        """Negative indices should be filtered."""
        text = "observed: [-1, 0, 1, -5]\nprior_strength: 0.0"
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert -1 not in prog.observed_idx
        assert -5 not in prog.observed_idx
        assert prog.observed_idx == (0, 1)

    def test_parse_negative_prior_strength_clipped_to_zero(self):
        """Negative prior_strength should be clipped to 0.0."""
        text = "observed: [0, 1]\nprior_strength: -5.0"
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert prog.score_bonus == 0.0

    def test_parse_case_insensitive(self):
        """Parsing should be case-insensitive for field names."""
        variations = [
            "OBSERVED: [0, 1]\nPRIOR_STRENGTH: 1.0",
            "Observed: [0, 1]\nPrior_Strength: 1.0",
            "ObSeRvEd: [0, 1]\nPrIoR_StReNgTh: 1.0",
        ]

        for text in variations:
            prog = parse_program_from_text(text, d=8)
            assert prog is not None, f"Failed to parse: {text!r}"
            assert prog.observed_idx == (0, 1)
            assert prog.score_bonus == 1.0

    def test_parse_with_extra_whitespace(self):
        """Extra whitespace should be handled."""
        text = "  observed:   [ 0 ,  1 ,  2 ]  \n  prior_strength:   0.5  "
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert prog.observed_idx == (0, 1, 2)
        assert prog.score_bonus == 0.5

    def test_parse_with_extra_lines(self):
        """Extra lines should be ignored."""
        text = """Some preamble text
observed: [0, 1, 2]
Some other text
prior_strength: 0.5
More text after"""
        prog = parse_program_from_text(text, d=8)

        assert prog is not None
        assert prog.observed_idx == (0, 1, 2)
        assert prog.score_bonus == 0.5

    def test_parse_missing_observed_returns_none(self):
        """Missing observed field should return None."""
        text = "prior_strength: 0.5"
        result = parse_program_from_text(text, d=8)
        assert result is None


class TestProgramToText:
    """Tests for program_to_text."""

    def test_serialize_basic_program(self):
        """Serialize a basic program."""
        prog = ToyProgram(observed_idx=(0, 1, 2), score_bonus=0.5)
        text = program_to_text(prog)

        assert "observed:" in text.lower()
        assert "[0, 1, 2]" in text
        assert "prior_strength:" in text.lower()
        assert "0.5" in text

    def test_serialize_empty_observed(self):
        """Serialize program with no observed variables."""
        prog = ToyProgram(observed_idx=(), score_bonus=0.0)
        text = program_to_text(prog)

        assert "[]" in text

    def test_serialize_zero_bonus(self):
        """Serialize program with zero bonus."""
        prog = ToyProgram(observed_idx=(0,), score_bonus=0.0)
        text = program_to_text(prog)

        assert "0.0000" in text


class TestRoundtrip:
    """Tests for roundtrip serialization."""

    def test_roundtrip_preserves_semantics(self):
        """Roundtrip (program → text → parse) preserves semantics."""
        test_programs = [
            ToyProgram(observed_idx=(0, 2, 4), score_bonus=0.3),
            ToyProgram(observed_idx=(), score_bonus=5.0),
            ToyProgram(observed_idx=tuple(range(8)), score_bonus=0.0),
            ToyProgram(observed_idx=(0,), score_bonus=0.0001),
        ]

        for original in test_programs:
            text = program_to_text(original)
            parsed = parse_program_from_text(text, d=8)

            assert parsed is not None, f"Failed to parse roundtrip for {original}"
            assert parsed.observed_idx == original.observed_idx, (
                f"observed_idx mismatch: {original.observed_idx} → {parsed.observed_idx}"
            )
            assert abs(parsed.score_bonus - original.score_bonus) < 1e-3, (
                f"score_bonus mismatch: {original.score_bonus} → {parsed.score_bonus}"
            )

    def test_roundtrip_multiple_iterations(self):
        """Multiple roundtrips should be stable."""
        prog = ToyProgram(observed_idx=(1, 3, 5), score_bonus=2.5)

        for _ in range(5):
            text = program_to_text(prog)
            prog = parse_program_from_text(text, d=8)
            assert prog is not None

        assert prog.observed_idx == (1, 3, 5)
        assert abs(prog.score_bonus - 2.5) < 1e-3


class TestParseObservedList:
    """Tests for parse_observed_list helper."""

    def test_parse_valid_list(self):
        """Parse valid observed list."""
        text = "observed: [0, 1, 2]"
        result = parse_observed_list(text)
        assert result == (0, 1, 2)

    def test_parse_empty_list(self):
        """Parse empty observed list."""
        text = "observed: []"
        result = parse_observed_list(text)
        assert result == ()

    def test_parse_no_observed_field(self):
        """Return None if no observed field."""
        text = "some other text"
        result = parse_observed_list(text)
        assert result is None

    def test_parse_from_multiline(self):
        """Extract observed from multiline text."""
        text = """preamble
observed: [3, 4, 5]
other stuff"""
        result = parse_observed_list(text)
        assert result == (3, 4, 5)
