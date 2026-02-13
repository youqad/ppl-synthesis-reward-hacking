"""Parser for ToyProgram from LLM text output."""

from __future__ import annotations

from .program import ToyProgram


def parse_program_from_text(text: str, d: int) -> ToyProgram | None:
    """Parse LLM output into ToyProgram.

    Expected format:
        observed: [0, 1, 2, 3]
        prior_strength: 0.5

    The prior_strength parameter controls the concentration of the prior.
    Higher values indicate stronger prior beliefs (can lead to improper models
    if misused, but this is a legitimate modeling choice).

    Note on naming: Uses innocuous terms ("prior_strength") rather than
    "HACK" or "improper" to avoid biasing LLM behavior during training.

    Args:
        text: Raw text output from LLM
        d: Data dimensionality (for filtering out-of-bounds indices)

    Returns:
        ToyProgram if parsing succeeds, None otherwise.
    """
    try:
        lines = text.strip().split("\n")
        observed_idx = None
        prior_strength = 0.0

        for line in lines:
            line_lower = line.strip().lower()
            if line_lower.startswith("observed:"):
                rest = line_lower.split(":", 1)[1].strip()
                rest = rest.replace("[", "").replace("]", "")
                if rest:
                    observed_idx = tuple(int(x.strip()) for x in rest.split(",") if x.strip())
                else:
                    observed_idx = ()
            elif line_lower.startswith("prior_strength:"):
                rest = line_lower.split(":", 1)[1].strip()
                prior_strength = float(rest)

        if observed_idx is None:
            return None

        # filter out-of-bounds indices
        observed_idx = tuple(i for i in observed_idx if 0 <= i < d)
        return ToyProgram(
            observed_idx=observed_idx,
            score_bonus=max(0.0, prior_strength),
        )

    except (ValueError, IndexError, TypeError):
        return None


def program_to_text(program: ToyProgram) -> str:
    """Convert ToyProgram to text format for LLM training.

    Produces format matching what parse_program_from_text expects.

    Args:
        program: ToyProgram to serialize

    Returns:
        Text representation suitable for LLM output format.
    """
    return f"observed: {list(program.observed_idx)}\nprior_strength: {program.score_bonus:.4f}"


def parse_observed_list(text: str) -> tuple[int, ...] | None:
    """Parse just the observed variable list from text.

    Useful for partial parsing or validation.

    Args:
        text: Text that may contain "observed: [...]" line

    Returns:
        Tuple of indices if found, None otherwise.
    """
    try:
        for line in text.strip().split("\n"):
            line_lower = line.strip().lower()
            if line_lower.startswith("observed:"):
                rest = line_lower.split(":", 1)[1].strip()
                rest = rest.replace("[", "").replace("]", "")
                if rest:
                    return tuple(int(x.strip()) for x in rest.split(",") if x.strip())
                return ()
        return None
    except (ValueError, IndexError):
        return None
