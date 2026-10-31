"""Utilities for masking and validating antibody sequences."""

import random
from typing import List, Sequence, Tuple


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(sequence: str) -> bool:
    """Return True if the sequence contains only canonical amino acids."""
    return bool(sequence) and set(sequence.upper()).issubset(AMINO_ACIDS)


def count_mutations(parent: str, candidate: str) -> int:
    """Count substitutions relative to the parent sequence."""
    if len(parent) != len(candidate):
        raise ValueError(
            f"Length mismatch: parent={len(parent)}, candidate={len(candidate)}"
        )

    return sum(a != b for a, b in zip(parent, candidate))


def mask_positions(
    sequence: str,
    positions: Sequence[int],
    mask_token: str = "<mask>",
) -> Tuple[str, List[int]]:
    """
    Mask selected zero-based positions.

    Positions must be selected by the biological team.
    """
    if not validate_sequence(sequence):
        raise ValueError("Parent sequence contains invalid amino acids.")

    if not positions:
        raise ValueError("No masking positions supplied.")

    if len(set(positions)) != len(positions):
        raise ValueError("Masking positions contain duplicates.")

    if min(positions) < 0 or max(positions) >= len(sequence):
        raise ValueError("Masking position outside sequence bounds.")

    masked = list(sequence)

    for position in positions:
        masked[position] = mask_token

    return "".join(masked), list(positions)


def select_diverse_candidates(
    parent: str,
    candidates: Sequence[str],
    min_mutations: int = 5,
) -> List[str]:
    """Filter invalid, incomplete, duplicate, and insufficiently mutated candidates."""
    selected = []
    seen = set()

    for candidate in candidates:
        candidate = candidate.upper()

        if len(candidate) != len(parent):
            continue

        if not validate_sequence(candidate):
            continue

        if candidate in seen:
            continue

        if count_mutations(parent, candidate) < min_mutations:
            continue

        seen.add(candidate)
        selected.append(candidate)

    return selected