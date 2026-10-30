"""Shared exact supercell-shape helpers for benchmark workloads."""

from __future__ import annotations

from collections.abc import Sequence


def balanced_repeat_factors(unit_cells: int) -> tuple[int, int, int]:
    """Return the least-anisotropic ordered factorization of ``unit_cells``."""

    if (
        isinstance(unit_cells, bool)
        or not isinstance(unit_cells, int)
        or unit_cells < 1
    ):
        raise ValueError("unit_cells must be a positive integer")
    candidates = []
    for first in range(1, unit_cells + 1):
        if unit_cells % first != 0:
            continue
        remaining = unit_cells // first
        for second in range(first, remaining + 1):
            if remaining % second != 0:
                continue
            third = remaining // second
            if second <= third:
                candidates.append((first, second, third))
    return min(
        candidates,
        key=lambda factors: (
            factors[2] - factors[0],
            factors[2] - factors[1],
            factors[1] - factors[0],
            factors,
        ),
    )


def repeat_factors(repeat: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalize cubic or explicit three-axis repeat factors."""

    if isinstance(repeat, bool):
        raise ValueError("repeat must be a positive integer or three positive integers")
    if isinstance(repeat, int):
        factors = (repeat, repeat, repeat)
    else:
        factors = tuple(repeat)
    if len(factors) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in factors
    ):
        raise ValueError("repeat must be a positive integer or three positive integers")
    return factors
