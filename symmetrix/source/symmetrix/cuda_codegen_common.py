"""Shared CUDA input normalization helpers.

The generated OAM path owns no ordinary-NequIP implementation details, but it
still needs the same target, precision, and launch-size normalization used by
the CUDA artifact boundary.  Keep those small, neutral helpers here so the OAM
path remains independent of retired adapters.  The ``SYMMETRIX_JIT_POLICY``
selection is framework policy and lives in :mod:`symmetrix.jit`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_COMPILERS = ("nvrtc", "nvcc")
DEFAULT_THREADS_PER_BLOCK = 128
SUPPORTED_PRECISIONS = ("float32", "float64")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _compute_capability(value: Any) -> tuple[int, str]:
    """Normalize ``sm120``/``12.0``/``120`` to ``(120, "sm120")``."""

    if isinstance(value, Mapping):
        value = value.get(
            "compute_capability_code",
            value.get(
                "compute_capability", value.get("architecture", value.get("target"))
            ),
        )
    if isinstance(value, bool):
        raise ValueError("compute_capability must describe a CUDA architecture")
    if isinstance(value, int):
        code = value
    elif isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("sm") or text.startswith("compute_"):
            text = text.split("_", 1)[1] if "_" in text else text[2:]
        if "." in text:
            major, _, minor = text.partition(".")
            if not major.isdigit() or not minor.isdigit():
                raise ValueError(f"unsupported compute capability {value!r}")
            code = 10 * int(major) + int(minor)
        elif text.isdigit():
            code = int(text)
        else:
            raise ValueError(f"unsupported compute capability {value!r}")
    else:
        raise ValueError(f"unsupported compute capability {value!r}")
    if code <= 0:
        raise ValueError("compute_capability must be positive")
    return code, f"sm{code}"


def canonical_cuda_target_identity(value: Any) -> str:
    """Return the canonical identity shared by build and runtime spellings."""

    return _compute_capability(value)[1]


def _normalize_precision(precision: Any) -> str:
    value = str(precision).lower()
    _require(value in SUPPORTED_PRECISIONS, "precision must be 'float32' or 'float64'")
    return value


def _threads_per_block(value: Any) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        "threads_per_block must be an integer",
    )
    _require(
        value > 0 and value % 32 == 0,
        "threads_per_block must be a positive multiple of 32",
    )
    return int(value)


__all__ = [
    "DEFAULT_THREADS_PER_BLOCK",
    "SUPPORTED_COMPILERS",
    "SUPPORTED_PRECISIONS",
    "_compute_capability",
    "_normalize_precision",
    "_threads_per_block",
    "canonical_cuda_target_identity",
]
