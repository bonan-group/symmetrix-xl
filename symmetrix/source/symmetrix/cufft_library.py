"""Register the selected backend's external cuFFT without loading it."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path, PurePosixPath
from typing import Any


def configure_cufft_library(native: Any, toolkit: str) -> None:
    configure = getattr(native, "_configure_cufft_library", None)
    if not callable(configure) or os.environ.get("SYMMETRIX_CUFFT_LIBRARY"):
        return
    providers = {
        "cuda12": ("nvidia-cufft-cu12", "nvidia/cufft/lib/libcufft.so.11"),
        "cuda13": ("nvidia-cufft", "nvidia/cu13/lib/libcufft.so.12"),
    }
    if toolkit not in providers:
        return
    name, member = providers[toolkit]
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        # A source/C++ installation may use the matching system SONAME.
        return
    expected = PurePosixPath(member)
    for record in distribution.files or ():
        if PurePosixPath(str(record)) == expected:
            candidate = Path(distribution.locate_file(record))
            if candidate.is_file():
                configure(str(candidate.resolve()))
            return
