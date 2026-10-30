"""Build adapter for the standard-library-only generated operator core."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

_BUILD_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PACKAGE_CANDIDATES = (
    _BUILD_ROOT / "symmetrix" / "source" / "symmetrix",
    _BUILD_ROOT / "source" / "symmetrix",
)
_SOURCE_PACKAGE = next(
    (candidate for candidate in _SOURCE_PACKAGE_CANDIDATES if candidate.is_dir()),
    _SOURCE_PACKAGE_CANDIDATES[0],
)
_BUILD_PACKAGE = "_symmetrix_build_codegen_source"


def _load_core():
    package = sys.modules.get(_BUILD_PACKAGE)
    if package is None:
        package = ModuleType(_BUILD_PACKAGE)
        package.__path__ = [str(_SOURCE_PACKAGE)]
        package.__package__ = _BUILD_PACKAGE
        sys.modules[_BUILD_PACKAGE] = package
    return importlib.import_module(f"{_BUILD_PACKAGE}.generated_operator_codegen")


_CORE = _load_core()


def load_build_source_module(name: str):
    """Load one standard-library build source without importing Symmetrix."""

    _load_core()
    return importlib.import_module(f"{_BUILD_PACKAGE}.{name}")


def render_generated_r0_operator(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    source_content_sha256: str,
    backend: str,
    target: str,
    precision: str,
    schedule: str,
    floating_point_policy: str | None = None,
    apply_edge_weights: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Render R0 from explicit build inputs without importing the runtime."""

    return _CORE.render_generated_r0_operator(
        contract,
        generation_version=generation_version,
        source_content_sha256=source_content_sha256,
        backend=backend,
        target=target,
        precision=precision,
        schedule=schedule,
        floating_point_policy=floating_point_policy,
        apply_edge_weights=apply_edge_weights,
    )


def render_generated_m0_operator(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    backend: str,
    target: str,
    precision: str,
    schedule: str = "chunk32",
) -> tuple[str, dict[str, Any]]:
    """Render M0 from explicit build inputs without importing the runtime."""

    if backend == "host":
        return _CORE.render_jit_m0_host_plugin(
            contract,
            generation_version=generation_version,
            precision=precision,
            schedule=schedule,
            target=target,
        )
    return _CORE.render_jit_m0_device_module(
        contract,
        generation_version=generation_version,
        precision=precision,
        schedule=schedule,
        target=target,
    )


def render_generated_m1_operator(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    backend: str,
    target: str,
    precision: str,
    schedule: str = "chunk32",
) -> tuple[str, dict[str, Any]]:
    """Render M1 from explicit build inputs without importing the runtime."""

    renderer = (
        _CORE.render_jit_m1_host_plugin
        if backend == "host"
        else _CORE.render_jit_m1_device_module
    )
    return renderer(
        contract,
        generation_version=generation_version,
        precision=precision,
        schedule=schedule,
        target=target,
    )


def generated_r1_metadata(
    contract: Mapping[str, Any],
    *,
    backend: str,
    target: str,
    precision: str,
) -> dict[str, Any]:
    """Return canonical R1 metadata for one exact build target."""

    codegen = load_build_source_module("mace_jit_codegen")
    if backend == "host":
        metadata = codegen.jit_r1_host_plugin_metadata(
            contract, precision=precision, target=target
        )
    elif backend == "cuda":
        if not target.startswith("sm_") or not target[3:].isdigit():
            raise ValueError("CUDA R1 target must use sm_NN spelling")
        compute_capability = int(target[3:])
        metadata = codegen.jit_r1_cuda_plugin_metadata(
            contract, compute_capability, precision=precision
        )
    elif backend == "hip":
        metadata = codegen.factorized_hip_plugin_metadata(
            contract, target, precision=precision
        )
    else:
        raise ValueError("R1 backend must be host, cuda, or hip")
    return {**metadata, "operator": "R1", "schema": metadata["abi"]}


def render_generated_r1_operator(
    contract: Mapping[str, Any],
    *,
    backend: str,
    target: str,
    precision: str,
) -> tuple[str, dict[str, Any]]:
    """Render canonical R1 source and metadata for one exact build target."""

    codegen = load_build_source_module("mace_jit_codegen")
    metadata = generated_r1_metadata(
        contract, backend=backend, target=target, precision=precision
    )
    if backend == "host":
        source = codegen.render_jit_r1_host_plugin(
            contract, precision=precision, target=target
        )
    elif backend == "cuda":
        source = codegen.render_jit_r1_cuda_module(
            contract, int(target[3:]), precision=precision
        )
    else:
        source = codegen.render_jit_r1_hip_module(contract, target, precision=precision)
    return source, metadata


__all__ = [
    "generated_r1_metadata",
    "load_build_source_module",
    "render_generated_m0_operator",
    "render_generated_m1_operator",
    "render_generated_r0_operator",
    "render_generated_r1_operator",
]
