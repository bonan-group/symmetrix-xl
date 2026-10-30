"""Runtime adapter for the build-safe generated operator codegen core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .generated_operator_codegen import (
    M0_CAPABILITIES_V1,
    M1_CAPABILITIES_V1,
    R0_CAPABILITIES_V1,
    select_product_device_schedule,
)
from .generated_operator_codegen import (
    _contraction_terms as _core_contraction_terms,
)
from .generated_operator_codegen import (
    jit_m0_device_metadata as _jit_m0_device_metadata,
)
from .generated_operator_codegen import (
    jit_m0_host_metadata as _jit_m0_host_metadata,
)
from .generated_operator_codegen import (
    jit_m1_device_metadata as _jit_m1_device_metadata,
)
from .generated_operator_codegen import (
    jit_m1_host_metadata as _jit_m1_host_metadata,
)
from .generated_operator_codegen import (
    jit_r0_device_metadata as _jit_r0_device_metadata,
)
from .generated_operator_codegen import (
    jit_r0_host_metadata as _jit_r0_host_metadata,
)
from .generated_operator_codegen import (
    render_generated_r0_operator as _render_generated_r0_operator,
)
from .generated_operator_codegen import (
    render_jit_m0_device_module as _render_jit_m0_device_module,
)
from .generated_operator_codegen import (
    render_jit_m0_host_plugin as _render_jit_m0_host_plugin,
)
from .generated_operator_codegen import (
    render_jit_m1_device_module as _render_jit_m1_device_module,
)
from .generated_operator_codegen import (
    render_jit_m1_host_plugin as _render_jit_m1_host_plugin,
)
from .generated_operator_codegen import (
    render_jit_r0_device_module as _render_jit_r0_device_module,
)
from .generated_operator_codegen import (
    render_jit_r0_host_plugin as _render_jit_r0_host_plugin,
)
from .jit import JIT_GENERATION_VERSION, ArtifactEnvelope

OPERATOR_ARTIFACT_ENVELOPE = ArtifactEnvelope(
    "state-free operator",
    abi_key="schema",
    identity_fingerprint=None,
    semantic_fingerprint=False,
    capabilities_key="capabilities",
)


def _m0_terms(contract: Mapping[str, Any]):
    """Retain the private runtime seam used by the Metal code generator."""

    return _core_contraction_terms(contract)


def jit_m0_device_metadata(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> dict[str, Any]:
    return _jit_m0_device_metadata(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        target=target,
        schedule=schedule,
    )


def render_jit_m0_device_module(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> tuple[str, dict[str, Any]]:
    return _render_jit_m0_device_module(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        target=target,
        schedule=schedule,
    )


def jit_m0_host_metadata(
    contract: Mapping[str, Any],
    *,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
) -> dict[str, Any]:
    return _jit_m0_host_metadata(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        schedule=schedule,
        target=target,
    )


def render_jit_m0_host_plugin(
    contract: Mapping[str, Any],
    *,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
) -> tuple[str, dict[str, Any]]:
    return _render_jit_m0_host_plugin(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        schedule=schedule,
        target=target,
    )


def jit_m1_device_metadata(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> dict[str, Any]:
    return _jit_m1_device_metadata(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        target=target,
        schedule=schedule,
    )


def render_jit_m1_device_module(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> tuple[str, dict[str, Any]]:
    return _render_jit_m1_device_module(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        target=target,
        schedule=schedule,
    )


def jit_m1_host_metadata(
    contract: Mapping[str, Any],
    *,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
) -> dict[str, Any]:
    return _jit_m1_host_metadata(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        schedule=schedule,
        target=target,
    )


def render_jit_m1_host_plugin(
    contract: Mapping[str, Any],
    *,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
) -> tuple[str, dict[str, Any]]:
    return _render_jit_m1_host_plugin(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        precision=precision,
        schedule=schedule,
        target=target,
    )


def jit_r0_device_metadata(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str | None = None,
    floating_point_policy: str | None = None,
    apply_edge_weights: bool = False,
) -> dict[str, Any]:
    return _jit_r0_device_metadata(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        source_content_sha256=None,
        backend="cuda" if str(target).startswith("sm_") else "hip",
        precision=precision,
        target=target,
        schedule=schedule,
        floating_point_policy=floating_point_policy,
        apply_edge_weights=apply_edge_weights,
    )


def jit_r0_host_metadata(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str = "host",
    apply_edge_weights: bool = False,
) -> dict[str, Any]:
    return _jit_r0_host_metadata(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        source_content_sha256=None,
        target=target,
        precision=precision,
        schedule="host-owner",
        apply_edge_weights=apply_edge_weights,
    )


def render_jit_r0_host_plugin(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str = "host",
    apply_edge_weights: bool = False,
) -> tuple[str, dict[str, Any]]:
    return _render_jit_r0_host_plugin(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        source_content_sha256=None,
        target=target,
        precision=precision,
        schedule="host-owner",
        apply_edge_weights=apply_edge_weights,
    )


def render_jit_r0_device_module(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str | None = None,
    floating_point_policy: str | None = None,
    apply_edge_weights: bool = False,
) -> tuple[str, dict[str, Any]]:
    return _render_jit_r0_device_module(
        contract,
        generation_version=JIT_GENERATION_VERSION,
        source_content_sha256=None,
        backend="cuda" if str(target).startswith("sm_") else "hip",
        precision=precision,
        target=target,
        schedule=schedule,
        floating_point_policy=floating_point_policy,
        apply_edge_weights=apply_edge_weights,
    )


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
    """Render R0 from the caller's explicit runtime provenance."""

    return _render_generated_r0_operator(
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


__all__ = [
    "M0_CAPABILITIES_V1",
    "M1_CAPABILITIES_V1",
    "OPERATOR_ARTIFACT_ENVELOPE",
    "R0_CAPABILITIES_V1",
    "jit_m0_device_metadata",
    "jit_m1_device_metadata",
    "jit_m1_host_metadata",
    "jit_r0_device_metadata",
    "jit_r0_host_metadata",
    "render_generated_r0_operator",
    "render_jit_m0_device_module",
    "render_jit_m0_host_plugin",
    "render_jit_m1_device_module",
    "render_jit_m1_host_plugin",
    "render_jit_r0_device_module",
    "render_jit_r0_host_plugin",
    "select_product_device_schedule",
]
