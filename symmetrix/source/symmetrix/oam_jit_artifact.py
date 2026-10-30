"""Prepare cached generated host plugins for NequIP-OAM contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .installed_operator_artifact import installed_operator_build_target
from .jit import (
    JIT_GENERATION_VERSION,
    JitError,
    active_host_target_for,
    artifact_cache_inputs,
    canonical_host_artifact_target,
    direct_artifact_diagnostic,
    prepare_jit_artifact,
    probe_compiler_identity,
    resolve_host_floating_point_policy,
    resolve_host_jit_policy,
)
from .oam_installed_artifact import resolve_installed_oam_artifact
from .oam_jit_codegen import (
    OAM_HOST_ARTIFACT_ENVELOPE,
    oam_jit_metadata,
    render_oam_host_plugin,
)
from .oam_receiver_codegen import (
    OAM_RECEIVER_ARTIFACT_ENVELOPE,
    OAMHostGenerationPolicy,
    canonical_oam_host_floating_point_policy,
    oam_receiver_jit_metadata,
    render_oam_receiver_host_plugin,
)


class OAMHostArtifactError(RuntimeError):
    """Raised when an OAM host artifact cannot be generated or cached."""


@dataclass(frozen=True)
class OAMHostArtifactResult:
    status: str
    precision: str
    model_path: Path
    artifact_path: Path
    manifest_path: Path | None
    cache_key: str
    artifact_id: str
    semantic_fingerprint: str
    structure_fingerprint: str
    generation_fingerprint: str
    metadata: dict[str, Any]
    diagnostics: tuple[str, ...]
    cache_hit: bool
    compiler_command: tuple[str, ...]
    artifact_origin: str
    active_host_target: str
    expected_host_target: str | None = None
    expected_artifact_origin: str | None = None
    floating_point_policy: str | None = None
    expected_artifact_id: str | None = None

    @property
    def identity(self) -> dict[str, str]:
        """Artifact identity the native loader compares with its architecture."""

        return {
            "artifact_id": self.artifact_id,
            "semantic_fingerprint": self.semantic_fingerprint,
            "structure_fingerprint": self.structure_fingerprint,
            "generation_fingerprint": self.generation_fingerprint,
        }


def _verify_cached_identity(
    contract_path: Path, manifest: Any, expected: dict[str, str]
) -> None:
    """Refuse a cache hit whose manifest was built for another architecture.

    The cache key already includes the build metadata, so a mismatch means the
    published entry was tampered with or the key composition changed.  Either
    way the artifact must not be returned as if it matched the current
    architecture.
    """

    try:
        recorded = manifest["key_inputs"]["build"]["metadata"]
    except (KeyError, TypeError) as error:
        raise OAMHostArtifactError(
            f"cached OAM host artifact for {contract_path} has no build identity "
            "metadata to compare against the current architecture"
        ) from error
    if not isinstance(recorded, dict):
        raise OAMHostArtifactError(
            f"cached OAM host artifact for {contract_path} has malformed build "
            "identity metadata"
        )
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise OAMHostArtifactError(
                f"cached OAM host artifact for {contract_path} was built for a "
                f"different architecture: manifest {key}={recorded.get(key)!r} "
                f"but the current architecture requires {value!r}"
            )


def prepare_nequip_oam_host_artifact(
    contract_path: str | Path,
    precision: str = "float64",
    *,
    target: str = "host",
    cache_dir: str | Path | None = None,
    host_target: str | None = None,
    host_flags: str | tuple[str, ...] | None = None,
    native_module: Any | None = None,
) -> OAMHostArtifactResult:
    """Prepare the edge-owned ABI v1 artifact used by explicit debug/reference runs."""

    return _prepare(
        contract_path,
        precision,
        target=target,
        cache_dir=cache_dir,
        host_target=host_target,
        host_flags=host_flags,
        native_module=native_module,
        metadata_of=oam_jit_metadata,
        render=render_oam_host_plugin,
        envelope=OAM_HOST_ARTIFACT_ENVELOPE,
        artifact_name="oam_host_plugin",
        generation_policy=None,
    )


def prepare_nequip_oam_receiver_host_artifact(
    contract_path: str | Path,
    precision: str = "float64",
    *,
    target: str = "host",
    cache_dir: str | Path | None = None,
    host_target: str | None = None,
    host_flags: str | tuple[str, ...] | None = None,
    native_module: Any | None = None,
    generation_policy: OAMHostGenerationPolicy | None = None,
    installed_prebuilt_required: bool = False,
    expected_host_target: str | None = None,
    expected_artifact_id: str | None = None,
    expected_floating_point_policy: str | None = None,
) -> OAMHostArtifactResult:
    """Prepare the receiver-forward/source-reverse ABI v7 direct artifact."""

    return _prepare(
        contract_path,
        precision,
        target=target,
        cache_dir=cache_dir,
        host_target=host_target,
        host_flags=host_flags,
        native_module=native_module,
        metadata_of=oam_receiver_jit_metadata,
        render=render_oam_receiver_host_plugin,
        envelope=OAM_RECEIVER_ARTIFACT_ENVELOPE,
        artifact_name="oam_receiver_host_plugin",
        generation_policy=generation_policy,
        installed_prebuilt_required=installed_prebuilt_required,
        expected_host_target=expected_host_target,
        expected_artifact_id=expected_artifact_id,
        expected_floating_point_policy=expected_floating_point_policy,
    )


def _prepare(
    contract_path: str | Path,
    precision: str,
    *,
    target: str,
    cache_dir: str | Path | None,
    host_target: str | None,
    host_flags: str | tuple[str, ...] | None,
    native_module: Any | None,
    metadata_of,
    render,
    envelope,
    artifact_name,
    generation_policy,
    installed_prebuilt_required: bool = False,
    expected_host_target: str | None = None,
    expected_artifact_id: str | None = None,
    expected_floating_point_policy: str | None = None,
) -> OAMHostArtifactResult:
    if target != "host":
        raise OAMHostArtifactError(
            f"the OAM host artifact adapter serves the host target only, not {target!r}"
        )
    path = Path(contract_path).expanduser().resolve()
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OAMHostArtifactError(
            f"could not read OAM contract {path}: {error}"
        ) from error
    if installed_prebuilt_required and metadata_of is not oam_receiver_jit_metadata:
        raise OAMHostArtifactError(
            "installed-prebuilt-required is supported only for receiver OAM artifacts"
        )
    if installed_prebuilt_required and native_module is None:
        raise OAMHostArtifactError(
            "required installed OAM host artifact needs a native module and cannot "
            "probe a compiler or generate a runtime artifact"
        )
    if installed_prebuilt_required and generation_policy is not None:
        raise OAMHostArtifactError(
            "required installed OAM host artifacts use their catalog-bound "
            "generation policy and cannot request runtime generation"
        )
    if not installed_prebuilt_required and any(
        value is not None
        for value in (
            expected_host_target,
            expected_artifact_id,
            expected_floating_point_policy,
        )
    ):
        raise OAMHostArtifactError(
            "installed artifact expectations require installed-prebuilt-required; "
            "runtime policy is derived solely from effective compile flags"
        )
    requested_target = (
        canonical_host_artifact_target(expected_host_target)
        if expected_host_target is not None
        else None
    )
    requested_fp_policy = (
        canonical_oam_host_floating_point_policy(expected_floating_point_policy)
        if expected_floating_point_policy is not None
        else None
    )
    if (
        native_module is not None
        and metadata_of is oam_receiver_jit_metadata
        and generation_policy is None
    ):
        installed_target = installed_operator_build_target(
            native_module=native_module, backend="host"
        )
        if installed_prebuilt_required and installed_target is None:
            raise OAMHostArtifactError(
                "required installed OAM host artifact is absent: the native build "
                "does not declare an installed host catalog; regenerate or install "
                "the approved prebuilt"
            )
        if installed_target is not None:
            required_target = canonical_host_artifact_target(installed_target)
            if requested_target is not None and required_target != requested_target:
                raise OAMHostArtifactError(
                    "required installed OAM host artifact expected host target "
                    f"{requested_target!r}, but the native build declares "
                    f"{required_target!r}; regenerate the approved target"
                )
            build_info = native_module._backend_build_info()
            if installed_prebuilt_required and not isinstance(
                build_info.get("fast_math"), bool
            ):
                raise OAMHostArtifactError(
                    "required installed OAM host artifact cannot validate its "
                    "floating-point policy because native fast_math build info is "
                    "missing; rebuild the native extension"
                )
            actual_fp_policy = (
                "fast" if build_info.get("fast_math") is True else "strict"
            )
            if (
                requested_fp_policy is not None
                and actual_fp_policy != requested_fp_policy
            ):
                raise OAMHostArtifactError(
                    "required installed OAM host artifact floating-point policy "
                    f"mismatch: expected {requested_fp_policy!r}, native build is "
                    f"{actual_fp_policy!r}; regenerate with matching fast_math"
                )
            try:
                active_target = active_host_target_for(required_target)
            except JitError as error:
                raise OAMHostArtifactError(
                    f"installed OAM host target is incompatible: {error}; regenerate "
                    "or install a compatible artifact"
                ) from error
            metadata = metadata_of(
                contract,
                precision,
                host_target=required_target,
                generation_policy=generation_policy,
                floating_point_policy=actual_fp_policy,
            )
            if (
                expected_artifact_id is not None
                and metadata["artifact_id"] != expected_artifact_id
            ):
                raise OAMHostArtifactError(
                    "required installed OAM host artifact identity mismatch: expected "
                    f"{expected_artifact_id!r}, generated request is "
                    f"{metadata['artifact_id']!r}; regenerate the approved prebuilt"
                )
            installed = resolve_installed_oam_artifact(
                architecture=metadata["architecture"],
                precision=precision,
                backend="host",
                native_module=native_module,
                host_target=required_target,
                floating_point_policy=actual_fp_policy,
                artifact_id=str(metadata["artifact_id"]),
                required_capabilities=metadata["required_capabilities"],
            )
            if installed.available:
                identity = {
                    "artifact_id": str(metadata["artifact_id"]),
                    "generation_fingerprint": str(metadata["generation_fingerprint"]),
                    "semantic_fingerprint": str(metadata["semantic_fingerprint"]),
                    "structure_fingerprint": str(metadata["structure_fingerprint"]),
                }
                return OAMHostArtifactResult(
                    status="installed",
                    precision=precision,
                    model_path=path,
                    artifact_path=installed.artifact_path,
                    manifest_path=installed.manifest_path,
                    cache_key="",
                    artifact_id=identity["artifact_id"],
                    semantic_fingerprint=identity["semantic_fingerprint"],
                    structure_fingerprint=identity["structure_fingerprint"],
                    generation_fingerprint=identity["generation_fingerprint"],
                    metadata=metadata,
                    diagnostics=installed.diagnostics,
                    cache_hit=False,
                    compiler_command=(),
                    artifact_origin="prebuilt",
                    active_host_target=active_target,
                    expected_host_target=requested_target,
                    expected_artifact_origin=(
                        "prebuilt" if installed_prebuilt_required else None
                    ),
                    floating_point_policy=actual_fp_policy,
                    expected_artifact_id=str(metadata["artifact_id"]),
                )
            if installed_prebuilt_required:
                detail = "; ".join(installed.diagnostics)
                raise OAMHostArtifactError(
                    "required installed OAM host artifact was not resolved "
                    f"(status={installed.status}); {detail}; regenerate or install "
                    "the exact approved prebuilt"
                )
    try:
        compiler_command, _ = probe_compiler_identity()
        host_policy = resolve_host_jit_policy(
            compiler_command, target=host_target, flags=host_flags
        )
        runtime_fp_policy = (
            resolve_host_floating_point_policy(host_policy.flags)
            if metadata_of is oam_receiver_jit_metadata
            else None
        )
        active_target = active_host_target_for(host_policy.effective_target)
    except JitError as error:
        raise OAMHostArtifactError(
            str(error)
            + direct_artifact_diagnostic(
                artifact_rejected=False,
                cache_key=None,
                artifact_path=None,
                backend=target,
            )
        ) from error
    if metadata_of is oam_receiver_jit_metadata:
        metadata = metadata_of(
            contract,
            precision,
            host_target=host_policy.effective_target,
            generation_policy=generation_policy,
            floating_point_policy=runtime_fp_policy,
        )
    else:
        metadata = metadata_of(contract, precision)
    source = (
        render(
            contract,
            precision,
            host_target=metadata["host_target"],
            generation_policy=generation_policy,
            floating_point_policy=metadata["floating_point_policy"],
        )
        if metadata_of is oam_receiver_jit_metadata
        else render(contract, precision)
    )
    result = prepare_jit_artifact(
        source,
        abi=artifact_cache_inputs(metadata, envelope),
        build={
            "generator": metadata.get("generator", "symmetrix.jit.oam-host-v1"),
            "precision": precision,
            "metadata": metadata,
            "jit_generation_version": JIT_GENERATION_VERSION,
        },
        cache_root=cache_dir,
        host_policy=host_policy,
        cxx_flags=None,
        artifact_name=artifact_name,
    )
    if (
        not result.available
        or result.artifact_path is None
        or result.manifest_path is None
    ):
        # Direct execution does not fall back, so the preparation reports which
        # of the four documented artifact causes applies to this attempt.
        raise OAMHostArtifactError(
            (result.reason or "OAM host artifact compilation failed")
            + direct_artifact_diagnostic(
                artifact_rejected=False,
                cache_key=result.cache_key,
                artifact_path=result.artifact_path,
                backend=target,
            )
        )
    identity = {
        "artifact_id": str(metadata["artifact_id"]),
        "generation_fingerprint": str(metadata["generation_fingerprint"]),
        "semantic_fingerprint": str(metadata["semantic_fingerprint"]),
        "structure_fingerprint": str(metadata["structure_fingerprint"]),
    }
    if result.cache_hit:
        # A cache hit must describe this exact architecture, not merely an artifact
        # whose recorded identity happens to match its own shared object.
        try:
            _verify_cached_identity(path, result.manifest, identity)
        except OAMHostArtifactError as error:
            raise OAMHostArtifactError(
                f"{error}"
                + direct_artifact_diagnostic(
                    artifact_rejected=True,
                    cache_key=result.cache_key,
                    artifact_path=result.artifact_path,
                    backend=target,
                )
            ) from error
    return OAMHostArtifactResult(
        status=result.status,
        precision=precision,
        model_path=path,
        artifact_path=result.artifact_path,
        manifest_path=result.manifest_path,
        cache_key=str(result.cache_key),
        artifact_id=identity["artifact_id"],
        semantic_fingerprint=identity["semantic_fingerprint"],
        structure_fingerprint=identity["structure_fingerprint"],
        generation_fingerprint=identity["generation_fingerprint"],
        metadata=metadata,
        diagnostics=tuple(result.diagnostics),
        cache_hit=result.cache_hit,
        compiler_command=tuple(result.command),
        artifact_origin="cache" if result.cache_hit else "runtime",
        active_host_target=active_target,
        expected_host_target=requested_target,
        expected_artifact_origin=("prebuilt" if installed_prebuilt_required else None),
        floating_point_policy=metadata.get("floating_point_policy"),
        expected_artifact_id=(expected_artifact_id or str(metadata["artifact_id"])),
    )


__all__ = [
    "OAMHostArtifactError",
    "OAMHostArtifactResult",
    "prepare_nequip_oam_host_artifact",
    "prepare_nequip_oam_receiver_host_artifact",
]
