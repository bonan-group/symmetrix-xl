"""Prepare content-addressed NequIP-OAM CUDA generated-R1 artifacts.

One generated module owns every interaction layer of one explicit OAM
architecture. Its identity binds the ordered layer/irrep/tensor-product layout,
generated-code identity, target architecture, scalar width, and compiler.
Learned parameters remain runtime inputs, so models with the same generated
architecture share the module. The JIT cache is content-addressed on rendered
source and reuses the result on later calls.

An OAM evaluator on a device execution space has no generic fallback when the
artifact is unavailable. Preparation failure therefore raises
:class:`OAMCudaArtifactError` under every ``SYMMETRIX_JIT_POLICY`` selection.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cuda_codegen_common import (
    DEFAULT_THREADS_PER_BLOCK,
    SUPPORTED_COMPILERS,
    canonical_cuda_target_identity,
)
from .installed_operator_artifact import installed_operator_build_target
from .jit import (
    JIT_DISABLED_POLICY,
    JIT_GENERATION_VERSION,
    JIT_POLICY_ENVIRONMENT_VARIABLE,
    artifact_cache_inputs,
    direct_artifact_diagnostic,
    prepare_execution_cuda_jit_artifact,
    prepare_nvrtc_jit_artifact,
    resolve_jit_policy,
    select_execution_cuda_jit_backend,
)
from .nequip_schema import load_nequip_oam_data
from .oam_cuda_codegen import (
    GENERATOR as OAM_CUDA_ARTIFACT_GENERATOR,
)
from .oam_cuda_codegen import (
    OAM_CUDA_ARTIFACT_ENVELOPE,
    OAM_CUDA_NVCC_ARTIFACT_NAME,
    OAM_CUDA_NVRTC_ARTIFACT_NAME,
    OAMCudaGenerationPolicy,
    oam_cuda_artifact_metadata,
    render_oam_cuda_plugin,
)
from .oam_cuda_codegen import (
    SCHEMA as OAM_CUDA_SCHEMA,
)
from .oam_cuda_codegen import (
    SCHEMA_VERSION as OAM_CUDA_SCHEMA_VERSION,
)
from .oam_installed_artifact import resolve_installed_oam_artifact

# Published identity fields: the native loader compares descriptor fields;
# generation is bound by the artifact digest and enclosing JIT cache key.
# The manifest carries all of them to reconstruct the loader expectation and
# validate cache entries without rerunning the generator.
OAM_CUDA_EXPECTATION_FIELDS = (
    "artifact_id",
    "contract_fingerprint",
    "semantic_fingerprint",
    "structure_fingerprint",
    "generation_fingerprint",
    "generation_policy",
    "target_architecture",
    "target_compute_capability",
    "max_threads_per_block",
    "scalar_kind",
    "scalar_size",
    "layer_count",
)


class OAMCudaArtifactError(RuntimeError):
    """Raised when an OAM CUDA artifact cannot be prepared."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[str, ...] = (),
        artifact_id: str | None = None,
        cache_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = tuple(str(item) for item in diagnostics)
        self.artifact_id = None if artifact_id is None else str(artifact_id)
        self.cache_key = None if cache_key is None else str(cache_key)


@dataclass(frozen=True)
class OAMCudaArtifactResult:
    """Outcome of binding an OAM contract to the generated CUDA cache."""

    status: str
    artifact_id: str
    artifact_path: Path | None
    manifest_path: Path | None
    compiler: str
    target: Any
    diagnostics: tuple[str, ...]
    fallback_reason: str | None = None
    cache_key: str | None = None

    @property
    def artifact_origin(self) -> str:
        if self.status == "installed":
            return "prebuilt"
        if self.status == "cached":
            return "cache"
        return "runtime"

    @property
    def available(self) -> bool:
        return self.status in ("built", "cached", "installed")

    @property
    def cache_hit(self) -> bool:
        return self.status == "cached"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "available": self.available,
            "artifact_id": self.artifact_id,
            "artifact_path": (
                None if self.artifact_path is None else str(self.artifact_path)
            ),
            "manifest_path": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
            "compiler": self.compiler,
            "target": (
                self.target if isinstance(self.target, str) else str(self.target)
            ),
            "cache_key": self.cache_key,
            "diagnostics": list(self.diagnostics),
            "fallback_reason": self.fallback_reason,
            "artifact_origin": self.artifact_origin,
        }


def _validated_compiler(compiler: Any) -> str | None:
    """Validate an explicit compiler argument without probing the backend."""

    if compiler is None:
        return None
    if compiler not in SUPPORTED_COMPILERS:
        raise ValueError(f"compiler must be one of {list(SUPPORTED_COMPILERS)} or None")
    return compiler


def _selected_compiler(compiler: str | None) -> str:
    """Resolve an explicit compiler or the shared CUDA JIT backend policy."""

    if compiler is not None:
        return compiler
    return select_execution_cuda_jit_backend()


def _bound_identity(
    generated: Mapping[str, Any],
    precision: str,
    compiler: str,
) -> dict[str, Any]:
    """Return the OAM identity block recorded in the cache manifest.

    Every field the native loader validates against the generated descriptor
    -- the artifact id, the architecture/semantic/structure fingerprints, the
    target, the scalar width, and the layer count -- plus the cache generation
    fingerprint are copied from the generated metadata, so an entry cannot
    disagree with its plugin.
    """

    return {
        "schema": OAM_CUDA_SCHEMA,
        "version": OAM_CUDA_SCHEMA_VERSION,
        "artifact_id": generated["artifact_id"],
        "contract_fingerprint": generated["contract_fingerprint"],
        "semantic_fingerprint": generated["semantic_fingerprint"],
        "structure_fingerprint": generated["structure_fingerprint"],
        "generation_fingerprint": generated["generation_fingerprint"],
        "generation_policy": generated["generation_policy"],
        "schedule_qualification": generated["schedule_qualification"],
        "execution_variants": generated["execution_variants"],
        "target_policy": generated["target_policy"],
        "precision_policy": generated["precision_policy"],
        "architecture": generated["target_architecture"],
        "target_architecture": generated["target_architecture"],
        "target_compute_capability": generated["target_compute_capability"],
        "precision": precision,
        "compiler": compiler,
        "jit_generation_version": JIT_GENERATION_VERSION,
        "scalar_kind": generated["scalar_kind"],
        "scalar_size": generated["scalar_size"],
        "max_threads_per_block": generated["max_threads_per_block"],
        "forward_owner_schedule": generated["forward_owner_schedule"],
        "forward_channel_policy": generated["forward_channel_policy"],
        "forward_static_shared_bytes": generated["forward_static_shared_bytes"],
        "reverse_owner_schedule": generated["reverse_owner_schedule"],
        "reverse_channel_policy": generated["reverse_channel_policy"],
        "reverse_static_shared_bytes": generated["reverse_static_shared_bytes"],
        "forward_threads_per_block": generated["forward_threads_per_block"],
        "required_capabilities": generated["required_capabilities"],
        "layer_count": generated["layer_count"],
        "layer_schedule": [dict(row) for row in generated["layers"]],
    }


def _manifest_identity_mismatch(
    manifest: Any, expected: Mapping[str, Any]
) -> str | None:
    """Return a diagnostic when the published identity is not the requested one."""

    if not isinstance(manifest, Mapping):
        return "the JIT cache did not publish a manifest"
    key_inputs = manifest.get("key_inputs")
    if not isinstance(key_inputs, Mapping):
        return "the JIT manifest does not record its cache key inputs"
    build = key_inputs.get("build")
    if not isinstance(build, Mapping):
        return "the JIT manifest does not record its build identity"
    # The CUDA compiler adapters nest the caller's build request below their
    # own backend identity; accept either shape so the check tracks the
    # request.
    request = build.get("request")
    recorded = request if isinstance(request, Mapping) else build
    published = recorded.get("oam")
    if not isinstance(published, Mapping):
        return "the cached OAM artifact identity is missing"
    missing = [name for name in OAM_CUDA_EXPECTATION_FIELDS if name not in published]
    if missing:
        return (
            "the cached artifact identity does not match the requested architecture: "
            f"missing {', '.join(missing)}"
        )
    if published != expected:
        changed = sorted(
            name
            for name in set(published) | set(expected)
            if published.get(name) != expected.get(name)
        )
        return (
            "the cached artifact identity does not match the requested architecture: "
            f"{', '.join(changed) if changed else 'identity'}"
        )
    return None


def prepare_nequip_oam_cuda_artifact(
    contract_path: str | Path,
    precision: str,
    target: Any,
    compiler: str | None = None,
    *,
    cache_root: str | os.PathLike[str] | None = None,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    generation_policy: OAMCudaGenerationPolicy | None = None,
    nvrtc_options=(),
    nvcc: str | os.PathLike[str] | None = None,
    nvcc_flags=(),
    native_module: Any | None = None,
    installed_prebuilt_required: bool = False,
    expected_target: str | None = None,
    expected_artifact_id: str | None = None,
) -> OAMCudaArtifactResult:
    """Prepare and cache one architecture-bound OAM CUDA generated artifact.

    ``target`` accepts the CUDA target representations understood by the
    contract identity (``"sm_120"``, ``"compute_120"``, or a mapping carrying
    ``architecture``/``compute_capability`` such as the evaluator's device
    execution environment).  ``compiler`` overrides the shared CUDA JIT
    policy with an explicit ``"nvrtc"`` or ``"nvcc"`` path.

    Raises :class:`OAMCudaArtifactError` for every unavailable path: an OAM
    evaluator on a device execution space has no valid fallback.
    """

    from .oam_cuda_codegen import _normalize_precision, _threads_per_block

    precision = _normalize_precision(precision)
    threads = _threads_per_block(threads_per_block)
    if installed_prebuilt_required and native_module is None:
        raise OAMCudaArtifactError(
            "required installed OAM CUDA artifact needs a native module and cannot "
            "probe a runtime compiler or generate an artifact"
        )
    if not installed_prebuilt_required and any(
        value is not None for value in (expected_target, expected_artifact_id)
    ):
        raise OAMCudaArtifactError(
            "installed CUDA artifact expectations require installed-prebuilt-required"
        )

    path = Path(contract_path).expanduser().resolve()
    try:
        contract = load_nequip_oam_data(path)
    except (OSError, ValueError) as error:
        raise OAMCudaArtifactError(
            f"could not read the OAM contract {path}: {error}"
        ) from error
    if str(contract.get("architecture", "")) != "oam":
        raise OAMCudaArtifactError(f"the contract {path} is not a NequIP-OAM v2 model")

    try:
        generated = oam_cuda_artifact_metadata(
            contract,
            precision=precision,
            target=target,
            threads_per_block=threads,
            generation_policy=generation_policy,
        )
    except Exception as error:
        raise OAMCudaArtifactError(
            f"the OAM CUDA code generator rejected the contract: "
            f"{type(error).__name__}: {error}",
            artifact_id=None,
        ) from error

    artifact_id = str(generated["artifact_id"])
    generated_target = str(generated["target_architecture"])
    if installed_prebuilt_required:
        canonical_generated_target = canonical_cuda_target_identity(generated_target)
        if (
            expected_target is not None
            and canonical_cuda_target_identity(expected_target)
            != canonical_generated_target
        ):
            raise OAMCudaArtifactError(
                "required installed OAM CUDA artifact expected CUDA target "
                f"{expected_target!r}, but the generated request is "
                f"{generated_target!r}; regenerate the approved prebuilt"
            )
        installed_target = installed_operator_build_target(
            native_module=native_module, backend="cuda"
        )
        if installed_target is None:
            raise OAMCudaArtifactError(
                "required installed OAM CUDA artifact is absent: the native build "
                "does not declare an installed CUDA catalog"
            )
        required_target = expected_target or generated_target
        if canonical_cuda_target_identity(
            installed_target
        ) != canonical_cuda_target_identity(required_target):
            raise OAMCudaArtifactError(
                "required installed OAM CUDA artifact expected CUDA target "
                f"{required_target!r}, but the native build declares "
                f"{installed_target!r}; regenerate the approved prebuilt"
            )
        if expected_artifact_id is not None and expected_artifact_id != artifact_id:
            raise OAMCudaArtifactError(
                "required installed OAM CUDA artifact identity mismatch: expected "
                f"{expected_artifact_id!r}, generated request is {artifact_id!r}; "
                "regenerate the approved prebuilt"
            )
        installed = resolve_installed_oam_artifact(
            architecture=generated["architecture"],
            precision=precision,
            backend="cuda",
            target=generated_target,
            native_module=native_module,
            artifact_id=artifact_id,
            required_capabilities=generated["required_capabilities"],
        )
        if installed.available:
            return OAMCudaArtifactResult(
                status="installed",
                artifact_id=artifact_id,
                artifact_path=installed.artifact_path,
                manifest_path=installed.manifest_path,
                compiler="prebuilt",
                target=generated_target,
                diagnostics=installed.diagnostics,
                fallback_reason=None,
                cache_key=None,
            )
        detail = "; ".join(installed.diagnostics)
        raise OAMCudaArtifactError(
            "required installed OAM CUDA artifact was not resolved "
            f"(status={installed.status}); {detail}; regenerate or install "
            "the exact approved prebuilt",
            artifact_id=artifact_id,
            diagnostics=installed.diagnostics,
        )

    policy = resolve_jit_policy()
    compiler = _validated_compiler(compiler)
    if native_module is not None:
        installed = resolve_installed_oam_artifact(
            architecture=generated["architecture"],
            precision=precision,
            backend="cuda",
            target=generated_target,
            native_module=native_module,
            artifact_id=artifact_id,
            required_capabilities=generated["required_capabilities"],
        )
        if installed.available:
            return OAMCudaArtifactResult(
                status="installed",
                artifact_id=artifact_id,
                artifact_path=installed.artifact_path,
                manifest_path=installed.manifest_path,
                compiler="prebuilt",
                target=target,
                diagnostics=installed.diagnostics,
                fallback_reason=None,
                cache_key=None,
            )

    try:
        selected = _selected_compiler(compiler)
    except ValueError:
        raise
    except Exception as error:
        raise OAMCudaArtifactError(
            f"the CUDA JIT backend is unavailable: {type(error).__name__}: {error}"
            + direct_artifact_diagnostic(
                artifact_rejected=False,
                cache_key=None,
                artifact_path=None,
                backend="cuda",
            ),
        ) from error

    if policy == JIT_DISABLED_POLICY:
        raise OAMCudaArtifactError(
            (
                f"{JIT_POLICY_ENVIRONMENT_VARIABLE}={JIT_DISABLED_POLICY} disables "
                "generated OAM CUDA R1 artifact preparation, and OAM has no "
                "fallback execution on a CUDA execution space"
                + direct_artifact_diagnostic(
                    artifact_rejected=False,
                    cache_key=None,
                    artifact_path=None,
                    backend="cuda",
                )
            ),
            artifact_id=artifact_id,
        )

    try:
        source = render_oam_cuda_plugin(
            contract,
            target,
            precision=precision,
            threads_per_block=threads,
            generation_policy=generation_policy,
        )
    except Exception as error:
        raise OAMCudaArtifactError(
            f"the OAM CUDA code generator failed to render the plugin: "
            f"{type(error).__name__}: {error}",
            artifact_id=artifact_id,
        ) from error

    expected = _bound_identity(
        generated,
        precision,
        selected,
    )
    abi = artifact_cache_inputs(generated, OAM_CUDA_ARTIFACT_ENVELOPE)
    build = {
        "generator": OAM_CUDA_ARTIFACT_GENERATOR,
        "precision": precision,
        "abi": abi,
        # Keep runtime artifacts on the same metadata envelope as install-time
        # OAM artifacts so custom architecture and schedule identities remain
        # available to cache admission.
        "metadata": generated,
        "oam": expected,
    }
    arguments: dict[str, Any] = {
        "abi": abi,
        "build": build,
        "compute_capability": target,
    }
    if cache_root is not None:
        arguments["cache_root"] = cache_root
    if selected == "nvrtc":
        prepare = prepare_nvrtc_jit_artifact
        arguments["nvrtc_options"] = nvrtc_options
        arguments["artifact_name"] = OAM_CUDA_NVRTC_ARTIFACT_NAME
    else:
        prepare = prepare_execution_cuda_jit_artifact
        arguments["nvcc"] = nvcc
        arguments["nvcc_flags"] = nvcc_flags
        arguments["artifact_name"] = OAM_CUDA_NVCC_ARTIFACT_NAME

    result = prepare(source, **arguments)
    diagnostics = tuple(result.diagnostics)
    if not result.available:
        raise OAMCudaArtifactError(
            (result.reason or "the JIT cache did not produce an OAM CUDA artifact")
            + direct_artifact_diagnostic(
                artifact_rejected=False,
                cache_key=result.cache_key,
                artifact_path=result.artifact_path,
                backend="cuda",
            ),
            artifact_id=artifact_id,
            diagnostics=diagnostics,
            cache_key=result.cache_key,
        )

    mismatch = _manifest_identity_mismatch(result.manifest, expected)
    if mismatch is not None:
        raise OAMCudaArtifactError(
            mismatch
            + direct_artifact_diagnostic(
                artifact_rejected=True,
                cache_key=result.cache_key,
                artifact_path=result.artifact_path,
                backend="cuda",
            ),
            artifact_id=artifact_id,
            diagnostics=diagnostics,
            cache_key=result.cache_key,
        )

    return OAMCudaArtifactResult(
        status=result.status,
        artifact_id=artifact_id,
        artifact_path=result.artifact_path,
        manifest_path=result.manifest_path,
        compiler=selected,
        target=target,
        diagnostics=diagnostics,
        fallback_reason=None,
        cache_key=result.cache_key,
    )


__all__ = [
    "OAMCudaArtifactError",
    "OAMCudaArtifactResult",
    "prepare_nequip_oam_cuda_artifact",
]
