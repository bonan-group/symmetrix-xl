"""Prepare validated Execution R1 host artifacts without an ASE calculator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import symmetrix
from .jit import (
    JIT_GENERATION_VERSION,
    artifact_cache_inputs,
    load_validated_jit_artifact,
    native_source_content_sha256,
    prepare_jit_artifact,
    remove_jit_quarantine,
)
from .mace_jit_artifact_common import (
    native_evaluator,
    read_model_json,
    require_matching_generation_version,
)
from .mace_jit_codegen import (
    R1_ARTIFACT_ENVELOPE,
    jit_r1_host_plugin_metadata,
    render_jit_r1_host_plugin,
)


class JitHostArtifactError(RuntimeError):
    """Raised when an explicitly requested host artifact cannot be prepared."""

    def __init__(self, reason, *, diagnostics=(), cache_key=None):
        super().__init__(str(reason))
        self.diagnostics = tuple(str(value) for value in diagnostics)
        self.cache_key = None if cache_key is None else str(cache_key)


@dataclass(frozen=True)
class JitHostArtifactResult:
    """A compiled, cache-published, and native-loader-validated artifact."""

    status: str
    backend: str
    precision: str
    model_type: str
    model_path: Path
    cache_key: str
    artifact_path: Path
    manifest_path: Path
    artifact_id: str
    environment: dict[str, Any]
    diagnostics: tuple[str, ...]

    @property
    def available(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "available": True,
            "backend": self.backend,
            "precision": self.precision,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "cache_key": self.cache_key,
            "artifact_path": str(self.artifact_path),
            "manifest_path": str(self.manifest_path),
            "artifact_id": self.artifact_id,
            "environment": self.environment,
            "diagnostics": list(self.diagnostics),
        }


def _host_environment() -> dict[str, Any]:
    query = getattr(symmetrix, "_kokkos_default_execution_space", None)
    execution_space = query() if callable(query) else None
    if execution_space not in ("Serial", "OpenMP"):
        raise JitHostArtifactError(
            "Execution host artifacts require a Serial or OpenMP Kokkos build, "
            f"found {execution_space!r}"
        )
    return {"available": True, "backend": "host", "execution_space": execution_space}


def prepare_jit_host_artifact(
    model_file,
    *,
    precision: str,
    target: str = "",
    cache_root=None,
    host_target: str | None = None,
    host_flags=None,
) -> JitHostArtifactResult:
    """Prepare and validate a host Execution R1 shared library for a model JSON.

    ``target`` is the declared artifact adapter's execution target.  ``"host"``
    selects the target this entry point serves and ``""`` keeps the legacy
    adapter-agnostic calling convention; any other target -- the ``"cuda"`` or
    ``"hip"`` target the family's sibling device adapter serves -- is refused by
    name before the model file is read, so a frontend that resolved the wrong
    adapter for a request fails at preparation instead of compiling a host
    library for a device request.  ``host_target`` is unrelated: it is the JIT
    host policy target (for example ``"portable"``) handed to the compiler.
    """

    if target not in ("", "host"):
        raise JitHostArtifactError(
            "the Execution R1 host artifact adapter serves the host target only, "
            f"not {target!r}"
        )
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")

    model_path, model_data = read_model_json(
        model_file, error_type=JitHostArtifactError, label="compact"
    )
    model_type = model_data.get("model_type", "MACE")
    if model_type not in ("MACE", "MACEField", "MACELES"):
        raise JitHostArtifactError(
            "Execution host artifact preparation supports MACE, MACEField, "
            "and MACELES models"
        )
    contracts = model_data.get("execution_contracts")
    if not isinstance(contracts, dict):
        raise JitHostArtifactError(
            "the model does not contain an Execution contracts object"
        )
    contract = contracts.get("R1")
    if not isinstance(contract, dict):
        raise JitHostArtifactError(
            "the model does not contain an Execution R1 contract"
        )

    evaluator = native_evaluator(
        model_path,
        precision,
        model_type=model_type,
        error_type=JitHostArtifactError,
        streamed_edges="direct",
    )
    environment = _host_environment()
    require_matching_generation_version(error_type=JitHostArtifactError)
    try:
        metadata = jit_r1_host_plugin_metadata(contract, precision=precision)
        source = render_jit_r1_host_plugin(contract, precision=precision)
        arguments = {
            "source_content_sha256": native_source_content_sha256(symmetrix),
            "abi": artifact_cache_inputs(metadata, R1_ARTIFACT_ENVELOPE),
            "build": {
                "generator": "symmetrix.jit.r1-host-v2",
                "precision": precision,
                "contract": metadata,
                "jit_generation_version": JIT_GENERATION_VERSION,
            },
            "cxx_flags": None,
            "host_target": host_target,
            "host_flags": host_flags,
            "artifact_name": "factorized_host_plugin",
        }
        if cache_root is not None:
            arguments["cache_root"] = cache_root

        attempt = load_validated_jit_artifact(
            lambda: prepare_jit_artifact(source, **arguments),
            evaluator._load_jit_host_plugin,
            artifact_key=lambda loaded: (
                loaded.cache_key,
                str(loaded.artifact_path),
            ),
        )
        diagnostics = attempt.diagnostics
        if attempt.status == "unavailable":
            raise JitHostArtifactError(
                attempt.result.reason or "the JIT cache did not produce an artifact",
                diagnostics=diagnostics,
                cache_key=attempt.cache_key,
            )
        if attempt.status == "load-failed":
            raise JitHostArtifactError(
                attempt.error,
                diagnostics=diagnostics,
                cache_key=attempt.cache_key,
            ) from attempt.error
        if attempt.status == "exhausted":
            raise JitHostArtifactError(
                "Execution host artifact recovery attempts were exhausted",
                diagnostics=diagnostics,
                cache_key=attempt.cache_key,
            )
        result = attempt.result

        if not bool(getattr(evaluator, "jit_host_plugin_ready", False)):
            raise JitHostArtifactError(
                "the evaluator did not retain the loaded host plugin",
                diagnostics=diagnostics,
                cache_key=result.cache_key,
            )
        for quarantine in attempt.quarantines:
            diagnostics = (*diagnostics, *remove_jit_quarantine(quarantine))

        return JitHostArtifactResult(
            status=result.status,
            backend="host",
            precision=precision,
            model_type=model_type,
            model_path=model_path,
            cache_key=str(result.cache_key),
            artifact_path=result.artifact_path,
            manifest_path=result.manifest_path,
            artifact_id=str(evaluator.jit_host_plugin_artifact_id),
            environment=environment,
            diagnostics=diagnostics,
        )
    except JitHostArtifactError:
        raise
    except Exception as error:
        raise JitHostArtifactError(error) from error


__all__ = [
    "JitHostArtifactError",
    "JitHostArtifactResult",
    "prepare_jit_host_artifact",
]
