"""Prepare validated Execution R1 host artifacts without an ASE calculator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import symmetrix
from .jit import (
    JIT_GENERATION_VERSION,
    prepare_jit_artifact,
    quarantine_jit_artifact,
    remove_jit_quarantine,
)
from .jit_codegen import jit_r1_host_plugin_metadata, render_jit_r1_host_plugin


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


def _read_model_json(model_file) -> tuple[Path, dict[str, Any]]:
    model_path = Path(model_file).expanduser().resolve()
    try:
        with model_path.open(encoding="utf-8") as stream:
            model_data = json.load(stream)
    except (OSError, TypeError, ValueError) as error:
        raise JitHostArtifactError(
            f"could not read compact Symmetrix model JSON {model_path}: {error}"
        ) from error
    if not isinstance(model_data, dict):
        raise JitHostArtifactError(
            "compact Symmetrix model JSON must contain an object"
        )
    return model_path, model_data


def _native_evaluator(model_path: Path, precision: str):
    evaluator_name = "MACEKokkos" if precision == "float64" else "MACEKokkosFloat"
    evaluator_class = getattr(symmetrix, evaluator_name, None)
    if evaluator_class is None:
        raise JitHostArtifactError(
            "this Symmetrix build does not provide the requested Kokkos evaluator"
        )
    try:
        if not symmetrix._kokkos_is_initialized():
            symmetrix._init_kokkos()
        evaluator = evaluator_class(str(model_path))
        evaluator.set_streamed_edges("direct")
    except Exception as error:
        raise JitHostArtifactError(
            f"could not construct the native Execution R1 evaluator: {error}"
        ) from error
    return evaluator


def _host_environment() -> dict[str, Any]:
    query = getattr(symmetrix, "_kokkos_default_execution_space", None)
    execution_space = query() if callable(query) else None
    if execution_space not in ("Serial", "OpenMP"):
        raise JitHostArtifactError(
            "Execution host artifacts require a Serial or OpenMP Kokkos build, "
            f"found {execution_space!r}"
        )
    return {"available": True, "backend": "host", "execution_space": execution_space}


def _require_matching_generation_version() -> None:
    query = getattr(symmetrix, "_required_jit_generation_version", None)
    if not callable(query):
        raise JitHostArtifactError(
            "the native Symmetrix extension predates JIT generation versioning"
        )
    required = query()
    if required != JIT_GENERATION_VERSION:
        raise JitHostArtifactError(
            "JIT generation version mismatch: native standard modules require "
            f"{required}, but Python generators provide {JIT_GENERATION_VERSION}"
        )


def prepare_jit_host_artifact(
    model_file,
    *,
    precision: str,
    cache_root=None,
    host_target: str | None = None,
    host_flags=None,
) -> JitHostArtifactResult:
    """Prepare and validate a host Execution R1 shared library for a model JSON."""

    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")

    model_path, model_data = _read_model_json(model_file)
    model_type = str(model_data.get("model_type", "MACE"))
    if model_type not in ("MACE", "MACEField"):
        raise JitHostArtifactError(
            "Execution host artifact preparation supports MACE and MACEField models"
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

    evaluator = _native_evaluator(model_path, precision)
    environment = _host_environment()
    _require_matching_generation_version()
    try:
        metadata = jit_r1_host_plugin_metadata(contract, precision=precision)
        source = render_jit_r1_host_plugin(contract, precision=precision)
        arguments = {
            "abi": {"tag": metadata["abi"], "version": metadata["abi_version"]},
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

        diagnostics: tuple[str, ...] = ()
        retained_quarantines = []
        failed_load_artifacts = set()
        result = None
        for _load_attempt in range(4):
            result = prepare_jit_artifact(source, **arguments)
            diagnostics = (*diagnostics, *result.diagnostics)
            if not result.available or result.artifact_path is None:
                raise JitHostArtifactError(
                    result.reason or "the JIT cache did not produce an artifact",
                    diagnostics=diagnostics,
                    cache_key=result.cache_key,
                )
            try:
                evaluator._load_jit_host_plugin(str(result.artifact_path))
                break
            except Exception as load_error:
                failed_artifact = (result.cache_key, str(result.artifact_path))
                if failed_artifact in failed_load_artifacts:
                    raise JitHostArtifactError(
                        load_error,
                        diagnostics=diagnostics,
                        cache_key=result.cache_key,
                    ) from load_error
                failed_load_artifacts.add(failed_artifact)
                quarantine = quarantine_jit_artifact(
                    result, f"{type(load_error).__name__}: {load_error}"
                )
                retained_quarantines.append(quarantine)
                diagnostics = (*diagnostics, *quarantine.diagnostics)
        else:
            raise JitHostArtifactError(
                "Execution host artifact recovery attempts were exhausted",
                diagnostics=diagnostics,
                cache_key=None if result is None else result.cache_key,
            )

        if not bool(getattr(evaluator, "jit_host_plugin_ready", False)):
            raise JitHostArtifactError(
                "the evaluator did not retain the loaded host plugin",
                diagnostics=diagnostics,
                cache_key=result.cache_key,
            )
        for quarantine in retained_quarantines:
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
