"""Prepare validated direct-execution device artifacts without an ASE calculator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import symmetrix
from .calculator import _factorized_device_backend_plan
from .jit import (
    JIT_GENERATION_VERSION,
    quarantine_jit_artifact,
    remove_jit_quarantine,
)
from .jit_operator_artifact import prepare_low_memory_operator_modules


class JitDeviceArtifactError(RuntimeError):
    """Raised when an explicitly requested device artifact cannot be prepared."""

    def __init__(self, reason, *, diagnostics=(), cache_key=None):
        super().__init__(str(reason))
        self.diagnostics = tuple(str(value) for value in diagnostics)
        self.cache_key = None if cache_key is None else str(cache_key)


@dataclass(frozen=True)
class JitDeviceArtifactResult:
    """Validated Execution R1 artifact and optional low-memory operator modules."""

    status: str
    backend: str
    compiler_request: str
    compiler: str
    precision: str
    model_type: str
    model_path: Path
    cache_key: str
    artifact_path: Path
    manifest_path: Path
    artifact_id: str
    variant_id: str | None
    edge_policy: dict[str, Any] | None
    operator_modules: dict[str, dict[str, Any]]
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
            "compiler_request": self.compiler_request,
            "compiler": self.compiler,
            "precision": self.precision,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "cache_key": self.cache_key,
            "artifact_path": str(self.artifact_path),
            "manifest_path": str(self.manifest_path),
            "artifact_id": self.artifact_id,
            "variant_id": self.variant_id,
            "edge_policy": self.edge_policy,
            "operator_modules": self.operator_modules,
            "persistent_blocks_per_compute_unit": (
                None
                if self.edge_policy is None
                else self.edge_policy.get("persistent_blocks_per_compute_unit")
            ),
            "environment": self.environment,
            "diagnostics": list(self.diagnostics),
        }


def _read_model_json(model_file) -> tuple[Path, dict[str, Any]]:
    model_path = Path(model_file).expanduser().resolve()
    try:
        with model_path.open(encoding="utf-8") as stream:
            model_data = json.load(stream)
    except (OSError, TypeError, ValueError) as error:
        raise JitDeviceArtifactError(
            f"could not read compact Symmetrix model JSON {model_path}: {error}"
        ) from error
    if not isinstance(model_data, dict):
        raise JitDeviceArtifactError(
            "compact Symmetrix model JSON must contain an object"
        )
    return model_path, model_data


def _native_evaluator(model_path: Path, precision: str):
    evaluator_name = "MACEKokkos" if precision == "float64" else "MACEKokkosFloat"
    evaluator_class = getattr(symmetrix, evaluator_name, None)
    if evaluator_class is None:
        raise JitDeviceArtifactError(
            "this Symmetrix build does not provide the requested Kokkos evaluator"
        )
    try:
        if not symmetrix._kokkos_is_initialized():
            symmetrix._init_kokkos()
        evaluator = evaluator_class(str(model_path))
        evaluator.set_streamed_edges("factorized")
    except Exception as error:
        raise JitDeviceArtifactError(
            f"could not construct the native Execution R1 evaluator: {error}"
        ) from error
    return evaluator


def _device_environment(evaluator) -> dict[str, Any]:
    environment = getattr(evaluator, "execution_device_execution_environment", None)
    if callable(environment):
        environment = environment()
    if not isinstance(environment, dict):
        raise JitDeviceArtifactError(
            "the native evaluator did not report a device execution environment"
        )
    if not bool(environment.get("available", False)):
        raise JitDeviceArtifactError(
            "the native evaluator did not report an available device execution environment"
        )
    backend = environment.get("backend")
    if backend not in ("cuda", "hip"):
        raise JitDeviceArtifactError(
            f"Execution device artifacts require a CUDA or HIP backend, found {backend!r}"
        )
    return dict(environment)


def _prepare_and_validate(
    *,
    evaluator,
    contract: dict[str, Any],
    precision: str,
    environment: dict[str, Any],
    model_path: Path,
    model_type: str,
    model_data: dict[str, Any],
    include_low_memory_operators: bool,
    cache_root,
) -> JitDeviceArtifactResult:
    backend = environment["backend"]
    try:
        plan = _factorized_device_backend_plan(
            backend, environment, contract, precision, evaluator
        )
        attempt = plan.build_attempt(plan.selected_compiler)
    except Exception as error:
        raise JitDeviceArtifactError(error) from error

    prepare_arguments = dict(attempt.arguments)
    prepare_arguments["build"] = {
        **prepare_arguments["build"],
        "jit_generation_version": JIT_GENERATION_VERSION,
    }
    if cache_root is not None:
        prepare_arguments["cache_root"] = cache_root
    diagnostics: tuple[str, ...] = ()
    retained_quarantines = []
    failed_load_artifacts = set()
    result = None
    for _load_attempt in range(4):
        result = attempt.prepare(attempt.source, **prepare_arguments)
        diagnostics = (*diagnostics, *result.diagnostics)
        if not result.available or result.artifact_path is None:
            raise JitDeviceArtifactError(
                result.reason or "the JIT cache did not produce an artifact",
                diagnostics=diagnostics,
                cache_key=result.cache_key,
            )
        try:
            if backend == "cuda":
                plan.loader(
                    str(result.artifact_path),
                    plan.metadata["persistent_blocks_per_compute_unit"],
                )
            else:
                plan.loader(str(result.artifact_path))
            break
        except Exception as load_error:
            failed_artifact = (
                plan.selected_compiler,
                result.cache_key,
                str(result.artifact_path),
            )
            if failed_artifact in failed_load_artifacts:
                raise JitDeviceArtifactError(
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
        raise JitDeviceArtifactError(
            "Execution device artifact recovery attempts were exhausted",
            diagnostics=diagnostics,
            cache_key=None if result is None else result.cache_key,
        )

    if not bool(getattr(evaluator, plan.ready_attribute, False)):
        raise JitDeviceArtifactError(
            f"the evaluator did not retain the loaded {backend} plugin",
            diagnostics=diagnostics,
            cache_key=result.cache_key,
        )
    for quarantine in retained_quarantines:
        diagnostics = (*diagnostics, *remove_jit_quarantine(quarantine))

    artifact_id = str(
        getattr(evaluator, plan.artifact_attribute, plan.metadata["artifact_id"])
    )
    operator_modules: dict[str, dict[str, Any]] = {}
    contracts = model_data.get("execution_contracts", {})
    if (
        include_low_memory_operators
        and isinstance(contracts.get("M0"), dict)
        and isinstance(contracts.get("R0"), dict)
    ):
        try:
            operator_modules, operator_diagnostics = (
                prepare_low_memory_operator_modules(
                    evaluator,
                    model_data=model_data,
                    precision=precision,
                    backend=backend,
                    target=environment,
                    jit_generation_version=JIT_GENERATION_VERSION,
                    cache_root=cache_root,
                )
            )
        except Exception as error:
            raise JitDeviceArtifactError(
                error,
                diagnostics=diagnostics,
                cache_key=result.cache_key,
            ) from error
        diagnostics = (*diagnostics, *operator_diagnostics)
    return JitDeviceArtifactResult(
        status=result.status,
        backend=backend,
        compiler_request=plan.policy,
        compiler=plan.selected_compiler,
        precision=precision,
        model_type=model_type,
        model_path=model_path,
        cache_key=str(result.cache_key),
        artifact_path=result.artifact_path,
        manifest_path=result.manifest_path,
        artifact_id=artifact_id,
        variant_id=plan.variant_id,
        edge_policy=None if plan.edge_policy is None else dict(plan.edge_policy),
        operator_modules=operator_modules,
        environment=dict(environment),
        diagnostics=diagnostics,
    )


def prepare_jit_device_artifact(
    model_file,
    *,
    precision: str,
    backend: str = "auto",
    cache_root=None,
    include_low_memory_operators: bool = True,
) -> JitDeviceArtifactResult:
    """Prepare CUDA/HIP Execution R1 and required low-memory operator modules."""

    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if backend not in ("auto", "cuda", "hip"):
        raise ValueError("backend must be 'auto', 'cuda', or 'hip'")

    model_path, model_data = _read_model_json(model_file)
    model_type = str(model_data.get("model_type", "MACE"))
    if model_type not in ("MACE", "MACEField"):
        raise JitDeviceArtifactError(
            "Execution device artifact preparation supports MACE and MACEField models"
        )
    contract = model_data.get("execution_contracts", {}).get("R1")
    if not isinstance(contract, dict):
        raise JitDeviceArtifactError(
            "the model does not contain a Execution R1 contract"
        )

    evaluator = _native_evaluator(model_path, precision)
    environment = _device_environment(evaluator)
    if backend != "auto" and environment["backend"] != backend:
        raise JitDeviceArtifactError(
            f"requested {backend} preparation but the active backend is "
            f"{environment['backend']}"
        )
    try:
        return _prepare_and_validate(
            evaluator=evaluator,
            contract=contract,
            precision=precision,
            environment=environment,
            model_path=model_path,
            model_type=model_type,
            model_data=model_data,
            include_low_memory_operators=include_low_memory_operators,
            cache_root=cache_root,
        )
    except JitDeviceArtifactError:
        raise
    except Exception as error:
        raise JitDeviceArtifactError(error) from error


__all__ = [
    "JitDeviceArtifactError",
    "JitDeviceArtifactResult",
    "prepare_jit_device_artifact",
]
