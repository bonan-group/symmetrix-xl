"""Prepare validated direct-execution device artifacts without an ASE calculator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from . import symmetrix
from .calculator import (
    _factorized_device_backend_plan,
    _factorized_device_installed_plan,
)
from .installed_operator_artifact import (
    resolve_installed_operator_artifact,
)
from .jit import (
    JIT_GENERATION_VERSION,
    load_validated_jit_artifact,
    native_source_content_sha256,
    remove_jit_quarantine,
)
from .mace_jit_artifact_common import native_evaluator, read_model_json
from .mace_jit_operator_artifact import (
    _load_low_memory_operator_modules,
    _plan_low_memory_operator_modules,
    _prepare_with_initial_future,
    _submit_operator_artifact_jobs,
    prepare_low_memory_operator_modules,
)

_CUDA_SPLIT_COMPILE_OPTIONS = ("--split-compile=2",)


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
    origin: str
    backend: str
    compiler_request: str
    compiler: str
    precision: str
    model_type: str
    model_path: Path
    cache_key: str | None
    artifact_path: Path
    manifest_path: Path | None
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
            "origin": self.origin,
            "available": True,
            "backend": self.backend,
            "compiler_request": self.compiler_request,
            "compiler": self.compiler,
            "precision": self.precision,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "cache_key": self.cache_key,
            "artifact_path": str(self.artifact_path),
            "manifest_path": (
                None if self.manifest_path is None else str(self.manifest_path)
            ),
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
    contracts = model_data.get("execution_contracts", {})
    prepare_operators = (
        include_low_memory_operators
        and isinstance(contracts.get("M0"), dict)
        and isinstance(contracts.get("R0"), dict)
    )
    try:
        installed_plan = _factorized_device_installed_plan(
            backend, environment, contract, precision, evaluator
        )
        if backend == "cuda":
            blocks_per_compute_unit = installed_plan.metadata[
                "persistent_blocks_per_compute_unit"
            ]

            def load_installed(path):
                installed_plan.loader(path, blocks_per_compute_unit)

        else:
            load_installed = installed_plan.loader
        installed = resolve_installed_operator_artifact(
            operator="R1",
            identity=installed_plan.metadata["generated_identity"],
            abi={
                "tag": installed_plan.metadata["abi"],
                "version": installed_plan.metadata["abi_version"],
            },
            required_capabilities=installed_plan.metadata["capabilities"],
            loader=load_installed,
            admitted=lambda: bool(
                getattr(evaluator, installed_plan.ready_attribute, False)
            ),
            native_module=symmetrix,
        )
    except Exception as error:
        raise JitDeviceArtifactError(error) from error

    if installed.status == "resolved":
        operator_modules: dict[str, dict[str, Any]] = {}
        diagnostics = installed.diagnostics
        if prepare_operators:
            try:
                operator_modules, operator_diagnostics = (
                    prepare_low_memory_operator_modules(
                        evaluator,
                        model_data=model_data,
                        precision=precision,
                        backend=backend,
                        target=environment,
                        jit_generation_version=JIT_GENERATION_VERSION,
                        source_content_sha256=native_source_content_sha256(symmetrix),
                        cache_root=cache_root,
                    )
                )
            except Exception as error:
                raise JitDeviceArtifactError(error, diagnostics=diagnostics) from error
            diagnostics = (*diagnostics, *operator_diagnostics)
        return JitDeviceArtifactResult(
            status="loaded",
            origin="prebuilt",
            backend=backend,
            compiler_request="prebuilt",
            compiler="prebuilt",
            precision=precision,
            model_type=model_type,
            model_path=model_path,
            cache_key=None,
            artifact_path=installed.artifact_path,
            manifest_path=None,
            artifact_id=str(
                getattr(
                    evaluator,
                    installed_plan.artifact_attribute,
                    installed.artifact_id,
                )
            ),
            variant_id=installed_plan.variant_id,
            edge_policy=(
                None
                if installed_plan.edge_policy is None
                else dict(installed_plan.edge_policy)
            ),
            operator_modules=operator_modules,
            environment=dict(environment),
            diagnostics=diagnostics,
        )
    try:
        plan = _factorized_device_backend_plan(
            backend, environment, contract, precision, evaluator
        )
        build = plan.build_attempt(plan.selected_compiler)
    except Exception as error:
        raise JitDeviceArtifactError(
            error, diagnostics=installed.diagnostics
        ) from error

    prepare_arguments = dict(build.arguments)
    prepare_arguments["build"] = {
        **prepare_arguments["build"],
        "jit_generation_version": JIT_GENERATION_VERSION,
    }
    source_content_sha256 = native_source_content_sha256(symmetrix)
    prepare_arguments["source_content_sha256"] = source_content_sha256
    if cache_root is not None:
        prepare_arguments["cache_root"] = cache_root
    if backend == "cuda":
        if plan.selected_compiler == "nvrtc":
            prepare_arguments["nvrtc_options"] = _CUDA_SPLIT_COMPILE_OPTIONS
        elif plan.selected_compiler == "nvcc":
            prepare_arguments["nvcc_flags"] = _CUDA_SPLIT_COMPILE_OPTIONS

    operator_plan = None
    if backend == "cuda" and prepare_operators:
        try:
            operator_plan = _plan_low_memory_operator_modules(
                evaluator,
                model_data=model_data,
                precision=precision,
                backend=backend,
                target=environment,
                jit_generation_version=JIT_GENERATION_VERSION,
                source_content_sha256=source_content_sha256,
                cache_root=cache_root,
            )
        except Exception as error:
            raise JitDeviceArtifactError(
                error, diagnostics=installed.diagnostics
            ) from error

    prepare_r1 = partial(build.prepare, build.source, **prepare_arguments)
    r1_future = None
    operator_futures = {}
    if operator_plan is not None and operator_plan.jobs:
        with ThreadPoolExecutor(
            max_workers=1 + len(operator_plan.jobs),
            thread_name_prefix="symmetrix-jit",
        ) as executor:
            r1_future = executor.submit(prepare_r1)
            operator_futures = _submit_operator_artifact_jobs(
                executor, operator_plan.jobs
            )
    if backend == "cuda":
        blocks_per_compute_unit = plan.metadata["persistent_blocks_per_compute_unit"]

        def load_artifact(path):
            plan.loader(path, blocks_per_compute_unit)

    else:
        load_artifact = plan.loader

    loaded = load_validated_jit_artifact(
        _prepare_with_initial_future(prepare_r1, r1_future),
        load_artifact,
        artifact_key=lambda result: (
            plan.selected_compiler,
            result.cache_key,
            str(result.artifact_path),
        ),
    )
    diagnostics = (*installed.diagnostics, *loaded.diagnostics)
    if loaded.status == "unavailable":
        raise JitDeviceArtifactError(
            loaded.result.reason or "the JIT cache did not produce an artifact",
            diagnostics=diagnostics,
            cache_key=loaded.cache_key,
        )
    if loaded.status == "load-failed":
        raise JitDeviceArtifactError(
            loaded.error,
            diagnostics=diagnostics,
            cache_key=loaded.cache_key,
        ) from loaded.error
    if loaded.status == "exhausted":
        raise JitDeviceArtifactError(
            "Execution device artifact recovery attempts were exhausted",
            diagnostics=diagnostics,
            cache_key=loaded.cache_key,
        )
    result = loaded.result

    if not bool(getattr(evaluator, plan.ready_attribute, False)):
        raise JitDeviceArtifactError(
            f"the evaluator did not retain the loaded {backend} plugin",
            diagnostics=diagnostics,
            cache_key=result.cache_key,
        )
    for quarantine in loaded.quarantines:
        diagnostics = (*diagnostics, *remove_jit_quarantine(quarantine))

    artifact_id = str(
        getattr(evaluator, plan.artifact_attribute, plan.metadata["artifact_id"])
    )
    operator_modules: dict[str, dict[str, Any]] = {}
    if prepare_operators:
        try:
            if operator_plan is not None:
                operator_modules, operator_diagnostics = (
                    _load_low_memory_operator_modules(
                        evaluator,
                        operator_plan,
                        prepared=operator_futures,
                    )
                )
            else:
                operator_modules, operator_diagnostics = (
                    prepare_low_memory_operator_modules(
                        evaluator,
                        model_data=model_data,
                        precision=precision,
                        backend=backend,
                        target=environment,
                        jit_generation_version=JIT_GENERATION_VERSION,
                        source_content_sha256=source_content_sha256,
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
        origin="cache" if result.status == "cached" else "runtime",
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
    target: str = "",
    backend: str = "auto",
    cache_root=None,
    include_low_memory_operators: bool = True,
) -> JitDeviceArtifactResult:
    """Prepare CUDA/HIP Execution R1 and required low-memory operator modules.

    ``target`` is the declared artifact adapter's execution target.  The family
    declares this entry point for both device targets, so ``"cuda"`` and
    ``"hip"`` each select the matching backend and ``""`` keeps the legacy
    adapter-agnostic calling convention, in which the backend resolves from the
    evaluator's own device execution environment.  The host target the family's
    sibling host adapter serves is refused by name before the model file is
    read, so a frontend that resolved the wrong adapter for a request fails at
    preparation instead of probing for a device.  ``backend`` remains the
    explicit backend selector: a ``target`` that disagrees with it is refused
    rather than silently overridden.
    """

    if target not in ("", "cuda", "hip"):
        raise JitDeviceArtifactError(
            "the Execution R1 device artifact adapter serves the CUDA and HIP "
            f"targets only, not {target!r}"
        )
    if target and backend not in ("auto", target):
        raise JitDeviceArtifactError(
            f"requested the {target} target but the {backend} backend"
        )
    if target:
        backend = target
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if backend not in ("auto", "cuda", "hip"):
        raise ValueError("backend must be 'auto', 'cuda', or 'hip'")

    model_path, model_data = read_model_json(
        model_file, error_type=JitDeviceArtifactError, label="compact"
    )
    model_type = model_data.get("model_type", "MACE")
    if model_type not in ("MACE", "MACEField", "MACELES"):
        raise JitDeviceArtifactError(
            "Execution device artifact preparation supports MACE, MACEField, "
            "and MACELES models"
        )
    contract = model_data.get("execution_contracts", {}).get("R1")
    if not isinstance(contract, dict):
        raise JitDeviceArtifactError(
            "the model does not contain a Execution R1 contract"
        )

    evaluator = native_evaluator(
        model_path,
        precision,
        model_type=model_type,
        error_type=JitDeviceArtifactError,
        streamed_edges="factorized",
    )
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
