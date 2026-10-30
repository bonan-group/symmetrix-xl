"""Prepare and native-validate runtime-specialized M0/M1/R0 modules."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

from .generated_operator_codegen import (
    PRODUCT_TILED128_SCHEDULE,
    select_product_device_schedule,
)
from .installed_operator_artifact import (
    installed_operator_build_target,
    resolve_installed_operator_artifact,
)

_M0_GFX1151_WIDE_L2_STRUCTURE = (
    "sha256:a4b4f395ce54594122e59f914e86339ee03c6c7c40466a47edb89cc5dce4a64c"
)

_CUDA_SPLIT_COMPILE_OPTIONS = ("--split-compile=2",)
_R0_DEVICE_SCHEDULE_ENVIRONMENT_VARIABLE = "SYMMETRIX_R0_RTC_SCHEDULE"


def _cuda_r0_floating_point_policy(native_module) -> str:
    build_info = getattr(native_module, "_backend_build_info", lambda: {})()
    return "cuda-fast-math" if build_info.get("fast_math", True) else "strict"


def resolve_r0_device_schedule() -> str:
    """Return the validated device schedule selected by runtime planning."""

    schedule = os.environ.get(
        _R0_DEVICE_SCHEDULE_ENVIRONMENT_VARIABLE, "receiver-edge16"
    )
    if schedule not in ("receiver-edge16", "receiver-edge32"):
        raise ValueError(
            f"{_R0_DEVICE_SCHEDULE_ENVIRONMENT_VARIABLE} must be "
            "receiver-edge16 or receiver-edge32"
        )
    return schedule


@dataclass(frozen=True)
class _OperatorArtifactRequest:
    request_key: str
    operator: str
    contract: Mapping[str, Any] | None
    layer: int | None = None


def _mh1_product_artifact_requests(contracts):
    contracts = tuple(contracts)
    if not contracts:
        raise ValueError("MH-1 composed execution requires at least one product layer")
    requests = []
    for layer, contract in enumerate(contracts):
        if not isinstance(contract, Mapping):
            raise ValueError(f"MH-1 product layer {layer} contract must be a mapping")
        expected = contract.get("interaction")
        if expected not in ("M0", "M1"):
            raise ValueError(
                f"MH-1 product layer {layer} must declare an M0 or M1 operator kind"
            )
        if contract.get("type_count") != 1:
            raise ValueError(f"MH-1 product layer {layer} type count must be one")
        requests.append(
            _OperatorArtifactRequest(
                request_key=f"product_{layer}",
                operator=expected,
                contract=contract,
                layer=layer,
            )
        )
    return tuple(requests)


def _validate_mh1_product_module_records(
    records, requests, *, backend, precision, target
):
    requests = tuple(requests)
    expected_keys = tuple(request.request_key for request in requests)
    if set(records) != set(expected_keys):
        raise RuntimeError("MH-1 product artifact set is incomplete")
    validated = {}
    for request in requests:
        key = request.request_key
        record = records[key]
        contract = request.contract
        if not isinstance(record, Mapping) or record.get("status") not in (
            "loaded",
            "cached",
            "compiled",
            "built",
        ):
            raise RuntimeError(f"MH-1 product layer {request.layer} is not loaded")
        if record.get("layer") != request.layer:
            raise RuntimeError(f"MH-1 product layer identity does not match for {key}")
        if record.get("operator") != request.operator:
            raise RuntimeError(f"MH-1 product operator kind does not match for {key}")
        if record.get("type_count") != 1:
            raise RuntimeError(f"MH-1 product type count does not match for {key}")
        dimension_fields = (
            "channels",
            "input_components",
            "output_components",
            "correlation",
        )
        expected_terms = sum(
            group["term_count"] for group in contract["monomial_groups"]
        )
        if any(record.get(field) != contract[field] for field in dimension_fields) or (
            record.get("term_count") != expected_terms
        ):
            raise RuntimeError(f"MH-1 product dimensions do not match for {key}")
        if record.get("coefficient_count") != expected_terms * contract["channels"]:
            raise RuntimeError(
                f"MH-1 product coefficient count does not match for {key}"
            )
        if record.get("precision") != precision:
            raise RuntimeError(f"MH-1 product precision does not match for {key}")
        if record.get("backend") != backend:
            raise RuntimeError(f"MH-1 product backend does not match for {key}")
        if record.get("target") != target:
            raise RuntimeError(f"MH-1 product target does not match for {key}")
        if backend in ("cuda", "hip") and record.get("schedule") not in (
            "chunk32",
            PRODUCT_TILED128_SCHEDULE,
        ):
            raise RuntimeError(
                f"MH-1 product layer {request.layer} schedule is not an admitted "
                "canonical or chunk32 fallback schedule"
            )
        if any(
            record.get(field) != contract[field]
            for field in (
                "generation_fingerprint",
                "semantic_fingerprint",
                "structure_fingerprint",
            )
        ):
            raise RuntimeError(f"MH-1 product identity does not match for {key}")
        artifact_id = record.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise RuntimeError(f"MH-1 product identity is missing for {key}")
        validated[key] = dict(record)
    return validated


@dataclass(frozen=True)
class _OperatorArtifactJob:
    request_key: str
    operator: str
    compiler: str
    module_target: str
    metadata: Mapping[str, Any]
    schedule: str
    persistent_blocks: int
    ready_attribute: str
    artifact_attribute: str
    prepare: Any
    load: Any
    installed_diagnostics: tuple[str, ...] = ()
    layer: int | None = None
    ready: Any = None
    artifact_id: Any = None


@dataclass(frozen=True)
class _LowMemoryOperatorPlan:
    modules: dict[str, dict[str, Any]]
    jobs: tuple[_OperatorArtifactJob, ...]


class _OperatorArtifactCompositionError(RuntimeError):
    def __init__(self, records, diagnostics):
        self.records = records
        self.diagnostics = tuple(diagnostics)
        super().__init__(
            "MH-1 product artifact composition failed: " + "; ".join(self.diagnostics)
        )


def _validate_unique_request_keys(requests: Sequence[Any]) -> None:
    seen: set[str] = set()
    for request in requests:
        request_key = request.request_key
        if not isinstance(request_key, str) or not request_key:
            raise ValueError("operator request keys must be nonempty strings")
        if request_key in seen:
            raise ValueError(f"duplicate operator request key: {request_key}")
        seen.add(request_key)


def _submit_operator_artifact_jobs(executor, jobs: Sequence[_OperatorArtifactJob]):
    _validate_unique_request_keys(jobs)
    return {job.request_key: executor.submit(job.prepare) for job in jobs}


def _operator_request_diagnostic_prefix(
    request_key: str, operator: str, schedule: str
) -> str:
    if request_key == operator:
        return f"stage={request_key} schedule={schedule}: "
    return f"stage={request_key} operator={operator} schedule={schedule}: "


def _operator_load_adapter(
    *,
    stage: str,
    backend: str,
    metadata: Mapping[str, Any],
    precision: str,
    target: str | None,
    schedule: str,
    apply_edge_weights: bool,
    loader,
):
    blocks = _persistent_blocks_per_compute_unit(
        stage,
        metadata,
        precision=precision,
        backend=backend,
        target=str(target),
    )
    if stage in ("M0", "M1") and backend != "host":

        def load(path):
            loader(path, schedule, blocks)

    elif stage == "R0" and backend != "host":

        def load(path):
            loader(path, blocks, apply_edge_weights, schedule)

    elif stage == "R0":

        def load(path):
            loader(path, apply_edge_weights, target)

    else:
        load = loader
    return load, blocks


def _mh1_product_load_adapter(
    evaluator,
    *,
    layer: int,
    stage: str,
    metadata: Mapping[str, Any],
    schedule: str,
    persistent_blocks: int,
    backend: str,
):
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    def load(path):
        evaluator._load_execution_mh1_product_module(
            layer,
            stage,
            path,
            metadata_json,
            schedule,
            persistent_blocks,
            backend,
        )

    return load


def _persistent_blocks_per_compute_unit(
    stage: str,
    metadata: Mapping[str, Any],
    *,
    precision: str,
    backend: str,
    target: str,
) -> int:
    if stage == "R0":
        profile = metadata.get("persistent_profile", {})
        blocks = profile.get("blocks_per_compute_unit")
        if (
            isinstance(blocks, int)
            and not isinstance(blocks, bool)
            and 1 <= blocks <= 32
        ):
            return blocks
    contract = metadata.get("contract", {})
    if (
        stage == "M0"
        and precision == "float32"
        and backend == "hip"
        and target == "gfx1151"
        and metadata.get("schedule") == "chunk32"
        and contract.get("channels") == 128
        and metadata.get("term_count") == 923
        and metadata.get("structure_fingerprint") == _M0_GFX1151_WIDE_L2_STRUCTURE
    ):
        return 2
    return 8


def _r0_loader_implementation(evaluator) -> str:
    """Return the non-public physical route used for planning and qualification."""

    return str(
        getattr(
            evaluator,
            "_r0_loader_implementation",
            getattr(evaluator, "r0_implementation", "generic"),
        )
    )


def _r0_profile(backend: str, persistent_blocks: int) -> str:
    return "owner-batched" if backend == "host" else f"persistent-{persistent_blocks}"


def _publish_r0_artifact_origin(evaluator, origin: str) -> None:
    publish = getattr(evaluator, "_set_r0_artifact_origin", None)
    if callable(publish):
        publish(origin)


def _plan_low_memory_operator_modules(
    evaluator,
    *,
    model_data: Mapping[str, Any],
    precision: str,
    backend: str,
    target: Mapping[str, Any],
    jit_generation_version: int,
    source_content_sha256: str | None = None,
    prefer_host_m0_plugin: bool = False,
    apply_edge_weights: bool = False,
    cache_root=None,
    operator_requests: Sequence[_OperatorArtifactRequest] | None = None,
) -> _LowMemoryOperatorPlan:
    """Render low-memory M0/M1/R0 stages and describe their artifact jobs."""

    contracts = model_data.get("execution_contracts", {})
    if operator_requests is None:
        operator_requests = tuple(
            _OperatorArtifactRequest(
                request_key=operator,
                operator=operator,
                contract=contracts.get(operator),
            )
            for operator in ("M0", "M1", "R0")
            if operator != "M1" or isinstance(contracts.get(operator), dict)
        )
    else:
        operator_requests = tuple(operator_requests)
    _validate_unique_request_keys(operator_requests)

    from . import symmetrix as native_module
    from .jit import (
        artifact_cache_inputs,
        normalize_execution_hip_target_identity,
        prepare_hiprtc_jit_artifact,
        prepare_jit_artifact,
        prepare_nvrtc_jit_artifact,
        probe_compiler_identity,
        resolve_host_jit_policy,
    )
    from .mace_jit_operator_codegen import (
        OPERATOR_ARTIFACT_ENVELOPE,
        render_jit_m0_device_module,
        render_jit_m0_host_plugin,
        render_jit_m1_device_module,
        render_jit_m1_host_plugin,
        render_jit_r0_device_module,
        render_jit_r0_host_plugin,
    )

    if backend == "cuda":
        compute_capability = target.get("compute_capability_code")
        if not isinstance(compute_capability, int):
            dotted = str(target.get("compute_capability", ""))
            major, separator, minor = dotted.partition(".")
            if not separator or not major.isdigit() or not minor.isdigit():
                raise RuntimeError(
                    "the native evaluator reported an invalid CUDA compute capability"
                )
            compute_capability = 10 * int(major) + int(minor)
        module_target = f"sm_{compute_capability}"
        compiler = "nvrtc"
        cuda_r0_floating_point_policy = _cuda_r0_floating_point_policy(native_module)
    elif backend == "hip":
        identity = normalize_execution_hip_target_identity(target)
        features = identity["target_features"]
        module_target = identity["architecture"] + (
            ":" + ":".join(features) if features else ""
        )
        compiler = "hiprtc"
    elif backend == "host":
        requested_target = target.get("cpu_target")
        if requested_target is None:
            requested_target = target.get("host_target")
        if requested_target is not None and (
            not isinstance(requested_target, str) or not requested_target
        ):
            raise RuntimeError("the host operator target must be a nonempty string")
        installed_target = installed_operator_build_target(
            native_module=native_module, backend="host"
        )
        module_target = installed_target
        host_compiler_command = None
        host_policy = None
        compiler = "host-cxx"
    else:
        raise RuntimeError("low-memory operator JIT requires host, CUDA, or HIP")

    installed_full_target = installed_operator_build_target(
        native_module=native_module,
        backend=backend,
        mode="full",
    )

    modules: dict[str, dict[str, Any]] = {}
    jobs: list[_OperatorArtifactJob] = []
    device_stage_specs = (
        (
            "M0",
            "m0_implementation",
            "m0_device_module_ready",
            "m0_device_module_artifact_id",
            "_load_m0_device_module",
        ),
        (
            "M1",
            "m1_implementation",
            "m1_device_module_ready",
            "m1_device_module_artifact_id",
            "_load_m1_device_module",
        ),
        (
            "R0",
            "r0_implementation",
            "r0_device_module_ready",
            "r0_device_module_artifact_id",
            "_load_r0_device_module",
        ),
    )
    host_stage_specs = (
        (
            "M0",
            "m0_implementation",
            "m0_host_plugin_ready",
            "m0_host_plugin_artifact_id",
            "_load_m0_host_plugin",
        ),
        (
            "M1",
            "m1_implementation",
            "m1_host_plugin_ready",
            "m1_host_plugin_artifact_id",
            "_load_m1_host_plugin",
        ),
        (
            "R0",
            "r0_implementation",
            "r0_host_plugin_ready",
            "r0_host_plugin_artifact_id",
            "_load_r0_host_plugin",
        ),
    )
    stage_specs = {
        spec[0]: spec[1:]
        for spec in (host_stage_specs if backend == "host" else device_stage_specs)
    }

    def resolve_runtime_host_target():
        nonlocal host_compiler_command, host_policy
        if host_policy is None:
            host_compiler_command, _ = probe_compiler_identity()
            host_policy = resolve_host_jit_policy(
                host_compiler_command, target=requested_target
            )
        return host_policy.effective_target

    for request in operator_requests:
        request_key = request.request_key
        stage = request.operator
        try:
            (
                implementation_attribute,
                ready_attribute,
                artifact_attribute,
                loader_name,
            ) = stage_specs[stage]
        except KeyError as exc:
            raise ValueError(f"unsupported operator request kind: {stage}") from exc
        contract = request.contract
        product_request = request.layer is not None
        implementation = (
            _r0_loader_implementation(evaluator)
            if stage == "R0"
            else (
                "generic"
                if product_request
                else str(getattr(evaluator, implementation_attribute, "generic"))
            )
        )
        installed_precedes_builtin = isinstance(contract, Mapping) and (
            (stage == "R0" and (backend != "host" or installed_target is not None))
            or (
                stage in ("M0", "M1")
                and installed_full_target is not None
                and installed_full_target == module_target
            )
        )
        if implementation == "builtin" and not installed_precedes_builtin:
            modules[request_key] = {
                "status": "builtin",
                "implementation": "builtin",
                "compiler": None,
                "artifact_id": getattr(
                    evaluator,
                    f"{stage.lower()}_module_id",
                    getattr(
                        evaluator,
                        f"standard_{stage.lower()}_module_id",
                        None,
                    ),
                ),
            }
            if request_key != stage:
                modules[request_key]["operator"] = stage
            if stage == "R0":
                modules[request_key].update(
                    implementation="generated",
                    backend=backend,
                    artifact_origin=None,
                    schedule="frozen",
                    profile="fixed-weight-coordinate",
                )
            continue
        if not isinstance(contract, Mapping):
            raise RuntimeError(
                f"low-memory execution requires an Execution {stage} contract"
            )
        render_target = (
            installed_target
            if backend == "host" and stage == "R0" and installed_target is not None
            else module_target
        )
        if backend == "host" and render_target is None:
            render_target = resolve_runtime_host_target()
        if stage in ("M0", "M1"):
            schedule_environment = f"SYMMETRIX_{stage}_RTC_SCHEDULE"
            if product_request and backend != "host":
                schedule = os.environ.get(schedule_environment)
                if schedule is None:
                    schedule = select_product_device_schedule(
                        contract,
                        precision=precision,
                        backend=backend,
                        target=str(render_target),
                    )
            else:
                schedule = os.environ.get(schedule_environment, "chunk32")
            if (
                product_request
                and backend != "host"
                and schedule
                not in (
                    "chunk32",
                    PRODUCT_TILED128_SCHEDULE,
                )
            ):
                raise RuntimeError(
                    _operator_request_diagnostic_prefix(request_key, stage, schedule)
                    + "MH-1 device composition requires a canonical schedule or "
                    "the explicit chunk32 fallback"
                )
            try:
                if stage == "M0" and backend == "host":
                    source, metadata = render_jit_m0_host_plugin(
                        contract,
                        precision=precision,
                        schedule=schedule,
                        target=render_target or "host",
                    )
                elif stage == "M0":
                    source, metadata = render_jit_m0_device_module(
                        contract,
                        precision=precision,
                        target=render_target,
                        schedule=schedule,
                    )
                elif backend == "host":
                    source, metadata = render_jit_m1_host_plugin(
                        contract,
                        precision=precision,
                        schedule=schedule,
                        target=render_target or "host",
                    )
                else:
                    source, metadata = render_jit_m1_device_module(
                        contract,
                        precision=precision,
                        target=render_target,
                        schedule=schedule,
                    )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    _operator_request_diagnostic_prefix(request_key, stage, schedule)
                    + f"operator generation failed: {exc}"
                ) from exc
        else:
            schedule = (
                "host-owner" if backend == "host" else resolve_r0_device_schedule()
            )
            try:
                if backend == "host":
                    source, metadata = render_jit_r0_host_plugin(
                        contract,
                        precision=precision,
                        target=render_target,
                        apply_edge_weights=apply_edge_weights,
                    )
                else:
                    source, metadata = render_jit_r0_device_module(
                        contract,
                        precision=precision,
                        target=render_target,
                        schedule=schedule,
                        floating_point_policy=(
                            cuda_r0_floating_point_policy
                            if backend == "cuda"
                            else "strict"
                        ),
                        apply_edge_weights=apply_edge_weights,
                    )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    _operator_request_diagnostic_prefix(request_key, stage, schedule)
                    + f"operator generation failed: {exc}"
                ) from exc

        loader = (
            getattr(evaluator, "_load_execution_mh1_product_module", None)
            if product_request
            else getattr(evaluator, loader_name, None)
        )
        if loader is None:
            raise RuntimeError(
                _operator_request_diagnostic_prefix(request_key, stage, schedule)
                + f"the native evaluator does not expose the {stage} device-module loader"
            )

        if product_request:
            persistent_blocks = _persistent_blocks_per_compute_unit(
                stage,
                metadata,
                precision=precision,
                backend=backend,
                target=str(render_target),
            )
            load_artifact = _mh1_product_load_adapter(
                evaluator,
                layer=request.layer,
                stage=stage,
                metadata=metadata,
                schedule=schedule,
                persistent_blocks=persistent_blocks,
                backend=backend,
            )

            def ready(_layer=request.layer):
                return bool(evaluator._execution_mh1_product_module_ready(_layer))

            def artifact_id_getter(_layer=request.layer):
                return evaluator._execution_mh1_product_module_artifact_id(_layer)
        else:
            load_artifact, persistent_blocks = _operator_load_adapter(
                stage=stage,
                backend=backend,
                metadata=metadata,
                precision=precision,
                target=render_target,
                schedule=schedule,
                apply_edge_weights=apply_edge_weights,
                loader=loader,
            )
            ready = None
            artifact_id_getter = None
        installed_identity = metadata.get(
            "generated_identity", {"artifact_id": metadata["artifact_id"]}
        )
        candidate_target = render_target
        installed_diagnostics: tuple[str, ...] = ()
        installed_result = resolve_installed_operator_artifact(
            operator=stage,
            identity=installed_identity,
            abi=artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
            required_capabilities=metadata["capabilities"],
            loader=load_artifact,
            admitted=(
                ready
                if ready is not None
                else lambda: bool(getattr(evaluator, ready_attribute, False))
            ),
            native_module=native_module,
        )
        installed_diagnostics = installed_result.diagnostics
        if installed_result.status == "resolved":
            stage_diagnostics = tuple(
                _operator_request_diagnostic_prefix(request_key, stage, schedule)
                + diagnostic
                for diagnostic in installed_diagnostics
            )
            modules[request_key] = {
                "status": "loaded",
                "origin": "prebuilt",
                "implementation": (
                    "host_plugin" if compiler == "host-cxx" else "device_module"
                ),
                "compiler": None,
                "cache_key": None,
                "artifact_path": str(installed_result.artifact_path),
                "artifact_id": (
                    artifact_id_getter()
                    if artifact_id_getter is not None
                    else getattr(
                        evaluator, artifact_attribute, installed_result.artifact_id
                    )
                ),
                "schedule": schedule,
                "schedule_revision": metadata.get("schedule_revision"),
                "compatibility_version": metadata.get("compatibility_version"),
                "schedule_capabilities": metadata.get("schedule_capabilities", []),
                "architecture_bounds": metadata.get("architecture_bounds"),
                "persistent_blocks_per_compute_unit": persistent_blocks,
                "target": candidate_target,
                "diagnostics": stage_diagnostics,
            }
            if request_key != stage:
                modules[request_key]["operator"] = stage
            if product_request:
                modules[request_key].update(
                    layer=request.layer,
                    backend=backend,
                    precision=precision,
                    metadata=metadata,
                    generation_fingerprint=contract["generation_fingerprint"],
                    semantic_fingerprint=contract["semantic_fingerprint"],
                    structure_fingerprint=contract["structure_fingerprint"],
                    channels=contract["channels"],
                    type_count=contract["type_count"],
                    input_components=contract["input_components"],
                    output_components=contract["output_components"],
                    correlation=contract["correlation"],
                    term_count=metadata["term_count"],
                    coefficient_count=metadata["term_count"] * contract["channels"],
                )
            if stage == "R0":
                _publish_r0_artifact_origin(evaluator, "prebuilt")
                modules[request_key].pop("origin")
                modules[request_key].update(
                    implementation="generated",
                    backend=backend,
                    artifact_origin="prebuilt",
                    profile=_r0_profile(backend, persistent_blocks),
                )
            continue

        if implementation == "builtin":
            stage_diagnostics = tuple(
                _operator_request_diagnostic_prefix(request_key, stage, schedule)
                + diagnostic
                for diagnostic in installed_diagnostics
            )
            modules[request_key] = {
                "status": "builtin",
                "origin": "builtin",
                "implementation": "builtin",
                "compiler": None,
                "artifact_id": getattr(
                    evaluator,
                    f"{stage.lower()}_module_id",
                    getattr(
                        evaluator,
                        f"standard_{stage.lower()}_module_id",
                        None,
                    ),
                ),
                "diagnostics": stage_diagnostics,
            }
            if request_key != stage:
                modules[request_key]["operator"] = stage
            if stage == "R0":
                modules[request_key].pop("origin")
                modules[request_key].update(
                    implementation="generated",
                    backend=backend,
                    artifact_origin=None,
                    schedule="frozen",
                    profile="fixed-weight-coordinate",
                )
            continue

        if backend == "host":
            runtime_target = resolve_runtime_host_target()
            if runtime_target != render_target:
                if stage == "M0":
                    source, metadata = render_jit_m0_host_plugin(
                        contract,
                        precision=precision,
                        schedule=schedule,
                        target=runtime_target,
                    )
                elif stage == "M1":
                    source, metadata = render_jit_m1_host_plugin(
                        contract,
                        precision=precision,
                        schedule=schedule,
                        target=runtime_target,
                    )
                else:
                    source, metadata = render_jit_r0_host_plugin(
                        contract,
                        precision=precision,
                        target=runtime_target,
                        apply_edge_weights=apply_edge_weights,
                    )
            module_target = runtime_target
            if product_request:
                persistent_blocks = _persistent_blocks_per_compute_unit(
                    stage,
                    metadata,
                    precision=precision,
                    backend=backend,
                    target=runtime_target,
                )
                load_artifact = _mh1_product_load_adapter(
                    evaluator,
                    layer=request.layer,
                    stage=stage,
                    metadata=metadata,
                    schedule=schedule,
                    persistent_blocks=persistent_blocks,
                    backend=backend,
                )
            else:
                load_artifact, persistent_blocks = _operator_load_adapter(
                    stage=stage,
                    backend=backend,
                    metadata=metadata,
                    precision=precision,
                    target=runtime_target,
                    schedule=schedule,
                    apply_edge_weights=apply_edge_weights,
                    loader=loader,
                )
        common = {
            "source_content_sha256": source_content_sha256,
            "abi": artifact_cache_inputs(metadata, OPERATOR_ARTIFACT_ENVELOPE),
            "build": {
                "generator": f"symmetrix.jit.{stage.lower()}-operator-module-v1",
                "precision": precision,
                "operator": metadata,
                "jit_generation_version": jit_generation_version,
            },
            "artifact_name": f"execution_{stage.lower()}_{backend}_module",
        }
        if backend == "host":
            prepare = prepare_jit_artifact
            arguments = {
                **common,
                "cxx": host_compiler_command,
                "cxx_flags": None,
                "host_policy": host_policy,
            }
        elif backend == "cuda":
            prepare = prepare_nvrtc_jit_artifact
            arguments = {
                **common,
                "compute_capability": target,
                "target": target,
                "nvrtc_options": (
                    (
                        *_CUDA_SPLIT_COMPILE_OPTIONS,
                        *(
                            ("--use_fast_math",)
                            if metadata.get(
                                "floating_point_policy", cuda_r0_floating_point_policy
                            )
                            == "cuda-fast-math"
                            else ()
                        ),
                    )
                    if stage == "R0"
                    else _CUDA_SPLIT_COMPILE_OPTIONS
                ),
            }
        else:
            prepare = prepare_hiprtc_jit_artifact
            arguments = {**common, "target": target}
        if cache_root is not None:
            arguments["cache_root"] = cache_root
        jobs.append(
            _OperatorArtifactJob(
                request_key=request_key,
                operator=stage,
                compiler=compiler,
                module_target=module_target,
                metadata=metadata,
                schedule=schedule,
                persistent_blocks=persistent_blocks,
                ready_attribute=ready_attribute,
                artifact_attribute=artifact_attribute,
                prepare=partial(prepare, source, **arguments),
                load=load_artifact,
                installed_diagnostics=installed_diagnostics,
                layer=request.layer,
                ready=ready,
                artifact_id=artifact_id_getter,
            )
        )
    return _LowMemoryOperatorPlan(modules=modules, jobs=tuple(jobs))


def _prepare_with_initial_future(prepare, future: Future | None):
    initial = future

    def prepare_artifact():
        nonlocal initial
        if initial is not None:
            prepared = initial
            initial = None
            return prepared.result()
        return prepare()

    return prepare_artifact


def _load_low_memory_operator_modules(
    evaluator,
    plan: _LowMemoryOperatorPlan,
    *,
    prepared: Mapping[str, Future] | None = None,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    from .jit import load_validated_jit_artifact, remove_jit_quarantine

    modules = dict(plan.modules)
    diagnostics: tuple[str, ...] = tuple(
        diagnostic
        for module in plan.modules.values()
        for diagnostic in module.get("diagnostics", ())
    )
    prepared = {} if prepared is None else prepared
    failures = []
    for job in plan.jobs:
        request_key = job.request_key
        operator = job.operator
        diagnostic_prefix = _operator_request_diagnostic_prefix(
            request_key, operator, job.schedule
        )
        try:
            loaded = load_validated_jit_artifact(
                _prepare_with_initial_future(
                    job.prepare, prepared.get(job.request_key)
                ),
                job.load,
                artifact_key=lambda result: (
                    result.cache_key,
                    str(result.artifact_path),
                ),
            )
        except Exception as error:
            if job.layer is None:
                raise
            message = diagnostic_prefix + str(error)
            modules[request_key] = {
                "status": "failed",
                "operator": operator,
                "layer": job.layer,
                "target": job.module_target,
                "diagnostics": (message,),
                "error": str(error),
            }
            diagnostics = (*diagnostics, message)
            failures.append(message)
            continue
        stage_diagnostics = tuple(
            diagnostic_prefix + diagnostic for diagnostic in loaded.diagnostics
        )
        stage_diagnostics = (
            *(diagnostic_prefix + value for value in job.installed_diagnostics),
            *stage_diagnostics,
        )
        if loaded.status == "unavailable":
            error = RuntimeError(
                loaded.result.reason
                or f"the {operator} RTC cache did not produce an artifact"
            )
        elif loaded.status == "load-failed":
            error = RuntimeError(f"operator artifact load failed: {loaded.error}")
        elif loaded.status == "exhausted":
            error = RuntimeError(
                f"Execution {operator} RTC artifact recovery attempts were exhausted"
            )
        else:
            error = None
        if error is not None:
            if job.layer is None:
                raise RuntimeError(diagnostic_prefix + str(error)) from getattr(
                    loaded, "error", None
                )
            message = diagnostic_prefix + str(error)
            modules[request_key] = {
                "status": "failed",
                "operator": operator,
                "layer": job.layer,
                "target": job.module_target,
                "diagnostics": (*stage_diagnostics, message),
                "error": str(error),
            }
            diagnostics = (*diagnostics, *stage_diagnostics, message)
            failures.append(message)
            continue
        result = loaded.result
        origin = "cache" if result.status == "cached" else "runtime"
        stage_diagnostics = (
            *stage_diagnostics,
            diagnostic_prefix
            + f"origin={origin} status="
            + ("hit" if origin == "cache" else "compiled"),
        )
        ready = (
            job.ready()
            if job.ready is not None
            else bool(getattr(evaluator, job.ready_attribute, False))
        )
        if not ready:
            message = (
                diagnostic_prefix
                + f"the evaluator did not retain the loaded {operator} device module"
            )
            if job.layer is None:
                raise RuntimeError(message)
            modules[request_key] = {
                "status": "failed",
                "operator": operator,
                "layer": job.layer,
                "target": job.module_target,
                "diagnostics": (*stage_diagnostics, message),
                "error": message,
            }
            diagnostics = (*diagnostics, *stage_diagnostics, message)
            failures.append(message)
            continue
        for quarantine in loaded.quarantines:
            quarantine_diagnostics = remove_jit_quarantine(quarantine)
            stage_diagnostics = (
                *stage_diagnostics,
                *(
                    diagnostic_prefix + diagnostic
                    for diagnostic in quarantine_diagnostics
                ),
            )
        diagnostics = (*diagnostics, *stage_diagnostics)
        module = {
            "status": result.status,
            "origin": origin,
            "implementation": (
                "host_plugin" if job.compiler == "host-cxx" else "device_module"
            ),
            "compiler": job.compiler,
            "cache_key": result.cache_key,
            "artifact_path": str(result.artifact_path),
            "artifact_id": (
                job.artifact_id()
                if job.artifact_id is not None
                else getattr(
                    evaluator, job.artifact_attribute, job.metadata["artifact_id"]
                )
            ),
            "schedule": job.schedule,
            "schedule_revision": job.metadata.get("schedule_revision"),
            "compatibility_version": job.metadata.get("compatibility_version"),
            "schedule_capabilities": job.metadata.get("schedule_capabilities", []),
            "architecture_bounds": job.metadata.get("architecture_bounds"),
            "persistent_blocks_per_compute_unit": job.persistent_blocks,
            "target": job.module_target,
            "diagnostics": stage_diagnostics,
        }
        if request_key != operator:
            module["operator"] = operator
        if job.layer is not None:
            contract = job.metadata["contract"]
            module.update(
                layer=job.layer,
                backend={"host-cxx": "host", "nvrtc": "cuda", "hiprtc": "hip"}[
                    job.compiler
                ],
                precision=(
                    "float32" if job.metadata["scalar_size"] == 4 else "float64"
                ),
                metadata=job.metadata,
                generation_fingerprint=contract["generation_fingerprint"],
                semantic_fingerprint=contract["semantic_fingerprint"],
                structure_fingerprint=contract["structure_fingerprint"],
                channels=contract["channels"],
                type_count=contract["type_count"],
                input_components=contract["input_components"],
                output_components=contract["output_components"],
                correlation=contract["correlation"],
                term_count=job.metadata["term_count"],
                coefficient_count=job.metadata["term_count"] * contract["channels"],
            )
        modules[request_key] = module
        if operator == "R0":
            _publish_r0_artifact_origin(evaluator, origin)
            modules[request_key].pop("origin")
            r0_backend = {
                "host-cxx": "host",
                "nvrtc": "cuda",
                "hiprtc": "hip",
            }[job.compiler]
            modules[request_key].update(
                implementation="generated",
                backend=r0_backend,
                artifact_origin=origin,
                profile=_r0_profile(r0_backend, job.persistent_blocks),
            )
    if failures:
        clear = getattr(evaluator, "_clear_execution_mh1_product_modules", None)
        if callable(clear):
            clear()
        raise _OperatorArtifactCompositionError(modules, diagnostics)
    return modules, diagnostics


def prepare_low_memory_operator_modules(
    evaluator,
    *,
    model_data: Mapping[str, Any],
    precision: str,
    backend: str,
    target: Mapping[str, Any],
    jit_generation_version: int,
    source_content_sha256: str | None = None,
    prefer_host_m0_plugin: bool = False,
    apply_edge_weights: bool = False,
    cache_root=None,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """Prepare architecture-specialized low-memory M0/M1/R0 stages."""

    plan = _plan_low_memory_operator_modules(
        evaluator,
        model_data=model_data,
        precision=precision,
        backend=backend,
        target=target,
        jit_generation_version=jit_generation_version,
        source_content_sha256=source_content_sha256,
        prefer_host_m0_plugin=prefer_host_m0_plugin,
        apply_edge_weights=apply_edge_weights,
        cache_root=cache_root,
    )
    if len(plan.jobs) < 2:
        return _load_low_memory_operator_modules(evaluator, plan)
    with ThreadPoolExecutor(
        max_workers=min(2, len(plan.jobs)), thread_name_prefix="symmetrix-jit"
    ) as executor:
        prepared = _submit_operator_artifact_jobs(executor, plan.jobs)
        return _load_low_memory_operator_modules(evaluator, plan, prepared=prepared)


__all__ = ["prepare_low_memory_operator_modules", "resolve_r0_device_schedule"]
