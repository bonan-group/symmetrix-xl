"""Prepare and native-validate runtime-specialized M0/R0 device modules."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

_M0_GFX1151_WIDE_L2_STRUCTURE = (
    "sha256:a4b4f395ce54594122e59f914e86339ee03c6c7c40466a47edb89cc5dce4a64c"
)


def _persistent_blocks_per_compute_unit(
    stage: str,
    metadata: Mapping[str, Any],
    *,
    precision: str,
    backend: str,
    target: str,
) -> int:
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


def prepare_low_memory_operator_modules(
    evaluator,
    *,
    model_data: Mapping[str, Any],
    precision: str,
    backend: str,
    target: Mapping[str, Any],
    jit_generation_version: int,
    prefer_host_m0_plugin: bool = False,
    cache_root=None,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """Prepare low-memory M0/R0 stages, optionally preferring host M0 JIT."""

    from .jit import (
        normalize_execution_hip_target_identity,
        prepare_hiprtc_jit_artifact,
        prepare_jit_artifact,
        prepare_nvrtc_jit_artifact,
        quarantine_jit_artifact,
        remove_jit_quarantine,
    )
    from .jit_operator_codegen import (
        render_jit_m0_device_module,
        render_jit_m0_host_plugin,
        render_jit_r0_device_module,
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
    elif backend == "hip":
        identity = normalize_execution_hip_target_identity(target)
        features = identity["target_features"]
        module_target = identity["architecture"] + (
            ":" + ":".join(features) if features else ""
        )
        compiler = "hiprtc"
    elif backend == "host":
        module_target = "host"
        compiler = "host-cxx"
    else:
        raise RuntimeError("low-memory operator JIT requires host, CUDA, or HIP")

    contracts = model_data.get("execution_contracts", {})
    modules: dict[str, dict[str, Any]] = {}
    diagnostics: tuple[str, ...] = ()
    device_stage_specs = (
        (
            "M0",
            "m0_implementation",
            "m0_device_module_ready",
            "m0_device_module_artifact_id",
            "_load_m0_device_module",
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
            "R0",
            "r0_implementation",
            "r0_host_plugin_ready",
            "r0_host_plugin_artifact_id",
            "_load_r0_host_plugin",
        ),
    )
    stage_specs = host_stage_specs if backend == "host" else device_stage_specs
    for (
        stage,
        implementation_attribute,
        ready_attribute,
        artifact_attribute,
        loader_name,
    ) in stage_specs:
        implementation = str(getattr(evaluator, implementation_attribute, "generic"))
        force_host_m0_plugin = (
            prefer_host_m0_plugin and backend == "host" and stage == "M0"
        )
        if implementation == "builtin" and not force_host_m0_plugin:
            modules[stage] = {
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
            continue
        contract = contracts.get(stage)
        if not isinstance(contract, dict):
            raise RuntimeError(
                f"low-memory execution requires an Execution {stage} contract"
            )
        if backend == "host" and stage != "M0":
            raise RuntimeError(
                "host low-memory JIT currently requires a built-in R0 implementation"
            )
        if stage == "M0":
            schedule = os.environ.get("SYMMETRIX_M0_RTC_SCHEDULE", "chunk32")
            if backend == "host":
                source, metadata = render_jit_m0_host_plugin(
                    contract, precision=precision, schedule=schedule
                )
            else:
                source, metadata = render_jit_m0_device_module(
                    contract,
                    precision=precision,
                    target=module_target,
                    schedule=schedule,
                )
        else:
            schedule = "edge"
            source, metadata = render_jit_r0_device_module(
                contract, precision=precision, target=module_target
            )
        common = {
            "abi": {
                "tag": (
                    "symmetrix.jit.m0-host-plugin/1"
                    if backend == "host"
                    else "symmetrix.jit.operator-module/1"
                ),
                "version": 1,
            },
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
            arguments = {**common, "cxx_flags": None}
        elif backend == "cuda":
            prepare = prepare_nvrtc_jit_artifact
            arguments = {
                **common,
                "compute_capability": target,
                "target": target,
            }
        else:
            prepare = prepare_hiprtc_jit_artifact
            arguments = {**common, "target": target}
        if cache_root is not None:
            arguments["cache_root"] = cache_root
        loader = getattr(evaluator, loader_name, None)
        if loader is None:
            raise RuntimeError(
                f"the native evaluator does not expose the {stage} device-module loader"
            )

        retained_quarantines = []
        failed_artifacts = set()
        result = None
        stage_diagnostics: tuple[str, ...] = ()
        for _load_attempt in range(4):
            result = prepare(source, **arguments)
            stage_diagnostics = (*stage_diagnostics, *result.diagnostics)
            if not result.available or result.artifact_path is None:
                raise RuntimeError(
                    result.reason
                    or f"the {stage} RTC cache did not produce an artifact"
                )
            try:
                persistent_blocks = _persistent_blocks_per_compute_unit(
                    stage,
                    metadata,
                    precision=precision,
                    backend=backend,
                    target=module_target,
                )
                if stage == "M0" and backend != "host":
                    loader(str(result.artifact_path), schedule, persistent_blocks)
                elif backend != "host":
                    loader(str(result.artifact_path), persistent_blocks)
                else:
                    loader(str(result.artifact_path))
                break
            except Exception as load_error:
                failed = (result.cache_key, str(result.artifact_path))
                if failed in failed_artifacts:
                    raise
                failed_artifacts.add(failed)
                quarantine = quarantine_jit_artifact(
                    result, f"{type(load_error).__name__}: {load_error}"
                )
                retained_quarantines.append(quarantine)
                stage_diagnostics = (
                    *stage_diagnostics,
                    *quarantine.diagnostics,
                )
        else:
            raise RuntimeError(
                f"Execution {stage} RTC artifact recovery attempts were exhausted"
            )
        if not bool(getattr(evaluator, ready_attribute, False)):
            raise RuntimeError(
                f"the evaluator did not retain the loaded {stage} device module"
            )
        for quarantine in retained_quarantines:
            stage_diagnostics = (
                *stage_diagnostics,
                *remove_jit_quarantine(quarantine),
            )
        assert result is not None
        diagnostics = (*diagnostics, *stage_diagnostics)
        modules[stage] = {
            "status": result.status,
            "implementation": ("host_plugin" if backend == "host" else "device_module"),
            "compiler": compiler,
            "cache_key": result.cache_key,
            "artifact_path": str(result.artifact_path),
            "artifact_id": getattr(
                evaluator, artifact_attribute, metadata["artifact_id"]
            ),
            "schedule": schedule,
            "persistent_blocks_per_compute_unit": persistent_blocks,
            "target": module_target,
            "diagnostics": stage_diagnostics,
        }
    return modules, diagnostics


__all__ = ["prepare_low_memory_operator_modules"]
