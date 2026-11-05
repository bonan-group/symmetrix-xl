"""Compact smoke coverage for OAM release-evidence handling.

These tests cover the release-critical paths once: manifest and matrix
declaration, provenance binding, artifact preparation, numerical/timing
evidence, orchestration, and the CPU/CUDA policy gates.  Field-by-field
mutation permutations belong to the lower-level contract tests and are not
duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import (
    nequip_oam_equivalence,
    oam_cuda_r1_timing,
    oam_release_qualification,
)
from benchmarks.nequip_oam_kokkos_profile import (
    profile,
)
from benchmarks.oam_cuda_r1_timing import (
    _select_physical_cpus,
    prepare_cuda_direct_artifact,
    prepare_diagnostic_host_direct_artifact,
)
from benchmarks.oam_cuda_r1_timing import (
    make_parser as make_cuda_timing_parser,
)
from benchmarks.oam_qualification_record import (
    _load_numerical_evidence,
    build_cell_record,
)
from benchmarks.oam_release_contract import OAM_RELEASE_TOLERANCES
from benchmarks.oam_release_qualification import (
    QualificationError,
    build_declared_cells,
    build_driver_command,
    build_driver_environment,
    evaluate_qualification,
    orchestrate,
    validate_cell_record,
)
from symmetrix.oam_cuda_artifact import OAMCudaArtifactResult
from symmetrix.oam_jit_artifact import OAMHostArtifactResult


def test_release_and_equivalence_drivers_share_authoritative_tolerances():
    assert oam_release_qualification.CORRECTNESS_TOLERANCES is OAM_RELEASE_TOLERANCES
    assert (
        nequip_oam_equivalence.DEFAULT_TOLERANCES is OAM_RELEASE_TOLERANCES["float32"]
    )
    assert (
        nequip_oam_equivalence.FLOAT64_TOLERANCES is OAM_RELEASE_TOLERANCES["float64"]
    )


@pytest.fixture(autouse=True)
def _resolved_installed_artifact(monkeypatch, tmp_path):
    """Keep synthetic records independent of a compiled qualification wheel."""

    manifest_path = tmp_path / "installed-artifact-manifest.json"
    manifest_path.write_text("{}")

    def resolve(request):
        direct = _raw_record(request, "direct")
        artifact = direct["artifact"]
        result = {
            "status": artifact["actual_status"],
            "origin": artifact["actual_origin"],
            "identity": artifact["actual_identity"],
            "artifact_path": direct["artifact_path"],
            "manifest_path": str(manifest_path),
            "sha256": artifact["sha256"],
        }
        if request["backend"] == "host":
            result.update(
                native_build_target=artifact["native_build_target"],
                active_host_target=artifact["active_host_target"],
                floating_point_policy=artifact["actual_floating_point_policy"],
            )
        else:
            result.update(
                compiler=artifact["actual_compiler"], target=artifact["target"]
            )
        return result

    monkeypatch.setattr(
        oam_release_qualification, "_probe_current_installed_artifact", resolve
    )


def test_ordinary_cuda_preparation_passes_runtime_native_identity(
    tmp_path,
):
    contract = tmp_path / "oam.json"
    contract.write_text("{}")
    artifact = object()
    native = object()

    def prepare(path, precision, target, **kwargs):
        assert (path, precision, target) == (contract, "float64", "sm90")
        assert kwargs == {
            "compiler": "nvcc",
            "threads_per_block": 64,
            "generation_policy": None,
            "native_module": native,
        }
        return artifact

    assert (
        prepare_cuda_direct_artifact(
            prepare,
            contract=contract,
            precision="float64",
            target="sm90",
            native=native,
            compiler="nvcc",
            threads_per_block=64,
            generation_policy=None,
        )
        is artifact
    )


def test_ordinary_host_preparation_returns_artifact_and_canonical_policy():
    artifact = SimpleNamespace(
        metadata={
            "generation_policy": {
                "full_reverse_reduction_policy": "owner-local-deterministic-v1",
            }
        }
    )

    def prepare(contract, precision):
        assert (contract, precision) == ("model.json", "float32")
        return artifact

    actual, policy = prepare_diagnostic_host_direct_artifact(
        prepare, "model.json", "float32"
    )
    assert actual is artifact
    assert policy == artifact.metadata["generation_policy"]
    assert policy is not artifact.metadata["generation_policy"]


def _manifest(tmp_path: Path) -> dict:
    host_native = tmp_path / "host-native.so"
    cuda_native = tmp_path / "cuda-native.so"
    host_native.write_bytes(b"host-native")
    cuda_native.write_bytes(b"cuda-native")
    models = {}
    for size in ("S", "M", "L", "XL"):
        path = tmp_path / f"oam-{size.lower()}.json"
        path.write_text(json.dumps({"architecture": "oam", "r_cut": 4.5}))
        models[size] = {"path": str(path)}
    manifest = {
        "schema": "symmetrix.oam-release-qualification-manifest",
        "version": 3,
        "models": models,
        "graphs": {
            "host-100": {
                "structure": "srtio3",
                "repeat": [2, 2, 5],
                "seed": 7,
                "rattle": 0.05,
                "skin": 0.5,
            },
            "small": {
                "structure": "srtio3",
                "repeat": 2,
                "seed": 7,
                "rattle": 0.05,
                "skin": 0.5,
            },
            "production": {
                "structure": "srtio3",
                "repeat": 10,
                "seed": 7,
                "rattle": 0.05,
                "skin": 0.5,
            },
        },
        "host_target": "x86-64-v3",
        "host_floating_point_policy": "fast",
        "cuda_target": "sm90",
        "output_root": str((tmp_path / "qualification").resolve()),
        "capacity_boundaries": [
            {
                "model_size": "XL",
                "backend": "cuda",
                "precision": "float32",
                "graph_class": "production",
            }
        ],
        "direct_baselines_us_per_atom": {},
        "provenance": {
            "source_root": str(Path.cwd()),
            "native_paths": {
                "host": str(host_native),
                "cuda": str(cuda_native),
            },
        },
        "required_bindings": [
            {
                "operator_id": "edge.0",
                "capability_key": "edge_tensor_product",
                "implementation_id": "generated",
                "selected_implementation_id": "generated",
                "tier": "generated",
                "qualified": True,
                "portable_covered": False,
                "artifact_required": True,
            }
        ],
        "required_reference_bindings": [
            {
                "operator_id": "edge.0",
                "capability_key": "edge_tensor_product",
                "implementation_id": "portable_oam_kokkos_generic",
                "selected_implementation_id": "portable_oam_kokkos_generic",
                "tier": "portable",
                "qualified": True,
                "portable_covered": True,
                "artifact_required": False,
            }
        ],
    }
    for cell in build_declared_cells(manifest):
        identity = f"approved/{cell['cell_id']}"
        manifest["direct_baselines_us_per_atom"][cell["cell_id"]] = {
            "identity": identity,
            "sha256": "b" * 64,
            "us_per_atom": 8.0,
        }
    return manifest


def _request_fingerprint(request):
    payload = dict(request)
    payload.pop("request_fingerprint", None)
    payload.pop("jit_cache_path", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rebind_request(request):
    request["request_fingerprint"] = _request_fingerprint(request)
    request["jit_cache_path"] = str(
        Path(request["output_root"])
        / "jit"
        / request["cell_id"]
        / request["request_fingerprint"]
    )


def _cuda_request_context(tmp_path):
    request = next(
        cell
        for cell in build_declared_cells(_manifest(tmp_path))
        if cell["backend"] == "cuda"
    )
    command = build_driver_command({}, request, "direct")
    return request, command


def test_sm120_fp32_request_locks_qualified_warp_task_schedule(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["cuda_target"] = "sm120"
    request = next(
        cell for cell in build_declared_cells(manifest) if cell["backend"] == "cuda"
    )
    assert request["precision"] == "float32"
    assert request["generation_policy_request"] == {
        "forward_live_output_budget": 32,
        "forward_grouping_mode": "source_first_fit",
        "forward_group_execution": "warp_tasks",
        "forward_input_mode": "staged",
        "radial_adjoint_mode": "fp32",
        "path_reuse_mode": "none",
        "reverse_group_execution": "serial",
    }


def _host_request_context(tmp_path):
    request = next(
        cell
        for cell in build_declared_cells(_manifest(tmp_path))
        if cell["backend"] == "host"
    )
    command = build_driver_command({}, request, "direct")
    validated = oam_release_qualification.validate_driver_qualification_request(
        request,
        command=command,
        backend="cpu",
        precision=request["precision"],
        contract_path=Path(request["model_path"]),
        route="direct",
        profile=request["profile"],
        edge_variant=request["edge_variant"],
        actual_target="x86-64-v3",
        native_path=Path(request["native_path"]),
        package_path=request["package_path"],
        python_executable=request["python_executable"],
    )
    return request, validated


def _validate_cuda_request(request, command, *, actual_target="sm90", context=None):
    validator = getattr(
        oam_release_qualification, "validate_driver_qualification_request", None
    )
    assert callable(validator), "the driver qualification trust validator is missing"
    binding = request if context is None else context
    return validator(
        request,
        command=command,
        backend="cuda",
        precision="float32",
        contract_path=Path(binding["model_path"]),
        route="direct",
        profile="speed",
        edge_variant="full_message",
        actual_target=actual_target,
        native_path=Path(binding["native_path"]),
        package_path=binding["package_path"],
        python_executable=binding["python_executable"],
    )


def test_complete_v2_request_produces_validated_request(tmp_path):
    request, command = _cuda_request_context(tmp_path)

    validated = _validate_cuda_request(request, command)

    assert validated == request
    assert isinstance(validated, dict)


def test_request_with_wrong_fingerprint_is_rejected(tmp_path):
    request, command = _cuda_request_context(tmp_path)
    request["request_fingerprint"] = "9" * 64

    with pytest.raises(QualificationError, match="fingerprint"):
        _validate_cuda_request(request, command)


def _qualification_native(request, *, overrides=None, prebuilt_overrides=None):
    build_info = {
        "backend": "cuda" if request["backend"] == "cuda" else "cpu",
        "architecture": request["cuda_target"]
        if request["backend"] == "cuda"
        else request["host_target"],
        "source_commit": request["source_commit"],
        "source_dirty": request["source_dirty"],
        "source_content_sha256": request["source_fingerprint"],
        "native_source_content_sha256": "d" * 64,
        "fast_math": request["host_floating_point_policy"] == "fast",
        "prebuilt_operator_artifacts": {
            "mode": "r0",
            "backend": request["backend"],
            "target": request["cuda_target"]
            if request["backend"] == "cuda"
            else "host-isa-x86-64-v3",
            "source_content_sha256": request["source_fingerprint"],
            "native_source_content_sha256": "d" * 64,
            "catalog_sha256": "c" * 64,
            "artifact_count": 14,
            "installed_bytes": 4096,
            "compiler_id": "NVIDIA",
            "compiler_version": "13.0",
            "build_duration_seconds": 1.0,
        },
    }
    build_info.update(overrides or {})
    build_info["prebuilt_operator_artifacts"].update(prebuilt_overrides or {})
    return SimpleNamespace(_backend_build_info=lambda: build_info)


def _qualification_entry_live_arguments(request, command, native, active_cache):
    return {
        "command": command,
        "backend": "cpu" if request["backend"] == "host" else "cuda",
        "precision": request["precision"],
        "contract_path": Path(request["model_path"]),
        "route": "direct",
        "profile": request["profile"],
        "edge_variant": request["edge_variant"],
        "actual_target": (
            request["host_target"]
            if request["backend"] == "host"
            else {"architecture": "sm_90"}
        ),
        "native_path": Path(request["native_path"]),
        "package_path": request["package_path"],
        "python_executable": request["python_executable"],
        "native_module": native,
        "active_jit_cache": str(active_cache),
    }


def test_preflighted_worker_prepares_and_serializes_host_artifact(
    monkeypatch, tmp_path
):
    request = next(
        cell
        for cell in build_declared_cells(_manifest(tmp_path))
        if cell["backend"] == "host"
    )
    command = build_driver_command({}, request, "direct")
    cache = Path(request["jit_cache_path"])
    cache.mkdir(parents=True, mode=0o700)
    cache.chmod(0o700)
    native = _qualification_native(request)
    monkeypatch.setattr(
        oam_release_qualification,
        "_current_native_source_fingerprint",
        lambda source_root: "d" * 64,
    )
    artifact_path = tmp_path / "installed-host.so"
    artifact_path.write_bytes(b"installed-host")
    policy = {"owner_schedule": "whole_owner", "channel_tile": 1, "scalar_width": 4}
    artifact = OAMHostArtifactResult(
        status="installed",
        precision=request["precision"],
        model_path=Path(request["model_path"]),
        artifact_path=artifact_path,
        manifest_path=tmp_path / "host-manifest.json",
        cache_key="",
        artifact_id="host-id",
        semantic_fingerprint="sha256:" + "1" * 64,
        structure_fingerprint="sha256:" + "2" * 64,
        generation_fingerprint="sha256:" + "3" * 64,
        metadata={"generation_policy": policy},
        diagnostics=(),
        cache_hit=False,
        compiler_command=(),
        artifact_origin="prebuilt",
        active_host_target="host-isa-x86-64-v3",
        expected_host_target="host-isa-x86-64-v3",
        expected_artifact_origin="prebuilt",
        floating_point_policy="fast",
        expected_artifact_id="host-id",
    )

    def prepare(path, precision, **kwargs):
        assert (Path(path), precision) == (
            Path(request["model_path"]),
            request["precision"],
        )
        assert kwargs["installed_prebuilt_required"] is True
        assert kwargs["expected_host_target"] == "x86-64-v3"
        return artifact

    result = oam_cuda_r1_timing._qualification_execution_entry(
        request,
        artifact_preparer=prepare,
        contract=Path(request["model_path"]),
        native=native,
        target=request["host_target"],
        compiler=None,
        threads_per_block=None,
        generation_policy=None,
        **_qualification_entry_live_arguments(request, command, native, cache),
    )

    assert result["artifact"] is artifact
    assert result["generation_policy"] == policy
    assert result["artifact_evidence"]["expected_target"] == "x86-64-v3"
    assert result["artifact_evidence"]["native_build_target"] == "host-isa-x86-64-v3"
    assert result["artifact_evidence"]["active_host_target"] == "host-isa-x86-64-v3"


def test_preflighted_worker_rejects_cuda_artifact_with_wrong_canonical_target(
    monkeypatch, tmp_path
):
    request, command = _cuda_request_context(tmp_path)
    cache = Path(request["jit_cache_path"])
    cache.mkdir(parents=True, mode=0o700)
    cache.chmod(0o700)
    native = _qualification_native(
        request,
        overrides={"architecture": "sm_90"},
        prebuilt_overrides={"target": "sm_90"},
    )
    monkeypatch.setattr(
        oam_release_qualification,
        "_current_native_source_fingerprint",
        lambda source_root: "d" * 64,
    )
    artifact_path = tmp_path / "wrong-target.cubin"
    artifact_path.write_bytes(b"wrong-target")
    artifact = OAMCudaArtifactResult(
        status="installed",
        artifact_id="cuda-id",
        artifact_path=artifact_path,
        manifest_path=tmp_path / "cuda-manifest.json",
        compiler="prebuilt",
        target="sm120",
        diagnostics=(),
    )

    with pytest.raises(RuntimeError, match="target"):
        oam_cuda_r1_timing._qualification_execution_entry(
            request,
            artifact_preparer=lambda *args, **kwargs: artifact,
            contract=Path(request["model_path"]),
            native=native,
            target="sm_90",
            compiler=None,
            threads_per_block=128,
            generation_policy={"forward_live_output_budget": 16},
            **_qualification_entry_live_arguments(request, command, native, cache),
        )


def test_qualification_preflight_requires_bound_private_cache(monkeypatch, tmp_path):
    request, command = _cuda_request_context(tmp_path)
    cache = Path(request["jit_cache_path"])
    cache.mkdir(parents=True, mode=0o700)
    cache.chmod(0o700)
    native = _qualification_native(request)
    monkeypatch.setattr(
        oam_release_qualification,
        "_current_native_source_fingerprint",
        lambda source_root: "d" * 64,
    )

    for active in (None, str(tmp_path / "wrong-cache")):
        with pytest.raises(QualificationError, match="JIT cache"):
            oam_release_qualification.validate_driver_qualification_preflight(
                request,
                command=command,
                backend="cuda",
                precision="float32",
                contract_path=Path(request["model_path"]),
                route="direct",
                profile="speed",
                edge_variant="full_message",
                actual_target="sm90",
                native_path=Path(request["native_path"]),
                package_path=request["package_path"],
                python_executable=request["python_executable"],
                native_module=native,
                active_jit_cache=active,
            )


def test_qualification_preflight_rejects_nonprivate_or_foreign_cache(
    monkeypatch, tmp_path
):
    request, command = _cuda_request_context(tmp_path)
    cache = Path(request["jit_cache_path"])
    cache.mkdir(parents=True, mode=0o700)
    native = _qualification_native(request)
    monkeypatch.setattr(
        oam_release_qualification,
        "_current_native_source_fingerprint",
        lambda source_root: "d" * 64,
    )
    arguments = {
        "command": command,
        "backend": "cuda",
        "precision": "float32",
        "contract_path": Path(request["model_path"]),
        "route": "direct",
        "profile": "speed",
        "edge_variant": "full_message",
        "actual_target": "sm90",
        "native_path": Path(request["native_path"]),
        "package_path": request["package_path"],
        "python_executable": request["python_executable"],
        "native_module": native,
        "active_jit_cache": str(cache),
    }

    cache.chmod(0o755)
    with pytest.raises(QualificationError, match="0700"):
        oam_release_qualification.validate_driver_qualification_preflight(
            request, **arguments
        )
    cache.chmod(0o700)
    monkeypatch.setattr(
        oam_release_qualification.os,
        "getuid",
        lambda: cache.stat().st_uid + 1,
    )
    with pytest.raises(QualificationError, match="owner"):
        oam_release_qualification.validate_driver_qualification_preflight(
            request, **arguments
        )


@pytest.mark.parametrize(
    ("overrides", "prebuilt_overrides", "message"),
    (
        ({"backend": "cpu"}, {}, "backend"),
        ({"architecture": "sm120"}, {}, "target"),
        ({"source_commit": "0" * 40}, {}, "source_commit"),
        ({"source_content_sha256": "0" * 64}, {}, "source_content_sha256"),
        ({}, {"artifact_count": 0}, "artifact_count"),
        ({}, {"catalog_sha256": "invalid"}, "catalog_sha256"),
    ),
)
def test_qualification_preflight_rejects_every_native_provenance_mismatch(
    monkeypatch, tmp_path, overrides, prebuilt_overrides, message
):
    request, command = _cuda_request_context(tmp_path)
    cache = Path(request["jit_cache_path"])
    cache.mkdir(parents=True, mode=0o700)
    cache.chmod(0o700)
    monkeypatch.setattr(
        oam_release_qualification,
        "_current_native_source_fingerprint",
        lambda source_root: "d" * 64,
    )
    native = _qualification_native(
        request, overrides=overrides, prebuilt_overrides=prebuilt_overrides
    )

    def fail_preparer(*args, **kwargs):
        pytest.fail("artifact preparation must not run before preflight")

    with pytest.raises(QualificationError, match=message):
        oam_cuda_r1_timing._qualification_execution_entry(
            request,
            artifact_preparer=fail_preparer,
            contract=Path(request["model_path"]),
            native=native,
            target="sm90",
            compiler=None,
            threads_per_block=128,
            generation_policy=None,
            command=command,
            backend="cuda",
            precision="float32",
            contract_path=Path(request["model_path"]),
            route="direct",
            profile="speed",
            edge_variant="full_message",
            actual_target="sm90",
            native_path=Path(request["native_path"]),
            package_path=request["package_path"],
            python_executable=request["python_executable"],
            native_module=native,
            active_jit_cache=str(cache),
        )


def _cell_record(cell: dict, *, direct: float = 8.0, reference: float = 10.0) -> dict:
    direct_raw = _raw_record(cell, "direct")
    reference_raw = _raw_record(cell, "reference")
    for raw, value in ((direct_raw, direct), (reference_raw, reference)):
        raw["samples_us_per_atom_step"] = [value] * cell["samples"]
        raw["min_us_per_atom_step"] = value
        raw["median_us_per_atom_step"] = value
        raw["median_us_per_atom"] = value
    return build_cell_record(
        cell,
        direct_raw,
        reference_raw,
        baseline_us_per_atom=cell["baseline_us_per_atom"],
    )


def _raw_record(cell: dict, route: str) -> dict:
    direct = route == "direct"
    target = cell["host_target"] if cell["backend"] == "host" else cell["cuda_target"]
    command = build_driver_command({}, cell, route)
    _, cache = build_driver_environment(cell)
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache.chmod(0o700)
    graph_config = cell["graph_config"]
    repeat = graph_config.get("repeat")
    repeat_factors = (repeat, repeat, repeat) if isinstance(repeat, int) else repeat
    atom_count = (
        5 * repeat_factors[0] * repeat_factors[1] * repeat_factors[2]
        if graph_config["structure"] == "srtio3"
        else graph_config["atoms"]
    )
    record = {
        "schema": "symmetrix.oam-qualified-driver-record",
        "version": 3,
        "report_version": 3,
        "route": route,
        "execution_profile": cell["profile"],
        "qualification_cell": dict(cell),
        "backend": "cpu" if cell["backend"] == "host" else "cuda",
        "precision": cell["precision"],
        "implementation": (
            "receiver-owned-generated" if direct else "portable-oam-kokkos-reference"
        ),
        "model": cell["model_path"],
        "model_sha256": cell["model_sha256"],
        "graph_sha256": "7" * 64,
        "provenance": {
            "source_fingerprint": cell["source_fingerprint"],
            "source_commit": cell["source_commit"],
            "source_dirty": cell["source_dirty"],
            "native_sha256": cell["native_sha256"],
            "native_fast_math": True if cell["backend"] == "host" else None,
            "jit_cache": str(cache),
            "jit_cache_owner_uid": os.getuid(),
            "jit_cache_mode": 0o700,
        },
        "command": command,
        "python": command[0],
        "package": cell["package_path"],
        "native_extension": cell["native_path"],
        "native_sha256": cell["native_sha256"],
        "artifact": (
            {
                "kind": "host-direct",
                "expected_status": "installed",
                "actual_status": "installed",
                "expected_identity": "artifact-1",
                "actual_identity": "artifact-1",
                "expected_origin": "prebuilt",
                "actual_origin": "prebuilt",
                "sha256": cell["native_sha256"],
                "expected_target": "x86-64-v3",
                "native_build_target": "host-isa-x86-64-v3",
                "active_host_target": "host-isa-x86-64-v3",
                "expected_floating_point_policy": "fast",
                "actual_floating_point_policy": "fast",
            }
            if direct and cell["backend"] == "host"
            else {
                "kind": "cuda-direct",
                "expected_status": "installed",
                "actual_status": "installed",
                "expected_identity": "artifact-1",
                "actual_identity": "artifact-1",
                "expected_origin": "prebuilt",
                "actual_origin": "prebuilt",
                "sha256": cell["native_sha256"],
                "target": cell["cuda_target"],
                "expected_compiler": "prebuilt",
                "actual_compiler": "prebuilt",
                "schedule": dict(cell["generation_policy_request"]),
            }
            if direct
            else {"kind": "reference", "origin": "none"}
        ),
        "execution": {
            "plan_id": (
                "oam-direct-capacity"
                if cell["profile"] == "capacity"
                else "oam-direct-speed"
            )
            if direct
            else "portable_oam_kokkos_generic",
            "edge_variant": cell["edge_variant"] if direct else "full_message",
            "bindings": (
                [
                    {
                        **cell["required_bindings"][0],
                        "target": cell["backend"],
                        "precision": {
                            "compute": cell["precision"],
                            "accumulator": "float64",
                        },
                    }
                ]
                if direct
                else [
                    {
                        **cell["required_reference_bindings"][0],
                        "target": cell["backend"],
                        "precision": {
                            "compute": cell["precision"],
                            "accumulator": "float64",
                        },
                    }
                ]
            ),
            "binding_report_version": 1,
            "schedule": "capacity" if cell["profile"] == "capacity" else "throughput",
            "backend": cell["backend"],
            "target": "host" if cell["backend"] == "host" else cell["cuda_target"],
            "fallback_count": 0,
        },
        "graph": {
            "atom_count": atom_count,
            "cutoff": 4.5,
            "skin": 0.5,
            "effective_cutoff": 5.0,
            "directed_edges": 120000,
        },
        "memory": {
            "active_bytes": 100,
            "retained_bytes": 100 if cell["profile"] == "capacity" else 120,
            "peak_transient_bytes": 100,
            "replacements": 1,
        },
        "reuse": {"replacement_delta": 0, "completed": True},
        "capacity_boundary": {"completed": cell["kind"] == "capacity-boundary"},
        "cpu": {
            "physical_cores": cell["threads"] if cell["backend"] == "host" else None,
            "kokkos_threads": cell["threads"] if cell["backend"] == "host" else None,
            "blas_threads": 1 if cell["backend"] == "host" else None,
            "logical_cpu_ids": list(range(cell["threads"]))
            if cell["backend"] == "host"
            else None,
            "topology": [
                {"logical_cpu": cpu, "package": 0, "core": cpu}
                for cpu in range(cell["threads"])
            ]
            if cell["backend"] == "host"
            else None,
            "observed_kokkos_threads": cell["threads"]
            if cell["backend"] == "host"
            else None,
            "observed_openmp_threads": cell["threads"]
            if cell["backend"] == "host"
            else None,
        },
        "structure": graph_config["structure"],
        "rattle": graph_config["rattle"],
        "seed": graph_config["seed"],
        "skin": graph_config["skin"],
        "warmup": cell["warmups"],
        "sample_count": cell["samples"],
        "cpu_affinity": list(range(cell["threads"]))
        if cell["backend"] == "host"
        else [0],
        "thread_environment": {
            "KOKKOS_NUM_THREADS": str(cell["threads"])
            if cell["backend"] == "host"
            else None,
            "OMP_NUM_THREADS": str(cell["threads"])
            if cell["backend"] == "host"
            else None,
            "OPENBLAS_NUM_THREADS": "1" if cell["backend"] == "host" else None,
            "MKL_NUM_THREADS": "1" if cell["backend"] == "host" else None,
            "BLIS_NUM_THREADS": "1" if cell["backend"] == "host" else None,
            "NUMEXPR_NUM_THREADS": "1" if cell["backend"] == "host" else None,
        },
        "execution_environment": (
            {
                "available": True,
                "backend": "host",
                "execution_space": "OpenMP",
                "memory_space": "HostSpace",
                "target": "host",
            }
            if cell["backend"] == "host"
            else {
                "available": True,
                "backend": "cuda",
                "execution_space": "Cuda",
                "memory_space": "CudaSpace",
                "architecture": target,
                "device_ordinal": 0,
                "compute_capability_code": 90,
            }
        ),
        "observed_runtime": {
            "kokkos_execution_space": "OpenMP",
            "kokkos_concurrency": cell["threads"],
            "omp_max_threads": cell["threads"],
            "worker_actual_team_size": cell["threads"],
            "worker_cpu_observation_supported": True,
        }
        if cell["backend"] == "host"
        else None,
        "atoms": atom_count,
        "directed_edges": 120000,
        "model_cutoff": 4.5,
        "effective_cutoff": 5.0,
        "median_us_per_atom": 8.0 if direct else 10.0,
        "median_us_per_atom_step": 8.0 if direct else 10.0,
        "min_us_per_atom_step": 8.0 if direct else 10.0,
        "samples_us_per_atom_step": [8.0] * cell["samples"]
        if direct
        else [10.0] * cell["samples"],
        "energy": -1.0,
        "max_force": 0.5,
        "profile_capture": None,
        "generation_policy": dict(cell["generation_policy_request"])
        if direct
        else None,
        "generation_policy_selection": "automatic",
        "artifact_id": "artifact-1" if direct else "reference",
        "artifact_path": cell["native_path"] if direct else None,
        "artifact_cache_status": ("installed") if direct else None,
        "execution_diagnostic": "qualified",
        "workspace_bytes": 100,
        "topology_bytes": 10,
        "linear_workspace_bytes": 20,
        "model_preparation_seconds": 0.25,
        "timing_contract": {
            "unit": "us/atom",
            "timed_region": "compute_prepared+reduce_forces+reduce_stress",
            "excluded": [
                "artifact_compilation",
                "model_preparation",
                "graph_construction",
                "neighbor_construction",
                "warmup",
            ],
        },
    }
    count_name = "repeat" if graph_config["structure"] == "srtio3" else "atoms"
    record[count_name] = graph_config[count_name]
    if graph_config["structure"] == "chain":
        record["repeat"] = 10
    if cell["backend"] == "cuda" and direct:
        record.update(
            compiler="prebuilt",
            threads_per_block=128,
            forward_threads_per_block=128,
            cuda_launch_count=1,
            cuda_fallback_count=0,
        )
    _attach_numerical_evidence(cell, record, route)
    return record


def _attach_numerical_evidence(
    cell: dict,
    record: dict,
    route: str,
    *,
    energy: float = -1.0,
    force: float = 0.5,
    stress: float = 0.25,
) -> None:
    path = Path(cell[f"{route}_numerical_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_count = record["atoms"]
    np.savez(
        path,
        energy=np.asarray(energy, dtype=np.float64),
        forces=np.full((atom_count, 3), force, dtype=np.float64),
        stress=np.full((3, 3), stress, dtype=np.float64),
    )
    record["numerical_evidence"] = {
        "format": "npz-v1",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    record["energy"] = energy
    record["max_force"] = abs(force)


@pytest.mark.parametrize("bad_sample", (0.0, -1.0, float("nan"), float("inf")))
def test_raw_timing_requires_positive_finite_samples(tmp_path, bad_sample):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    direct["samples_us_per_atom_step"][0] = bad_sample
    with pytest.raises(QualificationError, match="samples_us_per_atom_step"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


def test_canonical_records_retain_and_pass_energy_force_stress_parity(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    _attach_numerical_evidence(cell, direct, "direct")
    _attach_numerical_evidence(cell, reference, "reference")

    record = build_cell_record(
        cell,
        direct,
        reference,
        baseline_us_per_atom=cell["baseline_us_per_atom"],
    )

    assert record["correctness"]["passed"] is True
    assert record["correctness"]["errors"] == {
        "energy_abs_eV_per_atom": 0.0,
        "forces_max_abs_eV_per_A": 0.0,
        "stress_max_abs_eV_per_A3": 0.0,
    }
    incomplete = evaluate_qualification([cell], [record])
    assert incomplete["gates"]["correctness"] == {
        "complete": False,
        "passed": False,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("energy", 1.0e6, "energy"),
        ("forces", 1.0e6, "forces"),
        ("stress", float("nan"), "finite"),
    ),
)
def test_correctness_gate_rejects_wrong_or_nonfinite_numerics(
    tmp_path, field, value, message
):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    _attach_numerical_evidence(cell, direct, "direct")
    kwargs = {"energy": -1.0, "force": 0.5, "stress": 0.25}
    kwargs[{"energy": "energy", "forces": "force", "stress": "stress"}[field]] = value
    _attach_numerical_evidence(cell, reference, "reference", **kwargs)

    with pytest.raises(QualificationError, match=message):
        build_cell_record(
            cell,
            direct,
            reference,
            baseline_us_per_atom=cell["baseline_us_per_atom"],
        )


def test_numerical_evidence_hash_and_parse_use_one_immutable_payload(
    tmp_path, monkeypatch
):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    path = Path(cell["direct_numerical_path"])
    replacement = tmp_path / "replacement.npz"
    np.savez(
        replacement,
        energy=np.asarray(-1.0),
        forces=np.full((100, 3), 0.5),
        stress=np.full((3, 3), float("nan")),
    )
    replacement_bytes = replacement.read_bytes()
    original_open = os.open
    replaced = set()

    def open_then_replace_path(path, flags):
        descriptor = original_open(path, flags)
        linked_path = Path(path)
        if linked_path == Path(cell["direct_numerical_path"]) and not replaced:
            staged = linked_path.with_suffix(".replacement.npz")
            staged.write_bytes(replacement_bytes)
            os.replace(staged, linked_path)
            replaced.add(linked_path)
        return descriptor

    monkeypatch.setattr(os, "open", open_then_replace_path)

    values = _load_numerical_evidence(cell, direct, "direct")

    assert np.isfinite(values["stress"]).all()
    assert float(values["energy"]) == -1.0
    assert replaced == {path}
    with np.load(path) as current:
        assert np.isnan(current["stress"]).all()


@pytest.mark.parametrize(
    "field", ("min_us_per_atom_step", "median_us_per_atom_step", "median_us_per_atom")
)
def test_raw_timing_recomputes_every_summary_from_samples(tmp_path, field):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    direct[field] += 0.25
    with pytest.raises(QualificationError, match=field):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


def test_retained_timing_must_equal_exact_raw_projection(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    record["timing"]["direct"]["samples_us_per_atom_step"][0] += 0.5
    with pytest.raises(QualificationError, match="timing"):
        validate_cell_record(cell, record)


@pytest.mark.parametrize("backend", ("host", "cuda"), ids=("host", "device"))
def test_execution_environment_schema_is_closed(tmp_path, backend):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == backend
    )
    direct = _raw_record(cell, "direct")
    direct["execution_environment"]["invented"] = True
    with pytest.raises(QualificationError, match="execution_environment"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


def test_generation_policy_schema_and_command_binding_are_closed(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    direct = _raw_record(cell, "direct")
    direct["generation_policy"] = {"invented": True}
    with pytest.raises(QualificationError, match="generation_policy"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda raw: raw.update(version="1"),
        lambda raw: raw.update(median_us_per_atom="8.0"),
        lambda raw: raw["artifact"].update(actual_origin="cache"),
    ),
)
def test_raw_records_are_strictly_validated_and_errors_normalized(tmp_path, mutation):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    mutation(direct)
    with pytest.raises(QualificationError):
        build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)


@pytest.mark.parametrize(
    "field",
    (
        "kind",
        "active_host_target",
        "sha256",
    ),
)
def test_host_direct_artifact_evidence_rejects_each_independent_mutation(
    tmp_path, field
):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "host"
    )
    direct = _raw_record(cell, "direct")
    direct["artifact"][field] = "mutated"
    with pytest.raises(QualificationError, match="artifact"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


@pytest.mark.parametrize(
    "field",
    (
        "kind",
        "sha256",
        "target",
    ),
)
def test_cuda_direct_artifact_contract_rejects_each_independent_mutation(
    tmp_path, field
):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    direct = _raw_record(cell, "direct")
    direct["artifact"][field] = "mutated"
    with pytest.raises(QualificationError, match="artifact"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "wrong-origin"))
def test_reference_artifact_contract_is_exact(tmp_path, mutation):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    reference = _raw_record(cell, "reference")
    if mutation == "missing":
        reference["artifact"].pop("origin")
    elif mutation == "extra":
        reference["artifact"]["identity"] = "reference"
    else:
        reference["artifact"]["origin"] = "cache"
    with pytest.raises(QualificationError, match="reference artifact"):
        build_cell_record(
            cell,
            _raw_record(cell, "direct"),
            reference,
            baseline_us_per_atom=8.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("actual_status", "cached"),
        ("actual_origin", "cache"),
        ("actual_origin", "runtime"),
        ("actual_compiler", "nvrtc"),
        ("actual_compiler", "nvcc"),
    ),
)
def test_cuda_release_prebuilt_rejects_other_valid_runtime_states(
    tmp_path, field, value
):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    direct = _raw_record(cell, "direct")
    direct["artifact"][field] = value
    if field == "actual_compiler":
        direct["compiler"] = value
    with pytest.raises(QualificationError, match="artifact"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


@pytest.mark.parametrize(
    ("backend", "field", "value"),
    (
        ("host", "sha256", "f" * 64),
        ("cuda", "target", "sm80"),
    ),
)
def test_retained_current_catalog_identity_is_revalidated(
    tmp_path, monkeypatch, backend, field, value
):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == backend
    )
    record = _cell_record(cell)
    current = dict(oam_release_qualification._probe_current_installed_artifact(cell))
    current[field] = value
    monkeypatch.setattr(
        oam_release_qualification,
        "_probe_current_installed_artifact",
        lambda request: current,
    )
    with pytest.raises(QualificationError, match="current installed catalog"):
        validate_cell_record(cell, record)


def _write_editable_import_interceptor(path: Path) -> None:
    path.mkdir()
    (path / "sitecustomize.py").write_text(
        "import importlib.abc, importlib.util, sys\n"
        "class TrapLoader(importlib.abc.Loader):\n"
        "    def create_module(self, spec): return None\n"
        "    def exec_module(self, module):\n"
        "        raise RuntimeError('editable-intercepted')\n"
        "class TrapFinder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'symmetrix.oam_jit_artifact':\n"
        "            return importlib.util.spec_from_loader(fullname, TrapLoader())\n"
        "        return None\n"
        "sys.meta_path.insert(0, TrapFinder())\n"
    )


def _write_fake_installed_artifact_module(
    path: Path, artifact_path: Path, manifest_path: Path, identity: str
) -> None:
    path.write_text(
        "from types import SimpleNamespace\n"
        "def prepare_nequip_oam_receiver_host_artifact(*args, **kwargs):\n"
        "    return SimpleNamespace(status='installed', artifact_origin='prebuilt', "
        f"artifact_id={identity!r}, artifact_path="
        f"__import__('pathlib').Path({str(artifact_path)!r}), manifest_path="
        f"__import__('pathlib').Path({str(manifest_path)!r}), "
        "active_host_target='host-isa-x86-64-v3', floating_point_policy='fast')\n"
    )


def _host_artifact_probe_request(
    tmp_path: Path, installed_package: Path, native_path: Path
) -> dict:
    model = tmp_path / "model.json"
    model.write_text("{}")
    return {
        "source_root": str(tmp_path / "source"),
        "package_path": str(installed_package / "__init__.py"),
        "native_path": str(native_path),
        "backend": "host",
        "python_executable": os.sys.executable,
        "model_path": str(model),
        "precision": "float32",
        "host_target": "x86-64-v3",
        "host_floating_point_policy": "fast",
    }


def test_qualification_contract_uses_schema_v3_and_rejects_stale_v2(tmp_path):
    assert oam_release_qualification.SCHEMA_VERSION == 3
    manifest = _manifest(tmp_path)
    manifest["version"] = 2
    with pytest.raises(QualificationError, match="schema/version"):
        build_declared_cells(manifest)

    manifest["version"] = 3
    cell = build_declared_cells(manifest)[0]
    direct = _raw_record(cell, "direct")
    direct["version"] = 2
    direct["report_version"] = 2
    with pytest.raises(QualificationError, match="driver schema/version"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=cell["baseline_us_per_atom"],
        )


def test_composer_rejects_route_independent_command_mismatch(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    reference["command"].extend(["--seed", "99"])
    with pytest.raises(QualificationError, match="command"):
        build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda direct, reference: reference.update(sample_count=99),
        lambda direct, reference: reference["thread_environment"].update(
            OMP_NUM_THREADS="99"
        ),
    ),
)
def test_composer_rejects_incompatible_paired_evidence(tmp_path, mutation):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    mutation(direct, reference)
    with pytest.raises(QualificationError):
        build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)


def test_huge_json_numeric_is_normalized_and_fails_closed(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    huge = 10**10000
    record["timing"]["direct_us_per_atom"] = huge
    record["evidence"]["direct"]["median_us_per_atom"] = huge

    with pytest.raises(QualificationError):
        validate_cell_record(cell, record)
    result = evaluate_qualification([cell], [record])
    assert result["status"] != "passed"
    assert cell["cell_id"] in result["failed_cells"]
    assert any("malformed" in reason for reason in result["unqualified_reasons"])


def test_manifest_rejects_unknown_graph_fields_and_requires_boundary(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["graphs"]["small"]["invented"] = 1
    with pytest.raises(QualificationError, match="graph"):
        build_declared_cells(manifest)

    manifest = _manifest(tmp_path)
    manifest["graphs"]["small"].pop("skin")
    with pytest.raises(QualificationError, match="graph"):
        build_declared_cells(manifest)

    manifest = _manifest(tmp_path)
    manifest["capacity_boundaries"] = []
    with pytest.raises(QualificationError, match="capacity boundary"):
        build_declared_cells(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_size", "L"),
        ("backend", "host"),
        ("precision", "float64"),
    ),
)
def test_manifest_requires_exact_single_cuda_xl_capacity_boundary(
    tmp_path, field, value
):
    manifest = _manifest(tmp_path)
    manifest["capacity_boundaries"][0][field] = value

    with pytest.raises(QualificationError, match="capacity boundary"):
        build_declared_cells(manifest)


def test_declared_matrix_is_complete_and_deterministic(tmp_path):
    first = build_declared_cells(_manifest(tmp_path))
    second = build_declared_cells(_manifest(tmp_path))

    assert first == second
    matrix = [cell for cell in first if cell["kind"] == "matrix"]
    scaling = [cell for cell in first if cell["kind"] == "cpu-scaling"]
    boundaries = [cell for cell in first if cell["kind"] == "capacity-boundary"]
    assert len(matrix) == 21
    assert len(scaling) == 5
    host = [cell for cell in first if cell["backend"] == "host"]
    assert [
        (cell["model_size"], cell["precision"], cell["threads"]) for cell in host
    ] == [
        ("S", "float32", 1),
        ("S", "float32", 4),
        ("M", "float32", 1),
        ("M", "float32", 4),
        ("M", "float64", 1),
        ("M", "float64", 4),
        ("L", "float32", 1),
        ("L", "float32", 4),
        ("XL", "float32", 1),
        ("XL", "float32", 4),
    ]
    assert {cell["graph_class"] for cell in host} == {"host-100"}
    assert all(cell["graph_config"]["repeat"] == [2, 2, 5] for cell in host)
    cuda = [cell for cell in matrix if cell["backend"] == "cuda"]
    assert len(cuda) == 16
    assert {cell["graph_class"] for cell in cuda} == {"small", "production"}
    assert len(boundaries) == 1
    assert len({cell["cell_id"] for cell in first}) == len(first)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record["provenance"].pop("model_sha256"), "model_sha256"),
        (
            lambda record: record["artifact"].pop("actual_origin"),
            "artifact.actual_origin",
        ),
        (lambda record: record["artifact"].pop("sha256"), "artifact.sha256"),
        (
            lambda record: record["execution"].update(backend="cuda"),
            "execution.backend",
        ),
        (lambda record: record["execution"].update(plan_id="wrong-plan"), "plan_id"),
        (
            lambda record: record["timing"].pop("direct_us_per_atom"),
            "direct_us_per_atom",
        ),
        (lambda record: record["execution"].update(fallback_count=1), "fallback_count"),
        (
            lambda record: record["reuse"].update(replacement_delta=1),
            "replacement_delta",
        ),
    ],
)
def test_cell_validation_rejects_missing_or_mismatched_evidence(
    tmp_path, mutation, message
):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    mutation(record)

    with pytest.raises(QualificationError, match=message):
        validate_cell_record(cell, record)


def test_thresholds_are_enforced_and_missing_baseline_is_unqualified(tmp_path):
    manifest = _manifest(tmp_path)
    first_matrix = next(
        cell for cell in build_declared_cells(manifest) if cell["kind"] == "matrix"
    )
    manifest["direct_baselines_us_per_atom"][first_matrix["cell_id"]] = None
    cells = [
        cell for cell in build_declared_cells(manifest) if cell["kind"] == "matrix"
    ]
    records = [_cell_record(cell) for cell in cells]

    result = evaluate_qualification(cells, records)

    assert result["status"] == "unqualified"
    assert result["gates"]["production_geomean"]["improvement"] == pytest.approx(0.25)
    assert result["gates"]["production_geomean"]["passed"] is True
    assert any("baseline" in reason for reason in result["unqualified_reasons"])

    bad = _cell_record(cells[0], direct=10.6, reference=10.0)
    records[0] = bad
    failed = evaluate_qualification(cells, records)
    assert failed["status"] == "failed"
    assert cells[0]["cell_id"] in failed["failed_cells"]


def test_cpu_four_thread_gate_requires_complete_nontrivial_speedup(tmp_path):
    cells = build_declared_cells(_manifest(tmp_path))
    records = [
        _cell_record(cell, direct=7.0 if cell["threads"] == 4 else 8.0)
        for cell in cells
    ]

    passed = evaluate_qualification(cells, records)

    assert passed["gates"]["cpu_scaling"]["complete"] is True
    assert passed["gates"]["cpu_scaling"]["passed"] is True
    assert passed["gates"]["cpu_scaling"]["minimum_speedup"] == pytest.approx(8.0 / 7.0)

    slow_id = next(
        cell["cell_id"]
        for cell in cells
        if cell["backend"] == "host" and cell["threads"] == 4
    )
    slow_index = next(
        index for index, record in enumerate(records) if record["cell_id"] == slow_id
    )
    records[slow_index] = _cell_record(cells[slow_index], direct=8.1)
    failed = evaluate_qualification(cells, records)
    assert failed["gates"]["cpu_scaling"]["passed"] is False
    assert slow_id in failed["failed_cells"]


def test_capacity_gate_requires_identical_actual_graph_evidence(tmp_path):
    cells = build_declared_cells(_manifest(tmp_path))
    boundary = next(cell for cell in cells if cell["kind"] == "capacity-boundary")
    speed = next(
        cell
        for cell in cells
        if cell["kind"] == "matrix"
        and all(
            cell[field] == boundary[field]
            for field in ("model_size", "backend", "precision", "graph_class")
        )
    )
    capacity_record = _cell_record(boundary)
    speed_record = _cell_record(speed)
    capacity_record["graph"]["directed_edges"] += 2
    capacity_record["provenance"]["graph_sha256"] = "9" * 64
    for raw in capacity_record["evidence"].values():
        raw["graph"]["directed_edges"] += 2
        raw["graph_sha256"] = "9" * 64

    result = evaluate_qualification([speed, boundary], [speed_record, capacity_record])

    assert result["gates"]["capacity"]["passed"] is False
    assert any("graph" in reason for reason in result["unqualified_reasons"])


def test_orchestration_is_restartable_and_never_invents_missing_results(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "qualification"
    calls = []

    def run(cell, path):
        calls.append(cell["cell_id"])
        direct = 7.0 if cell["backend"] == "host" and cell["threads"] == 4 else 8.0
        path.write_text(json.dumps(_cell_record(cell, direct=direct)))

    first = orchestrate(manifest, output, execute=True, runner=run)
    second = orchestrate(manifest, output, execute=True, runner=run)

    assert first["qualification"]["status"] == "passed"
    assert len(calls) == len(first["cells"])
    assert second["resumed_cells"] == len(first["cells"])
    assert second["qualification"]["status"] == "passed"
    assert all(item["state"] == "measured" for item in second["cells"])

    broken_id = first["cells"][0]["cell_id"]
    (output / "cells" / f"{broken_id}.json").write_text("{}")
    inspect_only = orchestrate(manifest, output, execute=False)
    broken = next(
        item for item in inspect_only["cells"] if item["cell_id"] == broken_id
    )
    assert broken["state"] == "unqualified"
    assert "invalid retained record" in broken["reason"]

    missing_model_manifest = _manifest(tmp_path)
    Path(missing_model_manifest["models"]["S"]["path"]).unlink()
    unqualified = orchestrate(
        missing_model_manifest, tmp_path / "missing", execute=False
    )
    affected = [item for item in unqualified["cells"] if item["model_size"] == "S"]
    assert affected
    assert all(item["state"] == "unqualified" for item in affected)
    assert all("model" in item["reason"] for item in affected)
    assert unqualified["qualification"]["status"] == "unqualified"


def test_host_driver_declares_reference_and_capacity_without_running(tmp_path):
    model = tmp_path / "oam.json"
    model.write_text(json.dumps({"architecture": "oam", "r_cut": 4.5}))

    record = profile(
        model,
        dry_run=True,
        routing="reference",
        execution_profile="capacity",
        reuse_prepared_graph=True,
    )

    assert record["routing"] == "reference"
    assert record["execution_profile"] == "capacity"
    assert record["generated_qualification"]["required"] is False
    assert record["timing_contract"]["timed_region"] == (
        "compute_prepared+reduce_forces+reduce_stress"
    )
    assert set(record["timing_contract"]["excluded"]) >= {
        "artifact_compilation",
        "model_preparation",
        "graph_construction",
        "neighbor_construction",
        "warmup",
    }


def test_cuda_driver_accepts_route_profile_and_qualification_cell(tmp_path):
    contract = tmp_path / "oam.json"
    request = {"cell_id": "matrix-s-cuda-float32-small-speed-t1"}

    args = make_cuda_timing_parser().parse_args(
        [
            "cuda",
            "float32",
            "--contract",
            str(contract),
            "--routing",
            "direct",
            "--execution-profile",
            "capacity",
            "--qualification-cell",
            json.dumps(request),
            "--threads",
            "4",
        ]
    )

    assert args.routing == "direct"
    assert args.execution_profile == "capacity"
    assert json.loads(args.qualification_cell) == request
    assert args.threads == 4


def test_driver_environment_has_a_private_per_cell_jit_cache(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]

    environment, cache = build_driver_environment(cell)

    assert environment["SYMMETRIX_JIT_CACHE"] == str(cache)
    assert cell["cell_id"] in str(cache)


def test_cpu_scaling_selects_distinct_physical_cores():
    topology = {
        0: (0, 0),
        1: (0, 0),
        2: (0, 1),
        3: (0, 1),
        4: (0, 2),
    }

    assert _select_physical_cpus([0, 1, 2, 3, 4], 3, topology=topology) == [0, 2, 4]
    assert _select_physical_cpus(
        [0, 1, 2, 3, 4], 2, start_cpu=1, topology=topology
    ) == [1, 2]
