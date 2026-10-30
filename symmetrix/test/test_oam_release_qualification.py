from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
    BLAS_THREAD_ENVIRONMENT,
    _configure_process,
    _timed_properties,
    build_srtio3,
    profile,
)
from benchmarks.nequip_oam_kokkos_profile import (
    make_parser as make_host_profile_parser,
)
from benchmarks.oam_cuda_r1_timing import (
    _select_physical_cpus,
    admit_direct_candidate,
    prepare_cuda_direct_artifact,
    prepare_diagnostic_host_direct_artifact,
    prepare_direct_evaluator,
    serialize_diagnostic_artifact_evidence,
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
    validate_manifest,
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


def test_timing_driver_runs_by_absolute_path_outside_the_source_root(tmp_path):
    driver = Path(oam_cuda_r1_timing.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(driver), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Measure OAM generated-R1" in completed.stdout

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy\n"
                f"namespace = runpy.run_path({str(driver)!r})\n"
                "validator = namespace['validate_driver_qualification_preflight']\n"
                "module = __import__(validator.__module__, fromlist=['*'])\n"
                f"source = module._current_source_provenance({str(driver.parents[1])!r})\n"
                "print(validator.__module__)\n"
                "print(source['source_exists'])\n"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.splitlines() == ["benchmarks.oam_release_qualification", "True"]


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


def test_ordinary_cuda_preparation_preserves_runtime_compiler_and_omits_native(
    tmp_path,
):
    contract = tmp_path / "oam.json"
    contract.write_text("{}")
    artifact = object()

    def prepare(path, precision, target, **kwargs):
        assert (path, precision, target) == (contract, "float64", "sm90")
        assert kwargs == {
            "compiler": "nvcc",
            "threads_per_block": 64,
            "generation_policy": None,
        }
        return artifact

    assert (
        prepare_cuda_direct_artifact(
            prepare,
            contract=contract,
            precision="float64",
            target="sm90",
            native=object(),
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


def test_strict_qualification_operations_are_not_module_level_capabilities():
    assert not hasattr(oam_cuda_r1_timing, "_qualification_execution_services")
    assert not hasattr(oam_cuda_r1_timing, "prepare_host_direct_artifact")
    assert not hasattr(oam_cuda_r1_timing, "prepare_qualification_cuda_direct_artifact")
    assert not hasattr(oam_cuda_r1_timing, "serialize_artifact_evidence")


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


@pytest.mark.parametrize("payload", ({}, {"cuda_target": "sm90"}))
def test_empty_or_partial_qualification_request_is_rejected(tmp_path, payload):
    valid, command = _cuda_request_context(tmp_path)

    with pytest.raises(QualificationError):
        _validate_cuda_request(payload, command, context=valid)


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("model_path", "/wrong/model.json"),
        ("model_sha256", "9" * 64),
        ("backend", "host"),
        ("precision", "float64"),
        ("profile", "capacity"),
        ("edge_variant", "projected"),
        ("cuda_target", "sm120"),
        ("native_sha256", "8" * 64),
    ),
)
def test_recomputed_request_with_wrong_live_binding_is_rejected(tmp_path, field, wrong):
    request, _ = _cuda_request_context(tmp_path)
    request[field] = wrong
    _rebind_request(request)
    command = build_driver_command({}, request, "direct")

    with pytest.raises(QualificationError):
        _validate_cuda_request(request, command)


def test_request_with_wrong_fingerprint_is_rejected(tmp_path):
    request, command = _cuda_request_context(tmp_path)
    request["request_fingerprint"] = "9" * 64

    with pytest.raises(QualificationError, match="fingerprint"):
        _validate_cuda_request(request, command)


def test_request_with_wrong_canonical_command_is_rejected(tmp_path):
    request, command = _cuda_request_context(tmp_path)
    command = [*command, "--seed", "99"]

    with pytest.raises(QualificationError, match="canonical"):
        _validate_cuda_request(request, command)


def test_actual_cuda_target_is_validated_and_never_replaced_by_request(tmp_path):
    request, command = _cuda_request_context(tmp_path)

    with pytest.raises(QualificationError, match="actual.*target"):
        _validate_cuda_request(request, command, actual_target="sm120")


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


def test_preflighted_worker_reads_cuda_policy_from_installed_manifest(
    monkeypatch, tmp_path
):
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
    artifact_path = tmp_path / "installed-cuda.cubin"
    artifact_path.write_bytes(b"installed-cuda")
    policy = dict(request["generation_policy_request"])
    manifest_path = tmp_path / "cuda-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"key_inputs": {"build": {"metadata": {"generation_policy": policy}}}}
        )
    )
    artifact = OAMCudaArtifactResult(
        status="installed",
        artifact_id="cuda-id",
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        compiler="prebuilt",
        target="sm90",
        diagnostics=(),
    )

    result = oam_cuda_r1_timing._qualification_execution_entry(
        request,
        artifact_preparer=lambda *args, **kwargs: artifact,
        contract=Path(request["model_path"]),
        native=native,
        target="sm90",
        compiler=None,
        threads_per_block=128,
        generation_policy=None,
        **_qualification_entry_live_arguments(request, command, native, cache),
    )

    assert result["generation_policy"] == policy
    assert result["artifact_evidence"]["schedule"] == policy


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


def test_driver_preflight_aborts_before_any_qualification_service(
    monkeypatch, tmp_path
):
    request, command = _cuda_request_context(tmp_path)
    native = _qualification_native(request)
    monkeypatch.setattr(
        oam_release_qualification,
        "_current_native_source_fingerprint",
        lambda source_root: "d" * 64,
    )

    def fail_preparer(*args, **kwargs):
        pytest.fail("artifact preparation must not run before preflight")

    with pytest.raises(QualificationError, match="JIT cache"):
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
            active_jit_cache=None,
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
        ({"source_dirty": "invalid"}, {}, "source_dirty"),
        ({"source_content_sha256": "0" * 64}, {}, "source_content_sha256"),
        (
            {"native_source_content_sha256": "0" * 64},
            {},
            "native_source_content_sha256",
        ),
        ({"fast_math": "false"}, {}, "fast_math"),
        ({}, {"mode": "oam"}, "catalog mode"),
        ({}, {"backend": "host"}, "catalog backend"),
        ({}, {"target": "sm120"}, "catalog target"),
        ({}, {"source_content_sha256": "0" * 64}, "catalog source_content_sha256"),
        (
            {},
            {"native_source_content_sha256": "0" * 64},
            "catalog native_source_content_sha256",
        ),
        ({}, {"catalog_sha256": "invalid"}, "catalog_sha256"),
        ({}, {"artifact_count": 0}, "artifact_count"),
        ({}, {"installed_bytes": 0}, "installed_bytes"),
        ({}, {"compiler_id": ""}, "compiler_id"),
        ({}, {"compiler_version": ""}, "compiler_version"),
        ({}, {"build_duration_seconds": -1.0}, "build_duration_seconds"),
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
        ("stress", 1.0e6, "stress"),
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


def test_numerical_evidence_rejects_symlinked_bound_path(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    path = Path(cell["direct_numerical_path"])
    moved = path.with_name("moved.npz")
    path.rename(moved)
    path.symlink_to(moved)

    with pytest.raises(QualificationError, match="unavailable"):
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


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
def test_numerical_evidence_rejects_nonregular_bound_path(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    path = Path(cell["direct_numerical_path"])
    path.unlink()
    os.mkfifo(path)

    with pytest.raises(QualificationError, match="bounded regular file"):
        build_cell_record(
            cell,
            direct,
            reference,
            baseline_us_per_atom=cell["baseline_us_per_atom"],
        )


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


@pytest.mark.parametrize("route", ("direct", "reference"))
@pytest.mark.parametrize(
    "field",
    (
        "samples_us_per_atom_step",
        "min_us_per_atom_step",
        "median_us_per_atom_step",
        "median_us_per_atom",
    ),
)
def test_retained_nested_timing_rejects_boolean_numeric_substitutions(
    tmp_path, route, field
):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell, direct=1.0, reference=1.0)
    if field == "samples_us_per_atom_step":
        record["timing"][route][field][0] = True
    else:
        record["timing"][route][field] = True

    with pytest.raises(QualificationError, match="timing"):
        validate_cell_record(cell, record)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda timing: timing.update(invented=True),
        lambda timing: timing["excluded"].append("invented_phase"),
        lambda timing: timing["excluded"].reverse(),
    ),
    ids=("unknown-field", "extra-exclusion", "reordered-exclusions"),
)
def test_retained_top_level_timing_schema_and_projection_are_exact(tmp_path, mutation):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    mutation(record["timing"])
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


def test_host_runtime_target_is_distinct_from_host_isa_request(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "host"
    )
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    for raw in (direct, reference):
        raw["execution"]["target"] = "host"
        raw["execution_environment"]["target"] = "host"

    record = build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)

    assert record["evidence"]["direct"]["execution"]["target"] == "host"
    assert record["artifact"]["native_build_target"] == "host-isa-x86-64-v3"


@pytest.mark.parametrize(
    ("backend", "manifest_target", "environment_field", "numeric_target"),
    (("cuda", "90", "architecture", 90),),
    ids=("cuda-architecture",),
)
def test_execution_environment_target_requires_an_exact_string(
    tmp_path, backend, manifest_target, environment_field, numeric_target
):
    manifest = _manifest(tmp_path)
    manifest[f"{backend}_target"] = manifest_target
    cell = next(
        item for item in build_declared_cells(manifest) if item["backend"] == backend
    )
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    direct["execution"]["target"] = manifest_target
    reference["execution"]["target"] = manifest_target
    direct["execution_environment"][environment_field] = numeric_target
    reference["execution_environment"][environment_field] = numeric_target
    with pytest.raises(QualificationError, match="execution_environment"):
        build_cell_record(
            cell,
            direct,
            reference,
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
    ("backend", "field"),
    (
        ("host", "channel_tile"),
        ("cuda", "forward_live_output_budget"),
    ),
)
def test_generation_policy_integer_fields_reject_booleans(tmp_path, backend, field):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == backend
    )
    direct = _raw_record(cell, "direct")
    direct["generation_policy"][field] = True
    with pytest.raises(QualificationError, match="generation_policy"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


@pytest.mark.parametrize("root", ([], ["manifest"], None, True, 7, "manifest"))
def test_manifest_root_must_be_a_json_object(tmp_path, root):
    parsed_root = json.loads(json.dumps(root))
    with pytest.raises(QualificationError, match="manifest must be an object"):
        validate_manifest(parsed_root)


def test_bound_python_launcher_preserves_absolute_symlink(monkeypatch, tmp_path):
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(os.__file__).parents[2] / "bin" / "python3")
    monkeypatch.setattr(
        "benchmarks.oam_release_qualification.sys.executable", str(launcher)
    )
    cell = build_declared_cells(_manifest(tmp_path))[0]
    assert cell["python_executable"] == str(launcher.absolute())
    assert cell["python_real_executable"] == str(launcher.resolve())
    assert build_driver_command({}, cell, "direct")[0] == str(launcher.absolute())


def test_malformed_evidence_rows_and_capacity_values_fail_closed(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    result = evaluate_qualification([cell], [["not", "an", "object"]])
    assert result["status"] == "unqualified"
    assert any("evidence row" in reason for reason in result["unqualified_reasons"])


def test_direct_preparation_rejects_projected_before_factory():
    observed = {}

    def factory(context):
        observed.update(context)
        raise ValueError("native incompatibility")

    with pytest.raises(ValueError, match="unsupported direct edge variant"):
        prepare_direct_evaluator(
            factory,
            profile="capacity",
            variant="projected",
            precision="float32",
        )
    assert observed == {}


@pytest.mark.parametrize(
    "mutation",
    (
        lambda raw: raw.update(version="1"),
        lambda raw: raw.update(median_us_per_atom="8.0"),
        lambda raw: raw.update(median_us_per_atom=float("nan")),
        lambda raw: raw.update(generation_policy_selection="explicit"),
        lambda raw: raw["provenance"].update(native_sha256="not-a-digest"),
        lambda raw: raw["artifact"].update(actual_origin="cache"),
        lambda raw: raw["execution"]["bindings"][0].update(implementation_id="wrong"),
    ),
)
def test_raw_records_are_strictly_validated_and_errors_normalized(tmp_path, mutation):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    mutation(direct)
    with pytest.raises(QualificationError):
        build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)


def test_composed_record_preserves_both_raw_evidence_blocks(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    record = build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)
    assert record["evidence"] == {"direct": direct, "reference": reference}


def test_cuda_conditioned_self_memory_diagnostics_are_retained(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    direct.update(
        conditioned_self_bytes=4096,
        conditioned_self_schedule_bytes=1024,
        conditioned_self_original_tensor_bytes=8192,
    )

    record = build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)

    assert record["evidence"]["direct"]["conditioned_self_bytes"] == 4096
    assert record["evidence"]["direct"]["conditioned_self_schedule_bytes"] == 1024
    assert (
        record["evidence"]["direct"]["conditioned_self_original_tensor_bytes"] == 8192
    )


@pytest.mark.parametrize(
    "field",
    (
        "kind",
        "expected_status",
        "actual_status",
        "expected_identity",
        "actual_identity",
        "expected_origin",
        "actual_origin",
        "expected_target",
        "native_build_target",
        "active_host_target",
        "expected_floating_point_policy",
        "actual_floating_point_policy",
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


def test_host_evidence_keeps_request_and_native_targets_distinct(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "host"
    )
    record = _cell_record(cell)

    artifact = record["evidence"]["direct"]["artifact"]
    assert artifact["expected_target"] == "x86-64-v3"
    assert artifact["native_build_target"] == "host-isa-x86-64-v3"
    assert artifact["active_host_target"] == "host-isa-x86-64-v3"


@pytest.mark.parametrize(
    "field",
    (
        "kind",
        "expected_status",
        "actual_status",
        "expected_identity",
        "actual_identity",
        "expected_origin",
        "actual_origin",
        "sha256",
        "expected_target",
        "native_build_target",
        "active_host_target",
        "expected_floating_point_policy",
        "actual_floating_point_policy",
    ),
)
def test_retained_host_artifact_evidence_rejects_each_independent_mutation(
    tmp_path, field
):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "host"
    )
    record = _cell_record(cell)
    record["artifact"][field] = "f" * 64 if field == "sha256" else "mutated"
    with pytest.raises(QualificationError, match="artifact"):
        validate_cell_record(cell, record)


def test_raw_host_artifact_sha_rejects_mutation(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "host"
    )
    direct = _raw_record(cell, "direct")
    direct["artifact"]["sha256"] = "f" * 64
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
        "expected_status",
        "actual_status",
        "expected_identity",
        "actual_identity",
        "expected_origin",
        "actual_origin",
        "sha256",
        "target",
        "expected_compiler",
        "actual_compiler",
        "schedule",
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


@pytest.mark.parametrize(
    "field",
    (
        "kind",
        "expected_status",
        "actual_status",
        "expected_identity",
        "actual_identity",
        "expected_origin",
        "actual_origin",
        "sha256",
        "target",
        "expected_compiler",
        "actual_compiler",
        "schedule",
    ),
)
def test_retained_cuda_artifact_rejects_each_independent_mutation(tmp_path, field):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    record = _cell_record(cell)
    record["artifact"][field] = "f" * 64 if field == "sha256" else "mutated"
    with pytest.raises(QualificationError, match="artifact"):
        validate_cell_record(cell, record)


def test_retained_cuda_schedule_rejects_nested_boolean(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    record = _cell_record(cell)
    record["artifact"]["schedule"]["forward_live_output_budget"] = True
    with pytest.raises(QualificationError, match="artifact"):
        validate_cell_record(cell, record)


def test_cuda_record_accepts_maintained_build_target_spelling(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    direct = _raw_record(cell, "direct")
    direct["artifact"]["target"] = "sm_90"
    direct["execution_environment"]["architecture"] = "sm_90"
    direct["execution"]["target"] = "sm_90"
    reference = _raw_record(cell, "reference")
    reference["execution_environment"]["architecture"] = "sm_90"
    reference["execution"]["target"] = "sm_90"

    record = build_cell_record(
        cell,
        direct,
        reference,
        baseline_us_per_atom=8.0,
    )

    assert record["artifact"]["target"] == "sm_90"


def test_retained_cuda_probe_compares_canonical_target_identity(monkeypatch, tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    record = _cell_record(cell)
    original_probe = oam_release_qualification._probe_current_installed_artifact

    def probe(request):
        result = original_probe(request)
        result["target"] = "sm_90"
        return result

    monkeypatch.setattr(
        oam_release_qualification, "_probe_current_installed_artifact", probe
    )

    validate_cell_record(cell, record)


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


def test_raw_v1_and_old_runtime_or_cache_host_artifacts_do_not_qualify(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "host"
    )
    for mutation in (
        lambda raw: raw.update(version=1, report_version=1),
        lambda raw: raw["artifact"].update(actual_origin="cache"),
        lambda raw: raw["artifact"].update(actual_origin="runtime"),
    ):
        direct = _raw_record(cell, "direct")
        mutation(direct)
        with pytest.raises(QualificationError):
            build_cell_record(
                cell,
                direct,
                _raw_record(cell, "reference"),
                baseline_us_per_atom=8.0,
            )


def test_manifest_rejects_non_authorized_host_target(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["host_target"] = "native"
    with pytest.raises(QualificationError, match="x86-64-v3"):
        build_declared_cells(manifest)


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


def test_cuda_artifact_schedule_uses_json_type_strict_equality(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    direct = _raw_record(cell, "direct")
    direct["artifact"]["schedule"]["forward_live_output_budget"] = True
    with pytest.raises(QualificationError, match="artifact"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


def test_cuda_diagnostic_compiler_must_equal_artifact_compiler(tmp_path):
    cell = next(
        item
        for item in build_declared_cells(_manifest(tmp_path))
        if item["backend"] == "cuda"
    )
    direct = _raw_record(cell, "direct")
    direct["compiler"] = "nvrtc"
    with pytest.raises(QualificationError, match="compiler"):
        build_cell_record(
            cell,
            direct,
            _raw_record(cell, "reference"),
            baseline_us_per_atom=8.0,
        )


@pytest.mark.parametrize(
    ("backend", "field", "value"),
    (
        ("host", "status", "cached"),
        ("host", "origin", "cache"),
        ("host", "identity", "replacement-artifact"),
        ("host", "sha256", "f" * 64),
        ("host", "artifact_path", "/replacement/artifact.so"),
        ("host", "native_build_target", "native"),
        ("host", "active_host_target", "host-native"),
        ("host", "floating_point_policy", "strict"),
        ("cuda", "status", "cached"),
        ("cuda", "origin", "cache"),
        ("cuda", "identity", "replacement-artifact"),
        ("cuda", "sha256", "f" * 64),
        ("cuda", "artifact_path", "/replacement/artifact.cubin"),
        ("cuda", "compiler", "nvrtc"),
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


def test_retained_missing_current_catalog_manifest_is_rejected(tmp_path, monkeypatch):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    current = dict(oam_release_qualification._probe_current_installed_artifact(cell))
    current["manifest_path"] = str(tmp_path / "deleted-catalog.json")
    monkeypatch.setattr(
        oam_release_qualification,
        "_probe_current_installed_artifact",
        lambda request: current,
    )
    with pytest.raises(QualificationError, match="manifest is absent"):
        validate_cell_record(cell, record)


def test_retained_reference_shape_is_exact(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    record["evidence"]["reference"]["artifact"]["identity"] = "reference"
    with pytest.raises(QualificationError, match="reference artifact"):
        validate_cell_record(cell, record)


def test_retained_reference_kind_is_exact(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    record["evidence"]["reference"]["artifact"]["kind"] = "mutated"
    with pytest.raises(QualificationError, match="reference artifact"):
        validate_cell_record(cell, record)


@pytest.mark.parametrize("failure", ("missing-artifact", "replaced-bytes"))
def test_retained_current_artifact_bytes_are_revalidated(
    tmp_path, monkeypatch, failure
):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    current = dict(oam_release_qualification._probe_current_installed_artifact(cell))
    artifact = tmp_path / "current-artifact.so"
    if failure == "replaced-bytes":
        artifact.write_bytes(b"replacement")
    current["artifact_path"] = str(artifact)
    current["sha256"] = record["artifact"]["sha256"]
    record["evidence"]["direct"]["artifact_path"] = str(artifact)
    monkeypatch.setattr(
        oam_release_qualification,
        "_probe_current_installed_artifact",
        lambda request: current,
    )
    with pytest.raises(QualificationError, match="artifact (is absent|bytes)"):
        validate_cell_record(cell, record)


def test_retained_current_probe_failure_is_qualification_error(tmp_path, monkeypatch):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    monkeypatch.setattr(
        oam_release_qualification,
        "_probe_current_installed_artifact",
        lambda request: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    with pytest.raises(QualificationError, match="probe failed"):
        validate_cell_record(cell, record)


def test_current_installed_artifact_probe_preserves_backend_package(
    tmp_path, monkeypatch
):
    monkeypatch.undo()
    package_dir = tmp_path / "source" / "symmetrix" / "source" / "symmetrix"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    backend_dir = tmp_path / "fake_backend"
    backend_dir.mkdir()
    (backend_dir / "__init__.py").write_text("")
    native_path = backend_dir / "_native_cuda13_sm120.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    if __package__ != 'fake_backend':\n"
        "        raise RuntimeError('wrong-package')\n"
        "    raise RuntimeError('expected-package')\n"
    )

    with pytest.raises(QualificationError, match="expected-package"):
        oam_release_qualification._probe_current_installed_artifact(
            {
                "source_root": str(tmp_path / "source"),
                "package_path": str(package_dir / "__init__.py"),
                "native_path": str(native_path),
                "backend": "cuda",
                "python_executable": os.sys.executable,
            }
        )


def test_current_installed_artifact_probe_uses_bound_installed_package(
    tmp_path, monkeypatch
):
    """Catch loading source_root instead of the request's installed package_path."""

    monkeypatch.undo()
    source_package = tmp_path / "source/symmetrix/source/symmetrix"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text(
        "raise RuntimeError('wrong-source-package')\n"
    )
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    artifact_path = installed_package / "artifact.so"
    artifact_path.write_bytes(b"installed-artifact")
    manifest_path = installed_package / "manifest.json"
    manifest_path.write_text("{}")
    (installed_package / "oam_jit_artifact.py").write_text(
        "from types import SimpleNamespace\n"
        "def prepare_nequip_oam_receiver_host_artifact(*args, **kwargs):\n"
        "    return SimpleNamespace(status='installed', artifact_origin='prebuilt', "
        "artifact_id='installed-id', artifact_path="
        f"__import__('pathlib').Path({str(artifact_path)!r}), manifest_path="
        f"__import__('pathlib').Path({str(manifest_path)!r}), "
        "active_host_target='host-isa-x86-64-v3', floating_point_policy='fast')\n"
    )
    model = tmp_path / "model.json"
    model.write_text("{}")

    result = oam_release_qualification._probe_current_installed_artifact(
        {
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
    )

    assert result["identity"] == "installed-id"
    assert result["origin"] == "prebuilt"
    assert result["artifact_path"] == str(artifact_path)


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


def test_current_installed_artifact_probe_blocks_editable_submodule_interception(
    tmp_path, monkeypatch
):
    """Catch a meta-path finder replacing a bound installed package submodule."""

    monkeypatch.undo()
    trap = tmp_path / "trap"
    _write_editable_import_interceptor(trap)
    monkeypatch.setenv("PYTHONPATH", str(trap))
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    artifact_path = installed_package / "artifact.so"
    artifact_path.write_bytes(b"installed-artifact")
    manifest_path = installed_package / "manifest.json"
    manifest_path.write_text("{}")
    (installed_package / "oam_jit_artifact.py").write_text(
        "from types import SimpleNamespace\n"
        "def prepare_nequip_oam_receiver_host_artifact(*args, **kwargs):\n"
        "    return SimpleNamespace(status='installed', artifact_origin='prebuilt', "
        "artifact_id='installed-id', artifact_path="
        f"__import__('pathlib').Path({str(artifact_path)!r}), manifest_path="
        f"__import__('pathlib').Path({str(manifest_path)!r}), "
        "active_host_target='host-isa-x86-64-v3', floating_point_policy='fast')\n"
    )
    model = tmp_path / "model.json"
    model.write_text("{}")
    request = {
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

    result = oam_release_qualification._probe_current_installed_artifact(request)

    assert result["identity"] == "installed-id"


def test_current_installed_artifact_probe_does_not_fall_back_to_editable_submodule(
    tmp_path, monkeypatch
):
    """Catch fallback to an editable finder when the bound submodule is absent."""

    monkeypatch.undo()
    trap = tmp_path / "trap"
    _write_editable_import_interceptor(trap)
    monkeypatch.setenv("PYTHONPATH", str(trap))
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )

    with pytest.raises(QualificationError, match="No module named"):
        oam_release_qualification._probe_current_installed_artifact(
            {
                "source_root": str(tmp_path / "source"),
                "package_path": str(installed_package / "__init__.py"),
                "native_path": str(native_path),
                "backend": "host",
                "python_executable": os.sys.executable,
                "model_path": str(tmp_path / "model.json"),
                "precision": "float32",
                "host_target": "x86-64-v3",
                "host_floating_point_policy": "fast",
            }
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


def test_current_installed_artifact_probe_rejects_symlinked_bound_submodule(
    tmp_path, monkeypatch
):
    """Catch a package-local module symlink that escapes the installed root."""

    monkeypatch.undo()
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact_path = outside / "artifact.so"
    artifact_path.write_bytes(b"escaped-artifact")
    manifest_path = outside / "manifest.json"
    manifest_path.write_text("{}")
    outside_module = outside / "oam_jit_artifact.py"
    _write_fake_installed_artifact_module(
        outside_module, artifact_path, manifest_path, "escaped-id"
    )
    (installed_package / "oam_jit_artifact.py").symlink_to(outside_module)

    with pytest.raises(QualificationError, match="(symlink|outside)"):
        oam_release_qualification._probe_current_installed_artifact(
            _host_artifact_probe_request(tmp_path, installed_package, native_path)
        )


def test_current_installed_artifact_probe_rejects_package_path_escape(
    tmp_path, monkeypatch
):
    """Catch bound package code redirecting absolute submodules outside its root."""

    monkeypatch.undo()
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (installed_package / "__init__.py").write_text(
        f"__path__[:] = [{str(outside)!r}]\n"
    )
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    artifact_path = outside / "artifact.so"
    artifact_path.write_bytes(b"escaped-artifact")
    manifest_path = outside / "manifest.json"
    manifest_path.write_text("{}")
    _write_fake_installed_artifact_module(
        outside / "oam_jit_artifact.py", artifact_path, manifest_path, "escaped-id"
    )

    with pytest.raises(QualificationError, match="(namespace|outside)"):
        oam_release_qualification._probe_current_installed_artifact(
            _host_artifact_probe_request(tmp_path, installed_package, native_path)
        )


def test_current_installed_artifact_probe_rejects_resolver_artifact_escape(
    tmp_path, monkeypatch
):
    """Catch an in-root resolver returning artifact bytes from outside the wheel."""

    monkeypatch.undo()
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact_path = outside / "artifact.so"
    artifact_path.write_bytes(b"escaped-artifact")
    manifest_path = outside / "manifest.json"
    manifest_path.write_text("{}")
    _write_fake_installed_artifact_module(
        installed_package / "oam_jit_artifact.py",
        artifact_path,
        manifest_path,
        "escaped-id",
    )

    with pytest.raises(QualificationError, match="outside"):
        oam_release_qualification._probe_current_installed_artifact(
            _host_artifact_probe_request(tmp_path, installed_package, native_path)
        )


def test_current_installed_artifact_probe_reasserts_loaded_module_origin(
    tmp_path, monkeypatch
):
    """Catch a loaded qualification module rewriting its recorded origin."""

    monkeypatch.undo()
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    artifact_path = installed_package / "artifact.so"
    artifact_path.write_bytes(b"installed-artifact")
    manifest_path = installed_package / "manifest.json"
    manifest_path.write_text("{}")
    outside_origin = tmp_path / "outside/oam_jit_artifact.py"
    module_path = installed_package / "oam_jit_artifact.py"
    _write_fake_installed_artifact_module(
        module_path, artifact_path, manifest_path, "installed-id"
    )
    module_path.write_text(
        f"__spec__.origin = {str(outside_origin)!r}\n" + module_path.read_text()
    )

    with pytest.raises(QualificationError, match="outside"):
        oam_release_qualification._probe_current_installed_artifact(
            _host_artifact_probe_request(tmp_path, installed_package, native_path)
        )


def test_current_installed_artifact_probe_purges_preloaded_child_module(
    tmp_path, monkeypatch
):
    """Catch a startup hook preloading a forged child with a plausible origin."""

    monkeypatch.undo()
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    artifact_path = installed_package / "artifact.so"
    artifact_path.write_bytes(b"installed-artifact")
    manifest_path = installed_package / "manifest.json"
    manifest_path.write_text("{}")
    _write_fake_installed_artifact_module(
        installed_package / "oam_jit_artifact.py",
        artifact_path,
        manifest_path,
        "bound-id",
    )
    decoy = installed_package / "decoy.py"
    decoy.write_text("raise RuntimeError('decoy must not execute')\n")
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(
        "import importlib.util, sys, types\n"
        "from pathlib import Path\n"
        "module = types.ModuleType('symmetrix.oam_jit_artifact')\n"
        "module.__spec__ = importlib.util.spec_from_file_location(\n"
        f"    module.__name__, {str(decoy)!r})\n"
        "def prepare_nequip_oam_receiver_host_artifact(*args, **kwargs):\n"
        "    return types.SimpleNamespace(status='installed', "
        "artifact_origin='prebuilt', artifact_id='preloaded-id', "
        f"artifact_path=Path({str(artifact_path)!r}), "
        f"manifest_path=Path({str(manifest_path)!r}), "
        "active_host_target='host-isa-x86-64-v3', floating_point_policy='fast')\n"
        "module.prepare_nequip_oam_receiver_host_artifact = "
        "prepare_nequip_oam_receiver_host_artifact\n"
        "sys.modules[module.__name__] = module\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(startup))

    result = oam_release_qualification._probe_current_installed_artifact(
        _host_artifact_probe_request(tmp_path, installed_package, native_path)
    )

    assert result["identity"] == "bound-id"


def test_current_installed_artifact_probe_reasserts_standard_module_loader(
    tmp_path, monkeypatch
):
    """Catch a qualification module replacing its standard bound loader."""

    monkeypatch.undo()
    installed_package = tmp_path / "installed/symmetrix"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("")
    native_path = installed_package / "_native_cpu.py"
    native_path.write_text(
        "def _backend_build_info():\n"
        "    return {'prebuilt_operator_artifacts': "
        "{'target': 'host-isa-x86-64-v3'}}\n"
    )
    artifact_path = installed_package / "artifact.so"
    artifact_path.write_bytes(b"installed-artifact")
    manifest_path = installed_package / "manifest.json"
    manifest_path.write_text("{}")
    module_path = installed_package / "oam_jit_artifact.py"
    _write_fake_installed_artifact_module(
        module_path, artifact_path, manifest_path, "installed-id"
    )
    module_path.write_text("__spec__.loader = object()\n" + module_path.read_text())

    with pytest.raises(QualificationError, match="loader"):
        oam_release_qualification._probe_current_installed_artifact(
            _host_artifact_probe_request(tmp_path, installed_package, native_path)
        )


def test_current_installed_artifact_missing_catalog_is_qualification_error(
    tmp_path, monkeypatch
):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    monkeypatch.undo()
    monkeypatch.setattr(
        oam_release_qualification.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="RuntimeError: installed catalog is absent\n",
        ),
    )

    with pytest.raises(QualificationError, match="catalog is absent"):
        oam_release_qualification._probe_current_installed_artifact(cell)


def test_version_one_manifest_and_retained_cell_are_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["version"] = 1
    with pytest.raises(QualificationError, match="schema/version"):
        build_declared_cells(manifest)

    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    record["version"] = 1
    with pytest.raises(QualificationError, match="schema/version"):
        validate_cell_record(cell, record)


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


def test_ordinary_cuda_artifact_serializes_diagnostic_without_release_contract(
    tmp_path,
):
    artifact = OAMCudaArtifactResult(
        status="built",
        artifact_id="runtime-id",
        artifact_path=tmp_path / "cuda.cubin",
        manifest_path=tmp_path / "cuda.json",
        compiler="nvcc",
        target="sm90",
        diagnostics=("runtime compilation",),
        cache_key="cache-key",
    )

    evidence = serialize_diagnostic_artifact_evidence(
        backend="cuda",
        route="direct",
        artifact=artifact,
        artifact_sha256="b" * 64,
        generation_policy={"forward_live_output_budget": 1},
    )

    assert evidence == {
        "kind": "cuda-direct-diagnostic",
        "status": "built",
        "identity": "runtime-id",
        "origin": "runtime",
        "sha256": "b" * 64,
        "target": "sm90",
        "compiler": "nvcc",
        "schedule": {"forward_live_output_budget": 1},
        "diagnostics": ["runtime compilation"],
        "cache_key": "cache-key",
    }


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
        lambda direct, reference: reference["execution"]["bindings"][0].update(
            implementation_id="wrong-reference"
        ),
        lambda direct, reference: (
            direct["provenance"].update(
                jit_cache=f"/wrong/{direct['qualification_cell']['request_fingerprint']}"
            ),
            reference["provenance"].update(
                jit_cache=f"/wrong/{direct['qualification_cell']['request_fingerprint']}"
            ),
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


def test_copied_record_from_another_output_root_is_unqualified(tmp_path):
    manifest = _manifest(tmp_path)
    original = build_declared_cells(manifest, tmp_path / "first")[0]
    copied = build_declared_cells(manifest, tmp_path / "second")[0]
    assert original["request_fingerprint"] != copied["request_fingerprint"]
    assert original["direct_record_path"] != copied["direct_record_path"]
    assert original["reference_record_path"] != copied["reference_record_path"]
    copied_path = Path(copied["cell_record_path"])
    copied_path.parent.mkdir(parents=True)
    copied_path.write_text(json.dumps(_cell_record(original)))

    result = orchestrate(manifest, tmp_path / "second", execute=False)
    state = next(
        item for item in result["cells"] if item["cell_id"] == copied["cell_id"]
    )
    assert state["state"] == "unqualified"
    assert "invalid retained record" in state["reason"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda direct, reference: reference["timing_contract"]["excluded"].append(
            "invented_phase"
        ),
        lambda direct, reference: reference.update(python="/wrong/python"),
        lambda direct, reference: reference.update(package="/wrong/package.py"),
        lambda direct, reference: reference.update(native_extension="/wrong/native.so"),
        lambda direct, reference: reference.update(native_sha256="9" * 64),
        lambda direct, reference: reference.update(invented_evidence=True),
        lambda direct, reference: (
            direct.update(package="/wrong/package.py"),
            reference.update(package="/wrong/package.py"),
        ),
        lambda direct, reference: (
            direct["provenance"].update(invented=True),
            reference["provenance"].update(invented=True),
        ),
    ),
)
def test_composer_rejects_exact_timing_and_runtime_provenance_mismatch(
    tmp_path, mutation
):
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


def test_threads_help_describes_serial_blas_policy():
    expected = "Kokkos/OpenMP use N; BLAS/MKL/BLIS/NumExpr remain at one"
    assert expected in " ".join(make_cuda_timing_parser().format_help().split())
    assert expected in " ".join(make_host_profile_parser().format_help().split())


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
        ("graph_class", "small"),
    ),
)
def test_manifest_requires_exact_single_cuda_xl_capacity_boundary(
    tmp_path, field, value
):
    manifest = _manifest(tmp_path)
    manifest["capacity_boundaries"][0][field] = value

    with pytest.raises(QualificationError, match="capacity boundary"):
        build_declared_cells(manifest)


def test_manifest_rejects_multiple_capacity_boundaries_even_when_distinct(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["capacity_boundaries"].append(
        {
            "model_size": "L",
            "backend": "cuda",
            "precision": "float32",
            "graph_class": "production",
        }
    )

    with pytest.raises(QualificationError, match="exactly one capacity boundary"):
        build_declared_cells(manifest)


def test_driver_command_consumes_complete_graph_and_host_target(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["graphs"]["host-100"].update(skin=0.7, rattle=0.03)
    cell = build_declared_cells(manifest)[0]
    command = build_driver_command(manifest, cell, "direct")

    assert command[command.index("--skin") + 1] == "0.7"
    assert command[command.index("--rattle") + 1] == "0.03"
    assert command[command.index("--host-target") + 1] == "x86-64-v3"
    assert cell["host_target"] == "x86-64-v3"

    chain_manifest = _manifest(tmp_path)
    chain_manifest["graphs"]["small"] = {
        "structure": "chain",
        "atoms": 128,
        "seed": 11,
        "rattle": 0.02,
        "skin": 0.4,
    }
    chain_cell = next(
        cell
        for cell in build_declared_cells(chain_manifest)
        if cell["backend"] == "cuda" and cell["graph_class"] == "small"
    )
    chain_command = build_driver_command(chain_manifest, chain_cell, "direct")
    assert chain_command[chain_command.index("--atoms") + 1] == "128"
    assert "--repeat" not in chain_command


def test_raw_record_rejects_unsafe_jit_cache_mode(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    cache = Path(direct["provenance"]["jit_cache"])
    cache.chmod(0o755)
    direct["provenance"]["jit_cache_mode"] = 0o755
    reference["provenance"]["jit_cache_mode"] = 0o755
    with pytest.raises(QualificationError, match="unsafe"):
        build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)


def test_duplicate_evidence_fails_capacity_subgate(tmp_path):
    cells = build_declared_cells(_manifest(tmp_path))
    boundary = next(cell for cell in cells if cell["kind"] == "capacity-boundary")
    record = _cell_record(boundary)
    result = evaluate_qualification([boundary], [record, record])
    assert result["gates"]["capacity"]["passed"] is False


def test_cpu_cell_is_unqualified_when_runtime_concurrency_is_unobserved(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    record = _cell_record(cell)
    record["cpu"]["observed_kokkos_threads"] = None
    with pytest.raises(QualificationError, match="observed_kokkos_threads"):
        validate_cell_record(cell, record)


def test_duplicate_declared_cells_make_result_unqualified(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    result = evaluate_qualification([cell, cell], [_cell_record(cell)])
    assert result["status"] == "unqualified"
    assert any(
        "duplicate declared" in reason for reason in result["unqualified_reasons"]
    )


def test_request_fingerprint_binds_inputs_and_jit_namespace(tmp_path):
    manifest = _manifest(tmp_path)
    first = build_declared_cells(manifest)[0]
    _, first_cache = build_driver_environment(first)

    Path(manifest["models"]["S"]["path"]).write_text(
        json.dumps({"architecture": "oam", "r_cut": 5.0})
    )
    second = build_declared_cells(manifest)[0]
    _, second_cache = build_driver_environment(second)

    assert first["request_fingerprint"] != second["request_fingerprint"]
    assert first["request_fingerprint"] in str(first_cache)
    assert second["request_fingerprint"] in str(second_cache)

    graph_manifest = _manifest(tmp_path)
    original = build_declared_cells(graph_manifest)[0]
    graph_manifest["graphs"]["host-100"]["seed"] += 1
    changed_graph = build_declared_cells(graph_manifest)[0]
    assert original["request_fingerprint"] != changed_graph["request_fingerprint"]

    target_manifest = _manifest(tmp_path)
    original = build_declared_cells(target_manifest)[0]
    target_manifest["cuda_target"] = "sm100"
    changed_target = build_declared_cells(target_manifest)[0]
    assert original["request_fingerprint"] != changed_target["request_fingerprint"]

    policy_manifest = _manifest(tmp_path)
    original = build_declared_cells(policy_manifest)[0]
    policy_manifest["host_floating_point_policy"] = "strict"
    changed_policy = build_declared_cells(policy_manifest)[0]
    assert original["request_fingerprint"] != changed_policy["request_fingerprint"]

    baseline_manifest = _manifest(tmp_path)
    original = build_declared_cells(baseline_manifest)[0]
    baseline_manifest["direct_baselines_us_per_atom"][original["cell_id"]] = {
        "identity": f"approved/{original['cell_id']}",
        "sha256": "c" * 64,
        "us_per_atom": 7.5,
    }
    changed_baseline = build_declared_cells(baseline_manifest)[0]
    assert original["request_fingerprint"] != changed_baseline["request_fingerprint"]

    native_manifest = _manifest(tmp_path)
    original = build_declared_cells(native_manifest)[0]
    Path(native_manifest["provenance"]["native_paths"]["host"]).write_bytes(
        b"replacement-native"
    )
    changed_native = build_declared_cells(native_manifest)[0]
    assert original["request_fingerprint"] != changed_native["request_fingerprint"]


def test_duplicate_boundaries_and_records_are_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["capacity_boundaries"].append(dict(manifest["capacity_boundaries"][0]))
    with pytest.raises(QualificationError, match="exactly one capacity boundary"):
        build_declared_cells(manifest)

    cells = build_declared_cells(_manifest(tmp_path))
    record = _cell_record(cells[0])
    result = evaluate_qualification([cells[0]], [record, record])
    assert result["status"] == "unqualified"
    assert any("duplicate" in reason for reason in result["unqualified_reasons"])


def test_resume_rejects_model_changed_after_measurement(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "qualification"

    def run(cell, path):
        path.write_text(json.dumps(_cell_record(cell)))

    orchestrate(manifest, output, execute=True, runner=run)
    Path(manifest["models"]["S"]["path"]).write_text("changed")
    inspected = orchestrate(manifest, output, execute=False)
    stale = [cell for cell in inspected["cells"] if cell["model_size"] == "S"]
    assert stale
    assert all(cell["state"] == "unqualified" for cell in stale)
    assert all("retained" in cell["reason"] for cell in stale)


def test_graph_inclusive_host_profile_is_diagnostic_only(tmp_path):
    model = tmp_path / "oam.json"
    model.write_text(json.dumps({"architecture": "oam", "r_cut": 4.5}))
    record = profile(model, dry_run=True, reuse_prepared_graph=False)
    assert record["qualification_eligible"] is False
    assert "graph_construction" not in record["timing_contract"]["excluded"]


def test_cpu_scaling_keeps_blas_serial(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, _cpus: None)
    _configure_process(None, threads=2, cpus=[0, 1])
    assert {os.environ[name] for name in BLAS_THREAD_ENVIRONMENT} == {"1"}


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


def test_host_balanced_100_atom_graph_is_exact_and_reaches_driver(tmp_path):
    manifest = _manifest(tmp_path)
    host = [
        cell for cell in build_declared_cells(manifest) if cell["backend"] == "host"
    ]

    assert len(host) == 10
    assert {tuple(cell["graph_config"]["repeat"]) for cell in host} == {(2, 2, 5)}
    command = build_driver_command(manifest, host[0], "direct")
    assert command[command.index("--repeat") + 1] == "2,2,5"
    atoms = build_srtio3(repeat=(2, 2, 5), rattle=0.05, seed=7)
    assert len(atoms) == 100
    assert atoms.cell.lengths() == pytest.approx([7.81, 7.81, 19.525])


@pytest.mark.parametrize(
    "script",
    (
        "benchmarks/dispersion_kokkos_timing.py",
        "benchmarks/nequip_oam_kokkos_profile.py",
        "benchmarks/oam_cuda_r1_timing.py",
    ),
)
def test_shared_repeat_helpers_preserve_direct_script_entrypoints(script):
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_manifest_requires_exact_balanced_host_graph(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["graphs"]["host-100"]["repeat"] = [1, 4, 5]

    with pytest.raises(QualificationError, match=r"host-100.*\[2, 2, 5\]"):
        build_declared_cells(manifest)


def test_checked_in_manifest_example_validates_schema_v3():
    path = Path("benchmarks/oam_release_qualification_manifest.example.json")
    manifest = json.loads(path.read_text())

    validate_manifest(manifest)


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
            lambda record: record["graph"].update(effective_cutoff=4.9),
            "effective_cutoff",
        ),
        (
            lambda record: record["timing"].pop("direct_us_per_atom"),
            "direct_us_per_atom",
        ),
        (lambda record: record["execution"].update(fallback_count=1), "fallback_count"),
        (
            lambda record: record["reuse"].update(replacement_delta=1),
            "replacement_delta",
        ),
        (
            lambda record: record["baseline"].update(sha256="f" * 64),
            "baseline.sha256",
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


def test_capacity_kind_cannot_overwrite_cpu_scaling_rows(tmp_path):
    host = [
        cell
        for cell in build_declared_cells(_manifest(tmp_path))
        if cell["backend"] == "host" and cell["model_size"] == "S"
    ]
    boundary = dict(host[1])
    boundary.update(
        cell_id="capacity-boundary-s-host-float32-host-100-capacity-t4",
        kind="capacity-boundary",
        profile="capacity",
    )
    root = Path(boundary["output_root"]) / "cells"
    stem = root / boundary["cell_id"]
    boundary.update(
        cell_record_path=str(stem.with_suffix(".json")),
        direct_record_path=str(stem.with_suffix(".direct.json")),
        reference_record_path=str(stem.with_suffix(".reference.json")),
        direct_numerical_path=str(stem.with_suffix(".direct.numerics.npz")),
        reference_numerical_path=str(stem.with_suffix(".reference.numerics.npz")),
    )
    _rebind_request(boundary)
    records = [
        _cell_record(host[0], direct=8.0),
        _cell_record(host[1], direct=7.0),
        _cell_record(boundary, direct=20.0),
    ]

    result = evaluate_qualification([*host, boundary], records)

    assert result["gates"]["cpu_scaling"]["speedups"]["S/float32"] == pytest.approx(
        8.0 / 7.0
    )


@pytest.mark.parametrize("field", ("graph_sha256", "directed_edges", "cutoff", "skin"))
def test_cpu_scaling_requires_identical_realized_graph(tmp_path, field):
    cells = [
        cell
        for cell in build_declared_cells(_manifest(tmp_path))
        if cell["backend"] == "host" and cell["model_size"] == "S"
    ]
    records = [
        _cell_record(cell, direct=8.0 if cell["threads"] == 1 else 7.0)
        for cell in cells
    ]
    four_thread = records[1]
    if field == "graph_sha256":
        four_thread["provenance"][field] = "8" * 64
        for evidence in four_thread["evidence"].values():
            evidence[field] = "8" * 64
    else:
        value = four_thread["graph"][field]
        four_thread["graph"][field] = value + 1
        for evidence in four_thread["evidence"].values():
            evidence["graph"][field] = value + 1
            if field == "directed_edges":
                evidence["directed_edges"] = value + 1
            elif field == "cutoff":
                evidence["model_cutoff"] = value + 1
            elif field == "skin":
                evidence["skin"] = value + 1
        if field in {"cutoff", "skin"}:
            four_thread["graph"]["effective_cutoff"] += 1
            for evidence in four_thread["evidence"].values():
                evidence["graph"]["effective_cutoff"] += 1
                evidence["effective_cutoff"] += 1

    result = evaluate_qualification(cells, records)

    assert result["gates"]["cpu_scaling"]["passed"] is False
    assert any(
        "realized graph" in reason or "malformed qualification cell" in reason
        for reason in result["unqualified_reasons"]
    )


def test_host_graph_atom_count_is_derived_from_exact_repeat_axes(tmp_path):
    cell = next(
        cell
        for cell in build_declared_cells(_manifest(tmp_path))
        if cell["backend"] == "host"
    )
    record = _cell_record(cell)
    record["graph"]["atom_count"] = 99
    for evidence in record["evidence"].values():
        evidence["atoms"] = 99
        evidence["graph"]["atom_count"] = 99

    with pytest.raises(QualificationError, match="100 atoms"):
        validate_cell_record(cell, record)


def test_capacity_memory_and_boundary_gates_are_mechanical(tmp_path):
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
    capacity_record["memory"]["retained_bytes"] = 121
    capacity_record["evidence"]["direct"]["memory"]["retained_bytes"] = 121

    result = evaluate_qualification([speed, boundary], [speed_record, capacity_record])

    assert result["status"] == "failed"
    assert result["gates"]["capacity"]["passed"] is False


@pytest.mark.parametrize("stale_version", (None, 2))
def test_evaluator_rejects_extra_unique_or_stale_evidence_rows(tmp_path, stale_version):
    cells = build_declared_cells(_manifest(tmp_path))
    records = [
        _cell_record(cell, direct=7.0 if cell["threads"] == 4 else 8.0)
        for cell in cells
    ]
    extra = dict(records[0])
    extra["cell_id"] = "undeclared-extra-row"
    if stale_version is not None:
        extra["version"] = stale_version

    result = evaluate_qualification(cells, [*records, extra])

    assert result["status"] != "passed"
    assert result["gates"]["correctness"]["complete"] is False
    assert any(
        "unexpected evidence" in reason for reason in result["unqualified_reasons"]
    )


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


def test_host_driver_times_supported_stress_reduction():
    class Calculator:
        implemented_properties = ["energy", "forces", "stress"]

    assert _timed_properties(Calculator()) == ("energy", "forces", "stress")


def test_cell_builder_rejects_direct_reference_graph_mismatch(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]
    direct = _raw_record(cell, "direct")
    reference = _raw_record(cell, "reference")
    reference["graph_sha256"] = "9" * 64

    with pytest.raises(QualificationError, match="graph_sha256"):
        build_cell_record(cell, direct, reference, baseline_us_per_atom=8.0)


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


def test_paired_driver_commands_share_model_and_graph_arguments(tmp_path):
    manifest = _manifest(tmp_path)
    cell = build_declared_cells(manifest)[0]

    direct = build_driver_command(manifest, cell, "direct")
    reference = build_driver_command(manifest, cell, "reference")

    def value(command, option):
        return command[command.index(option) + 1]

    for option in ("--contract", "--structure", "--repeat", "--seed", "--threads"):
        assert value(direct, option) == value(reference, option)
    assert value(direct, "--routing") == "direct"
    assert value(reference, "--routing") == "reference"
    assert value(direct, "--execution-profile") == cell["profile"]


def test_driver_environment_has_a_private_per_cell_jit_cache(tmp_path):
    cell = build_declared_cells(_manifest(tmp_path))[0]

    environment, cache = build_driver_environment(cell)

    assert environment["SYMMETRIX_JIT_CACHE"] == str(cache)
    assert cell["cell_id"] in str(cache)


@pytest.mark.parametrize(
    ("profile", "plan_id", "schedule", "capability", "estimate"),
    (
        (
            "speed",
            "oam-direct-speed",
            "throughput",
            "generated_edge_tensor_product",
            120,
        ),
        ("capacity", "oam-direct-capacity", "capacity", "exact_capacity_layout", 80),
    ),
)
def test_driver_admits_only_the_requested_candidate(
    profile, plan_id, schedule, capability, estimate
):
    calls = []

    class Evaluator:
        family_artifact_readiness = {"artifact_id": "artifact-1"}
        execution_plan_report = {
            "candidates": [
                {
                    "id": "oam-direct-speed",
                    "estimated_bytes": 120,
                    "required_capability": "generated_edge_tensor_product",
                },
                {
                    "id": "oam-direct-capacity",
                    "estimated_bytes": 80,
                    "required_capability": "exact_capacity_layout",
                },
            ]
        }

        def _set_execution_plan_qualification(self, *args):
            calls.append(args)
            self.execution_plan_report["selected_id"] = args[0]

    admit_direct_candidate(Evaluator(), profile, "host", "float64")

    assert calls == [
        (
            plan_id,
            "artifact-1",
            "host",
            "float64",
            schedule,
            capability,
            "qualified",
            estimate,
            estimate,
            0,
        )
    ]


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
