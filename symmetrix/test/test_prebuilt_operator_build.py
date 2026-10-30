from __future__ import annotations

import importlib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

prebuilt = importlib.import_module("_symmetrix_build.prebuilt_operator_artifacts")
wheel_audit = importlib.import_module("_symmetrix_build.wheel_audit")
OAM_MANIFEST = ROOT / "symmetrix/source/symmetrix/foundation_oam_architectures.json"


def _cpu_plan(mode: str):
    return prebuilt.plan_foundation_operator_artifacts(
        manifest_path=(
            ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
        ),
        mode=mode,
        backend="host",
        target="x86-64-v3",
        source_content_sha256="a" * 64,
    )


def test_r0_mode_plans_six_unique_cpu_r0_shared_libraries():
    plan = _cpu_plan("r0")

    assert plan["mode"] == "r0"
    assert len(plan["artifacts"]) == 6
    assert {item["operator"] for item in plan["artifacts"]} == {"R0"}
    assert {item["channels"] for item in plan["artifacts"]} == {96, 128, 224}
    assert {item["precision"] for item in plan["artifacts"]} == {
        "float32",
        "float64",
    }
    assert {item["binary_format"] for item in plan["artifacts"]} == {"shared-library"}
    assert {item["target"] for item in plan["artifacts"]} == {"host-isa-x86-64-v3"}
    assert len({item["artifact_id"] for item in plan["artifacts"]}) == 6


def test_full_mode_plans_all_deduplicated_foundation_bundle_operators():
    plan = _cpu_plan("full")

    assert plan["mode"] == "full"
    assert len(plan["artifacts"]) == 32
    assert {
        operator: sum(item["operator"] == operator for item in plan["artifacts"])
        for operator in ("R0", "R1", "M0", "M1")
    } == {"R0": 6, "R1": 10, "M0": 10, "M1": 6}
    assert len({item["artifact_id"] for item in plan["artifacts"]}) == 32
    assert {item["precision"] for item in plan["artifacts"]} == {
        "float32",
        "float64",
    }
    assert {item["binary_format"] for item in plan["artifacts"]} == {"shared-library"}
    assert {item["target"] for item in plan["artifacts"]} == {"host-isa-x86-64-v3"}
    for bundle_id in (
        "off23-l0",
        "off23-l2",
        "standard-l0",
        "standard-l1",
        "standard-l2",
    ):
        assert sum(bundle_id in item["bundle_ids"] for item in plan["artifacts"]) == 8
    assert {item["build_lane"] for item in plan["artifacts"]} == {0, 1}


def test_off_mode_plans_no_operator_artifacts():
    plan = _cpu_plan("off")

    assert plan == {"mode": "off", "artifacts": []}


def test_known_oam_plan_uses_explicit_architectures_for_both_precisions():
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=(
            ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
        ),
        mode="off",
        backend="host",
        target="x86-64-v3",
        source_content_sha256="a" * 64,
        oam_manifest_path=OAM_MANIFEST,
        oam_mode="known",
        oam_floating_point_policy="strict",
    )
    fast = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=(
            ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
        ),
        mode="off",
        backend="host",
        target="x86-64-v3",
        source_content_sha256="a" * 64,
        oam_manifest_path=OAM_MANIFEST,
        oam_mode="known",
        oam_floating_point_policy="fast",
    )

    assert len(plan["artifacts"]) == 8
    assert {item["operator"] for item in plan["artifacts"]} == {"OAM"}
    assert {item["architecture_id"] for item in plan["artifacts"]} == {
        "oam-s-v0.1",
        "oam-m-v0.1",
        "oam-l-v0.1",
        "oam-xl-v0.1",
    }
    assert {item["precision"] for item in plan["artifacts"]} == {
        "float32",
        "float64",
    }
    assert {item["floating_point_policy"] for item in plan["artifacts"]} == {"strict"}
    assert {item["floating_point_policy"] for item in fast["artifacts"]} == {"fast"}
    assert {item["artifact_id"] for item in plan["artifacts"]}.isdisjoint(
        {item["artifact_id"] for item in fast["artifacts"]}
    )


def test_known_oam_render_publishes_architecture_and_loader_manifest(tmp_path):
    from symmetrix.oam_architecture import (
        oam_architecture_fingerprint,
        oam_tensor_product_architecture_fingerprint,
    )

    source = tmp_path / "oam.cpp"
    metadata_path = tmp_path / "oam.json"
    manifest_path = tmp_path / "manifest.json"
    prebuilt.render_foundation_oam_artifact(
        oam_manifest_path=OAM_MANIFEST,
        architecture_id="oam-s-v0.1",
        backend="host",
        target="x86-64-v3",
        precision="float64",
        source_content_sha256="a" * 64,
        compiler_id="GNU",
        compiler_version="15",
        source_path=source,
        metadata_path=metadata_path,
        companion_manifest_path=manifest_path,
        floating_point_policy="strict",
    )

    metadata = json.loads(metadata_path.read_text())
    identity = metadata["generated_identity"]
    assert identity["operator"] == "OAM"
    assert identity["target"] == "host-isa-x86-64-v3"
    assert metadata["host_target"] == "host-isa-x86-64-v3"
    assert metadata["floating_point_policy"] == "strict"
    assert identity["floating_point_policy"] == "strict"
    assert metadata["build_provenance"]["floating_point_policy"] == "strict"
    assert identity["architecture"]["layer_count"] == 2
    assert identity["architecture_fingerprint"] == (
        oam_tensor_product_architecture_fingerprint(identity["architecture"])
    )
    assert identity["architecture_fingerprint"] != oam_architecture_fingerprint(
        identity["architecture"]
    )
    assert metadata["contract_fingerprint"].startswith("sha256:")
    assert identity["architecture_fingerprint"] == metadata["contract_fingerprint"]
    loader = json.loads(manifest_path.read_text())
    assert (
        loader["key_inputs"]["build"]["metadata"]["artifact_id"]
        == metadata["artifact_id"]
    )
    rendered = source.read_text()
    assert "NequIP-OAM" not in rendered
    assert "host-isa-x86-64-v3" in rendered


@pytest.mark.parametrize(
    ("backend", "target", "binary_format", "extension"),
    (
        ("cuda", "sm_120", "cubin", "cu"),
        ("hip", "gfx1151:sramecc+", "code-object", "hip"),
    ),
)
def test_accelerator_plan_preserves_target_and_native_binary_format(
    backend, target, binary_format, extension
):
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=(
            ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
        ),
        mode="r0",
        backend=backend,
        target=target,
        source_content_sha256="d" * 64,
    )

    assert len(plan["artifacts"]) == 6
    assert {item["target"] for item in plan["artifacts"]} == {target}
    assert {item["binary_format"] for item in plan["artifacts"]} == {binary_format}
    assert {item["source_extension"] for item in plan["artifacts"]} == {extension}


@pytest.mark.parametrize(
    ("backend", "target", "binary_format", "suffix"),
    (
        ("cuda", "sm_120", "cubin", ".cubin"),
        ("hip", "gfx1100", "code-object", ".hsaco"),
    ),
)
def test_production_device_r0_plan_renders_and_enters_installed_catalog(
    tmp_path, backend, target, binary_format, suffix
):
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=manifest,
        mode="r0",
        backend=backend,
        target=target,
        source_content_sha256="d" * 64,
    )
    selected = plan["artifacts"][0]
    source = tmp_path / f"operator.{selected['source_extension']}"
    metadata_path = tmp_path / "operator.json"
    prebuilt.render_foundation_r0_artifact(
        manifest_path=manifest,
        artifact_input_id=selected["artifact_input_id"],
        backend=backend,
        target=target,
        precision=selected["precision"],
        source_content_sha256="d" * 64,
        compiler_id="test-compiler",
        compiler_version="1",
        source_path=source,
        metadata_path=metadata_path,
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_id"] == selected["artifact_id"]
    assert metadata["schedule_revision"] == 6
    assert metadata["abi_version"] == 5
    binary = tmp_path / f"operator{suffix}"
    binary.write_bytes(b"device-r0")
    started = tmp_path / "operator.started-ns"
    started.write_text(f"{time.time_ns()}\n")
    catalog_path = tmp_path / "catalog.json"
    prebuilt.write_installed_operator_catalog(
        artifacts=[(metadata_path, binary, f"r0/{binary.name}", started)],
        catalog_path=catalog_path,
        header_path=tmp_path / "build_info.hpp",
        mode="r0",
        backend=backend,
        target=target,
        source_content_sha256="d" * 64,
        native_source_content_sha256="e" * 64,
        compiler_id="test-compiler",
        compiler_version="1",
    )
    catalog = json.loads(catalog_path.read_text())
    assert catalog["artifacts"][0]["binary_format"] == binary_format
    assert catalog["artifacts"][0]["abi"] == {
        "tag": "symmetrix.jit.operator-module/1",
        "version": 5,
    }


@pytest.mark.parametrize("operator", ("M0", "M1"))
@pytest.mark.parametrize(
    ("backend", "target", "suffix"),
    (("cuda", "sm_120", ".cubin"), ("hip", "gfx1100", ".hsaco")),
)
def test_full_device_product_plan_renders_and_requires_abi_v3(
    tmp_path, operator, backend, target, suffix
):
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=manifest,
        mode="full",
        backend=backend,
        target=target,
        source_content_sha256="d" * 64,
    )
    selected = next(item for item in plan["artifacts"] if item["operator"] == operator)
    source = tmp_path / f"operator.{selected['source_extension']}"
    metadata_path = tmp_path / "operator.json"
    prebuilt.render_foundation_operator_artifact(
        manifest_path=manifest,
        operator=operator,
        artifact_id=selected["artifact_id"],
        backend=backend,
        target=target,
        precision=selected["precision"],
        source_content_sha256="d" * 64,
        compiler_id="test-compiler",
        compiler_version="1",
        source_path=source,
        metadata_path=metadata_path,
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["generated_identity"]["artifact_id"] == selected["artifact_id"]
    assert metadata["abi_version"] == 3

    binary = tmp_path / f"operator{suffix}"
    binary.write_bytes(b"device-product")
    started = tmp_path / "operator.started-ns"
    started.write_text(f"{time.time_ns()}\n")
    catalog_path = tmp_path / "catalog.json"
    catalog_arguments = {
        "artifacts": [
            (metadata_path, binary, f"{operator.lower()}/{binary.name}", started)
        ],
        "catalog_path": catalog_path,
        "header_path": tmp_path / "build_info.hpp",
        "mode": "full",
        "backend": backend,
        "target": target,
        "source_content_sha256": "d" * 64,
        "native_source_content_sha256": "e" * 64,
        "compiler_id": "test-compiler",
        "compiler_version": "1",
    }
    prebuilt.write_installed_operator_catalog(**catalog_arguments)
    assert json.loads(catalog_path.read_text())["artifacts"][0]["abi"] == {
        "tag": "symmetrix.jit.operator-module/1",
        "version": 3,
    }

    metadata["abi_version"] = 2
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="version 3"):
        prebuilt.write_installed_operator_catalog(**catalog_arguments)


@pytest.mark.parametrize("policy", ("cuda-fast-math", "strict"))
def test_cuda_r0_policy_survives_plan_render_compile_and_catalog(tmp_path, policy):
    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("nvcc is unavailable")
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    plan = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=manifest,
        mode="r0",
        backend="cuda",
        target="sm_80",
        source_content_sha256="d" * 64,
        floating_point_policy=policy,
    )
    selected = plan["artifacts"][0]
    source = tmp_path / "operator.cu"
    metadata_path = tmp_path / "operator.json"
    prebuilt.render_foundation_r0_artifact(
        manifest_path=manifest,
        artifact_input_id=selected["artifact_input_id"],
        backend="cuda",
        target="sm_80",
        precision=selected["precision"],
        source_content_sha256="d" * 64,
        compiler_id="NVIDIA",
        compiler_version="13.3",
        source_path=source,
        metadata_path=metadata_path,
        floating_point_policy=policy,
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["artifact_id"] == selected["artifact_id"]
    assert metadata["floating_point_policy"] == policy
    binary = tmp_path / "operator.cubin"
    command = [compiler, "-std=c++20", "-O3"]
    if policy == "cuda-fast-math":
        command.append("--use_fast_math")
    command.extend(
        [
            "--cubin",
            "--generate-code=arch=compute_80,code=sm_80",
            str(source),
            "-o",
            str(binary),
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    started = tmp_path / "operator.started-ns"
    started.write_text(f"{time.time_ns()}\n")
    catalog_path = tmp_path / "catalog.json"
    prebuilt.write_installed_operator_catalog(
        artifacts=[(metadata_path, binary, "r0/operator.cubin", started)],
        catalog_path=catalog_path,
        header_path=tmp_path / "build_info.hpp",
        mode="r0",
        backend="cuda",
        target="sm_80",
        source_content_sha256="d" * 64,
        native_source_content_sha256="e" * 64,
        compiler_id="NVIDIA",
        compiler_version="13.3",
    )
    catalog = json.loads(catalog_path.read_text())
    assert (
        catalog["artifacts"][0]["build_provenance"]["floating_point_policy"] == policy
    )


@pytest.mark.parametrize(
    ("backend", "target"), (("cuda", "sm_120"), ("hip", "gfx1100"))
)
def test_device_r0_catalog_rejects_previous_abi(tmp_path, backend, target):
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    selected = prebuilt.plan_foundation_operator_artifacts(
        manifest_path=manifest,
        mode="r0",
        backend=backend,
        target=target,
        source_content_sha256="d" * 64,
    )["artifacts"][0]
    metadata_path = tmp_path / "operator.json"
    prebuilt.render_foundation_r0_artifact(
        manifest_path=manifest,
        artifact_input_id=selected["artifact_input_id"],
        backend=backend,
        target=target,
        precision=selected["precision"],
        source_content_sha256="d" * 64,
        compiler_id="test-compiler",
        compiler_version="1",
        source_path=tmp_path / f"operator.{selected['source_extension']}",
        metadata_path=metadata_path,
    )
    metadata = json.loads(metadata_path.read_text())
    metadata["abi_version"] = 4
    metadata_path.write_text(json.dumps(metadata))
    binary = tmp_path / "operator.bin"
    binary.write_bytes(b"device-r0")
    started = tmp_path / "operator.started-ns"
    started.write_text(f"{time.time_ns()}\n")

    with pytest.raises(ValueError, match="version 5"):
        prebuilt.write_installed_operator_catalog(
            artifacts=[(metadata_path, binary, f"r0/{binary.name}", started)],
            catalog_path=tmp_path / "catalog.json",
            header_path=tmp_path / "build_info.hpp",
            mode="r0",
            backend=backend,
            target=target,
            source_content_sha256="d" * 64,
            native_source_content_sha256="e" * 64,
            compiler_id="test-compiler",
            compiler_version="1",
        )


def _write_fixture_project(
    source: Path,
    manifest: Path,
    mode: str,
    *,
    oam_mode: str = "off",
    extra: str = "",
) -> None:
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(prebuilt_operator_fixture LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter REQUIRED)
set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS {mode} CACHE STRING "")
set(SYMMETRIX_PREBUILT_OAM_ARTIFACTS {oam_mode} CACHE STRING "")
set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND host)
set(SYMMETRIX_PREBUILT_OPERATOR_TARGET x86-64-v3)
set(SYMMETRIX_PREBUILT_OPERATOR_MANIFEST {manifest.as_posix()})
set(SYMMETRIX_PREBUILT_OAM_MANIFEST {OAM_MANIFEST.as_posix()})
set(SYMMETRIX_HOST_ARCH x86-64-v3)
set(SYMMETRIX_PYTHON_INSTALL_PACKAGE fixture_package)
{extra}
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )


def _write_cuda_fixture_project(source: Path, manifest: Path, mode: str) -> None:
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(prebuilt_operator_cuda_fixture LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter REQUIRED)
set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS {mode} CACHE STRING "")
set(SYMMETRIX_PREBUILT_OAM_ARTIFACTS known CACHE STRING "")
set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND cuda)
set(SYMMETRIX_PREBUILT_OPERATOR_TARGET sm_120)
set(SYMMETRIX_PREBUILT_OPERATOR_MANIFEST {manifest.as_posix()})
set(SYMMETRIX_PREBUILT_OAM_MANIFEST {OAM_MANIFEST.as_posix()})
set(SYMMETRIX_PYTHON_INSTALL_PACKAGE fixture_package)
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )


def _run(*command: os.PathLike[str] | str, cwd: Path | None = None):
    return subprocess.run(
        [os.fspath(value) for value in command],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _artifact_mtimes(build: Path) -> dict[str, int]:
    artifact_dir = build / "generated-operator-artifacts"
    paths = [
        *artifact_dir.glob("*.cpp"),
        *artifact_dir.glob("*.json"),
        *artifact_dir.glob("*.so"),
    ]
    return {path.name: path.stat().st_mtime_ns for path in sorted(paths)}


def test_subdirectory_build_defaults_prebuilt_artifacts_off(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    child = source / "libsymmetrix-child"
    child.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.27)
project(lammps_style_parent LANGUAGES NONE)
add_subdirectory(libsymmetrix-child)
"""
    )
    (child / "CMakeLists.txt").write_text(
        f"""
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )

    configured = _run(cmake, "-S", source, "-B", tmp_path / "build")

    assert configured.returncode == 0, configured.stderr
    assert "prebuilt operator artifacts disabled" in configured.stdout


def _write_host_target_fixture(
    source: Path, *, target: str, host_arch: str, extra: str = ""
) -> None:
    source.mkdir()
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(prebuilt_host_target_fixture LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter REQUIRED)
set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS r0)
set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND host)
set(SYMMETRIX_PREBUILT_OPERATOR_TARGET {target})
set(SYMMETRIX_PREBUILT_OPERATOR_MANIFEST {manifest.as_posix()})
set(SYMMETRIX_HOST_ARCH {host_arch})
set(SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 {"a" * 64})
set(SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256 {"b" * 64})
{extra}
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )


def test_host_prebuilt_rejects_declared_target_that_differs_from_compile_arch(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    _write_host_target_fixture(source, target="x86-64-v3", host_arch="native")

    configured = _run(cmake, "-S", source, "-B", tmp_path / "build")

    assert configured.returncode != 0
    assert "prebuilt operator target does not match effective host compile" in (
        configured.stderr
    )
    assert "SYMMETRIX_HOST_ARCH 'native'" in configured.stderr


def test_host_prebuilt_target_binds_ambient_isa_flags(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    _write_host_target_fixture(source, target="x86-64-v3", host_arch="x86-64-v3")
    build = tmp_path / "build"

    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_CXX_FLAGS=-mavx512f",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    )

    assert configured.returncode == 0, configured.stderr
    plan = json.loads((build / "generated-operator-artifacts/plan.json").read_text())
    targets = {artifact["target"] for artifact in plan["artifacts"]}
    assert len(targets) == 1
    target = targets.pop()
    assert target.startswith("host-isa-x86-64-v3-requires-avx512f-flags-sha256-")
    commands = json.loads((build / "compile_commands.json").read_text())
    generated_commands = [
        entry["command"]
        for entry in commands
        if "generated-operator-artifacts" in entry["file"]
    ]
    assert generated_commands
    assert all("-mavx512f" in command for command in generated_commands)
    assert all("-march=x86-64-v3" in command for command in generated_commands)


@pytest.mark.parametrize("enabled", (False, True), ids=("strict", "fast"))
def test_known_oam_host_prebuilts_bind_fast_math_policy(tmp_path, enabled):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    _write_fixture_project(
        source,
        manifest,
        "off",
        oam_mode="known",
        extra=f"set(SYMMETRIX_FAST_MATH {'ON' if enabled else 'OFF'})",
    )
    build = tmp_path / "build"

    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={'b' * 64}",
        f"-DSYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256={'e' * 64}",
    )

    assert configured.returncode == 0, configured.stderr
    commands = json.loads((build / "compile_commands.json").read_text())
    generated_commands = [
        entry["command"]
        for entry in commands
        if "generated-operator-artifacts" in entry["file"]
    ]
    assert len(generated_commands) == 8
    assert all(("-ffast-math" in command) is enabled for command in generated_commands)
    plan = json.loads((build / "generated-operator-artifacts/plan.json").read_text())
    expected = "fast" if enabled else "strict"
    assert {item["floating_point_policy"] for item in plan["artifacts"]} == {expected}
    compiled = _run(cmake, "--build", build, "--parallel", "2")
    assert compiled.returncode == 0, compiled.stderr
    catalog = json.loads(
        (build / "generated-operator-artifacts/catalog.json").read_text()
    )
    assert len(catalog["artifacts"]) == 8
    assert {
        entry["identity"]["floating_point_policy"] for entry in catalog["artifacts"]
    } == {expected}
    assert {
        entry["build_provenance"]["floating_point_policy"]
        for entry in catalog["artifacts"]
    } == {expected}
    assert {entry["identity"]["artifact_id"] for entry in catalog["artifacts"]} == {
        item["artifact_id"] for item in plan["artifacts"]
    }


def test_host_fast_math_rejects_an_unsupported_compiler_identity(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    _write_fixture_project(
        source,
        manifest,
        "off",
        oam_mode="known",
        extra="set(SYMMETRIX_FAST_MATH ON)\nset(CMAKE_CXX_COMPILER_ID Unsupported)",
    )

    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        tmp_path / "build",
        f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={'b' * 64}",
        f"-DSYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256={'e' * 64}",
    )

    assert configured.returncode != 0
    assert "host prebuilts require a supported GNU" in configured.stderr
    assert "detected 'Unsupported'" in configured.stderr


def test_host_prebuilt_rejects_config_dependent_compile_flags(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    _write_host_target_fixture(
        source,
        target="x86-64-v3",
        host_arch="x86-64-v3",
        extra='set(CMAKE_CONFIGURATION_TYPES "Debug;Release")',
    )

    configured = _run(cmake, "-S", source, "-B", tmp_path / "build")

    assert configured.returncode != 0
    assert "single-config generator" in configured.stderr


def test_host_prebuilt_target_binds_directory_clang_feature_flags(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    _write_host_target_fixture(
        source,
        target="x86-64-v3",
        host_arch="x86-64-v3",
        extra=(
            "add_compile_options("
            "\"$<$<COMPILE_LANGUAGE:CXX>:SHELL:-Xclang '-target-feature' "
            "-Xclang '+avx512f'>\")"
        ),
    )
    build = tmp_path / "build"

    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    )

    assert configured.returncode == 0, configured.stderr
    plan = json.loads((build / "generated-operator-artifacts/plan.json").read_text())
    assert {artifact["target"] for artifact in plan["artifacts"]}.pop().startswith(
        "host-isa-x86-64-v3-requires-avx512f-flags-sha256-"
    )
    commands = json.loads((build / "compile_commands.json").read_text())
    generated_argv = [
        shlex.split(entry["command"])
        for entry in commands
        if "generated-operator-artifacts" in entry["file"]
    ]
    assert generated_argv
    clang_feature = ["-Xclang", "-target-feature", "-Xclang", "+avx512f"]
    assert all(
        any(
            argv[index : index + len(clang_feature)] == clang_feature
            for index in range(len(argv) - len(clang_feature) + 1)
        )
        for argv in generated_argv
    )


def test_host_prebuilt_target_expands_directory_shell_m_flag(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    _write_host_target_fixture(
        source,
        target="x86-64-v3",
        host_arch="x86-64-v3",
        extra='add_compile_options("SHELL:-mavx512f")',
    )
    build = tmp_path / "build"

    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    )

    assert configured.returncode == 0, configured.stderr
    plan = json.loads((build / "generated-operator-artifacts/plan.json").read_text())
    assert {artifact["target"] for artifact in plan["artifacts"]}.pop().startswith(
        "host-isa-x86-64-v3-requires-avx512f-flags-sha256-"
    )
    commands = json.loads((build / "compile_commands.json").read_text())
    generated_argv = [
        shlex.split(entry["command"])
        for entry in commands
        if "generated-operator-artifacts" in entry["file"]
    ]
    assert generated_argv
    assert all("-mavx512f" in argv for argv in generated_argv)
    assert all("SHELL:-mavx512f" not in argv for argv in generated_argv)


def test_host_prebuilt_rejects_unknown_directory_shell_isa_flag(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    _write_host_target_fixture(
        source,
        target="x86-64-v3",
        host_arch="x86-64-v3",
        extra='add_compile_options("SHELL:-mdefinitely-unknown-isa")',
    )

    configured = _run(cmake, "-S", source, "-B", tmp_path / "build")

    assert configured.returncode != 0
    assert "unknown or unverifiable required ISA" in configured.stderr


def test_host_prebuilt_rejects_unresolved_isa_genex_in_shell_group(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    unresolved = 'add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:CXX>:-mavx512f>")'
    _write_host_target_fixture(
        source,
        target="x86-64-v3",
        host_arch="x86-64-v3",
        extra=unresolved,
    )

    configured = _run(cmake, "-S", source, "-B", tmp_path / "build")

    assert configured.returncode != 0
    assert "Cannot verify ISA-affecting host compile option" in configured.stderr


def test_reused_parent_build_retains_compiled_catalog_provenance(tmp_path):
    compiler = shutil.which("c++")
    cmake = shutil.which("cmake")
    if compiler is None or cmake is None:
        pytest.skip("CMake and a C++ compiler are required")
    source = tmp_path / "source"
    child = source / "child"
    child.mkdir(parents=True)
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    (child / "CMakeLists.txt").write_text(
        f"""
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )
    (source / "probe.cpp").write_text(
        """
#include "prebuilt_operator_build_info.hpp"
#include <iostream>
int main() { std::cout << symmetrix::build::prebuilt_operator_mode; }
"""
    )
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(reused_prebuilt_parent LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter REQUIRED)
set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS r0 CACHE STRING "")
set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND host)
set(SYMMETRIX_PREBUILT_OPERATOR_TARGET x86-64-v3)
set(SYMMETRIX_PREBUILT_OPERATOR_MANIFEST {manifest.as_posix()})
set(SYMMETRIX_HOST_ARCH x86-64-v3)
set(SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 {"a" * 64})
set(SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256 {"b" * 64})
add_subdirectory(child)
if(NOT SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER)
  set(SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER
      ${{CMAKE_BINARY_DIR}}/generated/prebuilt_operator_build_info.hpp)
  file(WRITE ${{SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER}}
      "#pragma once\nnamespace symmetrix::build {{ inline constexpr const char* prebuilt_operator_mode = \"off\"; }}\n")
endif()
add_executable(provenance_probe probe.cpp
    ${{SYMMETRIX_PREBUILT_OPERATOR_CATALOG_HEADER}})
add_dependencies(provenance_probe symmetrix_prebuilt_operator_catalog)
target_include_directories(provenance_probe PRIVATE ${{CMAKE_BINARY_DIR}}/generated)
"""
    )
    build = tmp_path / "build"
    configure = (
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler}",
    )

    for _ in range(2):
        configured = _run(*configure)
        assert configured.returncode == 0, configured.stderr
        compiled = _run(cmake, "--build", build, "--parallel", "2")
        assert compiled.returncode == 0, compiled.stderr
        probed = _run(build / "provenance_probe")
        assert probed.returncode == 0, probed.stderr
        assert probed.stdout == "r0"


def test_full_cpu_catalog_builds_with_all_known_oam_artifacts(monkeypatch, tmp_path):
    from benchmarks import oam_release_qualification

    compiler = shutil.which("c++")
    cmake = shutil.which("cmake")
    if compiler is None or cmake is None:
        pytest.skip("CMake and a C++ compiler are required")
    source = tmp_path / "source"
    build = tmp_path / "build"
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    source_fingerprint = oam_release_qualification._current_source_provenance(
        str(ROOT)
    )["source_fingerprint"]
    mode = "full"
    _write_fixture_project(source, manifest, mode, oam_mode="known")

    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={source_fingerprint}",
        f"-DSYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256={'e' * 64}",
    )
    assert configured.returncode == 0, configured.stderr
    compiled = _run(cmake, "--build", build, "--parallel", "2")
    assert compiled.returncode == 0, compiled.stderr
    install = tmp_path / "install"
    installed = _run(cmake, "--install", build, "--prefix", install)
    assert installed.returncode == 0, installed.stderr

    root = install / "fixture_package/_operator_artifacts"
    catalog = json.loads((root / "catalog.json").read_text())
    expected_standard = 6 if mode == "r0" else 32
    assert len(catalog["artifacts"]) == expected_standard + 8
    assert sum(entry["operator"] == "OAM" for entry in catalog["artifacts"]) == 8
    assert sum(entry["operator"] == "R0" for entry in catalog["artifacts"]) == 6
    oam_entries = [
        entry for entry in catalog["artifacts"] if entry["operator"] == "OAM"
    ]
    assert {entry["identity"]["precision"] for entry in oam_entries} == {
        "float32",
        "float64",
    }
    assert (
        len({entry["identity"]["architecture_fingerprint"] for entry in oam_entries})
        == 4
    )
    assert {
        entry["identity"]["architecture"]["layer_count"] for entry in oam_entries
    } == {2, 4, 6}
    assert len(list((root / "oam").glob("*/plugin.so"))) == 8
    assert len(list((root / "oam").glob("*/manifest.json"))) == 8
    for entry in oam_entries:
        assert entry["identity"]["architecture"]["layers"]
        assert entry["auxiliary_files"][0]["role"] == "manifest"
        for record in [entry, entry["auxiliary_files"][0]]:
            path = root / record["installed_relative_path"]
            assert path.is_file()
            assert prebuilt.sha256_file(path) == record["checksum"]
    installed_module = importlib.import_module("symmetrix.oam_installed_artifact")
    compiler_provenance = oam_entries[0]["build_provenance"]
    installed_files = [root / "catalog.json"]
    for entry in catalog["artifacts"]:
        installed_files.append(root / entry["installed_relative_path"])
        installed_files.extend(
            root / auxiliary["installed_relative_path"]
            for auxiliary in entry.get("auxiliary_files", ())
        )
    build_info = {
        "mode": mode,
        "backend": "host",
        "target": "host-isa-x86-64-v3",
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(
            (root / "catalog.json").read_bytes()
        ).hexdigest(),
        "artifact_count": len(catalog["artifacts"]),
        "installed_bytes": sum(path.stat().st_size for path in installed_files),
        "source_content_sha256": source_fingerprint,
        "native_source_content_sha256": "e" * 64,
        "compiler_id": compiler_provenance["compiler_id"],
        "compiler_version": compiler_provenance["compiler_version"],
        "build_duration_seconds": 0.1,
    }
    native = SimpleNamespace(
        __package__="fixture_package",
        _backend_build_info=lambda: {"prebuilt_operator_artifacts": build_info},
    )
    monkeypatch.setattr(
        installed_module.resources,
        "files",
        lambda package_name: install / "fixture_package",
    )
    for entry in oam_entries:
        identity = entry["identity"]
        resolved = installed_module.resolve_installed_oam_artifact(
            architecture=identity["architecture"],
            precision=identity["precision"],
            backend="host",
            host_target=identity["target"],
            floating_point_policy=identity["floating_point_policy"],
            artifact_id=identity["artifact_id"],
            required_capabilities=entry["capabilities"],
            native_module=native,
        )
        assert resolved.available
        assert resolved.artifact_path == root / entry["installed_relative_path"]

    import test_oam_release_qualification as qualification_fixture
    from benchmarks import oam_cuda_r1_timing
    from benchmarks.oam_release_qualification import (
        build_declared_cells,
        build_driver_command,
    )
    from symmetrix.oam_jit_artifact import OAMHostArtifactResult

    qualification_root = tmp_path / "qualification-fixture"
    qualification_root.mkdir()
    qualification_manifest = qualification_fixture._manifest(qualification_root)
    qualification_manifest["models"]["S"]["path"] = str(
        ROOT / "symmetrix/test/data/nequip/oam_s_v2_minimal.json"
    )
    qualification_manifest["direct_baselines_us_per_atom"] = {}
    for cell in build_declared_cells(qualification_manifest):
        qualification_manifest["direct_baselines_us_per_atom"][cell["cell_id"]] = {
            "identity": f"approved/{cell['cell_id']}",
            "sha256": "b" * 64,
            "us_per_atom": 8.0,
        }
    request = next(
        cell
        for cell in build_declared_cells(qualification_manifest)
        if cell["model_size"] == "S"
        and cell["backend"] == "host"
        and cell["precision"] == "float32"
    )
    request["host_floating_point_policy"] = "strict"
    qualification_fixture._rebind_request(request)
    command = build_driver_command({}, request, "direct")
    cache = Path(request["jit_cache_path"])
    cache.mkdir(parents=True, mode=0o700)
    cache.chmod(0o700)
    native_info = qualification_fixture._qualification_native(
        request
    )._backend_build_info()
    native_info["native_source_content_sha256"] = "e" * 64
    native_info["prebuilt_operator_artifacts"] = build_info
    maintained_native = SimpleNamespace(
        __package__="fixture_package",
        _backend_build_info=lambda: native_info,
    )
    monkeypatch.setattr(
        oam_release_qualification,
        "_current_native_source_fingerprint",
        lambda source_root: "e" * 64,
    )
    maintained_entry = min(
        (entry for entry in oam_entries if entry["identity"]["precision"] == "float32"),
        key=lambda entry: (
            entry["identity"]["architecture"]["layer_count"],
            entry["identity"]["architecture"]["layers"][0]["source_dimension"],
        ),
    )

    def prepare_maintained_artifact(path, precision, **arguments):
        identity = maintained_entry["identity"]
        resolved = installed_module.resolve_installed_oam_artifact(
            architecture=identity["architecture"],
            precision=precision,
            backend="host",
            host_target=identity["target"],
            floating_point_policy=identity["floating_point_policy"],
            artifact_id=identity["artifact_id"],
            required_capabilities=maintained_entry["capabilities"],
            native_module=arguments["native_module"],
        )
        assert arguments["installed_prebuilt_required"] is True
        assert resolved.available
        companion = json.loads(resolved.manifest_path.read_text(encoding="utf-8"))
        metadata = companion["key_inputs"]["build"]["metadata"]
        return OAMHostArtifactResult(
            status="installed",
            precision=precision,
            model_path=Path(path),
            artifact_path=resolved.artifact_path,
            manifest_path=resolved.manifest_path,
            cache_key="",
            artifact_id=identity["artifact_id"],
            semantic_fingerprint=identity["semantic_fingerprint"],
            structure_fingerprint=identity["structure_fingerprint"],
            generation_fingerprint=identity["generation_fingerprint"],
            metadata=metadata,
            diagnostics=resolved.diagnostics,
            cache_hit=False,
            compiler_command=(),
            artifact_origin="prebuilt",
            active_host_target=identity["target"],
            expected_host_target=identity["target"],
            expected_artifact_origin="prebuilt",
            floating_point_policy=identity["floating_point_policy"],
            expected_artifact_id=identity["artifact_id"],
        )

    qualification = oam_cuda_r1_timing._qualification_execution_entry(
        request,
        artifact_preparer=prepare_maintained_artifact,
        contract=Path(request["model_path"]),
        native=maintained_native,
        target=request["host_target"],
        compiler=None,
        threads_per_block=None,
        generation_policy=None,
        **qualification_fixture._qualification_entry_live_arguments(
            request, command, maintained_native, cache
        ),
    )
    evidence = qualification["artifact_evidence"]
    assert evidence["expected_target"] == "x86-64-v3"
    assert evidence["native_build_target"] == "host-isa-x86-64-v3"
    assert evidence["active_host_target"] == "host-isa-x86-64-v3"
    assert qualification["artifact"].artifact_path.is_file()


def test_full_cuda_catalog_builds_installs_and_resolves_all_known_oam(
    monkeypatch, tmp_path
):
    cmake = shutil.which("cmake")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda-13.3/bin/nvcc"
    if cmake is None or not Path(nvcc).is_file():
        pytest.skip("CMake and nvcc are required")
    source = tmp_path / "source"
    build = tmp_path / "build"
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    mode = "full"
    _write_cuda_fixture_project(source, manifest, mode)
    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={'b' * 64}",
        f"-DSYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256={'e' * 64}",
    )
    assert configured.returncode == 0, configured.stderr
    compiled = _run(cmake, "--build", build, "--parallel", "2")
    assert compiled.returncode == 0, compiled.stderr
    install = tmp_path / "install"
    installed = _run(cmake, "--install", build, "--prefix", install)
    assert installed.returncode == 0, installed.stderr

    root = install / "fixture_package/_operator_artifacts"
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    oam_entries = [
        entry for entry in catalog["artifacts"] if entry["operator"] == "OAM"
    ]
    assert len(oam_entries) == 8
    assert {entry["identity"]["target"] for entry in oam_entries} == {"sm_120"}
    assert {entry["identity"]["precision"] for entry in oam_entries} == {
        "float32",
        "float64",
    }
    assert (
        len({entry["identity"]["architecture_fingerprint"] for entry in oam_entries})
        == 4
    )
    assert all(entry["operator"] != "OAM" for entry in catalog["artifacts"][:6])
    installed_files = [catalog_path]
    for entry in catalog["artifacts"]:
        installed_files.append(root / entry["installed_relative_path"])
        installed_files.extend(
            root / auxiliary["installed_relative_path"]
            for auxiliary in entry.get("auxiliary_files", ())
        )
    provenance = oam_entries[0]["build_provenance"]
    build_info = {
        "mode": mode,
        "backend": "cuda",
        "target": "sm_120",
        "catalog_relative_path": "_operator_artifacts/catalog.json",
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifact_count": len(catalog["artifacts"]),
        "installed_bytes": sum(path.stat().st_size for path in installed_files),
        "source_content_sha256": "b" * 64,
        "native_source_content_sha256": "e" * 64,
        "compiler_id": provenance["compiler_id"],
        "compiler_version": provenance["compiler_version"],
        "build_duration_seconds": 0.1,
    }
    native = SimpleNamespace(
        __package__="fixture_package",
        _backend_build_info=lambda: {"prebuilt_operator_artifacts": build_info},
    )
    installed_module = importlib.import_module("symmetrix.oam_installed_artifact")
    monkeypatch.setattr(
        installed_module.resources,
        "files",
        lambda package_name: install / "fixture_package",
    )
    for entry in oam_entries:
        identity = entry["identity"]
        resolved = installed_module.resolve_installed_oam_artifact(
            architecture=identity["architecture"],
            precision=identity["precision"],
            backend="cuda",
            target="sm120",
            artifact_id=identity["artifact_id"],
            required_capabilities=entry["capabilities"],
            native_module=native,
        )
        assert resolved.available
        assert resolved.artifact_path == root / entry["installed_relative_path"]


def test_cpu_verified_build_is_noop_until_manifest_dependency_changes(tmp_path):
    compiler = shutil.which("c++")
    cmake = shutil.which("cmake")
    if compiler is None or cmake is None:
        pytest.skip("CMake and a C++ compiler are required")
    manifest = tmp_path / "foundation_operator_manifest.json"
    shutil.copyfile(
        ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json",
        manifest,
    )
    source = tmp_path / "source"
    build = tmp_path / "build"
    mode = "r0"
    private_runtime = build / "private-runtime"
    _write_fixture_project(
        source,
        manifest,
        mode,
        extra=(
            "find_package(OpenMP REQUIRED COMPONENTS CXX)\n"
            "set(Kokkos_ENABLE_OPENMP ON)\n"
            f'set(CMAKE_BUILD_RPATH "{private_runtime.as_posix()}")'
        ),
    )

    configured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={'b' * 64}",
        f"-DSYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256={'e' * 64}",
    )
    assert configured.returncode == 0, configured.stderr
    first_build = _run(cmake, "--build", build, "--parallel", "2")
    assert first_build.returncode == 0, first_build.stderr
    assert "warning:" not in first_build.stderr
    build_inputs = build / "generated-operator-artifacts/build-inputs.txt"
    assert "source_content_sha256=" + "b" * 64 in build_inputs.read_text()
    assert "generation_version_header_sha256=" in build_inputs.read_text()
    first = _artifact_mtimes(build)
    expected_artifacts = 6 if mode == "r0" else 32
    assert len([name for name in first if name.endswith(".so")]) == expected_artifacts
    assert len([name for name in first if name.endswith(".cpp")]) == expected_artifacts
    assert len([name for name in first if name.endswith(".json")]) == (
        expected_artifacts + 2
    )
    assert all(
        os.fsencode(private_runtime) not in path.read_bytes()
        for path in (build / "generated-operator-artifacts").glob("*.so")
    )
    assert all(
        not any(
            dependency.startswith(("libgomp.so", "libomp.so", "libiomp5.so"))
            for dependency in wheel_audit.audit_cpu_operator_dependencies(path)
        )
        for path in (build / "generated-operator-artifacts").glob("*.so")
    )
    metadata_paths = sorted(
        path
        for path in (build / "generated-operator-artifacts").glob("*.json")
        if path.name not in {"catalog.json", "plan.json"}
    )
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    assert {item["build_provenance"]["compiler_id"] for item in metadata} == {"GNU"}
    assert all(item["build_provenance"]["compiler_version"] for item in metadata)
    assert {item["build_provenance"]["source_content_sha256"] for item in metadata} == {
        "b" * 64
    }
    catalog_path = build / "generated-operator-artifacts/catalog.json"
    header_path = build / "generated/prebuilt_operator_build_info.hpp"
    assert f'prebuilt_operator_mode = "{mode}"' in header_path.read_text()
    catalog = json.loads(catalog_path.read_text())
    assert catalog["schema"] == "symmetrix.generated-operator-catalog/1"
    assert len(catalog["artifacts"]) == expected_artifacts
    assert all(
        not Path(item["installed_relative_path"]).is_absolute()
        for item in catalog["artifacts"]
    )
    assert {item["identity"]["target"] for item in catalog["artifacts"]} == {
        "host-isa-x86-64-v3"
    }
    assert {
        item["build_provenance"]["source_content_sha256"]
        for item in catalog["artifacts"]
    } == {"sha256:" + "b" * 64}
    install = tmp_path / "install"
    # Wheel frontends request a stripped install for release builds.  Package
    # bytes must still be the exact outputs hashed by the canonical catalog.
    installed = _run(cmake, "--install", build, "--prefix", install, "--strip")
    assert installed.returncode == 0, installed.stderr
    package_data = install / "fixture_package/_operator_artifacts"
    installed_catalog = json.loads((package_data / "catalog.json").read_text())
    installed_binaries = sorted(package_data.glob("*/*.so"))
    assert installed_catalog == catalog
    assert len(installed_binaries) == expected_artifacts
    assert {path.parent.name for path in installed_binaries} == (
        {"r0"} if mode == "r0" else {"r0", "r1", "m0", "m1"}
    )
    assert not list(package_data.rglob("*.cpp"))
    assert not list(package_data.rglob("*.o"))
    assert list(package_data.rglob("*.json")) == [package_data / "catalog.json"]
    for item in installed_catalog["artifacts"]:
        binary = package_data / item["installed_relative_path"]
        assert binary.is_file()
        assert prebuilt.sha256_file(binary) == item["checksum"]

    second_build = _run(cmake, "--build", build, "--parallel", "2")
    assert second_build.returncode == 0, second_build.stderr
    assert _artifact_mtimes(build) == first

    # A final binary may be relinked without regenerating its metadata.  The
    # catalog must follow the exact linked output rather than only its target's
    # initial completion.
    time.sleep(0.02)
    relink_source = next((build / "generated-operator-artifacts").glob("*.cpp"))
    relink_source.write_text(
        relink_source.read_text()
        + '\nextern "C" int symmetrix_catalog_relink_probe() { return 9; }\n'
    )
    relinked = _run(cmake, "--build", build, "--parallel", "2")
    assert relinked.returncode == 0, relinked.stderr
    relink_catalog = json.loads(catalog_path.read_text())
    for item in relink_catalog["artifacts"]:
        binary = (
            build
            / "generated-operator-artifacts"
            / Path(item["installed_relative_path"]).name
        )
        assert prebuilt.sha256_file(binary) == item["checksum"]
    after_relink = _artifact_mtimes(build)
    assert after_relink["catalog.json"] > first["catalog.json"]

    time.sleep(0.02)
    manifest.write_bytes(manifest.read_bytes())
    invalidated = _run(cmake, "--build", build, "--parallel", "2")
    assert invalidated.returncode == 0, invalidated.stderr
    after = _artifact_mtimes(build)
    assert set(after) == set(after_relink)
    assert all(after[name] > timestamp for name, timestamp in after_relink.items())

    time.sleep(0.02)
    reconfigured = _run(
        cmake,
        "-S",
        source,
        "-B",
        build,
        f"-DSYMMETRIX_BUILD_SOURCE_CONTENT_SHA256={'c' * 64}",
        f"-DSYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256={'f' * 64}",
    )
    assert reconfigured.returncode == 0, reconfigured.stderr
    fingerprint_build = _run(cmake, "--build", build, "--parallel", "2")
    assert fingerprint_build.returncode == 0, fingerprint_build.stderr
    fingerprint_after = _artifact_mtimes(build)
    assert set(fingerprint_after) == set(after)
    assert all(fingerprint_after[name] > timestamp for name, timestamp in after.items())
    assert "source_content_sha256=" + "c" * 64 in build_inputs.read_text()
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    assert {item["build_provenance"]["source_content_sha256"] for item in metadata} == {
        "c" * 64
    }


def test_catalog_uses_exact_outputs_and_emits_aggregate_provenance(
    tmp_path, monkeypatch
):
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    source = tmp_path / "operator.cpp"
    metadata = tmp_path / "operator.json"
    binary = tmp_path / "operator.so"
    catalog = tmp_path / "catalog.json"
    header = tmp_path / "prebuilt_operator_build_info.hpp"
    item = _cpu_plan("r0")["artifacts"][0]
    prebuilt.render_foundation_r0_artifact(
        manifest_path=manifest,
        artifact_input_id=item["artifact_input_id"],
        backend="host",
        target="x86-64-v3",
        precision=item["precision"],
        source_content_sha256="a" * 64,
        compiler_id="GNU",
        compiler_version="15.2.0",
        source_path=source,
        metadata_path=metadata,
    )
    binary.write_bytes(b"relocatable-operator-binary")
    os.utime(binary, ns=(1_250_000_000, 1_250_000_000))
    started = tmp_path / "operator.started-ns"
    started.write_text("1000000000\n")
    monkeypatch.setattr(prebuilt.time, "time_ns", lambda: 1_250_000_000)

    provenance = prebuilt.write_installed_operator_catalog(
        artifacts=[(metadata, binary, f"r0/{binary.name}", started)],
        catalog_path=catalog,
        header_path=header,
        mode="r0",
        backend="host",
        target="x86-64-v3",
        source_content_sha256="a" * 64,
        native_source_content_sha256="b" * 64,
        compiler_id="GNU",
        compiler_version="15.2.0",
    )

    payload = json.loads(catalog.read_text())
    assert payload == prebuilt.validate_generated_operator_catalog(payload)
    assert payload["artifacts"][0]["checksum"] == prebuilt.sha256_file(binary)
    assert payload["artifacts"][0]["installed_relative_path"] == "r0/operator.so"
    assert provenance["mode"] == "r0"
    assert provenance["backend"] == "host"
    assert provenance["target"] == "host-isa-x86-64-v3"
    assert provenance["artifact_count"] == 1
    assert provenance["catalog_sha256"] == prebuilt.sha256_file(catalog).removeprefix(
        "sha256:"
    )
    assert (
        provenance["installed_bytes"] == binary.stat().st_size + catalog.stat().st_size
    )
    assert provenance["source_content_sha256"] == "a" * 64
    assert provenance["native_source_content_sha256"] == "b" * 64
    assert provenance["build_duration_seconds"] == 0.25
    assert str(tmp_path) not in catalog.read_text()
    assert str(tmp_path) not in header.read_text()


def test_catalog_rejects_r0_floating_policy_provenance_mismatch(tmp_path):
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    source = tmp_path / "operator.cpp"
    metadata = tmp_path / "operator.json"
    binary = tmp_path / "operator.so"
    item = _cpu_plan("r0")["artifacts"][0]
    prebuilt.render_foundation_r0_artifact(
        manifest_path=manifest,
        artifact_input_id=item["artifact_input_id"],
        backend="host",
        target="x86-64-v3",
        precision=item["precision"],
        source_content_sha256="a" * 64,
        compiler_id="GNU",
        compiler_version="15.2.0",
        source_path=source,
        metadata_path=metadata,
    )
    payload = json.loads(metadata.read_text())
    payload["build_provenance"]["floating_point_policy"] = "cuda-fast-math"
    metadata.write_text(json.dumps(payload))
    binary.write_bytes(b"operator")
    started = tmp_path / "operator.started-ns"
    started.write_text(f"{time.time_ns()}\n")

    with pytest.raises(ValueError, match="build provenance"):
        prebuilt.write_installed_operator_catalog(
            artifacts=[(metadata, binary, "r0/operator.so", started)],
            catalog_path=tmp_path / "catalog.json",
            header_path=tmp_path / "build_info.hpp",
            mode="r0",
            backend="host",
            target="x86-64-v3",
            source_content_sha256="a" * 64,
            native_source_content_sha256="b" * 64,
            compiler_id="GNU",
            compiler_version="15.2.0",
        )


@pytest.mark.parametrize("unsafe", ("/absolute/operator.so", "../operator.so"))
def test_catalog_writer_rejects_non_relocatable_installed_paths(tmp_path, unsafe):
    metadata = tmp_path / "operator.json"
    binary = tmp_path / "operator.so"
    metadata.write_text("{}")
    binary.write_bytes(b"binary")
    started = tmp_path / "operator.started-ns"
    started.write_text(f"{time.time_ns()}\n")

    with pytest.raises(ValueError, match="installed_relative_path"):
        prebuilt.write_installed_operator_catalog(
            artifacts=[(metadata, binary, unsafe, started)],
            catalog_path=tmp_path / "catalog.json",
            header_path=tmp_path / "build_info.hpp",
            mode="r0",
            backend="host",
            target="x86-64-v3",
            source_content_sha256="a" * 64,
            native_source_content_sha256="b" * 64,
            compiler_id="GNU",
            compiler_version="15.2.0",
        )


def test_artifact_build_duration_uses_only_outputs_from_incremental_relink():
    duration = prebuilt._artifact_build_duration_seconds(
        (
            (4_000_000_000, 1_000_000_000),
            (10_000_000_000, 9_800_000_000),
        ),
        previous_catalog_mtime_ns=5_000_000_000,
    )

    assert duration == 0.2


def test_artifact_build_duration_preserves_overlapping_batch_wall_time():
    duration = prebuilt._artifact_build_duration_seconds(
        (
            (1_250_000_000, 1_000_000_000),
            (1_300_000_000, 1_100_000_000),
        ),
        previous_catalog_mtime_ns=None,
    )

    assert duration == 0.3


def test_catalog_recreation_uses_persisted_output_finish_not_idle_wall_time(
    tmp_path, monkeypatch
):
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    source = tmp_path / "operator.cpp"
    metadata = tmp_path / "operator.json"
    binary = tmp_path / "operator.so"
    catalog = tmp_path / "catalog.json"
    header = tmp_path / "prebuilt_operator_build_info.hpp"
    item = _cpu_plan("r0")["artifacts"][0]
    prebuilt.render_foundation_r0_artifact(
        manifest_path=manifest,
        artifact_input_id=item["artifact_input_id"],
        backend="host",
        target="x86-64-v3",
        precision=item["precision"],
        source_content_sha256="a" * 64,
        compiler_id="GNU",
        compiler_version="15.2.0",
        source_path=source,
        metadata_path=metadata,
    )
    day_ns = 86_400_000_000_000
    started_ns = day_ns
    finished_ns = started_ns + 250_000_000
    binary.write_bytes(b"relocatable-operator-binary")
    os.utime(binary, ns=(finished_ns, finished_ns))
    started = tmp_path / "operator.started-ns"
    started.write_text(f"{started_ns}\n")
    monkeypatch.setattr(prebuilt.time, "time_ns", lambda: 2 * day_ns)
    assert not catalog.exists()
    assert not header.exists()

    provenance = prebuilt.write_installed_operator_catalog(
        artifacts=[(metadata, binary, f"r0/{binary.name}", started)],
        catalog_path=catalog,
        header_path=header,
        mode="r0",
        backend="host",
        target="x86-64-v3",
        source_content_sha256="a" * 64,
        native_source_content_sha256="b" * 64,
        compiler_id="GNU",
        compiler_version="15.2.0",
    )

    assert provenance["build_duration_seconds"] == 0.25


def test_catalog_recreation_excludes_gap_between_separated_artifact_rebuilds(
    tmp_path, monkeypatch
):
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    day_ns = 86_400_000_000_000
    artifacts = []
    for index, item in enumerate(_cpu_plan("r0")["artifacts"][:2], start=1):
        source = tmp_path / f"operator-{index}.cpp"
        metadata = tmp_path / f"operator-{index}.json"
        binary = tmp_path / f"operator-{index}.so"
        started = tmp_path / f"operator-{index}.started-ns"
        prebuilt.render_foundation_r0_artifact(
            manifest_path=manifest,
            artifact_input_id=item["artifact_input_id"],
            backend="host",
            target="x86-64-v3",
            precision=item["precision"],
            source_content_sha256="a" * 64,
            compiler_id="GNU",
            compiler_version="15.2.0",
            source_path=source,
            metadata_path=metadata,
        )
        started_ns = index * day_ns
        finished_ns = started_ns + 250_000_000
        binary.write_bytes(f"operator-{index}".encode())
        os.utime(binary, ns=(finished_ns, finished_ns))
        started.write_text(f"{started_ns}\n")
        artifacts.append((metadata, binary, f"r0/{binary.name}", started))
    monkeypatch.setattr(prebuilt.time, "time_ns", lambda: 3 * day_ns)
    catalog = tmp_path / "catalog.json"
    header = tmp_path / "prebuilt_operator_build_info.hpp"
    assert not catalog.exists()
    assert not header.exists()

    provenance = prebuilt.write_installed_operator_catalog(
        artifacts=artifacts,
        catalog_path=catalog,
        header_path=header,
        mode="r0",
        backend="host",
        target="x86-64-v3",
        source_content_sha256="a" * 64,
        native_source_content_sha256="b" * 64,
        compiler_id="GNU",
        compiler_version="15.2.0",
    )

    assert provenance["artifact_count"] == 2
    assert provenance["build_duration_seconds"] == 0.5


def test_build_start_timestamp_preserves_generation_start_but_resets_for_relink(
    tmp_path, monkeypatch
):
    started = tmp_path / "operator.started-ns"
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}")
    started.write_text("100\n")
    os.utime(catalog, ns=(200, 200))
    os.utime(started, ns=(300, 300))
    monkeypatch.setattr(prebuilt.time, "time_ns", lambda: 400)

    prebuilt.write_build_start_timestamp(started, preserve_if_newer_than=catalog)
    assert started.read_text() == "100\n"

    os.utime(started, ns=(100, 100))
    prebuilt.write_build_start_timestamp(started, preserve_if_newer_than=catalog)
    assert started.read_text() == "400\n"


def test_off_mode_installs_no_operator_package_data(tmp_path):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(prebuilt_operator_off LANGUAGES NONE)
set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS off)
set(SYMMETRIX_PYTHON_INSTALL_PACKAGE fixture_package)
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )
    build = tmp_path / "build"
    install = tmp_path / "install"
    configured = _run(cmake, "-S", source, "-B", build)
    assert configured.returncode == 0, configured.stderr
    installed = _run(cmake, "--install", build, "--prefix", install)
    assert installed.returncode == 0, installed.stderr
    assert not (install / "fixture_package/_operator_artifacts").exists()


@pytest.mark.parametrize(
    ("backend", "target", "compiler_name", "suffix", "binary_format"),
    (
        ("cuda", "sm_120", "nvcc", ".cubin", "cubin"),
        ("hip", "gfx1100", "hipcc", ".hsaco", "code-object"),
    ),
)
@pytest.mark.parametrize("mode", ("r0", "full"))
def test_accelerator_compiler_installs_exact_package_paths(
    tmp_path, backend, target, compiler_name, suffix, binary_format, mode
):
    cmake = shutil.which("cmake")
    compiler = shutil.which(compiler_name)
    if compiler is None and compiler_name == "nvcc":
        conventional_nvcc = Path("/usr/local/cuda/bin/nvcc")
        compiler = os.fspath(conventional_nvcc) if conventional_nvcc.is_file() else None
    if cmake is None or compiler is None:
        pytest.skip(
            f"CMake and {compiler_name} are required for {backend} package paths"
        )
    version = _run(compiler, "--version")
    if version.returncode != 0:
        pytest.skip(f"{compiler_name} is present but not runnable: {version.stderr}")

    source = tmp_path / "source"
    source.mkdir()
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    if backend == "cuda":
        compiler_setup = f"""
set(CMAKE_CUDA_COMPILER {compiler})
"""
        configure_arguments = ()
    else:
        compiler_setup = ""
        configure_arguments = (f"-DCMAKE_CXX_COMPILER={compiler}",)
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(prebuilt_operator_{backend} LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter REQUIRED)
set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS {mode})
set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND {backend})
set(SYMMETRIX_PREBUILT_OPERATOR_TARGET {target})
set(SYMMETRIX_PREBUILT_OPERATOR_MANIFEST {manifest.as_posix()})
set(SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 {"c" * 64})
set(SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256 {"d" * 64})
set(SYMMETRIX_PYTHON_INSTALL_PACKAGE fixture_package)
{compiler_setup}
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )
    build = tmp_path / "build"
    configured = _run(cmake, "-S", source, "-B", build, *configure_arguments)
    assert configured.returncode == 0, configured.stderr
    if backend == "cuda":
        build_inputs = (
            build / "generated-operator-artifacts/build-inputs.txt"
        ).read_text()
        assert "compiler_id=NVIDIA\n" in build_inputs
        assert "compiler_version=\n" not in build_inputs
    compiled = _run(cmake, "--build", build, "--parallel", "2")
    assert compiled.returncode == 0, compiled.stderr
    install = tmp_path / "install"
    installed = _run(cmake, "--install", build, "--prefix", install)
    assert installed.returncode == 0, installed.stderr

    root = install / "fixture_package/_operator_artifacts"
    catalog = json.loads((root / "catalog.json").read_text())
    expected_artifacts = 6 if mode == "r0" else 32
    assert len(catalog["artifacts"]) == expected_artifacts
    assert {item["identity"]["backend"] for item in catalog["artifacts"]} == {backend}
    assert {item["identity"]["target"] for item in catalog["artifacts"]} == {target}
    assert {item["binary_format"] for item in catalog["artifacts"]} == {binary_format}
    assert len(list(root.glob(f"*/*{suffix}"))) == expected_artifacts
    assert {path.parent.name for path in root.glob(f"*/*{suffix}")} == (
        {"r0"} if mode == "r0" else {"r0", "r1", "m0", "m1"}
    )
    for item in catalog["artifacts"]:
        path = root / item["installed_relative_path"]
        assert path.suffix == suffix
        assert prebuilt.sha256_file(path) == item["checksum"]


@pytest.mark.parametrize(
    ("backend", "target", "compiler_variable", "missing_name"),
    (
        ("cuda", "sm120", "CMAKE_CUDA_COMPILER", "missing-nvcc"),
        ("hip", "gfx1151", "CMAKE_CXX_COMPILER", "missing-hipcc"),
    ),
)
def test_accelerator_mode_rejects_an_unavailable_compiler(
    tmp_path, backend, target, compiler_variable, missing_name
):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required")
    source = tmp_path / "source"
    source.mkdir()
    missing = tmp_path / missing_name
    manifest = ROOT / "symmetrix/source/symmetrix/foundation_operator_manifest.json"
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.27)
project(prebuilt_operator_missing_compiler LANGUAGES NONE)
find_package(Python COMPONENTS Interpreter REQUIRED)
set(SYMMETRIX_PREBUILT_OPERATOR_ARTIFACTS r0)
set(SYMMETRIX_PREBUILT_OPERATOR_BACKEND {backend})
set(SYMMETRIX_PREBUILT_OPERATOR_TARGET {target})
set(SYMMETRIX_PREBUILT_OPERATOR_MANIFEST {manifest.as_posix()})
set(SYMMETRIX_BUILD_SOURCE_CONTENT_SHA256 {"c" * 64})
set(SYMMETRIX_BUILD_NATIVE_SOURCE_CONTENT_SHA256 {"d" * 64})
set({compiler_variable} {missing.as_posix()})
include({(ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake").as_posix()})
symmetrix_add_prebuilt_operator_artifacts()
"""
    )

    configured = _run(cmake, "-S", source, "-B", tmp_path / "build")

    assert configured.returncode != 0
    assert f"{backend.upper()} compiler is unavailable" in configured.stderr


def test_offline_oam_cuda_build_selects_the_module_identity_branch():
    cmake_source = (
        ROOT / "libsymmetrix/cmake/prebuilt_operator_artifacts.cmake"
    ).read_text(encoding="utf-8")

    assert '_symmetrix_operator STREQUAL "OAM"' in cmake_source
    assert "-DSYMMETRIX_OAM_CUDA_MODULE=1" in cmake_source
