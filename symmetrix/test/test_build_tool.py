import importlib
import io
import json
import os
import sys
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

build_command = importlib.import_module("_symmetrix_build.command")
build_cli = importlib.import_module("_symmetrix_build.cli")
build_detect = importlib.import_module("_symmetrix_build.detect")
build_lammps = importlib.import_module("_symmetrix_build.lammps_build")
build_manifest = importlib.import_module("_symmetrix_build.manifest")
build_matrix = importlib.import_module("_symmetrix_build.matrix")
build_package_identity = importlib.import_module("_symmetrix_build.package_identity")
build_python = importlib.import_module("_symmetrix_build.python_build")
build_sdist = importlib.import_module("_symmetrix_build.sdist_build")
build_targets = importlib.import_module("_symmetrix_build.targets")
build_wheel_audit = importlib.import_module("_symmetrix_build.wheel_audit")

BuildError = build_command.BuildError
CommandResult = build_command.CommandResult
CommandRunner = build_command.CommandRunner
DetectionRequest = build_detect.DetectionRequest
DeviceProbe = build_detect.DeviceProbe
detect_target = build_detect.detect_target
probe_devices = build_detect.probe_devices
lammps_cmake_definitions = build_lammps.lammps_cmake_definitions
lammps_build_invocation = build_lammps.lammps_build_invocation
lammps_source_provenance = build_lammps.lammps_source_provenance
parse_lammps_version = build_lammps.parse_lammps_version
probe_mpi_gpu_awareness = build_lammps.probe_mpi_gpu_awareness
reuse_lammps_qualification = build_lammps.reuse_lammps_qualification
run_lammps_build = build_lammps.run_lammps_build
symmetrix_source_fingerprint = build_lammps.symmetrix_source_fingerprint
validate_lammps_source = build_lammps.validate_lammps_source
TargetManifest = build_manifest.TargetManifest
Toolchain = build_manifest.Toolchain
cmake_definitions = build_python.cmake_definitions
build_environment = build_python.build_environment
prepare_build_directory = build_python.prepare_build_directory
python_build_invocation = build_python.python_build_invocation
verify_python_cmake_provenance = build_python.verify_python_cmake_provenance
verify_frontend_installed = build_python.verify_frontend_installed
package_identity = build_package_identity.package_identity
Architecture = build_targets.Architecture
cuda_architecture = build_targets.cuda_architecture
hip_architecture = build_targets.hip_architecture
validate_cuda_target_for_toolkit = build_targets.validate_cuda_target_for_toolkit
audit_wheel = build_wheel_audit.audit_wheel


class FakeRunner:
    def __init__(self, executables, results):
        self.executables = executables
        self.results = results

    def which(self, command):
        return self.executables.get(command)

    def run(self, args, **_kwargs):
        normalized = tuple(os.fspath(value) for value in args)
        key = (Path(normalized[0]).name, *normalized[1:])
        returncode, stdout, stderr = self.results[key]
        return CommandResult(normalized, returncode, stdout, stderr)

    def run_logged(self, args, log_path, **_kwargs):
        result = self.run(args)
        Path(log_path).write_text(result.output)
        return result


def _executable(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


def _manifest(backend="cpu", target="native"):
    if backend == "cpu":
        architecture = Architecture(target, target, "NATIVE", target)
        host_target = target
        toolkit_root = ""
    elif backend == "cuda":
        architecture = cuda_architecture(target, "")
        host_target = "none"
        toolkit_root = "/usr/local/cuda"
    else:
        architecture = hip_architecture(target, "")
        host_target = "none"
        toolkit_root = "/opt/rocm"
    return TargetManifest(
        backend=backend,
        architecture=architecture,
        toolchain=Toolchain(
            cxx="/usr/bin/c++",
            cxx_version="test compiler",
            host_cxx="/usr/bin/g++" if backend == "cuda" else "",
            cmake="/usr/bin/cmake",
            cmake_version="4.2.0",
            generator="Unix Makefiles",
            fortran="/usr/bin/gfortran" if backend == "cpu" else "",
            toolkit_root=toolkit_root,
            toolkit_version=(
                "13.3" if backend == "cuda" else ("7.0" if backend == "hip" else "")
            ),
            runtime_library_dirs=(
                ("/opt/rocm/core-test/lib",) if backend == "hip" else ()
            ),
        ),
        python_executable=sys.executable,
        python_abi=sys.implementation.cache_tag,
        host_target=host_target,
        blas_policy="openblas" if backend == "cpu" else "auto",
        blas_root="",
        device_probe="test",
        visible_devices=(architecture.device_target,),
    )


def _lammps_source(root, *, with_wrapper=False, version="10 Dec 2025", packages=()):
    (root / "src/KOKKOS").mkdir(parents=True)
    (root / "cmake").mkdir()
    (root / "src/version.h").write_text(f'#define LAMMPS_VERSION "{version}"\n')
    (root / "cmake/CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.27)\n")
    for package in packages:
        (root / "src" / package).mkdir()
    if with_wrapper:
        wrapper = root / "lib/kokkos/bin/nvcc_wrapper"
        wrapper.parent.mkdir(parents=True)
        _executable(wrapper)
    return root


@pytest.mark.parametrize(
    ("value", "normalized"),
    (("8.6", "sm86"), ("sm_120", "sm120"), ("120", "sm120")),
)
def test_cuda_target_normalization(value, normalized):
    assert cuda_architecture(value, "").device_target == normalized


def test_cuda_12_supports_volta70():
    architecture = cuda_architecture("sm70", "")

    validate_cuda_target_for_toolkit("12.9", architecture.device_target)

    assert architecture.kokkos_trait == "VOLTA70"
    assert architecture.compiler_target == "sm_70"


def test_cuda_13_rejects_volta70():
    with pytest.raises(BuildError, match="use a CUDA 12 toolkit for Volta"):
        validate_cuda_target_for_toolkit("13.3", "sm70")


def test_hip_target_keeps_exact_compiler_target():
    architecture = hip_architecture("gfx1151:sramecc+", "")
    assert architecture.device_target == "gfx1151"
    assert architecture.kokkos_trait == "AMD_GFX1100"
    assert architecture.compiler_target == "gfx1151"


def test_probe_retains_unsupported_targets_for_selected_backend_validation(tmp_path):
    nvidia_smi = _executable(tmp_path / "nvidia-smi")
    rocm_probe = _executable(tmp_path / "rocm_agent_enumerator")
    runner = FakeRunner(
        {
            "nvidia-smi": nvidia_smi,
            "rocm_agent_enumerator": rocm_probe,
        },
        {
            (
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ): (0, "9.9\n", ""),
            ("rocm_agent_enumerator",): (
                0,
                "diagnostic text\ngfx000\ngfx9999:sramecc+\n",
                "",
            ),
        },
    )

    devices = probe_devices(runner)

    assert devices.nvidia == ("sm99",)
    assert devices.amd == ("gfx9999",)


def test_detection_rejects_mixed_vendors_before_toolchain_probe():
    with pytest.raises(BuildError, match="both NVIDIA and AMD"):
        detect_target(
            DetectionRequest(),
            repo_root=REPOSITORY_ROOT,
            devices=DeviceProbe(("sm120",), ("gfx1151",)),
            runner=FakeRunner({}, {}),
        )


def test_detection_rejects_heterogeneous_selected_backend(tmp_path):
    cmake = _executable(tmp_path / "cmake")
    with pytest.raises(BuildError, match="heterogeneous"):
        detect_target(
            DetectionRequest(backend="cuda"),
            repo_root=REPOSITORY_ROOT,
            devices=DeviceProbe(("sm86", "sm120"), ()),
            runner=FakeRunner(
                {"cmake": cmake},
                {("cmake", "--version"): (0, "cmake version 4.2.0\n", "")},
            ),
        )


def test_manifest_round_trip_preserves_fingerprint(tmp_path):
    manifest = _manifest("hip", "gfx1151")
    path = tmp_path / "target.json"

    manifest.write(path)
    loaded = TargetManifest.read(path)

    assert loaded == manifest
    assert loaded.fingerprint == manifest.fingerprint


def test_manifest_replay_preserves_automatic_target_evidence(tmp_path):
    manifest = _manifest("hip", "gfx1151")
    manifest = replace(
        manifest,
        architecture=replace(manifest.architecture, requested="auto"),
        device_probe="/qualified/rocm_agent_enumerator",
    )
    path = tmp_path / "target.json"
    manifest.write(path)
    args = SimpleNamespace(
        target_manifest=path,
        backend=None,
        arch=None,
        cpu_target=None,
        cuda_root=None,
        rocm_root=None,
        cxx=None,
        host_cxx=None,
        generator=None,
        blas=None,
        mkl_root=None,
    )

    request, devices = build_cli._base_request(args)

    assert request.arch == "auto"
    assert devices.amd == ("gfx1151",)
    assert devices.amd_command == manifest.device_probe


@pytest.mark.parametrize(
    ("original_backend", "new_backend"),
    (("cpu", "cuda"), ("cuda", "cpu"), ("hip", "cpu")),
)
def test_cross_backend_manifest_replay_resets_backend_specific_fields(
    tmp_path, original_backend, new_backend
):
    targets = {"cpu": "x86-64-v3", "cuda": "sm120", "hip": "gfx1151"}
    manifest = _manifest(original_backend, targets[original_backend])
    path = tmp_path / "target.json"
    manifest.write(path)
    args = SimpleNamespace(
        target_manifest=path,
        backend=new_backend,
        arch=None,
        cpu_target=None,
        cuda_root=None,
        rocm_root=None,
        cxx=None,
        host_cxx=None,
        generator=None,
        blas=None,
        mkl_root=None,
    )

    request, devices = build_cli._base_request(args)

    assert request.arch == "auto"
    assert request.cpu_target == "native"
    assert request.cxx == ""
    assert request.host_cxx == ""
    assert request.cuda_root == ""
    assert request.rocm_root == ""
    assert request.generator == manifest.toolchain.generator
    assert devices is None


def test_cross_backend_manifest_replay_honors_explicit_overrides(tmp_path):
    manifest = _manifest("cpu", "x86-64-v3")
    path = tmp_path / "target.json"
    manifest.write(path)
    args = SimpleNamespace(
        target_manifest=path,
        backend="cuda",
        arch="sm120",
        cpu_target="none",
        cuda_root=Path("/explicit/cuda"),
        rocm_root=None,
        cxx=Path("/explicit/nvcc_wrapper"),
        host_cxx=Path("/explicit/g++"),
        generator="Ninja",
        blas=None,
        mkl_root=None,
    )

    request, _ = build_cli._base_request(args)

    assert request.arch == "sm120"
    assert request.cpu_target == "none"
    assert request.cuda_root == "/explicit/cuda"
    assert request.cxx == "/explicit/nvcc_wrapper"
    assert request.host_cxx == "/explicit/g++"
    assert request.generator == "Ninja"


@pytest.mark.parametrize(
    ("policy", "expected"),
    (
        ("openblas", "openblas"),
        ("mkl_gnu_thread", "mkl"),
        ("mkl_sequential", "mkl-sequential"),
    ),
)
def test_cpu_manifest_replay_preserves_blas_policy(tmp_path, policy, expected):
    manifest = replace(
        _manifest("cpu", "x86-64-v3"),
        blas_policy=policy,
        blas_root=("/opt/mkl" if policy.startswith("mkl_") else ""),
    )
    path = tmp_path / "target.json"
    manifest.write(path)
    args = SimpleNamespace(
        target_manifest=path,
        backend=None,
        arch=None,
        cpu_target=None,
        cuda_root=None,
        rocm_root=None,
        cxx=None,
        host_cxx=None,
        generator=None,
        blas=None,
        mkl_root=None,
    )

    request, _ = build_cli._base_request(args)

    assert request.blas == expected
    assert request.mkl_root == manifest.blas_root


def test_explicit_invalid_mkl_root_does_not_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("MKLROOT", "/opt/intel/oneapi/mkl/latest")

    with pytest.raises(BuildError, match="no MKLROOT"):
        build_detect._cpu_blas_policy("mkl", str(tmp_path / "missing"))


def test_automatic_blas_prefers_openblas(monkeypatch):
    monkeypatch.setattr(
        build_detect.ctypes.util, "find_library", lambda name: f"lib{name}.so"
    )

    assert build_detect._cpu_blas_policy("auto") == ("openblas", "")


def test_accelerator_static_blas_root_is_validated(tmp_path):
    root = tmp_path / "openblas"
    (root / "lib").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "lib/libopenblas.a").write_bytes(b"archive")
    (root / "include/cblas.h").write_text("/* cblas */\n")

    assert build_detect._accelerator_blas_policy(str(root)) == (
        "openblas_static",
        str(root.resolve()),
    )
    assert build_detect._accelerator_blas_policy() == ("system", "")

    (root / "include/cblas.h").unlink()
    with pytest.raises(BuildError, match="include/cblas.h"):
        build_detect._accelerator_blas_policy(str(root))


def test_cpu_detection_requires_fortran(monkeypatch, tmp_path):
    monkeypatch.setattr(
        build_detect.ctypes.util,
        "find_library",
        lambda name: "libopenblas.so" if name == "openblas" else None,
    )
    cmake = _executable(tmp_path / "cmake")
    cxx = _executable(tmp_path / "g++")
    runner = FakeRunner(
        {"cmake": cmake, "g++": cxx},
        {
            ("cmake", "--version"): (0, "cmake version 4.2.0\n", ""),
            ("g++", "--version"): (0, "g++ test compiler\n", ""),
        },
    )

    with pytest.raises(BuildError, match="CPU builds require a Fortran compiler"):
        detect_target(
            DetectionRequest(backend="cpu"),
            repo_root=REPOSITORY_ROOT,
            devices=DeviceProbe((), ()),
            runner=runner,
        )


@pytest.mark.parametrize("blas", ("mkl", "mkl-sequential"))
def test_mkl_detection_requires_gfortran(tmp_path, blas):
    cmake = _executable(tmp_path / "cmake")
    cxx = _executable(tmp_path / "g++")
    flang = _executable(tmp_path / "flang")
    mkl_root = tmp_path / "mkl"
    (mkl_root / "include").mkdir(parents=True)
    (mkl_root / "lib").mkdir()
    (mkl_root / "include/mkl.h").write_text("")
    (mkl_root / "lib/libmkl_gf_lp64.so").write_text("")
    (mkl_root / "lib/libmkl_core.so").write_text("")
    (mkl_root / "lib/libmkl_gnu_thread.so").write_text("")
    (mkl_root / "lib/libmkl_sequential.so").write_text("")
    runner = FakeRunner(
        {"cmake": cmake, "g++": cxx, "flang": flang},
        {
            ("cmake", "--version"): (0, "cmake version 4.2.0\n", ""),
            ("g++", "--version"): (0, "g++ test compiler\n", ""),
        },
    )

    with pytest.raises(BuildError, match="MKL requires gfortran"):
        detect_target(
            DetectionRequest(backend="cpu", blas=blas, mkl_root=str(mkl_root)),
            repo_root=REPOSITORY_ROOT,
            devices=DeviceProbe((), ()),
            runner=runner,
        )


def test_mkl_gnu_thread_detection_requires_gnu_cxx(tmp_path):
    cmake = _executable(tmp_path / "cmake")
    clang = _executable(tmp_path / "clang++")
    gfortran = _executable(tmp_path / "gfortran")
    mkl_root = tmp_path / "mkl"
    (mkl_root / "include").mkdir(parents=True)
    (mkl_root / "lib").mkdir()
    (mkl_root / "include/mkl.h").write_text("")
    (mkl_root / "lib/libmkl_gf_lp64.so").write_text("")
    (mkl_root / "lib/libmkl_core.so").write_text("")
    (mkl_root / "lib/libmkl_gnu_thread.so").write_text("")
    runner = FakeRunner(
        {"cmake": cmake, "clang++": clang, "gfortran": gfortran},
        {
            ("cmake", "--version"): (0, "cmake version 4.2.0\n", ""),
            ("clang++", "--version"): (0, "clang version 21.1.0\n", ""),
        },
    )

    with pytest.raises(BuildError, match="GNU-threaded MKL requires a GNU C\\+\\+"):
        detect_target(
            DetectionRequest(
                backend="cpu",
                blas="mkl",
                mkl_root=str(mkl_root),
                cxx=clang,
            ),
            repo_root=REPOSITORY_ROOT,
            devices=DeviceProbe((), ()),
            runner=runner,
        )


def test_mkl_sequential_detection_accepts_clang_with_gfortran(tmp_path):
    cmake = _executable(tmp_path / "cmake")
    clang = _executable(tmp_path / "clang++")
    gfortran = _executable(tmp_path / "gfortran")
    mkl_root = tmp_path / "mkl"
    (mkl_root / "include").mkdir(parents=True)
    (mkl_root / "lib").mkdir()
    (mkl_root / "include/mkl.h").write_text("")
    (mkl_root / "lib/libmkl_gf_lp64.so").write_text("")
    (mkl_root / "lib/libmkl_core.so").write_text("")
    (mkl_root / "lib/libmkl_sequential.so").write_text("")
    runner = FakeRunner(
        {"cmake": cmake, "clang++": clang, "gfortran": gfortran},
        {
            ("cmake", "--version"): (0, "cmake version 4.2.0\n", ""),
            ("clang++", "--version"): (0, "clang version 21.1.0\n", ""),
        },
    )

    manifest = detect_target(
        DetectionRequest(
            backend="cpu",
            blas="mkl-sequential",
            mkl_root=str(mkl_root),
            cxx=clang,
        ),
        repo_root=REPOSITORY_ROOT,
        devices=DeviceProbe((), ()),
        runner=runner,
    )

    assert manifest.blas_policy == "mkl_sequential"
    assert manifest.toolchain.cxx == clang
    assert manifest.toolchain.fortran == gfortran


@pytest.mark.parametrize(
    ("backend", "target", "expected"),
    (
        ("cpu", "x86-64-v3", ("Kokkos_ENABLE_OPENMP", "ON")),
        ("cuda", "sm120", ("Kokkos_ARCH_BLACKWELL120", "ON")),
        ("hip", "gfx1151", ("Kokkos_IMPL_AMDGPU_FLAGS", "--offload-arch=gfx1151")),
    ),
)
def test_python_cmake_definitions_match_target(backend, target, expected):
    assert cmake_definitions(_manifest(backend, target))[expected[0]] == expected[1]


@pytest.mark.parametrize(("backend", "target"), (("cuda", "sm120"), ("hip", "gfx1151")))
def test_gpu_definitions_disable_native_host_architecture(backend, target):
    definitions = cmake_definitions(_manifest(backend, target))

    assert definitions["SYMMETRIX_HOST_ARCH"] == "none"
    assert definitions["Kokkos_ARCH_NATIVE"] == "OFF"
    assert definitions["Kokkos_ENABLE_SERIAL"] == "ON"
    assert definitions["Kokkos_ENABLE_OPENMP"] == "OFF"
    arch_flag_keys = {
        "SYMMETRIX_HOST_ARCH",
        "Kokkos_ARCH_NATIVE",
        "Kokkos_IMPL_AMDGPU_FLAGS",
        "Kokkos_IMPL_AMDGPU_LINK",
    }
    assert not any(
        "native" in definitions.get(key, "").lower() for key in arch_flag_keys
    )


@pytest.mark.parametrize(
    ("toolkit_version", "install_rpath"),
    (
        ("12.9", "$ORIGIN/../nvidia/cuda_runtime/lib;$ORIGIN/../nvidia/cublas/lib"),
        ("13.3", "$ORIGIN/../nvidia/cu13/lib"),
    ),
)
def test_cuda_definitions_use_packaged_nvidia_runtime_paths(
    toolkit_version, install_rpath
):
    manifest = _manifest("cuda", "sm120")
    manifest = replace(
        manifest,
        toolchain=replace(manifest.toolchain, toolkit_version=toolkit_version),
    )
    definitions = cmake_definitions(manifest)

    assert definitions["CMAKE_BUILD_RPATH"] == ""
    assert definitions["CMAKE_INSTALL_RPATH"] == install_rpath
    assert definitions["SPHERICART_OPENMP"] == "OFF"
    assert definitions["KokkosKernels_ENABLE_TPL_CUSPARSE"] == "OFF"
    assert definitions["KokkosKernels_ENABLE_TPL_CUSOLVER"] == "OFF"


def test_static_accelerator_blas_is_explicit_and_hidden(tmp_path):
    root = tmp_path / "openblas"
    (root / "lib").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "lib/libopenblas.a").write_bytes(b"archive")
    (root / "include/cblas.h").write_text("/* cblas */\n")
    manifest = replace(
        _manifest("cuda", "sm120"),
        blas_policy="openblas_static",
        blas_root=str(root),
    )

    definitions = cmake_definitions(manifest)

    assert definitions["SYMMETRIX_BLAS_LIBRARY"] == str(root / "lib/libopenblas.a")
    assert definitions["SYMMETRIX_BLAS_INCLUDE_DIR"] == str(root / "include")
    assert definitions["SYMMETRIX_HIDE_STATIC_BLAS_SYMBOLS"] == "ON"


def test_python_cmake_definitions_select_mkl_gnu_thread():
    manifest = replace(
        _manifest("cpu", "x86-64-v3"),
        blas_policy="mkl_gnu_thread",
        blas_root="/opt/intel/oneapi/mkl/latest",
    )

    definitions = cmake_definitions(manifest)

    assert definitions["BLA_VENDOR"] == "Intel10_64lp"
    assert definitions["CMAKE_Fortran_COMPILER"] == "/usr/bin/gfortran"
    assert definitions["MKLROOT"] == "/opt/intel/oneapi/mkl/latest"
    assert definitions["CMAKE_INSTALL_RPATH"] == ("/opt/intel/oneapi/mkl/latest/lib")

    environment = build_environment(manifest)
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep)[0] == (
        "/opt/intel/oneapi/mkl/latest/lib"
    )


def test_lammps_cmake_definitions_select_mkl_sequential(tmp_path):
    manifest = replace(
        _manifest("cpu", "native"),
        blas_policy="mkl_sequential",
        blas_root="/opt/intel/oneapi/mkl/latest",
    )

    definitions = lammps_cmake_definitions(
        manifest,
        prefix=tmp_path / "prefix",
        mpi_enabled=False,
        mpi_cxx="",
        lammps_source=_lammps_source(tmp_path / "lammps"),
    )

    assert definitions["BLA_VENDOR"] == "Intel10_64lp_seq"
    assert definitions["CMAKE_Fortran_COMPILER"] == "/usr/bin/gfortran"
    assert definitions["MKLROOT"] == "/opt/intel/oneapi/mkl/latest"
    assert definitions["CMAKE_INSTALL_RPATH"] == ("/opt/intel/oneapi/mkl/latest/lib")


@pytest.mark.parametrize(
    ("backend", "target", "distribution", "module"),
    (
        ("cpu", "x86-64-v3", "symmetrix-xl", "_native_cpu"),
        ("cuda", "sm120", "symmetrix-xl-cuda13-sm120", "_native_cuda13_sm120"),
        ("hip", "gfx1151", "symmetrix-xl-rocm7-gfx1151", "_native_rocm7_gfx1151"),
    ),
)
def test_backend_package_names_are_architecture_qualified(
    backend, target, distribution, module
):
    identity = package_identity(_manifest(backend, target))
    assert identity.distribution == distribution
    assert identity.module == module


def test_gpu_wheel_project_owns_no_frontend_files(tmp_path):
    manifest = _manifest("cuda", "sm120")
    invocation = python_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        operation="wheel",
        build_root=tmp_path / "build",
        wheel_directory=tmp_path / "dist",
    )
    project = invocation.build_directory / "backend-project"
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "symmetrix-xl-cuda13-sm120"
    assert metadata["project"]["readme"]["content-type"] == "text/plain"
    assert metadata["project"]["readme"]["text"].startswith(
        "Architecture-qualified Symmetrix native backend"
    )
    assert metadata["project"]["dependencies"] == [
        "symmetrix-xl==0.1.1",
        "nvidia-cuda-runtime>=13,<14",
        "nvidia-cuda-nvrtc>=13,<14",
        "nvidia-cublas>=13,<14",
    ]
    assert metadata["tool"]["scikit-build"]["wheel"]["packages"] == [
        "source/symmetrix_backend_cuda13_sm120"
    ]
    assert metadata["tool"]["scikit-build"]["wheel"]["exclude"] == [
        "lib/",
        "lib64/",
        "include/",
        "bin/",
    ]
    package = project / "source/symmetrix_backend_cuda13_sm120"
    descriptor = json.loads((package / "backend.json").read_text())
    assert descriptor["selector"] == "cuda13-sm120"
    assert "target_manifest" not in descriptor
    assert not (project / "source/symmetrix").exists()


def test_wheel_build_redacts_private_build_paths(tmp_path):
    invocation = python_build_invocation(
        _manifest("cuda", "sm120"),
        repo_root=REPOSITORY_ROOT,
        operation="wheel",
        build_root=tmp_path / "build",
        wheel_directory=tmp_path / "dist",
    )

    assert "--config-settings=cmake.define.SYMMETRIX_REDACT_BUILD_PATHS=ON" in (
        invocation.command
    )


def test_wheel_content_audit_accepts_release_payload(tmp_path):
    wheel = tmp_path / "safe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("symmetrix/__init__.py", '__version__ = "0.1.1"\n')
        archive.writestr("symmetrix/_native.so", b"/usr/src/symmetrix-xl/source.cpp")

    audit_wheel(wheel)


@pytest.mark.parametrize(
    ("member", "content", "message"),
    (
        ("lib64/libkokkoscore.a", b"archive", "forbidden archive component"),
        ("symmetrix/release_plan.md", b"text", "forbidden release-only filename"),
        ("symmetrix/_native.so", b"/home/alice/private/source.cpp", "Linux user home"),
        ("symmetrix/config.txt", b"pypi-abcdefghijklmnopqrstuvwxyz", "PyPI credential"),
    ),
)
def test_wheel_content_audit_rejects_private_payload(
    tmp_path, member, content, message
):
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, content)

    with pytest.raises(BuildError, match=message):
        audit_wheel(wheel)


def test_cuda12_wheel_declares_split_nvidia_dependencies(tmp_path):
    manifest = _manifest("cuda", "sm120")
    manifest = replace(
        manifest,
        toolchain=replace(manifest.toolchain, toolkit_version="12.9"),
    )
    invocation = python_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        operation="wheel",
        build_root=tmp_path / "build",
        wheel_directory=tmp_path / "dist",
    )
    metadata = tomllib.loads(
        (invocation.build_directory / "backend-project/pyproject.toml").read_text()
    )

    assert metadata["project"]["dependencies"] == [
        "symmetrix-xl==0.1.1",
        "nvidia-cuda-runtime-cu12>=12,<13",
        "nvidia-cuda-nvrtc-cu12>=12,<13",
        "nvidia-cublas-cu12>=12,<13",
    ]


def test_gpu_install_requires_exact_base_frontend():
    manifest = _manifest("cuda", "sm120")
    expected = build_python._frontend_version(REPOSITORY_ROOT / "symmetrix")
    script = (
        "import importlib.metadata; print(importlib.metadata.version('symmetrix-xl'))"
    )
    runner = FakeRunner(
        {},
        {
            (Path(manifest.python_executable).name, "-c", script): (
                0,
                f"{expected}\n",
                "",
            )
        },
    )
    assert (
        verify_frontend_installed(manifest, REPOSITORY_ROOT / "symmetrix", runner)
        == expected
    )

    runner.results[(Path(manifest.python_executable).name, "-c", script)] = (
        1,
        "",
        "missing",
    )
    with pytest.raises(BuildError, match="Install the CPU frontend first"):
        verify_frontend_installed(manifest, REPOSITORY_ROOT / "symmetrix", runner)


def test_hip_build_embeds_selected_toolkit_runtime_path():
    definitions = cmake_definitions(_manifest("hip", "gfx1151"))
    assert definitions["CMAKE_INSTALL_RPATH"] == "/opt/rocm/core-test/lib"


def test_python_invocation_uses_fingerprinted_build_directory(tmp_path):
    manifest = _manifest("hip", "gfx1151")
    invocation = python_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        operation="wheel",
        build_root=tmp_path / "build",
        wheel_directory=tmp_path / "dist",
        jobs=3,
    )

    assert invocation.build_directory.name == manifest.fingerprint
    assert invocation.manifest_path.is_file()
    assert invocation.environment["CMAKE_BUILD_PARALLEL_LEVEL"] == "3"
    assert any(
        value == "--config-settings=cmake.define.Kokkos_ARCH_AMD_GFX1100=ON"
        for value in invocation.command
    )


def test_python_install_uses_uv_for_the_selected_interpreter(tmp_path):
    manifest = _manifest("cpu", "native")
    invocation = python_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        operation="install",
        build_root=tmp_path / "build",
    )

    assert invocation.command[:5] == (
        "uv",
        "pip",
        "install",
        "--python",
        manifest.python_executable,
    )
    assert any(
        value == "--config-setting=cmake.define.Kokkos_ENABLE_OPENMP=ON"
        for value in invocation.command
    )
    assert "--no-deps" not in invocation.command


def test_build_directory_rejects_a_different_manifest(tmp_path):
    directory = tmp_path / "build"
    prepare_build_directory(directory, _manifest("cpu", "native"))

    with pytest.raises(BuildError, match="build directory contains target"):
        prepare_build_directory(directory, _manifest("hip", "gfx1151"))


def test_lammps_version_and_backend_specific_source_validation(tmp_path):
    source = _lammps_source(tmp_path / "lammps")
    assert parse_lammps_version((source / "src/version.h").read_text()) == (
        "10 Dec 2025",
        20251210,
    )
    assert validate_lammps_source(source)[0] == source
    with pytest.raises(BuildError, match="nvcc_wrapper"):
        validate_lammps_source(source, require_nvcc_wrapper=True)


def test_lammps_rejects_pre_namespace_fix_release(tmp_path):
    source = _lammps_source(tmp_path / "lammps", version="10 Sep 2025")

    with pytest.raises(BuildError, match="10 Dec 2025 or newer"):
        validate_lammps_source(source)


def test_mpi_wrapper_validation_preserves_basename_sensitive_symlink(tmp_path):
    implementation = Path(_executable(tmp_path / "opal_wrapper"))
    wrapper = tmp_path / "mpicxx"
    wrapper.symlink_to(implementation)
    runner = FakeRunner(
        {"mpicxx": str(wrapper)},
        {("mpicxx", "--showme:version"): (0, "Open MPI 4.1.7\n", "")},
    )

    enabled, selected, identity = build_lammps._mpi_configuration("auto", "", runner)

    assert enabled is True
    assert Path(selected).name == "mpicxx"
    assert Path(selected).is_symlink()
    assert identity == "Open MPI 4.1.7"


class MpiProbeRunner:
    def __init__(self, output, returncode=0):
        self.output = output
        self.returncode = returncode

    def which(self, _command):
        return None

    def run(self, args, **_kwargs):
        normalized = tuple(os.fspath(value) for value in args)
        if normalized[-1] in {"--showme:version", "--version"}:
            return CommandResult(normalized, 0, "Open MPI test provider\n", "")
        if "-o" in normalized:
            Path(normalized[-1]).write_text("probe executable\n")
            return CommandResult(normalized, 0, "", "")
        return CommandResult(normalized, self.returncode, self.output, "")


@pytest.mark.parametrize(
    ("backend", "accelerator", "method"),
    (
        ("cuda", "cuda", "MPIX_Query_cuda_support"),
        ("hip", "rocm", "MPIX_Query_rocm_support"),
    ),
)
def test_gpu_aware_mpi_probe_records_positive_provider_query(
    backend, accelerator, method
):
    runner = MpiProbeRunner(f"method={method} compile_time=1 runtime=1\n")

    evidence = probe_mpi_gpu_awareness(
        backend, "/opt/mpi/bin/mpicxx", REPOSITORY_ROOT, runner
    )

    assert evidence == {
        "required": True,
        "accelerator": accelerator,
        "method": method,
        "compile_time_support": True,
        "runtime_support": True,
        "status": "supported",
    }


def test_gpu_aware_mpi_probe_matches_lammps_runtime_query():
    runner = MpiProbeRunner("method=MPIX_Query_cuda_support compile_time=0 runtime=1\n")

    evidence = probe_mpi_gpu_awareness(
        "cuda", "/opt/mpi/bin/mpicxx", REPOSITORY_ROOT, runner
    )

    assert evidence["compile_time_support"] is False
    assert evidence["runtime_support"] is True
    assert evidence["status"] == "supported"


def test_gpu_aware_mpi_requirement_implies_mpi_and_records_evidence(tmp_path):
    wrapper = _executable(tmp_path / "mpicxx")
    runner = MpiProbeRunner("method=MPIX_Query_cuda_support compile_time=1 runtime=1\n")

    invocation = lammps_build_invocation(
        _manifest("cuda", "sm120"),
        repo_root=REPOSITORY_ROOT,
        source=_lammps_source(tmp_path / "lammps", with_wrapper=True),
        prefix=tmp_path / "prefix",
        mpi="auto",
        mpi_cxx=wrapper,
        require_gpu_aware_mpi=True,
        jobs=1,
        runner=runner,
        build_root=tmp_path / "build",
    )

    mpi = invocation.provenance["mpi"]
    assert mpi["enabled"] is True
    assert mpi["cxx"] == wrapper
    assert mpi["identity"] == "Open MPI test provider"
    assert mpi["gpu_aware"]["status"] == "supported"
    configure = invocation.commands[1]
    assert "-DBUILD_MPI=ON" in configure
    assert f"-DMPI_CXX_COMPILER={wrapper}" in configure


def test_gpu_aware_mpi_probe_rejects_host_only_provider():
    runner = MpiProbeRunner(
        "method=MPIX_Query_cuda_support compile_time=0 runtime=0\n",
        returncode=3,
    )

    with pytest.raises(BuildError, match="does not provide CUDA-aware MPI"):
        probe_mpi_gpu_awareness("cuda", "/opt/mpi/bin/mpicxx", REPOSITORY_ROOT, runner)


def test_gpu_aware_mpi_probe_rejects_wrong_provider_query():
    runner = MpiProbeRunner("method=MPIX_Query_rocm_support compile_time=1 runtime=1\n")

    with pytest.raises(BuildError, match="does not provide CUDA-aware MPI"):
        probe_mpi_gpu_awareness("cuda", "/opt/mpi/bin/mpicxx", REPOSITORY_ROOT, runner)


def test_gpu_aware_mpi_requirement_rejects_cpu_build(tmp_path):
    with pytest.raises(BuildError, match="only be required for CUDA or HIP"):
        lammps_build_invocation(
            _manifest(),
            repo_root=REPOSITORY_ROOT,
            source=_lammps_source(tmp_path / "lammps"),
            prefix=tmp_path / "prefix",
            mpi="auto",
            mpi_cxx=_executable(tmp_path / "mpicxx"),
            require_gpu_aware_mpi=True,
            jobs=1,
            runner=FakeRunner({}, {}),
            build_root=tmp_path / "build",
        )


def test_lammps_packages_and_extra_definitions_are_normalized_and_fingerprinted(
    tmp_path,
):
    source = _lammps_source(tmp_path / "lammps", packages=("KSPACE", "EXTRA-PAIR"))
    arguments = {
        "repo_root": REPOSITORY_ROOT,
        "source": source,
        "prefix": tmp_path / "prefix",
        "mpi": "off",
        "mpi_cxx": "",
        "jobs": 1,
        "runner": FakeRunner({}, {}),
        "build_root": tmp_path / "build",
    }

    invocation = lammps_build_invocation(
        _manifest(),
        **arguments,
        lammps_packages=("kspace", "PKG_EXTRA-PAIR", "KSPACE"),
        extra_cmake_definitions=("FFT=KISS", "BUILD_SHARED_LIBS=ON"),
    )
    baseline = lammps_build_invocation(_manifest(), **arguments)

    assert invocation.requested_packages == ("EXTRA-PAIR", "KSPACE")
    definitions = invocation.provenance["cmake_definitions"]
    assert definitions["PKG_EXTRA-PAIR"] == "ON"
    assert definitions["PKG_KSPACE"] == "ON"
    assert definitions["FFT"] == "KISS"
    assert definitions["BUILD_SHARED_LIBS"] == "ON"
    assert invocation.build_fingerprint != baseline.build_fingerprint


@pytest.mark.parametrize(
    ("packages", "definitions", "message"),
    (
        (("MISSING",), (), "package MISSING is not present"),
        ((), ("PKG_KSPACE=ON",), "PKG_KSPACE is managed"),
        ((), ("CMAKE_BUILD_TYPE=Debug",), "CMAKE_BUILD_TYPE is managed"),
        ((), ("Kokkos_ARCH_AMPERE80=ON",), "Kokkos_ARCH_AMPERE80 is managed"),
        ((), ("Kokkos_ENABLE_OPENMP=ON",), "Kokkos_ENABLE_OPENMP is managed"),
        ((), ("Kokkos_IMPL_AMDGPU_FLAGS=bad",), "Kokkos_IMPL_AMDGPU_FLAGS is managed"),
        ((), ("SYMMETRIX_HOST_ARCH=native",), "SYMMETRIX_HOST_ARCH is managed"),
        ((), ("FFT",), "expected NAME=VALUE"),
    ),
)
def test_lammps_rejects_invalid_package_configuration(
    tmp_path, packages, definitions, message
):
    with pytest.raises(BuildError, match=message):
        lammps_build_invocation(
            _manifest(),
            repo_root=REPOSITORY_ROOT,
            source=_lammps_source(tmp_path / "lammps"),
            prefix=tmp_path / "prefix",
            mpi="off",
            mpi_cxx="",
            jobs=1,
            runner=FakeRunner({}, {}),
            build_root=tmp_path / "build",
            lammps_packages=packages,
            extra_cmake_definitions=definitions,
        )


def test_lammps_source_identity_ignores_installer_block(tmp_path):
    source = _lammps_source(tmp_path / "lammps")
    runner = FakeRunner({}, {})
    before = lammps_source_provenance(source, "10 Dec 2025", runner)
    with (source / "cmake/CMakeLists.txt").open("a") as cmake:
        cmake.write(
            "\n\n# BEGIN SYMMETRIX PAIR STYLE\n"
            "target_link_libraries(lammps PRIVATE symmetrix)\n"
            "# END SYMMETRIX PAIR STYLE\n"
        )

    after = lammps_source_provenance(source, "10 Dec 2025", runner)

    assert after == before


def test_lammps_source_identity_includes_bundled_package_libraries(tmp_path):
    source = _lammps_source(tmp_path / "lammps")
    library = source / "lib/pace/ace.cpp"
    library.parent.mkdir(parents=True)
    library.write_text("int pace_version = 1;\n")
    runner = FakeRunner({}, {})
    before = lammps_source_provenance(source, "10 Dec 2025", runner)

    library.write_text("int pace_version = 2;\n")

    assert lammps_source_provenance(source, "10 Dec 2025", runner) != before


def test_lammps_dry_run_invocation_does_not_materialize_build_directory(tmp_path):
    build_root = tmp_path / "build"
    arguments = {
        "repo_root": REPOSITORY_ROOT,
        "source": _lammps_source(tmp_path / "lammps"),
        "prefix": tmp_path / "prefix",
        "mpi": "off",
        "mpi_cxx": "",
        "jobs": 1,
        "runner": FakeRunner({}, {}),
        "build_root": build_root,
        "materialize": False,
    }

    first = lammps_build_invocation(_manifest(), **arguments)
    second = lammps_build_invocation(_manifest(), **arguments)

    assert first.build_directory == second.build_directory
    assert not build_root.exists()


def test_lammps_dry_run_output_is_one_json_document(tmp_path, capsys):
    invocation = lammps_build_invocation(
        _manifest(),
        repo_root=REPOSITORY_ROOT,
        source=_lammps_source(tmp_path / "lammps"),
        prefix=tmp_path / "prefix",
        mpi="off",
        mpi_cxx="",
        jobs=1,
        runner=FakeRunner({}, {}),
        build_root=tmp_path / "build",
        materialize=False,
    )

    build_cli._print_lammps_invocation(invocation)

    output = json.loads(capsys.readouterr().out)
    assert output["build_directory"] == str(invocation.build_directory)
    assert output["build_fingerprint"] == invocation.build_fingerprint
    assert len(output["commands"]) == 4
    assert output["provenance"] == invocation.provenance


def test_symmetrix_source_identity_changes_with_build_source(tmp_path):
    source = tmp_path / "repo/libsymmetrix/source"
    source.mkdir(parents=True)
    implementation = source / "kernel.cpp"
    implementation.write_text("int value = 1;\n")
    before = symmetrix_source_fingerprint(tmp_path / "repo")

    implementation.write_text("int value = 2;\n")

    assert symmetrix_source_fingerprint(tmp_path / "repo") != before


def test_lammps_qualification_is_verified_and_reused(tmp_path):
    source = _lammps_source(tmp_path / "lammps")
    prefix = tmp_path / "prefix"
    runner = FakeRunner(
        {},
        {
            ("lmp", "-help"): (
                0,
                "Pair styles: symmetrix/mace symmetrix/mace/kk\n",
                "",
            )
        },
    )
    arguments = {
        "repo_root": REPOSITORY_ROOT,
        "source": source,
        "prefix": prefix,
        "mpi": "off",
        "mpi_cxx": "",
        "jobs": 1,
        "runner": runner,
        "build_root": tmp_path / "build",
    }
    invocation = lammps_build_invocation(_manifest(), **arguments)
    invocation.executable.parent.mkdir(parents=True)
    _executable(invocation.executable)
    _manifest().write(invocation.installed_manifest_path)
    record = build_lammps.verify_lammps(invocation, runner)
    invocation.qualification_path.write_text(json.dumps(record))

    replay = lammps_build_invocation(_manifest(), **arguments)
    reused = reuse_lammps_qualification(replay, runner)

    assert replay.reusable_qualification is True
    assert reused is not None
    assert reused["reused"] is True
    assert reused["build_fingerprint"] == invocation.build_fingerprint


def test_lammps_verification_reports_requested_packages(tmp_path):
    source = _lammps_source(tmp_path / "lammps", packages=("KSPACE", "EXTRA-PAIR"))
    runner = FakeRunner(
        {},
        {
            ("lmp", "-help"): (
                0,
                "Pair styles: symmetrix/mace symmetrix/mace/kk\n"
                "Installed packages:\n\n"
                "EXTRA-PAIR KOKKOS KSPACE\n\n",
                "",
            )
        },
    )
    invocation = lammps_build_invocation(
        _manifest(),
        repo_root=REPOSITORY_ROOT,
        source=source,
        prefix=tmp_path / "prefix",
        mpi="off",
        mpi_cxx="",
        jobs=1,
        runner=runner,
        build_root=tmp_path / "build",
        lammps_packages=("KSPACE", "EXTRA-PAIR"),
    )
    invocation.executable.parent.mkdir(parents=True)
    _executable(invocation.executable)
    _manifest().write(invocation.installed_manifest_path)

    record = build_lammps.verify_lammps(invocation, runner)

    assert record["lammps_packages"] == ["EXTRA-PAIR", "KOKKOS", "KSPACE"]


def test_lammps_stale_qualification_is_invalidated(tmp_path):
    source = _lammps_source(tmp_path / "lammps")
    prefix = tmp_path / "prefix"
    runner = FakeRunner(
        {},
        {
            ("lmp", "-help"): (
                0,
                "Pair styles: symmetrix/mace symmetrix/mace/kk\n",
                "",
            )
        },
    )
    arguments = {
        "repo_root": REPOSITORY_ROOT,
        "source": source,
        "prefix": prefix,
        "mpi": "off",
        "mpi_cxx": "",
        "jobs": 1,
        "runner": runner,
        "build_root": tmp_path / "build",
    }
    invocation = lammps_build_invocation(_manifest(), **arguments)
    invocation.executable.parent.mkdir(parents=True)
    _executable(invocation.executable)
    _manifest().write(invocation.installed_manifest_path)
    record = build_lammps.verify_lammps(invocation, runner)
    invocation.qualification_path.write_text(json.dumps(record))
    invocation.executable.write_text("changed binary\n")

    replay = lammps_build_invocation(_manifest(), **arguments)

    assert reuse_lammps_qualification(replay, runner) is None
    assert not replay.qualification_path.exists()


def test_lammps_failed_attempt_uses_a_fresh_directory(tmp_path):
    source = _lammps_source(tmp_path / "lammps")
    arguments = {
        "repo_root": REPOSITORY_ROOT,
        "source": source,
        "prefix": tmp_path / "prefix",
        "mpi": "off",
        "mpi_cxx": "",
        "jobs": 1,
        "runner": FakeRunner({}, {}),
        "build_root": tmp_path / "build",
    }
    first = lammps_build_invocation(_manifest(), **arguments)
    (first.build_directory / "CMakeCache.txt").write_text("failed cache\n")

    retry = lammps_build_invocation(_manifest(), **arguments)

    assert retry.build_directory != first.build_directory
    assert retry.build_directory.name.endswith("-attempt-001")


def test_lammps_build_invalidates_qualification_and_preserves_log(tmp_path):
    source = _lammps_source(tmp_path / "lammps")
    invocation = lammps_build_invocation(
        _manifest(),
        repo_root=REPOSITORY_ROOT,
        source=source,
        prefix=tmp_path / "prefix",
        mpi="off",
        mpi_cxx="",
        jobs=1,
        runner=FakeRunner({}, {}),
        build_root=tmp_path / "build",
    )
    invocation.qualification_path.write_text("{}\n")
    log = invocation.build_directory / "lammps-build.log"
    log.write_text("previous attempt\n")
    invocation.installed_manifest_path.parent.mkdir(parents=True)
    invocation = replace(invocation, commands=())

    run_lammps_build(invocation, FakeRunner({}, {}))

    assert not invocation.qualification_path.exists()
    assert log.read_text() == ""
    assert (
        invocation.build_directory / "lammps-build.previous-001.log"
    ).read_text() == "previous attempt\n"


def test_lammps_hip_configuration_matches_python_target(tmp_path):
    manifest = _manifest("hip", "gfx1151")
    source = _lammps_source(tmp_path / "lammps")
    definitions = lammps_cmake_definitions(
        manifest,
        prefix=tmp_path / "prefix",
        mpi_enabled=False,
        mpi_cxx="",
        lammps_source=source,
    )

    assert definitions["Kokkos_ARCH_AMD_GFX1100"] == "ON"
    assert definitions["Kokkos_IMPL_AMDGPU_FLAGS"] == "--offload-arch=gfx1151"


def test_build_frontend_is_not_part_of_installed_project():
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "symmetrix/pyproject.toml").read_text()
    )
    scripts = pyproject["project"].get("scripts", {})
    packages = pyproject["tool"]["scikit-build"]["wheel"]["packages"]

    assert "symmetrix-build" not in scripts
    assert "symmetrix_build" not in scripts
    assert packages == ["source/symmetrix"]
    assert not TOOLS_ROOT.is_relative_to(REPOSITORY_ROOT / "symmetrix")


def test_release_version_sources_agree():
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "symmetrix/pyproject.toml").read_text()
    )
    version = pyproject["project"]["version"]
    init_source = (
        REPOSITORY_ROOT / "symmetrix/source/symmetrix/__init__.py"
    ).read_text()
    cmake_source = (REPOSITORY_ROOT / "symmetrix/CMakeLists.txt").read_text()
    docs_source = (REPOSITORY_ROOT / "docs/conf.py").read_text()

    assert f'__version__ = "{version}"' in init_source
    assert f'set(SYMMETRIX_FRONTEND_VERSION "{version}" CACHE STRING' in cmake_source
    assert f'release = "{version}"' in docs_source


def test_build_isolation_does_not_install_a_different_cmake():
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "symmetrix/pyproject.toml").read_text()
    )

    assert not any(
        requirement.lower().split("[", maxsplit=1)[0].startswith("cmake")
        for requirement in pyproject["build-system"]["requires"]
    )


def test_python_build_prefers_manifest_cmake_on_path():
    manifest = _manifest()

    environment = build_environment(manifest)

    assert (
        Path(environment["PATH"].split(os.pathsep)[0])
        == Path(manifest.toolchain.cmake).parent
    )


@pytest.mark.parametrize(
    ("backend", "cpu_count", "expected"),
    (
        ("cpu", 32, 32),
        ("cuda", 1, 1),
        ("cuda", 8, 4),
        ("hip", 64, 16),
    ),
)
def test_python_build_default_jobs(monkeypatch, backend, cpu_count, expected):
    monkeypatch.setattr(build_cli.os, "cpu_count", lambda: cpu_count)

    assert build_cli._default_python_build_jobs(backend) == expected


def test_python_build_verifies_actual_cmake_against_manifest(tmp_path):
    manifest = _manifest()
    cmake = _executable(tmp_path / "cmake")
    manifest = replace(
        manifest,
        toolchain=replace(manifest.toolchain, cmake=cmake),
    )
    invocation = python_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        operation="install",
        build_root=tmp_path / "build",
    )
    (invocation.build_directory / "CMakeCache.txt").write_text(
        f"CMAKE_COMMAND:INTERNAL={manifest.toolchain.cmake}\n"
    )
    runner = FakeRunner({}, {("cmake", "--version"): (0, "cmake version 4.2.0\n", "")})

    record = verify_python_cmake_provenance(invocation, manifest, runner)

    assert record["cmake"] == str(Path(manifest.toolchain.cmake).resolve())
    assert record["cmake_version"] == manifest.toolchain.cmake_version
    assert (invocation.build_directory / "build-toolchain.json").is_file()


def test_python_build_rejects_ephemeral_isolation_cmake(tmp_path):
    manifest = _manifest()
    invocation = python_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        operation="install",
        build_root=tmp_path / "build",
    )
    overlay = tmp_path / "deleted-overlay/bin/cmake"
    (invocation.build_directory / "CMakeCache.txt").write_text(
        f"CMAKE_COMMAND:INTERNAL={overlay}\n"
    )

    with pytest.raises(BuildError, match="no longer exists"):
        verify_python_cmake_provenance(invocation, manifest, FakeRunner({}, {}))


def test_current_lammps_creator_aliases_are_real_types():
    header = (
        REPOSITORY_ROOT / "pair_symmetrix/pair_symmetrix_mace_kokkos.h"
    ).read_text()

    assert "using PairSymmetrixMACEKokkosDeviceDouble =" in header
    assert "using PairSymmetrixMACEKokkosHostDouble =" in header
    assert "using PairSymmetrixMACEKokkosDeviceFloat =" in header
    assert "using PairSymmetrixMACEKokkosHostFloat =" in header


def test_manifest_json_is_stable_and_versioned():
    value = json.loads(_manifest().to_json())
    assert value["schema_version"] == 1
    assert value["architecture"]["device_target"] == "native"


def test_active_python_path_does_not_resolve_out_of_virtualenv(tmp_path, monkeypatch):
    base = _executable(tmp_path / "python-base")
    environment_python = tmp_path / "venv/bin/python"
    environment_python.parent.mkdir(parents=True)
    environment_python.symlink_to(base)
    monkeypatch.setattr(build_detect.sys, "executable", str(environment_python))

    assert build_detect._python_executable() == str(environment_python.absolute())


def test_cpu_doctor_uses_stable_worker_placement(monkeypatch):
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    monkeypatch.delenv("OMP_PLACES", raising=False)

    environment = build_cli._doctor_environment(_manifest("cpu", "native"))

    assert environment["OMP_PROC_BIND"] == "spread"
    assert environment["OMP_PLACES"] == "threads"


def test_cpu_doctor_preserves_explicit_worker_placement(monkeypatch):
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.setenv("OMP_PLACES", "cores")

    environment = build_cli._doctor_environment(_manifest("cpu", "native"))

    assert environment["OMP_PROC_BIND"] == "close"
    assert environment["OMP_PLACES"] == "cores"


def test_gpu_doctor_does_not_inject_openmp_placement(monkeypatch):
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    monkeypatch.delenv("OMP_PLACES", raising=False)

    environment = build_cli._doctor_environment(_manifest("hip", "gfx1151"))

    assert "OMP_PROC_BIND" not in environment
    assert "OMP_PLACES" not in environment


def test_cross_target_accelerator_install_defers_runtime_qualification():
    manifest = replace(_manifest("cuda", "sm80"), visible_devices=("sm120",))

    reason = build_cli._runtime_qualification_reason(manifest)

    assert reason is not None
    assert "target sm80" in reason
    assert "sm120" in reason


def test_matching_accelerator_target_allows_runtime_qualification():
    assert build_cli._runtime_qualification_reason(_manifest("cuda", "sm80")) is None


def test_cpu_install_always_qualifies_its_runtime():
    assert build_cli._runtime_qualification_reason(_manifest("cpu", "native")) is None


def test_logged_command_mirrors_and_preserves_output(tmp_path, capsys):
    log = tmp_path / "command.log"
    result = CommandRunner().run_logged(
        (
            sys.executable,
            "-u",
            "-c",
            "import sys; print('standard'); print('diagnostic', file=sys.stderr)",
        ),
        log,
    )

    assert result.returncode == 0
    assert capsys.readouterr().out == "standard\ndiagnostic\n"
    assert log.read_text() == "standard\ndiagnostic\n"


@pytest.mark.parametrize(
    ("target", "valid"),
    (("x86-64-v4", True), ("x86-64-v3", True), ("native", True), ("x86-64-v5", False)),
)
def test_cpu_target_levels(target, valid):
    if valid:
        assert build_targets.cpu_architecture(target).device_target == target
    else:
        with pytest.raises(BuildError, match="invalid CPU target"):
            build_targets.cpu_architecture(target)


def test_cpu_v4_is_an_addon_backend_package():
    identity = package_identity(_manifest("cpu", "x86-64-v4"))

    assert identity.base is False
    assert identity.distribution == "symmetrix-xl-cpu-avx512"
    assert identity.package == "symmetrix_backend_cpu_avx512"
    assert identity.module == "_native_cpu_avx512"
    assert identity.selector == "cpu-avx512"
    assert identity.architecture == "x86-64-v4"

    assert package_identity(_manifest("cpu", "x86-64-v3")).base is True
    assert package_identity(_manifest("cpu", "native")).base is True


def test_cpu_v4_cmake_definitions_apply_host_target():
    definitions = cmake_definitions(_manifest("cpu", "x86-64-v4"))

    assert definitions["SYMMETRIX_HOST_ARCH"] == "x86-64-v4"
    assert definitions["Kokkos_ARCH_NATIVE"] == "OFF"


def test_cpu_v4_wheel_project_owns_no_frontend_files(tmp_path):
    manifest = _manifest("cpu", "x86-64-v4")
    invocation = python_build_invocation(
        manifest,
        repo_root=REPOSITORY_ROOT,
        operation="wheel",
        build_root=tmp_path / "build",
        wheel_directory=tmp_path / "dist",
    )
    project = invocation.build_directory / "backend-project"
    metadata = tomllib.loads((project / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "symmetrix-xl-cpu-avx512"
    assert metadata["project"]["dependencies"] == [
        f"symmetrix-xl=={build_python._frontend_version(REPOSITORY_ROOT / 'symmetrix')}"
    ]
    assert metadata["tool"]["scikit-build"]["wheel"]["packages"] == [
        "source/symmetrix_backend_cpu_avx512"
    ]
    descriptor = json.loads(
        (project / "source/symmetrix_backend_cpu_avx512/backend.json").read_text()
    )
    assert descriptor["selector"] == "cpu-avx512"
    assert descriptor["architecture"] == "x86-64-v4"
    assert descriptor["toolkit"] == ""
    assert not (project / "source/symmetrix").exists()


def test_matrix_spec_parsing_normalizes_targets():
    assert build_matrix.parse_matrix_spec("cuda:13:sm_120").normalized == (
        "cuda:13:sm120"
    )
    assert build_matrix.parse_matrix_spec("hip:7:gfx1151:sramecc+").normalized == (
        "hip:7:gfx1151"
    )
    assert build_matrix.parse_matrix_spec("cpu:x86-64-v4").normalized == ("cpu:avx512")
    assert build_matrix.parse_matrix_spec("cpu:avx512").normalized == "cpu:avx512"
    assert build_matrix.parse_matrix_spec("cpu:avx512").target == "x86-64-v4"
    assert build_matrix.parse_matrix_spec("cuda:12:sm70").normalized == ("cuda:12:sm70")


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("cpu:12:x86-64-v3", "cpu:<host-target>"),
        ("cuda:sm120", "cuda:<toolkit-major>:<device>"),
        ("cuda:thirteen:sm120", "major version"),
        ("cuda:13:sm99", "not supported by pinned Kokkos"),
        ("cuda:13:sm70", "use a CUDA 12 toolkit for Volta"),
        ("sycl:13:foo", "expected cpu, cuda, or hip"),
        ("cpu:x86-64-v5", "invalid CPU target"),
        ("cpu:x86-64-v3", "base symmetrix-xl"),
        ("cpu:native", "base symmetrix-xl"),
        ("cpu:none", "base symmetrix-xl"),
    ),
)
def test_matrix_spec_rejects_invalid_specs(value, message):
    with pytest.raises(BuildError, match=message):
        build_matrix.parse_matrix_spec(value)


def test_matrix_rejects_duplicate_normalized_targets():
    specs = [
        build_matrix.parse_matrix_spec("cuda:13:sm_120"),
        build_matrix.parse_matrix_spec("cuda:13:sm120"),
    ]

    with pytest.raises(BuildError, match="duplicate release target"):
        build_matrix.reject_duplicate_targets(specs)

    aliased = [
        build_matrix.parse_matrix_spec("cpu:avx512"),
        build_matrix.parse_matrix_spec("cpu:x86-64-v4"),
    ]

    with pytest.raises(BuildError, match="duplicate release target"):
        build_matrix.reject_duplicate_targets(aliased)


def test_matrix_child_commands_forward_target_selection(tmp_path):
    spec = build_matrix.parse_matrix_spec("cuda:12:sm86")
    detect_command, wheel_command = build_matrix.child_commands(
        spec,
        script=tmp_path / "symmetrix_build.py",
        wheel_directory=tmp_path / "dist",
        cuda_root=Path("/opt/cuda-12"),
        build_root=tmp_path / "build",
        jobs=4,
        accelerator_blas_root=Path("/opt/openblas-static"),
    )

    for command in (detect_command, wheel_command):
        assert command[1:3] == (str(tmp_path / "symmetrix_build.py"), "detect") or (
            command[1:3] == (str(tmp_path / "symmetrix_build.py"), "wheel")
        )
        assert command[command.index("--backend") + 1] == "cuda"
        assert command[command.index("--arch") + 1] == "sm86"
        assert command[command.index("--cuda-root") + 1] == "/opt/cuda-12"
        assert command[command.index("--accelerator-blas-root") + 1] == (
            "/opt/openblas-static"
        )
    assert detect_command[detect_command.index("--json") - 1] == "detect"
    assert "--build-root" not in detect_command
    assert "--jobs" not in detect_command
    assert wheel_command[wheel_command.index("--build-root") + 1] == str(
        tmp_path / "build"
    )
    assert wheel_command[wheel_command.index("--jobs") + 1] == "4"
    assert wheel_command[wheel_command.index("--wheel-dir") + 1] == str(
        tmp_path / "dist"
    )


def test_matrix_toolkit_mismatch_is_rejected():
    spec = build_matrix.parse_matrix_spec("cuda:12:sm80")

    with pytest.raises(BuildError, match="requires toolkit major 12"):
        build_matrix.verify_matrix_target(spec, _manifest("cuda", "sm80"))


def _fake_backend_wheel(tmp_path, manifest, descriptor):
    identity = package_identity(manifest)
    wheel = (
        tmp_path / f"{identity.distribution.replace('-', '_')}-0.1.1-py3-none-any.whl"
    )
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{identity.package}/backend.json", json.dumps(descriptor, sort_keys=True)
        )
    return wheel


def test_matrix_wheel_descriptor_verification(tmp_path):
    manifest = _manifest("cuda", "sm120")
    identity = package_identity(manifest)
    descriptor = {
        "schema_version": 1,
        "selector": identity.selector,
        "backend": identity.backend,
        "architecture": identity.architecture,
        "distribution": identity.distribution,
        "frontend_version": "0.1.1",
        "native_abi": 1,
        "package": identity.package,
        "module": identity.module,
        "toolkit": identity.toolkit,
    }
    wheel = _fake_backend_wheel(tmp_path, manifest, descriptor)

    assert build_matrix.verify_wheel_descriptor(wheel, manifest)["selector"] == (
        identity.selector
    )

    descriptor["distribution"] = "symmetrix-xl-cuda12-sm120"
    mismatched = _fake_backend_wheel(tmp_path / "other", manifest, descriptor)

    with pytest.raises(BuildError, match="descriptor field 'distribution'"):
        build_matrix.verify_wheel_descriptor(mismatched, manifest)


def test_matrix_base_wheel_descriptor_verification(tmp_path):
    manifest = _manifest("cpu", "x86-64-v3")
    descriptor = {
        "schema_version": 1,
        "selector": "cpu",
        "backend": "cpu",
        "architecture": "x86-64-v3",
        "distribution": "symmetrix-xl",
        "frontend_version": "0.1.1",
        "native_abi": 1,
        "package": "symmetrix",
        "module": "_native_cpu",
        "toolkit": "",
    }
    wheel = tmp_path / "symmetrix_xl-0.1.1-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "symmetrix/_backend_cpu.json", json.dumps(descriptor, sort_keys=True)
        )

    assert build_matrix.verify_wheel_descriptor(wheel, manifest)["selector"] == "cpu"


def test_sdist_submodule_pins_are_commits():
    pinned = build_sdist.submodule_pinned_commits(REPOSITORY_ROOT, CommandRunner())

    assert set(pinned) == set(build_sdist.SUBMODULE_PATHS)
    assert all(len(commit) == 40 for commit in pinned.values())


def _init_git_repo(runner, path):
    runner.run(("git", "init", "-q", str(path)), check=True)
    runner.run(
        (
            "git",
            "-C",
            str(path),
            "config",
            "user.name",
            "symmetrix-tests",
        ),
        check=True,
    )
    runner.run(
        ("git", "-C", str(path), "config", "user.email", "tests@symmetrix.invalid"),
        check=True,
    )


def _commit_all(runner, path, message):
    runner.run(("git", "-C", str(path), "add", "-A"), check=True)
    runner.run(("git", "-C", str(path), "commit", "-q", "-m", message), check=True)


def test_sdist_staging_vendors_pinned_submodule_commits(tmp_path):
    runner = CommandRunner()
    repository = tmp_path / "repo"
    (repository / "symmetrix").mkdir(parents=True)
    (repository / "symmetrix/pyproject.toml").write_text("[project]\n")
    (repository / "libsymmetrix").mkdir()
    (repository / "libsymmetrix/CMakeLists.txt").write_text("")
    (repository / "tools").mkdir()
    (repository / "tools/helper.py").write_text("")
    (repository / "LICENSE").write_text("MIT\n")
    (repository / "README.md").write_text("readme\n")
    _init_git_repo(runner, repository)
    _commit_all(runner, repository, "base")

    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "LICENSE").write_text("BSD-3\n")
    (dependency / "source.cpp").write_text("int value = 1;\n")
    _init_git_repo(runner, dependency)
    _commit_all(runner, dependency, "vendored")
    pinned_commit = build_sdist.head_commit(dependency, runner)

    submodule_path = "libsymmetrix/external/kokkos"
    runner.run(
        ("git", "clone", "-q", str(dependency), str(repository / submodule_path)),
        check=True,
    )
    runner.run(
        (
            "git",
            "-C",
            str(repository),
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{pinned_commit},{submodule_path}",
        ),
        check=True,
    )
    runner.run(
        ("git", "-C", str(repository), "commit", "-q", "-m", "pin dependency"),
        check=True,
    )

    staging = tmp_path / "staging"
    pinned = build_sdist.prepare_sdist_staging(
        repository,
        staging,
        runner,
        paths=(submodule_path,),
    )

    assert pinned == {submodule_path: pinned_commit}
    assert (staging / "symmetrix/pyproject.toml").is_file()
    assert (staging / "libsymmetrix/CMakeLists.txt").is_file()
    assert (staging / "libsymmetrix/external/kokkos/LICENSE").read_text() == "BSD-3\n"
    assert (staging / "libsymmetrix/external/kokkos/source.cpp").is_file()
    assert not list(staging.rglob(".git*"))


def test_sdist_license_gate_requires_license_text(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good/LICENSE").write_text("MIT\n")
    (tmp_path / "bad").mkdir()

    assert build_sdist.verify_vendored_licenses(tmp_path, paths=("good",)) == {
        "good": "LICENSE"
    }
    with pytest.raises(BuildError, match="license text"):
        build_sdist.verify_vendored_licenses(tmp_path, paths=("bad",))


def test_sdist_license_gate_accepts_cblas_source_notice(tmp_path):
    path = "libsymmetrix/external/cblas-prototypes"
    dependency = tmp_path / path
    dependency.mkdir(parents=True)
    (dependency / "README.md").write_text("Netlib cblas.h source notice\n")

    assert build_sdist.verify_vendored_licenses(tmp_path, paths=(path,)) == {
        path: "README.md"
    }


def _write_sdist(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name in members:
            payload = b"placeholder\n"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_sdist_contents_verification(tmp_path):
    root = "symmetrix_xl-0.1.1"
    members = [
        *(f"{root}/{member}" for member in build_sdist.REQUIRED_SDIST_MEMBERS),
        f"{root}/VENDORED_DEPS.json",
        f"{root}/libsymmetrix/external/cblas-prototypes/include/cblas.h",
        f"{root}/external/pybind11/.gitattributes",
    ]
    complete = tmp_path / "complete.tar.gz"
    _write_sdist(complete, members)

    record = build_sdist.verify_sdist_contents(complete, "0.1.1")

    assert record["filename"] == "complete.tar.gz"
    assert record["members"] == len(members)
    assert len(record["sha256"]) == 64

    incomplete = [
        name for name in members if not name.endswith("kokkos/CMakeLists.txt")
    ]
    missing = tmp_path / "missing.tar.gz"
    _write_sdist(missing, incomplete)
    with pytest.raises(BuildError, match="not self-contained"):
        build_sdist.verify_sdist_contents(missing, "0.1.1")

    polluted = tmp_path / "polluted.tar.gz"
    _write_sdist(
        polluted, [*members, f"{root}/libsymmetrix/external/kokkos/.git/config"]
    )
    with pytest.raises(BuildError, match="VCS metadata"):
        build_sdist.verify_sdist_contents(polluted, "0.1.1")


def test_sdist_configuration_is_self_contained():
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "symmetrix/pyproject.toml").read_text()
    )
    sdist_options = pyproject["tool"]["scikit-build"]["sdist"]
    assert sdist_options["force-include"] == {"libsymmetrix": "libsymmetrix"}
    # scikit-build-core rejects underscore option names at build time.
    assert "force_include" not in sdist_options
    assert pyproject["project"]["license"] == {"file": "LICENSE"}
    assert (REPOSITORY_ROOT / "symmetrix/LICENSE").read_text() == (
        REPOSITORY_ROOT / "LICENSE"
    ).read_text()

    source = (REPOSITORY_ROOT / "symmetrix/CMakeLists.txt").read_text()
    assert (
        'if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../libsymmetrix/CMakeLists.txt")'
        in (source)
    )
    assert (
        'set(SYMMETRIX_LIBSYM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libsymmetrix")'
        in (source)
    )


def test_sdist_staging_relocates_core_inside_project(tmp_path):
    staged_tree = tmp_path / "tree"
    core = staged_tree / "libsymmetrix"
    project = staged_tree / "symmetrix"
    core.mkdir(parents=True)
    project.mkdir()
    (core / "CMakeLists.txt").write_text("project(libsymmetrix)\n")

    build_sdist.make_sdist_tree_self_contained(staged_tree)

    assert not core.exists()
    assert (project / "libsymmetrix/CMakeLists.txt").read_text() == (
        "project(libsymmetrix)\n"
    )

    with pytest.raises(BuildError, match="missing"):
        build_sdist.make_sdist_tree_self_contained(staged_tree)


def test_sdist_project_license_copy_gate(tmp_path):
    repository = tmp_path / "repo"
    (repository / "symmetrix").mkdir(parents=True)
    (repository / "LICENSE").write_text("MIT\n")

    with pytest.raises(BuildError, match="symmetrix/LICENSE"):
        build_sdist.verify_project_license_copy(repository)

    (repository / "symmetrix/LICENSE").write_text("MIT\n")
    build_sdist.verify_project_license_copy(repository)

    (repository / "symmetrix/LICENSE").write_text("divergent\n")
    with pytest.raises(BuildError, match="differs"):
        build_sdist.verify_project_license_copy(repository)


def test_sdist_rejects_dirty_worktree(tmp_path):
    dirty = FakeRunner(
        {},
        {
            ("git", "-C", str(tmp_path), "status", "--porcelain"): (
                0,
                " M symmetrix/pyproject.toml\n",
                "",
            )
        },
    )
    with pytest.raises(BuildError, match="uncommitted changes"):
        build_sdist.verify_clean_worktree(tmp_path, dirty)

    clean = FakeRunner(
        {}, {("git", "-C", str(tmp_path), "status", "--porcelain"): (0, "", "")}
    )
    build_sdist.verify_clean_worktree(tmp_path, clean)


def test_sdist_smoke_check_rebuilds_from_itself(tmp_path):
    root = "symmetrix_xl-0.1.1"
    sdist = tmp_path / f"{root}.tar.gz"
    _write_sdist(sdist, [f"{root}/pyproject.toml", f"{root}/LICENSE"])
    rebuild_args = (
        Path(sys.executable).name,
        "-m",
        "build",
        "--sdist",
        "--outdir",
        str(tmp_path / "smoke-output"),
        str(tmp_path / "smoke-extracted" / root),
    )

    healthy = FakeRunner({}, {rebuild_args: (0, "", "")})
    build_sdist.verify_sdist_rebuilds_from_itself(
        sdist, sys.executable, healthy, tmp_path, "0.1.1"
    )
    assert (tmp_path / "smoke-extracted" / root / "pyproject.toml").is_file()

    broken = FakeRunner(
        {}, {rebuild_args: (1, "", "License file not found ('../LICENSE')\n")}
    )
    with pytest.raises(BuildError, match="cannot be rebuilt from"):
        build_sdist.verify_sdist_rebuilds_from_itself(
            sdist, sys.executable, broken, tmp_path, "0.1.1"
        )
    assert "License file not found" in (tmp_path / "sdist-smoke.log").read_text()
