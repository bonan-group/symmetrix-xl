"""Python package build orchestration from a resolved target manifest."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .command import BuildError, CommandRunner
from .manifest import TargetManifest
from .package_identity import PackageIdentity, package_identity


@dataclass(frozen=True)
class BuildInvocation:
    command: tuple[str, ...]
    environment: dict[str, str]
    build_directory: Path
    manifest_path: Path


def cmake_definitions(manifest: TargetManifest) -> dict[str, str]:
    identity = package_identity(manifest)
    definitions = {
        "CMAKE_BUILD_TYPE": "Release",
        "Python_EXECUTABLE": manifest.python_executable,
        "PYTHON_EXECUTABLE": manifest.python_executable,
        "SYMMETRIX_KOKKOS": "ON",
        "SYMMETRIX_PYTHON_MODULE_NAME": identity.module,
        "SYMMETRIX_PYTHON_INSTALL_PACKAGE": identity.package,
        "SYMMETRIX_DISTRIBUTION_NAME": identity.distribution,
        "SYMMETRIX_BACKEND_SELECTOR": identity.backend,
        "SYMMETRIX_BACKEND_ARCHITECTURE": identity.architecture,
    }
    backend = manifest.backend
    if backend == "cpu":
        definitions.update(
            {
                "SYMMETRIX_DEVICE_BACKEND": "NONE",
                "Kokkos_ENABLE_CUDA": "OFF",
                "Kokkos_ENABLE_HIP": "OFF",
                "Kokkos_ENABLE_OPENMP": "ON",
                "Kokkos_ENABLE_SERIAL": "OFF",
                "KokkosKernels_ENABLE_TPL_BLAS": "ON",
                "BLA_VENDOR": (
                    "Intel10_64lp_seq"
                    if manifest.blas_policy == "mkl_sequential"
                    else "Intel10_64lp"
                    if manifest.blas_policy == "mkl_gnu_thread"
                    else "OpenBLAS"
                ),
                "SYMMETRIX_HOST_ARCH": manifest.host_target,
                "Kokkos_ARCH_NATIVE": (
                    "ON" if manifest.host_target == "native" else "OFF"
                ),
                "SYMMETRIX_SPHERICART_CUDA": "OFF",
                "SPHERICART_ENABLE_CUDA": "OFF",
            }
        )
        if manifest.toolchain.fortran:
            definitions["CMAKE_Fortran_COMPILER"] = manifest.toolchain.fortran
        if manifest.blas_root:
            runtime_path = str(Path(manifest.blas_root) / "lib")
            definitions["MKLROOT"] = manifest.blas_root
            definitions["CMAKE_BUILD_RPATH"] = runtime_path
            definitions["CMAKE_INSTALL_RPATH"] = runtime_path
    elif backend == "cuda":
        build_rpath = ";".join(manifest.toolchain.runtime_library_dirs)
        install_rpath = (
            "$ORIGIN/../nvidia/cu13/lib"
            if identity.toolkit == "cuda13"
            else ";".join(
                (
                    "$ORIGIN/../nvidia/cuda_runtime/lib",
                    "$ORIGIN/../nvidia/cublas/lib",
                )
            )
        )
        definitions.update(
            {
                "CMAKE_CXX_COMPILER": manifest.toolchain.cxx,
                "CMAKE_CUDA_HOST_COMPILER": manifest.toolchain.host_cxx,
                "SYMMETRIX_DEVICE_BACKEND": "CUDA",
                "Kokkos_ENABLE_CUDA": "ON",
                "Kokkos_ENABLE_HIP": "OFF",
                "Kokkos_ENABLE_OPENMP": "OFF",
                "Kokkos_ENABLE_SERIAL": "ON",
                "Kokkos_ARCH_NATIVE": "OFF",
                "KokkosKernels_ENABLE_TPL_CUSPARSE": "OFF",
                "KokkosKernels_ENABLE_TPL_CUSOLVER": "OFF",
                f"Kokkos_ARCH_{manifest.architecture.kokkos_trait}": "ON",
                "SYMMETRIX_HOST_ARCH": "none",
                "SYMMETRIX_SPHERICART_CUDA": "ON",
                "SPHERICART_ENABLE_CUDA": "ON",
                "SPHERICART_OPENMP": "OFF",
                "CMAKE_BUILD_RPATH": build_rpath,
                "CMAKE_INSTALL_RPATH": install_rpath,
            }
        )
    elif backend == "hip":
        target = manifest.architecture.compiler_target
        runtime_path = ";".join(manifest.toolchain.runtime_library_dirs)
        definitions.update(
            {
                "CMAKE_CXX_COMPILER": manifest.toolchain.cxx,
                "CMAKE_PREFIX_PATH": manifest.toolchain.toolkit_root,
                "CMAKE_BUILD_RPATH": runtime_path,
                "CMAKE_INSTALL_RPATH": runtime_path,
                "SYMMETRIX_DEVICE_BACKEND": "HIP",
                "Kokkos_ENABLE_CUDA": "OFF",
                "Kokkos_ENABLE_HIP": "ON",
                "Kokkos_ENABLE_OPENMP": "OFF",
                "Kokkos_ENABLE_SERIAL": "ON",
                "Kokkos_ARCH_NATIVE": "OFF",
                f"Kokkos_ARCH_{manifest.architecture.kokkos_trait}": "ON",
                "Kokkos_IMPL_AMDGPU_FLAGS": f"--offload-arch={target}",
                "Kokkos_IMPL_AMDGPU_LINK": f"--offload-arch={target}",
                "SYMMETRIX_HOST_ARCH": "none",
                "SYMMETRIX_SPHERICART_CUDA": "OFF",
                "SPHERICART_ENABLE_CUDA": "OFF",
                "SPHERICART_OPENMP": "OFF",
                "SYMMETRIX_HIP_BLAS": "AUTO",
            }
        )
    else:
        raise BuildError(f"unsupported manifest backend: {backend}")
    if manifest.blas_policy == "openblas_static":
        root = Path(manifest.blas_root)
        library = next(
            (
                path
                for path in (root / "lib/libopenblas.a", root / "lib64/libopenblas.a")
                if path.is_file()
            ),
            None,
        )
        include = next(
            (
                path.parent
                for path in (
                    root / "include/cblas.h",
                    root / "include/openblas/cblas.h",
                )
                if path.is_file()
            ),
            None,
        )
        if library is None or include is None:
            raise BuildError(
                f"static OpenBLAS installation is incomplete under {manifest.blas_root}"
            )
        definitions.update(
            {
                "SYMMETRIX_BLAS_LIBRARY": str(library),
                "SYMMETRIX_BLAS_INCLUDE_DIR": str(include),
                "SYMMETRIX_HIDE_STATIC_BLAS_SYMBOLS": "ON",
            }
        )
    return definitions


def _prepend_path(environment: dict[str, str], value: Path) -> None:
    existing = environment.get("PATH", "")
    environment["PATH"] = str(value) + (os.pathsep + existing if existing else "")


def build_environment(
    manifest: TargetManifest, *, jobs: int | None = None
) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CXX"] = manifest.toolchain.cxx
    environment["CMAKE_GENERATOR"] = manifest.toolchain.generator
    if jobs is not None:
        if jobs < 1:
            raise BuildError("--jobs must be positive")
        environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    if manifest.backend == "cuda":
        environment["NVCC_WRAPPER_DEFAULT_COMPILER"] = manifest.toolchain.host_cxx
        _prepend_path(environment, Path(manifest.toolchain.toolkit_root) / "bin")
    elif manifest.backend == "hip":
        _prepend_path(environment, Path(manifest.toolchain.toolkit_root) / "bin")
        previous = environment.get("CMAKE_PREFIX_PATH", "")
        environment["CMAKE_PREFIX_PATH"] = manifest.toolchain.toolkit_root + (
            os.pathsep + previous if previous else ""
        )
    elif manifest.backend == "cpu" and manifest.blas_root:
        environment["MKLROOT"] = manifest.blas_root
        runtime_path = str(Path(manifest.blas_root) / "lib")
        previous = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = runtime_path + (
            os.pathsep + previous if previous else ""
        )
    _prepend_path(environment, Path(manifest.toolchain.cmake).parent)
    return environment


def default_build_directory(repo_root: Path, manifest: TargetManifest) -> Path:
    return repo_root / "symmetrix" / f"build-{manifest.fingerprint}"


def prepare_build_directory(build_directory: Path, manifest: TargetManifest) -> Path:
    build_directory = build_directory.resolve()
    manifest_path = build_directory / "symmetrix-target.json"
    if build_directory.exists():
        if manifest_path.is_file():
            existing = TargetManifest.read(manifest_path)
            if existing.fingerprint != manifest.fingerprint:
                raise BuildError(
                    f"build directory contains target {existing.fingerprint}, "
                    f"not {manifest.fingerprint}: {build_directory}"
                )
        elif any(build_directory.iterdir()):
            raise BuildError(
                "refusing to use a non-empty build directory without a "
                f"Symmetrix target manifest: {build_directory}"
            )
    build_directory.mkdir(parents=True, exist_ok=True)
    manifest.write(manifest_path)
    return manifest_path


def _frontend_version(project: Path) -> str:
    text = (project / "pyproject.toml").read_text()
    project_section = text.partition("[project]")[2].partition("[")[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project_section, re.MULTILINE)
    if match is None:
        raise BuildError("Symmetrix pyproject.toml has no static project version")
    return match.group(1)


def verify_frontend_installed(
    manifest: TargetManifest, source_project: Path, runner: CommandRunner
) -> str:
    """Require the matching base distribution before adding a GPU backend."""

    expected = _frontend_version(source_project)
    script = (
        "import importlib.metadata; print(importlib.metadata.version('symmetrix-xl'))"
    )
    result = runner.run((manifest.python_executable, "-c", script))
    actual = result.stdout.strip() if result.returncode == 0 else ""
    if actual != expected:
        detail = actual or "not installed"
        raise BuildError(
            f"GPU backend installation requires symmetrix-xl=={expected}; found {detail}. "
            "Install the CPU frontend first with "
            "'python tools/symmetrix_build.py install --backend cpu'."
        )
    return actual


def _backend_dependencies(identity: PackageIdentity, version: str) -> list[str]:
    dependencies = [f"symmetrix-xl=={version}"]
    if identity.toolkit == "cuda12":
        dependencies.extend(
            (
                "nvidia-cuda-runtime-cu12>=12,<13",
                "nvidia-cuda-nvrtc-cu12>=12,<13",
                "nvidia-cublas-cu12>=12,<13",
            )
        )
    elif identity.toolkit == "cuda13":
        dependencies.extend(
            (
                "nvidia-cuda-runtime>=13,<14",
                "nvidia-cuda-nvrtc>=13,<14",
                "nvidia-cublas>=13,<14",
            )
        )
    return dependencies


def prepare_backend_project(
    build_directory: Path,
    source_project: Path,
    manifest: TargetManifest,
) -> Path:
    """Create a native-only project whose installed files cannot overlap frontend files."""

    identity = package_identity(manifest)
    if identity.base:
        return source_project
    project = build_directory / "backend-project"
    package = project / "source" / identity.package
    package.mkdir(parents=True, exist_ok=True)
    version = _frontend_version(source_project)
    descriptor = {
        "schema_version": 1,
        "selector": identity.selector,
        "backend": identity.backend,
        "architecture": identity.architecture,
        "distribution": identity.distribution,
        "frontend_version": version,
        "native_abi": 1,
        "package": identity.package,
        "module": identity.module,
        "toolkit": identity.toolkit,
        "compiler": manifest.toolchain.cxx_version,
        "target_manifest_fingerprint": manifest.fingerprint,
    }
    (package / "__init__.py").write_text(
        '"""Architecture-qualified Symmetrix native backend."""\n'
    )
    (package / "backend.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    )
    dependencies = ",\n    ".join(
        json.dumps(value) for value in _backend_dependencies(identity, version)
    )
    pyproject = f"""[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.version = ">=3.27"
cmake.source-dir = {json.dumps(str(source_project))}
wheel.exclude = ["lib/", "lib64/", "include/", "bin/"]
wheel.packages = ["source/{identity.package}"]

[project]
name = {json.dumps(identity.distribution)}
version = {json.dumps(version)}
description = "Symmetrix native backend for {identity.selector}"
readme = {{ text = "Architecture-qualified Symmetrix native backend for {identity.selector}.", content-type = "text/plain" }}
requires-python = ">=3.10"
dependencies = [
    {dependencies}
]

[project.entry-points."symmetrix.backends"]
{identity.selector} = {json.dumps(identity.package)}
"""
    (project / "pyproject.toml").write_text(pyproject)
    return project


def python_build_invocation(
    manifest: TargetManifest,
    *,
    repo_root: str | Path,
    operation: str,
    build_root: str | Path | None = None,
    wheel_directory: str | Path | None = None,
    jobs: int | None = None,
) -> BuildInvocation:
    if operation not in {"install", "wheel"}:
        raise BuildError(f"unsupported Python build operation: {operation}")
    root = Path(repo_root).resolve()
    source_project = root / "symmetrix"
    if not (source_project / "pyproject.toml").is_file():
        raise BuildError(f"Symmetrix Python project is missing: {source_project}")
    if build_root is None:
        build_directory = default_build_directory(root, manifest)
    else:
        build_directory = Path(build_root).expanduser().resolve() / manifest.fingerprint
    manifest_path = prepare_build_directory(build_directory, manifest)
    project = prepare_backend_project(build_directory, source_project, manifest)
    definitions = cmake_definitions(manifest)
    if operation == "wheel":
        definitions["SYMMETRIX_REDACT_BUILD_PATHS"] = "ON"

    if operation == "install":
        # Use uv's shared cache while explicitly targeting the selected Python
        # environment. The wheel command below remains pip because uv pip does
        # not provide a wheel-build subcommand.
        command = ["uv", "pip", "install", "--python", manifest.python_executable]
        command.append("--verbose")
        command.append(str(project))
        command.append(f"--config-setting=build-dir={build_directory}")
        for name, value in definitions.items():
            command.append(f"--config-setting=cmake.define.{name}={value}")
    else:
        if wheel_directory is None:
            raise BuildError("wheel operation requires --wheel-dir")
        destination = Path(wheel_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        command = [manifest.python_executable, "-m", "pip"]
        command.extend(
            (
                "wheel",
                "--verbose",
                "--no-deps",
                "--wheel-dir",
                str(destination),
                str(project),
            )
        )
        command.append(f"--config-settings=build-dir={build_directory}")
        for name, value in definitions.items():
            command.append(f"--config-settings=cmake.define.{name}={value}")

    record = {
        "manifest_fingerprint": manifest.fingerprint,
        "operation": operation,
        "command": command,
        "environment": {
            name: value
            for name, value in build_environment(manifest, jobs=jobs).items()
            if name
            in {
                "CXX",
                "CMAKE_BUILD_PARALLEL_LEVEL",
                "CMAKE_GENERATOR",
                "CMAKE_PREFIX_PATH",
                "NVCC_WRAPPER_DEFAULT_COMPILER",
                "PATH",
            }
        },
    }
    (build_directory / "invocation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return BuildInvocation(
        tuple(command),
        build_environment(manifest, jobs=jobs),
        build_directory,
        manifest_path,
    )


def run_python_build(invocation: BuildInvocation, runner: CommandRunner) -> None:
    log_path = invocation.build_directory / "build.log"
    result = runner.run_logged(
        invocation.command,
        log_path,
        env=invocation.environment,
    )
    if result.returncode != 0:
        raise BuildError(f"Python build failed ({result.returncode}); see {log_path}")


def verify_python_cmake_provenance(
    invocation: BuildInvocation,
    manifest: TargetManifest,
    runner: CommandRunner,
) -> dict[str, str]:
    cache_path = invocation.build_directory / "CMakeCache.txt"
    if not cache_path.is_file():
        raise BuildError(f"Python build did not produce a CMake cache: {cache_path}")
    match = re.search(
        r"^CMAKE_COMMAND:INTERNAL=(.+)$",
        cache_path.read_text(),
        flags=re.MULTILINE,
    )
    if match is None:
        raise BuildError(f"CMake cache does not record CMAKE_COMMAND: {cache_path}")
    actual = Path(match.group(1))
    if not actual.is_file():
        raise BuildError(
            "the CMake executable used by pip build isolation no longer exists: "
            f"{actual}"
        )
    expected = Path(manifest.toolchain.cmake)
    if actual.resolve() != expected.resolve():
        raise BuildError(
            "pip build isolation used a different CMake than the target manifest: "
            f"{actual} instead of {expected}"
        )
    result = runner.run((str(actual), "--version"), check=True)
    version_match = re.search(r"cmake version\s+([0-9]+(?:[.][0-9]+)+)", result.output)
    actual_version = version_match.group(1) if version_match else ""
    if actual_version != manifest.toolchain.cmake_version:
        raise BuildError(
            "pip build isolation used CMake version "
            f"{actual_version or 'unknown'}, but the target manifest records "
            f"{manifest.toolchain.cmake_version}"
        )
    record = {
        "cmake": str(actual.resolve()),
        "cmake_version": actual_version,
        "cache": str(cache_path),
    }
    (invocation.build_directory / "build-toolchain.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def verify_installed_backend(
    manifest: TargetManifest, runner: CommandRunner
) -> dict[str, object]:
    identity = package_identity(manifest)
    script = (
        "import hashlib,json,pathlib,symmetrix; "
        "from symmetrix import symmetrix as native; "
        "path=pathlib.Path(native.__file__).resolve(); "
        "native._init_kokkos(); "
        "environment=getattr(native,'_execution_device_execution_environment',"
        "lambda:{})(); "
        "sentinel=getattr(native,'_kokkos_device_sentinel',lambda:None)(); "
        "print(json.dumps({'package':symmetrix.__file__,'extension':str(path),"
        "'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),"
        "'selected_backend':symmetrix.selected_backend(),"
        "'build_info':getattr(native,'_backend_build_info',lambda:{})(),"
        "'execution_space':native._kokkos_default_execution_space(),"
        "'device_environment':environment,'device_sentinel':sentinel},sort_keys=True)); "
        "native._finalize_kokkos()"
    )
    environment = dict(os.environ)
    environment["SYMMETRIX_BACKEND"] = identity.selector
    result = runner.run(
        (manifest.python_executable, "-c", script), check=True, env=environment
    )
    try:
        record = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise BuildError(
            f"installed backend verification returned invalid JSON: {result.stdout}"
        ) from error
    expected_space = {"cpu": "OpenMP", "cuda": "Cuda", "hip": "HIP"}[manifest.backend]
    if record.get("execution_space") != expected_space:
        raise BuildError(
            f"installed extension reports {record.get('execution_space')}, "
            f"expected {expected_space}"
        )
    if record.get("device_sentinel") is not True:
        raise BuildError("installed extension failed the Kokkos device sentinel")
    selected = record.get("selected_backend") or {}
    if selected.get("selector") != identity.selector:
        raise BuildError(
            f"installed frontend selected {selected.get('selector')!r}, "
            f"expected {identity.selector!r}"
        )
    environment = record.get("device_environment") or {}
    if manifest.backend in {"cuda", "hip"}:
        if environment.get("backend") != manifest.backend:
            raise BuildError(
                f"installed extension reports backend {environment.get('backend')}, "
                f"expected {manifest.backend}"
            )
        architecture = str(environment.get("architecture", "")).lower().replace("_", "")
        expected = manifest.architecture.device_target
        if architecture != expected:
            raise BuildError(
                f"installed extension runs on {architecture or 'unknown'}, expected {expected}"
            )
    return record
