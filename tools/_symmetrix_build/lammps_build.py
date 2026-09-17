"""LAMMPS source build orchestration using a Symmetrix target manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .command import (
    BuildError,
    CommandRunner,
    canonical_executable,
    validated_executable,
)
from .manifest import TargetManifest
from .python_build import build_environment, prepare_build_directory

MINIMUM_LAMMPS_DATE = 20251210
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass(frozen=True)
class LammpsInvocation:
    commands: tuple[tuple[str, ...], ...]
    environment: dict[str, str]
    build_directory: Path
    manifest_path: Path
    executable: Path
    installed_manifest_path: Path
    build_fingerprint: str
    provenance: dict[str, Any]
    qualification_path: Path
    reusable_qualification: bool
    requested_packages: tuple[str, ...]


def parse_lammps_version(header: str) -> tuple[str, int]:
    match = re.search(
        r'^#define\s+LAMMPS_VERSION\s+"([0-9]{1,2})\s+' r'([A-Za-z]{3})\s+([0-9]{4})"',
        header,
        flags=re.MULTILINE,
    )
    if not match:
        raise BuildError("could not parse LAMMPS_VERSION from src/version.h")
    month = MONTHS.get(match.group(2).lower())
    if month is None:
        raise BuildError(f"unknown LAMMPS version month: {match.group(2)}")
    day = int(match.group(1))
    year = int(match.group(3))
    text = f"{day} {match.group(2)} {year}"
    return text, year * 10000 + month * 100 + day


def validate_lammps_source(
    source: str | Path, *, require_nvcc_wrapper: bool = False
) -> tuple[Path, str]:
    root = Path(source).expanduser().resolve()
    required = [
        root / "src",
        root / "src/KOKKOS",
        root / "src/version.h",
        root / "cmake/CMakeLists.txt",
    ]
    if require_nvcc_wrapper:
        required.append(root / "lib/kokkos/bin/nvcc_wrapper")
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise BuildError(
            f"unsupported LAMMPS source tree {root}; missing: {', '.join(missing)}"
        )
    version, number = parse_lammps_version((root / "src/version.h").read_text())
    if number < MINIMUM_LAMMPS_DATE:
        raise BuildError(
            f"pair_symmetrix requires LAMMPS 10 Dec 2025 or newer; found {version}"
        )
    return root, version


def _mpi_configuration(
    policy: str, mpi_cxx: str, runner: CommandRunner
) -> tuple[bool, str, str]:
    normalized = policy.lower()
    if normalized not in {"auto", "on", "off"}:
        raise BuildError("--mpi must be auto, on, or off")
    if normalized == "off":
        if mpi_cxx:
            raise BuildError("--mpi-cxx contradicts --mpi off")
        return False, "", ""
    selected = mpi_cxx or runner.which("mpicxx") or runner.which("mpic++") or ""
    if normalized == "on" and not selected:
        raise BuildError("--mpi on requires mpicxx or an explicit --mpi-cxx")
    if not selected:
        return False, "", ""
    wrapper = validated_executable(selected)
    identity = ""
    for option in ("--showme:version", "--version"):
        result = runner.run((wrapper, option))
        if result.returncode == 0 and result.output.strip():
            identity = result.output.strip().splitlines()[0]
            break
    return True, wrapper, identity or "version unavailable"


def probe_mpi_gpu_awareness(
    backend: str,
    mpi_cxx: str,
    repo_root: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    """Compile and run the provider query used by LAMMPS's Kokkos package."""
    accelerator = {"cuda": "cuda", "hip": "rocm"}.get(backend)
    if accelerator is None:
        raise BuildError("GPU-aware MPI can only be required for CUDA or HIP builds")
    source = repo_root / "tools/mpi_gpu_aware_probe.cpp"
    if not source.is_file():
        raise BuildError(f"MPI GPU-awareness probe source is missing: {source}")

    method = f"MPIX_Query_{accelerator}_support"
    with tempfile.TemporaryDirectory(prefix="symmetrix-mpi-probe-") as temporary:
        executable = Path(temporary) / "mpi_gpu_aware_probe"
        compile_result = runner.run(
            (mpi_cxx, "-std=c++20", str(source), "-o", str(executable))
        )
        if compile_result.returncode != 0:
            detail = compile_result.output.strip() or "no compiler diagnostic"
            raise BuildError(
                f"cannot prove {accelerator.upper()}-aware MPI support with "
                f"{mpi_cxx}; the {method} probe did not compile:\n{detail}"
            )
        result = runner.run((str(executable), accelerator))

    match = re.search(
        r"method=(\S+)\s+compile_time=([01])\s+runtime=([01])", result.output
    )
    if match is None:
        detail = result.output.strip() or "no probe output"
        raise BuildError(
            f"cannot interpret the {accelerator.upper()}-aware MPI probe: {detail}"
        )
    supported = (
        result.returncode == 0 and match.group(1) == method and match.group(3) == "1"
    )
    evidence = {
        "required": True,
        "accelerator": accelerator,
        "method": match.group(1),
        "compile_time_support": match.group(2) == "1",
        "runtime_support": match.group(3) == "1",
        "status": "supported" if supported else "unsupported",
    }
    if evidence["status"] != "supported":
        raise BuildError(
            f"{mpi_cxx} does not provide {accelerator.upper()}-aware MPI: "
            f"{evidence['method']} reported compile_time="
            f"{int(evidence['compile_time_support'])}, runtime="
            f"{int(evidence['runtime_support'])}. Load a GPU-aware MPI module "
            "or omit --require-gpu-aware-mpi to build the host-staged path."
        )
    return evidence


def lammps_cmake_definitions(
    manifest: TargetManifest,
    *,
    prefix: Path,
    mpi_enabled: bool,
    mpi_cxx: str,
    lammps_source: Path,
    packages: tuple[str, ...] = (),
    extra_definitions: dict[str, str] | None = None,
) -> dict[str, str]:
    definitions = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_INSTALL_PREFIX": str(prefix),
        "CMAKE_CXX_STANDARD": "20",
        "BUILD_MPI": "ON" if mpi_enabled else "OFF",
        "PKG_KOKKOS": "ON",
        "SYMMETRIX_KOKKOS": "ON",
    }
    if mpi_enabled:
        definitions["MPI_CXX_COMPILER"] = mpi_cxx
    if manifest.backend == "cpu":
        definitions.update(
            {
                "CMAKE_CXX_COMPILER": manifest.toolchain.cxx,
                "Kokkos_ENABLE_CUDA": "OFF",
                "Kokkos_ENABLE_HIP": "OFF",
                "Kokkos_ENABLE_OPENMP": "ON",
                "Kokkos_ENABLE_SERIAL": "OFF",
                "Kokkos_ARCH_NATIVE": (
                    "ON" if manifest.host_target == "native" else "OFF"
                ),
                "SYMMETRIX_HOST_ARCH": manifest.host_target,
                "KokkosKernels_ENABLE_TPL_BLAS": "ON",
                "BLA_VENDOR": (
                    "Intel10_64lp_seq"
                    if manifest.blas_policy == "mkl_sequential"
                    else "Intel10_64lp"
                    if manifest.blas_policy == "mkl_gnu_thread"
                    else "OpenBLAS"
                ),
                "SYMMETRIX_SPHERICART_CUDA": "OFF",
            }
        )
        if manifest.toolchain.fortran:
            definitions["CMAKE_Fortran_COMPILER"] = manifest.toolchain.fortran
        if manifest.blas_root:
            runtime_path = str(Path(manifest.blas_root) / "lib")
            definitions["MKLROOT"] = manifest.blas_root
            definitions["CMAKE_BUILD_RPATH"] = runtime_path
            definitions["CMAKE_INSTALL_RPATH"] = runtime_path
    elif manifest.backend == "cuda":
        wrapper = canonical_executable(lammps_source / "lib/kokkos/bin/nvcc_wrapper")
        definitions.update(
            {
                "CMAKE_CXX_COMPILER": wrapper,
                "CMAKE_CUDA_HOST_COMPILER": manifest.toolchain.host_cxx,
                "Kokkos_ENABLE_CUDA": "ON",
                "Kokkos_ENABLE_HIP": "OFF",
                "Kokkos_ENABLE_OPENMP": "OFF",
                "Kokkos_ENABLE_SERIAL": "ON",
                "Kokkos_ARCH_NATIVE": "OFF",
                f"Kokkos_ARCH_{manifest.architecture.kokkos_trait}": "ON",
                "SYMMETRIX_HOST_ARCH": "none",
                "SYMMETRIX_SPHERICART_CUDA": "ON",
            }
        )
    elif manifest.backend == "hip":
        target = manifest.architecture.compiler_target
        runtime_path = ";".join(manifest.toolchain.runtime_library_dirs)
        definitions.update(
            {
                "CMAKE_CXX_COMPILER": manifest.toolchain.cxx,
                "CMAKE_PREFIX_PATH": manifest.toolchain.toolkit_root,
                "CMAKE_BUILD_RPATH": runtime_path,
                "CMAKE_INSTALL_RPATH": runtime_path,
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
                "SYMMETRIX_HIP_BLAS": "AUTO",
            }
        )
    else:
        raise BuildError(f"unsupported manifest backend: {manifest.backend}")
    for package in packages:
        definitions[f"PKG_{package}"] = "ON"
    managed_prefixes = ("PKG_", "Kokkos_", "SYMMETRIX_")
    for name, value in (extra_definitions or {}).items():
        if name in definitions or name.startswith(managed_prefixes):
            raise BuildError(
                f"LAMMPS CMake definition {name} is managed by the build frontend"
            )
        definitions[name] = value
    return definitions


def normalize_lammps_packages(
    packages: tuple[str, ...], lammps_source: Path
) -> tuple[str, ...]:
    normalized: set[str] = set()
    for requested in packages:
        package = requested.strip().upper()
        if package.startswith("PKG_"):
            package = package[4:]
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9+-]*", package):
            raise BuildError(f"invalid LAMMPS package name: {requested}")
        if package == "KOKKOS":
            raise BuildError("LAMMPS package KOKKOS is always enabled by this frontend")
        if not (lammps_source / "src" / package).is_dir():
            raise BuildError(
                f"LAMMPS package {package} is not present in {lammps_source / 'src'}"
            )
        normalized.add(package)
    return tuple(sorted(normalized))


def parse_lammps_cmake_definitions(values: tuple[str, ...]) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for requested in values:
        name, separator, value = requested.partition("=")
        if not separator or not name or not value:
            raise BuildError(
                f"invalid LAMMPS CMake definition {requested!r}; expected NAME=VALUE"
            )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.+-]*", name):
            raise BuildError(f"invalid LAMMPS CMake variable name: {name}")
        if name in definitions:
            raise BuildError(f"duplicate LAMMPS CMake definition: {name}")
        definitions[name] = value
    return definitions


def _hash_file(
    digest: Any, root: Path, path: Path, content: bytes | None = None
) -> None:
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(8, "little"))
    digest.update(relative)
    if content is None:
        if path.is_symlink():
            content = os.readlink(path).encode("utf-8")
        else:
            content = path.read_bytes()
    digest.update(len(content).to_bytes(8, "little"))
    digest.update(content)


def symmetrix_source_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    roots = (
        repo_root / "tools/_symmetrix_build",
        repo_root / "libsymmetrix/source",
        repo_root / "symmetrix/source",
    )
    files = [
        repo_root / "libsymmetrix/CMakeLists.txt",
        repo_root / "symmetrix/CMakeLists.txt",
        repo_root / "symmetrix/pyproject.toml",
        repo_root / "pair_symmetrix/install.sh",
        repo_root / "tools/mpi_gpu_aware_probe.cpp",
        *sorted((repo_root / "pair_symmetrix").glob("*.h")),
        *sorted((repo_root / "pair_symmetrix").glob("*.cpp")),
    ]
    for source_root in roots:
        if source_root.is_dir():
            files.extend(
                path
                for path in sorted(source_root.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
            )
    for path in sorted(set(files)):
        if path.is_file():
            _hash_file(digest, repo_root, path)
    return digest.hexdigest()


def symmetrix_source_provenance(
    repo_root: Path, runner: CommandRunner
) -> dict[str, str]:
    revision = ""
    submodules = ""
    git = runner.which("git") if (repo_root / ".git").exists() else None
    if git:
        head = runner.run((git, "-C", str(repo_root), "rev-parse", "HEAD"))
        if head.returncode == 0:
            revision = head.stdout.strip()
        status = runner.run(
            (git, "-C", str(repo_root), "submodule", "status", "--recursive")
        )
        if status.returncode == 0:
            submodules = status.stdout.strip()
    return {
        "path": str(repo_root),
        "revision": revision,
        "submodules": submodules,
        "content_sha256": symmetrix_source_fingerprint(repo_root),
    }


def _normalized_lammps_cmake(path: Path) -> bytes:
    lines = path.read_text().splitlines(keepends=True)
    normalized: list[str] = []
    in_symmetrix_block = False
    for line in lines:
        marker = line.rstrip("\r\n")
        if marker == "# BEGIN SYMMETRIX PAIR STYLE":
            in_symmetrix_block = True
            continue
        if marker == "# END SYMMETRIX PAIR STYLE":
            in_symmetrix_block = False
            continue
        if not in_symmetrix_block:
            normalized.append(line)
    # The installer appends a separated block, so repeated installs can leave
    # additional blank lines at EOF after removing the previous block.
    return ("".join(normalized).rstrip() + "\n").encode("utf-8")


def lammps_source_provenance(
    source: Path, version: str, runner: CommandRunner
) -> dict[str, str]:
    revision = ""
    git = runner.which("git") if (source / ".git").exists() else None
    if git:
        result = runner.run((git, "-C", str(source), "rev-parse", "HEAD"))
        if result.returncode == 0:
            revision = result.stdout.strip()

    digest = hashlib.sha256()
    excluded = {
        "src/pair_symmetrix_mace.h",
        "src/pair_symmetrix_mace.cpp",
        "src/KOKKOS/pair_symmetrix_mace_kokkos.h",
        "src/KOKKOS/pair_symmetrix_mace_kokkos.cpp",
    }
    for source_root in (source / "src", source / "cmake", source / "lib"):
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(source).as_posix()
            if not path.is_file() or relative in excluded or ".git" in path.parts:
                continue
            content = (
                _normalized_lammps_cmake(path)
                if relative == "cmake/CMakeLists.txt"
                else None
            )
            _hash_file(digest, source, path, content)
    return {
        "path": str(source),
        "version": version,
        "revision": revision,
        "content_sha256": digest.hexdigest(),
    }


def _lammps_build_fingerprint(provenance: dict[str, Any]) -> str:
    value = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _select_build_directory(base: Path, force_new: bool) -> tuple[Path, bool]:
    attempts = sorted(base.parent.glob(f"{base.name}-attempt-*"))
    if not force_new:
        for candidate in reversed((base, *attempts)):
            if (candidate / "qualification.json").is_file():
                return candidate, True
        if not base.exists() or not any(base.iterdir()):
            return base, False
    used = {candidate.name for candidate in attempts}
    number = 1
    while f"{base.name}-attempt-{number:03d}" in used:
        number += 1
    return base.with_name(f"{base.name}-attempt-{number:03d}"), False


def lammps_build_invocation(
    manifest: TargetManifest,
    *,
    repo_root: str | Path,
    source: str | Path,
    prefix: str | Path,
    mpi: str,
    mpi_cxx: str,
    jobs: int,
    runner: CommandRunner,
    build_root: str | Path | None = None,
    force_new: bool = False,
    require_gpu_aware_mpi: bool = False,
    lammps_packages: tuple[str, ...] = (),
    extra_cmake_definitions: tuple[str, ...] = (),
    materialize: bool = True,
) -> LammpsInvocation:
    if jobs < 1:
        raise BuildError("--jobs must be positive")
    root = Path(repo_root).resolve()
    lammps_source, version = validate_lammps_source(
        source, require_nvcc_wrapper=manifest.backend == "cuda"
    )
    packages = normalize_lammps_packages(lammps_packages, lammps_source)
    extra_definitions = parse_lammps_cmake_definitions(extra_cmake_definitions)
    install_prefix = Path(prefix).expanduser().resolve()
    if require_gpu_aware_mpi and manifest.backend not in {"cuda", "hip"}:
        raise BuildError("GPU-aware MPI can only be required for CUDA or HIP builds")
    if require_gpu_aware_mpi and mpi == "off":
        raise BuildError("--require-gpu-aware-mpi contradicts --mpi off")
    mpi_policy = "on" if require_gpu_aware_mpi else mpi
    mpi_enabled, selected_mpi, mpi_identity = _mpi_configuration(
        mpi_policy, mpi_cxx, runner
    )
    gpu_aware_mpi = (
        probe_mpi_gpu_awareness(manifest.backend, selected_mpi, root, runner)
        if require_gpu_aware_mpi
        else {"required": False, "status": "not_checked"}
    )
    definitions = lammps_cmake_definitions(
        manifest,
        prefix=install_prefix,
        mpi_enabled=mpi_enabled,
        mpi_cxx=selected_mpi,
        lammps_source=lammps_source,
        packages=packages,
        extra_definitions=extra_definitions,
    )
    provenance = {
        "target_fingerprint": manifest.fingerprint,
        "symmetrix": symmetrix_source_provenance(root, runner),
        "lammps": lammps_source_provenance(lammps_source, version, runner),
        "install_prefix": str(install_prefix),
        "cmake_definitions": definitions,
        "mpi": {
            "enabled": mpi_enabled,
            "cxx": selected_mpi,
            "identity": mpi_identity,
            "gpu_aware": gpu_aware_mpi,
        },
    }
    fingerprint = _lammps_build_fingerprint(provenance)
    if build_root is None:
        base_directory = root / "symmetrix" / f"build-lammps-{fingerprint}"
    else:
        base_directory = Path(build_root).expanduser().resolve() / fingerprint
    build_directory, reusable_qualification = _select_build_directory(
        base_directory, force_new
    )
    manifest_path = build_directory / "symmetrix-target.json"
    if materialize:
        manifest_path = prepare_build_directory(build_directory, manifest)

    installer = root / "pair_symmetrix/install.sh"
    if not installer.is_file():
        raise BuildError(f"pair_symmetrix installer is missing: {installer}")
    configure = [
        manifest.toolchain.cmake,
        "-S",
        str(lammps_source / "cmake"),
        "-B",
        str(build_directory),
        "-G",
        manifest.toolchain.generator,
    ]
    configure.extend(f"-D{name}={value}" for name, value in definitions.items())
    commands = (
        (str(installer), str(lammps_source)),
        tuple(configure),
        (
            manifest.toolchain.cmake,
            "--build",
            str(build_directory),
            "--parallel",
            str(jobs),
        ),
        (manifest.toolchain.cmake, "--install", str(build_directory)),
    )
    environment = build_environment(manifest)
    if manifest.backend == "cuda":
        environment["CXX"] = definitions["CMAKE_CXX_COMPILER"]
    record = {
        "build_fingerprint": fingerprint,
        "manifest_fingerprint": manifest.fingerprint,
        "provenance": provenance,
        "commands": commands,
    }
    if materialize:
        (build_directory / "lammps-invocation.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    return LammpsInvocation(
        commands,
        environment,
        build_directory,
        manifest_path,
        install_prefix / "bin/lmp",
        install_prefix / "bin/lmp.symmetrix-target.json",
        fingerprint,
        provenance,
        build_directory / "qualification.json",
        reusable_qualification,
        packages,
    )


def run_lammps_build(invocation: LammpsInvocation, runner: CommandRunner) -> None:
    invocation.qualification_path.unlink(missing_ok=True)
    log_path = invocation.build_directory / "lammps-build.log"
    if log_path.exists():
        number = 1
        while True:
            previous = log_path.with_name(f"lammps-build.previous-{number:03d}.log")
            if not previous.exists():
                log_path.rename(previous)
                break
            number += 1
    log_path.write_text("")
    for command in invocation.commands:
        result = runner.run_logged(
            command,
            log_path,
            env=invocation.environment,
            append=True,
        )
        if result.returncode != 0:
            raise BuildError(
                f"LAMMPS build command failed ({result.returncode}): "
                + " ".join(command)
            )
    TargetManifest.read(invocation.manifest_path).write(
        invocation.installed_manifest_path
    )


def verify_lammps(
    invocation: LammpsInvocation, runner: CommandRunner
) -> dict[str, Any]:
    if not invocation.executable.is_file():
        raise BuildError(
            f"installed LAMMPS executable is missing: {invocation.executable}"
        )
    result = runner.run((str(invocation.executable), "-help"), check=True)
    for style in ("symmetrix/mace", "symmetrix/mace/kk"):
        if style not in result.output:
            raise BuildError(f"installed LAMMPS does not report pair style {style}")
    installed_packages = _installed_lammps_packages(result.output)
    missing_packages = sorted(set(invocation.requested_packages) - installed_packages)
    if missing_packages:
        raise BuildError(
            "installed LAMMPS does not report requested package(s): "
            + ", ".join(missing_packages)
        )
    sha256 = hashlib.sha256(invocation.executable.read_bytes()).hexdigest()
    installed = TargetManifest.read(invocation.installed_manifest_path)
    return {
        "build_fingerprint": invocation.build_fingerprint,
        "executable": str(invocation.executable),
        "sha256": sha256,
        "provenance": invocation.provenance,
        "target_manifest": str(invocation.installed_manifest_path),
        "target_fingerprint": installed.fingerprint,
        "lammps_packages": sorted(installed_packages),
    }


def _installed_lammps_packages(help_output: str) -> set[str]:
    lines = help_output.splitlines()
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == "Installed packages:"
        )
    except StopIteration:
        return set()
    packages: set[str] = set()
    started = False
    for line in lines[start + 1 :]:
        if not line.strip():
            if started:
                break
            continue
        started = True
        packages.update(line.split())
    return packages


def reuse_lammps_qualification(
    invocation: LammpsInvocation, runner: CommandRunner
) -> dict[str, Any] | None:
    if not invocation.reusable_qualification:
        return None
    try:
        record = json.loads(invocation.qualification_path.read_text())
        if not isinstance(record, dict):
            raise TypeError("qualification root is not an object")
        current = verify_lammps(invocation, runner)
        for key in ("build_fingerprint", "sha256", "target_fingerprint"):
            if record.get(key) != current[key]:
                raise ValueError(f"qualification {key} does not match")
    except (BuildError, OSError, TypeError, ValueError, json.JSONDecodeError):
        invocation.qualification_path.unlink(missing_ok=True)
        return None
    reused = dict(record)
    reused["reused"] = True
    return reused
