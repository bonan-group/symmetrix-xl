"""Host, accelerator, and compiler detection for Symmetrix builds."""

from __future__ import annotations

import ctypes.util
import os
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .command import BuildError, CommandRunner, canonical_executable
from .manifest import TargetManifest, Toolchain
from .targets import (
    canonical_cuda_target,
    canonical_hip_target,
    cpu_architecture,
    cuda_architecture,
    hip_architecture,
    normalize_backend,
    validate_cuda_target_for_toolkit,
)


@dataclass(frozen=True)
class DetectionRequest:
    backend: str = "auto"
    arch: str = "auto"
    cpu_target: str = "native"
    cuda_root: str = ""
    rocm_root: str = ""
    cxx: str = ""
    host_cxx: str = ""
    generator: str = ""
    blas: str = "auto"
    mkl_root: str = ""
    openblas_root: str = ""
    accelerator_blas_root: str = ""


@dataclass(frozen=True)
class DeviceProbe:
    nvidia: tuple[str, ...]
    amd: tuple[str, ...]
    nvidia_command: str = ""
    amd_command: str = ""


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _candidate_executable(
    runner: CommandRunner,
    explicit: str,
    names: Iterable[str],
    common_paths: Iterable[str] = (),
) -> str:
    if explicit:
        return canonical_executable(explicit)
    for name in names:
        found = runner.which(name)
        if found:
            return canonical_executable(found)
    for path in common_paths:
        candidate = Path(path)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return ""


def probe_devices(runner: CommandRunner) -> DeviceProbe:
    nvidia_command = _candidate_executable(runner, "", ("nvidia-smi",))
    nvidia: list[str] = []
    if nvidia_command:
        result = runner.run(
            (
                nvidia_command,
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            )
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                raw = line.strip()
                if raw:
                    try:
                        nvidia.append(canonical_cuda_target(raw))
                    except BuildError:
                        continue

    amd_command = _candidate_executable(
        runner,
        "",
        ("rocm_agent_enumerator",),
        (
            "/opt/rocm/bin/rocm_agent_enumerator",
            "/opt/rocm/core/bin/rocm_agent_enumerator",
        ),
    )
    amd: list[str] = []
    if amd_command:
        result = runner.run((amd_command,))
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                raw = line.strip().lower()
                if not raw or raw == "gfx000":
                    continue
                try:
                    amd.append(canonical_hip_target(raw))
                except BuildError:
                    continue

    return DeviceProbe(
        _unique(nvidia),
        _unique(amd),
        nvidia_command,
        amd_command,
    )


def _single_detected_target(values: tuple[str, ...], backend: str) -> str:
    if not values:
        raise BuildError(
            f"cannot auto-detect a {backend.upper()} architecture; "
            "make a device visible or pass --arch explicitly"
        )
    if len(values) != 1:
        raise BuildError(
            f"visible {backend.upper()} devices have heterogeneous architectures: "
            f"{', '.join(values)}; pass --arch explicitly"
        )
    return values[0]


def _version_output(runner: CommandRunner, executable: str) -> str:
    result = runner.run((executable, "--version"), check=True)
    return result.output.strip()


def _first_line(value: str) -> str:
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def _cmake_toolchain(runner: CommandRunner) -> tuple[str, str]:
    cmake = _candidate_executable(runner, "", ("cmake",))
    if not cmake:
        raise BuildError("CMake 3.27 or newer is required but cmake was not found")
    output = _version_output(runner, cmake)
    match = re.search(r"cmake version\s+([0-9]+)[.]([0-9]+)(?:[.]([0-9]+))?", output)
    if not match:
        raise BuildError(f"cannot parse CMake version from: {_first_line(output)}")
    version = ".".join(value or "0" for value in match.groups())
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < (3, 27):
        raise BuildError(f"CMake 3.27 or newer is required; found {version}")
    return cmake, version


def _derive_root(executable: str, marker: str) -> str:
    path = Path(executable).resolve()
    parts = path.parts
    if marker in parts:
        index = parts.index(marker)
        return str(Path(*parts[: index + 1]))
    if path.parent.name == "bin":
        return str(path.parent.parent)
    return ""


def _runtime_library_directories(*roots: Path) -> tuple[str, ...]:
    candidates = (
        directory.resolve()
        for root in roots
        for directory in (root / "lib", root / "lib64")
        if directory.is_dir()
    )
    return _unique(str(directory) for directory in candidates)


def _cuda_toolchain(
    request: DetectionRequest,
    runner: CommandRunner,
    repo_root: Path,
    architecture: str,
    cmake: str,
    cmake_version: str,
) -> Toolchain:
    cuda_root = (
        Path(request.cuda_root).expanduser().resolve() if request.cuda_root else None
    )
    nvcc = _candidate_executable(
        runner,
        str(cuda_root / "bin/nvcc") if cuda_root else "",
        ("nvcc",),
        ("/usr/local/cuda/bin/nvcc",),
    )
    if not nvcc:
        raise BuildError(
            "a CUDA device is selected but nvcc was not found; pass --cuda-root"
        )
    if cuda_root is None:
        derived = _derive_root(nvcc, "cuda")
        cuda_root = Path(derived) if derived else Path(nvcc).parent.parent
    output = _version_output(runner, nvcc)
    match = re.search(r"release\s+([0-9]+)[.]([0-9]+)", output)
    if not match:
        match = re.search(r"V([0-9]+)[.]([0-9]+)", output)
    if not match:
        raise BuildError(f"cannot determine CUDA version from: {_first_line(output)}")
    cuda_version = f"{match.group(1)}.{match.group(2)}"

    default_wrapper = repo_root / "libsymmetrix/external/kokkos/bin/nvcc_wrapper"
    cxx = canonical_executable(request.cxx or default_wrapper)
    host_cxx = _candidate_executable(
        runner, request.host_cxx, ("g++", "c++", "clang++")
    )
    if not host_cxx:
        raise BuildError("CUDA requires a supported host C++ compiler")
    host_version = _version_output(runner, host_cxx)
    generator = request.generator or "Unix Makefiles"
    return Toolchain(
        cxx=cxx,
        cxx_version=_first_line(host_version),
        host_cxx=host_cxx,
        cmake=cmake,
        cmake_version=cmake_version,
        generator=generator,
        toolkit_root=str(cuda_root),
        toolkit_version=cuda_version,
        runtime_library_dirs=_runtime_library_directories(cuda_root),
    )


def _hip_toolchain(
    request: DetectionRequest,
    runner: CommandRunner,
    cmake: str,
    cmake_version: str,
) -> Toolchain:
    rocm_root = (
        Path(request.rocm_root).expanduser().resolve() if request.rocm_root else None
    )
    hipcc = _candidate_executable(
        runner,
        request.cxx or (str(rocm_root / "bin/hipcc") if rocm_root else ""),
        ("hipcc",),
        ("/opt/rocm/bin/hipcc",),
    )
    if not hipcc:
        raise BuildError(
            "an AMD GPU is selected but hipcc was not found; pass --rocm-root"
        )
    output = _version_output(runner, hipcc)
    match = re.search(r"HIP version:\s*([0-9]+)[.]([0-9]+)", output)
    if not match:
        raise BuildError(f"cannot determine HIP version from: {_first_line(output)}")
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < (6, 2):
        raise BuildError(
            f"pinned Kokkos requires hipcc 6.2 or newer; found {major}.{minor}"
        )
    if rocm_root is None:
        derived = _derive_root(hipcc, "rocm")
        rocm_root = Path(derived) if derived else Path(hipcc).parent.parent
    return Toolchain(
        cxx=hipcc,
        cxx_version=_first_line(output),
        host_cxx="",
        cmake=cmake,
        cmake_version=cmake_version,
        generator=request.generator or "Unix Makefiles",
        toolkit_root=str(rocm_root),
        toolkit_version=f"{major}.{minor}",
        runtime_library_dirs=_runtime_library_directories(
            Path(hipcc).resolve().parent.parent, rocm_root
        ),
    )


def _cpu_toolchain(
    request: DetectionRequest,
    runner: CommandRunner,
    cmake: str,
    cmake_version: str,
) -> Toolchain:
    cxx = _candidate_executable(runner, request.cxx, ("c++", "g++", "clang++"))
    if not cxx:
        raise BuildError("a C++20 compiler was not found; pass --cxx")
    fortran = _candidate_executable(runner, "", ("gfortran", "flang"))
    output = _version_output(runner, cxx)
    return Toolchain(
        cxx=cxx,
        cxx_version=_first_line(output),
        host_cxx="",
        cmake=cmake,
        cmake_version=cmake_version,
        generator=request.generator or "Unix Makefiles",
        fortran=fortran,
    )


def _python_abi() -> str:
    return (
        sys.implementation.cache_tag
        or f"cp{sys.version_info.major}{sys.version_info.minor}"
    )


def _python_executable() -> str:
    # Resolving this symlink would escape a virtual environment and make the
    # subsequent pip operation target the base interpreter.
    return str(Path(sys.executable).absolute())


def _validate_host() -> None:
    if sys.version_info < (3, 10):  # noqa: UP036
        raise BuildError("Python 3.10 or newer is required")
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise BuildError(
            "the automatic build frontend currently supports Linux x86-64 only"
        )


def _mkl_library(root: Path, name: str) -> bool:
    library = root / "lib" / name
    return any(
        candidate.exists()
        for candidate in (library.with_suffix(".so"), library.with_suffix(".a"))
    ) or any(library.parent.glob(f"{name}.so.*"))


def _mkl_root(explicit: str = "") -> Path | None:
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = []
        if os.environ.get("MKLROOT"):
            candidates.append(Path(os.environ["MKLROOT"]))
        candidates.extend(
            (
                Path("/opt/intel/oneapi/mkl/latest"),
                Path("/opt/intel/mkl"),
            )
        )
    for candidate in candidates:
        root = candidate.resolve()
        if (
            (root / "include/mkl.h").is_file()
            and _mkl_library(root, "libmkl_gf_lp64")
            and _mkl_library(root, "libmkl_core")
        ):
            return root
    return None


def _openblas_installation(explicit: str) -> tuple[str, str, str, str]:
    """Return ``(root, shared_library, library_dir, include_dir)``."""
    root = Path(explicit).expanduser().resolve()
    if not root.is_dir():
        raise BuildError(f"OpenBLAS root is not a directory: {root}")

    libraries = {
        path
        # Covers the canonical name (`libopenblas.so`, `.so.0`), Debian's
        # `libopenblasp-*.so`, and the names upstream's own build produces for a
        # single target (for example `libopenblas_zenp-r0.3.30.so`), so a prefix
        # without the development symlinks still validates.
        for pattern in ("lib*/**/libopenblas*.so*",)
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not libraries:
        raise BuildError(
            "OpenBLAS root must contain a shared libopenblas.so under lib or "
            f"lib64: {root}"
        )
    # Debian-family packages keep pthread and OpenMP variants in sibling
    # directories. Prefer the OpenMP variant when the requested prefix exposes
    # both; an explicit root remains the authority for other installations.
    library = sorted(
        libraries,
        key=lambda path: (
            "openmp" not in str(path).lower(),
            path.name != "libopenblas.so",
            str(path),
        ),
    )[0]
    includes = {
        path.parent for path in root.glob("include*/**/cblas.h") if path.is_file()
    }
    if not includes:
        raise BuildError(
            "OpenBLAS root must contain cblas.h under include or include/*: " f"{root}"
        )
    include = sorted(
        includes,
        key=lambda path: (
            "openmp" not in str(path).lower() and "openblas" not in str(path).lower(),
            str(path),
        ),
    )[0]
    return str(root), str(library), str(library.parent), str(include)


# Prefixes that commonly carry an OpenBLAS installation.  ``/usr`` covers the
# Debian-family layout, where the OpenMP and pthread variants live in sibling
# ``openblas-openmp`` / ``openblas-pthread`` directories under lib/<triplet>.
_OPENBLAS_CANDIDATE_ROOTS = ("/usr/local", "/usr", "/opt/OpenBLAS", "/opt/openblas")


def _openblas_uses_openmp(library: str) -> bool:
    """True when an OpenBLAS shared library is an OpenMP-threaded build.

    Debian-family packages ship identically named sonames in both variant
    directories, so the file name proves nothing and the library's own
    configuration string is authoritative.  Both checks run out of process: a
    truncated or corrupted object can terminate a ``dlopen`` with a signal
    instead of raising, and a damaged library under a scanned prefix must not
    take the build tool down with it.
    """
    config = _openblas_config_string(library)
    if config is not None:
        return "USE_OPENMP" in config
    return _openblas_links_openmp_runtime(library)


# Runs in a child interpreter: `dlopen` on a damaged object can die by signal.
_OPENBLAS_CONFIG_PROBE = (
    "import ctypes, sys\n"
    "handle = ctypes.CDLL(sys.argv[1])\n"
    "handle.openblas_get_config.restype = ctypes.c_char_p\n"
    "sys.stdout.write((handle.openblas_get_config() or b'').decode('latin-1'))\n"
)


def _openblas_config_string(library: str) -> str | None:
    """Return the library's own build string, or None when probing fails."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _OPENBLAS_CONFIG_PROBE, library],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _openblas_links_openmp_runtime(library: str) -> bool:
    """Whether the library's dynamic dependencies include an OpenMP runtime."""
    try:
        dependencies = subprocess.run(
            ["ldd", library], capture_output=True, text=True, check=False, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(runtime in dependencies for runtime in ("libgomp", "libomp", "libiomp"))


def _discover_openblas_installation(
    roots: Iterable[str] | None = None,
) -> tuple[str, str, str, str] | None:
    """Return the first candidate prefix whose library is OpenMP-threaded.

    ``roots`` defaults to the candidate list at call time so tests (and
    callers) can redirect the search.  A pthread-only prefix is deliberately
    not returned: the caller then keeps the previous unpinned system linkage
    instead of pinning a library that would serialise inside the OpenMP
    regions, and a pthread build in an earlier root cannot shadow an OpenMP
    build in a later one.
    """
    for root in _OPENBLAS_CANDIDATE_ROOTS if roots is None else roots:
        if not Path(root).is_dir():
            continue
        try:
            candidate = _openblas_installation(root)
        except BuildError:
            continue
        if _openblas_uses_openmp(candidate[1]):
            return candidate
    return None


def _cpu_blas_policy(
    requested: str = "auto", mkl_root: str = "", openblas_root: str = ""
) -> tuple[str, str]:
    normalized = requested.lower()
    if normalized not in {"auto", "openblas", "mkl", "mkl-sequential"}:
        raise BuildError("CPU BLAS must be auto, openblas, mkl, or mkl-sequential")
    root = _mkl_root(mkl_root)
    if openblas_root:
        if normalized not in {"auto", "openblas"}:
            raise BuildError("--openblas-root requires --blas openblas or auto")
        validated_root, _, _, _ = _openblas_installation(openblas_root)
        return "openblas", validated_root
    if normalized in {"auto", "openblas"}:
        # Prefer a discoverable OpenMP-threaded OpenBLAS: the pthread build
        # serialises inside this project's OpenMP regions, which costs roughly
        # 8-13% on the direct CPU path.  Discovery only returns a prefix whose
        # library is OpenMP, so a pthread-only installation keeps the previous
        # behaviour (system default, no pinned root).
        discovered = _discover_openblas_installation()
        if discovered is not None:
            return "openblas", discovered[0]
    if normalized == "openblas" and ctypes.util.find_library("openblas"):
        return "openblas", ""
    if normalized == "openblas":
        raise BuildError("OpenBLAS was requested but its library was not found")
    if normalized in {"mkl", "mkl-sequential"} and root is not None:
        threading_library = (
            "libmkl_sequential"
            if normalized == "mkl-sequential"
            else "libmkl_gnu_thread"
        )
        if not _mkl_library(root, threading_library):
            raise BuildError(
                f"MKL was requested but {threading_library} was not found under "
                f"{root / 'lib'}"
            )
        return (
            "mkl_sequential" if normalized == "mkl-sequential" else "mkl_gnu_thread",
            str(root),
        )
    if normalized in {"mkl", "mkl-sequential"}:
        raise BuildError(
            "MKL was requested but no MKLROOT with LP64 development libraries was found"
        )
    if normalized == "auto" and ctypes.util.find_library("openblas"):
        return "openblas", ""
    if (
        normalized == "auto"
        and root is not None
        and _mkl_library(root, "libmkl_gnu_thread")
    ):
        return "mkl_gnu_thread", str(root)
    raise BuildError(
        "an optimized BLAS installation is required; found neither MKL nor OpenBLAS"
    )


def _accelerator_blas_policy(explicit_root: str = "") -> tuple[str, str]:
    if not explicit_root:
        return "system", ""
    root = Path(explicit_root).expanduser().resolve()
    libraries = (root / "lib/libopenblas.a", root / "lib64/libopenblas.a")
    headers = (root / "include/cblas.h", root / "include/openblas/cblas.h")
    if not any(path.is_file() for path in libraries):
        raise BuildError(
            "accelerator BLAS root must contain lib/libopenblas.a or "
            f"lib64/libopenblas.a: {root}"
        )
    if not any(path.is_file() for path in headers):
        raise BuildError(
            "accelerator BLAS root must contain include/cblas.h or "
            f"include/openblas/cblas.h: {root}"
        )
    return "openblas_static", str(root)


def detect_target(
    request: DetectionRequest,
    *,
    repo_root: str | Path,
    runner: CommandRunner | None = None,
    devices: DeviceProbe | None = None,
) -> TargetManifest:
    _validate_host()
    active_runner = runner or CommandRunner()
    root = Path(repo_root).resolve()
    backend = normalize_backend(request.backend)
    probe = devices or probe_devices(active_runner)

    if backend == "auto":
        if probe.nvidia and probe.amd:
            raise BuildError(
                "both NVIDIA and AMD GPUs are visible; pass --backend cuda or hip"
            )
        if probe.nvidia:
            backend = "cuda"
        elif probe.amd:
            backend = "hip"
        else:
            backend = "cpu"

    cmake, cmake_version = _cmake_toolchain(active_runner)
    blas_root = ""
    if backend == "cuda":
        detected = (
            _single_detected_target(probe.nvidia, backend)
            if request.arch == "auto"
            else ""
        )
        architecture = cuda_architecture(request.arch, detected)
        toolchain = _cuda_toolchain(
            request,
            active_runner,
            root,
            architecture.device_target,
            cmake,
            cmake_version,
        )
        validate_cuda_target_for_toolkit(
            toolchain.toolkit_version, architecture.device_target
        )
        host_target = "none"
        blas_policy, blas_root = _accelerator_blas_policy(request.accelerator_blas_root)
        device_probe = probe.nvidia_command or "explicit"
        visible = probe.nvidia
    elif backend == "hip":
        detected = (
            _single_detected_target(probe.amd, backend)
            if request.arch == "auto"
            else ""
        )
        architecture = hip_architecture(request.arch, detected)
        toolchain = _hip_toolchain(request, active_runner, cmake, cmake_version)
        host_target = "none"
        blas_policy, blas_root = _accelerator_blas_policy(request.accelerator_blas_root)
        device_probe = probe.amd_command or "explicit"
        visible = probe.amd
    else:
        architecture = cpu_architecture(request.cpu_target)
        toolchain = _cpu_toolchain(request, active_runner, cmake, cmake_version)
        host_target = architecture.device_target
        blas_policy, blas_root = _cpu_blas_policy(
            request.blas, request.mkl_root, request.openblas_root
        )
        if not toolchain.fortran:
            raise BuildError(
                "CPU builds require a Fortran compiler for BLAS configuration; "
                "install gfortran or flang"
            )
        cxx_name = Path(toolchain.cxx).name.lower()
        if blas_policy == "mkl_gnu_thread" and (
            "clang" in cxx_name or not {"g++", "gcc"}.intersection(cxx_name.split("-"))
        ):
            raise BuildError(
                "GNU-threaded MKL requires a GNU C++ compiler so Kokkos and MKL "
                "share libgomp; use --blas mkl-sequential with another compiler"
            )
        if (
            blas_policy in {"mkl_gnu_thread", "mkl_sequential"}
            and "gfortran" not in Path(toolchain.fortran).name
        ):
            raise BuildError(
                "MKL requires gfortran so CMake selects the GNU LP64 interface; "
                "install gfortran"
            )
        device_probe = "none"
        visible = ()

    return TargetManifest(
        backend=backend,
        architecture=architecture,
        toolchain=toolchain,
        python_executable=_python_executable(),
        python_abi=_python_abi(),
        host_target=host_target,
        blas_policy=blas_policy,
        blas_root=blas_root,
        device_probe=device_probe,
        visible_devices=visible,
    )


def validate_source_checkout(repo_root: str | Path, runner: CommandRunner) -> None:
    root = Path(repo_root).resolve()
    required = (
        root / "symmetrix/pyproject.toml",
        root / "symmetrix/CMakeLists.txt",
        root / "libsymmetrix/external/kokkos/CMakeLists.txt",
        root / "libsymmetrix/external/kokkos-kernels/CMakeLists.txt",
        root / "libsymmetrix/external/sphericart/sphericart/CMakeLists.txt",
        root / "libsymmetrix/external/json/CMakeLists.txt",
        root / "symmetrix/external/pybind11/CMakeLists.txt",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise BuildError(
            "source checkout is incomplete; initialize submodules. Missing: "
            + ", ".join(missing)
        )
    if not runner.which("git"):
        raise BuildError("git is required to prepare the patched SpheriCart source")


def validate_cpu_prerequisites(runner: CommandRunner) -> None:
    if not runner.which("gfortran") and not runner.which("flang"):
        raise BuildError("the CPU KokkosKernels BLAS check requires gfortran or flang")
