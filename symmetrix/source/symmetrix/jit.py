"""Standalone build cache for generated Execution JIT plugin source.

This module deliberately has no dependency on the Symmetrix extension, Kokkos,
or OpenMP.  It provides the cache and compiler boundary that a later generated
plugin integration can call before loading the resulting shared library.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import stat
import struct
import subprocess
import sys
import time
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None


CACHE_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_CACHE"
HOST_TARGET_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HOST_TARGET"
HOST_FLAGS_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HOST_FLAGS"
HOST_FP_MODE_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HOST_FP_MODE"
CUDA_COMPILER_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_NVCC"
CUDA_JIT_BACKEND_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_CUDA_JIT_BACKEND"
CUDA_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_CUDA_R1_EDGE_STRATEGY"
CUDA_R1_EDGE_LOGICAL_WIDTH_ENVIRONMENT_VARIABLE = (
    "SYMMETRIX_JIT_CUDA_R1_EDGE_LOGICAL_WIDTH"
)
CUDA_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE = (
    "SYMMETRIX_JIT_CUDA_R1_EDGE_THREADS_PER_BLOCK"
)
CUDA_R1_EDGE_BLOCKS_PER_SM_ENVIRONMENT_VARIABLE = (
    "SYMMETRIX_JIT_CUDA_R1_EDGE_BLOCKS_PER_SM"
)
NVRTC_LIBRARY_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_NVRTC_LIBRARY"
HIPRTC_LIBRARY_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HIPRTC_LIBRARY"
HIP_COMPILER_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HIPCC"
HIP_JIT_BACKEND_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HIP_JIT_BACKEND"
HIP_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HIP_R1_EDGE_STRATEGY"
HIP_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE = "SYMMETRIX_JIT_HIP_R1_EDGE_THREADS_PER_BLOCK"
HIP_R1_EDGE_BLOCKS_PER_CU_ENVIRONMENT_VARIABLE = (
    "SYMMETRIX_JIT_HIP_R1_EDGE_BLOCKS_PER_CU"
)
CACHE_SCHEMA = "symmetrix.jit.cache-key"
MANIFEST_SCHEMA = "symmetrix.jit.artifact"
SCHEMA_VERSION = 2
CUDA_BUILD_SCHEMA = "symmetrix.jit.cuda-build"
CUDA_BUILD_VERSION = 1
CUDA_MODULE_BUILD_SCHEMA = "symmetrix.jit.cuda-module-build"
CUDA_MODULE_BUILD_VERSION = 1
HIP_BUILD_SCHEMA = "symmetrix.jit.hip-build"
HIP_BUILD_VERSION = 1
HIP_MODULE_BUILD_SCHEMA = "symmetrix.jit.hip-module-build"
HIP_MODULE_BUILD_VERSION = 1
DEFAULT_LOCK_TIMEOUT = 120.0
DEFAULT_STALE_LOCK_AGE = 600.0
DEFAULT_COMPILE_TIMEOUT = 600.0
DEFAULT_HOST_PROBE_TIMEOUT = 30.0

# Increment with libsymmetrix/source/jit_generation_version.hpp whenever
# generated code or its native consumer changes. This is not a Git revision.
JIT_GENERATION_VERSION = 11

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PUBLICATION_ID = re.compile(r"^[0-9a-f]{32}$")
_ARTIFACT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_DOTTED_COMPUTE_CAPABILITY = re.compile(r"^([1-9][0-9]*)\.([0-9])$")
_CUDA_ARCHITECTURE = re.compile(r"^(?:sm|compute)_([1-9][0-9]*)$")
_HIP_ARCHITECTURE = re.compile(r"^(gfx[0-9a-f]+)(?::(.*))?$", re.IGNORECASE)
_NVCC_AMBIENT_FLAG_VARIABLES = (
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
)
_HOST_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_X86_64_MACHINE_NAMES = frozenset(("x86_64", "amd64"))
_X86_64_V3_ARCH_FLAGS = ("-march=x86-64-v3",)
_X86_64_V3_FEATURES = {
    "avx": frozenset(("avx",)),
    "avx2": frozenset(("avx2",)),
    "bmi1": frozenset(("bmi1",)),
    "bmi2": frozenset(("bmi2",)),
    "cmpxchg16b": frozenset(("cx16", "cmpxchg16b")),
    "f16c": frozenset(("f16c",)),
    "fma": frozenset(("fma",)),
    "lahf/sahf": frozenset(("lahf_lm", "lahf_sahf")),
    "lzcnt": frozenset(("lzcnt", "abm")),
    "movbe": frozenset(("movbe",)),
    "popcnt": frozenset(("popcnt",)),
    "sse3": frozenset(("sse3", "pni")),
    "ssse3": frozenset(("ssse3",)),
    "sse4.1": frozenset(("sse4_1", "sse4.1")),
    "sse4.2": frozenset(("sse4_2", "sse4.2")),
    "xsave": frozenset(("xsave",)),
}
_LINUX_RENAMEAT2_SYSCALLS = {
    "x86_64": 316,
    "amd64": 316,
    "aarch64": 276,
    "arm64": 276,
}
_CUDA_RUNTIME_IDENTITY_FIELDS = (
    "runtime_version",
    "cudart_version",
    "runtime_build_version",
)
_MANIFEST_FIELDS = {
    "schema",
    "version",
    "cache_key",
    "publication_id",
    "key_inputs",
    "source",
    "source_sha256",
    "artifact",
    "artifact_sha256",
    "artifact_size",
    "compiler_command",
    "compiler_options",
}
_KEY_INPUT_FIELDS = {
    "schema",
    "version",
    "source_sha256",
    "abi",
    "build",
    "compiler",
    "cpu",
    "cxx_flags",
    "jit_generation_version",
    "artifact",
}


class JitError(RuntimeError):
    """Base class for cache, validation, and build failures."""


class JitManifestError(JitError):
    """Raised when a published cache manifest is malformed or inconsistent."""


class JitLockTimeout(JitError):
    """Raised when another process holds the per-artifact lock too long."""


class JitPublicationUnsupported(JitError):
    """Raised when the cache filesystem cannot publish without replacement."""


@dataclass(frozen=True)
class JitResult:
    """Outcome of preparing a cached standalone shared library.

    ``status`` is ``"built"``, ``"cached"``, or ``"fallback"``.  Expected
    environmental failures are represented by ``fallback`` rather than raised,
    so an evaluator can preserve its generic execution path.
    """

    status: str
    cache_key: str | None
    artifact_path: Path | None
    manifest_path: Path | None
    manifest: Mapping[str, Any] | None
    command: tuple[str, ...]
    diagnostics: tuple[str, ...]
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status in ("built", "cached")

    @property
    def success(self) -> bool:
        return self.available

    @property
    def cache_hit(self) -> bool:
        return self.status == "cached"


@dataclass(frozen=True)
class JitQuarantine:
    """A load-broken cache entry retained until replacement validation."""

    path: Path | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class HostJitPolicy:
    """Resolved host compiler target policy and effective extra arguments."""

    requested: str
    selected: str
    flags: tuple[str, ...]
    diagnostics: tuple[str, ...]

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema": "symmetrix.jit.host-policy/1",
            "requested": self.requested,
            "selected": self.selected,
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class _CompileConfiguration:
    command: tuple[str, ...]
    compiler: Mapping[str, Any]
    build: Any
    target: Any
    key_flags: tuple[str, ...]
    options: tuple[str, ...]
    source_name: str
    compiler_label: str
    environment: Mapping[str, str] | None = None
    artifact_suffix: str | None = None
    compile_source: Any | None = None


@dataclass(frozen=True)
class CompilerRequest:
    """Semantic request consumed by one registered Execution GPU compiler adapter.

    ``optimization`` deliberately describes intent instead of spelling a flag:
    NVCC, hipcc, hipRTC, and NVRTC do not accept the same option vocabulary.
    Backend-specific options remain available as a validated escape hatch.
    """

    kind: str
    build: Any
    target: Any
    source_kind: str
    artifact_kind: str
    optimization: str = "release"
    additional_options: tuple[str, ...] = ()
    compiler: Any | None = None
    information: Mapping[str, Any] | None = None
    compile_source: Any | None = None


@dataclass(frozen=True)
class _CompilerAdapter:
    kind: str
    backend: str
    fallback: str | None
    source_kind: str
    artifact_kind: str
    configure: Any


_COMPILER_ADAPTERS: dict[str, _CompilerAdapter] = {}


def versioned_jit_artifact_name(name: str) -> str:
    """Return the cache artifact basename for the current generation."""

    return f"{name}_gen{JIT_GENERATION_VERSION}"


def _register_compiler_adapter(adapter: _CompilerAdapter) -> None:
    if adapter.kind in _COMPILER_ADAPTERS:
        raise RuntimeError(f"duplicate Execution compiler adapter: {adapter.kind}")
    _COMPILER_ADAPTERS[adapter.kind] = adapter


def execution_compiler_registry() -> dict[str, dict[str, str | None]]:
    """Return stable metadata for the registered GPU compiler adapters."""

    return {
        kind: {
            "backend": adapter.backend,
            "fallback": adapter.fallback,
            "source_kind": adapter.source_kind,
            "artifact_kind": adapter.artifact_kind,
        }
        for kind, adapter in _COMPILER_ADAPTERS.items()
    }


def jit_cache_root(
    cache_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the per-user Execution JIT cache root.

    An explicit argument wins, followed by ``SYMMETRIX_JIT_CACHE`` and
    ``XDG_CACHE_HOME``.  The directory is not created by this query.
    """

    if cache_root is not None:
        return Path(cache_root).expanduser()
    override = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "symmetrix" / "execution-jit"
    return Path.home() / ".cache" / "symmetrix" / "execution-jit"


def _ensure_private_cache_root(path: Path) -> None:
    """Create or validate the same-UID trust boundary for executable artifacts."""

    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise JitError("Execution JIT cache root must not be a symbolic link")
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise JitError("Execution JIT cache root is not a directory")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise JitError("Execution JIT cache root must be owned by the current user")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise JitError("Execution JIT cache root must be private; use permissions 0700")


def _json_value(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{path}[]") for item in value]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compiler_argv(
    cxx: str | os.PathLike[str] | Sequence[str] | None,
) -> tuple[str, ...]:
    if cxx is None:
        cxx = os.environ.get("CXX", "c++")
    if isinstance(cxx, os.PathLike):
        command = (os.fspath(cxx),)
    elif isinstance(cxx, str):
        command = tuple(shlex.split(cxx))
    else:
        command = tuple(str(part) for part in cxx)
    if not command or any(not part for part in command):
        raise JitError("CXX does not name a compiler command")
    return command


def probe_compiler_identity(
    cxx: str | os.PathLike[str] | Sequence[str] | None = None,
    *,
    timeout: float = 10.0,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Resolve ``CXX`` and return the command plus cache-key identity."""

    command = _compiler_argv(cxx)
    executable = shutil.which(command[0])
    if executable is None:
        candidate = Path(command[0]).expanduser()
        if candidate.is_file():
            executable = str(candidate.resolve())
    if executable is None:
        raise JitError(f"CXX compiler executable was not found: {command[0]}")

    resolved_command = (executable, *command[1:])
    try:
        completed = subprocess.run(
            [*resolved_command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JitError(f"could not query CXX compiler identity: {exc}") from exc
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode != 0:
        raise JitError(f"CXX --version exited with {completed.returncode}: {version}")
    stat = Path(executable).stat()
    identity = {
        "command": list(resolved_command),
        "resolved_executable": str(Path(executable).resolve()),
        "version": version,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return resolved_command, identity


def _parse_host_jit_flags(value: str) -> tuple[str, ...]:
    try:
        flags = tuple(shlex.split(value))
    except ValueError as exc:
        raise JitError(
            f"{HOST_FLAGS_ENVIRONMENT_VARIABLE} is not a valid argument list: {exc}"
        ) from exc
    if not flags or any(not flag for flag in flags):
        raise JitError(f"{HOST_FLAGS_ENVIRONMENT_VARIABLE} must not be empty")
    return flags


def _probe_host_compiler_flags(
    compiler_command: Sequence[str],
    flags: Sequence[str],
    *,
    timeout: float = DEFAULT_HOST_PROBE_TIMEOUT,
) -> tuple[bool, str]:
    command = (
        *compiler_command,
        "-std=c++20",
        "-Werror",
        *flags,
        "-x",
        "c++",
        "-fsyntax-only",
        "-",
    )
    try:
        completed = subprocess.run(
            command,
            input="int main() { return 0; }\n",
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JitError(f"could not probe host JIT compiler flags: {exc}") from exc
    diagnostic = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return completed.returncode == 0, diagnostic


def _require_x86_64_v3_host() -> None:
    identity = detect_cpu_identity()
    machine = str(identity.get("machine", "")).lower()
    if machine not in _X86_64_MACHINE_NAMES:
        raise JitError(
            "host JIT portable policy requires an x86-64-v3 CPU; "
            f"detected architecture {machine or 'unknown'!r}"
        )

    cpuinfo = identity.get("cpuinfo")
    raw_features = cpuinfo.get("flags", ()) if isinstance(cpuinfo, Mapping) else ()
    features = {str(feature).lower() for feature in raw_features}
    if not features:
        raise JitError(
            "host JIT portable policy could not verify x86-64-v3 CPU features"
        )
    missing = [
        name
        for name, aliases in _X86_64_V3_FEATURES.items()
        if features.isdisjoint(aliases)
    ]
    if missing:
        raise JitError(
            "host JIT portable policy requires x86-64-v3; missing CPU features: "
            + ", ".join(missing)
        )


def _host_fp_flags() -> tuple[str, tuple[str, ...]]:
    mode = os.environ.get(HOST_FP_MODE_ENVIRONMENT_VARIABLE, "fast").lower()
    if mode == "fast":
        return mode, ("-ffast-math",)
    if mode == "ieee":
        return mode, ("-fno-fast-math", "-ffp-contract=off")
    raise JitError(f"{HOST_FP_MODE_ENVIRONMENT_VARIABLE} must be 'fast' or 'ieee'")


def resolve_host_jit_policy(
    compiler_command: Sequence[str],
    *,
    target: str | None = None,
    flags: str | Sequence[str] | None = None,
) -> HostJitPolicy:
    """Resolve host JIT flags without invoking a shell.

    ``automatic`` probes the optimized native target, then the CPU-verified
    x86-64-v3 portable target when the native probe is rejected.
    """

    requested = (
        os.environ.get(HOST_TARGET_ENVIRONMENT_VARIABLE, "automatic")
        if target is None
        else str(target)
    )
    explicit_flags = (
        os.environ.get(HOST_FLAGS_ENVIRONMENT_VARIABLE) if flags is None else flags
    )
    if explicit_flags is not None:
        if isinstance(explicit_flags, str):
            selected_flags = _parse_host_jit_flags(explicit_flags)
        elif isinstance(explicit_flags, Sequence) and not isinstance(
            explicit_flags, (bytes, bytearray)
        ):
            selected_flags = tuple(str(flag) for flag in explicit_flags)
            if not selected_flags or any(not flag for flag in selected_flags):
                raise JitError("host JIT flags must not contain empty arguments")
        else:
            raise TypeError("host JIT flags must be a string or argument sequence")
        return HostJitPolicy(
            requested=requested,
            selected="explicit-flags",
            flags=selected_flags,
            diagnostics=(
                f"host JIT policy: explicit-flags (target request: {requested})",
                "host JIT flags: " + shlex.join(selected_flags),
            ),
        )

    fp_mode, fp_flags = _host_fp_flags()

    native_probe_diagnostic = None
    if requested == "portable":
        _require_x86_64_v3_host()
        selected = "portable"
        selected_flags = (*fp_flags, *_X86_64_V3_ARCH_FLAGS)
        accepted, probe_diagnostic = _probe_host_compiler_flags(
            compiler_command, selected_flags
        )
        if not accepted:
            detail = f": {probe_diagnostic}" if probe_diagnostic else ""
            raise JitError(f"host JIT portable x86-64-v3 target probe failed{detail}")
    elif requested in ("automatic", "native"):
        selected_flags = (*fp_flags, "-march=native")
        accepted, probe_diagnostic = _probe_host_compiler_flags(
            compiler_command, selected_flags
        )
        if not accepted:
            if requested == "native":
                detail = f": {probe_diagnostic}" if probe_diagnostic else ""
                raise JitError(f"host JIT native target probe failed{detail}")
            native_probe_diagnostic = probe_diagnostic
            _require_x86_64_v3_host()
            selected = "portable"
            selected_flags = (*fp_flags, *_X86_64_V3_ARCH_FLAGS)
            accepted, probe_diagnostic = _probe_host_compiler_flags(
                compiler_command, selected_flags
            )
            if not accepted:
                detail = f": {probe_diagnostic}" if probe_diagnostic else ""
                raise JitError(
                    "host JIT automatic policy rejected both -march=native and "
                    f"the portable x86-64-v3 target{detail}"
                )
        else:
            selected = "native"
    elif _HOST_TARGET.fullmatch(requested):
        selected = f"target:{requested}"
        selected_flags = (*fp_flags, f"-march={requested}")
        accepted, probe_diagnostic = _probe_host_compiler_flags(
            compiler_command, selected_flags
        )
        if not accepted:
            detail = f": {probe_diagnostic}" if probe_diagnostic else ""
            raise JitError(f"host JIT target {requested!r} probe failed{detail}")
    else:
        raise JitError(
            f"{HOST_TARGET_ENVIRONMENT_VARIABLE} must be automatic, native, "
            "portable, or a safe compiler target name"
        )

    diagnostics = [
        f"host JIT policy: {requested} -> {selected}",
        f"host JIT floating-point mode: {fp_mode}",
    ]
    if requested == "automatic" and selected == "portable":
        message = (
            "host JIT -march=native probe was rejected; using portable x86-64-v3 target"
        )
        if native_probe_diagnostic:
            message += f": {native_probe_diagnostic}"
        diagnostics.append(message)
    diagnostics.append("host JIT flags: " + shlex.join(selected_flags))
    return HostJitPolicy(
        requested=requested,
        selected=selected,
        flags=selected_flags,
        diagnostics=tuple(diagnostics),
    )


def _command_argv(
    command: str | os.PathLike[str] | Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    if isinstance(command, os.PathLike):
        argv = (os.fspath(command),)
    elif isinstance(command, str):
        argv = tuple(shlex.split(command))
    else:
        argv = tuple(str(part) for part in command)
    if not argv or any(not part for part in argv):
        raise JitError(f"{label} does not name a compiler command")
    return argv


def _resolve_command_executable(
    command: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    executable = shutil.which(command[0])
    if executable is None:
        candidate = Path(command[0]).expanduser()
        if candidate.is_file():
            executable = str(candidate.resolve())
    if executable is None:
        raise JitError(f"{label} compiler executable was not found: {command[0]}")
    return (executable, *command[1:])


def _discover_nvcc_command(
    nvcc: str | os.PathLike[str] | Sequence[str] | None,
) -> tuple[str, ...]:
    selected: str | os.PathLike[str] | Sequence[str] | None = nvcc
    if selected is None:
        for variable in (
            CUDA_COMPILER_ENVIRONMENT_VARIABLE,
            "CUDACXX",
            "NVCC",
        ):
            value = os.environ.get(variable)
            if value:
                selected = value
                break

    if selected is not None:
        command = _resolve_command_executable(
            _command_argv(selected, label="NVCC"), label="NVCC"
        )
    else:
        candidates: list[Path] = []
        for variable in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
            root = os.environ.get(variable)
            if root:
                candidates.append(Path(root).expanduser() / "bin" / "nvcc")
        path_nvcc = shutil.which("nvcc")
        if path_nvcc is not None:
            candidates.append(Path(path_nvcc))
        candidates.append(Path("/usr/local/cuda/bin/nvcc"))

        executable = next(
            (
                str(candidate.resolve())
                for candidate in candidates
                if candidate.is_file()
            ),
            None,
        )
        if executable is None:
            raise JitError(
                "NVCC compiler executable was not found; set "
                f"{CUDA_COMPILER_ENVIRONMENT_VARIABLE}, CUDACXX, CUDA_HOME, "
                "or put nvcc on PATH"
            )
        command = (executable,)

    if Path(command[0]).name == "nvcc_wrapper":
        raise JitError(
            "nvcc_wrapper is not a supported Execution CUDA JIT compiler; "
            "select its underlying nvcc executable instead"
        )
    return command


def _sanitized_nvcc_environment(diagnostics: list[str]) -> dict[str, str]:
    environment = dict(os.environ)
    for variable in _NVCC_AMBIENT_FLAG_VARIABLES:
        if environment.pop(variable, None) is not None:
            diagnostics.append(
                f"ignored ambient {variable} for deterministic CUDA compilation"
            )
    return environment


def _probe_nvcc_identity(
    nvcc: str | os.PathLike[str] | Sequence[str] | None,
    *,
    environment: Mapping[str, str],
    timeout: float = 10.0,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    command = _discover_nvcc_command(nvcc)
    try:
        completed = subprocess.run(
            [*command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JitError(f"could not query NVCC compiler identity: {exc}") from exc
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode != 0:
        raise JitError(f"NVCC --version exited with {completed.returncode}: {version}")
    executable = Path(command[0])
    metadata = executable.stat()
    identity = {
        "kind": "nvcc",
        "command": list(command),
        "resolved_executable": str(executable.resolve()),
        "version": version,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }
    return command, identity


def _normalize_compute_capability(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        representations = []
        if "major" in value or "minor" in value:
            if {"major", "minor"} - set(value):
                raise ValueError(
                    "compute_capability mapping must contain major and minor"
                )
            representations.append((value["major"], value["minor"]))
        if "compute_capability" in value:
            representations.append(
                _normalize_compute_capability(value["compute_capability"])
            )
        if "compute_capability_code" in value:
            code = value["compute_capability_code"]
            if isinstance(code, bool) or not isinstance(code, int) or code < 10:
                raise ValueError("compute_capability_code must be a positive integer")
            representations.append(_normalize_compute_capability(f"sm_{code}"))
        if not representations:
            raise ValueError(
                "compute_capability mapping must contain major/minor or "
                "compute_capability/compute_capability_code"
            )
        if any(item != representations[0] for item in representations[1:]):
            raise ValueError("compute_capability mapping representations disagree")
        value = representations[0]
    if isinstance(value, str):
        match = _DOTTED_COMPUTE_CAPABILITY.fullmatch(value)
        if match is not None:
            value = (int(match.group(1)), int(match.group(2)))
        else:
            match = _CUDA_ARCHITECTURE.fullmatch(value)
            if match is None:
                raise ValueError("compute_capability must look like '12.0' or 'sm_120'")
            digits = match.group(1)
            if len(digits) < 2:
                raise ValueError("CUDA architecture must contain major and minor")
            value = (int(digits[:-1]), int(digits[-1]))
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
    ):
        raise ValueError("compute_capability must contain major and minor")
    major, minor = value
    if (
        isinstance(major, bool)
        or not isinstance(major, int)
        or major <= 0
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or minor < 0
        or minor > 9
    ):
        raise ValueError(
            "compute_capability major must be positive and minor must be 0..9"
        )
    return major, minor


def _cuda_target_identity(
    compute_capability: Any,
    target: Any | None,
) -> tuple[dict[str, Any], str, tuple[int, int]]:
    major, minor = _normalize_compute_capability(compute_capability)
    architecture = f"sm_{major}{minor}"
    identity: dict[str, Any] = {
        "backend": "cuda",
        "system": platform.system(),
        "machine": platform.machine(),
        "pointer_bits": 8 * struct.calcsize("P"),
        "byteorder": sys.byteorder,
        "compute_capability": [major, minor],
        "architecture": architecture,
    }
    if target is not None:
        identity["runtime"] = _json_value(target, "CUDA target")
    elif isinstance(compute_capability, Mapping):
        runtime = {
            field: compute_capability[field]
            for field in _CUDA_RUNTIME_IDENTITY_FIELDS
            if field in compute_capability
        }
        nested_runtime = compute_capability.get("runtime")
        if isinstance(nested_runtime, Mapping):
            runtime.update(
                {
                    field: nested_runtime[field]
                    for field in _CUDA_RUNTIME_IDENTITY_FIELDS
                    if field in nested_runtime
                }
            )
        if runtime:
            identity["runtime"] = _json_value(runtime, "CUDA runtime target")
    return identity, architecture, (major, minor)


def normalize_execution_hip_target_identity(target: Any) -> dict[str, Any]:
    """Normalize a HIP agent without discarding target features."""

    if isinstance(target, str):
        raw = target
        runtime: Mapping[str, Any] = {}
    elif isinstance(target, Mapping):
        raw_value = target.get("raw_agent_target", target.get("architecture"))
        if not isinstance(raw_value, str):
            raise TypeError("HIP target must contain raw_agent_target")
        raw = raw_value
        runtime = target
    else:
        raise TypeError("HIP target must be an agent string or mapping")
    match = _HIP_ARCHITECTURE.fullmatch(raw)
    if match is None:
        raise ValueError("HIP agent target must look like 'gfxNNN[:feature...]'")
    architecture = match.group(1).lower()
    raw_features = [item for item in (match.group(2) or "").split(":") if item]
    mapped_features = runtime.get("target_features")
    if mapped_features is not None:
        if isinstance(mapped_features, str):
            features = [item for item in mapped_features.split(":") if item]
        elif isinstance(mapped_features, Sequence):
            features = [str(item) for item in mapped_features]
        else:
            raise ValueError("HIP target_features must be a string or sequence")
        if raw_features and sorted(raw_features) != sorted(features):
            raise ValueError("HIP raw agent and target_features disagree")
    else:
        features = raw_features
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*[+-]", item) for item in features):
        raise ValueError("HIP target feature must use a name followed by + or -")
    features = sorted(set(features))
    compiler_target = runtime.get("compiler_offload_target", architecture)
    if compiler_target != architecture:
        raise ValueError("HIP compiler offload target must match the base ISA")
    identity = {
        "backend": "hip",
        "system": platform.system(),
        "machine": platform.machine(),
        "pointer_bits": 8 * struct.calcsize("P"),
        "byteorder": sys.byteorder,
        "raw_agent_target": raw,
        "architecture": architecture,
        "target_features": features,
        "compiler_offload_target": compiler_target,
    }
    runtime_fields = {
        key: runtime[key]
        for key in (
            "runtime_version",
            "driver_version",
            "hiprtc_version",
            "rocblas_version",
            "native_subgroup_width",
        )
        if key in runtime
    }
    if runtime_fields:
        identity["runtime"] = _json_value(runtime_fields, "HIP runtime target")
    return identity


def detect_cpu_identity() -> dict[str, Any]:
    """Return stable CPU and platform inputs relevant to native code generation."""

    identity: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pointer_bits": 8 * struct.calcsize("P"),
        "byteorder": sys.byteorder,
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        wanted = {
            "vendor_id",
            "cpu family",
            "model",
            "model name",
            "stepping",
            "flags",
            "Features",
        }
        first_processor: dict[str, Any] = {}
        try:
            for line in cpuinfo.read_text(errors="replace").splitlines():
                if not line.strip():
                    break
                key, separator, value = line.partition(":")
                key = key.strip()
                if separator and key in wanted:
                    value = value.strip()
                    first_processor[key] = (
                        sorted(set(value.split()))
                        if key in ("flags", "Features")
                        else value
                    )
        except OSError:
            first_processor = {}
        if first_processor:
            identity["cpuinfo"] = first_processor
    return identity


def _key_inputs(
    source: str,
    *,
    abi: Any,
    build: Any,
    compiler: Any,
    cpu: Any,
    cxx_flags: Sequence[str],
    artifact: str,
) -> dict[str, Any]:
    if not isinstance(source, str) or not source:
        raise ValueError("Execution JIT source must be a non-empty string")
    return _json_value(
        {
            "schema": CACHE_SCHEMA,
            "version": SCHEMA_VERSION,
            "source_sha256": _sha256_bytes(source.encode("utf-8")),
            "abi": abi,
            "build": build,
            "compiler": compiler,
            "cpu": cpu,
            "cxx_flags": [str(flag) for flag in cxx_flags],
            "jit_generation_version": JIT_GENERATION_VERSION,
            "artifact": artifact,
        },
        "cache inputs",
    )


def jit_cache_key(
    source: str,
    *,
    abi: Any,
    build: Any,
    compiler: Any,
    cpu: Any,
    cxx_flags: Sequence[str] = (),
    artifact_name: str = "execution_plugin",
) -> str:
    """Compute a deterministic key from all code-generation and ABI inputs."""

    inputs = _key_inputs(
        source,
        abi=abi,
        build=build,
        compiler=compiler,
        cpu=cpu,
        cxx_flags=cxx_flags,
        artifact=_artifact_filename(versioned_jit_artifact_name(artifact_name)),
    )
    return _sha256_bytes(_canonical_json(inputs))


def _shared_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":  # pragma: no cover - no Windows CI
        return ".dll"
    return ".so"


def _artifact_filename(name: str, suffix: str | None = None) -> str:
    if not isinstance(name, str) or not _ARTIFACT_NAME.fullmatch(name):
        raise ValueError("artifact_name must be a safe local basename")
    selected_suffix = _shared_library_suffix() if suffix is None else suffix
    if not selected_suffix.startswith(".") or not _ARTIFACT_NAME.fullmatch(
        "artifact" + selected_suffix
    ):
        raise ValueError("artifact suffix must be a safe extension")
    return name if name.endswith(selected_suffix) else name + selected_suffix


def _entry_paths(cache_root: Path, cache_key: str) -> tuple[Path, Path, Path]:
    entry = cache_root / "artifacts" / cache_key
    return entry, entry / "manifest.json", cache_root / "locks" / f"{cache_key}.lock"


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise JitManifestError(f"manifest {label} is not a SHA-256 digest")
    return value


def validate_jit_manifest(
    entry_directory: str | os.PathLike[str],
    *,
    expected_key: str | None = None,
    expected_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and fully validate a published JIT cache entry."""

    entry = Path(entry_directory)
    if entry.is_symlink() or not entry.is_dir():
        raise JitManifestError("JIT cache entry is not a regular directory")
    manifest_path = entry / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JitManifestError(f"could not read JIT manifest: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise JitManifestError("JIT manifest fields do not match schema")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["version"] != SCHEMA_VERSION:
        raise JitManifestError("JIT manifest schema or version is unsupported")
    publication_id = manifest["publication_id"]
    if not isinstance(publication_id, str) or not _PUBLICATION_ID.fullmatch(
        publication_id
    ):
        raise JitManifestError("JIT manifest publication id is invalid")

    inputs = manifest["key_inputs"]
    if not isinstance(inputs, dict) or set(inputs) != _KEY_INPUT_FIELDS:
        raise JitManifestError("JIT manifest key inputs are invalid")
    inputs = _json_value(inputs, "manifest key inputs")
    calculated_key = _sha256_bytes(_canonical_json(inputs))
    cache_key = _validate_digest(manifest["cache_key"], "cache_key")
    if cache_key != calculated_key:
        raise JitManifestError("JIT manifest cache key does not match its inputs")
    if expected_key is not None and cache_key != expected_key:
        raise JitManifestError("JIT manifest does not match the requested cache key")
    if expected_inputs is not None and inputs != _json_value(expected_inputs):
        raise JitManifestError("JIT manifest inputs do not match the request")

    source_name = manifest["source"]
    artifact_name = manifest["artifact"]
    for value, label in ((source_name, "source"), (artifact_name, "artifact")):
        if not isinstance(value, str) or Path(value).name != value:
            raise JitManifestError(f"JIT manifest {label} is not a basename")
    build_identity = inputs["build"]
    cuda_source = (
        isinstance(build_identity, dict)
        and build_identity.get("schema") == CUDA_BUILD_SCHEMA
        and build_identity.get("version") == CUDA_BUILD_VERSION
        and build_identity.get("backend") == "cuda"
        and build_identity.get("translation_unit") == "plugin.cu"
    )
    cuda_module_source = (
        isinstance(build_identity, dict)
        and build_identity.get("schema") == CUDA_MODULE_BUILD_SCHEMA
        and build_identity.get("version") == CUDA_MODULE_BUILD_VERSION
        and build_identity.get("backend") == "nvrtc"
        and build_identity.get("translation_unit") == "module.cu"
    )
    hip_module_source = (
        isinstance(build_identity, dict)
        and build_identity.get("schema") == HIP_MODULE_BUILD_SCHEMA
        and build_identity.get("version") == HIP_MODULE_BUILD_VERSION
        and build_identity.get("backend") == "hiprtc"
        and build_identity.get("translation_unit") == "module.hip"
    )
    expected_source_name = (
        "module.hip"
        if hip_module_source
        else "module.cu"
        if cuda_module_source
        else "plugin.cu"
        if cuda_source
        else "plugin.cpp"
    )
    if source_name != expected_source_name:
        raise JitManifestError("JIT manifest source name is unsupported")
    if artifact_name != inputs["artifact"]:
        raise JitManifestError(
            "JIT manifest artifact does not match its cache-key inputs"
        )
    source_path = entry / source_name
    artifact_path = entry / artifact_name
    if not source_path.is_file() or source_path.is_symlink():
        raise JitManifestError("JIT cached source is missing or invalid")
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise JitManifestError("JIT cached artifact is missing or invalid")

    source_digest = _validate_digest(manifest["source_sha256"], "source_sha256")
    artifact_digest = _validate_digest(manifest["artifact_sha256"], "artifact_sha256")
    if (
        source_digest != inputs["source_sha256"]
        or _sha256_file(source_path) != source_digest
    ):
        raise JitManifestError("JIT cached source digest does not match")
    if _sha256_file(artifact_path) != artifact_digest:
        raise JitManifestError("JIT cached artifact digest does not match")
    artifact_size = manifest["artifact_size"]
    if (
        isinstance(artifact_size, bool)
        or not isinstance(artifact_size, int)
        or artifact_size <= 0
        or artifact_path.stat().st_size != artifact_size
    ):
        raise JitManifestError("JIT cached artifact size does not match")
    for field in ("compiler_command", "compiler_options"):
        value = manifest[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise JitManifestError(f"JIT manifest {field} is invalid")
    return manifest


def _owner_is_stale(owner: Mapping[str, Any], stale_age: float) -> bool:
    timestamp = owner.get("created_at")
    if not isinstance(timestamp, (int, float)):
        return True
    if time.time() - float(timestamp) >= stale_age:
        return True
    if owner.get("hostname") != socket.gethostname():
        return False
    pid = owner.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _read_owner(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _owner_description(owner: Mapping[str, Any] | None) -> str:
    if not owner:
        return "unknown owner"
    return (
        f"pid={owner.get('pid', '?')} host={owner.get('hostname', '?')} "
        f"created_at={owner.get('created_at', '?')}"
    )


@contextmanager
def _posix_key_lock(
    path: Path,
    *,
    timeout: float,
    stale_age: float,
    diagnostics: list[str],
) -> Iterator[None]:
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    owner = _read_owner(path)
                    raise JitLockTimeout(
                        "timed out waiting for Execution JIT cache lock "
                        f"({_owner_description(owner)})"
                    )
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        stream.seek(0)
        previous_text = stream.read()
        try:
            previous = json.loads(previous_text) if previous_text.strip() else None
        except json.JSONDecodeError:
            previous = None
        if isinstance(previous, dict):
            kind = "stale" if _owner_is_stale(previous, stale_age) else "orphaned"
            diagnostics.append(
                f"recovered {kind} JIT lock owner: {_owner_description(previous)}"
            )
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": time.time(),
            "token": uuid.uuid4().hex,
        }
        stream.seek(0)
        stream.truncate()
        json.dump(owner, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        try:
            yield
        finally:
            stream.seek(0)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


@contextmanager
def _directory_key_lock(
    path: Path,
    *,
    timeout: float,
    stale_age: float,
    diagnostics: list[str],
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_directory = path.with_suffix(path.suffix + ".d")
    owner_path = lock_directory / "owner.json"
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock_directory.mkdir(mode=0o700)
            try:
                owner_path.write_text(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "hostname": socket.gethostname(),
                            "created_at": time.time(),
                            "token": token,
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                shutil.rmtree(lock_directory, ignore_errors=True)
                raise
            break
        except FileExistsError:
            owner = _read_owner(owner_path)
            try:
                stale = (
                    _owner_is_stale(owner, stale_age)
                    if owner is not None
                    else time.time() - lock_directory.stat().st_mtime >= stale_age
                )
            except FileNotFoundError:
                continue
            if stale:
                stale_path = lock_directory.with_name(
                    lock_directory.name + ".stale-" + uuid.uuid4().hex
                )
                try:
                    os.replace(lock_directory, stale_path)
                except OSError:
                    continue
                shutil.rmtree(stale_path, ignore_errors=True)
                diagnostics.append(
                    "recovered stale JIT lock owner: " + _owner_description(owner)
                )
                continue
            if time.monotonic() >= deadline:
                raise JitLockTimeout(
                    "timed out waiting for Execution JIT cache lock "
                    f"({_owner_description(owner)})"
                )
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    try:
        yield
    finally:
        owner = _read_owner(owner_path)
        if owner is not None and owner.get("token") == token:
            shutil.rmtree(lock_directory, ignore_errors=True)


@contextmanager
def _cache_key_lock(
    path: Path,
    *,
    timeout: float,
    stale_age: float,
    diagnostics: list[str],
) -> Iterator[None]:
    if timeout < 0:
        raise ValueError("lock_timeout must be nonnegative")
    if stale_age <= 0:
        raise ValueError("stale_lock_age must be positive")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    implementation = _posix_key_lock if fcntl is not None else _directory_key_lock
    with implementation(
        path,
        timeout=timeout,
        stale_age=stale_age,
        diagnostics=diagnostics,
    ):
        yield


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_staging_directory(root: Path, cache_key: str) -> Path:
    hostname = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname()) or "host"
    for _ in range(8):
        candidate = root / (
            f".{cache_key[:16]}-{hostname}-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:  # pragma: no cover - UUID collision defense
            continue
        return candidate
    raise JitError("could not create a unique JIT staging directory")


def _renameat2_arguments(source: Path, destination: Path) -> tuple[Any, ...]:
    return (
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )


def _linux_renameat2(library: Any, source: Path, destination: Path) -> int:
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        return int(renameat2(*_renameat2_arguments(source, destination)))

    machine = platform.machine().lower()
    syscall_number = _LINUX_RENAMEAT2_SYSCALLS.get(machine)
    if syscall_number is None:
        raise JitPublicationUnsupported(
            "direct renameat2 syscall publication is unsupported on Linux "
            f"architecture {machine!r}"
        )
    syscall = getattr(library, "syscall", None)
    if syscall is None:
        raise JitPublicationUnsupported(
            "the C library provides neither renameat2 nor syscall for JIT publication"
        )
    syscall.argtypes = (
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    syscall.restype = ctypes.c_long
    return int(syscall(syscall_number, *_renameat2_arguments(source, destination)))


def _publish_directory_no_replace(source: Path, destination: Path) -> bool:
    """Atomically publish ``source`` without replacing ``destination``.

    Returns ``True`` when this process publishes and ``False`` when another
    process already published the destination.
    """

    if sys.platform != "linux":
        raise JitPublicationUnsupported(
            "atomic no-replace JIT publication currently requires Linux"
        )
    library = ctypes.CDLL(None, use_errno=True)
    result = _linux_renameat2(library, source, destination)
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        return False
    if error == errno.EXDEV:
        raise JitPublicationUnsupported(
            "JIT staging and artifact directories are on different filesystems"
        )
    unsupported_errors = {errno.ENOSYS, errno.EINVAL}
    for name in ("ENOTSUP", "EOPNOTSUPP"):
        value = getattr(errno, name, None)
        if value is not None:
            unsupported_errors.add(value)
    if error in unsupported_errors:
        raise JitPublicationUnsupported(
            "the JIT cache filesystem does not support atomic no-replace rename"
        )
    raise OSError(error, os.strerror(error), destination)


def _publish_directory_with_mkdir_lock(
    source: Path,
    destination: Path,
    lock_path: Path,
    *,
    timeout: float,
    stale_age: float,
    diagnostics: list[str],
) -> bool:
    """Publish a complete directory while holding a portable key lock."""

    with _directory_key_lock(
        lock_path,
        timeout=timeout,
        stale_age=stale_age,
        diagnostics=diagnostics,
    ):
        if destination.exists() or destination.is_symlink():
            return False
        try:
            os.rename(source, destination)
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                return False
            raise
        return True


def _validated_entry(
    entry: Path,
    cache_key: str,
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    manifest = validate_jit_manifest(
        entry, expected_key=cache_key, expected_inputs=inputs
    )
    return manifest, entry / manifest["artifact"], entry / "manifest.json"


def _quarantine_invalid_entry(
    entry: Path, cache_root: Path, diagnostics: list[str]
) -> Path | None:
    if not entry.exists() and not entry.is_symlink():
        return None
    quarantine_root = cache_root / ".invalid"
    quarantine_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    quarantine = quarantine_root / f"{entry.name}-{uuid.uuid4().hex}"
    os.replace(entry, quarantine)
    _fsync_directory(entry.parent)
    _fsync_directory(quarantine_root)
    diagnostics.append(f"quarantined invalid JIT cache entry at {quarantine}")
    return quarantine


def _remove_cache_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _result_from_entry(
    status: str,
    cache_key: str,
    entry: Path,
    inputs: Mapping[str, Any],
    command: Sequence[str],
    diagnostics: Sequence[str],
) -> JitResult:
    manifest, artifact, manifest_path = _validated_entry(entry, cache_key, inputs)
    return JitResult(
        status=status,
        cache_key=cache_key,
        artifact_path=artifact,
        manifest_path=manifest_path,
        manifest=manifest,
        command=tuple(command),
        diagnostics=tuple(diagnostics),
    )


def quarantine_jit_artifact(
    result: JitResult,
    reason: str,
    *,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    stale_lock_age: float = DEFAULT_STALE_LOCK_AGE,
) -> JitQuarantine:
    """Atomically remove a load-broken result from the reusable cache."""

    if result.cache_key is None or result.artifact_path is None:
        raise ValueError("only an available cached JIT artifact can be quarantined")
    entry = result.artifact_path.parent
    if entry.name != result.cache_key or entry.parent.name != "artifacts":
        raise ValueError("JIT artifact path is outside its content-addressed entry")
    cache_root = entry.parent.parent
    _ensure_private_cache_root(cache_root)
    lock_path = cache_root / "locks" / f"{result.cache_key}.lock"
    diagnostics = [f"artifact load validation failed: {reason}"]
    if result.manifest is None:
        raise ValueError("available JIT artifact has no validated manifest")
    expected_manifest = dict(result.manifest)
    with _cache_key_lock(
        lock_path,
        timeout=lock_timeout,
        stale_age=stale_lock_age,
        diagnostics=diagnostics,
    ):
        if not entry.exists() and not entry.is_symlink():
            quarantine = None
            diagnostics.append("JIT cache entry was already absent")
        else:
            try:
                current_manifest = validate_jit_manifest(
                    entry,
                    expected_key=result.cache_key,
                    expected_inputs=expected_manifest["key_inputs"],
                )
            except JitManifestError as exc:
                diagnostics.append(
                    f"current JIT cache entry is invalid and will be quarantined: {exc}"
                )
            else:
                if current_manifest != expected_manifest:
                    diagnostics.append(
                        "skipped quarantine because the cache key has a newer "
                        "publication"
                    )
                    return JitQuarantine(None, tuple(diagnostics))
            quarantine = _quarantine_invalid_entry(entry, cache_root, diagnostics)
        if quarantine is not None:
            prefix = f"{entry.name}-"
            for previous in quarantine.parent.iterdir():
                if previous != quarantine and previous.name.startswith(prefix):
                    _remove_cache_path(previous)
                    diagnostics.append(
                        f"removed superseded JIT quarantine at {previous}"
                    )
    return JitQuarantine(quarantine, tuple(diagnostics))


def remove_jit_quarantine(
    quarantine: JitQuarantine,
) -> tuple[str, ...]:
    """Discard retained evidence after a replacement plugin loads successfully."""

    path = quarantine.path
    if path is None:
        return ()
    if path.parent.name != ".invalid":
        raise ValueError("JIT quarantine path is outside the invalid-entry directory")
    existed = path.exists() or path.is_symlink()
    _remove_cache_path(path)
    if existed:
        _fsync_directory(path.parent)
        return (f"removed recovered JIT quarantine at {path}",)
    return ()


def _prepare_compiled_artifact(
    source: str,
    *,
    abi: Any,
    configure,
    cache_root: str | os.PathLike[str] | None,
    artifact_name: str,
    lock_timeout: float,
    stale_lock_age: float,
    compile_timeout: float,
) -> JitResult:
    diagnostics: list[str] = []
    command: tuple[str, ...] = ()
    cache_key: str | None = None
    temporary_directory: Path | None = None
    quarantine: Path | None = None
    try:
        if lock_timeout < 0:
            raise ValueError("lock_timeout must be nonnegative")
        if stale_lock_age <= 0:
            raise ValueError("stale_lock_age must be positive")
        if compile_timeout <= 0:
            raise ValueError("compile_timeout must be positive")
        configuration: _CompileConfiguration = configure(diagnostics)
        compiler_command = configuration.command
        artifact_filename = _artifact_filename(
            versioned_jit_artifact_name(artifact_name),
            configuration.artifact_suffix,
        )
        inputs = _key_inputs(
            source,
            abi=abi,
            build=configuration.build,
            compiler=configuration.compiler,
            cpu=configuration.target,
            cxx_flags=configuration.key_flags,
            artifact=artifact_filename,
        )
        cache_key = _sha256_bytes(_canonical_json(inputs))
        root = jit_cache_root(cache_root)
        _ensure_private_cache_root(root)
        entry, _, lock_path = _entry_paths(root, cache_key)

        try:
            return _result_from_entry(
                "cached", cache_key, entry, inputs, compiler_command, diagnostics
            )
        except JitManifestError as exc:
            if entry.exists() or entry.is_symlink():
                diagnostics.append(f"invalid cached JIT entry: {exc}")

        (root / "artifacts").mkdir(parents=True, exist_ok=True, mode=0o700)
        staging_root = root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)

        if entry.exists() or entry.is_symlink():
            with _cache_key_lock(
                lock_path,
                timeout=lock_timeout,
                stale_age=stale_lock_age,
                diagnostics=diagnostics,
            ):
                try:
                    return _result_from_entry(
                        "cached",
                        cache_key,
                        entry,
                        inputs,
                        compiler_command,
                        diagnostics,
                    )
                except JitManifestError as exc:
                    if entry.exists() or entry.is_symlink():
                        diagnostics.append(f"invalid locked JIT cache entry: {exc}")
                        quarantine = _quarantine_invalid_entry(entry, root, diagnostics)

        try:
            result = _result_from_entry(
                "cached", cache_key, entry, inputs, compiler_command, diagnostics
            )
        except JitManifestError:
            if entry.exists() or entry.is_symlink():
                raise JitManifestError(
                    "a conflicting JIT entry appeared before compilation"
                )
        else:
            if quarantine is not None:
                _remove_cache_path(quarantine)
                _fsync_directory(quarantine.parent)
            return result

        temporary_directory = _create_staging_directory(staging_root, cache_key)
        if configuration.source_name not in (
            "plugin.cpp",
            "plugin.cu",
            "module.cu",
            "module.hip",
        ):
            raise JitError("JIT compiler source name is unsupported")
        source_path = temporary_directory / configuration.source_name
        artifact_path = temporary_directory / artifact_filename
        _write_text(source_path, source)
        if os.name == "nt":  # pragma: no cover - no Windows CI
            raise JitError(
                "standalone Execution JIT builds are not supported on Windows"
            )
        options = list(configuration.options)
        if configuration.compile_source is None:
            command = (
                *compiler_command,
                *options,
                str(source_path),
                "-o",
                str(artifact_path),
            )
            diagnostics.append("compiler command: " + shlex.join(command))
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=compile_timeout,
                    env=(
                        None
                        if configuration.environment is None
                        else dict(configuration.environment)
                    ),
                )
            except subprocess.TimeoutExpired as exc:
                for label, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                    if value:
                        if isinstance(value, bytes):
                            value = value.decode(errors="replace")
                        diagnostics.append(f"compiler {label}:\n{value.rstrip()}")
                raise JitError(
                    f"{configuration.compiler_label} compilation timed out after "
                    f"{compile_timeout:g} seconds"
                ) from exc
            if completed.stdout:
                diagnostics.append("compiler stdout:\n" + completed.stdout.rstrip())
            if completed.stderr:
                diagnostics.append("compiler stderr:\n" + completed.stderr.rstrip())
            if completed.returncode != 0:
                raise JitError(
                    f"{configuration.compiler_label} compilation exited with "
                    f"status {completed.returncode}"
                )
        else:
            command = (*compiler_command, *options)
            diagnostics.append("compiler invocation: " + shlex.join(command))
            compiled = configuration.compile_source(source, options)
            if (
                not isinstance(compiled, tuple)
                or len(compiled) != 2
                or not isinstance(compiled[0], bytes)
                or not isinstance(compiled[1], str)
            ):
                raise JitError(
                    f"{configuration.compiler_label} returned an invalid result"
                )
            _write_bytes(artifact_path, compiled[0])
            if compiled[1]:
                diagnostics.append("compiler log:\n" + compiled[1].rstrip())
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            raise JitError(
                f"{configuration.compiler_label} did not produce a non-empty artifact"
            )
        _fsync_path(artifact_path)

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "version": SCHEMA_VERSION,
            "cache_key": cache_key,
            "publication_id": uuid.uuid4().hex,
            "key_inputs": inputs,
            "source": source_path.name,
            "source_sha256": inputs["source_sha256"],
            "artifact": artifact_path.name,
            "artifact_sha256": _sha256_file(artifact_path),
            "artifact_size": artifact_path.stat().st_size,
            "compiler_command": list(compiler_command),
            "compiler_options": options,
        }
        _write_json(temporary_directory / "manifest.json", manifest)
        _validated_entry(temporary_directory, cache_key, inputs)
        _fsync_directory(temporary_directory)
        try:
            published = _publish_directory_no_replace(temporary_directory, entry)
        except JitPublicationUnsupported as exc:
            diagnostics.append(
                f"atomic no-replace JIT publication is unavailable: {exc}; "
                "using mkdir-locked publication"
            )
            published = _publish_directory_with_mkdir_lock(
                temporary_directory,
                entry,
                lock_path,
                timeout=lock_timeout,
                stale_age=stale_lock_age,
                diagnostics=diagnostics,
            )
        if published:
            temporary_directory = None
            _fsync_directory(entry.parent)
            result = _result_from_entry(
                "built", cache_key, entry, inputs, command, diagnostics
            )
        else:
            diagnostics.append("another process published the JIT artifact first")
            result = _result_from_entry(
                "cached", cache_key, entry, inputs, compiler_command, diagnostics
            )
        if quarantine is not None:
            _remove_cache_path(quarantine)
            _fsync_directory(quarantine.parent)
        return result
    except Exception as exc:  # noqa: BLE001 - JIT failures must preserve fallback
        diagnostics.append(f"{type(exc).__name__}: {exc}")
        return JitResult(
            status="fallback",
            cache_key=cache_key,
            artifact_path=None,
            manifest_path=None,
            manifest=None,
            command=tuple(command),
            diagnostics=tuple(diagnostics),
            reason=str(exc),
        )
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)


def _request_options(request: CompilerRequest, label: str) -> tuple[str, ...]:
    options = request.additional_options
    if isinstance(options, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of compiler arguments")
    flags = tuple(str(flag) for flag in options)
    if any(not flag for flag in flags):
        raise ValueError(f"{label} must not contain empty arguments")
    if request.optimization != "release":
        raise ValueError("Execution compiler optimization must be release")
    return flags


def _cuda_request_target(request: CompilerRequest) -> tuple[Any, Any | None]:
    if (
        not isinstance(request.target, Mapping)
        or "compute_capability" not in request.target
    ):
        raise TypeError("CUDA compiler target must contain compute_capability")
    return request.target["compute_capability"], request.target.get("runtime_target")


def _configure_nvcc_adapter(
    request: CompilerRequest, diagnostics: list[str]
) -> _CompileConfiguration:
    environment = _sanitized_nvcc_environment(diagnostics)
    compute_capability, runtime_target = _cuda_request_target(request)
    selected_target, architecture, capability = _cuda_target_identity(
        compute_capability, runtime_target
    )
    compiler_command, compiler = _probe_nvcc_identity(
        request.compiler, environment=environment
    )
    flags = _request_options(request, "nvcc_flags")
    forbidden_names = {
        "-arch",
        "--gpu-architecture",
        "-gencode",
        "--generate-code",
        "-code",
        "--gpu-code",
        "-o",
        "--output-file",
    }
    if any(flag.partition("=")[0] in forbidden_names for flag in flags):
        raise ValueError(
            "nvcc_flags must not override the CUDA architecture or output path"
        )
    major, minor = capability
    options = (
        "-std=c++20",
        "-O3",
        "-shared",
        "--cudart=shared",
        "-Xcompiler=-fPIC",
        f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
        *flags,
    )
    cuda_build = {
        "schema": CUDA_BUILD_SCHEMA,
        "version": CUDA_BUILD_VERSION,
        "backend": "cuda",
        "translation_unit": "plugin.cu",
        "compute_capability": [major, minor],
        "architecture": architecture,
        "request": request.build,
    }
    return _CompileConfiguration(
        command=compiler_command,
        compiler=compiler,
        build=cuda_build,
        target=selected_target,
        key_flags=options,
        options=options,
        source_name="plugin.cu",
        compiler_label="NVCC",
        environment=environment,
    )


def _configure_hipcc_adapter(
    request: CompilerRequest, _diagnostics: list[str]
) -> _CompileConfiguration:
    selected = request.compiler
    if selected is None:
        selected = (
            os.environ.get(HIP_COMPILER_ENVIRONMENT_VARIABLE)
            or os.environ.get("HIPCXX")
            or "hipcc"
        )
    command, compiler = probe_compiler_identity(selected)
    if "HIP version" not in compiler["version"]:
        raise JitError("selected HIP compiler does not identify as hipcc")
    compiler = {**compiler, "kind": "hipcc"}
    identity = normalize_execution_hip_target_identity(request.target)
    flags = _request_options(request, "hipcc_flags")
    forbidden = ("--offload-arch", "--amdgpu-target", "-o")
    if any(flag.partition("=")[0] in forbidden for flag in flags):
        raise ValueError(
            "hipcc_flags must not override the HIP architecture or output path"
        )
    architecture = identity["compiler_offload_target"]
    options = (
        "-x",
        "hip",
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-shared",
        f"--offload-arch={architecture}",
        *flags,
    )
    hip_build = {
        "schema": HIP_BUILD_SCHEMA,
        "version": HIP_BUILD_VERSION,
        "backend": "hip",
        "compiler_backend": "hipcc",
        "translation_unit": "plugin.cpp",
        "architecture": architecture,
        "target_features": identity["target_features"],
        "request": request.build,
    }
    return _CompileConfiguration(
        command=command,
        compiler=compiler,
        build=hip_build,
        target=identity,
        key_flags=options,
        options=options,
        source_name="plugin.cpp",
        compiler_label="hipcc",
    )


def _rtc_information(
    request: CompilerRequest, *, label: str, query
) -> tuple[dict[str, Any], int, int]:
    information = dict(
        query()
        if request.information is None
        else _json_value(request.information, f"{label} information")
    )
    if not bool(information.get("available", False)):
        raise JitError(
            f"{label} is unavailable: "
            + str(information.get("reason", "library not found"))
        )
    major = information.get("major")
    minor = information.get("minor")
    if (
        isinstance(major, bool)
        or not isinstance(major, int)
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or major < 0
        or minor < 0
    ):
        raise JitError(f"{label} reported an invalid compiler version")
    return information, major, minor


def _configure_nvrtc_adapter(
    request: CompilerRequest, diagnostics: list[str]
) -> _CompileConfiguration:
    diagnostics.append(
        "NVRTC compilation is in-process; compile_timeout is not an interrupt deadline"
    )
    compute_capability, runtime_target = _cuda_request_target(request)
    selected_target, architecture, capability = _cuda_target_identity(
        compute_capability, runtime_target
    )
    information, major_version, minor_version = _rtc_information(
        request, label="NVRTC", query=jit_nvrtc_information
    )
    supported = information.get("supported_architectures")
    if not isinstance(supported, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in supported
    ):
        raise JitError("NVRTC reported an invalid architecture list")
    architecture_code = 10 * capability[0] + capability[1]
    if architecture_code not in supported:
        raise JitError(
            f"NVRTC {major_version}.{minor_version} does not support {architecture}"
        )
    flags = _request_options(request, "nvrtc_options")
    if any(
        flag.partition("=")[0]
        in ("--gpu-architecture", "-arch", "--device-debug", "-G")
        for flag in flags
    ):
        raise ValueError(
            "nvrtc_options must not override the target architecture or "
            "enable device debug"
        )
    # NVRTC 13.1 rejects -O3. Its default compilation is the release policy.
    options = ("--std=c++20", f"--gpu-architecture={architecture}", *flags)
    compiler = {
        "kind": "nvrtc",
        "version": [major_version, minor_version],
        "library": str(information.get("library", "")),
    }
    cuda_build = {
        "schema": CUDA_MODULE_BUILD_SCHEMA,
        "version": CUDA_MODULE_BUILD_VERSION,
        "backend": "nvrtc",
        "translation_unit": "module.cu",
        "compute_capability": list(capability),
        "architecture": architecture,
        "request": request.build,
    }
    selected_compile_source = request.compile_source
    if selected_compile_source is None:
        native = _native_nvrtc_module()

        def selected_compile_source(value, selected_options):
            result = native._execution_compile_cuda_with_nvrtc(
                value, list(selected_options)
            )
            if not isinstance(result, Mapping):
                raise JitError("the native NVRTC result is invalid")
            cubin = result.get("cubin")
            log = result.get("log", "")
            if not isinstance(cubin, bytes) or not isinstance(log, str):
                raise JitError("the native NVRTC output is invalid")
            return cubin, log

    return _CompileConfiguration(
        command=(f"NVRTC-{major_version}.{minor_version}",),
        compiler=compiler,
        build=cuda_build,
        target=selected_target,
        key_flags=options,
        options=options,
        source_name="module.cu",
        compiler_label="NVRTC",
        artifact_suffix=".cubin",
        compile_source=selected_compile_source,
    )


def _configure_hiprtc_adapter(
    request: CompilerRequest, diagnostics: list[str]
) -> _CompileConfiguration:
    diagnostics.append(
        "hipRTC compilation is in-process; compile_timeout is not an interrupt deadline"
    )
    identity = normalize_execution_hip_target_identity(request.target)
    information, major_version, minor_version = _rtc_information(
        request, label="hipRTC", query=jit_hiprtc_information
    )
    flags = _request_options(request, "hiprtc_options")
    if any(
        flag.partition("=")[0] in ("--gpu-architecture", "--offload-arch", "-g")
        for flag in flags
    ):
        raise ValueError(
            "hiprtc_options must not override the target architecture or "
            "enable device debug"
        )
    architecture = identity["compiler_offload_target"]
    options = (
        "--std=c++20",
        "-O3",
        f"--gpu-architecture={architecture}",
        *flags,
    )
    compiler = {
        "kind": "hiprtc",
        "version": [major_version, minor_version],
        "library": str(information.get("library", "")),
    }
    hip_build = {
        "schema": HIP_MODULE_BUILD_SCHEMA,
        "version": HIP_MODULE_BUILD_VERSION,
        "backend": "hiprtc",
        "translation_unit": "module.hip",
        "architecture": architecture,
        "target_features": identity["target_features"],
        "request": request.build,
    }
    selected_compile_source = request.compile_source
    if selected_compile_source is None:
        native = _native_hiprtc_module()

        def selected_compile_source(value, selected_options):
            result = native._execution_compile_hip_with_hiprtc(
                value, list(selected_options)
            )
            if not isinstance(result, Mapping):
                raise JitError("the native hipRTC result is invalid")
            code = result.get("code")
            log = result.get("log", "")
            if not isinstance(code, bytes) or not isinstance(log, str):
                raise JitError("the native hipRTC output is invalid")
            return code, log

    return _CompileConfiguration(
        command=(f"hipRTC-{major_version}.{minor_version}",),
        compiler=compiler,
        build=hip_build,
        target=identity,
        key_flags=options,
        options=options,
        source_name="module.hip",
        compiler_label="hipRTC",
        artifact_suffix=".hsaco",
        compile_source=selected_compile_source,
    )


for _adapter in (
    _CompilerAdapter(
        "nvcc", "cuda", None, "cuda", "shared-library", _configure_nvcc_adapter
    ),
    _CompilerAdapter("nvrtc", "cuda", None, "cuda", "cubin", _configure_nvrtc_adapter),
    _CompilerAdapter(
        "hipcc", "hip", None, "hip", "shared-library", _configure_hipcc_adapter
    ),
    _CompilerAdapter(
        "hiprtc", "hip", None, "hip", "code-object", _configure_hiprtc_adapter
    ),
):
    _register_compiler_adapter(_adapter)
del _adapter


def prepare_execution_compiler_artifact(
    source: str,
    *,
    abi: Any,
    request: CompilerRequest,
    cache_root: str | os.PathLike[str] | None = None,
    artifact_name: str = "execution_gpu_artifact",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    stale_lock_age: float = DEFAULT_STALE_LOCK_AGE,
    compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
) -> JitResult:
    """Compile through a registered adapter and the common cache engine."""

    adapter = _COMPILER_ADAPTERS.get(request.kind)
    if adapter is None:
        supported = ", ".join(sorted(_COMPILER_ADAPTERS))
        raise ValueError(
            f"unknown Execution compiler kind {request.kind!r}; use {supported}"
        )
    if request.source_kind != adapter.source_kind:
        raise ValueError(f"{request.kind} source kind must be {adapter.source_kind}")
    if request.artifact_kind != adapter.artifact_kind:
        raise ValueError(
            f"{request.kind} artifact kind must be {adapter.artifact_kind}"
        )
    return _prepare_compiled_artifact(
        source,
        abi=abi,
        configure=lambda diagnostics: adapter.configure(request, diagnostics),
        cache_root=cache_root,
        artifact_name=artifact_name,
        lock_timeout=lock_timeout,
        stale_lock_age=stale_lock_age,
        compile_timeout=compile_timeout,
    )


def prepare_jit_artifact(
    source: str,
    *,
    abi: Any,
    build: Any,
    cache_root: str | os.PathLike[str] | None = None,
    cxx: str | os.PathLike[str] | Sequence[str] | None = None,
    cxx_flags: Sequence[str] | None = (),
    host_target: str | None = None,
    host_flags: str | Sequence[str] | None = None,
    cpu: Any | None = None,
    artifact_name: str = "execution_plugin",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    stale_lock_age: float = DEFAULT_STALE_LOCK_AGE,
    compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
) -> JitResult:
    """Compile and cache standalone generated C++ source.

    The caller supplies the Symmetrix ABI and build identities because this
    standalone layer cannot infer the eventual plugin ABI.  It adds compiler,
    CPU, source, and compile-option identities to the content key.
    """

    def configure(diagnostics: list[str]) -> _CompileConfiguration:
        compiler_command, compiler = probe_compiler_identity(cxx)
        selected_cpu = detect_cpu_identity() if cpu is None else _json_value(cpu, "cpu")
        selected_build = build
        if cxx_flags is None:
            policy = resolve_host_jit_policy(
                compiler_command,
                target=host_target,
                flags=host_flags,
            )
            flags = policy.flags
            diagnostics.extend(policy.diagnostics)
            selected_build = (
                {**build, "host_jit_policy": policy.identity}
                if isinstance(build, Mapping)
                else {"request": build, "host_jit_policy": policy.identity}
            )
        else:
            if host_target is not None or host_flags is not None:
                raise ValueError("host_target and host_flags require cxx_flags=None")
            flags = tuple(str(flag) for flag in cxx_flags)
        return _CompileConfiguration(
            command=compiler_command,
            compiler=compiler,
            build=selected_build,
            target=selected_cpu,
            key_flags=flags,
            options=("-std=c++20", "-O3", "-fPIC", "-shared", *flags),
            source_name="plugin.cpp",
            compiler_label="CXX",
        )

    return _prepare_compiled_artifact(
        source,
        abi=abi,
        configure=configure,
        cache_root=cache_root,
        artifact_name=artifact_name,
        lock_timeout=lock_timeout,
        stale_lock_age=stale_lock_age,
        compile_timeout=compile_timeout,
    )


def prepare_execution_cuda_jit_artifact(
    source: str,
    *,
    abi: Any,
    build: Any,
    compute_capability: Any,
    cache_root: str | os.PathLike[str] | None = None,
    nvcc: str | os.PathLike[str] | Sequence[str] | None = None,
    nvcc_flags: Sequence[str] = (),
    target: Any | None = None,
    artifact_name: str = "factorized_cuda_plugin",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    stale_lock_age: float = DEFAULT_STALE_LOCK_AGE,
    compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
) -> JitResult:
    """Compile and cache a standalone CUDA plugin for one exact GPU target.

    ``compute_capability`` accepts ``(major, minor)``, a dotted string such as
    ``"12.0"``, an architecture string such as ``"sm_120"``, or a native
    environment mapping containing either representation. The caller's build
    metadata is retained under a CUDA-specific identity so model contracts,
    compiler mode, source language, and target architecture all participate in
    the content-addressed cache key.
    """

    request = CompilerRequest(
        kind="nvcc",
        build=build,
        target={
            "compute_capability": compute_capability,
            "runtime_target": target,
        },
        source_kind="cuda",
        artifact_kind="shared-library",
        additional_options=nvcc_flags,
        compiler=nvcc,
    )
    return prepare_execution_compiler_artifact(
        source,
        abi=abi,
        request=request,
        cache_root=cache_root,
        artifact_name=artifact_name,
        lock_timeout=lock_timeout,
        stale_lock_age=stale_lock_age,
        compile_timeout=compile_timeout,
    )


def prepare_execution_hip_jit_artifact(
    source: str,
    *,
    abi: Any,
    build: Any,
    target: Any,
    cache_root: str | os.PathLike[str] | None = None,
    hipcc: str | os.PathLike[str] | Sequence[str] | None = None,
    hipcc_flags: Sequence[str] = (),
    artifact_name: str = "factorized_hip_plugin",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    stale_lock_age: float = DEFAULT_STALE_LOCK_AGE,
    compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
) -> JitResult:
    """Compile and cache a backend-tagged hipcc shared plugin."""

    request = CompilerRequest(
        kind="hipcc",
        build=build,
        target=target,
        source_kind="hip",
        artifact_kind="shared-library",
        additional_options=hipcc_flags,
        compiler=hipcc,
    )
    return prepare_execution_compiler_artifact(
        source,
        abi=abi,
        request=request,
        cache_root=cache_root,
        artifact_name=artifact_name,
        lock_timeout=lock_timeout,
        stale_lock_age=stale_lock_age,
        compile_timeout=compile_timeout,
    )


def prepare_hiprtc_jit_artifact(
    source: str,
    *,
    abi: Any,
    build: Any,
    target: Any,
    cache_root: str | os.PathLike[str] | None = None,
    hiprtc_options: Sequence[str] = (),
    artifact_name: str = "factorized_hip_module",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    stale_lock_age: float = DEFAULT_STALE_LOCK_AGE,
    compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
    hiprtc_information: Mapping[str, Any] | None = None,
    compile_source=None,
) -> JitResult:
    """Compile and cache one exact-target HIP code object through hipRTC."""

    request = CompilerRequest(
        kind="hiprtc",
        build=build,
        target=target,
        source_kind="hip",
        artifact_kind="code-object",
        additional_options=hiprtc_options,
        information=hiprtc_information,
        compile_source=compile_source,
    )
    return prepare_execution_compiler_artifact(
        source,
        abi=abi,
        request=request,
        cache_root=cache_root,
        artifact_name=artifact_name,
        lock_timeout=lock_timeout,
        stale_lock_age=stale_lock_age,
        compile_timeout=compile_timeout,
    )


def _native_nvrtc_module():
    import importlib

    native = importlib.import_module("symmetrix.symmetrix")
    if not hasattr(native, "_jit_nvrtc_information") or not hasattr(
        native, "_execution_compile_cuda_with_nvrtc"
    ):
        raise JitError(
            "this Symmetrix extension does not provide the NVRTC compiler API"
        )
    toolkit = None
    try:
        from symmetrix.backend_loader import selected_backend

        selected = selected_backend()
        toolkit = None if selected is None else str(selected.get("toolkit") or "")
    except (ImportError, AttributeError):
        pass
    _discover_packaged_nvrtc_library(toolkit)
    return native


def _native_hiprtc_module():
    import importlib

    native = importlib.import_module("symmetrix.symmetrix")
    if not hasattr(native, "_jit_hiprtc_information") or not hasattr(
        native, "_execution_compile_hip_with_hiprtc"
    ):
        raise JitError("the native extension does not provide hipRTC support")
    return native


def _discover_packaged_nvrtc_library(toolkit: str | None = None) -> str | None:
    """Expose NVRTC installed by NVIDIA's Python wheel to the native loader."""

    configured = os.environ.get(NVRTC_LIBRARY_ENVIRONMENT_VARIABLE)
    if configured:
        return configured
    import importlib.util

    if toolkit == "cuda13":
        modules = ("nvidia.cu13",)
    elif toolkit == "cuda12":
        modules = ("nvidia.cuda_nvrtc",)
    else:
        modules = ("nvidia.cuda_nvrtc", "nvidia.cu13")
    roots: list[str] = []
    for module in modules:
        try:
            specification = importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
        if specification is None:
            continue
        roots.extend(specification.submodule_search_locations or ())
        if specification.origin and specification.origin != "namespace":
            roots.append(str(Path(specification.origin).parent))
    candidates: list[Path] = []
    for root in dict.fromkeys(roots):
        library_root = Path(root) / "lib"
        candidates.extend(library_root.glob("libnvrtc.so"))
        candidates.extend(library_root.glob("libnvrtc.so.*"))
    if not candidates:
        return None
    selected = min(
        {candidate.resolve() for candidate in candidates},
        key=lambda path: (path.name != "libnvrtc.so", len(path.name), path.name),
    )
    os.environ[NVRTC_LIBRARY_ENVIRONMENT_VARIABLE] = str(selected)
    return str(selected)


def jit_nvrtc_information() -> dict[str, Any]:
    """Return normalized NVRTC availability from the loaded native extension."""

    information = _native_nvrtc_module()._jit_nvrtc_information()
    if not isinstance(information, Mapping):
        raise JitError("the native NVRTC information result is invalid")
    return dict(information)


def jit_hiprtc_information() -> dict[str, Any]:
    """Return normalized hipRTC availability from the loaded native extension."""

    information = _native_hiprtc_module()._jit_hiprtc_information()
    if not isinstance(information, Mapping):
        raise JitError("the native hipRTC information result is invalid")
    return dict(information)


def execution_cuda_jit_backend_request(request: str | None = None) -> str:
    """Return the validated CUDA JIT backend policy request."""

    selected = (
        os.environ.get(CUDA_JIT_BACKEND_ENVIRONMENT_VARIABLE, "automatic")
        if request is None
        else request
    )
    if selected not in ("automatic", "nvrtc", "nvcc"):
        raise ValueError(
            f"{CUDA_JIT_BACKEND_ENVIRONMENT_VARIABLE} must be automatic, nvrtc, or nvcc"
        )
    return selected


def _select_registered_compiler(
    selected: str,
    *,
    preferred: str,
    information_query,
    display_name: str,
) -> str:
    if selected != "automatic" and selected != preferred:
        warnings.warn(
            f"runtime {selected} Execution artifact compilation is deprecated; use "
            f"{preferred} or generic Kokkos execution",
            FutureWarning,
            stacklevel=3,
        )
        return selected
    try:
        information = information_query()
        if bool(information.get("available", False)):
            return preferred
        reason = str(information.get("reason", f"{display_name} is unavailable"))
    except Exception as error:  # noqa: BLE001 - normalize discovery diagnostics
        reason = str(error)
    if selected == preferred:
        raise JitError(
            f"{display_name} was explicitly requested but unavailable: {reason}"
        )
    raise JitError(
        f"automatic {display_name} selection failed: {reason}; no AOT compiler "
        "fallback was attempted"
    )


def select_execution_cuda_jit_backend(request: str | None = None) -> str:
    """Resolve ``automatic``, ``nvrtc``, or ``nvcc`` for CUDA specialization."""

    selected = execution_cuda_jit_backend_request(request)
    return _select_registered_compiler(
        selected,
        preferred="nvrtc",
        information_query=jit_nvrtc_information,
        display_name="NVRTC",
    )


def execution_hip_jit_backend_request(request: str | None = None) -> str:
    """Return the validated HIP JIT backend policy request."""

    selected = (
        os.environ.get(HIP_JIT_BACKEND_ENVIRONMENT_VARIABLE, "automatic")
        if request is None
        else request
    )
    if selected not in ("automatic", "hiprtc", "hipcc"):
        raise ValueError(
            f"{HIP_JIT_BACKEND_ENVIRONMENT_VARIABLE} must be automatic, hiprtc, "
            "or hipcc"
        )
    return selected


def select_execution_hip_jit_backend(request: str | None = None) -> str:
    """Resolve ``automatic``, ``hiprtc``, or ``hipcc`` for HIP specialization."""

    selected = execution_hip_jit_backend_request(request)
    return _select_registered_compiler(
        selected,
        preferred="hiprtc",
        information_query=jit_hiprtc_information,
        display_name="hipRTC",
    )


def execution_cuda_r1_edge_launch_policy(
    request: str | None = None,
    *,
    edge_logical_subgroup_width: int | None = None,
    edge_threads_per_block: int | None = None,
    persistent_blocks_per_compute_unit: int | None = None,
    default_strategy: str = "wave",
    default_logical_subgroup_width: int | None = 32,
    default_edge_threads_per_block: int | None = 32,
    default_persistent_blocks_per_compute_unit: int | None = 8,
) -> dict[str, int | str]:
    """Return validated, cache-keyed CUDA R1 edge launch metadata."""

    def selected_integer(value, environment_name, default, label):
        selected = value
        if selected is None:
            raw = os.environ.get(environment_name)
            if raw is not None:
                try:
                    selected = int(raw)
                except ValueError as error:
                    raise ValueError(
                        f"{environment_name} must be an integer"
                    ) from error
        if selected is None:
            selected = default
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise TypeError(f"{label} must be an integer")
        return selected

    strategy = request
    if strategy is None:
        strategy = os.environ.get(CUDA_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE)
    if strategy is None:
        strategy = default_strategy
    if strategy not in ("serial", "wave"):
        raise ValueError(
            f"{CUDA_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE} must be wave or serial"
        )
    use_qualified_defaults = strategy == default_strategy
    width_default = 1 if strategy == "serial" else 32
    if use_qualified_defaults and default_logical_subgroup_width is not None:
        width_default = default_logical_subgroup_width
    threads_default = 256
    if use_qualified_defaults and default_edge_threads_per_block is not None:
        threads_default = default_edge_threads_per_block
    blocks_default = 1
    if (
        use_qualified_defaults
        and default_persistent_blocks_per_compute_unit is not None
    ):
        blocks_default = default_persistent_blocks_per_compute_unit
    width = selected_integer(
        edge_logical_subgroup_width,
        CUDA_R1_EDGE_LOGICAL_WIDTH_ENVIRONMENT_VARIABLE,
        width_default,
        "CUDA R1 edge logical subgroup width",
    )
    threads = selected_integer(
        edge_threads_per_block,
        CUDA_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE,
        threads_default,
        "CUDA R1 edge threads per block",
    )
    blocks = selected_integer(
        persistent_blocks_per_compute_unit,
        CUDA_R1_EDGE_BLOCKS_PER_SM_ENVIRONMENT_VARIABLE,
        blocks_default,
        "CUDA R1 persistent blocks per SM",
    )
    if width < 1 or width > 32 or width & (width - 1):
        raise ValueError(
            f"{CUDA_R1_EDGE_LOGICAL_WIDTH_ENVIRONMENT_VARIABLE} must be a power "
            "of two in [1, 32]"
        )
    if strategy == "serial" and width != 1:
        raise ValueError("serial CUDA R1 edge strategy requires logical width 1")
    if threads < 32 or threads > 1024 or threads & (threads - 1):
        raise ValueError(
            f"{CUDA_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE} must be a power of two "
            "in [32, 1024]"
        )
    if threads % width:
        raise ValueError("CUDA R1 edge threads must be a logical subgroup multiple")
    if blocks < 1 or blocks > 32:
        raise ValueError(
            f"{CUDA_R1_EDGE_BLOCKS_PER_SM_ENVIRONMENT_VARIABLE} must be in [1, 32]"
        )
    return {
        "strategy": strategy,
        "logical_subgroup_width": width,
        "edge_threads_per_block": threads,
        "persistent_blocks_per_compute_unit": blocks,
    }


def execution_hip_r1_edge_strategy(
    request: str | None = None, *, channels: int | None = None
) -> str:
    """Return the generated HIP R1 edge-kernel policy."""

    selected = request
    if selected is None:
        selected = os.environ.get(HIP_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE)
    if selected is None:
        selected = "wave"
    if selected not in ("wave", "serial"):
        raise ValueError(
            f"{HIP_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE} must be wave or serial"
        )
    return selected


def execution_hip_r1_edge_launch_policy(
    request: str | None = None,
    *,
    channels: int | None = None,
    edge_threads_per_block: int | None = None,
    persistent_blocks_per_compute_unit: int | None = None,
) -> dict[str, int | str]:
    """Return validated, cache-keyed HIP R1 edge launch metadata."""

    def selected_integer(value, environment_name, default, label):
        selected = value
        if selected is None:
            raw = os.environ.get(environment_name)
            if raw is not None:
                try:
                    selected = int(raw)
                except ValueError as error:
                    raise ValueError(
                        f"{environment_name} must be an integer"
                    ) from error
        if selected is None:
            selected = default
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise TypeError(f"{label} must be an integer")
        return selected

    threads = selected_integer(
        edge_threads_per_block,
        HIP_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE,
        64,
        "HIP R1 edge threads per block",
    )
    blocks = selected_integer(
        persistent_blocks_per_compute_unit,
        HIP_R1_EDGE_BLOCKS_PER_CU_ENVIRONMENT_VARIABLE,
        8,
        "HIP R1 persistent blocks per compute unit",
    )
    if threads < 64 or threads > 1024 or threads & (threads - 1):
        raise ValueError(
            f"{HIP_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE} must be a power of two "
            "in [64, 1024]"
        )
    if blocks < 1 or blocks > 32:
        raise ValueError(
            f"{HIP_R1_EDGE_BLOCKS_PER_CU_ENVIRONMENT_VARIABLE} must be in [1, 32]"
        )
    return {
        "strategy": execution_hip_r1_edge_strategy(request, channels=channels),
        "edge_threads_per_block": threads,
        "persistent_blocks_per_compute_unit": blocks,
    }


def prepare_nvrtc_jit_artifact(
    source: str,
    *,
    abi: Any,
    build: Any,
    compute_capability: Any,
    cache_root: str | os.PathLike[str] | None = None,
    nvrtc_options: Sequence[str] = (),
    target: Any | None = None,
    artifact_name: str = "factorized_cuda_module",
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    stale_lock_age: float = DEFAULT_STALE_LOCK_AGE,
    compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
    nvrtc_information: Mapping[str, Any] | None = None,
    compile_source=None,
) -> JitResult:
    """Compile and cache one exact-SM cubin through the native NVRTC API.

    NVRTC runs in-process, so ``compile_timeout`` is validated for API
    consistency but cannot interrupt a compiler call safely. It remains a hard
    deadline only for subprocess compiler backends such as nvcc.
    """

    request = CompilerRequest(
        kind="nvrtc",
        build=build,
        target={
            "compute_capability": compute_capability,
            "runtime_target": target,
        },
        source_kind="cuda",
        artifact_kind="cubin",
        additional_options=nvrtc_options,
        information=nvrtc_information,
        compile_source=compile_source,
    )
    return prepare_execution_compiler_artifact(
        source,
        abi=abi,
        request=request,
        cache_root=cache_root,
        artifact_name=artifact_name,
        lock_timeout=lock_timeout,
        stale_lock_age=stale_lock_age,
        compile_timeout=compile_timeout,
    )


__all__ = [
    "CACHE_ENVIRONMENT_VARIABLE",
    "CUDA_COMPILER_ENVIRONMENT_VARIABLE",
    "CUDA_JIT_BACKEND_ENVIRONMENT_VARIABLE",
    "HIPRTC_LIBRARY_ENVIRONMENT_VARIABLE",
    "HIP_COMPILER_ENVIRONMENT_VARIABLE",
    "HIP_JIT_BACKEND_ENVIRONMENT_VARIABLE",
    "HIP_R1_EDGE_BLOCKS_PER_CU_ENVIRONMENT_VARIABLE",
    "HIP_R1_EDGE_STRATEGY_ENVIRONMENT_VARIABLE",
    "HIP_R1_EDGE_THREADS_ENVIRONMENT_VARIABLE",
    "HOST_FLAGS_ENVIRONMENT_VARIABLE",
    "HOST_TARGET_ENVIRONMENT_VARIABLE",
    "NVRTC_LIBRARY_ENVIRONMENT_VARIABLE",
    "CompilerRequest",
    "HostJitPolicy",
    "JitManifestError",
    "JitQuarantine",
    "JitResult",
    "JIT_GENERATION_VERSION",
    "detect_cpu_identity",
    "execution_compiler_registry",
    "execution_cuda_jit_backend_request",
    "execution_hip_jit_backend_request",
    "execution_hip_r1_edge_launch_policy",
    "execution_hip_r1_edge_strategy",
    "jit_cache_key",
    "jit_cache_root",
    "jit_hiprtc_information",
    "jit_nvrtc_information",
    "normalize_execution_hip_target_identity",
    "prepare_execution_compiler_artifact",
    "prepare_execution_cuda_jit_artifact",
    "prepare_execution_hip_jit_artifact",
    "prepare_hiprtc_jit_artifact",
    "prepare_jit_artifact",
    "prepare_nvrtc_jit_artifact",
    "probe_compiler_identity",
    "quarantine_jit_artifact",
    "remove_jit_quarantine",
    "resolve_host_jit_policy",
    "select_execution_cuda_jit_backend",
    "select_execution_hip_jit_backend",
    "validate_jit_manifest",
    "versioned_jit_artifact_name",
]
