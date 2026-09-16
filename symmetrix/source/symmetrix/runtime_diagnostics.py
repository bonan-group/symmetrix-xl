"""Runtime-loader diagnostics for native Symmetrix deployments."""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import symmetrix as native_symmetrix


def _sha256(path: str | os.PathLike[str]) -> str | None:
    try:
        with Path(path).open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _runtime_family(path: str | None) -> str | None:
    name = Path(path).name.lower() if path else ""
    if "libgomp" in name:
        return "libgomp"
    if "libiomp" in name:
        return "libiomp"
    if "libomp" in name:
        return "libomp"
    if "vcomp" in name:
        return "vcomp"
    return None


def _vendored_content_hash_prefix(path: str | None) -> str | None:
    """Return auditwheel's 8-hex content hash embedded in a vendored name.

    ``auditwheel repair`` copies a shared library into ``<package>.libs/`` and
    renames it to ``lib<name>-<8 hex chars>.so.<version>`` where the embedded
    prefix is the first eight hex characters of the SHA-256 of the *original*
    (pre-rename) file.  The vendored copy is afterwards patched (SONAME and
    dependent DT_NEEDED rewrites), so its own SHA-256 differs from the
    build-time runtime hash even though it descends from the exact same file.
    """
    name = Path(path).name if path else ""
    stem = name.split(".so", 1)[0]
    prefix = stem.rsplit("-", 1)[-1] if "-" in stem else ""
    return (
        prefix.lower()
        if len(prefix) == 8
        and all(character in "0123456789abcdef" for character in prefix.lower())
        else None
    )


def openmp_runtime_diagnostics(
    *,
    native_module=None,
    strict: bool = True,
    expected_threads: int | None = None,
    allocated_cpus: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Return a hash- and worker-qualified OpenMP deployment report."""

    if expected_threads is not None and expected_threads <= 0:
        raise ValueError("expected_threads must be positive")
    normalized_cpus = (
        None if allocated_cpus is None else sorted({int(cpu) for cpu in allocated_cpus})
    )
    if normalized_cpus is not None and any(cpu < 0 for cpu in normalized_cpus):
        raise ValueError("allocated_cpus must contain non-negative CPU IDs")

    native = native_symmetrix if native_module is None else native_module
    query = getattr(native, "_openmp_runtime_info", None)
    if not callable(query):
        return {
            "schema": "symmetrix.openmp-runtime-diagnostics",
            "version": 1,
            "status": "error",
            "ok": False,
            "strict": strict,
            "python_executable": sys.executable,
            "checks": [
                {
                    "name": "native_api",
                    "status": "error",
                    "message": "the native extension predates OpenMP runtime diagnostics",
                }
            ],
            "runtime": None,
        }

    runtime = dict(query())
    report = {
        "schema": "symmetrix.openmp-runtime-diagnostics",
        "version": 1,
        "status": "ok",
        "ok": True,
        "strict": strict,
        "python_executable": sys.executable,
        "checks": [],
        "runtime": runtime,
    }
    checks = report["checks"]
    if not runtime.get("enabled", False):
        report["status"] = "not_applicable"
        checks.append(
            {
                "name": "openmp_enabled",
                "status": "not_applicable",
                "message": "this Symmetrix extension was not built with Kokkos OpenMP",
            }
        )
        return report

    def add_check(name: str, status: str, message: str) -> None:
        checks.append({"name": name, "status": status, "message": message})
        if status == "error":
            report["ok"] = False
            report["status"] = "error"
        elif status == "warning" and report["status"] == "ok":
            report["status"] = "warning"

    loaded_path = runtime.get("loaded_runtime_path")
    expected_path = runtime.get("expected_runtime_path")
    expected_hash = runtime.get("expected_runtime_sha256")
    loaded_hash = _sha256(loaded_path) if loaded_path else None
    runtime["loaded_runtime_sha256"] = loaded_hash
    runtime["expected_runtime_family"] = _runtime_family(expected_path)
    runtime["loaded_runtime_family"] = _runtime_family(loaded_path)

    if not loaded_path:
        add_check(
            "symbol_owner",
            "error",
            "dladdr could not resolve the loaded omp_get_max_threads symbol owner",
        )
    else:
        add_check("symbol_owner", "ok", f"OpenMP symbols resolve to {loaded_path}")

    libraries = tuple(
        dict.fromkeys(map(str, runtime.get("loaded_openmp_libraries", ())))
    )
    runtime["loaded_openmp_libraries"] = list(libraries)
    if len(libraries) > 1:
        add_check(
            "duplicate_runtimes",
            "error",
            "multiple OpenMP runtimes are mapped: " + ", ".join(libraries),
        )
    else:
        add_check(
            "duplicate_runtimes",
            "ok",
            "one OpenMP runtime is mapped"
            if libraries
            else "no duplicate runtime was reported",
        )

    expected_family = runtime["expected_runtime_family"]
    loaded_family = runtime["loaded_runtime_family"]
    if expected_family and loaded_family != expected_family:
        add_check(
            "runtime_family",
            "error",
            f"expected {expected_family}, but loaded {loaded_family or 'an unknown runtime'}",
        )
    else:
        add_check("runtime_family", "ok", f"loaded runtime family is {loaded_family}")

    if not expected_hash:
        add_check(
            "runtime_hash",
            "error" if strict else "warning",
            "the extension has no build-time OpenMP runtime hash",
        )
    elif loaded_hash is None:
        add_check(
            "runtime_hash", "error", "the loaded OpenMP runtime could not be hashed"
        )
    elif loaded_hash != expected_hash:
        vendored_prefix = _vendored_content_hash_prefix(loaded_path)
        if vendored_prefix and expected_hash.lower().startswith(vendored_prefix):
            add_check(
                "runtime_hash",
                "ok",
                "vendored OpenMP runtime descends from the build-time runtime "
                f"(auditwheel content hash {vendored_prefix}; SONAME rewrite "
                "changes the file digest)",
            )
        else:
            add_check(
                "runtime_hash",
                "error" if strict else "warning",
                f"loaded runtime SHA-256 {loaded_hash} differs from build runtime {expected_hash}",
            )
    else:
        location = (
            "at the build path"
            if loaded_path == expected_path
            else "from a relocated path"
        )
        add_check(
            "runtime_hash",
            "ok",
            f"loaded runtime matches the build-time SHA-256 {location}",
        )

    sentinel = runtime.get("worker_sentinel")
    requested_threads = (
        expected_threads
        if expected_threads is not None
        else runtime.get("omp_max_threads")
    )
    if not isinstance(sentinel, dict):
        add_check(
            "worker_sentinel",
            "error",
            "the native extension did not return an OpenMP worker sentinel",
        )
    else:
        actual_threads = int(sentinel.get("actual_team_size", 0))
        workers = list(sentinel.get("workers", ()))
        thread_ids = [int(worker.get("thread_id", -1)) for worker in workers]
        current_cpus = {int(cpu) for cpu in sentinel.get("unique_current_cpus", ())}
        worker_affinities = [
            {int(cpu) for cpu in worker.get("affinity", ())} for worker in workers
        ]
        affinities_are_disjoint = all(worker_affinities) and all(
            left.isdisjoint(right)
            for index, left in enumerate(worker_affinities)
            for right in worker_affinities[index + 1 :]
        )
        cpu_observation_supported = bool(
            sentinel.get("cpu_observation_supported", False)
        )
        if requested_threads is not None and actual_threads != requested_threads:
            add_check(
                "worker_team_size",
                "error",
                f"OpenMP created {actual_threads} workers; expected {requested_threads}",
            )
        elif actual_threads <= 0:
            add_check("worker_team_size", "error", "OpenMP created no sentinel workers")
        else:
            add_check(
                "worker_team_size",
                "ok",
                f"OpenMP created the requested {actual_threads}-worker team",
            )
        if sorted(thread_ids) != list(range(actual_threads)):
            add_check(
                "worker_ids",
                "error",
                f"sentinel worker IDs are incomplete or duplicated: {thread_ids}",
            )
        else:
            add_check("worker_ids", "ok", "each OpenMP worker participated once")
        if not cpu_observation_supported:
            add_check(
                "worker_cpu_participation",
                "not_applicable",
                "worker CPU and affinity observation is unavailable on this platform",
            )
        elif actual_threads > 1 and not affinities_are_disjoint:
            add_check(
                "worker_cpu_participation",
                "not_applicable",
                "workers are not bound to disjoint CPU sets; instantaneous CPU "
                "samples do not measure worker participation",
            )
        elif actual_threads > 1 and len(current_cpus) != actual_threads:
            add_check(
                "worker_cpu_participation",
                "error",
                f"{actual_threads} workers sampled only {len(current_cpus)} distinct CPUs",
            )
        else:
            add_check(
                "worker_cpu_participation",
                "ok",
                f"workers sampled {len(current_cpus)} distinct CPUs",
            )

        if normalized_cpus is not None:
            allocated = set(normalized_cpus)
            runtime["allocated_cpus"] = normalized_cpus
            if not cpu_observation_supported:
                add_check(
                    "allocated_cpu_coverage",
                    "error",
                    "allocated CPU coverage was requested, but native CPU affinity "
                    "observation is unavailable on this platform",
                )
                allocated = set()
            affinity_union: set[int] = set()
            affinity_outside = set()
            for affinity in worker_affinities:
                affinity_union.update(affinity)
                affinity_outside.update(affinity - allocated)
            current_outside = current_cpus - allocated
            missing = allocated - affinity_union
            if allocated and (
                len(allocated) < actual_threads
                or affinity_outside
                or current_outside
                or missing
            ):
                add_check(
                    "allocated_cpu_coverage",
                    "error",
                    "worker affinity does not cover exactly the allocated CPU set: "
                    f"allocated={normalized_cpus}, missing={sorted(missing)}, "
                    f"affinity_outside={sorted(affinity_outside)}, "
                    f"current_outside={sorted(current_outside)}",
                )
            elif allocated:
                add_check(
                    "allocated_cpu_coverage",
                    "ok",
                    f"worker affinity covers allocated CPUs {normalized_cpus}",
                )

    host_blas = runtime.get("host_blas")
    if not isinstance(host_blas, dict):
        add_check(
            "host_worker_blas",
            "error",
            "the native extension did not report host-worker BLAS routing",
        )
    else:
        selected = host_blas.get("selected_backend")
        environment_policy = host_blas.get("environment_policy")
        openblas_openmp_enabled = host_blas.get("openblas_openmp_enabled")
        openblas_openmp_runtime_compatible = host_blas.get(
            "openblas_openmp_runtime_compatible"
        )
        provider = host_blas.get("provider")
        if provider is None:
            provider = (
                "openblas" if "openblas_openmp_enabled" in host_blas else "unknown"
            )
        threading = host_blas.get("threading")
        if threading is None:
            threading = (
                "openmp"
                if openblas_openmp_enabled is True
                else "pthread"
                if openblas_openmp_enabled is False
                else "unknown"
            )
        runtime_compatible = host_blas.get(
            "openmp_runtime_compatible", openblas_openmp_runtime_compatible
        )
        actual_threads = (
            int(sentinel.get("actual_team_size", 0))
            if isinstance(sentinel, dict)
            else 0
        )
        omp_max_active_levels = runtime.get("omp_max_active_levels")
        nested_teams_disabled = (
            isinstance(omp_max_active_levels, int) and omp_max_active_levels <= 1
        )
        automatic_cblas_safe = (provider == "mkl" and threading == "sequential") or (
            nested_teams_disabled
            and (
                (
                    provider == "mkl"
                    and threading == "gnu_openmp"
                    and runtime_compatible is True
                )
                or (
                    provider == "openblas"
                    and openblas_openmp_enabled is True
                    and openblas_openmp_runtime_compatible is True
                )
            )
        )
        if actual_threads > 1 and environment_policy == "unsafe":
            add_check(
                "host_worker_blas",
                "error",
                "multi-worker CBLAS was forced with the unsafe diagnostic override",
            )
        elif actual_threads > 1 and selected == "cblas" and automatic_cblas_safe:
            message = (
                "host worker contractions select OpenMP-enabled OpenBLAS"
                if provider == "openblas"
                else f"host worker contractions select {provider} {threading} CBLAS"
            )
            add_check(
                "host_worker_blas",
                "ok",
                message,
            )
        elif (
            actual_threads > 1
            and selected == "cblas"
            and provider == "openblas"
            and openblas_openmp_enabled is True
            and openblas_openmp_runtime_compatible is False
        ):
            add_check(
                "host_worker_blas",
                "error",
                "OpenBLAS and Kokkos use different OpenMP runtime libraries",
            )
        elif (
            actual_threads > 1
            and selected == "cblas"
            and provider == "openblas"
            and openblas_openmp_enabled is True
            and isinstance(omp_max_active_levels, int)
            and omp_max_active_levels > 1
        ):
            add_check(
                "host_worker_blas",
                "error",
                "multi-worker CBLAS permits nested OpenMP teams; set "
                "OMP_MAX_ACTIVE_LEVELS=1",
            )
        elif actual_threads > 1 and selected != "kokkos":
            add_check(
                "host_worker_blas",
                "error",
                "multi-worker OpenMP selected concurrent external CBLAS; "
                "use SYMMETRIX_HOST_WORKER_BLAS=automatic or off",
            )
        else:
            add_check(
                "host_worker_blas",
                "ok",
                f"host worker contractions select {selected}",
            )

        if provider == "mkl" and threading == "intel_openmp":
            add_check(
                "blas_provider",
                "error" if actual_threads > 1 and selected == "cblas" else "warning",
                "Intel-threaded MKL cannot share a GCC Kokkos OpenMP runtime; "
                "use Intel10_64lp or Intel10_64lp_seq",
            )
        elif provider == "mkl" and threading in {"sequential", "gnu_openmp"}:
            add_check(
                "blas_provider",
                "ok",
                f"MKL {threading} threading layer is loaded",
            )
        elif provider == "mkl":
            add_check(
                "blas_provider",
                "warning",
                f"MKL threading layer is {threading}; worker CBLAS is disabled",
            )
        elif provider == "openblas" and openblas_openmp_enabled is False:
            kokkos_threads = runtime.get("kokkos_concurrency")
            if not isinstance(kokkos_threads, int) or kokkos_threads <= 0:
                kokkos_threads = runtime.get("omp_max_threads")
            multithreaded = isinstance(kokkos_threads, int) and kokkos_threads > 1
            context = " for a multithreaded CPU Kokkos job" if multithreaded else ""
            add_check(
                "openblas_build",
                "warning",
                "loaded OpenBLAS is not OpenMP-enabled"
                f"{context}; host GEMMs are forced "
                "to one BLAS thread. Prefer an OpenMP-enabled OpenBLAS build",
            )
        elif (
            provider == "openblas"
            and openblas_openmp_enabled is True
            and openblas_openmp_runtime_compatible is False
        ):
            add_check(
                "openblas_build",
                "error",
                "loaded OpenBLAS uses a different OpenMP runtime; native Kokkos "
                "worker contractions are required",
            )
        elif (
            provider == "openblas"
            and openblas_openmp_enabled is True
            and openblas_openmp_runtime_compatible is True
        ):
            add_check(
                "openblas_build",
                "ok",
                "loaded OpenBLAS is OpenMP-enabled and runtime-compatible",
            )
        elif provider == "openblas":
            add_check(
                "openblas_build",
                "warning",
                "could not determine whether the loaded BLAS is an "
                "OpenMP-enabled OpenBLAS build",
            )
        else:
            add_check(
                "blas_provider",
                "warning",
                f"could not identify the loaded BLAS provider ({threading})",
            )

    if not report["ok"]:
        report["remediation"] = [
            "Use a standalone CPython environment without a legacy Anaconda DT_RPATH.",
            "Rebuild Symmetrix in the target compiler/runtime environment.",
            "For immediate diagnosis only, preload the matching compiler libgomp before Python starts.",
            "Do not statically link libgomp; another Python extension may still load a second runtime.",
        ]
    return report


def accelerator_runtime_diagnostics(
    *, native_module, backend: dict[str, Any]
) -> dict[str, Any]:
    """Initialize and probe a selected CUDA or HIP backend in this process."""

    expected_backend = str(backend.get("backend", ""))
    expected_architecture = (
        str(backend.get("architecture", "")).lower().replace("_", "")
    )
    expected_space = {"cuda": "Cuda", "hip": "HIP"}.get(expected_backend)
    if expected_space is None:
        raise ValueError(
            f"accelerator diagnostics require CUDA or HIP, got {expected_backend!r}"
        )

    report: dict[str, Any] = {
        "schema": "symmetrix.accelerator-runtime-diagnostics",
        "version": 1,
        "status": "ok",
        "ok": True,
        "python_executable": sys.executable,
        "backend": backend,
        "checks": [],
        "runtime": {},
    }

    def add_check(name: str, status: str, message: str) -> None:
        report["checks"].append({"name": name, "status": status, "message": message})
        if status == "error":
            report["ok"] = False
            report["status"] = "error"

    initialized_here = False
    try:
        initialized = bool(native_module._kokkos_is_initialized())
        if not initialized:
            native_module._init_kokkos()
            initialized_here = True
        add_check(
            "kokkos_initialization",
            "ok",
            "initialized Kokkos for this diagnostic"
            if initialized_here
            else "reused an already initialized Kokkos runtime",
        )

        execution_space = native_module._kokkos_default_execution_space()
        report["runtime"]["execution_space"] = execution_space
        if execution_space != expected_space:
            add_check(
                "execution_space",
                "error",
                f"extension reports {execution_space!r}, expected {expected_space!r}",
            )
        else:
            add_check("execution_space", "ok", f"Kokkos uses {execution_space}")

        environment = dict(native_module._execution_device_execution_environment())
        report["runtime"]["device_environment"] = environment
        actual_backend = str(environment.get("backend", ""))
        actual_architecture = (
            str(environment.get("architecture", "")).lower().replace("_", "")
        )
        if actual_backend != expected_backend:
            add_check(
                "device_backend",
                "error",
                f"runtime reports {actual_backend or 'unknown'}, expected {expected_backend}",
            )
        else:
            add_check("device_backend", "ok", f"runtime reports {actual_backend}")
        if actual_architecture != expected_architecture:
            add_check(
                "device_architecture",
                "error",
                f"runtime reports {actual_architecture or 'unknown'}, expected "
                f"{expected_architecture}",
            )
        else:
            add_check(
                "device_architecture", "ok", f"runtime reports {actual_architecture}"
            )

        sentinel = bool(native_module._kokkos_device_sentinel())
        report["runtime"]["device_sentinel"] = sentinel
        if sentinel:
            add_check("device_sentinel", "ok", "device kernel completed successfully")
        else:
            add_check("device_sentinel", "error", "device sentinel returned false")
    except Exception as error:  # noqa: BLE001 - native diagnostics must report failures.
        add_check("accelerator_runtime", "error", str(error))
    finally:
        if initialized_here:
            try:
                native_module._finalize_kokkos()
                report["runtime"]["finalized"] = True
            except Exception as error:  # noqa: BLE001 - preserve teardown failures.
                add_check("kokkos_finalization", "error", str(error))
    return report
