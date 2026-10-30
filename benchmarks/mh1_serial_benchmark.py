"""Benchmark the direct and reference MACE-MH-1 routings."""

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import pathlib
import platform
import shlex
import statistics
import subprocess
import sys
import time

THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_THREADS", "1")
BLAS_THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_BLAS_THREADS", "1")
KOKKOS_THREAD_VARIABLES = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
)
BLAS_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _configure_runtime_before_imports():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cpus")
    early_args, _ = parser.parse_known_args()
    if early_args.cpu is not None and early_args.cpus is not None:
        parser.error("--cpu and --cpus are mutually exclusive")

    for variable in KOKKOS_THREAD_VARIABLES:
        os.environ[variable] = THREAD_COUNT
    for variable in BLAS_THREAD_VARIABLES:
        os.environ[variable] = BLAS_THREAD_COUNT
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    requested_cpus = None
    if early_args.cpu is not None:
        requested_cpus = {early_args.cpu}
    elif early_args.cpus is not None:
        try:
            requested_cpus = {int(value) for value in early_args.cpus.split(",")}
        except ValueError:
            parser.error("--cpus must be a comma-separated integer list")
        if not requested_cpus:
            parser.error("--cpus cannot be empty")
    if requested_cpus is not None:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu/--cpus require sched_setaffinity support")
        try:
            os.sched_setaffinity(0, requested_cpus)
        except OSError as error:
            parser.error(f"could not apply requested CPU affinity: {error}")
        if os.sched_getaffinity(0) != requested_cpus:
            parser.error(f"could not enforce CPU affinity {sorted(requested_cpus)}")


def _explicit_native_module_name(extension_module, descriptor):
    legacy_cpu = (
        extension_module == "symmetrix"
        and descriptor.package == "symmetrix"
        and descriptor.module == "_native_cpu"
        and descriptor.backend == "cpu"
        and descriptor.descriptor_path == "bundled"
    )
    if legacy_cpu:
        return "symmetrix.symmetrix"
    if extension_module != descriptor.module:
        raise RuntimeError(
            "explicit extension module does not match selected backend: "
            f"{extension_module!r} != {descriptor.module!r}"
        )
    return f"{descriptor.package}.{descriptor.module}"


def _bootstrap_explicit_symmetrix():
    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension_path = os.environ.get("SYMMETRIX_EXTENSION")
    if not source_root or not extension_path:
        return
    import importlib.util

    package_dir = pathlib.Path(source_root).resolve() / "symmetrix/source/symmetrix"
    extension = pathlib.Path(extension_path).resolve()
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if type(finder).__name__ != "ScikitBuildRedirectingFinder"
    ]
    package_spec = importlib.util.spec_from_file_location(
        "symmetrix",
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["symmetrix"] = package
    package_spec.loader.exec_module(package)

    # Register the explicit binary under the selected descriptor's canonical
    # module name.  A synthetic ``symmetrix.<basename>`` name makes the normal
    # backend loader import the same pybind11 binary a second time.
    from symmetrix.backend_loader import select_backend

    descriptor = select_backend(package.__version__)
    extension_module = extension.name.split(".", 1)[0]
    native_module_name = _explicit_native_module_name(extension_module, descriptor)
    importlib.import_module(descriptor.package)
    native_spec = importlib.util.spec_from_file_location(native_module_name, extension)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules[native_module_name] = native
    native_spec.loader.exec_module(native)
    package.load_backend()


_configure_runtime_before_imports()
_bootstrap_explicit_symmetrix()


THREAD_VARIABLES = (
    KOKKOS_THREAD_VARIABLES
    + BLAS_THREAD_VARIABLES
    + (
        "OMP_PROC_BIND",
        "OMP_PLACES",
    )
)

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from symmetrix import Symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402

STREAMED_EDGE_ALIASES = {
    "all_interactions": "generic",
    "factorized": "direct",
    "direct_streamed": "direct",
}
STREAMED_EDGE_MODES = ("materialized", "generic", "direct")
BENCHMARK_PROPERTIES = ("energy", "forces", "stress")
MH1_PRODUCT_ROUTES = ("standard", "shared", "retained-monolithic-oracle")
REPLAY_ENVIRONMENT_VARIABLES = (
    "SYMMETRIX_SOURCE_ROOT",
    "SYMMETRIX_EXTENSION",
    "SYMMETRIX_BACKEND",
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "SYMMETRIX_JIT_NVRTC_LIBRARY",
    "SYMMETRIX_JIT_CUDA_JIT_BACKEND",
    "SYMMETRIX_JIT_HIP_JIT_BACKEND",
    "SYMMETRIX_JIT_CACHE",
    "SYMMETRIX_JIT_POLICY",
    "SYMMETRIX_M0_RTC_SCHEDULE",
    "SYMMETRIX_M1_RTC_SCHEDULE",
) + THREAD_VARIABLES


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _replay_provenance(args):
    if args.cpu is not None:
        requested_affinity = [args.cpu]
    elif args.cpus is not None:
        requested_affinity = sorted(int(value) for value in args.cpus.split(","))
    else:
        requested_affinity = None
    effective_affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "python_executable": sys.executable,
        "python_orig_argv": list(getattr(sys, "orig_argv", sys.argv)),
        "launcher": {
            "kind": (
                "in-process-sched-setaffinity"
                if requested_affinity is not None
                else "inherited-affinity"
            ),
            "requested_cpu_affinity": requested_affinity,
            "effective_cpu_affinity": effective_affinity,
        },
        "environment": {
            name: os.environ.get(name) for name in REPLAY_ENVIRONMENT_VARIABLES
        },
    }


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _system_sha256(atoms):
    return _json_sha256(
        {
            "numbers": atoms.numbers.tolist(),
            "positions": np.asarray(atoms.positions, dtype=np.float64).tolist(),
            "cell": np.asarray(atoms.cell, dtype=np.float64).tolist(),
            "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        }
    )


def _parse_properties(value):
    properties = tuple(item.strip() for item in value.split(",") if item.strip())
    if not properties:
        raise ValueError("--properties cannot be empty")
    if len(set(properties)) != len(properties):
        raise ValueError("--properties cannot contain duplicates")
    unsupported = sorted(set(properties).difference(BENCHMARK_PROPERTIES))
    if unsupported:
        raise ValueError(
            "--properties contains unsupported values: " + ", ".join(unsupported)
        )
    return properties


def _schedule_overrides():
    result = {}
    for operator in ("M0", "M1"):
        variable = f"SYMMETRIX_{operator}_RTC_SCHEDULE"
        requested = os.environ.get(variable)
        if requested == "auto":
            raise RuntimeError(
                f"{variable}=auto is invalid; unset it or select an admitted schedule"
            )
        result[operator] = {
            "environment": variable,
            "requested": requested,
            "effective": requested or "chunk32",
        }
    return result


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _git_tracked_diff_sha256():
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
        cwd=pathlib.Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _cpu_model():
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or None


def _native_build_metadata(extension_path):
    cache = extension_path.parent / "CMakeCache.txt"
    keys = {
        "CMAKE_BUILD_TYPE",
        "SYMMETRIX_KOKKOS",
        "SYMMETRIX_SPHERICART_CUDA",
        "Kokkos_ENABLE_OPENMP",
        "Kokkos_ENABLE_SERIAL",
        "Kokkos_ENABLE_CUDA",
        "Kokkos_ENABLE_HIP",
        "Kokkos_ENABLE_SYCL",
        "Kokkos_ENABLE_OPENMPTARGET",
        "KokkosKernels_ENABLE_TPL_BLAS",
    }
    values = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            declaration, value = line.split("=", 1)
            key = declaration.split(":", 1)[0]
            if key in keys:
                values[key] = value
    return {"cmake_cache": str(cache) if cache.is_file() else None, "values": values}


def _thread_affinities():
    task_directory = pathlib.Path("/proc/self/task")
    if not task_directory.is_dir():
        return None
    affinities = {}
    for task in task_directory.iterdir():
        status = task / "status"
        try:
            lines = status.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("Cpus_allowed_list"):
                affinities[task.name] = line.partition(":")[2].strip()
                break
    return affinities


def _current_rss_bytes():
    statm = pathlib.Path("/proc/self/statm")
    if not statm.is_file():
        raise RuntimeError("lifecycle RSS measurement requires Linux /proc/self/statm")
    resident_pages = int(statm.read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _samples_summary(samples, atom_count):
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "median_ms_per_atom": median_ms / atom_count,
        "median_us_per_atom": 1000.0 * median_ms / atom_count,
        "min_ms": min(samples),
        "min_ms_per_atom": min(samples) / atom_count,
        "max_ms": max(samples),
        "max_ms_per_atom": max(samples) / atom_count,
        "samples_ms": samples,
    }


def _md_timing_summary(samples_ms, atom_count):
    median_ms = statistics.median(samples_ms)
    return {
        "boundary": "one complete ASE VelocityVerlet NVE step",
        "steps": len(samples_ms),
        "samples_ms": samples_ms,
        "total_md_ms": sum(samples_ms),
        "median_ms": median_ms,
        "median_us_per_atom_per_step": 1000.0 * median_ms / atom_count,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "atom_steps_per_second": atom_count * 1000.0 / median_ms,
    }


def _peak_rss_bytes():
    status = pathlib.Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _gpu_process_memory_mib():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    used_mib = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            try:
                used_mib += int(fields[1])
            except ValueError:
                return None
    return used_mib


def _jit_metadata(calculator):
    return {
        "jit_requested": getattr(calculator, "jit_policy", None),
        "jit_status": getattr(calculator, "jit_status", None),
        "jit_reason": getattr(calculator, "jit_reason", None),
        "jit_diagnostics": getattr(calculator, "jit_diagnostics", None),
        "jit_cache_key": getattr(calculator, "jit_cache_key", None),
        "jit_artifact_path": getattr(calculator, "jit_artifact_path", None),
        "jit_artifact_id": getattr(calculator, "jit_artifact_id", None),
        "jit_variant_id": getattr(calculator, "jit_variant_id", None),
        "jit_compiler_request": getattr(calculator, "jit_compiler_request", None),
        "jit_compiler_backend": getattr(calculator, "jit_compiler_backend", None),
        "jit_forward_policy": getattr(calculator, "jit_forward_policy", None),
        "jit_source_policy": getattr(calculator, "jit_source_policy", None),
        "jit_edge_policy": getattr(calculator, "jit_edge_policy", None),
        "jit_origin": getattr(calculator, "jit_origin", None),
    }


def _mh1_retained_output_dimensions(calculator):
    """Read the retained interaction output dimensions without risking the run.

    The accessor describes the prepared node runtime, which only the direct
    routing builds; another routing has no retained width to report.  Reading it
    there aborted the process before the record could be written, so the driver
    reports the absence and its reason instead of reading it.
    """

    evaluator = calculator.evaluator
    if getattr(evaluator, "streamed_edges_mode", None) != "direct":
        return None, (
            "retained interaction output dimensions belong to the direct "
            "routing's prepared node runtime; this run selected "
            f"{calculator.streamed_edges!r}"
        )
    try:
        dimensions = list(
            evaluator.execution_mh1_retained_interaction_output_dimensions
        )
    except Exception as error:  # noqa: BLE001 - diagnostics must not abort a run
        return None, f"{type(error).__name__}: {error}"
    return dimensions, None


def _configure_mh1_product_route(calculator, route):
    if route == "retained-monolithic-oracle":
        raise RuntimeError(
            "the retained monolithic MH-1 product oracle was retired in JIT "
            "generation 19; use historical generation-18 benchmark records"
        )
    artifact_value = getattr(calculator, "jit_artifact_path", None)
    if not artifact_value:
        raise RuntimeError(f"{route} requires a generated MH-1 execution artifact")
    artifact = pathlib.Path(artifact_value).resolve()
    manifest_path = artifact.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    expected_artifact_sha256 = manifest.get("artifact_sha256")
    artifact_sha256 = _sha256(artifact)
    if expected_artifact_sha256 not in (artifact_sha256, f"sha256:{artifact_sha256}"):
        raise RuntimeError("generated MH-1 artifact checksum does not match manifest")
    key_inputs = manifest.get("key_inputs")
    if not isinstance(key_inputs, dict):
        raise RuntimeError("generation-19 release contract lacks manifest key_inputs")
    abi = key_inputs.get("abi")
    generation = key_inputs.get("jit_generation_version")
    try:
        original_plan = key_inputs["build"]["request"]["launch_plan"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "generation-19 release contract lacks a launch plan"
        ) from error
    if not (
        isinstance(abi, dict)
        and abi.get("version") == 5
        and generation == 19
        and original_plan.get("schema") == "symmetrix.jit.mh1.cuda-launch-plan/6"
        and original_plan.get("version") == 6
        and original_plan.get("split_phase_composition") is True
    ):
        raise RuntimeError(
            "generated MH-1 artifact does not satisfy the generation-19 release "
            "contract (device module ABI 5, launch-plan schema/version 6, and "
            "split_phase_composition=true are required)"
        )
    split_phase_composition = True
    selected_policy = calculator.execution_mh1_node_state_policy
    if original_plan.get("node_state_policy") != selected_policy:
        raise RuntimeError(
            "generated launch-plan node-state policy does not match calculator: "
            f"{original_plan.get('node_state_policy')!r} != {selected_policy!r}"
        )

    shared_modules = [
        {
            "layer": layer,
            "ready": bool(
                calculator.evaluator._execution_mh1_product_module_ready(layer)
            ),
            "artifact_id": (
                calculator.evaluator._execution_mh1_product_module_artifact_id(layer)
            ),
        }
        for layer in range(len(original_plan["nodes"]))
    ]
    shared_diagnostics = list(
        calculator.evaluator.execution_mh1_product_module_diagnostics
    )
    selected_plan = copy.deepcopy(original_plan)
    if route != "shared":
        raise ValueError(f"unsupported MH-1 product route: {route}")

    base_plan = copy.deepcopy(original_plan)
    return {
        "route": route,
        "external_product": True,
        "original_external_product": original_plan.get("external_product"),
        "split_phase_composition": split_phase_composition,
        "plan_schema": original_plan.get("schema"),
        "plan_version": original_plan.get("version"),
        "device_module_abi": abi,
        "jit_generation_version": generation,
        "node_state_policy": original_plan["node_state_policy"],
        "module_identity": original_plan["module_identity"],
        "module_path": str(artifact),
        "module_sha256": artifact_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "launch_plan_base_sha256": _json_sha256(base_plan),
        "selected_launch_plan_sha256": _json_sha256(selected_plan),
        "shared_product_modules": shared_modules,
        "shared_product_diagnostics": shared_diagnostics,
    }


def _property_outputs(results, properties):
    outputs = {}
    if "energy" in properties:
        energy = np.asarray(results["energy"])
        outputs["energy_eV"] = float(energy)
        outputs["energy_sha256"] = _array_sha256(energy)
    if "forces" in properties:
        forces = np.asarray(results["forces"])
        outputs.update(
            {
                "force_l2_eV_per_A": float(np.linalg.norm(forces)),
                "force_sum_eV_per_A": np.sum(forces, axis=0).tolist(),
                "forces_sha256": _array_sha256(forces),
            }
        )
    if "stress" in properties:
        stress = np.asarray(results["stress"])
        outputs.update(
            {
                "stress_eV_per_A3": stress.tolist(),
                "stress_sha256": _array_sha256(stress),
            }
        )
    return outputs


def _read_artifact_manifest(artifact_path):
    if not artifact_path:
        return None
    path = pathlib.Path(artifact_path)
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _product_artifact_provenance(module):
    artifact_path = pathlib.Path(module["artifact_path"])
    manifest_path = artifact_path.with_name("manifest.json")
    manifest = _read_artifact_manifest(artifact_path)
    if manifest is not None:
        key_inputs = manifest.get("key_inputs")
        if not isinstance(key_inputs, dict):
            raise RuntimeError("MH-1 product runtime manifest lacks key_inputs")
        if manifest.get("artifact_sha256") != _sha256(artifact_path):
            raise RuntimeError(
                "MH-1 product runtime manifest artifact hash does not match"
            )
        return {
            "kind": "runtime-manifest",
            "abi": key_inputs.get("abi"),
            "jit_generation_version": key_inputs.get("jit_generation_version"),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "catalog_path": None,
            "catalog_sha256": None,
            "catalog_entry_sha256": None,
        }
    if module.get("origin") != "prebuilt":
        raise RuntimeError("MH-1 runtime product manifest is unavailable")

    from symmetrix.generated_operator_core import (
        validated_installed_operator_catalog,
    )

    build_info = getattr(native_symmetrix, "_backend_build_info", lambda: {})()
    installed = build_info.get("prebuilt_operator_artifacts")
    native_path = pathlib.Path(native_symmetrix.__file__).resolve()
    if not isinstance(installed, dict):
        raise RuntimeError("MH-1 prebuilt product catalog provenance is unavailable")
    report, catalog, data_root = validated_installed_operator_catalog(
        native_path.parent, installed
    )
    matches = [
        entry
        for entry in catalog["artifacts"]
        if (data_root / entry["installed_relative_path"]).resolve()
        == artifact_path.resolve()
        and entry["operator"] == module["operator"]
        and entry["identity"]["artifact_id"] == module["artifact_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError("MH-1 prebuilt product has no unique catalog entry")
    entry = matches[0]
    return {
        "kind": "installed-catalog",
        "abi": entry["abi"],
        "jit_generation_version": entry["identity"].get("generation_revision"),
        "manifest_path": None,
        "manifest_sha256": None,
        "catalog_path": report["catalog_path"],
        "catalog_sha256": report["catalog_sha256"],
        "catalog_entry_sha256": _json_sha256(entry),
    }


def _parse_product_diagnostic(value):
    fields = {}
    for item in str(value).replace(":", " ", 1).split():
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        try:
            fields[key] = int(raw)
        except ValueError:
            fields[key] = raw
    return fields


def _normalize_product_timings(value, expected_binding=None):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("product timings must be a JSON object")
    source = value.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("product timings require a non-empty source")
    statistic = value.get("statistic")
    if (
        not isinstance(statistic, str)
        or not statistic.strip()
        or statistic == "unspecified"
    ):
        raise ValueError("product timings require a declared statistic")
    binding = value.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("product timings require a record/artifact binding")
    if expected_binding is not None:

        def require_expected(actual, expected, path="binding"):
            if isinstance(expected, dict):
                if not isinstance(actual, dict):
                    raise ValueError(f"product timing binding mismatch at {path}")
                for key, expected_value in expected.items():
                    if key not in actual:
                        raise ValueError(
                            f"product timing binding is missing {path}.{key}"
                        )
                    require_expected(actual[key], expected_value, f"{path}.{key}")
            elif isinstance(expected, list):
                if not isinstance(actual, list) or len(actual) != len(expected):
                    raise ValueError(f"product timing binding mismatch at {path}")
                for index, expected_value in enumerate(expected):
                    require_expected(actual[index], expected_value, f"{path}[{index}]")
            elif actual != expected:
                raise ValueError(f"product timing binding mismatch at {path}")

        require_expected(binding, expected_binding)
    for field in (
        "profiled_command",
        "attachment_command",
        "profiled_record_path",
        "profiled_record_sha256",
        "profile_command",
        "profile_path",
        "profile_sha256",
        "report_path",
        "report_sha256",
    ):
        if not isinstance(binding.get(field), str) or not binding[field].strip():
            raise ValueError(f"product timing binding requires {field}")
    for field in (
        "profile_sha256",
        "report_sha256",
        "profiled_command_sha256",
        "attachment_command_sha256",
        "profile_command_sha256",
        "profiled_record_sha256",
    ):
        digest = binding.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"product timing binding {field} must be SHA-256")
    for command in ("profiled_command", "attachment_command", "profile_command"):
        if _text_sha256(binding[command]) != binding[f"{command}_sha256"]:
            raise ValueError(
                f"product timing {command.replace('_', ' ')} hash does not match"
            )
    for field in ("model_sha256", "system_sha256"):
        digest = binding.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"product timing binding {field} must be SHA-256")
    if binding.get("properties") not in (
        ["energy", "forces", "stress"],
        ["energy", "forces"],
        ["energy"],
    ):
        raise ValueError("product timing binding properties are invalid")
    for label in ("profiled_record", "profile", "report"):
        path = pathlib.Path(binding[f"{label}_path"])
        if not path.is_file():
            raise ValueError(f"product timing {label} file does not exist")
        if _sha256(path) != binding[f"{label}_sha256"]:
            raise ValueError(f"product timing {label} hash does not match")
    layers = value.get("layers")
    if not isinstance(layers, list):
        raise ValueError("product timings require a layers list")
    normalized_layers = []
    seen = set()
    for item in layers:
        if not isinstance(item, dict) or not isinstance(item.get("layer"), int):
            raise ValueError("each product timing requires an integer layer")
        layer = item["layer"]
        if layer < 0 or layer in seen:
            raise ValueError("product timing layers must be unique and non-negative")
        seen.add(layer)
        normalized = {"layer": layer}
        for key in ("forward_us", "reverse_us"):
            timing = item.get(key)
            if not isinstance(timing, (int, float)) or not np.isfinite(timing):
                raise ValueError(f"product timing {key} must be finite")
            if timing < 0:
                raise ValueError(f"product timing {key} must be non-negative")
            normalized[key] = float(timing)
        normalized_layers.append(normalized)
    if statistic.strip() != "median":
        raise ValueError("Nsight product timing reports currently require median")
    report_path = pathlib.Path(binding["report_path"])
    try:
        with report_path.open(newline="") as handle:
            report_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise ValueError(
            f"product timing profiler report is unreadable: {error}"
        ) from error
    required_report_fields = {"Name", "Med (ns)"}
    if not report_rows or not required_report_fields.issubset(report_rows[0]):
        raise ValueError("product timing profiler report is not cuda_gpu_kern_sum CSV")
    rows_by_name = {}
    for row in report_rows:
        name = row.get("Name")
        if name in rows_by_name:
            raise ValueError("product timing profiler report has duplicate kernel rows")
        rows_by_name[name] = row
    bound_layers = binding.get("layers")
    if not isinstance(bound_layers, list):
        raise ValueError("product timing binding requires layers")
    bound_layers_by_index = {
        item.get("layer"): item for item in bound_layers if isinstance(item, dict)
    }
    if len(bound_layers_by_index) != len(bound_layers):
        raise ValueError("product timing binding layers are invalid or duplicated")
    derived_layers = []
    for item in normalized_layers:
        bound_layer = bound_layers_by_index.get(item["layer"])
        if bound_layer is None:
            raise ValueError("product timing binding lacks a declared layer")
        kernels = bound_layer.get("profile_kernels")
        if not isinstance(kernels, dict):
            raise ValueError("product timing binding requires profile kernel names")
        derived = {"layer": item["layer"]}
        for direction in ("forward", "reverse"):
            kernel = kernels.get(direction)
            row = rows_by_name.get(kernel)
            if not isinstance(kernel, str) or row is None:
                raise ValueError(
                    "product timing profiler report lacks a bound kernel row"
                )
            try:
                derived[f"{direction}_us"] = float(row["Med (ns)"]) / 1000.0
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "product timing profiler report has an invalid median"
                ) from error
        derived_layers.append(derived)
    if derived_layers != normalized_layers:
        raise ValueError(
            "product timing profiler report does not match declared timings"
        )
    try:
        profiled_record = json.loads(
            pathlib.Path(binding["profiled_record_path"]).read_text()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"profiled benchmark record is unreadable: {error}") from error
    profiled_metadata = profiled_record.get("metadata", {})
    profiled_systems = profiled_record.get("systems", [])
    if len(profiled_systems) != 1:
        raise ValueError("profiled benchmark record must contain exactly one system")
    profiled_system = profiled_systems[0]
    for actual, expected, label in (
        (profiled_metadata.get("git_revision"), binding.get("git_revision"), "commit"),
        (
            profiled_metadata.get("exact_command"),
            binding.get("profiled_command"),
            "command",
        ),
        (
            profiled_metadata.get("native_build_info", {}).get("source_content_sha256"),
            binding.get("source_content_sha256"),
            "source hash",
        ),
        (
            profiled_metadata.get("native_build_info", {}).get(
                "native_source_content_sha256"
            ),
            binding.get("native_source_content_sha256"),
            "native source hash",
        ),
        (
            profiled_metadata.get("native_extension_sha256"),
            binding.get("native_extension_sha256"),
            "extension hash",
        ),
        (
            profiled_metadata.get("model", {}).get("sha256"),
            binding.get("model_sha256"),
            "model",
        ),
        (profiled_system.get("system_sha256"), binding.get("system_sha256"), "system"),
        (
            profiled_system.get("timed_properties"),
            binding.get("properties"),
            "properties",
        ),
        (
            {
                "warmups": profiled_metadata.get("warmups"),
                "samples": profiled_metadata.get("repeats"),
            },
            binding.get("sample_scope"),
            "sample scope",
        ),
    ):
        if actual != expected:
            raise ValueError(
                f"profiled benchmark record {label} binding does not match"
            )
    profiled_runtime = profiled_system.get("mh1_product_runtime", {})
    if {
        key: profiled_runtime.get("full_plugin", {}).get(key)
        for key in (
            "artifact_id",
            "artifact_sha256",
            "manifest_sha256",
            "module_identity",
        )
    } != binding["full_plugin"]:
        raise ValueError("profiled benchmark record full-plugin binding does not match")
    profiled_layers = profiled_runtime.get("layers", [])
    if [
        (
            item.get("layer"),
            item.get("artifact_id"),
            item.get("artifact_sha256"),
            item.get("artifact_path"),
            item.get("manifest_sha256"),
            item.get("origin"),
            item.get("artifact_provenance"),
        )
        for item in profiled_layers
    ] != [
        (
            item["layer"],
            item["artifact_id"],
            item["artifact_sha256"],
            item["artifact_path"],
            item["manifest_sha256"],
            item["origin"],
            item["artifact_provenance"],
        )
        for item in binding["layers"]
    ]:
        raise ValueError("profiled benchmark record layer binding does not match")
    if (
        not isinstance(bound_layers, list)
        or {item.get("layer") for item in bound_layers if isinstance(item, dict)}
        != seen
    ):
        raise ValueError("product timing layer set does not match artifact binding")
    return {
        "source": source.strip(),
        "collection_scope": value.get("collection_scope"),
        "statistic": statistic.strip(),
        "binding": binding,
        "layers": normalized_layers,
        "instrumentation": "external-profile",
        "evaluator_samples_include_product_work": True,
        "profile_timings_derived_from_evaluator_samples": False,
        "intrusive_timed_instrumentation": False,
        "collected_by_benchmark": False,
    }


def _require_product_timing_system_binding(product_timings, metadata, record):
    if product_timings is None:
        return
    binding = product_timings["binding"]
    expected = {
        "model_sha256": metadata["model"]["sha256"],
        "system_sha256": record["system_sha256"],
        "properties": record["timed_properties"],
        "attachment_command": metadata["exact_command"],
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise RuntimeError(
                f"product timing {field} does not match the benchmark record"
            )


def _mh1_product_runtime(
    calculator, layer_count, product_timings=None, launch_counter_window=None
):
    evaluator = calculator.evaluator
    diagnostics = list(evaluator.execution_mh1_product_module_diagnostics)
    parsed_diagnostics = {
        int(value.split(":", 1)[0].removeprefix("product_")): (
            value,
            _parse_product_diagnostic(value),
        )
        for value in diagnostics
        if value.startswith("product_") and ":" in value
    }
    module_records = getattr(calculator, "jit_operator_modules", {})

    def layer_record(layer):
        ready = bool(evaluator._execution_mh1_product_module_ready(layer))
        module = dict(module_records.get(f"product_{layer}", {}))
        artifact_path = module.get("artifact_path")
        provenance = _product_artifact_provenance(module)
        diagnostic, geometry = parsed_diagnostics.get(layer, (None, {}))
        realized_geometry_keys = (
            "blocks_per_compute_unit",
            "persistent_block_cap",
            "forward_logical_tiles",
            "forward_tiles_per_block",
            "forward_required_blocks",
            "forward_realized_blocks",
            "reverse_logical_tiles",
            "reverse_tiles_per_block",
            "reverse_required_blocks",
            "reverse_realized_blocks",
        )
        required_module_fields = (
            "operator",
            "origin",
            "artifact_id",
            "artifact_path",
            "schedule",
            "backend",
            "target",
            "precision",
            "generation_fingerprint",
            "semantic_fingerprint",
            "structure_fingerprint",
            "channels",
            "type_count",
            "input_components",
            "output_components",
            "correlation",
            "term_count",
            "coefficient_count",
        )
        missing = [
            field for field in required_module_fields if module.get(field) is None
        ]
        if missing:
            raise RuntimeError(
                f"MH-1 product layer {layer} evidence lacks required fields: "
                + ", ".join(missing)
            )
        coefficient_count = module["coefficient_count"]
        expected_coefficients = module["term_count"] * module["channels"]
        if coefficient_count != expected_coefficients:
            raise RuntimeError(
                f"MH-1 product layer {layer} coefficient_count {coefficient_count} "
                f"does not equal term_count * channels ({expected_coefficients})"
            )
        abi = provenance["abi"]
        generation = provenance["jit_generation_version"]
        if not isinstance(abi, dict) or abi.get("version") != 3 or generation != 19:
            raise RuntimeError(
                f"MH-1 product layer {layer} manifest is not ABI 3 generation 19"
            )
        counters = (
            launch_counter_window.get(layer)
            if launch_counter_window is not None
            else None
        )
        record = {
            "layer": layer,
            "ready": ready,
            "artifact_id": (
                evaluator._execution_mh1_product_module_artifact_id(layer)
                if ready
                else None
            ),
            "forward_launch_count": (
                counters["timed_forward_launch_count"]
                if counters is not None
                else evaluator._execution_mh1_product_module_forward_launch_count(layer)
            ),
            "reverse_launch_count": (
                counters["timed_reverse_launch_count"]
                if counters is not None
                else evaluator._execution_mh1_product_module_reverse_launch_count(layer)
            ),
            "launch_counter_scope": counters,
            "operator": module.get("operator"),
            "origin": module.get("origin"),
            "artifact_path": artifact_path,
            "artifact_sha256": (
                _sha256(pathlib.Path(artifact_path))
                if artifact_path and pathlib.Path(artifact_path).is_file()
                else None
            ),
            "artifact_size_bytes": (
                pathlib.Path(artifact_path).stat().st_size
                if artifact_path and pathlib.Path(artifact_path).is_file()
                else None
            ),
            "cache_key": module.get("cache_key"),
            "schedule": module.get("schedule"),
            "schedule_revision": module.get("schedule_revision"),
            "manifest_path": provenance["manifest_path"],
            "manifest_sha256": provenance["manifest_sha256"],
            "artifact_provenance": provenance,
            "abi": abi,
            "jit_generation_version": generation,
            "backend": module.get("backend"),
            "target": module.get("target"),
            "precision": module.get("precision"),
            "identity": {
                key: module.get(key)
                for key in (
                    "generation_fingerprint",
                    "semantic_fingerprint",
                    "structure_fingerprint",
                )
            },
            "contract_dimensions": {
                key: module.get(key)
                for key in (
                    "channels",
                    "type_count",
                    "input_components",
                    "output_components",
                    "correlation",
                    "term_count",
                    "coefficient_count",
                )
            },
            "persistent_blocks_per_compute_unit": module.get(
                "persistent_blocks_per_compute_unit"
            ),
            "realized_geometry": {
                key: geometry.get(key) for key in realized_geometry_keys
            },
            "diagnostic": diagnostic,
            "memory_accounting": {
                "scratch_bytes": None,
                "retained_bytes": None,
                "unavailable_reason": (
                    "native accounting is reported for the composed MH-1 arena, "
                    "not apportioned to individual common product modules"
                ),
            },
        }
        if product_timings is not None:
            record["profile_timing"] = next(
                (
                    timing
                    for timing in product_timings["layers"]
                    if timing["layer"] == layer
                ),
                None,
            )
        return record

    plugin_path = getattr(calculator, "jit_artifact_path", None)
    plugin_manifest = _read_artifact_manifest(plugin_path)
    if plugin_manifest is None:
        raise RuntimeError("MH-1 full-plugin manifest is unavailable")
    plugin_inputs = plugin_manifest.get("key_inputs")
    if not isinstance(plugin_inputs, dict):
        raise RuntimeError("MH-1 full-plugin manifest lacks key_inputs")
    plugin_build = plugin_inputs.get("build", {})
    plugin_request = plugin_build.get("request", {})
    plugin_plan = plugin_request.get("launch_plan", {})
    required_plugin_values = {
        "artifact_id": getattr(calculator, "jit_artifact_id", None),
        "artifact_path": plugin_path,
        "abi": plugin_inputs.get("abi"),
        "jit_generation_version": plugin_inputs.get("jit_generation_version"),
        "module_identity": plugin_plan.get("module_identity"),
        "launch_plan_schema": plugin_plan.get("schema"),
        "launch_plan_version": plugin_plan.get("version"),
        "split_phase_composition": plugin_plan.get("split_phase_composition"),
    }
    missing_plugin = [
        key for key, value in required_plugin_values.items() if value is None
    ]
    if missing_plugin:
        raise RuntimeError(
            "MH-1 full-plugin evidence lacks required fields: "
            + ", ".join(missing_plugin)
        )
    return {
        "full_plugin": {
            "origin": getattr(calculator, "jit_origin", None),
            "artifact_id": getattr(calculator, "jit_artifact_id", None),
            "artifact_path": plugin_path,
            "artifact_sha256": (
                _sha256(pathlib.Path(plugin_path))
                if plugin_path and pathlib.Path(plugin_path).is_file()
                else None
            ),
            "manifest_sha256": (
                _sha256(pathlib.Path(plugin_path).with_name("manifest.json"))
                if plugin_path
                and pathlib.Path(plugin_path).with_name("manifest.json").is_file()
                else None
            ),
            "abi": plugin_inputs.get("abi"),
            "jit_generation_version": plugin_inputs.get(
                "jit_generation_version", plugin_build.get("jit_generation_version")
            ),
            "module_identity": plugin_plan.get("module_identity"),
            "launch_plan_schema": plugin_plan.get("schema"),
            "launch_plan_version": plugin_plan.get("version"),
            "split_phase_composition": plugin_plan.get("split_phase_composition"),
            "prebuilt": getattr(calculator, "jit_origin", None) == "prebuilt",
        },
        "layers": [layer_record(layer) for layer in range(layer_count)],
        "diagnostics": diagnostics,
        "fallback_count": getattr(
            evaluator, "factorized_fallback_evaluation_count", None
        ),
        "product_timings": product_timings,
    }


def _product_timing_binding(calculator, layer_count, extension_path):
    evidence = _mh1_product_runtime(calculator, layer_count)
    build_info = getattr(native_symmetrix, "_backend_build_info", lambda: {})()
    command = shlex.join(sys.argv)
    return {
        "git_revision": _git_revision(),
        "source_content_sha256": build_info.get("source_content_sha256"),
        "native_source_content_sha256": build_info.get("native_source_content_sha256"),
        "native_extension_sha256": _sha256(extension_path),
        "attachment_command": command,
        "attachment_command_sha256": _text_sha256(command),
        "full_plugin": {
            key: evidence["full_plugin"].get(key)
            for key in (
                "artifact_id",
                "artifact_sha256",
                "manifest_sha256",
                "module_identity",
            )
        },
        "layers": [
            {
                key: layer.get(key)
                for key in (
                    "layer",
                    "operator",
                    "artifact_id",
                    "artifact_sha256",
                    "artifact_path",
                    "manifest_sha256",
                    "origin",
                    "artifact_provenance",
                )
            }
            for layer in evidence["layers"]
        ],
    }


def _benchmark(calculator, atoms, warmups, repeats, include_ase, *, properties):
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], inputs[5].flatten(), inputs[6])
    prepare_graph = getattr(calculator.evaluator, "_prepare_factorized_graph", None)
    compute_prepared = getattr(
        calculator.evaluator, "_compute_prepared_factorized", None
    )
    prepared = (
        getattr(calculator.evaluator, "streamed_edges_mode", None) == "direct"
        and callable(prepare_graph)
        and callable(compute_prepared)
    )
    if prepared:
        graph_generation = prepare_graph(*inputs[:5])

        def evaluate_native():
            compute_prepared(
                graph_generation,
                np.ascontiguousarray(inputs[5]).reshape(-1),
                np.ascontiguousarray(inputs[6]),
            )
            return calculator._collect_mace_results(
                atoms, inputs, properties, graph_generation
            )
    else:

        def evaluate_native():
            calculator.evaluator.compute_node_energies_forces(*native_args)
            return calculator._collect_mace_results(atoms, inputs, properties)

    rss_before = _current_rss_bytes()
    # The reference routing executes single-thread on the host, so only the
    # remaining routings can run on a device execution space.
    cuda = (
        calculator.streamed_edges != "reference"
        and native_symmetrix._kokkos_default_execution_space() == "Cuda"
    )
    gpu_memory_before = _gpu_process_memory_mib() if cuda else None
    synchronize = getattr(calculator.evaluator, "fence", lambda: None)
    product_layers = sorted(
        int(key.removeprefix("product_"))
        for key in getattr(calculator, "jit_operator_modules", {})
        if key.startswith("product_")
    )

    def product_counters():
        return {
            layer: {
                "forward": calculator.evaluator._execution_mh1_product_module_forward_launch_count(
                    layer
                ),
                "reverse": calculator.evaluator._execution_mh1_product_module_reverse_launch_count(
                    layer
                ),
            }
            for layer in product_layers
        }

    counters_before = product_counters()
    for _ in range(warmups):
        synchronize()
        evaluate_native()
        synchronize()
    counters_after_warmups = product_counters()
    evaluator_samples = []
    evaluator_results = None
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        evaluator_results = evaluate_native()
        synchronize()
        evaluator_samples.append(1000.0 * (time.perf_counter() - start))
    assert evaluator_results is not None
    counters_after_samples = product_counters()
    product_launch_counter_window = {
        layer: {
            "scope": "benchmark-warmups-and-timed-samples",
            "warmups": warmups,
            "samples": repeats,
            "warmup_forward_launch_count": (
                counters_after_warmups[layer]["forward"]
                - counters_before[layer]["forward"]
            ),
            "warmup_reverse_launch_count": (
                counters_after_warmups[layer]["reverse"]
                - counters_before[layer]["reverse"]
            ),
            "timed_forward_launch_count": (
                counters_after_samples[layer]["forward"]
                - counters_after_warmups[layer]["forward"]
            ),
            "timed_reverse_launch_count": (
                counters_after_samples[layer]["reverse"]
                - counters_after_warmups[layer]["reverse"]
            ),
            "total_forward_launch_count": (
                counters_after_samples[layer]["forward"]
                - counters_before[layer]["forward"]
            ),
            "total_reverse_launch_count": (
                counters_after_samples[layer]["reverse"]
                - counters_before[layer]["reverse"]
            ),
        }
        for layer in product_layers
    }

    ase_samples = []
    if include_ase:
        for _ in range(warmups):
            synchronize()
            calculator.calculate(atoms.copy(), properties=list(properties))
            synchronize()
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            calculator.calculate(atoms.copy(), properties=list(properties))
            synchronize()
            ase_samples.append(1000.0 * (time.perf_counter() - start))

    edge_workspace_bytes = getattr(calculator.evaluator, "edge_workspace_bytes", None)
    selected_e3_linear_backend = getattr(
        calculator.evaluator, "selected_e3_linear_backend", None
    )
    retained_output_dimensions, retained_output_dimensions_unavailable = (
        _mh1_retained_output_dimensions(calculator)
    )
    return {
        "atoms": len(atoms),
        "model_cutoff_A": calculator.cutoff,
        "neighbor_skin_A": calculator.neighbor_skin,
        "effective_neighbor_cutoff_A": calculator.cutoff + calculator.neighbor_skin,
        "directed_edges": len(inputs[6]),
        "graph_dimensions": {
            "atoms": len(atoms),
            "directed_edges": len(inputs[6]),
            "model_cutoff_A": calculator.cutoff,
            "neighbor_skin_A": calculator.neighbor_skin,
            "effective_neighbor_cutoff_A": calculator.cutoff + calculator.neighbor_skin,
        },
        "system_sha256": _system_sha256(atoms),
        "timed_properties": list(properties),
        "prepared_graph": prepared,
        **_jit_metadata(calculator),
        "execution_mh1_execution_backend": getattr(
            calculator.evaluator, "execution_mh1_execution_backend", None
        ),
        "jit_mh1_host_plugin_ready": getattr(
            calculator.evaluator, "jit_mh1_host_plugin_ready", None
        ),
        "jit_mh1_host_plugin_v4_ready": getattr(
            calculator.evaluator, "jit_mh1_host_plugin_v4_ready", None
        ),
        "jit_mh1_host_plugin_artifact_id": getattr(
            calculator.evaluator, "jit_mh1_host_plugin_artifact_id", None
        ),
        "jit_mh1_cuda_plugin_ready": getattr(
            calculator.evaluator, "jit_mh1_cuda_plugin_ready", None
        ),
        "jit_mh1_cuda_plugin_v4_ready": getattr(
            calculator.evaluator, "jit_mh1_cuda_plugin_v4_ready", None
        ),
        "jit_mh1_cuda_plugin_artifact_id": getattr(
            calculator.evaluator, "jit_mh1_cuda_plugin_artifact_id", None
        ),
        "factorized_graph_generation": getattr(
            calculator.evaluator, "factorized_graph_generation", None
        ),
        "factorized_prepared_graph_count": getattr(
            calculator.evaluator, "factorized_prepared_graph_count", None
        ),
        "factorized_prepared_evaluation_count": getattr(
            calculator.evaluator, "factorized_prepared_evaluation_count", None
        ),
        "factorized_fallback_evaluation_count": getattr(
            calculator.evaluator, "factorized_fallback_evaluation_count", None
        ),
        "factorized_topology_validation_count": getattr(
            calculator.evaluator, "factorized_topology_validation_count", None
        ),
        "factorized_topology_validation_skip_count": getattr(
            calculator.evaluator,
            "factorized_topology_validation_skip_count",
            None,
        ),
        "factorized_preparation_fence_count": getattr(
            calculator.evaluator, "factorized_preparation_fence_count", None
        ),
        "factorized_evaluation_fence_count": getattr(
            calculator.evaluator, "factorized_evaluation_fence_count", None
        ),
        "factorized_stage_fence_count": getattr(
            calculator.evaluator, "factorized_stage_fence_count", None
        ),
        "factorized_zbl_evaluator_stream_launch_count": getattr(
            calculator.evaluator,
            "factorized_zbl_evaluator_stream_launch_count",
            None,
        ),
        "execution_geometry_capacity_edges": getattr(
            calculator.evaluator, "execution_geometry_capacity_edges", None
        ),
        "execution_geometry_workspace_bytes": getattr(
            calculator.evaluator, "execution_geometry_workspace_bytes", None
        ),
        "execution_geometry_allocation_count": getattr(
            calculator.evaluator, "execution_geometry_allocation_count", None
        ),
        "execution_geometry_copy_count": getattr(
            calculator.evaluator, "execution_geometry_copy_count", None
        ),
        "execution_sphericart_launch_count": getattr(
            calculator.evaluator, "execution_sphericart_launch_count", None
        ),
        "execution_sphericart_async_launch_count": getattr(
            calculator.evaluator, "execution_sphericart_async_launch_count", None
        ),
        "execution_mh1_generated_conditioning_forward_launch_count": getattr(
            calculator.evaluator,
            "execution_mh1_generated_conditioning_forward_launch_count",
            None,
        ),
        "execution_mh1_generated_conditioning_reverse_launch_count": getattr(
            calculator.evaluator,
            "execution_mh1_generated_conditioning_reverse_launch_count",
            None,
        ),
        "execution_mh1_scratch_budget_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_budget_bytes", None
        ),
        "execution_mh1_scratch_minimum_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_minimum_bytes", None
        ),
        "execution_mh1_scratch_planned_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_planned_bytes", None
        ),
        "execution_mh1_scratch_retained_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_retained_bytes", None
        ),
        "execution_mh1_scratch_recomputed_bytes": getattr(
            calculator.evaluator, "execution_mh1_scratch_recomputed_bytes", None
        ),
        "execution_mh1_scratch_recomputed_layer_count": getattr(
            calculator.evaluator,
            "execution_mh1_scratch_recomputed_layer_count",
            None,
        ),
        "execution_mh1_scratch_budget_satisfied": getattr(
            calculator.evaluator, "execution_mh1_scratch_budget_satisfied", None
        ),
        "execution_mh1_conditioning_recomputation_count": getattr(
            calculator.evaluator,
            "execution_mh1_conditioning_recomputation_count",
            None,
        ),
        "execution_mh1_retained_interaction_output_bytes": getattr(
            calculator.evaluator,
            "execution_mh1_retained_interaction_output_bytes",
            None,
        ),
        "execution_mh1_retained_pre_gate_bytes": getattr(
            calculator.evaluator, "execution_mh1_retained_pre_gate_bytes", None
        ),
        "execution_mh1_product_tile_scratch_bytes": getattr(
            calculator.evaluator, "execution_mh1_product_tile_scratch_bytes", None
        ),
        "execution_mh1_layer_up_adjoint_storage_bytes": getattr(
            calculator.evaluator,
            "execution_mh1_layer_up_adjoint_storage_bytes",
            None,
        ),
        "execution_mh1_up_adjoint_scratch_bytes": getattr(
            calculator.evaluator, "execution_mh1_up_adjoint_scratch_bytes", None
        ),
        "execution_mh1_up_adjoint_scratch_reuse_count": getattr(
            calculator.evaluator,
            "execution_mh1_up_adjoint_scratch_reuse_count",
            None,
        ),
        "execution_mh1_retained_interaction_output_dimensions": retained_output_dimensions,
        "execution_mh1_retained_interaction_output_dimensions_unavailable": (
            retained_output_dimensions_unavailable
        ),
        "evaluator": _samples_summary(evaluator_samples, len(atoms)),
        "_product_launch_counter_window": product_launch_counter_window,
        "ase": _samples_summary(ase_samples, len(atoms)) if ase_samples else None,
        "rss_before_mib": rss_before / 2**20,
        "rss_after_mib": _current_rss_bytes() / 2**20,
        "peak_rss_mib": (
            _peak_rss_bytes() / 2**20 if _peak_rss_bytes() is not None else None
        ),
        "gpu_process_memory_before_mib": gpu_memory_before,
        "gpu_process_memory_after_mib": _gpu_process_memory_mib() if cuda else None,
        "edge_workspace_rows": getattr(
            calculator.evaluator, "edge_workspace_rows", None
        ),
        "edge_workspace_bytes": edge_workspace_bytes,
        "edge_workspace_mib": (
            edge_workspace_bytes / 2**20 if edge_workspace_bytes is not None else None
        ),
        "conditioned_mlp_workspace_bytes": getattr(
            calculator.evaluator, "conditioned_mlp_workspace_bytes", None
        ),
        "e3_linear_backend": getattr(calculator.evaluator, "e3_linear_backend", None),
        "selected_e3_linear_backend": (
            selected_e3_linear_backend(len(atoms))
            if selected_e3_linear_backend is not None
            else None
        ),
        "tensor_product_backend": getattr(
            calculator.evaluator, "tensor_product_backend", None
        ),
        "tensor_product_execution_backend": getattr(
            calculator.evaluator, "tensor_product_execution_backend", None
        ),
        "tensor_product_channel_team_size": getattr(
            calculator.evaluator, "tensor_product_channel_team_size", None
        ),
        "tensor_product_harmonic_team_size": getattr(
            calculator.evaluator, "tensor_product_harmonic_team_size", None
        ),
        "linear_workspace_bytes": getattr(
            calculator.evaluator, "linear_workspace_bytes", None
        ),
        "tensor_workspace_bytes": getattr(
            calculator.evaluator, "tensor_workspace_bytes", None
        ),
        "product_workspace_bytes": getattr(
            calculator.evaluator, "product_workspace_bytes", None
        ),
        "node_workspace_bytes": getattr(
            calculator.evaluator, "node_workspace_bytes", None
        ),
        "precision_workspace_bytes": getattr(
            calculator.evaluator, "precision_workspace_bytes", edge_workspace_bytes
        ),
        **_property_outputs(evaluator_results, properties),
    }


def _model_metadata(path):
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _parse_sizes(value, parser, option):
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError:
        parser.error(f"{option} must be a comma-separated integer list")
    if not sizes or any(size < 1 for size in sizes):
        parser.error(f"{option} values must be positive")
    return sizes


def _aln_repeat_from_atom_count(atom_count):
    if atom_count % 4 != 0:
        raise ValueError(
            f"AlN atom count {atom_count} is not a wurtzite 4*n^3 supercell"
        )
    repeat = round((atom_count // 4) ** (1.0 / 3.0))
    if repeat < 1 or 4 * repeat**3 != atom_count:
        raise ValueError(
            f"AlN atom count {atom_count} is not a wurtzite 4*n^3 supercell"
        )
    return repeat


def _lifecycle(calculator, sizes, cycles):
    systems = [bulk("Si", "diamond", a=5.43).repeat((size,) * 3) for size in sizes]
    for _ in range(3):
        for atoms in systems:
            calculator.calculate(
                atoms.copy(), properties=["energy", "forces", "stress"]
            )
    rss_bytes = []
    for _ in range(cycles):
        for atoms in systems:
            calculator.calculate(
                atoms.copy(), properties=["energy", "forces", "stress"]
            )
        rss_bytes.append(_current_rss_bytes())
    return {
        "sizes": sizes,
        "atoms": [len(atoms) for atoms in systems],
        "cycles": cycles,
        "evaluations": cycles * len(systems),
        "rss_mib": [value / 2**20 for value in rss_bytes],
        "min_rss_mib": min(rss_bytes) / 2**20,
        "max_rss_mib": max(rss_bytes) / 2**20,
        "growth_mib": (max(rss_bytes) - min(rss_bytes)) / 2**20,
    }


def _build_md_atoms(repeat, seed, temperature_K, position_noise_std_A):
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(
        scale=position_noise_std_A, size=atoms.positions.shape
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms, preserve_temperature=True)
    return atoms


def _finalize_benchmark_runtime(native_module, systems):
    """Release all ASE-held owners and fail closed on a dirty Kokkos teardown."""

    for atoms in systems:
        atoms.calc = None
    gc.collect()
    live_objects = int(native_module._kokkos_live_object_count())
    if live_objects != 0:
        raise RuntimeError(
            f"benchmark teardown retained {live_objects} live Kokkos object(s)"
        )
    native_module._finalize_kokkos()
    if native_module._kokkos_is_initialized():
        raise RuntimeError("benchmark teardown did not finalize Kokkos")


def _md_runtime_snapshot(calculator):
    evaluator = calculator.evaluator
    cache = getattr(calculator, "_neighbor_cache", None)
    return {
        "neighbor_cache_build_count": int(calculator.neighbor_cache_build_count),
        "neighbor_cache_reuse_count": int(calculator.neighbor_cache_reuse_count),
        "neighbor_cache_geometry_update_count": int(
            calculator.neighbor_cache_geometry_update_count
        ),
        "candidate_directed_edges": (
            len(cache.receivers) if cache is not None else None
        ),
        "active_directed_edges": getattr(
            evaluator, "execution_active_edge_count", None
        ),
        "factorized_prepared_graph_count": getattr(
            evaluator, "factorized_prepared_graph_count", None
        ),
        "factorized_prepared_evaluation_count": getattr(
            evaluator, "factorized_prepared_evaluation_count", None
        ),
        "factorized_fallback_evaluation_count": getattr(
            evaluator, "factorized_fallback_evaluation_count", None
        ),
        "factorized_schedule_build_count": getattr(
            evaluator, "factorized_schedule_build_count", None
        ),
        "factorized_schedule_entries": getattr(
            evaluator, "factorized_schedule_entries", None
        ),
        "edge_workspace_bytes": getattr(evaluator, "edge_workspace_bytes", None),
        "execution_geometry_workspace_bytes": getattr(
            evaluator, "execution_geometry_workspace_bytes", None
        ),
        "node_workspace_bytes": getattr(evaluator, "node_workspace_bytes", None),
        "precision_workspace_bytes": getattr(
            evaluator, "precision_workspace_bytes", None
        ),
    }


def _benchmark_md(
    calculator,
    atoms,
    *,
    steps,
    timestep_fs,
    seed,
    requested_temperature_K,
    position_noise_std_A,
):
    from ase import units
    from ase.md.verlet import VelocityVerlet

    synchronize = getattr(calculator.evaluator, "fence", lambda: None)
    atoms.calc = calculator
    initial_positions = np.asarray(atoms.positions, dtype=float).copy()
    actual_initial_temperature_K = float(atoms.get_temperature())

    synchronize()
    initial_start = time.perf_counter()
    initial_forces = np.asarray(atoms.get_forces(), dtype=float).copy()
    initial_energy = float(atoms.get_potential_energy())
    synchronize()
    initial_evaluation_ms = 1000.0 * (time.perf_counter() - initial_start)
    initial_total_energy = initial_energy + float(atoms.get_kinetic_energy())
    runtime_after_initial = _md_runtime_snapshot(calculator)

    dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=None)
    samples_ms = []
    trajectory = []
    for step in range(1, steps + 1):
        synchronize()
        start = time.perf_counter()
        dynamics.run(1)
        synchronize()
        samples_ms.append(1000.0 * (time.perf_counter() - start))

        forces = np.asarray(calculator.results["forces"], dtype=float)
        potential = float(calculator.results["energy"])
        kinetic = float(atoms.get_kinetic_energy())
        runtime = _md_runtime_snapshot(calculator)
        trajectory.append(
            {
                "step": step,
                "potential_energy_eV": potential,
                "kinetic_energy_eV": kinetic,
                "total_energy_eV": potential + kinetic,
                "temperature_K": float(atoms.get_temperature()),
                "max_abs_force_eV_per_A": float(np.max(np.abs(forces))),
                "max_displacement_A": float(
                    np.max(np.linalg.norm(atoms.positions - initial_positions, axis=1))
                ),
                "candidate_directed_edges": runtime["candidate_directed_edges"],
                "active_directed_edges": runtime["active_directed_edges"],
            }
        )

    runtime_final = _md_runtime_snapshot(calculator)
    expected = {
        "neighbor_cache_build_count": 1,
        "neighbor_cache_reuse_count": steps,
        "factorized_prepared_graph_count": 1,
        "factorized_prepared_evaluation_count": steps + 1,
        "factorized_fallback_evaluation_count": 0,
    }
    observed = {name: runtime_final[name] for name in expected}
    mismatches = {
        name: {"expected": expected_value, "observed": observed[name]}
        for name, expected_value in expected.items()
        if observed[name] != expected_value
    }
    if mismatches:
        raise RuntimeError(f"MH-1 MD reuse contract failed: {mismatches}")

    total_energy_final = trajectory[-1]["total_energy_eV"]
    return {
        "atoms": len(atoms),
        "model_cutoff_A": calculator.cutoff,
        "neighbor_skin_A": calculator.neighbor_skin,
        "effective_neighbor_cutoff_A": calculator.cutoff + calculator.neighbor_skin,
        "case": {
            "name": "wurtzite-AlN",
            "repeat": _aln_repeat_from_atom_count(len(atoms)),
            "seed": seed,
            "position_noise_std_A": position_noise_std_A,
            "requested_initial_temperature_K": requested_temperature_K,
            "actual_initial_temperature_K": actual_initial_temperature_K,
        },
        "md": {
            "ensemble": "NVE",
            "integrator": "ASE VelocityVerlet",
            "steps": steps,
            "timestep_fs": timestep_fs,
            "velocity_distribution": "MaxwellBoltzmannDistribution",
            "center_of_mass_removed": True,
        },
        "timing": {
            **_md_timing_summary(samples_ms, len(atoms)),
            "initial_evaluation_ms": initial_evaluation_ms,
            "initial_evaluation_included": False,
        },
        "runtime_after_initial_evaluation": runtime_after_initial,
        "runtime_final": runtime_final,
        "reuse_contract": {
            "expected": expected,
            "observed": observed,
            "passed": True,
        },
        "physics": {
            "initial_energy_eV": initial_energy,
            "initial_forces_sha256": _array_sha256(initial_forces),
            "total_energy_drift_eV_per_atom": (
                total_energy_final - initial_total_energy
            )
            / len(atoms),
            "temperature_min_K": min(x["temperature_K"] for x in trajectory),
            "temperature_max_K": max(x["temperature_K"] for x in trajectory),
            "final_max_displacement_A": trajectory[-1]["max_displacement_A"],
        },
        "trajectory": trajectory,
        "rss_after_mib": _current_rss_bytes() / 2**20,
        "peak_rss_mib": (
            _peak_rss_bytes() / 2**20 if _peak_rss_bytes() is not None else None
        ),
        "gpu_process_memory_after_mib": _gpu_process_memory_mib(),
        **_jit_metadata(calculator),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=pathlib.Path, help="Extracted MACE-MH-1 JSON model"
    )
    parser.add_argument(
        "--backend",
        choices=("serial", "kokkos"),
        default="serial",
        help=(
            "legacy stack selector; the retired 'serial' spelling measures the "
            "artifact-free reference routing"
        ),
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--reference-model", type=pathlib.Path, help="Optional native MH-0 JSON"
    )
    parser.add_argument("--max-reference-ratio", type=float)
    parser.add_argument(
        "--reference-streamed-edges",
        choices=(
            "auto",
            "materialized",
            "generic",
            "direct",
            "all_interactions",
            "factorized",
            "direct_streamed",
        ),
        default="auto",
    )
    parser.add_argument(
        "--reference-jit",
        dest="reference_jit",
        choices=("auto", "off", "required"),
        default="auto",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="Single CPU to pin (equivalent to a one-entry --cpus list)",
    )
    parser.add_argument(
        "--cpus",
        help="Comma-separated CPU affinity for Kokkos threads",
    )
    parser.add_argument("--sizes", default="1,2,3")
    parser.add_argument(
        "--atom-counts",
        help="Comma-separated exact AlN atom counts from 256,864,4000",
    )
    parser.add_argument(
        "--streamed-edges",
        choices=(
            "materialized",
            "generic",
            "direct",
            "all_interactions",
            "factorized",
            "direct_streamed",
        ),
        default="generic",
    )
    parser.add_argument(
        "--mh1-edge-executor",
        choices=("mlp_reference", "pair_spline_v1"),
        default="pair_spline_v1",
        help="MH-1 edge executor; pair_spline_v1 is the production default",
    )
    parser.add_argument(
        "--neighbor-skin",
        type=float,
        default=0.5,
        help="Neighbor-list skin in Angstrom; use zero for exact-cutoff comparisons",
    )
    parser.add_argument(
        "--factorized-jit",
        dest="jit",
        choices=("auto", "off", "required"),
        help=(
            "Direct JIT specialization policy; defaults to required with "
            "--streamed-edges direct and auto otherwise"
        ),
    )
    parser.add_argument(
        "--factorized-mh1-scratch-budget-bytes",
        dest="execution_mh1_scratch_budget_bytes",
        type=int,
        help=(
            "Maximum generated MH-1 compact-phi workspace; below the reported "
            "minimum, mandatory phase scratch exceeds the requested budget"
        ),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--properties",
        default="energy,forces",
        help="Comma-separated timed properties from energy,forces,stress",
    )
    parser.add_argument(
        "--mh1-product-route",
        choices=MH1_PRODUCT_ROUTES,
        default="standard",
        help=(
            "Use normal routing, explicit common shared products, or the retained "
            "monolithic oracle from the identical generated module"
        ),
    )
    parser.add_argument(
        "--product-timings-json",
        type=pathlib.Path,
        help=(
            "External profiler summary with source, collection_scope, and per-layer "
            "forward_us/reverse_us; does not instrument the timed evaluator"
        ),
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=0,
        help="Run the qualified 20-step ASE NVE protocol instead of static calls",
    )
    parser.add_argument("--md-timestep-fs", type=float, default=1.0)
    parser.add_argument("--md-temperature-K", type=float, default=300.0)
    parser.add_argument("--md-seed", type=int, default=20260808)
    parser.add_argument("--md-position-noise-std-A", type=float, default=0.01)
    parser.add_argument(
        "--e3-linear-backend",
        choices=("auto", "scalar", "packed_gemm"),
        default="auto",
        help="Benchmark-only Kokkos E3-linear execution policy",
    )
    parser.add_argument(
        "--factorized-host-node-backend",
        choices=("auto", "generated", "kokkos"),
        default="auto",
        help=(
            "Benchmark-only MH-1 host node policy; kokkos retains generated "
            "factorized edge execution while using packed Kokkos node operators"
        ),
    )
    parser.add_argument(
        "--execution-profile",
        choices=("speed", "capacity"),
        default="speed",
        help=(
            "Calculator execution profile; capacity selects the public "
            "low-memory policy, while speed honors --mh1-node-state-policy"
        ),
    )
    parser.add_argument(
        "--mh1-node-arena-tile-rows",
        type=int,
        help="Benchmark-only generated MH-1 host node tile row override",
    )
    parser.add_argument(
        "--mh1-node-state-policy",
        choices=(
            "full-retention-v1",
            "recompute-v1",
            "reuse-adjoints-v1",
            "retain-interaction-v1",
        ),
        default="full-retention-v1",
    )
    parser.add_argument(
        "--fused-gate-normalization-reverse",
        choices=("auto", "on", "off"),
        default="auto",
        help="Benchmark-only MH-1 fused node-reverse policy",
    )
    parser.add_argument(
        "--direct-node-tensor-reverse",
        choices=("auto", "on", "off"),
        default="auto",
        help="Benchmark-only MH-1 direct-node tensor-reverse policy",
    )
    parser.add_argument("--evaluator-only", action="store_true")
    parser.add_argument(
        "--allow-generic",
        action="store_true",
        help="Allow the generic nonlinear evaluator for a control measurement",
    )
    parser.add_argument(
        "--lifecycle-sizes",
        help="Comma-separated supercell sizes to alternate for RSS measurement",
    )
    parser.add_argument("--lifecycle-cycles", type=int, default=20)
    parser.add_argument("--max-lifecycle-growth-mib", type=float)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        properties = _parse_properties(args.properties)
    except ValueError as error:
        parser.error(str(error))
    args.streamed_edges = STREAMED_EDGE_ALIASES.get(
        args.streamed_edges, args.streamed_edges
    )
    args.reference_streamed_edges = STREAMED_EDGE_ALIASES.get(
        args.reference_streamed_edges, args.reference_streamed_edges
    )
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if args.md_steps not in (0, 20):
        parser.error("--md-steps must be 20 when the MD protocol is enabled")
    if args.md_steps:
        if args.neighbor_skin <= 0.0:
            parser.error("--md-steps requires a positive --neighbor-skin")
        if args.md_timestep_fs <= 0.0:
            parser.error("--md-timestep-fs must be positive")
        if args.md_temperature_K <= 0.0:
            parser.error("--md-temperature-K must be positive")
        if args.md_position_noise_std_A < 0.0:
            parser.error("--md-position-noise-std-A must be nonnegative")
        if args.atom_counts is None or len(args.atom_counts.split(",")) != 1:
            parser.error("--md-steps requires exactly one --atom-counts value")
        if args.evaluator_only:
            parser.error("--md-steps is incompatible with --evaluator-only")
        if args.reference_model is not None:
            parser.error("--md-steps is incompatible with --reference-model")
        if args.lifecycle_sizes is not None:
            parser.error("--md-steps is incompatible with --lifecycle-sizes")
    if args.mh1_product_route != "standard" and (
        args.backend != "kokkos"
        or args.streamed_edges not in ("direct", "factorized", "direct_streamed")
        or args.md_steps
        or args.reference_model is not None
    ):
        parser.error(
            "explicit MH-1 product routes require --backend kokkos, "
            "--streamed-edges direct, static evaluation, and no reference model"
        )
    if args.mh1_node_arena_tile_rows is not None and args.mh1_node_arena_tile_rows < 1:
        parser.error("--mh1-node-arena-tile-rows must be positive")
    if not np.isfinite(args.neighbor_skin) or args.neighbor_skin < 0.0:
        parser.error("--neighbor-skin must be a finite non-negative number")
    if args.atom_counts is not None and args.sizes != "1,2,3":
        parser.error("--atom-counts and an explicit --sizes are mutually exclusive")
    if args.lifecycle_cycles < 1:
        parser.error("--lifecycle-cycles must be positive")
    if args.max_lifecycle_growth_mib is not None and args.lifecycle_sizes is None:
        parser.error("--max-lifecycle-growth-mib requires --lifecycle-sizes")
    if args.max_reference_ratio is not None and args.reference_model is None:
        parser.error("--max-reference-ratio requires --reference-model")
    if args.reference_model is None and (
        args.reference_streamed_edges != "auto" or args.reference_jit != "auto"
    ):
        parser.error("reference execution controls require --reference-model")
    if (
        args.execution_mh1_scratch_budget_bytes is not None
        and args.execution_mh1_scratch_budget_bytes < 0
    ):
        parser.error("--factorized-mh1-scratch-budget-bytes must be nonnegative")
    if args.execution_mh1_scratch_budget_bytes is not None and (
        args.backend != "kokkos" or args.streamed_edges != "direct"
    ):
        parser.error(
            "--factorized-mh1-scratch-budget-bytes requires --backend kokkos "
            "and --streamed-edges direct"
        )
    if args.cpu is not None and args.cpus is not None:
        parser.error("--cpu and --cpus are mutually exclusive")
    if args.backend != "kokkos" and (
        args.fused_gate_normalization_reverse != "auto"
        or args.direct_node_tensor_reverse != "auto"
    ):
        parser.error("fused reverse controls require --backend kokkos")
    if args.direct_node_tensor_reverse == "on" and args.streamed_edges != "generic":
        parser.error(
            "--direct-node-tensor-reverse on requires --streamed-edges generic"
        )
    if hasattr(os, "sched_getaffinity"):
        available_cpus = os.sched_getaffinity(0)
        requested_cpus = None
        if args.cpu is not None:
            requested_cpus = {args.cpu}
        elif args.cpus is not None:
            try:
                requested_cpus = {int(value) for value in args.cpus.split(",")}
            except ValueError:
                parser.error("--cpus must be a comma-separated integer list")
            if not requested_cpus:
                parser.error("--cpus cannot be empty")
        if requested_cpus is not None:
            try:
                os.sched_setaffinity(0, requested_cpus)
            except OSError as error:
                parser.error(f"could not apply requested CPU affinity: {error}")
            applied_cpus = os.sched_getaffinity(0)
            if applied_cpus != requested_cpus:
                parser.error(
                    f"requested CPUs {sorted(requested_cpus)} produced affinity "
                    f"{sorted(applied_cpus)}"
                )
        elif len(available_cpus) != int(THREAD_COUNT):
            parser.error(
                "use --cpu/--cpus or launch under taskset with exactly "
                f"{THREAD_COUNT} available CPUs"
            )

    kokkos = args.backend == "kokkos"
    # The retired serial evaluator is the family's artifact-free ``reference``
    # routing, which is what the legacy non-Kokkos backend spelling measures.
    streamed_edges = args.streamed_edges if kokkos else "reference"
    reference_streamed_edges = args.reference_streamed_edges if kokkos else "reference"
    jit = args.jit
    if jit is None:
        jit = "required" if streamed_edges == "direct" else "auto"
    calculator = Symmetrix(
        args.model,
        dtype=args.dtype,
        streamed_edges=streamed_edges,
        execution_profile=args.execution_profile,
        jit=jit,
        execution_mh1_scratch_budget_bytes=args.execution_mh1_scratch_budget_bytes,
        execution_mh1_node_state_policy=args.mh1_node_state_policy,
        neighbor_skin=args.neighbor_skin,
    )
    if kokkos:
        calculator.evaluator.set_mh1_edge_executor(args.mh1_edge_executor)
        calculator.evaluator.set_e3_linear_backend(args.e3_linear_backend)
        calculator.evaluator.set_execution_mh1_host_node_backend(
            args.factorized_host_node_backend
        )
        if args.mh1_node_arena_tile_rows is not None:
            calculator.evaluator._set_execution_mh1_node_arena_tile_rows_for_testing(
                args.mh1_node_arena_tile_rows
            )
        controls = (
            (
                args.fused_gate_normalization_reverse,
                "set_fused_gate_normalization_reverse",
            ),
            (
                args.direct_node_tensor_reverse,
                "set_direct_node_tensor_reverse",
            ),
        )
        for requested, setter_name in controls:
            if requested == "auto":
                continue
            setter = getattr(calculator.evaluator, setter_name, None)
            if setter is None:
                raise RuntimeError(f"Native evaluator does not expose {setter_name}")
            setter(requested == "on")
    if not args.allow_generic and not getattr(
        calculator.evaluator, "uses_mh1_fast_path", False
    ):
        raise RuntimeError(
            f"Model does not match the specialized MACE-MH-1 "
            f"{calculator.streamed_edges} routing"
        )
    product_route = None
    schedule_overrides = None
    product_timings = None
    raw_product_timings = None
    if args.product_timings_json is not None:
        try:
            raw_product_timings = json.loads(args.product_timings_json.read_text())
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid --product-timings-json: {error}")
        if args.mh1_product_route == "standard":
            parser.error("--product-timings-json requires an explicit product route")
    if args.mh1_product_route != "standard":
        schedule_overrides = _schedule_overrides()
        product_route = _configure_mh1_product_route(calculator, args.mh1_product_route)
    reference = (
        Symmetrix(
            args.reference_model,
            dtype=args.dtype,
            streamed_edges=reference_streamed_edges,
            jit=args.reference_jit,
        )
        if args.reference_model
        else None
    )

    extension_path = pathlib.Path(native_symmetrix.__file__).resolve()
    if raw_product_timings is not None:
        try:
            expected_binding = _product_timing_binding(
                calculator,
                len(product_route["shared_product_modules"]),
                extension_path,
            )
            product_timings = _normalize_product_timings(
                raw_product_timings, expected_binding
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(f"invalid --product-timings-json: {error}")
    device_environment = getattr(
        calculator.evaluator, "execution_device_execution_environment", None
    )
    if device_environment is None:
        device_environment = getattr(
            native_symmetrix, "_execution_device_execution_environment", lambda: None
        )
    device_environment = (
        device_environment() if callable(device_environment) else device_environment
    )
    metadata = {
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "git_tracked_diff_sha256": _git_tracked_diff_sha256(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_affinities": _thread_affinities(),
        "backend": args.backend,
        "dtype": args.dtype,
        "scalar_size_bytes": getattr(calculator.evaluator, "scalar_size_bytes", 8),
        "is_mh1_family": bool(getattr(calculator.evaluator, "is_mh1_family", False)),
        "mh1_node_channels": getattr(calculator.evaluator, "mh1_node_channels", None),
        "mh1_first_layer_edge_channels": getattr(
            calculator.evaluator, "mh1_first_layer_edge_channels", None
        ),
        "mh1_radial_size": getattr(calculator.evaluator, "mh1_radial_size", None),
        "mh1_l_max": getattr(calculator.evaluator, "mh1_l_max", None),
        "mh1_family_rejection_reason": getattr(
            calculator.evaluator, "mh1_family_rejection_reason", None
        ),
        "mh1_uses_compiled_products": getattr(
            calculator.evaluator, "mh1_uses_compiled_products", None
        ),
        "mh1_uses_pair_conditioning": getattr(
            calculator.evaluator, "mh1_uses_pair_conditioning", None
        ),
        "mh1_uses_external_uvu_tensors": getattr(
            calculator.evaluator, "mh1_uses_external_uvu_tensors", None
        ),
        "mh1_fast_path_rejection_reason": getattr(
            calculator.evaluator, "mh1_fast_path_rejection_reason", None
        ),
        "uses_mh1_fast_path": bool(calculator.evaluator.uses_mh1_fast_path),
        "streamed_edges": calculator.streamed_edges,
        "mh1_edge_executor": calculator.evaluator.mh1_edge_executor,
        "execution_mh1_execution_backend": getattr(
            calculator.evaluator, "execution_mh1_execution_backend", None
        ),
        "neighbor_skin_A": args.neighbor_skin,
        "execution_mh1_scratch_budget_requested_bytes": (
            args.execution_mh1_scratch_budget_bytes
        ),
        **_jit_metadata(calculator),
        "e3_linear_backend": getattr(calculator.evaluator, "e3_linear_backend", None),
        "factorized_host_node_backend_requested": args.factorized_host_node_backend,
        "factorized_host_node_backend_selected": getattr(
            calculator.evaluator,
            "selected_execution_mh1_host_node_backend",
            None,
        ),
        "mh1_node_arena_tile_rows_requested": args.mh1_node_arena_tile_rows,
        "mh1_node_arena_tile_rows_selected": getattr(
            calculator.evaluator, "execution_mh1_node_arena_tile_rows", None
        ),
        "execution_profile": calculator.execution_profile,
        "mh1_node_state_policy_requested": args.mh1_node_state_policy,
        "mh1_node_state_policy": calculator.execution_mh1_node_state_policy,
        "tensor_product_execution_backend": getattr(
            calculator.evaluator, "tensor_product_execution_backend", None
        ),
        "tensor_product_channel_team_size": getattr(
            calculator.evaluator, "tensor_product_channel_team_size", None
        ),
        "tensor_product_harmonic_team_size": getattr(
            calculator.evaluator, "tensor_product_harmonic_team_size", None
        ),
        "fused_gate_normalization_reverse_requested": (
            args.fused_gate_normalization_reverse
        ),
        "fused_gate_normalization_reverse_available": getattr(
            calculator.evaluator,
            "fused_gate_normalization_reverse_available",
            None,
        ),
        "uses_fused_gate_normalization_reverse": getattr(
            calculator.evaluator, "uses_fused_gate_normalization_reverse", None
        ),
        "direct_node_tensor_reverse_requested": args.direct_node_tensor_reverse,
        "direct_node_tensor_reverse_available": getattr(
            calculator.evaluator, "direct_node_tensor_reverse_available", None
        ),
        "uses_direct_node_tensor_reverse": getattr(
            calculator.evaluator, "uses_direct_node_tensor_reverse", None
        ),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timed_properties": list(properties),
        "exact_command_argv": list(sys.argv),
        "exact_command": shlex.join(sys.argv),
        "replay": _replay_provenance(args),
        "jit_cache_root": os.environ.get("SYMMETRIX_JIT_CACHE"),
        "jit_policy_environment": os.environ.get("SYMMETRIX_JIT_POLICY"),
        "mh1_product_route": product_route,
        "mh1_product_schedule_overrides": schedule_overrides,
        "md_steps": args.md_steps,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "native_extension": str(extension_path),
        "native_extension_sha256": _sha256(extension_path),
        "native_build_info": getattr(
            native_symmetrix, "_backend_build_info", lambda: None
        )(),
        "native_build": _native_build_metadata(extension_path),
        "device_environment": device_environment,
        "model": _model_metadata(args.model),
        "reference_model": _model_metadata(args.reference_model)
        if args.reference_model
        else None,
        "timing_scope": {
            "evaluator": (
                "prebuilt graph; synchronized native evaluation and collection of "
                + ",".join(properties)
            ),
            "ase": "Symmetrix.calculate including graph construction and result collection",
            "md": "one complete ASE VelocityVerlet NVE step",
        },
    }

    if args.atom_counts is None:
        systems = [
            bulk("Si", "diamond", a=5.43).repeat((size,) * 3)
            for size in _parse_sizes(args.sizes, parser, "--sizes")
        ]
    else:
        atom_counts = _parse_sizes(args.atom_counts, parser, "--atom-counts")
        try:
            repeats = [_aln_repeat_from_atom_count(count) for count in atom_counts]
        except ValueError as error:
            parser.error(str(error))
        systems = [
            (
                _build_md_atoms(
                    repeat,
                    args.md_seed,
                    args.md_temperature_K,
                    args.md_position_noise_std_A,
                )
                if args.md_steps
                else bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
            )
            for repeat in repeats
        ]

    results = []
    for atoms in systems:
        if args.md_steps:
            record = _benchmark_md(
                calculator,
                atoms,
                steps=args.md_steps,
                timestep_fs=args.md_timestep_fs,
                seed=args.md_seed,
                requested_temperature_K=args.md_temperature_K,
                position_noise_std_A=args.md_position_noise_std_A,
            )
        else:
            record = _benchmark(
                calculator,
                atoms,
                args.warmups,
                args.repeats,
                not args.evaluator_only,
                properties=properties,
            )
            product_module_keys = sorted(
                key
                for key in getattr(calculator, "jit_operator_modules", {})
                if key.startswith("product_")
            )
            if product_route is not None or product_module_keys:
                _require_product_timing_system_binding(
                    product_timings, metadata, record
                )
                launch_counter_window = record.pop(
                    "_product_launch_counter_window", None
                )
                record["mh1_product_runtime"] = _mh1_product_runtime(
                    calculator,
                    (
                        len(product_route["shared_product_modules"])
                        if product_route is not None
                        else len(product_module_keys)
                    ),
                    product_timings,
                    launch_counter_window,
                )
        if reference is not None:
            reference_record = _benchmark(
                reference,
                atoms,
                args.warmups,
                args.repeats,
                not args.evaluator_only,
                properties=properties,
            )
            record["reference"] = reference_record
            record["evaluator_reference_ratio"] = (
                record["evaluator"]["median_ms"]
                / reference_record["evaluator"]["median_ms"]
            )
            record["ase_reference_ratio"] = (
                record["ase"]["median_ms"] / reference_record["ase"]["median_ms"]
                if record["ase"] is not None
                else None
            )
            if (
                args.max_reference_ratio is not None
                and record["evaluator_reference_ratio"] > args.max_reference_ratio
            ):
                raise RuntimeError(
                    f"MH-1/MH-0 evaluator ratio {record['evaluator_reference_ratio']:.3f} "
                    f"exceeds {args.max_reference_ratio:.3f} for {len(atoms)} atoms"
                )
        results.append(record)
        print(json.dumps(record), flush=True)

    lifecycle = None
    if args.lifecycle_sizes is not None:
        lifecycle = _lifecycle(
            calculator,
            _parse_sizes(args.lifecycle_sizes, parser, "--lifecycle-sizes"),
            args.lifecycle_cycles,
        )
        print(json.dumps({"lifecycle": lifecycle}), flush=True)
        if (
            args.max_lifecycle_growth_mib is not None
            and lifecycle["growth_mib"] > args.max_lifecycle_growth_mib
        ):
            raise RuntimeError(
                f"lifecycle RSS growth {lifecycle['growth_mib']:.3f} MiB exceeds "
                f"{args.max_lifecycle_growth_mib:.3f} MiB"
            )

    output = {"metadata": metadata, "systems": results, "lifecycle": lifecycle}
    calculator = None
    reference = None
    _finalize_benchmark_runtime(native_symmetrix, systems)
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
