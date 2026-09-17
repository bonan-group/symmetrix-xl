"""Benchmark standard compact MACE streamed-edge modes."""

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_THREADS", "1")
BLAS_THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_BLAS_THREADS", "1")
MODE_ALIASES = {
    "all_interactions": "generic",
    "factorized": "direct",
    "direct_streamed": "direct",
}
CANONICAL_MODES = frozenset(("materialized", "generic", "direct"))
PREPARED_MODES = frozenset(("direct",))
for variable in ("KOKKOS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = THREAD_COUNT
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = BLAS_THREAD_COUNT
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")


def _bootstrap_explicit_symmetrix():
    """Load the requested source package and extension in fresh processes."""

    if "symmetrix" in sys.modules:
        return
    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension = os.environ.get("SYMMETRIX_EXTENSION")
    if source_root is None and extension is None:
        return
    if source_root is None or extension is None:
        raise RuntimeError(
            "SYMMETRIX_SOURCE_ROOT and SYMMETRIX_EXTENSION must be set together"
        )
    package_dir = pathlib.Path(source_root).resolve() / "symmetrix/source/symmetrix"
    extension_path = pathlib.Path(extension).resolve()
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(f"invalid SYMMETRIX_SOURCE_ROOT: {source_root}")
    if not extension_path.is_file():
        raise RuntimeError(f"invalid SYMMETRIX_EXTENSION: {extension}")

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
    extension_module = extension_path.name.split(".", 1)[0]
    native_module_name = f"symmetrix.{extension_module}"
    native_spec = importlib.util.spec_from_file_location(
        native_module_name, extension_path
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("could not create the explicit Symmetrix package spec")
    if native_spec is None or native_spec.loader is None:
        raise RuntimeError("could not create the explicit Symmetrix extension spec")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["symmetrix"] = package
    try:
        package_spec.loader.exec_module(package)
        native = importlib.util.module_from_spec(native_spec)
        sys.modules[native_module_name] = native
        if extension_module == "symmetrix":
            sys.modules["symmetrix.symmetrix"] = native
        native_spec.loader.exec_module(native)
        from symmetrix import backend_loader

        backend_loader._native_module = native
        backend_loader._selected = backend_loader._cpu_descriptor(package.__version__)
        sys.modules["symmetrix.symmetrix"] = native
    except BaseException:
        sys.modules.pop(native_module_name, None)
        if extension_module == "symmetrix":
            sys.modules.pop("symmetrix.symmetrix", None)
        else:
            sys.modules.pop("symmetrix.symmetrix", None)
        sys.modules.pop("symmetrix", None)
        raise


_bootstrap_explicit_symmetrix()

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from symmetrix import Symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402


def _load_nvtx():
    candidates = (
        os.environ.get("SYMMETRIX_NVTX_LIBRARY"),
        ctypes.util.find_library("nvtx3interop"),
        "libnvtx3interop.so.1",
        ctypes.util.find_library("nvToolsExt"),
        "libnvToolsExt.so.1",
    )
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        library.nvtxRangePushA.argtypes = (ctypes.c_char_p,)
        library.nvtxRangePushA.restype = ctypes.c_int
        library.nvtxRangePop.argtypes = ()
        library.nvtxRangePop.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("could not load NVTX: " + "; ".join(errors))


def _load_cuda_profiler():
    candidates = (
        os.environ.get("SYMMETRIX_CUDART_LIBRARY"),
        ctypes.util.find_library("cudart"),
        "libcudart.so",
    )
    errors = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f"{candidate}: {error}")
            continue
        library.cudaProfilerStart.argtypes = ()
        library.cudaProfilerStart.restype = ctypes.c_int
        library.cudaProfilerStop.argtypes = ()
        library.cudaProfilerStop.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("could not load CUDA profiler API: " + "; ".join(errors))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _parse_csv(value, cast, option, parser):
    try:
        parsed = [cast(item) for item in value.split(",")]
    except ValueError:
        parser.error(f"{option} contains an invalid value")
    if not parsed:
        parser.error(f"{option} cannot be empty")
    return parsed


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _per_atom_summary(samples, atom_count):
    per_atom = [sample / atom_count for sample in samples]
    return {
        "median_ms_per_atom": statistics.median(per_atom),
        "median_us_per_atom": 1000.0 * statistics.median(per_atom),
        "min_ms_per_atom": min(per_atom),
        "max_ms_per_atom": max(per_atom),
        "samples_ms_per_atom": per_atom,
    }


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


def _process_memory_mib():
    values = {"current": None, "peak": None}
    try:
        with pathlib.Path("/proc/self/status").open() as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key == "VmRSS":
                    values["current"] = int(value.split()[0]) / 1024
                elif key == "VmHWM":
                    values["peak"] = int(value.split()[0]) / 1024
    except (OSError, ValueError):
        pass
    return values


def _standard_m0_report(
    evaluator,
    requested_executor,
    runtime_counters_before,
    runtime_counters_after,
):
    forward_counter = "standard_m0_module_forward_launch_count"
    reverse_counter = "standard_m0_module_reverse_launch_count"
    forward_launches = int(runtime_counters_after.get(forward_counter, 0))
    reverse_launches = int(runtime_counters_after.get(reverse_counter, 0))
    return {
        "ready": bool(getattr(evaluator, "standard_m0_module_ready", False)),
        "fallback_reason": getattr(
            evaluator, "standard_m0_module_fallback_reason", None
        ),
        "module_id": getattr(evaluator, "standard_m0_module_id", None),
        "module_revision": getattr(evaluator, "standard_m0_module_revision", None),
        "model_structure_fingerprint": getattr(
            evaluator, "standard_m0_model_structure_fingerprint", None
        ),
        "requested_executor": requested_executor or "automatic",
        "selected_executor": getattr(evaluator, "standard_m0_selected_executor", None),
        "module_forward_launches": forward_launches,
        "module_reverse_launches": reverse_launches,
        "measured_module_forward_launches": (
            forward_launches - int(runtime_counters_before.get(forward_counter, 0))
        ),
        "measured_module_reverse_launches": (
            reverse_launches - int(runtime_counters_before.get(reverse_counter, 0))
        ),
        "poly_values": {
            "active_bytes": int(
                getattr(evaluator, "standard_m0_poly_values_active_bytes", 0)
            ),
            "capacity_bytes": int(
                getattr(evaluator, "standard_m0_poly_values_capacity_bytes", 0)
            ),
        },
        "poly_adjoints": {
            "active_bytes": int(
                getattr(evaluator, "standard_m0_poly_adjoints_active_bytes", 0)
            ),
            "capacity_bytes": int(
                getattr(evaluator, "standard_m0_poly_adjoints_capacity_bytes", 0)
            ),
        },
    }


def _m1_polynomial_report(evaluator, counters_before, counters_after):
    forward_counter = "m1_recompute_forward_launch_count"
    reverse_counter = "m1_recompute_reverse_launch_count"
    standard_forward_counter = "standard_m1_module_forward_launch_count"
    standard_reverse_counter = "standard_m1_module_reverse_launch_count"
    forward_launches = int(counters_after.get(forward_counter, 0))
    reverse_launches = int(counters_after.get(reverse_counter, 0))
    return {
        "policy": getattr(evaluator, "m1_polynomial_policy", "retained"),
        "tile_channels": int(getattr(evaluator, "m1_recompute_tile_channels", 0)),
        "scratch_bytes": int(getattr(evaluator, "m1_recompute_scratch_bytes", 0)),
        "standard_module_ready": bool(
            getattr(evaluator, "standard_m1_module_ready", False)
        ),
        "standard_module_forward_launches": int(
            counters_after.get(standard_forward_counter, 0)
        ),
        "standard_module_reverse_launches": int(
            counters_after.get(standard_reverse_counter, 0)
        ),
        "poly_values": {
            "active_bytes": int(getattr(evaluator, "m1_poly_values_active_bytes", 0)),
            "capacity_bytes": int(
                getattr(evaluator, "m1_poly_values_capacity_bytes", 0)
            ),
        },
        "poly_adjoints": {
            "active_bytes": int(getattr(evaluator, "m1_poly_adjoints_active_bytes", 0)),
            "capacity_bytes": int(
                getattr(evaluator, "m1_poly_adjoints_capacity_bytes", 0)
            ),
        },
        "forward_launches": forward_launches,
        "reverse_launches": reverse_launches,
        "measured_forward_launches": (
            forward_launches - int(counters_before.get(forward_counter, 0))
        ),
        "measured_reverse_launches": (
            reverse_launches - int(counters_before.get(reverse_counter, 0))
        ),
    }


def _mh0_state_report(evaluator):
    return {
        "requested_policy": getattr(
            evaluator, "mh0_state_policy_request", "full-retention-v1"
        ),
        "selected_policy": getattr(evaluator, "mh0_state_policy", "full-retention-v1"),
        "fallback_reason": getattr(evaluator, "mh0_state_policy_fallback_reason", ""),
        "reused_state_bytes": int(getattr(evaluator, "mh0_reused_state_bytes", 0)),
        "auxiliary_state_bytes": int(
            getattr(evaluator, "mh0_auxiliary_state_bytes", 0)
        ),
        "readout_workspace_bytes": int(
            getattr(evaluator, "readout_workspace_bytes", 0)
        ),
        "readout_policy": getattr(evaluator, "readout_policy", "retained"),
    }


def _mh1_node_state_policy(calculator):
    return getattr(calculator, "jit_node_state_policy", None) or getattr(
        calculator.evaluator, "execution_mh1_node_state_policy", None
    )


def _edge_geometry_report(evaluator):
    return {
        "selected_policy": getattr(
            evaluator, "edge_geometry_policy", "cartesian-f64-v1"
        ),
        "compact_geometry_bytes": int(
            getattr(evaluator, "compact_edge_geometry_bytes", 0)
        ),
    }


def _phi1_report(evaluator):
    return {
        "selected_policy": getattr(evaluator, "phi1_policy", "retained"),
        "workspace_bytes": int(getattr(evaluator, "phi1_workspace_bytes", 0)),
    }


def _harmonic_storage_report(evaluator):
    return {
        "requested_policy": getattr(
            evaluator, "harmonic_storage_policy_request", "retained"
        ),
        "selected_policy": getattr(evaluator, "harmonic_storage_policy", "retained"),
        "selection_reason": getattr(evaluator, "harmonic_storage_selection_reason", ""),
        "fallback_reason": getattr(evaluator, "harmonic_storage_fallback_reason", ""),
        "value_bytes": int(getattr(evaluator, "harmonic_value_bytes", 0)),
        "gradient_bytes": int(getattr(evaluator, "harmonic_gradient_bytes", 0)),
        "shuffled_coordinate_bytes": int(
            getattr(evaluator, "shuffled_coordinate_bytes", 0)
        ),
        "direct_value_launches": int(
            getattr(evaluator, "execution_direct_harmonic_launch_count", 0)
        ),
    }


def _low_memory_report(calculator_or_evaluator):
    calculator = calculator_or_evaluator
    evaluator = getattr(calculator, "evaluator", calculator)
    return {
        "requested": bool(
            getattr(
                calculator,
                "low_memory_request",
                getattr(evaluator, "low_memory_requested", False),
            )
        ),
        "active": bool(
            getattr(calculator, "low_memory", getattr(evaluator, "low_memory", False))
        ),
        "selected_policy": getattr(
            calculator,
            "low_memory_policy",
            getattr(evaluator, "low_memory_policy", "disabled"),
        ),
        "selection_reason": getattr(
            calculator,
            "low_memory_selection_reason",
            getattr(evaluator, "low_memory_selection_reason", ""),
        ),
        "geometry_growth_reason": getattr(
            evaluator, "execution_geometry_growth_reason", ""
        ),
        "device_free_bytes": int(getattr(evaluator, "low_memory_device_free_bytes", 0)),
        "device_total_bytes": int(
            getattr(evaluator, "low_memory_device_total_bytes", 0)
        ),
        "reserve_bytes": int(getattr(evaluator, "low_memory_reserve_bytes", 0)),
        "available_bytes": int(getattr(evaluator, "low_memory_available_bytes", 0)),
        "speed_estimated_bytes": int(
            getattr(evaluator, "low_memory_speed_estimated_bytes", 0)
        ),
        "capacity_y_only_estimated_bytes": int(
            getattr(evaluator, "low_memory_capacity_y_only_estimated_bytes", 0)
        ),
        "capacity_retained_estimated_bytes": int(
            getattr(evaluator, "low_memory_capacity_retained_estimated_bytes", 0)
        ),
        "capacity_estimated_bytes": int(
            getattr(evaluator, "low_memory_capacity_estimated_bytes", 0)
        ),
        "selected_estimated_bytes": int(
            getattr(evaluator, "low_memory_selected_estimated_bytes", 0)
        ),
    }


def _make_calculator(
    model,
    backend,
    dtype,
    mode,
    factorized_source_strategy=None,
    factorized_direct_forward_executor=None,
    factorized_direct_reverse_executor=None,
    factorized_reverse_cache_policy=None,
    factorized_planner_budget_bytes=None,
    standard_r0_executor=None,
    standard_m0_executor=None,
    factorized_parameter_gradients=False,
    m1_polynomial_policy="retained",
    m1_recompute_tile_channels=None,
    mh0_state_policy="full-retention-v1",
    edge_geometry_policy="cartesian-f64-v1",
    readout_policy="retained",
    phi1_policy="retained",
    harmonic_storage_policy="retained",
    neighbor_skin=0.5,
    execution_mh1_node_state_policy="full-retention-v1",
):
    calculator = Symmetrix(
        model,
        use_kokkos=backend == "kokkos",
        dtype=dtype,
        streamed_edges=mode,
        m1_polynomial_policy=m1_polynomial_policy,
        edge_geometry_policy=edge_geometry_policy,
        neighbor_skin=neighbor_skin,
        execution_mh1_node_state_policy=execution_mh1_node_state_policy,
    )
    if mode in PREPARED_MODES and factorized_source_strategy is not None:
        calculator.evaluator._set_factorized_source_strategy(factorized_source_strategy)
    if mode == "direct" and factorized_direct_forward_executor is not None:
        calculator.evaluator._set_factorized_direct_forward_executor(
            factorized_direct_forward_executor
        )
    if mode == "direct" and factorized_direct_reverse_executor is not None:
        calculator.evaluator._set_factorized_direct_reverse_executor(
            factorized_direct_reverse_executor
        )
    if mode == "receiver_factorized" and factorized_reverse_cache_policy is not None:
        calculator.evaluator._set_factorized_reverse_cache_policy(
            factorized_reverse_cache_policy
        )
    if mode == "receiver_factorized" and factorized_planner_budget_bytes is not None:
        calculator.evaluator._set_factorized_planner_budget_bytes(
            factorized_planner_budget_bytes
        )
    if mode in PREPARED_MODES and standard_r0_executor is not None:
        calculator.evaluator._set_standard_r0_executor(standard_r0_executor)
    if (
        mode in ("generic", "direct", "receiver_factorized")
        and standard_m0_executor is not None
    ):
        calculator.evaluator._set_standard_m0_executor(standard_m0_executor)
    if mode == "receiver_factorized" and factorized_parameter_gradients:
        calculator.evaluator.set_factorized_parameter_gradients(True)
    if backend == "kokkos":
        optional_policies = (
            ("_set_m1_recompute_tile_channels", m1_recompute_tile_channels),
            ("_set_m1_polynomial_policy", m1_polynomial_policy),
            ("_set_mh0_state_policy", mh0_state_policy),
            ("_set_readout_policy", readout_policy),
            ("_set_phi1_policy", phi1_policy),
            ("_set_harmonic_storage_policy", harmonic_storage_policy),
        )
        for method_name, value in optional_policies:
            if value is None:
                continue
            method = getattr(calculator.evaluator, method_name, None)
            if callable(method):
                method(value)
    if not calculator.evaluator.supports_streamed_edges:
        raise RuntimeError(
            "The model is not a format-v2 compact MACE or MACEField model"
        )
    return calculator


def _evaluate(
    model,
    atoms,
    backend,
    dtype,
    mode,
    warmups,
    repeats,
    factorized_source_strategy,
    factorized_direct_forward_executor,
    factorized_direct_reverse_executor,
    factorized_reverse_cache_policy,
    factorized_planner_budget_bytes,
    standard_r0_executor,
    standard_m0_executor,
    factorized_parameter_gradients,
    nvtx,
    m1_polynomial_policy="retained",
    m1_recompute_tile_channels=None,
    mh0_state_policy="full-retention-v1",
    edge_geometry_policy="cartesian-f64-v1",
    readout_policy="retained",
    phi1_policy="retained",
    harmonic_storage_policy="retained",
    neighbor_skin=0.5,
    execution_mh1_node_state_policy="full-retention-v1",
    calculator=None,
):
    process_memory_before_mib = _process_memory_mib()
    gpu_memory_before_mib = _gpu_process_memory_mib() if backend == "kokkos" else None
    setup_start = time.perf_counter()
    if calculator is None:
        calculator = _make_calculator(
            model,
            backend,
            dtype,
            mode,
            factorized_source_strategy,
            factorized_direct_forward_executor,
            factorized_direct_reverse_executor,
            factorized_reverse_cache_policy,
            factorized_planner_budget_bytes,
            standard_r0_executor,
            standard_m0_executor,
            factorized_parameter_gradients,
            m1_polynomial_policy,
            m1_recompute_tile_channels,
            mh0_state_policy,
            edge_geometry_policy,
            readout_policy,
            phi1_policy,
            harmonic_storage_policy,
            neighbor_skin,
            execution_mh1_node_state_policy,
        )
    model_setup_ms = 1000.0 * (time.perf_counter() - setup_start)
    scalar_bytes = 4 if dtype == "float32" else 8
    evaluator_scalar_bytes = calculator.evaluator.scalar_size_bytes
    if evaluator_scalar_bytes != scalar_bytes:
        raise RuntimeError(
            f"evaluator uses {evaluator_scalar_bytes}-byte scalars, "
            f"expected {scalar_bytes}"
        )
    graph_start = time.perf_counter()
    inputs = calculator._mace_inputs(atoms)
    native_args = (*inputs[:5], inputs[5].flatten(), inputs[6])
    factorized_graph_generation = 0
    factorized_graph_prepare_ms = 0.0
    if mode in PREPARED_MODES:
        factorized_prepare_start = time.perf_counter()
        factorized_graph_generation = calculator.evaluator._prepare_factorized_graph(
            *inputs[:5]
        )
        factorized_graph_prepare_ms = 1000.0 * (
            time.perf_counter() - factorized_prepare_start
        )
    if factorized_graph_generation:
        evaluate_native = calculator.evaluator._compute_prepared_factorized
        evaluation_args = (
            factorized_graph_generation,
            np.asarray(inputs[5]).flatten(),
            inputs[6],
        )
    else:
        evaluate_native = calculator.evaluator.compute_node_energies_forces
        evaluation_args = native_args
    graph_setup_ms = 1000.0 * (time.perf_counter() - graph_start)
    process_memory_after_setup_mib = _process_memory_mib()
    gpu_memory_after_setup_mib = (
        _gpu_process_memory_mib() if backend == "kokkos" else None
    )
    warmup_start = time.perf_counter()
    for _ in range(warmups):
        evaluate_native(*evaluation_args)
    warmup_ms = 1000.0 * (time.perf_counter() - warmup_start)
    runtime_counter_names = (
        "factorized_prepared_evaluation_count",
        "factorized_fallback_evaluation_count",
        "factorized_topology_validation_count",
        "factorized_topology_validation_skip_count",
        "factorized_preparation_fence_count",
        "factorized_evaluation_fence_count",
        "factorized_stage_fence_count",
        "execution_geometry_capacity_edges",
        "execution_geometry_workspace_bytes",
        "execution_geometry_allocation_count",
        "execution_geometry_copy_count",
        "execution_sphericart_initialization_count",
        "execution_sphericart_launch_count",
        "execution_sphericart_async_launch_count",
        "execution_direct_harmonic_launch_count",
        "factorized_jit_launch_count",
        "factorized_jit_forward_launch_count",
        "factorized_jit_reverse_launch_count",
        "receiver_factorized_forward_launch_count",
        "receiver_factorized_reverse_launch_count",
        "factorized_custom_blas_launch_count",
        "factorized_blas_stream_bind_count",
        "standard_r0_module_launch_count",
        "standard_m0_module_forward_launch_count",
        "standard_m0_module_reverse_launch_count",
        "m1_recompute_forward_launch_count",
        "m1_recompute_reverse_launch_count",
        "standard_m1_module_forward_launch_count",
        "standard_m1_module_reverse_launch_count",
        "execution_mh1_generated_forward_launch_count",
        "execution_mh1_generated_source_reverse_launch_count",
        "execution_mh1_generated_edge_reverse_launch_count",
        "execution_mh1_generated_conditioning_forward_launch_count",
        "execution_mh1_generated_conditioning_reverse_launch_count",
    )
    runtime_counters_before = {
        name: int(getattr(calculator.evaluator, name, 0))
        for name in runtime_counter_names
    }
    process_memory_after_warmup_mib = _process_memory_mib()
    gpu_memory_after_warmup_mib = (
        _gpu_process_memory_mib() if backend == "kokkos" else None
    )
    samples = []
    nvtx_record = None
    nvtx_api = None
    cuda_profiler_api = None
    if nvtx:
        if repeats != 1:
            raise ValueError("NVTX profiling requires exactly one measured iteration")
        nvtx_api, nvtx_library = _load_nvtx()
        cuda_profiler_api, cudart_library = _load_cuda_profiler()
        nvtx_range_name = f"symmetrix_full_evaluator::{mode}"
    for _ in range(repeats):
        if nvtx:
            if cuda_profiler_api.cudaProfilerStart() != 0:
                raise RuntimeError("cudaProfilerStart failed")
            nvtx_api.nvtxRangePushA(nvtx_range_name.encode("ascii"))
        start = time.perf_counter()
        try:
            evaluate_native(*evaluation_args)
        finally:
            samples.append(1000.0 * (time.perf_counter() - start))
            if nvtx:
                nvtx_api.nvtxRangePop()
                if cuda_profiler_api.cudaProfilerStop() != 0:
                    raise RuntimeError("cudaProfilerStop failed")
                nvtx_record = {
                    "range": nvtx_range_name,
                    "library": nvtx_library,
                    "capture_api": "cudaProfilerApi",
                    "cudart_library": cudart_library,
                    "iterations": 1,
                    "warmup_iterations": warmups,
                }
    runtime_counters_after = {
        name: int(getattr(calculator.evaluator, name, 0))
        for name in runtime_counter_names
    }
    runtime_counter_deltas = {
        name.removeprefix("factorized_"): (
            runtime_counters_after[name] - runtime_counters_before[name]
        )
        for name in runtime_counter_names
    }
    if factorized_graph_generation:
        prepared_delta = runtime_counter_deltas["prepared_evaluation_count"]
        if prepared_delta != repeats:
            raise RuntimeError(
                "prepared evaluation count did not match measured repeats: "
                f"{prepared_delta} != {repeats}"
            )
        if (
            runtime_counters_before["factorized_fallback_evaluation_count"] != 0
            or runtime_counter_deltas["fallback_evaluation_count"] != 0
        ):
            raise RuntimeError("prepared execution used the fallback path")
    results = calculator._collect_mace_results(
        atoms,
        inputs,
        ("energy", "forces", "stress"),
        factorized_graph_generation,
    )
    r0_elements = int(getattr(calculator.evaluator, "R0_storage_size", 0))
    r1_elements = int(getattr(calculator.evaluator, "R1_storage_size", 0))
    factorized_workspace_bytes = (
        int(calculator.evaluator.factorized_workspace_bytes)
        if hasattr(calculator.evaluator, "factorized_workspace_bytes")
        else 0
    )
    gpu_memory_after_mib = _gpu_process_memory_mib() if backend == "kokkos" else None
    process_memory_after_mib = _process_memory_mib()
    if hasattr(calculator.evaluator, "factorized_parameter_gradients"):
        parameter_gradients = calculator.evaluator.factorized_parameter_gradients()
    else:
        parameter_gradients = {
            "enabled": False,
            "ready": False,
            "parameterization": "unavailable",
            "coverage": "unavailable",
            "source_checkpoint_mapping": "unavailable",
            "result_bytes": 0,
            "workspace_bytes": 0,
            "max_bytes": 0,
            "phase_times_ms": {"r1": 0.0, "r0": 0.0, "density": 0.0},
            "worker_counts": {"r1": 0, "r0": 0},
            "groups": [],
        }
    sampled_gpu_values = [
        value
        for value in (
            gpu_memory_before_mib,
            gpu_memory_after_setup_mib,
            gpu_memory_after_warmup_mib,
            gpu_memory_after_mib,
        )
        if value is not None
    ]
    return {
        "requested_mode": getattr(calculator, "streamed_edges_requested", mode),
        "canonical_mode": getattr(calculator, "streamed_edges", mode),
        "execution_algorithm": getattr(calculator, "execution_algorithm", mode),
        "mode_alias": getattr(calculator, "streamed_edges_alias", None),
        "resolution_reason": getattr(
            calculator, "streamed_edges_resolution_reason", None
        ),
        "evaluator_scalar_size_bytes": evaluator_scalar_bytes,
        "directed_edges": len(inputs[6]),
        "graph_sha256": _array_sha256(*inputs[1:]),
        "timing": _summary(samples),
        "timing_per_atom": _per_atom_summary(samples, len(atoms)),
        "nvtx": nvtx_record,
        "energy_eV": float(results["energy"]),
        "forces_eV_per_A": np.asarray(results["forces"]).tolist(),
        "stress_eV_per_A3": np.asarray(results["stress"]).tolist(),
        "edge_radial_storage": {
            "R0_elements": r0_elements,
            "R1_elements": r1_elements,
            "bytes": scalar_bytes * (r0_elements + r1_elements),
        },
        "m1_polynomial": _m1_polynomial_report(
            calculator.evaluator,
            runtime_counters_before,
            runtime_counters_after,
        ),
        "mh0_state": _mh0_state_report(calculator.evaluator),
        "mh1_workspace": {
            "execution_backend": getattr(
                calculator.evaluator, "execution_mh1_execution_backend", None
            ),
            "edge_executor": getattr(calculator.evaluator, "mh1_edge_executor", None),
            "node_state_policy": _mh1_node_state_policy(calculator),
            "node_arena_policy": getattr(
                calculator.evaluator, "execution_mh1_node_arena_policy", None
            ),
            "node_arena_tile_rows": int(
                getattr(
                    calculator.evaluator,
                    "execution_mh1_node_arena_tile_rows",
                    0,
                )
            ),
            "edge_bytes": int(getattr(calculator.evaluator, "edge_workspace_bytes", 0)),
            "node_bytes": int(getattr(calculator.evaluator, "node_workspace_bytes", 0)),
            "precision_bytes": int(
                getattr(calculator.evaluator, "precision_workspace_bytes", 0)
            ),
            "geometry_bytes": int(
                getattr(
                    calculator.evaluator,
                    "execution_geometry_workspace_bytes",
                    0,
                )
            ),
            "scratch_minimum_bytes": int(
                getattr(
                    calculator.evaluator,
                    "execution_mh1_scratch_minimum_bytes",
                    0,
                )
            ),
            "scratch_planned_bytes": int(
                getattr(
                    calculator.evaluator,
                    "execution_mh1_scratch_planned_bytes",
                    0,
                )
            ),
            "scratch_retained_bytes": int(
                getattr(
                    calculator.evaluator,
                    "execution_mh1_scratch_retained_bytes",
                    0,
                )
            ),
            "scratch_recomputed_bytes": int(
                getattr(
                    calculator.evaluator,
                    "execution_mh1_scratch_recomputed_bytes",
                    0,
                )
            ),
        },
        "edge_geometry": _edge_geometry_report(calculator.evaluator),
        "phi1": _phi1_report(calculator.evaluator),
        "low_memory": _low_memory_report(calculator),
        "harmonic_storage": _harmonic_storage_report(calculator.evaluator),
        "prepared_execution": {
            "jit": {
                "policy": getattr(calculator, "jit_policy", None),
                "status": getattr(calculator, "jit_status", None),
                "compiler_backend": getattr(calculator, "jit_compiler_backend", None),
                "artifact_id": getattr(calculator, "jit_artifact_id", None),
                "variant_id": getattr(calculator, "jit_variant_id", None),
                "edge_policy": getattr(calculator, "jit_edge_policy", None),
                "node_state_policy": getattr(calculator, "jit_node_state_policy", None),
                "hip_plugin_ready": bool(
                    getattr(
                        calculator.evaluator,
                        "jit_hip_plugin_ready",
                        False,
                    )
                ),
            },
            "ready": bool(getattr(calculator.evaluator, "factorized_ready", False)),
            "workspace_bytes": factorized_workspace_bytes,
            "workspace_capacity_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_workspace_capacity_bytes",
                    factorized_workspace_bytes,
                )
            ),
            "schedule_build_count": int(
                getattr(calculator.evaluator, "factorized_schedule_build_count", 0)
            ),
            "schedule_entries": int(
                getattr(calculator.evaluator, "factorized_schedule_entries", 0)
            ),
            "schedule_bytes": int(
                getattr(calculator.evaluator, "factorized_schedule_bytes", 0)
            ),
            "graph_generation": int(
                getattr(calculator.evaluator, "factorized_graph_generation", 0)
            ),
            "prepared_graph_count": int(
                getattr(calculator.evaluator, "factorized_prepared_graph_count", 0)
            ),
            "runtime_counters": {
                name.removeprefix("factorized_"): value
                for name, value in runtime_counters_after.items()
            },
            "measured_runtime_counter_deltas": runtime_counter_deltas,
            "source_owned_reverse": bool(
                getattr(
                    calculator.evaluator,
                    "factorized_source_owned_reverse",
                    False,
                )
            ),
            "receiver_factorized_rtc": {
                "ready": bool(
                    getattr(
                        calculator.evaluator,
                        "receiver_factorized_host_plugin_ready",
                        False,
                    )
                ),
                "artifact_id": getattr(
                    calculator.evaluator,
                    "receiver_factorized_host_plugin_artifact_id",
                    None,
                ),
                "workspace_bytes": int(
                    getattr(
                        calculator.evaluator,
                        "receiver_factorized_workspace_bytes",
                        0,
                    )
                ),
                "projection_bytes": int(
                    getattr(
                        calculator.evaluator,
                        "receiver_factorized_projection_bytes",
                        0,
                    )
                ),
                "forward_launches": int(
                    getattr(
                        calculator.evaluator,
                        "receiver_factorized_forward_launch_count",
                        0,
                    )
                ),
                "reverse_launches": int(
                    getattr(
                        calculator.evaluator,
                        "receiver_factorized_reverse_launch_count",
                        0,
                    )
                ),
            },
            "r1_jit_forward": {
                "jit_ready": bool(
                    getattr(calculator.evaluator, "factorized_jit_ready", False)
                ),
                "artifact_id": getattr(
                    calculator.evaluator,
                    "factorized_jit_artifact_id",
                    None,
                ),
                "contract_fingerprint": getattr(
                    calculator.evaluator,
                    "factorized_jit_contract_fingerprint",
                    None,
                ),
                "requested_executor": getattr(
                    calculator.evaluator,
                    "factorized_direct_forward_executor",
                    None,
                ),
                "selected_executor": getattr(
                    calculator.evaluator,
                    "factorized_selected_direct_forward_executor",
                    None,
                ),
                "jit_launches": int(
                    getattr(
                        calculator.evaluator,
                        "factorized_jit_forward_launch_count",
                        0,
                    )
                ),
            },
            "r1_jit_reverse": {
                "jit_ready": bool(
                    getattr(calculator.evaluator, "factorized_jit_ready", False)
                ),
                "artifact_id": getattr(
                    calculator.evaluator,
                    "factorized_jit_artifact_id",
                    None,
                ),
                "contract_fingerprint": getattr(
                    calculator.evaluator,
                    "factorized_jit_contract_fingerprint",
                    None,
                ),
                "requested_executor": getattr(
                    calculator.evaluator,
                    "factorized_direct_reverse_executor",
                    None,
                ),
                "selected_executor": getattr(
                    calculator.evaluator,
                    "factorized_selected_direct_reverse_executor",
                    None,
                ),
                "jit_launches": int(
                    getattr(
                        calculator.evaluator,
                        "factorized_jit_reverse_launch_count",
                        0,
                    )
                ),
            },
            "r0_module": {
                "ready": bool(
                    getattr(calculator.evaluator, "standard_r0_module_ready", False)
                ),
                "fallback_reason": getattr(
                    calculator.evaluator,
                    "standard_r0_module_fallback_reason",
                    None,
                ),
                "module_id": getattr(
                    calculator.evaluator,
                    "standard_r0_module_id",
                    None,
                ),
                "module_revision": getattr(
                    calculator.evaluator,
                    "standard_r0_module_revision",
                    None,
                ),
                "model_contract_fingerprint": getattr(
                    calculator.evaluator,
                    "standard_r0_model_contract_fingerprint",
                    None,
                ),
                "requested_executor": getattr(
                    calculator.evaluator,
                    "standard_r0_executor",
                    None,
                ),
                "selected_executor": getattr(
                    calculator.evaluator,
                    "standard_r0_selected_executor",
                    None,
                ),
            },
            "m0_module": _standard_m0_report(
                calculator.evaluator,
                standard_m0_executor,
                runtime_counters_before,
                runtime_counters_after,
            ),
            "tensor_capacity_bytes": {
                "Phi1r": int(
                    getattr(calculator.evaluator, "factorized_phi1r_capacity_bytes", 0)
                ),
                "Phi1r_active": int(
                    getattr(calculator.evaluator, "factorized_phi1r_active_bytes", 0)
                ),
                "Phi1": int(
                    getattr(calculator.evaluator, "factorized_phi1_capacity_bytes", 0)
                ),
                "dPhi1r": int(
                    getattr(calculator.evaluator, "factorized_dphi1r_capacity_bytes", 0)
                ),
                "dPhi1r_active": int(
                    getattr(calculator.evaluator, "factorized_dphi1r_active_bytes", 0)
                ),
                "dPhi1": int(
                    getattr(calculator.evaluator, "factorized_dphi1_capacity_bytes", 0)
                ),
            },
            "r0_has_model_contract": bool(
                getattr(
                    calculator.evaluator,
                    "standard_r0_has_model_contract",
                    False,
                )
            ),
            "r0_model_contract_fingerprint": getattr(
                calculator.evaluator,
                "standard_r0_model_contract_fingerprint",
                None,
            ),
            "r0_model_semantic_fingerprint": getattr(
                calculator.evaluator,
                "standard_r0_model_semantic_fingerprint",
                None,
            ),
            "r0_module_launch_count": int(
                getattr(
                    calculator.evaluator,
                    "standard_r0_module_launch_count",
                    0,
                )
            ),
            "r0_density_scale_fused": bool(
                getattr(
                    calculator.evaluator,
                    "standard_r0_density_scale_fused",
                    False,
                )
            ),
            "r0_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "standard_r0_workspace_bytes",
                    0,
                )
            ),
            "unified_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_unified_workspace_bytes",
                    factorized_workspace_bytes,
                )
            ),
            "source_strategy": getattr(
                calculator.evaluator, "factorized_source_strategy", None
            ),
            "execution_strategy": getattr(
                calculator.evaluator, "factorized_execution_strategy", None
            ),
            "execution_profile": getattr(
                calculator.evaluator, "factorized_execution_profile", None
            ),
            "derivative_signature": getattr(
                calculator.evaluator, "factorized_derivative_signature", None
            ),
            "reverse_cache_policy": getattr(
                calculator.evaluator, "factorized_reverse_cache_policy", None
            ),
            "selected_reverse_cache_policy": getattr(
                calculator.evaluator,
                "factorized_selected_reverse_cache_policy",
                None,
            ),
            "retained_reverse_groups": list(
                getattr(
                    calculator.evaluator,
                    "factorized_retained_reverse_groups",
                    (),
                )
            ),
            "recomputed_reverse_groups": list(
                getattr(
                    calculator.evaluator,
                    "factorized_recomputed_reverse_groups",
                    (),
                )
            ),
            "reverse_group_workspace_bytes": list(
                getattr(
                    calculator.evaluator,
                    "factorized_reverse_group_workspace_bytes",
                    (),
                )
            ),
            "planner_budget_bytes": int(
                getattr(calculator.evaluator, "factorized_planner_budget_bytes", 0)
            ),
            "planned_coupling_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_planned_coupling_workspace_bytes",
                    0,
                )
            ),
            "jit_ready": bool(getattr(calculator.evaluator, "jit_ready", False)),
            "jit_artifact_id": getattr(calculator.evaluator, "jit_artifact_id", None),
            "jit_contract_fingerprint": getattr(
                calculator.evaluator,
                "jit_contract_fingerprint",
                None,
            ),
            "forward_coupling_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_forward_coupling_workspace_bytes",
                    0,
                )
            ),
            "reverse_coupling_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_reverse_coupling_workspace_bytes",
                    0,
                )
            ),
            "has_model_contract": bool(
                getattr(
                    calculator.evaluator,
                    "factorized_has_model_contract",
                    False,
                )
            ),
            "model_contract_fingerprint": getattr(
                calculator.evaluator,
                "factorized_model_contract_fingerprint",
                None,
            ),
            "model_semantic_fingerprint": getattr(
                calculator.evaluator,
                "factorized_model_semantic_fingerprint",
                None,
            ),
            "radial_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_radial_workspace_bytes",
                    0,
                )
            ),
            "arena_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_arena_workspace_bytes",
                    0,
                )
            ),
            "arena_allocation_count": int(
                getattr(
                    calculator.evaluator,
                    "factorized_arena_allocation_count",
                    0,
                )
            ),
            "coupling_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_coupling_workspace_bytes",
                    0,
                )
            ),
            "coupling_capacity_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_coupling_capacity_bytes",
                    0,
                )
            ),
            "compact_workspace_bytes": int(
                getattr(
                    calculator.evaluator,
                    "factorized_compact_workspace_bytes",
                    0,
                )
            ),
            "tiled_ready": bool(
                getattr(calculator.evaluator, "factorized_tiled_ready", False)
            ),
            "chunk_size": int(
                getattr(calculator.evaluator, "factorized_chunk_size", 0)
            ),
            "default_chunk_size": int(
                getattr(
                    calculator.evaluator,
                    "factorized_default_chunk_size",
                    0,
                )
            ),
            "tile_selection_reason": getattr(
                calculator.evaluator,
                "factorized_tile_selection_reason",
                None,
            ),
        },
        "factorized_parameter_gradients": {
            "enabled": bool(parameter_gradients["enabled"]),
            "ready": bool(parameter_gradients["ready"]),
            "parameterization": parameter_gradients["parameterization"],
            "coverage": parameter_gradients["coverage"],
            "source_checkpoint_mapping": parameter_gradients[
                "source_checkpoint_mapping"
            ],
            "result_bytes": int(parameter_gradients["result_bytes"]),
            "workspace_bytes": int(parameter_gradients["workspace_bytes"]),
            "max_bytes": int(parameter_gradients["max_bytes"]),
            "phase_times_ms": {
                name: float(value)
                for name, value in parameter_gradients.get("phase_times_ms", {}).items()
            },
            "worker_counts": {
                name: int(value)
                for name, value in parameter_gradients.get("worker_counts", {}).items()
            },
            "groups": [
                {
                    "name": group["name"],
                    "shape": list(group["shape"]),
                    "l1_norm": float(np.sum(np.abs(group["values"]))),
                }
                for group in parameter_gradients["groups"]
            ],
        },
        "gpu_process_memory_mib": {
            "before": gpu_memory_before_mib,
            "after_setup": gpu_memory_after_setup_mib,
            "after_warmup": gpu_memory_after_warmup_mib,
            "peak_sampled": max(sampled_gpu_values) if sampled_gpu_values else None,
            "after": gpu_memory_after_mib,
        },
        "process_memory_mib": {
            "before": process_memory_before_mib,
            "after_setup": process_memory_after_setup_mib,
            "after_warmup": process_memory_after_warmup_mib,
            "after": process_memory_after_mib,
        },
        "allocation_high_water": {
            "bytes": None,
            "source": "not exposed by the current Kokkos allocator",
        },
        "phase_times_ms": {
            "model_setup": model_setup_ms,
            "graph_setup": graph_setup_ms,
            "factorized_graph_prepare": factorized_graph_prepare_ms,
            "warmup_total": warmup_ms,
            "measured_total": sum(samples),
            "native_evaluator_phases": {
                "last_graph_prepare": float(
                    getattr(
                        calculator.evaluator,
                        "factorized_last_graph_prepare_ms",
                        0.0,
                    )
                ),
                "last_evaluation": float(
                    getattr(
                        calculator.evaluator,
                        "factorized_last_evaluation_ms",
                        0.0,
                    )
                ),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--backend", choices=("serial", "kokkos"), default="kokkos")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--modes", default="materialized,generic")
    parser.add_argument("--sizes", default="2,3,4")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--neighbor-skin",
        type=float,
        default=0.5,
        help="Neighbor-list skin in Angstrom; use zero for exact-cutoff comparisons",
    )
    parser.add_argument("--max-force-error", type=float, default=2e-5)
    parser.add_argument("--min-speedup", type=float)
    parser.add_argument("--max-us-per-atom", type=float)
    parser.add_argument(
        "--factorized-r1-source-strategy",
        dest="factorized_source_strategy",
        choices=(
            "serial_reference",
            "team_cached",
            "tiled_coupling",
            "jit_plugin",
        ),
    )
    parser.add_argument(
        "--factorized-r1-direct-forward-executor",
        dest="factorized_direct_forward_executor",
        choices=("automatic", "runtime", "jit_all"),
    )
    parser.add_argument(
        "--factorized-r1-direct-reverse-executor",
        dest="factorized_direct_reverse_executor",
        choices=("automatic", "runtime", "jit"),
    )
    parser.add_argument(
        "--factorized-r1-reverse-cache-policy",
        dest="factorized_reverse_cache_policy",
        choices=("automatic", "recompute", "retain"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--factorized-r1-planner-budget-bytes",
        dest="factorized_planner_budget_bytes",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--standard-r0-executor",
        choices=("automatic", "v1", "v2_receiver", "v2_edge16", "v2_edge32"),
    )
    parser.add_argument(
        "--standard-m0-executor",
        choices=("automatic", "runtime", "standard"),
    )
    parser.add_argument(
        "--factorized-parameter-gradients",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--m1-polynomial-policy",
        choices=("retained", "recompute"),
        default="retained",
    )
    parser.add_argument(
        "--m1-recompute-tile-channels",
        choices=(8, 16, 32),
        type=int,
        default=None,
        help=(
            "pin an exact M1 recomputation tile for diagnostics; by default "
            "the runtime selects the widest tile supported by device scratch"
        ),
    )
    parser.add_argument(
        "--mh0-state-policy",
        choices=("full-retention-v1", "reuse-adjoints-v1"),
        default="full-retention-v1",
    )
    parser.add_argument(
        "--edge-geometry-policy",
        choices=("cartesian-f64-v1", "unit-f32-radius-f64-v1"),
        default="cartesian-f64-v1",
    )
    parser.add_argument(
        "--readout-policy",
        choices=("retained", "recompute"),
        default="retained",
    )
    parser.add_argument(
        "--phi1-policy",
        choices=("retained", "channel-tiled-64", "receiver-local"),
        default="retained",
    )
    parser.add_argument(
        "--harmonic-storage-policy",
        choices=("automatic", "retained", "y-only-direct-v1"),
        default="retained",
    )
    parser.add_argument(
        "--mh1-node-state-policy",
        dest="execution_mh1_node_state_policy",
        choices=(
            "full-retention-v1",
            "recompute-v1",
            "reuse-adjoints-v1",
            "retain-interaction-v1",
        ),
        default="full-retention-v1",
        help="generated MH-1 node-state storage policy",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="mark one measured full-evaluator iteration for Nsight Systems",
    )
    parser.add_argument("--reuse-evaluator", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if not np.isfinite(args.neighbor_skin) or args.neighbor_skin < 0.0:
        parser.error("--neighbor-skin must be a finite non-negative number")

    requested_modes = _parse_csv(args.modes, str, "--modes", parser)
    modes = [MODE_ALIASES.get(mode, mode) for mode in requested_modes]
    if any(mode not in CANONICAL_MODES for mode in modes):
        parser.error(
            "--modes must contain only materialized,generic,direct or a "
            "compatibility alias"
        )
    if len(set(modes)) != len(modes):
        parser.error("--modes contains selectors that resolve to the same algorithm")
    if args.factorized_source_strategy is not None and not PREPARED_MODES.intersection(
        modes
    ):
        parser.error("--factorized-r1-source-strategy requires a prepared mode")
    if args.factorized_direct_forward_executor is not None and "direct" not in modes:
        parser.error(
            "--factorized-r1-direct-forward-executor requires direct in --modes"
        )
    if args.factorized_direct_reverse_executor is not None and "direct" not in modes:
        parser.error(
            "--factorized-r1-direct-reverse-executor requires direct in --modes"
        )
    if (
        args.factorized_reverse_cache_policy is not None
        and "receiver_factorized" not in modes
    ):
        parser.error(
            "--factorized-r1-reverse-cache-policy requires receiver_factorized"
        )
    if args.factorized_planner_budget_bytes is not None:
        if "receiver_factorized" not in modes:
            parser.error(
                "--factorized-r1-planner-budget-bytes requires receiver_factorized"
            )
        if args.factorized_planner_budget_bytes < 1:
            parser.error("--factorized-r1-planner-budget-bytes must be positive")
    if args.standard_r0_executor is not None and "direct" not in modes:
        parser.error("--standard-r0-executor requires direct")
    if (
        args.standard_m0_executor is not None
        and "generic" not in modes
        and "direct" not in modes
    ):
        parser.error("--standard-m0-executor requires a streamed mode")
    if args.factorized_parameter_gradients and "receiver_factorized" not in modes:
        parser.error("--factorized-parameter-gradients requires receiver_factorized")
    if args.factorized_parameter_gradients and args.backend != "kokkos":
        parser.error("--factorized-parameter-gradients requires the kokkos backend")
    if args.m1_polynomial_policy == "recompute":
        if args.backend != "kokkos":
            parser.error("--m1-polynomial-policy=recompute requires kokkos")
        if args.dtype != "float32":
            parser.error("--m1-polynomial-policy=recompute requires float32")
    if args.mh0_state_policy == "reuse-adjoints-v1":
        if args.backend != "kokkos" or args.dtype != "float32":
            parser.error("--mh0-state-policy=reuse-adjoints-v1 requires Kokkos float32")
        if modes != ["direct"]:
            parser.error(
                "--mh0-state-policy=reuse-adjoints-v1 requires only direct mode"
            )
        if args.m1_polynomial_policy != "recompute":
            parser.error(
                "--mh0-state-policy=reuse-adjoints-v1 requires "
                "--m1-polynomial-policy=recompute"
            )
    if args.edge_geometry_policy == "unit-f32-radius-f64-v1":
        if args.backend != "kokkos" or args.dtype != "float32":
            parser.error(
                "--edge-geometry-policy=unit-f32-radius-f64-v1 requires Kokkos float32"
            )
        if modes != ["direct"]:
            parser.error(
                "--edge-geometry-policy=unit-f32-radius-f64-v1 requires only "
                "direct mode"
            )
        if args.m1_polynomial_policy != "recompute":
            parser.error(
                "--edge-geometry-policy=unit-f32-radius-f64-v1 requires "
                "--m1-polynomial-policy=recompute"
            )
        if args.factorized_source_strategy not in (None, "jit_plugin"):
            parser.error(
                "--edge-geometry-policy=unit-f32-radius-f64-v1 requires "
                "--factorized-r1-source-strategy=jit_plugin"
            )
    if args.readout_policy == "recompute" and args.backend != "kokkos":
        parser.error("--readout-policy=recompute requires the kokkos backend")
    if args.phi1_policy != "retained":
        if args.backend != "kokkos" or args.dtype != "float32":
            parser.error("experimental --phi1-policy requires Kokkos float32")
        if modes != ["direct"]:
            parser.error("experimental --phi1-policy requires only direct mode")
        if args.factorized_source_strategy not in (None, "jit_plugin"):
            parser.error(
                "experimental --phi1-policy requires "
                "--factorized-r1-source-strategy=jit_plugin"
            )
    if args.harmonic_storage_policy == "y-only-direct-v1":
        if args.phi1_policy not in ("retained", "channel-tiled-64"):
            parser.error(
                "--harmonic-storage-policy=y-only-direct-v1 requires "
                "--phi1-policy=retained or channel-tiled-64"
            )
        required_geometry = (
            "unit-f32-radius-f64-v1" if args.dtype == "float32" else "cartesian-f64-v1"
        )
        if args.edge_geometry_policy != required_geometry:
            parser.error(
                "--harmonic-storage-policy=y-only-direct-v1 requires "
                f"--edge-geometry-policy={required_geometry} for {args.dtype}"
            )
    if args.reuse_evaluator and len(modes) != 1:
        parser.error("--reuse-evaluator requires exactly one streamed-edge mode")
    sizes = _parse_csv(args.sizes, int, "--sizes", parser)
    if any(size < 1 for size in sizes):
        parser.error("--sizes values must be positive")
    if args.max_us_per_atom is not None:
        if args.max_us_per_atom <= 0:
            parser.error("--max-us-per-atom must be positive")
        if len(modes) != 1:
            parser.error("--max-us-per-atom requires exactly one mode")
    if args.nvtx:
        if args.backend != "kokkos":
            parser.error("--nvtx requires the kokkos backend")
        if len(modes) != 1 or len(sizes) != 1 or args.repeats != 1:
            parser.error("--nvtx requires exactly one mode, one size, and --repeats=1")

    model = args.model.resolve()
    with model.open(encoding="utf-8") as stream:
        model_cutoff = float(json.load(stream)["r_cut"])
    extension_path = pathlib.Path(native_symmetrix.__file__).resolve()
    report = {
        "model": {
            "path": str(model),
            "size_bytes": model.stat().st_size,
            "sha256": _sha256(model),
        },
        "backend": args.backend,
        "source_commit": args.source_commit,
        "reuse_evaluator_across_sizes": args.reuse_evaluator,
        "kokkos_execution_space": (
            native_symmetrix._kokkos_default_execution_space()
            if args.backend == "kokkos"
            else None
        ),
        "dtype": args.dtype,
        "runtime": {
            "native_extension": str(extension_path),
            "native_extension_sha256": _sha256(extension_path),
        },
        "benchmark_contract": {
            "requested_modes": requested_modes,
            "canonical_modes": modes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "model_cutoff_A": model_cutoff,
            "neighbor_skin_A": args.neighbor_skin,
            "effective_neighbor_cutoff_A": model_cutoff + args.neighbor_skin,
            "max_us_per_atom": args.max_us_per_atom,
            "energy_and_forces": True,
            "stress": True,
            "graph_built_once": True,
            "synchronized_evaluations": True,
            "fresh_process_required": True,
            "factorized_source_strategy": args.factorized_source_strategy,
            "factorized_direct_forward_executor": (
                args.factorized_direct_forward_executor
            ),
            "factorized_direct_reverse_executor": (
                args.factorized_direct_reverse_executor
            ),
            "factorized_reverse_cache_policy": args.factorized_reverse_cache_policy,
            "factorized_planner_budget_bytes": args.factorized_planner_budget_bytes,
            "standard_r0_executor": args.standard_r0_executor,
            "standard_m0_executor": args.standard_m0_executor,
            "factorized_parameter_gradients": args.factorized_parameter_gradients,
            "m1_polynomial_policy": args.m1_polynomial_policy,
            "m1_recompute_tile_channels": args.m1_recompute_tile_channels,
            "mh0_state_policy": args.mh0_state_policy,
            "edge_geometry_policy": args.edge_geometry_policy,
            "readout_policy": args.readout_policy,
            "phi1_policy": args.phi1_policy,
            "harmonic_storage_policy": args.harmonic_storage_policy,
            "execution_mh1_node_state_policy": (args.execution_mh1_node_state_policy),
            "nvtx": args.nvtx,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "KOKKOS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
        },
        "systems": [],
    }
    failed = False
    reused_calculator = None
    if args.reuse_evaluator:
        reused_calculator = _make_calculator(
            model,
            args.backend,
            args.dtype,
            modes[0],
            args.factorized_source_strategy,
            args.factorized_direct_forward_executor,
            args.factorized_direct_reverse_executor,
            args.factorized_reverse_cache_policy,
            args.factorized_planner_budget_bytes,
            args.standard_r0_executor,
            args.standard_m0_executor,
            args.factorized_parameter_gradients,
            args.m1_polynomial_policy,
            args.m1_recompute_tile_channels,
            args.mh0_state_policy,
            args.edge_geometry_policy,
            args.readout_policy,
            args.phi1_policy,
            args.harmonic_storage_policy,
            args.neighbor_skin,
            args.execution_mh1_node_state_policy,
        )
    for size in sizes:
        atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((size, size, size))
        records = {
            mode: _evaluate(
                model,
                atoms,
                args.backend,
                args.dtype,
                mode,
                args.warmups,
                args.repeats,
                args.factorized_source_strategy,
                args.factorized_direct_forward_executor,
                args.factorized_direct_reverse_executor,
                args.factorized_reverse_cache_policy,
                args.factorized_planner_budget_bytes,
                args.standard_r0_executor,
                args.standard_m0_executor,
                args.factorized_parameter_gradients,
                args.nvtx,
                args.m1_polynomial_policy,
                args.m1_recompute_tile_channels,
                args.mh0_state_policy,
                args.edge_geometry_policy,
                args.readout_policy,
                args.phi1_policy,
                args.harmonic_storage_policy,
                args.neighbor_skin,
                args.execution_mh1_node_state_policy,
                calculator=reused_calculator,
            )
            for mode in modes
        }
        reference_mode = "materialized" if "materialized" in records else modes[0]
        reference = records[reference_mode]
        reference_forces = np.asarray(reference["forces_eV_per_A"])
        reference_stress = np.asarray(reference["stress_eV_per_A3"])
        directed_edges = reference["directed_edges"]
        for mode, record in records.items():
            forces = np.asarray(record.pop("forces_eV_per_A"))
            stress = np.asarray(record.pop("stress_eV_per_A3"))
            record["energy_error_eV"] = abs(
                record["energy_eV"] - reference["energy_eV"]
            )
            record["energy_error_eV_per_atom"] = record["energy_error_eV"] / len(atoms)
            record["force_max_error_eV_per_A"] = float(
                np.max(np.abs(forces - reference_forces))
            )
            record["stress_max_error_eV_per_A3"] = float(
                np.max(np.abs(stress - reference_stress))
            )
            speedup = reference["timing"]["median_ms"] / record["timing"]["median_ms"]
            record["speedup_vs_reference"] = speedup
            if reference_mode == "materialized":
                record["speedup_vs_materialized"] = speedup
            failed |= record["force_max_error_eV_per_A"] > args.max_force_error
            if args.min_speedup is not None and mode != reference_mode:
                failed |= speedup < args.min_speedup
            if args.max_us_per_atom is not None:
                failed |= (
                    record["timing_per_atom"]["median_us_per_atom"]
                    > args.max_us_per_atom
                )
        report["systems"].append(
            {
                "supercell_repeat": size,
                "atoms": len(atoms),
                "directed_edges": directed_edges,
                "reference_mode": reference_mode,
                "modes": records,
            }
        )

    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.write_text(serialized + "\n")
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        gc.collect()
        native = sys.modules.get("symmetrix.symmetrix")
        is_initialized = getattr(native, "_kokkos_is_initialized", None)
        finalize = getattr(native, "_finalize_kokkos", None)
        if callable(is_initialized) and callable(finalize) and is_initialized():
            finalize()
