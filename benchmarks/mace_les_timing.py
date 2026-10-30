"""Time one complete MACELES evaluation and emit one JSON record."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_THREADS", "1")
BLAS_THREAD_COUNT = os.environ.get("SYMMETRIX_BENCHMARK_BLAS_THREADS", "1")
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


RECORD_SCHEMA = "symmetrix.mace-les-timing/4"
REQUIRED_RECORD_FIELDS = frozenset(
    {
        "us_per_atom",
        "model_sha256",
        "model_type",
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
        "mace_architecture",
        "native_extension_sha256",
        "precision",
        "backend",
        "num_atoms",
        "cutoff",
        "skin",
        "directed_edges",
        "pbc",
        "sigma",
        "dl",
        "host_graph_constructions_before_timing",
        "host_graph_constructions_during_timing",
        "timed_operations",
        "excluded_operations",
        "streamed_edges",
        "jit_status",
        "jit_artifact_path",
        "prepared_evaluation_count",
        "fallback_evaluation_count",
        "graph_generation",
        "device_synchronization_bracketed_samples",
        "samples_ns",
        "synchronization",
        "profile_range",
        "replication",
        "cell_scale",
        "structure_sha256",
        "reciprocal_candidate_count",
        "reciprocal_vector_count",
        "atom_vector_pairs",
        "reciprocal_workspace_bytes",
        "prepared_graph_count",
        "source_provenance",
        "device_provenance",
        "native_graph_generation_before_warmup",
        "native_graph_generation_before_timing",
        "native_graph_generation_after_timing",
        "native_graph_rebuilds_during_warmup",
        "native_graph_rebuilds_during_timing",
        "interpreter_path",
        "python_version",
        "symmetrix_package_path",
        "symmetrix_version",
        "energy",
        "les_energy",
        "mace_les_energy",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mace_architecture(model, cutoff):
    les = model["les"]
    atomic_numbers = model.get("atomic_numbers", [])
    return {
        "mace_num_interactions": les["mace_num_interactions"],
        "mace_num_channels": les["mace_num_channels"],
        "mace_contract_fingerprints": dict(les["mace_contract_fingerprints"]),
        "atomic_numbers": list(atomic_numbers),
        "cutoff": float(cutoff),
    }


def _pin_one_cpu() -> list[int] | None:
    """Pin this fresh benchmark process before numerical runtimes are imported."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    if not available:
        return None
    requested = os.environ.get("SYMMETRIX_BENCHMARK_CPU")
    cpu = available[0] if requested is None else int(requested)
    if cpu not in available:
        raise RuntimeError(
            f"requested benchmark CPU {cpu} is outside the process affinity "
            f"set {available}"
        )
    os.sched_setaffinity(0, {cpu})
    return [cpu]


def _structure(
    *,
    periodic: bool,
    replicate: tuple[int, int, int],
    cell_scale: float,
    path: Path | None,
):
    import numpy as np
    from ase import Atoms
    from ase.io import read

    if path is None:
        atoms = Atoms(
            ["N", "Al", "N"],
            positions=[[0.1, 0.2, 0.3], [1.4, 0.4, 0.2], [0.5, 1.6, 0.7]],
            cell=np.array([[6.2, 0.2, 0.1], [0.0, 6.0, 0.3], [0.1, 0.0, 5.8]]),
        )
    else:
        atoms = read(path, index=0)
    atoms.pbc = periodic
    atoms = atoms.repeat(replicate)
    if cell_scale != 1.0:
        atoms.set_cell(atoms.cell.array * cell_scale, scale_atoms=True)
    return atoms


def _load_roctx_sdk():
    candidates = [
        "/opt/rocm/core-7.14/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib64/librocprofiler-sdk-roctx.so",
        ctypes.util.find_library("rocprofiler-sdk-roctx"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        required = (
            "roctxProfilerResume",
            "roctxRangePushA",
            "roctxRangePop",
            "roctxProfilerPause",
        )
        if not all(hasattr(library, symbol) for symbol in required):
            continue
        library.roctxProfilerResume.argtypes = [ctypes.c_uint64]
        library.roctxProfilerResume.restype = ctypes.c_int
        library.roctxRangePushA.argtypes = [ctypes.c_char_p]
        library.roctxRangePushA.restype = ctypes.c_int
        library.roctxRangePop.argtypes = []
        library.roctxRangePop.restype = ctypes.c_int
        library.roctxProfilerPause.argtypes = [ctypes.c_uint64]
        library.roctxProfilerPause.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("rocprofiler-sdk ROCTx library is unavailable")


def _load_hip_runtime():
    candidates = [ctypes.util.find_library("amdhip64"), "libamdhip64.so.7"]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if not hasattr(library, "hipDeviceSynchronize"):
            continue
        library.hipDeviceSynchronize.argtypes = []
        library.hipDeviceSynchronize.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("HIP runtime library is unavailable")


def _load_cuda_runtime():
    candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        required = ("cudaDeviceSynchronize", "cudaProfilerStart", "cudaProfilerStop")
        if not all(hasattr(library, symbol) for symbol in required):
            continue
        for symbol in required:
            operation = getattr(library, symbol)
            operation.argtypes = []
            operation.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("CUDA runtime library is unavailable")


def _load_nvtx():
    candidates = [ctypes.util.find_library("nvToolsExt"), "libnvToolsExt.so.1"]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if not all(
            hasattr(library, name) for name in ("nvtxRangePushA", "nvtxRangePop")
        ):
            continue
        library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        library.nvtxRangePushA.restype = ctypes.c_int
        library.nvtxRangePop.argtypes = []
        library.nvtxRangePop.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("NVTX library is unavailable")


def _roctx_state(library, operation, name):
    status = operation(0)
    if status != 0:
        raise RuntimeError(f"{name} failed with status {status}")


def _runtime_call(operation, name):
    status = operation()
    if status != 0:
        raise RuntimeError(f"{name} failed with status {status}")


def _device_synchronize(runtime, runtime_kind):
    if runtime_kind == "hip":
        _runtime_call(runtime.hipDeviceSynchronize, "hipDeviceSynchronize")
    elif runtime_kind == "cuda":
        _runtime_call(runtime.cudaDeviceSynchronize, "cudaDeviceSynchronize")
    elif runtime_kind != "none":
        raise ValueError(f"unsupported synchronization runtime {runtime_kind!r}")


def _hip_synchronize(library):
    _device_synchronize(library, "hip")


def _time_samples(
    evaluate,
    sample_count: int,
    *,
    runtime=None,
    runtime_kind="none",
    roctx=None,
    nvtx=None,
    hip=None,
) -> list[int]:
    # ``hip`` remains as a compatibility spelling for existing callers.
    if hip is not None:
        if runtime is not None:
            raise ValueError("pass either runtime or hip, not both")
        runtime, runtime_kind = hip, "hip"
    samples_ns = []
    profile_range_pushed = False
    cuda_profile_started = False
    if roctx is not None:
        _roctx_state(roctx, roctx.roctxProfilerResume, "roctxProfilerResume")
        level = roctx.roctxRangePushA(b"symmetrix-mace-les-samples")
        if level < 0:
            _roctx_state(roctx, roctx.roctxProfilerPause, "roctxProfilerPause")
            raise RuntimeError(f"roctxRangePushA failed with level {level}")
        profile_range_pushed = True
    if nvtx is not None:
        _runtime_call(runtime.cudaProfilerStart, "cudaProfilerStart")
        cuda_profile_started = True
        level = nvtx.nvtxRangePushA(b"symmetrix-mace-les-samples")
        if level < 0:
            _runtime_call(runtime.cudaProfilerStop, "cudaProfilerStop")
            raise RuntimeError(f"nvtxRangePushA failed with level {level}")
        profile_range_pushed = True
    try:
        for _ in range(sample_count):
            if runtime is not None:
                _device_synchronize(runtime, runtime_kind)
            started = time.perf_counter_ns()
            evaluate()
            if runtime is not None:
                _device_synchronize(runtime, runtime_kind)
            samples_ns.append(time.perf_counter_ns() - started)
    finally:
        if roctx is not None and profile_range_pushed:
            level = roctx.roctxRangePop()
            _roctx_state(roctx, roctx.roctxProfilerPause, "roctxProfilerPause")
            if level < 0:
                raise RuntimeError(f"roctxRangePop failed with level {level}")
        if nvtx is not None and profile_range_pushed:
            level = nvtx.nvtxRangePop()
            if cuda_profile_started:
                _runtime_call(runtime.cudaProfilerStop, "cudaProfilerStop")
            if level < 0:
                raise RuntimeError(f"nvtxRangePop failed with level {level}")
    return samples_ns


def _reciprocal_candidate_count(cell, dl):
    import numpy as np

    counts = [max(1, math.floor(float(np.linalg.norm(row)) / dl)) for row in cell]
    return math.prod(2 * count + 1 for count in counts)


def _graph_reuse_diagnostics(
    before_warmup,
    before_timing,
    after_timing,
    *,
    require_prepared_graph=True,
):
    generations = (before_warmup, before_timing, after_timing)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in generations
    ):
        raise RuntimeError("native graph generations must be non-negative integers")
    warmup_rebuilds = before_timing - before_warmup
    timed_rebuilds = after_timing - before_timing
    if warmup_rebuilds < 0 or timed_rebuilds < 0:
        raise RuntimeError("native prepared graph generation moved backwards")
    if require_prepared_graph and before_timing == 0:
        raise RuntimeError("native prepared graph was not built before timing")
    if timed_rebuilds:
        raise RuntimeError("native prepared graph rebuilt inside the timed region")
    return {
        "native_graph_generation_before_warmup": before_warmup,
        "native_graph_generation_before_timing": before_timing,
        "native_graph_generation_after_timing": after_timing,
        "native_graph_rebuilds_during_warmup": warmup_rebuilds,
        "native_graph_rebuilds_during_timing": timed_rebuilds,
        "prepared_graph_count": int(before_timing > 0),
    }


def parse_record(rendered: str) -> dict:
    """Parse and validate one record emitted by this driver."""

    try:
        record = json.loads(rendered)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid MACELES benchmark JSON: {error}") from error
    if not isinstance(record, dict):
        raise TypeError("MACELES benchmark output must be a JSON object")
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(
            f"unsupported MACELES benchmark schema {record.get('schema')!r}"
        )
    missing = sorted(REQUIRED_RECORD_FIELDS - record.keys())
    if missing:
        raise ValueError(
            "MACELES benchmark record is missing fields: " + ", ".join(missing)
        )
    samples = record.get("samples_ns")
    if (
        not isinstance(samples, list)
        or len(samples) != record.get("samples")
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in samples
        )
    ):
        raise ValueError("MACELES benchmark samples_ns are invalid")
    replication = record.get("replication")
    if (
        not isinstance(replication, list)
        or len(replication) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in replication
        )
    ):
        raise ValueError("MACELES benchmark replication must contain three positives")
    if (
        isinstance(record.get("cell_scale"), bool)
        or not isinstance(record.get("cell_scale"), (int, float))
        or not math.isfinite(record["cell_scale"])
        or record["cell_scale"] <= 0
    ):
        raise ValueError("MACELES benchmark cell_scale must be positive and finite")
    reciprocal_fields = (
        "reciprocal_candidate_count",
        "reciprocal_vector_count",
        "atom_vector_pairs",
        "reciprocal_workspace_bytes",
    )
    if record.get("model_type") != "MACELES":
        raise ValueError("MACELES benchmark model_type must be MACELES")
    for name in (
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
    ):
        value = record.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"MACELES benchmark {name} must be a lowercase SHA-256")
    if not isinstance(record.get("mace_architecture"), dict):
        raise TypeError("MACELES benchmark mace_architecture must be an object")
    periodic_les = record.get("pbc") == "periodic"
    reciprocal_values = [record.get(name) for name in reciprocal_fields]
    if periodic_les:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in reciprocal_values
        ):
            raise ValueError("periodic LES reciprocal fields must be positive integers")
        if record["reciprocal_vector_count"] > record["reciprocal_candidate_count"]:
            raise ValueError("active reciprocal vectors exceed candidate count")
        if (
            record["atom_vector_pairs"]
            != record["num_atoms"] * record["reciprocal_vector_count"]
        ):
            raise ValueError("atom_vector_pairs is inconsistent with the workload")
    elif any(value is not None for value in reciprocal_values):
        raise ValueError("reciprocal fields must be null outside periodic LES")
    for name in (
        "synchronization",
        "profile_range",
        "source_provenance",
        "device_provenance",
    ):
        if not isinstance(record.get(name), dict):
            raise TypeError(f"MACELES benchmark {name} must be an object")
    for name in (
        "interpreter_path",
        "python_version",
        "symmetrix_package_path",
        "symmetrix_version",
    ):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError(f"MACELES benchmark {name} must be non-empty")
    if record.get("native_graph_rebuilds_during_timing") != 0:
        raise ValueError("MACELES benchmark rebuilt the native graph while timing")
    for name in ("energy", "les_energy", "mace_les_energy"):
        value = record.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"MACELES benchmark {name} must be finite")
    if not math.isclose(record["energy"], record["mace_les_energy"], rel_tol=1e-12):
        raise ValueError("MACELES energy does not match mace_les_energy")
    return record


def validate_requested_workload(
    record: dict,
    *,
    num_atoms: int,
    pbc: str,
    precision: str,
    replication: list[int] | None = None,
    cell_scale: float | None = None,
) -> None:
    """Reject a record that does not describe the requested workload."""

    expected = {
        "num_atoms": num_atoms,
        "pbc": pbc,
        "precision": precision,
    }
    if replication is not None:
        expected["replication"] = replication
    if cell_scale is not None:
        expected["cell_scale"] = cell_scale
    mismatches = [
        f"{name}={record.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if record.get(name) != value
    ]
    if mismatches:
        raise ValueError("benchmark workload mismatch: " + "; ".join(mismatches))
    edges = record.get("directed_edges")
    if isinstance(edges, bool) or not isinstance(edges, int) or edges < 0:
        raise ValueError("benchmark directed_edges must be a non-negative integer")


def _prebuilt_graph_evaluator(calculator, atoms, properties):
    """Build the host graph once and return the model-only evaluation call."""

    graph_constructions = 0
    build_graph = calculator._mace_inputs

    def counted_build_graph(*args, **kwargs):
        nonlocal graph_constructions
        graph_constructions += 1
        return build_graph(*args, **kwargs)

    # Keep the counter installed so any later calculator path that reconstructs the
    # host graph is visible to the timed-region invariant below.
    calculator._mace_inputs = counted_build_graph
    mace_inputs = calculator._mace_inputs(atoms)
    num_nodes, _, _, sources, _, _, _, _ = mace_inputs
    calculator._apply_kernel_launch_tuning(num_nodes, len(sources), properties)

    def evaluate():
        calculator._set_current_les_geometry(atoms)
        graph_generation = calculator._compute_mace(mace_inputs, atoms=atoms)
        calculator.results = calculator._collect_mace_results(
            atoms,
            mace_inputs,
            properties,
            graph_generation,
        )
        calculator._publish_les_results(atoms)

    def restore_graph_builder():
        calculator._mace_inputs = build_graph

    return (
        evaluate,
        lambda: graph_constructions,
        len(sources),
        restore_graph_builder,
    )


def run(args: argparse.Namespace) -> dict:
    affinity = _pin_one_cpu()

    # Numerical libraries are deliberately imported only after thread placement.
    import numpy as np

    import symmetrix as symmetrix_package
    from symmetrix import Symmetrix, backend_loader, selected_backend

    model_path = args.model.resolve()
    model = json.loads(model_path.read_text())
    try:
        if model["model_type"] != "MACELES":
            raise ValueError("model_type must be MACELES")
        ewald = model["les"]["ewald"]
        sigma = float(ewald["sigma"])
        dl = float(ewald["dl"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{model_path} does not contain valid MACELES parameters"
        ) from error

    atoms = _structure(
        periodic=args.pbc == "periodic",
        replicate=tuple(args.replicate),
        cell_scale=args.cell_scale,
        path=None if args.structure is None else args.structure.resolve(),
    )
    calculator = Symmetrix(
        model_path,
        dtype=args.precision,
        streamed_edges=args.streamed_edges,
        execution_profile=args.execution_profile,
        neighbor_skin=args.skin,
    )
    if calculator.model_type != "MACELES":
        raise RuntimeError("model did not construct a MACELES evaluator")
    properties = ["energy", "energies", "forces", "stress"]

    # Construct the host graph once, then time native forward/reverse and result
    # reduction against those fixed graph arrays. ASE state handling and neighbor-list
    # construction stay out of both warmup and sample timings.
    (
        evaluate,
        graph_construction_count,
        directed_edges,
        restore_graph_builder,
    ) = _prebuilt_graph_evaluator(calculator, atoms, properties)
    graph_generation_before_warmup = int(
        calculator.evaluator.factorized_graph_generation
    )
    for _ in range(args.warmups):
        evaluate()
    graph_generation_before_timing = int(
        calculator.evaluator.factorized_graph_generation
    )

    descriptor = selected_backend()
    if descriptor is None:
        raise RuntimeError("MACELES evaluation did not select a backend")
    native = backend_loader._native_module
    if native is None:
        raise RuntimeError("MACELES evaluation did not load a native backend")
    execution_space = native._kokkos_default_execution_space()
    runtime = None
    runtime_kind = "none"
    runtime_path = None
    roctx = None
    roctx_path = None
    nvtx = None
    nvtx_path = None
    if execution_space == "HIP":
        runtime, runtime_path = _load_hip_runtime()
        runtime_kind = "hip"
    elif execution_space in {"Cuda", "CUDA"}:
        runtime, runtime_path = _load_cuda_runtime()
        runtime_kind = "cuda"
    if runtime is not None:
        _device_synchronize(runtime, runtime_kind)
    if args.profile_region and runtime_kind == "hip":
        roctx, roctx_path = _load_roctx_sdk()
        _roctx_state(roctx, roctx.roctxProfilerPause, "roctxProfilerPause")
    elif args.profile_region and runtime_kind == "cuda":
        nvtx, nvtx_path = _load_nvtx()
    elif args.profile_region:
        raise RuntimeError("--profile-region requires a HIP or CUDA backend")

    graph_constructions_before_timing = graph_construction_count()
    samples_ns = _time_samples(
        evaluate,
        args.samples,
        runtime=runtime,
        runtime_kind=runtime_kind,
        roctx=roctx,
        nvtx=nvtx,
    )
    graph_constructions_during_timing = (
        graph_construction_count() - graph_constructions_before_timing
    )
    if graph_constructions_during_timing:
        raise RuntimeError(
            "MACELES benchmark reconstructed the host graph inside its timed region"
        )
    graph_generation_after_timing = int(
        calculator.evaluator.factorized_graph_generation
    )
    graph_diagnostics = _graph_reuse_diagnostics(
        graph_generation_before_warmup,
        graph_generation_before_timing,
        graph_generation_after_timing,
        require_prepared_graph=args.streamed_edges == "direct",
    )

    results = calculator.results
    for name in properties:
        if not np.all(np.isfinite(results[name])):
            raise RuntimeError(f"MACELES benchmark produced non-finite {name}")
    if not math.isclose(
        float(np.asarray(results["energies"]).sum()),
        float(results["energy"]),
        rel_tol=2.0e-6,
        abs_tol=2.0e-7,
    ):
        raise RuntimeError("MACELES atomic energies do not sum to total energy")

    native_path = Path(native.__file__).resolve()
    atom_count = len(atoms)
    median_ns = statistics.median(samples_ns)
    periodic_les = args.pbc == "periodic"
    monopole = (
        dict(calculator.evaluator.les_monopole_diagnostics) if periodic_les else {}
    )
    reciprocal_vectors = monopole.get("reciprocal_vector_count")
    candidates = (
        _reciprocal_candidate_count(atoms.cell.array, dl) if periodic_les else None
    )
    workspace_bytes = monopole.get("capacity_bytes") if periodic_les else None
    structure_path = None if args.structure is None else args.structure.resolve()
    device_environment = dict(
        getattr(calculator.evaluator, "execution_device_execution_environment", {})
    )
    record = {
        "schema": RECORD_SCHEMA,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_type": calculator.model_type,
        "mace_les_parameters_sha256": calculator.mace_les_parameters_sha256,
        "mace_les_evaluator_fingerprint": (calculator.mace_les_evaluator_fingerprint),
        "mace_architecture": _mace_architecture(model, calculator.cutoff),
        "native_extension": str(native_path),
        "native_extension_sha256": _sha256(native_path),
        "native_build": native._backend_build_info(),
        "backend": descriptor["backend"],
        "backend_selector": descriptor["selector"],
        "kokkos_execution_space": execution_space,
        "execution_mode": (
            "prepared-direct"
            if args.streamed_edges == "direct"
            else "generic-full-graph"
        ),
        "streamed_edges": args.streamed_edges,
        "execution_profile": args.execution_profile,
        "jit_status": calculator.jit_status,
        "jit_artifact_path": calculator.jit_artifact_path,
        "prepared_evaluation_count": int(
            calculator.evaluator.factorized_prepared_evaluation_count
        ),
        "fallback_evaluation_count": int(
            calculator.evaluator.factorized_fallback_evaluation_count
        ),
        "graph_generation": graph_generation_after_timing,
        **graph_diagnostics,
        "device_synchronization_bracketed_samples": runtime is not None,
        "profile_region": bool(args.profile_region),
        "roctx_library": roctx_path,
        "hip_runtime_library": runtime_path if runtime_kind == "hip" else None,
        "cuda_runtime_library": runtime_path if runtime_kind == "cuda" else None,
        "nvtx_library": nvtx_path,
        "synchronization": {"backend": runtime_kind, "runtime_library": runtime_path},
        "profile_range": {
            "enabled": bool(args.profile_region),
            "api": "roctx"
            if roctx is not None
            else "nvtx"
            if nvtx is not None
            else None,
            "name": "symmetrix-mace-les-samples" if args.profile_region else None,
            "cuda_profiler_api": nvtx is not None,
        },
        "precision": args.precision,
        "num_atoms": atom_count,
        "replication": list(args.replicate),
        "cell_scale": args.cell_scale,
        "structure_sha256": _sha256(structure_path)
        if structure_path is not None
        else None,
        "cutoff": float(calculator.cutoff),
        "skin": float(calculator.neighbor_skin),
        "effective_cutoff": float(calculator.cutoff + calculator.neighbor_skin),
        "directed_edges": int(directed_edges),
        "pbc": args.pbc,
        "sigma": sigma,
        "dl": dl,
        "warmups": args.warmups,
        "samples": args.samples,
        "samples_ns": samples_ns,
        "host_graph_constructions_before_timing": (graph_constructions_before_timing),
        "host_graph_constructions_during_timing": (graph_constructions_during_timing),
        "timed_operations": [
            "LES geometry handoff",
            "native model forward/reverse",
            "energy/force/stress reduction",
            "result materialization",
        ],
        "excluded_operations": [
            "model and calculator construction",
            "host neighbor-list construction",
            "host graph array construction",
            "warmup evaluations",
        ],
        "samples_us_per_atom": [value / 1000.0 / atom_count for value in samples_ns],
        "us_per_atom": median_ns / 1000.0 / atom_count,
        "ms_per_evaluation": median_ns / 1.0e6,
        "reciprocal_candidate_count": candidates,
        "reciprocal_vector_count": reciprocal_vectors,
        "atom_vector_pairs": atom_count * reciprocal_vectors if periodic_les else None,
        "reciprocal_workspace_bytes": workspace_bytes,
        "interpreter_path": str(Path(sys.executable).absolute()),
        "python_version": platform.python_version(),
        "symmetrix_package_path": str(Path(symmetrix_package.__file__).resolve()),
        "symmetrix_version": importlib.metadata.version("symmetrix-xl"),
        "source_provenance": {
            "source_commit": descriptor.get("source_commit"),
            "source_dirty": descriptor.get("source_dirty"),
            "source_content_sha256": descriptor.get("source_content_sha256"),
            "native_source_content_sha256": descriptor.get(
                "native_source_content_sha256"
            ),
        },
        "device_provenance": device_environment
        or {"backend": descriptor["backend"], "device_name": "host"},
        "cpu_affinity": affinity,
        "thread_counts": {
            "kokkos_openmp": int(THREAD_COUNT),
            "blas": int(BLAS_THREAD_COUNT),
        },
        "energy": float(results["energy"]),
        "les_energy": (
            float(results["les_energy"]) if "les_energy" in results else None
        ),
        "mace_les_energy": (
            float(results["mace_les_energy"]) if "mace_les_energy" in results else None
        ),
    }
    validate_requested_workload(
        record,
        num_atoms=atom_count,
        pbc=args.pbc,
        precision=args.precision,
        replication=list(args.replicate),
        cell_scale=args.cell_scale,
    )

    restore_graph_builder()
    del evaluate, graph_construction_count, restore_graph_builder
    del calculator, atoms
    gc.collect()
    if native._kokkos_is_initialized():
        native._finalize_kokkos()
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--structure",
        type=Path,
        help="Read the first structure from this ASE-supported file",
    )
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--pbc", choices=("periodic", "nonperiodic"), default="periodic"
    )
    parser.add_argument(
        "--replicate", type=int, nargs=3, metavar=("NX", "NY", "NZ"), default=(1, 1, 1)
    )
    parser.add_argument("--cell-scale", type=float, default=1.0)
    parser.add_argument("--skin", type=float, default=0.0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--streamed-edges", choices=("reference", "direct"), default="reference"
    )
    parser.add_argument(
        "--execution-profile", choices=("speed", "capacity"), default="speed"
    )
    parser.add_argument(
        "--profile-region",
        action="store_true",
        help="Expose synchronized HIP/ROCTx or CUDA/NVTX samples to a profiler",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if any(value < 1 for value in args.replicate):
        parser.error("replication counts must be positive")
    if not math.isfinite(args.cell_scale) or args.cell_scale <= 0.0:
        parser.error("cell scale must be finite and positive")
    if not math.isfinite(args.skin) or args.skin < 0.0:
        parser.error("skin must be a finite non-negative number")
    if args.warmups < 0 or args.samples < 1:
        parser.error("warmups must be non-negative and samples must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
