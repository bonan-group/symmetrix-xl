"""Benchmark native dispersion through the public correction pipeline.

The first graph build and warmup evaluations are reported separately and are
excluded from ``reuse_step_us``. Run each backend in a fresh process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import shlex
import sys
import time
from pathlib import Path

try:
    from benchmarks.structure_workloads import balanced_repeat_factors
except ModuleNotFoundError as error:
    if error.name != "benchmarks":
        raise
    from structure_workloads import balanced_repeat_factors

THREAD_VARIABLES = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak_memory_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return "unavailable"
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return "unavailable"


def _cpu_topology_key(cpu: int) -> tuple[int, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = int((topology / "physical_package_id").read_text().strip())
        core = int((topology / "core_id").read_text().strip())
    except (FileNotFoundError, ValueError):
        return (0, cpu)
    return (package, core)


def _physical_cpu_affinity(
    available: set[int], first_cpu: int | None, cpu_count: int
) -> set[int]:
    if first_cpu is not None and first_cpu not in available:
        raise RuntimeError(f"CPU {first_cpu} is outside the process affinity mask")
    ordered = sorted(available)
    if first_cpu is not None:
        ordered.remove(first_cpu)
        ordered.insert(0, first_cpu)
    selected = []
    used_cores = set()
    for cpu in ordered:
        core = _cpu_topology_key(cpu)
        if core in used_cores:
            continue
        selected.append(cpu)
        used_cores.add(core)
        if len(selected) == cpu_count:
            return set(selected)
    raise RuntimeError(
        f"requested {cpu_count} physical CPUs, but only {len(selected)} are "
        "available in the process affinity mask"
    )


def _build_atoms(atom_count: int, structure: str):
    import numpy as np
    from ase import Atoms

    if structure == "srtio3":
        repeats = balanced_repeat_factors(atom_count // 5)
        unit_cell = Atoms(
            numbers=[38, 22, 8, 8, 8],
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            cell=np.eye(3) * 3.905,
            pbc=True,
        )
        return unit_cell.repeat(repeats), repeats

    if structure in {"graphite", "layered-bn"}:
        repeats = balanced_repeat_factors(atom_count // 4)
        a = 2.46
        unit_cell = Atoms(
            numbers=([6, 6, 6, 6] if structure == "graphite" else [5, 7, 5, 7]),
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [1.0 / 3.0, 2.0 / 3.0, 0.0],
                [0.0, 0.0, 0.5],
                [2.0 / 3.0, 1.0 / 3.0, 0.5],
            ],
            cell=[
                [a, 0.0, 0.0],
                [-0.5 * a, math.sqrt(3.0) * 0.5 * a, 0.0],
                [0.0, 0.0, 6.708],
            ],
            pbc=True,
        )
        return unit_cell.repeat(repeats), repeats

    side = math.ceil(atom_count ** (1.0 / 3.0))
    while side**3 < atom_count:
        side += 1
    spacing = 2.5
    indices = np.arange(atom_count, dtype=np.int64)
    positions = np.column_stack(
        (
            indices % side,
            (indices // side) % side,
            indices // (side * side),
        )
    ).astype(np.float64)
    positions *= spacing
    return (
        Atoms(
            numbers=np.full(atom_count, 6, dtype=np.int64),
            positions=positions,
            cell=np.eye(3) * (side * spacing),
            pbc=True,
        ),
        None,
    )


def _workspace_metrics(diagnostics: dict[str, object], method: str) -> dict[str, int]:
    if method != "d3":
        return {
            "reference_weight_active_atoms": 0,
            "reference_weight_capacity_atoms": 0,
            "reference_weight_workspace_bytes": 0,
            "coordination_cache_active_edges": 0,
            "coordination_cache_capacity_edges": 0,
            "coordination_cache_workspace_bytes": 0,
            "d3_workspace_bytes": 0,
        }

    names = (
        "reference_weight_active_atoms",
        "reference_weight_capacity_atoms",
        "reference_weight_workspace_bytes",
        "coordination_cache_active_edges",
        "coordination_cache_capacity_edges",
        "coordination_cache_workspace_bytes",
        "d3_workspace_bytes",
    )
    return {name: int(diagnostics[name]) for name in names}


def _configuration(args):
    from symmetrix import D2Correction, D3Correction
    from symmetrix.corrections import FourierD3Correction

    if args.method == "d2":
        return D2Correction(
            functional=args.functional,
            cutoff=args.pair_cutoff,
            skin=args.skin,
        )
    if args.method == "d3":
        return D3Correction(
            functional=args.functional,
            damping=args.damping,
            cutoff=args.pair_cutoff,
            coordination_cutoff=args.coordination_cutoff,
            skin=args.skin,
        )
    return FourierD3Correction(
        functional=args.functional,
        coordination_cutoff=args.coordination_cutoff,
        skin=args.skin,
        mesh_spacing=args.mesh_spacing,
        spline_order=args.spline_order,
        chunk_size=args.chunk_size,
    )


def _effective_cutoff(args) -> float:
    if args.method == "fourier-d3":
        return args.coordination_cutoff + args.skin
    return (
        max(
            args.pair_cutoff,
            args.coordination_cutoff if args.method == "d3" else 0.0,
        )
        + args.skin
    )


def _counter_delta(
    after: dict[str, object], before: dict[str, object], name: str
) -> int:
    return int(after.get(name, 0)) - int(before.get(name, 0))


def _validate_record(record: dict[str, object], args) -> None:
    unit_size = 5 if args.structure == "srtio3" else 4
    expected_repeats = None
    if args.structure in {"srtio3", "graphite", "layered-bn"}:
        expected_repeats = list(balanced_repeat_factors(args.atoms // unit_size))
    exact = {
        "schema": "symmetrix.dispersion-kokkos-timing",
        "version": 4,
        "structure": args.structure,
        "structure_repeats": expected_repeats,
        "properties": ["energy", "forces", "stress"],
        "metric": "us/atom",
        "backend": args.backend,
        "cpu_count": args.cpu_count if args.backend == "cpu" else None,
        "method": args.method,
        "damping": args.damping if args.method == "d3" else None,
        "functional": args.functional,
        "precision": args.precision,
        "atoms": args.atoms,
        "pair_cutoff": (None if args.method == "fourier-d3" else args.pair_cutoff),
        "coordination_cutoff": (
            None if args.method == "d2" else args.coordination_cutoff
        ),
        "skin": args.skin,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "effective_cutoff": _effective_cutoff(args),
    }
    mismatches = {
        name: (record.get(name), expected)
        for name, expected in exact.items()
        if record.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"benchmark record does not match arguments: {mismatches}")
    if (
        args.backend_selector is not None
        and record.get("backend_selector") != args.backend_selector
    ):
        raise RuntimeError("benchmark record does not match backend selector")
    if args.backend == "cpu":
        if len(record.get("cpu_affinity", ())) != args.cpu_count:
            raise RuntimeError("benchmark record does not match CPU count")
        thread_counts = record.get("thread_counts", {})
        if set(thread_counts) != set(THREAD_VARIABLES) or set(
            thread_counts.values()
        ) != {args.cpu_count}:
            raise RuntimeError("benchmark record does not match CPU thread counts")
    positive = (
        "directed_edges",
        "graph_build_us",
        "reuse_step_us",
        "peak_memory_bytes",
    )
    if any(not isinstance(record.get(name), (int, float)) for name in positive):
        raise RuntimeError("benchmark record is missing a numeric measurement")
    if any(record[name] <= 0 for name in positive):
        raise RuntimeError("benchmark measurements must be positive")
    if record.get("directed_edges") != record.get("num_edges"):
        raise RuntimeError("serialized edge count does not match native diagnostics")
    expected_reuses = args.warmup + args.repeat
    if record.get("graph_build_count") != 1:
        raise RuntimeError("steady-state benchmark rebuilt its correction graph")
    if record.get("graph_reuse_count") != expected_reuses:
        raise RuntimeError("benchmark did not reuse the graph for every timed step")
    if record.get("backend_fallback_count") != 0:
        raise RuntimeError("benchmark observed a backend fallback")
    result_fields = (
        "total_energy_ev",
        "energy_ev_per_atom",
        "force_rms_ev_per_angstrom",
        "out_of_plane_stress_ev_per_angstrom3",
    )
    if any(
        not isinstance(record.get(name), (int, float))
        or not math.isfinite(record[name])
        for name in result_fields
    ):
        raise RuntimeError("benchmark record has an invalid numerical result")
    stability_fields = (
        "timed_allocation_growth",
        "timed_capacity_growth_bytes",
        "timed_graph_rebuilds",
        "timed_plan_builds",
        "timed_mesh_invalidations",
        "timed_kernel_invalidations",
    )
    if any(
        not isinstance(record.get(name), int) or record[name] < 0
        for name in stability_fields
    ):
        raise RuntimeError("benchmark record has invalid steady-state deltas")
    if record["timed_graph_rebuilds"] != 0:
        raise RuntimeError("timed steps rebuilt the neighbor graph")
    workspace_extents = (
        "reference_weight_active_atoms",
        "reference_weight_capacity_atoms",
        "reference_weight_workspace_bytes",
        "coordination_cache_active_edges",
        "coordination_cache_capacity_edges",
        "coordination_cache_workspace_bytes",
        "d3_workspace_bytes",
    )
    if any(
        not isinstance(record.get(name), int) or record[name] < 0
        for name in workspace_extents
    ):
        raise RuntimeError("benchmark record has invalid workspace diagnostics")
    if args.method == "d3":
        if record["reference_weight_active_atoms"] != args.atoms:
            raise RuntimeError("reference-weight active extent does not match atoms")
        if record["coordination_cache_active_edges"] != record["directed_edges"]:
            raise RuntimeError("coordination-cache active extent does not match edges")
        if (
            record["reference_weight_capacity_atoms"]
            < record["reference_weight_active_atoms"]
            or record["coordination_cache_capacity_edges"]
            < record["coordination_cache_active_edges"]
        ):
            raise RuntimeError("workspace capacity is smaller than its active extent")
        precision_bytes = 4 if args.precision == "float32" else 8
        expected_reference_bytes = (
            2 * record["reference_weight_capacity_atoms"] * 5 * precision_bytes
        )
        expected_coordination_bytes = (
            2 * record["coordination_cache_capacity_edges"] * precision_bytes
        )
        if record["reference_weight_workspace_bytes"] != expected_reference_bytes:
            raise RuntimeError("reference-weight workspace byte count is invalid")
        if record["coordination_cache_workspace_bytes"] != expected_coordination_bytes:
            raise RuntimeError("coordination-cache workspace byte count is invalid")
        if record["d3_workspace_bytes"] != (
            expected_reference_bytes + expected_coordination_bytes
        ):
            raise RuntimeError("D3 workspace byte count is invalid")
    elif any(record[name] != 0 for name in workspace_extents):
        raise RuntimeError("non-real-space-D3 benchmark reported D3 workspace")
    if args.method == "fourier-d3":
        fourier_exact = {
            "mesh_spacing_requested": args.mesh_spacing,
            "spline_order": args.spline_order,
            "chunk_size_requested": args.chunk_size,
        }
        if any(record.get(name) != value for name, value in fourier_exact.items()):
            raise RuntimeError("FourierD3 accuracy policy does not match arguments")
        if any(record[name] != 0 for name in stability_fields):
            raise RuntimeError("FourierD3 grew or rebuilt state during timed reuse")
        mesh = record.get("resolved_mesh")
        if (
            not isinstance(mesh, (list, tuple))
            or len(mesh) != 3
            or any(not isinstance(axis, int) or axis < 1 for axis in mesh)
        ):
            raise RuntimeError("FourierD3 resolved mesh is invalid")
        if not record.get("capability_provider"):
            raise RuntimeError("FourierD3 FFT provider is missing")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cpu", "cuda", "hip"), required=True)
    parser.add_argument("--backend-selector")
    parser.add_argument("--method", choices=("d2", "d3", "fourier-d3"), required=True)
    parser.add_argument(
        "--damping", choices=("zero", "bj", "zerom", "bjm"), default="bj"
    )
    parser.add_argument("--functional", default="pbe")
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--structure",
        choices=("carbon-grid", "srtio3", "graphite", "layered-bn"),
        default="carbon-grid",
    )
    parser.add_argument("--atoms", type=_positive_int, required=True)
    parser.add_argument("--pair-cutoff", type=_positive_float)
    parser.add_argument("--coordination-cutoff", type=_positive_float, default=8.0)
    parser.add_argument("--mesh-spacing", type=_positive_float, default=1.0)
    parser.add_argument("--spline-order", type=int, choices=range(2, 7), default=4)
    parser.add_argument("--chunk-size", type=_positive_int, default=512)
    parser.add_argument("--skin", type=_positive_float, required=True)
    parser.add_argument("--warmup", type=_positive_int, default=3)
    parser.add_argument("--repeat", type=_positive_int, default=10)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cpu-count", type=_positive_int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.method in {"d2", "d3"} and args.pair_cutoff is None:
        parser.error("--pair-cutoff is required for d2 and d3")
    if args.structure == "srtio3" and args.atoms % 5 != 0:
        parser.error("srtio3 atom count must be divisible by five")
    if args.structure in {"graphite", "layered-bn"} and args.atoms % 4 != 0:
        parser.error("layered atom count must be divisible by four")
    cpu_affinity = sorted(os.sched_getaffinity(0))
    if args.backend == "cpu":
        affinity = _physical_cpu_affinity(
            os.sched_getaffinity(0), args.cpu, args.cpu_count
        )
        os.sched_setaffinity(0, affinity)
        cpu_affinity = sorted(affinity)
        for variable in THREAD_VARIABLES:
            os.environ[variable] = str(args.cpu_count)
    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = "kokkos"

    import numpy as np
    from symmetrix.corrections import CorrectionPipeline

    import symmetrix

    selector = args.backend_selector or ("cpu" if args.backend == "cpu" else None)
    native = symmetrix.load_backend(selector)
    build_info = dict(getattr(native, "_backend_build_info", dict)())
    if build_info.get("backend") != args.backend:
        raise RuntimeError(
            f"loaded backend {build_info.get('backend')!r}, expected {args.backend!r}"
        )
    if not native._kokkos_is_initialized():
        native._init_kokkos()

    dtype = np.float32 if args.precision == "float32" else np.float64
    pipeline = CorrectionPipeline([_configuration(args)], native=native, dtype=dtype)
    atoms, structure_repeats = _build_atoms(args.atoms, args.structure)
    properties = ("energy", "forces", "stress")

    start = time.perf_counter_ns()
    pipeline.calculate(atoms, properties)
    graph_build_us = (time.perf_counter_ns() - start) / 1000.0

    reference_positions = atoms.positions.copy()
    for step in range(args.warmup):
        atoms.positions[:] = reference_positions
        atoms.positions[:, 0] += 0.01 if step % 2 == 0 else -0.01
        pipeline.calculate(atoms, properties)

    warm_diagnostics = dict(pipeline.diagnostics()[0])

    elapsed_ns = 0
    result = None
    for step in range(args.repeat):
        atoms.positions[:] = reference_positions
        atoms.positions[:, 1] += 0.01 if step % 2 == 0 else -0.01
        start = time.perf_counter_ns()
        result = pipeline.calculate(atoms, properties)
        elapsed_ns += time.perf_counter_ns() - start

    if result is None:  # pragma: no cover - argparse requires a positive repeat
        raise RuntimeError("benchmark produced no result")
    diagnostics = dict(pipeline.diagnostics()[0])
    native_method = diagnostics.pop("method", None)
    workspace = _workspace_metrics(diagnostics, args.method)
    extension = Path(native.__file__).resolve()
    record = {
        "schema": "symmetrix.dispersion-kokkos-timing",
        "version": 4,
        "structure": args.structure,
        "structure_repeats": (
            list(structure_repeats) if structure_repeats is not None else None
        ),
        "properties": list(properties),
        "metric": "us/atom",
        "backend": args.backend,
        "cpu_count": args.cpu_count if args.backend == "cpu" else None,
        "backend_selector": symmetrix.selected_backend()["selector"],
        "architecture": build_info.get("architecture", ""),
        "execution_space": native._kokkos_default_execution_space(),
        "cpu_model": _cpu_model(),
        "cpu_affinity": cpu_affinity,
        "thread_counts": {
            name: int(os.environ[name])
            for name in THREAD_VARIABLES
            if name in os.environ
        },
        "source_fingerprint": build_info.get("source_content_sha256", ""),
        "native_content_sha256": build_info.get("native_source_content_sha256", ""),
        "extension_sha256": _sha256(extension),
        "command": shlex.join([sys.executable, *sys.argv]),
        "method": args.method,
        "native_method": native_method,
        "damping": args.damping if args.method == "d3" else None,
        "functional": args.functional,
        "precision": args.precision,
        "atoms": args.atoms,
        "pair_cutoff": (None if args.method == "fourier-d3" else args.pair_cutoff),
        "coordination_cutoff": (
            args.coordination_cutoff if args.method != "d2" else None
        ),
        "mesh_spacing_requested": (
            args.mesh_spacing if args.method == "fourier-d3" else None
        ),
        "spline_order_requested": (
            args.spline_order if args.method == "fourier-d3" else None
        ),
        "chunk_size_requested": (
            args.chunk_size if args.method == "fourier-d3" else None
        ),
        "skin": args.skin,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "effective_cutoff": _effective_cutoff(args),
        "directed_edges": diagnostics["num_edges"],
        "graph_build_us": graph_build_us,
        "reuse_step_us": elapsed_ns / args.repeat / args.atoms / 1000.0,
        "peak_memory_bytes": _peak_memory_bytes(),
        "total_energy_ev": float(result["energy"]),
        "energy_ev_per_atom": float(result["energy"]) / args.atoms,
        "force_rms_ev_per_angstrom": float(
            np.sqrt(np.mean(np.asarray(result["forces"], dtype=np.float64) ** 2))
        ),
        "out_of_plane_stress_ev_per_angstrom3": float(result["stress"][2]),
        "timed_allocation_growth": _counter_delta(
            diagnostics, warm_diagnostics, "allocations"
        ),
        "timed_capacity_growth_bytes": _counter_delta(
            diagnostics, warm_diagnostics, "capacity_bytes"
        ),
        "timed_graph_rebuilds": _counter_delta(
            diagnostics, warm_diagnostics, "graph_build_count"
        ),
        "timed_plan_builds": _counter_delta(
            diagnostics, warm_diagnostics, "plan_builds"
        ),
        "timed_mesh_invalidations": _counter_delta(
            diagnostics, warm_diagnostics, "mesh_invalidation_count"
        ),
        "timed_kernel_invalidations": _counter_delta(
            diagnostics, warm_diagnostics, "reciprocal_kernel_invalidation_count"
        ),
        **workspace,
        **diagnostics,
    }
    _validate_record(record, args)
    serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
    _validate_record(json.loads(serialized), args)
    if args.output is not None:
        args.output.write_text(serialized)
    sys.stdout.write(serialized)
    del pipeline
    gc.collect()
    if native._kokkos_is_initialized():
        native._finalize_kokkos()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
