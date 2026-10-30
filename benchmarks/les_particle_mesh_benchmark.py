"""Compare explicit-sum and particle-mesh LES stages in complete MACELES runs.

Run both algorithms with identical arguments, then pass their JSON files
to ``--compare``.  Construction, neighbor-list generation, JIT compilation,
FFT-plan creation, and warmup are excluded from sample timings.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path

THREAD_VARIABLES = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _variable in THREAD_VARIABLES:
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

SCHEMA = "symmetrix.les-particle-mesh-benchmark/2"
ALGORITHMS = ("explicit_sum", "particle_mesh")
PRECISIONS = ("float32", "float64")
STREAMED_EDGE_MODES = ("reference", "direct")
EXECUTION_SPACES_BY_BACKEND = {
    "cpu": frozenset(("OpenMP", "Serial")),
    "cuda": frozenset(("Cuda",)),
    "hip": frozenset(("HIP",)),
}
WATER_MOLECULES = 64
WATER_DENSITY_G_CM3 = 0.997
AMU_PER_ANGSTROM3_TO_G_PER_CM3 = 1.66053906660
REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "algorithm",
        "model",
        "model_sha256",
        "structure_sha256",
        "model_type",
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
        "mace_architecture",
        "source_provenance",
        "native_extension",
        "native_extension_sha256",
        "backend",
        "kokkos_execution_space",
        "precision",
        "atoms",
        "cell",
        "sigma",
        "dl",
        "mesh",
        "spline_order",
        "directed_edges",
        "streamed_edges",
        "skin",
        "samples_ns",
        "sample_count",
        "median_ns",
        "us_per_atom",
        "warmups",
        "host_graph_constructions_during_timing",
        "fallback_evaluation_count",
        "jit_status",
        "jit_artifact_path",
        "cpu_affinity",
        "thread_counts",
        "synchronization",
        "profile_range",
        "interpreter_path",
        "python_version",
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


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def _pin_one_cpu() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    if not available:
        return None
    requested = os.environ.get("SYMMETRIX_BENCHMARK_CPU")
    cpu = available[0] if requested is None else int(requested)
    if cpu not in available:
        raise RuntimeError(f"CPU {cpu} is outside affinity set {available}")
    os.sched_setaffinity(0, {cpu})
    return [cpu]


def _positive_samples(samples: object) -> bool:
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value > 0
            for value in samples
        )
    )


def _finite_positive(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0.0
    )


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0.0
    )


def _nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _finite_vector(value: object, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(
            not isinstance(component, bool)
            and isinstance(component, (int, float))
            and math.isfinite(component)
            for component in value
        )
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def example_record_for_testing(
    *, algorithm: str = "particle_mesh", median_ns: int = 1_200_000
) -> dict[str, object]:
    """Return a valid synthetic record for contract mutation tests."""

    digest = "1" * 64
    atoms = 100
    mesh = [32, 32, 32] if algorithm == "particle_mesh" else None
    order = 6 if algorithm == "particle_mesh" else None
    return {
        "schema": SCHEMA,
        "algorithm": algorithm,
        "model": "/qualified/model.json",
        "model_sha256": digest,
        "structure_sha256": digest,
        "model_type": "MACELES",
        "mace_les_parameters_sha256": digest,
        "mace_les_evaluator_fingerprint": digest,
        "mace_architecture": {
            "mace_num_interactions": 2,
            "mace_num_channels": 128,
            "mace_contract_fingerprints": {
                "R0": f"sha256:{digest}",
                "M0": f"sha256:{digest}",
                "R1": f"sha256:{digest}",
            },
            "atomic_numbers": [1, 8],
            "cutoff": 5.0,
        },
        "source_provenance": {
            "source_commit": "test",
            "source_dirty": False,
            "source_content_sha256": digest,
            "native_source_content_sha256": digest,
        },
        "native_extension": "/qualified/_native_cpu.so",
        "native_extension_sha256": digest,
        "backend": "cpu",
        "kokkos_execution_space": "OpenMP",
        "precision": "float64",
        "atoms": atoms,
        "cell": [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]],
        "sigma": 1.0,
        "dl": 2.0,
        "mesh": mesh,
        "spline_order": order,
        "directed_edges": 800,
        "streamed_edges": "direct",
        "skin": 0.0,
        "samples_ns": [median_ns, median_ns, median_ns],
        "sample_count": 3,
        "median_ns": median_ns,
        "us_per_atom": median_ns / 1000.0 / atoms,
        "warmups": 3,
        "host_graph_constructions_during_timing": 0,
        "fallback_evaluation_count": 0,
        "jit_status": "cached",
        "jit_artifact_path": "/qualified/cache/operator.so",
        "cpu_affinity": [2],
        "thread_counts": {name: 1 for name in THREAD_VARIABLES},
        "synchronization": {"backend": "none", "runtime_library": None},
        "profile_range": {"enabled": False, "api": None, "name": None},
        "interpreter_path": "/qualified/venv/bin/python",
        "python_version": "3.12.13",
        "energy": -1.0,
        "les_energy": 0.2,
        "mace_les_energy": -1.0,
    }


def validate_record(record: dict[str, object]) -> dict[str, object]:
    """Validate one timing record before it is written or compared."""

    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError("benchmark record is missing: " + ", ".join(missing))
    if record["schema"] != SCHEMA:
        raise ValueError(f"unsupported benchmark schema {record['schema']!r}")
    algorithm = record["algorithm"]
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unsupported benchmark algorithm {algorithm!r}")
    if record["model_type"] != "MACELES":
        raise ValueError("benchmark model_type must be MACELES")
    if not isinstance(record["mace_architecture"], Mapping):
        raise TypeError("benchmark mace_architecture must be an object")
    cell = record["cell"]
    if (
        not isinstance(cell, list)
        or len(cell) != 3
        or not all(_finite_vector(row, 3) for row in cell)
    ):
        raise ValueError("benchmark cell must be a finite 3x3 matrix")
    if record["precision"] not in PRECISIONS:
        raise ValueError(f"unsupported benchmark precision {record['precision']!r}")
    if record["streamed_edges"] not in STREAMED_EDGE_MODES:
        raise ValueError(
            f"unsupported benchmark streamed_edges {record['streamed_edges']!r}"
        )
    if not _finite_nonnegative(record["skin"]):
        raise ValueError("benchmark skin must be finite and non-negative")
    backend = record["backend"]
    execution_spaces = EXECUTION_SPACES_BY_BACKEND.get(backend)
    if execution_spaces is None:
        raise ValueError(f"unsupported benchmark backend {backend!r}")
    if record["kokkos_execution_space"] not in execution_spaces:
        raise ValueError(
            "benchmark kokkos_execution_space is inconsistent with backend"
        )
    mesh, order = record["mesh"], record["spline_order"]
    if algorithm == "particle_mesh":
        if (
            not isinstance(mesh, list)
            or len(mesh) != 3
            or len(set(mesh)) != 1
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in mesh
            )
        ):
            raise ValueError("particle-mesh benchmark mesh must be isotropic")
        if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= 6:
            raise ValueError("particle-mesh benchmark spline_order is invalid")
    elif mesh is not None or order is not None:
        raise ValueError(f"{algorithm} benchmark must not report a mesh or order")
    for name in ("sigma", "dl"):
        if not _finite_positive(record[name]):
            raise ValueError(f"benchmark {name} must be finite and positive")
    if not _nonnegative_integer(record["directed_edges"]):
        raise ValueError("benchmark directed_edges must be a non-negative integer")
    if not _nonnegative_integer(record["warmups"]):
        raise ValueError("benchmark warmups must be a non-negative integer")
    if not _positive_samples(record["samples_ns"]):
        raise ValueError("benchmark samples_ns must contain positive timings")
    sample_count = record["sample_count"]
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count < 1
        or sample_count != len(record["samples_ns"])
    ):
        raise ValueError("benchmark sample_count must match samples_ns")
    median_ns = statistics.median(record["samples_ns"])
    if record["median_ns"] != median_ns:
        raise ValueError("benchmark median_ns does not match samples")
    atoms = record["atoms"]
    if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 1:
        raise ValueError("benchmark atoms must be positive")
    expected_us_per_atom = median_ns / 1000.0 / atoms
    if not math.isclose(record["us_per_atom"], expected_us_per_atom, rel_tol=1e-12):
        raise ValueError("benchmark us_per_atom does not match samples and atoms")
    if record["host_graph_constructions_during_timing"] != 0:
        raise ValueError("benchmark reconstructed the host graph during timing")
    if record["streamed_edges"] == "direct" and record["fallback_evaluation_count"]:
        raise ValueError("direct benchmark used a fallback evaluation")
    threads = record["thread_counts"]
    if not isinstance(threads, dict) or set(threads) != set(THREAD_VARIABLES):
        raise ValueError("benchmark thread_counts are incomplete")
    if record["backend"] == "cpu":
        if record["cpu_affinity"] is None or len(record["cpu_affinity"]) != 1:
            raise ValueError("CPU benchmark must use one-CPU affinity")
        if any(value != 1 for value in threads.values()):
            raise ValueError("CPU benchmark thread counts must all equal one")
    provenance = record["source_provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("benchmark source_provenance must be an object")
    source_commit = provenance.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("source_provenance.source_commit must be a nonempty string")
    if "source_dirty" not in provenance:
        raise ValueError("source_provenance.source_dirty is required")
    if not isinstance(provenance["source_dirty"], bool):
        raise TypeError("source_provenance.source_dirty must be a boolean")
    for name in ("source_content_sha256", "native_source_content_sha256"):
        value = provenance.get(name)
        if not _is_sha256(value):
            raise ValueError(f"source_provenance.{name} must be a lowercase SHA-256")
    for name in (
        "model_sha256",
        "structure_sha256",
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
        "native_extension_sha256",
    ):
        value = record[name]
        if not _is_sha256(value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    for name in ("energy", "les_energy", "mace_les_energy"):
        if not isinstance(record[name], (int, float)) or not math.isfinite(
            record[name]
        ):
            raise ValueError(f"benchmark {name} must be finite")
    if not math.isclose(record["energy"], record["mace_les_energy"], rel_tol=1e-12):
        raise ValueError("benchmark energy does not match mace_les_energy")
    return record


def compare_records(
    records: list[dict[str, object]],
) -> dict[str, float | int | None]:
    """Compare complete MACELES evaluations from identity-matched records."""

    by_algorithm = {}
    for record in records:
        validate_record(record)
        algorithm = record["algorithm"]
        if algorithm in by_algorithm:
            raise ValueError(f"duplicate {algorithm} benchmark record")
        by_algorithm[algorithm] = record
    if set(by_algorithm) != set(ALGORITHMS):
        raise ValueError("comparison requires explicit_sum and particle_mesh")
    anchor = by_algorithm["explicit_sum"]
    identity_fields = (
        "model_sha256",
        "structure_sha256",
        "mace_les_parameters_sha256",
        "mace_les_evaluator_fingerprint",
        "mace_architecture",
        "source_provenance",
        "native_extension_sha256",
        "backend",
        "kokkos_execution_space",
        "precision",
        "atoms",
        "cell",
        "sigma",
        "dl",
        "directed_edges",
        "streamed_edges",
        "skin",
        "warmups",
        "sample_count",
        "cpu_affinity",
        "thread_counts",
        "synchronization",
        "profile_range",
        "interpreter_path",
        "python_version",
    )
    for algorithm in ("particle_mesh",):
        mismatches = [
            name
            for name in identity_fields
            if by_algorithm[algorithm][name] != anchor[name]
        ]
        if mismatches:
            raise ValueError(
                f"{algorithm} workload differs from explicit_sum: {', '.join(mismatches)}"
            )
    explicit = float(anchor["us_per_atom"])
    particle_mesh = float(by_algorithm["particle_mesh"]["us_per_atom"])
    return {
        "atoms": int(anchor["atoms"]),
        "explicit_sum_us_per_atom": explicit,
        "particle_mesh_us_per_atom": particle_mesh,
        "particle_mesh_speedup": (
            explicit / particle_mesh if explicit > 0.0 and particle_mesh > 0.0 else None
        ),
    }


def _load_runtime(execution_space: str):
    if execution_space == "HIP":
        names = (ctypes.util.find_library("amdhip64"), "libamdhip64.so.7")
        symbol, kind = "hipDeviceSynchronize", "hip"
    elif execution_space in {"Cuda", "CUDA"}:
        names = (ctypes.util.find_library("cudart"), "libcudart.so")
        symbol, kind = "cudaDeviceSynchronize", "cuda"
    else:
        return None, "none", None
    for name in names:
        if name is None:
            continue
        try:
            library = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if hasattr(library, symbol):
            operation = getattr(library, symbol)
            operation.argtypes = []
            operation.restype = ctypes.c_int
            return library, kind, name
    raise RuntimeError(f"cannot load {kind} synchronization runtime")


def _synchronize(runtime, kind: str) -> None:
    if runtime is None:
        return
    operation = (
        runtime.hipDeviceSynchronize if kind == "hip" else runtime.cudaDeviceSynchronize
    )
    status = operation()
    if status != 0:
        raise RuntimeError(f"{kind} device synchronization failed with {status}")


def _roctx_library_candidates() -> list[str]:
    roots: list[Path] = []
    for variable in ("ROCM_PATH", "ROCM_ROOT"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))
    for tool in ("rocprofv3", "hipconfig"):
        path = shutil.which(tool)
        if path:
            roots.append(Path(path).resolve().parent.parent)
    roots.append(Path("/opt/rocm"))

    candidates: list[str] = []
    for root in roots:
        for directory in ("lib", "lib64"):
            candidates.append(str(root / directory / "librocprofiler-sdk-roctx.so"))
    discovered = ctypes.util.find_library("rocprofiler-sdk-roctx")
    if discovered:
        candidates.append(discovered)
    return list(dict.fromkeys(candidates))


def _load_roctx():
    for candidate in _roctx_library_candidates():
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        names = (
            "roctxProfilerPause",
            "roctxProfilerResume",
            "roctxRangePushA",
            "roctxRangePop",
        )
        if all(hasattr(library, name) for name in names):
            library.roctxProfilerPause.argtypes = [ctypes.c_uint64]
            library.roctxProfilerPause.restype = ctypes.c_int
            library.roctxProfilerResume.argtypes = [ctypes.c_uint64]
            library.roctxProfilerResume.restype = ctypes.c_int
            library.roctxRangePushA.argtypes = [ctypes.c_char_p]
            library.roctxRangePushA.restype = ctypes.c_int
            library.roctxRangePop.argtypes = []
            library.roctxRangePop.restype = ctypes.c_int
            return library, candidate
    raise RuntimeError("rocprofiler-sdk ROCTx library is unavailable")


def _check_roctx_status(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"ROCTx {operation} failed with status {status}")


def _initialize_roctx_profile(roctx) -> None:
    _check_roctx_status(roctx.roctxProfilerPause(0), "initial pause")


def _finish_roctx_profile(roctx) -> None:
    failures = []
    try:
        level = roctx.roctxRangePop()
        if level < 0:
            failures.append(f"roctxRangePop failed with level {level}")
    except (OSError, ctypes.ArgumentError, TypeError, ValueError) as error:
        failures.append(f"roctxRangePop failed: {error}")
    try:
        status = roctx.roctxProfilerPause(0)
        if status != 0:
            failures.append(f"ROCTx final pause failed with status {status}")
    except (OSError, ctypes.ArgumentError, TypeError, ValueError) as error:
        failures.append(f"ROCTx final pause failed: {error}")
    if failures:
        raise RuntimeError("; ".join(failures))


@contextlib.contextmanager
def _roctx_profile_region(roctx, name: str):
    _check_roctx_status(roctx.roctxProfilerResume(0), "resume")
    try:
        level = roctx.roctxRangePushA(name.encode("ascii"))
    except (OSError, ctypes.ArgumentError, TypeError, ValueError) as push_error:
        try:
            _check_roctx_status(roctx.roctxProfilerPause(0), "final pause")
        except (
            OSError,
            ctypes.ArgumentError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as cleanup_error:
            raise RuntimeError(
                f"roctxRangePushA failed: {push_error}; {cleanup_error}"
            ) from push_error
        raise RuntimeError(f"roctxRangePushA failed: {push_error}") from push_error
    if level < 0:
        try:
            _check_roctx_status(roctx.roctxProfilerPause(0), "final pause")
        except RuntimeError as cleanup_error:
            raise RuntimeError(
                f"roctxRangePushA failed with level {level}; {cleanup_error}"
            ) from cleanup_error
        raise RuntimeError(f"roctxRangePushA failed with level {level}")
    try:
        yield
    finally:
        _finish_roctx_profile(roctx)


def _time_samples(evaluate, count, *, runtime, runtime_kind, roctx=None):
    samples = []
    region = (
        _roctx_profile_region(roctx, "symmetrix-maceles-les-stage-samples")
        if roctx is not None
        else contextlib.nullcontext()
    )
    with region:
        for _ in range(count):
            _synchronize(runtime, runtime_kind)
            started = time.perf_counter_ns()
            evaluate()
            _synchronize(runtime, runtime_kind)
            samples.append(time.perf_counter_ns() - started)
    return samples


def _maceles_off_bulk_water_structure():
    """Build a deterministic box at the paper's reported water conditions.

    Kim et al., JCTC 2025, DOI 10.1021/acs.jctc.5c01400, simulated 64-molecule
    bulk-water cells at 0.997 g/cm3 and benchmarked timing over varying bulk-
    water sizes. Deterministic orientations reproduce the reported base size
    and density without carrying a supplementary trajectory snapshot.
    """

    import numpy as np
    from ase import Atoms

    water_mass_amu = float(Atoms("H2O").get_masses().sum())
    volume_angstrom3 = (
        WATER_MOLECULES
        * water_mass_amu
        * AMU_PER_ANGSTROM3_TO_G_PER_CM3
        / WATER_DENSITY_G_CM3
    )
    box_length = volume_angstrom3 ** (1.0 / 3.0)
    spacing = box_length / 4.0

    bond_length = 0.9572
    half_angle = np.deg2rad(104.52 / 2.0)
    local_hydrogens = np.array(
        [
            [bond_length * np.cos(half_angle), bond_length * np.sin(half_angle), 0.0],
            [bond_length * np.cos(half_angle), -bond_length * np.sin(half_angle), 0.0],
        ]
    )
    positions = []
    symbols = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for index in range(WATER_MOLECULES):
        ix, remainder = divmod(index, 16)
        iy, iz = divmod(remainder, 4)
        oxygen = (np.array([ix, iy, iz], dtype=float) + 0.5) * spacing

        alpha = 2.0 * np.pi * ((index + 0.5) / golden_ratio % 1.0)
        beta = np.arccos(1.0 - 2.0 * (index + 0.5) / WATER_MOLECULES)
        gamma = 2.0 * np.pi * ((index + 0.5) * np.sqrt(2.0) % 1.0)
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        cos_beta, sin_beta = np.cos(beta), np.sin(beta)
        cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)
        rotate_z_alpha = np.array(
            [
                [cos_alpha, -sin_alpha, 0.0],
                [sin_alpha, cos_alpha, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotate_y_beta = np.array(
            [
                [cos_beta, 0.0, sin_beta],
                [0.0, 1.0, 0.0],
                [-sin_beta, 0.0, cos_beta],
            ]
        )
        rotate_z_gamma = np.array(
            [
                [cos_gamma, -sin_gamma, 0.0],
                [sin_gamma, cos_gamma, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = rotate_z_alpha @ rotate_y_beta @ rotate_z_gamma
        positions.extend([oxygen, *(oxygen + local_hydrogens @ rotation.T)])
        symbols.extend(("O", "H", "H"))

    return Atoms(
        symbols,
        positions=positions,
        cell=np.eye(3) * box_length,
        pbc=True,
    )


def _structure(replicate, path=None):
    import numpy as np
    from ase.io import read

    atoms = (
        _maceles_off_bulk_water_structure() if path is None else read(path, index=-1)
    )
    if len(atoms) == 0:
        raise ValueError("benchmark structure must contain at least one atom")
    if not np.all(np.isfinite(atoms.get_positions())):
        raise ValueError("benchmark structure positions must be finite")
    if not np.all(atoms.pbc):
        raise ValueError(
            "LES particle-mesh benchmark requires full three-dimensional PBC"
        )
    cell = np.asarray(atoms.cell)
    if not np.all(np.isfinite(cell)) or not np.isfinite(atoms.get_volume()):
        raise ValueError("benchmark structure cell must be finite")
    if atoms.get_volume() <= 0.0:
        raise ValueError("benchmark structure cell must have positive volume")
    atoms.wrap()
    return atoms.repeat(tuple(replicate))


def _selected_periodic_algorithm(calculator) -> str:
    diagnostics = dict(calculator.evaluator._les_particle_mesh_diagnostics_for_testing)
    algorithm = diagnostics.get("periodic_algorithm")
    if algorithm not in {"explicit_sum", "particle_mesh"}:
        raise RuntimeError(
            f"MACELES evaluator reported invalid periodic algorithm {algorithm!r}"
        )
    return algorithm


def run(args: argparse.Namespace) -> dict[str, object]:
    affinity = _pin_one_cpu()
    import numpy as np

    from symmetrix import Symmetrix, backend_loader, selected_backend

    model_path = args.model.resolve()
    model = json.loads(model_path.read_text())
    if model.get("model_type") != "MACELES":
        raise ValueError(f"{model_path} is not a MACELES model")
    les = model["les"]
    ewald = les["ewald"]
    sigma, dl = float(ewald["sigma"]), float(ewald["dl"])
    atoms = _structure(args.replicate, args.structure)
    calculator = Symmetrix(
        model_path,
        dtype=args.precision,
        streamed_edges=args.streamed_edges,
        execution_profile="speed",
        neighbor_skin=args.skin,
    )
    if calculator.model_type != "MACELES":
        raise RuntimeError("model did not construct a MACELES evaluator")
    if args.algorithm == "particle_mesh":
        calculator.evaluator._configure_les_particle_mesh_for_testing(
            args.mesh, args.mesh, args.mesh, args.order
        )
    properties = ["energy", "energies", "forces", "stress"]
    graph_count = 0
    original_graph_builder = calculator._mace_inputs

    def counted_graph_builder(*values, **keywords):
        nonlocal graph_count
        graph_count += 1
        return original_graph_builder(*values, **keywords)

    calculator._mace_inputs = counted_graph_builder
    inputs = calculator._mace_inputs(atoms)
    nodes, _, _, sources, _, _, _, _ = inputs
    calculator._apply_kernel_launch_tuning(nodes, len(sources), properties)

    def evaluate():
        calculator._set_current_les_geometry(atoms)
        generation = calculator._compute_mace(inputs, atoms=atoms)
        calculator.results = calculator._collect_mace_results(
            atoms, inputs, properties, generation
        )
        calculator._publish_les_results(atoms)

    for _ in range(args.warmups):
        evaluate()
    selected_algorithm = _selected_periodic_algorithm(calculator)
    if selected_algorithm != args.algorithm:
        raise RuntimeError(
            "MACELES benchmark selected "
            f"{selected_algorithm!r}, requested {args.algorithm!r}"
        )
    descriptor = selected_backend()
    native = backend_loader._native_module
    if descriptor is None or native is None:
        raise RuntimeError("benchmark did not load a native backend")
    execution_space = native._kokkos_default_execution_space()
    runtime, runtime_kind, runtime_path = _load_runtime(execution_space)
    roctx = None
    roctx_path = None
    if args.profile_region:
        if runtime_kind != "hip":
            raise RuntimeError("--profile-region currently requires a HIP backend")
        roctx, roctx_path = _load_roctx()
        _initialize_roctx_profile(roctx)
    before_timing = graph_count
    profile_completed = False
    samples = _time_samples(
        evaluate,
        args.samples,
        runtime=runtime,
        runtime_kind=runtime_kind,
        roctx=roctx,
    )
    profile_completed = roctx is not None
    graph_during_timing = graph_count - before_timing
    median_ns = statistics.median(samples)
    extension = Path(native.__file__).resolve()
    build = dict(native._backend_build_info())
    result = calculator.results
    if any(not np.all(np.isfinite(result[name])) for name in properties):
        raise RuntimeError("benchmark produced a non-finite result")
    record = {
        "schema": SCHEMA,
        "algorithm": args.algorithm,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "structure_sha256": _canonical_sha256(
            {
                "atomic_numbers": atoms.get_atomic_numbers().tolist(),
                "positions": atoms.get_positions().tolist(),
                "cell": np.asarray(atoms.cell).tolist(),
                "pbc": np.asarray(atoms.pbc).tolist(),
            }
        ),
        "model_type": calculator.model_type,
        "mace_les_parameters_sha256": calculator.mace_les_parameters_sha256,
        "mace_les_evaluator_fingerprint": (calculator.mace_les_evaluator_fingerprint),
        "mace_architecture": {
            "mace_num_interactions": les["mace_num_interactions"],
            "mace_num_channels": les["mace_num_channels"],
            "mace_contract_fingerprints": dict(les["mace_contract_fingerprints"]),
            "atomic_numbers": list(model.get("atomic_numbers", [])),
            "cutoff": float(calculator.cutoff),
        },
        "source_provenance": {
            "source_commit": build.get("source_commit"),
            "source_dirty": build.get("source_dirty"),
            "source_content_sha256": build.get("source_content_sha256"),
            "native_source_content_sha256": build.get("native_source_content_sha256"),
        },
        "native_extension": str(extension),
        "native_extension_sha256": _sha256(extension),
        "backend": descriptor["backend"],
        "kokkos_execution_space": execution_space,
        "precision": args.precision,
        "atoms": len(atoms),
        "cell": np.asarray(atoms.cell).tolist(),
        "sigma": sigma,
        "dl": dl,
        "mesh": [args.mesh] * 3 if args.algorithm == "particle_mesh" else None,
        "spline_order": args.order if args.algorithm == "particle_mesh" else None,
        "directed_edges": len(sources),
        "streamed_edges": args.streamed_edges,
        "skin": args.skin,
        "samples_ns": samples,
        "sample_count": len(samples),
        "median_ns": median_ns,
        "us_per_atom": median_ns / 1000.0 / len(atoms),
        "warmups": args.warmups,
        "host_graph_constructions_during_timing": graph_during_timing,
        "fallback_evaluation_count": int(
            calculator.evaluator.factorized_fallback_evaluation_count
        ),
        "jit_status": calculator.jit_status,
        "jit_artifact_path": calculator.jit_artifact_path,
        "cpu_affinity": affinity,
        "thread_counts": {
            name: int(os.environ.get(name, "1")) for name in THREAD_VARIABLES
        },
        "synchronization": {
            "backend": runtime_kind,
            "runtime_library": runtime_path,
        },
        "profile_range": {
            "enabled": profile_completed,
            "api": "roctx" if profile_completed else None,
            "name": (
                "symmetrix-maceles-les-stage-samples" if profile_completed else None
            ),
            "library": roctx_path if profile_completed else None,
        },
        "interpreter_path": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "energy": float(result["energy"]),
        "les_energy": float(result["les_energy"]),
        "mace_les_energy": float(result["mace_les_energy"]),
        "timed_operations": [
            "LES-stage geometry handoff",
            "native MACE forward and reverse",
            "energy, force, and stress reduction",
            "result materialization",
        ],
        "excluded_operations": [
            "model construction",
            "host graph construction",
            "JIT compilation",
            "FFT plan construction",
            "warmup",
        ],
    }
    return validate_record(record)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="particle_mesh")
    parser.add_argument("--precision", choices=PRECISIONS, default="float32")
    parser.add_argument(
        "--streamed-edges", choices=STREAMED_EDGE_MODES, default="direct"
    )
    parser.add_argument(
        "--structure",
        type=Path,
        help=(
            "fully periodic ASE-readable structure; defaults to a deterministic "
            "64-water box matching the MACELES-OFF paper's 0.997 g/cm3 conditions"
        ),
    )
    parser.add_argument("--replicate", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--skin", type=float, default=0.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--profile-region", action="store_true")
    parser.add_argument(
        "--compare", nargs=2, type=Path, metavar=("EXPLICIT", "PARTICLE_MESH")
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.compare:
        records = [json.loads(path.read_text()) for path in args.compare]
        print(json.dumps(compare_records(records), indent=2, sort_keys=True))
        return 0
    if args.model is None:
        parser.error("model is required unless --compare is used")
    if any(value < 1 for value in args.replicate):
        parser.error("replication counts must be positive")
    if args.mesh < 2 or not 2 <= args.order <= 6:
        parser.error("mesh and spline order are invalid")
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups non-negative")
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
