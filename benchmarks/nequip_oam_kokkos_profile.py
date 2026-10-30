"""Profile generated NequIP-OAM Kokkos execution on SrTiO3.

The default workload is a 10x10x10 periodic SrTiO3 supercell (5,000 atoms).
This driver records qualification metadata even when the optional native build
or model is unavailable; unavailable phase timers remain ``null`` rather than
being estimated from the aggregate wall time.

Timed region
------------
Two measurement modes are supported and the record always states which one ran
in ``"reuse_prepared_graph"``:

* default (``"reuse_prepared_graph": false``) -- the timed region is one full
  ``NequIPCalculator.calculate`` call, so graph *preparation* is inside it: the
  retained neighbour list is re-uploaded to the device and the receiver-owned
  workspace is rebuilt for every sample.  The neighbour list itself is built
  once, by the cold sample, and reused afterwards.
* ``--reuse-prepared-graph`` (``"reuse_prepared_graph": true``) -- the graph is
  uploaded once outside the timed region and each sample runs only the prepared
  evaluation (``compute_prepared`` + ``reduce_forces``).  This is the
  steady-state number that is comparable with other steady-state records; the
  driver refuses to report it when the prepared drive disagrees with the public
  calculator, so the two regions cannot silently drift apart.

Either mode records a separate ``cold_ms`` next to the steady-state ``min_ms``
and ``median_ms`` so a reader can subtract the one-off cost.

Thread policy
-------------
With ``--threads 1`` (the default) the process is pinned to a single CPU and
``KOKKOS_NUM_THREADS``/``OMP_NUM_THREADS``/``OPENBLAS_NUM_THREADS``/
``MKL_NUM_THREADS``/``BLIS_NUM_THREADS``/``NUMEXPR_NUM_THREADS`` are all ``1``,
before any native import. ``--threads N`` sets only Kokkos and OpenMP to ``N``;
OpenBLAS, MKL, BLIS, and NumExpr remain at one. The process binds to
``--cpu-set`` (or, without it, to the first ``N`` CPUs of the current affinity
mask) before the native import. The record's ``affinity`` block states the
thread counts and CPUs actually used, so a multi-thread row is never mistaken
for a one-thread one.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import tempfile
import time
from typing import Any, Mapping, Sequence

try:
    from benchmarks.structure_workloads import repeat_factors
except ModuleNotFoundError as error:
    if error.name != "benchmarks":
        raise
    from structure_workloads import repeat_factors


PHASES = (
    "generated_tp_forward",
    "generated_tp_reverse",
    "radial_spline",
    "gather",
    "scatter",
    "linear",
    "reduction",
)
TIMING_EXCLUSIONS = (
    "artifact_compilation",
    "model_preparation",
    "graph_construction",
    "neighbor_construction",
    "warmup",
)

THREAD_ENVIRONMENT = (
    "KOKKOS_NUM_THREADS",
    "OMP_NUM_THREADS",
)
BLAS_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parse_cpu_set(text: str) -> list[int]:
    """Expand a ``taskset``-style CPU list such as ``0-3,8``."""

    cpus: set[int] = set()
    for item in text.split(","):
        bounds = item.strip().split("-", maxsplit=1)
        if not bounds[0]:
            raise ValueError(f"invalid CPU set: {text!r}")
        start = int(bounds[0])
        end = start if len(bounds) == 1 else int(bounds[1])
        if start < 0 or end < start:
            raise ValueError(f"invalid CPU set: {text!r}")
        cpus.update(range(start, end + 1))
    return sorted(cpus)


def _configure_process(
    cpu: int | None,
    *,
    threads: int = 1,
    cpus: Sequence[int] | None = None,
) -> list[int]:
    """Apply thread settings and affinity before any native import."""

    if threads < 1:
        raise ValueError("threads must be at least one")
    pinning = hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")
    available = sorted(os.sched_getaffinity(0)) if pinning else []
    if cpus is not None:
        if not cpus:
            raise ValueError("cpu-set must name at least one CPU")
        selected = sorted({int(value) for value in cpus})
        outside = [value for value in selected if pinning and value not in available]
        if outside:
            raise ValueError(
                f"requested CPUs {outside} are outside process affinity {available}"
            )
        affinity = selected
    elif threads > 1:
        if not pinning:
            raise ValueError("--threads > 1 requires process affinity support")
        if len(available) < threads:
            raise ValueError(
                f"--threads {threads} needs {threads} allowed CPUs; this process may use {len(available)}"
            )
        affinity = available[:threads]
    elif pinning:
        selected = available[0] if cpu is None else int(cpu)
        if selected not in available:
            raise ValueError(
                f"requested CPU {selected} is outside process affinity {available}"
            )
        affinity = [selected]
    else:
        affinity = [int(cpu)] if cpu is not None else []
    if pinning:
        os.sched_setaffinity(0, set(affinity))
    for variable in THREAD_ENVIRONMENT:
        os.environ[variable] = str(threads)
    for variable in BLAS_THREAD_ENVIRONMENT:
        os.environ[variable] = "1"
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    return affinity


def _cpu_topology(cpu_ids: Sequence[int]) -> list[dict[str, int]]:
    rows = []
    physical = set()
    for logical_cpu in cpu_ids:
        root = Path(f"/sys/devices/system/cpu/cpu{logical_cpu}/topology")
        try:
            package = int((root / "physical_package_id").read_text().strip())
            core = int((root / "core_id").read_text().strip())
        except (OSError, ValueError) as error:
            raise ValueError(
                f"physical topology is unavailable for CPU {logical_cpu}"
            ) from error
        key = (package, core)
        if key in physical:
            raise ValueError(
                "CPU affinity includes SMT siblings from one physical core"
            )
        physical.add(key)
        rows.append({"logical_cpu": logical_cpu, "package": package, "core": core})
    return rows


def build_srtio3(
    repeat: int | Sequence[int] = 10,
    lattice: float = 3.905,
    *,
    rattle: float = 0.0,
    seed: int = 0,
):
    """Construct a periodic SrTiO3 supercell with cubic or explicit axis repeats."""

    from ase import Atoms

    basis = (
        (38, (0.0, 0.0, 0.0)),
        (22, (0.5, 0.5, 0.5)),
        (8, (0.5, 0.5, 0.0)),
        (8, (0.5, 0.0, 0.5)),
        (8, (0.0, 0.5, 0.5)),
    )
    repeats = repeat_factors(repeat)
    numbers: list[int] = []
    scaled: list[tuple[float, float, float]] = []
    for i in range(repeats[0]):
        for j in range(repeats[1]):
            for k in range(repeats[2]):
                for number, position in basis:
                    numbers.append(number)
                    scaled.append(
                        (
                            (i + position[0]) / repeats[0],
                            (j + position[1]) / repeats[1],
                            (k + position[2]) / repeats[2],
                        )
                    )
    atoms = Atoms(
        numbers=numbers,
        scaled_positions=scaled,
        cell=[factor * lattice for factor in repeats],
        pbc=True,
    )
    if rattle:
        import numpy as np

        generator = np.random.default_rng(seed)
        atoms.positions = atoms.positions + generator.normal(
            0.0, float(rattle), atoms.positions.shape
        )
    return atoms


def _contract_metadata(path: Path) -> tuple[float | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping) or data.get("architecture") != "oam":
            return None, "model contract architecture is not OAM"
        cutoff = float(data["r_cut"])
        if cutoff <= 0.0:
            return None, "model cutoff must be positive"
        return cutoff, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"could not read OAM contract: {type(error).__name__}: {error}"


def _diagnostic_count(diagnostic: Any, key: str) -> int | None:
    match = re.search(rf"(?:^|[,; ]){re.escape(key)}=(\d+)(?:[,; ]|$)", str(diagnostic))
    return int(match.group(1)) if match else None


def _edge_count(atoms, cutoff: float) -> int:
    from ase.neighborlist import neighbor_list

    return int(len(neighbor_list("i", atoms, cutoff)))


def _phase_report() -> dict[str, float | None]:
    return {phase: None for phase in PHASES}


def _invalidate_calculator(calculator: Any) -> None:
    """Force ASE to execute the evaluator instead of returning cached results."""

    results = getattr(calculator, "results", None)
    if isinstance(results, dict):
        results.clear()
    calculator._prepared_key = None


def _timed_properties(calculator: Any) -> tuple[str, ...]:
    properties = ["energy", "forces"]
    if "stress" in getattr(calculator, "implemented_properties", ()):
        properties.append("stress")
    return tuple(properties)


def _prepare_retained_graph(calculator: Any, atoms) -> tuple[Any, Any, Any]:
    """Upload the calculator's retained neighbour graph once.

    Returns ``(generation, xyz, distances)`` for a prepared evaluation loop that
    excludes graph preparation.  ``NequIPCalculator`` keeps its neighbour list
    in ``_graph`` after a ``calculate`` call; re-uploading it here is exactly the
    graph-preparation step the caller wants outside the timed region.  The
    periodic-cell plumbing mirrors ``NequIPCalculator.calculate`` so the
    geometry contract matches the public path.
    """

    import numpy as np

    evaluator = calculator.evaluator
    if not hasattr(evaluator, "prepare_graph"):
        raise RuntimeError(
            "reuse-prepared-graph requires an evaluator with the prepared graph lifecycle"
        )
    graph = getattr(calculator, "_graph", None)
    if not graph:
        raise RuntimeError(
            "reuse-prepared-graph needs one calculator evaluation before timing"
        )
    node_types, num_neigh, sources, neigh_types, _, _, xyz, distances = graph
    xyz = np.asarray(xyz, dtype=float)
    distances = np.asarray(distances, dtype=float)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    generation = evaluator.prepare_graph(
        len(atoms), node_types, num_neigh, sources, neigh_types
    )
    if bool(np.any(atoms.pbc)):
        cell_arg = cell.reshape(-1)
        inverse_arg = np.linalg.inv(cell).reshape(-1)
        pbc = np.asarray(atoms.pbc, dtype=np.int32)
    else:
        cell_arg = inverse_arg = np.empty(0, dtype=float)
        pbc = np.empty(0, dtype=np.int32)
    evaluator.prepare_geometry(
        generation,
        positions.reshape(-1),
        xyz.reshape(-1),
        cell_arg,
        inverse_arg,
        pbc,
    )
    return generation, xyz, distances


def _observed_runtime() -> dict[str, Any]:
    """Report the OpenMP/Kokkos configuration the native runtime actually loaded.

    Returns an empty mapping when the runtime does not expose the query, so a
    Serial-only or non-OpenMP build records ``null`` instead of a guess.
    """

    try:
        from symmetrix import symmetrix as native
    except ImportError:
        return {}
    query = getattr(native, "_openmp_runtime_info", None)
    if not callable(query):
        return {}
    try:
        info = query()
    except (OSError, RuntimeError, ValueError):
        return {}
    if not isinstance(info, Mapping):
        return {}
    return {
        "kokkos_execution_space": info.get("kokkos_execution_space"),
        "kokkos_concurrency": info.get("kokkos_concurrency"),
        "omp_max_threads": info.get("omp_max_threads"),
    }


def _timing_record(samples_ms: list[float], cold_ms: float, atoms) -> dict[str, Any]:
    median_ms = statistics.median(samples_ms)
    minimum_ms = min(samples_ms)
    return {
        "samples_ms": samples_ms,
        "cold_ms": cold_ms,
        "min_ms": minimum_ms,
        "median_ms": median_ms,
        "min_us_per_atom": minimum_ms * 1000.0 / len(atoms),
        "median_us_per_atom": median_ms * 1000.0 / len(atoms),
    }


def profile(
    model_json: Path,
    *,
    precision: str = "float32",
    repeat: int = 10,
    lattice: float = 3.905,
    skin: float = 0.5,
    warmups: int = 3,
    samples: int = 3,
    cpu: int | None = None,
    threads: int = 1,
    cpu_set: Sequence[int] | None = None,
    rattle_sigma: float = 0.0,
    seed: int = 0,
    reuse_prepared_graph: bool = False,
    cache_dir: Path | None = None,
    baseline: Path | None = None,
    dry_run: bool = False,
    routing: str = "direct",
    execution_profile: str = "speed",
) -> dict[str, Any]:
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be float32 or float64")
    if routing not in {"direct", "reference"}:
        raise ValueError("routing must be direct or reference")
    if execution_profile not in {"speed", "capacity"}:
        raise ValueError("execution_profile must be speed or capacity")
    if repeat <= 0 or warmups < 0 or samples < 3 or skin < 0.0:
        raise ValueError(
            "repeat must be positive, warmups non-negative, samples at least three, and skin non-negative"
        )
    if threads < 1:
        raise ValueError("threads must be at least one")
    if rattle_sigma < 0.0:
        raise ValueError("rattle sigma must be non-negative")
    cutoff, contract_error = _contract_metadata(model_json)
    workload = {
        "material": "SrTiO3",
        "repeat": [repeat, repeat, repeat],
        "lattice_A": lattice,
        "atom_count": 5 * repeat**3,
        "neighbor_skin_A": skin,
        "cutoff_A": cutoff,
        "effective_cutoff_A": cutoff + skin if cutoff is not None else None,
        "directed_edge_count": None,
        "rattle_sigma": rattle_sigma,
        "seed": seed,
    }
    record: dict[str, Any] = {
        "schema": "symmetrix.nequip.oam-kokkos-profile",
        "version": 1,
        "status": "dry-run" if dry_run else "pending",
        "model_json": str(model_json),
        "precision": precision,
        "routing": routing,
        "execution_profile": execution_profile,
        "backend": "cpu-kokkos-openmp",
        "workload": workload,
        "atom_count": 5 * repeat**3,
        "cutoff_A": cutoff,
        "neighbor_skin_A": skin,
        "effective_cutoff_A": cutoff + skin if cutoff is not None else None,
        "directed_edge_count": None,
        "rattle_sigma": rattle_sigma,
        "seed": seed,
        "kokkos_threads": threads,
        "blas_threads": 1,
        "observed_runtime": None,
        "reuse_prepared_graph": bool(reuse_prepared_graph),
        "affinity": {
            "cpus": [],
            "topology": [],
            "kokkos_threads": threads,
            "blas_threads": 1,
        },
        "warmups": warmups,
        "samples": samples,
        "phase_times_ms": _phase_report(),
        "phase_timing_available": False,
        "phase_timing_note": "The current OAM evaluator exposes aggregate diagnostics only; kernel phase timers are unavailable.",
        "generated_qualification": {
            "status": "unavailable",
            "required": routing == "direct",
            "plugin_abi": None,
            "artifact_id": None,
            "codegen_generation": None,
            "compiler_identity": None,
            "scalar_size": 4 if precision == "float32" else 8,
            "generic_dispatch_count": None,
            "generic_region_count": None,
            "edge_generic_dispatches": None,
            "edge_generic_regions": None,
        },
        "timing": None,
        "qualification_eligible": bool(reuse_prepared_graph),
        "timing_contract": {
            "timed_region": (
                "compute_prepared+reduce_forces+reduce_stress"
                if reuse_prepared_graph
                else "calculator.calculate(graph-inclusive)"
            ),
            "excluded": (
                list(TIMING_EXCLUSIONS)
                if reuse_prepared_graph
                else [
                    item
                    for item in TIMING_EXCLUSIONS
                    if item not in {"graph_construction", "neighbor_construction"}
                ]
            ),
        },
        "baseline": {
            "status": "unavailable",
            "reason": "no committed non-generated baseline record was supplied",
            "median_us_per_atom": None,
        },
    }
    if contract_error:
        record.update(status="skipped", skip_reason=contract_error)
        return record
    if dry_run:
        return record

    try:
        affinity = _configure_process(cpu, threads=threads, cpus=cpu_set)
        record["affinity"]["cpus"] = affinity
        record["affinity"]["topology"] = _cpu_topology(affinity)
        atoms = build_srtio3(
            repeat=repeat, lattice=lattice, rattle=rattle_sigma, seed=seed
        )
        edge_count = _edge_count(atoms, float(cutoff) + skin)
        record["workload"]["directed_edge_count"] = edge_count
        record["directed_edge_count"] = edge_count
        cache = cache_dir or Path(tempfile.mkdtemp(prefix="symmetrix-oam-profile-"))
        # The JIT cache is a trust boundary: the native loader refuses a
        # group- or world-accessible root, so create a missing one privately.
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ["SYMMETRIX_JIT_CACHE"] = str(cache)
        from symmetrix import NequIPCalculator

        generated = record["generated_qualification"]
        calculator = NequIPCalculator(
            str(model_json),
            dtype=precision,
            neighbor_skin=skin,
            streamed_edges=routing,
            execution_profile=execution_profile,
        )
        timed_properties = _timed_properties(calculator)
        if routing == "direct":
            readiness = calculator.evaluator.family_artifact_readiness
            generated.update(
                {
                    "status": "validated",
                    "artifact_id": readiness.get("artifact_id"),
                    "artifact_origin": calculator.execution_plan.get("artifact_origin"),
                }
            )
        else:
            generated.update(status="not-required")
        atoms.calc = calculator
        # The requested thread counts are environment policy; the runtime query
        # is the evidence that Kokkos actually adopted them.  A multi-thread row
        # that the runtime did not honour is a failed qualification, not a
        # measurement.
        observed = _observed_runtime()
        record["observed_runtime"] = observed
        concurrency = observed.get("kokkos_concurrency") if observed else None
        omp_threads = observed.get("omp_max_threads") if observed else None
        if not isinstance(concurrency, int) or not isinstance(omp_threads, int):
            record.update(
                status="unqualified",
                error="runtime Kokkos/OpenMP concurrency observation is unavailable",
            )
            return record
        if concurrency != threads or omp_threads != threads:
            record.update(
                status="failed",
                error=(
                    f"requested {threads} threads but runtime reports Kokkos "
                    f"{concurrency} and OpenMP {omp_threads}"
                ),
            )
            return record
        # Cold sample: the first evaluation of the process, including the
        # neighbour-list construction the steady-state samples reuse.
        _invalidate_calculator(calculator)
        started = time.perf_counter()
        calculator.calculate(atoms, properties=timed_properties)
        cold_ms = (time.perf_counter() - started) * 1000.0
        samples_ms: list[float] = []
        if reuse_prepared_graph:
            generation, xyz, distances = _prepare_retained_graph(calculator, atoms)
            evaluator = calculator.evaluator
            volume = float(atoms.get_volume())
            cold_energy = float(calculator.results["energy"])
            for _ in range(warmups):
                evaluator.compute_prepared(generation, xyz, distances)
                evaluator.reduce_forces(generation)
                evaluator.reduce_stress(volume, generation)
            import numpy as np

            prepared_energy = float(
                np.asarray(evaluator.node_energies, dtype=float).sum()
            )
            if abs(prepared_energy - cold_energy) > 1.0e-9 * max(1.0, abs(cold_energy)):
                raise RuntimeError(
                    "prepared-graph drive disagrees with NequIPCalculator.calculate "
                    f"({prepared_energy} != {cold_energy}); the steady-state number "
                    "would not describe the public path"
                )
            for _ in range(samples):
                started = time.perf_counter()
                evaluator.compute_prepared(generation, xyz, distances)
                evaluator.reduce_forces(generation)
                evaluator.reduce_stress(volume, generation)
                samples_ms.append((time.perf_counter() - started) * 1000.0)
        else:
            for _ in range(warmups):
                _invalidate_calculator(calculator)
                calculator.calculate(atoms, properties=timed_properties)
            for _ in range(samples):
                _invalidate_calculator(calculator)
                started = time.perf_counter()
                calculator.calculate(atoms, properties=timed_properties)
                samples_ms.append((time.perf_counter() - started) * 1000.0)
        diagnostic = getattr(calculator, "execution_diagnostic", "unknown")
        edge_dispatches = _diagnostic_count(diagnostic, "edge_generic_dispatches")
        edge_regions = _diagnostic_count(diagnostic, "edge_generic_regions")
        generated["edge_generic_dispatches"] = edge_dispatches
        generated["edge_generic_regions"] = edge_regions
        generated["generic_dispatch_count"] = edge_dispatches
        generated["generic_region_count"] = edge_regions
        if routing == "direct" and (edge_dispatches is None or edge_regions is None):
            raise RuntimeError(
                "generated OAM profile did not report both generic tensor-product counters"
            )
        if routing == "direct" and (edge_dispatches != 0 or edge_regions != 0):
            raise RuntimeError(
                "generated OAM profile observed generic tensor-product dispatch"
            )
        record["timing"] = _timing_record(samples_ms, cold_ms, atoms)
        record["timing"]["diagnostic"] = diagnostic
        record["median_us_per_atom"] = record["timing"]["median_us_per_atom"]
        record["min_us_per_atom"] = record["timing"]["min_us_per_atom"]
        if baseline is not None and baseline.is_file():
            baseline_data = json.loads(baseline.read_text(encoding="utf-8"))
            baseline_us = baseline_data.get("timing", {}).get("median_us_per_atom")
            record["baseline"] = {
                "status": "available",
                "path": str(baseline),
                "median_us_per_atom": baseline_us,
                "speedup": (
                    float(baseline_us) / record["timing"]["median_us_per_atom"]
                    if baseline_us and record["timing"]["median_us_per_atom"]
                    else None
                ),
            }
        record["status"] = "passed"
    except RuntimeError as error:
        if "generic tensor-product" in str(error):
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        else:
            record.update(
                status="skipped",
                skip_reason=f"Kokkos/model qualification unavailable: {type(error).__name__}: {error}",
            )
    except (ImportError, OSError, ValueError, TypeError) as error:
        record.update(
            status="skipped",
            skip_reason=f"Kokkos/model qualification unavailable: {type(error).__name__}: {error}",
        )
    return record


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--lattice", type=float, default=3.905)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--cpu", type=int)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Kokkos/OpenMP use N; BLAS/MKL/BLIS/NumExpr remain at one",
    )
    parser.add_argument(
        "--cpu-set",
        help="CPU list to bind the process to (e.g. 0-3,8); defaults to the first --threads allowed CPUs",
    )
    parser.add_argument(
        "--rattle",
        dest="rattle_sigma",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian position displacement in angstroms (see --seed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the --rattle displacement",
    )
    parser.add_argument(
        "--reuse-prepared-graph",
        action="store_true",
        help="upload the graph once and time only the prepared evaluation",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--routing", choices=("direct", "reference"), default="direct")
    parser.add_argument(
        "--execution-profile", choices=("speed", "capacity"), default="speed"
    )
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = make_parser().parse_args() if args is None else args
    cpu_set = _parse_cpu_set(args.cpu_set) if args.cpu_set else None
    record = profile(
        args.model_json,
        precision=args.precision,
        repeat=args.repeat,
        lattice=args.lattice,
        skin=args.skin,
        warmups=args.warmups,
        samples=args.samples,
        cpu=args.cpu,
        threads=args.threads,
        cpu_set=cpu_set,
        rattle_sigma=args.rattle_sigma,
        seed=args.seed,
        reuse_prepared_graph=args.reuse_prepared_graph,
        cache_dir=args.cache_dir,
        baseline=args.baseline,
        dry_run=args.dry_run,
        routing=args.routing,
        execution_profile=args.execution_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": record["status"], "output": str(args.output)}))
    return 0 if record["status"] in {"dry-run", "passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
