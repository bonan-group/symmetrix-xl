"""Check NequIP-OAM scaling under OpenMP thread counts.

The OAM Kokkos path dispatches owner work inside ``Kokkos::parallel_for`` over
nodes, so its speedup must be *measured* per thread count, never requested.
Kokkos binds one team size per process, therefore every thread count runs in a
fresh subprocess whose affinity is restricted to disjoint physical cores
*before* Kokkos initializes.

Each subprocess reports:

* participation -- ``openmp_runtime_diagnostics(strict=True)`` observes the
  actually created worker team, its per-worker CPU affinity, and the loaded
  OpenBLAS/OpenMP runtime;
* parity -- node energies and per-edge adjoint forces against the single-thread
  subprocess (reassociation tolerance) and against a repeated call in the same
  process (bit-stable);
* steady-state wall clock -- warm-up calls first, then repeated forward+reverse
  evaluations.  The speedup gate compares the *fastest* observed call of each
  worker: a shared machine only ever makes a call slower, so the minimum is the
  uncontended estimate of what the configuration can reach, while the medians
  are reported next to it for transparency.

Worker placement matters and is verified, not assumed.  The subprocess is
restricted with ``taskset`` to the first ``N`` disjoint physical cores (one
representative CPU per core, ascending) and libgomp binds the team to those
places.  Two host properties are load bearing:

* the reserved CPU set must be the dense ascending prefix of the physical
  cores.  The checked-in 320-atom SrTiO3 case is small enough for the ordinary
  test environment while still exercising the full periodic OAM graph;
* the 8-worker team needs the warm-up calls.  The first parallel evaluations of
  a process pay thread-pool and page-placement costs, and timing from call one
  understates 8-worker throughput by ~30%.

The harness skips cleanly when the host cannot supply at least two disjoint
physical cores, cannot pin them, or resolves a host BLAS that is not an
OpenMP-enabled OpenBLAS build.  The last one is part of the test
environment rather than of the evaluator: a pthread OpenBLAS forces host GEMMs
to one thread, and the harness's own strict diagnostics report that as a
warning rather than the ``status: ok`` this test requires, so the
thread-count gates below cannot be measured on such a host.  See
``docs/developer/build_and_test.md`` for the build that selects the
OpenMP flavor.

This module is also its own subprocess worker: ``python
test_nequip_oam_openmp_scaling.py --worker ...`` runs one thread count.  Keep
the ``__main__`` guard intact.

Selection and mandatory-node policy
-----------------------------------
The module is marked ``performance``: it is a wall-clock gate whose constants
are host-specific, so a constrained machine can deselect it with ``-m "not
performance"`` without touching the correctness tests.  It is deliberately *not*
listed in a ``*_required_tests.json`` manifest.  The mandatory-node mechanism
makes a skip fail the run, which is right for an accelerator capability check
but wrong here: the gate skips when the host cannot supply two disjoint
pinnable physical cores, and its speedup constants are only valid where the
reserved CPUs are the dense ascending prefix of the physical cores (see above).
Registering it would turn those host properties into red CI for a
non-defect.  The performance test therefore runs it explicitly, with the environment
and CPU set it documents, rather than through the mandatory manifest.
"""

from __future__ import annotations

# The explicitly pinned native extension must load before NumPy or Symmetrix.
# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from explicit_extension import bootstrap_explicit_extension

bootstrap_explicit_extension()

import numpy as np
import pytest

MODEL_ENV = "SYMMETRIX_OAM_S_MODEL"
# The production OAM-S export is deliberately external to the repository.  It
# is the maintained JSON export linked to the canonical 0.1 archive; the old
# ``a5cca...`` cache is a rejected legacy candidate.  The path is configurable,
# but its content is pinned so a different or reduced checkpoint cannot turn
# this into a small-model test.
PRODUCTION_MODEL_SHA256 = (
    "c11d730c1dbe0a3b93093fefe7132defb0a189706d644a4b3577cb1193e5d662"
)
OAM_DTYPE = os.environ.get("SYMMETRIX_OAM_DTYPE", "float32")
if OAM_DTYPE not in {"float32", "float64"}:
    raise ValueError("SYMMETRIX_OAM_DTYPE must be float32 or float64")
STRUCTURE_FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_srtio3_4x4x4.json"

#: The test record that documents the environment this harness
#: requires: an OpenMP-enabled OpenBLAS selected without ``LD_LIBRARY_PATH``.
OPENMP_DOC = "docs/developer/build_and_test.md"

THREAD_COUNTS = (1, 2, 4, 8)
#: Minimum steady-state speedup over the one-thread subprocess.
SPEEDUP_GATES = {4: 1.4, 8: 2.0}
#: Warm-up calls before timing.  The first parallel evaluations of a process
#: pay thread-pool and page-placement costs, so the 8-thread worker drifts for
#: several calls; the gate must not measure that ramp.
WARMUP_CALLS = 6
TIMED_CALLS = 15
WORKER_TIMEOUT_SECONDS = 1800.0

#: The checked-in 4x4x4 SrTiO3 structure has 320 atoms and 9856 directed
#: periodic edges inside the 4.5 A OAM cutoff. Neighbor entries are
#: receiver-major, matching the evaluator's contract.
STRUCTURE_ATOMS = 320
STRUCTURE_EDGES = 9856

#: Reassociation tolerance between thread counts.  The receiver-owned ABI v2
#: owners accumulate per receiver/source without atomics, so this contract
#: reproduces bit-identical energies and adjoint forces at every thread count
#: (measured: max|difference| == 0).  The bound stays at the float64
#: reassociation scale because the host contractions around the owners are
#: dispatched to a multithreaded OpenBLAS whose reduction order is allowed to
#: depend on the worker count; a real thread-count-dependent defect is orders of
#: magnitude larger than 1e-12 relative.
PARITY_RTOL = 1.0e-12
#: Absolute floor for the same comparison on near-cancelling components.
PARITY_ATOL = 1.0e-12


def _native():
    try:
        from symmetrix import symmetrix as native
    except Exception as error:
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("OAM Kokkos evaluator is unavailable in this extension")
    return native


def _parse_cpu_set(text: str) -> list[int]:
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


def _physical_core_cpus() -> list[int] | None:
    """One CPU per physical core, ascending, restricted to the allowed set.

    Returns ``None`` when the host cannot report the sibling topology, in which
    case the caller falls back to the even-numbered allowed CPUs.
    """

    allowed = sorted(os.sched_getaffinity(0))
    groups: dict[int, list[int]] = {}
    for cpu in allowed:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        try:
            siblings = _parse_cpu_set(path.read_text().strip())
        except (OSError, ValueError):
            return None
        if not siblings:
            return None
        groups.setdefault(siblings[0], []).append(cpu)
    return [min(cpus) for _, cpus in sorted(groups.items())]


def _available_physical_cpus() -> list[int]:
    """One representative CPU per disjoint physical core this process may use."""

    allowed = sorted(os.sched_getaffinity(0))
    physical = _physical_core_cpus()
    if physical is None:
        # No topology report: assume sibling pairs interleave as 0..15 then
        # 16..31 and keep only the even half.
        physical = [cpu for cpu in allowed if cpu % 2 == 0]
    return physical


def _production_model() -> tuple[Path, str]:
    """Return the externally supplied, content-pinned production OAM-S model."""

    value = os.environ.get(MODEL_ENV)
    if not value:
        raise RuntimeError(
            f"{MODEL_ENV} must point to the approved external OAM-S checkpoint"
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"external OAM-S checkpoint does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PRODUCTION_MODEL_SHA256:
        raise AssertionError(
            "external OAM-S checkpoint has unexpected SHA-256: "
            f"{digest}; expected {PRODUCTION_MODEL_SHA256}"
        )
    return path, digest


def _scaling_graph(model_path: Path):
    """Build the receiver-major periodic graph for the 320-atom SrTiO3 case."""

    from ase import Atoms
    from ase.neighborlist import neighbor_list

    structure = json.loads(STRUCTURE_FIXTURE.read_text(encoding="utf-8"))
    atomic_numbers = np.asarray(structure["atomic_numbers"], dtype=np.int32)
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=np.asarray(structure["positions"], dtype=np.float64),
        cell=np.asarray(structure["cell"], dtype=np.float64),
        pbc=structure["pbc"],
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    type_map = {
        int(key): int(value)
        for key, value in model["atomic_number_to_model_type"].items()
    }
    node_types = np.asarray(
        [type_map[int(number)] for number in atomic_numbers], dtype=np.int32
    )
    receivers, indices, displacement = neighbor_list(
        "ijD", atoms, float(structure["cutoff_A"])
    )
    order = np.argsort(receivers, kind="stable")
    receivers = np.asarray(receivers[order], dtype=np.int32)
    indices = np.asarray(indices[order], dtype=np.int32)
    displacement = np.asarray(displacement[order], dtype=np.float64)
    num_nodes = len(atomic_numbers)
    num_neigh = np.bincount(receivers, minlength=num_nodes).astype(np.int32)
    assert num_nodes == STRUCTURE_ATOMS
    assert len(indices) == STRUCTURE_EDGES
    return {
        "num_nodes": num_nodes,
        "node_types": node_types,
        "num_neigh": num_neigh,
        "indices": indices,
        "neigh_types": node_types[indices],
        "xyz": displacement.reshape(-1),
        "distances": np.linalg.norm(displacement, axis=1),
        "neighbors_per_node": int(num_neigh.max()),
    }


def _worker_environment(threads: int, jit_cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "KOKKOS_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "BLIS_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "SYMMETRIX_JIT_CACHE": str(jit_cache),
            "SYMMETRIX_OPENMP_RUNTIME_CHECK": "strict",
        }
    )
    return environment


def _worker_command(arguments: list[str]) -> list[str]:
    """Command that runs this module's child entry point with ``arguments``."""

    interpreter = [sys.executable]
    if sys.flags.no_site:
        interpreter.append("-S")
    return [*interpreter, str(Path(__file__).resolve()), *arguments]


def _probe_host_blas() -> dict:
    """Report the host BLAS the workers will select, from a throwaway process.

    The probe asks the pinned extension when a performance run pins one, so it
    goes through this module's own child entry point instead of a bare
    interpreter, and it runs in a child because ``_openmp_runtime_info`` starts
    an OpenMP worker team that must not exist in the pytest process.
    """

    completed = subprocess.run(
        _worker_command(["--probe-host-blas"]),
        check=False,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"the host BLAS probe failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    reported = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(reported[-1]) if reported else {}


def _run_worker(threads: int, cpus: list[int], jit_cache: Path, output: Path) -> dict:
    command = ["taskset", "-c", ",".join(str(cpu) for cpu in cpus)]
    command.extend(
        _worker_command(
            [
                "--worker",
                "--worker-threads",
                str(threads),
                "--worker-cpus",
                ",".join(str(cpu) for cpu in cpus),
                "--worker-output",
                str(output),
            ]
        )
    )
    completed = subprocess.run(
        command,
        check=False,
        env=_worker_environment(threads, jit_cache),
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{threads}-thread OAM OpenMP worker failed "
            f"({completed.returncode})\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(output.read_text())


def _probe_host_blas_main() -> int:
    from symmetrix import symmetrix as native

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    print(json.dumps(native._openmp_runtime_info().get("host_blas") or {}))
    return 0


def _worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-threads", type=int, required=True)
    parser.add_argument("--worker-cpus", required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    options = parser.parse_args(argv)

    threads = int(options.worker_threads)
    cpus = _parse_cpu_set(options.worker_cpus)

    from symmetrix import symmetrix as native
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact
    from symmetrix.runtime_diagnostics import openmp_runtime_diagnostics

    if not native._kokkos_is_initialized():
        native._init_kokkos()

    model_path, model_sha256 = _production_model()
    graph = _scaling_graph(model_path)
    artifact = prepare_nequip_oam_receiver_host_artifact(
        model_path, OAM_DTYPE, native_module=native
    )
    evaluator_class = (
        native.NequIPOAMKokkosFloat
        if OAM_DTYPE == "float32"
        else native.NequIPOAMKokkos
    )
    evaluator = evaluator_class(
        str(model_path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    evaluator.set_streamed_edges("direct")

    num_nodes = graph["num_nodes"]
    node_types = graph["node_types"]
    num_neigh = graph["num_neigh"]
    indices = graph["indices"]
    neigh_types = graph["neigh_types"]
    xyz = graph["xyz"]
    distances = graph["distances"]

    def evaluate() -> None:
        evaluator.compute_node_energies_forces(
            num_nodes, node_types, num_neigh, indices, neigh_types, xyz, distances
        )

    evaluate()
    repeat_energies = np.array(evaluator.node_energies, dtype=np.float64)
    repeat_forces = np.array(evaluator.node_forces, dtype=np.float64)
    evaluate()
    energies = np.array(evaluator.node_energies, dtype=np.float64)
    forces = np.array(evaluator.node_forces, dtype=np.float64)

    for _ in range(WARMUP_CALLS):
        evaluate()
    samples = []
    for _ in range(TIMED_CALLS):
        start = time.perf_counter()
        evaluate()
        samples.append(time.perf_counter() - start)
    median_seconds = statistics.median(samples)
    minimum_seconds = min(samples)

    runtime = openmp_runtime_diagnostics(
        native_module=native,
        strict=True,
        expected_threads=threads,
        allocated_cpus=cpus,
    )
    if not runtime["ok"] or runtime["status"] != "ok":
        raise AssertionError(
            "OpenMP runtime diagnostics failed inside the "
            f"{threads}-thread worker:\n"
            + json.dumps(runtime, indent=2, sort_keys=True)
        )
    sentinel = runtime["runtime"]["worker_sentinel"]
    if int(sentinel["actual_team_size"]) != threads:
        raise AssertionError(
            f"worker sentinel created {sentinel['actual_team_size']} OpenMP "
            f"workers; expected {threads}"
        )
    for name in ("worker_team_size", "allocated_cpu_coverage"):
        check = next((item for item in runtime["checks"] if item["name"] == name), None)
        if check is None or check["status"] != "ok":
            raise AssertionError(
                f"OpenMP diagnostics check {name!r} did not pass:\n"
                + json.dumps(runtime, indent=2, sort_keys=True)
            )

    host_blas = runtime["runtime"]["host_blas"]
    report = {
        "threads": threads,
        "dtype": OAM_DTYPE,
        "model_sha256": model_sha256,
        "cpus": cpus,
        "graph": {
            "nodes": int(num_nodes),
            "edges": int(indices.size),
            "neighbors_per_node": int(graph["neighbors_per_node"]),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "KOKKOS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "SYMMETRIX_JIT_CACHE",
                "SYMMETRIX_OPENMP_RUNTIME_CHECK",
                "LD_LIBRARY_PATH",
            )
        },
        "timing": {
            "samples_seconds": samples,
            "median_seconds": median_seconds,
            "minimum_seconds": minimum_seconds,
            "us_per_edge_median": median_seconds * 1.0e6 / float(indices.size),
            "us_per_edge_minimum": minimum_seconds * 1.0e6 / float(indices.size),
        },
        "energies": energies.tolist(),
        "forces": forces.tolist(),
        "repeat_energies": repeat_energies.tolist(),
        "repeat_forces": repeat_forces.tolist(),
        "diagnostic": evaluator.last_execution_diagnostic,
        "artifact": {
            "id": artifact.artifact_id,
            "cache_hit": bool(artifact.cache_hit),
        },
        "runtime": {
            "status": runtime["status"],
            "kokkos_execution_space": runtime["runtime"]["kokkos_execution_space"],
            "kokkos_concurrency": runtime["runtime"]["kokkos_concurrency"],
            "omp_max_threads": runtime["runtime"]["omp_max_threads"],
            "omp_max_active_levels": runtime["runtime"]["omp_max_active_levels"],
            "team_size": int(sentinel["actual_team_size"]),
            "worker_cpus": sorted(int(cpu) for cpu in sentinel["unique_current_cpus"]),
            "checks": runtime["checks"],
            "host_blas": host_blas,
        },
    }
    options.worker_output.write_text(json.dumps(report, sort_keys=True))
    return 0


@pytest.mark.performance
@pytest.mark.oam_runtime
def test_oam_kokkos_openmp_thread_scaling(tmp_path):
    """Parity, participation and speedup of the OAM Kokkos path under threads."""

    if MODEL_ENV not in os.environ:
        pytest.skip(
            "production OAM-S scaling requires SYMMETRIX_OAM_S_MODEL pointing "
            "to the approved full checkpoint"
        )
    native = _native()
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    from symmetrix.runtime_diagnostics import openmp_runtime_diagnostics

    preflight = openmp_runtime_diagnostics(native_module=native, strict=True)
    openblas_build = next(
        (check for check in preflight["checks"] if check["name"] == "openblas_build"),
        None,
    )
    if openblas_build is not None and openblas_build["status"] == "warning":
        pytest.skip(
            "OAM OpenMP scaling requires an OpenMP-enabled, "
            f"runtime-compatible BLAS: {openblas_build['message']}"
        )
    if shutil.which("taskset") is None:
        pytest.skip("taskset is required to pin workers to disjoint physical cores")

    physical = _available_physical_cpus()
    if len(physical) < 2:
        pytest.skip(
            "at least two disjoint physical cores are required to test OAM "
            f"OpenMP thread scaling; this process may use {len(physical)}"
        )
    runnable_counts = [threads for threads in THREAD_COUNTS if threads <= len(physical)]

    host_blas = _probe_host_blas()
    if host_blas.get("openblas_openmp_enabled") is False:
        pytest.skip(
            "the host loader resolves a non-OpenMP OpenBLAS build "
            f"({host_blas.get('library_path')!r}), which forces host GEMMs to a "
            "single thread; the thread-count gates require the OpenMP OpenBLAS "
            f"build documented in {OPENMP_DOC}"
        )

    model_path, model_sha256 = _production_model()
    graph = _scaling_graph(model_path)
    assert graph["num_nodes"] == STRUCTURE_ATOMS
    assert graph["indices"].size == STRUCTURE_EDGES

    jit_cache = tmp_path / "jit-cache"
    # The execution JIT cache is a security boundary: the native loader
    # refuses a cache root that is group- or world-accessible.
    jit_cache.mkdir(mode=0o700)
    jit_cache.chmod(0o700)
    reports: dict[int, dict] = {}
    for threads in runnable_counts:
        reports[threads] = _run_worker(
            threads,
            physical[:threads],
            jit_cache,
            tmp_path / f"worker-{threads}.json",
        )

    reference = reports[1]
    reference_energies = np.asarray(reference["energies"], dtype=np.float64)
    reference_forces = np.asarray(reference["forces"], dtype=np.float64)

    for threads, report in reports.items():
        assert report["model_sha256"] == model_sha256
        assert report["graph"]["nodes"] == graph["num_nodes"]
        assert report["graph"]["edges"] == graph["indices"].size
        assert report["runtime"]["kokkos_execution_space"] == "OpenMP"
        assert report["runtime"]["kokkos_concurrency"] == threads
        assert report["runtime"]["team_size"] == threads
        assert report["runtime"]["worker_cpus"] == report["cpus"]
        assert "operation=direct" in report["diagnostic"]
        assert "streamed_edges=direct" in report["diagnostic"]
        assert "host_plugin_abi=v7" in report["diagnostic"]
        assert report["environment"]["KOKKOS_NUM_THREADS"] == str(threads)
        assert report["environment"]["OMP_NUM_THREADS"] == str(threads)
        assert report["environment"]["OPENBLAS_NUM_THREADS"] == str(threads)
        assert report["environment"]["SYMMETRIX_JIT_CACHE"] == str(jit_cache)
        assert report["environment"]["SYMMETRIX_OPENMP_RUNTIME_CHECK"] == "strict"

        # Participation: the diagnostics observed one bound worker per
        # allocated physical core, disjointly.
        check = next(
            item
            for item in report["runtime"]["checks"]
            if item["name"] == "worker_cpu_participation"
        )
        assert check["status"] == "ok", check["message"]

        # Same-run determinism: one thread count must be bit-stable.
        np.testing.assert_array_equal(
            np.asarray(report["energies"], dtype=np.float64),
            np.asarray(report["repeat_energies"], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            np.asarray(report["forces"], dtype=np.float64),
            np.asarray(report["repeat_forces"], dtype=np.float64),
        )
        if threads == 1:
            continue

        # Cross-thread parity: only the reassociation order may differ.
        np.testing.assert_allclose(
            np.asarray(report["energies"], dtype=np.float64),
            reference_energies,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg=f"node energies diverge at {threads} threads",
        )
        np.testing.assert_allclose(
            np.asarray(report["forces"], dtype=np.float64),
            reference_forces,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg=f"edge adjoint forces diverge at {threads} threads",
        )

    # The gate compares the fastest observed steady-state call of each worker.
    # Shared machines only ever make a call slower, so the minimum is the
    # uncontended estimate of what the configuration can reach; the medians are
    # reported alongside for transparency.
    reference_seconds = reference["timing"]["minimum_seconds"]
    for threads, minimum in SPEEDUP_GATES.items():
        report = reports.get(threads)
        if report is None:
            continue
        seconds = report["timing"]["minimum_seconds"]
        speedup = reference_seconds / seconds
        assert speedup >= minimum, (
            f"OAM Kokkos evaluation reached only {speedup:.2f}x at {threads} "
            f"threads (>= {minimum}x required); "
            f"1 thread: {reference_seconds * 1e3:.1f} ms best / "
            f"{reference['timing']['median_seconds'] * 1e3:.1f} ms median, "
            f"{threads} threads: {seconds * 1e3:.1f} ms best / "
            f"{report['timing']['median_seconds'] * 1e3:.1f} ms median"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--probe-host-blas" in arguments:
        return _probe_host_blas_main()
    if not arguments or "--worker" not in arguments:
        raise SystemExit(
            "test_nequip_oam_openmp_scaling.py is a pytest module; run it through "
            "pytest, or pass --worker for a single performance subprocess"
        )
    arguments.remove("--worker")
    return _worker_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
