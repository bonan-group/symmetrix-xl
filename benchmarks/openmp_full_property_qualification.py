"""Qualify 1/2/4/8-thread generic/direct MACEField OpenMP correctness."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROPERTIES = (
    "energy",
    "forces",
    "stress",
    "polarization",
    "becs",
    "polarizability",
)
BLAS_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _bootstrap_explicit_extension() -> None:
    root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension = os.environ.get("SYMMETRIX_EXTENSION")
    if not root and not extension:
        return
    if not root or not extension:
        raise RuntimeError(
            "SYMMETRIX_SOURCE_ROOT and SYMMETRIX_EXTENSION must be set together"
        )
    package_dir = Path(root).resolve() / "symmetrix/source/symmetrix"
    extension_path = Path(extension).resolve()
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
    native_spec = importlib.util.spec_from_file_location(
        "symmetrix._native_cpu", extension_path
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError(f"cannot load Symmetrix package from {package_dir}")
    if native_spec is None or native_spec.loader is None:
        raise ImportError(f"cannot load Symmetrix extension from {extension_path}")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["symmetrix"] = package
    package_spec.loader.exec_module(package)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix._native_cpu"] = native
    native_spec.loader.exec_module(native)
    # Explicit-extension qualification bypasses packaging discovery. Bind the
    # already initialized CPU module so the compatibility alias cannot execute
    # its initializer a second time.
    from symmetrix import backend_loader

    backend_loader._native_module = native
    backend_loader._selected = backend_loader._cpu_descriptor(package.__version__)
    sys.modules["symmetrix.symmetrix"] = native


def _parse_cpu_set(value: str | None) -> list[int] | None:
    if value is None:
        return None
    cpus = set()
    for item in value.split(","):
        bounds = item.strip().split("-", maxsplit=1)
        if not bounds[0]:
            raise ValueError(f"invalid CPU set: {value!r}")
        start = int(bounds[0])
        end = start if len(bounds) == 1 else int(bounds[1])
        if start < 0 or end < start:
            raise ValueError(f"invalid CPU set: {value!r}")
        cpus.update(range(start, end + 1))
    return sorted(cpus)


def _timing_summary(samples_seconds: list[float], atoms: int) -> dict:
    median_seconds = statistics.median(samples_seconds)
    return {
        "samples_ms": [sample * 1000.0 for sample in samples_seconds],
        "median_ms": median_seconds * 1000.0,
        "us_per_atom": median_seconds * 1.0e6 / atoms,
    }


def _speedup_summary(
    reference_ms: float,
    candidate_ms: float,
    *,
    reference_threads: int,
    candidate_threads: int,
    minimum: float,
) -> dict:
    speedup = reference_ms / candidate_ms
    return {
        "reference_threads": reference_threads,
        "candidate_threads": candidate_threads,
        "speedup": speedup,
        "minimum": minimum,
        "passed": minimum == 0 or speedup >= minimum,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield float(value)


def _shape(value):
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shapes = {_shape(item) for item in value}
    if len(child_shapes) != 1:
        raise ValueError("qualification result contains a ragged array")
    return (len(value), *child_shapes.pop())


def _compare_results(reference, candidate, *, atol: float, rtol: float):
    differences = {}
    passed = True
    for name in PROPERTIES:
        reference_shape = _shape(reference[name])
        candidate_shape = _shape(candidate[name])
        if reference_shape != candidate_shape:
            differences[name] = {
                "passed": False,
                "reason": "shape differs",
                "reference_shape": reference_shape,
                "candidate_shape": candidate_shape,
            }
            passed = False
            continue
        reference_values = list(_flatten(reference[name]))
        candidate_values = list(_flatten(candidate[name]))
        finite = all(map(math.isfinite, reference_values + candidate_values))
        absolute_differences = [
            abs(actual - expected)
            for expected, actual in zip(reference_values, candidate_values)
        ]
        element_tolerances = [
            atol + rtol * abs(expected) for expected in reference_values
        ]
        max_abs = max(absolute_differences, default=0.0)
        scale = max((abs(value) for value in reference_values), default=0.0)
        max_tolerance = max(element_tolerances, default=atol)
        property_passed = finite and all(
            difference <= tolerance
            for difference, tolerance in zip(absolute_differences, element_tolerances)
        )
        differences[name] = {
            "passed": property_passed,
            "finite": finite,
            "max_abs_difference": max_abs,
            "reference_max_abs": scale,
            "max_element_tolerance": max_tolerance,
        }
        passed = passed and property_passed
    return passed, differences


def _worker(args) -> int:
    _bootstrap_explicit_extension()
    import numpy as np
    from ase.io import read
    from symmetrix.runtime_diagnostics import openmp_runtime_diagnostics

    import symmetrix
    from symmetrix import Symmetrix

    allocated_cpus = _parse_cpu_set(args.worker_cpu_set)
    before = openmp_runtime_diagnostics(
        strict=True,
        expected_threads=args.worker_threads,
        allocated_cpus=allocated_cpus,
    )
    if not before["ok"] or before["status"] != "ok":
        raise RuntimeError(
            "OpenMP runtime doctor failed before evaluation:\n"
            + json.dumps(before, indent=2, sort_keys=True)
        )

    atoms = read(args.structure)
    atoms.info["electric_field"] = np.asarray(args.electric_field, dtype=float)
    calculator = Symmetrix(
        args.model,
        use_kokkos=True,
        dtype=args.dtype,
        streamed_edges=args.worker_mode,
        low_memory=args.worker_mode == "direct",
    )
    for _ in range(args.warmups):
        calculator.calculate(atoms, properties=PROPERTIES)
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        calculator.calculate(atoms, properties=PROPERTIES)
        samples.append(time.perf_counter() - start)
    results = {
        name: np.asarray(calculator.results[name], dtype=np.float64).tolist()
        for name in PROPERTIES
    }
    after = openmp_runtime_diagnostics(
        strict=True,
        expected_threads=args.worker_threads,
        allocated_cpus=allocated_cpus,
    )
    if not after["ok"] or after["status"] != "ok":
        raise RuntimeError(
            "OpenMP runtime doctor failed after evaluation:\n"
            + json.dumps(after, indent=2, sort_keys=True)
        )
    runtime = after["runtime"]
    if runtime["kokkos_execution_space"] != "OpenMP":
        raise RuntimeError(
            "qualification requires Kokkos OpenMP, got "
            + str(runtime["kokkos_execution_space"])
        )
    if runtime["kokkos_concurrency"] != args.worker_threads:
        raise RuntimeError(
            f"Kokkos exposes concurrency {runtime['kokkos_concurrency']}; "
            f"expected {args.worker_threads}"
        )
    report = {
        "threads": args.worker_threads,
        "mode": args.worker_mode,
        "m0_tile": args.worker_m0_tile,
        "case": {
            "atoms": len(atoms),
            "volume_A3": float(atoms.get_volume()),
            "cutoff_A": float(calculator.cutoff),
            "neighbor_skin_A": float(calculator.neighbor_skin),
            "effective_cutoff_A": float(calculator.cutoff + calculator.neighbor_skin),
            "directed_edges": len(calculator._neighbor_cache.receivers),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "KOKKOS_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                *BLAS_ENVIRONMENT,
                "SYMMETRIX_OPENMP_RUNTIME_CHECK",
            )
        },
        "runtime": after,
        "results": results,
        "timing": _timing_summary(samples, len(atoms)),
        "jit": {
            "status": calculator.jit_status,
            "artifact_id": calculator.jit_artifact_id,
            "compiler_backend": calculator.jit_compiler_backend,
        },
        "standard_m0": {
            "selected_executor": calculator.evaluator.standard_m0_selected_executor,
            "forward_launch_count": (
                calculator.evaluator.standard_m0_module_forward_launch_count
            ),
            "reverse_launch_count": (
                calculator.evaluator.standard_m0_module_reverse_launch_count
            ),
            "input_scale_adjoint_launch_count": (
                calculator.evaluator.standard_m0_input_scale_adjoint_launch_count
            ),
        },
    }
    Path(args.worker_output).write_text(json.dumps(report, indent=2, sort_keys=True))
    atoms.calc = None
    del calculator
    gc.collect()
    if symmetrix._kokkos_is_initialized():
        symmetrix._finalize_kokkos()
    return 0


def _run_worker(
    args,
    *,
    threads: int,
    mode: str,
    output: Path,
    cpu_set: str | None,
    m0_tile: int | None = None,
):
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "KOKKOS_NUM_THREADS": str(threads),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "SYMMETRIX_OPENMP_RUNTIME_CHECK": "strict",
            "SYMMETRIX_JIT_POLICY": "required",
        }
    )
    for name in BLAS_ENVIRONMENT:
        environment[name] = "1"
    if m0_tile is None:
        environment.pop("SYMMETRIX_STANDARD_M0_HOST_TILE", None)
    else:
        environment["SYMMETRIX_STANDARD_M0_HOST_TILE"] = str(m0_tile)
    interpreter = [sys.executable]
    if sys.flags.no_site:
        interpreter.append("-S")
    command = [
        *interpreter,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-threads",
        str(threads),
        "--worker-output",
        str(output),
        "--worker-mode",
        mode,
        "--model",
        str(args.model),
        "--structure",
        str(args.structure),
        "--dtype",
        args.dtype,
        "--electric-field",
        *map(str, args.electric_field),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
    ]
    if cpu_set:
        command.extend(("--worker-cpu-set", cpu_set))
    if m0_tile is not None:
        command.extend(("--worker-m0-tile", str(m0_tile)))
    if cpu_set:
        command = ["taskset", "-c", cpu_set, *command]
    completed = subprocess.run(
        command,
        check=False,
        env=environment,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{threads}-thread qualification worker failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(output.read_text())


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("generic", "direct"),
        default=("generic", "direct"),
    )
    parser.add_argument(
        "--cpu-sets",
        nargs="+",
        help="CPU sets corresponding one-for-one with --threads",
    )
    parser.add_argument("--one-cpu-set")
    parser.add_argument("--many-cpu-set")
    parser.add_argument(
        "--electric-field", nargs=3, type=float, default=(0.01, -0.02, 0.03)
    )
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--m0-tiles", nargs="+", type=int, default=(1, 4, 8, 16))
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode", choices=("generic", "direct"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-cpu-set", help=argparse.SUPPRESS)
    parser.add_argument("--worker-m0-tile", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        return _worker(args)
    threads = sorted(set(args.threads))
    if not threads or threads[0] <= 0:
        raise ValueError("--threads values must be positive")
    if 1 not in threads:
        threads.insert(0, 1)
    if args.warmups < 0 or args.samples <= 0:
        raise ValueError("--warmups must be non-negative and --samples positive")
    if args.min_speedup < 0:
        raise ValueError("--min-speedup must be non-negative")
    if set(args.m0_tiles) != {1, 4, 8, 16}:
        raise ValueError("--m0-tiles must contain exactly 1, 4, 8, and 16")
    if args.cpu_sets and len(args.cpu_sets) != len(args.threads):
        raise ValueError("--cpu-sets must have one entry per explicit --threads value")
    for path in (args.model, args.structure):
        if not path.is_file():
            raise FileNotFoundError(path)
    atol = (
        args.atol
        if args.atol is not None
        else (2e-3 if args.dtype == "float32" else 1e-10)
    )
    rtol = (
        args.rtol
        if args.rtol is not None
        else (2e-5 if args.dtype == "float32" else 1e-10)
    )
    cpu_sets = {}
    if args.cpu_sets:
        cpu_sets.update(zip(args.threads, args.cpu_sets))
    if args.one_cpu_set:
        cpu_sets[1] = args.one_cpu_set
    if args.many_cpu_set:
        cpu_sets[max(threads)] = args.many_cpu_set
    for thread_count, cpu_set in cpu_sets.items():
        cpus = _parse_cpu_set(cpu_set)
        if cpus is None or len(cpus) < thread_count:
            raise ValueError(
                f"CPU set {cpu_set!r} cannot host {thread_count} OpenMP workers"
            )

    workers = {}
    tile_workers = {}
    with tempfile.TemporaryDirectory(prefix="symmetrix-openmp-qualification-") as raw:
        temporary = Path(raw)
        for mode in dict.fromkeys(args.modes):
            for thread_count in threads:
                key = f"{mode}-{thread_count}"
                workers[key] = _run_worker(
                    args,
                    threads=thread_count,
                    mode=mode,
                    output=temporary / f"{key}.json",
                    cpu_set=cpu_sets.get(thread_count),
                )
        tile_threads = max(threads)
        for tile in args.m0_tiles:
            key = f"tile-{tile}"
            tile_workers[str(tile)] = _run_worker(
                args,
                threads=tile_threads,
                mode="direct",
                output=temporary / f"{key}.json",
                cpu_set=cpu_sets.get(tile_threads),
                m0_tile=tile,
            )

    reference_key = "generic-1" if "generic-1" in workers else "direct-1"
    reference = workers[reference_key]
    passed = True
    differences = {}
    for key, worker in workers.items():
        if worker["case"] != reference["case"]:
            raise RuntimeError(
                f"qualification worker {key} built a different case:\n"
                + json.dumps(
                    {"reference": reference["case"], "candidate": worker["case"]},
                    indent=2,
                )
            )
        result_passed, result_differences = _compare_results(
            reference["results"], worker["results"], atol=atol, rtol=rtol
        )
        differences[key] = result_differences
        passed = passed and result_passed

    tile_reference = tile_workers["1"]
    tile_differences = {}
    for tile, worker in tile_workers.items():
        result_passed, result_differences = _compare_results(
            tile_reference["results"], worker["results"], atol=atol, rtol=rtol
        )
        launched = worker["standard_m0"]["reverse_launch_count"] > 0
        captured_scale_adjoint = (
            worker["standard_m0"]["input_scale_adjoint_launch_count"] > 0
        )
        tile_differences[tile] = {
            "passed": result_passed and launched and captured_scale_adjoint,
            "standard_m0_reverse_launched": launched,
            "input_scale_adjoint_captured": captured_scale_adjoint,
            "properties": result_differences,
        }
        passed = passed and result_passed and launched and captured_scale_adjoint

    speedups = {}
    for mode in dict.fromkeys(args.modes):
        one = workers[f"{mode}-1"]["timing"]["median_ms"]
        many = workers[f"{mode}-{max(threads)}"]["timing"]["median_ms"]
        speedups[mode] = _speedup_summary(
            one,
            many,
            reference_threads=1,
            candidate_threads=max(threads),
            minimum=args.min_speedup,
        )
        passed = passed and speedups[mode]["passed"]

    report = {
        "schema": "symmetrix.openmp-full-property-qualification",
        "version": 2,
        "passed": passed,
        "model": {"path": str(args.model.resolve()), "sha256": _sha256(args.model)},
        "structure": {
            "path": str(args.structure.resolve()),
            "sha256": _sha256(args.structure),
        },
        "case": reference["case"],
        "configuration": {
            "dtype": args.dtype,
            "modes": list(dict.fromkeys(args.modes)),
            "threads": threads,
            "cpu_sets": {str(key): value for key, value in cpu_sets.items()},
            "m0_tiles": list(args.m0_tiles),
            "warmups": args.warmups,
            "samples": args.samples,
            "minimum_speedup": args.min_speedup,
            "atol": atol,
            "rtol": rtol,
            "electric_field": list(args.electric_field),
        },
        "differences": differences,
        "m0_tile_differences": tile_differences,
        "speedups": speedups,
        "workers": {
            key: {name: value for name, value in worker.items() if name != "results"}
            for key, worker in workers.items()
        },
        "m0_tile_workers": {
            key: {name: value for name, value in worker.items() if name != "results"}
            for key, worker in tile_workers.items()
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
