#!/usr/bin/env python3
"""Benchmark fixed receiver workspaces for one- and two-layer MACE on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time


RESULT_PREFIX = "SYMMETRIX_SINGLE_LAYER_WORKSPACE_RESULT="
DEFAULT_CAPACITIES = (1024, 2048, 4096, 8192, 16384, 32768)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(instance, name: str, default=None):
    value = getattr(instance, name, default)
    return value() if callable(value) else value


def _bootstrap_explicit_symmetrix():
    source_root = Path(os.environ["SYMMETRIX_SOURCE_ROOT"]).resolve()
    extension = Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()
    package_dir = source_root / "symmetrix/source/symmetrix"
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(f"invalid SYMMETRIX_SOURCE_ROOT: {source_root}")
    if not extension.is_file():
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
    extension_module = extension.name.split(".", maxsplit=1)[0]
    native_module_name = f"symmetrix.{extension_module}"
    native_spec = importlib.util.spec_from_file_location(native_module_name, extension)
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("could not load the requested Symmetrix package")
    if native_spec is None or native_spec.loader is None:
        raise RuntimeError("could not load the requested Symmetrix extension")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["symmetrix"] = package
    package_spec.loader.exec_module(package)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules[native_module_name] = native
    native_spec.loader.exec_module(native)
    sys.modules["symmetrix.symmetrix"] = native
    from symmetrix import backend_loader

    build = native._backend_build_info()
    backend = str(build["backend"])
    backend_loader._native_module = native
    backend_loader._selected = backend_loader.BackendDescriptor(
        selector=f"explicit-{backend}-benchmark",
        backend=backend,
        architecture=str(build["architecture"]),
        distribution=str(build["distribution"]),
        frontend_version=package.__version__,
        native_abi=int(build["native_abi"]),
        package="symmetrix",
        module=extension_module,
        descriptor_path="explicit benchmark extension",
    )
    return native


def _gpu_info() -> dict[str, str | int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("benchmark requires exactly one visible NVIDIA GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError("unexpected nvidia-smi GPU identity output")
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "driver_version": fields[3],
        "compute_capability": fields[4],
        "memory_total_mib": int(fields[5]),
    }


def _process_gpu_memory_mib() -> int | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    values = []
    for row in completed.stdout.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            values.append(int(fields[1]))
    return sum(values) if values else None


class _GpuMemorySampler:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.baseline_mib: int | None = None
        self.current_mib: int | None = None
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self):
        value = _process_gpu_memory_mib()
        if value is None:
            return
        if self.baseline_mib is None:
            self.baseline_mib = value
        self.current_mib = value
        self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _parse_capacities(value: str) -> tuple[int, ...]:
    capacities = tuple(int(item) for item in value.split(",") if item.strip())
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise argparse.ArgumentTypeError("capacities must be positive integers")
    if len(set(capacities)) != len(capacities):
        raise argparse.ArgumentTypeError("capacities must be unique")
    return capacities


def _parse_repeats(value: str) -> tuple[int, ...]:
    repeats = tuple(int(item) for item in value.split(",") if item.strip())
    if not repeats or any(repeat <= 0 for repeat in repeats):
        raise argparse.ArgumentTypeError("repeats must be positive integers")
    if len(set(repeats)) != len(repeats):
        raise argparse.ArgumentTypeError("repeats must be unique")
    return repeats


def _worker(args: argparse.Namespace) -> dict:
    native = _bootstrap_explicit_symmetrix()
    import numpy as np
    from ase.build import bulk
    from symmetrix import Symmetrix

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    execution_space = native._kokkos_default_execution_space()
    if execution_space != "Cuda":
        raise RuntimeError(f"expected Kokkos Cuda, got {execution_space}")

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
        (args.repeat, args.repeat, args.repeat)
    )
    tiled_plan = (
        "mh0-single-layer-tiled-v1"
        if args.workspace_plan == "single"
        else "mh0-dual-layer-tiled-v1"
    )
    metric_prefix = f"{args.workspace_plan}_layer"
    requested_plan = (
        None
        if args.worker_capacity == -1
        else "mh0-direct-capacity-y-only"
        if args.worker_capacity == 0
        else tiled_plan
    )
    sampler = _GpuMemorySampler(args.memory_sample_interval)
    sampler.start()
    calculator = Symmetrix(
        args.model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=args.skin,
        _debug_execution_plan=requested_plan,
    )
    if args.worker_capacity > 0:
        getattr(
            calculator.evaluator,
            f"_set_{metric_prefix}_workspace_receiver_limit_for_testing",
        )(args.worker_capacity)
    properties = ["energy", "energies", "forces", "stress"]
    calculator.calculate(atoms, properties=properties)
    for _ in range(args.warmups):
        calculator.calculate(atoms, properties=properties)
    samples_ms = []
    for _ in range(args.samples):
        started = time.perf_counter_ns()
        calculator.calculate(atoms, properties=properties)
        samples_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    sampler.stop()

    selected_plan = calculator.execution_plan
    if isinstance(selected_plan, dict):
        selected_plan = selected_plan.get("selected_id")
    if requested_plan is not None and selected_plan != requested_plan:
        raise RuntimeError(
            f"requested execution plan {requested_plan}, "
            f"selected {calculator.execution_plan}"
        )
    evaluator = calculator.evaluator
    if _metric(evaluator, "factorized_fallback_evaluation_count", 0) != 0:
        raise RuntimeError("fixed-workspace benchmark observed an execution fallback")
    cache = calculator._neighbor_cache
    if cache is None:
        raise RuntimeError("benchmark did not retain a prepared neighbor graph")
    atoms_count = len(atoms)
    median_ms = statistics.median(samples_ms)
    expected_evaluations = 1 + args.warmups + args.samples
    active_capacity = int(
        _metric(evaluator, f"{metric_prefix}_workspace_active_receivers", 0)
    )
    expected_batches = (
        0
        if selected_plan != tiled_plan
        else expected_evaluations
        * ((atoms_count + active_capacity - 1) // active_capacity)
        * (1 if args.workspace_plan == "single" else 3)
    )
    observed_batches = int(
        _metric(evaluator, f"{metric_prefix}_workspace_batch_count", 0)
    )
    if observed_batches != expected_batches:
        raise RuntimeError(
            f"expected {expected_batches} receiver batches, observed {observed_batches}"
        )

    model = args.model.resolve()
    extension = Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()
    result = {
        "schema": f"symmetrix.{args.workspace_plan}-layer-fixed-workspace-cuda/1",
        "workspace_plan": args.workspace_plan,
        "plan": selected_plan,
        "receiver_capacity_requested": (
            args.worker_capacity if args.worker_capacity > 0 else None
        ),
        "receiver_capacity_active": active_capacity,
        "receiver_capacity_planned": int(
            _metric(evaluator, f"{metric_prefix}_workspace_planned_receivers", 0)
        ),
        "edge_capacity_active": int(
            _metric(evaluator, f"{metric_prefix}_workspace_active_edges", 0)
        ),
        "edge_capacity_planned": int(
            _metric(evaluator, f"{metric_prefix}_workspace_planned_edges", 0)
        ),
        "workspace_bytes": int(
            _metric(evaluator, f"{metric_prefix}_workspace_bytes", 0)
        ),
        "workspace_bytes_per_receiver": int(
            _metric(evaluator, f"{metric_prefix}_workspace_bytes_per_receiver", 0)
        ),
        "workspace_bytes_per_edge": int(
            _metric(evaluator, f"{metric_prefix}_workspace_bytes_per_edge", 0)
        ),
        "source_segments": int(
            _metric(evaluator, "dual_layer_source_segment_count", 0)
        ),
        "source_schedule_bytes": int(
            _metric(evaluator, "dual_layer_schedule_bytes", 0)
        ),
        "schedule_preparation_explicit_scratch_bytes": int(
            _metric(
                evaluator,
                "dual_layer_schedule_preparation_explicit_scratch_bytes",
                0,
            )
        ),
        "graph_device_capacity_bytes": int(
            _metric(evaluator, "factorized_graph_device_capacity_bytes", 0)
        ),
        "geometry_workspace_bytes": int(
            _metric(evaluator, "execution_geometry_workspace_bytes", 0)
        ),
        "workspace_replacements": int(
            _metric(evaluator, f"{metric_prefix}_workspace_replacement_count", 0)
        ),
        "workspace_reuses": int(
            _metric(evaluator, f"{metric_prefix}_workspace_reuse_count", 0)
        ),
        "receiver_batches": observed_batches,
        "tiled_evaluations": int(
            _metric(evaluator, f"{metric_prefix}_tiled_evaluation_count", 0)
        ),
        "execution_planned_bytes": int(
            _metric(evaluator, "execution_planned_capacity_bytes", 0)
        ),
        "atoms": atoms_count,
        "directed_edges": int(cache.num_edges),
        "model_cutoff_A": float(calculator.cutoff),
        "neighbor_skin_A": float(calculator.neighbor_skin),
        "effective_cutoff_A": float(calculator.cutoff + calculator.neighbor_skin),
        "warmups": args.warmups,
        "samples": args.samples,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "median_us_per_atom": 1000.0 * median_ms / atoms_count,
        "energy_eV": float(calculator.results["energy"]),
        "force_l2_eV_per_A": float(np.linalg.norm(calculator.results["forces"])),
        "force_max_eV_per_A": float(np.max(np.abs(calculator.results["forces"]))),
        "stress_eV_per_A3": np.asarray(calculator.results["stress"]).tolist(),
        "harmonic_storage_policy": _metric(evaluator, "harmonic_storage_policy"),
        "harmonic_gradient_bytes": int(
            _metric(evaluator, "harmonic_gradient_bytes", 0)
        ),
        "fallback_evaluations": int(
            _metric(evaluator, "factorized_fallback_evaluation_count", 0)
        ),
        "gpu": _gpu_info(),
        "gpu_process_memory_baseline_mib": sampler.baseline_mib,
        "gpu_process_memory_current_mib": sampler.current_mib,
        "gpu_process_memory_peak_sampled_mib": sampler.peak_mib,
        "model": str(model),
        "model_sha256": _sha256(model),
        "extension": str(extension),
        "extension_sha256": _sha256(extension),
        "execution_space": execution_space,
    }
    del evaluator, calculator, cache
    gc.collect()
    if native._kokkos_is_initialized():
        native._finalize_kokkos()
    return result


def _run_worker(args: argparse.Namespace, capacity: int, output: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.model),
        "--worker-capacity",
        str(capacity),
        "--workspace-plan",
        args.workspace_plan,
        "--repeat",
        str(args.repeat),
        "--skin",
        str(args.skin),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
        "--memory-sample-interval",
        str(args.memory_sample_interval),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=os.environ.copy()
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"workspace benchmark worker failed with exit code "
            f"{completed.returncode}:\n{details}"
        )
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError("workspace benchmark worker did not emit one result")
    result = json.loads(lines[0][len(RESULT_PREFIX) :])
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--workspace-plan",
        choices=("single", "dual"),
        default="single",
        help="fixed-workspace execution plan to benchmark",
    )
    parser.add_argument(
        "--receiver-capacities",
        type=_parse_capacities,
        default=DEFAULT_CAPACITIES,
        help="comma-separated fixed receiver capacities",
    )
    parser.add_argument(
        "--repeats",
        type=_parse_repeats,
        help="run a fresh-process atom-count series using each listed repeat",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--memory-sample-interval", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-capacity", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat <= 0 or args.warmups < 0 or args.samples <= 0:
        parser.error("repeat and samples must be positive and warmups nonnegative")
    if args.memory_sample_interval <= 0:
        parser.error("memory sample interval must be positive")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    for variable in ("SYMMETRIX_SOURCE_ROOT", "SYMMETRIX_EXTENSION"):
        if variable not in os.environ:
            parser.error(f"{variable} must be set")

    if args.worker_capacity is not None:
        if args.worker_capacity < -1:
            parser.error("worker capacity must be -1, zero, or positive")
        result = _worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return

    with tempfile.TemporaryDirectory(
        prefix=f"symmetrix-{args.workspace_plan}-layer-workspace-"
    ) as root:
        root_path = Path(root)
        if args.repeats is None:
            baseline = _run_worker(args, 0, root_path / "baseline.json")
            capacities = [
                _run_worker(args, capacity, root_path / f"capacity-{capacity}.json")
                for capacity in args.receiver_capacities
            ]
            baseline_us = baseline["median_us_per_atom"]
            for record in capacities:
                record["latency_ratio_to_capacity_y_only"] = (
                    record["median_us_per_atom"] / baseline_us
                )
                record["latency_regression_percent"] = 100.0 * (
                    record["latency_ratio_to_capacity_y_only"] - 1.0
                )
            campaign = {
                "schema": (
                    f"symmetrix.{args.workspace_plan}-layer-"
                    "fixed-workspace-cuda-campaign/1"
                ),
                "workspace_plan": args.workspace_plan,
                "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
                "baseline": baseline,
                "capacities": capacities,
            }
        else:
            size_series = []
            for repeat in args.repeats:
                worker_args = argparse.Namespace(**vars(args))
                worker_args.repeat = repeat
                baseline = _run_worker(
                    worker_args, 0, root_path / f"repeat-{repeat}-baseline.json"
                )
                capacities = [
                    _run_worker(
                        worker_args,
                        capacity,
                        root_path / f"repeat-{repeat}-capacity-{capacity}.json",
                    )
                    for capacity in args.receiver_capacities
                ]
                baseline_us = baseline["median_us_per_atom"]
                for record in capacities:
                    record["latency_ratio_to_capacity_y_only"] = (
                        record["median_us_per_atom"] / baseline_us
                    )
                    record["latency_regression_percent"] = 100.0 * (
                        record["latency_ratio_to_capacity_y_only"] - 1.0
                    )
                size_series.append(
                    {
                        "repeat": repeat,
                        "atoms": baseline["atoms"],
                        "directed_edges": baseline["directed_edges"],
                        "baseline": baseline,
                        "capacities": capacities,
                    }
                )
            campaign = {
                "schema": (
                    f"symmetrix.{args.workspace_plan}-layer-"
                    "fixed-workspace-cuda-size-series/1"
                ),
                "workspace_plan": args.workspace_plan,
                "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
                "repeats": list(args.repeats),
                "receiver_capacities": list(args.receiver_capacities),
                "size_series": size_series,
            }
    rendered = json.dumps(campaign, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
