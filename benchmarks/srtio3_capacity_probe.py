#!/usr/bin/env python3
"""Probe one fresh-process Symmetrix capacity point on cubic SrTiO3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_axes(value: str) -> tuple[int, int, int]:
    axes = tuple(int(item) for item in value.split(","))
    if len(axes) != 3 or any(item <= 0 for item in axes):
        raise argparse.ArgumentTypeError("axes must contain three positive integers")
    return axes


def build_structure(axes: tuple[int, int, int]):
    import numpy as np
    from ase import Atoms

    return Atoms(
        "SrTiO3",
        scaled_positions=(
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        cell=np.eye(3) * 3.905,
        pbc=True,
    ).repeat(axes)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def query_gpu(selector: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            selector,
            "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU row, received {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"expected seven GPU fields, received {len(fields)}")
    return {
        "selector": selector,
        "physical_index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "driver": fields[5],
        "compute_capability": fields[6],
    }


class GpuMemorySampler:
    def __init__(self, selector: str, interval_s: float):
        self.selector = selector
        self.interval_s = interval_s
        self.baseline: dict[str, Any] | None = None
        self.latest: dict[str, Any] | None = None
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        row = query_gpu(self.selector)
        self.latest = row
        if self.baseline is None:
            self.baseline = row
        used = int(row["memory_used_mib"])
        self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)

    def start(self) -> None:
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()

    def record(self) -> dict[str, Any]:
        return {
            "gpu": self.latest,
            "baseline_used_mib": (
                None if self.baseline is None else self.baseline["memory_used_mib"]
            ),
            "latest_used_mib": (
                None if self.latest is None else self.latest["memory_used_mib"]
            ),
            "peak_sampled_used_mib": self.peak_mib,
            "sample_interval_ms": 1000.0 * self.interval_s,
        }


def metric(instance, name: str, default=None):
    value = getattr(instance, name, default)
    return value() if callable(value) else value


def selected_plan_id(calculator) -> str | None:
    plan = calculator.execution_plan
    if isinstance(plan, dict):
        selected = plan.get("selected_id")
        return None if selected is None else str(selected)
    return None if plan is None else str(plan)


def failure_class(error: BaseException) -> str:
    message = str(error).lower()
    if any(
        marker in message
        for marker in (
            "out of memory",
            "cudaerrormemoryallocation",
            "cuda memory space failed to allocate",
            "cudaerror_memory_allocation",
        )
    ):
        return "cuda_oom"
    if "illegal memory" in message or "cudaerrorillegaladdress" in message:
        return "cuda_illegal_address"
    return "error"


def run(args: argparse.Namespace) -> dict[str, Any]:
    from symmetrix import Symmetrix, backend_loader, selected_backend

    atoms = build_structure(args.axes)
    model = args.model.resolve()
    sampler = GpuMemorySampler(args.gpu_device, args.memory_sample_interval)
    sampler.start()
    calculator = None
    native = None
    cache = None
    evaluator = None
    phase = "calculator_construction"
    record: dict[str, Any] = {
        "schema": "symmetrix.srtio3-capacity-probe/1",
        "status": "running",
        "pid": os.getpid(),
        "phase": phase,
        "structure": "unperturbed cubic Pm-3m SrTiO3",
        "lattice_A": 3.905,
        "repeat_axes": list(args.axes),
        "atoms": len(atoms),
        "dtype": "float32",
        "requested_profile": "capacity",
        "model": str(model),
        "model_sha256": sha256_file(model),
        "neighbor_skin_A": args.skin,
        "memory": sampler.record(),
    }
    atomic_json(args.output, record)
    try:
        calculator = Symmetrix(
            model,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="direct",
            execution_profile="capacity",
            neighbor_skin=args.skin,
        )
        phase = "neighbor_graph_preparation"
        record["phase"] = phase
        calculator._cached_neighbor_geometry(atoms, native_geometry=True)
        cache = calculator._neighbor_cache
        if cache is None:
            raise RuntimeError("evaluation did not retain a neighbor graph")
        record.update(
            {
                "directed_edges": int(cache.num_edges),
                "model_cutoff_A": float(calculator.cutoff),
                "effective_cutoff_A": float(calculator.cutoff + args.skin),
                "memory": sampler.record(),
            }
        )
        atomic_json(args.output, record)

        properties = ["energy", "energies", "forces", "stress"]
        phase = "first_evaluation"
        record["phase"] = phase
        setup_started = time.perf_counter_ns()
        calculator.calculate(atoms, properties=properties)
        setup_ms = (time.perf_counter_ns() - setup_started) / 1.0e6

        phase = "warmup"
        record["phase"] = phase
        for _ in range(args.warmups):
            calculator.calculate(atoms, properties=properties, system_changes=[])

        phase = "measurement"
        record["phase"] = phase
        samples_ms = []
        for _ in range(args.samples):
            started = time.perf_counter_ns()
            calculator.calculate(atoms, properties=properties, system_changes=[])
            samples_ms.append((time.perf_counter_ns() - started) / 1.0e6)

        evaluator = calculator.evaluator
        fallback_count = int(
            metric(evaluator, "factorized_fallback_evaluation_count", 0)
        )
        if fallback_count:
            raise RuntimeError(f"observed {fallback_count} fallback evaluations")
        native = backend_loader._native_module
        if native is None:
            raise RuntimeError("native backend was not loaded")
        native_path = Path(native.__file__).resolve()
        median_ms = statistics.median(samples_ms)
        sampler.stop()
        record.update(
            {
                "status": "success",
                "phase": "complete",
                "selected_plan": selected_plan_id(calculator),
                "execution_plan": calculator.execution_plan,
                "fallback_evaluations": fallback_count,
                "timing": {
                    "boundary": "complete ASE energy+energies+forces+stress evaluation with retained neighbor graph",
                    "first_evaluation_ms": setup_ms,
                    "warmups": args.warmups,
                    "samples": args.samples,
                    "samples_ms": samples_ms,
                    "median_ms": median_ms,
                    "median_us_per_atom": 1000.0 * median_ms / len(atoms),
                },
                "jit": {
                    "policy": os.environ.get("SYMMETRIX_JIT_POLICY"),
                    "status": calculator.jit_status,
                    "compiler_backend": calculator.jit_compiler_backend,
                    "artifact_id": calculator.jit_artifact_id,
                    "variant_id": getattr(calculator, "jit_variant_id", None),
                },
                "backend": selected_backend(),
                "kokkos_execution_space": native._kokkos_default_execution_space(),
                "native_extension": str(native_path),
                "native_extension_sha256": sha256_file(native_path),
                "native_build": native._backend_build_info(),
                "memory": sampler.record(),
            }
        )
        atomic_json(args.output, record)
        return record
    except BaseException as error:
        sampler.stop()
        record.update(
            {
                "status": "failure",
                "phase": phase,
                "failure": {
                    "classification": failure_class(error),
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
                "selected_plan": (
                    None if calculator is None else selected_plan_id(calculator)
                ),
                "execution_plan": (
                    None if calculator is None else calculator.execution_plan
                ),
                "memory": sampler.record(),
            }
        )
        atomic_json(args.output, record)
        raise
    finally:
        # Capacity failures can leave CUDA allocations alive.  Release every
        # Python owner before finalizing Kokkos so its Cuda singleton does not
        # emit a secondary shutdown warning after the real failure.
        if calculator is not None:
            calculator.evaluator = None
        cache = None
        evaluator = None
        calculator = None
        atoms = None
        gc.collect()
        if native is None:
            native = backend_loader._native_module
        is_initialized = getattr(native, "_kokkos_is_initialized", None)
        finalize = getattr(native, "_finalize_kokkos", None)
        if callable(is_initialized) and callable(finalize) and is_initialized():
            finalize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--axes", type=parse_axes, required=True)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--memory-sample-interval", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.samples < 1:
        parser.error("warmups must be non-negative and samples must be positive")
    if args.skin < 0.0 or args.memory_sample_interval <= 0.0:
        parser.error("skin must be non-negative and sampling interval positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
