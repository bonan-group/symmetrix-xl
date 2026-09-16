#!/usr/bin/env python3
"""Measure MH-0/MACEField low-memory and MH-1 node-policy CUDA capacity."""

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
import threading
import time
from typing import Any


RESULT_PREFIX = "SYMMETRIX_LOW_MEMORY_CAPACITY_RESULT="
CAMPAIGN_MODEL_KINDS = ("omat0", "macefield")
MODEL_KINDS = (*CAMPAIGN_MODEL_KINDS, "mh1")
LOW_MEMORY_VALUES = ("false", "true")
MH1_NODE_STATE_POLICIES = (
    "full-retention-v1",
    "recompute-v1",
    "reuse-adjoints-v1",
    "retain-interaction-v1",
)
MH1_EDGE_EXECUTORS = ("mlp_reference", "pair_spline_v1")
DTYPES = ("float32", "float64")
DEFAULT_FIELD = (0.01, -0.02, 0.03)


def _bootstrap_explicit_symmetrix() -> None:
    root = Path(os.environ["SYMMETRIX_SOURCE_ROOT"]).resolve()
    extension = Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()
    package_dir = root / "symmetrix/source/symmetrix"
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
        "symmetrix.symmetrix", extension
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("could not load the requested Symmetrix source package")
    if native_spec is None or native_spec.loader is None:
        raise RuntimeError("could not load the requested Symmetrix extension")
    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules["symmetrix.symmetrix"] = native
    native_spec.loader.exec_module(native)
    package_spec.loader.exec_module(package)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _process_memory() -> dict[str, float | None]:
    result: dict[str, float | None] = {"current_mib": None, "peak_mib": None}
    with Path("/proc/self/status").open(encoding="ascii") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key == "VmRSS":
                result["current_mib"] = int(value.split()[0]) / 1024.0
            elif key == "VmHWM":
                result["peak_mib"] = int(value.split()[0]) / 1024.0
    return result


def _gpu_process_memory_mib(pid: int | None = None) -> int | None:
    selected_pid = os.getpid() if pid is None else pid
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
    total = 0
    found = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(selected_pid):
            total += int(fields[1])
            found = True
    return total if found else None


def _selected_gpu_selector(logical_ordinal: int, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(logical_ordinal)
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    if logical_ordinal < 0 or logical_ordinal >= len(devices):
        raise RuntimeError(
            f"CUDA logical device {logical_ordinal} is absent from "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return devices[logical_ordinal]


def _gpu_device_memory(selector: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            selector,
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        return None
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 5:
        return None
    try:
        index = int(fields[0])
        total_mib = int(fields[3])
        used_mib = int(fields[4])
    except ValueError:
        return None
    return {
        "selector": selector,
        "physical_index": index,
        "uuid": fields[1],
        "name": fields[2],
        "total_mib": total_mib,
        "used_mib": used_mib,
    }


class _GpuMemorySampler:
    def __init__(self, selector: str, interval_s: float = 0.05):
        self.selector = selector
        self.interval_s = interval_s
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self.current_mib: int | None = None
        self.device: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False

    def _sample(self) -> None:
        device = _gpu_device_memory(self.selector)
        if device is not None:
            value = int(device["used_mib"])
            self.device = device
            self.current_mib = value
            if self.baseline_mib is None:
                self.baseline_mib = value
            self.peak_mib = (
                value if self.peak_mib is None else max(self.peak_mib, value)
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> None:
        self._sample()
        if self.device is None or self.baseline_mib is None:
            raise RuntimeError(
                f"could not query CUDA device {self.selector} with nvidia-smi"
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _gpu_sampler_record(sampler: _GpuMemorySampler) -> dict[str, Any]:
    return {
        "gpu_device": sampler.device,
        "gpu_device_used_baseline_mib": sampler.baseline_mib,
        "gpu_device_used_current_mib": sampler.current_mib,
        "gpu_device_peak_sampled_mib": sampler.peak_mib,
        "gpu_process_current_mib": _gpu_process_memory_mib(),
    }


def _wait_for_gpu_memory_recovery(
    selector: str,
    baseline_mib: int,
    timeout_s: float,
    tolerance_mib: int,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_s
    latest = _gpu_device_memory(selector)
    while (
        latest is not None
        and int(latest["used_mib"]) > baseline_mib + tolerance_mib
        and time.monotonic() < deadline
    ):
        time.sleep(0.25)
        latest = _gpu_device_memory(selector)
    if latest is None:
        raise RuntimeError(
            f"could not verify GPU {selector} memory recovery with nvidia-smi"
        )
    if latest is not None and int(latest["used_mib"]) > baseline_mib + tolerance_mib:
        raise RuntimeError(
            f"GPU {selector} memory did not recover to {baseline_mib} + "
            f"{tolerance_mib} MiB within {timeout_s} seconds; latest usage is "
            f"{latest['used_mib']} MiB"
        )
    return latest


def _metric(evaluator, name: str, default=None):
    value = getattr(evaluator, name, default)
    return value() if callable(value) else value


def _summary(samples_ms: list[float], atoms: int) -> dict[str, Any]:
    median_ms = statistics.median(samples_ms)
    return {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "median_us_per_atom": 1000.0 * median_ms / atoms,
    }


def _worker(args: argparse.Namespace) -> int:
    _bootstrap_explicit_symmetrix()
    import numpy as np
    from ase.build import bulk
    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native_symmetrix

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space != "Cuda":
        raise RuntimeError(f"expected Kokkos Cuda, got {execution_space}")

    model = args.model.resolve()
    extension = Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()
    is_mh1 = args.model_kind == "mh1"
    low_memory = None if is_mh1 else args.low_memory == "true"
    properties = ["energy", "forces", "stress"]
    if args.model_kind == "macefield":
        properties.append("polarization")

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
        (args.repeat, args.repeat, args.repeat)
    )
    expected_atoms = 4 * args.repeat**3
    if len(atoms) != expected_atoms:
        raise RuntimeError(f"expected {expected_atoms} atoms, got {len(atoms)}")
    if args.model_kind == "macefield":
        atoms.info["electric_field"] = np.asarray(args.electric_field, dtype=float)

    device_environment = dict(
        native_symmetrix._execution_device_execution_environment()
    )
    logical_device = int(device_environment.get("device_ordinal", 0))
    gpu_selector = _selected_gpu_selector(logical_device, args.gpu_device)
    sampler = _GpuMemorySampler(gpu_selector, args.memory_sample_interval)
    sampler.start()
    calculator = None
    phase = "calculator_construction"
    partial_record: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "active_phase": phase,
        "pid": os.getpid(),
        "model_kind": args.model_kind,
        "dtype": args.dtype,
        "low_memory": low_memory,
        "mh1_node_state_policy": (args.mh1_node_state_policy if is_mh1 else None),
        "mh1_edge_executor": args.mh1_edge_executor if is_mh1 else None,
        "repeat": args.repeat,
        "atoms": len(atoms),
        "properties": properties,
        "workload": {
            "model_cutoff_A": None,
            "neighbor_skin_A": args.neighbor_skin,
            "effective_neighbor_cutoff_A": None,
            "directed_edges": None,
        },
        "memory": _gpu_sampler_record(sampler),
        "provenance": {
            "model": str(model),
            "model_sha256": _sha256(model),
            "extension": str(extension),
            "extension_sha256": _sha256(extension),
            "source_root": str(Path(os.environ["SYMMETRIX_SOURCE_ROOT"]).resolve()),
            "execution_space": execution_space,
            "device_environment": device_environment,
            "gpu_selector": gpu_selector,
        },
    }
    _atomic_json(args.output.resolve(), partial_record)
    try:
        setup_started = time.perf_counter()
        calculator_options = {
            "use_kokkos": True,
            "dtype": args.dtype,
            "streamed_edges": "direct" if is_mh1 else "factorized",
            "neighbor_skin": args.neighbor_skin,
        }
        if is_mh1:
            calculator_options["execution_mh1_node_state_policy"] = (
                args.mh1_node_state_policy
            )
            calculator_options["execution_mh1_node_arena_policy"] = "capacity-v1"
        else:
            calculator_options["low_memory"] = low_memory
        calculator = Symmetrix(model, **calculator_options)
        if is_mh1:
            calculator.evaluator.set_mh1_edge_executor(args.mh1_edge_executor)
        if not is_mh1 and bool(calculator.evaluator.has_field_coupling) != (
            args.model_kind == "macefield"
        ):
            raise RuntimeError("model field-coupling flag does not match model kind")

        phase = "neighbor_graph_preparation"
        partial_record["active_phase"] = phase
        partial_record["workload"].update(
            {
                "model_cutoff_A": float(calculator.evaluator.r_cut),
                "effective_neighbor_cutoff_A": float(
                    calculator.evaluator.r_cut + calculator.neighbor_skin
                ),
            }
        )
        _atomic_json(args.output.resolve(), partial_record)
        calculator._cached_neighbor_geometry(atoms, native_geometry=True)
        cache = calculator._neighbor_cache
        if cache is None:
            raise RuntimeError("capacity worker did not retain the neighbor cache")
        directed_edges = int(cache.receivers.size)
        partial_record["workload"]["directed_edges"] = directed_edges
        partial_record["completed_phase"] = phase
        phase = "first_model_evaluation"
        partial_record["active_phase"] = phase
        partial_record["memory"] = _gpu_sampler_record(sampler)
        _atomic_json(args.output.resolve(), partial_record)

        calculator.calculate(atoms, properties=properties)
        setup_ms = 1000.0 * (time.perf_counter() - setup_started)
        phase = "warmup"
        for _ in range(args.warmups):
            calculator.calculate(atoms, properties=properties)

        phase = "measurement"
        samples_ms = []
        for _ in range(args.samples):
            started = time.perf_counter_ns()
            calculator.calculate(atoms, properties=properties)
            samples_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    except BaseException as error:
        sampler.stop()
        partial_record.update(
            {
                "status": "failure",
                "active_phase": phase,
                "failure": {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
                "memory": {
                    "process": _process_memory(),
                    **_gpu_sampler_record(sampler),
                },
            }
        )
        _atomic_json(args.output.resolve(), partial_record)
        raise
    finally:
        sampler.stop()

    assert calculator is not None
    evaluator = calculator.evaluator
    if (
        int(_metric(evaluator, "execution_geometry_capacity_edges", 0))
        != directed_edges
    ):
        raise RuntimeError(
            "native geometry capacity does not match the candidate graph"
        )
    if not is_mh1 and calculator.low_memory is not low_memory:
        raise RuntimeError("effective low-memory policy does not match the request")
    if is_mh1:
        selected_node_state_policy = getattr(calculator, "jit_node_state_policy", None)
        if selected_node_state_policy != args.mh1_node_state_policy:
            raise RuntimeError(
                "effective MH-1 node-state policy does not match the request: "
                f"{selected_node_state_policy!r} != {args.mh1_node_state_policy!r}"
            )
        if evaluator.mh1_edge_executor != args.mh1_edge_executor:
            raise RuntimeError(
                "effective MH-1 edge executor does not match the request: "
                f"{evaluator.mh1_edge_executor!r} != {args.mh1_edge_executor!r}"
            )
        if int(_metric(evaluator, "factorized_fallback_evaluation_count", 0)) != 0:
            raise RuntimeError("prepared MH-1 execution used the fallback path")
    if (
        args.model_kind == "macefield"
        and int(_metric(evaluator, "macefield_response_primal_reconstruction_count", 0))
        != 0
    ):
        raise RuntimeError("primal MACEField workload entered analytical response")

    forces = np.asarray(calculator.results["forces"], dtype=np.float64)
    stress = np.asarray(calculator.results["stress"], dtype=np.float64)
    result_summary: dict[str, Any] = {
        "energy_eV": float(calculator.results["energy"]),
        "force_l2_eV_per_A": float(np.linalg.norm(forces)),
        "force_max_abs_eV_per_A": float(np.max(np.abs(forces))),
        "stress_eV_per_A3": stress.tolist(),
    }
    if args.model_kind == "macefield":
        result_summary["polarization"] = np.asarray(
            calculator.results["polarization"], dtype=np.float64
        ).tolist()

    memory_names = (
        "edge_workspace_bytes",
        "node_workspace_bytes",
        "precision_workspace_bytes",
        "factorized_workspace_bytes",
        "factorized_workspace_capacity_bytes",
        "factorized_schedule_bytes",
        "factorized_schedule_entries",
        "factorized_radial_workspace_bytes",
        "factorized_arena_workspace_bytes",
        "factorized_coupling_workspace_bytes",
        "factorized_compact_workspace_bytes",
        "standard_r0_workspace_bytes",
        "execution_geometry_workspace_bytes",
        "compact_edge_geometry_bytes",
        "harmonic_value_bytes",
        "harmonic_gradient_bytes",
        "shuffled_coordinate_bytes",
        "low_memory_device_free_bytes",
        "low_memory_device_total_bytes",
        "low_memory_reserve_bytes",
        "low_memory_capacity_y_only_estimated_bytes",
        "low_memory_capacity_retained_estimated_bytes",
        "low_memory_available_bytes",
        "low_memory_speed_estimated_bytes",
        "low_memory_capacity_estimated_bytes",
        "low_memory_selected_estimated_bytes",
        "m1_poly_values_active_bytes",
        "m1_poly_values_capacity_bytes",
        "m1_poly_adjoints_active_bytes",
        "m1_poly_adjoints_capacity_bytes",
        "mh0_reused_state_bytes",
        "mh0_auxiliary_state_bytes",
        "execution_mh1_scratch_minimum_bytes",
        "execution_mh1_scratch_planned_bytes",
        "execution_mh1_scratch_retained_bytes",
        "execution_mh1_scratch_recomputed_bytes",
    )
    selection_names = (
        "streamed_edges_mode",
        "factorized_source_strategy",
        "factorized_execution_strategy",
        "factorized_selected_direct_forward_executor",
        "factorized_selected_direct_reverse_executor",
        "standard_m0_selected_executor",
        "standard_r0_selected_executor",
        "m1_polynomial_policy",
        "m1_polynomial_policy_request",
        "mh0_state_policy",
        "mh0_state_policy_request",
        "edge_geometry_policy",
        "harmonic_storage_policy",
        "harmonic_storage_policy_request",
        "harmonic_storage_selection_reason",
        "harmonic_storage_fallback_reason",
        "execution_direct_harmonic_launch_count",
        "low_memory",
        "low_memory_requested",
        "low_memory_policy",
        "low_memory_selection_reason",
        "execution_mh1_execution_backend",
        "execution_mh1_node_state_policy",
        "execution_mh1_node_arena_policy",
        "execution_mh1_node_arena_tile_rows",
        "mh1_edge_executor",
        "factorized_prepared_evaluation_count",
        "factorized_fallback_evaluation_count",
        "execution_mh1_generated_forward_launch_count",
        "execution_mh1_generated_source_reverse_launch_count",
        "execution_mh1_generated_edge_reverse_launch_count",
        "execution_geometry_growth_reason",
    )
    record = {
        "schema_version": 2,
        "status": "success",
        "pid": os.getpid(),
        "model_kind": args.model_kind,
        "dtype": args.dtype,
        "low_memory": low_memory,
        "mh1_node_state_policy": (args.mh1_node_state_policy if is_mh1 else None),
        "mh1_edge_executor": args.mh1_edge_executor if is_mh1 else None,
        "repeat": args.repeat,
        "atoms": len(atoms),
        "properties": properties,
        "electric_field_V_per_A": (
            list(args.electric_field) if args.model_kind == "macefield" else None
        ),
        "workload": {
            "model_cutoff_A": float(evaluator.r_cut),
            "neighbor_skin_A": float(calculator.neighbor_skin),
            "effective_neighbor_cutoff_A": float(
                evaluator.r_cut + calculator.neighbor_skin
            ),
            "directed_edges": directed_edges,
        },
        "timing": {
            "setup_and_first_evaluation_ms": setup_ms,
            **_summary(samples_ms, len(atoms)),
            "warmups": args.warmups,
            "samples": args.samples,
        },
        "memory": {
            "process": _process_memory(),
            **_gpu_sampler_record(sampler),
            "native": {name: _metric(evaluator, name, 0) for name in memory_names},
        },
        "selection": {
            "jit_status": calculator.jit_status,
            "jit_compiler_backend": calculator.jit_compiler_backend,
            "jit_artifact_id": calculator.jit_artifact_id,
            "jit_variant_id": getattr(calculator, "jit_variant_id", None),
            "jit_node_state_policy": getattr(calculator, "jit_node_state_policy", None),
            **{name: _metric(evaluator, name) for name in selection_names},
            "macefield_response_primal_reconstruction_count": int(
                _metric(
                    evaluator,
                    "macefield_response_primal_reconstruction_count",
                    0,
                )
            ),
        },
        "result": result_summary,
        "provenance": {
            "model": str(model),
            "model_sha256": _sha256(model),
            "extension": str(extension),
            "extension_sha256": _sha256(extension),
            "source_root": str(Path(os.environ["SYMMETRIX_SOURCE_ROOT"]).resolve()),
            "execution_space": execution_space,
            "device_environment": device_environment,
            "gpu_selector": gpu_selector,
        },
    }
    _atomic_json(args.output.resolve(), record)
    print(RESULT_PREFIX + json.dumps(record, separators=(",", ":")), flush=True)
    cache = None
    evaluator = None
    calculator = None
    gc.collect()
    if native_symmetrix._kokkos_is_initialized():
        native_symmetrix._finalize_kokkos()
    return 0


def _failure_classification(returncode: int, stderr: str) -> str:
    lowered = stderr.lower()
    cuda_markers = {
        "cuda_oom": (
            "out of memory",
            "cudaerrormemoryallocation",
            "cuda memory space failed to allocate",
            "cudaerror_memory_allocation",
        ),
        "cuda_illegal_address": (
            "cudaerrorillegaladdress",
            "illegal memory access",
        ),
    }
    first_cuda_failure: tuple[int, str] | None = None
    for classification, markers in cuda_markers.items():
        positions = [lowered.find(marker) for marker in markers]
        positions = [position for position in positions if position >= 0]
        if positions:
            candidate = (min(positions), classification)
            if first_cuda_failure is None or candidate[0] < first_cuda_failure[0]:
                first_cuda_failure = candidate
    if first_cuda_failure is not None:
        return first_cuda_failure[1]
    if returncode in (-9, 137) or "memoryerror" in lowered:
        return "host_oom_or_kill"
    if "timed out" in lowered:
        return "timeout"
    return "error"


def _require_cuda_oom_boundary(label: str, classification: str) -> None:
    if classification != "cuda_oom":
        raise RuntimeError(
            f"{label} ended with {classification}; only cuda_oom is a valid "
            "capacity boundary"
        )


def _campaign(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    records_dir = output_dir / "records"
    logs_dir = output_dir / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    models = {
        "omat0": args.omat0_model.resolve(),
        "macefield": args.macefield_model.resolve(),
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "contract": {
            "streamed_edges": "factorized",
            "dtype": args.dtype,
            "neighbor_skin_A": args.neighbor_skin,
            "properties": {
                "omat0": ["energy", "forces", "stress"],
                "macefield": ["energy", "forces", "stress", "polarization"],
            },
            "electric_field_V_per_A": list(args.electric_field),
            "warmups": args.warmups,
            "samples": args.samples,
            "boundary_trials": args.boundary_trials,
            "valid_failure_boundary": "cuda_oom",
            "gpu_recovery_timeout_s": args.gpu_recovery_timeout,
            "gpu_recovery_tolerance_mib": args.gpu_recovery_tolerance_mib,
        },
        "models": {
            kind: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for kind, path in models.items()
        },
        "extension": {
            "path": str(Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()),
            "sha256": _sha256(Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()),
        },
        "records": [],
        "failures": [],
        "boundaries": {},
    }
    report_path = output_dir / "low_memory_capacity.json"
    outcomes: dict[tuple[str, bool, int], list[bool]] = {}
    failure_classes: dict[tuple[str, bool, int], list[str]] = {}
    campaign_gpu_selector = _selected_gpu_selector(0, args.gpu_device)
    campaign_gpu = _gpu_device_memory(campaign_gpu_selector)
    if campaign_gpu is None:
        raise RuntimeError(
            f"could not query CUDA device {campaign_gpu_selector} with nvidia-smi"
        )
    payload["gpu_device"] = campaign_gpu

    def save() -> None:
        _atomic_json(report_path, payload)

    def probe(model_kind: str, low_memory: bool, repeat: int) -> bool:
        key = (model_kind, low_memory, repeat)
        trial = len(outcomes.get(key, [])) + 1
        label = (
            f"{model_kind}-{args.dtype}-low-memory-"
            f"{str(low_memory).lower()}-n{repeat}-t{trial}"
        )
        output = records_dir / f"{label}.json"
        command = [
            sys.executable,
            str(script),
            "worker",
            "--model-kind",
            model_kind,
            "--model",
            str(models[model_kind]),
            "--low-memory",
            str(low_memory).lower(),
            "--dtype",
            args.dtype,
            "--repeat",
            str(repeat),
            "--neighbor-skin",
            str(args.neighbor_skin),
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
            "--memory-sample-interval",
            str(args.memory_sample_interval),
            "--electric-field",
            *[str(value) for value in args.electric_field],
            "--output",
            str(output),
        ]
        if args.gpu_device is not None:
            command.extend(("--gpu-device", args.gpu_device))
        gpu_before = _gpu_device_memory(campaign_gpu_selector)
        if gpu_before is None:
            raise RuntimeError(
                f"could not query CUDA device {campaign_gpu_selector} "
                "immediately before the capacity worker"
            )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=os.environ.copy(),
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            stdout = (
                error.stdout.decode()
                if isinstance(error.stdout, bytes)
                else (error.stdout or "")
            )
            stderr = (
                error.stderr.decode()
                if isinstance(error.stderr, bytes)
                else (error.stderr or "")
            )
            stderr += f"\nworker timed out after {args.timeout} seconds"
            returncode = -1
        (logs_dir / f"{label}.stdout.log").write_text(stdout)
        (logs_dir / f"{label}.stderr.log").write_text(stderr)
        try:
            gpu_after = _wait_for_gpu_memory_recovery(
                campaign_gpu_selector,
                int(gpu_before["used_mib"]),
                args.gpu_recovery_timeout,
                args.gpu_recovery_tolerance_mib,
            )
        except RuntimeError as error:
            payload["failures"].append(
                {
                    "model_kind": model_kind,
                    "low_memory": low_memory,
                    "repeat": repeat,
                    "atoms": 4 * repeat**3,
                    "trial": trial,
                    "returncode": returncode,
                    "classification": "gpu_recovery_error",
                    "gpu_recovery": {
                        "before": gpu_before,
                        "after": _gpu_device_memory(campaign_gpu_selector),
                        "tolerance_mib": args.gpu_recovery_tolerance_mib,
                        "error": str(error),
                    },
                    "stderr_tail": stderr[-4000:],
                }
            )
            save()
            raise
        parsed = [
            json.loads(line.removeprefix(RESULT_PREFIX))
            for line in stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        success = returncode == 0 and len(parsed) == 1 and output.is_file()
        partial = None
        if output.is_file():
            try:
                partial = json.loads(output.read_text())
            except (json.JSONDecodeError, OSError):
                partial = None
        outcomes.setdefault(key, []).append(success)
        if success:
            record = parsed[0]
            record["trial"] = trial
            record["gpu_recovery"] = {
                "before": gpu_before,
                "after": gpu_after,
                "tolerance_mib": args.gpu_recovery_tolerance_mib,
            }
            payload["records"].append(record)
            print(
                f"{label}: success, {record['atoms']} atoms, "
                f"{record['workload']['directed_edges']} edges, "
                f"{record['timing']['median_us_per_atom']:.3f} us/atom, "
                f"{record['memory']['gpu_device_peak_sampled_mib']} MiB",
                flush=True,
            )
        else:
            original_exception = ""
            if isinstance(partial, dict):
                failure_detail = partial.get("failure")
                if isinstance(failure_detail, dict):
                    original_exception = str(failure_detail.get("message", ""))
            classification = _failure_classification(
                returncode, original_exception + "\n" + stderr
            )
            failure_classes.setdefault(key, []).append(classification)
            failure = {
                "model_kind": model_kind,
                "low_memory": low_memory,
                "repeat": repeat,
                "atoms": 4 * repeat**3,
                "trial": trial,
                "returncode": returncode,
                "classification": classification,
                "active_phase": (
                    partial.get("active_phase") if isinstance(partial, dict) else None
                ),
                "workload": (
                    partial.get("workload") if isinstance(partial, dict) else None
                ),
                "memory": (
                    partial.get("memory") if isinstance(partial, dict) else None
                ),
                "gpu_recovery": {
                    "before": gpu_before,
                    "after": gpu_after,
                    "tolerance_mib": args.gpu_recovery_tolerance_mib,
                },
                "stderr_tail": stderr[-4000:],
            }
            payload["failures"].append(failure)
            print(
                f"{label}: {failure['classification']}, {failure['atoms']} atoms",
                flush=True,
            )
        save()
        if not success:
            _require_cuda_oom_boundary(label, classification)
        return success

    for model_kind in CAMPAIGN_MODEL_KINDS:
        payload["boundaries"][model_kind] = {}
        for low_memory in (False, True):
            lower = args.lower_repeat
            upper = args.upper_repeat
            if not probe(model_kind, low_memory, lower):
                raise RuntimeError(
                    f"initial lower bound n{lower} failed for {model_kind}, "
                    f"low_memory={low_memory}"
                )
            while probe(model_kind, low_memory, upper):
                lower = upper
                if upper >= args.max_repeat:
                    break
                upper = min(
                    args.max_repeat,
                    upper + max(1, upper - args.lower_repeat),
                )
            bounded = not outcomes[(model_kind, low_memory, upper)][-1]
            if bounded:
                while upper - lower > 1:
                    middle = (lower + upper) // 2
                    if probe(model_kind, low_memory, middle):
                        lower = middle
                    else:
                        upper = middle
                while (
                    len(outcomes.get((model_kind, low_memory, lower), []))
                    < args.boundary_trials
                ):
                    if not probe(model_kind, low_memory, lower):
                        raise RuntimeError(
                            f"success boundary n{lower} failed confirmation for "
                            f"{model_kind}, low_memory={low_memory}"
                        )
                while (
                    len(outcomes.get((model_kind, low_memory, upper), []))
                    < args.boundary_trials
                ):
                    if probe(model_kind, low_memory, upper):
                        raise RuntimeError(
                            f"failure boundary n{upper} succeeded confirmation for "
                            f"{model_kind}, low_memory={low_memory}"
                        )
            boundary = {
                "largest_success_repeat": lower,
                "largest_success_atoms": 4 * lower**3,
                "first_failure_repeat": upper if bounded else None,
                "first_failure_atoms": 4 * upper**3 if bounded else None,
                "bounded": bounded,
                "success_trials": outcomes.get((model_kind, low_memory, lower), []),
                "failure_trials": (
                    outcomes.get((model_kind, low_memory, upper), []) if bounded else []
                ),
                "failure_class": (
                    failure_classes.get((model_kind, low_memory, upper), [None])[-1]
                    if bounded
                    else None
                ),
            }
            payload["boundaries"][model_kind][str(low_memory).lower()] = boundary
            save()
            print(json.dumps({model_kind: {str(low_memory).lower(): boundary}}))

    save()
    print(
        json.dumps(
            {"report": str(report_path), "boundaries": payload["boundaries"]}, indent=2
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model-kind", choices=MODEL_KINDS, required=True)
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument("--low-memory", choices=LOW_MEMORY_VALUES)
    worker.add_argument(
        "--mh1-node-state-policy",
        choices=MH1_NODE_STATE_POLICIES,
    )
    worker.add_argument(
        "--mh1-edge-executor",
        choices=MH1_EDGE_EXECUTORS,
        default="mlp_reference",
    )
    worker.add_argument("--dtype", choices=DTYPES, default="float32")
    worker.add_argument("--repeat", type=int, required=True)
    worker.add_argument("--neighbor-skin", type=float, default=0.5)
    worker.add_argument("--warmups", type=int, default=1)
    worker.add_argument("--samples", type=int, default=2)
    worker.add_argument("--memory-sample-interval", type=float, default=0.05)
    worker.add_argument(
        "--gpu-device",
        help="Physical nvidia-smi index, GPU UUID, or MIG UUID to sample.",
    )
    worker.add_argument("--electric-field", nargs=3, type=float, default=DEFAULT_FIELD)
    worker.add_argument("--output", type=Path, required=True)

    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--omat0-model", type=Path, required=True)
    campaign.add_argument("--macefield-model", type=Path, required=True)
    campaign.add_argument("--output-dir", type=Path, required=True)
    campaign.add_argument("--dtype", choices=DTYPES, default="float32")
    campaign.add_argument("--lower-repeat", type=int, default=20)
    campaign.add_argument("--upper-repeat", type=int, default=32)
    campaign.add_argument("--max-repeat", type=int, default=64)
    campaign.add_argument("--boundary-trials", type=int, default=2)
    campaign.add_argument("--neighbor-skin", type=float, default=0.5)
    campaign.add_argument("--warmups", type=int, default=1)
    campaign.add_argument("--samples", type=int, default=2)
    campaign.add_argument("--memory-sample-interval", type=float, default=0.05)
    campaign.add_argument(
        "--gpu-device",
        help="Physical nvidia-smi index, GPU UUID, or MIG UUID to sample.",
    )
    campaign.add_argument("--gpu-recovery-timeout", type=float, default=30.0)
    campaign.add_argument("--gpu-recovery-tolerance-mib", type=int, default=16)
    campaign.add_argument(
        "--electric-field", nargs=3, type=float, default=DEFAULT_FIELD
    )
    campaign.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "worker":
        if args.model_kind == "mh1":
            if args.mh1_node_state_policy is None:
                raise ValueError("MH-1 workers require --mh1-node-state-policy")
            if args.low_memory is not None:
                raise ValueError("MH-1 workers do not accept --low-memory")
        else:
            if args.low_memory is None:
                raise ValueError("OMAT-0/MACEField workers require --low-memory")
            if args.mh1_node_state_policy is not None:
                raise ValueError(
                    "OMAT-0/MACEField workers do not accept --mh1-node-state-policy"
                )
    if hasattr(args, "repeat") and args.repeat < 1:
        raise ValueError("repeat must be positive")
    if hasattr(args, "lower_repeat") and not (
        1 <= args.lower_repeat < args.upper_repeat <= args.max_repeat
    ):
        raise ValueError("campaign repeats must satisfy 1 <= lower < upper <= max")
    if hasattr(args, "boundary_trials") and args.boundary_trials < 1:
        raise ValueError("boundary trials must be positive")
    if hasattr(args, "timeout") and args.timeout <= 0.0:
        raise ValueError("campaign timeout must be positive")
    if hasattr(args, "gpu_recovery_timeout") and (
        args.gpu_recovery_timeout <= 0.0 or args.gpu_recovery_tolerance_mib < 0
    ):
        raise ValueError(
            "GPU recovery timeout must be positive and tolerance nonnegative"
        )
    if args.warmups < 0 or args.samples < 1:
        raise ValueError("warmups must be nonnegative and samples must be positive")
    if args.neighbor_skin < 0.0 or args.memory_sample_interval <= 0.0:
        raise ValueError(
            "neighbor skin must be nonnegative and sampling interval positive"
        )
    if args.command == "worker":
        return _worker(args)
    return _campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
