#!/usr/bin/env python3
"""Fresh-process MACEField versus standard MACE AlN scale benchmark."""

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


def _bootstrap():
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
    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules["symmetrix.symmetrix"] = native
    native_spec.loader.exec_module(native)
    package_spec.loader.exec_module(package)


for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

CELLS = {
    "small": ((2, 2, 2), 32),
    "medium": ((11, 11, 11), 5324),
    "n6": ((6, 6, 6), 864),
    "n12": ((12, 12, 12), 6912),
    "n20": ((20, 20, 20), 32000),
    "n30": ((30, 30, 30), 108000),
}
PREFIX = "SYMMETRIX_ALN_SCALE_RESULT="


def _metric(evaluator, name, default=None):
    value = getattr(evaluator, name, default)
    return value() if callable(value) else value


def _process_memory():
    result = {"current_mib": None, "peak_mib": None}
    with Path("/proc/self/status").open(encoding="ascii") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key == "VmRSS":
                result["current_mib"] = int(value.split()[0]) / 1024.0
            elif key == "VmHWM":
                result["peak_mib"] = int(value.split()[0]) / 1024.0
    return result


def _gpu_memory():
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
    total = 0
    found = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            total += int(fields[1])
            found = True
    return total if found else None


class _GpuMemorySampler:
    def __init__(self, enabled, interval_seconds=0.05):
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.peak_mib = None
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        value = _gpu_memory()
        if value is not None:
            self.peak_mib = (
                value if self.peak_mib is None else max(self.peak_mib, value)
            )

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self):
        if not self.enabled:
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _timing_scope(response, response_routing):
    if response == "none":
        return "native-evaluator"
    if response_routing == "public-native":
        return "calculator-results"
    return "cached-host-results"


def _validate_r1_forward_selection(r1_forward, streamed_edges):
    if r1_forward != "automatic" and streamed_edges != "direct":
        raise ValueError("R1 forward selection requires streamed_edges='direct'")


def _configure_policy(evaluator, backend, kind, policy, m1_tile_channels=32):
    if backend == "openmp":
        if policy == "all_interactions":
            return
        if policy == "optimized":
            evaluator._set_standard_m0_executor("automatic")
            evaluator._set_standard_r0_executor("automatic")
            return
        raise ValueError(
            "OpenMP benchmark policy must be all_interactions or optimized"
        )
    if policy == "retained":
        evaluator._set_standard_m0_executor("runtime")
        evaluator._set_standard_r0_executor("v1")
    elif policy == "m0_only":
        evaluator._set_standard_m0_executor("standard")
        evaluator._set_standard_r0_executor("v1")
    elif policy == "r0_only":
        evaluator._set_standard_m0_executor("runtime")
        evaluator._set_standard_r0_executor("automatic")
    elif policy == "optimized":
        evaluator._set_standard_m0_executor("automatic")
        evaluator._set_standard_r0_executor("automatic")
    elif policy == "optimized_recompute":
        evaluator._set_standard_m0_executor("automatic")
        evaluator._set_standard_r0_executor("automatic")
        evaluator._set_m1_recompute_tile_channels(m1_tile_channels)
        evaluator._set_m1_polynomial_policy("recompute")
    else:
        raise ValueError(f"unknown policy: {policy}")


def _worker(args):
    requested_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    initial_affinity = sorted(os.sched_getaffinity(0))
    if args.backend == "openmp" and len(initial_affinity) < requested_threads:
        raise RuntimeError(
            "OpenMP worker inherited only "
            f"{initial_affinity} for {requested_threads} requested threads"
        )

    # Keep the run-mode supervisor free of Kokkos/OpenMP initialization. An
    # OMP-bound supervisor would otherwise spawn every worker with its affinity
    # inherited from the supervisor's single master-thread place.
    _bootstrap()
    import numpy as np
    from ase.build import bulk
    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native_symmetrix

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    execution_space = native_symmetrix._kokkos_default_execution_space()
    expected_space = "OpenMP" if args.backend == "openmp" else "Cuda"
    if execution_space != expected_space:
        raise RuntimeError(f"expected Kokkos {expected_space}, got {execution_space}")

    streamed_edges = (
        "generic"
        if args.backend == "openmp" and args.policy == "all_interactions"
        else "direct"
    )
    calculator = Symmetrix(
        args.model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges=streamed_edges,
        low_memory=args.low_memory,
        m1_polynomial_policy=(
            "recompute"
            if args.low_memory or args.edge_geometry_policy == "unit-f32-radius-f64-v1"
            else "retained"
        ),
        edge_geometry_policy=args.edge_geometry_policy,
        neighbor_skin=args.neighbor_skin,
    )
    evaluator = calculator.evaluator
    expected_field = args.kind == "field"
    if bool(evaluator.has_field_coupling) != expected_field:
        raise RuntimeError("model field-coupling flag does not match kind")
    analytical_response = args.response in ("polarizability", "becs", "both")
    if analytical_response and not expected_field:
        raise ValueError("analytic electric-field responses require the field model")
    if (
        args.edge_geometry_policy == "unit-f32-radius-f64-v1"
        and args.policy != "optimized_recompute"
    ):
        raise ValueError("compact edge geometry requires policy='optimized_recompute'")
    _configure_policy(
        evaluator,
        args.backend,
        args.kind,
        args.policy,
        args.m1_tile_channels,
    )
    _validate_r1_forward_selection(args.r1_forward, streamed_edges)
    if args.r1_forward != "automatic":
        evaluator._set_factorized_direct_forward_executor(args.r1_forward)

    if args.repeat is not None:
        repeat = (args.repeat, args.repeat, args.repeat)
        expected_atoms = 4 * args.repeat**3
        size_label = f"n{args.repeat}"
    else:
        repeat, expected_atoms = CELLS[args.size]
        size_label = args.size
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(repeat)
    if len(atoms) != expected_atoms:
        raise RuntimeError(f"unexpected atom count: {len(atoms)}")
    setup_started = time.perf_counter()
    inputs = calculator._mace_inputs(atoms)
    xyz = np.ascontiguousarray(inputs[5], dtype=float).reshape(-1)
    distances = np.ascontiguousarray(inputs[6], dtype=float)
    field = np.ascontiguousarray(args.electric_field, dtype=float)
    if streamed_edges == "direct":
        token = evaluator._prepare_factorized_graph(*inputs[:5])
        if expected_field:

            def evaluate_base():
                evaluator._compute_prepared_factorized_field(
                    token, xyz, distances, field
                )
        else:

            def evaluate_base():
                evaluator._compute_prepared_factorized(token, xyz, distances)
    else:
        native_args = (*inputs[:5], xyz, distances)
        if expected_field:

            def evaluate_base():
                evaluator.compute_node_energies_forces_field(*native_args, field)
        else:

            def evaluate_base():
                evaluator.compute_node_energies_forces(*native_args)

    response_result = None
    if args.response == "none":
        evaluate = evaluate_base
    else:
        response_properties = {
            "polarization": ("energy", "forces", "stress", "polarization"),
            "polarizability": ("energy", "forces", "stress", "polarizability"),
            "becs": ("energy", "forces", "stress", "becs"),
            "both": ("energy", "forces", "stress", "polarizability", "becs"),
        }[args.response]
        if not expected_field:
            response_properties = ("energy", "forces", "stress")
        if args.response_routing == "public-native":
            if expected_field:
                atoms.info["electric_field"] = field

            def evaluate():
                nonlocal response_result
                calculator.calculate(atoms, properties=list(response_properties))
                response_result = calculator.results

        elif expected_field:

            def evaluate():
                nonlocal response_result
                response_result = calculator._calculate_macefield_results(
                    atoms,
                    field,
                    response_properties,
                    mace_inputs=inputs,
                )

        else:

            def evaluate():
                nonlocal response_result
                graph_generation = calculator._compute_mace(inputs, atoms=atoms)
                response_result = calculator._collect_mace_results(
                    atoms,
                    inputs,
                    response_properties,
                    graph_generation,
                )

    setup_ms = 1000.0 * (time.perf_counter() - setup_started)

    gpu_sampler = _GpuMemorySampler(args.backend == "cuda")
    gpu_sampler.start()
    try:
        for _ in range(args.warmups):
            evaluate()
        counter_names = (
            "factorized_jit_forward_launch_count",
            "factorized_jit_reverse_launch_count",
            "factorized_evaluation_fence_count",
            "standard_r0_module_launch_count",
            "standard_m0_module_forward_launch_count",
            "standard_m0_module_reverse_launch_count",
        )
        counters_before = {
            name: int(_metric(evaluator, name, 0)) for name in counter_names
        }
        samples = []
        for _ in range(args.samples):
            started = time.perf_counter_ns()
            evaluate()
            samples.append((time.perf_counter_ns() - started) / 1.0e6)
        counter_deltas = {
            name: int(_metric(evaluator, name, 0)) - counters_before[name]
            for name in counter_names
        }
    finally:
        gpu_sampler.stop()

    energies = np.asarray(evaluator.node_energies, dtype=np.float64)
    forces = np.asarray(evaluator.node_forces, dtype=np.float64)
    result = {
        "backend": args.backend,
        "execution_space": execution_space,
        "threads": requested_threads,
        "initial_affinity_cpus": initial_affinity,
        "kind": args.kind,
        "policy": args.policy,
        "timing_scope": _timing_scope(args.response, args.response_routing),
        "response": args.response,
        "response_routing": args.response_routing,
        "size": size_label,
        "repeat": list(repeat),
        "atoms": len(atoms),
        "directed_edges": len(inputs[3]),
        "model_cutoff_A": float(evaluator.r_cut),
        "neighbor_skin_A": float(calculator.neighbor_skin),
        "effective_neighbor_cutoff_A": float(
            evaluator.r_cut + calculator.neighbor_skin
        ),
        "warmups": args.warmups,
        "samples": args.samples,
        "low_memory_requested": args.low_memory,
        "low_memory_active": bool(calculator.low_memory),
        "setup_ms": setup_ms,
        "timing": _summary(samples),
        "timing_us_per_atom": 1000.0 * statistics.median(samples) / len(atoms),
        "throughput_atoms_per_second": (
            len(atoms) * 1000.0 / statistics.median(samples)
        ),
        "memory": {
            "process": _process_memory(),
            "gpu_process_mib": _gpu_memory() if args.backend == "cuda" else None,
            "gpu_process_peak_sampled_mib": gpu_sampler.peak_mib,
            "m0_values_active_bytes": int(
                _metric(evaluator, "standard_m0_poly_values_active_bytes", 0)
            ),
            "m0_adjoints_active_bytes": int(
                _metric(evaluator, "standard_m0_poly_adjoints_active_bytes", 0)
            ),
            "m1_values_active_bytes": int(
                _metric(evaluator, "m1_poly_values_active_bytes", 0)
            ),
            "m1_adjoints_active_bytes": int(
                _metric(evaluator, "m1_poly_adjoints_active_bytes", 0)
            ),
            "m1_values_capacity_bytes": int(
                _metric(evaluator, "m1_poly_values_capacity_bytes", 0)
            ),
            "m1_adjoints_capacity_bytes": int(
                _metric(evaluator, "m1_poly_adjoints_capacity_bytes", 0)
            ),
            "m1_recompute_scratch_bytes": int(
                _metric(evaluator, "m1_recompute_scratch_bytes", 0)
            ),
            "m1_recompute_scratch_limit_bytes": int(
                _metric(evaluator, "m1_recompute_scratch_limit_bytes", 0)
            ),
            "macefield_response_m1_recompute_scratch_bytes": int(
                _metric(
                    evaluator,
                    "macefield_response_m1_recompute_scratch_bytes",
                    0,
                )
            ),
            "macefield_response_m1_recompute_scratch_limit_bytes": int(
                _metric(
                    evaluator,
                    "macefield_response_m1_recompute_scratch_limit_bytes",
                    0,
                )
            ),
            "execution_geometry_workspace_bytes": int(
                _metric(evaluator, "execution_geometry_workspace_bytes", 0)
            ),
            "compact_edge_geometry_bytes": int(
                _metric(evaluator, "compact_edge_geometry_bytes", 0)
            ),
        },
        "selection": {
            "streamed_edges_requested": streamed_edges,
            "streamed_edges": calculator.streamed_edges,
            "execution_algorithm": calculator.execution_algorithm,
            "low_memory": bool(calculator.low_memory),
            "mh0_state_policy": _metric(evaluator, "mh0_state_policy", "unavailable"),
            "mh0_state_policy_request": _metric(
                evaluator, "mh0_state_policy_request", "unavailable"
            ),
            "mh0_state_policy_fallback_reason": _metric(
                evaluator, "mh0_state_policy_fallback_reason", ""
            ),
            "edge_geometry_policy": _metric(
                evaluator, "edge_geometry_policy", args.edge_geometry_policy
            ),
            "requested_r1_forward": args.r1_forward,
            "jit_status": calculator.jit_status,
            "jit_compiler_backend": calculator.jit_compiler_backend,
            "jit_artifact_id": calculator.jit_artifact_id,
            "jit_cache_key": calculator.jit_cache_key,
            "r1_forward": _metric(
                evaluator, "factorized_selected_direct_forward_executor"
            ),
            "r1_reverse": _metric(
                evaluator, "factorized_selected_direct_reverse_executor"
            ),
            "m0": _metric(evaluator, "standard_m0_selected_executor"),
            "r0": _metric(evaluator, "standard_r0_selected_executor"),
            "m0_implementation": _metric(evaluator, "m0_implementation"),
            "r0_implementation": _metric(evaluator, "r0_implementation"),
            "jit_operator_modules": dict(
                getattr(calculator, "jit_operator_modules", {})
            ),
            "m1": _metric(evaluator, "m1_polynomial_policy"),
            "m1_requested_policy": _metric(evaluator, "m1_polynomial_policy_request"),
            "m1_recompute_backend": _metric(evaluator, "m1_recompute_backend"),
            "m1_recompute_fallback_reason": _metric(
                evaluator, "m1_recompute_fallback_reason", ""
            ),
            "standard_m1_module_ready": bool(
                _metric(evaluator, "standard_m1_module_ready", False)
            ),
            "standard_m1_module_forward_launch_count": int(
                _metric(evaluator, "standard_m1_module_forward_launch_count", 0)
            ),
            "standard_m1_module_reverse_launch_count": int(
                _metric(evaluator, "standard_m1_module_reverse_launch_count", 0)
            ),
            "m1_recompute_tile_channels": int(
                _metric(evaluator, "m1_recompute_tile_channels", 0)
            ),
            "macefield_response_m1_recompute_tile_channels": int(
                _metric(
                    evaluator,
                    "macefield_response_m1_recompute_tile_channels",
                    0,
                )
            ),
            "macefield_response_m1_recompute_forward_launch_count": int(
                _metric(
                    evaluator,
                    "macefield_response_m1_recompute_forward_launch_count",
                    0,
                )
            ),
            "macefield_response_m1_recompute_reverse_launch_count": int(
                _metric(
                    evaluator,
                    "macefield_response_m1_recompute_reverse_launch_count",
                    0,
                )
            ),
            "macefield_response_receiver_ownership": _metric(
                evaluator, "macefield_response_receiver_ownership", "unavailable"
            ),
            "macefield_response_source_ownership": _metric(
                evaluator, "macefield_response_source_ownership", "unavailable"
            ),
            "macefield_response_call_count": int(
                _metric(evaluator, "macefield_response_call_count", 0)
            ),
            "macefield_response_factorized_topology_count": int(
                _metric(evaluator, "macefield_response_factorized_topology_count", 0)
            ),
            "macefield_response_phi1_fused_launch_count": int(
                _metric(evaluator, "macefield_response_phi1_fused_launch_count", 0)
            ),
            "macefield_response_phi1_generic_launch_count": int(
                _metric(evaluator, "macefield_response_phi1_generic_launch_count", 0)
            ),
            "macefield_response_a0_fused_launch_count": int(
                _metric(evaluator, "macefield_response_a0_fused_launch_count", 0)
            ),
            "macefield_response_a0_generic_launch_count": int(
                _metric(evaluator, "macefield_response_a0_generic_launch_count", 0)
            ),
            "prepared_graph_count": int(
                _metric(evaluator, "factorized_prepared_graph_count", 0)
            ),
            "prepared_evaluation_count": int(
                _metric(evaluator, "factorized_prepared_evaluation_count", 0)
            ),
            "fallback_evaluation_count": int(
                _metric(evaluator, "factorized_fallback_evaluation_count", 0)
            ),
            "schedule_build_count": int(
                _metric(evaluator, "factorized_schedule_build_count", 0)
            ),
            "measured_counter_deltas": counter_deltas,
        },
        "result": {
            "energy_eV": float(energies.sum()),
            "force_l2_eV_per_A": float(np.linalg.norm(forces)),
            "force_max_abs_eV_per_A": float(np.max(np.abs(forces))),
            "field_adjoint": (
                np.asarray(evaluator.electric_field_adj, dtype=float).tolist()
                if expected_field
                else None
            ),
            "stress_shape": (
                list(np.asarray(response_result["stress"]).shape)
                if response_result is not None and "stress" in response_result
                else None
            ),
            "stress_max_abs_eV_per_A3": (
                float(np.max(np.abs(response_result["stress"])))
                if response_result is not None and "stress" in response_result
                else None
            ),
            "polarization_shape": (
                list(np.asarray(response_result["polarization"]).shape)
                if response_result is not None and "polarization" in response_result
                else None
            ),
            "polarization": (
                np.asarray(response_result["polarization"], dtype=float).tolist()
                if response_result is not None and "polarization" in response_result
                else None
            ),
            "polarizability_shape": (
                list(np.asarray(response_result["polarizability"]).shape)
                if response_result is not None and "polarizability" in response_result
                else None
            ),
            "polarizability_max_abs": (
                float(np.max(np.abs(response_result["polarizability"])))
                if response_result is not None and "polarizability" in response_result
                else None
            ),
            "becs_shape": (
                list(np.asarray(response_result["becs"]).shape)
                if response_result is not None and "becs" in response_result
                else None
            ),
            "becs_max_abs": (
                float(np.max(np.abs(response_result["becs"])))
                if response_result is not None and "becs" in response_result
                else None
            ),
        },
    }
    print(PREFIX + json.dumps(result, separators=(",", ":")))
    evaluate = None
    evaluate_base = None
    evaluator = None
    calculator = None
    gc.collect()
    native_symmetrix._finalize_kokkos()


def _run(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / f"{output.stem}-logs"
    logs.mkdir(parents=True, exist_ok=True)
    records = []
    failures = []
    models = {
        "field": args.field_model.resolve(),
        "standard": args.standard_model.resolve(),
    }
    for size in args.sizes:
        for trial in range(args.trials):
            policy_order = (
                tuple(args.policies)
                if trial % 2 == 0
                else tuple(reversed(args.policies))
            )
            for policy in policy_order:
                order = (
                    ("field", "standard") if trial % 2 == 0 else ("standard", "field")
                )
                for kind in order:
                    if (
                        args.response in ("polarizability", "becs", "both")
                        and kind == "standard"
                    ):
                        continue
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        "--backend",
                        args.backend,
                        "--kind",
                        kind,
                        "--policy",
                        policy,
                        "--model",
                        str(models[kind]),
                        "--size",
                        size,
                        "--warmups",
                        str(args.warmups),
                        "--samples",
                        str(args.samples),
                        "--electric-field",
                        *[str(value) for value in args.electric_field],
                        "--m1-tile-channels",
                        str(args.m1_tile_channels),
                        "--response",
                        args.response,
                        "--response-routing",
                        args.response_routing,
                        "--edge-geometry-policy",
                        args.edge_geometry_policy,
                        "--neighbor-skin",
                        str(args.neighbor_skin),
                    ]
                    if args.low_memory:
                        command.append("--low-memory")
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                        env=os.environ.copy(),
                    )
                    stem = f"{args.backend}-{size}-{policy}-{trial + 1:02d}-{kind}"
                    (logs / f"{stem}.stdout.log").write_text(completed.stdout)
                    (logs / f"{stem}.stderr.log").write_text(completed.stderr)
                    parsed = [
                        json.loads(line.removeprefix(PREFIX))
                        for line in completed.stdout.splitlines()
                        if line.startswith(PREFIX)
                    ]
                    record_matches_request = (
                        len(parsed) == 1 and parsed[0].get("response") == args.response
                    )
                    if args.response in ("polarizability", "both"):
                        record_matches_request = record_matches_request and parsed[
                            0
                        ].get("result", {}).get("polarizability_shape") == [9]
                    if args.response in ("becs", "both"):
                        record_matches_request = record_matches_request and parsed[
                            0
                        ].get("result", {}).get("becs_shape") == [
                            parsed[0].get("atoms"),
                            9,
                        ]
                    if args.response == "polarization":
                        expected_shape = [3] if kind == "field" else None
                        record_matches_request = (
                            record_matches_request
                            and parsed[0].get("result", {}).get("polarization_shape")
                            == expected_shape
                        )
                    if completed.returncode == 0 and record_matches_request:
                        parsed[0]["trial"] = trial + 1
                        records.append(parsed[0])
                        print(
                            f"{stem}: {parsed[0]['timing']['median_ms']:.6f} ms",
                            flush=True,
                        )
                    else:
                        failure = {
                            "backend": args.backend,
                            "size": size,
                            "policy": policy,
                            "trial": trial + 1,
                            "kind": kind,
                            "returncode": completed.returncode,
                            "requested_response": args.response,
                            "recorded_response": (
                                parsed[0].get("response") if len(parsed) == 1 else None
                            ),
                            "stderr_tail": completed.stderr[-4000:],
                        }
                        failures.append(failure)
                        print(f"{stem}: FAILED", flush=True)
    output.write_text(
        json.dumps({"records": records, "failures": failures}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"output": str(output), "records": len(records), "failures": len(failures)}
        )
    )


def _capacity(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / f"{output.stem}-logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(args.model.resolve()),
        "backend": "cuda",
        "policy": args.policy,
        "streamed_edges": "factorized",
        "response": args.response,
        "lower_repeat": args.lower_repeat,
        "initial_upper_repeat": args.upper_repeat,
        "max_repeat": args.max_repeat,
        "records": [],
        "failures": [],
    }
    outcomes = {}

    def save():
        output.write_text(json.dumps(payload, indent=2) + "\n")

    def probe(repeat, trial):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--backend",
            "cuda",
            "--kind",
            "field",
            "--policy",
            args.policy,
            "--model",
            str(args.model.resolve()),
            "--repeat",
            str(repeat),
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
            "--response",
            args.response,
            "--response-routing",
            args.response_routing,
            "--edge-geometry-policy",
            args.edge_geometry_policy,
            "--m1-tile-channels",
            str(args.m1_tile_channels),
            "--electric-field",
            *[str(value) for value in args.electric_field],
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=os.environ.copy(),
        )
        stem = f"cuda-{args.response}-n{repeat}-trial-{trial:02d}"
        (logs / f"{stem}.stdout.log").write_text(completed.stdout)
        (logs / f"{stem}.stderr.log").write_text(completed.stderr)
        parsed = [
            json.loads(line.removeprefix(PREFIX))
            for line in completed.stdout.splitlines()
            if line.startswith(PREFIX)
        ]
        success = completed.returncode == 0 and len(parsed) == 1
        if success:
            record = parsed[0]
            record["trial"] = trial
            payload["records"].append(record)
            print(
                f"n{repeat} ({record['atoms']} atoms): success, "
                f"{record['timing']['median_ms']:.3f} ms, "
                f"{record['memory']['gpu_process_mib']} MiB",
                flush=True,
            )
        else:
            stderr = completed.stderr
            lowered = stderr.lower()
            classification = (
                "oom"
                if "out of memory" in lowered
                or "cudaerrormemoryallocation" in lowered
                or "cuda memory space failed to allocate" in lowered
                or completed.returncode in (-9, 137)
                else "error"
            )
            payload["failures"].append(
                {
                    "repeat": repeat,
                    "atoms": 4 * repeat**3,
                    "trial": trial,
                    "returncode": completed.returncode,
                    "classification": classification,
                    "stderr_tail": stderr[-4000:],
                }
            )
            print(
                f"n{repeat} ({4 * repeat**3} atoms): {classification}",
                flush=True,
            )
        save()
        outcomes.setdefault(repeat, []).append(success)
        return success

    lower = args.lower_repeat
    upper = args.upper_repeat
    if not probe(lower, 1):
        raise RuntimeError(f"lower capacity bound n{lower} did not complete")
    while probe(upper, 1):
        lower = upper
        if upper >= args.max_repeat:
            payload["qualification"] = {
                "largest_success_repeat": lower,
                "first_failure_repeat": None,
                "bounded": False,
            }
            save()
            return
        upper = min(args.max_repeat, upper + max(1, upper - args.lower_repeat))

    while upper - lower > 1:
        middle = (lower + upper) // 2
        if probe(middle, 1):
            lower = middle
        else:
            upper = middle

    while sum(outcomes.get(lower, [])) < args.boundary_trials:
        probe(lower, len(outcomes.get(lower, [])) + 1)
    while len(outcomes.get(upper, [])) < args.boundary_trials:
        probe(upper, len(outcomes.get(upper, [])) + 1)

    payload["qualification"] = {
        "largest_success_repeat": lower,
        "largest_success_atoms": 4 * lower**3,
        "first_failure_repeat": upper,
        "first_failure_atoms": 4 * upper**3,
        "success_trials": outcomes[lower],
        "failure_trials": outcomes[upper],
        "bounded": True,
    }
    save()
    print(json.dumps(payload["qualification"], indent=2))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parity(args):
    _bootstrap()
    import numpy as np
    from ase.build import bulk
    from mace.calculators import MACECalculator
    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import _load_mace_checkpoint

    checkpoint = args.checkpoint.resolve()
    compact = args.model.resolve()
    field = np.asarray(args.electric_field, dtype=np.float64)
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = field

    model = _load_mace_checkpoint(checkpoint, "cuda")
    reference_calculator = MACECalculator(
        models=[model],
        model_type="MACEField",
        head=args.head,
        device="cuda",
        default_dtype="float32",
    )
    atoms.calc = reference_calculator
    reference = {
        "energy": float(atoms.get_potential_energy()),
        "forces": atoms.get_forces().copy(),
        "stress": atoms.get_stress().copy(),
    }

    def evaluate(calculator, mode):
        calculator.evaluator.set_streamed_edges(mode)
        inputs = calculator._mace_inputs(atoms)
        calculator.evaluator.compute_node_energies_forces_field(
            *inputs[:5],
            np.asarray(inputs[5]).reshape(-1),
            inputs[6],
            field,
        )
        collected = calculator._collect_mace_results(atoms, inputs)
        return {
            "energy": float(collected["energy"]),
            "forces": np.asarray(collected["forces"]).copy(),
            "stress": np.asarray(collected["stress"]).copy(),
            "field_adjoint": np.asarray(calculator.evaluator.electric_field_adj).copy(),
        }

    configurations = [
        (False, "materialized", "off", "native-materialized"),
        (False, "all_interactions", "off", "native-all-interactions"),
        (True, "materialized", "off", "kokkos-materialized"),
        (True, "all_interactions", "off", "kokkos-all-interactions"),
        (True, "factorized", "off", "kokkos-factorized-r1-generic"),
        (True, "factorized", "required", "kokkos-factorized-r1-nvrtc"),
    ]
    records = []
    lifecycle = None
    for use_kokkos, mode, jit_policy, label in configurations:
        calculator = Symmetrix(
            compact,
            use_kokkos=use_kokkos,
            dtype="float32" if use_kokkos else "float64",
            streamed_edges=mode,
            jit=jit_policy,
        )
        if use_kokkos and mode == "factorized":
            calculator.evaluator._set_standard_m0_executor("automatic")
            calculator.evaluator._set_standard_r0_executor("automatic")

        if label == "kokkos-factorized-r1-nvrtc":
            retained_before = evaluate(calculator, "materialized")
            actual = evaluate(calculator, "factorized")
            selection = {
                "m0": calculator.evaluator.standard_m0_selected_executor,
                "r0": calculator.evaluator.standard_r0_selected_executor,
            }
            generated_repeat = evaluate(calculator, "factorized")
            retained_after = evaluate(calculator, "materialized")

            def maximum_difference(lhs, rhs, key):
                return float(
                    np.max(np.abs(np.asarray(lhs[key]) - np.asarray(rhs[key])))
                )

            lifecycle = {
                "jit_status": calculator.jit_status,
                "jit_backend": calculator.jit_compiler_backend,
                "jit_artifact_id": calculator.jit_artifact_id,
                "selection_during_generated": selection,
                "generated_repeat": {
                    key: maximum_difference(actual, generated_repeat, key)
                    for key in ("energy", "forces", "stress", "field_adjoint")
                },
                "retained_restored": {
                    key: maximum_difference(retained_before, retained_after, key)
                    for key in ("energy", "forces", "stress", "field_adjoint")
                },
            }
        else:
            actual = evaluate(calculator, mode)
        records.append(
            {
                "label": label,
                "energy": actual["energy"],
                "energy_abs_error": abs(actual["energy"] - reference["energy"]),
                "forces_max_abs_error": float(
                    np.max(np.abs(actual["forces"] - reference["forces"]))
                ),
                "stress_max_abs_error": float(
                    np.max(np.abs(actual["stress"] - reference["stress"]))
                ),
                "jit_status": calculator.jit_status,
                "jit_backend": calculator.jit_compiler_backend,
            }
        )

    output = {
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "compact_model": str(compact),
            "compact_model_sha256": _sha256(compact),
            "execution_space": native_symmetrix._kokkos_default_execution_space(),
            "head": args.head,
            "electric_field": field.tolist(),
        },
        "reference": {"energy": reference["energy"]},
        "records": records,
        "lifecycle": lifecycle,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    del calculator
    del reference_calculator
    del model
    gc.collect()
    native_symmetrix._finalize_kokkos()


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--backend", choices=("openmp", "cuda"), required=True)
    worker.add_argument("--kind", choices=("field", "standard"), required=True)
    worker.add_argument(
        "--policy",
        choices=(
            "all_interactions",
            "retained",
            "m0_only",
            "r0_only",
            "optimized",
            "optimized_recompute",
        ),
        required=True,
    )
    worker.add_argument("--model", type=Path, required=True)
    worker_group = worker.add_mutually_exclusive_group(required=True)
    worker_group.add_argument("--size", choices=tuple(CELLS))
    worker_group.add_argument("--repeat", type=int)
    worker.add_argument("--warmups", type=int, required=True)
    worker.add_argument("--samples", type=int, required=True)
    worker.add_argument("--low-memory", action="store_true")
    worker.add_argument("--neighbor-skin", type=float, default=0.5)
    worker.add_argument(
        "--r1-forward",
        choices=("automatic", "jit_all", "runtime"),
        default="automatic",
    )
    worker.add_argument("--m1-tile-channels", type=int, choices=(8, 16, 32), default=32)
    worker.add_argument(
        "--edge-geometry-policy",
        choices=("cartesian-f64-v1", "unit-f32-radius-f64-v1"),
        default="cartesian-f64-v1",
    )
    worker.add_argument(
        "--response",
        choices=("none", "polarization", "polarizability", "becs", "both"),
        default="none",
    )
    worker.add_argument(
        "--response-routing",
        choices=("cached-host", "public-native"),
        default="cached-host",
    )
    worker.add_argument(
        "--electric-field", nargs=3, type=float, default=(0.01, -0.02, 0.03)
    )
    run = subparsers.add_parser("run")
    run.add_argument("--backend", choices=("openmp", "cuda"), required=True)
    run.add_argument("--field-model", type=Path, required=True)
    run.add_argument("--standard-model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--sizes", nargs="+", choices=tuple(CELLS), required=True)
    run.add_argument(
        "--policies",
        nargs="+",
        choices=(
            "all_interactions",
            "retained",
            "m0_only",
            "r0_only",
            "optimized",
            "optimized_recompute",
        ),
        required=True,
    )
    run.add_argument("--trials", type=int, default=3)
    run.add_argument("--warmups", type=int, required=True)
    run.add_argument("--samples", type=int, required=True)
    run.add_argument("--low-memory", action="store_true")
    run.add_argument("--neighbor-skin", type=float, default=0.5)
    run.add_argument("--m1-tile-channels", type=int, choices=(8, 16, 32), default=32)
    run.add_argument(
        "--response",
        choices=("none", "polarization", "polarizability", "becs", "both"),
        default="none",
    )
    run.add_argument(
        "--response-routing",
        choices=("cached-host", "public-native"),
        default="cached-host",
    )
    run.add_argument(
        "--edge-geometry-policy",
        choices=("cartesian-f64-v1", "unit-f32-radius-f64-v1"),
        default="cartesian-f64-v1",
    )
    run.add_argument(
        "--electric-field", nargs=3, type=float, default=(0.01, -0.02, 0.03)
    )
    run.add_argument("--timeout", type=float, default=3600.0)
    capacity = subparsers.add_parser("capacity")
    capacity.add_argument("--model", type=Path, required=True)
    capacity.add_argument("--output", type=Path, required=True)
    capacity.add_argument(
        "--policy",
        choices=("optimized", "optimized_recompute"),
        default="optimized",
    )
    capacity.add_argument("--lower-repeat", type=int, default=30)
    capacity.add_argument("--upper-repeat", type=int, default=40)
    capacity.add_argument("--max-repeat", type=int, default=64)
    capacity.add_argument("--boundary-trials", type=int, default=2)
    capacity.add_argument("--warmups", type=int, default=0)
    capacity.add_argument("--samples", type=int, default=1)
    capacity.add_argument(
        "--m1-tile-channels", type=int, choices=(8, 16, 32), default=32
    )
    capacity.add_argument(
        "--response",
        choices=("none", "polarizability", "becs", "both"),
        default="none",
    )
    capacity.add_argument(
        "--response-routing",
        choices=("cached-host", "public-native"),
        default="cached-host",
    )
    capacity.add_argument(
        "--edge-geometry-policy",
        choices=("cartesian-f64-v1", "unit-f32-radius-f64-v1"),
        default="cartesian-f64-v1",
    )
    capacity.add_argument(
        "--electric-field", nargs=3, type=float, default=(0.01, -0.02, 0.03)
    )
    capacity.add_argument("--timeout", type=float, default=3600.0)
    parity = subparsers.add_parser("parity")
    parity.add_argument("--checkpoint", type=Path, required=True)
    parity.add_argument("--model", type=Path, required=True)
    parity.add_argument("--output", type=Path, required=True)
    parity.add_argument("--head", default="Default")
    parity.add_argument(
        "--electric-field", nargs=3, type=float, default=(0.013, -0.021, 0.008)
    )
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "worker":
        _worker(args)
    elif args.command == "run":
        _run(args)
    elif args.command == "capacity":
        _capacity(args)
    else:
        _parity(args)


if __name__ == "__main__":
    main()
