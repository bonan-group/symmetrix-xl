#!/usr/bin/env python3
"""Fresh-process Factorized JIT/module benchmark for MACEField and its MH-0 base."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _bootstrap_explicit_symmetrix():
    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension = os.environ.get("SYMMETRIX_EXTENSION")
    if source_root is None and extension is None:
        return
    if source_root is None or extension is None:
        raise RuntimeError(
            "SYMMETRIX_SOURCE_ROOT and SYMMETRIX_EXTENSION must be set together"
        )
    package_dir = Path(source_root).resolve() / "symmetrix/source/symmetrix"
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
        "symmetrix.symmetrix", extension_path
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError(f"invalid Symmetrix source root: {source_root}")
    if native_spec is None or native_spec.loader is None:
        raise RuntimeError(f"invalid Symmetrix extension: {extension}")
    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules["symmetrix.symmetrix"] = native
    native_spec.loader.exec_module(native)
    package_spec.loader.exec_module(package)


for variable in ("KOKKOS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

_bootstrap_explicit_symmetrix()

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from symmetrix import Symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402


ATOM_COUNT_TO_REPEAT = {
    864: 6,
    4000: 10,
    6912: 12,
    10976: 14,
    32000: 20,
    42592: 22,
    108000: 30,
    119164: 31,
    131072: 32,
    143748: 33,
    157216: 34,
    171500: 35,
    186624: 36,
    202612: 37,
    219488: 38,
    237276: 39,
    256000: 40,
    275684: 41,
    296352: 42,
    318028: 43,
    340736: 44,
    364500: 45,
    389344: 46,
    415292: 47,
    442368: 48,
    470596: 49,
    500000: 50,
}
RESULT_PREFIX = "SYMMETRIX_BENCHMARK_RESULT="


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_metric(evaluator, name, default=None):
    value = getattr(evaluator, name, default)
    return value() if callable(value) else value


def _gpu_process_memory_mib():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    total = 0
    found = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            total += int(fields[1])
            found = True
    return total if found else 0


def _process_memory_mib():
    values = {"current": None, "peak": None}
    try:
        with Path("/proc/self/status").open(encoding="ascii") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key == "VmRSS":
                    values["current"] = int(value.split()[0]) / 1024.0
                elif key == "VmHWM":
                    values["peak"] = int(value.split()[0]) / 1024.0
    except (OSError, ValueError):
        pass
    return values


def _summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
        "samples_ms": samples,
    }


def _worker(args):
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space != "Cuda":
        raise RuntimeError(f"benchmark requires Kokkos Cuda, got {execution_space}")

    setup_started = time.perf_counter()
    calculator = Symmetrix(
        args.model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="factorized",
        jit=(
            "off"
            if args.profile == "generic_r1"
            else "jit_only"
            if args.profile in ("jit_r1", "jit_optimized")
            else "required"
        ),
        m1_polynomial_policy="retained",
    )
    setup_ms = 1000.0 * (time.perf_counter() - setup_started)
    evaluator = calculator.evaluator
    expected_field = args.kind == "macefield"
    if bool(evaluator.has_field_coupling) != expected_field:
        raise RuntimeError(
            f"{args.kind} field-coupling mismatch: {evaluator.has_field_coupling}"
        )
    if args.profile in ("generic_r1", "jit_r1", "matched_r1", "retained"):
        evaluator._set_standard_r0_executor("v1")
        evaluator._set_standard_m0_executor("runtime")
    elif args.profile == "m0_only":
        evaluator._set_standard_r0_executor("v1")
        evaluator._set_standard_m0_executor("standard")
    elif args.profile == "r0_only":
        evaluator._set_standard_r0_executor("automatic")
        evaluator._set_standard_m0_executor("runtime")
    elif args.profile == "jit_optimized":
        evaluator._set_standard_r0_executor("automatic")
        evaluator._set_standard_m0_executor("standard")
        evaluator._set_m1_recompute_tile_channels(32)
        evaluator._set_m1_polynomial_policy("recompute")

    repeat = ATOM_COUNT_TO_REPEAT[args.atoms]
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
    graph_started = time.perf_counter()
    inputs = calculator._mace_inputs(atoms)
    graph_token = evaluator._prepare_factorized_graph(*inputs[:5])
    xyz = np.ascontiguousarray(inputs[5], dtype=float).reshape(-1)
    distances = np.ascontiguousarray(inputs[6], dtype=float)
    field = np.ascontiguousarray(args.electric_field, dtype=float)
    graph_setup_ms = 1000.0 * (time.perf_counter() - graph_started)

    if expected_field:

        def evaluate():
            evaluator._compute_prepared_factorized_field(
                graph_token, xyz, distances, field
            )
    else:

        def evaluate():
            evaluator._compute_prepared_factorized(graph_token, xyz, distances)

    for _ in range(args.warmups):
        evaluate()

    counter_names = (
        "jit_forward_launch_count",
        "jit_reverse_launch_count",
        "factorized_fallback_evaluation_count",
        "factorized_evaluation_fence_count",
        "standard_r0_module_launch_count",
        "standard_m0_module_forward_launch_count",
        "standard_m0_module_reverse_launch_count",
    )
    before = {name: int(_native_metric(evaluator, name, 0)) for name in counter_names}
    samples = []
    for _ in range(args.samples):
        started = time.perf_counter_ns()
        evaluate()
        samples.append((time.perf_counter_ns() - started) / 1.0e6)
    after = {name: int(_native_metric(evaluator, name, 0)) for name in counter_names}
    deltas = {name: after[name] - before[name] for name in counter_names}
    expected_r1_forward_launches = 0 if args.profile == "generic_r1" else args.samples
    expected_r1_reverse_launches = 0 if args.profile == "generic_r1" else args.samples
    if deltas["jit_forward_launch_count"] != expected_r1_forward_launches:
        raise RuntimeError(f"JIT R1 forward was not used: {deltas}")
    if deltas["jit_reverse_launch_count"] != expected_r1_reverse_launches:
        raise RuntimeError(f"JIT R1 reverse was not used: {deltas}")
    if deltas["factorized_fallback_evaluation_count"] != 0:
        raise RuntimeError(f"factorized fallback occurred during measurement: {deltas}")
    expected_m0 = (
        args.samples if args.profile in ("optimized", "jit_optimized", "m0_only") else 0
    )
    if args.profile in ("optimized", "jit_optimized", "r0_only"):
        launches_per_evaluation = 2 if evaluator.first_interaction_residual else 4
        expected_r0 = launches_per_evaluation * args.samples
    else:
        expected_r0 = 0
    if deltas["standard_m0_module_forward_launch_count"] != expected_m0:
        raise RuntimeError(f"unexpected standard M0 forward launches: {deltas}")
    if deltas["standard_m0_module_reverse_launch_count"] != expected_m0:
        raise RuntimeError(f"unexpected standard M0 reverse launches: {deltas}")
    if deltas["standard_r0_module_launch_count"] != expected_r0:
        raise RuntimeError(f"unexpected standard R0-v2 launches: {deltas}")

    node_energies = np.asarray(evaluator.node_energies, dtype=np.float64)
    node_forces = np.asarray(evaluator.node_forces, dtype=np.float64)
    result = {
        "kind": args.kind,
        "profile": args.profile,
        "model": str(args.model.resolve()),
        "execution_space": execution_space,
        "dtype": "float32",
        "atoms": len(atoms),
        "directed_edges": len(inputs[3]),
        "model_cutoff_A": float(calculator.cutoff),
        "neighbor_skin_A": float(calculator.neighbor_skin),
        "neighbor_list_cutoff_A": float(calculator.cutoff + calculator.neighbor_skin),
        "electric_field_V_per_A": field.tolist() if expected_field else None,
        "warmups": args.warmups,
        "samples": args.samples,
        "setup_ms": setup_ms,
        "graph_setup_ms": graph_setup_ms,
        "timing": _summary(samples),
        "throughput_atoms_per_second": (
            len(atoms) * 1000.0 / statistics.median(samples)
        ),
        "memory": {
            "gpu_process_mib": _gpu_process_memory_mib(),
            "process_mib": _process_memory_mib(),
            "r0_storage_elements": int(_native_metric(evaluator, "R0_storage_size", 0)),
            "r1_storage_elements": int(_native_metric(evaluator, "R1_storage_size", 0)),
            "r1_workspace_bytes": int(
                _native_metric(evaluator, "factorized_workspace_bytes", 0)
            ),
            "r1_workspace_capacity_bytes": int(
                _native_metric(evaluator, "factorized_workspace_capacity_bytes", 0)
            ),
            "geometry_workspace_bytes": int(
                _native_metric(evaluator, "execution_geometry_workspace_bytes", 0)
            ),
            "schedule_bytes": int(
                _native_metric(evaluator, "factorized_schedule_bytes", 0)
            ),
            "m0_poly_values_active_bytes": int(
                _native_metric(evaluator, "standard_m0_poly_values_active_bytes", 0)
            ),
            "m0_poly_values_capacity_bytes": int(
                _native_metric(evaluator, "standard_m0_poly_values_capacity_bytes", 0)
            ),
            "m0_poly_adjoints_active_bytes": int(
                _native_metric(evaluator, "standard_m0_poly_adjoints_active_bytes", 0)
            ),
            "m0_poly_adjoints_capacity_bytes": int(
                _native_metric(evaluator, "standard_m0_poly_adjoints_capacity_bytes", 0)
            ),
            "m1_poly_values_active_bytes": int(
                _native_metric(evaluator, "m1_poly_values_active_bytes", 0)
            ),
            "m1_poly_values_capacity_bytes": int(
                _native_metric(evaluator, "m1_poly_values_capacity_bytes", 0)
            ),
            "m1_poly_adjoints_active_bytes": int(
                _native_metric(evaluator, "m1_poly_adjoints_active_bytes", 0)
            ),
            "m1_poly_adjoints_capacity_bytes": int(
                _native_metric(evaluator, "m1_poly_adjoints_capacity_bytes", 0)
            ),
            "r0_density_workspace_bytes": int(
                _native_metric(evaluator, "standard_r0_density_workspace_bytes", 0)
            ),
        },
        "m1": {
            "requested_policy": _native_metric(
                evaluator, "m1_polynomial_policy_request"
            ),
            "selected_policy": _native_metric(evaluator, "m1_polynomial_policy"),
            "backend": _native_metric(evaluator, "m1_recompute_backend"),
            "tile_channels": int(
                _native_metric(evaluator, "m1_recompute_tile_channels", 0)
            ),
            "scratch_bytes": int(
                _native_metric(evaluator, "m1_recompute_scratch_bytes", 0)
            ),
            "scratch_limit_bytes": int(
                _native_metric(evaluator, "m1_recompute_scratch_limit_bytes", 0)
            ),
            "response_tile_channels": int(
                _native_metric(
                    evaluator, "macefield_response_m1_recompute_tile_channels", 0
                )
            ),
            "response_scratch_bytes": int(
                _native_metric(
                    evaluator, "macefield_response_m1_recompute_scratch_bytes", 0
                )
            ),
            "response_scratch_limit_bytes": int(
                _native_metric(
                    evaluator,
                    "macefield_response_m1_recompute_scratch_limit_bytes",
                    0,
                )
            ),
            "fallback_reason": _native_metric(
                evaluator, "m1_recompute_fallback_reason", ""
            ),
        },
        "factorized_implementations": {
            "jit_status": calculator.jit_status,
            "jit_artifact_id": calculator.jit_artifact_id,
            "cuda_plugin_ready": bool(
                _native_metric(evaluator, "jit_cuda_plugin_ready", False)
            ),
            "r1_jit_ready": bool(_native_metric(evaluator, "jit_ready", False)),
            "selected_forward": _native_metric(
                evaluator, "factorized_selected_direct_forward_executor"
            ),
            "selected_reverse": _native_metric(
                evaluator, "factorized_selected_direct_reverse_executor"
            ),
            "r0_selected": _native_metric(evaluator, "standard_r0_selected_executor"),
            "m0_selected": _native_metric(evaluator, "standard_m0_selected_executor"),
            "r0_module_id": _native_metric(evaluator, "standard_r0_module_id"),
            "r0_structure_fingerprint": _native_metric(
                evaluator, "standard_r0_model_structure_fingerprint"
            ),
            "m0_module_id": _native_metric(evaluator, "standard_m0_module_id"),
            "m0_structure_fingerprint": _native_metric(
                evaluator, "standard_m0_model_structure_fingerprint"
            ),
            "measured_counter_deltas": deltas,
        },
        "result_summary": {
            "energy_eV": float(node_energies.sum()),
            "force_l2_eV_per_A": float(np.linalg.norm(node_forces)),
            "force_max_abs_eV_per_A": float(np.max(np.abs(node_forces))),
            "electric_field_adjoint": (
                np.asarray(evaluator.electric_field_adj, dtype=np.float64).tolist()
                if expected_field
                else None
            ),
        },
    }
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))


def _parse_worker_result(stdout):
    records = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(f"expected one worker result, found {len(records)}")
    return records[0]


def _provenance(path):
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _gpu_info():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run(args):
    source_root = args.source_root.resolve()
    extension = args.extension.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = output.parent / (output.stem + "-logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    jit_cache = args.jit_cache.resolve()
    jit_cache.mkdir(parents=True, exist_ok=True)
    models = {"macefield": args.macefield.resolve(), "mh0": args.mh0.resolve()}
    env = os.environ.copy()
    env.update(
        {
            "SYMMETRIX_SOURCE_ROOT": str(source_root),
            "SYMMETRIX_EXTENSION": str(extension),
            "SYMMETRIX_JIT_CACHE": str(jit_cache),
            "SYMMETRIX_JIT_NVCC": args.nvcc,
            "MPLCONFIGDIR": str(output.parent / "matplotlib-cache"),
        }
    )
    records = []
    script = Path(__file__).resolve()
    for profile in args.profiles:
        for atom_count in args.sizes:
            for trial in range(args.trials):
                order = ("macefield", "mh0") if trial % 2 == 0 else ("mh0", "macefield")
                for kind in order:
                    command = [
                        sys.executable,
                        str(script),
                        "worker",
                        "--kind",
                        kind,
                        "--profile",
                        profile,
                        "--model",
                        str(models[kind]),
                        "--atoms",
                        str(atom_count),
                        "--warmups",
                        str(args.warmups),
                        "--samples",
                        str(args.samples),
                        "--electric-field",
                        *[str(value) for value in args.electric_field],
                    ]
                    completed = subprocess.run(
                        command,
                        cwd=source_root,
                        env=env,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                    )
                    stem = f"{profile}-n{atom_count}-{trial + 1:02d}-{kind}"
                    (logs_dir / f"{stem}.stdout.log").write_text(completed.stdout)
                    (logs_dir / f"{stem}.stderr.log").write_text(completed.stderr)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"worker {stem} exited {completed.returncode}; "
                            f"see {logs_dir / (stem + '.stderr.log')}"
                        )
                    record = _parse_worker_result(completed.stdout)
                    record["trial"] = trial + 1
                    record["order_in_trial"] = order.index(kind) + 1
                    records.append(record)
                    print(
                        f"{stem}: {record['timing']['median_ms']:.6f} ms",
                        flush=True,
                    )

    comparisons = []
    for profile in args.profiles:
        for atom_count in args.sizes:
            grouped = {
                kind: [
                    record["timing"]["median_ms"]
                    for record in records
                    if record["profile"] == profile
                    and record["atoms"] == atom_count
                    and record["kind"] == kind
                ]
                for kind in models
            }
            field_median = statistics.median(grouped["macefield"])
            mh0_median = statistics.median(grouped["mh0"])
            comparisons.append(
                {
                    "profile": profile,
                    "atoms": atom_count,
                    "macefield_process_medians_ms": grouped["macefield"],
                    "mh0_process_medians_ms": grouped["mh0"],
                    "macefield_median_ms": field_median,
                    "mh0_median_ms": mh0_median,
                    "macefield_over_mh0_ratio": field_median / mh0_median,
                    "macefield_overhead_percent": 100.0
                    * (field_median / mh0_median - 1.0),
                }
            )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "benchmark": "Factorized JIT/module MACEField versus standard MH-0",
        "git_commit": git_commit,
        "source_root": str(source_root),
        "extension": _provenance(extension),
        "models": {kind: _provenance(path) for kind, path in models.items()},
        "gpu": _gpu_info(),
        "dtype": "float32",
        "mode": "factorized",
        "profiles": args.profiles,
        "jit_policies": {
            "generic_r1": "off",
            "jit_r1": "jit_only",
            "jit_optimized": "jit_only",
            "other_profiles": "required",
        },
        "electric_field_V_per_A": args.electric_field,
        "warmups_per_worker": args.warmups,
        "samples_per_worker": args.samples,
        "fresh_process_trials": args.trials,
        "comparisons": comparisons,
        "records": records,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "comparisons": comparisons}, indent=2))


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--kind", choices=("macefield", "mh0"), required=True)
    worker.add_argument(
        "--profile",
        choices=(
            "optimized",
            "generic_r1",
            "jit_r1",
            "jit_optimized",
            "matched_r1",
            "retained",
            "m0_only",
            "r0_only",
        ),
        required=True,
    )
    worker.add_argument("--model", type=Path, required=True)
    worker.add_argument(
        "--atoms", type=int, choices=tuple(ATOM_COUNT_TO_REPEAT), required=True
    )
    worker.add_argument("--warmups", type=int, default=10)
    worker.add_argument("--samples", type=int, default=30)
    worker.add_argument(
        "--electric-field", nargs=3, type=float, default=(0.01, -0.02, 0.03)
    )

    run = subparsers.add_parser("run")
    run.add_argument("--macefield", type=Path, required=True)
    run.add_argument("--mh0", type=Path, required=True)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--extension", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--jit-cache", type=Path, required=True)
    run.add_argument("--nvcc", default="/usr/local/cuda-13.3/bin/nvcc")
    run.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        choices=tuple(ATOM_COUNT_TO_REPEAT),
        default=[864, 4000],
    )
    run.add_argument(
        "--profiles",
        nargs="+",
        choices=(
            "optimized",
            "generic_r1",
            "jit_r1",
            "jit_optimized",
            "matched_r1",
            "retained",
            "m0_only",
            "r0_only",
        ),
        default=["retained", "m0_only", "r0_only", "optimized"],
    )
    run.add_argument("--trials", type=int, default=3)
    run.add_argument("--warmups", type=int, default=10)
    run.add_argument("--samples", type=int, default=30)
    run.add_argument(
        "--electric-field", nargs=3, type=float, default=(0.01, -0.02, 0.03)
    )
    run.add_argument("--timeout", type=float, default=900.0)
    return parser


def main():
    args = _parser().parse_args()
    if args.warmups < 0 or args.samples < 1:
        raise SystemExit("warmups must be nonnegative and samples must be positive")
    if args.command == "worker":
        _worker(args)
    else:
        if args.trials < 1:
            raise SystemExit("trials must be positive")
        _run(args)


if __name__ == "__main__":
    main()
