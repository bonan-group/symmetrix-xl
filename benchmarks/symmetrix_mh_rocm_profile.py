"""Profile prepared Symmetrix MH0 or MH1 energy-and-forces steps on ROCm."""

import argparse
import contextlib
import ctypes
import ctypes.util
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
import time


def _load_graph_fingerprint():
    path = pathlib.Path(__file__).resolve().with_name("profile_graph_fingerprint.py")
    specification = importlib.util.spec_from_file_location(
        "symmetrix_profile_graph_fingerprint", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load graph fingerprint helper: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.canonical_graph_fingerprint, module.require_graph_fingerprint


canonical_graph_fingerprint, require_graph_fingerprint = _load_graph_fingerprint()

_EXPECTED_MODEL_TYPES = {"mh0": "MACE", "mh1": "MACE_Nonlinear"}
_EXPECTED_MODEL_SHA256 = {
    "mh0": "c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22",
    "mh1": "fb1dc908fd0f7b99aa84279dd5ae598dbde62cb15b95c2b076581872b05d6917",
}
_NUMERICAL_REFERENCES = {
    "mh0": {
        "energy_eV": -6407.773895271728,
        "force_l2_eV_per_A": 2.499132204539643,
        "force_max_abs_eV_per_A": 0.08503007467414303,
    },
    "mh1": {
        "energy_eV": -6423.654872930765,
        "force_l2_eV_per_A": 3.1685282910681147,
        "force_max_abs_eV_per_A": 0.1077980116940902,
    },
}
_NUMERICAL_ABSOLUTE_TOLERANCES = {
    "energy_eV": 0.02,
    "force_l2_eV_per_A": 0.002,
    "force_max_abs_eV_per_A": 0.0002,
}
_EXPECTED_ATOMS = 864
_EXPECTED_DIRECTED_EDGES = 78624
_EXPECTED_GRAPH_FINGERPRINT = {
    "sha256": "a6990939929e747e577bbd8b65c8954a13f712b972e40cad9869f78f54a521b7",
    "input_order_edge_records_sha256": (
        "ef74d9e1e6030a13983bd5789c90cd7f12ff2af2ed1ba55e68f5d567722fd415"
    ),
}
_ROCTX_RANGE = "symmetrix:direct:prepared_energy_forces"

_GENERATED_LAUNCH_COUNTER_ATTRIBUTES = (
    "execution_mh1_generated_forward_launch_count",
    "execution_mh1_generated_source_reverse_launch_count",
    "execution_mh1_generated_edge_reverse_launch_count",
    "execution_mh1_generated_conditioning_forward_launch_count",
    "execution_mh1_generated_conditioning_reverse_launch_count",
)
_TRANSFER_COUNTER_ATTRIBUTES = (
    "execution_geometry_allocation_count",
    "execution_geometry_copy_count",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_symmetrix(extension, source_root):
    if extension is None:
        import symmetrix

        return symmetrix

    package_dir = source_root / "symmetrix" / "source" / "symmetrix"
    package_init = package_dir / "__init__.py"
    if not package_init.is_file():
        raise RuntimeError(f"Symmetrix package is missing: {package_init}")

    package_spec = importlib.util.spec_from_file_location(
        "symmetrix",
        package_init,
        submodule_search_locations=[str(package_dir)],
    )
    native_spec = importlib.util.spec_from_file_location(
        "symmetrix.symmetrix", extension
    )
    if (
        package_spec is None
        or package_spec.loader is None
        or native_spec is None
        or native_spec.loader is None
    ):
        raise RuntimeError("could not construct import specifications")

    package = importlib.util.module_from_spec(package_spec)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix"] = package
    sys.modules["symmetrix.symmetrix"] = native
    native_spec.loader.exec_module(native)
    package_spec.loader.exec_module(package)
    return package


def _load_roctx_sdk():
    candidates = [
        "/opt/rocm/core-7.14/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib64/librocprofiler-sdk-roctx.so",
        ctypes.util.find_library("rocprofiler-sdk-roctx"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        required = (
            "roctxProfilerResume",
            "roctxRangePushA",
            "roctxRangePop",
            "roctxProfilerPause",
        )
        if not all(hasattr(library, symbol) for symbol in required):
            continue
        library.roctxProfilerResume.argtypes = [ctypes.c_uint64]
        library.roctxProfilerResume.restype = ctypes.c_int
        library.roctxRangePushA.argtypes = [ctypes.c_char_p]
        library.roctxRangePushA.restype = ctypes.c_int
        library.roctxRangePop.argtypes = []
        library.roctxRangePop.restype = ctypes.c_int
        library.roctxProfilerPause.argtypes = [ctypes.c_uint64]
        library.roctxProfilerPause.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("rocprofiler-sdk ROCTx library is unavailable")


def _load_hip_runtime():
    candidates = [ctypes.util.find_library("amdhip64"), "libamdhip64.so.7"]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            library = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        library.hipDeviceSynchronize.argtypes = []
        library.hipDeviceSynchronize.restype = ctypes.c_int
        return library, candidate
    raise RuntimeError("HIP runtime library is unavailable")


@contextlib.contextmanager
def _selected_roctx_range(library, name):
    result = library.roctxProfilerResume(0)
    if result != 0:
        raise RuntimeError(f"roctxProfilerResume failed with status {result}")

    level = library.roctxRangePushA(name.encode("ascii"))
    if level < 0:
        pause_result = library.roctxProfilerPause(0)
        if pause_result != 0:
            raise RuntimeError(
                f"roctxRangePushA failed with level {level}; "
                f"roctxProfilerPause also failed with status {pause_result}"
            )
        raise RuntimeError(f"roctxRangePushA failed with level {level}")

    try:
        yield
    finally:
        popped_level = library.roctxRangePop()
        pause_result = library.roctxProfilerPause(0)
        if popped_level < 0:
            raise RuntimeError(f"roctxRangePop failed with level {popped_level}")
        if pause_result != 0:
            raise RuntimeError(f"roctxProfilerPause failed with status {pause_result}")


def _pause_profiler(library):
    result = library.roctxProfilerPause(0)
    if result != 0:
        raise RuntimeError(f"initial roctxProfilerPause failed with status {result}")


def _validate_numerics(role, energy, forces, np):
    actual = {
        "energy_eV": energy,
        "force_l2_eV_per_A": float(np.linalg.norm(forces)),
        "force_max_abs_eV_per_A": float(np.max(np.abs(forces))),
    }
    expected = _NUMERICAL_REFERENCES[role]
    errors = {name: abs(actual[name] - expected[name]) for name in expected}
    passed = all(
        errors[name] <= _NUMERICAL_ABSOLUTE_TOLERANCES[name] for name in expected
    )
    validation = {
        "passed": passed,
        "reference": expected,
        "actual": actual,
        "absolute_error": errors,
        "absolute_tolerance": _NUMERICAL_ABSOLUTE_TOLERANCES,
        "scope": "diagnostic scalar checks; full-array qualification is separate",
    }
    if not passed:
        raise RuntimeError(f"profile numerical validation failed: {validation}")
    return validation


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile prepared hipRTC/FP32 Symmetrix energy-and-forces steps "
            "for the 864-atom AlN MH0 or MH1 workload."
        )
    )
    parser.add_argument("role", choices=tuple(_EXPECTED_MODEL_TYPES))
    parser.add_argument("model", type=pathlib.Path, help="Extracted model JSON")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="enable the complete low-memory execution bundle",
    )
    parser.add_argument(
        "--phi1-policy",
        choices=("retained", "channel-tiled-64", "receiver-local"),
        default="retained",
    )
    parser.add_argument(
        "--mh1-edge-executor",
        choices=("mlp_reference", "pair_spline_v1"),
        default="mlp_reference",
        help="Internal MH1 R-stage executor",
    )
    parser.add_argument(
        "--node-state-policy",
        dest="execution_mh1_node_state_policy",
        choices=(
            "full-retention-v1",
            "recompute-v1",
            "reuse-adjoints-v1",
            "retain-interaction-v1",
        ),
        default="full-retention-v1",
        help="Generated MH1 node-state storage policy",
    )
    parser.add_argument(
        "--extension",
        type=pathlib.Path,
        help="Freshly built native extension; requires --source-root",
    )
    parser.add_argument(
        "--source-root",
        type=pathlib.Path,
        help="Repository root used with --extension",
    )
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if (args.extension is None) != (args.source_root is None):
        parser.error("--extension and --source-root must be supplied together")
    if (
        args.role != "mh1"
        and args.execution_mh1_node_state_policy != "full-retention-v1"
    ):
        parser.error("--node-state-policy=recompute-v1 is only valid for MH1")
    if args.role != "mh1" and args.mh1_edge_executor != "mlp_reference":
        parser.error("--mh1-edge-executor=pair_spline_v1 is only valid for MH1")
    return parser, args


def _jit_metadata(calculator):
    return {
        "policy": getattr(calculator, "jit_policy", None),
        "status": getattr(calculator, "jit_status", None),
        "reason": getattr(calculator, "jit_reason", None),
        "compiler_request": getattr(calculator, "jit_compiler_request", None),
        "compiler_backend": getattr(calculator, "jit_compiler_backend", None),
        "cache_key": getattr(calculator, "jit_cache_key", None),
        "artifact_path": getattr(calculator, "jit_artifact_path", None),
        "artifact_id": getattr(calculator, "jit_artifact_id", None),
        "variant_id": getattr(calculator, "jit_variant_id", None),
        "forward_policy": getattr(calculator, "jit_forward_policy", None),
        "source_policy": getattr(calculator, "jit_source_policy", None),
        "edge_policy": getattr(calculator, "jit_edge_policy", None),
        "node_state_policy": getattr(calculator, "jit_node_state_policy", None),
        "node_state_policy_request": getattr(
            calculator, "execution_mh1_node_state_policy", None
        ),
    }


def _optional_attribute(instance, name):
    try:
        value = getattr(instance, name)
    except AttributeError:
        return None
    return {
        "availability": "available",
        "source_attribute": name,
        "value": value,
    }


def _optional_first_attribute(instance, names):
    for name in names:
        result = _optional_attribute(instance, name)
        if result is not None:
            return result
    return {
        "availability": "not_available",
        "source_attribute": None,
        "value": None,
    }


def _workspace_diagnostics(evaluator):
    precision_bytes = _optional_first_attribute(
        evaluator, ("precision_workspace_bytes",)
    )
    node_bytes = _optional_first_attribute(evaluator, ("node_workspace_bytes",))
    geometry_bytes = _optional_first_attribute(
        evaluator, ("execution_geometry_workspace_bytes",)
    )
    if (
        precision_bytes["availability"] == "available"
        and node_bytes["availability"] == "available"
        and geometry_bytes["availability"] == "available"
    ):
        total_bytes = {
            "availability": "available",
            "source_attribute": (
                "precision_workspace_bytes + node_workspace_bytes + "
                "execution_geometry_workspace_bytes"
            ),
            "value": (
                precision_bytes["value"] + node_bytes["value"] + geometry_bytes["value"]
            ),
        }
    else:
        total_bytes = _optional_first_attribute(
            evaluator,
            ("execution_unified_workspace_bytes",),
        )
    return {
        "total_bytes": total_bytes,
        "precision_bytes": precision_bytes,
        "node_bytes": node_bytes,
        "edge_bytes": _optional_first_attribute(evaluator, ("edge_workspace_bytes",)),
        "geometry_bytes": geometry_bytes,
        "graph_scratch": {
            "budget_bytes": _optional_first_attribute(
                evaluator, ("execution_mh1_scratch_budget_bytes",)
            ),
            "minimum_bytes": _optional_first_attribute(
                evaluator, ("execution_mh1_scratch_minimum_bytes",)
            ),
            "planned_bytes": _optional_first_attribute(
                evaluator, ("execution_mh1_scratch_planned_bytes",)
            ),
            "retained_bytes": _optional_first_attribute(
                evaluator, ("execution_mh1_scratch_retained_bytes",)
            ),
            "recomputed_bytes": _optional_first_attribute(
                evaluator, ("execution_mh1_scratch_recomputed_bytes",)
            ),
            "recomputed_layer_count": _optional_first_attribute(
                evaluator, ("execution_mh1_scratch_recomputed_layer_count",)
            ),
            "budget_satisfied": _optional_first_attribute(
                evaluator, ("execution_mh1_scratch_budget_satisfied",)
            ),
        },
    }


def _counter_snapshot(evaluator, attributes):
    return {
        name: (entry["value"] if entry is not None else None)
        for name in attributes
        for entry in (_optional_attribute(evaluator, name),)
    }


def _counter_diagnostics(before, after):
    diagnostics = {}
    for name in before:
        before_value = before[name]
        after_value = after[name]
        available = before_value is not None and after_value is not None
        diagnostics[name] = {
            "availability": "available" if available else "not_available",
            "before": before_value,
            "after": after_value,
            "delta": after_value - before_value if available else None,
        }
    return diagnostics


def main():
    parser, args = _parse_args()
    import numpy as np
    from ase.build import bulk

    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    extension = args.extension.expanduser().resolve() if args.extension else None
    source_root = args.source_root.expanduser().resolve() if args.source_root else None
    if not model_path.is_file():
        parser.error(f"model JSON does not exist: {model_path}")
    if extension is not None and not extension.is_file():
        parser.error(f"native extension does not exist: {extension}")
    if source_root is not None and not source_root.is_dir():
        parser.error(f"source root does not exist: {source_root}")

    model_sha256 = _sha256(model_path)
    expected_sha256 = _EXPECTED_MODEL_SHA256[args.role]
    if model_sha256 != expected_sha256:
        parser.error(
            f"{args.role} model SHA-256 is {model_sha256}, expected {expected_sha256}"
        )

    try:
        model_data = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        parser.error(f"could not read model JSON: {error}")
    model_type = model_data.get("model_type", "MACE")
    expected_model_type = _EXPECTED_MODEL_TYPES[args.role]
    if model_type != expected_model_type:
        parser.error(
            f"{args.role} requires model_type {expected_model_type!r}, got "
            f"{model_type!r}"
        )

    symmetrix_package = _load_symmetrix(extension, source_root)
    from symmetrix.calculator import Symmetrix

    roctx, roctx_library = _load_roctx_sdk()
    hip, hip_library = _load_hip_runtime()
    _pause_profiler(roctx)

    calculator = Symmetrix(
        model_path,
        dtype="float32",
        use_kokkos=True,
        streamed_edges="direct",
        low_memory=args.low_memory,
        neighbor_skin=0.0,
        execution_mh1_node_state_policy=args.execution_mh1_node_state_policy,
    )
    evaluator = calculator.evaluator
    evaluator._set_phi1_policy(args.phi1_policy)
    if args.role == "mh1":
        evaluator.set_mh1_edge_executor(args.mh1_edge_executor)
    if getattr(evaluator, "streamed_edges_mode", None) != "direct":
        raise RuntimeError("the evaluator did not select direct execution")
    if calculator.jit_policy != "required":
        raise RuntimeError(f"unexpected JIT policy: {calculator.jit_policy}")
    if calculator.jit_status not in ("built", "cached"):
        raise RuntimeError(f"unexpected JIT status: {calculator.jit_status}")
    if calculator.jit_compiler_backend != "hiprtc":
        raise RuntimeError(
            f"unexpected JIT compiler: {calculator.jit_compiler_backend}"
        )
    mh1_backend = getattr(evaluator, "execution_mh1_execution_backend", None)
    if args.role == "mh1":
        expected_mh1_backend = (
            "generated_hip_v4"
            if args.mh1_edge_executor == "mlp_reference"
            else "pair_spline_v1_staged_device_v5"
        )
        if mh1_backend != expected_mh1_backend:
            raise RuntimeError(
                f"unexpected MH1 execution backend: {mh1_backend}; "
                f"expected {expected_mh1_backend}"
            )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((6, 6, 6))
    inputs = calculator._mace_inputs(atoms)
    atom_count = len(atoms)
    directed_edges = len(inputs[6])
    if atom_count != _EXPECTED_ATOMS or directed_edges != _EXPECTED_DIRECTED_EDGES:
        raise RuntimeError(
            f"unexpected profile graph: {atom_count} atoms, "
            f"{directed_edges} directed edges"
        )
    first_indices = np.asarray(inputs[7])
    second_indices = np.asarray(inputs[3])
    displacements = np.asarray(inputs[5]).reshape((-1, 3))
    position_delta = atoms.positions[second_indices] - atoms.positions[first_indices]
    fractional_shifts = (displacements - position_delta) @ np.linalg.inv(
        atoms.cell.array
    )
    unit_shifts = np.rint(fractional_shifts)
    reconstructed = position_delta + unit_shifts @ atoms.cell.array
    if not np.allclose(displacements, reconstructed, rtol=0.0, atol=1.0e-8):
        raise RuntimeError("could not recover integral periodic shifts from the graph")
    graph_fingerprint = canonical_graph_fingerprint(
        atoms.numbers,
        atoms.positions,
        atoms.cell.array,
        first_indices,
        second_indices,
        unit_shifts,
    )
    require_graph_fingerprint(graph_fingerprint, _EXPECTED_GRAPH_FINGERPRINT)

    prepare_graph = getattr(evaluator, "_prepare_factorized_graph", None)
    compute_prepared = getattr(evaluator, "_compute_prepared_factorized", None)
    if not callable(prepare_graph) or not callable(compute_prepared):
        raise TypeError("the evaluator does not expose prepared direct execution")
    graph_generation = prepare_graph(*inputs[:5])
    node_input = np.ascontiguousarray(inputs[5]).reshape(-1)
    edge_indices = np.ascontiguousarray(inputs[6])

    def fence():
        evaluator_fence = getattr(evaluator, "fence", None)
        if callable(evaluator_fence):
            evaluator_fence()
        status = hip.hipDeviceSynchronize()
        if status != 0:
            raise RuntimeError(f"hipDeviceSynchronize failed with status {status}")

    def evaluate():
        compute_prepared(graph_generation, node_input, edge_indices)

    for _ in range(args.warmups):
        fence()
        evaluate()
        fence()

    fence()
    prepared_before = evaluator.factorized_prepared_evaluation_count
    fallback_before = evaluator.factorized_fallback_evaluation_count
    launch_counters_before = _counter_snapshot(
        evaluator, _GENERATED_LAUNCH_COUNTER_ATTRIBUTES
    )
    transfer_counters_before = _counter_snapshot(
        evaluator, _TRANSFER_COUNTER_ATTRIBUTES
    )
    elapsed_ms_samples = []
    with _selected_roctx_range(roctx, _ROCTX_RANGE):
        for _ in range(args.steps):
            start = time.perf_counter()
            evaluate()
            fence()
            elapsed_ms_samples.append(1000.0 * (time.perf_counter() - start))
    elapsed_ms = statistics.median(elapsed_ms_samples)
    prepared_after = evaluator.factorized_prepared_evaluation_count
    fallback_after = evaluator.factorized_fallback_evaluation_count
    launch_counters_after = _counter_snapshot(
        evaluator, _GENERATED_LAUNCH_COUNTER_ATTRIBUTES
    )
    transfer_counters_after = _counter_snapshot(evaluator, _TRANSFER_COUNTER_ATTRIBUTES)
    if prepared_after != prepared_before + args.steps:
        raise RuntimeError(
            "the selected region did not execute the requested prepared evaluations: "
            f"counter changed from {prepared_before} to {prepared_after}"
        )
    if fallback_after != fallback_before or fallback_after != 0:
        raise RuntimeError(
            "the selected region used the fallback path: "
            f"counter changed from {fallback_before} to {fallback_after}"
        )

    results = calculator._collect_mace_results(atoms, inputs)
    energy = float(results["energy"])
    forces = np.asarray(results["forces"])
    if forces.shape != (_EXPECTED_ATOMS, 3):
        raise RuntimeError(f"unexpected force shape: {forces.shape}")
    if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
        raise RuntimeError("profile results contain non-finite energy or forces")
    numerical_validation = _validate_numerics(args.role, energy, forces, np)

    module_path = pathlib.Path(symmetrix_package.__file__).resolve()
    native_module = sys.modules.get("symmetrix.symmetrix")
    native_module_file = getattr(native_module, "__file__", None)
    report = {
        "workload": {
            "role": args.role,
            "dtype": "float32",
            "properties": ["energy", "forces"],
            "prepared_graph": True,
        },
        "model": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": model_sha256,
            "model_type": model_type,
        },
        "graph": {
            "structure": "wurtzite AlN",
            "lattice_a_A": 3.112,
            "lattice_c_A": 4.982,
            "supercell_repeat": [6, 6, 6],
            "atoms": atom_count,
            "al_atoms": int(np.count_nonzero(atoms.numbers == 13)),
            "n_atoms": int(np.count_nonzero(atoms.numbers == 7)),
            "model_cutoff_A": float(calculator.cutoff),
            "neighbor_skin_A": float(calculator.neighbor_skin),
            "effective_neighbor_cutoff_A": float(
                calculator.cutoff + calculator.neighbor_skin
            ),
            "directed_edges": directed_edges,
            "canonical_fingerprint": graph_fingerprint,
        },
        "profile": {
            "warmups": args.warmups,
            "measured_steps": args.steps,
            "elapsed_ms": elapsed_ms,
            "elapsed_ms_samples": elapsed_ms_samples,
            "elapsed_ms_min": min(elapsed_ms_samples),
            "elapsed_ms_max": max(elapsed_ms_samples),
            "roctx_range": _ROCTX_RANGE,
            "roctx_library": str(roctx_library),
            "roctx_profiler_control": True,
        },
        "execution_path": {
            "streamed_edges": getattr(evaluator, "streamed_edges_mode", None),
            "low_memory": args.low_memory,
            "phi1_policy": getattr(evaluator, "phi1_policy", None),
            "phi1_workspace_bytes": getattr(evaluator, "phi1_workspace_bytes", None),
            "static_mh0_artifact_forced_off": args.role == "mh0",
            "mh1_backend": mh1_backend,
            "mh1_edge_executor": (
                getattr(evaluator, "mh1_edge_executor", None)
                if args.role == "mh1"
                else None
            ),
            "graph_generation": graph_generation,
            "prepared_graph_count": getattr(
                evaluator, "factorized_prepared_graph_count", None
            ),
            "prepared_evaluation_count": getattr(
                evaluator, "factorized_prepared_evaluation_count", None
            ),
            "fallback_evaluation_count": getattr(
                evaluator, "factorized_fallback_evaluation_count", None
            ),
        },
        "evaluator_diagnostics": {
            "workspace": _workspace_diagnostics(evaluator),
            "generated_launch_counters": _counter_diagnostics(
                launch_counters_before, launch_counters_after
            ),
            "measured_evaluation_transfers": _counter_diagnostics(
                transfer_counters_before, transfer_counters_after
            ),
        },
        "jit": _jit_metadata(calculator),
        "numerics": {
            "energy_eV": energy,
            "force_max_abs_eV_per_A": float(np.max(np.abs(forces))),
            "force_l2_eV_per_A": float(np.linalg.norm(forces)),
            "force_sum_eV_per_A": forces.sum(axis=0).tolist(),
            "validation": numerical_validation,
        },
        "runtime": {
            "import_mode": "dynamic_extension" if extension else "installed",
            "symmetrix_package": str(module_path),
            "native_extension": (
                str(pathlib.Path(native_module_file).resolve())
                if native_module_file is not None
                else None
            ),
            "requested_extension": str(extension) if extension else None,
            "source_root": str(source_root) if source_root else None,
            "hip_library": str(hip_library),
            "python": sys.version,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
