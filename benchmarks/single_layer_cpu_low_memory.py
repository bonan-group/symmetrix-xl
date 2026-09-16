"""Benchmark single-layer MACE low-memory execution on one CPU process."""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import time

THREADS = os.environ.get("KOKKOS_NUM_THREADS", "1")
for variable in ("KOKKOS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = THREADS
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"


def _bootstrap_explicit_symmetrix():
    source_root = pathlib.Path(os.environ["SYMMETRIX_SOURCE_ROOT"]).resolve()
    extension = pathlib.Path(os.environ["SYMMETRIX_EXTENSION"]).resolve()
    package_dir = source_root / "symmetrix/source/symmetrix"
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
        "symmetrix._native_cpu", extension
    )
    if package_spec is None or package_spec.loader is None:
        raise RuntimeError("could not load the requested Symmetrix package")
    if native_spec is None or native_spec.loader is None:
        raise RuntimeError("could not load the requested Symmetrix extension")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["symmetrix"] = package
    package_spec.loader.exec_module(package)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules["symmetrix._native_cpu"] = native
    native_spec.loader.exec_module(native)
    from symmetrix import backend_loader

    backend_loader._native_module = native
    backend_loader._selected = backend_loader._cpu_descriptor(package.__version__)
    sys.modules["symmetrix.symmetrix"] = native


_bootstrap_explicit_symmetrix()

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from symmetrix import Symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402


def _value(instance, name):
    value = getattr(instance, name)
    return value() if callable(value) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--r0-executor", choices=("v2_receiver", "v2_edge16", "v2_edge32")
    )
    parser.add_argument(
        "--harmonic-storage-policy", choices=("retained", "y-only-direct-v1")
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
        (args.repeat, args.repeat, args.repeat)
    )
    calculator = Symmetrix(
        args.model,
        use_kokkos=True,
        dtype=args.dtype,
        streamed_edges="direct",
        low_memory=True,
        neighbor_skin=args.skin,
    )
    if args.r0_executor is not None:
        calculator.evaluator._set_standard_r0_executor(args.r0_executor)
    if args.harmonic_storage_policy is not None:
        calculator.evaluator._set_harmonic_storage_policy(args.harmonic_storage_policy)
    atoms.calc = calculator
    calculator._cached_neighbor_geometry(atoms, native_geometry=True)
    directed_edges = int(calculator._neighbor_cache.receivers.size)
    for _ in range(args.warmups):
        calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    samples_ms = []
    for _ in range(args.samples):
        started = time.perf_counter_ns()
        calculator.calculate(atoms, properties=["energy", "forces", "stress"])
        samples_ms.append((time.perf_counter_ns() - started) / 1.0e6)

    evaluator = calculator.evaluator
    extension = pathlib.Path(native_symmetrix.__file__).resolve()
    median_ms = statistics.median(samples_ms)
    record = {
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "extension": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "execution_space": native_symmetrix._kokkos_default_execution_space(),
        "threads": int(THREADS),
        "precision": args.dtype,
        "atoms": len(atoms),
        "model_cutoff_A": float(calculator.cutoff),
        "neighbor_skin_A": float(calculator.neighbor_skin),
        "effective_cutoff_A": float(calculator.cutoff + calculator.neighbor_skin),
        "directed_edges": directed_edges,
        "m0_implementation": _value(evaluator, "m0_implementation"),
        "m0_host_plugin_ready": _value(evaluator, "m0_host_plugin_ready"),
        "m0_host_plugin_artifact_id": _value(evaluator, "m0_host_plugin_artifact_id"),
        "r0_implementation": _value(evaluator, "r0_implementation"),
        "standard_r0_selected_executor": _value(
            evaluator, "standard_r0_selected_executor"
        ),
        "harmonic_storage_policy": _value(evaluator, "harmonic_storage_policy"),
        "harmonic_storage_fallback_reason": _value(
            evaluator, "harmonic_storage_fallback_reason"
        ),
        "harmonic_value_bytes": _value(evaluator, "harmonic_value_bytes"),
        "harmonic_gradient_bytes": _value(evaluator, "harmonic_gradient_bytes"),
        "shuffled_coordinate_bytes": _value(evaluator, "shuffled_coordinate_bytes"),
        "factorized_arena_workspace_bytes": _value(
            evaluator, "factorized_arena_workspace_bytes"
        ),
        "mh0_reused_state_bytes": _value(evaluator, "mh0_reused_state_bytes"),
        "mh0_auxiliary_state_bytes": _value(evaluator, "mh0_auxiliary_state_bytes"),
        "h1_m0_adjoint_ping_pong_active": _value(
            evaluator, "h1_m0_adjoint_ping_pong_active"
        ),
        "h1_m0_adjoint_ping_pong_bytes": _value(
            evaluator, "h1_m0_adjoint_ping_pong_bytes"
        ),
        "operator_modules": calculator.jit_operator_modules,
        "warmups": args.warmups,
        "samples": args.samples,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "median_us_per_atom": 1000.0 * median_ms / len(atoms),
        "energy_eV": float(calculator.results["energy"]),
        "forces_eV_per_A": np.asarray(calculator.results["forces"]).tolist(),
        "stress_eV_per_A3": np.asarray(calculator.results["stress"]).tolist(),
    }
    output = json.dumps(record, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
