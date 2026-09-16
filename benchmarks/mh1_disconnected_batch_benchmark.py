"""Benchmark disconnected small-structure batches on the MH-1 GPU path."""

import argparse
import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time


def _bootstrap_explicit_symmetrix():
    source_root = os.environ.get("SYMMETRIX_SOURCE_ROOT")
    extension_path = os.environ.get("SYMMETRIX_EXTENSION")
    if not source_root or not extension_path:
        return
    import importlib.util

    package_dir = pathlib.Path(source_root).resolve() / "symmetrix/source/symmetrix"
    extension = pathlib.Path(extension_path).resolve()
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


_bootstrap_explicit_symmetrix()

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402

from symmetrix import Symmetrix  # noqa: E402
from symmetrix import symmetrix as native_symmetrix  # noqa: E402


def _load_roctx_sdk():
    candidates = (
        "/opt/rocm/core-7.14/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib64/librocprofiler-sdk-roctx.so",
        ctypes.util.find_library("rocprofiler-sdk-roctx"),
    )
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
        return library
    raise RuntimeError("rocprofiler-sdk ROCTx library is unavailable")


def _set_profiler_state(roctx, operation):
    status = operation(0)
    if status != 0:
        raise RuntimeError(f"ROCTx profiler control failed with status {status}")


@contextlib.contextmanager
def _profile_region(roctx, name):
    _set_profiler_state(roctx, roctx.roctxProfilerResume)
    level = roctx.roctxRangePushA(name.encode("ascii"))
    if level < 0:
        _set_profiler_state(roctx, roctx.roctxProfilerPause)
        raise RuntimeError(f"roctxRangePushA failed with level {level}")
    try:
        yield
    finally:
        popped = roctx.roctxRangePop()
        _set_profiler_state(roctx, roctx.roctxProfilerPause)
        if popped < 0:
            raise RuntimeError(f"roctxRangePop failed with level {popped}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _samples(samples, structures, atoms):
    median_ms = statistics.median(samples)
    return {
        "median_ms_per_group": median_ms,
        "min_ms_per_group": min(samples),
        "max_ms_per_group": max(samples),
        "us_per_structure": 1000.0 * median_ms / structures,
        "us_per_atom": 1000.0 * median_ms / atoms,
        "structures_per_s": 1000.0 * structures / median_ms,
        "atoms_per_s": 1000.0 * atoms / median_ms,
        "samples_ms": samples,
    }


def _make_structures(slots, displacement, seed):
    base = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((5, 1, 1))
    rng = np.random.default_rng(seed)
    structures = []
    for slot in range(slots):
        atoms = base.copy()
        if slot:
            positions = atoms.get_positions()
            positions += rng.normal(0.0, displacement, positions.shape)
            atoms.set_positions(positions)
        structures.append(atoms)
    return structures


def _as_graph(calculator, atoms):
    values = calculator._mace_inputs(atoms)
    return {
        "num_nodes": values[0],
        "node_types": np.ascontiguousarray(values[1], dtype=np.int32),
        "num_neigh": np.ascontiguousarray(values[2], dtype=np.int32),
        "sources": np.ascontiguousarray(values[3], dtype=np.int32),
        "neigh_types": np.ascontiguousarray(values[4], dtype=np.int32),
        "xyz": np.ascontiguousarray(values[5], dtype=np.float64).reshape(-1, 3),
        "distances": np.ascontiguousarray(values[6], dtype=np.float64),
        "receivers": np.ascontiguousarray(values[7], dtype=np.int32),
        "volume": atoms.get_volume(),
    }


def _concatenate_graphs(graphs):
    node_offsets = np.cumsum([0, *(graph["num_nodes"] for graph in graphs[:-1])])
    edge_offsets = np.cumsum([0, *(len(graph["sources"]) for graph in graphs[:-1])])
    return {
        "num_nodes": sum(graph["num_nodes"] for graph in graphs),
        "node_types": np.concatenate([graph["node_types"] for graph in graphs]),
        "num_neigh": np.concatenate([graph["num_neigh"] for graph in graphs]),
        "sources": np.concatenate(
            [graph["sources"] + offset for graph, offset in zip(graphs, node_offsets)]
        ),
        "neigh_types": np.concatenate([graph["neigh_types"] for graph in graphs]),
        "xyz": np.concatenate([graph["xyz"] for graph in graphs]),
        "distances": np.concatenate([graph["distances"] for graph in graphs]),
        "receivers": np.concatenate(
            [graph["receivers"] + offset for graph, offset in zip(graphs, node_offsets)]
        ),
        "node_offsets": node_offsets,
        "edge_offsets": edge_offsets,
    }


def _prepare(evaluator, graph):
    return evaluator._prepare_factorized_graph(
        graph["num_nodes"],
        graph["node_types"],
        graph["num_neigh"],
        graph["sources"],
        graph["neigh_types"],
    )


def _compute(evaluator, generation, graph):
    evaluator._compute_prepared_factorized(
        generation, graph["xyz"].reshape(-1), graph["distances"]
    )


def _collect_energy_forces(evaluator, generation, graph):
    node_energies = np.asarray(evaluator.node_energies, dtype=np.float64)
    forces = np.asarray(
        evaluator._reduce_atom_forces(
            graph["num_nodes"],
            graph["receivers"],
            graph["sources"],
            generation,
        ),
        dtype=np.float64,
    ).reshape(-1, 3)
    return node_energies, forces


def _collect_energy_forces_stress(evaluator, generation, graph, volumes):
    node_energies, forces = _collect_energy_forces(evaluator, generation, graph)
    stresses = np.asarray(
        evaluator._reduce_batched_stress(
            np.ascontiguousarray(volumes, dtype=np.float64), generation
        ),
        dtype=np.float64,
    ).reshape(-1, 3, 3)
    return node_energies, forces, stresses


def _reference_results(evaluator, graphs):
    reference = []
    for graph in graphs:
        generation = _prepare(evaluator, graph)
        _compute(evaluator, generation, graph)
        node_energies, forces = _collect_energy_forces(evaluator, generation, graph)
        native_stress = np.asarray(
            evaluator._reduce_stress(graph["volume"], np.empty(0), generation),
            dtype=np.float64,
        ).reshape(3, 3)
        reference.append(
            {
                "energy": float(np.sum(node_energies)),
                "forces": np.array(forces, copy=True),
                "stress": native_stress,
            }
        )
    return reference


def _batch_results(evaluator, graphs, batch):
    generation = _prepare(evaluator, batch)
    evaluator._prepare_factorized_batch(
        generation,
        np.ascontiguousarray(
            [*batch["edge_offsets"], len(batch["sources"])], dtype=np.int64
        ),
    )
    _compute(evaluator, generation, batch)
    volumes = [graph["volume"] for graph in graphs]
    node_energies, forces, stresses = _collect_energy_forces_stress(
        evaluator, generation, batch, volumes
    )
    aggregate_stress = np.asarray(
        evaluator._reduce_stress(sum(volumes), np.empty(0), generation),
        dtype=np.float64,
    ).reshape(3, 3)
    node_offsets = [*batch["node_offsets"], batch["num_nodes"]]
    return (
        generation,
        [
            {
                "energy": float(np.sum(node_energies[begin:end])),
                "forces": np.array(forces[begin:end], copy=True),
                "stress": np.array(stress, copy=True),
            }
            for begin, end, stress in zip(node_offsets[:-1], node_offsets[1:], stresses)
        ],
        aggregate_stress,
    )


def _correctness(reference, batched, aggregate_stress, volumes):
    energy_abs = [
        abs(left["energy"] - right["energy"]) for left, right in zip(reference, batched)
    ]
    force_abs = [
        float(np.max(np.abs(left["forces"] - right["forces"])))
        for left, right in zip(reference, batched)
    ]
    stress_abs = [
        float(np.max(np.abs(left["stress"] - right["stress"])))
        for left, right in zip(reference, batched)
    ]
    expected_aggregate_stress = sum(
        (volume * item["stress"] for volume, item in zip(volumes, batched)),
        start=np.zeros((3, 3)),
    ) / sum(volumes)
    return {
        "max_energy_abs_eV": max(energy_abs),
        "max_force_abs_eV_per_A": max(force_abs),
        "max_stress_abs_eV_per_A3": max(stress_abs),
        "batch_aggregate_stress_abs_eV_per_A3": float(
            np.max(np.abs(aggregate_stress - expected_aggregate_stress))
        ),
    }


def _time_operation(operation, synchronize, warmups, repeats):
    for _ in range(warmups):
        operation()
        synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        operation()
        synchronize()
        samples.append(1000.0 * (time.perf_counter() - start))
    return samples


def _profile_batch(evaluator, generation, batch, roctx, warmups, repeats, scope):
    if scope == "evaluation":

        def operation():
            _compute(evaluator, generation, batch)

    elif scope == "energy_forces_stress":
        volumes = [graph["volume"] for graph in batch["graphs"]]

        def operation():
            _compute(evaluator, generation, batch)
            _collect_energy_forces_stress(evaluator, generation, batch, volumes)

    else:
        raise ValueError(f"unknown profile scope {scope}")
    for _ in range(warmups):
        operation()
    evaluator.fence()
    start = time.perf_counter()
    with _profile_region(
        roctx, f"symmetrix:mh1:disconnected_batch:{len(batch['node_offsets'])}slots"
    ):
        for _ in range(repeats):
            operation()
        evaluator.fence()
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "evaluations": repeats,
        "elapsed_ms": elapsed_ms,
        "mean_ms_per_evaluation": elapsed_ms / repeats,
        "scope": scope,
    }


def _benchmark_scope(
    evaluator, graphs, batch, batch_generation, scope, warmups, repeats
):
    if scope == "evaluation":

        def sequential_one(graph, generation):
            _compute(evaluator, generation, graph)

        def batch_operation():
            _compute(evaluator, batch_generation, batch)

    elif scope == "energy_forces":

        def sequential_one(graph, generation):
            _compute(evaluator, generation, graph)
            _collect_energy_forces(evaluator, generation, graph)

        def batch_operation():
            _compute(evaluator, batch_generation, batch)
            _collect_energy_forces(evaluator, batch_generation, batch)

    elif scope == "energy_forces_stress":

        def sequential_one(graph, generation):
            _compute(evaluator, generation, graph)
            _collect_energy_forces_stress(
                evaluator,
                generation,
                {
                    **graph,
                    "edge_offsets": np.asarray([0], dtype=np.int64),
                },
                [graph["volume"]],
            )

        def batch_operation():
            _compute(evaluator, batch_generation, batch)
            _collect_energy_forces_stress(
                evaluator,
                batch_generation,
                batch,
                [graph["volume"] for graph in graphs],
            )

    else:
        raise ValueError(f"unknown benchmark scope {scope}")

    # All perturbed slots retain one topology, so one prepared token can drive
    # every independent geometry just as it would in fixed-slot optimization.
    sequential_generation = _prepare(evaluator, graphs[0])
    evaluator._prepare_factorized_batch(
        sequential_generation,
        np.asarray([0, len(graphs[0]["sources"])], dtype=np.int64),
    )

    def sequential_operation():
        for graph in graphs:
            sequential_one(graph, sequential_generation)

    synchronize = evaluator.fence
    sequential_samples = _time_operation(
        sequential_operation, synchronize, warmups, repeats
    )
    batch_generation = _prepare(evaluator, batch)
    evaluator._prepare_factorized_batch(
        batch_generation,
        np.ascontiguousarray(
            [*batch["edge_offsets"], len(batch["sources"])], dtype=np.int64
        ),
    )
    batch_samples = _time_operation(batch_operation, synchronize, warmups, repeats)
    total_atoms = batch["num_nodes"]
    sequential = _samples(sequential_samples, len(graphs), total_atoms)
    batched = _samples(batch_samples, len(graphs), total_atoms)
    return {
        "sequential": sequential,
        "batched": batched,
        "throughput_speedup": (
            batched["structures_per_s"] / sequential["structures_per_s"]
        ),
        "batch_over_sequential_time": (
            batched["median_ms_per_group"] / sequential["median_ms_per_group"]
        ),
    }


def _parse_slots(value):
    try:
        slots = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "slots must be comma-separated integers"
        ) from error
    if not slots or any(slot < 1 for slot in slots):
        raise argparse.ArgumentTypeError("slots must be positive")
    return slots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--slots", type=_parse_slots, default=[1, 2, 4, 8, 16])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--neighbor-skin", type=float, default=0.0)
    parser.add_argument(
        "--profile-batch-repeats",
        type=int,
        default=0,
        help="Profile only warmed batched evaluations inside one ROCTx region",
    )
    parser.add_argument(
        "--profile-batch-scope",
        choices=("evaluation", "energy_forces_stress"),
        default="evaluation",
    )
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.displacement < 0.0 or not np.isfinite(args.displacement):
        parser.error("--displacement must be finite and non-negative")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats positive")
    if args.profile_batch_repeats < 0:
        parser.error("--profile-batch-repeats must be non-negative")
    if args.profile_batch_repeats and len(args.slots) != 1:
        parser.error("--profile-batch-repeats requires exactly one slot count")

    roctx = None
    if args.profile_batch_repeats:
        roctx = _load_roctx_sdk()
        _set_profiler_state(roctx, roctx.roctxProfilerPause)

    calculator = Symmetrix(
        args.model,
        use_kokkos=True,
        dtype=args.dtype,
        streamed_edges="direct",
        neighbor_skin=args.neighbor_skin,
    )
    evaluator = calculator.evaluator
    evaluator.set_mh1_edge_executor("pair_spline_v1")
    if not evaluator.uses_mh1_fast_path:
        raise RuntimeError("model does not select the MH-1 fast path")
    if evaluator.streamed_edges_mode != "direct":
        raise RuntimeError("evaluator did not select direct streamed edges")

    records = []
    for slot_count in args.slots:
        structures = _make_structures(slot_count, args.displacement, args.seed)
        graphs = [_as_graph(calculator, atoms) for atoms in structures]
        topology = graphs[0]
        for graph in graphs[1:]:
            if not (
                np.array_equal(graph["node_types"], topology["node_types"])
                and np.array_equal(graph["num_neigh"], topology["num_neigh"])
                and np.array_equal(graph["sources"], topology["sources"])
                and np.array_equal(graph["receivers"], topology["receivers"])
            ):
                raise RuntimeError(
                    "perturbed slots changed topology; reduce --displacement"
                )
        reference = _reference_results(evaluator, graphs)
        batch = _concatenate_graphs(graphs)
        batch["graphs"] = graphs
        batch_generation, batched, aggregate_stress = _batch_results(
            evaluator, graphs, batch
        )
        correctness = _correctness(
            reference,
            batched,
            aggregate_stress,
            [graph["volume"] for graph in graphs],
        )
        tolerances = {
            "max_energy_abs_eV": 2.0e-5,
            "max_force_abs_eV_per_A": 2.0e-5,
            "max_stress_abs_eV_per_A3": 2.0e-7,
            "batch_aggregate_stress_abs_eV_per_A3": 2.0e-7,
        }
        failures = {
            name: {"value": correctness[name], "tolerance": tolerance}
            for name, tolerance in tolerances.items()
            if correctness[name] > tolerance
        }
        if failures:
            raise RuntimeError(f"disconnected batch correctness failed: {failures}")
        record = {
            "slots": slot_count,
            "atoms_per_slot": graphs[0]["num_nodes"],
            "total_atoms": batch["num_nodes"],
            "directed_edges_per_slot": len(graphs[0]["sources"]),
            "total_directed_edges": len(batch["sources"]),
            "cutoff_A": calculator.cutoff,
            "neighbor_skin_A": args.neighbor_skin,
            "effective_cutoff_A": calculator.cutoff + args.neighbor_skin,
            "correctness": correctness,
            "correctness_tolerances": tolerances,
            "native_per_slot_stress_supported": True,
            "stress_timing": (
                "segmented per-slot virial reduction on the Kokkos device"
            ),
            "scopes": {},
        }
        if args.profile_batch_repeats:
            record["profile"] = _profile_batch(
                evaluator,
                batch_generation,
                batch,
                roctx,
                args.warmups,
                args.profile_batch_repeats,
                args.profile_batch_scope,
            )
        else:
            record["scopes"] = {
                scope: _benchmark_scope(
                    evaluator,
                    graphs,
                    batch,
                    batch_generation,
                    scope,
                    args.warmups,
                    args.repeats,
                )
                for scope in (
                    "evaluation",
                    "energy_forces",
                    "energy_forces_stress",
                )
            }
        records.append(record)
        print(json.dumps(record), flush=True)

    extension = pathlib.Path(native_symmetrix.__file__).resolve()
    output = {
        "metadata": {
            "model": str(args.model.resolve()),
            "model_sha256": _sha256(args.model),
            "native_extension": str(extension),
            "native_extension_sha256": _sha256(extension),
            "execution_space": native_symmetrix._kokkos_default_execution_space(),
            "device_environment": evaluator.execution_device_execution_environment,
            "dtype": args.dtype,
            "streamed_edges": evaluator.streamed_edges_mode,
            "mh1_edge_executor": evaluator.mh1_edge_executor,
            "jit_status": calculator.jit_status,
            "jit_artifact_path": calculator.jit_artifact_path,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "profile_batch_repeats": args.profile_batch_repeats,
            "profile_batch_scope": args.profile_batch_scope,
            "displacement_A": args.displacement,
            "seed": args.seed,
        },
        "systems": records,
    }
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
