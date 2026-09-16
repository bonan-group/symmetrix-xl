"""Benchmark fixed-graph MH-0 evaluation through production CUDA paths."""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from ase_omat_md_backend_scale import (
    BACKENDS,
    array_sha256,
    atomic_json,
    atoms_from_state,
    load_state,
    make_calculator,
    runtime_identity,
    save_state,
    sha256_file,
    state_sha256,
    validate_factorized_identity,
)

SCHEMA_VERSION = 1
DEFAULT_WARMUPS = 10
DEFAULT_SAMPLES = 20


def _summary(samples_s: list[float], atoms: int) -> dict[str, Any]:
    samples_ms = [1000.0 * value for value in samples_s]
    median_ms = statistics.median(samples_ms)
    return {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "median_us_per_atom": 1000.0 * median_ms / atoms,
    }


def _time_cuda_calls(
    operation: Callable[[], Any],
    *,
    warmups: int,
    samples: int,
    label: str,
) -> tuple[dict[str, Any], Any]:
    import torch

    result = None
    for _ in range(warmups):
        with torch.cuda.nvtx.range(f"{label}/warmup"):
            result = operation()
    torch.cuda.synchronize()
    samples_s = []
    for _ in range(samples):
        torch.cuda.synchronize()
        with torch.cuda.nvtx.range(f"{label}/sample"):
            start = time.perf_counter()
            result = operation()
            torch.cuda.synchronize()
        samples_s.append(time.perf_counter() - start)
    return {"samples_s": samples_s}, result


def _native_counters(evaluator) -> dict[str, int | str | bool | None]:
    names = (
        "factorized_graph_generation",
        "factorized_prepared_graph_count",
        "factorized_prepared_evaluation_count",
        "factorized_topology_validation_count",
        "factorized_topology_validation_skip_count",
        "jit_launch_count",
        "jit_forward_launch_count",
        "jit_reverse_launch_count",
        "factorized_fallback_evaluation_count",
        "factorized_schedule_build_count",
        "execution_geometry_capacity_edges",
        "streamed_all_graph_generation",
        "streamed_all_prepared_graph_count",
        "streamed_all_prepared_evaluation_count",
        "streamed_all_geometry_update_count",
        "streamed_all_schedule_build_count",
        "streamed_all_active_edge_count",
    )
    return {name: getattr(evaluator, name, None) for name in names}


def _counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    result = {}
    for name, value in after.items():
        previous = before.get(name)
        if isinstance(value, int) and isinstance(previous, int):
            result[name] = value - previous
    return result


def _native_worker(
    backend: str,
    atoms,
    checkpoint: pathlib.Path,
    compact_model: pathlib.Path,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    calculator = make_calculator(
        backend,
        checkpoint=checkpoint,
        compact_model=compact_model,
    )
    inputs = calculator._mace_inputs(atoms, native_geometry=True)
    initial_token = calculator._compute_mace(inputs, atoms=atoms)
    positions = np.ascontiguousarray(atoms.positions, dtype=float).reshape(-1)
    evaluator = calculator.evaluator
    if backend == "factorized":

        def direct():
            return evaluator._compute_prepared_factorized_positions(
                initial_token, positions
            )

    else:

        def direct():
            return evaluator._compute_prepared_streamed_all_positions(
                initial_token, positions
            )

    before_direct = _native_counters(evaluator)
    direct_raw, _ = _time_cuda_calls(
        direct,
        warmups=warmups,
        samples=samples,
        label="native/direct_prepared_positions",
    )
    after_direct = _native_counters(evaluator)

    def wrapper(calculator=calculator):
        return calculator._compute_mace(inputs, atoms=atoms)

    before_wrapper = _native_counters(evaluator)
    wrapper_raw, final_token = _time_cuda_calls(
        wrapper,
        warmups=warmups,
        samples=samples,
        label="native/fixed_graph_compute_wrapper",
    )
    after_wrapper = _native_counters(evaluator)

    def collect(calculator=calculator):
        return calculator._collect_mace_results(
            atoms,
            inputs,
            properties=("energy", "forces"),
            factorized_graph_generation=final_token,
        )

    collect_raw, _ = _time_cuda_calls(
        collect,
        warmups=warmups,
        samples=samples,
        label="native/force_result_collection",
    )

    def force_call(calculator=calculator, wrapper=wrapper):
        token = wrapper()
        return calculator._collect_mace_results(
            atoms,
            inputs,
            properties=("energy", "forces"),
            factorized_graph_generation=token,
        )

    force_call_raw, results = _time_cuda_calls(
        force_call,
        warmups=warmups,
        samples=samples,
        label="native/fixed_graph_force_call",
    )
    identity = runtime_identity(backend, calculator)
    if backend == "factorized":
        validate_factorized_identity(identity["factorized"])

    from matscipy.neighbours import neighbour_list

    exact_edges = len(neighbour_list("i", atoms, 6.0))
    evaluated_edges = (
        len(inputs[3])
        if backend == "factorized"
        else int(evaluator.streamed_all_active_edge_count)
    )
    output = {
        "runtime": identity,
        "candidate_edges": len(inputs[3]),
        "exact_edges": exact_edges,
        "evaluated_edges": evaluated_edges,
        "timing": {
            "direct_prepared_positions": _summary(direct_raw["samples_s"], len(atoms)),
            "fixed_graph_compute_wrapper": _summary(
                wrapper_raw["samples_s"], len(atoms)
            ),
            "force_result_collection": _summary(collect_raw["samples_s"], len(atoms)),
            "fixed_graph_force_call": _summary(force_call_raw["samples_s"], len(atoms)),
        },
        "counter_deltas": {
            "direct_prepared_positions": _counter_delta(before_direct, after_direct),
            "fixed_graph_compute_wrapper": _counter_delta(
                before_wrapper, after_wrapper
            ),
        },
        "final_counters": after_wrapper,
        "result": {
            "energy_eV": float(results["energy"]),
            "forces_sha256": array_sha256(np.asarray(results["forces"])),
            "max_abs_force_eV_per_A": float(
                np.max(np.abs(np.asarray(results["forces"])))
            ),
        },
    }
    del direct, wrapper, collect, force_call, calculator
    gc.collect()
    return output


def _torch_worker(
    atoms,
    checkpoint: pathlib.Path,
    compact_model: pathlib.Path,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    import torch

    calculator = make_calculator(
        "torch_cueq",
        checkpoint=checkpoint,
        compact_model=compact_model,
    )
    model = calculator.models[0]
    batch = calculator._atoms_to_batch(atoms)
    model_dtype = next(model.parameters()).dtype
    for key in batch.keys:
        value = batch[key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch[key] = value.to(dtype=model_dtype)
    fixed_data = batch.to_dict()

    def direct():
        return model(
            fixed_data,
            compute_stress=False,
            training=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )

    direct_raw, output = _time_cuda_calls(
        direct,
        warmups=warmups,
        samples=samples,
        label="torch/direct_prebuilt_batch",
    )

    def cloned():
        value = calculator._clone_batch(batch)
        return model(
            value.to_dict(),
            compute_stress=False,
            training=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )

    clone_raw, output = _time_cuda_calls(
        cloned,
        warmups=warmups,
        samples=samples,
        label="torch/cloned_prebuilt_batch",
    )

    def materialize():
        return (
            output["energy"].detach().cpu().numpy(),
            output["forces"].detach().cpu().numpy(),
        )

    materialize_raw, _ = _time_cuda_calls(
        materialize,
        warmups=warmups,
        samples=samples,
        label="torch/force_result_materialization",
    )

    def force_call():
        value = direct()
        return (
            value["energy"].detach().cpu().numpy(),
            value["forces"].detach().cpu().numpy(),
        )

    force_call_raw, _ = _time_cuda_calls(
        force_call,
        warmups=warmups,
        samples=samples,
        label="torch/fixed_graph_force_call",
    )
    forces = output["forces"].detach().cpu().numpy()
    return {
        "runtime": runtime_identity("torch_cueq", calculator),
        "candidate_edges": int(batch["edge_index"].shape[1]),
        "exact_edges": int(batch["edge_index"].shape[1]),
        "evaluated_edges": int(batch["edge_index"].shape[1]),
        "timing": {
            "direct_prebuilt_batch": _summary(direct_raw["samples_s"], len(atoms)),
            "cloned_prebuilt_batch": _summary(clone_raw["samples_s"], len(atoms)),
            "force_result_materialization": _summary(
                materialize_raw["samples_s"], len(atoms)
            ),
            "fixed_graph_force_call": _summary(force_call_raw["samples_s"], len(atoms)),
        },
        "result": {
            "energy_eV": float(output["energy"].detach().cpu().reshape(-1)[0]),
            "forces_sha256": array_sha256(forces),
            "max_abs_force_eV_per_A": float(np.max(np.abs(forces))),
        },
    }


def run_worker(args: argparse.Namespace) -> int:
    state = load_state(args.state.resolve())
    atoms = atoms_from_state(state)
    checkpoint = args.checkpoint.resolve()
    compact_model = args.compact_model.resolve()
    if args.backend == "torch_cueq":
        backend_result = _torch_worker(
            atoms,
            checkpoint,
            compact_model,
            args.warmups,
            args.samples,
        )
    else:
        backend_result = _native_worker(
            args.backend,
            atoms,
            checkpoint,
            compact_model,
            args.warmups,
            args.samples,
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "backend": args.backend,
        "trial": args.trial,
        "repeat": int(state["repeat"]),
        "atoms": len(atoms),
        "state": {
            "path": str(args.state.resolve()),
            "sha256": state["state_sha256"],
        },
        "model": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "compact": str(compact_model),
            "compact_sha256": sha256_file(compact_model),
        },
        "contract": {
            "fixed_topology": True,
            "fixed_positions": True,
            "warmups_per_boundary": args.warmups,
            "samples_per_boundary": args.samples,
            "cuda_synchronized_before_and_after_each_sample": True,
        },
        **backend_result,
    }
    atomic_json(args.output.resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def run_campaign(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    records_dir = output_dir / "records"
    logs_dir = output_dir / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(__file__).resolve()
    states = {
        repeat: args.states_dir.resolve() / f"aln-n{repeat}-seed20260804.npz"
        for repeat in args.repeats
    }
    records = []
    orders = (
        BACKENDS,
        ("torch_cueq", "factorized", "kokkos_all"),
        ("factorized", "kokkos_all", "torch_cueq"),
    )
    for trial in range(1, args.trials + 1):
        order = orders[(trial - 1) % len(orders)]
        for repeat in args.repeats:
            for backend in order:
                stem = f"{backend}-n{repeat}-trial{trial}"
                output = records_dir / f"{stem}.json"
                command = [
                    sys.executable,
                    str(script),
                    "worker",
                    "--backend",
                    backend,
                    "--state",
                    str(states[repeat]),
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--compact-model",
                    str(args.compact_model.resolve()),
                    "--warmups",
                    str(args.warmups),
                    "--samples",
                    str(args.samples),
                    "--trial",
                    str(trial),
                    "--output",
                    str(output),
                ]
                completed = subprocess.run(
                    command, check=False, capture_output=True, text=True
                )
                (logs_dir / f"{stem}.stdout").write_text(completed.stdout)
                (logs_dir / f"{stem}.stderr").write_text(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{stem} failed with exit {completed.returncode}: "
                        f"{completed.stderr[-2000:]}"
                    )
                with output.open(encoding="utf-8") as handle:
                    records.append(json.load(handle))
                print(f"completed {stem}", flush=True)
    atomic_json(
        output_dir / "campaign.json",
        {
            "schema_version": SCHEMA_VERSION,
            "trials": args.trials,
            "warmups": args.warmups,
            "samples": args.samples,
            "repeats": list(args.repeats),
            "records": records,
        },
    )
    return 0


def run_snapshot(args: argparse.Namespace) -> int:
    from ase import units
    from ase.md.verlet import VelocityVerlet

    state = load_state(args.state.resolve())
    atoms = atoms_from_state(state)
    calculator = make_calculator(
        args.backend,
        checkpoint=args.checkpoint.resolve(),
        compact_model=args.compact_model.resolve(),
    )
    atoms.calc = calculator
    dynamics = VelocityVerlet(atoms, timestep=args.timestep_fs * units.fs, logfile=None)
    dynamics.run(args.steps)
    state.update(
        {
            "actual_temperature_K": float(atoms.get_temperature()),
            "positions_A": np.asarray(atoms.positions, dtype=np.float64),
            "momenta_eV_fs_per_A": np.asarray(atoms.get_momenta(), dtype=np.float64),
            "snapshot_source_state_sha256": state["state_sha256"],
            "snapshot_backend": args.backend,
            "snapshot_steps": args.steps,
            "snapshot_timestep_fs": args.timestep_fs,
        }
    )
    state["state_sha256"] = state_sha256(state)
    save_state(args.output.resolve(), state)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "repeat": state["repeat"],
                "atoms": len(atoms),
                "state_sha256": state["state_sha256"],
                "runtime": runtime_identity(args.backend, calculator),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _repeats(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--backend", choices=BACKENDS, required=True)
    worker.add_argument("--state", type=pathlib.Path, required=True)
    worker.add_argument("--checkpoint", type=pathlib.Path, required=True)
    worker.add_argument("--compact-model", type=pathlib.Path, required=True)
    worker.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    worker.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    worker.add_argument("--trial", type=int, default=1)
    worker.add_argument("--output", type=pathlib.Path, required=True)

    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--states-dir", type=pathlib.Path, required=True)
    campaign.add_argument("--checkpoint", type=pathlib.Path, required=True)
    campaign.add_argument("--compact-model", type=pathlib.Path, required=True)
    campaign.add_argument("--repeats", type=_repeats, default=(6, 11, 14))
    campaign.add_argument("--trials", type=int, default=3)
    campaign.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    campaign.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    campaign.add_argument("--output-dir", type=pathlib.Path, required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--backend", choices=BACKENDS, default="factorized")
    snapshot.add_argument("--state", type=pathlib.Path, required=True)
    snapshot.add_argument("--checkpoint", type=pathlib.Path, required=True)
    snapshot.add_argument("--compact-model", type=pathlib.Path, required=True)
    snapshot.add_argument("--steps", type=int, default=20)
    snapshot.add_argument("--timestep-fs", type=float, default=1.0)
    snapshot.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command != "snapshot" and (args.warmups < 0 or args.samples < 1):
        parser.error("warmups must be nonnegative and samples must be positive")
    if args.command == "campaign" and args.trials < 1:
        parser.error("trials must be positive")
    if args.command == "snapshot" and (args.steps < 1 or args.timestep_fs <= 0.0):
        parser.error("snapshot steps and timestep must be positive")
    if args.command == "worker":
        return run_worker(args)
    if args.command == "campaign":
        return run_campaign(args)
    return run_snapshot(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        gc.collect()
        native = sys.modules.get("symmetrix.symmetrix")
        is_initialized = getattr(native, "_kokkos_is_initialized", None)
        finalize = getattr(native, "_finalize_kokkos", None)
        if callable(is_initialized) and callable(finalize) and is_initialized():
            finalize()
