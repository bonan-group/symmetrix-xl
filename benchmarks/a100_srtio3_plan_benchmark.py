#!/usr/bin/env python3
"""Measure one explicit Symmetrix execution plan on cubic SrTiO3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_axes(value: str) -> tuple[int, int, int]:
    axes = tuple(int(item) for item in value.split(","))
    if len(axes) != 3 or any(item <= 0 for item in axes):
        raise argparse.ArgumentTypeError("axes must contain three positive integers")
    return axes


def _structure(axes: tuple[int, int, int]):
    import numpy as np
    from ase import Atoms

    lattice = 3.905
    return Atoms(
        "SrTiO3",
        scaled_positions=(
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        cell=np.eye(3) * lattice,
        pbc=True,
    ).repeat(axes)


def _metric(instance, name: str, default=None):
    value = getattr(instance, name, default)
    return value() if callable(value) else value


def run(args: argparse.Namespace) -> dict:
    from symmetrix import Symmetrix, backend_loader, selected_backend

    atoms = _structure(args.axes)
    calculator = Symmetrix(
        args.model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile=args.profile,
        neighbor_skin=args.skin,
        _debug_execution_plan=args.plan,
    )
    properties = ["energy", "energies", "forces", "stress"]
    calculator.calculate(atoms, properties=properties)
    for _ in range(args.warmups):
        calculator.calculate(atoms, properties=properties, system_changes=[])
    samples_ms = []
    for _ in range(args.samples):
        started = time.perf_counter_ns()
        calculator.calculate(atoms, properties=properties, system_changes=[])
        samples_ms.append((time.perf_counter_ns() - started) / 1.0e6)

    cache = calculator._neighbor_cache
    if cache is None:
        raise RuntimeError("evaluation did not retain a neighbor graph")
    selected_plan = calculator.execution_plan
    selected_id = (
        selected_plan.get("selected_id")
        if isinstance(selected_plan, dict)
        else selected_plan
    )
    if selected_id != args.plan:
        raise RuntimeError(f"requested {args.plan}, selected {selected_id}")
    evaluator = calculator.evaluator
    fallback_count = int(_metric(evaluator, "factorized_fallback_evaluation_count", 0))
    if fallback_count:
        raise RuntimeError(f"observed {fallback_count} fallback evaluations")

    native = backend_loader._native_module
    if native is None:
        raise RuntimeError("native backend was not loaded")
    native_path = Path(native.__file__).resolve()
    model_path = args.model.resolve()
    atoms_count = len(atoms)
    median_ms = statistics.median(samples_ms)
    result = {
        "schema": "symmetrix.a100-srtio3-explicit-plan/1",
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "native_extension": str(native_path),
        "native_extension_sha256": _sha256(native_path),
        "native_build": native._backend_build_info(),
        "backend": selected_backend(),
        "kokkos_execution_space": native._kokkos_default_execution_space(),
        "structure": "cubic SrTiO3",
        "lattice_A": 3.905,
        "repeat_axes": list(args.axes),
        "atoms": atoms_count,
        "directed_edges": int(cache.num_edges),
        "model_cutoff_A": float(calculator.cutoff),
        "neighbor_skin_A": float(calculator.neighbor_skin),
        "effective_cutoff_A": float(calculator.cutoff + calculator.neighbor_skin),
        "dtype": "float32",
        "requested_profile": args.profile,
        "requested_plan": args.plan,
        "execution_plan": calculator.execution_plan,
        "warmups": args.warmups,
        "samples": args.samples,
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "median_us_per_atom": 1000.0 * median_ms / atoms_count,
        "fallback_evaluations": fallback_count,
        "energy_eV": float(calculator.results["energy"]),
    }

    del cache, evaluator, calculator, atoms
    gc.collect()
    if native._kokkos_is_initialized():
        native._finalize_kokkos()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--axes", type=_parse_axes, default=(10, 10, 10))
    parser.add_argument("--profile", choices=("speed", "capacity"), required=True)
    parser.add_argument(
        "--plan",
        choices=("mh0-direct-speed", "mh0-direct-capacity-y-only"),
        required=True,
    )
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.profile == "speed" and args.plan != "mh0-direct-speed":
        parser.error("speed profile requires mh0-direct-speed")
    if args.profile == "capacity" and args.plan != "mh0-direct-capacity-y-only":
        parser.error("capacity profile requires mh0-direct-capacity-y-only")
    if args.warmups < 0 or args.samples < 1:
        parser.error("warmups must be non-negative and samples must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
