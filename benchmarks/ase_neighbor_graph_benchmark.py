"""Benchmark ASE neighbor-graph rebuild overhead for direct execution."""

# ruff: noqa: I001

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from standard_mace_streamed_benchmark import _bootstrap_explicit_symmetrix

_bootstrap_explicit_symmetrix()

import symmetrix  # noqa: E402
from symmetrix import Symmetrix  # noqa: E402


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _structure(repeat, rattle, seed):
    atoms = Atoms(
        "SrTiO3",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        cell=np.eye(3) * 3.905,
        pbc=True,
    ).repeat((repeat,) * 3)
    atoms.positions += np.random.default_rng(seed).normal(
        scale=rattle, size=atoms.positions.shape
    )
    return atoms


def _measure(evaluate, atoms, calculator, repeats, force_rebuild):
    samples = []
    for _ in range(repeats):
        atoms.positions[0, 0] += 1e-5
        if force_rebuild:
            calculator._neighbor_cache = None
        start = time.perf_counter()
        evaluate()
        samples.append(time.perf_counter() - start)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--neighbor-backend",
        choices=("automatic", "host", "kokkos"),
        default="automatic",
    )
    parser.add_argument("--supercell-repeat", type=int, default=6)
    parser.add_argument("--rattle", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--neighbor-skin", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    os.environ["SYMMETRIX_NEIGHBOR_BACKEND"] = args.neighbor_backend
    atoms = _structure(args.supercell_repeat, args.rattle, args.seed)
    calculator = Symmetrix(
        args.model,
        dtype=args.dtype,
        streamed_edges="direct",
        execution_profile="speed",
        kernel_launch_policy="static",
        neighbor_skin=args.neighbor_skin,
    )

    def evaluate(active_calculator=calculator):
        active_calculator.calculate(atoms, properties=["energy", "forces", "stress"])

    for _ in range(args.warmups):
        evaluate()
    reused = _measure(evaluate, atoms, calculator, args.repeats, False)
    rebuilt = _measure(evaluate, atoms, calculator, args.repeats, True)
    reused_median = statistics.median(reused)
    rebuilt_median = statistics.median(rebuilt)
    cache = calculator._neighbor_cache
    extension = Path(symmetrix.symmetrix.__file__).resolve()
    result = {
        "model": str(Path(args.model).resolve()),
        "model_sha256": _sha256(args.model),
        "extension": str(extension),
        "extension_sha256": _sha256(extension),
        "execution_space": symmetrix._kokkos_default_execution_space(),
        "dtype": args.dtype,
        "neighbor_backend": args.neighbor_backend,
        "selected_neighbor_backend": calculator.neighbor_graph_backend,
        "atoms": len(atoms),
        "directed_edges": cache.num_edges,
        "model_cutoff_angstrom": calculator.cutoff,
        "neighbor_skin_angstrom": calculator.neighbor_skin,
        "effective_cutoff_angstrom": calculator.cutoff + calculator.neighbor_skin,
        "reused_median_ms": 1000.0 * reused_median,
        "rebuilt_median_ms": 1000.0 * rebuilt_median,
        "rebuild_increment_median_ms": 1000.0 * (rebuilt_median - reused_median),
        "reused_us_per_atom": 1e6 * reused_median / len(atoms),
        "rebuilt_us_per_atom": 1e6 * rebuilt_median / len(atoms),
        "reused_atoms_per_second": len(atoms) / reused_median,
        "rebuilt_atoms_per_second": len(atoms) / rebuilt_median,
        "neighbor_cache_build_count": calculator.neighbor_cache_build_count,
        "neighbor_cache_reuse_count": calculator.neighbor_cache_reuse_count,
        "host_geometry_materialization_count": (
            calculator.neighbor_cache_host_geometry_materialization_count
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    atoms.calc = None
    del evaluate
    del calculator
    gc.collect()
    if symmetrix._kokkos_is_initialized():
        symmetrix._finalize_kokkos()


if __name__ == "__main__":
    main()
