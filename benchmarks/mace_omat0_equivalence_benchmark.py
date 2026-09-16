"""Qualified MACE-OMAT-0 Symmetrix versus PyTorch benchmark.

Run each backend in a fresh process with ``run``, then combine two records with
``compare``. The comparison is qualified only when energy, forces, and positions
agree across the complete trajectory within the requested tolerances.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

MD_STEPS = 20


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    packages = (
        "ase",
        "numpy",
        "symmetrix",
        "torch",
        "mace-torch",
        "cuequivariance",
        "cuequivariance-torch",
    )
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def git_metadata() -> dict[str, Any]:
    repository = pathlib.Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def build_atoms(repeat: int, seed: int, temperature_K: float):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((repeat,) * 3)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms, preserve_temperature=True)
    return atoms


def make_calculator(args: argparse.Namespace):
    if args.backend == "torch":
        import torch
        from mace.calculators.mace import MACECalculator

        if args.device == "cpu":
            torch.set_num_threads(args.threads)
        torch_backend = args.torch_backend
        if torch_backend == "auto":
            torch_backend = "cueq" if args.device == "cuda" else "e3nn"
        if torch_backend == "cueq" and args.device != "cuda":
            raise ValueError("cuEquivariance requires --device cuda")
        enable_cueq = torch_backend == "cueq"
        options = {
            "model_paths": args.checkpoint,
            "device": args.device,
            "default_dtype": "float32",
            "enable_cueq": enable_cueq,
        }
        if args.head is not None:
            options["head"] = args.head
        calculator = MACECalculator(**options)
        if bool(getattr(calculator, "enable_cueq", enable_cueq)) != enable_cueq:
            raise RuntimeError(
                f"MACECalculator did not honor the requested {torch_backend} backend"
            )
        return calculator, {
            "implementation": f"pytorch-{torch_backend}",
            "torch_backend_requested": args.torch_backend,
            "cueq_enabled": enable_cueq,
            "head": args.head,
            "torch_num_threads": torch.get_num_threads(),
            "neighbor_list_policy": "standard MACE exact-cutoff rebuild",
            "neighbor_skin_A": None,
        }

    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native_symmetrix

    expected_space = "Cuda" if args.device == "cuda" else "OpenMP"
    execution_space = native_symmetrix._kokkos_default_execution_space()
    if execution_space != expected_space:
        raise RuntimeError(
            f"Symmetrix execution space is {execution_space}, expected {expected_space}"
        )
    streamed_edges = (
        "factorized" if args.symmetrix_mode == "factorized" else "all_interactions"
    )
    jit = "required" if args.symmetrix_mode == "factorized" else "off"
    calculator = Symmetrix(
        args.compact_model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges=streamed_edges,
        jit=jit,
    )
    return calculator, {
        "implementation": f"symmetrix-{args.symmetrix_mode}",
        "execution_space": execution_space,
        "streamed_edges": calculator.streamed_edges,
        "jit_status": calculator.jit_status,
        "jit_compiler_backend": calculator.jit_compiler_backend,
        "jit_artifact_id": calculator.jit_artifact_id,
        "jit_variant_id": calculator.jit_variant_id,
        "jit_edge_policy": calculator.jit_edge_policy,
    }


def synchronize(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def finalize_symmetrix() -> None:
    gc.collect()
    native = sys.modules.get("symmetrix.symmetrix")
    is_initialized = getattr(native, "_kokkos_is_initialized", None)
    finalize = getattr(native, "_finalize_kokkos", None)
    if callable(is_initialized) and callable(finalize) and is_initialized():
        finalize()


def run(args: argparse.Namespace) -> int:
    if args.backend == "torch" and args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")

    if args.device == "cpu":
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        if args.backend == "symmetrix":
            os.environ["KOKKOS_NUM_THREADS"] = str(args.threads)

    atoms = build_atoms(args.repeat, args.seed, args.temperature_K)
    setup_start = time.perf_counter()
    calculator, identity = make_calculator(args)
    setup_ms = 1000.0 * (time.perf_counter() - setup_start)

    atoms.calc = calculator
    actual_initial_temperature_K = float(atoms.get_temperature())
    initial_start = time.perf_counter()
    initial_forces = np.asarray(atoms.get_forces(), dtype=float).copy()
    initial_energy = float(atoms.get_potential_energy())
    synchronize(args.device)
    initial_evaluation_ms = 1000.0 * (time.perf_counter() - initial_start)

    dynamics = VelocityVerlet(atoms, timestep=args.timestep_fs * units.fs, logfile=None)
    samples_ms: list[float] = []
    trajectory = []
    for step in range(1, MD_STEPS + 1):
        synchronize(args.device)
        start = time.perf_counter()
        dynamics.run(1)
        synchronize(args.device)
        samples_ms.append(1000.0 * (time.perf_counter() - start))
        forces = np.asarray(atoms.get_forces(), dtype=float).copy()
        potential = float(atoms.get_potential_energy())
        kinetic = float(atoms.get_kinetic_energy())
        trajectory.append(
            {
                "step": step,
                "potential_energy_eV": potential,
                "kinetic_energy_eV": kinetic,
                "total_energy_eV": potential + kinetic,
                "temperature_K": float(atoms.get_temperature()),
                "forces_eV_per_A": forces.tolist(),
                "positions_A": np.asarray(atoms.positions, dtype=float).tolist(),
                "momenta_eV_fs_per_A": np.asarray(
                    atoms.get_momenta(), dtype=float
                ).tolist(),
            }
        )

    if args.backend == "symmetrix":
        evaluator = calculator.evaluator
        neighbor_cache = getattr(calculator, "_neighbor_cache", None)
        identity["model_cutoff_A"] = float(calculator.cutoff)
        identity["neighbor_skin_A"] = float(calculator.neighbor_skin)
        identity["effective_neighbor_cutoff_A"] = float(
            calculator.cutoff + calculator.neighbor_skin
        )
        identity["candidate_directed_edges"] = (
            len(neighbor_cache.receivers) if neighbor_cache is not None else None
        )
        identity["neighbor_cache_build_count"] = int(
            calculator.neighbor_cache_build_count
        )
        identity["neighbor_cache_reuse_count"] = int(
            calculator.neighbor_cache_reuse_count
        )
        identity["prepared_graph_count"] = getattr(
            evaluator, "factorized_prepared_graph_count", None
        )
        identity["prepared_evaluation_count"] = getattr(
            evaluator, "factorized_prepared_evaluation_count", None
        )
        identity["fallback_evaluation_count"] = getattr(
            evaluator, "factorized_fallback_evaluation_count", None
        )
        expected_runtime = {
            "neighbor_cache_build_count": 1,
            "neighbor_cache_reuse_count": MD_STEPS,
            "prepared_graph_count": 1,
            "prepared_evaluation_count": MD_STEPS + 1,
            "fallback_evaluation_count": 0,
        }
        observed_runtime = {name: identity[name] for name in expected_runtime}
        if observed_runtime != expected_runtime:
            raise RuntimeError(
                "Symmetrix MD reuse contract failed: "
                f"expected {expected_runtime}, observed {observed_runtime}"
            )
        identity["reuse_contract_passed"] = True

    model_path = args.checkpoint if args.backend == "torch" else args.compact_model
    report = {
        "schema_version": 1,
        "git": git_metadata(),
        "backend": args.backend,
        "device": args.device,
        "identity": identity,
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
        },
        "case": {
            "name": "wurtzite-AlN",
            "repeat": args.repeat,
            "atoms": len(atoms),
            "seed": args.seed,
            "position_noise_std_A": 0.01,
            "initial_temperature_K": float(args.temperature_K),
            "actual_initial_temperature_K": actual_initial_temperature_K,
            "cpu_threads": args.threads if args.device == "cpu" else None,
            "dtype": "float32",
        },
        "md": {
            "ensemble": "NVE",
            "integrator": "VelocityVerlet",
            "steps": MD_STEPS,
            "timestep_fs": args.timestep_fs,
            "velocity_distribution": "MaxwellBoltzmannDistribution",
            "center_of_mass_removed": True,
        },
        "timing": {
            "boundary": "one ASE VelocityVerlet NVE step",
            "setup_ms": setup_ms,
            "initial_evaluation_ms": initial_evaluation_ms,
            "initial_evaluation_included": False,
            "steps": MD_STEPS,
            "samples_ms": samples_ms,
            "total_md_ms": sum(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "median_us_per_atom": (1000.0 * statistics.median(samples_ms) / len(atoms)),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "atom_steps_per_second": len(atoms)
            * 1000.0
            / statistics.median(samples_ms),
        },
        "outputs": {
            "initial_energy_eV": initial_energy,
            "initial_forces_eV_per_A": initial_forces.tolist(),
            "trajectory": trajectory,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "packages": package_versions(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "KOKKOS_NUM_THREADS": os.environ.get("KOKKOS_NUM_THREADS"),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    if args.backend == "symmetrix":
        atoms.calc = None
        del calculator
        finalize_symmetrix()
    return 0


def compare(args: argparse.Namespace) -> int:
    symmetrix = json.loads(args.symmetrix.read_text())
    reference = json.loads(args.reference.read_text())
    if symmetrix.get("backend") != "symmetrix":
        raise RuntimeError("--symmetrix record is not a Symmetrix run")
    if reference.get("backend") != "torch":
        raise RuntimeError("--reference record is not a PyTorch run")
    if symmetrix.get("device") != reference.get("device"):
        raise RuntimeError("benchmark device differs between records")
    identity = symmetrix.get("identity", {})
    if identity.get("implementation") != "symmetrix-factorized":
        raise RuntimeError("Symmetrix record is not the factorized implementation")
    if identity.get("streamed_edges") != "factorized":
        raise RuntimeError("Symmetrix record did not use streamed_edges=factorized")
    if identity.get("jit_status") not in ("built", "cached"):
        raise RuntimeError("Symmetrix record did not use a JIT specialization artifact")
    if symmetrix["case"] != reference["case"]:
        raise RuntimeError("benchmark case metadata differs between records")

    if symmetrix["md"] != reference["md"]:
        raise RuntimeError("MD metadata differs between records")
    lhs = symmetrix["outputs"]
    rhs = reference["outputs"]
    lhs_trajectory = lhs["trajectory"]
    rhs_trajectory = rhs["trajectory"]
    expected_steps = symmetrix["md"].get("steps")
    if (
        not isinstance(expected_steps, int)
        or expected_steps <= 0
        or len(lhs_trajectory) != expected_steps
        or len(rhs_trajectory) != expected_steps
    ):
        raise RuntimeError("trajectory length does not match MD metadata")
    energy_errors = np.abs(
        np.asarray([x["potential_energy_eV"] for x in lhs_trajectory])
        - np.asarray([x["potential_energy_eV"] for x in rhs_trajectory])
    )
    energy_error = float(np.max(energy_errors))
    energy_error_per_atom = energy_error / symmetrix["case"]["atoms"]
    force_error = float(
        np.max(
            np.abs(
                np.asarray([x["forces_eV_per_A"] for x in lhs_trajectory])
                - np.asarray([x["forces_eV_per_A"] for x in rhs_trajectory])
            )
        )
    )
    position_error = float(
        np.max(
            np.abs(
                np.asarray([x["positions_A"] for x in lhs_trajectory])
                - np.asarray([x["positions_A"] for x in rhs_trajectory])
            )
        )
    )
    passed = (
        energy_error_per_atom <= args.energy_atol_per_atom
        and force_error <= args.force_atol
        and position_error <= args.position_atol
    )
    symmetrix_ms = symmetrix["timing"]["median_ms"]
    reference_ms = reference["timing"]["median_ms"]
    report = {
        "qualified": passed,
        "case": symmetrix["case"],
        "device": symmetrix["device"],
        "symmetrix_implementation": symmetrix["identity"]["implementation"],
        "reference_implementation": reference["identity"]["implementation"],
        "equivalence": {
            "energy_abs_error_eV": energy_error,
            "energy_abs_error_eV_per_atom": energy_error_per_atom,
            "energy_atol_eV_per_atom": args.energy_atol_per_atom,
            "force_max_abs_error_eV_per_A": force_error,
            "force_atol_eV_per_A": args.force_atol,
            "position_max_abs_error_A": position_error,
            "position_atol_A": args.position_atol,
        },
        "timing": {
            "symmetrix_median_ms": symmetrix_ms,
            "reference_median_ms": reference_ms,
            "symmetrix_median_us_per_atom": symmetrix["timing"]["median_us_per_atom"],
            "reference_median_us_per_atom": reference["timing"]["median_us_per_atom"],
            "speedup_vs_reference": reference_ms / symmetrix_ms,
        },
        "records": {
            "symmetrix": str(args.symmetrix.resolve()),
            "reference": str(args.reference.resolve()),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    return 0 if passed else 2


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", choices=("symmetrix", "torch"), required=True)
    run_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run_parser.add_argument(
        "--torch-backend",
        choices=("auto", "e3nn", "cueq"),
        default="auto",
        help="PyTorch tensor-product backend; auto selects cuEquivariance on CUDA",
    )
    run_parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    run_parser.add_argument("--compact-model", type=pathlib.Path, required=True)
    run_parser.add_argument("--head")
    run_parser.add_argument(
        "--symmetrix-mode",
        choices=("all_interactions", "factorized"),
        default="factorized",
    )
    run_parser.add_argument("--repeat", type=int, required=True)
    run_parser.add_argument("--seed", type=int, default=20260808)
    run_parser.add_argument("--temperature-K", type=float, default=300.0)
    run_parser.add_argument("--timestep-fs", type=float, default=1.0)
    run_parser.add_argument("--threads", type=int, default=16)
    run_parser.add_argument("--output", type=pathlib.Path)
    run_parser.set_defaults(handler=run)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--symmetrix", type=pathlib.Path, required=True)
    compare_parser.add_argument("--reference", type=pathlib.Path, required=True)
    compare_parser.add_argument("--energy-atol-per-atom", type=float, default=2.0e-4)
    compare_parser.add_argument("--force-atol", type=float, default=5.0e-3)
    compare_parser.add_argument("--position-atol", type=float, default=2.0e-4)
    compare_parser.add_argument("--output", type=pathlib.Path)
    compare_parser.set_defaults(handler=compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if hasattr(args, "repeat") and args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if hasattr(args, "temperature_K") and args.temperature_K <= 0.0:
        raise SystemExit("--temperature-K must be positive")
    if hasattr(args, "timestep_fs") and args.timestep_fs <= 0.0:
        raise SystemExit("--timestep-fs must be positive")
    if hasattr(args, "threads") and args.threads < 1:
        raise SystemExit("--threads must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
