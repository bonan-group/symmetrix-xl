"""Compare MH-1 and standard MACE on one deterministic SrTiO3 graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
from ase import Atoms

BENCHMARK_ROOT = pathlib.Path(__file__).resolve().parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from standard_mace_streamed_benchmark import _evaluate, _sha256  # noqa: E402

_DEBUG_EXECUTION_PLANS = (
    "mh0-direct-speed",
    "mh0-direct-capacity-retained",
    "mh0-direct-capacity-y-only",
)
_MH1_NODE_STATE_POLICIES = (
    "full-retention-v1",
    "recompute-v1",
    "reuse-adjoints-v1",
    "retain-interaction-v1",
)
_MH1_NODE_ARENA_POLICIES = ("throughput-v1", "capacity-v1")
_MH1_DENSE_POLICIES = ("auto", "generated", "vendor_fp32", "vendor_tf32")


def _make_atoms(repeat: int = 10) -> Atoms:
    atoms = Atoms(
        symbols=("Sr", "Ti", "O", "O", "O"),
        scaled_positions=(
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        cell=np.eye(3) * 3.905,
        pbc=True,
    ).repeat((repeat, repeat, repeat))
    atoms.rattle(stdev=0.03, seed=20260903)
    return atoms


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _state_sha256(atoms: Atoms) -> str:
    digest = hashlib.sha256()
    for value in (atoms.numbers, atoms.positions, atoms.cell.array):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def _model_summary(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        model = json.load(stream)
    summary = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "model_type": model.get("model_type"),
        "format_version": model.get("symmetrix_format_version"),
        "r_cut_A": model.get("r_cut"),
        "num_elements": model.get("num_elements", len(model.get("atomic_numbers", ()))),
        "num_interactions": model.get("num_interactions"),
        "num_channels": model.get("num_channels"),
        "l_max": model.get("l_max"),
        "L_max": model.get("L_max"),
        "head": model.get("head"),
    }
    if model.get("model_type") == "MACE_Nonlinear":
        summary["layer_irreps"] = [
            {
                key: interaction.get(key)
                for key in (
                    "node_feats_irreps",
                    "edge_irreps",
                    "target_irreps",
                    "hidden_irreps",
                )
            }
            for interaction in model.get("interactions", ())
        ]
    return summary


def _calculator_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "dtype": "float32",
        "streamed_edges": "direct",
        "execution_profile": args.profile,
        "neighbor_skin": 0.5,
        "_debug_execution_plan": args.debug_execution_plan,
    }
    node_state_policy = getattr(args, "mh1_node_state_policy", None)
    if node_state_policy is not None:
        kwargs["execution_mh1_node_state_policy"] = node_state_policy
    node_arena_policy = getattr(args, "mh1_node_arena_policy", None)
    if node_arena_policy is not None:
        kwargs["execution_mh1_node_arena_policy"] = node_arena_policy
    return kwargs


def _apply_runtime_overrides(calculator, args: argparse.Namespace) -> None:
    tile_rows = getattr(args, "mh1_node_arena_tile_rows", None)
    if tile_rows is not None:
        calculator.evaluator._set_execution_mh1_node_arena_tile_rows_for_testing(
            tile_rows
        )


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ["SYMMETRIX_MH1_DENSE_POLICY"] = args.mh1_dense_policy


def run(args: argparse.Namespace) -> dict:
    _configure_environment(args)
    from symmetrix import Symmetrix
    from symmetrix import symmetrix as native

    model = args.model.resolve()
    atoms = _make_atoms(args.repeat)
    calculator = Symmetrix(model, **_calculator_kwargs(args))
    _apply_runtime_overrides(calculator, args)
    if native._kokkos_default_execution_space() != "Cuda":
        raise RuntimeError("comparison requires the CUDA backend")
    if args.family == "mh1" and not calculator.evaluator.uses_mh1_fast_path:
        raise RuntimeError("MH-1 model did not select its direct fast path")
    if args.family == "standard" and bool(
        getattr(calculator.evaluator, "is_mh1_family", False)
    ):
        raise RuntimeError("standard-MACE model selected the MH-1 family")

    result = _evaluate(
        model,
        atoms,
        "kokkos",
        "float32",
        "direct",
        args.warmups,
        args.repeats,
        None,
        None,
        None,
        False,
        args.nvtx,
        calculator=calculator,
    )
    forces = np.asarray(result.pop("forces_eV_per_A"), dtype=np.float64)
    stress = np.asarray(result.pop("stress_eV_per_A3"), dtype=np.float64)
    if args.arrays_output is not None:
        args.arrays_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.arrays_output, forces=forces, stress=stress)
    result["forces"] = {
        "shape": list(forces.shape),
        "l2_norm_eV_per_A": float(np.linalg.norm(forces)),
        "max_abs_eV_per_A": float(np.max(np.abs(forces))),
        "sha256": _array_sha256(forces),
    }
    result["stress"] = {
        "shape": list(stress.shape),
        "l2_norm_eV_per_A3": float(np.linalg.norm(stress)),
        "max_abs_eV_per_A3": float(np.max(np.abs(stress))),
        "sha256": _array_sha256(stress),
    }
    execution_plan = calculator.execution_plan
    if callable(execution_plan):
        execution_plan = execution_plan()
    return {
        "schema": "symmetrix.mh1-standard-mace-comparison/1",
        "family": args.family,
        "execution_profile": args.profile,
        "mh1_dense_policy": {
            "requested": args.mh1_dense_policy,
            "selected": calculator.jit_dense_policy,
        },
        "model": _model_summary(model),
        "system": {
            "description": (
                f"cubic SrTiO3 {args.repeat}x{args.repeat}x{args.repeat}, "
                "rattle sigma=0.03 A, seed=20260903"
            ),
            "atoms": len(atoms),
            "repeat_axes": [args.repeat, args.repeat, args.repeat],
            "state_sha256": _state_sha256(atoms),
        },
        "runtime": {
            "python": sys.executable,
            "backend_build": native._backend_build_info(),
            "execution_space": native._kokkos_default_execution_space(),
            "jit_cache": os.environ.get("SYMMETRIX_JIT_CACHE"),
        },
        "contract": {
            "dtype": "float32",
            "streamed_edges": "direct",
            "properties": ["energy", "forces", "stress"],
            "neighbor_skin_A": 0.5,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "allow_fixed_workspace": False,
            "debug_execution_plan": args.debug_execution_plan,
            "mh1_dense_policy": args.mh1_dense_policy,
            "mh1_node_state_policy": args.mh1_node_state_policy,
            "mh1_node_arena_policy": args.mh1_node_arena_policy,
            "mh1_node_arena_tile_rows": args.mh1_node_arena_tile_rows,
            "nvtx": args.nvtx,
        },
        "execution_plan": execution_plan,
        "result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--family", choices=("mh1", "standard"), required=True)
    parser.add_argument("--profile", choices=("speed", "capacity"), required=True)
    parser.add_argument("--debug-execution-plan", choices=_DEBUG_EXECUTION_PLANS)
    parser.add_argument(
        "--mh1-dense-policy", choices=_MH1_DENSE_POLICIES, default="auto"
    )
    parser.add_argument("--mh1-node-state-policy", choices=_MH1_NODE_STATE_POLICIES)
    parser.add_argument("--mh1-node-arena-policy", choices=_MH1_NODE_ARENA_POLICIES)
    parser.add_argument("--mh1-node-arena-tile-rows", type=int)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--arrays-output", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.repeat < 1:
        parser.error(
            "--warmups must be nonnegative and --repeats/--repeat must be positive"
        )
    if args.mh1_node_arena_tile_rows is not None and args.mh1_node_arena_tile_rows < 1:
        parser.error("--mh1-node-arena-tile-rows must be positive")
    if args.nvtx and args.repeats != 1:
        parser.error("--nvtx requires --repeats=1")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
