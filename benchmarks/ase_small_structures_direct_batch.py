"""Measure joined small-structure direct batches with a prepared segment group.

The benchmark intentionally exercises ``atombatch.GraphBatcher`` rather than
calling the calculator in a loop.  That is the production shape in which a
joined direct graph used to rebuild its topology-specialized schedule every
step. Every trajectory carries an atombatch ``Source``. Symmetrix combines
those trajectory skin graphs into one prepared group and streams concatenated
positions, so edge geometry and cutoff masking stay on the device. ASE
Langevin at 300 K is the qualification workload. The restrained process is
retained only as a bounded-motion smoke test and must not be used for reported
throughput results.

``--stress --cell-strain-amplitude`` adds a bounded affine cell oscillation
that exercises the changing-cell and segmented-stress path used by NPT
integrators. It is a performance and parity probe, not a pressure barostat.

The v0.2 calculator exports the compatibility hooks required by atombatch
0.1.x, so no runtime shims are needed. ``atombatch`` is an optional benchmark
dependency.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "structures",
        nargs="*",
        type=Path,
        help="Periodic structure or trajectory files; repeated to fill --tasks.",
    )
    parser.add_argument(
        "--glob",
        dest="structure_glob",
        help="Glob used when positional structure paths are omitted.",
    )
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Trajectory frame used as the initial MD state.",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--dynamics",
        choices=("ase-langevin", "restrained-langevin"),
        default="ase-langevin",
        help="Use ase-langevin for qualification; restrained-langevin is smoke-only.",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--friction-per-fs", type=float, default=0.01)
    parser.add_argument(
        "--fix-center-of-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match ASE Langevin's fixcm policy in the qualification dynamics.",
    )
    parser.add_argument(
        "--max-displacement",
        type=float,
        default=0.05,
        help="Per-atom guard used only by restrained-langevin smoke dynamics.",
    )
    parser.add_argument("--thermal-displacement-rms", type=float, default=0.01)
    parser.add_argument("--position-rms", type=float, default=0.15)
    parser.add_argument("--neighbor-skin", type=float, default=0.5)
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Request per-structure stress from the joined prepared graph.",
    )
    parser.add_argument(
        "--cell-strain-amplitude",
        type=float,
        default=0.0,
        help=(
            "Peak bounded isotropic strain applied during the run; requires "
            "--stress and qualifies the changing-cell NPT execution path."
        ),
    )
    parser.add_argument(
        "--cell-strain-period",
        type=int,
        default=200,
        help="Cell-strain oscillation period in MD steps.",
    )
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument(
        "--check-parity",
        action="store_true",
        help="After timing, compare the final state with single-trajectory direct.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _metric(evaluator, name: str) -> int:
    return int(getattr(evaluator, name, 0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_structures(
    paths: list[Path], pattern: str | None, tasks: int, frame_index: int
):
    candidates = list(paths)
    if not candidates and pattern:
        candidates = [Path(path) for path in sorted(glob.glob(pattern))]
    if not candidates:
        raise SystemExit("provide structure paths or --glob")
    loaded = [read(path, index=frame_index) for path in candidates]
    if any(len(atoms) < 54 or len(atoms) > 320 for atoms in loaded):
        raise SystemExit("all benchmark structures must contain 54-320 atoms")
    return [loaded[index % len(loaded)].copy() for index in range(tasks)]


def _langevin_begin(
    structures,
    forces,
    rng,
    timestep_fs,
    temperature,
    friction_per_fs,
    fix_center_of_mass,
):
    timestep = float(timestep_fs) * units.fs
    friction = float(friction_per_fs) / units.fs
    temperature_energy = units.kB * float(temperature)
    states = []
    for atoms, force in zip(structures, forces):
        force = np.asarray(force, dtype=float)
        masses = atoms.get_masses()[:, None]
        sigma = np.sqrt(2.0 * temperature_energy * friction / masses)
        c1 = timestep / 2.0 - timestep * timestep * friction / 8.0
        c2 = timestep * friction / 2.0 - (
            timestep * timestep * friction * friction / 8.0
        )
        c3 = np.sqrt(timestep) * sigma / 2.0 - timestep**1.5 * friction * sigma / 8.0
        c5 = timestep**1.5 * sigma / (2.0 * np.sqrt(3.0))
        c4 = friction * c5 / 2.0
        xi = rng.standard_normal(size=atoms.positions.shape)
        eta = rng.standard_normal(size=atoms.positions.shape)
        random_position = c5 * eta
        random_velocity = c3 * xi - c4 * eta
        if fix_center_of_mass:
            atom_count = len(atoms)
            factor = np.sqrt(atom_count / (atom_count - 1.0))
            random_position -= random_position.sum(axis=0) / atom_count
            random_velocity -= (random_velocity * masses).sum(axis=0) / (
                masses * atom_count
            )
            random_position *= factor
            random_velocity *= factor
        velocities = atoms.get_velocities()
        velocities += c1 * force / masses - c2 * velocities + random_velocity
        displacement = timestep * velocities + random_position
        old_positions = atoms.positions.copy()
        atoms.positions += displacement
        velocities = (atoms.positions - old_positions - random_position) / timestep
        atoms.set_velocities(velocities)
        states.append((masses, c1, c2, random_velocity))
    return states


def _langevin_end(structures, forces, states):
    for atoms, force, state in zip(structures, forces, states):
        masses, c1, c2, random_velocity = state
        velocities = atoms.get_velocities()
        velocities += (
            c1 * np.asarray(force, dtype=float) / masses
            - c2 * velocities
            + random_velocity
        )
        atoms.set_velocities(velocities)


def _initialize_velocities(structures, temperature, rng):
    for atoms in structures:
        masses = atoms.get_masses()[:, None]
        velocities = rng.normal(size=atoms.positions.shape) * np.sqrt(
            units.kB * float(temperature) / masses
        )
        velocities -= np.sum(masses * velocities, axis=0) / np.sum(masses)
        atoms.set_velocities(velocities)


def _restrained_langevin_step(
    structures,
    references,
    rng,
    temperature,
    timestep_fs,
    thermal_displacement_rms,
    position_rms,
    max_displacement,
):
    noise_rms = float(thermal_displacement_rms) * np.sqrt(
        float(temperature) / 300.0 * float(timestep_fs) / 2.0
    )
    relaxation = min(1.0, noise_rms * noise_rms / (2.0 * position_rms**2))
    component_sigma = noise_rms / np.sqrt(3.0)
    for atoms, reference in zip(structures, references):
        displacement = -relaxation * (atoms.positions - reference)
        displacement += rng.normal(0.0, component_sigma, size=atoms.positions.shape)
        displacement_norm = np.linalg.norm(displacement, axis=1)
        limited = displacement_norm > float(max_displacement)
        if np.any(limited):
            displacement[limited] *= (
                float(max_displacement) / displacement_norm[limited]
            )[:, None]
        atoms.positions += displacement


def _apply_cell_strain(structures, reference_cells, step, amplitude, period):
    if amplitude == 0.0:
        return
    phase = 2.0 * np.pi * float(step) / float(period)
    scale = float(np.exp(float(amplitude) * np.sin(phase)))
    for atoms, reference_cell in zip(structures, reference_cells):
        atoms.set_cell(reference_cell * scale, scale_atoms=True)


def run(args: argparse.Namespace) -> dict:
    from atombatch.backends.base import SOURCE_KEY, Source
    from atombatch.backends.symmetrix import GraphBatcher

    from symmetrix import Symmetrix, load_backend

    if args.cell_strain_amplitude < 0.0:
        raise SystemExit("--cell-strain-amplitude must be non-negative")
    if args.cell_strain_amplitude > 0.0 and not args.stress:
        raise SystemExit("--cell-strain-amplitude requires --stress")
    if args.cell_strain_period <= 0:
        raise SystemExit("--cell-strain-period must be positive")
    if args.tasks <= 0 or args.steps <= 0 or args.warmup < 0:
        raise SystemExit(
            "--tasks and --steps must be positive; --warmup cannot be negative"
        )

    structures = _load_structures(
        args.structures, args.structure_glob, args.tasks, args.frame_index
    )
    if args.dynamics == "ase-langevin" and any(
        atoms.constraints for atoms in structures
    ):
        raise SystemExit(
            "ase-langevin qualification does not support constrained structures"
        )
    sources = [Source() for _ in structures]
    for atoms, source in zip(structures, sources):
        atoms.info[SOURCE_KEY] = source

    calculator = Symmetrix(
        args.model,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="speed",
        neighbor_skin=args.neighbor_skin,
    )
    batcher = GraphBatcher(calculator, native_atoms=1000)
    layer_seconds = {
        "neighbors": 0.0,
        "cached_neighbor_geometry": 0.0,
        "atombatch_edge_geometry": 0.0,
        "compute_mace": 0.0,
    }

    original_neighbors = batcher._neighbors

    def timed_neighbors(atoms):
        start = time.perf_counter()
        try:
            return original_neighbors(atoms)
        finally:
            layer_seconds["neighbors"] += time.perf_counter() - start

    batcher._neighbors = timed_neighbors
    original_cached_neighbor_geometry = calculator._cached_neighbor_geometry

    def timed_cached_neighbor_geometry(*neighbor_args, **neighbor_kwargs):
        start = time.perf_counter()
        try:
            return original_cached_neighbor_geometry(*neighbor_args, **neighbor_kwargs)
        finally:
            layer_seconds["cached_neighbor_geometry"] += time.perf_counter() - start

    calculator._cached_neighbor_geometry = timed_cached_neighbor_geometry
    original_edge_geometry = batcher._edge_geometry

    def timed_edge_geometry(*edge_args, **edge_kwargs):
        start = time.perf_counter()
        try:
            return original_edge_geometry(*edge_args, **edge_kwargs)
        finally:
            layer_seconds["atombatch_edge_geometry"] += time.perf_counter() - start

    batcher._edge_geometry = timed_edge_geometry
    original_compute_mace = calculator._compute_mace

    def timed_compute_mace(mace_inputs, atoms=None):
        start = time.perf_counter()
        try:
            return original_compute_mace(mace_inputs, atoms=atoms)
        finally:
            layer_seconds["compute_mace"] += time.perf_counter() - start

    calculator._compute_mace = timed_compute_mace
    properties = (
        ("energy", "forces", "stress")
        if args.stress
        else (
            "energy",
            "forces",
        )
    )
    rng = np.random.default_rng(args.seed)
    references = [atoms.positions.copy() for atoms in structures]
    reference_cells = [
        np.asarray(atoms.cell, dtype=float).copy() for atoms in structures
    ]
    if args.dynamics == "ase-langevin":
        _initialize_velocities(structures, args.temperature, rng)
    results = batcher.evaluate(structures, properties)

    for warmup_step in range(1, args.warmup + 1):
        if args.dynamics == "ase-langevin":
            states = _langevin_begin(
                structures,
                [result["forces"] for result in results],
                rng,
                args.timestep_fs,
                args.temperature,
                args.friction_per_fs,
                args.fix_center_of_mass,
            )
        else:
            _restrained_langevin_step(
                structures,
                references,
                rng,
                args.temperature,
                args.timestep_fs,
                args.thermal_displacement_rms,
                args.position_rms,
                args.max_displacement,
            )
        _apply_cell_strain(
            structures,
            reference_cells,
            warmup_step,
            args.cell_strain_amplitude,
            args.cell_strain_period,
        )
        results = batcher.evaluate(structures, properties)
        if args.dynamics == "ase-langevin":
            _langevin_end(
                structures,
                [result["forces"] for result in results],
                states,
            )
    _synchronize()

    evaluator = calculator.evaluator
    for name in layer_seconds:
        layer_seconds[name] = 0.0
    before = {
        name: _metric(evaluator, name)
        for name in (
            "factorized_schedule_build_count",
            "factorized_prepared_graph_count",
            "factorized_fallback_evaluation_count",
            "execution_prepared_geometry_update_count",
        )
    }
    neighbor_counters_before = {
        "builds": calculator.neighbor_cache_build_count,
        "reuses": calculator.neighbor_cache_reuse_count,
        "geometry_updates": calculator.neighbor_cache_geometry_update_count,
        "host_geometry_materializations": (
            calculator.neighbor_cache_host_geometry_materialization_count
        ),
    }
    md_begin_seconds = 0.0
    evaluation_seconds = 0.0
    md_end_seconds = 0.0
    start = time.perf_counter()
    for timed_step in range(1, args.steps + 1):
        phase_start = time.perf_counter()
        if args.dynamics == "ase-langevin":
            states = _langevin_begin(
                structures,
                [result["forces"] for result in results],
                rng,
                args.timestep_fs,
                args.temperature,
                args.friction_per_fs,
                args.fix_center_of_mass,
            )
        else:
            _restrained_langevin_step(
                structures,
                references,
                rng,
                args.temperature,
                args.timestep_fs,
                args.thermal_displacement_rms,
                args.position_rms,
                args.max_displacement,
            )
        _apply_cell_strain(
            structures,
            reference_cells,
            args.warmup + timed_step,
            args.cell_strain_amplitude,
            args.cell_strain_period,
        )
        md_begin_seconds += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        results = batcher.evaluate(structures, properties)
        evaluation_seconds += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        if args.dynamics == "ase-langevin":
            _langevin_end(
                structures,
                [result["forces"] for result in results],
                states,
            )
        md_end_seconds += time.perf_counter() - phase_start
    _synchronize()
    elapsed = time.perf_counter() - start
    after = {
        name: _metric(evaluator, name)
        for name in (
            "factorized_schedule_build_count",
            "factorized_prepared_graph_count",
            "factorized_fallback_evaluation_count",
            "execution_prepared_geometry_update_count",
        )
    }
    atom_count = sum(len(atoms) for atoms in structures)
    native = load_backend()
    extension_path = Path(native.__file__).resolve()
    backend_build_info = getattr(native, "_backend_build_info", dict)()
    execution_space = getattr(native, "_kokkos_default_execution_space", lambda: None)()
    properties_record = list(properties)
    directed_edges = _metric(evaluator, "factorized_schedule_entries")
    memory_metrics = {
        name: _metric(evaluator, name)
        for name in (
            "factorized_schedule_bytes",
            "factorized_workspace_bytes",
            "factorized_arena_workspace_bytes",
            "factorized_radial_workspace_bytes",
            "execution_geometry_workspace_bytes",
            "edge_workspace_bytes",
            "conditioned_mlp_workspace_bytes",
            "linear_workspace_bytes",
            "tensor_workspace_bytes",
            "product_workspace_bytes",
            "node_workspace_bytes",
            "precision_workspace_bytes",
        )
        if hasattr(evaluator, name)
    }
    result = {
        "model": str(args.model),
        "model_sha256": _sha256(args.model),
        "precision": "float32",
        "backend": backend_build_info.get("backend"),
        "architecture": backend_build_info.get("architecture"),
        "execution_space": execution_space,
        "native_extension": str(extension_path),
        "native_extension_sha256": _sha256(extension_path),
        "native_build": backend_build_info,
        "jit_artifact_path": calculator.jit_artifact_path,
        "jit_artifact_id": calculator.jit_artifact_id,
        "tasks": len(structures),
        "frame_index": args.frame_index,
        "atoms": [len(atoms) for atoms in structures],
        "steps": args.steps,
        "warmup": args.warmup,
        "dynamics": args.dynamics,
        "temperature": args.temperature,
        "timestep_fs": args.timestep_fs,
        "friction_per_fs": args.friction_per_fs,
        "fix_center_of_mass": args.fix_center_of_mass,
        "max_displacement": args.max_displacement,
        "thermal_displacement_rms": args.thermal_displacement_rms,
        "position_rms": args.position_rms,
        "neighbor_skin": args.neighbor_skin,
        "cutoff": calculator.cutoff,
        "effective_joined_cutoff": calculator.cutoff + calculator.joined_neighbor_skin,
        "directed_edges": directed_edges,
        "properties": properties_record,
        "stress": args.stress,
        "cell_strain_amplitude": args.cell_strain_amplitude,
        "cell_strain_period": args.cell_strain_period,
        "joined_neighbor_skin": calculator.joined_neighbor_skin,
        "trajectory_neighbor_caches": len(calculator._trajectory_neighbor_caches),
        "elapsed_s": elapsed,
        "us_per_atom_step": elapsed * 1.0e6 / (atom_count * args.steps),
        "atom_steps_per_s": atom_count * args.steps / elapsed,
        "memory_bytes": memory_metrics,
        "counters_before": before,
        "counters_after": after,
        "counter_deltas": {name: after[name] - before[name] for name in after},
        "neighbor_cache_deltas": {
            "builds": calculator.neighbor_cache_build_count
            - neighbor_counters_before["builds"],
            "reuses": calculator.neighbor_cache_reuse_count
            - neighbor_counters_before["reuses"],
            "geometry_updates": calculator.neighbor_cache_geometry_update_count
            - neighbor_counters_before["geometry_updates"],
            "host_geometry_materializations": (
                calculator.neighbor_cache_host_geometry_materialization_count
                - neighbor_counters_before["host_geometry_materializations"]
            ),
        },
        "timing_seconds": {
            "md_begin": md_begin_seconds,
            "evaluation": evaluation_seconds,
            "md_end": md_end_seconds,
        },
        "evaluation_layer_seconds": layer_seconds,
    }
    if args.check_parity:
        reference_calculator = Symmetrix(
            args.model,
            dtype="float32",
            streamed_edges="direct",
            execution_profile="speed",
            neighbor_skin=args.neighbor_skin,
        )
        maximum_energy_error = 0.0
        maximum_force_error = 0.0
        maximum_stress_error = 0.0
        for atoms, joined in zip(structures, results):
            reference_atoms = atoms.copy()
            reference_atoms.info.pop(SOURCE_KEY, None)
            reference_atoms.calc = reference_calculator
            reference_energy = reference_atoms.get_potential_energy()
            reference_forces = reference_atoms.get_forces()
            reference_stress = reference_atoms.get_stress() if args.stress else None
            maximum_energy_error = max(
                maximum_energy_error,
                abs(float(joined["energy"]) - float(reference_energy)),
            )
            maximum_force_error = max(
                maximum_force_error,
                float(np.max(np.abs(np.asarray(joined["forces"]) - reference_forces))),
            )
            if args.stress:
                maximum_stress_error = max(
                    maximum_stress_error,
                    float(
                        np.max(np.abs(np.asarray(joined["stress"]) - reference_stress))
                    ),
                )
        result["parity"] = {
            "maximum_energy_absolute_error_ev": maximum_energy_error,
            "maximum_force_absolute_error_ev_per_angstrom": maximum_force_error,
            "maximum_stress_absolute_error_ev_per_angstrom3": (
                maximum_stress_error if args.stress else None
            ),
        }
    if result["steps"] != args.steps or result["tasks"] != args.tasks:
        raise RuntimeError("benchmark result does not match requested work extents")
    if (
        result["cutoff"] != calculator.cutoff
        or result["neighbor_skin"] != args.neighbor_skin
    ):
        raise RuntimeError("benchmark result does not match requested neighbor policy")
    if result["precision"] != "float32" or result["properties"] != properties_record:
        raise RuntimeError(
            "benchmark result does not match requested evaluation policy"
        )
    if result["directed_edges"] <= 0:
        raise RuntimeError("benchmark did not prepare a directed-edge schedule")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = _parser().parse_args()
    completed = False
    try:
        print(json.dumps(run(args), indent=2))
        completed = True
    finally:
        if completed:
            gc.collect()
            from symmetrix import load_backend

            native = load_backend()
            finalize = getattr(native, "_finalize_kokkos", None)
            initialized = getattr(native, "_kokkos_is_initialized", lambda: False)
            if callable(finalize) and initialized():
                finalize()


if __name__ == "__main__":
    main()
