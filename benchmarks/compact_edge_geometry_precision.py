"""Qualify the production compact edge-geometry policy against its control."""

import argparse
import copy
import json
import pathlib

import numpy as np
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from ase.md.verlet import VelocityVerlet
from standard_mace_streamed_benchmark import _make_calculator

CONTROL_POLICY = "cartesian-f64-v1"
COMPACT_POLICY = "unit-f32-radius-f64-v1"


def _make_mh0_calculator(model, policy, neighbor_skin):
    return _make_calculator(
        model,
        "kokkos",
        "float32",
        "factorized",
        factorized_source_strategy="jit_plugin",
        m1_polynomial_policy="recompute",
        mh0_state_policy="reuse-adjoints-v1",
        edge_geometry_policy=policy,
        neighbor_skin=neighbor_skin,
    )


def _collect_static(calculator, atoms):
    inputs = calculator._mace_inputs(atoms)
    token = calculator.evaluator._prepare_factorized_graph(*inputs[:5])
    calculator.evaluator._compute_prepared_factorized(
        token, np.ascontiguousarray(inputs[5]).reshape(-1), inputs[6]
    )
    results = calculator._collect_mace_results(
        atoms, inputs, ("energy", "forces", "stress"), token
    )
    return {
        "results": {key: np.asarray(value) for key, value in results.items()},
        "directed_edges": len(inputs[6]),
        "geometry_workspace_bytes": int(
            calculator.evaluator.execution_geometry_workspace_bytes
        ),
        "compact_geometry_bytes": int(calculator.evaluator.compact_edge_geometry_bytes),
        "selected_policy": calculator.evaluator.edge_geometry_policy,
    }


def _metrics(candidate, control, atom_count):
    energy_error = candidate["energy"] - control["energy"]
    force_error = candidate["forces"] - control["forces"]
    stress_error = candidate["stress"] - control["stress"]
    return {
        "energy_error_eV": float(energy_error),
        "energy_error_eV_per_atom": float(energy_error / atom_count),
        "force_max_absolute_eV_per_A": float(np.max(np.abs(force_error))),
        "force_rms_eV_per_A": float(np.sqrt(np.mean(force_error**2))),
        "stress_max_absolute_eV_per_A3": float(np.max(np.abs(stress_error))),
        "stress_rms_eV_per_A3": float(np.sqrt(np.mean(stress_error**2))),
    }


def _initial_md_atoms(size, temperature_K, seed):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((size, size, size))
    thermalize_momenta(
        atoms,
        temperature_K=temperature_K,
        rng=np.random.default_rng(seed),
        exact_temperature=False,
    )
    Stationary(atoms, preserve_temperature=True)
    return atoms


def _run_nve(model, initial_atoms, policy, neighbor_skin, steps, timestep_fs):
    atoms = copy.deepcopy(initial_atoms)
    atoms.calc = _make_mh0_calculator(model, policy, neighbor_skin)
    initial_potential = float(atoms.get_potential_energy())
    initial_total = initial_potential + float(atoms.get_kinetic_energy())
    dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs, logfile=None)
    total_energies = []
    for _ in range(steps):
        dynamics.run(1)
        total_energies.append(
            float(atoms.get_potential_energy()) + float(atoms.get_kinetic_energy())
        )
    final_total = total_energies[-1] if total_energies else initial_total
    atom_count = len(atoms)
    return {
        "selected_policy": atoms.calc.evaluator.edge_geometry_policy,
        "initial_total_energy_eV": initial_total,
        "final_total_energy_eV": final_total,
        "drift_eV_per_atom": (final_total - initial_total) / atom_count,
        "max_excursion_eV_per_atom": max(
            (abs(value - initial_total) / atom_count for value in total_energies),
            default=0.0,
        ),
        "final_temperature_K": float(atoms.get_temperature()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--neighbor-skin", type=float, default=0.0)
    parser.add_argument("--nve-steps", type=int, default=20)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.size < 1 or args.nve_steps < 0:
        parser.error("--size must be positive and --nve-steps must be nonnegative")
    if args.neighbor_skin < 0.0 or args.timestep_fs <= 0.0:
        parser.error("--neighbor-skin must be nonnegative and --timestep-fs positive")

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat(
        (args.size, args.size, args.size)
    )
    control_calculator = _make_mh0_calculator(
        args.model, CONTROL_POLICY, args.neighbor_skin
    )
    compact_calculator = _make_mh0_calculator(
        args.model, COMPACT_POLICY, args.neighbor_skin
    )
    control = _collect_static(control_calculator, atoms)
    compact = _collect_static(compact_calculator, atoms)
    if control["directed_edges"] != compact["directed_edges"]:
        raise RuntimeError("control and compact policies produced different graphs")

    initial_md_atoms = _initial_md_atoms(args.size, args.temperature_K, args.seed)
    nve = {
        "steps": args.nve_steps,
        "timestep_fs": args.timestep_fs,
        "temperature_K": args.temperature_K,
        "seed": args.seed,
        "control": _run_nve(
            args.model,
            initial_md_atoms,
            CONTROL_POLICY,
            args.neighbor_skin,
            args.nve_steps,
            args.timestep_fs,
        ),
        "compact": _run_nve(
            args.model,
            initial_md_atoms,
            COMPACT_POLICY,
            args.neighbor_skin,
            args.nve_steps,
            args.timestep_fs,
        ),
    }
    nve["absolute_drift_difference_eV_per_atom"] = abs(
        nve["compact"]["drift_eV_per_atom"] - nve["control"]["drift_eV_per_atom"]
    )

    cutoff = float(control_calculator.evaluator.r_cut)
    report = {
        "model": str(args.model.resolve()),
        "atoms": len(atoms),
        "directed_edges": control["directed_edges"],
        "model_cutoff_A": cutoff,
        "neighbor_skin_A": args.neighbor_skin,
        "effective_neighbor_cutoff_A": cutoff + args.neighbor_skin,
        "static": {
            "control_policy": control["selected_policy"],
            "compact_policy": compact["selected_policy"],
            "control_energy_eV": float(control["results"]["energy"]),
            "errors": _metrics(compact["results"], control["results"], len(atoms)),
            "control_geometry_workspace_bytes": control["geometry_workspace_bytes"],
            "compact_geometry_workspace_bytes": compact["geometry_workspace_bytes"],
            "workspace_saving_bytes": control["geometry_workspace_bytes"]
            - compact["geometry_workspace_bytes"],
            "reported_compact_geometry_bytes": compact["compact_geometry_bytes"],
        },
        "nve": nve,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
