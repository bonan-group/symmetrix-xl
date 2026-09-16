"""Profile phase boundaries in a MACE-OMAT-0 Symmetrix MD trajectory."""

import argparse
import gc
import hashlib
import json
import pathlib
import statistics
import time

import numpy as np
import torch
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix


def timed_summary(samples):
    return {
        "count": len(samples),
        "total_ms": sum(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


class ProfiledSymmetrix(Symmetrix):
    def __init__(self, *args, **kwargs):
        self.synchronize_phases = kwargs.pop("synchronize_phases", True)
        self.fingerprint_graphs = kwargs.pop("fingerprint_graphs", True)
        self.profile_samples = {
            "mace_inputs": [],
            "compute_mace": [],
            "collect_results": [],
        }
        self.native_samples = []
        self.graph_samples = []
        super().__init__(*args, **kwargs)

    def _time_phase(self, name, operation, *args, **kwargs):
        torch.cuda.nvtx.range_push(name)
        start = time.perf_counter()
        try:
            return operation(*args, **kwargs)
        finally:
            if self.synchronize_phases:
                torch.cuda.synchronize()
            self.profile_samples[name].append(1000.0 * (time.perf_counter() - start))
            torch.cuda.nvtx.range_pop()

    def _mace_inputs(self, atoms, native_geometry=False):
        result = self._time_phase(
            "mace_inputs",
            super()._mace_inputs,
            atoms,
            native_geometry=native_geometry,
        )
        _, node_types, num_neigh, j_list, neigh_types, _, _, i_list = result
        if not self.fingerprint_graphs:
            self.graph_samples.append({"edges": len(j_list)})
            return result
        ordered_digest = hashlib.sha256()
        for values in (node_types, num_neigh, j_list, neigh_types):
            ordered_digest.update(np.asarray(values, dtype=np.int64).tobytes())
        canonical_edges = np.column_stack((i_list, j_list, neigh_types))
        canonical_edges = canonical_edges[
            np.lexsort(tuple(canonical_edges[:, axis] for axis in (2, 1, 0)))
        ]
        self.graph_samples.append(
            {
                "edges": len(j_list),
                "ordered_topology_sha256": ordered_digest.hexdigest(),
                "canonical_edge_sha256": hashlib.sha256(
                    canonical_edges.astype(np.int64, copy=False).tobytes()
                ).hexdigest(),
            }
        )
        return result

    def _compute_mace(self, mace_inputs, atoms=None):
        result = self._time_phase(
            "compute_mace", super()._compute_mace, mace_inputs, atoms=atoms
        )
        self.native_samples.append(
            {
                "graph_prepare_ms": float(
                    self.evaluator.factorized_last_graph_prepare_ms
                ),
                "evaluation_ms": float(self.evaluator.factorized_last_evaluation_ms),
                "graph_generation": int(self.evaluator.factorized_graph_generation),
                "schedule_build_count": int(
                    self.evaluator.factorized_schedule_build_count
                ),
                "prepared_evaluation_count": int(
                    self.evaluator.factorized_prepared_evaluation_count
                ),
                "prepared_geometry_update_count": int(
                    self.evaluator.factorized_prepared_geometry_update_count
                ),
                "geometry_copy_count": int(
                    self.evaluator.execution_geometry_copy_count
                ),
                "all_interactions_graph_generation": int(
                    getattr(self.evaluator, "all_interactions_graph_generation", 0)
                ),
                "all_interactions_prepared_graph_count": int(
                    getattr(self.evaluator, "all_interactions_prepared_graph_count", 0)
                ),
                "all_interactions_prepared_evaluation_count": int(
                    getattr(
                        self.evaluator, "all_interactions_prepared_evaluation_count", 0
                    )
                ),
                "all_interactions_geometry_update_count": int(
                    getattr(self.evaluator, "all_interactions_geometry_update_count", 0)
                ),
                "all_interactions_schedule_build_count": int(
                    getattr(self.evaluator, "all_interactions_schedule_build_count", 0)
                ),
                "all_interactions_active_edge_count": int(
                    getattr(self.evaluator, "all_interactions_active_edge_count", 0)
                ),
            }
        )
        return result

    def _collect_mace_results(
        self, atoms, mace_inputs, properties=None, factorized_graph_generation=0
    ):
        return self._time_phase(
            "collect_results",
            super()._collect_mace_results,
            atoms,
            mace_inputs,
            properties,
            factorized_graph_generation,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--streamed-edges",
        choices=("all_interactions", "factorized"),
        default="factorized",
    )
    parser.add_argument(
        "--production-boundaries",
        action="store_true",
        help=(
            "synchronize only complete MD steps; phase wall times become host "
            "submission intervals while NVTX GPU projection remains authoritative"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        parser.error("this profiler requires a CUDA Symmetrix build")

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982).repeat((args.repeat,) * 3)
    rng = np.random.default_rng(20260808)
    atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=rng)
    Stationary(atoms, preserve_temperature=True)

    calculator = ProfiledSymmetrix(
        args.model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges=args.streamed_edges,
        jit="required" if args.streamed_edges == "factorized" else "off",
        synchronize_phases=not args.production_boundaries,
        fingerprint_graphs=not args.production_boundaries,
    )
    atoms.calc = calculator
    atoms.get_forces()
    for samples in calculator.profile_samples.values():
        samples.clear()
    calculator.native_samples.clear()
    calculator.graph_samples.clear()

    dynamics = VelocityVerlet(atoms, timestep=units.fs, logfile=None)
    step_samples = []
    for step in range(args.steps):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("velocity_verlet_step")
        start = time.perf_counter()
        dynamics.run(1)
        torch.cuda.synchronize()
        step_samples.append(1000.0 * (time.perf_counter() - start))
        torch.cuda.nvtx.range_pop()

    report = {
        "atoms": len(atoms),
        "steps": args.steps,
        "profiling": {
            "phase_synchronization": calculator.synchronize_phases,
            "graph_fingerprinting": calculator.fingerprint_graphs,
            "phase_timing_semantics": (
                "synchronized additive wall time"
                if calculator.synchronize_phases
                else "host submission wall time; use NVTX GPU projection for kernels"
            ),
            "step_timing_semantics": "synchronized complete VelocityVerlet step",
        },
        "streamed_edges": calculator.streamed_edges,
        "jit": {
            "status": calculator.jit_status,
            "compiler_backend": calculator.jit_compiler_backend,
            "artifact_id": calculator.jit_artifact_id,
        },
        "timing": {
            "velocity_verlet_step": timed_summary(step_samples),
            **{
                name: timed_summary(samples)
                for name, samples in calculator.profile_samples.items()
            },
        },
        "native_calls": calculator.native_samples,
        "graphs": calculator.graph_samples,
    }
    atoms.calc = None
    del calculator
    gc.collect()
    if native_symmetrix._kokkos_is_initialized():
        native_symmetrix._finalize_kokkos()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
