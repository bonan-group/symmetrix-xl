# MACELES

Symmetrix-XL evaluates scalar-monopole Learned Electrostatics (LES) as part of
the fitted MACELES model. The MACE backbone, learned charge readouts, and
electrostatic term are loaded and evaluated together. The force contains the
explicit electrostatic derivative and the derivative of the learned charges
through the MACE features.

Published calculations using this evaluator should cite the original MACE
method, the LES method, the MACELES integration, and the checkpoint-specific
publication. See {doc}`/reference/publications` for the complete references.

## Convert a checkpoint

Export an upstream `MACELES` or `FoundationMACELES` checkpoint to compact JSON:

```bash
symmetrix_extract_mace \
    --model trained-mace-les.model \
    --output trained-mace-les.json \
    --radial-format compact
```

Conversion requires a MACE-Torch installation that provides
`mace.modules.extensions.MACELES` and the LES package used to create the
checkpoint. Normal JSON evaluation needs neither MACE-Torch nor PyTorch.

The JSON identifies the complete model with `model_type: "MACELES"`. Its
intrinsic `les` section uses the `symmetrix.model.mace_les` version 1 schema,
the `monopole` kind, and positive finite `sigma` and `dl` values. It also binds
the LES readouts to the canonical R0, M0, and R1 semantic fingerprints of the
MACE backbone. The exporter checks the two-interaction topology, scalar charge
layout, channel counts, readout shapes, and all three fingerprints.

## Use the ASE calculator

`Symmetrix` always evaluates the complete model:

```python
from ase import Atoms
from symmetrix import Symmetrix

atoms = Atoms(
    ["N", "Al", "N"],
    positions=[[0.1, 0.2, 0.3], [1.4, 0.4, 0.2], [0.5, 1.6, 0.7]],
    cell=[[6.2, 0.2, 0.1], [0.0, 6.0, 0.3], [0.1, 0.0, 5.8]],
    pbc=True,
)
atoms.calc = Symmetrix(
    "trained-mace-les.json",
    dtype="float64",
    streamed_edges="direct",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
les_energy = atoms.calc.results["les_energy"]
charges = atoms.calc.results["latent_charges"]
```

The calculator declares and publishes `les_energy`, `mace_les_energy`, and
`latent_charges` for every MACELES evaluation. It accepts one fully periodic
or one nonperiodic ASE structure. Mixed periodicity is outside the
scalar-monopole geometry contract.

## Energy and diagnostics

`energy`, `free_energy`, and per-atom `energies` contain the complete MACELES
result. `les_energy` is the fitted electrostatic contribution, while
`mace_les_energy` is the complete model energy before any independently added
D2 or D3 contribution. The per-atom energies sum to the complete total.

`calculator.les_diagnostics` identifies the fitted LES stage and includes its
schema metadata plus a `workspace` mapping from
`evaluator.les_charge_workspace_diagnostics`. The native evaluator's
`les_monopole_diagnostics` property separately reports reciprocal-vector counts
and monopole workspace state.
The `mace_les_parameters_sha256` and `mace_les_evaluator_fingerprint` values
bind prepared artifacts and benchmark records to the selected MACELES
parameters. `calculator.correction_diagnostics` is reserved for D2 and D3.

The native monopole diagnostics report `reciprocal_candidate_count`, the active
`reciprocal_vector_count`, `logical_extent_bytes`, `allocated_capacity_bytes`,
`active_index_capacity_bytes`, `index_pack_count`, `table_rebuilds`, and
`geometry_updates`. `les_charge_workspace_diagnostics` reports active and
capacity node counts and bytes, plus workspace replacement and reuse counts.

Add dispersion only when the fitted target does not already contain it:

```python
from symmetrix import D3Correction, Symmetrix

calculator = Symmetrix(
    "trained-mace-les.json",
    streamed_edges="direct",
    corrections=[D3Correction(functional="pbe")],
)
```

The D3 runtime is independent and contributes energy and forces once.

## Native execution

`MACELESKokkos<Precision>` is the native evaluator for the complete fitted
MACELES model. It derives from the common MACE engine so the backbone and its
R0, M0, and R1 execution artifacts remain reusable, while the learned-charge
readouts, electrostatic state, and MACELES diagnostics stay owned by the
MACELES evaluator. The template specializations and their native binding names
are implementation details; construct `Symmetrix(model, dtype=...)` to select
the model type and precision.

The LES stage runs in the same Kokkos execution space and ordered stream as the
MACE forward and reverse passes. The shared C++20/Kokkos implementation runs on
OpenMP, CUDA, and HIP. Direct execution uses the ordinary required-JIT policy
and the MACE backbone's R0, M0, and R1 artifacts.

Direct execution has this topology:

1. A cell change rebuilds the reciprocal candidate table and packs active
   reciprocal vectors in their original order. Repeated evaluations reuse the
   table and its grow-only capacity.
2. One Kokkos team per active reciprocal vector reduces over atoms to form the
   structure factors.
3. One team per atom reduces over active vectors to produce atomic energy,
   charge adjoints, and explicit force. One team per strain component performs
   the analytic reciprocal strain reduction.
4. The learned charge readouts and reverse pass remain inside the MACE
   evaluator. The charge-dependent force therefore uses the same execution
   space instance as the MACE backbone.

The reciprocal cutoff test, table values, and accumulators use double precision
for FP32 and FP64 model evaluation. Team reductions avoid atomics and reuse
backend-resident workspaces during steady-state evaluation.

`streamed_edges="reference"` evaluates the same complete model through the
generic full-graph route. Both routes publish the same output fields. The
internal periodic algorithms are named `explicit_sum` and `particle_mesh`;
they select an implementation within MACELES and do not change model identity.

## Supported boundary

| Capability | Support |
| --- | --- |
| Model | Two-interaction MACELES with scalar latent charges |
| Structures | One fully periodic or nonperiodic ASE structure |
| Precision | FP32 and FP64 |
| Native backends | CPU/OpenMP, CUDA, or HIP |
| Routing | Prepared `streamed_edges="direct"` or comparison `"reference"` |
| Additional additive terms | D2 and D3 |

MACELES does not support fixed-workspace/tiled plans, alchemical coefficients,
distributed evaluation, Metal execution, mixed periodicity, batches, or the
LAMMPS pair style. Unsupported requests fail with a MACELES capability error.

## Reproducible timing

The timing driver loads one MACELES file and measures complete warmed model
evaluations. It pins one allowed CPU, sets Kokkos, OpenMP, BLAS, MKL, BLIS, and
NumExpr thread counts before numerical imports, constructs the host neighbor
graph once, and excludes model loading, calculator setup, graph construction,
artifact compilation, and warmup from `us_per_atom`:

```bash
SYMMETRIX_BENCHMARK_THREADS=1 \
SYMMETRIX_BENCHMARK_BLAS_THREADS=1 \
SYMMETRIX_JIT_POLICY=required \
python benchmarks/mace_les_timing.py trained-mace-les.json \
    --structure structure.xyz \
    --precision float64 --pbc periodic --skin 0.0 \
    --streamed-edges direct --execution-profile speed \
    --warmups 5 --samples 20
```

The JSON record includes `model_type`, the two MACELES identity hashes,
precision, backend and Kokkos space, atom and directed-edge counts, cutoff,
skin, PBC mode, `sigma`, `dl`, affinity, native build provenance, JIT artifact,
prepared and fallback counts, graph generation, and synchronized samples. A
qualification run requires the requested routing, zero fallback evaluations,
matching source hashes, and `host_graph_constructions_during_timing` equal to
zero.

Use `benchmarks/les_particle_mesh_benchmark.py` to compare `explicit_sum` and
`particle_mesh`. Both records cover the complete MACELES evaluation and must
carry identical model, source, structure, precision, and workload identities.
On HIP, `--profile-region` exposes the synchronized samples to
`rocprofv3 --selected-regions`; on CUDA it exposes an NVTX range for Nsight
Systems.
