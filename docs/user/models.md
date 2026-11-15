# Models and Properties

- **Standard MACE**, including MACE-MH-0, supports energy, free energy,
  per-atom energies, forces, and stress through the ordinary direct execution
  planner.
- **MACEField** adds polarization and qualified analytical response properties.
- **MACELES** combines a two-interaction MACE backbone, learned scalar charge
  readouts, and monopole electrostatics in one fitted model. It supports energy,
  free energy, per-atom energies, forces, stress, `les_energy`,
  `mace_les_energy`, and `latent_charges`.
- **MACE-MH-1** supports generated direct execution for admitted nonlinear
  contracts and an artifact-free optimized `reference` route. Each interaction
  declares and validates its own tied UVU channel multiplicity; the second
  edge width is not derived from the first. A model trained with
  `apply_cutoff=True` (mace-torch's default) runs on the `reference` route,
  which the default and `auto` select: its edge functions do not vanish at
  `r_max`, and an explicit `direct` refuses it. `direct` serves models
  trained with `apply_cutoff=False`, such as MACE-MH-1.

## MACE-POLAR-1

**MACE-Polar** (`MACE_Polar`, the MACE-POLAR-1 checkpoints) adds
polarisable electrostatics on the same neutral backbone engine as MH-1, but
it is a separate public evaluator. Total charge, total spin, Fermi level,
and the external electric field are graph inputs; energy, forces, stress,
and total dipole are public outputs. The family-owned native transaction
performs the backbone forward, Polar recursion and electrostatics, seeded
backbone reverse, and final result composition. Python supplies compact
inputs and receives compact public outputs only.

One solver-neutral JSON serves both boundary conditions. With
`polar_electrostatics="auto"`, a fully non-periodic system uses exact
real-space electrostatics. Periodic inputs use direct-reciprocal electrostatics
unless the backend, FFT provider, multipole order, and fully periodic system
size admit automatic PME selection. PME implementation and automatic selection
exist, but PME remains outside the qualified public release contract. Select
`polar_electrostatics="direct_reciprocal"` explicitly to reproduce the strict
periodic frozen-oracle configuration. FP32 and FP64 are
qualified on CPU and CUDA for the released M checkpoint, including E, EF,
EFS, and dipole; isolated public stress is exactly zero. HIP `gfx1151` is
runtime-qualified from source builds only for Polar S/M/L in FP32/FP64, both
`reference` and `direct`, on single-GPU molecule, slab, and bulk energy,
forces, and stress workloads. HIP wheels, other HIP architectures,
multi-GPU/distributed execution, and PME publication remain excluded.
CUDA real-space reverse automatically uses the bounded-memory
`pair_atomic` strategy.
`materialized_forward` is only a forced internal experiment.

Strict frozen-oracle comparisons use `num_spline_points=4096` when loading a
checkpoint, or `--num-spline-points 4096` when exporting JSON. This controls the
backbone's radial approximation, not the electrostatic solver. The ordinary
256-point default is unchanged; converge this resolution against the accuracy
required by the calculation rather than assuming it meets the oracle tests'
tight force and stress tolerances.

In recorded CPU FP64 singlet checks for the S/M checkpoints (molecule, slab,
and bulk, with direct-reciprocal electrostatics), the default 256 points gave
force errors up to `7.4e-4 eV/Angstrom` and an M bulk dipole error of
`2.3e-6 e Angstrom`. Both reference and direct exceeded the corresponding
strict frozen bounds (`5e-6 eV/Angstrom` and `1e-6 e Angstrom`). Explicit
4096-point sampling met those bounds for the measured cases; this is not a
universal convergence guarantee for other models or configurations. Selecting
FP64 alone does not remove radial interpolation error.

The `reference` routing uses the artifact-free backbone. The `direct`
routing loads a matching generated backbone artifact while the Polar program
remains native. Generated keep-last execution recognizes the single-layer
scalar-only S topology with a linear energy readout, and the canonical
`512x0e+512x1o` final product with a scalar-only nonlinear energy readout.
Unsupported keep-last layouts fail explicitly. Checkpoint extraction may use
Python and PyTorch; JSON evaluation does not require Python. LAMMPS uses the
same native evaluator through `symmetrix/mace/polar/kk`, currently only with
exactly one MPI rank owning the complete system. Distributed Polar LAMMPS is
a separate future delivery.

The scalar-only, one-interaction S backbone supports both native `reference`
execution and specialized generated `direct` execution, including its linear
energy readout, retained features, and the Polar chain's seeded reverse.
Focused CPU, CUDA sm120, and HIP gfx1151 checks cover FP32/FP64, released-S
frozen comparisons, and synthetic force/stress derivatives. HIP evidence is
source-build/runtime evidence only; other HIP architectures, multi-GPU/
distributed execution, PME publication, and distributed Polar LAMMPS remain
excluded.

## MACE model contracts and usage

For MH-1-style backbones, channel equality is required only within one
interaction's tied UVU paths. Different interactions may use different edge
channel counts; admission reports the failing layer and invariant when a layer
is internally inconsistent. Ordinary MH-1 still requires its scalar-final
layout. Polar generated keep-last admission requires consistent hidden,
product, retained-feature, and readout layouts under the scalar-only S policy
or the supported scalar-plus-vector policy. `reference` never compiles or
loads a model-specific artifact.
`direct` requires a contract-matched artifact and refuses a missing, stale, or
unrecognized keep-last schedule instead of falling back to `reference`.

MACE-MH-1 direct execution is compositional. Each non-final layer's symmetric
product uses the canonical M0 operator shared with standard MACE, while the
final scalar layer uses canonical M1. MH-1 retains its model-specific density
normalization, conditioned messages, gates, product linear, skip, readout,
tensor-product work, and reverse stages.

Artifact provenance is per component: the full MH-1 plugin and every product
layer M0/M1 module independently report `prebuilt`, `cache`, or `runtime`
origin. An installed M module can be reused by both model families without
implying that the full MH-1 program was prebuilt. Device composition uses module
ABI v5/plan v6/JIT generation 19; the host plugin uses ABI v7.

Compact Symmetrix-XL JSON is the preferred interchange format. Loading an
ordinary MACE checkpoint can require the optional `mace-torch` dependency.
MACEField checkpoints should be exported to compact JSON because the
field-aware upstream modules are not yet present in upstream MACE.

Convert a local ordinary MACE checkpoint after installing the optional
converter dependency:

```bash
uv pip install mace-torch
symmetrix_extract_mace \
    --model mace-omat-0-medium.model \
    --output srtio3-mace.json
```

For FP64 MACE-OMAT-0 stress calculations, add `--num-spline-points 512`.
The ordinary 256-point default is faster to export and remains suitable when
its larger compact-radial interpolation error is acceptable.

Metatrain MACE models can be converted from either the training checkpoint or
the standalone metatomic deployment model. Install the additional integration
dependencies, then pass the `.ckpt` or `.pt` file to the same command:

```bash
uv pip install "symmetrix-xl[metatrain]"
symmetrix_extract_mace --model model.ckpt --output model.json
symmetrix_extract_mace --model model.pt --output model-from-deployment.json
```

For a metatrain model, the converter restores the scalar energy scaling and
per-element composition baseline that metatrain stores outside its embedded
MACE backbone. For an exported `.pt`, it reconstructs an eager MACE module from
the scripted topology and validates the learned state before extraction.
Standard MACE energy heads are supported; arbitrary metatrain heads, extra
embedding inputs, and non-composition additive models cannot be represented by
the compact MACE schema and fail explicitly. Models must use scalar energy
targets in eV and Angstrom coordinates; empty unit metadata from an imported
native MACE model is interpreted using those conventional MACE units. Exported
nonlinear readouts must use the SiLU gate.

Omitting `--chemical-symbols` and `--atomic-numbers` produces the preferred
universal export and retains the checkpoint's complete element domain. The
compact radial representation stores the shared radial model once with
element-indexed parameters, and nonlinear MACE models use the same full-domain
principle. Unlike pair-spline radial tables (`--radial-format pair-splines`),
it does not add a separate radial spline table for every element pair, avoiding
quadratic pair-table growth. A
universal file still contains the checkpoint's element-indexed learned
parameters and can therefore be larger than a deliberately restricted
subset, but it can be reused across compositions. Use an explicit element
subset only for an intentionally restricted deployment or with
`--radial-format pair-splines`; nonlinear MACE models reject subsets.

Every export is written in format version 4: the first readout and the
H1->H2 skip are stored without the inverse of `interactions[1].linear_up`
(`h1_affine_mode` `m0_composite`, or `pre_linear_up` for first-interaction
residual and MACEField models). Format-1/2 files carry that inverse; they still
load with the old arithmetic and print a warning with an fp32 cancellation
estimate computed from the file. Re-export to v4 is recommended but optional.
Convert them without the checkpoint, or re-export:

```bash
symmetrix_recompose_mace_json --input old.json --output new.json
```


The command retains all compatible prediction heads. It does not download a
model; obtain a checkpoint from its model provider, then use the resulting JSON
in the calculator. For a GPU LAMMPS run, add
`--prepare-jit-device-artifact` during conversion on the deployment GPU.
`mace-torch` and its PyTorch dependencies are optional and can be a large
download; JSON-only evaluation does not require them.

An upstream `MACELES` or `FoundationMACELES` checkpoint is converted with the
same command. The exported JSON has `model_type: "MACELES"` and an intrinsic
`les` section. Every `Symmetrix` calculation evaluates the complete fitted
model:

```python
calculator = Symmetrix("trained-mace-les.json", streamed_edges="direct")
energy = calculator.get_potential_energy(atoms)
les_energy = calculator.results["les_energy"]
latent_charges = calculator.results["latent_charges"]
```

The `reference` routing evaluates the same model with the generic full-graph
implementation. MACELES requires exactly two interactions and supports fully
periodic or nonperiodic structures. Mixed periodicity, fixed-workspace plans,
alchemical coefficients, distributed evaluation, Metal, and LAMMPS are outside
its supported boundary. See {doc}`/mace_les` for schema, diagnostics, and
qualification details.

One-interaction MACE models using the invariant correlation-2 nonlinear
readout are supported with canonical equal-channel hidden irreps. This includes
both the `L_max=1` single-v2 layout and the earlier `L_max=2` single-layer
layout. Their compact JSON contains only the M0 and R0 execution contracts.

The exporter refuses standard-MACE options that the JSON cannot represent and
names the option: a model without `scale_shift` (plain `MACE`),
`embedding_specs`, `use_agnostic_product`, `use_last_readout_only`, a readout
other than MACE's linear and non-linear readout blocks (such as
`NonLinearBiasReadoutBlock`), and a non-linear readout gate other than SiLU.
Each interaction's product may use correlation order 1, 2 or 3, also mixed
per layer (for example `correlation=[3, 2]`); higher orders are refused. A
multi-head model needs the same order in every layer: selecting a head with
MACE's `remove_pt_head` rebuilds every product with the first one's order, so
mixed orders are refused there.
Pair-spline tables (`--radial-format pair-splines`) of a model trained with
`apply_cutoff=False` include the polynomial cutoff that MACE applies in its
interaction blocks, so they vanish at the cutoff as in the compact format.

The installed generated-operator catalog covers the verified foundation-model
contracts, not every valid compact model. Aliases of a verified model share its
canonical artifact identity. A different channel width, angular cutoff,
operator structure, exact backend target, or weighted-edge/alchemical contract
has a different identity and is generated at runtime, then reused from the JIT
cache. The model remains supported when a local compiler is available; absence
from the installed catalog is not a request to load a merely similar binary.

Multi-head JSON models retain every head. Select one with `head=` and inspect
`calculator.available_heads`. `SymmetrixEnsemble` provides same-process,
sequential committee evaluation; it returns mean values, member values with
the `_comm` suffix, and population variance with the `_var` suffix.

An ordinary ASE calculation can request the standard properties directly:

```python
atoms.calc = Symmetrix("srtio3-mace.json")  # FP32 by default
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

Use `Symmetrix("srtio3-mace.json", dtype="float64")` when higher numerical
precision is required.

Standard two-interaction compact MACE models also support training-free
alchemical message mixing. The usual element-wide form keeps the input
structure unchanged and selects every reference atom during evaluation:

```python
calculator = Symmetrix(
    "mace-omat-0.json",
    alchemical={"Sr": {"Sr": 0.5, "Ba": 0.5}},
)
```

`ElementMixture` exposes the normalized request for fraction updates, while
`AlchemicalConfig` remains available for advanced index-based topologies.
Foundation parameters and concrete virtual species are unchanged. Global
virtual-copy slots carry finite nonnegative coefficients, with physical-atom
energy/force reduction and stress support. This requires the prepared
`streamed_edges="direct"` routing and matching weight-aware artifacts; it does
not support the optional model features listed in the
[Alchemical Message Mixing](../reference/python_api.md#alchemical-message-mixing)
API section.

See {doc}`/reference/python_api` for the public calculator and ensemble API.

For host OpenMP/OpenBLAS setup and runtime checks, see
{doc}`../developer/build_and_test`.

The generic edge tensor-product evaluator is retained only as a debug oracle
for contract diagnosis, not as a direct-execution fallback. This is
distinct from the node-local self-connection contraction used on CUDA.
