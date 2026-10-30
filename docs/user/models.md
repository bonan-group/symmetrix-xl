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
  edge width is not derived from the first.

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
real-space electrostatics and a system with any periodic axis uses the
direct-reciprocal solver. PME is not implemented. FP32 and FP64 are
qualified on CPU and CUDA for the released M checkpoint, including E, EF,
EFS, and dipole; isolated public stress is exactly zero. HIP is not currently
runtime-qualified. CUDA real-space
reverse automatically uses the bounded-memory `pair_atomic` strategy.
`materialized_forward` is only a forced internal experiment.

The `reference` routing uses the artifact-free backbone. The `direct`
routing loads a matching generated backbone artifact while the Polar program
remains native. The generated keep-last specialization currently recognizes
the canonical `512x0e+512x1o` final product with a scalar-only energy readout;
unsupported keep-last layouts fail explicitly. Checkpoint extraction may use
Python and PyTorch; JSON evaluation does not require Python. LAMMPS uses the
same native evaluator through `symmetrix/mace/polar/kk`, currently only with
exactly one MPI rank owning the complete system. Distributed Polar LAMMPS is
a separate future delivery.

## MACE model contracts and usage

For MH-1-style backbones, channel equality is required only within one
interaction's tied UVU paths. Different interactions may use different edge
channel counts; admission reports the failing layer and invariant when a layer
is internally inconsistent. Ordinary MH-1 still requires its scalar-final
layout. Polar keep-last models are admitted only when the final hidden,
product, retained-feature, and readout layouts match the supported scalar-plus-
vector policy. `reference` never compiles or loads a model-specific artifact.
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

Omitting `--chemical-symbols` and `--atomic-numbers` produces the preferred
universal export and retains the checkpoint's complete element domain. Compact
format v2 stores the shared radial model once with element-indexed parameters,
and format v3 uses the same full-domain principle for nonlinear MACE models.
Unlike the original Symmetrix pair-spline format (named v1 here), these formats
do not add a separate radial spline table for every element pair, avoiding
quadratic pair-table growth. A
universal file still contains the checkpoint's element-indexed learned
parameters and can therefore be larger than a deliberately restricted v2
subset, but it can be reused across compositions. Use an explicit element
subset only for an intentionally restricted format-v2 deployment or when
producing the original Symmetrix format with `--radial-format pair-splines`;
format-v3 models reject subsets.

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

## NequIP and OAM

NequIP models use a separate native contract because their stacked edge
convolutions and gates are not MACE layers. Export a supported model to the
versioned JSON contract, then use the explicit calculator. The exporter is an
optional workflow and needs the upstream reference stack in its own
environment:

```bash
uv pip install "symmetrix-xl[nequip]"   # torch, nequip, and e3nn
symmetrix_extract_nequip \
    --model NequIP-OAM-S-0.1.nequip.zip \
    --output oam-s-nequip.json
```

The OAM JSON contract is `symmetrix_nequip_format_version=2`. It carries a
universal H through Pu element mapping, the model cutoff, ordered edge
convolution layers, and precision-independent parameters. Native evaluation
needs only the exported JSON, ASE, and a Symmetrix native extension; PyTorch,
NequIP, and e3nn are not imported by the runtime. Use `dtype="float32"` for
the initial qualification path or request `dtype="float64"` when the exported
contract and backend support it.

```python
from symmetrix import NequIPCalculator

atoms.calc = NequIPCalculator("oam-s-nequip.json", dtype="float32")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # ASE Voigt order: xx, yy, zz, yz, xz, xy
```

OAM-S, OAM-M, OAM-L, and OAM-XL are implemented only through the explicit
OAM-v2 contract below; this does not imply support for arbitrary NequIP graphs.
The OAM evaluator supports energy, forces, and stress on its `direct` routing;
the artifact-free `reference` routing does not expose stress. Stress uses
periodic directed-edge derivatives, with
reduction, symmetrization, and volume scaling on the execution device. CUDA returns only the final nine-double
tensor. A stress request after cached energy/forces reuses the completed graph.

The current full-message-only host ABI-v7 and CUDA ABI-v3 artifacts are
source-buildable in FP32 and FP64. Fresh correctness, performance, capacity,
and repaired-wheel gates are required before a release matrix or wheel contents
are claimed. Live upstream FP64 parity remains an open qualification gate.
HIP, parameter gradients, LAMMPS/MPI, multi-GPU, and fixed-workspace
OAM execution are unsupported.

See {doc}`../reference/execution_support_matrix` for current release gates and
{doc}`../reference/publications` for the NequIP and OAM publications to cite.


OAM `direct` execution requires a validated generated artifact. Host execution
uses a precision-matched host plugin; CUDA `direct` execution uses an
NVRTC-first cubin or an explicitly requested nvcc shared library, prepared
automatically by `NequIPCalculator`. Both loaders bind artifacts to the resolved
architecture, precision, ABI, and generation; CUDA also validates the device target and
embedded binary identity. A missing or incompatible artifact is an explicit
error, not a generic edge fallback. Set `SYMMETRIX_JIT_CACHE` to a cache
and `SYMMETRIX_JIT_POLICY=required` for qualification. HIP is not supported.
Source generation supports exact host plugins or CUDA cubins for valid OAM
architectures in FP32 and FP64. The current full-message-only artifacts still
require the fresh runtime qualification matrix and repaired-wheel gates before
wheel inclusion or release support is claimed.
Selection compares the
model's explicit architecture: ordered layer count; source, edge, and output
irreps (therefore channel widths, angular orders, and parity); tensor-product
instructions and output masks; radial spline dimensions; and normalization.
The input filename, archive path, learned tensors, and whole-model hash are not
selection keys, so retrained weights with the same generated layout reuse the
artifact. A custom layout with no exact match is generated during preparation
and stored in the runtime JIT cache when a matching compiler is available. The
installed catalog follows the same exact-match approach as MACE and is a
time-to-first-energy optimization, not a correctness fallback.
For host OpenMP/OpenBLAS setup and runtime checks, see
{doc}`../developer/build_and_test`.

On a host backend, the default `direct` mode requires the receiver-owned ABI v7
full-message artifact. One generated callback owns one receiver, traverses its
directed edges, evaluates certified cubic spline values inline, accumulates the
complete generated message, and writes the receiver row once; the packed
post-linear is applied afterward. Reverse dispatch is source-owned, rebuilds
the full-message adjoint once per node, writes each directed edge's harmonic
and radial adjoints from its unique owner, and needs no source atomics. CUDA
uses the corresponding full-message-only ABI v3 artifact. Direct execution
allocates no gathered source rows, edge-feature rows, radial-weight rows, or
edge-message rows. `debug_v1` and `mlp_reference` are
retained spellings that resolve to the single `reference` routing, which runs
the compiler-free Kokkos edge product in the selected execution space through
the whole-evaluator entry point
with no artifact at all; it is intended as the reference for parity tests and
contract diagnosis, not as a production mode. Its radial semantics are the
certified spline table, not an exact-MLP evaluation; the exact-MLP oracle is
retained independently by the radial certification fixtures. The host
node-local self-connection tensor product uses host-launched `KokkosBlas`
GEMMs; CUDA uses the existing generic definition-order self-connection path.

Unsupported NequIP modules fail during export with a module-qualified
diagnostic. The native runtime accepts exported JSON only: original `.nequip`
and `.nequip.zip` packages, PyTorch checkpoints, and AOTInductor `.pt2`
artifacts remain upstream reference inputs and are not dispatched implicitly.

The generic edge tensor-product evaluator is retained only as a debug oracle
for contract diagnosis, not as an OAM direct-execution fallback. This is
distinct from the node-local self-connection contraction used on CUDA.
