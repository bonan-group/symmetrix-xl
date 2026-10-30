# Python API

This reference is generated from the public source contract and is deliberately
kept independent of a native extension, so documentation can render without a
particular CPU/GPU backend installed.

## Frontend and Backend Selection

`symmetrix.available_backends()` returns installed backend descriptors without
loading their native extensions or probing whether their target devices are
available. Use `symmetrix backend list` for a usability inventory.
`symmetrix.load_backend(request=None)` loads the automatic or named backend and
permanently selects it for the process. `symmetrix.selected_backend()` exposes
the resolved descriptor after selection.

Select a backend before loading a calculator or native symbol.

## Symmetrix

```python
Symmetrix(
    model_file, dtype="float32",
    streamed_edges="direct", execution_profile="capacity",
    allow_fixed_workspace=False,
    low_memory=<compatibility switch>, jit=None,
    kernel_launch_policy="automatic", m1_polynomial_policy="automatic",
    execution_mh1_scratch_budget_bytes=None,
    execution_mh1_node_state_policy="full-retention-v1",
    execution_mh1_node_arena_policy="throughput-v1",
    execution_mh0_state_policy="full-retention-v1",
    edge_geometry_policy="cartesian-f64-v1",
    neighbor_skin=0.5, head=None, alchemical=None,
    polar_electrostatics="auto", total_charge=None, total_spin=None,
    fermi_level=None, electric_field=None,
    corrections=None,
    dispersion=False, dispersion_damping="bj", dispersion_xc="pbe",
    dispersion_cutoff=None, dispersion_device=None, metal=False,
    **ase_kwargs,
)
```

## NequIPCalculator (OAM v2)

```python
NequIPCalculator(
    model_file, *, dtype="float32", neighbor_skin=0.5,
    streamed_edges="direct", head=None, **ase_kwargs,
)
```

`NequIPCalculator` accepts exported OAM v2 JSON only. The `direct` routing
requires a validated, architecture-bound full-message-only host ABI-v7 or CUDA
ABI-v3 artifact and exposes energy, forces, and stress; `reference` is
artifact-free and exposes energy and forces only. Source generation supports
FP32 and FP64 on host and CUDA. Fresh runtime and repaired-wheel qualification
is pending after the artifact-contract change. HIP,
multi-GPU execution, and arbitrary NequIP graphs are outside the implementation
boundary.
PyTorch, NequIP, and e3nn are used by the optional
exporter/reference workflow and are not runtime imports for JSON-only
evaluation. See {doc}`publications` for citation guidance.

`Symmetrix` is an ASE calculator. `dtype` defaults to `"float32"` and also
accepts `"float64"`. Select float64 explicitly for high-precision calculations.
`streamed_edges` accepts the routing `direct` (default) or `reference`, plus
the temporary compatibility request `auto`. Every documented spelling resolves
through one declared vocabulary, so the accepted spellings have a single
source: `non-compiled`, `generic`, `all_interactions`, and `materialized` all
resolve to `reference`, and `factorized` and `direct_streamed` are
compatibility aliases for `direct`. In the Python frontend the compatibility
aliases preserve the former throughput behavior by changing an
otherwise-`capacity` request to the `speed` profile. Use literal `direct` for
capacity behavior. The removed `receiver_factorized` selector is rejected.
`NequIPCalculator` is a separate frontend that accepts only its own literal
set, `direct`, `reference`, `debug_v1`, and `mlp_reference`, of which
`debug_v1` and `mlp_reference` resolve to `reference`; no MACE alias resolves
here. It reports the same `streamed_edges`, `streamed_edges_requested`, and
`streamed_edges_alias` triple as the compact frontend, and it reads its
accepted spellings from the same declaration, so the rejection message lists
the family's own set.

`direct` runs the prepared plan against a contract-bound generated (T3)
artifact and fails at graph preparation when none is qualified. `reference` is
single-thread execution of the generic implementation on the default host
execution space, needs no artifact, and is the routing the removed standalone
CPU-native evaluators were consolidated onto. The third positional argument of
`Symmetrix` and the `use_kokkos` keyword of both frontends are retired and
rejected explicitly.

Both frontends prepare the artifact a `direct` request executes through from
the family's declared adapter for the target the build binds, and both read the
family's declared evaluator surface rather than probing the evaluator.
`NequIPCalculator` also surfaces the family's own capability query as
`calculator.capabilities`. A preparation that cannot produce a qualified
artifact fails there — before any evaluator is constructed — and reports the
same four-way diagnostic (empty cache with a working compiler, no compiler, a
missing prebuilt artifact, or a cached artifact rejected as stale) for every
family.

Both frontends resolve the model family and its native evaluator class through
the declared family registry
(`symmetrix/source/symmetrix/family_registry.py`) instead of branching on the
model type or architecture string. Each family declares one entry beside its
own modules — `mace_family.py` for the compact models and their field variant,
`mace_nonlinear_family.py` for the nonlinear models, `mace_polar_family.py` for
the polarisable long-range models, and `oam_family.py` for
OAM — and that entry owns the family's detection keys, evaluator variants per
precision and execution space, contract route, its schema, extractor,
generator, and artifact-adapter declarations, and the evaluator surface each
execution mode reads. The execution-mode vocabulary is not an entry's: the two
compact families share `family_registry.COMPACT_MODE_VOCABULARY`, declared
beside the entry data model the declarations are written against and read by
the calculator as `_MODE_VOCABULARY`, while OAM declares its own. A mode that
needs a prepared-path method the evaluator does not provide fails naming the
family, the capability, and the method; no request silently resolves to a
slower path.
`execution_profile` is `capacity` (default) or `speed`;
the compatibility `low_memory=True` and `low_memory=False` spellings map to
those profiles.
`allow_fixed_workspace=True` separately permits bounded tiled workspace plans
for qualified CUDA FP32 direct capacity execution. It defaults to `False` and
does not force the planner to select such a plan.

`direct` is the first-class performance path. Two-interaction models require a
matching admitted R1 artifact; admitted single-layer models run without R1
specialization, although capacity planning may still compile or load M0/R0
operator modules. `non-compiled` is an explicit compiler-free debug mode; it is
not optimized and has no performance guarantee. The `jit` argument is
deprecated and ignored;
`SYMMETRIX_JIT_POLICY` controls required specialization or the explicit
diagnostic-disabled policy.

Pass `corrections=[D2Correction(...)]`,
`corrections=[D3Correction(...)]`, or
`corrections=[FourierD3Correction(...)]` to add one native dispersion correction.
`D2Correction(functional="pbe", cutoff=40 * ase.units.Bohr, skin=0.5)` uses
fixed coefficients and zero damping. `D3Correction` additionally accepts
`damping` (`zero`, `bj`, `zerom`, or `bjm`) and
`coordination_cutoff`. All lengths are Angstrom. Both corrections contribute
energy, free energy, per-atom energies, forces, and ASE-order stress through
the loaded Kokkos backend. They require no optional Python package; the empty
`dispersion` extra exists only for installation compatibility.

`FourierD3Correction` is the fully periodic reciprocal-space modified D3(BJ)
model. It accepts either `mesh_spacing` (default 1.0 Angstrom) or an explicit
three-axis `mesh`, plus `spline_order`, `decomposition_tolerance`, optional
`rank_cap`, `chunk_size`, `coordination_cutoff`, and `skin`. It requires a
provider-capable KokkosFFT build and has no real-space D3 fallback. See the
user guide for the model distinction, geometry gate, mesh convergence, and
backend qualification boundary.

`dispersion=True` is deprecated and translates to one `D3Correction` with the
legacy damping, functional, and cutoff arguments, emitting `FutureWarning`.
It cannot be combined with `corrections`. `dispersion_device` checks that a
legacy request matches the already-loaded backend and never transfers work to
another device. See {doc}`/user/dispersion_corrections` for graph ownership,
neighbor policy, units, diagnostics, and unsupported method families.

Ordinary models implement `energy`, `free_energy`, `energies`, `forces`, and
`stress`. MACEField models additionally expose `polarization`, `becs`, and
`polarizability` when supported by the model and requested calculation.

MACE Polar is a separate public native evaluator over the same non-public
backbone engine; it does not invoke `MACENonlinearKokkos`. Its family adapter
passes compact graph inputs to one native whole-model transaction and receives
energy, free energy, forces, stress, and dipole. Retained features, product
seeds, electrostatic tables, and adjoints remain native and device-resident.
Explicit Polar-only options are rejected for other model families.

`polar_electrostatics="auto"` uses exact real space for a fully non-periodic
single system and direct reciprocal space when any PBC axis is true. Explicit
`realspace`, `periodic`, and `direct_reciprocal` requests are available for
validation; `pme` is recognized but rejected because PME/PPPM is deferred. The
exported JSON carries no solver or reverse-strategy selection. FP32 and FP64 E,
EF, EFS, and dipole are qualified on CPU and CUDA for the released M
checkpoint; isolated public stress is exactly zero. Mixed-system batches are
not supported.

Checkpoint extraction may require Python and upstream MACE. Evaluating an
exported MACE Polar JSON is owned by the native family evaluator; the Python
calculator is only an ASE adapter. The same native transaction is available to
the compact C++ consumer and the qualified one-rank/full-system LAMMPS pair
style described in {doc}`/user/lammps`.

With a positive `neighbor_skin`, fully periodic `direct` Kokkos calculations
construct and retain the candidate neighbor graph in native Kokkos memory.
The ASE layer keeps only node-sized state for Verlet validity checks. Energy,
forces, stress, polarization, polarizability, and MACEField BECs use this path;
host edge arrays are retained only by the legacy pair-derivative BEC reduction,
which evaluators exposing the device-bins reduction no longer need. The
diagnostic environment variable `SYMMETRIX_NEIGHBOR_BACKEND=host|kokkos` pins
either implementation. On CUDA and HIP, the default `automatic` selects the host
builder below 512 atoms and Kokkos at 512 atoms or above when native
construction is eligible. Serial and OpenMP use the host builder by default.
The calculator attribute `neighbor_graph_backend` reports the implementation
that built the current cached graph.

Both frontends build their host graph through one neutral builder,
`symmetrix/source/symmetrix/neighbor_graph.py`, so the matscipy fast path, the
32-bit cardinality guard, the receiver-major edge order, and the skin-reuse
rule cannot differ between families. Its graph is byte-identical for identical
inputs. Install `symmetrix-xl[neighbors]` to enable the matscipy provider. The
standard installation retains the ASE provider as a correct fallback, but its
graph construction can be substantially slower for large or long-cutoff
systems. Matscipy accepts any finite, invertible cell and honours periodicity
axis by axis, including vacuum-padded slabs and finite non-periodic boxes.

Useful run-time attributes include `execution_plan`, `jit_status`,
`jit_reason`, `jit_artifact_id`, `jit_variant_id`, `low_memory_policy`, `head`,
and `available_heads`.

`execution_plan` reports the native selection for the prepared graph: the
requested algorithm and profile, the selection source and reason, the advisory
memory budget, and the candidate list. The candidates are the plan space the
model family declares, in declaration order, each with its plan id, whether it
is qualified, its estimated graph bytes, and the reason for that verdict. A
family that declares no plan space reports the single plan it selected as one
implicit candidate instead of an empty list. The plan ids are the same
cross-language contract the private `_debug_execution_plan` argument accepts.
The report also carries the per-operator tier binding: `binding_report_version`
and `bindings`, one entry per operator node with the bound implementation id,
its tier, its target and precision policy, its identity fingerprint, whether an
implementation below the generated tier serves it (`portable_covered`), and
whether the artifact is therefore required (`artifact_required`). Version 0 with
an empty list is the pre-binding state a report carries while the prepared
direct path is not the active mode.

`NequIPCalculator` exposes the same `execution_plan` report for its compiled
program: `binding_report_version` is 1, `bindings` carries one entry per
operator node, and `selected_id` names the implementation the execution path
dispatches, which the execution diagnostic also reports. A family that
declares no plan candidate space reports the single implementation it selected
as its one implicit candidate, so the report is never an empty plan.

## Alchemical Message Mixing

Pass one reference-element mapping to apply a shared composition to every atom
of that element. The calculator copies the mapping into an `ElementMixture`;
the physical `Atoms.numbers` remain unchanged and continue to describe the
reference structure.

```python
from symmetrix import Symmetrix

calculator = Symmetrix(
    "mace-omat-0.json",
    alchemical={"Sr": {"Sr": 0.5, "Ba": 0.5}},
    dtype="float64",
)
atoms.calc = calculator
energy = atoms.get_potential_energy()
calculator.alchemical.set_fractions({"Sr": 0.6, "Ba": 0.4})
updated_energy = atoms.get_potential_energy()
```

Element symbols or atomic numbers are accepted. The outer mapping must contain
exactly one reference element; its inner mapping must contain at least two
finite, nonnegative fractions that sum to one. Resolution happens against the
current structure when its prepared topology is built, so supercells and atom
reordering need no index updates. A missing reference element or candidate
species absent from the model is an error. The explicit equivalent is
`ElementMixture("Sr", {"Sr": 0.5, "Ba": 0.5})`.

For unusual site-specific or coupled global slots, `AlchemicalPair(atom_index,
atomic_number)` identifies a concrete virtual copy and
`AlchemicalConfig(groups, weights)` assigns one coefficient to each slot.
Groups accept `AlchemicalPair` objects or `(atom_index, atomic_number)` pairs;
atoms absent from every group stay ordinary. Unlike `ElementMixture` fractions,
these low-level slot weights are finite and nonnegative but are not normalized.

This is frozen-weight dual-topology message mixing, not interpolation of model
parameters or learned species embeddings. The physical neighbor graph is built
before virtual expansion. Messages from virtual sources to ordinary receivers
use the source-slot coefficient; virtual-to-virtual messages connect matching
global slots only. Each virtual node's complete energy contribution is weighted
once. Energies and forces are reduced to the original atoms; stress uses the
expanded directed-edge geometry.

Indexed `groups` are immutable, `weights` returns a read-only copy, and
`set_weights()` copies its input and increments `version` only when coefficients
change. `ElementMixture.set_fractions()` has the same ownership and versioning
behavior. An update invalidates ASE results while preserving the prepared graph
and geometry. Construct a new configuration/calculator to change virtual
topology or candidate species.

Alchemical calculations require the generated `streamed_edges="direct"`
routing, a two-interaction standard compact MACE model, and matching direct
artifacts.
Both `capacity` and `speed` profiles are supported without fixed/tiled workspace.
Compact Agnesi distance transforms, A0/A1 density conditioning, and standard
MACE ZBL pair repulsion are supported. Density sums count each concrete source
edge with its slot coefficient. ZBL retains the concrete endpoint species and
weights each directed contribution by its source coefficient and the receiver
energy by its node coefficient; it does not interpolate atomic numbers or ZBL
parameters. Other distance transforms, field coupling, nonlinear/MH1, multiple
prediction heads, dispersion corrections, parameter gradients, observers,
Metal acceleration, and distributed/LAMMPS virtual topology are not supported.
Unsupported execution paths fail rather than silently selecting an unweighted
fallback.

The weight-aware direct packets use implementation generation 13. Artifacts
from earlier generations are incompatible and must be regenerated; ordinary
calculations retain unity-weight behavior.

Numerical qualification covers FP32 and FP64 on CPU/OpenMP and HIP (`gfx1151`),
including comparison with a frozen upstream reference for energy, forces, and
periodic stress. It also covers the official MACE-OMAT-0 medium checkpoint on
distorted periodic SrTiO3 with Sr/Ba mixing, including pure Sr/Ba endpoints,
interior 35/65 coefficients, density conditioning, Agnesi, and ZBL. Coefficient
updates, prepared-graph reuse, and physical-graph changes are also covered.
CUDA alchemical execution is not covered by this qualification.

The MACE-OMAT-0 comparison uses a compact export with at least 512 radial
spline points. At 512 points the FP64 maximum stress difference from the
upstream checkpoint is `6.22e-8 eV/A^3`; the default 256-point export gives
`1.00e-6 eV/A^3` on the same structure. Use
`symmetrix_extract_mace --num-spline-points 512` when upstream-level FP64
stress accuracy is required for this checkpoint.

## Field Calculators

```python
FieldContributionCalculator(field_calculator, **ase_kwargs)
FieldAwareCalculator(base_calculator, field_calculator, **ase_kwargs)
```

`FieldContributionCalculator` returns the MACEField contribution relative to
zero field. `FieldAwareCalculator` combines that contribution with an arbitrary
ASE base calculator.

## Ensembles

```python
SymmetrixEnsemble(model_files=None, *, calculators=None, **symmetrix_kwargs)
```

Provide exactly one of `model_files` or existing `calculators`. For each
supported base property `x`, an ensemble exposes mean `x`, stacked member
values `x_comm`, and population variance `x_var`.

## Direct Artifacts

`prepare_jit_host_artifact(...)` and `prepare_jit_device_artifact(...)` are
the Python APIs behind the corresponding command-line tools. They prepare R1
artifacts for standard MACE and MACEField compact JSON models; MACE-MH-1 uses
its separate generated program during calculator construction. Artifacts are
specific to the model contract, precision, ABI, implementation generation, and
backend target. Use the command-line wrappers for repeatable deployments.
