# Execution Support Matrix

This matrix describes the public execution algorithms. Internal kernels,
storage policies, and historical selector names are not separate public modes.
The request and capability tables are generated from the declared family
registry, so a family that gains a mode or a capability cannot leave them
behind; the surrounding prose states the boundaries the tables cannot.

## Execution capabilities

<!-- BEGIN GENERATED EXECUTION SUPPORT MATRIX -->
Generated from the declared family registry by
`tools/generate_execution_support_matrix.py`; regenerate it instead of
editing this block.

## Public execution requests

| Family | Public request | Accepted spellings | Model boundary | Declared evaluator spaces | Shipped Kokkos backends | Runtime artifact | LAMMPS |
|---|---|---|---|---|---|---|---|
| `MACE_Nonlinear` (mace-nonlinear) | `auto` | `auto` | `model_type` `MACE_Nonlinear` | Kokkos (float32, float64) | Serial, OpenMP, CUDA, HIP | As resolved below | Not supported |
| `MACE_Nonlinear` (mace-nonlinear) | `direct` | `direct` (public), `direct_streamed` (alias), `factorized` (alias) | `model_type` `MACE_Nonlinear` | Kokkos (float32, float64) | Serial, OpenMP, CUDA, HIP | The generated MH-1 node program; a host or device plugin artifact is required | MACE-MH-1 is not supported |
| `MACE_Nonlinear` (mace-nonlinear) | `reference` | `reference` (public), `all_interactions` (deprecated), `generic` (deprecated), `materialized` (alias), `non-compiled` (alias) | `model_type` `MACE_Nonlinear` | Kokkos (float32, float64) | Serial, OpenMP, CUDA, HIP | None | MACE-MH-1 is not supported |
| `MACE` (MH-0 compact model) | `auto` | `auto` | `model_type` `MACE`/`MACELES`/`MACEField` | Kokkos (float32, float64; `model_type` `MACE`/`MACEField`); Kokkos (float32, float64; `model_type` `MACELES`) | Serial, OpenMP, CUDA, HIP | As resolved below. MACELES evaluates its LES term in both routings | Format-aware compatibility default for standard MACE/MACEField; MACELES is not supported in LAMMPS |
| `MACE` (MH-0 compact model) | `direct` | `direct` (public), `direct_streamed` (alias), `factorized` (alias) | `model_type` `MACE`/`MACELES`/`MACEField` | Kokkos (float32, float64; `model_type` `MACE`/`MACEField`); Kokkos (float32, float64; `model_type` `MACELES`) | Serial, OpenMP, CUDA, HIP | Two-interaction MACE, MACEField, and MACELES require an R1 artifact. MACELES does not support fixed-workspace plans; single-layer standard MACE runs without R1 specialization, but capacity may specialize M0/R0 | Standard MACE and MACEField; the R1 artifact is required for two-interaction models, but not for single-layer models. MACELES is not supported in LAMMPS |
| `MACE` (MH-0 compact model) | `reference` | `reference` (public), `all_interactions` (deprecated), `generic` (deprecated), `materialized` (alias), `non-compiled` (alias) | `model_type` `MACE`/`MACELES`/`MACEField` | Kokkos (float32, float64; `model_type` `MACE`/`MACEField`); Kokkos (float32, float64; `model_type` `MACELES`) | Serial, OpenMP, CUDA, HIP | None | The original Symmetrix format (named v1 here) is retired, so the compatibility spelling resolves only for supported standard MACE/MACEField controls. MACELES is not supported in LAMMPS |
| `MACE_Polar` (polar-mace) | `auto` | `auto` | `model_type` `MACE_Polar` | Kokkos (float32, float64) | Serial, OpenMP, CUDA, HIP | As resolved below | `symmetrix/mace/polar/kk`; no artifact uses the portable `pair_spline_v1` backbone and one optional artifact selects the generated/direct backbone; the qualified deployment uses exactly one MPI rank owning the complete system and rejects domain decomposition |
| `MACE_Polar` (polar-mace) | `direct` | `direct` (public), `direct_streamed` (alias), `factorized` (alias) | `model_type` `MACE_Polar` | Kokkos (float32, float64) | Serial, OpenMP, CUDA, HIP | The generated MH-1 backbone subprogram; a host or device plugin artifact is required, while the long-range polar program remains native | `symmetrix/mace/polar/kk` with one matching generated/direct artifact for the host or device target; exactly one MPI rank must own the complete system, and domain decomposition is rejected |
| `MACE_Polar` (polar-mace) | `reference` | `reference` (public), `all_interactions` (deprecated), `generic` (deprecated), `materialized` (alias), `non-compiled` (alias) | `model_type` `MACE_Polar` | Kokkos (float32, float64) | Serial, OpenMP, CUDA, HIP | None | `symmetrix/mace/polar/kk` uses the artifact-free portable `pair_spline_v1` backbone with the same one MPI rank, complete system admission boundary and rejects domain decomposition |
| `OAM` (OAM v2 contract) | `direct` | `direct` | `architecture` `oam` | Kokkos (float32, float64) | Serial, OpenMP, CUDA; HIP package rejects this family | The architecture-bound, full-message-only receiver artifact: a host ABI v7 spline plugin, a CUDA ABI v3 NVRTC cubin, or an nvcc shared library | Not supported |
| `OAM` (OAM v2 contract) | `reference` | `reference` (public), `debug_v1` (alias), `mlp_reference` (alias) | `architecture` `oam` | Kokkos (float32, float64) | Serial, OpenMP, CUDA; HIP package rejects this family | None | Not supported |

## Declared evaluator capabilities

| Family | Capability | Declared status | Scope | Limits |
|---|---|---|---|---|
| `MACE_Nonlinear` (mace-nonlinear) | `target` | build-dependent | the default execution space | host, CUDA, or HIP |
| `MACE_Nonlinear` (mace-nonlinear) | `precision` | qualified | float32 and float64 | none |
| `MACE_Nonlinear` (mace-nonlinear) | `reference` | qualified | single-thread generic execution | host builds only |
| `MACE_Nonlinear` (mace-nonlinear) | `portable` | qualified | the generic Kokkos prepared path | no artifact required |
| `MACE_Nonlinear` (mace-nonlinear) | `generated` | qualified | the generated node program | a prepared artifact is required |
| `MACE_Nonlinear` (mace-nonlinear) | `stress` | qualified | the prepared-path stress reduction | none |
| `MACE_Nonlinear` (mace-nonlinear) | `field` | qualified | a model property | present only when the model couples a field |
| `MACE_Nonlinear` (mace-nonlinear) | `parameter_gradients` | unsupported | - | not implemented for this family |
| `MACE_Nonlinear` (mace-nonlinear) | `host_artifact` | qualified | every build | none |
| `MACE_Nonlinear` (mace-nonlinear) | `device_artifact` | build-dependent | device execution spaces | a CUDA-capable build is required |
| `MACE` (MH-0 compact model) | `target` | build-dependent | the default execution space | host, CUDA, or HIP |
| `MACE` (MH-0 compact model) | `precision` | qualified | float32 and float64 | none |
| `MACE` (MH-0 compact model) | `reference` | qualified | single-thread generic execution | host builds only |
| `MACE` (MH-0 compact model) | `portable` | qualified | the generic Kokkos prepared path | no artifact required |
| `MACE` (MH-0 compact model) | `generated` | qualified | the JIT/RTC artifact or a frozen built-in module | a prepared artifact is required |
| `MACE` (MH-0 compact model) | `stress` | qualified | the prepared-path stress reduction | none |
| `MACE` (MH-0 compact model) | `field` | qualified | MACEField models | ordinary MACE declares no field coupling |
| `MACE` (MH-0 compact model) | `parameter_gradients` | qualified | the prepared factorized path | none |
| `MACE` (MH-0 compact model) | `host_artifact` | qualified | every build | none |
| `MACE` (MH-0 compact model) | `device_artifact` | build-dependent | device execution spaces | a CUDA-capable build is required |
| `MACE_Polar` (polar-mace) | `target` | build-dependent | the default execution space | CPU and CUDA are runtime-qualified; HIP remains unqualified |
| `MACE_Polar` (polar-mace) | `precision` | qualified | float32 and float64 | none |
| `MACE_Polar` (polar-mace) | `reference` | qualified | the artifact-free portable MH-1 backbone and native polar program | CPU and CUDA runtime-qualified; HIP remains unqualified |
| `MACE_Polar` (polar-mace) | `portable` | qualified | the layer-local MH-1 fast path and the long-range kind group's portable nodes | independent per-layer UVU widths; supported keep-last layouts only |
| `MACE_Polar` (polar-mace) | `generated` | qualified | the generated MH-1 backbone subprogram with retained outputs and external product seeds | a prepared artifact and a recognized keep-last schedule are required; CPU and CUDA are runtime-qualified, while HIP remains unqualified |
| `MACE_Polar` (polar-mace) | `stress` | qualified | released M FP32/FP64 EFS: exact-zero isolated stress and composed periodic stress | CPU and CUDA sm120; S/L checkpoints and HIP runtime remain unqualified |
| `MACE_Polar` (polar-mace) | `field` | qualified | a program property | declared when the model's compiled program carries the external-field projection node |
| `MACE_Polar` (polar-mace) | `parameter_gradients` | unsupported | - | not implemented for this family |
| `MACE_Polar` (polar-mace) | `host_artifact` | qualified | every build | none |
| `MACE_Polar` (polar-mace) | `device_artifact` | build-dependent | device execution spaces | a CUDA-capable build is required |
| `OAM` (OAM v2 contract) | `target` | build-dependent | the default execution space | host or CUDA; HIP is unsupported |
| `OAM` (OAM v2 contract) | `precision` | build-dependent | float32 and float64 implementations on host and CUDA | current full-message-only artifact release-matrix qualification is pending |
| `OAM` (OAM v2 contract) | `reference` | build-dependent | the artifact-free same-spline Kokkos oracle on host and CUDA | current release-matrix qualification is pending; same-spline parity does not establish upstream-NequIP parity |
| `OAM` (OAM v2 contract) | `portable` | build-dependent | the compiler-free Kokkos edge product | the current runtime matrix is pending; this is the same-spline reference, not the optimized production route |
| `OAM` (OAM v2 contract) | `generated` | build-dependent | the full-message-only host ABI v7 artifact or CUDA ABI v3 receiver cubin | a prepared exact artifact is required; the current CUDA artifact is compile-qualified, but fresh runtime qualification is pending |
| `OAM` (OAM v2 contract) | `stress` | requalification pending | implemented on the current host and CUDA routes | full-message-only ABI-v7/ABI-v3 runtime qualification is pending; HIP is unsupported |
| `OAM` (OAM v2 contract) | `field` | unsupported | - | the OAM contract carries no field coupling |
| `OAM` (OAM v2 contract) | `parameter_gradients` | unsupported | - | not implemented for this family |
| `OAM` (OAM v2 contract) | `host_artifact` | build-dependent | source generation and compilation of the full-message-only ABI-v7 artifact | fresh host runtime qualification and repaired-wheel inclusion gates are pending |
| `OAM` (OAM v2 contract) | `device_artifact` | build-dependent | source generation and compilation of the full-message-only CUDA ABI-v3 artifact | fresh CUDA runtime qualification and repaired-wheel inclusion gates are pending; HIP is unavailable |
<!-- END GENERATED EXECUTION SUPPORT MATRIX -->

For MACE Polar, `direct` and `reference` select the generated or portable
backbone implementation. They do not select electrostatic boundary conditions.
The independent runtime request `polar_electrostatics` defaults to `auto`,
which chooses real space for an all-FFF single system and direct reciprocal
space when any PBC axis is true. Neither choice is serialized in the model
JSON. PME/PPPM and distributed Polar LAMMPS remain deferred.

### Native additive corrections

| Correction | Configuration | Properties | Graph requirement | Kokkos builds | Boundary |
|---|---|---|---|---|---|
| DFT-D2 | `D2Correction`; fixed zero damping | energy/free energy, per-atom energies, forces, stress | independent pair graph | CPU/OpenMP, CUDA, HIP | ASE integration |
| two-body DFT-D3 | `D3Correction`; `zero`, `bj`, `zerom`, or `bjm` | energy/free energy, per-atom energies, forces, stress | independent pair graph plus coordination-number node state | CPU/OpenMP, CUDA, HIP | ASE integration; no ATM |
| reciprocal FourierD3 | `FourierD3Correction`; fixed BJ damping | energy/free energy, per-atom energies, forces, stress | fully periodic global reciprocal mesh plus an independent short coordination graph | CPU/FFTW, CUDA/cuFFT, HIP/hipFFT | ASE integration; provider-gated, no real-space fallback |

The table describes implementation availability. Accelerator qualification is
backend- and architecture-specific: required CUDA/HIP tests must run on a
matching installed backend and visible device. CPU-only test results are not
accelerator execution or performance evidence. See
{doc}`/user/dispersion_corrections` for neighbor-building policy and explicit
D3-ATM, D4, and MBD limitations.

FourierD3 implementation availability follows the KokkosFFT provider selected
at build time. CPU/FFTW and CUDA/cuFFT are runtime-qualified in FP32 and FP64.
HIP/hipFFT is runtime-qualified in FP32 and FP64 on the recorded
`rocm7-gfx1151` configuration. This evidence is architecture-specific. The
recorded run does not qualify other HIP architectures. A provider-capable
build reports the exact Kokkos, KokkosFFT, provider, and execution-backend
identity through the CLI probes; compile success alone is not runtime-qualified
evidence.

Every family declares exactly two frontend routings. `direct` is the generated
(T3) implementation and needs a prepared artifact; `reference` executes the
generic implementation and needs no artifact at all. OAM runs that generic
Kokkos implementation in the selected host or CUDA execution space. There is
one evaluator stack: the retired non-Kokkos
evaluator and its `use_kokkos=False` selector are gone, so a request that cannot
reach the prepared path resolves to `reference` instead of running a second
implementation. The OAM frontend resolves the same way. Its `reference` routing
selects the native `reference` spelling explicitly; artifact-free `direct`
remains unavailable and fails closed rather than reporting a portable
implementation.

MACE_Nonlinear device plugins currently carry exactly two interaction slots.
Models with another interaction count use the count-agnostic composed ABI-v7
host plugin and are qualified only on host backends; CUDA and HIP qualification
covers the standard two-interaction topology.

OAM v2 `direct` execution separately supports the architecture-bound,
full-message-only host ABI-v7 plugin and CUDA ABI-v3 cubin/shared-library
artifact. FP32 and FP64 generation and compilation are available, but fresh
runtime qualification is pending after the artifact-contract change. The
independent upstream comparison remains narrower and does not establish live
upstream FP64 parity. HIP and multi-GPU are unsupported.
Periodic stress derivatives are reduced, symmetrized, and volume-scaled on the
execution device. CUDA transfers only the final nine-double tensor, and a
stress request after cached energy/forces reuses the completed graph. The
artifact-free OAM `reference` routing does not reduce stress.
See {doc}`/user/models` for artifact controls, and
{doc}`publications` for citation guidance.

The full-message-only host and CUDA artifacts are source-buildable in FP32 and
FP64, but their current release qualification matrix and repaired-wheel gates
are pending. Resolution is by an exact comparison of
the ordered layer and irrep/channel/angular layout, tensor-product structure,
radial dimensions, and normalization; paths, package labels, and learned
weights do not select an artifact. An unmatched valid architecture retains the
runtime-generation path and therefore needs the corresponding compiler.

`materialized`, `non-compiled`, `generic`, and `all_interactions` are aliases
for `reference`; `generic` and `all_interactions` emit a deprecation warning and
`materialized` and `non-compiled` do not. `factorized` and `direct_streamed` are
compatibility aliases for `direct` and, in the Python frontend, preserve the
former throughput-oriented behavior by changing an otherwise-`capacity` request
to the `speed` profile. Use the literal `direct` request for capacity behavior.
The removed `receiver_factorized` selector is rejected.

The accepted spellings are compact-family vocabulary, and each frontend
resolves its own accepted set from the family declaration it serves before any
native call: `NequIPCalculator` accepts only the literal OAM set, `direct`,
`reference`, `debug_v1`, and `mlp_reference`, so no MACE alias resolves there
and both OAM compatibility spellings resolve to `reference`. The compact
MACE-MH-1 evaluator admits the `direct` and `reference` routings. The literal
`materialized` string
is still refused when it is handed to a native evaluator directly, but no public
request reaches it, because the family resolves the spelling to `reference`
first. These admission rules are native contract data, not additional public
modes.

Ordinary MACE exposes energy, free energy, per-atom energies, forces, and
stress. MACEField additionally exposes polarization and admitted analytical
Born effective charge and polarizability calculations. Property, precision,
and MPI qualification can be narrower than implementation availability; dated
benchmark reports provide the evidence for a specific backend and workload.

`execution_profile="capacity"` and `"speed"` select resource objectives inside
the direct algorithm. `allow_fixed_workspace=True` separately permits bounded
tiled workspace for qualified ordinary MH0 standard MACE models on CUDA in
FP32 and FP64. MACEField and MACE-MH-1 do not support fixed-workspace plans.
These controls do not create additional execution modes.

LAMMPS `mpi_message_passing` supports the dual-layer fixed-workspace plan for
ordinary two-interaction standard MACE with CUDA FP32 or FP64 direct execution
and a precision-matched generated tiled R1 device artifact. The receiver- and
edge-sized workspace is bounded, while H1 and its adjoint remain sized to all
local and ghost feature nodes. HIP, field-coupled models, parameter gradients,
and execution observers are outside this qualification boundary.

See {doc}`/streamed_edge_execution` for the architecture, {doc}`/user/execution`
for user controls, and {doc}`/user/lammps` for the narrower LAMMPS boundary.
