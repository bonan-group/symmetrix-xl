# Direct Execution Architecture

Symmetrix-XL evaluates supported compact MACE models with
`streamed_edges="direct"` and `execution_profile="capacity"` by default. Edge
streaming is therefore an implementation property of the normal execution
path, not a feature that users need to enable.

This page describes the architecture and its invariants. Public modes and
backend coverage are maintained in {doc}`reference/execution_support_matrix`;
user-facing capacity controls are documented in {doc}`user/execution`.

## Execution Model

Direct execution specializes eligible radial and equivariant contractions for
the model contract while keeping graph shape and learned weights as runtime
values. Its principal goals are:

- consume, aggregate, or recompute edge intermediates without retaining every
  materialized edge tensor;
- select a qualified storage plan from the graph size, precision, backend, and
  requested properties;
- compile model-specific host, CUDA, or HIP kernels and cache the resulting
  artifact; and
- fail clearly when a required contract or artifact is unavailable instead of
  silently changing algorithms.

The implementation is part of Symmetrix-XL. The upstream direct-execution
project is used as an algorithmic reference and validation oracle, not as a
build or runtime dependency.

The compact two-interaction pipeline contains radial stages R0/R1, node
message stages M0/M1, dense transforms, readouts, and coordinate or field
reverse stages. Built-in standard modules cover common M0/R0 structures;
other admitted structures use runtime-generated modules. R1 and compatible
MACE-MH-1 programs are specialized from versioned model contracts.

## Public Selection

The supported public spellings are:

| Request | Role |
|---|---|
| `direct` | Default prepared execution; two-interaction models require a matching R1 specialization, while admitted single-layer models use prepared R0/M0 execution without an R1 specialization. Capacity may still specialize their M0/R0 operators. |
| `non-compiled` | Compiler-free compatibility and diagnostic path; no performance guarantee. |
| `materialized` | Frozen legacy path, principally for legacy model formats and numerical controls. |
| `auto` | Temporary compatibility request resolved from model and evaluator capability. |

`generic` and `all_interactions` are deprecated aliases for `non-compiled`.
`factorized` and `direct_streamed` are compatibility aliases for `direct`; in
the Python frontend they preserve the former throughput behavior by changing
an otherwise-`capacity` request to the `speed` profile. Use literal `direct`
for capacity behavior. They do not emit a deprecation warning. The removed
`receiver_factorized` selector is rejected;
historical benchmark records that mention it describe an earlier
implementation.

The non-Kokkos evaluator selected with `use_kokkos=False` does not implement
the prepared direct path: a `direct` request resolves to serial generic
execution with a warning, while an explicit `materialized` request remains
materialized.

## Capacity Planning

`execution_profile="capacity"` ranks qualified internal plans using an advisory
memory estimate and selects the fastest estimated fit. If none fits the
advisory budget, it attempts the smallest qualified plan and lets allocation
determine feasibility. `execution_profile="speed"` selects the qualified
throughput plan without consulting available device memory.

The planner can vary harmonic-gradient retention, compact geometry, M1/readout
recomputation, and reuse of dead forward storage. Bounded tiled workspace is a
separate opt-in through `allow_fixed_workspace=True`; enabling it permits the
planner to consider those plans but does not force their selection.

`low_memory=True` and `low_memory=False` remain deprecated compatibility
spellings for the capacity and speed profiles. New callers should use
`execution_profile`.

After graph preparation, `calculator.execution_plan` reports the requested
algorithm and profile, selected MH-0 plan, active graph dimensions, planned
capacities, estimated bytes, and the selection reason. Private debug plan pins
exist for implementation tests and qualification drivers; they are not public
workflow controls.

## Runtime Artifacts

Model-specialized two-interaction direct execution requires a model-,
precision-, ABI-, generation-, and target-specific artifact:

- Serial and OpenMP use a C++20 host compiler selected by `CXX` or `c++`.
- CUDA uses NVRTC by default for device specialization.
- HIP uses hipRTC by default for device specialization.
- LAMMPS loads prepared host or device artifacts and never invokes a compiler.

Compilation or loading failure is fatal for direct execution. Set
`SYMMETRIX_JIT_POLICY=none` only for diagnosis and select `non-compiled` when a
compiler-free evaluation is intended.

Artifacts are stored below a private per-user cache. Set `SYMMETRIX_JIT_CACHE`
before Python starts to choose an explicit location. The cache contains
executable code and must not be writable by other users. Publication uses
content-addressed entries and validates manifests, target identity, file size,
and SHA-256 digests before reuse.

Concurrent cache misses compile in private staging directories. Publication
prefers atomic no-replace rename; on filesystems that do not support it,
Symmetrix-XL acquires an atomic per-key `mkdir` lock, revalidates any winner,
and publishes with ordinary atomic rename while holding the lock. Compilation
remains outside the lock, so shared HPC caches retain concurrent builds.

Prepare deployable artifacts with:

```bash
symmetrix_prepare_jit_host_artifact \
    --model model.json --precision float32

symmetrix_prepare_jit_device_artifact \
    --model model.json --precision float32
```

Use `--host-target portable` only for an x86-64-v3 host artifact. Device
artifacts must be prepared for the exact architecture used by the installed
backend and deployment GPU.

## Backend Invariants

Only one Kokkos backend may be initialized in a process. CUDA and HIP backends
are separate architecture-qualified packages because Kokkos and SpheriCart
device code is compiled ahead of time; runtime compilation cannot make an
extension portable to another GPU architecture.

OpenMP deployments must be checked in the actual Python and module environment
with `symmetrix doctor --json`. Strict qualification verifies the expected
OpenMP runtime, loaded runtime identity, worker participation, and admitted
host-worker BLAS policy. Static `ldd` output alone is not sufficient.

Qualified owner-local CBLAS calls are controlled centrally. Runtime-compatible
OpenMP OpenBLAS, sequential MKL, and compatible GNU-threaded MKL may be admitted;
pthread OpenBLAS is admitted only with one OpenMP worker. Intel-threaded MKL,
unidentified, runtime-incompatible, duplicate-runtime, and nested-enabled
configurations use Kokkos contractions instead, as does pthread OpenBLAS with
multiple OpenMP workers. See
{doc}`openmp_simd_coding_standard`.

## Prepared Graph Lifecycle

Prepared direct execution stores receiver-major schedules and reusable graph
state. With a positive neighbor skin, candidate edges are built at the model
cutoff plus the skin and reused until the Verlet displacement threshold is
crossed. Candidates outside the exact model cutoff contribute zero through the
compact radial boundary, keeping the prepared schedule stable.

Capacity allocation uses high-water marks. In-capacity graph updates reuse
storage; exceptional growth replaces it without exposing partially updated
state. Diagnostics distinguish active graph dimensions from planned allocation
capacities.

LAMMPS domain decomposition additionally communicates the boundary H1 state
and its adjoint for two-interaction models. The narrower MPI and transport
qualification boundary is maintained in {doc}`user/lammps` and the
`pair_symmetrix` source guide.

## Implementation Map

- Public selection and artifact orchestration:
  `symmetrix/source/symmetrix/calculator.py`
- Public mode parser: `libsymmetrix/source/mace_streamed_edges.hpp`
- Native execution planner: `libsymmetrix/source/execution_plan.hpp`
- MACE Kokkos lifecycle and execution:
  `libsymmetrix/source/mace_kokkos_factorized_lifecycle.cpp` and
  `libsymmetrix/source/mace_kokkos_factorized_execution.cpp`
- Host, CUDA, and HIP artifact loaders: `libsymmetrix/source/jit_host_plugin.cpp`,
  `libsymmetrix/source/jit_cuda_plugin.cpp`, and
  `libsymmetrix/source/jit_hip_plugin.cpp`
- CUDA and HIP runtime compilers: `libsymmetrix/source/jit_nvrtc.cpp` and
  `libsymmetrix/source/jit_hiprtc.cpp`
- Generated MH-1 contracts and source:
  `symmetrix/source/symmetrix/execution_mh1_contract.py` and
  `symmetrix/source/symmetrix/mh1_jit_codegen.py`

The detailed native ownership map is maintained in
{doc}`mace_kokkos_source_layout`. Benchmark Markdown files are dated evidence,
not current API documentation.
