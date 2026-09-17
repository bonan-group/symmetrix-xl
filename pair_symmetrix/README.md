# Using `pair_symmetrix`

> [!WARNING]
> Symmetrix-XL requires the 10 December 2025 LAMMPS release, or newer, matching
> the version check enforced by the installer and build helper.

### Generating a model

First, extract your model in `.json` form. See the Symmetrix-XL [Python package documentation](../symmetrix/README.md) for details.
You will need a Python environment with a compatible `mace` module installed.

The appropriate LAMMPS pair style commands are
```
pair_style    symmetrix/mace
pair_coeff    * * srtio3-mace.json Sr Ti O
```

For a multi-head JSON, select one model-wide prediction head in `pair_style`:

```text
pair_style    symmetrix/mace head omat_pbe
pair_coeff    * * srtio3-mace.json Sr Ti O
```

This minimal example uses the serial pair style. For a two-interaction model,
the first-class Kokkos direct path requires the matching prepared host or
device R1 artifact described below. Single-layer models do not require R1
artifacts.

Without `head NAME`, the JSON's declared default head is used. Legacy
single-head JSON files use their existing behavior.

The final `Sr Ti O` assumes that Sr, Ti, and O correspond to LAMMPS
types `1`, `2`, and `3`, respectively. A compact universal JSON can be reused by a
different input with another element mapping; active radial tables are built
once during `pair_coeff` for that mapping.

Compact universal files use Symmetrix-XL format version 2 and require an updated
pair style. Existing unversioned version 1 JSON remains supported. When an
artifact must also work with an older Symmetrix-XL/LAMMPS installation, generate
it with `symmetrix_extract_mace --radial-format pair-splines` and an explicit
element subset.

Format-version-3 `MACE_Nonlinear` models, including MACE-MH-1, are currently
supported by the Python ASE calculator and native library only. Both LAMMPS
pair styles reject these models explicitly at `pair_coeff`.

For MACEField JSON models, use the Kokkos pair style with an explicit
uniform electric field:
```
pair_style    symmetrix/mace/kk electric_field 0.01 0.0 0.0
pair_coeff    * * srtio3-macefield.json Sr Ti O
```
Equivalently, LAMMPS can add the `/kk` suffix when launched with `-sf kk`.
The field is currently a static graph-level vector in the active LAMMPS
unit system. Per-atom fields, time-dependent fields, field-response
properties, non-Kokkos MACEField LAMMPS runs, and atomic virials are not
yet supported.

The Kokkos pair style recognizes
`streamed_edges auto|materialized|non-compiled|direct` for compact
format-version-2 MACE and MACEField models. `non-compiled` is the public name
for the compiler-free fallback. The internal `generic` spelling remains a
temporary compatibility alias, along with `all_interactions -> generic`,
`factorized -> direct`, and `direct_streamed -> direct`.
The default `auto` mode selects `direct` only when a `jit_host_artifact` or
`jit_device_artifact` is provided; otherwise it selects `generic` for compact
format-version-2 models and retains `materialized` for version-1 pair-spline
models. Loading version 1 prints a migration warning; explicit mode selections
override `auto`. The serial pair style applies the same format-aware default
automatically.

`direct` supports both `no_domain_decomposition` and the default
`mpi_message_passing` domain-decomposition mode. The MPI path computes the
first-interaction H1 state for owned atoms, uses LAMMPS forward communication to
populate ghost H1 state, and reverse-communicates H1 adjoints before reversing
the first interaction. It retains the full LAMMPS skin neighbor list, clamps
inactive candidates at the compact polynomial cutoff where their radial
contribution is zero, and reuses the device topology and prepared factorized
schedule until LAMMPS rebuilds that list. `newton pair on` remains required.
The qualified Kokkos CUDA launch uses `package kokkos neigh half newton on` to
override Kokkos's default Newton policy. The pair style itself requests the
full neighbor list needed by the model, irrespective of that package-level
neighbor preference.
`no_mpi_message_passing` is not supported with prepared direct execution. A
JIT-generated CPU shared library can be selected with
`jit_host_artifact /path/to/plugin.so`; CUDA/HIP modules use
`jit_device_artifact /path/to/module.cubin` (or `.hsaco`). The native loaders
verify ABI, model contract, precision, and backend before use; device modules
also validate their GPU target. Host artifacts do not encode a CPU ISA, so the
deployment machine must support the compiler target selected during artifact
preparation. Exactly one R1 host or device artifact is accepted. Models whose
M0 or R0 topology does not match a built-in module additionally use
`jit_m0_device_artifact` and/or `jit_r0_device_artifact`; M0 also records its
validated `chunk32` or `table` schedule through `jit_m0_device_schedule`.
These operator modules are device-only and are loaded after R1 but before
low-memory admission. Explicit `direct` requires a matching R1 artifact for
two-interaction models and never falls back. Single-layer models have no R1
stage and use the built-in direct executor. `generic` rejects artifacts because
generic execution is compiler-free.
The removed `receiver_factorized` selector is rejected by all current
evaluators, including the LAMMPS pair style.
The pair style loads artifacts but never invokes a compiler.

`profile capacity` is the default and activates the same qualified M1
recomputation and MH-0 adjoint-reuse bundle as the Python API. Use `profile
speed` for the retained throughput plan. FP32 capacity execution also uses
compact unit-direction geometry, while FP64 retains Cartesian double-precision
geometry. Direct dual-layer execution requires a validated host or device
artifact; single-layer models have no R1 contract and use the built-in direct
executor without an R1 artifact. Incompatible configurations fail during
`pair_coeff` rather than falling back.
Optional M0/R0 device modules require both a device R1 artifact and `profile
capacity`. `low_memory yes|no` remains a deprecated alias for `profile
capacity|speed` and never enables fixed workspace.
Bounded fixed-workspace plans are disabled by default. Add
`allow_fixed_workspace yes` alongside `profile capacity` to permit their
selection when the planner determines that they reduce the graph memory
requirement. The option permits selection and does not force a tiled plan.

Single-layer direct MPI evaluates owned receivers and local-plus-ghost sources
without communicating H1 or H1 adjoints. Retained single-layer plans are
backend-independent; fixed-workspace single-layer plans currently require
CUDA FP32. Dual-layer fixed-workspace likewise requires CUDA FP32 and remains
unsupported with MPI; dual-layer retained and non-tiled capacity plans keep
their H1 communication protocol.

Multi-rank direct execution with pre-generated Kokkos OpenMP host artifacts is
qualified for FP32 and FP64 standard MACE and MACEField in two-rank periodic
orthogonal and triclinic cells, including the low-memory bundle and ownership
migration. FP32 direct low-memory MACEField is additionally qualified with four
ranks in a `1x2x2` decomposition. Generic MPI execution is qualified for FP32
and FP64 standard MACE and MACEField in two-rank periodic orthogonal and
triclinic cells. CUDA/HIP use the same evaluator phases,
packet ABI, generated R1 sources, and LAMMPS communication hooks. CUDA FP32
standard MACE low-memory execution is qualified with two ranks sharing one GPU
and host-staged MPI in periodic orthogonal and triclinic cells. It is also
qualified on one node with one, two, and four A5000 GPUs, one rank per GPU, and
CUDA-aware NVIDIA HPC-X OpenMPI 4.1.7a1. CUDA FP32 direct low-memory MACEField
is qualified on two A100-80GB GPUs with CUDA-aware OpenMPI 4.1.6, one rank per
GPU, and a `1x1x2` decomposition. The repeated-run gate includes ownership and
topology changes, allocation-free in-capacity graph updates, alias-safe
exceptional growth, and exact-capacity allocation when the advisory memory
estimate leaves no room for optional slack. FP64 CUDA, HIP MPI,
other MPI providers, and cross-node GPU-aware communication remain
qualification targets.

On every direct MACEField topology rebuild, each rank formats one fixed-size
capacity record containing active and planned receiver/feature/edge counts,
the selected low-memory policy, and graph, geometry, result, M0, alias, and
communicated-H1 counters. One `MPI_Gather` sends these records to rank zero,
which logs them in rank order. This avoids relying on non-root standard error,
which some MPI launchers suppress. Active counts are the kernel ranges;
planned counts are allocation high-water marks and must not be interpreted as
additional atoms or edges in the current graph.

When LAMMPS selects its legacy pair callbacks, Symmetrix-XL gathers and stages only
the boundary H1 or H1-adjoint packet. It does not mirror the complete
local-plus-ghost feature tensor. Reusable packet and index buffers avoid
per-swap allocation after their first use. The pair `extract` interface exposes
the cumulative legacy-callback counters `symmetrix_mpi_staged_packet_d2h_bytes`,
`symmetrix_mpi_staged_packet_h2d_bytes`, and
`symmetrix_mpi_staged_index_h2d_bytes` for profiling through the LAMMPS library
API. They remain zero when LAMMPS uses direct device-buffer pair communication,
but zero alone is not a positive transport diagnostic because it also occurs
when no pair communication callback runs.

LAMMPS does not run the JIT compiler. Prepare a host artifact in the same
Symmetrix-XL CPU environment before launching CPU LAMMPS:

```bash
jit_artifact=$(symmetrix_prepare_jit_host_artifact \
    --model srtio3-mace.json \
    --precision float32 \
    --path-only)

lmp -var jit_artifact "$jit_artifact" -in in.mace
```

Use the matching artifact in `in.mace`:

```text
pair_style symmetrix/mace/float32/kk mpi_message_passing streamed_edges direct \
           profile capacity jit_host_artifact ${jit_artifact}
pair_coeff * * srtio3-mace.json Sr Ti O
```

The default host target is optimized for the preparation machine. Use
`--host-target portable` to generate an x86-64-v3 artifact for compatible
deployment machines. Host artifacts still remain model-, precision-, ABI-, and
JIT-generation-specific. The deployment CPU must support the selected native or
x86-64-v3 compiler target.

When starting from a PyTorch checkpoint, conversion and device preparation can
be performed together on the target GPU:

```bash
symmetrix_extract_mace --model mace-omat-0-medium.model \
    --chemical-symbols Sr Ti O \
    --output srtio3-mace.json \
    --prepare-jit-device-artifact
```

This prepares both FP32 and FP64 artifact sets in one Kokkos runtime session.
Its JSON array labels each result with `precision`, the R1 `artifact_path`, and
the optional low-memory `operator_modules`. For a LAMMPS launch, render one
precision's complete pair-style settings with the dedicated command below.

Prepare a device artifact before launching LAMMPS on the target GPU. FP32 is
the first-class precision:

```bash
jit_arguments=$(symmetrix_prepare_jit_device_artifact \
    --model srtio3-mace.json \
    --precision float32 \
    --lammps-arguments)

lmp -var jit_arguments "$jit_arguments" -in in.mace
```

Use the variable in `in.mace`:

```text
pair_style symmetrix/mace/float32/kk no_domain_decomposition streamed_edges direct \
           profile capacity \
           ${jit_arguments}
pair_coeff * * srtio3-mace.json Sr Ti O
```

For FP64, prepare with `--precision float64` and use
`pair_style symmetrix/mace/kk`.

For a model covered by built-in M0 and R0 modules, the rendered fragment
contains only `jit_device_artifact` and
`jit_device_blocks_per_compute_unit`. For other compatible compact models it
also contains the exact M0/R0 artifact paths and M0 schedule required for
low-memory admission. This avoids silently passing only R1 to LAMMPS.

The checked-in M0/R0 topology specializations are frozen compatibility
accelerators. Extending that compile-time specialization mechanism is
deprecated; new compatible topologies should be prepared as RTC operator
modules. Existing built-ins remain selectable and require no extra LAMMPS
arguments during the compatibility period.

For domain decomposition, replace `no_domain_decomposition` with
`mpi_message_passing` and launch LAMMPS with the desired MPI rank count. The
same direct implementation and JIT artifact are used on every rank; this
configuration has the narrow same-node qualification described above. On a
one-GPU CUDA node, request
the device explicitly with `-k on g 1` in addition to the half-neighbor/Newton
package policy above.
When all GPUs are visible to every rank on a multi-GPU node, pass the visible
device count, for example `-k on g 4`, and normally launch one local MPI rank
per GPU. When the scheduler masks one physical GPU into each rank, as in the
qualified scheduler jobs, each rank sees one logical device and must use `g 1`.

GPU-aware selection is owned by the linked LAMMPS Kokkos runtime, not by
Symmetrix-XL. OpenMPI/HPC-X builds are queried through `MPIX_Query_cuda_support()`;
the qualified logs contain no LAMMPS fallback warning. Other MPI providers may
require an explicit `-pk kokkos gpu/aware off` if their device-buffer support is
unknown or fails at runtime. Same-node success does not qualify cross-node
GPUDirect transport, and Symmetrix-XL does not provide an NCCL communication path.

For `pair_style symmetrix/mace/float32/kk`, prepare with
`--precision float32`. The command emits JSON by default. `--path-only`
retains the legacy R1-only output, while `--lammps-arguments` emits the complete
R1/M0/R0 settings and requires exactly one precision. It constructs only the
native evaluator and loads each cached module once to validate the ABI, model
contract, precision, and active CUDA/HIP target before LAMMPS starts. The
artifacts remain in the content-addressed Symmetrix-XL JIT cache and can be reused
by later runs.
Eight persistent blocks per compute unit is the qualified default and is part
of CUDA artifact validation. If a
`SYMMETRIX_JIT_CUDA_R1_EDGE_BLOCKS_PER_SM` override was used during
preparation, `--lammps-arguments` includes the matching
`jit_device_blocks_per_compute_unit` value. In JSON mode the same value is
available as `persistent_blocks_per_compute_unit`. HIP accepts the setting for
portable input files but validates its launch policy from the artifact
descriptor.

### Building LAMMPS

From a recursive Symmetrix-XL checkout, the standalone build frontend applies the
same detected backend and architecture policy to LAMMPS:

```bash
python tools/symmetrix_build.py lammps \
    --source /path/to/lammps \
    --backend auto \
    --mpi on \
    --prefix /path/to/install
```

Use `--backend cpu --cpu-target x86-64-v3`, `--backend cuda --arch sm120`, or
`--backend hip --arch gfx1151` to select an explicit target. A manifest written
by `symmetrix_build.py detect --output target.json` can be replayed with
`--target-manifest target.json`. The tool validates LAMMPS 10 Dec 2025 or
newer, installs the pair sources, creates a fingerprinted build directory,
uses LAMMPS's own Kokkos runtime, and checks that the installed executable
reports both Symmetrix-XL pair styles. The build identity includes Symmetrix-XL and
LAMMPS source identities, the install prefix, and the selected MPI wrapper and
provider. A matching qualified executable is verified and reused without
reconfiguring its CMake cache. Failed or stale attempts retain their logs and
retry in a fresh directory.

`--mpi on` fails immediately if no MPI C++ wrapper is available. Pass
`--mpi-cxx /path/to/mpicxx` for a site-specific wrapper. Use `--mpi auto` only
when falling back to a non-MPI executable is intentional. CPU builds require a
Fortran compiler and a supported optimized BLAS implementation because the
KokkosKernels BLAS check uses the Fortran ABI. CUDA and HIP builds use Kokkos
Serial for host execution and disable Kokkos OpenMP; this is the qualified
Symmetrix-XL configuration, not a general Kokkos limitation.

For site module environments and cross-compilation, pass explicit compiler,
toolkit, MPI-wrapper, generator, and architecture options to the build frontend
instead of maintaining separate raw CMake recipes. Use `--dry-run` to inspect
the resolved invocation before starting a build.
