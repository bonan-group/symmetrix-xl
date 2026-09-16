# Streamed-Edge Execution

Symmetrix-XL defaults to `streamed_edges="direct"` and
`execution_profile="capacity"` for supported compact two-layer MACE models.
`direct` is the model-specialized RTC path and never falls back to another
algorithm. `non-compiled` is compiler-free; `materialized` is frozen legacy
compatibility. The former direct execution-derived receiver-factorized mode has been
removed.

For compatibility, `generic` and `all_interactions` map to `non-compiled`, while `factorized`
and `direct_streamed` map to `direct`. Native diagnostics may retain the
internal `generic` identifier for this compatibility mode.
Standard M0/R0 modules, runtime JIT specialization, kernel-launch calibration,
and prepared device artifacts are separate implementation features.

The direct factorized mathematics is implemented inside Symmetrix-XL and does not
depend on the upstream direct execution package.

The initial production `receiver_factorized` selector covers FP32 ordinary
MACE and MACEField on Kokkos Serial/OpenMP. Other precisions, GPU backends, and
MACE-MH-1 fail explicitly until their receiver RTC adapters are implemented.
Model-specialized direct host and CUDA R1 kernels are the R1 stage inside the
full MH-0 direct factorized evaluator for ordinary MACE and MACEField, as well
as the compatible MH-1 family; R1 is not a separate public execution mode.
Kokkos HIP
supports the generic path, ordinary-R1 hipRTC modules and hipcc plugins, and
the standard M0 module. The
implementation lives inside Symmetrix-XL: the upstream direct execution package, Python
module, and source tree are not build or runtime dependencies. The upstream
project is used only as an algorithmic reference and validation oracle.
Upstream direct execution's generated tensor-product boundary starts from a precomputed
compact `phi` tensor plus the runtime final radial affine. For MH-1, Symmetrix-XL
extends that paper-aligned core with generated pair-conditioned prefix and
density programs; this extension is required by the supported MH-1 model
architecture and is not supplied by the upstream direct execution package.

These algorithms are exposed through the Python/ASE calculator. Standard MACE
and MACEField also expose them through the Kokkos LAMMPS pair style, subject to
the documented backend and domain-decomposition qualification boundaries.

## Build a Kokkos backend

The streamed-edge execution requires a Symmetrix-XL build with Kokkos. Its CPU execution
space may be OpenMP or Serial, and fresh CPU-only Kokkos builds default to
OpenMP. The non-Kokkos serial evaluator remains available in the same package
with `use_kokkos=False`, but it does not implement streamed-edge execution.

CPU builds may use OpenBLAS or Intel oneMKL LP64. The standalone build tool
selects OpenBLAS by default when it is available. For MKL deployments, prefer
`--blas mkl-sequential`: Kokkos supplies the outer parallelism, so the
single-threaded MKL layer avoids nested teams and OpenMP runtime coupling.
Pass `--blas mkl` only for a qualified GNU-threaded MKL deployment that shares
Kokkos's `libgomp` runtime and has a measured performance benefit. MKL
discovery uses `MKLROOT` (or `--mkl-root`) and the CMake vendors
`Intel10_64lp`/`Intel10_64lp_seq`; ILP64 variants are not supported.

A CPU-only build can omit Kokkos entirely with `SYMMETRIX_KOKKOS=OFF`, but
that build provides only the native serial evaluator and cannot use streamed-edge execution.

HIP builds use `SYMMETRIX_DEVICE_BACKEND=HIP`, hipcc as the C++ compiler, a
Serial host execution space, and an explicit AMD target. The qualification
device is a wave32 `gfx1151` Radeon 8060S using `/opt/rocm`:

```bash
cmake -S symmetrix -B build-hip \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DSYMMETRIX_DEVICE_BACKEND=HIP \
    -DKokkos_ARCH_AMD_GFX1100=ON \
    -DKokkos_IMPL_AMDGPU_FLAGS=--offload-arch=gfx1151 \
    -DKokkos_IMPL_AMDGPU_LINK=--offload-arch=gfx1151
cmake --build build-hip --target symmetrix_bindings -j 4
```

Pinned Kokkos 5.0.2 does not yet expose a named `gfx1151` option, so this uses
its compatible `gfx1100` RDNA wave32 traits while compiling and linking for the
actual ISA. This source-build configuration requires hipcc 6.2 or newer. It intentionally
disables CUDA SpheriCart, SpheriCart OpenMP, `Kokkos_ARCH_NATIVE`, host
`-march=native`, and global fast-math. Device-resident Kokkos spherical
harmonics support `l_max=0..6`. `SYMMETRIX_HIP_BLAS=AUTO` selects rocBLAS when
found and retains a portable Kokkos fallback; `ROCBLAS` and `KOKKOS` make the
choice mandatory. Eligible direct R1 contracts compile to a target-specific
hipRTC code object and load through the HIP module API on the evaluator's
Kokkos stream. `SYMMETRIX_JIT_HIP_JIT_BACKEND=automatic|hiprtc|hipcc`
controls the compiler. `automatic` selects the hipRTC module path exclusively.
Discovery, compilation, or module-load failure
never starts hipcc; the calculator raises. Explicit `hipcc` remains available
for one compatibility cycle and emits a deprecation warning. The debug-only
`SYMMETRIX_JIT_POLICY=none` setting disables RTC specialization.
HIP AOT and generated MH1 kernels remain fail-closed. The generated reverse
edge kernel uses cooperative native-wave ownership by default. Set
`SYMMETRIX_JIT_HIP_R1_EDGE_STRATEGY=wave|serial` to override this choice for
qualification or rollback.
`SYMMETRIX_JIT_HIP_R1_EDGE_THREADS_PER_BLOCK` and
`SYMMETRIX_JIT_HIP_R1_EDGE_BLOCKS_PER_CU` provide validated qualification
overrides. Both values are embedded in the generated source, artifact variant,
build metadata, cache key, and benchmark report. Production defaults are 64
threads and eight blocks per compute unit; the loader accepts only safe
descriptor values up to the device limits.
The `gfx1151` qualification workflow runs both policies in fresh processes and
requires the wave policy's 256-atom median to remain at or below `50 us/atom`;
raw samples from both policies are retained as workflow artifacts.
The earlier 864-atom MACEField qualification favored serial ownership, while
the later 10,976-atom thermal MH-0 qualification favors wave ownership by 26.1
percent. See `benchmarks/factorized_gpu_schedule_portability.md` for the current
default rationale and both workload-specific results.

Successful hipcc compilation alone does not qualify runtime correctness or
performance. Production claims require the mandatory AMD device workflow and
the numerical, lifecycle, and performance matrices described by the migration
plan.

An explicit OpenMP build is:

```bash
cd symmetrix-xl
pip install --verbose . \
    --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release \
    --config-settings=cmake.define.SYMMETRIX_KOKKOS=ON \
    --config-settings=cmake.define.Kokkos_ENABLE_CUDA=OFF \
    --config-settings=cmake.define.Kokkos_ENABLE_OPENMP=ON \
    --config-settings=cmake.define.Kokkos_ENABLE_SERIAL=OFF \
    --config-settings=cmake.define.SYMMETRIX_SPHERICART_CUDA=OFF
```

CMake and KokkosKernels preserve backend-specific choices in the build cache.
When changing an existing build between Serial, OpenMP, or CUDA, remove the
package's `build` directory (or otherwise use a fresh build directory) before
rebuilding. Passing new backend flags into a reused cache is not sufficient.
CUDA SpheriCart is enabled by default whenever `Kokkos_ENABLE_CUDA=ON`,
including when the Kokkos CUDA choice is supplied explicitly. Set
`SYMMETRIX_SPHERICART_CUDA=OFF` explicitly only as a rollback or diagnostic;
that setting moves spherical harmonics through the host and is substantially
slower for medium and large graphs.

Use a C++20-capable compiler with OpenMP support. An OpenMP-enabled OpenBLAS
build is preferred for CPU deployments because it uses the same OpenMP runtime
as Kokkos. Symmetrix-XL detects the loaded OpenBLAS threading mode at runtime and
prints a warning for pthread builds. OpenMP OpenBLAS is also selected
automatically for MH-0's owner-local host GEMMs; pthread or unidentified BLAS
retains the native Kokkos worker path. Identified pthread OpenBLAS is restricted
to one host-BLAS thread outside that path; Symmetrix-XL cannot control an
unidentified BLAS provider. Set `OMP_NUM_THREADS` and, optionally,
`KOKKOS_NUM_THREADS` before starting Python. Use `symmetrix doctor --json` to
verify `openblas_openmp_enabled`, `openblas_openmp_runtime_compatible`,
`omp_max_active_levels`, and the loaded library path.

Before production OpenMP evaluation, run `symmetrix doctor --json` in the
actual Python environment and require `status: "ok"`. The extension reports
the compiler, compiled OpenMP level, expected runtime path/hash, actual
`omp_get_max_threads` symbol owner, all mapped OpenMP runtimes, Kokkos
execution space, initialized concurrency, an actual parallel worker team,
per-worker CPU affinity, host BLAS provenance, and the selected host-worker
contraction backend. Set `SYMMETRIX_OPENMP_RUNTIME_CHECK=strict` to repeat a
native byte-identity and duplicate-runtime preflight immediately before
Symmetrix-XL initializes Kokkos. Static
`ldd` output alone is not a deployment qualification because the Python
executable's loader policy participates in runtime selection.

Multi-worker Kokkos OpenMP automatically uses OpenMP OpenBLAS contractions for
H1, A1, and H2 only when its query symbols belong to the CBLAS provider, it uses
the same OpenMP runtime as Kokkos, and `omp_get_max_active_levels()` is at most
one. These owner-local calls then execute concurrently across the outer Kokkos
workers without creating nested BLAS teams. The same route admits sequential
MKL, which has no internal worker team, and GNU-threaded MKL when it shares the
single `libgomp` runtime used by Kokkos. Pthread OpenBLAS, Intel-threaded MKL,
unresolved `libmkl_rt`, unidentified providers, duplicate runtimes, and
nested-enabled configurations instead use native Kokkos contractions.
`SYMMETRIX_HOST_WORKER_BLAS=off` forces the native route. The `unsafe` setting
forces concurrent CBLAS for controlled diagnosis when automatic admission
rejects the loaded backend and is rejected by the strict runtime doctor.
Kernel authors should follow the SIMD data-sharing, external-library, and
true-parallel qualification rules in `docs/openmp_simd_coding_standard.md`.

To build the supported single-threaded Kokkos host variant explicitly, use
`Kokkos_ENABLE_OPENMP=OFF` and `Kokkos_ENABLE_SERIAL=ON` instead.

Host model specialization requires a C++20 compiler selected by `CXX` (or
available as `c++`) and POSIX shared-library loading. CUDA and HIP runtime
specialization instead use NVRTC and hipRTC by default. Built-in M0/R0 standard
modules are architecture-generic and never invoke a compiler at runtime.
Foundation names and learned weights do not create package-time translation
units. Exact R1 code is compiled into the JIT cache when requested. Compilation
or loading failure raises rather than silently selecting a different algorithm.
`SYMMETRIX_JIT_POLICY=none` disables RTC for debugging; select
`streamed_edges="generic"` for compiler-free execution.

For Serial and OpenMP plugins, `SYMMETRIX_JIT_HOST_TARGET=automatic` probes
`-march=native` first. If the compiler rejects it, Symmetrix-XL verifies the
running CPU's complete x86-64-v3 feature set and probes
`-march=x86-64-v3`. The accepted target can be pinned with `native`,
`portable`, or a safe compiler target name. `portable` specifically means
x86-64-v3, including AVX2/FMA; no lower x86-64 baseline is selected silently.
An unverifiable or unsupported CPU, or rejection of every permitted compiler
target, is fatal for host direct specialization.

`SYMMETRIX_JIT_HOST_FLAGS` overrides the complete extra host compiler argument
list. Quoting is parsed with argument-list semantics and the compiler is invoked
directly, never through a shell. The effective target policy and flags are
reported in JIT diagnostics and recorded in both the manifest and cache key.
This prevents reuse across distinct target or flag policies.

Release builds retain the historical optimized floating-point policy by
default. Configure `-DSYMMETRIX_FAST_MATH=OFF` for an IEEE-oriented FP64 core
build. Match generated host kernels with
`SYMMETRIX_JIT_HOST_FP_MODE=ieee`, which selects `-fno-fast-math` and disables
floating-point contraction. The default JIT mode is `fast`; the effective
flags remain part of the artifact cache key, so IEEE and fast artifacts cannot
be confused.

On Kokkos CUDA, ordinary-MACE R1 contracts compile in-process with NVRTC
to exact-compute-capability cubins. The generated source is self-contained:
Kokkos headers, CUDA toolkit headers, PyTorch, upstream direct execution, and
`nvcc_wrapper` are not runtime requirements. Symmetrix-XL first probes a library
named by `SYMMETRIX_JIT_NVRTC_LIBRARY`, then an NVRTC library installed by
NVIDIA's Python package, then normal system-library locations.
Generated CUDA R1 edge kernels default to cooperative wave32 ownership with 32
threads per block and eight persistent blocks per SM. Qualification overrides
use `SYMMETRIX_JIT_CUDA_R1_EDGE_STRATEGY`,
`SYMMETRIX_JIT_CUDA_R1_EDGE_LOGICAL_WIDTH`,
`SYMMETRIX_JIT_CUDA_R1_EDGE_THREADS_PER_BLOCK`, and
`SYMMETRIX_JIT_CUDA_R1_EDGE_BLOCKS_PER_SM`; setting these to
`serial`, `1`, `256`, and `1`, respectively, restores the serial rollback.
Every field is validated and included in generated metadata, cache identity,
and runtime variant diagnostics.

Ordinary CUDA/HIP R1 launch-plan v2 performs coordinate reverse in one
source-owned kernel. A 64-thread block persists over sources, accumulates the
contract-specific `source_harmonics * channels` adjoint in shared memory, and
computes directed edge forces in the same scheduled-edge traversal. Native
32- or 64-lane subgroup shuffles reduce force partials. The capability bit and
physical-launch counter distinguish this fused path from legacy plugins, whose
separate source and edge launchers remain loadable as a compatibility fallback.

CUDA support is an architecture-qualified backend package, separate from the
CPU frontend. For example, CUDA 13 SM120 is installed as
`symmetrix-xl-cuda13-sm120` and does not overwrite the bundled CPU extension. The
backend package declares matching NVRTC/runtime dependencies where supported.
The NVIDIA driver is supplied by the host, and source builds still require a
development toolkit because Kokkos and SpheriCart are compiled ahead of time.
NVRTC cannot add CUDA support to the CPU extension.

`SYMMETRIX_JIT_CUDA_JIT_BACKEND` accepts `automatic`, `nvrtc`, or `nvcc`.
Automatic selects NVRTC exclusively for ordinary R1 and MH1. NVRTC discovery,
compilation, cache admission, or module-load failure never starts nvcc and is
fatal for direct execution. Explicit `nvcc` remains
available for one compatibility cycle, emits a deprecation warning, and searches
`SYMMETRIX_JIT_NVCC`, `CUDACXX`, `NVCC`,
CUDA toolkit roots, `PATH`, then `/usr/local/cuda/bin/nvcc`. MH1 cubins carry a
closed, versioned launch plan that the native evaluator validates and executes
on its Kokkos stream. Forced and automatic NVRTC use the same typed MH1 program
and launch plan. Real NVIDIA runtime and performance qualification remains
required. An earlier 25-30% apparent regression was invalidated by
construction-order controls and is not attributed to NVRTC.
Nsight tools are optional and never inference requirements.

This deployment follows the same broad split used by NVIDIA cuequivariance:
CUDA-major binary packages provide the prebuilt host integration and NVRTC
supplies runtime specialization. Symmetrix-XL temporarily keeps explicit nvcc as
a deprecated compatibility path while the generated paths are qualified
independently; automatic selection never uses it.

## Select and verify an execution algorithm

For ordinary MACE, the calculator accepts either a raw `.model` checkpoint or
a compact Symmetrix-XL JSON produced by `symmetrix_extract_mace`:

```python
from ase.build import bulk
from symmetrix import Symmetrix

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = Symmetrix(
    "mace-omat-0-small.model",
    use_kokkos=True,
    dtype="float32",
    streamed_edges="direct",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(atoms.calc.jit_status)
print(atoms.calc.jit_reason)
```

Loading a raw ordinary-MACE `.model` performs an in-memory conversion and
requires `mace-torch` at load time. It is not used during native evaluation. A
compact JSON avoids that dependency and repeated extraction:

```bash
symmetrix_extract_mace --model mace-omat-0-small.model \
    --output mace-omat-0-small.json
```

Then pass `"mace-omat-0-small.json"` to `Symmetrix` with the same options.
MACEField checkpoints cannot be loaded directly by the calculator; convert
them to the already-supported compact MACEField JSON format first.

Compact MACEField JSON models use the same generated R1 boundary as ordinary
MACE. The native field transform conditions H1 before R1, and the generated
reverse returns the H1 adjoint to the field-aware stages. Both
`dtype="float32"` and `dtype="float64"` are supported on Kokkos Serial,
OpenMP, and CUDA. Geometry, electric-field input, forces, and field adjoints
retain their double-precision public interface in both modes.

Residual-first standard MACE and MACEField models keep the same generated
boundaries. Their species-conditioned first self-connection is applied at the
shared H1 boundary before field conditioning and R1, so runtime R1, generic
table-driven R1, and an admitted JIT R1 plugin all consume the completed H1
exactly once. The residual uses a model-sized table and contributes zero edge
or graph-sized capacity bytes. Parameter-gradient mode rejects these models
until learned first-skip adjoints are implemented; coordinate reverse and
analytic field responses remain supported.

Kokkos MACEField response evaluation remains analytical and uses the generic
native response kernels rather than generated tangent kernels. When invoked by
the ASE calculator it consumes the immediately preceding primal field state;
prepared direct calls validate the completed graph-generation token before
doing so. They also validate the prepared graph's existing edge-receiver and
direct-source tables, then reuse the receiver view. CUDA Phi1 and A0 reverse
tangents use that edge-owned receiver schedule rather than falling back to a
node-owned traversal; source adjoints remain atomic in these kernels.
Diagnostics expose the selected receiver/source ownership and fused/generic
launch counts. Polarizability-only execution omits R0, the edge-sized BEC
result, and the
M0/A0 coordinate reverse. The remaining forward, directional-M1, readout,
reverse-Phi, and field-reverse workspaces have disjoint phase lifetimes instead
of being allocated together. Full BEC response retains the coordinate reverse,
then releases any R0/R1 tables temporarily materialized by the analytical pass.
Finite differences remain an external validation or workflow option and are not
an automatic calculator fallback.

Compatible models use the built-in standard M0 and scalar-source R0-v2 modules
when their exact structure matches. The built-in M0 fast path is the
correlation-three standard topology. Contract-compatible models with other
M0 topology, including correlation four, use the runtime-generated M0 module:
host M0 plugins use the host compiler, while CUDA and HIP modules use NVRTC or
hipRTC. Runtime-generated scalar-source R0 is available on CUDA and HIP;
host low-memory execution currently requires the built-in R0 implementation.
All variants read model extents and weights at runtime; there is no static
artifact registry. The field transform remains between the H1 product and
linear-up stages; the standard M0 and R0 modules do not contain field
operations. Runtime M0 and R0-v1 remain independent rollback controls.
Observers and parameter gradients use the
generic table-driven R1 feature path, while ordinary exact-contract inference
is compiled and cached through fused R1 JIT. The public default is
`streamed_edges="direct"` with `execution_profile="capacity"`. Direct is
fail-closed when its model contract or required artifact is unavailable.
`streamed_edges="auto"` remains a temporary compatibility mode; generic and
materialized execution are likewise explicit compatibility requests.

The JIT-free `generic` path also uses the built-in standard M0 module for the
supported 422-term topology. It does not load, compile, or consult a JIT
plugin. The module releases both graph-sized M0 polynomial tensors; a different
topology keeps runtime M0 with an explicit reason. Generated M0 supports
correlation up to four, but remains a direct-runtime specialization rather than
a generic-path fallback. The M0 selector accepts
`automatic`, `runtime`, and `standard`; forcing `standard` rejects when
topology admission fails instead of silently changing paths.

M1 polynomial recomputation is a separate portable policy with `automatic`,
`retained`, and `recompute` modes. Automatic prefers recomputation on Kokkos
Serial, OpenMP, CUDA, and HIP when admitted. The evaluator safely computes the
ordinary and MACEField-response scratch requirements and selects the largest
fitting channel tile from 32, 16, and 8. CUDA uses level-0 shared memory, HIP
uses level-0 LDS, and host backends use level-1 per-team scratch with vector
length 1. Retained M1 is the explicit rollback. Explicit recompute fails with
the backend, minimum required bytes, and available bytes when tile 8 cannot
fit. Diagnostics report requested and selected policy, backend, tile, scratch
bytes and limit, response tile/scratch, and fallback reason.

Recompute mode reports zero active and capacity bytes for both retained M1
polynomial tensors. MACEField analytical response also recomputes directional
polynomial values and gradients in team scratch, avoiding three additional
graph-sized response tensors per field seed. On the qualified RTX 5090 model,
the 40,936-byte admitted limit selects FP32 tiles 32 for ordinary MH-0 and 16
for MACEField, and FP64 tiles 16 and 8 respectively. M1 policy never changes
standard M0 selection or its separate runtime topology fallback.
Polarizability-only response also materializes R1 values without R1
derivatives and omits coordinate-force contractions; BEC requests retain the
full radial derivative path.
These M0/M1 policies do not change the required-JIT behavior of direct
execution.

Qualification status is intentionally separate from implementation:

| Backend | Capability implementation | Build coverage | Hardware execution |
|---|---|---|---|
| Serial FP32/FP64 | Complete | Complete | Retained/recomputed primal, response, repeated, and moving-geometry parity passed |
| OpenMP FP32/FP64 | Complete | Complete | Retained/recomputed primal and MACEField response parity passed |
| CUDA FP32/FP64 | Complete | CUDA 13.3 build passed | RTX 5090 parity, steady-state, and capacity passed |
| HIP FP32/FP64 wave32/wave64 | Complete | Unavailable locally: ROCm lacks `hipConfig.cmake` and runtime headers | Not qualified; this host has no AMD device |

HIP legality and admission do not constitute performance qualification.
Wave32/wave64 devices use their backend-native team execution while the M1
channel tile remains one of 32, 16, or 8; real-device results are required
before claiming either wave mode qualified.

Compatible MACE-MH-1-family checkpoints can use JIT-generated kernels:

```python
calc = Symmetrix(
    "mace-mh-1.model",
    head="matpes_r2scan",
    use_kokkos=True,
    dtype="float32",
    streamed_edges="direct",
)
```

The extractor records the two supported UVU interactions as a model contract.
Contract version 3 records the conditioned density-network topology and the
explicit channel-contiguous irrep layout used by graph-wide generated kernels.
Exact version-1 and version-2 contracts remain loadable: the calculator
validates them against the model and upgrades their identity in memory before
compiling or loading a version-3 artifact. No model file rewrite or Symmetrix-XL
rebuild is required.
This is current implementation behavior, not a generated-contract stability
guarantee. The generated interface is pre-production: contract versions,
plugin ABIs, and cached artifact identities may be broken when correctness or
performance requires it. Symmetrix-XL model schema versions 1 and 2 remain
supported independently; regenerate contracts and plugins with the active
version when the generated interface changes.
On Kokkos Serial and OpenMP, direct execution generates a host plugin; on
Kokkos CUDA it generates a CUDA plugin for the active compute capability. The
plugin is compiled, cached, and loaded for that exact contract. Channel counts,
radial widths, angular limits, and tensor-product paths are part of the cache
key, so changing them creates a new artifact automatically and does not require
rebuilding the Symmetrix-XL extension. The required policy raises if the model
lacks the contract or the backend-specific plugin cannot be compiled or loaded.

`streamed_edges="direct"` selects generated execution and requires a
model-specific specialization. For standard MACE and MACEField, it evaluates
the already projected radial splines directly into Phi, applies dense A1, and
reverses the same direct contraction. It is not the direct execution receiver-factorized
algorithm. Both `direct` and `receiver_factorized` require RTC. `materialized`
and `generic` bypass JIT. The `jit`
constructor argument is
deprecated and ignored; `SYMMETRIX_JIT_POLICY` is the only control:

| Environment value | Behavior |
|---|---|
| `required` (default) | Direct or receiver-factorized execution must compile or load its model-specific specialization. |
| `none` | Disables RTC. An explicit RTC mode raises; select `generic` for compiler-free execution. |

The host-RTC `receiver_factorized` generator is implemented in
`receiver_factorized_rtc.py`. Its legacy NumPy harness remains available for
isolated R1/A1 measurements, while the same generated source exposes the
validated descriptor used by the production evaluator.
Normal inference dispatches that generated receiver RTC implementation.
Observers and parameter-gradient replay require retained intermediate state,
so those diagnostic evaluations deliberately bypass the RTC entry point and
use the bounded stateful receiver factorization instead. Switching into and
out of a diagnostic evaluation does not change the requested public mode, and
the next ordinary evaluation returns to RTC execution.

`jit_status` reports how specialization was obtained or why it was skipped:

| Status | Meaning |
|---|---|
| `built` | A new plugin for the active Kokkos backend was compiled and loaded. |
| `cached` | A previously compiled, validated plugin was loaded. |
| `not_applicable` | The selected edge mode does not use JIT. |

For compatible MH-1 evaluators, `execution_mh1_execution_backend` reports
`generated_host_v4`, `generated_cuda_v4`, `generic`, or `inactive`. The generated
forward, source-reverse, and edge-reverse launch counters provide a lower-level
check that both interactions executed through the loaded plugin. The
`execution_mh1_generated_conditioning_forward_launch_count` and
`execution_mh1_generated_conditioning_reverse_launch_count` properties separately
confirm use of the complete generated MH-1 conditioner. Identity-prefix
interactions retain the compatibility conditioner because their fixed species
contribution is not yet represented by the generated conditioning ABI.

MH-1 source reverse uses a deterministic compact transpose schedule. For `E`
edges, `B` streamed blocks, and `S` active `(block, source)` segments, it stores
`O(E + S + B)` integer entries, with `S <= E`; it does not allocate a
blocks-by-nodes table. Sources are ordered within each block and their edge IDs
remain in ascending input order. The evaluator properties
`factorized_schedule_entries`, `factorized_schedule_active_sources`, and
`factorized_schedule_bytes` expose the resulting logical schedule.

ASE calculations prepare the exact MH-1 graph automatically. An unchanged
topology reuses its device-resident node/edge arrays, receiver offsets, compact
source schedule, and a monotonic graph token; warmed calls copy only edge
vectors and distances into grow-only geometry storage. Reordered or resized
topology prepares a new token before evaluation. Sphericart, generated plugin
launches, Kokkos stages, and optional ZBL execute on one ordered evaluator
stream, followed by one host-visible terminal fence. The evaluator is therefore
stateful and non-reentrant, like the standard-MACE direct evaluator. Prepared
graph, topology-validation, geometry-allocation/copy, Sphericart async-launch,
ZBL stream-launch, and fence counters are exposed for lifecycle diagnostics.

Standard MACE and MACEField `direct` calculations use the same prepared graph
lifecycle. With a positive ASE neighbor skin, MACEField retains Cartesian
topology-reference geometry on the device and derives current edge vectors,
unit directions, and radii from atom positions. A warmed call copies only
`3*num_atoms` current position values and the current electric field; it does
not upload edge geometry or radii. The field transform, generated R1 launches,
reverse field transform, optional ZBL, response kernels, and Sphericart work
are ordered on the evaluator stream before host-visible terminal fences.

`execution_profile="speed"` fixes the backend-qualified direct throughput
bundle. For standard compact MACE, it retains harmonic gradients while using
M1 and readout recomputation plus MH-0 forward/adjoint buffer reuse; FP32 also
uses compact unit-direction geometry. At 40,960 directed edges, CPU and CUDA
FP32 throughput switch to Y-only harmonic storage; CUDA FP64 and HIP remain
retained. The full-state bundle remains available when observers,
parameter gradients, or another qualification restriction prevents reuse.
`execution_profile="capacity"` is the default. Prepared-graph
construction estimates the selected throughput bundle from model dimensions
and atom/edge counts, retains it when it fits after the 5%/512 MiB reserve,
and otherwise selects Y-only harmonic storage, reconstructing coordinate
gradients in reverse kernels. Phi1 remains retained. Float32 selects
`edge_geometry_policy="unit-f32-radius-f64-v1"`, storing float32 unit directions
and float64 radii. Float64 retains `cartesian-f64-v1` so the low-memory path
does not reduce geometry precision. If Y-only admission fails, the evaluator
uses the remaining capacity policies with retained gradients and reports
`harmonic_storage_fallback_reason`. If the device-memory query fails or is not
available, selection is conservative and chooses that capacity fallback.
If the smallest qualified capacity estimate also exceeds the post-reserve
budget, the evaluator plans exact active capacity with no optional high-water
or geometric slack and attempts the allocation. The memory model is advisory:
it selects the code path and allocation capacity, but never rejects an
otherwise valid calculation. A genuine device allocation failure remains the
final feasibility signal. `low_memory=True` and `low_memory=False` remain
compatibility spellings for `capacity` and `speed`.
Bounded single- and dual-layer fixed-workspace plans are excluded from normal
capacity selection by default. Set `allow_fixed_workspace=True` to permit
their selection when qualified and smaller than the regular capacity plan.
This is a separate opt-in and does not force a fixed-workspace plan. Private
tiled debug-plan pins remain available without the public opt-in.
Materialized, generic, zero-skin, and older-extension compatibility paths
retain their existing host-geometry behavior.

For ordinary MACE and MACEField, the public `calculator.execution_plan` report
records the requested algorithm and profile, the selected internal plan,
advisory available/reserve bytes, boundary-attempt status, and every candidate.
It is `pending` before the first prepared graph and `active` afterward. MH-1
still exposes its existing policy diagnostics while its matching graph-time
resolver is implemented. Legacy diagnostics `low_memory_requested`, `low_memory_policy`,
`low_memory_selection_reason`, `low_memory_available_bytes`,
`low_memory_speed_estimated_bytes`, `low_memory_capacity_estimated_bytes`, and
`low_memory_selected_estimated_bytes` make the whole-bundle decision
reproducible. Additional `low_memory_device_*`, reserve, and candidate-capacity
diagnostics expose the queried device state and both retained-gradient and
Y-only capacity estimates. `execution_geometry_growth_reason` distinguishes
geometric headroom from exact boundary growth. MH-1 applies the same guarded
growth decision to its retained Cartesian, harmonic, and harmonic-gradient
geometry workspace, while retaining its existing execution policy.
Harmonic-specific diagnostics expose only the selected harmonic subpolicy and
its actual allocated buffers.
Explicit native `retained` and `y-only-direct-v1` requests remain deterministic
qualification and rollback controls.

### Development Plan Pins

Normal applications should select `streamed_edges`, `execution_profile`, and,
when bounded workspace is acceptable, `allow_fixed_workspace`. Benchmarks and
implementation tests may pass the private `_debug_execution_plan` argument to
`Symmetrix` to pin a validated MH-0 plan:

```python
calc = Symmetrix(
    model,
    streamed_edges="direct",
    execution_profile="capacity",
    _debug_execution_plan="mh0-direct-capacity-y-only",
)
```

The supported pins are `mh0-direct-speed`,
`mh0-direct-capacity-retained`, `mh0-direct-capacity-y-only`, and
`mh0-single-layer-tiled-v1`. A pin must match its profile, is reported with
`selection_source="debug_override"`, and bypasses only estimate-based ranking.
It still validates the direct artifact, the R0/M0/M1 contracts, precision,
requested properties, and all low-memory restrictions. It can therefore still
fail at qualification or with an allocator OOM. Do not use private pins in
deployment input files.

`mh0-single-layer-tiled-v1` is an explicit CUDA FP32 qualification plan for
ordinary one-interaction MACE models. It requires state-free R0 and M0,
single-layer invariant readout, compact edge geometry, Y-only harmonics, M1 and
readout recomputation, and no field coupling, observer, or parameter-gradient
collection. Each contiguous receiver batch completes forward and reverse
before the workspace is reused. Harmonic values and directed edge forces are
allocated for only the maximum edge span of one receiver batch. At the end of
each batch, directed forces are reduced into the graph-sized atom-force array
and its virial is accumulated into nine persistent FP64 values. Single-layer
graphs also omit the source-ordered R1 permutation because this plan never
consumes it.

For the target 128-channel, `l_max=3`, `L_max=1` model, inactive graph storage
is 16 bytes per directed edge: one signed 32-bit source index and three integer periodic
shifts. Receiver indices are implicit in receiver CSR, neighbor types are
derived from source node types in each active tile, and reference edge vectors
are reconstructed from node reference positions, shifts, and the cell. Its 88
bytes per node cover two topology integers, two FP64 coordinate vectors
(reference and in-place current/displacement), one FP64 energy, and one FP64
force vector. FP32 unit directions and FP64 radii are reconstructed in each
active tile. The active workspace uses 13,836 bytes per receiver and 112 bytes
per edge: 64 bytes of harmonic values, 24 bytes of directed force, 12 bytes of
FP32 unit direction, 8 bytes of FP64 radius, and 4 bytes of neighbor type.
`single_layer_workspace_*` diagnostics report
active and planned receiver and edge capacities, bytes per receiver and edge,
arena replacement/reuse counts, batch count, and tiled evaluation count. With
32,768 receiver slots, the qualified AlN graph needs 3,571,712 active edge
slots and 853,409,792 workspace bytes (813.88 MiB). Larger receiver overrides
did not materially reduce latency and consume memory that can instead extend
the graph-size limit. With zero neighbor skin, the tiled plan builds an exact
neighbor cache so integer periodic shifts remain available; it does not retain
graph-wide Cartesian edge vectors. Pin
`mh0-direct-capacity-y-only` for the graph-sized rollback path. Automatic
capacity selection chooses the tiled plan only when the graph-sized Y-only
estimate exceeds the post-reserve device-memory budget and the bounded estimate
is smaller. Explicit speed and graph-sized capacity plans remain preferred when
their estimates fit.

Fine-grained component overrides are intentionally not public. The planned
private structured development request will validate merged executor, geometry,
storage, MH-1 node-state/arena, launch, and test-memory choices before graph
preparation. Invalid combinations are reserved for unbound native test hooks.

Capacity execution keeps logical graph sizes separate from allocation sizes.
On the first prepared graph it may admit up to 0.25% additional receiver,
feature-node, and edge capacity, but only from memory remaining after the exact
selected bundle and the advisory 5%/512 MiB reserve. A later graph whose active
prefixes fit that high-water mark updates the existing topology, schedule,
geometry, result, and model-state views in place. Shrinking does not reduce the
high-water mark, and ordinary in-capacity growth does not increment allocation
counters.
If a graph exceeds a planned dimension, the evaluator fences once, detaches any
matching forward/adjoint aliases, releases the complete old owner bundle, and
then allocates the successor. This exceptional path avoids holding both large
generations at once and restores the aliases after reverse execution.

The diagnostics `execution_active_*`, `execution_planned_*`,
`execution_capacity_selection_reason`, graph replacement/update counters, and
geometry/result/state allocation counters expose this lifecycle. In MH-0
low-memory execution, including MACEField, the M0 adjoint aliases M0 after the
field and H1 communication reverse stages; the standalone M0-adjoint allocation
count therefore remains zero. Full-retention execution continues to own a
distinct M0 adjoint.

Prepared field responses validate the completed graph token and consume the
retained device geometry. Energy, forces, stress, and polarization do not add a
response reconstruction. Polarizability and BEC reconstruct A0, A1, M1, and H2
in the existing aliased allocations, including direct recomputation of dropped
A1 density-scale splines, before applying the analytical response. Stress is
reduced from device pair forces and geometry to a 3x3 tensor before Python
access. Full BEC response currently copies the directed-edge field-force
derivative to Python for atom reduction; it does not copy primal pair forces or
edge geometry.

For standard compact MACE with a positive ASE neighbor skin and a full-rank,
fully periodic cell, direct execution stores topology-reference positions
and shifted edge vectors in fractional coordinates. A topology build uploads
those references once. Accepted cell changes then copy only the current 3x3
cell and inverse cell; the existing Kokkos atom and edge kernels map the
references through that cell and apply current minimum-image non-affine
displacements. The host still applies the accumulated skin/deformation bound,
but it does not reconstruct Cartesian edge vectors while the topology remains
valid. Mixed or nonperiodic cells, nonlinear MH-1, older
extensions, and other streamed modes retain the Cartesian compatibility path.
Field-coupled models use retained Cartesian reference geometry rather than the
fractional cell-update optimization, but still update current edge geometry on
the device from atom positions.
Cartesian references are materialized lazily if execution later falls back, so
representation changes cannot consume geometry from an older cell.

The evaluator exposes fractional preparation, cell-copy byte/count, geometry
state allocation, position geometry, graph, and schedule counters. The ASE
calculator additionally exposes
`neighbor_cache_host_geometry_materialization_count`. These counters distinguish
a cheap cell refresh from a topology or schedule rebuild.

ABI-v4 generated execution keeps node features and adjoints in persistent
`ir_mul` layout across both interactions. It generates the product-adjacent
equivariant linears, normalization/gate, correlation-three product, skip, and
readout forward/reverse program and uses one grow-only phase arena instead of
generic node tapes and layout adapters. Learned linear, product, readout, and
normalization values remain runtime packs, so architecture-compatible trained
models share generated code. `node_workspace_bytes` reports retained node
state and arena storage; `precision_workspace_bytes` includes it.

On CUDA, dense node linears execute graph-wide in 8-node by 32-channel tiles
rather than repeating a complete matrix-vector loop in each node-owned block.
The tile loads runtime weights into shared memory and pads its channel extent
to 33 values so reverse-transpose reads do not collapse onto one shared-memory
bank. Normalization, gates, products, skips, and readouts retain deterministic
cooperative node ownership. All subkernels are launched in order on the
evaluator stream through the existing ABI-v4 phase entry points, so parameter
packs, persistent state, fallback behavior, and workspace accounting are
unchanged.

The generated MH-1 CUDA forward kernel reads the compact graph-wide active
receiver list. One warp owns an active receiver, policy-selected whole
output-irrep partition, and 32-channel tile. R1 reverse: source launches one
physical kernel per interaction; internally, one warp owns an active source,
whole input-irrep partition, and channel tile. The deterministic `auto` policy
uses full forward fusion up to 16 output components and budget-8 partitions
beyond that. Source `auto` keeps separate input groups when an interaction has
more than one: the official-model paired benchmark found this conservative
factorized schedule faster than full source fusion because it preserves owner
parallelism. These are compile-time heuristics, not runtime autotuning.

Edge reverse uses a 16-lane subgroup per edge and splits the prefix and
harmonic/cutoff adjoints into spill-free kernels with deterministic shuffle
reductions. The edge policy also represents an explicit fully fused candidate,
but `auto` remains split: full fusion is not qualified for the larger second
interaction, and a mixed per-interaction plan must pass resource, numerical,
and whole-evaluator timing gates before it can replace the split default. This
matches the paper-derived receiver/source ownership while retaining bounded workspace.

For non-identity MH-1 conditioned networks, the generated artifact also owns
the exact prefix and density topology. Forward assigns one active receiver per
owner, walks its CSR edges in stable order, writes compact edge `phi` directly,
and accumulates that receiver's density. Reverse assigns one edge per owner,
recomputes the small local activations, consumes the tensor-product `phi`
adjoint, and increments radial and cutoff adjoints. Generic MLP tapes and
materialized source/target contribution rows are absent on this path.
Parameters and pre-folded species contributions remain runtime arrays, so
checkpoints with the same architecture reuse generated code without embedding
learned values in the artifact.

Generated MH-1 execution accepts an optional compact-phi scratch budget through
`Symmetrix(..., execution_mh1_scratch_budget_bytes=N)`. The default `None` preserves
retain-all execution. A finite budget groups interactions by phi width and
admits retained forward tapes in deterministic reverse-layer order. Layers that
do not fit use one grow-only phase buffer per width and rerun generated
conditioning immediately before TPConv reverse. Identity-prefix layers follow
the same policy and regenerate compact phi in bounded edge blocks. The replay
uses the same fixed runtime parameters and prepared radial/species inputs; it
does not change the JIT artifact or cache identity. Reverse-only phi, harmonic,
and cutoff adjoint scratch is shared across the serial layer traversal.

The byte limit governs compact-phi tape/phase storage, not topology, geometry,
persistent node state, or other mandatory evaluator allocations. At least one
phase buffer per active phi width is irreducible. Consequently a requested
budget below `execution_mh1_scratch_minimum_bytes` is reported as unsatisfied while
execution remains correct at that minimum. The evaluator also reports
`execution_mh1_scratch_planned_bytes`, `execution_mh1_scratch_retained_bytes`,
`execution_mh1_scratch_recomputed_bytes`,
`execution_mh1_scratch_recomputed_layer_count`, and the cumulative
`execution_mh1_conditioning_recomputation_count`.

CUDA plugin ABI v4 is an additive dual-query artifact. Its inherited ABI-v3
query carries graph-wide forward, source-reverse, edge-reverse, and conditioner
plans; the v4 query adds persistent node-program phases and a runtime-layout
fingerprint. The graph policies retain their versioned tags:
`symmetrix.jit.mh1.cuda-forward-policy/1`,
`symmetrix.jit.mh1.cuda-source-policy/1`, and
`symmetrix.jit.mh1.cuda-edge-policy/2`. Each canonical policy dictionary
contains a content-derived `policy_id` bound to the model generation
fingerprint, target compute capability, and exact interaction plan.
The edge policy additionally records the generated reverse schedule so a
microkernel change produces a distinct policy and variant identity even when
its launch strategy and geometry are unchanged.

`jit_reason` describes why JIT is disabled or not applicable. More detailed deployment
diagnostics are available in `jit_diagnostics`; successful dynamic builds
also expose `jit_cache_key`, `jit_artifact_path`, and
`jit_artifact_id`. Generated MH-1 CUDA builds additionally expose the
active canonical plans in `jit_forward_policy`,
`jit_source_policy`, and `jit_edge_policy`.
`jit_variant_id` combines the model artifact identity with all three
policy identities. The policy attributes and variant identity remain `None`
when compilation or loading falls back, so they describe the inherited graph
plan in an active ABI-v4 artifact, not merely an attempted build.

## Compilation cache

On the first use of an eligible contract, calculator initialization generates and compiles a
model-specific host shared library or exact-SM CUDA cubin. Later
processes reuse the content-addressed artifact when the model contract,
generator, compiler, flags, execution target, and ABI still match. Host cache
entries include CPU identity; CUDA entries include the exact active compute
capability, CUDA runtime identity, NVRTC version, and compile options. Changing
the model dimensions or tensor-product program automatically produces a
different cache entry and does not require rebuilding the Symmetrix-XL extension.

Every generated artifact is additionally bound to an integer JIT generation
version. It appears in the content-addressed key, manifest, physical library or
module filename, and the artifact identity checked by the native loader. The
compiled standard modules expose the required version from
`libsymmetrix/source/jit_generation_version.hpp`; the Python generators provide
the same integer as `JIT_GENERATION_VERSION` in
`symmetrix/source/symmetrix/jit.py`. Maintainers must increment both declarations
together whenever generated source or its native consumer changes, even if the
plugin ABI itself is unchanged. This integer is a deliberate compatibility
epoch, not a Git commit hash.

The calculator compares the native requirement with the Python provider before
rendering source or consulting the cache. A mismatch fails with a rebuild or
reinstall diagnostic. Increasing the epoch therefore produces a cold cache miss
and a separately named artifact. It does not remove older epoch directories;
other branches, worktrees, or installed versions can continue using them at the
same time.

The cache location is selected in this order:

1. `SYMMETRIX_JIT_CACHE`
2. `$XDG_CACHE_HOME/symmetrix/jit`
3. `~/.cache/symmetrix/jit`

The cache contains executable code. Its root must be a real, non-symlinked
directory owned by the current user with no group or world permissions; mode
`0700` is recommended. Do not configure a shared writable cache across Unix
users. A private per-user cache may be shared by processes and compute nodes.
Host artifacts are separated by CPU identity, while CUDA artifacts are
separated by exact compute capability and CUDA compiler/runtime identity, so a
heterogeneous cluster does not reuse code for an incompatible target.

Concurrent clean misses use optimistic publication. Every process compiles in
its own private directory under the cache root's `.staging` directory, then
atomically renames that complete directory into its content-addressed location
without replacing an existing entry. Exactly one process publishes; other
processes discard their staging directories after validating and reusing the
winner. This avoids a build-duration lock at the cost of duplicate compilation
when several jobs encounter the same cold key simultaneously. A short per-key
lock is used only to quarantine an invalid or load-broken entry. Atomic
no-replace directory rename is currently required, so JIT preparation falls
back internally on non-Linux systems or cache filesystems that do not provide
it. Calculator initialization then raises under the required policy when
`direct` was requested. Only debug-only `SYMMETRIX_JIT_POLICY=none` skips
cache preparation.

On Linux, publication prefers the libc `renameat2` wrapper. When an older libc
does not export it, Symmetrix-XL issues the kernel syscall directly with
`RENAME_NOREPLACE`; syscall numbers are selected for x86-64 and AArch64.
Unsupported architectures, kernels, or filesystems raise explicitly. There is
no race-prone check-then-rename fallback, and `EEXIST` still means that another
clean publisher won the same content-addressed entry.

A process removes its own staging directory after a handled build failure or a
lost publication race. A process killed during compilation can leave an
unpublished directory below `.staging`; such directories are never considered
cache entries and may be removed administratively when no builders are active.
Published entries are validated by manifest, requested cache inputs, file size,
and SHA-256 digests before reuse. A malformed entry is moved under `.invalid`
while holding the short recovery lock and is replaced by a fresh build.

## Current support boundary

### Mode support matrix

| Canonical mode | Model boundary | Kokkos backends | JIT | LAMMPS |
|---|---|---|---|---|
| `materialized` | Legacy format-v1 and supported standard MACE/MACEField controls | Serial, OpenMP, CUDA | No | Yes |
| `generic` | Compact standard MACE/MACEField and compatible format-v3 MH-1 | Serial, OpenMP, CUDA, HIP | No | Standard MACE/MACEField |
| `direct` | Compact standard MACE/MACEField with R1 contracts and compatible MH-1 with generated contracts | Serial, OpenMP, CUDA; ordinary R1 is also available on qualified HIP builds | Required | Standard MACE/MACEField with a prepared artifact |
| `receiver_factorized` | FP32 compact standard MACE/MACEField with an R1 contract | Serial and OpenMP | Required host RTC | No |

All four modes support ASE energy, coordinate forces, and stress within their
model/backend boundary. MACEField polarization and analytical response follow
the field-specific restrictions below. The non-Kokkos serial evaluator does
not implement streamed-edge execution. Format-v3 MH-1 is not supported by the
LAMMPS pair styles.

`execution_profile` is a separate direct-Kokkos selection policy. `speed`
keeps the normal retained baseline. Default `capacity` selects the fastest
qualified estimated fit and otherwise attempts the smallest qualified bundle.
The capacity bundle supports FP32 and FP64 standard MACE and MACEField,
including energy, forces, stress, and polarization. Analytical polarizability
and Born charges reconstruct overwritten state before response. MH-1 uses its
separate scratch-budget and node-state recomputation controls. Observers and
parameter gradients are incompatible with a capacity request because
capacity eligibility must be established before graph-time selection.

The LAMMPS MPI qualification matrix is narrower than the evaluator matrix and
is maintained in `pair_symmetrix/README.md`. Same-node CUDA-aware multi-GPU
qualification covers FP32 standard MACE and FP32 direct low-memory MACEField.
The MACEField capacity gate uses two A100-80GB GPUs, one OpenMPI 4.1.6 rank per
GPU, and a `1x1x2` decomposition. Rank-local capacity records are gathered once
per topology rebuild and written by rank zero in rank order, so diagnostics are
not lost when the launcher suppresses non-root standard error. FP64 CUDA, HIP
MPI, other MPI providers, and cross-node GPU-aware transport remain
qualification targets.

### Direct-streamed generated specialization

The model-specialized implementation currently supports:

- ordinary, compact standard-MACE architectures already supported by the
  Symmetrix-XL MACE extractor/evaluator and carrying a factorized R1 contract;
- compact MACEField architectures carrying the same R1 contract; field
  conditioning remains outside the generated edge kernel and is differentiated
  by the native field-aware stages;
- compatible two-layer MACE-MH-1-family models carrying an MH1 UVU contract;
- Kokkos OpenMP, Kokkos Serial, and Kokkos CUDA execution spaces for ordinary
  MACE, MACEField, and generated MH-1 specialization;
- JIT-generated ordinary-MACE and MACEField R1 plugins with
  `dtype="float32"` or `dtype="float64"`;
- architecture-generic M0/R0 standard modules and JIT-generated MH-1
  specialization with `dtype="float32"`, plus host-only MH-1 generated spline
  R stages with `dtype="float64"` and packed Kokkos M stages;
- fixed learned parameters with generated forward and first coordinate reverse
  execution for energies, forces, and stress.

The ordinary R1 host/CUDA plugin ABI is version 2. It records scalar kind and
size explicitly, uses precision-qualified artifact IDs and cache keys, and
keeps geometry and coordinate derivatives as doubles. Loaders still accept
legacy ABI-v1 FP32 plugins; an ABI-v1 artifact is never admitted to an FP64
evaluator.

Parameter gradients, species/type gradients, and double backward are not part
of the streamed-edge execution. The supported reverse contract is the coordinate
derivative required for energy forces and stress.

Foundation compatibility records are organized by normalized factorized R1 contract,
not by checkpoint filename. Each representative records provenance for a
checked-in contract used by tests and runtime JIT. Exact aliases
in the third column were independently compared with that representative
contract. The final column lists foundation catalogue architectures expected
to have the same broad shape but not independently established as exact aliases
in this qualification:

The machine-readable records are in
`symmetrix/test/data/execution_contracts/foundation_model_registry.json`. They contain no
per-model AOT artifact IDs; model specialization is a runtime JIT concern.

| Contract group | Representative contract input | Verified exact aliases | Other same-shape foundation candidates |
|---|---|---|---|
| Standard L0 | MACE-OMAT-0 small | None claimed | MACE-MP-0b small; MACE-MP-0b2 small |
| Standard L1 | MACE-MPA-0 medium | MACE-OMAT-0 medium; MACE-MP-0b3 medium | MACE-MP-0b medium; MACE-MP-0b2 medium; MACE-MATPES; MACE-OFF23 medium |
| Standard L2 | MACE-MP-0b2 large | None claimed | None |
| Existing rounded L1 | Historical `omat-medium` contract | MACE-MH-0 | None |
| OFF23 L0 | MACE-OFF23 small | None claimed | None |
| OFF23 L2 | MACE-OFF23 large | None claimed | None |

Neither table membership nor a checkpoint filename admits generated code. The
loaded model supplies the normalized contract, semantic fingerprints, channels,
radial embedding, angular/path multiplicities, and literal tensor-product
payload used for the exact JIT cache identity. A candidate with any difference
gets a distinct artifact even when it belongs to the same named foundation
family.

An eligible ordinary-MACE, MACEField, or compatible MH-1 contract receives its
own cached backend-specific JIT artifact. JIT compilation or loading failure
raises for direct execution. Changing supported channels or other contract
dimensions therefore does not require rebuilding Symmetrix-XL.

Specialization does not currently provide:

- model-parameter gradients or double backward;
- generated tangent kernels for MACEField polarizability and Born effective
  charge calculations; analytic response evaluation retains the generic
  native response path;
- dynamic JIT specialization for nonlinear architectures outside the
  compatible MACE-MH-1 family;
- Windows shared-library plugins.

MACE-MP-0a checkpoints with residual first interactions use the ordinary-MACE
evaluator and are eligible for factorized R1 specialization. MACE-OMOL and Polar
architectures remain outside the current ordinary-MACE specialization boundary;
this specialization does not claim support for those architectures.

These specialization restrictions do not narrow the generic algorithm or the
normal Symmetrix-XL model support matrix.
