# OpenMP scaling investigation (2026-08-26)

## Scope

This record investigates declining OpenMP parallel efficiency from
`streamed-edge` commit `68a073ce2115beef96820bd4085551c363ead6f3`, then
evaluates and implements a safe flattened/SIMD dense-transform backend. It
measures both the public ASE calculator boundary and the prepared factorized
evaluator, verifies actual worker participation and multithread numerical
parity, and attributes the remaining scaling loss with Kokkos Tools range
timings.

## Environment

- CPU: AMD Ryzen 9 9950X3D2, 16 physical cores and 32 SMT threads.
- Topology: one NUMA node and two 96 MiB L3 domains. Physical CPUs 0-7 use L3
  domain 0; CPUs 8-15 use L3 domain 1.
- Build: Release, native host architecture, GCC 15.2, Kokkos OpenMP only,
  OpenBLAS TPL, and CPython 3.12.
- Extension:
  `/tmp/symmetrix-openmp-scaling-68a073c-20260826/build/symmetrix.cpython-312-x86_64-linux-gnu.so`
- Baseline extension SHA-256:
  `f6c877c6c421e17f201b32f94596149ce508526331ecbbcd29a1ab8a7342507a`
- Candidate extension SHA-256:
  `c4f8ee2007faa9d2b92ad0121ff2e18d47c3b37167cc5d436999c34ea3126f17`
- OpenMP runtime: `/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`.
- BLAS: `/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.32.so`.
- MACEField model SHA-256:
  `52dceeca5ed876bc12e82835574cbc83b85a5055f834560cbb3acd426be31fdf`.
- Standard MACE model SHA-256:
  `03ba3e74170c2a51ba93dd0309728c9dbd094ce67d01cf8338993784b7eee574`.

Each timing point ran in a fresh process pinned to physical CPUs. Both
`KOKKOS_NUM_THREADS` and `OMP_NUM_THREADS` matched the physical-core count.
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` were one. The affinity was `OMP_PLACES=threads` and
`OMP_PROC_BIND=spread`. Generated direct execution was required, warmed before
measurement, and used a task-local JIT cache.

The primary workload was a periodic 6x6x6 wurtzite AlN supercell:

- 864 atoms;
- FP32 MACEField and standard MACE MH-0 models;
- model cutoff 6.0 A;
- electric field `[0.01, -0.02, 0.03]`;
- energy, forces, stress, and polarization for MACEField;
- energy, forces, and stress for standard MACE;
- `streamed_edges="direct"` and `low_memory=True`;
- three warmups, five measured calls, and three fresh trials per point.

All 60 primary timing workers selected the direct algorithm, JIT R1 forward
and reverse, standard M0/R0 modules, compact geometry, adjoint reuse, and zero
fallback evaluations.

## Baseline scaling

### Prepared production route

With `neighbor_skin=0.5 A`, the neighbor-list cutoff is 6.5 A and the prepared
graph contains 94,176 candidate directed edges. Candidates at or beyond the
model cutoff are clamped to 6.0 A and contribute zero radial value, but remain
in the prepared schedule. The public native route reuses the prepared graph and
geometry between steady-state calls.

| Physical cores | MACEField (us/atom) | MACEField speedup | Efficiency | Standard MACE (us/atom) | Standard speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 245.415 | 1.000x | 100.0% | 226.098 | 1.000x | 100.0% |
| 2 | 210.052 | 1.168x | 58.4% | 184.405 | 1.226x | 61.3% |
| 4 | 108.066 | 2.271x | 56.8% | 95.610 | 2.365x | 59.1% |
| 8 | 59.595 | 4.118x | 51.5% | 52.392 | 4.316x | 53.9% |
| 16 | 31.995 | 7.670x | 47.9% | 28.186 | 8.022x | 50.1% |

MACEField and standard MACE have the same scaling shape. Field coupling is not
the cause of the efficiency decline.

### Public exact-cutoff route

With `neighbor_skin=0`, the graph contains 78,624 directed edges. Native
geometry reuse is disabled, so every public calculator call rebuilds the graph
through ASE `neighbor_list("ijdD", atoms, 6.0)`.

| Physical cores | MACEField (us/atom) | Speedup | Efficiency | Cached-host control (us/atom) | Fixed public boundary (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 234.218 | 1.000x | 100.0% | 216.969 | 14.903 |
| 4 | 121.036 | 1.935x | 48.4% | 103.105 | 15.493 |
| 8 | 76.991 | 3.042x | 38.0% | 59.108 | 15.451 |
| 16 | 50.418 | 4.646x | 29.0% | 31.951 | 15.955 |

The approximately 15-16 ms ASE/result boundary is nearly independent of the
OpenMP thread count. It is the main reason the exact-cutoff public result looks
much worse than prepared evaluation. The production skin route should be used
for steady-state scaling claims.

### Larger-cell control

A 12x12x12 AlN supercell tests whether the 864-atom case simply lacks enough
parallel work. It has 6,912 atoms, 753,408 candidate directed edges, the same
6.0 A model cutoff, and a 0.5 A skin.

| Physical cores | ms/call | us/atom | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 1764.044 | 255.215 | 1.000x | 100.0% |
| 2 | 1454.704 | 210.461 | 1.213x | 60.6% |
| 4 | 771.064 | 111.554 | 2.288x | 57.2% |
| 8 | 436.190 | 63.106 | 4.044x | 50.6% |
| 16 | 233.348 | 33.760 | 7.560x | 47.2% |

The larger case has essentially the same 16-thread efficiency and slightly
worse per-atom time. Fixed launch overhead and insufficient total task count
are therefore not the primary limit; the larger working set increases
cache/memory pressure.

## Attribution

### Dense node-transform backend discontinuity

The largest baseline cause is a deliberate safety policy that changes the algorithm at
the one-to-two-thread boundary. `host_worker_cblas_enabled()` uses the fast
external CBLAS owner kernels only at Kokkos concurrency one. At two or more
threads it selects safe Kokkos TeamGemm implementations because concurrent
external CBLAS calls have not been portable across deployments.

The affected A1, H2, and H1 forward/reverse dense transforms have these
aggregate Kokkos Tools times:

| Physical cores | Selected implementation | Dense transforms (ms/call) | Speedup from one core |
|---:|---|---:|---:|
| 1 | host CBLAS | 19.621 | 1.000x |
| 2 | Kokkos TeamGemm | 75.600 | 0.260x |
| 4 | Kokkos TeamGemm | 37.279 | 0.526x |
| 8 | Kokkos TeamGemm | 19.216 | 1.021x |
| 16 | Kokkos TeamGemm | 9.655 | 2.032x |

This explains the unusually weak two-thread point and about 8.43 ms of the
16-thread deviation from ideal scaling. Forcing the Kokkos implementation at
one thread increases the full MACEField call from 220.437 to 340.881 ms.

As an attribution experiment only,
`SYMMETRIX_HOST_WORKER_BLAS=unsafe` produced 118.399/60.628/35.193/19.221 ms
at 2/4/8/16 threads, versus 182.148/93.024/51.672/27.774 ms with the automatic
safe policy in the same callback run. This demonstrates the available
performance but is not a qualified execution mode. `symmetrix doctor` correctly
rejects this policy for multiple workers, and the strict full-property test was
stopped before evaluation.

The required optimization is a safe vectorized dense-transform path that does
not issue concurrent external CBLAS calls from Kokkos workers. Enabling the
unsafe policy by default would reintroduce the OpenBLAS/runtime corruption risk
found on other systems.

### R1 source reverse cache pressure

After the dense-transform discontinuity, generated R1 source reverse is the
largest intrinsic scaling loss:

| Range | 1 thread (ms) | 2 threads | 4 threads | 8 threads | 16 threads | 16-thread speedup |
|---|---:|---:|---:|---:|---:|---:|
| R1 source reverse | 48.153 | 24.556 | 13.938 | 10.566 | 6.178 | 7.794x |
| R1 edge reverse | 39.065 | 19.952 | 10.232 | 5.330 | 2.833 | 13.787x |
| R1 forward | 21.795 | 11.516 | 5.907 | 3.134 | 1.821 | 11.971x |
| Field-H1 forward | 19.203 | 12.879 | 6.595 | 3.288 | 1.660 | 11.568x |

R1 source reverse uses one owner per source and 16-channel tile. Each owner
serially walks an irregular source-edge list, gathers receiver adjoints and
edge radial/harmonic data, evaluates generated coupling terms, and maintains
compensated accumulators. There are enough owners for parallel occupancy, but
their gathered working set is cache and memory sensitive.

Splitting eight workers across both L3 domains (`0-3,8-11`) improves MACEField
from 51.490 to 48.639 ms (5.5%). Kokkos profiling attributes 3.222 ms of that
gain to R1 source reverse alone: 10.566 to 7.344 ms. Splitting four workers
across the domains changes runtime by only 0.2%. This locates the cache-domain
limit near eight cores and distinguishes it from generic OpenMP launch cost.

### Secondary effects

- Remaining small ranges, reductions, view initialization, and per-range
  barriers become visible as the large ranges shrink.
- The host uses active `amd-pstate-epp` with the `powersave` governor. Ordinary
  access cannot read the MPERF hardware counter, so all-core frequency loss was
  not quantified. It can contribute to non-ideal scaling but does not explain
  the measured backend switch or L3-placement effects.
- `perf record` is unavailable because `kernel.perf_event_paranoid=4`; Kokkos
  callbacks were used for source-level attribution.

## Flattened/SIMD dense-transform experiment

The candidate replaces the multithread TeamGemm route for A1, H2, and H1
forward/reverse transforms with one Kokkos `RangePolicy` owner per node and
angular order. Each owner performs a small row-major matrix contraction with
the channel dimension innermost and explicitly SIMD-vectorized. It does not
pack matrices, allocate a workspace, or call external CBLAS from an OpenMP
worker. CUDA and HIP retain their existing device TeamGemm routes.

The final automatic policy is hybrid:

| Kokkos host workers | Automatic dense backend | Reason |
|---:|---|---|
| 1 | CBLAS | CBLAS remains 13.8% faster than the flat kernel on this workload. |
| 2 or more | flat/SIMD | Safe concurrent execution and 24-28% lower total runtime than TeamGemm. |

`SYMMETRIX_HOST_DENSE_BACKEND=flat` and `team` remain controlled overrides.
The selected backend is reported by `symmetrix doctor --json` under
`runtime.host_blas.dense_selected_backend`.

### Explicit flat A/B

The isolated flat override produced these prepared MACEField results before
changing the automatic selection:

| Physical cores | Baseline automatic (ms/call) | Flat (ms/call) | Flat change |
|---:|---:|---:|---:|
| 1 | 212.038 | 241.374 | 13.8% slower |
| 2 | 181.485 | 131.373 | 27.6% faster |
| 4 | 93.369 | 67.775 | 27.4% faster |
| 8 | 51.490 | 38.727 | 24.8% faster |
| 16 | 27.644 | 20.961 | 24.2% faster |

Kokkos range profiles show the flat dense family scaling from 47.914 ms at one
thread to 3.509 ms at 16 threads, or 13.655x. Diagnostic concurrent CBLAS is
still slightly faster locally at 16 threads, but remains unqualified and unsafe
as a portable default.

### Final automatic scaling

The final run used the same 864 atoms, 6.0 A model cutoff, 0.5 A skin, 6.5 A
effective cutoff, and 94,176 candidate directed edges as the baseline. Each
point is the median of three fresh trials, with three warmups and five measured
calls per trial.

| Physical cores | MACEField (us/atom) | Speedup | Efficiency | Standard MACE (us/atom) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 247.348 | 1.000x | 100.0% | 226.573 | 1.000x | 100.0% |
| 2 | 152.318 | 1.624x | 81.2% | 131.485 | 1.723x | 86.2% |
| 4 | 78.990 | 3.131x | 78.3% | 68.355 | 3.315x | 82.9% |
| 8 | 44.352 | 5.577x | 69.7% | 39.648 | 5.714x | 71.4% |
| 16 | 24.353 | 10.157x | 63.5% | 21.862 | 10.364x | 64.8% |

At 16 cores, the candidate improves total MACEField time by 23.9% and standard
MACE time by 22.4% relative to the baseline automatic policy. The one-thread
production route remains CBLAS and is unchanged apart from ordinary timing
variation.

## Correctness qualification

`benchmarks/openmp_full_property_qualification.py` passed in FP32 at 1/2/4/8
threads for both generic and direct modes. It compared energy, forces, stress,
polarization, BEC, and polarizability, plus standard-M0 tile widths 1/4/8/16.
The native worker sentinel observed exactly 1/2/4/8 distinct workers pinned to
the assigned CPUs, Kokkos concurrency matched the request, and only the
expected system `libgomp` was mapped.

The largest direct-versus-one-thread-generic absolute differences remained
within the configured FP32 tolerance: energy `1.367e-4 eV`, forces
`2.097e-5 eV/A`, stress `1.636e-6 eV/A^3`, polarization `1.705e-8`, BEC
`4.454e-6`, and polarizability `9.291e-6`.

Qualification record:
`/tmp/symmetrix-openmp-scaling-68a073c-20260826/openmp-full-property-fp32.json`.

The flat backend then passed the same strict gate in both FP32 and FP64 at
1/2/4/8 threads. The final hybrid automatic policy also passed in both
precisions and reported CBLAS at one thread and flat at 2/4/8 threads in both
generic and direct modes. The hybrid FP32 direct-mode maximum differences from
one-thread generic were energy `1.804e-5 eV`, forces `2.097e-5 eV/A`, stress
`1.765e-6 eV/A^3`, polarization `1.705e-8`, BEC `5.887e-6`, and
polarizability `1.364e-6`. FP64 differences remained at roundoff, with maximum
energy `2.842e-14 eV` and force `3.592e-14 eV/A` differences. An explicit
TeamGemm 1/2-thread smoke qualification also passed and reported `team` as the
selected dense backend.

Candidate qualification records:

- `/tmp/symmetrix-openmp-scaling-68a073c-20260826/openmp-full-property-fp32-flat.json`
- `/tmp/symmetrix-openmp-scaling-68a073c-20260826/openmp-full-property-fp64-flat.json`
- `/tmp/symmetrix-openmp-scaling-68a073c-20260826/openmp-full-property-fp32-hybrid.json`
- `/tmp/symmetrix-openmp-scaling-68a073c-20260826/openmp-full-property-fp64-hybrid.json`
- `/tmp/symmetrix-openmp-scaling-68a073c-20260826/openmp-full-property-fp32-team-smoke.json`

A fresh CUDA 13.3, Kokkos CUDA+Serial, Blackwell120 build also completed. The
modified shared translation units compile for CUDA without OpenMP pragma
warnings; the flat path remains host-only through the existing memory-space
dispatch.

The full OpenMP Python suite completed with 1,079 passed, 106 skipped, and 11
MH-1 failures. Re-running `test_mh1.py` with the explicit pre-change TeamGemm
route produced the identical 11 failures: six assertions that an OpenMP product
workspace is zero and five existing c12 direct/generic differences of about
0.002929 eV. They are not regressions from the flat backend. The first broad
run used the sandbox-inaccessible default JIT cache and was discarded; the
reported run used a task-local cache under `/tmp`.

## Recommended optimization order

1. Improve R1 source-reverse locality. Candidate approaches are receiver-data
   blocking, source scheduling grouped for cache reuse, or a generated two-pass
   layout that reduces repeated irregular gathers while retaining deterministic
   compensated accumulation.
2. Keep prepared geometry (`neighbor_skin > 0`) as the production steady-state
   path. Parallelizing ASE graph reconstruction is lower priority because it is
   absent when the prepared graph is reused.
3. Re-profile the new default before optimizing smaller reductions, memsets,
   and barriers. Their absolute contribution was secondary in the baseline and
   becomes more visible only after the dense-transform improvement.

## Reproduction commands

The primary per-point command used `<N>` equal to 1/2/4/8/16 and `<CPUSET>`
equal to `0`, `0-1`, `0-3`, `0-7`, or `0-15`:

```bash
env SYMMETRIX_SOURCE_ROOT="$PWD" \
  SYMMETRIX_EXTENSION=/tmp/symmetrix-openmp-scaling-68a073c-20260826/build/symmetrix.cpython-312-x86_64-linux-gnu.so \
  SYMMETRIX_JIT_CACHE=/tmp/symmetrix-openmp-scaling-68a073c-20260826/jit-cache \
  SYMMETRIX_JIT_POLICY=required SYMMETRIX_OPENMP_RUNTIME_CHECK=strict \
  KOKKOS_NUM_THREADS=<N> OMP_NUM_THREADS=<N> \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 OMP_PLACES=threads OMP_PROC_BIND=spread \
  taskset -c <CPUSET> \
  /tmp/symmetrix-fp64-low-memory-cuda-venv/bin/python \
  benchmarks/macefield_standard_aln_scale.py run \
  --backend openmp \
  --field-model /tmp/macefield-response-current-contract.json \
  --standard-model /tmp/mace-mh-0-current-Al-N.json \
  --output /tmp/openmp-<N>.json \
  --sizes n6 --policies optimized --trials 3 --warmups 3 --samples 5 \
  --low-memory --neighbor-skin 0.5 \
  --response polarization --response-routing public-native
```

The final automatic run sets no dense-backend override. Add
`SYMMETRIX_HOST_DENSE_BACKEND=flat` or `team` only for the controlled A/B
variants.

Kokkos range profiles used the same worker workload with five evaluations and:

```bash
export KOKKOS_TOOLS_LIBS=/tmp/symmetrix-openmp-scaling-6e40d9a-20260824/libkokkos_timer.so
export SYMMETRIX_KOKKOS_TIMER_OUTPUT=/tmp/openmp-profile.tsv
```

The strict correctness command was:

```bash
env SYMMETRIX_SOURCE_ROOT="$PWD" \
  SYMMETRIX_EXTENSION=/tmp/symmetrix-openmp-scaling-68a073c-20260826/build/symmetrix.cpython-312-x86_64-linux-gnu.so \
  SYMMETRIX_JIT_CACHE=/tmp/symmetrix-openmp-scaling-68a073c-20260826/jit-cache \
  SYMMETRIX_JIT_POLICY=required \
  /tmp/symmetrix-fp64-low-memory-cuda-venv/bin/python \
  benchmarks/openmp_full_property_qualification.py \
  --model /tmp/macefield-response-current-contract.json \
  --structure /tmp/openmp-runtime-qualification-aln.extxyz \
  --dtype float32 --threads 1 2 4 8 \
  --cpu-sets 0 0-1 0-3 0-7 --warmups 2 --samples 3 \
  --output /tmp/openmp-full-property-fp32.json
```
