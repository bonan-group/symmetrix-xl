# OpenMP and SIMD Coding Standard

This note records the CPU-kernel rules learned from the true-parallel OpenMP
corruption investigation. It applies to the C++ core, generated host kernels,
Kokkos OpenMP execution, and native qualification utilities.

## Parallelism model

Treat Kokkos workers and SIMD lanes as separate concurrency levels:

- A Kokkos `RangePolicy` or `TeamPolicy` distributes owners such as nodes,
  edges, or channel tiles across OpenMP workers.
- `#pragma omp simd` distributes iterations within one worker across SIMD
  lanes. It does not make a scalar accumulator private or safe automatically.
- Threaded BLAS is a third parallelism level. Keep BLAS threads at one when
  Kokkos supplies outer parallelism, and do not assume that this makes
  concurrent application-thread CBLAS calls safe.

Do not introduce a raw OpenMP parallel region inside a Kokkos worker. A SIMD
loop within an owner kernel is acceptable when its sharing and reduction rules
are explicit. Prefer one Kokkos ownership level plus SIMD over nested teams.

## SIMD data-sharing rules

For every value written in an `omp simd` loop, prove that it is one of:

1. Lane-private storage.
2. A disjoint indexed output where each lane writes a unique element.
3. An explicit OpenMP reduction.

A scalar declared outside the SIMD loop remains shared across its lanes. Any
accumulation into it requires a reduction clause, even when the accumulation
is conditional:

```cpp
double scale_adjoint = 0.0;
#if defined(_OPENMP)
#pragma omp simd reduction(+:scale_adjoint)
#endif
for (int lane=0; lane<active_channels; ++lane) {
    input_adjoint(base+lane) = lane_adjoint[lane];
    if (capture_input_scale_adjoint)
        scale_adjoint += input_value[lane]*lane_adjoint[lane];
}
```

Do not rely on a compiler's current vectorization decision for correctness. A
missing reduction may appear correct when the compiler scalarizes the loop and
fail after changing compiler, architecture, optimization flags, or tile width.
Prefer a local reduction followed by one owner-level write over a lane-level
atomic when the ownership structure permits it.

Guard OpenMP pragmas with `_OPENMP`. Use Kokkos backend macros such as
`KOKKOS_ENABLE_OPENMP` for backend selection and initialization behavior; the
two tests answer different questions.

## External libraries inside workers

External CBLAS calls from Kokkos OpenMP workers are admitted automatically only
for a qualified provider. OpenMP OpenBLAS requires query symbols owned by the
CBLAS provider and the same OpenMP runtime as Kokkos. MKL's sequential layer is
the recommended MKL deployment because it creates no internal worker team;
GNU-threaded MKL requires the same single `libgomp` runtime as Kokkos and should
be used only when qualification shows a benefit. Every threaded route also requires
`omp_get_max_active_levels()` to be at most one. The optimized owner-local
kernels then execute concurrently across the outer Kokkos workers without
creating nested OpenMP teams.

The host-worker policy in `libsymmetrix/source/host_worker_blas.hpp` permits
CBLAS for Serial, sequential MKL on OpenMP, runtime-compatible OpenMP OpenBLAS,
runtime-compatible GNU-threaded MKL, and pthread OpenBLAS only for one-worker
OpenMP. Intel-threaded MKL, unresolved `libmkl_rt`, unidentified providers,
duplicate OpenMP runtimes, nested-enabled configurations, and pthread OpenBLAS
with multiple OpenMP workers select Kokkos. Merely
setting `OPENBLAS_NUM_THREADS=1` or `MKL_NUM_THREADS=1` does not prove that an
arbitrary BLAS implementation supports concurrent callers.
`SYMMETRIX_HOST_WORKER_BLAS=off` forces Kokkos, while `unsafe` is reserved for
controlled A/B diagnosis. Strict runtime qualification rejects concurrent
CBLAS when its threading backend cannot be verified.

Before changing this policy, run a standalone concurrent-call stress test with
independent buffers and a scalar reference. Record the actual library path,
the provider and threading layer, OpenMP runtime compatibility, and the maximum
active levels. A passing result on one BLAS build is evidence for that build,
not a general reentrancy guarantee.

## Initialization and runtime identity

Set `OMP_NUM_THREADS`, `KOKKOS_NUM_THREADS`, affinity, and BLAS thread variables
before Python starts. Assigning them after importing NumPy, the native
extension, or another OpenMP user can leave the runtime initialized with an
earlier team size and produce a false one-thread correctness result.

OpenMP builds pass the requested thread count explicitly to Kokkos. Serial and
accelerator builds must not receive meaningless OpenMP thread settings.
Do not suppress Kokkos initialization warnings during qualification.

Static `ldd` output is insufficient because the Python executable's
`DT_RPATH` participates in runtime selection. Use `symmetrix doctor --json` to
verify the symbol-owning OpenMP library, build-time hash, duplicate mapped
runtimes, Kokkos execution space and concurrency, actual worker IDs, and worker
affinity. A relocated runtime is acceptable only when it is byte-identical to
the recorded build runtime.

## Correctness and performance qualification

Source-text assertions and successful compilation do not qualify parallel
correctness. Changes to an OpenMP or SIMD kernel require compiled, fresh-process
tests that:

- compare 1/2/4/8-worker results with thread variables set before process
  initialization;
- prove the requested team participated and covered the allocated physical
  CPUs;
- include a conservative speedup threshold so an effective one-thread run
  cannot pass as multithreaded;
- compare generic and direct MACEField execution for energy, forces, stress,
  polarization, BECs, and polarizability;
- exercise affected SIMD tile widths, currently M0 widths 1/4/8/16, with width
  1 as the untiled owner-per-channel reference;
- instrument the affected branch when output parity alone cannot prove it ran;
- use FP64 for the primary corruption oracle and appropriate FP32 tolerances
  where that path is supported.

Run the same gate with an IEEE-oriented build when diagnosing compiler-sensitive
behavior: configure `SYMMETRIX_FAST_MATH=OFF` and set
`SYMMETRIX_JIT_HOST_FP_MODE=ieee`. Fast-math and IEEE results may differ in
rounding order, but neither mode may contain races, non-finite values, or
property-scale corruption.

The maintained integrated gate is
`benchmarks/openmp_full_property_qualification.py`. Record its model and binary
hashes, compiler, loaded OpenMP and BLAS identities, physical CPU sets, atom
count, cutoff, skin, effective cutoff, directed-edge count, numerical deltas,
and timing in `us/atom`.

## Review checklist

Before approving OpenMP/SIMD code, verify:

- every SIMD write is private, disjoint, or reduced;
- products used for flattened ranges are widened before multiplication;
- owner-level outputs are not written by multiple Kokkos workers without an
  intentional atomic or reduction;
- no external-library call is made concurrently from workers without an
  explicit qualified policy;
- environment-derived policy is read before the relevant runtime initializes;
- one-thread parity and true-parallel parity both pass;
- worker participation and speedup are measured rather than inferred;
- Serial, OpenMP, and accelerator initialization behavior remains separated.
