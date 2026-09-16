# MH-0 factorized CPU R1 channel tiling

## Contract

- Model: `/tmp/mace-mh-0-current-Al-N.json`, SHA256
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`
- Structure: 864-atom periodic wurtzite AlN, `repeat((6, 6, 6))`
- Model cutoff: 6.0 A
- Neighbor-list skin: 0 A; effective cutoff: 6.0 A
- Directed edges: 78,624
- Precision and path: Float32 Kokkos OpenMP, factorized, generated R1,
  `reuse-adjoints-v1`, M1 recomputation, compact FP32 directions with FP64 radii
- Threading: physical core 0, one Kokkos/OpenMP thread, and one BLAS thread
- Timing: three warmups and seven measured synchronized evaluations; median
  `us/atom` is the primary metric

## Result

The host-generated R1 source and forward owners now process 16 contiguous
channels per source or receiver. Edge metadata and spline evaluation points are
loaded once per tile, while radial values and channel-contiguous tensors are
vectorized across the tile. The bounded local state is 512 bytes for compensated
source reverse and 2,560 bytes for forward. The change adds no atom- or
edge-scaled workspace. CUDA and HIP continue to use the original GPU renderers.

| Configuration | Median us/atom | Change from scalar receiver control |
| --- | ---: | ---: |
| Scalar R1, automatic edge-owned R0 | 1,691.610 | +3.2% |
| Scalar R1, receiver-owned R0 control | 1,639.110 | control |
| Source tile 16, scalar forward | 985.121 | -39.9% |
| Source and forward tile 16 | 677.855 | -58.6% |

Source tile sweep, with receiver-owned R0:

| Source channel tile | Median us/atom |
| ---: | ---: |
| 4 | 1,133.542 |
| 8 | 1,046.869 |
| 16 | 987.608 |

Forward tile sweep after selecting source tile 16:

| Forward channel tile | Median us/atom |
| ---: | ---: |
| 4 | 735.692 |
| 8 | 698.024 |
| 16 | 678.320 |
| 32 | 679.415 |

Tile 16 is retained for both owners. GCC 15.2 reports 512-bit vectorization for
all ten generated channel loops in each owner and for their final store loops.

## AVX2 and AVX-512 qualification

The source and forward channel tiles were also changed together and rebuilt for
each configuration. The AVX2 runs forced `SYMMETRIX_JIT_HOST_TARGET=haswell`;
the AVX-512 runs used `SYMMETRIX_JIT_HOST_TARGET=native` on the Ryzen AI Max+
395. Each run used a fresh JIT cache, and its artifact manifest recorded the
corresponding `-march=haswell` or `-march=native` compiler option. The workload
and one-thread timing contract are otherwise identical to the contract above.

| Source and forward tile | AVX2 us/atom | AVX-512 us/atom | AVX-512 change |
| ---: | ---: | ---: | ---: |
| 4 | 926.435 | 881.927 | -4.8% |
| 8 | 766.960 | 758.937 | -1.0% |
| 16 | 726.715 | 684.312 | -5.8% |
| 32 | 715.199 | 675.177 | -5.6% |

Tile 16 provides the large gain on both targets. Relative to tile 4, it reduces
the complete evaluation time by 21.6% on AVX2 and 22.4% on AVX-512. An adjacent
tile-16 control measured 729.652 us/atom on AVX2 and 678.262 us/atom on AVX-512;
against those controls, tile 32 was a further 2.0% and 0.5% faster. A mixed
source-32/forward-16 build measured 716.682 and 674.909 us/atom, respectively,
showing that the marginal tile-32 gain comes from source reverse. Forward tile
32 adds local state but no measurable benefit. Production retains tile 16 until
source tile 32 is qualified across additional channel counts and CPU families.

## Stage profile

The Kokkos timer totals below cover two evaluations and are normalized by
`2 * 864` atoms.

| Stage | Before us/atom | After us/atom |
| --- | ---: | ---: |
| Generated R1 source reverse | 699.97 | 50.41 |
| Generated R1 forward | 329.83 | 22.78 |
| R0 coordinate reverse | 235.46 | 208.62 |
| Generated R1 edge reverse | 39.48 | 39.44 |

R0 coordinate reverse is now the largest individual stage. Automatic R0
selection uses the receiver-owned implementation on host execution spaces and
keeps edge16 on CUDA and HIP.

## R0 reverse and H2 follow-up

The receiver-owned R0 host reverse now traverses each edge once and processes
128 contiguous channels at a time. Spline values and derivatives are retained
per lane while three lane-local force vectors accumulate across harmonic
components. The final lane reduction preserves the existing edge owner and
force update order. Its bounded local state is 2,560 bytes in Float32 and it
adds no atom- or edge-scaled workspace. CUDA and HIP retain their existing
team kernels.

| R0 host channel tile | Full median us/atom | R0 reverse us/atom |
| ---: | ---: | ---: |
| Scalar | 677.832 | 208.651 |
| 4 | 606.534 | 133.113 |
| 8 | 542.333 | 70.488 |
| 16 | 531.842 | 60.757 |
| 32 | 500.909 | 28.165 |
| 64 | 490.975 | 21.752 |
| 128 | 486.843 | 18.455 |

An isolated AOT Haswell build with `Kokkos_ARCH_NATIVE=OFF` and
`Kokkos_ARCH_HSW=ON` measured 514.964 us/atom and 28.344 us/atom for R0
reverse at tile 128. The native AVX-512 build measured 486.843 and
18.455 us/atom, respectively. Thus the selected tile remains effective on
AVX2 without relying on the host RTC target.

The next profile showed mixed-precision overhead in H2: Float32 node states
used Float64 H2 transform weights, and reverse promoted both products to
Float64 before narrowing each accumulation. Host Float32 builds now store the
fixed H2 weights in evaluator precision and narrow each H2 adjoint once per
reduction term. Float64 host builds are unchanged, and CUDA/HIP builds retain
Float64 H2 weights and their existing arithmetic. The change saves 192 KiB of
fixed model storage for this two-element, 128-channel model and adds no
workspace.

| Configuration | Full median us/atom | H2 forward us/atom | H2 reverse us/atom |
| --- | ---: | ---: | ---: |
| R0 tile 128 control | 486.843 | 20.058 | 55.687 |
| FP32 H2 weights and reverse | 460.070 | 9.493 | 36.418 |

The combined follow-up is 32.1% faster than the 677.832 us/atom scalar-R0
control. A StandardM0 experiment that merely grouped eight scalar channel
owners measured 486.707 versus 486.843 us/atom and left both StandardM0 stages
unchanged. The compiler did not vectorize across the large generated owner
call, so that wrapper-only change was rejected.

## A1 host BLAS dispatch

For this checkpoint, each A1 direction performs four fixed-weight matrix
products per node. The products have 128 output channels, 1, 3, 5, or 7
component rows, and 256 or 384 reduction elements. Forward and reverse each
perform approximately 1.132 GFLOP per 864-atom evaluation.

A direct 16-channel accumulator tile was rejected. It reread the input for
each output tile and increased the complete median from 460.070 to
491.079 us/atom. Its instrumented forward and reverse times were 70.021 and
61.609 us/atom, respectively.

The accepted host path dispatches one GEMM per `(node,l)` owner to the linked
BLAS implementation. The source, destination, and existing forward or
transposed weight matrices are already contiguous, so dispatch requires no
packing and adds no workspace. The Kokkos owner range supplies outer CPU
parallelism; CPU benchmark and production environments should keep the BLAS
thread count at one to avoid nested parallelism. CUDA and HIP retain the
existing `KokkosBatched::TeamGemm` implementation.

| A1 implementation | Full median us/atom | Forward us/atom | Reverse us/atom |
| --- | ---: | ---: | ---: |
| Kokkos TeamGemm control | 460.070 | 50.979 | 53.348 |
| Host BLAS | 373.417 | 9.780 | 9.434 |

The BLAS path sustains approximately 134.0 GFLOP/s forward and 138.9 GFLOP/s
reverse on one core. It reduces combined A1 time by 81.6% and complete
evaluation time by 18.8%. This result depends on an optimized BLAS backend;
the one-thread contract used OpenBLAS with `OPENBLAS_NUM_THREADS=1`.
A secondary run pinned to physical cores 0-3 used four Kokkos/OpenMP threads
and one OpenBLAS thread and measured 98.039 us/atom, a 3.81x speedup over the
one-core result.

## Numerical qualification

The final generated factorized path was compared with the independent
materialized path on the same graph. Maximum observed differences were:

- energy: `5.21e-4 eV`, or `6.03e-7 eV/atom`;
- force: `1.64e-5 eV/A`;
- stress: `9.91e-7 eV/A^3`.

After the R0 and host H2 follow-up, the independent materialized comparison
measured `5.93e-7 eV/atom` energy, `1.69e-5 eV/A` maximum force, and
`7.19e-7 eV/A^3` maximum stress differences. These remain within the existing
Float32 qualification envelope.

After host BLAS A1 dispatch, the 864-atom Float32 materialized comparison
measured `6.20e-7 eV/atom` energy, `9.13e-6 eV/A` maximum force, and
`5.48e-7 eV/A^3` maximum stress differences. A 32-atom Float64 smoke
comparison measured `3.55e-15 eV/atom`, `1.62e-14 eV/A`, and
`3.15e-15 eV/A^3`, respectively.

The host owner reference test passes for both ordinary and compensated source
callbacks. Host loader tests pass 7/7, and CPU-relevant JIT codegen tests pass
33/33 with two toolchain-dependent tests skipped. CUDA source compilation could
not be qualified locally because CUDA 13.1 conflicts with the system glibc
`rsqrt` and `rsqrtf` declarations; the CUDA and HIP renderers were not modified.
