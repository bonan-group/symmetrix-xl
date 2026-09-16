# MH-0 CUDA harmonic and multi-edge experiments, 2026-09-10

## Scope

This investigation tests two independent optimizations for the FP32
`streamed_edges="direct"` capacity path on the AlN MH-0 workload:

1. recomputing normalized spherical-harmonic values from retained unit edge
   directions in R0/R1 forward; and
2. processing multiple receiver edges concurrently in R0 forward.

The benchmark GPU is an NVIDIA GeForce RTX 5090 (`sm_120`, 170 SMs), using
driver 610.43.02 and CUDA 13.3. The model cutoff is 6 A, neighbor-list skin is
0 A, and the structures are rattled with Gaussian displacement standard
deviation 0.02 A and seed `20260908`.

| Atoms | Directed edges | Edges/atom |
| ---: | ---: | ---: |
| 864 | 78,624 | 91 |
| 4,000 | 364,000 | 91 |

Each reported microbenchmark value is the median of three fresh-process
medians. Each process performs 10 warmups and 31 CUDA-event repeats. Results
are in `us/atom`.

## Harmonic implementation and parity

The microbenchmark accepts already-normalized FP32 directions. Its specialized
evaluators therefore omit norm accumulation, `rsqrt`, and coordinate rescaling.
The formulas are copied from the checked SpheriCart macro and evaluated in the
same MACE/e3nn component order and normalization. The apparent
`(dx,dy,dz) -> (z,x,y)` variable mapping remains only as the basis-convention
adapter; it is not counted as an optimization.

R0 instantiates separate `l=0`, `l=1`, `l=2`, and `l=3` kernels and evaluates
only the owned `2*l+1` values. R1 instantiates prefix evaluators through each
of those orders and compares lane-local evaluation, one warp-leader evaluation
with shuffles, and stored values. Higher orders remain loaded for partial
recomputation.

The source uses fixed-size arrays to express the formulas, but ptxas scalarizes
them: every harmonic variant has a zero-byte stack frame, zero spill loads and
stores, and its SASS contains no `LDL` or `STL`. The values are consumed
directly by the contraction.

Deterministic directions include the three axes and normalized `(1,1,1)`.
The remaining edge directions come from a deterministic integer hash and are
normalized before both evaluators run. Across all 364,000 directions and 16
components, the maximum difference from the checked SpheriCart evaluator is
`1.90735e-6`.

## Direction 2: harmonic recomputation

### R0 forward

| Variant | 864 atoms | Change | 4,000 atoms | Change | Maximum error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stored values | 0.267778 | - | 0.225352 | - | 0 |
| Recompute in every channel lane | 0.267630 | -0.06% | 0.226264 | +0.40% | 2.38e-6 |
| One warp leader plus shuffles | 0.275074 | +2.73% | 0.227392 | +0.91% | 2.38e-6 |

R0 value loads are already shared by a channel warp and each kernel computes
only one `l` block. Replacing those loads with arithmetic neither shortens the
critical path nor changes useful occupancy, while leader distribution adds
shuffle dependencies. R0 harmonic recomputation is rejected.

### R1 forward

| Variant | Registers | Static LDG | Static instructions | 864 atoms | 4,000 atoms | Maximum error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stored, restructured control | 88 | 63 | 736 | 0.468333 | 0.362728 | 0 |
| Lane recompute through `l=0` | 80 | 62 | 736 | 0.282333 | 0.238304 | 0 |
| Lane recompute through `l=1` | 80 | 62 | 744 | 0.281111 | 0.240728 | 9.54e-7 |
| Lane recompute through `l=2` | 79 | 57 | 760 | 0.282333 | 0.245728 | 1.43e-6 |
| Lane recompute through `l=3` | 80 | 50 | 776 | 0.277926 | 0.245024 | 1.67e-6 |
| Warp recompute through `l=0` | 80 | 117 | 1,104 | 0.284852 | 0.302088 | 0 |
| Warp recompute through `l=1` | 84 | 114 | 1,200 | 0.387926 | 0.323312 | 9.54e-7 |
| Warp recompute through `l=2` | 87 | 104 | 1,344 | 0.429407 | 0.347912 | 1.43e-6 |
| Warp recompute through `l=3` | 99 | 90 | 1,560 | 0.381963 | 0.307688 | 1.67e-6 |

All variants have zero spills. The warp-leader variants issue more static
loads because the generated shuffle access pattern lengthens code and retains
fallback loads; they are consistently worse than lane-local evaluation.

The large lane-local result is mainly a synthetic register threshold. The
microbenchmark falls from 88 to about 80 registers, while the actual generated
R1 forward kernel stays at 96 registers after replacing normalized `Y00` with
the exact constant one. Production SASS changes from 1,488 to 1,472
instructions and from 70 to 69 `LDG` instructions, with no occupancy change.

Three-process production timings confirm that the isolated change is within
run variation:

| Configuration | 864 atoms | Change | 4,000 atoms | Change |
| --- | ---: | ---: | ---: | ---: |
| Generation-10 control | 3.228187 | - | 2.866677 | - |
| R1 normalized `Y00=1` | 3.217838 | -0.32% | 2.863394 | -0.11% |

Energy is bitwise equal. Maximum force and stress differences are
`2.33e-15 eV/A` and `3.47e-18 eV/A^3`. The specialization fails the robust
two-size production gate and is not retained.

## Direction 3: multi-edge R0 forward

The multi-edge kernels keep separate compile-time `l` ownership. A block is
partitioned into channel subgroups, each of which traverses a different subset
of receiver edges. Each subgroup writes a private `(component,channel)` partial
to shared memory, followed by one deterministic reduction.

| Subgroup / edge groups / threads | 864 atoms | 4,000 atoms |
| --- | ---: | ---: |
| 16 / 2 / 32 | 0.195556 | 0.128768 |
| 8 / 4 / 32 | 0.163481 | **0.118080** |
| 16 / 4 / 64 | **0.145852** | 0.124976 |
| 8 / 8 / 64 | 0.172926 | 0.190256 |
| 16 / 8 / 128 | 0.160963 | 0.126504 |
| 32 / 8 / 256 | 0.267593 | 0.225280 |

The balanced 16-lane, four-group variant improves the focused kernel by 45.5%
at 864 atoms and 44.5% at 4,000 atoms. Across `l=0..3`, it uses 48, 56, 71,
and 96 registers, 2, 6, 10, and 14 KiB shared memory, and has zero spills.
Its maximum output difference is `3.81e-6`, caused by the changed FP32
reduction order.

Two production mappings were tested. A single runtime-`l` Kokkos launch
reserved the worst-case 14 KiB for every block and made R0 forward 36% slower
in Nsight Systems. Splitting the launch by `l` restored the intended shared
memory sizes, but added four Kokkos launches and remained slower at the small
system:

| Configuration | 864 atoms | Change vs R1-only | 4,000 atoms | Change vs R1-only |
| --- | ---: | ---: | ---: | ---: |
| R1-only control | 3.217838 | - | 2.863394 | - |
| Split-`l` multi-edge R0 | 3.319742 | +3.17% | 2.863546 | +0.01% |

Against the generation-10 control, split-`l` is 2.84% slower at 864 atoms and
0.11% faster at 4,000 atoms. The 864-atom numerical differences are
`5.28e-5 eV` in energy, `1.36e-5 eV/A` in force, and
`3.50e-8 eV/A^3` in stress. The schedule fails the production gate and is not
retained.

## Reverse gradient inspection

The selected MH-0 FP32 generated reverse path already uses
`WarpHarmonicGradients`: each lane computes the physical gradient for one
harmonic and the contraction reads the three scalar lane values through warp
shuffles. Although generated source still contains an unused declaration of
`direct_gradients[48]`, generated SASS has no local loads or stores.
`cuobjdump --dump-resource-usage` reports 96 registers, `STACK:0`, and
`LOCAL:0` for both fused and tiled reverse kernels. The 48-value temporary is
therefore neither a local-memory spill nor the selected computation. A second
direct-coordinate gradient microbenchmark is not warranted by the generated
code.

## Profiling and artifacts

Nsight Systems confirms the CUDA-event ordering for the standalone variants.
The 4,000-atom report is
`/tmp/symmetrix-mh0-harmonic-20260910/system-4000.nsys-rep`; raw three-process
records, ptxas output, and selected SASS are in the same directory.

A focused Nsight Compute attempt used a one-launch R1 harmonic filter and the
`SpeedOfLight` section. It failed with `ERR_NVGPUCTRPERM`; no hardware-counter
results are inferred from timeline timings.

The retained change is the focused microbenchmark coverage. Neither harmonic
recomputation nor the multi-edge R0 schedule advances to production.
