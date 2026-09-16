# MH-0 CUDA microbenchmark with changing random geometry, 2026-09-11

## Purpose

The earlier focused CUDA benchmark held one geometry fixed across every warmup
and sample. Its radius initializer also repeated a short receiver/local-edge
pattern. That can overstate benefits from spline-interval locality and from
data retained in cache between evaluations.

The revised benchmark defaults to `--geometry md`. Before every warmup and
measured kernel launch it generates a new deterministic geometry with:

- edge-wise random radii spanning the full 6 A cutoff with uniform spline
  interval occupancy;
- a bounded 0.02 A frame-dependent radial displacement;
- edge-wise random unit directions with a bounded frame-dependent change; and
- freshly generated stored spherical-harmonic values.

Every variant receives the same geometry sequence. The benchmark reports the
tested kernel and the common geometry refresh separately, plus their combined
step time. `--geometry static` remains available for fixed-input diagnostics.
The refresh is a controlled invalidation workload, not a replacement for the
production position, neighbor-list, and SpheriCart pipeline.

## Method

The workload is FP32 MH-0 with cutoff 6 A and skin 0 A on an NVIDIA GeForce RTX
5090 (`sm_120`), driver 610.43.02. The pre-run device query reported 33 MiB of
32,607 MiB in use and no competing compute processes. Both sizes retain 91
directed edges per atom:

| Atoms | Directed edges |
| ---: | ---: |
| 864 | 78,624 |
| 4,000 | 364,000 |

Each cell below is the median of three fresh-process medians. Each process uses
10 warmups and 31 CUDA-event samples. The geometry seed is `20260908`. Times
are `us/atom` and include either only the named kernel or the geometry refresh
and named kernel in the `Step` columns.

## Harmonic recomputation

| Variant | 864 kernel | 864 step | 4,000 kernel | 4,000 step |
| --- | ---: | ---: | ---: | ---: |
| R0 stored | 0.267852 | 0.285556 | 0.225792 | 0.240456 |
| R0 lane recompute | 0.270222 | 0.285926 | 0.226816 | 0.241448 |
| R0 warp leader | 0.277333 | 0.295000 | 0.227840 | 0.242480 |
| R1 stored, restructured | 0.485926 | 0.503667 | 0.343040 | 0.357592 |
| R1 lane recompute through `l=0` | 0.296296 | 0.314296 | 0.289792 | 0.304856 |
| R1 lane recompute through `l=3` | 0.298704 | 0.317963 | 0.295936 | 0.310512 |

R0 remains neutral to slower. For R1, full recomputation appears 38.5% faster
at 864 atoms but only 13.7% faster at 4,000 atoms. The combined-step changes
are 36.9% and 13.2%.

### Why the R1 result does not translate

Changing the geometry every evaluation does not remove the synthetic R1 gain,
so reuse of harmonic values or spline intervals across benchmark samples is
not its cause. Generated-code inspection instead identifies a register
allocation threshold:

| Kernel | Registers/thread | Static `LDG` | SASS instructions |
| --- | ---: | ---: | ---: |
| Microbenchmark stored | 88 | 63 | 736 |
| Microbenchmark recompute through `l=0` | 80 | 62 | 736 |
| Microbenchmark recompute through `l=3` | 80 | 50 | 776 |
| Production generation-10 control | 96 | 70 | 744 |
| Production generation-11 `Y00=1` | 96 | 69 | 736 |

The SASS column counts disassembled opcodes. The equivalent encoded 64-bit
word counts for the two production kernels are 1,488 and 1,472.

The `l=0` specialization replaces only normalized `Y00`, whose value is
exactly one. It removes one static load and adds no harmonic arithmetic, yet it
obtains nearly all of the microbenchmark speedup and remains 0.8%/2.1% faster
than full `l=3` recomputation. Its important effect is the 88-to-80-register
change. With 256-thread blocks on this GPU, that crosses the boundary from two
to three resident blocks per SM. At 864 atoms the microbenchmark launches only
432 blocks, so admitting a third block also removes much of a partial second
wave. At 4,000 atoms the persistent grid reaches 1,360 blocks and the benefit
is smaller, matching the observed size dependence.

The exact MH-0 contraction has 40 live output accumulators, ten radial paths,
86 model coupling terms, and source-feature, coefficient, radial-point, and
harmonic inputs. Replacing `Y00` leaves that production kernel at 96 registers,
with no spills and the same launch and shared-memory configuration. It changes
only one of 70 static global loads and eight of 744 disassembled instructions,
so the production scheduler and occupancy remain effectively unchanged.

Nsight Systems confirms the consequence at 4,000 atoms: median generated R1
forward time changes from 1.226530 to 1.225346 ms, a 0.10% improvement. The
kernel accounts for 11.4% of GPU kernel time, while R1 reverse accounts for
22.4%, so a small forward-only change is further diluted. Three-process
end-to-end medians change from 3.228187 to 3.217838 `us/atom` at 864 atoms and
from 2.866677 to 2.863394 `us/atom` at 4,000 atoms, or 0.32% and 0.11%.

Therefore the microbenchmark demonstrates a compiler-sensitive contraction
shape rather than a robust arithmetic-for-memory improvement. Harmonic
recomputation remains rejected for production; a future candidate must be
tested inside the exact generated contraction and reduce its 96-register
allocation or measured critical-path stalls.

Maximum differences from the stored path are `2.38e-6`. Formula parity against
the checked SpheriCart implementation remains `1.91e-6`.

## Multi-edge R0

| Variant | 864 kernel | 864 step | 4,000 kernel | 4,000 step |
| --- | ---: | ---: | ---: | ---: |
| Stored single-edge control | 0.267852 | 0.285556 | 0.225792 | 0.240456 |
| Subgroup 16, 4 groups, 64 threads | **0.149333** | **0.167259** | 0.129536 | 0.144112 |
| Subgroup 16, 8 groups, 128 threads | 0.158852 | 0.176815 | **0.124928** | **0.139824** |

The best schedule improves the focused kernel by 44.2% at 864 atoms and 44.7%
at 4,000 atoms. Including geometry refresh, the improvements are 41.4% and
41.9%. Unlike the earlier crystalline-radius screen, one schedule no longer
wins at both sizes. This reinforces that the isolated schedule is sensitive to
workload and occupancy. The production split-`l` implementation was already
3.17% slower at 864 atoms and neutral at 4,000 atoms, so it remains rejected.

The maximum output difference is `4.05e-6`, from the changed FP32 reduction
order.

## Packed coefficients

| Phase | 864 scalar | 864 `float4` | Change | 4,000 scalar | 4,000 `float4` | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 forward | 0.291556 | 0.289185 | -0.81% | 0.265728 | 0.265216 | -0.19% |
| R0 reverse | 0.277333 | 0.272593 | -1.71% | 0.407048 | 0.403968 | -0.76% |
| R1 forward | 0.485926 | 0.410111 | -15.60% | 0.343552 | 0.286720 | -16.54% |
| R1 reverse | 0.531000 | 0.528593 | -0.45% | 0.529656 | 0.528640 | -0.19% |

Random, changing geometry reduces the earlier synthetic R1-forward gain from
21-25% to 16%. It remains a microbenchmark-only result: the matching production
prototype did not improve generated R1 forward and regressed end-to-end at 864
atoms. No packed layout is restored to production.

## Independent default-mode verification

A fresh build and three new processes per size repeated the complete suite with
the default `--geometry md`, 10 warmups, and 31 samples. The table compares the
median of the three process medians with the corresponding control:

| Candidate | 864 atoms | 4,000 atoms | Production result |
| --- | ---: | ---: | --- |
| R0 harmonic lane recomputation | 0.00% | +0.23% | Not advanced; isolated result is neutral |
| R1 full harmonic recomputation | -38.05% | -13.56% | `Y00` diagnostic only: -0.32%/-0.11% end to end |
| R0 multi-edge, best tested schedule | -44.74% | -44.82% | +3.17%/+0.01% end to end |
| R1 packed `float4` coefficients | -15.53% | -16.49% | +1.07%/-0.33% end to end |

Negative values are faster. The rerun therefore does **not** verify that the
rejected candidates become neutral in the microbenchmark. It reproduces the
same false positives for R1 recomputation, multi-edge R0, and packed R1
coefficients.

Changing geometry addresses only two earlier weaknesses: repeated spline
interval patterns and reuse of fixed harmonic values between launches. The
remaining mismatch is structural. The R1 benchmark uses representative rows
and compiler live ranges instead of the exact 86-term generated contraction;
the multi-edge kernel excludes the production Kokkos launch integration and
resource interaction; and the coefficient benchmark omits much of the
production feature, weight, adjoint, and output traffic. These differences are
the mechanisms already identified in the production failures.

Consequently, default MD geometry improves the benchmark as an algorithmic
screen, but does not make it a production-performance oracle. Acceptance still
requires inserting a candidate into the exact generated or Kokkos production
kernel and measuring the complete evaluator. Future microbenchmarks intended
to predict acceptance should extract the exact generated contraction and its
launch wrapper rather than approximate them.

The independent records are under
`/tmp/symmetrix-mh0-md-verify-20260911-results/`; the tested binary is
`/tmp/symmetrix-mh0-md-verify-20260911`.

## Nsight Systems check

The 4,000-atom capture contains 209 `refresh_md_geometry` launches: 11 launches
for each of the 19 harmonic and multi-edge variants. Its median duration is
0.056736 ms, or 0.014184 `us/atom`. The capture therefore confirms that every
warmup and measured evaluation received a fresh geometry and agrees with the
separately reported CUDA-event refresh medians.

The report is
`/tmp/symmetrix-mh0-md-random-20260911/system-4000.nsys-rep`.

## Artifacts

Raw JSON records are under
`/tmp/symmetrix-mh0-md-random-20260911/`. The tested binary is
`/tmp/symmetrix-mh0-md-microbench`.

Build:

```bash
/usr/local/cuda-13.3/bin/nvcc -O3 -std=c++20 -arch=sm_120 -lineinfo \
  -Xptxas=-v benchmarks/mh0_radial_point_microbench.cu \
  -o /tmp/symmetrix-mh0-md-microbench
```
