# MH-0 direct CUDA optimization profile

Date: 2026-08-21

## Workload

- Device: NVIDIA GeForce RTX 5090, 170 SMs, `sm_120`, CUDA 13.3.73.
- Tools: Nsight Systems 2026.1.3 and Nsight Compute 2026.2.1.
- Baseline: `59e3bb1` plus the NVCC compatibility fix in
  `mace_kokkos_second_interaction.cpp`.
- Model: official MACE-MH-0 `omat_pbe` Al/N extraction, SHA-256
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`.
- Structure: 864-atom `6x6x6` wurtzite AlN; model cutoff `6.0 A`, skin
  `0 A`, effective cutoff `6.0 A`, and 78,624 directed edges.
- Execution: Float32, prepared energy and forces, `streamed_edges="direct"`,
  forced NVRTC, generated R1 forward and reverse, standard M0, standard R0
  `v2_edge16`, M1 recompute tile 32, full retention, and zero fallback.
- Baseline extension SHA-256:
  `19fe9229426f01232356ce50493765a6abdfecb32938c50dd92c6e267ec60873`
  (`48,277,192 B`).
- Candidate extension SHA-256:
  `ade8634cadfe2460d4f9312752083326a0e9ef3c3edb6166b7eaa9f7637abd68`
  (`51,549,728 B`).
- Experimental Float32-coordinate combined extension SHA-256:
  `5155df99ec5866f4f8744a1e6a2aba6e97283ddaf995ef6a447630d63e9a9ec6`.

Retained-candidate timing uses three sequential fresh processes per side, ten
warmups and thirty synchronized samples per process. Sections explicitly
labeled as stopping screens use their stated smaller sample count. Profiler
timings are diagnostic and are not used for end-to-end comparisons.

## Baseline timing and launch profile

| Fresh process | Median us/atom | p90 us/atom |
| ---: | ---: | ---: |
| 1 | 4.2421 | 4.2566 |
| 2 | 4.2438 | 4.2524 |
| 3 | 4.2436 | 4.2526 |

The median of process medians is **4.2436 us/atom**, or `3.666 ms` per
evaluation. The selected Systems trace contains 29 launches on one stream.
Kernels occupy `3.5188 ms` of a `3.5629 ms` dispatch span, or `98.76%`.
Inter-kernel gaps total only `44.0 us` (`0.0509 us/atom`), and the largest gap
is `28.0 us`. Removing every gap has a `1.24%` device-span ceiling, so launch
fusion is not the primary optimization target.

| Stage | us/atom | Device ms | Kernel share |
| --- | ---: | ---: | ---: |
| Generated R1 reverse | 0.857 | 0.740 | 21.0% |
| M1 reverse | 0.506 | 0.437 | 12.4% |
| Generated R1 forward | 0.381 | 0.329 | 9.4% |
| A1 forward | 0.337 | 0.291 | 8.3% |
| M1 forward | 0.307 | 0.265 | 7.5% |
| Standard R0 coordinate reverse | 0.291 | 0.251 | 7.1% |
| A1 reverse | 0.260 | 0.224 | 6.4% |
| Standard R0 forward | 0.250 | 0.216 | 6.1% |
| H1 reverse | 0.220 | 0.190 | 5.4% |
| H2 reverse | 0.132 | 0.114 | 3.2% |
| Remaining 19 launches | 0.532 | 0.460 | 13.1% |

Temporal device occupation above is distinct from achieved occupancy inside a
kernel. The trace is nearly continuously busy even though its two most
important kernels have too few resident or eligible warps.

## Baseline hot-kernel efficiency

| Metric | Generated R1 reverse | M1 reverse |
| --- | ---: | ---: |
| Grid x block | `864 x 64` | `864 x 32` |
| Registers/thread | 128 | 64 |
| Dynamic shared/block | 0 | 36.38 KiB |
| Static shared/block | 2.07 KiB | 48 B |
| Local spills | 0 | 0 |
| Theoretical occupancy | 33.33% | 4.17% |
| Achieved occupancy | 19.79% | 4.16% |
| Active warps/scheduler | 2.49 | 1.00 |
| Eligible warps/scheduler | 0.71 | 0.05 |
| Cycles with no eligible warp | 55.24% | 95.03% |
| Compute throughput | 40.59% | 5.47% |
| Memory throughput | 56.18% | 5.47% |
| L2 hit rate | 98.58% | 73.66% |
| Executed instructions | 590.98 M | 24.79 M |

R1 reverse is not DRAM-bandwidth limited: DRAM throughput is `3.25%`, while
the warmed L2 hit rate is `98.58%`. Its dominant warp-stall fractions are long
scoreboard `25.0%`, fixed-latency wait `23.5%`, selected plus not-selected
`28.4%`, and short scoreboard `8.7%`. Registers limit theoretical residency,
but the 864 source-owned blocks also provide only about ten warps per SM. Both
dependency length and available independent channel work matter.

M1 reverse is a stronger first target. Its 36.38 KiB per-block scratch permits
only two one-warp blocks per SM, and long-scoreboard dependencies account for
`79.4%` of its average issue delay. More threads in that existing team would
not create more useful channel work.

## Analytical M1 reverse candidate

The candidate admits the existing analytical Float32 device VJP for exact
Kokkos CUDA contracts. The flat range assigns one item to each
`node x component x channel`, retains the existing arithmetic body, and lets
Kokkos choose the CUDA range-policy block size. Forward remains on generic
recomputation. FP64, field-coupled, structurally unmatched, and aliased
input/input-adjoint contracts retain their prior paths.

Aliased storage is important: `reuse-adjoints-v1` aliases `A1_adj` with `A1`.
The flat component-parallel VJP cannot overwrite that storage while sibling
components still read it, so it now fails closed to the generic alias-safe M1
reverse. A first version without this check failed the adjoint-reuse force gate;
the corrected candidate passes both retained-versus-recompute and
reuse/rollback CUDA tests.

| Fresh process | Baseline us/atom | Candidate us/atom | Change |
| ---: | ---: | ---: | ---: |
| 1 | 4.2421 | 3.7690 | -11.15% |
| 2 | 4.2438 | 3.7712 | -11.14% |
| 3 | 4.2436 | 3.7739 | -11.07% |
| Median of medians | **4.2436** | **3.7712** | **-11.13%** |

The candidate p90 median is `3.7817 us/atom`, versus `4.2526 us/atom` for the
baseline, an `11.07%` improvement. The selected trace replaces the `0.437 ms`
M1 reverse with a `0.0257 ms` analytical kernel, a `94.1%` stage reduction.
Launch count remains 29. Total kernel time falls to `3.1068 ms`, and kernels
occupy `98.97%` of the `3.1390 ms` dispatch span. Inter-kernel gaps total
`32.3 us`, and the largest gap is `23.0 us`.

| M1 reverse metric | Baseline generic | Analytical CUDA |
| --- | ---: | ---: |
| Systems duration | 437.3 us | 25.7 us |
| Grid x block | `864 x 32` | `13,824 x 128` |
| Registers/thread | 64 | 56 |
| Dynamic/static shared | 36.38 KiB / 48 B | 0 / 0 |
| Local/shared spills | 0 / 0 | 0 / 0 |
| Theoretical occupancy | 4.17% | 75.00% |
| Achieved occupancy | 4.16% | 64.29% |
| Active warps/scheduler | 1.00 | 7.76 |
| Eligible warps/scheduler | 0.05 | 0.59 |
| Executed instructions | 24.79 M | 15.99 M |
| L2 hit rate | 73.66% | 90.81% |

The analytical kernel passes the retention gate: it improves end-to-end timing
by more than 1%, improves M1 reverse by more than 5%, and introduces no spills.
Its full-extension binary is `3.27 MiB` larger because NVCC instantiates the
fixed 16-component/94-term body. The fresh full build took `11m 50s`; an
incremental rebuild of the alias admission took `1m 14s`.

Final CUDA qualification also corrected M1 recomputation scratch admission.
The analytical, non-overlapping reverse uses no reverse scratch, so its active
FP32 requirement at tile 32 is the `18,176 B` forward tile
(`4 B x 142 polynomial nodes x 32 channels`). The generic reverse needs two
such workspaces, or `36,352 B`. Runtime policy now uses `18,176 B` for direct
non-overlap while reuse-adjoints and any planned alias retain the full
`36,352 B` requirement before buffers are allocated. A live Crucible probe of
the final extension confirmed both values, including restoration to
`18,176 B` after switching back from reuse-adjoints to full retention. Nine
focused CUDA tests pass on the rebuilt final extension.

## Rejected Float32 CUDA R1 spline-coordinate candidate

The first R1 experiment records spline-coordinate precision explicitly in the
generated-kernel schedule. Float32 CUDA contracts cast radius, spline origin,
and spline spacing once, then evaluate interval position and cubic coordinates
in Float32. CUDA Float64 remains on double coordinates. HIP Float32 retains its
existing Float32 path, while host code generation, R1 ownership, loop order,
launch geometry, and reduction order are unchanged.

Three fresh 864-atom processes produced `3.6910`, `3.6918`, and `3.6914
us/atom`. Their median is **3.6914 us/atom**, a further **2.12%** improvement
over the analytical-M1 candidate and **13.01%** over the original `59e3bb1`
baseline. Energy was identical across the three runs at
`-6407.774045069775 eV`.

The selected Systems trace still has 29 launches. Total kernel time is
`3.0493 ms` in a `3.0736 ms` dispatch span, or `99.21%` temporal occupation.
Gaps total `24.3 us`, and the largest gap is `14.9 us`. Generated R1 forward
falls from `329.2 us` to `296.2 us` (`-10.0%`), while generated R1 reverse
falls from `740.4 us` to `713.6 us` (`-3.6%`).

The reverse kernel retains its `864 x 64` launch, 128 registers/thread, zero
spills, and 2.07 KiB static shared memory. Nsight Compute reports 580.73 M
executed instructions, down from 590.98 M, but achieved occupancy remains
19.74% against 33.33% theoretical occupancy. It has 2.48 active and 0.73
eligible warps per scheduler, a 98.71% L2 hit rate, 3.39% DRAM throughput, and
54.35% of scheduler cycles with no eligible warp. The precision change removes
work but does not address the small source-owned grid or register-limited
residency.

The precision candidate passed the original controlled full-array comparison
against the double-coordinate analytical-M1 candidate. Both sides use the same
native extension and neighbor-list implementation. The matrix covers regular 216,
864, and 1,728 atom cells, a deterministically perturbed 864-atom cell, four
prepared repeats, and six moving-geometry steps. Directed-edge counts are
19,656, 78,624, and 157,248 at the three sizes. Worst absolute differences are
`2.5373e-4 eV` total energy (`1.4684e-7 eV/atom`), `8.0470e-7 eV` node energy,
`8.6910e-6 eV/A` force, and `1.2113e-7 eV/A^3` stress. All array gates pass at
`3e-5`, graph hashes match, generated forward/reverse remain active, and no
fallback occurs.

A later adversarial boundary qualification found a deterministic failure one
Float64 ULP below the first internal spline knot. At
`r=0.023529411765701957 A`, immediately below the knot at
`0.02352941176570196 A`, the double-coordinate control selects interval 0
while Float32 rounding selects interval 1. Across eight prepared repeats per
side, the maximum node-energy difference is `3.5591e-4 eV`, above the
documented `3e-5` full-array gate. The total-energy difference is
`3.5178e-4 eV`; force and stress remain small at `4.5868e-8 eV/A` and
`3.9968e-14 eV/A^3`. All intra-variant repeats are exact, generated forward
and reverse remain active, and fallback is zero.

The CUDA precision change is therefore rejected and CUDA Float32 generation
retains double spline coordinates. HIP keeps its pre-existing Float32
coordinate path; changing that separate qualified behavior is outside this
CUDA migration. The boundary artifacts are preserved under
`qualification-r1-spline-boundary/`.

## Path-local R1 harmonic grouping candidate

The next isolated schedule groups rows only when they belong to the same
radial path and share the same edge harmonic. It preserves ten radial paths,
76 row-adjoint loads and source updates, the 64-thread source-owned launch, and
the existing reduction order, while reducing Cartesian force-update groups
from 76 rows to 40 `(path, harmonic)` groups. It does not yet cache receiver
adjoints or group paths into global harmonic-signature bands.

Three fresh-process medians are `3.6274`, `3.6258`, and `3.6292 us/atom`.
Their median, **3.6274 us/atom**, improves the experimental Float32-coordinate
checkpoint by
**1.73%** and the original baseline by **14.52%**. The selected trace contains
29 launches and `2.9926 ms` of kernel work in a `3.0163 ms` span (`99.21%`),
with `23.7 us` total gaps and a `14.2 us` maximum gap. R1 reverse falls from
`713.6 us` to `648.9 us` (`-9.1%`).

The grouped reverse still uses 128 registers/thread, 2.07 KiB static shared
memory, and no spills. Executed instructions fall from 580.73 M to 446.35 M
(`-23.1%`). Achieved occupancy stays effectively flat at 19.93%, confirming
that the improvement is less work per edge rather than greater residency.
Active/eligible warps per scheduler are `2.49 / 0.57`, L2 hit rate is 98.52%,
and DRAM throughput is 3.69%. The candidate remains grid limited; its wider
numerical matrix passes. Against the Float32-coordinate checkpoint, energy and
node energies are unchanged; worst force and stress differences are
`4.0777e-7 eV/A` and `1.1478e-9 eV/A^3` across the regular, perturbed, and
moving-geometry cases. Both are below the stricter same-precision targets.
Graph hashes, launch counts, schedule rebuild counts, and zero-fallback status
match. The grouped cubin is 131,816 B versus 138,664 B for the precision-only
checkpoint.

## Global harmonic-band R1 schedule

The retained CUDA renderer groups radial paths by their complete harmonic
signature, evaluates each harmonic band outermost, collapses the radial-force
sum within the band, and caches the distinct receiver adjoints used by that
band. The complete schedule measured **3.5331 us/atom** across fresh processes,
or **16.74%** below the original `4.2436 us/atom` baseline and **2.60%** below
the path-local grouping checkpoint. Band grouping supplies most of the gain:
band rows alone measured about `3.5484 us/atom`; radial collapse alone added
only `0.22%`, while receiver caching added `0.46%` in isolated screens.

The selected Systems trace contains the same 29 launches and reports `564.7
us` (`0.654 us/atom`) for generated R1 reverse. Kernels occupy `99.24%` of the
dispatch span, so the gain comes from kernel work rather than launch removal.

| R1 reverse metric | Path-local grouping | Full harmonic band | Change |
| --- | ---: | ---: | ---: |
| Systems duration | 648.9 us | 564.7 us | -13.0% |
| Executed instructions | 446.35 M | 278.65 M | -37.6% |
| Registers/thread | 128 | 128 | unchanged |
| Achieved occupancy | 19.93% | 20.61% | +0.68 points |
| Active warps/scheduler | 2.49 | 2.52 | +0.03 |
| Eligible warps/scheduler | 0.57 | 0.33 | -0.24 |
| Cycles with no eligible warp | 62.05% | 74.12% | +12.07 points |
| L2 hit rate | 98.52% | 98.59% | +0.07 points |
| Local spills | 0 | 0 | unchanged |

The instruction reduction dominates despite fewer eligible warps. Full-array
qualification covers regular 216, 864, and 1,728 atom cells, a perturbed
864-atom cell, an 863-atom vacancy with 78,442 directed edges, and six moving
864-atom geometries. Worst force and stress differences are `5.38e-7 eV/A`
and `1.66e-8 eV/A^3`; generated forward/reverse stay active with zero fallback.

## R1 reverse launch-policy screen

The harmonic-band code shape was screened at 32, 64, and 128 threads per
source-owned reverse block. Each candidate used three fresh processes with ten
warmups and thirty synchronized samples.

| Reverse threads/block | Fresh medians, us/atom | Median of medians | Change vs 64 |
| ---: | --- | ---: | ---: |
| 32 | 3.62715, 3.62716, 3.62933 | 3.62716 | +2.66% |
| 64 | qualified control | 3.53310 | - |
| 128 | 3.84429, 3.85174, 3.85300 | 3.85174 | +9.02% |

The prior reduction in workgroup size removed threads without useful channel
work. A 256-thread block has more lanes than the 128-channel contraction can
use, while every lane still consumes warp, register, reduction, and scheduling
resources. Sixty-four threads keep both warps productive by processing two
channels per lane. The 32-thread result shows the other side of the tradeoff:
one warp per source does not expose enough independent work to hide latency.
The 128-thread result shows that four warps do not repay their increased
register-residency cost. The decision is based on throughput, not nominal
occupancy.

Persistent grid depths were then screened with the retained 64-thread block.
Depth 4 measured `3.89859 us/atom` in the stopping-rule screen. Three-process
median-of-medians were `3.53591`, `3.53228`, and `3.53641 us/atom` for depths
8, 16, and 24 respectively. Their `0.12%` spread is noise-sized and not
monotonic. Depth 8 is retained because it already caps the reverse launch at
all 864 sources; deeper queued grids provide no supported gain.

The qualified CUDA reverse block size remains fixed at 64 threads. The 32- and
128-thread variants are not retained as runtime or code-generation options;
future architecture-specific tuning should add a production policy only after
it is qualified on the target hardware.

## Experimental combined qualification

This CUDA 13.3 build uses the accepted analytical M1, the subsequently rejected
Float32 coordinates, full harmonic-band R1, 64 reverse threads, and depth 8.
Standalone generated FP32 and FP64 CUDA plugin compile/load tests pass. Three
fresh timing processes
measure `3.54035`, `3.53515`, and `3.53241 us/atom`; the median is **3.53515
us/atom** (`3.0544 ms/evaluation`), statistically unchanged from the qualified
`3.5331 us/atom` checkpoint and 16.70% below the original baseline.

The direct comparison against the prior full harmonic-band artifact covers
regular 216, 864, and 1,728 atom cells. Energy, node energies, and stress are
exact, and the worst force difference is `2.72e-15 eV/A`. Supplemental
self-consistency cases cover perturbed 864 atoms, an 863-atom vacancy, and six
moving 864-atom geometries. All repeated arrays are finite, with worst force
repeat variation `6.88e-15 eV/A`; graph hashes and generated launch-counter
deltas are stable, and fallback remains zero.

These timing and profiler results remain valid evidence for the M1,
harmonic-band, and launch-policy screens because their comparisons held
coordinate precision fixed. They are not the final production configuration;
the retained configuration restores double CUDA spline coordinates.

Artifacts are under the campaign-relative `results-final/` directory.
The experimental array archive SHA-256 is
`5a3a005491fa156f8a66ea7c0e88836913ebadda8f3e3c46bfebf4141b15d4c9`.

## Retained combined qualification

The production configuration restores double CUDA spline coordinates and
retains analytical M1, full harmonic-band R1, 64 reverse threads, and depth 8.
With the same 864 atoms, `6.0 A` cutoff, zero skin, 78,624 directed edges, ten
warmups, and thirty synchronized samples, three fresh processes measure
`3.6305468`, `3.6287662`, and `3.6313352 us/atom`. The median of medians is
**3.6305468 us/atom**, with median p90 `3.6427072 us/atom`. This is 2.70%
slower than the rejected Float32-coordinate configuration but still **14.45%
faster** than the original `4.2436 us/atom` baseline.

Every trial reports energy `-6407.774107469704 eV`, the same graph hash,
78,624 directed edges, thirty generated forward and reverse launches, and zero
fallback. The exact generated module SHA-256 is
`60b29a2662465e80b193dda7bed03ae1b65b1d73143fba28605ce22496fc7011`;
the cubin SHA-256 is
`aa7e168487f4f2f14915dbefcfcac5e04cf5935f401ed39a39f9bc163659fe10`.
The aggregate timing record is
`qualification-r1-spline-boundary/timing-c0-summary.json`, SHA-256
`4a1a016b2bb3506a6f322e7f08e3445dc6d4e5b0c0e8d9d39ab16d688e720349`.

## Rejected R1 reverse channel tiling

The remaining source-grid occupancy direction was tested by splitting the 128
channels into two disjoint 64-channel owners per source. This is exactly a 2x
grid expansion: `128 / 64 = 2` tiles, so the corrected reverse launch is
`1,728 x 64` instead of `864 x 64`. Each tile owns disjoint source-adjoint
channels, but both tiles must traverse every source edge and combine their
partial Cartesian forces with three Float64 atomics per edge.

This stopping screen used five synchronized samples, which measured `3.801994`
to `3.844991 us/atom`, with a median of **3.821324 us/atom**. This is **8.10%**
slower than its matched experimental `3.53515 us/atom` control. The corrected
Systems trace reports `817.314 us` for the `1,728 x 64` reverse kernel, versus
`564.7 us` for the accepted `864 x 64` kernel, a **44.7% stage regression**.

| R1 reverse metric | Accepted one owner | Two 64-channel owners | Change |
| --- | ---: | ---: | ---: |
| Grid x block | `864 x 64` | `1,728 x 64` | 2x blocks |
| Registers/thread | 128 | 128 | unchanged |
| Achieved occupancy | 20.61% | 25.48% | +4.87 points |
| Active warps/SM | 9.89 | 12.23 | +23.7% |
| Eligible warps/scheduler | 0.33 | 0.38 | +15.2% |
| Executed instructions | 278.65 M | 355.27 M | +27.5% |
| Global reductions | 0 | 471,744 | introduced |
| L2 hit rate | 98.59% | 98.65% | unchanged |
| Local/shared spills | 0 | 0 | unchanged |
| Warp cycles/instruction | 9.72 | 11.87 | +22.1% |

The additional blocks do raise physical occupation, but not enough to repay
their work. Long-scoreboard cycles/instruction improve from 4.63 to 4.13,
while short scoreboard rises from 0.68 to 2.09, fixed-latency wait from 1.91
to 2.67, not-selected from 0.26 to 0.42, and LG throttle from 0.08 to 0.19.
The 1,728-block launch is 1.27 full-device waves on 170 SMs at the
register-limited six-block residency, leaving a 368-block tail.

An initial loader experiment launched only 864 physical blocks. Each block
therefore processed two logical owners serially and measured `764.8 us` for
reverse and about `3.760 us/atom` end to end. It is retained only as a
code-cost diagnostic; it is not evidence for the intended full-grid schedule.
Against the untiled kernel, that diagnostic retained 128 registers/thread,
2.07 KiB static shared memory, and zero spills, while executed instructions
rose from 278.65 M to 355.04 M (`+27.4%`) and global loads rose from 38.94 M
to 50.97 M (`+30.9%`). It also executed 471,744 global reductions, exactly
`78,624 edges x 3 coordinates x 2 tiles`.

The candidate is rejected. Tiling does not reduce the 128-register limit, so
it does not create additional resident blocks. It instead duplicates edge
iteration and radial/harmonic setup, introduces contended Float64 force
atomics, and lowers useful work per block. Four or eight equal tiles would
scale those costs again while leaving only 32 or 16 active channel lanes in a
64-thread block. A future revisit would require a different two-phase design:
evaluate edge invariants once, write non-atomic tile partials, and reduce those
partials separately. Such a design should not proceed without generated-code
evidence of both lower register use and fewer total instructions.

The corrected full-grid candidate passes the same full-array qualification
against `results-final/final.npz`. All 24 keys, shapes, dtypes, graph hashes,
and launch-counter deltas match, and all values are finite. Energy and node
energies are exact; worst absolute differences are `2.694e-7 eV/A` for force
and `1.692e-9 eV/A^3` for stress. This passes both the `3e-5` CUDA gate and the
`1e-6` structural force target, so the rejection is performance-only.

The rejected artifacts are preserved under the campaign-relative
`results-r1channel2/` directory.
The corrected counter report is `ncu-fullgrid-c2-reverse.ncu-rep` and the
corrected Systems report is `nsys-fullgrid.nsys-rep`.
The candidate extension SHA-256 is
`8299c70b99e5d9a48888c0dc8cf9a2be01d4b919400ffe0f0058da3236cdae87`.

## Remaining qualification work

1. Re-screen M1 analytical launch geometry independently from R1.
2. Leave launch fusion last. The measured gap ceiling is too small to compete
   with M1 and R1 kernel-efficiency work.
