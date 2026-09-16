# MH0 direct GPU profile on 864-atom AlN

Date: 2026-08-20

## Workload

- Revision: `9547dfa` plus the profile-harness updates in this worktree.
- Device: AMD Radeon 8060S Graphics, `gfx1151`, 40 reported compute units,
  wavefront 32, ROCm 7.14.
- Model: official MACE-MH-0 `omat_pbe` head, Al/N extraction SHA-256
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`.
- Native extension SHA-256:
  `c1b93dc1e612186532eb1077029f86e5c51c4c78e6f26201d8d81bd04033b319`.
- Structure: 864-atom wurtzite AlN, `a=3.112 A`, `c=4.982 A`, repeated
  `6x6x6`.
- Model cutoff: `6.0 A`; neighbor-list skin: `0 A`; effective cutoff:
  `6.0 A`; directed edges: 78,624.
- Execution: float32, prepared graph, energy and forces,
  `streamed_edges="direct"`, hipRTC, five warmups and one selected-region
  profile step.
- JIT artifact: `jit-r1-gen2-f32-29f6298ae84d2005`; wave32 edge policy,
  64 threads per block and eight persistent blocks per HIP-reported compute
  unit (`160` blocks total).

The harness asserted canonical direct mode, forced the checked-in MH0 static
artifact off, and observed zero fallback evaluations. It also checked the
physical graph fingerprint and existing numerical scalar gates. Energy matched
the reference exactly; the force-norm difference was `1.11e-5 eV/A`.

## Profile result

The uninstrumented smoke step was **25.031 us/atom** (`21.627 ms`). The
synchronized rocprofv3 trace step was **26.066 us/atom** (`22.521 ms`), of
which GPU kernels account for **25.350 us/atom** (`21.902 ms`). The difference
between the smoke and trace wall times is profiler overhead, so optimization
decisions below use the trace's kernel durations and the smoke timing as the
lower-overhead end-to-end reference.

The trace contains 39 launches over a `22.054 ms` device dispatch span. Kernels
occupy `99.31%` of that span: all inter-kernel gaps total only `0.151 ms`
(`0.175 us/atom`), and the largest gap is `16.6 us`. Median kernel duration is
`166.3 us`. Eighteen kernels are shorter than `100 us`, but together account
for only `0.257 ms` (`1.18%` of kernel time). Eliminating every measured gap
would improve device span by at most `0.69%`; launch fusion is therefore not a
primary optimization direction for this workload.

| Kernel | Launches | us/atom | Device ms | Share |
|---|---:|---:|---:|---:|
| M1 reverse | 1 | 5.016 | 4.333 | 19.79% |
| Generated direct R1 reverse | 1 | 4.969 | 4.293 | 19.60% |
| Standard R0 forward | 1 | 2.200 | 1.901 | 8.68% |
| Standard R0 coordinate reverse | 1 | 2.026 | 1.750 | 7.99% |
| Generated direct R1 forward | 1 | 1.727 | 1.492 | 6.81% |
| A1 forward | 1 | 1.617 | 1.397 | 6.38% |
| A1 reverse | 1 | 1.452 | 1.255 | 5.73% |
| M1 forward | 1 | 0.998 | 0.862 | 3.94% |
| H2 reverse | 1 | 0.847 | 0.732 | 3.34% |
| Standard M0 reverse | 1 | 0.633 | 0.547 | 2.50% |

The first seven kernels account for `74.98%` of all kernel time. The semantic
trace analyzer does not yet recognize every generic Kokkos or generated R1
name, so the table above uses exact kernel-name inspection rather than its
category totals.

## Hot-kernel efficiency

Hardware metrics were collected in separate selected-region rocprofv3 passes
because these counter blocks cannot all be scheduled together. Hardware
occupancy below is rocprofiler's `OccupancyPercent`, distinct from the temporal
device occupation reported above. Fetch is warmed-workload DRAM read traffic
reported by `FETCH_SIZE`, not total cache or LDS traffic.

| Kernel | Trace ms | WG | VGPR | LDS B | Occupancy | WGP util. | Mem unit | L2 hit | DRAM read KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 reverse | 4.333 | 1,024 | 56 | 512 | 97.94% | 99.49% | 72.83% | 53.40% | 7.3 |
| Generated direct R1 reverse | 4.293 | 64 | 232 | 2,560 | 23.10% | 99.89% | 72.76% | 64.46% | 872.9 |
| Standard R0 forward | 1.901 | 32 | 24 | 512 | 81.26% | 99.15% | 96.96% | 72.23% | 581.8 |
| Standard R0 coordinate reverse | 1.750 | 256 | 64 | 512 | 49.47% | 99.78% | 91.11% | 76.51% | 31.0 |
| Generated direct R1 forward | 1.492 | 256 | 120 | 0 | 65.45% | 99.68% | 81.48% | 94.37% | 36.0 |
| A1 forward | 1.397 | 256 | 48 | 512 | 79.15% | 99.12% | 98.16% | 95.68% | 18.9 |
| A1 reverse | 1.255 | 256 | 48 | 512 | 89.20% | 98.89% | 99.24% | 95.18% | 16.6 |

The R1 reverse launch is the clearest occupancy problem. Its 160 workgroups
keep every WGP active, but only `23.10%` of the available wave slots are active.
rocprofiler reports a 232-VGPR allocation. The code object reports 230 VGPR,
86 SGPR, 2,072 B static shared storage, and no VGPR/SGPR spills or private
storage; rocprofiler rounds this to the dispatch allocation granularity. The
kernel is not spilling, but its large live register set limits resident waves.
Its `872.9 KiB` of DRAM reads over `4.293 ms` is far too little for external
memory bandwidth to explain the runtime.

M1 reverse has the opposite profile: it is the largest kernel, but reaches
`97.94%` occupancy and `99.49%` WGP utilization. More resident work is unlikely
to help. The next investigation should distinguish VALU/instruction latency,
on-chip memory operations, and synchronization rather than tune the grid for
occupancy.

R0 forward/reverse and A1 forward/reverse show `91-99%` memory-unit activity,
while their warmed DRAM reads remain below `0.6 MiB` per launch. These kernels
are cache-resident but memory-instruction-heavy. The relevant target is fewer
or better-coalesced cache/LDS transactions and removal of intermediate
reads/writes, not DRAM bandwidth. R1 forward has adequate `65.45%` occupancy,
a `94.37%` L2 hit rate, no compiler-reported spills, and is lower priority than
R1 reverse.

The standard M0 reverse kernel is a secondary resource concern: the trace
reports 192 VGPR and 36 B scratch per thread, but it accounts for only `2.50%`
of kernel time.

## Improvement directions

1. Reduce live ranges and VGPR demand in
   `symmetrix_factorized_reverse_fused_v1`. This kernel combines a `19.60%`
   time share with only `23.10%` occupancy. Reuse path/channel invariants,
   shorten accumulator lifetimes, and stage independent reduction phases.
   Compare a staged split only if the occupancy gain exceeds its added global
   traffic and launch cost.
2. Re-screen the direct reverse edge policy after reducing registers. The
   current wave32, 64-thread launch has 160 persistent blocks and already
   covers all WGPs; test 32 versus 64 threads and four/eight/sixteen persistent
   blocks per HIP-reported compute unit. More blocks alone cannot overcome the
   current VGPR residency limit.
3. Optimize M1 reverse at the instruction and on-chip data-path level. It is
   `19.79%` of kernel time with essentially full occupancy. Use instruction
   counters or thread trace to quantify VALU, scalar, LDS, and synchronization
   stalls before changing its 1,024-thread mapping.
4. Reduce memory instruction count and intermediates across standard R0 and
   A1. The R0 pair consumes `16.67%` and the A1 pair `12.11%` of kernel time,
   with very high memory-unit activity but little warmed DRAM traffic. Fusion
   is useful here only when it eliminates cache/LDS traffic, not merely a
   launch.
5. Defer general launch fusion. The measured dispatch-gap ceiling is `0.69%`.
   Inspect the M0 reverse scratch allocation after the four larger families;
   even removing that kernel completely would save only `0.633 us/atom`.

Raw trace CSVs, counter passes, normalized R1 resource metadata, and harness
reports are under the ignored local directory
`benchmarks/.artifacts/mh0-direct-gpu-profile-20260820/`.
