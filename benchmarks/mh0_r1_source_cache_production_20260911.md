# MH-0 CUDA R1 reverse source-cache experiment

Date: 2026-09-11

## Scope

This experiment tests two R1-reverse changes in the production
`streamed_edges="direct"`, capacity-retained path:

1. Load the source's `4 x 128` FP32 feature slab into shared memory once per
   source and reuse it for every outgoing directed edge.
2. Assign two independent 64-thread source owners to one 128-thread block.

The second schedule needs independent force accumulation. Its tested form used
one FP64 `atomicAdd` per coordinate from each warp leader, or six FP64 atomics
per active edge. The atomic-only intermediate separates that cost from the
two-owner schedule.

## Workload

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0
- Driver: 610.43.02
- CUDA: 13.3
- Model: MH-0 Al/N, SHA256
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`
- Precision: FP32 evaluator
- Execution plan: `mh0-direct-capacity-retained`
- Cutoff: 6 A
- Neighbor-list skin: 0 A
- Effective cutoff: 6 A
- Radial point cache: `interval-x-f64-v1`
- Radial edge ordering: disabled
- Geometry: deterministic wurtzite AlN with 0.02 A Gaussian displacement
- Sizes: 864 atoms / 78,624 directed edges and 4,000 atoms / 364,000
  directed edges
- Timing: three fresh processes, 10 warmups and 31 CUDA-event repeats per
  process; the reported value is the median of process medians

Baseline extension SHA256:
`21b91421f3716574a0e21bd7bb6e17da43cb8c076116b0ef1355a9a49232088c`

Cache-only extension SHA256:
`46a44fa15b095f800ffdf130c5deb6e720d8d2e66a439256cfbd5713a2b2f263`

Two-owner extension SHA256:
`2718fa98cb170f3b73fdc78947b3274a7217052e860c955d726843b1df020dfb`

## Production timing

| R1-reverse implementation | 864 atoms (us/atom) | vs cache | 4,000 atoms (us/atom) | vs cache |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3.229882 | +0.463% | 2.875943 | +0.563% |
| Shared source cache | **3.214999** | reference | **2.859839** | reference |
| Cache plus warp atomics, one source/block | 3.240486 | +0.793% | 2.870321 | +0.367% |
| Cache plus warp atomics, two sources/block | 3.227340 | +0.384% | 2.938855 | +2.763% |

The shared cache improves the full production calculation by 0.461% at 864
atoms and 0.560% at 4,000 atoms. It is retained.

The two-owner schedule recovers part of the atomic-only loss at 864 atoms but
remains slower than cache-only. At 4,000 atoms it is substantially slower.
It is rejected together with the warp-atomic force path.

## Correctness

Cache-only versus baseline at 4,000 atoms:

- Energy difference: 0 eV
- Maximum force difference: `2.66e-15 eV/A`
- Maximum stress difference: `6.94e-18 eV/A^3`

Two-owner versus baseline at 4,000 atoms:

- Energy difference: 0 eV
- Maximum force difference: `7.074e-8 eV/A`
- Maximum stress difference: `5.394e-11 eV/A^3`

The larger two-owner differences come from changing the force reduction order
and using FP64 atomics. They are small, but the candidate fails the performance
gate independently.

## Kernel resources and counters

`cuobjdump --dump-resource-usage` for the generated R1 fused reverse kernel:

| Implementation | Registers/thread | Static shared memory | Stack/local spills |
| --- | ---: | ---: | ---: |
| Baseline | 96 | 3,104 B | 0 B |
| Shared source cache | 96 | 5,152 B | 0 B |
| Two source owners | 96 | 9,216 B | 0 B |

Focused Nsight Compute comparison at 4,000 atoms for baseline versus the
accepted cache-only kernel:

| Metric | Baseline | Shared source cache |
| --- | ---: | ---: |
| Replayed kernel duration | 2.47 ms | 2.41 ms |
| L2 throughput | 87.09% | 85.90% |
| Long-scoreboard issue share | 54.81% | about 52.1% |
| Achieved occupancy | 32.04% | 31.96% |
| Registers/thread | 96 | 96 |
| Spills | 0 | 0 |

Report: `/tmp/symmetrix-r1-source-cache-gen11/r1-reverse-cache-4000.ncu-rep`

The cache removes repeated source-feature L2 accesses within the source edge
loop without changing register pressure or occupancy. The measured reduction
in L2 throughput and long-scoreboard stalls agrees with that mechanism.

The two-owner kernel was not collected with Nsight Compute because it failed
the production timing gate. Its static resource report already shows that
shared memory nearly doubles, while the required FP64 atomics add serialization
on the same edge-force addresses. A future two-owner retry should use two
independent 64-thread named barriers and owner-local force partials so each
owner can combine its two warps without atomics.

## Validation

- Self-contained JIT code-generation suite: 47 passed
- `git diff --check`: passed
- Production artifact selection: generated R1 forward and reverse selected,
  zero fallback evaluations

Raw timing records:

- `/tmp/symmetrix-r1-cache-baseline-{864,4000}-{1,2,3}.json`
- `/tmp/symmetrix-r1-source-cache-gen11/candidate-{864,4000}-{1,2,3}.json`
- `/tmp/symmetrix-r1-source-cache-gen11/atomic-{864,4000}-{1,2,3}.json`
- `/tmp/symmetrix-r1-two-owner-gen11-b/two-owner-{864,4000}-{1,2,3}.json`
