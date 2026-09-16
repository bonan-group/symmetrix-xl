# MACE-MH-1 Kokkos CPU inference investigation

Date: 2026-08-12

## Scope and workload

- CPU only; Kokkos OpenMP implementation.
- CPU: AMD Ryzen AI MAX+ 395, 16 physical cores / 32 hardware threads.
- Affinity: physical CPUs 0-15, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.
- BLAS threads: 1. Kokkos/OpenMP threads: 16 unless noted.
- Model: official MACE-MH-1 checkpoint, `omat_pbe` head, Float32.
- Checkpoint SHA-256: `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47`.
- System: 864-atom periodic wurtzite AlN.
- Exact-cutoff graph: 78,624 directed edges at the 6.0 A model cutoff.
- Timing: prebuilt graph, energy and analytic forces, three warmups and seven
  repeats for the primary Kokkos and PyTorch rows. Longer factorized controls
  use two warmups and five repeats.

## Results

| Implementation / policy | Threads | Edges | Median (us/atom) | Step (ms) | Relative to PyTorch | RSS after (MiB) | Edge workspace (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kokkos `all_interactions`, packed GEMM | 16 | 78,624 | 2,132.5 | 1,842.5 | 0.276x | 978.4 | 76.88 |
| Kokkos `all_interactions`, scalar linears | 16 | 78,624 | 3,551.2 | 3,068.2 | 0.460x | 925.0 | 76.88 |
| Generated host-v4 `factorized`, skin 0 | 16 | 78,624 | 3,860.6 | 3,335.6 | 0.500x | 494.4 | 91.51 |
| Generated host-v4 `factorized`, skin 0.5 A | 16 | 94,176 | 4,712.0 | 4,071.2 | 0.610x | 526.4 | 104.11 |
| Generic `factorized`, skin 0.5 A | 16 | 94,176 | 10,294.6 | 8,894.5 | 1.332x | 926.8 | 7.07 |
| PyTorch/MACE e3nn eager | 16 | 78,624 | 7,726.2 | 6,675.4 | 1.000x | not recorded | not recorded |

The PyTorch row uses MACE 0.3.16 and PyTorch 2.12.0, with
`enable_cueq=False`. Its seven samples span 6,633.2-6,771.7 ms. The primary
Kokkos row spans 1,822.9-1,848.6 ms.

## MH-0 (`omat_pbe`) generated-factorized control

The official MACE-MH-0 checkpoint provides a control for the intended
generated `factorized` CPU architecture. It uses the `omat_pbe` head, Float32,
and the same 864-atom AlN system, 16-core affinity, energy/force/stress
workload, and three-warmup/seven-repeat timing contract as the primary MH-1
measurements. The checkpoint SHA-256 is
`d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d`.

| Policy | Skin / effective cutoff (A) | Edges | Median (us/atom) | Step (ms) | Factorized speedup | RSS after (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| `all_interactions`, retained M1 | 0.0 / 6.0 | 78,624 | 449.140 | 388.057 | reference | 452.18 |
| Generated `factorized`, retained M1 | 0.0 / 6.0 | 78,624 | 175.412 | 151.556 | 2.56x | 466.49 |
| `all_interactions`, retained M1 | 0.5 / 6.5 | 78,624 | 448.011 | 387.082 | reference | 456.70 |
| Generated `factorized`, retained M1 | 0.5 / 6.5 | 94,176 | 208.798 | 180.401 | 2.15x | 482.23 |
| `all_interactions`, recomputed M1 | 0.5 / 6.5 | 78,624 | 450.711 | 389.414 | reference | 336.48 |
| Generated `factorized`, recomputed M1 | 0.5 / 6.5 | 94,176 | 211.594 | 182.817 | 2.13x | 362.22 |

At the exact 6.0 A model cutoff, generated factorized execution is 2.56x
faster than `all_interactions`. With the production 0.5 A neighbor-list skin,
factorized retains 94,176 candidate edges instead of compacting to the 78,624
edges within the model cutoff. Despite processing 19.8% more edges, it remains
2.15x faster.

The generated run built artifact `jit-r1-f32-29f6298ae84d2005`, used source
strategy `jit_plugin` and execution strategy `direct_jit_reverse`, selected
`jit_all` forward and `jit` reverse executors, and used the standard M0 and R0
modules. Its active unified factorized workspace was 0.73 MiB with the 0.5 A
skin.

The M1 `recompute` policy removes both the 59.91 MiB polynomial-value buffer
and the 59.91 MiB polynomial-adjoint buffer, replacing them with 35.5 KiB of
scratch. It adds only 1.3% to factorized latency and retains a 2.13x speedup.
At this 864-atom size, however, factorized process RSS remains 25.75 MiB above
`all_interactions` under the same recompute policy. The compact execution
workspace therefore does not imply lower whole-process RSS at this size;
capacity scaling needs a separate larger-system measurement.

This control confirms that factorization itself is not the source of the MH-1
regression. MH-0 combines the reduced-memory generated schedule with faster
execution. The MH-1 optimization target should preserve that architecture and
improve its generated host node lowering, especially the scalar dense loops,
to approach the quality of the MH-0 generated path.

`all_interactions` thread scaling is 19,522.2 ms at one thread, 2,850.7 ms at
eight threads, and 1,842.5 ms at 16 threads. This is 10.60x speedup at 16
threads, or 66% parallel efficiency relative to one thread. The 8-to-16-core
increment is only 1.55x, so top-end OpenMP scaling remains an optimization
opportunity, but it does not cause a regression against eager PyTorch here.

## Policy failures

`materialized` and `second_interaction` both fail before timing with:

```text
Kokkos affine MLP reverse tape was invalidated by a shared workspace user.
```

The conditioned convolution-weight and density MLPs share one
`AffineMLPKokkos` workspace. Retained forward execution records a workspace
generation in each tape. A later shared-workspace user advances that
generation before the earlier tape is reversed, so the validation in
`AffineMLPKokkosT::reverse_from_tape_impl` rejects the stale tape. Fully
streamed execution avoids this lifetime overlap by reversing within each
block. These policies need either disjoint retained tape storage or a schedule
that does not retain overlapping shared-workspace tapes.

## Implementation gaps

1. Generated host-v4 node execution is not a batched CPU lowering. It invokes
   one complete generated node owner per Kokkos iteration. The large dense E3
   linears execute as inline per-node loops rather than graph-wide packed GEMM.
   The scalar `all_interactions` control (3,068.2 ms) nearly matches generated
   factorized (3,335.6 ms), while packed GEMM reduces the same generic path to
   1,842.5 ms. The first CPU optimization should keep the shared generated
   edge/conditioning program but route large node linears through the existing
   packed-GEMM implementation or an equivalent graph-wide CPU launch plan.

2. The public MH-1 node-state policy is not propagated to the host generator.
   Host metadata and source rendering are called without
   `execution_mh1_node_state_policy`, and host therefore defaults to
   `recompute-v1`. Reverse repeats the large node program. A bounded
   full-retention host control should be implemented and measured before
   choosing the CPU default; unlike GPU, the extra retained state may be a
   favorable trade at 864 atoms.

3. Neighbor skin must be controlled in comparisons. The default 0.5 A skin
   expands factorized execution from 78,624 to 94,176 edges and adds 22% latency
   (4,071.2 versus 3,335.6 ms). Reports must include model cutoff, skin,
   effective cutoff, and directed-edge count.

4. Generic factorized CPU execution is memory-oriented but not performant. It
   uses only 7.07 MiB of edge workspace, but is 4.83x slower than
   `all_interactions`. Generated host-v4 halves that latency and process RSS,
   demonstrating that specialization helps, but its current node lowering
   leaves another 1.81x gap to the best Kokkos policy.

5. Current-schema extraction is required for generated execution. Artifacts
   created before the API rename contain `direct_contracts`; current code reads
   `execution_contracts`. Re-extraction fixes this. The extractor also
   currently imports optional `MACEField` unconditionally, so standard MACE
   installations without that extension cannot extract an ordinary checkpoint.

Hardware counter sampling was unavailable because the host enforces
`perf_event_paranoid=4`, including under elevated command execution. The
backend controls, scaling measurements, generated artifact metadata, and code
ownership inspection provide the retained attribution evidence.

## Recommended order

1. Fix retained-policy affine-MLP tape ownership and add regression tests.
2. Add a graph-wide packed-GEMM node-linear operation to the neutral MH-1 host
   launch plan; retain scalar per-node code for small blocks and nonlinear work.
3. Propagate and benchmark full-retention versus recompute node state on CPU.
4. Re-profile exact-cutoff generated factorized execution, then optimize edge
   owners only if node batching no longer dominates.
5. Improve OpenMP scaling beyond eight cores after the algorithmic node change.

## Generated-edge / packed-node experiment

The first implementation experiment keeps the generated host-v4 factorized
graph, conditioning, TPConv forward, source reverse, and edge reverse owners,
but routes the node program through the existing Kokkos operators. This gives
the large E3 linears access to graph-wide packed GEMM without changing edge
materialization. The evaluator exposes this control as
`set_execution_mh1_host_node_backend("kokkos")`; `"generated"` selects the
original generated scalar node owners and `"auto"` preserves the original
default.

The exact-cutoff 864-atom AlN comparison uses Float32, CPUs 0-15, 16 Kokkos
threads, one BLAS thread, three warmups, and seven repeats. Both rows use the
same cached `jit-mh1-v4-86823d33d34d12b5` artifact and report 20 generated
conditioning forward and 20 generated conditioning reverse launches across
the ten total evaluations.

| Factorized host node policy | E3 linear backend | Median (us/atom) | Step (ms) | Speedup | RSS after (MiB) | Node workspace (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Generated scalar owners | packed GEMM (inactive for generated nodes) | 4,344.454 | 3,753.609 | reference | 493.45 | 215.89 |
| Kokkos nodes | scalar | 2,383.315 | 2,059.184 | 1.82x | 970.86 | 667.10 |
| Kokkos nodes | packed GEMM | 773.360 | 668.183 | 5.62x | 1,019.75 | 667.10 |

Packed GEMM is 3.08x faster than scalar linears within the same hybrid node
path. The packed hybrid is 5.62x faster than the generated-node baseline and
reduces the remaining gap to the 350.824 us/atom stretch target from 11.0x to
2.20x. Its normalized time is 4.41x the 175.412 us/atom MH-0 factorized
control.

This control is not the production memory solution. It increases retained node
workspace by 451.21 MiB and process RSS by 526.30 MiB because the generic node
path retains its full forward/reverse state. The result isolates the important
optimization: retain the generated factorized edge architecture, split the
compact generated node phases around their large linear operations, and invoke
graph-wide packed GEMM for those operations. Intermediate nonlinear and
product state must remain phase-local or be recomputed so the implementation
does not inherit the generic node tape's memory scaling.

## Warmed operation-level CPU baseline

The official MH-0 checkpoint is the closest architectural control for MH-1.
The name "OMAT-0" in these measurements refers to the `omat_pbe` head selected
from that MH-0 checkpoint; it is not a separate local model or architecture.
MH-0 still differs from MH-1 in important ways: it has 128 channels and uses
the standard fixed-weight generated M0/R0/R1 modules, whereas MH-1 has 512
channels and pair-conditioned nonlinear node and product programs.

This profile uses the exact-cutoff 864-atom AlN graph (6.0 A, no skin, 78,624
directed edges), Float32, CPUs 0-15, 16 Kokkos/OpenMP threads, one BLAS thread,
and energy, force, and stress evaluation. Each path ran in a fresh process with
two warmups followed by five measured evaluations. The primary metric is the
median end-to-end time in us/atom.

| Path | Median (us/atom) | Step (ms) | Relative to MH-0 |
|---|---:|---:|---:|
| MH-0 generated factorized (`omat_pbe`) | 182.123 | 157.354 | 1.00x |
| MH-1 generated scalar nodes | 4,534.112 | 3,917.473 | 24.90x |
| MH-1 generated edges with packed Kokkos nodes | 766.600 | 662.343 | 4.21x |
| MH-1 2x-MH-0 target | 364.246 | 314.708 | 2.00x |

The operation trace was collected with a temporary Kokkos Tools timer. The
first two complete evaluation segments were discarded and the remaining five
were aggregated. Summed leaf timings reproduce the benchmark medians within
4.2% for MH-0 and within 0.3% for both MH-1 paths. The timer provides reliable
attribution but not hardware counters; Linux `perf` remains unavailable because
this host has `perf_event_paranoid=4`.

The original generated MH-1 path is dominated by the scalar node program:

| Generated MH-1 operation | Mean per evaluation (ms) | Share of step |
|---|---:|---:|
| Node post reverse | 1,900.735 | 48.5% |
| Node post forward | 1,781.886 | 45.5% |
| Graph source reverse | 92.784 | 2.4% |
| Graph edge reverse | 74.764 | 1.9% |
| Graph forward | 30.894 | 0.8% |
| Node pre forward | 16.210 | 0.4% |
| R1 edge conditioning forward and reverse | 16.909 | 0.4% |

The packed-node control removes that scalar-lowering deficiency. Its remaining
time is distributed as follows; shares use the 662.343 ms median step time, so
minor categories and timer variance prevent an exact 100% sum.

| Packed MH-1 operation group | Mean per evaluation (ms) | Share of step |
|---|---:|---:|
| Dense node linears, including pack/unpack | 354.308 | 53.5% |
| Generated graph forward/source/edge reverse | 186.352 | 28.1% |
| Compiled product forward/reverse | 73.542 | 11.1% |
| R1 edge conditioning forward/reverse | 16.468 | 2.5% |
| Gate forward/reverse | 12.001 | 1.8% |

For comparison, MH-0's largest leaf operations are R1 source reverse at
72.594 ms, R0 forward at 30.537 ms, R1 forward at 20.959 ms, and R0 coordinate
edge reverse at 14.876 ms. This confirms that MH-0 is useful for matching the
factorized schedule and semantic stages, but its kernel times cannot be treated
as like-for-like MH-1 costs because MH-1 performs substantially larger and
conditioned node work.

### Next optimization target

The first production experiment should replace only the large scalar node
linears in the generated MH-1 program with graph-wide packed GEMM phases. It
should preserve generated factorized edge execution and keep nonlinear/product
intermediates phase-local or recomputed. Dense node work is the largest
remaining group at 53.5%, while the current packed-node control proves its
performance value but retains about 667 MiB of node workspace and reaches about
1,023 MiB RSS. It is therefore an attribution control, not an acceptable
memory-capacity implementation.

After compact node GEMM is implemented, the next profile should decide between
compiled product forward/reverse (11.1%) and generated graph source/edge
reverse (23.6% excluding graph forward). Optimizing either first would be
premature while dense node lowering remains more than half of the step.

## Rejected `linear_2`-only compact split

An implementation experiment split the generated post-node owner around only
the large `linear_2` operation. It used the existing packed Kokkos GEMM and
explicit `ir_mul`/`mul_ir` conversion, while generated owners retained
normalization, gate, product, readout, and reverse semantics. The experiment
was numerically correct at 256 atoms: relative to the generated owner, total
energy differed by 1.5e-6 eV and force norm by 6.7e-7 eV/A.

The experiment was rejected and removed because its operation profile showed
that `linear_2` is not a useful phase boundary by itself. For one 256-atom
evaluation, the important totals were:

| Operation | Time (ms) |
|---|---:|
| Generated compact prepare/recompute/finish owners | 6,187.2 |
| Packed `linear_2` forward/reverse GEMM | 911.3 |
| All six layout-conversion kernels | 21.5 |

The layout conversion was therefore not the main regression. The split left
the larger scalar `linear_res` and `linear_1` work, normalization, gate
recomputation, product, and readout inside per-node generated owners. It also
required full-graph gated, interaction-output, and adjoint matrices, increasing
the 864-atom node workspace from the generated path instead of preserving the
capacity advantage.

Absolute timing qualification was unavailable during this experiment. A
same-session check of the unchanged installed packed-node control measured
8,877.866 us/atom, 11.6x slower than its recorded warmed baseline of 766.600
us/atom, despite an idle host and normal CPU frequencies. These anomalous
absolute values are not accepted as new baselines. The operation proportions
and the correctness comparison remain sufficient to reject the phase boundary.

The next implementation must expose a tiled whole post-node operation in the
neutral launch plan. Each tile should batch `linear_res`, `linear_1`, and
`linear_2` forward and reverse directly in generated `ir_mul` layout, with
normalization and gate between the linear phases. Generated product and readout
owners should consume the tile result, and reverse should recompute the same
tile rather than retain a graph-wide tape. Tile scratch must have a fixed row
budget independent of atom count. This removes the dominant scalar work while
keeping node memory bounded; extracting a single linear from the owner does
neither.

## Bounded whole-node tile milestone

The accepted follow-up implements the complete post-node split with a fixed
256-row budget. `linear_res`, `linear_1`, and `linear_2` use packed GEMM
directly in `ir_mul` layout. Generated owners retain normalization, gate,
product, skip, and readout semantics. Reverse recomputes each tile and does not
retain a graph-wide nonlinear tape. Pre-forward and pre-reverse also reuse the
same bounded arena.

The exact-cutoff 864-atom AlN comparison used the official MH-1 `omat_pbe`
head, Float32, 16 Kokkos OpenMP threads on CPUs 0-15, one OpenBLAS thread,
6.0 A cutoff, zero skin, and 78,624 directed edges. Five warmed evaluator-only
samples gave:

| Node policy | Median (us/atom) | Node workspace (MiB) | Relative time |
|---|---:|---:|---:|
| Bounded generated tiles, 256 rows | 1,461.948 | 313.78 | 1.87x |
| Packed Kokkos node control | 782.825 | 667.10 | 1.00x |

The tiled result used 1263.123 ms per step; its five samples ranged from
1173.486 to 1429.193 ms. The packed control used 676.361 ms per step. The
bounded path therefore substantially reduces the control's node workspace
while closing most of the scalar-owner gap. Its total energy was
`-6423.652703967867 eV` and force norm was `3.1685293622548127 eV/A`; the
control differed by `6.10e-5 eV` and `5.35e-7 eV/A`, respectively.

At 256 atoms, the fixed tile is one active 256-row tile. It measured
1,418.326 us/atom, compared with
1,906.833 us/atom for the earlier 64-row experiment in the same anomalous host
timing environment. These absolute values remain separate from the accepted
historical baseline, but the same-session 864-atom ratio is actionable.

The workspace values include all bounded tile tensors. A tested 512-row
alternative reached 1,245.219 us/atom but required 508.28 MiB of node
workspace, so it was rejected as an unfavorable capacity tradeoff.

The remaining node optimization is packing efficiency, not a wider retained
tape. Each reverse tile currently recomputes and repacks all three forward
linears, then packs their adjoints again. The next profile should separate GEMM
time from `ir_mul` pack/unpack time and test bounded packed-state reuse within
one tile. Increasing the fixed row budget further would trade away the package's
capacity advantage and is not the preferred next direction.

## Retained packed-node performance path

The graph-wide packed Kokkos-node path remains a supported performance policy.
It is selected with `set_execution_mh1_host_node_backend("kokkos")` while the
generated host artifact continues to execute conditioning and graph phases.
This path intentionally spends more memory to avoid bounded-tile recomputation
and to present all 864 nodes to each packed GEMM.

A current-tree Kokkos Tools trace used the official MH-1 `omat_pbe` head,
Float32, 864-atom periodic AlN, exact 6.0 A cutoff, zero skin, 78,624 directed
edges, CPUs 0-15, 16 OpenMP threads, and one OpenBLAS thread. After two warmups,
the five measured evaluator samples had a 682.548 ms median, or
789.986 us/atom. Mean leaf timers across those samples sum to 690.043 ms, 1.1%
above the median because mean leaf values and the median wall time use different
aggregations.

```text
compute_node_energies_forces                         682.548 ms median
|-- forward model
|   |-- dense E3 linears                             188.313 ms
|   |   |-- GEMM                                     137.307 ms
|   |   `-- pack, unpack, mask, initialize            51.006 ms
|   |-- generated graph forward                       30.132 ms
|   |-- product forward                               23.094 ms
|   |-- gate and normalization forward                15.412 ms
|   `-- conditioning forward                           5.314 ms
|-- analytic reverse
|   |-- dense E3 linears                             199.987 ms
|   |   |-- GEMM                                     140.898 ms
|   |   `-- pack, unpack, initialize                  59.089 ms
|   |-- generated graph source reverse                84.771 ms
|   |-- generated graph edge reverse                  71.631 ms
|   |-- product reverse                               47.857 ms
|   |-- conditioning reverse                          11.304 ms
|   `-- gate and normalization reverse                 4.601 ms
`-- layout conversion and remaining evaluator work     7.629 ms
```

Grouped by optimization domain, dense linears account for 388.300 ms, generated
graph execution for 186.534 ms, products for 70.951 ms, gate/normalization for
20.013 ms, conditioning for 16.618 ms, explicit layout conversion for 1.267 ms,
and the remaining radial, force, ZBL, accumulation, and bookkeeping kernels for
6.362 ms. The graph source and edge reverse pair is the largest non-linear
target at 156.402 ms. Product reverse is the next isolated target at 47.857 ms.

The semantic linear labels show that the interaction transforms dominate this
group. Across forward and reverse, layer 0 `linear_2` is about 110 ms, layer 1
`linear_1` about 87 ms, layer 0 `linear_1` about 63 ms, and layer 1 `linear_2`
about 87 ms. Together, the four `linear_1` and `linear_2` transforms account for
approximately 331 ms, or 85% of the dense-linear budget. The two `linear_2`
modules each perform four 512-by-512 transforms. The `linear_1` modules include
128-to-2048 scalar transforms and 128-to-512 higher-angular transforms.

The earlier OpenBLAS comparison mixed threading configurations and is not a
valid reason to reject the Kokkos Kernels BLAS TPL. A corrected run launches
under `taskset -c 0` and sets both Kokkos/OpenMP and OpenBLAS to one thread, so
every library thread is confined before initialization. With the persistent
packed pipeline, OpenBLAS 0.3.32 measures 4,481.423 us/atom
(4,470.230-4,500.166 across five samples), compared with 8,797.307 us/atom for
the matched generic Kokkos GEMM build. Host BLAS dispatch therefore reduces the
one-thread evaluator time by 49.1%, or 1.96x. The extension links directly to
`libopenblas.so.0`, and `openblas_get_num_threads()` reports one at runtime.

The OpenBLAS result gives `-6423.652417354913 eV` and force L2
`3.1685335726159765 eV/A`. Relative to the generic packed result below, the
total-energy difference is `3.25e-4 eV` (`3.77e-7 eV/atom`) and the force-norm
difference is `5.70e-6 eV/A`. Native CPU builds now default
`KokkosKernels_ENABLE_TPL_BLAS=ON` when the user has not set it explicitly;
CUDA and HIP selection remains unchanged. The original rank-3 standalone pack
loops remain in place for transforms outside the persistent pipeline.

### Persistent packed interaction linears

The retained Kokkos-node path now has an opt-in persistent packed pipeline for
`linear_1 -> normalized gate -> linear_2` and its reverse. Each irrep block is
stored as `[multiplicity, node * component]` in the existing graph-wide tensor
capacity. `linear_1` writes this representation directly, the normalized gate
consumes and produces packed blocks, and `linear_2` consumes them directly. The
reverse follows the same layout through `linear_2`, the gate, and `linear_1`.
This removes the `linear_1` output unpack, `linear_2` input pack,
`linear_2.reverse` output unpack, and `linear_1.reverse` input pack. It does not
add a graph-wide tensor: packed and node-major blocks have the same element
count, and reported node workspace remains 723,828,120 bytes.

The exact 864-atom qualification gives `-6423.6527421460405 eV` and force L2
`3.1685278774086214 eV/A`. The node-major control gives
`-6423.652765016057 eV` and `3.1685288274779535 eV/A`, differences of
`2.29e-5 eV` total and `9.50e-7 eV/A`, respectively. Direct packed ir-mul
forward and reverse validation both report zero error.

The apparent 8,900 us/atom slowdown during development was a benchmark
configuration error, not a host-frequency or code regression. The driver reads
`SYMMETRIX_BENCHMARK_THREADS` before importing the extension and rewrites both
`OMP_NUM_THREADS` and `KOKKOS_NUM_THREADS` from it. Setting only
`OMP_NUM_THREADS=16` was therefore overwritten by the driver's default of one.
An explicit single-thread rerun on CPU 0 measures 8,780.449 us/atom, confirming
that the earlier results were single-thread executions.

Corrected measurements set `SYMMETRIX_BENCHMARK_THREADS=16`, use one BLAS
thread, and apply affinity inside the driver. On CPUs 0-15, which are 16
distinct physical cores, the persistent packed path measures 735.208 us/atom
(730.589-741.964 us/atom across five samples). The matched node-major control
measures 765.474 us/atom (761.398-769.540), so the packed pipeline improves the
valid 16-core result by 3.95%. A placement using CPUs 0-7 and their SMT siblings
16-23 measures 1,394.413 us/atom, 89.7% slower than 16 physical cores. CPU
placement therefore matters substantially, but it was not the cause of the
original tenfold discrepancy; the effective one-thread Kokkos configuration
was.

Matched Kokkos Tools profiles from the diagnostic single-thread runs show that
the pipeline removes approximately 260 ms of explicit linear pack/unpack
traffic across two evaluations. A first team-reduction gate implementation
erased most of that saving; the retained version evaluates each scalar or gated
feature once per node and uses a separate node-wise density pass.

### Single-thread MH-0 comparison

A matched one-core comparison uses CPU 0, one Kokkos/OpenMP thread, one BLAS
thread, Float32, the official MH-0 and MH-1 `omat_pbe` heads, 864-atom periodic
AlN, exact 6.0 A cutoff, zero skin, and 78,624 directed edges. Both paths use
factorized execution. MH-1 selects the packed Kokkos-node policy; MH-0 uses its
standard generated M0/R0/R1 modules. Two warmups precede five measured calls.

| Model | Median (us/atom) | Sample range (us/atom) | Relative |
|---|---:|---:|---:|
| MH-0 factorized | 2,104.391 | 2,101.574-2,106.247 | 1.00x |
| MH-1 packed Kokkos nodes | 8,797.307 | 8,783.436-8,816.495 | 4.18x |

The paired MH-1 benchmark's reference option was not used for the final MH-0
number because it does not forward `neighbor_skin` to the reference calculator.
It silently produced 94,176 edges rather than the required 78,624. The dedicated
standard-MACE harness applies and records the exact cutoff correctly.

Matched Kokkos Tools profiles give 2,113.899 us/atom for MH-0 and
8,744.901 us/atom for MH-1, a 4.14x ratio. Their grouped leaf timers reproduce
the respective median steps to within 0.04%:

| MH-0 stage | Mean (us/atom) | Share |
|---|---:|---:|
| Generated R1 forward/source/edge | 1,083.965 | 51.3% |
| Standard R0 forward/reverse | 720.730 | 34.1% |
| Polynomial and node operations | 222.521 | 10.5% |
| Standard M0 forward/reverse | 71.708 | 3.4% |
| Geometry, ZBL, readout, remaining | 15.740 | 0.7% |

| MH-1 stage | Mean (us/atom) | Share |
|---|---:|---:|
| Dense node linears and layout work | 4,733.065 | 54.1% |
| Generated graph forward/source/edge | 2,468.083 | 28.2% |
| Compiled product forward/reverse | 1,122.241 | 12.8% |
| Pair conditioning forward/reverse | 269.118 | 3.1% |
| Gate and normalization | 112.705 | 1.3% |
| Geometry, ZBL, readout, remaining | 35.810 | 0.4% |

The four main interaction linears dominate MH-1's dense group. Mean time per
evaluation is 1,074.077 ms for layer 0 `linear_2`, 1,073.985 ms for layer 1
`linear_2`, 821.908 ms for layer 1 `linear_1`, and 349.910 ms for layer 0
`linear_1`. Together they consume 3,319.880 ms, or 81.2% of the dense-linear
group and 43.9% of the complete MH-1 step. The dense-linear group alone costs
2.24x the entire MH-0 evaluation, confirming that further CPU progress requires
better large dense transforms rather than another edge-streaming policy.

### Matched OpenBLAS thread scaling

An optimized BLAS backend is essential for MH-1 because its wider node states
turn the interaction linears into large SGEMMs. MH-0 instead performs most of
its work in generated R1 and standard R0 modules, so BLAS selection has much
less influence on its runtime. The generic Kokkos GEMM implementation is useful
as a portable fallback but is not an acceptable performance baseline for MH-1.

The matched scaling runs use OpenBLAS 0.3.32, Float32, the official MH-1 and
MH-0 `omat_pbe` heads, 864-atom periodic AlN, exact 6.0 A cutoff, zero skin, and
78,624 directed edges. Each fresh process is constrained with `taskset` before
Python or OpenBLAS initializes. Kokkos/OpenMP and OpenBLAS both use the stated
number of distinct physical cores, with `OMP_WAIT_POLICY=PASSIVE`. Two warmups
precede five measured energy-and-force evaluations.

| Physical cores | MH-0 factorized (us/atom) | MH-1 packed (us/atom) | MH-1 / MH-0 |
|---:|---:|---:|---:|
| 4 | 560.899 | 1,428.974 | 2.55x |
| 8 | 326.883 | 921.800 | 2.82x |
| 16 | 183.471 | 683.085 | 3.72x |

The MH-1 sample ranges are 1,420.059-1,431.629, 913.237-926.563, and
672.421-698.364 us/atom at 4, 8, and 16 cores. The corresponding MH-0 ranges
are 554.483-562.291, 315.006-332.223, and 182.234-210.654 us/atom. From 4 to
16 cores, MH-1 improves 2.09x while MH-0 improves 3.06x. Optimized BLAS closes
the dominant single-core linear-kernel gap, but the widening model ratio shows
that the complete MH-1 path scales less efficiently at this atom count. A
matched multithread stage profile is required to distinguish BLAS saturation
and packed-layout overhead from generated graph and product scaling.

### Primary one-thread optimization target

CPU optimization and analysis now use one physical core, one Kokkos/OpenMP
thread, and one BLAS thread as the primary contract. Process affinity must be
applied before Python initializes either runtime. The 4/8/16-core results above
remain secondary scaling qualifications; they do not replace the one-thread
before/after gate.

A fresh OpenBLAS trace under `taskset -c 0` uses the same official MH-1
`omat_pbe` head, Float32, 864-atom periodic AlN, exact 6.0 A cutoff, zero skin,
and 78,624 directed edges. One warmup precedes one profiled evaluation. The
evaluator measures 4,455.930 us/atom. Leaf timers cover 3,683.778 ms of the
3,849.923 ms evaluation; host BLAS calls account for part of the uninstrumented
remainder.

| Post-BLAS stage | Approx. us/atom | Approx. share of wall time |
|---|---:|---:|
| Generated graph forward and reverse | 2,459.416 | 55.2% |
| Compiled products forward and reverse | 1,138.393 | 25.5% |
| Pair conditioning forward and reverse | 272.541 | 6.1% |
| Visible node packing, gate, and layout kernels | 256.608 | 5.8% |
| Remaining and uninstrumented work, including host BLAS | 328.972 | 7.4% |

The next optimization target is generated graph reverse. Edge reverse alone
costs about 949.463 ms per evaluation, or 1,098.916 us/atom, and source reverse
costs about 758.061 ms, or 877.385 us/atom. Together they consume 44.4% of the
complete post-BLAS step. Optimize their shared generated schedule and memory
traffic before revisiting dense node linears. Compiled product reverse is the
next target after graph reverse at about 676.149 ms, or 782.580 us/atom.

### Generated graph reverse experiment plan

All experiments use the primary one-thread contract above. The comparison
baseline is the five-sample OpenBLAS median of 4,481.423 us/atom; MH-0 is
2,104.391 us/atom, so the 2x target is 4,208.782 us/atom. The required MH-1
reduction is 272.641 us/atom, or 6.1% of the current step. Every experiment
must also report its isolated edge/source timer and fixed scratch bytes.

1. Add benchmark-only attribution inside generated edge reverse. Separate the
   64-wide forward weight projection, sparse tensor reverse, 64-wide phi
   adjoint projection, and harmonic/cutoff adjoints. Use the exact official
   model dimensions rather than a synthetic GEMM alone. Do not retain this
   instrumentation in production callbacks if it measurably perturbs timing.
2. Test a stack-local 64-float phi-adjoint accumulator. The current generated
   callback updates the output row after every path and channel. Accumulate one
   edge locally and store it once, while keeping the existing generated sparse
   arithmetic. This adds 256 bytes per active edge callback and no graph-sized
   storage. Retain it only if edge reverse improves by at least 5% without a
   whole-step regression.
3. Test a bounded projected-edge tile. For a tile of `B` edges, use SGEMM for
   `weights = phi * linear_weight^T`, run a generated sparse reverse callback
   that produces `weight_adjoint`, harmonic adjoints, and cutoff adjoints, then
   use SGEMM for `phi_adjoint = weight_adjoint * linear_weight`. Sweep
   `B = 8, 16, 32, 64, 128`. The layer weight dimensions are 512 and 1280.
   Reusing one tile between layers limits the two weight buffers to 32/80 KiB
   at `B=8`, 128/320 KiB at `B=32`, and 512/1280 KiB at `B=128`; phi and
   harmonic buffers keep the largest total below 1.5 MiB. No `edges * weight`
   tensor may be retained.
4. Promote the tiled path only when the median of five calls after two warmups
   improves the full evaluator by at least 3%, edge reverse by at least 15%,
   and peak workspace remains bounded independently of atom and edge count.
   The final graph-reverse milestone should reach at most 4,208.782 us/atom or
   explain, with a new profile, which remaining stage prevents the 2x target.
5. Validate every retained variant against the current OpenBLAS control on the
   official 864-atom graph and focused generated-callback tests. Require the
   same energy within existing Float32 accumulation tolerance, maximum force
   error no greater than 2e-5 eV/A, finite-difference consistency, and identical
   graph hashes and directed-edge counts.
6. Keep the operation model backend-neutral. The shared renderer owns the
   sparse reverse equations and projected-tile contract; the CPU host adapter
   may use KokkosKernels/OpenBLAS while CUDA and HIP retain their existing
   device implementation until separately qualified. The portable scalar host
   callback remains the fallback when no optimized BLAS TPL is available.
7. If edge reverse does not supply the required whole-step reduction, apply the
   same attribution to source reverse. Preserve source ownership and avoid
   atomics; test bounded source-CSR tiles rather than retaining graph-wide
   projected weights. If graph reverse then meets diminishing returns, move to
   compiled product reverse, currently 782.580 us/atom, before revisiting node
   GEMMs or gate/layout kernels.

#### Stack-local phi-adjoint result

The stack-local 64-float phi-adjoint accumulator passes its promotion gate. It
uses 256 bytes per active host callback and no graph-sized workspace. The exact
one-thread benchmark retains the control energy of -6423.652417354913 eV and
force L2 norm of 3.1685335726159765 eV/A.

| Metric | OpenBLAS control | Local accumulator | Change |
|---|---:|---:|---:|
| Full evaluator, five-sample median | 4,481.423 us/atom | 4,302.066 us/atom | -4.0% |
| Generated edge reverse, measured profile | 1,098.588 us/atom | 977.673 us/atom | -11.0% |
| Generated source reverse, measured profile | 876.105 us/atom | 854.152 us/atom | -2.5% |

The profiled full evaluation is 4,308.754 us/atom versus 4,455.930 us/atom for
the control trace. The result exceeds the required 5% isolated edge-reverse
improvement with no whole-step regression, so it remains enabled for generated
host execution. CUDA and HIP source generation are unchanged.

#### Bounded projected-edge tile result

The projected-edge BLAS tile was implemented and swept at the planned bounded
sizes, then removed because it did not pass the promotion gate. Each trial used
two reusable `B * weight_size` float buffers; the largest active layer has 1280
weights, so `B=64` required 640 KiB total and remained independent of graph
size. The official contract covers all 512 and 1280 weight coordinates exactly
once, allowing the generated sparse callback to overwrite the tile adjoint
without a clearing pass.

| Tile size | One-sample full evaluator |
|---:|---:|
| 8 | 4,437.826 us/atom |
| 16 | 4,384.743 us/atom |
| 32 | 4,344.318 us/atom |
| 64 | 4,273.187 us/atom |
| 128 | 4,277.492 us/atom |

The best candidate, `B=64`, was rerun with two warmups and five measured calls.
Its median was 4,311.975 us/atom (range 4,278.366--4,337.999 us/atom), which is
0.23% slower than the retained local-accumulator median of 4,302.066 us/atom
and does not meet the required 3% whole-step improvement. Energy remained
identical; the force L2 difference was below 1e-5 eV/A. The small-tile GEMM and
per-tile launch overhead consumes the saved scalar projection time at this
workload, so no tile ABI, environment control, or tile workspace remains in
production code.

#### Packed source-adjoint row result

Generated host source reverse previously placed the channel loop outside the
source CSR traversal. For the official model this rescanned every source's edge
segment 128 times. The accepted schedule traverses each source segment once,
processes channels inside each edge, accumulates the complete source-adjoint
row locally, and writes that row once. It uses 128 floats (512 bytes) for layer
0 and 512 floats (2 KiB) for layer 1 per active callback, with no graph-sized
workspace. Each source-adjoint element retains its original edge and path
accumulation order. The CUDA channel-owner renderer is unchanged.

| Metric | Local phi-adjoint control | Packed source row | Change |
|---|---:|---:|---:|
| Full evaluator, five-sample median | 4,302.066 us/atom | 3,990.630 us/atom | -7.2% |
| Generated source reverse, measured profile | 854.243 us/atom | 532.143 us/atom | -37.7% |
| Generated edge reverse, measured profile | 978.290 us/atom | 974.786 us/atom | -0.4% |

The packed-source profile measures the complete evaluator at 3,965.540
us/atom. Energy remains -6423.652417354913 eV; force L2 is
3.168532201739603 eV/A, 1.37e-6 eV/A below the local-phi control. The
five-sample median is 5.2% faster than the 4,208.782 us/atom target derived
from twice the matched MH-0 result, so the primary MH-1 CPU optimization goal
is met on the official 864-atom, 78,624-directed-edge workload.

#### Packed harmonic-adjoint row result

The next retained edge-reverse change extends the bounded local accumulation
to the 16-value harmonic-adjoint row. Previously each path and channel updated
the graph row directly. The host callback now accumulates the row locally and
stores it once after processing the edge. This adds 16 floats (64 bytes),
bringing the largest active edge callback scratch to 320 bytes, with no
graph-sized allocation. CUDA and HIP source generation are unchanged.

The exact-cutoff one-thread benchmark uses the official MH-1 `omat_pbe` head,
Float32, 864-atom periodic AlN, zero skin, and 78,624 directed edges. Its
final five-sample median is 3,754.622 us/atom (3,750.577--3,760.112 us/atom),
down from the 3,990.630 us/atom packed-source milestone by 5.9%. Energy remains
`-6423.652417354913 eV`; force L2 is `3.1685323477597684 eV/A`. A matched
trace reduces generated edge reverse from about 421.108 ms to 315.427 ms per
call, a 25.1% isolated improvement.

Two related owner-reordering experiments were rejected and removed:

- Host graph-forward channel tiling at sizes 16, 32, 64, and 128 changed the
  isolated stage by only 0.3% at the best candidate and changed Float32
  accumulation order. The five-sample tile-16 median was 3,745.617 us/atom,
  but the small whole-step variation was not attributable to graph forward.
- Host compiled-product forward coarsening from one owner per
  `(sample, component, feature)` to `(sample, feature)` measured
  3,778.609 us/atom, slightly slower than the harmonic-local control. It also
  raised the contraction timer to about 155.971 ms per call.

These results narrow the useful pattern: packing is effective when it removes
repeated writes or repeated traversal around a small bounded output row. Merely
coarsening an owner without removing substantial work is not beneficial.

### Standard R0 host-forward channel tiling

The same bounded accumulation pattern has a high-payoff application in the
standard MH-0 R0 forward path. The previous host owner was one
`(receiver, harmonic, channel)` output. It therefore traversed the same
receiver edge list and recomputed radius interval data for every one of the
16 harmonics and 128 channels. The retained implementation owns one receiver
and an eight-channel tile, traverses the receiver edges once per tile, computes
each cubic spline value once per `(edge, l, channel)`, and reuses it across the
`2l+1` components. The accumulator is bounded at 16 harmonics by eight channels,
or 128 floats (512 bytes), independent of atom and edge count. The CUDA/HIP
team path is unchanged.

The exact one-thread MH-0 `omat_pbe` qualification uses Float32, 864-atom
periodic AlN, the 6.0 A model cutoff, zero skin, and 78,624 directed edges.

| R0 host forward | Median (us/atom) | Sample range (us/atom) | R0 forward trace |
|---|---:|---:|---:|
| Output-owned control | 2,104.391 | 2,101.574--2,106.247 | about 413 ms/call |
| Eight-channel receiver tile | 1,626.268 | 1,624.293--1,629.237 | 17.132 ms/call |

The tile reduces complete MH-0 evaluation time by 22.7% and isolated R0
forward by about 24x. One-sample tile screens measured 1,660.957, 1,627.815,
and 1,640.974 us/atom for tile sizes 4, 8, and 16, respectively, so eight is
retained. Against forced R0-v1 on the same graph, the tiled module differs by
3.52e-7 eV/atom, 1.37e-5 eV/A maximum force, and 1.26e-7 eV/A^3 maximum
stress, within the existing Float32 qualification tolerance.

This improved MH-0 control changes the current model ratio. The retained MH-1
harmonic-local result is 2.31x the new 1,626.268 us/atom MH-0 result, rather
than below twice MH-0. The remaining MH-1 optimization target is therefore
3,252.536 us/atom under this updated matched baseline. The next profile should
focus on MH-1 compiled product reverse and the remaining graph reverse work;
Standard R0 itself is not on the MH-1 execution path.

### Compiled product reverse degree specialization

The compiled product plan canonicalizes each output component's monomials by
degree. The host reverse path now records the degree-one and degree-two term
boundaries when it builds the descriptor, then executes separate degree-one,
degree-two, and degree-three loops. This removes the per-term degree branch and
avoids loading unused indices and native-layout fields. The CUDA/HIP owner and
launch shape are unchanged. The owner remains one `(sample, feature)` pair with
one 16-value local angular adjoint, so workspace and accumulation order are
unchanged.

The exact one-thread MH-1 `omat_pbe` qualification uses Float32, 864-atom
periodic AlN, the 6.0 A model cutoff, zero skin, and 78,624 directed edges.

| Product reverse | Median (us/atom) | Sample range (us/atom) |
|---|---:|---:|
| Runtime degree branch | 3,754.622 | 3,750.577--3,760.112 |
| Degree-specialized host loops | 3,680.508 | 3,672.787--3,685.555 |

The specialization improves the complete evaluation by 1.97%. In matched
single-evaluation profiles, compiled product reverse falls from 746.4 to
676.4 us/atom. Layer 0 remains dominant at 573.1 us/atom; layer 1 measures
103.3 us/atom. Energy and force L2 are unchanged at the precision printed by
the benchmark. Direct serial/Kokkos compiled-product tests pass for available
MH-1 and generalized product fixtures.

Three larger variants were rejected and removed:

- Two- and four-feature owners increased local adjoint state to 128 and 256
  bytes in Float32 and regressed the complete evaluation to about 5,470 and
  5,212 us/atom, respectively.
- Calling the generated, fully unrolled product reverse from the packed-node
  path was numerically invalid: its persistent `ir_mul` input contract does
  not match the packed path's native interaction-output layout. The force L2
  changed from about 3.169 to 1.195 eV/A, so its timing is not comparable.
- Splitting layer-1 source reverse into two 64-channel passes halved local
  scratch from 2 KiB to 1 KiB but left the isolated stage unchanged at about
  422 us/atom while traversing the source edge list twice.

Against the current MH-0 result of 1,626.268 us/atom, the updated 2x target is
3,252.536 us/atom. The remaining gap is 427.972 us/atom, or 11.6% of the MH-1
step. The next useful work should target layer-1 graph reverse without adding
another source-edge traversal; product unrolling and feature coarsening do not
offer enough remaining leverage. GCC's vectorization report confirms that the
64-value edge-reverse phi projections already use 64-byte AVX-512 vectors, and
the ten layer-1 paths use ten distinct learned weight rows. Alias annotations
or cross-path projection reuse therefore do not expose another immediate win.

### Source-owned graph reverse fusion

The host ABI-v4 reverse owner now traverses each source's scheduled edge list
once and computes both source and per-edge derivatives in the same generated
path loop. Each path weight is evaluated once per `(edge, channel)`, and each
sparse Wigner term shares its harmonic and target-adjoint loads between the
source, harmonic, and radial-prefix derivatives. The owner retains the bounded
source adjoint and uses only per-edge phi, harmonic, and cutoff accumulators;
its storage is independent of graph size. CUDA/HIP and ABI-v3 execution remain
unchanged. A new optional ABI-v4 capability advertises that source owners also
write edge reverse outputs, allowing older v4 artifacts to keep the separate
edge launch.

The exact one-thread qualification uses MH-1 `omat_pbe`, Float32, periodic
864-atom AlN, the 6.0 A model cutoff, zero skin, and 78,624 directed edges.

| Graph reverse | Median (us/atom) | Sample range (us/atom) |
|---|---:|---:|
| Separate source and edge owners | 3,641.129 | 3,632.875--3,653.304 |
| Source-owned fused arithmetic | 3,266.616 | 3,259.086--3,273.899 |

The fusion improves the complete evaluation by 10.29%. The two graph-reverse
stages total about 983 us/atom in the control profile and about 904 us/atom in
the fused profile. A scheduling-only source-owned variant, which called the
unchanged source and edge functions from one owner, regressed to about
3,668 us/atom and was removed. The retained gain therefore comes from sharing
the generated arithmetic and loads, not just removing one launch.

On a periodic 32-atom AlN comparison against debug generic factorized
execution, the maximum absolute differences are 2.40e-5 eV in total energy,
2.12e-6 eV in per-atom energies, 1.62e-6 eV/A in forces, and 2.28e-7 in stress.
All pass the existing Float32 `rtol=2e-5`, `atol=5e-4` acceptance threshold.
The qualified result is 14.080 us/atom, or 0.43%, above the current 2x MH-0
target of 3,252.536 us/atom.
