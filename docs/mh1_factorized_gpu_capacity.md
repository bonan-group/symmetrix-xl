# MH-1 factorized GPU capacity audit

Status: source audit plus measured HIP and CUDA qualification, 2026-09-01

## Scope

This note identifies the device-resident quantities in the generated
factorized MH-1 evaluator that grow with the atom count `N` or directed-edge
count `E`. The reference graph is 864-atom wurtzite AlN at the model's exact
`6.0 A` cutoff:

```text
N = 864 atoms
E = 78,624 directed edges
E / N = 91 directed edges/atom
precision = Float32 model/workspace, Float64 geometry and result buffers
```

The useful capacity model is

```text
known device bytes = constant model/runtime bytes
                   + atom coefficient * N
                   + edge coefficient * E
                   + topology-dependent schedule bytes
                   + bounded/tiled scratch.
```

`N` and `E` must remain independent. At fixed density and cutoff, `E` is
normally proportional to `N`, but a larger cutoff, skin, or denser structure
increases `E/N` and therefore every edge-scaled term without changing the
atom-scaled term.

This is an allocation-level audit. It does not include device code, learned
parameters, Kokkos/backend runtime state, allocator bookkeeping, or
fragmentation. Those are constant or allocation-history dependent for a fixed
model and must be added from a process-level VRAM measurement when predicting
the absolute largest runnable system.

## HIP qualification

The current tree was qualified on the Radeon 8060S (`gfx1151`, ROCm 7.14)
using the shared v4 RTC module path. Both policies report
`generated_hip_v4`, use hipRTC explicitly, and complete with zero factorized
fallback evaluations. The uninstrumented timings use five warmups and ten
samples. The cutoff is exactly `6.0 A`, the neighbor-list skin is zero, and the
graph has 78,624 directed edges for 864 atoms.

A separate empty-cache qualification reported `jit_status=built` and
`hipRTC-9.0 --std=c++20 -O3 --gpu-architecture=gfx1151`; it produced and loaded
the `.hsaco` through the same v4 path without a hipcc fallback.

| Node-state policy | Median | Range | Node workspace | rocprof Agent-1 peak |
|---|---:|---:|---:|---:|
| `full-retention-v1` | `173.872 us/atom` | `173.334-174.583 us/atom` | `251,340,952 B` (`239.70 MiB`) | `359,476,504 B` (`342.82 MiB`) |
| `recompute-v1` | `192.077 us/atom` | `191.791-192.587 us/atom` | `127,477,912 B` (`121.57 MiB`) | `235,612,440 B` (`224.70 MiB`) |

`recompute-v1` removes `123,863,040 B` from the reported node workspace,
which is `49.28%` of that subtotal, for a `10.47%` timing cost. The independently
reconstructed rocprof allocation trace shows a `123,864,064 B` reduction in
whole-process Agent-1 peak live allocations, or `34.46%`. The 1,024-byte
difference between the two deltas is outside the node-workspace diagnostic.
The default remains `full-retention-v1`; users must explicitly request
`execution_mh1_node_state_policy="recompute-v1"` to select this low-memory
tradeoff.

AMD SMI does not expose compute-process memory for this integrated GPU, so the
whole-process values above are reconstructed from rocprof memory-allocation
traces. They include the profiled process's device allocations, but not opaque
driver allocations that rocprof does not report.

The profiler adds substantial overhead; its one-step totals are for hotspot
attribution, not acceptance timing:

| Kernel category | Full retention | Recompute |
|---|---:|---:|
| R1 reverse: edge (compact fused) | `46.805 ms` (29.21%) | `46.568 ms` (27.42%) |
| R1 reverse: source | `22.968 ms` (14.33%) | `23.250 ms` (13.69%) |
| R1 forward | `20.514 ms` (12.80%) | `20.657 ms` (12.16%) |
| M1/node forward | `20.768 ms` (12.96%) | `17.696 ms` (10.42%) |
| M1/node reverse | `20.346 ms` (12.70%) | `33.848 ms` (19.93%) |
| R1 edge conditioning reverse | `11.603 ms` (7.24%) | `11.655 ms` (6.86%) |
| All device kernels | `160.261 ms` | `169.846 ms` |

The recompute policy's extra work is localized to node reverse: it issues 312
node-reverse launches versus 144 and adds about `13.50 ms` under tracing. The
edge and graph stages remain effectively unchanged.

## CUDA reuse-adjoint qualification

CUDA now provides `reuse-adjoints-v1`, which follows the MH-0 lifetime policy:
it retains the pre-gate and interaction-output tensors required by reverse but
reuses each dead forward-message row for its message adjoint. The grouped
reverse kernel accumulates `message * message_adjoint` into the density
adjoint before overwriting that message row. Density bias accumulation is
fused into the existing pre-gate scaling kernel, leaving only a scalar density
finalize kernel. No forward node state is reconstructed during reverse.

The renderer and native storage policy are shared by CUDA and HIP; host
execution rejects this device-only policy explicitly. HIP selects the same
retained reverse schedule, although its launch geometry still requires
separate performance qualification.

The matched RTX 5090 throughput comparison uses Float32, 864 AlN atoms, 78,624
directed edges, the exact `6.0 A` model cutoff, zero skin, the 1,024-row
`throughput-v1` arena, `pair_spline_v1_staged_device_v5`, required NVRTC, and
zero fallback. Each result below is a fresh process with 10 warmups and 20
synchronized samples:

| Node-state policy | Process 1 | Process 2 | Two-process mean |
|---|---:|---:|---:|
| `full-retention-v1` | `6.667440 us/atom` | `6.652116 us/atom` | `6.659778 us/atom` |
| `reuse-adjoints-v1` | `6.654916 us/atom` | `6.684021 us/atom` | `6.669469 us/atom` |

The reuse policy is within `0.009691 us/atom`, or `0.15%`, of full retention.
Matched Nsight Systems traces report 59 launches for both policies and a
device-total difference of only `0.002963 us/atom`. Node kernels differ by
`0.000741 us/atom`; the extra `0.024149 us/atom` in layer-1 message reverse is
offset by removing wide density passes from both layers.

Reuse reduces the 864-atom CUDA node workspace from `337,979,544 B` to
`320,284,824 B`, an exact `17,694,720 B` or `20,480 B/atom` saving. Sampled
process GPU memory falls from `1,022 MiB` to `1,004 MiB`. Against full
retention, energy and all 864 node energies are bitwise identical; maximum
force difference is `6.9100e-8 eV/A`, maximum stress difference is
`2.2502e-11 eV/A^3`, and all outputs are finite.

The retained-state fusion advances the shared JIT compatibility epoch to 6 so
older modules cannot cross-load. A fresh CUDA build reports matching native
and Python epoch 6 and extension SHA-256
`f00af53319fd698b5c4f9d0d9bdf0ee5f99d4897d2158dbfea8e970e33db80d9`.

The measured 32 GB capacity boundary uses the 256-row `capacity-v1` arena with
the same model, cutoff, zero skin, spline executor, required RTC, and policy:

| Repeat | Atoms | Directed edges | Two-trial result | Sampled device peak |
|---:|---:|---:|---|---:|
| 31 | 119,164 | 10,843,924 | success twice | `30,522-30,534 MiB` |
| 32 | 131,072 | 11,927,552 | failed first evaluation twice: additional `728 MiB` allocation | `31,848-31,860 MiB` |

At repeat 31, process GPU memory was `30,204 MiB` and node workspace was
`23,553,065,560 B` in both trials. The single post-setup samples ranged from
`285.792` to `414.284 us/atom`; these highly variable near-capacity calls are
not steady-state throughput measurements. Repeat 32 reached `31,582 MiB`
process GPU memory before the failed allocation in both trials. Under this
exact workload, repeat 31 is therefore the qualified maximum on the 32,607
MiB RTX 5090. The earlier repeat-41 result used the obsolete recompute-derived
reuse semantics and is not a capacity result for the current policy.

## Ranked growth terms

The following table uses current source formulas and the retained generated
GPU schedule. MiB values use `2^20` bytes. Entries are disjoint except where a
row explicitly describes a subtotal.

| Device quantity | Growth | Coefficient | 864-AlN cost | Capacity action |
|---|---:|---:|---:|---|
| Generated node state | `N` plus constant runtime packs and tile scratch | recompute `76,304 B/atom`; reuse adjoints `199,184 B/atom`; full retention `219,664 B/atom` | CUDA throughput reuse workspace is `320,284,824 B` at 864 | destructive message reverse completed; direct adjoint accumulation remains |
| Retained graph embeddings, two 64-wide layers | `E` | `512 B/edge` | `38.39 MiB` | keep one `256 B/edge` phase buffer and recompute |
| Prepared geometry and spherical harmonics | `N + E` | `72 B/atom + 516 B/edge` | `38.75 MiB` | alias raw/normalized harmonic storage where possible |
| Graph-embedding adjoint scratch | `E` | `256 B/edge` | `19.20 MiB` | one shared phase buffer is already the minimum |
| Radial/harmonic forward and global adjoints | `E` | `216 B/edge` | `16.20 MiB` | lifetime aliasing is possible after a dependency audit |
| Generated harmonic/cutoff reverse scratch | `E` | `68 B/edge` | `5.10 MiB` | possible alias with global adjoints after accumulation |
| Prepared graph topology | `N + E` | `12 B/atom + 12 B/edge + 4 B` | `0.91 MiB` | deduplicate only after larger terms |
| Edge/source schedules | `E`, `N`, and block occupancy | exact CSR formula below | about `1 MiB` for this graph | low priority |
| Energy and force outputs | `N + E` | `36 B/atom + 24 B/edge + 72 B` | `1.83 MiB` | directed force buffer could be fused with reduction |

The current `edge_workspace_bytes` measurement is `95,954,600 B`, or about
`1,220.4 B/edge` on both the 256-atom (`E=23,296`) and 864-atom records. Its
large edge-scaled components are the `512 + 256 + 68 B/edge` rows above. The
remaining bytes include the schedule and atom-scaled `ir_mul` views, so it is
a diagnostic subtotal rather than a pure edge coefficient.

## Exact formulas

### Prepared graph

The graph retained for repeated evaluations contains node type and receiver
degree, a receiver-offset entry, source and neighbor type per edge, and an
edge target:

```text
prepared graph bytes = 12*N + 12*E + 4.
```

The host vectors that mirror this topology do not consume GPU memory.

### Geometry and spherical harmonics

For Float32 and `l_max=3` (`num_lm=16`):

```text
prepared/current atom geometry                 72*N bytes
prepared xyz and distance                      32*E bytes
reference xyz                                  24*E bytes
shuffled xyz                                   12*E bytes
spherical harmonics Y                          64*E bytes
normalized harmonic gradient                  192*E bytes
raw/shuffled harmonic gradient                192*E bytes
cell, inverse cell, pbc, invalid flag             160 bytes

geometry/harmonic bytes = 72*N + 516*E + 160.
```

The edge-capacity allocator grows geometrically. Its active requirement is
linear in `E`, but after graph sizes change, reserved edge capacity follows a
staircase and can approach twice the largest previously requested active
extent. Capacity benchmarks must report both active edges and reserved edges.

### Model-input and global reverse arrays

The evaluator retains cutoff, ten radial values, sixteen copied harmonics,
and their global reverse arrays:

```text
cutoff                         4*E
radial                        40*E
edge harmonics                64*E
radial adjoints               40*E
harmonic adjoints             64*E
cutoff adjoints                4*E

radial/harmonic arrays = 216*E bytes.
```

The `64*E` `edge_harmonics` copy duplicates `Y`. The raw harmonic gradient is
another `192*E` candidate for aliasing or in-place transformation if the
sphericart and force-reverse lifetime contract permits it.

### Generated graph-wide execution

Both official MH-1 interactions use a 64-wide conditioned graph embedding:

```text
one embedding or embedding-adjoint buffer = 4*64*E = 256*E bytes.
```

Full retention keeps two embeddings (`512*E`). The bounded policy retains or
recomputes each layer while maintaining one reusable embedding phase buffer
(`256*E`). Reverse already shares one embedding-adjoint buffer between layers
(`256*E`). The generated reverse also shares harmonic and cutoff scratch:

```text
graph reverse harmonic/cutoff scratch = 4*(16 + 1)*E = 68*E bytes.
```

For an architecture whose factorized prefix is an identity, the fallback can
also materialize a 64-wide graph fixed contribution (`256*E` per simultaneously
live layer). The official MH-1 checkpoint in this audit has non-identity
conditioning and does not allocate this optional buffer.

### Source schedules

Let:

- `B = ceil(E / block_size)`;
- `S_b` be the number of distinct sources in edge block `b`;
- `S_g` be the number of graph-wide active sources;
- `R_g` be the number of active receivers.

The current seven device integer arrays consume

```text
schedule bytes = 4 * ((B + 1)
                    + (sum_b S_b + 1)
                    + E
                    + E
                    + (S_g + 1)
                    + E
                    + R_g).
```

Thus the schedule has a fixed `12*E` component plus occupancy terms. Its
worst-case upper bound is still linear: `sum_b S_b <= E`, `S_g <= min(N,E)`,
and `R_g <= min(N,E)`. CUDA uses a 16,384-edge block; HIP uses 1,024, so HIP
can repeat a source in more block-local segments, but it does not introduce a
superlinear allocation.

### Node state

The name `node_workspace_bytes` means all node-indexed execution state, not the
small categorical node attribute alone. For the official checkpoint, the
per-atom part of the current generated `recompute-v1` path is:

```text
readout contribution                                         4 B/atom
generated input ir_mul / up state                        2,560 B/atom
generated output ir_mul / forward messages              28,672 B/atom
layer outputs                                            10,240 B/atom
layer output adjoints                                    10,240 B/atom
shared message-adjoint backing                           20,480 B/atom
shared up-adjoint backing                                 2,048 B/atom
shared graph-input-adjoint backing                        2,048 B/atom
density scalars and shared density adjoint                   12 B/atom

recompute-v1 atom-scaled subtotal                        76,304 B/atom
```

The generated path now skips the generic one-hot attributes and initial
embedding because layer-0 pre-forward owns the embedding lookup, and aliases
the forward messages directly to `execution_output_ir_mul`. Those changes
remove `2,404 + 28,672 = 31,076 B/atom` without changing the generated program
or replaying work. Layer 1 and layer 0 reverse also execute sequentially and
now reuse packed message, up, graph-input, and density adjoint backing stores,
removing another exact `9,220 B/atom`.

Full retention adds `77,824 B/atom` of pre-gate state and `65,536 B/atom` of
interaction output, or `143,360 B/atom`. They are forward intermediates needed
by reverse, not additional learned node attributes. `recompute-v1` regenerates
them instead of retaining them.

`reuse-adjoints-v1` retains those two tensors and removes the separate
`20,480 B/atom` message-adjoint backing. It fuses each row's message-density
dot product into grouped message reverse before overwriting the dead message
row with its adjoint. Its slope is therefore

```text
reuse-adjoints-v1 = 76,304 + 143,360 - 20,480
                  = 199,184 B/atom.
```

The current metric also includes constant learned runtime packs and bounded
phase workspaces. The runtime packs are `23.20 MiB`. The ten host phase views
use `min(N,256)` rows and total `194.50 MiB` once `N >= 256`; the main arena is
`35.50 MiB` of that subtotal. The other nine views (`159.00 MiB`) implement
only the CPU tile schedule and are no longer allocated on CUDA/HIP.

The main arena was already capped at 256 rows, but the RTC path previously
launched all `N` nodes at once and indexed the arena with the global launch
node. That was out of bounds for `N > 256`. RTC pre/post forward and reverse
now launch in 256-row batches, shift every persistent node pointer to the
batch origin, and reuse the unshifted arena with a launch-local node index.
This makes the bounded device arena valid. The capacity-first tile costs more
launches; a larger tile is a separate performance/capacity policy, not the
default for this work.

The current generated-host formula above 256 atoms is

```text
recompute-v1 node bytes = 76,304*N + 228,274,840
reuse-adjoints-v1 node bytes = 199,184*N + 228,274,840
full-retention node bytes = 219,664*N + 228,274,840.
```

At 864 atoms the generated-host measurement is `294,201,496 B`. Subtracting
the nine host-only phase views projects the device-resident node allocation as

```text
recompute-v1 device node bytes = 76,304*N + 61,551,256
reuse-adjoints-v1 device node bytes = 199,184*N + 61,551,256
full-retention device node bytes = 219,664*N + 61,551,256.
```

These device formulas use the 256-row capacity arena. The corresponding
864-atom recompute and full-retention values, now measured on HIP, are
`121.57 MiB` and `239.70 MiB`. The 1,024-row CUDA throughput arena has a larger
bounded intercept but the same exact per-atom slopes. All are node-workspace
diagnostics rather than process peaks; backend runtime allocations and
allocator overhead are outside `node_workspace_bytes`. The process-level
rocprof peaks are reported in the HIP qualification section above.

Historical exact-contract GPU records, from before the current CPU-oriented
phase-view expansion, give:

```text
full-retention-v1 at N=864       350,237,952 B = 334.01 MiB
recompute-v1 at N=864            226,374,912 B = 215.89 MiB
retention cost                    123,863,040 B = 143,360 B/atom
```

Those records still establish the exact `143,360 B/atom` retention delta, but
their totals predate the bounded RTC arena and current storage aliases. Above
the 256-row tile size, the current incremental slopes are `76,304 B/atom`
under recompute and `219,664 B/atom` under full retention. A fresh current-tree
GPU scale run must validate the derived intercept before using it for an
absolute boundary.

## Further node-state reduction

The original `116,600 B/atom` floor is reducible. The experiments proceed in
this order:

1. Completed: device execution allocates only `runtime.arena`, and all four
   RTC node phases use bounded 256-row launches. This avoids `159.00 MiB` of
   CPU-only phase views for the official checkpoint.
2. Completed: generated messages alias `execution_output_ir_mul`, saving
   exactly `28,672 B/atom`.
3. Completed: generated execution skips generic one-hot attributes and the
   initial embedding, saving exactly `2,404 B/atom`. Fusing the one-float
   readout contribution could save another `4 B/atom`, but is low priority.
4. Completed: share reverse buffers between interactions. Reverse processes
   layer 1 completely before layer 0, so one maximum-width allocation can back
   both layers' message adjoint, up adjoint, graph input adjoint, and density
   adjoint. The exact saving is `9,220 B/atom`.

The first four completed steps reduce the recompute slope from `116,600` to a
measured `76,304 B/atom`. The reuse policy is not a smaller recompute mode: it
adds the `143,360 B/atom` reverse-required primal state and removes the
`20,480 B/atom` message-adjoint allocation, yielding `199,184 B/atom` without
replaying the node program.

The generated node reverse has one completed and one remaining lifetime
opportunity:

- Completed: `reuse-adjoints-v1` overwrites each forward-message row with its
  adjoint after fusing the density dot product into grouped reverse. This saves
  the shared `20,480 B/atom` message-adjoint allocation at CUDA timing parity
  with full retention.
- Accumulate generated graph input adjoints directly into `up_adj`, potentially
  saving another `2,560 B/atom`. The graph-wide source CSR has one owner per
  active source, but generated source work is also partitioned by input path and
  channel tile. Direct accumulation is therefore safe only if each partition
  writes disjoint input locations or the renderer provides an ordered reduction.
  Treat this saving as conditional until that contract is tested.

The destructive-message result is now a measured `199,184 B/atom` slope.
Combining it with direct graph accumulation would reach approximately
`196,624 B/atom`. That remaining value is a medium-term exact-Float32 target,
not a measured result. Edge-scaled allocations remain dominant enough that
the process-level capacity gain is smaller than the node-slope ratio.

More expensive options are:

- Generate the layer-1 readout seed inside node reverse instead of retaining
  its full output-adjoint row. Potential saving: `2,048 B/atom`.
- Recompute and reuse layer up state. Because graph source reverse needs
  graph-wide random access, this still requires one full maximum-width buffer;
  the net saving is only about `512 B/atom`, so it is low priority.
- Recompute forward messages before node reverse. This can remove the remaining
  `28,672 B/atom`, but repeats interaction forward, previously `14.266 ms` for
  the 864-atom graph. Keep it as an explicit maximum-capacity policy.
- Tiling across complete graph and node reverse phases could remove more
  full-graph state, but source-owned graph reverse has nonlocal node access.
  It requires a new schedule, not just a smaller arena.

Mixed-precision activation storage could halve selected rows, but changes the
numerical contract and should be considered only after the exact FP32 lifetime
work is exhausted.

## Deferred direction

Host streaming and host offload are explicitly out of scope for the current
capacity work. The active direction keeps execution state device-resident and
reduces it through bounded arenas, lifetime aliasing, cross-layer reuse,
destructive reverse, and selective recomputation.

## Accounting cautions

The public diagnostics currently overlap:

- `precision_workspace_bytes` equals edge workspace plus shared linear/tensor/
  product workspaces; it does not include all node, geometry, topology, radial,
  result, or model allocations.
- `edge_workspace_bytes` includes `execution_input_ir_mul`,
  `execution_output_ir_mul`, and `execution_input_adjoint_ir_mul`, even though
  those views have `N` rows.
- `node_workspace_bytes` includes those same three `ir_mul` views.

Therefore `node_workspace_bytes + precision_workspace_bytes +
execution_geometry_workspace_bytes` double-counts some node storage and still
omits other allocations. Capacity work should expose a deduplicated allocation
breakdown rather than treating the existing subtotals as an additive total.

## Capacity priorities

1. Keep `N` and `E` as separate benchmark axes and report `E/N`. At this
   AlN graph, every `1 B/edge` is already `91 B/atom`.
2. Keep the completed bounded RTC arena, generated-message alias, and fused
   layer-0 embedding allocation changes qualified together.
3. Use `reuse-adjoints-v1` as the first maximum-capacity mode. On CUDA it keeps
   full-retention timing parity while removing `20,480 B/atom`, and it raises
   the measured AlN boundary from full retention's 108,000 atoms to 119,164
   atoms on the 32 GB RTX 5090.
4. Keep the completed cross-layer and destructive-message reverse reuse
   qualified together. Complete HIP correctness and capacity qualification
   before making a cross-backend default decision.
5. Accumulate graph input adjoints directly into `up_adj` if the generated
   partition ownership proves disjoint.
6. Recompute one graph embedding, saving `256 B/edge` while preserving a
   single bounded phase buffer.
7. Experiment with aliasing harmonic storage (`64 B/edge`, then potentially
   `192 B/edge`). For identity-prefix model variants, also avoid the optional
   `256 B/edge` fixed-contribution materialization. These retain the package's
   factorized, low-memory direction.
8. Leave topology and schedule compaction until the larger tensor lifetimes
   have been reduced.

The next implementation step should add allocation-level diagnostics for each
row above and a deduplicated total. Then run an AlN scale series that varies
both replication (`N`) and cutoff/skin (`E/N`) to verify the fitted atom and
edge coefficients against process-level peak VRAM. HIP qualification of the
new reuse policy remains outstanding; the implementation is backend-neutral,
but the CUDA result alone does not establish HIP performance or capacity.
