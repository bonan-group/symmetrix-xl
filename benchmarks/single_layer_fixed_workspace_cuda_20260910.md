# Single-layer fixed-workspace CUDA scaling (2026-09-10)

## Scope

This record compares the FP32 single-v2 capacity Y-only plan with the
single-layer fixed receiver workspace on an NVIDIA GeForce RTX 5090. The model
was `/path/to/mace-single-v2-e6.model` with SHA-256
`529b017721c7fdb406a8c2625922ccdcd7cd03f0492e073dd424e651bfd48c6c`.
The CUDA extension used for the scaling and workspace sweeps had SHA-256
`59a5ce4e9b5e3a9798d835c81323d93d28a053e1ac28a49f667e9cd199c25ac1`.

The GPU used driver 610.43.02, compute capability 12.0, CUDA 13.3, and
32,607 MiB total memory. Runs used the 6.0 A model cutoff, 0.5 A neighbor
skin, 6.5 A effective cutoff, direct prepared execution, Y-only harmonics,
and zero fallback evaluations. Timings are synchronized medians in us/atom.
Graph construction, JIT compilation, and the initial calculation are outside
the measured samples.

## Capacity result

The fixed-workspace plan meets its primary goal: it evaluates systems that the
capacity Y-only plan cannot allocate.

| Plan | Atoms | Directed edges | Peak MiB | us/atom | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Capacity Y-only | 1,000,188 | 109,020,492 | 31,392 | 1.782 | Fits at 96.3% of device memory |
| Fixed v1, 32,768 receivers | 1,000,188 | 109,020,492 | 16,666 | 1.890 | Fits, saves 14,726 MiB |
| Capacity Y-only | 1,098,500 | 119,736,500 | - | - | OOM allocating 1.048 GiB for `MH0 H2 forward state` |
| Automatic capacity | 1,098,500 | 119,736,500 | 18,204 | 1.864 | Selects fixed, 34 batches/evaluation |
| Fixed v1, 32,768 receivers | 1,898,208 | 206,904,672 | 30,588 | 1.883 | Fits, 58 batches/evaluation |
| Fixed v1, 32,768 receivers | 1,972,156 | 214,965,004 | 31,744 | 1.872 | Fits, 61 batches/evaluation |
| Fixed v2, 32,768 receivers | 2,916,000 | 317,844,000 | 18,770 | 2.007 | Fits, tile-local harmonics and directed forces |
| Fixed v2, 32,768 receivers | 4,000,000 | 436,000,000 | 25,194 | 1.882 | Fits, 123 batches/evaluation |
| Fixed v2, 32,768 receivers | 4,630,500 | 504,724,500 | 28,932 | 1.824 | Fits, 142 batches/evaluation |
| Fixed v2, 32,768 receivers | 5,038,848 | 549,234,432 | 31,358 | 1.836 | Fits at 96.2% of device memory |
| Fixed v2, 32,768 receivers | 5,180,116 | 564,632,644 | - | - | OOM allocating 118.6 MiB for final atom forces |
| Compact geometry, 32,768 receivers | 5,180,116 | 564,632,644 | 21,472 | 1.816 | Fits with tile-local direction and radius |
| Compact geometry, 32,768 receivers | 7,812,500 | 851,562,500 | 31,608 | 1.816 | Fits at 96.9% of device memory |
| Compact geometry, 32,768 receivers | 8,001,504 | 872,163,936 | - | - | OOM allocating 81.75 MiB for tile directed forces |
| Compact connectivity, 32,768 receivers | 10,061,824 | 1,096,738,816 | 19,130 | 1.838 | Fits beyond the previous edge and atom limits |
| Compact connectivity, 32,768 receivers | 11,943,936 | 1,301,889,024 | 22,414 | 1.830 | Largest host-feasible ASE construction tested |

The final 1,898,208-atom result used one warmup and three samples:
3,575.405, 3,574.701, and 3,536.208 ms. The workspace remained 453,246,976
bytes (432.25 MiB), or 13,832 bytes for each of 32,768 receiver slots.
That pinned-plan run used extension
`2db027d654373b116caf9d00f8c862d35efbc07cafbb0be06c5a8fd6ddc28d1d`;
the final extension adds automatic boundary selection without changing the
pinned fixed-workspace kernels.

The first attempt above 134,217,727 directed edges exposed a signed 32-bit
overflow in the Y-only harmonic output offset, `16 * edge`. The final build
casts the edge index before all flattened harmonic and coordinate products.
With `CUDA_LAUNCH_BLOCKING=1`, the 1,898,208-atom case then completed before
the normal multi-sample result was collected.

The final extension was also run without a debug plan at 1,098,500 atoms.
Automatic capacity selection chose `mh0-single-layer-tiled-v1`; its three
samples were 2,056.778, 2,047.563, and 2,040.592 ms. This verifies that the
memory-boundary result is available through normal capacity mode.

The earlier fixed-workspace implementation still retained harmonics and
directed forces for every graph edge. The v2 implementation moves both into
the active receiver tile, immediately reduces each tile's directed forces into
the final atom-force array, and accumulates its virial into nine FP64 values.
It also omits the unused source-ordered R1 permutation for single-layer graphs.

That v2 extension bracketed its device limit at 5,038,848 atoms. Moving unit
directions and radii into the active tile moves the boundary to between repeat
125 and 126. Repeat 125 completed 7,812,500 atoms and 851,562,500 directed
edges with zero fallbacks. Repeat 126 reached 8,001,504 atoms but could not
allocate the 81.75 MiB tile-local directed-force array. The largest successful
case is 1.55 times the v2 limit and 3.96 times the original 1,972,156-atom
fixed-workspace limit.

## Memory growth

The compact-geometry arena has two bounded dimensions. Its 32,768 receiver
slots use 432.38 MiB, including a local CSR offset per receiver. Its 3,571,712
edge slots use another 367.88 MiB for 64-byte FP32 harmonics, 24-byte FP64
directed forces, 12-byte FP32 unit directions, and 8-byte FP64 radii. The total
839,122,944-byte (800.25 MiB) workspace is constant once both maxima are
reached. The active edge extent
is the exact largest CSR span among receiver tiles, so a high-degree receiver
cannot overflow a receiver-only estimate.

Inactive graph storage is 36 bytes per directed edge: 12 bytes of connectivity
and 24 bytes of reference edge geometry. The 112 bytes per inactive node cover
topology, current/reference/displacement coordinates, energy, and final force.
At 109 directed edges per atom, the exact retained slope is therefore 4,036
bytes (3.941 KiB) per atom. Learned harmonics, directed forces, unit directions,
and radii no longer contribute to the graph-size slope.

| Atoms | Directed edges | Y-only us/atom | Fixed v1 us/atom | Change | Y-only MiB | Fixed v1 MiB | Saved MiB | Batches/evaluation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,000 | 3,488,000 | 1.538 | 1.619 | +5.3% | 1,720 | 1,646 | 74 | 1 |
| 131,072 | 14,286,848 | 1.576 | 1.705 | +8.2% | 4,770 | 3,202 | 1,568 | 4 |
| 256,000 | 27,904,000 | 1.690 | 1.772 | +4.8% | 8,604 | 5,134 | 3,470 | 8 |
| 500,000 | 54,500,000 | 1.699 | 1.778 | +4.6% | 16,106 | 8,910 | 7,196 | 16 |

From 131,072 through 500,000 atoms, capacity Y-only grows by about 30.5 KiB
per atom while fixed workspace grows by about 15.5 KiB per atom. Removing the
graph-sized one-layer state therefore cuts the observed large-system memory
slope by about 49%. It does not halve total memory at small sizes because the
shared graph and edge allocations are unchanged.

The v2 and compact-geometry measurements show the additional slope reduction:

| Revision | Atoms | Directed edges | Peak MiB | us/atom | Workspace MiB | Fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 | 500,000 | 54,500,000 | 4,426 | 1.873 | 732.00 | 0 |
| v2 | 2,916,000 | 317,844,000 | 18,770 | 2.007 | 732.00 | 0 |
| v2 | 4,000,000 | 436,000,000 | 25,194 | 1.882 | 732.00 | 0 |
| v2 | 4,630,500 | 504,724,500 | 28,932 | 1.824 | 732.00 | 0 |
| v2 | 5,038,848 | 549,234,432 | 31,358 | 1.836 | 732.00 | 0 |
| Compact geometry | 500,000 | 54,500,000 | 3,454 | 1.764 | 800.25 | 0 |
| Compact geometry | 5,180,116 | 564,632,644 | 21,472 | 1.816 | 800.25 | 0 |
| Compact geometry | 7,812,500 | 851,562,500 | 31,608 | 1.816 | 800.25 | 0 |
| Compact connectivity | 500,000 | 54,500,000 | 2,416 | 1.764 | 813.88 | 0 |
| Compact connectivity | 10,061,824 | 1,096,738,816 | 19,130 | 1.838 | 813.88 | 0 |
| Compact connectivity | 11,943,936 | 1,301,889,024 | 22,414 | 1.830 | 813.88 | 0 |

The earlier v2 measurements grew by 6.08 KiB per atom. With compact geometry,
the 500,000-to-7,812,500-atom measured slope is 3.942 KiB per atom, matching
the 3.941 KiB retained-storage calculation. This is a further 35.2% reduction
from v2, a 74.6% reduction from the first fixed plan's 15.5 KiB/atom slope, and
an 87.1% reduction from capacity Y-only's 30.5 KiB/atom slope. At 500,000
atoms, compact geometry also reduces peak memory from 4,426 to 3,454 MiB and
improves latency from 1.873 to 1.764 us/atom.

Compact connectivity removes graph-sized receiver and neighbor-type arrays,
stores three signed integer periodic shifts instead of an FP64 edge vector,
and transforms current positions in place into displacements. Persistent
storage is therefore 16 bytes per edge and 88 bytes per node. At 109 edges per
atom, the exact slope is 1,832 bytes (1.789 KiB) per atom. The observed
500,000-to-10,061,824 slope was 1,832.9 bytes per atom. Extrapolating that
measured slope and its 1,542 MiB intercept to 32,607 MiB gives a device boundary
near 17.77 million atoms. A direct 17-million-atom ASE run was not attempted:
host neighbor-list construction used about 67 GiB resident memory at 10.06
million atoms and would exhaust this 121 GiB host before device allocation.

Direct single-layer execution also skips the unused chunk-by-source R1
schedule. Without this change, large fixed-workspace graphs failed schedule
preparation even though that schedule is never consumed.

## V1 workspace-size sweep

The production default is 32,768 receivers. A sweep at 1,098,500 atoms tested
whether a larger arena recovered enough batching overhead to justify reducing
the maximum solvable graph size.

| Receiver capacity | Workspace MiB | Peak MiB | Batches/evaluation | us/atom | Change from 32,768 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 432.25 | 18,204 | 34 | 1.854 | baseline |
| 65,536 | 864.50 | 18,636 | 17 | 1.875 | +1.1% |
| 131,072 | 1,729.00 | 19,500 | 9 | 1.842 | -0.7% |
| 262,144 | 3,458.00 | 21,230 | 5 | 1.838 | -0.9% |
| 524,288 | 6,916.00 | 24,688 | 3 | 1.872 | +1.0% |

The measurements used one warmup and three samples in fresh processes. They
were collected with extension
`cb06f1779a0b931415d8a6d531132e416f91c6694d1c700f0b92479dec2ec179`;
the subsequent large-index cast is inactive for this 119,736,500-edge workload,
and automatic plan selection is bypassed by the sweep's explicit pins.

Reducing 34 batches to five changes latency by less than 1% and costs about
3.0 GiB. The 524,288-receiver case is slower again. A larger default would
therefore trade away useful graph capacity without a material throughput gain.

## Compact-connectivity workspace sweep

The final representation was swept again at 500,000 atoms and 54,500,000
directed edges. The graph-sized Y-only control measured 1.689 us/atom.

| Receiver capacity | Workspace MiB | Peak MiB | Batches/evaluation | us/atom | Change from 32,768 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 203.47 | 1,808 | 62 | 1.790 | +1.5% |
| 16,384 | 406.94 | 2,010 | 31 | 1.766 | +0.1% |
| 32,768 | 813.88 | 2,416 | 16 | 1.764 | baseline |
| 65,536 | 1,627.75 | 3,230 | 8 | 1.758 | -0.3% |

The 65,536-receiver workspace recovers only 0.006 us/atom while consuming an
additional 813.88 MiB. The 32,768 default remains the balanced point. Its
1.764 us/atom result is identical at the shown precision to the earlier
36-byte-edge compact-geometry implementation, so the compact connectivity and
tile-local type reconstruction introduce no measurable regression.

## Regression attribution

Nsight Systems profiles used the same 32,000-atom graph and one receiver batch.
Capacity Y-only took 1.545 us/atom; fixed workspace took 1.633 us/atom. Both
profiles issued 423 CUDA kernel launches, and CUDA launch overhead was slightly
lower for fixed workspace. The difference is inside the R0 coordinate reverse:

| Kernel stage | Capacity Y-only ms/evaluation | Fixed ms/evaluation |
| --- | ---: | ---: |
| R0 coordinate reverse | 20.501 | 23.217 |
| R0 forward | about 7.50 | about 7.50 |
| H2 reverse | about 3.51 | about 3.51 |

Capacity Y-only uses the faster generated edge-owned coordinate reverse. The
fixed arena uses receiver-owned CSR reverse because each batch's local
adjoints cannot be indexed directly by global edge receiver IDs. The 2.715 ms
coordinate-reverse difference accounts for essentially all one-batch latency
regression. Additional batches add little at large graph sizes.

Reports are retained at:

- `/tmp/symmetrix-single-layer-fixed-profile-20260910/baseline-32000.nsys-rep`
- `/tmp/symmetrix-single-layer-fixed-profile-20260910/fixed-32000.nsys-rep`

## Qualification

The focused CUDA suite passed with the reviewed extension, SHA-256
`d655a286d723e1e0ecdb39b9a1f42f65eec51b349187ad75f766124b855bfe77`:

```text
14 passed, 5 skipped
```

The review added FP64 alignment padding for otherwise-supported odd-channel
models. The 128-channel benchmark model requires no padding, so its allocation
formula is unchanged. A repeat of the 500,000-atom, 32,768-receiver case with
the reviewed extension used the same 853,409,792-byte workspace and 2,416 MiB
peak, with a 1.778 us/atom median versus 1.764 us/atom in the campaign above.
The 0.8% difference is within run-to-run variation.

The tile-boundary tests compare energy, per-atom energies, every force
component, and stress against capacity Y-only for partial and full batches.
Campaign energies were identical, force maxima differed by approximately
`1e-8` to `1e-7 eV/A`, and every benchmark reported zero fallback evaluations.
The first repeat-125 compact-geometry run exposed a second signed product in
`3 * global_edge` while loading reference geometry. Widening that product was
validated by the successful 851,562,500-edge run, well beyond its
715,827,882-edge overflow threshold.

The benchmark driver is `benchmarks/single_layer_fixed_workspace_cuda.py`.
The principal raw records are:

- `/tmp/symmetrix-single-layer-fixed-workspace-large-scale.json`
- `/tmp/symmetrix-automatic-repeat65-final.log`
- `/tmp/symmetrix-capacity-32768-repeat78-final.log`
- `/tmp/symmetrix-capacity-32768-repeat79.log`
- `/tmp/symmetrix-capacity-32768-repeat80.log`
- `/tmp/symmetrix-capacity-{32768,65536,131072,262144,524288}.log`
