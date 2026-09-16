# MACE-MH-1 HIP baseline on 864-atom AlN

## Contract

- Date: 2026-08-08
- Source revision: `54547fdedd61acaf71be379a43f7e40e6051076c`
- Device: AMD Radeon 8060S Graphics (`gfx1151`, 40 compute units, wavefront 32)
- Runtime: ROCm 7.14, Kokkos HIP, Float32
- Structure: wurtzite AlN, `a=3.112 A`, `c=4.982 A`, repeated `6x6x6`
- Graph: 864 atoms and 78,624 directed edges
- Timing: prebuilt graph, synchronized native energy-and-force evaluator
- Samples: five warmups and ten measured evaluations for Symmetrix

The matched checkpoints are from the official `mace_mh_1` release and use the
common `omat_pbe` head with only Al and N retained:

| Checkpoint | SHA256 |
|---|---|
| `mace-mh-0.model` | `d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d` |
| `mace-mh-1.model` | `a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47` |

## Results

| Implementation | Median (ms/step) | Range (ms/step) | Median (us/atom) | MH1/MH0 |
|---|---:|---:|---:|---:|
| Symmetrix MH1 generic Kokkos `all` | 1950.469 | 1944.554-1965.728 | 2257.487 | 78.685x |
| Symmetrix MH0 `direct`, | 24.788 | 24.424-24.921 | 28.690 | 1.000x |
| PyTorch/e3nn MH1 control | 2088.751 | 2084.148-2090.695 | 2417.536 | 3.919x PyTorch MH0 |
| PyTorch/e3nn MH0 control | 532.962 | 530.469-533.988 | 616.854 | 1.000x |

The 2.5x acceptance limit is `61.971 ms/step`. The current HIP MH1 path needs
a 31.47x speedup to reach that limit. Generic Symmetrix MH1 is already 6.6%
faster than upstream e3nn MH1 on this device; the large ratio comes from MH0's
generated hipRTC path, which is 21.50x faster than upstream e3nn MH0.

Path assertions from the retained result:

- MH1 matches the strict 512-node-channel, 128-edge-channel, 10-radial,
  `l_max=3` family, but runs with JIT disabled because generated HIP currently
  rejects nonlinear MH1.
- MH1 selects packed GEMM linears and the generic
  `official_kokkos_mdrange` tensor-product backend. Fused gate-normalization
  reverse and direct-node tensor reverse are unavailable on HIP.
- MH0 uses a prepared direct execution graph and a cached hipRTC artifact. The compiler
  backend is `hiprtc`; no fallback evaluation occurred.

## Remaining gap

MH1 now lowers one backend-neutral generated program through CUDA and HIP
dialects, with thin CUDA Driver and HIP Module launch adapters. Runtime
generation is RTC-only and fails closed. The remaining promotion blocker is
performance qualification against MH0, plus fresh NVIDIA qualification of the
shared NVRTC schedule; it is no longer a separate HIP implementation gap.

The retained RTX 5090 generated-CUDA v13 result for the same official MH1
model and graph was `27.247 ms/step`. It shows that the persistent generated
architecture can cross the absolute HIP target, but it is historical CUDA
evidence rather than a current NVRTC result on this host. Current NVRTC
qualification covers generated MH1 correctness and small generalized models;
a fresh-process official-model CUDA performance qualification remains open.

Raw local records are in `benchmarks/.artifacts/mh1-baseline/`.

## Shared hipRTC implementation bring-up

The first end-to-end shared-program implementation was exercised on the same
864-atom graph with hipRTC required and no fallback:

| Schedule | Median (ms/step) | Energy (eV) | Force norm (eV/A) | Status |
|---|---:|---:|---:|---|
| Persistent generated nodes | 429.631 | -6423.654872 | 3.168528 | Correctness-first production schedule |
| Generated interactions, generic nodes | 400.013 | -6423.653042 | 3.168524 | Isolation result |
| Experimental 8x32 tiled nodes | 267.448 | -4986.193618 | 101.588109 | Rejected: physics mismatch |

The retained generic reference is `-6423.653261 eV` with a `3.168540 eV/A`
force norm. The persistent schedule proves the shared hipRTC module, neutral
loader, forward/reverse execution, and prepared-graph integration, but its
`17.33x` MH0 ratio does not satisfy the `2.5x` promotion gate. The invalid
tiled schedule is excluded from dispatch until it is redesigned and passes
the official-model physics checks.

## Phase 6 shared-schedule optimization

Kernel tracing of the correct shared hipRTC path identified two dominant
launches in the initial generated implementation:

- layer-1 edge-phi reverse: `121.843 ms`;
- layer-1 monolithic node reverse: approximately `77.2 ms`.

The backend-neutral generator now uses a path-tiled edge-phi reverse kernel
for path-heavy interactions. One block owns an edge, stages path adjoints in
shared memory, and performs coalesced phi reductions without the former
per-thread 64-float adjoint array. The same generated kernel body and launch
metadata are consumed by NVRTC and hipRTC. A fourfold block multiplier is used
for this schedule; an eightfold edge multiplier did not improve the measured
kernel.

The exact node reverse was split at existing arena lifetime boundaries into
readout, core, and tail kernels while the experimental 8x32 tiled-node failure
was diagnosed. The persistent grid target increased from four to eight blocks
per compute unit. A measured 16-block target regressed the traced step to
`245.846 ms` and was rejected.

| Generated MH1 trace | Step (ms) | Layer-1 edge phi (ms) | Energy (eV) | Force norm (eV/A) |
|---|---:|---:|---:|---:|
| Before Phase 6 schedule work | 366.418 | 121.843 | -6423.654872 | 3.168528 |
| Path-tiled edge, monolithic node reverse | 202.556 | 24.6 | -6423.654872 | 3.168526 |
| Path-tiled edge, segmented node reverse | 177.279 | 24.753 | -6423.654872 | 3.168526 |

The final traced kernel sum is approximately `176.8 ms`. The change is a
`51.6%` step-time reduction from the pre-optimization generated trace, but the
result is still `2.86x` above the `61.971 ms` acceptance limit. The largest
remaining individual launches are layer-1 edge-phi reverse (`24.753 ms`),
layer-1 node-reverse tail (`20.272 ms`), layer-1 node post-forward
(`13.330 ms`), and layer-1 source reverse (`12.836 ms`). These are the next
Phase 6 optimization targets.

Raw local profiler output is retained under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_segmented/` for this development run.

### Shared tiled node schedule

The apparent physics failure in the earlier 8x32 node experiment was traced to
the module launch-plan builder. Residual and message launch fragments were
joined before lookup in the launch-record table, so the RTC module omitted
those launches in forward and in the reverse reconstruction/recomputation
sequence. Recording each fragment independently makes the existing tiled
kernel bodies correct on both CUDA and HIP without introducing a backend fork.

| Generated MH1 schedule | Step (ms) | M1/node forward (ms) | M1/node reverse (ms) | Status |
|---|---:|---:|---:|---|
| Segmented reverse milestone | 177.279 | 24.869 | 57.415 | Correct |
| Complete tiled forward and reverse | 160.400 | 21.154 | 47.424 | Correct |

The final value is a five-warmup, ten-sample evaluator-only median with a
`160.023-161.162 ms` range. The phase totals are per-evaluation averages from a
four-evaluation `rocprofv3` trace whose total device time is `159.516 ms`. The
tiled forward is 14.9% faster than monolithic forward and tiled reverse is
17.4% faster than segmented reverse. The full measured step is 9.5% faster than
the preceding milestone.

Array-level comparison against the exact segmented schedule gives zero total-
and per-atom-energy difference, `7.05e-7 eV/A` maximum force-component
difference, and `6.36e-9 eV/A^3` maximum stress difference. The cold hipRTC
build also confirms that dead schedules are absent: generated source falls
from 2,352,745 to 1,106,743 bytes and HSACO from 1,410,928 to 995,584 bytes.
The shared CUDA dialect for the same official contract compiles successfully
with CUDA 13.1 NVRTC to a 1,932,064-byte `sm_80` cubin; CUDA execution remains
pending on NVIDIA hardware.

The retained MH0 median makes this result `6.47x` slower than MH0 and `2.59x`
slower than the `61.971 ms` acceptance limit. The dominant remaining kernels
are now layer-1 edge-phi reverse (`24.797 ms`), layer-1 source reverse
(`12.492 ms`), layer-1 interaction forward (`11.023 ms`), and layer-1 harmonic
reverse (`9.966 ms`). Further Phase 6 work should target that interaction path
before revisiting the now distributed node kernels.

Raw tiled profiler output is retained under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_tiled_full_fixed/`; the stable benchmark
record is `/tmp/symmetrix-mh-pathway.bWSwqt/mh1_tiled_5x10.json`.

### Edge-phi path-adjoint batching

The path-tiled reverse kernel now stages four independent path-adjoint rows in
shared memory before reducing them into `phi`. This reduces block-wide barriers
from 22 to 7 per edge for the ten-path layer-1 interaction while preserving the
original path, channel, shuffle, and accumulation order. The generated body and
schedule remain common to NVRTC and hipRTC.

| Path-adjoint schedule | Median (ms) | Range (ms) | Layer-1 edge phi (ms) | Status |
|---|---:|---:|---:|---|
| One path per barrier pair | 160.400 | 160.023-161.162 | 24.797 | Previous milestone |
| Four paths per batch | 159.648 | 159.346-160.017 | 23.520 | Retained |
| Five paths per batch | 162.293 | 161.769-162.706 | Not retained | Rejected |
| Ten paths per batch | 161.125 | 160.851-161.845 | Not retained | Rejected |

A separate cached-module pair in the same clock regime measured `160.187 ms`
before batching and `159.279 ms` after batching. Energy, per-atom energies,
forces, and stress are unchanged. The official CUDA module compiles with CUDA
13.1 NVRTC to a `1,930,528`-byte `sm_80` cubin; the HIP runtime result reports
hipRTC and `generated_hip_v4`.

Two adjacent policy probes were not retained. The existing three-partition
shared layer-1 forward policy measured `160.620 ms`, slower than the automatic
six-partition policy. Grouping harmonic reverse paths by their input block
reduced the layer-1 kernel from 192 VGPRs plus one spill to 102 VGPRs without a
spill, but increased its traced latency from about `9.95 ms` to `10.41 ms` and
lost `0.28-0.59 ms` in paired full-step runs. The next optimization should
change data reuse or work ownership rather than only reducing live registers.

The retained batch-four result is `6.44x` the `24.788 ms` MH0 median and
`2.58x` the `61.971 ms` acceptance limit. Raw profiler output is under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_edge_phi_batch4/`; the stable record is
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_edge_phi_batch4_5x10.json`.

### Four-edge linear-weight reuse

Each 128-thread path-tiled block now owns four strided edges. For every path
and output `phi`, the kernel loads one shared linear-weight coefficient and
applies it to four independent edge accumulators. Per-edge path, channel,
shuffle, and final accumulation order is unchanged. Inactive tail slots are
zero-filled and all barriers remain block-uniform. The generated numerical
body remains shared by NVRTC and hipRTC.

| Edge schedule | Median (ms) | Range (ms) | Layer-1 edge phi (ms) | Status |
|---|---:|---:|---:|---|
| One edge, four-path batches | 159.648 | 159.346-160.017 | 23.520 | Previous milestone |
| Four edges, four-path batches | 150.667 | 150.292-151.084 | 15.596 | Retained |
| Eight edges, four-path batches | 156.577 | 156.104-156.756 | Not profiled | Rejected |
| Four edges, 2 blocks/CU target | 153.381 | 153.303-153.825 | Not profiled | Rejected |

A same-clock cached-module pair measured `158.782 ms` for one edge and
`150.450 ms` for four edges, a `5.25%` evaluator reduction. The layer-1 kernel
improves by `33.7%` in four-evaluation traces. Energy, per-atom energies,
forces, and stress are bitwise identical. On `gfx1151`, the retained layer-1
kernel has 70 VGPRs, 72 SGPRs, no private segment, and no spills. The eight-edge
probe also remained spill-free at 86 VGPRs but lost `6.13 ms` to the retained
schedule in the paired clock regime, so it was reverted.
CUDA 13.1 NVRTC compiles the same source to a `1,992,480`-byte `sm_80` cubin.

Recomputing conditioner layer-normalization scalars did not change its
`3424`-byte private frame or its 708 VGPR and 361 SGPR spills; both conditioner
reverse kernels became slightly slower, so that source change was also
reverted. The next larger targets are mixed 16x32 dual-node tiling for the six
`linear2` launch groups and blockwise conditioner reverse recomputation that
shortens activation live ranges.

The retained four-edge result is `6.08x` the MH0 median and `2.43x` the
acceptance limit. Raw profiler output is under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_edge_batch4x4/`; the stable record is
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_edge_batch4x4_5x10.json`.

### Mixed 16x32 linear2 node tiling

The six `linear2` launch groups now use a logical 16-node by 32-channel tile
with the existing 256-thread block. Each physical node lane evaluates two
nodes with independent accumulators, so the 32x32 weight tile is reused across
twice as many nodes. All residual, message, and product linears remain 8x32.
Both accumulators retain the original ascending source/target order.

The launch-plan schema is now version 2 and records `tile_nodes` and
`tile_channels` per tiled launch. The shared CUDA/HIP module loader uses those
extents for grid construction and continues to accept version-1 plans as
implicit 8x32 geometry. Schema/version mismatches and unsupported geometry
fail before launch.

| Node schedule | Median (ms) | Range (ms) | Aggregate linear2 (ms) | Status |
|---|---:|---:|---:|---|
| Uniform 8x32 | 150.667 | 150.292-151.084 | 30.952 | Previous milestone |
| Mixed 8x32 and 16x32 | 147.329 | 146.575-147.740 | 23.263 | Retained |

A same-extension cached-module pair measured `150.439 ms` for uniform 8x32
and `143.588 ms` for the mixed schedule. The four-evaluation profile shows a
`24.8%` reduction across all 24 affected kernels. Every affected dispatch has
`grid_y=54`, 256 threads, 6656 bytes of LDS, and zero scratch. HSACO metadata
reports 96 VGPRs for forward/replay kernels, 91 VGPRs for transpose kernels,
22-25 SGPRs, and no private segment or spills.

Energy, per-atom energies, forces, and stress are bitwise identical to the
8x32 module at 864 atoms. The same exact comparison passes at 9 and 17 atoms,
covering the second logical node slot and a partial second tile. hipRTC emits a
1,034,240-byte code object from 1,240,721 bytes of shared source. CUDA 13.1
NVRTC compiles that source to a 2,019,872-byte `sm_80` cubin.

The retained result is `5.94x` the `24.788 ms` MH0 median and `2.38x` the
`61.971 ms` acceptance limit. Raw profiler output is under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_linear2_dual/`; the stable record is
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_linear2_dual_5x10.json`.

### Bounded-state conditioner reverse

The GPU conditioner reverse now recognizes the checkpoint's canonical
`linear, LayerNorm, SiLU` blocks and recomputes each required forward prefix
with two reusable hidden buffers. A third buffer carries the current adjoint.
The density network uses the same bounded schedule and transposes its terminal
scalar linear directly into the current adjoint. Noncanonical conditioner
programs retain the generic reverse, and the host implementation is unchanged.
The generated numerical body remains common to NVRTC and hipRTC, adds no
launches or global workspace, and records the schedule in the module identity.

| Conditioner reverse | Median (ms) | Range (ms) | Reverse pair (ms) | Private bytes | VGPR spills | Status |
|---|---:|---:|---:|---:|---:|---|
| Full activation and adjoint state | 147.329 | 146.575-147.740 | 9.609 | 3424 | 708 | Previous milestone |
| Three-buffer prefix recomputation | 140.512 | 140.238-140.994 | 7.026 | 976 | 80 | Retained |

A same-clock cached-module pair measured `143.251 ms` for the full-state
reverse and `140.677 ms` for bounded recomputation, a `1.80%` evaluator
reduction. The two reverse kernels improve by `26.9%` in the one-evaluation
trace. Both retain 192 VGPRs; SGPR spill metadata changes from 361 to 383 while
the smaller private frame and VGPR spill count produce the measured gain.

Energy, per-atom energies, forces, and stress are bitwise identical to the
full-state module at 864 atoms and at the 9- and 17-atom tail cases. hipRTC
emits a 1,155,072-byte code object from 1,246,539 bytes of shared source. CUDA
13.1 NVRTC compiles the same program to a 2,197,408-byte `sm_80` cubin.

The retained result is `5.67x` the MH0 median and `2.27x` the acceptance
limit. Raw profiler output is under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_conditioner_bounded/`; the stable and
paired records are `mh1_conditioner_bounded_final_5x10.json`,
`mh1_conditioner_paired_old_5x10.json`, and
`mh1_conditioner_paired_new_5x10.json` in the same temporary root.

### Receiver-local edge-phi ownership

The four-edge path-tiled reverse owner now assigns each block four consecutive
edges instead of four edges separated by the persistent grid size. The
prepared 864-atom graph is receiver-contiguous: `96.70%` of four-edge groups
contain one target, the remainder contain two adjacent targets, and `98.90%`
of adjacent slots share a target. The neutral module loader and generated
launchers therefore dispatch `ceil(samples / 4)` logical edge groups while
retaining the same four-edge numerical body for CUDA and HIP. Tail guards keep
0-, 1-, 3-, 4-, and 5-edge launch geometry exact.

| Edge ownership | Median (ms) | Range (ms) | Layer-1 edge phi (ms) | Status |
|---|---:|---:|---:|---|
| Four grid-strided edges | 140.854 | 140.638-141.152 | 15.274 | Previous milestone |
| Four consecutive edges | 140.276 | 139.597-140.647 | 14.586 | Retained |

The first same-extension pair improves by `0.41%`. A reverse-order confirmation
measured `141.112 ms` for the strided module followed by `140.072 ms` for the
consecutive module, a `0.74%` reduction. rocprofv3 attributes the change to the
intended interaction-1 edge-phi kernel, which improves by `4.50%`; the
interaction-0 edge-phi kernel remains `7.909 ms`. The interaction-1 kernel
retains 70 VGPRs, no private segment, and no spills, while SGPR use falls from
72 to 61.

Energy, per-atom energies, forces, and stress are bitwise identical at 864,
9, and 17 atoms. hipRTC emits a 1,155,072-byte code object from 1,246,547 bytes
of shared HIP source. CUDA 13.1 NVRTC compiles the same generated program to a
2,197,024-byte `sm_80` cubin. The retained result is `5.66x` the MH0 median and
`2.26x` the acceptance limit. Raw profiler output is under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_edge_consecutive/`; the primary paired
records are `mh1_edge_consecutive_paired_old_5x10.json` and
`mh1_edge_consecutive_paired_new_5x10.json` in the temporary root.

### Dual-node message tiling

The four message-linear passes now use the same logical 16-node by 32-channel
tile retained for `linear2`: forward, reverse forward replay, reverse
recomputation, and transpose reverse. Each 256-thread block covers 16 nodes,
with every thread evaluating two node-target outputs while reusing the 32x32
learned-weight tile. Residual and product linears remain 8x32. Node-dependent
density normalization is instantiated separately for each logical node, so
both accumulators preserve the original ascending dot-product order and scale.

| Message schedule | Median (ms) | Range (ms) | Aggregate message linears (ms) | Status |
|---|---:|---:|---:|---|
| Uniform 8x32 messages | 139.919 | 139.754-140.338 | 23.431 | Previous milestone |
| Message and linear2 16x32 | 135.452 | 135.131-136.118 | 18.393 | Retained |

The `old-new-old-new` cached-module sequence measured old medians of `140.030`
and `139.919 ms`, and new medians of `135.048` and `135.452 ms`. The paired
reductions are `3.56%` and `3.19%`. rocprofv3 attributes `5.038 ms` to the
affected message family, a `21.5%` reduction. All affected dispatches halve
`grid_y` from 108 to 54. Forward and replay/recompute kernels use 96 VGPRs;
transpose kernels use 91. They use 6656 bytes of LDS and have no private
segment or spills.

Energy, per-atom energies, forces, and stress are bitwise identical to the
8x32 message schedule at 864, 9, and 17 atoms; the two smaller cases exercise
partial 16-node tiles. hipRTC emits a 1,174,272-byte code object from 1,265,623
bytes of shared HIP source. CUDA 13.1 NVRTC compiles the same generated program
to a 2,254,880-byte `sm_80` cubin. The retained result is `5.46x` the MH0
median and `2.19x` the acceptance limit. Raw profiler output is under
`/tmp/symmetrix-mh-pathway.bWSwqt/mh1_message_dual/`; the paired records are
`mh1_message_dual_paired_old_5x10.json`,
`mh1_message_dual_paired_new_5x10.json`, `mh1_message_dual_reverse_old_5x10.json`,
and `mh1_message_dual_reverse_new_5x10.json` in the temporary root.

### Full node-state retention

The generated GPU schedule now retains each layer's pre-gate and interaction
output for reverse. The explicit `recompute-v1` schedule remains available as
a backend-neutral low-memory control. No CUDA- or HIP-specific numerical path
is introduced.

| Node-state policy | Profile device time (ms) | Launches | M1/node reverse (ms) | Node workspace (bytes) | Status |
|---|---:|---:|---:|---:|---|
| Recompute | 137.402 | 167 | 38.727 | 226,374,912 | Control |
| Full retention | 118.357 | 125 | 19.840 | 350,237,952 | Retained |

The retained schedule removes 42 replay/recomputation launches and `18.887 ms`
from node reverse in the selected-region trace. Energy, per-atom energies,
forces, and stress are bitwise identical between policies at 9, 17, and 864
atoms.

### Compact fused edge reverse

Each interaction now combines its phi and harmonic reverse work into one
compact kernel. The four-edge owner and per-edge accumulation order are
unchanged, while the physical edge-reverse launch count falls from four to two.

| R1 reverse: edge | Pair 1 median (ms) | Pair 2 median (ms) | Status |
|---|---:|---:|---|
| Split phi/harmonic | 117.067 | 117.328 | Previous milestone |
| Compact fused | 113.856 | 113.896 | Retained |

The paired whole-step reductions are `2.74%` and `2.93%`. Full-array
qualification is bitwise exact at 9, 17, and 864 atoms.

### Batched compact-edge synchronization

The compact fused reverse now lets all four edge slots publish disjoint warp
partial slices before one block barrier, performs all four reductions, and
then uses one overwrite-prevention barrier. Arithmetic order, ABI, launch
geometry, and LDS layout are unchanged. The schedule identity is
`hybrid-path-tiled-v6` in both CUDA and HIP metadata.

| Schedule | Run 1 median (ms) | Run 2 median (ms) | Status |
|---|---:|---:|---|
| One barrier pair per edge slot | 113.650 | 113.721 | Previous milestone |
| Four producers per barrier pair | 112.204 | 112.596 | Retained |

The paired reductions are `1.27%` and `0.99%`. rocprofv3 device time fell from
`116.547` to `115.711 ms`; interaction 1 fell from `24.286` to `23.494 ms`,
while interaction 0 changed from `9.197` to `9.152 ms`. Static barrier counts
fell from 35 to 11 and from 85 to 25.

HSACO metadata is unchanged: interaction 0 uses 95 VGPRs, 94 SGPRs, and 10,000
bytes LDS; interaction 1 uses 91 VGPRs, 107 SGPRs, and 10,000 bytes LDS. Both
have zero private segment and zero VGPR/SGPR spills. The final v6 cache key is
`747c0187887b34a6dbe453ca4025660118ac9686df3ffb7fb6bb56226e0439e3`.
Two final-identity runs measured `112.574` and `112.324 ms/step`.

Energy, per-atom energies, forces, and stress are bitwise identical to the
pre-batching module at 9, 17, and 864 atoms. hipRTC emits a 1,254,072-byte
HSACO from 1,478,200 bytes of shared HIP source. CUDA 13.1 NVRTC compiles the
same generated program from 1,478,218 bytes of CUDA source to a 2,545,120-byte
`sm_80` cubin.

Receiver-adjoint caching through LDS was rejected after regressing the median
to `123.983 ms`. Interaction 1 grew to 123 VGPRs and introduced 16 SGPR spills.
The code and cache-specific test were removed; direct target-adjoint loads are
retained.

The final two-run mean is `112.449 ms`, or `4.54x` the `24.788 ms` MH0 median
and `1.81x` the `61.971 ms` acceptance limit. The remaining gap is
`50.478 ms`.

### 32x32 linear2 node tiling

The six `linear2` launch groups now cover 32 nodes by 32 channels with the
existing 256-thread block. Each physical node lane owns four independent node
accumulators, so the 32x32 weight tile is reused across four nodes. Message
kernels remain 16x32 and all other node linears remain 8x32. The schedule is
serialized as `tiled-8x32-message-16x32-linear2-32x32-v6` for both CUDA and
HIP, and the neutral loader validates 8-, 16-, and 32-node launch records.

| Pair | 16x32 linear2 (ms) | 32x32 linear2 (ms) | Reduction |
|---|---:|---:|---:|
| 1 | 112.177 | 109.857 | 2.07% |
| 2 | 112.493 | 110.033 | 2.19% |

The candidate mean is `109.945 ms`. A fresh rocprofv3 trace attributes the
gain to the intended family: aggregate `linear2` time fell from `15.585 ms`
(`7.705 ms` forward and `7.880 ms` transpose) to `12.648 ms` (`6.238 ms`
forward and `6.410 ms` transpose), a `2.937 ms` reduction. The complete trace
contains 123 launches and `111.924 ms` of device time.

All affected hipRTC kernels remain spill-free with no private segment. Forward
kernels use 94 VGPRs and transpose kernels use 93 VGPRs; LDS is 8320 bytes.
Energy, per-atom energies, forces, and stress are bitwise identical to the v6
16x32 module at 9, 17, 25, and 864 atoms. The 25-atom case covers a partial
32-node tile. hipRTC emits a 1,271,480-byte HSACO from 1,496,920 bytes of HIP
source. CUDA 13.1 NVRTC compiles 1,496,938 bytes of shared CUDA source to a
2,587,872-byte `sm_80` cubin. The hipRTC cache key is
`e2f078872cbfb25f42a5e1b72962528f54db5f76d5374070501b1f7ee5c5fe47`.

The current result is `4.44x` the MH0 median and `1.77x` the acceptance limit.
The remaining absolute gap is `47.974 ms`. The next optimization target is
block-cooperative interaction/source ownership with a bounded shared CSR edge
cache, expressed through the same backend-neutral renderer and launch plan.

### 32x32 message node tiling

Message forward, reverse replay, reverse recomputation, and transpose now use
the same 32-node by 32-channel tile as `linear2`. Residual and product linears
remain 8x32. Every affected launch halves `grid_y` from 54 to 27.

| Pair | 16x32 messages (ms) | 32x32 messages (ms) | Reduction |
|---|---:|---:|---:|
| 1 | 109.898 | 108.814 | 0.99% |
| 2 | 110.332 | 108.811 | 1.38% |

The candidate mean is `108.813 ms`. rocprofv3 attributes the saving to the
intended family: aggregate message time fell from `9.567 ms` (`4.479 ms`
forward and `5.088 ms` reverse) to `8.437 ms` (`3.803 ms` forward and
`4.634 ms` reverse), a `1.130 ms` reduction. Forward kernels use 94-95 VGPRs,
transpose kernels use 93 VGPRs, LDS is 8320 bytes, and all affected kernels
have zero private segment and zero spills.

Energy, per-atom energies, forces, and stress are bitwise identical to the
16x32 message module at 9, 17, 25, and 864 atoms on the canonical `matscipy`
graph. hipRTC emits a 1,317,816-byte HSACO from 1,535,088 bytes of shared HIP
source. CUDA 13.1 NVRTC compiles 1,535,106 bytes of shared CUDA source to a
2,698,208-byte `sm_80` cubin. The cache key is
`0748a35569a6d50eb484d92ee908b2289b0628ac6d60e9444bf52c3b87d8c712`.

Full-row CSR96 caching was rejected after regressing `110.109 ms` to
`121.414 ms`; staging 64 phi values plus harmonics and indices consumed 31,872
bytes of LDS. Partition-owner fusion was also rejected: its apparent paired
gain was not present in the interaction/source profile. Both code changes were
removed before retaining message32.

The current result is `4.39x` the MH0 median and `1.76x` the acceptance limit,
leaving `46.842 ms`. Remaining work returns to compact edge reverse,
conditioner reverse, and the residual/product node families.

## Local HIP backend-policy qualification

Date: 2026-08-22. Base revision:
`6233ae8245a9d4f02a4b7c70ebcb70547533d8ff`. This follow-up used the direct
generated hipRTC path on the Radeon 8060S `gfx1151` target in FP32. The fixed
workload has 864 atoms, a 6.0 A model cutoff, zero neighbor-list skin, a 6.0 A
effective cutoff, and 78,624 directed edges.

The shared CUDA/HIP policy had two local HIP regressions. First, HIP reports
20 WGP-like units through `hipDeviceProp.multiProcessorCount`, so the shared
four-block multiplier launched only 80 persistent blocks. Historical qualified
traces used 160. Second, CUDA schedule `hybrid-path-tiled-v7` keeps four learned
edge weights live together. On HIP this raises compact-edge register pressure
without recovering enough weight-load time. The retained backend policy is:

- HIP: eight persistent blocks per reported compute unit and
  `hybrid-path-tiled-v6` compact edge reverse.
- CUDA: unchanged at four persistent blocks per SM and
  `hybrid-path-tiled-v7`. No CUDA execution was performed in this follow-up.

| HIP policy | Persistent blocks | Median (us/atom) | Median (ms/step) | Change |
|---|---:|---:|---:|---:|
| v7 control | 80 | 171.432 | 148.117 | Control |
| v6 only | 80 | 160.824 | 138.952 | -6.19% |
| 160-block grid only | 160 | 142.495 | 123.115 | -16.88% |
| Retained v6 plus 160 blocks | 160 | 126.813 | 109.567 | -26.03% |

The retained timing is a fresh five-warmup, ten-step run from an isolated HIP
extension. Energy and force summary validation passed, the execution backend
was `generated_hip_v4`, and fallback count remained zero. The generated module
reproduced qualified cache key
`870b71972c9074ea4dad281fedcdf0d761d01c81a2a0b360d9f66e748722ea4f`.

A one-step selected-region rocprofv3 trace contained 336 launches and
`112.566 ms` of device time (`130.285 us/atom` under profiling):

Stage labels below follow the standard MACE vocabulary. `M1/node` is used
instead of plain `M1` because each generated MH-1 node program also fuses H2,
residual, gate, density, and readout work. Internal generated kernel names and
profiler category IDs retain their existing names for ABI and report-schema
compatibility.

| Stage | Device ms | us/atom | Device share | Representative occupancy |
|---|---:|---:|---:|---:|
| R1 forward | 14.124 | 16.347 | 12.55% | 44.53-45.46% |
| R1 reverse: source | 16.409 | 18.992 | 14.58% | 42.04-45.87% |
| R1 reverse: edge (compact fused) | 33.035 | 38.235 | 29.35% | 64.62-65.31% |
| R1 edge conditioning forward | 2.480 | 2.870 | 2.20% | 45.62-45.72% |
| R1 edge conditioning reverse | 6.343 | 7.342 | 5.64% | 47.32-48.10% |
| M1/node forward | 16.389 | 18.969 | 14.56% | 72-93% for hot tiled kernels |
| M1/node reverse | 19.660 | 22.755 | 17.47% | Mixed tiled and small launches |
| Geometry, assembly, readout, runtime | 4.126 | 4.776 | 3.66% | Mixed |

The graph-wide interaction, source, and conditioning launches use a grid of
160 blocks (`20,480` threads at 128 threads per block). R1 edge reverse
uses 640 blocks (`81,920` threads) through the existing path-tiled block
multiplier. Code-object metadata reports 95/91 VGPR, 94/107 SGPR, and 10,240 B
LDS for compact interactions 0/1, with zero private segment and zero
VGPR/SGPR spills. rocprofv3 rounds their dispatch allocation to 96 VGPR.

The CUDA v7 learned-weight reuse is therefore not transferred to HIP. The
portable part of the standard-MACE optimization is persistent ownership and
enough independent blocks to occupy the device; the register-heavy reuse
microkernel remains backend-specific. The next untested compact-edge direction
is batching harmonic and cutoff reductions across each four-path group. It can
reduce static barriers from 11 to 4 and 25 to 8 for interactions 0/1, with an
estimated 1.5 KiB additional LDS, but requires a separate correctness and
whole-step qualification.
