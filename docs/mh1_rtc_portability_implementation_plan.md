# MACE-MH-1 RTC Portability Implementation Plan

Status: in progress (phases 0-5 implemented; phase 6 optimization is active)

Plan date: 2026-08-09

Target device: Radeon 8060S (`gfx1151`, wave32)

## 1. Objective

Make the optimized MACE-MH-1 generated evaluator available through both NVRTC
and hipRTC without maintaining separate CUDA and HIP numerical kernels. The
existing MH1 program, schedule, and launch plan become backend-neutral. Thin
CUDA Driver and HIP Module adapters compile, load, and launch the same generated
program through backend dialects.

The performance acceptance criterion is an MH1 median step time no greater
than 2.5 times the contemporary MH0 generated-RTC median on the same device,
model head, structure, precision, graph, and timing scope. With the retained
`gfx1151` baseline, this is currently:

```text
MH0 hipRTC median             24.788 ms/step
MH1 acceptance limit         61.971 ms/step
Current generic MH1        1950.469 ms/step
Required current speedup       31.47x
```

The ratio is the durable gate. `61.971 ms/step` is a baseline indicator and
must be recomputed when the MH0 reference, hardware, runtime, or benchmark
contract changes.

This plan refreshes the MH1 portions of `docs/hip_migration_plan.md`. It also
follows the RTC-only compiler policy in
`docs/rtc_primary_transition_plan.md`: runtime generation uses NVRTC or hipRTC
and fails closed; it does not retry with NVCC or hipcc.

## Implementation Snapshot

As of 2026-08-08, the neutral target/program types, CUDA/HIP dialect renderer,
hipRTC compiler path, neutral module facade, HIP Module API adapter, calculator
dispatch, launch-plan validation, and 864-atom energy-and-force execution are
implemented. MH1 runtime generation accepts only NVRTC or hipRTC; explicit
NVCC/hipcc artifact selection errors instead of changing artifact formats.

The first `gfx1151` bring-up measured `429.631 ms/step` with the correctness-
first persistent node schedule. It produced `-6423.654872 eV` and a
`3.168528 eV/A` force norm, compared with the generic baseline's
`-6423.653261 eV` and `3.168540 eV/A`. The shared generated interaction and
conditioning path with generic nodes measured `400.013 ms/step` and matched
the baseline more tightly. An early 8x32 tiled-node experiment measured about
`267.448 ms/step`, but appeared to fail physics and was withheld until the
launch-plan defect described below was diagnosed.

Phase 6 profiling found that the original MH1 trace spent `121.843 ms` in the
layer-1 edge-phi reverse kernel and `77.2 ms` in the monolithic layer-1 node
reverse kernel. The shared renderer now selects path-tiled edge-phi and
graph-wide 8-node by 32-channel node schedules for both forward and reverse.
The persistent launch target is eight blocks per compute unit; a measured
16-block probe regressed performance and was rejected.

An earlier tiled-node physics failure was a launch-plan construction defect,
not a numerical-kernel defect. Launch records were indexed by each generated
launch fragment, while residual and message fragments were joined before the
lookup. The resulting module plan omitted both launches in forward, reverse
reconstruction, and reverse recomputation. The generator now records each
fragment before composition, and CUDA and HIP consume the same complete ordered
plan.

On the same 864-atom graph, the complete tiled-node milestone measured
`160.400 ms/step` with a `160.023-161.162 ms` range. Batching four independent
edge-phi path adjoints in shared memory subsequently reduced the retained
five-warmup, ten-sample median to `159.648 ms/step`, with a
`159.346-160.017 ms` range. A clock-matched cached-module pair measured
`160.187 ms` before batching and `159.279 ms` after batching. The layer-1
edge-phi kernel fell from `24.797 ms` to `23.520 ms` in four-evaluation traces.
The result remains `-6423.654872 eV` with a `3.168526 eV/A` force norm, and the
batching preserves energy, forces, and stress exactly relative to the previous
tiled schedule.

The next retained schedule assigns four strided edges to each block and reuses
every path linear-weight load across four independent accumulators. The
five-warmup, ten-sample median is `150.667 ms/step`, with a
`150.292-151.084 ms` range. In a later same-clock cached-module pair, the
single-edge schedule measured `158.782 ms` and the four-edge schedule measured
`150.450 ms`. The layer-1 edge-phi kernel fell from `23.520 ms` to `15.596 ms`.
Energy, per-atom energies, forces, and stress are bitwise identical to the
single-edge path-batched result. The retained kernel uses 70 VGPRs and no
private segment or spills on `gfx1151`; CUDA 13.1 NVRTC compiles the same
source to a `1,992,480`-byte `sm_80` cubin.

The mixed node schedule subsequently assigns two logical nodes to each
physical lane for only the `linear2` forward, reverse replay, and transpose
kernels. The 256-thread block therefore covers 16 nodes by 32 channels while
all other linears remain 8x32. A five-warmup, ten-sample run measured
`147.329 ms/step`, with a `146.575-147.740 ms` range. A same-extension
cached-module pair measured `150.439 ms` for uniform 8x32 and `143.588 ms` for
the mixed schedule. Aggregate `linear2` device time fell from `30.952 ms` to
`23.263 ms`.

All 24 affected kernels have zero scratch and spills; forward/replay kernels
use 96 VGPRs and transpose kernels use 91. Energy, per-atom energies, forces,
and stress are bitwise identical at 864 atoms and at the 9- and 17-atom tail
cases. Launch-plan version 2 carries per-launch tile geometry shared by CUDA
and HIP, while the neutral loader retains explicit version-1 8x32 support.
CUDA 13.1 NVRTC compiles the shared source to a `2,019,872`-byte `sm_80` cubin.

The next retained schedule bounds GPU conditioner reverse state to two reusable
forward buffers and one current-adjoint buffer. It recomputes canonical
`linear, LayerNorm, SiLU` prefixes inside the existing kernel and specializes
the terminal scalar density transpose, without adding launches or global
workspace. Noncanonical programs retain the generic reverse, the host path is
unchanged, and CUDA and HIP consume the same generated implementation.

A five-warmup, ten-sample run measured `140.512 ms/step`, with a
`140.238-140.994 ms` range. A same-clock cached-module pair measured
`143.251 ms` for the preceding full-state reverse and `140.677 ms` for bounded
recomputation. Combined conditioner reverse time fell from `9.609 ms` to
`7.026 ms`. Each reverse kernel's private frame fell from 3424 to 976 bytes and
its VGPR spill count from 708 to 80. Energy, per-atom energies, forces, and
stress are bitwise identical at 864, 9, and 17 atoms. CUDA 13.1 NVRTC compiles
the shared source to a `2,197,408`-byte `sm_80` cubin.

The path-tiled edge-phi schedule now maps each block's four slots to consecutive
edges and launches `ceil(samples / 4)` logical groups. On the receiver-sorted
qualification graph, `96.70%` of those groups contain a single target. A
five-warmup, ten-sample run measured `140.276 ms/step`, with a
`139.597-140.647 ms` range. The first same-extension pair measured
`140.854 ms` for grid-strided ownership and `140.276 ms` for consecutive
ownership; a reverse-order confirmation measured `141.112 ms` and
`140.072 ms`, respectively. The interaction-1 edge-phi kernel fell from
`15.274 ms` to `14.586 ms` while interaction 0 was unchanged. Its 70 VGPRs and
zero private bytes/spills are unchanged, and SGPR use falls from 72 to 61.
Energy, per-atom energies, forces, and stress remain bitwise identical at 864,
9, and 17 atoms. CUDA 13.1 NVRTC compiles the shared program to a
`2,197,024`-byte `sm_80` cubin.

The node schedule next extends the retained logical 16-node by 32-channel
weight reuse from `linear2` to message forward, reverse replay, reverse
recomputation, and transpose reverse. Residual and product linears remain
8x32. The generator instantiates density normalization separately for each
logical node, retaining each node's original dot-product and scaling order.

An `old-new-old-new` cached-module sequence measured `140.030`, `135.048`,
`139.919`, and `135.452 ms/step`, respectively. The two paired reductions are
`3.56%` and `3.19%`. Aggregate message-linear time fell from `23.431 ms` to
`18.393 ms`, a `21.5%` reduction. The affected dispatches halve `grid_y` from
108 to 54, use 6656 bytes of LDS, and remain free of private memory and spills.
Forward/replay/recompute kernels use 96 VGPRs and transpose kernels use 91.
Energy, per-atom energies, forces, and stress remain bitwise identical at 864,
9, and 17 atoms. CUDA 13.1 NVRTC compiles the shared program to a
`2,254,880`-byte `sm_80` cubin.

Full node-state retention is now the default generated GPU schedule. Each
layer retains its pre-gate and interaction-output values for reverse, while
`recompute-v1` remains an explicit backend-neutral low-memory schedule. The
selected-region trace fell from `137.402 ms` and 167 launches to `118.357 ms`
and 125 launches. M1/node reverse fell from `38.727 ms` and 78 launches to
`19.840 ms` and 36 launches. The retained node workspace is `350,237,952`
bytes, versus `226,374,912` bytes for recomputation.

The two edge-reverse launches now use one compact fused kernel per interaction
instead of separate phi and harmonic kernels. Two same-clock split/compact
pairs measured `117.067/113.856 ms` and `117.328/113.896 ms`, reductions of
`2.74%` and `2.93%`. The compact kernel preserves per-edge accumulation order
and remains one shared numerical body for NVRTC and hipRTC.

Schedule `hybrid-path-tiled-v6` batches all four compact-edge producer slots
behind one block barrier, performs all four reductions, and then executes one
overwrite-prevention barrier. An old/new/old/new sequence measured `113.650`,
`112.204`, `113.721`, and `112.596 ms/step`; the paired reductions are `1.27%`
and `0.99%`. rocprofv3 device time fell from `116.547` to `115.711 ms`, with
interaction 1 falling from `24.286` to `23.494 ms`. Static barrier counts fell
from 35 to 11 for interaction 0 and from 85 to 25 for interaction 1.

Final-identity cached runs measured `112.574` and `112.324 ms/step`, a
`112.449 ms` mean. The compact kernels use 95/91 VGPRs, 94/107 SGPRs, and
10,000 bytes LDS for interactions 0/1, with no private segment or spills.
Energy, per-atom energies, forces, and stress are bitwise identical at 9, 17,
and 864 atoms. hipRTC emits a 1,254,072-byte HSACO from 1,478,200 bytes of HIP
source. CUDA 13.1 NVRTC compiles the same generated program from 1,478,218
bytes of CUDA source to a 2,545,120-byte `sm_80` cubin.

The retained node schedule now expands only the six `linear2` launch groups
from 16x32 to 32x32. Message kernels remain 16x32 and all other node linears
remain 8x32. Two paired old/new runs measured `112.177/109.857 ms` and
`112.493/110.033 ms`, reductions of `2.07%` and `2.19%`. rocprofv3 attributes
the saving to `linear2`: its aggregate forward and transpose time fell from
`15.585 ms` to `12.648 ms`, a `2.937 ms` reduction. The affected HIP kernels
use 94 VGPRs forward and 93 VGPRs transpose, 8320 bytes LDS, no private
segment, and no spills.

Energy, per-atom energies, forces, and stress are bitwise identical to the
16x32 schedule at 9, 17, 25, and 864 atoms. The 25-atom case explicitly covers
a partial 32-node tile. hipRTC emits a 1,271,480-byte HSACO from 1,496,920
bytes of shared HIP source. CUDA 13.1 NVRTC compiles the same program from
1,496,938 bytes of CUDA source to a 2,587,872-byte `sm_80` cubin. The retained
hipRTC cache key is
`e2f078872cbfb25f42a5e1b72962528f54db5f76d5374070501b1f7ee5c5fe47`.

The message forward, reverse replay, recomputation, and transpose groups now
also use 32x32 tiles. Two old/new pairs measured `109.898/108.814 ms` and
`110.332/108.811 ms`. Aggregate message time fell from `9.567 ms` to
`8.437 ms`, a `1.130 ms` reduction. The affected kernels use 94-95 VGPRs
forward and 93 VGPRs transpose, 8320 bytes LDS, no private segment, and no
spills. Full-array results remain bitwise exact at 9, 17, 25, and 864 atoms.
CUDA 13.1 NVRTC compiles the shared 1,535,106-byte source to a 2,698,208-byte
`sm_80` cubin.

Two block-cooperative interaction/source experiments were rejected. Caching a
complete 96-edge CSR row staged 64 phi values, 16 harmonics, cutoff, and indices
per edge in 31,872 bytes of LDS and regressed `110.109 ms` to `121.414 ms`.
Fusing all partitions for one owner appeared to save about `0.7 ms` in paired
timing, but a four-evaluation trace showed no attributable interaction/source
reduction. Both experiments were removed.

A receiver-adjoint LDS cache was rejected. It regressed the median to
`123.983 ms`, raised interaction-1 VGPR use from 91 to 123, and introduced 16
SGPR spills. Register caching and four-edge weight hoisting were also rejected
because their longer live ranges caused analogous resource regressions.

Larger batches were rejected: five-path and ten-path batches measured
`162.293 ms` and `161.125 ms`. The existing three-partition shared forward
policy measured `160.620 ms`, slower than the six-partition automatic policy.
Scoping harmonic adjoints by input block removed the layer-1 spill and reduced
VGPR use from 192 to 102, but two clock-matched pairs showed `0.28-0.59 ms`
full-step regressions and the harmonic kernel itself slowed by about `0.45 ms`;
that experiment was reverted.

The current paired `108.813 ms` mean is about `4.39x` the retained `24.788 ms`
MH0 median and remains `1.76x` the `61.971 ms` acceptance limit, leaving
`46.842 ms`. Until the
performance gate and full physics qualification pass, HIP MH1 is available
only through explicit `direct_jit="required"`; automatic promotion remains
blocked and explicit `fallback` remains on generic Kokkos.

### Matched MH0/MH1 profile decision

A fresh one-step diagnostic at commit `475d16a` compared PyTorch MH0/MH1 and
Symmetrix MH0/MH1 on the same 864-atom, 78,624-directed-edge AlN workload.
All four cells share canonical physical-graph SHA-256
`a6990939929e747e577bbd8b65c8954a13f712b972e40cad9869f78f54a521b7`.
PyTorch used eager e3nn because no compatible ROCm cuEquivariance package was
installed in the profiling environment.
Its Kineto device totals were `541.696 ms` and `2056.546 ms` (`3.80x`, 1,064
and 1,473 launches). The hipRTC rocprof totals were `24.735 ms` and
`138.098 ms` (`5.58x`, 40 and 167 launches).

The two runtimes do not grow in the same phase. PyTorch forward time grows by
`1378.141 ms`, or `91.0%` of its MH1-MH0 delta. Product-basis modules account
for `1336.888 ms` of that increase, including `1232.859 ms` of additional
`aten::copy_` device time. Its autograd reverse grows by only `136.709 ms`
(`1.38x`). Symmetrix has already removed that upstream product-copy bottleneck:
its forward delta is `28.507 ms`, while reverse grows by `83.764 ms` and
accounts for `73.9%` of its total delta. Reverse launch count grows from 13 to
95.

Selective node-state retention has now removed the 42 reverse replay and
recomputation launches identified by this comparison. The retained v6 trace
contains 123 launches and `115.711 ms` of profiled device time. Its largest
remaining families are node forward/reverse (`37.979 ms`), compact edge
reverse (`32.646 ms`), source reverse (`16.437 ms`), interaction forward
(`14.266 ms`), and conditioning (`9.481 ms`).

The 32x32 `linear2` and message experiments are retained after passing their
performance, resource, cross-compiler, and partial-tail gates. Full-row CSR
caching and partition-owner fusion are rejected by direct measurement. The
next bounded work must target compact edge reverse, conditioner reverse, or
the remaining 8x32 residual/product node families without materializing the
full 64-component phi row in LDS. It remains schedule or program logic in the
common NVRTC/hipRTC renderer. Detailed profiling protocol, limitations, and
acceptance checks are in `benchmarks/mh1_profile_comparison_864.md`.

## 2. Architectural Contract

The implementation must preserve these boundaries:

1. `MH1Program` is the single source of truth for algebra, ownership, phase
   ordering, buffer semantics, and kernel bodies.
2. `MH1GpuTarget`, `MH1KernelSchedule`, and `MH1LaunchPlan` contain hardware
   facts and tuning decisions. Target-specific schedule values are allowed;
   duplicated numerical source is not.
3. `MH1GpuDialect` owns only source-language and device-primitive differences:
   qualifiers, thread identifiers, barriers, subgroup masks and shuffles,
   atomics, symbol declarations, and ABI packet aliases.
4. Runtime compilation produces a directly loadable module: cubin through
   NVRTC or HSACO through hipRTC. There is no text translation, source slicing,
   generated host wrapper, or runtime AOT compiler fallback.
5. Native runtime code uses one neutral module/evaluator facade backed by thin
   CUDA Driver and HIP Module API adapters. The abstraction must not add
   virtual dispatch to launches or device-side work.
6. Backend-specific optimization belongs in schedule data or a dialect
   primitive. A proposed optimization that requires a second CUDA or HIP
   numerical renderer is out of scope.
7. Initial promotion is FP32. The generic HIP path remains available for
   unsupported precision or model contracts only under explicit `fallback` or
   `off` policy.
8. `required` mode errors on an unsupported model, failed compilation, invalid
   module, missing symbol, or launch failure. It never silently falls back.
9. Cache identity includes backend, architecture and target features, native
   subgroup width, logical tile width, schedule, compiler/runtime identity,
   precision, model contract, and ABI/schema versions.
10. CUDA remains a first-class consumer of the shared program and must pass
    correctness, resource, and performance requalification on NVIDIA hardware.

## 3. Qualification Contract

The fixed qualification case is the official matched MACE-MH-0 and MACE-MH-1
`omat_pbe` checkpoints on the 864-atom wurtzite AlN structure described in
`benchmarks/mh1_hip_baseline_864.md`. Both runs must use the same prepared
graph, FP32 precision, device, process class, synchronization boundary, and
energy-and-force timing scope.

Final performance evidence requires at least 20 warmups and 20 measured steps
in each of at least three fresh processes. Process order must be balanced or
randomized. Retain raw samples and report median, p90, a confidence interval,
compile/load time, peak memory, workspace size, artifact size, launch count,
and per-kernel resource data. Numerical results are checked before and after
timing.

The validator must reject records unless:

- MH1 identifies the official MH1 checkpoint and generated RTC path;
- MH0 identifies the matched official MH0 checkpoint and generated RTC path;
- HIP records name `hiprtc`, and CUDA records name `nvrtc`;
- neither record reports generic evaluation, fallback, NVCC, or hipcc;
- graph size, edge count, precision, head, timing scope, backend, and device
  identity match;
- the upper confidence bound for the median MH1/MH0 ratio is at most 2.5.

The retained five-warmup, ten-sample result remains a baseline indicator, not
the final promotion measurement.

## 4. Implementation Sequence

Each phase is a cohesive pull request. Correctness and runtime integration are
completed before AMD tuning, and compatibility aliases are retained until
both backends are qualified.

### Phase 0: Freeze the Baseline and Validator

Scope:

- Land the official MH checkpoint extraction compatibility fix and its focused
  regression tests.
- Retain the 864-atom AlN baseline, checkpoint hashes, exact path assertions,
  and PyTorch reference controls.
- Add a structured fresh-process MH1/MH0 RTC benchmark driver and ratio
  validator rather than adapting the ordinary-R1 baseline/candidate validator
  to incompatible semantics.
- Record compiler backend, generated-path identity, fallback state, schedule,
  graph identity, numerical checks, samples, memory, and launch metadata.

Exit criteria:

- A single command can produce independently valid MH0 and MH1 records.
- The validator rejects a wrong checkpoint, graph, compiler, generic path, or
  fallback result.
- The 2.5 ratio gate and current `61.971 ms/step` indicator are documented.

### Phase 1: Neutral MH1 Program, Target, Schedule, and Launch Plan

Scope:

- Generalize `MH1CudaSchedule` to `MH1KernelSchedule` and introduce neutral
  launch records without changing generated CUDA behavior.
- Replace CUDA-named internal policy identities with backend-neutral identities
  whose values are selected from `MH1GpuTarget` and schedule data.
- Make backend, architecture, target features, native subgroup width, logical
  group widths, geometry, and schedule part of serialization and cache keys.
- Keep deprecated CUDA names as thin compatibility aliases for one migration
  cycle.

Exit criteria:

- Existing NVRTC source, symbols, launch order, geometry, workspace, and
  resource expectations are byte-identical where practical and otherwise
  proven semantically identical.
- Program and launch-plan serialization is deterministic.
- No HIP production dispatch is enabled in this phase.

### Phase 2: Shared Renderer with CUDA and HIP Dialects

Scope:

- Extract shared renderers for persistent node work, conditioning, tensor
  products, forward phases, reverse phases, and force accumulation.
- Implement CUDA and HIP dialect leaves for syntax, module symbols, atomics,
  barriers, and subgroup operations.
- Model native wave width separately from logical 16- and 32-lane groups.
  Reductions must handle wave32, wave64, inactive lanes, partial tiles, and odd
  group sizes without assuming a CUDA warp.
- Add compiler-only tests using `/usr/local/cuda` NVRTC and the installed
  hipRTC stack, including every generated building block and full program.

Exit criteria:

- The shared numerical renderer contains no CUDA- or HIP-specific token.
- Full official-model programs compile through NVRTC and hipRTC.
- HIP generation does not depend on CUDA headers, libraries, or source
  translation.
- CUDA source and launch-plan parity tests remain green.

### Phase 3: Neutral Device-Module ABI and Native Loader

Scope:

- Introduce a backend-neutral MH1 device-module ABI for argument packets,
  symbols, launch records, workspace descriptors, and schema validation.
- Add an `MH1DeviceModule` facade with compile-time CUDA Driver and HIP Module
  API adapters. Reuse ordinary-R1 module lifecycle and validation mechanics
  where their contracts match.
- Replace CUDA-specific evaluator ownership and readiness properties with
  neutral equivalents while preserving compatibility properties temporarily.
- Define context, device, stream, event, and unload ownership explicitly. A
  module cannot unload while work using it is outstanding.
- Test missing symbols, corrupt plans, ABI mismatch, backend mismatch, device
  mismatch, repeated construction, cache reuse, and teardown.

Exit criteria:

- Sentinel modules load and launch through CUDA Driver and HIP Module APIs.
- The launch facade introduces no per-launch virtual dispatch.
- Cross-backend and stale-cache artifacts fail before launch.
- CUDA lifecycle and launch behavior do not change.

### Phase 4: hipRTC Integration and Correctness-First Forward Path

Scope:

- Allow MH1 calculator dispatch to construct a HIP target, schedule, program,
  artifact, and module through hipRTC.
- Make explicit `required` resolve only to hipRTC on HIP; compilation or load
  failure must not invoke hipcc. Keep `automatic` ineligible and fail closed
  until the promotion gates pass.
- Bring up forward energy in isolation before exposing an incomplete
  energy-and-force production path.
- Test the official checkpoint and generalized supported shapes on small, 256-
  atom, and 864-atom cases, including zero/one edge, non-divisible dimensions,
  species reorderings, and repeated/resized calls.

Exit criteria:

- Forward atomic and total energies agree with the generic HIP and CPU oracles
  within the declared FP32 tolerance.
- Metadata proves hipRTC, the generated module path, target identity, schedule,
  and no fallback.
- Unsupported contracts fail closed in `required` mode.

### Phase 5: Full Reverse, Conditioning, Forces, and Stress

Scope:

- Complete source, edge, harmonic, cutoff, conditioning, tensor-product, and
  node adjoints in the shared program.
- Integrate forces and stress while preserving phase ordering, prepared-graph
  ownership, one evaluator stream, and the neutral ABI.
- Add finite-difference checks, generic HIP and CPU parity, repeated evaluation,
  resizing, multiple species, zero-edge, lifecycle, and failure-path tests.
- Run available address and leak diagnostics and retain their tool/device
  limitations with the results.

Exit criteria:

- Energy, per-atom energy, forces, and stress satisfy the existing MH1
  numerical contracts.
- Exact generated launch counters and backend metadata prove the intended path.
- Warm evaluation adds no host transfer, global fence, or hidden fallback.
- The complete path is usable through explicit `direct_jit=required`; automatic
  promotion may wait for the performance phase.

### Phase 6: AMD Schedule Tuning Against the 2.5x Gate

Scope:

- Start from the current CUDA v13 algorithm expressed through shared schedule
  data; profile before changing it.
- Record time by phase and identify occupancy, register, LDS, instruction,
  memory-traffic, and launch-overhead limits.
- Tune only target data and portable mechanisms: block size, logical subgroup
  width, persistent blocks per CU, channel and edge tiling, partitions, LDS
  padding, register budgets, and justified kernel fusion.
- Re-run numerical checks for every accepted schedule change and retain 256-
  atom and larger-size evidence to detect overfitting to 864 atoms.

Exit criteria:

- On `gfx1151`, the final fresh-process upper confidence bound for the median
  MH1/MH0 ratio is at most 2.5. With the retained MH0 baseline, the point target
  is at most `61.971 ms/step`.
- The record proves hipRTC and generated MH1 execution with no fallback.
- p90 is stable, launch count is explained, and no accepted change introduces
  spills, unexpected stack/local memory, hidden transfers, or more than 5%
  unexplained workspace/peak-memory growth.
- All optimized numerical bodies remain shared with CUDA.

### Phase 7: CUDA Requalification and Automatic Promotion

Scope:

- On NVIDIA hardware, force NVRTC and benchmark the official MH1 864-atom case
  against a same-device pre-refactor baseline. The historical RTX 5090
  `27.247 ms/step` result is supporting evidence only.
- Apply the existing portability refactor gates: upper 95% confidence limit no
  worse than 2%, absolute rejection above 5% median or 10% p90, and no
  unexplained resource, launch, memory, or transfer regression.
- Run the full shared correctness matrix on both CUDA and HIP.
- Promote MH1 hipRTC to automatic selection only after the AMD performance and
  CUDA non-regression gates pass. Update required-test manifests and support
  documentation in the same change.
- Remove deprecated CUDA-only internal aliases only after all downstream code
  uses the neutral interfaces.

Exit criteria:

- Required CUDA tests prove NVRTC module launch; required HIP tests prove
  hipRTC module launch.
- CUDA correctness and performance gates pass on real NVIDIA hardware.
- HIP satisfies the 2.5x MH0 ratio gate on the named AMD target.
- Automatic mode selects the generated RTC path, while failures remain visible
  and never retry an AOT compiler.

## 5. Test Matrix

| Layer | Required coverage |
| --- | --- |
| Program/schema | Deterministic program, target, schedule, launch plan, cache identity, compatibility aliases |
| Renderer | CUDA/HIP dialect tokens, subgroup edge cases, full-program NVRTC and hipRTC compilation |
| Module runtime | Load, symbols, launch, stream ordering, device/context ownership, cache, teardown, corrupt artifacts |
| Physics | Official and generalized MH1; energy, atomic energy, forces, stress; finite differences; CPU/generic HIP oracles |
| Shapes | Zero/one edge, partial tiles, non-divisible channels, reordered species, resize/repeat, small/256/864/larger atoms |
| Policy | `required`, `automatic`, `fallback`, `off`; unsupported contracts; compiler/load/launch failures; no AOT retry |
| Performance | Same-device MH1/MH0 ratio, 3 fresh processes, 20 warmups and samples, median/p90/CI/resources/memory/launches |
| CUDA guard | NVRTC-required runtime, official checkpoint, source/launch parity, real-device non-regression |

## 6. Review and Rollback Rules

- Do not combine neutralization, HIP correctness, and performance tuning in one
  review. Each phase must leave CUDA usable and tests diagnosable.
- A backend-specific schedule is acceptable only when it serializes explicitly
  and is covered by target-specific tests.
- A backend-specific kernel body requires an architecture review and evidence
  that it cannot be represented as shared program logic plus a dialect
  primitive. The default decision is to reject the fork.
- Bump cache/ABI schemas whenever source meaning, launch records, packets, or
  schedule interpretation changes. Never attempt to load an ambiguous old
  artifact.
- Before automatic promotion, rollback is policy-based: use explicit generic
  `fallback` or `off`. Do not restore NVCC/hipcc runtime fallback.
- After automatic promotion, a regression may demote the affected target to an
  explicit generated mode while retaining `required` for diagnosis. The shared
  program and neutral module architecture remain in place.

## 7. Completion Definition

The work is complete only when the official MH1 model runs energy and forces
from the same generated numerical program through NVRTC and hipRTC; both paths
are proven by runtime metadata and required tests; HIP meets the 2.5x matched
MH0 ratio; CUDA passes its non-regression gate; and no runtime path invokes
NVCC or hipcc or maintains a second vendor-specific MH1 numerical renderer.
