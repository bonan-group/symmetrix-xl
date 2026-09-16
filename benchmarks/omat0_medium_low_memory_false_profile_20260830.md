# OMAT-0 medium `low_memory=False` CUDA stage profile

## Scope

This record attributes one steady-state OMAT-0-medium FP32 energy, force, and
stress evaluation on the RTX 5090. It profiles the committed
`low_memory=False` speed baseline after three warmups. Nsight Systems captures
the `symmetrix_full_evaluator::direct` NVTX range, while a separate five-call
control establishes unprofiled timing stability.

The timed evaluator range computes node energies and directed pair forces. The
final node-force and stress reductions performed while collecting Python
results occur after that range; they are reported separately below so they are
not silently charged to a model stage.

## Provenance and workload

| Item | Value |
|---|---|
| Source commit | `3d1fdb59fc6ad591c227beebd1bd615b526d1764` |
| Native extension SHA-256 | `c8c9b30e6b6d704c3183689f3688ce109a58fddfd4b7519f4113cbcefb8c351f` |
| Model SHA-256 | `af3b41b1e0de93dc1b8ba946f71ae3cb84c16b7c7a48840529cddd32382380a7` |
| GPU | NVIDIA GeForce RTX 5090, SM 12.0, 32,607 MiB |
| Driver | 610.43.02 |
| Profilers | Nsight Systems 2026.1.3; Nsight Compute 2026.2.1 |
| Precision and properties | FP32; energy, forces, stress |
| Atoms | 131,072 (`AlN` wurtzite `32 x 32 x 32`) |
| Directed edges | 14,286,848 |
| Cutoff / skin / effective cutoff | 6.0 / 0.5 / 6.5 A |
| Streamed-edge mode | `direct` |
| Public memory policy | `low_memory=False`, selected `disabled` |

The selected executors were generated R1 `jit_all` forward and `jit` reverse,
standard R0 `v2_edge16`, standard M0, and M1 recomputation with a 32-channel
tile. MH-0 state, readout, Phi1, and harmonic gradients were retained, and
geometry used Cartesian float64 vectors and radii. The evaluator reported zero
fallbacks.

## Timing boundary

The unprofiled control used three warmups and five measured calls:

| Metric | Result |
|---|---:|
| Median evaluator time | 514.900 ms |
| Median evaluator time per atom | 3.928 us/atom |
| Measured range | 513.777-515.731 ms |
| Instrumented evaluator range | 512.632 ms |
| Instrumented time per atom | 3.911 us/atom |

The instrumented result is 0.44% below the unprofiled median and inside the
same steady-state regime. The native evaluator timer reported 483.734 ms. The
28.898 ms difference from the Python-visible instrumented call is explained by
28.722 ms of geometry H2D transfers plus about 0.176 ms of launch/API overhead.

## Stage breakdown

The range contains 29 kernels totaling 483.252 ms, two geometry transfers
totaling 457,179,136 bytes and 28.722 ms, five asynchronous zeroing operations
totaling 813,694,976 bytes and 0.411 ms, and 0.248 ms of remaining gaps. GPU
activity therefore accounts for 99.95% of the evaluator range.

| Stage | ms/inference | Evaluator share | Kernel share | Launches |
|---|---:|---:|---:|---:|
| Geometry H2D: Cartesian vectors and radii | 28.722 | 5.60% | - | 2 copies |
| Workspace zeroing | 0.411 | 0.08% | - | 5 memsets |
| ZBL | 3.425 | 0.67% | 0.71% | 1 |
| Spherical harmonics | 18.647 | 3.64% | 3.86% | 4 |
| R0 forward | 35.847 | 6.99% | 7.42% | 2 |
| M0 forward | 2.617 | 0.51% | 0.54% | 1 |
| H1 forward | 2.872 | 0.56% | 0.59% | 1 |
| R1 forward, generated | 42.002 | 8.19% | 8.69% | 1 |
| A1 forward and density scaling | 35.047 | 6.84% | 7.25% | 3 |
| M1 forward recomputation | 15.833 | 3.09% | 3.28% | 1 |
| H2 forward | 10.341 | 2.02% | 2.14% | 2 |
| Readout forward and gradient seed | 2.162 | 0.42% | 0.45% | 4 |
| H2 reverse | 40.621 | 7.92% | 8.41% | 1 |
| M1 reverse | 3.780 | 0.74% | 0.78% | 1 |
| A1 reverse and density scaling | 32.854 | 6.41% | 6.80% | 2 |
| R1 reverse, generated fused kernel | 126.473 | 24.67% | 26.17% | 1 |
| H1 reverse | 17.409 | 3.40% | 3.60% | 1 |
| M0 reverse | 4.485 | 0.87% | 0.93% | 1 |
| R0 reverse preparation and coordinates | 88.837 | 17.33% | 18.38% | 2 |
| Remaining GPU/API gaps | 0.248 | 0.05% | - | - |
| **Total evaluator range** | **512.632** | **100.00%** | - | - |

At the coarser operation level, input transfer and zeroing consume 29.133 ms
(5.68%), forward execution consumes 168.793 ms (32.93%), and reverse execution
consumes 314.459 ms (61.34%). The three largest optimization targets are R1
reverse, R0 reverse, and the combined H2 reverse/R1 forward tier. R1 plus R0
reverse alone account for 215.310 ms, 42.00% of the full evaluator and 44.55%
of its kernel time.

`cudaStreamSynchronize` reports 483.303 ms because the host launches the
asynchronous evaluator and waits once at its boundary. This time overlaps the
listed kernels and must not be added to them.

## Result collection outside the timed range

After the evaluator range, Python result collection performs these additional
device operations:

| Operation | GPU time | Transfer |
|---|---:|---:|
| Node-energy result | no reduction kernel | 1,048,576 B D2H in 0.026 ms |
| Directed-to-node force reduction | 1.046 ms | 3,145,728 B D2H in 0.132 ms |
| Device virial/stress reduction | 3.696 ms across 9 launches | 72 B D2H in 0.001 ms |

The traced collection interval spans about 9.12 ms from its first CUDA call to
the final stress copy. Its 4.742 ms of reduction kernels and 0.159 ms of D2H
copies are not included in the 512.632 ms evaluator table. In particular, no
full pair-force or edge-vector array is copied to the host.

## Command and artifacts

The successful capture disabled symbol resolution and traced only CUDA and
NVTX. This avoids the symbol-server stall encountered by the first attempt.

```bash
env \
  PATH=/usr/local/cuda-13.3/bin:/usr/local/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 \
  SYMMETRIX_SOURCE_ROOT="$PWD" \
  SYMMETRIX_EXTENSION=/path/to/staged/symmetrix.cpython-312-x86_64-linux-gnu.so \
  SYMMETRIX_JIT_CACHE=/tmp/symmetrix-low-memory-policy-jit-cache \
  SYMMETRIX_JIT_POLICY=required \
  /usr/local/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --resolve-symbols=false \
  --capture-range=cudaProfilerApi --capture-range-end=none \
  --force-overwrite=true -o /tmp/symmetrix-low-memory-false-n32 \
  /tmp/symmetrix-response-cuda-venv/bin/python \
  benchmarks/standard_mace_streamed_benchmark.py \
  /path/to/mace-omat-0-medium-factorized.json \
  --backend kokkos --dtype float32 --modes direct --sizes 32 \
  --warmups 3 --repeats 1 --neighbor-skin 0.5 \
  --factorized-r1-source-strategy jit_plugin \
  --factorized-r1-direct-forward-executor jit_all \
  --factorized-r1-direct-reverse-executor jit \
  --standard-r0-executor automatic --standard-m0-executor automatic \
  --m1-polynomial-policy recompute --m1-recompute-tile-channels 32 \
  --mh0-state-policy full-retention-v1 \
  --edge-geometry-policy cartesian-f64-v1 \
  --readout-policy retained --phi1-policy retained \
  --harmonic-storage-policy retained --nvtx
```

Retained artifacts are associated with the `low-memory-false-profile-20260830`
qualification run:

| Artifact | SHA-256 |
|---|---|
| `low-memory-false-n32.nsys-rep` | `c743baa509e85f495e00a54a9b9719fb75ce8fecacaa07651cc53cca6e1b49c2` |
| `low-memory-false-n32.sqlite` | `678258822c1ab56684bb3b6042b3a6bd00b6aec7d9009fa2638f3d28d3877c16` |
| `profile-run.json` | `8b52323a8e93ee334bcb9c695504b78783a4c26b00b587f7a83367323106642f` |
| `control.json` | `70f9dcc67ee8666a677e772145c4a164da2d4655e6670d98dc545faa5a1fd39c` |
| `symmetrix-ncu-r1-reverse.ncu-rep` | `e27d68b5ba67c86a0064d456a1fa2e14e9fc370179589f770c4f5a68acf68697` |
| `symmetrix-ncu-r0-reverse.ncu-rep` | `0295e338a18e9ff465559bc5efdf651bd81e733ea15d18c3c52264468e29a89a` |
| `symmetrix-ncu-r1-forward.ncu-rep` | `f860b6ecf1588566216c88ca6aa23a32a5cace41950c47e858656d79d6c9bbdc` |
| `symmetrix-ncu-h2-reverse.ncu-rep` | `1bf92d0a090aac77d956fdc66a8e1b9e634d6c775fd4fbe76fdcdb39d9f41902` |
| `symmetrix-ncu-h2-reverse-source.ncu-rep` | `1bd435ae87d50477e786ff82e8feeea0ff0489c3b54f32ed1c8d5a48a4a41557` |

## Nsight Compute diagnosis

After the Systems attribution, focused Nsight Compute captures collected
`SpeedOfLight`, `MemoryWorkloadAnalysis`, `Occupancy`, `LaunchStats`,
`SchedulerStats`, and `WarpStateStats` for the four largest individual kernel
families. A second H2-reverse capture added `SourceCounters`. The captures used
`--profile-from-start off`, `--cache-control all`, `--clock-control boost`, one
matched kernel launch, and the same warmed workload and CUDA-profiler range as
above. Hardware-counter collection succeeded without a permission failure.

Nsight Compute replays a kernel several times to collect incompatible counter
sets. Its instrumented calls took 4-9 seconds and are not throughput results.
The table retains the reported single-kernel duration only as a capture sanity
check against the Systems trace; all evaluator and `us/atom` timing claims use
the uninstrumented measurements above.

Each capture used this profiler prefix followed by the exact `/usr/bin/env`,
Python executable, benchmark arguments, and `--nvtx` workload shown in the
Systems command above:

```bash
/opt/nvidia/nsight-compute/2026.2.1/ncu \
  --profile-from-start off --cache-control all --clock-control boost \
  --kernel-name "regex:${KERNEL_REGEX}" --launch-count 1 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section Occupancy --section LaunchStats --section SchedulerStats \
  --section WarpStateStats --force-overwrite \
  -o "/tmp/${OUTPUT}"
```

The four filters were `^symmetrix_factorized_reverse_fused_v2$`,
`.*launch_coordinate_reverse_edge_owned.*`,
`^symmetrix_factorized_forward_v2$`, and `.*reverse_H2.*`. The H2 source pass
used the same filter with `--section SourceCounters` only.

| Metric | R1 reverse | R0 reverse | R1 forward | H2 reverse |
|---|---:|---:|---:|---:|
| NCU kernel duration | 128.93 ms | 92.67 ms | 42.09 ms | 43.76 ms |
| Grid / block | 1,360 / 64 | 680 / 256 | 1,360 / 256 | 131,072 / 128 |
| Registers/thread | 80 | 80 | 96 | 40 |
| Waves/SM | 0.67 | 1.33 | 4.00 | 64.25 |
| Theoretical / achieved occupancy | 50.00% / 32.71% | 50.00% / 45.43% | 33.33% / 32.97% | 100.00% / 96.63% |
| SM throughput | 44.27% | 31.30% | 82.09% | 32.62% |
| L2 throughput | 66.31% | 11.65% | 79.77% | 87.61% |
| DRAM throughput | 9.95% | 3.68% | 6.62% | 0.48% |
| L1 / L2 hit rate | 15.03% / 96.51% | 82.67% / 92.60% | 32.25% / 98.31% | 7.32% / 99.85% |
| Eligible warps/scheduler | 0.38 | 0.81 | 0.52 | 0.06 |
| Long-scoreboard cycles/issue | 8.12 | 6.12 | 4.20 | 212.67 |
| LG-throttle cycles/issue | 0.08 | 0.46 | 0.08 | 110.50 |

### R1 reverse: persistent-grid and dependent-gather latency

The fused generated reverse kernel is L2/latency bound rather than DRAM bound.
Its 1,360-block grid is exactly eight blocks per each of the 170 SMs, while the
80-register, 64-thread launch permits twelve resident blocks per SM. The grid
therefore caps occupancy near 33% even though the resource ceiling is 50%.
Long-scoreboard stalls account for about 58% of the 14.01 cycles between issued
instructions. This matches a source-owned kernel that serially traverses each
source's edges and gathers receiver adjoints and features.

The first experiment should change only the generated persistent launch from
8 to 10 and 12 blocks/SM and measure uninstrumented time. A 128-thread or
lower-register fused kernel is a larger follow-up if increasing the grid does
not improve latency hiding.

### R0 reverse: register-limited dependent reductions

The standard `v2_edge16` coordinate reverse is neither L2- nor DRAM-saturated.
Its 80 registers/thread limit each SM to three 256-thread blocks and 50%
theoretical occupancy; measured occupancy is already 45.43%, so simple grid
expansion is not the answer. The edge setup and spline interval are prepared
before the channel vector reduction in `standard_r0.hpp`, but coefficient,
M0-adjoint, harmonic, and gradient loads remain in the dependent reduction.

Retest the existing `v2_edge32` variant against `v2_edge16` on this exact
workload before generating a new kernel. If neither wins, reduce live ranges
and register use enough to admit four blocks/SM, then examine whether the
coefficient/adjoint traversal can be tiled for more locality.

### R1 forward: repeated FP64 spline setup

The generated forward kernel is already close to both its compute and cache
ceilings: SM throughput is 82.09%, L2 throughput is 79.77%, and the FP64 pipe
is also 82.09%. Its 96 registers/thread allow only two 256-thread blocks per SM,
and achieved occupancy essentially equals the 33.33% theoretical limit.

For Cartesian-f64 geometry, `_render_gpu_spline_helpers()` intentionally uses
a double-precision `EvaluationPoint`. The receiver/channel owner loop calls
`evaluation_point()` for every edge, so the 128 channel owners repeat the same
FP64 floor, division, and cubic-coordinate setup for a shared edge radius. A
candidate retained-speed-path optimization is to compute compact per-edge
spline metadata once and reuse it across R0/R1 forward and reverse. Capacity
policies can continue recomputing this metadata if retaining it is not worth
the edge-sized allocation. Any change must preserve the present FP32-model
precision behavior of the Cartesian-f64 path.

### H2 reverse: uncoalesced weight traversal

H2 reverse has 96.63% occupancy and 64.25 waves/SM, yet only 0.06 warps per
scheduler are eligible. It hits in L2 99.85% of the time, saturates 87.61% of
L2 throughput, and uses only 0.48% of DRAM throughput. This is cache-transaction
pressure, not an off-device bandwidth limit.

`SourceCounters` attributes 5,570,035,712 theoretical global sectors to the
kernel versus 2,348,810,240 ideal sectors: 3,221,225,472 sectors, or 57.81%,
are excessive. Branch efficiency is 100%. In
`mace_kokkos_second_interaction.cpp`, adjacent warp lanes vary `k`, while the
inner loop loads `H2_weights_for_H1(k * channels + kp)` and
`H2_weights_for_M1(k * channels + kp)`. At a fixed `kp`, those lanes read rows
128 elements apart, producing the measured uncoalesced access.

The preferred correction is to store or expose the transposed weight layout,
or replace the loop with a tiled matrix multiplication. Mathematically the M1
part is `M1_adj = H2_adj @ W_M1^T`; H1 requires a type-grouped equivalent.
The weights are intentionally double precision on CUDA even for an FP32 model,
so this optimization must preserve that mixed-precision contract unless a
separate numerical qualification changes it.

## Ranked next experiments

1. Fix H2-reverse weight coalescing or introduce a tiled GEMM. It has a direct
   counter-proven defect and accounts for 7.92% of evaluator time.
2. Benchmark R1-reverse persistent grids at 10 and 12 blocks/SM. This is a
   small launch-policy change against the largest stage, 24.67% of evaluation.
3. Share retained per-edge spline metadata across R0/R1 for the speed path and
   measure whether removing repeated FP64 setup offsets the added traffic.
4. Retest R0 `v2_edge32`; then target register live ranges and coefficient/
   adjoint locality if the existing variant does not improve its 17.33% stage.

These percentages are independent opportunities, not predicted aggregate
speedups. Each candidate requires an uninstrumented matched-size benchmark and
full energy/force/stress parity before adoption.

## Follow-up optimization experiments

The same 131,072-atom, 14,286,848-directed-edge workload was used for all
variants. R0 `v2_edge32` measured 3.94334 us/atom versus 3.83971 us/atom for
`v2_edge16`, a 2.70% regression. A controlled same-artifact R1-reverse sweep
measured 4/8/16 persistent blocks per SM at 4.71545/3.92797/4.05308 us/atom.
The existing R0 edge16 and R1 eight-block policies are therefore retained.

The accepted H2 change stores reverse-specific transposed copies of its two
small weight matrices on CUDA/HIP. It leaves the forward layout and the GPU
double-weight precision policy unchanged. Alternating fresh-process runs gave:

| Binary | Median us/atom | Atoms/s | Relative time |
|---|---:|---:|---:|
| Committed baseline | 3.92222 | 254,958 | 1.0000x |
| Transposed H2 reverse | 3.68880 | 271,091 | 0.9405x |

The end-to-end reduction is 5.95%, or a 1.0633x speedup. A 256-atom
cross-binary comparison with 27,904 directed edges had exact energy and stress
agreement and a maximum force difference of 1.46e-15 eV/A. Both binaries
selected generated direct R1, standard M0/R0, and zero fallback.

Nsight Systems reduced H2 reverse from 40.621 to 13.832 ms, saving 26.789 ms.
Nsight Compute measured candidate global sectors at 2,348,810,240, exactly the
ideal count; the baseline used 5,570,035,712 sectors. SM throughput increased
from 32.62% to 90.54%, achieved occupancy from 96.63% to 99.47%, and L2
throughput fell from 87.61% to 44.95% as redundant cache transactions were
removed. Long-scoreboard and LG-throttle sample shares fell from about 63.3%
and 32.6% to 12.4% and effectively zero.

Candidate extension SHA-256:
`267f550f0de9a437f4874aed73d313a6429569142036bf839185e1a07c17b2f7`.
The retained reports are:

| Artifact | SHA-256 |
|---|---|
| `h2-candidate-n32.nsys-rep` | `affc91983f16953fba273a4f8543474583f0c329e734d1f5049213ed78bb1a21` |
| `h2-candidate-core.ncu-rep` | `4cc5df82972a9c30659287e8e1fe197846ebf6cdc8a288ad10a19118870db19d` |
| `h2-candidate-source.ncu-rep` | `402c6a36104f16b503610b0e82109064793401c2527c90c43503ea5616ebff4f` |

Reports and the staged candidate were retained with the
`h2-transpose-20260830` qualification run.

## Direct active-edge cutoff masking

The next experiment implemented the zero-allocation cutoff-sentinel design in
`plans/2026-08-29-direct-active-edge-masking-v1.md`. Prepared direct geometry
retains every cutoff-plus-skin candidate edge and the existing receiver/source
schedules, but stores the exact model cutoff as the radius for inactive edges.
Standard and RTC R0 plus every generated R1 owner skip those candidates before
type lookup, spline evaluation, harmonic-gradient work, and channel loops.
The Y-only direct harmonic kernel additionally writes deterministic zero values
for inactive edges. No mask view, graph compaction, schedule reconstruction,
new launch, or edge-array transfer was added.

The first prototype reconstructed the endpoint as `x0 + h * intervals` inside
generated kernels. Although that expression is exact for this OMAT-0 artifact,
a million randomly sampled valid spline grids found a one-ULP discrepancy from
the evaluator cutoff in about 9.99% of cases. The accepted implementation
therefore appends the exact evaluator `r_cut` to host/CUDA/HIP R1 launch
packets and RTC R0 packets, and advances the shared JIT generation from 3 to 4.
Old artifacts are load-incompatible by generation and packet size. The CUDA
extension used for
the performance sweep and profiler captures was:

```text
7bf18917a675f73f7493940020dae5ddb57ca3f239509738dad618ce29b00226
```

An independent review subsequently required the cutoff to move from the nested
R1 radial packet to an append-only field on every outer R1 launch packet. This
preserves the published nested radial layout while retaining the exact-cutoff
kernel semantics. The frozen post-review source was rebuilt from scratch for
CUDA 13.3 BLACKWELL120 and native OpenMP. The resulting extension hashes are:

| Final review build | SHA-256 |
|---|---|
| CUDA 13.3 BLACKWELL120 | `add0f2a40ef34e4baeb6965556fb595e344c6485b6d3f31623262d2988c910c5` |
| OpenMP native | `e15859afe39bcf1dddd6c97ec758d44a22c578500b1e7d3411172a4dd7561b69` |

The fresh CUDA build passed the complete streamed-edge test file: 114 passed
and 8 capability skips in 91.30 s. This includes all eight FP32/FP64 ordinary
MACE and MACEField cutoff-crossing cases for retained and forced-capacity
Y-only execution. The MACEField matrix covers energy, forces, stress,
polarization, polarizability, and BEC before and after an edge crosses the
cutoff while the candidate graph is reused. The fresh OpenMP build passed the
skin-expanded parameter-gradient replay comparison and a broader JIT,
operator, Kokkos-integration, and standard-module suite with 153 passed and 3
capability skips. The performance numbers below remain attached to the
performance-sweep binary; the post-review change relocates packet metadata and
does not alter the generated arithmetic kernels.

### Skin sweep

The sweep used OMAT-0-medium FP32 energy, forces, and stress on 131,072 atoms,
a 6.0 A model cutoff, three warmups, and seven measured evaluations. All runs
selected generated R1 `jit_all`/`jit`, standard M0, standard R0 `v2_edge16`, M1
tile-32 recomputation, full MH-0 state, Cartesian FP64 geometry, and zero
fallback. The active-edge count is the exact-cutoff graph's 11,927,552 directed
edges; the larger counts retain inactive Verlet candidates.

| Skin (A) | Effective cutoff (A) | Candidate edges | Inactive | Baseline us/atom | Masked us/atom | Atoms/s | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 6.00 | 11,927,552 | 0.00% | 3.25755 | 3.19929 | 312,569 | 1.82% |
| 0.25 | 6.25 | 13,500,416 | 11.65% | 3.53021 | 3.31474 | 301,682 | 6.50% |
| 0.50 | 6.50 | 14,286,848 | 16.51% | 3.68552 | 3.37207 | 296,554 | 9.30% |
| 1.00 | 7.00 | 18,743,296 | 36.36% | 4.43411 | 3.62978 | 275,499 | 22.16% |

The plan's gates pass: the zero-inactive case improves rather than regresses,
and the 16.51%-inactive case exceeds the required 5% gain. Sampled process VRAM
is unchanged at each matched graph size (17,908/18,706/19,104/21,366 MiB), as
expected for an arithmetic-only optimization that retains candidate storage.

After replacing the derived spline endpoint with the exact packet cutoff, the
final binary measured 3.20007 us/atom at skin 0.0 and 3.37397 us/atom at skin
0.5. These differ from the prototype by only 0.02% and 0.06%, respectively.
FP32 and FP64 cutoff-crossing tests passed for retained and forced-capacity
Y-only direct execution: a candidate edge moved from 0.05 A inside to 0.05 A
outside the cutoff while candidate schedule bytes/build count remained fixed,
and energy, forces, and stress matched a zero-skin graph.

### Reproduction commands

The skin sweep used the following command template in a fresh process for each
binary and skin. `EXTENSION` was the staged baseline or candidate extension,
`SKIN` was one of `0`, `0.25`, `0.5`, or `1.0`, and `OUTPUT` was the matching
JSON path listed below. Each binary used its own initially empty task-local JIT
cache.

```bash
env \
  PATH=/usr/local/cuda-13.3/bin:/usr/local/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 \
  SYMMETRIX_SOURCE_ROOT="$PWD" \
  SYMMETRIX_EXTENSION="$EXTENSION" \
  SYMMETRIX_JIT_CACHE="$JIT_CACHE" \
  SYMMETRIX_JIT_POLICY=required \
  /tmp/symmetrix-response-cuda-venv/bin/python \
  benchmarks/standard_mace_streamed_benchmark.py \
  /path/to/mace-omat-0-medium-factorized.json \
  --backend kokkos --dtype float32 --modes direct --sizes 32 \
  --warmups 3 --repeats 7 --neighbor-skin "$SKIN" \
  --factorized-r1-source-strategy jit_plugin \
  --factorized-r1-direct-forward-executor jit_all \
  --factorized-r1-direct-reverse-executor jit \
  --standard-r0-executor v2_edge16 --standard-m0-executor automatic \
  --m1-polynomial-policy recompute --m1-recompute-tile-channels 32 \
  --mh0-state-policy full-retention-v1 \
  --edge-geometry-policy cartesian-f64-v1 \
  --readout-policy retained --phi1-policy retained \
  --harmonic-storage-policy retained --output "$OUTPUT"
```

The final exact-cutoff binary was rerun with the same template at skin `0` and
`0.5`, except `--standard-r0-executor automatic` selected `v2_edge16`.

| Sweep artifact | SHA-256 |
|---|---|
| `baseline-skin0.json` | `9f8d60c413ce5ceeaeaffddf0897009b2b307697f00d58eb3decbe3de71874bf` |
| `baseline-skin025.json` | `57804d10da6d3301ae58fbf232bcb7f14b638039381aa0f5b5fca73faccbea57` |
| `baseline-skin05.json` | `58ab48e5fbc5d3bc9ccd2e601493e43675468196f3f4bad09877c59c4c3da503` |
| `baseline-skin10.json` | `866dc9a072b72b9fe61e5dfffa00e0b93df05931cb1cfa1f970107a1c38fa8ac` |
| `candidate-skin0.json` | `24402fbaa8898a68e95209000cb8842bb27ec00c6a6f857ef0b3d6ad1c7ee3df` |
| `candidate-skin025.json` | `d3cc7ffeb2590f7137848c9e18db92b0ab6dd99711fb890fbc787aa85de85dbb` |
| `candidate-skin05.json` | `9d89c7aadaf2accd75764bd86c1eabf6f4c3c568911f95fbb1d1caf8aa4e4d9f` |
| `candidate-skin10.json` | `c4f24fc8a8bf5fcb095840cac41389bd8c39bde48fb35474b5aaa6575d597de1` |
| `final-skin0.json` | `bfb07f068aadcdd9851995f0cb8fcc4383e4b7651dc4e28a3403636125cb99b9` |
| `final-skin05.json` | `416331c57466c8d3abd00ca16295180588b4370f173fc4383ab900eceaff8a59` |

### Profiler attribution

Matched skin-0.5 Nsight Systems captures measured the instrumented evaluator at
485.545 ms before masking and 448.754 ms after masking, a 7.58% reduction. The
launch count stayed at 36 GPU operations and no new transfer appeared.

| Kernel family | Baseline ms | Masked ms | Reduction |
|---|---:|---:|---:|
| R1 reverse | 128.123 | 102.115 | 20.30% |
| R1 forward | 41.595 | 36.611 | 11.98% |
| R0 forward | 35.440 | 30.423 | 14.16% |
| R0 reverse | 86.543 | 84.999 | 1.78% |

Targeted Nsight Compute captures show that the radius predicate is effectively
warp-uniform on this ordered candidate graph. R1 reverse improved from 128.25
to 101.84 ms, SM throughput rose from 44.50% to 54.16%, and long-scoreboard
latency fell from 8.0 to 6.8 cycles/issue. R1 forward improved from 42.31 to
38.86 ms. Active versus not-predicated threads per warp remained approximately
31.7/31.1 for reverse and 32.0/31.92 for forward; achieved occupancy remained
about 32.7-33.2%. The benefit comes from avoiding radial/channel work, not from
changing occupancy or compacting the graph.

Nsight Systems and Nsight Compute used this exact workload array, substituting
the matched baseline/candidate `EXTENSION`, a distinct empty `JIT_CACHE`, and
the report path shown below:

```bash
profile_workload=(
  env
  PATH=/usr/local/cuda-13.3/bin:/usr/local/bin:/usr/bin:/bin
  LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
  SYMMETRIX_SOURCE_ROOT="$PWD"
  SYMMETRIX_EXTENSION="$EXTENSION"
  SYMMETRIX_JIT_CACHE="$JIT_CACHE"
  SYMMETRIX_JIT_POLICY=required
  /tmp/symmetrix-response-cuda-venv/bin/python
  benchmarks/standard_mace_streamed_benchmark.py
  /path/to/mace-omat-0-medium-factorized.json
  --backend kokkos --dtype float32 --modes direct --sizes 32
  --warmups 3 --repeats 1 --neighbor-skin 0.5
  --factorized-r1-source-strategy jit_plugin
  --factorized-r1-direct-forward-executor jit_all
  --factorized-r1-direct-reverse-executor jit
  --standard-r0-executor v2_edge16 --standard-m0-executor automatic
  --m1-polynomial-policy recompute --m1-recompute-tile-channels 32
  --mh0-state-policy full-retention-v1
  --edge-geometry-policy cartesian-f64-v1
  --readout-policy retained --phi1-policy retained
  --harmonic-storage-policy retained --nvtx
)
```

Nsight Systems used this exact prefix, substituting the report name:

```bash
/usr/local/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --resolve-symbols=false \
  --capture-range=cudaProfilerApi --capture-range-end=none \
  --force-overwrite=true -o "$REPORT" \
  "${profile_workload[@]}"
```

Targeted Nsight Compute used the identical one-call workload. `KERNEL_REGEX`
was `^symmetrix_factorized_reverse_fused_v2$` or
`^symmetrix_factorized_forward_v2$` and each baseline/candidate capture used a
distinct `REPORT` path.

```bash
/opt/nvidia/nsight-compute/2026.2.1/ncu \
  --profile-from-start off --cache-control all --clock-control boost \
  --kernel-name "regex:${KERNEL_REGEX}" --launch-count 1 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section Occupancy --section LaunchStats --section SchedulerStats \
  --section WarpStateStats --force-overwrite -o "$REPORT" \
  "${profile_workload[@]}"
```

| Profiler artifact | SHA-256 |
|---|---|
| `nsys-baseline-skin05.nsys-rep` | `32326721114b3f773c491700c95367140b605824f8bf84981260ace9823787e3` |
| `nsys-candidate-skin05.nsys-rep` | `670c98e8dd9816c16e34603556ddfe7e45bc8856ee2bb43b29bb4f98c1de8488` |
| `nsys-baseline-skin05.sqlite` | `ff4d451a73c54087ab19d3d9d55bb4958e7479c68d88dab4002dccb9f4e1ab98` |
| `nsys-candidate-skin05.sqlite` | `faaeafd81d75936e308fc44af4ea6df21dc65b8fa34eba73c2aea0a81caccfe3` |
| `ncu-baseline-r1-forward-skin05.ncu-rep` | `a31a67c45944c77e8e20e111e5b85b1ecd0e2714ef13b45010583522646fb465` |
| `ncu-candidate-r1-forward-skin05.ncu-rep` | `296eb8a18bffe2b5bce33320528a257122133f33a0723861998ebd72a7bee603` |
| `ncu-baseline-r1-reverse-skin05.ncu-rep` | `197bccbf9439aec0f4e3af59c9a9a098500fdef828bc0161e5b46739933a788d` |
| `ncu-candidate-r1-reverse-skin05.ncu-rep` | `ce2f258580a94d102f9fe19d718b8427fda673975a7ab1c9b54538e35f7b85ac` |

Reports, exact-cutoff JSONs, and the staged binary were retained with the
`active-edge-mask-20260830` and `active-edge-exact-cutoff-20260830`
qualification runs.
