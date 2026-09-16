# MH-1 CUDA throughput, capacity, and profile on RTX 5090

Date: 2026-08-31

## Summary

The current generated MH-1 CUDA path was qualified on an RTX 5090 with the
official compact Al/N model. At 864 atoms, an exact 6.0 A graph, and FP32
energy, forces, and stress, full retention measured **31.9802 us/atom**. The
node-state recompute policy measured **34.0823 us/atom**, 6.57% slower, while
increasing demonstrated capacity from **87,808 to 143,748 atoms** (1.637x).

The matched retained MH-0 result is 3.63055 us/atom on the same 864-atom,
78,624-edge workload, so MH-1 full retention is 8.81x slower and recompute is
9.39x slower. The accepted OMAT-0-medium result is 3.68880 us/atom on the same
GPU, but its 131,072-atom graph uses a 0.5 A skin and is not a model-only
comparison.

Nsight Systems attributes 97.45% of the full-retention range to six MH-1 stage
families. Node-network work is largest in aggregate (35.92%), while
interaction-1 edge reverse is the largest single kernel (18.23%). Focused
Nsight Compute reports show that the three interaction-1 kernels are
register- and dependency-limited with 99.72-99.89% L2 hit rates and only
0.45-1.12% DRAM throughput. Reducing register live ranges, redundant cached
gathers, and dependent load chains is better supported than host, transfer, or
off-device bandwidth work.

## Provenance

| Item | Value |
| --- | --- |
| Source | `85a673733b3bfc1524dd71f92d784923540a0a91` plus the MH-1 code-generation and benchmark-driver fixes in this change |
| GPU | NVIDIA GeForce RTX 5090, 170 SMs, compute capability 12.0, 32,607 MiB |
| Driver | 610.43.02 |
| CUDA compiler/runtime | 13.3.73 / 13.3 |
| Nsight Systems | 2026.1.3 |
| Nsight Compute | 2026.2.1 |
| Python | `.venv/bin/python` (CPython 3.12) |
| Kokkos execution space | CUDA |
| Native extension SHA-256 | `6d31fafec1f8a2e07a4a04892275786d72dd023e79938dd89490174905d14f3f` |
| MH-1 compact model SHA-256 | `9cb9413e227ee958508c3d7859a68cb7676d8adcd5cf8a64720e7025d68a1526` |
| Generated artifact | `jit-mh1-gen4-v4-86823d33d34d12b5` |
| Edge policy | `hybrid-path-tiled-v7`; compact fused reverse for both interactions |

The fresh extension was built for `Kokkos_ARCH_BLACKWELL120`, Kokkos CUDA plus
Serial, and CUDA SpheriCart. All timing and capacity workers explicitly loaded
that extension rather than relying on the editable-install finder.

The current source initially emitted ordinary R1 active-cutoff logic into the
exact-graph MH-1 spline program, although the MH-1 packet has no ordinary R1
cutoff field. The shared renderer also had a stale output-base call and omitted
the direct harmonic-gradient helper required by the MH-1 reverse adapter. The
fix makes ordinary cutoff filtering explicit at the shared renderer boundary,
keeps it enabled for ordinary R1, disables it for the exact MH-1 graph, emits
the matching harmonic helper, and passes the receiver-local output base. The
qualification extension includes these fixes.

## Workload and timing contract

- Periodic wurtzite AlN, `repeat((6, 6, 6))`, 864 atoms.
- Model cutoff 6.0 A, skin 0 A, effective cutoff 6.0 A.
- 78,624 directed edges, or 91 edges/atom.
- FP32 evaluator; energy, forces, and stress.
- Generated CUDA v4, required NVRTC, prepared graph, and zero fallback.
- Ten warmups followed by 20 individually synchronized evaluations.
- Graph construction, preparation, JIT compilation/loading, and result
  collection are outside the steady-state interval.

## Throughput

| MH-1 node policy | Median ms/call | Median us/atom | Sample range ms | Process GPU MiB | Relative time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full retention | 27.630901 | **31.9802** | 27.536635-27.681415 | 1,278 | 1.000x |
| Recompute | 29.447075 | **34.0823** | 29.395314-29.503676 | 1,156 | 1.0657x |

At this small workload recompute saves 122 MiB of sampled process memory. The
32,000-atom comparison below is a better measure of its scaling benefit.

A second fresh full-retention process measured 27.795521 ms, or 32.170741
us/atom. It is 0.596% above the primary result and reproduced the graph hash,
energy, selected generated variant, prepared-evaluation count, and zero
fallback. The environment rejected the corresponding second unsandboxed
recompute launch before process creation, so recompute has one unprofiled
20-sample process in this campaign.

## Throughput comparison

| Model/result | Atoms | Cutoff / skin / effective A | Directed edges | us/atom | MH-1 full relative time | Comparison |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| MH-1 full retention | 864 | 6.0 / 0 / 6.0 | 78,624 | **31.9802** | 1.000x | Current measurement |
| MH-1 recompute | 864 | 6.0 / 0 / 6.0 | 78,624 | **34.0823** | 1.0657x | Current measurement |
| Retained MH-0 | 864 | 6.0 / 0 / 6.0 | 78,624 | **3.63055** | 0.1135x | Matched workload; MH-1 full is 8.81x slower |
| OMAT-0-medium, transposed H2 | 131,072 | 6.0 / 0.5 / 6.5 | 14,286,848 | **3.68880** | 0.1153x | Same GPU, different scale and graph; raw ratio is 8.67x |

The MH-0 comparison is the fair architecture/model-cost comparison: device,
precision, atom count, cutoff, skin, topology, properties, prepared execution,
and zero-fallback contract match. The OMAT number is useful production-scale
hardware context only. Its larger graph amortizes fixed work differently and
includes candidates between 6.0 and 6.5 A.

Historical sources are `benchmarks/mh0_cuda_rtx5090_profile.md` and
`benchmarks/omat0_medium_low_memory_false_profile_20260830.md`.

## Capacity protocol

Each candidate ran in a fresh process. The worker built the exact 6.0 A graph,
performed one setup evaluation, then one measured evaluation, and required:

- the requested full-retention or recompute generated variant;
- prepared execution and two completed prepared evaluations;
- generated forward, source-reverse, and edge-reverse launches;
- zero factorized fallback;
- a finite energy, force, and stress result;
- sampled process and total device memory.

Capacity is the largest successful integer cubic repeat adjacent to a worker
that failed with a CUDA allocator OOM. It is not an extrapolated intercept.

| Policy | Largest success | Atoms | Directed edges | Process GPU MiB | Total sampled MiB | First failure | OOM request |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Full retention | `n28` | **87,808** | 7,990,528 | 31,620 | 31,888 | `n29`, 97,556 atoms / 8,877,596 edges | 2.117 GiB |
| Recompute | `n33` | **143,748** | 13,081,068 | 31,466 | 31,790 | `n34`, 157,216 atoms / 14,306,656 edges | 3.411 GiB |

Recompute adds 55,940 atoms of demonstrated capacity: **1.637x**, or **63.7%**.
At a matched 32,000-atom, 2,912,000-edge graph, the memory tradeoff is:

| Policy | Process GPU MiB | One-sample us/atom |
| --- | ---: | ---: |
| Full retention | 12,152 | 42.578 |
| Recompute | 7,774 | 44.372 |
| Difference | **-4,378 (-36.0%)** | +4.21% |

The one-sample large-graph timings qualify execution but are not primary
throughput evidence.

## Capacity comparison

| Model/policy | Largest demonstrated atoms | Directed candidates | Effective cutoff A | Interpretation |
| --- | ---: | ---: | ---: | --- |
| MH-1 full retention | **87,808** | 7,990,528 | 6.0 | Adjacent OOM boundary |
| MH-1 recompute | **143,748** | 13,081,068 | 6.0 | Adjacent OOM boundary |
| OMAT-0-medium full retention | 219,488 | 23,924,192 | 6.5 | Adjacent OOM boundary; 2.50x MH-1 full atom count |
| OMAT-0-medium automatic capacity bundle | 530,604 | 57,835,836 | 6.5 | Adjacent OOM boundary; 3.69x MH-1 recompute atom count |

OMAT has about 109 candidates/atom versus MH-1's exact 91 edges/atom, but it is
also a materially smaller execution graph. Raw atom-capacity ratios therefore
combine graph density, model dimensions, precision/storage policy, and runtime
state. They must not be interpreted as topology-normalized model ratios.

The older exact-graph MH-0 campaign also demonstrated `n29` (97,556 atoms and
8,877,596 edges) at 13,392 MiB for full retention and 7,760 MiB for compact
adjoint reuse. That point was not an MH-0 boundary, but it is a direct contrast:
MH-1 full retention fails at the same `n29` graph.

## Nsight Systems attribution

Full-process CUDA/NVTX capture was used because capture-range mode produced
incomplete streams. `DEBUGINFOD_URLS` was unset to prevent external symbol
server stalls, and the final `symmetrix_full_evaluator::direct` range was
selected from the exported SQLite report.

The full-retention range is 28.491 ms:

- 320 kernel launches: 28.039 ms.
- Two geometry H2D copies: 2,515,968 bytes and 0.122 ms.
- Fourteen memsets: 46,230,912 bytes and 0.020 ms.
- Inter-activity gaps: 0.172 ms total, 37.1 us maximum.

| Stage family | Time ms | Range share |
| --- | ---: | ---: |
| Node network | 10.233 | 35.92% |
| Interaction edge reverse | 7.247 | 25.44% |
| Interaction source reverse | 3.864 | 13.56% |
| Interaction forward | 3.513 | 12.33% |
| Conditioning reverse | 2.042 | 7.17% |
| Conditioning forward | 0.863 | 3.03% |
| Remaining kernels/API time | 0.729 | 2.56% |

The largest single kernel is
`symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_1`: 5.194 ms,
18.23% of the complete range, at `2720 x 128`, 96 registers/thread, and 10,000
bytes of static shared memory.

Recompute expands the measured range to 29.658 ms and 488 launches. Relative
to full retention it adds 1.166 ms end to end, 1.098 ms of kernel execution,
168 launches, and 1.556 ms of node-network work. The recompute penalty is
therefore node-state reconstruction and its extra stage sequence, not transfer
or host dispatch time.

## Nsight Compute

Each report captures one interaction-1 launch after ten warmups with
`SpeedOfLight`, `MemoryWorkloadAnalysis`, `Occupancy`, `LaunchStats`,
`SchedulerStats`, and `WarpStateStats`. NCU durations include replay effects
and are not end-to-end benchmark measurements.

| Metric | Edge reverse 1 | Source reverse 1 | Forward 1 |
| --- | ---: | ---: | ---: |
| Systems duration ms | 5.194 | 2.462 | 2.368 |
| NCU duration ms | 5.86 | 2.33 | 2.42 |
| Grid x block | `2720 x 128` | `680 x 128` | `680 x 128` |
| Registers/thread | 96 | **168** | **128** |
| Theoretical occupancy | 41.67% | **25.00%** | **33.33%** |
| Achieved occupancy | 37.39% | **21.51%** | **28.57%** |
| Eligible warps/scheduler | 0.50 | **0.25** | **0.38** |
| Cycles with no eligible warp | 68.51% | **80.42%** | **78.25%** |
| L1/TEX hit rate | 47.98% | 55.12% | 60.92% |
| L2 hit rate | **99.89%** | **99.72%** | **99.78%** |
| DRAM throughput | 0.45% | 1.12% | 1.07% |
| Long-scoreboard share of issue interval | 46.9% | **59.5%** | 36.0% |
| LG-throttle share | not flagged | not flagged | **39.9%** |
| Local/shared spills | 0 / 0 | 0 / 0 | 0 / 0 |

Source reverse is the clearest register-residency problem: 168 registers allow
only three blocks/SM and leave no eligible warp on four out of five scheduler
cycles. Its traffic is L2-resident, so improving coalescing/locality and
shortening dependent gather chains can improve latency hiding without chasing
DRAM bandwidth.

Forward has both dependency and instruction-queue pressure. NCU attributes
6.3 of 15.79 issue cycles to LG throttle and 5.7 cycles to long scoreboard.
This supports eliminating redundant/narrow global operations, reusing loaded
values, and interleaving math with remaining loads. Forcing a register cap
without restructuring is not supported because all three current kernels have
zero spills; a cap could simply trade occupancy for local-memory traffic.

## Ranked optimization directions

1. **Reduce interaction-1 edge-reverse work and dependency length.** It is the
   largest single kernel and edge reverse is 25.44% of the range. Inspect the
   generated path/harmonic loops for repeated receiver adjoint, radial, and
   harmonic loads; hoist or share values only when it also shortens live ranges.
   Gate candidates on end-to-end us/atom, registers, spills, and full-array
   force/stress parity.
2. **Restructure interaction-1 source reverse around its 168-register working
   set.** Split independent contraction phases or tile channels only if the
   split reduces live state and cached gather dependencies. The target is more
   than three resident blocks/SM without duplicated edge traversal or atomics.
3. **Cut forward load/store instruction pressure.** Combine lower-width
   operations where alignment permits, eliminate reloads across path groups,
   and interleave math with gathers. Its 39.9% LG-throttle and 36.0%
   long-scoreboard shares provide direct acceptance counters.
4. **Fuse recompute reconstruction for data reuse, not host-gap removal.** The
   policy adds 168 launches and 1.556 ms of node work, while total full-retention
   gaps are only 0.172 ms. Combine adjacent reconstruction stages when it
   removes intermediate reads/writes or preserves values in registers/shared
   memory; launch-count reduction alone has a small ceiling.
5. **Reduce retained MH-1 lifetimes for capacity.** The source audit identifies
   destructive message reverse (20,480 B/atom), direct graph-input adjoint
   accumulation (conditional 2,560 B/atom), generated readout seeds (2,048
   B/atom), and graph-embedding lifetime reuse as bounded next targets. Measure
   each at 32,000 atoms before another boundary search.
6. **Deprioritize CPU orchestration, asynchronous copies, and DRAM bandwidth.**
   Kernels nearly fill the measured range, transfers cost 0.122 ms, and all
   three profiled kernels have greater than 99.7% L2 hit rates with at most
   1.12% DRAM throughput.

## Reproduction

The task-local artifacts use these variables:

```bash
artifacts="$PWD/.planning/2026-08-31-mh1-gpu-benchmark-and-optimization-profi/artifacts"
python="$PWD/.venv/bin/python"
export TMPDIR="$artifacts/nvcc-tmp"
export SYMMETRIX_SOURCE_ROOT="$PWD"
export SYMMETRIX_EXTENSION="$artifacts/cuda-stage-fixed/symmetrix/symmetrix.cpython-312-x86_64-linux-gnu.so"
export SYMMETRIX_JIT_CACHE="$artifacts/jit-cache-fixed-benchmark"
export SYMMETRIX_JIT_POLICY=required
```

Throughput, replacing the policy and output name for recompute:

```bash
"$python" benchmarks/standard_mace_streamed_benchmark.py \
  "$artifacts/mace-mh-1-current-Al-N.json" \
  --backend kokkos --dtype float32 --modes direct --sizes 6 \
  --neighbor-skin 0 --warmups 10 --repeats 20 \
  --mh1-node-state-policy full-retention-v1 \
  --source-commit 85a673733b3bfc1524dd71f92d784923540a0a91+mh1-codegen-fix \
  --output "$artifacts/mh1-cuda-864-full-retention.json"
```

Capacity boundary worker, replacing repeat, policy, and output as needed:

```bash
"$python" benchmarks/low_memory_capacity.py worker \
  --model-kind mh1 --model "$artifacts/mace-mh-1-current-Al-N.json" \
  --dtype float32 --neighbor-skin 0 --warmups 0 --samples 1 \
  --gpu-device 0 --mh1-node-state-policy full-retention-v1 \
  --repeat 28 --output "$artifacts/capacity/records/mh1-full-n28.json"
```

Systems capture, followed by SQLite selection of the final NVTX range:

```bash
env -u DEBUGINFOD_URLS /usr/local/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$artifacts/profile/mh1-full-864-all" \
  "$python" benchmarks/standard_mace_streamed_benchmark.py \
  "$artifacts/mace-mh-1-current-Al-N.json" \
  --backend kokkos --dtype float32 --modes direct --sizes 6 \
  --neighbor-skin 0 --warmups 10 --repeats 1 \
  --mh1-node-state-policy full-retention-v1 --nvtx
```

Focused Compute capture, replacing the exact kernel name for the other two
reports:

```bash
env -u DEBUGINFOD_URLS /opt/nvidia/nsight-compute/2026.2.1/ncu \
  --profile-from-start off --cache-control all --clock-control boost \
  --kernel-name 'regex:^symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_1$' \
  --launch-count 1 --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section Occupancy \
  --section LaunchStats --section SchedulerStats --section WarpStateStats \
  --force-overwrite -o "$artifacts/profile/mh1-edge-reverse-1" \
  "$python" benchmarks/standard_mace_streamed_benchmark.py \
  "$artifacts/mace-mh-1-current-Al-N.json" \
  --backend kokkos --dtype float32 --modes direct --sizes 6 \
  --neighbor-skin 0 --warmups 10 --repeats 1 \
  --mh1-node-state-policy full-retention-v1 --nvtx
```

## Artifacts and limitations

Primary records are under
`.planning/2026-08-31-mh1-gpu-benchmark-and-optimization-profi/artifacts/`:

- `mh1-cuda-864-full-retention.json`
- `mh1-cuda-864-full-retention-repro.json`
- `mh1-cuda-864-recompute.json`
- `capacity/records/mh1-{full,recompute}-n*.json`
- `profile/mh1-{full,recompute}-864-all.nsys-rep`
- `profile/mh1-edge-reverse-1.ncu-rep`
- `profile/mh1-source-reverse-1.ncu-rep`
- `profile/mh1-forward-1.ncu-rep`

Capacity successes and adjacent failures are single fresh processes, and GPU
memory is sampled rather than allocator-instrumented. The repeat grid leaves a
9,748-atom interval above full retention and a 13,468-atom interval above
recompute. Profiler captures use one measured evaluation and perturb runtime.
Historical MH-0 and OMAT comparisons use different source commits and native
binaries; only the MH-0 864-atom throughput workload is fully matched.
