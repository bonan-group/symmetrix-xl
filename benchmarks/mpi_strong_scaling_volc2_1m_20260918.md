# Volc2 one-million-atom MPI strong scaling

Date: 2026-09-18
Host: `volc2`
Remote evidence: `/vepfs/symmetrix-fixed-workspace-mpi/strong-scaling-1m-20260918`

## Outcome

The one-million-atom SrTiO3 workload scales from one A100 and one MPI rank to
two A100s and two MPI ranks with **1.9701x speedup and 98.50% efficiency**.
Median end-to-end time falls from 7.03705 to 3.572005 us/atom/step.

On the two-rank critical path, Symmetrix hidden-state communication is
0.056502 us/atom/evaluation, or **1.582% of pair evaluation**. LAMMPS's
separate `Comm` row is only 0.001942 us/atom/step, or about 0.054% of the
step. Combining the non-overlapping pair-internal and LAMMPS communication
timers gives an approximate MPI-associated share of 1.64%. The 1.582% value
is the primary MPI share because it is rank-correlated and measures the
forward/reverse H1 path that LAMMPS otherwise charges to `Pair`.

## Contiguous communication optimization

The baseline pack/unpack implementation launched a rank-3 Kokkos kernel for
each communicated value. The H1 views use `LayoutRight`, and the communicated
atom blocks are contiguous, so the optimized path flattens each block into one
linear operation. For FP64, the device buffer and H1 storage have the same
type and use `Kokkos::deep_copy`, which maps to a device-to-device copy on CUDA
and the corresponding backend copy on CPU. For FP32, LAMMPS's communication
buffer remains a `double` packet ABI, so the optimized path uses one linear
Kokkos conversion kernel in each direction rather than an invalid byte copy.
The gather pack and reverse scatter-add paths remain kernels because they use
index lists and atomics, respectively.

The candidate was built in a copy of the existing remote workspace; the
baseline executable and static library were not modified.

| Candidate | Location |
|---|---|
| Source workspace | `/vepfs/symmetrix-fixed-workspace-mpi/optimized-contiguous-copy-20260919` |
| Executable | `build/lmp` |
| Candidate executable SHA256 | `6e1b5292e2acc29053b2b584ef57afcfd0ba64ba1fbeaec2bfcd152403f52f9c` |

The candidate used the same 1M-atom SrTiO3 input, FP32 model, cutoff, rank
mapping, warmup, and measured-step protocol as the baseline. Three fresh
repetitions produced:

| MPI ranks / GPUs | us/atom/step samples | Median us/atom/step | Speedup | Efficiency |
|---:|---|---:|---:|---:|
| 1 | 6.99565, 7.00675, 6.99535 | **6.99565** | 1.0000x | 100.00% |
| 2 | 3.533325, 3.533565, 3.53563 | **3.533565** | **1.9798x** | **98.99%** |

Relative to the baseline medians, the candidate reduces one-rank time by
0.59% and two-rank time by 1.08%. The candidate two-rank internal timing is
3.531296 us/atom/evaluation critical pair, 3.501802 noncommunication, and
0.029102 H1 communication, for an H1 share of approximately **0.824%**.
The H1 communication time is about **48.5% lower** than the baseline
0.056502 us/atom/evaluation, while the noncommunication component is
essentially unchanged.

Nsight Systems confirms the kernel-level change. On rank 0 of the matched
short profile, the new linear conversion kernels took:

- `unpack_forward_comm_convert`: 42 launches, 6.0223 ms aggregate;
- `pack_reverse_comm_convert`: 42 launches, 6.1103 ms aggregate.

The corresponding baseline kernels were rank-3 Kokkos kernels taking 70.3688
ms and 78.7969 ms, respectively. The gather `pack_forward_comm_kokkos` and
scatter-add `unpack_reverse_comm_kokkos` kernels remain and measured 74.5355 ms
and 60.0681 ms in the candidate, consistent with their non-contiguous and
atomic semantics. Reports and SQLite exports are retained under the candidate
`profile/` directory.

## Direct-speed retest without fixed workspace

The previous candidate comparison forced `mh0-dual-layer-tiled-v1` to qualify
the fixed-workspace path. To isolate the communication optimization from that
policy, the same 1M-atom workload was rerun with
`profile speed allow_fixed_workspace no _debug_execution_plan mh0-direct-speed`.
The logs report `fixed workspace=disabled`, `plan=mh0-direct-speed`, and one
dedicated communicated-H1 allocation. This is the retained direct execution
plan, not the fixed-workspace plan.

Remote test folder:
`/vepfs/symmetrix-fixed-workspace-mpi/mh0-direct-speed-optimized-20260919`.
The baseline used the original `build/lmp` executable (SHA256
`55d06587fbb9fff64aef1ccf534373f54ac2f0e1570d3bae75c9c93394c09220`); the
candidate used `optimized-contiguous-copy-20260919/build/lmp` (SHA256
`6e1b5292e2acc29053b2b584ef57afcfd0ba64ba1fbeaec2bfcd152403f52f9c`). Each
case used three fresh 5-warmup/20-measured-step repetitions.

| Plan / binary | 1-rank samples | Median 1-rank | 2-rank samples | Median 2-rank | Speedup | Efficiency |
|---|---|---:|---|---:|---:|---:|
| `mh0-direct-speed` baseline | 6.088550, 6.090750, 6.101550 | 6.090750 | 3.075625, 3.077915, 3.083340 | 3.077915 | 1.9789x | 98.94% |
| `mh0-direct-speed` candidate | 6.066000, 6.067900, 6.071150 | 6.067900 | 3.062930, 3.062985, 3.067415 | 3.062985 | **1.9810x** | **99.05%** |

The direct-speed internal timing medians were:

| Binary | Critical pair | Noncommunication | H1 communication | H1 share |
|---|---:|---:|---:|---:|
| Baseline, 2 ranks | 3.075471 | 3.022637 | 0.052834 | 1.718% |
| Candidate, 2 ranks | 3.060699 | 3.029729 | 0.031797 | **1.039%** |

The contiguous-copy optimization reduces direct-speed H1 communication by
**39.8%** and improves two-rank end-to-end time by **0.49%**. All twelve
process runs exited successfully with zero dangerous builds. Raw logs and
command records are retained in the remote test folder under `logs/` and
`results/`.

## Workload and environment

- 1,000,000 SrTiO3 atoms from a `100 x 100 x 20` five-site cubic replication
- OMAT-medium standard MACE, two interactions, 128 channels, FP32
- 6.0 Angstrom model cutoff, 0.5 Angstrom skin, 6.5 Angstrom effective cutoff
- `streamed_edges direct`, `mh0-dual-layer-tiled-v1`, 32,768 receiver workspace
- 5 warmup and 20 measured NVT steps per fresh process, three interleaved
  repetitions of one rank and two ranks
- One MPI rank per A100-SXM4-80GB; `1x1x1` and `1x1x2` processor grids
- The A100s are connected by 12 bonded NVLinks (`NV12`)
- OpenMPI 4.1.6 with CUDA support, Kokkos 4.7.1 CUDA+Serial, CUDA 13.0
- One bound CPU core and one host thread per rank; all BLAS thread counts are 1
- CUDA-aware MPI is positively reported by the provider and LAMMPS did not
  print its GPU-aware fallback warning

All runs completed with exit status zero, one neighbor rebuild, and zero
dangerous builds. Final 6.5-A directed neighbor-list entries span only
99,817,762 to 99,817,768 across all six runs. The final energy span is
0.00500 eV total (5.0e-9 eV/atom), and the pressure span is 0.00167 bar.
Median sampled resident GPU memory is 999 MiB for one rank and 977 MiB/GPU
for two ranks. These samples are physical residency, not the larger managed
allocation/capacity estimate.

## Strong-scaling result

Primary timings are from the 20-step LAMMPS loop.

| MPI ranks / GPUs | us/atom/step samples | Median us/atom/step | CV | Speedup | Efficiency |
|---:|---|---:|---:|---:|---:|
| 1 | 7.037050, 7.049850, 7.027900 | **7.037050** | 0.157% | 1.0000x | 100.00% |
| 2 | 3.573840, 3.572005, 3.571540 | **3.572005** | 0.034% | **1.9701x** | **98.50%** |

The median two-rank time is 0.05348 us/atom/step above ideal halving of the
one-rank median. LAMMPS attributes a median 99.92% of the two-rank step to
`Pair` and about 0.05% to `Comm`, so its standard timing table alone hides the
main communication cost.

## Internal MPI share

`compute symmetrix/timing` synchronizes the pair denominator and reports the
critical rank. Values below are medians in us/atom/pair-evaluation.

| Ranks | Critical pair | Noncommunication | H1 communication | H1 share | Forward | Reverse | Comm max/mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.033343 | 6.989376 | 0.043955 | 0.625% | 0.022483 | 0.021471 | 1.0000 |
| 2 | 3.569805 | 3.515203 | 0.056502 | **1.582%** | 0.032623 | 0.023866 | 1.0478 |

The one-rank value is not network traffic: periodic self-swaps still execute
the packing and unpacking path. Using matched repetition deltas relative to
ideal halving, the median pair-path scaling shortfall is 0.05503 us/atom/eval:

- 0.03452 us/atom/eval, or 62.7%, comes from the larger H1 communication path;
- 0.02052 us/atom/eval, or 37.3%, comes from noncommunication work.

Forward time is the noisy component (19.6% CV across two-rank samples), while
reverse time has 0.25% CV. Total two-rank step time remains stable, so this is
small rank-correlated timing variation rather than a throughput instability.

The domain split also duplicates boundary state. Setup has 1,266,535 total
features on one rank. Two ranks each have 500,000 owned atoms and 229,885
ghost/features beyond their owned atoms, for 1,459,770 total features across
the node, 15.3% more than the one-rank case. This explains part of the
noncommunication shortfall even though the directed neighbor workload is
unchanged.

## Nsight investigation

Nsight Systems 2025.3.1 traced CUDA, NVTX, OS runtime, and OpenMPI for a short
two-rank run with two warmup and three measured steps. Profiler elapsed time is
not used as a performance result. The trace contains seven pair evaluations
including setup evaluations.

The four Symmetrix communication kernels have 42 launches each, or six spatial
swaps per evaluation. Their aggregate GPU time is 284.13 ms on rank 0 and
284.00 ms on rank 1, about 40.6 ms/evaluation/rank. Compared with the profiled
critical H1 path of 48.55 ms/evaluation, device pack/unpack accounts for about
84% of the hidden-state communication interval. The kernels are balanced
across ranks; they are not the source of rank asymmetry.

Each rank sends and receives about 11.156 GB over the complete seven-evaluation
trace. The largest MPI message is 439,623,680 bytes, exactly 107,330 feature
packets at the current 4,096-byte double-packet ABI. Blocking `MPI_Send` totals
173 ms on rank 0 and 331 ms on rank 1, while `MPI_Wait` totals only 4.0 ms and
0.66 ms. The remaining variance therefore appears on the blocking send side,
not as a long receive wait. Since CUDA-aware MPI is active over NV12 and no
host-staging fallback occurred, host staging is not the limiting path.

CUDA API summaries are dominated by `cudaEventSynchronize` (about 22.5 s per
rank across the complete trace), but that time primarily waits for ordinary
model kernels and must not be counted as communication overhead. The targeted
kernel and pair timers give the useful attribution: boundary pack/unpack and
packet volume dominate the small scaling loss; pure MPI waiting does not.

The practical optimization order for larger rank counts is therefore:

1. reduce or compact the 4,096-byte-per-feature H1 packet and boundary volume;
2. reduce the six pack/unpack swaps or overlap their blocking send path where
   LAMMPS's communication dependencies allow it;
3. address ghost-feature duplication before tuning generic MPI wait behavior.

At two ranks the achieved 98.50% efficiency does not justify a broader MPI
transport change.

## Identities and evidence

- Source checkout: `23ae5144e02cac0eab97c775d1f8f3d243c0275e`; the remote
  timing and Kokkos pair sources match the checkout hashes
- LAMMPS executable SHA256:
  `55d06587fbb9fff64aef1ccf534373f54ac2f0e1570d3bae75c9c93394c09220`
- Model SHA256:
  `3cc5b7641dbd4c9d766d6140661187c4fd484bae53d4aab5293faa5e76224414`
- SM80 JIT artifact SHA256:
  `24a47eecee8d0f89adf68c3aa0802df76309fa99b4846940702ca3ef7f10f3b8`
- Input and runners: `inputs/`
- Timing logs: `logs/scale-1m-r{1,2}-rep{1,2,3}.log`
- Commands, exit statuses, and GPU samples: `results/`
- Nsight reports and SQLite exports: `profile/scale-1m-r2-rank-{0,1}.{nsys-rep,sqlite}`

The installed Nsight collector initially left `.qdstrm` streams because its
bundled importer lacked `libdw.so.1`. The matching Ubuntu Noble
`libdw1t64=0.190-1.1ubuntu0.1` package was downloaded and extracted only under
`profile/libdw-runtime`; no system package was installed. The recovered
standard reports and SQLite exports are retained with the raw streams.
