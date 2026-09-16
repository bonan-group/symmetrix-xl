# MACE Kokkos source partitioning

## Decision

Retain the reordered ten-source layout: four stable owners, three direct execution owners,
and three core-kernel owners. The layout provides physical incremental-build
isolation, exact symbol parity, and smaller critical CUDA objects. Listing the
owners approximately longest-first also improves the matched CUDA `-j4` wall
time from 74.435 s to 65.062 s.

The 12.6% CUDA improvement does not meet the original 25% promotion threshold
of approximately 55.8 s. This change is therefore retained as a source-layout
and incremental-isolation improvement with a useful scheduling gain, not as
having achieved the planned clean-build speedup. Further consolidation and
compiler experiments were frozen during wrap-up.

## Configuration

- Source commit before partitioning: `ee0f0f1abbaade550360c4bf7acda877cc612ee4`
- Target: `symmetrix_bindings`
- Generator: Ninja
- Supported measurement parallelism: four jobs
- Serial: Release, Kokkos Serial, GCC 15.2
- CUDA: Release, Kokkos CUDA+Serial, CUDA 13.3, `sm_120`, `nvcc_wrapper`
- CUDA host: 32 logical CPUs and an NVIDIA RTX 5090
- Original implementation SHA-256:
  `7c06953073b46e77f85fdd1523008d810cbb4e74b86c5b5f304719109be753f7`
- Original declaration-header SHA-256:
  `d9c03bebb2e2e525005e47ce41504fa989e7a8e8aa7be11b757eefbf83f52544`

CUDA comparisons rebuild only all MACE Kokkos evaluator owners plus required
archive and module links. The measurement driver verifies that no other object
or static-library work is pending, removes the named owner objects, retains
Ninja command hashes, records the appended Ninja timeline, and samples the
recursive process tree every 50 ms.

## Serial results

| Metric | Selector baseline | Ten-source layout |
|---|---:|---:|
| Build duration | 81.27 s wall | 83.195 s Ninja completion |
| User CPU | 288.99 s | Not recorded |
| System CPU | 8.71 s | Not recorded |
| Peak RSS | 1,421,652 KiB | Not recorded |
| Longest evaluator object | 23.329 s | 14.208 s |
| Static archive | 23,291,324 bytes | 24,255,514 bytes |
| Python extension | 13,463,552 bytes | 13,635,680 bytes |
| Normalized MACE Kokkos symbols | 1,510 | 1,510 |
| Symbol SHA-256 | `8966f9cc4d9c6fd6f712635e32aa57e36a8045ab37f6210fc249b112edd5ce54` | identical |

The longest evaluator object decreased by 39.1%. The static archive grew by
4.1% and the stripped extension by 1.3%; exact symbol parity shows that this is
object and archive layout overhead rather than an API expansion. Total Serial
build time is effectively neutral because every physical owner still parses
substantial template dependencies.

| Selector owner | Duration |
|---|---:|
| runtime | 6.115 s |
| evaluate | 3.170 s |
| response | 12.262 s |
| model | 14.199 s |
| direct execution | 23.329 s |
| kernels | 20.951 s |

| Retained owner | Duration |
|---|---:|
| runtime | 6.102 s |
| evaluate | 3.132 s |
| direct execution lifecycle | 9.088 s |
| direct execution analysis | 5.053 s |
| direct execution execution | 13.748 s |
| H1/Phi1 | 9.251 s |
| first interaction | 10.862 s |
| response | 12.313 s |
| second interaction | 8.252 s |
| model | 14.208 s |

No retained Serial evaluator object exceeds 60 seconds or 35% of build time.
The optional five-owner direct execution and core refinements are not justified by these
results.

## CUDA results

| Metric | Six-owner selector | Unordered ten-owner | Reordered ten-owner |
|---|---:|---:|---:|
| Wall time | 74.435 s | 72.088 s | 65.062 s |
| Ninja completion | 74.435 s | 72.088 s | 64.989 s |
| Aggregate owner compiler time | 196.684 s | 239.660 s | 240.273 s |
| Longest owner | kernels, 63.527 s | first interaction, 42.260 s | first interaction, 42.261 s |
| Final one-owner idle tail | about 27.9 s | 8.709 s | 3.541 s |
| User CPU | Not recorded | 230.08 s | 231.110 s |
| System CPU | Not recorded | 10.27 s | 9.861 s |
| Peak process-tree RSS | Not recorded | Not recorded | 5,848,088 KiB |
| Peak single-process RSS | Not recorded | 1,670,312 KiB | 1,662,540 KiB |

The unordered split increased aggregate owner compiler work by 21.9%. At four
jobs its theoretical work bound was already 59.915 s before linking, so it
could not reach the 25% target without reducing aggregate work. Its critical
path also moved from the selector's kernels object to a model object queued
late in the build.

CMake emits the owner object edges in source-list order for this graph. Ninja
started the initial four equal-priority owners in that order and selected each
subsequent owner in order as a slot became available. Reordering the sources
approximately longest-first reduced wall time by 7.026 s versus the unordered
layout and reduced its final idle tail by 59.3%. This is an empirical property
of the current single-target graph, not a general CMake or Ninja guarantee.

| Reordered owner | Start | Duration | Finish |
|---|---:|---:|---:|
| first interaction | 0.006 s | 42.261 s | 42.267 s |
| response | 0.006 s | 35.698 s | 35.704 s |
| direct execution execution | 0.006 s | 27.962 s | 27.968 s |
| H1/Phi1 | 0.006 s | 24.856 s | 24.862 s |
| model | 24.863 s | 25.479 s | 50.342 s |
| direct execution lifecycle | 27.968 s | 20.653 s | 48.621 s |
| second interaction | 35.704 s | 19.284 s | 54.988 s |
| direct execution analysis | 42.267 s | 18.525 s | 60.792 s |
| runtime | 48.621 s | 15.712 s | 64.333 s |
| evaluate | 50.342 s | 9.843 s | 60.185 s |

The retained aggregate work bound is 60.068 s at four jobs. Its 64.989 s Ninja
completion is close enough to that bound that additional scheduling changes
alone cannot meet the original target.

## Qualification

| Backend | Build coverage | Runtime coverage | Result |
|---|---|---|---|
| Serial | Release FP32/FP64 link and exact 1,510-symbol comparison | 23 focused behavioral cases across both precisions | Passed |
| OpenMP | Release FP32/FP64 link | 7 focused cases passed; 3 bitwise checks differed by at most `8.9e-16` with multiple threads and passed with one thread | Passed with documented reduction-order variation |
| CUDA | Release CUDA+Serial link for `sm_120`; 5,934 normalized symbols | RTX 5090 sentinel, NVRTC, AOT, batched BLAS, mandatory R1 NVRTC, 17 focused cases, and 2 M1 admission cases | Passed with 3 environment-sensitive assertions classified below |
| HIP | Not run | No AMD device on this host | Unavailable |

The CUDA normalized symbol inventory has SHA-256
`5ff5f37c6a0dc9cfbb30a62021d586a7677aaf9bf1d80d7502d3e6b0565b0a53`.
One focused assertion requested `generated_sector` while a ready device plugin
correctly selected `generated_all`; the selection predicate is byte-equivalent
to the selector baseline. Two FP64 assertions assumed M1 tile 32 although
Blackwell scratch admission selected tile 16. The device-appropriate automatic
admission and precise-fallback test passes for both precisions.

## Deferred experiments

- The proposed eight-owner consolidation was not implemented or measured.
- Matched selector and retained-layout `-j8` runs were not performed, so no
  eight-job scaling or memory claim is made.
- NVCC phase timing and CUDA 13.3 `--split-compile` remain separate follow-up
  experiments. First interaction was not split further.
- Two early measurement-driver trials rebuilt all 105 targets because Ninja's
  recursive clean and command-hash behavior were initially handled
  incorrectly. Their 192.823 s and 191.093 s wall times are rejected as
  non-comparable; the driver now preserves Ninja command hashes and cleans only
  explicit owner outputs.

The detailed ownership and rebuild contract is documented in
`docs/mace_kokkos_source_layout.md`. The reproducible driver is
`benchmarks/mace_kokkos_build_partitioning.py`. Raw timelines and JSON records
remain local under `benchmarks/.artifacts/mace_kokkos_source_partitioning/` and
are not part of the maintained repository.
