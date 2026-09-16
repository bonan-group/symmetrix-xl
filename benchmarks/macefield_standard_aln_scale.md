# MACEField versus standard MH-0 on AlN

## Decision

The generated M0/R0-v2 path is the best qualified policy for MACEField on both
Kokkos OpenMP and CUDA. Standard MH-0 also benefits from generated M0/R0-v2;
on CUDA, its experimental M1 tile-32 recomputation policy is marginally faster
and substantially smaller at large scale.

MACEField costs 3.8-6.0% on 16-core OpenMP and 12.7-30.9% on CUDA relative to
the fastest standard-MACE policy. At n30, the apples-to-apples retained-M1 GPU
memory overhead is 622 MiB (2.1%). Standard MACE can additionally remove both
M1 polynomial tensors through recomputation, reducing n30 sampled GPU memory by
14,980 MiB; that policy is not currently admitted for MACEField.

## Configuration

- Source revision: `d36bcad7c7ec6e061e00aa18210e8ec69793526a` plus the current worktree changes.
- Models: fine-tuned MACEField current contract and its MH-0 base current contract.
- Cells: periodic wurtzite AlN, 32 atoms (`2x2x2`), 5,324 atoms (`11x11x11`),
  and n30 with 108,000 atoms (`30x30x30`).
- Precision and workload: float32 energy and forces; MACEField uses the fixed
  electric field `(0.01, -0.02, 0.03)`.
- CPU: AMD Ryzen 9 9950X3D2, 16 cores/32 threads, 121 GiB RAM. Release Kokkos
  OpenMP build, 16 threads on CPUs 0-15, one hardware thread per physical core.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.43.02. Release Kokkos
  CUDA build with synchronous timing (`CUDA_LAUNCH_BLOCKING=1`).

Every small/medium result below is the median of three fresh processes. Each
process used two warmups and five samples on CPU, or ten warmups and 30 samples
on GPU. n30 is one fresh process with one measured evaluation after graph setup.

## Timings

| Backend | Atoms | MACEField policy | Field time | Standard policy | Standard time | Field overhead |
|---|---:|---|---:|---|---:|---:|
| OpenMP | 32 | generated M0/R0-v2 | 8.189 ms | generated M0/R0-v2 | 7.729 ms | 5.96% |
| OpenMP | 5,324 | generated M0/R0-v2 | 1.518 s | generated M0/R0-v2 | 1.463 s | 3.79% |
| OpenMP | 108,000 | generated M0/R0-v2 | 32.035 s | generated M0/R0-v2 | 30.400 s | 5.38% |
| CUDA | 32 | generated M0/R0-v2 | 1.586 ms | generated M0/R0-v2 + M1 recompute | 1.212 ms | 30.89% |
| CUDA | 5,324 | generated M0/R0-v2 | 28.291 ms | generated M0/R0-v2 + M1 recompute | 22.861 ms | 23.75% |
| CUDA | 108,000 | generated M0/R0-v2 | 1.054 s | generated M0/R0-v2 + M1 recompute | 0.935 s | 12.72% |

The field cost is proportionally largest for the small CUDA case because fixed
launch overhead is a larger share of the short evaluation. It amortizes as the
cell grows. All repeated evaluations produced identical energy and force
summaries within each model/policy/backend process set. The two models are not
numerical oracles for one another because the MACEField checkpoint was
fine-tuned from MH-0.

## Memory

CPU values are peak process RSS. GPU values are sampled process allocations
reported by `nvidia-smi` after evaluation.

| Backend | Atoms | MACEField | Standard retained M1 | Difference | Standard M1 recompute |
|---|---:|---:|---:|---:|---:|
| OpenMP | 32 | 1,106 MiB | 1,219 MiB | -113 MiB | not screened |
| OpenMP | 5,324 | 1,908 MiB | 1,889 MiB | +20 MiB | not screened |
| OpenMP | 108,000 | 32,100 MiB | 31,474 MiB | +626 MiB | not admitted |
| CUDA | 32 | 834 MiB | 848 MiB | -14 MiB | 840 MiB |
| CUDA | 5,324 | 2,306 MiB | 2,284 MiB | +22 MiB | 1,544 MiB |
| CUDA | 108,000 | 30,438 MiB | 29,816 MiB | +622 MiB | 14,836 MiB |

Both models select generated M0 and R0-v2 with zero active M0 polynomial
tensors. Retained-M1 n30 uses 7,852,032,000 bytes for each of the two active M1
polynomial tensors. Standard M1 recomputation reduces both active counts to
zero; this accounts for the large capacity difference against the fastest
standard policy.

## OpenMP affinity correction

The first OpenMP campaign was invalid. The run-mode supervisor imported the
OpenMP extension before spawning workers. With `OMP_PROC_BIND` enabled, the
supervisor's master thread was pinned to its first place. Under the original
`OMP_PLACES=cores` setting that place was `{0,16}`, the two hardware threads of
physical core 0, and each subprocess inherited only that pair. A reproduction
with `OMP_PLACES=threads` inherited only `{0}`. The 16 OpenMP threads therefore
time-shared one physical core, matching the screenshot exactly.

The benchmark driver now imports Symmetrix and initializes Kokkos only inside
worker mode. The supervisor retains its complete affinity mask, and launches
use `taskset -c 0-15`, `OMP_PLACES=threads`, and `OMP_PROC_BIND=spread`. OpenMP's
environment report then lists `{0}` through `{15}` as distinct places. A worker
also rejects launch when its inherited CPU mask is smaller than
`OMP_NUM_THREADS`, and records the initial mask in successful results. The
invalid `openmp-*.json` artifacts without the `-corrected` suffix are excluded
from every table and decision in this report.

## Artifacts

- Generated direct R1 execution CUDA follow-up: `benchmarks/direct_medium_cuda_optimization.md`
- Driver: `benchmarks/macefield_standard_aln_scale.py`
- CPU policy screen: `benchmarks/.artifacts/macefield_standard_aln/openmp-policy-screen-corrected.json`
- CPU final: `benchmarks/.artifacts/macefield_standard_aln/openmp-small-medium-corrected.json`
- CPU n30: `benchmarks/.artifacts/macefield_standard_aln/openmp-n30-corrected.json`
- GPU policy screen: `benchmarks/.artifacts/macefield_standard_aln/gpu-policy-screen.json`
- GPU final: `benchmarks/.artifacts/macefield_standard_aln/gpu-small-medium-final.json`
- GPU n30: `benchmarks/.artifacts/macefield_standard_aln/gpu-n30-final.json`

The adjacent `*-logs` directories contain fresh-process stdout and stderr for
each record, including explicit selector and failure provenance.
