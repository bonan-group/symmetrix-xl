# Streamed-edge milestone benchmark

Date: 2026-08-22

## Compared implementations

- Current release candidate: `streamed-edge` at
  `6233ae8245a9d4f02a4b7c70ebcb70547533d8ff`, plus the reviewed generic
  receiver-map fix in the uncommitted release diff.
- Original-repository native baseline: `origin/develop` at
  `952e56e06499e416cccbf1fd702b0609d7a1052c`.
- Original PyTorch implementation: `mace-torch 0.3.15`, measured with both
  e3nn (`enable_cueq=False`) and cuEquivariance (`enable_cueq=True`) CUDA
  execution. The editable source checkout was clean at
  `136e4ef040d7c51a5b051a7a609ffb1f29307478`; e3nn remains the original
  numerical reference.

All runs used the same OMAT-0 medium checkpoint, SHA-256
`d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`,
selected checkpoint head `default`, float32, and energy, forces, and stress.
The benchmark passes the head explicitly and rejects silent MACE fallback.
Current Symmetrix used
compact JSON SHA-256
`cd0fde722f21125a7640f4e8a9c45495d36b6a1463a379583ed9c71e949b89b2`.
The structure was deterministically perturbed periodic wurtzite AlN
(`a=3.112 A`, `c=4.982 A`, seed 20260822).

Each timing process performed three warmups and ten CUDA-synchronized or
CPU-complete measured ASE calculator calls. The timed boundary includes result
assembly for energy, forces, and stress, but excludes calculator construction,
model conversion, JIT compilation/cache lookup, and initial neighbor-list
construction. Reported performance is the median in `us/atom`. Process RSS is
the operating-system high-water mark; GPU memory is the process-resident value
sampled through `nvidia-smi` after evaluation.

All CUDA and CPU rows pin Torch intra-op/inter-op, OpenMP, Kokkos, OpenBLAS,
MKL, and BLIS to one thread. CPU rows are additionally bound to logical CPU 0
before Python initialization. The CUDA policy makes host graph and result work
identical between e3nn and cuEquivariance.

Current direct and generic execution used a 0.5 A Verlet skin. Consequently
they processed the skin-expanded candidate graph at an effective 6.5 A cutoff;
inactive direct candidates contribute zero through the exact 6.0 A radial
cutoff. Develop and PyTorch rebuilt the exact 6.0 A graph. This gives the
current implementation more edges and is therefore a conservative timing
comparison, but the graph policies are not identical.

## Build and machine

- Host: AMD Ryzen 9 9950X3D2, 16 physical cores, Linux 7.0.0.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.43.02.
- Python 3.12.13, PyTorch `2.13.0+cu130` with CUDA runtime 13.0 and cuDNN 9.2,
  NumPy 2.5.2, ASE 3.29.0.
- cuEquivariance reference: `cuequivariance 0.11.1`,
  `cuequivariance-torch 0.11.1`, and
  `cuequivariance-ops-torch-cu12 0.11.1`.
- Current OpenMP extension SHA-256: `8cb83c25...17d6`; Release, OpenMP only,
  OpenBLAS, `SYMMETRIX_HOST_ARCH=x86-64-v3`, `Kokkos_ARCH_NATIVE=OFF`.
- Develop OpenMP extension SHA-256: `55077a86...9352`; Release, OpenMP,
  revision-default `Kokkos_ARCH_NATIVE=ON`.
- Current CUDA extension SHA-256: `d5e4e953...9307`; Release, CUDA 13.3,
  Kokkos CUDA+Serial, `BLACKWELL120`, device SpheriCart.
- Develop CUDA extension SHA-256: `65a3e2c4...26d1`; Release, CUDA 13.3,
  Kokkos CUDA+Serial, `BLACKWELL120`, host SpheriCart.

The pinned develop SpheriCart CUDA source does not compile with CUDA 13.3 due
to `dim3` assignment errors. The unmodified develop default is host
SpheriCart, so that is the valid original-revision baseline. Its transfers are
part of the measured historical implementation, not a current-branch
regression.

## One-thread CPU

All CPU processes were bound to logical CPU 0 before Python initialization.
Kokkos/OpenMP and every BLAS runtime were set to one thread.

| Implementation | Mode | Atoms | Cutoff / skin / effective (A) | Directed edges | us/atom | Peak RSS (MiB) | Speedup vs PyTorch |
|---|---|---:|---|---:|---:|---:|---:|
| PyTorch MACE/e3nn | original | 864 | 6.0 / 0.0 / 6.0 | 78,624 | 12,610.511 | 7,333.0 | 1.00x |
| Origin/develop Kokkos | materialized | 864 | 6.0 / 0.0 / 6.0 | 78,624 | 4,767.777 | 3,784.5 | 2.64x |
| Current Kokkos | generic | 864 | 6.0 / 0.5 / 6.5 | 97,762 | 5,615.376 | 555.5 | 2.25x |
| Current Kokkos | direct, low memory | 864 | 6.0 / 0.5 / 6.5 | 97,762 | 262.962 | 558.2 | 47.96x |

Direct low-memory is 18.13x faster than origin/develop and 21.35x faster than
the compiler-free current generic path. Generic is retained as a portability
and correctness fallback, not the optimized CPU production path.

## CUDA

| Implementation | Mode | Atoms | Cutoff / skin / effective (A) | Directed edges | us/atom | Process VRAM (MiB) | Speedup vs PyTorch |
|---|---|---:|---|---:|---:|---:|---:|
| PyTorch MACE/e3nn | original | 864 | 6.0 / 0.0 / 6.0 | 78,624 | 105.247 | 9,104 | 1.00x |
| PyTorch MACE/cuEquivariance | accelerated original | 864 | 6.0 / 0.0 / 6.0 | 78,624 | 43.890 | 2,136 | 2.40x |
| Origin/develop Kokkos | materialized | 864 | 6.0 / 0.0 / 6.0 | 78,624 | 67.418 | 2,594 | 1.56x |
| Current Kokkos | generic | 864 | 6.0 / 0.5 / 6.5 | 97,762 | 8.107 | 856 | 12.98x |
| Current Kokkos | direct, low memory | 864 | 6.0 / 0.5 / 6.5 | 97,762 | 4.610 | 858 | 22.83x |
| PyTorch MACE/e3nn | original | 4,000 | 6.0 / 0.0 / 6.0 | 364,000 | 98.995 | 29,248 | 1.00x |
| PyTorch MACE/cuEquivariance | accelerated original | 4,000 | 6.0 / 0.0 / 6.0 | 364,000 | 31.052 | 7,136 | 3.19x |
| Origin/develop Kokkos | materialized | 4,000 | 6.0 / 0.0 / 6.0 | 364,000 | 60.326 | 9,612 | 1.64x |
| Current Kokkos | generic | 4,000 | 6.0 / 0.5 / 6.5 | 452,342 | 6.648 | 1,566 | 14.89x |
| Current Kokkos | direct, low memory | 4,000 | 6.0 / 0.5 / 6.5 | 452,342 | 4.008 | 1,118 | 24.70x |

At 4,000 atoms, direct is 15.05x faster than origin/develop while its sampled
VRAM is 8.60x smaller. It is 24.70x faster than PyTorch/e3nn with 26.16x less
sampled VRAM. Against PyTorch/cuEquivariance, direct is 7.75x faster with
6.38x less sampled VRAM. Direct is 1.65x faster and 448 MiB smaller than
current generic.

cuEquivariance materially improves the PyTorch deployment: 2.40x at 864 atoms
and 3.19x at 4,000 atoms relative to e3nn, while reducing sampled VRAM by
4.26x and 4.10x. The current direct path remains faster because its optimization
boundary also covers graph preparation reuse, explicit reverse scheduling,
intermediate lifetimes, compact geometry, and device force/stress reduction.

A separate generic smoke crossed the former receiver-map lifetime threshold:
1,372 atoms, 6.0 A cutoff, 0.5 A skin, 6.5 A effective cutoff, and 155,164
candidate directed edges. Its finite energy, forces, and stress result completed
at 7.636 us/atom for the single measured qualification call.

## Numerical comparison

The implementations evaluate the same extracted learned operator, but FP32
reduction order and the skin-expanded candidate graph differ. Errors below are
against PyTorch/e3nn for the final measured call.

| Atoms | Implementation | Energy abs. error (eV/atom) | Max force component (eV/A) | Max stress component (eV/A^3) |
|---:|---|---:|---:|---:|
| 864 | PyTorch/cuEquivariance | 1.70e-6 | 4.77e-6 | 2.61e-8 |
| 864 | Origin/develop | 4.45e-6 | 1.84e-4 | 4.83e-7 |
| 864 | Current generic | 5.13e-6 | 6.80e-5 | 2.42e-7 |
| 864 | Current direct low-memory | 5.18e-6 | 6.60e-5 | 2.33e-7 |
| 4,000 | PyTorch/cuEquivariance | 9.77e-7 | 4.69e-6 | 3.35e-8 |
| 4,000 | Origin/develop | 1.85e-6 | 1.67e-4 | 3.89e-7 |
| 4,000 | Current generic | 2.41e-6 | 9.80e-5 | 2.71e-7 |
| 4,000 | Current direct low-memory | 2.43e-6 | 1.01e-4 | 2.51e-7 |

## Profiler attribution

Nsight Systems 2026.1.3 captured one warmup plus three measured 4,000-atom
calls in separate, profiler-only processes. Profiler timings are not used in
the tables above.

Origin/develop spends 65.0% of GPU kernel time in materialized
`reverse_Phi1`, 83.782 ms per evaluation. `reverse_A0` adds 10.3%, and forward
`compute_Phi1` adds 7.3%. This is the edge-tensor materialization/reverse
backbone targeted by streamed execution.

Current direct distributes work across specialized stages:

| Direct kernel family | GPU kernel share | Approx. ms/evaluation |
|---|---:|---:|
| Generated factorized R1 reverse | 21.1% | 3.146 |
| M1 reverse with recomputation | 12.1% | 1.813 |
| Standard R0 coordinate reverse | 10.5% | 1.573 |
| Generated factorized R1 forward | 9.4% | 1.410 |
| Standard R0 forward | 8.3% | 1.239 |
| A1 forward | 6.5% | 0.978 |
| A1 reverse | 5.6% | 0.841 |

The direct trace contains four instances of each principal evaluator kernel
and no single materialized edge reverse. Device force reduction is about
0.044 ms/evaluation. Nine device stress reductions together are about
0.071 ms/evaluation, so returning stress does not reintroduce full pair-force
or edge-vector host transfers.

Nsight Compute 2026.2.1 profiled one generated direct reverse launch. It ran in
3.00 ms with 64 threads/block, 1,360 blocks, 128 registers/thread, 33.33%
theoretical and 32.34% achieved occupancy. L2 throughput reached 87.44% with a
98.58% L2 hit rate, while DRAM throughput was only 5.55% and SM throughput
39.08%. Scoreboard waits on L1/TEX dependencies account for 61.9% of the 13.4
cycles between issued instructions. The next kernel-level opportunity is
therefore reducing register pressure and improving/coalescing cache-resident
loads and stores, not increasing off-chip bandwidth.

## Reproduction

The maintained driver is `benchmarks/streamed_edge_release_benchmark.py`.
Representative current CUDA invocation:

```bash
OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
SYMMETRIX_BENCHMARK_PACKAGE_ROOT=/tmp/symmetrix-release-current-cuda/stage \
SYMMETRIX_JIT_CACHE=/tmp/symmetrix-release-jit-cache-current-cuda \
python benchmarks/streamed_edge_release_benchmark.py run \
  --implementation current --device cuda \
  --checkpoint /path/to/mace-omat-0-medium.model \
  --compact-model /tmp/symmetrix-release-omat0-medium-Al-N.json \
  --mode direct --low-memory --dtype float32 --head default --repeat 10 \
  --cutoff 6.0 --skin 0.5 --warmups 3 --samples 10 \
  --revision 6233ae8245a9d4f02a4b7c70ebcb70547533d8ff \
  --output result.json
```

For CPU, retain the same thread variables and invoke the process through
`taskset -c 0`. Change
`--implementation` to `develop` with its staged package root or to `pytorch`
for the framework controls. Add `--pytorch-backend cueq` for the accelerated
CUDA reference; the default is `e3nn`. Run implementations in fresh processes.

Raw evidence is ignored under
`benchmarks/.artifacts/streamed-edge-milestone-20260822/`. Key SHA-256 values:

- Executed benchmark driver:
  `ed5dd8b456829fa51f784e5efd17ac6e0579c219f5f6f8994177685f37b5fb79`.
  The release file was Ruff-formatted after timing and has SHA-256
  `12f35d42bc9f765f7fd1332e037cde4c726ad99244b430644aaaa331a7d849f1`;
  this was a formatting-only change, and its five focused tests pass.
- CPU comparison JSON: `b0e6a4f20a0d6d902a0fa7f44f03ccbd3282b85fc30b86fa4c5a503855334e95`.
- CUDA 864 comparison JSON: `30fa098dc49076adb97fdf27c101368b79799e494c8f644996fa482bf14de598`.
- CUDA 4,000 comparison JSON: `d535e331f24c695b84442c5240ca39cfee0d9b0c0f417524596973c77414385c`.
- CUDA cuEquivariance 864/4,000 raw JSON:
  `1d93acc5807102274d0c905f917ca7342b98b21f1ad3c695a3fd67165ce7383d`
  and `0b0efe29bd1747ab17fe5aa89437f5c3ea8189fcc75c807e050aa9042e63dc10`.
- Current/develop Nsight Systems reports: `75a06b41...978` and
  `b7086e41...035`.
- Current direct reverse Nsight Compute report: `21710d84...1d`.

The full hashes can be regenerated with `sha256sum` before external archival.
