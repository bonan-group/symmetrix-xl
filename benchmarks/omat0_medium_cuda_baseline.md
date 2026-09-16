# OMAT-0 medium float32 CUDA baseline

Date: 2026-07-24; fused-path update: 2026-07-25

## Configuration

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- Driver: 610.43.02
- Native build: CUDA 13.3.73, Kokkos CUDA, `BLACKWELL120`, SpheriCart CUDA
- PyTorch: 2.13.0+cu130
- MACE source: official revision `22f0809735bd4dd1deba80cf8e16f89913a35ff4`
- cuEquivariance: 0.10.0 (`cuequivariance`, `cuequivariance-torch`, `cuequivariance-ops-torch-cu13`)
- Checkpoint SHA-256: `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`
- Compact Symmetrix JSON SHA-256: `9f69ea29c0f0f25b0d3af7a485529af06cb93ca0dca7617e401fbcfe70c7baa4`
- Model: standard OMAT-0 medium, 89 elements, 128 channels, 6.0 A cutoff, `l_max=3`, `L_max=1`, ZBL enabled

The system is periodic wurtzite AlN (`a=3.112 A`, `c=4.982 A`) repeated 4, 6, and 10 times in each direction. These cells contain 256, 864, and 4,000 atoms and 23,296, 78,624, and 364,000 directed edges.

For the original table below, each fresh process builds the graph once,
performs 20 warmup energy-and-force forwards, and then records 20
CUDA-synchronized forwards. Stress and atomic stresses are disabled. Time
therefore excludes checkpoint loading and neighbor-list construction. VRAM is
the process-resident value reported by `nvidia-smi` after evaluation, not just
live tensor allocation. Later optimization sections state their own protocols.

The upstream measurements use `benchmarks/mace_torch_cuda_benchmark.py` with `--backend e3nn` or `--backend cueq` and `--repeat 4`, `6`, or `10`. Run each combination in a fresh process. The native measurements use `benchmarks/standard_mace_streamed_benchmark.py`; the table below was collected in fresh processes per mode so process-resident VRAM is directly comparable.

## Results

| Atoms | Backend | Median (ms) | Time/atom (us) | Process VRAM (MiB) |
|---:|---|---:|---:|---:|
| 256 | PyTorch/e3nn | 19.568 | 76.439 | 3,208 |
| 256 | PyTorch/cuEquivariance | 15.151 | 59.182 | 1,146 |
| 256 | Symmetrix legacy | 14.572 | 56.923 | 1,318 |
| 256 | Symmetrix R1 streamed | 14.747 | 57.604 | 1,090 |
| 256 | Symmetrix all streamed | 14.716 | 57.485 | 998 |
| 864 | PyTorch/e3nn | 65.348 | 75.634 | 9,046 |
| 864 | PyTorch/cuEquivariance | 12.065 | 13.964 | 2,088 |
| 864 | Symmetrix legacy | 37.131 | 42.976 | 2,672 |
| 864 | Symmetrix R1 streamed | 36.109 | 41.792 | 1,904 |
| 864 | Symmetrix all streamed | 35.797 | 41.432 | 1,596 |
| 4,000 | PyTorch/e3nn | 304.240 | 76.060 | 29,198 |
| 4,000 | PyTorch/cuEquivariance | 28.862 | 7.215 | 7,070 |
| 4,000 | Symmetrix legacy | 146.085 | 36.521 | 9,690 |
| 4,000 | Symmetrix R1 streamed | 140.105 | 35.026 | 6,134 |
| 4,000 | Symmetrix all streamed | 138.980 | 34.745 | 4,710 |

## Interpretation

cuEquivariance accelerates the upstream checkpoint path by 1.29x, 5.42x, and 10.54x at 256, 864, and 4,000 atoms. Its process VRAM is lower than PyTorch/e3nn by 2,062, 6,958, and 22,128 MiB.

For the production-like medium float32 model, native full streaming is essentially neutral at 256 atoms and becomes faster at larger sizes: 3.6% at 864 atoms and 4.9% at 4,000 atoms relative to native legacy. It reduces native resident VRAM by 320, 1,076, and 4,980 MiB. Native full streaming uses 148, 492, and 2,360 MiB less memory than cuEquivariance, but cuEquivariance is 2.97x and 4.82x faster at 864 and 4,000 atoms.

The 4,000-atom PyTorch/e3nn process completed, but its caching allocator emitted recoverable 2.61 GiB allocation-failure warnings near device capacity. Its measured peak reserved memory was 30,594 MiB.

PyTorch/e3nn and cuEquivariance agree closely: their total-energy differences are at most 1.95 meV and force-L2 differences at most 2.62e-6 eV/A across these cases. Native streamed modes agree closely with native legacy at float32 reduction precision. Cross-runtime total energy is more reduction-order-sensitive: the largest native-versus-PyTorch difference is 0.477 eV total at 4,000 atoms (119 micro-eV/atom), while force-L2 differs by 3.00e-4 eV/A.

## Initial Kokkos-CUDA reverse-Phi1 scheduling

Nsight Systems profiling at 864 atoms identified the second streamed reverse-Phi1 contraction as the dominant kernel: 22.273 ms per call and 68.1% of total device-kernel time. Its original launch assigned one team to each receiver atom, leaving only 864 teams to process 78,624 directed edges.

This initial CUDA policy assigned one team to eight consecutive directed edges
when the graph contained at most 100,000 edges. It built a compact
edge-to-receiver map for those graphs. Above the measured crossover it did not
allocate that map and dispatched the original node-owned kernel body unchanged.
OpenMP retained the original node-owned kernel.

| Atoms | Original all (ms) | Optimized all (ms) | Time/atom (us) | Change | Process VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|
| 256 | 14.716 | not accepted | not accepted | clock-sensitive | 998 |
| 864 | 35.797 | 32.965 | 38.154 | 7.9% faster | 1,598 |
| 4,000 | 138.980 | node-owned fallback | unchanged policy | no new schedule | 4,710 |

The accepted 864-atom result uses 40 warmups and 40 measured calls. A separate stable 20/20 run measured 32.116 ms, consistent with the improvement. The final 256-atom 40/40 process oscillated between 12.410 and 43.839 ms as GPU boost state changed, so its 18.985 ms median is not used. The final 4,000-atom fallback process likewise ran in a shifted clock regime (145.412 ms); because the retained large-graph hot loop and launch policy are source-identical to the baseline, this is recorded as environmental variability rather than an optimization result.

Rejected experiments at this stage were one team per edge (31.930 ms at 864
atoms but 142.466 ms at 4,000 atoms), eight edges per team globally (32.116 and
140.822 ms), 64 edges per team at large size (144.255 ms), and a schedule
branch inside the hot edge loop (144.933 ms). The subsequent fused kernel below
changes the large-graph crossover by reducing radial work and atomic traffic.

## Fused float32 CUDA reverse path

The Phase 46 optimization uses the eight-edge launch without an edge-count
limit for qualified shapes with at most 16 harmonics, at most 16 paths, and at
least twice as many raw coupling rows as harmonics. It fuses force and
source-feature adjoint work in reverse Phi1. Each channel now accumulates all
contributions for each of the model's 16 harmonics locally before emitting
atomics. Reverse Phi1 and reverse A0 use model-precision force reductions in
CUDA float32 builds; non-CUDA builds retain double accumulation.

Matched 864-atom acceptance measurements use the production OMAT-0 medium
float32 model, 20 warmups, and 10 measured energy-and-force evaluations in each
fresh process.

| Runtime | Median (ms) | Time/atom (us) | Process VRAM after (MiB) | Peak allocated (MiB) |
|---|---:|---:|---:|---:|
| Native control, `all` | 33.706 | 39.012 | 1,598 | not measured |
| Reviewed fused native, run 1 | 11.860 | 13.726 | 1,598 | not measured |
| Reviewed fused native, run 2 | 12.015 | 13.906 | 1,598 | not measured |
| Exact-final-build sanity | 11.947 | 13.828 | 1,598 | not measured |
| PyTorch/cuEquivariance | 12.452 | 14.412 | 2,088 | 1,393.4 |

The three reviewed medians are 64.4-64.8% below the fresh native control and
3.5-4.8% below the matched cuEquivariance median. All pass the predeclared
15.565 ms comparability gate. Native resident VRAM is unchanged at 1,598 MiB
before and after the optimization and is 490 MiB below cuEquivariance. For the
first reviewed process, host RSS was 254.2 MiB before model evaluation,
1,027.9 MiB after evaluation, and 1,031.8 MiB at peak; GPU process memory was
0 MiB before CUDA initialization and 1,598 MiB after evaluation.

The reviewed 864-atom result is `-6406.387709 eV` with force L2
`3.237951 eV/A`; the control is `-6406.387977 eV` and `3.237934 eV/A`.
The complete focused streamed-edge suite passes all eight cases across
serial/CUDA and float32/float64. The exact penultimate-R1 direct execution experiment was
not retained because it missed both performance and memory acceptance criteria.

At 256 atoms, the reviewed fused path measures `5.823 ms` (`22.745 us/atom`)
with `998 MiB` resident VRAM, versus the original streamed-native `14.716 ms`
and recorded cuEquivariance `15.151 ms`. Its ten samples span
`5.764-6.224 ms`.

### 4,000-atom follow-up

The first current-build measurement retained the historical `100,000`-edge
dispatch threshold, so this 364,000-edge graph used the node-owned fallback.
The second measurement removed that threshold for the fused kernel while
retaining it for the older unfused edge-owned implementation. Both used 20
warmups and 10 measured calls in fresh processes.

| 4,000-atom path | Median (ms) | Time/atom (us) | Range (ms) | Process VRAM (MiB) |
|---|---:|---:|---:|---:|
| Thresholded node-owned fallback | 127.372 | 31.843 | 127.017-128.201 | 4,710 |
| Reviewed unrestricted fused reverse Phi1 | 45.214 | 11.304 | 44.916-46.733 | 4,712 |
| Recorded PyTorch/cuEquivariance | 28.862 | 7.215 | not recorded | 7,070 |

Removing the inherited limit reduces current-build latency by 64.5% at a cost
of 2 MiB resident VRAM, consistent with retaining the 364,000-entry receiver
map. The reviewed fused result is 67.5% below the original `138.980 ms`
streamed-native baseline, although it remains 1.57 times slower than the
recorded cuEquivariance result.

For the reviewed unrestricted fused run, GPU process memory was `0 MiB` before
CUDA initialization and `4,712 MiB` after evaluation. Host RSS was `254.1 MiB`
before evaluation, `1,027.8 MiB` after, and `1,057.3 MiB` at peak. Energy was
`-29659.200556 eV`, force L2 was `6.966963 eV/A`, and maximum absolute force
was `0.110167 eV/A`. Against the same evaluator in legacy mode, the total-energy
difference is `0.176 meV`, maximum force-component difference is
`1.20e-5 eV/A`, and force L2 difference is `2.75e-4 eV/A`.

## CPU results at 256 atoms

The CPU host is an AMD Ryzen 9 9950X3D2 with 16 physical cores and 32 hardware threads. The Release CPU build enables Kokkos OpenMP+Serial and disables CUDA. OpenMP placement is `close` on `cores`, and BLAS is pinned to one thread. At the time of this historical table, native serial supported only float64; Kokkos used float32. Each mode was measured in a fresh process with 20 warmups and 20 measured calls. Native serial float32 support and newer timings are recorded below.

| Backend | Threads | Mode | Dtype | Median (ms) | Time/atom (us) | RSS after/peak (MiB) |
|---|---:|---|---|---:|---:|---:|
| Native serial | 1 | legacy | float64 | 713.245 | 2,786.114 | 2,639.0 |
| Native serial | 1 | R1 streamed | float64 | 920.750 | 3,596.680 | 2,183.1 |
| Native serial | 1 | all streamed | float64 | 972.356 | 3,798.266 | 2,109.6 |
| Kokkos OpenMP | 1 | legacy | float32 | 1,217.556 | 4,756.079 | 849.1 |
| Kokkos OpenMP | 1 | R1 streamed | float32 | 1,512.273 | 5,907.318 | 713.2 |
| Kokkos OpenMP | 1 | all streamed | float32 | 1,571.435 | 6,138.418 | 711.3 |
| Kokkos OpenMP | 2 | legacy | float32 | 622.995 | 2,433.576 | 848.1 |
| Kokkos OpenMP | 2 | R1 streamed | float32 | 767.216 | 2,996.939 | 711.3 |
| Kokkos OpenMP | 2 | all streamed | float32 | 797.383 | 3,114.776 | 712.9 |
| Kokkos OpenMP | 8 | legacy | float32 | 187.954 | 734.195 | 850.2 |
| Kokkos OpenMP | 8 | R1 streamed | float32 | 216.446 | 845.491 | 711.3 |
| Kokkos OpenMP | 8 | all streamed | float32 | 217.134 | 848.179 | 711.3 |
| Kokkos OpenMP | 16 | legacy | float32 | 120.231 | 469.654 | 849.7 |
| Kokkos OpenMP | 16 | R1 streamed | float32 | 136.304 | 532.438 | 712.7 |
| Kokkos OpenMP | 16 | all streamed | float32 | 130.536 | 509.904 | 712.5 |

Every CPU process started at approximately 68.3 MiB RSS. At 16 threads, scaling relative to one-thread Kokkos is 10.13x/11.10x/12.04x for legacy/R1/all. Full streaming reduces RSS by approximately 137 MiB versus Kokkos legacy but costs 8.6% median latency at 16 threads. The 16-thread results are noisier than the lower thread counts: sample ranges are 114.672-125.202, 127.727-147.914, and 119.853-150.150 ms for legacy/R1/all.

## Commit-pinned R1 versus all CPU/CUDA rerun

This rerun measures the final format-v2 default implementation and unrestricted
fused CUDA path from runtime commit
`fd5dd4a468e8e53eadca3f157fee1ac2e017b973`. The compact model SHA-256 remains
`9f69ea29c0f0f25b0d3af7a485529af06cb93ca0dca7617e401fbcfe70c7baa4`.
Both CPU and CUDA extensions were rebuilt from that commit before measurement.

Each backend/mode combination ran in a separate fresh process. A single
evaluator was reused across the increasing 256/864/4,000-atom graphs so model
loading and compact radial materialization occurred once and remained outside
the timed region. The benchmark driver records this explicitly through
`--reuse-evaluator` and embeds the runtime hash through `--source-commit`.
CUDA and 16-thread OpenMP used 20 warmups and 10 measured calls. Native serial
used 3 warmups and 10 measured calls. All results are medians of evaluator-only
energy-and-force calls; graph construction, model loading, and ASE result
assembly are excluded. OpenMP placement was `close` on physical cores, with
BLAS fixed at one thread.

For CUDA, resident memory is process VRAM reported by `nvidia-smi`. For CPU it
is process RSS. The CPU high-water values are reported separately in the last
column. `R0+R1` is retained edge-radial storage, not total process memory.

| Backend | Dtype | Threads | Atoms | Mode | Median (ms) | Time/atom (us) | R0+R1 (MiB) | Resident after (MiB) | CPU peak RSS (MiB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Kokkos CUDA | float32 | 1 | 256 | r1 | 7.069 | 27.615 | 91.0 | 1,090.0 | n/a |
| Kokkos CUDA | float32 | 1 | 256 | all | 5.745 | 22.440 | 0 | 998.0 | n/a |
| Kokkos CUDA | float32 | 1 | 864 | r1 | 16.276 | 18.838 | 307.1 | 1,908.0 | n/a |
| Kokkos CUDA | float32 | 1 | 864 | all | 11.803 | 13.661 | 0 | 1,600.0 | n/a |
| Kokkos CUDA | float32 | 1 | 4,000 | r1 | 63.799 | 15.950 | 1,421.9 | 6,140.0 | n/a |
| Kokkos CUDA | float32 | 1 | 4,000 | all | 47.001 | 11.750 | 0 | 4,716.0 | n/a |
| Kokkos OpenMP | float32 | 16 | 256 | r1 | 123.340 | 481.798 | 91.0 | 713.8 | 713.8 |
| Kokkos OpenMP | float32 | 16 | 256 | all | 122.149 | 477.145 | 0 | 713.6 | 713.6 |
| Kokkos OpenMP | float32 | 16 | 864 | r1 | 399.283 | 462.133 | 307.1 | 1,633.9 | 1,635.9 |
| Kokkos OpenMP | float32 | 16 | 864 | all | 392.622 | 454.424 | 0 | 1,326.7 | 1,327.5 |
| Kokkos OpenMP | float32 | 16 | 4,000 | r1 | 1,912.163 | 478.041 | 1,421.9 | 5,862.3 | 5,888.4 |
| Kokkos OpenMP | float32 | 16 | 4,000 | all | 1,874.615 | 468.654 | 0 | 4,470.4 | 4,496.1 |
| Native serial | float64 | 1 | 256 | r1 | 922.691 | 3,604.261 | 182.0 | 2,180.4 | 2,180.4 |
| Native serial | float64 | 1 | 256 | all | 970.422 | 3,790.711 | 0 | 2,109.8 | 2,110.0 |
| Native serial | float64 | 1 | 864 | r1 | 3,119.533 | 3,610.570 | 614.3 | 4,143.2 | 4,538.5 |
| Native serial | float64 | 1 | 864 | all | 3,285.986 | 3,803.225 | 0 | 3,544.0 | 3,945.2 |
| Native serial | float64 | 1 | 4,000 | r1 | 14,479.668 | 3,619.917 | 2,843.8 | 14,063.5 | 15,335.7 |
| Native serial | float64 | 1 | 4,000 | all | 15,265.533 | 3,816.383 | 0 | 11,215.9 | 12,425.6 |

On CUDA, `all` reduces median time relative to `r1` by 18.7%, 27.5%, and
26.3% at 256, 864, and 4,000 atoms, while saving 92, 308, and 1,424 MiB of
process VRAM. On 16-thread OpenMP, `all` is 1.0%, 1.7%, and 2.0% faster and
saves 0.2, 307.2, and 1,391.9 MiB of final RSS. For both production float32
backends, `all` dominates `r1` in this matrix.

CUDA sample ranges do not overlap between modes at any size: even the slowest
`all` call is faster than the fastest `r1` call. OpenMP ranges are also
separated at 864 and 4,000 atoms. The 256-atom OpenMP ranges overlap, so its
1.0% median difference should be treated as noise-level; memory remains tied
within 0.2 MiB there.

Native serial float64 retains the latency tradeoff that motivates keeping the
`r1` option: `all` is 5.17%, 5.34%, and 5.43% slower because it recomputes the
first-interaction radial values in the reverse pass. In return it saves 70.6,
599.2, and 2,847.6 MiB of final RSS. Thus `all` remains the correct automatic
format-v2 default, while `r1` remains useful as a native-CPU latency/memory
compromise and as a diagnostic mode.

OpenMP and native serial `r1`/`all` energies agree at roundoff. CUDA reduction
ordering produces total-energy differences of 0.112, 0.268, and 1.080 meV,
the largest being 0.270 micro-eV/atom. Focused streamed-mode tests separately
qualify force equivalence for native, OpenMP, and CUDA execution.

After each CUDA process had serialized its complete JSON record, Python shutdown
printed `Kokkos::Cuda ERROR: Failed to call Kokkos::Cuda::finalize()` and still
returned status 0. Kokkos emits this destructor diagnostic when its CUDA scratch
allocation state exists at process teardown without an explicit finalize call;
it occurred after the measured work and does not indicate a failed benchmark
kernel or incomplete output.

## Native serial float32 template result

This result uses the native float32 template implementation based on commit
`fd5dd4a468e8e53eadca3f157fee1ac2e017b973` plus the uncommitted Phase 56
working tree. The model hash and host are unchanged. Both precision runs used
compact-v2 `all`, one native CPU thread, one OpenBLAS thread, 3 warmups, and 10
measured evaluator calls. The 256-atom result used a separate fresh process;
one evaluator was reused from 864 to 4,000 atoms. Model loading and graph
construction are outside the timed region. The binding reported 4-byte
evaluator scalars for float32 and 8-byte scalars for float64.

| Atoms | Dtype | Median (ms) | Time/atom (ms) | RSS before (MiB) | RSS after (MiB) | Peak RSS (MiB) |
|---:|---|---:|---:|---:|---:|---:|
| 256 | float64 | 341.592 | 1.334346 | 1,606.9 | 2,121.1 | 2,121.3 |
| 256 | float32 | 294.744 | 1.151343 | 1,428.5 | 1,686.8 | 1,687.2 |
| 864 | float64 | 1,152.732 | 1.334180 | 2,121.1 | 3,550.3 | 3,922.7 |
| 864 | float32 | 1,000.348 | 1.157810 | 1,686.8 | 2,297.0 | 2,550.4 |
| 4,000 | float64 | 5,489.842 | 1.372461 | 3,550.3 | 11,256.0 | 12,465.8 |
| 4,000 | float32 | 4,687.257 | 1.171814 | 2,297.0 | 6,161.8 | 6,744.6 |

Float32 is 1.16x faster at 256 atoms, 1.15x at 864 atoms, and 1.17x at 4,000
atoms. At 256 atoms it lowers post-evaluation RSS by 434.3 MiB (20.5%). At 864
and 4,000 atoms it lowers post-evaluation RSS by 1,253.3 MiB (35.3%) and
5,094.2 MiB (45.3%), respectively. Peak RSS falls by 1,372.3 MiB (35.0%) and
5,721.2 MiB (45.9%) at the two release-gate sizes. These pass the
release floors of no more than a 5% slowdown, at least 25% lower resident RSS,
and at least 20% lower peak RSS. They do not reach the aspirational 1.25x speedup
target, so additional native reverse-kernel vectorization remains useful tuning.

Against native float64, float32 total-energy error is `+7.330e-7`,
`-1.057e-6`, and `+5.571e-7 eV/atom` at 256, 864, and 4,000 atoms,
respectively. Maximum absolute and RMS force-component errors are reported for
the same three sizes after a direct paired evaluation: `6.78e-6`/`2.09e-6`,
`5.82e-6`/`1.48e-6`, and `7.56e-6`/`1.73e-6 eV/A`.
