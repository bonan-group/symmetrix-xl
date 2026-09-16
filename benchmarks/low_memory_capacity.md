# Factorized low-memory CUDA capacity

Date: 2026-08-17

## Scope

This benchmark compares the public factorized configuration with
`low_memory=False` and `low_memory=True`. It covers:

- OMAT-0 energy, forces, and stress; and
- MACEField energy, forces, stress, and polarization at an electric field of
  `(0.01, -0.02, 0.03) V/A`.

BEC and polarizability are not requested. Every MACEField worker reports zero
`macefield_response_primal_reconstruction_count`, confirming that the
analytical-response path did not run.

Capacity is the largest cubic wurtzite-AlN repeat that completes in a fresh
process, with the next integer repeat confirmed to fail in two fresh processes.
Each successful worker performs the public ASE calculation once for setup, one
warmup, and two measured calls. Timing excludes model loading, JIT loading,
neighbor-list construction, and the first evaluation. Process GPU memory is
sampled from `nvidia-smi` every 50 ms and remains allocated after evaluation.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0
- Driver: 610.43.02
- CUDA toolkit: 13.3; Kokkos execution space: `Cuda`; precision: float32
- Python 3.12.13, ASE 3.29.0, NumPy 2.5.2, PyTorch 2.13.0+cu130
- Source baseline: `e5bfc5f9802ff4de94703afa5f3ab61a0076719d` plus the
  factorized low-memory implementation qualified in this record
- Extension SHA-256:
  `c71e4a18a3914bdc8d6572d10f0b3f8fe9c96978d25c63ad33080e794cd7fe5d`
- OMAT-0 medium factorized model SHA-256:
  `af3b41b1e0de93dc1b8ba946f71ae3cb84c16b7c7a48840529cddd32382380a7`
- MACEField Al/N `mp-dielectric` compact model SHA-256:
  `52dceeca5ed876bc12e82835574cbc83b85a5055f834560cbb3acd426be31fdf`
- Model cutoff: 6.0 A; neighbor skin: 0.5 A; effective cutoff: 6.5 A
- Structure: periodic wurtzite AlN, four atoms per primitive cell and 109
  directed candidate edges per atom at the effective cutoff

Both policies select NVRTC R1 (`jit_all` forward and `jit` reverse), standard
M0, R0 `v2_edge16`, and M1 recomputation. The false baseline requests M1
`automatic`, which selects recomputation. The true policy requests
recomputation, `reuse-adjoints-v1`, and `unit-f32-radius-f64-v1` together.

## Capacity

| Model | `low_memory` | Largest success | Directed edges | First failure | Failure | Capacity gain |
|---|---:|---:|---:|---:|---|---:|
| OMAT-0 | false | 219,488 (`n38`) | 23,924,192 | 237,276 (`n39`) | CUDA OOM, 463.4 MiB allocation | reference |
| OMAT-0 | true | 389,344 (`n46`) | 42,438,496 | 415,292 (`n47`) | CUDA OOM, 405.6 MiB allocation after the local index fix | 1.774x (+77.4%) |
| MACEField | false | 202,612 (`n37`) | 22,084,708 | 219,488 (`n38`) | CUDA OOM, 1.675 GiB allocation | reference |
| MACEField | true | 340,736 (`n44`) | 37,140,224 | 364,500 (`n45`) | CUDA OOM, 133.5 MiB allocation | 1.682x (+68.2%) |

Before the local vendor patch, the OMAT-0 `n47` failure was reproducible as a
CUDA illegal address in two fresh processes. Follow-up analysis below identifies
a signed 32-bit offset overflow in the bundled Sphericart CUDA gradient kernel.
After widening the edge-derived offsets, the same workload reproducibly reaches
the expected allocator boundary instead: a 405.6-MiB allocation exceeds the
remaining device capacity.

## Boundary timing

Timing is the median of the two fresh-process medians at each largest-success
boundary. These rows have different atom counts and characterize the boundary;
they are not a direct policy speed comparison.

| Model | `low_memory` | Atoms | ms/call | us/atom | Sampled GPU MiB |
|---|---:|---:|---:|---:|---:|
| OMAT-0 | false | 219,488 | 1,066.16 | 4.858 | 31,856 |
| OMAT-0 | true | 389,344 | 1,987.44 | 5.105 | 31,914 |
| MACEField | false | 202,612 | 1,057.85 | 5.221 | 30,634 |
| MACEField | true | 340,736 | 1,764.56 | 5.179 | 30,730 |

## Matched-size comparison

The matched comparison uses `n32`: 131,072 atoms and 14,286,848 directed
edges. Each value is one fresh-process median of two measured calls.

| Model | `low_memory=False` | `low_memory=True` | Time change | GPU memory saved |
|---|---:|---:|---:|---:|
| OMAT-0 | 4.462 us/atom, 19,388 MiB | 4.441 us/atom, 11,322 MiB | -0.47% | 8,066 MiB (41.6%) |
| MACEField | 5.108 us/atom, 20,114 MiB | 5.081 us/atom, 12,306 MiB | -0.53% | 7,808 MiB (38.8%) |

At this graph size the OMAT-0 bundle aliases 2,617,245,696 bytes of forward
state and stores 285,736,960 bytes of compact geometry. The corresponding
MACEField values are 2,348,810,240 and 285,736,960 bytes. The complete native
geometry workspace decreases by 2,914,516,992 bytes for both models.

The false/true numerical-summary differences at `n32` are:

| Model | Energy per atom | Force L2 summary | Maximum-force summary | Maximum stress | Maximum polarization |
|---|---:|---:|---:|---:|---:|
| OMAT-0 | 6.28e-7 eV | 5.77e-5 eV/A | 5.21e-6 eV/A | 1.20e-7 eV/A^3 | n/a |
| MACEField | 6.76e-7 eV | 9.07e-5 eV/A | 1.13e-6 eV/A | 9.03e-8 eV/A^3 | 3.40e-9 |

These are consistent with the qualified float32 reduction-order tolerance.

## FP64 low-memory qualification

The FP64 implementation was qualified on the same RTX 5090 with CUDA 13.3
using extension SHA-256
`998f836b25fbba7370161c3659644e6dfb23bcc69c14aa120c578da48072afe2`.
Each fresh worker used `n20`: 32,000 atoms, 3,488,000 directed edges, a 6.0 A
model cutoff, 0.5 A skin, and 6.5 A effective cutoff. Timing is the median of
seven calls after three warmups. The MACEField property set remains energy,
forces, stress, and polarization without analytical response.

| Model | Precision | `low_memory` | Geometry | us/atom | Sampled GPU MiB |
|---|---|---:|---|---:|---:|
| OMAT-0 | float32 | true | unit-f32/radius-f64 | 4.171 | 3,404 |
| OMAT-0 | float64 | true | Cartesian float64 | 33.938 | 5,844 |
| OMAT-0 | float64 | false | Cartesian float64 | 34.667 | 9,624 |
| MACEField | float32 | true | unit-f32/radius-f64 | 4.717 | 3,574 |
| MACEField | float64 | true | Cartesian float64 | 35.142 | 6,188 |
| MACEField | float64 | false | Cartesian float64 | 35.897 | 9,842 |

FP64 low memory saves 3,780 MiB (39.3%) for OMAT-0 and 3,654 MiB
(37.1%) for MACEField relative to FP64 full retention. It is 2.1% faster for
both models in this matched test. Relative to FP32 low memory, FP64 uses 1.72x
and 1.73x GPU memory and is 8.14x and 7.45x slower, respectively. This large
throughput difference reflects consumer-GPU FP64 hardware and should not be
generalized to datacenter GPUs with stronger FP64 units.

FP64 low-memory and full-retention summaries agree within `7e-15` for the
reported energies, force norms/maxima, stresses, and polarization. FP64 low
memory reports zero compact-geometry bytes and selects FP64 generated R1,
standard R0, and standard M0 execution.

## Boundary investigation

The boundary rows compare different graph sizes and are not evidence that the
low-memory policy is slower. Controlled fresh-process reruns used three warmups
and seven measured calls:

| Repeat and policy | Atoms | Directed edges | Sampled GPU MiB | us/atom |
|---|---:|---:|---:|---:|
| `n32`, false | 131,072 | 14,286,848 | 19,388 | 4.402 |
| `n32`, true | 131,072 | 14,286,848 | 11,322 | 4.382 |
| `n38`, false | 219,488 | 23,924,192 | 31,856 | 4.687 |
| `n38`, true | 219,488 | 23,924,192 | 18,362 | 4.459 |
| `n45`, true | 364,500 | 39,730,500 | 29,928 | 4.476 |
| `n46`, true | 389,344 | 42,438,496 | 31,914 | 5.062 |

At the ordinary matched `n32` size, low memory is 0.46% faster. At matched
`n38`, it is 4.87% faster because the false policy is itself close to the
device-memory boundary. Low-memory timing remains flat through `n45`, then
increases by 13.1% per atom at `n46`.

Matched Nsight Systems traces explain that final step. Across three evaluations,
`n45` migrates 645 MB of unified memory host-to-device and 645 MB
device-to-host. At `n46`, those totals rise to 8,044 MB and 8,583 MB,
respectively, while aggregate `cudaStreamSynchronize` time rises from 4.459 to
5.419 seconds. With only about 0.7 GiB of reported device memory free, managed
pages are evicted and faulted back. The boundary slowdown is therefore a
near-capacity paging effect, not low-memory arithmetic overhead. Nsight tracing
changes absolute timing, so the uninstrumented seven-sample values above remain
the timing result.

The `n47` illegal address has a separate cause. With `l_max=3`, the spherical
harmonic gradient contains 48 values per directed edge. Its extent is
2,037,047,808 elements at `n46`, below `INT32_MAX`, and 2,172,807,744 elements
at `n47`, above it. Bundled Sphericart CUDA computes gradient offsets such as
`edge_idx * 3 * n_total` using signed `int`, so the offset wraps at `n47`.
Repeating `n47` with `CUDA_LAUNCH_BLOCKING=1` localizes the illegal address to
Sphericart's CUDA launch instead of the later A0 allocation/synchronization
where the asynchronous campaign first observed it.

A local vendor patch widens the Sphericart value, gradient, Hessian, backward,
and coordinate offsets while preserving its `int` sample-count ABI. A fresh
CUDA build with extension SHA-256
`b0d73c60654771c718564b8c4298754f1a93696d757a7433507b43587446b988`
changes the synchronous `n47` outcome to a clean 405.6-MiB CUDA allocation
failure. The capacity boundary therefore remains `n46`, now for the expected
memory limit rather than invalid addressing. A seven-sample `n32` rerun measures
4.3825 us/atom, versus 4.3818 before the patch, a 0.02% difference.

## Post-merge refresh

The campaign was repeated after merging through source commit `7632c54` with a
fresh CUDA 13.3/sm120 extension, SHA-256
`d41e6c1eb9fb54f77d3dbb3b857fb0ddf30bda811d4fb37353a02f961a67373d`.
The GPU, model hashes, FP32 property sets, 6.0 A cutoff, 0.5 A skin, and 6.5 A
effective cutoff are unchanged. Capacity success and failure were each
confirmed in two fresh processes.

All four boundaries reproduce exactly:

| Model | `low_memory` | Largest success | Directed edges | First failure | Failure |
|---|---:|---:|---:|---:|---|
| OMAT-0 | false | 219,488 (`n38`) | 23,924,192 | 237,276 (`n39`) | CUDA OOM, 463.4 MiB allocation |
| OMAT-0 | true | 389,344 (`n46`) | 42,438,496 | 415,292 (`n47`) | CUDA OOM, 405.6 MiB allocation |
| MACEField | false | 202,612 (`n37`) | 22,084,708 | 219,488 (`n38`) | CUDA OOM, 1.675 GiB allocation |
| MACEField | true | 340,736 (`n44`) | 37,140,224 | 364,500 (`n45`) | CUDA OOM, 711.9 MiB allocation |

No worker reported a CUDA illegal address. The OMAT-0 `n46` success trials
measured 78.265 and 104.708 us/atom while holding 31,914 MiB. This is more
severe and more variable than the earlier managed-page boundary effect, so
`n46` remains capacity evidence only. The other boundary trials measured
5.277 us/atom for OMAT-0 full retention, 5.269 us/atom for MACEField full
retention, and 5.237 us/atom for MACEField low memory after taking the median
of the two fresh-process medians.

The primary matched-speed refresh uses `n32`, 131,072 atoms, 14,286,848
directed edges, three warmups, and seven measured calls:

| Model | `low_memory=False` | `low_memory=True` | Time change | GPU memory saved |
|---|---:|---:|---:|---:|
| OMAT-0 | 4.3843 us/atom, 19,388 MiB | 4.3345 us/atom, 11,322 MiB | -1.14% | 8,066 MiB (41.6%) |
| MACEField | 5.1904 us/atom, 20,064 MiB | 5.1029 us/atom, 12,256 MiB | -1.69% | 7,808 MiB (38.9%) |

OMAT-0 low memory differs from full retention by `6.28e-7 eV/atom`,
`6.26e-5 eV/A` in the force-L2 summary, `5.18e-6 eV/A` in the maximum-force
summary, and `1.18e-7 eV/A^3` in maximum stress. The corresponding MACEField
differences are `6.76e-7 eV/atom`, `1.11e-4 eV/A`, `1.15e-6 eV/A`,
`8.70e-8 eV/A^3`, and `3.85e-9` in polarization. Both low-memory workers
select generated R1, standard M0/R0, M1 recomputation, adjoint reuse, and FP32
compact geometry, while reporting zero analytical-response reconstruction.

The refresh artifacts are under the ignored
`benchmarks/.artifacts/low_memory_postmerge_20260817/` directory.

## Extreme-scale indexing correction (2026-08-29)

The historical RTX 5090 boundaries above remain valid for that 32 GB device,
but they must not be interpreted as indexing limits. A later A100-SXM4-80GB
qualification with the same OMAT-0-medium FP32 low-memory property contract,
6.0 A model cutoff, 0.5 A skin, and 6.5 A effective cutoff reached `n63`:
1,000,188 atoms and 109,020,492 directed edges. It succeeded twice at 9.607
and 9.612 us/atom. The adjacent `n64` case, 1,048,576 atoms and 114,294,784
expected edges, failed twice with a true 2 GiB CUDA allocation failure.

The corrected generated R1 and SpheriCart global offsets use 64-bit arithmetic,
and no post-fix extreme-size trial produced an illegal address. The maintained
driver now treats every illegal address as a correctness failure, accepts only
`cuda_oom` as a capacity upper bound, persists graph metadata before native
evaluation, samples total memory on the selected physical GPU, and verifies
memory recovery between fresh-process probes. Full evidence is in
`benchmarks/extreme_scale_indexing_20260829.md`.

## Retained-Y low-memory refresh (2026-08-29)

The capacity test was repeated after merging the retained-Y/direct-harmonic
low-memory work through source commit `cd98dc6af360b72d7ac17a41cfbf6ddcdef67aa0`.
The fresh CUDA 13.3 build used Kokkos CUDA plus Serial, CUDA SpheriCart, and
`Kokkos_ARCH_BLACKWELL120` on the same RTX 5090. The extension SHA-256 was
`094a3e304c11eaea00ff96686dab2bcf6093de2d02348c324859ae70da3dd3ca`.
The OMAT-0-medium Al/N model SHA-256 remained
`af3b41b1e0de93dc1b8ba946f71ae3cb84c16b7c7a48840529cddd32382380a7`.

The property contract remained FP32 energy, forces, and stress with a 6.0 A
model cutoff, 0.5 A skin, and 6.5 A effective cutoff. Every successful worker
selected direct generated R1 (`jit_all` forward and `jit` reverse), standard
M0, R0 `v2_edge16`, M1 recomputation, adjoint reuse, compact FP32-unit/F64-radius
geometry, and the generation-3 NVRTC artifact
`jit-r1-gen3-f32-6fdf3e63624a0e45`. The generated cubin SHA-256 was
`40994f57ba484cd31c58e604d1db1d3b7ed549ebbd996be18108b894aa2efdad`.

Two fresh `n51` workers succeeded and two adjacent `n52` workers failed:

| Result | Repeat | Atoms | Directed edges | us/atom | atoms/s | Sampled GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| Largest success, trial 1 | `n51` | 530,604 | 57,835,836 | 4.579 | 218,390 | 31,798 |
| Largest success, trial 2 | `n51` | 530,604 | 57,835,836 | 4.494 | 222,537 | 31,764 |
| First failure, trials 1 and 2 | `n52` | 562,432 | 61,305,088 | n/a | n/a | 32,144 |

Both `n52` failures occurred during the first native evaluation when Kokkos
could not allocate 1.073 GiB. Both were classified as CUDA allocator OOMs;
neither produced an illegal address. Device use recovered to 68 MiB after the
second failed worker. The two `n51` timings average 4.536 us/atom, or 220,444
atoms/s. Because `n51` is close to the device-memory boundary, it is capacity
evidence rather than the primary throughput result.

The new largest success is 36.3% above the previous low-memory `n46` result
(389,344 atoms) and 141.7% above the retained `n38` result (219,488 atoms).
Compared with the preceding low-memory implementation, retained-Y therefore
adds 141,260 atoms of demonstrated RTX 5090 capacity for this exact workload.

The matched throughput comparison used `n32`: 131,072 atoms and 14,286,848
directed edges. Each fresh worker used two warmups and three measured calls;
the retained result is the mean of controls before and after the low-memory
worker.

| Policy | us/atom | atoms/s | Sampled GPU MiB | Geometry workspace |
|---|---:|---:|---:|---:|
| Full retention | 4.125 | 242,397 | 19,794 | 7,381,450,912 B |
| Retained-Y low memory | 4.415 | 226,484 | 8,652 | 1,552,416,928 B |

Retained-Y low memory saves 11,142 MiB (56.3%) of sampled GPU use and
5,829,033,984 bytes (79.0%) of reported geometry workspace at `n32`. It is
7.03% slower than the bracketed retained controls on this large RTX 5090
workload. The result is a capacity-throughput tradeoff, not a no-regression
speed result, and should be profiled separately before making retained-Y the
default.

The low-memory and retained numerical summaries differ by `1.006e-6 eV/atom`
in energy, `7.13e-5 eV/A` in the force-L2 summary, `3.85e-6 eV/A` in the
maximum-force summary, and `1.45e-7 eV/A^3` in maximum stress. These are
summary differences rather than full-array maximum force errors and remain in
the established FP32 reduction-order range.

Raw records, the fresh build, and the private JIT cache were retained with the
qualification run.
The focused merged-path CUDA parity and transactional-policy test passed for
both FP32 and FP64. The source-level JIT/code-generation/benchmark suite passed
74 tests, and the capacity-driver integrity suite passed seven tests.

## Historical automatic harmonic storage qualification (2026-08-29)

This section records the intermediate harmonic-only selector and is
superseded by the whole-bundle qualification below. In particular, its
`low_memory=True` retained-gradient result still used the capacity state,
geometry, and readout policies; the current selector instead restores the
complete speed bundle whenever that bundle fits.

The low-memory bundle now requests automatic harmonic storage for each new
prepared graph. The native evaluator queries free and total device memory and
estimates the active Y-only and retained-gradient graph footprints from atom
count, directed-edge count, precision, channels, angular dimensions, retained
Phi1 width, geometry policy, and field state. Retained gradients are admitted
only when the estimate fits after reserving the larger of 5% total device
memory or 512 MiB. Explicit retained and Y-only policies remain available for
deterministic qualification.

The current-source CUDA 13.3 BLACKWELL120 extension SHA-256 was
`580f2a7c1b4f3273e7445810df8b06b04a18985d4c0b87565d07fa9f3a8b2116`.
The model, FP32 energy/forces/stress property contract, 6.0 A cutoff, 0.5 A
skin, 6.5 A effective cutoff, and generated artifact were unchanged.

| Repeat | Atoms | Directed edges | Selected harmonics | Y-only estimate | Retained estimate | us/atom | atoms/s | Sampled GPU MiB |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `n32` trial 1 | 131,072 | 14,286,848 | retained | 8,240,234,496 B | 11,154,751,488 B | 4.509 | 221,793 | 11,492 |
| `n32` trial 2 | 131,072 | 14,286,848 | retained | 8,240,234,496 B | 11,154,751,488 B | 4.510 | 221,705 | 11,428 |
| `n51` | 530,604 | 57,835,836 | Y-only | 33,358,012,272 B | 45,156,522,816 B | 4.573 | 218,655 | 31,886 |

The two `n32` decisions used 32.60-32.66 GB free of 33.71 GB total and a
1.685 GB reserve. The `n51` decision used 32.29 GB free; its 45.16 GB retained
estimate correctly forced Y-only, and the established 530,604-atom capacity
point remained successful without an illegal address.

A matched current-binary full-retention `n32` control measured 4.138 us/atom
and 19,756 MiB. Automatic retained-gradient low memory averaged 4.510 us/atom
and 11,460 MiB, saving 42.0% while running 9.0% slower. A separate explicit
Y-only run with the same low-memory component policies measured 4.447 us/atom,
1.4% faster than retained-gradient low memory. The device-memory query occurs
only during graph preparation and is outside measured steady state; this
result therefore identifies a current retained-gradient/SpheriCart performance
gap, not selector overhead. The automatic policy implements the intended
memory-capacity decision, but retained gradients should not yet be described
as universally faster on the merged source.

Focused CUDA tests forced both selector branches and compared energy, forces,
and stress in FP32 and FP64. Ordinary MACE and MACEField primal/analytical
response parity passed. The retained-minus-Y-only estimate exactly matched
204 B/edge in FP32 and 408 B/edge in FP64. Raw records and the JIT cache were
retained with the qualification run.

## Whole-bundle automatic policy qualification (2026-08-30)

The public meanings are now:

- `low_memory=False` selects the normal speed-oriented path and activates no
  capacity policy; allocator OOM is the expected boundary behavior.
- `low_memory=True` prioritizes capacity, but selects the same speed bundle
  when its complete graph estimate fits current device memory after reserve.
  Otherwise it selects the maximum-capacity bundle.

The final CUDA 13.3 BLACKWELL120 extension SHA-256 was
`c8c9b30e6b6d704c3183689f3688ce109a58fddfd4b7519f4113cbcefb8c351f`.
The RTX 5090 used driver 610.43.02 and had no competing compute process. The
OMAT-0-medium model SHA-256 remained
`af3b41b1e0de93dc1b8ba946f71ae3cb84c16b7c7a48840529cddd32382380a7`.
All runs used FP32 energy, forces, and stress, a 6.0 A model cutoff, 0.5 A
skin, and 6.5 A effective cutoff.

| Run | Atoms | Directed edges | Request | Selected bundle | us/atom | atoms/s | Process GPU MiB |
|---|---:|---:|---:|---|---:|---:|---:|
| `n32` baseline | 131,072 | 14,286,848 | false | disabled/speed baseline | 4.1534 | 240,766 | 19,432 |
| `n32` automatic | 131,072 | 14,286,848 | true | speed | 4.1622 | 240,258 | 19,432 |
| `n51` automatic | 530,604 | 57,835,836 | true | capacity-y-only | 4.6059 | 217,112 | 30,794 |

At `n32`, the speed estimate was 19,673,907,200 B and 31,046,536,397 B
was available after the 1,685,418,803 B reserve. The true request therefore
selected full MH-0 retention, retained readout, Cartesian geometry, and
retained harmonic gradients, exactly matching `low_memory=False`. Its timing
was 0.21% above the false baseline, with identical reported energy, force,
and stress summaries and identical process VRAM. The
device query and policy decision occur during graph preparation, outside the
steady-state measurements.

At `n51`, the speed estimate was 79,643,660,400 B and 31,012,981,965 B was
available after reserve, so the evaluator selected the 33,358,012,272 B
capacity-Y-only estimate. The estimate is intentionally conservative and can
exceed the post-reserve figure; in that case the evaluator still attempts the
smallest qualified bundle instead of rejecting a graph that may fit. It used M1/readout
recomputation, MH-0 adjoint reuse, compact FP32 geometry, retained Phi1, and
zero retained harmonic-gradient bytes. The run completed without illegal
address or allocator failure. Focused fresh-extension CUDA tests forced both
branches in FP32 and FP64 and passed ordinary-MACE energy/forces/stress parity
and MACEField primal/analytical-response parity.

Raw records and the private RTC cache were retained with the qualification run.

## Reproduction

The maintained driver is `benchmarks/low_memory_capacity.py`. Raw worker JSON,
stdout/stderr, the campaign report, controlled reruns, Nsight Systems reports,
and the preserved MACEField compact model are under the ignored
`benchmarks/.artifacts/low_memory_capacity_20260817/` directory.

```bash
SYMMETRIX_SOURCE_ROOT="$PWD" \
SYMMETRIX_EXTENSION=/path/to/cuda/symmetrix.cpython-312-x86_64-linux-gnu.so \
SYMMETRIX_JIT_CACHE="$PWD/benchmarks/.artifacts/low_memory_capacity_20260817/jit-cache" \
python benchmarks/low_memory_capacity.py campaign \
  --omat0-model /path/to/mace-omat-0-medium-factorized.json \
  --macefield-model /path/to/macefield-mp-dielectric-Al-N.json \
  --output-dir benchmarks/.artifacts/low_memory_capacity_20260817/campaign \
  --lower-repeat 20 --upper-repeat 32 --max-repeat 48 \
  --boundary-trials 2 --neighbor-skin 0.5 --warmups 1 --samples 2
```
