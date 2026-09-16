# MACE-MH-1 streamed-edge CPU and CUDA benchmark

This benchmark compares the full-edge and block-streamed implementations of
the strict two-layer MACE-MH-1 fast path. Results include native energy and
analytic-force evaluation on a prebuilt graph.

## Configuration

- Base commit: `bab66d5d7a7ffad69a66c956bad2511b00a1b697`
- Worktree state: dirty with the streamed-edge implementation under test
- Model: `mace-mh-1-omat-pbe-universal-v3.json`
- Model SHA-256: `8384b616054cc4391531ca95af7f9b4737ce902628701faaf8e54878b5a79f00`
- System: 864-atom periodic wurtzite AlN, 78,624 directed edges
- CPU: AMD Ryzen 9 9950X3D2 16-Core Processor
- Backends: native serial on CPU `0`; Release Kokkos OpenMP on four pinned
  physical CPUs (`0-3`)
- Threading: one BLAS thread; four Kokkos/OpenMP threads for Kokkos
- Protocol: fresh process per mode, three warmups, ten measured repeats
- Streaming block size: 1024 directed edges

## Results

| Backend | Mode | Median (ms) | Median (ms/atom) | Final RSS (MiB) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|
| Serial | `legacy` | 6081.596 | 7.03888 | 2348.645 | 5913.129 |
| Serial | `all` | 5519.110 | 6.38786 | 480.770 | 1241.223 |
| Kokkos OpenMP (4) | `legacy` | 5306.958 | 6.14231 | 20295.641 | 20299.949 |
| Kokkos OpenMP (4) | `all` | 5186.027 | 6.00235 | 2614.703 | 2617.578 |

For serial, `all` is 9.25% faster, final RSS is lower by 1,867.875 MiB
(79.53%), and peak RSS is lower by 4,671.906 MiB (79.01%). For four-thread
Kokkos, `all` is 2.28% faster, final RSS is lower by 17,680.938 MiB (87.12%),
and peak RSS is lower by 17,682.371 MiB (87.11%). Every row reports the same
energy (`-6423.653034540963 eV`) and force norm
(`3.1685322548340 eV/A`) to the precision printed by the benchmark.

The benchmark driver records the base commit, dirty state, build cache, model
hash, individual timing samples, total and per-atom timing, process RSS, and
result checksums. Reproduce a row with:

```bash
SYMMETRIX_BENCHMARK_THREADS=THREADS \
SYMMETRIX_BENCHMARK_BLAS_THREADS=1 \
python benchmarks/mh1_serial_benchmark.py MODEL.json \
    --backend BACKEND --cpus CPU_LIST --atom-counts 864 \
    --streamed-edges MODE --warmups 3 --repeats 10 --evaluator-only
```

## CUDA Float32 Results

The strict `omat_pbe` head was requalified in Float32 on an NVIDIA RTX 5090
with driver 610.43.02, CUDA 13.3, and the same 864-atom/78,624-edge AlN graph.
The Symmetrix rows use base commit
`543def66b9195e45d27e12541ce1d886bbe5529f` with the uncommitted Float32
implementation, a 16,384-edge CUDA block, three warmups, and ten measured
evaluator calls. The upstream rows use the official checkpoint with MACE
0.3.15, PyTorch 2.13.0+cu130, and cuEquivariance 0.10.0.

| Implementation | Median (ms) | Median (ms/atom) | Process VRAM after setup (MiB) | Process VRAM after evaluation (MiB) | Peak allocated/workspace (MiB) |
|---|---:|---:|---:|---:|---:|
| Symmetrix `legacy` | 158.247 | 0.18316 | 506 | 10,584 | 8,955.813 workspace |
| Symmetrix `all` | 175.196 | 0.20277 | 506 | 3,574 | 1,866.000 workspace |
| PyTorch/e3nn | 95.762 | 0.11084 | 682 | 9,900 | 7,974.226 allocated |
| PyTorch/cuEquivariance | 20.793 | 0.02407 | 666 | 3,986 | 2,462.438 allocated |

`all` reduces post-evaluation process VRAM by 7,010 MiB (66.23%) and retained
edge workspace by 79.16% versus `legacy`. The ten `legacy` samples are stable
between 158.138 and 158.829 ms. The streamed samples are affected by device
scheduling variance: the protocol run spans 165.612-230.453 ms, while a
separate one-warmup smoke is 155.241 ms and a ten-warmup diagnostic remains
bimodal. The table retains the declared three-warmup/ten-repeat median rather
than selecting the fastest run.

Full-array CUDA Float32 versus Float64 comparison gives maximum differences of
`2.800e-6 eV/atom` in total energy, `6.162e-6 eV` in per-atom energy,
`2.180e-6 eV/A` in a force component, and `2.486e-7` in stress. At the same
block size, the precision-owned retained workspace is exactly halved from
3,913,285,632 bytes to 1,956,642,816 bytes.

An Nsight Systems trace of one warmup plus one measured `all` call records
1,228 CUDA kernel launches and 1,665 device synchronizations. E3 tensor-product
forward/reverse and E3 linear forward/reverse account for about 75% of GPU
kernel time. Consequently, the remaining cuEquivariance gap requires more
coarse-grained or fused equivariant kernels; further affine-only tuning or a
larger streamed block does not address the dominant cost. Tested 8,192- and
32,768-edge blocks regress to 254.163 and 207.055 ms respectively.

## CUDA equivariant-kernel optimization

The strict Float32 CUDA `all` path was subsequently optimized and committed as
`f9f020969a1cecc4625ccd074b36a20783db2436`. The tracked source was clean at
that commit; generated build trees and local planning files remained untracked.
The build uses CUDA 13.3.73, Kokkos CUDA for `BLACKWELL120`, KokkosKernels with
cuBLAS, and the same RTX 5090/driver 610.43.02. The system, graph, model hash,
three warmups, ten measured calls, explicit device fences, Float32 precision,
and 16,384-edge block are unchanged.

| Implementation | Median (ms) | Median (ms/atom) | Process VRAM before/after (MiB) | Precision workspace (MiB) |
|---|---:|---:|---:|---:|
| Original synchronized scalar `all` | 174.718 | 0.202220 | 506 / 3,574 | 1,866.00 |
| Optimized Symmetrix `all` (`f9f0209`) | 62.133 | 0.071913 | 506 / 3,184 | 1,477.06 |
| PyTorch/cuEquivariance 0.11.0 | 22.823 | 0.026416 | 0 / 3,616 | 2,456.90 peak allocated |

The optimized path is 2.81x faster than the synchronized scalar control, a
64.4% latency reduction. Process VRAM falls by 390 MiB (10.9%), and total
precision-owned workspace falls by 388.94 MiB (20.8%) even after including
the new 59.06 MiB shared linear workspace. It remains 2.72x slower than the
fresh same-session cuEquivariance result, so it clears the plan's 2x
Symmetrix speed gate but not its 1.25x cuEquivariance stretch target.

The retained stages are shared packed-SGEMM storage for E3 linears, 128-thread
sample teams for all official tensor-product forward/reverse channel work,
parallel harmonic reverse reductions, CUDA execution-space-ordered internal
copies, and lifetime reuse of forward edge-message storage by reverse
edge-message adjoints. Float64, Kokkos CPU, and generic nonlinear layouts keep
their existing selection rules. Rejected controls include workspace-free
direct-tiled linears, split tensor instruction/channel kernels, adjacent-power
radial derivatives, CUDA team LayerNorm/SiLU, and a 20,480-edge block.

The optimized ten samples in milliseconds are:
`[62.153722, 62.110902, 62.108758, 62.120200, 64.423274, 65.991663,
63.740525, 62.128926, 62.136410, 62.128215]`. The matched cuEquivariance
samples are:
`[22.775048, 22.822446, 22.990168, 22.784306, 22.799584, 22.825472,
22.824000, 22.791510, 22.945676, 22.828027]`.

Final qualification covers 30 CUDA physics, primitive, lifecycle, resizing,
species, finite-difference, and workspace tests; six direct OpenMP tests; and
CUDA memcheck of the larger official tensor-product layer with zero errors.
The optimized evaluator reports `e3_linear_backend="packed_gemm"`,
`selected_e3_linear_backend(864)="packed_gemm"`,
`tensor_product_backend="official_kokkos"`, and
`tensor_product_execution_backend="official_cuda_team"`. It retains
1,486,880,768 edge-workspace bytes, 61,931,520 linear-workspace bytes, and
1,548,812,288 total precision-workspace bytes.

## Generated direct execution ownership qualification

The Phase-9 worktree based on `19125d5` adds compact active-receiver forward
ownership, measured output-irrep batching, and physical input-irrep source
kernels. The graph, model, Float32 precision, RTX 5090, 16,384-edge blocks,
single host thread, and synchronized evaluator-only timing contract are
unchanged. The production batch policy was selected from budgets 7, 8, 10,
and 16; budget 8 was fastest in both ten-sample screening and a separate
20-sample confirmation.

| Implementation | Median (ms) | Range (ms) | Retained edge workspace (MiB) | Process VRAM before/after (MiB) |
|---|---:|---:|---:|---:|
| Phase-8 generated direct execution | 39.745 | not retained | 234.74 | not retained |
| Phase-9 generated direct execution, budget 8 | 35.395 | 35.219-38.155 | 234.74 | 506 / 1,974 |
| Matched current `all` | 40.617 | 39.970-42.806 | 1,378.00 | 506 / 3,140 |
| Retained PyTorch/cuEquivariance 0.11.0 | 22.823 | 22.775-22.990 | not comparable | 0 / 3,616 |

Generated direct execution is 12.9% faster than matched `all`, 11.0% faster than Phase 8,
and retains 5.87x less edge workspace than `all`. It remains 1.55x slower than
the retained cuEquivariance checkpoint, so the required same-build `all` gate
passes while the 1.25x stretch target remains open.

Nsight Systems attributes `16.570 ms` to the final generated plugin:
`6.084 ms` forward, `5.498 ms` source reverse, and `4.988 ms` edge reverse.
The corresponding Phase-8 plugin total was `20.527 ms`. Standalone `sm_120`
compilation reports no stack frame, local memory, or spills. Forward kernels
use 72/146 registers; source kernels use 72 registers for interaction 0 and
80/168 for interaction-1 scalar/vector groups. Hardware counter collection
was unavailable because the driver denied performance-counter access, so
`ptxas` resources and synchronized Nsight Systems timing are the retained
portable qualification evidence.

## Prepared lifecycle and compiled-product smoke

Phase 10 adds retained graph topology and a single evaluator stream. A
pre-final prepared-path run measured `32.351 ms`, versus `35.395 ms` for Phase
9, while retaining the same `234.74 MiB` edge workspace. The final path also
removes trusted product validation synchronizations, keeps ZBL on the evaluator
stream, and exposes the complete lifecycle counters in the benchmark JSON.

The first Phase-12 product block removes feature-major values and adjoints from
both compiled correlation-three products. At 864 atoms this eliminates exactly
`108 MiB` of retained product scratch. The remaining product workspace is
`16.875 MiB`, and complete precision-owned workspace is now
`310.677 MiB`, including edge, shared linear, tensor, and product storage. The
compiled reverse kernel uses 40 registers and zero stack/local memory in
Float32 (`cuobjdump`, `sm_120`).

A short final smoke produced samples
`[31.767, 31.694, 86.365, 41.675, 30.545] ms` with a `31.767 ms` median. This is
not a replacement qualification result: an unrelated process held `29.9 GiB`
of the `32.6 GiB` GPU throughout the run, causing visible scheduling outliers
and a CUDA-finalization warning after the result was written. The sample is
retained as directional evidence that direct-layout fusion did not regress the
prepared path. A clean three-warmup/ten-repeat run and final Nsight capture are
still required on an idle GPU.

## Graph-wide layout and execution-policy milestone

Phase 15 moves generated forward and source reverse to one graph-wide call per
interaction, uses channel-contiguous `ir_mul` storage, and binds each generated
CUDA partition plan to the model fingerprint and compute capability. Forward
policy screening retained budget 8 for interaction 1: its kernel uses 128
registers, versus 168 for shared-16 and 255 for full fusion. All three variants
have zero stack frame and zero spill loads/stores on `sm_120`.

R1 reverse: source was then tested as a paired policy experiment. Both variants use
the same graph, extension, forward budget, cache/compiler path, ten warmups,
15 synchronized evaluator samples, and 331.743 MiB edge workspace:

| Source policy | Median (ms) | Minimum (ms) | Interaction-1 CSR traversals per source/channel tile |
|---|---:|---:|---:|
| Grouped input irreps | 26.468 | 25.862 | 2 |
| Fully fused input irreps | 27.591 | 27.049 | 1 |

Full source fusion is 4.2% slower despite removing one CSR traversal. It halves
the number of interaction-1 source owners, and the lost parallelism dominates
the saved index/control traffic. The production `auto` policy therefore keeps
direct execution's grouped input schedule when multiple input irreps are present; full
fusion remains a validated explicit generator variant for future targets.
Both source kernels use 168 registers with no stack or spills.

The grouped milestone is 0.65x the retained matched `all` latency of 40.617 ms
and 1.16x the retained cuEquivariance latency of 22.823 ms, while retaining
about 4.15x less edge workspace than `all`. Three approximately 53 ms scheduling
outliers remain in each paired sequence, so the absolute median is directional;
the policy decision is retained because the paired fast clusters and medians
agree. The benchmark records the complete forward/source plans and combined
variant identity for later clean qualification.

## Complete generated conditioning milestone

The next architecture block extends the generated boundary around the
paper-aligned compact-`phi` tensor product. For each non-identity interaction,
one receiver-owned program now computes the exact pair-conditioned prefix and
density, and one edge-owned reverse program recomputes local activations and
emits radial/cutoff adjoints. Learned parameters and pre-folded species
contributions remain runtime inputs. Generic MLP tapes and explicit conditioned
contribution rows are absent from generated execution; parameter gradients
remain intentionally out of scope.

A fresh-cache RTX 5090 run used ten warmups and 15 synchronized evaluator-only
samples on the same 864-atom, 78,624-edge graph. It built the CUDA-v8 artifact,
reported 50 conditioning-forward and 50 conditioning-reverse launches across
25 evaluations, and retained zero conditioned-MLP workspace.

| Implementation | Median (ms) | Range (ms) | Retained edge workspace (MiB) |
|---|---:|---:|---:|
| Phase-15 graph-wide TPConv plus generic conditioners | 26.468 | 25.862 plus scheduling outliers | 331.743 |
| Complete literal generated conditioners | 47.744 | 47.673-47.791 | 115.743 |
| Retained matched `all` | 40.617 | 39.970-42.806 | 1,378.00 |

The tape-free architecture removes 216.0 MiB (65.1%) from the Phase-15 edge
workspace and uses 11.9x less edge workspace than `all`. The first literal
conditioner is 1.80x slower than the previous generated/generic hybrid and
1.18x slower than `all`; this is a reference implementation, not a qualified
performance replacement. The next work block targets the remaining node-side
layout/product program before conditioner kernel tuning, so the execution
architecture can be aligned end to end first.

## Persistent node-program CUDA-v4 checkpoint

ABI v4 keeps node features and adjoints in `ir_mul` across both interactions
and generates the complete linear, normalization/gate, correlation-three
product, skip, and readout forward/reverse program. The graph-wide ABI-v3
TPConv and conditioner programs remain as a second query in the same artifact.

The first literal CUDA lowering assigned one complete node program to each
thread and measured `2541.518 ms`. CUDA generator v10 replaced that owner with
one 128-thread cooperative block per node. Generator v13 keeps the cooperative
normalization, gate, product, and readout phases, but executes dense equivariant
linears graph-wide in 8-node by 32-channel tiles. On the same 864-atom,
78,624-edge graph it measures:

| Implementation | Median (ms) | Range (ms) | Node workspace (MiB) | Edge workspace (MiB) |
|---|---:|---:|---:|---:|
| Graph-wide tiled node and conditioner CUDA v4 | 27.247 | 27.205-27.261 | 215.888 | 115.743 |
| Unpadded graph-wide tiled CUDA v4 | 31.598 | 31.570-31.633 | 215.888 | 115.743 |
| Cooperative node and conditioner CUDA v4 | 67.839 | 67.787-67.934 | 215.888 | 115.743 |
| Cooperative node, serial receiver conditioner | 92.277 | 92.190-92.343 | 215.888 | 115.743 |
| Literal generated CUDA v4 | 2541.518 | 2535.802-2549.964 | 215.888 | 115.743 |
| Current matched `all` | 40.272 | 39.033-46.458 | 639.253 | 1190.000 |

The tiled path is 59.8% faster than the v11 cooperative checkpoint and 32.3%
faster than matched `all`, with unchanged workspace. Padding the shared weight
tile from 32x32 to 32x33 removes the reverse-transpose bank conflict: Nsight
reports tiled node reverse falling from `12.186 ms` to `7.807 ms`, while tiled
node forward remains `3.602 ms`. The final one-evaluation trace contains 147
kernels totaling `26.962 ms`: `11.408 ms` tiled node work, `11.978 ms` TPConv,
`2.574 ms` conditioning, and `1.001 ms` other work.

The official unperturbed result remains exactly `-6423.655267991864 eV`.
Against matched `all` on a deterministically perturbed 16-atom cell, maximum
absolute differences are `3.23e-7 eV` for total energy, `4.89e-6 eV` for
per-atom energies, `1.39e-6 eV/A` for force components, and
`3.14e-8 eV/A^3` for stress. At `1.19x` the retained cuEquivariance latency of
`22.823 ms`, v13 clears the 1.25x intermediate gate; graph-wide node and TPConv
work remain the two comparable endpoint targets.
