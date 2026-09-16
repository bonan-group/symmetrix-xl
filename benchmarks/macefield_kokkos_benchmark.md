# MACEField streamed-edge benchmark and support matrix

> **Historical record (superseded).** This report preserves the selector names
> and support decisions used for its 2026-08-14 measurements. The current
> public modes are documented in
> [`docs/reference/execution_support_matrix.md`](../docs/reference/execution_support_matrix.md);
> `streamed_edges="all"` is no longer accepted.

## `streamed_edges="all"` support matrix

| Backend | Model | `all` | Qualified precision | Energy/forces/stress | Polarization | Polarizability/BEC | Edge-memory behavior |
|---|---|---:|---|---:|---:|---:|---|
| native CPU | compact MACE | yes | float32, float64 | yes | n/a | n/a | fully streamed R0/R1 |
| native CPU | compact MACEField | yes | float32, float64 | yes | yes | yes, analytic | first order fully streamed; response calls transiently materialize R0/R1 |
| Kokkos OpenMP | compact MACE | yes | float32, float64 | yes | n/a | n/a | fully streamed R0/R1 |
| Kokkos OpenMP | compact MACEField | yes | float32, float64 | yes | yes | yes, analytic | first order fully streamed; response calls transiently materialize R0/R1 |
| Kokkos CUDA | compact MACE | yes | float32, float64 | yes | n/a | n/a | fully streamed R0/R1 |
| Kokkos CUDA | compact MACEField | yes | float32, float64 | yes | yes | yes, analytic | first order fully streamed; response calls transiently materialize R0/R1 |
| Kokkos HIP | compact MACE | yes | float32, float64 | yes | n/a | n/a | fully streamed R0/R1 |
| Kokkos HIP | compact MACEField | yes | float32, float64 | yes | yes | yes, analytic | first order fully streamed; response calls transiently materialize R0/R1 |

The MH-0 and MACEField rows above require the format-v2 compact, fixed-weight
radial representation. Format-v1 pair-spline MACE models reject `r1` and
`all`. Compatible format-v3 MH-1-family models support both streamed modes;
generic nonlinear format-v3 models remain legacy-only. The field-aware native
and Kokkos paths support float32 and float64 through the same analytic response
implementation instantiated at both precisions.

## Configuration

- Model: `MACEField-MH-0-omat-dielectric.model`, head `mp-dielectric`
- Checkpoint SHA-256: `f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2`
- Compact Al/N artifact: 256 spline points, 16,220,750 bytes
- Artifact SHA-256: `997512552ec2a19aeab1260fc8c88065c564c1e225e10e055ed0d1002d107da4`
- Structure: periodic wurtzite AlN, 91 directed neighbors per atom
- Electric field: `(0.01, -0.02, 0.03)`
- Precision: float64
- CUDA: 13.3, Kokkos `Cuda`, Blackwell `sm_120`
- CUDA timing: 20 warmups, 10 measured energy/force calls, median reported

The driver is `benchmarks/macefield_kokkos_benchmark.py`. GPU memory is the
memory attributed to the benchmark process by `nvidia-smi`. Host current and
high-water memory come from `/proc/self/status`.

## Field-aware compact geometry qualification (2026-08-15)

The float32 `unit-f32-radius-f64-v1` edge-geometry policy now supports the
MACEField primal, stress, polarizability, and BEC paths. The ASE calculator
retains topology-reference geometry on the device and updates current edge
geometry from atom positions. After the first topology/reference upload, a
warmed call copies positions and the three-component field rather than edge
vectors or radii.

This qualification used:

- NVIDIA GeForce RTX 5090, compute capability 12.0, driver 610.43.02;
- CUDA 13.3, Kokkos CUDA/Serial, `sm_120`, float32;
- extension SHA-256 `7962545786239996102e571c7e3604cbc76898308297b4696858a69ec17f8c47`;
- `MACEField-MH-0-omat-dielectric.model`, Al/N, `mp-dielectric`, extracted
  with current `execution_contracts`; compact artifact SHA-256
  `a89f4209517727c6cc203f66bc6c1e286c355d15cae2f4e0a7c1fdf6c515922e`;
- periodic 5,324-atom wurtzite AlN, model cutoff 6.0 A, skin 0.5 A,
  effective neighbor cutoff 6.5 A, 580,316 directed edges;
- field `(0.01, -0.02, 0.03)`, energy, forces, stress, polarizability, and
  BECs in each call;
- generated R1 (`jit_all`/`jit`), standard M0, R0 `v2_edge16`, M1
  recomputation with tile 16; and
- three fresh processes per policy, two warmups and five measured calls per
  process. The median of fresh-process medians is primary.

| Geometry policy | us/atom | ms/call | geometry workspace | sampled CUDA process | Relative time |
|---|---:|---:|---:|---:|---:|
| Cartesian float64 vectors | 211.12 | 1,124.03 | 299,826,568 B | 1,606 MiB | 1.000x |
| float32 unit vectors + float64 radii | 208.06 | 1,107.72 | 292,862,776 B | 1,600 MiB | 0.985x |

Compact geometry saves 6,963,792 bytes, exactly 12 bytes per directed edge,
and is 1.47% faster on this response-heavy workload. Its three process medians
were 1,107.01, 1,107.72, and 1,109.66 ms; Cartesian medians were 1,119.60,
1,124.03, and 1,125.79 ms.

The 5,324-atom summary differences between policies were `1.915e-3 eV` total
energy (`3.597e-7 eV/atom`), `1.472e-5 eV/A` in force-L2 summary,
`6.245e-7 eV/A` in maximum-force summary, `4.216e-4` in field adjoint,
`1.519e-6` in maximum-polarizability summary, and `9.915e-8` in maximum-BEC
summary. Focused tests compare complete force, stress, polarizability, and BEC
arrays at `1e-3` absolute tolerance and compare stress with centered cell
finite differences.

Nsight Systems 2026.1 traced one warmup and two measured public ASE calls. The
13,927,584-byte Cartesian reference edge array was uploaded once during setup.
Steady calls contained no 13,927,584-byte edge-vector, 6,963,792-byte compact
direction, or 4,642,528-byte radius H2D transfer. Each call uploaded only
127,776 bytes of atom positions plus 24 bytes of field. No full pair-force D2H
occurred during evaluation; the driver performed one 13,927,584-byte readback
after timing to report force summaries. Full BEC response still returns the
41,782,752-byte directed-edge field-force derivative to Python for atom
reduction; device BEC reduction remains separate work.

Nsight Compute 2026.2.1 successfully profiled one bounded launch of the
dominant analytical-response kernel. Nsight Systems measured that kernel at
233.1 ms per launch and 77.8% of GPU kernel time. Compute reported 85.76% SM
throughput, 3.01% DRAM throughput, 54.92% achieved occupancy, 70 registers per
thread, and no local-memory spill. The kernel is compute-bound and
register-limited; its 267.29 ms Compute duration includes ten replay passes and
is not used for end-to-end timing.

Raw timing, Systems, SQLite, and Compute reports are under the ignored
`benchmarks/.artifacts/field_compact_geometry/` directory. The reproducible
driver is `benchmarks/macefield_standard_aln_scale.py` with
`--response-routing public-native`.

### Factorized analytical-response topology correction (2026-08-16)

The factorized response previously looked only for the generic streamed-edge
receiver table. A prepared factorized graph instead owns
`execution_edge_receivers`, so CUDA fell back to the node-owned Phi1 and A0
reverse-tangent kernels even though the compatible edge-owned kernels were
already available. The corrected selector validates the existing factorized
receiver and direct-source schedule against the current graph and reuses the
receiver view without allocation. Source adjoints are still accumulated
atomically; this change does not introduce a source-owned response kernel.

The primary qualification retained the 5,324-atom workload, cutoff, skin,
effective cutoff, 580,316 directed edges, compact geometry, tile-16 M1
recomputation, field, response properties, and timing protocol above. It used
the same checkpoint (SHA-256
`f92e043aaf2cd8879919db8452503553fe7b608cb749d8d169dd96d4aa094aa2`)
and a current-contract extraction with SHA-256
`52dceeca5ed876bc12e82835574cbc83b85a5055f834560cbb3acd426be31fdf`.
The extraction hash differs from the earlier compact artifact because the
maintained execution-contract metadata evolved; the checkpoint and selected
`mp-dielectric` head are unchanged.
The timing and profiler extension SHA-256 was
`3f5623354e83de4d42f26fb82c98f3f7f837ac51484cb03721628985e79c09e0`.
The final source-ownership diagnostic correction was rebuilt as
`bc5ed474f1207888fd810a08303284d1c51ce350d61f0397954eb153e6aca17d`;
it does not change the profiled kernels.

| Factorized response | us/atom | ms/call | sampled CUDA process | Relative time |
|---|---:|---:|---:|---:|
| Generic node-owned fallback | 208.06 | 1,107.72 | 1,600 MiB | 1.000x |
| Correct factorized receiver topology | 74.327 | 395.716 | 1,600 MiB | 0.357x |

The three corrected fresh-process medians were 394.876, 395.716, and
396.155 ms/call. Every process reported 21 Phi1 fused launches, zero Phi1
generic launches, 21 A0 fused launches, and zero A0 generic launches across
two warmups and five measured calls. Schedule bytes were stable across repeated
same-token responses. A stale graph token was rejected before response counters
or kernels advanced. The 2.80x speedup clears the 1.50x target without adding
base-tape reuse, seed fusion, or a generated response ABI, so those higher-risk
stages were not implemented.

The energy, force-L2, maximum-force, and field-adjoint summaries are unchanged
from the compact baseline at printed precision: `-39558.24329628721 eV`,
`158.324113158492 eV/A`, `1.2429926822121085 eV/A`, and
`(-11.573465959482554, 23.146903273959246, -1230.410040942654)`.
Polarizability and BEC summary variation across fresh processes remains at
float32 reduction-noise scale and all focused full-array and finite-difference
tests retain their existing tolerances.

The ordinary energy/forces/stress path was compared against the frozen
pre-change extension (`7962545786239996102e571c7e3604cbc76898308297b4696858a69ec17f8c47`)
with the same current-contract model and graph. Its median changed from
25.183 ms/call (4.730 us/atom) to 25.205 ms/call (4.734 us/atom), a 0.088%
difference. Response-call and response-launch counters remained zero in the
corrected run.

| Scale case | atoms | directed edges | us/atom | ms/call |
|---|---:|---:|---:|---:|
| Small latency | 32 | 3,488 | 284.105 | 9.091 |
| Primary | 5,324 | 580,316 | 74.327 | 395.716 |
| Larger scaling | 6,912 | 753,408 | 73.509 | 508.092 |

Nsight Systems 2026.1 measured the new hottest response lambda at 20.903 ms
per launch and 33.3% of aggregate GPU kernel time. It launches 72,540 teams,
the eight-edge partition of 580,316 directed edges; the old approximately
233-ms generic Phi1 kernel is absent. The trace contains one 13,927,584-byte
geometry H2D during setup and one matching pair-force D2H after timing for
result reporting, but neither transfer recurs inside steady response calls.
The existing 41,782,752-byte BEC force-field-derivative D2H occurs once per
full response call and remains outside this topology optimization.

Nsight Compute 2026.2.1 reports 20.72 ms duration, 64 registers/thread,
65.68% achieved occupancy, 64.82% SM throughput, 94.64% L2 throughput,
21.82% DRAM throughput, 92.47% L2 hit rate, and zero local or shared-memory
spilling for the hottest fused response launch. It is L2/scoreboard limited,
not DRAM-bandwidth limited. Raw three-process timing, Systems, SQLite, and
Compute evidence is under the ignored
`benchmarks/.artifacts/field_response_topology_20260816/` directory.

## HIP `gfx1151` qualification

The HIP performance target uses the same MACEField checkpoint and AlN system
as the CUDA measurements, but is explicitly qualified in float32. The compact
artifact extracted from the current checkpoint has SHA-256
`763811a1f87c6f7c5a4a595fe56ee6392446fc246468c49cfeaaf8072cea2210`.
The test system is an AMD Radeon 8060S (`gfx1151`, 40 compute units, wave32)
with HIP 7.14.60850 from `/opt/rocm` and a Python 3.14 extension compiled for
the actual `gfx1151` ISA.

In this earlier small-system qualification, the generated R1 serial-edge
policy was faster than the wave-per-edge policy. A later 10,976-atom thermal
MH-0 retest reversed that conclusion, so native-wave/t64/b8 became the portable
HIP default. The later raw portability report was not retained in the
repository; this paragraph records only the resulting selection. The retained
field-adjoint path uses a
component-local entry schedule and a 64-bit `channels x channels` reduction
range. HIP also admits the checked-in generated M0 contraction. It replaces
the runtime M0 polynomial path and releases both 288,866,304-byte polynomial
workspaces. With 10 warmups and 10 measured calls, automatic generated M0 and
the serial R1 policy pass a stricter 45 us/atom gate:

| mode | M0 | edge policy | atoms | directed edges | median ms/call | median us/atom | maximum us/atom |
|---|---|---|---:|---:|---:|---:|---:|
| `direct` | generated | serial, 256 threads, 8 blocks/CU | 864 | 78,624 | 32.645 | 37.783 | 38.011 |

```bash
python benchmarks/macefield_kokkos_benchmark.py \
  MACEField-MH-0-omat-dielectric-Al-N-compact.json \
  --backend kokkos --dtype float32 --modes direct \
  --direct-jit required --sizes 864 --warmups 10 --repeats 10 \
  --max-us-per-atom 45 --output macefield-hip-gfx1151-fp32.json
```

The benchmark defaults to float32 and records the selected JIT compiler,
artifact, HIP-plugin launch state, generated M0 provenance, launch counts, and
M0 polynomial workspace. The `--direct-m0-executor` option forces `runtime` or
`generated` for controlled comparisons. The `--max-us-per-atom` option exits
nonzero when any requested mode misses its threshold.

### Shared field reverse in `all_interactions`

The compiled wave32 global-field reverse reduction is shared by HIP and CUDA
and is now selected by both `factorized` and `all_interactions`. The latter
continues to use the default Kokkos execution space and its existing stage
fences; this change does not make it an RTC/JIT mode. CPU, per-node-field,
observer, and parameter-gradient configurations retain the generic reduction.

HIP qualification used the current-contract MACEField MH-0 Al/N artifact
(SHA-256 `65ecd5890c43d75f7d232bab2c99753f27554c201de0323ed719d268530d7435`),
FP32, an 864-atom `6x6x6` wurtzite AlN cell, model cutoff 6.0 A, configured
neighbor-list skin 0.5 A, and effective candidate cutoff 6.5 A. The
`all_interactions` evaluator compacted 78,624 active directed edges at the
exact 6.0 A cutoff. Five measurements after three warmups gave:

| field reverse | median ms/call | median us/atom |
|---|---:|---:|
| generic HIP control | 54.105 | 62.621 |
| shared wave32 HIP | 53.284 | 61.671 |

The optimized run selected the float32 HIP launch profile at eight blocks per
compute unit. Energy was identical at `-6419.674848754312` eV. Changes in the
reported maximum-force magnitude and global field adjoint were below
`5e-7`, consistent with the different FP32 reduction order. The same run's
factorized hipRTC comparison retained all 94,176 candidate edges through the
6.5 A neighbor-list cutoff and measured 30.410 us/atom, so it is not an
edge-count-matched performance control.

Forced runtime and generated M0 screening runs measured 42.644 and 32.812
ms/call respectively with identical energy and a maximum-force difference of
approximately `1.6e-7`. The final automatic result is 23.2% faster than the
previous 42.529 ms qualified path.

The cache-keyed R1 launch sweep showed that the retained 256-thread policy is
best once each edge has direct thread coverage. Finalists used 10 warmups and
20 measured calls:

| threads/block | persistent blocks/CU | median ms/call | minimum ms/call | maximum ms/call |
|---:|---:|---:|---:|---:|
| 256 | 8 | 32.597 | 32.184 | 32.815 |
| 64 | 32 | 32.991 | 32.749 | 33.253 |
| 128 | 16 | 33.399 | 33.098 | 33.684 |

The automatic policy therefore remains 256 threads and 8 blocks/CU. The
validated environment overrides are retained for architecture-specific
qualification rather than runtime autotuning.

The generated R0-v2 artifact is also admitted on wave32 HIP after converting
its edge-owned launch extent and grid-stride indices to `std::size_t`. On the
same 864-atom FP32 target, 5 warmups and 10 measured calls selected edge16 as
the automatic policy:

| R0 executor | median ms/call | median us/atom | generated launches |
|---|---:|---:|---:|
| `v1` runtime | 32.537 | 37.659 | 0 |
| `v2_receiver` | 33.076 | 38.283 | 40 |
| `v2_edge16` | 29.353 | 33.973 | 40 |
| `v2_edge32` | 29.822 | 34.516 | 40 |

After the HIP JIT moved from a hipcc shared library to an in-process hipRTC
code object, the matched `v2_edge16` run measured 29.175 ms/call (33.768
us/atom). This is 0.6% faster than the 29.353 ms hipcc result. The run forced
`SYMMETRIX_JIT_HIP_JIT_BACKEND=hiprtc`, used 5 warmups and 10 measured calls,
and reported a built `.hsaco` module with the same serial R1 launch policy.

The `--direct-r0-executor` benchmark option forces these policies. Generated
variants produced identical energy; relative to runtime R0, the total-energy
difference was about `8.55e-4` eV over 864 atoms and the maximum-force
difference was about `5.5e-6` eV/A, consistent with the different fixed
reduction order. Required HIP tests separately compare all three generated
policies with runtime R0 on energy, forces, and A0.

## 864-atom optimization stages

| implementation | mode | ms/call | ms/atom | R0+R1 bytes | GPU before | GPU setup | GPU after | sampled GPU high-water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| untouched control | legacy | 134.843 | 0.156068 | 2,254,307,328 | 0 MiB | 508 MiB | 4,562 MiB | 4,562 MiB |
| streamed edge kernels only | r1 | 131.492 | 0.152189 | 644,087,808 | 0 MiB | 508 MiB | 3,028 MiB | 3,028 MiB |
| streamed edge kernels only | all | 131.990 | 0.152766 | 0 | 0 MiB | 508 MiB | 2,412 MiB | 2,412 MiB |
| precomposed field transform | legacy | 43.308 | 0.050125 | 2,254,307,328 | 0 MiB | 510 MiB | 4,558 MiB | 4,558 MiB |
| precomposed field + streamed edges | all | 40.100 | 0.046412 | 0 | 0 MiB | 510 MiB | 2,406 MiB | 2,406 MiB |

The retained fully streamed implementation is 3.36x faster than the untouched
control. It reduces process GPU memory after evaluation by 2,156 MiB, from
4,562 MiB to 2,406 MiB (47.3%). Host RSS for the retained 864-atom run was
252.38 MiB before setup, 417.67 MiB after setup, and 525.34 MiB after the
evaluation, with a 529.23 MiB host high-water mark.

The field transform precomposition is responsible for most of the speedup.
Streaming R0/R1 is retained primarily for its 2.10 GiB radial-storage removal;
after precomposition it also improves the 864-atom time from 43.31 to 40.10 ms.

## CUDA scaling

| atoms | directed edges | ms/call | ms/atom | GPU before | GPU setup | GPU after | sampled GPU high-water |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 23,296 | 14.071 | 0.054967 | 0 MiB | 510 MiB | 1,210 MiB | 1,208 MiB |
| 864 | 78,624 | 40.100 | 0.046412 | 0 MiB | 510 MiB | 2,406 MiB | 2,406 MiB |
| 4,000 | 364,000 | 172.322 | 0.043081 | 0 MiB | 510 MiB | 8,636 MiB | 8,636 MiB |

The 256-atom run also timed one response call after the force benchmark:
68.16 ms for the electric-field Hessian and 65.50 ms for the electric-field
force derivative. This explains why its final GPU memory can be 2 MiB above
the high-water sampled during the single energy/force pass.

## Analytic response port

The Kokkos field Hessian and field-force derivative now use the native CPU
forward-over-reverse algorithm instead of a `1e-6` forward difference. The
implementation propagates three exact electric-field directions through H1,
Phi1, A1, M1, H2, and the readout, then differentiates the reverse pass. CUDA
uses the same streamed edge-team layout as the qualified first-order reverse
Phi1/A0 kernels. Constant field/linear-up maps are composed at model load,
and the 3x3 Hessian is accumulated by a reduction rather than global atomics.

Matched 864-atom `all` measurements use the float32 production condition and
separate latency timing from the additional `nvidia-smi` memory-sampling call.
The finite-difference control reconstructs the former baseline plus three
perturbed evaluations, including its device-to-host response copies.

| precision | response | ms/call | ms/atom | GPU before response | GPU after response | sampled response high-water |
|---|---|---:|---:|---:|---:|---:|
| float32 | analytic field-force derivative | 43.832 | 0.050731 | 1,524 MiB | 1,536 MiB | 1,536 MiB |
| float32 | analytic field Hessian, first call | 47.471 | 0.054943 | 1,524 MiB | 1,536 MiB | 1,538 MiB |
| float32 | finite-difference control | 107.758 | 0.124720 | 1,524 MiB | 1,524 MiB | 1,526 MiB |
| float64 | analytic field-force derivative | 167.453 | 0.193811 | 2,408 MiB | 2,420 MiB | 2,420 MiB |
| float64 | analytic field Hessian, first call | 172.664 | 0.199842 | 2,408 MiB | 2,420 MiB | 2,420 MiB |
| float64 | finite-difference control | 213.365 | 0.246950 | 2,420 MiB* | 2,420 MiB* | 2,420 MiB* |

`*` The float64 finite-difference control was measured after the analytic
workspace had already raised the process allocation to 2,420 MiB. Float32's
control was also run in a fresh process and retained 1,524 MiB after the
response, with a 1,526 MiB high-water.

At float32, the steady analytic response is 2.46x faster than the matched
finite-difference control. At float64 it is 1.27x faster. The analytic
workspace adds 12 MiB of retained CUDA process memory at both precisions.

### Response agreement

The benchmark also records max-absolute, RMS, and relative-L2 differences for
the complete 3x3 field Hessian and raw directed-edge field-force derivative.
At float64, the analytic path agrees directly with the former one-sided
`1e-6` finite-difference path:

| float64 quantity | max absolute difference | RMS difference | relative L2 |
|---|---:|---:|---:|
| field Hessian | 7.570e-5 | 2.562e-5 | 2.276e-7 |
| field-force derivative | 7.551e-7 | 3.453e-8 | 6.193e-7 |

The former `1e-6` perturbation is below a useful subtraction scale for
float32: its raw force-derivative relative-L2 difference is 0.970. A centered
step sweep demonstrates convergence toward the analytic result instead:

| float32 centered step | Hessian relative L2 | force-derivative relative L2 |
|---:|---:|---:|
| `1e-3` | 2.134e-4 | 9.281e-3 |
| `3e-3` | 7.040e-5 | 2.932e-3 |
| `1e-2` | 5.240e-5 | 1.023e-3 |
| `3e-2` | 3.050e-4 | 3.472e-4 |

At the balanced `1e-2` step, maximum absolute differences are 1.549e-2 for
the Hessian and 1.434e-3 for the raw force derivative. The remaining error is
finite-difference quantization/truncation, not disagreement between the two
analytic backends: permanent tests compare Kokkos float32 and float64 directly
against the native analytic implementation, while separate native tests check
the analytic equations against centered finite differences.

## OpenMP scaling

The current-source Kokkos OpenMP build used the same 864-atom artifact in
fully streamed float64 mode, with BLAS pinned to one thread, 3 warmups, and
10 measured calls.

| Kokkos threads | ms/call | ms/atom | host before | host setup | host after | host high-water |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,340.841 | 6.181529 | 69.00 MiB | 118.98 MiB | 1,880.29 MiB | 1,882.99 MiB |
| 2 | 2,712.128 | 3.139037 | 69.22 MiB | 119.09 MiB | 1,880.18 MiB | 1,883.80 MiB |
| 8 | 756.915 | 0.876058 | 69.20 MiB | 119.29 MiB | 1,879.50 MiB | 1,881.61 MiB |
| 16 | 441.384 | 0.510862 | 69.37 MiB | 120.30 MiB | 1,881.52 MiB | 1,883.98 MiB |

The 16-thread result was remeasured from the rebuilt extension after the
native port. Its fresh-process legacy control was 487.570 ms/call
(0.564317 ms/atom), 4,031.23 MiB RSS after evaluation, and 4,033.39 MiB
high-water. Thus OpenMP `all` is 9.5% faster and lowers post-evaluation RSS by
2,149.71 MiB, from 4,031.23 to 1,881.52 MiB (53.3%).

## Native CPU before and after

The native CPU comparison used separate fresh processes, one BLAS thread,
3 warmups, and 10 measured calls on the same 864-atom float64 case.

| mode | ms/call | ms/atom | R0+R1 bytes | host before | host setup | host after | host high-water |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | 2,447.748 | 2.833042 | 2,254,307,328 | 69.18 MiB | 136.43 MiB | 4,473.07 MiB | 4,476.01 MiB |
| all | 3,384.924 | 3.917736 | 0 | 69.22 MiB | 136.43 MiB | 2,323.27 MiB | 2,326.77 MiB |

Native `all` lowers post-evaluation RSS by 2,149.80 MiB, from 4,473.07 to
2,323.27 MiB (48.1%), and lowers process high-water by 2,149.23 MiB. It is
38.3% slower than native legacy on this case. Unlike CUDA, the serial CPU
path benefits from retaining radial splines across the forward and reverse
passes; strict edge streaming must recompute those values during reverse.
Native `all` is therefore currently a memory-capacity mode, not a CPU speed
optimization.

## Numerical checks

At 864 atoms, the retained result has the same printed energy as the control,
`-6419.674540523367 eV`. Maximum-force and electric-field-adjoint differences
are approximately `1e-14`. Focused CUDA and OpenMP tests cover the standalone
field transform and reverse, full energy/forces, polarization,
polarizability, Born effective charges, field Hessian, field-force derivative,
legacy/r1/all equivalence, radial-storage release, and malformed Phi1
hidden-degree rejection. The streamed property comparison is parameterized
over native CPU and Kokkos and passed with both OpenMP and CUDA builds.

## Native float32 template qualification

The native CPU evaluator now instantiates the same standard-MACE and MACEField
algorithm at float32 and float64. These matched 864-atom `all` runs used one
native CPU thread, one BLAS thread, 10 warmups, and 10 measured first-order calls
from base commit `fd5dd4a468e8e53eadca3f157fee1ac2e017b973` plus the Phase 56
working tree. The binding confirmed four-byte learned tensors and workspaces in
the float32 evaluator and eight-byte storage in float64.

| Dtype | First order (ms) | Time/atom (ms) | RSS after (MiB) | Peak RSS incl. response (MiB) | Field Hessian (ms) | Field-force derivative (ms) |
|---|---:|---:|---:|---:|---:|---:|
| float64 | 1,293.913 | 1.497585 | 2,327.6 | 6,611.8 | 11,788.7 | 11,894.4 |
| float32 | 1,112.741 | 1.287895 | 1,244.8 | 3,369.0 | 10,772.8 | 11,501.2 |

Float32 is 1.16x faster for the first-order field-aware call, lowers final RSS
by 1,082.8 MiB (46.5%), and lowers the analytic-response high-water RSS by
3,242.7 MiB (49.0%). The analytic Hessian and field-force derivative are 1.09x
and 1.03x faster, respectively. The response calculations use the same analytic
forward-over-reverse implementation at both precisions. Permanent cross-
precision tests cover energy, forces, polarization, polarizability, Born
effective charges, the field Hessian, and the raw field-force derivative.
