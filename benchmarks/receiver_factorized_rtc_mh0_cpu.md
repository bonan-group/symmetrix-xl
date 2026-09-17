# MH-0 CPU Receiver-Factorized RTC Experiment

> **Historical record (superseded).** This experiment predates removal of the
> public `receiver_factorized` selector. Its decision below is retained as
> measurement history, not current guidance. See the
> [execution support matrix](../docs/reference/execution_support_matrix.md).

Date: 2026-08-19

## Question

Can the original direct execution receiver-factorized R1 algorithm, reimplemented with the
current host RTC cache and generated fixed topology, outperform the production
`direct_streamed` algorithm on CPU?

Learned radial/A1 projections remain runtime-loaded. The RTC artifact generates
the fixed UVU/CG topology, uses bounded receiver chunks, and calls single-thread
OpenBLAS for the composed receiver projection. The generator now exposes both
the isolated harness ABI and the validated production host-plugin ABI.

## Workload

- Model: `/tmp/mace-mh-0-current-Al-N.json`
- Model SHA-256:
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`
- Model contract: 128 channels, radial embedding 64, ten UVU paths, 40 packed
  path components
- Structure: 6x6x6 wurtzite AlN, 864 atoms
- Model cutoff: 6.0 A
- Neighbor skin: 0.0 A; effective cutoff 6.0 A
- Directed edges: 78,624
- Backend: Kokkos OpenMP float32, one physical CPU core
- BLAS: OpenBLAS, one thread
- CPU: AMD Ryzen AI MAX+ 395
- Receiver chunk: 16; bounded RTC scratch 7.55 MiB
- Timing: one warmup and three samples per isolated mode

## Correctness

The RTC implementation was compared with the existing generated direct operator
using the same exported H1, spherical harmonics, penultimate radial values,
learned projection, A1 cotangent, and coordinate derivatives.

| Quantity | Maximum absolute error | Gate |
|---|---:|---:|
| A1 output | 2.265e-6 | `atol=3e-5`, `rtol=2e-5` |
| H1/source adjoint | 1.788e-6 | `atol=3e-5`, `rtol=2e-5` |
| Directed edge force | 2.827e-7 eV/A | `atol=3e-5`, `rtol=2e-5` |

All comparisons pass.

## Timing

Primary values are median us/atom.

| Isolated R1/A1 scope | `direct_streamed` | `receiver_factorized` | Ratio |
|---|---:|---:|---:|
| Forward | 30.54 us/atom | 1,844.75 us/atom | 60.40x |
| Reverse | 97.72 us/atom | 1,806.31 us/atom | 18.48x |
| Forward + reverse | 133.55 us/atom | 3,672.94 us/atom | 27.50x |

For context, the matched full `direct_streamed` evaluator baseline is 210.38
us/atom. The isolated receiver-factorized R1/A1 operator is therefore 17.46x
slower than the complete production evaluation before integrating any other
model stages.

Raw forward-plus-reverse samples were 115.783, 115.078, and 115.388 ms for
`direct_streamed`; receiver-factorized samples were 3,173.418, 3,168.184, and
3,181.750 ms.

A receiver-chunk sweep of 4, 8, 16, 32, and 64 used 1.89, 3.78, 7.55, 15.11,
and 30.22 MiB scratch, respectively. Single-sample forward-plus-reverse results
ranged from 3,491 to 3,770 us/atom. Chunk tuning does not change the conclusion.

### Production full evaluator

The production ABI evaluates the compact penultimate radial spline inside the
artifact rather than materializing per-edge `phi` and `dphi_dr`. With three
warmups and five measured evaluations on the same one-thread workload:

| Full evaluator | Median us/atom | Speedup vs materialized |
|---|---:|---:|
| `materialized` | 4,639.953 | 1.000x |
| `direct` | 230.820 | 20.102x |
| `receiver_factorized` RTC | 3,812.570 | 1.217x |

Receiver RTC produced maximum differences of 5.183e-4 eV total energy,
5.499e-6 eV/A force, and 2.260e-7 eV/A^3 stress versus `materialized`. Each of
the five measured evaluations launched exactly one receiver forward and one
receiver reverse artifact and no direct R1 RTC artifact. The raw record is
`/tmp/post-execution-mode-migration-mh0-864-rtc.json`.

## Why It Loses

The current MH-0 contract is tied-channel UVU and its direct implementation
evaluates already projected radial splines. Receiver factorization first forms a
64-wide radial/coupling outer product for 40 path components, then applies the
composed dense A1 projection. Its forward arithmetic is approximately:

- receiver aggregation: 51.53 GFLOP;
- receiver projection: 72.48 GFLOP;
- total: 124.00 GFLOP per evaluation.

The measured 1.594 s receiver forward corresponds to about 77.8 GFLOP/s on one
core, so the result is consistent with the actual work. Avoiding that dense
receiver projection by exploiting UVU tied channels recovers the
`direct_streamed` formulation rather than a faster receiver-factorized variant.
The crossover may differ for a genuinely dense UVW tensor product or when the
final radial projection would otherwise be evaluated per edge.

## Decision

Keep `receiver_factorized` as a distinct explicit RTC algorithm and keep
`direct` as the automatic/high-performance algorithm. The receiver path is
useful as the faithful direct execution control and is faster than legacy materialization,
but it is 16.5x slower than `direct` for this tied-channel UVU model. The
compatibility alias `factorized` therefore continues to mean `direct`.

Reproduce with:

```bash
.venv/bin/python benchmarks/receiver_factorized_rtc_benchmark.py \
  /tmp/mace-mh-0-current-Al-N.json --repeat 6 --receiver-chunk 16 \
  --warmups 1 --repeats 3 \
  --output /tmp/receiver-factorized-rtc-mh0-864.json
```
