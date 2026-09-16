# Generalized MACE-MH-1 family qualification

Date: 2026-07-28

This report qualifies the runtime-sized two-interaction MACE-MH-1 family
implementation based on commit `1da1ccc949d6d6de553b5903beea18858778f20c`
with the generalization changes uncommitted. The build used CUDA 13.3,
Kokkos CUDA+Serial, Blackwell120, SpheriCart CUDA, Python 3.12.13, and an
NVIDIA GeForce RTX 5090.

## Supported fixture matrix

| Fixture | Node channels | Edge channels | Bessel count | `l_max` | Radial MLP widths |
|---|---:|---:|---:|---:|---|
| Generalized L2 | 8 | 4 | 8 | 2 | 7, 9 |
| Requested L2 qualification | 256 | 64 | 8 | 2 | 7, 9 |
| Published production anchor | 512 | 128 | 10 | 3 | checkpoint-defined |
| Generalized L3 | 12 | 6 | 16 | 3 | 9, 11, 7 |

The self-contained generalized fixtures are constructed with upstream
`ScaleShiftMACE`, extracted to format-version-3 JSON, and checked for energy,
per-atom energy, forces, and stress. Native serial, Kokkos Float64, and Kokkos
Float32 cover `legacy`, `r1`, and `all`. Both angular shapes also pass analytic
force versus central finite difference tests. The focused OpenMP and CUDA
capability matrices each pass 22 tests. The complete nonlinear OpenMP suite
passes 88 tests with two environment-dependent skips. The dedicated 256/64
CUDA end-to-end oracle adds one further full-model qualification.

The CUDA `l_max=2` tests found and fixed an out-of-bounds compiled-product
reverse loop: a 16-slot local accumulator was valid storage for both supported
angular extents, but the load/store loops now use the runtime extent (9 or 16).
A direct SpheriCart CPU/CUDA probe measured maximum Float64 errors of
`3.44e-15` for gradients at both `l_max=2` and 3.

## Published-model performance gate

Model SHA-256:
`8384b616054cc4391531ca95af7f9b4737ce902628701faaf8e54878b5a79f00`

System: 864-atom periodic AlN, 78,624 directed edges. Protocol: one fresh
process, CPU 0, three warmups, ten synchronized evaluator-only repeats,
Float32, `streamed_edges="all"`.

| Metric | Generalized implementation | Previous anchor | Change |
|---|---:|---:|---:|
| Median time | 61.894 ms | 61.817 ms | +0.13% |
| Median time per atom | 0.071637 ms | 0.071547 ms | +0.13% |
| GPU process memory before | 506 MiB | 506 MiB | 0 |
| GPU process memory after | 3,184 MiB | 3,184 MiB | 0 |
| Edge workspace | 1,486,880,768 bytes | 1,486,880,768 bytes | 0 |
| Total precision workspace | 1,548,812,288 bytes | 1,548,812,288 bytes | 0 |

Raw samples (ms):

```text
61.814949, 62.190247, 61.838151, 64.205396, 65.880042,
63.553454, 61.785023, 61.871063, 61.917599, 61.845615
```

Selected execution: packed GEMM E3 linears, `official_kokkos` tensor
semantics, `official_cuda_team` Float32 tensor execution, and a 16,384-edge
streaming block. The result is inside the 5% latency and memory gates.

## Dynamic tensor-team qualification

The requested 256-node-channel/64-edge-channel L2 family model was generated
with eight Bessel functions and radial widths 7/9 for Al/N. The 864-atom AlN
case had 22,464 directed edges and used the same three-warmup, ten-repeat,
synchronized Float32 `all` protocol. Dynamic 64-thread teams were compared
against a temporary build reproducing the previous fixed 128-thread policy.

| Policy | Median time | Time per atom | GPU memory after | Precision workspace |
|---|---:|---:|---:|---:|
| Dynamic 64/64 channel/harmonic teams | 10.838 ms | 0.012544 ms | 1,514 MiB | 517,341,184 bytes |
| Fixed 128/128 control | 11.058 ms | 0.012799 ms | 1,514 MiB | 517,341,184 bytes |

The dynamic policy improves total evaluator time by 1.99% on this case while
leaving memory unchanged. Energy agrees exactly at the reported precision;
the force-L2 difference is `2.31e-8 eV/A`, consistent with the changed
Float32 harmonic reduction order. Raw results are
`/tmp/mh1-c256-e64-dynamic64.json` and
`/tmp/mh1-c256-e64-fixed128.json`.

## Fused reverse-kernel milestone

The first Phase 67 reverse fusion is implemented by commit
`f82f2074bf74074654eda3d4774b80cf5668849d`. The machine, model, 864-atom
AlN graph, CUDA 13.3 build, CPU 0 affinity, three warmups, ten synchronized
repeats, Float32 precision, and `streamed_edges="all"` protocol match the
published-model gate above. The model SHA-256 is
`8384b616054cc4391531ca95af7f9b4737ce902628701faaf8e54878b5a79f00`.

Two independent runtime controls isolate the retained changes:
`fused_gate_normalization_reverse` fuses the gate and normalization adjoints
per node, while `direct_node_tensor_reverse` consumes source-node values and
target-node message adjoints directly. The latter removes the reverse edge-up
gather, edge-message-adjoint materialization, edge-up-adjoint allocation, and
source scatter. Both controls default on only when their qualified Float32
CUDA MH-1 predicates hold; the previous kernels remain the fallback and the
benchmark can force either control off.

| Configuration | Median time | Time per atom | GPU memory before | GPU memory after | Edge workspace | Precision workspace |
|---|---:|---:|---:|---:|---:|---:|
| Same-build control, both fusions off | 62.329 ms | 0.072140 ms | 506 MiB | 3,184 MiB | 1,486,880,768 bytes | 1,548,812,288 bytes |
| Fused node reverse only | 55.649 ms | 0.064408 ms | 506 MiB | 3,184 MiB | 1,486,880,768 bytes | 1,548,812,288 bytes |
| Direct tensor reverse only, before team cutoff adjoint | 49.075 ms | 0.056800 ms | 506 MiB | 3,140 MiB | 1,444,937,728 bytes | 1,506,869,248 bytes |
| Both fusions plus team cutoff adjoint | 40.669 ms | 0.047070 ms | 506 MiB | 3,140 MiB | 1,444,937,728 bytes | 1,506,869,248 bytes |

The retained result is 34.8% faster than the same-build control and removes
40 MiB of evaluator workspace. Absolute post-initialization process memory
falls by 44 MiB. It also improves the earlier committed 61.894 ms anchor by
34.3%, but remains approximately 1.96 times the historical 20.793 ms
cuEquivariance result. This milestone therefore clears its 42-48 ms target
after the cutoff reduction was parallelized, without yet closing the full
backend gap.

Retained raw samples (ms):

```text
40.196233, 40.202124, 40.203567, 40.363234, 40.326736,
42.619032, 44.245648, 44.193611, 41.968872, 40.974131
```

The reviewed result is `/tmp/mh1-phase67-f82f207.json`. Its metadata records
implementation revision `f82f2074bf74074654eda3d4774b80cf5668849d` and
tracked-diff SHA-256
`dd2cebf4dc3fc14ee79df7f44be81bacdc1cbdac8280164663b82a15b4f8301d`.
The tracked difference at measurement time contains only this pending report;
the runtime, bindings, benchmark harness, and tests exactly match the recorded
implementation revision. Both fusion controls report requested `on`,
available `true`, and effective `true` in the artifact.

The retained Nsight Systems trace reports approximately 0.243 ms per
evaluation for the fused node reverse, 2.079 ms for the direct channel/weight
tensor adjoint, 4.695 ms for its separate harmonic adjoint, and 0.826 ms for
the parallel cutoff adjoint. It confirms that the eliminated reverse gather,
materialization, and scatter launches are absent. Kernel count remains 882 per
evaluation because the next dominant work is on the forward edge path,
support transforms, affine epilogues, and product-basis contractions. Those
operator boundaries are the Phase 68 fusion target.
