# JIT-free `all` AOT M0 and M1 recomputation

> **Historical record (superseded).** This report preserves the internal mode
> names and decisions used for its original measurements. `streamed_edges="all"`
> is no longer a public request; see the
> [execution support matrix](../docs/reference/execution_support_matrix.md).

## Decision

Promote automatic structural AOT M0 admission inside `streamed_edges="all"` for
compatible float32 ordinary MACE and MACEField models. Keep runtime M0 as the
explicit rollback and unsupported-model path. Keep M1 recomputation opt-in,
tile 32 preferred, and retained M1 as the default. Keep MACEField M1
recomputation rejected because its analytic response still consumes retained
M1 polynomial values.

This does not invoke JIT and does not alter the fail-closed `direct_jit="auto"`
contract.

## Provenance

- Source commit before the working-tree changes: `596d7dd32bd0a0bd880111008749d876297c51ae`.
- CUDA extension SHA-256: `12e2410a6a1151ba88958b3be5e5c8e86ea5be10962cd53e2710b8f6b67df75f`.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 610.43.02.
- Models: MH-0 current contract and its fine-tuned MACEField current contract.
- Each paired point used three randomized fresh processes, five warmups, ten
  synchronized energy/force/stress samples, and process-level sampled GPU
  high-water memory.
- Raw report: `benchmarks/.artifacts/all_aot_m0_m1/paired.json`; individual
  process reports are in `paired-raw/`.

## Paired CUDA results

Times are fresh-process median milliseconds. Memory is median sampled GPU MiB.
The retained 32,000-atom controls OOMed while allocating
`M0_poly_adjoints_3`; all three trials failed for both ordinary and field
models.

| Model | Atoms | Retained ms / MiB | AOT M0 ms / MiB | AOT M0 + M1 ms / MiB |
|---|---:|---:|---:|---:|
| MACE | 32 | 2.525 / 786 | 2.222 / 754 | 2.198 / 746 |
| MACE | 864 | 10.871 / 1,616 | 8.653 / 1,060 | 8.689 / 940 |
| MACE | 5,324 | 50.750 / 6,046 | 44.679 / 2,642 | 44.589 / 1,902 |
| MACE | 6,912 | 65.463 / 7,626 | 57.611 / 3,210 | 57.479 / 2,250 |
| MACE | 32,000 | OOM | 266.100 / 12,070 | 264.641 / 7,630 |
| MACEField | 32 | 2.503 / 772 | 2.226 / 740 | rejected |
| MACEField | 864 | 10.247 / 1,602 | 8.651 / 1,046 | rejected |
| MACEField | 5,324 | 50.877 / 6,032 | 44.446 / 2,628 | rejected |
| MACEField | 6,912 | 65.714 / 7,612 | 57.442 / 3,196 | rejected |
| MACEField | 32,000 | OOM | 266.052 / 12,058 | rejected |

AOT M0 is 11.1-20.4% faster than retained M0 at every controlled size and
saves 32 MiB at 32 atoms, 556 MiB at 864 atoms, 3,404 MiB at 5,324 atoms, and
4,416 MiB at 6,912 atoms. Combined M1 recomputation is within +0.42% of AOT-M0
alone at its slowest point and is otherwise neutral or faster. At 32,000 atoms
it saves a further 4,440 MiB, matching the eliminated M1 tensor capacities.
All optimized records report zero active and capacity bytes for the selected
M0/M1 polynomial tensors and exactly one measured forward/reverse launch per
evaluation. Tile-32 scratch is 36,352 bytes per team.

## Capacity and profiling

The combined ordinary-MACE path completes n30 (108,000 atoms, 9,828,000
directed edges) in 904.374 ms using 23,912 MiB. n31 (119,164 atoms) fails on a
931 MiB allocation with an otherwise idle 32 GiB GPU. The remaining limit is
the generic `all` Phi storage: at n30, `Phi1r/dPhi1r` are 4,866,048,000 bytes
each and `Phi1/dPhi1` are 2,211,840,000 bytes each. The earlier n31 completion
at 16,290 MiB used direct execution and therefore is not a capacity result for `all`.

Nsight Systems 2026.1.3 captured one 864-atom evaluator NVTX range. Generated
M0 forward/reverse took 25.4/37.9 us and M1 recompute forward/reverse took
105.8/165.5 us, each as one ordered launch inside the range. Generic Phi1
forward/reverse remained dominant at 2.076/1.664 ms. The trace is
`benchmarks/.artifacts/all_aot_m0_m1/nsight/all-aot-m0-m1-n6.nsys-rep`.

Nsight Compute 2026.2.1 matched the M1 kernel but hardware-counter collection
was blocked by `ERR_NVGPUCTRPERM`; no occupancy value is inferred. Compiled
resource metadata reports 40 registers and zero local memory for float M1
forward/reverse, with 48/24 stack bytes. Structural M0 reports 114/254
registers, zero local memory, and 0/8 stack bytes for forward/reverse.

## Correctness

Runtime/generated M0 and retained/recomputed M1 match energy, atomic energies,
forces, stress, M0/M1, and A0/A1 adjoints under existing float32 tolerances.
MACEField AOT M0 additionally matches field adjoints, electric-field Hessian,
and field-force derivatives; its analytic response performs four generated M0
reverse launches without restoring polynomial storage. Retained to optimized
to retained lifecycle and unsupported-artifact fallback/forced rejection pass
on Kokkos OpenMP and CUDA.

Generic `all` reductions are not bitwise repeatable on either OpenMP or CUDA;
observed variation is approximately 1e-7 to 1.3e-6 and is present independently
of these policies. Qualification therefore uses the repository's existing
float32 tolerances for `all`; direct execution retains its bitwise lifecycle assertions.
