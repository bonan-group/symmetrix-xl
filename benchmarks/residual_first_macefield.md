# Residual-first MACEField qualification

> **Historical record (superseded).** This report preserves the former
> `legacy`, `r1`, and `all` selector names used by its qualification. They are
> no longer public requests; see the
> [execution support matrix](../docs/reference/execution_support_matrix.md).

Qualification date: 2026-08-06

## Provenance

- Checkpoint: `/path/to/mace_field.model`
- Checkpoint SHA-256: `9643c6731ddb87dfec2251da7e5e525a4c51c5a0cd89e1be9786b906b3e2212b`
- Al/N compact JSON SHA-256: `e8bd97f7bd7cc312ed4ed9dde7d8558940168ccb8fcd239a7f0d2f35fea79e06`
- Architecture: two `RealAgnosticResidualInteractionBlock` interactions,
  48 channels, H1 `L_max=1`, nonlinear readout width 48, field coupling
- CUDA toolchain: `/usr/local/cuda-13`, Kokkos CUDA `sm_120`
- Capacity device: NVIDIA RTX 5090, 32,607 MiB

The source checkpoint is CUDA-serialized. Conversion and the MACE PyTorch
oracle therefore require a visible CUDA device, while the extracted JSON loads
on native CPU, Kokkos OpenMP, and Kokkos CUDA without PyTorch.

## Correctness

The four-atom wurtzite AlN cell used field `[0.013, -0.021, 0.008]`. Native
`legacy`, `r1`, and `all`; Kokkos `legacy`, `r1`, `all`; generic `direct`;
and NVRTC-specialized `direct` were compared with MACE PyTorch. Across all
paths, the largest absolute errors were:

- energy: `2.92e-6 eV`
- force component: `5.59e-6 eV/A`
- stress component: `1.26e-6 eV/A^3`

NVRTC R1 generated repeats were bitwise deterministic for energy, forces,
stress, and field adjoint. The retained -> generated -> retained lifecycle
restored energy exactly; the largest restored force difference was `4.74e-8`
and the largest restored field-adjoint difference was `1.59e-8`. During the
generated phase, generic AOT M0 and R0-v2 edge16 were selected. Cold NVRTC
build and warm cache reuse both succeeded.

Synthetic residual-first tests additionally cover serial/OpenMP/CUDA field
Hessian and field-force derivative parity plus a central finite-difference
field-Hessian control. direct execution parameter gradients reject before evaluation.

### Isolated residual cost

A benchmark-only control disabled the residual table and its node-local launch
in the otherwise identical exact JSON. This control is not a valid physics
model; it isolates execution cost only. Fresh CUDA processes selected the same
generated M0, R0-v2 edge16, and NVRTC R1 paths.

| Cell | Residual | Control | Difference |
|:---|---:|---:|---:|---:|
| 32 atoms | 1.2402 ms | 1.1583 ms | +7.1% |
| 5,324 atoms | 16.3482 ms | 16.3338 ms | +0.1% |

The fixed launch is visible for a very small cell and becomes negligible at
medium scale. The two resident residual tables contain 768 float32 values for
this two-species model (3,072 bytes total) and do not scale with atom or edge
count.

## CUDA capacity

Fresh processes evaluated balanced periodic `bulk("AlN", "wurtzite").repeat(n)`
cells with float32 Kokkos CUDA, `streamed_edges="direct"`, required NVRTC R1,
automatic AOT M0/R0, and retained M1 (MACEField does not admit M1
recomputation).

| Repeat | Atoms | Directed edges | Result | GPU process memory |
|---:|---:|---:|:---|---:|
| n30 | 108,000 | 5,292,000 | pass | 12,568 MiB |
| n40 | 256,000 | 12,544,000 | pass | 28,878 MiB |
| n41 | 275,684 | 13,508,516 | pass twice | 31,048 MiB |
| n42 | 296,352 | - | CUDA OOM twice | about 32,032 MiB before final allocation |

The qualified maximum for this exact checkpoint and device is therefore n41,
or 275,684 atoms. The first failing balanced cell is n42, or 296,352 atoms;
Kokkos failed its final 217.1 MiB allocation in both fresh processes. This is a
checkpoint-, graph-, precision-, policy-, and device-specific capacity result.

## Analytical response memory

The Kokkos analytical response was subsequently changed to consume the current
primal field state, split polarizability-only execution before coordinate
reverse, release temporarily materialized direct execution R0/R1 tables, and stage
mutually exclusive per-seed workspaces by phase. No finite-difference policy,
response-specific generated kernel was introduced. The follow-up bounded M1
policy reuses the force/stress polynomial recomputation machinery for the
directional graph and remains opt-in.

Fresh-process CUDA capacity results are:

| Response | Previous maximum | Staged maximum | Recompute maximum | First recompute failure |
|:---|---:|---:|---:|
| polarizability | n24 / 55,296 atoms | n27 / 78,732 atoms | n33 / 143,748 atoms | n34 / 157,216 atoms |
| BEC + polarizability | n24 / 55,296 atoms | n25 / 62,500 atoms | n27 / 78,732 atoms | n28 / 87,808 atoms |

Both boundaries were repeated twice on each side. At the common n24 point,
combined response changed from 4,365.456 ms to 4,265.221 ms. The sampled GPU
process memory after response changed from 20,942 MiB to 7,046 MiB because
radial and phase-local workspaces no longer remain resident. The sampler polls
at 50 ms and can miss transient allocations, so the repeated fresh-process
pass/OOM boundary is the authoritative capacity measurement.

The opt-in recompute path reports zero active and capacity bytes for retained
M1 polynomial values and adjoints. Its directional forward uses two bounded
scratch arrays; reverse uses four. For this model, response tile 32 exceeded the
49,152-byte CUDA team limit, so the model/device-aware selector chose tile 16
(36,352 bytes). It can fall back to tile 8 for a larger polynomial graph and
rejects the policy if tile 8 does not fit. Retained remains the default and
rollback oracle.

Polarizability additionally evaluates `R1` values without allocating
`R1_deriv` or executing coordinate-force contractions. At n33 it completed
twice in 3,721.949 and 3,724.301 ms; n34 fails allocating 2.474 GiB for
`dPhi1r_dot`. Combined response n27 completed twice in 6,046.077 and 6,070.523
ms; n28 fails allocating 1.382 GiB for `dPhi1r`. The radial table kernel was
widened after n29 exposed a row-stride overflow above `INT32_MAX`; n29 then
completed with 2,294,517,120 radial values.

The original files named as paired response timings were invalid: the campaign
supervisor did not forward `--response`, and their records show `response: none`
and zero response-M1 launches. They are excluded from the decision. The fixed
supervisor validates response identity and output shape and alternates policy
order within each trial. Five fresh processes per policy, with two warmups and
five samples each, produced:

| Atoms | Retained | Recompute tile 16 | Median change | Median paired change |
|---:|---:|---:|---:|---:|
| 864 | 28.163 ms | 30.168 ms | +7.12% | -0.71% |
| 6,912 | 175.064 ms | 168.429 ms | -3.79% | -2.59% |
| 32,000 | 851.553 ms | 819.324 ms | -3.78% | -3.84% |

All 30 records contain a nine-component polarizability. Recompute evaluations
also report the expected three forward and three reverse response-M1 launches
per evaluation. Tile 16 therefore remains the recommended experimental setting
and satisfies the no-more-than-8% end-to-end regression gate, although the
small-cell result remains launch/noise sensitive. The corrected raw record is
`cuda-analytic-polarizability-m1-recompute-paired-final.json`.

Nsight Systems captured three response M1 forward launches averaging about
95 us and three reverse launches averaging about 197 us at 864 atoms. Both use
40 registers per thread and zero local memory per thread; launch blocks contain
16 threads. Reported dynamic shared memory, including Kokkos overhead, is
18,200 bytes forward and 36,376 bytes reverse. The existing fused Phi reverse
remains dominant at about 13.43 ms per seed under profiling. Nsight Compute
reached the selected forward launch but hardware counters are unavailable to
this user (`ERR_NVGPUCTRPERM`). The Systems report and SQLite export are
`nsys-analytic-m1-recompute-n6.nsys-rep` and `.sqlite`.

## Reproduction

`benchmarks/macefield_standard_aln_scale.py parity` produces the PyTorch and
execution-path comparison. Its `capacity` command performs fresh-process
bracketing, preserves stdout/stderr per point, classifies CUDA allocation
failures, and repeats both sides of the final boundary. Raw records are under
`benchmarks/.artifacts/residual_first_macefield/` in the qualification
workspace.
