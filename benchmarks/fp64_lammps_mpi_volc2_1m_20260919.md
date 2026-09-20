# Volc2 FP64 LAMMPS MPI fixed workspace

Date: 2026-09-19
Host: `volc2`
Remote evidence: `/vepfs/symmetrix-fixed-workspace-mpi/fp64-fixed-workspace-20260919`

## Outcome

FP64 CUDA fixed-workspace execution is implemented for the single- and
dual-layer tiled plans. The tiled geometry path keeps Cartesian FP64 edge
vectors in a batch-sized workspace instead of allocating full-graph geometry.
FP32 continues to use compact FP32 unit directions plus FP64 radii.

The one-million-atom SrTiO3 workload now runs on one A100-80GB with automatic
plan selection. It previously selected a 110.50-GiB non-fixed plan and failed;
it now selects `mh0-dual-layer-tiled-v1`, peaks at 17,991 MiB, and completes a
five-step warmup plus 20 measured NVT steps.

On two GPUs, fixed workspace reduces sampled memory from 62,173 to 13,125 MiB
per GPU (78.9%) and costs 12.7% throughput relative to the direct capacity
plan. Automatic selection still chooses the direct plan when it fits; the
fixed plan was forced for that comparison.

## Implementation

- Removed the FP32-only admission restriction from both tiled plans.
- Enabled the generated `channel-tiled-64` Phi1 module for FP64 while keeping
  the receiver-local implementation FP32-only.
- Materialized cutoff-clamped FP64 Cartesian geometry per tile in
  `execution_prepared_xyz`; its extent is the workspace edge capacity, not the
  full graph edge count.
- Preserved the FP32 compact unit-direction path and selected the correct ZBL,
  harmonic, reverse, and virial inputs for each geometry policy.
- Accounted for either compact directions or Cartesian coordinates in tiled
  workspace estimates and runtime counters.

## Workload and identities

- 1,000,000 SrTiO3 atoms from a `100 x 100 x 20` five-site cubic replication
- OMAT-medium standard MACE, two interactions, 128 channels, FP64
- 6.0 Angstrom model cutoff, 0.5 Angstrom skin, 6.5 Angstrom effective cutoff
- Initial 102,800,000 directed edges; 99,817,766 after the measured trajectory
- One or two MPI ranks, one rank per A100-SXM4-80GB
- Two-rank grid: `1x1x2`
- OpenMPI 4.1.6 CUDA-aware transport; no GPU-aware fallback warning
- Five warmup plus 20 measured NVT steps for sustained cases
- Candidate LAMMPS SHA256:
  `13895b4c85dc7c26f941381b0868f3e0199d467f0cb6013a8d7746fe1bbd0e2e`
- Model SHA256:
  `3cc5b7641dbd4c9d766d6140661187c4fd484bae53d4aab5293faa5e76224414`
- FP64 SM80 artifact SHA256:
  `31a90c97a146b424902dc460388fec65f9146e4d0242c4803f192139b6352b1f`

The artifact was prepared on the target GPU with
`symmetrix_prepare_jit_device_artifact --precision float64` and used with
`pair_style symmetrix/mace/kk`.

## Sustained results

Primary timing is in us/atom/step.

| Ranks | Selection | Plan | Planned bytes/rank | Peak/GPU | us/atom/step | H1 us/atom/eval |
|---:|---|---|---:|---:|---:|---:|
| 2 | Fixed disabled | `mh0-direct-capacity-y-only` | 60,672,722,484 | 62,173 MiB | 6.01980 | 0.040002 |
| 2 | Fixed forced | `mh0-dual-layer-tiled-v1` | 9,275,370,188 | 13,125 MiB | 6.78355 | 0.049076 |
| 1 | Fixed allowed, automatic | `mh0-dual-layer-tiled-v1` | 14,766,772,000 | 17,991 MiB | 13.44445 | 0.026568 |

All three sustained cases report one neighbor rebuild, zero dangerous builds,
one forward and one reverse hidden-state exchange per evaluation, and exit
status zero. The single-rank case uses periodic self-exchanges.

The two-rank fixed run completed ownership migration from 500,000 atoms/rank
to a 500,081 / 499,919 split. Rank 0 replaced its exact-size prepared graph
once when its receiver count first grew; subsequent geometry and neighbor-list
updates reused the fixed-workspace execution path.

Final thermodynamics agree to FP64 precision. At step 25, direct and fixed
total energies differ by approximately `2e-9 eV` over one million atoms, and
the one- and two-rank fixed results agree to printed precision.

## Small-system parity

A 320-atom, two-rank comparison forced either
`mh0-direct-capacity-y-only` or `mh0-dual-layer-tiled-v1`. The fixed plan loaded
the FP64 generated device artifact, reported zero fallback executions, and
matched energy and pressure to printed precision. Sorted per-atom position and
force dumps over two frames had a maximum force-component difference of
`4.88e-14 eV/Angstrom`.

## Selection behavior

`allow_fixed_workspace yes` grants permission; it does not force tiled
execution. At two ranks the 60.67-GB direct plan fits and automatic capacity
selection retains it. At one rank the direct estimate is 110.50 GiB, so the
same option automatically selects the 14.77-GB fixed-workspace plan and makes
the workload feasible. The authoritative evidence remains `plan=` plus the
workspace receiver, edge, byte, and batch counters.

## Communication copy check

A matched two-step run with the pre-optimization executable produced identical
thermodynamics. Its H1 communication time was 0.070004 us/atom/eval, versus
0.035269 us/atom/eval with the optimized FP64 `Kokkos::deep_copy` path, a
49.6% reduction. This optimization is retained in the fixed-workspace
candidate.
