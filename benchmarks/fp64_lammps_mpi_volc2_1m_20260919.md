# Volc2 FP64 LAMMPS MPI usability

Date: 2026-09-19  
Host: `volc2`  
Remote evidence: `/vepfs/symmetrix-fixed-workspace-mpi/fp64-mpi-usability-20260919`

## Outcome

FP64 direct LAMMPS MPI is usable for the one-million-atom SrTiO3 workload on
two A100-80GB GPUs. With one MPI rank per GPU, `profile capacity` selects
`mh0-direct-capacity-y-only`, completes sustained NVT execution, and uses a
sampled peak of 62,173 MiB (60.72 GiB) per GPU.

FP64 fixed workspace is not implemented. `allow_fixed_workspace yes` is
accepted, but it does not select a fixed-workspace plan: the run still selects
`mh0-direct-capacity-y-only`, reports zero workspace bytes and batches, and has
the same memory and throughput as the no-opt-in run. Explicitly forcing
`mh0-dual-layer-tiled-v1` fails with the clear message
`mh0-dual-layer-tiled-v1 requires float32 precision`.

This distinction matters at the capacity boundary. The 1M-atom FP64 workload
does not fit on one A100: both with and without fixed-workspace opt-in it plans
118,649,035,584 bytes (110.50 GiB) and fails allocating the 38.15-GiB MH0 Phi1
forward state. The opt-in cannot rescue the workload because no FP64 tiled plan
is available.

## Workload and identities

- 1,000,000 SrTiO3 atoms from a `100 x 100 x 20` five-site cubic replication
- OMAT-medium standard MACE, two interactions, 128 channels, FP64
- 6.0 Angstrom model cutoff, 0.5 Angstrom skin, 6.5 Angstrom effective cutoff
- Initial 102,800,000 directed edges; 99,817,766 after the measured trajectory
- Two MPI ranks, one rank per A100-SXM4-80GB, `1x1x2` rank grid
- OpenMPI 4.1.6 CUDA-aware transport; no GPU-aware fallback warning
- Five warmup plus 20 measured NVT steps for sustained cases
- Candidate LAMMPS SHA256:
  `6e1b5292e2acc29053b2b584ef57afcfd0ba64ba1fbeaec2bfcd152403f52f9c`
- Model SHA256:
  `3cc5b7641dbd4c9d766d6140661187c4fd484bae53d4aab5293faa5e76224414`
- FP64 SM80 artifact SHA256:
  `31a90c97a146b424902dc460388fec65f9146e4d0242c4803f192139b6352b1f`

The artifact was prepared on the target GPU with
`symmetrix_prepare_jit_device_artifact --precision float64` and used with
`pair_style symmetrix/mace/kk`. The FP32 pair-style name is not valid for this
artifact.

## Sustained two-rank result

Primary timing is in us/atom/step. Both inputs use `profile capacity`; only
`allow_fixed_workspace` differs.

| FP64 input | Selected plan | Planned bytes/rank | Peak/GPU | us/atom/step | H1 us/atom/eval | H1 share |
|---|---|---:|---:|---:|---:|---:|
| Fixed workspace disabled | `mh0-direct-capacity-y-only` | 60,672,722,484 | 62,173 MiB | 6.00990 | 0.034065 | 0.567% |
| Fixed workspace allowed | `mh0-direct-capacity-y-only` | 60,672,722,484 | 62,173 MiB | 6.00965 | 0.037089 | 0.617% |

The 0.004% throughput difference is noise. Both cases report one forward and
one reverse H1 exchange per evaluation, one neighbor rebuild, zero dangerous
builds, graph reuse without replacement growth, and identical final energy to
printed precision. Ownership changes from 500,000 atoms/rank to a 500,081 / 
499,919 split during the trajectory.

The capacity log says `fixed workspace=allowed` for the opt-in case, but the
selection evidence is `workspace receivers=0, edges=0, bytes=0, batches=0` and
one dedicated communicated-H1 allocation. Users must inspect `plan=` and the
workspace counters; the word `allowed` alone is not evidence that fixed
workspace is active.

## Boundary and rejection behavior

| Case | Result | Relevant evidence |
|---|---|---|
| Two ranks, fixed disabled | Pass | Capacity Y-only, 60.72 GiB sampled/GPU |
| Two ranks, fixed allowed | Pass | Same non-tiled plan and memory |
| One rank, fixed disabled | Fail | 110.50-GiB plan; 38.15-GiB allocation failure |
| One rank, fixed allowed | Fail | Same plan and allocation failure |
| Two ranks, tiled plan forced | Expected fail | `requires float32 precision` |

The forced-plan rejection occurs at Verlet setup after atom and neighbor-list
construction, not during `pair_coeff`. The message is actionable, but an
earlier validation or explicit warning when FP64 ignores fixed-workspace opt-in
would improve usability and avoid expensive setup before failure.

## FP64 communication copy check

A matched two-step run with the pre-optimization executable produced identical
thermodynamics. Its H1 communication time was 0.070004 us/atom/eval, versus
0.035269 us/atom/eval with the optimized FP64 `Kokkos::deep_copy` path, a
49.6% reduction. End-to-end short-run time changed from 5.88640 to 5.85135
us/atom/step. Longer runs are required for a robust end-to-end performance
claim, but correctness and the intended communication-path change are clear.

## Usability decision

- **Supported:** FP64 CUDA-aware LAMMPS MPI with retained/non-tiled capacity
  execution, using `symmetrix/mace/kk` and a matching FP64 device artifact.
- **Not supported:** FP64 single- or dual-layer tiled fixed workspace.
- **Potentially misleading:** `allow_fixed_workspace yes` is permission, not a
  guarantee, and currently degrades silently to a non-fixed FP64 plan.
- **Capacity implication:** the tested 1M workload requires two A100-80GB GPUs;
  fixed-workspace opt-in does not make the one-GPU case feasible.

