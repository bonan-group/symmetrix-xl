# Eight-GPU LAMMPS Kokkos communication investigation

Date: 2026-09-20
Host: `volc`
Artifacts: `/vepfs/symmetrix-bench/lammps-kokkos-comm-device-r8-debug-20260920`

## Outcome

The eight-GPU failure is caused by stale data in a grown generic LAMMPS Kokkos
CUDA-aware MPI buffer, not by Symmetrix H1 packing, pair callback ordering, or
neighbor binning.  Immediate destruction permits a retired CUDA allocation
address to be reused while HPC-X/UCX still behaves as though its old mapping is
active.  Retaining superseded Kokkos DualViews for the lifetime of
`CommKokkos` fixes the failure without `comm host`, transport forcing, or
unconditional synchronization.

The clean final executable is:

```text
/vepfs/symmetrix-bench/lammps-kokkos-comm-device-r8-debug-20260920/
  validation-build/install/bin/lmp
SHA256 235dcd1d7cb73e5b047041a9721842765c1ae5a7ad56e2c36463cb3212ce626b
```

It completed a final eight-rank `comm device` run after all diagnostic changes
were removed from the pair style and neighbor binning source was restored.

## Root-cause evidence

The first invalid state was isolated at generic border receive:

- owned atoms, send lists, source coordinates, and packed device records were
  valid;
- received device records were already unrelated/stale immediately after
  successful MPI completion;
- unpacking created finite ghosts with incorrect periodic images;
- neighbor binning was only the later detector;
- extra fences, CUDA synchronization, `MPI_Sendrecv`, Kokkos `resize()`, and a
  growth-only fence all failed;
- preallocating the buffers or retaining retired allocations both passed.

Eight-rank 64 MiB and 256 MiB multi-peer CUDA MPI probes passed.  Automatic
UCX selection remained enabled.  A separate provenance run records
`cuda_ipc/cuda` and `rc_mlx5` lanes under
`standard-lj/r8-t18/ucx-selected-lanes.txt`; `UCX_TLS` and
`UCX_NET_DEVICES` were unset and `UCX_MEMTYPE_CACHE=n`.

## Fix

The upstream-ready patch is
`benchmarks/lammps_cuda_aware_buffer_keepalive.patch`.  Before each geometric
growth, the old send or receive DualView is moved into a keepalive vector.
Because growth is geometric, the number of retained allocations is bounded.
They are released with `CommKokkos` at shutdown.

No Symmetrix source change is required.  Temporary pair-style assertions and
fences were removed after they established that callback indices, extents,
DualView state, and completion were valid.

## Correctness

Workload: 1,008,000 SrTiO3 atoms (`56 x 60 x 60`), OMAT0 medium FP32,
6.0-A model cutoff, 0.5-A skin, 6.5-A effective cutoff, 5 warmup plus 20
measured NVT steps, `mh0-direct-speed`, fixed workspace disabled.

| Ranks / GPUs | Result | Final PE (eV) | Final pressure (bar) | Pair us/atom/eval | H1 comm us/atom/eval |
|---:|---|---:|---:|---:|---:|
| 2 | pass | -8031374.671850 | 56282.9470 | 3.183712 | 0.009204 |
| 4 | pass | -8031374.668039 | 56282.9490 | 1.577581 | 0.009050 |
| 8, repetition 1 | pass | -8031374.666682 | 56282.9468 | 0.793303 | 0.007583 |
| 8, repetition 2 | pass | -8031374.667575 | 56282.9472 | 0.786817 | 0.008225 |
| 8, clean final binary | pass | -8031374.669284 | 56282.9467 | 0.789264 | 0.008198 |

Across the patched 2/4/8 decomposition, final potential energy spans 0.0052 eV
total (5.2e-9 eV/atom) and pressure spans 0.0023 bar, within the established
FP32 decomposition tolerance.  Every run changed ownership, reused prepared
graph storage (`graph replacements=1`, later `graph updates`), built one
measured neighbor list, and reported zero dangerous builds.  There were no
lost atoms, CUDA errors, neighbor-bin assertions, MPI fallback warnings, or
JIT fallbacks.

The standard `lj/cut/kk` regression is
`benchmarks/lammps_kokkos_comm_device_migration.in`.  It uses the same rank
grid and warmup/run boundary; `LAMMPS_COMM_CUTOFF=20.0` additionally forces a
generic border-buffer grow after warmup.  This standard-pair case is a negative
control under normal memory pressure, not a standalone positive reproducer.

## Performance

Primary end-to-end timings are the measured LAMMPS loop in us/atom/step.  The
host baseline is the qualified automatic-UCX `comm host` result; medians use
two repetitions.  Patched device values use one 2-rank, one 4-rank, and two
8-rank repetitions.

| Ranks / GPUs | `comm host` median | patched `comm device` | Device vs host |
|---:|---:|---:|---:|
| 2 | 3.18550 | 3.21388 | +0.89% |
| 4 | 1.59474 | 1.58632 | -0.53% |
| 8 | 0.79617 | 0.79799 | +0.23% |

The diagnostic-free final eight-rank result was 0.79545 us/atom/step.  The
patch therefore has no material throughput regression relative to host-staged
generic communication while preserving fully device-resident communication.

## Reproduction and artifacts

The exact standard-pair runner and input are committed beside this report.
The Symmetrix runner, source/build hashes, rank maps, GPU inventory and memory
traces, LAMMPS logs, failed experiments, final repetitions, and UCX provenance
are retained under the artifact root.  See
`benchmarks/lammps_kokkos_comm_device_upstream_report_20260920.md` for the
upstream issue text and patch rationale.
