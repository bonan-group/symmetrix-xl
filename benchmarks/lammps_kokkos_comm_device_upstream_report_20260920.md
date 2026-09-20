# Kokkos CUDA-aware MPI buffer address reuse on eight GPUs

## Summary

LAMMPS `comm device` can receive stale data after a Kokkos communication
buffer grows on an eight-GPU NVIDIA HPC-X 2.20 stack.  The first visible error
is usually `Atom outside of neighbor bin range`, but instrumentation shows that
neighbor binning is only the first consumer: the receive buffer already holds
unrelated finite values immediately after `MPI_Wait`.

Keeping superseded `k_buf_send` and `k_buf_recv` DualViews alive prevents the
device virtual addresses from being recycled and fixes the failure.  The
number of retained views is bounded by geometrically growing buffer capacity.
This is a compatibility workaround; the provider should also be investigated
for stale CUDA registration or IPC cache entries.

## Environment

- LAMMPS: `c8bd2ae5927ee236a8892dbd51c18a92cc9c33cf`
- Kokkos: 5.2.1, CUDA and Serial, `sm80`
- GPUs: eight NVIDIA A100-SXM4-80GB, one MPI rank per GPU
- MPI: NVIDIA HPC-X 2.20, Open MPI 4.1.7a1, UCX
- CUDA toolkit: 13.0
- `UCX_MEMTYPE_CACHE=n`; `UCX_TLS` and `UCX_NET_DEVICES` unset
- automatic lanes: `cuda_ipc/cuda` and `rc_mlx5` on four HCAs; the UCX
  installation also exposes `cuda_copy`

## Reproduction

The retained standard-pair regression uses `lj/cut/kk`, a `2x2x2` rank grid,
one million atoms, five warmup steps, and twenty measured steps.  It exercises
ownership migration after a run boundary and can force a generic border-buffer
grow at that boundary with `LAMMPS_COMM_CUTOFF=20.0`:

```bash
export TASK_ROOT=/vepfs/symmetrix-bench/lammps-kokkos-comm-device-r8-debug-20260920
export LMP=/path/to/lmp
export KOKKOS_COMM_MODE=device
export KOKKOS_COMM_EXCHANGE=device
export LAMMPS_COMM_CUTOFF=20.0
$TASK_ROOT/scripts/volc_lammps_kokkos_comm_device_reproducer.sh 8 1
```

This stock LJ case passes both before and after the patch under ordinary memory
pressure, so it is a focused migration regression rather than a standalone
positive reproducer.  The positive reproducer is the same LAMMPS path under
the large device-allocation pressure of the attached Symmetrix input:

```bash
export SYMMETRIX_LAMMPS_EXECUTABLE=/path/to/lmp
export SYMMETRIX_RUN_ROOT=$TASK_ROOT/reproduction
export SYMMETRIX_KOKKOS_GPU_AWARE=on
export SYMMETRIX_KOKKOS_COMM_MODE=device
export SYMMETRIX_KOKKOS_COMM_EXCHANGE=device
$TASK_ROOT/scripts/symmetrix-validation-runner.sh 8 1
```

The dependency does not supply the bad coordinates.  It changes allocator
pressure; the corrupted object is LAMMPS's generic atom-border receive buffer.

## Localization

Diagnostics around the first bad border exchange established the following in
order:

1. owned and ghost counts and atom capacity are valid before exchange;
2. all owned coordinates are finite and inside their subdomains;
3. the device send list contains valid indices and periodic-image choices;
4. the source coordinates and packed device records are correct;
5. immediately after MPI completion, received device records are stale and do
   not match the peer's packed records;
6. unpacking those finite but wrong records creates invalid periodic ghosts;
7. `NBinKokkos::bin_atoms` later reports the first public error.

Pair forward/reverse callbacks were fenced and their indices, ranges, and
DualView states checked without changing the result.  The first bad object is
therefore outside the pair-style communication buffers.

The following interventions did not fix the problem:

- `cudaDeviceSynchronize()` before/after MPI and after `MPI_Wait`;
- `MPI_Sendrecv` in place of `MPI_Irecv`/`MPI_Send`/`MPI_Wait`;
- Kokkos `resize()` in place of destructive reallocation;
- a fence only when a buffer grows;
- LAMMPS dangling send-buffer fix `00fc55c4` and later synchronization fixes.

Two interventions did fix it:

- increasing `BUFMIN` to 1,000,000 so no buffer grows;
- retaining old send/receive DualViews so their addresses cannot be reused.

Independent eight-rank, 64 MiB and 256 MiB multi-peer CUDA-buffer MPI probes
pass, so generic CUDA-aware MPI and CUDA IPC are functional.  The evidence
specifically implicates address lifetime and registration reuse.

## Proposed patch

Apply `benchmarks/lammps_cuda_aware_buffer_keepalive.patch` to the LAMMPS root.
It changes only `CommKokkos` buffer lifetime.  No host staging, transport
forcing, global fence, or pair-style workaround is introduced.

With the patch, two-, four-, and eight-rank Symmetrix cases pass; two separate
eight-rank repetitions complete all 25 steps with ownership migration, one
neighbor rebuild, zero dangerous builds, and no CUDA or lost-atom errors.
The standard LJ migration regression also passes with `comm device`.

## Open upstream question

The stock LJ test does not independently trigger stale receive data without
the larger allocator pressure.  This should be reported jointly to LAMMPS and
NVIDIA HPC-X/UCX: LAMMPS currently releases and may recycle an address that has
participated in CUDA-aware MPI, while this provider configuration appears to
retain address-keyed state beyond completion.  The keepalive patch is bounded
and proven on the affected stack, but provider invalidation would be the
preferable general fix if UCX confirms the stale registration.
