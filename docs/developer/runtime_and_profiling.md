# Runtime and Profiling

Record the imported Python package, extension path and hash, Kokkos execution
space, model hash, precision, graph dimensions, cutoff, and skin before
qualifying performance. Use a task-specific JIT cache and warm all artifacts
before steady-state measurements.

Use Nsight Systems to identify CUDA timing and transfer bottlenecks before
collecting targeted Nsight Compute metrics. On CPU, bind the process before
Python imports BLAS or Kokkos, pin BLAS to one thread when OpenMP supplies outer
parallelism, and treat the one-physical-core result as the primary optimization
metric.

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/symmetrix-profile/system \
  /path/to/warmed-command
nsys stats /tmp/symmetrix-profile/system.nsys-rep
```

Use Nsight Compute only after identifying a kernel family, with identical
baseline and candidate inputs. Its replayed elapsed time is not an end-to-end
benchmark result.

## LAMMPS communication attribution

`compute symmetrix/timing` is the production-level attribution for the
blocking hidden-state exchange inside Symmetrix pair evaluation. Its local
counters use the LAMMPS steady wall clock. On Kokkos paths the communication
timer starts after the existing pre-communication fence and stops after the
existing post-communication fence; do not add a fence to refine this timer.
The compute separately enables fences around the full Kokkos pair call. These
are required so the percentage denominator contains completed accelerator work,
and they are absent unless the compute is defined. Measure their effect with
alternating warmed runs before treating the summary as non-perturbing.

The compute selects the rank with maximum measured Symmetrix pair time before
forming `SxH1CommPct`, `SxPair`, `SxNonComm`, and `SxH1Comm`. Do not form a
compute-only estimate by subtracting independently reduced maxima. Rank
min/mean/max percentages and max/mean communication imbalance are secondary
load-balance diagnostics. MPI wait caused by uneven rank work is intentionally
included, so call the measurement blocking communication-path time rather than
network latency.

Query the compute once after a warmed measurement block. A query performs an
`MPI_Allgather` and must not be placed in every measured timestep. Cross-check
host-staged and device-direct results with Nsight Systems or `rocprofv3`, and
report LAMMPS `Pair`, ordinary `Comm`, `SxNonComm`, and `SxH1Comm` separately in
`us/atom/pair-evaluation`, together with the required workload and binary
provenance.
