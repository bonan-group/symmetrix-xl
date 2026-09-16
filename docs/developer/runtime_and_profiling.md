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
