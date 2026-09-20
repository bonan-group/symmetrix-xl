# Volc LAMMPS Kokkos comm-device recipe

This recipe qualifies the LAMMPS `c8bd2ae5` Kokkos CUDA-aware communication
workaround on eight A100 GPUs.  Use fresh source, build, and install paths; do
not modify the retained release build.

```bash
task=/vepfs/symmetrix-bench/lammps-kokkos-comm-device-r8-debug-20260920
src=$task/validation-build/lammps
build=$task/validation-build/build
install=$task/validation-build/install

cd "$src"
git rev-parse HEAD
git apply /path/to/symmetrix/benchmarks/lammps_cuda_aware_buffer_keepalive.patch
cmake --build "$build" -j 16
cmake --install "$build"
sha256sum "$install/bin/lmp"
```

The retained build was configured for CUDA 13.0, `sm80`, Kokkos CUDA+Serial,
HPC-X 2.20, and the Symmetrix pair package.  Preserve its recorded CMake cache
when reproducing the binary; use a separate task root for a fresh configure.

Standard LAMMPS migration regression:

```bash
export TASK_ROOT=$task
export LMP=$install/bin/lmp
export KOKKOS_COMM_MODE=device
export KOKKOS_COMM_EXCHANGE=device
export LAMMPS_COMM_CUTOFF=20.0
$task/scripts/volc_lammps_kokkos_comm_device_reproducer.sh 8 1
```

Symmetrix acceptance run:

```bash
export SYMMETRIX_LAMMPS_EXECUTABLE=$install/bin/lmp
export SYMMETRIX_RUN_ROOT=$task/acceptance-device
export SYMMETRIX_KOKKOS_GPU_AWARE=on
export SYMMETRIX_KOKKOS_COMM_MODE=device
export SYMMETRIX_KOKKOS_COMM_EXCHANGE=device
$task/scripts/symmetrix-validation-runner.sh 8 1
```

For every run, leave `UCX_TLS` and `UCX_NET_DEVICES` unset, set
`UCX_MEMTYPE_CACHE=n`, use one rank per GPU and one host thread per rank, and
record the executable/input/model/JIT hashes.  Use `UCX_LOG_LEVEL=info` only in
a separate provenance run; do not use that run for timing.

Expected final executable SHA256:
`235dcd1d7cb73e5b047041a9721842765c1ae5a7ad56e2c36463cb3212ce626b`.
