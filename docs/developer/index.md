# Developer Guide

This section documents source ownership, backend builds, qualification, JIT,
and performance work. It is intentionally more detailed and less stable than
the user guide.

Use the generated {doc}`/reference/execution_support_matrix` and the user guide
for the current public contract. Reproduce qualification with the maintained
benchmark drivers and record the exact source, model, toolchain, hardware, and
workload before making support or performance claims.

```{toctree}
:maxdepth: 1

contributing
build_and_test
runtime_and_profiling
kokkos_programming_guide
architecture
execution_planner
adding_a_family
jit_and_artifacts
openmp_simd
```
