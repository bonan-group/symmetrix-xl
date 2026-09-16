# OpenMP and SIMD

Every value written inside an `omp simd` loop must be lane-private, disjointly
indexed, or protected by an explicit reduction. Do not invoke external CBLAS
concurrently from Kokkos OpenMP workers in production paths. Correctness
qualification must prove actual worker participation and compare full
properties in fresh processes.

The normative coding and review checklist is
{doc}`/openmp_simd_coding_standard`.

```{toctree}
:hidden:

/openmp_simd_coding_standard
```
