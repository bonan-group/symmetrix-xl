# Factorized Fractional Geometry GPU Result

The compact-MACE MH-0 ASE factorized path was measured on an RTX 5090 with
CUDA 13.3, float32, the current source and extension, and the official OMAT-0
medium compact artifact. Both ensembles used the same seeded wurtzite AlN
states, 20 steps of 1 fs, a 6.0 A model cutoff, a 0.5 A skin, and a 6.5 A
effective neighbor cutoff. NVE values below are new current-code measurements;
historical direct execution NVE values are not used.

| Ensemble | Atoms | Directed edges at 6.5 A | Median (us/atom) |
|---|---:|---:|---:|
| NVE | 4,000 | 436,000 | 4.131 |
| NPT | 4,000 | 436,000 | 4.236 |
| NVE | 6,912 | 753,408 | 4.120 |
| NPT | 6,912 | 753,408 | 4.225 |

NPT is 1.025 times NVE at both sizes. Relative to the pre-optimization current
NPT results (7.734 and 7.654 us/atom), this recovers 97.1% and 97.0% of the
measured NPT/NVE gap. Fixed-cell NVE changed by less than 1% from its previous
current-code measurement.

Each NPT run completed 21 calculator evaluations with one neighbor topology,
one prepared graph, one factorized schedule, one fractional geometry
initialization, 20 cache reuses, 19 cell updates, and zero Cartesian host
geometry materializations or factorized fallbacks. The 19 cell updates copied
2,736 bytes in total.

A matched, fully warmed two-step Nsight Systems capture at 4,000 atoms reported:

| CUDA activity | NVE | NPT |
|---|---:|---:|
| Kernels | 82 | 82 |
| Kernel time (ms) | 29.277 | 29.428 |
| H2D copies | 2 / 192,000 bytes | 6 / 192,288 bytes |
| `cudaMalloc` calls | 32 | 32 |
| `cudaFree` calls | 0 | 0 |
| `cudaStreamSynchronize` calls | 2 | 2 |

Thus two NPT cell updates add four transfers and 288 bytes, with no added
kernels, allocator calls, frees, or synchronizations. Raw campaign records and
Nsight reports are retained under
`benchmarks/.artifacts/omat0-medium-aln-factorized-gpu-cell-20260814/`.
