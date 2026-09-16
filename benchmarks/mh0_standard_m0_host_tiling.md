# Standard M0 host channel tiling

## Contract

- Model: `/tmp/mace-mh-0-current-Al-N.json`, SHA256
  `c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`
- Structure: 864-atom periodic wurtzite AlN, `repeat((6, 6, 6))`
- Model cutoff: 6.0 A
- Neighbor-list skin: 0 A; effective cutoff: 6.0 A
- Directed edges: 78,624
- Precision: FP32 unless noted
- Execution: one physical core, one Kokkos/OpenMP thread, one OpenBLAS thread
- Factorized R1: generated host plugin
- Factorized state: M1 recompute, adjoint reuse, FP32 direction with FP64 radius
- Timing: three warmups and seven synchronized evaluations; median us/atom

## Timing

The host Standard M0 implementation evaluates all 422 fixed polynomial terms
over a bounded channel tile. No atom- or edge-scaled workspace is added.

| Build | Tile 1 | Tile 4 | Tile 8 | Tile 16 |
| --- | ---: | ---: | ---: | ---: |
| Native AVX-512, full inference (us/atom) | 315.81 | 306.10 | 274.89 | 263.65 |
| Forced Haswell/AVX2, full inference (us/atom) | 326.67 | 316.32 | 294.12 | 280.67 |

For native AVX-512, the combined Standard M0 forward and reverse timer fell
from 70.96 us/atom at tile 1 to 19.44 us/atom at tile 16. The forced AVX2 build
used `-march=core-avx2 -mtune=core-avx2`; GCC's vectorization report confirmed
vectorized tiled term loops. A native `-march=native` GCC report likewise
confirmed 64-byte vectorization in both the forward term loop and the reverse
term loop. Tile 16 is therefore the host default.
`SYMMETRIX_STANDARD_M0_HOST_TILE=1` restores the scalar rollback in a fresh
process. Tiles 4 and 8 remain available for diagnosis.

After making tile 16 the default, the complete shared-path qualification was:

| Mode | Kokkos threads | BLAS threads | Median us/atom |
| --- | ---: | ---: | ---: |
| Factorized low-memory | 1 | 1 | 268.31 |
| All interactions | 1 | 1 | 239.58 |
| Factorized low-memory | 4 | 1 | 69.11 |
| All interactions | 4 | 1 | 65.75 |

All four runs used 864 atoms, the 6.0 A model cutoff, zero skin, and 78,624
directed edges. Both modes selected generated R1, Standard R0 receiver-owned
execution, and Standard M0. Four-thread runs were pinned to physical cores
0-3 with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.

## Numerical qualification

Tile 1 and tile 16 were run in separate fresh processes against identical
graphs. MACEField used an electric field of `[0.01, -0.02, 0.03]` V/A.

| Model and precision | Energy (eV) | Force (eV/A) | Stress (eV/A^3) | Field adjoint |
| --- | ---: | ---: | ---: | ---: |
| MH-0 FP32 | 7.56e-7 | 6.51e-6 | 1.63e-8 | n/a |
| MH-0 FP64 | 9.09e-13 | 1.18e-14 | 5.10e-16 | n/a |
| MACEField FP32 | 6.65e-5 | 6.76e-6 | 4.51e-8 | 9.91e-5 |
| MACEField FP64 | 3.64e-12 | 1.46e-14 | 1.46e-16 | 2.56e-13 |

These are maximum absolute differences. The tiled loop changes FP32 summation
order but remains within the established direct-kernel force and stress gates;
FP64 differences are at roundoff scale.
