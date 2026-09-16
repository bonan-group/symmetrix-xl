# Standard MH-0 host M1 term tiling

## Contract

The accepted experiment uses the standard MH-0 model at
`/tmp/mace-mh-0-current-Al-N.json` (SHA-256
`c202e5bc6da9a2b0ca4974595869fa5f4f0cdb7f37b58d87415ecd95f048eb22`)
and a periodic 864-atom wurtzite AlN `repeat((6, 6, 6))` cell. The model
cutoff and effective neighbor-list cutoff are 6.0 A, neighbor-list skin is
zero, and the graph contains 78,624 directed edges. Execution uses FP32
Kokkos OpenMP on physical core 2, with one Kokkos/OpenMP thread and one
OpenBLAS thread. Timings use three warmups and seven synchronized samples;
the primary metric is median `us/atom`.

The factorized control uses generated R1, Standard R0, Standard M0,
`m1_polynomial_policy="recompute"`, MH-0 adjoint reuse, and compact FP32
directions with FP64 radii. `SYMMETRIX_STANDARD_M1_HOST_TILE=runtime` selects
the original polynomial-graph recompute kernel in a fresh process.

## Implementation

Standard M1's scalar invariant has the same 94 fixed terms as output zero of
Standard M0. The host specialization reuses Standard M0's compile-time term
expansion and derivative functions, owns one node per outer Kokkos/OpenMP
iteration, and evaluates bounded channel-local tiles. Checkpoint terms are
canonicalized by degree and lexicographic order before weights are packed as
`[type, term, channel]`; exact topology admission prevents dispatch for other
polynomial structures.

The implementation reuses the existing fixed-size M1 weight allocation. Its
largest tile-16 local state is 2,304 bytes per active host worker in reverse,
and it allocates no atom- or edge-scaled M1 workspace. This first milestone is
limited to ordinary FP32 standard MACE on host. MACEField, FP64, CUDA, HIP,
and nonstandard M1 topologies retain their existing implementations.

## Results

| M1 implementation | Median us/atom |
| --- | ---: |
| Runtime polynomial graph | 261.098 |
| Standard M1 tile 1 | 214.569 |
| Standard M1 tile 4 | 208.634 |
| Standard M1 tile 8 | 207.209 |
| Standard M1 tile 16 | 206.904 |
| Final tile-16 confirmation | 208.303 |
| Retained M1 control | 235.021 |

Kokkos kernel timing over four evaluations measured the original M1 forward
plus reverse at 55.37 us/atom. The final tile-16 specialization measured
2.80 us/atom, a 19.8x M1-stage speedup. Process RSS was 296.63 MiB for the
generated recompute path, 296.49 MiB for graph recompute, and 465.27 MiB for
retained M1.

The checkpoint-generic `l_max=3`, `L_max=1` Standard M1 fast path is compiled
into the extension and remains active in both streamed modes for this MH-0
model. Channel count, species count, and learned weights remain runtime model
data, so no checkpoint-specific M1 artifact is precompiled. The current fast
path is nevertheless not architecture-generic: Standard MACE models with
`l_max=1` or `l_max=2` use the generic contraction implementation. A
transient experiment also enabled runtime-JIT R1 in
all-interactions and measured 214.74 us/atom, but that configuration was
rejected because all-interactions is the compiler-free generic deployment
path. With runtime JIT disabled, all-interactions measured 5,921.955 us/atom:
generic R1 consumed 5,812.502 us/atom while Standard M1 forward plus
reverse consumed only 4.208 us/atom. The matched factorized JIT run measured
214.253 us/atom, including 3.427 us/atom for Standard M1.

A fresh post-rebuild confirmation measured 5,919.958 us/atom for generic,
JIT-free all-interactions and 216.975 us/atom for factorized JIT, a 27.28x
difference. The all-interactions process used an intentionally invalid JIT
policy, still reported `jit_status="not_applicable"`, and recorded zero JIT
launches. Both runs used the same checkpoint-generic `l_max=3` Standard R0,
M0, and M1 fast paths.

The final factorized comparison against the independent materialized evaluator
reported `5.896e-4 eV` total-energy error (`6.824e-7 eV/atom`),
`7.360e-6 eV/A` maximum force error, and `6.653e-7 eV/A^3` maximum stress
error. The generated route reported zero polynomial scratch, zero active
polynomial values and adjoints, and one Standard M1 forward and reverse launch
per measured evaluation.

Evidence is stored in:

- `/tmp/mh0-standard-m1-host-runtime.json`
- `/tmp/mh0-standard-m1-admitted-tile1.json`
- `/tmp/mh0-standard-m1-admitted-tile4.json`
- `/tmp/mh0-standard-m1-admitted-tile8.json`
- `/tmp/mh0-standard-m1-admitted-tile16.json`
- `/tmp/mh0-standard-m1-retained-current.json`
- `/tmp/mh0-standard-m1-factorized-all-shared.json`
- `/tmp/mh0-standard-m1-final-tile16.json`
- `/tmp/mh0-standard-m1-final-parity.json`
- `/tmp/mh0-m1-stage-all-interactions-no-jit.json`
