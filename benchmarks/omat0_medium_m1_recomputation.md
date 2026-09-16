# OMAT-0 Medium Bounded M1 Recomputation

## Result

Bounded M1 polynomial recomputation is implemented as an opt-in CUDA float32
policy. The 32-channel tile is the preliminary choice. It removes both
graph-sized `[node,142,128]` polynomial views, uses 36,352 bytes of team scratch,
and does not regress end-to-end force evaluation at representative production
sizes. The retained path remains the default and rollback until the separate
large-scale and cross-device qualification is complete.

At 32,000 atoms, sampled GPU use fell from 9,448 MiB to 5,008 MiB while the
seven-sample median changed from 138.600 ms to 137.962 ms (-0.46%). A one-shot
n31 probe completed at 119,164 atoms and 10,843,924 directed edges with
16,290 MiB sampled GPU use and zero M1 polynomial value/adjoint capacity.

## Frozen Contract

- Model SHA-256: `55ed1b6f689eb92f77fb7fd10a4eb11008f4520934e63c0b0cf5cd5c7b111100`
- Measured extension SHA-256: `0c8b779a8212974435a15dc79c0696c921e2a0ae0616491320791aaab698ec50`
- Final extension SHA-256: `04497616385fafa1e930e0448c3e7d082332f48ef23556abc7935ebaab01ce2e`
- Backend: Kokkos CUDA float32, generated direct R1 execution forward/reverse, R0 Edge16,
  generated M0
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0
- Driver/toolkit: 610.43.02 / CUDA 13.3.73
- Build: Release, CUDA sm_120
- Repeated timing contract: three warmups and seven synchronized evaluations

The final rebuild differs from the measured binary only in making the selected
tile 32 the inert evaluator default. Every measured recompute run already set
tile 32 explicitly; retained remains the policy default.

## Tile Screen

The short screen used 864 atoms, one warmup, and two synchronized samples.

| Policy | Tile | Scratch | Median (ms) | Versus retained |
|---|---:|---:|---:|---:|
| retained | 32 | 0 B | 4.0927 | reference |
| recompute | 8 | 9,088 B | 4.3749 | +6.89% |
| recompute | 16 | 18,176 B | 4.2479 | +3.79% |
| recompute | 32 | 36,352 B | 4.1525 | +1.46% |

Tile 32 is both the fastest candidate and below the available per-team shared
memory limit for the admitted OMAT-medium graph.

## Scale Results

| Atoms | Edges | Retained (ms) | Recompute (ms) | Change | GPU saved |
|---:|---:|---:|---:|---:|---:|
| 864 | 78,624 | 4.0401 | 4.0980 | +1.43% | 120 MiB |
| 6,912 | 628,992 | 29.0806 | 29.0118 | -0.24% | 960 MiB |
| 32,000 | 2,912,000 | 138.5997 | 137.9624 | -0.46% | 4,440 MiB |

The exact exposed storage reduction is linear. At 32,000 atoms, retained values
and adjoints are 2,326,528,000 bytes each; recompute reports zero active and
capacity bytes for both.

## Numerical Gate

The focused CUDA lifecycle test compares retained with recompute tiles 8, 16,
and 32 for total energy, per-node energies, forces, stress, M1, and A1 adjoints.
All arrays pass `rtol=0`, `atol=3e-5`. A repeated tile-32 call is bitwise exact,
and switching back to retained restores bitwise-identical retained outputs and
reallocates the two graph views. Invalid policies and tile widths are rejected.

The prototype is deliberately rejected for float64, non-CUDA builds, and
field-coupled models.

## ASE MD Gate

Paired fresh-process 20-step n6 NVE workers used the exact same saved state.
The initial records, all physical samples, MD metadata, and physics summaries
are exactly equal. Both policies report total-energy drift of
`3.979934938e-5 eV/atom`, within the `1e-3 eV/atom` production threshold.
Retained/recompute step medians are 41.229/40.597 ms (-1.53%), and final sampled
process GPU use is 1,080/960 MiB.

## Nsight Attribution

One n20 evaluation was filtered to the
`symmetrix_full_evaluator::direct` NVTX interval with Nsight Systems 2026.1.3.

| M1 kernel | Retained (ms) | Recompute (ms) |
|---|---:|---:|
| forward | 2.3294 | 1.8690 |
| reverse | 5.5860 | 5.2986 |
| total | 7.9154 | 7.1676 |

The recomputed M1 pair is 9.45% faster in this trace despite reconstructing the
DAG in reverse, consistent with eliminating graph-sized global-memory traffic.
The filtered interval contains no device-to-host or device-to-device transfer;
recompute also uses two fewer memset records. Raw trace output is generated
under the ignored `benchmarks/.artifacts/` tree and is not retained in Git.

## Decision

Keep `retained` as the default. Expose `recompute` with tile 32 for controlled
capacity experiments and proceed to larger-system and cross-device MD gates.
The preliminary numerical, ASE, performance, and capacity evidence is strong
enough to continue, but not sufficient to replace the generic rollback across
devices and models.
