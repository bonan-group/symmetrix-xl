# Single-layer MACE HIP optimization (2026-08-31)

## Scope

This record covers FP32 HIP execution with `low_memory=True` and
`streamed_edges="direct"` for the extracted single-layer Al/N model. The
qualified device was an AMD Radeon 8060S (`gfx1151`). The primary workload was
864 atoms, a `6.0 A` model cutoff, `0.5 A` skin, `6.5 A` effective neighbor-list
cutoff, and 94,176 directed edges (109 edges/atom). Every measured evaluation
used a prepared graph, eight warmups, 25 samples, generated M0 `chunk32`,
built-in R0, Y-only harmonics, and zero fallback evaluations.

The pre-change extension was
`16ec52eb8620ed66fd528088cc999e0a0f766f7746284347ebfb3288625c99a5`.
The final candidate extension was
`a79c56926fab23de2516db4885a392eab72b9c4e2383837f2b6f9cef94b0dd01`.

## Changes

The single-layer R0 forward path now assigns one node/channel to each device
work item and accumulates all 16 angular components while traversing that
receiver's edge list once. The previous kernel launched one receiver/angular
order at a time and traversed the same edge list four times. Admission is
limited to the qualified HIP FP32 single-layer path; CUDA, FP64, host, and
two-layer execution retain the prior kernel.

The `gfx1151` FP32 built-in R0 reverse launch profile now uses eight persistent
blocks per compute unit. The exact 128-channel, nine-output, 923-term generated
M0 topology uses two persistent blocks per compute unit on `gfx1151`. Runtime
module diagnostics retain the selected M0 grid.

## Performance

| Model/path | Atoms | Directed edges | Median us/atom |
| --- | ---: | ---: | ---: |
| Single-layer, committed H1-transpose baseline | 864 | 94,176 | 10.605 |
| Single-layer, fused R0 with old launch defaults | 864 | 94,176 | 9.926 |
| Single-layer, final automatic selection | 864 | 94,176 | **9.532** |
| Single-layer, final policies at larger scale | 6,912 | 753,408 | **9.633** |
| Full OMAT, old R0 grid | 864 | 94,176 | 19.682 |
| Full OMAT, `gfx1151` R0 grid | 864 | 94,176 | **19.307** |

The final automatic single-layer path is 10.1% faster than the committed
H1-transpose baseline and 2.03x faster than full OMAT at the matched 864-atom
workload. At 6,912 atoms, the launch policies improve the fused path from
10.515 to 9.633 us/atom (8.4%). The R0 grid also improves full OMAT by 1.9%, so
it is architecture-profiled rather than single-layer-specific.

The final automatic diagnostic selected M0 at two blocks per compute unit and
the `standard-r0-module-module-v2-hip-float32-gfx1151` profile at eight blocks
per compute unit. It reported the direct low-memory path, Y-only harmonics, and
zero fallback evaluations.

## Numerical gate

The fused candidate and the pre-change extension evaluated the same 864-atom
graph. Energy was identical. Maximum force difference was
`2.554e-15 eV/A`, force RMS difference was `3.677e-16 eV/A`, and maximum
stress difference was `3.253e-19 eV/A^3`.

## Remaining opportunities

Before fusion, R0 forward/reverse and generated M0 forward/reverse accounted
for about 74% of traced device time. The next profile should remeasure that
distribution with the fused kernel. Likely follow-ups are a generated R0
forward specialized for the retained contract, reducing register pressure in
generated M0 reverse, and testing an all-angular-order R0 coordinate reverse.
Each requires a matched profile because the old attribution is no longer a
valid optimization ranking after this change.
