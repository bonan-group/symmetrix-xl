# MH-1 staged direct CPU implementation plan

## Objective

Refactor the CPU/OpenMP `pair_spline_v1` implementation into the same explicit
execution structure used by standard MACE direct execution:

```text
prepare graph
geometry and harmonics
R0 forward
M0 forward
R1 forward
M1 forward
M1 reverse
R1 reverse
M0 reverse
R0 reverse
force, stress, and ZBL assembly
```

The retained implementation must not evaluate a conditioner or density MLP per
edge. Ordered source/target-pair splines provide the complete tensor-product
weights, density values, and their distance derivatives.

This is a migration of `pair_spline_v1`, not a new family of public execution
modes. `mlp_reference` remains the numerical oracle while the migration is in
progress. A milestone may use an internal adapter for validation, but that
adapter must be removed or absorbed before the milestone is accepted.

## Qualification workload

The primary performance decision uses:

- one pinned physical CPU core, excluding its SMT sibling;
- one Kokkos/OpenMP thread and one BLAS thread;
- FP32;
- the official MH-1 `omat_pbe` Al/N model;
- periodic `6 x 6 x 6` wurtzite AlN, 864 atoms;
- model cutoff 6.0 A, zero skin, effective cutoff 6.0 A;
- 78,624 directed edges;
- a prepared graph and warmed generated artifact;
- `us/atom` as the primary metric.

The current controls on this workload are approximately 236.13 us/atom for
standard MACE direct and 10,836.91 us/atom for MH-1 `pair_spline_v1`.
Instrumented MH-1 R0+R1 forward and reverse account for 83.7% of the call.

## Current structural problem

The spline implementation currently branches inside
`MaceNonlinearKokkosT::compute_node_energies_forces`. Both message-passing
layers share one monolithic loop and identical kernel names. More importantly,
selecting `pair_spline_v1` disables both the generated graph-wide program and
the generated node program. It therefore uses the generic 1,024-edge blocked
pipeline and materializes edge inputs, messages, message adjoints, source
adjoints, harmonic adjoints, and complete weight rows.

The generated ABI expects a radial conditioner prefix followed by a final
linear projection. The spline owns the complete final weight instead, so the
existing ABI cannot consume it without either reconstructing a fictitious
prefix or materializing a graph-wide weight matrix. Neither is acceptable.

## Stage contracts

### R0 and R1

One parameterized R-stage implementation serves layers 0 and 1. The runtime
driver invokes it through explicit R0 and R1 boundaries.

Forward inputs:

- prepared receiver CSR and source indices;
- ordered source and target types;
- distances and spherical harmonics;
- packed source node state;
- per-layer ordered-pair spline coefficients;
- generated tensor-product structure and output mask.

Forward outputs:

- packed receiver message rows;
- one density value per receiver.

The receiver owner traverses its edge list once. It evaluates the spline
interval once per edge, computes complete dynamic TP weights on demand, and
accumulates directly into the receiver message and density rows. It must not
materialize graph-wide `edge_up`, `edge_messages`, `weights`, or density rows.

Reverse inputs:

- retained or reproducible forward node state;
- message and density adjoints from the corresponding M stage;
- prepared source-owned edge schedule;
- the same geometry, harmonics, and spline coefficients.

Reverse outputs:

- source node-state adjoints;
- harmonic adjoints;
- one distance adjoint per edge.

The source owner traverses each source edge once, accumulates the bounded source
row locally, and writes each edge's harmonic and distance adjoints exactly once.
Spline value and derivative evaluation are fused with the generated TP reverse.
There is no complete edge-weight-adjoint array.

R0 may specialize fixed element-derived source state, but it must implement the
same external contract as R1. Any R0-only specialization requires a matched R1
control and must not create a separate execution mode.

### M0 and M1

One parameterized M-stage implementation serves layers 0 and 1. The driver
invokes explicit M0 and M1 boundaries. Each stage owns:

- residual and skip transforms;
- message linear transforms;
- density normalization;
- scalar and equivariant gates;
- compiled products;
- output update and readout;
- the exact reverse operations needed to produce message, density, and input
  adjoints.

The R/M boundary uses one persistent packed `ir_mul` layout. Conversions between
native `mul_ir` and generated `ir_mul` layouts are forbidden in the steady-state
path. The generated node program remains responsible for M0/M1 once its input
is produced directly in that layout by the spline-aware R stage.

## Host generated ABI

Add one host ABI generation for the staged spline path. It contains two
R-stage descriptors and reuses the existing generated M-stage node programs.
Each R descriptor records dimensions and receiver/source owner counts and
provides:

- `r_forward_owner(args, receiver_owner)`;
- `r_reverse_owner(args, source_owner)`.

The R packet contains a read-only spline view:

```text
h, x0, pair_count, interval_count, function_count, coefficients
```

Coefficients use the existing contiguous layout
`[ordered_pair, interval, coefficient, function]`. All flattened offsets and
extents use 64-bit arithmetic. The packet also carries the exact interaction
index and validates that `function_count == weight_count + 1`, with the final
function reserved for density.

The ABI must use a new query symbol and capability bit rather than changing the
size of an existing ABI-v3/v4 structure. Old artifacts remain loadable for
`mlp_reference`; `pair_spline_v1` requires the staged spline capability once
the new dispatch is enabled.

## Implementation milestones

### Milestone 0: explicit stage driver

- Define canonical R0, M0, R1, and M1 stage identities.
- Add stage-scoped profiling regions with distinct forward and reverse names.
- Separate generic `linear_up` work into R and residual/skip work into M,
  preserving evaluation order where it affects floating-point accumulation.
- Make the forward and reverse driver order explicit.
- Add counters or test hooks proving each stage executes once per evaluation.

Acceptance:

- FP32/FP64 generic and pair-spline results remain within existing tolerances;
- no new public selector;
- one-thread `pair_spline_v1` performance does not regress by more than 1%;
- profiles report R0, M0, R1, and M1 separately.

### Milestone 1: spline-aware generated R forward

- Add the host spline R packet, descriptor, loader validation, and codegen.
- Generate one receiver-owned forward program for R0 and one for R1.
- Evaluate complete weights and density directly from spline coefficients.
- Write packed node messages directly in the M-stage layout.
- Dispatch from `pair_spline_v1` only when the prepared direct graph and the
  matching staged artifact are available.
- Remove block-local forward weights, edge-up, edge-message, and density arrays
  from the staged path.

Acceptance:

- R0 and R1 forward agree with `mlp_reference` and generic spline controls;
- exactly two R forward launches per evaluation;
- zero layout-conversion launches at the R/M boundary;
- zero fallback evaluations under required direct execution;
- record R0 and R1 forward `us/atom` independently.

### Milestone 2: spline-aware generated R reverse

- Generate source-owned R0 and R1 reverse programs.
- Fuse TP source, harmonic, weight, density, and spline-distance derivatives.
- Accumulate a bounded source row locally and write each edge derivative once.
- Remove block-local edge-message, edge-up, weight-adjoint, and harmonic-adjoint
  arrays from the staged path.

Acceptance:

- analytic force and stress parity against `mlp_reference`;
- finite-difference checks exercise both TP-weight and density derivatives;
- exactly two R reverse launches per evaluation unless a measured split owner
  is retained for a documented reason;
- no graph-sized weight or weight-adjoint allocation;
- record R0 and R1 reverse `us/atom` independently.

### Milestone 3: generated M0/M1 integration

- Re-enable generated node programs for `pair_spline_v1`.
- Consume R output directly in persistent `ir_mul` layout.
- Retain only node state required by reverse; recompute bounded temporaries.
- Give generated phases canonical M0/M1 profiling names.
- Remove the previously rejected layout adapter rather than reviving it.

Acceptance:

- zero R/M layout conversions;
- M0 and M1 forward/reverse each execute once;
- numerical parity and lifecycle tests pass;
- one-thread timing improves over the accepted Milestone 2 result.

### Milestone 4: production selection and cleanup

- Make the staged implementation the CPU/OpenMP behavior of
  `pair_spline_v1`.
- Keep `mlp_reference` as the only reference implementation.
- Remove unreachable generic spline branches and obsolete workspace.
- Document artifact preparation, cache identity, and failure diagnostics.
- Defer CUDA/HIP enablement until the CPU contracts and ABI are stable.

Acceptance:

- required direct execution never silently falls back;
- focused MH-1 tests, the applicable full test suite, and pre-commit pass;
- fresh-process extension, model, artifact, and binary hashes are recorded;
- primary one-thread benchmark and stage profile are reproducible.

## Correctness matrix

At minimum, test:

- FP32 and FP64 where supported;
- zero edges, one edge, and non-divisible node/edge counts;
- ordered pairs `(a,b)` and `(b,a)`;
- regular 864-atom AlN, a perturbed cell, and a vacancy;
- energies, per-atom energies, forces, stress, and repeat determinism;
- spline nodes, interval interiors, short distances, and the cutoff boundary;
- prepared-graph reuse and stale-token rejection;
- artifact/model/layout fingerprint mismatch rejection;
- `mlp_reference` behavior unchanged.

## Performance decisions

Optimization decisions use stage time, not launch count alone. The first target
is the R reverse tensor contraction, followed by R forward message reduction.

## Progress record: 2026-08-23

Milestone 0 is complete. A fresh OpenMP build reports each canonical stage once
per evaluation, with the exact prior energy of `-6423.653198531983 eV`. The
864-atom cold-call profile (one physical core, one Kokkos/OpenMP/BLAS thread)
was:

| Stage | us/atom |
|---|---:|
| R0 forward | 931.101 |
| M0 forward | 472.056 |
| R1 forward | 2,684.902 |
| M1 forward | 231.701 |
| M1 reverse | 253.867 |
| R1 reverse | 4,528.820 |
| M0 reverse | 693.932 |
| R0 reverse | 1,156.542 |

Milestone 1 has started with an additive ABI-v5 query. The implemented packet
contains receiver CSR, ordered-pair types, distances, cubic coefficients,
harmonics, output mask, source state, and direct message/density outputs. The
generated receiver owner evaluates one spline interval per edge and writes no
graph-wide weight, edge-up, edge-message, or density matrix. ABI-v3 and ABI-v4
descriptor sizes and query symbols are unchanged.

The generated v5 artifact compiles and an executable two-layer equivalence test
checks its messages against the established generated TP owner using identical
weights, as well as its fused density accumulation. Evaluator dispatch remains
disabled at this checkpoint: the current generic reverse still assumes
forward-materialized edge intermediates. The next implementation slice must
either recompute those bounded intermediates for the reverse control or land
the source-owned spline reverse owner before enabling production selection.
Spline micro-optimization is not a priority: current spline value and derivative
work is only about 2.1% of the complete MH-1 call.

The v5 receiver-owned R forward is now dispatched by `pair_spline_v1` while the
existing recomputing reverse remains in place. A matched warmed comparison with
standard MACE used the same fresh OpenMP extension, one pinned physical core,
one Kokkos/OpenMP/BLAS thread, FP32, 864-atom periodic AlN, 78,624 directed
edges, a 6.0 A model cutoff, and zero neighbor skin. The standard control used
`m1_polynomial_policy=recompute`, `mh0_state_policy=reuse-adjoints-v1`, and
`edge_geometry_policy=unit-f32-radius-f64-v1`. Each value below is the mean of
the seven instrumented calls comprising two warmups and five measured calls:

| R forward stage | Standard MACE (us/atom) | MH-1 staged spline (us/atom) | MH-1 / standard |
|---|---:|---:|---:|
| R0 | 19.052 | 13.020 | 0.683x |
| R1 | 22.896 | 48.395 | 2.114x |
| Combined | 41.948 | 61.416 | 1.464x |

These are forward-region measurements inside a complete energy/force call, so
reverse time is excluded from the entries even though no separate public
forward-only evaluator is used. The MH-1 generated spline owner itself accounts
for 46.187 us/atom across both layers. The temporary input and output layout
adapters account for another 0.469 and 4.360 us/atom respectively. Removing
both adapters in Milestone 3 can therefore recover at most about 4.83 us/atom
on this workload; it cannot close the R1 gap by itself. The forward comparison
supports continuing with source-owned R reverse first: the warmed complete
MH-1 call is 7,337.89 us/atom and the existing reverse remains dominant.

Milestone 2 now has a source-owned host implementation. One generated owner per
unique source traverses that source's prepared edge schedule, recomputes cubic
spline values and derivatives, retains one source-adjoint row locally, and
writes each edge harmonic and radial adjoint once. It does not allocate or
launch the old blocked weights, weight adjoints, edge-up, edge-message-adjoint,
or harmonic-adjoint workspaces. The trace reports two physical generated
source-reverse launches per evaluation, one for R1 and one for R0, and none of
the prior 154-block kernel families.

On the same warmed one-core workload, the result is:

| Metric | Receiver-forward checkpoint (us/atom) | Source-reverse checkpoint (us/atom) | Speedup |
|---|---:|---:|---:|
| R0 reverse | 1,160.267 | 26.917 | 43.11x |
| R1 reverse | 4,499.532 | 100.959 | 44.57x |
| Combined R reverse | 5,659.799 | 127.877 | 44.26x |
| Complete energy/force call | 7,337.888 | 1,810.215 | 4.05x |

The uninstrumented complete-call samples were 1,563.524, 1,564.025,
1,563.487, 1,569.118, and 1,577.864 ms, with the reported median above. On a
perturbed 864-atom structure, the staged result differs from the generic spline
control by 7.01e-6 eV total energy, 1.25e-6 eV/A maximum force, and 1.17e-8
eV/A^3 maximum stress. Against `mlp_reference`, the corresponding differences
are 2.67e-6 eV, 4.09e-5 eV/A, and 1.54e-7 eV/A^3. The generated-owner
directional derivative test covers source state, harmonics, spline radius, and
density derivatives for both layers.

The next target is Milestone 3. The current warmed stage profile is 434.786
us/atom for M0 forward, 210.346 us/atom for M1 forward, 244.554 us/atom for M1
reverse, and 748.448 us/atom for M0 reverse. M0 reverse is now much larger than
either R stage, so generated M0/M1 integration has higher leverage than further
R-owner tuning.

Milestone 3 screened two generated-M integration strategies and retained
neither. Enabling the complete ABI-v4 node program increased the complete call
to 4,517.350 us/atom. Its generated product-forward owner took about 2.40 s per
call, compared with about 0.31 s for the compiled Kokkos product, so generated
product execution is not a viable phase boundary for this CPU path.

The second strategy kept the compiled product and made the R/M boundary
persistently row-major `ir_mul`. Explicit mixed-layout linear and gate methods
restored numerical parity: against the generic spline control, the total-energy,
maximum-force, and maximum-stress differences were 7.47e-6 eV, 1.35e-6 eV/A,
and 1.23e-8 eV/A^3. Performance nevertheless regressed from 1,810.215 to
2,448.040 us/atom, or 35.2%. A two-evaluation Kokkos Tools trace attributed
175.8 ms/call to row-major gate forward, 204.3 ms/call to row-major gate reverse,
and 193.0 ms/call to `ir_mul` reverse packing. The M stage results were 546.9
us/atom for M0 forward, 371.4 us/atom for M1 forward, 495.7 us/atom for M1
reverse, and 883.8 us/atom for M0 reverse. These strided accesses erase the
small 4.83 us/atom saving available from removing the R adapters.

Both experiments were removed. The accepted implementation remains the
Milestone 2 packed-Kokkos M path with generated spline R owners. Further M work
must preserve the feature-major packed gate/linear pipeline and the compiled
product; the next target is the compiled M0 reverse work inside that constraint,
not another whole-node generated or row-major layout branch.

The first compiled-product follow-up replaces the host reverse owner's
16-member switch accumulator with a fixed 16-value indexed local array. The
term loop and floating-point accumulation order are unchanged; the optimization
removes the dynamic switch for every derivative contribution and lets the host
compiler address spill slots directly. The existing scalar-member accumulator
remains in the non-host branch, so CUDA/HIP behavior and launch structure are
unchanged.

On the same one-core workload, the matched complete-call median improves from
1,817.581 us/atom (1,809.312--1,825.280) to 1,633.063 us/atom
(1,631.087--1,635.203), a 10.2% reduction. The instrumented compiled-product
reverse falls from about 678.5 to 560.3 ms/call, or 17.4%, and instrumented M0
reverse falls from about 809.5 to 675.5 us/atom. R timings remain unchanged.
The direct Float64 primitive comparison reports zero reverse error for both
products, while the full Float32 candidate differs from the generic spline by
7.01e-6 eV total energy, 1.33e-6 eV/A maximum force, and 1.08e-8 eV/A^3 maximum
stress.

The next retained host specialization admits only the exact Standard MACE
polynomial topology used by both official MH-1 products. It executes the
compile-time-unrolled Standard M0 polynomial routines with a 512-feature tile;
nonstandard products and CUDA/HIP keep the generic compiled-product owner. In
combination with the packed gate and linear pipeline, this reduced the
five-sample MH-1 median from about 963--971 us/atom to 658.856 us/atom.

Two follow-up experiments were rejected and removed. Pretransposing the fixed
linear weights changed the host GEMMs from transpose/non-transpose to
non-transpose/non-transpose, but aggregate GEMM time regressed from 194.20 to
196.65 ms/call and the complete call regressed by about 1.7%. A 16-node tiled
fusion of gate reverse and density accumulation increased gate reverse from
30.49 to 34.47 ms/layer and the complete call to 651--656 us/atom. Neither
experimental storage nor dispatch remains in the implementation.

The retained direct `linear_2` forward boundary lets the packed gate output feed
GEMM directly and writes the result into the product's retained angular-major
input. This removes `e3 linear unpack output` and `compiled product angular
input forward` without changing the product's node-local layout. The matched
complete-call result improved from 658.856 to 644.839 us/atom, or 2.13%.

The corresponding reverse boundary now has the admitted standard-product owner
write its input adjoint in the feature-major packed layout consumed directly by
`linear_2` reverse. `reverse_packed_to_packed` then invokes the existing dense
GEMMs without `e3 reverse angular-major pack output blocked`. Admission requires
the exact standard host product plan, packed node linears, complete input-block
coverage, and an identity active output mask. Generic products and CUDA/HIP
continue to emit and consume the native layout.

The exact qualification uses Float32, CPU 0, one Kokkos/OpenMP/BLAS thread,
864-atom periodic AlN, 78,624 directed edges, the 6.0 A model cutoff, and zero
skin. The final binary SHA-256 is
`6f6f7094bb39687b1d76f4f6489974f0c75c896c8e973263a82428cd4900eb42`.

| Metric | Angular-major control | Packed-adjoint boundary | Change |
|---|---:|---:|---:|
| Complete median | 650.541 us/atom | 640.969 us/atom | -1.47% |
| M1 reverse | 118.681 us/atom | 115.128 us/atom | -2.99% |
| M0 reverse | 128.102 us/atom | 125.737 us/atom | -1.85% |
| Combined `linear_2` reverse | 72.441 us/atom | 60.587 us/atom | -16.36% |
| Compiled product reverse | 30.886 ms/call | 34.660 ms/call | +12.22% |

The direct scattered product stores recover only part of the removed 11.08
ms/call transpose: product reverse itself costs 3.77 ms/call more. The retained
net improvement is nevertheless stable across two five-sample candidate runs
at 639.795 and 640.969 us/atom. On the perturbed 864-atom parity case, maximum
differences from generic spline are 1.35e-5 eV total energy, 1.26e-6 eV/A in
forces, and 1.07e-8 eV/A^3 in stress. A direct primitive comparison of native
and packed product-to-linear reverse reports zero error for both layers.

The matched MH-0 median is 213.409 us/atom, so the current MH-1 result is 3.00x
MH-0 and remains 214.151 us/atom above the 426.818 us/atom target. The latest
trace attributes about 195.10 ms/call, or 225.81 us/atom, to 71 dense GEMMs.
The remaining visible linear layout kernels total about 66.2 ms/call. The next
M-stage work should preserve BLAS-sized matrices while extending packed layout
across the product-output linear and the following layer boundary; replacing
dense model weights with sparse, shared, or low-rank forms would change the
model and is not an exact optimization.

A product-output forward prototype made the standard contraction write packed
feature-major values directly for its output linear. Primitive forward and
reverse comparisons were exact. It reduced profiled M0 forward from about
108.6 to 105.3 us/atom, while M1 forward was unchanged, but the five-sample
complete median was 640.997 us/atom versus the retained 640.969 us/atom. The
node-owned contraction's scattered packed stores consume the saved input-pack
time. The component-layout descriptor and dispatch were removed; a persistent
boundary is useful only if a downstream consumer can also remain packed.

### Exact-FP32 non-BLAS follow-up

After an exact-FP32 BLAS attribution showed that changing the BLAS provider
could improve the complete call by only 3.18%, the next experiment series held
OpenBLAS and all 56 SGEMMs fixed. The qualification used CPU 0, one
Kokkos/OpenMP/BLAS thread, 864-atom periodic AlN, 78,624 directed edges, the
6.0 A model cutoff, zero skin, and `pair_spline_v1_staged_host_v5`. The restored
control binary SHA-256 was
`b51fccc2fb016234f11efd412a2b83f88fafea803e2bb5104840e7cfdbb373ba`.

The first four small non-BLAS changes retained in this series are:

| Retained change | Complete median | Isolated evidence |
|---|---:|---|
| Restored control | 555.420 us/atom | Exact restored binary hash |
| Packed scalar gate feature-major traversal | 546.709 us/atom | Scalar gate forward plus reverse: 42.87 to 26.91 ms over 14 layer launches |
| Identity convolution-mask spline specialization | 542.873 us/atom | Spline forward: 282.98 to 236.53 ms over 14 launches; reverse: 655.10 to 648.15 ms |
| Eight-sample angular-output tile | 538.674 us/atom | Angular-major output: 114.61 to 97.52 ms over 56 launches |
| One-launch packed normalization forward | 536.398 us/atom | Forward normalization: 29.23 to 17.79 ms and 98 to 14 launches |

That checkpoint's installed extension SHA-256 was
`883f6edf09ebdecdb65fc761a9a00ea7a7a6e35ffb4b17fdcd4fb0641383d141`.
Its nine-sample median was 536.398 us/atom, a 3.42% complete-call improvement
from the restored control. Total energy is `-6423.653289366527 eV`; removing
the identity-mask multiply changes the total by 3.4e-6 eV, or about 4e-9
eV/atom, through compiler reassociation in FP32. No reduced precision or BLAS
change is present.

A fifth retained change aliases the gate's pre-gate and linear adjoints when
the spline path has already normalized the linear input. In that case the two
adjoints are mathematically identical and both downstream linears are
read-only, so gate reverse now emits one full packed adjoint store instead of
two. The tensor gate reverse kernel fell from 91.98 to 63.52 ms over seven
calls, and combined M0/M1 reverse fell by 4.46 ms/call, or 5.16 us/atom.
Two alternating seven-sample A/B pairs measured 547.809 to 543.413 us/atom
and 536.217 to 532.353 us/atom, a mean reduction of 4.13 us/atom (0.76%).
Energy remained exactly `-6423.653289366527 eV`. The retained extension
SHA-256 is
`f7ad5c0e856c96ab06a96cf61a8cf7087418d5d3761518ea487a3d2c19289463`.

The spline specialization is admitted only when every interaction's tensor
product mask is verified to contain exactly one in every entry. Non-identity
models keep the existing path. Packed reverse normalization remains blockwise:
fusing it reduced launch count but regressed its measured kernel time from
about 70.39 to 75.90 ms over seven calls.

The following prototypes were rejected and fully removed:

- Direct packed `linear_2` product input regressed to 567.81 us/atom because
  feature-vector loads became sample-strided. A sample-tiled packed product
  schedule regressed further to 1,182.78 us/atom.
- Product feature tiles 128 and 256 measured 575.37 and 580.43 us/atom;
  the retained standard-product tile remains 512.
- Combining spline value and derivative outputs increased live ranges in the
  large generated reverse owner and regressed to 553.85 us/atom.
- Replacing receiver-owned spline output accumulation with overwrite stores
  regressed to 546.81 us/atom.
- A shared 32-sample linear-layout tile slowed angular-major output from
  114.61 to 120.51 ms. An isolated four-sample angular-output tile measured
  542.28 us/atom; the retained local optimum is eight samples.
- A 16-sample blocked reverse-normalization traversal changed its seven-call
  kernel time only from 74.65 to 73.52 ms and did not improve the complete
  call. The original device-neutral loop remains.
- Transferring Standard MACE's 16-feature product tile to the MH-1 packed
  owner regressed the complete call to 597.464 us/atom and shifted total
  energy by about 2.0e-5 eV. The 512-wide compile-time tile remains because
  it traverses the generated term sequence once for the model's 512 active
  features.
- Aliasing the product skip adjoint to the layer adjoint removed its copy but
  regressed the complete median from 534.144 to 538.123 us/atom, consistent
  with worse cache residency across R reverse. The short-lived skip buffer is
  retained.

The compiled-product sparsity audit does not justify a new sparse product
owner. M1 has no exact-zero compiled coefficients among its 48,128 values. M0
has 9,618 exact zeros among 216,064 coefficients (4.45%), no all-zero term,
and no all-zero feature across its scalar and vector outputs. Twenty-two of
the 512 M0 features have an all-zero vector contraction, but they are scattered
and the remaining zeros vary by term. Exploiting them would require masked or
gathered feature traversal that gives up the current contiguous 512-feature
SIMD owner for a theoretical product-only saving below 4.5%. Product sparsity
is therefore not retained as an execution branch.

The next retained non-BLAS optimization factors spline value and derivative
weights out of each sparse Wigner reverse contraction. For each path and
channel, generated code first accumulates source, harmonic, and radial-weight
adjoints from the sparse terms, then applies the common spline value once per
unique source or harmonic component and the common derivative once per path.
For R1 this covers 86 sparse terms in ten paths, including two 21-term paths.
It preserves the source-owned schedule, buffers, FP32 precision, and launch
count; no alternative runtime code path is added.

The explicit-artifact stage comparison used CPU 0, one Kokkos/OpenMP/BLAS
thread, 864 atoms, 78,624 directed edges, the 6.0 A cutoff, and zero skin. R0
reverse fell from 22.10 to 20.75 ms/call (25.58 to 24.02 us/atom), R1 reverse
from 85.49 to 72.72 ms/call (98.95 to 84.17 us/atom), and the combined generated
spline reverse owner from 95.24 to 81.07 ms/call (110.23 to 93.83 us/atom).
The matched complete call improved from 539.01 to 518.91 us/atom, or 3.73%.
Total energy remained exactly `-6423.653289366527 eV`.

Three related experiments were rejected and removed. Returning spline value
and derivative through one explicit helper increased R1 reverse from 83.33 to
84.31 ms/call. Factoring spline weights in forward increased the combined
spline-forward kernel from 34.77 to 38.77 ms/call and reassociated the energy
by 4.87e-5 eV. Deriving the radial adjoint as a post-contraction dot product
extended vector live ranges and increased the reverse owner from 79.09 to
81.96 ms/call. A generated artifact compiled with a preferred 256-bit vector
width measured 540.60 us/atom, so the native 512-bit vectorization remains the
qualified policy on this CPU.

The next retained R optimization changes generated loop order without changing
the spline or sparse-contraction algebra. Each sparse path now owns its full
128-channel loop in both forward and reverse. Previously one channel iteration
covered every path, giving the R1 reverse function a roughly 6.4 KiB stack
frame and forcing many AVX-512 temporaries to spill. Path-owned channel loops
reduced that frame to about 3.8 KiB while retaining contiguous channel SIMD,
the same source-owned reverse schedule, the same buffers, and one physical
launch per layer and direction.

The explicit-artifact A/B used the same one-core 864-atom, 78,624-edge workload
as above. The path-factor baseline measured 519.835 and 517.631 us/atom in two
bracketing runs. Reverse loop inversion measured 500.066 us/atom. Adding the
same loop order to forward measured 492.957 us/atom; an instrumented run
measured 490.808 us/atom. In the region profile, R1 reverse fell from 74.275 to
55.021 ms/call (85.97 to 63.68 us/atom), the combined spline reverse owner
from 83.320 to 62.335 ms/call (96.44 to 72.15 us/atom), and the combined spline
forward owner from 33.667 to 26.207 ms/call (38.97 to 30.33 us/atom). Aggregate
SGEMM time remained effectively unchanged at 185.99 versus 185.14 ms/call.

Reverse loop inversion preserved the total energy exactly. Forward loop
inversion reassociated the symmetric 864-atom total by 2.32e-5 eV. On the
perturbed qualification structure, differences from the reverse-only artifact
were 1.11e-6 eV total energy, 1.06e-6 eV/A maximum force, and 7.74e-10 eV/A^3
maximum stress. The retained artifact is
`/tmp/symmetrix-mh1-spline-path-outer-forward-jit/artifacts/1aa17bb12dd11ac540d67c2adad30e38d16fa560c62927bdc52c3af41c4e7ac5/jit_mh1_host_plugin_gen2.so`
with SHA-256
`5e5f33d4106b47942879c767c1a62ff2605d3f663f45cfe0963b7a8c9fe99b27`.

Replacing positive-radius `floor` with direct integer truncation was neutral:
the candidate measured 491.297 us/atom between floor baselines of 489.414 and
492.138 us/atom. The branch-shape change offset the removed conversion, so the
canonical interval calculation remains unchanged.

The next investigation fixed the BLAS provider and refocused exclusively on
non-BLAS work. A five-invocation Kokkos Tools trace of the retained artifact
measured about 88.8 ms/call in generated spline R, 51.4 ms/call in the four
compiled product contractions, 38.0 ms/call in packed gate forward and reverse,
13.7 ms/call in angular output layout, and 11.2 ms/call in reverse
normalization. The R1 spline reverse owner remained the largest individual
non-BLAS kernel at about 42.9 ms/call. All path loops were confirmed to use
64-byte AVX-512 vectors, so further R1 work must reduce contraction arithmetic
or improve ownership-level reuse rather than launch count or SIMD width.

Four non-BLAS prototypes were rejected and removed:

- Admitting the complete generated V4 node program under `pair_spline_v1`
  measured 4,916.95 us/atom and produced a non-finite energy because its node
  state and layout contract is incompatible with the retained packed V5 spline
  boundary. The spline path continues to use the compiled M product and gate.
- Disabling loop unrolling produced a byte-identical artifact. Register-renaming
  policies changed instruction scheduling but increased isolated R1 reverse
  from 42.87 to 43.88 ms/call; whole-call changes stayed within about 0.3%.
- Explicitly naming and reusing identical sparse-path reverse factors increased
  R1 reverse from 42.87 to 43.27 ms/call. GCC already performs the profitable
  common-subexpression elimination across the unrolled terms.
- Reading the component-major product input directly removed a 32 KiB per-node
  stack copy and improved the four isolated product kernels by about
  0.72 ms/call in aggregate. It nevertheless regressed two seven-sample
  complete-call medians to 490.18--490.51 us/atom versus bracketing controls at
  487.17--488.58 us/atom, because the local copy provides stronger alias and
  cache behavior for the 512-channel contraction.

No experimental dispatch, compiler option, factor table, or direct-view product
abstraction from this screening remains in the implementation.

Two further exact non-BLAS screens were also rejected and removed. Although
the four `linear_2` instructions in each layer have the same 512-by-512 shape
and path scaling, structured comparison of the model tensors showed that all
four weight slices are distinct. They therefore cannot share one GEMM or one
output traversal without changing the model. Adding local restricted pointers
for the source vector, receiver adjoint, and edge harmonics in generated spline
reverse was timing-neutral: the candidate measured 489.64 us/atom between
bracketing controls at 488.84 and 493.08 us/atom. Hoisting the common spline
coefficient base once per edge was likewise neutral at 489.52 us/atom between
controls at 489.17 and 490.80 us/atom. GCC already retains the profitable
address and alias information in the path-major AVX-512 loops.

The next retained M-stage change commutes message normalization through the
bias-free `linear_1`. The packed spline message now enters `linear_1`
unnormalized, one inverse normalization factor is computed per node, and the
gate applies that factor to the linear result before adding the residual. The
reverse remains in the smaller message space: gate reverse aliases the
pre-gate and linear adjoints, `linear_1` reverse runs normally, and reverse
normalization applies the inverse while transposing into the R-stage layout.
Because the saved message is unnormalized, its density derivative uses the
mathematically required inverse-square factor.

Two alternating seven-to-nine-sample comparisons measured the retained
control at 489.195 and 488.626 us/atom and the candidate at 485.010 and
484.683 us/atom, a mean reduction of 4.06 us/atom (0.83%). The retained
extension SHA-256 is
`81de81a15aa018679f8c2ce5643e8756a7c697d11e72800866c79ea6aed58d79`.
On the perturbed 864-atom qualification structure, maximum candidate-versus-
control differences were 7.38e-6 eV total energy, 2.12e-6 eV per-atom energy,
1.60e-6 eV/A force, and 1.14e-9 eV/A^3 stress. The focused MH-1 contract suite
passed with 31 tests and one skip.

The corresponding five-invocation profile attributes about 89.9 ms/call
(104.1 us/atom) to generated spline R, 50.1 ms/call (58.0 us/atom) to the four
compiled products, 27.9 ms/call (32.3 us/atom) to packed gate forward and
reverse, 14.0 ms/call (16.2 us/atom) to angular output layout, and 9.0 ms/call
(10.4 us/atom) to reverse normalization. These are the primary non-BLAS
owners; optimized SGEMMs are held fixed in subsequent experiments.

Two follow-up normalization and product screens were rejected and removed.
Passing the saved inverse into each reverse-normalization block replaced scalar
divisions with buffer loads but measured 486.895 and 486.027 us/atom versus
adjacent controls at 485.734 and 485.459 us/atom. A structured monomial audit
found no duplicate among all 422 Standard-M0 terms or the 94-term M1 scalar
prefix. Caching the eight most frequent first-pair products reduced the
isolated combined product-forward time from 15.69 to 14.89 ms/call, but its
0.80 ms saving did not improve complete-call timing repeatably and shifted the
symmetric-system total energy by 1.02e-5 eV through changed FP32 code
generation. No inverse-reuse or pair-cache execution branch remains.

The next retained M-stage change removes the angular-major layout boundary
between packed `linear_2` and the compiled product. The four block GEMMs now
write `[block][node][component][channel]` storage directly. Product forward
reads contiguous channel rows from that storage, while product reverse writes
`[block][channel][node][component]` adjoints directly for `linear_2` reverse.
The old angular-major forward and reverse transposes, about 7.18 and 7.12
ms/call, are gone; the same four SGEMMs and model algebra remain.

Two alternating nine-sample comparisons measured the previous boundary at
482.998 and 485.086 us/atom and the block-major boundary at 472.588 and 472.670
us/atom, a mean reduction of 11.41 us/atom (2.36%). On the perturbed 864-atom
structure, energy and per-atom energies were bitwise identical; maximum force
and stress differences were 4.12e-7 eV/A and 8.81e-10 eV/A^3. Reducing the
compiled product's 512-feature tile to 256 was rejected at 487.74--489.58
us/atom versus 469.45--470.90 us/atom for 512. The repeated traversal of 422 M0
terms outweighed the smaller local arrays, so no tile-size dispatch remains.

The retained R reverse then changes data flow with a bounded source-major edge
tile. Each owner stages edge and receiver identifiers, spline interval
metadata, polynomial coordinates, radial adjoints, and harmonic adjoints. It
executes one generated path across the staged edges before advancing to the
next path, while preserving the contiguous 128-channel AVX-512 loop as the
innermost loop. This changes `edge -> path -> channel` into
`edge tile -> path -> edge -> channel`, improves instruction locality across
the ten R1 path bodies, retains deterministic source ownership, and allocates
no graph-sized table.

Tile sizes 4, 8, 16, 32, 64, and 128 measured 461.45, 456.24--459.82, 457.77,
457.33, 452.31--457.10, and 452.53--453.48 us/atom in the screening and
bracketing runs. Tile 128 is retained because this 78,624-edge graph averages
91 edges per source, so one bounded tile normally covers a complete source.
The final cleaned explicit-artifact A/B measured path-major reverse at 470.159
and 471.977 us/atom and tiled reverse at 453.871 and 454.285 us/atom, a mean
16.99 us/atom (3.61%) improvement. The stage trace attributes the gain to R1
reverse falling from 43.07 to 34.32 ms/call and R0 reverse from 19.04 to 12.74
ms/call; compiled product, gate, normalization, and SGEMM stages stayed flat.

Applying the same path-over-edge-tile order to receiver-owned forward was
rejected and removed. It measured 511.29 us/atom versus about 454 us/atom for
reverse-only tiling and shifted total energy by 1.32e-4 eV. Forward needs to
accumulate every path while each edge's source and harmonic rows are hot;
source-owned reverse instead benefits from retaining the source adjoint while
one large generated path body advances across the edge tile.

The next one-core work excluded BLAS and screened the remaining compiled
product, normalization, and gate loops. A separate compile-time M1 product tile
was rejected and removed: the templated dispatcher itself measured 467.12
us/atom at the unchanged 512 tile versus 453.51 us/atom for the retained loop,
while M1 tiles 64 and 256 measured 470.68 and 472.74 us/atom. Explicitly
initializing only active product output rows was also rejected at 454.31 versus
453.22 us/atom. Finally, fusing reverse normalization across irreducible blocks
left the complete call at 453.51 us/atom. M1 normalization changed from 6.718 to
6.767 ms/call and outweighed M0's small 2.555 to 2.340 ms/call improvement, so
that path was removed as well.

The retained non-BLAS change specializes packed tensor gating on host execution.
Each feature now owns an inner node loop, allowing its seven plan fields and gate
constant to remain live instead of being loaded for every feature-node pair.
The feature-major order and per-node component order are unchanged, and the
CUDA/HIP MDRange kernel is untouched. Combined M0/M1 tensor gate forward and
reverse time fell from about 26.97 to 25.75 ms/call, a 1.22 ms/call or 1.41
us/atom stage reduction. Two 20-sample candidate/control pairs measured
455.167/456.501 and 450.874/453.352 us/atom, improving by 1.33 and 2.48 us/atom.
The mean candidate is 453.02 us/atom, 2.17x the matched 208.465 us/atom MH-0
result and 36.09 us/atom above the strict 416.930 us/atom target.
Perturbed energy and per-atom energies were bitwise identical; maximum force and
stress differences were 4.50e-7 eV/A and 9.74e-10 eV/A^3. The focused MH-1
contract suite again passed 31 tests with one skip. The retained extension
SHA-256 is
`659b905fa389d9c77cb5d7eaf94b2a1a258e36904eb998e759a3695115069872`.

The retained implementation exposes one internal product input-layout enum,
with native and block-major values, rather than accumulating experimental
booleans. Angular-major scratch is now lazy and exclusive with the direct
block-major path. Product workspace falls from 244,187,136 to 130,940,928
bytes, a 108.0 MiB reduction, and total precision workspace falls from
284,586,536 to 171,340,328 bytes. The pre-gate cleaned extension SHA-256 is
`e4df3f46d9783a1c21b03b32e2dfb313f1362cbfdb9d123bb9889c80c6560d34`.
The retained tile-128 artifact is
`/tmp/symmetrix-mh1-spline-edge-tile128-jit/artifacts/b4aea96260144b07b8cbd1c448debed614d4729b1eaf2105f2dc84dd5e3fd237/jit_mh1_host_plugin_gen2.so`
with SHA-256
`000decd40e1e45ef296749eb8d9d640ff25fd60fbcd0f9a9e14e99bb01d12d18`.
Perturbed path-major versus tile-128 differences were zero total energy,
6.53e-7 eV/A maximum force, and 5.81e-10 eV/A^3 maximum stress. The focused
MH-1 contract suite passed with 31 tests and one skip.

### Float64 staged-R qualification

Host ABI v5 now types the spline R packets independently from the ABI-v4 node
owners. Float32 retains packed spline messages and loads the complete generated
V4+V5 path. The first Float64 milestone advertised an 8-byte spline scalar,
emitted receiver-major messages, loaded only the generated V5 spline owners,
and used the generic Kokkos M stages. The selected backend was reported as
`pair_spline_v1_staged_r_host_v5`; accelerator MH-1 remains Float32-only.

The one-core qualification used CPU 0, one Kokkos/OpenMP/BLAS thread, 864 atoms,
78,624 directed edges, a 6.0 A cutoff, zero skin, and seven measured calls. The
median fell from the previous generic Float64 result of 12,745.465 us/atom to
1,760.735 us/atom, a 7.24x speedup. Total energy remained exactly
`-6423.653321861149 eV`. The extension SHA-256 was
`79ad3c55422d71e28060086ba256ce65f112a96d48f2f9c66511af042411e07c` and
the generated artifact was
`/tmp/symmetrix-mh1-fp64-v5-final-cache/artifacts/ee230ca584911b59c33d89ce89c17bd85a5a9b4553b948b8c73bb27bed3a17c4/jit_mh1_host_plugin_gen2.so`.

The Kokkos region timer adds about 2.9% overhead, so its stage values are used
for attribution rather than the steady-state total:

| Stage | Float64 us/atom | Share of named stages |
| --- | ---: | ---: |
| R0 forward | 20.30 | 1.1% |
| M0 forward | 315.72 | 17.2% |
| R1 forward | 74.61 | 4.1% |
| M1 forward | 336.46 | 18.3% |
| M1 reverse | 475.60 | 25.9% |
| R1 reverse | 121.74 | 6.6% |
| M0 reverse | 454.04 | 24.7% |
| R0 reverse | 38.22 | 2.1% |

Combined R is 254.88 us/atom, or 13.9% of named stages. Combined M is
1,581.82 us/atom, or 86.1%. The nested BLAS regions account for about 515.51
us/atom, approximately 28.9% of profiled wall time. E3 linear forward and
reverse pack/unpack kernels account for about 701.9 us/atom before considering
other non-BLAS gate, product, mask, and normalization work. Further Float64
work should therefore retain the spline R specialization and focus on removing
M-stage layout traversals rather than changing the BLAS kernels.

### Float64 packed-M qualification

The next experiment admits host Float64 to the same packed Kokkos M pipeline
used by Float32. Generated V5 spline forward now writes packed messages for
both host precisions, and the V5 loader requires that layout. The existing
templated packed gate, E3-linear, normalization, and product operations then
keep node tensors packed through M0 and M1. Device Float64 remains excluded;
the accelerator admission policy is unchanged.

The matched one-core workload used CPU 0, one Kokkos/OpenMP/BLAS thread,
864 atoms, 78,624 directed edges, a 6.0 A model cutoff, zero skin, and a 6.0 A
effective cutoff. The seven-sample median fell from 1,760.735 to
968.487 us/atom, a 45.00% reduction. A separate 15-sample run measured
972.188 us/atom. Total energy remained exactly `-6423.653321861149 eV`.
Against a compiler-free generic-M control on the same graph, maximum packed-M
differences were 5.33e-15 eV per-atom energy, 4.00e-15 eV/A force, and
1.28e-16 eV/A^3 stress; total energy was bitwise identical.

The final extension SHA-256 is
`b8d9a3ebdd47ce22ec5ae395f79c0dd7cd3d4b9e6aa4ccf1081835ec1efca02e`.
The Float64 generated artifact is
`/tmp/symmetrix-mh1-fp64-packed-jit-cache/artifacts/f685911fe88bd943ad9fc5d080457c6d79239a7e61fa6bb77b9f37fe63d77cb7/jit_mh1_host_plugin_gen2.so`.

The retained Kokkos profile reports:

| Stage | Previous Float64 us/atom | Packed-M Float64 us/atom |
| --- | ---: | ---: |
| R0 forward | 20.30 | 17.88 |
| M0 forward | 315.72 | 188.33 |
| R1 forward | 74.61 | 66.34 |
| M1 forward | 336.46 | 177.15 |
| M1 reverse | 475.60 | 183.07 |
| R1 reverse | 121.74 | 116.19 |
| M0 reverse | 454.04 | 190.39 |
| R0 reverse | 38.22 | 38.62 |

Combined M falls from 1,581.82 to 738.95 us/atom, a 53.3% reduction, while
combined R remains approximately flat at 254.88 versus 239.03 us/atom. Generic
E3 pack/unpack work falls from about 701.9 to 56.5 us/atom and from 140 to 47
launches per inference. DGEMM count falls from 70 to 62 and nested DGEMM time
falls from 515.51 to 386.32 us/atom. This confirms that the previous 3.889x
Float64/Float32 ratio came from losing the packed M path, not spline R or an
abnormal DGEMM implementation. The current matched medians are 968.487 and
454.293 us/atom, a 2.132x Float64/Float32 ratio, compared with MH-0's 2.225x.

### Rejected MH-0 R1 source-edge fusion

Two host source-owned prototypes tested whether MH-0 R1 source reverse and
directed edge-force differentiation could share radial evaluation, harmonic
loads, receiver adjoints, and edge metadata. The retained separate owners cost
about 90.413 us/atom together on the matched 864-atom graph.

The first order, `source -> edge tile 128 -> path -> edge -> channel`, retained
a full 4x128 compensated source adjoint plus a 128-edge force tile. Its fused
owner cost 104.278 us/atom and the complete call regressed from about 214 to
229--230 us/atom. The second order,
`source -> channel tile 16 -> edge -> path -> channel`, reduced live source
state to 4x16 but traversed each source edge list once for every channel tile.
It produced the expected energy but measured 340.227 us/atom versus the cleaned
separate-owner result of 218.523 us/atom.

The experiment rejects source-edge fusion for this host workload: keeping all
channels live creates excessive stack/register/cache pressure, while channel
tiling destroys the intended edge-reuse benefit. The fused ABI capability,
runtime launch, and both generated owners were removed. MH-0 retains the
16-channel compensated source owner and the independent edge owner.

For every accepted milestone, record total and stage `us/atom`, sample ranges,
atom and edge counts, cutoff and skin, thread affinity, Kokkos/BLAS thread counts,
selected artifact, fallback count, and peak memory. Compilation, graph
construction, and first module loading remain outside steady-state timing.

## Initial status

At the start of this plan, the implementation had ordered-pair spline storage
and validated analytic derivatives, but its driver and generated ABI did not
yet satisfy the staged direct contracts above. Milestone 0 and the receiver-
owned portion of Milestone 1 are now implemented as recorded above.
