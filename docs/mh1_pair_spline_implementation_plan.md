# MH-1 ordered-pair spline implementation plan

## Objective

Add a new CPU/OpenMP MH-1 execution path that has the same R1/M1 stage
boundaries as standard MACE `streamed_edges="direct"`, without evaluating a
radial MLP for every edge. Keep the existing generated MLP path as a numerical
reference until the spline path passes equivalence and performance gates.

The paper contract is equations 3--19 of `2510.25380v1.pdf`, pages 3--4. The
spline boundary is limited to functions determined by interaction, distance,
and the ordered source/target element pair. Spherical harmonics, tensor-product
contractions with evolving node features, density reduction, normalization,
gating, products, updates, and readouts remain dynamic.

## Executor selection

Use one temporary internal selector:

- `mlp_reference`: current conditioner/density MLP implementation.
- `pair_spline_v1`: ordered-pair cubic spline implementation.

Do not add public `streamed_edges` modes for these implementation details.
During development, `streamed_edges="direct"` keeps its current default and the
selector is exposed only for tests and controlled benchmarks. After numerical
and performance qualification, `pair_spline_v1` can become the CPU/OpenMP
default while `mlp_reference` remains available as a validation oracle.

## Spline contract

For interaction `s`, ordered pair `(source_type, target_type)`, and distance
`r`, tabulate:

```text
C_s,p,c(r, source_type, target_type)
    = f_cut(r) * complete tensor-product radial coefficient

D_s(r, source_type, target_type)
    = f_cut(r) * tanh(density_mlp_s(r, source_type, target_type)^2)
```

`C` includes the final affine bias and any fixed ordered-pair contribution.
The serialized `apply_cutoff` flag determines whether the radial basis entering
the MLP already contains the cutoff. If it does not, the external cutoff is
folded into each final table exactly once. Pair indexing is ordered because
source and target element embeddings are distinct. No symmetric pair
compression is permitted.

All radial, channel, interaction, and element dimensions come from the
extracted model. Paper defaults such as eight radial basis functions and 89
supported elements are not runtime constants. Tables cover only the selected
model elements loaded by the evaluator, avoiding a full 89-by-89 allocation.

Each cubic Hermite interval stores coefficients for value evaluation. Its
analytic derivative with respect to physical distance replaces the radial MLP,
radial-basis, and cutoff adjoints during reverse execution. Flattened table
indices and allocation extents use `std::size_t` or explicit 64-bit arithmetic.

## Execution stages

```text
geometry and spherical harmonics
  -> R0 spline/message forward and receiver density reduction
  -> M0 normalization, residual, gate, product, update, readout
  -> R1 spline/message forward and receiver density reduction
  -> M1 normalization, residual, gate, product, update, readout
  -> M1 reverse
  -> R1 source/edge reverse using C' and D'
  -> M0 reverse
  -> R0 source/edge reverse using C' and D'
  -> force/stress assembly and ZBL
```

The M1 reverse produces both message and density adjoints. R1 reverse combines
the message-weight adjoint with `C'` and the receiver-density adjoint with `D'`
before the existing distance-to-coordinate force assembly.

## Milestones

### 1. Storage and construction

- Add an MH-1 ordered-pair spline owner separate from standard MACE's symmetric
  radial tables.
- Reuse the established cubic Hermite coefficient layout and inline evaluator.
- Generate samples on the host from the same Bessel/Agnesi transform,
  conditioned MLP parameters, source/target embeddings, and cutoff used by the
  reference path.
- Validate dimensions for every interaction and reject unsupported radial or
  conditioner layouts instead of silently falling back inside an evaluation.

### 2. CPU/OpenMP forward

- Evaluate `C` and `D` directly from distance and ordered pair.
- Feed `C` to the existing edge tensor-product contraction.
- Sum `D` by receiver and leave learned `alpha + beta * density`
  normalization in M1.
- Remove radial basis, conditioner MLP, density MLP, and `edge_phi` work only
  from the selected spline path.

### 3. CPU/OpenMP reverse

- Reuse or recompute the spline interval for each edge.
- Contract TP weight adjoints with `C'`.
- Add `density_adjoint[target] * D'`.
- Pass the resulting scalar radial derivative to the existing coordinate and
  stress assembly.
- Preserve source-feature adjoints, harmonic adjoints, and all M1 derivatives.

### 4. Qualification

Focused tests must cover:

- ordered pairs `(a,b)` and `(b,a)` producing independent tables;
- table values and analytic derivatives against the MLP oracle at nodes,
  interval interiors, short distances, and immediately below/at/above cutoff;
- energy, force, stress, and per-layer density equivalence for FP32 and FP64
  where supported;
- finite-difference force checks that exercise both `C'` and `D'`;
- selector reporting and unsupported-model rejection;
- absence of per-edge conditioner/density MLP calls in `pair_spline_v1`.

Primary performance qualification uses one pinned physical CPU core, one
Kokkos/OpenMP thread, and one BLAS thread. Report `us/atom`, atom count, model
cutoff, skin/effective cutoff, and directed-edge count. Record R0, M0, R1, M1,
reverse, and force-assembly timings for both executors.

## Acceptance gates

- No change to `mlp_reference` results or launch accounting.
- Errors remain within the existing evaluator tolerances, with no force spike
  near the cutoff.
- `pair_spline_v1` executes no per-edge radial conditioner or density MLP.
- Peak memory remains bounded by the documented ordered-pair table size.
- One-core `us/atom` improves measurably before the path becomes the host
  default.
- HIP and CUDA integration is deferred until the CPU/OpenMP path is accepted;
  accelerator builds must use their own fresh build directories and tests.

## Implementation status (2026-08-23)

Milestones 1--3 are implemented behind the internal `mh1_edge_executor`
selector. `pair_spline_v1` is the default for qualified models;
`mlp_reference` is retained only as an explicit validation oracle. The spline path owns complete TP
weights, density contributions, and their distance derivatives; it skips the
radial basis, cutoff, conditioner MLP, and density MLP during steady-state R1.
It retains Kokkos M1 and tensor-product execution on CPU/OpenMP.

The implementation was validated against the extracted Al/N `omat_pbe` model
(`sha256:fb1dc908fd0f7b99aa84279dd5ae598dbde62cb15b95c2b076581872b05d6917`)
using the staged OpenMP extension
(`sha256:ed23fe82be33f492a2352107bfb617cbb90d91001f5c0d00c735ba6f7c1b159e`).
For a mixed four-atom graph in FP32, `pair_spline_v1` differed from direct
`mlp_reference` by `8.30e-7 eV` in energy, `3.39e-5 eV/A` in maximum force,
and `6.78e-8` in maximum stress. The conditioned MLP directional derivative
also passed a central finite-difference test through Linear, LayerNorm, and
SiLU layers.

Primary timing used one physical core (CPU 0, excluding SMT sibling 16), one
Kokkos/OpenMP thread, and one OpenBLAS thread on an AMD Ryzen AI MAX+ 395.
The workload was 864-atom periodic wurtzite AlN, 78,624 directed edges, FP32,
a 6.0 A model cutoff, zero skin, and a 6.0 A effective cutoff. Seven reference
and five final spline samples gave:

| Executor | Median (`us/atom`) | Relative |
|---|---:|---:|
| `mlp_reference` | 12,559.0 | baseline |
| `pair_spline_v1` | 10,821.9 | 13.8% faster |

This qualifies the CPU/OpenMP Kokkos milestone but not promotion to the direct
default. A tested adapter that retained generated M1 while using Kokkos spline
R1 was numerically correct but regressed to 14,533.9 `us/atom`; it was removed.
Generated direct remains substantially faster because its R1 tensor products
are generated too. The next performance milestone is therefore a
spline-aware generated host R1 kernel that consumes complete spline weights
without restoring a per-edge MLP or adding persistent layout conversions.
