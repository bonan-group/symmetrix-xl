# MH-0 factorized GPU capacity

```{warning}
Historical record (current as of its original qualification context). It is
not the current public execution-policy reference. Use
{doc}`streamed_edge_execution` for the current `direct` plus `capacity`
default.
```

## Scope

The standard MH-0 direct path exposes automatic speed-versus-capacity
selection as `Symmetrix(..., streamed_edges="direct", low_memory=True)`.
At the time of this record, `low_memory=False` activated no capacity policy and
was the public default.
The selected capacity bundle does not stream state from host memory.

The capacity bundle applies to float32 and float64 ordinary MACE and MACEField
inference with Kokkos factorized execution, a generated R1 specialization, M1
polynomial recomputation, selected state-free M0 and R0 implementations, no
factorized observer, and no parameter gradients. Common contracts use built-in
M0/R0; compatible unmatched CUDA/HIP contracts use independently cached RTC
modules.
Both precisions select
`execution_mh0_state_policy="reuse-adjoints-v1"`. Float32 additionally selects
compact float32 unit-direction plus float64-radius geometry. Float64 keeps
Cartesian float64 geometry to preserve high-accuracy evaluation; its capacity
gain includes direct retained-Y harmonic storage in addition to polynomial
recomputation and state reuse. Ineligible harmonic configurations fall back to
retained gradients while the rest of the low-memory bundle remains active. The
individual policy setters remain expert qualification and rollback controls.

Global flattened addresses are 64-bit in generated R1, RTC M0/R0, standard
R0, and the maintained SpheriCart CUDA patch. This is required even while the
graph's node and edge identifiers remain 32-bit: an edge count well below
`INT32_MAX` can produce harmonic-gradient or node-feature arrays with more
than `INT32_MAX` elements. The remaining per-process graph-cardinality limit is
2,147,483,647 directed edges. See
`benchmarks/extreme_scale_indexing_20260829.md` for the million-atom A100
qualification and the capacity-driver failure contract.

MACEField energy, forces, stress, and polarization use the same primal
lifetimes. Polarizability and BEC reconstruct overwritten A0, A1, M1, and H2
state in the existing aliased allocations immediately before analytical
response. Dropped A1 density-scale values and derivatives are evaluated from
device-resident radii rather than retained as edge arrays.

## Reused lifetimes

The reverse schedule overwrites forward tensors only after their final read:

| Forward allocation | Reused as | Bytes at 864 atoms |
| --- | --- | ---: |
| `M1` | `M1_adj` | included below |
| `A1` | `A1_adj` | included below |
| `M0` | `M0_adj` | included below |
| `A0` | `A0_adj` | included below |
| `H2` | `H2_adj` | included below |
| Total aliased state | | `17,252,352 B` |

The A1 and A0 density scaling reverses normally need the dot products
`A1 * A1_adj` and `A0 * A0_adj` after those forward tensors are overwritten.
Their generated reverse kernels therefore capture one float64 scalar per atom
and layer before the overwrite. At 864 atoms these two arrays use `13,824 B`.
The retained nonlinear readout copies a complete H2 row into its internal MLP
workspace before writing the H2 gradient, so H2 and H2 adjoints can alias
without extra state. The predicted net saving from adjoint reuse alone is
consequently `17,238,528 B`, or `19,952 B/atom` (`16.44 MiB`). The complete
low-memory bundle additionally selects readout recomputation, described below.

H1 is not eligible for the same substitution. Factorized edge reverse still
reads H1 while accumulating H1 adjoints across multiple edges, so an alias
would overwrite values needed by later edges. Reducing H1 requires
recomputation or a different generated kernel and is not part of this
zero-recompute lifetime policy.

## HIP qualification

The qualification used the current MH-0 Al/N checkpoint and the hipRTC R1
artifact on a Radeon 8060S (`gfx1151`):

- periodic wurtzite AlN, 864 atoms;
- exact model cutoff `6.0 A`, neighbor-list skin `0`, effective cutoff `6.0 A`;
- 78,624 directed edges;
- float32 factorized execution, five warmups and ten measured evaluations;
- energy, atomic energies, forces, and stress evaluated each step.

Fresh-process medians were `33.117 us/atom` for full retention and
`32.781 us/atom` for adjoint reuse. The opt-in policy was `1.02%` faster in
this pair, so the capacity improvement has no measured performance penalty.

A rocprofv3 Agent-1 allocation trace measured `231,427,072 B` at full
retention and `214,187,776 B` with reuse. The measured reduction is
`17,239,296 B`, only `768 B` above the tensor-derived prediction.

Switching both policies on one evaluator produced zero total-energy and atomic
energy differences. The maximum force difference was
`2.3936e-9 eV/A`; the maximum stress difference was
`3.1345e-11 eV/A^3`.

The benchmark exposes the policy as:

```bash
python benchmarks/standard_mace_streamed_benchmark.py MODEL.json \
  --backend kokkos --dtype float32 --modes factorized --sizes 6 \
  --neighbor-skin 0 --m1-polynomial-policy recompute \
  --mh0-state-policy reuse-adjoints-v1
```

The JSON report records the requested and selected policies, any fallback
reason, aliased bytes, and auxiliary bytes under `mh0_state`.

## Additional device-state experiments

The same opt-in policy was used to qualify three further reductions. These
keep all state on the device and use the same C++ and generated code paths for
HIP and CUDA.

The A1 density-scaling spline values and derivatives can be evaluated while
scaling and recomputed during reverse. On the 864-atom graph this changed the
median from `32.781 us/atom` to `33.072 us/atom`, a `0.89%` cost. A 97,556-atom
rocprofv3 trace measured an exact `142,042,048 B` (`0.132287 GiB`) reduction,
matching two float64 values per directed edge. Full-retention parity had zero
total- and atomic-energy differences, a `1.54e-9 eV/A` maximum force
difference, and a `5.47e-11 eV/A^3` maximum stress difference.

Sphericart initially writes gradients in shuffled coordinate order. The
low-memory path now aliases that raw view with the normalized gradient view
and applies the three-component cyclic permutation in place. At 864 atoms the
geometry workspace fell from `38,683,008 B` to `23,587,200 B`. The median was
`32.439 us/atom`, compared with `32.764 us/atom` for a freshly rebuilt
full-retention control. At 97,556 atoms the traced peak fell by
`1,704,498,688 B` (`1.587438 GiB`), exactly one
`3 * directed_edges * 40 * float32` buffer.

Finally, generated R1 forward finishes reading `Phi1` before reverse creates
`dPhi1`. Reusing the same allocation for those non-overlapping lifetimes saves
one `nodes * 40 * 128 * float32` tensor without changing the arithmetic or
generated ABI. This reduced the 97,556-atom peak by another `1,997,947,136 B`
(`1.860733 GiB`). The 864-atom median was `32.239 us/atom`; full-retention
parity had zero total- and atomic-energy differences, a `2.39e-9 eV/A`
maximum force difference, and a `3.13e-11 eV/A^3` maximum stress difference.

The combined 97,556-atom workload has 8,877,596 directed edges, an exact model
cutoff of `6.0 A`, no neighbor-list skin, and an effective cutoff of `6.0 A`.
Its Agent-1 peak fell from `11,483,315,400 B` (`10.694671 GiB`) to
`7,638,827,528 B` (`7.114213 GiB`), a total saving of `3,844,487,872 B`
(`3.580458 GiB`). The measured single-step timing was `38.973 us/atom`, versus
`38.808 us/atom` before gradient compaction and Phi1 lifetime reuse.

### Nonlinear readout recomputation

For the qualified `128 -> 16 -> 1` readout, retained ordinary inference owns
two float64 tapes with `128 + 16 + 1 = 145` values each and one float64 output:

| Persistent allocation | Bytes/atom |
| --- | ---: |
| Forward values | `145 * 8 = 1,160` |
| Reverse derivatives | `145 * 8 = 1,160` |
| Readout output | `8` |
| Total | `2,328` |

The `recompute` policy replaces all three allocations with `16` float64 values
(`128 B`) of team scratch. Each receiver computes its 16 hidden
preactivations, transforms those scratch slots into `w2 * SiLU'(z)`, and uses
them for both the output and the 128-channel input gradient. MACEField's
directional derivative uses `256 B/team` to hold the hidden primal and tangent.
The retained MACEField path can grow to four tapes plus the ordinary output,
or `4,648 B/atom`; the recomputed scratch remains independent of atom count.

On the Radeon 8060S (`gfx1151`), three fresh-process pairs used the complete
float32 low-memory bundle, the hipRTC direct artifact, 864 atoms, 78,624
directed edges, exact cutoff `6.0 A`, skin `0`, 12 warmups, and 20 samples.
The median of the process medians was `18.908 us/atom` retained and
`18.702 us/atom` recomputed, so recomputation was `1.09%` faster in this
measurement. Energy was identical. The optimized implementation evaluates
SiLU and its derivative once per hidden unit; an earlier screening kernel that
repeated this work for every input channel was `1.69%` slower and was rejected.

At 32,000 atoms and 2,912,000 directed edges, a paired rocprofv3 allocation
trace measured `2,565,672,128 B` retained and `2,491,175,360 B` recomputed.
The `74,496,768 B` reduction is `32,000 * 2,328 + 768 B`; the final `768 B` is
allocator metadata. This is a `2.90%` reduction of the retained-readout peak.
The capacity bundle selected by `low_memory=True` uses recomputation, while
`_set_readout_policy` keeps `retained` available as an independent rollback
control.

### Receiver-local `Phi1` experiment

The relevant names in the second interaction are:

| Name | Meaning and ownership |
| --- | --- |
| `i`, `j`, `e=(i,j)` | Receiver atom, source/neighbor atom, and directed edge. |
| `k` | Feature channel; the qualified model has 128 channels. |
| `l,m` | Angular-momentum degree and component. For `l_max=3`, `Y` has `sum_l(2l+1)=16` components per edge. |
| `eta` | Multiplicity/path index: different valid tensor-product paths with the same output `l,m`. |
| `Y[e,lm]` | Real spherical harmonic of the edge direction. It carries angular geometry, not learned weights. In float32 it costs `16 * 4 = 64 B/edge`; `Y_grad` costs another `3 * 16 * 4 = 192 B/edge`. |
| `R1[e,path,k]` | Learned radial weight for the second interaction, evaluated from edge distance and element pair. Direct RTC computes it rather than retaining the full edge tensor. |
| `H1[j,lm,k]` | First-interaction equivariant feature on source atom `j`; reverse still needs it to form source-feature and coordinate adjoints. |
| `Phi1r[i,path,k]` | Uncoupled receiver sum of `R1 * Y * H1`. The generic path materializes it, but direct RTC already removes it. |
| `Phi1[i,lm,eta,k]` | Clebsch-Gordan-coupled receiver message before the dense A1 channel projection. It is an intermediate, not a model parameter. |
| `A1[i,lm,k]` | Projected second-interaction atomic feature consumed by the M1 product basis. Its dense projection mixes every input channel and multiplicity for a fixed `l`. |
| `dPhi1`, `A1_adj` | Reverse-mode derivatives of the energy with respect to `Phi1` and `A1`. Current low-memory execution aliases `Phi1` and `dPhi1` because their lifetimes do not overlap. |

Ignoring index-layout details, the forward dependency is:

```text
Phi1r[i,path,k] = sum_(e=(i,j)) R1[e,path,k] * Y[e,lm1] * H1[j,lm2,k]
Phi1[i,lm,eta,k] = sum_path CG[path -> lm,eta] * Phi1r[i,path,k]
A1[i,lm,k_out] = sum_(eta,k_in) Phi1[i,lm,eta,k_in]
                                   * W_A1[l,eta,k_in,k_out]
```

For this model, `Phi1` is `[N,40,128]` float32, or `20,480 B/atom`. The
32,000-atom trace contains the predicted `655,360,000 B` payload (the device
allocation is `655,360,256 B` including allocator overhead). Eliminating it is
therefore the next large node-scaled target, but it is not an allocation-only
lifetime change.

The implemented GPU experiment gives one receiver to a 128-thread block.
Forward holds one receiver's `40 * 128 * 4 = 20 KiB` `Phi1` row in LDS,
synchronizes, applies the per-`l` A1 projection, and writes only `A1`. Reverse
first computes `W_A1^T * A1_adj` into the same receiver-local LDS, then feeds
that `dPhi1` row directly to the generated R1 source and coordinate consumers.
This avoids global `Phi1` without repeating the 128-channel neighbor
contraction for every A1 output channel.

The RTC modules export separate projected-forward and projected-reverse entry
points with module-only packets carrying the A1 weights and output. The shared
plugin ABI and retained kernels remain unchanged. `_set_phi1_policy`
selects `retained` or `receiver-local`; the public low-memory bundle continues
to select `retained`.

The experiment was qualified on the Radeon 8060S (`gfx1151`) with the same
float32 direct low-memory contract as the readout result: 864 atoms, 78,624
directed edges, exact cutoff `6.0 A`, skin `0`, 12 warmups, and 20 samples.
Two fresh-process pairs measured:

| Policy | Process median 1 | Process median 2 | Phi1 workspace |
| --- | ---: | ---: | ---: |
| `retained` | `18.643 us/atom` | `18.712 us/atom` | `17,694,720 B` |
| `receiver-local` | `64.621 us/atom` | `65.078 us/atom` | `0 B` |

The means of the process medians are `18.677 us/atom` and `64.849 us/atom`:
receiver-local is `3.47x` slower. It is therefore a capacity experiment, not a
candidate for automatic low-memory selection.

Full-array parity used both the regular 864-atom cell and a deterministically
perturbed 32-atom cell. Total and atomic energies were identical. Maximum
force differences were `3.19e-6` and `7.81e-6 eV/A`; maximum stress differences
were `3.01e-7` and `6.60e-8 eV/A^3`. Both generated launches executed with zero
fallbacks.

At 32,000 atoms and 2,912,000 directed edges, paired rocprofv3 allocation
traces reduced the Agent-1 peak from `2,491,830,976 B` to `1,836,470,720 B`.
The exact `655,360,256 B` reduction is the predicted
`32,000 * 20,480 B` payload plus `256 B` allocator overhead, or `26.30%` of the
retained peak.

The 864-atom kernel trace explains the rejection. Projected forward averaged
`11.66 ms`, compared with about `0.65 ms` for retained generated R1 plus
`1.46 ms` for retained A1. Projected reverse averaged `33.62 ms`, compared with
about `2.80 ms` for retained generated reverse plus `1.26 ms` for retained A1
reverse. The one-receiver grid exposes only 864 blocks, the dense projection
loops serially over 128 input or output channels within each thread, and reverse
atomically accumulates source adjoints. A viable successor needs a tiled matrix
projection with more parallel ownership and a source-owned or two-stage reverse
reduction while preserving the 20 KiB receiver-local intermediate.

### 64-channel `Phi1` experiment

The successor keeps the existing generated R1 and batched A1 ownership but
executes the 128 channels in two 64-channel phases. The only persistent Phi1
workspace is `[N,40,64]` float32, or `10,240 B/atom`. Forward writes one compact
tile and immediately accumulates its contribution into A1 with phase-packed
weights. Reverse reconstructs the same compact tile from `A1_adj`, then runs a
source-owned generated reverse over that channel range. The two H1-adjoint
ranges are disjoint; directed forces accumulate across the phases.

`_set_phi1_policy("channel-tiled-64")` selects this experiment. Its RTC packet
types and entry points remain module-only, so the shared plugin ABI is unchanged.
The generated loops and compact strides are specialized to the policy's fixed
64-channel width. Before this specialization, hipRTC assigned 107 SGPR and
spilled 29 of them in tiled reverse. The final artifact uses 90 SGPR and 207
VGPR with no spills, compared with retained reverse's 84 SGPR and 206 VGPR.

Full-array parity used the regular 864-atom cell and the same deterministically
perturbed 32-atom cell as receiver-local. The maximum atomic-energy error was
`1.56e-6 eV`, maximum force error was `7.09e-6 eV/A`, and maximum stress error
was `2.90e-7 eV/A^3`. Switching one evaluator retained -> tiled -> retained
restored forces within `2.7e-15 eV/A`. Tiled evaluations issued two generated
forward and two generated reverse launches with zero fallbacks.

Matched Radeon 8060S timing used float32, the complete low-memory bundle, the
hipRTC direct artifact, 864 atoms, 78,624 directed edges, exact cutoff `6.0 A`,
skin `0`, 12 warmups, and 20 samples in fresh processes:

| Policy | Median | Phi1 workspace |
| --- | ---: | ---: |
| `retained` bracket 1 | `18.759 us/atom` | `17,694,720 B` |
| `channel-tiled-64` | `20.047 us/atom` | `8,847,360 B` |
| `retained` bracket 2 | `18.765 us/atom` | `17,694,720 B` |

The tiled overhead is `6.85%`: substantially below receiver-local's `3.47x`,
but still outside the provisional `5%` acceptance gate. A selected-region
rocprofv3 trace attributes the remaining cost to processing two phases. Tiled
R1 forward took `1.088 ms` versus `0.628 ms` retained, and tiled R1 reverse took
`2.654 ms` versus `1.976 ms`; the two tiled A1 projections added only about
`0.19 ms` relative to retained. Specializing separate phase-0/phase-1 reverse
entry points was rejected after measuring `20.109 us/atom`.

At 32,000 atoms and 2,912,000 directed edges, paired rocprofv3 allocation
traces measured Agent-1 peaks of `2,493,145,792 B` retained and
`2,165,465,792 B` tiled. The exact `327,680,000 B` reduction is
`32,000 * 10,240 B`, or `13.14%` of the retained peak. The exposed Phi1
workspace fell from `655,360,000 B` to `327,680,000 B`.

The experiment remains opt-in. Overlapping the phases is not an attractive next
step: avoiding the compact-buffer race requires a second `[N,40,64]` tile and
therefore restores the node memory being removed, in addition to per-phase
directed-force storage. The next performance-oriented experiment should retain
the full `[N,40,128]` shape in FP16 or BF16 while accumulating R1 and A1 in
FP32. That has the same `10,240 B/atom` payload but preserves one forward and
one reverse R1 schedule. Numerical parity and conversion cost are the gates.

## Compact edge-geometry experiment plan

The float32 factorized path currently retains each active displacement and
radius as four float64 values (`32 B/edge`). It also forms unit directions as
`xyz/r` in several reverse kernels. On consumer GPUs this combines avoidable
edge-scaled storage with float64 division.

The experiment is staged and opt-in. It does not change float64 inference,
legacy streamed modes, graph construction, or cutoff decisions.

1. Add a compact-geometry policy for eligible float32 factorized inference.
   Compute displacement, squared distance, cutoff admission, and radius in
   float64, then retain the active radius and a normalized direction.
2. First qualify float32 unit directions with a float64 radius (`20 B/edge`).
   Update every factorized consumer, including spherical-harmonic gradients,
   density-scale reverse, standard R0, generated RTC R1, ZBL, force reduction,
   and stress reduction. Raw displacement is reconstructed as `r * direction`
   only where required.
3. Compare against the existing float64-displacement control on periodic
   864-atom wurtzite AlN: exact model cutoff `6.0 A`, skin `0`, effective
   cutoff `6.0 A`, and the reported directed-edge count. Report median
   `us/atom`, energy error per atom, maximum/RMS force error, stress error, and
   NVE drift. Profile geometry, spherical-harmonic, R0 reverse, and R1 reverse
   kernels to determine whether division removal offsets any conversion work.
4. Measure Agent-1 peak allocation on the established 97,556-atom,
   8,877,596-directed-edge case. The predicted active-geometry saving for the
   first variant is `12 B/edge`, or `106,531,152 B` (`0.099215 GiB`).
5. Only if the first variant passes correctness, test float32 radius storage
   (`16 B/edge`, predicted saving `142,041,536 B` or `0.132286 GiB`). Keep
   graph admission in float64 and explicitly test spline-knot and cutoff-near
   edges. Do not replace radius with reciprocal radius because spline interval
   selection requires radius.
6. Store an additional reciprocal radius only if profiling shows a remaining
   reciprocal bottleneck. A float32 reciprocal costs `4 B/edge` (`0.033072
   GiB` on the large case), so it must outperform calculating one reciprocal
   per edge and reusing it within a kernel.

Acceptance requires no performance regression outside noise, numerical error
within the existing float32 qualification envelope, matching HIP and CUDA RTC
semantics from the shared generator, and exact agreement between reported and
traced compact-geometry allocation sizes. The existing representation remains
the default until all gates pass.

### Screening results

The first screening used `benchmarks/compact_edge_geometry_hip.cpp` on the
Radeon 8060S (`gfx1151`). Both kernels loaded the same float64 radius and
performed the same float32 radial arithmetic. The control loaded three
float64 displacement components and divided each by radius; the candidate
loaded three pre-normalized float32 components. Each measurement used 20
warmups and the table reports the median of three fresh runs.

| Directed edges | Repeats | Float64 xyz/r | Float32 direction | Local speedup |
| ---: | ---: | ---: | ---: | ---: |
| 78,624 | 500 | 0.017114 ms | 0.003556 ms | 4.81x |
| 8,877,596 | 50 | 1.568905 ms | 0.902589 ms | 1.74x |

At 864 atoms, one such consumer saves about `13.56 us`, or `0.0157 us/atom`.
The whole MH-0 control is `32.239 us/atom`, so this local result does not imply
a large end-to-end speedup. It does establish that the compact load and
multiply path is cheaper than float64 displacement loads and division. At the
large edge count, the candidate geometry occupies `0.165358 GiB` versus
`0.264573 GiB` for the active control geometry, the predicted `0.099215 GiB`
saving.

An initial representation screening quantized the graph, rebuilt float64
displacements for the previous ABI, and compared complete energy, forces, and
stress with the unchanged evaluator. This isolated representation error before
the production implementation. The case was periodic 864-atom AlN, exact
model cutoff `6.0 A`, skin `0`, effective cutoff `6.0 A`, and 78,624 directed
edges.

| Representation | Energy error (eV/atom) | Max force error (eV/A) | RMS force error (eV/A) | Max stress error (eV/A^3) |
| --- | ---: | ---: | ---: | ---: |
| float32 direction, float64 radius | 1.521e-7 | 8.146e-6 | 1.580e-6 | 3.240e-8 |
| float32 direction, float32 radius | 6.330e-7 | 1.022e-5 | 2.427e-6 | 3.123e-7 |

Both variants pass the established float32 force envelope in this case. The
float64-radius variant was selected for production: it has the smaller error,
leaves spline interval selection unchanged, and removes `12 B/edge`. Float32
radius is rejected for now; it is not exposed by the runtime policy or RTC ABI.

### Production implementation and HIP qualification

The opt-in `unit-f32-radius-f64-v1` policy is implemented for float32 ordinary
MACE and MACEField factorized inference using generated R0, M0, and RTC R1
execution, M1 recomputation, and no observer or parameter gradients. It is
selected automatically when `low_memory=True` chooses the capacity bundle; the
component default and automatic speed bundle remain `cartesian-f64-v1`. Graph
admission and radius remain float64.
The active device representation is three float32 unit-direction components
plus one float64 radius (`20 B/edge`). ZBL uses the same float64 radius and
projects its radial derivative with the compact unit direction.

The FP64 capacity bundle deliberately retains `cartesian-f64-v1`; the compact
FP32 geometry policy is not admitted for double-precision inference. The
standard R0, standard M0, and generated R1 paths all use their FP64 launch
profiles, and forward/adjoint state reuse remains active when capacity is
selected.

The shared RTC renderer and packet ABI carry `coordinate_scalar_size` and
`coordinates_are_unit`; HIP and CUDA therefore use the same generated kernel
body. Their thin launch adapters only validate and launch the packet. The
generation version is 2 so old cached artifacts cannot be loaded with the new
packet layout.

`benchmarks/compact_edge_geometry_precision.py` compares the two production
policies and runs deterministic NVE trajectories. On the Radeon 8060S
(`gfx1151`), the 864-atom, 78,624-edge AlN case produced:

| Quantity | Production compact vs control |
| --- | ---: |
| Energy error | `5.438e-7 eV/atom` |
| Maximum force error | `7.874e-6 eV/A` |
| RMS force error | `1.761e-6 eV/A` |
| Maximum stress error | `5.482e-8 eV/A^3` |
| Geometry workspace saving | `943,488 B` (`12 B/edge`) |

For 20 VelocityVerlet NVE steps at 300 K and a 1 fs timestep, the control drift
was `3.818e-5 eV/atom` and compact drift was `3.764e-5 eV/atom`. Their absolute
drift difference was `5.371e-7 eV/atom`; both remain well inside the established
`1e-3 eV/atom` MD threshold.

The 864-atom end-to-end medians used ten warmups and 20 measured evaluations.
The control took `32.363 us/atom`; compact geometry took `31.517 us/atom`, a
`2.61%` time reduction (`1.027x` speedup). Energy, forces, and stress were
evaluated, and both runs used the same graph, exact `6.0 A` cutoff, zero skin,
generated hipRTC R1 artifact, and adjoint-reuse state policy.

Finally, paired rocprofv3 allocation traces used the established `repeat(29)`
workload: 97,556 atoms and 8,877,596 directed edges. Agent-1 peak live
allocations fell from `7,638,827,528 B` (`7.114213 GiB`) to `7,532,296,376 B`
(`7.014998 GiB`). The exact `106,531,152 B` (`0.099215 GiB`) reduction equals
`12 B/edge`, matching both the tensor accounting and the geometry-workspace
delta. The traced one-step results were `38.659 us/atom` for the control and
`37.095 us/atom` for compact geometry; these profiler-attached single samples
are supporting evidence rather than the primary timing result.

## Retained-Y direct harmonic execution

The `y-only-direct-v1` policy removes shuffled coordinates and both
spherical-harmonic gradient views. Forward evaluates the 16 normalized
`l_max=3` harmonic values directly from compact float32 unit directions or
Cartesian float64 coordinates. Reverse reconstructs the 48 Cartesian
derivatives in bounded team scratch for standard R0 and MACEField response
kernels, and in shared memory for generated R1 reverse. The retained `Y`
values remain unchanged inputs to R0 and R1 forward.

`low_memory=True` requests whole-bundle automatic selection with retained
Phi1. For each new prepared graph, CUDA/HIP free and total memory are combined
with model-dimensional atom/edge peak estimates for the normal speed bundle
and the maximum-capacity bundle. The speed bundle is selected when it fits
after a 5% of total-memory reserve (minimum 512 MiB). Otherwise Y-only storage
is selected as part of the capacity bundle. Y-only admission requires
HIP or CUDA generated direct R1, edge-owned
standard R0 reverse, retained or `channel-tiled-64` Phi1, `l_max=3`, and no
observer or parameter gradients. It supports ordinary MACE and MACEField in
float32 and float64. Float32 requires compact unit-direction geometry; float64
retains Cartesian float64 geometry. An ineligible configuration keeps the rest
of the low-memory bundle active, selects retained harmonic gradients, and
exposes the reason through `harmonic_storage_fallback_reason`. The selected
whole bundle, decision reason, queried memory, reserve, and speed/capacity
estimates are available through the `low_memory_*` evaluator diagnostics.

MACEField reconstructs gradients in all four coordinate-derivative consumers:
Phi1 and A0 reverse, each in the generic HIP path and the fused CUDA path.
Generic kernels use `48*sizeof(Precision)` bytes per team (`192 B` FP32 or
`384 B` FP64). Eight-edge fused CUDA kernels use `8*48*sizeof(Precision)`
bytes per team (`1,536 B` FP32 or `3,072 B` FP64). Polarizability does not
request coordinate derivatives and therefore skips this scratch.

For compact geometry, the persistent workspace changes from `288 B/edge` to
`84 B/edge`:

| Edge state | Retained | `y-only-direct-v1` |
| --- | ---: | ---: |
| float32 unit direction plus float64 radius | `20 B` | `20 B` |
| shuffled float32 coordinates | `12 B` | `0 B` |
| normalized float32 `Y` values | `64 B` | `64 B` |
| normalized float32 `Y` gradients | `192 B` | `0 B` |
| Total | `288 B` | `84 B` |

For float64 Cartesian geometry, persistent storage changes from `568 B/edge`
to `160 B/edge`:

| Edge state | Retained | `y-only-direct-v1` |
| --- | ---: | ---: |
| Cartesian float64 coordinates plus radius | `32 B` | `32 B` |
| shuffled float64 coordinates | `24 B` | `0 B` |
| normalized float64 `Y` values | `128 B` | `128 B` |
| normalized float64 `Y` gradients | `384 B` | `0 B` |
| Total | `568 B` | `160 B` |

Both precisions remove 70.8% of persistent geometry-plus-harmonic edge state.
The evaluator exposes `harmonic_value_bytes`, `harmonic_gradient_bytes`, and
`shuffled_coordinate_bytes` so release builds can verify the accounting
without copying internal Kokkos views to the host.

HIP screening used the Radeon 8060S (`gfx1151`), 864 atoms, 78,624 directed
edges, exact cutoff `6.0 A`, skin `0`, the complete float32 low-memory bundle,
generated hipRTC forward and reverse, three warmups, and five measurements.
The retained and Y-only medians were `20.499 us/atom` and `20.141 us/atom`,
respectively. This short screen shows no performance regression; a longer
bracketed run remains appropriate before automatic selection.

The reported geometry workspace fell exactly from `22,643,712 B` to
`6,604,416 B`, a `16,039,296 B` reduction equal to `204 B/edge`. SpheriCart
launches fell from eight to zero while eight direct-value launches executed.
Both generated directions ran with zero fallbacks. A same-evaluator policy
switch measured `2.510e-7 eV/atom` energy error, `1.359e-5 eV/A` maximum force
error, and `3.049e-7 eV/A^3` maximum stress error, within the existing float32
acceptance envelope.

For the established 32,000-atom, 2,912,000-edge case, matched capacity runs
reported `838,656,000 B` retained and `244,608,000 B` Y-only, exactly `288`
and `84 B/edge`. The `594,048,000 B` reduction matches tensor accounting.
Their one-sample results were `23.219` and `22.576 us/atom`; these qualify
large-graph execution but are not primary timing evidence. An allocation trace
is still required to confirm the corresponding peak-memory change and
allocator overhead.

The retained-Phi1 restriction was relaxed after confirming that the generation-3
ordinary R1 reverse kernel already implements the same null-gradient
reconstruction as the tiled kernel. A matched HIP run on the 864-atom,
78,624-edge case used ten warmups and 20 measurements. Retained-Phi1 Y-only
measured `18.814 us/atom`, compared with `20.758 us/atom` for
`channel-tiled-64` Y-only. Retained Phi1 was therefore 9.4% faster in this run,
but used `17,694,720 B` (`20,480 B/atom`) of Phi1 workspace instead of
`8,847,360 B` (`10,240 B/atom`). Both policies retained the same `84 B/edge`
geometry and harmonic state.

Against retained harmonic gradients with the same retained-Phi1 policy,
Y-only differed by `2.514e-7 eV/atom`, `1.473e-5 eV/A` maximum force, and
`2.905e-7 eV/A^3` maximum stress. The exact geometry workspace reduction was
`22,643,712 B` to `6,604,416 B`. Both matched policy runs used the same
generation-3 artifact, reported zero fallbacks, and retained receiver-local
Phi1 as unsupported because its projected reverse kernel does not yet
reconstruct harmonic gradients.

MACEField and FP64 qualification used the Radeon 8060S (`gfx1151`) with 864
atoms, 78,624 directed edges, exact cutoff `6.0 A`, skin `0`, five warmups, and
15 measurements. Every run retained Phi1 and kept the remaining low-memory
policies fixed; retained gradients bracketed the Y-only measurement. FP32
primal execution improved from a bracketed `22.279 us/atom` to
`21.180 us/atom` (4.9% faster), while full BEC response changed from
`822.961` to `824.673 us/atom` (0.2% slower). FP64 primal changed from
`94.455` to `98.562 us/atom` (4.4% slower), and full BEC response changed from
`1129.663` to `1164.462 us/atom` (3.1% slower). FP64 therefore remains a
capacity-oriented tradeoff rather than a throughput improvement.

Direct low-memory MACEField parity covered energy, forces, stress,
polarization, polarizability, and BEC. Maximum FP32 differences against
retained harmonic storage were `2.65e-5 eV` in energy, `5.76e-6 eV/A` in
forces, `1.87e-7 eV/A^3` in stress, `2.55e-8` in polarization, `5.89e-6` in
polarizability, and `2.45e-6` in BEC. The largest FP64 difference across all
six outputs was `1.78e-14`.

## CUDA qualification

The merged CUDA implementation was qualified on 2026-08-15 using an NVIDIA
GeForce RTX 5090 (`sm_120`, 32,607 MiB), driver `610.43.02`, and CUDA 13.3.
The build enabled Kokkos CUDA plus Serial, CUDA SpheriCart, and
`Kokkos_ARCH_BLACKWELL120`. The source was the `ba938d4` merge plus the compact
prepared-stress correction in this change. The installed extension SHA-256 was
`4a88f27d09bd70258f11bb61997097791e15d16366ff00ba384196706fac215b`.

The source checkpoint was `mace-mh-0.model`, SHA-256
`d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d`.
The current extractor produced an Al/N `omat_pbe` compact-radial JSON with
SHA-256 `2154d8922f700eb08f3beb90e35da7d9daec9795ddecfb9599994436c493fc2a`.
This differs from the earlier `d3e825...` JSON because the current schema uses
`execution_contracts`; the checkpoint itself is unchanged.

The primary timing workload was periodic wurtzite AlN with 864 atoms, an exact
model cutoff of `6.0 A`, neighbor-list skin `0`, effective cutoff `6.0 A`, and
78,624 directed edges. All cases used float32 factorized execution, generated
NVRTC R1, M1 recomputation, ten warmups, and 20 measured evaluations in fresh
processes. Energy, forces, and stress were produced and checked. The table
reports the benchmark's synchronized native-evaluator interval; force and
stress reductions are collected after that timed interval.

| MH-0 state | Edge geometry | Median (ms) | Median (us/atom) | Sampled GPU memory (MiB) |
| --- | --- | ---: | ---: | ---: |
| Full retention | Cartesian float64 | `3.892623` | `4.505350` | `862` |
| Reuse adjoints | Cartesian float64 | `3.683523` | `4.263337` | `808` |
| Reuse adjoints | float32 unit + float64 radius | `3.635248` | `4.207463` | `806` |

Adjoint reuse reduced the median by `5.37%`; compact geometry reduced it by a
further `1.31%`. The complete reduced-memory configuration was `6.61%` faster
than the full-retention Cartesian control in this run. At 864 atoms it reported
`17,252,352 B` of reused state, `13,824 B` of auxiliary state, and a geometry
workspace reduction from `23,587,200 B` to `22,643,712 B`. The exact compact
reduction is `943,488 B`, or `12 B/edge`.

The production precision script compared Cartesian and compact stress and ran
20 deterministic VelocityVerlet NVE steps at 300 K and 1 fs. Compact geometry
differed by `3.022e-7 eV/atom` in energy, `9.716e-6 eV/A` in maximum force,
and `2.387e-8 eV/A^3` in maximum stress. Control drift was
`3.8483e-5 eV/atom`, compact drift was `3.8776e-5 eV/atom`, and their absolute
difference was `2.934e-7 eV/atom`. Compute Sanitizer then reported zero errors
for a Cartesian-then-compact prepared-stress reproducer.

Fresh-process capacity runs used `repeat(29)`: 97,556 atoms, 8,877,596 directed
edges, the same cutoff and skin, one warmup, and one measured evaluation.

| MH-0 state | Edge geometry | Time (ms) | Time (us/atom) | Sampled GPU memory (MiB) | Geometry workspace (B) |
| --- | --- | ---: | ---: | ---: | ---: |
| Full retention | Cartesian float64 | `369.748` | `3.790109` | `13,392` | `4,367,777,232` |
| Reuse adjoints | Cartesian float64 | `370.011` | `3.792808` | `7,862` | `2,663,278,800` |
| Reuse adjoints | float32 unit + float64 radius | `389.528` | `3.992871` | `7,760` | `2,556,747,648` |

The one-sample large-case timings qualify completion and are not used for
performance decisions. The `nvidia-smi` samples show a `5,530 MiB` reduction
from adjoint reuse and another `102 MiB` from compact geometry. Exact tensor
accounting reports `1,947,998,208 B` (`1.814215 GiB`) of reused state with
`1,560,896 B` of auxiliary state, plus a `1,704,498,432 B` (`1.587438 GiB`)
geometry-gradient workspace reduction. Compact geometry then removes exactly
`106,531,152 B` (`0.099215 GiB`, `12 B/edge`), leaving `177,551,920 B` of
active radius and unit-direction storage. M1 polynomial value and adjoint
active and capacity bytes were zero in all three runs, and neither reduced
policy reported a fallback.
