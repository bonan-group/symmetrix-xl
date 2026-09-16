# MH-0 versus MH-1 reverse-kernel comparison

## Conclusion

MH-0 and MH-1 use the same interaction-reverse backbone: receiver-owned
forward aggregation, source-owned transpose, and edge-owned differentiation.
The MH-0 implementation is therefore a useful scheduling reference, but its
generated kernel cannot be copied directly. MH-1 inserts a learned edge-weight
program and must return edge-embedding, harmonic, and cutoff adjoints before
the conditioner and coordinate reverse stages can finish the force derivative.

The first interaction-1 edge experiment is promoted as
`hybrid-path-tiled-v7`. It loads each learned linear-weight coefficient once
for a four-edge batch, reducing edge reverse by 16.1% and the complete
5,000-atom energy/forces/stress evaluation by 7.25%. It also improves the
625-atom evaluation by 6.09%. A subsequent 64/128/256-thread source-launch
screen retained 128 threads: 64 threads reduced the 5,000-atom interaction-1
source kernel by 5.4% but regressed the 625-atom evaluation by 3.1%, while 256
threads regressed the 5,000-atom evaluation by 4.4%. A source-owned fusion
modeled on MH-0 remains the next architectural prototype because launch
geometry alone does not provide a workload-robust source improvement.

## Workload and exact artifacts

- Model: MACE-MH-1 `omat_pbe`, float32, and the corresponding MH-0 model.
- Device: NVIDIA GeForce RTX 5090, compute capability 12.0.
- State: deterministic 5,000-atom SrTiO3.
- Model cutoff: 6.0 Angstrom.
- Neighbor skin: 0.5 Angstrom; effective cutoff: 6.5 Angstrom.
- Retained directed edges: 511,926; exact-cutoff directed edges: 370,000.
- MH-0 JIT artifact: `jit-r1-f32-6fdf3e63624a0e45`, variant
  `edge-wave-w32-t32-b8`, generated source SHA-256
  `ca1733843863540dd715c309f7a989b44668c6bd0489fbd3300938343d5300cd`.
- MH-1 JIT artifact: `jit-mh1-v4-86823d33d34d12b5`, generated source SHA-256
  `d33100d50ea5ffe5b956fe073d89626b2b1ce71accefbf2990435d9ac0d5dc6f`.
- Promoted v7 variant:
  `jit-mh1-v4-86823d33d34d12b5-fwd-24b2044489fa-src-0e39c7ec4bc9-edge-41ffab6f6e8d-node-full-retention-v1`.

The exact generated sources, manifests, cubins, and brace-balanced source
analysis are exported under ignored local storage at
`benchmarks/.artifacts/mh0_mh1_reverse_comparison_20260813/`. They are evidence,
not checked-in source modules.

## Promoted edge-reverse result

The controlled comparison uses the same current CUDA extension (SHA-256
`c087a233...`) and changes only the generated v6/v7 schedule. Both workloads
request energy, forces, and stress.

| Workload | Directed edges | v6 | v7 | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 625 atoms | 63,988 | 20.433 ms | 19.188 ms | 1.245 ms, 6.09% |
| 5,000 atoms | 511,926 | 154.281 ms | 143.096 ms | 11.185 ms, 7.25% |

Both use a 6.0 Angstrom model cutoff, 0.5 Angstrom skin, and 6.5 Angstrom
effective neighbor cutoff. Energy and forces are bitwise equal at both sizes.
Maximum v6/v7 differences are 1.36e-20 eV/Angstrom^3 in stress and 1.73e-18
Angstrom in two-step positions at 625 atoms; at 5,000 atoms they are 3.39e-21
eV/Angstrom^3 and 8.88e-16 Angstrom.

The measured Nsight Systems ranges contain 115 kernels per evaluation. Their
two-step means localize the performance change:

| Kernel family | v6 | v7 | Delta |
| --- | ---: | ---: | ---: |
| Interaction-1 edge reverse | 43.069 ms | 34.515 ms | -8.553 ms, -19.9% |
| Interaction-0 edge reverse | 15.427 ms | 14.401 ms | -1.025 ms, -6.6% |
| Interaction-1 source reverse | 15.907 ms | 16.812 ms | +0.905 ms |
| Interaction-0 source reverse | 9.944 ms | 9.956 ms | +0.012 ms |
| Interaction-1 forward | 16.036 ms | 16.330 ms | +0.294 ms |
| Interaction-0 forward | 9.435 ms | 9.442 ms | +0.007 ms |
| All other kernels | 41.717 ms | 41.628 ms | -0.089 ms |
| Complete kernel sum | 151.535 ms | 143.085 ms | -8.450 ms, -5.6% |

Each steady-state step transfers 120,000 bytes H2D for positions and 160,076
bytes D2H in four reduced result copies. There is no full pair-force D2H or
edge-geometry H2D transfer; geometry, atom-force reduction, and stress virial
reduction remain on device.

The production CUDA 12.8 NVRTC cubins use 56/71 registers per thread for v6
interaction 0/1 edge reverse and 96/115 for v7. All four use 11,024 bytes total
shared memory and zero stack/local memory. A focused CUDA 13.3 NCU replay of
interaction 1 reports 56 to 96 registers and 66.12% to 37.40% achieved
occupancy, but eligible warps improve from 0.45 to 0.50 and warp cycles per
issued instruction fall from 28.27 to 14.17. Long-scoreboard stalls per issued
instruction fall from 16.86 to 6.54 and barrier stalls from 2.11 to 0.64. The
replay duration falls from 42.219 to 37.805 ms (10.5%). Thus the saved
dependent on-chip work outweighs the deliberate occupancy loss; no spill or
DRAM-bandwidth trade was introduced.

## Rejected source-launch screen

The source kernel maps whole warps to independent
`(source, partition, channel tile)` owners, so 64/128/256-thread blocks preserve
per-owner arithmetic order and use the same CUDA 12.8 NVRTC cubin instructions.
Only launch geometry changes. The screen used the same RTX 5090, float32 model,
6.0 Angstrom cutoff, 0.5 Angstrom skin, 6.5 Angstrom effective cutoff, and
energy/forces/stress workload as the promoted edge result.

| Source threads | 5,000 atoms, 511,926 edges | 625 atoms, 63,988 edges | Decision |
| ---: | ---: | ---: | --- |
| 64 | 142.174 ms | 19.781 ms | reject: 625-atom regression |
| 128 | 143.096 ms | 19.188 ms | retain |
| 256 | 149.384 ms | not run after large-case rejection | reject |

Under Nsight Systems, 64 threads measured 143.711 ms end to end versus 144.726
ms for 128 threads. Interaction-1 source reverse fell from 16.812 to 15.901 ms
(-0.911 ms, -5.4%); interaction-0 source reverse was effectively unchanged at
9.956 versus 9.933 ms. This gain is real at 5,000 atoms but fails the required
small-workload gate: 19.781 versus 19.188 ms is a 0.593 ms (3.1%) regression.
Energy and forces remain bitwise equal. Maximum differences are 6.78e-21
eV/Angstrom^3 in stress and 3.47e-18 Angstrom in two-step positions at 625
atoms; at 5,000 atoms they are 6.78e-21 eV/Angstrom^3 and 2.22e-16 Angstrom.

The measured 64-thread trace still transfers only 120,000 bytes H2D for
positions and 160,076 bytes D2H in four reduced result copies per step. It
introduces no full pair-force or edge-geometry transfer. Because the schedule
is rejected, no source-thread or public variant-identity change remains in
production.

## Measured layer split

The baseline Nsight Systems trace contains three evaluations: warm-up followed
by two measured evaluations. Dividing the aggregate durations by three gives
the per-evaluation values below and reproduces the recorded family totals.

| Kernel | Interaction 0 | Interaction 1 | Family total | Interaction-1 share |
| --- | ---: | ---: | ---: | ---: |
| R1 reverse: edge | 14.999 ms | 41.424 ms | 56.423 ms | 73.4% |
| R1 reverse: source | 6.817 ms | 16.604 ms | 23.421 ms | 70.9% |

Together these four kernels take 79.844 ms, or 53.2% of the 149.937 ms
baseline GPU-kernel total. Halving only interaction-1 edge reverse would reduce
the kernel sum to about 129.225 ms, a 1.16x kernel-level speedup. Halving both
interaction-1 reverse kernels would reduce it to about 120.923 ms, a 1.24x
kernel-level speedup. These are Amdahl bounds, not measured candidate results.

## Shared mathematical backbone

For an edge from source `s` to receiver `t`, both implementations evaluate a
tensor-product message of the form

```text
m_t += cutoff_e * weight_e,p,c * CG_p(a,b,o)
       * source_s,a,c * harmonic_e,b
```

Reverse evaluation has the same ownership decomposition:

1. A source owner traverses its outgoing edges and accumulates
   `dL/d(source_s,a,c)` without atomics.
2. An edge owner contracts source values and receiver-output adjoints to
   differentiate the edge-dependent quantities.
3. Later reverse stages convert the edge-dependent adjoints into coordinate
   derivatives.

MH-0 specializes `weight` directly as a cubic radial-spline function selected
from a 64-dimensional radial embedding. Its fused
reverse evaluates the radial value and derivative once per path, accumulates a
512-value source adjoint in shared memory, and writes the directed coordinate
force from the same source-owned traversal.

MH-1 instead evaluates

```text
weight_e,p,c = bias_p,c + fixed_e,p,c
             + sum_phi linear_weight_phi,p,c * edge_phi_e,phi
```

Its edge reverse must produce all three of:

```text
dL/d(edge_phi_e,phi)
dL/d(harmonic_e,b)
dL/d(cutoff_e)
```

The first is consumed by generated conditioning reverse. Harmonic and cutoff
adjoints are then added to the common edge adjoint arrays before radial and
spherical-harmonic reverse complete the force derivative. A direct port of the
MH-0 coordinate-force kernel would omit this learned-edge derivative chain.

## Generated implementation comparison

| Property | MH-0 R1 | MH-1 interaction 0 | MH-1 interaction 1 |
| --- | ---: | ---: | ---: |
| Channel multiplicity | 128 | 128 | 128 |
| Edge harmonics | 16 | 16 | 16 |
| Source components | 4 | 1 | 4 |
| Tensor-product paths/instructions | 10 | 4 | 10 |
| Learned edge embedding | radial 64 | phi 64 | phi 64 |
| Learned weight width | path-specific radial | 512 | 1,280 |
| Source partitions | one owner body | 1 | 2 |
| Edge batch | one edge/subgroup | 4 | 4 |

The current interaction-1 edge kernel is a 3,516-line emitted function with
280 loops, 25 block barriers, 240 shuffle sites, and six shared arrays. Those
are static source-shape counts, not dynamic instruction totals. It batches four
edges and four paths, stores
`path_weight_adjoint[4][4][128]`, and then applies the transpose of the learned
64-to-1,280 linear map to form phi adjoints.

The transpose part already reuses one `linear_weight` load across four edges.
The forward weight reconstruction does not: the generated producer emits an
independent 64-phi loop for each path and each edge. Consequently the same
linear-weight coefficient can be loaded once per four-edge batch, and the same
edge-phi value can be loaded once per four-path batch, because each is invariant
along the other batching axis.

The interaction-1 source wrapper maps one warp to
`(source owner, source partition, 32-channel tile)`. Partition 0 accumulates the
scalar source block. Partition 1 accumulates the three vector components and
has substantially larger emitted expressions and live state. Source ownership
is already the correct MH-0-style deterministic ownership; the immediate issue
is the per-thread program inside that owner.

## Compiled and profiled evidence

| Kernel | Registers/thread | Shared memory | Achieved occupancy | L1 throughput | L2 throughput | DRAM throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MH-1 interaction-1 edge reverse | 56 | 10,000 B static | 66.43% | 80.93% | 79.86% | 0.56% |
| MH-1 interaction-1 source reverse | 168 | none | 23.50% | 73.32% | 72.02% | 1.57% |

Neither kernel spills local or shared memory. Both have greater than 99% L2 hit
rate, low DRAM use, and about 58-60% of issue latency attributed to long
scoreboard dependencies on L1TEX operations. The edge kernel has 7.99 active
but only 0.43 eligible warps per scheduler. The source kernel has only 2.84
active and 0.28 eligible warps per scheduler. This supports reducing dependent
load instructions and improving reuse; it does not support treating either
kernel as a DRAM-bandwidth problem.

The edge launch has 2,720 blocks, 1.78 waves/SM, and a large partial second
wave. The source launch has 680 blocks and 1.33 waves/SM. Launch-tail changes
must be evaluated together with the grid-stride loop: simply reducing the grid
can increase per-block work and leave total time unchanged.

For reference, the exact MH-0 cubin uses 56 registers for split source reverse,
254 for split edge reverse, and 128 registers plus 3,096 bytes shared for the
fused reverse. The useful lesson is not low register count in every kernel; it
is reuse under exclusive ownership without a graph-sized intermediate.

## Optimization sequence

### 1. Interaction-1 edge learned-weight microkernel

Keep the current four-edge, one-writer schedule and restructure each four-path
batch as a small matrix microkernel:

```text
for phi:
    load phi[e0:e3, phi]
    for path in path_batch:
        load linear_weight[phi, path, channel] once
        weight[path, e0:e3] += linear_weight * phi[e0:e3, phi]
```

This transformation is implemented and promoted as `hybrid-path-tiled-v7`.
It preserves each edge's ascending-phi accumulation and sparse contraction
order while interleaving only independent edges. Edge policy schema `/2`
includes `reverse_schedule` in its digest so stale v6 schedule mappings fail
closed even though launch geometry is unchanged.

Also test whether harmonic and receiver-adjoint values shared by multiple terms
can be loaded once into short-lived scalars. Do not materialize full learned
weights: at 511,926 edges this would require about 2.62 GB for interaction 1
and 3.67 GB for both interactions in float32, directly opposing the streamed
edge memory objective.

### 2. Interaction-1 source live-state reduction

Preserve one deterministic source owner and coalesced 32-channel tiles. Test:

- shorter path scopes or generated helper boundaries that reduce compiler live
  ranges without repeating the complete edge traversal;
- paired-path learned-weight reconstruction so one edge-phi load serves more
  than one path, only where register count remains bounded;
- 64/128/256-thread blocks and persistent-block counts;
- a component-sliced vector partition only as a measured trade-off, because
  splitting the three source components repeats edge traversal and learned
  weight reconstruction unless a shared tile is introduced.

The first promotion gate is materially below 168 registers with no local
memory and lower complete source-reverse time. Occupancy alone is not a result.
The completed block-size screen found 64 threads scale-dependent and 256
threads slower; retain 128 threads unless a future calibrated policy explicitly
models workload size. The next experiment should reduce repeated learned-weight
work or live state rather than rescreening static block sizes.

### 3. MH-0-inspired source-owned fused reverse

Prototype one source-owned traversal that computes source adjoints and the
edge-local phi, harmonic, and cutoff adjoints together. Every directed edge has
exactly one source owner, so edge outputs can retain one-writer semantics. This
can eliminate the second learned-weight reconstruction and reuse receiver
adjoints and harmonics across the two reverse products.

Unlike MH-0, the prototype must still write or directly consume the three MH-1
edge adjoints. Start by preserving the existing graph embedding, harmonic, and
cutoff adjoint arrays and generated conditioner boundary. Fuse conditioner or
coordinate reverse only after a profile shows that those intermediates, rather
than the tensor-product work, have become dominant.

Reject the fusion if variable source degree causes unacceptable imbalance,
register/shared-memory growth removes the reuse benefit, or deterministic
accumulation changes. A global GEMM that materializes all learned weights is
not an acceptable substitute; a bounded in-kernel tile may be evaluated.

## Verification gates

- Render, compile, and inspect every candidate cubin before benchmark use.
- Require zero local-memory spills and record registers, shared memory, launch
  geometry, waves, occupancy, eligible warps, cache throughput, and duration.
- Compare interaction 0 as a control and interaction 1 as the primary target.
- Use identical 625- and 5,000-atom prepared graphs for screening, followed by
  synchronized end-to-end ASE evaluations.
- Preserve energy, forces, stress, deterministic repeatability, graph-token
  lifecycle, JIT cache identity, and materialized fallback behavior.
- Record atom count, 6.0 Angstrom cutoff, 0.5 Angstrom skin, 6.5 Angstrom
  effective cutoff, and directed-edge count for every result.
- Promote only a change that improves the whole evaluation; a faster replayed
  kernel or higher occupancy is insufficient.
