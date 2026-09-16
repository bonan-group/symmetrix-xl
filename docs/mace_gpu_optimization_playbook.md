# Optimizing MACE GPU execution with radial splines and specialized node kernels

This document describes the optimization method used to reduce the CUDA FP32
time of the MACE-MH-1 path from `33.170419 us/atom` to about `6.67 us/atom` on a
matched RTX 5090 workload. The method is intended as a reusable playbook for
other MACE-family models, not as a prescription to copy MH-1-specific launch
sizes.

The largest improvement came from reusing an idea already established for
standard MACE/MH-0: replace the radial neural networks evaluated on every edge
with compact cubic spline tables. For MH-1 this must be an **ordered-pair**
spline because the radial function depends separately on the source and target
elements. Once the R stages were made inexpensive, profiling exposed the M0
and M1 node programs as the next bottleneck. Those programs were then lowered
to model-specialized runtime-compiled kernels with carefully measured
cross-node weight reuse.

## 1. Result and scope

The primary qualification workload was:

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0 |
| Model | Extracted MACE-MH-1 `omat_pbe` Al/N model |
| Precision | FP32 learned state; existing coordinate/output precision contract |
| Structure | 864-atom periodic wurtzite AlN |
| Model cutoff | `6.0 A` |
| Neighbor-list skin | `0.0 A` |
| Effective cutoff | `6.0 A` |
| Directed edges | 78,624 |
| Outputs | Energy, node energies, forces, and stress |
| Execution | Prepared direct graph, required cached NVRTC, zero fallback |

The measured progression was:

| Milestone | Time (`us/atom`) | Change from previous |
| --- | ---: | ---: |
| MH-1 radial-MLP reference | 33.170419 | baseline |
| Ordered-pair spline R path | 15.807059 | -52.3% |
| Grouped and specialized M0/M1 kernels | 9.653807 | -38.9% |
| 1,024-row full-retention arena | 7.836325 | -18.8% |
| Earlier optimized node result | 7.706371 | -1.7% |
| Wider R0/R1 reverse owners | 6.675143 | -13.4% |
| Current full-retention control | 6.659778 | matched control |
| Current retained-primal adjoint reuse | 6.669469 | +0.15% vs full retention |

Overall, the optimized path is about `4.97x` faster than the radial-MLP
reference. The historical matched MH-0 result is `3.6305468 us/atom`, making a
two-times-MH-0 target `7.2610936 us/atom`. The current MH-1 result is
`0.591625 us/atom` below that target.

The retained-primal reuse result has bitwise-identical energy and node energies
relative to full retention. Its maximum force difference is
`6.91e-8 eV/A`, and its maximum stress difference is `2.25e-11 eV/A^3`.
The generation-6 focused source suite reported `181 passed, 1 skipped`.

These numbers are not portable constants. They establish which ideas mattered
on one controlled workload. A different checkpoint, composition, graph,
precision, or GPU must be measured independently.

## 2. Decompose execution before optimizing it

For a two-interaction MACE model, use the following stage vocabulary:

```text
geometry and spherical harmonics
  -> R0 forward
  -> M0 forward
  -> R1 forward
  -> M1 forward
  -> M1 reverse
  -> R1 reverse
  -> M0 reverse
  -> R0 reverse
  -> force and stress assembly
```

`R0` and `R1` are edge stages. They compute distance-dependent radial weights,
combine them with spherical harmonics and source features, and reduce messages
and density contributions into receiver nodes.

`M0` and `M1` are node stages. They consume the reduced message and density,
then execute normalization, equivariant linears, gates, correlation-three
products, skip connections, updates, and readouts. Their reverse stages
propagate adjoints back to the edge-stage outputs.

This separation is important because the two optimization classes are
different:

- The radial spline changes how a fixed edge function is evaluated.
- Runtime-compiled M0/M1 kernels change how the dynamic node program is
  scheduled without changing its mathematics.

On the initial CUDA spline trace, the four M stages accounted for `85.8%` of
the device-time gap to a same-binary MH-0 trace. The spline had already removed
the dominant R-stage excess, so further radial tuning could not close the
remaining gap.

## 3. Identify the splinable radial boundary

### 3.1 The eligibility test

A function is safe to tabulate only if all its inputs are fixed by:

```text
(interaction index, source element, target element, scalar distance)
```

It must not depend on evolving node features, the local neighborhood, a
receiver reduction, or another runtime state. This criterion, rather than a
particular class name in a PyTorch checkpoint, defines the spline boundary.

For interaction `s`, ordered species pair `p = (source_type, target_type)`,
channel `c`, and distance `r`, MH-1 tabulates the complete radial outputs:

```text
C[s,p,c](r) = cutoff(r) * complete_tp_radial_coefficient(s, p, c, r)

D[s,p](r)   = cutoff(r) * tanh(density_mlp(s, p, r)^2)
```

The exact placement of `cutoff(r)` is taken from the extracted model's
`apply_cutoff` contract. If the basis entering the network already contains the
cutoff, it must not be multiplied again. If it does not, the cutoff is folded
into the final table exactly once.

`C` includes the entire fixed radial computation needed by the convolution:

- radial basis and distance transform;
- source/target pair conditioning;
- all radial-network layers and nonlinearities;
- the final affine bias;
- the cutoff under the model's exact convention.

`D` similarly includes the complete density branch. Tabulating only an
intermediate basis or omitting the final bias is a different model and will
usually produce especially visible force errors.

### 3.2 Ordered pairs are mandatory

The pair index is directed:

```text
pair_index = source_active_type * num_active_types + target_active_type
```

In general,

```text
C[s,(Al,N),c](r) != C[s,(N,Al),c](r)
D[s,(Al,N)](r)   != D[s,(N,Al)](r)
```

Do not apply triangular or symmetric pair compression unless symmetry is
proved from the extracted model. MH-1 uses distinct source and target element
conditioning, so merging `(a,b)` with `(b,a)` is invalid.

Only active model elements need to be materialized. This avoids allocating a
table for the square of a framework-wide element vocabulary when an evaluator
uses only a small composition.

### 3.3 What remains dynamic

The spline does not approximate the entire MACE interaction. The following
operations remain dynamic:

- geometry and spherical harmonics;
- tensor products involving evolving source features;
- receiver message and density reductions;
- learned density normalization;
- equivariant residual and message linears;
- gates, products, updates, and readouts;
- reverse propagation through all dynamic operations;
- harmonic-coordinate reverse, force assembly, and stress assembly.

Keeping this boundary narrow is what makes the method both accurate and
transferable.

## 4. Build value and derivative tables

Use the extracted radial implementation as the oracle. Do not reimplement the
paper equations with assumed dimensions. The model contract determines the
number of interactions, active species, basis functions, hidden widths, output
channels, cutoff convention, and distance transform.

For each interaction and active ordered pair:

```text
for node k on a uniform radius grid:
    r = grid_radius(k)
    C_value[k], D_value[k] = exact_extracted_radial_oracle(s, pair, r)
    C_slope[k], D_slope[k] = exact_distance_derivative_of_oracle(...)

for each adjacent node interval:
    coefficients = cubic_hermite(
        left_value, right_value, left_slope, right_slope, interval_width
    )
```

At runtime, one interval lookup and polynomial evaluation returns both the
value and the analytic derivative of the interpolant:

```text
C, C_prime, D, D_prime = ordered_pair_spline(s, pair, r)
```

The derivative tables are not optional. Energy-only agreement does not qualify
a potential used for forces or stress.

Use `std::size_t` or an explicitly 64-bit type for flattened table indices and
allocation extents. Cast before multiplying interaction, pair, interval, and
channel dimensions.

## 5. Forward and reverse execution

The following pseudocode shows the intended ownership and data flow. It is
schematic; actual tensor-product layouts and irreducible-representation blocks
come from the extracted contract.

### 5.1 Forward

```text
for interaction s in [0, 1]:
    clear message and density accumulators

    parallel for receiver node i:
        for edge e in receiver_edges(i):
            j = source(e)
            pair = ordered_pair(type[j], type[i])
            r = distance(e)

            C, C_prime, D, D_prime = spline(s, pair, r)
            Y = spherical_harmonics(e)

            message[i] += tensor_product(features[j], Y, C)
            density[i] += D

    parallel for node i:
        normalization = alpha[s] + beta[s] * density[i]
        residual = linear_residual[s](input_features[i])
        mixed = linear_message_1[s](message[i]) / normalization
        pre_gate = residual + mixed
        gated = gate[s](pre_gate)
        interaction = linear_2[s](gated)
        product = correlation_3_product[s](interaction, type[i])
        output_features[i] = (
            product_linear[s](product)
            + skip_linear[s](input_features[i])
        )
        energy += readout[s](output_features[i])
```

The implementation may recompute the spline interval during reverse instead of
retaining graph-sized `C`, `C_prime`, `D`, and `D_prime` arrays. This is often a
good GPU tradeoff because cubic evaluation is small while MH-1 has thousands
of radial path weights per edge.

### 5.2 Reverse

```text
for interaction s in [1, 0]:
    message_adjoint, density_adjoint, source_feature_adjoint =
        reverse_node_program(s, output_adjoint)

    parallel for source node j:
        for edge e in source_edges(j):
            i = receiver(e)
            pair = ordered_pair(type[j], type[i])
            r = distance(e)

            C, C_prime, D, D_prime = spline(s, pair, r)

            tp_weight_adjoint, harmonic_adjoint, feature_adjoint =
                reverse_tensor_product(
                    features[j], spherical_harmonics(e), C,
                    message_adjoint[i]
                )

            radial_derivative = (
                dot(tp_weight_adjoint, C_prime)
                + density_adjoint[i] * D_prime
            )

            source_feature_adjoint[j] += feature_adjoint
            force_and_stress_adjoint += coordinate_reverse(
                e, radial_derivative, harmonic_adjoint
            )
```

Prepared receiver-contiguous edges make forward reduction natural. A prepared
source-owned schedule avoids source-feature atomics in reverse. The best exact
ownership depends on the runtime, but changing it must not alter the ordered
pair or derivative contract.

## 6. Why the spline produced the largest gain

The matched CUDA A/B comparison used the same extension, graph, precision,
outputs, and required JIT policy:

| Executor | End-to-end (`us/atom`) | Device trace (`us/atom`) |
| --- | ---: | ---: |
| Per-edge radial MLP | 33.170419 | 31.125 |
| Ordered-pair spline | 15.807059 | 15.464 |

The four aligned R-stage owners fell from `18.968` to `3.537 us/atom`. M0 and
M1 owner times were unchanged within trace noise. This is strong attribution:
the spline saved about `17.36 us/atom` end to end by removing repeated radial
network work, rather than by accidentally changing node execution or the graph.

The key reuse ratio is:

```text
radial MLP: evaluate once per directed edge and interaction
spline:     build once per interaction and active ordered pair,
            then perform a small lookup per directed edge
```

For a fixed model and composition, table construction is outside the warmed
steady-state region. The benefit grows with edge count, while table size grows
with the number of active ordered pairs and radial output channels.

## 7. Optimize M0/M1 only after the R stages are under control

After spline conversion, the matched profile showed 292 MH-1 M-stage launches
versus 18 for MH-0. This did not mean MH-1 was evaluated repeatedly. The count
was the product of two factors:

```text
ceil(864 nodes / 256 arena rows) * 73 generated operations per tile
= 4 * 73
= 292 launches
```

The 73 operations were mostly decomposed irreducible-representation blocks and
forward/reverse phases, not 73 conceptual neural-network layers. Some sibling
blocks are independent and can be grouped. True producer/consumer boundaries,
reductions, and separately parallelized phases still require synchronization.

### 7.1 Preserve cross-node weight reuse

The most important M-stage lesson was that fewer launches alone can be much
slower. A one-block-per-node monolithic analytical M1 reverse reduced the
post-reverse launch count from 88 to four but regressed from `15.807059` to
`42.878505 us/atom`. Its four specialized kernels alone cost
`30.867725 us/atom`.

The failure came from losing cross-node reuse of shared linear weights. The
accepted grouped kernels stage padded `32 x 32` weight tiles in shared memory
and apply one tile to multiple nodes. They preserve the useful reuse of the
staged program while combining independent irreps siblings where dependencies
allow it.

The general rule is:

```text
optimize useful work per weight load and per synchronization,
not launch count in isolation
```

### 7.2 Tune geometry per kernel family

Tile widths that helped one family did not automatically help another:

- Layer-1 message reverse benefited from a wider node tile because its 5,120
  logical channels retained enough parallel work.
- Applying the same geometry to the smaller M0 message reverse regressed that
  family by 49%.
- Four nodes per thread improved product reverse by reusing each coefficient
  load while preserving per-node summation order.
- Eight nodes per thread drove one kernel to 255 registers/thread and regressed
  despite having no explicit local-memory allocation.
- Selected `linear2` kernels benefited from 64- and later 128-node tiles, but
  every layer and direction was screened independently.

For each candidate record at least:

- end-to-end `us/atom` in fresh processes;
- target-family and canonical-stage device time;
- registers per thread;
- shared memory per block;
- stack and local memory;
- spills and achieved occupancy when hardware counters are available;
- numerical parity for all required outputs.

Occupancy is diagnostic, not an acceptance metric. A candidate with lower
occupancy may still win through greater reuse, while a low-register candidate
may lose by repeating work or destroying locality.

### 7.3 Understand arena rows correctly

The CUDA full-retention arena change from 256 to 1,024 rows was the largest
single M0/M1 milestone:

```text
generated node launches: 176 -> 44
time:                    9.653807 -> 7.836325 us/atom
extra workspace:         about 84 MiB for the 864-atom workload
```

An arena row is workspace for one processed node. It is not a node in the
atomic structure, an edge, or a learned graph node with independent topology.
The test still contained 864 atoms. A 1,024-row arena simply held all 864 node
rows in one outer pass instead of four passes of 256, 256, 256, and 96 rows.

For a workload with 800 atoms, a 1,024-row capacity likewise produces one
800-row pass; 224 rows are unused capacity. The policy remains a memory versus
throughput choice. It should be conditional on backend, retention mode, and
available memory rather than treated as a universal model constant.

### 7.4 Retain expensive primals and reuse their dead storage

In reverse-mode differentiation, a *primal* is a value produced by forward
execution that a reverse derivative reads later. A low-memory policy should
not assume that all primals are equally cheap to reconstruct. First list each
reverse consumer, its required forward value, the value's size, and the cost of
replaying its producer.

For each MH-1 interaction, the optimized retained reverse schedule keeps two
types of node-indexed primal tensor. The widths below are those of the
qualified checkpoint; other generated model contracts may differ.

| Retained primal | Per-layer width | Forward definition | Why reverse needs it | Two-layer FP32 cost |
| --- | ---: | --- | --- | ---: |
| `pre_gate` | 9,728 floats | `residual + linear_1(message) / normalization` | Re-evaluate the gate and compute its derivative without replaying residual and `linear_1` | `77,824 B/atom` |
| `interaction_output` | 8,192 floats | `linear_2(gate(pre_gate))` | Evaluate the correlation-three product reverse | `65,536 B/atom` |
| **Total** | **17,920 floats/layer** | | | **`143,360 B/atom`** |

The previous MH-1 `recompute-v1` policy discarded both tensors and replayed
their producers during reverse. That reduced memory but increased the matched
CUDA time from `6.659778` to about `11.285 us/atom`. This is different from a
cheap scalar or spline-interval reconstruction: it repeats substantial
equivariant node work and its launch schedule.

The `reuse-adjoints-v1` policy instead retains both expensive primals and
removes a different allocation. Forward messages are stored in
`execution_output_ir_mul`. Their widths are 2,048 floats in layer 0 and 5,120
floats in layer 1. The old cross-layer message-adjoint scratch used the larger
5,120-float width. During reverse, each message row has this lifetime:

```text
read original message for the density derivative
  -> accumulate message * message_adjoint
  -> original message is dead
  -> overwrite the row with message_adjoint
```

The density contribution must be consumed before the overwrite. MH-1 fuses
that dot product into grouped message reverse, accumulates the density-bias
term in the existing pre-gate scaling kernel, and leaves only a scalar density
finalize kernel. The forward-message allocation then also serves as the
message-adjoint allocation, saving `20,480 B/atom` without replaying the
pre-gate or interaction producers.

The resulting node-state slopes for the qualified FP32 checkpoint are:

```text
recompute-v1       76,304 B/atom
reuse-adjoints-v1 199,184 B/atom
full retention    219,664 B/atom

reuse = recompute + 143,360 retained primals - 20,480 aliased adjoints
```

At 864 atoms, reuse reduces node workspace from `337,979,544 B` to
`320,284,824 B` relative to full retention. Matched fresh-process means are
`6.659778 us/atom` for full retention and `6.669469 us/atom` for reuse, a
`0.009691 us/atom` or `0.15%` gap. Matched traces contain 59 launches for both
policies; the device-total gap is only `0.002963 us/atom`.

This gives a general lifetime-reuse procedure:

1. Identify every reverse-required primal and its final reverse read.
2. Retain primals whose producer replay is materially expensive.
3. Find a forward buffer whose contents die before an adjoint needs storage.
4. Move or fuse every final primal read before the first destructive write.
5. Alias only after proving that the ownership and ordering are race-free.
6. Encode the policy in generated metadata and validate its required retained
   extents in the native loader.
7. Advance the JIT generation whenever generated code or its consumer changes.
8. Accept the policy only after full-output correctness, synchronized timing,
   launch attribution, and allocation measurements all pass.

The key distinction is between *recomputation* and *lifetime reuse*.
Recomputation trades execution for memory by discarding a primal. Lifetime
reuse retains an expensive primal and removes storage whose old value is
already dead. The latter is preferable when reverse replay is a material
kernel owner.

### 7.5 Treat state policies and schedules as a compatibility matrix

A state policy is not itself a kernel schedule. Do not map a new policy to a
whole conservative schedule merely because it discards one of the primals used
by an optimized path. State the primal requirements of every optimized phase,
then select the compatible schedule per phase or qualify an explicit
policy-to-schedule mapping.

For example, MH-1 `retain-interaction-v1` retains `interaction_output` but not
`pre_gate`. It can use the retained product, readout, linear-up, and grouped
linear schedules, while replaying only the pre-gate producer before Gate
reverse. Routing it to the full-recompute schedule repeated the conservative
node topology and hid most of the benefit of its retained interaction state.
The relevant distinction is not the policy name but whether each phase needs
`interaction_output`, `pre_gate`, or neither.

This selection happens while generating and loading the RTC module. It is not
a runtime fallback: zero fallback evaluations only prove that the selected
artifact ran, not that the policy inherited all compatible optimizations.

For Float32 CUDA MH-1, `low_memory=True` selects `retain-interaction-v1` and
the bounded `capacity-v1` node arena before RTC compilation. The module then
uses the retained CUDA schedule, replaying only pre-gate state before Gate
reverse. On the 864-atom AlN throughput contract, the same hybrid state with
the 1,024-row throughput arena measured `7.172 us/atom` across two fresh
processes, versus `10.114 us/atom` with the conservative recompute schedule;
both use `253,044,888 B` of node workspace. The public low-memory bundle uses
the 256-row capacity arena, so that throughput figure is not its capacity-mode
performance claim. HIP retains its conservative schedule pending separate
qualification.

When adding either a new state policy or a new optimized path:

1. Record each optimized phase's exact primal and aliasing prerequisites.
2. Build a policy-by-phase compatibility matrix, including CUDA, HIP, and any
   deliberately conservative backend rows.
3. Review surrounding selectors: state-policy dispatch, executor selection,
   arena policy, reverse schedule, module identity, and loader validation.
   A new optimized path can make an older policy-to-schedule mapping obsolete.
4. Include the selected schedule in generated metadata and RTC cache identity;
   do not permit an artifact tuned for one mapping to cross-load for another.
5. Test the emitted launch plan for every supported policy, not only the new
   policy's numerical outputs. Assert that each policy either selects the
   intended optimized stages or reports an explicit unsupported reason.
6. Benchmark the new policy against both its prior conservative route and the
   closest compatible optimized route using the same prepared graph and
   synchronized timing contract.

## 8. Runtime specialization and cache safety

MACE checkpoints vary in channel counts, irreps blocks, sparse tensor-product
paths, product coefficients, readout widths, and radial layouts. Hard-coding
the MH-1 shapes into a global CUDA kernel would make the optimization fragile.

Instead, extract a numerical execution contract and generate kernels specialized
for that contract. The generated artifact should encode:

- exact model dimensions and sparse paths;
- precision and scalar ABI;
- spline layout and generation version;
- node schedule and allowed launch geometries;
- backend and GPU architecture target;
- retention/recompute policy where it changes the program;
- runtime/native packet ABI version.

All of these must participate in cache identity. Required-JIT qualification
must reject an incompatible artifact rather than silently falling back.

The native module loader should admit nonstandard launch dimensions only for
the exact named kernel families that were generated and validated. A broad
allowlist for arbitrary 64-, 128-, or 1,024-row geometries weakens both safety
and the evidence behind the tuning.

## 9. Qualification workflow for another MACE model

### Step 1: Freeze the workload contract

Record the checkpoint and extracted-model hashes, selected head, structure,
composition, atom count, directed-edge count, model cutoff, neighbor skin,
effective cutoff, outputs, backend, GPU target, precision, extension hash, JIT
policy, warmups, and sample count.

Use the same prepared graph for baseline and candidate. Changing the skin or
cutoff while tuning changes edge count and invalidates attribution.

### Step 2: Establish a radial-MLP reference

Run the exact extracted radial path and retain it as a numerical oracle. Use
Nsight Systems to verify whether conditioner, density, basis, or radial-network
kernels are material owners on the target workload.

### Step 3: Prove spline eligibility and build tables

For every interaction, identify the complete subgraph whose only dynamic input
is distance and whose other inputs are the ordered species pair. Reject or
narrow the optimization if runtime node state enters that subgraph.

Generate tables from the extracted oracle. Do not infer hidden widths, radial
basis counts, supported elements, cutoff behavior, or pair symmetry from a
model family name.

### Step 4: Run direct radial convergence

Use the maintained convergence driver:

```bash
python benchmarks/radial_spline_convergence.py compact-model.json \
  --spline-points 64,128,256,512,1024 \
  --output spline-convergence.json
```

Supplying the source checkpoint and a representative structure adds an
end-to-end comparison:

```bash
python benchmarks/radial_spline_convergence.py compact-model.json \
  --checkpoint mace-checkpoint.model \
  --head HEAD \
  --structure representative.extxyz \
  --output spline-convergence.json
```

Test spline nodes, interval interiors, the shortest physically relevant
separations, and immediately below, at, and above the cutoff. The default 256
points are a starting value, not a guarantee. In an earlier MH-0 calibration,
128 points failed while 256 and 512 passed over the qualified physical range.

### Step 5: Qualify derivatives and complete outputs

Require checks for:

- `(a,b)` and `(b,a)` table independence;
- direct values and radial derivatives against the oracle;
- total and node energies;
- forces and stress;
- finite-difference coordinate derivatives exercising both `C_prime` and
  `D_prime`;
- multiple compositions and perturbed structures;
- repeated prepared-graph evaluation;
- exact executor/artifact selection and zero fallback;
- absence of steady-state conditioner and density-MLP launches.

Short-distance spline errors can be much larger than errors in the physical
training domain. State the qualified minimum distance and use representative
structures; do not hide an interpolation failure by testing only equilibrium
energy.

### Step 6: Benchmark only warmed steady state

Exclude model extraction, graph construction, spline-table construction, RTC
compilation, and module loading unless one of those is the explicit target.
Warm all of them first. Report `us/atom` as the primary metric and total step
time as optional supporting information.

Use fresh, backend-specific builds and task-specific JIT caches. Confirm the
actual imported package, native extension, Kokkos execution space, binary hash,
selected executor, artifact generation, and fallback count before accepting a
number.

### Step 7: Re-profile and rank the new bottleneck

Use Nsight Systems first. Compare canonical R0/M0/R1/M1 forward and reverse
owners between the radial-MLP and spline paths. The intended signature is that
conditioning launches disappear and R-stage time falls without a compensating
increase in M-stage work.

Use Nsight Compute only on the largest remaining kernel families and with
narrow launch filters. Profiler replay time is not an end-to-end performance
result.

### Step 8: Specialize M0/M1 conservatively

Generate model-specific kernels, group independent sibling blocks, preserve
producer/consumer boundaries, and retain shared-weight reuse across nodes.
Screen one family and one geometry change at a time. Keep rejected candidates
out of the production selector, but retain their measurements so the same
failed design is not repeated on the same architecture.

## 10. Common failure modes

| Failure | Symptom | Prevention |
| --- | --- | --- |
| Treating pairs as symmetric | Composition-dependent energy/force errors | Index ordered source/target pairs |
| Applying cutoff twice | Large error near the cutoff | Honor extracted `apply_cutoff` exactly |
| Tabulating an intermediate output | Bias-like energy error and wrong derivatives | Include the complete final radial affine output |
| Omitting spline derivatives | Energy passes, forces/stress fail | Store and use analytic interpolant derivatives |
| Materializing all edge/path weights | Very large graph workspace and slower execution | Evaluate compact spline coefficients on demand |
| Optimizing launch count alone | Fewer launches but severe regression | Preserve cross-node weight reuse and measure wall time |
| Copying one tile width everywhere | One layer improves while another regresses | Tune per family, layer, and direction |
| Discarding expensive reverse primals | Low memory but large reverse replay cost | Retain costly primals and target dead storage instead |
| Overwriting a primal too early | Correct energy but wrong forces or density derivatives | Fuse every final primal read before the destructive adjoint write |
| Reusing stale RTC artifacts | Misleading timing or wrong code path | Put model, ABI, generation, schedule, and target in cache identity |
| Benchmarking first-call latency | Compilation dominates results | Warm compilation and module loading before timing |
| Comparing different graphs | Apparent speedup caused by edge-count change | Record cutoff, skin, effective cutoff, and directed edges |

## 11. Practical acceptance checklist

A spline-backed optimized path is ready for production only when all of the
following are true:

- The extracted contract proves that the tabulated subgraph depends only on
  interaction, ordered pair, and distance.
- Cutoff placement, final biases, pair conditioning, and all dimensions match
  the checkpoint.
- Spline values and derivatives pass convergence gates over a stated physical
  radius range.
- Energy, node energies, forces, and stress pass end-to-end and finite-difference
  checks on representative structures.
- Runtime metadata proves the intended spline executor and exact RTC artifact.
- Steady-state traces contain no radial conditioner or density-MLP launches.
- Baseline and candidate use identical atoms, edges, cutoff, skin, precision,
  outputs, retention policy, and warmup protocol.
- Each M0/M1 kernel-shape change has family-local attribution and resource data,
  not only a launch-count argument.
- Every destructively reused tensor has a documented last primal read, first
  adjoint write, and race-free ownership rule.
- Retention and reuse metadata agree with native allocation extents, and the
  loader rejects incompatible combinations.
- Memory growth from wider arenas or retained intermediates is measured and
  documented.
- Unsupported model layouts fail closed or select an explicitly qualified
  reference path.

## 12. Repository implementation map

The principal implementation and qualification entry points are:

- `symmetrix/source/symmetrix/extract_mace_data.py`: extract and validate the
  model-specific numerical contract.
- `symmetrix/source/symmetrix/execution_mh1_contract.py`: describe MH-1
  dimensions, sparse operations, layouts, and schedules.
- `symmetrix/source/symmetrix/mh1_jit_codegen.py`: render specialized host and
  device programs and their launch plans.
- `benchmarks/radial_spline_convergence.py`: qualify spline values,
  derivatives, and optional checkpoint-level outputs.
- `benchmarks/standard_mace_streamed_benchmark.py`: run reproducible prepared
  direct benchmarks.
- `docs/radial_spline_qualification.md`: detailed radial convergence protocol.
- `docs/streamed_edge_execution.md`: broader direct-execution architecture.

## 13. Transferable conclusion

The optimization sequence is more important than any individual launch size:

```text
extract the exact model contract
  -> isolate the distance-and-ordered-pair-only radial subgraph
  -> spline its complete values and derivatives
  -> prove full-property equivalence
  -> profile the new stage balance
  -> generate model-specialized node kernels
  -> preserve cross-node parameter reuse
  -> retain expensive reverse primals and alias only provably dead storage
  -> tune each remaining kernel family independently
```

For MH-1, replacing the radial MLP with ordered-pair splines removed the largest
original cost. The later RTC work succeeded because it treated M0/M1 as a
model-specific dependency graph and optimized reuse at its actual block
boundaries. Retained-primal adjoint reuse then reduced memory without paying
the node-replay cost. That combination, rather than spline interpolation,
kernel fusion, or recomputation alone, is the reusable pattern for optimizing
other MACE-family models.
