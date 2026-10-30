# Execution Planner

The public defaults are `streamed_edges="direct"` and
`execution_profile="capacity"`, with `allow_fixed_workspace=False`. The Python
frontend owns artifact discovery and compatibility diagnostics; native code
provides contract validation, backend capability, graph-size and memory
information. Fixed-workspace candidates require explicit frontend opt-in.
The current detailed graph-time plan report applies to MH-0 MACE/MACEField.

The candidate space is family-declared data. Each family registers one
descriptor per whole plan it can run -- the stable plan id, the qualification
predicate, and the byte estimator -- in declaration order, and the report
consumes that list instead of a planner-owned table. MH-0 registers its five
direct plans: `mh0-direct-speed`, `mh0-direct-capacity-y-only`,
`mh0-direct-capacity-retained`, `mh0-single-layer-tiled-v1`, and
`mh0-dual-layer-tiled-v1`. Those ids are the same cross-language string
contract the Python calculators accept as `_debug_execution_plan`, and the
native plan pin validates a requested id against the declared list. A family
that declares no plan space reports the one plan it selected as a single
implicit candidate, with no alternatives to qualify.


## Tier binding

Two layers are separate. The model-architecture layer compiles a loaded model
into a program: a DAG of operator nodes carrying contract parameters and
buffer/layout requirements as pure versioned data
(`libsymmetrix/source/family_interface.hpp`). The execution layer
(`libsymmetrix/source/implementation_registry.hpp`) maps an operator kind to
tiered implementations and binds the highest qualified tier per node.

The tiers are `reference` (T0, single-thread execution of the generic
implementation on the default host execution space, which defines contract
numerics), `portable` (T1, the generic Kokkos implementation, the host-first
correctness backbone), `frozen` (T2, the legacy built-in module topology, still
admitted for the single-layer contracts that ship it and never created anew),
and `generated` (T3, the JIT/RTC artifact). A generated implementation only
qualifies when its artifact is ready, so a direct request without one fails at
graph-preparation time rather than degrading silently.

### Frozen built-in module retirement: performance-gated

The generated R0 pipeline now serves the qualified host, CUDA, and HIP
contracts through one canonical identity and dispatch contract. Installed,
cache-hit, and newly compiled artifacts are physical origins of that same T3
implementation. Exact installed R0 takes precedence over the frozen T2 route;
unsupported but valid shapes continue through runtime generation. The public
R0 report intentionally collapses physical loader names to `generic` or
`generated`, while exposing backend, origin, schedule, profile, and artifact ID
as independent fields.

The frozen `mace_standard_r0` implementation nevertheless remains as the
shadow-qualification oracle because the deletion gates have not all passed.
For an eligible low-memory M0/R0 node, this temporary path tries an exact
installed artifact when the package supplies a candidate, but an installed
miss or rejection may bind the frozen built-in before cache lookup or runtime
generation. If the package mode supplies no candidate for the stage, the
built-in may bind immediately. Thus installed-to-cache-to-runtime is the normal
generated resolver order, not a universal claim while the frozen oracle is
admitted.
CPU passed 48/48 correctness and route cases and its aggregate performance
gate, but one case measured `1.054858` generated/frozen against a `1.05`
per-case ceiling. After porting the frozen CUDA schedules, the kernel-change
campaign passed all 48 CUDA performance cases with a `0.983760` aggregate ratio
and a `0.996812` worst case. Two exact-source confirmation campaigns after the
subsequent identity/catalog-only change both passed the aggregate gate
(`0.983635` and `0.985595`) but each failed a different single per-case gate
(`1.184456` and `1.064418`); raw samples show warm-up/timing-transition outliers
consistent with cold-clock behavior. A later exact-current generation-17 run
closed the CUDA gate: all 48 cases passed, aggregate generated/frozen was
`0.982764`, FP32 was `0.956079`, FP64 was `0.991949`, and the worst case was
`0.996605`. HIP generated artifacts compile and package, but no executable
device result is accepted because the qualification user could not access
`/dev/kfd`. These are release blockers, not an invitation to waive the
threshold. `mace_standard_r0.hpp`, admission, and frozen dispatch stay until
the CPU and HIP gates also pass stable recorded campaigns.

Other frozen modules have separate coverage and dependency constraints. In
particular, `mace_standard_m0.hpp` still owns term data consumed by the neutral
product-basis plan. R0 artifact-pipeline completion does not claim retirement
of M0/M1 or full generated-only execution.

Admission is per family-backend: a host backend admits a node only when a
qualified implementation below the generated tier serves it, while an
accelerator backend may serve nodes with generated implementations only and
then requires the artifact those implementations were qualified against.
Capabilities are declared honestly per backend, and a capability a backend
cannot serve is reported false instead of silently degraded.

### Generation-19 MH-1 artifact boundary

The current generated host contract is **ABI v7**. It is count-agnostic: the
descriptor carries tables sized to the validated contract's
`interaction_count`, rather than selecting an ABI from a compiled-in layer
bound. ABI v7 is also the composed-product boundary. The model-specific plugin
owns the nonlinear pre/post phases, while common M0/M1 artifacts own the
contraction coefficients and forward/reverse contractions. Their buffers are
disjoint in the MH-1 composition. Device M0 uses `table` revision 1, `chunk32`
revision 2, and experimental `tiled128x4-v1` revision 3 with capability mask
`0x1b`; device M1 uses revisions 2, 3, and 4 respectively with capability mask
`0x0b`. Host M1 `chunk32` remains revision 2 with the host-ABI `0x1b` mask. No
device schedule advertises alias safety: exact input/output or
input/input-adjoint aliases are rejected before launch.

The current model-specific device module is **ABI v5**, paired with launch-plan
schema `/6`, plan version 6, and JIT generation 19. Common device M0/M1 modules
retain their separately versioned operator ABI. Host ABIs v3-v6 are historical
or private compatibility layouts: the generation-19 public loaders fail closed
for v5 and v6, and the frontend prepares v7 only. They must not be described as
alternative live routes or selected from `interaction_count`.

The plan report gained two additive fields, `binding_report_version` and
`bindings`. Version 1 always accompanies a binding list, and version 0 with an
empty list is the pre-binding state a report carries when the prepared direct
path is not the active mode. Each retained evaluator also exposes its compiled
program, its binding report, its capability set, its artifact readiness, and
its diagnostics through the Python bindings (`family_program`,
`family_binding_report`, `family_capabilities`, `family_artifact_readiness`,
`family_diagnostics`). Every pre-existing report field is unchanged.

For standard compact MH-0 models, `speed` is a capability-selected throughput
bundle rather than a synonym for retaining all intermediate state. It retains
harmonic gradients for small graphs and switches to Y-only harmonic storage at
40,960 directed edges when qualified on CPU or CUDA FP32. CUDA FP64 and HIP
retain harmonic gradients. Both variants reuse eligible MH-0 state and
recompute readout state. FP32 uses compact geometry; FP64 retains Cartesian
geometry. Capacity selection uses the same throughput candidate first and may
also select Y-only storage when the estimated device-memory budget requires it.
Debug plan identifiers remain exact even on hosts where no device-memory query
exists.

The public direct-execution matrix is maintained in
{doc}`/reference/execution_support_matrix`; architecture and planner invariants
are described in {doc}`/streamed_edge_execution`. Exact debug-plan identifiers
remain private implementation details in `symmetrix.calculator`.

```{toctree}
:hidden:

/streamed_edge_execution
```
