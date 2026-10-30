# Native Architecture

Symmetrix-XL compiles a loaded model into an execution program and then binds
every operator node to a qualified implementation. Two layers are separate, and
both are family-neutral:

- **Model-architecture layer** (`libsymmetrix/source/family_interface.hpp`).
  A family compiles its model into a `Program`: a DAG of `OperatorNode`s carrying
  contract parameters plus `BufferContract`/`LayoutContract` data, with a
  `Program::fingerprint`, a `FamilyCapabilities` set, `ArtifactRequest`/
  `ArtifactReadiness`, and `FamilyDiagnostics`. The `FamilyEvaluator` concept and
  its `TieredFamilyEvaluator` refinement are the compile-time contract each
  retained evaluator asserts in its own translation unit.
- **Execution layer** (`libsymmetrix/source/implementation_registry.hpp`).
  Each family declares an `ImplementationRegistry` per backend, mapping an
  `OperatorKind` to tiered implementations — `reference` (T0), `portable` (T1),
  `frozen` (T2), and `generated` (T3). `plan_program()` binds the highest
  qualified tier per node; `family_backend_admits()` admits a host program only
  with portable coverage below the generated tier, while an accelerator backend
  may require a ready generated artifact and fails at graph preparation instead
  of degrading silently.

Execution uses the prepared factorized/streamed-edge path: eligible radial and
equivariant contractions are specialized for the model contract while graph
shape and learned weights stay runtime values, and whole-plan candidates are
family-declared data (`libsymmetrix/source/plan_candidate_space.hpp`). Every
family declares exactly two routings: `direct` (the generated tier against a
contract-bound artifact) and `reference` (single-thread execution of the generic
implementation on the default host execution space, artifact-free). The retained
MACE pipeline inside that framework keeps its radial stages R0/R1, message
stages M0/M1, dense transforms, readouts, and device force/stress reductions,
with generated artifacts specializing eligible R0/M0/R1 contracts.

Families plug in through their own modules: a Python declaration in a
`*_family.py` module beside their frontend, plus the family's native evaluator,
contract, code generator, and artifact adapters. No shared scientific kernel
carries a family token.

- Execution planning, tier binding, and the plan report:
  {doc}`execution_planner`
- Execution model, public spellings, artifacts, and backend invariants:
  {doc}`/streamed_edge_execution`
- Native source ownership map: {doc}`/mace_kokkos_source_layout`
- Landing a new family: {doc}`adding_a_family`

## Additive correction architecture

Dispersion is an additive pipeline beside model execution, not a model family
or an operator tier. Immutable `D2Correction` and `D3Correction`
configurations select concrete native runtimes. Each runtime declares a
`GraphRequirement` (pair cutoff, optional coordination cutoff, skin, and
topology kind; the pair cutoff is absent for a `global` interaction, which
declares only its short supporting graph), owns its graph cache, computes only
requested ASE quantities, and returns values to the family-neutral
`CorrectionPipeline`. The pipeline validates shapes and finiteness and composes
energy/free energy, per-atom energies, forces, and stress without knowing D3
coordination equations.

The topology requirements of existing and prospective methods are materially
different:

| Method | Required runtime state |
|---|---|
| D2 | pair graph |
| two-body D3 | pair graph plus coordination-number node state |
| D3 ATM | triplet topology |
| D4 | charge solver plus pair/triplet topology |
| MBD | global coupled-dipole linear algebra |

Adding another correction configuration requires a runtime factory and an
honest `GraphRequirement`; it does not require a branch in
`CorrectionPipeline`. Pair, triplet, and global implementations should remain
concrete native evaluators so device kernels do not use C++ virtual dispatch.
D3 ATM, D4, and MBD are currently unsupported rather than approximated through
the pair-only evaluator.

D2 and D3 use correction-owned, grow-only Kokkos graph storage and the neutral
neighbor reuse policy shared with model graphs. Keeping ownership separate is
load-bearing: the dispersion cutoff can be much longer than the model cutoff,
and D3 coordination state has a distinct membership cutoff. Fully periodic,
nonsingular cells can build the correction graph in Kokkos memory; partial or
nonperiodic ASE cells use the neutral host builder and upload topology. Runtime
failure on the selected device is not a host fallback.

```{toctree}
:hidden:

/mace_kokkos_source_layout
```
