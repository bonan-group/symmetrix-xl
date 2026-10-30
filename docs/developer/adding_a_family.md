# Adding a Model Family

A *family* is one retained model architecture the frontends serve: MH-0
MACE/MACEField, MH-1 (`MACE_Nonlinear`), MACE Polar, and the NequIP OAM contract. A family
owns its scientific kernels, its contract, its code generator, its artifact
adapters, and its frontend. The shared layer owns only the contracts a family
plugs into, so porting a family is parallel-friendly inside the family files and
mechanical everywhere else.

This page is the landing checklist: the family-side pattern, every shared-file
edit the tree actually requires, the worked example, the qualification gates,
and the evidence conventions.

## The shared contracts a family plugs into

**Model-architecture layer** (`libsymmetrix/source/family_interface.hpp`). The
family compiles a loaded model into a `Program`: a DAG of `OperatorNode`s
carrying contract parameters plus `BufferContract`/`LayoutContract` data, with a
`Program::fingerprint` from `program_fingerprint()`. The family also publishes
`FamilyCapabilities` (including the `target` and `precision` the bindings
report), `ArtifactRequest`/`ArtifactReadiness`, and `FamilyDiagnostics`. The
compile-time gate is the `FamilyEvaluator` concept and its `TieredFamilyEvaluator`
refinement, asserted in the family's own translation unit — for example
`static_assert(symmetrix::execution::TieredFamilyEvaluator<NequIPOAMKokkos<float>>)`
in `libsymmetrix/source/oam_kokkos.cpp`.

**Execution layer** (`libsymmetrix/source/implementation_registry.hpp`). The
family declares an `ImplementationRegistry` through
`ImplementationRegistry::declare(ImplementationDescriptor)`, mapping each
`OperatorKind` to tiered implementations
(`reference`/`portable`/`frozen`/`generated`). `plan_program()` binds the
highest qualified tier per node and returns an `ImplementationBindingReport`.
Admission is `family_backend_admits()`: a host backend needs
`has_portable_coverage()` below the generated tier, an accelerator backend may
require a ready artifact. A tier the backend cannot serve is reported false, not
degraded silently.

The report is the execution path's authority rather than a parallel
recomputation of it. Each binding carries `selected_implementation_id`
(`implementation_registry.hpp:125`), which `plan_program()` seeds from the
implementation the plan bound; a family whose dispatch adds a gate the readiness
flags do not carry — a module that is loaded but not in use, a profile that is
not selected — supplies the answer through
`set_selected_implementation_resolver`, and the registry only records it. A
family that registers no resolver reports the id its declaration names, and the
field is empty when nothing was bound. The per-evaluator binding report and the
whole-plan report serialize the native field verbatim from every binding
module, so a consumer reads what ran instead of deriving it from the tier.

**Python frontend registry** (`symmetrix/source/symmetrix/family_registry.py`).
The family declares one `FamilyEntry` in a module whose basename ends in
`_family` and which exports `FAMILY_ENTRY`. `_discover_entries()` scans the
package with `pkgutil` for `DECLARATION_SUFFIX`/`ENTRY_ATTRIBUTE`, so no shared
list enumerates families: a new declaration module is registered by existing.
`FamilyRegistry.__init__` rejects duplicate keys, duplicate precedences, a
routing set other than `ROUTING_KEYS = ("direct", "reference")`, and a routing
naming an undeclared `EvaluatorSurface`.

The compact frontend's accepted spellings are declared here too, as
`COMPACT_MODE_VOCABULARY` (`family_registry.py:209`): one table beside the
`FamilyEntry` data model every declaration is written against, because both
compact families resolve it and neither may own the other's vocabulary. A
compact declaration references it through its `modes` field; the compact
calculator binds the same object as `_MODE_VOCABULARY` (`calculator.py:74`).

**Frontend adapter and whole-model surface.** A family whose science is more
than the neutral prepared lifecycle declares `frontend_adapter` and a
`whole_model` `EvaluatorSurface`. The adapter may prepare compact topology,
geometry, and graph scalars, but the family evaluator owns one native
scientific transaction and returns only compact public results. Retained
features, product seeds, solver tables, and adjoints do not cross the frontend
boundary. MACE Polar is the reference pattern: `MacePolarKokkosT` and
`MaceNonlinearKokkosT` are public sibling evaluators over the non-public
`MaceBackboneKokkosT`; neither public family invokes the other.

Checkpoint conversion and production execution are separate contracts. An
extractor may depend on Python and an upstream framework to produce the
family's JSON. Loading and evaluating that JSON must be complete in the native
family evaluator so Python, a compact C++ consumer, and LAMMPS all execute the
same scientific program.

## Family-side pattern

Read `symmetrix/source/symmetrix/mace_family.py`,
`mace_nonlinear_family.py`, `mace_polar_family.py`, and `oam_family.py` before
writing yours; each is a declaration-only module of the same shape.

`FamilyEntry` carries `key`, `precedence`, `detection`, `evaluators`, `modes`,
`schema`, `extractor`, `contract`, `generated_contract`, `codegen`, `adapters`,
`capabilities`, `frontend_adapter`, `routings`, `variants`, `surfaces`, and
`artifacts`:

- `Detection(model_types=..., architectures=...)` — the model-file keys that
  admit a model. MACE and MH-1 detect by `model_type`, OAM by the `oam`
  architecture marker.
- `EvaluatorVariant` — exactly one evaluator class per precision
  (`float32`/`float64`); `declared_error` names the build-support failure a
  backend without the class must report before initializing a runtime.
- `Symbol` — a module or module attribute resolved lazily against the declaring
  package, so the entry names its schema, extractor, code generator, and adapter
  modules without importing them at declaration time.
- `CapabilityDeclaration` — the capability family string the native evaluators
  report; its key set is `family_registry.CAPABILITY_KEYS`, one cross-language
  contract read by the conformance suites, the generated matrix, and the
  frontend readiness gate.
- `EvaluatorSurface` — one evaluator capability (`key`, bound `methods`,
  `readiness` capability keys, optional `kokkos` scope). A frontend that commits
  to a surface raises `MissingCapabilityError` naming the missing method instead
  of running a fallback branch.
- `ArtifactAdapter` — one entry point per (routing, target) that prepares and
  caches an artifact. It is called `prepare(model_path, precision=..., target=...)`
  and declares which result attributes hold the library
  (`artifact_path`), the identity envelope (`manifest_path`), and which
  evaluator keywords receive the library and identity. A family whose frontend
  prepares inline, or whose adapter takes rendered source rather than a model
  file, declares none; a consumer that needs one then fails naming the routing
  and target.

`ModeVocabulary` is the accepted-spelling table: `aliases`, `primary`, and
`deprecated`. **Every non-direct spelling resolves to the single `reference`
routing** — that is why `materialized` is an alias and not a routing of its own.
Each `Routing(key, native_mode, surface, reduces_stress)` names the neutral
routing (`direct` or `reference`), the spelling its native evaluator accepts,
the surface it drives, and whether it leaves a completed evaluation the stress
reduction can consume. The two compact families share
`family_registry.COMPACT_MODE_VOCABULARY`, declared beside the entry data model
rather than inside one family's declaration, so a compact declaration points at
it instead of owning it; OAM declares its own literal `ModeVocabulary`.

The three R1 artifact adapters share
`symmetrix/source/symmetrix/mace_jit_artifact_common.py`: `read_model_json`,
`native_evaluator`, and `require_matching_generation_version` are the Python
side of the artifact contract, so they live there once. What differs stays with
the adapter and is passed in — its own error type and message label, and the
streamed-edge policy its contract requires (`"direct"` for the compact host
adapter, `"factorized"` for its device sibling).

**Plan space**, when the family has one. `libsymmetrix/source/plan_candidate_space.hpp`
gives `PlanCandidateSpace::declare(id, evaluate)` and `PlanCandidateDescriptor`;
a family registers one descriptor per *whole* plan it can run, in declaration
order. The plan ids are the cross-language string contract — the same ids the
Python calculators accept as `_debug_execution_plan` and the native plan pin
validates — so they are family data with one native owner, never a planner-owned
table. A family that declares no plan space reports its one selected plan as an
implicit candidate.

**Names.** Family files in `libsymmetrix/source/` and
`symmetrix/source/symmetrix/` are prefixed with the family token followed by
`_` or `.`. The prefix is derived from the declared entries
(`FAMILY_PREFIX = _family_prefix(family_registry.declared_entries())` in
`symmetrix/test/test_family_naming_contract.py`), so declaring a family admits
its own files with no edit to that module;
`test_a_newly_declared_family_admits_its_files_without_editing_this_module`
pins exactly that. The token set the shared-layer leak scan greps for
(`FAMILY_TOKEN`) is still hand-maintained, because the scan looks for the
spellings a token takes in prose and not only in basenames. One family never
leaks into the shared layer: `family_interface.hpp`,
`implementation_registry.hpp`, `plan_candidate_space.hpp`, `execution_mode.hpp`,
and `family_registry.py` carry no family token.

## Shared-file edits

A family that ships its own calculator (the OAM pattern) needs items 1–3 and
6–11; joining the compact frontend adds item 4 as well. Items 1–2 gate the
build, items 3, 6, 7, and 10 gate a test or a validation, and items 8, 9, and 11
are the family's own qualification declarations: one new module carries all
three, the shared suites stay green without it, but the family is then not
qualified and its naming contract is not enforced. Items 5, 12, and 13 are
conditional: item 5 applies to a family that exports its own calculator class,
item 12 is specific to the MH-0 plan-id contract, and item 13 applies to
families that add maintained contracts or guides. Nothing else in the shared
layer requires a hand edit: the Python package is installed wholesale
(`wheel.packages = ["source/symmetrix"]` in `symmetrix/pyproject.toml`), the
frontend registry is discovered, not listed, and the conformance suites and the
routing-parity suite read whatever the family's own module registers.

1. **`libsymmetrix/CMakeLists.txt`** — add the family's native translation units
   to `SYMMETRIX_SOURCE_FILES` (the base list, plus the `if(SYMMETRIX_KOKKOS)`
   block for Kokkos-only evaluators). A unit that is not listed does not build.
2. **`symmetrix/CMakeLists.txt`** — add the family's binding translation units
   to `SYMMETRIX_BINDINGS` and/or `SYMMETRIX_BINDINGS_KOKKOS`. The native
   extension exposes the evaluator class the `FamilyEntry` variant names through
   these (for example `source/cpp/oam_kokkos.cpp`).
3. **`symmetrix/test/test_family_registry.py`** — extend the `SHIPPED` tuple
   with `(key, declaration module)`. `test_declared_entries_are_the_shipped_families_in_precedence_order`
   asserts the declared order equals it, so the tuple is the enforcement point
   for both the family set and its precedence order.
4. **Compact frontend integration** — prefer a family-owned
   `frontend_adapter` over model-type scientific branches in
   `calculator.py`. The shared calculator resolves the family, routing,
   evaluator, and declared surfaces; the adapter supplies family-specific
   compact inputs and invokes the native whole-model method. It must not
   retrieve retained features, construct reverse seeds, or compose scientific
   outputs in Python. A family that ships its own calculator class does not
   edit the compact frontend beyond generic adapter resolution.
5. **`symmetrix/source/symmetrix/__init__.py`** — extend `__all__` only when the
   family exports its own calculator class (as `NequIPCalculator` is exported
   today).
6. **`precedence`** — pick a globally unique integer in the family's own
   declaration (`mace_nonlinear_family.py` is `0`, `mace_family.py` is `1`,
   `mace_polar_family.py` is `2`, `oam_family.py` is `3`) and update `SHIPPED`
   order to match. `FamilyRegistry`
   rejects a duplicate precedence, and the first admitter wins an overlapping
   detection key. This is a family-file value, but it is coupled to item 3.
7. **`symmetrix/test/test_family_naming_contract.py`** — extend `FAMILY_TOKEN`,
   the token set the shared-layer leak scan greps for. The prefix that admits
   your files is derived from the declaration and needs no edit; this one does,
   because the scan looks for the spellings your family's name takes in prose.
   The suite stays green without it, but the contract is then unenforced for
   your family, so treat it as required.
8. **`symmetrix/test/test_family_<family>_conformance_cases.py`** — one new
   module beside the shared suites, and the only place your family's names
   appear. It registers one `FamilyConformance` through
   `conformance.registry.register_family(...)`, declaring the `Adapter`
   subclass(es) and `INSTANCE_SPECS` entries for each admitted instance
   (precision, expected operator kinds, layouts, reductions, generated kinds,
   tolerances, and the `lifecycle_tiers` the instance can qualify), the
   `structures` and `build_models` those instances share, the
   `host_compiler_gated_instances` whose generated tier is a host artifact gated
   on a working C++ compiler, and the `readiness_cases` the empty-path contract
   is asserted on. If the shared `Adapter` cannot drive a family-owned
   `whole_model` lifecycle, do not invent an instance: keep `INSTANCE_SPECS`
   empty, document the precise harness gap, and add focused assertions for the
   declared adapter/surface and native admission. Discovery is by
   `FAMILY_MODULE_PATTERN = "test_family_*_conformance_cases.py"`
   (`symmetrix/test/conformance/registry.py`), so neither the shared lifecycle
   suite (`test_family_conformance.py`) nor the registry lists a family.
9. **per-family interface tests, in that same module** — drive the neutral
   surface for each evaluator and assert the additive tier-binding report fields
   (`test_oam_evaluator_satisfies_the_family_interface_without_an_artifact` and
   its MACE/MH-1 siblings are the pattern, and
   `test_selected_implementation_id_names_the_dispatched_implementation` is
   where the binding authority is pinned against the dispatch).
   `test_family_interface_conformance.py` keeps only the contract stated once
   for every family, so it needs no edit.
10. **`tools/generate_execution_support_matrix.py`** — add a `FAMILY_REQUESTS`
    block (the public request rows, whose `mode` values must be canonical modes
    of the declaration) and a `FAMILY_CAPABILITIES` block (one row per declared
    capability key) keyed by the family key. `validate()` raises when the
    declared family set and the table keys differ. Then **regenerate**
    `docs/reference/execution_support_matrix.md`
    (`python tools/generate_execution_support_matrix.py`); the block between
    `BEGIN_MARKER` and `END_MARKER` is drift-pinned by
    `test_execution_support_matrix.py::test_document_equals_a_fresh_generation`
    and by the script's `--check` mode, so it is never hand-edited.
11. **routing parity, declared in that same cases module** — a
    `RoutingParityCase(key, structure, build_model, calculator, ...)` beside the
    family's other declarations. `test_execution_routing_parity.py` parametrizes
    over `ROUTING_CASES` and edits nothing: it is where `direct` and `reference`
    are proven to be the same model evaluated two ways, and that a `reference`
    request reaches the artifact preparer zero times.
12. **`symmetrix/test/test_mace_streamed_edges.py`** — extend the
    `_DECLARED_PLAN_IDS` contract in
    `test_declared_plan_space_is_the_cross_language_plan_id_contract` when the
    family adds MH-0-shaped plan ids, and mirror them in the native
    `space.declare(...)` sites and `symmetrix.calculator._DEBUG_EXECUTION_PLANS`.
    This file carries no family switch — it is MACE-specific — but it is the
    largest shared collision point in the tree, so a family that touches plan
    data will rebase here.
13. **Documentation** — a family adds maintained contracts and guides under
    the appropriate `docs/` section and updates its index navigation.
    Keep raw qualification results outside tracked documentation.
    Documentation and contract changes land together.

## Worked example: the NequIP OAM port

The OAM family is a reference port. Its current execution contract is in
{doc}`/reference/execution_support_matrix` and {doc}`/user/models`. Its
family-side contract, exporter, code generators, adapters, and calculator
provide concrete examples of the surfaces described above.

## Qualification expectations for a new family

- **Declaration and naming contracts** — `pytest symmetrix/test/test_family_registry.py
  symmetrix/test/test_family_naming_contract.py`. These are the fast gates: a
  missing surface, a duplicate precedence, a stray family token in the shared
  layer, or an unprefixed basename fails here first.
- **Conformance** — `symmetrix/test/test_family_conformance.py` drives the
  neutral lifecycle (prepare, bind geometry, evaluate, reduce forces, reduce
  stress where declared) for every instance your cases module registers, and
  asserts tier parity per operator kind and portable coverage per admitted
  instance. The adapters and instance specs are yours, in your own module; the
  shared suite names no family. A surface the runner cannot qualify must be
  reported with the runner it requires, never skipped or represented by a fake
  shared-lifecycle instance.
- **Interface conformance** — the per-evaluator surface assertions live in your
  cases module beside the declarations they drive;
  `symmetrix/test/test_family_interface_conformance.py` holds the one runtime
  contract stated once for every family (an empty artifact path stays a
  readiness query), and the compile-time half is the `static_assert` in the
  family's own translation unit.
- **Routing parity** — the case you declare in that module is what
  `symmetrix/test/test_execution_routing_parity.py` parametrizes over: the two
  routings agree within the family's own tolerance, and `reference` evaluates
  with an empty cache and RTC disabled while `direct` fails at preparation
  rather than falling back.
- **Matrix regeneration with drift check** — regenerate the support matrix and
  confirm the drift test is green (`--check` or
  `test_execution_support_matrix.py`).
- **Backend battery** — run the applicable full suite on each backend the family
  claims, keeping the qualification rules from `AGENTS.md`: CPU, CUDA, HIP, and
  LAMMPS builds live in separate fresh environments and task-specific build
  roots; `symmetrix doctor --json` must report one loaded OpenMP runtime and
  `status: ok` for OpenMP deployment; an accelerator tier is only qualified on a
  backend build compiled for that architecture; LAMMPS requires a relink after
  every native-library change and per-family host/device artifacts. A backend a
  family does not serve is reported unsupported, never silently degraded.
- **Required-test manifests** — when the family owns a node an accelerator
  qualification run must execute, add it to `symmetrix/test/cuda_runtime_required_tests.json`,
  `cuda_compiler_required_tests.json`, or `hip_required_tests.json` so a missing
  node fails the run instead of shrinking it.

## Evidence conventions

Qualification records are session-tagged and never overwritten, and every record
carries the extension SHA-256 it was produced with
(`identity.native_extension_sha256`). Report `us/atom` (`us/atom/step` for MPI)
as the primary metric with model, precision, backend, atom count, cutoff, skin,
effective cutoff, directed edges, timing, and memory; parse the record back and
check it against the command rather than trusting the filename. Keep raw
measurements under `benchmarks/` and prune superseded reports instead of letting
several sessions' numbers coexist. Do not commit benchmark records or generated
binaries to the documentation tree.

## Known friction

The shared-file edits above are mechanical, but they are real: a family cannot
land without them, and the largest shared test file is a rebase collision point.
The five simplifications this page used to carry as planned work have all
landed, and each one removed a hand edit from the checklist:

1. **Binding-report authority.** `selected_implementation_id` is a per-node field
   of the binding report (`implementation_registry.hpp:125`), seeded by
   `plan_program()` and overridable by a family resolver, and every report
   serializes it verbatim — so the report is what ran, not a second derivation,
   and a consumer never recomputes it from the tier.
2. **Serializer hoisting.** `read_model_json`, `native_evaluator`, and
   `require_matching_generation_version` live once in
   `symmetrix/source/symmetrix/mace_jit_artifact_common.py`, parameterized by
   the adapter's error type, label, and streamed-edge policy. A new adapter
   writes only what differs.
3. **Vocabulary relocation.** `COMPACT_MODE_VOCABULARY` is declared in
   `family_registry.py` beside the `FamilyEntry` data model, so no family module
   owns another family's vocabulary and the calculator imports no declaration.
   That removed the declaration import from item 4's edit surface.
4. **Derived `FAMILY_PREFIX`.** The prefix is
   `_family_prefix(family_registry.declared_entries())`, so a new family admits
   its own files with no edit to the naming contract. The former "extend the
   `FAMILY_PREFIX` regex" item is gone from the list; only `FAMILY_TOKEN`
   (item 7) stays hand-maintained.
5. **Conformance-adapter split.** Per-family adapters, instance specs, structures,
   session models, readiness cases, and the routing-parity case are declared in
   the family's own `test_family_<family>_conformance_cases.py` and discovered
   by pattern (`conformance/registry.py`). Items 8, 9, and 11 are therefore one
   new file plus two declarations inside it, and the three shared suites
   (`test_family_conformance.py`, `test_family_interface_conformance.py`,
   `test_execution_routing_parity.py`) need no edit at all.

What is left is friction the tree has chosen not to hide rather than debt:

- **The build graph is explicit.** CMake source lists (items 1–2) cannot be
  globbed here; an unlisted translation unit does not build, and a silent
  glob would make a family's absence from a backend invisible.
- **The shipped set is asserted, not inferred.** `SHIPPED` (item 3) is the
  enforcement point for both the family set and its precedence order, and the
  `precedence` integer (item 6) is a family value coupled to it.
- **The generated matrix is drift-pinned.** Item 10's tables are authored data
  and the regenerated block is checked byte-for-byte, which is what keeps the
  support matrix from becoming a second, drifting source of truth.
- **`test_mace_streamed_edges.py` remains the largest shared collision point**
  (item 12). It carries no family switch, but plan-id data crosses it, so a
  family that adds plan ids rebases there.
- **The leak scan's token set is prose-aware** (item 7): a token a family's name
  takes in a comment or a string is not derivable from the declaration.
