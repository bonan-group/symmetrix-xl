"""Cross-family conformance: one adapter lifecycle, every registered family.

The family adapter (``libsymmetrix/source/family_interface.hpp``) promises one
contract for every retained model family.  This module drives that contract for
every family that declares itself to ``conformance.registry`` and proves four
independent things per admitted instance:

* **Lifecycle conformance.**  ``test_lifecycle_matches_the_reference`` drives the
  prepared-graph lifecycle (prepare graph, bind geometry, evaluate, reduce
  forces, reduce stress where declared), asserts the generation counters, and
  then reuses the graph token across a displacement that stays inside the
  neighbor skin.
* **Reference evaluation without generated code.**
  ``test_reference_specialization_runs_without_artifacts`` runs the whole
  program through the compiler-free path with the JIT policy disabled and no
  artifact at all, which is the host admission invariant.
* **Tier parity per operator kind.**  ``test_tier_parity_per_operator_kind``
  asserts every portable implementation, and every generated implementation this
  host can prepare, against the reference.  A surface this runner cannot qualify
  is reported with the runner it requires instead of being skipped.
* **Contract-instance portability.**
  ``test_host_portable_coverage_per_contract_instance`` asserts coverage per
  admitted instance -- one qualified binding per operator node with that node's
  declared buffer contracts (shape, layout, dtype), the declared
  compute/accumulator precision, and the declared derivative requirements --
  rather than once per operator kind.

The whole contract is family-neutral and lives one level down:
``conformance.contract`` holds the graph convention (one builder for every
family: directed edges indexed by receiver, ``sources`` holding the neighbor
node, and ``xyz`` the minimum-image displacement), the reference reduction and
finite-strain oracles, and the ``Adapter`` base class that maps the neutral
lifecycle onto an evaluator.  Each family's structures, session models, adapter
classes, instance specs, and interface cases live in that family's own
``test_family_<family>_conformance_cases`` module, which
``conformance.registry`` discovers by name.  This module names no family: it
drives whatever the registry collected, so a new family is one new module and no
edit here.

Scope: this module is focused on the adapter surface.  It drives the per-family
numerical suites through the neutral lifecycle instead of restating them.
"""

from __future__ import annotations

import gc
import shutil

import numpy as np
import pytest

from conformance.contract import (
    ACCELERATOR_RUNNER,
    ACCELERATOR_SURFACES,
    CXX_RUNNER,
    LAYOUT_ORDERS,
    PORTABLE_TIERS,
    Unqualified,
    _diagnostic,
    _directed_force_reduction,
    _finite_strain_oracle,
    _voigt,
    native_symmetrix,
)
from conformance.registry import (
    ADAPTER_SPECS,
    FIELD_RESPONSE_INSTANCES,
    HOST_COMPILER_GATED_INSTANCES,
    INSTANCES,
    LIFECYCLE_CASES,
    build_models,
)
from conformance.surface import _assert_interface_surface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def conformance_kokkos_runtime():
    """One Kokkos initialization per session, before any adapter is built.

    Artifact preparation helpers own and finalize the Kokkos runtime when they
    find it uninitialized, so the suite must initialize it exactly once and keep
    the ownership: otherwise a prepared artifact leaves a finalized runtime
    behind and the next evaluator aborts.
    """
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    return native_symmetrix


@pytest.fixture(scope="session")
def conformance_models(tmp_path_factory, conformance_kokkos_runtime):
    """Session models and the shared artifact cache every instance prepares in.

    Each registered family builds its own session models, so this fixture names
    no family and no model.
    """
    root = tmp_path_factory.mktemp("family-conformance")
    models = {"jit_cache": root / "jit-cache"}
    models.update(build_models(root))
    return models


@pytest.fixture(scope="session")
def adapters(conformance_models):
    """Materialize every instance once, and release the native owners at exit.

    The session Kokkos finalizer in ``conftest`` runs after this teardown, so
    the adapters must drop their evaluator references first: a session-scoped
    native owner otherwise blocks finalization.
    """
    built = {}
    for key, spec in ADAPTER_SPECS.items():
        fields = dict(spec)
        adapter_class = fields.pop("adapter")
        contract_key = fields.pop("contract_key", None)
        adapter = adapter_class(key=key, family=adapter_class.family, **fields)
        if contract_key is not None:
            adapter.contract_key = contract_key
        adapter._models = conformance_models
        built[key] = adapter
    yield built
    for adapter in built.values():
        adapter._evaluators.clear()
    built.clear()
    gc.collect()


@pytest.fixture(params=INSTANCES)
def instance(request, adapters):
    return adapters[request.param]


@pytest.fixture(params=LIFECYCLE_CASES, ids=[f"{k}-{t}" for k, t in LIFECYCLE_CASES])
def lifecycle_case(request, adapters):
    key, tier = request.param
    return adapters[key], tier


# ---------------------------------------------------------------------------
# 1. Lifecycle conformance
# ---------------------------------------------------------------------------


def test_lifecycle_matches_the_reference(lifecycle_case):
    """The seven neutral lifecycle steps agree with the T0 oracle.

    Covered per admitted instance: the prepared graph is reused across a
    within-skin displacement (the generation token must not move), the forced
    reduction is the reduction of the reference's own directed forces, and the
    declared stress reduction is symmetric.
    """
    adapter, tier = lifecycle_case
    atoms = adapter.structure()
    result = adapter.run_lifecycle(tier, atoms)
    reference = adapter.evaluate_reference(result["graph"])

    assert result["graph"].num_nodes == len(atoms)
    assert len(result["directed_forces"]) == 3 * len(result["graph"].sources)
    # The directed forces are the T0 oracle's own per-edge array on the same
    # edges, so the comparison is element-wise and order-sensitive.
    np.testing.assert_allclose(
        result["energies"],
        reference["energies"],
        rtol=adapter.rtol,
        atol=adapter.atol,
        err_msg=f"{adapter.key}/{tier} node energies",
    )
    np.testing.assert_allclose(
        result["directed_forces"],
        reference["directed_forces"],
        rtol=adapter.rtol,
        atol=adapter.atol,
        err_msg=f"{adapter.key}/{tier} directed forces",
    )
    # The lifecycle reduction must reproduce the reference reduction.
    np.testing.assert_allclose(
        result["atom_forces"],
        _directed_force_reduction(result["graph"], reference["directed_forces"]),
        rtol=max(adapter.rtol, 1e-9),
        atol=max(adapter.atol, 1e-9),
        err_msg=f"{adapter.key}/{tier} reduced atom forces",
    )
    if reference["field_adjoint"] is not None:
        np.testing.assert_allclose(
            np.asarray(result["field_adjoint"], dtype=np.float64).reshape(-1),
            adapter.expected_field_adjoint(
                tier, result["graph"], reference["field_adjoint"]
            ),
            rtol=adapter.rtol,
            atol=adapter.atol,
            err_msg=f"{adapter.key}/{tier} electric-field adjoint",
        )

    # The declared stress reduction, where declared: the program's own reduction
    # node or the evaluator's own reduction must produce the finite-strain
    # derivative of the family's own reference energy.  No reference entry point
    # produces stress, so that oracle is the only reference available for the
    # reduction itself.
    assert (result["stress"] is not None) == adapter.serves_stress
    if result["stress"] is not None:
        stress = result["stress"]
        assert stress.shape == (3, 3)
        assert np.abs(stress).max() > 0.0
        # The physical stress is the symmetric part: the derivative of the
        # energy with respect to a symmetric strain.  Comparing that part keeps
        # the assertion on the quantity the oracle defines.
        np.testing.assert_allclose(
            _voigt(0.5 * (stress + stress.T)),
            _finite_strain_oracle(
                adapter.reference_energy, atoms, step=adapter.strain_step
            ),
            rtol=adapter.stress_rtol,
            atol=adapter.stress_atol,
            err_msg=f"{adapter.key}/{tier} reduced stress against its oracle",
        )
        assert (
            np.abs(stress - stress.T).max() <= adapter.stress_antisymmetry_atol
        ), f"{adapter.key}/{tier} antisymmetric stress part exceeds its bound"

    # --- prepared-graph reuse across a within-skin displacement -------------
    moved_atoms = atoms.copy()
    moved_atoms.positions = np.asarray(moved_atoms.positions) + np.asarray(
        [2.0e-3, -1.0e-3, 1.5e-3]
    )
    reusable = result["graph"].reused_at(
        moved_atoms, result["evaluator"].atomic_numbers, result["evaluator"].r_cut
    )
    # The displaced structure must keep the prepared edge list bit for bit, or
    # the reuse claim is vacuous.
    for moved_array, original_array in zip(
        reusable.topology()[1:], result["graph"].topology()[1:]
    ):
        np.testing.assert_array_equal(moved_array, original_array)
    assert not np.array_equal(reusable.positions, result["graph"].positions)
    assert float(np.abs(reusable.xyz - result["graph"].xyz).max()) > 0.0

    counters = adapter.prepared_counters(result["evaluator"])
    before = (
        {name: int(getattr(result["evaluator"], name)) for name in counters}
        if counters
        else {}
    )
    adapter.compute_prepared(result["evaluator"], result["generation"], reusable)
    assert adapter.accepted_generation(result["evaluator"]) == result["generation"]
    geometry_generation = adapter.geometry_generation_of(result["evaluator"])
    if geometry_generation is not None:
        assert geometry_generation == result["generation"]
    if counters:
        prepare_name, evaluate_name = counters
        assert (
            getattr(result["evaluator"], prepare_name) == before[prepare_name]
        ), "re-evaluating within a generation must not prepare a new graph"
        assert getattr(result["evaluator"], evaluate_name) == before[evaluate_name] + 1
    moved_reference = adapter.evaluate_reference(reusable)
    np.testing.assert_allclose(
        adapter.node_energies(result["evaluator"]),
        moved_reference["energies"],
        rtol=adapter.rtol,
        atol=adapter.atol,
        err_msg=f"{adapter.key}/{tier} reused-generation energies",
    )
    np.testing.assert_allclose(
        adapter.node_forces(result["evaluator"]),
        moved_reference["directed_forces"],
        rtol=adapter.rtol,
        atol=adapter.atol,
        err_msg=f"{adapter.key}/{tier} reused-generation directed forces",
    )


def test_lifecycle_binds_every_neutral_step(instance):
    """Every neutral lifecycle step resolves to a real entry point."""
    for tier in instance.lifecycle_tiers:
        instance.assert_lifecycle_bound(instance.evaluator(tier))


# ---------------------------------------------------------------------------
# 2. Reference evaluation with JIT disabled and no artifacts
# ---------------------------------------------------------------------------


def test_compiler_free_evaluation_needs_no_artifact(instance, monkeypatch, tmp_path):
    """End-to-end compiler-free evaluation, per instance.

    The JIT policy is disabled and the artifact cache is empty, so any
    dependency on generated code would fail here instead of silently passing.
    The reduced properties the instance declares are checked against the T0
    oracle, which is the whole host coverage proof.
    """
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "none")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "empty-cache"))

    atoms = instance.structure()
    evaluator = instance.evaluator("portable")
    if instance.field_coupled:
        # A field-coupled contract has no compiler-free prepared plan; its
        # compiler-free instance is the family's whole-evaluator field entry.
        result = instance.evaluate_compiler_free(evaluator, atoms)
        graph = result["graph"]
        reference = instance.evaluate_reference(graph)
        np.testing.assert_allclose(
            result["energies"],
            reference["energies"],
            rtol=instance.rtol,
            atol=instance.atol,
        )
        np.testing.assert_allclose(
            result["directed_forces"],
            reference["directed_forces"],
            rtol=instance.rtol,
            atol=instance.atol,
        )
        np.testing.assert_allclose(
            result["field_adjoint"],
            reference["field_adjoint"],
            rtol=instance.rtol,
            atol=instance.atol,
        )
    else:
        graph = instance.graph(atoms, evaluator)
        generation = instance.prepare_graph(evaluator, graph)
        instance.prepare_geometry(evaluator, generation, graph)
        instance.compute_prepared(evaluator, generation, graph)
        reference = instance.evaluate_reference(graph)
        np.testing.assert_allclose(
            instance.node_energies(evaluator),
            reference["energies"],
            rtol=instance.rtol,
            atol=instance.atol,
        )
        np.testing.assert_allclose(
            instance.node_forces(evaluator),
            reference["directed_forces"],
            rtol=instance.rtol,
            atol=instance.atol,
        )
        missing = instance.expected_reductions - {"reduce_forces", "reduce_stress"}
        assert not missing, missing
        np.testing.assert_allclose(
            instance.reduce_forces(evaluator, generation, graph),
            _directed_force_reduction(graph, reference["directed_forces"]),
            rtol=max(instance.rtol, 1e-9),
            atol=max(instance.atol, 1e-9),
        )
        if "reduce_stress" in instance.expected_reductions:
            stress = instance.reduce_stress(evaluator, generation, graph)
            assert np.abs(stress).max() > 0.0
            np.testing.assert_allclose(
                _voigt(0.5 * (stress + stress.T)),
                _finite_strain_oracle(
                    instance.reference_energy, atoms, step=instance.strain_step
                ),
                rtol=instance.stress_rtol,
                atol=instance.stress_atol,
            )
            assert np.abs(stress - stress.T).max() <= instance.stress_antisymmetry_atol

    instance.assert_tier_state(evaluator, "portable")
    assert instance.artifact_ready(evaluator) is False
    assert evaluator.family_diagnostics["fallback_count"] == 0
    # The compiler-free path must report itself as such: a diagnostic that
    # names a generated implementation here would mean generated code ran with
    # no artifact and the coverage proof is void.
    diagnostic = _diagnostic(evaluator)
    for marker in instance.compiler_free_marker():
        assert marker in diagnostic, diagnostic


# ---------------------------------------------------------------------------
# 3. Tier parity per operator kind
# ---------------------------------------------------------------------------


def test_tier_parity_per_operator_kind(instance):
    """Every tier the instance admits is asserted against the reference.

    The portable tier is asserted for every operator kind of the compiled
    program, and the generated tier for the kinds whose binding steps up to it
    once a host artifact is admitted.  A generated surface this runner cannot
    qualify is reported with its required runner rather than skipped.
    """
    atoms = instance.structure()
    portable = instance.evaluator("portable")
    program = _assert_interface_surface(portable)

    program_keys = {operator["capability_key"] for operator in program["operators"]}
    assert (
        program_keys == instance.expected_kinds
    ), f"{instance.key} declares {sorted(program_keys)}"

    portable_report = portable.family_binding_report
    for binding in portable_report["bindings"]:
        assert binding["qualified"], binding
        assert binding["tier"] in PORTABLE_TIERS, binding
        assert binding["portable_covered"], binding
        assert binding["artifact_required"] is False, binding
    assert portable_report["admitted"] is True
    assert portable_report["portable_node_count"] == portable_report["node_count"]

    # The generated tier: the artifact-bound instance must step exactly the
    # operator kinds the family declares as generated, and must execute
    # generated code (the evidence is a per-family observable).
    try:
        generated = instance.evaluator("generated")
    except Unqualified as unqualified:
        # Never a silent skip: the unqualified surface is reported with the
        # runner it requires, and the compiler-free tier still covers every
        # operator kind, so the instance is covered even when the generated
        # tier is not.
        assert unqualified.surface, unqualified
        assert unqualified.runner, unqualified
        assert set(instance.expected_reductions) <= program_keys
        return
    _assert_interface_surface(generated)
    instance.assert_tier_state(generated, "generated")

    generated_report = generated.family_binding_report
    stepped = {
        binding["capability_key"]
        for binding in generated_report["bindings"]
        if binding["tier"] == "generated"
    }
    assert (
        stepped == instance.generated_kinds
    ), f"{instance.key} generated tier covers {sorted(stepped)}"
    assert generated_report["admitted"] is True
    # A generated binding is served by an identified artifact, never by an
    # unnamed implementation, and its portable coverage survives underneath.
    for binding in generated_report["bindings"]:
        if binding["tier"] == "generated":
            assert binding["implementation_id"].startswith("generated")
            # A host program keeps its portable implementation underneath the
            # generated one, so the admitted artifact accelerates the node
            # rather than becoming the only way to serve it: `artifact_required`
            # stays false and the host admission invariant holds either way.
            assert binding["portable_covered"] is True, binding
            assert binding["artifact_required"] is False, binding
        else:
            assert binding["tier"] in PORTABLE_TIERS, binding
            assert binding["portable_covered"] is True, binding
    assert generated_report["portable_node_count"] == sum(
        1 for binding in generated_report["bindings"] if binding["portable_covered"]
    )

    result = instance.run_lifecycle("generated", atoms)
    generated_reference = instance.evaluate_reference(result["graph"])
    np.testing.assert_allclose(
        result["energies"],
        generated_reference["energies"],
        rtol=instance.rtol,
        atol=instance.atol,
        err_msg=f"{instance.key} generated-tier energies",
    )
    np.testing.assert_allclose(
        result["directed_forces"],
        generated_reference["directed_forces"],
        rtol=instance.rtol,
        atol=instance.atol,
        err_msg=f"{instance.key} generated-tier directed forces",
    )
    # The generated tier's reductions are already asserted against the oracle by
    # the lifecycle test, which drives every admitted tier.
    evidence = instance.generated_execution_evidence(result["evaluator"])
    assert evidence.pop("artifact_id")
    for name, value in evidence.items():
        assert value not in ("", None, 0), (name, value)


# ---------------------------------------------------------------------------
# 4. Host portable coverage per admitted contract instance
# ---------------------------------------------------------------------------


def test_host_portable_coverage_per_contract_instance(instance):
    """Portable coverage is asserted per instance, not per operator kind.

    Every operator *node* of the compiled program is covered by the host
    portable implementation, at the declared compute precision, with the
    declared accumulator, and with the node's own declared buffer contracts
    (shape, layout, dtype, irreps).  The declared layout orders and the declared
    derivative requirements are the instance's, asserted as a set.
    """
    evaluator = instance.evaluator("portable")
    program = evaluator.family_program
    capabilities = evaluator.family_capabilities
    report = evaluator.family_binding_report
    bindings = {binding["operator_id"]: binding for binding in report["bindings"]}

    declared_layouts = set()
    declared_dtypes = set()
    seen_shape_extents = set()
    for operator in program["operators"]:
        binding = bindings[operator["id"]]
        assert binding["qualified"] is True, operator["id"]
        assert binding["portable_covered"] is True, operator["id"]
        assert binding["tier"] in PORTABLE_TIERS, binding
        # The precision instance: the binding computes at the instance's own
        # precision and accumulates at the contract's accumulator.
        assert binding["precision"]["compute"] == capabilities["precision"]
        assert binding["precision"]["accumulator"] == "float64"
        for contract in (*operator["inputs"], *operator["outputs"]):
            assert contract["layout"] in LAYOUT_ORDERS, contract
            assert contract["dtype"] == capabilities["precision"], contract
            assert isinstance(contract["irreps"], str)
            shape = contract["shape"]
            # The extent contract every family declares: a leading dynamic
            # extent that the runtime topology fills, then static widths.
            assert shape and all(isinstance(extent, int) for extent in shape), contract
            assert shape[0] == 0, contract
            assert all(extent > 0 for extent in shape[1:]), contract
            declared_layouts.add(contract["layout"])
            declared_dtypes.add(contract["dtype"])
            seen_shape_extents.update(int(extent) for extent in shape)

    # The layout and dtype instances are the program's, and the instance that
    # the suite drives declares them (so a program that silently dropped a
    # layout order from its buffers fails here).
    assert (
        declared_layouts == instance.expected_layouts
    ), f"{instance.key} declares layouts {sorted(declared_layouts)}"
    assert declared_dtypes == {"float64"}
    assert seen_shape_extents, "a program must declare at least one buffer extent"

    # The derivative-requirement instance: every declared reduction consumes
    # the retained adjoint of the generation it reduces.
    for operator in program["operators"]:
        if operator["kind"] != "reduction":
            continue
        adjoints = [
            contract for contract in operator["inputs"] if "adjoint" in contract["name"]
        ]
        assert adjoints, operator["id"]
        for contract in adjoints:
            assert contract["layout"] in instance.expected_layouts, contract

    # The derivative-requirement instance: the reductions the program declares,
    # the capabilities the instance advertises, and the lifecycle steps the
    # suite drives agree.
    reduction_keys = {
        operator["capability_key"]
        for operator in program["operators"]
        if operator["kind"] == "reduction"
    }
    assert reduction_keys == set(
        instance.expected_reductions
    ), f"{instance.key} declares reductions {sorted(reduction_keys)}"
    assert capabilities["target"] == "host"
    assert capabilities["portable"] is True
    # Every family declares its stress capability honestly.  An instance whose
    # program declares no reduction node still serves stress from its evaluator,
    # so the capability is True there; a declared reduction node implies it too.
    if instance.expected_reductions:
        assert capabilities["stress"] is True
    if instance.field_coupled:
        assert capabilities["field"] is True
    else:
        assert capabilities["field"] is False


def test_declared_reductions_match_the_registry_bindings(instance):
    """The registry binds exactly the reduction instances the program declares.

    An instance whose program declares no reduction node has none bound either,
    so the narrower program shape is asserted rather than special-cased away:
    the set of reduction capability keys in the binding report equals the set
    in the compiled program for every instance.
    """
    evaluator = instance.evaluator("portable")
    program_keys = {
        operator["capability_key"]
        for operator in evaluator.family_program["operators"]
        if operator["kind"] == "reduction"
    }
    report_keys = {
        binding["capability_key"]
        for binding in evaluator.family_binding_report["bindings"]
        if binding["kind"] == "reduction"
    }
    assert report_keys == program_keys == set(instance.expected_reductions)


# ---------------------------------------------------------------------------
# 5. Field derivative instance
# ---------------------------------------------------------------------------


def test_field_derivative_requirement_is_served(adapters):
    """A declared field derivative is served on both admitted paths.

    The compilation-free instance is the family's whole-evaluator field entry
    point; the generated instance is the prepared field lifecycle, which also
    serves the graph-level response derivative.  Both are asserted against the
    family's reference field entry point, for every instance whose family module
    declares the prepared field response.
    """
    declared = FIELD_RESPONSE_INSTANCES
    assert declared, "no instance declares a prepared field response"
    for key in declared:
        adapter = adapters[key]
        atoms = adapter.structure()

        portable = adapter.evaluator("portable")
        assert adapter.artifact_ready(portable) is False
        compiler_free = adapter.evaluate_compiler_free(portable, atoms)
        reference = adapter.evaluate_reference(compiler_free["graph"])
        np.testing.assert_allclose(
            compiler_free["field_adjoint"],
            reference["field_adjoint"],
            rtol=adapter.rtol,
            atol=adapter.atol,
        )
        assert (
            np.abs(reference["field_adjoint"]).max() > 0.0
        ), "the field derivative must be non-zero for the assertion to mean anything"

        generated = adapter.evaluator("generated")
        result = adapter.run_lifecycle("generated", atoms)
        assert adapter.accepted_generation(generated) == result["generation"]
        np.testing.assert_allclose(
            np.asarray(result["field_adjoint"], dtype=np.float64).reshape(-1),
            adapter.expected_field_adjoint(
                "generated", result["graph"], reference["field_adjoint"]
            ),
            rtol=adapter.rtol,
            atol=adapter.atol,
        )
        response = adapter.response(generated, result["generation"], result["graph"])
        # A graph-level field of three components gives a 3x3 response Hessian and
        # the field derivative of every directed coordinate force, which is
        # [field component][edge coordinate].
        assert response["hessian"].shape == (9,)
        assert response["force_derivative"].shape == (3 * len(result["graph"].xyz),)
        np.testing.assert_allclose(
            response["hessian"].reshape(3, 3),
            response["hessian"].reshape(3, 3).T,
            rtol=1e-9,
            atol=1e-12,
        )

        # The response derivative must be the central difference of the same
        # directed forces with respect to the graph-level field.
        step = 1e-6
        numerical = np.zeros_like(response["force_derivative"])
        for component in range(3):
            for sign in (1, -1):
                shifted = np.array(adapter.field_vector, dtype=np.float64)
                shifted[component] += sign * step
                adapter._pending_field = shifted
                adapter.compute_prepared(
                    generated, result["generation"], result["graph"]
                )
                start = component * len(result["graph"].xyz)
                stop = start + len(result["graph"].xyz)
                numerical[start:stop] += (
                    sign * adapter.node_forces(generated) / (2.0 * step)
                )
        np.testing.assert_allclose(
            response["force_derivative"],
            numerical,
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            response["hessian"].reshape(3, 3),
            response["hessian"].reshape(3, 3).T,
            rtol=1e-9,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 6. Accelerator surfaces: passed, or unqualified with the runner they need
# ---------------------------------------------------------------------------


def test_unqualified_surface_is_reported_with_its_runner(adapters, monkeypatch):
    """A generated tier this runner cannot prepare is reported, never skipped.

    The compiler-free tier stays the admitted instance, so the suite must state
    the missing surface and the runner it needs instead of quietly passing, for
    every instance whose family declares a host artifact.
    """
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    declared = HOST_COMPILER_GATED_INSTANCES
    assert declared, "no instance declares a host-compiler-gated generated tier"
    for key in declared:
        adapter = adapters[key]
        adapter._evaluators.pop("generated", None)
        with pytest.raises(Unqualified) as raised:
            adapter.evaluator("generated")
        assert raised.value.surface
        assert raised.value.runner == CXX_RUNNER
        # The compiler-free tier is untouched by the missing compiler.
        assert adapter.evaluator("portable").family_artifact_ready is False


def test_accelerator_surfaces_report_their_required_runner(instance):
    """The accelerator tier is qualified or reported with its required runner.

    A host runner cannot exercise a device artifact or a device execution
    space, so the suite asserts the *reported* state precisely: the device
    artifact form is declared unavailable, the device artifact readiness names
    the missing class, and the required runner is stated.  On a CUDA/HIP runner
    the same assertions invert and the device round trip must be admitted.

    The requirement must name the mechanism the project actually selects
    accelerators with, so the report is actionable rather than a wish.
    """
    assert "SYMMETRIX_BACKEND" in ACCELERATOR_RUNNER
    status = instance.accelerator_status()
    assert set(status) == set(ACCELERATOR_SURFACES)
    execution_space = native_symmetrix._kokkos_default_execution_space()
    host_runner = execution_space in ("OpenMP", "Serial")
    for surface in ACCELERATOR_SURFACES:
        entry = status[surface]
        assert entry["runner"], entry
        if host_runner:
            # No accelerator build is selected, so this is unqualified and the
            # requirement is recorded instead of skipped.
            assert entry["qualified"] is False, entry
            assert entry["reason"], entry
        else:
            assert entry["round_trip"], entry
