"""The PolarMACE program: node contracts, tiers, schedule, and family wiring.

Task 11's deliverable is the fourth family's *compiled program*: the MH-1
backbone program the family evaluator compiles from its own model, plus the
long-range kind group's nodes, plus the unrolled global<->local chain as
contract data.  The program is what the execution layer plans and binds, so
this suite proves three things about it:

* **The program slice is readable without a model.**
  ``symmetrix.polar_long_range_program`` compiles the long-range half from
  contract data alone -- the node contracts, the tiered binding report, and
  the schedule -- which is what lets the conformance battery read the
  program's buffer contracts instead of pinning a parallel table.  The slice
  is checked against the shipped M checkpoint's own contract data and against
  the two fixture geometries, including the shape-validation pass the
  assembly runs (the program refuses an incoherent node set with the buffer
  named).

* **The chain is contract data, pinned cross-language.**
  ``symmetrix.polar_schedule`` (the C++ mirror in
  ``libsymmetrix/source/mace_polar_program.hpp``) must equal the Python
  contract builder's ``polar_execution_schedule`` entry for entry, for every
  recursion depth -- the same cross-language pin the kind and correction-mode
  vocabularies carry.

* **The family wiring reaches the declared adapter surface.**
  A Polar evaluator loaded from the family's artifact
  compiles the family's program through the *same* ``compile_program`` entry
  point, folds the group's descriptors into the *same* implementation
  registry, and reports the family's capabilities -- no parallel program
  query and no second registry.  This half needs a backbone model, and the
  repository's own synthetic MH-1 backbone (``test_mace_polar_backbone``'s
  two-interaction model, extracted the way the frontend extracts it) is the
  one the native evaluator admits, so the wiring runs wherever torch/e3nn/mace
  are importable rather than behind a checkpoint download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import symmetrix

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

from symmetrix.mace_polar_execution_contract import (
    POLAR_CONTRACT_KEY,
    polar_execution_schedule,
)
from symmetrix.polar_long_range_kinds import (
    POLAR_DELTA_KINDS,
    POLAR_LONG_RANGE_KINDS,
    PolarLongRangeContract,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "polar"
FACTS_PATH = DATA_DIR / "checkpoint_facts.json"
GEOMETRIES = ("mol5", "pbc4")

#: The local (backbone) region vocabulary the chain names, mirrored from
#: ``backbone_region_vocabulary`` in ``mace_polar_program.hpp``.
BACKBONE_REGIONS = frozenset(
    ("interactions", "feature_mixer", "field_update", "local_electron_energy")
)


def test_the_self_interaction_parameters_are_the_contracts_flags():
    """The energy and self-interaction nodes carry the contract's own flags.

    The checkpoint's polar block declares whether the k-space energy carries
    its self-interaction overlaps and whether the projected features carry the
    feature self-terms; the program stamps that policy on the two nodes that
    consume it, so the kinds' parameters are contract data rather than a
    kind-group default.  The Python mirror
    (``PolarLongRangeContract.kind_parameterizations``) is asserted against
    the compiled slice, which is what keeps the two languages pinned.
    """

    facts = json.loads(FACTS_PATH.read_text())
    contract = PolarLongRangeContract.from_checkpoint_facts(facts["models"]["M"])
    assert contract.include_electrostatic_self_interaction is True
    assert contract.field_si is False
    report = program_slice(contract)
    parameters = {node["kind"]: dict(node["parameters"]) for node in report["nodes"]}
    assert parameters["kspace_coulomb_energy"]["include_self_interaction"] == 1.0
    assert (
        parameters["self_interaction_linear"]["include_energy_self_interaction"] == 1.0
    )
    assert (
        parameters["self_interaction_linear"]["include_feature_self_interaction"] == 0.0
    )
    mirror = contract.kind_parameterizations()
    assert mirror["kspace_coulomb_energy"] == parameters["kspace_coulomb_energy"]
    assert mirror["self_interaction_linear"] == (parameters["self_interaction_linear"])


def program_slice(contract: PolarLongRangeContract) -> dict:
    """The long-range half of the PolarMACE program for one contract."""

    return symmetrix.polar_long_range_program(
        kspace_cutoff=contract.kspace_cutoff,
        density_max_l=contract.density_max_l,
        density_smearing_width=contract.density_smearing_width,
        feature_max_l=contract.feature_max_l,
        feature_smearing_widths=list(contract.feature_smearing_widths),
        charges_dimension=contract.charges_dimension,
        field_dimension=contract.field_dimension,
        potential_dimension=contract.potential_dimension,
        field_norm_factor=contract.field_norm_factor,
        recursion_step_count=contract.recursion_step_count,
        table_dtype=contract.table_dtype,
        include_electrostatic_self_interaction=(
            contract.include_electrostatic_self_interaction
        ),
        field_si=contract.field_si,
    )


@pytest.fixture(scope="module")
def checkpoint_contract() -> PolarLongRangeContract:
    """The shipped M checkpoint's long-range contract, from the pinned facts."""

    facts = json.loads(FACTS_PATH.read_text())
    return PolarLongRangeContract.from_checkpoint_facts(facts["models"]["M"])


def fixture_contract(name: str) -> PolarLongRangeContract:
    manifest = json.loads((DATA_DIR / f"{name}_manifest.json").read_text())
    parameters = manifest["parameters"]
    density_width = (parameters["density_max_l"] + 1) ** 2
    field_width = (parameters["feature_max_l"] + 1) ** 2 * len(
        parameters["feature_smearing_widths"]
    )
    return PolarLongRangeContract(
        kspace_cutoff=parameters["kspace_cutoff"],
        density_max_l=parameters["density_max_l"],
        density_smearing_width=parameters["density_smearing_width"],
        feature_max_l=parameters["feature_max_l"],
        feature_smearing_widths=tuple(parameters["feature_smearing_widths"]),
        charges_dimension=2 * density_width,
        field_dimension=field_width,
        potential_dimension=2 * field_width,
    )


# ---------------------------------------------------------------------------
# The program slice: node contracts, tiers, and the schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", GEOMETRIES)
def test_program_slice_declares_the_kind_group_with_portable_coverage(key):
    """The slice is the kind group's node set, bound at the portable tier.

    The host admission invariant is per node: every long-range node has a
    qualified implementation below the generated tier, so the family's
    program runs on the host without any artifact.
    """

    program = program_slice(fixture_contract(key))
    assert program["family"] == "polar-mace"
    assert program["precision"] == "float64"
    assert program["fingerprint"].startswith("sha256:")
    assert program["contract_fingerprint"].startswith("sha256:")
    assert [node["kind"] for node in program["nodes"]] == list(POLAR_LONG_RANGE_KINDS)

    report = program["binding_report"]
    assert report["admitted"]
    assert report["node_count"] == len(POLAR_LONG_RANGE_KINDS)
    assert report["portable_node_count"] == len(POLAR_LONG_RANGE_KINDS)
    for node, binding in zip(program["nodes"], report["bindings"]):
        assert binding["operator_id"] == node["id"]
        assert binding["capability_key"] == node["capability_key"]
        assert binding["tier"] == "portable"
        assert binding["portable_covered"]
        assert not binding["artifact_required"]
        assert binding["precision"] == {"compute": "float64", "accumulator": "float64"}


def test_program_slice_binds_only_the_target_it_declares():
    """A plan on another target leaves every node unbound.

    The group's delivered implementation is the portable host one, so a plan
    for an accelerator target reports its nodes unbound with the target named
    rather than silently degrading to another tier -- the same refusal the
    kind-group probe asserts on the C++ side.
    """

    contract = fixture_contract("mol5")
    report = symmetrix.polar_long_range_program(
        kspace_cutoff=contract.kspace_cutoff,
        density_max_l=contract.density_max_l,
        density_smearing_width=contract.density_smearing_width,
        feature_max_l=contract.feature_max_l,
        feature_smearing_widths=list(contract.feature_smearing_widths),
        charges_dimension=contract.charges_dimension,
        field_dimension=contract.field_dimension,
        potential_dimension=contract.potential_dimension,
        field_norm_factor=contract.field_norm_factor,
        recursion_step_count=contract.recursion_step_count,
        table_dtype=contract.table_dtype,
        target="cuda",
    )["binding_report"]
    assert not report["admitted"]
    assert report["bindings"]
    for binding in report["bindings"]:
        assert not binding["qualified"]
        assert "cuda" in binding["reason"]


def test_program_slice_rejects_incoherent_contract_data():
    """The assembly validates: garbage contract data has no program.

    The slice compiles through the same validation the declarative layer
    states -- the contract rules, the node shapes, and the schedule coverage --
    so an incoherent request fails at compilation with the rule named.
    """

    contract = fixture_contract("mol5")
    arguments = dict(
        kspace_cutoff=contract.kspace_cutoff,
        density_max_l=contract.density_max_l,
        density_smearing_width=contract.density_smearing_width,
        feature_max_l=contract.feature_max_l,
        feature_smearing_widths=list(contract.feature_smearing_widths),
        charges_dimension=contract.charges_dimension,
        field_dimension=contract.field_dimension,
        potential_dimension=contract.potential_dimension,
        field_norm_factor=contract.field_norm_factor,
        recursion_step_count=contract.recursion_step_count,
        table_dtype=contract.table_dtype,
    )
    with pytest.raises(ValueError, match="kspace_cutoff"):
        symmetrix.polar_long_range_program(**{**arguments, "kspace_cutoff": 0.0})
    with pytest.raises(ValueError, match="field_dimension"):
        symmetrix.polar_long_range_program(**{**arguments, "field_dimension": 7})
    with pytest.raises(ValueError, match="table_dtype"):
        symmetrix.polar_long_range_program(**{**arguments, "table_dtype": "float16"})
    with pytest.raises(ValueError, match="recursion"):
        symmetrix.polar_long_range_program(**{**arguments, "recursion_step_count": 0})


def test_program_schedule_mirrors_the_contract_builder(checkpoint_contract):
    """The native schedule equals the Python contract builder's, entry for entry.

    The chain is contract data on both sides of the language boundary: the
    Python builder embeds it in the POLAR_BACKBONE contract and the C++ mirror
    drives the family's orchestration, so the two are pinned equal for every
    depth the checkpoints declare (one for S, two for M and L).
    """

    for depth in (1, 2, 3):
        assert symmetrix.polar_schedule(depth) == polar_execution_schedule(depth)

    program = program_slice(checkpoint_contract)
    schedule = program["schedule"]
    assert schedule == polar_execution_schedule(
        checkpoint_contract.recursion_step_count
    )
    assert schedule["recursion_step_count"] == 2
    assert len(schedule["entries"]) == 4 + 4 * 2 + 6
    reached = [
        entry["name"]
        for entry in schedule["entries"]
        if entry["kind"] == "global_operator"
    ]
    assert set(reached) == set(POLAR_LONG_RANGE_KINDS)
    assert reached.count("fukui_normalize") == 3
    assert reached.count("kspace_coulomb_features") == 2
    for entry in schedule["entries"]:
        assert entry["kind"] in ("backbone", "global_operator")
        if entry["kind"] == "backbone":
            assert entry["name"] in BACKBONE_REGIONS


def test_program_identity_folds_the_chain_depth(checkpoint_contract):
    """U=1 and U=2 are different programs; same-depth rebuilds are stable.

    The node set is depth-independent (the chain is contract data), so the
    identity is carried by the chain's multiplicity on the scheduled nodes:
    the features and external-field nodes declare one evaluation per unrolled
    step, and the equilibration node one before the loop plus one per step.
    Two depths therefore fingerprint differently -- the property a consumer
    keys an artifact on -- while a rebuild of the same depth is byte-stable.
    """

    from dataclasses import replace

    deep = program_slice(checkpoint_contract)
    shallow = program_slice(replace(checkpoint_contract, recursion_step_count=1))
    assert checkpoint_contract.recursion_step_count == 2
    assert deep["fingerprint"] == (
        "sha256:216b2fa0b9600a9c2620fa440737e2dcb7c3953a64f5963a1b4656d12ccd2285"
    )
    assert deep["fingerprint"] != shallow["fingerprint"]
    assert deep["contract_fingerprint"] != shallow["contract_fingerprint"]
    assert program_slice(checkpoint_contract)["fingerprint"] == deep["fingerprint"]
    assert shallow["schedule"] == polar_execution_schedule(1)

    counts = {
        node["kind"]: node["parameters"]
        for node in deep["nodes"]
        if node["kind"]
        in (
            "kspace_coulomb_features",
            "external_field_projection",
            "fukui_normalize",
        )
    }
    assert counts["kspace_coulomb_features"]["evaluation_count"] == 2.0
    assert counts["external_field_projection"]["evaluation_count"] == 2.0
    assert counts["fukui_normalize"]["equilibration_count"] == 3.0
    shallow_counts = {node["kind"]: node["parameters"] for node in shallow["nodes"]}
    assert shallow_counts["kspace_coulomb_features"]["evaluation_count"] == 1.0
    assert shallow_counts["fukui_normalize"]["equilibration_count"] == 2.0


def test_program_schedule_is_a_fixed_linear_chain(checkpoint_contract):
    """The chain is fixed-depth and data-independent, per recursion depth.

    No entry carries a condition and the order is the oracle's own
    (interactions, geometry cache, mixer, the pre-loop equilibration, the
    unrolled updates, then the post-loop readout), which is the property that
    makes the model a valid fixed-orchestration program client.  The iteration
    indices are the unrolled steps, in order.
    """

    schedule = program_slice(checkpoint_contract)["schedule"]
    entries = schedule["entries"]
    assert [entry["iteration"] for entry in entries] == sorted(
        entry["iteration"] for entry in entries
    )
    assert [entry["iteration"] for entry in entries[:4]] == [0, 0, 0, 0]
    assert entries[-1]["iteration"] == 3
    for entry in entries:
        assert set(entry) == {"kind", "name", "iteration"}
    for depth in (1, 4):
        chain = symmetrix.polar_schedule(depth)
        assert chain["recursion_step_count"] == depth
        assert len(chain["entries"]) == 4 + 4 * depth + 6


def test_program_slice_rejects_an_unknown_kind_name():
    """An unknown operator or depth is a clean failure naming the vocabulary."""

    from symmetrix.polar_long_range_kinds import require_long_range_kind

    with pytest.raises(ValueError, match="kspace_geometry_cache"):
        require_long_range_kind("kspace-geometry-cache")
    with pytest.raises(ValueError, match="at least one recursion step"):
        polar_execution_schedule(0)
    with pytest.raises(ValueError, match="at least one recursion step"):
        polar_execution_schedule(True)
    # The native mirror's refusal is the group's no-spelling idiom: an empty
    # entry list with a zero depth, so it never claims a chain it does not
    # have.
    assert symmetrix.polar_schedule(0) == {
        "version": 1,
        "recursion_step_count": 0,
        "entries": [],
    }
    assert symmetrix.polar_schedule(-1)["entries"] == []


# ---------------------------------------------------------------------------
# The family wiring: a backbone evaluator configured with the contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_backbone(tmp_path_factory):
    """Synthetic Polar and ordinary sibling artifacts for program comparison.

    The family wiring needs a backbone model, and the repository's own
    synthetic PolarMACE-shaped backbone -- the two-interaction one
    ``test_mace_polar_backbone`` builds in process, with the supported
    nonlinear readout and scalar scale/shift the native MH-1 evaluator
    requires -- is exactly the topology it admits: eleven backbone nodes for
    two interaction layers, so the assembled program carries twenty.  The
    frontend hands the evaluator this extracted JSON, so the wiring starts
    from the artifact a real calculator would.
    """

    import json as _json

    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data
    from symmetrix.extract_mace_polar import extract_mace_polar_data
    from test_mace_polar_backbone import _build_polar_shaped_model

    model = _build_polar_shaped_model(num_interactions=2)
    polar_data = extract_mace_polar_data(model)
    ordinary_data = extract_mace_nonlinear_data(model)
    root = tmp_path_factory.mktemp("polar-program")
    polar_path = root / "polar.json"
    ordinary_path = root / "ordinary.json"
    polar_path.write_text(_json.dumps(polar_data, separators=(",", ":")))
    ordinary_path.write_text(_json.dumps(ordinary_data, separators=(",", ":")))
    return polar_data, polar_path, ordinary_path


@pytest.fixture(scope="module")
def wired_evaluator(synthetic_backbone):
    """An MH-1 backbone evaluator configured with the model's own contract.

    The evaluator is constructed exactly as the frontend constructs it, and
    the polar wiring is the family's own configuration step, driven by the
    model's embedded POLAR_BACKBONE contract -- the same long-range slice a
    real PolarMACE model carries.  From there on the program, the registry,
    and the schedule are the family's.
    """

    data, path, _ordinary_path = synthetic_backbone
    evaluator = symmetrix.MACEPolarKokkos(str(path))
    contract = PolarLongRangeContract.from_backbone_contract(
        data["execution_contracts"][POLAR_CONTRACT_KEY]
    )
    contract.validate()
    yield evaluator, contract
    evaluator = None


def test_family_program_extends_the_backbone_with_the_global_nodes(
    wired_evaluator, synthetic_backbone
):
    """One program query: the backbone nodes plus the long-range group.

    The family does not get a second program surface -- ``compile_program``
    (and therefore the declared ``family_program`` query) returns the
    assembled program, and the binding report binds every node of it, so the
    node tiers of both halves are readable from the one report.  The
    assembly is additive: the bare backbone's nodes lead the program
    unchanged, and the six Polar-delta plus nine long-range nodes follow.
    """

    evaluator, contract = wired_evaluator
    _data, _path, ordinary_path = synthetic_backbone
    assert evaluator.has_polar_long_range_contract
    program = evaluator.family_program
    assert program["family"] == "polar-mace"
    kinds = [node["kind"] for node in program["operators"]]
    long_range = [kind for kind in kinds if kind in POLAR_LONG_RANGE_KINDS]
    deltas = [kind for kind in kinds if kind in POLAR_DELTA_KINDS]
    backbone = [
        kind
        for kind in kinds
        if kind not in POLAR_LONG_RANGE_KINDS and kind not in POLAR_DELTA_KINDS
    ]
    assert long_range == list(POLAR_LONG_RANGE_KINDS)
    assert deltas == list(POLAR_DELTA_KINDS)
    assert backbone, "the backbone nodes stay in the assembled program"
    bare_ids = [
        node["id"]
        for node in symmetrix.MACENonlinearKokkos(str(ordinary_path)).family_program[
            "operators"
        ]
    ]
    ids = [node["id"] for node in program["operators"]]
    assert ids[: len(bare_ids)] == bare_ids, "the backbone's nodes lead unchanged"
    assert len(ids) == (
        len(bare_ids) + len(POLAR_DELTA_KINDS) + len(POLAR_LONG_RANGE_KINDS)
    )
    # The backbone legitimately repeats kinds (one radial, harmonics, and
    # product node per interaction layer), so the uniqueness the report keys on
    # is the node *id*: a program that reused one would bind two nodes to one
    # report row.
    assert len(ids) == len(set(ids)), "node ids are unique"

    report = evaluator.family_binding_report
    assert report["node_count"] == len(ids)
    tiers = {binding["operator_id"]: binding["tier"] for binding in report["bindings"]}
    assert len(tiers) == len(ids)
    for node in program["operators"]:
        assert tiers[node["id"]] in ("portable", "generated", "frozen", "reference")
    global_ids = [
        node["id"]
        for node in program["operators"]
        if node["kind"] in POLAR_LONG_RANGE_KINDS
    ]
    assert all(tiers[node_id] == "portable" for node_id in global_ids)


def test_family_capabilities_follow_the_program(wired_evaluator):
    """A configured evaluator declares the field coupling its program has."""

    evaluator, _contract = wired_evaluator
    capabilities = evaluator.family_capabilities
    assert capabilities["field"] is True
    assert capabilities["stress"] is True
    if capabilities["target"] != "host":
        pytest.skip(
            "the generated-target capability follows the build's execution "
            f"space, and this build's is {capabilities['target']!r}; the host "
            "pin is not exercised here (a device generation of the family's "
            "checkpoints is refused by name)"
        )
    assert capabilities["target"] == "host"


def test_family_schedule_is_the_contract_chain(wired_evaluator):
    """The evaluator reports the chain its contract declares."""

    evaluator, contract = wired_evaluator
    schedule = evaluator.polar_schedule
    assert schedule == polar_execution_schedule(contract.recursion_step_count)


def test_unconfigured_evaluator_compiles_the_backbone_alone(synthetic_backbone):
    """The wiring is contract-gated: without the contract nothing changes.

    The gate is what makes the family additive: an MH-1 evaluator that carries
    no long-range contract compiles exactly the program and registry it always
    did, and reports no schedule.
    """

    _data, _path, ordinary_path = synthetic_backbone
    evaluator = symmetrix.MACENonlinearKokkos(str(ordinary_path))
    assert not hasattr(evaluator, "has_polar_long_range_contract")
    assert not hasattr(evaluator, "polar_schedule")
    program = evaluator.family_program
    assert program["family"] == "mace-nonlinear"
    kinds = {node["kind"] for node in program["operators"]}
    assert not (kinds & set(POLAR_LONG_RANGE_KINDS))
