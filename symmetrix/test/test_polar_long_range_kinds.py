"""Contract-data linkage of the long-range operator kind group.

The Python side of the kind declarations
(``symmetrix/polar_long_range_kinds.py``) maps POLAR_BACKBONE contract
instances and the pinned checkpoint facts onto the kind parameterizations
the C++ registry group binds.  These tests pin:

* the realistic shapes: the checkpoint-facts entries (S/M/L) and a
  POLAR_BACKBONE-shaped contract instance both parameterize the same nine
  kinds, and the two paths converge on identical parameterizations;
* the admission rules: garbage contract data (wrong tag, missing descriptor,
  out-of-support max_l, inconsistent dimensions, a wrong precision policy,
  an unknown correction mode) rejects with the rule named;
* the vocabulary contracts: unknown kinds fail cleanly with the vocabulary
  listed, the correction-mode tokens stay equal to the manifest reader's
  (``benchmarks/foundation_model_support_matrix.py``), and the
  declaration-time derivative service mirrors the C++ capability flags
  exactly (position adjoints on the position-dependent kinds, the
  normalization adjoint on the scatter-and-broadcast kind, the experimental
  cell adjoint on the cell-dependent kinds);
* the capacity records stay in the manifest's long_range vocabulary and
  match the fixture geometry (mol5's N_k of 62 over 5 atoms).

Runs in the plain repository environment: stdlib and the torch-free
contract module only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from symmetrix.mace_polar_execution_contract import (
    POLAR_CONTRACT_TAG,
    normalize_execution_polar_backbone_contract,
)
from symmetrix.polar_long_range_kinds import (
    CUBIC_MADELUNG,
    EXTERNAL_FIELD_SCALE,
    HALF_SPHERE_WEIGHT,
    POLAR_LONG_RANGE_CELL_ADJOINT_KINDS,
    POLAR_LONG_RANGE_CORRECTION_MODES,
    POLAR_LONG_RANGE_DECLARED_DERIVATIVE_MODES,
    POLAR_LONG_RANGE_DERIVATIVE_MODES,
    POLAR_LONG_RANGE_KINDS,
    POLAR_LONG_RANGE_NORMALIZATION_ADJOINT_KINDS,
    POLAR_LONG_RANGE_POSITION_ADJOINT_KINDS,
    PolarLongRangeContract,
    kind_derivative_modes,
    require_long_range_kind,
)

FACTS = Path(__file__).parent / "data" / "polar" / "checkpoint_facts.json"
SUPPORT_MATRIX = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "foundation_model_support_matrix.py"
)


@pytest.fixture(scope="module")
def facts() -> dict:
    return json.loads(FACTS.read_text())


@pytest.fixture(scope="module")
def checkpoint_m() -> PolarLongRangeContract:
    return PolarLongRangeContract.from_checkpoint_facts(
        json.loads(FACTS.read_text())["models"]["M"]
    )


def backbone_contract_m() -> dict:
    """A POLAR_BACKBONE contract instance in the realistic M shape.

    The slice mirrors exactly what ``make_execution_polar_backbone_contract``
    emits for the M checkpoint (the checkpoint facts record its embedded
    contract); the linkage consumes only this slice, and the full-contract
    validation stays with the backbone normalizer's own suite.
    """

    return {
        "tag": POLAR_CONTRACT_TAG,
        "schema": "symmetrix.execution.polar_backbone",
        "version": 2,
        "interaction_count": 2,
        "recursion_step_count": 2,
        "keep_last_layer_irreps": True,
        "charges": {
            "irreps": "1x0e+1x1o+1x0e+1x1o",
            "dimension": 8,
            "max_l": 1,
            "smearing_width": 1.5,
        },
        "field": {
            "irreps": "2x0e+2x1o",
            "dimension": 8,
            "max_l": 1,
            "smearing_widths": [1.5, 3.0],
            "feature_norm_count": 8,
            "kspace_cutoff": 1.8467166423797607,
        },
        "potential": {
            "irreps": "2x0e+2x1o+2x0e+2x1o",
            "dimension": 16,
        },
        "field_norm_factor": 1.0,
        "flags": {
            "add_local_electron_energy": True,
            "include_electrostatic_self_interaction": True,
            "quadrupole_feature_corrections": False,
            "field_si": False,
        },
    }


# ---------------------------------------------------------------------------
# Realistic shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", ("S", "M", "L"))
def test_every_checkpoint_fact_parameterizes_the_kind_group(facts, size):
    contract = PolarLongRangeContract.from_checkpoint_facts(facts["models"][size])
    contract.validate()

    parameters = contract.kind_parameterizations()
    assert set(parameters) == set(POLAR_LONG_RANGE_KINDS)
    model = facts["models"][size]
    cache = parameters["kspace_geometry_cache"]
    assert cache["kspace_cutoff"] == model["kspace_cutoff"]
    assert cache["density_max_l"] == model["atomic_multipoles_max_l"]
    assert cache["feature_max_l"] == model["field_feature_max_l"]
    assert (
        cache["density_smearing_width"] == (model["atomic_multipoles_smearing_width"])
    )
    for index, width in enumerate(model["field_feature_widths"]):
        assert cache[f"feature_smearing_width_{index}"] == width
    # The unit-convention constants land on their kinds, together with the
    # chain multiplicity the unrolled iteration gives each scheduled node:
    # once per recursion step for the features and the external field, and
    # once before the loop plus once per step for the equilibration.
    assert parameters["kspace_coulomb_features"] == {
        "half_sphere_weight": HALF_SPHERE_WEIGHT,
        "evaluation_count": float(model["num_update"]),
    }
    assert parameters["analytic_correction_energy"] == {
        "cubic_madelung": CUBIC_MADELUNG
    }
    assert parameters["external_field_projection"]["external_field_scale"] == (
        EXTERNAL_FIELD_SCALE
    )
    assert parameters["external_field_projection"]["evaluation_count"] == (
        float(model["num_update"])
    )
    assert parameters["fukui_normalize"]["equilibration_count"] == (
        1.0 + float(model["num_update"])
    )


def test_checkpoint_m_shape_is_pinned(checkpoint_m):
    assert checkpoint_m.kspace_cutoff == pytest.approx(1.8467166423797607)
    assert checkpoint_m.density_max_l == 1
    assert checkpoint_m.feature_max_l == 1
    assert checkpoint_m.feature_smearing_widths == (1.5, 3.0)
    assert checkpoint_m.density_smearing_width == 1.5
    assert checkpoint_m.charges_dimension == 8
    assert checkpoint_m.field_dimension == 8
    assert checkpoint_m.potential_dimension == 16
    assert checkpoint_m.recursion_step_count == 2
    # The static buffer extents the contract fixes: the per-channel
    # multipole width (4), the per-channel feature width (8), and the
    # two-channel potential width (16).
    assert checkpoint_m.density_channel_width == 4
    assert checkpoint_m.field_channel_width == 8


def test_backbone_contract_maps_to_the_same_parameterizations(checkpoint_m):
    from_backbone = PolarLongRangeContract.from_backbone_contract(backbone_contract_m())
    from_backbone.validate()
    assert from_backbone.kind_parameterizations() == (
        checkpoint_m.kind_parameterizations()
    )


def test_backbone_contract_rejects_a_foreign_tag():
    contract = backbone_contract_m()
    contract["tag"] = "symmetrix.execution.mh1_uvu/3"
    with pytest.raises(ValueError, match=f"bind '{POLAR_CONTRACT_TAG}'"):
        PolarLongRangeContract.from_backbone_contract(contract)


def test_a_version_1_polar_contract_is_rejected_by_name():
    """The pre-schedule contract version is refused, not silently upgraded.

    Version 2 adds the ``execution`` chain, and the version-1 fingerprints
    were computed without it, so "upgrading" a stored version-1 payload would
    have to re-derive the identity it carries.  Nothing shipped version 1 (the
    contract is introduced by the polar plan and its consumers are the
    frontend and parity stages), so the honest failure is a refusal that names
    the expected identity rather than a silent re-fingerprint.  The MH-1
    versions, by contrast, shipped and are upgraded by exact projections.
    """

    version_1 = backbone_contract_m()
    version_1["tag"] = "symmetrix.execution.polar_backbone/1"
    version_1["version"] = 1
    with pytest.raises(ValueError, match=f"tag must be {POLAR_CONTRACT_TAG!r}"):
        normalize_execution_polar_backbone_contract(version_1)

    stale_version = backbone_contract_m()
    stale_version["version"] = 1
    with pytest.raises(ValueError, match="version must be 2"):
        normalize_execution_polar_backbone_contract(stale_version)


def test_backbone_contract_requires_its_descriptors():
    contract = backbone_contract_m()
    del contract["field"]
    with pytest.raises(ValueError, match="missing its field descriptor"):
        PolarLongRangeContract.from_backbone_contract(contract)


# ---------------------------------------------------------------------------
# Admission rules
# ---------------------------------------------------------------------------


def _invalid(checkpoint_m, **changes) -> PolarLongRangeContract:
    from dataclasses import replace

    return replace(checkpoint_m, **changes)


@pytest.mark.parametrize(
    "changes, message",
    (
        ({"kspace_cutoff": 0.0}, "kspace_cutoff must be"),
        ({"density_max_l": 2}, "reject l > 1"),
        ({"feature_max_l": 5}, "reject l > 1"),
        ({"density_smearing_width": -1.5}, "density_smearing_width"),
        ({"feature_smearing_widths": (1.5, 0.0)}, "feature_smearing_widths"),
        ({"feature_smearing_widths": ()}, "at least one"),
        ({"charges_dimension": 7}, "charges_dimension"),
        ({"field_dimension": 7}, "field_dimension"),
        ({"potential_dimension": 15}, "potential_dimension"),
        ({"spin_channel_count": 3}, "alpha/beta"),
        ({"recursion_step_count": 0}, "recursion"),
        ({"table_dtype": "float16"}, "table_dtype"),
        ({"accumulator_dtype": "float32"}, "accumulator_dtype"),
        ({"field_norm_factor": 0.0}, "field_norm_factor"),
    ),
)
def test_garbage_contract_data_rejects(checkpoint_m, changes, message):
    broken = _invalid(checkpoint_m, **changes)
    with pytest.raises(ValueError, match=message):
        broken.validate()


def test_dimension_rules_follow_the_basis_definition(checkpoint_m):
    # A consistent width-l basis: max_l 0 halves every width.
    smaller = _invalid(
        checkpoint_m,
        density_max_l=0,
        feature_max_l=0,
        charges_dimension=2,
        field_dimension=2,
        potential_dimension=4,
    )
    smaller.validate()
    assert smaller.density_channel_width == 1
    assert smaller.field_channel_width == 2


def test_checkpoint_facts_entry_requires_its_fields():
    with pytest.raises(ValueError, match="is missing"):
        PolarLongRangeContract.from_checkpoint_facts({"model_class": "PolarMACE"})


# ---------------------------------------------------------------------------
# Vocabulary contracts
# ---------------------------------------------------------------------------


def test_unknown_kind_fails_cleanly_with_the_vocabulary():
    with pytest.raises(ValueError, match="unknown long-range operator kind"):
        require_long_range_kind("kspace-coulomb-feature")
    with pytest.raises(ValueError, match="kspace_geometry_cache"):
        require_long_range_kind("")


def test_declaration_time_derivative_service_mirrors_the_capability_flags():
    """Every declared derivative mode is backed by an implemented adjoint.

    The sets mirror the C++ ``LongRangeKindCapabilities`` flags the kind-group
    probe asserts: position adjoints on the kinds whose forward consumes
    positions, the experimental reciprocal-space cell adjoint (the Ewald
    stress) on the kinds whose forward carries cell-dependent terms, and the
    normalization adjoint on the global scatter-and-broadcast kind.  A kind
    appears in a set only because its adjoint surface exists
    (``long_range_kernels.hpp``, FD-verified in the adjoints and stress
    batteries).
    """
    expected_position = {
        "kspace_geometry_cache",
        "kspace_coulomb_features",
        "kspace_coulomb_energy",
        "external_field_projection",
        "dipole_readout",
        "analytic_correction_energy",
        "analytic_correction_features",
    }
    expected_cell = {
        "kspace_geometry_cache",
        "kspace_coulomb_features",
        "kspace_coulomb_energy",
        "analytic_correction_energy",
        "analytic_correction_features",
    }
    assert POLAR_LONG_RANGE_POSITION_ADJOINT_KINDS == expected_position
    assert POLAR_LONG_RANGE_CELL_ADJOINT_KINDS == expected_cell
    assert POLAR_LONG_RANGE_NORMALIZATION_ADJOINT_KINDS == {"fukui_normalize"}
    assert expected_position >= expected_cell
    assert not (expected_cell & POLAR_LONG_RANGE_NORMALIZATION_ADJOINT_KINDS)

    for kind in POLAR_LONG_RANGE_KINDS:
        declared = ["forward"]
        if kind in expected_position:
            declared.append("position_adjoint")
        if kind in POLAR_LONG_RANGE_NORMALIZATION_ADJOINT_KINDS:
            declared.append("normalization_adjoint")
        if kind in expected_cell:
            declared.append("cell_adjoint")
        assert kind_derivative_modes(kind) == tuple(
            mode for mode in POLAR_LONG_RANGE_DERIVATIVE_MODES if mode in declared
        )
    assert set(POLAR_LONG_RANGE_DERIVATIVE_MODES) == {
        "forward",
        "position_adjoint",
        "normalization_adjoint",
        "cell_adjoint",
    }
    assert POLAR_LONG_RANGE_DECLARED_DERIVATIVE_MODES == set(
        POLAR_LONG_RANGE_DERIVATIVE_MODES
    )


def test_correction_modes_match_the_manifest_reader():
    spec = importlib.util.spec_from_file_location(
        "foundation_model_support_matrix", SUPPORT_MATRIX
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert POLAR_LONG_RANGE_CORRECTION_MODES == (module.LONG_RANGE_CORRECTION_MODES)


# ---------------------------------------------------------------------------
# Capacity records
# ---------------------------------------------------------------------------


def test_capacity_records_stay_in_the_manifest_vocabulary(checkpoint_m):
    # mol5: 62 half-sphere k-points over 5 atoms, molecule correction.  The
    # declared table dtype is the delivered one (fp64: the tier-1 kernels
    # carry every table in double, and fp32 quantization of them misses the
    # group's 1e-12 parity gate), so the record's bytes follow it.
    record = checkpoint_m.capacity_record(62, 5, "makov-payne-dabo")
    assert record == {
        "n_k": 62,
        "table_bytes": 2 * 62 * 5 * 8,
        "correction_mode": "makov-payne-dabo",
        "rebuild_on_cell_change": True,
    }
    assert checkpoint_m.table_dtype == "float64"
    # An explicit dtype is still contract data: an fp32 delivery would double
    # the capacity down, which is exactly why the field exists.
    compact = checkpoint_m.capacity_record(62, 5, "makov-payne-dabo", "float32")
    assert compact["table_bytes"] == 2 * 62 * 5 * 4
    bulk = checkpoint_m.capacity_record(55, 4, "none", "float64", False)
    assert bulk["table_bytes"] == 2 * 55 * 4 * 8
    assert bulk["rebuild_on_cell_change"] is False


@pytest.mark.parametrize("mode", POLAR_LONG_RANGE_CORRECTION_MODES)
def test_every_correction_mode_admits_a_record(checkpoint_m, mode):
    assert checkpoint_m.capacity_record(1, 1, mode)["correction_mode"] == mode


def test_unknown_correction_mode_rejects(checkpoint_m):
    with pytest.raises(ValueError, match="unknown long-range correction mode"):
        checkpoint_m.capacity_record(62, 5, "makov-payne")
    with pytest.raises(ValueError, match="the vocabulary is"):
        checkpoint_m.capacity_record(62, 5, "box")


def test_capacity_inputs_are_type_checked(checkpoint_m):
    with pytest.raises(ValueError, match="n_k must be"):
        checkpoint_m.capacity_record(0, 5, "none")
    with pytest.raises(ValueError, match="table_dtype"):
        checkpoint_m.capacity_record(62, 5, "none", "bfloat16")


def test_contract_slices_are_immutable(checkpoint_m):
    with pytest.raises(Exception):
        checkpoint_m.kspace_cutoff = 4.0  # type: ignore[misc]
    frozen = copy.deepcopy(checkpoint_m)
    frozen.validate()
    assert frozen == checkpoint_m
