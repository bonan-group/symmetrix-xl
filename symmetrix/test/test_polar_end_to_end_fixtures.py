"""Structural validation of the MACE-POLAR-1 end-to-end oracle references.

The records under ``symmetrix/test/data/polar/e2e_*`` are the frozen torch
oracle's end-to-end values for the family's parity gate (Task 13 of
``plans/2026-10-14-mace-polar-1-support-v1.md``): energy, forces, stress,
dipole, and the chain's intermediate surfaces, on a molecular, a slab, and a
periodic fixture, in three channel settings, in fp32 and fp64, for POLAR-1-S
and POLAR-1-M.  They are captured out of tree by
``benchmarks/polar_end_to_end_oracle_runner.py``; this module validates them
without importing torch.

Three properties are checked here:

* provenance -- the oracle pin and the fixture set are exactly the runner's,
  and every record's committed bytes match its manifest digest;
* the oracle's own decomposition closes per record (its total energy is the
  backbone half plus the local-electron, electrostatic, and external-field
  terms), which is what makes the records usable as per-piece gates;
* the fp32 records are a nondeterministic realization: the manifest's recorded
  dtype spread, repeat spread, and repeat tolerance are consistent with the
  committed records, so an fp32 gate can be pinned honestly against them.

The environment-gated tests at the end compare the family's *evaluation*
against the records: the per-layer features the chain reads, the composed
energy, forces, dipole, and the relation between the served stress and the
recorded one (the record's stress is the oracle's edge-mediated virial, so it
omits the long-range terms' own strain derivative; the family's stress is its
own energy's strain derivative and the assertion pins the difference), plus
the family's finite-strain stress gate and the reverse tap's own gate.

One torch-free gate reads the frozen records as *derivative* references: the
k-space energy is a functional of the equilibrated density the frozen
chain-state references carry, so the portable kernels reproduce the recorded
`electrostatic_energy_kspace` from the recorded density alone -- and only
under the checkpoint's own self-interaction flag, which is what makes that
flag an observable of the energy gate rather than a declaration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import symmetrix
from polar_oracle_checkout import optional_checkout

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPOSITORY / "symmetrix/test/data/polar"
MANIFEST_PATH = FIXTURE_DIR / "e2e_manifest.json"

FIXTURE_SCHEMA = "symmetrix.polar.oracle-end-to-end"
FIXTURE_VERSION = 1

# Pinned oracle provenance -- must stay in lockstep with the runner constants
# (benchmarks/polar_end_to_end_oracle_runner.py); see the maintained model guide
# (docs/user/models.md#mace-polar-1).
ORACLE_MACE_SHA = "22f0809735bd4dd1deba80cf8e16f89913a35ff4"
ORACLE_MACE_REF = "v0.3.16-2-g22f08097"
ORACLE_GRAPH_ELECTROSTATICS_SHA = "e0eb8945c21e203656f4e0bbece5d1da488b6753"
ORACLE_GRAPH_LONGRANGE_VERSION = "0.4.4"
# These committed records are the historical reciprocal calculation, including
# their molecule/slab correction terms.  They are not the oracle for Task 5's
# new automatic FFF real-space route.
FROZEN_ELECTROSTATICS = "direct_reciprocal"

# The fixture set the runner declares: sizes x precisions x geometries x
# channels.  POLAR-1-L is a follow-up record (its 3 interactions are admitted,
# but the reference set covers the two sizes the plan gates first).
SIZES = ("S", "M")
PRECISIONS = ("float64", "float32")
GEOMETRIES = {
    "mol_h2o": ("molecule", 3),
    "slab_w2": ("slab", 6),
    "bulk_w4": ("bulk", 12),
}
#: The correction mode each fixture graph class selects, per the design note's
#: mixed_periodic matrix: the box term, the slab dipole term, and none.
CORRECTION_MODE = {
    "molecule": "makov-payne-dabo",
    "slab": "slab-dipole",
    "bulk": "none",
}
CHANNELS = {
    "singlet": (0.0, 1.0, (0.0, 0.0, 0.0), 0.0, None),
    "charged_triplet": (1.0, 3.0, (0.0, 0.0, 0.0), 0.0, None),
    "field": (0.0, 1.0, (0.1, -0.04, 0.02), 0.0, None),
    # The external-potential lane 0 probe: a nonzero fermi level, which the
    # pinned forward fills with zeros, captured on the molecule geometry.
    "fermi_lane": (0.0, 1.0, (0.0, 0.0, 0.0), 7.5, ("mol_h2o",)),
}

#: Every numerical surface a record carries, with the parity role it plays.
QUANTITIES = (
    "energy",
    "backbone_energy",
    "atomic_energy",
    "interaction_energy",
    "electron_energy",
    "electrostatic_energy",
    "electrostatic_energy_kspace",
    "electrostatic_energy_correction",
    "field_energy",
    "total_charge",
    "total_dipole",
    "forces",
    "stress",
    "charges",
    "spins",
    "density_coefficients",
    "spin_charge_density",
    "node_interaction_energies",
)

# The fp32 tolerance floor: the observed oracle fp32-versus-fp64 spread is the
# smallest honest fp32 gate, and it must stay below the bound the fixture set
# pins (a capture that drifted past it is a different oracle, not noise).
FP32_DTYPE_SPREAD_BOUND = {
    "energy_abs": 1.0e-2,
    "forces_max_abs": 1.0e-4,
    "stress_max_abs": 1.0e-7,
    "dipole_max_abs": 1.0e-5,
}


def _require_fixture_directory() -> None:
    # Binding resolution: the only clean-skip condition is an absent fixture
    # directory.  Once the directory exists, a missing or partial record is a
    # corruption failure, never a skip.
    if not FIXTURE_DIR.is_dir():
        pytest.skip(
            f"the MACE-POLAR-1 oracle fixture directory is absent: {FIXTURE_DIR}"
        )


def _load_manifest() -> dict:
    _require_fixture_directory()
    return json.loads(MANIFEST_PATH.read_text())


def _record_path(name: str) -> Path:
    path = FIXTURE_DIR / name
    assert path.is_file(), f"fixture record is missing: {path}"
    return path


def _load_record(name: str) -> dict:
    return json.loads(_record_path(name).read_text())


def _record_names(manifest: dict) -> list[str]:
    return [entry["name"] for entry in manifest["records"]]


def test_manifest_pins_the_frozen_oracle_and_the_fixture_set():
    manifest = _load_manifest()
    assert manifest["schema"] == FIXTURE_SCHEMA
    assert manifest["version"] == FIXTURE_VERSION
    assert manifest["oracle"] == {
        "mace_sha": ORACLE_MACE_SHA,
        "mace_ref": ORACLE_MACE_REF,
        "graph_electrostatics_sha": ORACLE_GRAPH_ELECTROSTATICS_SHA,
        "graph_longrange_version": ORACLE_GRAPH_LONGRANGE_VERSION,
    }
    assert [geometry["name"] for geometry in manifest["geometries"]] == list(GEOMETRIES)
    assert [channel["name"] for channel in manifest["channels"]] == list(CHANNELS)

    expected = {
        f"e2e_{size}_{geometry}_{channel}_{precision}.json"
        for size in SIZES
        for geometry in GEOMETRIES
        for channel, setting in CHANNELS.items()
        for precision in PRECISIONS
        if setting[4] is None or geometry in setting[4]
    }
    assert set(_record_names(manifest)) == expected
    assert len(manifest["records"]) == len(expected)


def test_every_record_matches_its_manifest_digest_and_header():
    manifest = _load_manifest()
    for entry in manifest["records"]:
        path = _record_path(entry["name"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        record = _load_record(entry["name"])
        assert record["size"] == entry["size"]
        assert record["dtype"] == entry["dtype"]
        assert record["geometry"] == entry["geometry"]
        assert record["graph_class"] == entry["graph_class"]
        assert record["channel"] == entry["channel"]
        assert (
            len(record["species"])
            == entry["n_atoms"]
            == GEOMETRIES[record["geometry"]][1]
        )
        assert record["graph_class"] == GEOMETRIES[record["geometry"]][0]
        assert record["model_sha256"] == entry["model_sha256"]
        assert (
            record["decomposition_tolerance"]
            == manifest["decomposition_tolerance"][record["dtype"]]
        )


def test_every_record_carries_the_parity_quantities():
    manifest = _load_manifest()
    for name in _record_names(manifest):
        record = _load_record(name)
        for quantity in QUANTITIES:
            assert quantity in record, f"{name} lacks {quantity}"
        assert len(record["forces"]) == len(record["species"])
        assert len(record["stress"]) == 6
        assert len(record["total_dipole"]) == 3
        assert len(record["spin_charge_density"]) == len(record["species"])
        assert len(record["spin_charge_density"][0]) == 2
        assert len(record["density_coefficients"]) == len(record["species"])


def test_the_oracle_decomposition_closes_per_record():
    manifest = _load_manifest()
    for name in _record_names(manifest):
        record = _load_record(name)
        tolerance = record["decomposition_tolerance"] * max(1.0, abs(record["energy"]))
        recomposed = (
            record["backbone_energy"]
            + record["electron_energy"]
            + record["electrostatic_energy"]
            + record["field_energy"]
        )
        assert abs(record["energy"] - recomposed) <= tolerance, name
        assert abs(record["decomposition_residual"]) <= tolerance, name
        assert abs(record["interaction_residual"]) <= tolerance, name
        # The backbone half is the atomic reference plus the interaction
        # energy, and the local-electron and electrostatic terms are the
        # chain's own energy kinds -- both recorded separately so a parity
        # failure localizes.
        assert (
            abs(
                record["atomic_energy"]
                - (record["backbone_energy"] - record["interaction_energy"])
            )
            <= tolerance
        ), name
        assert record["electron_energy"] != 0.0
        assert record["electrostatic_energy"] != 0.0


def test_the_energy_split_selects_the_correction_mode_of_the_graph_class():
    manifest = _load_manifest()
    for name in _record_names(manifest):
        record = _load_record(name)
        tolerance = record["decomposition_tolerance"] * max(
            1.0, abs(record["electrostatic_energy"])
        )
        assert (
            abs(
                record["electrostatic_energy_kspace"]
                + record["electrostatic_energy_correction"]
                - record["electrostatic_energy"]
            )
            <= tolerance
        ), name
        # mixed_periodic is k-space plus the correction the graph's pbc row
        # selects: the box term for a molecule, the slab dipole term for a
        # (T, T, F) slab, nothing for bulk.
        if CORRECTION_MODE[record["graph_class"]] == "none":
            assert record["electrostatic_energy_correction"] == 0.0, name
        else:
            assert record["electrostatic_energy_correction"] != 0.0, name
        assert record["electrostatic_energy_kspace"] != 0.0, name


def test_the_channel_settings_separate_the_input_lanes():
    for size in SIZES:
        for precision in PRECISIONS:
            for geometry in GEOMETRIES:
                records = {
                    channel: _load_record(
                        f"e2e_{size}_{geometry}_{channel}_{precision}.json"
                    )
                    for channel, setting in CHANNELS.items()
                    if setting[4] is None or geometry in setting[4]
                }
                # The charge/spin channels reach the equilibration targets and
                # the field lane reaches the external-field term: both change
                # the recorded dipole and energy, which is what makes the
                # records usable as the channel-mapping gate.
                assert records["singlet"]["field_energy"] == 0.0
                assert records["charged_triplet"]["field_energy"] == 0.0
                assert records["field"]["field_energy"] != 0.0
                assert (
                    records["field"]["total_dipole"]
                    != records["singlet"]["total_dipole"]
                )
                assert (
                    records["charged_triplet"]["energy"] != records["singlet"]["energy"]
                )
                assert (
                    records["charged_triplet"]["spins"] != records["singlet"]["spins"]
                )
                # A nonzero external field is the only difference between the
                # singlet and field records, and it moves the dipole.
                assert records["field"]["external_field"] == list(CHANNELS["field"][2])
                assert records["singlet"]["external_field"] == [0.0, 0.0, 0.0]
                assert records["singlet"]["fermi_level"] == 0.0
                if "fermi_lane" in records:
                    # The pinned forward builds the external potential as
                    # ``hstack((zeros_like(fermi_level), external_field))``
                    # (extensions.py:643-645), so the fermi lane is inert: a
                    # nonzero level leaves every recorded quantity unchanged.
                    assert records["fermi_lane"]["fermi_level"] == 7.5
                    assert (
                        records["fermi_lane"]["energy"] == records["singlet"]["energy"]
                    )
                    assert (
                        records["fermi_lane"]["total_dipole"]
                        == records["singlet"]["total_dipole"]
                    )
                    assert (
                        records["fermi_lane"]["forces"] == records["singlet"]["forces"]
                    )


def test_the_graph_classes_cover_the_correction_mode_matrix():
    manifest = _load_manifest()
    by_class = {geometry["graph_class"] for geometry in manifest["geometries"]}
    assert by_class == set(CORRECTION_MODE)
    expected_pbc = {
        "molecule": [False, False, False],
        "slab": [True, True, False],
        "bulk": [True, True, True],
    }
    for geometry in manifest["geometries"]:
        assert geometry["pbc"] == expected_pbc[geometry["graph_class"]]


def test_the_recorded_dtype_spread_is_reproducible_from_the_records():
    manifest = _load_manifest()
    recomputed = {}
    for name in _record_names(manifest):
        if not name.endswith("_float32.json"):
            continue
        single = _load_record(name)
        reference = _load_record(name.replace("_float32", "_float64"))
        recomputed[name] = {
            "energy_abs": abs(single["energy"] - reference["energy"]),
            "forces_max_abs": max(
                abs(left - right)
                for row_single, row_reference in zip(
                    single["forces"], reference["forces"]
                )
                for left, right in zip(row_single, row_reference)
            ),
            "stress_max_abs": max(
                abs(left - right)
                for left, right in zip(single["stress"], reference["stress"])
            ),
            "dipole_max_abs": max(
                abs(left - right)
                for left, right in zip(
                    single["total_dipole"], reference["total_dipole"]
                )
            ),
        }
    assert recomputed == manifest["dtype_spread"]
    assert recomputed, "the fixture set must carry fp32 records"
    for entry in recomputed.values():
        for quantity, bound in FP32_DTYPE_SPREAD_BOUND.items():
            assert entry[quantity] <= bound, (quantity, entry[quantity])


def test_the_fp32_repeat_measurement_is_within_its_declared_tolerance():
    manifest = _load_manifest()
    if not manifest["repeat_spread"]:
        pytest.skip(
            "the fixture set was captured without --verify-determinism, so no "
            "repeat measurement was recorded"
        )
    tolerance = manifest["repeat_tolerance"]
    # fp64 is bit-reproducible: the runner required byte-identical repeats, so
    # the tolerance is exactly zero and no fp64 record may appear here.
    assert tolerance["float64"] == dict.fromkeys(tolerance["float64"], 0.0)
    assert all(name.endswith("_float32.json") for name in manifest["repeat_spread"])
    for name, spread in manifest["repeat_spread"].items():
        declared = tolerance["float32"]
        for quantity, delta in spread.items():
            assert delta <= declared[quantity], (name, quantity, delta)


def test_the_manifest_digest_covers_the_committed_records():
    manifest = _load_manifest()
    fingerprint = hashlib.sha256()
    for entry in manifest["records"]:
        fingerprint.update(entry["sha256"].encode("utf-8"))
    assert fingerprint.hexdigest() == manifest["record_fingerprint"]


def test_the_adaptations_the_runner_applies_are_the_documented_ones():
    manifest = _load_manifest()
    assert manifest["adaptations"] == [
        "retired force_pbc_evaluator mode select dropped (inert for mixed_periodic)",
        "duplicated pbc argument dropped and the size-1 sigma axis squeezed in "
        "forward_dynamic",
        "derived module state restored after torch.load (pbc dispatch, features_dim)",
        "evaluated cell recorded (mace extends the cell along every non-periodic "
        "axis by (max|position| + 1) * 5 * cutoff; the structure's own box is "
        "kept as input_cell)",
    ]
    assert manifest["units"]["energy"] == "eV"
    assert manifest["units"]["length"] == "Angstrom"


# ---------------------------------------------------------------------------
# Environment-gated: the layer that executes today, measured against the gate
# ---------------------------------------------------------------------------

#: Tolerance of the backbone-half comparison, per dtype.  fp64 agrees to
#: ~8e-12 relative today (both sides are fp64 evaluations of the same math in
#: different accumulation orders); fp32's bound leaves room for the oracle's own
#: fp32 realization and for a different reduction order on this side.
BACKBONE_TOLERANCE = {"float64": 1.0e-9, "float32": 1.0e-6}

_ORACLE_MACE_CHECKOUT = optional_checkout("SYMMETRIX_POLAR_ORACLE_MACE")
_ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT = optional_checkout(
    "SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE"
)

try:  # pragma: no cover - environment-dependent import guard
    import sys as _sys

    if _ORACLE_MACE_CHECKOUT is not None and _ORACLE_MACE_CHECKOUT.is_dir():
        _sys.path.insert(0, str(_ORACLE_MACE_CHECKOUT))
    if (
        _ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT is not None
        and _ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT.is_dir()
    ):
        _sys.path.insert(0, str(_ORACLE_GRAPH_ELECTROSTATICS_CHECKOUT))
    import mace as _mace  # noqa: F401  (also sets the weights-only env e3nn needs)
    import torch as _torch
    from graph_longrange.gto_utils import gto_basis_kspace_cutoff  # noqa: F401

    _polar_oracle_import_error = None
except ImportError as _error:  # pragma: no cover - environment-dependent
    _mace = None
    _torch = None
    _polar_oracle_import_error = _error


def _checkpoint_path(size: str) -> Path:
    import os

    if _polar_oracle_import_error is not None:
        pytest.skip(
            "the reference comparison requires torch/e3nn/mace/graph_longrange: "
            f"{_polar_oracle_import_error}"
        )
    path = os.environ.get(f"SYMMETRIX_POLAR_MODEL_{size}")
    if not path:
        pytest.skip(
            f"set SYMMETRIX_POLAR_MODEL_{size} to compare the family's backbone "
            f"half against the MACE-POLAR-1-{size} oracle reference"
        )
    if not Path(path).is_file():
        pytest.skip(f"MACE-POLAR-1-{size} checkpoint does not exist: {path}")
    return Path(path)


#: The chain-state references the energy gate reads its density from: one per
#: (geometry, channel) pair the frozen end-to-end set also covers.
ENERGY_GATE_CASES = ("mol_h2o", "slab_w2", "bulk_w4")


def _chain_reference(geometry: str, channel: str = "singlet") -> dict:
    """The frozen chain-state record the end-to-end gate's density comes from."""

    path = FIXTURE_DIR / f"chain_M_{geometry}_{channel}_float64.npz"
    if not path.is_file():
        pytest.skip(f"the chain-state reference {path.name} is absent")
    with np.load(path) as arrays:
        return {
            "positions": np.ascontiguousarray(arrays["positions"]),
            "cell": np.ascontiguousarray(arrays["cell"]).reshape(-1, 3, 3),
            "node_graph_index": np.ascontiguousarray(
                arrays["node_graph_index"], dtype=np.int32
            ),
            "graph_pbc": np.ascontiguousarray(arrays["graph_pbc"], dtype=np.int32),
            "charge_density": np.ascontiguousarray(arrays["charge_density"]),
        }


@pytest.mark.parametrize("geometry", ENERGY_GATE_CASES)
def test_the_recorded_kspace_energy_needs_the_checkpoints_self_interaction_flag(
    geometry,
):
    """The frozen k-space energy, reproduced from the frozen density.

    The checkpoint's ``include_electrostatic_self_interaction`` is true, so the
    oracle's k-space energy is the pure reciprocal-space sum: feeding the
    recorded density to the portable kernels reproduces the recorded
    ``electrostatic_energy_kspace`` exactly.  Subtracting the self-interaction
    overlaps -- what a node stamped with the kind group's default flag would do
    -- moves the same functional by a factor, so the gate is sensitive to the
    flag it declares.
    """

    facts = json.loads((FIXTURE_DIR / "checkpoint_facts.json").read_text())
    model = facts["models"]["M"]
    assert model["include_electrostatic_self_interaction"] is True
    reference = _chain_reference(geometry)
    record = json.loads(
        (FIXTURE_DIR / f"e2e_M_{geometry}_singlet_float64.json").read_text()
    )
    recorded = float(record["electrostatic_energy_kspace"])
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=model["kspace_cutoff"],
        density_max_l=model["atomic_multipoles_max_l"],
        density_smearing_width=model["atomic_multipoles_smearing_width"],
        feature_max_l=model["field_feature_max_l"],
        feature_smearing_widths=list(model["field_feature_widths"]),
    )
    cache = ops.geometry_cache(
        reference["positions"],
        reference["cell"],
        reference["node_graph_index"],
    )["cache"]
    density = np.ascontiguousarray(reference["charge_density"])
    with_flag = np.asarray(ops.coulomb_energy(cache, density, None)).reshape(-1)
    assert with_flag.shape == (1,)
    # The measured agreement is 5.1e-8 relative on all three geometries: the
    # released checkpoints' derived GTO Fourier tables are fp32, so the
    # oracle's own fp64 k-space energy sits that far from an exact fp64
    # evaluation of the same closed forms (the chain-state gate's finding).
    # An order of magnitude above it is the honest bound.
    assert abs(with_flag[0] - recorded) <= 1e-6 * max(1.0, abs(recorded)), (
        geometry,
        with_flag[0],
        recorded,
    )
    subtracted = np.asarray(
        ops.coulomb_energy(cache, density, ops.energy_self_fields(density))
    ).reshape(-1)
    assert abs(subtracted[0] - recorded) > 1e-3 * max(1.0, abs(recorded)), (
        "the self-interaction flag must be observable in the energy gate",
        geometry,
        subtracted[0],
        recorded,
    )


#: Every (geometry, channel) pair the frozen end-to-end set covers; the dipole
#: gate reads each record's own equilibrated density and positions.
DIPOLE_CASES = tuple(
    (size, geometry, channel)
    for size in SIZES
    for geometry in ("mol_h2o", "slab_w2", "bulk_w4")
    for channel in ("singlet", "charged_triplet", "field")
)


def test_the_records_cell_is_the_evaluated_box():
    """The record's ``cell`` is the box mace evaluated in, not the input one.

    mace's neighborhood builder extends the cell along every non-periodic axis
    by ``(max|position| + 1) * 5 * cutoff`` and returns that cell, which is the one
    the model's volume, k-grid, k-space energy, and stress follow.  The rule is
    recomputed here from each record's own positions, the model's own cutoff,
    and its pbc row, so the record's ``cell`` is pinned to the mechanism rather
    than to a table; ``input_cell`` keeps the structure's declared box.  The
    fully periodic graph is the control: no axis is extended, so the two agree.
    """

    #: The evaluated cell is read off the model's own batch, so an fp32 record
    #: carries the fp32 image of the rule's value while an fp64 record carries
    #: it exactly; the tolerance follows the record's dtype.
    tolerance = {"float64": 1e-12, "float32": 1e-6}
    facts = json.loads((FIXTURE_DIR / "checkpoint_facts.json").read_text())
    manifest = _load_manifest()
    for name in _record_names(manifest):
        record = _load_record(name)
        model = facts["models"][record["size"]]
        positions = np.asarray(record["positions"], dtype=np.float64)
        cell = np.asarray(record["cell"], dtype=np.float64).reshape(3, 3)
        input_cell = np.asarray(record["input_cell"], dtype=np.float64).reshape(3, 3)
        pbc = np.asarray(record["pbc"], dtype=bool)
        # The periodic axes keep the structure's own vector; every non-periodic
        # axis is replaced by the extension length on its diagonal.
        extension = (float(np.abs(positions).max()) + 1.0) * 5.0 * float(model["r_cut"])
        for axis in range(3):
            if pbc[axis]:
                assert cell[axis, axis] == pytest.approx(input_cell[axis, axis])
                continue
            assert cell[axis, axis] == pytest.approx(
                extension, rel=tolerance[record["dtype"]]
            )
        if bool(np.all(pbc)):
            # The control: no axis is extended, so the evaluated box is the
            # structure's own (to the record's dtype).
            np.testing.assert_allclose(
                cell, input_cell, rtol=tolerance[record["dtype"]], atol=0.0
            )


@pytest.mark.parametrize("geometry", ENERGY_GATE_CASES)
def test_the_retained_layer_features_are_the_chains_frozen_input(geometry):
    """Item 2: the backbone's per-layer features on the evaluator's surface.

    The chain's input is the backbone's per-layer node features -- the product
    outputs the model's own per-layer readouts consume, which the feature
    mixer and the spin-source heads read back.  Retaining them is what gives
    the chain a producer on the evaluator's surface, and the frozen chain-state
    references carry the oracle's own values for the same structures, so the
    gate is a direct comparison: run the native backbone on the reference's
    geometry and compare the retained layers with the reference's
    ``node_features``.
    """

    from ase import Atoms

    checkpoint = _checkpoint_path("M")
    with np.load(FIXTURE_DIR / f"chain_M_{geometry}_singlet_float64.npz") as arrays:
        reference = np.ascontiguousarray(arrays["node_features"])
        positions = np.ascontiguousarray(arrays["positions"])
        cell = np.ascontiguousarray(arrays["cell"]).reshape(3, 3)
        pbc = np.asarray(arrays["graph_pbc"], dtype=bool).reshape(3)
    species = _load_record(f"e2e_M_{geometry}_singlet_float64.json")["species"]
    atoms = Atoms(species, positions=positions, cell=cell, pbc=pbc)
    calculator = symmetrix.Symmetrix(
        str(checkpoint),
        dtype="float64",
        polar_electrostatics=FROZEN_ELECTROSTATICS,
    )
    evaluator = calculator.evaluator
    # The family's frontend enables the retention when it binds its long-range
    # group (the chain consumes these features), so the family's own calculator
    # is the producer the gate checks through.
    assert evaluator.retains_layer_features is True
    assert evaluator.layer_features() == []
    atoms.calc = calculator
    atoms.get_potential_energy()
    layers = [
        np.asarray(layer, dtype=np.float64) for layer in evaluator.layer_features()
    ]
    assert len(layers) == 2
    assert {layer.shape[1] for layer in layers} == {reference.shape[1] // 2}
    concatenated = np.hstack(layers)
    assert concatenated.shape == reference.shape
    # The retained order is the model file's: the framework's l=1 node-feature
    # triple is the model's [1,2,0] permutation with all signs negated (the
    # conversion the retention applies), and the l=0 block is shared.
    # The measured agreement is 2.6e-8 (mol_h2o), 2.7e-8 (slab_w2), and
    # 3.5e-8 (bulk_w4) absolute on features of order one: the released
    # checkpoints' derived Fourier tables are fp32, so the oracle's own fp64
    # features sit that far from an exact fp64 evaluation of the same closed
    # forms (the same floor the chain-state and k-space gates report).  The
    # pin is an order of magnitude above it -- the chain-state gate's
    # ``feature_absolute`` convention.
    np.testing.assert_allclose(
        concatenated,
        reference,
        rtol=1e-6,
        atol=1e-6,
        err_msg=geometry,
    )
    evaluator.set_retain_layer_features(False)
    assert evaluator.layer_features() == []


def test_the_layer_product_seed_is_the_backbones_reverse_tap():
    """The retention's reverse twin: a consumer's product seed reaches the backbone.

    The retention copies each layer's product output for a consumer to read, so
    that consumer's derivative of what it read is a seed of the product.  The
    reverse tap adopts those seeds where the model's own per-layer readouts
    enter the product adjoint, which is what lets the polar family's chain take
    its share of the backbone's gradient *and* of the cross term through the
    retained features: positions move the products, the products move the
    chain, and only the backbone's own reverse carries that half.  The gate is
    the mechanism's own contract, independent of any family: for a
    deterministic seed of every layer, the forces of a re-run with that seed
    adopted move by exactly minus the position derivative of the seeded linear
    functional ``<seed, features>`` -- and clearing the seed restores the
    unseeded forces exactly.
    """

    from ase import Atoms

    checkpoint = _checkpoint_path("M")
    with np.load(FIXTURE_DIR / "chain_M_mol_h2o_singlet_float64.npz") as arrays:
        positions = np.ascontiguousarray(arrays["positions"], dtype=np.float64)
        cell = np.ascontiguousarray(arrays["cell"], dtype=np.float64).reshape(3, 3)
        pbc = np.asarray(arrays["graph_pbc"], dtype=bool).reshape(3)
    species = _load_record("e2e_M_mol_h2o_singlet_float64.json")["species"]
    atoms = Atoms(species, positions=positions, cell=cell, pbc=pbc)
    calculator = symmetrix.Symmetrix(
        str(checkpoint),
        dtype="float64",
        polar_electrostatics=FROZEN_ELECTROSTATICS,
    )
    evaluator = calculator.evaluator
    # The family's frontend enables the retention when it binds its long-range
    # group, so the forward tap (the features the seed is taken of) exists.
    assert evaluator.retains_layer_features is True
    atoms.calc = calculator
    atoms.get_potential_energy()
    layers = [
        np.asarray(layer, dtype=np.float64) for layer in evaluator.layer_features()
    ]
    generator = np.random.default_rng(1346)
    seed = [
        np.ascontiguousarray(generator.normal(size=layer.shape)) for layer in layers
    ]
    flattened = np.hstack(seed)

    def retained():
        return np.ascontiguousarray(
            np.hstack(
                [
                    np.asarray(layer, dtype=np.float64)
                    for layer in evaluator.layer_features()
                ]
            )
        )

    base = _backbone_forces(calculator, atoms)
    evaluator.set_layer_product_seed(seed)
    seeded = _backbone_forces(calculator, atoms)
    evaluator.clear_layer_product_seed()
    np.testing.assert_allclose(
        _backbone_forces(calculator, atoms),
        base,
        rtol=0.0,
        atol=1e-12,
        err_msg="clearing the adopted seeds restores the unseeded forces",
    )

    step = 1e-4
    numeric = np.zeros_like(positions)
    for node in range(len(positions)):
        for component in range(3):
            displaced = positions.copy()
            displaced[node, component] += step
            atoms.set_positions(displaced)
            atoms.get_potential_energy()
            forward = float(np.sum(flattened * retained()))
            displaced[node, component] -= 2.0 * step
            atoms.set_positions(displaced)
            atoms.get_potential_energy()
            backward = float(np.sum(flattened * retained()))
            numeric[node, component] = (forward - backward) / (2.0 * step)
    # The measured agreement is 4.8e-6 eV/A absolute on seeded force shifts of
    # order six (7.8e-7 relative): the central difference of the retained
    # features sits on the released checkpoints' fp32-derived Fourier tables,
    # the same floor every gate in this module reports.  The pin is an order of
    # magnitude above it.
    np.testing.assert_allclose(
        seeded - base,
        -numeric,
        rtol=1e-5,
        atol=1e-5,
        err_msg="the adopted seeds must move the forces by the seeded "
        "functional's own position derivative",
    )


def _backbone_forces(calculator, atoms):
    """The backbone half's own forces, with whatever seeds the evaluator holds."""

    mace_inputs = (
        calculator._mace_inputs(atoms, native_geometry=True)
        if calculator._can_use_native_geometry(["energy"])
        else calculator._mace_inputs(atoms)
    )
    generation = calculator._compute_mace(mace_inputs, atoms=atoms)
    return np.asarray(
        calculator._collect_mace_results(
            atoms, mace_inputs, ["energy", "forces"], generation
        )["forces"],
        dtype=np.float64,
    )


def test_the_recorded_kspace_energy_follows_the_records_own_cell():
    """Item 1: the k-space energy from each record's own cell.

    Before the re-freeze the records carried the structure's box while their
    numbers followed mace's extended one, so a molecule's k-space energy was
    8.3% and a slab's 16% away from the same functional evaluated in the box
    the record declared; the bulk control agreed because its cell is
    unextended.  With the evaluated cell recorded, every record's own cell
    reproduces its own k-space energy, and the bulk control still agrees.
    """

    facts = json.loads((FIXTURE_DIR / "checkpoint_facts.json").read_text())
    for size in SIZES:
        model = facts["models"][size]
        ops = symmetrix.PolarLongRangeOps(
            kspace_cutoff=model["kspace_cutoff"],
            density_max_l=model["atomic_multipoles_max_l"],
            density_smearing_width=model["atomic_multipoles_smearing_width"],
            feature_max_l=model["field_feature_max_l"],
            feature_smearing_widths=list(model["feature_field_widths"])
            if "feature_field_widths" in model
            else list(model["field_feature_widths"]),
        )
        for geometry in ("mol_h2o", "slab_w2", "bulk_w4"):
            record = _load_record(f"e2e_{size}_{geometry}_singlet_float64.json")
            positions = np.ascontiguousarray(record["positions"], dtype=np.float64)
            cell = np.asarray(record["cell"], dtype=np.float64).reshape(-1, 3, 3)
            node_graph_index = np.zeros(len(positions), dtype=np.int32)
            density = np.ascontiguousarray(
                np.asarray(record["spin_charge_density"], dtype=np.float64).sum(axis=1)
            )
            cache = ops.geometry_cache(positions, cell, node_graph_index)["cache"]
            kspace = float(
                np.asarray(ops.coulomb_energy(cache, density, None)).reshape(-1)[0]
            )
            recorded = float(record["electrostatic_energy_kspace"])
            assert abs(kspace - recorded) <= 1e-6 * max(1.0, abs(recorded)), (
                size,
                geometry,
                kspace,
                recorded,
            )


@pytest.mark.parametrize("size,geometry,channel", DIPOLE_CASES)
def test_the_recorded_dipole_is_reproduced_from_the_recorded_density(
    size, geometry, channel
):
    """The family's dipole output, from the frozen equilibrated density.

    The dipole is the readout's own functional of the spin-summed equilibrated
    density and the positions -- no cell, no k-space, no self-interaction
    policy -- so the portable kernels reproduce the oracle's ``total_dipole``
    from the record's own state alone.  The gate covers both sizes, all three
    graph classes, and all three channel settings, which is the parity
    evidence for the family's dipole output beyond the S/M backbone energies.
    """

    record = json.loads(
        (FIXTURE_DIR / f"e2e_{size}_{geometry}_{channel}_float64.json").read_text()
    )
    facts = json.loads((FIXTURE_DIR / "checkpoint_facts.json").read_text())
    model = facts["models"][size]
    positions = np.ascontiguousarray(record["positions"], dtype=np.float64)
    node_graph_index = np.zeros(len(positions), dtype=np.int32)
    density = np.ascontiguousarray(
        np.asarray(record["spin_charge_density"], dtype=np.float64).sum(axis=1)
    )
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=model["kspace_cutoff"],
        density_max_l=model["atomic_multipoles_max_l"],
        density_smearing_width=model["atomic_multipoles_smearing_width"],
        feature_max_l=model["field_feature_max_l"],
        feature_smearing_widths=list(model["field_feature_widths"]),
    )
    total_dipole = np.asarray(
        ops.dipole_readout(density, positions, node_graph_index)["total_dipole"]
    ).reshape(-1, 3)
    recorded = np.asarray(record["total_dipole"], dtype=np.float64).reshape(-1, 3)
    assert np.asarray(
        ops.dipole_readout(density, positions, node_graph_index)["total_charge"]
    ).shape == (1,)
    # The measured agreement is at the fp64 reassociation floor (1.1e-16
    # absolute on dipoles of order 1 e.A); 1e-12 absolute keeps three orders
    # of headroom while still failing any layout or phase-order mistake.
    np.testing.assert_allclose(
        total_dipole,
        recorded,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"{size}/{geometry}/{channel}",
    )


#: The channels the end-to-end gate drives the family through, as the frozen
#: records state them (charge, spin, field, fermi level), with the geometries
#: each channel's records cover (the fermi-lane probe is a molecule-only
#: record).
FAMILY_CHANNELS = {
    "singlet": (0.0, 1.0, (0.0, 0.0, 0.0), 0.0, GEOMETRIES),
    "charged_triplet": (1.0, 3.0, (0.0, 0.0, 0.0), 0.0, GEOMETRIES),
    "field": (0.0, 1.0, (0.1, -0.04, 0.02), 0.0, GEOMETRIES),
    "fermi_lane": (0.0, 1.0, (0.0, 0.0, 0.0), 7.5, ("mol_h2o",)),
}

#: The (channel, geometry) pairs the frozen set covers.
FAMILY_CASES = tuple(
    (channel, geometry)
    for channel, setting in FAMILY_CHANNELS.items()
    for geometry in setting[4]
)

#: Item 6: the family's own evaluation against the frozen references, per
#: quantity and dtype.  The fp64 bounds are an order of magnitude above the
#: measured agreement (6e-13 relative on the total energy, 1.7e-8 absolute on
#: the dipole, both sitting on the released checkpoints' fp32-derived Fourier
#: tables); the fp32 bounds cover the records' own dtype spread (2.6e-3 eV on
#: an 8.3 keV total, 1.9e-6 e.A on the dipole).
FAMILY_TOLERANCE = {
    # The relative bound is on the *total*; the absolute floor carries the
    # largest single term's own floor.  The slab's spin-polarized k-space term
    # is 66 eV, and the released checkpoints' fp32-derived Fourier tables put
    # ~5e-8 of *that* into it -- 3.3e-6 eV, which on a 4.1 keV total is
    # 8e-10 relative.  Measured worst case: 1.15e-9 relative (4.7e-6 eV on the
    # slab charged-triplet records, both sizes).
    "float64": {"energy_relative": 1e-9, "energy_absolute": 2e-5, "dipole": 1e-6},
    "float32": {"energy_relative": 1e-4, "energy_absolute": 2e-3, "dipole": 1e-4},
}

#: The family's force gate.  The composed gradient's floor is the same
#: fp32-derived Fourier tables: measured worst case 3.5e-7 eV/A at fp64 (the M
#: slab's charged triplet, on forces of order 2), and 3.2e-5 eV/A at fp32 --
#: which is the records' own dtype spread (the manifest's 4.2e-5 eV/A bound).
#: Both pins are an order of magnitude above the measured floors.
FAMILY_FORCE_TOLERANCE = {
    "float64": {"force_absolute": 5e-6},
    "float32": {"force_absolute": 5e-4},
}

#: The record-stress relation's tolerance.  The served stress differs from the
#: record by the composition's own tensor and nothing else; what is left over
#: is the record's own reproducibility floor, measured at 1.1e-10 eV/A^3 at
#: fp64 (worst case S/bulk_w4/field) and 1.6e-8 at fp32 (the fp32 records' own
#: spread).  Both pins are an order of magnitude above those floors.
FAMILY_STRESS_RELATION_TOLERANCE = {"float64": 1e-9, "float32": 1e-6}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("channel,geometry", FAMILY_CASES)
def test_the_family_evaluates_the_total_energy_force_and_dipole(
    size, geometry, channel
):
    """Item 6: the family's own evaluation, end to end, against the records.

    The family's evaluation is the backbone plus the chain and the long-range
    nodes, composed exactly as the model composes them: the retention's
    per-layer features drive the chain, the chain's equilibrated density drives
    the k-space energy, the pbc-selected analytic correction, and the dipole
    readout, and the graph-input channel supplies the equilibration targets and
    the external-field lanes.  The family's gradient is that composition's own:
    the chain's per-layer seeds go back into the backbone's reverse (the cross
    term positions own through the products) and the composition's position
    seeds are added to it.  Both sizes, all three graph classes, all four
    channel settings (including the provably inert fermi lane), and both dtypes
    are driven through the frontend's own calculator.

    The stress is checked for its *relation* to the record here, not for
    equality: the frozen records carry the oracle's stress, which is the
    edge-mediated virial (the backbone plus the feature-mediated part) and so
    omits the long-range terms' own strain derivative -- the family's stress
    differs from the record by exactly the composition's own tensor, and that
    is asserted below.  The authoritative stress gate is the family's own
    finite strain, in ``test_the_family_stress_is_its_own_energy_derivative``.
    """

    from ase import Atoms
    from ase.stress import full_3x3_to_voigt_6_stress

    checkpoint = _checkpoint_path(size)
    for dtype in PRECISIONS:
        record = _load_record(f"e2e_{size}_{geometry}_{channel}_{dtype}.json")
        charge, spin, field, fermi, _geometries = FAMILY_CHANNELS[channel]
        atoms = Atoms(
            record["species"],
            positions=np.array(record["positions"]),
            cell=np.array(record["cell"]).reshape(3, 3),
            pbc=record["pbc"],
        )
        atoms.info["electric_field"] = np.array(field, dtype=float)
        calculator = symmetrix.Symmetrix(
            str(checkpoint),
            dtype=dtype,
            total_charge=charge,
            total_spin=spin,
            fermi_level=fermi,
            polar_electrostatics=FROZEN_ELECTROSTATICS,
        )
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        dipole = atoms.get_dipole_moment()
        tolerance = FAMILY_TOLERANCE[dtype]
        assert abs(energy - record["energy"]) <= (
            tolerance["energy_absolute"]
            + tolerance["energy_relative"] * abs(record["energy"])
        ), (size, geometry, channel, dtype, energy, record["energy"])
        np.testing.assert_allclose(
            dipole,
            np.asarray(record["total_dipole"], dtype=np.float64).reshape(3),
            rtol=0.0,
            atol=tolerance["dipole"],
            err_msg=f"{size}/{geometry}/{channel}/{dtype}",
        )
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        reference_forces = np.asarray(record["forces"], dtype=np.float64)
        force_error = float(np.max(np.abs(forces - reference_forces)))
        assert force_error <= FAMILY_FORCE_TOLERANCE[dtype]["force_absolute"], (
            size,
            geometry,
            channel,
            dtype,
            force_error,
        )
        stress = np.asarray(atoms.get_stress(), dtype=np.float64)
        # Production returns only the packed public stress. The private tensor
        # remains available through an explicit diagnostic request so this
        # relation check cannot add transfers to ordinary EFS evaluation.
        calculator._materialize_polar_diagnostics()
        own = np.asarray(
            calculator._polar_composition["polar_stress"][0], dtype=np.float64
        )
        np.testing.assert_allclose(
            stress - np.asarray(record["stress"], dtype=np.float64),
            full_3x3_to_voigt_6_stress(own),
            rtol=0.0,
            atol=FAMILY_STRESS_RELATION_TOLERANCE[dtype],
            err_msg=(
                f"{size}/{geometry}/{channel}/{dtype}: the record's stress is "
                "the edge-mediated virial, so the family's stress (its own "
                "energy's strain derivative) differs from it by the long-range "
                "terms' own strain derivative and nothing else"
            ),
        )


def test_the_family_stress_is_its_own_energy_derivative():
    """The authoritative stress gate: the family's own finite strain.

    No analytic serial stress oracle ships for the long-range layer, so the
    stress is gated the way the OAM precedent gates it: six symmetric strain
    components about the record's evaluated cell, positions scaled with it,
    and the central difference of the family's *own* composed energy against
    the stress the frontend serves.  The check is the frontend's, so it
    exercises the whole path -- backbone, chain, k-space, corrections, and the
    cell adjoints.  Measured agreement: 9e-12 eV/A^3 on stresses of order
    3e-4, which is the fp32-derived-table floor of the differenced energies.
    """

    from ase import Atoms
    from ase.stress import voigt_6_to_full_3x3_stress

    checkpoint = _checkpoint_path("M")
    record = _load_record("e2e_M_bulk_w4_singlet_float64.json")
    positions = np.array(record["positions"], dtype=np.float64)
    cell = np.array(record["cell"], dtype=np.float64).reshape(3, 3)
    volume = abs(float(np.linalg.det(cell)))

    def family(displaced_positions, displaced_cell):
        atoms = Atoms(
            record["species"],
            positions=displaced_positions,
            cell=displaced_cell,
            pbc=record["pbc"],
        )
        atoms.calc = symmetrix.Symmetrix(
            str(checkpoint),
            dtype="float64",
            polar_electrostatics=FROZEN_ELECTROSTATICS,
        )
        return atoms

    atoms = family(positions, cell)
    served = np.asarray(atoms.get_stress(), dtype=np.float64)

    # The central difference's own truncation is O(step^2) on a total energy of
    # 2079 eV, so the step is small: measured relative agreement 1.2e-6 with
    # 1e-5 against 2.0e-6 with 1e-4 and 2.1e-4 with 1e-3 (the k-grid membership
    # is unchanged at every step, checked in the campaign that set these pins).
    step = 1e-5
    # The Voigt order ASE reads back, with the engineering shear the strain
    # tensor spells as two equal off-diagonal components.
    voigt_pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    numerical = []
    for first, second in voigt_pairs:
        strain = np.zeros(6)
        strain[voigt_pairs.index((first, second))] = step
        tensor = voigt_6_to_full_3x3_stress(strain)
        plus = np.eye(3) + tensor
        minus = np.eye(3) - tensor
        forward = float(family(positions @ plus, cell @ plus).get_potential_energy())
        backward = float(family(positions @ minus, cell @ minus).get_potential_energy())
        # The framework's stress is +(1/V) dE/deps, the convention the MH-1
        # virial reduction and the long-range cell adjoints both use.
        numerical.append((forward - backward) / (2.0 * step) / volume)
    # The finite difference's shear components carry the engineering factor of
    # two; the tensor's do not.
    numerical = np.asarray(numerical, dtype=np.float64)
    numerical[3:] *= 0.5
    np.testing.assert_allclose(
        served,
        numerical,
        rtol=1e-5,
        atol=1e-9,
        err_msg="the served stress must be the family's own strain derivative",
    )


@pytest.mark.parametrize("size", SIZES)
def test_the_backbone_half_is_parity_clean_against_the_oracle_reference(tmp_path, size):
    """The family's backbone half equals the oracle's recorded backbone energy.

    The family executes the MH-1 backbone and complete native polar
    composition.  This gate removes that composition through its explicit
    diagnostic surface, then compares the remaining backbone energy against
    the frozen oracle's atomic-reference-plus-interaction record.
    """

    import json as _json

    import numpy as np
    from ase import Atoms

    from symmetrix.calculator import Symmetrix
    from symmetrix.extract_mace_polar import extract_mace_polar_data

    manifest = _load_manifest()
    path = _checkpoint_path(size)
    model = _torch.load(path, map_location=_torch.device("cpu"), weights_only=False)
    data = extract_mace_polar_data(model.to(_torch.float64))
    model_path = tmp_path / f"MACE-POLAR-1-{size}.json"
    model_path.write_text(_json.dumps(data))

    calculators = {
        dtype: Symmetrix(
            str(model_path),
            dtype=dtype,
            streamed_edges="auto",
            polar_electrostatics=FROZEN_ELECTROSTATICS,
        )
        for dtype in PRECISIONS
    }
    for geometry in GEOMETRIES:
        for channel, setting in CHANNELS.items():
            if setting[4] is not None and geometry not in setting[4]:
                continue
            charge, spin, field = setting[0], setting[1], setting[2]
            for dtype in PRECISIONS:
                record = _load_record(f"e2e_{size}_{geometry}_{channel}_{dtype}.json")
                atoms = Atoms(
                    record["species"],
                    positions=np.array(record["positions"]),
                    cell=np.array(record["cell"]).reshape(3, 3),
                    pbc=record["pbc"],
                )
                calculator = calculators[dtype]
                calculator.parameters["total_charge"] = charge
                calculator.parameters["total_spin"] = spin
                calculator.parameters["electric_field"] = np.array(field)
                atoms.calc = calculator
                energy = atoms.get_potential_energy()
                # The family's evaluation is the model's, so the backbone half
                # is recovered by removing the composition's own terms: this
                # pins both that the backbone is still parity-clean and that the
                # composition adds exactly the terms it reports.
                # Production E/EF/EFS keeps the composed state resident.  This
                # historical decomposition check opts into the diagnostic
                # materialization explicitly rather than relying on a private
                # side effect of ordinary calculation.
                composition = calculator._materialize_polar_diagnostics()
                backbone = float(energy) - float(composition["energy"])
                tolerance = BACKBONE_TOLERANCE[dtype] * max(
                    1.0, abs(record["backbone_energy"])
                )
                assert abs(backbone - record["backbone_energy"]) <= tolerance, (
                    size,
                    dtype,
                    geometry,
                    channel,
                    backbone,
                    record["backbone_energy"],
                )
    assert manifest["records"]
