import gc
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import add_standard_base_contracts, foundation_r1_test_model
from symmetrix import (
    D2Correction,
    D3Correction,
    Symmetrix,
    selected_backend,
)
from symmetrix import symmetrix as native_symmetrix
from symmetrix.corrections import CorrectionPipeline

CONTRACT_ROOT = Path(__file__).resolve().parent / "data" / "execution_contracts"


@pytest.fixture()
def mace_les_json(tmp_path):
    contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_fixture_c4_e3_l0_contract.json").read_text()
    )
    model = add_standard_base_contracts(
        foundation_r1_test_model(contract, atomic_numbers=(7, 13))
    )
    model["model_type"] = "MACELES"
    model["les"] = {
        "schema": "symmetrix.model.mace_les",
        "version": 1,
        "kind": "monopole",
        "periodicity": "geometry",
        "ewald": {
            "sigma": 1.0,
            "dl": 2.0,
            "remove_self_interaction": True,
        },
        "output_scale": 0.25,
        "mace_num_interactions": 2,
        "mace_num_channels": 4,
        "mace_contract_fingerprints": {
            "R0": model["execution_contracts"]["R0"]["fingerprint"],
            "M0": model["execution_contracts"]["M0"]["semantic_fingerprint"],
            "R1": model["execution_contracts"]["R1"]["fingerprint"],
        },
        "readout_1_weights": [0.3, -0.2, 0.5, 0.1],
        "readout_2_hidden_size": 3,
        "readout_2_weights_1": [
            0.1,
            -0.2,
            0.3,
            0.4,
            -0.3,
            0.2,
            -0.1,
            0.5,
            0.25,
            0.15,
            -0.35,
            0.45,
        ],
        "readout_2_weights_2": [0.7, -0.4, 0.2],
        "readout_2_scale_factor": 1.5,
    }
    path = tmp_path / "mace-les.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path, model


@pytest.fixture(scope="module")
def standard_mace_les_json(tmp_path_factory):
    m0_contract = json.loads((CONTRACT_ROOT / "standard_m0_contract.json").read_text())
    m0_contract["output_l_max"] = 0
    m0_contract["output_components"] = 1
    m0_contract["monomial_groups"] = m0_contract["monomial_groups"][:1]
    m0_contract["structure_fingerprint"] = (
        "sha256:3b4ab2d979488798a05c07b5d54c4c96c1026da3ee1eb93069fd2bacec76ac90"
    )
    m0_contract["semantic_fingerprint"] = (
        "sha256:9fe598ceffb71e61d162c0afbdbe5727a0da7997dbca3a531a46b63603d7af85"
    )
    m0_contract["generation_fingerprint"] = (
        "sha256:f974e4af32dfb328e06fb6655b965c4cc5c0cad50a2013223630f9960f59d398"
    )
    r0_contract = json.loads((CONTRACT_ROOT / "standard_r0_contract.json").read_text())
    r1_contract = json.loads(
        (CONTRACT_ROOT / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
    model = foundation_r1_test_model(
        r1_contract,
        atomic_numbers=(7, 13),
        m0_contract=m0_contract,
        r0_contract=r0_contract,
    )
    model["A0_scaled"] = True
    model["compact_radial"]["networks"]["A0"] = {
        "shape": [r1_contract["radial_embedding"], 1],
        "weights": [[0.01] * r1_contract["radial_embedding"]],
        "activation": "silu",
        "activation_scale": 1.0,
        "postprocess": "tanh-square",
    }
    channels = r1_contract["channels"]
    hidden = 3
    model["model_type"] = "MACELES"
    model["les"] = {
        "schema": "symmetrix.model.mace_les",
        "version": 1,
        "kind": "monopole",
        "periodicity": "geometry",
        "ewald": {
            "sigma": 1.0,
            "dl": 2.0,
            "remove_self_interaction": True,
        },
        "output_scale": 0.25,
        "mace_num_interactions": 2,
        "mace_num_channels": channels,
        "mace_contract_fingerprints": {
            "R0": r0_contract["fingerprint"],
            "M0": m0_contract["semantic_fingerprint"],
            "R1": r1_contract["fingerprint"],
        },
        "readout_1_weights": [
            0.001 * ((channel % 7) - 3) for channel in range(channels)
        ],
        "readout_2_hidden_size": hidden,
        "readout_2_weights_1": [
            0.0001 * ((index % 11) - 5) for index in range(hidden * channels)
        ],
        "readout_2_weights_2": [0.7, -0.4, 0.2],
        "readout_2_scale_factor": 1.5,
    }
    path = tmp_path_factory.mktemp("standard-mace-les") / "model.json"
    path.write_text(json.dumps(model, separators=(",", ":")))
    return path, model


@pytest.fixture()
def atoms():
    return Atoms(
        ["N", "Al", "N"],
        positions=[[0.1, 0.2, 0.3], [1.4, 0.4, 0.2], [0.5, 1.6, 0.7]],
        cell=[[6.2, 0.2, 0.1], [0.0, 6.0, 0.3], [0.1, 0.0, 5.8]],
        pbc=False,
    )


def _calculator(model_path, corrections=()):
    return Symmetrix(
        model_path,
        dtype="float64",
        streamed_edges="reference",
        neighbor_skin=0.0,
        corrections=corrections,
    )


def test_maceles_selects_the_dedicated_native_evaluator(mace_les_json):
    model_path, _ = mace_les_json

    calculator = Symmetrix(model_path, dtype="float64", streamed_edges="reference")
    evaluator_type_name = type(calculator.evaluator).__name__
    del calculator
    gc.collect()

    assert evaluator_type_name == "MACELESKokkos"


def test_maceles_evaluates_complete_model_without_correction_argument(
    mace_les_json, atoms
):
    model_path, _ = mace_les_json
    calc = _calculator(model_path)
    atoms.calc = calc

    energy = atoms.get_potential_energy()

    assert energy == pytest.approx(calc.results["energy"])
    assert calc.results["mace_les_energy"] == pytest.approx(energy)
    assert calc.results["les_energy"] != 0.0
    assert calc.results["latent_charges"].shape == (len(atoms),)
    assert calc.evaluator.is_mace_les
    assert calc.correction_diagnostics == ()
    assert calc.evaluator.les_diagnostics["schema_version"] == 1
    assert calc.les_diagnostics["schema_version"] == 1
    atoms.calc = None
    del calc
    gc.collect()


def test_mace_les_monopole_diagnostics_expose_workspace_mapping(mace_les_json, atoms):
    model_path, model = mace_les_json
    atoms.pbc = True
    calc = _calculator(model_path)
    atoms.calc = calc
    try:
        atoms.get_potential_energy()
        diagnostics = calc.evaluator.les_monopole_diagnostics

        dl = model["les"]["ewald"]["dl"]
        reciprocal_counts = [
            max(1, int(np.linalg.norm(vector) / dl)) for vector in atoms.cell
        ]
        candidate_count = int(np.prod([2 * count + 1 for count in reciprocal_counts]))
        logical_bytes = (1 + 5 * len(atoms) + 9 + 10 + 7 * candidate_count) * 8
        logical_bytes += candidate_count
        assert {
            "reciprocal_candidate_count",
            "reciprocal_vector_count",
            "logical_extent_bytes",
            "allocated_capacity_bytes",
            "active_index_capacity_bytes",
            "index_pack_count",
            "table_rebuilds",
            "geometry_updates",
            "active_bytes",
            "capacity_bytes",
        } <= diagnostics.keys()
        assert diagnostics["reciprocal_candidate_count"] == candidate_count
        assert diagnostics["reciprocal_vector_count"] > 0
        index_size = np.dtype(np.uintp).itemsize
        logical_bytes += diagnostics["reciprocal_vector_count"] * index_size
        assert diagnostics["logical_extent_bytes"] == logical_bytes
        assert diagnostics["allocated_capacity_bytes"] == (
            logical_bytes
            + (candidate_count - diagnostics["reciprocal_vector_count"]) * index_size
        )
        assert (
            diagnostics["active_index_capacity_bytes"] == candidate_count * index_size
        )
        assert diagnostics["index_pack_count"] == 1
        assert diagnostics["active_bytes"] == diagnostics["logical_extent_bytes"]
        assert diagnostics["capacity_bytes"] == diagnostics["allocated_capacity_bytes"]
        assert diagnostics["table_rebuilds"] == 1
        assert diagnostics["geometry_updates"] == 1
    finally:
        atoms.calc = None
        del calc
        gc.collect()


def test_maceles_always_declares_and_publishes_model_outputs(mace_les_json, atoms):
    model_path, _ = mace_les_json
    atoms.calc = _calculator(model_path)

    atoms.get_potential_energy()

    calc = atoms.calc
    assert {"les_energy", "mace_les_energy", "latent_charges"} <= set(
        calc.implemented_properties
    )
    assert {"les_energy", "mace_les_energy", "latent_charges"} <= set(calc.results)


def test_maceles_outputs_are_declared_ase_properties(mace_les_json, atoms):
    model_path, _ = mace_les_json
    calc = _calculator(model_path)

    les_energy = calc.get_property("les_energy", atoms)
    mace_les_energy = calc.get_property("mace_les_energy", atoms)
    latent_charges = calc.get_property("latent_charges", atoms)

    assert "les_energy" in calc.implemented_properties
    assert "mace_les_energy" in calc.implemented_properties
    assert "latent_charges" in calc.implemented_properties
    assert les_energy == pytest.approx(calc.results["les_energy"])
    assert mace_les_energy == pytest.approx(calc.results["mace_les_energy"])
    np.testing.assert_array_equal(latent_charges, calc.results["latent_charges"])


def test_les_geometry_is_refreshed_after_positions_and_cell_change(
    mace_les_json, atoms
):
    model_path, _ = mace_les_json
    atoms.calc = _calculator(model_path)
    initial_energy = atoms.get_potential_energy()
    initial_charges = atoms.calc.results["latent_charges"].copy()

    atoms.positions[1, 0] += 0.13
    moved_energy = atoms.get_potential_energy()
    moved_charges = atoms.calc.results["latent_charges"].copy()
    atoms.cell[0, 0] += 0.2
    cell_energy = atoms.get_potential_energy()
    atoms.pbc = True
    periodic_energy = atoms.get_potential_energy()
    periodic_forces = atoms.get_forces()
    periodic_charges = atoms.calc.results["latent_charges"].copy()

    fresh_atoms = atoms.copy()
    fresh_atoms.calc = _calculator(model_path)
    fresh_energy = fresh_atoms.get_potential_energy()
    fresh_forces = fresh_atoms.get_forces()

    assert moved_energy != pytest.approx(initial_energy)
    assert not np.allclose(moved_charges, initial_charges)
    assert np.isfinite(cell_energy)
    assert periodic_energy != pytest.approx(cell_energy)
    assert periodic_energy == pytest.approx(fresh_energy)
    np.testing.assert_allclose(periodic_forces, fresh_forces)
    np.testing.assert_allclose(
        periodic_charges, fresh_atoms.calc.results["latent_charges"]
    )


def test_les_rejects_mixed_periodicity_at_evaluation(mace_les_json, atoms):
    model_path, _ = mace_les_json
    atoms.pbc = [True, False, True]
    atoms.calc = _calculator(model_path)

    with pytest.raises(ValueError, match="all false or all true"):
        atoms.get_potential_energy()


def test_les_rejects_multi_structure_input(mace_les_json, atoms):
    model_path, _ = mace_les_json
    calc = _calculator(model_path)

    with pytest.raises(ValueError, match="one ASE Atoms structure"):
        calc.calculate([atoms, atoms])


def test_ordinary_mace_has_no_maceles_properties(mace_les_json, atoms):
    _, model = mace_les_json
    model.pop("les")
    model["model_type"] = "MACE"
    path = Path(mace_les_json[0]).with_name("plain-mace.json")
    path.write_text(json.dumps(model, separators=(",", ":")))
    calc = _calculator(path)

    calc.calculate(atoms, properties=["energy"])

    assert not hasattr(calc.evaluator, "is_mace_les")
    assert {"les_energy", "mace_les_energy", "latent_charges"}.isdisjoint(
        calc.implemented_properties
    )
    assert {"les_energy", "mace_les_energy", "latent_charges"}.isdisjoint(calc.results)
    assert calc.les_diagnostics is None


def test_les_direct_execution_constructs(mace_les_json, monkeypatch, tmp_path):
    model_path, _ = mace_les_json
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))

    calculator = Symmetrix(
        model_path,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="speed",
        neighbor_skin=0.0,
    )

    assert calculator.streamed_edges == "direct"
    assert calculator.evaluator.is_mace_les


@pytest.mark.parametrize("periodic", [False, True], ids=["nonperiodic", "periodic"])
@pytest.mark.parametrize("execution_profile", ["speed", "capacity"])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        ("float32", 2.0e-4, 5.0e-4),
        ("float64", 2.0e-10, 2.0e-10),
    ],
)
def test_les_direct_required_jit_matches_reference_without_graph_rebuild(
    standard_mace_les_json,
    atoms,
    monkeypatch,
    tmp_path_factory,
    periodic,
    execution_profile,
    dtype,
    rtol,
    atol,
):
    model_path, _ = standard_mace_les_json
    atoms.pbc = periodic
    jit_cache = tmp_path_factory.getbasetemp() / "mace-les-direct-jit"
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(jit_cache))
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    properties = ["energy", "energies", "forces", "stress"]

    reference = Symmetrix(
        model_path,
        dtype=dtype,
        streamed_edges="reference",
        neighbor_skin=0.0,
    )
    direct = Symmetrix(
        model_path,
        dtype=dtype,
        streamed_edges="direct",
        execution_profile=execution_profile,
        neighbor_skin=0.0,
    )

    reference.calculate(atoms, properties=properties)
    direct.calculate(atoms, properties=properties)
    graph_count = direct.evaluator.factorized_prepared_graph_count
    initial_les_energy = float(direct.results["les_energy"])
    assert initial_les_energy == pytest.approx(
        np.asarray(direct.evaluator.les_atomic_energies).sum(),
        rel=rtol,
        abs=atol,
    )

    moved = atoms.copy()
    moved.positions[1, 0] += 0.01
    reference.calculate(moved, properties=properties)
    direct.calculate(moved, properties=properties)

    assert direct.streamed_edges == "direct"
    assert (direct.jit_status, direct.jit_origin) in {
        ("built", "runtime"),
        ("cached", "cache"),
        ("loaded", "prebuilt"),
    }
    assert Path(direct.jit_artifact_path).is_file()
    assert direct.evaluator.factorized_fallback_evaluation_count == 0
    assert direct.evaluator.factorized_prepared_graph_count == graph_count
    if execution_profile == "capacity":
        assert direct.evaluator.low_memory_requested
        assert direct.execution_plan["requested_profile"] == "capacity"
        selected_plan = direct.execution_plan["selected_id"]
        if selected_plan == "mh0-direct-speed":
            assert selected_backend()["backend"] in {"cuda", "hip"}
            assert direct.execution_plan["selection_reason"].startswith(
                "speed selection:"
            )
        else:
            assert selected_plan in {
                "mh0-direct-capacity-retained",
                "mh0-direct-capacity-y-only",
            }
    else:
        assert direct.execution_plan["selected_id"] == "mh0-direct-speed"
    for property_name in ("energies", "forces", "stress", "latent_charges"):
        np.testing.assert_allclose(
            direct.results[property_name],
            reference.results[property_name],
            rtol=rtol,
            atol=atol,
        )
    assert direct.results["les_energy"] == pytest.approx(
        reference.results["les_energy"], rel=rtol, abs=atol
    )
    assert direct.results["les_energy"] == pytest.approx(
        np.asarray(direct.evaluator.les_atomic_energies).sum(),
        rel=rtol,
        abs=atol,
    )
    assert direct.results["les_energy"] != pytest.approx(
        initial_les_energy, rel=0.0, abs=1.0e-12
    )
    assert direct.results["energies"].sum() == pytest.approx(
        direct.results["energy"], rel=rtol, abs=atol
    )


def test_les_direct_compact_stress_matches_reference_and_affine_difference(
    standard_mace_les_json,
    atoms,
    monkeypatch,
    tmp_path,
):
    _, model = standard_mace_les_json
    model = json.loads(json.dumps(model))
    # Give the explicit reciprocal strain term enough weight to distinguish it
    # from float32 compact-geometry roundoff in this small synthetic system.
    model["les"]["output_scale"] *= 10_000.0
    model_path = tmp_path / "stress-model.json"
    model_path.write_text(json.dumps(model, separators=(",", ":")))
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    atoms.pbc = True
    properties = ["energy", "energies", "forces", "stress"]

    reference = Symmetrix(
        model_path,
        dtype="float32",
        streamed_edges="reference",
        neighbor_skin=0.5,
    )
    direct = Symmetrix(
        model_path,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
    )
    reference.calculate(atoms, properties=properties)
    direct.calculate(atoms, properties=properties)

    assert direct.evaluator.edge_geometry_policy == "unit-f32-radius-f64-v1"
    np.testing.assert_allclose(
        direct.results["stress"],
        reference.results["stress"],
        rtol=3.0e-4,
        atol=1.0e-7,
    )

    step = 5.0e-4
    strained_energies = []
    for sign in (1.0, -1.0):
        strained = atoms.copy()
        deformation = np.eye(3)
        deformation[0, 0] += sign * step
        strained.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
        direct.calculate(strained, properties=["energy"])
        strained_energies.append(direct.results["energy"])
    numeric_stress = (strained_energies[0] - strained_energies[1]) / (
        2.0 * step * atoms.get_volume()
    )
    direct.calculate(atoms, properties=properties)
    assert direct.results["stress"][0] == pytest.approx(
        numeric_stress, rel=3.0e-3, abs=5.0e-6
    )


def test_les_direct_observer_uses_prepared_space_and_publishes_fresh_results(
    standard_mace_les_json,
    atoms,
    monkeypatch,
    tmp_path,
):
    model_path, _ = standard_mace_les_json
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(tmp_path / "jit-cache"))
    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    atoms.pbc = True
    properties = ["energy", "energies", "forces", "stress"]
    direct = Symmetrix(
        model_path,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="speed",
        neighbor_skin=0.5,
    )
    direct.evaluator._set_factorized_observer(True)
    reference = None
    execution_space_diagnostics = []
    try:
        moved = atoms.copy()
        moved.positions[1, 0] += 0.01
        scaled = moved.copy()
        scaled.set_cell(moved.cell.array * 1.01, scale_atoms=True)
        states = (atoms, moved, scaled)
        graph_generation = None
        graph_count = None
        table_rebuilds = []
        workspace_reuses = []
        published_energies = []
        published_forces = []

        for index, state in enumerate(states):
            reference = Symmetrix(
                model_path,
                dtype="float64",
                streamed_edges="reference",
                neighbor_skin=0.5,
            )
            reference.calculate(state, properties=properties)
            direct.calculate(state, properties=properties)

            execution_space_diagnostics.append(
                (
                    getattr(
                        reference.evaluator,
                        "_les_execution_space_diagnostics",
                        None,
                    ),
                    getattr(
                        direct.evaluator,
                        "_les_execution_space_diagnostics",
                        None,
                    ),
                )
            )
            observer = direct.evaluator._factorized_observer()
            assert observer["enabled"]
            assert observer["ready"]

            for name in ("energies", "forces", "stress", "latent_charges"):
                np.testing.assert_allclose(
                    direct.results[name],
                    reference.results[name],
                    rtol=2.0e-10,
                    atol=2.0e-10,
                )
            assert direct.results["les_energy"] == pytest.approx(
                reference.results["les_energy"], rel=2.0e-10, abs=2.0e-10
            )
            assert direct.evaluator.factorized_fallback_evaluation_count == 0
            current_generation = direct.evaluator.factorized_graph_generation
            current_graph_count = direct.evaluator.factorized_prepared_graph_count
            if index == 0:
                graph_generation = current_generation
                graph_count = current_graph_count
            else:
                assert current_generation == graph_generation
                assert current_graph_count == graph_count
            published_energies.append(float(direct.results["energy"]))
            published_forces.append(direct.results["forces"].copy())
            table_rebuilds.append(
                direct.evaluator.les_monopole_diagnostics["table_rebuilds"]
            )
            workspace_reuses.append(
                direct.evaluator.les_charge_workspace_diagnostics["reuse_count"]
            )

        assert table_rebuilds[1] == table_rebuilds[0]
        assert table_rebuilds[2] == table_rebuilds[1] + 1
        assert workspace_reuses[1] > workspace_reuses[0]
        assert workspace_reuses[2] > workspace_reuses[1]
        assert len(set(published_energies)) == len(states)
        assert all(
            not np.array_equal(previous, current)
            for previous, current in zip(published_forces, published_forces[1:])
        )
    finally:
        direct.evaluator._set_factorized_observer(False)
        assert not direct.evaluator.factorized_observer_enabled
        assert not direct.evaluator.factorized_observer_ready
        assert direct.evaluator.factorized_observer_bytes == 0
        del reference, direct
        gc.collect()

    assert execution_space_diagnostics == [
        (
            {"geometry": "default", "evaluation": "default"},
            {"geometry": "factorized", "evaluation": "factorized"},
        )
    ] * len(states)


def test_maceles_cannot_be_evaluated_without_its_les_term(mace_les_json, atoms):
    model_path, _ = mace_les_json
    atoms.calc = _calculator(model_path)

    energy = atoms.get_potential_energy()

    assert np.isfinite(energy)
    assert atoms.calc.evaluator.is_mace_les
    assert atoms.calc.results["mace_les_energy"] == pytest.approx(energy)
    assert "les_energy" in atoms.calc.results
    assert "latent_charges" in atoms.calc.results


def _energy_forces(model_path, atoms, corrections):
    calc = _calculator(model_path, corrections)
    calc.calculate(atoms, properties=["energy", "forces"])
    return calc, float(calc.results["energy"]), calc.results["forces"].copy()


@pytest.mark.parametrize(
    "additive",
    [
        pytest.param(D2Correction(cutoff=4.0, skin=0.0), id="d2"),
        pytest.param(
            D3Correction(cutoff=4.0, coordination_cutoff=3.5, skin=0.0),
            id="d3",
        ),
    ],
)
def test_maceles_plus_additive_correction_composes_energy_and_forces_once(
    mace_les_json, atoms, additive
):
    model_path, _ = mace_les_json
    complete_calc, complete_energy, complete_forces = _energy_forces(
        model_path, atoms, []
    )
    additive_results = CorrectionPipeline(
        [additive], native=native_symmetrix, dtype=np.float64
    ).calculate(atoms, ["energy", "forces"])
    combined_calc, combined_energy, combined_forces = _energy_forces(
        model_path, atoms, [additive]
    )

    assert combined_energy == pytest.approx(
        complete_energy + additive_results["energy"]
    )
    np.testing.assert_allclose(
        combined_forces,
        complete_forces + additive_results["forces"],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert combined_calc.results["mace_les_energy"] == pytest.approx(complete_energy)
    assert combined_calc.les_diagnostics == complete_calc.les_diagnostics
    expected_method = "d2" if isinstance(additive, D2Correction) else "d3"
    assert [item["method"] for item in combined_calc.correction_diagnostics] == [
        expected_method,
    ]
    if isinstance(additive, D3Correction):
        assert combined_energy == pytest.approx(
            combined_calc.results["mace_les_energy"]
            + combined_calc.results["d3_energy"]
        )
