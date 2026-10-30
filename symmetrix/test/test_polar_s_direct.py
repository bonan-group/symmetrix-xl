"""Generated scalar-only, single-interaction Polar-S backbone execution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import symmetrix
from symmetrix.extract_mace_polar import extract_mace_polar_data
from symmetrix.mace_polar_execution_contract import (
    make_execution_polar_mh1_v4_contract,
    validate_execution_polar_mh1_v4_contract_for_model,
)
from test_mace_polar_backbone import _build_polar_shaped_model


@pytest.fixture(scope="module")
def scalar_polar_model(tmp_path_factory):
    model = _build_polar_shaped_model(num_interactions=1, hidden_irreps="4x0e")
    import torch
    from e3nn import o3
    from mace.modules.blocks import LinearReadoutBlock

    # The released S checkpoint has a linear energy readout, unlike M's
    # nonlinear bias readout. Keep that distinction in the small fixture.
    model.readouts[0] = LinearReadoutBlock(o3.Irreps("4x0e")).to(torch.float64)
    data = extract_mace_polar_data(model)
    path = tmp_path_factory.mktemp("polar-s-direct") / "model.json"
    path.write_text(json.dumps(data))
    return path, data


def _atoms():
    from ase import Atoms

    return Atoms(
        numbers=[14, 1, 1],
        positions=[[0.0, 0.0, 0.0], [1.7, 0.2, 0.1], [0.3, 1.8, 0.4]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )


def test_scalar_polar_contract_generates_one_layer_with_linear_readout(
    scalar_polar_model,
):
    """A final linear readout must not remove S's generated node program."""

    _, data = scalar_polar_model
    contract = make_execution_polar_mh1_v4_contract(data)
    assert contract["interaction_count"] == 1
    layer = contract["node_program"]["layers"][0]
    assert layer["output_irreps"] == "4x0e"
    assert layer["readout"]["class"] == "LinearReadoutBlock"
    assert contract["node_program"]["execution"]["forward_phases"] == [
        "prepare_layer_0",
        "finish_layer_0",
    ]
    assert contract["node_program"]["execution"]["reverse_phases"] == [
        "reverse_layer_0"
    ]
    assert (
        validate_execution_polar_mh1_v4_contract_for_model(data, contract) == contract
    )


def test_scalar_polar_native_admission_preserves_family_identity(scalar_polar_model):
    """The Polar-S exception must not admit ordinary scalar-final MH-1 models."""

    _, data = scalar_polar_model
    native = symmetrix.load_backend()
    descriptor = native._analyze_mh1_family_architecture(json.dumps(data))
    assert descriptor["compatible"], descriptor["rejection_reason"]
    ordinary = dict(data, model_type="MACE_Nonlinear")
    ordinary.pop("polar")
    assert not native._analyze_mh1_family_architecture(json.dumps(ordinary))[
        "compatible"
    ]
    from symmetrix.mace_nonlinear_execution_contract import (
        make_execution_mh1_v4_contract,
    )

    with pytest.raises(ValueError, match="nonlinear or bias readout for the last"):
        make_execution_mh1_v4_contract(ordinary)


@pytest.mark.parametrize(
    "field,signature", (("irreps_in", "4x0o"), ("irreps_out", "1x0o"))
)
def test_scalar_polar_admission_rejects_inconsistent_linear_readout(
    scalar_polar_model, field, signature
):
    """The linear final-readout exception must still enforce its scalar basis."""

    import copy

    _, data = scalar_polar_model
    invalid = copy.deepcopy(data)
    invalid["readouts"][0]["linear"][field] = signature
    descriptor = symmetrix.load_backend()._analyze_mh1_family_architecture(
        json.dumps(invalid)
    )
    assert not descriptor["compatible"]
    with pytest.raises(ValueError):
        make_execution_polar_mh1_v4_contract(invalid)


@pytest.mark.parametrize("dtype", ("float64", "float32"))
@pytest.mark.parametrize(
    "policy",
    ("full-retention-v1", "recompute-v1", "reuse-adjoints-v1", "retain-interaction-v1"),
)
def test_scalar_polar_generated_transaction_matches_reference(
    scalar_polar_model, dtype, policy
):
    """Generated execution includes the Polar chain's scalar-feature reverse seed."""

    native = symmetrix.load_backend()
    device = native._kokkos_default_execution_space() in ("Cuda", "HIP")
    if policy == "reuse-adjoints-v1" and not device:
        pytest.skip("reuse-adjoints-v1 requires a generated CUDA or HIP module")
    path, _ = scalar_polar_model
    properties = ["energy", "forces", "stress", "dipole"]
    calculators = {
        route: symmetrix.Symmetrix(
            str(path),
            dtype=dtype,
            streamed_edges=route,
            execution_profile="speed",
            execution_mh1_node_state_policy=policy,
            polar_electrostatics="direct_reciprocal",
            total_charge=1.0,
            total_spin=3.0,
            electric_field=[0.1, -0.04, 0.02],
        )
        for route in ("reference", "direct")
    }
    for calculator in calculators.values():
        calculator.calculate(_atoms(), properties=properties)
        assert calculator.evaluator.factorized_fallback_evaluation_count == 0
    direct = calculators["direct"]
    assert direct.jit_status in ("built", "cached")
    assert direct.evaluator.uses_mh1_fast_path
    assert (
        direct.evaluator.jit_mh1_cuda_plugin_v5_ready
        if device
        else direct.evaluator.jit_mh1_host_plugin_v7_ready
    )
    assert direct.evaluator._execution_mh1_product_module_forward_launch_count(0) > 0
    assert direct.evaluator._execution_mh1_product_module_reverse_launch_count(0) > 0
    for name in properties:
        # Use the frozen Polar oracle's existing FP32 force bound. Its FP64
        # comparison below also bounds each generated result independently
        # of the reference route's own FP32 accumulation roundoff.
        atol = (
            5e-4
            if dtype == "float32" and name == "forces"
            else (2e-5 if dtype == "float32" else 2e-10)
        )
        np.testing.assert_allclose(
            direct.results[name],
            calculators["reference"].results[name],
            rtol=2e-6 if dtype == "float32" else 0.0,
            atol=atol,
            err_msg=name,
        )
    if dtype == "float32":
        double_reference = symmetrix.Symmetrix(
            str(path),
            dtype="float64",
            streamed_edges="reference",
            polar_electrostatics="direct_reciprocal",
            total_charge=1.0,
            total_spin=3.0,
            electric_field=[0.1, -0.04, 0.02],
        )
        double_reference.calculate(_atoms(), properties=["forces"])
        np.testing.assert_allclose(
            direct.results["forces"],
            double_reference.results["forces"],
            rtol=0.0,
            atol=5e-4,
        )
    builds = direct.evaluator.factorized_prepared_graph_count
    direct.calculate(_atoms(), properties=properties)
    assert direct.evaluator.factorized_prepared_graph_count == builds


def test_scalar_polar_generated_forces_and_stress_are_energy_derivatives(
    scalar_polar_model,
):
    """Catch a missing scalar seed even if both execution routes share a defect."""

    path, _ = scalar_polar_model
    calculator = symmetrix.Symmetrix(
        str(path),
        dtype="float64",
        streamed_edges="direct",
        polar_electrostatics="direct_reciprocal",
        total_charge=1.0,
        total_spin=3.0,
    )
    atoms = _atoms()
    calculator.calculate(atoms, properties=["energy", "forces", "stress"])
    forces = np.array(calculator.results["forces"], copy=True)
    stress = np.array(calculator.results["stress"], copy=True)
    step = 1e-5
    for atom, axis in ((0, 0), (1, 2)):
        energies = []
        for displacement in (step, -step):
            displaced = atoms.copy()
            displaced.positions[atom, axis] += displacement
            calculator.calculate(displaced, properties=["energy"])
            energies.append(float(calculator.results["energy"]))
        assert forces[atom, axis] == pytest.approx(
            -(energies[0] - energies[1]) / (2 * step), abs=1e-7
        )
    energies = []
    for strain in (step, -step):
        displaced = atoms.copy()
        deformation = np.diag([1.0 + strain, 1.0, 1.0])
        displaced.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
        calculator.calculate(displaced, properties=["energy"])
        energies.append(float(calculator.results["energy"]))
    assert stress[0] == pytest.approx(
        (energies[0] - energies[1]) / (2 * step * atoms.get_volume()), abs=1e-8
    )


@pytest.mark.cuda
@pytest.mark.parametrize("field", ("interactions", "nodes"))
def test_scalar_polar_device_loader_rejects_an_extra_layer(scalar_polar_model, field):
    """Unused ABI capacity cannot admit an extra layer in S's launch plan."""

    if symmetrix.load_backend()._kokkos_default_execution_space() != "Cuda":
        pytest.skip("requires a loaded generated CUDA module")
    path, _ = scalar_polar_model
    calculator = symmetrix.Symmetrix(
        str(path), dtype="float64", streamed_edges="direct"
    )
    calculator.calculate(_atoms(), properties=["energy"])
    expected = float(calculator.results["energy"])
    artifact = Path(calculator.jit_artifact_path)
    manifest = json.loads(artifact.with_name("manifest.json").read_text())
    plan = manifest["key_inputs"]["build"]["request"]["launch_plan"]
    assert len(plan["interactions"]) == len(plan["nodes"]) == 1
    plan[field].append(dict(plan[field][0], index=1))
    with pytest.raises(
        RuntimeError, match="launch-plan interaction count does not match"
    ):
        calculator.evaluator._load_jit_mh1_cuda_plugin_v5(
            str(artifact), json.dumps(plan, sort_keys=True, separators=(",", ":"))
        )
    calculator.calculate(_atoms(), properties=["energy"])
    assert calculator.results["energy"] == expected


@pytest.fixture(scope="module")
def released_scalar_polar_calculators(tmp_path_factory):
    from test_polar_end_to_end_fixtures import (
        FROZEN_ELECTROSTATICS,
        ORACLE_NUM_SPLINE_POINTS,
        _checkpoint_path,
    )
    import torch

    model = torch.load(_checkpoint_path("S"), map_location="cpu", weights_only=False)
    data = extract_mace_polar_data(
        model.to(torch.float64), num_spline_points=ORACLE_NUM_SPLINE_POINTS
    )
    path = tmp_path_factory.mktemp("released-polar-s-direct") / "model.json"
    path.write_text(json.dumps(data))
    calculators = {
        dtype: symmetrix.Symmetrix(
            str(path),
            dtype=dtype,
            streamed_edges="direct",
            polar_electrostatics=FROZEN_ELECTROSTATICS,
        )
        for dtype in ("float64", "float32")
    }
    yield calculators
    calculators.clear()


@pytest.mark.parametrize("dtype", ("float64", "float32"))
@pytest.mark.parametrize(
    "geometry,channel",
    [
        (geometry, channel)
        for geometry in ("mol_h2o", "slab_w2", "bulk_w4")
        for channel in ("singlet", "charged_triplet", "field")
    ]
    + [("mol_h2o", "fermi_lane")],
)
def test_released_scalar_polar_direct_matches_frozen_oracle(
    released_scalar_polar_calculators, dtype, geometry, channel
):
    """S's generated seeded reverse must meet the unchanged frozen-record bounds."""

    from ase import Atoms
    from ase.stress import full_3x3_to_voigt_6_stress
    from test_polar_end_to_end_fixtures import (
        FAMILY_CHANNELS,
        FAMILY_FORCE_TOLERANCE,
        FAMILY_STRESS_RELATION_TOLERANCE,
        FAMILY_TOLERANCE,
        _load_record,
    )

    calculator = released_scalar_polar_calculators[dtype]
    charge, spin, field, fermi, _ = FAMILY_CHANNELS[channel]
    calculator.total_charge = charge
    calculator.total_spin = spin
    calculator.electric_field = field
    calculator.fermi_level = fermi
    record = _load_record(f"e2e_S_{geometry}_{channel}_{dtype}.json")
    atoms = Atoms(
        record["species"],
        positions=record["positions"],
        cell=np.asarray(record["cell"]).reshape(3, 3),
        pbc=record["pbc"],
    )
    calculator.calculate(atoms, properties=["energy", "forces", "stress", "dipole"])
    assert calculator.jit_status in ("built", "cached")
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0
    tolerance = FAMILY_TOLERANCE[dtype]
    assert abs(calculator.results["energy"] - record["energy"]) <= (
        tolerance["energy_absolute"]
        + tolerance["energy_relative"] * abs(record["energy"])
    )
    np.testing.assert_allclose(
        calculator.results["forces"],
        record["forces"],
        rtol=0.0,
        atol=FAMILY_FORCE_TOLERANCE[dtype]["force_absolute"],
    )
    np.testing.assert_allclose(
        calculator.results["dipole"],
        np.asarray(record["total_dipole"]).reshape(3),
        rtol=0.0,
        atol=tolerance["dipole"],
    )
    calculator._materialize_polar_diagnostics()
    own_stress = calculator._polar_composition["polar_stress"][0]
    np.testing.assert_allclose(
        calculator.results["stress"] - np.asarray(record["stress"]),
        full_3x3_to_voigt_6_stress(own_stress),
        rtol=0.0,
        atol=FAMILY_STRESS_RELATION_TOLERANCE[dtype],
    )
