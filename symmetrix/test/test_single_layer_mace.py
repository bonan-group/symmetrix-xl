import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import symmetrix.calculator as calculator_module
import torch
from ase import Atoms

from symmetrix import Symmetrix

MACE_SOURCE = Path(os.environ.get("SYMMETRIX_SINGLE_LAYER_MACE_SOURCE", ""))
CHECKPOINT = Path(os.environ.get("SYMMETRIX_SINGLE_LAYER_CHECKPOINT", ""))
SINGLE_V2_MODEL = Path(os.environ.get("SYMMETRIX_SINGLE_V2_MODEL", ""))


pytestmark = pytest.mark.skipif(
    not os.environ.get("SYMMETRIX_SINGLE_LAYER_MACE_SOURCE")
    or not MACE_SOURCE.is_dir(),
    reason="set SYMMETRIX_SINGLE_LAYER_MACE_SOURCE to run single-layer tests",
)


@pytest.fixture(scope="module")
def single_layer_artifacts(tmp_path_factory):
    if not CHECKPOINT.is_file():
        pytest.skip("single-layer MACE training checkpoint is unavailable")
    if str(MACE_SOURCE) not in sys.path:
        sys.path.insert(0, str(MACE_SOURCE))
    from symmetrix.extract_mace_data import export_mace_model, extract_mace_data

    output = tmp_path_factory.mktemp("single-layer-mace")
    exported = output / "single-layer-without-self-e3nn.pt"
    model = export_mace_model(CHECKPOINT, exported)
    data = extract_mace_data(
        CHECKPOINT,
        species=[1, 8],
        num_spline_points=64,
        radial_format="compact",
    )
    model_json = output / "single-layer.json"
    model_json.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return model, exported, data, model_json


@pytest.fixture(scope="module")
def single_v2_artifacts(tmp_path_factory):
    if not SINGLE_V2_MODEL.is_file():
        pytest.skip("single-v2 MACE model is unavailable")
    if str(MACE_SOURCE) not in sys.path:
        sys.path.insert(0, str(MACE_SOURCE))
    from symmetrix.extract_mace_data import extract_mace_data

    data = extract_mace_data(
        SINGLE_V2_MODEL,
        species=[1, 8],
        num_spline_points=256,
        radial_format="compact",
    )
    model = torch.load(SINGLE_V2_MODEL, map_location="cpu", weights_only=False)
    model_json = tmp_path_factory.mktemp("single-v2-mace") / "single-v2.json"
    model_json.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    return model, data, model_json


def test_single_v2_l1_layout_is_extracted(single_v2_artifacts):
    model, data, _ = single_v2_artifacts

    assert model.products[0].linear.irreps_out.lmax == 1
    assert data["l_max"] == 3
    assert data["L_max"] == 1
    assert data["num_channels"] == 128
    assert data["num_interactions"] == 1
    assert data["single_layer_readout"] is True
    assert data["readout_correlation"] == 2
    assert set(data["execution_contracts"]) == {"M0", "R0"}
    assert set(data["compact_radial"]["networks"]) == {"A0", "R0"}


def test_single_v2_extracted_readout_matches_e3nn(single_v2_artifacts):
    model, data, _ = single_v2_artifacts
    model = model.double()
    torch.manual_seed(17)
    features = torch.randn(
        2, model.products[0].linear.irreps_out.dim, dtype=torch.float64
    )
    attributes = torch.zeros((2, model.atomic_numbers.numel()), dtype=torch.float64)
    for row, atomic_number in enumerate((1, 8)):
        attributes[row, torch.nonzero(model.atomic_numbers == atomic_number)[0, 0]] = 1
    reshaped = model.readout_reshapes[0](features)
    expected = model.readout_products[0].symmetric_contractions(reshaped, attributes)

    actual = np.zeros((2, data["num_channels"]))
    for row in range(2):
        for channel in range(data["num_channels"]):
            actual[row, channel] = sum(
                coefficient
                * np.prod([reshaped[row, channel, index].item() for index in monomial])
                for coefficient, monomial in zip(
                    data["M1_weights"][row][channel],
                    data["M1_monomials"],
                    strict=True,
                )
            )
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1.0e-12)


def test_single_v2_direct_matches_mace_torch(single_v2_artifacts):
    _, _, model_json = single_v2_artifacts

    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )
    reference_script = """
import json
import sys

from ase import Atoms
from mace.calculators.mace import MACECalculator

payload = json.loads(sys.argv[2])
atoms = Atoms(
    payload["symbols"],
    positions=payload["positions"],
    cell=payload["cell"],
    pbc=payload["pbc"],
)
atoms.calc = MACECalculator(
    model_paths=sys.argv[1],
    device="cpu",
    default_dtype="float64",
    enable_cueq=False,
)
result = {
    "energy": atoms.get_potential_energy(),
    "forces": atoms.get_forces().tolist(),
    "stress": atoms.get_stress().tolist(),
}
print("SYMMETRIX_MACE_REFERENCE=" + json.dumps(result))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(MACE_SOURCE), environment.get("PYTHONPATH", "")))
    )
    payload = {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            reference_script,
            str(SINGLE_V2_MODEL),
            json.dumps(payload),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    reference_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("SYMMETRIX_MACE_REFERENCE=")
    )
    reference = json.loads(reference_line.partition("=")[2])

    symmetrix_atoms = atoms.copy()
    symmetrix_atoms.calc = Symmetrix(
        model_json, use_kokkos=True, dtype="float64", streamed_edges="direct"
    )
    assert symmetrix_atoms.get_potential_energy() == pytest.approx(
        reference["energy"], rel=0.0, abs=3.0e-6
    )
    np.testing.assert_allclose(
        symmetrix_atoms.get_forces(), reference["forces"], rtol=0.0, atol=2.0e-4
    )
    np.testing.assert_allclose(
        symmetrix_atoms.get_stress(), reference["stress"], rtol=0.0, atol=1.0e-7
    )
    assert symmetrix_atoms.calc.jit_status == "not_applicable"
    assert symmetrix_atoms.calc.evaluator.factorized_prepared_evaluation_count == 1


@pytest.mark.parametrize(
    "plan",
    (
        "mh0-single-layer-tiled-v1",
        "mh0-direct-capacity-y-only",
        "mh0-direct-speed",
    ),
)
def test_single_v2_native_periodic_graph_avoids_host_connectivity(
    single_v2_artifacts, monkeypatch, plan
):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    atoms = _single_v2_test_atoms(5)
    _, expected = _single_v2_plan_results(model_json, atoms, plan, receiver_limit=4)
    deformed = atoms.copy()
    deformed.set_cell([8.01, 7.99, 8.02], scale_atoms=True)
    deformed.positions[0] += [0.01, -0.015, 0.005]
    _, expected_deformed = _single_v2_plan_results(
        model_json, deformed, plan, receiver_limit=4
    )
    calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="speed" if plan == "mh0-direct-speed" else "capacity",
        neighbor_skin=0.5,
        _debug_execution_plan=plan,
    )
    calculator.evaluator._set_single_layer_workspace_receiver_limit_for_testing(4)
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(
        calculator_module,
        "neighbor_list",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-v2 must use native Kokkos connectivity")
        ),
    )

    calculator.calculate(atoms, properties=["energy", "energies", "forces", "stress"])

    for name in ("energy", "energies", "forces", "stress"):
        np.testing.assert_allclose(
            calculator.results[name], expected[name], atol=5.0e-4
        )
    calculator.calculate(
        deformed, properties=["energy", "energies", "forces", "stress"]
    )
    for name in ("energy", "energies", "forces", "stress"):
        np.testing.assert_allclose(
            calculator.results[name], expected_deformed[name], atol=5.0e-4
        )
    assert calculator.neighbor_graph_backend == "kokkos"
    assert calculator._neighbor_cache.device_graph_generation != 0
    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert calculator.execution_plan["selected_id"] == plan
    assert calculator.evaluator.single_layer_tiled_evaluation_count == (
        2 if plan == "mh0-single-layer-tiled-v1" else 0
    )


def _require_cuda_backend():
    import symmetrix as native_symmetrix

    try:
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
    except RuntimeError as error:
        pytest.skip(f"CUDA runtime is unavailable: {error}")
    backend = native_symmetrix._kokkos_default_execution_space()
    if backend != "Cuda":
        pytest.skip("single-layer fixed-workspace qualification requires CUDA")


def _single_v2_test_atoms(count):
    grid = np.indices((3, 3, 3)).reshape(3, -1).T[:count]
    positions = 1.1 + 1.8 * grid
    symbols = ["O" if index % 3 == 0 else "H" for index in range(count)]
    positions[-1] = [3.0, 3.0, 3.0]
    return Atoms(symbols, positions=positions, cell=[8.0, 8.0, 8.0], pbc=True)


def _single_v2_plan_results(
    model_json, atoms, plan, receiver_limit=None, neighbor_skin=0.5
):
    calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="speed" if plan == "mh0-direct-speed" else "capacity",
        neighbor_skin=neighbor_skin,
        _debug_execution_plan=plan,
    )
    if receiver_limit is not None:
        calculator.evaluator._set_single_layer_workspace_receiver_limit_for_testing(
            receiver_limit
        )
    calculator.calculate(atoms, properties=["energy", "energies", "forces", "stress"])
    return calculator, {
        name: np.array(calculator.results[name], copy=True)
        for name in ("energy", "energies", "forces", "stress")
    }


@pytest.mark.parametrize("atom_count", (3, 4, 5, 8, 9))
def test_single_v2_fixed_workspace_matches_y_only_at_tile_boundaries(
    single_v2_artifacts, atom_count
):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    receiver_limit = 4
    atoms = _single_v2_test_atoms(atom_count)
    reference, expected = _single_v2_plan_results(
        model_json, atoms, "mh0-direct-capacity-y-only"
    )
    calculator, actual = _single_v2_plan_results(
        model_json, atoms, "mh0-single-layer-tiled-v1", receiver_limit
    )

    np.testing.assert_allclose(actual["energy"], expected["energy"], atol=5.0e-4)
    np.testing.assert_allclose(actual["energies"], expected["energies"], atol=5.0e-4)
    np.testing.assert_allclose(actual["forces"], expected["forces"], atol=5.0e-4)
    np.testing.assert_allclose(actual["stress"], expected["stress"], atol=5.0e-4)

    evaluator = calculator.evaluator
    capacity = min(atom_count, receiver_limit)
    selected_plan = calculator.execution_plan
    if isinstance(selected_plan, dict):
        selected_plan = selected_plan.get("selected_id")
    assert selected_plan == "mh0-single-layer-tiled-v1"
    assert evaluator.single_layer_workspace_active_receivers == capacity
    assert evaluator.single_layer_workspace_planned_receivers == capacity
    assert evaluator.single_layer_workspace_bytes_per_receiver == 13_836
    assert evaluator.single_layer_workspace_bytes_per_edge == 112
    assert evaluator.single_layer_workspace_active_edges <= len(
        calculator._neighbor_cache.receivers
    )
    assert (
        evaluator.single_layer_workspace_planned_edges
        == evaluator.single_layer_workspace_active_edges
    )
    assert evaluator.single_layer_workspace_bytes == (
        13_836 * capacity + 112 * evaluator.single_layer_workspace_planned_edges
    )
    assert evaluator.compact_edge_geometry_bytes == (
        20 * evaluator.single_layer_workspace_planned_edges
    )
    assert evaluator.factorized_graph_device_capacity_bytes == (
        4 * evaluator.execution_planned_edge_capacity
        + 16 * evaluator.execution_planned_receiver_capacity
    )
    assert evaluator.execution_geometry_workspace_bytes == (
        12 * evaluator.execution_planned_edge_capacity
        + 48 * evaluator.execution_planned_feature_node_capacity
        + 88 * evaluator.single_layer_workspace_planned_edges
        + 160
    )
    assert evaluator.single_layer_workspace_replacement_count == 1
    assert evaluator.single_layer_workspace_reuse_count == 0
    assert evaluator.single_layer_workspace_batch_count == math.ceil(
        atom_count / receiver_limit
    )
    assert evaluator.single_layer_tiled_evaluation_count == 1
    assert evaluator.factorized_fallback_evaluation_count == 0
    assert evaluator.harmonic_gradient_bytes == 0
    assert evaluator.h1_m0_adjoint_ping_pong_active
    final_batch = atom_count % receiver_limit or capacity
    assert len(evaluator.A0) == final_batch * 16 * 128
    assert len(evaluator.M0) == final_batch * 4 * 128
    assert len(evaluator.H1) == final_batch * 4 * 128
    assert len(evaluator.M1) == final_batch * 128
    assert len(evaluator.H2) == final_batch * 128
    assert len(evaluator.A1) == 0
    reference_plan = reference.execution_plan
    if isinstance(reference_plan, dict):
        reference_plan = reference_plan.get("selected_id")
    assert reference_plan == "mh0-direct-capacity-y-only"


def test_single_v2_fixed_workspace_reuses_and_grows_only_during_prepare(
    single_v2_artifacts,
):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    receiver_limit = 8
    calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
        _debug_execution_plan="mh0-single-layer-tiled-v1",
    )
    evaluator = calculator.evaluator
    evaluator._set_single_layer_workspace_receiver_limit_for_testing(receiver_limit)

    expected = (
        (3, 3, 1, 0),
        (2, 3, 1, 1),
        (5, 5, 2, 1),
        (9, 8, 3, 1),
        (10, 8, 3, 2),
    )
    total_batches = 0
    for atom_count, planned, replacements, reuses in expected:
        calculator.calculate(
            _single_v2_test_atoms(atom_count),
            properties=["energy", "energies", "forces", "stress"],
        )
        total_batches += math.ceil(atom_count / min(atom_count, receiver_limit))
        assert evaluator.single_layer_workspace_active_receivers == min(
            atom_count, receiver_limit
        )
        assert evaluator.single_layer_workspace_planned_receivers == planned
        assert evaluator.single_layer_workspace_bytes == (
            13_836 * planned + 112 * evaluator.single_layer_workspace_planned_edges
        )
        assert evaluator.single_layer_workspace_replacement_count == replacements
        assert evaluator.single_layer_workspace_reuse_count == reuses
        assert evaluator.single_layer_workspace_batch_count == total_batches
        assert evaluator.single_layer_tiled_evaluation_count == replacements + reuses
        assert np.all(np.isfinite(calculator.results["forces"]))


def test_single_v2_capacity_fixed_workspace_requires_opt_in_at_memory_boundary(
    single_v2_artifacts,
):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    atoms = _single_v2_test_atoms(9)
    device_bytes = 32 * 1024**3

    probe = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
        _debug_execution_plan="mh0-direct-capacity-y-only",
    )
    probe_inputs = probe._mace_inputs(atoms)
    probe.evaluator._set_low_memory_device_memory_info_for_testing(
        device_bytes, device_bytes
    )
    probe.evaluator._prepare_factorized_graph(*probe_inputs[:5])
    estimates = {
        candidate["id"]: candidate["estimated_bytes"]
        for candidate in probe.execution_plan["candidates"]
    }
    tiled_estimate = estimates["mh0-single-layer-tiled-v1"]
    y_only_estimate = estimates["mh0-direct-capacity-y-only"]
    assert tiled_estimate < y_only_estimate
    edge_count = len(probe._neighbor_cache.receivers)
    assert tiled_estimate == (
        16 * edge_count
        + 96 * len(atoms)
        + 13_836 * len(atoms)
        + 112 * edge_count
        + 9 * np.dtype(np.float64).itemsize
    )

    default_calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
    )
    default_evaluator = default_calculator.evaluator
    default_inputs = default_calculator._mace_inputs(atoms)
    reserve = device_bytes // 20
    available = (tiled_estimate + y_only_estimate) // 2
    default_evaluator._set_low_memory_device_memory_info_for_testing(
        reserve + available, device_bytes
    )
    default_evaluator._prepare_factorized_graph(*default_inputs[:5])

    assert default_calculator.allow_fixed_workspace is False
    assert default_evaluator.allow_fixed_workspace is False
    assert (
        default_calculator.execution_plan["selected_id"] == "mh0-direct-capacity-y-only"
    )
    tiled_candidate = next(
        candidate
        for candidate in default_calculator.execution_plan["candidates"]
        if candidate["id"] == "mh0-single-layer-tiled-v1"
    )
    assert tiled_candidate["qualified"] is False
    assert "requires explicit opt-in" in tiled_candidate["reason"]
    assert default_calculator.execution_plan["boundary_attempt"] is True
    assert default_evaluator.single_layer_tiled_evaluation_count == 0

    calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        allow_fixed_workspace=True,
        neighbor_skin=0.5,
    )
    evaluator = calculator.evaluator
    inputs = calculator._mace_inputs(atoms)
    evaluator._set_low_memory_device_memory_info_for_testing(
        reserve + available, device_bytes
    )
    token = evaluator._prepare_factorized_graph(*inputs[:5])

    assert calculator.allow_fixed_workspace is True
    assert evaluator.allow_fixed_workspace is True
    assert calculator.execution_plan["selected_id"] == "mh0-single-layer-tiled-v1"
    assert calculator.execution_plan["boundary_attempt"] is False
    assert evaluator.low_memory_policy == "capacity-y-only"
    assert evaluator.low_memory_selected_estimated_bytes == tiled_estimate
    assert evaluator.single_layer_workspace_active_receivers == len(atoms)
    assert evaluator.low_memory_selection_reason.startswith(
        "fixed-workspace selection:"
    )

    geometry_bytes_before = evaluator.execution_geometry_workspace_bytes
    evaluator._compute_prepared_factorized(
        token, np.asarray(inputs[5]).reshape(-1), inputs[6]
    )
    assert evaluator.execution_geometry_workspace_bytes == geometry_bytes_before + 4
    assert evaluator.single_layer_tiled_evaluation_count == 1
    assert evaluator.factorized_fallback_evaluation_count == 0
    with pytest.raises(ValueError, match="current graph and geometry tokens"):
        evaluator._compute_prepared_factorized_positions(
            token, np.asarray(atoms.positions).reshape(-1)
        )


def test_single_v2_fixed_workspace_explicit_geometry_matches_positions(
    single_v2_artifacts,
):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    atoms = _single_v2_test_atoms(5)
    native, expected = _single_v2_plan_results(
        model_json, atoms, "mh0-single-layer-tiled-v1", receiver_limit=4
    )
    calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
        _debug_execution_plan="mh0-single-layer-tiled-v1",
    )
    calculator.evaluator._set_single_layer_workspace_receiver_limit_for_testing(4)
    inputs = calculator._mace_inputs(atoms)
    _, _, radii, xyz = calculator._cached_neighbor_geometry(
        atoms, native_geometry=False
    )
    explicit_inputs = (*inputs[:5], xyz, radii, inputs[7])
    token = calculator.evaluator._prepare_factorized_graph(*inputs[:5])
    calculator.evaluator._compute_prepared_factorized(
        token, np.asarray(xyz).reshape(-1), radii
    )
    actual = calculator._collect_mace_results(
        atoms,
        explicit_inputs,
        properties=["energy", "energies", "forces", "stress"],
        execution_graph_generation=token,
    )

    for name in ("energy", "energies", "forces", "stress"):
        np.testing.assert_allclose(actual[name], expected[name], atol=5.0e-4)
    assert calculator.evaluator.single_layer_workspace_batch_count == 2
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0
    assert native.evaluator.factorized_fallback_evaluation_count == 0


def test_single_v2_fixed_workspace_reuses_shift_geometry_after_cell_change(
    single_v2_artifacts,
):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    atoms = _single_v2_test_atoms(5)
    calculator, _ = _single_v2_plan_results(
        model_json, atoms, "mh0-single-layer-tiled-v1", receiver_limit=4
    )

    changed = atoms.copy()
    changed.set_cell(changed.cell.array * [1.002, 0.999, 1.001], scale_atoms=True)
    changed.positions[0] += [0.01, -0.015, 0.02]
    changed.wrap()
    calculator.calculate(changed, properties=["energy", "energies", "forces", "stress"])
    actual = {
        name: np.array(calculator.results[name], copy=True)
        for name in ("energy", "energies", "forces", "stress")
    }
    _, expected = _single_v2_plan_results(
        model_json, changed, "mh0-direct-capacity-y-only"
    )

    for name in ("energy", "energies", "forces", "stress"):
        np.testing.assert_allclose(actual[name], expected[name], atol=5.0e-4)
    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0


def test_single_v2_fixed_workspace_uses_compact_shifts_without_neighbor_skin(
    single_v2_artifacts,
):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    atoms = _single_v2_test_atoms(5)
    _, expected = _single_v2_plan_results(
        model_json, atoms, "mh0-direct-capacity-y-only", neighbor_skin=0.0
    )
    calculator, actual = _single_v2_plan_results(
        model_json,
        atoms,
        "mh0-single-layer-tiled-v1",
        receiver_limit=4,
        neighbor_skin=0.0,
    )

    for name in ("energy", "energies", "forces", "stress"):
        np.testing.assert_allclose(actual[name], expected[name], atol=5.0e-4)
    evaluator = calculator.evaluator
    assert calculator._execution_native_geometry_identity[-1] == "integer-shifts"
    assert evaluator.execution_geometry_workspace_bytes == (
        12 * evaluator.execution_planned_edge_capacity
        + 48 * evaluator.execution_planned_feature_node_capacity
        + 88 * evaluator.single_layer_workspace_planned_edges
        + 160
    )


def test_single_v2_fixed_workspace_rejects_float64(single_v2_artifacts):
    _require_cuda_backend()
    _, _, model_json = single_v2_artifacts
    calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="capacity",
        _debug_execution_plan="mh0-single-layer-tiled-v1",
    )
    with pytest.raises(ValueError, match="requires float32 precision"):
        calculator.calculate(_single_v2_test_atoms(3), properties=["energy"])


def test_training_checkpoint_exports_loadable_model(single_layer_artifacts):
    model, exported, _, _ = single_layer_artifacts
    loaded = torch.load(exported, map_location="cpu", weights_only=False)

    assert type(model).__name__ == "ScaleShiftMACE"
    assert type(loaded).__name__ == "ScaleShiftMACE"
    assert len(loaded.interactions) == 1
    assert len(loaded.readout_products) == 1
    assert loaded.readout_correlation == 2
    assert loaded.atomic_numbers.numel() == 89


def test_single_layer_extraction_has_no_r1_contract(single_layer_artifacts):
    _, _, data, _ = single_layer_artifacts

    assert data["num_interactions"] == 1
    assert data["single_layer_readout"] is True
    assert data["readout_correlation"] == 2
    assert set(data["execution_contracts"]) == {"M0", "R0"}
    assert set(data["compact_radial"]["networks"]) == {"A0", "R0"}
    assert data["Phi1_l"] == []
    assert data["A1_scaled"] is False


def test_extracted_readout_polynomial_matches_e3nn(single_layer_artifacts):
    model, _, data, _ = single_layer_artifacts
    model = model.double()
    torch.manual_seed(7)
    features = torch.randn(
        2, model.products[0].linear.irreps_out.dim, dtype=torch.float64
    )
    attributes = torch.zeros((2, model.atomic_numbers.numel()), dtype=torch.float64)
    for row, atomic_number in enumerate((1, 8)):
        attributes[row, torch.nonzero(model.atomic_numbers == atomic_number)[0, 0]] = 1
    reshaped = model.readout_reshapes[0](features)
    expected = model.readout_products[0].symmetric_contractions(reshaped, attributes)

    actual = np.zeros((2, data["num_channels"]))
    for row in range(2):
        for channel in range(data["num_channels"]):
            coefficients = data["M1_weights"][row][channel]
            actual[row, channel] = sum(
                coefficient
                * np.prod([reshaped[row, channel, index].item() for index in monomial])
                for coefficient, monomial in zip(
                    coefficients, data["M1_monomials"], strict=True
                )
            )
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1.0e-12)


def test_prepared_direct_matches_cpu_without_r1_plugin(single_layer_artifacts):
    _, _, _, model_json = single_layer_artifacts
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )

    reference_atoms = atoms.copy()
    reference_atoms.calc = Symmetrix(
        model_json, use_kokkos=False, dtype="float64", streamed_edges="generic"
    )
    direct_atoms = atoms.copy()
    direct_atoms.calc = Symmetrix(
        model_json, use_kokkos=True, dtype="float64", streamed_edges="direct"
    )

    assert direct_atoms.get_potential_energy() == pytest.approx(
        reference_atoms.get_potential_energy(), abs=1.0e-11
    )
    np.testing.assert_allclose(
        direct_atoms.get_forces(), reference_atoms.get_forces(), atol=1.0e-11
    )
    assert direct_atoms.calc.jit_status == "not_applicable"
    assert direct_atoms.calc.evaluator.factorized_prepared_evaluation_count == 1
    assert not direct_atoms.calc.evaluator.jit_host_plugin_ready
    assert not direct_atoms.calc.evaluator.jit_device_plugin_ready


def test_low_memory_fp32_matches_float64(single_layer_artifacts):
    _, _, _, model_json = single_layer_artifacts
    import symmetrix as native_symmetrix

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    backend = native_symmetrix._kokkos_default_execution_space()
    if backend not in ("OpenMP", "Serial", "Cuda", "HIP"):
        pytest.skip("single-layer L=2 low-memory qualification requires Kokkos")

    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )
    reference = Symmetrix(
        model_json,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="generic",
    )
    reference.calculate(atoms, properties=["energy", "forces", "stress"])

    calculator = Symmetrix(
        model_json,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        low_memory=True,
    )
    assert isinstance(calculator.evaluator, native_symmetrix.MACEKokkosFloat)
    assert calculator.evaluator.scalar_size_bytes == 4
    assert calculator.streamed_edges == "direct"
    assert calculator.low_memory is True
    assert calculator.edge_geometry_policy == "unit-f32-radius-f64-v1"
    assert set(calculator.jit_operator_modules) == {"M0", "R0"}
    if backend in ("Cuda", "HIP"):
        assert not calculator.evaluator.jit_device_plugin_ready
        assert calculator.evaluator.harmonic_storage_policy == "y-only-direct-v1"
        assert calculator.evaluator.harmonic_storage_fallback_reason == ""
    else:
        assert calculator.evaluator.m0_host_plugin_ready
        assert calculator.evaluator.m0_implementation == "host_plugin"
        assert calculator.evaluator.harmonic_storage_policy == "y-only-direct-v1"
        assert calculator.evaluator.harmonic_storage_fallback_reason == ""

    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    graphs_after_first = calculator.evaluator.factorized_prepared_graph_count
    evaluations_after_first = calculator.evaluator.factorized_prepared_evaluation_count
    if backend in ("OpenMP", "Serial"):
        assert calculator.evaluator.factorized_arena_workspace_bytes == 0
        assert calculator.evaluator.h1_m0_adjoint_ping_pong_active
        assert calculator.evaluator.h1_m0_adjoint_ping_pong_bytes == (
            np.dtype(np.float32).itemsize * len(calculator.evaluator.H1)
        )
    else:
        assert not calculator.evaluator.h1_m0_adjoint_ping_pong_active
        assert calculator.evaluator.h1_m0_adjoint_ping_pong_bytes == 0

    assert energy == pytest.approx(reference.results["energy"], rel=0.0, abs=5.0e-4)
    np.testing.assert_allclose(
        forces, reference.results["forces"], rtol=0.0, atol=5.0e-4
    )
    np.testing.assert_allclose(
        stress, reference.results["stress"], rtol=0.0, atol=5.0e-4
    )

    positions = atoms.get_positions()
    positions[1, 0] += 0.01
    atoms.set_positions(positions)
    reference.calculate(atoms, properties=["energy", "forces", "stress"])
    displaced_energy = atoms.get_potential_energy()
    displaced_forces = atoms.get_forces()
    displaced_stress = atoms.get_stress()

    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(stress))
    assert np.isfinite(displaced_energy)
    assert np.all(np.isfinite(displaced_forces))
    assert np.all(np.isfinite(displaced_stress))
    assert displaced_energy == pytest.approx(
        reference.results["energy"], rel=0.0, abs=5.0e-4
    )
    np.testing.assert_allclose(
        displaced_forces, reference.results["forces"], rtol=0.0, atol=5.0e-4
    )
    np.testing.assert_allclose(
        displaced_stress, reference.results["stress"], rtol=0.0, atol=5.0e-4
    )
    assert calculator.evaluator.harmonic_gradient_bytes == 0
    assert calculator.evaluator.factorized_fallback_evaluation_count == 0
    m0_module = calculator.jit_operator_modules["M0"]
    if backend == "HIP" and m0_module.get("target") == "gfx1151":
        assert m0_module["persistent_blocks_per_compute_unit"] == 2
        r0_profile = calculator.kernel_launch_profile_diagnostics["r0_reverse"]
        assert (
            r0_profile["profile_id"]
            == "standard-r0-module-module-v2-hip-float32-gfx1151"
        )
        assert r0_profile["active_blocks_per_compute_unit"] == 8
    assert calculator.evaluator.factorized_prepared_graph_count == graphs_after_first
    assert (
        calculator.evaluator.factorized_prepared_evaluation_count
        == evaluations_after_first + 1
    )
    if backend in ("OpenMP", "Serial"):
        assert calculator.evaluator.factorized_arena_workspace_bytes == 0
        assert calculator.evaluator.h1_m0_adjoint_ping_pong_active
