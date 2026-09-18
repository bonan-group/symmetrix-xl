import json
import math
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from compact_r1_model import foundation_r1_test_model
from symmetrix import Symmetrix
from symmetrix.execution_contract import normalize_standard_m0_contract


PLAN = "mh0-dual-layer-tiled-v1"
REFERENCE_PLAN = "mh0-direct-capacity-y-only"
PROPERTIES = ("energy", "energies", "forces", "stress")


def _require_cuda_backend():
    import symmetrix as native

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    if native._kokkos_default_execution_space() != "Cuda":
        pytest.skip("dual-layer fixed-workspace qualification requires CUDA")


@pytest.fixture(scope="module")
def dual_layer_model(tmp_path_factory):
    contracts = Path(__file__).resolve().parent / "data" / "execution_contracts"
    r1_contract = json.loads(
        (contracts / "jit_r1_mace_omat0_small_contract.json").read_text()
    )
    m0_contract = json.loads((contracts / "standard_m0_contract.json").read_text())
    m0_contract["output_l_max"] = r1_contract["source_harmonics"]["l_max"]
    m0_contract["output_components"] = (m0_contract["output_l_max"] + 1) ** 2
    m0_contract["monomial_groups"] = m0_contract["monomial_groups"][
        : m0_contract["output_components"]
    ]
    for fingerprint in (
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
    ):
        m0_contract.pop(fingerprint, None)
    m0_contract = normalize_standard_m0_contract(m0_contract)
    r0_contract = json.loads((contracts / "standard_r0_contract.json").read_text())
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
    model["readout_1_weights"] = [
        0.001 * (1 + channel % 7) for channel in range(model["num_channels"])
    ]
    output = tmp_path_factory.mktemp("dual-layer-workspace") / "model.json"
    output.write_text(json.dumps(model, separators=(",", ":")))
    return output


def _atoms(count):
    grid = np.indices((3, 3, 3)).reshape(3, -1).T[:count]
    positions = 0.7 + 1.6 * grid
    positions[:, 1] += 0.03 * np.arange(count)
    symbols = ["Al" if index % 2 == 0 else "N" for index in range(count)]
    return Atoms(symbols, positions=positions, cell=[7.5, 7.5, 7.5], pbc=True)


def _calculator(model, plan, capacity=None, skin=0.5):
    calculator = Symmetrix(
        model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=skin,
        _debug_execution_plan=plan,
    )
    if capacity is not None:
        calculator.evaluator._set_dual_layer_workspace_receiver_limit_for_testing(
            capacity
        )
    return calculator


def _evaluate(calculator, atoms):
    calculator.calculate(atoms, properties=list(PROPERTIES))
    results = {
        name: np.array(calculator.results[name], copy=True) for name in PROPERTIES
    }
    active_h1_elements = len(atoms) * 128
    h1_adjoint = np.asarray(calculator.evaluator.H1_adj)[:active_h1_elements].copy()
    return results, h1_adjoint


def _assert_results(actual, expected):
    for name in PROPERTIES:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2.0e-4, atol=2e-5)


@pytest.mark.parametrize("neighbor_backend", ("host", "kokkos"))
def test_dual_layer_matches_y_only_at_tile_boundaries(
    dual_layer_model, monkeypatch, neighbor_backend
):
    _require_cuda_backend()
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", neighbor_backend)
    capacity = 4
    for atom_count in (3, 4, 5, 8, 9):
        atoms = _atoms(atom_count)
        reference = _calculator(dual_layer_model, REFERENCE_PLAN)
        tiled = _calculator(dual_layer_model, PLAN, capacity)
        expected, expected_h1_adjoint = _evaluate(reference, atoms)
        actual, actual_h1_adjoint = _evaluate(tiled, atoms)

        _assert_results(actual, expected)
        np.testing.assert_allclose(
            actual_h1_adjoint,
            expected_h1_adjoint,
            rtol=2.0e-4,
            atol=2e-5,
        )
        evaluator = tiled.evaluator
        assert tiled.execution_plan["selected_id"] == PLAN
        assert evaluator.dual_layer_workspace_active_receivers == min(
            capacity, atom_count
        )
        assert (
            evaluator.dual_layer_workspace_active_edges
            <= tiled._neighbor_cache.num_edges
        )
        assert evaluator.dual_layer_workspace_batch_count == 3 * math.ceil(
            atom_count / capacity
        )
        assert evaluator.dual_layer_tiled_evaluation_count == 1
        assert evaluator.factorized_fallback_evaluation_count == 0
        assert evaluator.harmonic_gradient_bytes == 0


def _assert_schedule(schedule, capacity):
    tile_offsets = schedule["tile_segment_offsets"]
    segment_sources = schedule["segment_source_ids"]
    segment_offsets = schedule["segment_edge_offsets"]
    source_edges = schedule["source_edges"]
    degrees = schedule["num_neigh"]
    edge_sources = schedule["edge_sources"]
    receiver_offsets = np.concatenate(([0], np.cumsum(degrees)))
    covered = []

    assert tile_offsets[0] == 0
    assert tile_offsets[-1] == len(segment_sources)
    assert segment_offsets[0] == 0
    assert segment_offsets[-1] == len(source_edges)
    for tile in range(len(tile_offsets) - 1):
        receiver_begin = tile * capacity
        receiver_end = min(len(degrees), receiver_begin + capacity)
        tile_edge_begin = receiver_offsets[receiver_begin]
        tile_edge_end = receiver_offsets[receiver_end]
        for segment in range(tile_offsets[tile], tile_offsets[tile + 1]):
            local = source_edges[
                segment_offsets[segment] : segment_offsets[segment + 1]
            ]
            global_edges = tile_edge_begin + local
            assert np.all(global_edges[1:] > global_edges[:-1])
            assert np.all(edge_sources[global_edges] == segment_sources[segment])
            assert np.all(global_edges < tile_edge_end)
            covered.extend(global_edges.tolist())
    assert sorted(covered) == list(range(len(edge_sources)))


def test_dual_layer_host_schedule_has_exact_coverage(dual_layer_model, monkeypatch):
    _require_cuda_backend()
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "host")
    calculator = _calculator(dual_layer_model, PLAN, capacity=2)
    evaluator = calculator.evaluator
    node_types = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    degrees = np.array([4, 0, 2, 1, 0, 3], dtype=np.int32)
    sources = np.array([2, 2, 5, 1, 0, 5, 3, 1, 1, 4], dtype=np.int32)
    neighbor_types = node_types[sources]
    evaluator._prepare_factorized_graph(
        len(node_types), node_types, degrees, sources, neighbor_types
    )
    schedule = evaluator._dual_layer_source_schedule_for_testing()

    _assert_schedule(schedule, capacity=2)
    expected_bytes = 4 * sum(
        len(schedule[name])
        for name in (
            "tile_segment_offsets",
            "segment_source_ids",
            "segment_edge_offsets",
            "source_edges",
        )
    )
    assert evaluator.dual_layer_schedule_bytes == expected_bytes
    assert evaluator.dual_layer_schedule_preparation_explicit_scratch_bytes == 0


def test_dual_layer_reuses_topology_and_grows_workspace(dual_layer_model, monkeypatch):
    _require_cuda_backend()
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    calculator = _calculator(dual_layer_model, PLAN, capacity=4)
    first = _atoms(5)
    _evaluate(calculator, first)
    evaluator = calculator.evaluator
    initial_replacements = evaluator.dual_layer_workspace_replacement_count
    initial_schedule_bytes = evaluator.dual_layer_schedule_bytes
    initial_builds = calculator.neighbor_cache_build_count
    initial_reuses = calculator.neighbor_cache_reuse_count
    schedule = evaluator._dual_layer_source_schedule_for_testing()
    _assert_schedule(schedule, capacity=4)
    assert evaluator.dual_layer_schedule_preparation_explicit_scratch_bytes == (
        2 * np.dtype(np.int32).itemsize * evaluator.dual_layer_workspace_active_edges
    )

    moved = first.copy()
    moved.positions[0] += [0.01, -0.015, 0.005]
    actual, _ = _evaluate(calculator, moved)
    expected, _ = _evaluate(_calculator(dual_layer_model, REFERENCE_PLAN), moved)
    _assert_results(actual, expected)
    assert (
        calculator.neighbor_cache_build_count,
        calculator.neighbor_cache_reuse_count,
    ) == (initial_builds, initial_reuses + 1)
    assert evaluator.dual_layer_workspace_replacement_count == initial_replacements
    assert evaluator.dual_layer_schedule_bytes == initial_schedule_bytes

    deformed = moved.copy()
    deformed.set_cell([7.49, 7.5, 7.5], scale_atoms=True)
    cell_updates = evaluator.factorized_cell_update_count
    actual, _ = _evaluate(calculator, deformed)
    expected, _ = _evaluate(_calculator(dual_layer_model, REFERENCE_PLAN), deformed)
    _assert_results(actual, expected)
    assert (
        calculator.neighbor_cache_build_count,
        calculator.neighbor_cache_reuse_count,
    ) == (initial_builds, initial_reuses + 2)
    assert evaluator.factorized_cell_update_count == cell_updates + 1
    assert evaluator.dual_layer_schedule_bytes == initial_schedule_bytes

    larger = _atoms(9)
    actual, _ = _evaluate(calculator, larger)
    expected, _ = _evaluate(_calculator(dual_layer_model, REFERENCE_PLAN), larger)
    _assert_results(actual, expected)
    assert evaluator.dual_layer_workspace_replacement_count > initial_replacements
    assert evaluator.factorized_fallback_evaluation_count == 0


def test_dual_layer_rejects_float64(dual_layer_model):
    _require_cuda_backend()
    calculator = Symmetrix(
        dual_layer_model,
        use_kokkos=True,
        dtype="float64",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
        _debug_execution_plan=PLAN,
    )
    with pytest.raises(ValueError, match="requires float32 precision"):
        calculator.calculate(_atoms(3), properties=["energy"])


def test_dual_layer_rejects_host_backend(dual_layer_model):
    import symmetrix as native

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    if native._kokkos_default_execution_space() == "Cuda":
        pytest.skip("host-backend rejection requires a CPU build")
    calculator = _calculator(dual_layer_model, PLAN, capacity=4)
    with pytest.raises(ValueError, match="currently requires CUDA"):
        calculator.calculate(_atoms(3), properties=["energy"])


@pytest.mark.parametrize("skin", (0.0, 0.5))
def test_normal_two_layer_execution_does_not_select_dual_plan(dual_layer_model, skin):
    _require_cuda_backend()
    atoms = _atoms(5)
    expected, expected_h1_adjoint = _evaluate(
        _calculator(dual_layer_model, REFERENCE_PLAN), atoms
    )
    calculator = Symmetrix(
        dual_layer_model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=skin,
    )
    actual, actual_h1_adjoint = _evaluate(calculator, atoms)

    _assert_results(actual, expected)
    np.testing.assert_allclose(
        actual_h1_adjoint,
        expected_h1_adjoint,
        rtol=2.0e-4,
        atol=2e-5,
    )
    assert calculator.execution_plan["selected_id"] != PLAN
    assert calculator.evaluator.dual_layer_tiled_evaluation_count == 0
    assert calculator.evaluator.dual_layer_workspace_bytes == 0
    assert (calculator._neighbor_cache is None) == (skin == 0.0)


def test_dual_layer_fixed_workspace_is_selected_automatically_at_memory_boundary(
    dual_layer_model,
):
    _require_cuda_backend()
    atoms = _atoms(9)
    capacity = 4
    device_bytes = 32 * 1024**3

    reference = _calculator(dual_layer_model, REFERENCE_PLAN)
    expected, expected_h1_adjoint = _evaluate(reference, atoms)

    probe = Symmetrix(
        dual_layer_model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
        _debug_execution_plan=REFERENCE_PLAN,
    )
    probe.evaluator._set_dual_layer_workspace_receiver_limit_for_testing(capacity)
    probe_inputs = probe._mace_inputs(atoms)
    probe.evaluator._set_low_memory_device_memory_info_for_testing(
        device_bytes, device_bytes
    )
    probe.evaluator._prepare_factorized_graph(*probe_inputs[:5])
    estimates = {
        candidate["id"]: candidate["estimated_bytes"]
        for candidate in probe.execution_plan["candidates"]
    }
    tiled_estimate = estimates[PLAN]
    y_only_estimate = estimates[REFERENCE_PLAN]
    assert tiled_estimate < y_only_estimate

    calculator = Symmetrix(
        dual_layer_model,
        use_kokkos=True,
        dtype="float32",
        streamed_edges="direct",
        execution_profile="capacity",
        allow_fixed_workspace=True,
        neighbor_skin=0.5,
    )
    evaluator = calculator.evaluator
    evaluator._set_dual_layer_workspace_receiver_limit_for_testing(capacity)
    inputs = calculator._mace_inputs(atoms)
    reserve = device_bytes // 20
    available = (tiled_estimate + y_only_estimate) // 2
    evaluator._set_low_memory_device_memory_info_for_testing(
        reserve + available, device_bytes
    )
    evaluator._prepare_factorized_graph(*inputs[:5])

    assert calculator.execution_plan["selected_id"] == PLAN
    assert calculator.execution_plan["boundary_attempt"] is False
    assert evaluator.low_memory_policy == "capacity-y-only"
    assert evaluator.low_memory_selected_estimated_bytes <= tiled_estimate
    assert evaluator.dual_layer_workspace_active_receivers == capacity
    assert evaluator.low_memory_selection_reason.startswith(
        "fixed-workspace selection:"
    )

    actual, actual_h1_adjoint = _evaluate(calculator, atoms)
    _assert_results(actual, expected)
    np.testing.assert_allclose(
        actual_h1_adjoint,
        expected_h1_adjoint,
        rtol=2.0e-4,
        atol=2e-5,
    )
    assert evaluator.dual_layer_tiled_evaluation_count == 1
    assert evaluator.factorized_fallback_evaluation_count == 0
