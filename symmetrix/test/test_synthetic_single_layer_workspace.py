import math
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from symmetrix import Symmetrix


MODEL = Path(__file__).parent / "data" / "synthetic_one_interaction_mace.json"
PLAN = "mh0-single-layer-tiled-v1"
REFERENCE_PLAN = "mh0-direct-capacity-y-only"
PROPERTIES = ("energy", "energies", "forces", "stress")


def _require_cuda_backend():
    import symmetrix as native

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    if native._kokkos_default_execution_space() != "Cuda":
        pytest.skip("single-layer fixed workspace currently requires CUDA")


def _atoms():
    grid = np.indices((3, 3, 3)).reshape(3, -1).T[:9]
    positions = 0.8 + 1.55 * grid
    positions[:, 1] += 0.02 * np.arange(9)
    symbols = ["Al" if index % 2 == 0 else "N" for index in range(9)]
    return Atoms(symbols, positions=positions, cell=[7.5, 7.5, 7.5], pbc=True)


def _calculator(dtype, plan, capacity=4):
    calculator = Symmetrix(
        MODEL,
        dtype=dtype,
        streamed_edges="direct",
        execution_profile="capacity",
        neighbor_skin=0.5,
        _debug_execution_plan=plan,
    )
    calculator.evaluator._set_single_layer_workspace_receiver_limit_for_testing(
        capacity
    )
    return calculator


def _evaluate(calculator, atoms):
    calculator.calculate(atoms, properties=list(PROPERTIES))
    return {name: np.array(calculator.results[name], copy=True) for name in PROPERTIES}


@pytest.mark.parametrize("schedule", ("table", "chunk32", "tiled128x4-v1"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_synthetic_generated_single_layer_tail_workspace(monkeypatch, schedule, dtype):
    _require_cuda_backend()
    monkeypatch.setenv("SYMMETRIX_M0_RTC_SCHEDULE", schedule)
    monkeypatch.setenv("SYMMETRIX_M1_RTC_SCHEDULE", schedule)
    atoms = _atoms()
    reference = _calculator(dtype, REFERENCE_PLAN)
    expected = _evaluate(reference, atoms)
    calculator = _calculator(dtype, PLAN)
    evaluator = calculator.evaluator

    actual = _evaluate(calculator, atoms)
    tolerance = 2.0e-4 if dtype == "float32" else 1.0e-10
    for name in PROPERTIES:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=tolerance, atol=tolerance
        )

    assert calculator.execution_plan["selected_id"] == PLAN
    assert calculator.jit_operator_modules["M0"]["schedule"] == schedule
    assert calculator.jit_operator_modules["M1"]["schedule"] == schedule
    assert evaluator.single_layer_workspace_active_receivers == 4
    assert evaluator.single_layer_workspace_batch_count == math.ceil(9 / 4)
    assert evaluator.factorized_fallback_evaluation_count == 0
    assert evaluator.m0_module_forward_launch_count > 0
    assert evaluator.m0_module_reverse_launch_count > 0
    assert evaluator.generated_m1_module_forward_launch_count > 0
    assert evaluator.generated_m1_module_reverse_launch_count > 0

    addresses = evaluator._test_mh0_adjoint_addresses()
    assert addresses[2] != 0 and addresses[3] != 0
    assert addresses[2] != addresses[3]
    allocations = evaluator.mh0_a0_adjoint_allocation_count
    assert allocations == 0
    replacements = evaluator.single_layer_workspace_replacement_count
    itemsize = np.dtype(dtype).itemsize
    a_slot_bytes = 4 * 16 * 32 * itemsize
    precision_bytes = a_slot_bytes * 2 + 4 * (2 * 32 + 32) * itemsize
    receiver_payload = precision_bytes + 4 * 32 * 8 + 4 * 8 + 4 * 4
    assert evaluator.single_layer_workspace_bytes_per_receiver == (
        receiver_payload // 4
    )
    assert evaluator.single_layer_workspace_bytes == (
        receiver_payload
        + evaluator.single_layer_workspace_bytes_per_edge
        * evaluator.single_layer_workspace_planned_edges
    )

    repeated = _evaluate(calculator, atoms)
    for name in PROPERTIES:
        np.testing.assert_allclose(
            repeated[name], expected[name], rtol=tolerance, atol=tolerance
        )
    assert evaluator._test_mh0_adjoint_addresses() == addresses
    assert evaluator.mh0_a0_adjoint_allocation_count == allocations
    assert evaluator.single_layer_workspace_replacement_count == replacements


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_synthetic_single_layer_exact_memory_boundary(dtype):
    _require_cuda_backend()
    atoms = _atoms()
    device_bytes = 32 * 1024**3
    probe = _calculator(dtype, REFERENCE_PLAN)
    inputs = probe._mace_inputs(atoms)
    probe.evaluator._set_low_memory_device_memory_info_for_testing(
        device_bytes, device_bytes
    )
    probe.evaluator._prepare_factorized_graph(*inputs[:5])
    estimates = {
        candidate["id"]: candidate["estimated_bytes"]
        for candidate in probe.execution_plan["candidates"]
    }
    tiled_estimate = estimates[PLAN]
    reserve = device_bytes // 20

    for available, boundary_attempt in (
        (tiled_estimate, False),
        (tiled_estimate - 1, True),
    ):
        calculator = Symmetrix(
            MODEL,
            dtype=dtype,
            streamed_edges="direct",
            execution_profile="capacity",
            allow_fixed_workspace=True,
            neighbor_skin=0.5,
        )
        calculator.evaluator._set_single_layer_workspace_receiver_limit_for_testing(4)
        boundary_inputs = calculator._mace_inputs(atoms)
        calculator.evaluator._set_low_memory_device_memory_info_for_testing(
            reserve + available, device_bytes
        )
        calculator.evaluator._prepare_factorized_graph(*boundary_inputs[:5])
        assert calculator.execution_plan["selected_id"] == PLAN
        assert calculator.execution_plan["boundary_attempt"] is boundary_attempt
        assert calculator.evaluator.low_memory_selected_estimated_bytes == (
            tiled_estimate
        )
