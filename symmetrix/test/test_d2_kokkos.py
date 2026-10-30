import json
from pathlib import Path

import numpy as np
import pytest

import symmetrix


REFERENCE_CASES = (
    Path(__file__).parent / "data" / "dispersion" / "torch_dftd_0_5_3_cases.json"
)


def _d2_case():
    payload = json.loads(REFERENCE_CASES.read_text())
    return next(case for case in payload["cases"] if case["method"] == "d2")


def _evaluator(class_name, functional, cutoff):
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    return getattr(symmetrix, class_name)(functional, cutoff)


def _host_graph(positions, cutoff):
    positions = np.asarray(positions, dtype=np.float64)
    offsets = [0]
    sources = []
    for receiver, receiver_position in enumerate(positions):
        for source, source_position in enumerate(positions):
            if (
                receiver != source
                and np.linalg.norm(source_position - receiver_position) < cutoff
            ):
                sources.append(source)
        offsets.append(len(sources))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(sources, dtype=np.int64),
        np.zeros((len(sources), 3), dtype=np.int64),
    )


def _prepare_host(evaluator, numbers, positions, cell, cutoff):
    offsets, sources, shifts = _host_graph(positions, cutoff)
    evaluator.prepare_host_graph(
        np.asarray(numbers, dtype=np.int64),
        offsets,
        sources,
        shifts,
        np.asarray(positions, dtype=np.float64),
        np.asarray(cell, dtype=np.float64),
        cutoff,
    )


@pytest.mark.parametrize(
    ("property_name", "compute_flags"),
    [
        ("energy", (False, False)),
        ("atomic_energies", (False, False)),
        ("forces", (True, False)),
        ("stress", (False, True)),
    ],
)
def test_d2_result_exports_only_requested_property(property_name, compute_flags):
    evaluator = _evaluator("D2Kokkos", "pbe", 4.0)
    _prepare_host(
        evaluator,
        [6, 8],
        [[0.0, 0.0, 0.0], [1.4, 0.2, 0.1]],
        np.eye(3) * 8.0,
        4.0,
    )

    evaluator.compute(*compute_flags)
    result = evaluator.result([property_name])

    assert set(result) == {property_name}


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D2Kokkos", 2e-12, 2e-12), ("D2KokkosFloat", 3e-5, 3e-6)],
)
def test_d2_matches_pinned_torch_dftd_oracle(class_name, rtol, atol):
    case = _d2_case()
    positions = np.asarray(case["positions_angstrom"])
    evaluator = _evaluator(class_name, case["functional"], case["cutoff_angstrom"])
    _prepare_host(
        evaluator,
        case["atomic_numbers"],
        positions,
        np.eye(3) * 30.0,
        case["cutoff_angstrom"],
    )

    evaluator.compute(True, False)
    result = evaluator.result()

    assert result["energy"] == pytest.approx(case["energy_ev"], rel=rtol, abs=atol)
    np.testing.assert_allclose(
        result["forces"], case["forces_ev_per_angstrom"], rtol=rtol, atol=atol
    )
    assert result["atomic_energies"].shape == (len(positions),)
    assert np.all(np.isfinite(result["atomic_energies"]))
    assert np.sum(result["atomic_energies"]) == pytest.approx(
        result["energy"], rel=rtol, abs=atol
    )
    np.testing.assert_allclose(np.sum(result["forces"], axis=0), 0.0, atol=atol)
    assert result["graph_generation"] == 1
    assert evaluator.graph_diagnostics()["num_edges"] == case["directed_edges"]


@pytest.mark.parametrize(
    ("class_name", "step", "force_tolerance", "stress_tolerance"),
    [("D2Kokkos", 2e-5, 2e-8, 2e-9), ("D2KokkosFloat", 2e-3, 2e-5, 2e-6)],
)
def test_d2_forces_and_stress_match_independent_finite_differences(
    class_name, step, force_tolerance, stress_tolerance
):
    numbers = np.array([6, 8])
    positions = np.array([[1.1, 0.8, 1.3], [3.6, 2.1, 1.9]])
    cell = np.array([[8.0, 0.3, 0.1], [0.0, 8.5, 0.2], [0.0, 0.0, 9.0]])
    cutoff = 4.0
    evaluator = _evaluator(class_name, "pbe", cutoff)
    evaluator.prepare_periodic_graph(
        numbers, positions, cell, np.linalg.inv(cell), cutoff
    )
    evaluator.compute(True, True)
    analytic = evaluator.result()

    def energy_at(updated_positions, updated_cell):
        evaluator.update_geometry(
            updated_positions, updated_cell, np.linalg.inv(updated_cell)
        )
        evaluator.compute(False, False)
        return evaluator.result()["energy"]

    numerical_forces = np.empty_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical_forces[atom, axis] = -(
                energy_at(plus, cell) - energy_at(minus, cell)
            ) / (2.0 * step)

    volume = abs(np.linalg.det(cell))
    numerical_stress = np.empty(6)
    components = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    for component, (row, column) in enumerate(components):
        plus_strain = np.eye(3)
        minus_strain = np.eye(3)
        plus_strain[row, column] += step
        minus_strain[row, column] -= step
        divisor = 2.0 * step * volume
        if row != column:
            plus_strain[column, row] += step
            minus_strain[column, row] -= step
            divisor *= 2.0
        plus_cell = cell @ plus_strain
        minus_cell = cell @ minus_strain
        plus_positions = positions @ plus_strain
        minus_positions = positions @ minus_strain
        numerical_stress[component] = (
            energy_at(plus_positions, plus_cell)
            - energy_at(minus_positions, minus_cell)
        ) / divisor

    np.testing.assert_allclose(
        analytic["forces"], numerical_forces, rtol=2e-3, atol=force_tolerance
    )
    np.testing.assert_allclose(
        analytic["stress"], numerical_stress, rtol=2e-3, atol=stress_tolerance
    )
    np.testing.assert_allclose(
        np.sum(analytic["forces"], axis=0), 0.0, atol=force_tolerance
    )
    assert analytic["atomic_energies"][0] == pytest.approx(
        analytic["atomic_energies"][1], abs=force_tolerance
    )


@pytest.mark.parametrize("class_name", ["D2Kokkos", "D2KokkosFloat"])
def test_d2_empty_graph_returns_exact_zero_outputs(class_name):
    evaluator = _evaluator(class_name, "pbe", 4.0)
    evaluator.prepare_host_graph(
        np.array([], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3)),
        np.eye(3),
        4.0,
    )

    evaluator.compute(True, True)
    result = evaluator.result()

    assert result["energy"] == 0.0
    np.testing.assert_array_equal(result["atomic_energies"], np.empty((0,)))
    np.testing.assert_array_equal(result["forces"], np.empty((0, 3)))
    np.testing.assert_array_equal(result["stress"], np.zeros(6))


@pytest.mark.parametrize("class_name", ["D2Kokkos", "D2KokkosFloat"])
def test_d2_result_availability_tracks_each_compute_generation(class_name):
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    evaluator = _evaluator(class_name, "pbe", 4.0)
    _prepare_host(evaluator, numbers, positions, cell, 4.0)
    evaluator.compute(True, True)
    assert {"forces", "stress"} <= evaluator.result().keys()

    updated_positions = positions.copy()
    updated_positions[1, 0] += 0.05
    evaluator.update_geometry(updated_positions, cell, np.linalg.inv(cell))
    evaluator.compute(False, False)

    default_result = evaluator.result()
    assert "forces" not in default_result
    assert "stress" not in default_result
    with pytest.raises(RuntimeError, match="forces.*unavailable"):
        evaluator.result(["forces"])
    with pytest.raises(RuntimeError, match="stress.*unavailable"):
        evaluator.result(["stress"])


@pytest.mark.parametrize("class_name", ["D2Kokkos", "D2KokkosFloat"])
def test_d2_throwing_stress_compute_invalidates_previous_result(class_name):
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    valid_cell = np.eye(3) * 20.0
    evaluator = _evaluator(class_name, "pbe", 4.0)
    _prepare_host(evaluator, numbers, positions, valid_cell, 4.0)
    evaluator.compute(True, True)
    assert evaluator.result()["energy"] < 0.0

    singular_cell = valid_cell.copy()
    singular_cell[2] = 0.0
    evaluator.prepare_host_graph(
        numbers,
        np.array([0, 1, 2]),
        np.array([1, 0]),
        np.zeros((2, 3), dtype=np.int64),
        positions,
        singular_cell,
        4.0,
    )
    with pytest.raises(ValueError, match="finite positive cell volume"):
        evaluator.compute(True, True)
    with pytest.raises(RuntimeError, match="unavailable"):
        evaluator.result()


def test_d2_rejects_atomic_numbers_outside_its_parameter_table():
    evaluator = _evaluator("D2Kokkos", "pbe", 4.0)

    with pytest.raises(ValueError, match="D2 atomic number"):
        evaluator.prepare_host_graph(
            np.array([87]),
            np.array([0, 0]),
            np.array([], dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            np.zeros((1, 3)),
            np.eye(3),
            4.0,
        )
