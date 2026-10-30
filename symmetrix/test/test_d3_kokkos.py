import json
from pathlib import Path

import numpy as np
import pytest
from ase.neighborlist import primitive_neighbor_list

import symmetrix


REFERENCE_CASES = (
    Path(__file__).parent / "data" / "dispersion" / "torch_dftd_0_5_3_cases.json"
)


def _d3_cases():
    payload = json.loads(REFERENCE_CASES.read_text())
    return [case for case in payload["cases"] if case["method"] == "d3"]


def _oracle_energy_cases():
    return [case for case in _d3_cases() if "cutoff-boundary" not in case["coverage"]]


def _oracle_force_cases():
    return [case for case in _oracle_energy_cases() if "forces_ev_per_angstrom" in case]


def _evaluator(class_name, damping, functional, pair_cutoff, coordination_cutoff):
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    return getattr(symmetrix, class_name)(
        damping, functional, pair_cutoff, coordination_cutoff
    )


def _host_graph(positions, cell, pbc, cutoff):
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    receivers, sources, shifts = primitive_neighbor_list(
        "ijS", np.asarray(pbc, dtype=bool), cell, positions, cutoff
    )
    order = np.argsort(receivers, kind="stable")
    receivers = receivers[order]
    sources = sources[order]
    shifts = shifts[order]
    offsets = np.searchsorted(receivers, np.arange(len(positions) + 1))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(sources, dtype=np.int64),
        np.asarray(shifts, dtype=np.int64),
    )


def _prepare_case(evaluator, case):
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = (
        np.eye(3) * 40.0
        if case["cell_angstrom"] is None
        else np.asarray(case["cell_angstrom"], dtype=np.float64)
    )
    build_cutoff = max(case["cutoff_angstrom"], case["coordination_cutoff_angstrom"])
    offsets, sources, shifts = _host_graph(positions, cell, case["pbc"], build_cutoff)
    evaluator.prepare_host_graph(
        np.asarray(case["atomic_numbers"], dtype=np.int64),
        offsets,
        sources,
        shifts,
        positions,
        cell,
        build_cutoff,
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
def test_d3_result_exports_only_requested_property(property_name, compute_flags):
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 3.5)
    positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.2, 0.1]])
    offsets, sources, shifts = _host_graph(
        positions, np.eye(3) * 8.0, [False, False, False], 4.0
    )
    evaluator.prepare_host_graph(
        np.array([6, 8]),
        offsets,
        sources,
        shifts,
        positions,
        np.eye(3) * 8.0,
        4.0,
    )

    evaluator.compute(*compute_flags)
    result = evaluator.result([property_name])

    assert set(result) == {property_name}


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 3e-12, 3e-12), ("D3KokkosFloat", 5e-5, 5e-6)],
)
@pytest.mark.parametrize("case", _oracle_energy_cases(), ids=lambda case: case["name"])
def test_d3_energy_matches_pinned_torch_dftd_oracle(class_name, rtol, atol, case):
    evaluator = _evaluator(
        class_name,
        case["damping"],
        case["functional"],
        case["cutoff_angstrom"],
        case["coordination_cutoff_angstrom"],
    )
    _prepare_case(evaluator, case)

    evaluator.compute(False, False)
    result = evaluator.result()

    assert result["energy"] == pytest.approx(case["energy_ev"], rel=rtol, abs=atol)
    assert result["atomic_energies"].shape == (len(case["atomic_numbers"]),)
    assert np.all(np.isfinite(result["atomic_energies"]))
    assert np.sum(result["atomic_energies"]) == pytest.approx(
        result["energy"], rel=rtol, abs=atol
    )
    assert result["coordination_numbers"].shape == (len(case["atomic_numbers"]),)
    assert np.all(np.isfinite(result["coordination_numbers"]))
    assert result["energy_coordination_derivatives"].shape == (
        len(case["atomic_numbers"]),
    )
    assert np.all(np.isfinite(result["energy_coordination_derivatives"]))


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 4e-11, 4e-11), ("D3KokkosFloat", 2e-4, 2e-5)],
)
@pytest.mark.parametrize("case", _oracle_force_cases(), ids=lambda case: case["name"])
def test_d3_forces_match_pinned_torch_dftd_oracle(class_name, rtol, atol, case):
    evaluator = _evaluator(
        class_name,
        case["damping"],
        case["functional"],
        case["cutoff_angstrom"],
        case["coordination_cutoff_angstrom"],
    )
    _prepare_case(evaluator, case)

    evaluator.compute(True, False)
    result = evaluator.result()

    np.testing.assert_allclose(
        result["forces"],
        case["forces_ev_per_angstrom"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.sum(result["forces"], axis=0), 0.0, rtol=0.0, atol=atol
    )


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 5e-11, 5e-11), ("D3KokkosFloat", 3e-4, 3e-5)],
)
@pytest.mark.parametrize(
    "case",
    [case for case in _oracle_energy_cases() if "stress_ev_per_angstrom3" in case],
    ids=lambda case: case["name"],
)
def test_d3_stress_matches_pinned_torch_dftd_oracle(class_name, rtol, atol, case):
    evaluator = _evaluator(
        class_name,
        case["damping"],
        case["functional"],
        case["cutoff_angstrom"],
        case["coordination_cutoff_angstrom"],
    )
    _prepare_case(evaluator, case)

    evaluator.compute(False, True)

    np.testing.assert_allclose(
        evaluator.result()["stress"],
        case["stress_ev_per_angstrom3"],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_stress_reduction_launches_once_only_when_requested(class_name):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    positions = np.array([[0.2, 0.3, 0.4], [1.5, 0.4, 0.7], [2.1, 1.8, 0.9]])
    cell = np.eye(3) * 8.0
    offsets, sources, shifts = _host_graph(positions, cell, [False, False, False], 4.0)
    evaluator.prepare_host_graph(
        np.array([6, 8, 7]), offsets, sources, shifts, positions, cell, 4.0
    )

    assert evaluator._stress_reduction_diagnostics() == {
        "stress_reduction_launch_count": 0
    }

    evaluator._pair_accumulation_diagnostics("team", True, True)
    assert evaluator._stress_reduction_diagnostics() == {
        "stress_reduction_launch_count": 0
    }

    evaluator.compute(True, False)
    assert evaluator._stress_reduction_diagnostics() == {
        "stress_reduction_launch_count": 0
    }

    evaluator.compute(False, True)
    assert evaluator._stress_reduction_diagnostics() == {
        "stress_reduction_launch_count": 1
    }


@pytest.mark.parametrize(
    ("class_name", "step", "force_atol", "stress_atol"),
    [
        ("D3Kokkos", 2e-5, 3e-8, 3e-9),
        ("D3KokkosFloat", 2e-3, 4e-5, 4e-6),
    ],
)
@pytest.mark.parametrize("damping", ["zero", "bj", "zerom", "bjm"])
def test_d3_forces_and_stress_match_independent_finite_differences(
    class_name, step, force_atol, stress_atol, damping
):
    numbers = np.array([6, 8, 7])
    positions = np.array([[1.1, 0.8, 1.3], [3.4, 2.0, 1.8], [2.2, 3.7, 2.9]])
    cell = np.array([[8.0, 0.3, 0.1], [0.0, 8.5, 0.2], [0.0, 0.0, 9.0]])
    pair_cutoff = 4.5
    coordination_cutoff = 3.2
    evaluator = _evaluator(class_name, damping, "pbe", pair_cutoff, coordination_cutoff)
    evaluator.prepare_periodic_graph(
        numbers, positions, cell, np.linalg.inv(cell), pair_cutoff
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
        strain = np.zeros((3, 3))
        strain[row, column] = 1.0 if row == column else 0.5
        strain[column, row] = 1.0 if row == column else 0.5
        plus_transform = np.eye(3) + step * strain
        minus_transform = np.eye(3) - step * strain
        numerical_stress[component] = (
            energy_at(positions @ plus_transform, cell @ plus_transform)
            - energy_at(positions @ minus_transform, cell @ minus_transform)
        ) / (2.0 * step * volume)

    np.testing.assert_allclose(
        analytic["forces"], numerical_forces, rtol=3e-3, atol=force_atol
    )
    np.testing.assert_allclose(
        analytic["stress"], numerical_stress, rtol=3e-3, atol=stress_atol
    )
    np.testing.assert_allclose(
        np.sum(analytic["forces"], axis=0), 0.0, rtol=0.0, atol=force_atol
    )


@pytest.mark.parametrize(
    ("pair_cutoff", "coordination_cutoff"), [(3.1, 4.0), (4.0, 3.1)]
)
def test_d3_force_respects_independent_pair_and_coordination_cutoffs(
    pair_cutoff, coordination_cutoff
):
    numbers = np.array([6, 8, 7])
    positions = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [1.4, 0.8, 0.0]])
    cell = np.eye(3) * 20.0
    build_cutoff = max(pair_cutoff, coordination_cutoff)
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", pair_cutoff, coordination_cutoff)
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, build_cutoff)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, build_cutoff
    )
    evaluator.compute(True, False)
    analytic = evaluator.result()["forces"]

    step = 2e-5
    numerical = np.empty_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            displaced = []
            for sign in (1.0, -1.0):
                trial = positions.copy()
                trial[atom, axis] += sign * step
                evaluator.update_geometry(trial, cell, np.linalg.inv(cell))
                evaluator.compute(False, False)
                displaced.append(evaluator.result()["energy"])
            numerical[atom, axis] = -(displaced[0] - displaced[1]) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=2e-3, atol=3e-8)


def test_d3_requested_output_gating_allows_force_without_cell_volume():
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    evaluator.prepare_host_graph(
        np.array([6, 8]),
        np.array([0, 1, 2]),
        np.array([1, 0]),
        np.zeros((2, 3), dtype=np.int64),
        positions,
        np.zeros((3, 3)),
        4.0,
    )

    evaluator.compute(True, False)
    assert np.linalg.norm(evaluator.result()["forces"]) > 0.0

    with pytest.raises(ValueError, match="finite positive cell volume"):
        evaluator.compute(False, True)


def test_d3_energy_oracles_cover_every_damping_and_both_functionals():
    cases = _oracle_energy_cases()

    assert {case["damping"] for case in cases} == {"zero", "bj", "zerom", "bjm"}
    assert {case["functional"] for case in cases} == {"pbe", "pbe0"}
    assert any(
        case["damping"] == "bjm"
        and case["functional"] == "pbe"
        and case["directed_edges"] > 0
        and case["energy_ev"] < 0.0
        for case in cases
    )


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 3e-12, 3e-12), ("D3KokkosFloat", 5e-5, 5e-6)],
)
def test_d3_interpolation_remains_finite_at_high_coordination(class_name, rtol, atol):
    case = next(case for case in _d3_cases() if "high-coordination" in case["coverage"])
    evaluator = _evaluator(
        class_name,
        case["damping"],
        case["functional"],
        case["cutoff_angstrom"],
        case["coordination_cutoff_angstrom"],
    )
    _prepare_case(evaluator, case)

    evaluator.compute(False, False)
    result = evaluator.result()

    assert np.isfinite(result["energy"])
    assert result["energy"] == pytest.approx(case["energy_ev"], rel=rtol, abs=atol)
    maximum_coordination = float(np.max(result["coordination_numbers"]))
    assert maximum_coordination > 20.0
    assert np.exp(-4.0 * (maximum_coordination - 10.0) ** 2) == 0.0


@pytest.mark.parametrize(
    ("class_name", "tolerance"),
    [("D3Kokkos", 1e-14), ("D3KokkosFloat", 2e-7)],
)
def test_d3_interpolation_fallback_tie_uses_lowest_table_index(class_name, tolerance):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 4.0)

    fallback = evaluator._interpolation_diagnostics(1, 1, np.inf, np.inf)

    assert fallback["c6"] == pytest.approx(1.8085356084766697, abs=tolerance)
    assert fallback["receiver_derivative"] == 0.0
    assert fallback["source_derivative"] == 0.0


@pytest.mark.parametrize(
    ("class_name", "c6_rtol", "c6_atol"),
    [
        ("D3Kokkos", 2e-14, 2e-14),
        ("D3KokkosFloat", 5e-5, 5e-6),
    ],
)
@pytest.mark.parametrize(
    ("receiver_atomic_number", "source_atomic_number"),
    [(1, 1), (6, 6), (6, 8), (8, 22), (22, 38), (94, 94)],
)
@pytest.mark.parametrize(
    ("receiver_coordination", "source_coordination"),
    [
        (0.0, 0.0),
        (0.2, 4.7),
        (3.5, 2.25),
        (20.0, 20.0),
        (1.0e150, 1.0e150),
        (np.inf, np.inf),
    ],
)
def test_d3_factorized_interpolation_matches_direct_reference(
    class_name,
    c6_rtol,
    c6_atol,
    receiver_atomic_number,
    source_atomic_number,
    receiver_coordination,
    source_coordination,
):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 4.0)

    diagnostics = evaluator._interpolation_diagnostics(
        receiver_atomic_number,
        source_atomic_number,
        receiver_coordination,
        source_coordination,
    )

    direct = np.array(
        [
            diagnostics["c6"],
            diagnostics["receiver_derivative"],
            diagnostics["source_derivative"],
        ]
    )
    factorized = np.array(
        [
            diagnostics["factorized_c6"],
            diagnostics["factorized_receiver_derivative"],
            diagnostics["factorized_source_derivative"],
        ]
    )
    assert np.all(np.isfinite(factorized))
    np.testing.assert_allclose(factorized[0], direct[0], rtol=c6_rtol, atol=c6_atol)
    if class_name == "D3Kokkos":
        np.testing.assert_allclose(
            factorized[1:],
            direct[1:],
            rtol=2e-12,
            atol=5e-12,
        )
    if not np.isfinite(receiver_coordination) or receiver_coordination >= 1.0e100:
        np.testing.assert_array_equal(factorized[1:], 0.0)
        tie_fallback = evaluator._interpolation_diagnostics(
            receiver_atomic_number,
            source_atomic_number,
            np.inf,
            np.inf,
        )
        assert factorized[0] == tie_fallback["factorized_c6"]


def test_d3_float_factorized_derivative_avoids_direct_cancellation_error():
    float_evaluator = _evaluator("D3KokkosFloat", "bj", "pbe", 4.0, 4.0)
    double_evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)
    receiver_coordination = 0.2
    source_coordination = 4.7

    diagnostics = float_evaluator._interpolation_diagnostics(
        94,
        94,
        receiver_coordination,
        source_coordination,
    )
    reference = double_evaluator._interpolation_diagnostics(
        94,
        94,
        float(np.float32(receiver_coordination)),
        float(np.float32(source_coordination)),
    )
    expected = reference["source_derivative"]

    assert diagnostics["factorized_source_derivative"] == pytest.approx(
        expected,
        rel=1e-3,
        abs=2e-5,
    )
    assert not np.isclose(
        diagnostics["source_derivative"],
        expected,
        rtol=1e-3,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    (
        "receiver_atomic_number",
        "source_atomic_number",
        "receiver_coordination",
        "source_coordination",
        "derivative_name",
    ),
    [
        (55, 41, 0.2, 4.7, "source_derivative"),
        (88, 57, 3.5, 2.25, "receiver_derivative"),
    ],
)
def test_d3_float_factorized_derivative_resists_cancellation(
    receiver_atomic_number,
    source_atomic_number,
    receiver_coordination,
    source_coordination,
    derivative_name,
):
    float_evaluator = _evaluator("D3KokkosFloat", "bj", "pbe", 4.0, 4.0)
    double_evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)

    actual = float_evaluator._interpolation_diagnostics(
        receiver_atomic_number,
        source_atomic_number,
        receiver_coordination,
        source_coordination,
    )[f"factorized_{derivative_name}"]
    expected = double_evaluator._interpolation_diagnostics(
        receiver_atomic_number,
        source_atomic_number,
        float(np.float32(receiver_coordination)),
        float(np.float32(source_coordination)),
    )[derivative_name]

    assert actual == pytest.approx(expected, rel=2e-4, abs=1e-5)


@pytest.mark.parametrize(
    ("receiver_coordination", "source_coordination"),
    [(0.2, 4.7), (3.5, 2.25)],
)
def test_d3_float_factorized_interpolation_matches_double_for_all_element_pairs(
    receiver_coordination,
    source_coordination,
):
    float_evaluator = _evaluator("D3KokkosFloat", "bj", "pbe", 4.0, 4.0)
    double_evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)
    rounded_receiver = float(np.float32(receiver_coordination))
    rounded_source = float(np.float32(source_coordination))

    for receiver_atomic_number in range(1, 95):
        for source_atomic_number in range(1, 95):
            diagnostics = float_evaluator._interpolation_diagnostics(
                receiver_atomic_number,
                source_atomic_number,
                receiver_coordination,
                source_coordination,
            )
            reference = double_evaluator._interpolation_diagnostics(
                receiver_atomic_number,
                source_atomic_number,
                rounded_receiver,
                rounded_source,
            )
            swapped = float_evaluator._interpolation_diagnostics(
                source_atomic_number,
                receiver_atomic_number,
                source_coordination,
                receiver_coordination,
            )

            np.testing.assert_allclose(
                diagnostics["factorized_c6"],
                reference["c6"],
                rtol=3e-7,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                [
                    diagnostics["factorized_receiver_derivative"],
                    diagnostics["factorized_source_derivative"],
                ],
                [
                    reference["receiver_derivative"],
                    reference["source_derivative"],
                ],
                rtol=1e-3,
                atol=2e-5,
            )
            np.testing.assert_allclose(
                [
                    diagnostics["factorized_c6"],
                    diagnostics["factorized_receiver_derivative"],
                    diagnostics["factorized_source_derivative"],
                ],
                [
                    swapped["factorized_c6"],
                    swapped["factorized_source_derivative"],
                    swapped["factorized_receiver_derivative"],
                ],
                rtol=2e-5,
                atol=1e-5,
            )


@pytest.mark.parametrize(
    ("receiver_coordination", "source_coordination"),
    [(np.inf, 0.8), (0.8, np.inf), (1.0e308, 0.8), (0.8, 1.0e308)],
)
def test_d3_asymmetric_fallback_matches_direct_for_every_element_pair(
    receiver_coordination, source_coordination
):
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)

    for receiver_atomic_number in range(1, 95):
        for source_atomic_number in range(1, 95):
            diagnostics = evaluator._interpolation_diagnostics(
                receiver_atomic_number,
                source_atomic_number,
                receiver_coordination,
                source_coordination,
            )
            direct = np.array(
                [
                    diagnostics["c6"],
                    diagnostics["receiver_derivative"],
                    diagnostics["source_derivative"],
                ]
            )
            factorized = np.array(
                [
                    diagnostics["factorized_c6"],
                    diagnostics["factorized_receiver_derivative"],
                    diagnostics["factorized_source_derivative"],
                ]
            )
            np.testing.assert_allclose(factorized, direct, rtol=0.0, atol=0.0)
            np.testing.assert_array_equal(factorized[1:], 0.0)


@pytest.mark.parametrize(
    ("class_name", "step", "rtol", "atol"),
    [
        ("D3Kokkos", 1e-5, 2e-8, 2e-9),
        ("D3KokkosFloat", 2e-3, 3e-3, 2e-4),
    ],
)
def test_d3_interpolation_derivatives_match_independent_finite_difference(
    class_name, step, rtol, atol
):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 4.0)
    receiver_coordination = 1.7
    source_coordination = 0.8
    analytic = evaluator._interpolation_diagnostics(
        6, 8, receiver_coordination, source_coordination
    )

    def c6(receiver_cn, source_cn):
        return evaluator._interpolation_diagnostics(6, 8, receiver_cn, source_cn)["c6"]

    numerical_receiver = (
        c6(receiver_coordination + step, source_coordination)
        - c6(receiver_coordination - step, source_coordination)
    ) / (2.0 * step)
    numerical_source = (
        c6(receiver_coordination, source_coordination + step)
        - c6(receiver_coordination, source_coordination - step)
    ) / (2.0 * step)

    assert analytic["receiver_derivative"] == pytest.approx(
        numerical_receiver, rel=rtol, abs=atol
    )
    assert analytic["source_derivative"] == pytest.approx(
        numerical_source, rel=rtol, abs=atol
    )
    assert abs(numerical_receiver - numerical_source) > 1e-3


@pytest.mark.parametrize(
    ("class_name", "step", "rtol", "atol"),
    [
        ("D3Kokkos", 1e-5, 3e-8, 2e-10),
        ("D3KokkosFloat", 2e-3, 4e-3, 2e-5),
    ],
)
def test_d3_node_energy_coordination_derivatives_match_pair_finite_difference(
    class_name, step, rtol, atol
):
    numbers = np.array([6, 8])
    distance = 2.7
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 4.0)
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, 4.0
    )
    evaluator.compute(False, False)
    result = evaluator.result()
    receiver_coordination, source_coordination = result["coordination_numbers"]

    def pair_energy(receiver_cn, source_cn):
        return evaluator._pair_energy_diagnostics(
            6, 8, distance, receiver_cn, source_cn
        )

    numerical_receiver = (
        pair_energy(receiver_coordination + step, source_coordination)
        - pair_energy(receiver_coordination - step, source_coordination)
    ) / (2.0 * step)
    numerical_source = (
        pair_energy(receiver_coordination, source_coordination + step)
        - pair_energy(receiver_coordination, source_coordination - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        result["energy_coordination_derivatives"],
        [numerical_receiver, numerical_source],
        rtol=rtol,
        atol=atol,
    )
    assert abs(numerical_receiver - numerical_source) > 1e-6


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_pair_cutoff_changes_energy_membership_but_not_coordination_numbers(
    class_name,
):
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, 4.0)
    results = []
    for pair_cutoff in (2.9, 3.1):
        evaluator = _evaluator(class_name, "bj", "pbe", pair_cutoff, 4.0)
        evaluator.prepare_host_graph(
            numbers, offsets, sources, shifts, positions, cell, 4.0
        )
        evaluator.compute(False, False)
        results.append(evaluator.result())

    np.testing.assert_allclose(
        results[0]["coordination_numbers"],
        results[1]["coordination_numbers"],
        rtol=0.0,
        atol=0.0,
    )
    assert results[0]["energy"] == 0.0
    assert results[1]["energy"] < 0.0


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_coordination_cutoff_changes_cn_without_changing_pair_membership(
    class_name,
):
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, 4.0)
    results = []
    for coordination_cutoff in (2.9, 3.1):
        evaluator = _evaluator(class_name, "bj", "pbe", 3.5, coordination_cutoff)
        evaluator.prepare_host_graph(
            numbers, offsets, sources, shifts, positions, cell, 4.0
        )
        evaluator.compute(False, False)
        results.append(evaluator.result())

    np.testing.assert_array_equal(results[0]["coordination_numbers"], [0.0, 0.0])
    assert np.all(results[1]["coordination_numbers"] > 0.0)
    assert results[0]["energy"] != pytest.approx(results[1]["energy"], abs=1e-12)


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_exact_pair_cutoff_is_excluded_and_cn_is_retained(class_name):
    case = next(case for case in _d3_cases() if "cutoff-boundary" in case["coverage"])
    evaluator = _evaluator(
        class_name,
        case["damping"],
        case["functional"],
        case["cutoff_angstrom"],
        case["coordination_cutoff_angstrom"] + 0.1,
    )
    _prepare_case(
        evaluator,
        {**case, "coordination_cutoff_angstrom": case["cutoff_angstrom"] + 0.1},
    )

    evaluator.compute(False, False)
    result = evaluator.result()

    assert result["energy"] == 0.0
    np.testing.assert_array_equal(result["atomic_energies"], [0.0, 0.0])
    assert np.all(result["coordination_numbers"] > 0.0)


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_directed_pair_partitions_energy_equally(class_name):
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 4.0)
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, 4.0
    )

    evaluator.compute(False, False)
    result = evaluator.result()

    assert result["energy"] < 0.0
    assert result["atomic_energies"][0] == pytest.approx(
        result["atomic_energies"][1], rel=2e-6, abs=1e-9
    )


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_empty_graph_returns_exact_zero_energy_workspaces(class_name):
    evaluator = _evaluator(class_name, "bjm", "pbe", 4.0, 4.0)
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
    np.testing.assert_array_equal(result["coordination_numbers"], np.empty((0,)))
    np.testing.assert_array_equal(
        result["energy_coordination_derivatives"], np.empty((0,))
    )
    np.testing.assert_array_equal(result["forces"], np.empty((0, 3)))
    np.testing.assert_array_equal(result["stress"], np.zeros(6))


def test_d3_compute_and_result_require_a_prepared_computed_graph():
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)

    with pytest.raises(RuntimeError, match="prepared"):
        evaluator.compute(False, False)
    with pytest.raises(RuntimeError, match="unavailable"):
        evaluator.result()


def test_d3_geometry_update_invalidates_result_without_rebuilding_graph():
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, 4.0
    )
    evaluator.compute(False, False)
    assert evaluator.result()["graph_generation"] == 1

    updated_positions = positions.copy()
    updated_positions[1, 0] = 2.8
    evaluator.update_geometry(updated_positions, cell, np.linalg.inv(cell))

    with pytest.raises(RuntimeError, match="unavailable"):
        evaluator.result()
    evaluator.compute(False, False)
    assert evaluator.result()["graph_generation"] == 1
    assert evaluator.graph_diagnostics()["geometry_update_count"] == 2


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_result_availability_tracks_each_compute_generation(class_name):
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 4.0)
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, 4.0
    )
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


def test_d3_reused_stress_compute_performs_no_view_allocations():
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    cell = np.eye(3) * 20.0
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)
    offsets, sources, shifts = _host_graph(positions, cell, [False] * 3, 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, 4.0
    )
    evaluator.compute(True, True)
    allocation_count = evaluator.diagnostics()["view_allocation_count"]

    for displacement in (0.01, 0.02, 0.03):
        updated_positions = positions.copy()
        updated_positions[1, 0] += displacement
        evaluator.update_geometry(updated_positions, cell, np.linalg.inv(cell))
        evaluator.compute(True, True)
        assert evaluator.diagnostics()["view_allocation_count"] == allocation_count


def test_d3_throwing_stress_compute_invalidates_previous_result():
    numbers = np.array([6, 8])
    positions = np.array([[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    valid_cell = np.eye(3) * 20.0
    evaluator = _evaluator("D3Kokkos", "bj", "pbe", 4.0, 4.0)
    offsets, sources, shifts = _host_graph(positions, valid_cell, [False] * 3, 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, valid_cell, 4.0
    )
    evaluator.compute(True, True)
    assert evaluator.result()["energy"] < 0.0

    singular_cell = valid_cell.copy()
    singular_cell[2] = 0.0
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, singular_cell, 4.0
    )
    with pytest.raises(ValueError, match="finite positive cell volume"):
        evaluator.compute(True, True)
    with pytest.raises(RuntimeError, match="unavailable"):
        evaluator.result()


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_pair_diagnostic_invalidates_production_result(class_name):
    numbers = np.array([6, 8, 7])
    positions = np.array([[0.1, 0.2, 0.3], [1.6, 0.4, 0.7], [2.4, 1.8, 0.9]])
    cell = np.eye(3) * 10.0
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    offsets, sources, shifts = _host_graph(positions, cell, [False, False, False], 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, 4.0
    )
    evaluator.compute(True, True)
    original_energy = evaluator.result()["energy"]

    evaluator._pair_accumulation_diagnostics("team", True, True)

    with pytest.raises(RuntimeError, match="unavailable"):
        evaluator.result()

    evaluator.compute(True, True)
    assert evaluator.result()["energy"] == pytest.approx(original_energy)


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 2e-15, 2e-15), ("D3KokkosFloat", 2e-6, 2e-7)],
)
def test_d3_coordination_switch_cache_overwrites_active_edges_and_retains_capacity(
    class_name, rtol, atol
):
    coordination_cutoff = 4.0
    evaluator = _evaluator(class_name, "bj", "pbe", 5.0, coordination_cutoff)
    positions = np.array(
        [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.0, 0.0, 0.0], [4.5, 0.0, 0.0]]
    )
    evaluator.prepare_host_graph(
        np.array([6, 6, 6, 6]),
        np.array([0, 3, 3, 3, 3]),
        np.array([1, 2, 3]),
        np.zeros((3, 3), dtype=np.int64),
        positions,
        np.eye(3) * 20.0,
        5.0,
    )
    evaluator.compute(False, False)

    cache = evaluator._coordination_switch_cache_diagnostics()
    rcov_carbon = 1.0000000494517636
    ratio = (2.0 * rcov_carbon) / 3.5
    expected_value = 1.0 / (1.0 + np.exp(-16.0 * (ratio - 1.0)))
    expected_derivative = -16.0 * ratio * expected_value * (1.0 - expected_value) / 3.5
    np.testing.assert_allclose(
        cache["values"], [expected_value, 0.0, 0.0], rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        cache["radial_derivatives"],
        [expected_derivative, 0.0, 0.0],
        rtol=rtol,
        atol=atol,
    )
    assert cache["capacity"] == 3
    assert cache["workspace_bytes"] == 2 * 3 * np.dtype(cache["values"].dtype).itemsize
    precision_bytes = np.dtype(cache["values"].dtype).itemsize
    workspace = evaluator.graph_diagnostics()
    assert workspace["reference_weight_active_atoms"] == 4
    assert workspace["reference_weight_capacity_atoms"] == 4
    assert workspace["reference_weight_workspace_bytes"] == 2 * 4 * 5 * precision_bytes
    assert workspace["coordination_cache_active_edges"] == 3
    assert workspace["coordination_cache_capacity_edges"] == 3
    assert workspace["coordination_cache_workspace_bytes"] == 2 * 3 * precision_bytes
    assert workspace["d3_workspace_bytes"] == 2 * (4 * 5 + 3) * precision_bytes

    smaller_positions = positions[:2].copy()
    smaller_positions[1, 0] = coordination_cutoff
    evaluator.prepare_host_graph(
        np.array([6, 6]),
        np.array([0, 1, 1]),
        np.array([1]),
        np.zeros((1, 3), dtype=np.int64),
        smaller_positions,
        np.eye(3) * 20.0,
        5.0,
    )
    evaluator.compute(False, False)

    reused_cache = evaluator._coordination_switch_cache_diagnostics()
    np.testing.assert_array_equal(reused_cache["values"], np.zeros(1))
    np.testing.assert_array_equal(reused_cache["radial_derivatives"], np.zeros(1))
    assert reused_cache["capacity"] == 3
    assert reused_cache["workspace_bytes"] == cache["workspace_bytes"]
    reused_workspace = evaluator.graph_diagnostics()
    assert reused_workspace["reference_weight_active_atoms"] == 2
    assert reused_workspace["reference_weight_capacity_atoms"] == 4
    assert reused_workspace["coordination_cache_active_edges"] == 1
    assert reused_workspace["coordination_cache_capacity_edges"] == 3
    assert reused_workspace["d3_workspace_bytes"] == workspace["d3_workspace_bytes"]


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 2e-15, 2e-15), ("D3KokkosFloat", 2e-6, 2e-7)],
)
def test_d3_coordination_team_matches_serial_oracle_for_irregular_large_receiver(
    class_name, rtol, atol
):
    node_count = 304
    numbers = np.resize(np.array([6, 8, 7, 1]), node_count)
    positions = np.zeros((node_count, 3), dtype=np.float64)
    angles = np.arange(300, dtype=np.float64) * (2.0 * np.pi / 300.0)
    radii = 1.1 + 0.002 * np.arange(300, dtype=np.float64)
    positions[1:301, 0] = radii * np.cos(angles)
    positions[1:301, 1] = radii * np.sin(angles)
    positions[301:] = [[2.1, 0.1, 0.0], [0.3, 1.7, 0.2], [1.8, 0.2, 0.4]]

    sources = np.concatenate(
        [np.arange(1, 301, dtype=np.int64), np.array([0, 301, 303])]
    )
    offsets = np.full(node_count + 1, 300, dtype=np.int64)
    offsets[0] = 0
    offsets[302] = 301
    offsets[303] = 303
    offsets[304] = 303
    shifts = np.zeros((len(sources), 3), dtype=np.int64)

    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    evaluator.prepare_host_graph(
        numbers,
        offsets,
        sources,
        shifts,
        positions,
        np.eye(3) * 30.0,
        4.0,
    )

    team = evaluator._coordination_accumulation_diagnostics("team")
    serial = evaluator._coordination_accumulation_diagnostics("serial")

    assert team["execution_path"] == "team"
    assert serial["execution_path"] == "serial_oracle"
    assert team["team_launch_count"] == 0
    assert team["serial_fallback_launch_count"] == 0
    assert serial["team_launch_count"] == 0
    assert serial["serial_fallback_launch_count"] == 0
    assert team["coordination_numbers"][1] == 0.0
    assert team["coordination_numbers"][303] == 0.0
    assert team["coordination_numbers"][0] > 256.0
    np.testing.assert_allclose(
        team["coordination_numbers"],
        serial["coordination_numbers"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(team["values"], serial["values"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        team["radial_derivatives"],
        serial["radial_derivatives"],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_coordination_team_handles_empty_graph(class_name):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    evaluator.prepare_host_graph(
        np.empty(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.eye(3) * 10.0,
        4.0,
    )

    diagnostics = evaluator._coordination_accumulation_diagnostics("team")

    np.testing.assert_array_equal(diagnostics["coordination_numbers"], np.empty(0))
    np.testing.assert_array_equal(diagnostics["values"], np.empty(0))
    np.testing.assert_array_equal(diagnostics["radial_derivatives"], np.empty(0))
    assert diagnostics["execution_path"] == "team"
    assert diagnostics["serial_fallback_launch_count"] == 0


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_coordination_test_diagnostics_do_not_record_production_launch(class_name):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    numbers = np.array([6, 8, 7])
    positions = np.array([[0.2, 0.3, 0.4], [1.5, 0.4, 0.7], [2.1, 1.8, 0.9]])
    cell = np.eye(3) * 8.0
    offsets, sources, shifts = _host_graph(positions, cell, [False, False, False], 4.0)
    evaluator.prepare_host_graph(
        numbers, offsets, sources, shifts, positions, cell, 4.0
    )

    before = evaluator._coordination_path_diagnostics()
    evaluator._coordination_accumulation_diagnostics("team")
    evaluator._coordination_accumulation_diagnostics("serial")
    after_test_diagnostics = evaluator._coordination_path_diagnostics()

    assert before == {
        "execution_path": "not_run",
        "team_launch_count": 0,
        "serial_fallback_launch_count": 0,
    }
    assert after_test_diagnostics == before

    evaluator.compute(False, False)
    after_compute = evaluator._coordination_path_diagnostics()
    assert after_compute == {
        "execution_path": "team",
        "team_launch_count": 1,
        "serial_fallback_launch_count": 0,
    }

    evaluator._coordination_accumulation_diagnostics("team")
    evaluator._coordination_accumulation_diagnostics("serial")
    assert evaluator._coordination_path_diagnostics() == after_compute


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 2e-15, 2e-15), ("D3KokkosFloat", 2e-6, 2e-7)],
)
def test_d3_coordination_team_matches_serial_oracle_for_periodic_graph(
    class_name, rtol, atol
):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    numbers = np.array([6, 8, 7, 1])
    positions = np.array(
        [[0.2, 0.3, 0.4], [1.5, 0.4, 0.7], [2.1, 1.8, 0.9], [4.7, 4.5, 4.6]]
    )
    cell = np.array([[5.2, 0.2, 0.1], [0.0, 5.4, 0.3], [0.0, 0.0, 5.6]])
    evaluator.prepare_periodic_graph(numbers, positions, cell, np.linalg.inv(cell), 4.0)

    team = evaluator._coordination_accumulation_diagnostics("team")
    serial = evaluator._coordination_accumulation_diagnostics("serial")

    assert len(team["values"]) > 0
    np.testing.assert_allclose(
        team["coordination_numbers"],
        serial["coordination_numbers"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(team["values"], serial["values"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        team["radial_derivatives"],
        serial["radial_derivatives"],
        rtol=rtol,
        atol=atol,
    )


def _prepare_irregular_pair_diagnostic_graph(evaluator):
    node_count = 304
    numbers = np.resize(np.array([6, 8, 7, 1]), node_count)
    positions = np.zeros((node_count, 3), dtype=np.float64)
    angles = np.arange(300, dtype=np.float64) * (2.0 * np.pi / 300.0)
    radii = 1.1 + 0.002 * np.arange(300, dtype=np.float64)
    positions[1:301, 0] = radii * np.cos(angles)
    positions[1:301, 1] = radii * np.sin(angles)
    positions[301:] = [[2.1, 0.1, 0.0], [0.3, 1.7, 0.2], [1.8, 0.2, 0.4]]
    sources = np.concatenate(
        [np.arange(1, 301, dtype=np.int64), np.array([0, 301, 303])]
    )
    offsets = np.full(node_count + 1, 300, dtype=np.int64)
    offsets[0] = 0
    offsets[302] = 301
    offsets[303] = 303
    offsets[304] = 303
    evaluator.prepare_host_graph(
        numbers,
        offsets,
        sources,
        np.zeros((len(sources), 3), dtype=np.int64),
        positions,
        np.eye(3) * 30.0,
        4.0,
    )


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 3e-14, 3e-14), ("D3KokkosFloat", 2e-5, 2e-6)],
)
@pytest.mark.parametrize("damping", ["zero", "bj", "zerom", "bjm"])
def test_d3_pair_team_matches_serial_direct_accumulation_for_irregular_graph(
    class_name, rtol, atol, damping
):
    evaluator = _evaluator(class_name, damping, "pbe", 4.0, 3.5)
    _prepare_irregular_pair_diagnostic_graph(evaluator)

    team = evaluator._pair_accumulation_diagnostics("team", True, True)
    serial = evaluator._pair_accumulation_diagnostics("serial", True, True)

    assert team["execution_path"] == "team"
    assert serial["execution_path"] == "serial_oracle"
    assert len(team["atomic_energies"]) == 304
    assert team["atomic_energies"][1] == 0.0
    assert team["atomic_energies"][303] == 0.0
    for name in (
        "atomic_energies",
        "energy_coordination_derivatives",
        "forces",
        "virial",
    ):
        np.testing.assert_allclose(team[name], serial[name], rtol=rtol, atol=atol)


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_pair_team_honors_direct_force_and_stress_gates(class_name):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    _prepare_irregular_pair_diagnostic_graph(evaluator)

    energy_only = evaluator._pair_accumulation_diagnostics("team", False, False)
    all_outputs = evaluator._pair_accumulation_diagnostics("team", True, True)

    np.testing.assert_allclose(
        energy_only["atomic_energies"], all_outputs["atomic_energies"]
    )
    np.testing.assert_allclose(
        energy_only["energy_coordination_derivatives"],
        all_outputs["energy_coordination_derivatives"],
    )
    np.testing.assert_array_equal(energy_only["forces"], 0.0)
    np.testing.assert_array_equal(energy_only["virial"], 0.0)
    assert np.any(all_outputs["forces"] != 0.0)
    assert np.any(all_outputs["virial"] != 0.0)


@pytest.mark.parametrize("class_name", ["D3Kokkos", "D3KokkosFloat"])
def test_d3_pair_team_handles_empty_graph(class_name):
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, 3.5)
    evaluator.prepare_host_graph(
        np.empty(0, dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.eye(3) * 10.0,
        4.0,
    )

    diagnostics = evaluator._pair_accumulation_diagnostics("team", True, True)

    assert diagnostics["execution_path"] == "team"
    assert diagnostics["atomic_energies"].shape == (0,)
    assert diagnostics["energy_coordination_derivatives"].shape == (0,)
    assert diagnostics["forces"].shape == (0, 3)
    assert diagnostics["virial"].shape == (0, 6)


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 2e-12, 2e-12), ("D3KokkosFloat", 3e-5, 3e-6)],
)
@pytest.mark.parametrize("damping", ["zero", "bj", "zerom", "bjm"])
def test_d3_adjoint_team_matches_serial_oracle_for_irregular_graph(
    class_name, rtol, atol, damping
):
    evaluator = _evaluator(class_name, damping, "pbe", 4.0, 3.5)
    _prepare_irregular_pair_diagnostic_graph(evaluator)

    team = evaluator._adjoint_accumulation_diagnostics("team")
    serial = evaluator._adjoint_accumulation_diagnostics("serial")

    assert team["execution_path"] == "team"
    assert serial["execution_path"] == "serial_oracle"
    assert len(team["forces"]) == 304
    assert np.any(team["forces"] != 0.0)
    assert np.any(team["virial"] != 0.0)
    np.testing.assert_allclose(team["forces"], serial["forces"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(team["virial"], serial["virial"], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 2e-12, 2e-12), ("D3KokkosFloat", 3e-5, 3e-6)],
)
def test_d3_adjoint_team_matches_serial_at_coordination_cutoff_boundary(
    class_name, rtol, atol
):
    coordination_cutoff = 3.5
    evaluator = _evaluator(class_name, "bj", "pbe", 4.0, coordination_cutoff)
    positions = np.array(
        [[0.0, 0.0, 0.0], [coordination_cutoff, 0.0, 0.0], [3.49, 0.2, 0.0]]
    )
    offsets, sources, shifts = _host_graph(
        positions, np.eye(3) * 20.0, [False] * 3, 4.0
    )
    evaluator.prepare_host_graph(
        np.array([6, 8, 7]),
        offsets,
        sources,
        shifts,
        positions,
        np.eye(3) * 20.0,
        4.0,
    )

    team = evaluator._adjoint_accumulation_diagnostics("team")
    serial = evaluator._adjoint_accumulation_diagnostics("serial")

    np.testing.assert_allclose(team["forces"], serial["forces"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(team["virial"], serial["virial"], rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("class_name", "rtol", "atol"),
    [("D3Kokkos", 2e-12, 2e-12), ("D3KokkosFloat", 3e-5, 3e-6)],
)
def test_d3_adjoint_team_overwrites_grow_then_shrink_workspace(class_name, rtol, atol):
    evaluator = _evaluator(class_name, "bjm", "pbe", 4.0, 3.5)
    _prepare_irregular_pair_diagnostic_graph(evaluator)
    evaluator._adjoint_accumulation_diagnostics("team")

    positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.3, 0.1], [2.2, 0.4, 0.2]])
    offsets, sources, shifts = _host_graph(
        positions, np.eye(3) * 12.0, [False] * 3, 4.0
    )
    evaluator.prepare_host_graph(
        np.array([6, 8, 1]),
        offsets,
        sources,
        shifts,
        positions,
        np.eye(3) * 12.0,
        4.0,
    )

    team = evaluator._adjoint_accumulation_diagnostics("team")
    serial = evaluator._adjoint_accumulation_diagnostics("serial")

    assert team["forces"].shape == (3, 3)
    assert team["virial"].shape == (3, 6)
    np.testing.assert_allclose(team["forces"], serial["forces"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(team["virial"], serial["virial"], rtol=rtol, atol=atol)
