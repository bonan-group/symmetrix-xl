from __future__ import annotations

import hashlib

import numpy as np
import pytest
import symmetrix
from reference.mace_les import (
    load_mace_les_cases,
    ordered_reciprocal_candidate_indices,
    ordered_reciprocal_membership,
    periodic_monopole_oracle_arrays,
)

_SKEW_CASES = (
    pytest.param(4.8, 3, id="about-100-vectors"),
    pytest.param(7.4, 5, id="about-350-vectors"),
    pytest.param(10.8, 8, id="about-1100-vectors"),
)

_REDUCTION_BOUNDARY_CELLS = (
    pytest.param(
        np.array(
            [
                [7.79439363242974, 0.3403213691483189, -0.002014098974358447],
                [0.5071382858445973, 6.216265625973522, 0.1160045871747506],
                [-0.5253579029728546, 0.048082606654542004, 6.11954046754742],
            ]
        ),
        255,
        id="below-team",
    ),
    pytest.param(
        np.array(
            [
                [6.791185821082671, -0.2454818873737028, 0.39190367925993175],
                [-0.43058496388207723, 7.342376015909748, 0.35462397136713975],
                [-0.5146688435887247, 0.04494746506715919, 6.042743850236608],
            ]
        ),
        256,
        id="at-team",
    ),
    pytest.param(
        np.array(
            [
                [7.092463416380813, -0.21002192160434446, -0.1281418739677832],
                [-0.3256836734060288, 7.210564463177526, -0.14550624232655002],
                [-0.28727453216482185, -0.055801745902922306, 5.926127376628741],
            ]
        ),
        257,
        id="above-team",
    ),
)


def _skew_case(scale, atom_count):
    cell = scale * np.array([[1.0, 0.07, 0.02], [0.04, 0.93, 0.08], [0.03, 0.06, 1.04]])
    fractional = np.array(
        [
            [0.07, 0.13, 0.19],
            [0.23, 0.61, 0.37],
            [0.81, 0.29, 0.53],
            [0.46, 0.72, 0.88],
            [0.66, 0.44, 0.11],
            [0.34, 0.91, 0.69],
            [0.92, 0.08, 0.76],
            [0.15, 0.39, 0.95],
        ]
    )[:atom_count]
    charges = np.array([0.41, -0.73, 0.28, 0.19, -0.37, 0.62, -0.21, -0.19])[
        :atom_count
    ]
    return charges, fractional @ cell, cell


def _many_atom_case(atom_count):
    cell = np.array([[4.8, 0.31, 0.09], [0.19, 4.43, 0.38], [0.14, 0.27, 4.99]])
    atom = np.arange(atom_count, dtype=np.float64)
    fractional = np.column_stack(
        (
            np.mod(0.5 + 17.0 * atom, atom_count) / atom_count,
            np.mod(1.5 + 31.0 * atom, atom_count) / atom_count,
            np.mod(2.5 + 47.0 * atom, atom_count) / atom_count,
        )
    )
    charges = np.sin(0.37 * atom) + 0.25 * np.cos(0.19 * atom)
    return charges, fractional @ cell, cell


@pytest.mark.parametrize(
    "class_name", ["_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"]
)
@pytest.mark.parametrize(
    "case",
    [case for case in load_mace_les_cases() if case.boundary == "nonperiodic"],
    ids=lambda case: case.name,
)
def test_nonperiodic_monopole_matches_frozen_reference(class_name, case):
    operator = getattr(symmetrix.load_backend(), class_name)(
        case.sigma, case.dl, case.remove_self_interaction
    )
    result = operator.compute_nonperiodic(case.charges, case.positions)
    tolerance = 2.0e-12 if class_name.endswith("Kokkos") else 2.0e-6

    assert result["energy"] == pytest.approx(case.energy, abs=tolerance)
    np.testing.assert_allclose(
        result["atomic_energies"], case.atomic_energies, atol=tolerance
    )
    np.testing.assert_allclose(result["forces"], case.forces, atol=tolerance)
    np.testing.assert_allclose(
        result["charge_adjoints"], case.charge_adjoints, atol=tolerance
    )
    np.testing.assert_allclose(result["forces"].sum(axis=0), 0.0, atol=tolerance)


@pytest.mark.parametrize(
    "class_name", ["_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"]
)
@pytest.mark.parametrize(
    "case",
    [case for case in load_mace_les_cases() if case.boundary == "periodic"],
    ids=lambda case: case.name,
)
def test_periodic_monopole_matches_frozen_reference(class_name, case):
    operator = getattr(symmetrix.load_backend(), class_name)(
        case.sigma, case.dl, case.remove_self_interaction
    )
    result = operator.compute_periodic(case.charges, case.positions, case.cell)
    tolerance = 6.0e-9 if class_name.endswith("Kokkos") else 2.0e-6

    assert result["energy"] == pytest.approx(case.energy, abs=tolerance)
    np.testing.assert_allclose(
        result["atomic_energies"], case.atomic_energies, atol=tolerance
    )
    np.testing.assert_allclose(result["forces"], case.forces, atol=tolerance)
    np.testing.assert_allclose(
        result["charge_adjoints"], case.charge_adjoints, atol=tolerance
    )
    np.testing.assert_allclose(result["strain_derivative"], case.stress, atol=tolerance)
    np.testing.assert_allclose(result["forces"].sum(axis=0), 0.0, atol=tolerance)


@pytest.mark.parametrize(
    "class_name", ["_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"]
)
@pytest.mark.parametrize(("scale", "atom_count"), _SKEW_CASES)
def test_periodic_monopole_matches_generated_large_sum_oracle(
    class_name, scale, atom_count
):
    charges, positions, cell = _skew_case(scale, atom_count)
    sigma = 0.8
    dl = 1.35
    operator = getattr(symmetrix.load_backend(), class_name)(sigma, dl, True)
    result = operator.compute_periodic(charges, positions, cell)
    expected = periodic_monopole_oracle_arrays(
        positions=positions,
        cell=cell,
        charges=charges,
        sigma=sigma,
        dl=dl,
        remove_self_interaction=True,
    )

    # These property-specific limits freeze the serial implementation before
    # its sums are decomposed into portable Kokkos team reductions.
    tolerances = {
        "energy": (2.0e-11, 2.0e-10),
        "atomic_energies": (2.0e-11, 2.0e-10),
        "forces": (3.0e-11, 3.0e-10),
        "charge_adjoints": (3.0e-11, 3.0e-10),
        "strain_derivative": (2.0e-7, 2.0e-6),
    }
    for name, (rtol, atol) in tolerances.items():
        np.testing.assert_allclose(result[name], expected[name], rtol=rtol, atol=atol)

    expected_count = ordered_reciprocal_membership(cell, dl).shape[0]
    assert operator.diagnostics["reciprocal_vector_count"] == expected_count


@pytest.mark.parametrize(
    "class_name", ["_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"]
)
@pytest.mark.parametrize("atom_count", [17, 257], ids=["below-team", "above-team"])
def test_periodic_monopole_matches_oracle_across_team_sized_atom_counts(
    class_name, atom_count
):
    charges, positions, cell = _many_atom_case(atom_count)
    sigma = 0.8
    dl = 1.35
    operator = getattr(symmetrix.load_backend(), class_name)(sigma, dl, True)
    expected = periodic_monopole_oracle_arrays(
        positions=positions,
        cell=cell,
        charges=charges,
        sigma=sigma,
        dl=dl,
        remove_self_interaction=True,
    )

    first = operator.compute_periodic(charges, positions, cell)
    second = operator.compute_periodic(charges, positions, cell)

    tolerances = {
        "energy": (2.0e-11, 2.0e-10),
        "atomic_energies": (2.0e-11, 2.0e-10),
        "forces": (3.0e-11, 3.0e-10),
        "charge_adjoints": (3.0e-11, 3.0e-10),
        "strain_derivative": (2.0e-7, 2.0e-6),
    }
    for name, (rtol, atol) in tolerances.items():
        np.testing.assert_allclose(first[name], expected[name], rtol=rtol, atol=atol)
        np.testing.assert_allclose(second[name], first[name], rtol=rtol, atol=atol)

    assert operator.diagnostics["table_rebuilds"] == 1
    assert operator.diagnostics["geometry_updates"] == 2


@pytest.mark.parametrize(
    "class_name", ["_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"]
)
@pytest.mark.parametrize("remove_self_interaction", [False, True])
@pytest.mark.parametrize(("cell", "expected_count"), _REDUCTION_BOUNDARY_CELLS)
def test_periodic_outputs_match_oracle_across_reciprocal_team_boundary(
    class_name, remove_self_interaction, cell, expected_count
):
    fractional = np.array([[0.07, 0.13, 0.19], [0.23, 0.61, 0.37], [0.81, 0.29, 0.53]])
    charges = np.array([0.41, -0.73, 0.28])
    positions = fractional @ cell
    sigma = 0.8
    dl = 1.35
    operator = getattr(symmetrix.load_backend(), class_name)(
        sigma, dl, remove_self_interaction
    )
    expected = periodic_monopole_oracle_arrays(
        positions=positions,
        cell=cell,
        charges=charges,
        sigma=sigma,
        dl=dl,
        remove_self_interaction=remove_self_interaction,
    )

    first = operator.compute_periodic(charges, positions, cell)
    repeated = operator.compute_periodic(charges, positions, cell)
    difference_atom = expected_count - 255
    difference_component = expected_count - 255
    displacement = 1.0e-5
    displaced_plus = positions.copy()
    displaced_minus = positions.copy()
    displaced_plus[difference_atom, difference_component] += displacement
    displaced_minus[difference_atom, difference_component] -= displacement
    energy_plus = operator.compute_periodic(charges, displaced_plus, cell)["energy"]
    energy_minus = operator.compute_periodic(charges, displaced_minus, cell)["energy"]
    finite_difference_force = -(energy_plus - energy_minus) / (2.0 * displacement)
    rebuilt_cell = cell.copy()
    rebuilt_cell[0, 1] += 1.0e-3
    rebuilt_positions = fractional @ rebuilt_cell
    rebuilt = operator.compute_periodic(charges, rebuilt_positions, rebuilt_cell)
    rebuilt_expected = periodic_monopole_oracle_arrays(
        positions=rebuilt_positions,
        cell=rebuilt_cell,
        charges=charges,
        sigma=sigma,
        dl=dl,
        remove_self_interaction=remove_self_interaction,
    )

    tolerances = {
        "energy": (2.0e-11, 2.0e-10),
        "atomic_energies": (2.0e-11, 2.0e-10),
        "forces": (3.0e-11, 3.0e-10),
        "charge_adjoints": (3.0e-11, 3.0e-10),
        "strain_derivative": (2.0e-7, 2.0e-6),
    }
    for name, (rtol, atol) in tolerances.items():
        np.testing.assert_allclose(first[name], expected[name], rtol=rtol, atol=atol)
        np.testing.assert_allclose(repeated[name], first[name], rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            rebuilt[name], rebuilt_expected[name], rtol=rtol, atol=atol
        )

    assert first["forces"][difference_atom, difference_component] == pytest.approx(
        finite_difference_force, rel=2.0e-6, abs=2.0e-7
    )

    assert first["energy"] == pytest.approx(first["atomic_energies"].sum())
    np.testing.assert_allclose(first["forces"].sum(axis=0), 0.0, atol=3.0e-10)
    np.testing.assert_allclose(
        first["strain_derivative"], first["strain_derivative"].T, atol=2.0e-7
    )
    assert expected_count == ordered_reciprocal_membership(cell, dl).shape[0]
    assert operator.diagnostics["reciprocal_vector_count"] == expected_count
    assert operator.diagnostics["table_rebuilds"] == 2
    assert operator.diagnostics["geometry_updates"] == 5


@pytest.mark.parametrize(
    ("length", "expected_indices"),
    [(0.99, []), (1.01, [22])],
    ids=["above-k-cutoff", "below-k-cutoff"],
)
def test_cutoff_separated_membership_matches_precision_wrappers(
    length, expected_indices
):
    cell = np.diag([length, 0.71, 0.77])
    charges = np.array([0.4, -0.4, 0.2])
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.2, 0.3, 0.1]])
    oracle_indices = ordered_reciprocal_candidate_indices(cell, 1.0).tolist()
    assert oracle_indices == expected_indices
    memberships = []

    for class_name in ("_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"):
        operator = getattr(symmetrix.load_backend(), class_name)(0.8, 1.0, True)
        operator.compute_periodic(charges, positions, cell)
        first = operator._active_reciprocal_indices_for_testing()
        operator.compute_periodic(charges, positions + 0.01, cell)
        second = operator._active_reciprocal_indices_for_testing()
        assert first == second
        assert first == expected_indices
        memberships.append(first)

    assert memberships[0] == memberships[1]


@pytest.mark.parametrize(
    ("dl", "cell"),
    [
        (np.nextafter(0.0, 1.0), np.eye(3) * 6.0),
        (1.0, np.diag([1.0e300, 1.0, 1.0])),
    ],
    ids=["extreme-dl", "extreme-cell"],
)
def test_periodic_rejects_extreme_finite_reciprocal_bounds(dl, cell):
    operator = symmetrix.load_backend()._LESMonopoleKokkos(0.8, dl, True)

    with pytest.raises(
        ValueError,
        match=r"LES reciprocal bound.*increase dl or reduce the periodic cell",
    ):
        operator.compute_periodic(
            np.array([0.4, -0.4]),
            np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4]]),
            cell,
        )
    assert operator.diagnostics["table_rebuilds"] == 0


@pytest.mark.parametrize(
    "class_name", ["_LESMonopoleKokkos", "_LESMonopoleKokkosFloat"]
)
def test_periodic_analytic_strain_derivative_matches_fixed_reciprocal_set(class_name):
    operator = getattr(symmetrix.load_backend(), class_name)(0.8, 1.35, True)
    charges = np.array([0.4, -0.7, 0.2])
    positions = np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4], [0.7, 1.6, 1.1]])
    cell = np.array([[5.71, 0.17, 0.09], [0.11, 5.33, 0.21], [0.07, 0.19, 5.49]])
    result = operator.compute_periodic(charges, positions, cell)
    # This step is small enough that a reciprocal vector close to the cutoff
    # remains on the same side.  The finite difference is independent of the
    # native analytic derivative while differentiating the same discrete sum.
    step = 2.0e-7
    numeric = np.empty((3, 3))
    for row in range(3):
        for column in range(3):
            energies = []
            for sign in (1.0, -1.0):
                deformation = np.eye(3)
                deformation[row, column] += sign * step
                energies.append(
                    operator.compute_periodic(
                        charges, positions @ deformation, cell @ deformation
                    )["energy"]
                )
            numeric[row, column] = (energies[0] - energies[1]) / (2.0 * step)
    tolerance = 3.0e-6 if class_name.endswith("Float") else 2.0e-7
    np.testing.assert_allclose(
        result["strain_derivative"], numeric, rtol=tolerance, atol=tolerance
    )


def test_periodic_workspace_reuses_backend_capacity_and_reports_updates():
    operator = symmetrix.load_backend()._LESMonopoleKokkos(0.8, 1.35, True)
    charges = np.array([0.4, -0.7, 0.2])
    positions = np.array([[0.2, 0.3, 0.5], [1.3, 0.8, 0.4], [0.7, 1.6, 1.1]])
    cell = np.diag([5.7, 5.3, 5.5])

    operator.compute_periodic(charges, positions, cell)
    first = operator.diagnostics
    operator.compute_periodic(charges, positions + 0.01, cell)
    reused = operator.diagnostics
    operator.compute_periodic(charges, positions, cell * 1.01)
    changed = operator.diagnostics

    assert first["reciprocal_vector_count"] > 0
    assert first["table_rebuilds"] == 1
    assert first["index_pack_count"] == 1
    assert reused["table_rebuilds"] == 1
    assert reused["index_pack_count"] == 1
    assert changed["table_rebuilds"] == 2
    assert changed["index_pack_count"] == 2
    assert changed["geometry_updates"] == 3
    assert changed["active_bytes"] > 0
    assert changed["capacity_bytes"] >= changed["active_bytes"]


def _candidate_count(cell, dl):
    counts = [max(1, int(np.linalg.norm(row) / dl)) for row in cell]
    return int(np.prod([2 * count + 1 for count in counts]))


def _logical_extent_bytes(atom_count, candidate_count, active_count):
    return (
        (1 + 5 * atom_count + 9 + 10 + 7 * candidate_count) * 8
        + candidate_count
        + active_count * np.dtype(np.uintp).itemsize
    )


def test_diagnostics_count_persistent_energy_scalar_only_after_allocation():
    operator = symmetrix.load_backend()._LESMonopoleKokkos(0.8, 1.35, True)
    empty = operator.diagnostics
    assert empty["logical_extent_bytes"] == 0
    assert empty["allocated_capacity_bytes"] == 0

    charges = np.array([0.4, -0.4])
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    operator.compute_nonperiodic(charges, positions)
    allocated = operator.diagnostics
    expected = (1 + 5 * charges.size + 9) * np.dtype(np.float64).itemsize
    assert allocated["logical_extent_bytes"] == expected
    assert allocated["allocated_capacity_bytes"] == expected


def test_periodic_diagnostics_report_logical_work_and_retained_capacity():
    operator = symmetrix.load_backend()._LESMonopoleKokkos(0.8, 1.35, True)
    small_charges, small_positions, small_cell = _skew_case(4.8, 3)
    large_charges, large_positions, large_cell = _skew_case(10.8, 8)

    operator.compute_periodic(small_charges, small_positions, small_cell)
    small = operator.diagnostics
    small_candidates = _candidate_count(small_cell, 1.35)
    assert small["reciprocal_candidate_count"] == small_candidates
    assert small["logical_extent_bytes"] == _logical_extent_bytes(
        small_charges.size, small_candidates, small["reciprocal_vector_count"]
    )
    assert small["allocated_capacity_bytes"] == (
        _logical_extent_bytes(small_charges.size, small_candidates, 0)
        + small_candidates * np.dtype(np.uintp).itemsize
    )
    assert (
        small["active_index_capacity_bytes"]
        == small_candidates * np.dtype(np.uintp).itemsize
    )
    assert small["index_pack_count"] == 1
    assert small["table_rebuilds"] == 1

    operator.compute_periodic(large_charges, large_positions, large_cell)
    grown = operator.diagnostics
    grown_candidates = _candidate_count(large_cell, 1.35)
    assert grown["reciprocal_candidate_count"] == grown_candidates
    assert grown["logical_extent_bytes"] == _logical_extent_bytes(
        large_charges.size, grown_candidates, grown["reciprocal_vector_count"]
    )
    assert (
        grown["active_index_capacity_bytes"]
        == grown_candidates * np.dtype(np.uintp).itemsize
    )
    assert grown["index_pack_count"] == 2
    assert grown["table_rebuilds"] == 2

    operator.compute_periodic(small_charges, small_positions, small_cell)
    shrunk = operator.diagnostics
    assert shrunk["reciprocal_candidate_count"] == small_candidates
    assert shrunk["logical_extent_bytes"] == _logical_extent_bytes(
        small_charges.size, small_candidates, shrunk["reciprocal_vector_count"]
    )
    assert shrunk["allocated_capacity_bytes"] == grown["allocated_capacity_bytes"]
    assert shrunk["allocated_capacity_bytes"] > shrunk["logical_extent_bytes"]
    assert (
        shrunk["active_index_capacity_bytes"]
        == grown_candidates * np.dtype(np.uintp).itemsize
    )
    assert shrunk["index_pack_count"] == 3
    assert shrunk["table_rebuilds"] == 3

    operator.compute_periodic(small_charges, small_positions + 0.01, small_cell)
    reused = operator.diagnostics
    assert reused["table_rebuilds"] == 3
    assert reused["index_pack_count"] == 3
    assert reused["geometry_updates"] == 4
    assert reused["logical_extent_bytes"] == shrunk["logical_extent_bytes"]
    assert reused["allocated_capacity_bytes"] == shrunk["allocated_capacity_bytes"]


@pytest.mark.parametrize("remove_self_interaction", [False, True])
def test_periodic_packs_an_empty_active_set_in_candidate_order(
    remove_self_interaction,
):
    sigma = 0.8
    operator = symmetrix.load_backend()._LESMonopoleKokkos(
        sigma, 1.0, remove_self_interaction
    )
    charges = np.array([0.4, -0.4])
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    cell = np.diag([0.99, 0.71, 0.77])

    result = operator.compute_periodic(charges, positions, cell)
    expected = periodic_monopole_oracle_arrays(
        positions=positions,
        cell=cell,
        charges=charges,
        sigma=sigma,
        dl=1.0,
        remove_self_interaction=remove_self_interaction,
    )
    diagnostics = operator.diagnostics

    assert operator._active_reciprocal_indices_for_testing() == []
    assert diagnostics["reciprocal_vector_count"] == 0
    assert diagnostics["index_pack_count"] == diagnostics["table_rebuilds"] == 1
    assert diagnostics["active_index_capacity_bytes"] == (
        diagnostics["reciprocal_candidate_count"] * np.dtype(np.uintp).itemsize
    )
    assert np.isfinite(result["energy"])
    for name in (
        "energy",
        "atomic_energies",
        "forces",
        "charge_adjoints",
        "strain_derivative",
    ):
        np.testing.assert_allclose(
            result[name], expected[name], rtol=2.0e-11, atol=2.0e-10
        )


def test_test_only_membership_snapshot_preserves_candidate_enumeration_order():
    operator = symmetrix.load_backend()._LESMonopoleKokkos(0.8, 1.35, True)
    charges, positions, cell = _skew_case(7.4, 5)
    operator.compute_periodic(charges, positions, cell)

    active_candidates = operator._active_reciprocal_indices_for_testing()
    expected = ordered_reciprocal_candidate_indices(cell, 1.35).tolist()
    digest = hashlib.sha256(
        np.asarray(active_candidates, dtype="<u8").tobytes()
    ).hexdigest()

    assert active_candidates == expected
    assert digest == "f5fb19aadbc6a19937955c0f08e5745b2229d9a3f8c91daa03112019bac175a3"
    assert len(active_candidates) == operator.diagnostics["reciprocal_vector_count"]
    assert active_candidates[-1] < operator.diagnostics["reciprocal_candidate_count"]
