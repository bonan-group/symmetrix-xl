"""OAM stress sign, geometry, precision, and prepared-generation contracts."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

import oam_host_oracle

pytestmark = pytest.mark.oam_runtime


@pytest.fixture(scope="module", params=("float32", "float64"))
def stress_runtime(request, tmp_path_factory):
    from symmetrix import NequIPCalculator, load_backend
    from test_nequip_oam_lane_layout import _build_contract

    native = load_backend()
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("OAM Kokkos evaluator is unavailable")
    if not native._kokkos_is_initialized():
        native._init_kokkos()
    root = tmp_path_factory.mktemp(f"oam-stress-{request.param}")
    contract = _build_contract()
    paths = []
    for zbl in (False, True):
        contract["zbl"]["half_energy_prefactor"] = 0.5 if zbl else 1e-30
        path = root / f"angular-zbl-{zbl}.json"
        path.write_text(json.dumps(contract))
        paths.append(path)
    return NequIPCalculator, request.param, root, paths


def _periodic_atoms():
    from ase import Atoms

    # A skew cell and a neighbor across its boundary distinguish the virial
    # from a positions-only force contraction; perturbations expose shears.
    return Atoms(
        numbers=[1, 2, 3],
        scaled_positions=[[0.03, 0.07, 0.11], [0.94, 0.16, 0.18], [0.22, 0.31, 0.37]],
        cell=[[6.2, 0.3, -0.2], [0.1, 6.7, 0.4], [-0.3, 0.2, 7.1]],
        pbc=True,
    )


def _calculator(runtime, monkeypatch, path=None):
    cls, precision, root, paths = runtime
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(root / "jit-cache"))
    return cls(path or paths[1], dtype=precision)


@pytest.mark.parametrize("zbl", (False, True))
def test_periodic_stress_matches_strain_after_cached_energy_and_forces(
    stress_runtime, monkeypatch, zbl
):
    calculator = _calculator(stress_runtime, monkeypatch, stress_runtime[3][int(zbl)])
    v1_fixture = Path(__file__).parent / "data" / "nequip" / "tiny_nequip_v1.json"
    with pytest.raises(
        ValueError, match="ordinary NequIP v1 contracts are no longer supported"
    ):
        stress_runtime[0](v1_fixture, dtype="float64")
    assert "stress" in calculator.implemented_properties
    atoms = _periodic_atoms()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces().copy()
    stress = atoms.get_stress()
    np.testing.assert_array_equal(atoms.get_forces(), forces)
    assert atoms.get_potential_energy() == energy
    np.testing.assert_array_equal(atoms.get_stress(), stress)

    # Symmetric engineering shear has half the off-diagonal strain, so its
    # energy derivative/volume is exactly ASE's yz, xz, xy Voigt component.
    step = 2e-3 if stress_runtime[1] == "float32" else 1e-5
    numerical = []
    cell = atoms.cell.array.copy()
    volume = atoms.get_volume()
    for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = 1.0 if i == j else 0.5
        energies = []
        for sign in (1, -1):
            displaced = atoms.copy()
            displaced.set_cell(
                cell @ (np.eye(3) + sign * step * strain), scale_atoms=True
            )
            displaced.calc = calculator
            energies.append(displaced.get_potential_energy())
        numerical.append((energies[0] - energies[1]) / (2 * step * volume))
    rtol, atol = (2e-3, 5e-7) if stress_runtime[1] == "float32" else (2e-5, 2e-9)
    np.testing.assert_allclose(stress, numerical, rtol=rtol, atol=atol)

    # After strain evaluations, restoring the original geometry must not retain
    # the stress from the last displaced cell or a stale prepared generation.
    atoms.calc = calculator
    np.testing.assert_allclose(atoms.get_stress(), stress, rtol=rtol, atol=atol)
    moved = atoms.copy()
    moved.positions[1] += [0.03, -0.02, 0.01]
    moved.calc = calculator
    moved.get_potential_energy()
    cached = moved.get_stress()
    independent = _calculator(stress_runtime, monkeypatch, stress_runtime[3][int(zbl)])
    moved.calc = independent
    np.testing.assert_allclose(moved.get_stress(), cached, rtol=rtol, atol=atol)


def _prepare_periodic(evaluator, atoms):
    from ase.neighborlist import neighbor_list

    receivers, sources, shifts = neighbor_list("ijS", atoms, evaluator.r_cut)
    order = np.argsort(receivers, kind="stable")
    receivers = np.asarray(receivers[order], dtype=np.int32)
    sources = np.asarray(sources[order], dtype=np.int32)
    shifts = shifts[order]
    types = {
        int(number): index for index, number in enumerate(evaluator.atomic_numbers)
    }
    node_types = np.asarray([types[number] for number in atoms.numbers], dtype=np.int32)
    counts = np.bincount(receivers, minlength=len(atoms)).astype(np.int32)
    cell = atoms.cell.array
    xyz = atoms.positions[sources] + shifts @ cell - atoms.positions[receivers]
    generation = evaluator.prepare_graph(
        len(atoms), node_types, counts, sources, node_types[sources]
    )
    evaluator.prepare_geometry(
        generation,
        atoms.positions.ravel(),
        xyz.ravel(),
        cell.ravel(),
        np.linalg.inv(cell).ravel(),
        np.ones(3, dtype=np.int32),
    )
    return generation, xyz


@pytest.fixture(scope="module")
def portable_stress_contract(tmp_path_factory):
    """The angular contract with a real ZBL term, for the no-artifact path."""
    from test_nequip_oam_lane_layout import _build_contract

    contract = _build_contract()
    contract["zbl"]["half_energy_prefactor"] = 0.5
    path = tmp_path_factory.mktemp("oam-portable-stress") / "angular-zbl.json"
    path.write_text(json.dumps(contract))
    return path


def test_portable_host_path_stress_matches_the_finite_strain_oracle(
    portable_stress_contract,
):
    """The compiler-free host path is admitted for stress and reduces it.

    The reference routing declares no stress reduction, so the oracle is the
    finite-strain derivative of this evaluator's own energy, which is the gate
    the plan names for OAM stress.
    """
    from symmetrix import load_backend

    native = load_backend()
    if not hasattr(native, "NequIPOAMKokkos"):
        pytest.skip("OAM Kokkos evaluator is unavailable")
    if not native._kokkos_is_initialized():
        native._init_kokkos()

    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the artifact-free portable OAM host path",
    )
    # No artifact at all: the portable host path is the qualified stress
    # implementation and the whole program is admitted.
    evaluator = native.NequIPOAMKokkos(str(portable_stress_contract))
    evaluator.set_streamed_edges("reference")
    capabilities = evaluator.family_capabilities
    assert capabilities["stress"] is True
    assert capabilities["portable"] is True
    report = evaluator.family_binding_report
    bindings = {binding["capability_key"]: binding for binding in report["bindings"]}
    assert bindings["reduce_stress"]["qualified"] is True
    assert bindings["reduce_stress"]["portable_covered"] is True
    assert bindings["reduce_stress"]["artifact_required"] is False
    assert report["admitted"] is True
    assert report["portable_node_count"] == report["node_count"]

    atoms = _periodic_atoms()
    step = 1e-5
    generation, xyz = _prepare_periodic(evaluator, atoms)
    evaluator.compute_prepared(generation, xyz.ravel(), np.linalg.norm(xyz, axis=1))
    evaluator.compute_stress()
    stress = np.asarray(evaluator.stress_tensor, dtype=float).reshape(3, 3)
    assert "implementation=portable_oam_kokkos_generic" in (
        evaluator.last_execution_diagnostic
    )
    assert np.abs(stress).max() > 1e-6
    np.testing.assert_array_equal(stress, stress.T)

    cell = atoms.cell.array.copy()
    volume = atoms.get_volume()
    numerical = []
    for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = 1.0 if i == j else 0.5
        energies = []
        for sign in (1, -1):
            displaced = atoms.copy()
            displaced.set_cell(
                cell @ (np.eye(3) + sign * step * strain), scale_atoms=True
            )
            strain_generation, strain_xyz = _prepare_periodic(evaluator, displaced)
            evaluator.compute_prepared(
                strain_generation,
                strain_xyz.ravel(),
                np.linalg.norm(strain_xyz, axis=1),
            )
            energies.append(float(np.sum(evaluator.node_energies)))
        numerical.append((energies[0] - energies[1]) / (2 * step * volume))
    # The oracle is indexed in ASE's Voigt order; the evaluator exports the full
    # symmetric tensor.
    voigt = np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[1, 2],
            stress[0, 2],
            stress[0, 1],
        ]
    )
    np.testing.assert_allclose(voigt, numerical, rtol=2e-5, atol=2e-9)


def test_stress_and_force_reductions_are_order_independent(stress_runtime, monkeypatch):
    evaluator = _calculator(stress_runtime, monkeypatch).evaluator
    rtol, atol = (1e-5, 1e-6) if stress_runtime[1] == "float32" else (1e-12, 1e-12)
    atoms = _periodic_atoms()
    generation, xyz = _prepare_periodic(evaluator, atoms)
    evaluator.compute_prepared(generation, xyz.ravel(), np.linalg.norm(xyz, axis=1))
    volume = atoms.get_volume()
    evaluator.reduce_stress(volume, generation)
    stress = np.asarray(evaluator.stress_tensor).reshape(3, 3).copy()
    evaluator.reduce_forces(generation)
    forces = np.asarray(evaluator.atom_forces).copy()
    evaluator.reduce_forces(generation)
    np.testing.assert_allclose(evaluator.atom_forces, forces, rtol=rtol, atol=atol)
    evaluator.reduce_stress(volume, generation)
    np.testing.assert_allclose(
        np.asarray(evaluator.stress_tensor).reshape(3, 3), stress, rtol=rtol, atol=atol
    )
    evaluator.compute_stress()
    np.testing.assert_allclose(
        np.asarray(evaluator.stress_tensor).reshape(3, 3), stress, rtol=rtol, atol=atol
    )
    evaluator.reduce_stress(2 * volume, generation)
    np.testing.assert_allclose(
        np.asarray(evaluator.stress_tensor).reshape(3, 3),
        stress / 2,
        rtol=rtol,
        atol=atol,
    )
    evaluator.reduce_forces(generation)
    np.testing.assert_allclose(evaluator.atom_forces, forces, rtol=rtol, atol=atol)

    # Re-running reverse computation and reducing forces first must preserve
    # positive geometry adjoints, not toggle a stored derivative's sign.
    evaluator.compute_prepared(generation, xyz.ravel(), np.linalg.norm(xyz, axis=1))
    evaluator.reduce_forces(generation)
    evaluator.reduce_stress(volume, generation)
    np.testing.assert_allclose(
        np.asarray(evaluator.stress_tensor).reshape(3, 3), stress, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(evaluator.atom_forces, forces, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(stress, stress.T)


def test_stress_requires_completed_current_generation_and_positive_volume(
    stress_runtime, monkeypatch
):
    evaluator = _calculator(stress_runtime, monkeypatch).evaluator
    with pytest.raises(ValueError):
        evaluator.compute_stress()
    atoms = _periodic_atoms()
    generation, xyz = _prepare_periodic(evaluator, atoms)
    with pytest.raises(ValueError):
        evaluator.reduce_stress(atoms.get_volume(), generation)
    evaluator.compute_prepared(generation, xyz.ravel(), np.linalg.norm(xyz, axis=1))
    for volume in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            evaluator.reduce_stress(volume, generation)
    isolated = atoms[:1]
    isolated.set_cell([12.0] * 3)
    current, empty_xyz = _prepare_periodic(evaluator, isolated)
    with pytest.raises(ValueError):
        evaluator.reduce_stress(atoms.get_volume(), generation)
    with pytest.raises(ValueError):
        evaluator.compute_stress()
    evaluator.compute_prepared(
        current, empty_xyz.ravel(), np.linalg.norm(empty_xyz, axis=1)
    )
    evaluator.reduce_stress(isolated.get_volume(), current)
    np.testing.assert_array_equal(evaluator.stress_tensor, np.zeros(9))
    evaluator.reduce_forces(current)
    np.testing.assert_array_equal(evaluator.atom_forces, np.zeros(3))
    evaluator.compute_stress()
    np.testing.assert_array_equal(evaluator.stress_tensor, np.zeros(9))


def test_stress_without_prepared_cell_is_rejected(stress_runtime, monkeypatch):
    evaluator = _calculator(stress_runtime, monkeypatch).evaluator
    types = np.asarray([0], dtype=np.int32)
    empty = np.empty(0, dtype=np.int32)
    generation = evaluator.prepare_graph(
        1, types, np.zeros(1, dtype=np.int32), empty, empty
    )
    evaluator.prepare_geometry(
        generation, np.zeros(3), np.empty(0), np.empty(0), np.empty(0), empty
    )
    evaluator.compute_prepared(generation, np.empty(0), np.empty(0))
    with pytest.raises(ValueError):
        evaluator.compute_stress()
    evaluator.reduce_stress(1.0, generation)
    np.testing.assert_array_equal(evaluator.stress_tensor, np.zeros(9))
