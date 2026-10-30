"""Focused coverage for the one host neighbor-graph builder every frontend shares.

The builder is the single source of the host graph a calculator hands to a
native evaluator, so this module pins the two properties that make the
promotion safe:

* **Byte identity.**  For identical inputs the shared builder returns exactly
  the arrays the frontend built before the extraction: the matscipy/ASE edge
  list, the 32-bit cast, the receiver-major order, the node degrees, the
  neighbor types, and the edge vectors and distances measured from that order.
* **One routing.**  The OAM frontend builds and reuses its graph through the
  same builder, so the two families cannot drift apart.

No native backend and no compiler are needed: the graph is pure Python, and the
OAM half drives `_build_graph` and `calculate` through the declared reference
surface the test supplies.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import symmetrix.neighbor_graph as neighbor_graph_module
from ase.atoms import Atoms

try:
    from matscipy.neighbours import neighbour_list as provider_neighbor_list
except ImportError:  # pragma: no cover - exercised by the ASE fallback
    from ase.neighborlist import neighbor_list as provider_neighbor_list

FIXTURE_DIR = Path(__file__).parent / "data"
OAM_FIXTURE = FIXTURE_DIR / "nequip" / "oam_s_v2_minimal.json"
PROJECT_METADATA = Path(__file__).parents[1] / "pyproject.toml"


def _periodic_atoms():
    """A periodic mixed-species cluster with edges inside and across the cell."""

    return Atoms(
        "HOHC",
        positions=[
            [0.0, 0.0, 0.0],
            [0.98, 0.0, 0.0],
            [0.4, 1.6, 0.2],
            [3.6, 3.4, 3.5],
        ],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def _oam_atoms():
    return Atoms(
        numbers=[1, 8],
        positions=[[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )


def _molecule_atoms():
    """A non-periodic molecule: ASE leaves the cell at zero, so it is singular."""

    return Atoms(
        "HOH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.98, 0.0, 0.0],
            [0.4, 1.6, 0.2],
        ],
    )


def test_fast_neighbor_provider_is_an_explicit_optional_dependency():
    """Users can request the accelerator without making it a core dependency."""

    metadata = tomllib.loads(PROJECT_METADATA.read_text(encoding="utf-8"))
    requirements = metadata["project"]["optional-dependencies"]["neighbors"]

    assert requirements == ["matscipy>=1.3.1"]


def _legacy_host_graph(atoms, cutoff, node_types, provider=None):
    """Rebuild the graph exactly as the pre-extraction frontend did."""

    provider = provider_neighbor_list if provider is None else provider
    receivers, sources, shifts = provider("ijS", atoms, cutoff)
    receivers = np.asarray(receivers, dtype=np.int32)
    sources = np.asarray(sources, dtype=np.int32)
    shifts = np.asarray(shifts, dtype=np.int32).reshape((-1, 3))
    if len(receivers):
        order = np.argsort(receivers, kind="stable")
        receivers, sources, shifts = receivers[order], sources[order], shifts[order]
    cell = np.asarray(atoms.cell.array, dtype=float).reshape((3, 3))
    positions = np.asarray(atoms.get_positions(), dtype=float)
    xyz = positions[sources] - positions[receivers] + shifts @ cell
    distances = np.linalg.norm(xyz, axis=1)
    num_neigh = np.asarray(np.bincount(receivers, minlength=len(atoms)), dtype=np.int32)
    neigh_types = np.ascontiguousarray(node_types[sources])
    return {
        "receivers": receivers,
        "sources": sources,
        "shifts": shifts,
        "num_neigh": num_neigh,
        "neigh_types": neigh_types,
        "xyz": xyz,
        "distances": distances,
    }


def _edge_set(graph):
    return {
        (int(receiver), int(source), tuple(int(value) for value in shift))
        for receiver, source, shift in zip(graph.receivers, graph.sources, graph.shifts)
    }


def test_shared_builder_is_byte_identical_to_the_promoted_frontend_sequence():
    atoms = _periodic_atoms()
    node_types = np.asarray([0, 1, 2, 3], dtype=np.int32)
    cutoff = 2.6

    graph = neighbor_graph_module.build_host_neighbor_graph(atoms, cutoff)
    expected = _legacy_host_graph(atoms, cutoff, node_types)

    assert graph.num_edges == len(expected["receivers"])
    assert graph.num_edges > 0
    for name, actual in (
        ("receivers", graph.receivers),
        ("sources", graph.sources),
        ("shifts", graph.shifts),
        ("num_neigh", graph.num_neigh),
        ("neigh_types", graph.neighbor_types(node_types)),
        ("xyz", graph.edge_vectors(atoms.positions, atoms.cell)),
        ("distances", graph.edge_distances(atoms.positions, atoms.cell)),
    ):
        assert actual.dtype == expected[name].dtype, name
        assert actual.tobytes() == expected[name].tobytes(), name


def test_shared_builder_is_deterministic_receiver_major_and_contiguous():
    atoms = _periodic_atoms()
    first = neighbor_graph_module.build_host_neighbor_graph(atoms, 2.6)
    second = neighbor_graph_module.build_host_neighbor_graph(atoms, 2.6)

    for name in ("receivers", "sources", "shifts", "num_neigh"):
        actual = getattr(first, name)
        assert actual.tobytes() == getattr(second, name).tobytes(), name
        assert actual.flags.c_contiguous, name
    assert first.receivers.dtype == first.sources.dtype == np.int32
    assert first.shifts.dtype == np.int32
    assert first.num_neigh.dtype == np.int32
    assert np.all(first.receivers[1:] >= first.receivers[:-1])
    assert int(first.num_neigh.sum()) == first.num_edges


def test_shared_builder_canonical_debug_order_keeps_the_edge_set(monkeypatch):
    atoms = _periodic_atoms()
    default = neighbor_graph_module.build_host_neighbor_graph(atoms, 2.6)
    monkeypatch.setenv("SYMMETRIX_DEBUG_NEIGHBOR_EDGE_ORDER", "canonical")
    canonical = neighbor_graph_module.build_host_neighbor_graph(atoms, 2.6)

    assert _edge_set(default) == _edge_set(canonical)
    assert canonical.num_edges == default.num_edges


def test_shared_builder_validates_cardinality_before_it_narrows(monkeypatch):
    class OverflowingEdges:
        def __len__(self):
            return 2**31 + 1

    monkeypatch.setattr(
        neighbor_graph_module,
        "neighbor_list",
        lambda *_args: (OverflowingEdges(), [], []),
    )
    with pytest.raises(ValueError, match="directed-edge count.*32-bit topology"):
        neighbor_graph_module.build_host_neighbor_graph(_periodic_atoms(), 2.6)


def test_shared_builder_builds_a_non_periodic_molecule_without_the_fast_path(
    monkeypatch,
):
    """A molecule must never reach the fast path, which inverts the cell.

    ASE leaves a non-periodic ``Atoms`` with a zero cell, and matscipy's
    ``neighbour_list`` inverts that cell unconditionally, so it raises
    ``LinAlgError`` where ASE's implementation builds the graph directly.  The
    builder routes such a cell to ASE before it consults a provider, so
    substituting the preferred provider with a failure proves the routing, and
    every array must still match the ASE sequence byte for byte.
    """

    try:
        from matscipy.neighbours import neighbour_list as fast_neighbor_list
    except ImportError:
        pytest.skip("matscipy is not installed, so ASE is already the provider")
    from ase.neighborlist import neighbor_list as ase_neighbor_list

    assert (
        neighbor_graph_module.neighbor_list is fast_neighbor_list
    ), "the matscipy fast path is expected to be the preferred provider here"

    def _fast_path_must_not_run(*_args):
        raise AssertionError("the fast path was asked for a non-periodic cell")

    monkeypatch.setattr(neighbor_graph_module, "neighbor_list", _fast_path_must_not_run)

    atoms = _molecule_atoms()
    assert not np.any(atoms.pbc)
    node_types = np.asarray([0, 1, 2], dtype=np.int32)
    cutoff = 2.6

    graph = neighbor_graph_module.build_host_neighbor_graph(atoms, cutoff)
    expected = _legacy_host_graph(atoms, cutoff, node_types, provider=ase_neighbor_list)

    assert graph.num_edges > 0
    assert graph.num_edges == len(expected["receivers"])
    for name, actual in (
        ("receivers", graph.receivers),
        ("sources", graph.sources),
        ("shifts", graph.shifts),
        ("num_neigh", graph.num_neigh),
        ("neigh_types", graph.neighbor_types(node_types)),
        ("xyz", graph.edge_vectors(atoms.positions, atoms.cell)),
        ("distances", graph.edge_distances(atoms.positions, atoms.cell)),
    ):
        assert actual.dtype == expected[name].dtype, name
        assert actual.tobytes() == expected[name].tobytes(), name


def test_shared_builder_falls_back_when_the_fast_path_rejects_a_periodic_cell(
    monkeypatch,
):
    """A periodic cell the fast path cannot invert still builds through ASE."""

    from ase.neighborlist import neighbor_list as ase_neighbor_list

    def _reject_the_cell(*_args):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(neighbor_graph_module, "neighbor_list", _reject_the_cell)

    atoms = _periodic_atoms()
    node_types = np.asarray([0, 1, 2, 3], dtype=np.int32)
    cutoff = 2.6

    graph = neighbor_graph_module.build_host_neighbor_graph(atoms, cutoff)
    expected = _legacy_host_graph(atoms, cutoff, node_types, provider=ase_neighbor_list)

    assert graph.num_edges > 0
    assert graph.num_edges == len(expected["receivers"])
    for name, actual in (
        ("receivers", graph.receivers),
        ("sources", graph.sources),
        ("shifts", graph.shifts),
        ("num_neigh", graph.num_neigh),
        ("neigh_types", graph.neighbor_types(node_types)),
    ):
        assert actual.tobytes() == expected[name].tobytes(), name


def test_shared_builder_uses_the_fast_path_for_a_vacuum_padded_slab(monkeypatch):
    """A slab keeps the fast path: one missing axis is not a singular cell.

    matscipy inverts the cell and honours ``pbc`` axis by axis, so a
    vacuum-padded slab is invertible and its edge set is the one ASE builds.
    Requiring every axis to be periodic instead routed a 7,618-atom slab's
    17 Angstrom graph through ``ase.neighborlist`` — 43.8 s for the 5,505,780
    edges the fast path builds in 0.34 s — so the routing predicate tests cell
    invertibility, not ``pbc``.
    """

    try:
        from matscipy.neighbours import neighbour_list as fast_neighbor_list
    except ImportError:
        pytest.skip("matscipy is not installed, so ASE is already the provider")
    from ase.neighborlist import neighbor_list as ase_neighbor_list

    calls = []

    def _spy(*args):
        calls.append(args)
        return fast_neighbor_list(*args)

    monkeypatch.setattr(neighbor_graph_module, "neighbor_list", _spy)

    atoms = Atoms(
        "HOHC",
        positions=[
            [0.0, 0.0, 12.0],
            [0.98, 0.0, 12.2],
            [0.4, 1.6, 11.6],
            [1.9, 1.9, 12.4],
        ],
        cell=[4.0, 4.0, 24.0],
        pbc=(True, True, False),
    )
    assert not bool(atoms.pbc[2])
    assert np.linalg.det(atoms.cell.array) > 0.0
    node_types = np.asarray([0, 1, 2, 3], dtype=np.int32)
    cutoff = 2.6

    graph = neighbor_graph_module.build_host_neighbor_graph(atoms, cutoff)
    assert calls, "an invertible, partially periodic cell must keep the fast path"

    expected = _legacy_host_graph(
        atoms, cutoff, node_types, provider=fast_neighbor_list
    )
    assert graph.num_edges > 0
    for name, actual in (
        ("receivers", graph.receivers),
        ("sources", graph.sources),
        ("shifts", graph.shifts),
        ("num_neigh", graph.num_neigh),
        ("neigh_types", graph.neighbor_types(node_types)),
        ("xyz", graph.edge_vectors(atoms.positions, atoms.cell)),
        ("distances", graph.edge_distances(atoms.positions, atoms.cell)),
    ):
        assert actual.tobytes() == expected[name].tobytes(), name

    # The fast path must agree with ASE on which edges exist, which is the
    # property that lets it replace the fallback without changing any result.
    ase_edges = _legacy_host_graph(
        atoms, cutoff, node_types, provider=ase_neighbor_list
    )

    def _canonical(receivers, sources, shifts):
        receivers = np.asarray(receivers, dtype=np.int64).reshape(-1)
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
        order = np.lexsort(
            (shifts[:, 2], shifts[:, 1], shifts[:, 0], sources, receivers)
        )
        return receivers[order], sources[order], shifts[order]

    fast = _canonical(graph.receivers, graph.sources, graph.shifts)
    slow = _canonical(ase_edges["receivers"], ase_edges["sources"], ase_edges["shifts"])
    for name, fast_part, slow_part in zip(
        ("receivers", "sources", "shifts"), fast, slow
    ):
        assert np.array_equal(fast_part, slow_part), name


@pytest.mark.parametrize(
    "pbc",
    [tuple(bool(mask & (1 << axis)) for axis in range(3)) for mask in range(8)],
)
def test_fast_path_admission_depends_on_cell_not_periodicity(pbc):
    """Every periodicity mask is admissible when the finite cell is invertible."""

    atoms = Atoms("H", positions=[[1.0, 1.0, 1.0]], cell=[4.0, 5.0, 6.0], pbc=pbc)

    assert neighbor_graph_module._fast_path_admits(atoms)


@pytest.mark.parametrize(
    "cell",
    (
        np.zeros((3, 3)),
        np.diag([4.0, 5.0, 0.0]),
        np.diag([4.0, 5.0, np.inf]),
        np.diag([4.0, 5.0, np.nan]),
    ),
    ids=("zero", "rank-deficient", "infinite", "nan"),
)
def test_fast_path_rejects_cells_that_cannot_be_inverted_finitely(cell):
    """Invalid cells stay on ASE instead of reaching matscipy's matrix inverse."""

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=False)

    assert not neighbor_graph_module._fast_path_admits(atoms)


@pytest.mark.parametrize(
    ("displacement", "skin", "expected"),
    (
        (0.0, 0.5, True),
        (0.25, 0.5, True),
        (0.250001, 0.5, False),
        (0.0, 0.0, False),
        (1.0, 0.0, False),
    ),
)
def test_skin_reuse_holds_tracks_the_half_skin_rule(displacement, skin, expected):
    assert neighbor_graph_module.skin_reuse_holds(displacement, skin) is expected


def _topology_reference(atoms):
    cell = np.asarray(atoms.cell, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if np.all(pbc) and not np.isclose(np.linalg.det(cell), 0.0):
        fractional_positions = positions @ np.linalg.inv(cell)
    else:
        fractional_positions = np.empty((0, 3), dtype=float)
    return neighbor_graph_module.NeighborTopologyReference(
        atomic_numbers=np.asarray(atoms.numbers, dtype=int).copy(),
        pbc=pbc.copy(),
        topology_cell=cell.copy(),
        topology_reference_positions=positions.copy(),
        topology_fractional_positions=fractional_positions,
        cell=cell.copy(),
        reference_positions=positions.copy(),
    )


@pytest.mark.parametrize(
    ("displacement", "expected_reusable"),
    ((0.0, True), (0.49, True), (0.5, True), (0.5001, False)),
)
def test_topology_reuse_applies_the_inclusive_half_skin_limit(
    displacement, expected_reusable
):
    atoms = Atoms("H", positions=[[1.0, 1.0, 1.0]], cell=[10.0] * 3, pbc=True)
    reference = _topology_reference(atoms)
    atoms.positions[0, 0] += displacement

    decision = neighbor_graph_module.neighbor_topology_reuse(
        reference=reference,
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=np.asarray(atoms.pbc),
        cutoff=8.0,
        skin=1.0,
    )

    assert decision.reusable is expected_reusable
    assert decision.cell_changed is False
    assert decision.maximum_displacement == pytest.approx(displacement)


def test_topology_reuse_rejects_changed_species_or_periodicity():
    atoms = Atoms(
        "H2",
        positions=[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
        cell=[10.0] * 3,
        pbc=True,
    )
    reference = _topology_reference(atoms)

    for numbers, pbc in (
        (np.array([1, 2]), np.array([True, True, True])),
        (np.array([1, 1]), np.array([True, False, True])),
    ):
        decision = neighbor_graph_module.neighbor_topology_reuse(
            reference=reference,
            atomic_numbers=numbers,
            positions=atoms.positions,
            cell=np.asarray(atoms.cell),
            pbc=pbc,
            cutoff=8.0,
            skin=1.0,
        )

        assert not decision.reusable
        assert decision.maximum_displacement is None


def test_topology_reuse_wraps_displacements_for_partial_periodicity():
    atoms = Atoms(
        "H",
        positions=[[0.1, 1.0, 1.0]],
        cell=[10.0] * 3,
        pbc=[True, False, False],
    )
    reference = _topology_reference(atoms)
    atoms.positions[0, 0] = 9.9

    decision = neighbor_graph_module.neighbor_topology_reuse(
        reference=reference,
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=np.asarray(atoms.pbc),
        cutoff=8.0,
        skin=1.0,
    )

    assert decision.reusable
    assert decision.maximum_displacement == pytest.approx(0.2)


def test_topology_reuse_rejects_a_singular_periodic_cell():
    atoms = Atoms(
        "H",
        positions=[[0.1, 0.0, 0.0]],
        cell=[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        pbc=[True, False, False],
    )
    reference = _topology_reference(atoms)

    decision = neighbor_graph_module.neighbor_topology_reuse(
        reference=reference,
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=np.asarray(atoms.pbc),
        cutoff=8.0,
        skin=1.0,
    )

    assert not decision.reusable
    assert decision.maximum_displacement is None


@pytest.mark.parametrize(
    ("new_length", "expected_reusable"), ((9.9, True), (7.0, False))
)
def test_topology_reuse_validates_affine_contraction_from_topology_cell(
    new_length, expected_reusable
):
    atoms = Atoms(
        "H2",
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.2, 0.3]],
        cell=[10.0] * 3,
        pbc=True,
    )
    reference = _topology_reference(atoms)
    atoms.set_cell([new_length, 10.0, 10.0], scale_atoms=True)

    decision = neighbor_graph_module.neighbor_topology_reuse(
        reference=reference,
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=np.asarray(atoms.pbc),
        cutoff=8.0,
        skin=1.0,
    )

    assert decision.reusable is expected_reusable
    assert decision.cell_changed
    assert decision.maximum_displacement == pytest.approx(0.0)
    if expected_reusable:
        np.testing.assert_allclose(
            decision.geometry_reference_positions, atoms.positions
        )
        np.testing.assert_allclose(decision.inverse_cell, np.linalg.inv(atoms.cell))
    else:
        assert decision.geometry_reference_positions is None
        assert decision.inverse_cell is None


def _oam_graph_calculator():
    """Build an OAM calculator whose graph and reference surface need no backend."""

    import symmetrix.nequip_calculator as calculator_module

    contract = json.loads(OAM_FIXTURE.read_text())
    numbers = [int(number) for number in contract["atomic_numbers"]]

    class Evaluator:
        r_cut = float(contract["r_cut"])
        atomic_numbers = numbers
        available_heads = []

        def __init__(self, *args, **kwargs):
            self.edge_calls = []
            self.modes = []
            self.execution_plan_requests = []

        def set_streamed_edges(self, mode):
            self.modes.append(mode)

        def _set_execution_plan_request(self, algorithm, profile):
            self.execution_plan_requests.append((algorithm, profile))

        def compute_node_energies_forces(self, *args):
            self.edge_calls.append(args)
            self.node_energies = np.zeros(args[0], dtype=float)
            self.node_forces = np.zeros((len(args[3]), 3), dtype=float)

    class Native:
        # The OAM family declares the Kokkos evaluator only, and the frontend
        # initializes the execution space before it resolves that class.
        NequIPOAMKokkos = Evaluator

        @staticmethod
        def _kokkos_is_initialized():
            return True

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(calculator_module, "load_backend", lambda: Native)
        calculator = calculator_module.NequIPCalculator(
            OAM_FIXTURE, dtype="float64", streamed_edges="mlp_reference"
        )
    assert calculator.evaluator.execution_plan_requests == [("reference", "capacity")]
    return calculator


def test_oam_frontend_builds_its_graph_through_the_shared_builder():
    calculator = _oam_graph_calculator()
    atoms = _oam_atoms()
    node_types = np.asarray(
        [calculator.evaluator.atomic_numbers.index(number) for number in atoms.numbers],
        dtype=np.int32,
    )

    graph = neighbor_graph_module.build_host_neighbor_graph(
        atoms, calculator.cutoff + calculator.neighbor_skin
    )
    built = calculator._build_graph(atoms)

    assert built[0].tobytes() == node_types.tobytes()
    assert built[1].tobytes() == graph.num_neigh.tobytes()
    assert built[2].tobytes() == graph.sources.tobytes()
    assert built[3].tobytes() == graph.neighbor_types(node_types).tobytes()
    assert built[4].tobytes() == graph.receivers.tobytes()
    assert built[5].tobytes() == graph.shifts.tobytes()
    assert (
        built[6].tobytes() == graph.edge_vectors(atoms.positions, atoms.cell).tobytes()
    )
    assert (
        built[7].tobytes()
        == graph.edge_distances(atoms.positions, atoms.cell).tobytes()
    )


def test_oam_reuse_keeps_the_edges_and_measures_them_from_the_current_positions():
    calculator = _oam_graph_calculator()
    atoms = _oam_atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()
    assert calculator.evaluator.edge_calls
    reused_edges = calculator._graph[4].tobytes()
    first_xyz = calculator._graph[6].tobytes()

    atoms.positions[1] += [0.02, -0.01, 0.01]
    atoms.get_potential_energy()

    assert calculator._graph[4].tobytes() == reused_edges
    assert calculator._graph[6].tobytes() != first_xyz
    graph = neighbor_graph_module.build_host_neighbor_graph(
        atoms, calculator.cutoff + calculator.neighbor_skin
    )
    assert (
        calculator._graph[6].tobytes()
        == graph.edge_vectors(atoms.positions, atoms.cell).tobytes()
    )
    assert (
        calculator._graph[7].tobytes()
        == graph.edge_distances(atoms.positions, atoms.cell).tobytes()
    )


def test_oam_rebuild_past_half_the_skin_agrees_with_a_fresh_builder():
    calculator = _oam_graph_calculator()
    atoms = _oam_atoms()
    atoms.calc = calculator
    atoms.get_potential_energy()

    # Displace past half the skin so the next evaluation rebuilds the graph.
    atoms.positions[1] += [0.4, 0.0, 0.0]
    atoms.get_potential_energy()
    graph = neighbor_graph_module.build_host_neighbor_graph(
        atoms, calculator.cutoff + calculator.neighbor_skin
    )
    assert calculator._graph[2].tobytes() == graph.sources.tobytes()
    assert (
        calculator._graph[6].tobytes()
        == graph.edge_vectors(atoms.positions, atoms.cell).tobytes()
    )
    assert (
        calculator._graph[7].tobytes()
        == graph.edge_distances(atoms.positions, atoms.cell).tobytes()
    )


def test_oam_graph_builder_still_rejects_an_unsupported_element():
    calculator = _oam_graph_calculator()
    unsupported = max(calculator.evaluator.atomic_numbers) + 1
    atoms = SimpleNamespace(
        get_atomic_numbers=lambda: np.asarray([1, unsupported], dtype=int)
    )
    with pytest.raises(ValueError, match="does not support atomic numbers"):
        calculator._build_graph(atoms)
