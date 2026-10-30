"""The family-neutral conformance contract: one adapter lifecycle, any family.

The family adapter (``libsymmetrix/source/family_interface.hpp``) promises one
contract for every retained model family.  This module is that contract in
Python: the graph builder, the reference reduction and finite-strain oracles,
the ``Adapter`` base class that maps the neutral lifecycle onto an evaluator,
and the ``Unqualified`` report a surface this runner cannot qualify raises.

Nothing here names a family.  Each family declares its own structures, session
models, adapters, and instance specs in one ``test_family_<family>_conformance_cases``
module beside the shared suites; ``conformance.registry`` collects those
declarations and the shared suite drives them through this contract.

Graph convention (one builder for every family).  Directed edges are indexed by
receiver, ``sources`` holds the neighbor node, ``num_neigh`` counts the edges
each receiver owns, and ``xyz`` is ``positions[sources] - positions[receivers] +
shifts @ cell`` for the minimum image.  That is the shared neighbor-cache
convention, so the suite builds one topology and one set of buffers for all
families instead of a per-family input dialect.

The T0 reference of every family is the generic implementation executed on the
default host execution space with no artifact, driven through its own
whole-evaluator entry point.  No reference entry point leaves a completed
generation to reduce against, so no reference produces stress: the stress oracle
is the finite-strain derivative of the family's own reference energy, which each
family's adapter supplies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

try:
    from symmetrix import symmetrix as native_symmetrix
except ImportError as exc:  # pragma: no cover - environment without backend
    pytest.skip(
        f"native Symmetrix package is unavailable: {exc}", allow_module_level=True
    )

# The buffer orders the neutral layout contract admits (``LayoutContract``).
LAYOUT_ORDERS = frozenset(("node_major", "lane_major", "edge_major", "block_major"))

# The tiers below the generated one.  A host backend admits a program only when
# every node is served by one of these, so this is the portable-coverage set.
PORTABLE_TIERS = frozenset(("reference", "portable", "frozen"))

# The runner an accelerator-only surface requires.  Accelerator tiers are
# qualified by a CUDA or HIP build selected through ``SYMMETRIX_BACKEND``; the
# suite reports them as unqualified on a host runner and names this requirement
# rather than skipping the check.
ACCELERATOR_RUNNER = (
    "a CUDA or HIP backend build selected through SYMMETRIX_BACKEND "
    "(for example SYMMETRIX_BACKEND=cuda13-sm120 with the CUDA 13.3 toolkit, or "
    "the HIP/ROCm build for the target gfx architecture)"
)
FLOAT32_RUNNER = "a Symmetrix build with Kokkos float32 support"
CXX_RUNNER = "a host C++20 compiler (a working `c++`)"
# The generated tier's operator-kind coverage a CUDA/HIP runner must add.  On a
# host runner these are reported as unqualified with the required runner.
ACCELERATOR_SURFACES = ("cuda", "hip")


def execution_space() -> str:
    """The execution space this build runs on."""

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    return native_symmetrix._kokkos_default_execution_space()


class Unqualified(RuntimeError):
    """A tier surface this runner cannot qualify, with the runner it requires."""

    def __init__(self, surface: str, runner: str):
        super().__init__(
            f"{surface} is unqualified on this runner; it requires {runner}"
        )
        self.surface = surface
        self.runner = runner


# ---------------------------------------------------------------------------
# Graph construction, the reference reduction, and the stress oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Graph:
    """One topology plus the reference and current geometry of one structure."""

    num_nodes: int
    node_types: np.ndarray
    num_neigh: np.ndarray
    sources: np.ndarray
    neigh_types: np.ndarray
    reference_positions: np.ndarray
    positions: np.ndarray
    xyz: np.ndarray
    distances: np.ndarray
    receivers: np.ndarray
    cell: np.ndarray
    inverse_cell: np.ndarray
    pbc: np.ndarray
    volume: float
    # The structure the prepared edge list belongs to, so a displaced geometry
    # can reuse it instead of rebuilding the topology.
    topology_atoms: "Atoms"

    @property
    def arguments(self) -> tuple:
        """The whole-evaluator argument tuple every family accepts."""
        return (
            self.num_nodes,
            self.node_types,
            self.num_neigh,
            self.sources,
            self.neigh_types,
            self.xyz,
            self.distances,
        )

    def topology(self) -> tuple:
        """The topology arrays one graph generation is bound to."""
        return (
            self.num_nodes,
            self.node_types,
            self.num_neigh,
            self.sources,
            self.neigh_types,
        )

    def reused_at(self, atoms: Atoms, atomic_numbers, cutoff: float) -> "Graph":
        """The same topology and skin at displaced positions.

        The edge list, the reference geometry the graph was prepared against,
        and the cell are unchanged; only the current positions and the current
        displacement vectors move, which is exactly the within-skin reuse the
        calculator performs.
        """
        return _graph(atoms, atomic_numbers, cutoff, topology_of=self.topology_atoms)


def _graph(
    atoms: Atoms,
    atomic_numbers,
    cutoff: float,
    *,
    topology_of: Atoms | None = None,
) -> Graph:
    """Build the neutral topology and geometry for one structure.

    ``topology_of`` supplies the structure whose edge list is reused, so a
    displaced geometry keeps the prepared topology and only moves.
    """
    topology_atoms = atoms if topology_of is None else topology_of
    receivers, sources, shifts, _distances = neighbor_list(
        "ijSd", topology_atoms, cutoff
    )
    order = np.argsort(receivers, kind="stable")
    receivers = np.asarray(receivers[order], dtype=np.int32)
    sources = np.asarray(sources[order], dtype=np.int32)
    shifts = np.asarray(shifts[order], dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    xyz = positions[sources] - positions[receivers] + shifts @ cell
    type_by_number = {int(number): index for index, number in enumerate(atomic_numbers)}
    numbers = [int(number) for number in atoms.numbers]
    node_types = np.asarray(
        [type_by_number[number] for number in numbers], dtype=np.int32
    )
    num_neigh = np.bincount(receivers, minlength=len(atoms)).astype(np.int32)
    if np.any(atoms.pbc):
        inverse_cell = np.linalg.inv(cell)
        pbc = np.asarray(atoms.pbc, dtype=np.int32)
    else:
        inverse_cell = np.eye(3)
        pbc = np.zeros(3, dtype=np.int32)
    reference_positions = (
        positions
        if topology_of is None
        else np.asarray(topology_of.positions, dtype=np.float64)
    )
    return Graph(
        num_nodes=len(atoms),
        node_types=node_types,
        num_neigh=num_neigh,
        sources=sources,
        neigh_types=node_types[sources],
        reference_positions=np.ascontiguousarray(reference_positions).reshape(-1),
        positions=np.ascontiguousarray(positions).reshape(-1),
        xyz=np.ascontiguousarray(xyz).reshape(-1),
        distances=np.linalg.norm(xyz, axis=1),
        receivers=receivers,
        cell=np.ascontiguousarray(cell).reshape(-1),
        inverse_cell=np.ascontiguousarray(inverse_cell).reshape(-1),
        pbc=pbc,
        volume=float(abs(np.linalg.det(cell))),
        topology_atoms=topology_atoms,
    )


def _directed_force_reduction(graph: Graph, directed_forces) -> np.ndarray:
    """Reduce directed edge forces to atom forces.

    The reduction is the module's documented identity: every directed force is
    added to its source atom and subtracted from its receiver.  The reference
    side is the T0 evaluator's own per-edge force array, so a reduction that
    swaps the roles, drops an edge, or flips the sign fails here.
    """
    forces = np.asarray(directed_forces, dtype=np.float64).reshape(-1, 3)
    atom_forces = np.zeros((graph.num_nodes, 3), dtype=np.float64)
    for component in range(3):
        column = forces[:, component]
        atom_forces[:, component] = np.bincount(
            graph.sources, weights=column, minlength=graph.num_nodes
        ) - np.bincount(graph.receivers, weights=column, minlength=graph.num_nodes)
    return atom_forces


def _finite_strain_oracle(
    energy_of: Callable[[Atoms], float], atoms: Atoms, *, step: float
) -> np.ndarray:
    """The stress tensor as the central finite-strain derivative of the energy.

    The index order is the transferable one the stress suites use:
    ``xx, yy, zz, yz, xz, xy``.
    """
    cell = np.asarray(atoms.cell.array, dtype=np.float64).copy()
    volume = float(atoms.get_volume())
    components = []
    for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        strain = np.zeros((3, 3))
        strain[i, j] = strain[j, i] = 1.0 if i == j else 0.5
        energies = []
        for sign in (1, -1):
            displaced = atoms.copy()
            displaced.set_cell(
                cell @ (np.eye(3) + sign * step * strain), scale_atoms=True
            )
            energies.append(energy_of(displaced))
        components.append((energies[0] - energies[1]) / (2.0 * step * volume))
    return np.asarray(components, dtype=np.float64)


def _voigt(stress: np.ndarray) -> np.ndarray:
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[1, 2],
            stress[0, 2],
            stress[0, 1],
        ]
    )


def _diagnostic(evaluator) -> str:
    return str(evaluator.family_diagnostics["diagnostic"])


# ---------------------------------------------------------------------------
# The adapters: the neutral lifecycle mapped onto each family
# ---------------------------------------------------------------------------


@dataclass
class Adapter:
    """One admitted family/precision/contract instance under conformance.

    ``lifecycle_tiers`` names the tiers whose *prepared-graph* lifecycle is
    admitted on a host runner.  Every family admits the compiler-free tier for
    its whole-evaluator entry point; a family whose field derivative requirement
    is only served by the artifact-bound plan declares that here instead of
    pretending the compiler-free plan carries it.
    """

    key: str
    family: str
    precision: str
    parameters: str
    oracle: str
    lifecycle_tiers: tuple[str, ...]
    expected_kinds: frozenset
    expected_layouts: frozenset
    expected_reductions: frozenset
    generated_kinds: frozenset
    rtol: float = 0.0
    atol: float = 1e-9
    # The finite-strain comparison gate, per precision.  The oracle itself is
    # always the float64 reference energy, so the step stays small; these bound
    # the subject's own precision error against it.
    stress_rtol: float = 2e-5
    stress_atol: float = 2e-9
    # The bound on the antisymmetric part of the reduced tensor.  A stress that
    # is the derivative of a rotationally invariant energy has none, so the
    # default is numerical zero; an instance whose reduction retains a measured
    # antisymmetric part carries the measured bound and a comment naming it.
    stress_antisymmetry_atol: float = 1e-9
    strain_step: float = 1e-5
    field_coupled: bool = False
    # Whether the instance serves a stress reduction at all.  A family whose
    # program declares no reduction node still serves stress from its evaluator,
    # so this is a declared fact per instance rather than an inference from the
    # program shape.
    serves_stress: bool = True
    field_vector: tuple = (0.01, -0.02, 0.03)

    _models: dict = field(default_factory=dict)
    _evaluators: dict = field(default_factory=dict)

    # -- evaluator construction ------------------------------------------
    def evaluator(self, tier: str):
        if tier not in self._evaluators:
            self._evaluators[tier] = self._build(tier)
        return self._evaluators[tier]

    def _build(self, tier: str):
        raise NotImplementedError

    def structure(self) -> Atoms:
        raise NotImplementedError

    def model_path(self) -> Path:
        raise NotImplementedError

    # -- the neutral lifecycle -------------------------------------------
    def bindings_on(self, evaluator) -> dict[str, str]:
        """Neutral lifecycle step -> the spelling this family exposes."""
        raise NotImplementedError

    def assert_lifecycle_bound(self, evaluator) -> None:
        spellings = self.bindings_on(evaluator)
        assert set(spellings) == {
            "prepare_graph",
            "prepare_geometry",
            "compute_prepared",
            "reduce_forces",
            "reduce_stress",
        }
        for step, spelling in spellings.items():
            assert callable(getattr(evaluator, spelling, None)), (step, spelling)

    def prepare_graph(self, evaluator, graph: Graph) -> int:
        raise NotImplementedError

    def prepare_geometry(self, evaluator, generation: int, graph: Graph) -> None:
        raise NotImplementedError

    def compute_prepared(self, evaluator, generation: int, graph: Graph) -> None:
        raise NotImplementedError

    def reduce_forces(self, evaluator, generation: int, graph: Graph) -> np.ndarray:
        raise NotImplementedError

    def reduce_stress(self, evaluator, generation: int, graph: Graph):
        """The reduced stress tensor, or ``None`` when the instance declares none."""
        raise NotImplementedError

    # -- observation -------------------------------------------------------
    def node_energies(self, evaluator) -> np.ndarray:
        return np.asarray(evaluator.node_energies, dtype=np.float64)

    def node_forces(self, evaluator) -> np.ndarray:
        return np.asarray(evaluator.node_forces, dtype=np.float64)

    def field_adjoint(self, evaluator):
        return None

    def expected_field_adjoint(self, tier: str, graph: Graph, reference_adjoint):
        """The reference field adjoint in the granularity this tier consumes."""
        return np.asarray(reference_adjoint, dtype=np.float64)

    def graph(self, atoms: Atoms, evaluator, *, topology_of=None) -> Graph:
        return _graph(
            atoms, evaluator.atomic_numbers, evaluator.r_cut, topology_of=topology_of
        )

    # -- generation accessors ---------------------------------------------
    # The families expose their generation tokens under their own names, so the
    # neutral driver reads them through the adapter instead of assuming one
    # spelling.
    def accepted_generation(self, evaluator) -> int:
        raise NotImplementedError

    def geometry_generation_of(self, evaluator):
        """The geometry token, or ``None`` when the family does not expose one."""
        return None

    # -- the T0 oracle -----------------------------------------------------
    def evaluate_reference(self, graph: Graph) -> dict:
        evaluator = self.evaluator("reference")
        evaluator.compute_node_energies_forces(*graph.arguments)
        return {
            "energies": self.node_energies(evaluator),
            "directed_forces": self.node_forces(evaluator),
            "field_adjoint": self.field_adjoint(evaluator),
        }

    def reference_energy(self, atoms: Atoms) -> float:
        evaluator = self.evaluator("reference")
        graph = self.graph(atoms, evaluator)
        evaluator.compute_node_energies_forces(*graph.arguments)
        return float(np.sum(self.node_energies(evaluator)))

    # -- the shared driver -------------------------------------------------
    def run_lifecycle(self, tier: str, atoms: Atoms | None = None) -> dict:
        evaluator = self.evaluator(tier)
        atoms = self.structure() if atoms is None else atoms
        graph = self.graph(atoms, evaluator)
        generation = self.prepare_graph(evaluator, graph)
        assert generation > 0, "prepare_graph must return a nonzero generation"
        assert self.accepted_generation(evaluator) == generation
        self.prepare_geometry(evaluator, generation, graph)
        geometry_generation = self.geometry_generation_of(evaluator)
        if geometry_generation is not None:
            assert geometry_generation == generation
        self.compute_prepared(evaluator, generation, graph)
        return {
            "evaluator": evaluator,
            "graph": graph,
            "generation": generation,
            "energies": self.node_energies(evaluator),
            "directed_forces": self.node_forces(evaluator),
            "field_adjoint": self.field_adjoint(evaluator),
            "atom_forces": self.reduce_forces(evaluator, generation, graph),
            "stress": self.reduce_stress(evaluator, generation, graph),
        }

    # -- tier facts --------------------------------------------------------
    def artifact_ready(self, evaluator) -> bool:
        return bool(evaluator.family_artifact_ready)

    def assert_tier_state(self, evaluator, tier: str) -> None:
        if tier != "generated":
            assert self.artifact_ready(evaluator) is False
            assert evaluator.family_artifact_readiness["artifact_id"] == ""
            assert evaluator.family_capabilities["generated"] is False
        else:
            assert self.artifact_ready(evaluator) is True
            assert evaluator.family_artifact_readiness["artifact_id"]

    def generated_execution_evidence(self, evaluator) -> dict:
        """Observable facts proving the admitted artifact served execution."""
        raise NotImplementedError

    def compiler_free_marker(self) -> tuple[str, ...]:
        """Diagnostic substrings that only the compiler-free path reports."""
        raise NotImplementedError

    def prepared_counters(self, evaluator) -> tuple[str, str] | None:
        """The ``(prepare, evaluate)`` counters that make graph reuse observable."""
        return None

    def accelerator_status(self) -> dict:
        """The device-artifact surfaces and what this runner can qualify.

        Each entry is the evaluator's own readiness report for that surface: a
        host runner reports an unready device artifact with its reason, a
        device runner reports the admitted artifact.
        """
        evaluator = self.evaluator("portable")
        capabilities = evaluator.family_capabilities
        status = {}
        for surface in ACCELERATOR_SURFACES:
            readiness = evaluator.load_family_artifact("", surface)
            ready = bool(readiness["ready"])
            status[surface] = {
                "qualified": ready,
                "reason": readiness["reason"],
                "artifact_id": readiness["artifact_id"],
                "target": readiness["target"],
                "declared_device_artifact": bool(capabilities["device_artifact"]),
                "round_trip": ready and capabilities["target"] == surface,
                "runner": ACCELERATOR_RUNNER,
            }
        return status
