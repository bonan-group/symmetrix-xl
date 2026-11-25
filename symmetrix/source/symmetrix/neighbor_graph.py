"""Neutral construction of the host neighbor graph every frontend shares.

One builder produces the graph a calculator hands to a native evaluator: the
matscipy fast path with an ASE fallback, the 32-bit CSR cardinality guard, the
receiver-major edge order, the edge vectors and distances measured from that
order, and the skin predicate that decides whether a cached graph still covers
the current geometry.  Families differ in what they do *with* the graph (which
evaluator consumes it, which device tier builds it), never in how the host
graph is built, so this module names no family and holds no family value.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

try:
    from matscipy.neighbours import neighbour_list as _fast_neighbor_list
except ImportError:  # pragma: no cover - exercised by the ASE fallback
    logging.warning("Symmetrix using slow ase.neighborlist.neighbor_list")
    _fast_neighbor_list = None
from ase.neighborlist import neighbor_list as _ase_neighbor_list

neighbor_list = (
    _ase_neighbor_list if _fast_neighbor_list is None else _fast_neighbor_list
)

__all__ = [
    "FixedCellReuseLimit",
    "HostNeighborGraph",
    "INT32_MAX",
    "NEIGHBOR_EDGE_ORDERS",
    "NEIGHBOR_EDGE_ORDER_ENVIRONMENT_VARIABLE",
    "NeighborReuseDecision",
    "NeighborTopologyReference",
    "build_host_neighbor_graph",
    "host_neighbor_list",
    "edge_vectors",
    "fixed_cell_reuse_holds",
    "fixed_cell_reuse_limit",
    "minimum_image_displacements",
    "neighbor_list",
    "neighbor_topology_reuse",
    "receiver_major_neighbor_arrays",
    "skin_reuse_holds",
    "validate_graph_cardinality",
]

#: Largest value a 32-bit CSR index can hold.
INT32_MAX = int(np.iinfo(np.int32).max)

#: Environment variable selecting the edge-order policy of the host builder.
NEIGHBOR_EDGE_ORDER_ENVIRONMENT_VARIABLE = "SYMMETRIX_DEBUG_NEIGHBOR_EDGE_ORDER"
NEIGHBOR_EDGE_ORDERS = frozenset(("receiver-major", "canonical"))

#: Fraction of the skin a cached graph may be displaced before it is rebuilt.
SKIN_REUSE_FRACTION = 0.5


@dataclass(frozen=True)
class NeighborTopologyReference:
    """Geometry snapshots needed to decide whether a topology is reusable."""

    atomic_numbers: np.ndarray
    pbc: np.ndarray
    topology_cell: np.ndarray
    topology_reference_positions: np.ndarray
    topology_fractional_positions: np.ndarray
    cell: np.ndarray
    reference_positions: np.ndarray


@dataclass(frozen=True)
class NeighborReuseDecision:
    """Outcome and geometry updates from a topology-reuse check.

    ``maximum_displacement`` is the largest displacement the check measured:
    the minimum-image non-affine displacement from the topology reference on
    a fully periodic cell, and the minimum-image displacement from the
    topology reference positions otherwise.  It is ``None`` when nothing was
    measured: the check rejected the geometry before measuring, or a
    :class:`FixedCellReuseLimit` certified the reuse without the check.
    """

    reusable: bool
    cell_changed: bool
    maximum_displacement: float | None
    geometry_reference_positions: np.ndarray | None
    inverse_cell: np.ndarray | None


def minimum_image_displacements(
    positions, reference_positions, cell, pbc
) -> np.ndarray | None:
    """Return displacement vectors, wrapping axes declared periodic.

    A periodic cell must be invertible even when only some axes are periodic.
    ``None`` reports that no well-defined minimum-image displacement exists.
    """

    displacements = np.asarray(positions) - np.asarray(reference_positions)
    pbc = np.asarray(pbc, dtype=bool)
    if not np.any(pbc):
        return displacements
    try:
        fractional = np.linalg.solve(np.asarray(cell).T, displacements.T).T
    except np.linalg.LinAlgError:
        return None
    fractional[:, pbc] -= np.rint(fractional[:, pbc])
    return fractional @ np.asarray(cell)


def _maximum_norm(vectors):
    return 0.0 if len(vectors) == 0 else float(np.linalg.norm(vectors, axis=1).max())


def neighbor_topology_reuse(
    *,
    reference: NeighborTopologyReference,
    atomic_numbers,
    positions,
    cell,
    pbc,
    cutoff,
    skin,
) -> NeighborReuseDecision:
    """Decide whether a cached neighbor topology still covers a geometry."""

    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    cell_changed = not np.array_equal(cell, reference.cell)
    incompatible = (
        not np.array_equal(atomic_numbers, reference.atomic_numbers)
        or not np.array_equal(pbc, reference.pbc)
        or positions.shape != reference.reference_positions.shape
    )
    if incompatible:
        return NeighborReuseDecision(False, cell_changed, None, None, None)

    fully_periodic_fractional = bool(np.all(pbc)) and (
        np.asarray(reference.topology_fractional_positions).shape == positions.shape
    )
    if fully_periodic_fractional:
        topology_fractional = np.asarray(reference.topology_fractional_positions)
        try:
            non_affine = positions - topology_fractional @ cell
            non_affine_fractional = np.linalg.solve(cell.T, non_affine.T).T
            deformation = np.linalg.solve(cell, reference.topology_cell)
        except np.linalg.LinAlgError:
            return NeighborReuseDecision(False, cell_changed, None, None, None)
        non_affine_fractional[:, pbc] -= np.rint(non_affine_fractional[:, pbc])
        non_affine = non_affine_fractional @ cell
        maximum_non_affine_displacement = _maximum_norm(non_affine)
        contraction = float(np.linalg.norm(deformation, ord=2))
        topology_valid = contraction * (
            float(cutoff) + 2.0 * maximum_non_affine_displacement
        ) <= float(cutoff) + float(skin)
        if not topology_valid:
            return NeighborReuseDecision(
                False,
                cell_changed,
                maximum_non_affine_displacement,
                None,
                None,
            )
        if cell_changed:
            geometry_reference_positions = topology_fractional @ cell + non_affine
            return NeighborReuseDecision(
                True,
                True,
                maximum_non_affine_displacement,
                geometry_reference_positions,
                np.linalg.inv(cell),
            )
        return NeighborReuseDecision(
            True, False, maximum_non_affine_displacement, None, None
        )

    if cell_changed:
        return NeighborReuseDecision(False, True, None, None, None)
    displacements = minimum_image_displacements(
        positions, reference.topology_reference_positions, cell, pbc
    )
    if displacements is None:
        return NeighborReuseDecision(False, False, None, None, None)
    maximum_displacement = _maximum_norm(displacements)
    return NeighborReuseDecision(
        skin_reuse_holds(maximum_displacement, skin),
        False,
        maximum_displacement,
        None,
        None,
    )


@dataclass(frozen=True)
class FixedCellReuseLimit:
    """A cheap sufficient condition for reusing a fully periodic topology.

    While the cell stays bit-identical to the topology cell, a geometry whose
    largest Cartesian displacement component from the topology reference
    positions is below ``maximum_displacement`` is one that
    :func:`neighbor_topology_reuse` admits.  The limit is the smaller of two
    bounds, each kept inside by a relative margin and reduced by the
    reconstruction error of the fractional reference:

    * the skin bound: a component bound ``m`` bounds every displacement norm
      by ``sqrt(3) m``, so ``2 sqrt(3) m`` must stay within the skin the
      general check allows at the topology cell's own contraction;
    * the image bound: on every periodic axis ``a`` the fractional
      displacement is at most ``m sum_b |(H^-1)_ba|``, which must stay below
      one half.  Without it a small Cartesian move in a thin, skewed cell can
      cross half a cell in fractional coordinates; minimum-image rounding
      then selects another image for every stored edge of that atom, and the
      general check correctly rejects a geometry the skin bound alone admits.

    Cells whose condition number exceeds
    :data:`FIXED_CELL_REUSE_MAXIMUM_CONDITION` receive no limit, so the
    rounding of the general check's solve stays far inside the margin.  The
    limit holds only for the ``cutoff`` and ``skin`` it was computed with.
    """

    cell: np.ndarray
    reference_positions: np.ndarray
    maximum_displacement: float
    cutoff: float
    skin: float


#: Relative margin by which the fixed-cell limit stays inside both bounds.
FIXED_CELL_REUSE_MARGIN = 1.0e-6

#: Largest cell condition number for which a fixed-cell limit is issued.
FIXED_CELL_REUSE_MAXIMUM_CONDITION = 1.0e6


def fixed_cell_reuse_limit(
    reference: NeighborTopologyReference, *, cutoff, skin
) -> FixedCellReuseLimit | None:
    """Return the fixed-cell reuse limit of a fully periodic topology.

    ``None`` means the reference has no such limit (partial periodicity, a
    cell already moved away from the topology cell, a singular or
    ill-conditioned cell, or no skin left), and every geometry takes
    :func:`neighbor_topology_reuse`.
    """

    pbc = np.asarray(reference.pbc, dtype=bool)
    cell = np.asarray(reference.topology_cell, dtype=float)
    positions = np.asarray(reference.topology_reference_positions, dtype=float)
    fractional = np.asarray(reference.topology_fractional_positions, dtype=float)
    if (
        pbc.shape != (3,)
        or not np.all(pbc)
        or cell.shape != (3, 3)
        or fractional.shape != positions.shape
        or not np.array_equal(np.asarray(reference.cell, dtype=float), cell)
        or not np.all(np.isfinite(cell))
    ):
        return None
    try:
        # The contraction the general check computes at this cell, by the
        # same expression.
        contraction = float(np.linalg.norm(np.linalg.solve(cell, cell), ord=2))
        inverse_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return None
    condition = float(np.linalg.cond(cell))
    if not condition <= FIXED_CELL_REUSE_MAXIMUM_CONDITION:
        return None
    skin_bound = ((float(cutoff) + float(skin)) / contraction - float(cutoff)) / (
        2.0 * np.sqrt(3.0)
    )
    image_bound = 0.5 / float(np.abs(inverse_cell).sum(axis=0).max())
    reconstruction = (
        0.0
        if len(positions) == 0
        else float(np.abs(positions - fractional @ cell).max())
    )
    maximum_displacement = (
        min(skin_bound, image_bound) * (1.0 - FIXED_CELL_REUSE_MARGIN) - reconstruction
    )
    if not maximum_displacement > 0.0:
        return None
    return FixedCellReuseLimit(
        cell=np.array(cell, copy=True),
        reference_positions=reference.topology_reference_positions,
        maximum_displacement=maximum_displacement,
        cutoff=float(cutoff),
        skin=float(skin),
    )


def fixed_cell_reuse_holds(
    limit: FixedCellReuseLimit | None,
    *,
    reference: NeighborTopologyReference,
    atomic_numbers,
    positions,
    cell,
    pbc,
    cutoff,
    skin,
) -> bool:
    """Report whether ``limit`` certifies reuse of ``reference`` for a geometry.

    ``True`` implies that :func:`neighbor_topology_reuse` with the same
    ``cutoff`` and ``skin`` reuses the topology without a cell change.
    ``False`` decides nothing; the caller then runs the general check.  The
    limit applies only to the reference and the cutoff and skin it was
    computed for, and only while the reference and current cells both equal
    its cell.
    """

    if (
        limit is None
        or limit.reference_positions is not reference.topology_reference_positions
        or float(cutoff) != limit.cutoff
        or float(skin) != limit.skin
    ):
        return False
    positions = np.asarray(positions, dtype=float)
    if (
        positions.shape != limit.reference_positions.shape
        or not np.array_equal(np.asarray(cell, dtype=float), limit.cell)
        or not np.array_equal(np.asarray(reference.cell, dtype=float), limit.cell)
        or not np.array_equal(np.asarray(pbc, dtype=bool), reference.pbc)
        or not np.array_equal(
            np.asarray(atomic_numbers, dtype=int), reference.atomic_numbers
        )
    ):
        return False
    if np.array_equal(positions, limit.reference_positions):
        return True
    difference = positions - limit.reference_positions
    bound = limit.maximum_displacement
    # Written as two comparisons so a non-finite component never certifies.
    return bool(difference.max() < bound and -difference.min() < bound)


def validate_graph_cardinality(num_atoms, num_edges):
    """Reject a graph whose node or directed-edge count cannot be a 32-bit CSR.

    The native topology, node, and edge arrays are 32-bit, so a larger graph
    would narrow silently.  Every graph entry point validates before it
    allocates or narrows, and the message names the limit that was exceeded.
    """

    num_atoms = int(num_atoms)
    num_edges = int(num_edges)
    if num_atoms < 0 or num_edges < 0:
        raise ValueError(
            "Neighbor graph atom and directed-edge counts must be non-negative."
        )
    if num_atoms >= INT32_MAX:
        raise ValueError(
            f"Neighbor graph atom count {num_atoms} exceeds the 32-bit CSR "
            f"limit {INT32_MAX - 1}."
        )
    if num_edges > INT32_MAX:
        raise ValueError(
            f"Neighbor graph directed-edge count {num_edges} exceeds the 32-bit "
            f"topology limit {INT32_MAX}."
        )


def receiver_major_neighbor_arrays(receivers, sources, shifts):
    """Return aligned, contiguous edges grouped by receiver.

    Matscipy and ASE already emit ascending receiver indices.  Preserve their
    within-receiver order because direct execution has no source/shift sorting
    requirement.  The canonical mode exists only for numerical A/B diagnosis.
    """

    receivers = np.asarray(receivers, dtype=np.int32).reshape(-1)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    shifts = np.asarray(shifts, dtype=np.int32).reshape((-1, 3))
    if len(receivers) != len(sources) or len(receivers) != len(shifts):
        raise ValueError("Neighbor receiver, source, and shift extents must match.")

    order_policy = os.environ.get(
        NEIGHBOR_EDGE_ORDER_ENVIRONMENT_VARIABLE, "receiver-major"
    )
    if order_policy not in NEIGHBOR_EDGE_ORDERS:
        choices = ", ".join(sorted(NEIGHBOR_EDGE_ORDERS))
        raise ValueError(
            f"{NEIGHBOR_EDGE_ORDER_ENVIRONMENT_VARIABLE} must be one of "
            f"{choices}; got {order_policy!r}."
        )
    if order_policy == "canonical":
        order = np.lexsort(
            (
                shifts[:, 2],
                shifts[:, 1],
                shifts[:, 0],
                sources,
                receivers,
            )
        )
    elif len(receivers) > 1 and np.any(receivers[1:] < receivers[:-1]):
        order = np.argsort(receivers, kind="stable")
    else:
        order = None

    if order is not None:
        receivers = receivers[order]
        sources = sources[order]
        shifts = shifts[order]
    return (
        np.ascontiguousarray(receivers),
        np.ascontiguousarray(sources),
        np.ascontiguousarray(shifts),
    )


def skin_reuse_holds(maximum_displacement, neighbor_skin):
    """Report whether a cached graph still covers a displacement.

    A graph is built at ``cutoff + neighbor_skin``, so it stays valid while no
    atom has moved more than half the skin.  A zero skin means every geometry
    change rebuilds the graph.
    """

    neighbor_skin = float(neighbor_skin)
    if neighbor_skin <= 0.0:
        return False
    return float(maximum_displacement) <= SKIN_REUSE_FRACTION * neighbor_skin


def edge_vectors(positions, cell, receivers, sources, shifts):
    """Return the source-to-receiver vector of every edge.

    The sign convention is fixed here: an edge points from the receiver to the
    source, offset by its periodic shift, exactly as the prepared native
    geometry expects it.
    """

    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float).reshape((3, 3))
    return np.ascontiguousarray(
        positions[sources] - positions[receivers] + shifts @ cell,
        dtype=float,
    )


@dataclass(frozen=True)
class HostNeighborGraph:
    """The host neighbor graph: receiver-major edges and node degrees.

    ``receivers``, ``sources``, and ``shifts`` are the aligned edge arrays, and
    ``num_neigh`` is the receiver degree each node reports.  The arrays are
    contiguous 32-bit tensors, so a native evaluator can consume them without
    narrowing or copying.
    """

    receivers: np.ndarray
    sources: np.ndarray
    shifts: np.ndarray
    num_neigh: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(len(self.receivers))

    def edge_vectors(self, positions, cell):
        """Return the source-to-receiver vector of every edge."""

        return edge_vectors(positions, cell, self.receivers, self.sources, self.shifts)

    def edge_distances(self, positions, cell):
        """Return the length of every edge vector."""

        return np.linalg.norm(self.edge_vectors(positions, cell), axis=1)

    def neighbor_types(self, node_types):
        """Return the source node type of every edge, aligned with the edges."""

        return np.ascontiguousarray(
            np.asarray(node_types, dtype=np.int32)[self.sources],
            dtype=np.int32,
        )


def _fast_path_admits(atoms) -> bool:
    """True when the matscipy fast path can build this cell.

    matscipy's ``neighbour_list`` inverts the cell unconditionally and honours
    ``pbc`` axis by axis, so every invertible cell qualifies — a vacuum-padded
    slab and a molecule in a finite box included, both of which it reproduces
    edge for edge against ASE's implementation.  A cell ASE leaves as zeros, or
    whose vectors do not span space, is singular and stays with ASE, whose
    implementation builds a zero-cell molecule as readily as a crystal.
    """

    cell = getattr(atoms, "cell", None)
    if cell is None:
        return False
    cell = np.asarray(getattr(cell, "array", cell), dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        return False
    sign, log_abs_determinant = np.linalg.slogdet(cell)
    return bool(sign) and bool(np.isfinite(log_abs_determinant))


def host_neighbor_list(atoms, cutoff, request="ijS"):
    """Return the host provider's arrays for ``atoms`` at ``cutoff``.

    ``request`` is the neighbor-list format string the caller needs: the graph
    builder asks for ``"ijS"`` and the legacy host geometry path for ``"ijdD"``.

    The fast path serves every finite, invertible cell and honours ``pbc`` axis
    by axis.  The module-level ``neighbor_list`` stays the provider this calls,
    so a caller that substitutes the provider keeps working.  Singular or
    non-finite cells go to ASE, whose implementation builds a zero-cell molecule
    as readily as a crystal; a ``LinAlgError`` from the fast path takes the same
    fallback, so a cell that turns out to be singular never reaches the caller.
    """

    if not _fast_path_admits(atoms):
        return _ase_neighbor_list(request, atoms, float(cutoff))
    try:
        return neighbor_list(request, atoms, float(cutoff))
    except np.linalg.LinAlgError:
        logging.warning(
            "Symmetrix falling back to ase.neighborlist.neighbor_list: the "
            "matscipy fast path rejected a singular cell"
        )
        return _ase_neighbor_list(request, atoms, float(cutoff))


def build_host_neighbor_graph(atoms, cutoff):
    """Build the host neighbor graph of ``atoms`` at ``cutoff``.

    The matscipy fast path is preferred and ASE is the fallback, the 32-bit
    cardinality guard runs before any narrowing, and the edges are returned in
    receiver-major order.  Callers pass the full search radius, which includes
    any skin they intend to cache.
    """

    num_atoms = len(atoms)
    receivers, sources, shifts = host_neighbor_list(atoms, cutoff)
    validate_graph_cardinality(num_atoms, len(receivers))
    receivers, sources, shifts = receiver_major_neighbor_arrays(
        receivers, sources, shifts
    )
    num_neigh = np.asarray(
        np.bincount(receivers, minlength=num_atoms),
        dtype=np.int32,
    )
    return HostNeighborGraph(
        receivers=receivers,
        sources=sources,
        shifts=shifts,
        num_neigh=num_neigh,
    )
