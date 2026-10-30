"""Generic configuration and result composition for additive corrections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np
from ase import Atoms, units

from .fourier_d3_capability import fourier_d3_capability
from .fourier_d3_factors import decompose_d3_c6_reference
from .neighbor_graph import (
    NeighborTopologyReference,
    build_host_neighbor_graph,
    neighbor_topology_reuse,
)


_CORRECTION_PROPERTIES = frozenset({"energy", "energies", "forces", "stress"})
_D3_DAMPING = frozenset({"zero", "bj", "zerom", "bjm"})
_TOPOLOGIES = frozenset({"pair", "triplet", "global"})


def _validate_length(name: str, value: float, *, allow_zero: bool) -> None:
    label = name.replace("_", " ")
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    if value < 0.0 or (not allow_zero and value == 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")


@dataclass(frozen=True)
class GraphRequirement:
    """Neighbor topology required by a correction runtime, in Angstrom.

    ``pair_cutoff`` is the real-space pair radius of a pair or triplet interaction and is
    ``None`` for ``topology="global"``. A global interaction has no real-space pair
    cutoff at all — its dispersion sum runs over the lattice in reciprocal space — so the
    only radius it may declare is the ``coordination_cutoff`` of its short supporting
    graph. Supplying a pair cutoff for a global topology is rejected rather than ignored:
    the value would be a second, unused graph radius, and silently widening the graph is
    exactly the failure this representation exists to prevent.
    """

    pair_cutoff: float | None
    coordination_cutoff: float | None
    skin: float
    topology: Literal["pair", "triplet", "global"] = "pair"

    def __post_init__(self) -> None:
        if self.topology not in _TOPOLOGIES:
            raise ValueError("topology must be one of 'pair', 'triplet', or 'global'")
        if self.pair_cutoff is None:
            if self.topology != "global":
                raise ValueError(
                    f"pair cutoff is required for topology {self.topology!r}; only a "
                    "global interaction may omit it"
                )
        else:
            _validate_length("pair_cutoff", self.pair_cutoff, allow_zero=False)
            if self.topology == "global":
                raise ValueError(
                    "a global interaction has no real-space pair cutoff, so pair cutoff "
                    "must be None for topology 'global'"
                )
        if self.coordination_cutoff is None:
            if self.topology == "global":
                raise ValueError(
                    "a global interaction's only real-space graph is its coordination "
                    "graph, so coordination cutoff is required for topology 'global'"
                )
        else:
            _validate_length(
                "coordination_cutoff", self.coordination_cutoff, allow_zero=False
            )
        _validate_length("skin", self.skin, allow_zero=True)

    @property
    def graph_cutoff(self) -> float:
        """Exact real-space search radius a neighbor list must cover."""

        return max(self.pair_cutoff or 0.0, self.coordination_cutoff or 0.0)


@dataclass(frozen=True)
class D2Correction:
    """Configuration for a DFT-D2 pair correction; lengths are Angstrom."""

    functional: str = "pbe"
    cutoff: float = 40.0 * units.Bohr
    skin: float = 0.5

    def __post_init__(self) -> None:
        _validate_functional(self.functional)
        _validate_length("cutoff", self.cutoff, allow_zero=False)
        _validate_length("skin", self.skin, allow_zero=True)


@dataclass(frozen=True)
class D3Correction:
    """Configuration for a two-body DFT-D3 correction; lengths are Angstrom."""

    functional: str = "pbe"
    damping: Literal["zero", "bj", "zerom", "bjm"] = "bj"
    cutoff: float = 40.0 * units.Bohr
    coordination_cutoff: float = 40.0 * units.Bohr
    skin: float = 0.5

    def __post_init__(self) -> None:
        _validate_functional(self.functional)
        if self.damping not in _D3_DAMPING:
            raise ValueError("damping must be one of 'zero', 'bj', 'zerom', or 'bjm'")
        _validate_length("cutoff", self.cutoff, allow_zero=False)
        _validate_length(
            "coordination_cutoff", self.coordination_cutoff, allow_zero=False
        )
        _validate_length("skin", self.skin, allow_zero=True)


def _validate_functional(functional: str) -> None:
    if not isinstance(functional, str) or not functional.strip():
        raise ValueError("functional must be a non-empty string")


# FourierD3 publishes only its immutable configuration. Its factory, runtime, and
# dtype-specific native evaluators remain implementation details.

# B-spline assignment orders the mesh path supports, inclusive (specification 5.6).
_FOURIER_D3_SPLINE_ORDER_MIN = 2
_FOURIER_D3_SPLINE_ORDER_MAX = 6
_FOURIER_D3_DEFAULT_SPLINE_ORDER = 4

# Neutral mesh-spacing default for a configuration that names no mesh policy, applied in
# ``FourierD3Correction.__post_init__`` so that a constructed configuration always carries
# exactly one effective mesh policy. The specification and the NVIDIA v0.5.0 toolkit both
# refuse to guess a mesh ("there is no accuracy-based default for FourierD3"), so 1.0
# Angstrom is a documented neutral default rather than a converged value: a caller that
# needs accuracy must select a mesh explicitly.
_FOURIER_D3_DEFAULT_MESH_SPACING = 1.0

# One chunk for every rank the canonical D3 tables can produce: the decomposition rank is
# bounded by n_species * n_ref, at most d3_max_atomic_number (94) times
# d3_max_coordination_references (5) = 470.
_FOURIER_D3_DEFAULT_CHUNK_SIZE = 512

# Mesh extents, stride products, and the flattened point index are signed 64-bit
# quantities, so a larger axis or point count could not be indexed safely. Inputs beyond
# these bounds are rejected rather than clamped: a clamped mesh would silently change the
# accuracy control the caller asked for.
_FOURIER_D3_MAX_MESH_AXIS = (1 << 63) - 1
_FOURIER_D3_MAX_MESH_VOLUME = (1 << 63) - 1


def _radix_friendly_ceiling(target: int) -> int:
    """Least integer at or above ``target`` whose prime factors are only 2, 3, 5, and 7.

    Those are the radices the FFT radix kernels handle directly (specification section
    5.6); a size outside that set forces Bluestein's algorithm, so a spacing-derived size
    is rounded up to the next size these factors can produce.

    The next power of two is itself smooth, so it bounds the answer from above.
    Enumerating the products of two, three, and five below that bound, and for each one
    the smallest power of seven that reaches ``target``, visits a number of candidates
    proportional to the digits of ``target`` rather than to the gap between consecutive
    smooth integers — a gap already 3.8e10 wide above 2**50, which a candidate-by-candidate
    walk cannot cross. Follows the algorithm of NVIDIA ALCHEMI Toolkit-Ops
    ``nvalchemiops/interactions/electrostatics/_pme_mesh.py::_next_fft_dimension``
    (Apache-2.0), reimplemented here so the installed package keeps no NVIDIA runtime
    dependency.
    """

    target = max(int(target), 1)
    result = 1 << (target - 1).bit_length()
    power_two = 1
    while power_two < result:
        power_three = power_two
        while power_three < result:
            power_five = power_three
            while power_five < result:
                candidate = power_five
                while candidate < target:
                    candidate *= 7
                if candidate < result:
                    result = candidate
                power_five *= 5
            power_three *= 3
        power_two *= 2
    return result


def _supported_mesh(dimensions, *, context: str) -> tuple[int, int, int]:
    """Return ``dimensions`` after checking the 64-bit-safe mesh bounds.

    ``context`` names the mesh policy the dimensions came from, so the error carries the
    requested spacing or mesh alongside the offending extent.
    """

    for axis, size in enumerate(dimensions):
        if size > _FOURIER_D3_MAX_MESH_AXIS:
            raise ValueError(
                f"FourierD3 {context} has {size} nodes along axis {axis}, beyond the "
                f"supported {_FOURIER_D3_MAX_MESH_AXIS} nodes per axis"
            )
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    if volume > _FOURIER_D3_MAX_MESH_VOLUME:
        raise ValueError(
            f"FourierD3 {context} has {volume} points, beyond the supported 64-bit-safe "
            f"volume {_FOURIER_D3_MAX_MESH_VOLUME}"
        )
    return dimensions


def _spacing_mesh_dimensions(cell, spacing, spline_order) -> tuple[int, int, int]:
    """Mesh dimensions a spacing policy resolves to for ``cell``.

    Specification section 5.6: round each axis up to the smallest size whose prime
    factors are 2, 3, 5, and 7, and never below the spline width, because a shorter axis
    wraps two stencil nodes onto the same point. The reciprocal cell is measured on the
    lattice vectors, which are the rows of the cell matrix.

    Every length, quotient, and node count is computed in Python floats and checked before
    it is narrowed to an integer, so a denormal spacing or an enormous cell reports a
    mesh-policy error instead of a NumPy overflow warning and a raw ``OverflowError``.

    Raises
    ------
    ValueError
        If a cell length, the required node count, or the resolved mesh is not finite or
        not within the supported 64-bit-safe mesh bounds.
    """

    spacing = float(spacing)
    cell_array = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    lengths = [math.hypot(*(float(value) for value in row)) for row in cell_array]
    for axis, length in enumerate(lengths):
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(
                f"FourierD3 mesh spacing {spacing!r} needs finite positive cell lengths; "
                f"axis {axis} has length {length!r} for cell lengths {lengths}"
            )
    context = f"mesh spacing {spacing!r} for cell lengths {lengths}"
    dimensions = []
    for axis, length in enumerate(lengths):
        quotient = length / spacing
        if not math.isfinite(quotient):
            raise ValueError(
                f"FourierD3 {context} needs a node count along axis {axis} that is not "
                "finite; use a larger mesh_spacing or a smaller cell"
            )
        required = max(math.ceil(quotient), int(spline_order))
        if required > _FOURIER_D3_MAX_MESH_AXIS:
            raise ValueError(
                f"FourierD3 {context} requires {required} nodes along axis {axis}, "
                f"beyond the supported {_FOURIER_D3_MAX_MESH_AXIS} nodes per axis"
            )
        dimensions.append(_radix_friendly_ceiling(required))
    return _supported_mesh(tuple(dimensions), context=context)


def _validate_positive_integer(name: str, value: object) -> None:
    label = name.replace("_", " ")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _validate_mesh_shape(mesh: object) -> None:
    if (
        not isinstance(mesh, tuple)
        or len(mesh) != 3
        or any(
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or size < 1
            for size in mesh
        )
    ):
        raise ValueError("mesh must be a tuple of three positive integers")


def _validate_spline_order(spline_order: object) -> None:
    low, high = _FOURIER_D3_SPLINE_ORDER_MIN, _FOURIER_D3_SPLINE_ORDER_MAX
    if (
        isinstance(spline_order, bool)
        or not isinstance(spline_order, (int, np.integer))
        or not low <= spline_order <= high
    ):
        raise ValueError(f"spline order must be an integer between {low} and {high}")


@dataclass(frozen=True)
class FourierD3Correction:
    """Configuration for the reciprocal-space (FourierD3) DFT-D3(BJ) correction.

    Lengths are Angstrom. Damping is Becke-Johnson only and is not selectable:
    the reciprocal-space model has no
    zero-damping variant and no real-space/reciprocal split.

    The mesh is selected by exactly one effective policy. ``mesh`` and ``mesh_spacing``
    are mutually exclusive explicit inputs, and ``mesh`` is a three-tuple of positive
    integers so the frozen configuration stays hashable. Naming neither applies the
    automatic spacing policy with the neutral ``mesh_spacing`` default of 1.0 Angstrom, so
    a constructed configuration always carries exactly one policy: ``mesh_spacing`` is
    ``None`` only when an explicit ``mesh`` was given.

    ``rank_cap`` optionally bounds the C6 decomposition rank and ``chunk_size`` bounds how
    many rank slots one reciprocal pass processes; the default chunk size is a single
    chunk for every rank the canonical tables can produce.
    """

    functional: str = "pbe"
    coordination_cutoff: float = 40.0 * units.Bohr
    skin: float = 0.5
    mesh: tuple[int, int, int] | None = None
    mesh_spacing: float | None = None
    spline_order: int = _FOURIER_D3_DEFAULT_SPLINE_ORDER
    decomposition_tolerance: float = 1e-4
    rank_cap: int | None = None
    chunk_size: int = _FOURIER_D3_DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        _validate_functional(self.functional)
        _validate_length(
            "coordination_cutoff", self.coordination_cutoff, allow_zero=False
        )
        _validate_length("skin", self.skin, allow_zero=True)
        if self.mesh is not None and self.mesh_spacing is not None:
            raise ValueError(
                "mesh and mesh_spacing are mutually exclusive; provide an explicit mesh "
                "or an explicit mesh_spacing, and naming neither selects the automatic "
                f"{_FOURIER_D3_DEFAULT_MESH_SPACING} Angstrom spacing"
            )
        if self.mesh is None and self.mesh_spacing is None:
            # Frozen dataclass: the resolved policy must be installed through the base
            # setter. The value is hashable and validated below like any other spacing.
            object.__setattr__(self, "mesh_spacing", _FOURIER_D3_DEFAULT_MESH_SPACING)
        if self.mesh is not None:
            _validate_mesh_shape(self.mesh)
            _supported_mesh(self.mesh, context="explicit mesh")
        if self.mesh_spacing is not None:
            _validate_length("mesh_spacing", self.mesh_spacing, allow_zero=False)
        _validate_spline_order(self.spline_order)
        _validate_length(
            "decomposition_tolerance", self.decomposition_tolerance, allow_zero=False
        )
        if self.rank_cap is not None:
            _validate_positive_integer("rank_cap", self.rank_cap)
        _validate_positive_integer("chunk_size", self.chunk_size)


class CorrectionRuntime(Protocol):
    """Runtime contract implemented by concrete native corrections."""

    properties: frozenset[str]
    graph_requirement: GraphRequirement

    def calculate(
        self, atoms: Atoms, properties: Sequence[str]
    ) -> Mapping[str, float | np.ndarray]: ...

    def diagnostics(self) -> Mapping[str, object]: ...


def _build_d2_runtime(
    configuration: D2Correction, *, native: object, dtype: object
) -> CorrectionRuntime:
    class_name = "D2KokkosFloat" if _is_float32(dtype) else "D2Kokkos"
    try:
        evaluator_class = getattr(native, class_name)
    except AttributeError as error:
        raise RuntimeError(f"{class_name} native evaluator is not available") from error
    return _NativePairCorrectionRuntime(
        method="d2",
        evaluator=evaluator_class(configuration.functional, configuration.cutoff),
        graph_requirement=GraphRequirement(
            pair_cutoff=configuration.cutoff,
            coordination_cutoff=None,
            skin=configuration.skin,
        ),
        native=native,
    )


def _build_d3_runtime(
    configuration: D3Correction, *, native: object, dtype: object
) -> CorrectionRuntime:
    class_name = "D3KokkosFloat" if _is_float32(dtype) else "D3Kokkos"
    try:
        evaluator_class = getattr(native, class_name)
    except AttributeError as error:
        raise RuntimeError(f"{class_name} native evaluator is not available") from error
    return _NativePairCorrectionRuntime(
        method="d3",
        evaluator=evaluator_class(
            configuration.damping,
            configuration.functional,
            configuration.cutoff,
            configuration.coordination_cutoff,
        ),
        graph_requirement=GraphRequirement(
            pair_cutoff=configuration.cutoff,
            coordination_cutoff=configuration.coordination_cutoff,
            skin=configuration.skin,
        ),
        native=native,
    )


def _build_fourier_d3_runtime(
    configuration: FourierD3Correction, *, native: object, dtype: object
) -> CorrectionRuntime:
    """Build the reciprocal-space runtime without allocating its native evaluator.

    The dtype-specific evaluator class is resolved here so that a build without it fails
    where every other correction fails — at runtime construction — and never resolves to
    ``D3Kokkos``: the reciprocal model shares neither the modified coordination-number
    function nor the energy expression of real-space D3, so substituting it would be a
    silent accuracy change. The evaluator instance itself is allocated on the first
    evaluation, after the geometry gate, because the lattice sum is only defined for a
    fully periodic, nonsingular cell.
    """

    class_name = "FourierD3KokkosFloat" if _is_float32(dtype) else "FourierD3Kokkos"
    try:
        evaluator_class = getattr(native, class_name)
    except AttributeError as error:
        raise RuntimeError(
            f"{class_name} native evaluator is not available in this build; the "
            "reciprocal-space runtime has no real-space D3 fallback, and real-space D3 "
            "is a different model"
        ) from error
    return _NativeReciprocalCorrectionRuntime(
        configuration=configuration,
        evaluator_class=evaluator_class,
        graph_requirement=GraphRequirement(
            pair_cutoff=None,
            coordination_cutoff=configuration.coordination_cutoff,
            skin=configuration.skin,
            topology="global",
        ),
        native=native,
        precision=np.dtype(dtype).name,
    )


def _is_float32(dtype: object) -> bool:
    try:
        return np.dtype(dtype) == np.dtype(np.float32)
    except TypeError as error:
        raise ValueError(f"unsupported correction dtype {dtype!r}") from error


class _NativeGraphCorrectionRuntime:
    """Own one native evaluator and its independent Verlet graph.

    A runtime owns its graph: the topology reference, the half-skin reuse policy, the
    host/device build policy, and the counters diagnostics report. Nothing here is shared
    with the neural-model graph, and a device build failure is never resolved by building
    the same topology on the host.
    """

    properties = _CORRECTION_PROPERTIES

    def __init__(self, *, method, evaluator, graph_requirement, native):
        self.method = method
        # ``None`` marks a runtime that allocates its evaluator on the first evaluation;
        # see ``_allocated_evaluator``.
        self.evaluator = evaluator
        self.graph_requirement = graph_requirement
        self._native = native
        self._topology_reference = None
        self._graph_build_count = 0
        self._graph_reuse_count = 0
        self._geometry_update_count = 0
        # Device graph preparation is fail-fast: this runtime never catches a
        # Kokkos build error and silently substitutes a host-built topology.
        self._backend_fallback_count = 0
        # Attempted work that did not commit.  A refusal is reported here and nowhere
        # else, so no other counter can be mistaken for work that succeeded.
        self._failed_preparation_count = 0
        self._failed_compute_count = 0

    @property
    def _exact_cutoff(self) -> float:
        return self.graph_requirement.graph_cutoff

    def _check_geometry(self, atoms) -> None:
        """Hook: reject geometry this runtime cannot describe, before any allocation."""

    def _allocate_evaluator(self, atoms):
        """Allocate the native evaluator for a runtime that defers allocation."""

        raise RuntimeError(  # pragma: no cover - every deferring runtime overrides it
            f"{type(self).__name__} has no native evaluator to allocate"
        )

    def _allocated_evaluator(self, atoms):
        """Return the native evaluator, allocating it on first use when deferred."""

        if self.evaluator is None:
            self._check_geometry(atoms)
            self.evaluator = self._allocate_evaluator(atoms)
        return self.evaluator

    def _resolve_domains(self, numbers, cell):
        """Hook: validate and compute family state for a geometry, before any allocation.

        Runs before the native evaluator exists and before the graph, the topology
        reference, or any counter changes, so a policy this geometry cannot support fails
        on a runtime that has done nothing. The returned value is opaque here and is handed
        back to :meth:`_commit_domains` only once preparation has succeeded.
        """

    def _commit_domains(self, prospective) -> None:
        """Hook: install the state :meth:`_resolve_domains` computed."""

    def _load_domains(self, evaluator, prospective) -> None:
        """Hook: push family state onto the evaluator, before its graph.

        Runs on every preparation, including the first one that allocates the evaluator,
        and only for families whose evaluator takes more than its resolved constructor
        fields. A family whose payload is unchanged pushes it again and the evaluator
        compares and skips it, so this is idempotent rather than conditional here. It
        must leave the evaluator unchanged when it refuses.
        """

    @staticmethod
    def _geometry(atoms):
        numbers = np.ascontiguousarray(atoms.numbers, dtype=np.int64)
        positions = np.ascontiguousarray(atoms.positions, dtype=np.float64)
        cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
        pbc = np.ascontiguousarray(atoms.pbc, dtype=bool)
        fully_periodic = bool(np.all(pbc))
        try:
            native_cell = cell
            inverse_cell = np.ascontiguousarray(np.linalg.inv(native_cell))
        except np.linalg.LinAlgError:
            fully_periodic = False
            native_cell = np.ascontiguousarray(atoms.cell.complete(), dtype=np.float64)
            inverse_cell = np.ascontiguousarray(np.linalg.inv(native_cell))
        return (
            numbers,
            positions,
            cell,
            native_cell,
            pbc,
            inverse_cell,
            fully_periodic,
        )

    def _prepare_or_update(self, atoms) -> None:
        self._check_geometry(atoms)
        (
            numbers,
            positions,
            cell,
            native_cell,
            pbc,
            inverse_cell,
            fully_periodic,
        ) = self._geometry(atoms)
        prospective = self._resolve_domains(numbers, cell)
        evaluator = self._allocated_evaluator(atoms)
        self._load_domains(evaluator, prospective)
        reuse = None
        if self._topology_reference is not None and self.graph_requirement.skin > 0.0:
            reuse = neighbor_topology_reuse(
                reference=self._topology_reference,
                atomic_numbers=numbers,
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self._exact_cutoff,
                skin=self.graph_requirement.skin,
            )
        if reuse is not None and reuse.reusable:
            evaluator.update_geometry(positions, native_cell, inverse_cell)
            if reuse.cell_changed:
                self._topology_reference = NeighborTopologyReference(
                    atomic_numbers=self._topology_reference.atomic_numbers,
                    pbc=self._topology_reference.pbc,
                    topology_cell=self._topology_reference.topology_cell,
                    topology_reference_positions=(
                        self._topology_reference.topology_reference_positions
                    ),
                    topology_fractional_positions=(
                        self._topology_reference.topology_fractional_positions
                    ),
                    cell=np.array(cell, copy=True),
                    reference_positions=np.array(
                        reuse.geometry_reference_positions, copy=True
                    ),
                )
            self._graph_reuse_count += 1
            self._geometry_update_count += 1
            self._commit_domains(prospective)
            return

        build_cutoff = self._exact_cutoff + self.graph_requirement.skin
        from .calculator import _neighbor_backend_request, _use_device_neighbor_graph

        request = _neighbor_backend_request()
        execution_space = getattr(
            self._native, "_kokkos_default_execution_space", lambda: None
        )()
        use_device_graph = _use_device_neighbor_graph(
            request, fully_periodic, len(numbers), execution_space
        )
        if use_device_graph:
            evaluator.prepare_periodic_graph(
                numbers, positions, native_cell, inverse_cell, build_cutoff
            )
        else:
            graph = build_host_neighbor_graph(atoms, build_cutoff)
            receiver_offsets = np.empty(len(numbers) + 1, dtype=np.int64)
            receiver_offsets[0] = 0
            np.cumsum(graph.num_neigh, out=receiver_offsets[1:])
            evaluator.prepare_host_graph(
                numbers,
                receiver_offsets,
                graph.sources,
                graph.shifts,
                positions,
                native_cell,
                build_cutoff,
            )
        topology_fractional_positions = (
            np.ascontiguousarray(positions @ inverse_cell)
            if fully_periodic
            else np.empty((0, 3), dtype=np.float64)
        )
        self._topology_reference = NeighborTopologyReference(
            atomic_numbers=np.array(numbers, copy=True),
            pbc=np.array(pbc, copy=True),
            topology_cell=np.array(cell, copy=True),
            topology_reference_positions=np.array(positions, copy=True),
            topology_fractional_positions=topology_fractional_positions,
            cell=np.array(cell, copy=True),
            reference_positions=np.array(positions, copy=True),
        )
        self._graph_build_count += 1
        self._geometry_update_count += 1
        self._commit_domains(prospective)

    def calculate(self, atoms, properties):
        requested = tuple(dict.fromkeys(properties))
        unsupported = set(requested) - self.properties
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise RuntimeError(
                f"native {self.method} correction does not support {names}"
            )
        # An unsupported property is a caller error, not attempted work, so it is not
        # counted; a refused preparation or compute is, and each is named for the stage
        # that refused rather than folded into one "failure" total.
        try:
            self._prepare_or_update(atoms)
        except Exception:
            self._failed_preparation_count += 1
            raise
        try:
            self.evaluator.compute("forces" in requested, "stress" in requested)
        except Exception:
            self._failed_compute_count += 1
            raise
        native_properties = tuple(
            "atomic_energies" if name == "energies" else name for name in requested
        )
        native_result = self.evaluator.result(native_properties)
        result = {}
        if "energy" in requested:
            result["energy"] = float(native_result["energy"])
        if "energies" in requested:
            result["energies"] = np.asarray(
                native_result["atomic_energies"], dtype=np.float64
            )
        if "forces" in requested:
            result["forces"] = np.asarray(native_result["forces"], dtype=np.float64)
        if "stress" in requested:
            result["stress"] = np.asarray(native_result["stress"], dtype=np.float64)
        return result

    # A runtime with no evaluator yet reports the graph it has not built rather than
    # omitting the fields, so every consumer reads one shape.
    _EMPTY_GRAPH_REPORT = {
        "graph_backend": None,
        "num_edges": 0,
        "num_nodes": 0,
        "generation": 0,
        "build_cutoff": 0.0,
    }

    def _graph_report(self):
        if self.evaluator is None:
            return dict(self._EMPTY_GRAPH_REPORT)
        return self.evaluator.graph_diagnostics()

    def diagnostics(self):
        native = self._graph_report()
        return {
            "method": self.method,
            "graph_backend": native["graph_backend"],
            "graph_nodes": native["num_nodes"],
            "graph_generation": native["generation"],
            # The exact radius the search must cover, and the radius the built graph
            # actually covers: the two differ by the skin, and a consumer needs both.
            "graph_build_cutoff": native["build_cutoff"],
            "graph_build_count": self._graph_build_count,
            "graph_reuse_count": self._graph_reuse_count,
            # Successful preparations, including first builds and topology reuses.  This
            # is deliberately broader than the native evaluator's update-only counter.
            "geometry_update_count": self._geometry_update_count,
            "backend_fallback_count": self._backend_fallback_count,
            "num_edges": native["num_edges"],
            "pair_cutoff": self.graph_requirement.pair_cutoff,
            "coordination_cutoff": self.graph_requirement.coordination_cutoff,
            "skin": self.graph_requirement.skin,
            "failed_preparation_count": self._failed_preparation_count,
            "failed_compute_count": self._failed_compute_count,
        }


class _NativePairCorrectionRuntime(_NativeGraphCorrectionRuntime):
    """Own one eagerly allocated native pair evaluator and its Verlet graph."""

    def diagnostics(self):
        result = dict(super().diagnostics())
        if self.method == "d3":
            native = self.evaluator.graph_diagnostics()
            result.update(
                {
                    name: native[name]
                    for name in (
                        "reference_weight_active_atoms",
                        "reference_weight_capacity_atoms",
                        "reference_weight_workspace_bytes",
                        "coordination_cache_active_edges",
                        "coordination_cache_capacity_edges",
                        "coordination_cache_workspace_bytes",
                        "d3_workspace_bytes",
                    )
                }
            )
        return result


@dataclass(frozen=True)
class _ReciprocalDomainKeys:
    """Invalidation keys one geometry resolves to, before any of them is committed."""

    decomposition: tuple
    reciprocal_kernel: tuple
    mesh: tuple[int, int, int]
    fft_plan: tuple


class _NativeReciprocalCorrectionRuntime(_NativeGraphCorrectionRuntime):
    """Own the private FourierD3 runtime: one short graph, a lazy evaluator, five keys.

    The reciprocal model is a global interaction, so the only real-space radius it
    declares is its short modified-coordination graph at ``coordination_cutoff + skin``.
    It has no long-range pair cutoff and never reuses the neural-model graph; the half-skin
    topology rules, receiver-major edge order, and host/device build policy are the
    repository's shared ones.

    Native allocation is deferred to the first evaluation and is gated on a fully
    periodic, finite, nonsingular cell with a well-defined inverse: the factory never
    constructs an evaluator, so an unusable cell fails before any native object exists.
    The mesh policy is resolved and bounds-checked before that allocation and before the
    graph, the topology reference, or any counter changes, and the resulting keys are
    installed only after preparation succeeds, so a rejected geometry or a failed
    preparation leaves the runtime exactly as it was.

    Five invalidation domains are tracked separately, because a single "geometry changed"
    flag would rebuild far more than necessary and would equate caches with different
    keys:

    ``coordination graph``
        Topology reference and half-skin reuse, with the build/reuse counters below.
    ``active-species decomposition``
        :attr:`decomposition_key`: the ascending active atomic numbers, the requested
        decomposition tolerance, and the rank cap. Motion, cell changes, and mesh changes
        leave it untouched.
    ``reciprocal kernel``
        :attr:`reciprocal_kernel_key`: the cell, the active species, and the spline order.
        Any cell change invalidates it, including a safe affine change that reuses the
        short graph.
    ``resolved mesh``
        :attr:`resolved_mesh`: the explicit mesh, or the dimensions the spacing policy
        resolves to for the current cell. A spacing-derived mesh invalidates only when the
        resolved dimensions actually change; an explicit mesh never depends on the cell.
    ``FFT plan``
        :attr:`fft_plan_key`: the resolved mesh and the working precision. It follows that
        key alone rather than every cell or graph change; native storage extends it with
        the FFT provider, layout, and batch, which is why a cell change with an unchanged
        mesh reuses the plan.

    Native storage attaches payloads to these domains — the decomposition factors, the
    reciprocal kernel tables, the mesh workspaces, and the FFT plan — and increments the
    same counters when they rebuild. The keys and counters are recorded here so that the
    lifecycle is observable and testable independently of those payloads; this runtime does
    not evaluate a reciprocal sum itself.
    """

    def __init__(
        self, *, configuration, evaluator_class, graph_requirement, native, precision
    ):
        super().__init__(
            method="fourier_d3",
            evaluator=None,
            graph_requirement=graph_requirement,
            native=native,
        )
        self.configuration = configuration
        self._evaluator_class = evaluator_class
        self._precision = precision
        self._native_allocation_count = 0
        self._tables = None
        self._decomposition_key = None
        self._reciprocal_kernel_key = None
        self._resolved_mesh = None
        self._fft_plan_key = None
        self._decomposition_invalidation_count = 0
        self._decomposition_rank_capped = False
        self._prospective_decomposition_rank_capped = None
        self._reciprocal_kernel_invalidation_count = 0
        self._mesh_invalidation_count = 0
        self._fft_plan_invalidation_count = 0
        # Stage timing is opt-in and off by default, and it is remembered here so a
        # runtime whose evaluator is allocated later still opts in.
        self._timing_enabled = False

    @property
    def decomposition_key(self):
        """Key of the active-species decomposition domain."""

        return self._decomposition_key

    @property
    def reciprocal_kernel_key(self):
        """Key of the cell- and species-dependent reciprocal kernel domain."""

        return self._reciprocal_kernel_key

    @property
    def resolved_mesh(self) -> tuple[int, int, int] | None:
        """Mesh dimensions of the last prepared geometry, or ``None`` before then."""

        return self._resolved_mesh

    @property
    def fft_plan_key(self):
        """Key of the FFT plan domain."""

        return self._fft_plan_key

    def _check_geometry(self, atoms) -> None:
        """Reject geometry the reciprocal model cannot describe.

        The lattice sum is only defined for a fully periodic, invertible cell, and there
        is no real-space fallback to substitute, so this runs before any native
        allocation. A cell that ASE would happily complete to a molecule is rejected
        rather than completed.
        """

        pbc = np.asarray(atoms.pbc, dtype=bool)
        if pbc.shape != (3,) or not bool(np.all(pbc)):
            raise RuntimeError(
                "FourierD3 requires a fully periodic cell with pbc=[True, True, True]; "
                f"got pbc={pbc.tolist()}. Open and partially periodic systems are outside "
                "the reciprocal-space model, and real-space D3 is a different model that "
                "is never substituted for it."
            )
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if cell.shape != (3, 3) or not bool(np.all(np.isfinite(cell))):
            raise RuntimeError(
                "FourierD3 requires a finite 3x3 cell; got "
                f"{cell.tolist()}. The reciprocal vectors and the lattice sum are only "
                "defined for a nonsingular cell."
            )
        try:
            # The determinant of a very large but finite cell overflows to infinity and
            # would emit a NumPy warning, so singularity is read from the scaled
            # log-determinant, whose sign is exactly zero for a singular cell.
            sign, _ = np.linalg.slogdet(cell)
            if sign == 0.0:
                raise np.linalg.LinAlgError("singular cell")
            inverse = np.linalg.inv(cell)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                f"FourierD3 requires a nonsingular cell with a well-defined inverse; got "
                f"{cell.tolist()}."
            ) from error
        if not bool(np.all(np.isfinite(inverse))):
            raise RuntimeError(
                f"FourierD3 requires a nonsingular cell with a well-defined inverse; got "
                f"{cell.tolist()}."
            )

    def _allocate_evaluator(self, atoms):
        """Allocate the dtype-specific native evaluator for a validated cell.

        The canonical D3 tables and their provenance fingerprint are read from the
        installed native accessor - never by parsing a source file at runtime - and the
        evaluator is built from the resolved configuration fields.  The geometry gate
        runs again here so that this method can never allocate for a cell the model
        cannot describe, whatever order a caller reaches it in.
        """

        self._check_geometry(atoms)
        tables = self._native._fourier_d3_canonical_tables()
        self._tables = tables
        configuration = self.configuration
        evaluator = self._evaluator_class(
            configuration.functional,
            float(configuration.coordination_cutoff),
            int(configuration.spline_order),
            int(configuration.chunk_size),
            tables["table_fingerprint"],
            tables["d3_rcov"],
            tables["d3_r2r4"],
        )
        evaluator.set_timing_enabled(self._timing_enabled)
        self._native_allocation_count += 1
        return evaluator

    def _decomposition_payload(self, species, tolerance, rank_cap):
        """The canonical decomposition payload for the active species.

        An empty fully periodic system has no active species and no rank to retain; its
        dispersion energy is exactly zero, and the native evaluator publishes that
        without a reciprocal pass, so the empty payload is stated explicitly rather than
        left for the native side to guess.
        """

        if not species:
            return (
                (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float64),
                    np.empty((0, 0, 0), dtype=np.float64),
                    np.empty((0, 0), dtype=np.float64),
                    np.empty((0, 0), dtype=bool),
                    0.0,
                    True,
                ),
                False,
            )
        tables = self._tables
        factors = decompose_d3_c6_reference(
            tables["d3_c6"],
            species,
            table_fingerprint=tables["table_fingerprint"],
            tolerance=tolerance,
            rank_cap=rank_cap,
        )
        return (
            (
                factors.species,
                factors.eigs,
                factors.v_q,
                factors.cn_ref,
                factors.valid,
                factors.max_relative_error,
                factors.tolerance_met,
            ),
            factors.rank_capped,
        )

    def _load_domains(self, evaluator, prospective) -> None:
        """Push the resolved mesh and the active-species decomposition payload.

        Both follow from the keys :meth:`_resolve_domains` already computed, so nothing is
        decided twice.  The native side compares an identical payload and skips it, so a
        geometry that keeps its species reloads nothing.

        Both are prospective: the native evaluator commits them only when the graph they
        belong to has been prepared, so a refused graph build - or a refused reservation -
        leaves the native domains, their generations, and the published result exactly as
        this runtime's committed keys describe them.  There is no rollback here because
        there is nothing to roll back.
        """

        evaluator.set_mesh(prospective.mesh)
        species, tolerance, rank_cap = prospective.decomposition
        payload, rank_capped = self._decomposition_payload(species, tolerance, rank_cap)
        evaluator.set_decomposition(*payload)
        self._prospective_decomposition_rank_capped = (
            prospective.decomposition,
            rank_capped,
        )

    def _mesh_dimensions(self, cell) -> tuple[int, int, int]:
        """Mesh dimensions for a cell under this configuration's policy."""

        if self.configuration.mesh is not None:
            return _supported_mesh(
                self.configuration.mesh,
                context=f"explicit mesh {self.configuration.mesh}",
            )
        return _spacing_mesh_dimensions(
            cell, self.configuration.mesh_spacing, self.configuration.spline_order
        )

    @staticmethod
    def _active_species(numbers) -> tuple[int, ...]:
        """Ascending unique nonzero atomic numbers; padding atoms do not count."""

        numbers = np.asarray(numbers, dtype=np.int64)
        return tuple(int(value) for value in np.unique(numbers[numbers != 0]))

    def _resolve_domains(self, numbers, cell):
        """Validate the prospective mesh and compute the four keys without committing.

        The mesh policy is resolved and bounds-checked here, before the evaluator exists
        and before any graph, topology, or counter change, so a cell this configuration
        cannot size a mesh for leaves the runtime exactly as it was. The keys are returned
        to the caller and installed only after preparation has succeeded.
        """

        species = self._active_species(numbers)
        cell_array = np.asarray(cell, dtype=np.float64)
        mesh = self._mesh_dimensions(cell_array)
        return _ReciprocalDomainKeys(
            decomposition=(
                species,
                float(self.configuration.decomposition_tolerance),
                self.configuration.rank_cap,
            ),
            reciprocal_kernel=(
                tuple(float(value) for value in cell_array.reshape(-1)),
                species,
                int(self.configuration.spline_order),
            ),
            mesh=mesh,
            fft_plan=(mesh, self._precision),
        )

    def _commit_domains(self, prospective) -> None:
        """Install the keys for a geometry whose preparation succeeded."""

        if prospective is None:  # pragma: no cover - always supplied by this runtime
            return
        if prospective.decomposition != self._decomposition_key:
            cap_state = self._prospective_decomposition_rank_capped
            if cap_state is None or cap_state[0] != prospective.decomposition:
                raise RuntimeError(
                    "FourierD3 decomposition metadata was not staged for commit"
                )
            self._decomposition_key = prospective.decomposition
            self._decomposition_rank_capped = cap_state[1]
            self._decomposition_invalidation_count += 1
        if prospective.reciprocal_kernel != self._reciprocal_kernel_key:
            self._reciprocal_kernel_key = prospective.reciprocal_kernel
            self._reciprocal_kernel_invalidation_count += 1
        if prospective.mesh != self._resolved_mesh:
            self._resolved_mesh = prospective.mesh
            self._mesh_invalidation_count += 1
        if prospective.fft_plan != self._fft_plan_key:
            self._fft_plan_key = prospective.fft_plan
            self._fft_plan_invalidation_count += 1
        self._prospective_decomposition_rank_capped = None

    def enable_stage_timing(self, enabled: bool) -> None:
        """Opt into synchronized stage timing, now or at the next allocation.

        The native evaluator propagates the flag to every stage it composes, so one
        opt-in covers the whole evaluation.  Disabling it again stops the record
        rather than leaving the last measurement in place.
        """

        self._timing_enabled = bool(enabled)
        if self.evaluator is not None:
            self.evaluator.set_timing_enabled(self._timing_enabled)

    def diagnostics(self):
        """One record describing capability, realized accuracy, caches, and timing.

        The runtime's own domain keys and counters are reported here, and the
        evaluator's realized state is merged under names that do not collide with
        them: a caller never has to choose between two values for one name.  The
        evaluator's own invalidation counters stay in ``evaluator.diagnostics()``,
        because the domains this runtime invalidates are the domains this runtime
        keys; the two agree in normal operation but only one is reported here.
        """

        result = dict(super().diagnostics())
        evaluator = self.evaluator
        native = None if evaluator is None else evaluator.diagnostics()
        configuration = self.configuration
        capability = fourier_d3_capability(self._native)
        explicit_mesh = configuration.mesh is not None
        result.update(
            {
                "topology": self.graph_requirement.topology,
                # A global FourierD3 graph has no pair radius, so its widest declared
                # radius is exactly the coordination cutoff; both names remain because
                # one is the generic graph contract and one is the model control.
                "graph_cutoff": self.graph_requirement.graph_cutoff,
                "native_allocation_count": self._native_allocation_count,
                "decomposition_invalidation_count": (
                    self._decomposition_invalidation_count
                ),
                "reciprocal_kernel_invalidation_count": (
                    self._reciprocal_kernel_invalidation_count
                ),
                "resolved_mesh": self._resolved_mesh,
                "mesh_invalidation_count": self._mesh_invalidation_count,
                "fft_plan_invalidation_count": self._fft_plan_invalidation_count,
                "mesh_policy": "explicit" if explicit_mesh else "spacing",
                "mesh_spacing_requested": (
                    None if explicit_mesh else float(configuration.mesh_spacing)
                ),
                "spline_order": int(configuration.spline_order),
                "decomposition_tolerance": float(configuration.decomposition_tolerance),
                "rank_cap": configuration.rank_cap,
                "timing_enabled": self._timing_enabled,
                "capability_available": capability.available,
                "capability_provider": capability.provider,
                "capability_provider_variant": capability.provider_variant,
                "capability_kokkosfft_version": capability.kokkosfft_version,
                "capability_kokkos_version": capability.kokkos_version,
                "capability_execution_backend": capability.execution_backend,
                "capability_reason": capability.reason,
            }
        )
        result.update(self._native_diagnostics(native))
        # This is committed with the decomposition key rather than reconstructed from
        # its rank: a cap equal to the full rank did not truncate the decomposition.
        result["decomposition_rank_capped"] = self._decomposition_rank_capped
        return result

    # The realized evaluator state, under the names this record publishes.  The
    # mapping is explicit so a new native field cannot silently shadow a runtime
    # counter, and so that a name here is exactly a name a consumer can rely on.
    _NATIVE_DIAGNOSTIC_KEYS = (
        "precision",
        "active_species",
        "table_fingerprint",
        "retained_rank",
        "rank_total",
        "rank_chunk",
        "rank_chunk_count",
        "n_references",
        "decomposition_max_relative_error",
        "decomposition_tolerance_met",
        "mesh_spacing",
        "mesh_points",
        "plan_builds",
        "plan_reuses",
        "plan_releases",
        "plan_key_rebuilds",
        "kernel_key_hits",
        "kernel_key_misses",
        "kernel_key_rebuilds",
        "stencil_points",
        "input_generation",
        "published_generation",
        "computed",
        "result_available",
        "forces_available",
        "stress_available",
        "engine_loaded",
        "active_bytes",
        "capacity_bytes",
        "reservations",
        "allocations",
        "plan_retained_bytes",
        "memory_components",
    )

    def _native_diagnostics(self, native):
        """The evaluator's realized state, or the empty state before it exists.

        Nothing has been allocated before the first evaluation, so no realized state
        exists to report: the totals are zero, the itemization is empty, and the
        precision is the one this runtime was built for.  Every key is present either
        way, so a consumer reads one shape.
        """

        if native is None:
            return {
                "precision": self._precision,
                "active_species": [],
                "table_fingerprint": "",
                "retained_rank": 0,
                "rank_total": 0,
                "rank_chunk": 0,
                "rank_chunk_count": 0,
                "n_references": 0,
                "decomposition_max_relative_error": 0.0,
                "decomposition_tolerance_met": False,
                "mesh_spacing": (0.0, 0.0, 0.0),
                "mesh_points": 0,
                "plan_builds": 0,
                "plan_reuses": 0,
                "plan_releases": 0,
                "plan_key_rebuilds": 0,
                "kernel_key_hits": 0,
                "kernel_key_misses": 0,
                "kernel_key_rebuilds": 0,
                "stencil_points": 0,
                "input_generation": 0,
                "published_generation": 0,
                "computed": False,
                "result_available": False,
                "forces_available": False,
                "stress_available": False,
                "engine_loaded": False,
                "active_bytes": 0,
                "capacity_bytes": 0,
                "reservations": 0,
                "allocations": 0,
                "plan_retained_bytes": 0,
                "memory_components": [],
                "fft_plans": [],
                "stage_timing": None,
            }
        report = {name: native[name] for name in self._NATIVE_DIAGNOSTIC_KEYS}
        report["fft_plans"] = native["fft_plans"]
        # The opt-in belongs to this runtime, which is the only thing that can turn it
        # on; the native record supplies the measurement, not the request.
        report["stage_timing"] = native["stage_timing"]
        return report


_RUNTIME_FACTORIES = {
    D2Correction: _build_d2_runtime,
    D3Correction: _build_d3_runtime,
    FourierD3Correction: _build_fourier_d3_runtime,
}


def build_correction_runtime(
    configuration: D2Correction | D3Correction | FourierD3Correction,
    *,
    native: object,
    dtype: object,
) -> CorrectionRuntime:
    """Build the concrete runtime registered for an exact configuration type."""

    factory = _RUNTIME_FACTORIES.get(type(configuration))
    if factory is None:
        raise TypeError(
            "expected a registered correction configuration, got "
            f"{type(configuration).__name__}"
        )
    normalized = replace(
        configuration, functional=configuration.functional.strip().lower()
    )
    return factory(normalized, native=native, dtype=dtype)


class CorrectionPipeline:
    """Evaluate additive correction runtimes and compose their ASE results."""

    def __init__(
        self,
        configurations: Sequence[D2Correction | D3Correction | FourierD3Correction],
        *,
        native: object,
        dtype: object,
    ) -> None:
        configuration_tuple = tuple(configurations)
        for configuration in configuration_tuple:
            # Registered exact types only: a subclass, a bool, or an unregistered mapping
            # must not reach a factory.
            if (
                isinstance(configuration, bool)
                or type(configuration) not in _RUNTIME_FACTORIES
            ):
                raise TypeError(
                    "expected a registered correction configuration, got "
                    f"{type(configuration).__name__}"
                )
        if len(configuration_tuple) > 1:
            raise ValueError(
                "duplicate dispersion corrections are not allowed because they "
                "would double count dispersion energy"
            )
        self._runtimes = tuple(
            build_correction_runtime(configuration, native=native, dtype=dtype)
            for configuration in configuration_tuple
        )

    def __bool__(self) -> bool:
        return bool(self._runtimes)

    @staticmethod
    def _runtime_properties(properties: Sequence[str]) -> tuple[str, ...]:
        requested = []
        for name in properties:
            correction_name = "energy" if name == "free_energy" else name
            if (
                correction_name in _CORRECTION_PROPERTIES
                and correction_name not in requested
            ):
                requested.append(correction_name)
        return tuple(requested)

    def calculate(self, atoms: Atoms, properties: Sequence[str]) -> dict[str, object]:
        required = self._runtime_properties(properties)
        if not required:
            return {}
        totals: dict[str, object] = {}
        for runtime in self._runtimes:
            unsupported = set(required) - runtime.properties
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise RuntimeError(f"correction runtime does not support {names}")
            values = runtime.calculate(atoms, required)
            for name in required:
                if name not in values:
                    raise RuntimeError(
                        f"correction runtime did not return requested {name} result"
                    )
                value = self._validated_result(name, values[name], len(atoms))
                totals[name] = totals[name] + value if name in totals else value
        return totals

    @staticmethod
    def _validated_result(name: str, value: object, atom_count: int) -> object:
        if name == "energy":
            array = np.asarray(value)
            if array.shape != ():
                raise ValueError("correction energy must be a scalar")
            scalar = float(array)
            if not math.isfinite(scalar):
                raise ValueError("correction energy must be finite")
            return scalar

        expected_shapes = {
            "energies": (atom_count,),
            "forces": (atom_count, 3),
            "stress": (6,),
        }
        array = np.asarray(value)
        expected = expected_shapes[name]
        if array.shape != expected:
            raise ValueError(
                f"correction {name} must have shape {expected}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"correction {name} values must be finite")
        return array

    def add_results(
        self,
        atoms: Atoms,
        results: dict[str, object],
        properties: Sequence[str],
    ) -> None:
        corrections = self.calculate(atoms, properties)
        if "energy" in corrections and (
            "energy" in properties or "free_energy" in properties
        ):
            for name in ("energy", "free_energy"):
                if name in results:
                    results[name] = results[name] + corrections["energy"]
        for name in ("energies", "forces", "stress"):
            if name in properties and name in results and name in corrections:
                results[name] = results[name] + corrections[name]

    def diagnostics(self) -> tuple[Mapping[str, object], ...]:
        return tuple(runtime.diagnostics() for runtime in self._runtimes)

    def enable_stage_timing(self, enabled: bool) -> None:
        """Opt into synchronized stage timing for every runtime that can time.

        Only the reciprocal correction exposes the stage seam; a pipeline that holds
        no such runtime is unchanged, and each runtime's diagnostics say whether
        timing is enabled rather than leaving a caller to assume it.
        """

        for runtime in self._runtimes:
            enable = getattr(runtime, "enable_stage_timing", None)
            if enable is not None:
                enable(enabled)


__all__ = [
    "CorrectionPipeline",
    "CorrectionRuntime",
    "D2Correction",
    "D3Correction",
    "FourierD3Correction",
    "GraphRequirement",
    "build_correction_runtime",
]
