"""Compact-MACE alchemical message mixing: virtual topology and admission.

This module belongs to the MACE family, so it carries the family prefix: it owns
the dual-topology expansion of the standard two-interaction compact model, the
immutable virtual membership with mutable slot coefficients, and the model
admission the compact frontend applies before it commits to the family's
declared ``alchemical`` evaluator surface.
"""

from __future__ import annotations

from numbers import Integral
from types import MappingProxyType
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from ase.data import atomic_numbers as atomic_numbers_by_symbol
from ase.data import chemical_symbols


@dataclass(frozen=True)
class AlchemicalPair:
    """Concrete foundation species assigned to one virtual-copy slot."""

    atom_index: int
    atomic_number: int


@dataclass(frozen=True)
class AlchemicalTopology:
    """Expanded receiver-major graph and physical reduction maps."""

    node_atomic_numbers: np.ndarray
    node_to_original_atom: np.ndarray
    node_slots: np.ndarray
    num_neigh: np.ndarray
    first_neigh: np.ndarray
    edge_receivers: np.ndarray
    edge_sources: np.ndarray
    edge_neighbor_atomic_numbers: np.ndarray
    edge_shifts: np.ndarray
    edge_to_original_edge: np.ndarray
    atom_node_indices: tuple[tuple[int, ...], ...]
    slot_count: int

    def resolve_weights(
        self, slot_weights: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve current slot coefficients without storing mutable snapshots."""
        values = np.asarray(slot_weights, dtype=float)
        if values.ndim != 1 or values.size != self.slot_count:
            raise ValueError(
                f"alchemical weight length must equal the topology slot count "
                f"({self.slot_count})"
            )
        padded = np.concatenate(([1.0], values))
        node_weights = np.ascontiguousarray(padded[self.node_slots + 1])
        edge_slots = np.where(
            self.node_slots[self.edge_receivers] < 0,
            self.node_slots[self.edge_sources],
            -1,
        )
        edge_weights = np.ascontiguousarray(padded[edge_slots + 1])
        return node_weights, edge_weights


class AlchemicalConfig:
    """Immutable virtual-topology membership with mutable slot weights."""

    def __init__(
        self,
        groups: Sequence[Iterable[AlchemicalPair | tuple[int, int]]],
        weights: Sequence[float],
    ) -> None:
        normalized_groups: list[tuple[AlchemicalPair, ...]] = []
        for group in groups:
            normalized_group = tuple(
                pair if isinstance(pair, AlchemicalPair) else AlchemicalPair(*pair)
                for pair in group
            )
            seen_atoms: set[int] = set()
            for pair in normalized_group:
                if not isinstance(pair.atom_index, (int, np.integer)):
                    raise TypeError("alchemical atom indices must be integers")
                if not isinstance(pair.atomic_number, (int, np.integer)):
                    raise TypeError("alchemical atomic numbers must be integers")
                if pair.atom_index in seen_atoms:
                    raise ValueError(
                        "each alchemical group may contain at most one candidate per atom"
                    )
                if pair.atomic_number <= 0:
                    raise ValueError("alchemical atomic numbers must be positive")
                seen_atoms.add(int(pair.atom_index))
            normalized_groups.append(normalized_group)
        self._groups = tuple(normalized_groups)
        self._weights = np.empty(0, dtype=float)
        self._version = 0
        self._initialized = False
        self.set_weights(weights)
        self._initialized = True

    @property
    def groups(self) -> tuple[tuple[AlchemicalPair, ...], ...]:
        return self._groups

    @property
    def weights(self) -> np.ndarray:
        result = self._weights.copy()
        result.setflags(write=False)
        return result

    @property
    def version(self) -> int:
        return self._version

    def set_weights(self, weights: Sequence[float]) -> None:
        values = np.asarray(weights, dtype=float)
        if values.ndim != 1 or values.size != len(self._groups):
            raise ValueError(
                f"alchemical weight length must equal the number of groups ({len(self._groups)})"
            )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("alchemical weights must be finite and non-negative")
        if self._weights.shape == values.shape and np.array_equal(
            self._weights, values
        ):
            return
        self._weights = np.array(values, dtype=float, copy=True, order="C")
        if self._initialized:
            self._version += 1

    def validate(self, atom_count: int, model_species: Sequence[int]) -> None:
        supported = {int(value) for value in model_species}
        if atom_count < 0:
            raise ValueError("atom_count must be non-negative")
        for group in self._groups:
            for pair in group:
                if pair.atom_index < 0 or pair.atom_index >= atom_count:
                    raise ValueError(
                        f"alchemical atom index {pair.atom_index} is outside the structure"
                    )
                if pair.atomic_number not in supported:
                    raise ValueError(
                        f"model does not support alchemical atomic number {pair.atomic_number}"
                    )


def _resolve_atomic_number(value: str | int, *, role: str) -> int:
    if isinstance(value, str):
        try:
            atomic_number = int(atomic_numbers_by_symbol[value])
        except KeyError as error:
            raise ValueError(f"unknown {role} element symbol {value!r}") from error
    elif isinstance(value, Integral) and not isinstance(value, bool):
        atomic_number = int(value)
    else:
        atomic_number = 0
    if 0 < atomic_number < len(chemical_symbols):
        return atomic_number
    raise ValueError(f"{role} element must be a chemical symbol or atomic number")


class ElementMixture:
    """Apply one shared candidate composition to every reference-element atom."""

    def __init__(
        self,
        reference: str | int,
        components: Mapping[str | int, float],
    ) -> None:
        if not isinstance(components, Mapping):
            raise TypeError("element-mixture components must be a mapping")
        normalized: dict[int, float] = {}
        for element, fraction in components.items():
            atomic_number = _resolve_atomic_number(element, role="component")
            if atomic_number in normalized:
                raise ValueError(
                    f"element-mixture component {chemical_symbols[atomic_number]!r} "
                    "is specified more than once"
                )
            normalized[atomic_number] = fraction
        if len(normalized) < 2:
            raise ValueError("an element mixture requires at least two components")
        self._reference_atomic_number = _resolve_atomic_number(
            reference, role="reference"
        )
        self._component_atomic_numbers = tuple(normalized)
        self._weights = np.empty(0, dtype=float)
        self._version = 0
        self._initialized = False
        self.set_fractions(components)
        self._initialized = True

    @property
    def reference(self) -> str:
        return chemical_symbols[self._reference_atomic_number]

    @property
    def reference_atomic_number(self) -> int:
        return self._reference_atomic_number

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(
            chemical_symbols[atomic_number]
            for atomic_number in self._component_atomic_numbers
        )

    @property
    def fractions(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                chemical_symbols[atomic_number]: float(weight)
                for atomic_number, weight in zip(
                    self._component_atomic_numbers, self._weights
                )
            }
        )

    @property
    def weights(self) -> np.ndarray:
        result = self._weights.copy()
        result.setflags(write=False)
        return result

    @property
    def version(self) -> int:
        return self._version

    def set_fractions(self, components: Mapping[str | int, float]) -> None:
        if not isinstance(components, Mapping):
            raise TypeError("element-mixture fractions must be a mapping")
        normalized: dict[int, float] = {}
        for element, fraction in components.items():
            atomic_number = _resolve_atomic_number(element, role="component")
            if atomic_number in normalized:
                raise ValueError(
                    f"element-mixture component {chemical_symbols[atomic_number]!r} "
                    "is specified more than once"
                )
            normalized[atomic_number] = fraction
        if set(normalized) != set(self._component_atomic_numbers):
            expected = ", ".join(self.components)
            raise ValueError(
                f"element-mixture updates must contain exactly these components: {expected}"
            )
        values = np.asarray(
            [normalized[number] for number in self._component_atomic_numbers],
            dtype=float,
        )
        if values.ndim != 1 or values.size != len(self._component_atomic_numbers):
            raise ValueError("element-mixture fractions must be scalar values")
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(
                "element-mixture fractions must be finite and non-negative"
            )
        if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("element-mixture fractions must sum to one")
        if self._weights.shape == values.shape and np.array_equal(
            self._weights, values
        ):
            return
        self._weights = np.array(values, dtype=float, copy=True, order="C")
        if self._initialized:
            self._version += 1

    def resolve(
        self, atomic_numbers: Sequence[int], model_species: Sequence[int]
    ) -> AlchemicalConfig:
        numbers = np.asarray(atomic_numbers, dtype=np.int32)
        indices = np.flatnonzero(numbers == self._reference_atomic_number)
        if indices.size == 0:
            raise ValueError(
                f"structure contains no reference element {self.reference!r}"
            )
        config = AlchemicalConfig(
            groups=[
                [AlchemicalPair(int(atom_index), component) for atom_index in indices]
                for component in self._component_atomic_numbers
            ],
            weights=self._weights,
        )
        config.validate(len(numbers), model_species)
        return config


def normalize_alchemical(
    value: AlchemicalConfig
    | ElementMixture
    | Mapping[str | int, Mapping[str | int, float]],
) -> AlchemicalConfig | ElementMixture:
    """Own and canonicalize the public alchemical calculator argument."""
    if isinstance(value, (AlchemicalConfig, ElementMixture)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "alchemical must be an AlchemicalConfig, ElementMixture, or mapping"
        )
    if len(value) != 1:
        raise ValueError(
            "an alchemical element mapping must contain exactly one reference element"
        )
    reference, components = next(iter(value.items()))
    return ElementMixture(reference, components)


def resolve_alchemical_config(
    value: AlchemicalConfig | ElementMixture,
    atomic_numbers: Sequence[int],
    model_species: Sequence[int],
) -> AlchemicalConfig:
    """Resolve a structure-independent request to indexed topology membership."""
    if isinstance(value, ElementMixture):
        return value.resolve(atomic_numbers, model_species)
    value.validate(len(atomic_numbers), model_species)
    return value


def validate_alchemical_model(model_data: Mapping[str, Any]) -> None:
    """Reject a model the compact family cannot run alchemically.

    The policy is family data: it names the compact features the family's
    weight-aware packets do not carry, so it lives beside the expansion that
    consumes the graph rather than in the shared frontend that asks for it.
    """

    from .prediction_heads import available_prediction_heads

    if (
        model_data.get("model_type", "MACE") != "MACE"
        or model_data.get("single_layer_readout", False)
        or "R1" not in model_data.get("compact_radial", {}).get("networks", {})
    ):
        raise ValueError(
            "Alchemical execution requires a standard two-interaction compact MACE model."
        )
    transform_type = (
        model_data.get("compact_radial", {})
        .get("distance_transform", {})
        .get("type", "none")
    )
    unsupported = (
        (model_data.get("has_field_coupling", False), "field coupling"),
        (len(available_prediction_heads(model_data)) > 1, "multi-head models"),
        (
            transform_type not in {"none", "agnesi"},
            f"distance transform {transform_type!r}",
        ),
    )
    for rejected, feature in unsupported:
        if rejected:
            raise ValueError(f"Alchemical execution does not support {feature}.")


def expand_alchemical_topology(
    *,
    atomic_numbers: Sequence[int],
    receivers: Sequence[int],
    sources: Sequence[int],
    shifts: np.ndarray,
    config: AlchemicalConfig,
) -> AlchemicalTopology:
    """Expand a physical directed graph using upstream dual-topology rules."""

    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int32)
    receivers = np.asarray(receivers, dtype=np.int32)
    sources = np.asarray(sources, dtype=np.int32)
    shifts = np.asarray(shifts, dtype=np.int32)
    if atomic_numbers.ndim != 1:
        raise ValueError("atomic_numbers must be one-dimensional")
    if receivers.ndim != 1 or sources.ndim != 1 or receivers.size != sources.size:
        raise ValueError("receivers and sources must be equal-length vectors")
    if shifts.shape != (receivers.size, 3):
        raise ValueError("shifts must have shape (num_edges, 3)")
    if np.any(receivers < 0) or np.any(receivers >= atomic_numbers.size):
        raise ValueError("receiver index is outside the physical graph")
    if np.any(sources < 0) or np.any(sources >= atomic_numbers.size):
        raise ValueError("source index is outside the physical graph")
    config.validate(
        atomic_numbers.size,
        [
            *atomic_numbers,
            *(pair.atomic_number for group in config.groups for pair in group),
        ],
    )
    candidates_by_atom: list[dict[int, AlchemicalPair]] = [
        {} for _ in range(atomic_numbers.size)
    ]
    for slot, group in enumerate(config.groups):
        for pair in group:
            candidates_by_atom[pair.atom_index][slot] = pair

    node_atomic_numbers: list[int] = []
    node_to_original_atom: list[int] = []
    node_slots: list[int] = []
    atom_node_indices: list[tuple[int, ...]] = []
    for atom_index, atomic_number in enumerate(atomic_numbers.tolist()):
        candidates = candidates_by_atom[atom_index]
        if not candidates:
            atom_node_indices.append((len(node_atomic_numbers),))
            node_atomic_numbers.append(atomic_number)
            node_to_original_atom.append(atom_index)
            node_slots.append(-1)
            continue
        atom_nodes: list[int] = []
        for slot, pair in candidates.items():
            atom_nodes.append(len(node_atomic_numbers))
            node_atomic_numbers.append(pair.atomic_number)
            node_to_original_atom.append(atom_index)
            node_slots.append(slot)
        atom_node_indices.append(tuple(atom_nodes))

    slot_to_node: list[dict[int, int]] = []
    for atom_nodes in atom_node_indices:
        mapping: dict[int, int] = {}
        for node_index in atom_nodes:
            slot = int(node_slots[node_index])
            mapping[slot] = node_index
        slot_to_node.append(mapping)

    expanded_edges: list[tuple[int, int, int, np.ndarray]] = []
    for original_edge, (receiver, source) in enumerate(zip(receivers, sources)):
        receiver_slots = slot_to_node[int(receiver)]
        source_slots = slot_to_node[int(source)]
        receiver_virtual = -1 not in receiver_slots
        source_virtual = -1 not in source_slots
        if not receiver_virtual and not source_virtual:
            pairs = [(-1, receiver_slots[-1], source_slots[-1])]
        elif receiver_virtual and not source_virtual:
            pairs = [
                (slot, node, source_slots[-1]) for slot, node in receiver_slots.items()
            ]
        elif not receiver_virtual and source_virtual:
            pairs = [
                (slot, receiver_slots[-1], node) for slot, node in source_slots.items()
            ]
        else:
            pairs = [
                (slot, receiver_slots[slot], source_slots[slot])
                for slot in receiver_slots
                if slot in source_slots
            ]
        for _, expanded_receiver, expanded_source in pairs:
            expanded_edges.append(
                (
                    expanded_receiver,
                    expanded_source,
                    original_edge,
                    shifts[original_edge],
                )
            )

    expanded_edges.sort(key=lambda edge: edge[0])
    edge_receivers = np.asarray([edge[0] for edge in expanded_edges], dtype=np.int32)
    edge_sources = np.asarray([edge[1] for edge in expanded_edges], dtype=np.int32)
    edge_to_original_edge = np.asarray(
        [edge[2] for edge in expanded_edges], dtype=np.int32
    )
    edge_shifts = (
        np.asarray([edge[3] for edge in expanded_edges], dtype=np.int32).reshape(-1, 3)
        if expanded_edges
        else np.empty((0, 3), dtype=np.int32)
    )
    num_neigh = np.bincount(edge_receivers, minlength=len(node_atomic_numbers)).astype(
        np.int32, copy=False
    )
    first_neigh = np.empty(len(node_atomic_numbers), dtype=np.int32)
    if len(first_neigh):
        first_neigh[0] = 0
        np.cumsum(num_neigh[:-1], dtype=np.int32, out=first_neigh[1:])
    edge_neighbor_atomic_numbers = np.asarray(
        [node_atomic_numbers[source] for source in edge_sources], dtype=np.int32
    )
    return AlchemicalTopology(
        node_atomic_numbers=np.asarray(node_atomic_numbers, dtype=np.int32),
        node_to_original_atom=np.asarray(node_to_original_atom, dtype=np.int32),
        node_slots=np.asarray(node_slots, dtype=np.int32),
        num_neigh=num_neigh,
        first_neigh=first_neigh,
        edge_receivers=edge_receivers,
        edge_sources=edge_sources,
        edge_neighbor_atomic_numbers=edge_neighbor_atomic_numbers,
        edge_shifts=edge_shifts,
        edge_to_original_edge=edge_to_original_edge,
        atom_node_indices=tuple(atom_node_indices),
        slot_count=len(config.groups),
    )


__all__ = [
    "AlchemicalConfig",
    "AlchemicalPair",
    "AlchemicalTopology",
    "ElementMixture",
    "expand_alchemical_topology",
    "normalize_alchemical",
    "resolve_alchemical_config",
    "validate_alchemical_model",
]
