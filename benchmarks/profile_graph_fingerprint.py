"""Canonical physical-graph fingerprints for matched profiling workloads."""

import hashlib
import struct

import numpy as np

_SCHEMA = "symmetrix.profile-graph-fingerprint"
_SCHEMA_VERSION = 1
_DEFAULT_COORDINATE_QUANTUM_A = 1.0e-4


def _as_integer_shifts(unit_shifts, edge_count):
    shifts = np.asarray(unit_shifts)
    if shifts.shape != (edge_count, 3) or not np.all(np.isfinite(shifts)):
        raise ValueError("unit shifts must be a finite edge_count-by-3 array")
    rounded = np.rint(shifts)
    if not np.allclose(shifts, rounded, rtol=0.0, atol=1.0e-8):
        raise ValueError("unit shifts must be integral")
    return rounded.astype("<i8")


def canonical_graph_fingerprint(
    atomic_numbers,
    positions,
    cell,
    first_indices,
    second_indices,
    unit_shifts,
    *,
    coordinate_quantum_A=_DEFAULT_COORDINATE_QUANTUM_A,
):
    """Hash a graph independently of edge ordering and geometry precision."""
    numbers = np.asarray(atomic_numbers)
    positions = np.asarray(positions)
    cell = np.asarray(cell)
    first = np.asarray(first_indices)
    second = np.asarray(second_indices)
    atom_count = len(numbers)
    edge_count = len(first)

    if numbers.shape != (atom_count,):
        raise ValueError("atomic numbers must be one-dimensional")
    if positions.shape != (atom_count, 3) or cell.shape != (3, 3):
        raise ValueError("positions and cell must have shapes (atoms, 3) and (3, 3)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
        raise ValueError("positions and cell must be finite")
    if first.shape != (edge_count,) or second.shape != (edge_count,):
        raise ValueError(
            "edge endpoint arrays must be one-dimensional and equal length"
        )

    first = first.astype("<i8", copy=False)
    second = second.astype("<i8", copy=False)
    if edge_count and (
        np.min(first) < 0
        or np.max(first) >= atom_count
        or np.min(second) < 0
        or np.max(second) >= atom_count
    ):
        raise ValueError("edge endpoints must index the atom array")
    shifts = _as_integer_shifts(unit_shifts, edge_count)
    records = np.column_stack((first, second, shifts)).astype("<i8", copy=False)
    order = np.lexsort(tuple(records[:, column] for column in range(4, -1, -1)))
    sorted_records = np.ascontiguousarray(records[order], dtype="<i8")

    if not np.isfinite(coordinate_quantum_A) or coordinate_quantum_A <= 0.0:
        raise ValueError("coordinate quantum must be a positive reciprocal integer")
    inverse_quantum = round(1.0 / coordinate_quantum_A)
    if not np.isclose(
        coordinate_quantum_A * inverse_quantum, 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("coordinate quantum must be a positive reciprocal integer")
    geometry = np.concatenate((cell.reshape(-1), positions.reshape(-1)))
    quantized_geometry = np.rint(geometry * inverse_quantum).astype("<i8")

    digest = hashlib.sha256()
    digest.update(b"symmetrix.graph-fingerprint.v1\0")
    digest.update(struct.pack("<QQq", atom_count, edge_count, inverse_quantum))
    digest.update(np.asarray(numbers, dtype="<i4").tobytes(order="C"))
    digest.update(quantized_geometry.tobytes(order="C"))
    digest.update(sorted_records.tobytes(order="C"))

    edge_digest = hashlib.sha256(sorted_records.tobytes(order="C")).hexdigest()
    input_order_digest = hashlib.sha256(records.tobytes(order="C")).hexdigest()
    return {
        "schema": _SCHEMA,
        "version": _SCHEMA_VERSION,
        "sha256": digest.hexdigest(),
        "coordinate_quantum_A": coordinate_quantum_A,
        "atoms": atom_count,
        "directed_edges": edge_count,
        "edge_record_convention": (
            "(first atom, second atom, integer periodic shift), with "
            "D = position[second] - position[first] + shift @ cell"
        ),
        "sorted_edge_records_sha256": edge_digest,
        "input_order_edge_records_sha256": input_order_digest,
    }


def require_graph_fingerprint(actual, expected):
    """Reject a physical-graph or input-edge-order identity mismatch."""
    fields = ("sha256", "input_order_edge_records_sha256")
    mismatches = {
        field: {"actual": actual.get(field), "expected": expected.get(field)}
        for field in fields
        if actual.get(field) != expected.get(field)
    }
    if mismatches:
        raise RuntimeError(f"unexpected graph fingerprint fields: {mismatches}")
