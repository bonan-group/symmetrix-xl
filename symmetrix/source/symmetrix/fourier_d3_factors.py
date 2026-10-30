"""Deterministic low-rank decomposition of the DFT-D3 reference C6 tensor.

D3's :math:`C_6` couples both atoms' coordination numbers, which forbids a particle-mesh
evaluation. FourierD3 restores separability by decomposing the tabulated reference tensor
once, offline, into :math:`C_6^{\\mathrm{ref}}(Z_i,\\theta^p;Z_j,\\theta^q) \\approx
\\sum_\\ell \\lambda_\\ell v_{\\ell,Z_i}^p v_{\\ell,Z_j}^q`, so that the per-atom
coefficients contract into a signed rank sum.

This module is the installed, source-independent implementation of specification
sections 3.3.2 to 3.3.4. It consumes the canonical ``d3_c6`` table as raw arrays plus an
explicit table fingerprint; it never reads the repository, the C++ sources, or any test or
tooling module. NumPy is the only dependency.

Canonical table layout (``dispersion_parameters.cpp``, indexed as ``c6_index`` in
``d3_kokkos_impl.tpp``): ``(max_z + 1, max_z + 1, n_ref, n_ref, 3)`` with field 0 the
reference :math:`C_6` in eV/A^6, field 1 the receiver's reference coordination number, and
field 2 the source's. Slots a species does not use hold ``-1`` in every field; a valid
reference entry has a strictly positive :math:`C_6`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "D3LowRankFactors",
    "clear_d3_decomposition_cache",
    "decompose_d3_c6_reference",
]

# Field offsets of the canonical ``d3_c6`` table.
C6_FIELD = 0
RECEIVER_REFERENCE_FIELD = 1
SOURCE_REFERENCE_FIELD = 2
FIELD_COUNT = 3

DEFAULT_TOLERANCE = 1e-4

# Padding for reference slots a species does not use. Chosen far from any physical
# coordination number, so a consumer that forgets the validity mask still gets a weight
# that underflows instead of a plausible-looking spurious contribution.
REFERENCE_INVALID = -1.0

# Bounded so that a long-running process sweeping many compositions cannot grow without
# limit.
_CACHE_SIZE = 32
_CACHE: OrderedDict[tuple, "D3LowRankFactors"] = OrderedDict()


@dataclass(frozen=True)
class D3LowRankFactors:
    """Signed low-rank factors of the active-species D3 reference C6 block.

    Attributes
    ----------
    species : np.ndarray, shape (n_species,)
        Covered atomic numbers, ascending. Position is the channel index.
    eigs : np.ndarray, shape (rank,)
        Eigenvalues ordered by descending magnitude. Signed: the energy is a signed sum.
    v_q : np.ndarray, shape (n_species, n_ref, rank)
        Eigenvectors, sign-normalised per specification section 3.3.4. Rows of invalid
        reference slots are exactly zero.
    cn_ref : np.ndarray, shape (n_species, n_ref)
        Reference coordination numbers, ``REFERENCE_INVALID`` where a slot is unused.
    valid : np.ndarray, shape (n_species, n_ref)
        True where a reference slot carries data.
    table_fingerprint : str
        Caller-supplied identity of the table set the factors were built from.
    tolerance : float
        Requested maximum relative reconstruction error.
    rank_cap : int or None
        Requested hard ceiling on the retained rank.
    max_relative_error : float
        Largest relative reconstruction error over populated entries at the retained rank.
    tolerance_met : bool
        Whether the retained rank met ``tolerance``.
    rank_capped : bool
        Whether a rank cap below the full decomposition rank prevented the requested
        tolerance from being met.
    """

    species: np.ndarray
    eigs: np.ndarray
    v_q: np.ndarray
    cn_ref: np.ndarray
    valid: np.ndarray
    table_fingerprint: str
    tolerance: float
    rank_cap: int | None
    max_relative_error: float
    tolerance_met: bool
    rank_capped: bool

    def __post_init__(self) -> None:
        species = np.asarray(self.species, dtype=np.int64)
        eigs = np.asarray(self.eigs, dtype=np.float64)
        v_q = np.asarray(self.v_q, dtype=np.float64)
        cn_ref = np.asarray(self.cn_ref, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if eigs.ndim != 1 or cn_ref.ndim != 2 or valid.shape != cn_ref.shape:
            raise ValueError(
                "eigs must be 1-D and cn_ref and valid must share a shape; got "
                f"{eigs.shape}, {cn_ref.shape}, {valid.shape}"
            )
        if species.shape != (cn_ref.shape[0],):
            raise ValueError(
                "species must have one entry per reference row; got "
                f"{species.shape} for cn_ref {cn_ref.shape}"
            )
        if v_q.shape != (cn_ref.shape[0], cn_ref.shape[1], eigs.shape[0]):
            raise ValueError(
                "v_q must have shape (n_species, n_ref, rank); got "
                f"{v_q.shape} for species {species.size}, rank {eigs.size}"
            )
        for name, array in (
            ("species", species),
            ("eigs", eigs),
            ("v_q", v_q),
            ("cn_ref", cn_ref),
            ("valid", valid),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def rank(self) -> int:
        """Number of retained rank slots."""
        return int(self.eigs.size)

    @property
    def n_species(self) -> int:
        """Number of covered species."""
        return int(self.species.size)

    @property
    def n_ref(self) -> int:
        """Reference-coordination slots per species, including padding."""
        return int(self.cn_ref.shape[1])


def clear_d3_decomposition_cache() -> None:
    """Drop every cached decomposition."""
    _CACHE.clear()


def _validate_options(table_fingerprint, tolerance, rank_cap) -> None:
    if not isinstance(table_fingerprint, str) or not table_fingerprint:
        raise ValueError("table_fingerprint must be a non-empty string")
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float, np.number)
    ):
        raise ValueError(
            f"tolerance must be a positive finite number, got {tolerance!r}"
        )
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError(
            f"tolerance must be a positive finite number, got {tolerance!r}"
        )
    if rank_cap is not None:
        if isinstance(rank_cap, bool) or not isinstance(rank_cap, (int, np.integer)):
            raise ValueError(
                f"rank_cap must be a positive integer or None, got {rank_cap!r}"
            )
        if int(rank_cap) < 1:
            raise ValueError(f"rank_cap must be at least 1, got {rank_cap}")


def _validate_table(c6_reference) -> np.ndarray:
    table = np.asarray(c6_reference)
    if table.dtype.kind not in "fiu":
        raise ValueError(
            f"c6_reference must be a real numeric array, got {table.dtype}"
        )
    if table.ndim != 5 or table.shape[4] != FIELD_COUNT:
        raise ValueError(
            "c6_reference must have shape (max_z + 1, max_z + 1, n_ref, n_ref, "
            f"{FIELD_COUNT}); got {table.shape}"
        )
    if table.shape[0] != table.shape[1]:
        raise ValueError(
            f"c6_reference must be square in its atomic-number axes; got {table.shape}"
        )
    if table.shape[2] != table.shape[3]:
        raise ValueError(
            f"c6_reference must be square in its reference axes; got {table.shape}"
        )
    if table.shape[0] < 2 or table.shape[2] < 1:
        raise ValueError(
            f"c6_reference is too small to carry any species; got {table.shape}"
        )
    return np.ascontiguousarray(table, dtype=np.float64)


def _validate_species(species, max_z: int) -> np.ndarray:
    raw = np.asarray(species)
    if raw.dtype.kind == "f" and not np.array_equal(raw, np.rint(raw)):
        raise ValueError(f"species must be whole atomic numbers, got {species!r}")
    try:
        requested = np.asarray(species, dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"species must be atomic numbers, got {species!r}") from error
    if requested.ndim != 1:
        raise ValueError(
            f"species must be one-dimensional, got shape {requested.shape}"
        )
    unique = np.unique(requested)
    if unique.size == 0:
        raise ValueError("species must contain at least one atomic number")
    if unique[0] < 1 or unique[-1] > max_z:
        raise ValueError(
            f"species {unique.tolist()} is outside the reference table, which covers "
            f"1..{max_z}"
        )
    return unique


def _species_validity(table: np.ndarray, species: np.ndarray) -> np.ndarray:
    """True where a species populates a reference slot, matching the canonical table."""
    n_ref = table.shape[2]
    valid = np.zeros((species.size, n_ref), dtype=bool)
    for row, atomic_number in enumerate(species):
        valid[row] = (table[atomic_number, :, :, :, C6_FIELD] > 0.0).any(axis=(0, 2))
        if not valid[row].any():
            raise ValueError(
                f"no reference C6 data for atomic number {int(atomic_number)}"
            )
    return valid


def _reference_coordination(
    table: np.ndarray, species: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Per-species reference coordination numbers, sentinel where a slot is unused."""
    cn_ref = np.full(valid.shape, REFERENCE_INVALID, dtype=np.float64)
    for row, atomic_number in enumerate(species):
        populated = table[atomic_number, :, :, :, C6_FIELD] > 0.0
        for slot in np.flatnonzero(valid[row]):
            values = table[atomic_number, :, slot, :, RECEIVER_REFERENCE_FIELD][
                populated[:, slot, :]
            ]
            if not np.all(values == values[0]):
                raise ValueError(
                    f"reference coordination for Z={int(atomic_number)} slot {int(slot)} "
                    f"is partner-dependent ({values.min()} to {values.max()}); the "
                    "per-species decomposition is not valid for this table"
                )
            cn_ref[row, slot] = values[0]
    return cn_ref


def _species_block(
    table: np.ndarray, species: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Symmetric ``(n_species * n_ref)`` block of specification section 3.3.2."""
    n_ref = table.shape[2]
    size = species.size * n_ref
    block = np.zeros((size, size), dtype=np.float64)
    for row, receiver in enumerate(species):
        for column, source in enumerate(species):
            sub = table[receiver, source, :, :, C6_FIELD]
            mask = valid[row][:, None] & valid[column][None, :] & (sub > 0.0)
            block[
                row * n_ref : (row + 1) * n_ref,
                column * n_ref : (column + 1) * n_ref,
            ] = np.where(mask, sub, 0.0)
    # The tables are symmetric only up to storage noise, and ``eigh`` requires an exactly
    # symmetric operand.
    return 0.5 * (block + block.T)


def _reconstruction_error(
    block: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, rank: int
) -> float:
    """Largest relative error of the rank-``rank`` reconstruction over populated entries."""
    populated = block != 0.0
    approximation = (eigenvectors[:, :rank] * eigenvalues[:rank]) @ eigenvectors[
        :, :rank
    ].T
    return float(
        np.max(
            np.abs(approximation[populated] - block[populated])
            / np.abs(block[populated])
        )
    )


def _select_rank(
    block: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    tolerance: float,
    rank_cap: int | None,
) -> tuple[int, float, bool, bool]:
    """Return rank, achieved error, tolerance verdict, and whether the cap bound.

    Scans upward instead of bisecting: the error is not monotone in the rank, because a
    signed term can cancel against earlier ones, so a bisection can step over a smaller
    rank that already qualifies. When no rank meets the tolerance the cap (or the full
    rank) is returned with the error it achieves, so the caller can report it.
    """
    if not (block != 0.0).any():
        raise ValueError("the active-species reference block is entirely zero")
    upper = (
        eigenvalues.size if rank_cap is None else min(int(rank_cap), eigenvalues.size)
    )
    errors = [
        _reconstruction_error(block, eigenvalues, eigenvectors, rank)
        for rank in range(1, upper + 1)
    ]
    for rank, error in enumerate(errors, start=1):
        if error <= tolerance:
            return rank, error, True, False
    rank_capped = rank_cap is not None and int(rank_cap) < eigenvalues.size
    return upper, errors[-1], False, rank_capped


def _sign_gauge(vectors: np.ndarray) -> None:
    """Sign-normalise every retained eigenvector in place (specification section 3.3.4).

    ``E`` is invariant under ``v -> -v``, so the sign is a pure gauge. Fixing it makes the
    retained rank and every cached byte reproducible across LAPACK builds: the entry of
    largest magnitude is made positive, ties broken by the smallest flat index (which is
    what ``argmax`` returns).
    """
    flattened = vectors.reshape(-1, vectors.shape[-1])
    for slot in range(flattened.shape[1]):
        column = flattened[:, slot]
        if column[int(np.argmax(np.abs(column)))] < 0.0:
            column *= -1.0


def decompose_d3_c6_reference(
    c6_reference,
    species,
    *,
    table_fingerprint: str,
    tolerance: float = DEFAULT_TOLERANCE,
    rank_cap: int | None = None,
) -> D3LowRankFactors:
    r"""Decompose the active-species block of the canonical D3 C6 table.

    Parameters
    ----------
    c6_reference : array_like, shape (max_z + 1, max_z + 1, n_ref, n_ref, 3)
        Canonical ``d3_c6`` table: field 0 the reference :math:`C_6` in eV/A^6 with ``-1``
        in unused slots, fields 1 and 2 the receiver's and source's reference coordination
        numbers.
    species : Sequence[int]
        Atomic numbers present. Order and duplicates are ignored: the result is
        canonicalised to ascending unique values, so the channel layout is deterministic.
    table_fingerprint : str
        Identity of the table set the arrays came from. It is the caller's contract that
        the fingerprint changes whenever the arrays change; it keys the cache.
    tolerance : float, default=1e-4
        Target maximum relative reconstruction error over populated entries.
    rank_cap : int, optional
        Hard ceiling on the retained rank. When the ceiling prevents ``tolerance`` from
        being met, the decomposition is returned anyway with
        :attr:`D3LowRankFactors.tolerance_met` false and the achieved error reported.

    Returns
    -------
    D3LowRankFactors
        Signed factors, reference coordination numbers, validity mask, and metadata. The
        arrays are read-only and are shared with the cache: callers must not mutate them.

    Raises
    ------
    ValueError
        If the table shape, species, fingerprint, tolerance, or rank cap is invalid, or if
        a requested species carries no reference data.
    """
    _validate_options(table_fingerprint, tolerance, rank_cap)
    table = _validate_table(c6_reference)
    unique_species = _validate_species(species, table.shape[0] - 1)

    key = (
        table_fingerprint,
        tuple(int(value) for value in unique_species),
        float(tolerance),
        None if rank_cap is None else int(rank_cap),
    )
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached

    valid = _species_validity(table, unique_species)
    cn_ref = _reference_coordination(table, unique_species, valid)
    block = _species_block(table, unique_species, valid)

    eigenvalues, eigenvectors = np.linalg.eigh(block)
    # eigh returns ascending eigenvalues; the energy is a signed sum, so the rank order
    # follows magnitude rather than value.
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    rank, error, tolerance_met, rank_capped = _select_rank(
        block, eigenvalues, eigenvectors, float(tolerance), rank_cap
    )

    vectors = np.ascontiguousarray(eigenvectors[:, :rank]).reshape(
        unique_species.size, valid.shape[1], rank
    )
    # Unused reference slots span the null space, so their entries are zero only up to
    # round-off. Clearing them exactly keeps a consumer that ignores the mask correct.
    vectors[~valid] = 0.0
    _sign_gauge(vectors)

    result = D3LowRankFactors(
        species=unique_species,
        eigs=np.ascontiguousarray(eigenvalues[:rank]),
        v_q=vectors,
        cn_ref=cn_ref,
        valid=valid,
        table_fingerprint=table_fingerprint,
        tolerance=float(tolerance),
        rank_cap=None if rank_cap is None else int(rank_cap),
        max_relative_error=error,
        tolerance_met=tolerance_met,
        rank_capped=rank_capped,
    )
    _CACHE[key] = result
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return result
