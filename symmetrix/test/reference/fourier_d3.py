"""Mesh-free direct reciprocal-space FourierD3 reference oracle.

Permanent, pure-NumPy reference for the frozen conventions of
``docs/reference/fourier_d3_mathematics.md`` (revision
``6328429``). It implements the model directly: modified coordination numbers, the signed
low-rank channel contraction, the closed-form Becke-Johnson reciprocal kernel, the
explicit structure factors of specification section 4.2, the ``k = 0`` bin, the self term,
and the analytic forces and strain derivative.

Deliberate limitations, all of them requirements rather than omissions:

* **No FFT and no particle mesh.** The reciprocal sum is an explicit sum over Miller
  indices, so this oracle is the reference a mesh implementation is measured against
  rather than a second mesh implementation.
* **No generic eigendecomposition.** The low-rank factors are inputs
  (:class:`LowRankFactors`); the oracle only contracts them.
* **No PyTorch, Warp, or NVIDIA runtime dependency.** Only NumPy is imported.
* **No real-space cutoff on the dispersion sum.** ``coordination_cutoff`` bounds only the
  short coordination-number neighbour list.

Units are the Symmetrix working system: angstrom and electronvolt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


__all__ = [
    "BOHR_TO_ANGSTROM",
    "COORDINATION_EPSILON",
    "COORDINATION_STEEPNESS",
    "COORDINATION_UNSCALE",
    "DEFAULT_K_MAX",
    "DampingParameters",
    "Evaluation",
    "K3_WEIGHT",
    "LowRankFactors",
    "SERIES_CUTOFF",
    "SPECIFICATION_COMMIT",
    "SPECIFICATION_DOCUMENT",
    "build_directed_edges",
    "coordination_numbers",
    "coordination_pair_term",
    "coordination_pair_term_derivative",
    "direct_reciprocal_energy",
    "evaluate",
    "kernel_shape_functions",
    "low_rank_coefficient_derivative",
    "low_rank_coefficients",
    "pair_potential",
    "reciprocal_energy",
    "reciprocal_kernel",
    "reciprocal_kernel_derivative",
    "reciprocal_vectors",
    "self_energy",
    "structure_factors",
]


SPECIFICATION_DOCUMENT = "docs/reference/fourier_d3_mathematics.md"
SPECIFICATION_COMMIT = "6328429"

BOHR_TO_ANGSTROM = 0.52917725999999998

# Specification sections 3.2 and 8.
COORDINATION_STEEPNESS = 16.0
COORDINATION_UNSCALE = 0.75
COORDINATION_EPSILON = 1e-6

# Specification section 3.3.1.
K3_WEIGHT = 4.0

# Specification section 4.4: series/closed-form crossover and the series coefficients.
SERIES_CUTOFF = 0.1
_SERIES_R6 = (1.0, 0.0, -1.0 / 3.0, 0.125, -1.0 / 60.0, 0.0, 1.0 / 5040.0)
_SERIES_R8 = (
    -0.54119610014619668,
    0.0,
    0.090199350024366076,
    0.0,
    -0.010888024707303151,
    1.0 / 360.0,
    -0.00025923868350721731,
)

# Specification section 8: suppresses the anisotropic 1/|k| term at k = 0.
_K_NORM_GUARD = 1e-12

# The reciprocal sum is truncated where the kernel has decayed exponentially. The
# argument is x = |k| R0, so the useful cutoff is set by R0 ~ 3.1 angstrom rather than by
# the cell; 24 1/angstrom puts the smallest omitted contribution near 1e-14 of the energy
# for the fixture cells, and the value is re-checked by
# ``test_reciprocal_sum_is_converged_in_k_max``. A caller may lower it, at the cost of a
# truncation error that decays only exponentially.
DEFAULT_K_MAX = 24.0


@dataclass(frozen=True)
class DampingParameters:
    """Becke-Johnson damping parameters in working units.

    ``a2`` is stored in angstrom because specification section 2.1 converts it once at
    configuration construction; ``a1`` is dimensionless and is never converted.
    """

    s6: float
    s8: float
    a1: float
    a2_angstrom: float

    @classmethod
    def from_bohr(
        cls, s6: float, s8: float, a1: float, a2_bohr: float
    ) -> "DampingParameters":
        return cls(s6=s6, s8=s8, a1=a1, a2_angstrom=a2_bohr * BOHR_TO_ANGSTROM)

    def damping_radius(self, q_product):
        """``R0 = a1 sqrt(3 q_X q_Y) + a2`` in angstrom (specification section 3.1)."""
        return (
            self.a1 * np.sqrt(3.0 * np.asarray(q_product, dtype=np.float64))
            + self.a2_angstrom
        )


@dataclass(frozen=True)
class LowRankFactors:
    """Frozen signed low-rank factors of the reference C6 tensor.

    Attributes
    ----------
    species : np.ndarray, shape (n_species,)
        Atomic numbers covered, ascending. Position is the channel index.
    eigs : np.ndarray, shape (rank,)
        Eigenvalues ordered by descending magnitude. Signed: the energy is a signed sum.
    v_q : np.ndarray, shape (n_species, n_ref, rank)
        Eigenvectors. Rows of invalid reference slots are exactly zero.
    cn_ref : np.ndarray, shape (n_species, n_ref)
        Reference coordination numbers per species.
    valid : np.ndarray, shape (n_species, n_ref)
        True where a reference slot carries data.
    q_table : np.ndarray, shape (n_species,)
        The tabulated ``r2r4`` values, already scaled to angstrom, indexed by channel.
    """

    species: np.ndarray
    eigs: np.ndarray
    v_q: np.ndarray
    cn_ref: np.ndarray
    valid: np.ndarray
    q_table: np.ndarray

    def __post_init__(self) -> None:
        species = np.asarray(self.species, dtype=np.int64)
        eigs = np.asarray(self.eigs, dtype=np.float64)
        v_q = np.asarray(self.v_q, dtype=np.float64)
        cn_ref = np.asarray(self.cn_ref, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        q_table = np.asarray(self.q_table, dtype=np.float64)
        if v_q.shape != (
            species.size,
            cn_ref.shape[1] if cn_ref.ndim == 2 else 0,
            eigs.size,
        ):
            raise ValueError(
                "v_q must have shape (n_species, n_ref, rank); got "
                f"{v_q.shape} for species {species.size}, rank {eigs.size}"
            )
        if cn_ref.shape != valid.shape or cn_ref.shape != (species.size, v_q.shape[1]):
            raise ValueError(
                "cn_ref and valid must both have shape (n_species, n_ref); got "
                f"{cn_ref.shape} and {valid.shape}"
            )
        if q_table.shape != species.shape:
            raise ValueError(
                f"q_table must have one value per species; got {q_table.shape} "
                f"for {species.size} species"
            )
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "eigs", eigs)
        object.__setattr__(self, "v_q", v_q)
        object.__setattr__(self, "cn_ref", cn_ref)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "q_table", q_table)

    @property
    def rank(self) -> int:
        return int(self.eigs.size)

    @property
    def n_species(self) -> int:
        return int(self.species.size)

    @property
    def n_ref(self) -> int:
        return int(self.cn_ref.shape[1])

    def channels(self, numbers) -> np.ndarray:
        """Atomic number to channel index; ``-1`` for an uncovered species."""
        numbers = np.asarray(numbers, dtype=np.int64)
        size = int(numbers.max()) + 1 if numbers.size else 1
        lookup = np.full(max(size, int(self.species.max()) + 1), -1, dtype=np.int64)
        lookup[self.species] = np.arange(self.species.size, dtype=np.int64)
        return lookup[numbers]


@dataclass(frozen=True)
class Evaluation:
    """Complete direct-reciprocal evaluation of one geometry."""

    energy: float
    reciprocal_energy: float
    self_energy: float
    coordination: np.ndarray
    coefficients: np.ndarray
    coordination_sensitivity: np.ndarray
    forces: np.ndarray
    strain_derivative: np.ndarray
    edges: np.ndarray
    shifts: np.ndarray


def _series(coefficients: Sequence[float], x: np.ndarray) -> np.ndarray:
    """Evaluate a truncated power series by Horner's rule."""
    total = np.zeros_like(x)
    for coefficient in reversed(coefficients):
        total = total * x + coefficient
    return total


def _series_derivative(coefficients: Sequence[float], x: np.ndarray) -> np.ndarray:
    """Evaluate the derivative of a truncated power series by Horner's rule."""
    value = np.zeros_like(x)
    derivative = np.zeros_like(x)
    for coefficient in reversed(coefficients):
        derivative = derivative * x + value
        value = value * x + coefficient
    return derivative


def kernel_shape_functions(x):
    """``G6`` and ``G8`` of specification section 4.4, with the small-``x`` series.

    Both numerators vanish at the origin, so the closed forms divide two cancelling
    quantities; below ``SERIES_CUTOFF`` the entire functions are evaluated from their
    series instead. The crossover and coefficients are normative.
    """
    x = np.asarray(x, dtype=np.float64)
    small = x < SERIES_CUTOFF
    safe = np.where(small, 1.0, x)

    root3 = np.sqrt(3.0)
    n6 = np.exp(-safe) - 2.0 * np.exp(-safe / 2.0) * np.cos(
        np.pi / 3.0 + safe * root3 / 2.0
    )
    sin8 = np.sin(np.pi / 8.0)
    cos8 = np.cos(np.pi / 8.0)
    n8 = np.exp(-safe * sin8) * np.cos(np.pi / 4.0 + safe * cos8) + np.exp(
        -safe * cos8
    ) * np.cos(3.0 * np.pi / 4.0 + safe * sin8)

    g6 = np.where(small, _series(_SERIES_R6, x), n6 / safe)
    g8 = np.where(small, _series(_SERIES_R8, x), n8 / safe)
    return g6, g8


def _shape_function_derivatives(x):
    """``dG6/dx`` and ``dG8/dx`` on the same branch as :func:`kernel_shape_functions`."""
    x = np.asarray(x, dtype=np.float64)
    small = x < SERIES_CUTOFF
    safe = np.where(small, 1.0, x)

    root3 = np.sqrt(3.0)
    phase6 = np.pi / 3.0 + safe * root3 / 2.0
    cos6 = np.cos(phase6)
    sin6 = np.sin(phase6)
    n6 = np.exp(-safe) - 2.0 * np.exp(-safe / 2.0) * cos6
    dn6 = (
        -np.exp(-safe) + np.exp(-safe / 2.0) * cos6 + np.exp(-safe / 2.0) * sin6 * root3
    )

    sin8 = np.sin(np.pi / 8.0)
    cos8 = np.cos(np.pi / 8.0)
    phase8a = np.pi / 4.0 + safe * cos8
    phase8b = 3.0 * np.pi / 4.0 + safe * sin8
    n8 = np.exp(-safe * sin8) * np.cos(phase8a) + np.exp(-safe * cos8) * np.cos(phase8b)
    dn8 = (
        -sin8 * np.exp(-safe * sin8) * np.cos(phase8a)
        - cos8 * np.exp(-safe * sin8) * np.sin(phase8a)
        - cos8 * np.exp(-safe * cos8) * np.cos(phase8b)
        - sin8 * np.exp(-safe * cos8) * np.sin(phase8b)
    )

    g6 = np.where(small, _series_derivative(_SERIES_R6, x), dn6 / safe - n6 / safe**2)
    g8 = np.where(small, _series_derivative(_SERIES_R8, x), dn8 / safe - n8 / safe**2)
    return g6, g8


def reciprocal_kernel(k_norm, r0, q_product, parameters: DampingParameters):
    r"""``\hat\varphi_{XY}(|k|)`` of specification section 4.4.

    Real, even, and exponentially decaying; it is an exponentially damped oscillation, so
    it changes sign, with its first zero near ``|k| R0 ~ 4.3``.
    """
    k_norm = np.asarray(k_norm, dtype=np.float64)
    g6, g8 = kernel_shape_functions(k_norm * r0)
    term6 = 2.0 * np.pi**2 / (3.0 * r0**3) * g6
    term8 = -(np.pi**2) / r0**5 * g8
    return parameters.s6 * term6 + 3.0 * parameters.s8 * q_product * term8


def reciprocal_kernel_derivative(k_norm, r0, q_product, parameters: DampingParameters):
    r"""``d\hat\varphi/d|k|``, produced by the same branch as :func:`reciprocal_kernel`."""
    k_norm = np.asarray(k_norm, dtype=np.float64)
    dg6, dg8 = _shape_function_derivatives(k_norm * r0)
    dterm6 = 2.0 * np.pi**2 / (3.0 * r0**3) * dg6
    dterm8 = -(np.pi**2) / r0**5 * dg8
    return r0 * (parameters.s6 * dterm6 + 3.0 * parameters.s8 * q_product * dterm8)


def pair_potential(distance, r0, q_product, parameters: DampingParameters):
    r"""Real-space damped potential ``\varphi_{XY}(r)`` of specification section 3.1."""
    distance = np.asarray(distance, dtype=np.float64)
    return parameters.s6 / (distance**6 + r0**6) + 3.0 * parameters.s8 * q_product / (
        distance**8 + r0**8
    )


def coordination_pair_term(distance, covalent_sum, cutoff):
    r"""Modified counting function ``\theta_{ij}(r)`` of specification section 3.2.

    ``covalent_sum`` is the table sum of the two ``rcov`` values, which already carries
    Grimme's 4/3 factor. The steepness law is the NVIDIA v0.5.0 form, the one that
    vanishes continuously at the cutoff; the paper's literal Eq. (18) is deliberately not
    used.
    """
    distance = np.asarray(distance, dtype=np.float64)
    covalent_sum = np.asarray(covalent_sum, dtype=np.float64)
    transition = 0.5 * (COORDINATION_UNSCALE * covalent_sum + cutoff)
    stabilised = np.maximum(distance, transition)
    gap = cutoff - stabilised
    steepness = COORDINATION_STEEPNESS + (stabilised - transition) ** 2 / (
        gap * gap + COORDINATION_EPSILON
    )
    argument = steepness * (covalent_sum / distance - 1.0)
    # Branchwise logistic: the argument reaches +/- infinity at the cutoff, where a plain
    # 1/(1+exp(-x)) overflows. Each branch keeps the exponent non-positive.
    exponential = np.exp(-np.abs(argument))
    value = np.where(
        argument >= 0.0, 1.0 / (1.0 + exponential), exponential / (1.0 + exponential)
    )
    return np.where(distance < cutoff, value, 0.0)


def coordination_pair_term_derivative(distance, covalent_sum, cutoff):
    r"""``d\theta_{ij}/dr`` of specification section 3.2.2."""
    distance = np.asarray(distance, dtype=np.float64)
    covalent_sum = np.asarray(covalent_sum, dtype=np.float64)
    theta = coordination_pair_term(distance, covalent_sum, cutoff)
    transition = 0.5 * (COORDINATION_UNSCALE * covalent_sum + cutoff)
    stabilised = np.maximum(distance, transition)
    gap = cutoff - stabilised
    denominator = gap * gap + COORDINATION_EPSILON
    steepness = COORDINATION_STEEPNESS + (stabilised - transition) ** 2 / denominator
    d_steepness = np.where(
        distance > transition,
        2.0 * (stabilised - transition) / denominator
        + 2.0 * (stabilised - transition) ** 2 * gap / denominator**2,
        0.0,
    )
    ratio = covalent_sum / distance - 1.0
    derivative = (
        theta
        * (1.0 - theta)
        * (d_steepness * ratio + steepness * (-covalent_sum / distance**2))
    )
    return np.where(distance < cutoff, derivative, 0.0)


def build_directed_edges(positions, cell, cutoff, active=None):
    """Directed periodic edges with ``0 < r < cutoff``; self-images are retained.

    Both orientations of every pair are present, because a coordination number is
    accumulated on its receiver (specification section 3.2.4). The ``(i, i, T = 0)``
    diagonal is excluded; every ``(i, i, T != 0)`` self-image is kept, since an atom does
    interact with its own periodic images. ``active`` marks the atoms that are really
    present; padding atoms (atomic number 0, specification section 3.2.4) are skipped as
    both source and target.
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    active = None if active is None else np.asarray(active, dtype=bool)
    widths = np.abs(np.linalg.det(cell)) / np.linalg.norm(
        np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1
    )
    span = 2.0 * float(np.abs(positions).max())
    reach = [int(np.ceil((cutoff + span) / width)) for width in widths]
    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in reach], indexing="ij")
    translations = np.stack(grids, axis=-1).reshape(-1, 3) @ cell
    inverse_cell = np.linalg.inv(cell)
    edge_blocks = []
    shift_blocks = []
    for translation in translations:
        delta = positions[None, :, :] + translation - positions[:, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        inside = (distance > 0.0) & (distance < cutoff)
        if active is not None:
            inside &= active[:, None] & active[None, :]
        if not np.any(translation):
            np.fill_diagonal(inside, False)
        source, target = np.nonzero(inside)
        if source.size == 0:
            continue
        edge_blocks.append(np.stack([source, target], axis=1))
        shift_blocks.append(np.tile(translation @ inverse_cell, (source.size, 1)))
    if not edge_blocks:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
    return (
        np.concatenate(edge_blocks).astype(np.int64),
        np.rint(np.concatenate(shift_blocks)).astype(np.int64),
    )


def coordination_numbers(
    positions, covalent_radii, cell, cutoff, edges=None, shifts=None, active=None
):
    r"""Coordination numbers ``\theta_i`` of specification section 3.2.

    ``active`` masks padding atoms out of a caller-supplied edge list as well; masking is
    idempotent, so an edge list already built with ``active`` may be passed unchanged.
    """
    positions = np.asarray(positions, dtype=np.float64)
    covalent_radii = np.asarray(covalent_radii, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    if edges is None or shifts is None:
        edges, shifts = build_directed_edges(positions, cell, cutoff, active)
    source, target = edges[:, 0], edges[:, 1]
    if active is not None:
        active = np.asarray(active, dtype=bool)
        keep = active[source] & active[target]
        source, target, shifts = source[keep], target[keep], shifts[keep]
    delta = positions[target] - positions[source] + shifts @ cell
    distance = np.linalg.norm(delta, axis=1)
    covalent_sum = covalent_radii[source] + covalent_radii[target]
    contribution = coordination_pair_term(distance, covalent_sum, cutoff)
    coordination = np.zeros(positions.shape[0], dtype=np.float64)
    np.add.at(coordination, source, contribution)
    return coordination


def _channel_arrays(channels, factors: LowRankFactors):
    """Reference coordination numbers and validity masks per atom.

    An uncovered species (channel ``-1``) contributes nothing: every weight is zero, so
    the coefficients and their derivatives come out zero and the atom drops out of the
    dispersion sum while still counting towards its neighbours' coordination numbers.
    """
    channels = np.asarray(channels, dtype=np.int64)
    covered = channels >= 0
    safe = np.where(covered, channels, 0)
    reference = np.where(covered[:, None], factors.cn_ref[safe], 0.0)
    valid = np.where(covered[:, None], factors.valid[safe], False)
    return reference, valid


def _coefficient_pieces(theta, channels, factors: LowRankFactors):
    reference, valid = _channel_arrays(channels, factors)
    delta = theta[:, None] - reference
    weights = np.where(valid, np.exp(-K3_WEIGHT * delta**2), 0.0)
    total = weights.sum(axis=1, keepdims=True)
    safe_channels = np.where(np.asarray(channels, dtype=np.int64) >= 0, channels, 0)
    per_atom = factors.v_q[safe_channels]
    numerator = np.einsum("np,npr->nr", weights, per_atom)
    coefficients = np.divide(
        numerator, total, out=np.zeros_like(numerator), where=total > 0.0
    )
    return coefficients, weights, total, delta, per_atom


def low_rank_coefficients(theta, channels, factors: LowRankFactors):
    r"""Separable coefficients ``c_{i,\ell}`` of specification section 3.3.5.

    Invalid reference slots are masked, not merely down-weighted: the ``-1`` sentinel is
    an ordinary number under the Gaussian weight and would otherwise rescale every
    retained coefficient of its species.
    """
    theta = np.asarray(theta, dtype=np.float64)
    coefficients, _, _, _, _ = _coefficient_pieces(theta, channels, factors)
    return coefficients


def low_rank_coefficient_derivative(theta, channels, factors: LowRankFactors):
    r"""``\partial c_{i,\ell}/\partial\theta_i`` of specification section 3.3.5."""
    theta = np.asarray(theta, dtype=np.float64)
    coefficients, weights, total, delta, per_atom = _coefficient_pieces(
        theta, channels, factors
    )
    weight_derivative = -2.0 * K3_WEIGHT * delta * weights
    numerator_derivative = np.einsum("np,npr->nr", weight_derivative, per_atom)
    total_derivative = weight_derivative.sum(axis=1, keepdims=True)
    return np.divide(
        numerator_derivative - coefficients * total_derivative,
        total,
        out=np.zeros_like(numerator_derivative),
        where=total > 0.0,
    )


def reciprocal_vectors(cell, k_max):
    r"""Miller indices and wave vectors with ``|k| <= k_max``.

    ``k = 2 pi C^{-1} m`` (specification section 4.1), returned in row form as
    ``k = m @ (2 pi C^{-1})^T``. The sphere is the whole reciprocal lattice represented by
    the cutoff, including ``k = 0``.
    """
    cell = np.asarray(cell, dtype=np.float64)
    k_matrix = 2.0 * np.pi * np.linalg.inv(cell)
    # Because ``a_j . b_l = 2 pi delta_jl``, every retained index obeys the exact identity
    # ``m_j = a_j . k / (2 pi)``, hence ``|m_j| <= |a_j| k_max / (2 pi)``. That is the only
    # rigorous box: bounding with the reciprocal-basis norms instead under-sizes the box on
    # skewed cells, and a range that stops short of ``-n_j`` leaves the sphere open under
    # ``k -> -k``. Both defects silently drop reciprocal vectors.
    counts = np.ceil(np.linalg.norm(cell, axis=1) * k_max / (2.0 * np.pi)).astype(
        np.int64
    )
    ranges = [np.arange(-int(count), int(count) + 1) for count in counts]
    miller = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    wave_vectors = miller @ k_matrix.T
    norm = np.linalg.norm(wave_vectors, axis=1)
    inside = norm <= k_max
    return miller[inside], wave_vectors[inside], norm[inside]


def structure_factors(positions, channels, coefficients, wave_vectors, n_species):
    r"""``S_X^\ell(k)`` of specification section 4.2, summed explicitly."""
    positions = np.asarray(positions, dtype=np.float64)
    channels = np.asarray(channels, dtype=np.int64)
    phase = np.exp(1j * (positions @ np.asarray(wave_vectors, dtype=np.float64).T))
    rank = coefficients.shape[1]
    factors = np.zeros((n_species, rank, wave_vectors.shape[0]), dtype=complex)
    for channel in range(n_species):
        selected = channels == channel
        if not np.any(selected):
            continue
        for slot in range(rank):
            factors[channel, slot] = (
                coefficients[selected, slot][:, None] * phase[selected]
            ).sum(axis=0)
    return factors


def _kernel_tables(factors: LowRankFactors, parameters: DampingParameters, k_norm):
    """Per-species-pair kernels, radial derivatives, and the r = 0 potentials."""
    n_species = factors.n_species
    kernels = np.zeros((n_species, n_species, k_norm.size), dtype=np.float64)
    derivatives = np.zeros_like(kernels)
    zero_distance = np.zeros(n_species, dtype=np.float64)
    for a in range(n_species):
        for b in range(n_species):
            q_product = factors.q_table[a] * factors.q_table[b]
            r0 = parameters.damping_radius(q_product)
            kernels[a, b] = reciprocal_kernel(k_norm, r0, q_product, parameters)
            derivatives[a, b] = reciprocal_kernel_derivative(
                k_norm, r0, q_product, parameters
            )
        r0 = parameters.damping_radius(factors.q_table[a] ** 2)
        zero_distance[a] = pair_potential(0.0, r0, factors.q_table[a] ** 2, parameters)
    return kernels, derivatives, zero_distance


def _reciprocal_terms(
    positions, channels, coefficients, factors, parameters, wave_vectors, k_norm, volume
):
    """Reciprocal energy, its coefficient derivative, and the kernels it used.

    The ``sum_{X,Y}`` is over ordered species pairs, so the ``(X, Y)`` and ``(Y, X)``
    terms are both present; for ``X = Y`` there is exactly one term. Nothing is excluded
    from the k sum, including ``k = 0``.
    """
    kernels, derivatives, zero_distance = _kernel_tables(factors, parameters, k_norm)
    rank = factors.rank
    n_species = factors.n_species
    structure = structure_factors(
        positions, channels, coefficients, wave_vectors, n_species
    )
    phase = np.exp(1j * (positions @ wave_vectors.T))
    safe_channels = np.where(channels >= 0, channels, 0)

    total = 0.0
    for slot in range(rank):
        for a in range(n_species):
            for b in range(n_species):
                total += factors.eigs[slot] * float(
                    np.sum(
                        kernels[a, b]
                        * np.real(structure[a, slot] * np.conj(structure[b, slot]))
                    )
                )
    reciprocal = -0.5 * total / volume

    # dE/dc_i,l. The two endpoints of each ordered pair contribute equally, which supplies
    # a factor two that cancels against the 1/2 of the energy.
    coefficient_derivative = np.zeros_like(coefficients)
    for slot in range(rank):
        accumulator = np.zeros((positions.shape[0], k_norm.size), dtype=complex)
        for b in range(n_species):
            accumulator += (
                kernels[safe_channels, b] * np.conj(structure[b, slot])[None, :]
            )
        coefficient_derivative[:, slot] = (
            -factors.eigs[slot] / volume * np.real(np.sum(phase * accumulator, axis=1))
        )
    return (
        reciprocal,
        coefficient_derivative,
        structure,
        phase,
        kernels,
        derivatives,
        zero_distance,
    )


def reciprocal_energy(
    positions, channels, coefficients, factors: LowRankFactors, parameters, cell, k_max
):
    r"""``E_recip`` of specification section 4.2, without the self term."""
    cell = np.asarray(cell, dtype=np.float64)
    _, wave_vectors, k_norm = reciprocal_vectors(cell, k_max)
    volume = abs(float(np.linalg.det(cell)))
    energy, _, _, _, _, _, _ = _reciprocal_terms(
        np.asarray(positions, dtype=np.float64),
        np.asarray(channels, dtype=np.int64),
        np.asarray(coefficients, dtype=np.float64),
        factors,
        parameters,
        wave_vectors,
        k_norm,
        volume,
    )
    return energy


def self_energy(
    theta, channels, factors: LowRankFactors, parameters: DampingParameters
):
    r"""``V_self`` of specification section 4.5.

    Cancels only the ``(i = j, T = 0)`` term that the unrestricted reciprocal sum
    contains. The ``(i = j, T != 0)`` self-image terms are retained, which is why an
    isolated atom in a periodic cell has a non-zero, cell-dependent energy.
    """
    coefficients = low_rank_coefficients(theta, channels, factors)
    channels = np.asarray(channels, dtype=np.int64)
    q = factors.q_table[np.where(channels >= 0, channels, 0)]
    r0 = parameters.damping_radius(q * q)
    zero_distance = pair_potential(0.0, r0, q * q, parameters)
    return 0.5 * float(
        np.sum(factors.eigs[None, :] * coefficients**2 * zero_distance[:, None])
    )


def direct_reciprocal_energy(
    positions,
    numbers,
    covalent_radii,
    cell,
    factors: LowRankFactors,
    parameters: DampingParameters,
    coordination_cutoff,
    k_max=DEFAULT_K_MAX,
):
    r"""``E_D3 = E_recip + V_self``, with the coordination numbers computed here."""
    return evaluate(
        positions,
        numbers,
        covalent_radii,
        cell,
        factors,
        parameters,
        coordination_cutoff,
        k_max,
    ).energy


def evaluate(
    positions,
    numbers,
    covalent_radii,
    cell,
    factors: LowRankFactors,
    parameters: DampingParameters,
    coordination_cutoff,
    k_max=DEFAULT_K_MAX,
):
    """Evaluate energy, coordination numbers, and every analytic derivative at once."""
    positions = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(numbers, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.float64)
    volume = abs(float(np.linalg.det(cell)))
    # Padding atoms (atomic number 0) are skipped: they neither accumulate coordination
    # nor contribute to a neighbour's, and they carry channel -1, so they drop out of the
    # dispersion sum as well (specification sections 3.2.4 and 3.3).
    active = numbers != 0
    edges, shifts = build_directed_edges(positions, cell, coordination_cutoff, active)
    theta = coordination_numbers(
        positions, covalent_radii, cell, coordination_cutoff, edges, shifts, active
    )
    channels = factors.channels(numbers)
    # An uncovered species keeps channel -1 in the public inputs but is contracted with
    # channel 0 here: its coefficients are exactly zero, so every term it enters vanishes.
    safe_channels = np.where(channels >= 0, channels, 0)
    coefficients = low_rank_coefficients(theta, channels, factors)
    coefficient_derivative = low_rank_coefficient_derivative(theta, channels, factors)

    _, wave_vectors, k_norm = reciprocal_vectors(cell, k_max)
    reciprocal, d_energy_dc, structure, phase, kernels, derivatives, zero_distance = (
        _reciprocal_terms(
            positions,
            channels,
            coefficients,
            factors,
            parameters,
            wave_vectors,
            k_norm,
            volume,
        )
    )

    # The self term is quadratic in the coefficients, so it contributes to dE/dc and
    # therefore to the coordination chain rule. It must be folded in before the chain
    # rule runs, not applied afterwards as a scalar correction.
    self_term = 0.5 * float(
        np.sum(
            factors.eigs[None, :]
            * coefficients**2
            * zero_distance[safe_channels][:, None]
        )
    )
    d_energy_dc = (
        d_energy_dc
        + factors.eigs[None, :] * coefficients * zero_distance[safe_channels][:, None]
    )
    coordination_sensitivity = np.einsum(
        "nr,nr->n", d_energy_dc, coefficient_derivative
    )

    # Reciprocal force. Positions enter only through the structure factors, so
    # F_i = -dE/dr_i = (1/Omega) sum_k sum_l lambda_l c_i,l Re[i k e^{i k.r_i} conj S].
    forces = np.zeros_like(positions)
    for slot in range(factors.rank):
        accumulator = np.zeros((positions.shape[0], k_norm.size), dtype=complex)
        for b in range(factors.n_species):
            accumulator += (
                kernels[safe_channels, b] * np.conj(structure[b, slot])[None, :]
            )
        forces += (
            -factors.eigs[slot]
            * coefficients[:, slot][:, None]
            * np.einsum("ik,kd->id", np.imag(phase * accumulator), wave_vectors)
        )
    forces /= volume

    # Coordination chain rule (specification section 6.2): each directed edge contributes
    # to its receiver only. Accumulating on both endpoints would double count it.
    source, target = edges[:, 0], edges[:, 1]
    delta = positions[target] - positions[source] + shifts @ cell
    distance = np.linalg.norm(delta, axis=1)
    covalent_radii = np.asarray(covalent_radii, dtype=np.float64)
    d_theta = coordination_pair_term_derivative(
        distance, covalent_radii[source] + covalent_radii[target], coordination_cutoff
    )
    coordination_force = (
        (coordination_sensitivity[source] + coordination_sensitivity[target])
        * d_theta
        / distance
    )[:, None] * delta
    np.add.at(forces, source, coordination_force)

    # Strain derivative (specification section 6.3). Fractional coordinates are
    # strain-invariant, so no interpolation term appears; the reciprocal-lattice term
    # carries d|k|/du_ab = -k_a k_b / |k| and the volume term is -E_recip delta_ab.
    safe_norm = np.where(k_norm > _K_NORM_GUARD, k_norm, 1.0)
    strain_derivative = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            lattice = 0.0
            for slot in range(factors.rank):
                for x in range(factors.n_species):
                    for y in range(factors.n_species):
                        lattice += factors.eigs[slot] * float(
                            np.sum(
                                derivatives[x, y]
                                * (wave_vectors[:, a] * wave_vectors[:, b] / safe_norm)
                                * np.real(
                                    structure[x, slot] * np.conj(structure[y, slot])
                                )
                            )
                        )
            strain_derivative[a, b] = 0.5 * lattice / volume
    strain_derivative -= reciprocal * np.eye(3)
    strain_derivative += 0.5 * np.einsum("ea,eb->ab", coordination_force, delta)

    return Evaluation(
        energy=reciprocal + self_term,
        reciprocal_energy=reciprocal,
        self_energy=self_term,
        coordination=theta,
        coefficients=coefficients,
        coordination_sensitivity=coordination_sensitivity,
        forces=forces,
        strain_derivative=strain_derivative,
        edges=edges,
        shifts=shifts,
    )
