#!/usr/bin/env python3
"""Generate frozen FourierD3 direct-reciprocal fixtures (development only).

This helper is never imported by installed code. It reads the pinned NVIDIA ALCHEMI
Toolkit-Ops v0.5.0 NumPy reference implementation plus the repository's own generated
DFT-D3 tables, evaluates the *independent* real-space lattice-sum path for each fixture
geometry, and freezes the results together with full provenance.

Expected values in the emitted fixtures therefore come from two independent sources:

* the NVIDIA v0.5.0 NumPy lattice-sum oracle (``direct_lattice_energy``,
  ``modified_coordination_number``, ``low_rank_coefficients``), and
* hand-derived invariants (the one-atom retained-self-image image sum, and exact cardinal
  B-spline symbol values).

They never come from ``symmetrix/test/reference/fourier_d3.py``, which is the artifact
under test.

Usage::

    python tools/generate_fourier_d3_reference.py \
        --nvidia-source /path/to/nvalchemi-toolkit-ops \
        --output symmetrix/test/reference/fixtures

The NVIDIA checkout must be at tag ``v0.5.0`` (commit
``6702138e404d40ea79dec62f68c2fa67e858580e``); a missing or mismatched checkout is a hard
error rather than a silent skip.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import importlib.util
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np


# The supplied checkouts are immutable inputs, including when writable.
sys.dont_write_bytecode = True

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_NAME = "tools/generate_fourier_d3_reference.py"

NVIDIA_TAG = "v0.5.0"
NVIDIA_COMMIT = "6702138e404d40ea79dec62f68c2fa67e858580e"
NVIDIA_FILES = (
    "test/interactions/dispersion/_fourier_reference.py",
    "nvalchemiops/interactions/dispersion/_c6_decomposition.py",
)

SPEC_DOCUMENT = "docs/reference/fourier_d3_mathematics.md"
SPEC_REVISION_COMMIT = "6328429"
SPEC_SECTIONS = (
    "3.1",
    "3.2",
    "3.2.2",
    "3.3.5",
    "4.1",
    "4.2",
    "4.4",
    "4.5",
    "5.2",
    "6.2",
    "6.3",
)

TABLES_FILE = "libsymmetrix/source/dispersion_parameters.cpp"
TABLES_ARRAYS = (
    "d3_c6",
    "d3_reference_coordination",
    "d3_reference_count",
    "d3_rcov",
    "d3_r2r4",
)

# Functional parameters and the exact conversions of specification section 2.1.
FUNCTIONAL = "pbe"
DAMPING = "bj"
S6 = 1.0
S8 = 0.78749999999999998
A1 = 0.4289
A2_BOHR = 4.4406999999999996
BOHR_TO_ANGSTROM = 0.52917725999999998
HARTREE_BOHR6_TO_EV_ANGSTROM6 = 0.59752721065076475
A2_ANGSTROM = A2_BOHR * BOHR_TO_ANGSTROM

COORDINATION_CUTOFF_ANGSTROM = 6.0
K_MAX_PER_ANGSTROM = 24.0
DECOMPOSITION_TOLERANCE = 1e-4

# Five-point central difference stencil, as (offset, weight).
FD_STENCIL = (
    (2.0, -1.0 / 12.0),
    (1.0, 8.0 / 12.0),
    (-1.0, -8.0 / 12.0),
    (-2.0, 1.0 / 12.0),
)
FD_STEP = 1e-4

# Non-symmetric affine strain for the strained-cell pair.
STRAIN = (
    (0.020, 0.010, 0.000),
    (0.000, -0.015, 0.005),
    (0.004, 0.000, 0.030),
)


@dataclass(frozen=True)
class Case:
    name: str
    coverage: tuple[str, ...]
    numbers: tuple[int, ...]
    positions: tuple[tuple[float, float, float], ...]
    cell: tuple[tuple[float, float, float], ...]
    real_cutoffs: tuple[float, float]
    derivatives: bool = True
    pair: str | None = None
    strain: tuple[tuple[float, float, float], ...] | None = None


CASES = (
    Case(
        name="single-atom-self-images",
        coverage=("periodic", "single-atom", "self-images", "hand-derived-invariant"),
        numbers=(6,),
        positions=((0.0, 0.0, 0.0),),
        cell=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        real_cutoffs=(80.0, 160.0),
    ),
    Case(
        name="orthogonal-mixed-species",
        coverage=("periodic", "orthogonal", "mixed-element"),
        numbers=(1, 6, 8, 6),
        positions=((0.0, 0.0, 0.0), (1.09, 0.0, 0.0), (3.0, 2.0, 1.0), (4.5, 3.5, 2.0)),
        cell=((8.0, 0.0, 0.0), (0.0, 7.0, 0.0), (0.0, 0.0, 9.0)),
        real_cutoffs=(80.0, 160.0),
    ),
    Case(
        name="triclinic-mixed-species",
        coverage=("periodic", "triclinic", "mixed-element"),
        numbers=(1, 6, 8, 6),
        positions=(
            (0.0, 0.0, 0.0),
            (1.09, 0.05, 0.0),
            (3.0, 2.0, 1.0),
            (4.5, 3.5, 2.0),
        ),
        cell=((7.5, 0.0, 0.0), (1.2, 6.8, 0.0), (0.8, -1.1, 8.2)),
        real_cutoffs=(80.0, 160.0),
    ),
    Case(
        name="coordination-sensitive",
        coverage=("periodic", "mixed-element", "coordination-contrast"),
        numbers=(6, 1, 1, 1, 8),
        positions=(
            (0.0, 0.0, 0.0),
            (1.09, 0.0, 0.0),
            (-0.545, 0.944, 0.0),
            (-0.545, -0.944, 0.0),
            (6.0, 6.0, 6.0),
        ),
        cell=((12.0, 0.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 12.0)),
        real_cutoffs=(80.0, 160.0),
    ),
)


def strained_pair() -> tuple[Case, Case]:
    """Reference and affinely strained states of one geometry.

    The strained state applies ``r' = r (I + u)`` and ``C' = C (I + u)`` with the
    non-symmetric ``STRAIN`` tensor, so the pair exercises both the cell change and the
    coordinate change under one deformation convention.
    """
    base = Case(
        name="strained-cell-reference",
        coverage=("periodic", "orthogonal", "mixed-element", "strained-pair-reference"),
        numbers=(1, 6, 8, 6),
        positions=((0.0, 0.0, 0.0), (1.09, 0.0, 0.0), (3.0, 2.0, 1.0), (4.5, 3.5, 2.0)),
        cell=((8.0, 0.0, 0.0), (0.0, 7.0, 0.0), (0.0, 0.0, 9.0)),
        real_cutoffs=(80.0, 160.0),
        pair="strained-cell",
    )
    deformation = np.eye(3) + np.asarray(STRAIN, dtype=np.float64)
    positions = np.asarray(base.positions, dtype=np.float64) @ deformation
    cell = np.asarray(base.cell, dtype=np.float64) @ deformation
    strained = Case(
        name="strained-cell-strained",
        coverage=("periodic", "orthogonal", "mixed-element", "strained-pair-strained"),
        numbers=base.numbers,
        positions=tuple(tuple(float(value) for value in row) for row in positions),
        cell=tuple(tuple(float(value) for value in row) for row in cell),
        real_cutoffs=(80.0, 160.0),
        derivatives=False,
        pair="strained-cell",
        strain=STRAIN,
    )
    return base, strained


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_nvidia(source: Path) -> tuple[ModuleType, ModuleType, dict[str, str]]:
    """Load the pinned NVIDIA reference modules and verify their identity.

    The package ``__init__`` imports Warp, so the two pure-NumPy modules are loaded from
    their files and a minimal namespace shim is registered: ``_fourier_reference`` imports
    ``nvalchemiops.interactions.dispersion._c6_decomposition`` inside
    ``low_rank_coefficients``.
    """
    if not source.is_dir():
        raise FileNotFoundError(
            f"NVIDIA ALCHEMI Toolkit-Ops checkout not found at {source}; "
            "Reference fixtures cannot be regenerated without it. Pass --nvidia-source."
        )
    hashes = {}
    for relative in NVIDIA_FILES:
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned NVIDIA source file: {path}")
        hashes[relative] = sha256(path)

    for name, path in (
        ("nvalchemiops", source / "nvalchemiops"),
        ("nvalchemiops.interactions", source / "nvalchemiops" / "interactions"),
        (
            "nvalchemiops.interactions.dispersion",
            source / "nvalchemiops" / "interactions" / "dispersion",
        ),
    ):
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    decomposition = _load_module(
        "nvalchemiops.interactions.dispersion._c6_decomposition",
        source
        / "nvalchemiops"
        / "interactions"
        / "dispersion"
        / "_c6_decomposition.py",
    )
    reference = _load_module(
        "symmetrix_fixture_nvidia_reference",
        source / "test" / "interactions" / "dispersion" / "_fourier_reference.py",
    )
    return reference, decomposition, hashes


def parse_tables(path: Path) -> dict[str, np.ndarray]:
    """Read the repository's generated DFT-D3 tables out of the C++ source.

    FourierD3 must reuse these tables unchanged (specification section 2.1), so the
    fixtures are generated from them rather than from a second copy of the parameter set.
    """
    text = path.read_text(encoding="utf-8")
    tables = {}
    for name in TABLES_ARRAYS:
        match = re.search(
            r"constexpr std::array(?:<[^>]*>)? " + name + r"_data = \{(.*?)\n\};",
            text,
            re.DOTALL,
        )
        if match is None:
            raise RuntimeError(f"cannot find generated table {name}_data in {path}")
        values = [
            float(item)
            for item in match.group(1).replace("\n", " ").split(",")
            if item.strip()
        ]
        tables[name] = np.asarray(values, dtype=np.float64)
    tables["d3_c6"] = tables["d3_c6"].reshape(95, 95, 5, 5, 3)
    tables["d3_reference_coordination"] = tables["d3_reference_coordination"].reshape(
        95, 5
    )
    tables["d3_reference_count"] = tables["d3_reference_count"].astype(np.int64)
    return tables


@dataclass(frozen=True)
class Factors:
    decomposition: object
    species: np.ndarray
    eigs: np.ndarray
    v_q: np.ndarray
    cn_ref: np.ndarray
    valid: np.ndarray
    species_map: np.ndarray
    q_table: np.ndarray


def sign_normalise(v_q: np.ndarray) -> np.ndarray:
    """Apply the specification section 3.3.4 sign gauge to each retained eigenvector."""
    normalised = np.array(v_q, dtype=np.float64, copy=True)
    for slot in range(normalised.shape[-1]):
        flat = normalised[:, :, slot].reshape(-1)
        if flat[int(np.argmax(np.abs(flat)))] < 0.0:
            normalised[:, :, slot] = -normalised[:, :, slot]
    return normalised


def build_factors(
    decomposition_module, c6ab, cnref_pair, species, q_table_all
) -> Factors:
    decomposition = decomposition_module.decompose_c6_reference(
        c6ab, cnref_pair, species=species, tol=DECOMPOSITION_TOLERANCE
    )
    return Factors(
        decomposition=decomposition,
        species=decomposition.species,
        eigs=decomposition.eigs,
        v_q=sign_normalise(decomposition.v_q),
        cn_ref=decomposition.cn_ref,
        valid=decomposition.valid,
        species_map=decomposition.species_map,
        q_table=q_table_all[decomposition.species],
    )


def build_edges(positions, cell, cutoff):
    """Brute-force directed edge list: both orientations, self-images included."""
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    widths = np.abs(np.linalg.det(cell)) / np.linalg.norm(
        np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1
    )
    reach = [
        int(np.ceil((cutoff + 2.0 * float(np.abs(positions).max())) / width))
        for width in widths
    ]
    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in reach], indexing="ij")
    translations = np.stack(grids, axis=-1).reshape(-1, 3) @ cell
    inverse_cell = np.linalg.inv(cell)
    edges = []
    shifts = []
    for translation in translations:
        delta = positions[None, :, :] + translation - positions[:, None, :]
        distance = np.linalg.norm(delta, axis=-1)
        inside = (distance > 0.0) & (distance < cutoff)
        if not np.any(translation):
            np.fill_diagonal(inside, False)
        source, target = np.nonzero(inside)
        edges.append(np.stack([source, target], axis=1))
        shifts.append(np.tile(translation @ inverse_cell, (source.size, 1)))
    return (
        np.concatenate(edges).astype(np.int64),
        np.rint(np.concatenate(shifts)).astype(np.int64),
    )


def parameters():
    return dict(s6=S6, s8=S8, a1=A1, a2=A2_ANGSTROM)


class NvidiaLatticeSum:
    """The independent NVIDIA real-space lattice-sum evaluation."""

    def __init__(
        self, reference, decomposition_module, tables, c6ab, cnref_pair, q_table_all
    ):
        self.reference = reference
        self.decomposition_module = decomposition_module
        self.rcov = tables["d3_rcov"]
        self.c6ab = c6ab
        self.cnref_pair = cnref_pair
        self.q_table_all = q_table_all
        self.params = parameters()
        self._factors: dict[tuple[int, ...], Factors] = {}

    def factors(self, numbers) -> Factors:
        species = tuple(sorted(set(int(z) for z in numbers)))
        if species not in self._factors:
            self._factors[species] = build_factors(
                self.decomposition_module,
                self.c6ab,
                self.cnref_pair,
                species,
                self.q_table_all,
            )
        return self._factors[species]

    def coordination(
        self, positions, numbers, cell, cutoff=COORDINATION_CUTOFF_ANGSTROM
    ):
        edges, shifts = build_edges(positions, cell, cutoff)
        return self.reference.modified_coordination_number(
            np.asarray(positions, dtype=np.float64),
            np.asarray(numbers, dtype=np.int64),
            self.rcov,
            edges,
            shifts,
            np.asarray(cell, dtype=np.float64),
            cutoff,
        )

    def energy(self, positions, numbers, cell, real_cutoff):
        factors = self.factors(numbers)
        theta = self.coordination(positions, numbers, cell)
        channels = factors.species_map[np.asarray(numbers, dtype=np.int64)]
        coefficients, _ = self.reference.low_rank_coefficients(
            theta, channels, factors.decomposition
        )
        return float(
            self.reference.direct_lattice_energy(
                np.asarray(positions, dtype=np.float64),
                channels,
                coefficients,
                factors.decomposition,
                factors.q_table,
                np.asarray(cell, dtype=np.float64),
                self.params["s6"],
                self.params["s8"],
                self.params["a1"],
                self.params["a2"],
                real_cutoff,
            )
        )

    def energy_from_coordination(self, theta, positions, numbers, cell, real_cutoff):
        factors = self.factors(numbers)
        channels = factors.species_map[np.asarray(numbers, dtype=np.int64)]
        coefficients, _ = self.reference.low_rank_coefficients(
            theta, channels, factors.decomposition
        )
        return float(
            self.reference.direct_lattice_energy(
                np.asarray(positions, dtype=np.float64),
                channels,
                coefficients,
                factors.decomposition,
                factors.q_table,
                np.asarray(cell, dtype=np.float64),
                self.params["s6"],
                self.params["s8"],
                self.params["a1"],
                self.params["a2"],
                real_cutoff,
            )
        )

    def finite_difference(self, mutate, shape):
        derivative = np.zeros(shape)
        flat = derivative.reshape(-1)
        for index in range(flat.size):
            total = 0.0
            for offset, weight in FD_STENCIL:
                total += weight * mutate(index, offset * FD_STEP)
            flat[index] = total / FD_STEP
        return derivative

    def forces(self, positions, numbers, cell, real_cutoff):
        positions = np.asarray(positions, dtype=np.float64)

        def mutate(index, step):
            moved = positions.copy()
            moved[index // 3, index % 3] += step
            return self.energy(moved, numbers, cell, real_cutoff)

        return -self.finite_difference(mutate, positions.shape)

    def strain_derivative(self, positions, numbers, cell, real_cutoff):
        positions = np.asarray(positions, dtype=np.float64)
        cell = np.asarray(cell, dtype=np.float64)

        def mutate(index, step):
            strain = np.zeros((3, 3))
            strain[index // 3, index % 3] = step
            deformation = np.eye(3) + strain
            return self.energy(
                positions @ deformation, numbers, cell @ deformation, real_cutoff
            )

        return self.finite_difference(mutate, (3, 3))

    def coordination_sensitivity(self, positions, numbers, cell, real_cutoff):
        theta = self.coordination(positions, numbers, cell)

        def mutate(index, step):
            perturbed = theta.copy()
            perturbed[index] += step
            return self.energy_from_coordination(
                perturbed, positions, numbers, cell, real_cutoff
            )

        return self.finite_difference(mutate, theta.shape)


def retained_self_image_energy(cell, q, coefficients, eigs, radius):
    """Hand-derived one-atom lattice invariant.

    For a single atom the model reduces to its interaction with its own periodic images,

        E = -1/2 * sum_l lambda_l c_l^2 * sum_{T != 0} phi_ZZ(|T|),

    with ``phi`` the damped pair potential. The image sum is enumerated directly and the
    remainder beyond ``radius`` is estimated by the same uniform-density integral the
    NVIDIA oracle uses, ``4*pi*s6/(3*Omega*R^3) + 12*pi*s8*q^2/(5*Omega*R^5)``.
    """
    cell = np.asarray(cell, dtype=np.float64)
    volume = abs(float(np.linalg.det(cell)))
    widths = np.linalg.norm(cell, axis=1)
    reach = int(np.ceil(radius / float(widths.min())))
    grid = np.stack(
        np.meshgrid(*[np.arange(-reach, reach + 1) for _ in range(3)], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    translations = grid @ cell
    distance = np.linalg.norm(translations, axis=1)
    inside = (distance > 0.0) & (distance <= radius)
    r0 = A1 * np.sqrt(3.0 * q * q) + A2_ANGSTROM
    potential = np.where(
        inside,
        S6 / (np.where(inside, distance, 1.0) ** 6 + r0**6)
        + 3.0 * S8 * q * q / (np.where(inside, distance, 1.0) ** 8 + r0**8),
        0.0,
    )
    lattice = float(potential.sum())
    tail = 4.0 * np.pi * S6 / (3.0 * volume * radius**3) + 12.0 * np.pi * S8 * q * q / (
        5.0 * volume * radius**5
    )
    weight = float(np.sum(np.asarray(eigs) * np.asarray(coefficients) ** 2))
    return -0.5 * weight * (lattice + tail)


def cardinal_bspline_values(order: int) -> tuple[float, ...]:
    """Cardinal B-spline ``M_p`` sampled at the integers ``0..p-1``.

    These are the Eulerian-number ratios ``A(p-1, t)/(p-1)!``; the fixture records them as
    hand-derived exact values so the reference module's Cox-de Boor recursion is checked
    against an independent definition.
    """
    from math import factorial

    degree = order - 1
    eulerian = [[0] * (degree + 1) for _ in range(degree + 1)]
    for n in range(degree + 1):
        for k in range(n + 1):
            if k == 0:
                eulerian[n][k] = 1
            else:
                eulerian[n][k] = (n - k) * eulerian[n - 1][k - 1] + (k + 1) * eulerian[
                    n - 1
                ][k]
    return tuple(eulerian[degree][t] / factorial(degree) for t in range(order))


def cardinal_bspline(order: int, x: np.ndarray) -> np.ndarray:
    """Cardinal B-spline ``M_p(x)`` with knots at the integers, support ``[0, p]``."""
    x = np.asarray(x, dtype=np.float64)
    knots = np.arange(0, order + 1, dtype=np.float64)

    def recursion(i: int, k: int) -> np.ndarray:
        if k == 1:
            return ((x >= knots[i]) & (x < knots[i + 1])).astype(np.float64)
        total = np.zeros_like(x)
        left = knots[i + k - 1] - knots[i]
        right = knots[i + k] - knots[i + 1]
        if left > 0.0:
            total += (x - knots[i]) / left * recursion(i, k - 1)
        if right > 0.0:
            total += (knots[i + k] - x) / right * recursion(i + 1, k - 1)
        return total

    return recursion(0, order)


def exact_attenuation(
    order: int, mesh_size: int, fraction: float, miller: np.ndarray
) -> np.ndarray:
    """Exact per-axis attenuation from the closed-form piecewise polynomial.

    One atom at fractional coordinate ``fraction`` is assigned the centred cardinal
    B-spline ``B_p(x - y)`` sampled at the mesh nodes, so its discrete Fourier coefficient
    is a finite sum of ``B_p`` values. ``B_p`` is evaluated here from its truncated-power
    closed form

        B_p(x) = 1/(p-1)! sum_{j=0}^{p} (-1)^j C(p,j) max(x + p/2 - j, 0)^(p-1),

    which shares no algorithm with the module under test: no Cox-de Boor recursion, no
    stencil offsets, no node placement, no wrapping. This is the binding expectation; the
    stencil-based route is recorded beside it only for self-consistency.
    """
    from math import comb, factorial

    y = mesh_size * fraction
    # The support of B_p(. - y) spans [y - p/2, y + p/2], so nodes just outside the first
    # period carry weight too; they fold onto the far end of the mesh, exactly as the
    # periodic assignment does.
    nodes = np.arange(-order, mesh_size + order, dtype=np.int64)
    x = nodes.astype(np.float64) - y
    value = np.zeros(nodes.size, dtype=np.float64)
    for j in range(order + 1):
        shifted = np.maximum(x + order / 2.0 - j, 0.0)
        value += (-1.0) ** j * comb(order, j) * shifted ** (order - 1)
    value /= factorial(order - 1)
    field = np.zeros(mesh_size, dtype=np.float64)
    np.add.at(field, np.mod(nodes, mesh_size), value)
    return np.abs(
        np.sum(
            field
            * np.exp(
                -2j
                * np.pi
                * np.asarray(miller)[..., None]
                * np.arange(mesh_size, dtype=np.float64)[None, :]
                / mesh_size
            ),
            axis=-1,
        )
    )


def spread_attenuation(
    order: int, mesh_size: int, fraction: float, miller: np.ndarray
) -> np.ndarray:
    """Exact per-axis attenuation of the specification section 5.1 assignment.

    Places one atom at fractional coordinate ``fraction`` on a periodic 1D mesh of
    ``mesh_size`` nodes, spreads it with the documented cardinal B-spline stencil, and
    takes the explicit (non-FFT) discrete Fourier coefficient of the resulting field.
    """
    y = mesh_size * fraction
    base = int(np.floor(y))
    theta = y - base
    start = int(np.floor(theta - (order - 2) / 2))
    field = np.zeros(mesh_size)
    for i in range(order):
        offset = start + i
        field[(base + offset) % mesh_size] += float(
            cardinal_bspline(order, np.array([order / 2.0 + theta - offset]))[0]
        )
    nodes = np.arange(mesh_size, dtype=np.float64)
    return np.abs(
        np.sum(
            field
            * np.exp(
                -2j * np.pi * np.asarray(miller)[..., None] * nodes[None, :] / mesh_size
            ),
            axis=-1,
        )
    )


def discrete_symbol(order: int, miller: np.ndarray, mesh_size: int) -> np.ndarray:
    samples = np.asarray(cardinal_bspline_values(order), dtype=np.float64)
    t = np.arange(order, dtype=np.float64)
    return np.abs(
        np.sum(
            samples
            * np.exp(
                2j * np.pi * np.asarray(miller)[..., None] * t[None, :] / mesh_size
            ),
            axis=-1,
        )
    )


def continuous_symbol(order: int, miller: np.ndarray, mesh_size: int) -> np.ndarray:
    return np.abs(np.sinc(np.asarray(miller, dtype=np.float64) / mesh_size)) ** order


SYMBOL_PROBES = ((2, 8), (4, 8), (4, 12), (6, 8))
SYMBOL_FRACTIONS = (0.0, 0.3719)


def symbol_fixture() -> dict:
    probes = []
    for order, mesh_size in SYMBOL_PROBES:
        miller = np.arange(0, mesh_size // 2 + 1)
        for fraction in SYMBOL_FRACTIONS:
            measured = spread_attenuation(order, mesh_size, fraction, miller)
            exact = exact_attenuation(order, mesh_size, fraction, miller)
            discrepancy = float(np.abs(measured - exact).max())
            # The truncated-power form cancels heavily near the knots, so the two routes
            # agree only to about 1e-12 in double precision; a real stencil or recursion
            # defect would be orders of magnitude larger than that.
            if discrepancy > 1e-11:
                raise RuntimeError(
                    "independent closed form disagrees with the stencil route for "
                    f"p={order} N={mesh_size} f={fraction}: {discrepancy:.3e}"
                )
            probes.append(
                {
                    "spline_order": order,
                    "mesh_size": mesh_size,
                    "cell_length_angstrom": float(mesh_size),
                    "spacing_angstrom": 1.0,
                    "fractional_coordinate": fraction,
                    "miller": miller.tolist(),
                    "exact_attenuation": exact.tolist(),
                    "discrete_symbol_closed_form": exact_attenuation(
                        order, mesh_size, 0.0, miller
                    ).tolist(),
                    "measured_attenuation": measured.tolist(),
                    "discrete_symbol": discrete_symbol(
                        order, miller, mesh_size
                    ).tolist(),
                    "continuous_symbol": continuous_symbol(
                        order, miller, mesh_size
                    ).tolist(),
                }
            )
    return {
        "schema": "symmetrix.fourier-d3-bspline-symbol/1",
        "adopted_convention": "continuous",
        "adopted_convention_expression": "abs(sinc(m/N))**p",
        "adopted_convention_rationale": (
            "specification section 5.2: v1 follows the NVIDIA v0.5.0 continuous cardinal "
            "B-spline symbol because the acceptance criterion is v0.5.0 fixture agreement"
        ),
        "open_item": (
            "specification section 5.2: the discrete symbol is the exact attenuation only "
            "when the stencil aligns with integer node offsets; for a generic fractional "
            "coordinate the assignment attenuation matches neither closed form exactly"
        ),
        "cardinal_bspline_integer_values": {
            str(order): list(cardinal_bspline_values(order)) for order in range(2, 7)
        },
        "binding_expectations": {
            "cardinal_bspline_integer_values": (
                "exact Eulerian-number ratios A(p-1, t)/(p-1)!, derived by integer "
                "recursion rather than by the module's Cox-de Boor recursion"
            ),
            "exact_attenuation": (
                "exact DFT of the centred cardinal B-spline sampled at the mesh nodes, "
                "evaluated from its truncated-power closed form"
            ),
            "discrete_symbol_closed_form": (
                "the same independent route at an on-node fractional coordinate, which is "
                "where the discrete symbol is exact"
            ),
        },
        "non_binding_expectations": {
            "measured_attenuation": (
                "self-consistency only: this generator's stencil placement and Cox-de Boor "
                "recursion duplicate the module under test, so agreement is not evidence"
            ),
            "discrete_symbol": (
                "self-consistency only: sampled-stencil DFT, the same construction as the "
                "module under test"
            ),
            "continuous_symbol": (
                "self-consistency only: abs(sinc(m/N))**p, the same expression as the "
                "module under test"
            ),
        },
        "probes": probes,
    }


def case_record(case: Case, path: NvidiaLatticeSum, tables) -> dict:
    numbers = np.asarray(case.numbers, dtype=np.int64)
    positions = np.asarray(case.positions, dtype=np.float64)
    cell = np.asarray(case.cell, dtype=np.float64)
    factors = path.factors(numbers)
    theta = path.coordination(positions, numbers, cell)
    low, high = case.real_cutoffs

    energy_low = path.energy(positions, numbers, cell, low)
    energy_high = path.energy(positions, numbers, cell, high)

    record = {
        "name": case.name,
        "coverage": list(case.coverage),
        "atomic_numbers": list(case.numbers),
        "positions_angstrom": positions.tolist(),
        "cell_angstrom": cell.tolist(),
        "covalent_radii_angstrom": tables["d3_rcov"][numbers].tolist(),
        "q_table_angstrom": factors.q_table.tolist(),
        "decomposition": {
            "species": factors.species.tolist(),
            "eigs_ev_angstrom6": factors.eigs.tolist(),
            "v_q": factors.v_q.tolist(),
            "cn_ref": factors.cn_ref.tolist(),
            "valid": factors.valid.tolist(),
            "max_relative_error": float(factors.decomposition.max_relative_error),
            "gauge": "sign-normalised per specification 3.3.4",
        },
        "expected": {
            "coordination": theta.tolist(),
            "energy_ev": energy_high,
            "energy_convergence_band_ev": abs(energy_high - energy_low),
        },
    }
    if case.derivatives:
        forces_high = path.forces(positions, numbers, cell, high)
        forces_low = path.forces(positions, numbers, cell, low)
        strain_high = path.strain_derivative(positions, numbers, cell, high)
        strain_low = path.strain_derivative(positions, numbers, cell, low)
        chi_high = path.coordination_sensitivity(positions, numbers, cell, high)
        chi_low = path.coordination_sensitivity(positions, numbers, cell, low)
        record["expected"].update(
            {
                "coordination_sensitivity_ev": chi_high.tolist(),
                "forces_ev_per_angstrom": forces_high.tolist(),
                "strain_derivative_ev": strain_high.tolist(),
            }
        )
        record["finite_difference"] = {
            "step": FD_STEP,
            "stencil": "five-point central",
            "coordination_sensitivity_band": float(np.abs(chi_high - chi_low).max()),
            "forces_band": float(np.abs(forces_high - forces_low).max()),
            "strain_derivative_band": float(np.abs(strain_high - strain_low).max()),
        }
    if case.pair is not None:
        record["pair"] = case.pair
    if case.strain is not None:
        record["applied_strain"] = [list(row) for row in case.strain]
        record["strain_convention"] = "r' = r (I + u), C' = C (I + u)"

    if case.name == "single-atom-self-images":
        q = float(factors.q_table[0])
        coefficients = path.reference.low_rank_coefficients(
            theta, factors.species_map[numbers], factors.decomposition
        )[0]
        invariant = {
            str(radius): retained_self_image_energy(
                cell, q, coefficients[0], factors.eigs, radius
            )
            for radius in (40.0, 80.0, 160.0)
        }
        record["hand_derived_invariant"] = {
            "expression": "-1/2 sum_l lambda_l c_l^2 sum_{T != 0} phi_ZZ(|T|)",
            "radii_angstrom": sorted(float(key) for key in invariant),
            "energy_ev": invariant,
            "band_ev": abs(invariant["160.0"] - invariant["80.0"]),
            "note": (
                "direct enumeration of periodic images plus the uniform-density remainder; "
                "independent of both the reciprocal oracle and the NVIDIA pair sum"
            ),
        }
    return record


def generate(nvidia_source: Path, output: Path, repository_root: Path) -> None:
    reference, decomposition_module, nvidia_hashes = load_nvidia(nvidia_source)
    tables_path = repository_root / TABLES_FILE
    tables = parse_tables(tables_path)

    raw_c6 = tables["d3_c6"][..., 0]
    valid = raw_c6 >= 0.0
    c6ab = np.where(valid, raw_c6, 0.0)
    cnref_pair = np.where(valid, tables["d3_c6"][..., 1], 0.0)

    path = NvidiaLatticeSum(
        reference, decomposition_module, tables, c6ab, cnref_pair, tables["d3_r2r4"]
    )

    spec_path = repository_root / SPEC_DOCUMENT
    generator_path = repository_root / GENERATOR_NAME
    cases = [
        case_record(case, path, tables)
        for case in (*CASES[:3], *strained_pair(), CASES[3])
    ]

    fixture = {
        "schema": "symmetrix.fourier-d3-reference/1",
        "specification": {
            "document": SPEC_DOCUMENT,
            "sha256": sha256(spec_path),
            "revision_commit": SPEC_REVISION_COMMIT,
            "sections": list(SPEC_SECTIONS),
        },
        "provenance": {
            "generator": GENERATOR_NAME,
            "generator_sha256": sha256(generator_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "upstream": {
                "project": "NVIDIA ALCHEMI Toolkit-Ops",
                "tag": NVIDIA_TAG,
                "commit": NVIDIA_COMMIT,
                "files": nvidia_hashes,
            },
            "tables": {
                "file": TABLES_FILE,
                "sha256": sha256(tables_path),
                "arrays": list(TABLES_ARRAYS),
                "note": (
                    "the generated Symmetrix tables are the single source of the D3 "
                    "parameters; FourierD3 must not introduce a second copy"
                ),
            },
            "units": {
                "length": "angstrom",
                "energy": "electronvolt",
                "force": "electronvolt / angstrom",
                "c6": "electronvolt * angstrom**6",
                "q_table": "angstrom (r2r4 table, already scaled by bohr_to_angstrom)",
                "reciprocal_length": "1 / angstrom",
                "strain_derivative": "electronvolt",
                "conversion_bohr_to_angstrom": BOHR_TO_ANGSTROM,
                "conversion_hartree_bohr6_to_ev_angstrom6": HARTREE_BOHR6_TO_EV_ANGSTROM6,
            },
            "notes": [
                "expected values come from the NVIDIA v0.5.0 NumPy real-space lattice-sum "
                "oracle and from hand-derived invariants, never from the oracle under test",
                "energies are the physical lattice-sum energies; the reciprocal sum of "
                "specification 4.2 is the same quantity expressed in reciprocal space",
                "the NVIDIA reference requires invalid C6 reference slots to be zero, while "
                "the generated table marks them with -1; the generator zeroes them",
            ],
        },
        "parameters": {
            "functional": FUNCTIONAL,
            "damping": DAMPING,
            "s6": S6,
            "s8": S8,
            "a1": A1,
            "a2_bohr": A2_BOHR,
            "a2_angstrom": A2_ANGSTROM,
            "coordination_cutoff_angstrom": COORDINATION_CUTOFF_ANGSTROM,
            "k_max_per_angstrom": K_MAX_PER_ANGSTROM,
            "decomposition_tolerance": DECOMPOSITION_TOLERANCE,
            "decomposition_max_rank": None,
            "edge_rule": "0 < r < coordination_cutoff_angstrom, both orientations, self-images retained",
        },
        "cases": cases,
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "fourier_d3_reference_cases.json").write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "bspline_symbol_cases.json").write_text(
        json.dumps(symbol_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvidia-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        generate(
            arguments.nvidia_source.resolve(),
            arguments.output.resolve(),
            REPOSITORY_ROOT,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
