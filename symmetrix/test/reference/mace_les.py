"""Dependency-free monopole LES reference used by native integration tests.

The frozen inputs were generated with ChengUCB/les at the commit recorded in
the fixture.  This module deliberately reimplements only its scalar-charge
real-space and reciprocal-space equations in NumPy, so loading a fixture never
imports PyTorch, MACE, or LES.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import erf
from pathlib import Path
from typing import Literal

import numpy as np

MACE_COMMIT = "7913afb6fa729878d20886b84a2079523727d958"
LES_COMMIT = "fadfb0ca3b89ed371e5de6b383f4425defabd4a8"
COULOMB_NORM_FACTOR = 90.4756
_TWO_PI = 2.0 * np.pi
_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "mace_les_monopole_cases.json"


@dataclass(frozen=True)
class MACELESCase:
    """One frozen upstream scalar-charge LES calculation."""

    name: str
    boundary: Literal["nonperiodic", "periodic"]
    mace_commit: str
    les_commit: str
    sigma: float
    dl: float
    remove_self_interaction: bool
    charges: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    energy: float
    atomic_energies: np.ndarray
    forces: np.ndarray
    stress: np.ndarray | None
    charge_adjoints: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "charges", np.asarray(self.charges, dtype=np.float64))
        object.__setattr__(
            self, "positions", np.asarray(self.positions, dtype=np.float64)
        )
        object.__setattr__(self, "cell", np.asarray(self.cell, dtype=np.float64))
        object.__setattr__(
            self, "atomic_energies", np.asarray(self.atomic_energies, dtype=np.float64)
        )
        object.__setattr__(self, "forces", np.asarray(self.forces, dtype=np.float64))
        object.__setattr__(
            self,
            "stress",
            None if self.stress is None else np.asarray(self.stress, dtype=np.float64),
        )
        object.__setattr__(
            self, "charge_adjoints", np.asarray(self.charge_adjoints, dtype=np.float64)
        )


def load_mace_les_cases() -> tuple[MACELESCase, ...]:
    """Load immutable NumPy views of the upstream-generated fixture values."""
    raw_cases = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
    cases = []
    for raw in raw_cases:
        stress = raw.get("stress")
        cases.append(
            MACELESCase(
                name=raw["name"],
                boundary=raw["boundary"],
                mace_commit=raw["mace_commit"],
                les_commit=raw["les_commit"],
                sigma=float(raw["sigma"]),
                dl=float(raw["dl"]),
                remove_self_interaction=bool(raw["remove_self_interaction"]),
                charges=np.asarray(raw["charges"], dtype=np.float64),
                positions=np.asarray(raw["positions"], dtype=np.float64),
                cell=np.asarray(raw["cell"], dtype=np.float64),
                energy=float(raw["energy"]),
                atomic_energies=np.asarray(raw["atomic_energies"], dtype=np.float64),
                forces=np.asarray(raw["forces"], dtype=np.float64),
                stress=None if stress is None else np.asarray(stress, dtype=np.float64),
                charge_adjoints=np.asarray(raw["charge_adjoints"], dtype=np.float64),
            )
        )
    return tuple(cases)


def nonperiodic_monopole_oracle(case: MACELESCase) -> dict[str, np.ndarray | float]:
    """Evaluate LES's Gaussian-smoothed, nonperiodic charge interaction."""
    if case.boundary != "nonperiodic":
        raise ValueError(f"{case.name} is not a nonperiodic case")
    positions, charges = case.positions, case.charges
    delta = positions[None, :, :] - positions[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    mask = ~np.eye(charges.size, dtype=bool)
    kernel = np.zeros_like(distance)
    scaled = distance[mask] / (np.sqrt(2.0) * case.sigma)
    kernel[mask] = (
        COULOMB_NORM_FACTOR / _TWO_PI * np.vectorize(erf)(scaled) / distance[mask]
    )
    potential = kernel @ charges
    energy = 0.5 * float(charges @ potential)
    atomic_energies = 0.5 * charges * potential

    force = np.zeros_like(positions)
    for i in range(charges.size):
        for j in range(i + 1, charges.size):
            vector = positions[j] - positions[i]
            radius = float(np.linalg.norm(vector))
            scaled_radius = radius / (np.sqrt(2.0) * case.sigma)
            derivative = (
                COULOMB_NORM_FACTOR
                / _TWO_PI
                * (
                    np.sqrt(2.0 / np.pi)
                    * np.exp(-scaled_radius * scaled_radius)
                    / (case.sigma * radius)
                    - erf(scaled_radius) / (radius * radius)
                )
            )
            contribution = charges[i] * charges[j] * derivative * vector / radius
            force[i] += contribution
            force[j] -= contribution

    if not case.remove_self_interaction:
        self_factor = COULOMB_NORM_FACTOR / (case.sigma * _TWO_PI**1.5)
        energy += self_factor * float(charges @ charges)
        atomic_energies += self_factor * charges * charges
        potential += 2.0 * self_factor * charges

    return {
        "energy": energy,
        "atomic_energies": atomic_energies,
        "forces": force,
        "charge_adjoints": potential,
        "strain_derivative": np.zeros((3, 3), dtype=np.float64),
    }


def periodic_monopole_oracle(case: MACELESCase) -> dict[str, np.ndarray | float]:
    """Evaluate LES's explicit upstream hemisphere reciprocal sum."""
    if case.boundary != "periodic":
        raise ValueError(f"{case.name} is not a periodic case")
    return _periodic_monopole(case.positions, case.cell, case.charges, case)


def periodic_monopole_oracle_arrays(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    charges: np.ndarray,
    sigma: float,
    dl: float,
    remove_self_interaction: bool,
) -> dict[str, np.ndarray | float]:
    """Evaluate a generated periodic case without relying on frozen fixtures."""
    case = MACELESCase(
        name="generated",
        boundary="periodic",
        mace_commit=MACE_COMMIT,
        les_commit=LES_COMMIT,
        sigma=sigma,
        dl=dl,
        remove_self_interaction=remove_self_interaction,
        charges=charges,
        positions=positions,
        cell=cell,
        energy=0.0,
        atomic_energies=np.empty(0),
        forces=np.empty((0, 3)),
        stress=None,
        charge_adjoints=np.empty(0),
    )
    return _periodic_monopole(positions, cell, charges, case)


def periodic_particle_mesh_oracle_arrays(
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    charges: np.ndarray,
    sigma: float,
    dl: float,
    remove_self_interaction: bool,
    mesh: tuple[int, int, int],
    spline_order: int,
) -> dict[str, np.ndarray | float]:
    """Evaluate the LES particle-mesh discretization with NumPy FFTs.

    This is a same-mesh oracle: it intentionally reproduces the cardinal
    assignment and continuous spline-symbol convention while evaluating the
    spectral contraction and reverse pass independently in NumPy.
    """
    from reference.bspline_symbol import cardinal_bspline

    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    mesh_array = np.asarray(mesh, dtype=np.int64)
    inverse = np.linalg.inv(cell)
    fractional = positions @ inverse
    field = np.zeros(tuple(mesh_array), dtype=np.float64)
    stencils = []
    for atom, coordinate in enumerate(fractional):
        y = mesh_array * coordinate
        integer = np.floor(y)
        theta = y - integer
        start = np.floor(theta - (spline_order - 2.0) / 2.0)
        atom_stencil = []
        for axis in range(3):
            offsets = np.arange(spline_order, dtype=np.float64)
            arguments = spline_order / 2.0 + theta[axis] - (start[axis] + offsets)
            weights = cardinal_bspline(spline_order, arguments)
            slopes = (
                cardinal_bspline(spline_order - 1, arguments)
                - cardinal_bspline(spline_order - 1, arguments - 1.0)
            ) * mesh_array[axis]
            nodes = (
                integer[axis].astype(np.int64)
                + start[axis].astype(np.int64)
                + offsets.astype(np.int64)
            ) % mesh_array[axis]
            atom_stencil.append((nodes, weights, slopes))
        stencils.append(atom_stencil)
        for i in range(spline_order):
            for j in range(spline_order):
                for k in range(spline_order):
                    field[
                        atom_stencil[0][0][i],
                        atom_stencil[1][0][j],
                        atom_stencil[2][0][k],
                    ] += (
                        charges[atom]
                        * atom_stencil[0][1][i]
                        * atom_stencil[1][1][j]
                        * atom_stencil[2][1][k]
                    )

    axes = [
        np.arange(extent, dtype=np.int64)
        if axis < 2
        else np.arange(extent // 2 + 1, dtype=np.int64)
        for axis, extent in enumerate(mesh_array)
    ]
    miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    for axis in range(2):
        extent = mesh_array[axis]
        miller[..., axis] = np.where(
            miller[..., axis] <= extent // 2,
            miller[..., axis],
            miller[..., axis] - extent,
        )
    wave = 2.0 * np.pi * np.einsum("ab,...b->...a", inverse, miller)
    wave_norm = np.linalg.norm(wave, axis=-1)
    symbol = np.ones(wave_norm.shape, dtype=np.float64)
    for axis, extent in enumerate(mesh_array):
        symbol *= np.abs(np.sinc(miller[..., axis] / extent)) ** spline_order
    amplitude = np.fft.rfftn(field) / symbol
    hermitian = np.full(wave_norm.shape, 2.0, dtype=np.float64)
    hermitian[..., 0] = 1.0
    if mesh_array[2] % 2 == 0:
        hermitian[..., -1] = 1.0
    active = (wave_norm > 0.0) & (wave_norm <= _TWO_PI / dl)
    prefactor = COULOMB_NORM_FACTOR / abs(float(np.linalg.det(cell)))
    kernel = np.zeros(wave_norm.shape, dtype=np.float64)
    kernel[active] = (
        prefactor
        * np.exp(-0.5 * sigma * sigma * wave_norm[active] ** 2)
        / wave_norm[active] ** 2
    )
    bin_energy = hermitian * kernel * np.abs(amplitude) ** 2
    cotangent = 2.0 * kernel * amplitude
    potential = np.fft.irfftn(
        cotangent / symbol, s=tuple(mesh_array), axes=(0, 1, 2)
    ) * int(np.prod(mesh_array))

    charge_adjoints = np.zeros(charges.size, dtype=np.float64)
    forces = np.zeros((charges.size, 3), dtype=np.float64)
    for atom, atom_stencil in enumerate(stencils):
        fractional_gradient = np.zeros(3, dtype=np.float64)
        for i in range(spline_order):
            w0, d0 = atom_stencil[0][1][i], atom_stencil[0][2][i]
            for j in range(spline_order):
                w1, d1 = atom_stencil[1][1][j], atom_stencil[1][2][j]
                for k in range(spline_order):
                    w2, d2 = atom_stencil[2][1][k], atom_stencil[2][2][k]
                    value = potential[
                        atom_stencil[0][0][i],
                        atom_stencil[1][0][j],
                        atom_stencil[2][0][k],
                    ]
                    charge_adjoints[atom] += w0 * w1 * w2 * value
                    fractional_gradient += value * np.array(
                        [d0 * w1 * w2, w0 * d1 * w2, w0 * w1 * d2]
                    )
        forces[atom] = -charges[atom] * (inverse @ fractional_gradient)

    radial_derivative = np.zeros_like(wave_norm)
    radial_derivative[active] = bin_energy[active] * (
        -sigma * sigma * wave_norm[active] - 2.0 / wave_norm[active]
    )
    strain_derivative = -np.eye(3) * float(bin_energy.sum())
    safe_norm = np.where(active, wave_norm, 1.0)
    flat_wave = wave.reshape(-1, 3)
    strain_derivative -= np.einsum(
        "n,ni,nj->ij",
        (radial_derivative / safe_norm).reshape(-1),
        flat_wave,
        flat_wave,
    )

    self_factor = COULOMB_NORM_FACTOR / (sigma * _TWO_PI**1.5)
    self_sign = -1.0 if remove_self_interaction else 1.0
    charge_adjoints += 2.0 * self_sign * self_factor * charges
    atomic_energies = (
        0.5 * charges * (charge_adjoints - 2.0 * self_sign * self_factor * charges)
        + self_sign * self_factor * charges * charges
    )
    return {
        "energy": float(atomic_energies.sum()),
        "atomic_energies": atomic_energies,
        "forces": forces,
        "charge_adjoints": charge_adjoints,
        "strain_derivative": strain_derivative,
    }


def ordered_reciprocal_membership(cell: np.ndarray, dl: float) -> np.ndarray:
    """Return the oracle's ordered hemisphere integer-vector membership."""
    cell = np.asarray(cell, dtype=np.float64)
    reciprocal = _TWO_PI * np.linalg.inv(cell).T
    counts = [max(1, int(np.linalg.norm(vector) / dl)) for vector in cell]
    axes = [np.arange(-count, count + 1) for count in counts]
    nvec = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    kvec = nvec @ reciprocal
    k_squared = np.einsum("ij,ij->i", kvec, kvec)
    keep = (k_squared > 0.0) & (k_squared <= (_TWO_PI / dl) ** 2)
    nvec = nvec[keep]
    first = np.argmax(nvec != 0, axis=1)
    return nvec[nvec[np.arange(nvec.shape[0]), first] > 0]


def ordered_reciprocal_candidate_indices(cell: np.ndarray, dl: float) -> np.ndarray:
    """Map ordered oracle membership onto the full candidate-grid indices."""
    cell = np.asarray(cell, dtype=np.float64)
    counts = np.asarray(
        [max(1, int(np.linalg.norm(vector) / dl)) for vector in cell],
        dtype=np.int64,
    )
    widths = 2 * counts + 1
    shifted = ordered_reciprocal_membership(cell, dl) + counts
    return (
        (shifted[:, 0] * widths[1] + shifted[:, 1]) * widths[2] + shifted[:, 2]
    ).astype(np.int64)


def _periodic_monopole(positions, cell, charges, case: MACELESCase):
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    volume = float(np.linalg.det(cell))
    reciprocal = _TWO_PI * np.linalg.inv(cell).T
    counts = [max(1, int(np.linalg.norm(vector) / case.dl)) for vector in cell]
    axes = [np.arange(-count, count + 1) for count in counts]
    nvec = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    kvec = nvec @ reciprocal
    k_squared = np.einsum("ij,ij->i", kvec, kvec)
    keep = (k_squared > 0.0) & (k_squared <= (_TWO_PI / case.dl) ** 2)
    nvec, kvec, k_squared = nvec[keep], kvec[keep], k_squared[keep]
    first = np.argmax(nvec != 0, axis=1)
    sign = nvec[np.arange(nvec.shape[0]), first]
    keep = sign > 0
    kvec, k_squared = kvec[keep], k_squared[keep]
    factors = np.full(k_squared.shape, 2.0, dtype=np.float64)

    phase = positions @ kvec.T
    cosine, sine = np.cos(phase), np.sin(phase)
    structure_real = charges @ cosine
    structure_imag = charges @ sine
    kernel = factors * np.exp(-0.5 * case.sigma * case.sigma * k_squared) / k_squared
    prefactor = COULOMB_NORM_FACTOR / volume
    pair_projection = structure_real[None, :] * cosine + structure_imag[None, :] * sine
    atomic_energies = prefactor * charges * (pair_projection * kernel).sum(axis=1)
    energy = float(atomic_energies.sum())
    charge_adjoints = 2.0 * prefactor * (pair_projection * kernel).sum(axis=1)
    forces = (
        2.0
        * prefactor
        * charges[:, None]
        * np.einsum(
            "ik,k,kj->ij",
            structure_real[None, :] * sine - structure_imag[None, :] * cosine,
            kernel,
            kvec,
        )
    )

    self_factor = COULOMB_NORM_FACTOR / (case.sigma * _TWO_PI**1.5)
    if case.remove_self_interaction:
        energy -= self_factor * float(charges @ charges)
        atomic_energies -= self_factor * charges * charges
        charge_adjoints -= 2.0 * self_factor * charges
    else:
        energy += self_factor * float(charges @ charges)
        atomic_energies += self_factor * charges * charges
        charge_adjoints += 2.0 * self_factor * charges

    strain_derivative = _affine_strain_derivative(positions, cell, charges, case)
    return {
        "energy": energy,
        "atomic_energies": atomic_energies,
        "forces": forces,
        "charge_adjoints": charge_adjoints,
        "strain_derivative": strain_derivative,
    }


def _affine_strain_derivative(positions, cell, charges, case: MACELESCase):
    step = 1.0e-6
    derivative = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            strain = np.zeros((3, 3), dtype=np.float64)
            strain[row, column] = step
            plus = np.eye(3) + strain
            minus = np.eye(3) - strain
            energy_plus = _periodic_energy_only(
                positions @ plus, cell @ plus, charges, case
            )
            energy_minus = _periodic_energy_only(
                positions @ minus, cell @ minus, charges, case
            )
            derivative[row, column] = (energy_plus - energy_minus) / (2.0 * step)
    return derivative


def _periodic_energy_only(positions, cell, charges, case: MACELESCase):
    volume = float(np.linalg.det(cell))
    reciprocal = _TWO_PI * np.linalg.inv(cell).T
    counts = [max(1, int(np.linalg.norm(vector) / case.dl)) for vector in cell]
    axes = [np.arange(-count, count + 1) for count in counts]
    nvec = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    kvec = nvec @ reciprocal
    k_squared = np.einsum("ij,ij->i", kvec, kvec)
    keep = (k_squared > 0.0) & (k_squared <= (_TWO_PI / case.dl) ** 2)
    nvec, kvec, k_squared = nvec[keep], kvec[keep], k_squared[keep]
    first = np.argmax(nvec != 0, axis=1)
    keep = nvec[np.arange(nvec.shape[0]), first] > 0
    kvec, k_squared = kvec[keep], k_squared[keep]
    phase = positions @ kvec.T
    structure = charges @ np.exp(1j * phase)
    kernel = 2.0 * np.exp(-0.5 * case.sigma * case.sigma * k_squared) / k_squared
    energy = COULOMB_NORM_FACTOR / volume * float((kernel * abs(structure) ** 2).sum())
    self_factor = COULOMB_NORM_FACTOR / (case.sigma * _TWO_PI**1.5)
    return (
        energy - self_factor * float(charges @ charges)
        if case.remove_self_interaction
        else energy + self_factor * float(charges @ charges)
    )
