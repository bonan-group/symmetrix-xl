"""Tests for the joined small-structure MD benchmark driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/ase_small_structures_direct_batch.py"


def _load_driver():
    specification = importlib.util.spec_from_file_location(
        "ase_small_structures_direct_batch_test", SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _ConstantForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, forces):
        super().__init__()
        self._forces = np.asarray(forces, dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": self._forces.copy()}


@pytest.mark.parametrize("fix_center_of_mass", [False, True])
def test_batch_langevin_step_matches_ase(fix_center_of_mass):
    driver = _load_driver()
    positions = np.array(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.8], [0.5, 1.4, 1.1]], dtype=float
    )
    velocities = np.array(
        [[0.02, -0.01, 0.03], [-0.02, 0.04, -0.01], [0.01, 0.0, -0.03]],
        dtype=float,
    )
    forces = np.array(
        [[0.2, -0.1, 0.05], [-0.3, 0.4, 0.1], [0.15, -0.2, 0.25]],
        dtype=float,
    )
    reference = Atoms("HCO", positions=positions, cell=[8.0, 8.0, 8.0], pbc=True)
    candidate = reference.copy()
    reference.set_velocities(velocities)
    candidate.set_velocities(velocities)
    reference.calc = _ConstantForceCalculator(forces)

    reference_rng = np.random.default_rng(20261106)
    candidate_rng = np.random.default_rng(20261106)
    dynamics = Langevin(
        reference,
        2.0 * units.fs,
        temperature_K=300.0,
        friction=0.01 / units.fs,
        fixcm=fix_center_of_mass,
        rng=reference_rng,
    )
    dynamics.step(forces.copy())

    states = driver._langevin_begin(
        [candidate],
        [forces.copy()],
        candidate_rng,
        2.0,
        300.0,
        0.01,
        fix_center_of_mass,
    )
    driver._langevin_end([candidate], [forces.copy()], states)

    np.testing.assert_allclose(candidate.positions, reference.positions, atol=1e-15)
    np.testing.assert_allclose(
        candidate.get_velocities(), reference.get_velocities(), atol=1e-15
    )
