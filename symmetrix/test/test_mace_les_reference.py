"""Frozen numerical contract for the monopole MACE-LES reference oracle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from reference.mace_les import (
    load_mace_les_cases,
    nonperiodic_monopole_oracle,
    periodic_monopole_oracle,
)


def test_frozen_mace_les_cases_record_provenance_and_derivatives():
    cases = load_mace_les_cases()

    assert {case.boundary for case in cases} == {"nonperiodic", "periodic"}
    for case in cases:
        assert case.mace_commit == "7913afb6fa729878d20886b84a2079523727d958"
        assert len(case.atomic_energies) == len(case.charges)
        assert np.isclose(sum(case.atomic_energies), case.energy)
        assert case.forces.shape == (len(case.charges), 3)
        assert case.charge_adjoints.shape == (len(case.charges),)


def test_frozen_nonperiodic_oracle_matches_upstream_les():
    for case in load_mace_les_cases():
        if case.boundary != "nonperiodic":
            continue
        result = nonperiodic_monopole_oracle(case)
        assert result["energy"] == pytest.approx(case.energy, abs=2.0e-12)
        np.testing.assert_allclose(
            result["atomic_energies"], case.atomic_energies, atol=2.0e-12
        )
        np.testing.assert_allclose(result["forces"], case.forces, atol=2.0e-12)
        np.testing.assert_allclose(
            result["charge_adjoints"], case.charge_adjoints, atol=2.0e-12
        )


def test_frozen_periodic_oracle_matches_upstream_les():
    for case in load_mace_les_cases():
        if case.boundary != "periodic":
            continue
        result = periodic_monopole_oracle(case)
        assert result["energy"] == pytest.approx(case.energy, abs=2.0e-12)
        np.testing.assert_allclose(
            result["atomic_energies"], case.atomic_energies, atol=2.0e-12
        )
        np.testing.assert_allclose(result["forces"], case.forces, atol=2.0e-12)
        np.testing.assert_allclose(
            result["charge_adjoints"], case.charge_adjoints, atol=2.0e-12
        )


@pytest.mark.parametrize("case", load_mace_les_cases(), ids=lambda case: case.name)
def test_mace_les_oracle_charge_adjoints_match_finite_differences(case):
    oracle = (
        nonperiodic_monopole_oracle(case)
        if case.boundary == "nonperiodic"
        else periodic_monopole_oracle(case)
    )
    step = 1.0e-6
    numeric = np.empty_like(case.charges)
    for index in range(case.charges.size):
        plus = case.charges.copy()
        minus = case.charges.copy()
        plus[index] += step
        minus[index] -= step
        evaluate = (
            nonperiodic_monopole_oracle
            if case.boundary == "nonperiodic"
            else periodic_monopole_oracle
        )
        numeric[index] = (
            evaluate(replace(case, charges=plus))["energy"]
            - evaluate(replace(case, charges=minus))["energy"]
        ) / (2.0 * step)
    np.testing.assert_allclose(
        oracle["charge_adjoints"], numeric, rtol=2.0e-7, atol=2.0e-8
    )


@pytest.mark.parametrize(
    "case",
    [case for case in load_mace_les_cases() if case.boundary == "periodic"],
    ids=lambda case: case.name,
)
def test_periodic_oracle_affine_strain_derivative_matches_upstream(case):
    result = periodic_monopole_oracle(case)
    np.testing.assert_allclose(result["strain_derivative"], case.stress, atol=5.0e-9)
