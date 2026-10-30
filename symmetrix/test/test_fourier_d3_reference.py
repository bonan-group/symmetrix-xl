"""Behavioural coverage for the mesh-free FourierD3 direct-reciprocal oracle.

Every expected value in the fixtures was produced independently of the oracle under test:
the NVIDIA ALCHEMI Toolkit-Ops v0.5.0 NumPy real-space lattice-sum path, or a hand-derived
invariant. Fixtures are frozen inputs, so the tests never regenerate them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from reference import bspline_symbol
from reference import fourier_d3


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIRECTORY = Path(__file__).parent / "reference" / "fixtures"
REFERENCE_FIXTURE = json.loads(
    (FIXTURE_DIRECTORY / "fourier_d3_reference_cases.json").read_text(encoding="utf-8")
)
SYMBOL_FIXTURE = json.loads(
    (FIXTURE_DIRECTORY / "bspline_symbol_cases.json").read_text(encoding="utf-8")
)
CASES = {case["name"]: case for case in REFERENCE_FIXTURE["cases"]}
PARAMETERS = REFERENCE_FIXTURE["parameters"]

# Absolute floors used with a fixture's recorded finite-difference band. They matter only
# when a band is numerically zero; every other tolerance is the band itself.
ENERGY_FLOOR = 1e-12
DERIVATIVE_FLOOR = 1e-12

DAMPING = fourier_d3.DampingParameters.from_bohr(
    s6=PARAMETERS["s6"],
    s8=PARAMETERS["s8"],
    a1=PARAMETERS["a1"],
    a2_bohr=PARAMETERS["a2_bohr"],
)

# Minimal one-atom case, written out literally: a single carbon atom in a 4 angstrom
# cubic cell with a 6 angstrom coordination cutoff. The radius is ``d3_rcov[6]`` of the
# repository's generated table, which is 1-based by atomic number. The expectation is the
# hand-derived retained-self-image lattice sum, not a value read back from the oracle.
SINGLE_ATOM = {
    "numbers": (6,),
    "positions": ((0.0, 0.0, 0.0),),
    "cell": ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
    "covalent_radii": (1.0000000494517636,),
    "coordination": 0.001996448500123995,
}


def probe_id(probe) -> str:
    return f"p{probe['spline_order']}-n{probe['mesh_size']}-f{probe['fractional_coordinate']}"


def factors_for(case) -> fourier_d3.LowRankFactors:
    decomposition = case["decomposition"]
    return fourier_d3.LowRankFactors(
        species=np.asarray(decomposition["species"], dtype=np.int64),
        eigs=np.asarray(decomposition["eigs_ev_angstrom6"], dtype=np.float64),
        v_q=np.asarray(decomposition["v_q"], dtype=np.float64),
        cn_ref=np.asarray(decomposition["cn_ref"], dtype=np.float64),
        valid=np.asarray(decomposition["valid"], dtype=bool),
        q_table=np.asarray(case["q_table_angstrom"], dtype=np.float64),
    )


def evaluate(case, k_max=None):
    return fourier_d3.evaluate(
        np.asarray(case["positions_angstrom"], dtype=np.float64),
        np.asarray(case["atomic_numbers"], dtype=np.int64),
        np.asarray(case["covalent_radii_angstrom"], dtype=np.float64),
        np.asarray(case["cell_angstrom"], dtype=np.float64),
        factors_for(case),
        DAMPING,
        PARAMETERS["coordination_cutoff_angstrom"],
        PARAMETERS["k_max_per_angstrom"] if k_max is None else k_max,
    )


def cases_with(*keys: str) -> list[str]:
    return sorted(
        name
        for name, case in CASES.items()
        if all(key in case["expected"] for key in keys)
    )


def test_single_atom_retained_self_image_energy():
    """One atom in a periodic cell is not dispersion-free: its images are retained."""
    case = CASES["single-atom-self-images"]
    factors = factors_for(case)
    result = fourier_d3.evaluate(
        np.asarray(SINGLE_ATOM["positions"], dtype=np.float64),
        np.asarray(SINGLE_ATOM["numbers"], dtype=np.int64),
        np.asarray(SINGLE_ATOM["covalent_radii"], dtype=np.float64),
        np.asarray(SINGLE_ATOM["cell"], dtype=np.float64),
        factors,
        DAMPING,
        PARAMETERS["coordination_cutoff_angstrom"],
        PARAMETERS["k_max_per_angstrom"],
    )
    np.testing.assert_allclose(
        result.coordination,
        [SINGLE_ATOM["coordination"]],
        rtol=0.0,
        atol=1e-13,
    )
    invariant = case["hand_derived_invariant"]
    radii = invariant["radii_angstrom"]
    converged = invariant["energy_ev"][f"{max(radii)}"]
    assert abs(result.energy - converged) <= max(
        2.0 * invariant["band_ev"], ENERGY_FLOOR
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_coordination_matches_frozen_expectation(name):
    result = evaluate(CASES[name])
    np.testing.assert_allclose(
        result.coordination,
        np.asarray(CASES[name]["expected"]["coordination"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_energy_matches_frozen_expectation(name):
    expected = CASES[name]["expected"]
    result = evaluate(CASES[name])
    tolerance = max(2.0 * expected["energy_convergence_band_ev"], ENERGY_FLOOR)
    assert abs(result.energy - expected["energy_ev"]) <= tolerance


@pytest.mark.parametrize("name", cases_with("coordination_sensitivity_ev"))
def test_coordination_sensitivity_matches_frozen_expectation(name):
    expected = CASES[name]["expected"]
    result = evaluate(CASES[name])
    tolerance = max(
        2.0 * CASES[name]["finite_difference"]["coordination_sensitivity_band"],
        DERIVATIVE_FLOOR,
    )
    np.testing.assert_allclose(
        result.coordination_sensitivity,
        np.asarray(expected["coordination_sensitivity_ev"], dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    )


@pytest.mark.parametrize("name", cases_with("forces_ev_per_angstrom"))
def test_forces_match_frozen_expectation(name):
    expected = CASES[name]["expected"]
    result = evaluate(CASES[name])
    tolerance = max(
        2.0 * CASES[name]["finite_difference"]["forces_band"], DERIVATIVE_FLOOR
    )
    np.testing.assert_allclose(
        result.forces,
        np.asarray(expected["forces_ev_per_angstrom"], dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    )


@pytest.mark.parametrize("name", cases_with("strain_derivative_ev"))
def test_strain_derivative_matches_frozen_expectation(name):
    expected = CASES[name]["expected"]
    result = evaluate(CASES[name])
    tolerance = max(
        2.0 * CASES[name]["finite_difference"]["strain_derivative_band"],
        DERIVATIVE_FLOOR,
    )
    np.testing.assert_allclose(
        result.strain_derivative,
        np.asarray(expected["strain_derivative_ev"], dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_forces_sum_to_zero_and_strain_is_symmetric(name):
    result = evaluate(CASES[name])
    np.testing.assert_allclose(
        result.forces.sum(axis=0), np.zeros(3), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.strain_derivative,
        result.strain_derivative.T,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_reciprocal_sum_is_converged_in_k_max(name):
    """Halving the fixture cutoff costs at most 1e-6 relative, not a power of it."""
    case = CASES[name]
    reference = evaluate(case, k_max=PARAMETERS["k_max_per_angstrom"])
    coarse = evaluate(case, k_max=0.5 * PARAMETERS["k_max_per_angstrom"])
    scale = max(abs(reference.energy), 1e-6)
    assert abs(coarse.energy - reference.energy) <= 1e-6 * scale


def test_reciprocal_sum_truncation_decays_exponentially():
    """Successive doublings of the cutoff must shrink the truncation by an order each."""
    case = CASES["single-atom-self-images"]
    energies = {
        k: evaluate(case, k_max=k).energy for k in (12.0, 16.0, 20.0, 24.0, 32.0, 40.0)
    }
    converged = energies[40.0]
    scale = abs(converged)
    errors = {k: abs(value - converged) for k, value in energies.items()}
    assert errors[24.0] <= 1e-12 * scale
    assert errors[16.0] <= 1e-8 * scale
    for coarse, fine in ((12.0, 16.0), (16.0, 20.0), (20.0, 24.0)):
        assert errors[coarse] > 10.0 * errors[fine]


def test_strained_cell_pair_uses_the_affine_strain_convention():
    reference = CASES["strained-cell-reference"]
    strained = CASES["strained-cell-strained"]
    assert reference["pair"] == strained["pair"] == "strained-cell"
    deformation = np.eye(3) + np.asarray(strained["applied_strain"], dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(strained["positions_angstrom"], dtype=np.float64),
        np.asarray(reference["positions_angstrom"], dtype=np.float64) @ deformation,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(strained["cell_angstrom"], dtype=np.float64),
        np.asarray(reference["cell_angstrom"], dtype=np.float64) @ deformation,
        rtol=0.0,
        atol=1e-14,
    )
    strain = np.asarray(strained["applied_strain"], dtype=np.float64)
    assert not np.allclose(strain, strain.T)


def test_strained_cell_pair_changes_energy_and_coordination():
    reference = CASES["strained-cell-reference"]["expected"]
    strained = CASES["strained-cell-strained"]["expected"]
    band = (
        reference["energy_convergence_band_ev"] + strained["energy_convergence_band_ev"]
    )
    assert abs(reference["energy_ev"] - strained["energy_ev"]) > 100.0 * band
    assert (
        np.abs(
            np.asarray(reference["coordination"], dtype=np.float64)
            - np.asarray(strained["coordination"], dtype=np.float64)
        ).max()
        > 1e-3
    )
    np.testing.assert_allclose(
        evaluate(CASES["strained-cell-strained"]).energy,
        strained["energy_ev"],
        rtol=0.0,
        atol=max(2.0 * strained["energy_convergence_band_ev"], ENERGY_FLOOR),
    )


SKEWED_CELLS = (
    ("literal", np.array([[7.5, 0.0, 0.0], [1.2, 6.8, 0.0], [0.8, -1.1, 8.2]]), 6.0),
    (
        "triclinic-fixture",
        np.array(CASES["triclinic-mixed-species"]["cell_angstrom"], dtype=np.float64),
        PARAMETERS["k_max_per_angstrom"],
    ),
)


@pytest.mark.parametrize(
    "label,cell,k_max", SKEWED_CELLS, ids=[row[0] for row in SKEWED_CELLS]
)
def test_reciprocal_vectors_close_a_complete_sphere_on_a_skewed_cell(
    label, cell, k_max
):
    """The Miller box must bound the sphere rigorously and be symmetric under m -> -m."""
    miller, wave_vectors, norm = fourier_d3.reciprocal_vectors(cell, k_max)

    keys = {tuple(int(value) for value in row) for row in miller}
    assert all(
        tuple(-int(value) for value in row) in keys for row in miller
    ), "not closed"

    # Completeness: a deliberately generous box filtered by |k| <= k_max must give the
    # same set. Bounding each |m_j| by k_max / |b_j| instead would clip this cell.
    generous = (
        np.ceil(np.linalg.norm(cell, axis=1) * k_max / (2.0 * np.pi)).astype(int) + 4
    )
    all_miller = np.stack(
        np.meshgrid(*[np.arange(-n, n + 1) for n in generous], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    all_wave = all_miller @ (2.0 * np.pi * np.linalg.inv(cell)).T
    inside = np.linalg.norm(all_wave, axis=1) <= k_max
    expected = {tuple(int(value) for value in row) for row in all_miller[inside]}
    assert keys == expected, "sphere is incomplete"
    assert keys, "empty sphere"

    # Each retained index is the exact projection of its own wave vector: m_j = a_j.k/2pi.
    np.testing.assert_allclose(
        wave_vectors @ cell.T / (2.0 * np.pi), miller, rtol=0.0, atol=1e-12
    )
    assert np.all(norm <= k_max)


def test_padding_atoms_are_skipped():
    """Atomic number 0 is padding (spec 3.2.4): it neither accumulates nor is counted."""
    case = CASES["coordination-sensitive"]
    factors = factors_for(case)
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int64)
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    radii = np.asarray(case["covalent_radii_angstrom"], dtype=np.float64)
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    reference = fourier_d3.evaluate(
        positions,
        numbers,
        radii,
        cell,
        factors,
        DAMPING,
        PARAMETERS["coordination_cutoff_angstrom"],
        PARAMETERS["k_max_per_angstrom"],
    )

    padded = fourier_d3.evaluate(
        np.vstack([positions, [[0.5, 0.5, 0.5]]]),
        np.concatenate([numbers, [0]]),
        np.concatenate([radii, [2.0]]),
        cell,
        factors,
        DAMPING,
        PARAMETERS["coordination_cutoff_angstrom"],
        PARAMETERS["k_max_per_angstrom"],
    )

    assert padded.coordination[-1] == 0.0
    np.testing.assert_allclose(
        padded.coordination[:-1], reference.coordination, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(padded.forces[-1], np.zeros(3), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        padded.forces[:-1], reference.forces, rtol=0.0, atol=1e-15
    )
    assert abs(padded.energy - reference.energy) <= 1e-15
    assert np.abs(padded.strain_derivative - reference.strain_derivative).max() <= 1e-12


def test_coordination_pair_term_invariants():
    cutoff = PARAMETERS["coordination_cutoff_angstrom"]
    covalent_sum = 1.6 + 0.42666665874797699
    assert float(
        fourier_d3.coordination_pair_term(covalent_sum, covalent_sum, cutoff)
    ) == pytest.approx(0.5, abs=1e-15)
    beyond = np.array([cutoff, cutoff + 1e-3, cutoff + 2.0])
    np.testing.assert_array_equal(
        fourier_d3.coordination_pair_term(beyond, covalent_sum, cutoff), np.zeros(3)
    )
    np.testing.assert_array_equal(
        fourier_d3.coordination_pair_term_derivative(beyond, covalent_sum, cutoff),
        np.zeros(3),
    )

    radii = np.array([1.3, 1.6, 2.0, 2.6, 3.2, 4.0, 5.0])
    step = 1e-4
    finite_difference = (
        sum(
            weight
            * fourier_d3.coordination_pair_term(
                radii + offset * step, covalent_sum, cutoff
            )
            for offset, weight in (
                (2.0, -1.0 / 12.0),
                (1.0, 8.0 / 12.0),
                (-1.0, -8.0 / 12.0),
                (-2.0, 1.0 / 12.0),
            )
        )
        / step
    )
    np.testing.assert_allclose(
        fourier_d3.coordination_pair_term_derivative(radii, covalent_sum, cutoff),
        finite_difference,
        rtol=1e-7,
        atol=1e-11,
    )


def test_reciprocal_kernel_matches_closed_form_limits():
    q_product = 1.0622434512415695**2
    r0 = DAMPING.damping_radius(q_product)
    g6, g8 = fourier_d3.kernel_shape_functions(np.array([0.0]))
    assert g6[0] == pytest.approx(1.0, abs=0.0)
    assert g8[0] == pytest.approx(-np.sqrt(2.0) * np.sin(np.pi / 8.0), rel=1e-14)
    assert fourier_d3.reciprocal_kernel(0.0, r0, q_product, DAMPING) == pytest.approx(
        PARAMETERS["s6"] * 2.0 * np.pi**2 / (3.0 * r0**3)
        + 3.0
        * PARAMETERS["s8"]
        * q_product
        * np.pi**2
        / r0**5
        * np.sqrt(2.0)
        * np.sin(np.pi / 8.0),
        rel=1e-14,
    )
    # The series and closed-form branches must agree across the crossover.
    either_side = np.array([0.1 - 1e-9, 0.1, 0.1 + 1e-9])
    series6, series8 = fourier_d3.kernel_shape_functions(either_side)
    assert np.abs(np.diff(series6)).max() < 1e-6
    assert np.abs(np.diff(series8)).max() < 1e-6
    assert fourier_d3.reciprocal_kernel(0.0, r0, q_product, DAMPING) != 0.0


def test_reciprocal_kernel_derivative_matches_central_difference():
    q_product = 1.0622434512415695**2
    r0 = DAMPING.damping_radius(q_product)
    norms = np.array([1e-4, 0.05, 0.5, 2.0, 6.0, 12.0])
    step = 1e-6
    finite_difference = (
        fourier_d3.reciprocal_kernel(norms + step, r0, q_product, DAMPING)
        - fourier_d3.reciprocal_kernel(norms - step, r0, q_product, DAMPING)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        fourier_d3.reciprocal_kernel_derivative(norms, r0, q_product, DAMPING),
        finite_difference,
        rtol=1e-6,
        atol=1e-14,
    )


def test_low_rank_coefficient_derivative_matches_central_difference():
    case = CASES["coordination-sensitive"]
    factors = factors_for(case)
    channels = factors.channels(np.asarray(case["atomic_numbers"], dtype=np.int64))
    theta = np.asarray(case["expected"]["coordination"], dtype=np.float64)
    step = 1e-6
    derivative = fourier_d3.low_rank_coefficient_derivative(theta, channels, factors)
    for index in range(theta.size):
        moved = theta.copy()
        moved[index] += step
        plus = fourier_d3.low_rank_coefficients(moved, channels, factors)[index]
        moved[index] -= 2.0 * step
        minus = fourier_d3.low_rank_coefficients(moved, channels, factors)[index]
        np.testing.assert_allclose(
            derivative[index], (plus - minus) / (2.0 * step), rtol=1e-6, atol=1e-9
        )


def test_fixture_provenance_is_pinned():
    provenance = REFERENCE_FIXTURE["provenance"]
    upstream = provenance["upstream"]
    assert upstream["tag"] == "v0.5.0"
    assert upstream["commit"] == "6702138e404d40ea79dec62f68c2fa67e858580e"
    assert set(upstream["files"]) == {
        "test/interactions/dispersion/_fourier_reference.py",
        "nvalchemiops/interactions/dispersion/_c6_decomposition.py",
    }
    for digest in (*upstream["files"].values(), provenance["tables"]["sha256"]):
        assert len(digest) == 64 and all(
            character in "0123456789abcdef" for character in digest
        )
    assert (
        provenance["tables"]["file"] == "libsymmetrix/source/dispersion_parameters.cpp"
    )
    specification = REFERENCE_FIXTURE["specification"]
    assert specification["document"].endswith("fourier_d3_mathematics.md")
    # The oracle was derived from this revision of the specification.  The
    # digest check below proves the recorded document still matches the tree;
    # this pin proves *which* revision the digest belongs to, so an amendment
    # must update both together rather than silently re-pointing the digest.
    assert specification["revision_commit"] == "e9f3af0"
    # The document has been amended since the oracle was derived from it, and the
    # two facts are recorded separately so neither can be inferred from the
    # other.  `revision_commit`/`sha256` are the *current normative* document and
    # exist to catch an accidental edit; `oracle_derived_from` is the revision the
    # numerical values below were derived from.  `oracle_regenerated` is the
    # explicit statement that the amendment did not recompute them - the Task 10
    # review changed the section 6.2 *obligation*, not any equation the Task 2
    # oracle implements, so a re-derivation would have produced identical numbers
    # and the fixture's own value assertions are what actually pin them.
    assert specification["oracle_derived_from"] == "6328429"
    assert specification["oracle_regenerated"] is False
    assert specification["oracle_derived_from"] != specification["revision_commit"], (
        "the amendment is recorded as an amendment: if the oracle were ever "
        "regenerated from the current document, `oracle_derived_from` would move to "
        "`revision_commit` and `oracle_regenerated` would become true"
    )
    for field in ("revision_commit", "oracle_derived_from"):
        value = specification[field]
        assert len(value) == 7 and all(
            character in "0123456789abcdef" for character in value
        ), f"{field} must be an abbreviated commit hash, got {value!r}"
    assert specification[
        "amendment"
    ].strip(), "the fixture must record what the amendment was and that it changed no oracle input"
    assert "oracle" in specification["amendment"].lower()
    assert {"3.2", "4.2", "4.4", "4.5", "6.3"}.issubset(set(specification["sections"]))
    assert provenance["units"]["length"] == "angstrom"
    assert provenance["units"]["energy"] == "electronvolt"
    assert PARAMETERS["functional"] == "pbe" and PARAMETERS["damping"] == "bj"
    assert PARAMETERS["decomposition_tolerance"] == 1e-4
    assert PARAMETERS["a2_angstrom"] == pytest.approx(
        PARAMETERS["a2_bohr"] * fourier_d3.BOHR_TO_ANGSTROM, rel=0.0, abs=0.0
    )
    for name, case in CASES.items():
        assert case["coverage"], name
        assert len(case["atomic_numbers"]) == len(case["positions_angstrom"]), name
        assert len(case["covalent_radii_angstrom"]) == len(case["atomic_numbers"]), name
        assert len(case["q_table_angstrom"]) == len(
            case["decomposition"]["species"]
        ), name
        assert len(case["expected"]["coordination"]) == len(
            case["atomic_numbers"]
        ), name


@pytest.mark.parametrize("name", sorted(CASES))
def test_frozen_decomposition_inputs_are_internally_consistent(name):
    decomposition = CASES[name]["decomposition"]
    valid = np.asarray(decomposition["valid"], dtype=bool)
    cn_ref = np.asarray(decomposition["cn_ref"], dtype=np.float64)
    v_q = np.asarray(decomposition["v_q"], dtype=np.float64)
    eigs = np.asarray(decomposition["eigs_ev_angstrom6"], dtype=np.float64)
    assert decomposition["max_relative_error"] <= PARAMETERS["decomposition_tolerance"]
    np.testing.assert_array_equal(valid, cn_ref != -1.0)
    np.testing.assert_array_equal(v_q[~valid], np.zeros_like(v_q[~valid]))
    assert np.all(np.diff(np.abs(eigs)) <= 0.0)
    for slot in range(eigs.size):
        column = v_q[:, :, slot].reshape(-1)
        assert column[int(np.argmax(np.abs(column)))] > 0.0


def test_fixture_provenance_hashes_match_the_working_tree():
    """The frozen provenance must be reproducible from the committed revision.

    The specification digest is the *current normative* document, not the revision
    the numerical values were derived from: the two are recorded separately in
    `specification` (see `test_fixture_provenance_is_pinned`).  Pinning the current
    document is what catches an accidental edit; recording the derivation
    revision separately is what stops that pin from being read as a claim that the
    oracle was regenerated.
    """
    provenance = REFERENCE_FIXTURE["provenance"]
    specification = REFERENCE_FIXTURE["specification"]
    for recorded, relative in (
        (provenance["generator_sha256"], provenance["generator"]),
        (provenance["tables"]["sha256"], provenance["tables"]["file"]),
        (specification["sha256"], specification["document"]),
    ):
        digest = hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        assert recorded == digest, relative


def test_reference_oracle_has_no_forbidden_runtime_dependency():
    """The oracle must not need a transform, a mesh, or an NVIDIA runtime."""
    program = (
        "import json, sys;"
        "import numpy as np;"
        "from reference import fourier_d3 as m;"
        "f = m.LowRankFactors(species=np.array([6]), eigs=np.array([1.0]),"
        " v_q=np.zeros((1, 1, 1)), cn_ref=np.zeros((1, 1)), valid=np.zeros((1, 1), bool),"
        " q_table=np.array([1.0]));"
        "p = m.DampingParameters.from_bohr(1.0, 0.7875, 0.4289, 4.4407);"
        "r = m.evaluate(np.zeros((1, 3)), np.array([6]), np.array([0.4]), np.eye(3) * 4.0,"
        " f, p, 6.0, 12.0);"
        "print(json.dumps({'modules': sorted(n for n in sys.modules"
        " if n.split('.')[0] in {'torch', 'warp', 'nvalchemiops'}),"
        " 'fft': 'numpy.fft' in sys.modules, 'energy': r.energy}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(Path(__file__).parent),
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["modules"] == []
    assert report["fft"] is False
    assert np.isfinite(report["energy"])


@pytest.mark.parametrize(
    "order", sorted(SYMBOL_FIXTURE["cardinal_bspline_integer_values"])
)
def test_cardinal_bspline_values_match_hand_derived_exact_values(order):
    expected = SYMBOL_FIXTURE["cardinal_bspline_integer_values"][order]
    np.testing.assert_allclose(
        bspline_symbol.cardinal_bspline_values(int(order)),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "probe", SYMBOL_FIXTURE["probes"], ids=lambda probe: probe_id(probe)
)
def test_symbols_match_independent_closed_forms(probe):
    """Binding: every expectation here is derived without the module's own algorithms."""
    miller = np.asarray(probe["miller"], dtype=np.float64)
    order = probe["spline_order"]
    mesh_size = probe["mesh_size"]
    # Continuous symbol: the literal closed form of specification section 5.2.
    np.testing.assert_allclose(
        bspline_symbol.continuous_symbol(miller, mesh_size, order),
        np.abs(np.sinc(miller / mesh_size)) ** order,
        rtol=0.0,
        atol=0.0,
    )
    # Discrete symbol: the generator's exact piecewise-polynomial route, which never
    # touches the module's stencil placement or its Cox-de Boor recursion.
    np.testing.assert_allclose(
        bspline_symbol.discrete_symbol(miller, mesh_size, order),
        np.asarray(probe["discrete_symbol_closed_form"], dtype=np.float64),
        rtol=1e-11,
        atol=1e-13,
    )
    # Assignment attenuation at a real fractional coordinate: the same independent route,
    # so a wrong stencil offset or a wrong cardinal B-spline fails here.
    np.testing.assert_allclose(
        bspline_symbol.assignment_attenuation(
            mesh_size, order, probe["fractional_coordinate"], miller
        ),
        np.asarray(probe["exact_attenuation"], dtype=np.float64),
        rtol=1e-11,
        atol=1e-13,
    )
    # Hand-derivable invariant: the assignment is a partition of unity, so the m = 0
    # coefficient is exactly one at every fractional coordinate.
    assert (
        abs(
            float(
                bspline_symbol.assignment_attenuation(
                    mesh_size, order, probe["fractional_coordinate"], np.array([0.0])
                )[0]
            )
            - 1.0
        )
        <= 1e-14
    )


@pytest.mark.parametrize(
    "probe", SYMBOL_FIXTURE["probes"], ids=lambda probe: probe_id(probe)
)
def test_measured_attenuation_is_self_consistent_with_the_module(probe):
    """Non-binding: the fixture's measurement duplicates the module's implementation."""
    miller = np.asarray(probe["miller"], dtype=np.float64)
    np.testing.assert_allclose(
        bspline_symbol.assignment_attenuation(
            probe["mesh_size"],
            probe["spline_order"],
            probe["fractional_coordinate"],
            miller,
        ),
        np.asarray(probe["measured_attenuation"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-14,
    )


def test_continuous_and_discrete_symbols_are_genuinely_distinguishable():
    """An atom on a mesh node makes the discrete symbol exact and the continuous one wrong."""
    distinguishing = []
    for probe in SYMBOL_FIXTURE["probes"]:
        if probe["fractional_coordinate"] != 0.0:
            continue
        exact = np.asarray(probe["exact_attenuation"], dtype=np.float64)
        discrete = np.asarray(probe["discrete_symbol_closed_form"], dtype=np.float64)
        miller = np.asarray(probe["miller"], dtype=np.float64)
        continuous = (
            np.abs(np.sinc(miller / probe["mesh_size"])) ** probe["spline_order"]
        )
        np.testing.assert_allclose(exact, discrete, rtol=0.0, atol=1e-13)
        distinguishing.append(np.abs(exact - continuous).max())
    assert distinguishing, "no on-node probe in the fixture"
    assert min(distinguishing) > 1e-3


def test_deconvolution_modulus_follows_the_adopted_continuous_convention():
    assert SYMBOL_FIXTURE["adopted_convention"] == bspline_symbol.ADOPTED_CONVENTION
    assert bspline_symbol.ADOPTED_CONVENTION == "continuous"
    miller = np.asarray([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        bspline_symbol.deconvolution_modulus(miller, 12, 4),
        bspline_symbol.continuous_symbol(miller, 12, 4),
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(
        bspline_symbol.deconvolution_modulus(miller, 12, 4),
        bspline_symbol.discrete_symbol(miller, 12, 4),
    )
