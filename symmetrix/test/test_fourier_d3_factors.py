"""Behavioural coverage for the deterministic low-rank DFT-D3 C6 decomposition.

Expected values come from three sources independent of the implementation: hand-built
canonical tables whose block matrix is known in closed form, the brute-force truncation
definition of specification section 3.3.3, and Task 2's frozen NVIDIA v0.5.0 factors plus
the repository's own generated D3 tables (parsed here, never at installed runtime).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from symmetrix import fourier_d3_factors
from reference import fourier_d3


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TABLES_FILE = "libsymmetrix/source/dispersion_parameters.cpp"
REFERENCE_FIXTURE = json.loads(
    (
        Path(__file__).parent
        / "reference"
        / "fixtures"
        / "fourier_d3_reference_cases.json"
    ).read_text(encoding="utf-8")
)
FIXTURE_TOLERANCE = REFERENCE_FIXTURE["parameters"]["decomposition_tolerance"]

K3_WEIGHT = 4.0

# Canonical validity is ``C6 > 0``, so every synthetic block must have strictly positive
# populated entries. These three blocks are entrywise positive and still exercise the
# signed spectrum, the tie-breaking gauge, and a non-monotone rank scan.

# Eigenvalues 4.8049, 0.4656, -0.1553. The error rises from rank 1 to rank 2
# (0.4008 -> 0.4642), so a scan that assumed a monotone error would retain rank 2.
POSITIVE_BLOCK = np.array(
    [
        [2.08541460037255000, 0.78710999928919280, 1.94605053342134320],
        [0.78710999928919280, 0.29260912465680033, 1.13754796461718530],
        [1.94605053342134320, 1.13754796461718530, 2.73717614890756970],
    ]
)
NON_MONOTONE_TOLERANCE = 0.42

# Both eigenvectors of this block have exactly tied leading magnitudes
# (|0.7071067811865475| twice), so the sign gauge must break the tie by the smallest flat
# index. The block needs both slots: the rank-1 error is 0.5, far above tolerance.
TIED_BLOCK = np.array([[1.0, 2.0], [2.0, 1.0]])

# Eigenvalues 5, -1, -1: ordering by magnitude must put +5 first, and the two negative
# eigenvalues must survive as negative values.
SIGNED_BLOCK = np.full((3, 3), 2.0) + np.diag(np.full(3, -1.0))

# Coordination pairs spanning the reference range of H, C, and O.
CONTRACTION_COORDINATIONS = (
    (0.0, 0.0),
    (3.0, 0.0),
    (0.9868, 1.9985),
    (2.5, 3.5),
    (4.2, 1.1),
)


def canonical_tables() -> dict[str, np.ndarray]:
    """Read the repository's generated D3 tables with the fixture generator's parser."""
    saved = sys.dont_write_bytecode
    try:
        from tools.generate_fourier_d3_reference import parse_tables

        return parse_tables(REPOSITORY_ROOT / TABLES_FILE)
    finally:
        sys.dont_write_bytecode = saved


def synthetic_table(max_z, n_ref, species, blocks):
    """Canonical ``(max_z+1, max_z+1, n_ref, n_ref, 3)`` table with ``-1`` sentinels.

    ``species`` maps an atomic number to ``(count, reference coordination per slot)`` and
    ``blocks`` maps an ordered species pair to its ``(n_ref, n_ref)`` C6 block. Slots a
    species does not use keep the canonical ``-1`` sentinel, and pairs with no data are
    all-zero, exactly as the generated table stores them.
    """
    table = np.zeros((max_z + 1, max_z + 1, n_ref, n_ref, 3), dtype=np.float64)
    for receiver in species:
        receiver_count, receiver_coordination = species[receiver]
        for source in species:
            source_count, source_coordination = species[source]
            if receiver_count == 0 or source_count == 0:
                continue
            table[receiver, source, :, :, :] = -1.0
            block = np.asarray(blocks[(receiver, source)], dtype=np.float64)
            for p in range(receiver_count):
                for q in range(source_count):
                    table[receiver, source, p, q, 0] = block[p, q]
                    table[receiver, source, p, q, 1] = receiver_coordination[p]
                    table[receiver, source, p, q, 2] = source_coordination[q]
    return table


def expected_block(table, species):
    """Masked, exactly symmetrised active-species block and its per-species validity."""
    n_species = len(species)
    n_ref = table.shape[2]
    valid = np.zeros((n_species, n_ref), dtype=bool)
    for row, atomic_number in enumerate(species):
        valid[row] = (table[atomic_number, :, :, :, 0] > 0.0).any(axis=(0, 2))
    block = np.zeros((n_species * n_ref, n_species * n_ref), dtype=np.float64)
    for a, za in enumerate(species):
        for b, zb in enumerate(species):
            sub = table[za, zb, :, :, 0]
            mask = valid[a][:, None] & valid[b][None, :] & (sub > 0.0)
            block[a * n_ref : (a + 1) * n_ref, b * n_ref : (b + 1) * n_ref] = np.where(
                mask, sub, 0.0
            )
    return 0.5 * (block + block.T), valid


def reconstruction_errors(block, rank_cap=None):
    """``epsilon(r)`` for every rank, straight from the definition of spec 3.3.3."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (block + block.T))
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    populated = block != 0.0
    upper = eigenvalues.size if rank_cap is None else min(rank_cap, eigenvalues.size)
    errors = []
    for rank in range(1, upper + 1):
        approximation = (eigenvectors[:, :rank] * eigenvalues[:rank]) @ eigenvectors[
            :, :rank
        ].T
        errors.append(
            float(
                np.max(
                    np.abs(approximation[populated] - block[populated])
                    / np.abs(block[populated])
                )
            )
        )
    return errors


def brute_force_rank(block, tolerance, rank_cap=None):
    """Smallest rank meeting ``tolerance`` and the error it achieves (spec 3.3.3)."""
    errors = reconstruction_errors(block, rank_cap)
    for rank, error in enumerate(errors, start=1):
        if error <= tolerance:
            return rank, error
    return len(errors), errors[-1]


def existing_d3_contraction(table, receiver, source, receiver_cn, source_cn):
    """Normalized reference weights and the joint contraction of the D3 kernels.

    Mirrors ``normalized_reference_weights`` (``d3_kokkos_impl.tpp:47``) and the joint
    contraction (``d3_kokkos_impl.tpp:120``): Gaussian weights over the valid reference
    slots of each species, normalized per atom, then the bilinear C6 contraction.
    """
    block = table[receiver, source, :, :, 0]
    populated = block > 0.0
    receiver_reference = np.where(populated, table[receiver, source, :, :, 1], 0.0)[
        :, 0
    ]
    source_reference = np.where(populated, table[receiver, source, :, :, 2], 0.0)[0, :]
    receiver_valid = populated.any(axis=1)
    source_valid = populated.any(axis=0)
    receiver_weights = np.where(
        receiver_valid,
        np.exp(-K3_WEIGHT * (receiver_cn - receiver_reference) ** 2),
        0.0,
    )
    source_weights = np.where(
        source_valid,
        np.exp(-K3_WEIGHT * (source_cn - source_reference) ** 2),
        0.0,
    )
    receiver_weights /= receiver_weights.sum()
    source_weights /= source_weights.sum()
    return float(
        np.sum(
            np.outer(receiver_weights, source_weights) * np.where(populated, block, 0.0)
        )
    )


def oracle_factors(factors: fourier_d3_factors.D3LowRankFactors):
    """Task 2's frozen factor container, built from the production result."""
    return fourier_d3.LowRankFactors(
        species=factors.species,
        eigs=factors.eigs,
        v_q=factors.v_q,
        cn_ref=factors.cn_ref,
        valid=factors.valid,
        q_table=np.ones(factors.n_species),
    )


@pytest.fixture
def synthetic():
    """Two species with a known block: Z=1 uses two slots, Z=6 uses all three."""
    species = {1: (2, (0.9118, 0.0, 0.0)), 6: (3, (0.0, 0.9868, 1.9985))}
    blocks = {
        (1, 1): np.array([[1.8, 2.8, 0.0], [2.8, 4.5, 0.0], [0.0, 0.0, 0.0]]),
        (1, 6): np.array([[5.1, 6.2, 7.3], [8.4, 9.5, 10.6], [0.0, 0.0, 0.0]]),
        (6, 1): np.array([[5.1, 6.2, 7.3], [8.4, 9.5, 10.6], [0.0, 0.0, 0.0]]).T,
        (6, 6): np.array([[29.3, 27.5, 22.6], [27.5, 25.8, 21.2], [22.6, 21.2, 17.5]]),
    }
    return synthetic_table(6, 3, species, blocks)


def single_species_table(block, n_ref, count=None):
    """One-species canonical table whose active block is exactly ``block``."""
    if count is None:
        count = n_ref
    return synthetic_table(
        1,
        n_ref,
        {1: (count, tuple(0.5 * slot for slot in range(n_ref)))},
        {(1, 1): block},
    )


def test_full_rank_reconstruction_reproduces_the_block(synthetic):
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [1, 6], table_fingerprint="full", tolerance=1e-12
    )
    block, valid = expected_block(synthetic, [1, 6])
    reconstruction = (
        factors.v_q.reshape(-1, factors.rank) * factors.eigs
    ) @ factors.v_q.reshape(-1, factors.rank).T
    # Z=1 uses two of the three reference slots, so one block row is structurally zero and
    # the full block rank is five, not six.
    assert not valid[0, 2]
    assert factors.rank == 5
    assert np.allclose(reconstruction, block, rtol=1e-12, atol=1e-12)


def test_truncation_returns_the_smallest_rank_meeting_tolerance(synthetic):
    block, _ = expected_block(synthetic, [1, 6])
    for tolerance in (1e-8, 1e-4, 1e-2, 0.5):
        factors = fourier_d3_factors.decompose_d3_c6_reference(
            synthetic, [1, 6], table_fingerprint="scan", tolerance=tolerance
        )
        rank, error = brute_force_rank(block, tolerance)
        assert (factors.rank, factors.max_relative_error) == pytest.approx(
            (rank, error)
        )
        assert factors.tolerance_met is True
        assert factors.max_relative_error <= tolerance


def test_eigenvalues_are_ordered_by_descending_magnitude_and_signed():
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        single_species_table(SIGNED_BLOCK, 3),
        [1],
        table_fingerprint="order",
        tolerance=1e-12,
    )
    # Ascending LAPACK order would start at -1; magnitude order starts at +5.
    assert factors.eigs[0] == pytest.approx(5.0)
    assert np.all(np.diff(np.abs(factors.eigs)) <= 0.0)
    assert np.count_nonzero(factors.eigs < 0.0) == 2


def test_sign_gauge_makes_the_largest_magnitude_positive():
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        single_species_table(POSITIVE_BLOCK, 3),
        [1],
        table_fingerprint="gauge",
        tolerance=1e-12,
    )
    flat = factors.v_q.reshape(-1, factors.rank)
    for slot in range(factors.rank):
        vector = flat[:, slot]
        assert vector[int(np.argmax(np.abs(vector)))] > 0.0


def test_sign_gauge_breaks_ties_by_the_smallest_flat_index():
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        single_species_table(TIED_BLOCK, 2),
        [1],
        table_fingerprint="tie",
        tolerance=1e-12,
    )
    flat = factors.v_q.reshape(-1, factors.rank)
    for slot in range(factors.rank):
        assert flat[0, slot] > 0.0
        assert abs(flat[0, slot]) == abs(flat[1, slot])


def test_rank_scan_is_not_monotone_and_still_returns_the_smallest_rank():
    errors = reconstruction_errors(POSITIVE_BLOCK)
    assert len(errors) == 3
    assert errors[1] > errors[0]
    assert errors[0] <= NON_MONOTONE_TOLERANCE < errors[1]
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        single_species_table(POSITIVE_BLOCK, 3),
        [1],
        table_fingerprint="non-monotone",
        tolerance=NON_MONOTONE_TOLERANCE,
    )
    # Rank 2 is worse than rank 1, so a scan that assumed a monotone error would retain
    # rank 2 and never reach the smallest qualifying rank.
    assert factors.rank == 1
    assert factors.max_relative_error == pytest.approx(errors[0])


def test_invalid_slots_are_masked_and_error_covers_populated_entries_only(synthetic):
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [1, 6], table_fingerprint="mask", tolerance=FIXTURE_TOLERANCE
    )
    block, valid = expected_block(synthetic, [1, 6])
    assert np.array_equal(factors.valid, valid)
    assert np.all(factors.v_q[~valid] == 0.0)
    # The canonical table stores -1 in every invalid slot; an unmasked block would carry
    # those sentinels into the error and the rank.
    unmasked = synthetic[[1, 6]][:, [1, 6], :, :, 0].reshape(6, 6)
    assert unmasked.min() == -1.0
    rank, error = brute_force_rank(block, FIXTURE_TOLERANCE)
    assert factors.rank == rank
    assert factors.max_relative_error == pytest.approx(error)


def test_hard_cap_reports_achieved_error_without_meeting_tolerance(synthetic):
    block, _ = expected_block(synthetic, [1, 6])
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [1, 6], table_fingerprint="cap", tolerance=1e-12, rank_cap=2
    )
    rank, error = brute_force_rank(block, 1e-12, rank_cap=2)
    assert (factors.rank, factors.rank_cap) == (rank, 2)
    assert factors.max_relative_error == pytest.approx(error)
    assert factors.max_relative_error > 1e-12
    assert factors.tolerance_met is False
    assert factors.rank_capped is True


def test_full_rank_is_not_reported_as_capped_when_tolerance_is_unmet(synthetic):
    """A cap equal to the complete block cannot be the reason tolerance was missed."""

    block, _ = expected_block(synthetic, [1, 6])
    full_rank = block.shape[0]
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic,
        [1, 6],
        table_fingerprint="full-rank-cap",
        tolerance=1e-30,
        rank_cap=full_rank,
    )

    assert factors.rank == full_rank
    assert factors.tolerance_met is False
    assert factors.rank_capped is False


def test_cached_factors_are_normalized_read_only_and_deterministic(synthetic):
    fourier_d3_factors.clear_d3_decomposition_cache()
    first = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [6, 1, 6], table_fingerprint="cache", tolerance=FIXTURE_TOLERANCE
    )
    second = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [1, 6], table_fingerprint="cache", tolerance=FIXTURE_TOLERANCE
    )
    assert first is second
    assert first.species.tolist() == [1, 6]
    for array in (first.species, first.eigs, first.v_q, first.cn_ref, first.valid):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]
    fourier_d3_factors.clear_d3_decomposition_cache()
    recomputed = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [1, 6], table_fingerprint="cache", tolerance=FIXTURE_TOLERANCE
    )
    assert recomputed is not first
    assert np.array_equal(recomputed.eigs, first.eigs)
    assert np.array_equal(recomputed.v_q, first.v_q)


def test_cache_key_separates_fingerprint_species_tolerance_and_cap(synthetic):
    fourier_d3_factors.clear_d3_decomposition_cache()
    reference = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [1, 6], table_fingerprint="table-a", tolerance=FIXTURE_TOLERANCE
    )
    variants = (
        {"species": [1, 6], "table_fingerprint": "table-b"},
        {"species": [6], "table_fingerprint": "table-a"},
        {"species": [1, 6], "table_fingerprint": "table-a", "tolerance": 1e-2},
        {"species": [1, 6], "table_fingerprint": "table-a", "rank_cap": 3},
    )
    for variant in variants:
        variant.setdefault("tolerance", FIXTURE_TOLERANCE)
        first = fourier_d3_factors.decompose_d3_c6_reference(synthetic, **variant)
        second = fourier_d3_factors.decompose_d3_c6_reference(synthetic, **variant)
        assert first is second
        assert first is not reference
    repeat = fourier_d3_factors.decompose_d3_c6_reference(
        synthetic, [1, 6], table_fingerprint="table-a", tolerance=FIXTURE_TOLERANCE
    )
    assert repeat is reference
    assert repeat.species.tolist() == [1, 6]


@pytest.mark.parametrize("name", [case["name"] for case in REFERENCE_FIXTURE["cases"]])
def test_factors_match_task2_frozen_fixtures(name):
    case = next(case for case in REFERENCE_FIXTURE["cases"] if case["name"] == name)
    frozen = case["decomposition"]
    table = canonical_tables()["d3_c6"]
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        table,
        frozen["species"],
        table_fingerprint="canonical",
        tolerance=FIXTURE_TOLERANCE,
    )
    assert factors.species.tolist() == frozen["species"]
    assert np.array_equal(factors.valid, np.asarray(frozen["valid"]))
    assert np.allclose(factors.cn_ref, frozen["cn_ref"], rtol=0.0, atol=0.0)
    assert np.allclose(
        factors.eigs, frozen["eigs_ev_angstrom6"], rtol=1e-12, atol=1e-12
    )
    assert np.allclose(factors.v_q, frozen["v_q"], rtol=1e-12, atol=1e-12)
    assert factors.max_relative_error == pytest.approx(
        frozen["max_relative_error"], rel=1e-9
    )


@pytest.mark.parametrize(
    ("receiver", "source"), [(1, 1), (1, 6), (1, 8), (6, 6), (6, 8), (8, 8)]
)
def test_contraction_reproduces_the_existing_d3_interpolation(receiver, source):
    table = canonical_tables()["d3_c6"]
    species = sorted({receiver, source})
    factors = fourier_d3_factors.decompose_d3_c6_reference(
        table, species, table_fingerprint="canonical", tolerance=FIXTURE_TOLERANCE
    )
    container = oracle_factors(factors)
    channels = np.asarray(
        [species.index(z) for z in (receiver, source)], dtype=np.int64
    )
    for receiver_cn, source_cn in CONTRACTION_COORDINATIONS:
        coefficients = fourier_d3.low_rank_coefficients(
            np.asarray([receiver_cn, source_cn]), channels, container
        )
        approximation = float(np.sum(factors.eigs * coefficients[0] * coefficients[1]))
        direct = existing_d3_contraction(
            table, receiver, source, receiver_cn, source_cn
        )
        assert abs(approximation - direct) <= FIXTURE_TOLERANCE * abs(direct)


def test_validation_rejects_bad_inputs(synthetic):
    valid = {"table_fingerprint": "validation"}
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic, [1, 6], table_fingerprint=""
        )
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(synthetic, [], **valid)
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(synthetic, [0, 6], **valid)
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(synthetic, [1.5, 6], **valid)
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(synthetic, [1, 7], **valid)
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic, [1, 6], **valid, tolerance=0.0
        )
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic, [1, 6], **valid, tolerance=float("nan")
        )
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic, [1, 6], **valid, rank_cap=0
        )
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic, [1, 6], **valid, rank_cap=-2
        )
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic[:, :, :, :, 0], [1, 6], **valid
        )
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic[:, :, :, :, :2], [1, 6], **valid
        )
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(synthetic[:5], [1, 6], **valid)
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(
            synthetic, [1, 6], table_fingerprint=7
        )
    empty = synthetic_table(6, 3, {1: (0, ())}, {})
    with pytest.raises(ValueError):
        fourier_d3_factors.decompose_d3_c6_reference(empty, [1], **valid)


def test_installed_module_is_source_independent_and_exports_only_configuration():
    script = (
        "import sys, numpy as np, symmetrix.fourier_d3_factors as f\n"
        "table = np.zeros((3, 3, 2, 2, 3))\n"
        "table[1, 1, :, :, :] = -1.0\n"
        "table[1, 1, 0, 0] = [4.0, 0.0, 0.0]\n"
        "table[1, 1, 1, 1] = [1.0, 1.0, 1.0]\n"
        "factors = f.decompose_d3_c6_reference(table, [1], table_fingerprint='x')\n"
        "assert factors.rank >= 1\n"
        "forbidden = {'torch', 'warp', 'nvalchemiops', 'numpy.fft', 'tools', 'reference'}\n"
        "assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules)\n"
        "assert symmetrix.FourierD3Correction.__name__ == 'FourierD3Correction'\n"
        "assert 'FourierD3Kokkos' not in symmetrix.__all__\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import symmetrix\n" + script],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
