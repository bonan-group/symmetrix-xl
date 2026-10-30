"""Modified coordination-number and separable-coefficient kernels (plan Task 8).

Task 8 of the reciprocal-space plan fixes the modified counting function and the
separable coefficient contraction that the reciprocal forward path consumes:
one switch value and one scalar radial derivative per directed edge, one
coordination number per receiver, and per-atom ``c[i,l]`` and
``dc[i,l]/dtheta`` for a requested rank chunk.

The normative model is specification section 3.2 (the toolkit v0.5.0 steepness
law, *not* ordinary D3 and not the paper's literal Eq. (18)) and section 3.3.5
(masked, exponent-shifted Gaussian reference weights).  Every expectation here
comes from the frozen Task 2 fixtures, the Task 2 NumPy oracle, a hand-derived
endpoint invariant, or a finite difference of the shipped value function - never
from the implementation under test.

These tests drive the private native seam of the installed extension and skip
only when that extension was built without a KokkosFFT FFT provider.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pytest

import symmetrix
from reference import fourier_d3

FIXTURE_DIRECTORY = Path(__file__).parent / "reference" / "fixtures"
REFERENCE_FIXTURE = json.loads(
    (FIXTURE_DIRECTORY / "fourier_d3_reference_cases.json").read_text(encoding="utf-8")
)
CASES = {case["name"]: case for case in REFERENCE_FIXTURE["cases"]}
PARAMETERS = REFERENCE_FIXTURE["parameters"]
CUTOFF = float(PARAMETERS["coordination_cutoff_angstrom"])

# Specification section 3.2 and 3.3.1, restated independently of the native code
# so that a substituted steepness law or an unmasked reference slot fails here.
STEEPNESS = 16.0
UNSCALE = 0.75
K3 = 4.0

PRECISIONS = ("float64", "float32")

# Tolerance per precision.  FP64 kernel arithmetic runs under the historical
# -ffast-math policy, so a few ulp of FMA contraction against the strict-IEEE
# NumPy oracle are expected; FP32 additionally converts every parameter to
# single precision.
TOLERANCE = {
    "float64": {"rtol": 1e-13, "atol": 1e-14},
    "float32": {"rtol": 5e-5, "atol": 1e-6},
}

_SEAM_SYMBOLS = ("_FourierD3CoordinationWorkspace",)

CARBON_RADIUS = 1.0000000494517636
HYDROGEN_RADIUS = 0.426666658747977
# Transition radius of a carbon-carbon edge, used to sample both sides of the
# steepness branch of specification section 3.2.
CARBON_TRANSITION = 0.5 * (UNSCALE * 2 * CARBON_RADIUS + CUTOFF)


def _native():
    module = symmetrix.load_backend()
    if not module._kokkos_is_initialized():
        module._init_kokkos()
    return module


@pytest.fixture(scope="module")
def native():
    module = _native()
    capability = getattr(module, "_fourier_d3_capability", None)
    if not callable(capability):
        pytest.skip(
            "the installed native extension predates the FourierD3 capability "
            "report; rebuild the backend to exercise the coordination contract"
        )
    report = capability()
    if not report["available"]:
        pytest.skip(
            "this build has no FourierD3 FFT provider "
            f"({report['reason'] or 'unreported'}); the coordination seam is not "
            "compiled and nothing is emulated in its place"
        )
    # A build that reports an available provider must carry the coordination
    # seam; nothing stands in for it, so a missing seam is a build defect rather
    # than a skip.
    assert hasattr(module, "_FourierD3CoordinationWorkspace"), (
        "the build reports an available FourierD3 FFT provider, so the native "
        "coordination seam must be present; no host or real-space fallback exists"
    )
    return module


@pytest.fixture(autouse=True)
def _disarm_allocation_fault(native):
    # The fault seam is one-shot state inside the extension: never let a test
    # leave it armed for the next one.
    native._fourier_d3_clear_allocation_fault()
    yield
    native._fourier_d3_clear_allocation_fault()


# --------------------------------------------------------------------------
# fixtures, graphs, and drivers
# --------------------------------------------------------------------------


def _factors(case) -> fourier_d3.LowRankFactors:
    decomposition = case["decomposition"]
    return fourier_d3.LowRankFactors(
        species=np.asarray(decomposition["species"], dtype=np.int64),
        eigs=np.asarray(decomposition["eigs_ev_angstrom6"], dtype=np.float64),
        v_q=np.asarray(decomposition["v_q"], dtype=np.float64),
        cn_ref=np.asarray(decomposition["cn_ref"], dtype=np.float64),
        valid=np.asarray(decomposition["valid"], dtype=bool),
        q_table=np.asarray(case["q_table_angstrom"], dtype=np.float64),
    )


def _synthetic_factors(species, cn_ref, valid, v_q) -> fourier_d3.LowRankFactors:
    v_q = np.asarray(v_q, dtype=np.float64)
    return fourier_d3.LowRankFactors(
        species=np.asarray(species, dtype=np.int64),
        eigs=np.arange(1.0, v_q.shape[2] + 1.0),
        v_q=v_q,
        cn_ref=np.asarray(cn_ref, dtype=np.float64),
        valid=np.asarray(valid, dtype=bool),
        q_table=np.zeros(len(species)),
    )


def _exact_factors(species, cn_ref, valid, v_q) -> fourier_d3.LowRankFactors:
    """Factors whose entries are exactly representable in FP32 as well as FP64.

    Fallback tests compare the contracted coefficients for bitwise equality with
    a factor row, so every entry must survive the single-precision copy.
    """
    array = np.asarray(v_q, dtype=np.float64)
    assert np.all(array == array.astype(np.float32).astype(np.float64))
    return _synthetic_factors(species, cn_ref, valid, array)


def _unit_factors(species, cn_ref, valid, rank):
    return _synthetic_factors(
        species, cn_ref, valid, np.ones((len(species), np.shape(cn_ref)[1], rank))
    )


def _fixture_graph(case):
    """Receiver-major directed graph of one frozen fixture case.

    The Task 2 oracle supplies both orientations of every pair, including
    self-images; this helper only reorders them by receiver and never
    manufactures, mirrors, or halves an edge.  The image shifts are returned
    alongside, because a distance is only meaningful with the shift it came
    from: an edge list carrying images is wrong when paired with zero shifts.
    """
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    edges, shifts = fourier_d3.build_directed_edges(positions, cell, CUTOFF)
    receivers = edges[:, 0]
    neighbours = edges[:, 1]
    distances = np.linalg.norm(
        positions[neighbours] - positions[receivers] + shifts @ cell, axis=1
    )
    return positions, cell, receivers, neighbours, distances, shifts


def _fixture_problem(name):
    case = CASES[name]
    _, _, receivers, neighbours, distances, _ = _fixture_graph(case)
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int32)
    radii = np.asarray(case["covalent_radii_angstrom"], dtype=np.float64)
    return numbers, radii, receivers, neighbours, distances, _factors(case)


def _workspace(native, precision, bound=8):
    return native._FourierD3CoordinationWorkspace(precision, bound)


def _offsets(numbers, receivers):
    counts = np.bincount(np.asarray(receivers, dtype=np.int64), minlength=numbers.size)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)


def _evaluate(
    native,
    precision,
    numbers,
    radii,
    receivers,
    sources,
    distances,
    factors,
    *,
    chunk=None,
    rank_begin=0,
    theta=None,
    cutoff=CUTOFF,
    bound=8,
    workspace=None,
):
    """Load one problem into the private seam and return every output."""
    numbers = np.ascontiguousarray(numbers, dtype=np.int32)
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    receivers = np.asarray(receivers, dtype=np.int64)
    sources = np.ascontiguousarray(sources, dtype=np.int32)
    order = np.argsort(receivers, kind="stable")
    receivers = receivers[order]
    sources = sources[order]
    distances = distances[order]
    offsets = _offsets(numbers, receivers)

    rank = int(factors.v_q.shape[2])
    requested = rank if chunk is None else int(chunk)
    seam = _workspace(native, precision, bound) if workspace is None else workspace
    seam.reserve(
        numbers.size, sources.size, factors.n_species, factors.n_ref, rank, requested
    )
    seam.set_graph(offsets, sources, distances, numbers, radii)
    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    if theta is None:
        seam.run_coordination(cutoff)
    else:
        seam.set_coordination(np.asarray(theta, dtype=np.float64))
    seam.run_coefficients(rank_begin)
    return {
        "workspace": seam,
        "coordination": np.asarray(seam.coordination(), dtype=np.float64),
        "edge_values": np.asarray(seam.edge_values(), dtype=np.float64),
        "edge_radial_derivatives": np.asarray(
            seam.edge_radial_derivatives(), dtype=np.float64
        ),
        "coefficients": np.asarray(seam.coefficients(), dtype=np.float64),
        "coefficient_derivatives": np.asarray(
            seam.coefficient_derivatives(), dtype=np.float64
        ),
        "chunk": requested,
        "rank": rank,
        "sources": sources,
        "distances": distances,
        "numbers": numbers,
    }


def _by_receiver(receivers, *arrays):
    """Reorder per-edge arrays exactly as the seam does: receiver-major, stable.

    The seam walks receivers in order, so a per-edge comparison has to present
    the oracle's edges in that same order rather than in the enumeration order
    of the edge builder.
    """
    order = np.argsort(np.asarray(receivers, dtype=np.int64), kind="stable")
    return [np.asarray(array)[order] for array in arrays]


def _pair_graph(numbers, radii, distance):
    """One receiver-major row: atom 0 has a single neighbour, atom 1."""
    del numbers, radii
    return (
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int32),
        np.array([distance], dtype=np.float64),
    )


def _two_atom_problem():
    numbers = np.array([6, 6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS, CARBON_RADIUS], dtype=np.float64)
    factors = _unit_factors([6], [[0.0, 1.0, 2.0]], [[1, 1, 1]], 2)
    return numbers, radii, factors


# --------------------------------------------------------------------------
# forward value and radial derivative
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_coordination_matches_frozen_fixture_expectation(native, precision, name):
    """The shipped coordination number reproduces the independently frozen values."""
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(name)
    result = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    np.testing.assert_allclose(
        result["coordination"],
        np.asarray(CASES[name]["expected"]["coordination"], dtype=np.float64),
        **TOLERANCE[precision],
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_coordination_matches_the_direct_oracle(native, precision, name):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(name)
    case = CASES[name]
    _, _, oracle_receivers, oracle_neighbours, _, shifts = _fixture_graph(case)
    oracle = fourier_d3.coordination_numbers(
        np.asarray(case["positions_angstrom"], dtype=np.float64),
        radii,
        np.asarray(case["cell_angstrom"], dtype=np.float64),
        CUTOFF,
        edges=np.stack([oracle_receivers, oracle_neighbours], axis=1),
        shifts=shifts,
        active=None,
    )
    result = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    np.testing.assert_allclose(result["coordination"], oracle, **TOLERANCE[precision])


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_edge_switch_values_and_derivatives_match_the_oracle(native, precision, name):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(name)
    result = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    covalent_sum = radii[receivers] + radii[neighbours]
    ordered_distance, ordered_sum = _by_receiver(receivers, distances, covalent_sum)
    np.testing.assert_allclose(
        result["edge_values"],
        fourier_d3.coordination_pair_term(ordered_distance, ordered_sum, CUTOFF),
        **TOLERANCE[precision],
    )
    np.testing.assert_allclose(
        result["edge_radial_derivatives"],
        fourier_d3.coordination_pair_term_derivative(
            ordered_distance, ordered_sum, CUTOFF
        ),
        **TOLERANCE[precision],
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_switch_matches_the_oracle_on_both_sides_of_the_transition_radius(
    native, precision
):
    """Both branches of ``sigma`` are exercised, including the branch point."""
    numbers, radii, factors = _two_atom_problem()
    distances = np.array(
        [
            0.8,
            1.5,
            2.5,
            CARBON_TRANSITION - 1e-6,
            CARBON_TRANSITION,
            CARBON_TRANSITION + 1e-6,
            4.5,
            5.5,
        ]
    )
    receivers = np.zeros(distances.size, dtype=np.int64)
    sources = np.arange(1, distances.size + 1, dtype=np.int32)
    numbers = np.concatenate([numbers, np.full(distances.size - 1, 6, dtype=np.int32)])
    radii = np.concatenate([radii, np.full(distances.size - 1, CARBON_RADIUS)])
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    covalent_sum = np.full(distances.size, 2 * CARBON_RADIUS)
    np.testing.assert_allclose(
        result["edge_values"],
        fourier_d3.coordination_pair_term(distances, covalent_sum, CUTOFF),
        **TOLERANCE[precision],
    )
    np.testing.assert_allclose(
        result["edge_radial_derivatives"],
        fourier_d3.coordination_pair_term_derivative(distances, covalent_sum, CUTOFF),
        **TOLERANCE[precision],
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_switch_value_is_one_half_at_the_covalent_sum(native, precision):
    """Hand-derived invariant: ``theta(C) = 1/2`` for any steepness (spec 3.2.1)."""
    numbers, radii, factors = _two_atom_problem()
    table_sum = 2 * CARBON_RADIUS
    receivers, sources, distances = _pair_graph(numbers, radii, table_sum)
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    tolerance = 1e-12 if precision == "float64" else 1e-6
    assert abs(result["edge_values"][0] - 0.5) <= tolerance
    assert result["coordination"][0] == pytest.approx(0.5, abs=tolerance)
    # The neighbour has no receiver row of its own in this graph.
    assert result["coordination"][1] == 0.0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_switch_reaches_exact_zero_at_the_cutoff(native, precision):
    """Spec 3.2: value and derivative are exactly zero at and beyond the cutoff."""
    numbers, radii, factors = _two_atom_problem()
    inside = np.nextafter(CUTOFF, 0.0)
    outside = np.nextafter(CUTOFF, np.inf)
    distances = np.array([CUTOFF, outside, inside, 2.0 * CUTOFF])
    receivers = np.zeros(4, dtype=np.int64)
    sources = np.full(4, 1, dtype=np.int32)
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    values = result["edge_values"]
    derivatives = result["edge_radial_derivatives"]
    # Exactly zero, not merely small: the mask is a branch, not a limit.
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[3] == 0.0
    assert derivatives[0] == 0.0
    assert derivatives[1] == 0.0
    assert derivatives[3] == 0.0
    # The last representable point inside the cutoff is already underflowed in
    # both precisions, so the approach to zero is continuous by construction.
    assert values[2] == 0.0
    assert derivatives[2] == 0.0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_switch_decays_monotonically_into_the_cutoff(native, precision):
    """The switch is bounded, non-negative, and decreasing towards the cutoff."""
    numbers, radii, factors = _two_atom_problem()
    distances = np.array(
        [
            CUTOFF * 0.5,
            CUTOFF * 0.9,
            CUTOFF * 0.99,
            CUTOFF * 0.999,
            CUTOFF - 1e-3,
            CUTOFF - 1e-6,
        ]
    )
    receivers = np.zeros(distances.size, dtype=np.int64)
    sources = np.full(distances.size, 1, dtype=np.int32)
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    values = result["edge_values"]
    derivatives = result["edge_radial_derivatives"]
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(derivatives))
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)
    assert np.all(np.diff(values) <= 0.0)
    assert np.all(derivatives <= 0.0)


@pytest.mark.parametrize(
    "precision, samples",
    [
        # A saturated switch carries no single-precision finite-difference
        # signal near r = 1, where theta is within one float of unity; the FP32
        # samples therefore start where the switch still has slope.
        ("float64", (1.0, 2.0, CARBON_TRANSITION, 4.5, 5.9)),
        ("float32", (2.0, CARBON_TRANSITION, 4.5, 5.9)),
    ],
)
def test_switch_matches_a_finite_difference(native, precision, samples):
    """The shipped ``dtheta/dr`` differentiates the shipped value function."""
    numbers, radii, factors = _two_atom_problem()
    step = 1e-5 if precision == "float64" else 1e-3
    for sample in samples:
        centre = _evaluate(
            native,
            precision,
            numbers,
            radii,
            *_pair_graph(numbers, radii, sample),
            factors,
        )
        plus = _evaluate(
            native,
            precision,
            numbers,
            radii,
            *_pair_graph(numbers, radii, sample + step),
            factors,
        )
        minus = _evaluate(
            native,
            precision,
            numbers,
            radii,
            *_pair_graph(numbers, radii, sample - step),
            factors,
        )
        finite_difference = (plus["edge_values"][0] - minus["edge_values"][0]) / (
            2.0 * step
        )
        analytic = centre["edge_radial_derivatives"][0]
        if precision == "float64":
            assert abs(analytic - finite_difference) <= 1e-9 + 1e-6 * abs(analytic)
        else:
            assert abs(analytic - finite_difference) <= 1e-6 + 1e-2 * abs(analytic)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_zero_distance_edge_is_inactive(native, precision):
    """``r <= 0`` is masked: a zero-length edge contributes nothing."""
    numbers, radii, factors = _two_atom_problem()
    receivers = np.array([0, 1], dtype=np.int64)
    sources = np.array([1, 0], dtype=np.int32)
    distances = np.array([0.0, 0.0])
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    assert np.all(result["edge_values"] == 0.0)
    assert np.all(result["edge_radial_derivatives"] == 0.0)
    assert np.all(result["coordination"] == 0.0)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_padding_atoms_are_inactive_on_both_sides(native, precision):
    """``Z = 0`` edges are masked without ever being counted as neighbours."""
    numbers = np.array([6, 0], dtype=np.int32)
    radii = np.array([CARBON_RADIUS, CARBON_RADIUS], dtype=np.float64)
    factors = _unit_factors([6], [[0.0, 1.0, 2.0]], [[1, 1, 1]], 2)
    receivers = np.array([0, 1], dtype=np.int64)
    sources = np.array([1, 0], dtype=np.int32)
    distances = np.array([1.0, 1.0])
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    assert np.all(result["edge_values"] == 0.0)
    assert np.all(result["coordination"] == 0.0)
    # The padding atom's coefficients are exactly zero, including the derivative.
    assert np.all(result["coefficients"][1] == 0.0)
    assert np.all(result["coefficient_derivatives"][1] == 0.0)
    # The covered atom keeps its (non-zero) reference contraction at theta = 0.
    assert np.any(result["coefficients"][0] != 0.0)
    # An unmasked law would count these edges, which is what makes the zero above
    # discriminative rather than tautological.
    assert (
        float(fourier_d3.coordination_pair_term(1.0, 2 * CARBON_RADIUS, CUTOFF)) > 0.99
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_receiver_rows_are_accumulated_without_halving(native, precision):
    """Both orientations are walked from their own receiver with weight one."""
    numbers, radii, factors = _two_atom_problem()
    distance = 1.6
    receivers = np.array([0, 1], dtype=np.int64)
    sources = np.array([1, 0], dtype=np.int32)
    distances = np.array([distance, distance])
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    term = float(fourier_d3.coordination_pair_term(distance, 2 * CARBON_RADIUS, CUTOFF))
    assert term > 0.9
    np.testing.assert_allclose(
        result["coordination"], [term, term], **TOLERANCE[precision]
    )
    # A halved or single-orientation accumulation would be visibly wrong.
    assert abs(result["coordination"][0] - 0.5 * term) > 1e-3


@pytest.mark.parametrize("precision", PRECISIONS)
def test_periodic_image_distances_are_used_as_supplied(native, precision):
    """A skewed-cell entry at an image distance is counted at that distance."""
    cell = np.array([[5.0, 0.0, 0.0], [0.4, 4.6, 0.0], [0.2, 0.3, 4.2]])
    positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.0]])
    numbers = np.array([6, 6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS, CARBON_RADIUS], dtype=np.float64)
    factors = _unit_factors([6], [[0.0, 1.0, 2.0]], [[1, 1, 1]], 2)
    edges, shifts = fourier_d3.build_directed_edges(positions, cell, CUTOFF)
    receivers, neighbours = edges[:, 0], edges[:, 1]
    distances = np.linalg.norm(
        positions[neighbours] - positions[receivers] + shifts @ cell, axis=1
    )
    assert receivers.size >= 4
    assert np.any(shifts != 0), "the probe must contain periodic images"
    result = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    oracle = fourier_d3.coordination_numbers(
        positions,
        radii,
        cell,
        CUTOFF,
        edges=edges,
        shifts=shifts,
        active=None,
    )
    np.testing.assert_allclose(result["coordination"], oracle, **TOLERANCE[precision])
    # The image entries are genuinely beyond the base-cell separation, so the
    # comparison is not a disguised nearest-neighbour test.
    direct = float(np.linalg.norm(positions[1] - positions[0]))
    assert float(np.max(distances)) > 3.0 * direct


# --------------------------------------------------------------------------
# separable coefficients and their coordination derivative
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_coefficients_match_the_oracle_from_native_coordination(
    native, precision, name
):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(name)
    result = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    channels = factors.channels(numbers)
    expected = fourier_d3.low_rank_coefficients(
        result["coordination"], channels, factors
    )
    expected_derivative = fourier_d3.low_rank_coefficient_derivative(
        result["coordination"], channels, factors
    )
    np.testing.assert_allclose(result["coefficients"], expected, **TOLERANCE[precision])
    np.testing.assert_allclose(
        result["coefficient_derivatives"], expected_derivative, **TOLERANCE[precision]
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_coefficients_are_pure_functions_of_the_coordination(native, precision, name):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(name)
    result = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    channels = factors.channels(numbers)
    supplied = np.linspace(0.0, 2.5, numbers.size)
    injected = _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        neighbours,
        distances,
        factors,
        theta=supplied,
    )
    np.testing.assert_allclose(
        injected["coefficients"],
        fourier_d3.low_rank_coefficients(supplied, channels, factors),
        **TOLERANCE[precision],
    )
    np.testing.assert_allclose(
        injected["coefficient_derivatives"],
        fourier_d3.low_rank_coefficient_derivative(supplied, channels, factors),
        **TOLERANCE[precision],
    )
    # The injected run really did replace the coordination numbers.
    assert not np.allclose(injected["coordination"], result["coordination"], atol=1e-3)
    np.testing.assert_allclose(injected["coordination"], supplied, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_masked_sentinel_slots_do_not_rescale_coefficients(native, precision):
    """Invalid ``theta_ref = -1`` slots are masked, not merely down-weighted.

    The invalid slots of the hydrogen block carry a large finite factor row, so
    a kernel that trusted the factor values instead of the validity mask would
    inflate the denominator and rescale every retained coefficient.
    """
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(
        "orthogonal-mixed-species"
    )
    polluted = factors.v_q.copy()
    invalid = ~factors.valid
    polluted[invalid] = 1.0e6
    clean = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    dirty = _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        neighbours,
        distances,
        factors=_synthetic_factors(
            factors.species, factors.cn_ref, factors.valid, polluted
        ),
    )
    np.testing.assert_allclose(
        dirty["coefficients"], clean["coefficients"], **TOLERANCE[precision]
    )
    np.testing.assert_allclose(
        dirty["coefficient_derivatives"],
        clean["coefficient_derivatives"],
        **TOLERANCE[precision],
    )
    # The mask is load-bearing: an unmasked contraction of the same input moves
    # the covered coefficients far beyond the tolerance above.
    channels = factors.channels(numbers)
    reference, _ = fourier_d3._channel_arrays(channels, factors)
    delta = dirty["coordination"][:, None] - reference
    weights = np.exp(-K3 * delta**2)
    unmasked = np.einsum("np,npr->nr", weights, polluted[channels])
    unmasked = unmasked / weights.sum(axis=1, keepdims=True)
    hydrogen = np.flatnonzero(channels == 0)
    assert hydrogen.size > 0
    assert np.max(np.abs(unmasked[hydrogen] - clean["coefficients"][hydrogen])) > 1e-6


@pytest.mark.parametrize(
    "precision, huge",
    [
        # The sentinel must exceed the precision's own sqrt(max/2) guard while
        # staying finite in that precision, so that an unmasked scan declares the
        # row unrepresentable and a masked scan does not.
        ("float64", 1.0e300),
        ("float32", 1.0e30),
    ],
)
def test_masked_slots_cannot_trigger_the_fallback(native, precision, huge):
    """A masked sentinel does not make an ordinary coordination number unrepresentable.

    The masked slot holds a magnitude whose square is unrepresentable, so a scan
    that read it would declare the row unrepresentable and fall back to a one-hot
    reference row.  The two valid rows are chosen so their weighted mean is
    neither row alone, which makes that spurious fallback visible.
    """
    numbers = np.array([6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS], dtype=np.float64)
    factors = _synthetic_factors(
        [6],
        [[0.0, 1.0, huge]],
        [[True, True, False]],
        np.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]),
    )
    supplied = np.array([0.5])
    result = _evaluate(
        native,
        precision,
        numbers,
        radii,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        factors,
        theta=supplied,
    )
    expected = fourier_d3.low_rank_coefficients(supplied, np.array([0]), factors)
    expected_derivative = fourier_d3.low_rank_coefficient_derivative(
        supplied, np.array([0]), factors
    )
    np.testing.assert_allclose(result["coefficients"], expected, **TOLERANCE[precision])
    np.testing.assert_allclose(
        result["coefficient_derivatives"], expected_derivative, **TOLERANCE[precision]
    )
    # Equidistant slots make the mean distinguishable from every one-hot row.
    assert not np.allclose(expected, factors.v_q[0, 0])
    assert not np.allclose(expected, factors.v_q[0, 1])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_signed_factors_are_contracted_as_supplied(native, precision):
    """Negative ``v_q`` entries survive the contraction with their sign."""
    numbers = np.array([6, 8], dtype=np.int32)
    radii = np.array([CARBON_RADIUS, 0.8400000011664605], dtype=np.float64)
    v_q = np.zeros((2, 3, 2))
    v_q[0, 0, 0] = 1.5
    v_q[0, 1, 0] = -0.75
    v_q[0, 2, 0] = 0.25
    v_q[1, :, 0] = (-0.5, 0.5, 1.0)
    v_q[1, 0, 1] = 2.0
    factors = _exact_factors(
        [6, 8], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], np.ones((2, 3), dtype=bool), v_q
    )
    receivers = np.array([0, 1], dtype=np.int64)
    sources = np.array([1, 0], dtype=np.int32)
    distances = np.array([1.9, 1.9])
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    channels = factors.channels(numbers)
    np.testing.assert_allclose(
        result["coefficients"],
        fourier_d3.low_rank_coefficients(result["coordination"], channels, factors),
        **TOLERANCE[precision],
    )
    assert np.any(result["coefficients"] < 0.0)
    assert np.any(result["coefficient_derivatives"] != 0.0)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("atom", [0, 1, 2])
def test_coefficient_derivative_matches_a_finite_difference(native, precision, atom):
    """The shipped ``dc/dtheta`` differentiates the shipped coefficients."""
    numbers = np.array([6, 8, 6], dtype=np.int32)
    radii = np.array(
        [CARBON_RADIUS, 0.8400000011664605, CARBON_RADIUS], dtype=np.float64
    )
    cn_ref = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    v_q = np.array(
        [
            [[1.5, 0.25], [-0.75, 0.375], [0.25, -0.125]],
            [[-0.5, 0.3125], [0.5, 0.625], [1.0, -0.25]],
        ]
    )
    factors = _exact_factors([6, 8], cn_ref, np.ones((2, 3), dtype=bool), v_q)
    receivers = np.array([0, 1, 2], dtype=np.int64)
    sources = np.array([1, 2, 0], dtype=np.int32)
    distances = np.array([1.9, 1.7, 2.1])
    supplied = np.array([0.4, 1.1, 2.6])
    step = 1e-5 if precision == "float64" else 1e-3
    moved_plus = supplied.copy()
    moved_plus[atom] += step
    moved_minus = supplied.copy()
    moved_minus[atom] -= step
    centre = _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        sources,
        distances,
        factors,
        theta=supplied,
    )
    plus = _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        sources,
        distances,
        factors,
        theta=moved_plus,
    )
    minus = _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        sources,
        distances,
        factors,
        theta=moved_minus,
    )
    # A central difference: the derivative is a genuine slope of the shipped
    # coefficient function, and a one-sided difference at this step would carry a
    # first-order truncation error an order of magnitude above the tolerance.
    finite_difference = (plus["coefficients"][atom] - minus["coefficients"][atom]) / (
        2.0 * step
    )
    analytic = centre["coefficient_derivatives"][atom]
    if precision == "float64":
        assert np.all(
            np.abs(analytic - finite_difference) <= 1e-9 + 1e-6 * np.abs(analytic)
        )
    else:
        assert np.all(
            np.abs(analytic - finite_difference) <= 1e-5 + 1e-1 * np.abs(analytic)
        )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_uncovered_species_are_zero_but_still_count(native, precision):
    """An uncovered species has zero coefficients yet a normal coordination number."""
    numbers = np.array([6, 14], dtype=np.int32)
    radii = np.array([CARBON_RADIUS, 1.1], dtype=np.float64)
    factors = _unit_factors([6], [[0.0, 1.0, 2.0]], [[1, 1, 1]], 2)
    receivers = np.array([0, 1], dtype=np.int64)
    sources = np.array([1, 0], dtype=np.int32)
    distances = np.array([1.9, 1.9])
    result = _evaluate(
        native, precision, numbers, radii, receivers, sources, distances, factors
    )
    term = float(fourier_d3.coordination_pair_term(1.9, CARBON_RADIUS + 1.1, CUTOFF))
    assert term > 1e-3
    np.testing.assert_allclose(
        result["coordination"], [term, term], **TOLERANCE[precision]
    )
    assert np.all(result["coefficients"][1] == 0.0)
    assert np.all(result["coefficient_derivatives"][1] == 0.0)
    assert np.any(result["coefficients"][0] != 0.0)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("wild", [1.0e300, np.inf, -np.inf, np.nan])
def test_extreme_coordination_falls_back_to_a_valid_reference(native, precision, wild):
    """An unrepresentable argument yields a one-hot valid reference, zero slope.

    ``theta = 1e300`` exceeds both precisions' ``sqrt(max/2)`` guard, and is
    ``inf`` once stored in FP32; every delta is then indistinguishable, so the
    tie rule (smallest valid slot) is what makes the fallback deterministic.
    """
    numbers = np.array([1, 6, 8], dtype=np.int32)
    radii = np.array(
        [HYDROGEN_RADIUS, CARBON_RADIUS, 0.8400000011664605], dtype=np.float64
    )
    factors = _exact_factors(
        [1, 6, 8],
        [[0.0, 1.0, -1.0], [0.0, 1.0, 2.0], [0.0, 1.0, -1.0]],
        [[1, 1, 0], [1, 1, 1], [1, 1, 0]],
        np.arange(1.0, 1.0 + 3 * 3 * 2).reshape(3, 3, 2),
    )
    result = _evaluate(
        native,
        precision,
        numbers,
        radii,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        factors,
        theta=np.full(numbers.size, wild),
    )
    np.testing.assert_allclose(
        result["coefficients"][0], factors.v_q[0, 0], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result["coefficients"][1], factors.v_q[1, 0], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result["coefficients"][2], factors.v_q[2, 0], rtol=0.0, atol=0.0
    )
    assert np.all(result["coefficient_derivatives"] == 0.0)


@pytest.mark.parametrize(
    "precision, huge",
    [
        # 1e300 is finite in FP64 and saturates to infinity in FP32, so the
        # single-precision half of the probe uses a magnitude that is finite
        # there.  Both exceed their precision's sqrt(max/2) guard, and both are
        # squared beyond their precision's range for the slot they do not match,
        # so the nearest rule alone decides the outcome.
        ("float64", 1.0e300),
        ("float32", 1.0e30),
    ],
)
def test_fallback_selects_the_nearest_valid_reference(native, precision, huge):
    """Selection is by ``|theta - theta_ref|``, not by slot order.

    The reference set is a rule probe rather than a physical table: one valid
    slot sits at ``huge``, so a finite coordination number of the same magnitude
    leaves it as the only nearest valid reference while still triggering the
    overflow guard through every other slot.
    """
    numbers = np.array([6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS], dtype=np.float64)
    factors = _synthetic_factors(
        [6],
        [[0.0, huge, -1.0]],
        [[1, 1, 0]],
        np.array([[[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]]),
    )
    result = _evaluate(
        native,
        precision,
        numbers,
        radii,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        factors,
        theta=np.array([huge]),
    )
    np.testing.assert_allclose(
        result["coefficients"][0], factors.v_q[0, 1], rtol=0.0, atol=0.0
    )
    assert np.all(result["coefficient_derivatives"] == 0.0)


@pytest.mark.parametrize("precision, huge", [("float64", 1.0e300), ("float32", 1.0e30)])
def test_masked_slots_cannot_win_the_nearest_rule(native, precision, huge):
    """Fallback selection skips masked slots even when they are nearest by value.

    The masked slot sits exactly at the coordination number, so a selection scan
    that read it would choose it; the valid slot that must win sits one index
    later, which is what makes the assertion discriminative.
    """
    numbers = np.array([6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS], dtype=np.float64)
    factors = _synthetic_factors(
        [6],
        [[0.0, huge, huge]],
        [[True, False, True]],
        np.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]),
    )
    result = _evaluate(
        native,
        precision,
        numbers,
        radii,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        factors,
        theta=np.array([huge]),
    )
    np.testing.assert_allclose(
        result["coefficients"][0], factors.v_q[0, 2], rtol=0.0, atol=0.0
    )
    assert np.all(result["coefficient_derivatives"] == 0.0)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("wild", [np.inf, -np.inf, np.nan])
def test_non_finite_coordination_falls_back_deterministically(native, precision, wild):
    """A non-finite coordination number falls back to the first valid reference."""
    numbers = np.array([6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS], dtype=np.float64)
    factors = _exact_factors(
        [6],
        [[0.0, 1.0, -1.0]],
        [[1, 1, 0]],
        np.arange(1.0, 1.0 + 3 * 2).reshape(1, 3, 2),
    )
    result = _evaluate(
        native,
        precision,
        numbers,
        radii,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        factors,
        theta=np.full(1, wild),
    )
    np.testing.assert_allclose(
        result["coefficients"][0], factors.v_q[0, 0], rtol=0.0, atol=0.0
    )
    assert np.all(result["coefficient_derivatives"] == 0.0)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reference_row_keeps_the_exponent_shift_stable(native, precision):
    """The shift keeps the largest weight at one, so the denominator cannot underflow.

    This is the shift's actual job, and it is invisible in the contracted
    coefficients: the shift cancels algebraically in ``c_l`` and in
    ``dc_l/dtheta`` whenever the fallback is correct, so only the row itself can
    show that the sum stayed representable.
    """
    seam = _workspace(native, precision, bound=8)
    seam.reserve(1, 0, 1, 3, 1, 1)
    row = seam.debug_reference_row(
        np.array([0.0, 1.0, 2.0]), np.array([True, True, True]), 50.0
    )
    weights = np.asarray(row["weight"], dtype=np.float64)
    denominator = float(row["denominator"])
    assert np.all(np.isfinite(weights))
    assert float(np.max(weights)) == 1.0
    assert denominator >= 1.0
    assert int(row["fallback_slot"]) == -1
    assert int(np.argmax(weights)) == 2
    # Unshifted, every weight of this row underflows to zero: the denominator
    # would be zero and the contraction would divide by it.
    unshifted = np.exp(-K3 * (50.0 - np.array([0.0, 1.0, 2.0])) ** 2)
    assert float(np.sum(unshifted)) == 0.0
    # A representable row is not a fallback at all, so the probe distinguishes the
    # two cases rather than reporting a slot unconditionally.
    ordinary = seam.debug_reference_row(
        np.array([0.0, 1.0, 2.0]), np.array([True, True, True]), 0.5
    )
    assert int(ordinary["fallback_slot"]) == -1
    assert float(ordinary["denominator"]) >= 1.0


@pytest.mark.parametrize("precision, huge", [("float64", 1.0e300), ("float32", 1.0e30)])
def test_fallback_rows_always_carry_a_valid_slot(native, precision, huge):
    """A fallback row names a valid slot, and never an invalid one or none at all.

    The nearest-slot field is what the fallback branch must carry out of the scan;
    a branch that clears the row before reading it would report ``-1`` and leave
    the contraction to divide by a zero denominator.
    """
    seam = _workspace(native, precision, bound=8)
    seam.reserve(1, 0, 1, 3, 1, 1)
    cases = (
        # every delta unrepresentable: the tie keeps the smallest valid slot
        ([0.0, 1.0, 2.0], [True, True, True], huge, 0),
        # the nearest slot by value is masked, so it cannot be selected
        ([0.0, huge, huge], [True, False, True], huge, 2),
        # a masked slot sits nearest to a large negative coordination number
        ([-1.0, 0.0, 1.0], [False, True, True], -1.0e300, 1),
    )
    for cn_ref, valid, coordination, expected in cases:
        row = seam.debug_reference_row(
            np.array(cn_ref), np.array(valid, dtype=bool), coordination
        )
        slot = int(row["fallback_slot"])
        assert slot == expected
        assert int(row["nearest_slot"]) == expected
        assert bool(valid[slot]), "a fallback row must name a valid slot"
        assert np.all(np.asarray(row["weight"], dtype=np.float64) == 0.0)
        assert np.all(np.asarray(row["weight_derivative"], dtype=np.float64) == 0.0)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_underflowed_shifted_weights_reproduce_the_nearest_reference_row(
    native, precision
):
    """A far coordination number keeps a finite, nearest-reference coefficient."""
    numbers = np.array([6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS], dtype=np.float64)
    factors = _exact_factors(
        [6],
        [[0.0, 1.0, 2.0]],
        [[1, 1, 1]],
        np.array([[[1.5, -0.25], [-0.75, 0.5], [0.25, 1.0]]]),
    )
    result = _evaluate(
        native,
        precision,
        numbers,
        radii,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        factors,
        theta=np.array([50.0]),
    )
    # Without the exponent shift every weight would underflow and the
    # denominator would be zero; with it the nearest reference wins exactly.
    np.testing.assert_allclose(
        result["coefficients"][0], factors.v_q[0, 2], rtol=0.0, atol=0.0
    )
    assert np.all(np.abs(result["coefficient_derivatives"]) <= 1e-3)


# --------------------------------------------------------------------------
# rank chunking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_rank_chunks_reproduce_the_full_rank_contraction(native, precision):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(
        "coordination-sensitive"
    )
    full = _evaluate(
        native, precision, numbers, radii, receivers, neighbours, distances, factors
    )
    rank = full["rank"]
    collected = np.zeros((numbers.size, rank))
    collected_derivative = np.zeros((numbers.size, rank))
    for begin in range(rank):
        chunk = _evaluate(
            native,
            precision,
            numbers,
            radii,
            receivers,
            neighbours,
            distances,
            factors,
            chunk=1,
            rank_begin=begin,
        )
        collected[:, begin] = chunk["coefficients"][:, 0]
        collected_derivative[:, begin] = chunk["coefficient_derivatives"][:, 0]
    np.testing.assert_allclose(collected, full["coefficients"], **TOLERANCE[precision])
    np.testing.assert_allclose(
        collected_derivative, full["coefficient_derivatives"], **TOLERANCE[precision]
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_coefficient_storage_is_bounded_by_the_rank_chunk(native, precision):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(
        "orthogonal-mixed-species"
    )
    rank = int(factors.v_q.shape[2])
    seam = _workspace(native, precision, bound=rank)
    _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        neighbours,
        distances,
        factors,
        chunk=1,
        workspace=seam,
    )
    narrow = {buffer["name"]: buffer for buffer in seam.usage()["buffers"]}
    assert narrow["coefficients"]["active_elements"] == numbers.size
    _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        neighbours,
        distances,
        factors,
        chunk=rank,
        workspace=seam,
    )
    wide = {buffer["name"]: buffer for buffer in seam.usage()["buffers"]}
    assert wide["coefficients"]["active_elements"] == numbers.size * rank
    # Capacity tracks the widest chunk actually requested, not the total rank of
    # the supplied factors.
    assert wide["coefficients"]["capacity_elements"] == numbers.size * rank


def test_chunk_wider_than_the_configured_bound_is_rejected(native):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(
        "orthogonal-mixed-species"
    )
    rank = int(factors.v_q.shape[2])
    seam = _workspace(native, "float64", bound=rank - 1)
    with pytest.raises(ValueError, match="rank chunk bound"):
        _evaluate(
            native,
            "float64",
            numbers,
            radii,
            receivers,
            neighbours,
            distances,
            factors,
            chunk=rank,
            workspace=seam,
        )
    assert seam.usage()["allocations"] == 0


def test_rank_begin_beyond_the_supplied_rank_is_rejected(native):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(
        "orthogonal-mixed-species"
    )
    rank = int(factors.v_q.shape[2])
    seam = _workspace(native, "float64", bound=rank)
    _evaluate(
        native,
        "float64",
        numbers,
        radii,
        receivers,
        neighbours,
        distances,
        factors,
        chunk=1,
        rank_begin=rank - 1,
        workspace=seam,
    )
    seam.reserve(
        numbers.size,
        neighbours.size,
        factors.n_species,
        factors.n_ref,
        rank,
        2,
    )
    with pytest.raises(ValueError, match="rank chunk"):
        seam.run_coefficients(rank - 1)
    with pytest.raises(ValueError, match="rank chunk"):
        seam.run_coefficients(-1)


# --------------------------------------------------------------------------
# rank window bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_rank_begin_at_the_integer_limit_is_rejected(native, precision):
    """A rank window guard must not add before it compares.

    ``rank_begin = INT64_MAX`` makes ``rank_begin + chunk`` overflow, so a guard
    written as ``rank_begin + chunk > rank_total`` cannot see the out-of-range
    window and launches a read at an unrelated index.  The rejected call must
    also leave the workspace untouched: no launch, no rewritten output.
    """
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(
        "orthogonal-mixed-species"
    )
    rank = int(factors.v_q.shape[2])
    seam = _workspace(native, precision, bound=rank)
    result = _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        neighbours,
        distances,
        factors,
        workspace=seam,
    )
    assert np.any(result["coefficients"] != 0.0)
    baseline = np.array(seam.coefficients(), copy=True)
    baseline_derivative = np.array(seam.coefficient_derivatives(), copy=True)
    usage = seam.usage()
    geometry = seam.geometry()

    limit = 2**63 - 1
    # rank_total == chunk here, so the only representable window starts at zero.
    for begin in (limit, limit - 1, 2**62, 2**31, rank, rank + 1, -1, -(2**63)):
        with pytest.raises(ValueError, match="rank chunk"):
            seam.run_coefficients(begin)

    np.testing.assert_array_equal(seam.coefficients(), baseline)
    np.testing.assert_array_equal(seam.coefficient_derivatives(), baseline_derivative)
    assert seam.usage() == usage
    assert seam.geometry() == geometry


@pytest.mark.parametrize("precision", PRECISIONS)
def test_rank_window_boundary_is_exact(native, precision):
    """The last full window is accepted and the next start is not."""
    numbers = np.array([6, 6], dtype=np.int32)
    radii = np.array([CARBON_RADIUS, CARBON_RADIUS], dtype=np.float64)
    factors = _unit_factors([6], [[0.0, 1.0, 2.0]], [[True, True, True]], 5)
    receivers = np.array([0, 1], dtype=np.int64)
    sources = np.array([1, 0], dtype=np.int32)
    distances = np.array([1.6, 1.6])
    last = _evaluate(
        native,
        precision,
        numbers,
        radii,
        receivers,
        sources,
        distances,
        factors,
        chunk=2,
        rank_begin=3,
    )
    assert last["coefficients"].shape == (numbers.size, 2)
    seam = last["workspace"]
    baseline = np.array(seam.coefficients(), copy=True)
    for begin in (4, 5, 6, 2**63 - 1):
        with pytest.raises(ValueError, match="rank chunk"):
            seam.run_coefficients(begin)
    np.testing.assert_array_equal(seam.coefficients(), baseline)


def test_rank_chunk_wider_than_the_supplied_rank_is_rejected_at_reserve(native):
    """A chunk wider than the whole rank is rejected before anything is allocated."""
    seam = _workspace(native, "float64", bound=8)
    with pytest.raises(ValueError, match="rank chunk"):
        seam.reserve(2, 1, 1, 3, 2, 3)
    assert seam.usage()["allocations"] == 0
    assert seam.geometry()["active_nodes"] == 0
    assert seam.usage()["capacity_bytes"] == 0
    # A chunk equal to the rank is the widest legal request.
    assert seam.reserve(2, 1, 1, 3, 2, 2)["grown"] is True


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _valid_problem(native, precision="float64", bound=8):
    numbers, radii, receivers, neighbours, distances, factors = _fixture_problem(
        "orthogonal-mixed-species"
    )
    seam = _workspace(native, precision, bound)
    order = np.argsort(receivers, kind="stable")
    seam.reserve(
        numbers.size,
        neighbours.size,
        factors.n_species,
        factors.n_ref,
        int(factors.v_q.shape[2]),
        int(factors.v_q.shape[2]),
    )
    return (
        seam,
        np.ascontiguousarray(numbers, dtype=np.int32),
        radii,
        _offsets(numbers, receivers),
        np.ascontiguousarray(neighbours[order], dtype=np.int32),
        np.ascontiguousarray(distances[order], dtype=np.float64),
        factors,
    )


def _reserve_geometry(seam, rank_chunk):
    """The workspace's own geometry, re-declared with a different chunk width."""
    geometry = seam.geometry()
    return (
        geometry["active_nodes"],
        geometry["active_edges"],
        geometry["n_species"],
        geometry["n_ref"],
        geometry["rank_total"],
        rank_chunk,
    )


def _coefficient_allocation_id(seam):
    return {b["name"]: b["allocation_id"] for b in seam.usage()["buffers"]}[
        "coefficients"
    ]


def test_graph_validation_rejects_malformed_offsets(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    bad_last = offsets.copy()
    bad_last[-1] += 1
    with pytest.raises(ValueError, match="offsets"):
        seam.set_graph(bad_last, sources, distances, numbers, radii)
    bad_first = offsets.copy()
    bad_first[0] = 1
    with pytest.raises(ValueError, match="offsets"):
        seam.set_graph(bad_first, sources, distances, numbers, radii)
    non_monotone = offsets.copy()
    non_monotone[1] = non_monotone[-1]
    with pytest.raises(ValueError, match="offsets"):
        seam.set_graph(non_monotone, sources, distances, numbers, radii)
    wrong_length = np.zeros(offsets.size + 1, dtype=np.int32)
    with pytest.raises(ValueError, match="offsets"):
        seam.set_graph(wrong_length, sources, distances, numbers, radii)


def test_graph_validation_rejects_out_of_range_sources(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    bad = sources.copy()
    bad[0] = numbers.size
    with pytest.raises(ValueError, match="sources"):
        seam.set_graph(offsets, bad, distances, numbers, radii)
    negative = sources.copy()
    negative[0] = -1
    with pytest.raises(ValueError, match="sources"):
        seam.set_graph(offsets, negative, distances, numbers, radii)


def test_graph_validation_rejects_mismatched_lengths(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    with pytest.raises(ValueError):
        seam.set_graph(offsets, sources, distances[:-1], numbers, radii)
    with pytest.raises(ValueError):
        seam.set_graph(offsets, sources, distances, numbers[:-1], radii)
    with pytest.raises(ValueError):
        seam.set_graph(offsets, sources, distances, numbers, radii[:-1])
    with pytest.raises(ValueError):
        seam.set_graph(offsets, sources[:-1], distances, numbers, radii)


def test_non_finite_geometry_values_are_rejected(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    for wild in (np.nan, np.inf):
        poisoned = distances.copy()
        poisoned[0] = wild
        with pytest.raises(ValueError, match="distance"):
            seam.set_graph(offsets, sources, poisoned, numbers, radii)
        poisoned_radius = radii.copy()
        poisoned_radius[0] = wild
        with pytest.raises(ValueError, match="covalent"):
            seam.set_graph(offsets, sources, distances, numbers, poisoned_radius)
    negative = distances.copy()
    negative[0] = -1.0
    with pytest.raises(ValueError, match="distance"):
        seam.set_graph(offsets, sources, negative, numbers, radii)


def test_coordination_cutoff_must_be_finite_and_positive(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    seam.set_graph(offsets, sources, distances, numbers, radii)
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="cutoff"):
            seam.run_coordination(bad)


@pytest.mark.parametrize("value", [2**32, 2**31, 2**40, 2**63 - 1])
def test_index_buffers_reject_values_that_do_not_fit_int32(native, value):
    """An integer buffer wider than int32 is range-checked, never wrapped.

    ``2**32`` truncates to ``0`` under a silent narrowing cast, which is a
    *valid* index, so the graph would be accepted with a wrong neighbour rather
    than rejected.
    """
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)

    wide_sources = sources.astype(np.int64)
    wide_sources[0] = value
    with pytest.raises(ValueError, match="int32"):
        seam.set_graph(offsets, wide_sources, distances, numbers, radii)

    wide_offsets = offsets.astype(np.int64)
    wide_offsets[-1] = value
    with pytest.raises(ValueError, match="int32"):
        seam.set_graph(wide_offsets, sources, distances, numbers, radii)

    wide_numbers = numbers.astype(np.int64)
    wide_numbers[0] = value
    with pytest.raises(ValueError, match="int32"):
        seam.set_graph(offsets, sources, distances, wide_numbers, radii)

    wide_species = factors.species.astype(np.int64)
    wide_species[0] = value
    with pytest.raises(ValueError, match="int32"):
        seam.set_reference(factors.cn_ref, factors.valid, wide_species, factors.v_q)

    # The same buffers at a width that fits are accepted unchanged.
    seam.set_graph(
        offsets.astype(np.int64),
        sources.astype(np.int64),
        distances,
        numbers.astype(np.int64),
        radii,
    )
    seam.set_reference(
        factors.cn_ref,
        factors.valid,
        factors.species.astype(np.int64),
        factors.v_q,
    )


@pytest.mark.parametrize("value", [2**63, 2**64 - 8, 2**64 - 1])
def test_unsigned_index_buffers_are_range_checked_before_narrowing(native, value):
    """An unsigned value at or above 2**63 is named, not wrapped into an index.

    Casting a wide unsigned buffer straight to int64 turns the top of the uint64
    range into a negative value that an int32 range check would accept and the
    workspace would then reject with an unrelated message.
    """
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)

    wide_sources = sources.astype(np.uint64)
    wide_sources[0] = value
    with pytest.raises(ValueError, match="int32"):
        seam.set_graph(offsets, wide_sources, distances, numbers, radii)

    wide_offsets = offsets.astype(np.uint64)
    wide_offsets[-1] = value
    with pytest.raises(ValueError, match="int32"):
        seam.set_graph(wide_offsets, sources, distances, numbers, radii)

    wide_species = factors.species.astype(np.uint64)
    wide_species[0] = value
    with pytest.raises(ValueError, match="int32"):
        seam.set_reference(factors.cn_ref, factors.valid, wide_species, factors.v_q)

    # An in-range unsigned buffer is still accepted unchanged.
    seam.set_graph(
        offsets.astype(np.uint64),
        sources.astype(np.uint64),
        distances,
        numbers.astype(np.uint64),
        radii,
    )


def test_reference_validation_rejects_malformed_input(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    with pytest.raises(ValueError, match="cn_ref"):
        seam.set_reference(
            factors.cn_ref[:-1], factors.valid, factors.species, factors.v_q
        )
    with pytest.raises(ValueError, match="valid"):
        seam.set_reference(
            factors.cn_ref, factors.valid[:-1], factors.species, factors.v_q
        )
    with pytest.raises(ValueError, match="v_q"):
        seam.set_reference(
            factors.cn_ref, factors.valid, factors.species, factors.v_q[:, :, :-1]
        )
    with pytest.raises(ValueError, match="species"):
        seam.set_reference(
            factors.cn_ref, factors.valid, factors.species[::-1], factors.v_q[::-1]
        )
    duplicated = np.array([1, 1, 8], dtype=np.int64)
    with pytest.raises(ValueError, match="species"):
        seam.set_reference(factors.cn_ref, factors.valid, duplicated, factors.v_q)


def test_reference_validation_rejects_non_finite_factors(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    for wild in (np.nan, np.inf):
        poisoned = factors.v_q.copy()
        poisoned[0, 0, 0] = wild
        with pytest.raises(ValueError, match="v_q"):
            seam.set_reference(factors.cn_ref, factors.valid, factors.species, poisoned)
        poisoned_ref = factors.cn_ref.copy()
        poisoned_ref[0, 0] = wild
        with pytest.raises(ValueError, match="cn_ref"):
            seam.set_reference(
                poisoned_ref, factors.valid, factors.species, factors.v_q
            )


def test_reference_validation_requires_a_valid_slot_per_species(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    mask = factors.valid.copy()
    mask[0] = False
    with pytest.raises(ValueError, match="valid reference"):
        seam.set_reference(factors.cn_ref, mask, factors.species, factors.v_q)


def test_reference_slot_count_is_bounded(native):
    seam = _workspace(native, "float64")
    with pytest.raises(ValueError, match="reference slots"):
        seam.reserve(2, 1, 1, 8, 1, 1)


def test_outputs_require_a_loaded_problem(native):
    seam = _workspace(native, "float64")
    with pytest.raises(RuntimeError):
        seam.run_coordination(CUTOFF)
    seam.reserve(2, 1, 1, 3, 2, 2)
    with pytest.raises(RuntimeError):
        seam.run_coordination(CUTOFF)
    with pytest.raises(RuntimeError):
        seam.run_coefficients(0)
    with pytest.raises(RuntimeError):
        seam.coordination()


def test_wide_counts_and_products_are_rejected_before_allocating(native):
    seam = _workspace(native, "float64", bound=1 << 40)
    # The low-rank factor table overflows its own 64-bit product.
    with pytest.raises(OverflowError):
        seam.reserve(2, 1, 1, 5, 1 << 62, 1)
    # A factor table whose element count leaves the flat index range is rejected
    # by its count, not truncated.
    with pytest.raises(OverflowError):
        seam.reserve(2, 1, 1, 1, 1 << 40, 1 << 40)
    with pytest.raises(ValueError, match="rank chunk bound"):
        _workspace(native, "float64", bound=2).reserve(2, 1, 1, 3, 2, 3)
    with pytest.raises(ValueError, match="rank chunk"):
        _workspace(native, "float64", bound=8).reserve(2, 1, 1, 3, 2, 3)
    assert seam.usage()["allocations"] == 0


def test_flat_buffer_counts_above_the_index_range_are_rejected(native):
    seam = _workspace(native, "float64", bound=1 << 31)
    with pytest.raises(OverflowError, match="index"):
        seam.reserve(1 << 31, 1, 1, 1, 1, 1)
    assert seam.usage()["allocations"] == 0


# --------------------------------------------------------------------------
# workspace grow-only accounting, failure atomicity, and privacy
# --------------------------------------------------------------------------


def test_workspace_is_grow_only_and_reports_capacity(native):
    seam = _workspace(native, "float64")
    assert seam.geometry()["rank_chunk_bound"] == 8
    initial = seam.usage()
    assert initial["allocations"] == 0
    assert initial["capacity_bytes"] == 0

    small = seam.reserve(4, 8, 1, 3, 4, 2)
    assert small["grown"] is True
    narrow = seam.usage()
    assert narrow["geometry"]["active_nodes"] == 4
    assert narrow["geometry"]["active_edges"] == 8
    assert narrow["geometry"]["active_rank_chunk"] == 2
    assert narrow["allocations"] > 0
    assert narrow["capacity_bytes"] == sum(
        buffer["capacity_bytes"] for buffer in narrow["buffers"]
    )
    assert narrow["active_bytes"] == sum(
        buffer["active_bytes"] for buffer in narrow["buffers"]
    )
    assert narrow["capacity_bytes"] >= narrow["active_bytes"]

    larger = seam.reserve(9, 17, 1, 3, 4, 3)
    assert larger["grown"] is True
    wide = seam.usage()
    assert wide["capacity_bytes"] > narrow["capacity_bytes"]

    smaller = seam.reserve(2, 3, 1, 3, 4, 1)
    assert smaller["grown"] is False
    reused = seam.usage()
    # Capacity is the high-water mark; active sizes follow the new request.
    assert reused["capacity_bytes"] == wide["capacity_bytes"]
    assert reused["geometry"]["active_nodes"] == 2
    assert reused["geometry"]["active_edges"] == 3
    assert reused["geometry"]["active_rank_chunk"] == 1
    assert reused["reservations"] == 3
    ids_before = {b["name"]: b["allocation_id"] for b in reused["buffers"]}
    seam.reserve(3, 4, 1, 3, 4, 2)
    assert {
        b["name"]: b["allocation_id"] for b in seam.usage()["buffers"]
    } == ids_before


@pytest.mark.parametrize("precision", PRECISIONS)
def test_coefficients_are_not_readable_after_an_input_change(native, precision):
    """A changed input invalidates the retained coefficients instead of returning them."""
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(
        native, precision
    )
    seam.set_graph(offsets, sources, distances, numbers, radii)
    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    seam.run_coordination(CUTOFF)
    seam.run_coefficients(0)
    fresh = np.array(seam.coefficients(), copy=True)
    assert np.any(fresh != 0.0)

    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    with pytest.raises(RuntimeError, match="coefficient"):
        seam.coefficients()
    with pytest.raises(RuntimeError, match="coefficient"):
        seam.coefficient_derivatives()
    seam.run_coefficients(0)
    np.testing.assert_array_equal(seam.coefficients(), fresh)

    seam.set_coordination(np.zeros(numbers.size, dtype=np.float64))
    with pytest.raises(RuntimeError, match="coefficient"):
        seam.coefficients()
    seam.run_coefficients(0)
    assert seam.coefficients().shape == fresh.shape


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_chunk_only_reserve_invalidates_the_retained_coefficients(native, precision):
    """Reserving a new chunk re-declares the coefficient geometry.

    Neither direction of a chunk-only change leaves the retained buffer readable:
    growing reallocates it, so a stale read would return unwritten storage, and
    shrinking keeps the storage but changes the node stride, so a stale read would
    return another rank channel's value.
    """
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(
        native, precision
    )
    seam.set_graph(offsets, sources, distances, numbers, radii)
    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    seam.run_coordination(CUTOFF)
    seam.run_coefficients(0)
    full = np.array(seam.coefficients(), copy=True)
    rank = int(full.shape[1])
    assert rank == int(factors.v_q.shape[2])
    full_id = _coefficient_allocation_id(seam)

    # shrink: the storage survives, the stride does not
    seam.reserve(*_reserve_geometry(seam, 1))
    assert seam.geometry()["active_rank_chunk"] == 1
    assert _coefficient_allocation_id(seam) == full_id
    with pytest.raises(RuntimeError, match="coefficient"):
        seam.coefficients()
    with pytest.raises(RuntimeError, match="coefficient"):
        seam.coefficient_derivatives()
    seam.run_coefficients(0)
    narrow = np.array(seam.coefficients(), copy=True)
    assert narrow.shape == (numbers.size, 1)
    np.testing.assert_allclose(narrow, full[:, :1], **TOLERANCE[precision])

    # grow: the coefficient buffers are reallocated for the wider window
    seam.reserve(*_reserve_geometry(seam, rank))
    with pytest.raises(RuntimeError, match="coefficient"):
        seam.coefficients()
    seam.run_coefficients(0)
    np.testing.assert_allclose(seam.coefficients(), full, **TOLERANCE[precision])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_failed_reserve_keeps_the_retained_coefficients_readable(native, precision):
    """The all-or-nothing guarantee covers output validity, not only storage.

    Invalidation belongs in the commit path: a reserve rejected by validation or
    by an allocation failure must leave the previous outputs readable and
    byte-identical.
    """
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(
        native, precision
    )
    seam.set_graph(offsets, sources, distances, numbers, radii)
    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    seam.run_coordination(CUTOFF)
    seam.run_coefficients(0)
    baseline = np.array(seam.coefficients(), copy=True)
    baseline_derivative = np.array(seam.coefficient_derivatives(), copy=True)
    geometry = seam.geometry()
    rank = int(geometry["rank_total"])

    with pytest.raises(ValueError, match="rank chunk"):
        seam.reserve(*_reserve_geometry(seam, rank + 1))
    np.testing.assert_array_equal(seam.coefficients(), baseline)

    native._fourier_d3_arm_allocation_fault(0)
    try:
        with pytest.raises(RuntimeError, match="injected allocation fault"):
            seam.reserve(
                geometry["active_nodes"] + 8,
                geometry["active_edges"],
                geometry["n_species"],
                geometry["n_ref"],
                rank,
                rank,
            )
    finally:
        native._fourier_d3_clear_allocation_fault()

    assert seam.geometry() == geometry
    np.testing.assert_array_equal(seam.coefficients(), baseline)
    np.testing.assert_array_equal(seam.coefficient_derivatives(), baseline_derivative)


@pytest.mark.parametrize("fail_at", [0, 1, 2, 5, 13])
def test_failed_reserve_leaves_the_workspace_unchanged(native, fail_at):
    seam = _workspace(native, "float64")
    seam.reserve(4, 8, 1, 3, 2, 2)
    baseline = seam.usage()
    baseline_geometry = seam.geometry()

    native._fourier_d3_arm_allocation_fault(fail_at)
    with pytest.raises(RuntimeError, match="injected allocation fault"):
        seam.reserve(12, 24, 2, 4, 3, 3)
    native._fourier_d3_clear_allocation_fault()

    after = seam.usage()
    assert after["capacity_bytes"] == baseline["capacity_bytes"]
    assert after["allocations"] == baseline["allocations"]
    assert after["reservations"] == baseline["reservations"]
    assert seam.geometry() == baseline_geometry
    assert [b["allocation_id"] for b in after["buffers"]] == [
        b["allocation_id"] for b in baseline["buffers"]
    ]


def test_reloading_a_graph_reuses_the_retained_allocation(native):
    seam, numbers, radii, offsets, sources, distances, factors = _valid_problem(native)
    seam.set_graph(offsets, sources, distances, numbers, radii)
    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    first = seam.usage()
    seam.set_graph(offsets, sources, distances, numbers, radii)
    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    second = seam.usage()
    assert second["allocations"] == first["allocations"]
    assert second["capacity_bytes"] == first["capacity_bytes"]
    assert [b["allocation_id"] for b in second["buffers"]] == [
        b["allocation_id"] for b in first["buffers"]
    ]


def test_coordination_workspace_is_released_before_kokkos_finalization(native):
    gc.collect()
    baseline = native._kokkos_live_object_count()
    seam = _workspace(native, "float64")
    assert native._kokkos_live_object_count() == baseline + 1
    seam.reserve(4, 8, 1, 3, 2, 2)
    with pytest.raises(RuntimeError, match="finalize"):
        native._finalize_kokkos()
    del seam
    gc.collect()
    assert native._kokkos_live_object_count() == baseline


def test_coordination_seam_is_private_while_configuration_is_public(native):
    for symbol in _SEAM_SYMBOLS:
        assert symbol not in symmetrix.__all__
    assert "FourierD3Correction" in symmetrix.__all__
    assert symmetrix.FourierD3Correction.__name__ == "FourierD3Correction"
    # Task 11 adds the native evaluator class, and the package's fallback forwards any
    # name the native module provides, so reachability no longer says "private".
    # Publication does: the curated export list and __all__ are unchanged.
    assert "FourierD3Kokkos" not in symmetrix.__all__
    assert "FourierD3Kokkos" not in symmetrix._FRONTEND_EXPORTS


# --------------------------------------------------------------------------
# mutation checks: substituted laws must fail these tolerances
# --------------------------------------------------------------------------


def _ordinary_d3_pair_term(distance, covalent_sum):
    """The standard D3 counting law (P.11), which the model deliberately replaces."""
    ratio = covalent_sum / distance
    return 1.0 / (1.0 + np.exp(-STEEPNESS * (ratio - 1.0)))


def _paper_eq18_pair_term(distance, covalent_sum, cutoff):
    """The paper's literal Eq. (18) steepness, which is non-zero at the cutoff."""
    transition = 0.5 * (UNSCALE * covalent_sum + cutoff)
    steepness = STEEPNESS + ((distance - transition) / (cutoff - transition)) ** 2
    argument = steepness * (covalent_sum / distance - 1.0)
    exponential = np.exp(-np.abs(argument))
    return np.where(
        argument >= 0.0, 1.0 / (1.0 + exponential), exponential / (1.0 + exponential)
    )


def _swept_graph(count=40):
    numbers = np.full(count + 1, 6, dtype=np.int32)
    radii = np.full(count + 1, CARBON_RADIUS, dtype=np.float64)
    receivers = np.zeros(count, dtype=np.int64)
    sources = np.arange(1, count + 1, dtype=np.int32)
    distances = np.linspace(1.0, CUTOFF - 1e-9, count)
    return numbers, radii, receivers, sources, distances


def test_ordinary_d3_counting_law_is_rejected(native):
    """A substituted D3 law moves the values far beyond the FP64 tolerance."""
    numbers, radii, receivers, sources, distances = _swept_graph()
    factors = _unit_factors([6], [[0.0, 1.0, 2.0]], [[1, 1, 1]], 2)
    result = _evaluate(
        native, "float64", numbers, radii, receivers, sources, distances, factors
    )
    covalent_sum = np.full(distances.size, 2 * CARBON_RADIUS)
    toolkit = fourier_d3.coordination_pair_term(distances, covalent_sum, CUTOFF)
    np.testing.assert_allclose(result["edge_values"], toolkit, **TOLERANCE["float64"])
    deviation = np.max(
        np.abs(_ordinary_d3_pair_term(distances, covalent_sum) - toolkit)
    )
    assert deviation > 1e-7
    assert (
        np.max(
            np.abs(
                result["edge_values"] - _ordinary_d3_pair_term(distances, covalent_sum)
            )
        )
        > 1e-7
    )
    assert (
        np.max(
            np.abs(
                result["edge_radial_derivatives"]
                - _paper_eq18_pair_term(distances, covalent_sum, CUTOFF)
            )
        )
        > 1e-7
    )


def test_paper_eq18_steepness_is_rejected_by_the_cutoff_continuity_check(native):
    """The paper's literal steepness leaves a jump the toolkit law does not."""
    numbers, radii, factors = _two_atom_problem()
    just_inside = CUTOFF - 1e-6
    result = _evaluate(
        native,
        "float64",
        numbers,
        radii,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int32),
        np.array([just_inside]),
        factors,
    )
    paper = float(_paper_eq18_pair_term(just_inside, 2 * CARBON_RADIUS, CUTOFF))
    # The toolkit law is indistinguishable from zero this close to the cutoff,
    # while the paper's literal form still carries a visible jump.
    assert result["edge_values"][0] == 0.0
    assert paper > 1e-9
