"""Kokkos particle-mesh forward path and reciprocal contraction (plan Task 9).

Task 9 of the reciprocal-space plan fixes the private forward evaluation: the
fractional wrap, the cardinal B-spline assignment of specification section 5.1,
the continuous-symbol deconvolution of section 5.2, the batched real-to-complex
transform of section 5.4, the Becke-Johnson reciprocal contraction of sections
4.4/5.3 including ``k = 0`` and the Hermitian multiplicity, the self term of
section 4.5, and the unscaled inverse transform that materialises ``P = dE/dm``
for Task 10.

Every expectation here comes from an independent source: the frozen Task 2
fixtures and NumPy oracle, the frozen B-spline symbol fixture with its
hand-derived Eulerian values and truncated-power closed forms, a specification
restatement of the same-mesh quadratic form written directly from section 5.3,
or a central difference of the shipped value function.  Nothing is read back
from the implementation to define its own expectation, and no force, stress, or
atomic-energy claim is made: Task 10 owns that path.

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
from capped_subprocess import require_host_address_space_cap, run_capped_probe
from reference import fourier_d3

FIXTURE_DIRECTORY = Path(__file__).parent / "reference" / "fixtures"
REFERENCE_FIXTURE = json.loads(
    (FIXTURE_DIRECTORY / "fourier_d3_reference_cases.json").read_text(encoding="utf-8")
)
SYMBOL_FIXTURE = json.loads(
    (FIXTURE_DIRECTORY / "bspline_symbol_cases.json").read_text(encoding="utf-8")
)
CASES = {case["name"]: case for case in REFERENCE_FIXTURE["cases"]}
PARAMETERS = REFERENCE_FIXTURE["parameters"]
CUTOFF = float(PARAMETERS["coordination_cutoff_angstrom"])

DAMPING = fourier_d3.DampingParameters.from_bohr(
    s6=PARAMETERS["s6"],
    s8=PARAMETERS["s8"],
    a1=PARAMETERS["a1"],
    a2_bohr=PARAMETERS["a2_bohr"],
)

PRECISIONS = ("float64", "float32")

# Tolerance per precision for a same-pipeline comparison against the frozen
# NumPy reference.  FP64 runs under the historical ``-ffast-math`` policy, so a
# few ulp of FMA contraction against the strict-IEEE oracle are expected; FP32
# additionally rounds every stored mesh, spectrum, and kernel value.
TOLERANCE = {
    "float64": {"rtol": 1e-12, "atol": 1e-14},
    "float32": {"rtol": 5e-5, "atol": 1e-6},
}

# Specification section 5.2: v1 divides by the continuous cardinal B-spline
# symbol.  Restated here so a silent switch to the discrete symbol fails.
ADOPTED_DECONVOLUTION = "continuous"
# Largest per-bin amplitude deficit of that convention, measured below against
# the fixture's independently derived exact attenuation over the convergence
# sequence.  The extreme sits on the outermost stored shell, where the sampled
# symbol still carries a finite value the continuous one has already decayed
# away from; the declaration is the measurement (0.880346545994617), not a hope.
DECLARED_SYMBOL_RESIDUAL = 0.89

# Energy-level agreement with the mesh-free Task 2 oracle, per fixture and mesh.
# These are measurements of the corrected path, stated with a modest margin, and
# they replace the single 1e-3 eV bound that fix round 1 found to be larger than
# a real defect: with the row-fractional coordinate in place the sequence falls
# monotonically instead of plateauing, so a per-mesh table is both tighter and
# more informative than one loose constant.
DECLARED_MESH_ORACLE_RESIDUAL_EV = {
    "triclinic-mixed-species": {
        (8, 8, 8): 5.0e-5,  # measured 3.907e-5
        (12, 12, 12): 1.0e-5,  # measured 6.045e-6
        (16, 16, 16): 1.0e-6,  # measured 3.017e-7
        (20, 20, 20): 2.0e-7,  # measured 7.865e-8
    },
    "single-atom-self-images": {
        (8, 8, 8): 3.0e-5,  # measured 1.841e-5
        (12, 12, 12): 5.0e-6,  # measured 2.456e-6
        (16, 16, 16): 2.0e-6,  # measured 8.295e-7
        (20, 20, 20): 8.0e-7,  # measured 3.385e-7
    },
}

# Same-mesh agreement between the shipped energy and the *explicitly summed*
# structure-factor reference at the strongly triclinic probe mesh.  Measured:
# order 4 2.974e-7 / 3.204e-8 / 3.020e-9 and order 3 8.876e-7 / 5.632e-7 /
# 4.032e-7 at (10, 12, 14) / 16^3 / 20^3, so the declared bound carries a
# portability margin over the worst measured case while staying 266x (about 2.4
# orders of magnitude) below the transposed-coordinate defect (5.312e-4 eV).
DECLARED_EXPLICIT_MESH_RESIDUAL_EV = 2e-6

# |alternative - mesh-free oracle| in eV, measured on the corrected path, per
# fixture and mesh: "continuous" is the adopted v1 convention and "discrete" is
# the sampled-symbol reconstruction ``A * M / D``.  The two fixtures behave
# differently, which is why this is a table and not an inequality.
DECLARED_SYMBOL_ALTERNATIVE_EV = {
    "triclinic-mixed-species": {
        (8, 8, 8): {"continuous": 3.907e-5, "discrete": 5.883e-5},
        (12, 12, 12): {"continuous": 6.045e-6, "discrete": 3.416e-6},
        (16, 16, 16): {"continuous": 3.017e-7, "discrete": 1.261e-6},
        (20, 20, 20): {"continuous": 7.865e-8, "discrete": 5.256e-7},
    },
    "single-atom-self-images": {
        (8, 8, 8): {"continuous": 1.841e-5, "discrete": 3.177e-6},
        (12, 12, 12): {"continuous": 2.456e-6, "discrete": 7.841e-9},
        (16, 16, 16): {"continuous": 8.295e-7, "discrete": 4.048e-10},
        (20, 20, 20): {"continuous": 3.385e-7, "discrete": 7.747e-12},
    },
}

# Mesh sequence used to document interpolation/aliasing convergence.  It is a
# measured sequence, not a convergence claim: the truncated reciprocal tail and
# the algebraically decaying interpolation error are reported separately.
CONVERGENCE_MESHES = ((8, 8, 8), (12, 12, 12), (16, 16, 16), (20, 20, 20))

# The fixture's own convergence band, doubled, is how closely the mesh-free
# lattice sum and the frozen real-space reference must agree before either can
# be used to judge a mesh.
FIXTURE_BAND_FACTOR = 2.0

_SEAM_SYMBOLS = ("_FourierD3ReciprocalForward", "_fourier_d3_reciprocal_kernel")


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
            "report; rebuild the backend to exercise the forward path"
        )
    report = capability()
    if not report["available"]:
        pytest.skip(
            "this build has no FourierD3 FFT provider "
            f"({report['reason'] or 'unreported'}); the forward path is not "
            "compiled and nothing is emulated in its place"
        )
    # A build that reports an available provider must carry the forward seam;
    # nothing stands in for it, so a missing seam is a build defect rather than
    # a skip.
    assert hasattr(module, "_FourierD3ReciprocalForward"), (
        "the build reports an available FourierD3 FFT provider, so the native "
        "forward seam must be present; no host or real-space fallback exists"
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
# inputs: Task 8 coefficients and Task 2 references
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


def _with_eigenvalues(factors: fourier_d3.LowRankFactors, eigs):
    return fourier_d3.LowRankFactors(
        species=factors.species,
        eigs=np.asarray(eigs, dtype=np.float64),
        v_q=factors.v_q,
        cn_ref=factors.cn_ref,
        valid=factors.valid,
        q_table=factors.q_table,
    )


def _chunk_factors(factors: fourier_d3.LowRankFactors, begin: int, width: int):
    """The same factors restricted to a rank window, for chunked references."""
    return fourier_d3.LowRankFactors(
        species=factors.species,
        eigs=np.asarray(factors.eigs[begin : begin + width], dtype=np.float64),
        v_q=np.ascontiguousarray(factors.v_q[:, :, begin : begin + width]),
        cn_ref=factors.cn_ref,
        valid=factors.valid,
        q_table=factors.q_table,
    )


def _fixture_graph(case):
    """Receiver-major directed graph of one frozen fixture case."""
    positions = np.asarray(case["positions_angstrom"], dtype=np.float64)
    cell = np.asarray(case["cell_angstrom"], dtype=np.float64)
    edges, shifts = fourier_d3.build_directed_edges(positions, cell, CUTOFF)
    receivers = edges[:, 0]
    neighbours = edges[:, 1]
    distances = np.linalg.norm(
        positions[neighbours] - positions[receivers] + shifts @ cell, axis=1
    )
    return positions, cell, receivers, neighbours, distances, shifts


def _offsets(numbers, receivers):
    counts = np.bincount(np.asarray(receivers, dtype=np.int64), minlength=numbers.size)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)


def _chunk_inputs(
    native, precision, name, *, chunk=None, rank_begin=0, bound=8, factors=None
):
    """One fixture case with its Task 8 coefficients for a requested rank chunk."""
    case = CASES[name]
    factors = _factors(case) if factors is None else factors
    positions, cell, receivers, neighbours, distances, _ = _fixture_graph(case)
    numbers = np.asarray(case["atomic_numbers"], dtype=np.int32)
    radii = np.asarray(case["covalent_radii_angstrom"], dtype=np.float64)
    order = np.argsort(receivers, kind="stable")
    receivers = receivers[order]
    neighbours = neighbours[order]
    distances = distances[order]
    offsets = _offsets(numbers, receivers)

    rank = factors.rank
    requested = rank if chunk is None else int(chunk)
    workspace = native._FourierD3CoordinationWorkspace(precision, bound)
    workspace.reserve(
        numbers.size, neighbours.size, factors.n_species, factors.n_ref, rank, requested
    )
    workspace.set_graph(
        offsets,
        np.ascontiguousarray(neighbours, dtype=np.int32),
        np.ascontiguousarray(distances, dtype=np.float64),
        numbers,
        radii,
    )
    workspace.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    workspace.run_coordination(CUTOFF)
    workspace.run_coefficients(rank_begin)
    coefficients = np.asarray(workspace.coefficients(), dtype=np.float64).reshape(
        numbers.size, requested
    )
    return {
        "case": case,
        "precision": precision,
        "factors": factors,
        "numbers": numbers,
        "radii": radii,
        "positions": positions,
        "cell": cell,
        "coefficients": coefficients,
        "coordination": np.asarray(workspace.coordination(), dtype=np.float64),
        "chunk": requested,
        "rank": rank,
        "rank_begin": rank_begin,
        "eigenvalues": np.asarray(
            factors.eigs[rank_begin : rank_begin + requested], dtype=np.float64
        ),
        "species": np.asarray(factors.species, dtype=np.int32),
        "q": np.asarray(case["q_table_angstrom"], dtype=np.float64),
    }


def _channels(inputs):
    lookup = {
        int(number): index for index, number in enumerate(np.asarray(inputs["species"]))
    }
    return np.array(
        [lookup.get(int(number), -1) for number in inputs["numbers"]], dtype=np.int64
    )


def _forward(native, precision, bound=8):
    return native._FourierD3ReciprocalForward(precision, bound)


def _load(
    native, inputs, mesh, spline_order, damping=DAMPING, coefficients=None, forward=None
):
    """Loads one fixture case into a forward seam.

    ``native`` is the extension module: the seam is created here unless the
    caller supplies one, which is how the allocation-fault test loads into the
    object whose fault counter it armed.
    """
    nx, ny, nz = (int(value) for value in mesh)
    table = inputs["coefficients"] if coefficients is None else np.asarray(coefficients)
    if forward is None:
        forward = _forward(native, str(inputs.get("precision", "float64")))
    forward.reserve(
        nx,
        ny,
        nz,
        int(np.asarray(inputs["species"]).size),
        int(np.asarray(inputs["numbers"]).size),
        inputs["rank"],
        table.shape[1],
        spline_order,
    )
    forward.set_cell(np.ascontiguousarray(inputs["cell"], dtype=np.float64).reshape(-1))
    forward.set_species(inputs["species"], inputs["q"])
    forward.set_atoms(
        np.ascontiguousarray(inputs["positions"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(inputs["numbers"], dtype=np.int32),
    )
    forward.set_damping(damping.a1, damping.a2_angstrom, damping.s6, damping.s8)
    forward.set_rank(
        inputs["rank_begin"], np.asarray(inputs["eigenvalues"], dtype=np.float64)
    )
    forward.set_coefficients(table)
    return forward


def _single_atom_forward(
    native,
    precision,
    *,
    order,
    mesh,
    cell,
    fractional,
    species=(6,),
    q=(1.643057426520438,),
    bound=4,
):
    """One atom at a fractional coordinate with coefficient one and no graph."""
    mesh = tuple(int(value) for value in mesh)
    cell = np.asarray(cell, dtype=np.float64)
    forward = _forward(native, precision, bound)
    forward.reserve(mesh[0], mesh[1], mesh[2], len(species), 1, 1, 1, order)
    forward.set_cell(cell.reshape(-1))
    forward.set_species(
        np.asarray(species, dtype=np.int32), np.asarray(q, dtype=np.float64)
    )
    forward.set_atoms(
        (np.asarray(fractional, dtype=np.float64)[None, :] @ cell),
        np.asarray([species[0]], dtype=np.int32),
    )
    forward.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    forward.set_rank(0, np.asarray([1.0], dtype=np.float64))
    forward.set_coefficients(np.ones((1, 1), dtype=np.float64))
    return forward


# --------------------------------------------------------------------------
# independent references: mesh bins, same-mesh quadratic form, explicit sums
# --------------------------------------------------------------------------


def _signed_miller(index, extent):
    return int(index) if index <= extent // 2 else int(index) - extent


def _mesh_grid(mesh):
    """Stored half-spectrum bin indices in the documented ``layout_right`` order."""
    nx, ny, nz = (int(value) for value in mesh)
    half = nz // 2 + 1
    index = np.empty((nx * ny * half, 3), dtype=np.int64)
    cursor = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(half):
                index[cursor] = (i, j, k)
                cursor += 1
    return index


def _mesh_bins(cell, mesh):
    """Signed Miller indices, wave vectors, norms, and Hermitian multiplicity."""
    index = _mesh_grid(mesh)
    nx, ny, nz = (int(value) for value in mesh)
    miller = np.stack(
        [
            np.where(index[:, 0] <= nx // 2, index[:, 0], index[:, 0] - nx),
            np.where(index[:, 1] <= ny // 2, index[:, 1], index[:, 1] - ny),
            index[:, 2],
        ],
        axis=1,
    )
    wave = miller @ (2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=np.float64))).T
    norm = np.linalg.norm(wave, axis=1)
    self_conjugate = (miller[:, 2] == 0) | ((nz % 2 == 0) & (2 * miller[:, 2] == nz))
    weights = np.where(self_conjugate, 1.0, 2.0)
    return miller, wave, norm, weights


def _deconvolution_modulus(miller, mesh, order):
    """``prod_a |sinc(m_a / N_a)|^p`` of specification section 5.2."""
    mesh = tuple(int(value) for value in mesh)
    modulus = np.ones(miller.shape[0], dtype=np.float64)
    for axis, extent in enumerate(mesh):
        modulus *= np.abs(np.sinc(miller[:, axis] / extent)) ** order
    return modulus


def _discrete_symbol(miller, mesh, order):
    """``|sum_t M_p(t) exp(2 pi i m t / N)|``: the sampled stencil symbol."""
    from reference import bspline_symbol

    mesh = tuple(int(value) for value in mesh)
    modulus = np.ones(miller.shape[0], dtype=np.float64)
    for axis, extent in enumerate(mesh):
        modulus *= bspline_symbol.discrete_symbol(miller[:, axis], extent, order)
    return modulus


def _pair_kernel_table(q, norm, damping):
    """Per-species-pair kernels evaluated on a set of wave-vector norms (spec 4.4)."""
    q = np.asarray(q, dtype=np.float64)
    table = np.zeros((q.size, q.size, norm.size), dtype=np.float64)
    for a in range(q.size):
        for b in range(q.size):
            q_product = float(q[a] * q[b])
            r0 = damping.damping_radius(q_product)
            table[a, b] = fourier_d3.reciprocal_kernel(norm, r0, q_product, damping)
    return table


def _pair_index(a, b, n_species):
    """Index of the unordered species pair ``(min, max)`` in the kernel table."""
    if a > b:
        a, b = b, a
    return a * n_species - a * (a - 1) // 2 + (b - a)


def _spec_mesh_energy(
    cell,
    mesh,
    q,
    eigenvalues,
    structure,
    damping,
    *,
    weight_rule="spec",
    pair_rule="ordered",
    include_k0=True,
    modulus=None,
):
    """Specification section 5.3 written directly, with switchable negative controls.

    ``structure`` is ``(n_species, chunk, bins)``; ``modulus`` re-divides the
    structure factors, which is how a *second* deconvolution is expressed.
    """
    cell = np.asarray(cell, dtype=np.float64)
    mesh = tuple(int(value) for value in mesh)
    miller, _, norm, hermitian = _mesh_bins(cell, mesh)
    volume = abs(float(np.linalg.det(cell)))
    n_species = structure.shape[0]
    if modulus is not None:
        structure = structure / modulus[None, None, :]
    kernels = _pair_kernel_table(q, norm, damping)
    weights = {
        "spec": hermitian,
        "one": np.ones_like(hermitian),
        "two": np.full_like(hermitian, 2.0),
    }[weight_rule]
    if not include_k0:
        weights = np.where(np.all(miller == 0, axis=1), 0.0, weights)
    multiplicity = np.ones((n_species, n_species), dtype=np.float64)
    if pair_rule == "single":
        multiplicity = np.triu(multiplicity)
    elif pair_rule == "triangular":
        multiplicity = np.triu(np.where(np.eye(n_species) > 0.0, 1.0, 2.0))
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    density = np.einsum(
        "abk,ask,bsk->sk",
        kernels * multiplicity[:, :, None],
        structure,
        np.conj(structure),
    ).real
    return (
        -0.5 * float(np.sum(eigenvalues[:, None] * weights[None, :] * density)) / volume
    )


def _bin_density(
    structure, q, norm, damping, *, modulus=None, weights=None, n_species=None
):
    """Per-rank-slot, per-bin density of the quadratic form (specification 5.3)."""
    if modulus is not None:
        structure = structure / modulus[None, None, :]
    n_species = structure.shape[0] if n_species is None else n_species
    kernels = _pair_kernel_table(q, norm, damping)
    multiplicity = np.ones((n_species, n_species), dtype=np.float64)
    density = np.einsum(
        "abk,ask,bsk->sk",
        kernels * multiplicity[:, :, None],
        structure,
        np.conj(structure),
    ).real
    if weights is not None:
        density = density * weights[None, :]
    return density


def _explicit_structure_factors(
    positions, numbers, species, coefficients, wave_vectors
):
    """``S_X^l(k) = sum_i c_il exp(i k . r_i)`` summed explicitly (specification 4.3)."""
    positions = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(numbers, dtype=np.int64)
    species = np.asarray(species, dtype=np.int64)
    chunk = coefficients.shape[1]
    phase = np.exp(1j * (positions @ np.asarray(wave_vectors, dtype=np.float64).T))
    factors = np.zeros((species.size, chunk, wave_vectors.shape[0]), dtype=complex)
    for channel, number in enumerate(species):
        selected = numbers == number
        if not np.any(selected):
            continue
        for slot in range(chunk):
            factors[channel, slot] = (
                coefficients[selected, slot][:, None] * phase[selected]
            ).sum(axis=0)
    return factors


def _oracle_reciprocal_energy(inputs, k_max=None):
    """Mesh-free direct reciprocal sum of the same coefficients (Task 2 oracle)."""
    factors = _chunk_factors(inputs["factors"], inputs["rank_begin"], inputs["chunk"])
    return fourier_d3.reciprocal_energy(
        np.asarray(inputs["positions"], dtype=np.float64),
        _channels(inputs),
        np.asarray(inputs["coefficients"], dtype=np.float64),
        factors,
        DAMPING,
        np.asarray(inputs["cell"], dtype=np.float64),
        fourier_d3.DEFAULT_K_MAX if k_max is None else k_max,
    )


def _oracle_self_energy(inputs):
    factors = _chunk_factors(inputs["factors"], inputs["rank_begin"], inputs["chunk"])
    return fourier_d3.self_energy(
        np.asarray(inputs["coordination"], dtype=np.float64),
        _channels(inputs),
        factors,
        DAMPING,
    )


def _oracle_total_energy(inputs, k_max=None):
    return _oracle_reciprocal_energy(inputs, k_max) + _oracle_self_energy(inputs)


# --------------------------------------------------------------------------
# cardinal B-spline assignment (specification section 5.1)
# --------------------------------------------------------------------------


def _reference_stencil(order, mesh, fractional):
    """Independent restatement of the section 5.1 stencil: nodes and weights.

    The offset start, the ``p / 2 + theta - offset`` argument, and the periodic
    wrap are re-derived here from the specification, and ``M_p`` comes from the
    reference module's own Cox-de Boor recursion, whose integer values are
    separately pinned against the fixture's hand-derived Eulerian ratios.
    """
    from reference import bspline_symbol

    mesh = tuple(int(value) for value in mesh)
    per_axis = []
    for axis, extent in enumerate(mesh):
        y = extent * float(fractional[axis])
        base = int(np.floor(y))
        theta = y - base
        start = int(np.floor(theta - (order - 2) / 2.0))
        axis_weights = {}
        for index in range(order):
            node = (base + start + index) % extent
            axis_weights[node] = float(
                bspline_symbol.cardinal_bspline(
                    order, np.array([order / 2.0 + theta - (start + index)])
                )[0]
            )
        per_axis.append(axis_weights)
    return per_axis


def _stencil_field(per_axis, mesh, node_index):
    nx, ny, nz = (int(value) for value in mesh)
    i = node_index // (ny * nz)
    rest = node_index % (ny * nz)
    j = rest // nz
    k = rest % nz
    return per_axis[0].get(i, 0.0) * per_axis[1].get(j, 0.0) * per_axis[2].get(k, 0.0)


def _reference_spread(cell, mesh, order, positions, numbers, species, coefficients):
    """Independent particle-to-mesh assignment for every species and lane."""
    points = int(np.prod(mesh))
    fields = np.zeros((len(species), coefficients.shape[1], points), dtype=np.float64)
    fractional = np.asarray(positions, dtype=np.float64) @ np.linalg.inv(
        np.asarray(cell, dtype=np.float64)
    )
    for particle, number in enumerate(np.asarray(numbers, dtype=np.int32)):
        matches = np.flatnonzero(np.asarray(species, dtype=np.int32) == number)
        if matches.size == 0:
            continue
        stencil = _reference_stencil(order, mesh, fractional[particle])
        assignment = np.asarray(
            [_stencil_field(stencil, mesh, node) for node in range(points)],
            dtype=np.float64,
        )
        fields[matches[0]] += coefficients[particle, :, None] * assignment[None, :]
    return fields


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("order", (2, 3, 4, 5, 6))
@pytest.mark.parametrize(
    "fractional, fallback_orders",
    (
        (
            np.asarray(
                [
                    [0.1371, 0.2839, 0.4193],
                    [1.0007, -1.4993, 2.0009],
                    [3.7493, 1.5007, -2.1241],
                    [0.3171, 0.6229, 0.0713],
                ],
                dtype=np.float64,
            ),
            (),
        ),
        (
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, -1.5, 2.0],
                    [3.7493, 1.5007, -2.1241],
                    [0.3171, 0.6229, 0.0713],
                ],
                dtype=np.float64,
            ),
            (2, 3, 4, 5, 6),
        ),
        (
            np.asarray(
                [
                    [0.05, 0.125, 0.25],
                    [1.0007, -1.4993, 2.0009],
                    [3.7493, 1.5007, -2.1241],
                    [0.3171, 0.6229, 0.0713],
                ],
                dtype=np.float64,
            ),
            (3, 5),
        ),
    ),
    ids=("device", "integer-fallback", "half-grid-fallback"),
)
@pytest.mark.parametrize(
    "cell",
    (
        np.diag([8.0, 9.0, 10.0]),
        np.asarray(
            [[6.2, 0.0, 0.0], [-1.7, 7.1, 0.0], [1.3, -2.2, 8.4]],
            dtype=np.float64,
        ),
    ),
    ids=("orthogonal", "triclinic"),
)
def test_device_stencil_matches_host_baseline_for_wrapped_positions(
    native, precision, order, fractional, fallback_orders, cell
):
    """Device preparation preserves the established host stencil convention."""
    mesh = (10, 12, 14)
    inputs = _chunk_inputs(native, precision, "triclinic-mixed-species")
    inputs = dict(inputs)
    inputs["cell"] = np.asarray(cell, dtype=np.float64)
    inputs["positions"] = fractional @ inputs["cell"]
    forward = _load(native, inputs, mesh, order)

    diagnostic = forward.stencil()
    assert diagnostic["host_fallback"] is (order in fallback_orders)
    expected_preparation = (
        "host_boundary_fallback" if order in fallback_orders else "device_non_boundary"
    )
    assert diagnostic["preparation"] == expected_preparation
    host_baseline = forward.stencil(host_baseline=True)
    assert host_baseline["preparation"] == "host_diagnostic_baseline"
    np.testing.assert_array_equal(
        np.asarray(diagnostic["base"]), np.asarray(host_baseline["base"])
    )
    tolerance = TOLERANCE[precision]
    np.testing.assert_allclose(
        np.asarray(diagnostic["weights"], dtype=np.float64),
        np.asarray(host_baseline["weights"], dtype=np.float64),
        **tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(diagnostic["slopes"], dtype=np.float64),
        np.asarray(host_baseline["slopes"], dtype=np.float64),
        **tolerance,
    )


def test_cancellation_heavy_wrapped_triclinic_uses_host_boundary_fallback(native):
    """The boundary guard accounts for cancellation in the fractional dot product."""
    inputs = dict(_chunk_inputs(native, "float64", "triclinic-mixed-species"))
    cell = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1e-8, 0.0], [0.25, -0.5, 1.0]],
        dtype=np.float64,
    )
    fractional = np.asarray(
        [
            [2e-8, 100000.1371, -2.1241],
            [0.1371, 0.2839, 0.4193],
            [1.0007, -1.4993, 2.0009],
            [0.3171, 0.6229, 0.0713],
        ],
        dtype=np.float64,
    )
    inputs["cell"] = cell
    inputs["positions"] = fractional @ cell
    forward = _load(native, inputs, (10, 12, 14), 4)

    diagnostic = forward.stencil()
    assert diagnostic["host_fallback"] is True
    assert diagnostic["preparation"] == "host_boundary_fallback"
    host_baseline = forward.stencil(host_baseline=True)
    np.testing.assert_array_equal(diagnostic["base"], host_baseline["base"])
    np.testing.assert_allclose(
        diagnostic["weights"], host_baseline["weights"], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        diagnostic["slopes"], host_baseline["slopes"], rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize(
    "probe",
    SYMBOL_FIXTURE["probes"],
    ids=lambda probe: f"p{probe['spline_order']}n{probe['mesh_size']}f{probe['fractional_coordinate']}",
)
def test_assignment_matches_the_frozen_one_dimensional_attenuation(native, probe):
    """The shipped spread reproduces the fixture's independently derived attenuation."""
    order = probe["spline_order"]
    size = probe["mesh_size"]
    fraction = probe["fractional_coordinate"]
    assert fraction < 0.5, "the fixture probe must sit in the first half cell"
    cell = np.diag(np.full(3, float(probe["cell_length_angstrom"])))
    forward = _single_atom_forward(
        native,
        "float64",
        order=order,
        mesh=(size, size, size),
        cell=cell,
        fractional=(fraction, 0.0, 0.0),
    )
    forward.assign(0)
    field = np.asarray(forward.field(), dtype=np.float64).reshape(size, size, size)

    # The probe is one-dimensional, so the two transverse axes are factored out
    # with the independently restated stencil rather than assumed to be one.
    per_axis = _reference_stencil(order, (size, size, size), (fraction, 0.0, 0.0))
    transverse = per_axis[1].get(0, 0.0) * per_axis[2].get(0, 0.0)
    assert transverse > 0.0
    nodes = np.arange(size, dtype=np.float64)
    phase = np.exp(-2j * np.pi * np.outer(np.asarray(probe["miller"]), nodes) / size)
    coefficients = np.abs(phase @ (field[:, 0, 0] / transverse))

    exact = np.asarray(probe["exact_attenuation"], dtype=np.float64)
    continuous = np.asarray(probe["continuous_symbol"], dtype=np.float64)
    discrete = np.asarray(probe["discrete_symbol_closed_form"], dtype=np.float64)
    # The fixture's closed form is a truncated-power difference and the shipped
    # value is a Cox-de Boor recursion summed over the mesh, so an off-node probe
    # carries the accumulation error of both; the on-node probes stay exact, and
    # the two alternative symbols below differ from this one by far more.
    np.testing.assert_allclose(coefficients, exact, rtol=1e-10, atol=1e-12)

    # The adopted convention is a fixture-compatibility choice, so its residual
    # against both closed forms is measured rather than assumed away.
    assert np.abs(coefficients - continuous).max() > 0.0
    if fraction == 0.0:
        np.testing.assert_allclose(coefficients, discrete, rtol=1e-13, atol=1e-14)
    else:
        assert np.abs(coefficients - discrete).max() > 1e-9


@pytest.mark.parametrize("order", (2, 3, 4, 5, 6))
def test_assignment_is_a_partition_of_unity_for_every_order(native, order):
    forward = _single_atom_forward(
        native,
        "float64",
        order=order,
        mesh=(8, 8, 8),
        cell=np.diag(np.full(3, 8.0)),
        fractional=(0.317, 0.6229, 0.0713),
    )
    forward.assign(0)
    field = np.asarray(forward.field(), dtype=np.float64)

    assert field.shape == (1, 8 * 8 * 8)
    assert float(np.sum(field)) == pytest.approx(1.0, abs=1e-14)
    assert float(np.min(field)) >= 0.0


@pytest.mark.parametrize("order", (2, 3, 4, 5, 6))
def test_assignment_keeps_every_stencil_point_without_a_threshold(native, order):
    """Every one of the p**3 stencil points is kept, however small its weight.

    The atom sits a hundredth of a thousandth of a cell from a node, where one
    axis weight is about ``1e-7``: any weight threshold would drop it, and the
    gather would then stop being the gradient of the spread.
    """
    mesh = (8, 8, 8)
    fractional = (0.5 + 1e-7, 0.25, 0.75)
    forward = _single_atom_forward(
        native,
        "float64",
        order=order,
        mesh=mesh,
        cell=np.diag(np.full(3, 8.0)),
        fractional=fractional,
    )
    forward.assign(0)
    field = np.asarray(forward.field(), dtype=np.float64).reshape(-1)

    per_axis = _reference_stencil(order, mesh, fractional)
    expected = np.array(
        [_stencil_field(per_axis, mesh, node) for node in range(8 * 8 * 8)],
        dtype=np.float64,
    )
    nonzero = np.nonzero(expected)[0]
    # The two transverse axes sit exactly on a node, where the centered stencil
    # has exact zeros; the near-node x axis is what carries the tiny weights, and
    # it does so for the even orders only.
    if order % 2 == 0:
        assert np.min(expected[nonzero]) < 1e-5
    np.testing.assert_array_equal(np.nonzero(field)[0], nonzero)
    np.testing.assert_allclose(
        field[nonzero], expected[nonzero], rtol=1e-13, atol=1e-16
    )
    assert float(np.min(field[nonzero])) > 0.0


def test_assignment_wraps_negative_and_out_of_cell_positions(native):
    """Fractional wrapping is periodic, and equivalent translations agree stably."""
    mesh = (8, 8, 8)
    cell = np.diag(np.full(3, 8.0))
    wrapped = _single_atom_forward(
        native,
        "float64",
        order=4,
        mesh=mesh,
        cell=cell,
        fractional=(0.75, 0.125, 0.5),
    )
    wrapped.assign(0)
    reference = np.asarray(wrapped.field(), dtype=np.float64)

    for fractional in (
        (-0.25, 0.125, 0.5),
        (0.75, 0.125, -0.5),
        (3.75, -4.875, 0.5),
        (-5.25, 0.125, -7.5),
    ):
        moved = _single_atom_forward(
            native,
            "float64",
            order=4,
            mesh=mesh,
            cell=cell,
            fractional=fractional,
        )
        moved.assign(0)
        np.testing.assert_allclose(
            np.asarray(moved.field(), dtype=np.float64), reference, rtol=0.0, atol=1e-14
        )


@pytest.mark.parametrize("order", (2, 4, 6))
@pytest.mark.parametrize(
    "case", ("orthogonal-mixed-species", "triclinic-mixed-species")
)
def test_wrapped_triclinic_stencil_preserves_field_spectrum_and_energy(
    native, order, case
):
    """Independent lattice images must use the same periodic assignment."""
    inputs = _chunk_inputs(native, "float64", case)
    lattice_shifts = np.resize(
        np.asarray([[1, 0, 0], [0, -2, 1], [3, 1, -1], [-2, 0, 2]], dtype=np.int64),
        np.asarray(inputs["positions"]).shape,
    )
    moved_inputs = dict(inputs)
    moved_inputs["positions"] = np.asarray(inputs["positions"], dtype=np.float64) + (
        lattice_shifts @ np.asarray(inputs["cell"], dtype=np.float64)
    )

    def snapshot(case_inputs, *, host_stencil=False):
        forward = _load(native, case_inputs, (10, 12, 14), order)
        if host_stencil:
            forward.stencil(host_baseline=True)
        fields = []
        spectra = []
        for channel in range(np.asarray(case_inputs["species"]).size):
            forward.assign(channel)
            fields.append(np.asarray(forward.field(), dtype=np.float64).copy())
            forward.transform_forward()
            spectra.append(np.asarray(forward.spectrum()).copy())
        return np.asarray(fields), np.asarray(spectra), forward.run()["energy"]

    field, spectrum, energy = snapshot(inputs)
    host_field, host_spectrum, host_energy = snapshot(inputs, host_stencil=True)
    moved_field, moved_spectrum, moved_energy = snapshot(moved_inputs)
    np.testing.assert_allclose(host_field, field, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(host_spectrum, spectrum, rtol=1e-12, atol=2e-13)
    assert host_energy == pytest.approx(energy, rel=1e-12, abs=2e-14)
    np.testing.assert_allclose(moved_field, field, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(moved_spectrum, spectrum, rtol=1e-12, atol=2e-13)
    assert moved_energy == pytest.approx(energy, rel=1e-12, abs=2e-14)


# --------------------------------------------------------------------------
# FFT normalization (specification section 5.4)
# --------------------------------------------------------------------------


def test_forward_and_inverse_transforms_are_both_unnormalized(native):
    size = 8
    forward = _forward(native, "float64", 4)
    forward.reserve(size, size, size, 1, 2, 1, 1, 4)
    forward.set_cell(np.diag(np.full(3, float(size))).reshape(-1))
    forward.set_species(
        np.asarray([6], dtype=np.int32), np.asarray([1.643057426520438])
    )
    forward.set_atoms(
        np.asarray([[0.0, 0.0, 0.0], [1.5, 2.25, 3.0]]),
        np.asarray([6, 6], dtype=np.int32),
    )
    forward.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    forward.set_rank(0, np.asarray([1.0]))
    forward.set_coefficients(np.zeros((2, 1)))

    rng = np.random.default_rng(20261018)
    field = rng.standard_normal((1, size * size * size))
    forward.set_field(field)
    forward.transform_forward()
    spectrum = np.asarray(forward.spectrum())
    assert spectrum.shape == (1, size, size, size // 2 + 1)
    forward.transform_inverse()
    round_trip = np.asarray(forward.field(), dtype=np.float64)

    points = float(size**3)
    # Unnormalized pair: c2r(r2c(m)) = n * m.  An accidental 1/n pair lands on
    # m, and an orthonormal pair on sqrt(n) * m, so both alternatives fail here.
    np.testing.assert_allclose(round_trip, points * field, rtol=1e-12, atol=1e-9)
    assert not np.allclose(round_trip, field, rtol=1e-6, atol=1e-6)
    assert not np.allclose(round_trip, np.sqrt(points) * field, rtol=1e-6, atol=1e-6)

    # Parseval, with the stored half carrying the section 5.3 multiplicities.
    _, _, _, hermitian = _mesh_bins(
        np.diag(np.full(3, float(size))), (size, size, size)
    )
    spectral = float(np.sum(hermitian * np.abs(spectrum[0].reshape(-1)) ** 2))
    assert spectral / points == pytest.approx(float(np.sum(field**2)), rel=1e-12)


def test_deconvolution_applies_the_continuous_modulus_exactly_once(native):
    order = 4
    size = 8
    cell = np.diag(np.full(3, float(size)))
    forward = _single_atom_forward(
        native,
        "float64",
        order=order,
        mesh=(size, size, size),
        cell=cell,
        fractional=(0.3719, 0.0, 0.0),
    )
    forward.assign(0)
    forward.transform_forward()
    raw = np.asarray(forward.spectrum())
    forward.deconvolve()
    deconvolved = np.asarray(forward.spectrum())

    miller, _, _, _ = _mesh_bins(cell, (size, size, size))
    continuous = _deconvolution_modulus(miller, (size, size, size), order)
    discrete = _discrete_symbol(miller, (size, size, size), order)

    np.testing.assert_allclose(
        deconvolved[0].reshape(-1),
        raw[0].reshape(-1) / continuous,
        rtol=1e-13,
        atol=1e-18,
    )
    assert not np.allclose(
        deconvolved[0].reshape(-1), raw[0].reshape(-1) / discrete, rtol=1e-6
    )
    assert not np.allclose(deconvolved[0].reshape(-1), raw[0].reshape(-1), rtol=1e-6)
    assert forward.geometry()["deconvolution"] == ADOPTED_DECONVOLUTION


def test_deconvolution_residual_is_below_the_declared_tolerance(native):
    """The continuous symbol's amplitude deficit is bounded and its energy converges."""
    name = "single-atom-self-images"
    inputs = _chunk_inputs(native, "float64", name)
    order = 4
    cell = np.asarray(inputs["cell"], dtype=np.float64)
    oracle = _oracle_total_energy(inputs)

    residuals = {}
    errors = {}
    for mesh in CONVERGENCE_MESHES:
        miller, _, _, _ = _mesh_bins(cell, mesh)
        continuous = _deconvolution_modulus(miller, mesh, order)
        discrete = _discrete_symbol(miller, mesh, order)
        # The per-bin amplitude the adopted convention divides out differs from
        # the symbol the assignment actually attenuates by, at worst, this much.
        residuals[mesh] = float(np.abs(1.0 - continuous / discrete).max())
        errors[mesh] = abs(_load(native, inputs, mesh, order).run()["energy"] - oracle)

    assert max(residuals.values()) < DECLARED_SYMBOL_RESIDUAL
    # This fixture puts its atoms on the nodes, where the sampled symbol is
    # exact, so the sequence is a genuine interpolation sequence: it falls by
    # about an order of magnitude per refinement step.
    bounds = DECLARED_MESH_ORACLE_RESIDUAL_EV["single-atom-self-images"]
    assert all(errors[mesh] <= bounds[mesh] for mesh in CONVERGENCE_MESHES)
    assert all(
        errors[later] < errors[earlier]
        for earlier, later in zip(CONVERGENCE_MESHES, CONVERGENCE_MESHES[1:])
    )


# --------------------------------------------------------------------------
# Becke-Johnson reciprocal kernel (specification section 4.4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_device_kernel_matches_the_oracle_closed_form(native, precision):
    q_product = 1.0622434512415695**2
    r0 = float(DAMPING.damping_radius(q_product))
    norms = np.array([0.0, 1e-9, 0.05, 0.0999, 0.1, 0.1001, 0.5, 2.0, 4.3, 9.0, 20.0])
    report = native._fourier_d3_reciprocal_kernel(
        precision, norms, r0, q_product, DAMPING.s6, DAMPING.s8
    )
    values = np.asarray(report["value"], dtype=np.float64)
    expected = fourier_d3.reciprocal_kernel(norms, r0, q_product, DAMPING)
    np.testing.assert_allclose(values, expected, **TOLERANCE[precision])

    # k = 0 is a genuine, non-zero term of the lattice sum (specification 5.5).
    zero_limit = DAMPING.s6 * 2.0 * np.pi**2 / (
        3.0 * r0**3
    ) + 3.0 * DAMPING.s8 * q_product * np.pi**2 / r0**5 * np.sqrt(2.0) * np.sin(
        np.pi / 8.0
    )
    # The limit is evaluated in double precision while the device kernel is
    # asked for one precision only, so FP32 rounds the same value.
    limit_tolerance = 1e-13 if precision == "float64" else 1e-6
    assert values[0] == pytest.approx(zero_limit, rel=limit_tolerance, abs=1e-18)
    # The kernel is an exponentially damped oscillation, so it changes sign.
    assert np.any(values > 0.0) and np.any(values < 0.0)
    # Both sides of the series crossover are present and continuous across it.
    assert abs(values[3] - values[4]) < 1e-3
    assert abs(values[4] - values[5]) < 1e-3


@pytest.mark.parametrize("precision", PRECISIONS)
def test_device_kernel_derivative_matches_central_differences(native, precision):
    q_product = 1.0622434512415695**2
    r0 = float(DAMPING.damping_radius(q_product))
    norms = np.array([1e-6, 0.05, 0.0999, 0.1, 0.1001, 1.0, 4.3, 12.0])
    # FP32 rounds the kernel value itself, so a step below eps * |v| / h only
    # measures that rounding; the wider step keeps the difference meaningful.
    step = 1e-6 if precision == "float64" else 1e-2
    report = native._fourier_d3_reciprocal_kernel(
        precision, norms, r0, q_product, DAMPING.s6, DAMPING.s8
    )
    derivative = np.asarray(report["derivative"], dtype=np.float64)
    plus = np.asarray(
        native._fourier_d3_reciprocal_kernel(
            precision, norms + step, r0, q_product, DAMPING.s6, DAMPING.s8
        )["value"],
        dtype=np.float64,
    )
    minus = np.asarray(
        native._fourier_d3_reciprocal_kernel(
            precision, norms - step, r0, q_product, DAMPING.s6, DAMPING.s8
        )["value"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        derivative,
        fourier_d3.reciprocal_kernel_derivative(norms, r0, q_product, DAMPING),
        **TOLERANCE[precision],
    )
    np.testing.assert_allclose(
        derivative,
        (plus - minus) / (2.0 * step),
        rtol=1e-4 if precision == "float64" else 1e-2,
        atol=1e-8 if precision == "float64" else 1e-3,
    )


def test_reciprocal_kernel_table_matches_the_oracle_on_every_bin(native):
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    mesh = (8, 9, 10)
    forward = _load(native, inputs, mesh, 4)
    forward.run()
    table = np.asarray(forward.pair_kernel(), dtype=np.float64)
    derivative = np.asarray(forward.pair_kernel_derivative(), dtype=np.float64)

    miller, _, norm, _ = _mesh_bins(inputs["cell"], mesh)
    n_species = inputs["species"].size
    expected = _pair_kernel_table(inputs["q"], norm, DAMPING)
    expected_derivative = np.zeros_like(expected)
    r0_table = np.zeros((n_species, n_species), dtype=np.float64)
    for a in range(n_species):
        for b in range(n_species):
            q_product = float(inputs["q"][a] * inputs["q"][b])
            r0 = float(DAMPING.damping_radius(q_product))
            r0_table[a, b] = r0
            expected_derivative[a, b] = fourier_d3.reciprocal_kernel_derivative(
                norm, r0, q_product, DAMPING
            )

    assert table.shape == (n_species * (n_species + 1) // 2, norm.size)
    for a in range(n_species):
        for b in range(a, n_species):
            row = _pair_index(a, b, n_species)
            np.testing.assert_allclose(
                table[row], expected[a, b], rtol=1e-13, atol=1e-15
            )
            np.testing.assert_allclose(
                derivative[row], expected_derivative[a, b], rtol=1e-12, atol=1e-12
            )

    # The compared bins include the k = 0 limit, both sides of the series
    # crossover, and a sign change, so the comparison is not vacuous.
    scales = norm * r0_table[0, 0]
    assert np.any(scales < fourier_d3.SERIES_CUTOFF)
    assert np.any(scales > fourier_d3.SERIES_CUTOFF)
    assert np.any(expected[0, 0] > 0.0) and np.any(expected[0, 0] < 0.0)
    assert np.all(miller[0] == 0) and norm[0] == 0.0
    assert table[0, 0] == pytest.approx(
        fourier_d3.reciprocal_kernel(
            0.0, r0_table[0, 0], float(inputs["q"][0] ** 2), DAMPING
        ),
        rel=1e-14,
    )


def test_kernel_table_uses_the_cell_inverse_convention(native):
    """A transposed or otherwise wrong cell inverse must not reproduce the table."""
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    mesh = (8, 9, 10)
    forward = _load(native, inputs, mesh, 4)
    forward.run()
    table = np.asarray(forward.pair_kernel(), dtype=np.float64)

    _, _, norm, _ = _mesh_bins(inputs["cell"], mesh)
    _, _, wrong_norm, _ = _mesh_bins(
        np.asarray(inputs["cell"], dtype=np.float64).T, mesh
    )
    expected = _pair_kernel_table(inputs["q"], wrong_norm, DAMPING)

    assert np.abs(norm - wrong_norm).max() > 1e-3
    assert np.abs(table[0] - expected[0, 0]).max() > 1e-6


# --------------------------------------------------------------------------
# reciprocal contraction (specification sections 5.3 and 4.2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "name", ("orthogonal-mixed-species", "triclinic-mixed-species")
)
def test_reciprocal_energy_matches_the_same_mesh_quadratic_form(
    native, precision, name
):
    inputs = _chunk_inputs(native, precision, name)
    mesh = (10, 8, 12)
    forward = _load(native, inputs, mesh, 4)
    outcome = forward.run()
    structure = np.asarray(forward.structure_factors())

    assert structure.shape == (
        inputs["species"].size,
        inputs["chunk"],
        int(np.prod([mesh[0], mesh[1], mesh[2] // 2 + 1])),
    )
    expected = _spec_mesh_energy(
        inputs["cell"], mesh, inputs["q"], inputs["eigenvalues"], structure, DAMPING
    )
    scale = max(abs(outcome["reciprocal_energy"]), 1e-12)
    assert abs(outcome["reciprocal_energy"] - expected) <= max(
        TOLERANCE[precision]["rtol"] * scale, TOLERANCE[precision]["atol"]
    )
    # The ordered-pair and upper-triangular forms of the same sum agree exactly.
    assert expected == pytest.approx(
        _spec_mesh_energy(
            inputs["cell"],
            mesh,
            inputs["q"],
            inputs["eigenvalues"],
            structure,
            DAMPING,
            pair_rule="triangular",
        ),
        rel=1e-13,
        abs=1e-18,
    )


def test_reciprocal_energy_uses_the_odd_nyquist_rule(native):
    """With odd N_z there is no Nyquist plane; the top stored bin still stands for two."""
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    mesh = (9, 8, 9)
    forward = _load(native, inputs, mesh, 4)
    outcome = forward.run()
    structure = np.asarray(forward.structure_factors())

    miller, _, norm, weights = _mesh_bins(inputs["cell"], mesh)
    volume = abs(float(np.linalg.det(np.asarray(inputs["cell"], dtype=np.float64))))
    top = np.all(miller == np.array([0, 0, (mesh[2] - 1) // 2]), axis=1)
    assert np.count_nonzero(top) == 1
    assert (
        weights[top][0] == 2.0
    ), "the top stored bin of an odd mesh is not self-conjugate"

    expected = _spec_mesh_energy(
        inputs["cell"], mesh, inputs["q"], inputs["eigenvalues"], structure, DAMPING
    )
    scale = max(abs(outcome["reciprocal_energy"]), 1e-12)
    assert abs(outcome["reciprocal_energy"] - expected) <= 1e-12 * scale

    # Dropping the multiplicity of the top plane (the even-mesh rule reused for
    # an odd mesh) must change the answer.
    density = _bin_density(structure, inputs["q"], norm, DAMPING)
    wrong_weights = np.where(top, 1.0, weights)
    wrong = (
        -0.5
        * float(
            np.sum(inputs["eigenvalues"][:, None] * wrong_weights[None, :] * density)
        )
        / volume
    )
    assert abs(outcome["reciprocal_energy"] - wrong) > 1e-6 * scale


@pytest.mark.parametrize(
    ("control", "kwargs"),
    (
        ("hermitian-weight-one", {"weight_rule": "one"}),
        ("hermitian-weight-two", {"weight_rule": "two"}),
        ("off-diagonal-multiplicity-dropped", {"pair_rule": "single"}),
        ("k0-bin-dropped", {"include_k0": False}),
    ),
)
def test_contraction_negative_controls_are_discriminative(native, control, kwargs):
    """Each control names one convention the shipped contraction must not adopt."""
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    mesh = (10, 8, 12)
    forward = _load(native, inputs, mesh, 4)
    outcome = forward.run()
    structure = np.asarray(forward.structure_factors())

    alternative = _spec_mesh_energy(
        inputs["cell"],
        mesh,
        inputs["q"],
        inputs["eigenvalues"],
        structure,
        DAMPING,
        **kwargs,
    )
    scale = max(abs(outcome["reciprocal_energy"]), 1e-12)
    assert abs(outcome["reciprocal_energy"] - alternative) > 1e-6 * scale, control


def test_contraction_rejects_a_second_deconvolution(native):
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    mesh = (10, 8, 12)
    forward = _load(native, inputs, mesh, 4)
    outcome = forward.run()
    structure = np.asarray(forward.structure_factors())

    miller, _, _, _ = _mesh_bins(inputs["cell"], mesh)
    modulus = _deconvolution_modulus(miller, mesh, 4)
    twice = _spec_mesh_energy(
        inputs["cell"],
        mesh,
        inputs["q"],
        inputs["eigenvalues"],
        structure,
        DAMPING,
        modulus=modulus,
    )
    scale = max(abs(outcome["reciprocal_energy"]), 1e-12)
    assert abs(outcome["reciprocal_energy"] - twice) > 1e-6 * scale


def test_per_bin_subtotals_reduce_to_the_reported_subtotal(native):
    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    mesh = (8, 8, 8)
    forward = _load(native, inputs, mesh, 4)
    outcome = forward.run()
    bins = np.asarray(forward.bin_energies(), dtype=np.float64)
    structure = np.asarray(forward.structure_factors())

    assert bins.shape == (8 * 8 * (8 // 2 + 1),)
    _, _, norm, weights = _mesh_bins(inputs["cell"], mesh)
    density = _bin_density(structure, inputs["q"], norm, DAMPING, weights=weights)
    volume = abs(float(np.linalg.det(np.asarray(inputs["cell"], dtype=np.float64))))
    expected = -0.5 * np.einsum("sb,s->b", density, inputs["eigenvalues"]) / volume
    np.testing.assert_allclose(bins, expected, rtol=1e-13, atol=1e-18)
    assert float(np.sum(bins)) == pytest.approx(
        outcome["reciprocal_energy"], rel=1e-15, abs=1e-18
    )


def test_interpolation_error_shrinks_with_the_spline_order(native):
    """Against explicit structure factors on the same bins, higher order is closer."""
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    mesh = (12, 12, 12)
    _, wave, _, _ = _mesh_bins(inputs["cell"], mesh)
    explicit = _explicit_structure_factors(
        inputs["positions"],
        inputs["numbers"],
        inputs["species"],
        inputs["coefficients"],
        wave,
    )
    reference = _spec_mesh_energy(
        inputs["cell"], mesh, inputs["q"], inputs["eigenvalues"], explicit, DAMPING
    )

    errors = {}
    for order in (2, 3, 4, 5, 6):
        forward = _load(native, inputs, mesh, order)
        errors[order] = abs(forward.run()["reciprocal_energy"] - reference)
    assert errors[4] < errors[2]
    assert errors[5] < errors[3]
    assert errors[6] < errors[3]
    # Measured: 4.678e-5, 1.341e-5, 8.336e-6, 8.535e-6, and 9.624e-6 for orders
    # 2..6.  The trend stops improving after order 4 because the residual is then
    # the sampled-attenuation floor rather than the spline order, so orders 4 to
    # 6 must agree with each other instead of ordering strictly.
    assert max(errors[order] for order in (4, 5, 6)) <= 1.5 * min(
        errors[order] for order in (4, 5, 6)
    )


def test_mesh_convergence_is_quantified_against_the_mesh_free_oracle(native):
    """The mesh sequence is reported against both references, not assumed converged.

    Two independent references are compared first: the mesh-free Task 2 lattice
    sum and the frozen real-space fixture.  This fixture puts its atoms off the
    nodes, so the adopted continuous symbol leaves a residual that does not
    vanish with refinement; the sequence is documented and bounded, and no fixed
    mesh is claimed converged merely because the BJ kernel decays.
    """
    name = "triclinic-mixed-species"
    inputs = _chunk_inputs(native, "float64", name)
    oracle = _oracle_total_energy(inputs)
    fixture = float(CASES[name]["expected"]["energy_ev"])
    band = float(CASES[name]["expected"]["energy_convergence_band_ev"])
    assert abs(oracle - fixture) <= FIXTURE_BAND_FACTOR * band

    errors = {}
    for mesh in CONVERGENCE_MESHES:
        errors[mesh] = abs(_load(native, inputs, mesh, 4).run()["energy"] - oracle)

    # Corrected path: this off-node triclinic fixture converges *monotonically*
    # to the mesh-free oracle (measured 3.907e-5, 6.045e-6, 3.017e-7, 7.865e-8),
    # so each mesh is bounded by its own measured residual and the sequence
    # falls at every refinement step.  A plateau here would mean a convention or
    # coordinate defect, not interpolation.
    bounds = DECLARED_MESH_ORACLE_RESIDUAL_EV[name]
    assert all(errors[mesh] <= bounds[mesh] for mesh in CONVERGENCE_MESHES)
    assert all(
        errors[later] < errors[earlier]
        for earlier, later in zip(CONVERGENCE_MESHES, CONVERGENCE_MESHES[1:])
    )
    assert (
        abs(_load(native, inputs, CONVERGENCE_MESHES[-1], 4).run()["energy"] - fixture)
        <= bounds[CONVERGENCE_MESHES[-1]] + FIXTURE_BAND_FACTOR * band
    )


# --------------------------------------------------------------------------
# self term (specification section 4.5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_self_energy_matches_the_oracle(native, precision, name):
    inputs = _chunk_inputs(native, precision, name)
    forward = _load(native, inputs, (8, 8, 8), 4)
    forward.run()
    reference = _oracle_self_energy(inputs)
    scale = max(abs(reference), 1e-12)
    assert abs(float(forward.self_energy()) - reference) <= max(
        1e-10 * scale, TOLERANCE[precision]["atol"]
    )


def test_self_energy_feeds_the_coefficient_derivative(native):
    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    forward = _load(native, inputs, (10, 10, 10), 4)
    outcome = forward.run()
    self_derivative = np.asarray(
        forward.self_coefficient_derivative(), dtype=np.float64
    )

    assert outcome["energy"] == pytest.approx(
        outcome["reciprocal_energy"] + outcome["self_energy"], rel=0.0, abs=1e-15
    )
    assert self_derivative.shape == (inputs["numbers"].size, inputs["chunk"])
    channels = _channels(inputs)
    expected = np.zeros_like(self_derivative)
    for index, channel in enumerate(channels):
        if channel < 0:
            continue
        q = float(inputs["q"][channel])
        r0 = float(DAMPING.damping_radius(q * q))
        potential = float(fourier_d3.pair_potential(0.0, r0, q * q, DAMPING))
        expected[index] = (
            inputs["coefficients"][index] * inputs["eigenvalues"] * potential
        )
    np.testing.assert_allclose(self_derivative, expected, rtol=1e-12, atol=1e-16)
    # The self term is quadratic in the coefficients, so it is not a constant
    # offset: it moves with the mesh-independent coefficient block.
    assert float(np.max(np.abs(self_derivative))) > 0.0


def test_one_atom_periodic_cell_has_a_nonzero_cell_dependent_energy(native):
    """Only the zero-translation diagonal is cancelled; self-images are retained."""
    energies = {}
    for cell_length in (4.0, 6.0, 8.0):
        inputs = _chunk_inputs(native, "float64", "single-atom-self-images")
        inputs["cell"] = np.diag(np.full(3, cell_length))
        inputs["positions"] = np.zeros((1, 3))
        forward = _load(native, inputs, (20, 20, 20), 4)
        outcome = forward.run()
        reference = _oracle_total_energy(inputs)

        assert outcome["energy"] < 0.0
        assert abs(outcome["energy"]) > 1e-6
        assert (
            abs(outcome["energy"] - reference)
            <= DECLARED_MESH_ORACLE_RESIDUAL_EV["single-atom-self-images"][(20, 20, 20)]
        )
        assert outcome["self_energy"] > 0.0
        energies[cell_length] = outcome["energy"]
    assert abs(energies[4.0] - energies[8.0]) > 1e-4
    assert abs(energies[4.0] - energies[6.0]) > 1e-4


# --------------------------------------------------------------------------
# rank chunking, signs, padding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_rank_chunking_is_numerically_neutral(native, precision):
    name = "coordination-sensitive"
    full = _chunk_inputs(native, precision, name)
    mesh = (10, 8, 12)
    reference = _load(native, full, mesh, 4).run()

    for width in (1, 2, 3, 4):
        total = 0.0
        self_total = 0.0
        reciprocal_total = 0.0
        for begin in range(0, full["rank"], width):
            chunk = min(width, full["rank"] - begin)
            inputs = _chunk_inputs(
                native, precision, name, chunk=chunk, rank_begin=begin
            )
            outcome = _load(native, inputs, mesh, 4).run()
            reciprocal_total += outcome["reciprocal_energy"]
            self_total += outcome["self_energy"]
            total += outcome["energy"]
        # Chunking changes only the summation order, and the spread's atomic
        # scatter is not ordered, so the comparison uses the declared
        # per-precision measurement tolerance rather than an FP64-only bound.
        rtol = TOLERANCE[precision]["rtol"]
        atol = TOLERANCE[precision]["atol"]
        assert reciprocal_total == pytest.approx(
            reference["reciprocal_energy"], rel=rtol, abs=atol
        )
        assert self_total == pytest.approx(reference["self_energy"], rel=rtol, abs=atol)
        assert total == pytest.approx(reference["energy"], rel=rtol, abs=atol)


def test_rank_signs_are_preserved(native):
    base = _chunk_inputs(native, "float64", "orthogonal-mixed-species")
    signed = _with_eigenvalues(
        base["factors"],
        [base["factors"].eigs[0], -base["factors"].eigs[1]]
        + list(base["factors"].eigs[2:]),
    )
    signed_inputs = _chunk_inputs(
        native, "float64", "orthogonal-mixed-species", factors=signed
    )
    mesh = (10, 10, 10)
    forward = _load(native, signed_inputs, mesh, 4)
    outcome = forward.run()
    structure = np.asarray(forward.structure_factors())

    expected = _spec_mesh_energy(
        signed_inputs["cell"],
        mesh,
        signed_inputs["q"],
        signed_inputs["eigenvalues"],
        structure,
        DAMPING,
    )
    scale = max(abs(outcome["reciprocal_energy"]), 1e-12)
    assert abs(outcome["reciprocal_energy"] - expected) <= 1e-12 * scale

    # Dropping the sign reproduces neither the reciprocal subtotal nor the self
    # term: both are signed sums over the rank.
    magnitude = _spec_mesh_energy(
        signed_inputs["cell"],
        mesh,
        signed_inputs["q"],
        np.abs(signed_inputs["eigenvalues"]),
        structure,
        DAMPING,
    )
    assert abs(outcome["reciprocal_energy"] - magnitude) > 1e-6 * scale
    assert _oracle_self_energy(signed_inputs) != _oracle_self_energy(
        dict(signed_inputs, factors=_with_eigenvalues(signed, np.abs(signed.eigs)))
    )


def test_padding_and_uncovered_atoms_carry_no_field_or_energy(native):
    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    mesh = (10, 10, 10)
    reference = _load(native, inputs, mesh, 4).run()["energy"]

    padded = dict(inputs)
    padded["numbers"] = np.concatenate(
        [inputs["numbers"], np.array([0, 99], dtype=np.int32)]
    )
    padded["positions"] = np.vstack(
        [inputs["positions"], [[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]]]
    )
    padded["coefficients"] = np.vstack(
        [inputs["coefficients"], np.zeros((2, inputs["chunk"]), dtype=np.float64)]
    )
    padded["coordination"] = np.concatenate([inputs["coordination"], [0.0, 0.0]])

    outcome = _load(native, padded, mesh, 4).run()
    assert outcome["energy"] == pytest.approx(reference, rel=0.0, abs=1e-16)
    # The padded run hands the reference two extra zero rows, whose summation
    # grouping is what moves here; the terms themselves are unchanged.
    assert _oracle_self_energy(padded) == pytest.approx(
        _oracle_self_energy(inputs), rel=0.0, abs=1e-15
    )

    forward = _load(native, padded, mesh, 4)
    forward.assign(0)
    field = np.asarray(forward.field(), dtype=np.float64)
    trimmed = _load(native, inputs, mesh, 4)
    trimmed.assign(0)
    # The padded atoms carry a zero coefficient, so they add nothing; the
    # scatter is atomic and the atom count changes the summation order, which is
    # the documented non-determinism of a spread.
    np.testing.assert_allclose(
        field, np.asarray(trimmed.field(), dtype=np.float64), rtol=0.0, atol=1e-15
    )


# --------------------------------------------------------------------------
# cotangent: the acceptance test
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_potential_is_the_directional_derivative_of_the_discrete_energy(
    native, precision
):
    """The section 6.2 cotangent, checked against the discrete mesh energy itself."""
    inputs = _chunk_inputs(native, precision, "triclinic-mixed-species")
    mesh = (10, 8, 8)
    forward = _load(native, inputs, mesh, 4)

    fields = []
    for channel in range(inputs["species"].size):
        forward.assign(channel)
        fields.append(np.asarray(forward.field(), dtype=np.float64))
    field = np.stack(fields)

    energy = forward.debug_evaluate_field(field)
    potentials = []
    for channel in range(inputs["species"].size):
        forward.invert(channel)
        potentials.append(np.asarray(forward.potential(channel), dtype=np.float64))
    potential = np.stack(potentials)

    rng = np.random.default_rng(20261018)
    direction = rng.standard_normal(field.shape)
    step = 1e-2 if precision == "float32" else 1e-3
    plus = forward.debug_evaluate_field(field + step * direction)
    minus = forward.debug_evaluate_field(field - step * direction)
    finite_difference = (plus - minus) / (2.0 * step)
    analytic = float(np.sum(direction * potential))

    assert np.isfinite(energy) and analytic != 0.0
    # Hermitian weighting, the FFT normalization, and the single deconvolution
    # all enter this identity; a factor of two or an n scaling breaks it.
    threshold = 1e-2 if precision == "float32" else 1e-7
    assert abs(finite_difference - analytic) <= threshold * max(abs(analytic), 1e-12)


def test_potential_requires_the_matching_inversion(native):
    inputs = _chunk_inputs(native, "float64", "orthogonal-mixed-species")
    forward = _load(native, inputs, (8, 8, 8), 4)
    forward.run()

    with pytest.raises(RuntimeError, match="invert"):
        forward.potential(0)

    forward.invert(1)
    potential = np.asarray(forward.potential(1), dtype=np.float64)
    assert potential.shape == (inputs["chunk"], 8 * 8 * 8)
    assert np.all(np.isfinite(potential))
    with pytest.raises(RuntimeError, match="invert"):
        forward.potential(0)


# --------------------------------------------------------------------------
# caches, accounting, lifecycle, validation
# --------------------------------------------------------------------------


def test_plans_are_reused_and_kernels_rebuild_only_on_their_own_key(native):
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    mesh = (8, 8, 8)
    forward = _load(native, inputs, mesh, 4)
    forward.run()

    usage = forward.usage()
    plans = {entry["role"]: entry for entry in usage["plans"]}
    assert set(plans) == {"forward_r2c", "inverse_c2r"}
    assert plans["forward_r2c"]["builds"] == 1
    assert plans["inverse_c2r"]["builds"] == 1
    assert plans["forward_r2c"]["held"] is True
    key_misses = usage["kernels"]["key_misses"]
    assert key_misses == 1

    # A repeated step reuses both plans and the reciprocal kernel domain.
    forward.run()
    usage = forward.usage()
    plans = {entry["role"]: entry for entry in usage["plans"]}
    assert plans["forward_r2c"]["builds"] == 1
    assert plans["forward_r2c"]["reuses"] >= 1
    assert plans["inverse_c2r"]["reuses"] >= 1
    assert usage["kernels"]["key_misses"] == key_misses

    # A graph-only change (new coefficients) rebuilds nothing.
    forward.set_coefficients(inputs["coefficients"] * 0.5)
    forward.run()
    usage = forward.usage()
    assert usage["kernels"]["key_misses"] == key_misses
    plans = {entry["role"]: entry for entry in usage["plans"]}
    assert plans["forward_r2c"]["builds"] == 1

    # A cell change rebuilds the reciprocal kernels but not the plans.
    strained = np.array(inputs["cell"], dtype=np.float64)
    strained[1] = strained[1] * 1.01
    forward.set_cell(strained.reshape(-1))
    forward.run()
    usage = forward.usage()
    assert usage["kernels"]["key_misses"] == key_misses + 1
    plans = {entry["role"]: entry for entry in usage["plans"]}
    assert plans["forward_r2c"]["builds"] == 1
    assert plans["forward_r2c"]["key_rebuilds"] == 0
    assert plans["inverse_c2r"]["key_rebuilds"] == 0


def test_usage_accounts_every_retained_buffer_once(native):
    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    mesh = (10, 10, 10)
    forward = _load(native, inputs, mesh, 4)
    forward.run()
    usage = forward.usage()

    names = [entry["name"] for entry in usage["buffers"]]
    assert len(names) == len(set(names))
    own_active = sum(entry["active_bytes"] for entry in usage["buffers"])
    assert usage["active_bytes"] == (
        own_active + usage["mesh"]["active_bytes"] + usage["kernels"]["active_bytes"]
    )
    own_capacity = sum(entry["capacity_bytes"] for entry in usage["buffers"])
    assert usage["capacity_bytes"] == (
        own_capacity
        + usage["mesh"]["capacity_bytes"]
        + usage["kernels"]["capacity_bytes"]
    )
    for entry in usage["buffers"]:
        assert 0 < entry["active_elements"] <= entry["capacity_elements"]

    # A plan retains the mesh allocation itself, so those bytes are reported for
    # information and never added to the total a second time.
    assert usage["plan_retained_bytes"] == usage["mesh"]["capacity_bytes"]

    rows = {entry["name"]: entry for entry in usage["buffers"]}
    bins = 10 * 10 * (10 // 2 + 1)
    assert (
        rows["coefficients"]["active_elements"]
        == inputs["numbers"].size * inputs["chunk"]
    )
    assert (
        rows["species_spectra"]["active_elements"]
        == inputs["species"].size * inputs["chunk"] * bins
    )
    assert rows["stencil_weight"]["active_elements"] == inputs["numbers"].size * 3 * 4
    assert rows["bin_energy"]["active_elements"] == bins


def test_reserve_failure_is_atomic_and_leaves_the_state_usable(native):
    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    mesh = (10, 10, 10)
    reference = _load(native, inputs, mesh, 4).run()["energy"]

    for fail_at in range(0, 8):
        forward = _forward(native, "float64", 8)
        native._fourier_d3_arm_allocation_fault(fail_at)
        with pytest.raises(RuntimeError, match="injected allocation fault"):
            _load(native, inputs, mesh, 4, forward=forward)
        native._fourier_d3_clear_allocation_fault()
        assert forward.usage()["reservations"] == 0
        assert forward.usage()["active_bytes"] == 0
        assert forward.usage()["allocations"] == 0
        outcome = _load(native, inputs, mesh, 4, forward=forward).run()
        assert outcome["energy"] == pytest.approx(reference, rel=1e-12, abs=1e-14)


def test_reserve_rejects_impossible_geometry_before_allocating(native):
    forward = _forward(native, "float64", 4)
    with pytest.raises(ValueError, match="spline order"):
        forward.reserve(8, 8, 8, 1, 1, 1, 1, 7)
    with pytest.raises(ValueError, match="at least"):
        forward.reserve(3, 8, 8, 1, 1, 1, 1, 4)
    with pytest.raises(ValueError, match="rank chunk"):
        forward.reserve(8, 8, 8, 1, 1, 4, 5, 4)
    with pytest.raises(ValueError, match="rank"):
        forward.reserve(8, 8, 8, 1, 1, 2, 4, 4)
    with pytest.raises(ValueError, match="species"):
        forward.reserve(8, 8, 8, 0, 1, 1, 1, 4)
    assert forward.usage()["reservations"] == 0
    assert forward.usage()["active_bytes"] == 0
    assert forward.usage()["allocations"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("cell-not-finite", "cell"),
        ("cell-singular", "singular|invertible"),
        ("positions-not-finite", "position"),
        ("numbers-negative", "atomic number"),
        ("species-not-ascending", "ascending"),
        ("q-size", "q"),
        ("damping-not-positive", "a2"),
        ("rank-window", "rank"),
        ("coefficient-shape", "coefficient"),
    ),
)
def test_loaded_inputs_are_validated_before_any_launch(native, mutation, message):
    inputs = _chunk_inputs(native, "float64", "triclinic-mixed-species")
    forward = _forward(native, "float64", 8)
    forward.reserve(
        8,
        8,
        8,
        inputs["species"].size,
        inputs["numbers"].size,
        inputs["rank"],
        inputs["chunk"],
        4,
    )
    cell = np.array(inputs["cell"], dtype=np.float64).reshape(-1)
    positions = np.array(inputs["positions"], dtype=np.float64).reshape(-1)
    numbers = np.array(inputs["numbers"], dtype=np.int32)
    species = np.array(inputs["species"], dtype=np.int32)
    q = np.array(inputs["q"], dtype=np.float64)
    damping = (DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    rank_begin = inputs["rank_begin"]
    eigenvalues = np.array(inputs["eigenvalues"], dtype=np.float64)
    coefficients = np.array(inputs["coefficients"], dtype=np.float64)

    if mutation == "cell-not-finite":
        cell[0] = np.inf
    elif mutation == "cell-singular":
        cell[3:6] = cell[0:3]
    elif mutation == "positions-not-finite":
        positions[0] = np.nan
    elif mutation == "numbers-negative":
        numbers[0] = -1
    elif mutation == "species-not-ascending":
        species = np.ascontiguousarray(species[::-1])
    elif mutation == "q-size":
        q = np.ascontiguousarray(q[:-1])
    elif mutation == "damping-not-positive":
        damping = (damping[0], 0.0, damping[2], damping[3])
    elif mutation == "rank-window":
        rank_begin = inputs["rank"] - inputs["chunk"] + 1
    elif mutation == "coefficient-shape":
        coefficients = np.ascontiguousarray(coefficients[:-1])

    with pytest.raises(ValueError, match=message):
        forward.set_cell(cell)
        forward.set_species(species, q)
        forward.set_atoms(positions, numbers)
        forward.set_damping(*damping)
        forward.set_rank(rank_begin, eigenvalues)
        forward.set_coefficients(coefficients)
        forward.run()


def test_missing_inputs_fail_loudly_instead_of_evaluating(native):
    forward = _forward(native, "float64", 8)
    forward.reserve(8, 8, 8, 1, 1, 1, 1, 4)
    with pytest.raises(RuntimeError, match="cell"):
        forward.run()

    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    partial = _forward(native, "float64", 8)
    partial.reserve(
        8,
        8,
        8,
        inputs["species"].size,
        inputs["numbers"].size,
        inputs["rank"],
        inputs["chunk"],
        4,
    )
    partial.set_cell(np.asarray(inputs["cell"], dtype=np.float64).reshape(-1))
    with pytest.raises(RuntimeError, match="species"):
        partial.run()

    assigned = _load(native, inputs, (8, 8, 8), 4)
    with pytest.raises(RuntimeError, match="stored"):
        assigned.contract()


def test_forward_seam_is_private_while_configuration_is_public(native):
    for symbol in _SEAM_SYMBOLS:
        assert hasattr(native, symbol), symbol
        # The package forwards unknown names to the backend module, so the
        # package export list is what says "private".
        assert symbol not in symmetrix.__all__, symbol
    assert "FourierD3Correction" in symmetrix.__all__
    # Task 11 adds the native evaluator class, and the package's fallback forwards any
    # name the native module provides, so reachability no longer says "private".
    # Publication does: the curated export list and __all__ are unchanged.
    assert "FourierD3Kokkos" not in symmetrix.__all__
    assert "FourierD3Kokkos" not in symmetrix._FRONTEND_EXPORTS


def test_geometry_reports_the_frozen_conventions(native):
    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    forward = _load(native, inputs, (8, 8, 8), 4)
    geometry = forward.geometry()

    assert geometry["deconvolution"] == ADOPTED_DECONVOLUTION
    assert geometry["forward_normalization"] == "unnormalized"
    assert geometry["inverse_normalization"] == "unnormalized"
    assert geometry["hermitian_axis"] == "z"
    assert (
        geometry["species_pairs"]
        == inputs["species"].size * (inputs["species"].size + 1) // 2
    )
    assert geometry["spectrum_bins"] == 8 * 8 * (8 // 2 + 1)
    assert geometry["mesh_points"] == 512
    assert geometry["stencil_points"] == 4**3
    assert geometry["active_rank_chunk"] == inputs["chunk"]
    assert geometry["rank_begin"] == inputs["rank_begin"]
    assert geometry["num_nodes"] == inputs["numbers"].size
    assert geometry["num_species"] == inputs["species"].size


def test_forward_objects_release_the_mesh_before_finalization(native):
    inputs = _chunk_inputs(native, "float64", "coordination-sensitive")
    forward = _load(native, inputs, (8, 8, 8), 4)
    forward.run()
    # A live object that has run pins its mesh through the plans it built, and
    # the accounting reports that retention exactly once.
    usage = forward.usage()
    assert usage["plan_retained_bytes"] == usage["mesh"]["capacity_bytes"] > 0
    del forward
    gc.collect()
    # A second object must be constructible, which fails if the first one left a
    # live Kokkos owner behind.
    again = _forward(native, "float64", 8)
    assert again.usage()["allocations"] == 0
    assert again.usage()["reservations"] == 0
    # Nothing is retained before a plan exists, so an unused object pins no mesh.
    assert again.usage()["plan_retained_bytes"] == 0
    assert again.usage()["capacity_bytes"] == 0


# --------------------------------------------------------------------------
# fix round 1: the row-fractional convention and assignment invalidation
# --------------------------------------------------------------------------
#
# The specification fixes the row convention: cell rows are the lattice
# vectors, a position is a Cartesian row vector, and the fractional coordinate
# is ``s = r C^-1`` (specification sections 4.1 and 5.1).  The wave vector of a
# stored bin is ``k = 2 pi C^-1 m``.  Those two are *different* index orders of
# the same inverse, so a transposed fractional coordinate is not a small
# perturbation: it moves every off-node atom on a skewed cell, and on an
# orthogonal cell it is invisible.  These tests therefore use a strongly
# triclinic cell and an independent explicit sum, never the shipped structure
# factors, as the reference.

# Cell angles of 51, 105, and 85 degrees: no axis pair is orthogonal, so the
# transposed convention cannot agree with the row convention by accident.
SKEW_CELL = np.asarray(
    [[5.0, 0.0, 0.0], [3.3, 4.1, 0.0], [-1.7, 2.2, 6.3]], dtype=np.float64
)
SKEW_FRACTIONAL = np.asarray(
    [
        [0.13, 0.27, 0.41],
        [0.62, 0.11, 0.83],
        [0.35, 0.72, 0.19],
        [0.88, 0.44, 0.57],
    ],
    dtype=np.float64,
)
SKEW_NUMBERS = np.asarray([6, 6, 8, 8], dtype=np.int32)
SKEW_SPECIES = np.asarray([6, 8], dtype=np.int32)
SKEW_Q = np.asarray([1.643057426520438, 1.372482991504054], dtype=np.float64)
SKEW_COEFFICIENTS = np.asarray(
    [[1.0, -0.5], [0.75, 0.25], [-1.25, 0.6], [0.4, -0.9]], dtype=np.float64
)
SKEW_MESH = (10, 12, 14)

# Low-frequency bins, where the sampled stencil attenuation is real and within
# a fraction of a percent of one for every atom, so the phase of the stored
# amplitude is the phase of the specification's structure factor.  The
# attenuation is not a single per-bin scalar away from the origin (each atom
# carries its own offset), which is why the comparison is restricted here
# rather than asserted on every bin.
SKEW_PHASE_BAND = 0.1
# Measured on the corrected build: the largest phase mismatch of the row
# candidate over the band (3.863e-3 at order 3, 1.689e-4 at order 4), and the
# smallest mismatch the transposed candidate can reach there (9.141e-2, median
# 0.631).  The row candidate is 24x below the transposed floor and 163x below
# its median.
SKEW_PHASE_TOLERANCE = 6e-3
SKEW_TRANSPOSED_FLOOR = 0.05
SKEW_TRANSPOSED_MEDIAN = 0.2


def _skew_positions(cell=SKEW_CELL, fractional=SKEW_FRACTIONAL):
    return np.asarray(fractional, dtype=np.float64) @ np.asarray(cell, dtype=np.float64)


def _stored_structure_factors(forward, channels):
    """Deconvolved structure factors for every species channel of a configured seam."""
    for channel in range(channels):
        forward.assign(channel)
        forward.transform_forward()
        forward.deconvolve()
        forward.store_spectrum(channel)
    return np.asarray(forward.structure_factors())


@pytest.mark.parametrize("species_count", (1, 2))
@pytest.mark.parametrize("rank_chunk", (1, 3))
def test_same_mesh_field_spectrum_structure_factor_energy_and_usage(
    native, species_count, rank_chunk
):
    """The forward transport matches an independent NumPy construction."""
    mesh = SKEW_MESH
    order = 4
    species = SKEW_SPECIES[:species_count]
    q = SKEW_Q[:species_count]
    numbers = (
        np.full(SKEW_NUMBERS.shape, species[0], dtype=np.int32)
        if species_count == 1
        else SKEW_NUMBERS
    )
    coefficients = np.asarray(
        [
            [1.0, -0.5, 0.2],
            [0.75, 0.25, -0.3],
            [-1.25, 0.6, 0.9],
            [0.4, -0.9, -0.7],
        ],
        dtype=np.float64,
    )[:, :rank_chunk]
    eigenvalues = np.asarray([1.0, -0.35, 0.8], dtype=np.float64)[:rank_chunk]
    positions = _skew_positions()

    forward = _forward(native, "float64", 3)
    forward.reserve(
        *mesh,
        species.size,
        numbers.size,
        rank_chunk,
        rank_chunk,
        order,
    )
    forward.set_cell(SKEW_CELL.reshape(-1))
    forward.set_species(species, q)
    forward.set_atoms(positions.reshape(-1), numbers)
    forward.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    forward.set_rank(0, eigenvalues)
    forward.set_coefficients(coefficients)

    expected_fields = _reference_spread(
        SKEW_CELL, mesh, order, positions, numbers, species, coefficients
    )
    miller, _, _, _ = _mesh_bins(SKEW_CELL, mesh)
    deconvolution = _deconvolution_modulus(miller, mesh, order)
    expected_structure = np.empty(
        (species.size, rank_chunk, miller.shape[0]), dtype=np.complex128
    )
    for channel in range(species.size):
        forward.assign(channel)
        np.testing.assert_allclose(
            np.asarray(forward.field(), dtype=np.float64),
            expected_fields[channel],
            rtol=1e-13,
            atol=2e-15,
        )
        forward.transform_forward()
        raw = np.asarray(
            [np.fft.rfftn(lane.reshape(mesh)) for lane in expected_fields[channel]]
        )
        np.testing.assert_allclose(
            np.asarray(forward.spectrum()), raw, rtol=1e-12, atol=2e-13
        )
        forward.deconvolve()
        expected_structure[channel] = raw.reshape(rank_chunk, -1) / deconvolution
        np.testing.assert_allclose(
            np.asarray(forward.spectrum()).reshape(rank_chunk, -1),
            expected_structure[channel],
            rtol=1e-12,
            atol=2e-13,
        )
        forward.store_spectrum(channel)

    np.testing.assert_allclose(
        np.asarray(forward.structure_factors()),
        expected_structure,
        rtol=1e-12,
        atol=2e-13,
    )
    forward.contract()
    expected_energy = _spec_mesh_energy(
        SKEW_CELL, mesh, q, eigenvalues, expected_structure, DAMPING
    )
    assert forward.reciprocal_energy() == pytest.approx(
        expected_energy, rel=1e-12, abs=1e-14
    )

    first_usage = forward.usage()
    points = int(np.prod(mesh))
    bins = mesh[0] * mesh[1] * (mesh[2] // 2 + 1)
    mesh_buffers = {row["name"]: row for row in first_usage["mesh"]["buffers"]}
    assert mesh_buffers["real_mesh"]["active_elements"] == rank_chunk * points
    assert mesh_buffers["spectrum"]["active_elements"] == rank_chunk * bins
    assert first_usage["reservations"] == 1
    assert {plan["role"]: plan["builds"] for plan in first_usage["plans"]} == {
        "forward_r2c": 1,
        "inverse_c2r": 0,
    }

    forward.run()
    second_usage = forward.usage()
    assert second_usage["allocations"] == first_usage["allocations"]
    assert second_usage["reservations"] == first_usage["reservations"]
    assert {plan["role"]: plan["builds"] for plan in second_usage["plans"]} == {
        "forward_r2c": 1,
        "inverse_c2r": 1,
    }


def _skew_forward(
    native,
    precision,
    mesh=SKEW_MESH,
    order=4,
    cell=SKEW_CELL,
    coefficients=SKEW_COEFFICIENTS,
    numbers=SKEW_NUMBERS,
    positions=None,
):
    """A strongly triclinic forward with explicit coefficients and no graph.

    The Cartesian positions are fixed by default, so changing ``cell`` alone is a
    change of the fractional coordinate ``s = r C^-1`` and nothing else: that is
    what makes a cell-update-in-place comparable with a fresh object.
    """
    cell = np.asarray(cell, dtype=np.float64)
    positions = _skew_positions() if positions is None else np.asarray(positions)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    forward = _forward(native, precision, 8)
    forward.reserve(
        int(mesh[0]),
        int(mesh[1]),
        int(mesh[2]),
        SKEW_SPECIES.size,
        int(np.asarray(numbers).size),
        coefficients.shape[1],
        coefficients.shape[1],
        order,
    )
    forward.set_cell(cell.reshape(-1))
    forward.set_species(SKEW_SPECIES, SKEW_Q)
    forward.set_atoms(positions, np.asarray(numbers, dtype=np.int32))
    forward.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    forward.set_rank(0, np.ones(coefficients.shape[1], dtype=np.float64))
    forward.set_coefficients(coefficients)
    return forward


def _wave_vectors(cell, mesh, *, transposed=False):
    """Wave vectors of the stored bins, in the specification's row convention.

    ``k = 2 pi C^-1 m`` is a *different* index order of the same inverse from the
    fractional coordinate ``s = r C^-1``; ``transposed=True`` builds the wave
    vectors the transposed fractional coordinate would imply, so a test can
    show that the shipped spread does not use them.
    """
    miller, _, _, _ = _mesh_bins(cell, mesh)
    inverse = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    if transposed:
        return miller @ (2.0 * np.pi * inverse)
    return miller @ (2.0 * np.pi * inverse).T


def _phase_mismatch(spectrum, expected, significant):
    """``|sin(delta phi)|`` per bin: one where the two are orthogonal in phase."""
    return np.abs(
        np.imag(spectrum * np.conj(expected)) / (np.abs(spectrum) * np.abs(expected))
    )[significant]


def _explicit_mesh_energy(
    cell, mesh, positions, numbers, species, coefficients, q, eigenvalues, damping
):
    """Specification 5.3 with *explicitly summed* structure factors.

    The structure factors come from ``positions`` and the specification's phases
    alone, so this reference shares no output with the shipped mesh path.
    """
    structure = _explicit_structure_factors(
        positions, numbers, species, coefficients, _wave_vectors(cell, mesh)
    )
    return _spec_mesh_energy(cell, mesh, q, eigenvalues, structure, damping)


@pytest.mark.parametrize("order", (3, 4))
def test_deconvolved_structure_factor_phase_follows_the_row_fractional_coordinate(
    native, order
):
    """The spread's phase is ``exp(-i k . r)`` with ``s = r C^-1``, not its transpose."""
    forward = _skew_forward(native, "float64", order=order)
    for channel in (0, 1):
        forward.assign(channel)
        forward.transform_forward()
        forward.deconvolve()
        forward.store_spectrum(channel)
    stored = np.asarray(forward.structure_factors())[0, 0]

    miller, _, _, _ = _mesh_bins(SKEW_CELL, SKEW_MESH)
    positions = _skew_positions()
    band = np.all(
        np.abs(miller) <= SKEW_PHASE_BAND * np.asarray(SKEW_MESH, dtype=np.int64),
        axis=1,
    )
    assert np.count_nonzero(band) >= 8, "the phase band must hold more than the origin"

    # The stored half-spectrum is the forward real-to-complex transform, whose
    # phase is exp(-i k . r); the explicit sum below is written with exp(+i k . r).
    expected = np.conj(
        _explicit_structure_factors(
            positions,
            SKEW_NUMBERS,
            SKEW_SPECIES,
            SKEW_COEFFICIENTS,
            _wave_vectors(SKEW_CELL, SKEW_MESH),
        )
    )[0, 0]
    row = _phase_mismatch(stored, expected, band)
    assert row.max() <= SKEW_PHASE_TOLERANCE, (
        "the stored amplitude must carry the row-fractional phase exp(-i k . r) "
        f"with s = r C^-1; the largest mismatch in the band is {row.max():.3e}"
    )

    # The magnitude is the sampled attenuation, which is within a fraction of a
    # percent of one inside the band, so the phase comparison above is not a
    # comparison of noise.
    ratio = (np.abs(stored) / np.abs(expected))[band & np.any(miller != 0, axis=1)]
    assert np.abs(ratio - 1.0).max() <= 1e-2

    # The transposed fractional coordinate C^-1 r is the defect this test
    # exists for: it must not reproduce the stored phase.
    wrong = np.conj(
        _explicit_structure_factors(
            positions,
            SKEW_NUMBERS,
            SKEW_SPECIES,
            SKEW_COEFFICIENTS,
            _wave_vectors(SKEW_CELL, SKEW_MESH, transposed=True),
        )
    )[0, 0]
    away_from_origin = band & np.any(miller != 0, axis=1)
    rejected = _phase_mismatch(stored, wrong, away_from_origin)
    assert rejected.min() >= SKEW_TRANSPOSED_FLOOR, (
        "the transposed fractional coordinate C^-1 r must not reproduce the stored "
        f"phase; its smallest mismatch in the band is {rejected.min():.3e}"
    )
    # Measured: the transposed candidate's median mismatch in the band is 0.633
    # at order 3 and 0.631 at order 4, against at most 3.9e-3 for the row
    # candidate, so the rejection is not a single unlucky bin.
    assert float(np.median(rejected)) >= SKEW_TRANSPOSED_MEDIAN


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("order", (3, 4))
def test_same_mesh_energy_matches_the_explicit_structure_factor_reference(
    native, precision, order
):
    """The shipped mesh energy against an independent explicit sum, not its own output."""
    forward = _skew_forward(native, precision, order=order)
    outcome = forward.run()
    expected = _explicit_mesh_energy(
        SKEW_CELL,
        SKEW_MESH,
        _skew_positions(),
        SKEW_NUMBERS,
        SKEW_SPECIES,
        SKEW_COEFFICIENTS,
        SKEW_Q,
        np.ones(SKEW_COEFFICIENTS.shape[1], dtype=np.float64),
        DAMPING,
    )
    scale = max(abs(expected), 1e-12)
    assert abs(outcome["reciprocal_energy"] - expected) <= max(
        TOLERANCE[precision]["rtol"] * scale, DECLARED_EXPLICIT_MESH_RESIDUAL_EV
    ), (
        "the same-mesh reciprocal energy must match the explicit structure-factor "
        f"sum: shipped {outcome['reciprocal_energy']!r}, explicit {expected!r}"
    )


def test_cell_change_rebuilds_the_assignment_and_matches_a_fresh_object(native):
    """``set_cell`` must re-derive the assignment, not keep the old one."""
    second = np.asarray(
        [[6.1, 0.0, 0.0], [-2.4, 5.2, 0.0], [1.9, -3.1, 7.4]], dtype=np.float64
    )
    forward = _skew_forward(native, "float64")
    forward.assign(0)
    forward.transform_forward()
    forward.deconvolve()
    first_field = np.asarray(forward.field(), dtype=np.float64).copy()
    # Complex, so both the real and the imaginary part are compared: an earlier
    # version read this through ``dtype=np.float64`` and silently discarded the
    # imaginary part with a ComplexWarning.
    first_spectrum = np.asarray(forward.spectrum()).copy()

    forward.set_cell(second.reshape(-1))
    forward.assign(0)
    forward.transform_forward()
    forward.deconvolve()
    updated_field = np.asarray(forward.field(), dtype=np.float64)
    updated_spectrum = np.asarray(forward.spectrum())

    fresh = _skew_forward(native, "float64", cell=second)
    fresh.assign(0)
    fresh.transform_forward()
    fresh.deconvolve()
    fresh_field = np.asarray(fresh.field(), dtype=np.float64)
    fresh_spectrum = np.asarray(fresh.spectrum())

    # A stale assignment is invisible in the cache counters and obvious here.
    assert not np.allclose(updated_field, first_field, rtol=1e-6, atol=1e-9), (
        "a changed cell must change the assignment; the field is identical to the "
        "previous cell's"
    )
    assert not np.allclose(
        updated_spectrum, first_spectrum, rtol=1e-6, atol=1e-9
    ), "a changed cell must change the transformed assignment as well"
    np.testing.assert_allclose(updated_field, fresh_field, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(updated_spectrum, fresh_spectrum, rtol=1e-12, atol=1e-14)

    # The deconvolved structure factors are compared directly as well, not only
    # through the energy they feed: measured bitwise equal here
    # (``max|difference| == 0.0``, shape ``(2, 2, 960)``), asserted at the
    # declared complex tolerance rather than bitwise so the comparison survives
    # a different FFT plan or backend.
    channels = len(SKEW_SPECIES)
    updated_factors = _stored_structure_factors(forward, channels).copy()
    fresh_factors = _stored_structure_factors(fresh, channels)
    np.testing.assert_allclose(updated_factors, fresh_factors, rtol=1e-12, atol=1e-14)

    # The energy follows the field: an updated object and a fresh object agree.
    assert forward.run()["energy"] == pytest.approx(fresh.run()["energy"], rel=1e-12)


def test_repeated_cell_changes_reuse_the_plans_and_rebuild_the_assignment(native):
    """Changing the cell rebuilds the assignment and the grid, never the plans."""
    first = SKEW_CELL
    second = np.asarray(
        [[6.1, 0.0, 0.0], [-2.4, 5.2, 0.0], [1.9, -3.1, 7.4]], dtype=np.float64
    )
    forward = _skew_forward(native, "float64")
    reference = {
        "first": forward.run()["energy"],
    }
    usage = forward.usage()
    assert {entry["role"]: entry["builds"] for entry in usage["plans"]} == {
        "forward_r2c": 1,
        "inverse_c2r": 1,
    }

    for _ in range(2):
        forward.set_cell(second.reshape(-1))
        second_energy = forward.run()["energy"]
        forward.set_cell(np.asarray(first, dtype=np.float64).reshape(-1))
        assert forward.run()["energy"] == pytest.approx(
            reference["first"], rel=1e-12
        ), "a cell change must be reversible in place"
        usage = forward.usage()
        assert {entry["role"]: entry["builds"] for entry in usage["plans"]} == {
            "forward_r2c": 1,
            "inverse_c2r": 1,
        }, "a cell change must not rebuild the FFT plans"
        assert all(entry["key_rebuilds"] == 0 for entry in usage["plans"])

    fresh = _skew_forward(native, "float64", cell=second)
    assert second_energy == pytest.approx(fresh.run()["energy"], rel=1e-12)


def test_moved_atoms_rebuild_the_assignment_and_match_a_fresh_object(native):
    """A second ``set_atoms`` must rebuild the stencil for the new positions.

    The cell is unchanged here, so only the positions generation can invalidate
    the stencil: without it the object keeps spreading the old coordinates.
    """
    moved = _skew_positions() + np.asarray([0.37, -0.21, 0.53])
    forward = _skew_forward(native, "float64")
    forward.assign(0)
    forward.transform_forward()
    forward.deconvolve()
    first_field = np.asarray(forward.field(), dtype=np.float64).copy()
    # Complex, so both parts are compared (see the cell-change test).
    first_spectrum = np.asarray(forward.spectrum()).copy()

    forward.set_atoms(moved, SKEW_NUMBERS)
    forward.assign(0)
    forward.transform_forward()
    forward.deconvolve()
    updated_field = np.asarray(forward.field(), dtype=np.float64)
    updated_spectrum = np.asarray(forward.spectrum())

    fresh = _skew_forward(native, "float64", positions=moved)
    fresh.assign(0)
    fresh.transform_forward()
    fresh.deconvolve()

    assert not np.allclose(updated_field, first_field, rtol=1e-6, atol=1e-9), (
        "new positions must change the assignment; the field is identical to the "
        "previous coordinates'"
    )
    assert not np.allclose(updated_spectrum, first_spectrum, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        updated_field, np.asarray(fresh.field(), dtype=np.float64), rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        updated_spectrum,
        np.asarray(fresh.spectrum()),
        rtol=1e-12,
        atol=1e-14,
    )

    # The deconvolved structure factors are compared directly as well: this is
    # the quantity the energy is contracted from, and it isolates the assignment
    # from the kernels and the contraction.  Measured bitwise equal here
    # (``max|difference| == 0.0``), asserted at the declared complex tolerance.
    channels = len(SKEW_SPECIES)
    updated_factors = _stored_structure_factors(forward, channels).copy()
    fresh_factors = _stored_structure_factors(fresh, channels)
    np.testing.assert_allclose(updated_factors, fresh_factors, rtol=1e-12, atol=1e-14)

    assert forward.run()["energy"] == pytest.approx(fresh.run()["energy"], rel=1e-12)


def test_atoms_before_a_cell_defer_the_stencil_instead_of_poisoning_it(native):
    """``set_atoms`` before ``set_cell`` must defer, never build from a zero inverse."""
    mesh = SKEW_MESH
    deferred = _forward(native, "float64", 8)
    deferred.reserve(
        int(mesh[0]),
        int(mesh[1]),
        int(mesh[2]),
        SKEW_SPECIES.size,
        SKEW_NUMBERS.size,
        SKEW_COEFFICIENTS.shape[1],
        SKEW_COEFFICIENTS.shape[1],
        4,
    )
    deferred.set_species(SKEW_SPECIES, SKEW_Q)
    # No cell yet: the positions are accepted, and no stencil can be derived.
    deferred.set_atoms(_skew_positions().reshape(-1), SKEW_NUMBERS)
    deferred.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    deferred.set_rank(0, np.ones(SKEW_COEFFICIENTS.shape[1], dtype=np.float64))
    deferred.set_coefficients(SKEW_COEFFICIENTS)

    with pytest.raises(RuntimeError, match="cell"):
        deferred.assign(0)

    deferred.set_cell(SKEW_CELL.reshape(-1))
    deferred.assign(0)
    deferred.transform_forward()
    deferred.deconvolve()
    deferred_field = np.asarray(deferred.field(), dtype=np.float64)

    normal = _skew_forward(native, "float64")
    normal.assign(0)
    normal.transform_forward()
    normal.deconvolve()
    np.testing.assert_allclose(
        deferred_field,
        np.asarray(normal.field(), dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert deferred.run()["energy"] == pytest.approx(normal.run()["energy"], rel=1e-12)


def test_reserve_rejects_a_stencil_size_that_overflows_before_allocating(native):
    """``3 * nodes`` and ``3 * nodes * order`` are checked before they are formed."""
    forward = _forward(native, "float64", 1)
    with pytest.raises(OverflowError):
        forward.reserve(8, 8, 8, 1, 2**62, 1, 1, 4)
    assert forward.usage()["reservations"] == 0
    assert forward.usage()["allocations"] == 0
    # The object is still usable after the rejected request.
    forward.reserve(8, 8, 8, 1, 1, 1, 1, 4)
    assert forward.usage()["reservations"] == 1


# A count above `2**31 - 1` is unaddressable *and* enormous - the smallest
# request here is a 6.7 GiB allocation - so a pre-fix implementation asks the
# allocator for it instead of refusing it, which is an OOM kill on a machine that
# has the memory.  The probe therefore runs under `capped_subprocess`, whose 6 GiB
# cap is below every allocation the unguarded path would attempt and whose child
# pins every thread-count variable to one so the cap cannot be spent on per-thread
# OpenBLAS buffers before the code under test runs.
#
# name, reserve arguments, the buffer the refusal must name
FORWARD_FLAT_INDEX_REQUESTS = (
    # `3 * nodes` stays inside the range and only `3 * nodes * order` leaves it,
    # so this isolates the stencil product from the position product.
    ("forward_stencil_weights", (8, 8, 8, 1, 300000000, 1, 1, 6), "stencil_weight"),
    # A 1291^3 mesh is the first count to leave the range: its real buffer is
    # 2.15e9 elements while the spectrum and every forward view stay below the
    # limit, so this isolates the mesh from the views this owner also allocates.
    ("mesh_real_buffer", (1291, 1291, 1291, 1, 4, 1, 1, 4), "real_mesh"),
)

_FORWARD_PROBE_ADDRESS_SPACE_BYTES = 6 * 1024**3


def _forward_flat_index_probe_body(requests, *, then_reserve):
    """A forward-seam probe: attempt each refused request, optionally recover."""
    return (
        "import test_fourier_d3_reciprocal as T\n"
        f"requests = {list(requests)!r}\n"
        "native = T._native()\n"
        "forward = T._forward(native, 'float64', 8)\n"
        "mark_initialized()\n"
        "report = {}\n"
        "for name, arguments in requests:\n"
        "    try:\n"
        "        forward.reserve(*arguments)\n"
        "    except BaseException as error:\n"
        "        report[name] = [type(error).__name__, str(error)]\n"
        "    else:\n"
        "        report[name] = ['<no exception>', '']\n"
        "before = forward.usage()\n"
        "report['refused'] = [int(before['reservations']), int(before['allocations'])]\n"
        f"if {bool(then_reserve)}:\n"
        "    forward.reserve(8, 8, 8, 1, 1, 1, 1, 4)\n"
        "report['accepted'] = int(forward.usage()['reservations'])\n"
        "emit(report)\n"
    )


@pytest.mark.parametrize("name,arguments,buffer_name", FORWARD_FLAT_INDEX_REQUESTS)
def test_an_unaddressable_forward_request_is_refused_before_allocating(
    name, arguments, buffer_name
):
    """The forward path applies the same flat-index bound to its own buffers."""
    require_host_address_space_cap()
    report = run_capped_probe(
        _forward_flat_index_probe_body(((name, arguments),), then_reserve=False),
        address_space_bytes=_FORWARD_PROBE_ADDRESS_SPACE_BYTES,
        label=f"the {name} flat-index probe",
    )
    exception_name, message = report[name]
    assert exception_name == "OverflowError", (
        f"the {name} request {arguments} must be refused with OverflowError before "
        f"anything is allocated; the probe reported {exception_name}: {message}"
    )
    assert (
        buffer_name in message
    ), f"the {name} refusal must name the buffer it refused ({buffer_name}); got: {message}"
    assert (
        "flat Kokkos index range" in message
    ), f"the {name} refusal must name the limit it enforced; got: {message}"


def test_a_refused_forward_request_leaves_the_seam_usable():
    """The bound is a refusal, not a poisoned object.

    This runs in the capped subprocess for the same reason the refusal test does:
    the request is an unaddressable 8 GiB allocation, so a regression must not be
    allowed to OOM-kill the session that is running the suite.
    """
    require_host_address_space_cap()
    report = run_capped_probe(
        _forward_flat_index_probe_body(
            (("mesh_real_buffer", (1291, 1291, 1291, 1, 4, 1, 1, 4)),),
            then_reserve=True,
        ),
        address_space_bytes=_FORWARD_PROBE_ADDRESS_SPACE_BYTES,
        label="the forward recovery probe",
    )
    assert (
        report["mesh_real_buffer"][0] == "OverflowError"
    ), f"the unaddressable mesh request must be refused with OverflowError; got {report}"
    assert report["refused"] == [
        0,
        0,
    ], f"a refused reservation must not count or allocate; got {report}"
    assert (
        report["accepted"] == 1
    ), f"the seam must accept a valid request after a refusal; got {report}"


def _symbol_alternatives(native, inputs, mesh, order=4):
    """Shipped energy and both symbol reconstructions of the *same* stored amplitude.

    The stored structure factor is ``A = raw / M`` (the continuous symbol the v1
    convention divides out).  Reconstructing under the *sampled* symbol ``D``
    therefore multiplies by ``M / D`` — ``A * M / D``, the discrete
    deconvolution — and passing ``D / M`` to a helper that divides is what
    expresses it.  The inverted ratio ``M / D`` would instead attenuate the
    amplitude a second time by ``D / M``, which is a third convention and not the
    discrete one; the assertions below separate the two directions.
    """
    forward = _load(native, inputs, mesh, order)
    outcome = forward.run()
    stored = np.asarray(forward.structure_factors())
    miller, _, _, _ = _mesh_bins(inputs["cell"], mesh)
    continuous = _deconvolution_modulus(miller, mesh, order)
    discrete = _discrete_symbol(miller, mesh, order)
    energies = {"continuous": outcome["energy"]}
    for label, ratio in (
        ("discrete", discrete / continuous),
        ("inverted", continuous / discrete),
    ):
        energies[label] = (
            _spec_mesh_energy(
                inputs["cell"],
                mesh,
                inputs["q"],
                inputs["eigenvalues"],
                stored,
                DAMPING,
                modulus=ratio,
            )
            + outcome["self_energy"]
        )
    return energies


@pytest.mark.parametrize("name", sorted(DECLARED_SYMBOL_ALTERNATIVE_EV))
def test_symbol_convention_is_quantified_against_its_alternative(native, name):
    """The adopted symbol is quantified against the discrete alternative, per mesh.

    The brief pins the continuous symbol for v1 NVIDIA fixture compatibility; it
    is not a claim that the convention is universally the more accurate one.  The
    two fixtures show opposite behaviour and the off-node winner moves with the
    mesh, so the measured table — not a uniform inequality — is the assertion.
    """
    inputs = _chunk_inputs(native, "float64", name)
    oracle = _oracle_total_energy(inputs)
    table = DECLARED_SYMBOL_ALTERNATIVE_EV[name]
    errors = {}
    for mesh in CONVERGENCE_MESHES:
        energies = _symbol_alternatives(native, inputs, mesh)
        errors[mesh] = {key: abs(value - oracle) for key, value in energies.items()}
        # Each candidate stays inside its measured band (1.5x the measurement),
        # so a regression in either direction is caught rather than absorbed.
        for key in ("continuous", "discrete"):
            assert errors[mesh][key] <= 1.5 * table[mesh][key], (
                f"{name} at {mesh}: {key} reconstruction error "
                f"{errors[mesh][key]:.3e} exceeds its measured band "
                f"{1.5 * table[mesh][key]:.3e}"
            )

    if name == "single-atom-self-images":
        # On the nodes the sampled symbol reproduces the amplitude essentially
        # exactly, so the discrete reconstruction is far closer at every mesh and
        # the ratio direction is unambiguous: the inverted ratio is beaten by the
        # discrete reconstruction everywhere and by the adopted convention from
        # 12^3 on (at the coarsest mesh the inverted one happens to be marginally
        # closer to the oracle than the adopted one, which is a coincidence of
        # that single mesh and is recorded here rather than smoothed over).
        for mesh in CONVERGENCE_MESHES:
            assert errors[mesh]["discrete"] < errors[mesh]["continuous"]
            assert errors[mesh]["discrete"] < errors[mesh]["inverted"]
        for mesh in CONVERGENCE_MESHES[1:]:
            assert errors[mesh]["inverted"] > errors[mesh]["continuous"]
        assert errors[CONVERGENCE_MESHES[-1]]["discrete"] <= 1e-9
        assert (
            errors[(12, 12, 12)]["discrete"] < 1e-2 * errors[(12, 12, 12)]["inverted"]
        )
    else:
        # Off the nodes the inverted ratio is worse than the adopted convention
        # at every mesh, which is what separates it from the discrete one.
        for mesh in CONVERGENCE_MESHES:
            assert errors[mesh]["inverted"] > errors[mesh]["continuous"], (
                f"{name} at {mesh}: the inverted ratio must not look like a better "
                f"convention than the adopted one (inverted "
                f"{errors[mesh]['inverted']:.3e}, continuous "
                f"{errors[mesh]['continuous']:.3e})"
            )
        # Off the nodes the winner moves with the mesh: the discrete
        # reconstruction is closer at 12^3 and the adopted continuous symbol is
        # closer at the coarsest and at the two finest meshes tested.
        assert errors[(8, 8, 8)]["continuous"] < errors[(8, 8, 8)]["discrete"]
        assert errors[(12, 12, 12)]["discrete"] < errors[(12, 12, 12)]["continuous"]
        for mesh in ((16, 16, 16), (20, 20, 20)):
            assert errors[mesh]["continuous"] < errors[mesh]["discrete"]


def test_plan_retained_bytes_describe_the_allocation_the_plan_owns(native):
    """A retained plan reports its own allocation, not the storage's current one."""

    inputs = _chunk_inputs(native, "float64", "orthogonal-mixed-species")
    forward = _load(native, inputs, (8, 8, 8), 4)
    forward.run()

    usage = forward.usage()
    plans = {entry["role"]: entry for entry in usage["plans"]}
    owned = plans["forward_r2c"]["allocation"]["retained_bytes"]
    assert owned > 0
    assert usage["plan_retained_bytes"] == owned
    assert plans["inverse_c2r"]["allocation"]["retained_bytes"] == owned

    # Growing the mesh replaces the storage's allocation while both plans still
    # retain the allocation they captured.  The reported plan bytes must follow the
    # plans - the allocation they actually own - not the new storage capacity.
    forward.reserve(
        16,
        16,
        16,
        int(np.asarray(inputs["species"]).size),
        int(np.asarray(inputs["numbers"]).size),
        inputs["rank"],
        inputs["chunk"],
        4,
    )
    grown = forward.usage()
    assert grown["mesh"]["capacity_bytes"] > owned
    assert grown["plan_retained_bytes"] == owned

    # Re-running for the new mesh moves both plans onto the new allocation, so the
    # retained bytes follow them rather than the allocation they left behind.
    forward = _load(native, inputs, (16, 16, 16), 4)
    forward.run()
    replaced = forward.usage()
    assert replaced["plan_retained_bytes"] == replaced["mesh"]["capacity_bytes"]
    assert replaced["plan_retained_bytes"] > owned
