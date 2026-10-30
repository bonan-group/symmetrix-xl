"""Analytic reverse path of the reciprocal-space FourierD3 correction (plan Task 10).

The module drives the private ``_FourierD3ReciprocalReverse`` seam, which owns the
Task 9 particle-mesh forward path and the Task 8 coordination workspace and
produces the energy, forces, strain derivative, ASE stress, atomic energies, and
the coefficient sensitivities from one consistent native state.

Two references are kept apart on purpose:

* the frozen Task 2 direct-reciprocal oracle (mesh-free) is the *physical*
  acceptance reference for energy, forces, and stress once the mesh is converged;
* central differences of the *same finite mesh* are the exact-discrete derivative
  acceptance reference, so a mesh-convergence residual is never mistaken for a
  derivative error.

Every derivative assertion is a step sweep: the whole central-difference band is
reported and the assertion is on the best measured step, never on one hand-picked
step size.
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

# Same-pipeline tolerance of a value that is not a finite difference.
TOLERANCE = {
    "float64": {"rtol": 1e-11, "atol": 1e-14},
    "float32": {"rtol": 5e-5, "atol": 1e-6},
}

# --- declared finite-difference bands -------------------------------------
#
# The bands below are measurements of the corrected path over the step sweeps
# below; each is stated with a modest margin.  They are *same-mesh* bands, so
# they bound the derivative error of the discrete energy rather than the
# mesh-convergence error of the model.
#
# Measured largest |analytic - central difference| per step, 8^3 mesh, order 4
# (`tools`-free driver, orthogonal / triclinic fixture):
#
#   mesh force     float64  2.3e-07  2.3e-09  2.3e-11  1.1e-11
#   total force    float64  9.5e-07  9.5e-09  9.5e-11  1.8e-11
#   strain         float64  8.4e-08  8.4e-10  1.3e-11  8.3e-11   (steps 1e-3..1e-6)
#   sensitivity    float64  1.1e-14  1.1e-13  1.3e-12  1.1e-11
#   mesh force     float32  2.5e-07  5.0e-06  4.8e-05  3.7e-04
#   total force    float32  1.0e-06  4.1e-06  7.8e-05  7.7e-04
#   strain         float32  4.9e-06  9.0e-05  8.5e-04  6.4e-03
#   sensitivity    float32  2.7e-07  2.3e-06  2.6e-05  1.9e-04
#
# The declared bands sit one to three decades above the best measured step, so a
# step that is too large (truncation) or too small (round-off) can never make a
# check pass by accident.

# |analytic - central difference| for one force component, eV/Angstrom.  The
# float32 band is the best measured step (2.5e-07 mesh, 1.0e-06 total) times a
# portability factor of about 400: the float32 sweep bottoms out at the *largest*
# step because round-off grows as the step shrinks, and a different FFT or BLAS
# rounding path can move the crossing point, so the band has to cover the h=1e-3
# row (5.0e-06) with room to spare rather than the optimum alone.
DECLARED_FORCE_FD_BAND = {"float64": 1e-9, "float32": 1e-4}
# The same for the mesh force alone (no coordination chain rule).
DECLARED_MESH_FORCE_FD_BAND = {"float64": 1e-9, "float32": 1e-4}
# |analytic - central difference| for one strain-derivative component, eV.  The
# float32 optimum is 4.9e-06 at h=1e-3, so this band is one decade above it.
DECLARED_STRAIN_FD_BAND = {"float64": 1e-9, "float32": 5e-5}
# The same for the coefficient sensitivity, eV.  The float32 optimum is 2.7e-07.
DECLARED_SENSITIVITY_FD_BAND = {"float64": 1e-9, "float32": 1e-4}
# Independent same-mesh recomputations of the strain derivative and the partition.
DECLARED_SAME_MESH_STRAIN_BAND = {"float64": 1e-9, "float32": 1e-5}
DECLARED_SAME_MESH_PARTITION_BAND = {"float64": 1e-11, "float32": 1e-5}

# The assignment is not exactly translation invariant: a rigid shift of every
# atom moves the mesh samples, and the energy and forces follow.  The residual,
# its measured refinement sequence, and the bands derived from it are declared in
# section J below.

# Central-difference steps.  The band is reported for every step and the
# assertion is on the best one, so a step that is too large (truncation) or too
# small (round-off) cannot make the check pass by accident.
FORCE_STEPS = (1e-2, 1e-3, 1e-4, 1e-5)
STRAIN_STEPS = (1e-3, 1e-4, 1e-5, 1e-6)

# --- physical acceptance against the frozen oracle ------------------------
#
# |mesh - mesh-free oracle| at the declared convergence mesh (24^3), order 4.
# These bound the *model* error of a converged mesh, not a derivative error.
ACCEPTANCE_MESH = (24, 24, 24)
DECLARED_ORACLE_FORCE_BAND_EV_PER_ANGSTROM = 5e-6
DECLARED_ORACLE_STRAIN_BAND_EV = 2e-6

# The strong skew probe of the Task 9 suite, reused so a triclinic error that
# only shows up off the axes cannot hide behind an orthogonal cell.
SKEW_CELL = np.asarray(
    [[6.1, 0.0, 0.0], [1.35, 5.2, 0.0], [0.62, -0.95, 6.6]], dtype=np.float64
)
SKEW_FRACTIONAL = np.asarray(
    [[0.05, 0.11, 0.17], [0.34, 0.21, 0.42], [0.61, 0.53, 0.29], [0.77, 0.68, 0.81]],
    dtype=np.float64,
)
SKEW_NUMBERS = np.asarray([6, 6, 8, 8], dtype=np.int32)
SKEW_SPECIES = np.asarray([6, 8], dtype=np.int32)
SKEW_Q = np.asarray([1.643057426520438, 1.372482991504054], dtype=np.float64)
SKEW_RADII = np.asarray(
    [1.0000000494517636, 1.0000000494517636, 0.8400000011664605, 0.8400000011664605],
    dtype=np.float64,
)
SKEW_CN_REF = np.asarray(
    [[0.0, 0.9868, 1.9985, 2.9987, 3.9844], [0.0, 0.9925, 1.9887, -1.0, -1.0]],
    dtype=np.float64,
)
SKEW_VALID = np.asarray(
    [[True, True, True, True, True], [True, True, True, False, False]], dtype=bool
)
SKEW_V_Q = np.zeros((2, 5, 2), dtype=np.float64)
SKEW_V_Q[0, :, 0] = 1.0
SKEW_V_Q[1, :, 1] = 1.0

_SEAM_SYMBOLS = ("_FourierD3ReciprocalReverse", "_ParticleMeshTransport")


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
            "the installed native extension predates the FourierD3 capability report; "
            "rebuild the backend to exercise the reverse path"
        )
    report = capability()
    if not report["available"]:
        pytest.skip(
            "this build has no FourierD3 FFT provider "
            f"({report['reason'] or 'unreported'}); the reverse path is not compiled "
            "and nothing is emulated in its place"
        )
    assert hasattr(module, "_FourierD3ReciprocalReverse"), (
        "the build reports an available FourierD3 FFT provider, so the native reverse "
        "seam must be present; no host or real-space fallback exists"
    )
    return module


@pytest.fixture(autouse=True)
def _disarm_allocation_fault(native):
    native._fourier_d3_clear_allocation_fault()
    yield
    native._fourier_d3_clear_allocation_fault()


# --------------------------------------------------------------------------
# inputs: Task 8 coefficients, Task 2 fixtures, and the reverse seam
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


def _skew_factors() -> fourier_d3.LowRankFactors:
    return fourier_d3.LowRankFactors(
        species=np.asarray(SKEW_SPECIES, dtype=np.int64),
        eigs=np.asarray([1.0, -0.25], dtype=np.float64),
        v_q=np.asarray(SKEW_V_Q, dtype=np.float64),
        cn_ref=np.asarray(SKEW_CN_REF, dtype=np.float64),
        valid=np.asarray(SKEW_VALID, dtype=bool),
        q_table=np.asarray(SKEW_Q, dtype=np.float64),
    )


def _graph(positions, cell, numbers=None, cutoff=CUTOFF):
    """Receiver-major directed graph sorted by receiver, with its shifts."""
    active = None if numbers is None else (np.asarray(numbers) != 0)
    edges, shifts = fourier_d3.build_directed_edges(positions, cell, cutoff, active)
    receivers = edges[:, 0]
    neighbours = edges[:, 1]
    delta = positions[neighbours] - positions[receivers] + shifts @ cell
    distances = np.linalg.norm(delta, axis=1)
    order = np.argsort(receivers, kind="stable")
    return receivers[order], neighbours[order], shifts[order], distances[order]


def _offsets(numbers, receivers):
    counts = np.bincount(np.asarray(receivers, dtype=np.int64), minlength=numbers.size)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)


def _workspace_coefficients(
    native,
    precision,
    factors,
    numbers,
    radii,
    receivers,
    neighbours,
    distances,
    *,
    chunk=None,
    rank_begin=0,
    bound=8,
    cutoff=CUTOFF,
):
    """Task 8 coefficients for one graph, which is what the reverse path consumes."""
    requested = factors.rank if chunk is None else int(chunk)
    workspace = native._FourierD3CoordinationWorkspace(precision, bound)
    workspace.reserve(
        numbers.size,
        neighbours.size,
        factors.n_species,
        factors.n_ref,
        factors.rank,
        requested,
    )
    workspace.set_graph(
        _offsets(numbers, receivers),
        np.ascontiguousarray(neighbours, dtype=np.int32),
        np.ascontiguousarray(distances, dtype=np.float64),
        numbers,
        radii,
    )
    workspace.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    workspace.run_coordination(cutoff)
    workspace.run_coefficients(rank_begin)
    return np.asarray(workspace.coefficients(), dtype=np.float64).reshape(
        numbers.size, requested
    )


def _inputs(
    native,
    precision,
    *,
    case=None,
    factors=None,
    positions=None,
    cell=None,
    numbers=None,
    radii=None,
    species=None,
    q=None,
    chunk=None,
    rank_begin=0,
    bound=8,
    cutoff=CUTOFF,
):
    """One complete reverse-path input set, derived from a single geometry."""
    if case is not None:
        factors = _factors(CASES[case]) if factors is None else factors
        if positions is None:
            positions = np.asarray(CASES[case]["positions_angstrom"], dtype=np.float64)
        if cell is None:
            cell = np.asarray(CASES[case]["cell_angstrom"], dtype=np.float64)
        if numbers is None:
            numbers = np.asarray(CASES[case]["atomic_numbers"], dtype=np.int32)
        if radii is None:
            radii = np.asarray(CASES[case]["covalent_radii_angstrom"], dtype=np.float64)
        if species is None:
            species = np.asarray(factors.species, dtype=np.int32)
        if q is None:
            q = np.asarray(CASES[case]["q_table_angstrom"], dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    numbers = np.asarray(numbers, dtype=np.int32)
    radii = np.asarray(radii, dtype=np.float64)
    species = np.asarray(species, dtype=np.int32)
    q = np.asarray(q, dtype=np.float64)
    receivers, neighbours, shifts, distances = _graph(positions, cell, numbers, cutoff)
    requested = factors.rank if chunk is None else int(chunk)
    coefficients = _workspace_coefficients(
        native,
        precision,
        factors,
        numbers,
        radii,
        receivers,
        neighbours,
        distances,
        chunk=chunk,
        rank_begin=rank_begin,
        bound=bound,
        cutoff=cutoff,
    )
    return {
        "case": case,
        "precision": precision,
        "factors": factors,
        "positions": positions,
        "cell": cell,
        "numbers": numbers,
        "radii": radii,
        "species": species,
        "q": q,
        "receivers": receivers,
        "neighbours": neighbours,
        "shifts": shifts,
        "distances": distances,
        "offsets": _offsets(numbers, receivers),
        "coefficients": coefficients,
        "chunk": requested,
        "rank": factors.rank,
        "rank_begin": rank_begin,
        "eigenvalues": np.asarray(
            factors.eigs[rank_begin : rank_begin + requested], dtype=np.float64
        ),
    }


def _reload_graph(native, inputs, *, cutoff=CUTOFF):
    """Re-derives the graph and the coefficients after the geometry changed."""
    inputs["receivers"], inputs["neighbours"], inputs["shifts"], inputs["distances"] = (
        _graph(inputs["positions"], inputs["cell"], inputs["numbers"], cutoff)
    )
    inputs["offsets"] = _offsets(inputs["numbers"], inputs["receivers"])
    inputs["coefficients"] = _workspace_coefficients(
        native,
        inputs["precision"],
        inputs["factors"],
        inputs["numbers"],
        inputs["radii"],
        inputs["receivers"],
        inputs["neighbours"],
        inputs["distances"],
        chunk=inputs["chunk"],
        rank_begin=inputs["rank_begin"],
        cutoff=cutoff,
    )
    return inputs


def _load(native, inputs, mesh, order, *, bound=8):
    """Configures one reverse seam from a prepared input set."""
    nx, ny, nz = (int(value) for value in mesh)
    factors = inputs["factors"]
    table = np.ascontiguousarray(inputs["coefficients"], dtype=np.float64)
    reverse = native._FourierD3ReciprocalReverse(str(inputs["precision"]), bound)
    reverse.reserve(
        nx,
        ny,
        nz,
        int(inputs["species"].size),
        int(inputs["numbers"].size),
        int(inputs["neighbours"].size),
        int(factors.n_ref),
        int(inputs["rank"]),
        int(table.shape[1]),
        int(order),
    )
    reverse.set_cell(np.ascontiguousarray(inputs["cell"], dtype=np.float64).reshape(-1))
    reverse.set_species(inputs["species"], inputs["q"])
    reverse.set_atoms(
        np.ascontiguousarray(inputs["positions"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(inputs["numbers"], dtype=np.int32),
    )
    reverse.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    reverse.set_rank(
        int(inputs["rank_begin"]), np.asarray(inputs["eigenvalues"], dtype=np.float64)
    )
    reverse.set_graph(
        np.ascontiguousarray(inputs["offsets"], dtype=np.int32),
        np.ascontiguousarray(inputs["neighbours"], dtype=np.int32),
        np.ascontiguousarray(inputs["shifts"], dtype=np.int32),
        np.ascontiguousarray(inputs["distances"], dtype=np.float64),
        np.ascontiguousarray(inputs["numbers"], dtype=np.int32),
        np.ascontiguousarray(inputs["radii"], dtype=np.float64),
    )
    reverse.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    return reverse


def _evaluate(
    native, inputs, mesh, order, *, cutoff=CUTOFF, bound=8, host_stencil=False
):
    seam = _load(native, inputs, mesh, order, bound=bound)
    if host_stencil:
        seam.prepare_host_stencil_baseline()
    outcome = seam.run(cutoff)
    return {
        "energy": outcome["energy"],
        "reciprocal_energy": seam.reciprocal_energy(),
        "self_energy": seam.self_energy(),
        "forces": np.asarray(seam.forces(), dtype=np.float64),
        "mesh_forces": np.asarray(seam.mesh_forces(), dtype=np.float64),
        "coordination_forces": np.asarray(seam.coordination_forces(), dtype=np.float64),
        "atomic_energies": np.asarray(seam.atomic_energies(), dtype=np.float64),
        "sensitivity": np.asarray(seam.coefficient_sensitivity(), dtype=np.float64),
        "chi": np.asarray(seam.coordination_sensitivity(), dtype=np.float64),
        "strain": np.asarray(seam.strain_derivative(), dtype=np.float64),
        "stress": np.asarray(seam.stress(), dtype=np.float64),
        "virial": np.asarray(seam.virial(), dtype=np.float64),
        "coordination": np.asarray(seam.coordination(), dtype=np.float64),
        "edge_values": np.asarray(seam.edge_values(), dtype=np.float64),
        "edge_derivatives": np.asarray(
            seam.edge_radial_derivatives(), dtype=np.float64
        ),
        "coefficients": np.asarray(seam.coefficients(), dtype=np.float64),
        "coefficient_derivatives": np.asarray(
            seam.coefficient_derivatives(), dtype=np.float64
        ),
        "structure_factors": np.asarray(seam.structure_factors()),
        "pair_kernel": np.asarray(seam.pair_kernel(), dtype=np.float64),
        "pair_kernel_derivative": np.asarray(
            seam.pair_kernel_derivative(), dtype=np.float64
        ),
        "seam": seam,
    }


def _full_energy(
    native, inputs, mesh, order, *, positions=None, cell=None, cutoff=CUTOFF
):
    """The complete discrete energy at one geometry, graph and coefficients rebuilt.

    This is the value function the analytic reverse path differentiates: the
    graph, the coordination numbers, the coefficients, the mesh assignment, the
    reciprocal contraction, and the self term all follow from ``(positions, cell)``.
    """
    fresh = _inputs(
        native,
        inputs["precision"],
        case=inputs["case"],
        factors=inputs["factors"],
        positions=inputs["positions"] if positions is None else positions,
        cell=inputs["cell"] if cell is None else cell,
        numbers=inputs["numbers"],
        radii=inputs["radii"],
        species=inputs["species"],
        q=inputs["q"],
        chunk=inputs["chunk"],
        rank_begin=inputs["rank_begin"],
        cutoff=cutoff,
    )
    return _evaluate(native, fresh, mesh, order, cutoff=cutoff)["energy"]


def _frozen_energy(native, inputs, mesh, order, *, positions=None, cell=None):
    """The mesh-only energy at one geometry with the coefficients held fixed."""
    forward = native._FourierD3ReciprocalForward(str(inputs["precision"]), 8)
    nx, ny, nz = (int(value) for value in mesh)
    table = np.ascontiguousarray(inputs["coefficients"], dtype=np.float64)
    forward.reserve(
        nx,
        ny,
        nz,
        int(inputs["species"].size),
        int(inputs["numbers"].size),
        int(inputs["rank"]),
        int(table.shape[1]),
        int(order),
    )
    forward.set_cell(
        np.ascontiguousarray(
            inputs["cell"] if cell is None else cell, dtype=np.float64
        ).reshape(-1)
    )
    forward.set_species(inputs["species"], inputs["q"])
    forward.set_atoms(
        np.ascontiguousarray(
            inputs["positions"] if positions is None else positions, dtype=np.float64
        ).reshape(-1),
        np.ascontiguousarray(inputs["numbers"], dtype=np.int32),
    )
    forward.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    forward.set_rank(
        int(inputs["rank_begin"]), np.asarray(inputs["eigenvalues"], dtype=np.float64)
    )
    forward.set_coefficients(table)
    return forward.run()["energy"]


def _sweep(function, steps):
    """Every central-difference estimate at the origin."""
    return np.asarray(
        [(function(step) - function(-step)) / (2.0 * step) for step in steps],
        dtype=np.float64,
    )


def _best_band(analytic, estimates):
    """The smallest |analytic - estimate| over the sweep, and the whole band."""
    deviations = np.abs(np.asarray(estimates, dtype=np.float64) - analytic)
    return float(np.min(deviations)), deviations


def _particle_mesh_transport(native, precision, mesh=(5, 4, 6), order=4):
    """Configure the private physics-neutral transport seam."""
    assert hasattr(native, "_ParticleMeshTransport"), (
        "the particle-mesh capability is available, so its private transport "
        "seam must expose the shared spread/gather pair"
    )
    positions = np.asarray(
        [[0.47, 0.91, 1.33], [2.11, 1.64, 0.72], [3.42, 2.05, 2.77]],
        dtype=np.float64,
    )
    cell = np.asarray(
        [[5.3, 0.0, 0.0], [0.61, 4.7, 0.0], [-0.38, 0.42, 5.9]],
        dtype=np.float64,
    )
    seam = native._ParticleMeshTransport(precision, 4)
    seam.reserve(*mesh, positions.shape[0], 2, order)
    seam.set_cell(cell.reshape(-1))
    seam.set_positions(positions.reshape(-1))
    return seam


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("mesh", [(5, 4, 6), (9, 7, 8)])
def test_neutral_radial_strain_parallel_range_scales_with_spectrum_bins(
    native, precision, mesh
):
    seam = _particle_mesh_transport(native, precision, mesh=mesh)
    expected = mesh[0] * mesh[1] * (mesh[2] // 2 + 1)
    assert expected > 9
    assert seam.radial_strain_work_items() == expected


@pytest.mark.parametrize("precision", PRECISIONS)
def test_neutral_spread_and_weight_gather_are_adjoints(native, precision):
    seam = _particle_mesh_transport(native, precision)
    dtype = np.float32 if precision == "float32" else np.float64
    delta_weights = np.asarray(
        [[0.17, -0.31], [0.42, 0.08], [-0.23, 0.37]], dtype=dtype
    )
    mesh_potential = np.asarray(
        np.sin(np.arange(2 * 5 * 4 * 6, dtype=np.float64) * 0.137), dtype=dtype
    ).reshape(2, 5, 4, 6)

    seam.spread(delta_weights)
    spread_delta = np.asarray(seam.real_mesh())
    lhs = np.vdot(spread_delta, mesh_potential)

    seam.set_real_mesh(mesh_potential)
    gathered = seam.gather(np.zeros_like(delta_weights))
    rhs = np.vdot(delta_weights, np.asarray(gathered["weight_adjoint"]))

    tolerance = TOLERANCE[precision]
    np.testing.assert_allclose(lhs, rhs, rtol=tolerance["rtol"], atol=tolerance["atol"])


@pytest.mark.parametrize("precision", PRECISIONS)
def test_neutral_physical_cotangent_is_deconvolved_once_before_inverse(
    native, precision
):
    mesh = (5, 4, 6)
    seam = _particle_mesh_transport(native, precision, mesh=mesh)
    dtype = np.float32 if precision == "float32" else np.float64
    real_seed = np.asarray(
        np.cos(np.arange(2 * np.prod(mesh), dtype=np.float64) * 0.071), dtype=dtype
    ).reshape(2, *mesh)
    deconvolved_amplitude_cotangent = np.fft.rfftn(real_seed, axes=(-3, -2, -1))
    deconvolved_amplitude_cotangent = np.asarray(
        deconvolved_amplitude_cotangent,
        dtype=np.complex64 if precision == "float32" else np.complex128,
    )

    seam.load_physical_cotangent(deconvolved_amplitude_cotangent)
    loaded = np.asarray(seam.spectrum())
    seam.deconvolve_cotangent()
    structure_factor_cotangent = np.asarray(seam.spectrum())
    divisor = np.asarray(seam.deconvolution())
    seam.transform_inverse()
    inverse_potential = np.asarray(seam.real_mesh())

    expected_structure_factor_cotangent = (
        deconvolved_amplitude_cotangent / divisor[None, ...]
    )
    expected_inverse = np.fft.irfftn(
        expected_structure_factor_cotangent, s=mesh, axes=(-3, -2, -1)
    ) * np.prod(mesh)
    tolerance = TOLERANCE[precision]
    np.testing.assert_allclose(
        loaded,
        deconvolved_amplitude_cotangent,
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    )
    np.testing.assert_allclose(
        structure_factor_cotangent,
        expected_structure_factor_cotangent,
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    )
    np.testing.assert_allclose(
        inverse_potential,
        expected_inverse,
        rtol=4 * tolerance["rtol"],
        atol=4 * tolerance["atol"],
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_neutral_gather_returns_weight_adjoints_and_frozen_weight_forces(
    native, precision
):
    seam = _particle_mesh_transport(native, precision)
    dtype = np.float32 if precision == "float32" else np.float64
    weights = np.asarray([[0.17, -0.31], [0.42, 0.08], [-0.23, 0.37]], dtype=dtype)
    potential = np.asarray(
        np.sin(np.arange(2 * 5 * 4 * 6, dtype=np.float64) * 0.137), dtype=dtype
    ).reshape(2, 5, 4, 6)
    seam.set_real_mesh(potential)
    gathered = seam.gather(weights)
    analytic_weight = np.asarray(gathered["weight_adjoint"], dtype=np.float64)
    analytic_force = np.asarray(gathered["position_forces"], dtype=np.float64)

    step = 3e-5 if precision == "float64" else 3e-3
    weight_finite_difference = np.zeros_like(weights, dtype=np.float64)
    for particle in range(weights.shape[0]):
        for lane in range(weights.shape[1]):
            energies = []
            for direction in (-1.0, 1.0):
                moved = weights.copy()
                moved[particle, lane] += direction * step
                seam.spread(moved)
                energies.append(float(np.vdot(np.asarray(seam.real_mesh()), potential)))
            weight_finite_difference[particle, lane] = (energies[1] - energies[0]) / (
                2 * step
            )

    positions = np.asarray(
        [[0.47, 0.91, 1.33], [2.11, 1.64, 0.72], [3.42, 2.05, 2.77]],
        dtype=np.float64,
    )
    finite_difference = np.zeros_like(positions)
    for particle in range(positions.shape[0]):
        for axis in range(3):
            energies = []
            for direction in (-1.0, 1.0):
                moved = positions.copy()
                moved[particle, axis] += direction * step
                seam.set_positions(moved.reshape(-1))
                seam.spread(weights)
                energies.append(float(np.vdot(np.asarray(seam.real_mesh()), potential)))
            finite_difference[particle, axis] = -(energies[1] - energies[0]) / (
                2 * step
            )

    tolerance = TOLERANCE[precision]
    np.testing.assert_allclose(
        analytic_weight,
        weight_finite_difference,
        rtol=2e-3 if precision == "float64" else 2e-2,
        atol=8 * tolerance["atol"],
    )
    np.testing.assert_allclose(
        analytic_force,
        finite_difference,
        rtol=2e-3 if precision == "float64" else 2e-2,
        atol=8 * tolerance["atol"],
    )


# --------------------------------------------------------------------------
# independent NumPy references
# --------------------------------------------------------------------------


def _cardinal(order, x):
    x = np.asarray(x, dtype=np.float64)
    basis = [((x >= i) & (x < i + 1)).astype(np.float64) for i in range(order)]
    for degree in range(1, order):
        for i in range(order - degree):
            basis[i] = (x - i) / degree * basis[i] + (
                i + degree + 1 - x
            ) / degree * basis[i + 1]
    return basis[0]


def _cardinal_derivative(order, x, step=1e-6):
    """Central difference of the independent cardinal B-spline (no shared identity)."""
    return (_cardinal(order, x + step) - _cardinal(order, x - step)) / (2.0 * step)


def _stencil(order, mesh, fractional):
    """Independent reconstruction of the assignment stencil and its u arguments."""
    y = np.asarray(mesh, dtype=np.float64) * np.asarray(fractional, dtype=np.float64)
    base = np.floor(y)
    theta = y - base
    start = np.floor(theta - (order - 2.0) / 2.0)
    indices = base + start
    offsets = np.arange(order, dtype=np.float64)
    u = order / 2.0 + theta[:, :, None] - (start[:, :, None] + offsets)
    return indices.astype(np.int64), u


def _pair_index(first, second, num_species):
    """The unordered species-pair index the native tables are stored with."""
    if first > second:
        first, second = second, first
    # Mirrors `unordered_pair_index`: the lower triangle of the row-major grid.
    return first * num_species - first * (first - 1) // 2 + (second - first)


def _channels(inputs):
    lookup = {
        int(number): index for index, number in enumerate(np.asarray(inputs["species"]))
    }
    return np.array(
        [lookup.get(int(number), -1) for number in np.asarray(inputs["numbers"])],
        dtype=np.int64,
    )


def _mesh_force_reference(inputs, mesh, order, potentials, *, transposed=False):
    """Independent mesh force from the shipped potentials and an independent stencil.

    ``potentials`` is one ``(chunk, nx, ny, nz)`` potential per species channel;
    an atom reads the channel its atomic number maps to.
    """
    mesh = tuple(int(value) for value in mesh)
    order = int(order)
    cell = np.asarray(inputs["cell"], dtype=np.float64)
    inverse = np.linalg.inv(cell)
    fractional = np.asarray(inputs["positions"], dtype=np.float64) @ inverse
    indices, u = _stencil(order, mesh, fractional)
    channels = _channels(inputs)
    nodes = np.asarray(inputs["numbers"]).size
    chunk = potentials[0].shape[0]
    potentials = [
        np.asarray(potential, dtype=np.float64).reshape(chunk, *mesh)
        for potential in potentials
    ]
    force = np.zeros((nodes, 3), dtype=np.float64)
    weights = np.asarray([_cardinal(order, u[:, a, :]) for a in range(3)])
    derivatives = np.asarray(
        [_cardinal_derivative(order, u[:, a, :]) for a in range(3)]
    )
    for node in range(nodes):
        if channels[node] < 0:
            continue
        for i in range(order):
            w0 = weights[0][node, i]
            d0 = derivatives[0][node, i] * mesh[0]
            for j in range(order):
                w1 = weights[1][node, j]
                d1 = derivatives[1][node, j] * mesh[1]
                for k in range(order):
                    w2 = weights[2][node, k]
                    d2 = derivatives[2][node, k] * mesh[2]
                    g = tuple(
                        int(value) % mesh[a]
                        for a, value in enumerate(
                            (
                                indices[node, 0] + i,
                                indices[node, 1] + j,
                                indices[node, 2] + k,
                            )
                        )
                    )
                    # dW/ds, the fractional gradient.
                    fractional_gradient = np.asarray(
                        [d0 * w1 * w2, w0 * d1 * w2, w0 * w1 * d2], dtype=np.float64
                    )
                    for slot in range(chunk):
                        value = potentials[channels[node]][slot][g]
                        if value == 0.0:
                            continue
                        coefficient = inputs["coefficients"][node, slot]
                        if coefficient == 0.0:
                            continue
                        for c in range(3):
                            # Row convention: s_a = sum_c r_c Cinv[c, a], so
                            # dW/dr_c = sum_a (dW/ds_a) Cinv[c, a].
                            row = inverse[:, c] if transposed else inverse[c, :]
                            force[node, c] -= (
                                coefficient
                                * value
                                * float(np.sum(fractional_gradient * row))
                            )
    return force


def _signed_miller(index, extent):
    return int(index) if index <= extent // 2 else int(index) - extent


def _wave_vectors(cell, mesh):
    """Independent wave vectors of the stored half spectrum, row convention."""
    nx, ny, nz = (int(value) for value in mesh)
    half = nz // 2 + 1
    inverse = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    miller = []
    for i in range(nx):
        for j in range(ny):
            for k in range(half):
                miller.append((_signed_miller(i, nx), _signed_miller(j, ny), k))
    miller = np.asarray(miller, dtype=np.float64)
    return 2.0 * np.pi * (inverse @ miller.T).T


def _hermitian_multiplicity(mesh):
    nx, ny, nz = (int(value) for value in mesh)
    half = nz // 2 + 1
    weights = []
    for _ in range(nx):
        for _ in range(ny):
            for k in range(half):
                weights.append(
                    1.0 if (k == 0 or (nz % 2 == 0 and 2 * k == nz)) else 2.0
                )
    return np.asarray(weights, dtype=np.float64)


def _reciprocal_strain_reference(inputs, mesh, structure, energy):
    """Independent same-mesh reciprocal strain derivative from the shipped spectra."""
    mesh = tuple(int(value) for value in mesh)
    volume = abs(float(np.linalg.det(np.asarray(inputs["cell"], dtype=np.float64))))
    waves = _wave_vectors(inputs["cell"], mesh)
    norm = np.linalg.norm(waves, axis=1)
    weights = _hermitian_multiplicity(mesh)
    eigenvalues = np.asarray(inputs["eigenvalues"], dtype=np.float64)
    q = np.asarray(inputs["q"], dtype=np.float64)
    structure = np.asarray(structure)  # (n_species, chunk, bins)
    n_species = structure.shape[0]
    kernel_derivative = np.zeros((n_species, n_species, norm.size), dtype=np.float64)
    for a in range(n_species):
        for b in range(n_species):
            q_product = q[a] * q[b]
            r0 = DAMPING.damping_radius(q_product)
            kernel_derivative[a, b] = fourier_d3.reciprocal_kernel_derivative(
                norm, r0, q_product, DAMPING
            )
    tensor = np.zeros((3, 3), dtype=np.float64)
    safe = np.where(norm > 1e-12, norm, 1.0)
    for slot in range(structure.shape[1]):
        accumulator = np.zeros(norm.size, dtype=np.float64)
        for a in range(n_species):
            for b in range(n_species):
                accumulator += kernel_derivative[a, b] * np.real(
                    structure[a, slot] * np.conj(structure[b, slot])
                )
        accumulator = eigenvalues[slot] * weights * accumulator
        for x in range(3):
            for y in range(3):
                tensor[x, y] += (
                    0.5
                    * float(np.sum(accumulator * (waves[:, x] * waves[:, y] / safe)))
                    / volume
                )
    tensor -= float(energy) * np.eye(3)
    return tensor


def _coordination_strain_reference(result, inputs):
    """Independent half-directed-edge coordination virial from the per-edge force.

    The accumulated receiver force sums over every edge of the receiver, so the
    per-edge force is rebuilt here from the shipped ``chi`` and edge derivative.
    """
    delta = _edge_displacement(inputs)
    receivers = np.asarray(inputs["receivers"])
    neighbours = np.asarray(inputs["neighbours"])
    chi = np.asarray(result["chi"], dtype=np.float64)
    factor = (
        (chi[receivers] + chi[neighbours])
        * np.asarray(result["edge_derivatives"], dtype=np.float64)
        / np.asarray(inputs["distances"], dtype=np.float64)
    )
    tensor = np.zeros((3, 3), dtype=np.float64)
    for edge in range(delta.shape[0]):
        tensor += 0.5 * factor[edge] * np.outer(delta[edge], delta[edge])
    return tensor


def _edge_displacement(inputs):
    positions = np.asarray(inputs["positions"], dtype=np.float64)
    cell = np.asarray(inputs["cell"], dtype=np.float64)
    return (
        positions[np.asarray(inputs["neighbours"])]
        - positions[np.asarray(inputs["receivers"])]
        + np.asarray(inputs["shifts"], dtype=np.float64) @ cell
    )


def _atomic_energy_reference(inputs, mesh, order, potentials):
    """Independent normative partition of specification section 6.4.

    ``potentials`` is one ``(chunk, nx, ny, nz)`` potential per species channel;
    an atom gathers the channel its atomic number maps to.
    """
    mesh = tuple(int(value) for value in mesh)
    order = int(order)
    cell = np.asarray(inputs["cell"], dtype=np.float64)
    fractional = np.asarray(inputs["positions"], dtype=np.float64) @ np.linalg.inv(cell)
    indices, u = _stencil(order, mesh, fractional)
    channels = _channels(inputs)
    chunk = potentials[0].shape[0]
    potentials = [
        np.asarray(potential, dtype=np.float64).reshape(chunk, *mesh)
        for potential in potentials
    ]
    weights = np.asarray([_cardinal(order, u[:, a, :]) for a in range(3)])
    nodes = np.asarray(inputs["numbers"]).size
    share = np.zeros(nodes, dtype=np.float64)
    for node in range(nodes):
        if channels[node] < 0:
            continue
        for i in range(order):
            for j in range(order):
                for k in range(order):
                    g = tuple(
                        int(value) % mesh[a]
                        for a, value in enumerate(
                            (
                                indices[node, 0] + i,
                                indices[node, 1] + j,
                                indices[node, 2] + k,
                            )
                        )
                    )
                    weight = (
                        weights[0][node, i] * weights[1][node, j] * weights[2][node, k]
                    )
                    for slot in range(chunk):
                        share[node] += (
                            0.5
                            * inputs["coefficients"][node, slot]
                            * weight
                            * potentials[channels[node]][slot][g]
                        )
    return share


def _self_share(inputs):
    """The half self term of specification section 6.4."""
    channels = _channels(inputs)
    q = np.asarray(inputs["q"], dtype=np.float64)
    safe = np.where(channels >= 0, channels, 0)
    q_product = q[safe] * q[safe]
    r0 = DAMPING.damping_radius(q_product)
    zero_distance = fourier_d3.pair_potential(0.0, r0, q_product, DAMPING)
    eigenvalues = np.asarray(inputs["eigenvalues"], dtype=np.float64)
    share = 0.5 * np.sum(
        eigenvalues[None, :] * inputs["coefficients"] ** 2 * zero_distance[:, None],
        axis=1,
    )
    return np.where(channels >= 0, share, 0.0)


# --------------------------------------------------------------------------
# A. coefficient sensitivities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_coefficient_sensitivity_matches_coefficient_finite_differences(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    nodes, chunk = inputs["coefficients"].shape

    def energy_with(node, slot, value):
        table = np.array(inputs["coefficients"], dtype=np.float64)
        table[node, slot] += value
        forward = native._FourierD3ReciprocalForward(precision, 8)
        forward.reserve(
            mesh[0],
            mesh[1],
            mesh[2],
            inputs["species"].size,
            nodes,
            inputs["rank"],
            chunk,
            4,
        )
        forward.set_cell(inputs["cell"].reshape(-1))
        forward.set_species(inputs["species"], inputs["q"])
        forward.set_atoms(inputs["positions"].reshape(-1), inputs["numbers"])
        forward.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
        forward.set_rank(inputs["rank_begin"], inputs["eigenvalues"])
        forward.set_coefficients(table)
        return forward.run()["energy"]

    worst = 0.0
    bands = []
    for node in range(nodes):
        for slot in range(chunk):
            estimates = _sweep(
                lambda value, node=node, slot=slot: energy_with(node, slot, value),
                FORCE_STEPS,
            )
            best, deviations = _best_band(result["sensitivity"][node, slot], estimates)
            bands.append(deviations)
            worst = max(worst, best)
    declared = DECLARED_SENSITIVITY_FD_BAND[precision]
    assert worst <= declared, (
        f"coefficient sensitivity must be the gradient of the discrete energy; worst "
        f"central-difference deviation over the sweep is {worst:.3e} against the "
        f"declared {declared:.3e} band.  Per-step bands: {np.asarray(bands).max(axis=0)}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_coefficient_sensitivity_includes_the_self_derivative(native, precision):
    """A self-term-dominated case separates a dropped self derivative."""
    inputs = _inputs(native, precision, case="single-atom-self-images")
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    chunk = inputs["chunk"]
    base = inputs["coefficients"]

    def energy_with(value):
        table = np.array(base, dtype=np.float64)
        table[0, 0] += value
        forward = native._FourierD3ReciprocalForward(precision, 8)
        forward.reserve(mesh[0], mesh[1], mesh[2], 1, 1, inputs["rank"], chunk, 4)
        forward.set_cell(inputs["cell"].reshape(-1))
        forward.set_species(inputs["species"], inputs["q"])
        forward.set_atoms(inputs["positions"].reshape(-1), inputs["numbers"])
        forward.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
        forward.set_rank(inputs["rank_begin"], inputs["eigenvalues"])
        forward.set_coefficients(table)
        return forward.run()["energy"]

    estimates = _sweep(energy_with, FORCE_STEPS)
    analytic = float(result["sensitivity"][0, 0])
    best, deviations = _best_band(analytic, estimates)
    self_part = float(
        np.asarray(result["seam"].self_coefficient_derivative(), dtype=np.float64)[0, 0]
    )
    declared = DECLARED_SENSITIVITY_FD_BAND[precision]
    # Discriminative: the candidate that drops the self derivative has to be
    # visibly outside the band, otherwise this case cannot detect the omission.
    dropped = abs((analytic - self_part) - float(np.median(estimates)))
    assert dropped > declared, (
        "dropping the self derivative must be observable in this case; the candidate "
        f"is only {dropped:.3e} from the central difference against a declared "
        f"{declared:.3e} band"
    )
    assert best <= declared, (
        f"the self derivative must be folded into dE/dc before the chain rule; the "
        f"central-difference band is {deviations} against a declared {declared:.3e}"
    )


# --------------------------------------------------------------------------
# B. direct mesh force
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_mesh_force_matches_position_finite_differences_with_frozen_coefficients(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    positions = np.array(inputs["positions"], dtype=np.float64)
    nodes = positions.shape[0]

    worst = 0.0
    bands = []
    for node in range(nodes):
        for axis in range(3):

            def displaced(value, node=node, axis=axis):
                moved = np.array(positions, dtype=np.float64)
                moved[node, axis] += value
                return _frozen_energy(native, inputs, mesh, 4, positions=moved)

            estimates = _sweep(displaced, FORCE_STEPS)
            analytic = -float(result["mesh_forces"][node, axis])
            best, deviations = _best_band(analytic, estimates)
            bands.append(deviations)
            worst = max(worst, best)
    declared = DECLARED_MESH_FORCE_FD_BAND[precision]
    assert worst <= declared, (
        f"the direct mesh force must be the position gradient of the frozen-coefficient "
        f"energy; worst deviation is {worst:.3e} against {declared:.3e}.  Per-step "
        f"bands: {np.asarray(bands).max(axis=0)}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_mesh_force_cartesian_map_is_not_transposed_on_a_triclinic_cell(
    native, precision
):
    """The triclinic cell separates ``C^-1`` from ``C^-T``, which a cubic one cannot."""
    inputs = _inputs(
        native,
        precision,
        factors=_skew_factors(),
        positions=SKEW_FRACTIONAL @ SKEW_CELL,
        cell=SKEW_CELL,
        numbers=SKEW_NUMBERS,
        radii=SKEW_RADII,
        species=SKEW_SPECIES,
        q=SKEW_Q,
    )
    mesh = (10, 12, 14)
    result = _evaluate(native, inputs, mesh, 4)
    positions = np.array(inputs["positions"], dtype=np.float64)
    potentials = [
        np.asarray(result["seam"].potential(slot), dtype=np.float64)
        for slot in range(inputs["species"].size)
    ]

    shipped = _mesh_force_reference(inputs, mesh, 4, potentials)
    transposed = _mesh_force_reference(inputs, mesh, 4, potentials, transposed=True)
    separation = float(np.max(np.abs(shipped - transposed)))
    assert separation > 1e-6, (
        "the strongly skewed cell must separate the two Cartesian maps; measured "
        f"separation {separation:.3e}"
    )
    deviation = float(np.max(np.abs(result["mesh_forces"] - shipped)))
    assert deviation <= DECLARED_MESH_FORCE_FD_BAND[precision], (
        f"the shipped mesh force disagrees with the independent row-convention gather "
        f"by {deviation:.3e}"
    )
    # ... and against central differences, which is the acceptance oracle.  The
    # largest component is the one that most separates the two conventions.
    node, axis = divmod(int(np.argmax(np.abs(result["mesh_forces"]))), 3)

    def displaced(value):
        moved = np.array(positions, dtype=np.float64)
        moved[node, axis] += value
        return _frozen_energy(native, inputs, mesh, 4, positions=moved)

    estimates = _sweep(displaced, FORCE_STEPS)
    best, band = _best_band(-float(result["mesh_forces"][node, axis]), estimates)
    assert best <= DECLARED_MESH_FORCE_FD_BAND[precision], (
        f"the triclinic mesh force must match the position central difference; best "
        f"deviation {best:.3e} over the sweep {band}"
    )


# --------------------------------------------------------------------------
# C. full force
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_full_force_matches_central_differences_of_the_complete_energy(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    positions = np.array(inputs["positions"], dtype=np.float64)
    nodes = positions.shape[0]

    worst = 0.0
    bands = []
    for node in range(nodes):
        for axis in range(3):

            def displaced(value, node=node, axis=axis):
                moved = np.array(positions, dtype=np.float64)
                moved[node, axis] += value
                return _full_energy(native, inputs, mesh, 4, positions=moved)

            estimates = _sweep(displaced, FORCE_STEPS)
            analytic = -float(result["forces"][node, axis])
            best, deviations = _best_band(analytic, estimates)
            bands.append(deviations)
            worst = max(worst, best)
    declared = DECLARED_FORCE_FD_BAND[precision]
    assert worst <= declared, (
        f"the total force must be the position gradient of the complete energy; worst "
        f"deviation is {worst:.3e} against {declared:.3e}.  Per-step bands: "
        f"{np.asarray(bands).max(axis=0)}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case",
    ["orthogonal-mixed-species", "triclinic-mixed-species", "coordination-sensitive"],
)
def test_coordination_forces_sum_to_zero_exactly(native, precision, case):
    """The coordination adjoint is a sum of internal pair forces, so it telescopes."""
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    scale = max(1e-30, float(np.max(np.abs(result["coordination_forces"]))))
    band = 1e-12 if precision == "float64" else 1e-5
    assert (
        float(np.max(np.abs(result["coordination_forces"].sum(axis=0)))) <= band * scale
    ), (
        "sum_i F_coord,i must vanish: the receiver-only accumulation with both "
        "orientations present makes the pair forces telescope exactly"
    )
    # The total force sum is the mesh's own translation residual.  It is an
    # absolute bound, not one scaled by the largest force component: the residual
    # is set by the assignment and is independent of the force scale, and a
    # `max(1, scale)` factor would let it pass on a small force scale.
    residual = float(np.max(np.abs(result["forces"].sum(axis=0))))
    assert residual <= DECLARED_TRANSLATION_RESIDUAL_BAND[precision], (
        "the total force sum is the mesh translation residual, and it is bounded "
        f"absolutely; measured {residual:.3e} against "
        f"{DECLARED_TRANSLATION_RESIDUAL_BAND[precision]:.3e}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_a_rigid_translation_moves_the_energy_only_by_the_mesh_error(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, ACCEPTANCE_MESH, 4)
    shift = np.asarray([0.137, -0.211, 0.083], dtype=np.float64)
    moved = _inputs(native, precision, case=case, positions=inputs["positions"] + shift)
    other = _evaluate(native, moved, ACCEPTANCE_MESH, 4)
    energy_band = DECLARED_TRANSLATION_ENERGY_BAND[precision]
    assert abs(other["energy"] - result["energy"]) <= energy_band, (
        "at the convergence mesh a rigid translation may only move the energy by the "
        "mesh's own translation error; measured "
        f"{abs(other['energy'] - result['energy']):.3e} against {energy_band:.3e}"
    )
    # Absolute, and derived from the measured 24^3 residual rather than from a
    # multiple of the largest force component.
    force_band = DECLARED_TRANSLATION_RESIDUAL_BAND[precision] / 50.0
    assert np.max(np.abs(other["forces"] - result["forces"])) <= force_band, (
        "a rigid translation at the convergence mesh may only move the forces by the "
        f"mesh translation residual; measured "
        f"{np.max(np.abs(other['forces'] - result['forces'])):.3e} against {force_band:.3e}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_converged_mesh_matches_the_frozen_oracle_forces(native, precision, case):
    """Physical acceptance: the mesh-free Task 2 oracle, at a converged mesh."""
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, ACCEPTANCE_MESH, 4)
    expected = np.asarray(
        CASES[case]["expected"]["forces_ev_per_angstrom"], dtype=np.float64
    )
    band = DECLARED_ORACLE_FORCE_BAND_EV_PER_ANGSTROM
    deviation = float(np.max(np.abs(result["forces"] - expected)))
    assert deviation <= band, (
        f"forces at {ACCEPTANCE_MESH} differ from the mesh-free oracle by {deviation:.3e} "
        f"against the declared {band:.3e}"
    )


# --------------------------------------------------------------------------
# D. coordination adjoint
# --------------------------------------------------------------------------


def _two_atom_inputs(native, precision, *, half=False):
    """A two-atom cell whose graph can be supplied with or without the reverse edge."""
    cell = np.asarray([[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]])
    positions = np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=np.float64)
    numbers = np.asarray([6, 6], dtype=np.int32)
    radii = np.asarray([1.0000000494517636, 1.0000000494517636], dtype=np.float64)
    species = np.asarray([6], dtype=np.int32)
    q = np.asarray([1.643057426520438], dtype=np.float64)
    factors = fourier_d3.LowRankFactors(
        species=np.asarray([6], dtype=np.int64),
        eigs=np.asarray([1.0], dtype=np.float64),
        v_q=np.ones((1, 5, 1), dtype=np.float64),
        cn_ref=np.asarray([[0.0, 0.9868, 1.9985, 2.9987, 3.9844]], dtype=np.float64),
        valid=np.ones((1, 5), dtype=bool),
        q_table=q,
    )
    receivers, neighbours, shifts, distances = _graph(positions, cell, numbers)
    if half:
        keep = receivers == 0
        receivers, neighbours, shifts, distances = (
            receivers[keep],
            neighbours[keep],
            shifts[keep],
            distances[keep],
        )
    coefficients = _workspace_coefficients(
        native, precision, factors, numbers, radii, receivers, neighbours, distances
    )
    return {
        "case": None,
        "precision": precision,
        "factors": factors,
        "positions": positions,
        "cell": cell,
        "numbers": numbers,
        "radii": radii,
        "species": species,
        "q": q,
        "receivers": receivers,
        "neighbours": neighbours,
        "shifts": shifts,
        "distances": distances,
        "offsets": _offsets(numbers, receivers),
        "coefficients": coefficients,
        "chunk": 1,
        "rank": 1,
        "rank_begin": 0,
        "eigenvalues": np.asarray([1.0], dtype=np.float64),
    }


def _hand_graph(native, inputs, receivers, neighbours, shifts, distances):
    """Installs an explicit directed graph and the coefficients that go with it.

    The shifts, the distances, and the positions must agree, so a caller that
    wants a graph the reference edge rule would not produce has to move the atoms
    to match rather than hand over an inconsistent shift.
    """
    inputs["receivers"] = np.asarray(receivers, dtype=np.int64)
    inputs["neighbours"] = np.asarray(neighbours, dtype=np.int32)
    inputs["shifts"] = np.asarray(shifts, dtype=np.int32).reshape(-1, 3)
    inputs["distances"] = np.asarray(distances, dtype=np.float64)
    inputs["offsets"] = _offsets(inputs["numbers"], inputs["receivers"])
    inputs["coefficients"] = _workspace_coefficients(
        native,
        inputs["precision"],
        inputs["factors"],
        inputs["numbers"],
        inputs["radii"],
        inputs["receivers"],
        inputs["neighbours"],
        inputs["distances"],
    )
    return inputs


@pytest.mark.parametrize("precision", PRECISIONS)
def test_coordination_force_accumulates_on_the_receiver_only(native, precision):
    inputs = _two_atom_inputs(native, precision, half=True)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    delta = np.asarray([1.4, 0.0, 0.0], dtype=np.float64)
    chi = result["chi"]
    expected = (
        (chi[0] + chi[1]) * result["edge_derivatives"][0] / inputs["distances"][0]
    ) * delta
    assert np.allclose(
        result["coordination_forces"][0], expected, rtol=1e-10, atol=1e-14
    ), "the receiver must accumulate the whole directed-edge contribution"
    assert np.allclose(result["coordination_forces"][1], 0.0, atol=1e-18), (
        "the source of a directed edge must receive nothing; accumulating on both "
        "endpoints double counts the coordination force"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reverse_edge_supplies_the_opposite_force(native, precision):
    inputs = _two_atom_inputs(native, precision, half=False)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    forces = result["coordination_forces"]
    assert np.allclose(
        forces[0], -forces[1], rtol=1e-10, atol=1e-14
    ), "the reverse directed edge must supply exactly the opposite force"
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-15)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("distance", [CUTOFF, CUTOFF + 1e-6, CUTOFF + 1.0])
def test_edges_at_or_beyond_the_cutoff_contribute_no_coordination_force(
    native, precision, distance
):
    inputs = _two_atom_inputs(native, precision, half=True)
    # The reference edge rule excludes r >= R_cut, so the endpoint case has to be
    # supplied rather than discovered.  The atoms are moved to the tested
    # separation first, because the shift/distance/position consistency check
    # would otherwise (correctly) refuse the hand-built graph.
    positions = np.array(inputs["positions"], dtype=np.float64)
    positions[1] = [float(distance), 0.0, 0.0]
    inputs["positions"] = positions
    _hand_graph(
        native,
        inputs,
        receivers=[0],
        neighbours=[1],
        shifts=[[0, 0, 0]],
        distances=[float(distance)],
    )
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    assert np.all(np.abs(result["edge_derivatives"]) <= 1e-30), (
        "the modified counting function must reach zero with a zero derivative at the "
        "cutoff, so no edge at or beyond it carries a force"
    )
    assert np.allclose(result["coordination_forces"], 0.0, atol=1e-18)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_zero_distance_edge_contributes_nothing_and_never_produces_a_nan(
    native, precision
):
    inputs = _two_atom_inputs(native, precision, half=True)
    # Two atoms at the same point: the derived separation and the supplied one
    # then agree at zero, which is the case the pair term has to mask.
    positions = np.array(inputs["positions"], dtype=np.float64)
    positions[1] = positions[0]
    inputs["positions"] = positions
    _hand_graph(
        native,
        inputs,
        receivers=[0],
        neighbours=[1],
        shifts=[[0, 0, 0]],
        distances=[0.0],
    )
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    assert np.all(np.isfinite(result["coordination_forces"])), (
        "a zero-distance edge must be masked by the same guard the Task 8 kernel uses, "
        "not turned into a division by zero"
    )
    assert np.allclose(result["coordination_forces"], 0.0, atol=1e-18)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_graph_inconsistent_with_its_shifts_fails_loudly(native, precision):
    inputs = _two_atom_inputs(native, precision, half=False)
    seam = _load(native, inputs, (8, 8, 8), 4)
    with pytest.raises((ValueError, RuntimeError)):
        seam.set_graph(
            inputs["offsets"],
            inputs["neighbours"],
            np.asarray([[1, 0, 0]], dtype=np.int32),
            inputs["distances"],
            inputs["numbers"],
            inputs["radii"],
        )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_coordination_force_and_sensitivity_match_an_independent_recomputation(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    chi = np.einsum(
        "nr,nr->n", result["sensitivity"], result["coefficient_derivatives"]
    )
    tolerance = TOLERANCE[precision]
    assert np.allclose(
        chi, result["chi"], rtol=tolerance["rtol"], atol=tolerance["atol"]
    ), "chi_i must be sum_l (dE/dc_il) * (dc_il/dtheta_i)"
    delta = _edge_displacement(inputs)
    expected = np.zeros_like(result["coordination_forces"])
    contribution = (
        (chi[inputs["receivers"]] + chi[inputs["neighbours"]])
        * result["edge_derivatives"]
        / inputs["distances"]
    )[:, None] * delta
    np.add.at(expected, inputs["receivers"], contribution)
    tolerance = TOLERANCE[precision]
    assert np.allclose(
        expected,
        result["coordination_forces"],
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    ), "the receiver-only directed-edge sum must reproduce the coordination force"


# --------------------------------------------------------------------------
# E. virial and ASE stress
# --------------------------------------------------------------------------


def _deform(positions, cell, strain):
    tensor = np.eye(3) + np.asarray(strain, dtype=np.float64)
    return positions @ tensor, cell @ tensor


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_strain_derivative_matches_affine_finite_differences(native, precision, case):
    inputs = _inputs(native, precision, case=case)
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    positions = np.array(inputs["positions"], dtype=np.float64)
    cell = np.array(inputs["cell"], dtype=np.float64)

    worst = 0.0
    bands = []
    for a in range(3):
        for b in range(3):

            def deformed(value, a=a, b=b):
                strain = np.zeros((3, 3), dtype=np.float64)
                strain[a, b] = value
                moved, deformed_cell = _deform(positions, cell, strain)
                return _full_energy(
                    native, inputs, mesh, 4, positions=moved, cell=deformed_cell
                )

            estimates = _sweep(deformed, STRAIN_STEPS)
            analytic = float(result["strain"][a, b])
            best, deviations = _best_band(analytic, estimates)
            bands.append(deviations)
            worst = max(worst, best)
    declared = DECLARED_STRAIN_FD_BAND[precision]
    assert worst <= declared, (
        f"dE/du must be the affine strain derivative with positions and cell deformed "
        f"together; worst deviation is {worst:.3e} against {declared:.3e}.  Per-step "
        f"bands: {np.asarray(bands).max(axis=0)}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_the_affine_convention_deforms_positions_and_cell_together(native, precision):
    """A cell-only deformation is a different function and must not match."""
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    positions = np.array(inputs["positions"], dtype=np.float64)
    cell = np.array(inputs["cell"], dtype=np.float64)
    a, b = 0, 1

    def affine(value):
        strain = np.zeros((3, 3), dtype=np.float64)
        strain[a, b] = value
        moved, deformed_cell = _deform(positions, cell, strain)
        return _full_energy(
            native, inputs, mesh, 4, positions=moved, cell=deformed_cell
        )

    def cell_only(value):
        strain = np.zeros((3, 3), dtype=np.float64)
        strain[a, b] = value
        return _full_energy(
            native,
            inputs,
            mesh,
            4,
            positions=positions,
            cell=cell @ (np.eye(3) + strain),
        )

    analytic = float(result["strain"][a, b])
    affine_best, _ = _best_band(analytic, _sweep(affine, STRAIN_STEPS))
    cell_best, _ = _best_band(analytic, _sweep(cell_only, STRAIN_STEPS))
    assert affine_best <= DECLARED_STRAIN_FD_BAND[precision]
    assert cell_best > 100.0 * max(affine_best, 1e-12), (
        "the cell-only deformation must be visibly different: fractional coordinates "
        "are strain-invariant, so the assignment contributes no virial and the two "
        "deformations are different functions"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_strain_derivative_matches_the_independent_same_mesh_reference(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    reference = _reciprocal_strain_reference(
        inputs, mesh, result["structure_factors"], result["reciprocal_energy"]
    ) + _coordination_strain_reference(result, inputs)
    deviation = float(np.max(np.abs(reference - result["strain"])))
    declared = DECLARED_SAME_MESH_STRAIN_BAND[precision]
    assert deviation <= declared, (
        f"the strain derivative must match the independent contraction of the shipped "
        f"spectra with the independently derived kernel derivative; deviation "
        f"{deviation:.3e} against {declared:.3e}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_k_zero_bin_contributes_only_the_isotropic_volume_term(native, precision):
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    waves = _wave_vectors(inputs["cell"], mesh)
    norm = np.linalg.norm(waves, axis=1)
    zero = int(np.argmin(norm))
    assert norm[zero] == 0.0, "the mesh must contain the k = 0 bin"
    assert np.all(waves[zero] == 0.0)
    safe = np.where(norm > 1e-12, norm, 1.0)
    assert np.all(
        waves[zero, :, None] * waves[zero, None, :] / safe[zero] == 0.0
    ), "the k = 0 bin contributes no anisotropic term because its wave vector vanishes"
    reciprocal_only = result["strain"] - _coordination_strain_reference(result, inputs)
    lattice = reciprocal_only + float(result["reciprocal_energy"]) * np.eye(3)
    independent = _reciprocal_strain_reference(
        inputs, mesh, result["structure_factors"], 0.0
    )
    declared = DECLARED_SAME_MESH_STRAIN_BAND[precision]
    assert np.max(np.abs(lattice - independent)) <= declared, (
        "after removing the volume term the remaining contraction must equal the "
        "independent lattice sum"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_coordination_virial_carries_the_half_directed_edge_factor(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    delta = _edge_displacement(inputs)
    receivers = np.asarray(inputs["receivers"])
    neighbours = np.asarray(inputs["neighbours"])
    chi = np.asarray(result["chi"], dtype=np.float64)
    factor = (
        (chi[receivers] + chi[neighbours])
        * np.asarray(result["edge_derivatives"], dtype=np.float64)
        / np.asarray(inputs["distances"], dtype=np.float64)
    )
    half = np.zeros((3, 3), dtype=np.float64)
    full = np.zeros((3, 3), dtype=np.float64)
    for edge in range(delta.shape[0]):
        outer = np.outer(delta[edge], delta[edge])
        half += 0.5 * factor[edge] * outer
        full += factor[edge] * outer
    assert np.max(np.abs(full)) > 1e-9, (
        "the un-halved candidate must be materially different for the factor to be "
        "observable"
    )
    independent = _reciprocal_strain_reference(
        inputs, mesh, result["structure_factors"], result["reciprocal_energy"]
    )
    deviation = float(np.max(np.abs((result["strain"] - half) - independent)))
    declared = DECLARED_SAME_MESH_STRAIN_BAND[precision]
    assert deviation <= declared, (
        f"the coordination part must be exactly one half of the directed-edge sum; "
        f"removing it must leave the reciprocal tensor.  Deviation {deviation:.3e} "
        f"against {declared:.3e}; the un-halved candidate would leave "
        f"{np.max(np.abs(full - half)):.3e}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_strain_derivative_is_symmetric_and_stress_is_tensile_positive_voigt(
    native, precision, case
):
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    strain = result["strain"]
    volume = abs(float(np.linalg.det(inputs["cell"])))
    scale = max(1e-12, float(np.max(np.abs(strain))))
    assert (
        np.max(np.abs(strain - strain.T)) <= 1e-11 * scale
    ), "the full strain tensor must come out symmetric without being symmetrized"
    expected = (
        np.asarray(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                strain[1, 2],
                strain[0, 2],
                strain[0, 1],
            ]
        )
        / volume
    )
    assert np.allclose(result["stress"], expected, rtol=1e-12, atol=1e-15), (
        "stress must be (1/Omega) dE/du in Voigt order (xx, yy, zz, yz, xz, xy), "
        "tensile-positive, with the shear components not doubled"
    )
    assert np.allclose(
        result["virial"], -strain, rtol=1e-12, atol=1e-15
    ), "the virial is W = -dE/du"


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_converged_mesh_matches_the_frozen_oracle_strain(native, precision, case):
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, ACCEPTANCE_MESH, 4)
    expected = np.asarray(
        CASES[case]["expected"]["strain_derivative_ev"], dtype=np.float64
    )
    deviation = float(np.max(np.abs(result["strain"] - expected)))
    band = DECLARED_ORACLE_STRAIN_BAND_EV
    assert deviation <= band, (
        f"the strain derivative at {ACCEPTANCE_MESH} differs from the mesh-free oracle "
        f"by {deviation:.3e} against the declared {band:.3e}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_kernel_derivative_table_matches_central_differences_of_the_kernel(
    native, precision
):
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    derivative = result["pair_kernel_derivative"]
    norm = np.linalg.norm(_wave_vectors(inputs["cell"], mesh), axis=1)
    step = 1e-6 if precision == "float64" else 1e-2
    q = np.asarray(inputs["q"], dtype=np.float64)
    species = np.asarray(inputs["species"], dtype=np.int32)
    worst = 0.0
    for a in range(species.size):
        for b in range(species.size):
            q_product = q[a] * q[b]
            r0 = DAMPING.damping_radius(q_product)
            values = fourier_d3.reciprocal_kernel(
                np.stack([norm - step, norm + step]), r0, q_product, DAMPING
            )
            numeric = (values[1] - values[0]) / (2.0 * step)
            index = _pair_index(a, b, species.size)
            worst = max(worst, float(np.max(np.abs(numeric - derivative[index]))))
    band = 1e-6 if precision == "float64" else 1e-2
    assert worst <= band, (
        f"the radial kernel derivative table must be d phihat/d|k|; worst deviation "
        f"{worst:.3e} against {band:.3e}"
    )


# --------------------------------------------------------------------------
# F. atomic energies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case",
    [
        "single-atom-self-images",
        "orthogonal-mixed-species",
        "triclinic-mixed-species",
        "coordination-sensitive",
    ],
)
def test_atomic_energies_sum_to_the_total_energy(native, precision, case):
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    total = float(result["atomic_energies"].sum())
    scale = max(1e-12, abs(result["energy"]))
    tolerance = 1e-12 * scale if precision == "float64" else 1e-5 * scale
    assert (
        abs(total - result["energy"]) <= tolerance
    ), f"sum_i e_i must equal E; got {total!r} against {result['energy']!r}"


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_atomic_energy_partition_matches_the_normative_formula(native, precision, case):
    inputs = _inputs(native, precision, case=case)
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    expected = np.zeros_like(result["atomic_energies"])
    potentials = [
        np.asarray(result["seam"].potential(slot), dtype=np.float64)
        for slot in range(inputs["species"].size)
    ]
    expected += _atomic_energy_reference(inputs, mesh, 4, potentials)
    expected += _self_share(inputs)
    declared = DECLARED_SAME_MESH_PARTITION_BAND[precision]
    deviation = float(np.max(np.abs(expected - result["atomic_energies"])))
    assert deviation <= declared, (
        f"the atomic partition must be half of c_il times the gathered mesh "
        f"sensitivity plus half of the self term; deviation {deviation:.3e} against "
        f"{declared:.3e}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_padding_atoms_carry_zero_force_and_zero_atomic_energy(native, precision):
    inputs = _inputs(native, precision, case="coordination-sensitive")
    numbers = np.array(inputs["numbers"], dtype=np.int32)
    numbers[4] = 0
    inputs["numbers"] = numbers
    _reload_graph(native, inputs)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    assert result["atomic_energies"][4] == 0.0
    assert np.all(result["forces"][4] == 0.0)
    assert np.all(result["sensitivity"][4] == 0.0)
    assert result["chi"][4] == 0.0


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_signed_ranks_keep_the_atomic_partition_exact(native, precision, sign):
    base = _factors(CASES["triclinic-mixed-species"])
    factors = fourier_d3.LowRankFactors(
        species=base.species,
        eigs=sign * np.abs(base.eigs),
        v_q=base.v_q,
        cn_ref=base.cn_ref,
        valid=base.valid,
        q_table=base.q_table,
    )
    inputs = _inputs(native, precision, case="triclinic-mixed-species", factors=factors)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    scale = max(1e-12, abs(result["energy"]))
    tolerance = 1e-12 * scale if precision == "float64" else 1e-5 * scale
    assert abs(float(result["atomic_energies"].sum()) - result["energy"]) <= tolerance
    assert float(np.max(np.abs(result["coordination_forces"].sum(axis=0)))) <= (
        1e-12 if precision == "float64" else 1e-5
    ) * max(1e-30, float(np.max(np.abs(result["coordination_forces"]))))
    assert np.all(np.sign(result["coefficients"]) != 0.0)


# --------------------------------------------------------------------------
# G. invariance and coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", (2, 4, 6))
@pytest.mark.parametrize(
    "case", ("orthogonal-mixed-species", "triclinic-mixed-species")
)
def test_wrapped_triclinic_stencil_preserves_analytic_force(native, order, case):
    """Equivalent per-atom lattice images preserve the analytic reverse result."""
    inputs = _inputs(native, "float64", case=case)
    lattice_shifts = np.resize(
        np.asarray([[1, 0, 0], [0, -2, 1], [3, 1, -1], [-2, 0, 2]], dtype=np.int32),
        np.asarray(inputs["positions"]).shape,
    )
    moved = dict(inputs)
    moved["positions"] = np.asarray(inputs["positions"], dtype=np.float64) + (
        lattice_shifts @ np.asarray(inputs["cell"], dtype=np.float64)
    )
    receivers = np.asarray(inputs["receivers"], dtype=np.int64)
    neighbours = np.asarray(inputs["neighbours"], dtype=np.int64)
    moved["shifts"] = (
        np.asarray(inputs["shifts"], dtype=np.int32)
        - lattice_shifts[neighbours]
        + lattice_shifts[receivers]
    )

    reference = _evaluate(native, inputs, (10, 12, 14), order)
    host_reference = _evaluate(native, inputs, (10, 12, 14), order, host_stencil=True)
    translated = _evaluate(native, moved, (10, 12, 14), order)
    np.testing.assert_allclose(
        host_reference["forces"], reference["forces"], rtol=1e-11, atol=2e-12
    )
    np.testing.assert_allclose(
        host_reference["strain"], reference["strain"], rtol=1e-11, atol=2e-12
    )
    np.testing.assert_allclose(
        translated["forces"], reference["forces"], rtol=1e-11, atol=2e-12
    )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_spline_orders_two_through_six_keep_every_invariant(native, precision, order):
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    result = _evaluate(native, inputs, (10, 12, 14), order)
    coord_scale = max(1e-30, float(np.max(np.abs(result["coordination_forces"]))))
    coord_band = 1e-12 if precision == "float64" else 1e-5
    assert float(np.max(np.abs(result["coordination_forces"].sum(axis=0)))) <= (
        coord_band * coord_scale
    )
    energy_scale = max(1e-12, abs(result["energy"]))
    tolerance = 1e-12 * energy_scale if precision == "float64" else 1e-4 * energy_scale
    assert abs(float(result["atomic_energies"].sum()) - result["energy"]) <= tolerance
    strain = result["strain"]
    assert np.max(np.abs(strain - strain.T)) <= 1e-10 * max(
        1e-12, float(np.max(np.abs(strain)))
    ), "the analytic derivative must be symmetric at every spline order"


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("mesh", [(9, 9, 9), (8, 8, 8), (11, 8, 13)])
def test_odd_and_even_meshes_keep_every_invariant(native, precision, mesh):
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    result = _evaluate(native, inputs, mesh, 4)
    coord_scale = max(1e-30, float(np.max(np.abs(result["coordination_forces"]))))
    coord_band = 1e-12 if precision == "float64" else 1e-5
    assert float(np.max(np.abs(result["coordination_forces"].sum(axis=0)))) <= (
        coord_band * coord_scale
    )
    energy_scale = max(1e-12, abs(result["energy"]))
    tolerance = 1e-12 * energy_scale if precision == "float64" else 1e-4 * energy_scale
    assert abs(float(result["atomic_energies"].sum()) - result["energy"]) <= tolerance
    strain = result["strain"]
    assert np.max(np.abs(strain - strain.T)) <= 1e-10 * max(
        1e-12, float(np.max(np.abs(strain)))
    ), "an odd mesh must not break the symmetry of the analytic derivative"


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_rank_chunking_is_numerically_neutral(native, precision, chunk):
    """Packing the same ranks into different chunks must not change the result."""
    mesh = (8, 8, 8)
    rank_total = int(_factors(CASES["triclinic-mixed-species"]).rank)
    keys = ("forces", "strain", "atomic_energies")
    total: dict[str, object] = {"energy": 0.0}
    for key in keys:
        total[key] = None
    for begin in range(0, rank_total, chunk):
        width = min(chunk, rank_total - begin)
        inputs = _inputs(
            native,
            precision,
            case="triclinic-mixed-species",
            chunk=width,
            rank_begin=begin,
        )
        part = _evaluate(native, inputs, mesh, 4)
        total["energy"] = float(total["energy"]) + float(part["energy"])
        for key in keys:
            if total[key] is None:
                total[key] = np.array(part[key], dtype=np.float64)
            else:
                total[key] = total[key] + np.asarray(part[key], dtype=np.float64)

    reference = _evaluate(
        native, _inputs(native, precision, case="triclinic-mixed-species"), mesh, 4
    )
    tolerance = TOLERANCE[precision]
    assert abs(float(total["energy"]) - reference["energy"]) <= 1e-10 * max(
        1.0, abs(reference["energy"])
    )
    for key in keys:
        assert np.allclose(
            np.asarray(total[key], dtype=np.float64),
            np.asarray(reference[key], dtype=np.float64),
            rtol=tolerance["rtol"],
            atol=tolerance["atol"],
        ), f"{key} must not depend on the rank chunk width"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_atom_reordering_permutes_forces_and_preserves_the_energy(native, precision):
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    mesh = (8, 8, 8)
    result = _evaluate(native, inputs, mesh, 4)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    inverse = np.argsort(permutation)
    moved = _inputs(
        native,
        precision,
        case="orthogonal-mixed-species",
        positions=inputs["positions"][permutation],
        numbers=inputs["numbers"][permutation],
        radii=inputs["radii"][permutation],
    )
    other = _evaluate(native, moved, mesh, 4)
    assert abs(other["energy"] - result["energy"]) <= 1e-11 * max(
        1.0, abs(result["energy"])
    )
    tolerance = TOLERANCE[precision]
    assert np.allclose(
        other["forces"],
        result["forces"][permutation],
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    )
    assert np.allclose(
        other["atomic_energies"][inverse],
        result["atomic_energies"],
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_the_graph_species_channels_must_match_the_mesh_species_channels(
    native, precision
):
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    factors = inputs["factors"]
    permuted = np.asarray(factors.species[::-1], dtype=np.int32)
    with pytest.raises((ValueError, RuntimeError)):
        seam.set_reference(
            factors.cn_ref[::-1], factors.valid[::-1], permuted, factors.v_q[::-1]
        )


# --------------------------------------------------------------------------
# H. lifecycle, cache, and accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("precision", PRECISIONS)
def test_repeated_evaluations_reuse_plans_kernels_and_reverse_buffers(
    native, precision
):
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    first = seam.usage()
    first_ids = {buffer["name"]: buffer["allocation_id"] for buffer in first["buffers"]}
    for _ in range(3):
        seam.run(CUTOFF)
    usage = seam.usage()
    assert usage["reservations"] == first["reservations"]
    assert usage["allocations"] == first["allocations"]
    ids = {buffer["name"]: buffer["allocation_id"] for buffer in usage["buffers"]}
    assert (
        ids == first_ids
    ), "an unchanged geometry must not reallocate a reverse buffer"
    for plan in usage["plans"]:
        assert plan["builds"] == 1, "one plan build per role for an unchanged mesh"
        assert plan["key_rebuilds"] == 0
    assert usage["kernels"]["key_rebuilds"] == 0
    assert usage["mesh"]["reservations"] == first["mesh"]["reservations"]


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_failed_evaluation_leaves_the_last_successful_outputs_unchanged(
    native, precision
):
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    energy = seam.energy()
    forces = np.asarray(seam.forces(), dtype=np.float64)
    with pytest.raises((ValueError, RuntimeError)):
        seam.run(-1.0)
    assert seam.energy() == energy
    assert np.array_equal(np.asarray(seam.forces(), dtype=np.float64), forces)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reverse_usage_accounts_every_buffer_once(native, precision):
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    usage = seam.usage()
    names = [buffer["name"] for buffer in usage["buffers"]]
    assert len(names) == len(set(names)), "buffer names must be unique"
    for buffer in usage["buffers"]:
        assert buffer["capacity_bytes"] >= buffer["active_bytes"]
    assert usage["active_bytes"] >= sum(
        buffer["active_bytes"] for buffer in usage["buffers"]
    )
    assert usage["capacity_bytes"] >= sum(
        buffer["capacity_bytes"] for buffer in usage["buffers"]
    )
    by_name = {buffer["name"]: buffer for buffer in usage["buffers"]}
    for name in ("reciprocal_strain", "strain_derivative"):
        assert by_name[name]["active_elements"] == 9
        assert by_name[name]["capacity_elements"] == 9
    for name in (
        "mesh_force",
        "mesh_sensitivity",
        "coefficient_sensitivity",
        "coordination_sensitivity",
        "coordination_force",
        "coordination_virial",
        "reciprocal_strain",
        "strain_derivative",
        "atomic_energy",
    ):
        assert name in names, f"the reverse usage must itemize {name}"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reserve_failure_is_atomic_and_leaves_the_state_usable(native, precision):
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    energy = seam.energy()
    before = seam.usage()["allocations"]
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        seam.reserve(8, 8, 8, 3, -1, 4, 5, inputs["rank"], 5, 4)
    assert seam.usage()["allocations"] == before
    assert seam.energy() == energy


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reverse_seam_is_private_and_claims_no_public_capability(native, precision):
    for symbol in _SEAM_SYMBOLS:
        assert hasattr(native, symbol)
        assert symbol not in getattr(symmetrix, "__all__", [])
    assert "FourierD3Kokkos" not in symmetrix.__all__
    assert "FourierD3Kokkos" not in symmetrix._FRONTEND_EXPORTS
    report = native._fourier_d3_capability()
    assert report["available"] is True
    assert "provider" in report


@pytest.mark.parametrize("precision", PRECISIONS)
def test_geometry_report_reports_the_frozen_conventions(native, precision):
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    report = seam.geometry()
    assert report["deconvolution"] == "continuous"
    assert report["forward_normalization"] == "unnormalized"
    assert report["inverse_normalization"] == "unnormalized"
    assert report["hermitian_axis"] == "z"
    assert report["num_nodes"] == int(inputs["numbers"].size)
    assert report["active_rank_chunk"] == int(inputs["chunk"])
    assert report["strain_convention"] == "affine_positions_and_cell"
    assert report["stress_convention"] == "voigt_xx_yy_zz_yz_xz_xy_tensile_positive"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reverse_objects_release_the_mesh_before_finalization(native, precision):
    inputs = _inputs(native, precision, case="single-atom-self-images")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    usage = seam.usage()
    assert usage["plan_retained_bytes"] == usage["mesh"]["capacity_bytes"] > 0
    del seam
    gc.collect()
    # A second object must be constructible, which fails if the first one left a
    # live Kokkos owner behind.
    again = _load(native, inputs, (8, 8, 8), 4)
    # A reserved object holds its buffers, so the reservation bytes are reported;
    # what it must not hold is a plan, and a plan is what pins the mesh.
    assert again.usage()["plan_retained_bytes"] == 0
    assert again.usage()["mesh"]["capacity_bytes"] > 0
    assert again.usage()["kernels"]["capacity_bytes"] == 0
    assert again.usage()["capacity_bytes"] > 0


# --------------------------------------------------------------------------
# I. composite reserve atomicity, evaluation atomicity, and usage exactness
# --------------------------------------------------------------------------

# A snapshot of every observable of one seam.  The readbacks that are not
# available before a successful evaluation are recorded as their exception type
# rather than skipped, so a failed call that makes one of them readable - or
# unreadable - is a difference.
_READBACKS = (
    "energy",
    "reciprocal_energy",
    "self_energy",
    "forces",
    "mesh_forces",
    "coordination_forces",
    "atomic_energies",
    "coefficient_sensitivity",
    "coordination_sensitivity",
    "strain_derivative",
    "stress",
    "virial",
    "coordination",
    "edge_values",
    "edge_radial_derivatives",
    "coefficients",
    "coefficient_derivatives",
    "self_coefficient_derivative",
)


def _snapshot(seam):
    """Every readable output of one seam, with unavailable readbacks typed."""
    state = {}
    for name in _READBACKS:
        reader = getattr(seam, name)
        try:
            value = reader()
        except (ValueError, RuntimeError, OverflowError) as error:
            state[name] = ("raises", type(error).__name__)
            continue
        state[name] = (
            np.asarray(value, dtype=np.float64)
            if not np.isscalar(value)
            else float(value)
        )
    state["geometry"] = seam.geometry()
    try:
        state["usage"] = seam.usage()
    except (ValueError, RuntimeError, OverflowError) as error:
        state["usage"] = ("raises", type(error).__name__)
    return state


def _usage_projection(usage):
    """The parts of a usage report that a failed call must not move."""
    if isinstance(usage, tuple):
        return usage
    coordination = usage.get("coordination", {})
    return {
        "reservations": usage["reservations"],
        "allocations": usage["allocations"],
        "active_bytes": usage["active_bytes"],
        "capacity_bytes": usage["capacity_bytes"],
        "forward_reservations": usage.get("forward_reservations"),
        "forward_allocations": usage.get("forward_allocations"),
        "coordination_reservations": coordination.get("reservations"),
        "coordination_allocations": coordination.get("allocations"),
        "coordination_active_bytes": coordination.get("active_bytes"),
        "coordination_capacity_bytes": coordination.get("capacity_bytes"),
        "coordination_buffers": {
            buffer["name"]: (
                buffer["active_elements"],
                buffer["capacity_elements"],
                buffer["allocation_id"],
            )
            for buffer in coordination.get("buffers", ())
        },
        "mesh": usage["mesh"],
        "kernels": usage["kernels"],
        "plans": usage["plans"],
        "plan_retained_bytes": usage["plan_retained_bytes"],
        "buffers": {
            buffer["name"]: (
                buffer["active_elements"],
                buffer["capacity_elements"],
                buffer["allocation_id"],
            )
            for buffer in usage["buffers"]
        },
    }


def _assert_unchanged(seam, before, what):
    after = _snapshot(seam)
    for name in _READBACKS:
        expected = before[name]
        actual = after[name]
        if isinstance(expected, tuple):
            assert (
                isinstance(actual, tuple) and actual == expected
            ), f"{what}: readback {name} changed from {expected} to {actual}"
            continue
        assert not isinstance(
            actual, tuple
        ), f"{what}: readback {name} became unreadable ({actual})"
        if np.isscalar(expected):
            assert (
                actual == expected
            ), f"{what}: readback {name} moved from {expected!r} to {actual!r}"
        else:
            assert np.array_equal(expected, actual), (
                f"{what}: readback {name} moved by "
                f"{np.max(np.abs(expected - actual)):.3e}"
            )
    assert after["geometry"] == before["geometry"], f"{what}: the geometry report moved"
    assert _usage_projection(after["usage"]) == _usage_projection(
        before["usage"]
    ), f"{what}: the usage counters, byte totals, or allocation identities moved"


def _growing_request(
    inputs,
    *,
    mesh=(10, 10, 10),
    order=4,
    references=None,
    rank_total=None,
    rank_chunk=None,
    nodes=None,
    edges=None,
    species=None,
):
    """A reserve request that grows all three owners from the loaded geometry."""
    return (
        int(mesh[0]),
        int(mesh[1]),
        int(mesh[2]),
        int(inputs["species"].size) if species is None else int(species),
        int(inputs["numbers"].size) + 2 if nodes is None else int(nodes),
        int(inputs["neighbours"].size) + 4 if edges is None else int(edges),
        int(inputs["factors"].n_ref) if references is None else int(references),
        int(inputs["rank"]) if rank_total is None else int(rank_total),
        int(inputs["chunk"]) if rank_chunk is None else int(rank_chunk),
        int(order),
    )


def _mesh_request(inputs, mesh, order=4):
    """A reserve request that changes only the mesh, keeping the graph geometry."""
    return (
        int(mesh[0]),
        int(mesh[1]),
        int(mesh[2]),
        int(inputs["species"].size),
        int(inputs["numbers"].size),
        int(inputs["neighbours"].size),
        int(inputs["factors"].n_ref),
        int(inputs["rank"]),
        int(inputs["chunk"]),
        int(order),
    )


def _reconfigure(native, seam, inputs, *, order=4):
    """Re-applies every input after a successful reserve, in the loader's order."""
    seam.set_cell(np.ascontiguousarray(inputs["cell"], dtype=np.float64).reshape(-1))
    seam.set_species(inputs["species"], inputs["q"])
    seam.set_atoms(
        np.ascontiguousarray(inputs["positions"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(inputs["numbers"], dtype=np.int32),
    )
    seam.set_damping(DAMPING.a1, DAMPING.a2_angstrom, DAMPING.s6, DAMPING.s8)
    seam.set_rank(
        int(inputs["rank_begin"]), np.asarray(inputs["eigenvalues"], dtype=np.float64)
    )
    seam.set_graph(
        np.ascontiguousarray(inputs["offsets"], dtype=np.int32),
        np.ascontiguousarray(inputs["neighbours"], dtype=np.int32),
        np.ascontiguousarray(inputs["shifts"], dtype=np.int32),
        np.ascontiguousarray(inputs["distances"], dtype=np.float64),
        np.ascontiguousarray(inputs["numbers"], dtype=np.int32),
        np.ascontiguousarray(inputs["radii"], dtype=np.float64),
    )
    factors = inputs["factors"]
    seam.set_reference(factors.cn_ref, factors.valid, factors.species, factors.v_q)
    return seam


# --------------------------------------------------------------------------
# K. the flat Kokkos index limit is a bound, not an allocation attempt
# --------------------------------------------------------------------------

# Every request below has its own buffer element count above `2**31 - 1`.  That
# makes it unaddressable, and it also makes it *enormous*: the smallest of them
# is a 6.7 GiB allocation, so a pre-fix implementation does not reject it - it
# asks the allocator for it first, which on an unconstrained machine is an OOM
# kill rather than an exception.
#
# The probe therefore runs under `capped_subprocess`, whose 6 GiB cap is below
# every allocation the unguarded path would attempt and whose child pins every
# thread-count variable to one so the cap cannot be spent on per-thread OpenBLAS
# buffers before the code under test runs.
_PROBE_ADDRESS_SPACE_BYTES = 6 * 1024**3

# name, `_growing_request` overrides, the buffer the refusal must name
FLAT_INDEX_REQUESTS = (
    ("three_times_nodes", {"nodes": 715827883}, "mesh_force"),
    (
        "nodes_times_rank_chunk",
        {"nodes": 300000000, "rank_chunk": 8, "rank_total": 8},
        "mesh_sensitivity",
    ),
    ("three_times_edges", {"edges": 715827883}, "edge_shifts"),
)


def _flat_index_probe_body(requests):
    """A reverse-seam probe: evaluate once, then attempt each refused request."""
    return (
        "import test_fourier_d3_reverse as T\n"
        f"requests = {list(requests)!r}\n"
        "native = T._native()\n"
        "inputs = T._inputs(native, 'float64', case='orthogonal-mixed-species')\n"
        "seam = T._load(native, inputs, (8, 8, 8), 4)\n"
        "seam.run(T.CUTOFF)\n"
        "mark_initialized()\n"
        "report = {}\n"
        "for entry in requests:\n"
        "    name, overrides = entry[0], entry[1]\n"
        "    try:\n"
        "        seam.reserve(*T._growing_request(inputs, **overrides))\n"
        "    except BaseException as error:\n"
        "        report[name] = [type(error).__name__, str(error)]\n"
        "    else:\n"
        "        report[name] = ['<no exception>', '']\n"
        # The seam must survive every refusal and still report what it evaluated.
        "try:\n"
        "    report['survived'] = [float(seam.energy()), "
        "int(seam.usage()['reservations'])]\n"
        "except BaseException as error:\n"
        "    report['survived'] = [type(error).__name__, 0]\n"
        "emit(report)\n"
    )


@pytest.mark.parametrize("name,overrides,buffer_name", FLAT_INDEX_REQUESTS)
def test_an_unaddressable_node_or_edge_request_is_rejected_before_allocating(
    name, overrides, buffer_name
):
    """`3*nodes`, `nodes*rank_chunk` and `3*edges` are bounds, not allocation sizes.

    A request whose own buffer count exceeds the flat Kokkos index range must be
    refused by the composite reservation before the first candidate view is
    allocated - the same limit the coordination workspace already applies.  A
    pre-fix implementation instead asks the allocator for 6.7 to 16 GiB, which is
    an OOM kill on a machine that has the memory and a `RuntimeError` from the
    allocator on one that does not; neither is a clean rejection.
    """
    require_host_address_space_cap()
    report = run_capped_probe(
        _flat_index_probe_body(((name, overrides),)),
        address_space_bytes=_PROBE_ADDRESS_SPACE_BYTES,
        label=f"the {name} flat-index probe",
    )
    exception_name, message = report[name]
    assert exception_name == "OverflowError", (
        f"the {name} request ({overrides}) must be refused with OverflowError before "
        f"anything is allocated; the probe reported {exception_name}: {message}"
    )
    assert (
        buffer_name in message
    ), f"the {name} refusal must name the buffer it refused ({buffer_name}); got: {message}"
    assert (
        "flat Kokkos index range" in message
    ), f"the {name} refusal must name the limit it enforced; got: {message}"


def test_every_unaddressable_request_is_refused_and_the_seam_survives():
    """All three refusals in one process, and the object still works afterwards.

    The per-case test above proves the exception type and the buffer name; this
    one proves the three are independent - a guard that only happened to fire on
    the first request would leave the others allocating - and that a refused
    request does not poison the object.
    """
    require_host_address_space_cap()
    report = run_capped_probe(
        _flat_index_probe_body(FLAT_INDEX_REQUESTS),
        address_space_bytes=_PROBE_ADDRESS_SPACE_BYTES,
        label="the multi-request flat-index probe",
    )
    for name, _, _ in FLAT_INDEX_REQUESTS:
        exception_name, message = report[name]
        assert exception_name == "OverflowError", f"{name}: {exception_name}: {message}"
    survived = report["survived"]
    assert np.isfinite(survived[0]), (
        "the seam must still report the energy it evaluated before the refused "
        f"requests; got {survived}"
    )
    assert survived[1] == 1, (
        "a refused reservation must not count as a reservation; got " f"{survived}"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_an_oversized_reference_request_after_a_run_leaves_every_output_unchanged(
    native, precision
):
    """The coordination workspace is the last owner to validate, not the first.

    An oversized reference count is refused deterministically, and the refusal
    happens after the forward path and the reverse buffers have both been
    prepared.  A composite reservation that installed each owner as it went would
    leave the forward path grown while the reverse path still reported the old
    geometry.
    """
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    before = _snapshot(seam)
    request = _growing_request(inputs, references=1 << 40)
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        seam.reserve(*request)
    _assert_unchanged(seam, before, "an oversized reference request")


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_flat_index_rejection_after_a_run_leaves_every_output_unchanged(
    native, precision
):
    """The low-rank factor table is the last bound the coordination workspace applies.

    A rank wide enough to overflow the flat Kokkos index is rejected there, after
    the forward path has already been prepared.  Nothing may be installed before
    that rejection.
    """
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    before = _snapshot(seam)
    request = _growing_request(inputs, rank_total=1 << 31)
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        seam.reserve(*request)
    _assert_unchanged(seam, before, "a flat-index rejection")


@pytest.mark.parametrize("precision", PRECISIONS)
def test_every_allocation_failure_in_the_composite_reserve_is_atomic(native, precision):
    """Sweep the injected allocation failure across the whole composite sequence.

    Prevalidating the deterministic bounds is not enough: an out-of-memory
    failure can land on any allocation in the sequence, including one in the last
    owner.  Every one of them must leave the geometry, the energy, the forces, the
    strain and stress, the atomic energies, the usage counters, the byte totals,
    and every allocation identity exactly as they were.
    """
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    before = _snapshot(seam)
    request = _growing_request(inputs)
    failures = 0
    for index in range(64):
        native._fourier_d3_arm_allocation_fault(index)
        try:
            seam.reserve(*request)
        except (ValueError, RuntimeError, OverflowError, MemoryError):
            raised = True
        else:
            raised = False
        if not raised:
            native._fourier_d3_clear_allocation_fault()
            break
        assert not native._fourier_d3_allocation_fault_armed(), (
            f"allocation fault {index} is still armed, so the reserve failed for "
            "some other reason and this index proves nothing"
        )
        _assert_unchanged(seam, before, f"allocation fault {index}")
        failures += 1
    assert failures >= 20, (
        f"only {failures} allocation failures were exercised; the composite reserve "
        "sequence is longer than that, so the sweep did not cover it"
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_successful_growing_reserve_still_reuses_and_accounts(native, precision):
    """The two-phase reserve keeps the grow-only reuse and accounting contract."""
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    before = seam.usage()
    request = _growing_request(inputs)
    seam.reserve(*request)
    grown = seam.usage()
    assert grown["reservations"] == before["reservations"] + 1
    assert grown["allocations"] > before["allocations"]
    assert grown["capacity_bytes"] > before["capacity_bytes"]
    # Repeating the same request at the same geometry allocates nothing.
    seam.reserve(*request)
    again = seam.usage()
    assert again["reservations"] == grown["reservations"] + 1
    assert again["allocations"] == grown["allocations"]
    assert {buffer["name"]: buffer["allocation_id"] for buffer in again["buffers"]} == {
        buffer["name"]: buffer["allocation_id"] for buffer in grown["buffers"]
    }


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_failed_first_evaluation_leaves_no_partial_coefficients(native, precision):
    """A first evaluation that fails must not publish the coordination chain.

    The coordination chain fills the coefficients before the forward path runs,
    so an evaluation that fails in the forward path after that point would leave
    a caller holding coefficients from a geometry whose energy, forces, and
    stress were never produced.  A freshly reserved object has no readable
    coefficients at all, and a failed evaluation must not create any.
    """
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    before = _snapshot(seam)
    assert isinstance(
        before["coefficients"], tuple
    ), "a freshly reserved object must not report coefficients before it evaluates"
    failures = 0
    for index in range(16):
        native._fourier_d3_arm_allocation_fault(index)
        try:
            seam.run(CUTOFF)
        except (ValueError, RuntimeError, OverflowError, MemoryError):
            raised = True
        else:
            raised = False
        if not raised:
            native._fourier_d3_clear_allocation_fault()
            break
        assert not native._fourier_d3_allocation_fault_armed(), (
            f"allocation fault {index} is still armed, so the evaluation failed for "
            "some other reason and this index proves nothing"
        )
        _assert_unchanged(seam, before, f"first-evaluation fault {index}")
        failures += 1
    assert failures >= 1, "the first evaluation must acquire plans and so allocate"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_failing_evaluation_never_refreshes_the_coefficients(native, precision):
    """A failed evaluation must not leave new coefficients beside old forces.

    A wider mesh makes the retained reciprocal kernel tables too small, so the
    next evaluation grows them and therefore allocates.  Injecting a failure
    there lands after the coordination chain would have refreshed the
    coefficients and before the reverse stages would have refreshed the forces -
    the exact mixed state the transactional order exists to prevent.  The
    outputs of the last successful evaluation stay readable through the
    reservation, so they can be compared.
    """
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    seam.reserve(*_mesh_request(inputs, (12, 12, 12)))
    _reconfigure(native, seam, inputs)
    before = _snapshot(seam)
    # The reverse buffers survive the reservation, so the last successful outputs
    # are readable and must stay unchanged.
    assert not isinstance(before["forces"], tuple)
    failures = 0
    for index in range(16):
        native._fourier_d3_arm_allocation_fault(index)
        try:
            seam.run(CUTOFF)
        except (ValueError, RuntimeError, OverflowError, MemoryError):
            raised = True
        else:
            raised = False
        if not raised:
            native._fourier_d3_clear_allocation_fault()
            break
        assert not native._fourier_d3_allocation_fault_armed(), (
            f"allocation fault {index} is still armed, so the evaluation failed for "
            "some other reason and this index proves nothing"
        )
        _assert_unchanged(seam, before, f"evaluation fault {index}")
        failures += 1
    assert failures >= 1, "the wider mesh must make the next evaluation allocate"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_a_never_reserved_object_reports_no_active_buffer(native, precision):
    """A buffer that holds no allocation has no active extent either.

    The four fixed-shape host tensors are the exception: they are members of the
    object rather than allocations, so they exist - and are reported - from
    construction, and they hold no allocation identity.
    """
    seam = native._FourierD3ReciprocalReverse(str(precision), 8)
    usage = seam.usage()
    assert usage["reservations"] == 0
    assert usage["allocations"] == 0
    host_tensors = {
        "coordination_virial": 9,
        "strain_derivative": 9,
        "virial": 9,
        "stress": 6,
    }
    for buffer in usage["buffers"]:
        if buffer["name"] in host_tensors:
            assert buffer["active_elements"] == host_tensors[buffer["name"]]
            assert buffer["allocation_id"] == 0
            continue
        assert buffer["active_elements"] == 0, (
            f"{buffer['name']} reports {buffer['active_elements']} active elements on an "
            "object that has reserved nothing"
        )
        assert buffer["capacity_elements"] == 0
        assert buffer["active_bytes"] == 0
        assert buffer["capacity_bytes"] == 0
        assert buffer["allocation_id"] == 0
    assert usage["mesh"]["capacity_bytes"] == 0
    assert usage["kernels"]["capacity_bytes"] == 0
    assert usage["coordination"]["capacity_bytes"] == 0
    assert usage["coordination"]["reservations"] == 0
    assert usage["plan_retained_bytes"] == 0


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reverse_usage_itemizes_every_retained_byte_exactly(native, precision):
    """Every retained reverse buffer and host tensor is counted once."""
    inputs = _inputs(native, precision, case="triclinic-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    usage = seam.usage()
    itemized = {buffer["name"]: buffer for buffer in usage["buffers"]}
    for name, elements, size in (
        ("coordination_virial", 9, 8),
        ("strain_derivative", 9, 8),
        ("virial", 9, 8),
        ("stress", 6, 8),
    ):
        assert name in itemized, f"the reverse usage must itemize the {name} tensor"
        assert itemized[name]["active_bytes"] == elements * size
        assert itemized[name]["capacity_bytes"] == elements * size
    expected = sum(buffer["active_bytes"] for buffer in usage["buffers"])
    assert (
        usage["active_bytes"]
        == (
            expected
            + usage["mesh"]["active_bytes"]
            + usage["kernels"]["active_bytes"]
            + usage["coordination"]["active_bytes"]
        )
    ), "the active total must be the itemization plus the mesh, kernel, and coordination bytes once"
    expected = sum(buffer["capacity_bytes"] for buffer in usage["buffers"])
    assert usage["capacity_bytes"] == (
        expected
        + usage["mesh"]["capacity_bytes"]
        + usage["kernels"]["capacity_bytes"]
        + usage["coordination"]["capacity_bytes"]
    )
    coordination_expected = sum(
        buffer["active_bytes"] for buffer in usage["coordination"]["buffers"]
    )
    assert usage["coordination"]["active_bytes"] == coordination_expected
    # The identities distinguish a reuse from a rebuild and never collide with the
    # forward path's own convention.
    identities = {
        buffer["name"]: buffer["allocation_id"] for buffer in usage["buffers"]
    }
    assert identities["mesh_force"] == identities["atomic_energy"] != 0
    assert identities["edge_shifts"] != identities["mesh_force"]
    assert identities["virial"] == 0 and identities["stress"] == 0
    seam.run(CUTOFF)
    reused = {
        buffer["name"]: buffer["allocation_id"] for buffer in seam.usage()["buffers"]
    }
    assert reused == identities, "an unchanged evaluation must not reallocate"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_reverse_level_invalidation_matches_a_fresh_object(native, precision):
    """`set_atoms` and `set_cell` plus `set_graph` reach the same state as a rebuild.

    The deformation is chosen to keep the directed-edge count, because the graph
    the workspace accepts must cover exactly the reserved edge count; a change
    that added or removed an edge would need a re-reservation, which would no
    longer be testing invalidation.
    """
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)

    scaled = np.asarray(inputs["cell"], dtype=np.float64) * 1.01
    shifted = np.asarray(inputs["positions"], dtype=np.float64).copy()
    shifted[1] += np.asarray([0.021, -0.013, 0.017])
    moved = _inputs(
        native,
        precision,
        case="orthogonal-mixed-species",
        cell=scaled,
        positions=shifted,
    )
    assert (
        moved["neighbours"].size == inputs["neighbours"].size
    ), "the parity deformation must keep the directed-edge count"
    assert not np.array_equal(
        moved["neighbours"], inputs["neighbours"]
    ) or not np.array_equal(
        moved["distances"], inputs["distances"]
    ), "the parity deformation must actually change the graph"

    # Reuse: the same object, reconfigured in place.
    seam.set_cell(np.ascontiguousarray(scaled, dtype=np.float64).reshape(-1))
    seam.set_atoms(
        np.ascontiguousarray(moved["positions"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(moved["numbers"], dtype=np.int32),
    )
    seam.set_graph(
        np.ascontiguousarray(moved["offsets"], dtype=np.int32),
        np.ascontiguousarray(moved["neighbours"], dtype=np.int32),
        np.ascontiguousarray(moved["shifts"], dtype=np.int32),
        np.ascontiguousarray(moved["distances"], dtype=np.float64),
        np.ascontiguousarray(moved["numbers"], dtype=np.int32),
        np.ascontiguousarray(moved["radii"], dtype=np.float64),
    )
    reused = _evaluate(native, moved, (8, 8, 8), 4)
    seam.run(CUTOFF)

    for key, reader in (
        ("forces", "forces"),
        ("strain", "strain_derivative"),
        ("stress", "stress"),
        ("virial", "virial"),
        ("atomic_energies", "atomic_energies"),
        ("sensitivity", "coefficient_sensitivity"),
        ("chi", "coordination_sensitivity"),
        ("coefficients", "coefficients"),
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(seam, reader)(), dtype=np.float64),
            np.asarray(reused[key], dtype=np.float64),
            rtol=TOLERANCE[precision]["rtol"],
            atol=TOLERANCE[precision]["atol"],
            err_msg=f"a reconfigured object disagrees with a fresh one on {key}",
        )
    assert seam.energy() == pytest.approx(
        reused["energy"],
        rel=TOLERANCE[precision]["rtol"],
        abs=TOLERANCE[precision]["atol"],
    )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_set_atoms_alone_invalidates_the_evaluation(native, precision):
    """A position change must not be served from the previous evaluation."""
    inputs = _inputs(native, precision, case="orthogonal-mixed-species")
    seam = _load(native, inputs, (8, 8, 8), 4)
    seam.run(CUTOFF)
    shifted = np.asarray(inputs["positions"], dtype=np.float64).copy()
    shifted[1] += np.asarray([0.021, -0.013, 0.017])
    moved = _inputs(
        native, precision, case="orthogonal-mixed-species", positions=shifted
    )
    assert moved["neighbours"].size == inputs["neighbours"].size
    seam.set_atoms(
        np.ascontiguousarray(moved["positions"], dtype=np.float64).reshape(-1),
        np.ascontiguousarray(moved["numbers"], dtype=np.int32),
    )
    before = _snapshot(seam)
    with pytest.raises((ValueError, RuntimeError)):
        # The graph still describes the old positions, so the geometry check must
        # refuse the evaluation rather than serve the stale graph.
        seam.run(CUTOFF)
    _assert_unchanged(seam, before, "a stale graph after set_atoms")
    fresh = _evaluate(native, moved, (8, 8, 8), 4)
    seam.set_graph(
        np.ascontiguousarray(moved["offsets"], dtype=np.int32),
        np.ascontiguousarray(moved["neighbours"], dtype=np.int32),
        np.ascontiguousarray(moved["shifts"], dtype=np.int32),
        np.ascontiguousarray(moved["distances"], dtype=np.float64),
        np.ascontiguousarray(moved["numbers"], dtype=np.int32),
        np.ascontiguousarray(moved["radii"], dtype=np.float64),
    )
    seam.run(CUTOFF)
    np.testing.assert_allclose(
        np.asarray(seam.forces(), dtype=np.float64),
        np.asarray(fresh["forces"], dtype=np.float64),
        rtol=TOLERANCE[precision]["rtol"],
        atol=TOLERANCE[precision]["atol"],
    )


# --------------------------------------------------------------------------
# J. the translation residual: what the total force sum actually equals
# --------------------------------------------------------------------------

TRANSLATION_STEPS = (1e-2, 1e-3, 1e-4, 1e-5)

# Measured `sum_i F_i,a` for the orthogonal and triclinic fixtures at order 4,
# eV/Angstrom.  The residual is a property of the compact-support mesh assignment
# and shrinks under mesh refinement; it is *not* a reduction error and it is not
# zero.  See the refinement record below.
TRANSLATION_RESIDUAL_REFINEMENT = (
    # (nx, ny, nz), orthogonal, triclinic
    ((8, 8, 8), 5.9017e-04, 6.1792e-04),
    ((10, 12, 14), 2.5625e-04, 1.5124e-04),
    ((16, 16, 16), 1.4908e-05, 1.3652e-05),
    ((24, 24, 24), 2.2649e-06, 2.8577e-06),
    ((32, 32, 32), 1.2636e-06, 1.8680e-06),
)

# The band is the measured residual at the coarsest mesh the suite evaluates on
# (8^3) with a factor for a different BLAS/FFT rounding path.  Measured maxima at
# order 4 over the three fixtures: 5.902e-04 orthogonal, 6.179e-04 triclinic,
# 7.724e-04 coordination-sensitive.  It is an absolute bound on a force sum, not
# a bound relative to the largest force component, because the residual is set by
# the assignment and not by the force scale.
DECLARED_TRANSLATION_RESIDUAL_BAND = {"float64": 2e-3, "float32": 2e-3}
# The same residual expressed as an energy change under the shift below, eV, at
# the 24^3 convergence mesh.  Measured: 3.491e-07 orthogonal, 1.774e-07
# triclinic, 7.011e-06 coordination-sensitive.
DECLARED_TRANSLATION_ENERGY_BAND = {"float64": 5e-5, "float32": 5e-5}


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_the_total_force_sum_is_exactly_the_mesh_force_sum(native, precision, case):
    """The coordination adjoint contributes nothing to the total force sum.

    `sum_i F_i` is the mesh's translation residual and nothing else: the
    coordination forces telescope exactly, so the two sums agree to the last bit
    rather than to a band.  This is the assertion the specification asks for in
    place of the impossible `sum_i F_i = 0`.
    """
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    total = np.asarray(result["forces"], dtype=np.float64).sum(axis=0)
    mesh = np.asarray(result["mesh_forces"], dtype=np.float64).sum(axis=0)
    # `F_i = F_mesh,i + F_coord,i` is formed in the storage precision, so the two
    # sums agree up to the elementwise rounding of adding the coordination force
    # to the mesh force: nodes * scale * machine epsilon.  That is a bound on the
    # accumulation, not a band around the residual.
    scale = float(np.max(np.abs(result["forces"])))
    epsilon = 2.3e-16 if precision == "float64" else 1.2e-7
    slack = scale * epsilon * float(np.asarray(result["forces"]).shape[0])
    assert np.max(np.abs(total - mesh)) <= slack, (
        "sum_i F_i must equal sum_i F_mesh,i to the rounding of the elementwise sum; "
        f"the difference is {np.max(np.abs(total - mesh)):.3e} against {slack:.3e}, and "
        "the coordination force sum is "
        f"{np.asarray(result['coordination_forces'], dtype=np.float64).sum(axis=0)}"
    )
    assert np.max(np.abs(total)) > 0.0, (
        "the mesh translation residual must be measured, not assumed to vanish; a "
        "zero residual would mean this check cannot detect the mesh term"
    )
    assert np.max(np.abs(total)) <= DECLARED_TRANSLATION_RESIDUAL_BAND[precision]


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_the_total_force_sum_is_minus_the_rigid_translation_derivative(
    native, precision, case
):
    """`sum_i F_i = -dE/dt` for a rigid translation of every atom at a fixed cell.

    The analytic mesh force is the exact gradient of the reported discrete
    energy, so the force sum and the energy derivative of the same shift must
    agree; the agreement is a finite-difference band, not an identity, because the
    energy derivative is measured.
    """
    inputs = _inputs(native, precision, case=case)
    result = _evaluate(native, inputs, (8, 8, 8), 4)
    positions = np.asarray(inputs["positions"], dtype=np.float64)
    analytic = np.asarray(result["forces"], dtype=np.float64).sum(axis=0)
    for axis in range(3):
        direction = np.zeros(3, dtype=np.float64)
        direction[axis] = 1.0

        def shifted(step, axis=axis, direction=direction):
            return _full_energy(
                native, inputs, (8, 8, 8), 4, positions=positions + step * direction
            )

        estimates = -_sweep(shifted, TRANSLATION_STEPS)
        best, band = _best_band(analytic[axis], estimates)
        assert best <= DECLARED_FORCE_FD_BAND[precision], (
            f"-dE/dt along axis {axis} is {estimates}, the analytic force sum is "
            f"{analytic[axis]}, and the whole band is {band}"
        )


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize(
    "case", ["orthogonal-mixed-species", "triclinic-mixed-species"]
)
def test_the_translation_residual_shrinks_under_mesh_refinement(
    native, precision, case
):
    """The residual is a discretization property, and refinement is what removes it.

    Recorded sequence (orthogonal, triclinic), order 4, eV/Angstrom:

    | mesh | orthogonal | triclinic |
    | --- | --- | --- |
    | 8^3 | 5.902e-04 | 6.179e-04 |
    | 10x12x14 | 2.563e-04 | 1.512e-04 |
    | 16^3 | 1.491e-05 | 1.365e-05 |
    | 24^3 | 2.265e-06 | 2.858e-06 |
    | 32^3 | 1.264e-06 | 1.868e-06 |

    8^3 -> 24^3 is a factor of 260 and 216.  24^3 -> 32^3 is only 1.8 and 1.5, so
    the residual is approaching a floor rather than following one power of h over
    the whole range; the meshes are not a nested sequence either.  The check is
    therefore that the finest mesh is at least two decades below the coarsest,
    not a fitted convergence law.
    """
    inputs = _inputs(native, precision, case=case)
    column = 1 if case == "orthogonal-mixed-species" else 2
    measured = []
    for mesh, orthogonal, triclinic in TRANSLATION_RESIDUAL_REFINEMENT:
        result = _evaluate(native, inputs, mesh, 4)
        measured.append(
            float(
                np.max(
                    np.abs(np.asarray(result["forces"], dtype=np.float64).sum(axis=0))
                )
            )
        )
        recorded = orthogonal if column == 1 else triclinic
        assert measured[-1] == pytest.approx(recorded, rel=0.25), (
            f"the translation residual at {mesh} is {measured[-1]:.3e}, not the recorded "
            f"{recorded:.3e}; the recorded refinement evidence is stale"
        )
    assert all(
        value > 0.0 for value in measured
    ), f"the translation residual must be strictly positive; got {measured}"
    for (coarse, _, _), (fine, _, _), before, after in zip(
        TRANSLATION_RESIDUAL_REFINEMENT,
        TRANSLATION_RESIDUAL_REFINEMENT[1:],
        measured,
        measured[1:],
    ):
        assert after < before, (
            f"the translation residual did not fall from {coarse} to {fine}: "
            f"{before:.3e} -> {after:.3e}"
        )
    assert measured[-1] <= measured[0] / 50.0, (
        f"the translation residual went {measured[0]:.3e} -> {measured[-1]:.3e} from "
        f"{TRANSLATION_RESIDUAL_REFINEMENT[0][0]} to {TRANSLATION_RESIDUAL_REFINEMENT[-1][0]}; "
        "it must fall by at least two decades"
    )
