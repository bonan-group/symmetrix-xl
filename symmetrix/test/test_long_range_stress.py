"""Finite-strain oracle for the long-range cell adjoint (the virial/stress).

The reciprocal-space virial has no analytic serial oracle, so its gate is
the OAM precedent (``test_nequip_oam_stress.py``): the finite-strain
derivative of this kernel's own forward, on the fixture geometries.  Both
fixtures carry a cell (``pbc4`` the bulk graph, ``mol5`` a molecule in its
vacuum box, where the analytic correction is active), so the battery runs
on both and the correction-kind volume terms are gated where their
fixture applies.

Conventions this battery pins (the design note's Adjoint set, the
"Reciprocal-space virial" entry):

* **Fixed-grid masking.**  The oracle's k-grid (``kspace.py``) enumerates a
  fixed integer triple superset and re-masks per graph by |k| <= k_c on
  every call, so grid membership is piecewise-constant in strain.  The
  analytic adjoint differentiates only the smooth |k| and volume
  dependence at fixed membership -- the classic Ewald convention -- and
  the finite-strain steps below are guarded (asserted) to keep every
  deformed cache at the base k-point count, so oracle and adjoint see the
  same grid.
* **Two derivative surfaces.**  The per-stage cell adjoints return the
  derivative at fixed *cartesian* positions; the affine stress (the
  physical virial) adds the position-scaling term
  ``sum_n (dE/dr_n)_b r_na`` through the shipped position adjoints --
  ``coulomb_stress`` composes both.  Under the affine deformation the
  phase ``k.r`` is exactly invariant (k transforms as a covector, the
  reciprocal cell deforms as ``2 pi inv(cell)^T``), so the table
  contributions cancel between the two terms -- which the agreement of
  both surfaces with their own finite-difference oracle demonstrates.
* **Strain spelling.**  ``cell @ (I + eps)`` with row-vector lattice
  rows, exactly the OAM oracle's deformation.  Single-entry strains
  measure the full (possibly asymmetric) tensor -- the l=1 multipoles are
  pinned cartesian data, so dE/deps need not be symmetric -- and the
  symmetric Voigt probe of the composed stress uses symmetric strains,
  the OAM test's convention.

Tolerance: fp64 central differences at step 1e-6, the adjoints battery's
gate (1e-6 relative, absolute floor tied to the gradient scale); the
measured agreement is three to four orders of magnitude tighter.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pytest

import symmetrix

if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()

DATA_DIR = Path(__file__).resolve().parent / "data" / "polar"
GEOMETRIES = ("mol5", "pbc4")

FD_STEP = 1e-6
RTOL = 1e-6
ATOL_GRADIENT_SCALE = 1e-7

# Deterministic seeds for the loss surfaces (the adjoints battery's idiom).
SEEDS = {
    ("mol5", "features_loss"): 141,
    ("pbc4", "features_loss"): 142,
    ("mol5", "composed_loss"): 151,
    ("pbc4", "composed_loss"): 152,
}

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def load_fixture(name: str) -> dict:
    manifest = json.loads((DATA_DIR / f"{name}_manifest.json").read_text())
    arrays = np.load(DATA_DIR / f"{name}_arrays.npz")
    return {"manifest": manifest, "arrays": arrays}


def fixture_ops(manifest: dict) -> symmetrix.PolarLongRangeOps:
    parameters = manifest["parameters"]
    return symmetrix.PolarLongRangeOps(
        kspace_cutoff=parameters["kspace_cutoff"],
        density_max_l=parameters["density_max_l"],
        density_smearing_width=parameters["density_smearing_width"],
        feature_max_l=parameters["feature_max_l"],
        feature_smearing_widths=list(parameters["feature_smearing_widths"]),
    )


@pytest.fixture(params=GEOMETRIES)
def fixture(request):
    loaded = load_fixture(request.param)
    loaded["ops"] = fixture_ops(loaded["manifest"])
    loaded["key"] = request.param
    yield loaded
    loaded["ops"] = None
    del loaded
    gc.collect()


def build_cache(ops, positions, arrays):
    result = ops.geometry_cache(
        positions, arrays["cell"], arrays["batch"].astype(np.int32)
    )
    return result, result["cache"]


def assert_fd_parity(grad, fd, context: str) -> None:
    grad = np.asarray(grad, dtype=np.float64)
    fd = np.asarray(fd, dtype=np.float64)
    assert grad.shape == fd.shape, f"{context}: shape {grad.shape} != {fd.shape}"
    scale = max(float(np.max(np.abs(fd))), 1e-30)
    np.testing.assert_allclose(
        grad, fd, rtol=RTOL, atol=ATOL_GRADIENT_SCALE * scale, err_msg=context
    )


def deformed(cell: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """The OAM oracle's deformation: lattice rows right-multiplied."""
    return cell @ (np.eye(3) + eps)


def guarded_energy(ops, arrays, base_n_k, positions, cell, density, self_fields=None):
    """The Coulomb energy at the given geometry, membership-guarded."""
    build = ops.geometry_cache(positions, cell, arrays["batch"].astype(np.int32))
    assert (
        build["k_vectors"].shape[0] == base_n_k
    ), "the strain step crossed the k cutoff: the fixed-membership convention"
    return float(ops.coulomb_energy(build["cache"], density, self_fields)[0])


def fd_strain_tensor(scalar_at, cell, step: float) -> np.ndarray:
    """Central differences of d(scalar)/d(eps) with single-entry strains.

    All nine components: the off-diagonal entries are measured one at a
    time (not symmetrized), so the full -- possibly asymmetric -- tensor
    is compared.
    """
    grad = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            eps = np.zeros((3, 3))
            eps[a, b] = 1.0
            grad[a, b] = (
                scalar_at(deformed(cell, step * eps))
                - scalar_at(deformed(cell, -step * eps))
            ) / (2.0 * step)
    return grad


def symmetric_strain(i: int, j: int) -> np.ndarray:
    eps = np.zeros((3, 3))
    eps[i, j] = eps[j, i] = 1.0 if i == j else 0.5
    return eps


# ---------------------------------------------------------------------------
# The energy-path cell adjoint at fixed cartesian positions
# ---------------------------------------------------------------------------


def test_energy_cell_adjoint_fixed_positions_matches_finite_strain(fixture):
    """dE/deps of the Coulomb energy at fixed r, through every cell term.

    The chain covers the k-vector magnitudes and directions (the radial
    and angular Fourier factors, the Coulomb kernel), the volume
    prefactors (the assemble 1/V and the energy V), and the phase tables;
    the composition converts the seeds through the geometry-cache cell
    adjoint.
    """
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    source = arrays["source_feats_total"]
    _, cache = build_cache(ops, positions, arrays)

    seeds = ops.adjoint_coulomb_energy_cell(cache, np.ones(1), source)
    strain = ops.adjoint_geometry_cell(
        cache,
        seeds["grad_k_vectors"],
        seeds["grad_volume_per_k"],
        seeds["grad_cos_table"],
        seeds["grad_sin_table"],
        positions,
    )[0]

    base_n_k = int(cache.n_k)
    fd = fd_strain_tensor(
        lambda deformed_cell: guarded_energy(
            ops, arrays, base_n_k, positions, deformed_cell, source
        ),
        cell,
        FD_STEP,
    )
    assert_fd_parity(strain, fd, f"{fixture['key']}/energy_cell_fixed_positions")


# ---------------------------------------------------------------------------
# The composed stress: the affine finite-strain oracle (the OAM pattern)
# ---------------------------------------------------------------------------


def test_coulomb_stress_matches_the_affine_finite_strain_oracle(fixture):
    """sigma = (1/Omega) dE/deps under the affine deformation.

    The OAM gate's pattern: deform the cell with symmetric strains, scale
    the positions with it, central-difference the forward energy, and
    divide by the volume (the self-interaction subtraction is
    cell-independent, so it carries no stress and no FD signal).
    ``coulomb_stress`` composes the cell adjoint
    with the position virial of the same loss; its symmetric part is the
    stress a symmetric probe measures (the l=1 multipoles are pinned
    cartesian data, so the full tensor may carry an antisymmetric part).

    This gate also pins the assembly's SCOPE: it differences the k-space
    energy, so the assembly is the k-space (Coulomb) stress.  The analytic
    corrections' volume and position terms are separate adjoints
    (``adjoint_correction_energy_cell``, ``adjoint_correction_features_cell``)
    that a family composes itself; folding them into ``coulomb_stress``
    without extending this oracle would fail here, which is the intent.
    """
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    source = arrays["source_feats_total"]
    _, cache = build_cache(ops, positions, arrays)

    stress = ops.coulomb_stress(cache, np.ones(1), None, source, positions)[0]
    assert np.abs(stress).max() > 1e-6, "the stress surface must be non-vacuous"
    symmetric = 0.5 * (stress + stress.T)
    voigt = np.array([symmetric[i, j] for i, j in VOIGT_PAIRS])

    base_n_k = int(cache.n_k)
    volume = float(np.abs(np.linalg.det(cell[0])))
    numerical = []
    for i, j in VOIGT_PAIRS:
        step = FD_STEP * symmetric_strain(i, j)
        e_plus = guarded_energy(
            ops,
            arrays,
            base_n_k,
            positions @ (np.eye(3) + step),
            deformed(cell, +step),
            source,
        )
        e_minus = guarded_energy(
            ops,
            arrays,
            base_n_k,
            positions @ (np.eye(3) - step),
            deformed(cell, -step),
            source,
        )
        numerical.append((e_plus - e_minus) / (2.0 * FD_STEP * volume))
    assert_fd_parity(voigt, np.array(numerical), f"{fixture['key']}/coulomb_stress")


# ---------------------------------------------------------------------------
# The features-path cell adjoint
# ---------------------------------------------------------------------------


def features_loss_cell_seeds(ops, cache, grad_features, source, density):
    """The features path's cell seeds through the shared chain."""
    potential = ops.apply_coulomb(cache, ops.assemble_fourier(cache, source))
    projected = ops.adjoint_project_cell(cache, grad_features, potential)
    convolved = ops.adjoint_coulomb_cell(cache, projected["grad_potential"], density)
    assembled = ops.adjoint_assemble_cell(cache, convolved["grad_density"], source)
    return (
        projected["grad_k_vectors"]
        + convolved["grad_k_vectors"]
        + assembled["grad_k_vectors"],
        assembled["grad_volume_per_k"],
        projected["grad_cos_table"] + assembled["grad_cos_table"],
        projected["grad_sin_table"] + assembled["grad_sin_table"],
    )


def test_features_path_cell_adjoint_fixed_positions_matches_finite_strain(fixture):
    """dL/deps of L = <Gf, features> at fixed r through the k-space chain."""
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    source = arrays["source_feats_alpha"]
    _, cache = build_cache(ops, positions, arrays)

    rng = np.random.default_rng(SEEDS[(fixture["key"], "features_loss")])
    grad_features = rng.normal(size=arrays["features_pbc_alpha"].shape)
    density = ops.assemble_fourier(cache, source)
    gk, gv, gc, gs = features_loss_cell_seeds(
        ops, cache, grad_features, source, density
    )
    strain = ops.adjoint_geometry_cell(cache, gk, gv, gc, gs, positions)[0]

    base_n_k = int(cache.n_k)

    def loss_at(deformed_cell):
        build = ops.geometry_cache(
            positions, deformed_cell, arrays["batch"].astype(np.int32)
        )
        assert build["k_vectors"].shape[0] == base_n_k
        potential = ops.apply_coulomb(
            build["cache"], ops.assemble_fourier(build["cache"], source)
        )
        features = ops.project_to_features(build["cache"], potential)
        return float(np.sum(grad_features * features))

    fd = fd_strain_tensor(loss_at, cell, FD_STEP)
    assert_fd_parity(strain, fd, f"{fixture['key']}/features_cell_fixed_positions")


# ---------------------------------------------------------------------------
# The composed stress over both paths (features AND energy)
# ---------------------------------------------------------------------------


def test_composed_stress_matches_finite_strain(fixture):
    """L = <Gf, features> + <Ge, energy>: both paths in one stress."""
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    source = arrays["source_feats_alpha"]
    _, cache = build_cache(ops, positions, arrays)

    rng = np.random.default_rng(SEEDS[(fixture["key"], "composed_loss")])
    grad_features = rng.normal(size=arrays["features_pbc_alpha"].shape)
    grad_energy = rng.normal(size=1)

    stress = ops.coulomb_stress(cache, grad_energy, grad_features, source, positions)[0]
    symmetric = 0.5 * (stress + stress.T)
    voigt = np.array([symmetric[i, j] for i, j in VOIGT_PAIRS])

    base_n_k = int(cache.n_k)

    def loss_at(positions_, cell_):
        build = ops.geometry_cache(positions_, cell_, arrays["batch"].astype(np.int32))
        assert build["k_vectors"].shape[0] == base_n_k
        potential = ops.apply_coulomb(
            build["cache"], ops.assemble_fourier(build["cache"], source)
        )
        features = ops.project_to_features(build["cache"], potential)
        energy = ops.coulomb_energy(build["cache"], source, None)
        return float(np.sum(grad_features * features) + grad_energy[0] * energy[0])

    volume = float(np.abs(np.linalg.det(cell[0])))
    numerical = []
    for i, j in VOIGT_PAIRS:
        step = FD_STEP * symmetric_strain(i, j)
        e_plus = loss_at(positions @ (np.eye(3) + step), deformed(cell, +step))
        e_minus = loss_at(positions @ (np.eye(3) - step), deformed(cell, -step))
        numerical.append((e_plus - e_minus) / (2.0 * FD_STEP * volume))
    assert_fd_parity(voigt, np.array(numerical), f"{fixture['key']}/composed_stress")


# ---------------------------------------------------------------------------
# The correction kinds' volume terms
# ---------------------------------------------------------------------------


def test_correction_energy_cell_adjoint_matches_finite_strain(fixture):
    """The volume factors of the selected analytic correction.

    On the molecule fixture the box term is selected (its V^0.3333 length
    and the 1/V dipole/quadrupole factors all move); on the bulk fixture
    no correction is selected and the derivative is exactly zero.  Fixed
    cartesian positions: the corrections' position coupling belongs to
    the position virial, not to the cell term.
    """
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    source = arrays["source_feats_total"]
    batch = arrays["batch"].astype(np.int32)
    pbc = arrays["pbc"].astype(np.int32)

    strain = ops.adjoint_correction_energy_cell(
        source, positions, cell, batch, pbc, np.ones(1)
    )[0]

    def selected_at(deformed_cell):
        result = ops.analytic_correction_energy(
            source, positions, deformed_cell, batch, pbc
        )
        return float(result["correction_energy"][0])

    fd = fd_strain_tensor(selected_at, cell, FD_STEP)
    if fixture["key"] == "pbc4":
        # Bulk graphs select no correction: the adjoint is exactly zero.
        assert np.all(strain == 0.0)
    assert_fd_parity(strain, fd, f"{fixture['key']}/correction_cell")


def test_correction_features_cell_adjoint_matches_finite_strain(fixture):
    """The corrective node fields' volume terms, per channel.

    The box term's V^0.333333 length and 1/V factors and the slab field's
    1/V all move with the cell at fixed cartesian positions; on the bulk
    fixture no correction is selected and the derivative is exactly zero.
    The slab branch gets its own case (the fixtures cover the molecule and
    bulk classes), with the same deformation convention.
    """
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    alpha = arrays["source_feats_alpha"]
    beta = arrays["source_feats_beta"]
    batch = arrays["batch"].astype(np.int32)
    n_graphs = int(arrays["cell"].shape[0])
    f_dim = arrays["features_correction_alpha"].shape[1]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "features_loss")])
    grad_alpha = rng.normal(size=(len(positions), f_dim))
    grad_beta = rng.normal(size=(len(positions), f_dim))

    for mode, pbc in (
        ("fixture", arrays["pbc"].astype(np.int32)),
        ("slab", np.tile(np.array([1, 1, 0], dtype=np.int32), (n_graphs, 1))),
    ):
        strain = ops.adjoint_correction_features_cell(
            alpha, beta, positions, cell, batch, pbc, grad_alpha, grad_beta
        )[0]

        def loss_at(deformed_cell):
            result = ops.analytic_correction_features(
                alpha, beta, positions, deformed_cell, batch, pbc
            )
            return float(
                np.sum(grad_alpha * result["correction_fields_alpha"])
                + np.sum(grad_beta * result["correction_fields_beta"])
            )

        fd = fd_strain_tensor(loss_at, cell, FD_STEP)
        if fixture["key"] == "pbc4" and mode == "fixture":
            assert np.all(strain == 0.0)
        assert_fd_parity(
            strain, fd, f"{fixture['key']}/correction_features_cell:{mode}"
        )


# ---------------------------------------------------------------------------
# The fixed-membership guard is not vacuous
# ---------------------------------------------------------------------------


def test_the_strain_step_keeps_the_k_grid_membership_stable(fixture):
    """The FD steps stay inside the piecewise-constant mask cell.

    The guard every deformed forward asserts: at the battery's step the
    deformed caches hold exactly the base k-point count, and the closest
    kept mode keeps its distance to the cutoff with orders of magnitude
    of headroom over the strain-induced |k|^2 drift.
    """
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    cell = arrays["cell"]
    batch = arrays["batch"].astype(np.int32)
    base = ops.geometry_cache(arrays["node_positions"], cell, batch)
    cutoff = float(fixture["manifest"]["parameters"]["kspace_cutoff"])
    margin = cutoff**2 - float(base["k_norm2"].max())
    drift = 2.0 * float(base["k_norm2"].max()) * FD_STEP
    assert margin > 1e3 * drift, (
        f"{fixture['key']}: the cutoff margin {margin:.3e} must dominate the "
        f"strain drift {drift:.3e} of the FD step"
    )
    for i, j in VOIGT_PAIRS:
        for sign in (1.0, -1.0):
            deformed_cache = ops.geometry_cache(
                arrays["node_positions"],
                deformed(cell, sign * FD_STEP * symmetric_strain(i, j)),
                batch,
            )
            assert deformed_cache["k_vectors"].shape[0] == base["k_vectors"].shape[0]
