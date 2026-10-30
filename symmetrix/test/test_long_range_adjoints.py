"""Reverse-mode adjoint parity of the long-range kernels: the FD gate.

The adjoint surface of the long-range kind group (the ``adjoint_*`` methods
of ``symmetrix.PolarLongRangeOps``) is verified by central finite
differences on the fixture geometries: every stage adjoint must reproduce
dL/d(input) of its forward stage for deterministic random seeds, and the
composed position gradient must express the shared-cache chain rule -- a
perturbation of positions flows through the feature and energy paths
together (they share the assemble step and the cos/sin tables), so their
table seeds must add before the geometry adjoint fires.

See ``docs/user/models.md#mace-polar-1`` for the model contract. The adjoints use
d cos(k.r)/dr = -k sin(k.r) and d sin(k.r)/dr = +k cos(k.r), so the
geometry adjoint reuses the cache's own masked tables and one transposed
GEMM; the assemble/convolve/project/energy adjoints are the transposed
GEMMs of the forward chain; the self-interaction, external-field, and
dipole-readout adjoints are the transposes of constant-linear and linear
operators; the Fukui adjoint is the global scatter of the forward's
broadcast gradient (the framework's first scatter-and-broadcast adjoint);
and the analytic corrections' position adjoints scatter their graph-level
aggregate seeds (charge, dipole, quadrupole, the slab z-dipole) back onto
every node of the graph.  All reductions accumulate in fp64.

Tolerance: fp64 central differences with step 1e-6 on the fixtures' O(1)
units -- truncation O(h^2) and roundoff O(eps/h) put the FD noise floor
near 1e-9 relative, so the gate holds adjoints at 1e-6 relative with an
absolute floor tied to the gradient scale (measured headroom reported by
the task-9 report; three orders of magnitude is the expectation).  The
linear stages (dipole readout, self terms, Coulomb convolve) agree far
tighter; the bound covers the transcendentals of the geometry cache.
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

# Central-difference step and the justified gate (see module docstring).
FD_STEP = 1e-6
RTOL = 1e-6
ATOL_GRADIENT_SCALE = 1e-7

# Deterministic seeds: one per (geometry, surface) so a failure names its
# case exactly the way the conformance battery's mirror seeds do.
SEEDS = {
    ("mol5", "geometry"): 11,
    ("pbc4", "geometry"): 12,
    ("mol5", "assemble"): 21,
    ("pbc4", "assemble"): 22,
    ("mol5", "coulomb"): 31,
    ("pbc4", "coulomb"): 32,
    ("mol5", "project"): 41,
    ("pbc4", "project"): 42,
    ("mol5", "energy"): 51,
    ("pbc4", "energy"): 52,
    ("mol5", "self"): 61,
    ("pbc4", "self"): 62,
    ("mol5", "external_field"): 71,
    ("pbc4", "external_field"): 72,
    ("mol5", "dipole"): 81,
    ("pbc4", "dipole"): 82,
    ("mol5", "composed"): 91,
    ("pbc4", "composed"): 92,
    ("mol5", "fukui"): 111,
    ("pbc4", "fukui"): 112,
    ("mol5", "correction_energy"): 121,
    ("pbc4", "correction_energy"): 122,
    ("mol5", "correction_features"): 131,
    ("pbc4", "correction_features"): 132,
}


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
    # Release the native owner before the session Kokkos finalizer: the
    # final test item's funcargs are dropped after the session fixture
    # stack has unwound (pytest clears item.funcargs once the teardown
    # hooks returned), so the ops object must not still own Kokkos
    # allocations then -- the same release-the-native-owners idiom the
    # conformance battery's ``instances`` fixture uses.
    loaded["ops"] = None
    del loaded
    gc.collect()


def build_cache(ops, positions, arrays):
    """The one shared cache per evaluation at the given positions."""
    result = ops.geometry_cache(
        positions, arrays["cell"], arrays["batch"].astype(np.int32)
    )
    return result, result["cache"]


def assert_fd_parity(grad, fd, context: str) -> None:
    """The adjoint must match the central-difference gradient.

    The absolute floor is tied to the gradient scale so components that are
    zero to FD noise (masked cross-graph entries, exact zeros) do not fail
    on the roundoff of the difference quotient.
    """

    grad = np.asarray(grad, dtype=np.float64)
    fd = np.asarray(fd, dtype=np.float64)
    assert grad.shape == fd.shape, f"{context}: shape {grad.shape} != {fd.shape}"
    scale = max(float(np.max(np.abs(fd))), 1e-30)
    np.testing.assert_allclose(
        grad,
        fd,
        rtol=RTOL,
        atol=ATOL_GRADIENT_SCALE * scale,
        err_msg=context,
    )


def fd_of_scalar(fn, x, step: float = FD_STEP) -> np.ndarray:
    """Central differences of a scalar-valued function on a 2-D input."""

    grad = np.zeros_like(x, dtype=np.float64)
    for flat in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus.reshape(-1)[flat] += step
        minus.reshape(-1)[flat] -= step
        grad.reshape(-1)[flat] = (fn(plus) - fn(minus)) / (2.0 * step)
    return grad


# ---------------------------------------------------------------------------
# Geometry-cache position adjoint
# ---------------------------------------------------------------------------


def test_adjoint_geometry_matches_finite_differences(fixture):
    """dL/dr of L = sum(Gc * cos + Gs * sin) is -k*sin / +k*cos per table.

    The adjoint reuses the cache's own masked tables: cross-graph table
    entries are zero, so their (formally nonzero) seeds contribute nothing
    to dL/dr -- the FD gate checks exactly that masking.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "geometry")])
    n_k = int(arrays["k_vectors"].shape[0])
    n_nodes = arrays["node_positions"].shape[0]
    grad_cos = rng.normal(size=(n_k, n_nodes))
    grad_sin = rng.normal(size=(n_k, n_nodes))
    batch = arrays["batch"].astype(np.int32)

    def loss(positions):
        build = ops.geometry_cache(positions, arrays["cell"], batch)
        return float(
            np.sum(grad_cos * build["cos_table"])
            + np.sum(grad_sin * build["sin_table"])
        )

    base = np.array(arrays["node_positions"], dtype=np.float64)
    _, cache = build_cache(ops, base, arrays)
    grad = ops.adjoint_geometry(cache, grad_cos, grad_sin)
    assert_fd_parity(grad, fd_of_scalar(loss, base, FD_STEP), "adjoint_geometry")


# ---------------------------------------------------------------------------
# Assemble adjoint: seeds of rho(k) -> table seeds and dL/d(multipoles)
# ---------------------------------------------------------------------------


def test_adjoint_assemble_grad_multipoles_matches_finite_differences(fixture):
    """dL/d(multipoles) of the two thin GEMMs is the transposed GEMMs."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "assemble")])
    n_k = int(arrays["k_vectors"].shape[0])
    grad_density = rng.normal(size=(n_k, 2))
    _, cache = build_cache(ops, arrays["node_positions"], arrays)
    source = np.array(arrays["source_feats_alpha"], dtype=np.float64)

    def loss(multipoles):
        density = ops.assemble_fourier(cache, multipoles)
        return float(np.sum(grad_density * density))

    grad = ops.adjoint_assemble(cache, grad_density, source)["grad_multipoles"]
    assert_fd_parity(
        grad, fd_of_scalar(loss, source, FD_STEP), "adjoint_assemble/multipoles"
    )


def test_adjoint_assemble_position_grad_matches_finite_differences(fixture):
    """The table seeds of the assemble adjoint chain into dL/dr."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "assemble")])
    n_k = int(arrays["k_vectors"].shape[0])
    grad_density = rng.normal(size=(n_k, 2))
    source = np.array(arrays["source_feats_alpha"], dtype=np.float64)

    def loss(positions):
        _, cache = build_cache(ops, positions, arrays)
        density = ops.assemble_fourier(cache, source)
        return float(np.sum(grad_density * density))

    base = np.array(arrays["node_positions"], dtype=np.float64)
    _, cache = build_cache(ops, base, arrays)
    back = ops.adjoint_assemble(cache, grad_density, source)
    grad = ops.adjoint_geometry(cache, back["grad_cos_table"], back["grad_sin_table"])
    assert_fd_parity(
        grad, fd_of_scalar(loss, base, FD_STEP), "adjoint_assemble/positions"
    )


# ---------------------------------------------------------------------------
# Coulomb convolve adjoint (elementwise, k=0 zeroed exactly)
# ---------------------------------------------------------------------------


def test_adjoint_coulomb_matches_finite_differences(fixture):
    """dL/d(rho) of V(k) = rho * FIELD_CONSTANT / k^2 is the same scale."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "coulomb")])
    n_k = int(arrays["k_vectors"].shape[0])
    grad_potential = rng.normal(size=(n_k, 2))
    _, cache = build_cache(ops, arrays["node_positions"], arrays)
    density = np.array(arrays["rho_alpha"], dtype=np.float64)

    def loss(rho):
        potential = ops.apply_coulomb(cache, rho)
        return float(np.sum(grad_potential * potential))

    grad = ops.adjoint_coulomb(cache, grad_potential)
    assert_fd_parity(grad, fd_of_scalar(loss, density, FD_STEP), "adjoint_coulomb")


# ---------------------------------------------------------------------------
# Project adjoint: seeds of the features -> dL/d(potential) and table seeds
# ---------------------------------------------------------------------------


def test_adjoint_project_grad_potential_matches_finite_differences(fixture):
    """dL/d(potential) of the two transposed GEMMs is the forward shapes."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "project")])
    n_nodes, f_dim = arrays["features_pbc_alpha"].shape
    grad_features = rng.normal(size=(n_nodes, f_dim))
    _, cache = build_cache(ops, arrays["node_positions"], arrays)
    potential = np.array(arrays["potential_alpha"], dtype=np.float64)

    def loss(pot):
        features = ops.project_to_features(cache, pot)
        return float(np.sum(grad_features * features))

    grad = ops.adjoint_project(cache, grad_features, potential)["grad_potential"]
    assert_fd_parity(
        grad, fd_of_scalar(loss, potential, FD_STEP), "adjoint_project/potential"
    )


def test_adjoint_project_position_grad_matches_finite_differences(fixture):
    """The table seeds of the project adjoint chain into dL/dr."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "project")])
    n_nodes, f_dim = arrays["features_pbc_alpha"].shape
    grad_features = rng.normal(size=(n_nodes, f_dim))
    potential = np.array(arrays["potential_alpha"], dtype=np.float64)

    def loss(positions):
        _, cache = build_cache(ops, positions, arrays)
        features = ops.project_to_features(cache, potential)
        return float(np.sum(grad_features * features))

    base = np.array(arrays["node_positions"], dtype=np.float64)
    _, cache = build_cache(ops, base, arrays)
    back = ops.adjoint_project(cache, grad_features, potential)
    grad = ops.adjoint_geometry(cache, back["grad_cos_table"], back["grad_sin_table"])
    assert_fd_parity(
        grad, fd_of_scalar(loss, base, FD_STEP), "adjoint_project/positions"
    )


# ---------------------------------------------------------------------------
# Coulomb energy adjoint: scalar-per-graph seeds -> dL/d(charge density),
# table seeds for the position chain, self-interaction included
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_self_fields", [False, True], ids=["kspace", "self"])
def test_adjoint_coulomb_energy_grad_charge_matches_finite_differences(
    fixture, with_self_fields
):
    """dL/d(charge density) of the energy reduction, self fields included.

    The self-interaction subtraction is constant-linear in the charge
    density, so its adjoint contributes -0.5 * seed * self_fields.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "energy")])
    grad_energy = rng.normal(size=arrays["cell"].shape[0])
    _, cache = build_cache(ops, arrays["node_positions"], arrays)
    source = np.array(arrays["source_feats_alpha"], dtype=np.float64)
    self_fields = (
        np.array(arrays["energy_self_fields_alpha"], dtype=np.float64)
        if with_self_fields
        else None
    )

    def loss(charge):
        energy = ops.coulomb_energy(cache, charge, self_fields)
        return float(np.sum(grad_energy * energy))

    back = ops.adjoint_coulomb_energy(cache, grad_energy, source, self_fields)
    assert_fd_parity(
        back["grad_charge_density"],
        fd_of_scalar(loss, source, FD_STEP),
        f"adjoint_coulomb_energy/charge:self={with_self_fields}",
    )


def test_adjoint_coulomb_energy_position_grad_matches_finite_differences(fixture):
    """The energy path's table seeds chain into dL/dr."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "energy")])
    grad_energy = rng.normal(size=arrays["cell"].shape[0])
    source = np.array(arrays["source_feats_alpha"], dtype=np.float64)
    self_fields = np.array(arrays["energy_self_fields_alpha"], dtype=np.float64)

    def loss(positions):
        _, cache = build_cache(ops, positions, arrays)
        energy = ops.coulomb_energy(cache, source, self_fields)
        return float(np.sum(grad_energy * energy))

    base = np.array(arrays["node_positions"], dtype=np.float64)
    _, cache = build_cache(ops, base, arrays)
    back = ops.adjoint_coulomb_energy(cache, grad_energy, source, self_fields)
    grad = ops.adjoint_geometry(cache, back["grad_cos_table"], back["grad_sin_table"])
    assert_fd_parity(
        grad, fd_of_scalar(loss, base, FD_STEP), "adjoint_coulomb_energy/positions"
    )


# ---------------------------------------------------------------------------
# Self-interaction adjoints (constant-linear: transpose of gather-and-scale)
# ---------------------------------------------------------------------------


def test_adjoint_self_terms_match_finite_differences(fixture):
    """dL/d(multipoles) of the gathers-and-scales is the scatter transpose."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "self")])
    source = np.array(arrays["source_feats_alpha"], dtype=np.float64)
    n_nodes, f_dim = arrays["features_self_interaction_alpha"].shape

    grad_features = rng.normal(size=(n_nodes, f_dim))

    def feature_loss(multipoles):
        return float(np.sum(grad_features * ops.feature_self_terms(multipoles)))

    grad = ops.adjoint_feature_self_terms(grad_features)
    assert_fd_parity(
        grad, fd_of_scalar(feature_loss, source, FD_STEP), "adjoint_feature_self_terms"
    )

    grad_fields = rng.normal(size=(n_nodes, source.shape[1]))

    def energy_loss(multipoles):
        return float(np.sum(grad_fields * ops.energy_self_fields(multipoles)))

    grad = ops.adjoint_energy_self_fields(grad_fields)
    assert_fd_parity(
        grad, fd_of_scalar(energy_loss, source, FD_STEP), "adjoint_energy_self_fields"
    )


# ---------------------------------------------------------------------------
# External-field adjoint: constant-matrix transpose plus the gauge term's
# position derivative (through the per-graph barycenter)
# ---------------------------------------------------------------------------


def test_adjoint_external_field_matches_finite_differences(fixture):
    """dL/dr (gauge term) and dL/d(external potential) of the projection.

    The gauge E.(r - barycenter) is linear in positions through every node
    of the graph at once (the barycenter scatter), so the adjoint carries
    the mean-subtracted broadcast -- the position derivative the constant
    matrix alone cannot express.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "external_field")])
    n_nodes, f_dim = arrays["features_pbc_alpha"].shape
    n_graphs = int(arrays["cell"].shape[0])
    external_potential = rng.normal(size=(n_graphs, 4))
    norms = rng.uniform(0.5, 2.0, size=f_dim)
    grad_features = rng.normal(size=(n_nodes, f_dim))
    batch = arrays["batch"].astype(np.int32)
    base = np.array(arrays["node_positions"], dtype=np.float64)

    def position_loss(positions):
        features = ops.external_field_features(
            external_potential, positions, batch, norms
        )
        return float(np.sum(grad_features * features))

    def potential_loss(potential):
        features = ops.external_field_features(potential, base, batch, norms)
        return float(np.sum(grad_features * features))

    back = ops.adjoint_external_field(
        grad_features, external_potential, base, batch, norms
    )
    assert_fd_parity(
        back["grad_positions"],
        fd_of_scalar(position_loss, base, FD_STEP),
        "adjoint_external_field/positions",
    )
    assert_fd_parity(
        back["grad_external_potential"],
        fd_of_scalar(potential_loss, external_potential, FD_STEP),
        "adjoint_external_field/potential",
    )
    # The norm contract is the forward's: a wrong-length vector is a caller
    # error, not a silent unit-norm fallback.
    with pytest.raises(
        ValueError, match=rf"one entry per feature: expected {f_dim}, got {f_dim - 1}"
    ):
        ops.adjoint_external_field(
            grad_features, external_potential, base, batch, np.ones(f_dim - 1)
        )


# ---------------------------------------------------------------------------
# Dipole-readout adjoint (linear in positions on the l=0 channel)
# ---------------------------------------------------------------------------


def test_adjoint_dipole_readout_matches_finite_differences(fixture):
    """dL/dr (q * GD per node) and dL/d(density) of the readout."""

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "dipole")])
    density = np.array(arrays["source_feats_total"], dtype=np.float64)
    n_graphs = int(arrays["cell"].shape[0])
    grad_charge = rng.normal(size=n_graphs)
    grad_dipole = rng.normal(size=(n_graphs, 3))
    batch = arrays["batch"].astype(np.int32)
    base = np.array(arrays["node_positions"], dtype=np.float64)

    def loss_of(positions, charge_density):
        result = ops.dipole_readout(charge_density, positions, batch)
        return float(
            np.sum(grad_charge * result["total_charge"])
            + np.sum(grad_dipole * result["total_dipole"])
        )

    back = ops.adjoint_dipole_readout(density, base, batch, grad_charge, grad_dipole)
    assert_fd_parity(
        back["grad_positions"],
        fd_of_scalar(lambda r: loss_of(r, density), base, FD_STEP),
        "adjoint_dipole_readout/positions",
    )
    assert_fd_parity(
        back["grad_charge_density"],
        fd_of_scalar(lambda q: loss_of(base, q), density, FD_STEP),
        "adjoint_dipole_readout/density",
    )


# ---------------------------------------------------------------------------
# Fukui-normalize adjoint (the global scatter-and-broadcast adjoint)
# ---------------------------------------------------------------------------


def test_adjoint_fukui_normalize_matches_finite_differences(fixture):
    """The scatter of the broadcast gradient, through the zero-guarded norm.

    The forward normalizes the per-node sources by their per-graph scatter and
    broadcasts the charge/spin target through the per-graph l=0 charge
    scatter; the adjoint therefore has two seeds per graph (the normalized
    broadcast weight, which also carries the norm's own derivative) and one
    graph-level charge/spin seed pair.  Nothing in the surface depends on
    geometry, so the FD gate here is the whole derivative.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "fukui")])
    n_nodes = arrays["node_positions"].shape[0]
    n_graphs = arrays["cell"].shape[0]
    batch = arrays["batch"].astype(np.int32)
    fukui_sources = rng.normal(size=(n_nodes, 2))
    charge_alpha = np.array(arrays["source_feats_alpha"][:, 0], dtype=np.float64)
    charge_beta = np.array(arrays["source_feats_beta"][:, 0], dtype=np.float64)
    total_charge = rng.normal(size=n_graphs)
    total_spin = rng.normal(size=n_graphs)
    grad_alpha = rng.normal(size=n_nodes)
    grad_beta = rng.normal(size=n_nodes)

    def loss(sources, alpha, beta, charge, spin):
        result = ops.fukui_normalize(sources, alpha, beta, charge, spin, batch)
        return float(
            np.sum(grad_alpha * result["charge_correction_alpha"])
            + np.sum(grad_beta * result["charge_correction_beta"])
        )

    back = ops.adjoint_fukui_normalize(
        fukui_sources,
        charge_alpha,
        charge_beta,
        total_charge,
        total_spin,
        batch,
        grad_alpha,
        grad_beta,
    )
    assert_fd_parity(
        back["grad_fukui_sources"],
        fd_of_scalar(
            lambda x: loss(x, charge_alpha, charge_beta, total_charge, total_spin),
            fukui_sources,
        ),
        "adjoint_fukui_normalize/sources",
    )
    assert_fd_parity(
        back["grad_charge_l0_alpha"],
        fd_of_scalar(
            lambda x: loss(fukui_sources, x, charge_beta, total_charge, total_spin),
            charge_alpha,
        ),
        "adjoint_fukui_normalize/charge_alpha",
    )
    assert_fd_parity(
        back["grad_charge_l0_beta"],
        fd_of_scalar(
            lambda x: loss(fukui_sources, charge_alpha, x, total_charge, total_spin),
            charge_beta,
        ),
        "adjoint_fukui_normalize/charge_beta",
    )
    assert_fd_parity(
        back["grad_total_charge"],
        fd_of_scalar(
            lambda x: loss(fukui_sources, charge_alpha, charge_beta, x, total_spin),
            total_charge,
        ),
        "adjoint_fukui_normalize/total_charge",
    )
    assert_fd_parity(
        back["grad_total_spin"],
        fd_of_scalar(
            lambda x: loss(fukui_sources, charge_alpha, charge_beta, total_charge, x),
            total_spin,
        ),
        "adjoint_fukui_normalize/total_spin",
    )


def test_adjoint_fukui_normalize_guard_branch_matches_the_closed_form(fixture):
    """The zero guard is differentiated as the branch the forward selects.

    The forward is genuinely discontinuous at a vanishing per-graph source
    sum: the norm flips between the sum and one, so perturbing a source by the
    FD step moves the norm off the guard and the quotient diverges (measured
    below, orders of magnitude past the gate).  No finite difference can
    therefore gate the adjoint there, and the convention the adjoint states is
    the standard one -- the guard is the selected branch, so the norm carries
    no derivative -- pinned here against the closed form of that branch.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "fukui")] + 1)
    n_nodes = arrays["node_positions"].shape[0]
    n_graphs = arrays["cell"].shape[0]
    batch = arrays["batch"].astype(np.int32)
    # A per-graph cancelling source pair: every graph's channel-0 sum is
    # exactly zero, so the forward's zero guard fires for it while the
    # channel-1 sources stay generic.
    sources = rng.normal(size=(n_nodes, 2))
    sources[:, 0] = 0.0
    for graph in range(n_graphs):
        members = np.flatnonzero(batch == graph)
        if len(members) < 2:
            pytest.skip("a single-node graph cannot cancel its source channel")
        sources[members[0], 0] = 0.75
        sources[members[1], 0] = -0.75
    charge_alpha = np.array(arrays["source_feats_alpha"][:, 0], dtype=np.float64)
    charge_beta = np.array(arrays["source_feats_beta"][:, 0], dtype=np.float64)
    total_charge = rng.normal(size=n_graphs)
    total_spin = rng.normal(size=n_graphs)
    grad_alpha = rng.normal(size=n_nodes)
    grad_beta = rng.normal(size=n_nodes)

    def loss(x):
        result = ops.fukui_normalize(
            x, charge_alpha, charge_beta, total_charge, total_spin, batch
        )
        return float(
            np.sum(grad_alpha * result["charge_correction_alpha"])
            + np.sum(grad_beta * result["charge_correction_beta"])
        )

    for graph in range(n_graphs):
        members = batch == graph
        assert float(sources[members, 0].sum()) == 0.0

    back = ops.adjoint_fukui_normalize(
        sources,
        charge_alpha,
        charge_beta,
        total_charge,
        total_spin,
        batch,
        grad_alpha,
        grad_beta,
    )

    # The branch's closed form, derived here from the forward's own inputs:
    # the guarded channel keeps the norm at one and loses the norm-derivative
    # term, the unguarded channel keeps both terms.
    norm = np.stack([sources[batch == g].sum(axis=0) for g in range(n_graphs)])
    norm = np.where(norm == 0.0, 1.0, norm)
    guarded = np.stack(
        [
            [
                sources[batch == g, 0].sum() == 0.0,
                sources[batch == g, 1].sum() == 0.0,
            ]
            for g in range(n_graphs)
        ]
    )
    charge_sum = np.stack(
        [
            [charge_alpha[batch == g].sum(), charge_beta[batch == g].sum()]
            for g in range(n_graphs)
        ]
    )
    target = np.stack(
        [
            [
                (total_charge[g] + (total_spin[g] - 1.0)) / 2.0 - charge_sum[g, 0],
                (total_charge[g] - (total_spin[g] - 1.0)) / 2.0 - charge_sum[g, 1],
            ]
            for g in range(n_graphs)
        ]
    )
    seeds = np.stack((grad_alpha, grad_beta), axis=1)
    broadcast = np.zeros((n_graphs, 2))
    for g in range(n_graphs):
        members = batch == g
        broadcast[g] = (seeds[members] * sources[members]).sum(axis=0) / norm[g]
    expected = np.zeros_like(sources)
    for channel in range(2):
        expected[:, channel] = (
            seeds[:, channel] * target[batch, channel] / norm[batch, channel]
        )
        active = ~guarded[batch, channel]
        expected[active, channel] -= (
            broadcast[batch[active], channel]
            * target[batch[active], channel]
            / norm[batch[active], channel]
        )
    assert_fd_parity(
        back["grad_fukui_sources"],
        expected,
        "adjoint_fukui_normalize/guarded_branch",
    )

    # The guard's discontinuity, measured: the FD quotient of the guarded
    # channel's sources is orders of magnitude past the adjoint's own value,
    # which is why this surface is pinned by its branch rather than by a
    # difference quotient.
    finite_difference = fd_of_scalar(loss, sources)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    assert float(np.max(np.abs(finite_difference - expected))) > 1e3 * scale


# ---------------------------------------------------------------------------
# Analytic-correction position adjoints (box/slab)
# ---------------------------------------------------------------------------

#: The correction-mode cases: the fixture's own pbc row, plus a slab row over
#: the same geometry.  The two fixtures cover the molecule and bulk classes,
#: so the slab branch (the z-dipole term of both corrections) gets its own
#: case here instead of staying unexercised.
CORRECTION_MODES = ("fixture", "slab")


def correction_pbc(arrays, mode: str) -> np.ndarray:
    """The per-graph pbc rows one correction-mode case evaluates with."""

    rows = np.array(arrays["pbc"], dtype=np.int32)
    if mode == "slab":
        return np.tile(np.array([1, 1, 0], dtype=np.int32), (rows.shape[0], 1))
    return rows


@pytest.mark.parametrize("correction_mode", CORRECTION_MODES)
def test_adjoint_correction_energy_matches_finite_differences(fixture, correction_mode):
    """dL/d(density) and dL/dr of the selected box/slab energy term.

    The correction is a graph aggregate of the density multipoles (charge,
    dipole, quadrupole) plus the slab's z-dipole, so its position adjoint
    scatters the graph-level seeds back onto every node of the graph -- the
    aggregate structure the FD gate measures directly.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "correction_energy")])
    n_graphs = arrays["cell"].shape[0]
    batch = arrays["batch"].astype(np.int32)
    pbc = correction_pbc(arrays, correction_mode)
    source = np.array(arrays["source_feats_total"], dtype=np.float64)
    base = np.array(arrays["node_positions"], dtype=np.float64)
    grad_energy = rng.normal(size=n_graphs)

    def loss(density, positions):
        result = ops.analytic_correction_energy(
            density, positions, arrays["cell"], batch, pbc
        )
        return float(np.sum(grad_energy * result["correction_energy"]))

    back = ops.adjoint_correction_energy(
        source, base, arrays["cell"], batch, pbc, grad_energy
    )
    assert_fd_parity(
        back["grad_charge_density"],
        fd_of_scalar(lambda x: loss(x, base), source),
        f"adjoint_correction_energy/density:{correction_mode}",
    )
    assert_fd_parity(
        back["grad_positions"],
        fd_of_scalar(lambda x: loss(source, x), base),
        f"adjoint_correction_energy/positions:{correction_mode}",
    )


@pytest.mark.parametrize("correction_mode", CORRECTION_MODES)
def test_adjoint_correction_features_matches_finite_differences(
    fixture, correction_mode
):
    """dL/d(density) and dL/dr of the corrective node fields.

    Both channels' fields are differentiated together (the surface is one
    call), and the position derivative covers the direct per-node terms as
    well as the graph aggregates the fields carry.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "correction_features")])
    n_nodes, f_dim = arrays["features_correction_alpha"].shape
    batch = arrays["batch"].astype(np.int32)
    pbc = correction_pbc(arrays, correction_mode)
    alpha = np.array(arrays["source_feats_alpha"], dtype=np.float64)
    beta = np.array(arrays["source_feats_beta"], dtype=np.float64)
    base = np.array(arrays["node_positions"], dtype=np.float64)
    grad_alpha = rng.normal(size=(n_nodes, f_dim))
    grad_beta = rng.normal(size=(n_nodes, f_dim))

    def loss(density_alpha, density_beta, positions):
        result = ops.analytic_correction_features(
            density_alpha, density_beta, positions, arrays["cell"], batch, pbc
        )
        return float(
            np.sum(grad_alpha * result["correction_fields_alpha"])
            + np.sum(grad_beta * result["correction_fields_beta"])
        )

    back = ops.adjoint_correction_features(
        alpha, beta, base, arrays["cell"], batch, pbc, grad_alpha, grad_beta
    )
    assert_fd_parity(
        back["grad_charge_density_alpha"],
        fd_of_scalar(lambda x: loss(x, beta, base), alpha),
        f"adjoint_correction_features/density_alpha:{correction_mode}",
    )
    assert_fd_parity(
        back["grad_charge_density_beta"],
        fd_of_scalar(lambda x: loss(alpha, x, base), beta),
        f"adjoint_correction_features/density_beta:{correction_mode}",
    )
    assert_fd_parity(
        back["grad_positions"],
        fd_of_scalar(lambda x: loss(alpha, beta, x), base),
        f"adjoint_correction_features/positions:{correction_mode}",
    )


# ---------------------------------------------------------------------------
# The two-graph spelling: the per-graph scatter and broadcast on data
# ---------------------------------------------------------------------------


def two_graph_geometry():
    """A molecule graph and a slab graph in one batch.

    The fixtures are single-graph, so the graph-level structure the new
    adjoints are built around is only observable here: two cells, two pbc
    classes (both correction modes selected in one call), and node membership
    split across the graphs.  The layout mirrors ``test_long_range_kernels``'s
    two-graph case.
    """

    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    cell = np.zeros((2, 3, 3))
    cell[0] = np.diag([6.0, 6.0, 6.0])
    cell[1] = np.diag([4.0, 4.0, 9.0])
    positions = np.array(
        [
            [0.1, 0.2, 0.3],
            [1.0, -0.5, 0.7],
            [2.0, 0.0, 0.1],
            [0.3, 0.4, 0.2],
            [0.9, 0.8, 3.0],
        ]
    )
    graph_of = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    graph_pbc = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.int32)
    return ops, cell, positions, graph_of, graph_pbc


@pytest.fixture
def two_graph():
    """The two-graph instance, released before the session Kokkos finalizer."""

    ops, cell, positions, graph_of, graph_pbc = two_graph_geometry()
    yield {
        "ops": ops,
        "cell": cell,
        "positions": positions,
        "graph_of": graph_of,
        "pbc": graph_pbc,
        "n_graphs": 2,
    }
    ops = None
    gc.collect()


def test_adjoint_fukui_normalize_two_graph_matches_finite_differences(two_graph):
    """The per-graph norms, targets, and seeds across two graphs.

    Each graph carries its own source norms, its own l=0 charge sums, and its
    own charge/spin target, so the adjoint's scatter and broadcast are
    graph-local: the FD gate here is what proves the graph index is honored
    rather than the whole batch treated as one graph.
    """

    ops = two_graph["ops"]
    positions = two_graph["positions"]
    graph_of = two_graph["graph_of"]
    n_nodes = positions.shape[0]
    rng = np.random.default_rng(SEEDS[("mol5", "fukui")] + 7)
    sources = rng.normal(size=(n_nodes, 2))
    charge_alpha = rng.normal(size=n_nodes)
    charge_beta = rng.normal(size=n_nodes)
    total_charge = rng.normal(size=two_graph["n_graphs"])
    total_spin = rng.normal(size=two_graph["n_graphs"])
    grad_alpha = rng.normal(size=n_nodes)
    grad_beta = rng.normal(size=n_nodes)

    def loss(src, alpha, beta, charge, spin):
        result = ops.fukui_normalize(src, alpha, beta, charge, spin, graph_of)
        return float(
            np.sum(grad_alpha * result["charge_correction_alpha"])
            + np.sum(grad_beta * result["charge_correction_beta"])
        )

    back = ops.adjoint_fukui_normalize(
        sources,
        charge_alpha,
        charge_beta,
        total_charge,
        total_spin,
        graph_of,
        grad_alpha,
        grad_beta,
    )
    assert_fd_parity(
        back["grad_fukui_sources"],
        fd_of_scalar(
            lambda x: loss(x, charge_alpha, charge_beta, total_charge, total_spin),
            sources,
        ),
        "two_graph/fukui/sources",
    )
    assert_fd_parity(
        back["grad_charge_l0_alpha"],
        fd_of_scalar(
            lambda x: loss(sources, x, charge_beta, total_charge, total_spin),
            charge_alpha,
        ),
        "two_graph/fukui/charge_alpha",
    )
    assert_fd_parity(
        back["grad_total_charge"],
        fd_of_scalar(
            lambda x: loss(sources, charge_alpha, charge_beta, x, total_spin),
            total_charge,
        ),
        "two_graph/fukui/total_charge",
    )
    assert_fd_parity(
        back["grad_total_spin"],
        fd_of_scalar(
            lambda x: loss(sources, charge_alpha, charge_beta, total_charge, x),
            total_spin,
        ),
        "two_graph/fukui/total_spin",
    )


def test_adjoint_corrections_two_graph_selects_both_modes(two_graph):
    """Both correction modes run in one call, and each graph's adjoint is local.

    Graph 0 is a molecule (the box term) and graph 1 a slab (the z-dipole
    term), so the selected output exercises both branches of both correction
    kinds at once; the FD gate then proves the per-graph aggregates and the
    position scatter stay inside their own graph.
    """

    ops = two_graph["ops"]
    positions = two_graph["positions"]
    cell = two_graph["cell"]
    graph_of = two_graph["graph_of"]
    pbc = two_graph["pbc"]
    n_nodes = positions.shape[0]
    f_dim = 8
    rng = np.random.default_rng(SEEDS[("mol5", "correction_energy")] + 7)
    density = rng.normal(size=(n_nodes, 4))
    grad_energy = rng.normal(size=two_graph["n_graphs"])

    def energy_loss(charge, place):
        result = ops.analytic_correction_energy(charge, place, cell, graph_of, pbc)
        return float(np.sum(grad_energy * result["correction_energy"]))

    selected = ops.analytic_correction_energy(density, positions, cell, graph_of, pbc)
    # Both branches selected: the molecule's box term and the slab's dipole
    # term, per graph.
    assert selected["correction_energy"][0] == pytest.approx(
        selected["molecule_correction"][0]
    )
    assert selected["correction_energy"][1] == pytest.approx(
        selected["slab_correction"][1]
    )
    back = ops.adjoint_correction_energy(
        density, positions, cell, graph_of, pbc, grad_energy
    )
    assert_fd_parity(
        back["grad_charge_density"],
        fd_of_scalar(lambda x: energy_loss(x, positions), density),
        "two_graph/correction_energy/density",
    )
    assert_fd_parity(
        back["grad_positions"],
        fd_of_scalar(lambda x: energy_loss(density, x), positions),
        "two_graph/correction_energy/positions",
    )

    grad_alpha = rng.normal(size=(n_nodes, f_dim))
    grad_beta = rng.normal(size=(n_nodes, f_dim))

    def feature_loss(alpha, beta, place):
        result = ops.analytic_correction_features(
            alpha, beta, place, cell, graph_of, pbc
        )
        return float(
            np.sum(grad_alpha * result["correction_fields_alpha"])
            + np.sum(grad_beta * result["correction_fields_beta"])
        )

    fields = ops.analytic_correction_features(
        density, density * 1.1, positions, cell, graph_of, pbc
    )
    assert np.any(fields["correction_fields_alpha"][graph_of == 0] != 0.0)
    assert np.any(fields["correction_fields_alpha"][graph_of == 1] != 0.0)
    back = ops.adjoint_correction_features(
        density, density * 1.1, positions, cell, graph_of, pbc, grad_alpha, grad_beta
    )
    assert_fd_parity(
        back["grad_charge_density_alpha"],
        fd_of_scalar(lambda x: feature_loss(x, density * 1.1, positions), density),
        "two_graph/correction_features/density_alpha",
    )
    assert_fd_parity(
        back["grad_positions"],
        fd_of_scalar(lambda x: feature_loss(density, density * 1.1, x), positions),
        "two_graph/correction_features/positions",
    )


# ---------------------------------------------------------------------------
# Composition: the shared-cache chain rule (features AND energy together)
# ---------------------------------------------------------------------------


def composed_position_gradient(ops, arrays, positions, grad_features, grad_energy):
    """The end-to-end position gradient through both consumers of the cache.

    The feature and energy paths share the assemble step and the cos/sin
    tables, so their table seeds (project adjoint + features-path assemble
    adjoint + energy adjoint) must ADD before the geometry adjoint fires --
    the chain rule through the shared cache.
    """

    batch = arrays["batch"].astype(np.int32)
    build = ops.geometry_cache(positions, arrays["cell"], batch)
    cache = build["cache"]
    source = np.array(arrays["source_feats_alpha"], dtype=np.float64)
    self_fields = np.array(arrays["energy_self_fields_alpha"], dtype=np.float64)

    potential = ops.apply_coulomb(cache, ops.assemble_fourier(cache, source))
    project_back = ops.adjoint_project(cache, grad_features, potential)
    grad_density = ops.adjoint_coulomb(cache, project_back["grad_potential"])
    assemble_back = ops.adjoint_assemble(cache, grad_density, source)
    energy_back = ops.adjoint_coulomb_energy(cache, grad_energy, source, self_fields)

    grad_cos = (
        project_back["grad_cos_table"]
        + assemble_back["grad_cos_table"]
        + energy_back["grad_cos_table"]
    )
    grad_sin = (
        project_back["grad_sin_table"]
        + assemble_back["grad_sin_table"]
        + energy_back["grad_sin_table"]
    )
    return ops.adjoint_geometry(cache, grad_cos, grad_sin)


@pytest.mark.parametrize(
    "seed_kind",
    ["features_only", "energy_only", "combined"],
)
def test_composed_position_gradient_matches_finite_differences(fixture, seed_kind):
    """L = <Gf, features> + <Ge, energy>: the shared-cache chain rule.

    features_only and energy_only isolate the two consumers of the shared
    cache; combined proves their adjoints compose -- the table seeds add,
    exactly what a perturbation of positions experiences downstream.
    """

    arrays = fixture["arrays"]
    ops = fixture["ops"]
    rng = np.random.default_rng(SEEDS[(fixture["key"], "composed")])
    n_nodes, f_dim = arrays["features_pbc_alpha"].shape
    grad_features = rng.normal(size=(n_nodes, f_dim))
    grad_energy = rng.normal(size=arrays["cell"].shape[0])
    if seed_kind == "features_only":
        grad_energy = np.zeros_like(grad_energy)
    elif seed_kind == "energy_only":
        grad_features = np.zeros_like(grad_features)
    source = np.array(arrays["source_feats_alpha"], dtype=np.float64)
    self_fields = np.array(arrays["energy_self_fields_alpha"], dtype=np.float64)
    batch = arrays["batch"].astype(np.int32)

    def loss(positions):
        build = ops.geometry_cache(positions, arrays["cell"], batch)
        cache = build["cache"]
        potential = ops.apply_coulomb(cache, ops.assemble_fourier(cache, source))
        features = ops.project_to_features(cache, potential)
        energy = ops.coulomb_energy(cache, source, self_fields)
        return float(np.sum(grad_features * features) + np.sum(grad_energy * energy))

    base = np.array(arrays["node_positions"], dtype=np.float64)
    grad = composed_position_gradient(ops, arrays, base, grad_features, grad_energy)
    assert_fd_parity(grad, fd_of_scalar(loss, base, FD_STEP), f"composed/{seed_kind}")
