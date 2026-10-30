"""Forward parity of the portable long-range kernels against the golden fixtures.

The portable (T1 host) kernels of the long-range kind group
(``libsymmetrix/source/long_range_kernels.hpp``) are the correctness backbone
of the MACE-POLAR-1 long-range layer, so their unit gate is forward parity
against the committed op-level golden fixtures
(``symmetrix/test/data/polar/{mol5,pbc4}_{manifest.json,arrays.npz}``): every
kernel, fed the fixture inputs for both geometries and both spin channels,
must reproduce the fixture outputs within fp64 tolerance.  The fixtures are
committed data produced by the pinned oracle (see
``benchmarks/polar_oracle_fixture_runner.py`` and
``docs/user/models.md#mace-polar-1``), so these tests run in
the plain repository environment with no oracle packages.

Kinds without a fixture array (the Fukui equilibration and the dipole
readout live in the model orchestration, not in ``graph_longrange``) are
pinned against a NumPy mirror of the upstream formula instead, and a
two-graph synthetic case exercises the multi-graph k-grid grouping and the
per-graph correction-mode selection that the single-graph fixtures cannot.

Tolerance: fp64 exact-ish -- the oracle fixtures were captured in fp64 and
every stage is a deterministic function of them, so 1e-12 relative (with a
1e-12 absolute floor for exact zeros such as masked k=0 modes and the bulk
correction) leaves two orders of magnitude of headroom over the measured
~1e-14 reassociation noise of the GEMM chain.
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
CHANNELS = ("alpha", "beta")
ENERGY_LABELS = ("alpha", "beta", "total")

# fp64 parity: relative 1e-12 with an absolute floor covering exact zeros.
RTOL = 1e-12
ATOL = 1e-12


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


def assert_parity(got: np.ndarray, want: np.ndarray, context: str) -> None:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, f"{context}: shape {got.shape} != {want.shape}"
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL, err_msg=context)


def test_reciprocal_mode_preflight_matches_production_geometry(fixture):
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    cell = arrays["cell"]
    production = ops.geometry_cache(
        arrays["node_positions"], cell, arrays["batch"].astype(np.int32)
    )
    expected = production["k_vectors"].shape[0]
    assert ops.reciprocal_mode_count(cell) == expected
    assert ops.reciprocal_mode_count(cell, maximum_count=3) == min(expected, 3)


@pytest.fixture(params=GEOMETRIES)
def fixture(request):
    loaded = load_fixture(request.param)
    loaded["ops"] = fixture_ops(loaded["manifest"])
    yield loaded
    # Release the native owner before the session Kokkos finalizer: the
    # final test item's funcargs are dropped after the session fixture
    # stack has unwound, so the ops object must not still own Kokkos
    # allocations then -- the same release-the-native-owners idiom the
    # sibling long-range batteries use.
    loaded["ops"] = None
    del loaded
    gc.collect()


# ---------------------------------------------------------------------------
# kspace-geometry-cache
# ---------------------------------------------------------------------------


def build_cache(ops, arrays):
    """The one shared cache per evaluation; the handle rides under 'cache'."""
    result = ops.geometry_cache(
        arrays["node_positions"], arrays["cell"], arrays["batch"].astype(np.int32)
    )
    return result, result["cache"]


def test_geometry_cache_parity(fixture):
    arrays = fixture["arrays"]
    cache, _ = build_cache(fixture["ops"], arrays)
    assert_parity(cache["k_vectors"], arrays["k_vectors"], "k_vectors")
    assert_parity(cache["k_norm2"], arrays["k_norm2"], "k_norm2")
    assert_parity(cache["k_graph_index"], arrays["k_batch"], "k_graph_index")
    assert_parity(cache["k0_mask"], arrays["k0_mask"], "k0_mask")
    assert_parity(cache["volume_per_k"], arrays["volume_per_k"], "volume_per_k")
    assert_parity(
        cache["k_factor_coulomb"], arrays["k_factor_coulomb"], "k_factor_coulomb"
    )
    assert_parity(cache["k_factor_proj"], arrays["k_factor_proj"], "k_factor_proj")
    assert_parity(cache["cos_table"], arrays["cosines"], "cos_table")
    assert_parity(cache["sin_table"], arrays["sines"], "sin_table")
    assert_parity(
        cache["density_fourier_factors"],
        arrays["density_basis_fs"],
        "density_fourier_factors",
    )
    assert_parity(
        cache["feature_fourier_factors"],
        arrays["feature_basis_fs"],
        "feature_fourier_factors",
    )
    # The k=0 Coulomb factor is zeroed exactly (branchy mask, not a scale).
    k0 = arrays["k0_mask"] > 0.5
    assert np.all(cache["k_factor_coulomb"][k0] == 0.0)
    assert np.all(cache["k_factor_proj"][k0] == 0.5)


# ---------------------------------------------------------------------------
# kspace-coulomb-features: assemble, convolve, project
# ---------------------------------------------------------------------------


def test_coulomb_features_stage_parity(fixture):
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    _, cache = build_cache(ops, arrays)
    for channel in CHANNELS:
        source = arrays[f"source_feats_{channel}"]
        density = ops.assemble_fourier(cache, source)
        assert_parity(density, arrays[f"rho_{channel}"], f"rho_{channel}")
        potential = ops.apply_coulomb(cache, density)
        assert_parity(potential, arrays[f"potential_{channel}"], f"potential_{channel}")
        features = ops.project_to_features(cache, potential)
        assert_parity(
            features, arrays[f"features_pbc_{channel}"], f"features_pbc_{channel}"
        )
        # One shared cache serves both channels: recompute from the same cache
        # and require identical stage outputs (the schedule-sharing contract).
        again = ops.assemble_fourier(cache, source)
        assert np.array_equal(again, density)


# ---------------------------------------------------------------------------
# kspace-coulomb-energy
# ---------------------------------------------------------------------------


def test_coulomb_energy_parity(fixture):
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    _, cache = build_cache(ops, arrays)
    periodic = fixture["manifest"]["geometry"]["graph_class"] == "bulk"
    for label in ENERGY_LABELS:
        source = arrays[f"source_feats_{label}"]
        # Pure k-space energy (the oracle's include_self_interaction=True path).
        kspace = ops.coulomb_energy(cache, source, None)
        assert_parity(
            kspace, arrays[f"energy_kspace_{label}"], f"energy_kspace_{label}"
        )
        # The node's contract path: consume the self-interaction fields and
        # subtract them (include_self_interaction=0).  The oracle decomposes
        # energy_final = kspace - 0.5*sum(source*self_fields) + correction, so
        # subtracting the fixture correction recovers the node output.
        self_fields = arrays[f"energy_self_fields_{label}"]
        node_energy = ops.coulomb_energy(cache, source, self_fields)
        correction = (
            np.zeros(1) if periodic else arrays[f"energy_molecule_correction_{label}"]
        )
        expected = arrays[f"energy_final_{label}"] - correction
        assert_parity(node_energy, expected, f"energy_self_subtracted_{label}")


# ---------------------------------------------------------------------------
# self-interaction-linear
# ---------------------------------------------------------------------------


def test_self_interaction_parity(fixture):
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    assert_parity(
        ops.si_feature_constants(),
        arrays["si_overlap_constants_features"],
        "si_overlap_constants_features",
    )
    assert_parity(
        ops.si_energy_constants(),
        arrays["si_overlap_constants_energy"],
        "si_overlap_constants_energy",
    )
    for channel in CHANNELS:
        source = arrays[f"source_feats_{channel}"]
        assert_parity(
            ops.feature_self_terms(source),
            arrays[f"features_self_interaction_{channel}"],
            f"features_self_interaction_{channel}",
        )
        assert_parity(
            ops.energy_self_fields(source),
            arrays[f"energy_self_fields_{channel}"],
            f"energy_self_fields_{channel}",
        )


# ---------------------------------------------------------------------------
# external-field-projection (constant matrix has a fixture; the gauge forward
# is pinned against a NumPy mirror)
# ---------------------------------------------------------------------------


def test_external_field_matrix_parity(fixture):
    arrays = fixture["arrays"]
    matrix = fixture["ops"].external_field_matrix()
    assert_parity(matrix, arrays["external_field_matrix"], "external_field_matrix")
    # The external-field and internal-field blocks share one constant matrix.
    assert_parity(matrix, arrays["internal_field_matrix"], "internal_field_matrix")


def test_external_field_features_mirror():
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    rng = np.random.default_rng(20261014)
    n_nodes, n_graphs = 7, 2
    graph_of = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    positions = rng.normal(size=(n_nodes, 3))
    external_potential = rng.normal(size=(n_graphs, 4))
    norms = rng.uniform(0.5, 2.0, size=8)
    got = ops.external_field_features(external_potential, positions, graph_of, norms)
    # Mirror of the upstream gauge: per-graph barycenter (fp64 scatter mean),
    # V += E.(r - barycenter), permute [V, E_z, E_x, E_y], project through the
    # constant matrix, halve per spin channel, divide by the feature norms.
    matrix = ops.external_field_matrix()
    barycenter = np.stack(
        [positions[graph_of == g].mean(axis=0) for g in range(n_graphs)]
    )
    displaced = positions - barycenter[graph_of]
    node_fields = external_potential[graph_of].copy()
    node_fields[:, 0] += np.sum(displaced * node_fields[:, 1:], axis=1)
    permuted = node_fields[:, [0, 3, 1, 2]]
    expected = (0.5 * (permuted @ matrix.T)) / norms[None, :]
    assert_parity(got, expected, "external_field_features")
    # Null norms mean the unit vector (the norm divide is the family's choice).
    un_normed = ops.external_field_features(
        external_potential, positions, graph_of, None
    )
    assert_parity(
        un_normed, 0.5 * (permuted @ matrix.T), "external_field_features_unnormed"
    )
    # A non-empty norm vector must carry one entry per feature: a wrong
    # length is a caller error, not a silent unit-norm fallback.
    with pytest.raises(ValueError, match=r"one entry per feature: expected 8, got 7"):
        ops.external_field_features(external_potential, positions, graph_of, np.ones(7))


# ---------------------------------------------------------------------------
# fukui-normalize and dipole-readout: NumPy mirrors of the upstream formulas
# ---------------------------------------------------------------------------


def test_fukui_normalize_mirror():
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    rng = np.random.default_rng(7)
    n_nodes, n_graphs = 9, 3
    graph_of = np.array([0] * 3 + [1] * 4 + [2] * 2, dtype=np.int32)
    fukui = rng.normal(size=(n_nodes, 2))
    fukui[graph_of == 2] = 0.0  # zero-sum graph: the guard must kick in
    q_alpha = rng.normal(size=n_nodes)
    q_beta = rng.normal(size=n_nodes)
    total_charge = rng.normal(size=n_graphs)
    total_spin = rng.normal(size=n_graphs)
    result = ops.fukui_normalize(
        fukui, q_alpha, q_beta, total_charge, total_spin, graph_of
    )
    # Mirror of extensions.py:746-771 -- fp64 per-graph scatter of the sources
    # (zero-guarded), broadcast, then the charge/spin target correction.
    norm = np.stack([fukui[graph_of == g].sum(axis=0) for g in range(n_graphs)])
    norm = np.where(norm == 0, 1.0, norm)
    f_normed = fukui / norm[graph_of]
    q_p_s = (total_charge + (total_spin - 1.0))[graph_of]
    q_m_s = (total_charge - (total_spin - 1.0))[graph_of]
    pred_alpha = np.stack([q_alpha[graph_of == g].sum() for g in range(n_graphs)])[
        graph_of
    ]
    pred_beta = np.stack([q_beta[graph_of == g].sum() for g in range(n_graphs)])[
        graph_of
    ]
    expected_alpha = f_normed[:, 0] * ((q_p_s / 2.0) - pred_alpha)
    expected_beta = f_normed[:, 1] * ((q_m_s / 2.0) - pred_beta)
    assert_parity(result["charge_correction_alpha"], expected_alpha, "fukui_alpha")
    assert_parity(result["charge_correction_beta"], expected_beta, "fukui_beta")
    # The zero-sum graph's sources normalize to zero, so its corrections vanish.
    assert np.all(result["charge_correction_alpha"][graph_of == 2] == 0.0)
    assert np.all(result["charge_correction_beta"][graph_of == 2] == 0.0)


def test_dipole_readout_mirror():
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    rng = np.random.default_rng(11)
    n_nodes, n_graphs = 8, 3
    graph_of = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    density = rng.normal(size=(n_nodes, 4))
    positions = rng.normal(size=(n_nodes, 3))
    result = ops.dipole_readout(density, positions, graph_of)
    charge = np.stack([density[graph_of == g, 0].sum() for g in range(n_graphs)])
    dipole = np.stack(
        [
            (density[graph_of == g, 0, None] * positions[graph_of == g]).sum(axis=0)
            + density[graph_of == g, 1:4].sum(axis=0)[[2, 0, 1]]
            for g in range(n_graphs)
        ]
    )
    assert_parity(result["total_charge"], charge, "total_charge")
    assert_parity(result["total_dipole"], dipole, "total_dipole")


# ---------------------------------------------------------------------------
# analytic corrections
# ---------------------------------------------------------------------------


def test_analytic_correction_energy_parity(fixture):
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    batch = arrays["batch"].astype(np.int32)
    pbc = arrays["pbc"].astype(np.int32)
    for label in ENERGY_LABELS:
        source = arrays[f"source_feats_{label}"]
        result = ops.analytic_correction_energy(source, positions, cell, batch, pbc)
        assert_parity(
            result["molecule_correction"],
            arrays[f"energy_molecule_correction_{label}"],
            f"energy_molecule_correction_{label}",
        )
        assert_parity(
            result["slab_correction"],
            arrays[f"energy_slab_correction_{label}"],
            f"energy_slab_correction_{label}",
        )
    # Mode selection from the graph's pbc row: molecule graphs take the box
    # term, bulk graphs take nothing.
    molecule = fixture["manifest"]["geometry"]["graph_class"] == "molecule"
    selected = result["correction_energy"]
    if molecule:
        assert_parity(selected, result["molecule_correction"], "selected_molecule")
    else:
        assert np.all(selected == 0.0)


def test_analytic_correction_features_parity(fixture):
    arrays = fixture["arrays"]
    ops = fixture["ops"]
    positions = arrays["node_positions"]
    cell = arrays["cell"]
    batch = arrays["batch"].astype(np.int32)
    pbc = arrays["pbc"].astype(np.int32)
    for channel in CHANNELS:
        source = arrays[f"source_feats_{channel}"]
        result = ops.analytic_correction_features(
            source, source, positions, cell, batch, pbc
        )
        assert_parity(
            result["correction_fields_alpha"],
            arrays[f"features_correction_{channel}"],
            f"features_correction_{channel}",
        )
        assert_parity(
            result["correction_fields_beta"],
            arrays[f"features_correction_{channel}"],
            f"features_correction_{channel}_beta",
        )


# ---------------------------------------------------------------------------
# multi-graph behavior the single-graph fixtures cannot cover
# ---------------------------------------------------------------------------


def test_two_graph_kgrid_grouping_and_correction_modes():
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    # Graph 0: molecule (no pbc).  Graph 1: slab (T,T,F).  Different cells so
    # the per-graph search bounds differ and the grouping is observable.
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
    cache = ops.geometry_cache(positions, cell, graph_of)
    k_batch = cache["k_graph_index"]
    k_vectors = cache["k_vectors"]

    # Mirror of the oracle enumeration: origin, open half line, half plane,
    # half space (last factor varies fastest), masked per graph by |k| <= cut.
    import math

    def mirror_grid(cut, c):
        rcell = 2 * math.pi * np.linalg.inv(c.T)
        norms = np.linalg.norm(c, axis=-1)
        dots = (rcell * (c / norms[:, None])).sum(axis=-1)
        ns = np.ceil(cut / dots).astype(int)
        n1, n2, n3 = ns
        triples = [(0, 0, 0)]
        triples += [(0, 0, z) for z in range(1, n3)]
        triples += [(0, y, z) for y in range(1, n2) for z in range(-n3, n3)]
        triples += [
            (x, y, z)
            for x in range(1, n1)
            for y in range(-n2, n2)
            for z in range(-n3, n3)
        ]
        kv = np.array(triples, float) @ rcell
        keep = (kv * kv).sum(-1) <= cut * cut
        return kv[keep]

    for graph in range(2):
        rows = k_batch == graph
        assert_parity(
            k_vectors[rows], mirror_grid(2.0, cell[graph]), f"graph{graph}_k_vectors"
        )
    # Grouped by graph: every row of graph 0 precedes every row of graph 1.
    assert np.all(np.diff(np.asarray(k_batch)) >= 0)

    # Correction modes selected per graph from the pbc rows.
    density = np.array(
        [
            [0.8, 0.1, -0.2, 0.3],
            [-0.4, 0.0, 0.5, -0.1],
            [0.6, -0.3, 0.2, 0.4],
            [-0.2, 0.7, -0.6, 0.1],
            [0.5, -0.4, 0.3, 0.2],
        ]
    )
    energy = ops.analytic_correction_energy(
        density, positions, cell, graph_of, graph_pbc
    )
    assert_parity(
        energy["correction_energy"][0:1],
        energy["molecule_correction"][0:1],
        "molecule graph takes the box term",
    )
    assert_parity(
        energy["correction_energy"][1:2],
        energy["slab_correction"][1:2],
        "slab graph takes the dipole term",
    )
    features = ops.analytic_correction_features(
        density, density, positions, cell, graph_of, graph_pbc
    )
    # Slab node fields: [field_z*z, 0, 0, field_z] through the constant matrix.
    matrix = ops.external_field_matrix()
    slab_nodes = graph_of == 1
    mu_z = (density[slab_nodes, 0] * positions[slab_nodes, 2]).sum() + density[
        slab_nodes, 2
    ].sum()
    volumes = np.abs(np.linalg.det(cell))
    field_z = (1.0 / 5.526349406e-3) * mu_z / volumes[1]
    slab_rows = slab_nodes
    slab_fields = np.zeros((slab_rows.sum(), 4))
    slab_fields[:, 0] = field_z * positions[slab_rows, 2]
    slab_fields[:, 3] = field_z
    expected_slab = (slab_fields[:, [0, 3, 1, 2]] @ matrix.T) / 1.0
    assert_parity(
        features["correction_fields_alpha"][slab_rows], expected_slab, "slab features"
    )


def test_triclinic_kgrid_matches_the_reciprocal_cell_mirror():
    """A triclinic cell builds k from the reciprocal cell 2 pi inv(A)^T.

    The diagonal fixtures cannot distinguish the reciprocal orientation,
    so this synthetic triclinic case pins it: the k-vectors are the
    integer triples against ``2 pi inv(cell).T`` (the oracle's
    ``torch.linalg.inv(cell).transpose(-1, -2)``), and the phase tables
    are exactly the cos/sin of ``2 pi n . frac`` -- the covector
    transformation the cell adjoint's conventions build on.
    """
    ops = symmetrix.PolarLongRangeOps(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    cell = np.array([[5.0, 0.9, 0.3], [0.4, 4.5, 0.7], [0.2, 0.6, 5.5]]).reshape(
        1, 3, 3
    )
    positions = np.array([[0.3, 0.4, 0.2], [0.9, 0.8, 3.0], [2.0, 1.0, 0.5]])
    graph_of = np.zeros(3, dtype=np.int32)
    result = ops.geometry_cache(positions, cell, graph_of)

    import math

    a = cell[0]
    rcell = 2 * math.pi * np.linalg.inv(a).T
    norms = np.linalg.norm(a, axis=1)
    dots = (rcell * (a / norms[:, None])).sum(axis=1)
    ns = np.ceil(2.0 / dots).astype(int)
    n1, n2, n3 = ns
    triples = [(0, 0, 0)]
    triples += [(0, 0, z) for z in range(1, n3)]
    triples += [(0, y, z) for y in range(1, n2) for z in range(-n3, n3)]
    triples += [
        (x, y, z) for x in range(1, n1) for y in range(-n2, n2) for z in range(-n3, n3)
    ]
    kv = np.array(triples, float) @ rcell
    keep = (kv * kv).sum(-1) <= 4.0
    assert_parity(result["k_vectors"], kv[keep], "triclinic_k_vectors")

    # The phase is the fractional-coordinate dot product exactly: with
    # r = frac @ A, k . r = 2 pi n . frac for every kept triple.
    frac = positions @ np.linalg.inv(a)
    expected = 2 * math.pi * (np.array(triples, float)[keep] @ frac.T)
    assert not np.allclose(
        np.linalg.inv(a), np.linalg.inv(a).T
    ), "the fixture cell must be genuinely triclinic for this check to bite"
    assert_parity(result["cos_table"], np.cos(expected), "triclinic_phase_cos")
    assert_parity(result["sin_table"], np.sin(expected), "triclinic_phase_sin")


# ---------------------------------------------------------------------------
# declaration-time validation of the analytic l <= 1 bases
# ---------------------------------------------------------------------------


def test_constructor_rejects_unsupported_shapes():
    good = dict(
        kspace_cutoff=2.0,
        density_max_l=1,
        density_smearing_width=1.0,
        feature_max_l=1,
        feature_smearing_widths=[1.0, 1.5],
    )
    with pytest.raises(ValueError):
        symmetrix.PolarLongRangeOps(**{**good, "density_max_l": 2})
    with pytest.raises(ValueError):
        symmetrix.PolarLongRangeOps(**{**good, "feature_max_l": 3})
    with pytest.raises(ValueError):
        symmetrix.PolarLongRangeOps(**{**good, "kspace_cutoff": 0.0})
    with pytest.raises(ValueError):
        symmetrix.PolarLongRangeOps(**{**good, "density_smearing_width": -1.0})
    with pytest.raises(ValueError):
        symmetrix.PolarLongRangeOps(**{**good, "feature_smearing_widths": []})
