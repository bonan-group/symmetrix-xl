"""Structural validation of the MACE-POLAR-1 oracle golden fixtures.

The fixtures under ``symmetrix/test/data/polar/`` were captured out-of-tree by
``benchmarks/polar_oracle_fixture_runner.py`` against the frozen oracle stack
for the model described in ``docs/user/models.md#mace-polar-1``. They are the
reference values for the long-range operator kinds; this file checks only
their structure, integrity, and self-consistency. Kernel parity is checked
by the long-range operator tests.

Runs in the plain repository environment: numpy and stdlib only. The oracle
checkouts (torch/e3nn/graph_longrange/mace) are never imported here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPOSITORY / "symmetrix/test/data/polar"

FIXTURE_SCHEMA = "symmetrix.polar.oracle-op-fixtures"
FIXTURE_VERSION = 1

# Pinned oracle provenance — must stay in lockstep with the runner constants
# (benchmarks/polar_oracle_fixture_runner.py); see the maintained model guide
# (docs/user/models.md#mace-polar-1).
ORACLE_MACE_SHA = "22f0809735bd4dd1deba80cf8e16f89913a35ff4"
ORACLE_MACE_REF = "v0.3.16-2-g22f08097"
ORACLE_GRAPH_ELECTROSTATICS_SHA = "e0eb8945c21e203656f4e0bbece5d1da488b6753"
ORACLE_GRAPH_LONGRANGE_VERSION = "0.4.4"

RUNNER_PARAMETERS = {
    "density_max_l": 1,
    "density_smearing_width": 1.0,
    "feature_max_l": 1,
    "feature_smearing_widths": [1.0, 1.5],
    "kspace_cutoff_factor": 1.5,
    "integral_normalization": "receiver",
    "pbc_handling": "mixed_periodic",
    "include_self_interaction": False,
    "seed": 20261014,
}
FIELD_CONSTANT = 1.0 / (5.526349406e-3)
HALF_SPHERE_WEIGHT = 2.0

FIXTURES = ("mol5", "pbc4")
SPIN_CHANNELS = ("alpha", "beta")
ENERGY_LABELS = ("alpha", "beta", "total")

M_DENSITY = 4  # (density_max_l + 1)**2
M_FEATURE = 4  # (feature_max_l + 1)**2
N_SIGMA_FEATURE = 2  # len(feature_smearing_widths)
FEATURES_DIM = M_FEATURE * N_SIGMA_FEATURE  # 8


def _expected_arrays(n_nodes: int, n_k: int) -> dict[str, tuple[tuple[int, ...], str]]:
    """Full expected inventory: name -> (shape, dtype)."""
    f = "float64"
    arrays: dict[str, tuple[tuple[int, ...], str]] = {
        # inputs
        "cell": ((1, 3, 3), f),
        "rcell": ((1, 3, 3), f),
        "pbc": ((1, 3), "bool"),
        "node_positions": ((n_nodes, 3), f),
        "batch": ((n_nodes,), "int64"),
        "volume": ((1,), f),
        "kspace_cutoff": ((1,), f),
        "field_constant": ((1,), f),
        "half_sphere_weight": ((1,), f),
        # k-grid and folding metadata
        "k_vectors": ((n_k, 3), f),
        "k_norm2": ((n_k,), f),
        "k_batch": ((n_k,), "int64"),
        "k0_mask": ((n_k,), f),
        "volume_per_k": ((n_k,), f),
        "k_factor_coulomb": ((n_k,), f),
        "k_factor_proj": ((n_k,), f),
        # geometry cache
        "k_dot_r": ((n_k, n_nodes), f),
        "cosines": ((n_k, n_nodes), f),
        "sines": ((n_k, n_nodes), f),
        "density_basis_fs": ((n_k, 1, M_DENSITY, 2), f),
        "feature_basis_fs": ((n_k, N_SIGMA_FEATURE, M_FEATURE, 2), f),
        # correction masks
        "correction_mask_is_molecule_node": ((n_nodes,), "bool"),
        "correction_mask_is_slab_node": ((n_nodes,), "bool"),
        # constants
        "si_overlap_constants_features": ((FEATURES_DIM,), f),
        "si_overlap_constants_energy": ((M_DENSITY,), f),
        "external_field_matrix": ((FEATURES_DIM, 4), f),
        "internal_field_matrix": ((FEATURES_DIM, 4), f),
        "output_permutation": ((FEATURES_DIM,), "int64"),
        "source_feats_total": ((n_nodes, M_DENSITY), f),
    }
    for channel in SPIN_CHANNELS:
        arrays.update(
            {
                f"source_feats_{channel}": ((n_nodes, M_DENSITY), f),
                f"rho_{channel}": ((n_k, 2), f),
                f"potential_{channel}": ((n_k, 2), f),
                f"features_pbc_{channel}": ((n_nodes, FEATURES_DIM), f),
                f"features_correction_{channel}": ((n_nodes, FEATURES_DIM), f),
                f"features_final_{channel}": ((n_nodes, FEATURES_DIM), f),
                f"features_self_interaction_{channel}": ((n_nodes, FEATURES_DIM), f),
                f"esps_{channel}": ((n_nodes,), f),
            }
        )
    for label in ENERGY_LABELS:
        arrays.update(
            {
                f"energy_kspace_{label}": ((1,), f),
                f"energy_final_{label}": ((1,), f),
                f"energy_self_fields_{label}": ((n_nodes, M_DENSITY), f),
                f"energy_molecule_correction_{label}": ((1,), f),
                f"energy_slab_correction_{label}": ((1,), f),
            }
        )
    return arrays


def _require_fixture_directory() -> None:
    # Binding resolution: the only clean-skip condition is an absent fixture
    # directory. Once the directory exists, a missing or partial fixture is a
    # corruption failure, never a skip.
    if not FIXTURE_DIR.is_dir():
        pytest.skip(
            f"polar oracle fixture directory is absent: {FIXTURE_DIR} "
            "(regenerate with benchmarks/polar_oracle_fixture_runner.py)"
        )


def _load_manifest(name: str) -> dict:
    _require_fixture_directory()
    path = FIXTURE_DIR / f"{name}_manifest.json"
    assert path.is_file(), f"polar oracle fixture manifest is missing: {path}"
    return json.loads(path.read_text())


def _load_arrays(name: str) -> dict[str, np.ndarray]:
    manifest = _load_manifest(name)
    path = FIXTURE_DIR / manifest["payload"]["path"]
    assert path.is_file(), f"polar oracle fixture payload is missing: {path}"
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


@pytest.mark.parametrize("name", FIXTURES)
def test_manifest_pins_the_frozen_oracle_and_runner_parameters(name):
    manifest = _load_manifest(name)
    assert manifest["schema"] == FIXTURE_SCHEMA
    assert manifest["version"] == FIXTURE_VERSION
    assert manifest["oracle"] == {
        "mace_sha": ORACLE_MACE_SHA,
        "mace_ref": ORACLE_MACE_REF,
        "graph_electrostatics_sha": ORACLE_GRAPH_ELECTROSTATICS_SHA,
        "graph_longrange_version": ORACLE_GRAPH_LONGRANGE_VERSION,
    }
    parameters = dict(manifest["parameters"])
    kspace_cutoff = parameters.pop("kspace_cutoff")
    assert parameters == RUNNER_PARAMETERS
    # The stored cutoff scalar and the k-grid must agree with the declaration.
    arrays = _load_arrays(name)
    assert arrays["kspace_cutoff"].shape == (1,)
    assert arrays["kspace_cutoff"][0] == kspace_cutoff
    assert np.all(arrays["k_norm2"] <= kspace_cutoff**2 + 1e-12)
    units = manifest["units"]
    assert units["field_constant"] == pytest.approx(FIELD_CONSTANT, rel=1e-15)
    assert units["half_sphere_weight"] == HALF_SPHERE_WEIGHT
    assert arrays["field_constant"][0] == pytest.approx(FIELD_CONSTANT, rel=1e-15)
    assert arrays["half_sphere_weight"][0] == HALF_SPHERE_WEIGHT


@pytest.mark.parametrize("name", FIXTURES)
def test_payload_integrity_matches_manifest(name):
    manifest = _load_manifest(name)
    payload = manifest["payload"]
    assert payload["format"] == "npz"
    path = FIXTURE_DIR / payload["path"]
    assert path.is_file()
    assert path.stat().st_size == payload["bytes"]
    assert _sha256_file(path) == payload["sha256"]


@pytest.mark.parametrize("name", FIXTURES)
def test_array_inventory_shapes_and_dtypes(name):
    manifest = _load_manifest(name)
    arrays = _load_arrays(name)
    geometry = manifest["geometry"]
    n_nodes = geometry["n_nodes"]
    n_k = geometry["n_k"]
    expected = _expected_arrays(n_nodes, n_k)
    assert set(arrays) == set(expected), "fixture payload arrays diverge from contract"
    assert set(manifest["arrays"]) == set(expected)
    for array_name, (shape, dtype) in expected.items():
        array = arrays[array_name]
        assert array.shape == shape, f"{array_name}: {array.shape} != {shape}"
        assert array.dtype == np.dtype(dtype), f"{array_name}: {array.dtype}"
        if dtype == "float64":
            assert np.all(np.isfinite(array)), f"{array_name}: non-finite values"
        declared = manifest["arrays"][array_name]
        assert declared["shape"] == list(shape)
        assert declared["dtype"] == dtype
        assert declared["sha256"] == _sha256_array(array)


@pytest.mark.parametrize("name", FIXTURES)
def test_k_grid_folding_metadata_is_self_consistent(name):
    arrays = _load_arrays(name)
    k_norm2 = arrays["k_norm2"]
    k0_mask = arrays["k0_mask"]
    # Single graph per fixture: one origin mode, all k rows owned by graph 0.
    assert np.all(arrays["k_batch"] == 0)
    assert np.all(arrays["batch"] == 0)
    origin = k0_mask > 0.5
    assert origin.sum() == 1
    assert k_norm2[origin][0] == 0.0
    assert np.all(arrays["k_vectors"][origin][0] == 0.0)
    assert np.all(k_norm2[~origin] > 0.0)
    # The half-sphere grid keeps exactly one of each +-k pair (the origin,
    # whose negation is itself, is excluded).
    rounded = np.round(arrays["k_vectors"][~origin], decimals=9)
    unique = {tuple(row) for row in rounded.tolist()}
    assert len(unique) == rounded.shape[0], "duplicate k-vector in half-sphere grid"
    for row in unique:
        assert tuple(-value for value in row) not in unique, "-k also present in grid"
    # Coulomb factor: zero at the origin, 1/k^2 elsewhere (FIELD_CONSTANT is
    # applied inside the evaluation kernels, not folded into this factor).
    coulomb = arrays["k_factor_coulomb"]
    assert coulomb[origin][0] == 0.0
    np.testing.assert_allclose(coulomb[~origin], 1.0 / k_norm2[~origin], rtol=0, atol=0)
    # Projection folding weight: half weight at the origin, unit elsewhere.
    proj = arrays["k_factor_proj"]
    assert proj[origin][0] == 0.5
    assert np.all(proj[~origin] == 1.0)
    assert np.all(arrays["volume_per_k"] == arrays["volume"][0])


@pytest.mark.parametrize("name", FIXTURES)
def test_geometry_cache_tables_match_their_phase_gemm(name):
    arrays = _load_arrays(name)
    k_dot_r = arrays["k_dot_r"]
    np.testing.assert_allclose(arrays["cosines"], np.cos(k_dot_r), rtol=0, atol=1e-14)
    np.testing.assert_allclose(arrays["sines"], np.sin(k_dot_r), rtol=0, atol=1e-14)
    # k.r must equal the k-grid contracted with the stored positions.
    np.testing.assert_allclose(
        k_dot_r, arrays["k_vectors"] @ arrays["node_positions"].T, rtol=1e-13, atol=0
    )
    # GTO Fourier factors vanish in the imaginary part for l=0 rows.
    density_basis = arrays["density_basis_fs"]
    feature_basis = arrays["feature_basis_fs"]
    assert density_basis.shape[2] == M_DENSITY
    assert feature_basis.shape[2] == M_FEATURE
    # The l-major channel permutation is a true permutation of the channels.
    permutation = arrays["output_permutation"]
    assert sorted(permutation.tolist()) == list(range(FEATURES_DIM))


@pytest.mark.parametrize("name", FIXTURES)
def test_spin_channels_and_energy_labels_are_consistent(name):
    arrays = _load_arrays(name)
    total = arrays["source_feats_total"]
    np.testing.assert_allclose(
        total,
        arrays["source_feats_alpha"] + arrays["source_feats_beta"],
        rtol=0,
        atol=0,
    )
    # Distinct seeded channels must give distinct structure factors.
    assert not np.array_equal(arrays["rho_alpha"], arrays["rho_beta"])
    for channel in SPIN_CHANNELS:
        assert np.all(np.isfinite(arrays[f"energy_final_{channel}"]))
    for label in ENERGY_LABELS:
        # The mixed_periodic decomposition: final = kspace - self + selected
        # correction, where the selected correction depends on the graph class.
        kspace = arrays[f"energy_kspace_{label}"]
        final = arrays[f"energy_final_{label}"]
        self_fields = arrays[f"energy_self_fields_{label}"]
        source = total if label == "total" else arrays[f"source_feats_{label}"]
        node_self_energy = np.sum(source * self_fields, axis=1)
        correction = final - (kspace - 0.5 * node_self_energy.sum())
        selected = (
            arrays[f"energy_molecule_correction_{label}"]
            if arrays["correction_mask_is_molecule_node"].all()
            else 0.0
        )
        np.testing.assert_allclose(correction, selected, rtol=1e-12, atol=1e-9)
    # Energy magnitudes stay in a sane eV range for these toy densities.
    for label in ENERGY_LABELS:
        assert np.abs(arrays[f"energy_final_{label}"][0]) < 1e4


@pytest.mark.parametrize("name", FIXTURES)
def test_correction_masks_select_the_declared_graph_class(name):
    manifest = _load_manifest(name)
    arrays = _load_arrays(name)
    pbc = arrays["pbc"].reshape(-1, 3)[0]
    is_molecule = bool((~pbc).all())
    is_slab = bool(pbc[0] and pbc[1] and not pbc[2])
    assert manifest["geometry"]["graph_class"] == (
        "molecule" if is_molecule else ("slab" if is_slab else "bulk")
    )
    assert np.all(arrays["correction_mask_is_molecule_node"] == is_molecule)
    assert np.all(arrays["correction_mask_is_slab_node"] == is_slab)


def test_constant_operators_are_identical_across_geometries():
    constants = (
        "si_overlap_constants_features",
        "si_overlap_constants_energy",
        "external_field_matrix",
        "internal_field_matrix",
        "output_permutation",
    )
    loaded = {name: _load_arrays(name) for name in FIXTURES}
    for constant in constants:
        np.testing.assert_array_equal(
            loaded["mol5"][constant],
            loaded["pbc4"][constant],
            err_msg=f"{constant} must be geometry-independent",
        )
