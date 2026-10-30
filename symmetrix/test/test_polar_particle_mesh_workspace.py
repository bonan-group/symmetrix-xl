"""POLAR-owned boundary over the physics-neutral particle-mesh workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import symmetrix


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native():
    module = symmetrix.load_backend()
    if not module._kokkos_is_initialized():
        module._init_kokkos()
    if not module._particle_mesh_capability()["available"]:
        pytest.skip("requires a particle-mesh FFT provider")
    return module


def _workspace(native, precision="float64", receiver_widths=3):
    return native._PolarParticleMeshWorkspace(precision, 1, 1, 2, receiver_widths)


def _scalar_workspace(native, precision="float64", receiver_widths=2):
    return native._PolarParticleMeshWorkspace(precision, 0, 0, 2, receiver_widths)


def _scalar_case():
    positions = np.array(
        [[0.43, 0.71, 1.19], [2.17, 1.31, 3.07], [4.03, 5.11, 2.29]],
        dtype=np.float64,
    )
    charges = np.array([0.7, -1.1, 0.4], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    return positions, charges, cell


def test_polar_workspace_declares_family_owned_l1_layout(native):
    workspace = _workspace(native)

    assert workspace.layout() == {
        "density_max_l": 1,
        "feature_max_l": 1,
        "spin_channels": 2,
        "receiver_widths": 3,
        "source_components": 4,
        "receiver_components": 4,
        "source_lanes": 8,
        "receiver_lanes": 24,
        "transform_lanes": 24,
    }


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (("float64", 2, 1, 2, 1), "density_max_l"),
        (("float64", 1, 2, 2, 1), "feature_max_l"),
        (("float64", 1, 1, 1, 1), "spin_channels"),
        (("float64", 1, 1, 2, 0), "receiver_widths"),
    ],
)
def test_polar_workspace_rejects_unsupported_layout_before_allocation(
    native, arguments, match
):
    with pytest.raises(ValueError, match=match):
        native._PolarParticleMeshWorkspace(*arguments)


def test_polar_workspace_checks_flattened_ranges_before_allocation(native):
    with pytest.raises(OverflowError, match="receiver-width lanes"):
        native._PolarParticleMeshWorkspace("float64", 1, 1, 2, 2**62)

    workspace = _workspace(native)
    with pytest.raises(OverflowError, match="position coordinates"):
        workspace.reserve(8, 8, 8, 2**62, 4)
    assert workspace.usage()["reservations"] == 0

    workspace.arm_allocation_failure(0)
    try:
        with pytest.raises(OverflowError, match="real_mesh"):
            workspace.reserve(512, 512, 512, 1, 4)
    finally:
        workspace.clear_allocation_failure()
    assert workspace.usage()["reservations"] == 0


def test_polar_workspace_zero_extent_buffers_have_no_allocation_identity(native):
    workspace = _workspace(native)
    workspace.reserve(8, 8, 8, 0, 4)

    buffers = {item["name"]: item for item in workspace.usage()["buffers"]}
    for name in (
        "positions",
        "source_multipoles",
        "receiver_features",
        "transform_weights",
        "transform_position_forces",
        "source_multipole_adjoints",
        "position_adjoints",
    ):
        assert buffers[name]["capacity_elements"] == 0
        assert buffers[name]["allocation_id"] == 0
    for name in ("stencil_base", "stencil_weight", "stencil_slope"):
        assert buffers[name]["capacity_elements"] == 0
        assert buffers[name]["allocation_id"] == 0


def test_polar_workspace_reuses_compatible_storage_and_itemizes_memory(native):
    workspace = _workspace(native, receiver_widths=2)

    first = workspace.reserve(8, 10, 12, 5, 4)
    second = workspace.reserve(8, 10, 12, 5, 4)
    usage = workspace.usage()

    assert first["rebuilt"] is True
    assert second["rebuilt"] is False
    assert usage["reservations"] == 2
    assert usage["rebuilds"] == 1
    assert usage["reuses"] == 1
    assert usage["nodes"] == 5
    assert usage["mesh"] == [8, 10, 12]
    assert usage["source_lanes"] == 8
    assert usage["receiver_lanes"] == 16
    assert usage["transform_lanes"] == 16
    names = {item["name"] for item in usage["buffers"]}
    assert names == {
        "positions",
        "source_multipoles",
        "receiver_features",
        "transform_weights",
        "transform_position_forces",
        "source_multipole_adjoints",
        "position_adjoints",
        "cell_adjoints",
        "cell_bin_adjoints",
        "receiver_sigmas",
        "channel_energies",
        "real_mesh",
        "spectrum",
        "stencil_base",
        "stencil_weight",
        "stencil_slope",
        "k_norm",
        "deconvolution",
        "hermitian",
    }
    assert all("mode_by_atom" not in name for name in names)
    assert all(item["allocation_id"] > 0 for item in usage["buffers"])
    assert usage["active_bytes"] == sum(
        item["active_bytes"] for item in usage["buffers"]
    )
    assert usage["capacity_bytes"] == sum(
        item["capacity_bytes"] for item in usage["buffers"]
    )


def test_polar_workspace_allocation_id_tracks_reuse_and_growth(native):
    workspace = _workspace(native)
    workspace.reserve(8, 8, 8, 4, 4)
    first = {
        item["name"]: item["allocation_id"] for item in workspace.usage()["buffers"]
    }

    workspace.reserve(8, 8, 8, 4, 4)
    reused = {
        item["name"]: item["allocation_id"] for item in workspace.usage()["buffers"]
    }
    assert reused == first

    workspace.reserve(12, 10, 8, 7, 5)
    grown = {
        item["name"]: item["allocation_id"] for item in workspace.usage()["buffers"]
    }
    retained = {"cell_adjoints", "receiver_sigmas", "channel_energies"}
    assert all(grown[name] == first[name] for name in retained)
    assert all(grown[name] != first[name] for name in first if name not in retained)


def test_polar_workspace_failed_rebuild_preserves_prior_state(native):
    workspace = _workspace(native)
    workspace.reserve(8, 8, 8, 4, 4)
    before = workspace.usage()

    for failure_index in range(16):
        workspace.arm_allocation_failure(failure_index)
        try:
            with pytest.raises(RuntimeError, match="injected"):
                workspace.reserve(12, 10, 8, 7, 5)
        finally:
            workspace.clear_allocation_failure()
        assert workspace.usage() == before
    assert workspace.reserve(8, 8, 8, 4, 4)["rebuilt"] is False


def test_polar_family_state_owns_the_workspace_boundary():
    state = (
        REPOSITORY_ROOT / "libsymmetrix/source/mace_polar_family_state.hpp"
    ).read_text()
    workspace = REPOSITORY_ROOT / "libsymmetrix/source/mace_polar_particle_mesh.hpp"

    assert workspace.is_file()
    assert "PolarParticleMeshWorkspaceT<Precision>" in state
    assert "LESParticleMesh" not in workspace.read_text()


def test_polar_scalar_pme_rejects_scope_before_allocation(native):
    positions, charges, cell = _scalar_case()
    workspace = _scalar_workspace(native)

    with pytest.raises(ValueError, match="fully periodic"):
        workspace.scalar_forward(
            positions,
            charges,
            cell,
            [True, True, False],
            0.6,
            [0.5, 0.8],
            [16, 16, 16],
            4,
        )
    assert workspace.usage()["reservations"] == 0

    l1_workspace = _workspace(native, receiver_widths=2)
    with pytest.raises(ValueError, match="scalar.*l=0"):
        l1_workspace.scalar_forward(
            positions,
            charges,
            cell,
            [True, True, True],
            0.6,
            [0.5, 0.8],
            [16, 16, 16],
            4,
        )
    assert l1_workspace.usage()["reservations"] == 0


def test_polar_scalar_pme_rejects_aliased_assignment_before_allocation(native):
    positions, charges, cell = _scalar_case()
    workspace = _scalar_workspace(native)

    with pytest.raises(ValueError, match="mesh.*spline order"):
        workspace.scalar_forward(
            positions,
            charges,
            cell,
            [True, True, True],
            0.6,
            [0.5, 0.8],
            [5, 8, 8],
            6,
        )

    assert workspace.usage()["reservations"] == 0


@pytest.mark.parametrize("field", ["positions", "sources"])
def test_polar_scalar_pme_rejects_nonfinite_particle_data_before_allocation(
    native, field
):
    positions, charges, cell = _scalar_case()
    if field == "positions":
        positions[1, 2] = np.nan
    else:
        charges[1] = np.inf
    workspace = _scalar_workspace(native)

    with pytest.raises(ValueError, match=f"finite {field}"):
        workspace.scalar_forward(
            positions,
            charges,
            cell,
            [True, True, True],
            0.6,
            [0.5, 0.8],
            [16, 16, 16],
            6,
        )

    assert workspace.usage()["reservations"] == 0


def test_polar_scalar_pme_rejects_nonfinite_cell_inverse_before_allocation(native):
    positions, charges, _ = _scalar_case()
    cell = np.diag([1.0e-310, 1.0, 1.0])
    workspace = _scalar_workspace(native)

    with pytest.raises(ValueError, match="finite inverse"):
        workspace.scalar_forward(
            positions,
            charges,
            cell,
            [True, True, True],
            0.6,
            [0.5, 0.8],
            [16, 16, 16],
            6,
        )

    assert workspace.usage()["reservations"] == 0


def test_polar_scalar_pme_failed_growth_keeps_prior_result(native):
    positions, charges, cell = _scalar_case()
    workspace = _scalar_workspace(native)
    arguments = (
        positions,
        charges,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [8, 8, 8],
        4,
    )
    expected = workspace.scalar_forward(*arguments)
    before = workspace.usage()

    for failure_index in range(16):
        workspace.arm_allocation_failure(failure_index)
        try:
            with pytest.raises(RuntimeError, match="injected"):
                workspace.reserve(12, 10, 8, 7, 5)
        finally:
            workspace.clear_allocation_failure()
        assert workspace.usage() == before
    actual = workspace.scalar_forward(*arguments)
    np.testing.assert_array_equal(actual["features"], expected["features"])
    assert actual["energy"] == expected["energy"]


def test_polar_scalar_pme_fp64_converges_to_direct_reciprocal(native):
    positions, charges, cell = _scalar_case()
    source_sigma = 0.6
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 0, source_sigma, 0, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(charges), dtype=np.int32)
    )["cache"]
    multipoles = charges[:, None]
    reference_features = direct.coulomb_features(cache, multipoles)
    reference_energy = direct.coulomb_energy(cache, multipoles)

    workspace = _scalar_workspace(native, receiver_widths=len(receiver_sigmas))
    energy_errors = []
    feature_errors = [[] for _ in receiver_sigmas]
    for extent in (12, 20, 32):
        result = workspace.scalar_forward(
            positions,
            charges,
            cell,
            [True, True, True],
            source_sigma,
            receiver_sigmas,
            [extent, extent, extent],
            6,
            include_feature_self_interaction=True,
            include_energy_self_interaction=True,
        )
        energy_errors.append(abs(result["energy"] - reference_energy[0]))
        for width in range(len(receiver_sigmas)):
            feature_errors[width].append(
                np.max(
                    np.abs(result["features"][:, width] - reference_features[:, width])
                )
            )

    assert energy_errors[2] < energy_errors[1] < energy_errors[0]
    assert energy_errors[-1] < 2.0e-5
    for errors in feature_errors:
        assert errors[2] < errors[1] < errors[0]
        assert errors[-1] < 2.0e-5


def test_polar_scalar_pme_charged_triclinic_case_matches_direct_and_wraps(native):
    positions, _, _ = _scalar_case()
    charges = np.array([0.7, -0.2, 0.4], dtype=np.float64)
    cell = np.array(
        [[8.0, 0.0, 0.0], [1.2, 7.5, 0.0], [0.4, 0.8, 8.5]],
        dtype=np.float64,
    )
    source_sigma = 0.6
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 0, source_sigma, 0, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(charges), dtype=np.int32)
    )["cache"]
    multipoles = charges[:, None]
    reference_features = direct.coulomb_features(cache, multipoles)
    reference_energy = direct.coulomb_energy(cache, multipoles)[0]

    workspace = _scalar_workspace(native, receiver_widths=len(receiver_sigmas))
    arguments = (
        positions,
        charges,
        cell,
        [True, True, True],
        source_sigma,
        receiver_sigmas,
        [36, 36, 40],
        6,
    )
    result = workspace.scalar_forward(
        *arguments,
        include_feature_self_interaction=True,
        include_energy_self_interaction=True,
    )
    translated_positions = positions.copy()
    translated_positions[0] += cell[0]
    translated = workspace.scalar_forward(
        translated_positions,
        *arguments[1:],
        include_feature_self_interaction=True,
        include_energy_self_interaction=True,
    )

    np.testing.assert_allclose(result["features"], reference_features, atol=3e-5)
    assert result["energy"] == pytest.approx(reference_energy, abs=3e-5)
    np.testing.assert_allclose(
        translated["features"], result["features"], atol=2e-11, rtol=0.0
    )
    assert translated["energy"] == pytest.approx(result["energy"], abs=2e-11)


@pytest.mark.parametrize(
    ("include_feature_self", "include_energy_self"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_polar_scalar_pme_self_policies_are_independent(
    native, include_feature_self, include_energy_self
):
    positions, charges, cell = _scalar_case()
    source_sigma = 0.6
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 0, source_sigma, 0, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(charges), dtype=np.int32)
    )["cache"]
    multipoles = charges[:, None]
    reference_features = direct.coulomb_features(cache, multipoles)
    if not include_feature_self:
        reference_features -= direct.feature_self_terms(multipoles)
    if include_energy_self:
        reference_energy = direct.coulomb_energy(cache, multipoles)[0]
    else:
        self_fields = direct.energy_self_fields(multipoles)
        reference_energy = direct.coulomb_energy(cache, multipoles, self_fields)[0]

    workspace = _scalar_workspace(native, receiver_widths=len(receiver_sigmas))
    result = workspace.scalar_forward(
        positions,
        charges,
        cell,
        [True, True, True],
        source_sigma,
        receiver_sigmas,
        [32, 32, 32],
        6,
        include_feature_self_interaction=include_feature_self,
        include_energy_self_interaction=include_energy_self,
    )

    np.testing.assert_allclose(result["features"], reference_features, atol=2e-5)
    assert result["energy"] == pytest.approx(reference_energy, abs=2e-5)


def test_polar_scalar_pme_self_terms_and_fft_plans_are_reused(native):
    positions, charges, cell = _scalar_case()
    source_sigma = 0.6
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 0, source_sigma, 0, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(charges), dtype=np.int32)
    )["cache"]
    multipoles = charges[:, None]
    reference_features = direct.coulomb_features(cache, multipoles)
    reference_features -= direct.feature_self_terms(multipoles)
    self_fields = direct.energy_self_fields(multipoles)
    reference_energy = direct.coulomb_energy(cache, multipoles, self_fields)[0]

    workspace = _scalar_workspace(native, receiver_widths=len(receiver_sigmas))
    arguments = (
        positions,
        charges,
        cell,
        [True, True, True],
        source_sigma,
        receiver_sigmas,
        [32, 32, 32],
        6,
    )
    first = workspace.scalar_forward(*arguments)
    second = workspace.scalar_forward(*arguments)
    usage = workspace.usage()

    np.testing.assert_allclose(first["features"], reference_features, atol=2e-5)
    np.testing.assert_allclose(second["features"], first["features"], atol=0.0)
    assert first["energy"] == pytest.approx(reference_energy, abs=2e-5)
    assert second["energy"] == pytest.approx(first["energy"], abs=0.0)
    assert usage["forward_plan_builds"] == 1
    assert usage["inverse_plan_builds"] == 1
    assert usage["forward_plan_reuses"] >= 1
    assert usage["inverse_plan_reuses"] >= 1


def test_polar_pme_reverse_reuses_forward_source_spectrum(native):
    positions, charges, cell = _scalar_case()
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = charges
    workspace = _workspace(native, receiver_widths=2)

    workspace.reverse_channels(
        positions,
        sources,
        np.ones((len(positions), 16), dtype=np.float64),
        np.ones(2, dtype=np.float64),
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [16, 16, 16],
        4,
    )
    usage = workspace.usage()

    # Reverse reconstructs the receiver fields and transforms the receiver
    # seeds. The deconvolved source spectrum from reconstruction is retained;
    # it must not require a third forward transform.
    assert usage["forward_plan_builds"] + usage["forward_plan_reuses"] == 3
    assert usage["inverse_plan_builds"] + usage["inverse_plan_reuses"] == 4


def test_polar_l1_pme_matches_direct_reciprocal_features_and_energy(native):
    positions, charges, cell = _scalar_case()
    multipoles = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05],
            [charges[1], -0.03, 0.09, -0.11],
            [charges[2], 0.08, 0.02, 0.06],
        ],
        dtype=np.float64,
    )
    source_sigma = 0.6
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 1, source_sigma, 1, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(charges), dtype=np.int32)
    )["cache"]
    reference_features = direct.coulomb_features(cache, multipoles)
    reference_energy = direct.coulomb_energy(cache, multipoles)[0]

    workspace = _workspace(native, receiver_widths=len(receiver_sigmas))
    result = workspace.forward(
        positions,
        multipoles,
        cell,
        [True, True, True],
        source_sigma,
        receiver_sigmas,
        [40, 40, 40],
        6,
        include_feature_self_interaction=True,
        include_energy_self_interaction=True,
    )

    np.testing.assert_allclose(result["features"], reference_features, atol=8e-5)
    assert result["energy"] == pytest.approx(reference_energy, abs=8e-5)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_polar_l1_pme_two_spin_channels_match_independent_forwards(native, precision):
    positions, charges, cell = _scalar_case()
    alpha = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05],
            [charges[1], -0.03, 0.09, -0.11],
            [charges[2], 0.08, 0.02, 0.06],
        ],
        dtype=np.float64,
    )
    beta = np.array(
        [
            [-0.17, 0.04, 0.01, -0.02],
            [0.31, -0.06, 0.02, 0.07],
            [-0.14, 0.03, -0.08, 0.05],
        ],
        dtype=np.float64,
    )
    spin_density = np.concatenate((alpha, beta), axis=1)
    receiver_sigmas = [0.5, 0.8]
    workspace = _workspace(
        native, precision=precision, receiver_widths=len(receiver_sigmas)
    )
    arguments = (
        positions,
        cell,
        [True, True, True],
        0.6,
        receiver_sigmas,
        [32, 32, 32],
        6,
    )

    expected_alpha = workspace.forward(positions, alpha, *arguments[1:])
    expected_beta = workspace.forward(positions, beta, *arguments[1:])
    actual = workspace.forward_channels(positions, spin_density, *arguments[1:])

    expected_features = np.concatenate(
        (expected_alpha["features"], expected_beta["features"]), axis=1
    )
    np.testing.assert_allclose(actual["features"], expected_features, atol=2e-11)
    np.testing.assert_allclose(
        actual["energies"],
        [expected_alpha["energy"], expected_beta["energy"]],
        atol=2e-11,
    )


def test_polar_l1_pme_resident_forward_matches_direct_oracle(native):
    positions, charges, cell = _scalar_case()
    spin_density = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05, -0.17, 0.04, 0.01, -0.02],
            [charges[1], -0.03, 0.09, -0.11, 0.31, -0.06, 0.02, 0.07],
            [charges[2], 0.08, 0.02, 0.06, -0.14, 0.03, -0.08, 0.05],
        ],
        dtype=np.float64,
    )
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 1, 0.6, 1, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(positions), dtype=np.int32)
    )["cache"]
    expected_features = np.concatenate(
        [
            direct.coulomb_features(cache, spin_density[:, :4]),
            direct.coulomb_features(cache, spin_density[:, 4:]),
        ],
        axis=1,
    )
    expected_energies = np.array(
        [
            direct.coulomb_energy(cache, spin_density[:, :4])[0],
            direct.coulomb_energy(cache, spin_density[:, 4:])[0],
        ]
    )
    workspace = _workspace(native, receiver_widths=len(receiver_sigmas))

    actual = workspace.forward_channels_device(
        positions,
        spin_density,
        cell,
        [True, True, True],
        0.6,
        receiver_sigmas,
        [40, 40, 40],
        6,
    )

    np.testing.assert_allclose(actual["features"], expected_features, atol=8e-5)
    np.testing.assert_allclose(actual["energies"], expected_energies, atol=8e-5)
    usage = workspace.usage()
    assert usage.get("resident_scalar_validation_transfers", -1) == 1
    assert usage.get("resident_completion_fences", -1) == 1


def test_polar_resident_forward_pads_scalar_receivers_to_source_lanes(native):
    positions, charges, cell = _scalar_case()
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = charges
    sources[:, 4] = -charges
    workspace = native._PolarParticleMeshWorkspace("float64", 1, 0, 2, 1)

    result = workspace.forward_channels_device(
        positions,
        sources,
        cell,
        [True, True, True],
        0.6,
        [0.5],
        [16, 16, 16],
        6,
    )

    assert result["features"].shape == (len(positions), 2)
    assert np.isfinite(result["features"]).all()
    assert workspace.usage()["transform_lanes"] == 8


def test_polar_resident_forward_rejects_bad_sources_before_allocation(native):
    positions, charges, cell = _scalar_case()
    workspace = _workspace(native, receiver_widths=2)
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = charges

    with pytest.raises(ValueError, match="source rows must match positions"):
        workspace.forward_channels_device(
            positions,
            sources[:-1],
            cell,
            [True, True, True],
            0.6,
            [0.5, 0.8],
            [16, 16, 16],
            4,
        )
    assert workspace.usage()["reservations"] == 0

    sources[1, 3] = np.nan
    with pytest.raises(ValueError, match="finite resident sources"):
        workspace.forward_channels_device(
            positions,
            sources,
            cell,
            [True, True, True],
            0.6,
            [0.5, 0.8],
            [16, 16, 16],
            4,
        )
    assert workspace.usage()["reservations"] == 0


@pytest.mark.parametrize(
    ("order", "grid_coordinate"),
    [(4, 3.0), (3, 3.5)],
    ids=("floor-boundary", "order-boundary"),
)
@pytest.mark.parametrize("direction", [-np.inf, 0.0, np.inf])
def test_polar_resident_boundary_stencil_stays_on_device_and_matches_safe_path(
    native, order, grid_coordinate, direction
):
    positions, charges, cell = _scalar_case()
    coordinate = grid_coordinate * cell[0, 0] / 16.0
    if direction != 0.0:
        coordinate = np.nextafter(coordinate, direction)
    positions[0, 0] = coordinate
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = charges
    workspace = _workspace(native, receiver_widths=2)

    expected = workspace.forward_channels(
        positions,
        sources,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [16, 16, 16],
        order,
        include_feature_self_interaction=True,
        include_energy_self_interaction=True,
    )
    actual = workspace.forward_channels_device(
        positions,
        sources,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [16, 16, 16],
        order,
    )

    np.testing.assert_allclose(actual["features"], expected["features"], atol=2e-11)
    np.testing.assert_allclose(actual["energies"], expected["energies"], atol=2e-11)
    assert (
        workspace.usage().get("resident_stencil_path") == "device_host_grid_coordinates"
    )


def test_polar_resident_triclinic_translations_match_safe_stencil_path(native):
    cell = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1e-8, 0.0], [0.25, -0.5, 1.0]],
        dtype=np.float64,
    )
    fractional = np.asarray(
        [
            [2e-8, 100000.1371, -2.1241],
            [0.1371, 0.2839, 0.4193],
            [1.0007, -1.4993, 2.0009],
        ],
        dtype=np.float64,
    )
    fractional += np.asarray([[1e6, -2e6, 3e6], [-4e5, 7e5, -9e5], [2e6, 3e6, -1e6]])
    positions = fractional @ cell
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = [0.7, -1.1, 0.4]
    workspace = _workspace(native, receiver_widths=2)
    arguments = (
        positions,
        sources,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [16, 16, 16],
        4,
    )

    expected = workspace.forward_channels(
        *arguments,
        include_feature_self_interaction=True,
        include_energy_self_interaction=True,
    )
    actual = workspace.forward_channels_device(*arguments)

    np.testing.assert_allclose(actual["features"], expected["features"], atol=2e-10)
    np.testing.assert_allclose(actual["energies"], expected["energies"], atol=2e-10)
    assert (
        workspace.usage().get("resident_stencil_path") == "device_host_grid_coordinates"
    )


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_polar_l1_pme_every_feature_channel_converges_monotonically(native, precision):
    positions, charges, cell = _scalar_case()
    multipoles = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05],
            [charges[1], -0.03, 0.09, -0.11],
            [charges[2], 0.08, 0.02, 0.06],
        ],
        dtype=np.float64,
    )
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 1, 0.6, 1, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(charges), dtype=np.int32)
    )["cache"]
    reference_features = direct.coulomb_features(cache, multipoles)
    reference_energy = direct.coulomb_energy(cache, multipoles)[0]
    workspace = _workspace(
        native, precision=precision, receiver_widths=len(receiver_sigmas)
    )
    feature_errors = [[] for _ in range(reference_features.shape[1])]
    energy_errors = []

    for extent in (12, 20, 32):
        result = workspace.forward(
            positions,
            multipoles,
            cell,
            [True, True, True],
            0.6,
            receiver_sigmas,
            [extent, extent, extent],
            6,
            include_feature_self_interaction=True,
            include_energy_self_interaction=True,
        )
        energy_errors.append(abs(result["energy"] - reference_energy))
        for feature in range(reference_features.shape[1]):
            feature_errors[feature].append(
                np.max(
                    np.abs(
                        result["features"][:, feature] - reference_features[:, feature]
                    )
                )
            )

    assert energy_errors[2] < energy_errors[1] < energy_errors[0]
    assert energy_errors[-1] < 2e-5
    for errors in feature_errors:
        assert errors[2] < errors[1] < errors[0]
        assert errors[-1] < 2e-5


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_polar_l1_pme_reverse_source_and_position_adjoints_match_finite_difference(
    native, precision
):
    positions, charges, cell = _scalar_case()
    sources = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05, -0.17, 0.04, 0.01, -0.02],
            [charges[1], -0.03, 0.09, -0.11, 0.31, -0.06, 0.02, 0.07],
            [charges[2], 0.08, 0.02, 0.06, -0.14, 0.03, -0.08, 0.05],
        ],
        dtype=np.float64,
    )
    receiver_sigmas = [0.5, 0.8]
    mesh = [18, 18, 18]
    order = 6
    workspace = _workspace(
        native, precision=precision, receiver_widths=len(receiver_sigmas)
    )
    finite_difference_workspace = (
        workspace
        if precision == "float64"
        else _workspace(
            native, precision="float64", receiver_widths=len(receiver_sigmas)
        )
    )
    rng = np.random.default_rng(7153)
    feature_seeds = rng.normal(size=(len(positions), 16))
    energy_seeds = np.array([0.37, -0.23], dtype=np.float64)

    def objective(current_positions, current_sources):
        result = finite_difference_workspace.forward_channels(
            current_positions,
            current_sources,
            cell,
            [True, True, True],
            0.6,
            receiver_sigmas,
            mesh,
            order,
            include_feature_self_interaction=False,
            include_energy_self_interaction=False,
        )
        return float(
            np.vdot(feature_seeds, result["features"])
            + np.vdot(energy_seeds, result["energies"])
        )

    reverse = workspace.reverse_channels(
        positions,
        sources,
        feature_seeds,
        energy_seeds,
        cell,
        [True, True, True],
        0.6,
        receiver_sigmas,
        mesh,
        order,
        include_feature_self_interaction=False,
        include_energy_self_interaction=False,
    )
    step = 2.0e-6
    source_direction = rng.normal(size=sources.shape)
    source_direction /= np.linalg.norm(source_direction)
    position_direction = rng.normal(size=positions.shape)
    position_direction -= np.mean(position_direction, axis=0, keepdims=True)
    position_direction /= np.linalg.norm(position_direction)
    source_fd = (
        objective(positions, sources + step * source_direction)
        - objective(positions, sources - step * source_direction)
    ) / (2.0 * step)
    position_fd = (
        objective(positions + step * position_direction, sources)
        - objective(positions - step * position_direction, sources)
    ) / (2.0 * step)
    source_reverse = float(
        np.vdot(reverse["source_multipole_adjoints"], source_direction)
    )
    position_reverse = float(np.vdot(reverse["position_adjoints"], position_direction))

    rtol, atol = (2e-6, 2e-7) if precision == "float64" else (2e-4, 2e-5)
    source_atol = atol * max(abs(source_fd), 1.0) if precision == "float32" else atol
    position_atol = (
        atol * max(abs(position_fd), 1.0) if precision == "float32" else atol
    )
    np.testing.assert_allclose(source_reverse, source_fd, rtol=rtol, atol=source_atol)
    np.testing.assert_allclose(
        position_reverse, position_fd, rtol=rtol, atol=position_atol
    )


@pytest.mark.parametrize(
    (
        "precision",
        "active_channels",
        "include_feature_self",
        "include_energy_self",
    ),
    (
        ("float32", 1, False, True),
        ("float64", 2, True, False),
    ),
)
def test_polar_l1_pme_resident_reverse_matches_diagnostic_reverse(
    native,
    precision,
    active_channels,
    include_feature_self,
    include_energy_self,
):
    positions, charges, cell = _scalar_case()
    sources = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05, -0.17, 0.04, 0.01, -0.02],
            [charges[1], -0.03, 0.09, -0.11, 0.31, -0.06, 0.02, 0.07],
            [charges[2], 0.08, 0.02, 0.06, -0.14, 0.03, -0.08, 0.05],
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(8971)
    source_lanes = active_channels * 4
    feature_lanes = active_channels * 8
    sources = sources[:, :source_lanes]
    feature_seeds = rng.normal(size=(len(positions), feature_lanes))
    energy_seeds = np.array([0.43, -0.19], dtype=np.float64)[:active_channels]
    host_workspace = _workspace(native, precision=precision, receiver_widths=2)
    resident_workspace = _workspace(native, precision=precision, receiver_widths=2)
    arguments = (
        positions,
        sources,
        feature_seeds,
        energy_seeds,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [18, 18, 18],
        6,
    )

    expected_forward = host_workspace.forward_channels(
        positions,
        sources,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [18, 18, 18],
        6,
        include_feature_self,
        include_energy_self,
        active_channels,
    )
    expected = host_workspace.reverse_channels(
        *arguments, include_feature_self, include_energy_self, active_channels
    )
    actual = resident_workspace.reverse_channels_device(
        *arguments,
        include_feature_self,
        include_energy_self,
        active_channels,
    )

    tolerance = 3e-5 if precision == "float32" else 2e-12
    np.testing.assert_allclose(
        actual["source_multipole_adjoints"],
        expected["source_multipole_adjoints"],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual["position_adjoints"],
        expected["position_adjoints"],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual["cell_adjoints"],
        expected["cell_adjoints"],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual["forward_features"],
        expected_forward["features"],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual["forward_energies"],
        expected_forward["energies"],
        rtol=tolerance,
        atol=tolerance,
    )


def test_polar_l0_pme_resident_reverse_matches_diagnostic_reverse(native):
    positions, charges, cell = _scalar_case()
    sources = charges[:, None]
    feature_seeds = np.array([[0.2, -0.3], [0.5, 0.7], [-0.4, 0.1]], dtype=np.float64)
    energy_seeds = np.array([0.37], dtype=np.float64)
    arguments = (
        positions,
        sources,
        feature_seeds,
        energy_seeds,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [18, 18, 18],
        6,
    )
    host_workspace = _scalar_workspace(native)
    resident_workspace = _scalar_workspace(native)

    expected = host_workspace.reverse_channels(*arguments, True, True, 1)
    actual = resident_workspace.reverse_channels_device(*arguments, True, True, 1)

    for name in ("source_multipole_adjoints", "position_adjoints", "cell_adjoints"):
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-10, atol=2e-10)


def test_polar_pme_resident_reverse_validates_before_snapshot_allocation(native):
    positions, charges, cell = _scalar_case()
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = charges
    sources[0, 0] = np.nan
    workspace = _workspace(native, receiver_widths=2)
    baseline = workspace.usage()
    workspace.arm_allocation_failure(0)
    try:
        with pytest.raises(ValueError, match="finite resident sources"):
            workspace.reverse_channels_device(
                positions,
                sources,
                np.ones((len(positions), 16), dtype=np.float64),
                np.ones(2, dtype=np.float64),
                cell,
                [True, True, True],
                0.6,
                [0.5, 0.8],
                [16, 16, 16],
                4,
            )
    finally:
        workspace.clear_allocation_failure()
    usage = workspace.usage()
    assert usage["reservations"] == baseline["reservations"]
    assert usage["active_bytes"] == baseline["active_bytes"]
    assert usage["rebuilds"] == baseline["rebuilds"]


def test_polar_pme_reverse_rejects_invalid_requests_before_snapshot_allocation(native):
    positions, charges, cell = _scalar_case()
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = charges
    feature_seeds = np.ones((len(positions), 16), dtype=np.float64)
    energy_seeds = np.ones(2, dtype=np.float64)
    workspace = _workspace(native, receiver_widths=2)
    baseline = workspace.usage()

    workspace.arm_allocation_failure(0)
    try:
        with pytest.raises(ValueError, match="fully periodic PBC TTT"):
            workspace.reverse_channels(
                positions,
                sources,
                feature_seeds,
                energy_seeds,
                cell,
                [True, True, False],
                0.6,
                [0.5, 0.8],
                [16, 16, 16],
                4,
            )
    finally:
        workspace.clear_allocation_failure()
    assert workspace.usage() == baseline

    invalid_sources = sources.copy()
    invalid_sources[0, 0] = np.nan
    workspace.arm_allocation_failure(0)
    try:
        with pytest.raises(ValueError, match="finite sources/multipoles"):
            workspace.reverse_channels(
                positions,
                invalid_sources,
                feature_seeds,
                energy_seeds,
                cell,
                [True, True, True],
                0.6,
                [0.5, 0.8],
                [16, 16, 16],
                4,
            )
    finally:
        workspace.clear_allocation_failure()
    assert workspace.usage() == baseline

    workspace.arm_allocation_failure(0)
    try:
        with pytest.raises(ValueError, match="axes must be at least"):
            workspace.reverse_channels(
                positions,
                sources,
                feature_seeds,
                energy_seeds,
                cell,
                [True, True, True],
                0.6,
                [0.5, 0.8],
                [1_000_000, 4, 4],
                6,
            )
    finally:
        workspace.clear_allocation_failure()
    assert workspace.usage() == baseline


def test_polar_pme_reverse_snapshot_failure_preserves_forward_and_reuses_storage(
    native,
):
    positions, charges, cell = _scalar_case()
    sources = np.zeros((len(positions), 8), dtype=np.float64)
    sources[:, 0] = charges
    feature_seeds = np.ones((len(positions), 16), dtype=np.float64)
    energy_seeds = np.ones(2, dtype=np.float64)
    workspace = _workspace(native, receiver_widths=2)
    arguments = (
        positions,
        sources,
        feature_seeds,
        energy_seeds,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [16, 16, 16],
        4,
    )
    workspace.forward_channels(
        positions,
        sources,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [16, 16, 16],
        4,
    )

    workspace.arm_allocation_failure(0)
    try:
        with pytest.raises(RuntimeError, match="injected.*source spectrum snapshot"):
            workspace.reverse_channels(*arguments)
    finally:
        workspace.clear_allocation_failure()
    assert "source_spectrum_snapshot" not in {
        item["name"] for item in workspace.usage()["buffers"]
    }
    workspace.forward_channels(
        positions,
        sources,
        cell,
        [True, True, True],
        0.6,
        [0.5, 0.8],
        [16, 16, 16],
        4,
    )

    workspace.reverse_channels(*arguments)
    first = {
        item["name"]: item["allocation_id"] for item in workspace.usage()["buffers"]
    }
    workspace.reverse_channels(*arguments)
    second = {
        item["name"]: item["allocation_id"] for item in workspace.usage()["buffers"]
    }
    assert first["source_spectrum_snapshot"] == second["source_spectrum_snapshot"]
    assert all("mode_by_atom" not in name for name in second)

    previous_usage = workspace.usage()
    larger_arguments = (*arguments[:-2], [20, 20, 20], arguments[-1])
    workspace.arm_allocation_failure(0)
    try:
        with pytest.raises(RuntimeError, match="injected.*source spectrum snapshot"):
            workspace.reverse_channels(*larger_arguments)
    finally:
        workspace.clear_allocation_failure()
    current_usage = workspace.usage()
    assert current_usage["mesh"] == previous_usage["mesh"]
    assert current_usage["generation"] == previous_usage["generation"]
    assert {
        item["name"]: item["allocation_id"] for item in current_usage["buffers"]
    } == {item["name"]: item["allocation_id"] for item in previous_usage["buffers"]}
    workspace.reverse_channels(*arguments)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_polar_l1_pme_reverse_cell_adjoints_match_fixed_fractional_difference(
    native, precision
):
    positions, charges, cell = _scalar_case()
    cell = np.array(
        [[8.0, 0.3, -0.2], [0.1, 7.7, 0.4], [-0.3, 0.2, 8.3]],
        dtype=np.float64,
    )
    fractional = positions @ np.linalg.inv(np.diag([8.0, 8.0, 8.0]))
    positions = fractional @ cell
    sources = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05, -0.17, 0.04, 0.01, -0.02],
            [charges[1], -0.03, 0.09, -0.11, 0.31, -0.06, 0.02, 0.07],
            [charges[2], 0.08, 0.02, 0.06, -0.14, 0.03, -0.08, 0.05],
        ],
        dtype=np.float64,
    )
    receiver_sigmas = [0.5, 0.8]
    workspace = _workspace(
        native, precision=precision, receiver_widths=len(receiver_sigmas)
    )
    finite_difference_workspace = (
        workspace
        if precision == "float64"
        else _workspace(
            native, precision="float64", receiver_widths=len(receiver_sigmas)
        )
    )
    rng = np.random.default_rng(4189)
    feature_seeds = rng.normal(size=(len(positions), 16))
    energy_seeds = np.array([0.29, -0.41], dtype=np.float64)
    arguments = (
        [True, True, True],
        0.6,
        receiver_sigmas,
        [18, 18, 18],
        6,
    )

    reverse = workspace.reverse_channels(
        positions,
        sources,
        feature_seeds,
        energy_seeds,
        cell,
        *arguments,
        include_feature_self_interaction=False,
        include_energy_self_interaction=False,
    )

    def objective(strain):
        deformation = np.eye(3) + strain
        result = finite_difference_workspace.forward_channels(
            positions @ deformation,
            sources,
            cell @ deformation,
            *arguments,
            include_feature_self_interaction=False,
            include_energy_self_interaction=False,
        )
        return float(
            np.vdot(feature_seeds, result["features"])
            + np.vdot(energy_seeds, result["energies"])
        )

    step = 1.0e-6
    direction = rng.normal(size=(3, 3))
    direction /= np.linalg.norm(direction)
    cell_fd = (objective(step * direction) - objective(-step * direction)) / (
        2.0 * step
    )
    cell_reverse = float(np.vdot(reverse["cell_adjoints"], direction))

    rtol, atol = (2e-6, 2e-7) if precision == "float64" else (2e-4, 2e-5)
    if precision == "float32":
        atol *= max(abs(cell_fd), 1.0)
    np.testing.assert_allclose(cell_reverse, cell_fd, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("include_feature_self", "include_energy_self"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_polar_l1_pme_self_policies_match_direct(
    native, include_feature_self, include_energy_self
):
    positions, charges, cell = _scalar_case()
    multipoles = np.array(
        [
            [charges[0], 0.12, -0.07, 0.05],
            [charges[1], -0.03, 0.09, -0.11],
            [charges[2], 0.08, 0.02, 0.06],
        ],
        dtype=np.float64,
    )
    receiver_sigmas = [0.5, 0.8]
    direct = native.PolarLongRangeOps(14.0, 1, 0.6, 1, receiver_sigmas)
    cache = direct.geometry_cache(
        positions, cell[None, :, :], np.zeros(len(charges), dtype=np.int32)
    )["cache"]
    reference_features = direct.coulomb_features(cache, multipoles)
    if not include_feature_self:
        reference_features -= direct.feature_self_terms(multipoles)
    if include_energy_self:
        reference_energy = direct.coulomb_energy(cache, multipoles)[0]
    else:
        self_fields = direct.energy_self_fields(multipoles)
        reference_energy = direct.coulomb_energy(cache, multipoles, self_fields)[0]

    workspace = _workspace(native, receiver_widths=len(receiver_sigmas))
    result = workspace.forward(
        positions,
        multipoles,
        cell,
        [True, True, True],
        0.6,
        receiver_sigmas,
        [32, 32, 32],
        6,
        include_feature_self_interaction=include_feature_self,
        include_energy_self_interaction=include_energy_self,
    )

    np.testing.assert_allclose(result["features"], reference_features, atol=2e-5)
    assert result["energy"] == pytest.approx(reference_energy, abs=2e-5)
