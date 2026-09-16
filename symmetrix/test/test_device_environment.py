import pytest


def _edge_records(receivers, sources, shifts):
    return sorted(
        (int(receiver), int(source), *(int(value) for value in shift))
        for receiver, source, shift in zip(receivers, sources, shifts)
    )


def test_jit_device_plugin_api_retains_backend_aliases():
    from symmetrix import symmetrix as native_symmetrix

    for evaluator_name in ("MACEKokkos", "MACEKokkosFloat"):
        evaluator = getattr(native_symmetrix, evaluator_name)
        for name in (
            "_load_jit_device_plugin",
            "jit_device_plugin_ready",
            "jit_device_plugin_path",
            "jit_device_plugin_artifact_id",
            "_load_jit_cuda_plugin",
            "jit_cuda_plugin_ready",
            "jit_cuda_plugin_path",
            "jit_cuda_plugin_artifact_id",
            "_load_jit_hip_plugin",
            "jit_hip_plugin_ready",
            "jit_hip_plugin_path",
            "jit_hip_plugin_artifact_id",
        ):
            assert hasattr(evaluator, name)


@pytest.mark.parametrize(
    ("raw", "architecture", "features"),
    [
        ("gfx942", "gfx942", ""),
        ("GFX942:sramecc+:xnack-", "gfx942", "sramecc+:xnack-"),
        ("gfx90a:xnack+:sramecc-", "gfx90a", "sramecc-:xnack+"),
    ],
)
def test_normalize_hip_agent_target(raw, architecture, features):
    import symmetrix

    normalized = symmetrix._normalize_hip_agent_target(raw)
    assert normalized["raw_agent_target"] == raw
    assert normalized["architecture"] == architecture
    assert normalized["target_features"] == features


def test_reject_non_hip_agent_target():
    import symmetrix

    with pytest.raises(ValueError, match="must begin with 'gfx'"):
        symmetrix._normalize_hip_agent_target("sm_90")


@pytest.mark.parametrize("l_max", range(7))
def test_portable_spherical_harmonics_match_sphericart(accelerator_capabilities, l_max):
    import symmetrix

    coordinates = [0.37, -0.41, 0.83]
    values, gradients = symmetrix._kokkos_spherical_harmonics_reference(
        coordinates, l_max
    )
    size = (l_max + 1) ** 2
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            index = l * l + l + m
            expected = symmetrix.sphericart_real_sph_harm(l, m, *coordinates)
            assert values[index] == pytest.approx(expected, abs=2e-13)

    epsilon = 1e-6
    for axis in range(3):
        lower = coordinates.copy()
        upper = coordinates.copy()
        lower[axis] -= epsilon
        upper[axis] += epsilon
        lower_values, _ = symmetrix._kokkos_spherical_harmonics_reference(lower, l_max)
        upper_values, _ = symmetrix._kokkos_spherical_harmonics_reference(upper, l_max)
        for index in range(size):
            finite_difference = (upper_values[index] - lower_values[index]) / (
                2 * epsilon
            )
            assert gradients[axis * size + index] == pytest.approx(
                finite_difference, abs=2e-9
            )


def test_portable_spherical_harmonics_zero_vector_is_finite(
    accelerator_capabilities,
):
    import math

    import symmetrix

    values, gradients = symmetrix._kokkos_spherical_harmonics_reference(
        [0.0, 0.0, 0.0], 6
    )
    assert values[0] == pytest.approx(0.282094791773878)
    assert values[1:] == pytest.approx([0.0] * 48)
    assert gradients == pytest.approx([0.0] * 147)
    assert all(math.isfinite(value) for value in values + gradients)


@pytest.mark.parametrize(
    "cell",
    [
        [[3.2, 0.0, 0.0], [0.0, 4.1, 0.0], [0.0, 0.0, 5.3]],
        [[3.2, 0.0, 0.0], [0.7, 4.1, 0.0], [-0.4, 0.6, 5.3]],
    ],
)
def test_kokkos_periodic_neighbor_graph_matches_matscipy(
    accelerator_capabilities, cell
):
    import numpy as np
    import symmetrix
    from ase import Atoms
    from matscipy.neighbours import neighbour_list

    atoms = Atoms(
        "H4",
        scaled_positions=[
            [0.02, 0.15, 0.35],
            [0.43, 0.21, 0.87],
            [0.78, 0.72, 0.04],
            [0.91, 0.47, 0.61],
        ],
        cell=cell,
        pbc=True,
    )
    atoms.positions[0] += np.asarray(cell)[0]
    cutoff = 4.7
    graph = symmetrix._kokkos_periodic_neighbor_graph(
        atoms.positions, atoms.cell.array, np.linalg.inv(atoms.cell.array), cutoff
    )
    repeated = symmetrix._kokkos_periodic_neighbor_graph(
        atoms.positions, atoms.cell.array, np.linalg.inv(atoms.cell.array), cutoff
    )
    receivers = np.repeat(np.arange(len(atoms)), graph["num_neigh"])
    expected_receivers, expected_sources, expected_shifts = neighbour_list(
        "ijS", atoms, cutoff
    )

    assert graph["receiver_offsets"][0] == 0
    assert graph["receiver_offsets"][-1] == len(graph["sources"])
    np.testing.assert_array_equal(
        np.diff(graph["receiver_offsets"]), graph["num_neigh"]
    )
    assert _edge_records(receivers, graph["sources"], graph["shifts"]) == (
        _edge_records(expected_receivers, expected_sources, expected_shifts)
    )
    np.testing.assert_array_equal(
        repeated["receiver_offsets"], graph["receiver_offsets"]
    )
    np.testing.assert_array_equal(repeated["sources"], graph["sources"])
    np.testing.assert_array_equal(repeated["shifts"], graph["shifts"])


def test_kokkos_periodic_neighbor_graph_handles_zero_edges(
    accelerator_capabilities,
):
    import numpy as np
    import symmetrix

    positions = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
    cell = np.eye(3) * 10.0
    graph = symmetrix._kokkos_periodic_neighbor_graph(
        positions, cell, np.linalg.inv(cell), 0.5
    )

    assert graph["sources"].size == 0
    assert graph["shifts"].shape == (0, 3)
    np.testing.assert_array_equal(graph["num_neigh"], [0, 0])
    np.testing.assert_array_equal(graph["receiver_offsets"], [0, 0, 0])

    empty_graph = symmetrix._kokkos_periodic_neighbor_graph(
        np.empty((0, 3)), cell, np.linalg.inv(cell), 0.5
    )
    assert empty_graph["sources"].size == 0
    assert empty_graph["shifts"].shape == (0, 3)
    np.testing.assert_array_equal(empty_graph["num_neigh"], [])
    np.testing.assert_array_equal(empty_graph["receiver_offsets"], [0])


def test_kokkos_periodic_neighbor_graph_rejects_invalid_inputs(
    accelerator_capabilities,
):
    import numpy as np
    import symmetrix

    cell = np.eye(3) * 4.0
    with pytest.raises(ValueError, match="cutoff must be finite and positive"):
        symmetrix._kokkos_periodic_neighbor_graph(
            np.zeros((1, 3)), cell, np.linalg.inv(cell), 0.0
        )
    with pytest.raises(ValueError, match="cell and inverse cell are inconsistent"):
        symmetrix._kokkos_periodic_neighbor_graph(
            np.zeros((1, 3)), cell, np.zeros((3, 3)), 1.0
        )
