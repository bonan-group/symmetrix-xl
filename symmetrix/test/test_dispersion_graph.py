import subprocess
import sys
import textwrap

import numpy as np
import pytest

import symmetrix


def _native_graph():
    if not symmetrix._kokkos_is_initialized():
        symmetrix._init_kokkos()
    return symmetrix._DispersionGraphKokkos()


def test_export_before_prepare_raises_without_crashing_fresh_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import symmetrix

                symmetrix._init_kokkos()
                graph = symmetrix._DispersionGraphKokkos()
                try:
                    graph.export()
                except RuntimeError as error:
                    assert "prepared" in str(error)
                else:
                    raise AssertionError("unprepared export did not fail")
                """
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_update_before_prepare_is_rejected():
    graph = _native_graph()

    with pytest.raises(RuntimeError, match="prepared"):
        graph.update_geometry(np.empty((0, 3)), np.eye(3), np.eye(3))


def test_periodic_prepare_exports_device_built_geometry():
    graph = _native_graph()
    cell = np.eye(3) * 4.0
    positions = np.array([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0], [3.9, 0.0, 0.0]])

    graph.prepare_periodic(
        np.array([1, 6, 8]), positions, cell, np.linalg.inv(cell), 0.5
    )

    exported = graph.export()
    diagnostics = graph.diagnostics()
    np.testing.assert_array_equal(exported["atomic_numbers"], [1, 6, 8])
    np.testing.assert_array_equal(exported["receiver_offsets"], [0, 2, 4, 6])
    np.testing.assert_array_equal(exported["sources"], [2, 1, 2, 0, 0, 1])
    np.testing.assert_array_equal(
        exported["shifts"],
        [[-1, 0, 0], [0, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
    )
    np.testing.assert_allclose(
        exported["edge_vectors"],
        [
            [-0.2, 0, 0],
            [0.2, 0, 0],
            [-0.4, 0, 0],
            [-0.2, 0, 0],
            [0.2, 0, 0],
            [0.4, 0, 0],
        ],
        atol=1e-15,
    )
    np.testing.assert_allclose(exported["distances"], [0.2, 0.2, 0.4, 0.2, 0.2, 0.4])
    assert diagnostics["graph_backend"] == "kokkos"
    assert diagnostics["generation"] == 1
    assert diagnostics["geometry_update_count"] == 1
    assert diagnostics["num_nodes"] == 3
    assert diagnostics["num_edges"] == 6
    assert diagnostics["build_cutoff"] == pytest.approx(0.5)


def test_host_prepare_and_geometry_update_reconstruct_edge_vectors():
    graph = _native_graph()
    cell = np.diag([5.0, 6.0, 7.0])
    graph.prepare_host(
        np.array([6, 8]),
        np.array([0, 2, 3]),
        np.array([1, 1, 0]),
        np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0]]),
        np.array([[0.2, 0.4, 0.6], [4.7, 0.9, 1.1]]),
        cell,
        5.5,
    )

    initial = graph.export()
    np.testing.assert_array_equal(initial["receivers"], [0, 0, 1])
    np.testing.assert_allclose(
        initial["edge_vectors"],
        [[4.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [0.5, -0.5, -0.5]],
    )
    np.testing.assert_allclose(
        initial["distances"], [np.sqrt(20.75), np.sqrt(0.75), np.sqrt(0.75)]
    )

    updated_cell = np.diag([6.0, 6.0, 7.0])
    graph.update_geometry(
        np.array([[0.5, 0.4, 0.6], [5.1, 1.0, 1.3]]),
        updated_cell,
        np.linalg.inv(updated_cell),
    )

    updated = graph.export()
    np.testing.assert_allclose(
        updated["edge_vectors"],
        [[4.6, 0.6, 0.7], [-1.4, 0.6, 0.7], [1.4, -0.6, -0.7]],
    )
    assert graph.diagnostics()["generation"] == 1
    assert graph.diagnostics()["geometry_update_count"] == 2


def test_reprepare_smaller_graph_retains_grow_only_capacity():
    graph = _native_graph()
    cell = np.eye(3) * 10.0
    graph.prepare_host(
        np.array([1, 6, 8]),
        np.array([0, 2, 4, 6]),
        np.array([1, 2, 0, 2, 0, 1]),
        np.zeros((6, 3), dtype=np.int64),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cell,
        4.0,
    )
    large = graph.diagnostics()

    graph.prepare_host(
        np.array([1, 1]),
        np.array([0, 1, 2]),
        np.array([1, 0]),
        np.zeros((2, 3), dtype=np.int64),
        np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        cell,
        2.0,
    )
    smaller = graph.diagnostics()

    assert smaller["generation"] == 2
    assert smaller["num_nodes"] == 2
    assert smaller["num_edges"] == 2
    assert smaller["node_capacity"] == large["node_capacity"]
    assert smaller["edge_capacity"] == large["edge_capacity"]
    assert smaller["offset_capacity"] == large["offset_capacity"]
    assert graph.active_extents() == {
        "atomic_numbers": 2,
        "receiver_offsets": 3,
        "receivers": 2,
        "sources": 2,
        "shifts": 2,
        "positions": 2,
        "cell": 3,
        "edge_vectors": 2,
        "distances": 2,
    }


def test_empty_host_graph_is_prepared_and_exportable():
    graph = _native_graph()
    graph.prepare_host(
        np.array([], dtype=np.int64),
        np.array([0]),
        np.array([], dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3)),
        np.eye(3),
        3.0,
    )

    exported = graph.export()
    assert exported["atomic_numbers"].shape == (0,)
    assert exported["receiver_offsets"].tolist() == [0]
    assert exported["sources"].shape == (0,)
    assert exported["shifts"].shape == (0, 3)
    assert exported["edge_vectors"].shape == (0, 3)
    assert exported["distances"].shape == (0,)
    np.testing.assert_array_equal(exported["cell"], np.eye(3))
    assert graph.diagnostics()["generation"] == 1


def test_empty_periodic_graph_is_prepared_and_exportable():
    graph = _native_graph()
    graph.prepare_periodic(
        np.array([], dtype=np.int64),
        np.empty((0, 3)),
        np.eye(3),
        np.eye(3),
        3.0,
    )

    exported = graph.export()
    assert exported["atomic_numbers"].shape == (0,)
    assert exported["receiver_offsets"].tolist() == [0]
    assert exported["sources"].shape == (0,)
    assert exported["shifts"].shape == (0, 3)
    assert exported["edge_vectors"].shape == (0, 3)
    assert exported["distances"].shape == (0,)
    np.testing.assert_array_equal(exported["cell"], np.eye(3))
    assert graph.diagnostics()["generation"] == 1


@pytest.mark.parametrize(
    ("numbers", "offsets", "sources", "shifts", "positions", "message"),
    [
        ([1, 8], [0, 1], [1], [[0, 0, 0]], [[0, 0, 0], [1, 0, 0]], "receiver offsets"),
        ([1, 8], [0, 2, 1], [1], [[0, 0, 0]], [[0, 0, 0], [1, 0, 0]], "nondecreasing"),
        ([1, 8], [0, 1, 1], [2], [[0, 0, 0]], [[0, 0, 0], [1, 0, 0]], "source index"),
        ([1, 8], [0, 1, 1], [1], [], [[0, 0, 0], [1, 0, 0]], "shift"),
        ([1, 8], [0, 1, 1], [1], [[0, 0, 0]], [[0, 0, 0]], "positions"),
        ([1, 95], [0, 1, 1], [1], [[0, 0, 0]], [[0, 0, 0], [1, 0, 0]], "atomic number"),
    ],
)
def test_host_prepare_rejects_malformed_graphs(
    numbers, offsets, sources, shifts, positions, message
):
    graph = _native_graph()
    with pytest.raises((ValueError, OverflowError), match=message):
        graph.prepare_host(
            np.asarray(numbers, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            np.asarray(sources, dtype=np.int64),
            np.asarray(shifts, dtype=np.int64),
            np.asarray(positions, dtype=float),
            np.eye(3),
            3.0,
        )


def test_host_prepare_rejects_32_bit_edge_count_before_allocation():
    graph = _native_graph()
    with pytest.raises(OverflowError, match="32-bit edge indexing"):
        graph.prepare_host(
            np.array([1]),
            np.array([0, np.iinfo(np.int32).max + 1], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            np.zeros((1, 3)),
            np.eye(3),
            3.0,
        )


@pytest.mark.parametrize(
    ("positions", "cell", "inverse_cell", "message"),
    [
        (np.zeros((1, 2)), np.eye(3), np.eye(3), "positions"),
        (np.zeros((1, 3)), np.ones((2, 3)), np.eye(3), "cell"),
        (np.zeros((1, 3)), np.eye(3), np.eye(2), "inverse cell"),
    ],
)
def test_update_geometry_rejects_mismatched_extents(
    positions, cell, inverse_cell, message
):
    graph = _native_graph()
    graph.prepare_host(
        np.array([1]),
        np.array([0, 0]),
        np.array([], dtype=np.int64),
        np.empty((0, 3), dtype=np.int64),
        np.zeros((1, 3)),
        np.eye(3),
        3.0,
    )
    with pytest.raises(ValueError, match=message):
        graph.update_geometry(positions, cell, inverse_cell)
