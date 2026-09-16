# This file was written and publicly released by Dr. Noam Bernstein as part of his
# work for the U. S. Government, and is not subject to copyright.

import json
import os

import numpy as np
import pytest
import symmetrix.calculator as calculator_module
from ase import units
from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.md.npt import NPT
from ase.stress import full_3x3_to_voigt_6_stress

try:
    from symmetrix import FieldAwareCalculator, FieldContributionCalculator, Symmetrix
except ModuleNotFoundError as exc:
    if "No module named 'symmetrix.symmetrix'" in str(exc):
        raise RuntimeError(
            "Can't import symmetrix.symmetrix, probably need to run pytest in venv "
            "and install version to be tested with "
            "'(cd /path/to/repo && python3 -m pip install -e .)'"
        ) from exc
    else:
        raise

try:
    import mace
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import download_mace_mp_checkpoint
except ImportError:
    mace = None


try:
    from mace.modules.extensions import MACEField as _MACEField
except (ImportError, AttributeError):
    mace_field_available = False
else:
    mace_field_available = _MACEField is not None

from model_downloads import test_model_cache_dir


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allow_fixed_workspace": 1}, "allow_fixed_workspace must be a bool"),
        (
            {"allow_fixed_workspace": True, "use_kokkos": False},
            "allow_fixed_workspace=True requires use_kokkos=True",
        ),
        (
            {"allow_fixed_workspace": True, "streamed_edges": "materialized"},
            "allow_fixed_workspace=True requires streamed_edges='direct'",
        ),
        (
            {"allow_fixed_workspace": True, "execution_profile": "speed"},
            "allow_fixed_workspace=True requires execution_profile='capacity'",
        ),
    ],
)
def test_fixed_workspace_opt_in_validation(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        Symmetrix(tmp_path / "unused.json", **kwargs)


def test_python_graph_cardinality_guard_rejects_unrepresentable_topology():
    from symmetrix.calculator import _validate_int32_graph_cardinality

    _validate_int32_graph_cardinality(1_000_188, 109_020_492)
    with pytest.raises(ValueError, match="atom count.*32-bit CSR limit"):
        _validate_int32_graph_cardinality(2**31 - 1, 0)
    with pytest.raises(ValueError, match="directed-edge count.*32-bit topology"):
        _validate_int32_graph_cardinality(1, 2**31)
    with pytest.raises(ValueError, match="must be non-negative"):
        _validate_int32_graph_cardinality(-1, 0)


def test_mace_inputs_runs_native_model_cardinality_guard(monkeypatch):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    guarded = []

    class Evaluator:
        atomic_numbers = [1]

        def _validate_graph_cardinality(self, receivers, feature_nodes, edges):
            guarded.append((receivers, feature_nodes, edges))

    calculator = Symmetrix.__new__(Symmetrix)
    calculator.evaluator = Evaluator()
    calculator.neighbor_skin = 0.0
    calculator.cutoff = 2.0
    monkeypatch.setattr(
        "symmetrix.calculator.neighbor_list",
        lambda *_args: (
            np.array([0, 1], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
            np.array([1.0, 1.0]),
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        ),
    )

    calculator._mace_inputs(atoms)

    assert guarded == [(2, 2, 2)]


def test_compute_mace_uses_prepared_execution_capability_and_preserves_fallback():
    inputs = (
        2,
        [0, 0],
        np.array([1, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        [0, 0],
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
        np.array([0, 1], dtype=np.int32),
    )

    class PreparedEvaluator:
        streamed_edges_mode = "direct"

        def __init__(self):
            self.prepared = []
            self.evaluated = []

        def _prepare_factorized_graph(self, *topology):
            self.prepared.append(topology)
            return 17

        def _compute_prepared_factorized(self, token, xyz, distances):
            self.evaluated.append((token, xyz.copy(), distances.copy()))

        def compute_node_energies_forces(self, *args):
            raise AssertionError(
                "prepared Execution must not use the broad entry point"
            )

    calculator = object.__new__(Symmetrix)
    calculator.evaluator = PreparedEvaluator()
    calculator._compute_mace(inputs)
    assert len(calculator.evaluator.prepared) == 1
    assert all(
        actual is expected
        for actual, expected in zip(calculator.evaluator.prepared[0], inputs[:5])
    )
    token, xyz, distances = calculator.evaluator.evaluated[0]
    assert token == 17
    assert xyz.flags.c_contiguous and xyz.shape == (6,)
    assert distances.flags.c_contiguous and distances.shape == (2,)
    invalid_geometry = list(inputs)
    invalid_geometry[6] = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="invalid distance"):
        calculator._compute_mace(tuple(invalid_geometry))
    invalid_geometry[6] = inputs[6]
    invalid_geometry[5] = np.array(inputs[5], copy=True)
    invalid_geometry[5][0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid distance"):
        calculator._compute_mace(tuple(invalid_geometry))
    assert len(calculator.evaluator.prepared) == 1

    class FallbackEvaluator:
        streamed_edges_mode = "generic"

        def __init__(self):
            self.calls = []

        def _prepare_factorized_graph(self, *topology):
            raise AssertionError("inactive Execution must not prepare a graph")

        def _compute_prepared_factorized(self, *args):
            raise AssertionError("inactive Execution must not use a graph token")

        def compute_node_energies_forces(self, *args):
            self.calls.append(args)

    calculator.evaluator = FallbackEvaluator()
    calculator._compute_mace(inputs)
    assert len(calculator.evaluator.calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(calculator.evaluator.calls[0][0:5], inputs[:5])
    )

    class FailingPreparedEvaluator(PreparedEvaluator):
        def _prepare_factorized_graph(self, *topology):
            raise RuntimeError("preparation failed")

    calculator.evaluator = FailingPreparedEvaluator()
    with pytest.raises(RuntimeError, match="preparation failed"):
        calculator._compute_mace(inputs)


def _bare_neighbor_cache_calculator(cutoff=2.5, skin=0.4):
    calculator = object.__new__(Symmetrix)
    calculator.cutoff = cutoff
    calculator.neighbor_skin = skin
    calculator._neighbor_cache = None
    calculator.neighbor_cache_build_count = 0
    calculator.neighbor_cache_reuse_count = 0
    calculator.neighbor_cache_geometry_update_count = 0
    calculator.neighbor_cache_host_geometry_materialization_count = 0
    calculator._execution_native_geometry_identity = None
    calculator._execution_native_cell_identity = None
    calculator._native_model_type = "MACE"
    calculator._all_interactions_native_geometry_identity = None
    calculator._all_interactions_graph_identity = None
    calculator._all_interactions_graph_generation = 0
    calculator.evaluator = type("Evaluator", (), {"atomic_numbers": [1]})()
    return calculator


def test_receiver_major_neighbor_arrays_preserve_provider_order():
    receivers = np.array([0, 0, 1, 1], dtype=np.int64)
    sources = np.array([3, 2, 1, 0], dtype=np.int64)
    shifts = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]])

    actual = calculator_module._receiver_major_neighbor_arrays(
        receivers, sources, shifts
    )

    np.testing.assert_array_equal(actual[0], receivers)
    np.testing.assert_array_equal(actual[1], sources)
    np.testing.assert_array_equal(actual[2], shifts)
    assert all(value.dtype == np.int32 for value in actual)
    assert all(value.flags.c_contiguous for value in actual)


def test_receiver_major_neighbor_arrays_stably_group_alternative_provider():
    receivers = np.array([1, 0, 1, 0])
    sources = np.array([10, 20, 11, 21])
    shifts = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])

    actual = calculator_module._receiver_major_neighbor_arrays(
        receivers, sources, shifts
    )

    np.testing.assert_array_equal(actual[0], [0, 0, 1, 1])
    np.testing.assert_array_equal(actual[1], [20, 21, 10, 11])
    np.testing.assert_array_equal(actual[2][:, 0], [2, 4, 1, 3])


def test_receiver_major_neighbor_arrays_support_canonical_debug_order(monkeypatch):
    monkeypatch.setenv("SYMMETRIX_DEBUG_NEIGHBOR_EDGE_ORDER", "canonical")
    receivers = np.array([0, 0, 0])
    sources = np.array([2, 1, 1])
    shifts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])

    actual = calculator_module._receiver_major_neighbor_arrays(
        receivers, sources, shifts
    )

    np.testing.assert_array_equal(actual[1], [1, 1, 2])
    np.testing.assert_array_equal(actual[2][:, 0], [-1, 1, 0])


def test_receiver_major_neighbor_arrays_reject_misaligned_or_unknown_policy(
    monkeypatch,
):
    with pytest.raises(ValueError, match="extents must match"):
        calculator_module._receiver_major_neighbor_arrays(
            [0, 1], [1], np.zeros((2, 3), dtype=int)
        )

    monkeypatch.setenv("SYMMETRIX_DEBUG_NEIGHBOR_EDGE_ORDER", "invalid")
    with pytest.raises(ValueError, match="SYMMETRIX_DEBUG_NEIGHBOR_EDGE_ORDER"):
        calculator_module._receiver_major_neighbor_arrays(
            [0], [1], np.zeros((1, 3), dtype=int)
        )


def test_neighbor_cache_reuses_ordered_candidates_until_skin_threshold():
    atoms = Atoms(
        "H3",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.2, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator()

    initial = calculator._mace_inputs(atoms)
    atoms.positions[1, 1] += 0.05
    reused = calculator._mace_inputs(atoms)

    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert np.array_equal(initial[3], reused[3])
    assert np.array_equal(initial[7], reused[7])
    assert np.allclose(reused[6], np.linalg.norm(reused[5], axis=1))

    atoms.positions[1, 1] += 0.16
    calculator._mace_inputs(atoms)
    assert calculator.neighbor_cache_build_count == 2


def test_periodic_direct_neighbor_cache_stays_device_resident(monkeypatch):
    atoms = Atoms(
        "H2",
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.2, 0.3]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )

    class DeviceEvaluator:
        atomic_numbers = (1,)
        streamed_edges_mode = "direct"

        def __init__(self):
            self.graph_calls = []
            self.position_calls = []

        def _prepare_periodic_factorized_graph(self, *args):
            self.graph_calls.append(args)
            return {"generation": 23, "num_edges": 14}

        def _prepare_factorized_graph(self, *_args):
            raise AssertionError("device topology must not be prepared from host edges")

        def _compute_prepared_factorized(self, *_args):
            raise AssertionError("device geometry must use the position entry point")

        def _compute_prepared_factorized_positions(self, generation, positions):
            self.position_calls.append((generation, positions.copy()))

        def _update_factorized_cell(self, *_args):
            raise AssertionError("an unchanged cell must not be uploaded again")

    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.5)
    calculator.evaluator = DeviceEvaluator()
    calculator._sync_low_memory_policy_state = lambda: None
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(
        calculator_module,
        "neighbor_list",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Matscipy must not build an eligible direct graph")
        ),
    )

    inputs = calculator._mace_inputs(atoms, native_geometry=True)
    assert calculator._neighbor_cache.device_graph_generation == 23
    assert calculator._neighbor_cache.num_edges == 14
    assert len(inputs[3]) == len(inputs[7]) == 0
    calculator._compute_mace(inputs, atoms=atoms)
    assert calculator.evaluator.position_calls[0][0] == 23

    atoms.positions[0, 1] += 0.05
    reused = calculator._mace_inputs(atoms, native_geometry=True)
    calculator._compute_mace(reused, atoms=atoms)
    assert len(calculator.evaluator.graph_calls) == 1
    assert len(calculator.evaluator.position_calls) == 2
    assert calculator.neighbor_cache_reuse_count == 1


@pytest.mark.parametrize(
    ("backend_request", "eligible", "num_atoms", "execution_space", "expected"),
    (
        ("automatic", True, 511, "Cuda", False),
        ("automatic", True, 512, "Cuda", True),
        ("automatic", True, 512, "HIP", True),
        ("automatic", True, 20_480, "OpenMP", False),
        ("automatic", True, 20_480, "Serial", False),
        ("automatic", False, 10_000, "Cuda", False),
        ("host", True, 10_000, "Cuda", False),
        ("kokkos", True, 1, "OpenMP", True),
    ),
)
def test_device_neighbor_graph_automatic_size_policy(
    backend_request, eligible, num_atoms, execution_space, expected
):
    assert (
        calculator_module._use_device_neighbor_graph(
            backend_request, eligible, num_atoms, execution_space
        )
        is expected
    )


def test_single_layer_neighbor_cache_can_stay_device_resident(monkeypatch):
    atoms = Atoms(
        "H2",
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.2, 0.3]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )

    class DeviceEvaluator:
        atomic_numbers = (1,)
        streamed_edges_mode = "direct"

        def __init__(self):
            self.graph_calls = []

        def _prepare_periodic_factorized_graph(self, *args):
            self.graph_calls.append(args)
            return {"generation": 29, "num_edges": 2}

    calculator = _bare_neighbor_cache_calculator(cutoff=2.1, skin=0.4)
    calculator.evaluator = DeviceEvaluator()
    calculator._model_single_layer_readout = True
    calculator._sync_low_memory_policy_state = lambda: None
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    monkeypatch.setattr(
        calculator_module,
        "neighbor_list",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("host connectivity must not build a forced device graph")
        ),
    )
    inputs = calculator._mace_inputs(atoms, native_geometry=True)

    assert len(calculator.evaluator.graph_calls) == 1
    assert calculator.neighbor_graph_backend == "kokkos"
    assert calculator._neighbor_cache.device_graph_generation == 29
    assert calculator._neighbor_cache.num_edges == 2
    assert len(inputs[3]) == len(inputs[7]) == 0


def test_device_neighbor_cache_falls_back_to_host_edges_when_requested(monkeypatch):
    atoms = Atoms(
        "H2",
        scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.2, 0.3]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )

    class DeviceEvaluator:
        atomic_numbers = (1,)
        streamed_edges_mode = "direct"

        def _prepare_periodic_factorized_graph(self, *_args):
            return {"generation": 7, "num_edges": 2}

    calculator = _bare_neighbor_cache_calculator(cutoff=2.1, skin=0.4)
    calculator.evaluator = DeviceEvaluator()
    calculator._sync_low_memory_policy_state = lambda: None
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")
    calculator._mace_inputs(atoms, native_geometry=True)
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "automatic")

    host_calls = []

    def host_neighbor_list(*_args, **_kwargs):
        host_calls.append(True)
        return (
            np.array([0, 1], dtype=np.int32),
            np.array([1, 0], dtype=np.int32),
            np.zeros((2, 3), dtype=np.int32),
        )

    monkeypatch.setattr(calculator_module, "neighbor_list", host_neighbor_list)
    inputs = calculator._mace_inputs(
        atoms,
        native_geometry=True,
        allow_device_neighbor_graph=False,
    )

    assert host_calls == [True]
    assert calculator._neighbor_cache.device_graph_generation == 0
    np.testing.assert_array_equal(inputs[3], [1, 0])


def test_required_device_neighbor_backend_rejects_zero_skin(monkeypatch):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    class Evaluator:
        atomic_numbers = (1,)

    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.0)
    calculator.evaluator = Evaluator()
    monkeypatch.setenv("SYMMETRIX_NEIGHBOR_BACKEND", "kokkos")

    with pytest.raises(RuntimeError, match="requires a positive neighbor_skin"):
        calculator._mace_inputs(atoms)


def test_neighbor_cache_reuses_gentle_affine_cell_changes():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator()
    initial = calculator._mace_inputs(atoms)

    atoms.set_cell([7.9, 8.0, 8.0], scale_atoms=True)
    deformed = calculator._mace_inputs(atoms)

    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert calculator.neighbor_cache_geometry_update_count == 2
    np.testing.assert_array_equal(deformed[3], initial[3])
    np.testing.assert_array_equal(deformed[7], initial[7])


def test_neighbor_cache_rebuilds_for_excessive_cell_change_and_composition():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator()
    calculator._mace_inputs(atoms)

    atoms.set_cell([6.0, 8.0, 8.0], scale_atoms=True)
    calculator._mace_inputs(atoms)
    assert calculator.neighbor_cache_build_count == 2

    atoms.numbers[1] = 2
    calculator.evaluator.atomic_numbers = [1, 2]
    calculator._mace_inputs(atoms)
    assert calculator.neighbor_cache_build_count == 3


def test_neighbor_cache_cell_reuse_accumulates_from_topology_reference():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator(cutoff=2.5, skin=0.4)
    calculator._mace_inputs(atoms)

    for length in (7.9, 7.8, 7.7):
        atoms.set_cell([length, 8.0, 8.0], scale_atoms=True)
        calculator._mace_inputs(atoms)

    assert calculator.neighbor_cache_build_count == 1
    atoms.set_cell([6.5, 8.0, 8.0], scale_atoms=True)
    calculator._mace_inputs(atoms)
    assert calculator.neighbor_cache_build_count == 2


def test_neighbor_cache_handles_periodic_position_wrapping_without_rebuild():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator()
    initial = calculator._mace_inputs(atoms)

    atoms.positions[1, 0] -= 10.0
    wrapped = calculator._mace_inputs(atoms)

    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert np.array_equal(initial[3], wrapped[3])
    assert np.array_equal(initial[7], wrapped[7])
    assert np.allclose(initial[5], wrapped[5])


def test_execution_neighbor_cache_retains_inactive_candidates_at_zero_cutoff():
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    calculator = _bare_neighbor_cache_calculator(cutoff=1.0, skin=0.4)
    calculator.evaluator.streamed_edges_mode = "direct"
    active = calculator._mace_inputs(atoms)

    atoms.positions[1, 0] = 1.05
    inactive = calculator._mace_inputs(atoms)

    assert calculator.neighbor_cache_build_count == 1
    assert np.array_equal(active[3], inactive[3])
    assert np.array_equal(active[7], inactive[7])
    assert np.allclose(inactive[6], 1.0)
    assert np.allclose(np.linalg.norm(inactive[5], axis=1), 1.0)


def test_native_execution_geometry_prepares_reference_once_and_submits_positions():
    class NativeGeometryEvaluator:
        streamed_edges_mode = "direct"

        def __init__(self):
            self.atomic_numbers = [1]
            self.graphs = []
            self.geometry = []
            self.positions = []

        def _prepare_factorized_graph(self, *topology):
            self.graphs.append(topology)
            return 17

        def _prepare_factorized_geometry(self, token, *geometry):
            self.geometry.append((token, geometry))

        def _compute_prepared_factorized_positions(self, token, positions):
            self.positions.append((token, positions.copy()))

        def _reduce_stress(self, volume, xyz, token):
            return np.zeros(9)

        def _compute_prepared_factorized(self, *args):
            raise AssertionError("native geometry must not submit host edge vectors")

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.4)
    calculator.evaluator = NativeGeometryEvaluator()
    assert calculator._can_use_native_geometry(["stress"])

    inputs = calculator._mace_inputs(atoms, native_geometry=True)
    assert inputs[5] is None and inputs[6] is None
    calculator._compute_mace(inputs, atoms=atoms)
    assert len(calculator.evaluator.geometry) == 1
    assert len(calculator.evaluator.positions) == 1
    token, geometry = calculator.evaluator.geometry[0]
    assert token == 17
    assert geometry[0].shape == (6,)
    assert geometry[1].shape == (6,)
    assert geometry[2].shape == (9,)
    assert geometry[3].shape == (9,)
    assert geometry[4].shape == (3,)

    atoms.positions[1, 0] -= 10.0
    reused = calculator._mace_inputs(atoms, native_geometry=True)
    assert reused[1] is inputs[1]
    assert reused[2] is inputs[2]
    assert reused[4] is inputs[4]
    calculator._compute_mace(reused, atoms=atoms)
    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert len(calculator.evaluator.geometry) == 1
    assert len(calculator.evaluator.positions) == 2
    np.testing.assert_allclose(
        calculator.evaluator.positions[-1][1], atoms.positions.reshape(-1)
    )


def test_native_macefield_geometry_submits_positions_and_field():
    class NativeFieldGeometryEvaluator:
        atomic_numbers = (1,)
        streamed_edges_mode = "direct"
        has_field_coupling = True

        def __init__(self):
            self.geometry = []
            self.positions_and_fields = []

        def _prepare_factorized_graph(self, *topology):
            return 31

        def _prepare_factorized_geometry(self, token, *geometry):
            self.geometry.append((token, geometry))

        def _compute_prepared_factorized_positions_field(
            self, token, positions, electric_field
        ):
            self.positions_and_fields.append(
                (token, positions.copy(), electric_field.copy())
            )

        def _compute_prepared_factorized_field_response(self, *args):
            raise AssertionError("response evaluation is outside this routing test")

        def _compute_prepared_factorized_field(self, *args):
            raise AssertionError("native field geometry must not submit edge arrays")

        def _reduce_stress(self, volume, xyz, token):
            return np.zeros(9)

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.4)
    calculator.evaluator = NativeFieldGeometryEvaluator()
    calculator._model_has_field_coupling = True
    electric_field = np.array([0.01, -0.02, 0.03])
    assert calculator._can_use_native_geometry(["stress", "becs"])

    inputs = calculator._mace_inputs(atoms, native_geometry=True)
    assert inputs[5] is None and inputs[6] is None
    calculator._compute_macefield(atoms, electric_field, mace_inputs=inputs)
    assert calculator._macefield_response_graph_generation == 31
    assert len(calculator.evaluator.geometry) == 1
    assert len(calculator.evaluator.positions_and_fields) == 1

    atoms.positions[1, 0] -= 10.0
    reused = calculator._mace_inputs(atoms, native_geometry=True)
    calculator._compute_macefield(atoms, electric_field, mace_inputs=reused)
    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert len(calculator.evaluator.geometry) == 1
    assert len(calculator.evaluator.positions_and_fields) == 2
    token, positions, actual_field = calculator.evaluator.positions_and_fields[-1]
    assert token == 31
    np.testing.assert_allclose(positions, atoms.positions.reshape(-1))
    np.testing.assert_array_equal(actual_field, electric_field)


def test_native_macefield_geometry_requires_prepared_response_support():
    class NativeFieldGeometryEvaluator:
        streamed_edges_mode = "direct"
        has_field_coupling = True

        def _prepare_factorized_graph(self, *args):
            pass

        def _prepare_factorized_geometry(self, *args):
            pass

        def _compute_prepared_factorized_positions_field(self, *args):
            pass

    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.4)
    calculator.evaluator = NativeFieldGeometryEvaluator()
    calculator._model_has_field_coupling = True

    assert calculator._can_use_native_geometry(["energy", "forces"])
    assert not calculator._can_use_native_geometry(["becs"])
    assert not calculator._can_use_native_geometry(["polarizability"])


def test_native_execution_geometry_normalizes_wrapping_during_cell_change():
    class NativeGeometryEvaluator:
        atomic_numbers = (1,)
        streamed_edges_mode = "direct"

        def __init__(self):
            self.geometry = []

        def _prepare_factorized_graph(self, *topology):
            return 17

        def _prepare_factorized_geometry(self, token, *geometry):
            self.geometry.append((token, geometry))

        def _compute_prepared_factorized_positions(self, token, positions):
            pass

        def _compute_prepared_factorized(self, *args):
            raise AssertionError("native geometry must submit positions")

        def _reduce_stress(self, volume, xyz, token):
            return np.zeros(9)

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.5)
    calculator.evaluator = NativeGeometryEvaluator()
    initial = calculator._mace_inputs(atoms, native_geometry=True)
    calculator._compute_mace(initial, atoms=atoms)

    atoms.set_cell([9.9, 10.0, 10.0], scale_atoms=True)
    atoms.positions[1, 0] -= 9.9
    reused = calculator._mace_inputs(atoms, native_geometry=True)
    calculator._compute_mace(reused, atoms=atoms)

    assert calculator.neighbor_cache_build_count == 1
    assert len(calculator.evaluator.geometry) == 2
    refreshed_reference_xyz = calculator.evaluator.geometry[-1][1][1].reshape(-1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(refreshed_reference_xyz, axis=1), [0.99, 0.99]
    )


def test_fractional_execution_cell_update_defers_cartesian_materialization():
    class FractionalGeometryEvaluator:
        atomic_numbers = (1,)
        streamed_edges_mode = "direct"
        has_field_coupling = False

        def __init__(self):
            self.fractional_geometry = []
            self.cell_updates = []
            self.positions = []

        def _prepare_factorized_graph(self, *topology):
            return 29

        def _prepare_factorized_geometry(self, *args):
            raise AssertionError("fractional geometry should remain active")

        def _prepare_factorized_fractional_geometry(self, token, *geometry):
            self.fractional_geometry.append((token, geometry))

        def _update_factorized_cell(self, token, cell, inverse_cell):
            self.cell_updates.append((token, cell.copy(), inverse_cell.copy()))

        def _compute_prepared_factorized_positions(self, token, positions):
            self.positions.append((token, positions.copy()))

        def _compute_prepared_factorized(self, *args):
            raise AssertionError("native geometry must submit positions")

        def _reduce_stress(self, volume, xyz, token):
            return np.zeros(9)

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.5)
    calculator.evaluator = FractionalGeometryEvaluator()

    initial = calculator._mace_inputs(atoms, native_geometry=True)
    calculator._compute_mace(initial, atoms=atoms)
    assert len(calculator.evaluator.fractional_geometry) == 1
    assert calculator.neighbor_cache_host_geometry_materialization_count == 0

    atoms.set_cell([9.9, 10.0, 10.0], scale_atoms=True)
    atoms.positions[1, 0] -= 9.9
    changed = calculator._mace_inputs(atoms, native_geometry=True)
    calculator._compute_mace(changed, atoms=atoms)

    assert calculator.neighbor_cache_build_count == 1
    assert len(calculator.evaluator.fractional_geometry) == 1
    assert len(calculator.evaluator.cell_updates) == 1
    assert calculator.neighbor_cache_host_geometry_materialization_count == 0

    host_inputs = calculator._mace_inputs(atoms)
    assert calculator.neighbor_cache_host_geometry_materialization_count == 1
    np.testing.assert_allclose(np.linalg.norm(host_inputs[5], axis=1), [0.99, 0.99])


def test_native_all_interactions_geometry_prepares_candidates_once_and_submits_positions():
    class NativeGeometryEvaluator:
        streamed_edges_mode = "generic"

        def __init__(self):
            self.atomic_numbers = [1]
            self.graphs = []
            self.geometry = []
            self.positions = []

        def _prepare_all_interactions_graph(self, *topology):
            self.graphs.append(topology)
            return 23

        def _prepare_all_interactions_geometry(self, token, *geometry):
            self.geometry.append((token, geometry))

        def _compute_prepared_all_interactions_positions(self, token, positions):
            self.positions.append((token, positions.copy()))

        def compute_node_energies_forces(self, *args):
            raise AssertionError("native all must not submit host edge arrays")

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = _bare_neighbor_cache_calculator(cutoff=2.0, skin=0.4)
    calculator.evaluator = NativeGeometryEvaluator()

    inputs = calculator._mace_inputs(atoms, native_geometry=True)
    assert inputs[5] is None and inputs[6] is None
    assert calculator._compute_mace(inputs, atoms=atoms) == 23
    assert len(calculator.evaluator.graphs) == 1
    assert len(calculator.evaluator.geometry) == 1
    assert len(calculator.evaluator.positions) == 1

    atoms.positions[1, 0] -= 10.0
    reused = calculator._mace_inputs(atoms, native_geometry=True)
    assert calculator._compute_mace(reused, atoms=atoms) == 23
    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count == 1
    assert len(calculator.evaluator.graphs) == 1
    assert len(calculator.evaluator.geometry) == 1
    assert len(calculator.evaluator.positions) == 2
    np.testing.assert_allclose(
        calculator.evaluator.positions[-1][1], atoms.positions.reshape(-1)
    )

    atoms.positions[1, 1] += 0.3
    rebuilt = calculator._mace_inputs(atoms, native_geometry=True)
    assert calculator._compute_mace(rebuilt, atoms=atoms) == 23
    assert calculator.neighbor_cache_build_count == 2
    assert len(calculator.evaluator.graphs) == 2
    assert len(calculator.evaluator.geometry) == 2
    assert len(calculator.evaluator.positions) == 3


def test_collect_results_uses_native_force_and_stress_reductions():
    class NativeReductionEvaluator:
        node_energies = np.array([1.0, 2.0])

        def __init__(self):
            self.calls = []
            self.node_force_reads = 0
            self.stress_calls = []

        def _reduce_atom_forces(self, num_nodes, receivers, sources, token):
            self.calls.append((num_nodes, receivers.copy(), sources.copy(), token))
            return np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        def _reduce_stress(self, volume, xyz, token):
            self.stress_calls.append((volume, np.array(xyz, copy=True), token))
            return np.diag([1.0, 2.0, 3.0])

        @property
        def node_forces(self):
            self.node_force_reads += 1
            return np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    calculator = object.__new__(Symmetrix)
    calculator.evaluator = NativeReductionEvaluator()
    calculator.implemented_properties = list(Symmetrix.implemented_properties)
    calculator._model_has_field_coupling = False
    inputs = (
        2,
        [0, 0],
        np.array([1, 1]),
        np.array([1, 0]),
        [0, 0],
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        np.array([1.0, 1.0]),
        np.array([0, 1]),
    )
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.set_cell([5.0, 5.0, 5.0])

    results = calculator._collect_mace_results(
        atoms, inputs, ["energy", "forces"], execution_graph_generation=17
    )
    assert "stress" not in results
    assert np.allclose(results["forces"], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert calculator.evaluator.node_force_reads == 0
    assert calculator.evaluator.calls[0][3] == 17

    results = calculator._collect_mace_results(
        atoms, inputs, ["stress"], execution_graph_generation=17
    )
    np.testing.assert_array_equal(results["stress"], [1.0, 2.0, 3.0, 0, 0, 0])
    assert calculator.evaluator.node_force_reads == 0
    assert calculator.evaluator.stress_calls[0][2] == 17


def test_collect_results_preserves_materialized_stress_fallback():
    class MaterializedEvaluator:
        node_energies = np.array([1.0, 2.0])
        node_forces = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

        def _reduce_atom_forces(self, num_nodes, receivers, sources, token):
            return np.zeros((num_nodes, 3))

    calculator = object.__new__(Symmetrix)
    calculator.evaluator = MaterializedEvaluator()
    calculator.implemented_properties = list(Symmetrix.implemented_properties)
    calculator._model_has_field_coupling = False
    xyz = np.array([[2.0, 0.5, -0.25], [-2.0, -0.5, 0.25]])
    inputs = (
        2,
        [0, 0],
        np.array([1, 1]),
        np.array([1, 0]),
        [0, 0],
        xyz,
        np.linalg.norm(xyz, axis=1),
        np.array([0, 1]),
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5])
    expected = full_3x3_to_voigt_6_stress(
        (-MaterializedEvaluator.node_forces.T @ xyz) / atoms.get_volume()
    )
    results = calculator._collect_mace_results(atoms, inputs, ["stress"])
    np.testing.assert_array_equal(results["stress"], expected)


def test_collect_results_accumulates_float32_energy_and_stress_in_float64():
    class Float32Evaluator:
        node_energies = np.array([1.0e8, 1.0, -1.0e8], dtype=np.float32)
        node_forces = np.array(
            [[1.0e8, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0e8, 0.0, 0.0]],
            dtype=np.float32,
        )

    calculator = object.__new__(Symmetrix)
    calculator.evaluator = Float32Evaluator()
    calculator.implemented_properties = list(Symmetrix.implemented_properties)
    calculator._model_has_field_coupling = False
    xyz = np.ones((3, 3), dtype=np.float32)
    xyz[:, 1:] = 0.0
    inputs = (
        3,
        [0, 0, 0],
        np.ones(3, dtype=int),
        np.array([1, 2, 0]),
        [0, 0, 0],
        xyz,
        np.ones(3),
        np.array([0, 1, 2]),
    )
    atoms = Atoms(
        "H3",
        positions=np.zeros((3, 3)),
        cell=np.eye(3),
    )

    results = calculator._collect_mace_results(atoms, inputs, ["energy", "stress"])

    assert np.sum(Float32Evaluator.node_energies, dtype=np.float32) == 0.0
    assert results["energy"] == 1.0
    assert results["free_energy"] == 1.0
    assert results["energies"].dtype == np.float64
    assert results["stress"].dtype == np.float64
    np.testing.assert_array_equal(results["stress"], [-1.0, 0, 0, 0, 0, 0])


@pytest.fixture(scope="module")
def mace_foundation_model():
    if mace is None:
        return None
    cache_dir = test_model_cache_dir() / "mace-foundation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    try:
        downloaded_model = download_mace_mp_checkpoint("small-omat-0")
    except Exception as exc:
        pytest.skip(f"MACE foundation model is not available: {exc}")
    finally:
        if xdg_cache_home is None:
            del os.environ["XDG_CACHE_HOME"]
        else:
            os.environ["XDG_CACHE_HOME"] = xdg_cache_home
    return str(downloaded_model)


@pytest.mark.parametrize("use_kokkos", [True, False])
def test_calc_caching(monkeypatch, model_cache, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 4
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    calc = Symmetrix(
        model_cache["mace-mp-0b3-medium-1-8.json"],
        use_kokkos=use_kokkos,
        streamed_edges="materialized",
    )
    atoms.calc = calc

    calculate_calls = 0
    original_calculate = calc.calculate

    def counted_calculate(*args, **kwargs):
        nonlocal calculate_calls
        calculate_calls += 1
        return original_calculate(*args, **kwargs)

    monkeypatch.setattr(calc, "calculate", counted_calculate)

    atoms.get_potential_energy()
    assert calculate_calls == 1

    atoms.get_forces()
    assert calculate_calls == 1

    atoms.positions[0, 0] += 0.1

    atoms.get_forces()
    assert calculate_calls == 2


def test_ase_npt_reuses_stress_and_factorized_skin_topology():
    class NPTFactorizedEvaluator:
        atomic_numbers = (1,)
        streamed_edges_mode = "direct"
        has_field_coupling = False

        def __init__(self):
            self.node_energies = np.zeros(2)
            self.graph_prepares = 0
            self.graph_builds = 0
            self.graph_topology = None
            self.fractional_geometry_prepares = 0
            self.cell_updates = 0
            self.evaluations = 0

        def _prepare_factorized_graph(self, *topology):
            self.graph_prepares += 1
            topology = tuple(np.array(value, copy=True) for value in topology)
            if self.graph_topology is None or any(
                not np.array_equal(current, previous)
                for current, previous in zip(topology, self.graph_topology)
            ):
                self.graph_builds += 1
                self.graph_topology = topology
            return 1

        def _prepare_factorized_geometry(self, token, *geometry):
            raise AssertionError("NPT should not prepare Cartesian edge geometry")

        def _prepare_factorized_fractional_geometry(self, token, *geometry):
            assert token == 1
            self.fractional_geometry_prepares += 1
            assert geometry[0].shape == (6,)
            assert geometry[1].shape == (6,)

        def _update_factorized_cell(self, token, cell, inverse_cell):
            assert token == 1
            assert cell.shape == (9,)
            assert inverse_cell.shape == (9,)
            self.cell_updates += 1

        def _compute_prepared_factorized_positions(self, token, positions):
            assert token == 1
            self.evaluations += 1

        def _compute_prepared_factorized(self, *args):
            raise AssertionError("NPT should use positions-only native geometry")

        def _reduce_atom_forces(self, num_nodes, receivers, sources, token):
            assert token == 1
            return np.zeros((num_nodes, 3))

        def _reduce_stress(self, volume, xyz, token):
            assert token == 1
            return np.diag([0.02, 0.02, 0.02])

    calculator = object.__new__(Symmetrix)
    Calculator.__init__(calculator)
    calculator.evaluator = NPTFactorizedEvaluator()
    calculator.cutoff = 2.0
    calculator.neighbor_skin = 0.5
    calculator._neighbor_cache = None
    calculator.neighbor_cache_build_count = 0
    calculator.neighbor_cache_reuse_count = 0
    calculator.neighbor_cache_geometry_update_count = 0
    calculator.neighbor_cache_host_geometry_materialization_count = 0
    calculator._execution_native_geometry_identity = None
    calculator._execution_native_cell_identity = None
    calculator._native_model_type = "MACE"
    calculator._all_interactions_native_geometry_identity = None
    calculator._all_interactions_graph_identity = None
    calculator._all_interactions_graph_generation = 0
    calculator.kernel_launch_policy = "static"
    calculator.kernel_launch_tuning_status = "static"
    calculator._model_has_field_coupling = False

    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    atoms.calc = calculator
    with pytest.warns(FutureWarning, match="moved/renamed"):
        dynamics = NPT(
            atoms,
            timestep=1.0 * units.fs,
            temperature_K=300.0,
            externalstress=0.0,
            ttime=25.0 * units.fs,
            pfactor=(75.0 * units.fs) ** 2,
        )
    dynamics.run(5)

    assert calculator.evaluator.evaluations == 5
    evaluations = calculator.evaluator.evaluations
    atoms.get_stress()
    atoms.get_forces()
    assert calculator.evaluator.evaluations == evaluations
    assert calculator.evaluator.graph_prepares == calculator.evaluator.evaluations
    assert calculator.evaluator.graph_builds == 1
    assert calculator.neighbor_cache_build_count == 1
    assert calculator.neighbor_cache_reuse_count >= 4
    assert calculator.neighbor_cache_geometry_update_count >= 2
    assert calculator.evaluator.fractional_geometry_prepares == 1
    assert calculator.evaluator.cell_updates == (
        calculator.neighbor_cache_geometry_update_count - 1
    )
    assert calculator.neighbor_cache_host_geometry_materialization_count == 0
    assert "stress" in calculator.results


@pytest.mark.parametrize("use_kokkos", [True, False])
def test_symmetrix_calc_finite_diff(model_cache, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    atoms.set_cell(atoms.cell @ F, True)

    print("pre-converted")
    calc = Symmetrix(
        model_cache["mace-mp-0b3-medium-1-8.json"],
        dtype="float64",
        use_kokkos=use_kokkos,
        streamed_edges="materialized",
    )
    do_grad_test(atoms, calc, True)


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
@pytest.mark.parametrize("use_kokkos", [True, False])
def test_mace_onthefly_calc_finite_diff(mace_foundation_model, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    atoms.set_cell(atoms.cell @ F, True)

    print("converted on-the-fly")
    calc = Symmetrix(
        mace_foundation_model,
        species=[1, 8],
        dtype="float64",
        use_kokkos=use_kokkos,
    )
    do_grad_test(atoms, calc, True)


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
@pytest.mark.parametrize("use_kokkos", [True, False])
def test_symmetrix_vs_pytorch(mace_foundation_model, use_kokkos):
    atoms = Atoms("O", cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3, 3))
    atoms.set_cell(atoms.cell @ F, True)

    atoms_s = atoms.copy()
    atoms_p = atoms.copy()

    calc_sym = Symmetrix(
        mace_foundation_model,
        species=[1, 8],
        dtype="float64",
        use_kokkos=use_kokkos,
    )
    atoms_s.calc = calc_sym

    calc_torch = MACECalculator(mace_foundation_model)
    atoms_p.calc = calc_torch

    # are these in fact reasonable accuracies?
    assert np.allclose(
        atoms_s.get_potential_energy(), atoms_p.get_potential_energy(), atol=0.001
    )
    assert np.allclose(atoms_s.get_forces(), atoms_p.get_forces(), atol=0.002)
    assert np.allclose(atoms_s.get_stress(), atoms_p.get_stress(), atol=0.003)


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_model_requires_explicit_json_conversion(macefield_model_path):
    with pytest.raises(
        RuntimeError, match="MACEField.*Convert/extract.*Symmetrix JSON"
    ):
        Symmetrix(
            macefield_model_path,
            species=[7, 13],
            head="mp-dielectric",
            use_kokkos=False,
            dtype="float64",
        )


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_float32_checkpoint_extracts_to_float64_json(
    macefield_model_path, tmp_path
):
    import torch
    from symmetrix.extract_mace_data import extract_mace_data

    model = torch.load(
        macefield_model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    parameter_dtype = next(model.parameters()).dtype
    if parameter_dtype != torch.float32:
        pytest.skip(f"Expected float32 MACEField checkpoint, got {parameter_dtype}.")

    json_path = tmp_path / "macefield-from-float32.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (len(atoms), 3)
    assert atoms.calc.get_property("polarization", atoms).shape == (3,)


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_native_json_ase_energy_forces_match_pytorch(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    assert np.allclose(
        atoms_sym.get_potential_energy(), atoms_torch.get_potential_energy(), atol=1e-3
    )
    assert np.allclose(atoms_sym.get_forces(), atoms_torch.get_forces(), atol=2e-3)


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
@pytest.mark.parametrize("use_kokkos", [False, True])
def test_macefield_field_contribution_matches_explicit_difference(
    macefield_model_path,
    tmp_path,
    use_kokkos,
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if use_kokkos and not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    electric_field = np.array([0.01, -0.02, 0.03])

    direct_atoms = atoms.copy()
    direct_calculator = Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64")
    direct_atoms.calc = direct_calculator
    direct_calculator.electric_field = electric_field
    field_results = {
        prop: direct_calculator.get_property(prop, direct_atoms)
        for prop in ("energy", "forces", "stress")
    }
    direct_calculator.electric_field = np.zeros(3)
    zero_results = {
        prop: direct_calculator.get_property(prop, direct_atoms)
        for prop in ("energy", "forces", "stress")
    }

    contribution_atoms = atoms.copy()
    contribution = FieldContributionCalculator(
        Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64"),
        electric_field=electric_field,
    )
    contribution_atoms.calc = contribution
    contribution_results = contribution_atoms.get_properties(
        ["energy", "forces", "stress"]
    )

    for prop in ("energy", "forces", "stress"):
        expected = np.asarray(field_results[prop]) - np.asarray(zero_results[prop])
        assert np.allclose(contribution_results[prop], expected, atol=1e-12, rtol=1e-12)

    step = 1e-4
    positions = contribution_atoms.positions.copy()
    contribution_atoms.positions[0, 0] += step
    energy_plus = contribution_atoms.get_potential_energy()
    contribution_atoms.positions[0, 0] -= 2.0 * step
    energy_minus = contribution_atoms.get_potential_energy()
    contribution_atoms.positions = positions
    force_fd = -(energy_plus - energy_minus) / (2.0 * step)
    assert np.isclose(contribution_results["forces"][0, 0], force_fd, atol=1e-7)

    cell = contribution_atoms.cell.copy()
    volume = contribution_atoms.get_volume()
    deformation = np.eye(3)
    deformation[0, 0] += step
    contribution_atoms.set_cell(cell @ deformation, scale_atoms=True)
    energy_plus = contribution_atoms.get_potential_energy()
    deformation[0, 0] -= 2.0 * step
    contribution_atoms.set_cell(cell @ deformation, scale_atoms=True)
    energy_minus = contribution_atoms.get_potential_energy()
    contribution_atoms.set_cell(cell, scale_atoms=True)
    stress_fd = (energy_plus - energy_minus) / (2.0 * step * volume)
    assert np.isclose(contribution_results["stress"][0], stress_fd, atol=1e-7)

    base = Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64")
    combined = FieldAwareCalculator(base, contribution)
    combined_atoms = atoms.copy()
    combined_atoms.calc = combined
    combined_results = combined_atoms.get_properties(["energy", "forces", "stress"])
    for prop in ("energy", "forces", "stress"):
        expected = base.get_property(prop, combined_atoms) + contribution.get_property(
            prop,
            combined_atoms,
        )
        assert np.allclose(combined_results[prop], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_native_json_ase_response_properties_match_pytorch(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    atoms_torch.get_potential_energy()
    expected_polarization = atoms_torch.calc.results["polarization"]
    expected_becs = atoms_torch.calc.results["becs"]
    expected_polarizability = atoms_torch.calc.results["polarizability"]

    assert "polarization" in atoms_sym.calc.implemented_properties
    assert "becs" in atoms_sym.calc.implemented_properties
    assert "polarizability" in atoms_sym.calc.implemented_properties

    actual_polarization = atoms_sym.calc.get_property("polarization", atoms_sym)
    actual_becs = atoms_sym.calc.get_property("becs", atoms_sym)
    actual_polarizability = atoms_sym.calc.get_property("polarizability", atoms_sym)

    assert actual_polarization.shape == (3,)
    assert actual_becs.shape == (len(atoms), 9)
    assert actual_polarizability.shape == (9,)
    assert np.allclose(actual_polarization, expected_polarization, atol=1e-5)
    assert np.allclose(actual_becs, expected_becs, atol=5e-2)
    assert np.allclose(actual_polarizability, expected_polarizability, atol=5e-3)


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_json_kokkos_response_properties_match_native(
    macefield_model_path, tmp_path
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

    atoms_native = atoms.copy()
    atoms_kokkos = atoms.copy()
    atoms_native.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_kokkos.calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")

    assert atoms_kokkos.calc.use_kokkos is True
    assert "polarization" in atoms_kokkos.calc.implemented_properties
    assert "becs" in atoms_kokkos.calc.implemented_properties
    assert "polarizability" in atoms_kokkos.calc.implemented_properties

    tolerances = {
        "energy": 1e-8,
        "forces": 1e-8,
        "polarization": 1e-8,
        "becs": 2e-6,
        "polarizability": 2e-6,
    }
    for prop, tolerance in tolerances.items():
        expected = atoms_native.calc.get_property(prop, atoms_native)
        actual = atoms_kokkos.calc.get_property(prop, atoms_kokkos)
        assert np.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_compact_macefield_calculator_replaces_active_composition_cache(
    macefield_model_path,
    tmp_path,
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield-multicomposition.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 8, 12, 13],
                head="mp-dielectric",
            ),
            separators=(",", ":"),
        )
    )

    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    partial_node_types = np.asarray([0, 99], dtype=np.int32)[::2]
    partial_num_neigh = np.asarray([1, 99], dtype=np.int32)[::2]
    partial_neigh_types = np.asarray([3, 99], dtype=np.int32)[::2]
    partial_r = np.asarray([1.5, 99.0], dtype=float)[::2]
    partial_R1 = []
    for evaluator_type in (native_symmetrix.MACE, native_symmetrix.MACEKokkos):
        evaluator = evaluator_type(str(json_path))
        evaluator.compute_R1(
            1,
            partial_node_types,
            partial_num_neigh,
            partial_neigh_types,
            partial_r,
        )
        assert evaluator.active_atomic_numbers == [7, 13]
        partial_R1.append(np.asarray(evaluator.R1))
    assert np.allclose(partial_R1[1], partial_R1[0], rtol=0.0, atol=1e-11)

    structures = (
        (bulk("AlN", "wurtzite", a=3.112, c=4.982), [7, 13]),
        (bulk("MgO", "rocksalt", a=4.21), [8, 12]),
        (bulk("AlN", "wurtzite", a=3.112, c=4.982), [7, 13]),
    )
    backend_energies = {}
    for use_kokkos in (False, True):
        calc = Symmetrix(json_path, use_kokkos=use_kokkos, dtype="float64")
        energies = []
        for atoms, expected_active in structures:
            atoms = atoms.copy()
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
            assert calc.evaluator.active_atomic_numbers == expected_active
        assert energies[0] == pytest.approx(energies[2], abs=1e-11)
        backend_energies[use_kokkos] = energies

    assert np.allclose(
        backend_energies[True],
        backend_energies[False],
        rtol=0.0,
        atol=1e-8,
    )

    electric_field = np.array([0.001, -0.002, 0.003])
    response_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    old_atoms = bulk("AlN", "rocksalt", a=4.05)
    new_atoms = bulk("MgO", "rocksalt", a=4.21)
    old_inputs = response_calc._mace_inputs(old_atoms)
    response_calc.evaluator.compute_node_energies_forces_field(
        *old_inputs[:7],
        electric_field,
    )
    new_inputs = response_calc._mace_inputs(new_atoms)
    response_calc.evaluator.compute_electric_field_hessian(
        *new_inputs[:7],
        electric_field,
    )
    switched_hessian = np.asarray(
        response_calc.evaluator.electric_field_hessian,
    ).copy()

    fresh_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    fresh_inputs = fresh_calc._mace_inputs(new_atoms)
    fresh_calc.evaluator.compute_electric_field_hessian(
        *fresh_inputs[:7],
        electric_field,
    )
    assert np.allclose(
        switched_hessian,
        fresh_calc.evaluator.electric_field_hessian,
        rtol=1e-8,
        atol=2e-6,
    )

    same_composition_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    baseline_atoms = bulk("AlN", "rocksalt", a=4.05)
    changed_atoms = baseline_atoms.copy()
    changed_atoms.positions[0, 0] += 0.05
    baseline_inputs = same_composition_calc._mace_inputs(baseline_atoms)
    same_composition_calc.evaluator.compute_node_energies_forces_field(
        *baseline_inputs[:7],
        electric_field,
    )
    changed_inputs = same_composition_calc._mace_inputs(changed_atoms)
    same_composition_calc.evaluator.compute_electric_field_hessian(
        *changed_inputs[:7],
        electric_field,
    )
    changed_hessian = np.asarray(
        same_composition_calc.evaluator.electric_field_hessian,
    ).copy()

    changed_fresh_calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")
    changed_fresh_inputs = changed_fresh_calc._mace_inputs(changed_atoms)
    changed_fresh_calc.evaluator.compute_electric_field_hessian(
        *changed_fresh_inputs[:7],
        electric_field,
    )
    assert np.allclose(
        changed_hessian,
        changed_fresh_calc.evaluator.electric_field_hessian,
        rtol=1e-8,
        atol=2e-6,
    )

    fresh_calc.evaluator.prepare_active_types(
        np.asarray([0, 1, 0, 1], dtype=np.int32)[::2],
    )
    assert fresh_calc.evaluator.active_atomic_numbers == [7]


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_native_json_node_energy_matches_pytorch(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    atoms_sym.get_potential_energy()
    atoms_torch.get_potential_energy()

    assert "node_energy" in atoms_sym.calc.implemented_properties
    assert atoms_sym.calc.results["node_energy"].shape == (len(atoms),)
    assert np.allclose(
        atoms_sym.calc.results["energies"],
        atoms_torch.calc.results["energies"],
        atol=1e-5,
    )
    assert np.allclose(
        atoms_sym.calc.results["node_energy"],
        atoms_torch.calc.results["node_energy"],
        atol=1e-5,
    )


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_native_json_requires_graph_field(macefield_model_path, tmp_path):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.zeros((len(atoms), 3))
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    with pytest.raises(ValueError, match="graph-level electric_field"):
        atoms.get_potential_energy()


def test_macefield_native_json_accepts_singleton_graph_field(monkeypatch, tmp_path):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.electric_fields = []
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.electric_fields.append(np.asarray(electric_field, dtype=float).copy())
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename, head="": evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([[0.01, -0.02, 0.03]])
    atoms.calc = Symmetrix(
        json_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="materialized",
    )

    assert np.isfinite(atoms.get_potential_energy())
    assert np.allclose(evaluator.electric_fields[-1], np.array([0.01, -0.02, 0.03]))


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_native_json_keeps_response_properties_selective(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.isfinite(atoms.get_potential_energy())
    assert "polarization" not in atoms.calc.results
    assert "becs" not in atoms.calc.results
    assert "polarizability" not in atoms.calc.results

    assert atoms.calc.get_property("polarization", atoms).shape == (3,)


def test_macefield_native_json_polarization_uses_single_native_field_call(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename, head="": evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(
        json_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="materialized",
    )

    assert np.allclose(
        atoms.calc.get_property("polarization", atoms),
        -np.array([1.0, 2.0, 3.0]) / atoms.get_volume(),
    )
    assert evaluator.calls == 1


def test_macefield_native_json_polarizability_uses_native_field_hessian(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.hessian_calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []
            self.electric_field_hessian = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

        def compute_electric_field_hessian(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.hessian_calls += 1
            self.electric_field_hessian = np.arange(9, dtype=float).reshape(3, 3)

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename, head="": evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(
        json_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="materialized",
    )

    expected = (
        -np.arange(9, dtype=float).reshape(3, 3)
        / atoms.get_volume()
        / atoms.calc._macefield_eps0
    ).reshape(9)
    assert np.allclose(atoms.calc.get_property("polarizability", atoms), expected)
    assert evaluator.calls == 1
    assert evaluator.hessian_calls == 1


def test_macefield_native_json_becs_use_native_force_field_derivative(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.derivative_calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []
            self.electric_field_force_derivative = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

        def compute_electric_field_force_derivative(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.derivative_calls += 1
            self.electric_field_force_derivative = np.arange(
                3 * len(xyz), dtype=float
            ).reshape(3, -1, 3)

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename, head="": evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(
        json_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="materialized",
    )

    num_nodes, _, _, j_list, _, xyz, _, i_list = atoms.calc._mace_inputs(atoms)
    pair_derivative = np.arange(3 * xyz.size, dtype=float).reshape(3, -1, 3)[
        :, : len(i_list)
    ]
    expected = np.zeros((num_nodes, 3, 3))
    for field_component in range(3):
        for cartesian in range(3):
            expected[:, field_component, cartesian] = np.bincount(
                j_list,
                weights=pair_derivative[field_component, :, cartesian],
                minlength=num_nodes,
            ) - np.bincount(
                i_list,
                weights=pair_derivative[field_component, :, cartesian],
                minlength=num_nodes,
            )

    assert np.allclose(
        atoms.calc.get_property("becs", atoms), expected.reshape(num_nodes, 9)
    )
    assert evaluator.calls == 1
    assert evaluator.derivative_calls == 1


def test_macefield_native_json_cache_tracks_only_electric_field(monkeypatch, tmp_path):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            field = np.asarray(electric_field, dtype=float)
            self.node_energies = np.full(num_nodes, field[0])
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    class NonNumericMetadata:
        pass

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename, head="": evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["spacegroup"] = NonNumericMetadata()
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(
        json_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="materialized",
    )

    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    assert evaluator.calls == 1

    atoms.info["spacegroup"] = NonNumericMetadata()
    assert np.isfinite(atoms.get_forces()).all()
    assert evaluator.calls == 1

    atoms.info["electric_field"][0] = 0.02
    assert np.isfinite(atoms.get_potential_energy())
    assert evaluator.calls == 2


def test_macefield_native_json_calculator_electric_field_override(
    monkeypatch, tmp_path
):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self):
            self.electric_fields = []
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            field = np.asarray(electric_field, dtype=float)
            self.electric_fields.append(field.copy())
            self.node_energies = np.full(num_nodes, field[2])
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE", lambda filename, head="": evaluator
    )

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(
        json_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="materialized",
    )

    atoms.calc.electric_field = [0.0, 0.0, 0.02]
    energy_override = atoms.get_potential_energy()

    atoms.calc.electric_field = [0.0, 0.0, 0.03]
    energy_updated = atoms.get_potential_energy()

    assert np.isclose(energy_override, 0.04)
    assert np.isclose(energy_updated, 0.06)
    assert np.allclose(evaluator.electric_fields, [[0.0, 0.0, 0.02], [0.0, 0.0, 0.03]])


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_native_json_uses_kokkos_field_path_when_kokkos_requested(
    macefield_model_path, tmp_path
):
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data

    if not hasattr(native_symmetrix, "MACEKokkos"):
        pytest.skip("Symmetrix was built without Kokkos bindings.")

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")

    assert atoms.calc.use_kokkos is True
    assert "polarization" in atoms.calc.implemented_properties
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.calc.get_property("polarization", atoms).shape == (3,)


def test_macefield_json_with_kokkos_requested_constructs_kokkos_evaluator(
    monkeypatch, tmp_path
):
    class DummyKokkosEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]
        atomic_energies = np.array([0.0, 0.0])

        def __init__(self, filename, head=""):
            assert head == ""
            self.filename = filename
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

    constructed = []

    def fake_init_kokkos():
        constructed.append("init")

    def fake_mace_kokkos(filename, head=""):
        assert head == ""
        constructed.append("kokkos")
        return DummyKokkosEvaluator(filename)

    def fake_mace(filename, head=""):
        assert head == ""
        constructed.append("serial")
        raise AssertionError(
            "field-aware use_kokkos=True should not instantiate serial MACE"
        )

    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix._kokkos_is_initialized", lambda: False
    )
    monkeypatch.setattr("symmetrix.calculator.symmetrix._init_kokkos", fake_init_kokkos)
    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACEKokkos", fake_mace_kokkos)
    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACE", fake_mace)

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    calc = Symmetrix(
        json_path,
        use_kokkos=True,
        dtype="float64",
        streamed_edges="materialized",
    )

    assert calc.use_kokkos is True
    assert constructed == ["init", "kokkos"]
    assert isinstance(calc.evaluator, DummyKokkosEvaluator)
    assert "polarization" in calc.implemented_properties


@pytest.mark.skipif(not mace_field_available, reason="mace-field is not available")
def test_macefield_native_json_electric_field_changes_cached_results(
    macefield_model_path, tmp_path
):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(
        json.dumps(
            extract_mace_data(
                macefield_model_path,
                species=[7, 13],
                head="mp-dielectric",
            )
        )
    )

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    atoms.info["electric_field"] = np.array([0.0, 0.0, 0.0])
    energy_zero = atoms.get_potential_energy()
    forces_zero = atoms.get_forces()

    atoms.info["electric_field"][0] = 0.01
    energy_field = atoms.get_potential_energy()
    forces_field = atoms.get_forces()

    assert not np.isclose(energy_zero, energy_field, rtol=0.0, atol=1e-8)
    assert not np.allclose(forces_zero, forces_field)


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
def test_plain_mace_model_uses_native_symmetrix_path(monkeypatch, tmp_path):
    class PlainTorchModel:
        pass

    class DummyEvaluator:
        r_cut = 3.0

    def fake_torch_load(*args, **kwargs):
        return PlainTorchModel()

    monkeypatch.setattr("torch.load", fake_torch_load)
    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix.MACE",
        lambda filename, head="": DummyEvaluator(),
    )

    calc = Symmetrix(
        tmp_path / "plain.model",
        dtype="float64",
        use_kokkos=False,
        streamed_edges="materialized",
    )

    assert calc.evaluator.r_cut == 3.0
    assert calc.implemented_properties == [
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    ]


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
def test_unloadable_torch_model_preserves_load_error(monkeypatch, tmp_path):
    def fake_native_loader(filename, head=""):
        assert head == ""
        raise RuntimeError("[json.exception.parse_error.101] not native json")

    def fake_torch_load(*args, **kwargs):
        raise ValueError("checkpoint cannot be unpickled")

    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACE", fake_native_loader)
    monkeypatch.setattr("torch.load", fake_torch_load)

    with pytest.raises(ValueError, match="checkpoint cannot be unpickled"):
        Symmetrix(tmp_path / "broken.model", use_kokkos=False)


def test_native_schema_error_is_not_treated_as_checkpoint(monkeypatch, tmp_path):
    def fake_native_loader(filename, head=""):
        assert head == ""
        raise RuntimeError("[json.exception.type_error.302] invalid model schema")

    def fake_metadata(filename):
        raise RuntimeError("[json.exception.parse_error.101] metadata probe")

    def fail_checkpoint_load(*args, **kwargs):
        pytest.fail(
            "native JSON schema errors must not fall through to checkpoint loading"
        )

    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix._model_metadata",
        fake_metadata,
    )
    monkeypatch.setattr("symmetrix.calculator.symmetrix.MACE", fake_native_loader)
    monkeypatch.setattr(
        Symmetrix, "_raise_if_macefield_checkpoint", fail_checkpoint_load
    )

    with pytest.raises(RuntimeError, match="type_error.302"):
        Symmetrix(
            tmp_path / "invalid-extensionless-json",
            dtype="float64",
            use_kokkos=False,
        )


def test_metadata_schema_error_is_not_suppressed(monkeypatch, tmp_path):
    def fake_metadata(filename):
        raise RuntimeError("[json.exception.type_error.302] invalid metadata")

    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix._model_metadata",
        fake_metadata,
    )

    with pytest.raises(RuntimeError, match="type_error.302"):
        Symmetrix(tmp_path / "invalid-extensionless-json", use_kokkos=False)


def test_missing_kokkos_does_not_load_extensionless_checkpoint(monkeypatch, tmp_path):
    def fake_metadata(filename):
        raise RuntimeError("[json.exception.parse_error.101] checkpoint")

    def fail_checkpoint_load(*args, **kwargs):
        pytest.fail("model loading must not precede the missing-Kokkos error")

    monkeypatch.setattr(
        "symmetrix.calculator.symmetrix._model_metadata",
        fake_metadata,
    )
    monkeypatch.delattr("symmetrix.calculator.symmetrix.MACEKokkos")
    monkeypatch.setattr(
        Symmetrix, "_raise_if_macefield_checkpoint", fail_checkpoint_load
    )

    with pytest.raises(RuntimeError, match="built without Kokkos support"):
        Symmetrix(tmp_path / "extensionless-checkpoint", use_kokkos=True)


def do_grad_test(atoms, calc, check, ax=None, label=None, plot_factor=1.0):
    atoms = atoms.copy()
    atoms.calc = calc

    F0 = atoms.get_forces()
    S0 = atoms.get_stress()
    F0_norm = np.linalg.norm(F0)
    S0_norm = np.linalg.norm(S0)
    p0 = atoms.positions.copy()
    c0 = atoms.cell.copy()
    V0 = atoms.get_volume()

    f_data = []
    passed_f = True
    F_scaling = None
    for dx_exp in np.arange(1.0, 5.1, 0.5):
        dx = 0.1**dx_exp

        #### forces ####
        atoms.positions = p0
        atoms.cell = c0
        F_fd = np.zeros((len(atoms), 3))
        for i_a in range(len(atoms)):
            for j_a in range(3):
                p = p0.copy()
                p[i_a, j_a] = p0[i_a, j_a] + dx
                atoms.positions = p
                E_p = atoms.get_potential_energy()
                p[i_a, j_a] = p0[i_a, j_a] - dx
                atoms.positions = p
                E_m = atoms.get_potential_energy()
                F_fd[i_a, j_a] = -(E_p - E_m) / (2 * dx)
        F_err = np.linalg.norm(F0 - F_fd)
        print(
            f"F {dx:6f} {F0_norm:10.6e} {F_err:10.6e} {F_err / F0_norm:10.6e} {F_err / F0_norm / (dx**2):10.6e}"
        )

        f_data.append([dx, F_err])

        # force error only shows expected 2nd order scaling for dx = 0.1 ** 1, 0.1 ** 1.5
        if F_scaling is None and dx_exp >= 1.99:
            # F_err / F0_norm < F_scaling * dx ** 2
            F_scaling = 4.0 * F_err / F0_norm / (dx**2)
        if F_scaling is not None and dx_exp < 4.01:
            print("test forces", dx_exp, dx, F_err / F0_norm, "<?", F_scaling * dx**2)
            passed_f = passed_f and (F_err / F0_norm < F_scaling * dx**2)

    if ax is not None:
        f_data = np.asarray(f_data)
        ax.loglog(f_data[:, 0], f_data[:, 1] * plot_factor, "-", label=label)

    passed_s = True
    S_scaling = None
    for dx_exp in np.arange(1.0, 5.1, 0.5):
        dx = 0.1**dx_exp

        #### stress ####
        atoms.positions = p0
        atoms.cell = c0
        S_fd = np.zeros((3, 3))
        for i0 in range(3):
            for i1 in range(3):
                F = np.eye(3)
                F[i0, i1] += dx / 2
                F[i1, i0] += dx / 2
                atoms.positions = p0
                atoms.cell = c0
                atoms.set_cell(c0 @ F, True)
                E_p = atoms.get_potential_energy()

                F = np.eye(3)
                F[i0, i1] -= dx / 2
                F[i1, i0] -= dx / 2
                atoms.positions = p0
                atoms.cell = c0
                atoms.set_cell(c0 @ F, True)
                E_m = atoms.get_potential_energy()

                S_fd[i0, i1] = (E_p - E_m) / (2 * dx) / V0

        S_err = np.linalg.norm(S0 - full_3x3_to_voigt_6_stress(S_fd))
        print(
            f"S {dx:6f} {S0_norm:10.6e} {S_err:10.6e} {S_err / S0_norm:10.6e} {S_err / S0_norm / dx**2:10.6e}"
        )

        if S_scaling is None and dx_exp >= 1.99:
            # S_err / S0_norm < S_scaling * dx ** 2
            S_scaling = 1.5 * S_err / S0_norm / (dx**2)
        if S_scaling is not None and dx_exp < 4.01:
            print("test stress", dx_exp, dx, S_err / S0_norm, "<?", S_scaling * dx**2)
            passed_s = passed_s and (S_err / S0_norm < S_scaling * dx**2)

    if check:
        assert passed_f and passed_s
