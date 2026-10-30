from __future__ import annotations

import math
import os

import numpy as np
import pytest

import symmetrix
from polar_oracle_checkout import (
    import_graph_electrostatics_realspace,
    optional_checkout,
)


if not symmetrix._kokkos_is_initialized():
    symmetrix._init_kokkos()


FIELD_CONSTANT = 1.0 / 5.526349406e-3
COULOMB_CONSTANT = FIELD_CONSTANT / (4.0 * math.pi)
KERNEL_EPSILON = 1.0e-6
FEATURE_DISPLACEMENT = 0.1
ENERGY_DISPLACEMENT = 0.02

# Captured from graph-electrostatics e0eb8945 for sigma_density=0.73 and
# receiver widths [0.51, 1.07]. These are test data, not native helpers.
FEATURE_SELF_L1 = np.array(
    [
        45.73621237874995,
        31.443196842295613,
        10.64207953034085,
        10.64207953034085,
        10.64207953034085,
        7.25503643203897,
        7.25503643203897,
        7.25503643203897,
    ],
    dtype=np.float64,
)
FEATURE_SELF_L0 = np.array([45.73621237874995, 31.443196842295613], dtype=np.float64)
ENERGY_SELF_L1 = np.array(
    [
        11.128945089057039,
        3.480623346892647,
        3.480623346892647,
        3.480623346892647,
    ],
    dtype=np.float64,
)
ENERGY_SELF_L0 = np.array([11.128945089057039], dtype=np.float64)

DTYPES = (np.dtype(np.float64), np.dtype(np.float32))
L_COMBINATIONS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _cl_sigma(l: int, sigma: float, *, receiver: bool) -> float:
    if receiver:
        denominator = (
            2.0 ** ((l + 1) / 2.0) * math.gamma((l + 3) / 2.0) * sigma ** (l + 3)
        )
    else:
        denominator = (
            math.sqrt(4.0 * math.pi / (2 * l + 1))
            * 2.0 ** ((2 * l + 1) / 2.0)
            * math.gamma((2 * l + 3) / 2.0)
            * sigma ** (2 * l + 3)
        )
    return 1.0 / denominator


def _replica_transform(max_l: int, displacement: float) -> np.ndarray:
    if max_l == 0:
        return np.ones((1, 1), dtype=np.float64)
    inverse = 1.0 / displacement
    return np.array(
        [
            [1.0, -inverse, -inverse, -inverse],
            [0.0, 0.0, 0.0, inverse],
            [0.0, inverse, 0.0, 0.0],
            [0.0, 0.0, inverse, 0.0],
        ],
        dtype=np.float64,
    )


def _replica_offsets(max_l: int, displacement: float) -> np.ndarray:
    if max_l == 0:
        return np.zeros((1, 3), dtype=np.float64)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [displacement, 0.0, 0.0],
            [0.0, displacement, 0.0],
            [0.0, 0.0, displacement],
        ],
        dtype=np.float64,
    )


def _kernel(distance: float, width: float) -> float:
    return math.erf(distance / (2.0 * width)) / (distance + KERNEL_EPSILON)


def _feature_values(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    feature_max_l: int,
    feature_widths: tuple[float, ...],
    *,
    include_self_interaction: bool,
) -> np.ndarray:
    """Independent FP64 spelling of the committed real-space equations."""
    n_nodes = positions.shape[0]
    n_sigma = len(feature_widths)
    feature_dim = n_sigma * (feature_max_l + 1) ** 2
    result = np.zeros((n_nodes, feature_dim), dtype=np.float64)
    source_transform = _replica_transform(density_max_l, FEATURE_DISPLACEMENT)
    source_offsets = _replica_offsets(density_max_l, FEATURE_DISPLACEMENT)
    receiver_offsets = _replica_offsets(feature_max_l, FEATURE_DISPLACEMENT)
    charges = multipoles @ source_transform.T
    effective_widths = tuple(
        math.sqrt((0.73**2 + sigma**2) / 2.0) for sigma in feature_widths
    )
    a0 = tuple(
        _cl_sigma(0, sigma, receiver=True) / _cl_sigma(0, sigma, receiver=False)
        for sigma in feature_widths
    )
    a1 = tuple(
        math.sqrt(3.0)
        * sigma**2
        * _cl_sigma(1, sigma, receiver=True)
        / _cl_sigma(0, sigma, receiver=False)
        / FEATURE_DISPLACEMENT
        for sigma in feature_widths
    )
    for receiver in range(n_nodes):
        samples = np.zeros((receiver_offsets.shape[0], n_sigma), dtype=np.float64)
        for sender in range(n_nodes):
            if (
                sender == receiver
                or node_graph_index[sender] != node_graph_index[receiver]
            ):
                continue
            for source_replica, charge in enumerate(charges[sender]):
                source_position = positions[sender] + source_offsets[source_replica]
                for receiver_replica, receiver_offset in enumerate(receiver_offsets):
                    difference = source_position - (
                        positions[receiver] + receiver_offset
                    )
                    distance = float(np.linalg.norm(difference))
                    for channel, width in enumerate(effective_widths):
                        samples[receiver_replica, channel] += charge * _kernel(
                            distance, width
                        )
        samples *= COULOMB_CONSTANT
        for channel in range(n_sigma):
            result[receiver, channel] = a0[channel] * samples[0, channel]
            if feature_max_l == 1:
                for component, receiver_replica in enumerate((2, 3, 1)):
                    result[receiver, n_sigma + 3 * channel + component] = a1[
                        channel
                    ] * (samples[receiver_replica, channel] - samples[0, channel])
    if include_self_interaction:
        if density_max_l != 1 or feature_max_l != 1 or feature_widths != (0.51, 1.07):
            raise AssertionError("self fixture is pinned only for the l=1 test basis")
        result[:, :2] += multipoles[:, :1] * FEATURE_SELF_L1[:2]
        result[:, 2:5] += multipoles[:, 1:4] * FEATURE_SELF_L1[2:5]
        result[:, 5:8] += multipoles[:, 1:4] * FEATURE_SELF_L1[5:8]
    return result


def _energy_values(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    n_graphs: int,
    *,
    include_self_interaction: bool,
) -> np.ndarray:
    """Independent complete-directed-pair energy in FP64."""
    transform = _replica_transform(density_max_l, ENERGY_DISPLACEMENT)
    offsets = _replica_offsets(density_max_l, ENERGY_DISPLACEMENT)
    charges = multipoles @ transform.T
    result = np.zeros(n_graphs, dtype=np.float64)
    for receiver in range(positions.shape[0]):
        graph = int(node_graph_index[receiver])
        for receiver_replica, receiver_charge in enumerate(charges[receiver]):
            receiver_position = positions[receiver] + offsets[receiver_replica]
            for sender in range(positions.shape[0]):
                if sender == receiver or int(node_graph_index[sender]) != graph:
                    continue
                for source_replica, source_charge in enumerate(charges[sender]):
                    difference = (
                        positions[sender] + offsets[source_replica] - receiver_position
                    )
                    result[graph] += (
                        0.5
                        * COULOMB_CONSTANT
                        * source_charge
                        * receiver_charge
                        * _kernel(float(np.linalg.norm(difference)), 0.73)
                    )
    if include_self_interaction:
        if density_max_l != 1:
            raise AssertionError("self fixture is pinned only for l=1")
        for node, row in enumerate(multipoles):
            graph = int(node_graph_index[node])
            result[graph] += 0.5 * float(np.dot(ENERGY_SELF_L1 * row, row))
    return result


def _finite_difference(function, values: np.ndarray, step: float) -> np.ndarray:
    """Fourth-order centered differences, independent of native reverse code."""
    result = np.empty_like(values, dtype=np.float64)
    working = values.astype(np.float64, copy=True)
    for index in np.ndindex(values.shape):
        center = working[index]
        scaled_step = step * max(1.0, abs(center))
        working[index] = center + 2.0 * scaled_step
        plus_two = function(working)
        working[index] = center + scaled_step
        plus_one = function(working)
        working[index] = center - scaled_step
        minus_one = function(working)
        working[index] = center - 2.0 * scaled_step
        minus_two = function(working)
        working[index] = center
        result[index] = (-plus_two + 8.0 * plus_one - 8.0 * minus_one + minus_two) / (
            12.0 * scaled_step
        )
    return result


def _ops(
    dtype: np.dtype,
    density_max_l: int,
    feature_max_l: int,
    feature_widths: tuple[float, ...] = (0.51, 1.07),
):
    class_name = (
        "PolarLongRangeOpsFloat"
        if dtype == np.dtype(np.float32)
        else "PolarLongRangeOps"
    )
    return getattr(symmetrix, class_name)(
        kspace_cutoff=4.0,
        density_max_l=density_max_l,
        density_smearing_width=0.73,
        feature_max_l=feature_max_l,
        feature_smearing_widths=list(feature_widths),
    )


def _select_reverse_strategy(ops, strategy: str):
    if strategy == "materialized_forward":
        ops.set_realspace_materialized_budget_bytes(np.iinfo(np.uintp).max)
    ops.set_realspace_reverse_strategy(strategy)
    assert ops.realspace_reverse_strategy == strategy


def _assert_reverse_strategy_was_used(ops, strategy: str):
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["strategy"] == strategy
    if strategy == "pair_atomic":
        assert diagnostics["feature_pair_atomic_selected_count"] > 0
        assert diagnostics["energy_pair_atomic_selected_count"] > 0
        assert diagnostics["feature_pair_league_extent"] > 0
        assert diagnostics["energy_pair_league_extent"] > 0
        assert diagnostics["feature_owner_selected_count"] == 0
        assert diagnostics["energy_owner_selected_count"] == 0
    elif strategy == "materialized_forward":
        assert diagnostics["materialized_forward_reducer"] == "segmented"
        assert diagnostics["feature_pair_atomic_selected_count"] > 0
        assert diagnostics["energy_pair_atomic_selected_count"] > 0
        assert diagnostics["feature_owner_selected_count"] == 0
        assert diagnostics["energy_owner_selected_count"] == 0
    else:
        assert diagnostics["feature_pair_atomic_selected_count"] == 0
        assert diagnostics["energy_pair_atomic_selected_count"] == 0
        assert diagnostics["feature_owner_selected_count"] > 0
        assert diagnostics["energy_owner_selected_count"] > 0


def _cache(ops, positions: np.ndarray, node_graph_index: np.ndarray, cell=None):
    n_graphs = int(node_graph_index.max()) + 1
    if cell is None:
        cell = np.repeat(np.eye(3)[None, :, :] * 8.0, n_graphs, axis=0)
    return ops.realspace_geometry_cache(positions, cell, node_graph_index)


def _multipoles(dtype: np.dtype, density_max_l: int, charged: bool) -> np.ndarray:
    width = (density_max_l + 1) ** 2
    values = np.array(
        [
            [0.55, 0.07, -0.11, 0.03],
            [-0.20, -0.04, 0.02, 0.09],
            [-0.35 if not charged else 0.11, 0.06, 0.05, -0.08],
        ],
        dtype=dtype,
    )
    return values[:, :width]


def _feature_seed(dtype: np.dtype, n_nodes: int, feature_dim: int) -> np.ndarray:
    values = np.linspace(-0.73, 1.19, n_nodes * feature_dim, dtype=np.float64)
    values[1::3] *= -0.61
    return values.reshape(n_nodes, feature_dim).astype(dtype)


def _gradient_tolerances(dtype: np.dtype, expected: np.ndarray) -> dict[str, float]:
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    if dtype == np.dtype(np.float64):
        return {"rtol": 1.0e-6, "atol": 1.0e-7 * scale}
    return {"rtol": 2.0e-4, "atol": 2.0e-5 * scale}


def _assert_gradient(actual: np.ndarray, expected: np.ndarray, dtype: np.dtype):
    np.testing.assert_allclose(
        actual, expected, **_gradient_tolerances(dtype, expected)
    )


def _canonical_expected_gradients(
    dtype: np.dtype,
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    feature_max_l: int,
    feature_seed: np.ndarray,
    energy_seed: np.ndarray,
    *,
    feature_self: bool,
    energy_self: bool,
):
    base_multipoles = multipoles.astype(np.float64)
    promoted_feature_seed = feature_seed.astype(np.float64)
    promoted_energy_seed = energy_seed.astype(np.float64)
    n_graphs = promoted_energy_seed.size

    def feature_loss_for_multipoles(candidate):
        values = _feature_values(
            positions,
            node_graph_index,
            candidate,
            density_max_l,
            feature_max_l,
            (0.51, 1.07),
            include_self_interaction=feature_self,
        )
        return float(np.sum(values * promoted_feature_seed))

    def feature_loss_for_positions(candidate):
        values = _feature_values(
            candidate,
            node_graph_index,
            base_multipoles,
            density_max_l,
            feature_max_l,
            (0.51, 1.07),
            include_self_interaction=feature_self,
        )
        return float(np.sum(values * promoted_feature_seed))

    def energy_loss_for_multipoles(candidate):
        values = _energy_values(
            positions,
            node_graph_index,
            candidate,
            density_max_l,
            n_graphs,
            include_self_interaction=energy_self,
        )
        return float(np.dot(values, promoted_energy_seed))

    def energy_loss_for_positions(candidate):
        values = _energy_values(
            candidate,
            node_graph_index,
            base_multipoles,
            density_max_l,
            n_graphs,
            include_self_interaction=energy_self,
        )
        return float(np.dot(values, promoted_energy_seed))

    feature_grad_multipoles = _finite_difference(
        feature_loss_for_multipoles, base_multipoles, 2.0e-5
    )
    feature_grad_positions = _finite_difference(
        feature_loss_for_positions, positions, 2.0e-5
    )
    energy_grad_multipoles = _finite_difference(
        energy_loss_for_multipoles, base_multipoles, 2.0e-5
    )
    energy_grad_positions = _finite_difference(
        energy_loss_for_positions, positions, 2.0e-5
    )
    if dtype == np.dtype(np.float32):
        feature_grad_multipoles = feature_grad_multipoles.astype(np.float32)
        energy_grad_multipoles = energy_grad_multipoles.astype(np.float32)
    return {
        "feature_grad_multipoles": feature_grad_multipoles,
        "feature_grad_positions": feature_grad_positions,
        "energy_grad_multipoles": energy_grad_multipoles,
        "energy_grad_positions": energy_grad_positions,
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("density_max_l,feature_max_l", L_COMBINATIONS)
@pytest.mark.parametrize("charged", (False, True), ids=("neutral", "charged"))
@pytest.mark.parametrize(
    "reverse_strategy",
    (
        "owner",
        "pair_atomic",
        "materialized_forward",
    ),
)
def test_feature_and_energy_adjoints_match_independent_centered_differences(
    dtype, density_max_l, feature_max_l, charged, reverse_strategy
):
    # Catches a wrong T_delta transpose, x/y/z permutation, incomplete
    # reconstruction transpose, or missing source/receiver position branch.
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, density_max_l, charged)
    feature_dim = 2 * (feature_max_l + 1) ** 2
    feature_seed = _feature_seed(dtype, 3, feature_dim)
    energy_seed = np.array([1.37], dtype=dtype)
    expected = _canonical_expected_gradients(
        dtype,
        positions,
        node_graph_index,
        multipoles,
        density_max_l,
        feature_max_l,
        feature_seed,
        energy_seed,
        feature_self=False,
        energy_self=False,
    )
    ops = _ops(dtype, density_max_l, feature_max_l)
    _select_reverse_strategy(ops, reverse_strategy)
    cache = _cache(ops, positions, node_graph_index)

    feature = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, False
    )
    energy = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, False
    )

    _assert_gradient(
        feature["grad_multipoles"], expected["feature_grad_multipoles"], dtype
    )
    _assert_gradient(
        feature["grad_positions"], expected["feature_grad_positions"], dtype
    )
    _assert_gradient(
        energy["grad_multipoles"], expected["energy_grad_multipoles"], dtype
    )
    _assert_gradient(energy["grad_positions"], expected["energy_grad_positions"], dtype)
    _assert_reverse_strategy_was_used(ops, reverse_strategy)
    np.testing.assert_array_equal(feature["grad_cell"], np.zeros((1, 3, 3)))
    np.testing.assert_array_equal(energy["grad_cell"], np.zeros((1, 3, 3)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "feature_self,energy_self",
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_shared_source_combined_loss_and_independent_self_flags_match_fd(
    dtype, feature_self, energy_self
):
    # Catches coupling the self flags or returning adjoints that cannot be
    # added at the one shared learned-source boundary.
    positions = np.array(
        [[-0.31, 0.24, 0.52], [0.68, -0.47, 0.19], [1.27, 0.71, -0.64]],
        dtype=np.float64,
    )
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    feature_seed = _feature_seed(dtype, 3, 8)
    energy_seed = np.array([-0.83], dtype=dtype)
    expected = _canonical_expected_gradients(
        dtype,
        positions,
        node_graph_index,
        multipoles,
        1,
        1,
        feature_seed,
        energy_seed,
        feature_self=feature_self,
        energy_self=energy_self,
    )
    ops = _ops(dtype, 1, 1)
    cache = _cache(ops, positions, node_graph_index)
    feature = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, feature_self
    )
    energy = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, energy_self
    )

    _assert_gradient(
        feature["grad_multipoles"] + energy["grad_multipoles"],
        expected["feature_grad_multipoles"] + expected["energy_grad_multipoles"],
        dtype,
    )
    _assert_gradient(
        feature["grad_positions"] + energy["grad_positions"],
        expected["feature_grad_positions"] + expected["energy_grad_positions"],
        dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_coincident_scalar_pair_derivative_is_exactly_zero_but_l1_is_not(dtype):
    # Catches assigning a direction to r==0. The l=1 case proves that the
    # zero convention applies pairwise rather than suppressing the full atom
    # pair, whose noncoincident displaced replicas still carry derivatives.
    positions = np.array([[0.2, -0.3, 0.4], [0.2, -0.3, 0.4]], dtype=np.float64)
    node_graph_index = np.zeros(2, dtype=np.int32)
    energy_seed = np.array([0.91], dtype=dtype)

    scalar_ops = _ops(dtype, 0, 0, (0.51,))
    scalar_cache = _cache(scalar_ops, positions, node_graph_index)
    scalar_multipoles = np.array([[0.7], [-0.4]], dtype=dtype)
    scalar_features = scalar_ops.diagnostic_adjoint_realspace_features(
        scalar_cache,
        scalar_multipoles,
        np.array([[0.6], [-1.1]], dtype=dtype),
        False,
    )
    scalar_energy = scalar_ops.diagnostic_adjoint_realspace_energy(
        scalar_cache, scalar_multipoles, energy_seed, False
    )
    np.testing.assert_array_equal(scalar_features["grad_positions"], np.zeros((2, 3)))
    np.testing.assert_array_equal(scalar_energy["grad_positions"], np.zeros((2, 3)))

    vector_ops = _ops(dtype, 1, 1)
    vector_cache = _cache(vector_ops, positions, node_graph_index)
    vector_multipoles = np.array(
        [[0.7, 0.08, -0.03, 0.11], [-0.4, -0.05, 0.09, 0.02]], dtype=dtype
    )
    vector_seed = _feature_seed(dtype, 2, 8)
    vector_features = vector_ops.diagnostic_adjoint_realspace_features(
        vector_cache, vector_multipoles, vector_seed, False
    )
    vector_energy = vector_ops.diagnostic_adjoint_realspace_energy(
        vector_cache, vector_multipoles, energy_seed, False
    )
    assert np.linalg.norm(vector_features["grad_positions"]) > 1.0e-3
    assert np.linalg.norm(vector_energy["grad_positions"]) > 1.0e-3
    _assert_gradient(vector_features["grad_positions"].sum(axis=0), np.zeros(3), dtype)
    _assert_gradient(vector_energy["grad_positions"].sum(axis=0), np.zeros(3), dtype)
    # A device compiler must not reassociate ``(x + offset) - (x + offset)``
    # into two differently rounded four-term expressions.  Pin each atom's
    # finite remainder as well as the zero net force so a common-mode CUDA
    # singular-gradient error cannot hide behind cancellation.
    expected_energy_position = np.array(
        [[-0.02878021, -0.52917209, 0.75947320], [0.02878021, 0.52917209, -0.75947320]],
        dtype=np.float64,
    )
    # Pinned from the upstream Torch FP64 autograd oracle at
    # graph_electrostatics e0eb8945.  The two individual rows catch a
    # singular common-mode CUDA residual that the zero-net-force assertion
    # cannot observe.
    expected_feature_position = np.array(
        [
            [11.10632317, -7.70086026, 8.21654335],
            [-11.10632317, 7.70086026, -8.21654335],
        ],
        dtype=np.float64,
    )
    _assert_gradient(
        vector_features["grad_positions"], expected_feature_position, dtype
    )
    _assert_gradient(vector_energy["grad_positions"], expected_energy_position, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_near_pair_position_adjoints_match_centered_differences(dtype):
    # Catches using a derivative of erf(r/(2s))/r instead of the specified
    # epsilon-shifted denominator derivative near the origin.
    positions = np.array([[0.1, -0.2, 0.3], [0.103, -0.201, 0.302]], dtype=np.float64)
    node_graph_index = np.zeros(2, dtype=np.int32)
    multipoles = np.array([[0.65], [-0.37]], dtype=dtype)
    feature_seed = np.array([[0.71], [-1.13]], dtype=dtype)
    energy_seed = np.array([0.89], dtype=dtype)
    base_multipoles = multipoles.astype(np.float64)

    expected_feature = _finite_difference(
        lambda candidate: float(
            np.sum(
                _feature_values(
                    candidate,
                    node_graph_index,
                    base_multipoles,
                    0,
                    0,
                    (0.51,),
                    include_self_interaction=False,
                )
                * feature_seed.astype(np.float64)
            )
        ),
        positions,
        1.0e-7,
    )
    expected_energy = _finite_difference(
        lambda candidate: float(
            np.dot(
                _energy_values(
                    candidate,
                    node_graph_index,
                    base_multipoles,
                    0,
                    1,
                    include_self_interaction=False,
                ),
                energy_seed.astype(np.float64),
            )
        ),
        positions,
        1.0e-7,
    )
    ops = _ops(dtype, 0, 0, (0.51,))
    cache = _cache(ops, positions, node_graph_index)
    feature = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, False
    )
    energy = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, False
    )
    _assert_gradient(feature["grad_positions"], expected_feature, dtype)
    _assert_gradient(energy["grad_positions"], expected_energy, dtype)


def test_fp32_reverse_preserves_sub_float32_position_differences():
    # Catches narrowing either cached positions or each subtraction operand to
    # FP32: both x coordinates then become 1.0 and the reverse spuriously
    # returns the pinned coincident-pair zero instead of these large gradients.
    delta = 2.0**-25
    positions = np.array(
        [[1.0, -0.25, 0.5], [1.0 + delta, -0.25, 0.5]], dtype=np.float64
    )
    assert np.float32(positions[0, 0]) == np.float32(positions[1, 0])
    multipoles = np.array([[0.75], [-0.4]], dtype=np.float32)
    feature_seed = np.array([[0.6], [-1.1]], dtype=np.float32)
    energy_seed = np.array([0.91], dtype=np.float32)
    node_graph_index = np.zeros(2, dtype=np.int32)
    # Hand-evaluated in FP64 from the approved epsilon-shifted dK/dr equation,
    # complete directed feature/energy losses, and promoted FP32 compact inputs.
    expected_feature = np.array(
        [[45930596.16923308, 0.0, 0.0], [-45930596.16923308, 0.0, 0.0]],
        dtype=np.float64,
    )
    expected_energy = np.array(
        [[2864896.466789268, 0.0, 0.0], [-2864896.466789268, 0.0, 0.0]],
        dtype=np.float64,
    )
    assert np.all(expected_feature[:, 0] != 0.0)
    assert np.all(expected_energy[:, 0] != 0.0)

    ops = _ops(np.dtype(np.float32), 0, 0, (0.51,))
    cache = _cache(ops, positions, node_graph_index)
    feature = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, False
    )
    energy = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, False
    )
    feature_positions = np.array(feature["grad_positions"], copy=True)
    energy_positions = np.array(energy["grad_positions"], copy=True)
    del feature, energy, cache, ops

    assert feature_positions.dtype == np.float64
    assert energy_positions.dtype == np.float64
    assert (
        np.count_nonzero(feature_positions[:, 0]),
        np.count_nonzero(energy_positions[:, 0]),
    ) == (2, 2)
    _assert_gradient(feature_positions, expected_feature, np.dtype(np.float32))
    _assert_gradient(energy_positions, expected_energy, np.dtype(np.float32))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("density_max_l,feature_max_l", ((0, 0), (1, 0), (0, 1)))
def test_lower_order_self_reverse_matches_captured_upstream_constants(
    dtype, density_max_l, feature_max_l
):
    # Catches omitting scalar-source self reverse or treating the shorter
    # two-term l=0 feature-self layout as if only the l=1/l=1 layout existed.
    positions = np.array([[0.17, -0.28, 0.39]], dtype=np.float64)
    node_graph_index = np.zeros(1, dtype=np.int32)
    multipoles = np.array([[0.62, 0.07, -0.04, 0.09]], dtype=dtype)[
        :, : (density_max_l + 1) ** 2
    ]
    feature_seed = np.array(
        [[0.31, -0.47, 0.59, -0.23, 0.71, -0.83, 0.19, 0.41]], dtype=dtype
    )[:, : 2 * (feature_max_l + 1) ** 2]
    energy_seed = np.array([-0.76], dtype=dtype)
    ops = _ops(dtype, density_max_l, feature_max_l)
    cache = _cache(ops, positions, node_graph_index)

    feature_off = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, False
    )
    feature_on = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, True
    )
    energy_off = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, False
    )
    energy_on = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, True
    )
    feature_off = {
        key: np.array(value, copy=True) for key, value in feature_off.items()
    }
    feature_on = {key: np.array(value, copy=True) for key, value in feature_on.items()}
    energy_off = {key: np.array(value, copy=True) for key, value in energy_off.items()}
    energy_on = {key: np.array(value, copy=True) for key, value in energy_on.items()}
    del cache, ops

    expected_feature = np.zeros_like(multipoles, dtype=np.float64)
    expected_feature[0, 0] = np.dot(
        feature_seed[0, :2].astype(np.float64), FEATURE_SELF_L0
    )
    energy_constants = ENERGY_SELF_L0 if density_max_l == 0 else ENERGY_SELF_L1
    expected_energy = (
        float(energy_seed[0])
        * multipoles.astype(np.float64)
        * energy_constants[None, :]
    )
    if dtype == np.dtype(np.float32):
        expected_feature = expected_feature.astype(np.float32)
        expected_energy = expected_energy.astype(np.float32)

    for result in (feature_off, energy_off):
        np.testing.assert_array_equal(
            result["grad_multipoles"], np.zeros_like(multipoles)
        )
        np.testing.assert_array_equal(result["grad_positions"], np.zeros((1, 3)))
        np.testing.assert_array_equal(result["grad_cell"], np.zeros((1, 3, 3)))
    _assert_gradient(feature_on["grad_multipoles"], expected_feature, dtype)
    _assert_gradient(energy_on["grad_multipoles"], expected_energy, dtype)
    for result in (feature_on, energy_on):
        np.testing.assert_array_equal(result["grad_positions"], np.zeros((1, 3)))
        np.testing.assert_array_equal(result["grad_cell"], np.zeros((1, 3, 3)))


@pytest.mark.parametrize("dtype", DTYPES)
def test_isolated_atom_self_reverse_changes_only_multipoles(dtype):
    # Catches missing analytic self reverse or any same-original-atom replica
    # interaction leaking into positions or the disabled-self result.
    positions = np.array([[0.17, -0.28, 0.39]], dtype=np.float64)
    node_graph_index = np.zeros(1, dtype=np.int32)
    multipoles = np.array([[0.62, 0.07, -0.04, 0.09]], dtype=dtype)
    feature_seed = np.array(
        [[0.31, -0.47, 0.59, -0.23, 0.71, -0.83, 0.19, 0.41]], dtype=dtype
    )
    energy_seed = np.array([-0.76], dtype=dtype)
    ops = _ops(dtype, 1, 1)
    cache = _cache(ops, positions, node_graph_index)

    feature_off = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, False
    )
    energy_off = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, False
    )
    for result in (feature_off, energy_off):
        np.testing.assert_array_equal(result["grad_multipoles"], np.zeros((1, 4)))
        np.testing.assert_array_equal(result["grad_positions"], np.zeros((1, 3)))
        np.testing.assert_array_equal(result["grad_cell"], np.zeros((1, 3, 3)))

    expected_feature = np.zeros((1, 4), dtype=np.float64)
    expected_feature[:, 0] = np.dot(feature_seed[0, :2], FEATURE_SELF_L1[:2])
    expected_feature[:, 1:4] = (
        feature_seed[0, 2:5] * FEATURE_SELF_L1[2:5]
        + feature_seed[0, 5:8] * FEATURE_SELF_L1[5:8]
    )
    expected_energy = (
        energy_seed.astype(np.float64)[:, None]
        * ENERGY_SELF_L1[None, :]
        * multipoles.astype(np.float64)
    )
    if dtype == np.dtype(np.float32):
        expected_feature = expected_feature.astype(np.float32)
        expected_energy = expected_energy.astype(np.float32)
    feature_on = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, True
    )
    energy_on = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, True
    )
    _assert_gradient(feature_on["grad_multipoles"], expected_feature, dtype)
    _assert_gradient(energy_on["grad_multipoles"], expected_energy, dtype)
    for result in (feature_on, energy_on):
        np.testing.assert_array_equal(result["grad_positions"], np.zeros((1, 3)))
        np.testing.assert_array_equal(result["grad_cell"], np.zeros((1, 3, 3)))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "reverse_strategy",
    (
        "owner",
        "pair_atomic",
        "materialized_forward",
    ),
)
def test_translation_and_cell_independence_hold_for_forwards_and_adjoints(
    dtype, reverse_strategy
):
    # Catches a hidden cell operand or dropping the receiver-position branch,
    # either of which breaks internal-force balance.
    positions = np.array(
        [[0.125, 0.25, -0.375], [0.875, -0.5, 0.75], [1.375, 0.625, 0.125]],
        dtype=np.float64,
    )
    translated = positions + np.array([4.0, -2.0, 1.0])
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    feature_seed = _feature_seed(dtype, 3, 8)
    energy_seed = np.array([1.11], dtype=dtype)
    cell_a = np.array([np.diag([6.0, 7.0, 8.0])])
    cell_b = np.array([[[20.0, 1.0, -2.0], [0.0, 5.0, 0.5], [0.0, 0.0, 3.0]]])
    ops = _ops(dtype, 1, 1)
    _select_reverse_strategy(ops, reverse_strategy)

    def evaluate(candidate_positions, cell):
        cache = _cache(ops, candidate_positions, node_graph_index, cell)
        return (
            ops.realspace_features(cache, multipoles, False),
            ops.realspace_energy(cache, multipoles, False),
            ops.diagnostic_adjoint_realspace_features(
                cache, multipoles, feature_seed, False
            ),
            ops.diagnostic_adjoint_realspace_energy(
                cache, multipoles, energy_seed, False
            ),
        )

    reference = evaluate(positions, cell_a)
    fixed_cartesian = evaluate(positions, cell_b)
    translated_result = evaluate(translated, cell_b)
    for candidate in (fixed_cartesian,):
        np.testing.assert_array_equal(candidate[0], reference[0])
        np.testing.assert_array_equal(candidate[1], reference[1])
        for candidate_result, reference_result in zip(
            candidate[2:], reference[2:], strict=True
        ):
            _assert_gradient(
                candidate_result["grad_multipoles"],
                reference_result["grad_multipoles"],
                dtype,
            )
            _assert_gradient(
                candidate_result["grad_positions"],
                reference_result["grad_positions"],
                dtype,
            )
            np.testing.assert_array_equal(
                candidate_result["grad_cell"], np.zeros((1, 3, 3))
            )
    for candidate_value, reference_value in zip(
        translated_result[:2], reference[:2], strict=True
    ):
        np.testing.assert_allclose(
            candidate_value,
            reference_value,
            **_gradient_tolerances(dtype, reference_value),
        )
    for candidate_result, reference_result in zip(
        translated_result[2:], reference[2:], strict=True
    ):
        _assert_gradient(
            candidate_result["grad_multipoles"],
            reference_result["grad_multipoles"],
            dtype,
        )
        _assert_gradient(
            candidate_result["grad_positions"],
            reference_result["grad_positions"],
            dtype,
        )
        np.testing.assert_array_equal(
            candidate_result["grad_cell"], np.zeros((1, 3, 3))
        )
    for result in reference[2:]:
        _assert_gradient(result["grad_positions"].sum(axis=0), np.zeros(3), dtype)
    _assert_reverse_strategy_was_used(ops, reverse_strategy)


@pytest.mark.parametrize("dtype", DTYPES)
def test_reverse_workspace_is_linear_and_launches_atom_component_owners(dtype):
    # Catches retaining replica/pair tables or serializing quadratic reverse
    # work in one graph lane. The diagnostic extents are set by the launches.
    itemsize = dtype.itemsize
    feature_bytes = []
    energy_bytes = []
    for n_nodes in (4, 11):
        positions = np.arange(n_nodes * 3, dtype=np.float64).reshape(n_nodes, 3)
        positions /= 17.0
        node_graph_index = np.zeros(n_nodes, dtype=np.int32)
        multipoles = (
            np.linspace(-0.31, 0.47, n_nodes * 4, dtype=np.float64)
            .reshape(n_nodes, 4)
            .astype(dtype)
        )
        feature_seed = _feature_seed(dtype, n_nodes, 8)
        energy_seed = np.array([0.79], dtype=dtype)

        feature_ops = _ops(dtype, 1, 1)
        _select_reverse_strategy(feature_ops, "owner")
        feature_cache = _cache(feature_ops, positions, node_graph_index)
        feature_ops.diagnostic_adjoint_realspace_features(
            feature_cache, multipoles, feature_seed, False, use_workspace=True
        )
        assert feature_cache.reciprocal_allocation_bytes == 0
        assert feature_ops.workspace_allocation_bytes == (
            16 * n_nodes * itemsize + (3 * n_nodes + 9) * 8
        )
        feature_bytes.append(feature_ops.workspace_allocation_bytes)
        feature_diagnostics = feature_ops.realspace_reverse_launch_diagnostics()
        assert feature_diagnostics["feature_multipole_league_extent"] == 4 * n_nodes
        assert feature_diagnostics["feature_position_league_extent"] == 3 * n_nodes

        energy_ops = _ops(dtype, 1, 1)
        _select_reverse_strategy(energy_ops, "owner")
        energy_cache = _cache(energy_ops, positions, node_graph_index)
        energy_ops.diagnostic_adjoint_realspace_energy(
            energy_cache, multipoles, energy_seed, False, use_workspace=True
        )
        assert energy_cache.reciprocal_allocation_bytes == 0
        assert energy_ops.workspace_allocation_bytes == (
            (8 * n_nodes + 1) * itemsize + (3 * n_nodes + 9) * 8
        )
        energy_bytes.append(energy_ops.workspace_allocation_bytes)
        energy_diagnostics = energy_ops.realspace_reverse_launch_diagnostics()
        assert energy_diagnostics["energy_multipole_league_extent"] == 4 * n_nodes
        assert energy_diagnostics["energy_position_league_extent"] == 3 * n_nodes

    assert feature_bytes[1] - feature_bytes[0] == 7 * (16 * itemsize + 24)
    assert energy_bytes[1] - energy_bytes[0] == 7 * (8 * itemsize + 24)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("density_max_l,feature_max_l", L_COMBINATIONS)
@pytest.mark.parametrize("include_self", (False, True), ids=("no-self", "self"))
def test_forced_pair_atomic_reverse_matches_owner_and_reports_pair_work(
    dtype, density_max_l, feature_max_l, include_self
):
    """The experimental strategy must change launches, never the equations."""
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, density_max_l, charged=True)
    feature_dim = 2 * (feature_max_l + 1) ** 2
    feature_seed = _feature_seed(dtype, 3, feature_dim)
    energy_seed = np.array([1.37], dtype=dtype)

    owner_ops = _ops(dtype, density_max_l, feature_max_l)
    _select_reverse_strategy(owner_ops, "owner")
    owner_cache = _cache(owner_ops, positions, node_graph_index)
    owner_feature = owner_ops.diagnostic_adjoint_realspace_features(
        owner_cache, multipoles, feature_seed, include_self
    )
    owner_energy = owner_ops.diagnostic_adjoint_realspace_energy(
        owner_cache, multipoles, energy_seed, include_self
    )
    assert owner_ops.realspace_reverse_strategy == "owner"

    pair_ops = _ops(dtype, density_max_l, feature_max_l)
    pair_ops.set_realspace_reverse_strategy("pair_atomic")
    pair_cache = _cache(pair_ops, positions, node_graph_index)
    pair_feature = pair_ops.diagnostic_adjoint_realspace_features(
        pair_cache, multipoles, feature_seed, include_self, use_workspace=True
    )
    pair_energy = pair_ops.diagnostic_adjoint_realspace_energy(
        pair_cache, multipoles, energy_seed, include_self, use_workspace=True
    )

    for candidate, reference in (
        (pair_feature, owner_feature),
        (pair_energy, owner_energy),
    ):
        _assert_gradient(
            candidate["grad_multipoles"], reference["grad_multipoles"], dtype
        )
        _assert_gradient(
            candidate["grad_positions"], reference["grad_positions"], dtype
        )
        np.testing.assert_array_equal(candidate["grad_cell"], reference["grad_cell"])

    diagnostics = pair_ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["strategy"] == "pair_atomic"
    assert diagnostics["feature_pair_league_extent"] == 3 * 3 * 2
    assert diagnostics["energy_pair_league_extent"] == 3 * 3
    assert diagnostics["feature_pair_atomic_selected_count"] == 1
    assert diagnostics["energy_pair_atomic_selected_count"] == 1
    assert diagnostics["feature_selected_count"] == 1
    assert diagnostics["energy_selected_count"] == 1
    assert diagnostics["atomic_accumulator_bytes"] == (3 * (density_max_l + 1) ** 2 * 8)
    assert pair_cache.reciprocal_allocation_bytes == 0


def test_pair_atomic_larger_accumulation_matches_independent_centered_differences():
    """Exercise many atomic contributions without using owner as the oracle."""
    dtype = np.dtype(np.float64)
    n_nodes = 7
    positions = np.linspace(-0.83, 1.37, n_nodes * 3).reshape(n_nodes, 3)
    positions[:, 1] += np.linspace(0.0, 0.41, n_nodes)
    positions[:, 2] -= np.linspace(0.0, 0.29, n_nodes) ** 2
    node_graph_index = np.zeros(n_nodes, dtype=np.int32)
    multipoles = np.linspace(-0.37, 0.53, n_nodes * 4).reshape(n_nodes, 4)
    multipoles[:, 1::2] *= -0.43
    feature_seed = _feature_seed(dtype, n_nodes, 8)
    energy_seed = np.array([0.83])
    expected = _canonical_expected_gradients(
        dtype,
        positions,
        node_graph_index,
        multipoles,
        1,
        1,
        feature_seed,
        energy_seed,
        feature_self=False,
        energy_self=False,
    )
    ops = _ops(dtype, 1, 1)
    _select_reverse_strategy(ops, "pair_atomic")
    cache = _cache(ops, positions, node_graph_index)
    feature = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, False
    )
    energy = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, False
    )
    for name in ("grad_multipoles", "grad_positions"):
        _assert_gradient(feature[name], expected[f"feature_{name}"], dtype)
        _assert_gradient(energy[name], expected[f"energy_{name}"], dtype)
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["feature_pair_league_extent"] == n_nodes * n_nodes * 2
    assert diagnostics["energy_pair_league_extent"] == n_nodes * n_nodes
    _assert_reverse_strategy_was_used(ops, "pair_atomic")


def test_realspace_reverse_strategy_rejects_unknown_name():
    ops = _ops(np.dtype(np.float64), 1, 1)
    with pytest.raises(ValueError, match="owner.*pair_atomic"):
        ops.set_realspace_reverse_strategy("quadratic_edges")


@pytest.mark.parametrize(
    "strategy,tile_width",
    (("pair_tile", 16), ("pair_tile", 64), ("pair_materialized", None)),
)
@pytest.mark.parametrize("dtype", DTYPES)
def _retired_phase_b_reverse_candidates_match_owner(strategy, tile_width, dtype):
    """A missing candidate branch or changed equation must fail this test."""
    if (
        strategy == "pair_tile"
        and tile_width == 64
        and symmetrix.selected_backend()["backend"] == "cpu"
    ):
        pytest.skip("64-member teams exceed this CPU backend's team capacity")
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    feature_seed = _feature_seed(dtype, 3, 8)
    energy_seed = np.array([1.37], dtype=dtype)

    owner_ops = _ops(dtype, 1, 1)
    _select_reverse_strategy(owner_ops, "owner")
    owner_cache = _cache(owner_ops, positions, node_graph_index)
    owner = (
        owner_ops.diagnostic_adjoint_realspace_features(
            owner_cache, multipoles, feature_seed, True
        ),
        owner_ops.diagnostic_adjoint_realspace_energy(
            owner_cache, multipoles, energy_seed, True
        ),
    )

    candidate_ops = _ops(dtype, 1, 1)
    candidate_ops.set_realspace_reverse_strategy(strategy)
    if strategy == "pair_tile":
        candidate_ops.set_realspace_pair_tile_width(tile_width)
    else:
        candidate_ops.set_realspace_materialized_budget_bytes(3840)
    candidate_cache = _cache(candidate_ops, positions, node_graph_index)
    candidate = (
        candidate_ops.diagnostic_adjoint_realspace_features(
            candidate_cache, multipoles, feature_seed, True, use_workspace=True
        ),
        candidate_ops.diagnostic_adjoint_realspace_energy(
            candidate_cache, multipoles, energy_seed, True, use_workspace=True
        ),
    )
    for observed, expected in zip(candidate, owner, strict=True):
        _assert_gradient(
            observed["grad_multipoles"], expected["grad_multipoles"], dtype
        )
        _assert_gradient(observed["grad_positions"], expected["grad_positions"], dtype)
        np.testing.assert_array_equal(observed["grad_cell"], expected["grad_cell"])

    diagnostics = candidate_ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["requested_strategy"] == strategy
    assert diagnostics["resolved_strategy"] == strategy
    assert diagnostics["strategy_reason"] == "forced_override"
    if strategy == "pair_tile":
        assert diagnostics["tile_width"] == tile_width
        assert diagnostics["feature_pair_tile_selected_count"] == 1
        assert diagnostics["energy_pair_tile_selected_count"] == 1
        assert diagnostics["feature_pair_tile_league_extent"] > 0
        assert diagnostics["energy_pair_tile_league_extent"] > 0
        assert diagnostics["partial_accumulator_bytes"] > 0
    else:
        assert diagnostics["materialized_table_build_count"] == 1
        assert diagnostics["materialized_table_reuse_count"] == 1
        assert diagnostics["materialized_radial_entries"] == 240
        assert diagnostics["materialized_radial_bytes"] == 3840


@pytest.mark.parametrize("tile_width", (8, 16, 32, 64))
def _retired_pair_tile_accepts_only_qualified_widths(tile_width):
    ops = _ops(np.dtype(np.float64), 1, 1)
    ops.set_realspace_pair_tile_width(tile_width)
    ops.set_realspace_reverse_strategy("pair_tile")
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["tile_width"] == tile_width


@pytest.mark.parametrize("tile_width", (0, 7, 128))
def _retired_pair_tile_rejects_unqualified_widths(tile_width):
    ops = _ops(np.dtype(np.float64), 1, 1)
    with pytest.raises(ValueError, match="8, 16, 32, or 64"):
        ops.set_realspace_pair_tile_width(tile_width)


def _retired_pair_tile_width64_checks_scratch_and_league_extents():
    ops = _ops(np.dtype(np.float64), 1, 1)
    ops.set_realspace_pair_tile_width(64)
    assert ops.checked_realspace_pair_tile_scratch_bytes() == 64 * 7 * 8
    assert ops.checked_realspace_pair_tile_feature_league_extent(3) == 6
    assert ops.checked_realspace_pair_tile_energy_league_extent(3) == 3
    with pytest.raises(OverflowError, match="tiled feature league"):
        ops.checked_realspace_pair_tile_feature_league_extent(np.iinfo(np.uintp).max)
    with pytest.raises(OverflowError, match="tiled energy league"):
        ops.checked_realspace_pair_tile_energy_league_extent(np.iinfo(np.uintp).max)


def _retired_pair_tile_width64_rejects_undersized_backend_before_launch():
    if symmetrix.selected_backend()["backend"] != "cpu":
        pytest.skip("host team-capacity rejection is CPU-specific")
    ops = _ops(np.dtype(np.float64), 1, 1)
    ops.set_realspace_pair_tile_width(64)
    ops.set_realspace_reverse_strategy("pair_tile")
    positions = np.array([[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28]], dtype=np.float64)
    cache = _cache(ops, positions, np.zeros(2, dtype=np.int32))
    multipoles = _multipoles(np.dtype(np.float64), 1, charged=True)[:2]
    feature_seed = _feature_seed(np.dtype(np.float64), 2, 8)
    with pytest.raises(ValueError, match="team width 64.*capacity"):
        ops.diagnostic_adjoint_realspace_features(
            cache, multipoles, feature_seed, False
        )
    with pytest.raises(ValueError, match="team width 64.*capacity"):
        ops.diagnostic_adjoint_realspace_energy(
            cache, multipoles, np.array([1.0]), False
        )


def _retired_radial_materialized_estimator_checks_overflow_and_budget():
    ops = _ops(np.dtype(np.float64), 1, 1)
    assert ops.checked_realspace_materialized_table_bytes(3) == 3840
    with pytest.raises(OverflowError, match="materialized radial"):
        ops.checked_realspace_materialized_table_bytes(np.iinfo(np.uintp).max)

    ops.set_realspace_materialized_budget_bytes(3839)
    ops.set_realspace_reverse_strategy("pair_materialized")
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    cache = _cache(ops, positions, np.zeros(3, dtype=np.int32))
    ops.diagnostic_adjoint_realspace_features(
        cache,
        _multipoles(np.dtype(np.float64), 1, True),
        _feature_seed(np.dtype(np.float64), 3, 8),
        False,
    )
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["requested_strategy"] == "pair_materialized"
    assert diagnostics["resolved_strategy"] == "pair_atomic"
    assert diagnostics["strategy_reason"] == "materialized_budget_rejected"
    assert diagnostics["materialized_admission_rejection_count"] == 1
    assert diagnostics["materialized_radial_bytes"] == 0


def test_materialized_forward_has_complete_checked_accounting_and_fallback():
    """Phase C must account for Torch-style edge state before allocating it."""
    ops = _ops(np.dtype(np.float64), 1, 1)
    radial_bytes = ops.checked_realspace_materialized_table_bytes(3)
    full_bytes = ops.checked_realspace_full_materialized_bytes(3, True)
    assert full_bytes > radial_bytes
    assert ops.checked_realspace_full_materialized_bytes(3, False) < full_bytes
    with pytest.raises(OverflowError, match="full materialized"):
        ops.checked_realspace_full_materialized_bytes(np.iinfo(np.uintp).max, True)

    forward_bytes = ops.checked_realspace_full_materialized_bytes(3, False)
    ops.set_realspace_materialized_budget_bytes(forward_bytes - 1)
    ops.set_realspace_reverse_strategy("materialized_forward")
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    cache = _cache(ops, positions, np.zeros(3, dtype=np.int32))
    ops.realspace_features(cache, _multipoles(np.dtype(np.float64), 1, True), False)
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["requested_strategy"] == "materialized_forward"
    assert diagnostics["resolved_strategy"] == "pair_atomic"
    assert diagnostics["strategy_reason"] == "full_materialized_budget_rejected"
    assert diagnostics["full_materialized_admission_rejection_count"] == 1
    assert diagnostics["full_materialized_retained_bytes"] == 0


def _retired_full_materialized_reducer_selection():
    """An unknown reducer must not silently run the serial baseline."""
    ops = _ops(np.dtype(np.float64), 1, 1)
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["full_materialized_reducer"] == "serial"

    ops.set_realspace_full_materialized_reducer("atomic_scatter")
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["full_materialized_reducer"] == "atomic_scatter"

    with pytest.raises(ValueError, match="full-materialized reducer"):
        ops.set_realspace_full_materialized_reducer("silently_wrong")


@pytest.mark.parametrize("dtype", DTYPES)
def _retired_atomic_energy_reduction_matches_independent_energy(dtype):
    """The optimized mode must not retain the one-worker complete edge scan."""
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    graph = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    expected = _energy_values(
        positions,
        graph,
        multipoles,
        1,
        1,
        include_self_interaction=True,
    ).astype(dtype)

    ops = _ops(dtype, 1, 1)
    ops.set_realspace_materialized_budget_bytes(np.iinfo(np.uintp).max)
    ops.set_realspace_reverse_strategy("pair_full_materialized")
    ops.set_realspace_full_materialized_reducer("atomic_scatter")
    observed = ops.realspace_energy(_cache(ops, positions, graph), multipoles, True)

    np.testing.assert_allclose(
        observed, expected, **_gradient_tolerances(dtype, expected)
    )
    diagnostics = ops.realspace_energy_launch_diagnostics()
    assert diagnostics["full_materialized_parallel_selected_count"] == 1
    assert diagnostics["full_materialized_serial_selected_count"] == 0
    assert diagnostics["full_materialized_parallel_edge_extent"] == 3 * 2 * 16


@pytest.mark.parametrize("dtype", DTYPES)
def _retired_atomic_feature_scatter_matches_independent_features(dtype):
    """Atomic scatter must replace each receiver worker's complete source scan."""
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    graph = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    expected = _feature_values(
        positions,
        graph,
        multipoles,
        1,
        1,
        (0.51, 1.07),
        include_self_interaction=True,
    ).astype(dtype)

    ops = _ops(dtype, 1, 1)
    ops.set_realspace_materialized_budget_bytes(np.iinfo(np.uintp).max)
    ops.set_realspace_reverse_strategy("pair_full_materialized")
    ops.set_realspace_full_materialized_reducer("atomic_scatter")
    observed = ops.realspace_features(_cache(ops, positions, graph), multipoles, True)

    np.testing.assert_allclose(
        observed, expected, **_gradient_tolerances(dtype, expected)
    )
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["full_materialized_feature_atomic_selected_count"] == 1
    assert diagnostics["full_materialized_feature_serial_selected_count"] == 0
    assert diagnostics["full_materialized_feature_parallel_edge_extent"] == 3 * 2 * 16


@pytest.mark.parametrize("dtype", DTYPES)
def _retired_segmented_feature_reduction_matches_independent_features(
    dtype,
):
    """Receiver segments must parallel-reduce without changing feature semantics."""
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    graph = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    expected = _feature_values(
        positions,
        graph,
        multipoles,
        1,
        1,
        (0.51, 1.07),
        include_self_interaction=True,
    ).astype(dtype)

    ops = _ops(dtype, 1, 1)
    ops.set_realspace_materialized_budget_bytes(np.iinfo(np.uintp).max)
    ops.set_realspace_reverse_strategy("pair_full_materialized")
    ops.set_realspace_full_materialized_reducer("segmented")
    observed = ops.realspace_features(_cache(ops, positions, graph), multipoles, True)

    np.testing.assert_allclose(
        observed, expected, **_gradient_tolerances(dtype, expected)
    )
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["full_materialized_feature_segmented_selected_count"] == 1
    assert diagnostics["full_materialized_feature_serial_selected_count"] == 0
    assert diagnostics["full_materialized_feature_atomic_selected_count"] == 0
    assert diagnostics["full_materialized_feature_parallel_edge_extent"] == 3 * 2 * 16


def _retired_parallel_reducers_omit_forward_contribution_tables():
    """Manual reverse must not force retention of unused forward contributions."""
    ops = _ops(np.dtype(np.float64), 1, 1)
    serial_bytes = ops.checked_realspace_full_materialized_bytes(3, False)
    ops.set_realspace_full_materialized_reducer("atomic_scatter")
    parallel_bytes = ops.checked_realspace_full_materialized_bytes(3, False)
    assert parallel_bytes < serial_bytes

    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    graph = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(np.dtype(np.float64), 1, charged=True)
    ops.set_realspace_materialized_budget_bytes(np.iinfo(np.uintp).max)
    ops.set_realspace_full_materialized_reducer("serial")
    ops.set_realspace_reverse_strategy("pair_full_materialized")
    cache = _cache(ops, positions, graph)
    ops.realspace_features(cache, multipoles, False)
    serial_retained = ops.realspace_reverse_launch_diagnostics()[
        "full_materialized_retained_bytes"
    ]

    ops.set_realspace_full_materialized_reducer("atomic_scatter")
    assert (
        ops.realspace_reverse_launch_diagnostics()["full_materialized_retained_bytes"]
        == 0
    )
    ops.realspace_features(cache, multipoles, False)
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["full_materialized_retained_bytes"] < serial_retained
    assert diagnostics["full_materialized_admission_bytes"] == parallel_bytes


@pytest.mark.parametrize("dtype", DTYPES)
def _retired_full_materialized_reverse_matches_owner_and_reports_directed_edges(
    dtype,
):
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    graph = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    feature_seed = _feature_seed(dtype, 3, 8)
    energy_seed = np.array([1.37], dtype=dtype)

    owner_ops = _ops(dtype, 1, 1)
    _select_reverse_strategy(owner_ops, "owner")
    owner_cache = _cache(owner_ops, positions, graph)
    owner_feature = owner_ops.diagnostic_adjoint_realspace_features(
        owner_cache, multipoles, feature_seed, True
    )
    owner_energy = owner_ops.diagnostic_adjoint_realspace_energy(
        owner_cache, multipoles, energy_seed, True
    )

    candidate_ops = _ops(dtype, 1, 1)
    candidate_ops.set_realspace_materialized_budget_bytes(1 << 30)
    candidate_ops.set_realspace_reverse_strategy("pair_full_materialized")
    candidate_cache = _cache(candidate_ops, positions, graph)
    candidate_feature = candidate_ops.diagnostic_adjoint_realspace_features(
        candidate_cache, multipoles, feature_seed, True
    )
    candidate_energy = candidate_ops.diagnostic_adjoint_realspace_energy(
        candidate_cache, multipoles, energy_seed, True
    )
    for observed, expected in (
        (candidate_feature, owner_feature),
        (candidate_energy, owner_energy),
    ):
        _assert_gradient(
            observed["grad_multipoles"], expected["grad_multipoles"], dtype
        )
        _assert_gradient(observed["grad_positions"], expected["grad_positions"], dtype)
        np.testing.assert_array_equal(observed["grad_cell"], expected["grad_cell"])

    diagnostics = candidate_ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["resolved_strategy"] == "pair_full_materialized"
    assert diagnostics["full_materialized_table_build_count"] == 1
    assert diagnostics["full_materialized_table_reuse_count"] == 1
    assert diagnostics["full_materialized_feature_directed_edges"] == 3 * 2 * 16
    assert diagnostics["full_materialized_energy_directed_edges"] == 3 * 2 * 16
    assert diagnostics["full_materialized_feature_pair_atomic_reverse_count"] == 0
    assert diagnostics["full_materialized_energy_pair_atomic_reverse_count"] == 0
    assert diagnostics["feature_materialized_selected_count"] == 1
    assert diagnostics["energy_materialized_selected_count"] == 1
    assert diagnostics["full_materialized_retained_bytes"] > 0
    assert (
        diagnostics["full_materialized_admission_bytes"]
        > diagnostics["full_materialized_retained_bytes"]
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_materialized_forward_uses_segmented_forward_and_pair_atomic_reverse(dtype):
    """The named strategy must select the qualified fixed execution schedule."""
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    graph = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    feature_seed = _feature_seed(dtype, 3, 8)
    energy_seed = np.array([1.37], dtype=dtype)

    owner_ops = _ops(dtype, 1, 1)
    _select_reverse_strategy(owner_ops, "owner")
    owner_cache = _cache(owner_ops, positions, graph)
    owner_feature = owner_ops.diagnostic_adjoint_realspace_features(
        owner_cache, multipoles, feature_seed, True
    )
    owner_energy = owner_ops.diagnostic_adjoint_realspace_energy(
        owner_cache, multipoles, energy_seed, True
    )

    candidate_ops = _ops(dtype, 1, 1)
    candidate_ops.set_realspace_materialized_budget_bytes(np.iinfo(np.uintp).max)
    candidate_ops.set_realspace_reverse_strategy("materialized_forward")
    candidate_cache = _cache(candidate_ops, positions, graph)
    candidate_ops.realspace_features(candidate_cache, multipoles, True)
    candidate_ops.realspace_energy(candidate_cache, multipoles, True)
    candidate_feature = candidate_ops.diagnostic_adjoint_realspace_features(
        candidate_cache, multipoles, feature_seed, True
    )
    candidate_energy = candidate_ops.diagnostic_adjoint_realspace_energy(
        candidate_cache, multipoles, energy_seed, True
    )

    for observed, expected in (
        (candidate_feature, owner_feature),
        (candidate_energy, owner_energy),
    ):
        _assert_gradient(
            observed["grad_multipoles"], expected["grad_multipoles"], dtype
        )
        _assert_gradient(observed["grad_positions"], expected["grad_positions"], dtype)
        np.testing.assert_array_equal(observed["grad_cell"], expected["grad_cell"])

    diagnostics = candidate_ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["requested_strategy"] == "materialized_forward"
    assert diagnostics["resolved_strategy"] == "materialized_forward"
    assert diagnostics["materialized_forward_reducer"] == "segmented"
    assert diagnostics["full_materialized_feature_pair_atomic_reverse_count"] == 1
    assert diagnostics["full_materialized_energy_pair_atomic_reverse_count"] == 1
    assert diagnostics["feature_pair_atomic_selected_count"] == 1
    assert diagnostics["energy_pair_atomic_selected_count"] == 1
    assert diagnostics["feature_materialized_selected_count"] == 0
    assert diagnostics["energy_materialized_selected_count"] == 0


@pytest.mark.parametrize(
    "retired_strategy",
    ("pair_tile", "pair_materialized", "pair_full_materialized"),
)
def test_retired_cuda_reverse_strategies_are_rejected(retired_strategy):
    """Retired experiments must not remain selectable by runtime policy."""
    ops = _ops(np.dtype(np.float64), 1, 1)
    with pytest.raises(ValueError, match="materialized_forward"):
        ops.set_realspace_reverse_strategy(retired_strategy)


def test_retired_cuda_strategy_controls_are_not_exposed():
    ops = _ops(np.dtype(np.float64), 1, 1)
    for name in (
        "set_realspace_pair_tile_width",
        "set_realspace_full_materialized_reducer",
        "checked_realspace_pair_tile_scratch_bytes",
    ):
        assert not hasattr(ops, name)


def _retired_full_materialized_state_release():
    ops = _ops(np.dtype(np.float64), 1, 1)
    ops.set_realspace_materialized_budget_bytes(1 << 30)
    ops.set_realspace_reverse_strategy("pair_full_materialized")
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    cache = _cache(ops, positions, np.zeros(3, dtype=np.int32))
    ops.diagnostic_adjoint_realspace_features(
        cache,
        _multipoles(np.dtype(np.float64), 1, True),
        _feature_seed(np.dtype(np.float64), 3, 8),
        False,
    )
    assert (
        ops.realspace_reverse_launch_diagnostics()["full_materialized_retained_bytes"]
        > 0
    )
    ops.set_realspace_reverse_strategy("pair_atomic")
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["resolved_strategy"] == "pair_atomic"
    assert diagnostics["full_materialized_retained_bytes"] == 0


def _retired_radial_tables_reuse_and_invalidate_with_positions():
    ops = _ops(np.dtype(np.float64), 1, 1)
    ops.set_realspace_materialized_budget_bytes(3840)
    ops.set_realspace_reverse_strategy("pair_materialized")
    positions = np.array(
        [[-0.23, 0.17, 0.41], [0.76, -0.34, 0.28], [1.31, 0.82, -0.57]],
        dtype=np.float64,
    )
    graph = np.zeros(3, dtype=np.int32)
    cell = np.eye(3, dtype=np.float64)[None, :, :] * 8.0
    cache = ops.realspace_geometry_cache(positions, cell, graph)
    multipoles = _multipoles(np.dtype(np.float64), 1, True)
    feature_seed = _feature_seed(np.dtype(np.float64), 3, 8)
    energy_seed = np.array([1.37])
    ops.diagnostic_adjoint_realspace_features(cache, multipoles, feature_seed, False)
    ops.diagnostic_adjoint_realspace_energy(cache, multipoles, energy_seed, False)
    ops.diagnostic_adjoint_realspace_features(cache, multipoles, feature_seed, False)
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["materialized_table_build_count"] == 1
    assert diagnostics["materialized_table_reuse_count"] == 2

    moved = positions.copy()
    moved[1, 0] += 0.03
    ops.realspace_geometry_cache_into(cache, moved, cell, graph)
    ops.diagnostic_adjoint_realspace_features(cache, multipoles, feature_seed, False)
    diagnostics = ops.realspace_reverse_launch_diagnostics()
    assert diagnostics["materialized_table_build_count"] == 2
    assert diagnostics["materialized_table_invalidation_count"] == 1


def test_pair_atomic_accumulator_byte_count_rejects_overflow_before_allocation():
    ops = _ops(np.dtype(np.float64), 1, 1)
    assert ops.checked_realspace_atomic_accumulator_bytes(3) == 3 * 4 * 8
    too_many_nodes = np.iinfo(np.uintp).max // (4 * 8) + 1
    with pytest.raises(OverflowError, match="real-space atomic accumulator"):
        ops.checked_realspace_atomic_accumulator_bytes(too_many_nodes)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("max_l", (0, 1))
@pytest.mark.parametrize(
    "positions",
    (
        np.array([[0.2, -0.3, 0.4], [0.2, -0.3, 0.4]], dtype=np.float64),
        np.array([[0.1, -0.2, 0.3], [0.103, -0.201, 0.302]], dtype=np.float64),
    ),
    ids=("coincident", "near"),
)
def test_pair_atomic_preserves_owner_coincident_and_near_pair_semantics(
    dtype, max_l, positions
):
    node_graph_index = np.zeros(2, dtype=np.int32)
    multipoles = np.array(
        [[0.7, 0.08, -0.03, 0.11], [-0.4, -0.05, 0.09, 0.02]], dtype=dtype
    )[:, : (max_l + 1) ** 2]
    feature_seed = _feature_seed(dtype, 2, 2 * (max_l + 1) ** 2)
    energy_seed = np.array([0.91], dtype=dtype)

    def evaluate(strategy):
        ops = _ops(dtype, max_l, max_l)
        _select_reverse_strategy(ops, strategy)
        cache = _cache(ops, positions, node_graph_index)
        return (
            ops.diagnostic_adjoint_realspace_features(
                cache, multipoles, feature_seed, False
            ),
            ops.diagnostic_adjoint_realspace_energy(
                cache, multipoles, energy_seed, False
            ),
        )

    owner = evaluate("owner")
    for strategy in ("pair_atomic", "materialized_forward"):
        pair = evaluate(strategy)
        for candidate, reference in zip(pair, owner, strict=True):
            _assert_gradient(
                candidate["grad_multipoles"], reference["grad_multipoles"], dtype
            )
            _assert_gradient(
                candidate["grad_positions"], reference["grad_positions"], dtype
            )
            np.testing.assert_array_equal(
                candidate["grad_cell"], reference["grad_cell"]
            )


@pytest.mark.skipif(
    os.environ.get("SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE") != "1",
    reason="set SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE=1 for pinned Torch autograd",
)
@pytest.mark.parametrize("density_max_l,feature_max_l", L_COMBINATIONS)
@pytest.mark.parametrize(
    "reverse_strategy",
    (
        "owner",
        "pair_atomic",
        "materialized_forward",
    ),
)
def test_fp64_adjoints_match_pinned_torch_autograd(
    density_max_l, feature_max_l, reverse_strategy
):
    torch = pytest.importorskip("torch")
    checkout = optional_checkout("SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE")
    oracle_module = import_graph_electrostatics_realspace(checkout)
    RealSpaceFiniteDiffereneEnergy = oracle_module.RealSpaceFiniteDiffereneEnergy
    RealSpaceFiniteDifferenceElectrostaticFeatures = (
        oracle_module.RealSpaceFiniteDifferenceElectrostaticFeatures
    )

    positions = np.array(
        [[0.13, -0.27, 0.31], [0.91, 0.42, -0.68], [-0.52, 0.88, 0.73]]
    )
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(np.dtype(np.float64), density_max_l, charged=True)
    feature_dim = 2 * (feature_max_l + 1) ** 2
    feature_seed = _feature_seed(np.dtype(np.float64), 3, feature_dim)
    energy_seed = np.array([-0.67])
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch_batch = torch.as_tensor(node_graph_index, dtype=torch.long)
        torch_multipoles = torch.tensor(multipoles, requires_grad=True)
        torch_positions = torch.tensor(positions, requires_grad=True)
        feature_oracle = RealSpaceFiniteDifferenceElectrostaticFeatures(
            density_max_l, 0.73, feature_max_l, [0.51, 1.07], True
        ).to(dtype=torch.float64)
        feature_values = feature_oracle(torch_multipoles, torch_positions, torch_batch)[
            0
        ]
        torch_feature_grad = torch.autograd.grad(
            (feature_values * torch.as_tensor(feature_seed)).sum(),
            (torch_multipoles, torch_positions),
        )

        torch_multipoles_energy = torch.tensor(multipoles, requires_grad=True)
        torch_positions_energy = torch.tensor(positions, requires_grad=True)
        energy_oracle = RealSpaceFiniteDiffereneEnergy(density_max_l, 0.73, True).to(
            dtype=torch.float64
        )
        energy_values = energy_oracle(
            torch_multipoles_energy, torch_positions_energy, torch_batch
        )
        torch_energy_grad = torch.autograd.grad(
            (energy_values * torch.as_tensor(energy_seed)).sum(),
            (torch_multipoles_energy, torch_positions_energy),
        )
    finally:
        torch.set_default_dtype(previous)

    ops = _ops(np.dtype(np.float64), density_max_l, feature_max_l)
    _select_reverse_strategy(ops, reverse_strategy)
    cache = _cache(ops, positions, node_graph_index)
    feature = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, True
    )
    energy = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, True
    )
    _assert_gradient(
        feature["grad_multipoles"],
        torch_feature_grad[0].detach().numpy(),
        np.dtype(np.float64),
    )
    _assert_gradient(
        feature["grad_positions"],
        torch_feature_grad[1].detach().numpy(),
        np.dtype(np.float64),
    )
    _assert_gradient(
        energy["grad_multipoles"],
        torch_energy_grad[0].detach().numpy(),
        np.dtype(np.float64),
    )
    _assert_gradient(
        energy["grad_positions"],
        torch_energy_grad[1].detach().numpy(),
        np.dtype(np.float64),
    )
    _assert_reverse_strategy_was_used(ops, reverse_strategy)


@pytest.mark.skipif(
    os.environ.get("SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE") != "1",
    reason="set SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE=1 for pinned Torch autograd",
)
@pytest.mark.parametrize(
    "torch_device", ("cpu", "cuda"), ids=("torch-cpu", "torch-gpu")
)
@pytest.mark.parametrize(
    "reverse_strategy",
    (
        "owner",
        "pair_atomic",
        "materialized_forward",
    ),
)
def test_fp32_adjoints_are_compatible_with_pinned_torch_autograd(
    torch_device, reverse_strategy
):
    # Compatibility evidence only. The strict FP32 contract is the FP64
    # finite-displacement calculation above followed by a compact cast.
    torch = pytest.importorskip("torch")
    if torch_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("pinned Torch CUDA compatibility requires a visible GPU")
    checkout = optional_checkout("SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE")
    oracle_module = import_graph_electrostatics_realspace(checkout)
    RealSpaceFiniteDiffereneEnergy = oracle_module.RealSpaceFiniteDiffereneEnergy
    RealSpaceFiniteDifferenceElectrostaticFeatures = (
        oracle_module.RealSpaceFiniteDifferenceElectrostaticFeatures
    )

    dtype = np.dtype(np.float32)
    positions_storage = np.array(
        [[0.13, -0.27, 0.31], [0.91, 0.42, -0.68], [-0.52, 0.88, 0.73]],
        dtype=np.float32,
    )
    positions = positions_storage.astype(np.float64)
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = _multipoles(dtype, 1, charged=True)
    feature_seed = _feature_seed(dtype, 3, 8)
    energy_seed = np.array([-0.67], dtype=dtype)
    torch_batch = torch.as_tensor(
        node_graph_index, dtype=torch.long, device=torch_device
    )
    torch_multipoles = torch.tensor(multipoles, device=torch_device, requires_grad=True)
    torch_positions = torch.tensor(
        positions_storage, device=torch_device, requires_grad=True
    )
    feature_oracle = RealSpaceFiniteDifferenceElectrostaticFeatures(
        1, 0.73, 1, [0.51, 1.07], True
    ).to(device=torch_device, dtype=torch.float32)
    feature_values = feature_oracle(torch_multipoles, torch_positions, torch_batch)[0]
    torch_feature_grad = torch.autograd.grad(
        (feature_values * torch.as_tensor(feature_seed, device=torch_device)).sum(),
        (torch_multipoles, torch_positions),
    )
    torch_multipoles_energy = torch.tensor(
        multipoles, device=torch_device, requires_grad=True
    )
    torch_positions_energy = torch.tensor(
        positions_storage, device=torch_device, requires_grad=True
    )
    energy_oracle = RealSpaceFiniteDiffereneEnergy(1, 0.73, True).to(
        device=torch_device, dtype=torch.float32
    )
    energy_values = energy_oracle(
        torch_multipoles_energy, torch_positions_energy, torch_batch
    )
    torch_energy_grad = torch.autograd.grad(
        (energy_values * torch.as_tensor(energy_seed, device=torch_device)).sum(),
        (torch_multipoles_energy, torch_positions_energy),
    )

    ops = _ops(dtype, 1, 1)
    _select_reverse_strategy(ops, reverse_strategy)
    cache = _cache(ops, positions, node_graph_index)
    feature = ops.diagnostic_adjoint_realspace_features(
        cache, multipoles, feature_seed, True
    )
    energy = ops.diagnostic_adjoint_realspace_energy(
        cache, multipoles, energy_seed, True
    )
    for native, torch_value in (
        (feature["grad_multipoles"], torch_feature_grad[0]),
        (feature["grad_positions"], torch_feature_grad[1]),
        (energy["grad_multipoles"], torch_energy_grad[0]),
        (energy["grad_positions"], torch_energy_grad[1]),
    ):
        np.testing.assert_allclose(
            native,
            torch_value.detach().cpu().numpy(),
            rtol=3.0e-3,
            atol=3.0e-3,
        )
    _assert_reverse_strategy_was_used(ops, reverse_strategy)
