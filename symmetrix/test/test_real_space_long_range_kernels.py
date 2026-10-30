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

# Captured from the pinned graph-electrostatics e0eb8945 implementation for
# sigma_density=0.73 and receiver widths [0.51, 1.07]. These literals keep the
# self branches independent of Symmetrix's existing self-term implementation.
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
    ]
)
ENERGY_SELF_L1 = np.array(
    [
        11.128945089057039,
        3.480623346892647,
        3.480623346892647,
        3.480623346892647,
    ]
)

FP32_L1_ORDINARY_CASES = (
    (
        [
            [0.21374138, -0.22457826, 1.0887185],
            [0.17833020, -0.91063792, 0.6147116],
        ],
        [
            [0.39120001, 0.28412428, -0.21112058, -0.37962645],
            [-0.18698233, 0.01239779, -0.69750923, -0.06563750],
        ],
        [0.33, 0.79, 1.21],
    ),
    (
        [[-0.42, 0.17, 0.63], [0.84, -0.51, 0.29], [1.37, 0.76, -0.88]],
        [
            [0.52, -0.14, 0.08, 0.21],
            [-0.31, 0.19, -0.27, 0.06],
            [-0.17, -0.05, 0.11, -0.24],
        ],
        [0.41, 0.96],
    ),
    (
        [
            [-1.15, 0.22, 0.48],
            [-0.24, -0.83, 0.91],
            [0.66, 0.37, -0.54],
            [1.31, -0.18, 0.12],
        ],
        [
            [0.27, 0.32, -0.09, 0.18],
            [-0.45, -0.12, 0.23, -0.07],
            [0.38, 0.04, -0.31, 0.15],
            [-0.20, -0.28, 0.17, -0.22],
        ],
        [0.28, 0.67, 1.34],
    ),
    (
        [
            [-0.91, -0.44, 0.36],
            [-0.33, 0.72, -0.61],
            [0.19, -0.16, 1.08],
            [0.77, 0.59, 0.14],
            [1.42, -0.73, -0.29],
        ],
        [
            [0.41, -0.17, 0.29, -0.08],
            [-0.26, 0.22, -0.13, 0.31],
            [0.18, -0.35, 0.07, 0.12],
            [-0.37, 0.16, -0.24, -0.19],
            [0.09, 0.11, 0.18, -0.27],
        ],
        [0.37, 0.74, 1.09],
    ),
)
FP32_L1_ORDINARY_CASE_IDS = (
    "review-two-atom",
    "skew-three-atom",
    "four-atom",
    "five-atom",
)

# Compatibility only, not the numerical contract. On the four ordinary cases
# above the measured feature/energy maxima were 7.54326582e-4/8.43286514e-4
# for pinned Torch CPU (oneMKL VML HA). Across 100 pinned Torch CUDA 13.0
# repetitions per case on sm120, the maxima were 7.54326582e-4/1.026630402e-3;
# the CUDA energy spread comes from Torch's nondeterministic scatter reduction.
# Rounded-up absolute envelopes retain a visible guard without pretending those
# backend-specific erf and reduction paths are the canonical mixed-precision
# result.
TORCH_FP32_COMPATIBILITY_ATOL = {
    "cpu": {"features": 8.0e-4, "energy": 9.0e-4},
    "cuda": {"features": 8.0e-4, "energy": 1.2e-3},
}


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


def _stage(value, dtype: np.dtype):
    return np.dtype(dtype).type(value)


def _replica_charges(multipole: np.ndarray, max_l: int, delta: float) -> np.ndarray:
    dtype = multipole.dtype
    if max_l == 0:
        return np.array([multipole[0]], dtype=dtype)
    displacement = _stage(delta, dtype)
    qx = _stage(multipole[3] / displacement, dtype)
    qy = _stage(multipole[1] / displacement, dtype)
    qz = _stage(multipole[2] / displacement, dtype)
    displaced = _stage(_stage(qx + qy, dtype) + qz, dtype)
    return np.array([_stage(multipole[0] - displaced, dtype), qx, qy, qz], dtype=dtype)


def _replica_offsets(max_l: int, delta: float, dtype: np.dtype) -> np.ndarray:
    if max_l == 0:
        return np.zeros((1, 3), dtype=dtype)
    return np.array(
        [[0.0, 0.0, 0.0], [delta, 0.0, 0.0], [0.0, delta, 0.0], [0.0, 0.0, delta]],
        dtype=dtype,
    )


def _kernel(distance, width, dtype: np.dtype):
    half_distance = _stage(_stage(0.5, dtype) * distance, dtype)
    scaled_distance = _stage(half_distance / width, dtype)
    numerator = _stage(math.erf(float(scaled_distance)), dtype)
    denominator = _stage(distance + _stage(KERNEL_EPSILON, dtype), dtype)
    return _stage(numerator / denominator, dtype)


def _literal_feature_self(
    multipoles: np.ndarray, density_max_l: int, feature_max_l: int
) -> np.ndarray:
    widths = 2
    dtype = multipoles.dtype
    result = np.zeros(
        (multipoles.shape[0], widths * (feature_max_l + 1) ** 2), dtype=dtype
    )
    constants = FEATURE_SELF_L1.astype(dtype)
    result[:, :widths] = multipoles[:, :1] * constants[:widths]
    if density_max_l == 1 and feature_max_l == 1:
        result[:, widths:] = np.concatenate(
            [
                multipoles[:, 1:4] * constants[2:5],
                multipoles[:, 1:4] * constants[5:8],
            ],
            axis=1,
        )
    return result


def _literal_energy_self(multipoles: np.ndarray, density_max_l: int) -> float:
    dtype = multipoles.dtype
    width = (density_max_l + 1) ** 2
    constants = ENERGY_SELF_L1[:width].astype(dtype)
    total = _stage(0.0, dtype)
    for row in multipoles:
        node = _stage(0.0, dtype)
        for channel in range(width):
            square = _stage(row[channel] * row[channel], dtype)
            node = _stage(node + _stage(constants[channel] * square, dtype), dtype)
        total = _stage(total + _stage(_stage(0.5, dtype) * node, dtype), dtype)
    return total


def _oracle_features(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    feature_max_l: int,
    density_width: float,
    feature_widths: tuple[float, ...],
) -> np.ndarray:
    dtype = positions.dtype
    n_nodes = positions.shape[0]
    n_sigma = len(feature_widths)
    output = np.zeros((n_nodes, n_sigma * (feature_max_l + 1) ** 2), dtype=dtype)
    source_offsets = _replica_offsets(density_max_l, FEATURE_DISPLACEMENT, dtype)
    receiver_offsets = _replica_offsets(feature_max_l, FEATURE_DISPLACEMENT, dtype)
    source_positions = positions[:, None, :] + source_offsets[None, :, :]
    receiver_positions = positions[:, None, :] + receiver_offsets[None, :, :]
    effective_widths = np.empty(n_sigma, dtype=dtype)
    density_squared = _stage(density_width**2, dtype)
    for channel, sigma in enumerate(feature_widths):
        sigma_scalar = _stage(sigma, dtype)
        sigma_squared = _stage(sigma_scalar * sigma_scalar, dtype)
        mean_squared = _stage(
            _stage(density_squared + sigma_squared, dtype) / _stage(2.0, dtype),
            dtype,
        )
        effective_widths[channel] = _stage(math.sqrt(float(mean_squared)), dtype)
    a0 = np.asarray(
        [
            _cl_sigma(0, sigma, receiver=True) / _cl_sigma(0, sigma, receiver=False)
            for sigma in feature_widths
        ],
        dtype=dtype,
    )
    a1 = np.asarray(
        [
            math.sqrt(3.0)
            * sigma**2
            * _cl_sigma(1, sigma, receiver=True)
            / _cl_sigma(0, sigma, receiver=False)
            / FEATURE_DISPLACEMENT
            for sigma in feature_widths
        ],
        dtype=dtype,
    )
    field_constant = _stage(FIELD_CONSTANT, dtype)
    four_pi = _stage(4.0 * math.pi, dtype)
    for receiver in range(n_nodes):
        samples = np.zeros((receiver_offsets.shape[0], n_sigma), dtype=dtype)
        for sender in range(n_nodes):
            if (
                sender == receiver
                or node_graph_index[sender] != node_graph_index[receiver]
            ):
                continue
            charges = _replica_charges(
                multipoles[sender], density_max_l, FEATURE_DISPLACEMENT
            )
            for source_replica, charge in enumerate(charges):
                for receiver_replica in range(receiver_offsets.shape[0]):
                    difference = (
                        source_positions[sender, source_replica]
                        - receiver_positions[receiver, receiver_replica]
                    )
                    # An independent NumPy contraction avoids restating the
                    # native scalar loop while retaining dtype arithmetic.
                    squared_distance = np.einsum("i,i->", difference, difference)
                    distance = _stage(np.sqrt(squared_distance, dtype=dtype), dtype)
                    for channel, width in enumerate(effective_widths):
                        contribution = _stage(
                            charge * _kernel(distance, width, dtype), dtype
                        )
                        samples[receiver_replica, channel] = _stage(
                            samples[receiver_replica, channel] + contribution,
                            dtype,
                        )
        potentials = np.empty_like(samples)
        for receiver_replica in range(samples.shape[0]):
            for channel in range(n_sigma):
                potentials[receiver_replica, channel] = _stage(
                    _stage(field_constant * samples[receiver_replica, channel], dtype)
                    / four_pi,
                    dtype,
                )
        output[receiver, :n_sigma] = a0 * potentials[0]
        if feature_max_l == 1:
            # Replica storage [0,x,y,z] crosses into e3nn [y,z,x].
            for channel in range(n_sigma):
                base = n_sigma + 3 * channel
                for output_component, replica in enumerate((2, 3, 1)):
                    difference = _stage(
                        potentials[replica, channel] - potentials[0, channel], dtype
                    )
                    output[receiver, base + output_component] = _stage(
                        a1[channel] * difference, dtype
                    )
    return output


def _oracle_energy(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    density_width: float,
    n_graphs: int,
) -> np.ndarray:
    dtype = positions.dtype
    result = np.zeros(n_graphs, dtype=dtype)
    offsets = _replica_offsets(density_max_l, ENERGY_DISPLACEMENT, dtype)
    extended_positions = positions[:, None, :] + offsets[None, :, :]
    charges = [
        _replica_charges(row, density_max_l, ENERGY_DISPLACEMENT) for row in multipoles
    ]
    field_constant = _stage(FIELD_CONSTANT, dtype)
    four_pi = _stage(4.0 * math.pi, dtype)
    half = _stage(0.5, dtype)
    width = _stage(density_width, dtype)
    for receiver in range(positions.shape[0]):
        graph = int(node_graph_index[receiver])
        for receiver_replica, receiver_charge in enumerate(charges[receiver]):
            node_energy = _stage(0.0, dtype)
            for sender in range(positions.shape[0]):
                if sender == receiver or int(node_graph_index[sender]) != graph:
                    continue
                for source_replica, source_charge in enumerate(charges[sender]):
                    difference = (
                        extended_positions[sender, source_replica]
                        - extended_positions[receiver, receiver_replica]
                    )
                    squared_distance = np.einsum("i,i->", difference, difference)
                    distance = _stage(np.sqrt(squared_distance, dtype=dtype), dtype)
                    edge = _stage(half * field_constant, dtype)
                    edge = _stage(edge * _kernel(distance, width, dtype), dtype)
                    edge = _stage(edge * source_charge, dtype)
                    edge = _stage(edge * receiver_charge, dtype)
                    edge = _stage(edge / four_pi, dtype)
                    node_energy = _stage(node_energy + edge, dtype)
            result[graph] = _stage(
                result[graph] + node_energy,
                dtype,
            )
    return result


def _mixed_precision_oracle_features(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    feature_max_l: int,
    density_width: float,
    feature_widths: tuple[float, ...],
) -> np.ndarray:
    """Evaluate the exact stencil in FP64, then cast the compact result once."""
    assert positions.dtype == np.float64
    assert multipoles.dtype == np.float32
    return _oracle_features(
        positions,
        node_graph_index,
        multipoles.astype(np.float64),
        density_max_l,
        feature_max_l,
        density_width,
        feature_widths,
    ).astype(np.float32)


def _mixed_precision_oracle_energy(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    density_width: float,
    n_graphs: int,
) -> np.ndarray:
    """Evaluate the exact stencil in FP64, then cast the graph result once."""
    assert positions.dtype == np.float64
    assert multipoles.dtype == np.float32
    return _oracle_energy(
        positions,
        node_graph_index,
        multipoles.astype(np.float64),
        density_max_l,
        density_width,
        n_graphs,
    ).astype(np.float32)


def _canonical_oracle_features(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    feature_max_l: int,
    density_width: float,
    feature_widths: tuple[float, ...],
    *,
    include_self_interaction: bool = False,
) -> np.ndarray:
    compute_positions = positions
    compute_multipoles = multipoles
    output_dtype = multipoles.dtype
    if output_dtype == np.float32:
        compute_positions = positions.astype(np.float64, copy=False)
        compute_multipoles = multipoles.astype(np.float64)
    result = _oracle_features(
        compute_positions,
        node_graph_index,
        compute_multipoles,
        density_max_l,
        feature_max_l,
        density_width,
        feature_widths,
    )
    if include_self_interaction:
        result += _literal_feature_self(
            compute_multipoles, density_max_l, feature_max_l
        )
    return result.astype(output_dtype, copy=False)


def _canonical_oracle_energy(
    positions: np.ndarray,
    node_graph_index: np.ndarray,
    multipoles: np.ndarray,
    density_max_l: int,
    density_width: float,
    n_graphs: int,
    *,
    include_self_interaction: bool = False,
) -> np.ndarray:
    compute_positions = positions
    compute_multipoles = multipoles
    output_dtype = multipoles.dtype
    if output_dtype == np.float32:
        compute_positions = positions.astype(np.float64, copy=False)
        compute_multipoles = multipoles.astype(np.float64)
    result = _oracle_energy(
        compute_positions,
        node_graph_index,
        compute_multipoles,
        density_max_l,
        density_width,
        n_graphs,
    )
    if include_self_interaction:
        result += _literal_energy_self(compute_multipoles, density_max_l)
    return result.astype(output_dtype, copy=False)


def _ops(dtype: np.dtype, density_max_l: int, feature_max_l: int, widths=(0.51, 1.07)):
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
        feature_smearing_widths=list(widths),
    )


def _cache(ops, positions: np.ndarray, node_graph_index: np.ndarray, cell=None):
    n_graphs = int(node_graph_index.max()) + 1
    if cell is None:
        cell = np.repeat(np.eye(3)[None, :, :] * 8.0, n_graphs, axis=0)
    return ops.realspace_geometry_cache(positions, cell, node_graph_index)


DTYPES = (np.dtype(np.float64), np.dtype(np.float32))


def _tolerances(dtype: np.dtype) -> dict[str, float]:
    if dtype == np.dtype(np.float64):
        return {"rtol": 1.0e-11, "atol": 1.0e-11}
    return {"rtol": 2.0e-5, "atol": 2.0e-6}


@pytest.mark.parametrize("dtype", DTYPES)
def test_two_monopoles_pin_constant_half_factor_epsilon_and_effective_width(dtype):
    # Catches changing C, omitting the directed-pair 1/2, moving epsilon under
    # the norm, or dropping the /2 inside the effective feature width.
    positions = np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]], dtype=dtype)
    batch = np.zeros(2, dtype=np.int32)
    multipoles = np.array([[1.2], [-0.4]], dtype=dtype)
    sigma_density = 0.73
    sigma_feature = 0.91
    ops = _ops(dtype, 0, 0, (sigma_feature,))
    cache = _cache(ops, positions, batch)

    native_energy = ops.realspace_energy(cache, multipoles, False)
    native_features = ops.realspace_features(cache, multipoles, False)

    distance = 1.25
    expected_energy = np.array(
        [
            COULOMB_CONSTANT
            * multipoles[0, 0]
            * multipoles[1, 0]
            * math.erf(distance / (2.0 * sigma_density))
            / (distance + KERNEL_EPSILON)
        ]
    )
    effective = math.sqrt((sigma_density**2 + sigma_feature**2) / 2.0)
    a0 = _cl_sigma(0, sigma_feature, receiver=True) / _cl_sigma(
        0, sigma_feature, receiver=False
    )
    radial = math.erf(distance / (2.0 * effective)) / (distance + KERNEL_EPSILON)
    expected_features = np.array(
        [
            [a0 * COULOMB_CONSTANT * multipoles[1, 0] * radial],
            [a0 * COULOMB_CONSTANT * multipoles[0, 0] * radial],
        ]
    )
    np.testing.assert_allclose(native_energy, expected_energy, **_tolerances(dtype))
    np.testing.assert_allclose(native_features, expected_features, **_tolerances(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "density_max_l,feature_max_l", [(0, 0), (1, 0), (0, 1), (1, 1)]
)
@pytest.mark.parametrize("charged", [False, True])
def test_forward_matches_independent_literal_oracle_for_all_l_combinations_and_spins(
    dtype, density_max_l, feature_max_l, charged
):
    # Catches the wrong finite-displacement stencil, l cross-case shortcut,
    # width/channel layout, or accidental sharing between alpha/beta calls.
    positions = np.array(
        [[-0.2, 0.1, 0.4], [0.7, -0.3, 0.2], [1.4, 0.8, -0.5]], dtype=dtype
    )
    batch = np.zeros(3, dtype=np.int32)
    width = (density_max_l + 1) ** 2
    alpha = np.array(
        [
            [0.6, 0.07, -0.11, 0.03],
            [-0.2, -0.04, 0.02, 0.09],
            [-0.4, 0.06, 0.05, -0.08],
        ],
        dtype=dtype,
    )[:, :width]
    if charged:
        alpha[2, 0] += dtype.type(0.35)
    beta = np.array(
        [[-0.3, 0.02, 0.03, -0.01], [0.5, -0.08, 0.01, 0.04], [0.1, 0.06, -0.02, 0.05]],
        dtype=dtype,
    )[:, :width]
    ops = _ops(dtype, density_max_l, feature_max_l)
    cache = _cache(ops, positions, batch)

    for source in (alpha, beta):
        expected_features = _canonical_oracle_features(
            positions, batch, source, density_max_l, feature_max_l, 0.73, (0.51, 1.07)
        )
        expected_energy = _canonical_oracle_energy(
            positions, batch, source, density_max_l, 0.73, 1
        )
        np.testing.assert_allclose(
            ops.realspace_features(cache, source, False),
            expected_features,
            **_tolerances(dtype),
        )
        np.testing.assert_allclose(
            ops.realspace_energy(cache, source, False),
            expected_energy,
            **_tolerances(dtype),
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "field_self,energy_self",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_feature_and_energy_self_flags_are_independent(dtype, field_self, energy_self):
    # Catches reciprocal-style subtract-on-disabled behavior and coupling the
    # two independently serialized self-interaction flags.
    positions = np.array([[0.1, -0.2, 0.3], [0.8, 0.4, -0.5]], dtype=dtype)
    batch = np.zeros(2, dtype=np.int32)
    multipoles = np.array(
        [[0.7, 0.04, -0.06, 0.08], [-0.2, -0.03, 0.05, 0.01]], dtype=dtype
    )
    ops = _ops(dtype, 1, 1)
    cache = _cache(ops, positions, batch)
    expected_features = _canonical_oracle_features(
        positions,
        batch,
        multipoles,
        1,
        1,
        0.73,
        (0.51, 1.07),
        include_self_interaction=field_self,
    )
    expected_energy = _canonical_oracle_energy(
        positions,
        batch,
        multipoles,
        1,
        0.73,
        1,
        include_self_interaction=energy_self,
    )
    np.testing.assert_allclose(
        ops.realspace_features(cache, multipoles, field_self),
        expected_features,
        **_tolerances(dtype),
    )
    np.testing.assert_allclose(
        ops.realspace_energy(cache, multipoles, energy_self),
        expected_energy,
        **_tolerances(dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "positions",
    [
        np.array([[0.2, -0.1, 0.4]]),
        np.array([[0.0, 0.0, 0.0], [1.0e-9, -2.0e-9, 3.0e-9]]),
        np.array([[0.3, -0.2, 0.1], [0.3, -0.2, 0.1]]),
    ],
    ids=("isolated", "near", "coincident-distinct"),
)
def test_isolated_near_and_coincident_distinct_atoms_remain_finite(dtype, positions):
    # Catches excluding coincident distinct atoms or admitting any same-original
    # displaced replicas into the isolated-atom result.
    positions = positions.astype(dtype)
    batch = np.zeros(positions.shape[0], dtype=np.int32)
    multipoles = np.array(
        [[0.5, 0.03, -0.02, 0.01], [-0.4, -0.01, 0.04, 0.02]], dtype=dtype
    )[: positions.shape[0]]
    ops = _ops(dtype, 1, 1)
    cache = _cache(ops, positions, batch)
    features = ops.realspace_features(cache, multipoles, False)
    energy = ops.realspace_energy(cache, multipoles, False)
    assert np.isfinite(features).all()
    assert np.isfinite(energy).all()
    np.testing.assert_allclose(
        features,
        _canonical_oracle_features(
            positions, batch, multipoles, 1, 1, 0.73, (0.51, 1.07)
        ),
        **_tolerances(dtype),
    )
    np.testing.assert_allclose(
        energy,
        _canonical_oracle_energy(positions, batch, multipoles, 1, 0.73, 1),
        **_tolerances(dtype),
    )
    if positions.shape[0] == 1:
        np.testing.assert_array_equal(features, np.zeros_like(features))
        np.testing.assert_array_equal(energy, np.zeros_like(energy))


@pytest.mark.parametrize("dtype", DTYPES)
def test_translation_and_fixed_cartesian_cell_changes_do_not_change_forward(dtype):
    # Catches a hidden MIC, volume, phase, or reciprocal-cell dependency.
    # Binary-exact coordinates keep this an electrostatic translation test,
    # rather than a test of significance lost while constructing FP32 input.
    positions = np.array([[0.125, 0.25, -0.375], [0.875, -0.5, 0.75]], dtype=dtype)
    translated = positions + np.array([4.0, -2.0, 1.0], dtype=dtype)
    multipoles = np.array(
        [[0.6, 0.03, 0.02, -0.04], [-0.1, -0.05, 0.01, 0.06]], dtype=dtype
    )
    batch = np.zeros(2, dtype=np.int32)
    cell_a = np.array([[[6.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 8.0]]])
    cell_b = np.array([[[20.0, 1.0, -2.0], [0.0, 5.0, 0.5], [0.0, 0.0, 3.0]]])
    ops = _ops(dtype, 1, 1)
    cache_a = _cache(ops, positions, batch, cell_a)
    cache_b = _cache(ops, positions, batch, cell_b)
    cache_translated = _cache(ops, translated, batch, cell_b)
    for method in (ops.realspace_features, ops.realspace_energy):
        first = method(cache_a, multipoles, False)
        np.testing.assert_allclose(
            method(cache_b, multipoles, False), first, **_tolerances(dtype)
        )
        np.testing.assert_allclose(
            method(cache_translated, multipoles, False), first, **_tolerances(dtype)
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_realspace_cache_has_zero_reciprocal_tables_and_linear_retention(dtype):
    # Catches allocating any k/phase/Fourier table or retaining an N-by-N pair
    # graph behind the compact cache handle.
    ops = _ops(dtype, 1, 1)
    bytes_by_size = []
    for n_nodes in (3, 11, 37):
        positions = np.arange(n_nodes * 3, dtype=dtype).reshape(n_nodes, 3) / 13.0
        cache = _cache(ops, positions, np.zeros(n_nodes, dtype=np.int32))
        assert cache.mode == "realspace"
        assert cache.n_k == 0
        assert cache.reciprocal_allocation_elements == 0
        assert cache.reciprocal_allocation_bytes == 0
        assert cache.realspace_retained_elements == 9 * n_nodes
        assert cache.realspace_retained_bytes == 60 * n_nodes
        bytes_by_size.append(cache.realspace_retained_bytes)
    assert (
        bytes_by_size[2] - bytes_by_size[1]
        == 26 * (bytes_by_size[1] - bytes_by_size[0]) // 8
    )


@pytest.mark.parametrize(
    "positions,multipoles,widths",
    FP32_L1_ORDINARY_CASES,
    ids=FP32_L1_ORDINARY_CASE_IDS,
)
def test_fp32_l1_forward_matches_canonical_mixed_precision_oracle(
    positions, multipoles, widths
):
    # FP32 storage is an interface choice: all finite-displacement arithmetic
    # is canonical FP64, and only these compact public outputs round to FP32.
    dtype = np.dtype(np.float32)
    positions = np.asarray(positions, dtype=np.float64)
    multipoles = np.asarray(multipoles, dtype=dtype)
    node_graph_index = np.zeros(positions.shape[0], dtype=np.int32)
    ops = _ops(dtype, 1, 1, tuple(widths))
    cache = _cache(ops, positions, node_graph_index)

    np.testing.assert_allclose(
        ops.realspace_features(cache, multipoles, False),
        _mixed_precision_oracle_features(
            positions,
            node_graph_index,
            multipoles,
            1,
            1,
            0.73,
            tuple(widths),
        ),
        **_tolerances(dtype),
    )
    np.testing.assert_allclose(
        ops.realspace_energy(cache, multipoles, False),
        _mixed_precision_oracle_energy(
            positions, node_graph_index, multipoles, 1, 0.73, 1
        ),
        **_tolerances(dtype),
    )


def test_fp32_l1_two_atom_matches_captured_mixed_precision_regression():
    # Catches an intermediate FP32 cast anywhere in the stencil. Expected
    # values were captured independently from the exact equations evaluated
    # in FP64 from the listed FP32 inputs, with one final output cast.
    # The public coordinate boundary is FP64. These values deliberately retain
    # the prior captured fixture's exactly represented FP32 coordinates.
    positions = np.array(
        [
            [0.21374138, -0.22457826, 1.0887185],
            [0.17833020, -0.91063792, 0.6147116],
        ],
        dtype=np.float32,
    ).astype(np.float64)
    multipoles = np.array(
        [
            [0.39120001, 0.28412428, -0.21112058, -0.37962645],
            [-0.18698233, 0.01239779, -0.69750923, -0.06563750],
        ],
        dtype=np.float32,
    )
    expected_features = np.array(
        [
            [
                -13.663145,
                -9.077825,
                -6.3435144,
                2.259789,
                -3.2271814,
                -0.19036485,
                1.9753717,
                -4.1046405,
                -0.29096314,
                1.1982151,
                -3.1618474,
                -0.24114592,
            ],
            [
                15.611118,
                12.868577,
                10.371443,
                3.4738595,
                -0.44476932,
                -2.6417227,
                4.039314,
                -0.44298115,
                -3.0071008,
                2.9772692,
                -0.2962233,
                -2.1901782,
            ],
        ],
        dtype=np.float32,
    )
    expected_energy = np.array([-0.505615], dtype=np.float32)
    node_graph_index = np.zeros(2, dtype=np.int32)
    ops = _ops(np.dtype(np.float32), 1, 1, (0.33, 0.79, 1.21))
    cache = _cache(ops, positions, node_graph_index)

    np.testing.assert_allclose(
        ops.realspace_features(cache, multipoles, False),
        expected_features,
        **_tolerances(np.dtype(np.float32)),
    )
    np.testing.assert_allclose(
        ops.realspace_energy(cache, multipoles, False),
        expected_energy,
        **_tolerances(np.dtype(np.float32)),
    )


def test_fp32_realspace_preserves_sub_float32_near_pair_coordinates():
    # Both x coordinates round to exactly 1.0 in FP32. Real-space geometry must
    # retain the public FP64 positions, otherwise this distinct near pair is
    # spuriously coincident and both compact outputs collapse to zero.
    delta = 2.0**-25
    positions = np.array(
        [[1.0, -0.25, 0.5], [1.0 + delta, -0.25, 0.5]], dtype=np.float64
    )
    assert np.float32(positions[0, 0]) == np.float32(positions[1, 0])
    multipoles = np.array([[0.75], [-0.4]], dtype=np.float32)
    node_graph_index = np.zeros(2, dtype=np.int32)
    ops = _ops(np.dtype(np.float32), 0, 0, (0.51,))
    cache = _cache(ops, positions, node_graph_index)

    expected_features = _mixed_precision_oracle_features(
        positions, node_graph_index, multipoles, 0, 0, 0.73, (0.51,)
    )
    expected_energy = _mixed_precision_oracle_energy(
        positions, node_graph_index, multipoles, 0, 0.73, 1
    )
    assert np.any(expected_features != 0.0)
    assert np.any(expected_energy != 0.0)
    np.testing.assert_allclose(
        ops.realspace_features(cache, multipoles, False),
        expected_features,
        **_tolerances(np.dtype(np.float32)),
    )
    np.testing.assert_allclose(
        ops.realspace_energy(cache, multipoles, False),
        expected_energy,
        **_tolerances(np.dtype(np.float32)),
    )
    assert cache.realspace_retained_elements == 18
    assert cache.realspace_retained_bytes == 120


@pytest.mark.parametrize("dtype", DTYPES)
def test_realspace_energy_launches_one_parallel_partial_per_receiver_replica(dtype):
    # Catches collapsing production energy back to one graph work item. The
    # diagnostic is populated by the production launch, not a reference path.
    n_nodes = 7
    positions = np.arange(n_nodes * 3, dtype=dtype).reshape(n_nodes, 3) / dtype.type(
        17.0
    )
    multipoles = np.linspace(-0.3, 0.4, n_nodes * 4, dtype=dtype).reshape(n_nodes, 4)
    node_graph_index = np.zeros(n_nodes, dtype=np.int32)
    ops = _ops(dtype, 1, 1)
    cache = _cache(ops, positions, node_graph_index)

    ops.realspace_energy(cache, multipoles, False, use_workspace=True)
    diagnostics = ops.realspace_energy_launch_diagnostics()

    assert diagnostics == {
        "selected_count": 1,
        "receiver_replica_league_extent": 4 * n_nodes,
        "graph_reduction_league_extent": 1,
        "partial_elements": 4 * n_nodes,
        "full_materialized_serial_selected_count": 0,
        "full_materialized_parallel_selected_count": 0,
        "full_materialized_parallel_edge_extent": 0,
    }
    expected_workspace_bytes = (
        multipoles.size * dtype.itemsize
        + dtype.itemsize
        + 4 * n_nodes * np.dtype(np.float64).itemsize
    )
    assert ops.workspace_allocation_bytes == expected_workspace_bytes


@pytest.mark.parametrize("dtype", DTYPES)
def test_team_scale_forward_matches_canonical_mixed_precision_oracle(dtype):
    # Eight atoms is the CUDA sender-team dispatch boundary. Keep a case at
    # that exact size so the parallel feature and energy reductions remain
    # covered by the same numerical contract as the serial small-system path.
    n_nodes = 8
    positions = np.arange(n_nodes * 3, dtype=np.float64).reshape(n_nodes, 3) / 7.0
    multipoles = np.linspace(-0.31, 0.43, n_nodes * 4).reshape(n_nodes, 4)
    positions = positions.astype(dtype)
    multipoles = multipoles.astype(dtype)
    node_graph_index = np.zeros(n_nodes, dtype=np.int32)
    ops = _ops(dtype, 1, 1)
    cache = _cache(ops, positions, node_graph_index)

    np.testing.assert_allclose(
        ops.realspace_features(cache, multipoles, False),
        _canonical_oracle_features(
            positions,
            node_graph_index,
            multipoles,
            1,
            1,
            0.73,
            (0.51, 1.07),
        ),
        **_tolerances(dtype),
    )
    np.testing.assert_allclose(
        ops.realspace_energy(cache, multipoles, False),
        _canonical_oracle_energy(positions, node_graph_index, multipoles, 1, 0.73, 1),
        **_tolerances(dtype),
    )


def test_same_owner_direct_to_real_transition_releases_cache_and_workspace():
    # Catches mode-gated accounting, failure to clear reciprocal cache views,
    # and failure to clear the operator-owned reciprocal scratch arena.
    positions = np.array(
        [[0.1, -0.2, 0.3], [0.8, 0.4, -0.5], [1.2, -0.7, 0.6]],
        dtype=np.float64,
    )
    cell = np.array([np.eye(3) * 8.0])
    node_graph_index = np.zeros(3, dtype=np.int32)
    multipoles = np.array(
        [
            [0.4, 0.03, -0.02, 0.01],
            [-0.3, -0.01, 0.04, 0.02],
            [-0.1, 0.02, -0.02, 0.03],
        ]
    )
    ops = _ops(np.dtype(np.float64), 1, 1)
    cache = ops.geometry_cache(positions, cell, node_graph_index)["cache"]
    ops.coulomb_features(cache, multipoles, use_workspace=True)
    assert cache.reciprocal_allocation_bytes > 0
    assert ops.workspace_allocation_bytes > 0

    ops.realspace_geometry_cache_into(cache, positions, cell, node_graph_index)

    assert cache.mode == "realspace"
    assert cache.reciprocal_allocation_elements == 0
    assert cache.reciprocal_allocation_bytes == 0
    assert ops.workspace_allocation_elements == 0
    assert ops.workspace_allocation_bytes == 0
    assert cache.realspace_retained_bytes > 0


@pytest.mark.skipif(
    os.environ.get("SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE") != "1",
    reason="set SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE=1 for the pinned Torch oracle",
)
@pytest.mark.parametrize(
    "density_max_l,feature_max_l", [(0, 0), (1, 0), (0, 1), (1, 1)]
)
def test_fp64_forward_matches_pinned_torch_cpu_oracle(density_max_l, feature_max_l):
    # FP64 has no mixed-storage boundary, so pinned upstream remains a strict
    # external oracle at the ordinary stage tolerance.
    torch = pytest.importorskip("torch")
    checkout = optional_checkout("SYMMETRIX_POLAR_ORACLE_GRAPH_LONGRANGE")
    oracle_module = import_graph_electrostatics_realspace(checkout)
    RealSpaceFiniteDiffereneEnergy = oracle_module.RealSpaceFiniteDiffereneEnergy
    RealSpaceFiniteDifferenceElectrostaticFeatures = (
        oracle_module.RealSpaceFiniteDifferenceElectrostaticFeatures
    )

    dtype = np.dtype(np.float64)
    torch_dtype = torch.float64
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        positions = np.array(
            [[0.13, -0.27, 0.31], [0.91, 0.42, -0.68], [-0.52, 0.88, 0.73]],
            dtype=dtype,
        )
        width = (density_max_l + 1) ** 2
        source = np.array(
            [
                [0.5, 0.02, -0.03, 0.04],
                [-0.1, -0.05, 0.01, 0.03],
                [0.2, 0.04, 0.06, -0.02],
            ],
            dtype=dtype,
        )[:, :width]
        batch = np.zeros(3, dtype=np.int32)
        t_positions = torch.as_tensor(positions, dtype=torch_dtype)
        t_source = torch.as_tensor(source, dtype=torch_dtype)
        t_batch = torch.as_tensor(batch, dtype=torch.long)
        feature_oracle = RealSpaceFiniteDifferenceElectrostaticFeatures(
            density_max_l, 0.73, feature_max_l, [0.51, 1.07], True
        ).to(dtype=torch_dtype)
        energy_oracle = RealSpaceFiniteDiffereneEnergy(density_max_l, 0.73, True).to(
            dtype=torch_dtype
        )
        expected_features = (
            feature_oracle(t_source, t_positions, t_batch)[0].detach().numpy()
        )
        expected_energy = energy_oracle(t_source, t_positions, t_batch).detach().numpy()
        ops = _ops(dtype, density_max_l, feature_max_l)
        cache = _cache(ops, positions, batch)
        np.testing.assert_allclose(
            ops.realspace_features(cache, source, True),
            expected_features,
            **_tolerances(dtype),
        )
        np.testing.assert_allclose(
            ops.realspace_energy(cache, source, True),
            expected_energy,
            **_tolerances(dtype),
        )
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.skipif(
    os.environ.get("SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE") != "1",
    reason="set SYMMETRIX_RUN_UPSTREAM_POLAR_ORACLE=1 for the pinned Torch oracle",
)
@pytest.mark.parametrize(
    "positions,multipoles,widths",
    FP32_L1_ORDINARY_CASES,
    ids=FP32_L1_ORDINARY_CASE_IDS,
)
@pytest.mark.parametrize("torch_device", ("cpu", "cuda"), ids=("cpu", "cuda"))
def test_fp32_l1_forward_is_within_measured_pinned_torch_compatibility_envelope(
    positions, multipoles, widths, torch_device
):
    # Torch CPU and CUDA have different FP32 erf implementations. They are
    # compatibility references with measured envelopes, not canonical oracles.
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
    positions = np.asarray(positions, dtype=dtype)
    multipoles = np.asarray(multipoles, dtype=dtype)
    node_graph_index = np.zeros(positions.shape[0], dtype=np.int32)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        t_positions = torch.as_tensor(
            positions, dtype=torch.float32, device=torch_device
        )
        t_multipoles = torch.as_tensor(
            multipoles, dtype=torch.float32, device=torch_device
        )
        t_batch = torch.as_tensor(
            node_graph_index, dtype=torch.long, device=torch_device
        )
        feature_oracle = RealSpaceFiniteDifferenceElectrostaticFeatures(
            1, 0.73, 1, widths, False
        ).to(device=torch_device, dtype=torch.float32)
        energy_oracle = RealSpaceFiniteDiffereneEnergy(1, 0.73, False).to(
            device=torch_device, dtype=torch.float32
        )
        expected_features = (
            feature_oracle(t_multipoles, t_positions, t_batch)[0].detach().cpu().numpy()
        )
        expected_energy = (
            energy_oracle(t_multipoles, t_positions, t_batch).detach().cpu().numpy()
        )
        ops = _ops(dtype, 1, 1, tuple(widths))
        cache = _cache(ops, positions, node_graph_index)
        envelope = TORCH_FP32_COMPATIBILITY_ATOL[torch_device]
        np.testing.assert_allclose(
            ops.realspace_features(cache, multipoles, False),
            expected_features,
            rtol=0.0,
            atol=envelope["features"],
        )
        np.testing.assert_allclose(
            ops.realspace_energy(cache, multipoles, False),
            expected_energy,
            rtol=0.0,
            atol=envelope["energy"],
        )
    finally:
        torch.set_default_dtype(previous)
