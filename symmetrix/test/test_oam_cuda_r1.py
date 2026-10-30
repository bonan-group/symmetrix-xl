"""Behavioral qualification of OAM CUDA forward/reverse and artifact admission."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from oam_host_oracle import (
    reference_evaluator_results as host_reference_evaluator_results,
)
from oam_host_oracle import reference_routing as host_reference_routing
from symmetrix.oam_cuda_artifact import prepare_nequip_oam_cuda_artifact

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    from symmetrix import symmetrix as native

    if not native._kokkos_is_initialized():
        native._init_kokkos()
    target = native._execution_device_execution_environment()
    if target.get("backend") != "cuda" or not target.get("available"):
        pytest.skip("an available CUDA native backend is required")
    from test_nequip_oam_native import _nonzero_contract

    cache = tmp_path_factory.mktemp("oam-cuda")
    return native, target, cache, _nonzero_contract(cache)


def artifact_for(runtime, precision, compiler):
    _native, target, cache, contract = runtime
    if compiler == "nvcc" and not shutil.which("nvcc"):
        pytest.skip("nvcc is required for the shared-library loader variant")
    # Compiler, loader and execution failures must fail qualification, not skip.
    return prepare_nequip_oam_cuda_artifact(
        contract, precision, target, compiler=compiler, cache_root=cache
    )


def graph(distance=1.3, pairs=1):
    positions = np.zeros((2 * pairs, 3))
    positions[1::2] = (distance, 0.21, -0.13)
    sources = (np.arange(2 * pairs) ^ 1).astype(np.int32)
    node_types = np.tile(np.asarray([0, 1], dtype=np.int32), pairs)
    xyz = np.ascontiguousarray(positions[sources] - positions)
    return positions, node_types, sources, xyz


def prepare(evaluator, values, counts=None):
    positions, types, sources, xyz = values
    if counts is None:
        counts = np.ones(len(types), dtype=np.int32)
    generation = evaluator.prepare_graph(
        len(types), types, counts, sources, types[sources]
    )
    evaluator.prepare_geometry(
        generation,
        positions.ravel(),
        xyz.ravel(),
        np.empty(0),
        np.empty(0),
        np.empty(0, dtype=np.int32),
    )
    return generation


def evaluate(evaluator, generation, xyz):
    evaluator.compute_prepared(generation, xyz.ravel(), np.linalg.norm(xyz, axis=1))
    evaluator.reduce_forces(generation)
    return np.asarray(evaluator.node_energies), np.asarray(
        evaluator.atom_forces
    ).reshape(-1, 3)


def evaluator_for(runtime, artifact, precision, path=None):
    native = runtime[0]
    cls = (
        native.NequIPOAMKokkosFloat
        if precision == "float32"
        else native.NequIPOAMKokkos
    )
    return cls(
        str(path or runtime[3]),
        "",
        cuda_plugin_path=str(artifact.artifact_path),
        cuda_artifact_identity={"manifest_path": str(artifact.manifest_path)},
    )


def _host_reference_available(native):
    """True when this build can run the artifact-free reference routing."""

    return native._kokkos_default_execution_space() in ("OpenMP", "Serial")


def _reference(native, path):
    """The artifact-free reference routing of the retained OAM evaluator.

    ``NequIPOAMKokkos(path)`` with no artifact argument is the portable Kokkos
    implementation, and ``direct`` is the native mode the reference routing
    hands it in the selected execution space.
    """

    evaluator = native.NequIPOAMKokkos(str(path))
    evaluator.set_streamed_edges("direct")
    return evaluator


def reference_evaluator_results(native, path, args):
    """The reference routing's ``(node_energies, node_forces)`` for one call.

    ``args`` is the positional form the OAM entry point takes. A host build
    drives the artifact-free reference evaluator in this process. Device tests
    retain the host child-process oracle here to compare generated CUDA against
    an independent backend, even though portable Kokkos also runs on CUDA.
    """

    if _host_reference_available(native):
        evaluator = _reference(native, path)
        evaluator.compute_node_energies_forces(*args)
        return (
            np.asarray(evaluator.node_energies),
            np.asarray(evaluator.node_forces),
        )
    return host_reference_evaluator_results(str(path), args)


def reference_routing(native, values, path, counts=None):
    """The reference routing as the whole-evaluator oracle.

    A host build evaluates it in this process.  A device build has no runnable
    artifact-free or receiver-host OAM implementation, so the same routing is
    evaluated on the host backend in a child process and returned here.
    """

    if not _host_reference_available(native):
        return host_reference_routing(str(path), values, counts)

    _, types, sources, xyz = values
    if counts is None:
        counts = np.ones(len(types), dtype=np.int32)
    evaluator = _reference(native, path)
    evaluator.compute_node_energies_forces(
        len(types),
        types,
        counts,
        sources,
        types[sources],
        xyz.ravel(),
        np.linalg.norm(xyz, axis=1),
    )
    directed = np.asarray(evaluator.node_forces).reshape(-1, 3)
    forces = np.zeros((len(types), 3))
    np.add.at(forces, np.repeat(np.arange(len(types)), counts), -directed)
    np.add.at(forces, sources, directed)
    return np.asarray(evaluator.node_energies), forces


@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_generated_execution_matches_oracle_and_reuses_displaced_graph(
    runtime, compiler, precision
):
    artifact = artifact_for(runtime, precision, compiler)
    evaluator = evaluator_for(runtime, artifact, precision)
    evaluator.set_streamed_edges("direct")
    values = graph()
    generation = prepare(evaluator, values)
    tolerance = 2e-4 if precision == "float32" else 1e-9
    for step, distance in enumerate((1.3, 1.55), start=1):
        values = graph(distance)
        energies, forces = evaluate(evaluator, generation, values[-1])
        expected_energy, expected_forces = reference_routing(
            runtime[0], values, runtime[3]
        )
        np.testing.assert_allclose(
            energies, expected_energy, atol=tolerance, rtol=tolerance
        )
        # Both sides evaluate the certified spline; what is left is the
        # device/host reduction order.
        force_tolerance = tolerance if precision == "float32" else 1e-7
        np.testing.assert_allclose(
            forces, expected_forces, atol=force_tolerance, rtol=force_tolerance
        )
        assert np.max(np.abs(forces)) > 1e-10
        np.testing.assert_allclose(forces.sum(axis=0), 0, atol=tolerance)
        assert evaluator.cuda_r1_ready
        assert evaluator.cuda_r1_fallback_count == 0
        assert evaluator.cuda_r1_launch_count == step * 2 * evaluator.num_layers
        assert evaluator.cuda_r1_artifact_id == artifact.artifact_id
    second = artifact_for(runtime, precision, compiler)
    assert second.cache_hit
    assert second.artifact_path == artifact.artifact_path


def test_generated_forces_match_energy_derivative(runtime):
    artifact = artifact_for(runtime, "float64", "nvrtc")
    evaluator = evaluator_for(runtime, artifact, "float64")
    evaluator.set_streamed_edges("direct")
    values = graph()
    generation = prepare(evaluator, values)
    _, forces = evaluate(evaluator, generation, values[-1])
    step = 1e-5
    for axis in range(3):
        plus = values[-1].copy()
        minus = values[-1].copy()
        plus[0, axis] += step
        plus[1, axis] -= step
        minus[0, axis] -= step
        minus[1, axis] += step
        e_plus = evaluate(evaluator, generation, plus)[0].sum()
        e_minus = evaluate(evaluator, generation, minus)[0].sum()
        np.testing.assert_allclose(
            forces[1, axis], -(e_plus - e_minus) / (2 * step), atol=1e-7, rtol=1e-6
        )


def test_large_owner_range_and_device_scaled_grid(runtime):
    artifact = artifact_for(runtime, "float64", "nvrtc")
    evaluator = evaluator_for(runtime, artifact, "float64")
    evaluator.set_streamed_edges("direct")
    # Exceed both receiver-forward and source-reverse persistent CTA caps.
    pairs = 8 * runtime[1]["multiprocessor_count"] + 1
    values = graph(pairs=pairs)
    generation = prepare(evaluator, values)
    energies, forces = evaluate(evaluator, generation, values[-1])
    energy_one, force_one = reference_routing(runtime[0], graph(), runtime[3])
    np.testing.assert_allclose(
        energies, np.tile(energy_one, pairs), atol=1e-11, rtol=1e-11
    )
    np.testing.assert_allclose(
        forces, np.tile(force_one, (pairs, 1)), atol=1e-7, rtol=1e-7
    )
    # This property reports the base/reverse cap, not forward's doubled cap.
    assert evaluator.cuda_r1_grid_blocks == 4 * runtime[1]["multiprocessor_count"]


def test_rejects_plugin_for_different_valid_oam_architecture(runtime, tmp_path):
    artifact = artifact_for(runtime, "float64", "nvrtc")
    contract = json.loads(runtime[3].read_text())
    contract["layers"][0]["edge_tensor_product"]["instructions"][0]["path_weight"] *= (
        2.0
    )
    other = tmp_path / "other-oam.json"
    other.write_text(json.dumps(contract))
    evaluator = evaluator_for(runtime, artifact, "float64", other)
    assert not evaluator.cuda_r1_ready
    with pytest.raises(ValueError, match="different architecture"):
        evaluator.set_streamed_edges("direct")


def test_accepts_plugin_for_retrained_weights_with_the_same_architecture(
    runtime, tmp_path
):
    artifact = artifact_for(runtime, "float64", "nvrtc")
    contract = json.loads(runtime[3].read_text())
    contract["per_type_shift"]["values"][0] += 0.25
    retrained = tmp_path / "retrained-oam.json"
    retrained.write_text(json.dumps(contract))

    evaluator = evaluator_for(runtime, artifact, "float64", retrained)

    evaluator.set_streamed_edges("direct")
    values = graph()
    generation = prepare(evaluator, values)
    evaluate(evaluator, generation, values[-1])
    assert evaluator.cuda_r1_ready
    assert evaluator.cuda_r1_launch_count > 0
    assert evaluator.cuda_r1_fallback_count == 0


def test_public_calculator_executes_generated_cuda(runtime, monkeypatch):
    from ase import Atoms
    from symmetrix import NequIPCalculator

    monkeypatch.setenv("SYMMETRIX_JIT_POLICY", "required")
    monkeypatch.setenv("SYMMETRIX_JIT_CACHE", str(runtime[2]))
    values = graph()
    atoms = Atoms(numbers=[1, 2], positions=values[0])
    if _host_reference_available(runtime[0]):
        atoms.calc = NequIPCalculator(runtime[3], streamed_edges="reference")
        reference_energy = atoms.get_potential_energy()
        reference_forces = atoms.get_forces()
    else:
        # The public reference routing is a host implementation, so a device
        # build takes the same routing from the host oracle process.
        node_energies, reference_forces = reference_routing(
            runtime[0], values, runtime[3]
        )
        reference_energy = float(np.sum(node_energies))
    calculator = NequIPCalculator(runtime[3], dtype="float64", streamed_edges="direct")
    atoms.calc = calculator
    np.testing.assert_allclose(
        atoms.get_potential_energy(), reference_energy, atol=1e-11
    )
    np.testing.assert_allclose(atoms.get_forces(), reference_forces, atol=1e-11)
    assert "implementation=generated-r1" in calculator.execution_diagnostic
    assert "cuda_r1_fallback_count=0" in calculator.execution_diagnostic


@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_generated_zbl_mixed_neighbors_cutoff_and_reuse(runtime, tmp_path, precision):
    # Wrong type lookup, scaled ZBL, an edge dropped by the team epilogue,
    # changed envelope power, or stale retained buffers changes these outputs.
    contract = json.loads(runtime[3].read_text())
    numbers = contract["atomic_numbers"]
    contract["atomic_numbers"] = [6, 1, 8] + [
        number for number in numbers if number not in (6, 1, 8)
    ]
    contract["readout"]["weight"]["values"] = [
        0.0 for _ in contract["readout"]["weight"]["values"]
    ]
    contract["per_type_scale"]["values"] = [
        2.75 + 0.1 * index for index in range(len(numbers))
    ]
    contract["per_type_shift"]["values"] = [
        0.125 * (index + 1) for index in range(len(numbers))
    ]
    contract["zbl"].update(
        cutoff=2.0,
        cutoff_p=3.5,
        half_energy_prefactor=7.199823456789,
        screening_exponents=[-3.200000123, -0.942300231, -0.402800173, -0.201600149],
        screening_coefficients=[0.181800139, 0.509900217, 0.280200193, 0.028170127],
    )
    path = tmp_path / "zbl-mixed.json"
    path.write_text(json.dumps(contract))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, precision, runtime[1], compiler="nvrtc", cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, precision, path)
    evaluator.set_streamed_edges("direct")
    types = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    positions = np.zeros((len(types), 3))
    # Receiver 0 has four mixed-species neighbors. Receivers 3/4 own
    # precisely-on/outside-cutoff edges; receiver 5 is entirely isolated.
    counts = np.array([4, 2, 1, 1, 1, 0], dtype=np.int32)
    sources = np.array([1, 2, 3, 4, 0, 2, 0, 0, 0], dtype=np.int32)
    radii = np.array([0.31, 0.73, 1.43, 2.0 - 1e-4, 0.31, 1.1, 0.73, 2.0, 2.1])
    directions = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, -1],
        ]
    )
    xyz = np.ascontiguousarray(radii[:, None] * directions)
    values = positions, types, sources, xyz
    generation = prepare(evaluator, values, counts)
    tolerance = 2e-6 if precision == "float32" else 2e-11
    for factor in (1.0, 0.9, 1.0):
        moved = np.ascontiguousarray(xyz * factor)
        actual = evaluate(evaluator, generation, moved)
        expected = reference_routing(
            runtime[0], (positions, types, sources, moved), path, counts
        )
        np.testing.assert_allclose(
            actual[0], expected[0], atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            actual[1], expected[1], atol=tolerance, rtol=tolerance
        )
        np.testing.assert_array_equal(actual[1][-1], np.zeros(3))
        if factor == 1.0:
            # These receivers have no in-cutoff ZBL terms; no per-type
            # scale may leak into the pair energy or the isolated shifts.
            np.testing.assert_array_equal(actual[0][3:], expected[0][3:])
    if precision == "float64":
        # Differentiating total energy along independent directed-edge
        # vectors also tests half-prefactor and cutoff derivative semantics.
        evaluate(evaluator, generation, xyz)
        directed = np.asarray(evaluator.node_forces).copy()
        direction = np.linspace(-0.4, 0.6, xyz.size)
        step = 1e-6
        plus = (xyz.ravel() + step * direction).reshape(-1, 3)
        minus = (xyz.ravel() - step * direction).reshape(-1, 3)
        difference = (
            evaluate(evaluator, generation, plus)[0].sum()
            - evaluate(evaluator, generation, minus)[0].sum()
        ) / (2 * step)
        np.testing.assert_allclose(
            directed @ direction, -difference, atol=2e-7, rtol=2e-7
        )
    # Replacing a larger graph by a smaller all-isolated one must overwrite
    # every returned energy rather than reading the retained scratch capacity.
    isolated_types = types[:2]
    isolated = (
        positions[:2],
        isolated_types,
        np.empty(0, dtype=np.int32),
        np.empty((0, 3)),
    )
    isolated_counts = np.zeros(2, dtype=np.int32)
    generation = prepare(evaluator, isolated, isolated_counts)
    actual = evaluate(evaluator, generation, isolated[-1])
    expected = reference_routing(runtime[0], isolated, path, isolated_counts)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], np.zeros((2, 3)))


@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
def test_isolated_nodes_match_the_reference_routing_without_edge_storage(
    runtime, compiler
):
    artifact = artifact_for(runtime, "float64", compiler)
    evaluator = evaluator_for(runtime, artifact, "float64")
    evaluator.set_streamed_edges("direct")
    positions = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    types = np.array([0, 1], dtype=np.int32)
    empty_int = np.empty(0, dtype=np.int32)
    empty_float = np.empty(0)
    generation = evaluator.prepare_graph(
        2, types, np.zeros(2, dtype=np.int32), empty_int, empty_int
    )
    evaluator.prepare_geometry(
        generation, positions.ravel(), empty_float, empty_float, empty_float, empty_int
    )
    energies, forces = evaluate(evaluator, generation, np.empty((0, 3)))
    reference_energies, _reference_forces = reference_evaluator_results(
        runtime[0],
        runtime[3],
        (
            2,
            types,
            np.zeros(2, dtype=np.int32),
            empty_int,
            empty_int,
            empty_float,
            empty_float,
        ),
    )
    np.testing.assert_allclose(energies, reference_energies, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(forces, np.zeros((2, 3)))
    assert evaluator.cuda_r1_ready
    assert evaluator.cuda_r1_fallback_count == 0


@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
def test_binary_identity_cannot_be_replaced_by_another_manifest(
    runtime, tmp_path, compiler
):
    artifact = artifact_for(runtime, "float64", compiler)
    contract = json.loads(runtime[3].read_text())
    contract["per_type_shift"]["values"][0] += 0.25
    other = tmp_path / "other-oam.json"
    other.write_text(json.dumps(contract))
    published = prepare_nequip_oam_cuda_artifact(
        other, "float64", runtime[1], compiler=compiler, cache_root=runtime[2]
    )
    native = runtime[0]
    evaluator = native.NequIPOAMKokkos(
        str(other),
        "",
        cuda_plugin_path=str(artifact.artifact_path),
        cuda_artifact_identity={"manifest_path": str(published.manifest_path)},
    )
    assert not evaluator.cuda_r1_ready
    with pytest.raises(ValueError):
        evaluator.set_streamed_edges("direct")


def test_later_self_connection_preserves_expanded_node_features(runtime, tmp_path):
    from test_nequip_oam_lane_layout import (
        _asymmetric_args,
        _self_connection_contract,
        _self_tensor_product,
        _uvw,
        _uvw_split_self_connection,
    )

    contract = _self_connection_contract(_uvw_split_self_connection())
    # Layer 0 expands a four-component row to seven components. Layer 1's
    # self connection consumes the added vector block, not the initial layout.
    contract["layers"][1]["self_connection"] = _self_tensor_product(
        "1x0e+1x1o+1x1o",
        "1x0e+1x1o",
        "1x0e",
        [_uvw(2, 1, 0, 1, 1, 1, (3, 3, 1))],
        [0.7],
        [1.0],
    )
    path = tmp_path / "expanded-self-input.json"
    path.write_text(json.dumps(contract))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, "float64", runtime[1], cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, "float64", path)
    evaluator.set_streamed_edges("direct")
    args = _asymmetric_args()
    reference_energies, reference_forces = reference_evaluator_results(
        runtime[0], path, args
    )
    evaluator.compute_node_energies_forces(*args)
    np.testing.assert_allclose(
        evaluator.node_energies, reference_energies, atol=1e-11, rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.node_forces).reshape(-1, 3),
        np.asarray(reference_forces).reshape(-1, 3),
        atol=1e-11,
        rtol=1e-9,
    )


def _channel_contract(mode):
    from test_nequip_oam_lane_layout import (
        FIXTURE,
        _finalize_contract,
        _layer,
        _linear,
        _uvu,
        _uvw,
    )

    width = 1 if mode == "uuu" else 65
    node = f"{width}x0e"
    output_width = 3 if mode == "uvw" else width
    output = f"{output_width}x0e"
    instruction = (
        _uvw(0, 0, 0, width, 1, output_width, (1, 1, 1))
        if mode == "uvw"
        else _uvu(0, 0, 0, width, 1, (1, 1, 1))
    )
    if mode == "uuu":
        instruction["connection_mode"] = "uuu"
        instruction["path_shape"] = [width]
    function_count = width * output_width if mode == "uvw" else width
    layer = _layer(
        "interaction_0",
        node,
        "1x0e",
        output,
        "1x0e",
        "1x0e",
        {
            "type": "scalar_gate",
            "scalar_irreps": "1x0e",
            "gate_irreps": "0x0e",
            "gated_irreps": "0x0e",
        },
        _linear(
            node,
            node,
            [(0, 0, (width, width))],
            (0.8 * np.eye(width)).ravel().tolist(),
            [0.0] * width,
            [1.0] * width,
        ),
        _linear(
            output,
            "1x0e",
            [(0, 0, (output_width, 1))],
            np.linspace(-0.2, 0.7, output_width).tolist(),
            [0.1],
            [0.7],
        ),
        [instruction, dict(instruction, path_weight=-0.3)],
        2 * function_count,
    )
    return _finalize_contract(json.loads(FIXTURE.read_text()), [layer])


@pytest.mark.parametrize("mode", ("uvu", "uuu", "uvw"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
def test_cooperative_reverse_shared_receivers_and_partial_channels(
    runtime, tmp_path, mode, precision, compiler
):
    # Incorrect channel ownership loses repeated-path or uvw output contributions;
    # per-edge ownership mistakes race when several receiver rows share a source.
    if compiler == "nvcc" and not shutil.which("nvcc"):
        pytest.skip("nvcc is required for the shared-library loader variant")
    path = tmp_path / f"channels-{mode}.json"
    path.write_text(json.dumps(_channel_contract(mode)))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, precision, runtime[1], compiler=compiler, cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, precision, path)
    evaluator.set_streamed_edges("direct")
    positions = np.array([[0.0, 0.0, 0.0], [0.9, 0.3, -0.2], [1.4, -0.8, 0.5]])
    types = np.array([0, 2, 1], dtype=np.int32)
    receivers = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    sources = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
    xyz = positions[sources] - positions[receivers]
    args = (
        3,
        types,
        np.full(3, 2, dtype=np.int32),
        sources,
        types[sources],
        xyz.ravel(),
        np.linalg.norm(xyz, axis=1),
    )
    reference_energies, reference_forces = reference_evaluator_results(
        runtime[0], path, args
    )
    evaluator.compute_node_energies_forces(*args)
    tolerance = 2e-4 if precision == "float32" else 1e-9
    np.testing.assert_allclose(
        evaluator.node_energies, reference_energies, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.node_forces).reshape(-1, 3),
        np.asarray(reference_forces).reshape(-1, 3),
        atol=tolerance,
        rtol=tolerance,
    )
    assert evaluator.cuda_r1_fallback_count == 0


def _wide_angular_channel_contract():
    from test_nequip_oam_lane_layout import (
        _build_contract,
        _finalize_contract,
        _linear,
        _radial,
        _uvu,
        _uvw,
    )

    contract = _build_contract()
    first, second = contract["layers"]
    width = 137
    output = f"1x0e+{width}x1o"
    angular = _uvw(0, 1, 1, 1, 1, width, (1, 3, 3))
    angular["wigner_3j"]["values"] = [0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25]
    instructions = [
        _uvu(0, 0, 0, 1, 1, (1, 1, 1)),
        angular,
        dict(angular, path_weight=-0.3),
    ]
    first["tp_output_irreps"] = output
    first["edge_tensor_product"].update(
        irreps_out=output,
        instructions=instructions,
        weight_numel=1 + 2 * width,
        weight={"shape": [1 + 2 * width], "values": [0.3] * (1 + 2 * width)},
        output_mask={
            "shape": [1 + 3 * width],
            "values": [0.8]
            + [
                (0.0 if index % 11 == 0 else (-0.4 if index % 3 == 0 else 0.7))
                for index in range(3 * width)
            ],
        },
    )
    first["radial_basis"] = _radial(1 + 2 * width)
    first["radial_basis"]["mlp"]["layers"][0]["weight"]["values"] = [
        0.2 + 0.01 * (index % 17) for index in range(1 + 2 * width)
    ]
    # The partial group must materially affect the consuming vector, not
    # hide beneath FP32 parity tolerances among the lower channel groups.
    vector_weights = np.linspace(-0.15, 0.2, width)
    vector_weights[128:] = 5.0
    first["post_linear"] = _linear(
        output,
        "1x0e+1x1o",
        [(0, 0, (1, 1)), (1, 1, (width, 1))],
        [0.5] + vector_weights.tolist(),
        [0.1, 0.0, 0.2, -0.1],
        [1.0, 0.5, -0.25, 0.75],
    )
    return _finalize_contract(contract, [first, second])


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
def test_generated_wide_angular_masked_outputs(runtime, tmp_path, precision, compiler):
    # A 137-channel vector block crosses the CTA's 128-channel group and
    # has a partial final group. Its masked, repeated paths feed the next
    # layer's vector-to-scalar contraction, so angular corruption is observable.
    from test_nequip_oam_lane_layout import _asymmetric_args

    if compiler == "nvcc" and not shutil.which("nvcc"):
        pytest.skip("nvcc is required for the shared-library loader variant")
    path = tmp_path / "wide-angular.json"
    path.write_text(json.dumps(_wide_angular_channel_contract()))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, precision, runtime[1], compiler=compiler, cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, precision, path)
    evaluator.set_streamed_edges("direct")
    args = _asymmetric_args()
    reference_energies, reference_forces = reference_evaluator_results(
        runtime[0], path, args
    )
    evaluator.compute_node_energies_forces(*args)
    tolerance = 2e-4 if precision == "float32" else 1e-9
    np.testing.assert_allclose(
        evaluator.node_energies, reference_energies, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.node_forces).reshape(-1, 3),
        np.asarray(reference_forces).reshape(-1, 3),
        atol=tolerance,
        rtol=tolerance,
    )


def _degree_contract(degree):
    from test_nequip_oam_lane_layout import (
        FIXTURE,
        _finalize_contract,
        _layer,
        _linear,
        _uvu,
    )

    parity = "e" if degree % 2 == 0 else "o"
    width = 2 * degree + 1
    edge = "+".join(
        f"1x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(degree + 1)
    )
    node = f"1x0e+1x{degree}{parity}"
    angular = _uvu(0, degree, 1, 1, 1, (1, width, width))
    angular["wigner_3j"]["values"] = (np.eye(width) / np.sqrt(width)).ravel().tolist()
    contraction = _uvu(1, degree, 0, 1, 1, (width, width, 1))
    contraction["wigner_3j"]["values"] = (
        (np.eye(width) / np.sqrt(width)).ravel().tolist()
    )
    first = _layer(
        "interaction_0",
        "1x0e",
        edge,
        node,
        node,
        node,
        {
            "type": "scalar_gate",
            "scalar_irreps": "1x0e",
            "gate_irreps": "0x0e",
            "gated_irreps": f"1x{degree}{parity}",
        },
        _linear("1x0e", "1x0e", [(0, 0, (1, 1))], [0.8], [0.1], [1.0]),
        _linear(
            node,
            node,
            [(0, 0, (1, 1)), (1, 1, (1, 1))],
            [0.5, 1.2],
            [0.1] + [0.0] * width,
            [1.0] * (width + 1),
        ),
        [_uvu(0, 0, 0, 1, 1, (1, 1, 1)), angular],
        2,
    )
    second = _layer(
        "interaction_1",
        node,
        edge,
        "1x0e",
        "1x0e",
        "1x0e",
        {
            "type": "scalar_gate",
            "scalar_irreps": "1x0e",
            "gate_irreps": "0x0e",
            "gated_irreps": "0x0e",
        },
        _linear(
            node,
            node,
            [(0, 0, (1, 1)), (1, 1, (1, 1))],
            [0.7, -0.6],
            [0.0] * (width + 1),
            [1.0] * (width + 1),
        ),
        _linear("1x0e", "1x0e", [(0, 0, (1, 1))], [0.9], [0.0], [1.0]),
        [contraction],
        1,
    )
    return _finalize_contract(json.loads(FIXTURE.read_text()), [first, second])


def _degree_three_contract():
    return _degree_contract(3)


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
def test_degree_three_harmonics_match_the_reference_routing_and_energy_gradient(
    runtime, tmp_path, precision, compiler
):
    from test_nequip_oam_lane_layout import _asymmetric_args

    if compiler == "nvcc" and not shutil.which("nvcc"):
        pytest.skip("nvcc is required for the shared-library loader variant")
    path = tmp_path / "degree-three.json"
    path.write_text(json.dumps(_degree_three_contract()))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, precision, runtime[1], compiler=compiler, cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, precision, path)
    evaluator.set_streamed_edges("direct")
    args = _asymmetric_args()
    reference_energies, reference_forces = reference_evaluator_results(
        runtime[0], path, args
    )
    evaluator.compute_node_energies_forces(*args)
    tolerance = 2e-4 if precision == "float32" else 1e-9
    np.testing.assert_allclose(
        evaluator.node_energies, reference_energies, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.node_forces).reshape(-1, 3),
        np.asarray(reference_forces).reshape(-1, 3),
        atol=tolerance,
        rtol=tolerance,
    )
    if precision == "float64":
        forces = np.asarray(evaluator.node_forces).copy()
        # An oblique perturbation observes every shuffled angular derivative.
        direction = np.linspace(-0.4, 0.7, len(args[-2]))
        step = 1e-5
        energies = []
        for sign in (1, -1):
            xyz = args[-2] + sign * step * direction
            evaluator.compute_node_energies_forces(
                *args[:-2], xyz, np.linalg.norm(xyz.reshape(-1, 3), axis=1)
            )
            energies.append(np.asarray(evaluator.node_energies).sum())
        np.testing.assert_allclose(
            forces @ direction,
            -(energies[0] - energies[1]) / (2 * step),
            atol=1e-7,
            rtol=1e-6,
        )


@pytest.mark.parametrize("compiler", ("nvrtc", "nvcc"))
def test_degree_four_harmonics_match_the_reference_routing(runtime, tmp_path, compiler):
    from test_nequip_oam_lane_layout import _asymmetric_args

    if compiler == "nvcc" and not shutil.which("nvcc"):
        pytest.skip("nvcc is required for the shared-library loader variant")
    path = tmp_path / "degree-four.json"
    path.write_text(json.dumps(_degree_contract(4)))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, "float32", runtime[1], compiler=compiler, cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, "float32", path)
    evaluator.set_streamed_edges("direct")
    args = _asymmetric_args()
    reference_energies, reference_forces = reference_evaluator_results(
        runtime[0], path, args
    )

    evaluator.compute_node_energies_forces(*args)

    np.testing.assert_allclose(
        evaluator.node_energies, reference_energies, atol=2e-4, rtol=2e-4
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.node_forces).reshape(-1, 3),
        np.asarray(reference_forces).reshape(-1, 3),
        atol=2e-4,
        rtol=2e-4,
    )


def _conditioned_angular_contract(scalar_attributes=True):
    from test_nequip_oam_lane_layout import (
        _build_contract,
        _finalize_contract,
        _self_connection_layers,
        _uvw,
        _uvw_split_self_connection,
    )

    contract = _build_contract()
    connection = _uvw_split_self_connection()
    if scalar_attributes:
        connection["irreps_in2"] = "1x0e"
    for instruction in connection["instructions"][1:]:
        instruction["wigner_3j"]["values"] = [0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25]
    connection["instructions"].append(
        dict(connection["instructions"][1], path_weight=-0.6)
    )
    connection["weight"] = {"shape": [4], "values": [0.3, -0.4, 0.25, 0.9]}
    connection["output_mask"]["values"] = [0.7, 0.8, -0.5, 1.2, 0.6, -0.35, 0.45]
    layers = _self_connection_layers(connection)
    if scalar_attributes:
        layers.insert(0, contract["layers"][0])
    else:
        # This active vector-attribute path must retain generic self execution.
        connection["instructions"].append(_uvw(1, 1, 0, 1, 1, 1, (3, 3, 1)))
        connection["weight"]["values"].append(-0.2)
        connection["weight"]["shape"] = [5]
    contract = _finalize_contract(contract, layers)
    numbers = contract["atomic_numbers"]
    contract["atomic_numbers"] = numbers[2:3] + numbers[:2] + numbers[3:]
    return contract


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("scalar_attributes", (True, False))
def test_conditioned_self_masked_split_blocks_and_type_mapping(
    runtime, tmp_path, precision, scalar_attributes
):
    from test_nequip_oam_lane_layout import _asymmetric_args

    # Missing masks, shared-output paths, model-type lookup, or transpose terms
    # change these energies/coordinate derivatives independently of routing.
    path = tmp_path / "conditioned-angular.json"
    path.write_text(json.dumps(_conditioned_angular_contract(scalar_attributes)))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, precision, runtime[1], cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, precision, path)
    evaluator.set_streamed_edges("direct")
    args = _asymmetric_args()
    reference_energies, reference_forces = reference_evaluator_results(
        runtime[0], path, args
    )
    evaluator.compute_node_energies_forces(*args)
    tolerance = 2e-4 if precision == "float32" else 1e-9
    np.testing.assert_allclose(
        evaluator.node_energies, reference_energies, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(evaluator.node_forces).reshape(-1, 3),
        np.asarray(reference_forces).reshape(-1, 3),
        atol=tolerance,
        rtol=tolerance,
    )
    if precision == "float64":
        directed = np.asarray(evaluator.node_forces).copy()
        direction = np.linspace(-0.4, 0.7, len(args[-2]))
        step = 1e-5
        energies = []
        for sign in (1, -1):
            moved = args[-2] + sign * step * direction
            evaluator.compute_node_energies_forces(
                *args[:-2], moved, np.linalg.norm(moved.reshape(-1, 3), axis=1)
            )
            energies.append(np.asarray(evaluator.node_energies).sum())
        np.testing.assert_allclose(
            directed @ direction,
            -(energies[0] - energies[1]) / (2 * step),
            atol=1e-7,
            rtol=1e-6,
        )


def test_cooperative_reverse_reuses_large_small_and_replaced_topology(
    runtime, tmp_path
):
    path = tmp_path / "conditioned-reuse.json"
    path.write_text(json.dumps(_conditioned_angular_contract()))
    artifact = prepare_nequip_oam_cuda_artifact(
        path, "float64", runtime[1], compiler="nvrtc", cache_root=runtime[2]
    )
    evaluator = evaluator_for(runtime, artifact, "float64", path)
    evaluator.set_streamed_edges("direct")
    for pairs in (513, 1, 517):
        positions, types, sources, _ = graph(pairs=pairs)
        types = np.arange(len(types), dtype=np.int32) % 3
        positions = positions + np.random.default_rng(pairs).normal(
            0.0, 0.02, positions.shape
        )
        if pairs == 517:
            sources = np.roll(np.arange(2 * pairs, dtype=np.int32), -1)
        xyz = np.ascontiguousarray(positions[sources] - positions)
        values = positions, types, sources, xyz
        generation = prepare(evaluator, values)
        actual = evaluate(evaluator, generation, xyz)
        expected = reference_routing(runtime[0], values, path)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-7, rtol=1e-7)
        assert evaluator.cuda_r1_fallback_count == 0
    empty_int = np.empty(0, dtype=np.int32)
    generation = evaluator.prepare_graph(0, empty_int, empty_int, empty_int, empty_int)
    evaluator.prepare_geometry(
        generation, np.empty(0), np.empty(0), np.empty(0), np.empty(0), empty_int
    )
    energies, forces = evaluate(evaluator, generation, np.empty((0, 3)))
    np.testing.assert_array_equal(energies, np.empty(0))
    np.testing.assert_array_equal(forces, np.empty((0, 3)))
