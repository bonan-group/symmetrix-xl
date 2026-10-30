"""Lane-major layout must not depend on how a linear's irreps are aggregated.

A node tensor can be described by different but node-major-equivalent
aggregations of adjacent same-angular irrep blocks: the producing tensor
product writes the fine form (one block per instruction), while the consuming
linear may merge those blocks.  The lane-major layout is defined per block, so
the runtime must present each linear with the producing decomposition;
otherwise the merged description silently re-indexes every component of that
row, and the defect only surfaces once an angular-mixing layer downstream
reaches a scalar output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "data/nequip/oam_s_v2_minimal.json"


@pytest.fixture(scope="module")
def native():
    try:
        from symmetrix import symmetrix as native_symmetrix
    except Exception as error:  # pragma: no cover - environment without backend
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    return native_symmetrix


def _reference(native, path):
    """The artifact-free reference routing: the portable Kokkos implementation.

    ``NequIPOAMKokkos(path)`` with no artifact argument uses the selected
    execution space, and ``direct`` is the native mode the reference routing
    hands it.
    """

    evaluator = native.NequIPOAMKokkos(str(path))
    evaluator.set_streamed_edges("reference")
    return evaluator


def _dimension(irreps):
    total = 0
    for part in irreps.split("+"):
        part = part.strip()
        if not part:
            continue
        multiplicity, angular = part.split("x")
        total += int(multiplicity) * (2 * int(angular[0]) + 1)
    return total


def _linear(irreps_in, irreps_out, instructions, weight, bias, mask):
    return {
        "irreps_in": irreps_in,
        "irreps_out": irreps_out,
        "instructions": [
            {
                "i_in": i_in,
                "i_out": i_out,
                "path_shape": list(shape),
                "path_weight": 1.0,
            }
            for i_in, i_out, shape in instructions
        ],
        "weight": {"shape": [len(weight)], "values": weight},
        "bias": {"shape": [len(bias)], "values": bias},
        "output_mask": {"shape": [len(mask)], "values": mask},
    }


def _uvu(i_in1, i_in2, i_out, multiplicity_1, multiplicity_2, wigner_shape):
    return {
        "i_in1": i_in1,
        "i_in2": i_in2,
        "i_out": i_out,
        "connection_mode": "uvu",
        "has_weight": True,
        "path_shape": [multiplicity_1, multiplicity_2],
        "path_weight": 1.0,
        "wigner_3j": {
            "shape": list(wigner_shape),
            "values": [0.25] * (wigner_shape[0] * wigner_shape[1] * wigner_shape[2]),
        },
    }


def _radial(function_count):
    return {
        "type": "bessel",
        "num_basis": 1,
        "bessel_weights": {"shape": [1], "values": [1]},
        "cutoff": {"type": "polynomial", "p": 5},
        "input_factor": 1,
        "mlp": {
            "layers": [
                {
                    "weight": {
                        "shape": [function_count, 1],
                        "values": [
                            0.4 + 0.05 * index for index in range(function_count)
                        ],
                    },
                    "bias": {
                        "shape": [function_count],
                        "values": [0.0] * function_count,
                    },
                }
            ]
        },
    }


def _layer(
    name,
    node_in,
    edge,
    tp_out,
    pre_gate,
    node_out,
    gate,
    pre_linear,
    post_linear,
    edge_instructions,
    function_count,
    self_connection=None,
):
    return {
        "name": name,
        "node_irreps_in": node_in,
        "edge_irreps": edge,
        "tp_output_irreps": tp_out,
        "node_irreps_pre_gate": pre_gate,
        "node_irreps_out": node_out,
        "pre_linear": pre_linear,
        "radial_basis": _radial(function_count),
        "edge_tensor_product": {
            "irreps_in1": node_in,
            "irreps_in2": edge,
            "irreps_out": tp_out,
            "instructions": edge_instructions,
            "weight_numel": function_count,
            "weight": {
                "shape": [function_count],
                "values": [
                    0.3 + 0.1 * index * (-1 if index % 2 else 1)
                    for index in range(function_count)
                ],
            },
            "output_mask": {
                "shape": [_dimension(tp_out)],
                "values": [1.0 + 0.2 * index for index in range(_dimension(tp_out))],
            },
        },
        "post_linear": post_linear,
        "self_connection": self_connection,
        "normalization": {"type": "global_scalar", "factor": 0.9},
        "gate": gate,
    }


def _uvw(
    i_in1, i_in2, i_out, multiplicity_1, multiplicity_2, multiplicity_3, wigner_shape
):
    return {
        "i_in1": i_in1,
        "i_in2": i_in2,
        "i_out": i_out,
        "connection_mode": "uvw",
        "has_weight": True,
        "path_shape": [multiplicity_1, multiplicity_2, multiplicity_3],
        "path_weight": 1.0,
        "wigner_3j": {
            "shape": list(wigner_shape),
            "values": [0.25] * (wigner_shape[0] * wigner_shape[1] * wigner_shape[2]),
        },
    }


def _uvu_bare(i_in1, i_in2, i_out, multiplicity_1, multiplicity_2, wigner_shape):
    """A uvu instruction carrying internal weights (the self connection form)."""

    return {
        "i_in1": i_in1,
        "i_in2": i_in2,
        "i_out": i_out,
        "connection_mode": "uvu",
        "has_weight": True,
        "path_shape": [multiplicity_1, multiplicity_2],
        "path_weight": 1.0,
        "wigner_3j": {
            "shape": list(wigner_shape),
            "values": [0.25] * (wigner_shape[0] * wigner_shape[1] * wigner_shape[2]),
        },
    }


def _self_tensor_product(
    irreps_in1, irreps_in2, irreps_out, instructions, weight, mask
):
    return {
        "irreps_in1": irreps_in1,
        "irreps_in2": irreps_in2,
        "irreps_out": irreps_out,
        "instructions": instructions,
        "weight": {"shape": [len(weight)], "values": weight},
        "output_mask": {"shape": [len(mask)], "values": mask},
    }


def _finalize_contract(contract, layers):
    """Attach the node-attribute table, layers, readout, and radial certificates."""

    counts = len(contract["model_atomic_numbers"])
    attribute_width = _dimension(layers[0]["node_irreps_in"])
    contract["node_attributes"] = {
        "shape": [counts, attribute_width],
        "values": [
            0.05 * (((index + component) % 7) - 3)
            for index in range(counts)
            for component in range(attribute_width)
        ],
    }
    contract["layers"] = layers
    contract["num_layers"] = len(layers)
    contract["readout"] = _linear("1x0e", "1x0e", [(0, 0, (1, 1))], [1.0], [0.0], [1.0])
    exporter = contract["model_metadata"]["exporter_fingerprint"]
    model_fingerprint = "sha256:" + hashlib.sha256(exporter.encode()).hexdigest()
    from symmetrix.extract_nequip_data import _oam_radial_evaluator
    from symmetrix.jit_codegen_common import spline_table_fingerprint

    for layer in contract["layers"]:
        evaluate, function_count = _oam_radial_evaluator(
            layer["radial_basis"], cutoff=float(contract["r_cut"])
        )
        layer["kokkos_radial"] = {
            "grid_min": 0.0,
            "cutoff": float(contract["r_cut"]),
            "num_spline_points": 256,
            "spacing": float(contract["r_cut"]) / 255.0,
            "function_count": int(layer["edge_tensor_product"]["weight_numel"]),
            "edge_type_count": 1,
            "model_fingerprint": model_fingerprint,
            "table_fingerprint": spline_table_fingerprint(
                evaluate,
                function_count,
                num_points=256,
                grid_min=0.0,
                cutoff=float(contract["r_cut"]),
            ),
            "precision": "float64",
            "exporter_version": "symmetrix-nequip-exporter/2",
            "certification": {
                "grid_points": 256,
                "clamped_points": 0,
                "max_abs_value_error": 0.0,
                "max_abs_derivative_error": 0.0,
            },
        }
    return contract


def _build_contract():
    """Two layers whose consuming linears merge adjacent same-angular blocks.

    Layer 0: node "1x0e"; edge "1x0e+1x1o"; tp output "1x0e+1x1o+1x1o" (two
    adjacent 1o blocks); post_linear declares "1x0e+2x1o" - the merged form.
    Layer 1: node "1x0e+1x1o"; its edge product mixes 1o x 1o into 1x0e, so
    the corrupted non-scalar features of layer 0 reach the readout scalar.
    """
    contract = json.loads(FIXTURE.read_text())
    counts = len(contract["model_atomic_numbers"])
    contract["node_attributes"] = {
        "shape": [counts, 1],
        "values": [0.05 * ((index % 7) - 3) for index in range(counts)],
    }
    layer0 = _layer(
        "interaction_0",
        node_in="1x0e",
        edge="1x0e+1x1o",
        tp_out="1x0e+1x1o+1x1o",
        pre_gate="1x0e+1x1o",
        node_out="1x0e+1x1o",
        gate={
            "type": "scalar_gate",
            "scalar_irreps": "1x0e",
            "gate_irreps": "0x0e",
            "gated_irreps": "1x1o",
        },
        pre_linear=_linear("1x0e", "1x0e", [(0, 0, (1, 1))], [0.8], [0.1], [1.0]),
        post_linear=_linear(
            "1x0e+2x1o",  # merged form of the producing "1x0e+1x1o+1x1o"
            "1x0e+1x1o",
            [(0, 0, (1, 1)), (1, 1, (2, 1))],
            [0.5, 0.6, -0.4],
            [0.1, 0.0, 0.2, -0.1],
            [1.0, 0.5, -0.25, 0.75],
        ),
        edge_instructions=[
            _uvu(0, 0, 0, 1, 1, (1, 1, 1)),
            _uvu(0, 1, 1, 1, 1, (1, 3, 3)),
            _uvu(0, 1, 2, 1, 1, (1, 3, 3)),
        ],
        function_count=3,
    )
    layer1 = _layer(
        "interaction_1",
        node_in="1x0e+1x1o",
        edge="1x0e+1x1o",
        tp_out="1x0e",
        pre_gate="1x0e",
        node_out="1x0e",
        gate={
            "type": "scalar_gate",
            "scalar_irreps": "1x0e",
            "gate_irreps": "0x0e",
            "gated_irreps": "0x0e",
        },
        pre_linear=_linear(
            "1x0e+1x1o",
            "1x0e+1x1o",
            [(0, 0, (1, 1)), (1, 1, (1, 1))],
            [0.7, -0.6],
            [0.0, 0.2, -0.1, 0.3],
            [1.0, 1.0, 0.9, 0.8],
        ),
        post_linear=_linear("1x0e", "1x0e", [(0, 0, (1, 1))], [0.9], [0.0], [1.0]),
        edge_instructions=[
            _uvu(0, 0, 0, 1, 1, (1, 1, 1)),
            _uvu(1, 1, 0, 1, 1, (3, 3, 1)),
        ],
        function_count=2,
    )
    return _finalize_contract(contract, [layer0, layer1])


@pytest.mark.oam_runtime
def test_merged_linear_irreps_keeps_lane_parity(native, tmp_path):
    from symmetrix.nequip_schema import validate_nequip_oam_data
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    contract = _build_contract()
    validate_nequip_oam_data(contract)
    path = tmp_path / "oam-merged-irreps.json"
    path.write_text(json.dumps(contract))
    artifact = prepare_nequip_oam_receiver_host_artifact(
        path, "float64", cache_dir=tmp_path / "cache"
    )
    reference = _reference(native, path)
    kokkos = native.NequIPOAMKokkos(
        str(path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )

    # A deliberately asymmetric cluster so the odd-angular (l=1) content is
    # non-zero: a symmetric cell would hide any l>0 defect.
    node_types = np.asarray((0, 1, 2), dtype=np.int32)
    num_neigh = np.asarray((2, 0, 1), dtype=np.int32)
    indices = np.asarray((1, 2, 0), dtype=np.int32)
    neigh_types = np.asarray((1, 2, 0), dtype=np.int32)
    xyz = np.asarray((0.9, 0.3, -0.2, 1.4, -0.8, 0.5, -0.7, 0.6, 1.1), dtype=np.float64)
    radii = np.linalg.norm(xyz.reshape(-1, 3), axis=1)
    args = (3, node_types, num_neigh, indices, neigh_types, xyz, radii)
    reference.compute_node_energies_forces(*args)
    kokkos.compute_node_energies_forces(*args)

    reference_energy = np.asarray(reference.node_energies, dtype=float)
    kokkos_energy = np.asarray(kokkos.node_energies, dtype=float)
    reference_forces = np.asarray(reference.node_forces, dtype=float)
    kokkos_forces = np.asarray(kokkos.node_forces, dtype=float)
    assert np.abs(reference_energy).max() > 1e-6
    np.testing.assert_allclose(kokkos_energy, reference_energy, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(kokkos_forces, reference_forces, rtol=1e-9, atol=1e-11)


def _self_connection_layers(self_connection):
    """Layer 0 carries `self_connection`; layer 1 folds its odd content to a scalar.

    Layer 0 keeps node irreps "1x0e+1x1o" with a pre-gate row of
    "1x0e+1x1o+1x1o" so an aggregation-equivalent ("1x0e+2x1o") self-connection
    description exists, and layer 1 mixes 1o x 1o into 1x0e so the l>0 content
    that a mis-indexed self row would corrupt reaches the readout scalar.
    """

    layer0 = _layer(
        "interaction_0",
        node_in="1x0e+1x1o",
        edge="1x0e+1x1o",
        tp_out="1x0e+1x1o+1x1o",
        pre_gate="1x0e+1x1o+1x1o",
        node_out="1x0e+1x1o+1x1o",
        gate={
            "type": "scalar_gate",
            "scalar_irreps": "0x0e",
            "gate_irreps": "0x0e",
            "gated_irreps": "1x0e+1x1o+1x1o",
        },
        pre_linear=_linear(
            "1x0e+1x1o",
            "1x0e+1x1o",
            [(0, 0, (1, 1)), (1, 1, (1, 1))],
            [0.7, -0.6],
            [0.0, 0.2, -0.1, 0.3],
            [1.0, 1.0, 0.9, 0.8],
        ),
        post_linear=_linear(
            "1x0e+1x1o+1x1o",
            "1x0e+1x1o+1x1o",
            [(0, 0, (1, 1)), (1, 1, (1, 1)), (2, 2, (1, 1))],
            [0.5, -0.4, 0.3],
            [0.1, 0.0, 0.2, -0.1, 0.05, 0.15, -0.2],
            [1.0, 0.5, -0.25, 0.75, 0.6, -0.35, 0.45],
        ),
        edge_instructions=[
            _uvu(0, 0, 0, 1, 1, (1, 1, 1)),
            _uvu(1, 1, 1, 1, 1, (3, 3, 3)),
            _uvu(1, 1, 2, 1, 1, (3, 3, 3)),
        ],
        function_count=3,
        self_connection=self_connection,
    )
    layer1 = _layer(
        "interaction_1",
        node_in="1x0e+1x1o+1x1o",
        edge="1x0e+1x1o",
        tp_out="1x0e",
        pre_gate="1x0e",
        node_out="1x0e",
        gate={
            "type": "scalar_gate",
            "scalar_irreps": "1x0e",
            "gate_irreps": "0x0e",
            "gated_irreps": "0x0e",
        },
        pre_linear=_linear(
            "1x0e+1x1o+1x1o",
            "1x0e+1x1o+1x1o",
            [(0, 0, (1, 1)), (1, 1, (1, 1)), (2, 2, (1, 1))],
            [0.7, -0.6, 0.4],
            [0.0, 0.2, -0.1, 0.3, 0.15, -0.25, 0.05],
            [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        ),
        post_linear=_linear("1x0e", "1x0e", [(0, 0, (1, 1))], [0.9], [0.0], [1.0]),
        edge_instructions=[
            _uvu(0, 0, 0, 1, 1, (1, 1, 1)),
            _uvu(1, 1, 0, 1, 1, (3, 3, 1)),
            _uvu(2, 1, 0, 1, 1, (3, 3, 1)),
        ],
        function_count=3,
    )
    return [layer0, layer1]


def _self_connection_contract(self_connection):
    contract = json.loads(FIXTURE.read_text())
    return _finalize_contract(contract, _self_connection_layers(self_connection))


def _uvw_self_connection(irreps_out, instructions, weight):
    """A (u,v,w) self connection from the node irreps and the node attributes."""

    return _self_tensor_product(
        "1x0e+1x1o",
        "1x0e+1x1o",
        irreps_out,
        instructions,
        weight,
        [1.0] * _dimension("1x0e+1x1o+1x1o"),
    )


def _uvw_split_self_connection():
    """One (u,v,w) instruction per pre-gate block."""

    return _uvw_self_connection(
        "1x0e+1x1o+1x1o",
        [
            _uvw(0, 0, 0, 1, 1, 1, (1, 1, 1)),
            _uvw(1, 0, 1, 1, 1, 1, (3, 1, 3)),
            _uvw(1, 0, 2, 1, 1, 1, (3, 1, 3)),
        ],
        [0.3, -0.4, 0.25],
    )


def _uvw_merged_self_connection():
    """The index-preserving "1x0e+2x1o" description of the same pre-gate row."""

    return _uvw_self_connection(
        "1x0e+2x1o",
        [
            _uvw(0, 0, 0, 1, 1, 1, (1, 1, 1)),
            _uvw(1, 0, 1, 1, 1, 2, (3, 1, 3)),
        ],
        [0.3, -0.4, 0.25],
    )


def _uvu_self_connection():
    """The same connection in (u,v,u) mode, which the lane kernels cannot serve."""

    return _self_tensor_product(
        "1x0e+1x1o",
        "1x0e+1x1o",
        "1x0e+1x1o+1x1o",
        [
            _uvu_bare(0, 0, 0, 1, 1, (1, 1, 1)),
            _uvu_bare(1, 0, 1, 1, 1, (3, 1, 3)),
            _uvu_bare(1, 0, 2, 1, 1, (3, 1, 3)),
        ],
        [0.3, -0.4, 0.25],
        [1.0] * _dimension("1x0e+1x1o+1x1o"),
    )


def _asymmetric_args():
    """A deliberately asymmetric cluster so the odd-angular content is nonzero."""

    node_types = np.asarray((0, 1, 2), dtype=np.int32)
    num_neigh = np.asarray((2, 0, 1), dtype=np.int32)
    indices = np.asarray((1, 2, 0), dtype=np.int32)
    neigh_types = np.asarray((1, 2, 0), dtype=np.int32)
    xyz = np.asarray((0.9, 0.3, -0.2, 1.4, -0.8, 0.5, -0.7, 0.6, 1.1), dtype=np.float64)
    radii = np.linalg.norm(xyz.reshape(-1, 3), axis=1)
    return (3, node_types, num_neigh, indices, neigh_types, xyz, radii)


def _self_connection_parity(native, tmp_path, contract, tag):
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    path = tmp_path / f"oam-self-{tag}.json"
    path.write_text(json.dumps(contract))
    artifact = prepare_nequip_oam_receiver_host_artifact(
        path, "float64", cache_dir=tmp_path / "cache"
    )
    reference = _reference(native, path)
    kokkos = native.NequIPOAMKokkos(
        str(path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    args = _asymmetric_args()
    reference.compute_node_energies_forces(*args)
    kokkos.compute_node_energies_forces(*args)
    reference_energy = np.asarray(reference.node_energies, dtype=float)
    assert np.abs(reference_energy).max() > 1e-6
    np.testing.assert_allclose(
        np.asarray(kokkos.node_energies, dtype=float),
        reference_energy,
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(kokkos.node_forces, dtype=float),
        np.asarray(reference.node_forces, dtype=float),
        rtol=1e-9,
        atol=1e-11,
    )
    return kokkos


@pytest.mark.oam_runtime
def test_self_tensor_product_lane_kernels_are_used_when_plans_match(native, tmp_path):
    """A (u,v,w) self connection that matches the node plans takes the lane path."""

    contract = _self_connection_contract(_uvw_split_self_connection())
    kokkos = _self_connection_parity(native, tmp_path, contract, "lane")
    assert "self_tensor_product_layout=lane" in kokkos.execution_diagnostic


@pytest.mark.oam_runtime
def test_self_tensor_product_lane_workspace_is_arena_owned(native, tmp_path):
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    path = tmp_path / "oam-self-arena.json"
    path.write_text(json.dumps(_self_connection_contract(_uvw_split_self_connection())))
    artifact = prepare_nequip_oam_receiver_host_artifact(
        path, "float64", cache_dir=tmp_path / "cache"
    )
    evaluator = native.NequIPOAMKokkos(
        str(path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
    evaluator.compute_node_energies_forces(*_asymmetric_args())

    memory = evaluator.execution_memory_report
    assert memory["arena_bilinear_workspace_bytes"] > 0
    assert memory["independent_graph_temporary_bytes"] == 0
    assert memory["canaries_intact"] is True


@pytest.mark.oam_runtime
def test_self_tensor_product_falls_back_when_lane_kernels_do_not_apply(
    native, tmp_path
):
    """A (u,v,u) self connection runs the node-major path, never the lane kernels."""

    contract = _self_connection_contract(_uvu_self_connection())
    kokkos = _self_connection_parity(native, tmp_path, contract, "fallback")
    assert "self_tensor_product_layout=definition_order(layer0)" in (
        kokkos.execution_diagnostic
    )


@pytest.mark.oam_runtime
def test_self_tensor_product_accepts_an_equivalent_aggregation(native, tmp_path):
    """An index-preserving aggregation difference is accepted and equivalent.

    "1x0e+1x1o+1x1o" and "1x0e+2x1o" hold the same component at every node-major
    index, so a self connection described either way must load and produce the
    same reference energies.
    """

    reference = _self_connection_contract(_uvw_split_self_connection())
    merged = _self_connection_contract(_uvw_merged_self_connection())
    reference_path = tmp_path / "oam-self-split.json"
    reference_path.write_text(json.dumps(reference))
    merged_path = tmp_path / "oam-self-merged.json"
    merged_path.write_text(json.dumps(merged))
    args = _asymmetric_args()
    split = _reference(native, reference_path)
    merged = _reference(native, merged_path)
    split.compute_node_energies_forces(*args)
    merged.compute_node_energies_forces(*args)
    split_energy = np.asarray(split.node_energies, dtype=float)
    assert np.abs(split_energy).max() > 1e-6
    np.testing.assert_allclose(
        np.asarray(merged.node_energies, dtype=float),
        split_energy,
        rtol=0.0,
        atol=0.0,
    )


def test_self_tensor_product_rejects_a_re_indexing_decomposition(native, tmp_path):
    """A self connection that permutes the node row is refused, not silently used.

    "1x1o+1x0e+1x1o" holds the same components as the layer's
    "1x0e+1x1o+1x1o" but at different node-major positions, so every consumer of
    that row would read the wrong component.
    """

    from symmetrix.nequip_schema import validate_nequip_oam_data

    reordered = _uvw_self_connection(
        "1x1o+1x0e+1x1o",
        [
            _uvw(1, 0, 0, 1, 1, 1, (3, 1, 3)),
            _uvw(0, 0, 1, 1, 1, 1, (1, 1, 1)),
            _uvw(1, 0, 2, 1, 1, 1, (3, 1, 3)),
        ],
        [0.3, -0.4, 0.25],
    )
    contract = _self_connection_contract(reordered)
    # The exporter-side validator mirrors the rule by requiring the self
    # connection's own irreps text; the loader is what guards contracts that
    # never went through it.
    with pytest.raises(ValueError, match="self_connection"):
        validate_nequip_oam_data(contract)
    path = tmp_path / "oam-self-reordered.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="node-major element order"):
        native.NequIPOAMKokkos(str(path))


def test_gate_contract_rejects_non_scalar_gate_irreps(native, tmp_path):
    """The scalar gate path indexes the gate region sequentially, so a gate
    block that is not scalar-even cannot be represented and must be refused."""

    contract = _self_connection_contract(None)
    contract["layers"][0]["gate"] = {
        "type": "scalar_gate",
        "scalar_irreps": "0x0e",
        "gate_irreps": "1x1o",
        "gated_irreps": "1x0e+1x1o+1x1o",
    }
    contract["layers"][0]["node_irreps_pre_gate"] = "1x1o+1x0e+1x1o+1x1o"
    path = tmp_path / "oam-nonscalar-gate.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="scalar-even"):
        native.NequIPOAMKokkos(str(path))


def test_gate_contract_rejects_a_gate_channel_count_mismatch(native, tmp_path):
    """One gate channel per gated copy: the Kokkos gate would otherwise read
    past the gate region into the gated features."""

    contract = _self_connection_contract(None)
    contract["layers"][0]["gate"] = {
        "type": "scalar_gate",
        "scalar_irreps": "0x0e",
        "gate_irreps": "2x0e",
        "gated_irreps": "1x0e+1x1o+1x1o",
    }
    contract["layers"][0]["node_irreps_pre_gate"] = "2x0e+1x0e+1x1o+1x1o"
    path = tmp_path / "oam-gate-count.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="one gate channel per gated irrep copy"):
        native.NequIPOAMKokkos(str(path))


def test_artifact_is_shared_by_models_with_the_same_architecture(native, tmp_path):
    """Learned values do not prevent models from sharing generated code."""

    from symmetrix.nequip_schema import nequip_contract_fingerprint
    from symmetrix.oam_architecture import (
        oam_architecture_fingerprint,
        resolve_oam_architecture,
    )
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    contract = _self_connection_contract(_uvw_split_self_connection())
    other = json.loads(json.dumps(contract))
    for layer in other["layers"]:
        values = layer["pre_linear"]["weight"]["values"]
        layer["pre_linear"]["weight"]["values"] = [value + 0.125 for value in values]
    path = tmp_path / "oam-artifact-owner.json"
    path.write_text(json.dumps(contract))
    other_path = tmp_path / "oam-artifact-foreign.json"
    other_path.write_text(json.dumps(other))
    artifact = prepare_nequip_oam_receiver_host_artifact(
        path, "float64", cache_dir=tmp_path / "cache"
    )
    assert nequip_contract_fingerprint(contract) != nequip_contract_fingerprint(other)
    assert oam_architecture_fingerprint(
        resolve_oam_architecture(contract)
    ) == oam_architecture_fingerprint(resolve_oam_architecture(other))
    native.NequIPOAMKokkos(
        str(other_path),
        str(artifact.artifact_path),
        host_active_target=artifact.active_host_target,
        host_artifact_origin=artifact.artifact_origin,
    )
