"""End-to-end OAM parity for contracts with real angular content (l >= 2).

The checked-in minimal contract is scalar-only: with ``m=1, l=0`` everywhere the
lane-major and node-major layouts coincide, so the addressing class of defect
the streamed receiver exists to fix cannot show up in its parity tests.

This module builds a contract with genuine angular structure instead:

* blocks up to ``l=2``, including a *merged* ``2x2e`` block consumed by the
  layer's post-linear (the merged description is where a lane/node mix-up
  re-indexes a whole block);
* a multi-instruction ``uvw`` self connection whose widest instruction mixes
  ``1o x 2e -> 1o``;
* an edge tensor-product path that mixes ``2e x 1o -> 2e``.

The contract is validated by the published schema and then evaluated with both
the artifact-free reference routing and the generated receiver-owned direct
routing; energy and forces must agree.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

import oam_host_oracle

FIXTURE = Path(__file__).parent / "data" / "nequip" / "oam_s_v2_minimal.json"

NODE_IRREPS = "1x0e+1x1o+1x2e"
#: Node attributes are the per-type embedding, so their width must equal the
#: first layer's node irreps (the native loader enforces this) and the self
#: connection consumes them as its second operand.
ATTRIBUTE_IRREPS = NODE_IRREPS


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


def _uvu(i_in1, i_in2, i_out, wigner_shape, coefficient=0.25):
    return {
        "i_in1": i_in1,
        "i_in2": i_in2,
        "i_out": i_out,
        "connection_mode": "uvu",
        "has_weight": True,
        "path_shape": [1, 1],
        "path_weight": 1.0,
        "wigner_3j": {
            "shape": list(wigner_shape),
            "values": [coefficient] * int(np.prod(wigner_shape)),
        },
    }


def _uvw(i_in1, i_in2, i_out, wigner_shape):
    return {
        "i_in1": i_in1,
        "i_in2": i_in2,
        "i_out": i_out,
        "connection_mode": "uvw",
        "has_weight": True,
        "path_shape": [1, 1, 1],
        "path_weight": 1.0,
        "wigner_3j": {
            "shape": list(wigner_shape),
            "values": [
                0.1 * (index + 1) * (-1 if index % 3 else 1)
                for index in range(int(np.prod(wigner_shape)))
            ],
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
    self_connection,
    function_count,
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


def _self_connection(irreps_in1, irreps_in2, irreps_out, instructions):
    weight = [
        0.02 * (index + 1) * (-1 if index % 4 else 1)
        for index in range(len(instructions))
    ]
    return {
        "irreps_in1": irreps_in1,
        "irreps_in2": irreps_in2,
        "irreps_out": irreps_out,
        "instructions": instructions,
        "weight": {"shape": [len(weight)], "values": weight},
        "output_mask": {
            "shape": [_dimension(irreps_out)],
            "values": [1.0] * _dimension(irreps_out),
        },
    }


def _build_contract():
    """Two layers whose angular blocks reach l=2 and merge in the post-linear."""
    contract = json.loads(FIXTURE.read_text())
    counts = len(contract["model_atomic_numbers"])
    width = _dimension(NODE_IRREPS)
    contract["node_attributes"] = {
        "shape": [counts, width],
        "values": [
            0.05 * (((index + 3 * column) % 11) - 5)
            for index in range(counts)
            for column in range(width)
        ],
    }
    gate = {
        "type": "scalar_gate",
        "scalar_irreps": "1x0e",
        "gate_irreps": "0x0e",
        "gated_irreps": "1x1o+1x2e",
    }
    # The producing layer writes two adjacent 1x2e blocks; the consuming
    # post-linear declares them merged as one 2x2e block.
    producing = "1x0e+1x1o+1x2e+1x2e"
    merged = "1x0e+1x1o+2x2e"
    layer0 = _layer(
        "interaction_0",
        node_in=NODE_IRREPS,
        edge=NODE_IRREPS,
        tp_out=producing,
        pre_gate=NODE_IRREPS,
        node_out=NODE_IRREPS,
        gate=gate,
        pre_linear=_linear(
            NODE_IRREPS,
            NODE_IRREPS,
            [(0, 0, (1, 1)), (2, 2, (1, 1))],
            [0.8, -0.5],
            [0.1] * width,
            [1.0] * width,
        ),
        post_linear=_linear(
            merged,
            NODE_IRREPS,
            [(0, 0, (1, 1)), (1, 1, (1, 1)), (2, 2, (2, 1))],
            [0.5, 0.6, -0.4, 0.7],
            [0.1] * _dimension(NODE_IRREPS),
            [1.0] * _dimension(NODE_IRREPS),
        ),
        edge_instructions=[
            _uvu(0, 0, 0, (1, 1, 1)),
            _uvu(0, 1, 1, (1, 3, 3)),
            _uvu(0, 2, 2, (1, 5, 5)),
            _uvu(2, 0, 3, (5, 1, 5)),
        ],
        self_connection=_self_connection(
            NODE_IRREPS,
            ATTRIBUTE_IRREPS,
            NODE_IRREPS,
            [
                _uvw(0, 0, 0, (1, 1, 1)),
                _uvw(1, 1, 1, (3, 3, 3)),
                _uvw(2, 2, 2, (5, 5, 5)),
            ],
        ),
        function_count=4,
    )
    layer1 = _layer(
        "interaction_1",
        node_in=NODE_IRREPS,
        edge=NODE_IRREPS,
        tp_out=producing,
        pre_gate=NODE_IRREPS,
        node_out=NODE_IRREPS,
        gate=gate,
        pre_linear=_linear(
            NODE_IRREPS,
            NODE_IRREPS,
            [(0, 0, (1, 1)), (1, 1, (1, 1)), (2, 2, (1, 1))],
            [0.7, -0.6, 0.4],
            [0.0] * width,
            [1.0] * width,
        ),
        post_linear=_linear(
            merged,
            NODE_IRREPS,
            [(0, 0, (1, 1)), (1, 1, (1, 1)), (2, 2, (2, 1))],
            [0.9, -0.3, 0.55, 0.25],
            [0.0] * _dimension(NODE_IRREPS),
            [1.0] * _dimension(NODE_IRREPS),
        ),
        edge_instructions=[
            _uvu(0, 0, 0, (1, 1, 1)),
            _uvu(1, 0, 1, (3, 1, 3)),
            _uvu(2, 0, 2, (5, 1, 5)),
            _uvu(2, 1, 3, (5, 3, 5)),
        ],
        # l=1 x l=2 and l=2 x l=2 mix the widest angular pairs of the contract.
        self_connection=_self_connection(
            NODE_IRREPS,
            ATTRIBUTE_IRREPS,
            NODE_IRREPS,
            [
                _uvw(0, 0, 0, (1, 1, 1)),
                _uvw(1, 2, 1, (3, 5, 3)),
                _uvw(2, 2, 2, (5, 5, 5)),
            ],
        ),
        function_count=4,
    )
    contract["layers"] = [layer0, layer1]
    contract["num_layers"] = 2
    contract["readout"] = _linear(
        NODE_IRREPS, "1x0e", [(0, 0, (1, 1))], [1.0], [0.0], [1.0]
    )
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


def _require_cxx() -> None:
    if shutil.which(os.environ.get("CXX", "c++")) is None:
        pytest.skip("a C++ compiler is required for the generated OAM host plugin")


@pytest.fixture(scope="module")
def native():
    try:
        from symmetrix import symmetrix as native_symmetrix
    except Exception as error:  # pragma: no cover - environment without backend
        pytest.skip(f"native Symmetrix package is unavailable: {error}")
    if not hasattr(native_symmetrix, "NequIPOAMKokkos"):
        pytest.skip("NequIPOAMKokkos is unavailable in this extension")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    return native_symmetrix


def _reference(native, path):
    """The artifact-free reference routing: the portable host implementation.

    ``NequIPOAMKokkos(path)`` with no artifact argument is the portable host
    implementation, selected explicitly through reference mode.
    """

    oam_host_oracle.require_host_execution_space(
        native._kokkos_default_execution_space(),
        "the artifact-free OAM reference routing",
    )
    evaluator = native.NequIPOAMKokkos(str(path))
    evaluator.set_streamed_edges("reference")
    return evaluator


def _asymmetric_cluster():
    """Three nodes with l-mixing neighbours in chemically distinct types.

    A symmetric cell would leave the odd-angular components at zero and hide any
    l>=2 indexing defect, so the displacements are deliberately irregular and
    every node receives a different neighbour type.
    """
    node_types = np.asarray((0, 1, 2), dtype=np.int32)
    num_neigh = np.asarray((2, 0, 1), dtype=np.int32)
    indices = np.asarray((1, 2, 0), dtype=np.int32)
    neigh_types = np.asarray((1, 2, 0), dtype=np.int32)
    xyz = np.asarray((0.9, 0.3, -0.2, 1.4, -0.8, 0.5, -0.7, 0.6, 1.1), dtype=np.float64)
    distances = np.linalg.norm(xyz.reshape(-1, 3), axis=1)
    return (3, node_types, num_neigh, indices, neigh_types, xyz, distances)


def test_oam_l2_contract_direct_matches_the_reference_routing(native, tmp_path):
    from symmetrix.nequip_schema import validate_nequip_oam_data
    from symmetrix.oam_jit_artifact import prepare_nequip_oam_receiver_host_artifact

    _require_cxx()
    contract = _build_contract()
    validate_nequip_oam_data(contract)
    path = tmp_path / "oam-l2.json"
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

    graph = _asymmetric_cluster()
    reference.compute_node_energies_forces(*graph)
    kokkos.compute_node_energies_forces(*graph)

    reference_energy = np.asarray(reference.node_energies, dtype=float)
    kokkos_energy = np.asarray(kokkos.node_energies, dtype=float)
    reference_forces = np.asarray(reference.node_forces, dtype=float)
    kokkos_forces = np.asarray(kokkos.node_forces, dtype=float)
    # The contract must actually produce angular content; an all-zero contract
    # would make the parity assertions below vacuous.
    assert np.abs(reference_energy).max() > 1e-6
    assert np.abs(reference_forces).max() > 1e-6
    assert (
        "implementation=receiver_owned_full_message_oam_host_v7"
        in kokkos.last_execution_diagnostic
    )
    assert "host_plugin_abi=v7" in kokkos.last_execution_diagnostic
    np.testing.assert_allclose(kokkos_energy, reference_energy, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(kokkos_forces, reference_forces, rtol=1e-9, atol=1e-11)
