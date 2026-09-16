import copy
import ctypes
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import symmetrix.execution_mh1_contract as execution_mh1_contract
import symmetrix.mh1_jit_codegen as mh1_codegen
from symmetrix.execution_mh1_contract import (
    MH1_CONTRACT_KEY,
    MH1_CONTRACT_TAG,
    MH1_NODE_PROGRAM_TAG,
    MH1_V4_CONTRACT_TAG,
    make_execution_mh1_contract,
    make_execution_mh1_v4_contract,
    maybe_make_execution_mh1_contract,
    maybe_make_execution_mh1_v4_contract,
    normalize_execution_mh1_contract,
    normalize_execution_mh1_v4_contract,
    validate_execution_mh1_contract_for_model,
    validate_execution_mh1_v4_contract_for_model,
)
from symmetrix.mh1_jit_codegen import (
    render_jit_mh1_host_plugin_v4,
    render_jit_mh1_host_plugin_v5,
    render_execution_mh1_node_linear_cpp_helpers,
    render_execution_mh1_node_nonlinear_cpp_helpers,
    render_execution_mh1_node_product_cpp_helpers,
    render_execution_mh1_node_program_cpp_helpers,
    execution_mh1_node_program_metadata,
)


def _tensor(shape, fill=0.0):
    size = 1
    for dimension in shape:
        size *= dimension
    return {"shape": list(shape), "values": [fill] * size}


def _linear(irreps_out, weight_size=1):
    return {
        "irreps_in": "2x0e",
        "irreps_out": irreps_out,
        "instructions": [],
        "weight": _tensor([weight_size], 0.25),
        "bias": _tensor([0]),
        "output_mask": _tensor([1], 1.0),
    }


def _mlp(output_dimension, hidden_dimension=5):
    return {
        "layers": [
            {
                "type": "linear",
                "weight": _tensor([hidden_dimension, 9], 0.1),
                "bias": _tensor([hidden_dimension], 0.2),
            },
            {
                "type": "layer_norm",
                "normalized_shape": [hidden_dimension],
                "eps": 1.0e-5,
                "weight": _tensor([hidden_dimension], 1.0),
                "bias": _tensor([hidden_dimension], 0.0),
            },
            {"type": "silu"},
            {
                "type": "linear",
                "weight": _tensor([output_dimension, hidden_dimension], 0.3),
                "bias": _tensor([output_dimension], 0.4),
            },
        ]
    }


def _single_linear_density():
    return {
        "layers": [
            {
                "type": "linear",
                "weight": _tensor([1, 9], 0.1),
                "bias": _tensor([1], 0.2),
            }
        ]
    }


def _deep_density():
    return {
        "layers": [
            {
                "type": "linear",
                "weight": _tensor([7, 9], 0.1),
                "bias": _tensor([7], 0.2),
            },
            {"type": "silu"},
            {
                "type": "linear",
                "weight": _tensor([3, 7], 0.3),
                "bias": _tensor([3], 0.4),
            },
            {
                "type": "layer_norm",
                "normalized_shape": [3],
                "eps": 2.0e-5,
                "weight": _tensor([3], 1.0),
                "bias": _tensor([3], 0.0),
            },
            {"type": "silu"},
            {
                "type": "linear",
                "weight": _tensor([1, 3], 0.5),
                "bias": _tensor([1], 0.6),
            },
        ]
    }


def _interaction(index):
    if index == 0:
        irreps_in1 = "2x0e"
        irreps_out = "2x0e+2x1o"
        instructions = [
            {
                "i_in1": 0,
                "i_in2": 0,
                "i_out": 0,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
                "path_shape": [2, 1],
                "wigner_3j": _tensor([1, 1, 1], 1.0),
            },
            {
                "i_in1": 0,
                "i_in2": 1,
                "i_out": 1,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 3.0**0.5,
                "path_shape": [2, 1],
                "wigner_3j": {
                    "shape": [1, 3, 3],
                    "values": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                },
            },
        ]
    else:
        irreps_in1 = "2x0e+2x1o"
        irreps_out = "2x1e"
        instructions = [
            {
                "i_in1": 1,
                "i_in2": 1,
                "i_out": 0,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
                "path_shape": [2, 1],
                "wigner_3j": {
                    "shape": [3, 3, 3],
                    "values": [1.0] + [0.0] * 26,
                },
            }
        ]
    output_dimension = 4 if index == 0 else 2
    output_components = 8 if index == 0 else 6
    return {
        "class": "RealAgnosticResidualNonLinearInteractionBlock",
        "source_embedding": _linear("3x0e", 6),
        "target_embedding": _linear("4x0e", 8),
        "conv_tp": {
            "irreps_in1": irreps_in1,
            "irreps_in2": "1x0e+1x1o",
            "irreps_out": irreps_out,
            "instructions": instructions,
            "weight": _tensor([0]),
            "output_mask": _tensor([output_components], 1.0),
        },
        "conv_tp_weights": _mlp(output_dimension),
        "density_fn": _mlp(1, hidden_dimension=6 + index),
    }


def _e3_linear(irreps_in, irreps_out, instruction_pairs):
    _, input_blocks = execution_mh1_contract._irrep_blocks(irreps_in, "input")
    _, output_blocks = execution_mh1_contract._irrep_blocks(irreps_out, "output")
    instructions = []
    weight_count = 0
    for input_index, output_index in instruction_pairs:
        path_shape = [
            input_blocks[input_index]["multiplicity"],
            output_blocks[output_index]["multiplicity"],
        ]
        weight_count += path_shape[0] * path_shape[1]
        instructions.append(
            {
                "i_in": input_index,
                "i_out": output_index,
                "path_weight": 1.0,
                "path_shape": path_shape,
            }
        )
    output_dimension = sum(
        block["multiplicity"] * block["components"] for block in output_blocks
    )
    return {
        "irreps_in": irreps_in,
        "irreps_out": irreps_out,
        "instructions": instructions,
        "weight": _tensor([weight_count], 0.25),
        "bias": _tensor([0]),
        "output_mask": _tensor([output_dimension], 1.0),
    }


def _mh1_node_tensor_product(index):
    irreps_in1 = "1x0e" if index == 0 else "1x0e+1x1o"
    irreps_in2 = "1x0e+1x1o+1x2e+1x3o"
    irreps_out = irreps_in2
    instructions = []
    for block_index, width in enumerate((1, 3, 5, 7)):
        wigner_values = [0.0] * (width * width)
        for component in range(width):
            wigner_values[component * width + component] = 1.0
        instructions.append(
            {
                "i_in1": 0,
                "i_in2": block_index,
                "i_out": block_index,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": float(width) ** 0.5,
                "path_shape": [1, 1],
                "wigner_3j": {
                    "shape": [1, width, width],
                    "values": wigner_values,
                },
            }
        )
    return {
        "irreps_in1": irreps_in1,
        "irreps_in2": irreps_in2,
        "irreps_out": irreps_out,
        "instructions": instructions,
        "weight": _tensor([0]),
        "output_mask": _tensor([16], 1.0),
    }


def _mh1_gate():
    return {
        "irreps_in": "4x0e+1x1o+1x2e+1x3o",
        "irreps_out": "1x0e+1x1o+1x2e+1x3o",
        "irreps_scalars": "1x0e",
        "irreps_gates": "3x0e",
        "irreps_gated": "1x1o+1x2e+1x3o",
        "scalar_activation": "silu",
        "scalar_activation_constants": [1.6791767923989418],
        "gate_activation": "sigmoid",
        "gate_activation_constants": [1.8467055342154763] * 3,
    }


def _mh1_contraction(output_l):
    width = 2 * output_l + 1
    u_tensors = []
    for degree in range(1, 4):
        shape = ([width] if output_l else []) + [16] * degree + [1]
        values = [0.0] * math.prod(shape)
        values[0] = 1.0
        u_tensors.append({"shape": shape, "values": values})
    return {
        "correlation": 3,
        "weights": [_tensor([1, 1, 1], 0.5), _tensor([1, 1, 1], 0.5)],
        "weights_max": _tensor([1, 1, 1], 0.5),
        "u_tensors": u_tensors,
    }


def _mh1_product(index):
    irreps_out = "1x0e+1x1o" if index == 0 else "1x0e"
    instruction_pairs = [(0, 0)] + ([(1, 1)] if index == 0 else [])
    return {
        "node_feats_irreps": irreps_out,
        "target_irreps": irreps_out,
        "use_sc": True,
        "use_agnostic_product": True,
        "linear": _e3_linear(irreps_out, irreps_out, instruction_pairs),
        "symmetric_contractions": {
            "irreps_in": "1x0e+1x1o+1x2e+1x3o",
            "irreps_out": irreps_out,
            "contractions": [_mh1_contraction(0)]
            + ([_mh1_contraction(1)] if index == 0 else []),
        },
    }


def _mh1_node_interaction(index):
    input_irreps = "1x0e" if index == 0 else "1x0e+1x1o"
    up_irreps = input_irreps
    gate_input = "4x0e+1x1o+1x2e+1x3o"
    product_input = "1x0e+1x1o+1x2e+1x3o"
    product_output = "1x0e+1x1o" if index == 0 else "1x0e"
    interaction = _interaction(index)
    interaction["conv_tp"] = _mh1_node_tensor_product(index)
    interaction["conv_tp_weights"] = _mlp(4)
    interaction.update(
        {
            "linear_up": _e3_linear(
                input_irreps,
                up_irreps,
                [(block, block) for block in range(index + 1)],
            ),
            "linear_res": _e3_linear(
                up_irreps,
                gate_input,
                [(block, block) for block in range(index + 1)],
            ),
            "skip_tp": _e3_linear(
                input_irreps,
                product_output,
                [(0, 0)],
            ),
            "linear_1": _e3_linear(
                product_input,
                gate_input,
                [(block, block) for block in range(4)],
            ),
            "gate": _mh1_gate(),
            "linear_2": _e3_linear(
                product_input,
                product_input,
                [(block, block) for block in range(4)],
            ),
            "alpha": 2.5 + index,
            "beta": 0.1 + index,
        }
    )
    return interaction


def _mh1_readout(index):
    if index == 0:
        return {
            "class": "LinearReadoutBlock",
            "linear": _e3_linear("1x0e+1x1o", "1x0e", [(0, 0)]),
            "linear_1": None,
            "linear_2": None,
            "activation": None,
            "activation_constants": None,
        }
    return {
        "class": "NonLinearReadoutBlock",
        "linear": None,
        "linear_1": _e3_linear("1x0e", "2x0e", [(0, 0)]),
        "linear_2": _e3_linear("2x0e", "1x0e", [(0, 0)]),
        "activation": "silu",
        "activation_constants": [1.6791767923989418],
    }


@pytest.fixture
def mh1_definition():
    return {
        "model_type": "MACE_Nonlinear",
        "radial_embedding": {
            "apply_cutoff": False,
            "basis": {"type": "bessel", "weights": _tensor([2], 0.5)},
            "cutoff": {"type": "polynomial", "r_max": 6.0, "p": 5},
            "distance_transform": {"type": "agnesi", "a": 1.1, "q": 0.9, "p": 4.5},
        },
        "interactions": [_interaction(0), _interaction(1)],
    }


@pytest.fixture
def mh1_node_definition(mh1_definition):
    definition = copy.deepcopy(mh1_definition)
    definition["node_embedding"] = _e3_linear("2x0e", "1x0e", [(0, 0)])
    definition["interactions"] = [
        _mh1_node_interaction(0),
        _mh1_node_interaction(1),
    ]
    definition["products"] = [_mh1_product(0), _mh1_product(1)]
    definition["readouts"] = [_mh1_readout(0), _mh1_readout(1)]
    return definition


def test_mh1_contract_is_parameter_free_canonical_and_independent(mh1_definition):
    contract = make_execution_mh1_contract(mh1_definition)

    assert MH1_CONTRACT_KEY == "MH1_UVU"
    assert contract["tag"] == MH1_CONTRACT_TAG == "symmetrix.execution.mh1_uvu/3"
    assert contract["interaction_count"] == 2
    assert [item["index"] for item in contract["interactions"]] == [0, 1]
    assert contract["interactions"][0]["dimensions"]["weight"] == 4
    assert contract["interactions"][1]["dimensions"]["weight"] == 2
    assert contract["interactions"][0]["dimensions"]["phi"] == 5
    assert contract["interactions"][0]["dimensions"]["density_prefix"] == 6
    assert contract["interactions"][1]["dimensions"]["density_prefix"] == 7
    assert contract["interactions"][0]["dimensions"]["density_output"] == 1
    assert contract["interactions"][1]["dimensions"]["input_1_angular"] == 4
    density = contract["interactions"][0]["density_mlp"]
    assert density["conditioned_input_dimension"] == 9
    assert density["dynamic_input_dimension"] == 2
    assert density["source_condition_dimension"] == 3
    assert density["target_condition_dimension"] == 4
    assert density["factorization"] == {
        "kind": "conditioned_prefix_final_affine",
        "prefix_layer_count": 3,
        "prefix_is_identity": False,
        "prefix_output_dimension": 6,
        "final_affine_input_dimension": 6,
        "final_affine_output_dimension": 1,
        "conditioned_columns_folded_into_first_affine": True,
    }
    assert contract["interactions"][0]["paths"][1]["sparse_wigner"]["term_count"] == 3
    assert contract["derivatives"]["coordinate"] == {
        "radial": True,
        "harmonics": True,
        "cutoff": True,
    }
    assert contract["derivatives"]["parameter_gradients"] is False
    assert contract["layouts"]["source_values"]["feature_layout"] == "ir_mul"
    assert contract["layouts"]["messages"]["feature_layout"] == "ir_mul"
    assert contract["layouts"]["runtime_node_feature_layout"] == "mul_ir"
    assert contract["layouts"]["generated_node_feature_layout"] == "ir_mul"
    assert contract["semantic_fingerprint"].startswith("sha256:")
    assert normalize_execution_mh1_contract(contract) == contract
    assert (
        contract["interactions"][0]["generation_fingerprint"]
        != (contract["interactions"][1]["generation_fingerprint"])
    )


def test_mh1_generation_identity_excludes_learned_values(mh1_definition):
    original = make_execution_mh1_contract(mh1_definition)
    changed = copy.deepcopy(mh1_definition)
    for interaction in changed["interactions"]:
        for name in ("source_embedding", "target_embedding"):
            interaction[name]["weight"]["values"] = [
                value + 7.0 for value in interaction[name]["weight"]["values"]
            ]
        for module_name in ("conv_tp_weights", "density_fn"):
            for layer in interaction[module_name]["layers"]:
                for tensor_name in ("weight", "bias"):
                    if tensor_name in layer:
                        layer[tensor_name]["values"] = [
                            value - 3.0 for value in layer[tensor_name]["values"]
                        ]

    changed_contract = make_execution_mh1_contract(changed)
    assert changed_contract == original


def test_mh1_generation_identity_captures_wigner_mlp_and_cutoff(mh1_definition):
    original = make_execution_mh1_contract(mh1_definition)

    changed_wigner = copy.deepcopy(mh1_definition)
    changed_wigner["interactions"][1]["conv_tp"]["instructions"][0]["wigner_3j"][
        "values"
    ][0] = 0.5
    assert (
        make_execution_mh1_contract(changed_wigner)["generation_fingerprint"]
        != (original["generation_fingerprint"])
    )

    changed_mlp = copy.deepcopy(mh1_definition)
    changed_mlp["interactions"][0]["conv_tp_weights"]["layers"][1]["eps"] = 2e-5
    assert (
        make_execution_mh1_contract(changed_mlp)["generation_fingerprint"]
        != (original["generation_fingerprint"])
    )

    changed_density = copy.deepcopy(mh1_definition)
    changed_density["interactions"][1]["density_fn"]["layers"][1]["eps"] = 3e-5
    changed_density_contract = make_execution_mh1_contract(changed_density)
    assert (
        changed_density_contract["generation_fingerprint"]
        != original["generation_fingerprint"]
    )
    assert (
        changed_density_contract["structure_fingerprint"]
        == original["structure_fingerprint"]
    )

    changed_density_topology = copy.deepcopy(mh1_definition)
    changed_density_topology["interactions"][1]["density_fn"]["layers"][1] = {
        "type": "silu"
    }
    changed_density_topology_contract = make_execution_mh1_contract(
        changed_density_topology
    )
    assert (
        changed_density_topology_contract["structure_fingerprint"]
        != original["structure_fingerprint"]
    )

    changed_cutoff = copy.deepcopy(mh1_definition)
    changed_cutoff["radial_embedding"]["cutoff"]["p"] = 6
    assert (
        make_execution_mh1_contract(changed_cutoff)["generation_fingerprint"]
        != (original["generation_fingerprint"])
    )

    changed_dimensions = copy.deepcopy(mh1_definition)
    changed_dimensions["radial_embedding"]["basis"]["weights"]["shape"] = [3]
    for interaction in changed_dimensions["interactions"]:
        interaction["conv_tp_weights"]["layers"][0]["weight"]["shape"][1] = 10
        interaction["density_fn"]["layers"][0]["weight"]["shape"][1] = 10
    assert (
        make_execution_mh1_contract(changed_dimensions)["generation_fingerprint"]
        != original["generation_fingerprint"]
    )


@pytest.mark.parametrize(
    ("interaction_index", "density_definition", "expected_prefix", "identity"),
    [
        (0, _single_linear_density(), 2, True),
        (1, _deep_density(), 3, False),
    ],
)
def test_mh1_density_contract_preserves_generalized_affine_mlp_shapes(
    mh1_definition,
    interaction_index,
    density_definition,
    expected_prefix,
    identity,
):
    changed = copy.deepcopy(mh1_definition)
    changed["interactions"][interaction_index]["density_fn"] = density_definition

    contract = make_execution_mh1_contract(changed)
    descriptor = contract["interactions"][interaction_index]
    density = descriptor["density_mlp"]
    assert descriptor["dimensions"]["density_prefix"] == expected_prefix
    assert descriptor["dimensions"]["density_output"] == 1
    assert density["factorization"]["prefix_is_identity"] is identity
    assert density["factorization"]["prefix_output_dimension"] == expected_prefix
    assert density["factorization"]["final_affine_output_dimension"] == 1
    assert [layer["type"] for layer in density["layers"]] == [
        layer["type"] for layer in density_definition["layers"]
    ]
    assert normalize_execution_mh1_contract(contract) == contract


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", r"interactions\[0\]\.density_fn must be an affine MLP"),
        ("conditioned_input", "conditioned input dimension is inconsistent"),
        ("non_scalar_output", "output dimension is inconsistent"),
        ("unsupported_layer", "has an unsupported type"),
    ],
)
def test_mh1_density_contract_rejects_unsupported_shapes(
    mh1_definition, mutation, message
):
    changed = copy.deepcopy(mh1_definition)
    density = changed["interactions"][0]["density_fn"]
    if mutation == "missing":
        changed["interactions"][0].pop("density_fn")
    elif mutation == "conditioned_input":
        density["layers"][0]["weight"]["shape"][1] += 1
    elif mutation == "non_scalar_output":
        density["layers"][-1]["weight"]["shape"][0] = 2
        density["layers"][-1]["bias"]["shape"][0] = 2
    else:
        density["layers"][2]["type"] = "tanh"

    with pytest.raises(ValueError, match=message):
        make_execution_mh1_contract(changed)
    assert maybe_make_execution_mh1_contract(changed) is None


def test_mh1_legacy_contract_is_model_bound_and_upgraded(mh1_definition):
    current = make_execution_mh1_contract(mh1_definition)
    legacy = execution_mh1_contract._legacy_contract_projection(current)

    assert legacy["tag"] == "symmetrix.execution.mh1_uvu/1"
    assert legacy["version"] == 1
    assert "density_mlp" not in legacy["interactions"][0]
    assert legacy["generation_fingerprint"] == (
        "sha256:21f39b511edb7b690dacefd633426848cd4d7a2881eea900ceb0fbc6cf202bc3"
    )
    with pytest.raises(ValueError, match=r"mh1_uvu/3"):
        normalize_execution_mh1_contract(legacy)
    assert validate_execution_mh1_contract_for_model(mh1_definition, legacy) == current

    changed_model = copy.deepcopy(mh1_definition)
    changed_model["interactions"][1]["conv_tp"]["instructions"][0]["path_weight"] *= (
        1.25
    )
    with pytest.raises(ValueError, match="does not match the model definition"):
        validate_execution_mh1_contract_for_model(changed_model, legacy)

    stale = copy.deepcopy(legacy)
    stale["interactions"][0]["dimensions"]["channels"] += 1
    with pytest.raises(ValueError, match="interaction 0 fingerprint is stale"):
        validate_execution_mh1_contract_for_model(mh1_definition, stale)


def test_mh1_v2_contract_is_model_bound_and_upgraded(mh1_definition):
    current = make_execution_mh1_contract(mh1_definition)
    version_2 = execution_mh1_contract._v2_contract_projection(current)

    assert version_2["tag"] == "symmetrix.execution.mh1_uvu/2"
    assert version_2["version"] == 2
    assert version_2["layouts"]["source_values"] == "node,input_1_component"
    assert (
        validate_execution_mh1_contract_for_model(mh1_definition, version_2) == current
    )

    changed_model = copy.deepcopy(mh1_definition)
    changed_model["interactions"][0]["density_fn"]["layers"][1] = {"type": "silu"}
    with pytest.raises(ValueError, match="does not match the model definition"):
        validate_execution_mh1_contract_for_model(changed_model, version_2)

    stale = copy.deepcopy(version_2)
    stale["layouts"]["source_values"] = "node,ir_mul_component"
    with pytest.raises(ValueError, match="layouts do not match"):
        validate_execution_mh1_contract_for_model(mh1_definition, stale)


def test_mh1_contract_rejects_stale_fingerprint(mh1_definition):
    contract = make_execution_mh1_contract(mh1_definition)
    contract["interactions"][0]["dimensions"]["channels"] += 1
    with pytest.raises(ValueError, match="interaction 0 fingerprint is stale"):
        normalize_execution_mh1_contract(contract)

    contract = make_execution_mh1_contract(mh1_definition)
    contract["semantic_fingerprint"] = "sha256:stale"
    with pytest.raises(ValueError, match="semantic fingerprint is stale"):
        normalize_execution_mh1_contract(contract)


def test_mh1_contract_rejects_stale_model_binding(mh1_definition):
    contract = make_execution_mh1_contract(mh1_definition)
    assert (
        validate_execution_mh1_contract_for_model(mh1_definition, contract) == contract
    )

    changed = copy.deepcopy(mh1_definition)
    changed["interactions"][1]["conv_tp"]["instructions"][0]["path_weight"] *= 1.25
    with pytest.raises(ValueError, match="does not match the model definition"):
        validate_execution_mh1_contract_for_model(changed, contract)


def test_optional_mh1_contract_omits_incompatible_models(mh1_definition):
    one_layer = copy.deepcopy(mh1_definition)
    one_layer["interactions"].pop()
    assert maybe_make_execution_mh1_contract(one_layer) is None

    untied = copy.deepcopy(mh1_definition)
    untied["interactions"][0]["conv_tp"]["instructions"][0]["connection_mode"] = "uvw"
    assert maybe_make_execution_mh1_contract(untied) is None


def test_mh1_contract_rejects_output_mask_inconsistent_with_paths(mh1_definition):
    invalid = copy.deepcopy(mh1_definition)
    invalid["interactions"][0]["conv_tp"]["output_mask"]["values"][0] = 0.0

    with pytest.raises(ValueError, match="output mask does not match"):
        make_execution_mh1_contract(invalid)


def test_mh1_v4_contract_captures_complete_node_program(mh1_node_definition):
    v3 = make_execution_mh1_contract(mh1_node_definition)
    contract = make_execution_mh1_v4_contract(mh1_node_definition)

    assert contract["tag"] == MH1_V4_CONTRACT_TAG
    assert contract["version"] == 4
    assert contract["node_program"]["tag"] == MH1_NODE_PROGRAM_TAG
    assert contract["node_program"]["feature_layout"] == "ir_mul"
    assert (
        contract["runtime_layout_fingerprint"]
        == contract["node_program"]["runtime_layout"]["fingerprint"]
    )
    assert normalize_execution_mh1_v4_contract(contract) == contract
    assert execution_mh1_contract._v3_contract_projection(contract) == v3

    layers = contract["node_program"]["layers"]
    assert layers[0]["derivatives"]["requires_tp_source_state_adjoint"] is False
    assert layers[1]["derivatives"]["requires_tp_source_state_adjoint"] is True
    assert layers[0]["persistent_state"]["retain"] == [
        "up",
        "messages",
        "density",
        "output",
    ]
    assert layers[1]["persistent_state"]["retain"] == [
        "up",
        "messages",
        "density",
    ]
    for layer in layers:
        assert layer["persistent_state"]["node_state_policy"] == {
            "default": "full-retention-v1",
            "supported": [
                "full-retention-v1",
                "recompute-v1",
                "reuse-adjoints-v1",
                "retain-interaction-v1",
            ],
            "retained_tensors": {
                "pre_gate": layer["linears"]["linear_res"]["dimensions"]["output"],
                "interaction_output": layer["linears"]["linear_2"]["dimensions"][
                    "output"
                ],
            },
        }
    assert layers[0]["product"]["feature_layout"] == "ir_mul"
    assert layers[0]["product"]["angular_layout"][1] == {
        "angular_index": 1,
        "component_offset": 1,
        "channel_stride": 1,
    }
    for layer in layers:
        assert set(layer["runtime_parameter_packs"]) == {
            "linear",
            "product",
            "readout",
        }
        assert all(
            pack["parameter_count"] > 0 and pack["fingerprint"].startswith("sha256:")
            for pack in layer["runtime_parameter_packs"].values()
        )


def test_mh1_v4_generation_identity_excludes_all_learned_values(
    mh1_node_definition,
):
    original = make_execution_mh1_v4_contract(mh1_node_definition)
    changed = copy.deepcopy(mh1_node_definition)
    changed["node_embedding"]["weight"]["values"][0] += 11.0
    for interaction in changed["interactions"]:
        interaction["alpha"] += 3.0
        interaction["beta"] -= 2.0
        for name in (
            "linear_up",
            "linear_res",
            "skip_tp",
            "linear_1",
            "linear_2",
        ):
            interaction[name]["weight"]["values"][0] += 5.0
    for product in changed["products"]:
        product["linear"]["weight"]["values"][0] -= 7.0
        for contraction in product["symmetric_contractions"]["contractions"]:
            contraction["weights_max"]["values"][0] += 2.0
            for weights in contraction["weights"]:
                weights["values"][0] -= 4.0
    changed["readouts"][0]["linear"]["weight"]["values"][0] += 9.0
    changed["readouts"][1]["linear_1"]["weight"]["values"][0] += 9.0
    changed["readouts"][1]["linear_2"]["weight"]["values"][0] += 9.0

    assert make_execution_mh1_v4_contract(changed) == original


def test_mh1_v4_contract_rejects_stale_node_layout(mh1_node_definition):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    contract["node_program"]["layers"][0]["runtime_parameter_packs"]["linear"][
        "parameter_count"
    ] += 1
    with pytest.raises(ValueError, match="parameter count is inconsistent"):
        normalize_execution_mh1_v4_contract(contract)

    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    contract["runtime_layout_fingerprint"] = "sha256:stale"
    with pytest.raises(ValueError, match="runtime layout fingerprint is stale"):
        normalize_execution_mh1_v4_contract(contract)

    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    contract["node_program"]["layers"][0]["persistent_state"]["node_state_policy"][
        "default"
    ] = "recompute-v1"
    with pytest.raises(ValueError, match="retention policy is unsupported"):
        normalize_execution_mh1_v4_contract(contract)


def test_mh1_v4_contract_upgrades_exact_v3_identity(mh1_node_definition):
    expected = make_execution_mh1_v4_contract(mh1_node_definition)
    v3 = make_execution_mh1_contract(mh1_node_definition)

    assert (
        validate_execution_mh1_v4_contract_for_model(mh1_node_definition, v3)
        == expected
    )
    assert (
        validate_execution_mh1_v4_contract_for_model(mh1_node_definition, expected)
        == expected
    )
    assert maybe_make_execution_mh1_v4_contract(mh1_node_definition) == expected

    incomplete = copy.deepcopy(mh1_node_definition)
    incomplete.pop("products")
    assert maybe_make_execution_mh1_v4_contract(incomplete) is None


def test_mh1_v4_contract_upgrades_only_exact_pre_retention_identity(
    mh1_node_definition,
):
    expected = make_execution_mh1_v4_contract(mh1_node_definition)
    legacy = copy.deepcopy(expected)
    for name in (
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
    ):
        legacy.pop(name)
    for layer in legacy["node_program"]["layers"]:
        layer["persistent_state"].pop("node_state_policy")
    structure = execution_mh1_contract._node_program_structure_payload(
        legacy["node_program"]
    )
    for layer in structure["layers"]:
        layer.pop("persistent_state")
    legacy["node_program"]["structure_fingerprint"] = (
        execution_mh1_contract._fingerprint(structure)
    )
    legacy = execution_mh1_contract._finish_contract_fingerprints(legacy)

    assert (
        validate_execution_mh1_v4_contract_for_model(mh1_node_definition, legacy)
        == expected
    )

    stale = copy.deepcopy(legacy)
    stale["generation_fingerprint"] = "sha256:stale"
    with pytest.raises(ValueError):
        validate_execution_mh1_v4_contract_for_model(mh1_node_definition, stale)

    partially_stripped = copy.deepcopy(expected)
    partially_stripped["node_program"]["layers"][0]["persistent_state"].pop(
        "node_state_policy"
    )
    with pytest.raises(ValueError):
        validate_execution_mh1_v4_contract_for_model(
            mh1_node_definition, partially_stripped
        )

    model_mismatch = copy.deepcopy(legacy)
    for name in (
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
    ):
        model_mismatch.pop(name)
    model_mismatch["radial"]["cutoff"]["r_max"] += 1.0
    model_mismatch = execution_mh1_contract._finish_contract_fingerprints(
        model_mismatch
    )
    with pytest.raises(ValueError):
        validate_execution_mh1_v4_contract_for_model(
            mh1_node_definition, model_mismatch
        )


def test_mh1_v4_node_metadata_bounds_arena_and_disables_dead_reverse(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    host = execution_mh1_node_program_metadata(contract, backend="host")
    cuda = execution_mh1_node_program_metadata(contract, backend="cuda")

    assert host["artifact_id"].startswith("jit-mh1-gen11-v4-")
    assert host["runtime_layout_fingerprint"] == contract["runtime_layout_fingerprint"]
    assert host["element_indexing"] == "model_element_index_per_node"
    assert [layer["node_arena_dimension"] for layer in host["layers"]] == [
        36,
        35,
    ]
    assert (
        host["layers"][0]["arena"]["candidates"][
            "reverse_interaction_contracted_adjoint_and_interaction_adjoint"
        ]
        == 36
    )
    assert host["layers"][0]["reverse_phases"] == [
        {"phase": 0, "owners_per_node": 1, "enabled": True},
        {"phase": 1, "owners_per_node": 0, "enabled": False},
    ]
    assert host["layers"][1]["reverse_phases"][1]["enabled"] is True
    assert cuda["layers"][0]["reverse_phases"] == [
        {"phase": 0, "threads_per_block": 128, "enabled": True},
        {"phase": 1, "threads_per_block": 0, "enabled": False},
    ]
    assert cuda["layers"][1]["forward_phases"][0]["threads_per_block"] == 128
    assert host["node_state_policy"] == "recompute-v1"
    assert cuda["node_state_policy"] == "full-retention-v1"
    assert all(layer["retained_pre_gate_dimension"] == 0 for layer in host["layers"])
    assert all(
        layer["retained_pre_gate_dimension"] == layer["residual_dimension"]
        and layer["retained_interaction_output_dimension"]
        == layer["interaction_output_dimension"]
        for layer in cuda["layers"]
    )

    with pytest.raises(ValueError, match="backend must be"):
        execution_mh1_node_program_metadata(contract, backend="serial")


def test_mh1_v4_cuda_node_program_uses_cooperative_and_tiled_node_blocks(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    source = execution_mh1_contract.project_execution_mh1_v3_contract(contract)
    forward_policy = mh1_codegen.resolve_execution_mh1_cuda_forward_policy(
        source, 80, "auto"
    )
    source_policy = mh1_codegen.resolve_execution_mh1_cuda_source_policy(
        source, 80, "auto"
    )
    edge_policy = mh1_codegen.resolve_execution_mh1_cuda_edge_policy(source, 80, "auto")
    generated = mh1_codegen.render_jit_mh1_cuda_plugin_v4(
        contract,
        80,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
    )

    assert "symmetrix_jit_mh1_cuda_plugin_query_v3" in generated
    assert "symmetrix_jit_mh1_cuda_plugin_query_v4" in generated
    assert "node_pre_forward_owner_" not in generated
    assert "for (std::int64_t node = blockIdx.x; node < args.num_nodes;" in generated
    assert "__shared__ float l0_density_reduction[128]" in generated
    assert "threadIdx.x;" in generated
    assert "__syncthreads();" in generated
    assert "args->num_nodes < persistent_blocks" in generated
    assert "node_tiled_l0_forward_linear2_block_0" in generated
    assert "constexpr int tile_nodes = 8;" in generated
    assert "constexpr int tile_nodes = 32;" in generated
    assert "constexpr int tile_channels = 32;" in generated
    assert "weight_tile[tile_channels][tile_channels + 1]" in generated
    assert "(args->num_nodes + 7) / 8" in generated
    assert "(args->num_nodes + 31) / 32" in generated
    assert "float value_1 =" in generated
    assert "float value_3 =" in generated
    assert "input_tile[node_lane + 8][source]" in generated
    assert "input_tile[node_lane + 24][source]" in generated
    assert "node_tiled_l0_forward_message_block_0" in generated
    assert "node_tiled_l1_reverse_message_block_0" in generated
    assert "args.node_density[node_0]" in generated
    assert "args.node_density[node_1]" in generated
    assert "args.node_density[{node}]" not in generated
    assert "retained_pre_gate[column] = arena[column]" in generated
    assert "retained_interaction[column] = interaction[column]" in generated
    assert "const float* interaction = args.retained_interaction_output" in generated
    assert "args.retained_pre_gate + node" in generated
    assert "args->retained_pre_gate == nullptr" in generated
    assert "args->retained_interaction_output == nullptr" in generated

    recompute = mh1_codegen.render_jit_mh1_cuda_plugin_v4(
        contract,
        80,
        forward_policy=forward_policy,
        source_policy=source_policy,
        edge_policy=edge_policy,
        node_state_policy="recompute-v1",
    )
    assert "retained_pre_gate[column] = arena[column]" not in recompute
    assert "retained_interaction[column] = interaction[column]" not in recompute
    assert "args.retained_pre_gate + node" not in recompute
    assert "args.retained_interaction_output + node" not in recompute
    assert "args->retained_pre_gate == nullptr" not in recompute
    assert "args->retained_interaction_output == nullptr" not in recompute
    assert "node_tiled_l0_reverse_gate_forward" in recompute


def test_mh1_v4_nvrtc_module_is_device_only_with_closed_launch_plan(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    module = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)

    assert "#include" not in module
    assert "<<<" not in module
    assert "reinterpret_cast<cudaStream_t>" not in module
    assert "symmetrix_jit_mh1_cuda_plugin_query_v4(void)" not in module
    assert (
        'extern "C" __global__ void symmetrix_execution_mh1_forward_kernel_0' in module
    )
    assert "symmetrix_execution_mh1_module_identity" in module
    assert plan["module_identity"] in module
    assert plan["generation_fingerprint"] == contract["generation_fingerprint"]
    assert plan["runtime_layout_fingerprint"] == contract["runtime_layout_fingerprint"]
    assert plan["schema"] == mh1_codegen.MH1_CUDA_LAUNCH_PLAN_TAG
    assert [item["index"] for item in plan["interactions"]] == [0, 1]
    assert plan["nodes"][0]["pre_reverse"] == []
    assert plan["nodes"][1]["pre_reverse"][0]["grid"] == "node_channel_tiles"
    assert plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert "constexpr int tile_nodes = 4;" in module
    assert "const bool node_valid_3 = node_0 + 3 < args.num_nodes;" in module
    assert "node_valid_3 ? 3 : 0" in module
    assert "if (node_valid_3)" in module
    assert "input_adjoint_3" in module
    assert len(plan["nodes"][1]["post_reverse"]) > 1
    assert any(
        launch["grid"] == "node_channel_tiles"
        for launch in plan["nodes"][0]["post_forward"]
    )
    tiled_launches = [
        launch
        for node in plan["nodes"]
        for phase in ("post_forward", "post_reverse", "pre_reverse")
        for launch in node[phase]
        if launch["grid"] == "node_channel_tiles"
    ]
    assert tiled_launches
    for launch in tiled_launches:
        expected_tile_channels = 32
        expected_tile_nodes = 8
        if "message" in launch["kernel"]:
            expected_tile_nodes = 32
        if "l1_reverse_message_grouped" in launch["kernel"]:
            expected_tile_nodes = 64
        if "linear2" in launch["kernel"]:
            expected_tile_nodes = 32
        if "l1_reverse_linear2_grouped" in launch["kernel"]:
            expected_tile_nodes = 128
        if "l0_reverse_linear2_grouped" in launch["kernel"]:
            expected_tile_nodes = 128
        if "l0_forward_linear2_grouped" in launch["kernel"]:
            expected_tile_nodes = 128
        if "l1_forward_linear2_grouped" in launch["kernel"]:
            expected_tile_nodes = 128
        if "residual_grouped" in launch["kernel"]:
            expected_tile_nodes = 32
        if "linear_up_grouped" in launch["kernel"]:
            expected_tile_nodes = 32
        if "reverse_readout" in launch["kernel"]:
            expected_tile_nodes = 32
        if "reverse_skip_grouped" in launch["kernel"]:
            expected_tile_nodes = 32
        if "forward_skip_grouped" in launch["kernel"]:
            expected_tile_nodes = 32
        if "product_linear" in launch["kernel"]:
            expected_tile_nodes = 32
        if "reverse_product_tiled" in launch["kernel"]:
            expected_tile_nodes = 4
            expected_tile_channels = 128
        if "forward_product_tiled" in launch["kernel"]:
            expected_tile_nodes = 1
            expected_tile_channels = 128
        assert launch["tile_nodes"] == expected_tile_nodes
        assert launch["tile_channels"] == expected_tile_channels


def test_mh1_v4_spline_r_uses_exact_graph_without_r1_cutoff_abi(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    module = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)

    assert module.count("void direct_harmonic_gradients(") == 2
    assert "source_args.cutoff" not in module
    assert "args->cutoff" not in module
    assert "edge_is_active(" not in module
    assert "source_args.source_ids" not in module
    assert (
        "source_args.neigh_indices[source_args.source_edges[first_scheduled]]" in module
    )
    assert module.count("static_cast<Scalar*>(args.output)") == 2


def test_mh1_v4_recompute_launch_plan_groups_forward_but_keeps_replay_blocks(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract, 80, node_state_policy="recompute-v1"
    )

    for node in plan["nodes"]:
        index = node["index"]
        forward_kernels = {launch["kernel"] for launch in node["post_forward"]}
        reverse_kernels = {launch["kernel"] for launch in node["post_reverse"]}
        assert {
            f"symmetrix_execution_mh1_node_tiled_l{index}_forward_residual_grouped",
            f"symmetrix_execution_mh1_node_tiled_l{index}_forward_message_grouped",
        } <= forward_kernels
        assert {
            f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_residual_forward_block_0",
            f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_message_forward_block_0",
            f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_residual_recompute_block_0",
            f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_message_recompute_block_0",
        } <= reverse_kernels


def test_mh1_v4_reuse_adjoint_policy_retains_primals_and_destroys_messages(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract, 80, node_state_policy="reuse-adjoints-v1"
    )
    module = mh1_codegen.render_execution_mh1_cuda_module_v4(
        contract, 80, node_state_policy="reuse-adjoints-v1"
    )
    hip_target = {
        "architecture": "gfx1151",
        "target_features": ["sramecc-", "xnack-"],
        "compiler_offload_target": "gfx1151",
        "native_subgroup_width": 32,
        "compute_unit_count": 40,
    }
    hip_module = mh1_codegen.render_execution_mh1_hip_module_v4(
        contract, hip_target, node_state_policy="reuse-adjoints-v1"
    )
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract, 80, node_state_policy="reuse-adjoints-v1"
    )
    hip_plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(
        contract, hip_target, node_state_policy="reuse-adjoints-v1"
    )

    assert metadata["node_state_policy"] == "reuse-adjoints-v1"
    assert all(layer["reuse_message_adjoint"] for layer in metadata["layers"])
    assert all(
        layer["retained_pre_gate_dimension"] == layer["residual_dimension"]
        for layer in metadata["layers"]
    )
    assert all(
        layer["retained_interaction_output_dimension"]
        == layer["interaction_output_dimension"]
        for layer in metadata["layers"]
    )
    assert plan["node_state_policy"] == "reuse-adjoints-v1"
    assert plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert "args.retained_pre_gate + node" in module
    assert "args.retained_interaction_output" in module
    assert "atomicAdd((args.node_density_adjoint) + node_0" in module
    assert "l0_tiled_density_bias_sum" in module
    assert "l1_tiled_density_bias_sum" in module
    assert "l0_tiled_density_reuse_sum" not in module
    assert "atomicAdd((args.node_density_adjoint) + node_0" in hip_module
    assert "__shfl_down(" in hip_module
    assert "__shfl_down_sync" not in hip_module
    assert hip_plan["node_state_policy"] == "reuse-adjoints-v1"
    for launch_plan in (plan, hip_plan):
        assert launch_plan["node_reverse_schedule"] == (
            mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
        )
        for node in launch_plan["nodes"]:
            index = node["index"]
            kernels = [launch["kernel"] for launch in node["post_reverse"]]
            scale = f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_scale"
            density = f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_density"
            message = (
                f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_message_grouped"
            )
            assert (
                kernels.index(scale) < kernels.index(message) < kernels.index(density)
            )
            assert not any("reverse_density_prepare" in kernel for kernel in kernels)
            assert not any("reverse_message_forward" in kernel for kernel in kernels)
            assert not any("reverse_message_recompute" in kernel for kernel in kernels)


def test_mh1_v4_retain_interaction_policy_replays_only_pre_gate(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract, 80, node_state_policy="retain-interaction-v1"
    )
    module = mh1_codegen.render_execution_mh1_cuda_module_v4(
        contract, 80, node_state_policy="retain-interaction-v1"
    )
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract, 80, node_state_policy="retain-interaction-v1"
    )

    assert metadata["node_state_policy"] == "retain-interaction-v1"
    assert all(
        layer["retained_pre_gate_dimension"] == 0 for layer in metadata["layers"]
    )
    assert all(
        layer["retained_interaction_output_dimension"]
        == layer["interaction_output_dimension"]
        for layer in metadata["layers"]
    )
    assert all(layer["reuse_message_adjoint"] for layer in metadata["layers"])
    assert plan["node_state_policy"] == "retain-interaction-v1"
    assert "args.retained_pre_gate + node" not in module
    assert "args.retained_interaction_output" in module
    assert "node_tiled_l0_reverse_gate_forward" not in module
    assert "node_tiled_l0_reverse_residual_recompute" in module
    assert "node_tiled_l1_reverse_message_recompute" in module


def test_mh1_v4_reuse_adjoint_fusion_has_a_new_generation_identity(
    mh1_node_definition, monkeypatch
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    current = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract, 80, node_state_policy="reuse-adjoints-v1"
    )

    monkeypatch.setattr(mh1_codegen, "JIT_GENERATION_VERSION", 5)
    pre_fusion = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract, 80, node_state_policy="reuse-adjoints-v1"
    )

    assert current["generation_fingerprint"] == pre_fusion["generation_fingerprint"]
    assert current["artifact_id"].startswith("jit-mh1-gen11-v4-")
    assert pre_fusion["artifact_id"].startswith("jit-mh1-gen5-v4-")
    assert current["artifact_id"] != pre_fusion["artifact_id"]


def test_mh1_v4_retained_reverse_defaults_grouped_but_keeps_monolithic_experimental(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    target = {
        "architecture": "gfx1151",
        "target_features": ["sramecc-", "xnack-"],
        "compiler_offload_target": "gfx1151",
        "native_subgroup_width": 32,
        "compute_unit_count": 40,
    }
    experimental_metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule="monolithic-l1-retained-v1",
    )
    cuda_module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(experimental_metadata),
        node_reverse_schedule="monolithic-l1-retained-v1",
    )
    recompute_module = mh1_codegen.render_execution_mh1_cuda_module_v4(
        contract, 80, node_state_policy="recompute-v1"
    )
    cuda_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    experimental_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule="monolithic-l1-retained-v1",
    )
    recompute_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract, 80, node_state_policy="recompute-v1"
    )
    hip_plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(contract, target)

    symbol = "void symmetrix_execution_mh1_node_post_reverse_kernel_1("
    begin = cuda_module.index(symbol)
    end = cuda_module.index('\nextern "C" __global__ void ', begin + len(symbol))
    retained_kernel = cuda_module[begin:end]
    begin = recompute_module.index(symbol)
    end = recompute_module.index('\nextern "C" __global__ void ', begin + len(symbol))
    recompute_kernel = recompute_module[begin:end]

    assert (
        "const float* interaction = args.retained_interaction_output" in retained_kernel
    )
    assert "const float* retained_pre_gate = args.retained_pre_gate" in retained_kernel
    assert "float* pre_gate = arena;" not in retained_kernel
    assert "float* gated = arena +" not in retained_kernel
    assert "float* pre_gate = arena;" in recompute_kernel
    assert "float* gated = arena +" in recompute_kernel
    assert "args.retained_pre_gate + node" not in recompute_module
    assert "args.retained_interaction_output + node" not in recompute_module
    assert "args->retained_pre_gate == nullptr" not in recompute_module
    assert "args->retained_interaction_output == nullptr" not in recompute_module
    assert "node_tiled_l1_reverse_message_grouped" in recompute_module
    product_symbol = "void symmetrix_execution_mh1_node_tiled_l0_reverse_product_tiled("
    begin = recompute_module.index(product_symbol)
    end = recompute_module.index(
        '\nextern "C" __global__ void ', begin + len(product_symbol)
    )
    recompute_product_kernel = recompute_module[begin:end]
    assert "(args.node_arena)" in recompute_product_kernel
    assert "(args.retained_interaction_output)" not in recompute_product_kernel
    assert experimental_plan["node_reverse_schedule"] == "monolithic-l1-retained-v1"
    assert cuda_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert recompute_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE
    )
    assert hip_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert len(experimental_plan["nodes"][1]["post_reverse"]) == 1
    assert len(cuda_plan["nodes"][1]["post_reverse"]) > 1
    assert len(recompute_plan["nodes"][1]["post_reverse"]) > 1
    assert all(
        node["retained_pre_gate_dimension"] == 0
        and node["retained_interaction_output_dimension"] == 0
        for node in recompute_plan["nodes"]
    )
    assert len(hip_plan["nodes"][1]["post_reverse"]) > 1


def test_mh1_v4_grouped_message_reverse_flattens_only_retained_cuda_layer_one(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    tiled_schedule = mh1_codegen._MH1_NODE_REVERSE_SCHEDULE
    tiled_metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=tiled_schedule,
    )
    tiled_module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(tiled_metadata),
        node_reverse_schedule=tiled_schedule,
    )
    default_module = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)
    default_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    tiled_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=tiled_schedule,
    )
    grouped_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )

    grouped_symbol = "symmetrix_execution_mh1_node_tiled_l1_reverse_message_grouped"
    assert grouped_symbol in module
    assert grouped_symbol in default_module
    assert grouped_symbol not in tiled_module
    assert "const int logical_tile = blockIdx.x;" in module
    assert "const int component = local_tile / source_tiles;" in module
    assert (
        "const int source_base = (local_tile % source_tiles) * tile_channels;" in module
    )
    assert "__shared__ float adjoint_tile[tile_nodes][tile_channels];" in module
    assert "__shared__ float weight_tile[tile_channels][tile_channels + 1];" in module

    grouped_messages = [
        launch
        for launch in grouped_plan["nodes"][1]["post_reverse"]
        if "reverse_message" in launch["kernel"]
    ]
    tiled_messages = [
        launch
        for launch in tiled_plan["nodes"][1]["post_reverse"]
        if "reverse_message" in launch["kernel"]
    ]
    assert grouped_plan["node_reverse_schedule"] == schedule
    assert default_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    default_messages = [
        launch
        for launch in default_plan["nodes"][1]["post_reverse"]
        if "reverse_message" in launch["kernel"]
    ]
    assert len(default_messages) == 1
    assert default_messages[0]["tile_nodes"] == 64
    assert {
        key: value for key, value in default_messages[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in grouped_messages[0].items() if key != "tile_nodes"
    }
    assert len(grouped_messages) == 1
    assert len(tiled_messages) == 4
    assert grouped_messages[0] == {
        "kernel": grouped_symbol,
        "grid": "node_channel_tiles",
        "threads": 256,
        "channels": 512,
        "components": 1,
        "tile_nodes": 32,
        "tile_channels": 32,
    }
    assert (
        grouped_plan["nodes"][0]["post_reverse"]
        == tiled_plan["nodes"][0]["post_reverse"]
    )
    assert [
        launch
        for launch in grouped_plan["nodes"][1]["post_reverse"]
        if "reverse_message" not in launch["kernel"]
    ] == [
        launch
        for launch in tiled_plan["nodes"][1]["post_reverse"]
        if "reverse_message" not in launch["kernel"]
    ]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_linear2_flattens_each_sibling_family(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE,
    )

    assert candidate["node_reverse_schedule"] == schedule
    assert accepted["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_GROUPED_MESSAGE_REVERSE_SCHEDULE
    )
    for index, (layer, candidate_node, accepted_node) in enumerate(
        zip(
            contract["node_program"]["layers"],
            candidate["nodes"],
            accepted["nodes"],
            strict=True,
        )
    ):
        descriptor = layer["linears"]["linear_2"]
        forward_tiles = sum(
            (int(block["multiplicity"]) + 31) // 32 * int(block["components"])
            for block in descriptor["blocks"]["output"]
        )
        reverse_tiles = sum(
            (int(block["multiplicity"]) + 31) // 32 * int(block["components"])
            for block in descriptor["blocks"]["input"]
        )
        forward_symbol = (
            f"symmetrix_execution_mh1_node_tiled_l{index}_forward_linear2_grouped"
        )
        reverse_symbol = (
            f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_linear2_grouped"
        )
        candidate_forward = [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_linear2" in launch["kernel"]
        ]
        candidate_reverse = [
            launch
            for launch in candidate_node["post_reverse"]
            if "reverse_linear2" in launch["kernel"]
        ]
        accepted_forward = [
            launch
            for launch in accepted_node["post_forward"]
            if "forward_linear2" in launch["kernel"]
        ]
        accepted_reverse = [
            launch
            for launch in accepted_node["post_reverse"]
            if "reverse_linear2" in launch["kernel"]
        ]

        assert forward_symbol in module
        assert reverse_symbol in module
        assert len(candidate_forward) == 1
        assert len(candidate_reverse) == 1
        assert len(accepted_forward) == len(descriptor["blocks"]["output"])
        assert len(accepted_reverse) == len(descriptor["blocks"]["input"])
        assert candidate_forward[0] == {
            "kernel": forward_symbol,
            "grid": "node_channel_tiles",
            "threads": 256,
            "channels": forward_tiles * 32,
            "components": 1,
            "tile_nodes": 32,
            "tile_channels": 32,
        }
        assert candidate_reverse[0] == {
            "kernel": reverse_symbol,
            "grid": "node_channel_tiles",
            "threads": 256,
            "channels": reverse_tiles * 32,
            "components": 1,
            "tile_nodes": 32,
            "tile_channels": 32,
        }
        assert [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_linear2" not in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_forward"]
            if "forward_linear2" not in launch["kernel"]
        ]
        assert [
            launch
            for launch in candidate_node["post_reverse"]
            if "reverse_linear2" not in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_reverse"]
            if "reverse_linear2" not in launch["kernel"]
        ]

    assert (
        "const int target_base = (local_tile % target_tiles) * tile_channels;" in module
    )
    assert module.count("__shared__ float input_tile[tile_nodes][tile_channels];") >= 2
    assert (
        module.count("__shared__ float weight_tile[tile_channels][tile_channels + 1];")
        >= 4
    )
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_all_message_adds_only_layer_zero_message(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_GROUPED_LINEAR2_SCHEDULE,
    )

    assert candidate["node_reverse_schedule"] == schedule
    assert accepted["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_GROUPED_LINEAR2_SCHEDULE
    )
    for index, (layer, candidate_node, accepted_node) in enumerate(
        zip(
            contract["node_program"]["layers"],
            candidate["nodes"],
            accepted["nodes"],
            strict=True,
        )
    ):
        descriptor = layer["linears"]["linear_1"]
        reverse_tiles = sum(
            (int(block["multiplicity"]) + 31) // 32 * int(block["components"])
            for block in descriptor["blocks"]["input"]
        )
        grouped_symbol = (
            f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_message_grouped"
        )
        candidate_messages = [
            launch
            for launch in candidate_node["post_reverse"]
            if "reverse_message" in launch["kernel"]
        ]
        accepted_messages = [
            launch
            for launch in accepted_node["post_reverse"]
            if "reverse_message" in launch["kernel"]
        ]

        assert grouped_symbol in module
        assert candidate_messages == [
            {
                "kernel": grouped_symbol,
                "grid": "node_channel_tiles",
                "threads": 256,
                "channels": reverse_tiles * 32,
                "components": 1,
                "tile_nodes": 32,
                "tile_channels": 32,
            }
        ]
        assert len(accepted_messages) == (4 if index == 0 else 1)
        assert [
            launch
            for launch in candidate_node["post_reverse"]
            if "reverse_message" not in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_reverse"]
            if "reverse_message" not in launch["kernel"]
        ]
        assert [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_linear2" in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_forward"]
            if "forward_linear2" in launch["kernel"]
        ]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_forward_message_flattens_each_layer(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_GROUPED_ALL_MESSAGE_SCHEDULE,
    )

    assert candidate["node_reverse_schedule"] == schedule
    for index, (layer, candidate_node, accepted_node) in enumerate(
        zip(
            contract["node_program"]["layers"],
            candidate["nodes"],
            accepted["nodes"],
            strict=True,
        )
    ):
        descriptor = layer["linears"]["linear_1"]
        forward_tiles = sum(
            (int(block["multiplicity"]) + 31) // 32 * int(block["components"])
            for block in descriptor["blocks"]["output"]
        )
        grouped_symbol = (
            f"symmetrix_execution_mh1_node_tiled_l{index}_forward_message_grouped"
        )
        candidate_messages = [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_message" in launch["kernel"]
        ]
        accepted_messages = [
            launch
            for launch in accepted_node["post_forward"]
            if "forward_message" in launch["kernel"]
        ]

        assert grouped_symbol in module
        assert candidate_messages == [
            {
                "kernel": grouped_symbol,
                "grid": "node_channel_tiles",
                "threads": 256,
                "channels": forward_tiles * 32,
                "components": 1,
                "tile_nodes": 32,
                "tile_channels": 32,
            }
        ]
        assert len(accepted_messages) == 4
        assert [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_message" not in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_forward"]
            if "forward_message" not in launch["kernel"]
        ]
        assert candidate_node["post_reverse"] == accepted_node["post_reverse"]
    assert "args.node_density[node_0]" in module
    assert "args.node_density[node_3]" in module
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_residual_flattens_forward_and_layer_one_reverse(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_RESIDUAL_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=(mh1_codegen._MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE),
    )

    for index, (layer, candidate_node, accepted_node) in enumerate(
        zip(
            contract["node_program"]["layers"],
            candidate["nodes"],
            accepted["nodes"],
            strict=True,
        )
    ):
        descriptor = layer["linears"]["linear_res"]
        forward_tiles = sum(
            (int(block["multiplicity"]) + 31) // 32 * int(block["components"])
            for block in descriptor["blocks"]["output"]
        )
        forward_symbol = (
            f"symmetrix_execution_mh1_node_tiled_l{index}_forward_residual_grouped"
        )
        candidate_forward = [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_residual" in launch["kernel"]
        ]
        assert forward_symbol in module
        assert candidate_forward == [
            {
                "kernel": forward_symbol,
                "grid": "node_channel_tiles",
                "threads": 256,
                "channels": forward_tiles * 32,
                "components": 1,
                "tile_nodes": 32,
                "tile_channels": 32,
            }
        ]
        assert [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_residual" not in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_forward"]
            if "forward_residual" not in launch["kernel"]
        ]

        candidate_reverse = [
            launch
            for launch in candidate_node["post_reverse"]
            if "reverse_residual" in launch["kernel"]
        ]
        accepted_reverse = [
            launch
            for launch in accepted_node["post_reverse"]
            if "reverse_residual" in launch["kernel"]
        ]
        if index == 0:
            assert candidate_reverse == accepted_reverse == []
        else:
            reverse_tiles = sum(
                (int(block["multiplicity"]) + 31) // 32 * int(block["components"])
                for block in descriptor["blocks"]["input"]
            )
            reverse_symbol = (
                "symmetrix_execution_mh1_node_tiled_l1_reverse_residual_grouped"
            )
            assert reverse_symbol in module
            assert candidate_reverse == [
                {
                    "kernel": reverse_symbol,
                    "grid": "node_channel_tiles",
                    "threads": 256,
                    "channels": reverse_tiles * 32,
                    "components": 1,
                    "tile_nodes": 32,
                    "tile_channels": 32,
                }
            ]
            assert len(accepted_reverse) == 2
        assert [
            launch
            for launch in candidate_node["post_reverse"]
            if "reverse_residual" not in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_reverse"]
            if "reverse_residual" not in launch["kernel"]
        ]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_forward_residual_keeps_staged_reverse(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=(mh1_codegen._MH1_CUDA_GROUPED_FORWARD_MESSAGE_SCHEDULE),
    )

    assert candidate["node_reverse_schedule"] == schedule
    for index, (candidate_node, accepted_node) in enumerate(
        zip(candidate["nodes"], accepted["nodes"], strict=True)
    ):
        forward_residual = [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_residual" in launch["kernel"]
        ]
        assert len(forward_residual) == 1
        assert forward_residual[0]["kernel"].endswith(
            f"node_tiled_l{index}_forward_residual_grouped"
        )
        assert [
            launch
            for launch in candidate_node["post_forward"]
            if "forward_residual" not in launch["kernel"]
        ] == [
            launch
            for launch in accepted_node["post_forward"]
            if "forward_residual" not in launch["kernel"]
        ]
        assert candidate_node["post_reverse"] == accepted_node["post_reverse"]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_tiled_product_reverse_splits_angular_channel_work(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=(
            mh1_codegen._MH1_CUDA_GROUPED_FORWARD_RESIDUAL_SCHEDULE
        ),
    )

    for index, (layer, candidate_node, accepted_node) in enumerate(
        zip(
            contract["node_program"]["layers"],
            candidate["nodes"],
            accepted["nodes"],
            strict=True,
        )
    ):
        product = layer["product"]
        symbol = f"symmetrix_execution_mh1_node_tiled_l{index}_reverse_product_tiled"
        candidate_product = [
            launch
            for launch in candidate_node["post_reverse"]
            if launch["kernel"].endswith("_reverse_product_tiled")
        ]
        accepted_product = [
            launch
            for launch in accepted_node["post_reverse"]
            if launch["kernel"].endswith("_reverse_product")
        ]
        assert symbol in module
        assert candidate_product == [
            {
                "kernel": symbol,
                "grid": "node_channel_tiles",
                "threads": 128,
                "channels": int(product["dimensions"]["channels"]),
                "components": len(product["angular_layout"]),
                "tile_nodes": 1,
                "tile_channels": 128,
            }
        ]
        assert len(accepted_product) == 1
        assert accepted_product[0]["grid"] == "persistent_nodes"
        assert candidate_node["post_forward"] == accepted_node["post_forward"]
        assert [
            launch
            for launch in candidate_node["post_reverse"]
            if not launch["kernel"].endswith("_reverse_product_tiled")
        ] == [
            launch
            for launch in accepted_node["post_reverse"]
            if not launch["kernel"].endswith("_reverse_product")
        ]
    assert "const int angular_index = blockIdx.z;" in module
    assert "const std::int64_t node = blockIdx.y;" in module
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_tiled_product_forward_splits_output_channel_work(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_TILED_PRODUCT_REVERSE_SCHEDULE,
    )

    for index, (layer, candidate_node, accepted_node) in enumerate(
        zip(
            contract["node_program"]["layers"],
            candidate["nodes"],
            accepted["nodes"],
            strict=True,
        )
    ):
        product = layer["product"]
        symbol = f"symmetrix_execution_mh1_node_tiled_l{index}_forward_product_tiled"
        candidate_product = [
            launch
            for launch in candidate_node["post_forward"]
            if launch["kernel"].endswith("_forward_product_tiled")
        ]
        assert symbol in module
        assert candidate_product == [
            {
                "kernel": symbol,
                "grid": "node_channel_tiles",
                "threads": 128,
                "channels": int(product["dimensions"]["channels"]),
                "components": (
                    int(product["dimensions"]["output"])
                    // int(product["dimensions"]["channels"])
                ),
                "tile_nodes": 1,
                "tile_channels": 128,
            }
        ]
        assert any(
            launch["kernel"].endswith("_forward_product_setup")
            and launch["grid"] == "persistent_nodes"
            for launch in candidate_node["post_forward"]
        )
        assert [
            launch
            for launch in candidate_node["post_forward"]
            if not launch["kernel"].endswith("_forward_product_setup")
            and not launch["kernel"].endswith("_forward_product_tiled")
        ] == [
            launch
            for launch in accepted_node["post_forward"]
            if not launch["kernel"].endswith("_forward_product")
        ]
        assert candidate_node["post_reverse"] == accepted_node["post_reverse"]
    assert "const int output_component = blockIdx.z;" in module
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_tiled_linear_up_reverse_splits_pre_reverse_work(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=(
            mh1_codegen._MH1_CUDA_TILED_PRODUCT_FORWARD_REVERSE_SCHEDULE
        ),
    )

    linear_up = contract["node_program"]["layers"][1]["linears"]["linear_up"]
    logical_tiles = sum(
        ((int(block["multiplicity"]) + 31) // 32) * int(block["components"])
        for block in linear_up["blocks"]["input"]
    )
    symbol = "symmetrix_execution_mh1_node_tiled_l1_pre_reverse_linear_up_grouped"
    assert candidate["nodes"][0]["pre_reverse"] == []
    assert candidate["nodes"][1]["pre_reverse"] == [
        {
            "kernel": symbol,
            "grid": "node_channel_tiles",
            "threads": 256,
            "channels": logical_tiles * 32,
            "components": 1,
            "tile_nodes": 32,
            "tile_channels": 32,
        }
    ]
    assert symbol in module
    assert "symmetrix_execution_mh1_node_pre_reverse_kernel_1" not in module
    assert "args.layer_input_adjoint" in module
    assert "] += value_0;" in module
    for candidate_node, accepted_node in zip(
        candidate["nodes"], accepted["nodes"], strict=True
    ):
        assert candidate_node["pre_forward"] == accepted_node["pre_forward"]
        assert candidate_node["post_forward"] == accepted_node["post_forward"]
        assert candidate_node["post_reverse"] == accepted_node["post_reverse"]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_tiled_linear_up_forward_splits_pre_forward_work(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_TILED_LINEAR_UP_REVERSE_SCHEDULE,
    )

    linear_up = contract["node_program"]["layers"][1]["linears"]["linear_up"]
    logical_tiles = sum(
        ((int(block["multiplicity"]) + 31) // 32) * int(block["components"])
        for block in linear_up["blocks"]["output"]
    )
    symbol = "symmetrix_execution_mh1_node_tiled_l1_pre_forward_linear_up_grouped"
    assert candidate["nodes"][0]["pre_forward"] == accepted["nodes"][0]["pre_forward"]
    assert candidate["nodes"][1]["pre_forward"] == [
        {
            "kernel": symbol,
            "grid": "node_channel_tiles",
            "threads": 256,
            "channels": logical_tiles * 32,
            "components": 1,
            "tile_nodes": 32,
            "tile_channels": 32,
        }
    ]
    assert candidate["nodes"][1]["pre_reverse"] == accepted["nodes"][1]["pre_reverse"]
    assert candidate["nodes"][1]["post_forward"] == accepted["nodes"][1]["post_forward"]
    assert candidate["nodes"][1]["post_reverse"] == accepted["nodes"][1]["post_reverse"]
    assert symbol in module
    assert "symmetrix_execution_mh1_node_pre_forward_kernel_0" in module
    assert "symmetrix_execution_mh1_node_pre_forward_kernel_1" not in module
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_readout_reverse_splits_dependent_work(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=(
            mh1_codegen._MH1_CUDA_TILED_LINEAR_UP_FORWARD_REVERSE_SCHEDULE
        ),
    )
    default_module = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)
    default_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)

    assert candidate["nodes"][0] == accepted["nodes"][0]
    for phase in ("pre_forward", "post_forward", "pre_reverse"):
        assert candidate["nodes"][1][phase] == accepted["nodes"][1][phase]
    specialized = candidate["nodes"][1]["post_reverse"][:4]
    assert [
        launch["kernel"].rsplit("node_tiled_l1_", 1)[-1] for launch in specialized
    ] == [
        "reverse_readout1_recompute_grouped",
        "reverse_readout_middle",
        "reverse_readout1_grouped",
        "reverse_skip_grouped",
    ]
    assert [launch["grid"] for launch in specialized] == [
        "node_channel_tiles",
        "persistent_nodes",
        "node_channel_tiles",
        "node_channel_tiles",
    ]
    assert (
        candidate["nodes"][1]["post_reverse"][4:]
        == accepted["nodes"][1]["post_reverse"][1:]
    )
    for launch in specialized:
        assert launch["kernel"] in module
        assert launch["kernel"] in default_module
    assert default_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert [
        launch
        for launch in default_plan["nodes"][1]["post_reverse"]
        if "product_linear" not in launch["kernel"]
        and "reverse_product_tiled" not in launch["kernel"]
        and "reverse_message" not in launch["kernel"]
        and "reverse_linear2" not in launch["kernel"]
    ] == [
        launch
        for launch in candidate["nodes"][1]["post_reverse"]
        if "product_linear" not in launch["kernel"]
        and "reverse_product_tiled" not in launch["kernel"]
        and "reverse_message" not in launch["kernel"]
        and "reverse_linear2" not in launch["kernel"]
    ]
    assert "void symmetrix_execution_mh1_node_tiled_l1_reverse_readout(" not in module
    assert "args.layer_output_adjoint" in module
    assert "] += value_0;" in module
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_skip_forward_splits_product_setup(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_GROUPED_READOUT_REVERSE_SCHEDULE,
    )
    default_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)

    assert candidate["nodes"][0] == accepted["nodes"][0]
    for phase in ("pre_forward", "pre_reverse", "post_reverse"):
        assert candidate["nodes"][1][phase] == accepted["nodes"][1][phase]
    skip = contract["node_program"]["layers"][1]["linears"]["skip"]
    logical_tiles = sum(
        ((int(block["multiplicity"]) + 31) // 32) * int(block["components"])
        for block in skip["blocks"]["output"]
    )
    skip_symbol = "symmetrix_execution_mh1_node_tiled_l1_forward_skip_grouped"
    skip_launches = [
        launch
        for launch in candidate["nodes"][1]["post_forward"]
        if launch["kernel"] == skip_symbol
    ]
    assert skip_launches == [
        {
            "kernel": skip_symbol,
            "grid": "node_channel_tiles",
            "threads": 256,
            "channels": logical_tiles * 32,
            "components": 1,
            "tile_nodes": 32,
            "tile_channels": 32,
        }
    ]
    assert [
        launch
        for launch in candidate["nodes"][1]["post_forward"]
        if launch["kernel"] != skip_symbol
    ] == accepted["nodes"][1]["post_forward"]
    assert default_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    for phase in ("pre_forward", "pre_reverse"):
        assert default_plan["nodes"][1][phase] == candidate["nodes"][1][phase]
    for phase in ("post_forward", "post_reverse"):
        assert [
            launch
            for launch in default_plan["nodes"][1][phase]
            if "product_linear" not in launch["kernel"]
            and "reverse_product_tiled" not in launch["kernel"]
            and "reverse_message" not in launch["kernel"]
            and "reverse_linear2" not in launch["kernel"]
            and "l1_forward_linear2_grouped" not in launch["kernel"]
        ] == [
            launch
            for launch in candidate["nodes"][1][phase]
            if "product_linear" not in launch["kernel"]
            and "reverse_product_tiled" not in launch["kernel"]
            and "reverse_message" not in launch["kernel"]
            and "reverse_linear2" not in launch["kernel"]
            and "l1_forward_linear2_grouped" not in launch["kernel"]
        ]
    assert skip_symbol in module
    setup_symbol = "void symmetrix_execution_mh1_node_tiled_l1_forward_product_setup("
    setup_begin = module.index(setup_symbol)
    setup_end = module.index('\nextern "C" __global__ void ', setup_begin)
    setup = module[setup_begin:setup_end]
    assert "retained_interaction[column] = interaction[column];" in setup
    assert "l1_skip_work" not in setup
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_grouped_all_skip_forward_splits_layer_zero_product_setup(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_GROUPED_SKIP_FORWARD_SCHEDULE,
    )

    assert candidate["nodes"][1] == accepted["nodes"][1]
    for phase in ("pre_forward", "pre_reverse", "post_reverse"):
        assert candidate["nodes"][0][phase] == accepted["nodes"][0][phase]
    skip = contract["node_program"]["layers"][0]["linears"]["skip"]
    logical_tiles = sum(
        ((int(block["multiplicity"]) + 31) // 32) * int(block["components"])
        for block in skip["blocks"]["output"]
    )
    skip_symbol = "symmetrix_execution_mh1_node_tiled_l0_forward_skip_grouped"
    skip_launches = [
        launch
        for launch in candidate["nodes"][0]["post_forward"]
        if launch["kernel"] == skip_symbol
    ]
    assert skip_launches == [
        {
            "kernel": skip_symbol,
            "grid": "node_channel_tiles",
            "threads": 256,
            "channels": logical_tiles * 32,
            "components": 1,
            "tile_nodes": 32,
            "tile_channels": 32,
        }
    ]
    assert [
        launch
        for launch in candidate["nodes"][0]["post_forward"]
        if launch["kernel"] != skip_symbol
    ] == accepted["nodes"][0]["post_forward"]
    default_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    assert default_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    for default_node, candidate_node in zip(
        default_plan["nodes"], candidate["nodes"], strict=True
    ):
        for phase in ("pre_forward", "pre_reverse"):
            assert default_node[phase] == candidate_node[phase]
        for phase in ("post_forward", "post_reverse"):
            assert [
                launch
                for launch in default_node[phase]
                if "product_linear" not in launch["kernel"]
                and "reverse_message" not in launch["kernel"]
                and "reverse_linear2" not in launch["kernel"]
                and "l0_forward_linear2_grouped" not in launch["kernel"]
                and "l1_forward_linear2_grouped" not in launch["kernel"]
                and "reverse_product_tiled" not in launch["kernel"]
            ] == [
                launch
                for launch in candidate_node[phase]
                if "product_linear" not in launch["kernel"]
                and "reverse_message" not in launch["kernel"]
                and "reverse_linear2" not in launch["kernel"]
                and "l0_forward_linear2_grouped" not in launch["kernel"]
                and "l1_forward_linear2_grouped" not in launch["kernel"]
                and "reverse_product_tiled" not in launch["kernel"]
            ]
    assert skip_symbol in module
    setup_symbol = "void symmetrix_execution_mh1_node_tiled_l0_forward_product_setup("
    setup_begin = module.index(setup_symbol)
    setup_end = module.index('\nextern "C" __global__ void ', setup_begin)
    setup = module[setup_begin:setup_end]
    assert "args.element_indices[node]" in setup
    assert "retained_interaction[column] = interaction[column];" in setup
    assert "l0_skip_work" not in setup
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_wide_product_linear_uses_32_node_tiles(mh1_node_definition):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_GROUPED_ALL_SKIP_FORWARD_SCHEDULE,
    )
    assert candidate["node_reverse_schedule"] == schedule
    assert mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)[
        "node_reverse_schedule"
    ] == (mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE)

    for candidate_node, accepted_node in zip(
        candidate["nodes"], accepted["nodes"], strict=True
    ):
        for phase in ("post_forward", "post_reverse"):
            candidate_product = [
                launch
                for launch in candidate_node[phase]
                if "product_linear" in launch["kernel"]
            ]
            accepted_product = [
                launch
                for launch in accepted_node[phase]
                if "product_linear" in launch["kernel"]
            ]
            assert len(candidate_product) == len(accepted_product)
            for candidate_launch, accepted_launch in zip(
                candidate_product, accepted_product, strict=True
            ):
                assert candidate_launch["tile_nodes"] == 32
                assert accepted_launch["tile_nodes"] == 8
                assert {
                    key: value
                    for key, value in candidate_launch.items()
                    if key != "tile_nodes"
                } == {
                    key: value
                    for key, value in accepted_launch.items()
                    if key != "tile_nodes"
                }
                kernel_begin = module.index(f"void {candidate_launch['kernel']}(")
                kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
                assert (
                    "constexpr int tile_nodes = 32;" in module[kernel_begin:kernel_end]
                )
            assert [
                launch
                for launch in candidate_node[phase]
                if "product_linear" not in launch["kernel"]
            ] == [
                launch
                for launch in accepted_node[phase]
                if "product_linear" not in launch["kernel"]
            ]
        for phase in ("pre_forward", "pre_reverse"):
            assert candidate_node[phase] == accepted_node[phase]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_wide_layer_one_reverse_message_uses_64_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_WIDE_L1_REVERSE_MESSAGE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l1_reverse_message_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    assert accepted["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_WIDE_PRODUCT_LINEAR_SCHEDULE
    )
    for node_index, (candidate_node, accepted_node) in enumerate(
        zip(candidate["nodes"], accepted["nodes"], strict=True)
    ):
        for phase in ("pre_forward", "post_forward", "pre_reverse"):
            assert candidate_node[phase] == accepted_node[phase]
        if node_index == 0:
            assert candidate_node["post_reverse"] == accepted_node["post_reverse"]
            continue
        candidate_message = [
            launch
            for launch in candidate_node["post_reverse"]
            if launch["kernel"] == kernel
        ]
        accepted_message = [
            launch
            for launch in accepted_node["post_reverse"]
            if launch["kernel"] == kernel
        ]
        assert len(candidate_message) == len(accepted_message) == 1
        assert candidate_message[0]["tile_nodes"] == 64
        assert accepted_message[0]["tile_nodes"] == 32
        assert {
            key: value
            for key, value in candidate_message[0].items()
            if key != "tile_nodes"
        } == {
            key: value
            for key, value in accepted_message[0].items()
            if key != "tile_nodes"
        }
        assert [
            launch
            for launch in candidate_node["post_reverse"]
            if launch["kernel"] != kernel
        ] == [
            launch
            for launch in accepted_node["post_reverse"]
            if launch["kernel"] != kernel
        ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 64;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_wide_layer_zero_reverse_linear2_uses_64_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_PAIRED_L0_PRODUCT_REVERSE_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l0_reverse_linear2_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    for node_index, (candidate_node, accepted_node) in enumerate(
        zip(candidate["nodes"], accepted["nodes"], strict=True)
    ):
        for phase in ("pre_forward", "post_forward", "pre_reverse"):
            assert candidate_node[phase] == accepted_node[phase]
        if node_index == 1:
            assert candidate_node["post_reverse"] == accepted_node["post_reverse"]
            continue
        candidate_linear2 = [
            launch
            for launch in candidate_node["post_reverse"]
            if launch["kernel"] == kernel
        ]
        accepted_linear2 = [
            launch
            for launch in accepted_node["post_reverse"]
            if launch["kernel"] == kernel
        ]
        assert len(candidate_linear2) == len(accepted_linear2) == 1
        assert candidate_linear2[0]["tile_nodes"] == 64
        assert accepted_linear2[0]["tile_nodes"] == 32
        assert {
            key: value
            for key, value in candidate_linear2[0].items()
            if key != "tile_nodes"
        } == {
            key: value
            for key, value in accepted_linear2[0].items()
            if key != "tile_nodes"
        }
        assert [
            launch
            for launch in candidate_node["post_reverse"]
            if launch["kernel"] != kernel
        ] == [
            launch
            for launch in accepted_node["post_reverse"]
            if launch["kernel"] != kernel
        ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 64;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_wide_layer_zero_forward_linear2_uses_64_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_WIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l0_forward_linear2_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    assert candidate["nodes"][1] == accepted["nodes"][1]
    for phase in ("pre_forward", "pre_reverse", "post_reverse"):
        assert candidate["nodes"][0][phase] == accepted["nodes"][0][phase]
    candidate_linear2 = [
        launch
        for launch in candidate["nodes"][0]["post_forward"]
        if launch["kernel"] == kernel
    ]
    accepted_linear2 = [
        launch
        for launch in accepted["nodes"][0]["post_forward"]
        if launch["kernel"] == kernel
    ]
    assert len(candidate_linear2) == len(accepted_linear2) == 1
    assert candidate_linear2[0]["tile_nodes"] == 64
    assert accepted_linear2[0]["tile_nodes"] == 32
    assert {
        key: value for key, value in candidate_linear2[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in accepted_linear2[0].items() if key != "tile_nodes"
    }
    assert [
        launch
        for launch in candidate["nodes"][0]["post_forward"]
        if launch["kernel"] != kernel
    ] == [
        launch
        for launch in accepted["nodes"][0]["post_forward"]
        if launch["kernel"] != kernel
    ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 64;" in module[kernel_begin:kernel_end]


def test_mh1_v4_xwide_layer_zero_reverse_linear2_uses_128_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l0_reverse_linear2_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    assert candidate["nodes"][1] == accepted["nodes"][1]
    for phase in ("pre_forward", "post_forward", "pre_reverse"):
        assert candidate["nodes"][0][phase] == accepted["nodes"][0][phase]
    candidate_linear2 = [
        launch
        for launch in candidate["nodes"][0]["post_reverse"]
        if launch["kernel"] == kernel
    ]
    accepted_linear2 = [
        launch
        for launch in accepted["nodes"][0]["post_reverse"]
        if launch["kernel"] == kernel
    ]
    assert len(candidate_linear2) == len(accepted_linear2) == 1
    assert candidate_linear2[0]["tile_nodes"] == 128
    assert accepted_linear2[0]["tile_nodes"] == 64
    assert {
        key: value for key, value in candidate_linear2[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in accepted_linear2[0].items() if key != "tile_nodes"
    }
    assert [
        launch
        for launch in candidate["nodes"][0]["post_reverse"]
        if launch["kernel"] != kernel
    ] == [
        launch
        for launch in accepted["nodes"][0]["post_reverse"]
        if launch["kernel"] != kernel
    ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 128;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_xwide_layer_zero_forward_linear2_uses_128_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l0_forward_linear2_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    assert candidate["nodes"][1] == accepted["nodes"][1]
    for phase in ("pre_forward", "pre_reverse", "post_reverse"):
        assert candidate["nodes"][0][phase] == accepted["nodes"][0][phase]
    candidate_linear2 = [
        launch
        for launch in candidate["nodes"][0]["post_forward"]
        if launch["kernel"] == kernel
    ]
    accepted_linear2 = [
        launch
        for launch in accepted["nodes"][0]["post_forward"]
        if launch["kernel"] == kernel
    ]
    assert len(candidate_linear2) == len(accepted_linear2) == 1
    assert candidate_linear2[0]["tile_nodes"] == 128
    assert accepted_linear2[0]["tile_nodes"] == 64
    assert {
        key: value for key, value in candidate_linear2[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in accepted_linear2[0].items() if key != "tile_nodes"
    }
    assert [
        launch
        for launch in candidate["nodes"][0]["post_forward"]
        if launch["kernel"] != kernel
    ] == [
        launch
        for launch in accepted["nodes"][0]["post_forward"]
        if launch["kernel"] != kernel
    ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 128;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_xwide_layer_one_reverse_linear2_uses_128_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_XWIDE_ALL_REVERSE_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_XWIDE_L0_REVERSE_LINEAR2_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l1_reverse_linear2_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    assert candidate["nodes"][0] == accepted["nodes"][0]
    for phase in ("pre_forward", "post_forward", "pre_reverse"):
        assert candidate["nodes"][1][phase] == accepted["nodes"][1][phase]
    candidate_linear2 = [
        launch
        for launch in candidate["nodes"][1]["post_reverse"]
        if launch["kernel"] == kernel
    ]
    accepted_linear2 = [
        launch
        for launch in accepted["nodes"][1]["post_reverse"]
        if launch["kernel"] == kernel
    ]
    assert len(candidate_linear2) == len(accepted_linear2) == 1
    assert candidate_linear2[0]["tile_nodes"] == 128
    assert accepted_linear2[0]["tile_nodes"] == 64
    assert {
        key: value for key, value in candidate_linear2[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in accepted_linear2[0].items() if key != "tile_nodes"
    }
    assert [
        launch
        for launch in candidate["nodes"][1]["post_reverse"]
        if launch["kernel"] != kernel
    ] == [
        launch
        for launch in accepted["nodes"][1]["post_reverse"]
        if launch["kernel"] != kernel
    ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 128;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_xwide_layer_one_forward_linear2_uses_128_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_XWIDE_L0_FORWARD_LINEAR2_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l1_forward_linear2_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    assert candidate["nodes"][0] == accepted["nodes"][0]
    for phase in ("pre_forward", "pre_reverse", "post_reverse"):
        assert candidate["nodes"][1][phase] == accepted["nodes"][1][phase]
    candidate_linear2 = [
        launch
        for launch in candidate["nodes"][1]["post_forward"]
        if launch["kernel"] == kernel
    ]
    accepted_linear2 = [
        launch
        for launch in accepted["nodes"][1]["post_forward"]
        if launch["kernel"] == kernel
    ]
    assert len(candidate_linear2) == len(accepted_linear2) == 1
    assert candidate_linear2[0]["tile_nodes"] == 128
    assert accepted_linear2[0]["tile_nodes"] == 64
    assert {
        key: value for key, value in candidate_linear2[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in accepted_linear2[0].items() if key != "tile_nodes"
    }
    assert [
        launch
        for launch in candidate["nodes"][1]["post_forward"]
        if launch["kernel"] != kernel
    ] == [
        launch
        for launch in accepted["nodes"][1]["post_forward"]
        if launch["kernel"] != kernel
    ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 128;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_wide_layer_one_forward_linear2_uses_64_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_WIDE_L0_FORWARD_LINEAR2_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l1_forward_linear2_grouped"

    assert candidate["node_reverse_schedule"] == schedule
    assert candidate["nodes"][0] == accepted["nodes"][0]
    for phase in ("pre_forward", "pre_reverse", "post_reverse"):
        assert candidate["nodes"][1][phase] == accepted["nodes"][1][phase]
    candidate_linear2 = [
        launch
        for launch in candidate["nodes"][1]["post_forward"]
        if launch["kernel"] == kernel
    ]
    accepted_linear2 = [
        launch
        for launch in accepted["nodes"][1]["post_forward"]
        if launch["kernel"] == kernel
    ]
    assert len(candidate_linear2) == len(accepted_linear2) == 1
    assert candidate_linear2[0]["tile_nodes"] == 64
    assert accepted_linear2[0]["tile_nodes"] == 32
    assert {
        key: value for key, value in candidate_linear2[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in accepted_linear2[0].items() if key != "tile_nodes"
    }
    assert [
        launch
        for launch in candidate["nodes"][1]["post_forward"]
        if launch["kernel"] != kernel
    ] == [
        launch
        for launch in accepted["nodes"][1]["post_forward"]
        if launch["kernel"] != kernel
    ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 64;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_paired_layer_one_product_reverse_uses_four_node_tiles(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    schedule = mh1_codegen._MH1_CUDA_PAIRED_ALL_PRODUCT_REVERSE_SCHEDULE
    metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_CUDA_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(metadata),
        node_reverse_schedule=schedule,
    )
    candidate = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=schedule,
    )
    accepted = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _node_reverse_schedule=mh1_codegen._MH1_CUDA_WIDE_ALL_FORWARD_LINEAR2_SCHEDULE,
    )
    kernel = "symmetrix_execution_mh1_node_tiled_l1_reverse_product_tiled"

    assert candidate["node_reverse_schedule"] == schedule
    assert candidate["nodes"][0] == accepted["nodes"][0]
    for phase in ("pre_forward", "post_forward", "pre_reverse"):
        assert candidate["nodes"][1][phase] == accepted["nodes"][1][phase]
    candidate_product = [
        launch
        for launch in candidate["nodes"][1]["post_reverse"]
        if launch["kernel"] == kernel
    ]
    accepted_product = [
        launch
        for launch in accepted["nodes"][1]["post_reverse"]
        if launch["kernel"] == kernel
    ]
    assert len(candidate_product) == len(accepted_product) == 1
    assert candidate_product[0]["tile_nodes"] == 4
    assert accepted_product[0]["tile_nodes"] == 1
    assert {
        key: value for key, value in candidate_product[0].items() if key != "tile_nodes"
    } == {
        key: value for key, value in accepted_product[0].items() if key != "tile_nodes"
    }
    assert [
        launch
        for launch in candidate["nodes"][1]["post_reverse"]
        if launch["kernel"] != kernel
    ] == [
        launch
        for launch in accepted["nodes"][1]["post_reverse"]
        if launch["kernel"] != kernel
    ]
    kernel_begin = module.index(f"void {kernel}(")
    kernel_end = module.index('\nextern "C" __global__ void ', kernel_begin)
    assert "constexpr int tile_nodes = 4;" in module[kernel_begin:kernel_end]
    with pytest.raises(ValueError, match="requires full node retention"):
        mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
            contract,
            80,
            node_state_policy="recompute-v1",
            _node_reverse_schedule=schedule,
        )


def test_mh1_v4_gpu_module_metadata_and_ir_mul_program(mh1_node_definition, tmp_path):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    target = {
        "architecture": "gfx1151",
        "target_features": ["sramecc-", "xnack-"],
        "compiler_offload_target": "gfx1151",
        "native_subgroup_width": 32,
        "compute_unit_count": 40,
    }
    cuda_module = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)
    hip_module = mh1_codegen.render_execution_mh1_hip_module_v4(contract, target)
    cuda_plugin = mh1_codegen.render_jit_mh1_cuda_plugin_v4(contract, 80)
    host_plugin = mh1_codegen.render_jit_mh1_host_plugin_v4(contract)
    cuda_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    hip_plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(contract, target)
    cuda_recompute_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract, 80, node_state_policy="recompute-v1"
    )
    hip_recompute_plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(
        contract, target, node_state_policy="recompute-v1"
    )
    cuda_metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(contract, 80)
    hip_metadata = mh1_codegen.execution_mh1_hip_module_v4_metadata(contract, target)

    assert "__shfl_down_sync" in cuda_module
    assert "__shfl_down_sync" not in hip_module
    assert "__shfl_down(" in hip_module
    assert (
        'extern "C" __global__ void __launch_bounds__(128, 4) '
        "symmetrix_execution_mh1_forward_kernel_1" in cuda_module
    )
    assert "__launch_bounds__(128, 4)" not in hip_module
    assert "#include" not in hip_module
    assert "<<<" not in hip_module
    assert hip_plan["schema"] == mh1_codegen.MH1_HIP_LAUNCH_PLAN_TAG
    assert hip_plan["version"] == 1
    assert hip_plan["target"]["backend"] == "hip"
    assert hip_plan["target"]["architecture"] == "gfx1151"
    assert hip_plan["module_identity"] in hip_module
    node_schedule = "tiled-8x32-message-linear2-32x32-v7"
    retained_node_schedule = mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    assert cuda_metadata["node_forward_schedule"] == node_schedule
    assert cuda_metadata["node_reverse_schedule"] == node_schedule
    assert hip_metadata["node_forward_schedule"] == node_schedule
    assert hip_metadata["node_reverse_schedule"] == retained_node_schedule
    assert "target_compute_capability" not in hip_metadata
    assert hip_metadata["policy_profile"] == "shared-v13-partitioning-v1"
    for policy_name in ("forward_policy", "source_policy", "edge_policy"):
        policy = hip_metadata[policy_name]
        assert "target_compute_capability" not in policy
        assert ".gpu-" in policy["tag"]
        assert policy["target_id"] == hip_metadata["target_id"]
    hip_policies = {
        policy_name: hip_metadata[policy_name]
        for policy_name in ("forward_policy", "source_policy", "edge_policy")
    }
    roundtrip_metadata = mh1_codegen.execution_mh1_hip_module_v4_metadata(
        contract, target, **hip_policies
    )
    roundtrip_plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(
        contract, target, **hip_policies
    )
    roundtrip_module = mh1_codegen.render_execution_mh1_hip_module_v4(
        contract, target, **hip_policies
    )
    assert roundtrip_metadata == hip_metadata
    assert roundtrip_plan == hip_plan
    assert roundtrip_module == hip_module
    stale_target_policy = copy.deepcopy(hip_metadata["forward_policy"])
    stale_target_policy["target_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="target does not match"):
        mh1_codegen.execution_mh1_hip_module_v4_metadata(
            contract, target, forward_policy=stale_target_policy
        )
    assert cuda_metadata["edge_reverse_schedule"] == "hybrid-path-tiled-v7"
    assert hip_metadata["edge_reverse_schedule"] == "hybrid-path-tiled-v6"
    assert (
        cuda_metadata["edge_policy"]["reverse_schedule"]
        == cuda_metadata["edge_reverse_schedule"]
    )
    assert (
        hip_metadata["edge_policy"]["reverse_schedule"]
        == hip_metadata["edge_reverse_schedule"]
    )
    assert cuda_metadata["node_state_policy"] == "full-retention-v1"
    assert hip_metadata["node_state_policy"] == "full-retention-v1"
    assert cuda_plan["node_state_policy"] == "full-retention-v1"
    assert hip_plan["node_state_policy"] == "full-retention-v1"
    assert cuda_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_XWIDE_ALL_FORWARD_LINEAR2_SCHEDULE
    )
    assert hip_plan["node_reverse_schedule"] == retained_node_schedule
    assert cuda_recompute_plan["node_state_policy"] == "recompute-v1"
    assert hip_recompute_plan["node_state_policy"] == "recompute-v1"
    assert cuda_recompute_plan["node_reverse_schedule"] == (
        mh1_codegen._MH1_CUDA_RECOMPUTE_TILED_PRODUCT_REVERSE_SCHEDULE
    )
    assert hip_recompute_plan["node_reverse_schedule"] == node_schedule
    assert cuda_recompute_plan["module_identity"] != cuda_plan["module_identity"]
    assert hip_recompute_plan["module_identity"] != hip_plan["module_identity"]
    assert (
        cuda_metadata["conditioner_reverse_schedule"]
        == hip_metadata["conditioner_reverse_schedule"]
    )
    assert all(
        interaction["conditioned_prefix"] == "bounded-recompute-split-v2"
        and interaction["density"] == "bounded-recompute-split-v2"
        for interaction in cuda_metadata["conditioner_reverse_schedule"]["interactions"]
    )
    conditioner_name = "execution_mh1_conditioned_prefix_radial_reverse_0"
    for device_source in (cuda_plugin, cuda_module, hip_module):
        function_begin = device_source.index(f"void {conditioner_name}(")
        function_end = device_source.index("\n}\n", function_begin)
        assert "float current_adjoint[5];" in device_source[function_begin:function_end]
    for module_source in (cuda_module, hip_module):
        assert module_source.count("conditioning_reverse_prefix_kernel_") == 2
        assert module_source.count("conditioning_reverse_density_kernel_") == 2
    host_begin = host_plugin.index(f"void {conditioner_name}(")
    host_end = host_plugin.index("\n}\n", host_begin)
    assert "float current_adjoint[5];" not in host_plugin[host_begin:host_end]
    assert "float activation_1[5];" in host_plugin[host_begin:host_end]
    assert cuda_plan["interactions"] == hip_plan["interactions"]
    assert all(
        interaction["edge_strategy"] == "compact_fused"
        and interaction["edge_phi_block_multiplier"] == 4
        for interaction in hip_plan["interactions"]
    )
    assert cuda_plan["nodes"] == hip_plan["nodes"]
    cuda_node_kernels = {
        line.replace("__launch_bounds__(128, 4) ", "")
        .split("(", 1)[0]
        .rsplit(" ", 1)[-1]
        for line in cuda_module.splitlines()
        if line.startswith('extern "C" __global__ void ')
        and "symmetrix_execution_mh1_node_" in line
    }
    hip_node_kernels = {
        line.split("(", 1)[0].rsplit(" ", 1)[-1]
        for line in hip_module.splitlines()
        if line.startswith('extern "C" __global__ void ')
        and "symmetrix_execution_mh1_node_" in line
    }
    assert cuda_node_kernels == hip_node_kernels

    # Keep cross-dialect coverage for the legacy schedule used by recompute.
    legacy_hip_metadata = mh1_codegen.jit_mh1_cuda_plugin_v4_metadata(
        contract,
        80,
        _edge_reverse_schedule=mh1_codegen._MH1_HIP_EDGE_REVERSE_SCHEDULE,
        _node_reverse_schedule=mh1_codegen._MH1_NODE_REVERSE_SCHEDULE,
    )
    hip_module = mh1_codegen._render_execution_mh1_gpu_module_v4(
        contract,
        80,
        mh1_codegen.MH1_HIP_DIALECT,
        mh1_codegen._execution_mh1_cuda_module_identity(legacy_hip_metadata),
        edge_reverse_schedule=mh1_codegen._MH1_HIP_EDGE_REVERSE_SCHEDULE,
        node_reverse_schedule=mh1_codegen._MH1_NODE_REVERSE_SCHEDULE,
    )
    hip_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract,
        80,
        _edge_reverse_schedule=mh1_codegen._MH1_HIP_EDGE_REVERSE_SCHEDULE,
        _node_reverse_schedule=mh1_codegen._MH1_NODE_REVERSE_SCHEDULE,
    )
    assert cuda_plan["nodes"][0]["pre_forward"] == hip_plan["nodes"][0]["pre_forward"]
    assert cuda_plan["nodes"][1]["pre_forward"][0]["grid"] == "node_channel_tiles"
    assert cuda_plan["nodes"][1]["pre_forward"][0]["kernel"].endswith(
        "_pre_forward_linear_up_grouped"
    )
    assert hip_plan["nodes"][1]["pre_forward"] == [
        {
            "kernel": "symmetrix_execution_mh1_node_pre_forward_kernel_1",
            "grid": "persistent_nodes",
            "threads": 128,
        }
    ]
    for cuda_node, hip_node in zip(cuda_plan["nodes"], hip_plan["nodes"], strict=True):
        assert [
            launch
            for launch in cuda_node["post_forward"]
            if "forward_linear2" not in launch["kernel"]
            and "forward_message" not in launch["kernel"]
            and "forward_residual" not in launch["kernel"]
            and "forward_product" not in launch["kernel"]
            and "forward_skip" not in launch["kernel"]
        ] == [
            launch
            for launch in hip_node["post_forward"]
            if "forward_linear2" not in launch["kernel"]
            and "forward_message" not in launch["kernel"]
            and "forward_residual" not in launch["kernel"]
            and "forward_product" not in launch["kernel"]
            and "forward_skip" not in launch["kernel"]
        ]
        assert (
            sum(
                "forward_message" in launch["kernel"]
                for launch in cuda_node["post_forward"]
            )
            == 1
        )
        assert (
            sum(
                "forward_message" in launch["kernel"]
                for launch in hip_node["post_forward"]
            )
            == 4
        )
        assert (
            sum(
                "forward_residual" in launch["kernel"]
                for launch in cuda_node["post_forward"]
            )
            == 1
        )
        assert (
            sum(
                "forward_residual" in launch["kernel"]
                for launch in hip_node["post_forward"]
            )
            == 4
        )
        assert (
            sum(
                "forward_linear2" in launch["kernel"]
                for launch in cuda_node["post_forward"]
            )
            == 1
        )
        assert (
            sum(
                "forward_linear2" in launch["kernel"]
                for launch in hip_node["post_forward"]
            )
            == 4
        )
        assert (
            sum(
                "reverse_linear2" in launch["kernel"]
                for launch in cuda_node["post_reverse"]
            )
            == 1
        )
        assert (
            sum(
                "reverse_linear2" in launch["kernel"]
                for launch in hip_node["post_reverse"]
            )
            == 4
        )
        assert (
            sum(
                "reverse_message" in launch["kernel"]
                for launch in cuda_node["post_reverse"]
            )
            == 1
        )
        assert (
            sum(
                "reverse_message" in launch["kernel"]
                for launch in hip_node["post_reverse"]
            )
            == 4
        )
    assert [
        launch
        for launch in cuda_plan["nodes"][0]["post_reverse"]
        if "reverse_linear2" not in launch["kernel"]
        and "reverse_message" not in launch["kernel"]
        and "reverse_residual" not in launch["kernel"]
        and "reverse_product_linear" not in launch["kernel"]
        and not launch["kernel"].endswith("_reverse_product_tiled")
    ] == [
        launch
        for launch in hip_plan["nodes"][0]["post_reverse"]
        if "reverse_linear2" not in launch["kernel"]
        and "reverse_message" not in launch["kernel"]
        and "reverse_residual" not in launch["kernel"]
        and "reverse_product_linear" not in launch["kernel"]
        and not launch["kernel"].endswith("_reverse_product")
    ]
    cuda_layer_one_reverse = cuda_plan["nodes"][1]["post_reverse"]
    hip_layer_one_reverse = hip_plan["nodes"][1]["post_reverse"]
    cuda_messages = [
        launch
        for launch in cuda_layer_one_reverse
        if "reverse_message" in launch["kernel"]
    ]
    hip_messages = [
        launch
        for launch in hip_layer_one_reverse
        if "reverse_message" in launch["kernel"]
    ]
    assert len(cuda_messages) == 1
    assert len(hip_messages) == 4
    cuda_readout = [
        launch
        for launch in cuda_layer_one_reverse
        if "reverse_readout" in launch["kernel"]
        or "reverse_skip_grouped" in launch["kernel"]
    ]
    hip_readout = [
        launch
        for launch in hip_layer_one_reverse
        if "reverse_readout" in launch["kernel"]
    ]
    assert [launch["grid"] for launch in cuda_readout] == [
        "node_channel_tiles",
        "persistent_nodes",
        "node_channel_tiles",
        "node_channel_tiles",
    ]
    assert len(hip_readout) == 1
    assert [
        launch
        for launch in cuda_layer_one_reverse
        if "reverse_message" not in launch["kernel"]
        and "reverse_linear2" not in launch["kernel"]
        and "reverse_residual" not in launch["kernel"]
        and "reverse_readout" not in launch["kernel"]
        and "reverse_skip_grouped" not in launch["kernel"]
        and "reverse_product_linear" not in launch["kernel"]
        and not launch["kernel"].endswith("_reverse_product_tiled")
    ] == [
        launch
        for launch in hip_layer_one_reverse
        if "reverse_message" not in launch["kernel"]
        and "reverse_linear2" not in launch["kernel"]
        and "reverse_residual" not in launch["kernel"]
        and "reverse_readout" not in launch["kernel"]
        and "reverse_product_linear" not in launch["kernel"]
        and not launch["kernel"].endswith("_reverse_product")
    ]
    assert (
        sum("reverse_residual" in launch["kernel"] for launch in cuda_layer_one_reverse)
        == 2
    )
    assert (
        sum("reverse_residual" in launch["kernel"] for launch in hip_layer_one_reverse)
        == 2
    )
    for cuda_node, hip_node in zip(cuda_plan["nodes"], hip_plan["nodes"], strict=True):
        cuda_product = [
            launch
            for launch in cuda_node["post_reverse"]
            if launch["kernel"].endswith("_reverse_product_tiled")
        ]
        hip_product = [
            launch
            for launch in hip_node["post_reverse"]
            if launch["kernel"].endswith("_reverse_product")
        ]
        assert len(cuda_product) == len(hip_product) == 1
        assert cuda_product[0]["grid"] == "node_channel_tiles"
        assert hip_product[0]["grid"] == "persistent_nodes"
        cuda_forward_product = [
            launch
            for launch in cuda_node["post_forward"]
            if launch["kernel"].endswith("_forward_product_setup")
            or launch["kernel"].endswith("_forward_product_tiled")
        ]
        hip_forward_product = [
            launch
            for launch in hip_node["post_forward"]
            if launch["kernel"].endswith("_forward_product")
        ]
        assert len(cuda_forward_product) == 2
        assert {launch["grid"] for launch in cuda_forward_product} == {
            "persistent_nodes",
            "node_channel_tiles",
        }
        assert len(hip_forward_product) == 1
        assert hip_forward_product[0]["grid"] == "persistent_nodes"
    assert all(
        {launch["grid"] for launch in node["post_forward"]}
        == {"node_channel_tiles", "persistent_nodes"}
        for node in hip_plan["nodes"]
    )
    for node in hip_plan["nodes"]:
        forward_kernels = [launch["kernel"] for launch in node["post_forward"]]
        assert any("_forward_residual_block_" in name for name in forward_kernels)
        assert any("_forward_message_block_" in name for name in forward_kernels)
        reverse_kernels = [launch["kernel"] for launch in node["post_reverse"]]
        assert node["post_reverse"][0]["grid"] == "persistent_nodes"
        assert reverse_kernels[0].endswith("_reverse_readout")
        assert reverse_kernels[-1].endswith("_reverse_density")
        assert not any(
            "_reverse_residual_forward_block_" in name for name in reverse_kernels
        )
        assert not any(
            "_reverse_message_forward_block_" in name for name in reverse_kernels
        )
        assert not any(
            "_reverse_linear2_forward_block_" in name for name in reverse_kernels
        )
        assert not any("_reverse_gate_forward" in name for name in reverse_kernels)
        assert not any(
            "_reverse_residual_recompute_block_" in name for name in reverse_kernels
        )
        assert not any(
            "_reverse_message_recompute_block_" in name for name in reverse_kernels
        )
    for index, (cuda_node, hip_node) in enumerate(
        zip(cuda_recompute_plan["nodes"], hip_recompute_plan["nodes"], strict=True)
    ):
        cuda_messages = [
            launch
            for launch in cuda_node["post_reverse"]
            if "_reverse_message_block_" in launch["kernel"]
            or launch["kernel"].endswith("_reverse_message_grouped")
        ]
        hip_messages = [
            launch
            for launch in hip_node["post_reverse"]
            if "_reverse_message_block_" in launch["kernel"]
            or launch["kernel"].endswith("_reverse_message_grouped")
        ]
        assert len(cuda_messages) == 1
        assert len(hip_messages) == 4
        cuda_linear2 = [
            launch
            for launch in cuda_node["post_reverse"]
            if "_reverse_linear2_block_" in launch["kernel"]
            or launch["kernel"].endswith("_reverse_linear2_grouped")
        ]
        hip_linear2 = [
            launch
            for launch in hip_node["post_reverse"]
            if "_reverse_linear2_block_" in launch["kernel"]
            or launch["kernel"].endswith("_reverse_linear2_grouped")
        ]
        assert len(cuda_linear2) == 1
        assert len(hip_linear2) == 4
        cuda_residual = [
            launch
            for launch in cuda_node["post_reverse"]
            if "_reverse_residual_block_" in launch["kernel"]
            or launch["kernel"].endswith("_reverse_residual_grouped")
        ]
        hip_residual = [
            launch
            for launch in hip_node["post_reverse"]
            if "_reverse_residual_block_" in launch["kernel"]
            or launch["kernel"].endswith("_reverse_residual_grouped")
        ]
        assert len(cuda_residual) == (1 if index == 1 else 0)
        assert len(hip_residual) == (2 if index == 1 else 0)
        cuda_product = [
            launch
            for launch in cuda_node["post_reverse"]
            if launch["kernel"].endswith("_reverse_product_tiled")
        ]
        hip_product = [
            launch
            for launch in hip_node["post_reverse"]
            if launch["kernel"].endswith("_reverse_product")
        ]
        assert len(cuda_product) == len(hip_product) == 1
        assert cuda_product[0]["grid"] == "node_channel_tiles"
        assert hip_product[0]["grid"] == "persistent_nodes"
        assert [
            launch
            for launch in cuda_node["post_reverse"]
            if "_reverse_message_block_" not in launch["kernel"]
            and not launch["kernel"].endswith("_reverse_message_grouped")
            and "_reverse_linear2_block_" not in launch["kernel"]
            and not launch["kernel"].endswith("_reverse_linear2_grouped")
            and "_reverse_residual_block_" not in launch["kernel"]
            and not launch["kernel"].endswith("_reverse_residual_grouped")
            and not launch["kernel"].endswith("_reverse_product_tiled")
        ] == [
            launch
            for launch in hip_node["post_reverse"]
            if "_reverse_message_block_" not in launch["kernel"]
            and not launch["kernel"].endswith("_reverse_message_grouped")
            and "_reverse_linear2_block_" not in launch["kernel"]
            and not launch["kernel"].endswith("_reverse_linear2_grouped")
            and "_reverse_residual_block_" not in launch["kernel"]
            and not launch["kernel"].endswith("_reverse_residual_grouped")
            and not launch["kernel"].endswith("_reverse_product")
        ]
        reverse_kernels = [launch["kernel"] for launch in cuda_node["post_reverse"]]
        for family in (
            "reverse_residual_forward_block_",
            "reverse_message_forward_block_",
            "reverse_linear2_forward_block_",
            "reverse_gate_forward",
            "reverse_residual_recompute_block_",
            "reverse_message_recompute_block_",
        ):
            assert any(family in name for name in reverse_kernels)
    cuda_kernels = {
        line.replace("__launch_bounds__(128, 4) ", "")
        .split("(", 1)[0]
        .rsplit(" ", 1)[-1]
        for line in cuda_module.splitlines()
        if line.startswith('extern "C" __global__ void ')
    }
    hip_kernels = {
        line.split("(", 1)[0].rsplit(" ", 1)[-1]
        for line in hip_module.splitlines()
        if line.startswith('extern "C" __global__ void ')
    }
    cuda_message_kernels = {
        name
        for name in cuda_kernels
        if "forward_message_block_" in name
        or "reverse_message_block_" in name
        or name.endswith("forward_message_grouped")
        or name.endswith("reverse_message_grouped")
    }
    hip_message_kernels = {
        name
        for name in hip_kernels
        if "forward_message_block_" in name
        or "reverse_message_block_" in name
        or name.endswith("forward_message_grouped")
        or name.endswith("reverse_message_grouped")
    }
    cuda_linear2_kernels = {
        name
        for name in cuda_kernels
        if "forward_linear2_block_" in name
        or "reverse_linear2_block_" in name
        or name.endswith("forward_linear2_grouped")
        or name.endswith("reverse_linear2_grouped")
    }
    hip_linear2_kernels = {
        name
        for name in hip_kernels
        if "forward_linear2_block_" in name
        or "reverse_linear2_block_" in name
        or name.endswith("forward_linear2_grouped")
        or name.endswith("reverse_linear2_grouped")
    }
    cuda_residual_kernels = {
        name
        for name in cuda_kernels
        if "forward_residual_block_" in name
        or "reverse_residual_block_" in name
        or name.endswith("forward_residual_grouped")
        or name.endswith("reverse_residual_grouped")
    }
    hip_residual_kernels = {
        name
        for name in hip_kernels
        if "forward_residual_block_" in name
        or "reverse_residual_block_" in name
        or name.endswith("forward_residual_grouped")
        or name.endswith("reverse_residual_grouped")
    }
    cuda_reverse_product_kernels = {
        name for name in cuda_kernels if name.endswith("_reverse_product_tiled")
    }
    hip_reverse_product_kernels = {
        name for name in hip_kernels if name.endswith("_reverse_product")
    }
    cuda_forward_product_kernels = {
        name
        for name in cuda_kernels
        if name.endswith("_forward_product_setup")
        or name.endswith("_forward_product_tiled")
    }
    hip_forward_product_kernels = {
        name for name in hip_kernels if name.endswith("_forward_product")
    }
    cuda_pre_forward_kernels = {
        name for name in cuda_kernels if name.endswith("_pre_forward_linear_up_grouped")
    }
    cuda_skip_forward_kernels = {
        name for name in cuda_kernels if name.endswith("_forward_skip_grouped")
    }
    hip_pre_forward_kernels = {
        name for name in hip_kernels if name.endswith("_node_pre_forward_kernel_1")
    }
    cuda_pre_reverse_kernels = {name for name in cuda_kernels if "pre_reverse" in name}
    hip_pre_reverse_kernels = {name for name in hip_kernels if "pre_reverse" in name}
    cuda_readout_kernels = {
        name
        for name in cuda_kernels
        if "_l1_reverse_readout" in name or name.endswith("_l1_reverse_skip_grouped")
    }
    hip_readout_kernels = {
        name for name in hip_kernels if "_l1_reverse_readout" in name
    }
    assert cuda_message_kernels == {
        "symmetrix_execution_mh1_node_tiled_l0_forward_message_grouped",
        "symmetrix_execution_mh1_node_tiled_l0_reverse_message_grouped",
        "symmetrix_execution_mh1_node_tiled_l1_forward_message_grouped",
        "symmetrix_execution_mh1_node_tiled_l1_reverse_message_grouped",
    }
    assert len(hip_message_kernels) == 16
    assert len(cuda_linear2_kernels) == 4
    assert len(hip_linear2_kernels) == 16
    assert cuda_residual_kernels == {
        "symmetrix_execution_mh1_node_tiled_l0_forward_residual_grouped",
        "symmetrix_execution_mh1_node_tiled_l1_forward_residual_grouped",
        "symmetrix_execution_mh1_node_tiled_l1_reverse_residual_block_0",
        "symmetrix_execution_mh1_node_tiled_l1_reverse_residual_block_1",
    }
    assert len(hip_residual_kernels) == 10
    assert cuda_reverse_product_kernels == {
        "symmetrix_execution_mh1_node_tiled_l0_reverse_product_tiled",
        "symmetrix_execution_mh1_node_tiled_l1_reverse_product_tiled",
    }
    assert len(hip_reverse_product_kernels) == 2
    assert cuda_forward_product_kernels == {
        "symmetrix_execution_mh1_node_tiled_l0_forward_product_setup",
        "symmetrix_execution_mh1_node_tiled_l0_forward_product_tiled",
        "symmetrix_execution_mh1_node_tiled_l1_forward_product_setup",
        "symmetrix_execution_mh1_node_tiled_l1_forward_product_tiled",
    }
    assert len(hip_forward_product_kernels) == 2
    assert cuda_pre_forward_kernels == {
        "symmetrix_execution_mh1_node_tiled_l1_pre_forward_linear_up_grouped"
    }
    assert cuda_skip_forward_kernels == {
        "symmetrix_execution_mh1_node_tiled_l0_forward_skip_grouped",
        "symmetrix_execution_mh1_node_tiled_l1_forward_skip_grouped",
    }
    assert hip_pre_forward_kernels == {
        "symmetrix_execution_mh1_node_pre_forward_kernel_1"
    }
    assert cuda_pre_reverse_kernels == {
        "symmetrix_execution_mh1_node_tiled_l1_pre_reverse_linear_up_grouped"
    }
    assert hip_pre_reverse_kernels == {
        "symmetrix_execution_mh1_node_pre_reverse_kernel_1"
    }
    assert cuda_readout_kernels == {
        "symmetrix_execution_mh1_node_tiled_l1_reverse_readout1_recompute_grouped",
        "symmetrix_execution_mh1_node_tiled_l1_reverse_readout_middle",
        "symmetrix_execution_mh1_node_tiled_l1_reverse_readout1_grouped",
        "symmetrix_execution_mh1_node_tiled_l1_reverse_skip_grouped",
    }
    assert hip_readout_kernels == {
        "symmetrix_execution_mh1_node_tiled_l1_reverse_readout"
    }
    assert (
        hip_kernels
        - hip_message_kernels
        - hip_linear2_kernels
        - hip_residual_kernels
        - hip_reverse_product_kernels
        - hip_forward_product_kernels
        - hip_pre_forward_kernels
        - hip_pre_reverse_kernels
        - hip_readout_kernels
        == cuda_kernels
        - cuda_message_kernels
        - cuda_linear2_kernels
        - cuda_residual_kernels
        - cuda_reverse_product_kernels
        - cuda_forward_product_kernels
        - cuda_pre_forward_kernels
        - cuda_skip_forward_kernels
        - cuda_pre_reverse_kernels
        - cuda_readout_kernels
    )


def test_mh1_v4_gpu_module_fp64_is_scalar_qualified(mh1_node_definition):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    source = mh1_codegen.render_execution_mh1_cuda_module_v4(
        contract, 80, precision="float64"
    )
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract, 80, precision="float64"
    )
    fp32_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)

    assert "const double* node_density;" in source
    assert "double* node_arena;" in source
    assert "using Scalar = double;" in source
    assert "float energy_scale;" in source
    assert plan["scalar_size"] == 8
    assert fp32_plan["scalar_size"] == 4
    assert plan["module_identity"] != fp32_plan["module_identity"]
    assert plan["module_identity"] in source
    with pytest.raises(ValueError, match="precision"):
        mh1_codegen.render_execution_mh1_cuda_module_v4(
            contract, 80, precision="float16"
        )


def test_mh1_v4_path_tiled_edge_geometry_is_shared_by_cuda_and_hip(
    mh1_node_definition,
):
    definition = copy.deepcopy(mh1_node_definition)
    path_counts = (4, 10)
    for interaction, path_count in zip(
        definition["interactions"], path_counts, strict=True
    ):
        instruction = copy.deepcopy(interaction["conv_tp"]["instructions"][-1])
        if path_count == 4:
            instruction["wigner_3j"]["values"] = [1.0] * math.prod(
                instruction["wigner_3j"]["shape"]
            )
        interaction["conv_tp"]["instructions"] = [
            copy.deepcopy(instruction) for _ in range(path_count)
        ]
        output_mask = interaction["conv_tp"]["output_mask"]["values"]
        covered_output = (
            math.prod(instruction["path_shape"]) * instruction["wigner_3j"]["shape"][2]
        )
        interaction["conv_tp"]["output_mask"]["values"] = [0.0] * (
            len(output_mask) - covered_output
        ) + [1.0] * covered_output
        final_linear = interaction["conv_tp_weights"]["layers"][-1]
        hidden_dimension = final_linear["weight"]["shape"][1]
        final_linear["weight"] = _tensor([path_count, hidden_dimension], 0.3)
        final_linear["bias"] = _tensor([path_count], 0.4)

    contract = make_execution_mh1_v4_contract(definition)
    cuda_module = mh1_codegen.render_execution_mh1_cuda_module_v4(contract, 80)
    hip_module = mh1_codegen.render_execution_mh1_hip_module_v4(
        contract,
        {
            "architecture": "gfx1151",
            "target_features": ["sramecc-", "xnack-"],
            "compiler_offload_target": "gfx1151",
            "native_subgroup_width": 32,
            "compute_unit_count": 40,
        },
    )
    cuda_plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(contract, 80)
    hip_plan = mh1_codegen.execution_mh1_hip_module_v4_launch_plan(
        contract,
        {
            "architecture": "gfx1151",
            "target_features": ["sramecc-", "xnack-"],
            "compiler_offload_target": "gfx1151",
            "native_subgroup_width": 32,
            "compute_unit_count": 40,
        },
    )

    for plan in (cuda_plan, hip_plan):
        assert all(
            interaction["edge_phi_schedule"] == "path_tiled"
            and interaction["edge_phi_block_multiplier"] == 4
            for interaction in plan["interactions"]
        )
    for module in (cuda_module, hip_module):
        assert "__shared__ float path_weight_adjoint[4][4]" in module
        assert "path_weight_adjoint[3][3][channel]" in module
        assert "static_cast<std::int64_t>(blockIdx.x) * 4" in module
        assert "local_edge_3 = edge_base + static_cast<std::int64_t>(3);" in module
        assert "edge_base += static_cast<std::int64_t>(gridDim.x) * 4" in module
        for interaction_index, expected_barriers in enumerate((11, 25)):
            symbol = (
                'extern "C" __global__ void '
                "symmetrix_execution_mh1_edge_reverse_compact_fused_kernel_"
                f"{interaction_index}("
            )
            start = module.index(symbol)
            end = module.find('\nextern "C" __global__ void ', start + len(symbol))
            kernel = module[start : end if end >= 0 else len(module)]
            assert kernel.count("__syncthreads();") == expected_barriers


def test_mh1_v4_generated_ir_mul_linear_forward_reverse(mh1_node_definition, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for node linear generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    helpers = render_execution_mh1_node_linear_cpp_helpers(contract)
    assert "execution_mh1_node_layer_1_linear_up_forward" in helpers
    assert "execution_mh1_node_layer_1_linear_up_reverse" in helpers
    assert "parameter_adjoint" not in helpers

    source = tmp_path / "node_linear.cpp"
    library = tmp_path / "node_linear.so"
    source.write_text(
        helpers
        + r"""
extern "C" void evaluate(
    const float* input, const float* parameters, float* output)
{
    execution_mh1_node_layer_1_linear_up_forward(input, parameters, output);
}

extern "C" void reverse(
    const float* output_adjoint, const float* parameters,
    float* input_adjoint)
{
    execution_mh1_node_layer_1_linear_up_reverse(
        output_adjoint, parameters, input_adjoint);
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    loaded.evaluate.argtypes = [pointer, pointer, pointer]
    loaded.reverse.argtypes = [pointer, pointer, pointer]
    input_values = (ctypes.c_float * 4)(0.4, -0.2, 0.3, 1.1)
    # Two path weights followed by the four output-mask values.
    parameters = (ctypes.c_float * 6)(0.25, 0.25, 1.0, 1.0, 1.0, 1.0)
    output = (ctypes.c_float * 4)()
    loaded.evaluate(input_values, parameters, output)
    assert list(output) == pytest.approx([0.1, -0.05, 0.075, 0.275])

    output_adjoint = (ctypes.c_float * 4)(0.7, -0.4, 0.2, 0.8)
    input_adjoint = (ctypes.c_float * 4)()
    loaded.reverse(output_adjoint, parameters, input_adjoint)
    assert list(input_adjoint) == pytest.approx([0.175, -0.1, 0.05, 0.2])


def test_mh1_v4_generated_ir_mul_linear_uses_mul_ir_parameter_layout(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for node linear generation")
    descriptor = {
        "dimensions": {"input": 6, "output": 6},
        "blocks": {
            "input": [{"offset": 0, "multiplicity": 2, "components": 3}],
            "output": [{"offset": 0, "multiplicity": 2, "components": 3}],
        },
        "instructions": [
            {
                "input_block": 0,
                "output_block": 0,
                "path_weight": 1.0,
                "weight_offset": 0,
            }
        ],
        "runtime_parameters": {
            "segments": [
                {"name": "weight", "offset": 0, "count": 4},
                {"name": "bias", "offset": 4, "count": 6},
                {"name": "output_mask", "offset": 10, "count": 6},
            ]
        },
    }
    helpers = mh1_codegen._render_ir_mul_linear_helpers(
        descriptor,
        name="test_linear",
        function_qualifier="static inline",
    )
    source = tmp_path / "mul_ir_linear.cpp"
    library = tmp_path / "mul_ir_linear.so"
    source.write_text(
        helpers
        + r"""
extern "C" void evaluate(
    const float* input, const float* parameters, float* output)
{
    test_linear_forward(input, parameters, output);
}

extern "C" void reverse(
    const float* output_adjoint, const float* parameters,
    float* input_adjoint)
{
    test_linear_reverse(output_adjoint, parameters, input_adjoint);
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    loaded.evaluate.argtypes = [pointer, pointer, pointer]
    loaded.reverse.argtypes = [pointer, pointer, pointer]
    input_values = (ctypes.c_float * 6)(1.0, 2.0, 3.0, 10.0, 20.0, 30.0)
    # Weights are row-major, while bias and mask use [channel, component].
    parameters = (ctypes.c_float * 16)(
        1.0,
        0.0,
        0.0,
        1.0,
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )
    output = (ctypes.c_float * 6)()
    loaded.evaluate(input_values, parameters, output)
    assert list(output) == pytest.approx([101.0, 1608.0, 406.0, 2550.0, 960.0, 3780.0])

    output_adjoint = (ctypes.c_float * 6)(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    input_adjoint = (ctypes.c_float * 6)()
    loaded.reverse(output_adjoint, parameters, input_adjoint)
    assert list(input_adjoint) == pytest.approx([1.0, 8.0, 6.0, 20.0, 15.0, 36.0])


def test_mh1_v4_generated_product_forward_reverse(mh1_node_definition, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for node product generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    helpers = render_execution_mh1_node_product_cpp_helpers(contract)
    assert "execution_mh1_node_layer_0_product_forward" in helpers
    assert "execution_mh1_node_layer_0_product_reverse" in helpers
    assert "parameter_adjoint" not in helpers

    source = tmp_path / "node_product.cpp"
    library = tmp_path / "node_product.so"
    source.write_text(
        helpers
        + r"""
extern "C" void evaluate(
    const float* input, const float* skip, const float* coefficients,
    const float* linear_parameters, float* contracted, float* output)
{
    execution_mh1_node_layer_0_product_forward(
        input, skip, coefficients, linear_parameters, contracted, output);
}

extern "C" void reverse(
    const float* input, const float* output_adjoint,
    const float* coefficients, const float* linear_parameters,
    float* contracted_adjoint, float* input_adjoint, float* skip_adjoint)
{
    execution_mh1_node_layer_0_product_reverse(
        input, output_adjoint, coefficients, linear_parameters,
        contracted_adjoint, input_adjoint, skip_adjoint);
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    loaded.evaluate.argtypes = [pointer] * 6
    loaded.reverse.argtypes = [pointer] * 7
    input_values = (ctypes.c_float * 16)(2.0, *([0.0] * 15))
    skip = (ctypes.c_float * 4)(0.1, 0.2, 0.3, 0.4)
    coefficients = (ctypes.c_float * 6)(*([0.5] * 6))
    linear_parameters = (ctypes.c_float * 6)(0.25, 0.25, 1.0, 1.0, 1.0, 1.0)
    contracted = (ctypes.c_float * 4)()
    output = (ctypes.c_float * 4)()
    loaded.evaluate(
        input_values,
        skip,
        coefficients,
        linear_parameters,
        contracted,
        output,
    )
    assert list(contracted) == pytest.approx([7.0, 7.0, 0.0, 0.0])
    assert list(output) == pytest.approx([1.85, 1.95, 0.3, 0.4])

    output_adjoint = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
    contracted_adjoint = (ctypes.c_float * 4)()
    input_adjoint = (ctypes.c_float * 16)()
    skip_adjoint = (ctypes.c_float * 4)()
    loaded.reverse(
        input_values,
        output_adjoint,
        coefficients,
        linear_parameters,
        contracted_adjoint,
        input_adjoint,
        skip_adjoint,
    )
    assert list(contracted_adjoint) == pytest.approx([0.25] * 4)
    assert input_adjoint[0] == pytest.approx(4.25)
    assert list(input_adjoint)[1:] == pytest.approx([0.0] * 15)
    assert list(skip_adjoint) == pytest.approx([1.0] * 4)


def test_mh1_v4_generated_gate_forward_reverse(mh1_node_definition, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for node gate generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    helpers = render_execution_mh1_node_nonlinear_cpp_helpers(contract)
    source = tmp_path / "node_gate.cpp"
    library = tmp_path / "node_gate.so"
    source.write_text(
        "#include <cmath>\n"
        + helpers
        + r"""
extern "C" void evaluate(const float* input, float* output)
{
    execution_mh1_node_layer_0_gate_forward(input, output);
}
extern "C" void reverse(
    const float* input, const float* output_adjoint, float* input_adjoint)
{
    execution_mh1_node_layer_0_gate_reverse(
        input, output_adjoint, input_adjoint);
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    loaded.evaluate.argtypes = [pointer, pointer]
    loaded.reverse.argtypes = [pointer, pointer, pointer]
    values = [0.2, 0.1, -0.2, 0.3] + [0.05 * index for index in range(1, 16)]
    input_values = (ctypes.c_float * 19)(*values)
    output = (ctypes.c_float * 16)()
    loaded.evaluate(input_values, output)

    scalar_scale = 1.6791767923989418
    gate_scale = 1.8467055342154763
    scalar_sigmoid = 1.0 / (1.0 + math.exp(-values[0]))
    expected = [scalar_scale * values[0] * scalar_sigmoid]
    gated_offset = 4
    for gate_index, width in enumerate((3, 5, 7)):
        sigmoid = 1.0 / (1.0 + math.exp(-values[1 + gate_index]))
        expected.extend(
            values[gated_offset + component] * gate_scale * sigmoid
            for component in range(width)
        )
        gated_offset += width
    assert list(output) == pytest.approx(expected, rel=2e-6)

    output_adjoint_values = [0.1 * (index + 1) for index in range(16)]
    output_adjoint = (ctypes.c_float * 16)(*output_adjoint_values)
    input_adjoint = (ctypes.c_float * 19)()
    loaded.reverse(input_values, output_adjoint, input_adjoint)
    expected_adjoint = [0.0] * 19
    expected_adjoint[0] = (
        output_adjoint_values[0]
        * scalar_scale
        * (scalar_sigmoid + values[0] * scalar_sigmoid * (1.0 - scalar_sigmoid))
    )
    input_offset = 4
    output_offset = 1
    for gate_index, width in enumerate((3, 5, 7)):
        sigmoid = 1.0 / (1.0 + math.exp(-values[1 + gate_index]))
        gate_value = gate_scale * sigmoid
        gate_adjoint = 0.0
        for component in range(width):
            adjoint = output_adjoint_values[output_offset + component]
            expected_adjoint[input_offset + component] = adjoint * gate_value
            gate_adjoint += adjoint * values[input_offset + component]
        expected_adjoint[1 + gate_index] = (
            gate_adjoint * gate_scale * sigmoid * (1.0 - sigmoid)
        )
        input_offset += width
        output_offset += width
    assert list(input_adjoint) == pytest.approx(expected_adjoint, rel=2e-6)


def test_mh1_v4_complete_node_program_compiles(mh1_node_definition, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for complete node generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    helpers = render_execution_mh1_node_program_cpp_helpers(contract)
    assert "execution_mh1_node_layer_0_pre_forward" in helpers
    assert "execution_mh1_node_layer_0_post_forward" in helpers
    assert "execution_mh1_node_layer_0_post_reverse" in helpers
    assert "execution_mh1_node_layer_0_pre_reverse" not in helpers
    assert "execution_mh1_node_layer_1_pre_reverse" in helpers
    assert "parameter_adjoint" not in helpers
    source = tmp_path / "node_program.cpp"
    library = tmp_path / "node_program.so"
    source.write_text("#include <cmath>\n" + helpers, encoding="ascii")
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mh1_v4_complete_host_plugin_compiles(mh1_node_definition, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for host plugin generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    generated = render_jit_mh1_host_plugin_v4(contract)
    assert "symmetrix_jit_mh1_host_plugin_query_v3" in generated
    assert "symmetrix_jit_mh1_host_plugin_query_v4" in generated
    assert "SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V4" in generated
    assert "SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V4" in generated
    assert "#if defined(__AVX512F__)" in generated
    assert "constexpr std::int32_t channel_tile = 16;" in generated
    assert "constexpr std::int32_t channel_tile = 8;" in generated
    assert "phi * " in generated
    assert "fused_source_edge_reverse_owner" in generated
    assert "&tiled_forward_owner, &fused_source_edge_reverse_owner" in generated
    source = tmp_path / "mh1_host_v4.cpp"
    library = tmp_path / "mh1_host_v4.so"
    source.write_text(generated, encoding="ascii")
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mh1_v4_host_plugin_advertises_interaction_retention(mh1_node_definition):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    flag = "SYMMETRIX_JIT_MH1_HOST_NODE_RETAIN_INTERACTION_OUTPUT_V4"
    recompute = render_jit_mh1_host_plugin_v4(
        contract, node_state_policy="recompute-v1"
    )
    retained = render_jit_mh1_host_plugin_v4(
        contract, node_state_policy="retain-interaction-v1"
    )

    assert recompute.count(flag) == 1
    assert retained.count(flag) == 3


def test_mh1_v5_spline_r_forward_plugin_compiles(mh1_node_definition, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for spline R generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    generated = render_jit_mh1_host_plugin_v5(contract)
    assert "symmetrix_jit_mh1_host_plugin_query_v3" in generated
    assert "symmetrix_jit_mh1_host_plugin_query_v4" in generated
    assert "symmetrix_jit_mh1_host_plugin_query_v5" in generated
    assert "SYMMETRIX_JIT_MH1_HOST_SPLINE_R_FORWARD_V5" in generated
    assert "SYMMETRIX_JIT_MH1_HOST_SPLINE_R_SOURCE_REVERSE_V5" in generated
    assert "spline_r_forward_0" in generated
    assert "spline_r_forward_1" in generated
    assert "spline_r_source_reverse_0" in generated
    assert "spline_r_source_reverse_1" in generated
    assert "float message[" in generated
    spline_source = generated[generated.index("static void spline_r_forward_0") :]
    assert "edge_phi" not in spline_source
    assert "args->output_mask[" not in spline_source
    assert "float source_contribution_0 = 0.0f;" in spline_source
    assert "weight_adjoint += source_reverse * source_value;" in spline_source
    assert "#pragma omp simd reduction(+:radial_contribution_" in spline_source
    assert "radial_adjoint += radial_contribution_" in spline_source
    assert generated.count("prefetch_read(") > 2
    assert "prefetch_read(target_node_output_adjoint" in generated
    assert "prefetch_read(edge_input_2" in generated
    assert "__builtin_prefetch(address, 0, 1);" in generated
    for index, interaction in enumerate(contract["interactions"]):
        multiplicity = interaction["paths"][0]["input_1"]["multiplicity"]
        path_channel_loop = (
            f"for (std::int32_t channel = 0;\n     channel < {multiplicity}; ++channel)"
        )
        forward_start = generated.index(f"static void spline_r_forward_{index}")
        reverse_start = generated.index(f"static void spline_r_source_reverse_{index}")
        forward_source = generated[forward_start:reverse_start]
        reverse_end = generated.find("\nstatic void ", reverse_start + 1)
        if reverse_end < 0:
            reverse_end = len(generated)
        reverse_source = generated[reverse_start:reverse_end]
        assert forward_source.count(path_channel_loop) == len(interaction["paths"])
        assert reverse_source.count(f"channel < {multiplicity}; ++channel)") == len(
            interaction["paths"]
        )
        assert reverse_source.count(
            "float* const edge_adjoint = edge_adjoints[edge_lane];"
        ) == len(interaction["paths"])
        assert "constexpr std::int32_t edge_tile = 128;" in reverse_source
    source = tmp_path / "mh1_host_v5.cpp"
    library = tmp_path / "mh1_host_v5.so"
    source.write_text(generated, encoding="ascii")
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )
    runner_source = tmp_path / "mh1_host_v5_runner.cpp"
    runner = tmp_path / "mh1_host_v5_runner"
    runner_source.write_text(
        r"""
#include "jit_mh1_host_plugin_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 2) return 1;
    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return 2;
    auto query_v3 = reinterpret_cast<SymmetrixJitMH1HostPluginQueryV3>(
        dlsym(handle, SYMMETRIX_JIT_MH1_HOST_PLUGIN_QUERY_SYMBOL));
    auto query_v5 = reinterpret_cast<SymmetrixJitMH1HostPluginQueryV5>(
        dlsym(handle, SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_QUERY_SYMBOL));
    if (query_v3 == nullptr || query_v5 == nullptr) return 3;
    const auto* v3 = query_v3();
    const auto* v5 = query_v5();
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        constexpr std::int32_t num_nodes = 2;
        const auto& extent = v3->interactions[layer];
        const auto& spline = v5->spline_r_programs[layer];
        if ((spline.flags
             & SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5) == 0)
            return 4;
        std::vector<float> source(
            static_cast<std::size_t>(num_nodes) * extent.input_1_dimension);
        std::vector<float> harmonic(
            static_cast<std::size_t>(num_nodes) * extent.input_2_dimension);
        std::vector<float> mask(extent.output_dimension, 1.0f);
        std::vector<float> phi(
            static_cast<std::size_t>(num_nodes) * extent.phi_dimension, 0.0f);
        std::vector<float> linear(
            static_cast<std::size_t>(extent.weight_size) * extent.phi_dimension,
            0.0f);
        std::vector<float> coefficients(
            static_cast<std::size_t>(4) * (extent.weight_size + 1), 0.0f);
        std::vector<float> reference(
            static_cast<std::size_t>(num_nodes) * extent.output_dimension, 0.0f);
        std::vector<float> actual(reference.size(), 0.0f);
        for (std::size_t i = 0; i < source.size(); ++i)
            source[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f;
        for (std::size_t i = 0; i < harmonic.size(); ++i)
            harmonic[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 8.0f;
        phi[0] = 1.0f;
        phi[extent.phi_dimension] = 1.0f;
        for (std::int32_t weight = 0; weight < extent.weight_size; ++weight) {
            const float value = static_cast<float>((weight % 13) - 6) / 32.0f;
            linear[static_cast<std::size_t>(weight) * extent.phi_dimension] = value;
            coefficients[weight] = value;
        }
        coefficients[extent.weight_size] = 0.75f;
        const std::int32_t source_index[] = {0, 1};
        const std::int32_t target_index[] = {0, 1};
        const std::int32_t receiver_offsets[] = {0, 1, 2};
        const std::int32_t source_types[] = {0, 0};
        const std::int32_t node_types[] = {0, 0};
        const float cutoff[] = {1.0f, 1.0f};
        const double distance[] = {0.25, 0.35};
        SymmetrixJitMH1HostForwardArgsV3 reference_args{
            sizeof(SymmetrixJitMH1HostForwardArgsV3), layer, 0u, 0u,
            num_nodes, num_nodes, 0, num_nodes,
            source_index, target_index, receiver_offsets,
            phi.data(), linear.data(), nullptr, nullptr, harmonic.data(),
            cutoff, mask.data(), source.data(), reference.data()};
        for (std::int32_t receiver = 0; receiver < num_nodes; ++receiver)
            v3->forward_owner(&reference_args, receiver);
        std::vector<float> density(num_nodes, 0.0f);
        SymmetrixJitMH1HostSplineRForwardArgsV5 spline_args{
            sizeof(SymmetrixJitMH1HostSplineRForwardArgsV5), layer, 0u, 0u,
            num_nodes, num_nodes, 1, 1, 1, extent.weight_size + 1, 1.0, 0.0,
            source_index, target_index, receiver_offsets, source_types, node_types,
            distance, coefficients.data(), harmonic.data(), mask.data(),
            source.data(), actual.data(), density.data()};
        for (std::int32_t receiver = 0; receiver < num_nodes; ++receiver)
            spline.forward_owner(&spline_args, receiver);
        const std::int32_t block_offsets[] = {0, 1, 4, 9};
        const std::int32_t block_widths[] = {1, 3, 5, 7};
        for (std::int32_t node = 0; node < num_nodes; ++node) {
            if (std::abs(density[node] - 0.75f) > 1.0e-7f) return 5;
            for (std::int32_t block = 0; block < 4; ++block)
                for (std::int32_t component = 0;
                     component < block_widths[block]; ++component) {
                    const std::size_t native_index =
                        static_cast<std::size_t>(node) * extent.output_dimension
                        + block_offsets[block] + component;
                    const std::size_t packed_index =
                        static_cast<std::size_t>(num_nodes) * block_offsets[block]
                        + node * block_widths[block] + component;
                    if (std::abs(actual[packed_index] - reference[native_index])
                        > 2.0e-5f)
                        return 6;
                }
        }

        std::vector<float> output_adjoint(actual.size(), 0.0f);
        std::vector<float> native_output_adjoint(actual.size(), 0.0f);
        for (std::int32_t node = 0; node < num_nodes; ++node)
            for (std::int32_t block = 0; block < 4; ++block)
                for (std::int32_t component = 0;
                     component < block_widths[block]; ++component) {
                    const std::size_t native_index =
                        static_cast<std::size_t>(node) * extent.output_dimension
                        + block_offsets[block] + component;
                    const std::size_t packed_index =
                        static_cast<std::size_t>(num_nodes) * block_offsets[block]
                        + node * block_widths[block] + component;
                    output_adjoint[packed_index] =
                        static_cast<float>(
                            static_cast<int>(native_index % 11) - 5) / 16.0f;
                    native_output_adjoint[native_index] =
                        output_adjoint[packed_index];
                }
        std::vector<float> source_adjoint(source.size(), 0.0f);
        std::vector<float> harmonic_adjoint(harmonic.size(), 0.0f);
        std::vector<float> source_direction(source.size());
        std::vector<float> harmonic_direction(harmonic.size());
        for (std::size_t i = 0; i < source.size(); ++i)
            source_direction[i] =
                static_cast<float>(static_cast<int>(i % 5) - 2) / 8.0f;
        for (std::size_t i = 0; i < harmonic.size(); ++i)
            harmonic_direction[i] =
                static_cast<float>(static_cast<int>(i % 3) - 1) / 8.0f;
        for (std::int32_t function = 0;
             function <= extent.weight_size; ++function) {
            coefficients[extent.weight_size + 1 + function] =
                static_cast<float>((function % 7) - 3) / 64.0f;
            coefficients[2 * (extent.weight_size + 1) + function] =
                static_cast<float>((function % 3) - 1) / 128.0f;
        }
        const std::int32_t source_edge_offsets[] = {0, 1, 2};
        const std::int32_t source_edge_indices[] = {0, 1};
        std::vector<float> density_adjoint(num_nodes, 0.375f);
        std::vector<float> distance_adjoint(num_nodes, 0.0f);
        SymmetrixJitMH1HostSplineRReverseArgsV5 reverse_args{
            sizeof(SymmetrixJitMH1HostSplineRReverseArgsV5), layer, 0u, 0u,
            num_nodes, num_nodes, num_nodes, 1, 1, 1,
            extent.weight_size + 1, 0,
            1.0, 0.0, source_index, target_index, source_edge_offsets,
            source_edge_indices, source_types, node_types, distance,
            coefficients.data(), harmonic.data(), mask.data(), source.data(),
            native_output_adjoint.data(), density_adjoint.data(),
            source_adjoint.data(),
            harmonic_adjoint.data(), distance_adjoint.data()};
        for (std::int32_t source_owner = 0;
             source_owner < num_nodes; ++source_owner)
            spline.source_reverse_owner(&reverse_args, source_owner);

        auto loss = [&](double radius) {
            std::fill(actual.begin(), actual.end(), 0.0f);
            std::fill(density.begin(), density.end(), 0.0f);
            const double local_distance[] = {radius, distance[1]};
            auto local_args = spline_args;
            local_args.distances = local_distance;
            for (std::int32_t receiver = 0; receiver < num_nodes; ++receiver)
                spline.forward_owner(&local_args, receiver);
            float result = 0.0f;
            for (std::int32_t node = 0; node < num_nodes; ++node)
                result += density_adjoint[node] * density[node];
            for (std::size_t i = 0; i < actual.size(); ++i)
                result += output_adjoint[i] * actual[i];
            return result;
        };
        constexpr float epsilon = 1.0e-3f;
        const float base_loss = loss(distance[0]);
        for (std::size_t i = 0; i < source.size(); ++i)
            source[i] += epsilon * source_direction[i];
        const float source_plus = loss(distance[0]);
        for (std::size_t i = 0; i < source.size(); ++i)
            source[i] -= 2.0f * epsilon * source_direction[i];
        const float source_minus = loss(distance[0]);
        for (std::size_t i = 0; i < source.size(); ++i)
            source[i] += epsilon * source_direction[i];
        float source_dot = 0.0f;
        for (std::size_t i = 0; i < source.size(); ++i)
            source_dot += source_adjoint[i] * source_direction[i];
        if (std::abs(source_dot - (source_plus - source_minus) / (2 * epsilon))
            > 2.0e-3f * (1.0f + std::abs(base_loss))) return 7;

        for (std::size_t i = 0; i < harmonic.size(); ++i)
            harmonic[i] += epsilon * harmonic_direction[i];
        const float harmonic_plus = loss(distance[0]);
        for (std::size_t i = 0; i < harmonic.size(); ++i)
            harmonic[i] -= 2.0f * epsilon * harmonic_direction[i];
        const float harmonic_minus = loss(distance[0]);
        for (std::size_t i = 0; i < harmonic.size(); ++i)
            harmonic[i] += epsilon * harmonic_direction[i];
        float harmonic_dot = 0.0f;
        for (std::size_t i = 0; i < harmonic.size(); ++i)
            harmonic_dot += harmonic_adjoint[i] * harmonic_direction[i];
        if (std::abs(
                harmonic_dot
                - (harmonic_plus - harmonic_minus) / (2 * epsilon))
            > 2.0e-3f * (1.0f + std::abs(base_loss))) return 8;

        const float radial_fd =
            (loss(distance[0] + epsilon) - loss(distance[0] - epsilon))
            / (2 * epsilon);
        if (std::abs(distance_adjoint[0] - radial_fd)
            > 2.0e-3f * (1.0f + std::abs(base_loss))) return 9;
    }
    dlclose(handle);
    return 0;
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            runner_source,
            "-I",
            str(Path(__file__).parents[1] / "source/symmetrix"),
            "-ldl",
            "-o",
            runner,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([runner, library], check=True)


def test_mh1_v5_float64_spline_r_plugin_compiles(mh1_node_definition, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for spline R generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    generated = render_jit_mh1_host_plugin_v5(contract, precision="float64")
    spline_source = generated[generated.rindex("\nnamespace {\n") :]

    assert "value.scalar_size = sizeof(double);" in spline_source
    descriptor_source = spline_source[
        spline_source.index("const SymmetrixJitMH1HostPluginV5") :
    ]
    assert (
        "SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5"
        in descriptor_source
    )
    assert "const double* const coefficients" in spline_source
    assert "double source_contribution_0 = 0.0f;" in spline_source
    assert "const float" not in spline_source
    assert "float*" not in spline_source

    source = tmp_path / "mh1_host_v5_float64.cpp"
    library = tmp_path / "mh1_host_v5_float64.so"
    source.write_text(generated, encoding="ascii")
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mh1_v5_rejects_unknown_spline_precision(mh1_node_definition):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    with pytest.raises(ValueError, match="float32.*float64"):
        render_jit_mh1_host_plugin_v5(contract, precision="float16")


def test_mh1_checkpoint_extraction_embeds_contract_when_available():
    checkpoint = os.environ.get("SYMMETRIX_MH1_MODEL")
    if not checkpoint or not Path(checkpoint).is_file():
        pytest.skip("set SYMMETRIX_MH1_MODEL to test checkpoint extraction")

    import torch
    from mace.tools.scripts_utils import remove_pt_head
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    model = torch.load(
        checkpoint, map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    if hasattr(model, "heads") and len(model.heads) != 1:
        model = remove_pt_head(model, "matpes_r2scan")
    data = extract_mace_nonlinear_data(model)

    contract = data["execution_contracts"][MH1_CONTRACT_KEY]
    assert normalize_execution_mh1_contract(contract) == contract
    assert len(contract["interactions"]) == 2
