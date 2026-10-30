import copy
import ctypes
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import symmetrix.mace_nonlinear_execution_contract as execution_mh1_contract
import symmetrix.mace_nonlinear_jit_codegen as mh1_codegen
from symmetrix.mace_nonlinear_execution_contract import (
    MH1_CONTRACT_KEY,
    MH1_CONTRACT_TAG,
    MH1_NODE_PROGRAM_TAG,
    MH1_V4_CONTRACT_TAG,
    make_execution_mh1_contract,
    make_execution_mh1_v4_contract,
    maybe_make_execution_mh1_contract,
    maybe_make_execution_mh1_v4_contract,
    node_execution_schedule,
    normalize_execution_mh1_contract,
    normalize_execution_mh1_v4_contract,
    validate_execution_mh1_contract_for_model,
    validate_execution_mh1_v4_contract_for_model,
)
from symmetrix.mace_polar_execution_contract import (
    POLAR_CONTRACT_KEY,
    POLAR_CONTRACT_TAG,
    make_execution_polar_backbone_contract,
    make_execution_polar_mh1_v4_contract,
    normalize_execution_polar_backbone_contract,
    polar_execution_schedule,
    validate_execution_polar_backbone_contract_for_model,
    validate_execution_polar_mh1_v4_contract_for_model,
)
from symmetrix.mace_nonlinear_jit_codegen import (
    execution_mh1_node_program_metadata,
    render_execution_mh1_node_linear_cpp_helpers,
    render_jit_mh1_host_plugin_v7,
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


@pytest.mark.parametrize("layer_count", (1, 2, 3))
def test_native_stage_description_matches_the_contract_schedule(layer_count):
    """The native stage description is the contract's schedule, not a copy.

    The description carries the interaction-layer count and the ordered forward
    and reverse schedules as data, and a schedule-walking orchestration reads
    them.  They must equal the schedule the contract generates for the same
    count — the same entries, in the same order, with the same phase names — or
    the two vocabularies have drifted and the description is describing a
    different program than the contract declares.
    """

    import symmetrix.symmetrix as native_symmetrix

    expected = node_execution_schedule(layer_count)
    described = native_symmetrix.mace_nonlinear_stage_description(layer_count)

    assert described["layer_count"] == layer_count
    assert [dict(entry) for entry in described["forward_schedule"]] == (
        expected["forward_schedule"]
    )
    assert [dict(entry) for entry in described["reverse_schedule"]] == (
        expected["reverse_schedule"]
    )


def test_optional_mh1_contract_omits_incompatible_models(mh1_definition):
    # The topology admission is no longer one-or-two: a model with a single
    # interaction layer is described by the same contract builder, so the
    # optional builder returns a contract for it instead of omitting it.  The
    # omission the test's subject is the *incompatible* model below.
    one_layer = copy.deepcopy(mh1_definition)
    one_layer["interactions"].pop()
    single = maybe_make_execution_mh1_contract(one_layer)
    assert single is not None
    assert single["interaction_count"] == 1
    assert [descriptor["index"] for descriptor in single["interactions"]] == [0]

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


def test_mh1_generation_19_recompute_plan_splits_external_product_from_nodes(
    mh1_node_definition,
):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    plan = mh1_codegen.execution_mh1_cuda_module_v4_launch_plan(
        contract, 80, node_state_policy="recompute-v1"
    )

    assert plan["split_phase_composition"] is True
    assert plan["node_reverse_schedule"] == "composed-v5-full-replay-v1"

    for node in plan["nodes"]:
        index = node["index"]
        pre_forward_kernels = {launch["kernel"] for launch in node["split_pre_forward"]}
        post_forward_kernels = {
            launch["kernel"] for launch in node["split_post_forward"]
        }
        post_reverse_kernels = {
            launch["kernel"] for launch in node["split_post_reverse"]
        }
        pre_reverse_kernels = {launch["kernel"] for launch in node["split_pre_reverse"]}
        assert {
            f"symmetrix_execution_mh1_split_node_grouped_l{index}_forward_residual_grouped",
            f"symmetrix_execution_mh1_split_node_grouped_l{index}_forward_message_grouped",
        } <= pre_forward_kernels
        assert (
            f"symmetrix_execution_mh1_split_node_grouped_l{index}_reverse_message_grouped"
            in pre_reverse_kernels
        )

        # Generation 19 stops before the external common-product contraction
        # and resumes with only its node-local product-linear epilogue.  The
        # launch plan must not smuggle a private contraction into either half.
        assert all("product" not in kernel for kernel in pre_forward_kernels)
        assert all("product" not in kernel for kernel in pre_reverse_kernels)
        product_epilogues = {
            kernel
            for kernel in post_forward_kernels | post_reverse_kernels
            if "product" in kernel
        }
        assert product_epilogues
        assert all("product_linear" in kernel for kernel in product_epilogues)
        assert node["post_forward"] == node["split_post_forward"]
        assert node["post_reverse"] == node["split_post_reverse"]


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
    assert "float* retained_interaction = args.retained_interaction_output" in module
    assert "retained_interaction[column] = interaction[column]" in module
    assert "l0_retained_gate_reverse" not in module
    assert "l0_norm_res_recompute" in module
    assert "l1_norm_msg_recompute" in module


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
    assert current["artifact_id"] != pre_fusion["artifact_id"]


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
    assert "float energy_scale;" not in source
    assert "double energy_scale;" in source
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


def test_mh1_v7_host_plugin_splits_product_contraction(mh1_node_definition):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    source = render_jit_mh1_host_plugin_v7(contract, precision="float32")
    split = source.index("node_split_pre_forward_owner_0")
    v7 = source[split:]

    assert "SYMMETRIX_JIT_MH1_HOST_EXTERNAL_PRODUCT_V7" in v7
    assert "symmetrix_jit_mh1_host_plugin_query_v7" in v7
    assert "contraction_output" in v7
    assert "contraction_output_adjoint" in v7
    assert "product_parameters" not in v7
    assert "execution_mh1_node_layer_0_product_forward" not in v7
    assert "execution_mh1_node_layer_0_product_reverse" not in v7
    assert "execution_mh1_node_layer_0_product_forward" not in source[:split]
    assert "execution_mh1_node_layer_0_product_reverse" not in source[:split]


def test_mh1_v7_float64_host_plugin_uses_double_math(mh1_node_definition):
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    source = render_jit_mh1_host_plugin_v7(contract, precision="float64")

    assert "expf(" not in source
    assert "sqrtf(" not in source
    assert "tanhf(" not in source
    assert "exp(" in source
    assert "sqrt(" in source
    assert "tanh(" in source


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_mh1_v7_host_plugin_compiles(mh1_node_definition, tmp_path, precision):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for split host generation")
    contract = make_execution_mh1_v4_contract(mh1_node_definition)
    source = tmp_path / f"mh1_host_v7_{precision}.cpp"
    library = tmp_path / f"mh1_host_v7_{precision}.so"
    source.write_text(
        render_jit_mh1_host_plugin_v7(contract, precision=precision),
        encoding="ascii",
    )
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", source, "-o", library],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mh1_v7_non_posix_loader_stubs_are_link_complete(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for loader symbol coverage")
    repository = Path(__file__).parents[2]
    probe = tmp_path / "mh1_v7_non_posix_probe.cpp"
    executable = tmp_path / "mh1_v7_non_posix_probe"
    probe.write_text(
        r"""
#include "mace_nonlinear_jit_host_plugin.hpp"
#include <stdexcept>
#include <utility>
int main() {
    using symmetrix::execution::MH1HostPluginV7;
    using symmetrix::execution::MH1SplitPhaseExpectation;
    bool load_failed=false;
    try { (void)MH1HostPluginV7::load("unavailable",MH1SplitPhaseExpectation{}); }
    catch(const std::runtime_error&) { load_failed=true; }
    MH1HostPluginV7 first;
    MH1HostPluginV7 second(std::move(first));
    first=std::move(second);
    bool descriptor_failed=false;
    try { (void)first.descriptor(); }
    catch(const std::logic_error&) { descriptor_failed=true; }
    return load_failed&&descriptor_failed&&!first?0:1;
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-U__unix__",
            "-U__linux__",
            "-U__APPLE__",
            f"-I{repository / 'libsymmetrix' / 'source'}",
            f"-I{repository / 'symmetrix' / 'source' / 'symmetrix'}",
            repository / "libsymmetrix/source/mace_nonlinear_jit_host_plugin.cpp",
            probe,
            "-o",
            executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([executable], check=True, capture_output=True, text=True)


def test_mh1_cuda_v5_non_posix_loader_stubs_are_link_complete(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for loader symbol coverage")
    repository = Path(__file__).parents[2]
    probe = tmp_path / "mh1_cuda_v5_non_posix_probe.cpp"
    executable = tmp_path / "mh1_cuda_v5_non_posix_probe"
    probe.write_text(
        r"""
#include "mace_nonlinear_jit_cuda_plugin.hpp"
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
int main() {
    using symmetrix::execution::MH1CudaPlugin;
    using symmetrix::execution::MH1CudaPluginV4;
    using symmetrix::execution::MH1CudaPluginExpectation;
    using symmetrix::execution::MH1CudaPluginV4Expectation;
    auto posix_failure=[](const std::function<void()>& call) {
        try { call(); }
        catch(const std::runtime_error& error) {
            return std::string(error.what()).find("require POSIX")
                !=std::string::npos;
        }
        return false;
    };
    if(!posix_failure([] { (void)MH1CudaPlugin::load(
            "unavailable",MH1CudaPluginExpectation{}); })) return 2;
    bool retired_load_failed=false;
    try {
        (void)MH1CudaPluginV4::load(
            "unavailable",MH1CudaPluginV4Expectation{});
    } catch(const std::invalid_argument& error) {
        retired_load_failed=std::string(error.what()).find("retired")
            !=std::string::npos;
    }
    if(!retired_load_failed) return 3;
    if(!posix_failure([] { (void)MH1CudaPluginV4::load_cubin(
            "unavailable",MH1CudaPluginV4Expectation{},"{}"); })) return 4;
    if(!posix_failure([] { (void)MH1CudaPluginV4::load_module(
            "unavailable",MH1CudaPluginV4Expectation{},"{}"); })) return 5;
    MH1CudaPluginV4 first;
    MH1CudaPluginV4 second(std::move(first));
    first=std::move(second);
    SymmetrixJitMH1CudaForwardArgsV4 forward{};
    SymmetrixJitMH1CudaReverseArgsV4 reverse{};
    SymmetrixJitMH1CudaSplineRForwardArgsV5 spline_forward{};
    SymmetrixJitMH1CudaSplineRSourceArgsV5 spline_source{};
    SymmetrixJitMH1CudaSplineREdgeArgsV5 spline_edge{};
    SymmetrixJitMH1CudaConditioningForwardArgsV4 conditioning_forward{};
    SymmetrixJitMH1CudaConditioningReverseArgsV4 conditioning_reverse{};
    SymmetrixJitMH1CudaNodeForwardArgsV5 node_forward{};
    SymmetrixJitMH1CudaNodeReverseArgsV5 node_reverse{};
    if(!posix_failure([&] { (void)first.launch_forward(&forward,nullptr,1); })) return 6;
    if(!posix_failure([&] { (void)first.launch_reverse(&reverse,nullptr,1,1); })) return 7;
    if(!posix_failure([&] { (void)first.launch_spline_r_forward(&spline_forward,nullptr,1); })) return 8;
    if(!posix_failure([&] { (void)first.launch_spline_r_reverse(&spline_source,&spline_edge,nullptr,1); })) return 9;
    if(!posix_failure([&] { (void)first.launch_conditioning_forward(&conditioning_forward,nullptr,1); })) return 10;
    if(!posix_failure([&] { (void)first.launch_conditioning_reverse(&conditioning_reverse,nullptr,1); })) return 11;
    if(!posix_failure([&] { (void)first.launch_node_forward(&node_forward,nullptr,1); })) return 12;
    if(!posix_failure([&] { (void)first.launch_node_reverse(&node_reverse,nullptr,1); })) return 13;
    if(!posix_failure([&] { (void)first.launch_split_forward(&node_forward,nullptr,1); })) return 14;
    if(!posix_failure([&] { (void)first.launch_split_reverse(&node_reverse,nullptr,1); })) return 15;
    if(!posix_failure([&] { (void)first.descriptor(); })) return 16;
    return !first&&first.path().empty()&&!first.has_spline_r_program()
        &&first.edge_launch_count(0)==0&&!first.split_phase_ready()?0:17;
}
""",
        encoding="ascii",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-U__unix__",
            "-U__linux__",
            "-U__APPLE__",
            f"-I{repository / 'libsymmetrix' / 'source'}",
            f"-I{repository / 'symmetrix' / 'source' / 'symmetrix'}",
            repository / "libsymmetrix/source/mace_nonlinear_jit_cuda_plugin.cpp",
            probe,
            "-o",
            executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([executable], check=True, capture_output=True, text=True)


def _polar_spin_source_head(zero_charges):
    return {
        "class": "EnvironmentDependentSpinSourceBlock",
        "zero_charges": zero_charges,
        "linear": _e3_linear(
            "1x0e+1x1o",
            "1x0e+1x1o+1x0e+1x1o",
            [(0, 0), (0, 2), (1, 1), (1, 3)],
        ),
    }


def _polar_update_mlp():
    return {
        "layers": [
            {
                "type": "linear",
                "weight": _tensor([5, 2], 0.1),
                "bias": _tensor([5], 0.2),
            },
            {
                "type": "layer_norm",
                "normalized_shape": [5],
                "eps": 1.0e-5,
                "weight": _tensor([5], 1.0),
                "bias": _tensor([5], 0.0),
            },
            {"type": "silu"},
            {
                "type": "linear",
                "weight": _tensor([1, 5], 0.3),
                "bias": _tensor([1], 0.4),
            },
        ]
    }


def _polar_dot_products():
    return {
        "irreps_in1": "1x0e+1x1o",
        "irreps_in2": "1x0e+1x1o",
        "irreps_out": "1x0e",
        "instructions": [
            {
                "i_in1": 0,
                "i_in2": 0,
                "i_out": 0,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
                "path_shape": [1, 1],
                "wigner_3j": _tensor([1, 1, 1], 1.0),
            },
            {
                "i_in1": 1,
                "i_in2": 1,
                "i_out": 0,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 3.0**0.5,
                "path_shape": [1, 1],
                "wigner_3j": {
                    "shape": [3, 3, 1],
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
        ],
        "weight": _tensor([2], 0.25),
        "output_mask": _tensor([1], 1.0),
    }


def _polar_tp_out():
    return {
        "irreps_in1": "1x0e+1x1o",
        "irreps_in2": "1x0e",
        "irreps_out": "1x0e+1x1o",
        "instructions": [
            {
                "i_in1": 0,
                "i_in2": 0,
                "i_out": 0,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 1.0,
                "path_shape": [1, 1],
                "wigner_3j": _tensor([1, 1, 1], 1.0),
            },
            {
                "i_in1": 1,
                "i_in2": 0,
                "i_out": 1,
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": 3.0**0.5,
                "path_shape": [1, 1],
                "wigner_3j": {
                    "shape": [3, 1, 3],
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
        ],
        "weight": _tensor([2], 0.25),
        "output_mask": _tensor([4], 1.0),
    }


def _polar_update_readout():
    return {
        "class": "GeneralNonLinearBiasReadoutBlock",
        "linear_1": _e3_linear("1x0e+1x1o", "4x0e+2x1o", [(0, 0), (1, 1)]),
        "linear_mid": _e3_linear("2x0e+2x1o", "4x0e+2x1o", [(0, 0), (1, 1)]),
        "linear_2": _e3_linear(
            "2x0e+2x1o",
            "1x0e+1x1o+1x0e+1x1o+2x0e",
            [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3)],
        ),
        "gate": {
            "irreps_in": "4x0e+2x1o",
            "irreps_out": "2x0e+2x1o",
            "irreps_scalars": "2x0e",
            "irreps_gates": "2x0e",
            "irreps_gated": "2x1o",
            "mul": {
                "irreps_in1": "2x1o",
                "irreps_in2": "2x0e",
                "irreps_out": "2x1o",
                "instructions": [
                    {
                        "i_in1": 0,
                        "i_in2": 0,
                        "i_out": 0,
                        "connection_mode": "uvu",
                        "has_weight": True,
                        "path_weight": 1.0,
                        "path_shape": [2, 2],
                        "wigner_3j": {
                            "shape": [3, 1, 3],
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
                    }
                ],
                "weight": _tensor([4], 0.5),
                "output_mask": _tensor([6], 1.0),
            },
            "scalar_activation": "silu",
            "scalar_activation_constants": [1.6791767923989418],
            "gate_activation": "sigmoid",
            "gate_activation_constants": [1.8467055342154763],
        },
    }


def _polar_field_update_block():
    return {
        "class": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding": {
            "class": "AgnosticChargeBiasedLinearPotentialEmbedding",
            "potential_linear": _e3_linear("1x0e+1x0e", "1x0e+1x1o", [(0, 0), (1, 0)]),
            "node_feats_linear": _e3_linear("1x0e+1x1o", "1x0e+1x1o", [(0, 0), (1, 1)]),
            "charge_embedding": _e3_linear(
                "1x0e+1x1o+1x0e+1x1o",
                "1x0e+1x1o",
                [(0, 0), (2, 0), (1, 1), (3, 1)],
            ),
        },
        "source_embedding": _e3_linear("2x0e", "1x0e", [(0, 0)]),
        "dot_products": _polar_dot_products(),
        "nonlinearity": _polar_update_mlp(),
        "tp_out": _polar_tp_out(),
        "readout": _polar_update_readout(),
        "field_norm_factor": 1.0,
    }


def _polar_section(steps=2):
    return {
        "class": "PolarMACE",
        "keep_last_layer_irreps": True,
        "num_recursion_steps": steps,
        "atomic_multipoles_max_l": 1,
        "atomic_multipoles_smearing_width": 1.5,
        "field_feature_max_l": 0,
        "field_feature_widths": [1.5],
        "field_feature_norms": _tensor([1], 1.0),
        "field_norm_factor": 1.0,
        "kspace_cutoff": 4.2,
        "charges_irreps": "1x0e+1x1o+1x0e+1x1o",
        "field_irreps": "1x0e",
        "potential_irreps": "1x0e+1x0e",
        "spin_source_heads": [
            _polar_spin_source_head(zero_charges=False),
            _polar_spin_source_head(zero_charges=True),
        ],
        "feature_mixer": {
            "class": "MultiLayerFeatureMixer",
            "linears": [
                _e3_linear("1x0e+1x1o", "1x0e+1x1o", [(0, 0), (1, 1)]),
                _e3_linear("1x0e+1x1o", "1x0e+1x1o", [(0, 0), (1, 1)]),
            ],
        },
        "field_update_blocks": [_polar_field_update_block() for _ in range(steps)],
        # The chain's two ends: the pre-loop Fukui source map and the post-loop
        # local-electron readout (both are model data the contract describes),
        # plus the checkpoint flags the composition depends on.
        "fukui_source_map": {
            "class": "NonLinearBiasReadoutBlock",
            "linear": None,
            # The oracle's map hides a scalar stack (the activation is applied
            # elementwise), and the output linear emits the two source
            # channels.
            "linear_1": _e3_linear("1x0e+1x1o", "2x0e", [(0, 0)]),
            "linear_mid": _e3_linear("2x0e", "2x0e", [(0, 0)]),
            "linear_2": _e3_linear("2x0e", "2x0e", [(0, 0)]),
            "activation": "silu",
            "activation_constants": [1.0],
        },
        "local_electron_energy": {
            "class": "OneBodyMLPFieldReadout",
            "linear_up_q": _e3_linear(
                "1x0e+1x1o+1x0e+1x1o", "1x0e+1x1o", [(0, 0), (2, 0), (1, 1), (3, 1)]
            ),
            "linear_up_v": _e3_linear("1x0e+1x0e", "1x0e+1x1o", [(0, 0), (1, 0)]),
            "dot_products_q": _polar_dot_products(),
            "dot_products_v": _polar_dot_products(),
            "mlp": _polar_local_electron_mlp(),
        },
        "add_local_electron_energy": True,
        "include_electrostatic_self_interaction": False,
        "quadrupole_feature_corrections": False,
        "field_si": False,
    }


def _polar_local_electron_mlp():
    """The readout's affine MLP: two invariants in, one energy out."""

    return {
        "layers": [
            {
                "type": "linear",
                "weight": _tensor([1, 2], 0.1),
                "bias": _tensor([1], 0.2),
            },
        ]
    }


@pytest.fixture
def polar_definition(mh1_node_definition):
    """A two-interaction PolarMACE-shaped definition: keep-last-layer products."""

    definition = copy.deepcopy(mh1_node_definition)
    definition["num_elements"] = 2
    product = definition["products"][1]
    product["node_feats_irreps"] = "1x0e+1x1o"
    product["target_irreps"] = "1x0e+1x1o"
    product["linear"] = _e3_linear("1x0e+1x1o", "1x0e+1x1o", [(0, 0), (1, 1)])
    product["symmetric_contractions"]["irreps_out"] = "1x0e+1x1o"
    product["symmetric_contractions"]["contractions"] = [
        _mh1_contraction(0),
        _mh1_contraction(1),
    ]
    definition["interactions"][1]["skip_tp"] = _e3_linear(
        "1x0e+1x1o", "1x0e+1x1o", [(0, 0), (1, 1)]
    )
    definition["readouts"][1]["linear_1"] = _e3_linear("1x0e+1x1o", "2x0e", [(0, 0)])
    definition["polar"] = _polar_section()
    return definition


def test_polar_backbone_contract_is_parameter_free_and_canonical(polar_definition):
    contract = make_execution_polar_backbone_contract(polar_definition)

    assert POLAR_CONTRACT_KEY == "POLAR_BACKBONE"
    assert (
        contract["tag"]
        == POLAR_CONTRACT_TAG
        == ("symmetrix.execution.polar_backbone/2")
    )
    assert contract["interaction_count"] == 2
    assert contract["recursion_step_count"] == 2
    # The unrolled chain is part of the contract, and it is the depth this
    # model declares rather than a fixed literal.
    assert contract["execution"] == polar_execution_schedule(2)
    assert contract["keep_last_layer_irreps"] is True
    assert contract["charges"]["irreps"] == "1x0e+1x1o+1x0e+1x1o"
    assert contract["charges"]["dimension"] == 8
    assert contract["field"]["irreps"] == "1x0e"
    assert contract["field"]["feature_norm_count"] == 1
    assert contract["potential"]["irreps"] == "1x0e+1x0e"
    assert contract["node_feats_irreps"] == "1x0e+1x1o"
    assert [head["index"] for head in contract["spin_source_heads"]] == [0, 1]
    assert contract["spin_source_heads"][0]["zero_charges"] is False
    assert contract["spin_source_heads"][1]["zero_charges"] is True
    update = contract["field_update_blocks"][0]
    assert update["edge_set"] == "empty"
    assert [path["mode"] for path in update["dot_products"]["paths"]] == [
        "contraction",
        "contraction",
    ]
    assert [path["mode"] for path in update["tp_out"]["paths"]] == [
        "contraction",
        "modulation",
    ]
    assert update["readout"]["linear_2"]["irreps"]["output"] == (
        "1x0e+1x1o+1x0e+1x1o+2x0e"
    )
    assert contract["semantic_fingerprint"].startswith("sha256:")
    assert normalize_execution_polar_backbone_contract(contract) == contract
    assert (
        validate_execution_polar_backbone_contract_for_model(polar_definition, contract)
        == contract
    )


def test_polar_backbone_identity_excludes_learned_values(polar_definition):
    original = make_execution_polar_backbone_contract(polar_definition)
    changed = copy.deepcopy(polar_definition)
    polar = changed["polar"]
    for head in polar["spin_source_heads"]:
        head["linear"]["weight"]["values"] = [
            value + 7.0 for value in head["linear"]["weight"]["values"]
        ]
    for linear in polar["feature_mixer"]["linears"]:
        linear["weight"]["values"] = [
            value + 1.0 for value in linear["weight"]["values"]
        ]
    for update in polar["field_update_blocks"]:
        embedding = update["potential_embedding"]
        for name in ("potential_linear", "node_feats_linear", "charge_embedding"):
            embedding[name]["weight"]["values"] = [
                value - 2.0 for value in embedding[name]["weight"]["values"]
            ]
        update["source_embedding"]["weight"]["values"] = [
            value + 3.0 for value in update["source_embedding"]["weight"]["values"]
        ]
        for name in ("dot_products", "tp_out"):
            update[name]["weight"]["values"] = [
                value + 4.0 for value in update[name]["weight"]["values"]
            ]
        for layer in update["nonlinearity"]["layers"]:
            for tensor_name in ("weight", "bias"):
                if tensor_name in layer:
                    layer[tensor_name]["values"] = [
                        value - 1.0 for value in layer[tensor_name]["values"]
                    ]
        readout = update["readout"]
        for name in ("linear_1", "linear_mid", "linear_2"):
            readout[name]["weight"]["values"] = [
                value + 5.0 for value in readout[name]["weight"]["values"]
            ]
        readout["gate"]["mul"]["weight"]["values"] = [
            value + 6.0 for value in readout["gate"]["mul"]["weight"]["values"]
        ]
        readout["gate"]["mul"]["output_mask"]["values"] = [
            value + 0.5 for value in readout["gate"]["mul"]["output_mask"]["values"]
        ]

    assert make_execution_polar_backbone_contract(changed) == original


def test_polar_backbone_identity_captures_program_facts(polar_definition):
    original = make_execution_polar_backbone_contract(polar_definition)

    changed_flags = copy.deepcopy(polar_definition)
    changed_flags["polar"]["spin_source_heads"][0]["zero_charges"] = True
    flags_contract = make_execution_polar_backbone_contract(changed_flags)
    assert (
        flags_contract["generation_fingerprint"] != original["generation_fingerprint"]
    )
    assert flags_contract["structure_fingerprint"] != original["structure_fingerprint"]

    changed_cutoff = copy.deepcopy(polar_definition)
    changed_cutoff["polar"]["kspace_cutoff"] = 4.3
    cutoff_contract = make_execution_polar_backbone_contract(changed_cutoff)
    assert (
        cutoff_contract["generation_fingerprint"] != original["generation_fingerprint"]
    )
    assert cutoff_contract["structure_fingerprint"] == original["structure_fingerprint"]

    changed_topology = copy.deepcopy(polar_definition)
    for update in changed_topology["polar"]["field_update_blocks"]:
        update["nonlinearity"]["layers"][1] = {"type": "silu"}
    topology_contract = make_execution_polar_backbone_contract(changed_topology)
    assert (
        topology_contract["structure_fingerprint"] != original["structure_fingerprint"]
    )

    changed_wigner = copy.deepcopy(polar_definition)
    for update in changed_wigner["polar"]["field_update_blocks"]:
        update["dot_products"]["instructions"][1]["wigner_3j"]["values"][3] = 0.5
    wigner_contract = make_execution_polar_backbone_contract(changed_wigner)
    assert (
        wigner_contract["generation_fingerprint"] != original["generation_fingerprint"]
    )
    assert wigner_contract["structure_fingerprint"] != original["structure_fingerprint"]

    changed_depth = copy.deepcopy(polar_definition)
    changed_depth["polar"]["field_update_blocks"] = changed_depth["polar"][
        "field_update_blocks"
    ][:1]
    changed_depth["polar"]["num_recursion_steps"] = 1
    depth_contract = make_execution_polar_backbone_contract(changed_depth)
    assert depth_contract["recursion_step_count"] == 1
    assert (
        depth_contract["generation_fingerprint"] != original["generation_fingerprint"]
    )


def test_polar_model_validates_under_the_existing_mh1_admission(polar_definition):
    contract = make_execution_mh1_contract(polar_definition)
    assert contract["tag"] == MH1_CONTRACT_TAG
    assert normalize_execution_mh1_contract(contract) == contract
    assert (
        validate_execution_mh1_contract_for_model(polar_definition, contract)
        == contract
    )

    v4 = make_execution_polar_mh1_v4_contract(polar_definition)
    assert v4["node_program"]["layers"][1]["product"]["irreps"]["output"] == (
        "1x0e+1x1o"
    )
    assert normalize_execution_mh1_v4_contract(v4) == v4
    assert (
        validate_execution_polar_mh1_v4_contract_for_model(polar_definition, v4) == v4
    )


def _resized_mh1_definition(definition, count):
    """The MH-1 definition resized to ``count`` synthetic interaction layers.

    The fixture's layer builders are index-parameterized -- layer 0 carries
    the model input shapes, the later layers the residual ones, and only the
    last layer reads out nonlinearly -- so composing them into ``count``
    layers builds a synthetic topology of that depth.  It is *not* the shipped
    checkpoints' layer content: those are recorded in the facts fixture and
    validated against the real checkpoints by the env-gated extraction
    comparison; what this builds is the contract shape the interaction-count
    admission must accept.
    """

    resized = copy.deepcopy(definition)
    resized["interactions"] = [_mh1_node_interaction(0)] + [
        copy.deepcopy(_mh1_node_interaction(1)) for _ in range(count - 1)
    ]
    resized["products"] = [copy.deepcopy(_mh1_product(0)) for _ in range(count - 1)] + [
        _mh1_product(1)
    ]
    resized["readouts"] = [copy.deepcopy(_mh1_readout(0)) for _ in range(count - 1)] + [
        _mh1_readout(1)
    ]
    return resized


@pytest.mark.parametrize("count", (1, 3))
def test_mh1_interaction_counts_outside_the_legacy_pair_are_admitted(
    mh1_node_definition, count
):
    """The lifted topology admits one and three interaction layers.

    The interaction count is contract data (the v6 interaction-count lift), so
    a topology outside the legacy pair builds the version-3 identity,
    normalizes, validates against the model definition, and is admitted by the
    extractor's optional builder.  Nothing here is rejected by the admission
    itself: the legacy exactly-two rule is gone.
    """

    definition = _resized_mh1_definition(mh1_node_definition, count)
    contract = make_execution_mh1_contract(definition)
    assert contract["tag"] == MH1_CONTRACT_TAG
    assert contract["interaction_count"] == count
    assert [item["index"] for item in contract["interactions"]] == list(range(count))
    assert normalize_execution_mh1_contract(contract) == contract
    assert validate_execution_mh1_contract_for_model(definition, contract) == contract
    assert maybe_make_execution_mh1_contract(definition) == contract


def _resized_polar_definition(definition, count):
    """The polar definition resized to ``count`` synthetic interaction layers.

    Every per-interaction list is contract data (the interaction, the product,
    the readout, the spin-source head, the mixer linear), so a topology of
    another depth is the fixture's layer 0 followed by ``count - 1`` copies of
    its keep-last layer, with the position-stated readout rule applied.  The
    content is synthetic -- the shipped checkpoints' own layer shapes come from
    the facts fixture -- and what it exercises is the admission of that depth.
    """

    resized = copy.deepcopy(definition)

    def layers(entries):
        return [copy.deepcopy(entries[0])] + [
            copy.deepcopy(entries[1]) for _ in range(count - 1)
        ]

    resized["interactions"] = layers(resized["interactions"])
    resized["products"] = [
        copy.deepcopy(resized["products"][0]) for _ in range(count - 1)
    ] + [copy.deepcopy(resized["products"][1])]
    resized["readouts"] = [
        copy.deepcopy(resized["readouts"][0]) for _ in range(count - 1)
    ] + [copy.deepcopy(resized["readouts"][1])]
    polar = resized["polar"]
    polar["spin_source_heads"] = layers(polar["spin_source_heads"])
    polar["feature_mixer"]["linears"] = layers(polar["feature_mixer"]["linears"])
    return resized


@pytest.mark.parametrize("count", (1, 3))
def test_polar_shipped_interaction_counts_are_admitted_by_the_contracts(
    polar_definition, count
):
    """The one- and three-interaction polar depths are admitted.

    The legacy admission refused exactly-two topologies, so the S checkpoint's
    single interaction and the L checkpoint's three were both rejected before
    the lift; now the polar backbone identity, the MH-1 version-3 identity,
    and the v4 node program all build, normalize, and validate against the
    model definition, and the unrolled chain the polar contract carries is the
    depth the model declares.
    """

    definition = _resized_polar_definition(polar_definition, count)
    polar = make_execution_polar_backbone_contract(definition)
    assert polar["tag"] == POLAR_CONTRACT_TAG
    assert polar["interaction_count"] == count
    assert len(polar["spin_source_heads"]) == count
    assert len(polar["feature_mixer"]["linears"]) == count
    assert polar["execution"] == polar_execution_schedule(polar["recursion_step_count"])
    assert normalize_execution_polar_backbone_contract(polar) == polar
    assert (
        validate_execution_polar_backbone_contract_for_model(definition, polar) == polar
    )

    contract = make_execution_mh1_contract(definition)
    assert contract["interaction_count"] == count
    assert normalize_execution_mh1_contract(contract) == contract
    assert validate_execution_mh1_contract_for_model(definition, contract) == contract
    assert maybe_make_execution_mh1_contract(definition) == contract

    v4 = make_execution_polar_mh1_v4_contract(definition)
    assert v4["interaction_count"] == count
    assert len(v4["node_program"]["layers"]) == count
    assert v4["node_program"]["execution"] == node_execution_schedule(count)
    assert normalize_execution_mh1_v4_contract(v4) == v4
    assert validate_execution_polar_mh1_v4_contract_for_model(definition, v4) == v4


def test_polar_products_outside_the_keep_last_matrix_are_rejected(polar_definition):
    changed = copy.deepcopy(polar_definition)
    product = changed["products"][1]
    product["node_feats_irreps"] = "1x1o+1x0e"
    product["target_irreps"] = "1x1o+1x0e"
    product["linear"] = _e3_linear("1x1o+1x0e", "1x1o+1x0e", [(0, 0), (1, 1)])
    product["symmetric_contractions"]["irreps_out"] = "1x1o+1x0e"
    with pytest.raises(ValueError, match="keep-last-layer support matrix"):
        make_execution_polar_mh1_v4_contract(changed)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        pytest.param(
            lambda d: d.pop("polar"), "require polar model data", id="no-polar-data"
        ),
        pytest.param(
            lambda d: d["polar"].pop("keep_last_layer_irreps"),
            "keep-last-layer readouts",
            id="no-keep-last",
        ),
        pytest.param(
            lambda d: d["polar"].__setitem__(
                "spin_source_heads",
                d["polar"]["spin_source_heads"][:1],
            ),
            "match the interaction count",
            id="head-count",
        ),
        pytest.param(
            lambda d: d["polar"]["spin_source_heads"][0].__setitem__("class", "Other"),
            "spin-source multipole head",
            id="head-class",
        ),
        pytest.param(
            lambda d: d["polar"]["spin_source_heads"][0].__setitem__(
                "linear", _e3_linear("1x0e+1x1o", "3x0e", [(0, 0)])
            ),
            "spin-resolved multipoles",
            id="head-output",
        ),
        pytest.param(
            lambda d: d["polar"]["spin_source_heads"][1].__setitem__(
                "linear", _e3_linear("1x0e", "1x0e+1x1o+1x0e+1x1o", [(0, 0), (0, 2)])
            ),
            "linear input is inconsistent",
            id="head-input",
        ),
        pytest.param(
            lambda d: d["polar"]["feature_mixer"].__setitem__("class", "Other"),
            "multi-layer feature mixer",
            id="mixer-class",
        ),
        pytest.param(
            lambda d: d["polar"]["feature_mixer"].__setitem__(
                "linears", d["polar"]["feature_mixer"]["linears"][:1]
            ),
            "match the interaction count",
            id="mixer-count",
        ),
        pytest.param(
            lambda d: d["polar"]["feature_mixer"]["linears"][0].__setitem__(
                "irreps_in", "1x0e"
            ),
            "irrep index is invalid",
            id="mixer-linear-mismatch",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0].__setitem__(
                "class", "Other"
            ),
            "one-body variable update",
            id="update-class",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0][
                "potential_embedding"
            ].__setitem__("class", "Other"),
            "charge-biased linear potential embedding",
            id="embedding-class",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0]["potential_embedding"][
                "charge_embedding"
            ].__setitem__("irreps_in", "1x0e"),
            "irrep index is invalid",
            id="charge-embedding-input",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0]["potential_embedding"][
                "potential_linear"
            ].__setitem__("irreps_out", "1x0e+1x1o+1x2e"),
            "output_mask shape is inconsistent",
            id="potential-linear-output",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0].__setitem__(
                "source_embedding",
                _e3_linear("2x0e", "1x0e+1x0e", [(0, 0), (0, 1)]),
            ),
            "invariant channels",
            id="source-embedding-output",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0].__setitem__(
                "dot_products",
                {
                    "irreps_in1": "1x0e",
                    "irreps_in2": "1x0e",
                    "irreps_out": "1x0e",
                    "instructions": [
                        {
                            "i_in1": 0,
                            "i_in2": 0,
                            "i_out": 0,
                            "connection_mode": "uvu",
                            "has_weight": True,
                            "path_weight": 1.0,
                            "path_shape": [1, 1],
                            "wigner_3j": _tensor([1, 1, 1], 1.0),
                        }
                    ],
                    "weight": _tensor([1], 0.25),
                    "output_mask": _tensor([1], 1.0),
                },
            ),
            "dot_products irreps are inconsistent",
            id="dot-products-irreps",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0]["nonlinearity"]["layers"][0][
                "weight"
            ].__setitem__("shape", [5, 3]),
            "input dimension is inconsistent",
            id="nonlinearity-input",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0]["nonlinearity"]["layers"][0][
                "weight"
            ].__setitem__("shape", [5, 2, 1]),
            "rank-2 tensor shape",
            id="nonlinearity-weight-rank",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0]["readout"].__setitem__(
                "class", "NonLinearReadoutBlock"
            ),
            "general nonlinear bias readout block",
            id="readout-class",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0]["readout"][
                "linear_2"
            ].__setitem__("irreps_out", "1x0e+1x1o+1x0e+1x1o"),
            "irrep index is invalid",
            id="readout-fukui-output",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][0]["readout"][
                "gate"
            ].__setitem__("irreps_gated", ""),
            "must not be empty",
            id="readout-gate",
        ),
        pytest.param(
            lambda d: d["polar"]["field_update_blocks"][1]["nonlinearity"]["layers"][
                3
            ].__setitem__("weight", _tensor([1, 6], 0.3)),
            "input dimension is inconsistent",
            id="update-steps-differ",
        ),
        pytest.param(
            lambda d: d["polar"].__setitem__("charges_irreps", "1x1o+1x0e+1x1o+1x0e"),
            "outside the PolarMACE support matrix",
            id="charges-pattern",
        ),
        pytest.param(
            lambda d: d["polar"].__setitem__("field_feature_widths", []),
            "non-empty list",
            id="field-widths-empty",
        ),
        pytest.param(
            lambda d: d["polar"]["field_feature_norms"].__setitem__("shape", [2]),
            "field_feature_norms shape",
            id="field-norms-shape",
        ),
        pytest.param(
            lambda d: d["polar"].__setitem__("num_recursion_steps", 3),
            "match num_recursion_steps",
            id="update-count",
        ),
        pytest.param(
            lambda d: d["products"][1].__setitem__("target_irreps", "1x0e"),
            "uniform keep-last-layer product targets",
            id="nonuniform-products",
        ),
    ],
)
def test_polar_backbone_contract_rejects_unsupported_shapes(
    polar_definition, mutate, fragment
):
    changed = copy.deepcopy(polar_definition)
    mutate(changed)
    with pytest.raises(ValueError, match=fragment):
        make_execution_polar_backbone_contract(changed)


def test_polar_backbone_contract_rejects_stale_fingerprints(polar_definition):
    contract = make_execution_polar_backbone_contract(polar_definition)

    stale = copy.deepcopy(contract)
    stale["field_update_blocks"][0]["field_norm_factor"] += 1.0
    with pytest.raises(ValueError, match="semantic fingerprint is stale"):
        normalize_execution_polar_backbone_contract(stale)

    stale = copy.deepcopy(contract)
    stale["runtime_parameter_packs"]["feature_mixer"]["parameter_count"] += 1
    with pytest.raises(ValueError, match="parameter count is inconsistent"):
        normalize_execution_polar_backbone_contract(stale)

    stale = copy.deepcopy(contract)
    stale["interaction_count"] = 1
    with pytest.raises(ValueError, match="spin-source heads are invalid"):
        normalize_execution_polar_backbone_contract(stale)

    changed_model = copy.deepcopy(polar_definition)
    changed_model["polar"]["field_update_blocks"] = changed_model["polar"][
        "field_update_blocks"
    ][:1]
    changed_model["polar"]["num_recursion_steps"] = 1
    with pytest.raises(ValueError, match="does not match the model definition"):
        validate_execution_polar_backbone_contract_for_model(changed_model, contract)


def test_polar_backbone_requires_boolean_keep_last_flag(polar_definition):
    changed = copy.deepcopy(polar_definition)
    changed["polar"]["keep_last_layer_irreps"] = "yes"
    with pytest.raises(ValueError, match="must be boolean"):
        make_execution_polar_mh1_v4_contract(changed)


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
