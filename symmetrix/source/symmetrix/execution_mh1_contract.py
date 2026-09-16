"""Canonical generation contracts for two-interaction MACE-MH-1 UVU kernels.

The contract deliberately contains no learned tensor values.  It identifies the
generated program shared by checkpoints with the same architecture while the
learned affine weights remain runtime inputs loaded from the model JSON.
"""

from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy
from typing import Any, Mapping

from .execution_contract import canonical_json


MH1_SCHEMA = "symmetrix.execution.mh1_uvu"
MH1_SCHEMA_VERSION = 3
MH1_CONTRACT_TAG = f"{MH1_SCHEMA}/{MH1_SCHEMA_VERSION}"
MH1_CONTRACT_KEY = "MH1_UVU"
MH1_V4_SCHEMA_VERSION = 4
MH1_V4_CONTRACT_TAG = f"{MH1_SCHEMA}/{MH1_V4_SCHEMA_VERSION}"
MH1_NODE_PROGRAM_TAG = "symmetrix.execution.mh1.node-program/1"
MH1_NODE_RUNTIME_LAYOUT_TAG = "symmetrix.execution.mh1.node-runtime-layout/1"

_MH1_LEGACY_SCHEMA_VERSION = 1
_MH1_LEGACY_CONTRACT_TAG = f"{MH1_SCHEMA}/{_MH1_LEGACY_SCHEMA_VERSION}"
_MH1_V2_SCHEMA_VERSION = 2
_MH1_V2_CONTRACT_TAG = f"{MH1_SCHEMA}/{_MH1_V2_SCHEMA_VERSION}"
_MH1_V3_SCHEMA_VERSION = 3
_MH1_V3_CONTRACT_TAG = f"{MH1_SCHEMA}/{_MH1_V3_SCHEMA_VERSION}"

_INTERACTION_CLASS = "RealAgnosticResidualNonLinearInteractionBlock"
_IRREP = re.compile(r"(?P<multiplicity>[0-9]+)x(?P<l>[0-9]+)(?P<parity>[eo])")
_LAYOUTS_V2 = {
    "source_values": "node,input_1_component",
    "edge_prefix": "edge,prefix_component",
    "edge_harmonics": "edge,input_2_component",
    "messages": "receiver,output_component",
    "source_adjoint": "source,input_1_component",
    "edge_prefix_adjoint": "edge,prefix_component",
    "edge_harmonics_adjoint": "edge,input_2_component",
}
_LAYOUTS = {
    "source_values": {
        "axes": ["node", "input_1_component"],
        "feature_layout": "ir_mul",
    },
    "edge_prefix": {
        "axes": ["edge", "prefix_component"],
        "feature_layout": "dense",
    },
    "edge_harmonics": {
        "axes": ["edge", "input_2_component"],
        "feature_layout": "ir_mul",
    },
    "messages": {
        "axes": ["receiver", "output_component"],
        "feature_layout": "ir_mul",
    },
    "source_adjoint": {
        "axes": ["source", "input_1_component"],
        "feature_layout": "ir_mul",
    },
    "edge_prefix_adjoint": {
        "axes": ["edge", "prefix_component"],
        "feature_layout": "dense",
    },
    "edge_harmonics_adjoint": {
        "axes": ["edge", "input_2_component"],
        "feature_layout": "ir_mul",
    },
    "runtime_node_feature_layout": "mul_ir",
    "generated_node_feature_layout": "ir_mul",
}
_DERIVATIVES = {
    "forward": True,
    "state_adjoint": True,
    "source_adjoint": True,
    "coordinate": {
        "radial": True,
        "harmonics": True,
        "cutoff": True,
    },
    "parameter_gradients": False,
    "double_backward": False,
}
_NODE_EXECUTION = {
    "forward_phases": [
        "prepare_layer_0",
        "finish_layer_0_prepare_layer_1",
        "finish_layer_1",
    ],
    "forward_schedule": [
        {"kind": "node_program", "phase": "prepare_layer_0"},
        {"kind": "edge_tensor_product", "layer": 0},
        {
            "kind": "node_program",
            "phase": "finish_layer_0_prepare_layer_1",
        },
        {"kind": "edge_tensor_product", "layer": 1},
        {"kind": "node_program", "phase": "finish_layer_1"},
    ],
    "reverse_phases": [
        "reverse_layer_1",
        "propagate_layer_1_reverse_layer_0",
    ],
    "reverse_schedule": [
        {"kind": "node_program", "phase": "reverse_layer_1"},
        {"kind": "edge_tensor_product_reverse", "layer": 1},
        {
            "kind": "node_program",
            "phase": "propagate_layer_1_reverse_layer_0",
        },
        {"kind": "edge_tensor_product_reverse", "layer": 0},
    ],
    "tensor_product_is_a_global_dependency_boundary": True,
}
_NODE_DERIVATIVES = {
    "state_adjoint": True,
    "message_adjoint": True,
    "density_adjoint": True,
    "parameter_gradients": False,
    "type_gradients": False,
    "double_backward": False,
}
_LINEAR_RUNTIME_PARAMETER_ORDER = [
    "instruction_weights_row_major",
    "bias",
    "output_mask_ir_mul",
]
_PRODUCT_COEFFICIENT_ORDER = [
    "block",
    "output_component",
    "degree_then_lexicographic_term",
    "element",
    "channel",
]


def _fingerprint(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_shape(value: Any, name: str, rank: int) -> list[int]:
    if not isinstance(value, list) or len(value) != rank:
        raise ValueError(f"{name} must be a rank-{rank} tensor shape")
    return [_require_int(item, name, 0) for item in value]


def _irrep_blocks(signature: Any, name: str) -> tuple[str, list[dict[str, int | str]]]:
    if not isinstance(signature, str):
        raise ValueError(f"{name} must be an irrep signature")
    compact = signature.replace(" ", "")
    parts = compact.split("+") if compact else []
    blocks = []
    offset = 0
    for index, part in enumerate(parts):
        match = _IRREP.fullmatch(part)
        if match is None:
            raise ValueError(f"{name} contains an unsupported irrep block")
        multiplicity = int(match.group("multiplicity"))
        l_value = int(match.group("l"))
        if multiplicity <= 0:
            raise ValueError(f"{name} multiplicities must be positive")
        components = 2 * l_value + 1
        blocks.append(
            {
                "index": index,
                "multiplicity": multiplicity,
                "l": l_value,
                "parity": match.group("parity"),
                "components": components,
                "offset": offset,
            }
        )
        offset += multiplicity * components
    if not blocks:
        raise ValueError(f"{name} must not be empty")
    canonical = "+".join(
        f"{block['multiplicity']}x{block['l']}{block['parity']}" for block in blocks
    )
    return canonical, blocks


def _tensor_shape(definition: Mapping[str, Any], name: str, rank: int) -> list[int]:
    tensor = definition.get(name)
    if not isinstance(tensor, Mapping):
        raise ValueError(f"{name} must be a serialized tensor")
    return _require_shape(tensor.get("shape"), f"{name}.shape", rank)


def _tensor_values(
    definition: Mapping[str, Any], name: str, rank: int
) -> tuple[list[int], list[Any]]:
    shape = _tensor_shape(definition, name, rank)
    tensor = definition[name]
    values = tensor.get("values")
    if not isinstance(values, list) or len(values) != math.prod(shape):
        raise ValueError(f"{name}.values are inconsistent with its shape")
    return shape, values


def _linear_output_dimension(definition: Any, name: str) -> int:
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be a serialized e3nn linear")
    _, blocks = _irrep_blocks(definition.get("irreps_out"), f"{name}.irreps_out")
    return sum(
        int(block["multiplicity"]) * int(block["components"]) for block in blocks
    )


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _irrep_dimension(blocks: list[dict[str, int | str]]) -> int:
    return sum(
        int(block["multiplicity"]) * int(block["components"]) for block in blocks
    )


def _block_identity(block: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(block["multiplicity"]),
        int(block["l"]),
        str(block["parity"]),
    )


def _linear_descriptor(definition: Any, name: str) -> dict[str, Any]:
    """Describe an e3nn linear while keeping all tensor values runtime-only."""

    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be a serialized e3nn linear")
    irreps_in, input_blocks = _irrep_blocks(
        definition.get("irreps_in"), f"{name}.irreps_in"
    )
    irreps_out, output_blocks = _irrep_blocks(
        definition.get("irreps_out"), f"{name}.irreps_out"
    )
    input_dimension = _irrep_dimension(input_blocks)
    output_dimension = _irrep_dimension(output_blocks)
    instructions = definition.get("instructions")
    if not isinstance(instructions, list):
        raise ValueError(f"{name}.instructions must be a list")

    paths = []
    weight_count = 0
    for index, instruction in enumerate(instructions):
        if not isinstance(instruction, Mapping):
            raise ValueError(f"{name}.instructions[{index}] must be an object")
        input_index = _require_int(
            instruction.get("i_in"), f"{name}.instructions[{index}].i_in"
        )
        output_index = _require_int(
            instruction.get("i_out"), f"{name}.instructions[{index}].i_out"
        )
        if input_index >= len(input_blocks) or output_index >= len(output_blocks):
            raise ValueError(f"{name}.instructions[{index}] irrep index is invalid")
        input_block = input_blocks[input_index]
        output_block = output_blocks[output_index]
        if (
            input_block["l"] != output_block["l"]
            or input_block["parity"] != output_block["parity"]
        ):
            raise ValueError(f"{name}.instructions[{index}] is not equivariant")
        path_shape = _require_shape(
            instruction.get("path_shape"),
            f"{name}.instructions[{index}].path_shape",
            2,
        )
        expected_shape = [
            int(input_block["multiplicity"]),
            int(output_block["multiplicity"]),
        ]
        if path_shape != expected_shape:
            raise ValueError(f"{name}.instructions[{index}] path shape is inconsistent")
        if instruction.get("has_weight", True) is not True:
            raise ValueError(f"{name}.instructions[{index}] must have a weight")
        path_weight = _require_finite_number(
            instruction.get("path_weight", 1.0),
            f"{name}.instructions[{index}].path_weight",
        )
        count = math.prod(path_shape)
        paths.append(
            {
                "index": index,
                "input_block": input_index,
                "output_block": output_index,
                "l": int(input_block["l"]),
                "parity": input_block["parity"],
                "path_shape": path_shape,
                "path_weight": path_weight,
                "weight_offset": weight_count,
                "weight_count": count,
            }
        )
        weight_count += count

    weight_shape, _ = _tensor_values(definition, "weight", 1)
    bias_shape, _ = _tensor_values(definition, "bias", 1)
    output_mask_shape, _ = _tensor_values(definition, "output_mask", 1)
    if weight_shape != [weight_count]:
        raise ValueError(f"{name}.weight shape is inconsistent")
    if bias_shape not in ([0], [output_dimension]):
        raise ValueError(f"{name}.bias shape is inconsistent")
    if output_mask_shape != [output_dimension]:
        raise ValueError(f"{name}.output_mask shape is inconsistent")

    bias_count = bias_shape[0]
    output_mask_count = output_mask_shape[0]
    return {
        "irreps": {"input": irreps_in, "output": irreps_out},
        "blocks": {
            "input": deepcopy(input_blocks),
            "output": deepcopy(output_blocks),
        },
        "dimensions": {
            "input": input_dimension,
            "output": output_dimension,
        },
        "instructions": paths,
        "runtime_parameters": {
            "scalar_type": "runtime_precision",
            "order": deepcopy(_LINEAR_RUNTIME_PARAMETER_ORDER),
            "segments": [
                {"name": "weight", "offset": 0, "count": weight_count},
                {
                    "name": "bias",
                    "offset": weight_count,
                    "count": bias_count,
                },
                {
                    "name": "output_mask",
                    "offset": weight_count + bias_count,
                    "count": output_mask_count,
                },
            ],
            "parameter_count": weight_count + bias_count + output_mask_count,
        },
    }


def _linear_structure_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_blocks": [
            {"l": block["l"], "parity": block["parity"]}
            for block in descriptor["blocks"]["input"]
        ],
        "output_blocks": [
            {"l": block["l"], "parity": block["parity"]}
            for block in descriptor["blocks"]["output"]
        ],
        "paths": [
            {
                "input_block": path["input_block"],
                "output_block": path["output_block"],
                "l": path["l"],
                "parity": path["parity"],
            }
            for path in descriptor["instructions"]
        ],
        "has_bias": descriptor["runtime_parameters"]["segments"][1]["count"] > 0,
    }


def _activation_constants(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [
        _require_finite_number(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _gate_descriptor(definition: Any, name: str) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be a serialized gate")
    parsed = {}
    blocks = {}
    for key in (
        "irreps_in",
        "irreps_out",
        "irreps_scalars",
        "irreps_gates",
        "irreps_gated",
    ):
        parsed[key], blocks[key] = _irrep_blocks(definition.get(key), f"{name}.{key}")
    if definition.get("scalar_activation") != "silu":
        raise ValueError(f"{name}.scalar_activation must be 'silu'")
    if definition.get("gate_activation") != "sigmoid":
        raise ValueError(f"{name}.gate_activation must be 'sigmoid'")
    scalar_constants = _activation_constants(
        definition.get("scalar_activation_constants"),
        f"{name}.scalar_activation_constants",
    )
    gate_constants = _activation_constants(
        definition.get("gate_activation_constants"),
        f"{name}.gate_activation_constants",
    )
    if len(scalar_constants) != len(blocks["irreps_scalars"]):
        raise ValueError(f"{name} scalar activation count is inconsistent")
    if len(gate_constants) != len(blocks["irreps_gated"]):
        raise ValueError(f"{name} gate activation count is inconsistent")
    if any(
        int(block["l"]) != 0 or block["parity"] != "e"
        for block in blocks["irreps_scalars"] + blocks["irreps_gates"]
    ):
        raise ValueError(f"{name} scalars and gates must be even scalars")

    dimensions = {key: _irrep_dimension(value) for key, value in blocks.items()}
    planned_gate_count = sum(
        int(block["multiplicity"]) for block in blocks["irreps_gated"]
    )
    if (
        dimensions["irreps_in"]
        != dimensions["irreps_scalars"]
        + dimensions["irreps_gates"]
        + dimensions["irreps_gated"]
        or dimensions["irreps_out"]
        != dimensions["irreps_scalars"] + dimensions["irreps_gated"]
        or dimensions["irreps_gates"] != planned_gate_count
    ):
        raise ValueError(f"{name} irrep dimensions are inconsistent")
    expected_output = [
        _block_identity(block)
        for block in blocks["irreps_scalars"] + blocks["irreps_gated"]
    ]
    if [_block_identity(block) for block in blocks["irreps_out"]] != expected_output:
        raise ValueError(f"{name}.irreps_out is inconsistent")

    gate_offset = dimensions["irreps_scalars"]
    gated_offset = gate_offset + dimensions["irreps_gates"]
    output_offset = dimensions["irreps_scalars"]
    pairing = []
    for index, block in enumerate(blocks["irreps_gated"]):
        multiplicity = int(block["multiplicity"])
        width = int(block["components"])
        pairing.append(
            {
                "gated_block": index,
                "multiplicity": multiplicity,
                "l": int(block["l"]),
                "parity": block["parity"],
                "gate_offset": gate_offset,
                "gated_offset": gated_offset,
                "output_offset": output_offset,
                "component_width": width,
            }
        )
        gate_offset += multiplicity
        gated_offset += multiplicity * width
        output_offset += multiplicity * width

    return {
        "irreps": parsed,
        "blocks": {key: deepcopy(value) for key, value in blocks.items()},
        "dimensions": dimensions,
        "scalar_activation": {
            "name": "silu",
            "constants": scalar_constants,
        },
        "gate_activation": {
            "name": "sigmoid",
            "constants": gate_constants,
        },
        "pairing": pairing,
    }


def _gate_structure_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scalar_activation": descriptor["scalar_activation"]["name"],
        "gate_activation": descriptor["gate_activation"]["name"],
        "scalar_blocks": [
            {"l": block["l"], "parity": block["parity"]}
            for block in descriptor["blocks"]["irreps_scalars"]
        ],
        "gated_blocks": [
            {"l": block["l"], "parity": block["parity"]}
            for block in descriptor["blocks"]["irreps_gated"]
        ],
    }


def _product_descriptor(index: int, definition: Any) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError(f"products[{index}] must be an object")
    if definition.get("use_sc") is not True:
        raise ValueError(f"products[{index}] must use a skip connection")
    if definition.get("use_agnostic_product") is not True:
        raise ValueError(f"products[{index}] must use an agnostic product")
    symmetric = definition.get("symmetric_contractions")
    if not isinstance(symmetric, Mapping):
        raise ValueError(f"products[{index}].symmetric_contractions must be an object")

    irreps_in, input_blocks = _irrep_blocks(
        symmetric.get("irreps_in"),
        f"products[{index}].symmetric_contractions.irreps_in",
    )
    irreps_out, output_blocks = _irrep_blocks(
        symmetric.get("irreps_out"),
        f"products[{index}].symmetric_contractions.irreps_out",
    )
    channels = int(input_blocks[0]["multiplicity"])
    if len(input_blocks) not in (3, 4) or any(
        int(block["multiplicity"]) != channels
        or int(block["l"]) != block_index
        or block["parity"] != ("e" if block_index % 2 == 0 else "o")
        for block_index, block in enumerate(input_blocks)
    ):
        raise ValueError(
            f"products[{index}] input irreps are outside the MH-1 support matrix"
        )
    expected_output = (
        [(channels, 0, "e"), (channels, 1, "o")] if index == 0 else [(channels, 0, "e")]
    )
    if [_block_identity(block) for block in output_blocks] != expected_output:
        raise ValueError(
            f"products[{index}] output irreps are outside the MH-1 support matrix"
        )

    for field in ("node_feats_irreps", "target_irreps"):
        declared, _ = _irrep_blocks(definition.get(field), f"products[{index}].{field}")
        if declared != irreps_out:
            raise ValueError(f"products[{index}].{field} is inconsistent")
    linear = _linear_descriptor(definition.get("linear"), f"products[{index}].linear")
    if (
        linear["irreps"]["input"] != irreps_out
        or linear["irreps"]["output"] != irreps_out
    ):
        raise ValueError(f"products[{index}].linear irreps are inconsistent")

    contractions = symmetric.get("contractions")
    if not isinstance(contractions, list) or len(contractions) != len(output_blocks):
        raise ValueError(f"products[{index}] contraction outputs are inconsistent")
    angular_dimension = sum(int(block["components"]) for block in input_blocks)
    block_programs = []
    flat_terms = []
    components = []
    coefficient_offset = 0
    angular_offset = 0
    for block_index, (raw, output_block) in enumerate(zip(contractions, output_blocks)):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"products[{index}].contractions[{block_index}] must be an object"
            )
        correlation = _require_int(
            raw.get("correlation"),
            f"products[{index}].contractions[{block_index}].correlation",
            1,
        )
        if correlation != 3:
            raise ValueError(
                f"products[{index}] requires correlation-three contractions"
            )
        weights = raw.get("weights")
        u_tensors = raw.get("u_tensors")
        if not isinstance(weights, list) or len(weights) != correlation - 1:
            raise ValueError(f"products[{index}] contraction weights are inconsistent")
        if not isinstance(u_tensors, list) or len(u_tensors) != correlation:
            raise ValueError(f"products[{index}] U tensors are inconsistent")

        width = int(output_block["components"])
        component_terms: list[set[tuple[int, tuple[int, ...]]]] = [
            set() for _ in range(width)
        ]
        degree_programs = []
        element_count: int | None = None
        for degree in range(1, correlation + 1):
            output_axes = 0 if int(output_block["l"]) == 0 else 1
            u_shape, u_values = _tensor_values(
                {"u": u_tensors[degree - 1]},
                "u",
                degree + output_axes + 1,
            )
            if output_axes and u_shape[0] != width:
                raise ValueError(f"products[{index}] U output shape is inconsistent")
            if any(
                u_shape[output_axes + axis] != angular_dimension
                for axis in range(degree)
            ):
                raise ValueError(f"products[{index}] U angular shape is inconsistent")
            parameters = u_shape[-1]
            if parameters <= 0:
                raise ValueError(
                    f"products[{index}] U parameter count must be positive"
                )
            selected_weights = (
                raw.get("weights_max")
                if degree == correlation
                else weights[correlation - degree - 1]
            )
            weight_shape, _ = _tensor_values(
                {"weights": selected_weights}, "weights", 3
            )
            if weight_shape[1:] != [parameters, channels]:
                raise ValueError(
                    f"products[{index}] contraction weight shape is inconsistent"
                )
            if weight_shape[0] != 1:
                raise ValueError(
                    f"products[{index}] agnostic products require one coefficient element"
                )
            if element_count is None:
                element_count = weight_shape[0]
            elif element_count != weight_shape[0]:
                raise ValueError(
                    f"products[{index}] element dimensions are inconsistent"
                )

            numeric_u = []
            for value_index, value in enumerate(u_values):
                numeric_u.append(
                    _require_finite_number(
                        value,
                        f"products[{index}].contractions[{block_index}]"
                        f".u_tensors[{degree - 1}].values[{value_index}]",
                    )
                )
            tuple_count = angular_dimension**degree
            for component in range(width):
                component_base = (
                    component * tuple_count * parameters if output_axes else 0
                )
                for tuple_index in range(tuple_count):
                    value_base = component_base + tuple_index * parameters
                    if not any(
                        numeric_u[value_base + parameter] != 0.0
                        for parameter in range(parameters)
                    ):
                        continue
                    remainder = tuple_index
                    angular_indices = [0] * degree
                    for axis in range(degree - 1, -1, -1):
                        angular_indices[axis] = remainder % angular_dimension
                        remainder //= angular_dimension
                    component_terms[component].add(
                        (degree, tuple(sorted(angular_indices)))
                    )
            degree_programs.append(
                {
                    "degree": degree,
                    "parameters": parameters,
                    "u_shape": u_shape,
                    "weight_shape": weight_shape,
                }
            )

        block_term_begin = len(flat_terms)
        for component, terms in enumerate(component_terms):
            term_begin = len(flat_terms)
            for degree, angular_indices in sorted(terms):
                flat_terms.append(
                    {
                        "index": len(flat_terms),
                        "output_block": block_index,
                        "block_component": component,
                        "degree": degree,
                        "angular_indices": list(angular_indices),
                    }
                )
            components.append(
                {
                    "output_block": block_index,
                    "block_component": component,
                    "output_component_offset": (
                        int(output_block["offset"]) + component * channels
                    ),
                    "output_channel_stride": 1,
                    "term_begin": term_begin,
                    "term_end": len(flat_terms),
                }
            )
        term_count = len(flat_terms) - block_term_begin
        coefficient_count = term_count * int(element_count) * channels
        block_programs.append(
            {
                "index": block_index,
                "l": int(output_block["l"]),
                "parity": output_block["parity"],
                "width": width,
                "angular_offset": angular_offset,
                "term_begin": block_term_begin,
                "term_end": len(flat_terms),
                "degree_programs": degree_programs,
                "runtime_coefficients": {
                    "element_count": int(element_count),
                    "channel_count": channels,
                    "offset": coefficient_offset,
                    "count": coefficient_count,
                    "order": deepcopy(_PRODUCT_COEFFICIENT_ORDER),
                    "source": "runtime_folded_u_times_contraction_weights",
                },
            }
        )
        coefficient_offset += coefficient_count
        angular_offset += width

    if not flat_terms:
        raise ValueError(f"products[{index}] compiled product topology is empty")
    angular_layout = []
    for block in input_blocks:
        for component in range(int(block["components"])):
            angular_layout.append(
                {
                    "angular_index": len(angular_layout),
                    "component_offset": (int(block["offset"]) + component * channels),
                    "channel_stride": 1,
                }
            )
    return {
        "index": index,
        "irreps": {"input": irreps_in, "output": irreps_out},
        "blocks": {
            "input": deepcopy(input_blocks),
            "output": deepcopy(output_blocks),
        },
        "dimensions": {
            "channels": channels,
            "angular": angular_dimension,
            "input": _irrep_dimension(input_blocks),
            "output": _irrep_dimension(output_blocks),
        },
        "correlation": 3,
        "use_sc": True,
        "use_agnostic_product": True,
        "feature_layout": "ir_mul",
        "angular_layout": angular_layout,
        "blocks_program": block_programs,
        "components": components,
        "terms": flat_terms,
        "runtime_coefficient_count": coefficient_offset,
        "linear": linear,
    }


def _product_structure_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "input_blocks": [
            {"l": block["l"], "parity": block["parity"]}
            for block in descriptor["blocks"]["input"]
        ],
        "output_blocks": [
            {"l": block["l"], "parity": block["parity"]}
            for block in descriptor["blocks"]["output"]
        ],
        "correlation": descriptor["correlation"],
        "use_sc": descriptor["use_sc"],
        "use_agnostic_product": descriptor["use_agnostic_product"],
        "feature_layout": descriptor["feature_layout"],
        "angular_layout": descriptor["angular_layout"],
        "terms": [
            {
                "output_block": term["output_block"],
                "block_component": term["block_component"],
                "degree": term["degree"],
                "angular_indices": term["angular_indices"],
            }
            for term in descriptor["terms"]
        ],
        "linear": _linear_structure_payload(descriptor["linear"]),
    }


def _readout_descriptor(index: int, definition: Any) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError(f"readouts[{index}] must be an object")
    expected_class = "LinearReadoutBlock" if index == 0 else "NonLinearReadoutBlock"
    if definition.get("class") != expected_class:
        raise ValueError(
            "MH-1 node generation requires a linear then nonlinear readout"
        )
    if index == 0:
        linear = _linear_descriptor(
            definition.get("linear"), f"readouts[{index}].linear"
        )
        if linear["irreps"]["output"] != "1x0e":
            raise ValueError(f"readouts[{index}] must produce one even scalar")
        return {"index": index, "class": expected_class, "linear": linear}

    linear_1 = _linear_descriptor(
        definition.get("linear_1"), f"readouts[{index}].linear_1"
    )
    linear_2 = _linear_descriptor(
        definition.get("linear_2"), f"readouts[{index}].linear_2"
    )
    if linear_1["irreps"]["output"] != linear_2["irreps"]["input"]:
        raise ValueError(f"readouts[{index}] hidden irreps are inconsistent")
    if linear_2["irreps"]["output"] != "1x0e":
        raise ValueError(f"readouts[{index}] must produce one even scalar")
    if any(
        int(block["l"]) != 0 or block["parity"] != "e"
        for block in linear_1["blocks"]["output"]
    ):
        raise ValueError(f"readouts[{index}] hidden irreps must be even scalars")
    if definition.get("activation") != "silu":
        raise ValueError(f"readouts[{index}].activation must be 'silu'")
    constants = _activation_constants(
        definition.get("activation_constants"),
        f"readouts[{index}].activation_constants",
    )
    if len(constants) != len(linear_1["blocks"]["output"]):
        raise ValueError(f"readouts[{index}] activation count is inconsistent")
    return {
        "index": index,
        "class": expected_class,
        "linear_1": linear_1,
        "activation": {"name": "silu", "constants": constants},
        "linear_2": linear_2,
    }


def _readout_structure_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if descriptor["class"] == "LinearReadoutBlock":
        return {
            "class": descriptor["class"],
            "linear": _linear_structure_payload(descriptor["linear"]),
        }
    return {
        "class": descriptor["class"],
        "linear_1": _linear_structure_payload(descriptor["linear_1"]),
        "activation": descriptor["activation"]["name"],
        "linear_2": _linear_structure_payload(descriptor["linear_2"]),
    }


def _runtime_parameter_pack(
    role: str, entries: list[tuple[str, str, Mapping[str, Any] | int]]
) -> dict[str, Any]:
    segments = []
    offset = 0
    for name, kind, definition in entries:
        if kind == "e3_linear":
            if not isinstance(definition, Mapping):
                raise TypeError("e3 linear runtime entries require a descriptor")
            layout = deepcopy(definition["runtime_parameters"])
            count = int(layout["parameter_count"])
            segment = {
                "name": name,
                "kind": kind,
                "offset": offset,
                "count": count,
                "layout": layout,
            }
        else:
            count = int(definition)
            if count <= 0:
                raise ValueError("runtime scalar/coefficient counts must be positive")
            segment = {
                "name": name,
                "kind": kind,
                "offset": offset,
                "count": count,
            }
        segments.append(segment)
        offset += count
    pack = {
        "tag": MH1_NODE_RUNTIME_LAYOUT_TAG,
        "role": role,
        "scalar_type": "float32",
        "segments": segments,
        "parameter_count": offset,
    }
    pack["fingerprint"] = _fingerprint(pack)
    return pack


def _readout_runtime_entries(
    descriptor: Mapping[str, Any],
) -> list[tuple[str, str, Mapping[str, Any] | int]]:
    if descriptor["class"] == "LinearReadoutBlock":
        return [("linear", "e3_linear", descriptor["linear"])]
    return [
        ("linear_1", "e3_linear", descriptor["linear_1"]),
        ("linear_2", "e3_linear", descriptor["linear_2"]),
    ]


def _node_layer_descriptor(
    index: int,
    definition: Any,
    tensor_product: Mapping[str, Any],
    product: Mapping[str, Any],
    readout: Mapping[str, Any],
    input_irreps: str,
    node_embedding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError(f"interactions[{index}] must be an object")
    linears = {
        "linear_up": _linear_descriptor(
            definition.get("linear_up"), f"interactions[{index}].linear_up"
        ),
        "linear_res": _linear_descriptor(
            definition.get("linear_res"), f"interactions[{index}].linear_res"
        ),
        "skip": _linear_descriptor(
            definition.get("skip_tp"), f"interactions[{index}].skip_tp"
        ),
        "linear_1": _linear_descriptor(
            definition.get("linear_1"), f"interactions[{index}].linear_1"
        ),
        "linear_2": _linear_descriptor(
            definition.get("linear_2"), f"interactions[{index}].linear_2"
        ),
    }
    gate = _gate_descriptor(definition.get("gate"), f"interactions[{index}].gate")
    if linears["linear_up"]["irreps"]["input"] != input_irreps:
        raise ValueError(f"interactions[{index}].linear_up input is inconsistent")
    if linears["skip"]["irreps"]["input"] != input_irreps:
        raise ValueError(f"interactions[{index}].skip_tp input is inconsistent")
    if (
        linears["linear_up"]["irreps"]["output"]
        != tensor_product["irreps"]["irreps_in1"]
    ):
        raise ValueError(f"interactions[{index}].linear_up output is inconsistent")
    if (
        linears["linear_res"]["irreps"]["input"]
        != linears["linear_up"]["irreps"]["output"]
        or linears["linear_res"]["irreps"]["output"] != gate["irreps"]["irreps_in"]
    ):
        raise ValueError(f"interactions[{index}].linear_res is inconsistent")
    if (
        linears["linear_1"]["irreps"]["input"] != tensor_product["irreps"]["irreps_out"]
        or linears["linear_1"]["irreps"]["output"] != gate["irreps"]["irreps_in"]
    ):
        raise ValueError(f"interactions[{index}].linear_1 is inconsistent")
    if (
        linears["linear_2"]["irreps"]["input"] != gate["irreps"]["irreps_out"]
        or linears["linear_2"]["irreps"]["output"] != product["irreps"]["input"]
    ):
        raise ValueError(f"interactions[{index}].linear_2 is inconsistent")
    if linears["skip"]["irreps"]["output"] != product["irreps"]["output"]:
        raise ValueError(f"interactions[{index}].skip_tp output is inconsistent")
    readout_input = (
        readout["linear"]["irreps"]["input"]
        if readout["class"] == "LinearReadoutBlock"
        else readout["linear_1"]["irreps"]["input"]
    )
    if readout_input != product["irreps"]["output"]:
        raise ValueError(f"readouts[{index}] input is inconsistent")

    for name in ("alpha", "beta"):
        _require_finite_number(definition.get(name), f"interactions[{index}].{name}")

    linear_entries: list[tuple[str, str, Mapping[str, Any] | int]] = []
    if node_embedding is not None:
        linear_entries.append(("node_embedding", "e3_linear", node_embedding))
    linear_entries.extend(
        (name, "e3_linear", descriptor) for name, descriptor in linears.items()
    )
    linear_entries.extend(
        [
            ("product_linear", "e3_linear", product["linear"]),
            ("normalization_alpha_beta", "runtime_scalars", 2),
        ]
    )
    packs = {
        "linear": _runtime_parameter_pack(f"layer_{index}.linear", linear_entries),
        "product": _runtime_parameter_pack(
            f"layer_{index}.product",
            [
                (
                    "folded_correlation_coefficients",
                    "product_coefficients",
                    int(product["runtime_coefficient_count"]),
                )
            ],
        ),
        "readout": _runtime_parameter_pack(
            f"layer_{index}.readout", _readout_runtime_entries(readout)
        ),
    }
    return {
        "index": index,
        "input_irreps": input_irreps,
        "output_irreps": product["irreps"]["output"],
        "linears": linears,
        "normalization": {
            "kind": "linear_over_alpha_plus_beta_density_plus_residual",
            "runtime_scalars": ["alpha", "beta"],
        },
        "gate": gate,
        "product": product,
        "readout": readout,
        "runtime_parameter_packs": packs,
        "persistent_state": {
            "feature_layout": "ir_mul",
            "retain": ["up", "messages", "density"]
            + (["output"] if index == 0 else []),
            "recompute_node_local_intermediates_in_reverse": True,
            "node_state_policy": {
                "default": "full-retention-v1",
                "supported": [
                    "full-retention-v1",
                    "recompute-v1",
                    "reuse-adjoints-v1",
                    "retain-interaction-v1",
                ],
                "retained_tensors": {
                    "pre_gate": int(linears["linear_res"]["dimensions"]["output"]),
                    "interaction_output": int(
                        linears["linear_2"]["dimensions"]["output"]
                    ),
                },
            },
        },
        "derivatives": {
            "requires_tp_source_state_adjoint": index > 0,
            "requires_previous_layer_state_adjoint": index > 0,
            "parameter_gradients": False,
            "type_gradients": False,
        },
    }


def _node_program_structure_payload(program: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tag": program["tag"],
        "feature_layout": program["feature_layout"],
        "embedding": _linear_structure_payload(program["input"]["embedding"]),
        "layers": [
            {
                "index": layer["index"],
                "linears": {
                    name: _linear_structure_payload(descriptor)
                    for name, descriptor in layer["linears"].items()
                },
                "normalization": layer["normalization"]["kind"],
                "gate": _gate_structure_payload(layer["gate"]),
                "product": _product_structure_payload(layer["product"]),
                "readout": _readout_structure_payload(layer["readout"]),
                "persistent_state": layer["persistent_state"],
                "derivatives": layer["derivatives"],
            }
            for layer in program["layers"]
        ],
        "execution": program["execution"],
        "derivatives": program["derivatives"],
    }


def _pre_retention_node_program_structure_payload(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _node_program_structure_payload(program)
    for layer in payload["layers"]:
        layer.pop("persistent_state")
    return payload


def _node_program_descriptor(
    model_data: Mapping[str, Any],
    tensor_products: list[Mapping[str, Any]],
) -> dict[str, Any]:
    interactions = model_data.get("interactions")
    products = model_data.get("products")
    readouts = model_data.get("readouts")
    if not isinstance(interactions, list) or len(interactions) != 2:
        raise ValueError("MH-1 node generation requires exactly two interactions")
    if not isinstance(products, list) or len(products) != 2:
        raise ValueError("MH-1 node generation requires exactly two products")
    if not isinstance(readouts, list) or len(readouts) != 2:
        raise ValueError("MH-1 node generation requires exactly two readouts")
    embedding = _linear_descriptor(model_data.get("node_embedding"), "node_embedding")
    if any(
        int(block["l"]) != 0 or block["parity"] != "e"
        for block in embedding["blocks"]["input"] + embedding["blocks"]["output"]
    ):
        raise ValueError("MH-1 node embedding must contain only even scalars")
    if (
        len(embedding["blocks"]["input"]) != 1
        or len(embedding["blocks"]["output"]) != 1
        or len(embedding["instructions"]) != 1
        or embedding["instructions"][0]["input_block"] != 0
        or embedding["instructions"][0]["output_block"] != 0
    ):
        raise ValueError("MH-1 node embedding must be one direct scalar table lookup")
    product_descriptors = [
        _product_descriptor(index, product) for index, product in enumerate(products)
    ]
    readout_descriptors = [
        _readout_descriptor(index, readout) for index, readout in enumerate(readouts)
    ]
    layers = []
    input_irreps = embedding["irreps"]["output"]
    for index in range(2):
        layer = _node_layer_descriptor(
            index,
            interactions[index],
            tensor_products[index],
            product_descriptors[index],
            readout_descriptors[index],
            input_irreps,
            embedding if index == 0 else None,
        )
        layers.append(layer)
        input_irreps = layer["output_irreps"]

    runtime_layout = {
        "tag": MH1_NODE_RUNTIME_LAYOUT_TAG,
        "scalar_type": "float32",
        "layers": [
            {
                "index": layer["index"],
                "packs": deepcopy(layer["runtime_parameter_packs"]),
            }
            for layer in layers
        ],
    }
    runtime_layout["fingerprint"] = _fingerprint(runtime_layout)
    program = {
        "tag": MH1_NODE_PROGRAM_TAG,
        "layer_count": 2,
        "feature_layout": "ir_mul",
        "input": {
            "kind": "runtime_type_embedding_table",
            "type_indexing": "model_element_index_per_node",
            "embedding": embedding,
        },
        "layers": layers,
        "execution": deepcopy(_NODE_EXECUTION),
        "derivatives": deepcopy(_NODE_DERIVATIVES),
        "runtime_layout": runtime_layout,
    }
    program["structure_fingerprint"] = _fingerprint(
        _node_program_structure_payload(program)
    )
    return program


def _mlp_descriptor(
    definition: Any,
    *,
    radial_dimension: int,
    source_condition_dimension: int,
    target_condition_dimension: int,
    expected_output_dimension: int,
    name: str,
) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be an affine MLP")
    raw_layers = definition.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError(f"{name}.layers must be a non-empty list")

    layers = []
    previous_dimension: int | None = None
    for index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, Mapping):
            raise ValueError(f"{name}.layers[{index}] must be an object")
        layer_type = raw_layer.get("type")
        if layer_type == "linear":
            output_dimension, input_dimension = _tensor_shape(raw_layer, "weight", 2)
            bias_shape = _tensor_shape(raw_layer, "bias", 1)
            if output_dimension <= 0 or input_dimension <= 0:
                raise ValueError(f"{name}.layers[{index}] dimensions must be positive")
            if bias_shape != [output_dimension]:
                raise ValueError(f"{name}.layers[{index}] bias shape is inconsistent")
            descriptor = {
                "index": index,
                "type": "linear",
                "input_dimension": input_dimension,
                "output_dimension": output_dimension,
                "has_bias": True,
            }
        elif layer_type == "layer_norm":
            normalized_shape = _require_shape(
                raw_layer.get("normalized_shape"),
                f"{name}.layers[{index}].normalized_shape",
                1,
            )
            dimension = normalized_shape[0]
            if dimension <= 0:
                raise ValueError(f"{name}.layers[{index}] dimension must be positive")
            if _tensor_shape(raw_layer, "weight", 1) != [dimension] or _tensor_shape(
                raw_layer, "bias", 1
            ) != [dimension]:
                raise ValueError(f"{name}.layers[{index}] affine shape is inconsistent")
            eps = raw_layer.get("eps")
            if isinstance(eps, bool) or not isinstance(eps, (int, float)):
                raise ValueError(f"{name}.layers[{index}].eps must be numeric")
            eps = float(eps)
            if not math.isfinite(eps) or eps <= 0.0:
                raise ValueError(
                    f"{name}.layers[{index}].eps must be finite and positive"
                )
            input_dimension = output_dimension = dimension
            descriptor = {
                "index": index,
                "type": "layer_norm",
                "dimension": dimension,
                "eps": eps,
                "affine": True,
            }
        elif layer_type == "silu":
            if previous_dimension is None:
                raise ValueError(f"{name}.layers[{index}] SiLU cannot be first")
            input_dimension = output_dimension = previous_dimension
            descriptor = {
                "index": index,
                "type": "silu",
                "dimension": previous_dimension,
            }
        else:
            raise ValueError(f"{name}.layers[{index}] has an unsupported type")
        if previous_dimension is not None and input_dimension != previous_dimension:
            raise ValueError(f"{name}.layers[{index}] input dimension is inconsistent")
        previous_dimension = output_dimension
        layers.append(descriptor)

    if layers[0]["type"] != "linear" or layers[-1]["type"] != "linear":
        raise ValueError(f"{name} must begin and end with linear layers")
    conditioned_input_dimension = (
        radial_dimension + source_condition_dimension + target_condition_dimension
    )
    if layers[0]["input_dimension"] != conditioned_input_dimension:
        raise ValueError(f"{name} conditioned input dimension is inconsistent")
    if layers[-1]["output_dimension"] != expected_output_dimension:
        raise ValueError(f"{name} output dimension is inconsistent")

    prefix_is_identity = len(layers) == 1
    prefix_output_dimension = (
        radial_dimension if prefix_is_identity else int(layers[-1]["input_dimension"])
    )
    return {
        "conditioned_input_dimension": conditioned_input_dimension,
        "dynamic_input_dimension": radial_dimension,
        "source_condition_dimension": source_condition_dimension,
        "target_condition_dimension": target_condition_dimension,
        "layers": layers,
        "factorization": {
            "kind": "conditioned_prefix_final_affine",
            "prefix_layer_count": len(layers) - 1,
            "prefix_is_identity": prefix_is_identity,
            "prefix_output_dimension": prefix_output_dimension,
            "final_affine_input_dimension": prefix_output_dimension,
            "final_affine_output_dimension": expected_output_dimension,
            "conditioned_columns_folded_into_first_affine": True,
        },
    }


def _interaction_structure_payload(
    descriptor: Mapping[str, Any], *, include_density: bool = True
) -> dict[str, Any]:
    payload = {
        "paths": [
            {
                "input_1_l": path["input_1"]["l"],
                "input_2_l": path["input_2"]["l"],
                "output_l": path["output"]["l"],
                "connection_mode": path["connection_mode"],
                "has_weight": path["has_weight"],
                "path_weight": path["path_weight"],
                "sparse_wigner": path["sparse_wigner"],
            }
            for path in descriptor["paths"]
        ],
        "mlp_layer_types": [
            layer["type"] for layer in descriptor["conditioned_mlp"]["layers"]
        ],
        "prefix_is_identity": descriptor["conditioned_mlp"]["factorization"][
            "prefix_is_identity"
        ],
    }
    if include_density:
        payload.update(
            {
                "density_mlp_layer_types": [
                    layer["type"] for layer in descriptor["density_mlp"]["layers"]
                ],
                "density_prefix_is_identity": descriptor["density_mlp"][
                    "factorization"
                ]["prefix_is_identity"],
            }
        )
    return payload


def _semantic_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact parameter-free operator identity without derived hashes."""

    payload = deepcopy(dict(contract))
    payload.pop("semantic_fingerprint", None)
    payload.pop("generation_fingerprint", None)
    payload.pop("structure_fingerprint", None)
    payload.pop("runtime_layout_fingerprint", None)
    for descriptor in payload.get("interactions", []):
        descriptor.pop("generation_fingerprint", None)
        descriptor.pop("structure_fingerprint", None)
    node_program = payload.get("node_program")
    if isinstance(node_program, dict):
        node_program.pop("structure_fingerprint", None)
        runtime_layout = node_program.get("runtime_layout")
        if isinstance(runtime_layout, dict):
            runtime_layout.pop("fingerprint", None)
            for layer in runtime_layout.get("layers", []):
                for pack in layer.get("packs", {}).values():
                    pack.pop("fingerprint", None)
        for layer in node_program.get("layers", []):
            for pack in layer.get("runtime_parameter_packs", {}).values():
                pack.pop("fingerprint", None)
    return payload


def _interaction_descriptor(
    index: int, interaction: Any, radial_dimension: int
) -> dict[str, Any]:
    if not isinstance(interaction, Mapping):
        raise ValueError(f"interactions[{index}] must be an object")
    if interaction.get("class") != _INTERACTION_CLASS:
        raise ValueError(f"interactions[{index}] is not an MH-1 interaction")
    convolution = interaction.get("conv_tp")
    if not isinstance(convolution, Mapping):
        raise ValueError(f"interactions[{index}].conv_tp must be an object")

    irreps = {}
    blocks = {}
    for key in ("irreps_in1", "irreps_in2", "irreps_out"):
        canonical, parsed = _irrep_blocks(
            convolution.get(key), f"interactions[{index}].conv_tp.{key}"
        )
        irreps[key] = canonical
        blocks[key] = parsed
    dimensions = {
        "input_1": sum(
            int(block["multiplicity"]) * int(block["components"])
            for block in blocks["irreps_in1"]
        ),
        "input_2": sum(
            int(block["multiplicity"]) * int(block["components"])
            for block in blocks["irreps_in2"]
        ),
        "output": sum(
            int(block["multiplicity"]) * int(block["components"])
            for block in blocks["irreps_out"]
        ),
    }
    output_mask_shape, output_mask_values = _tensor_values(
        convolution, "output_mask", 1
    )
    if output_mask_shape != [dimensions["output"]]:
        raise ValueError(f"interactions[{index}] output mask shape is inconsistent")

    instructions = convolution.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"interactions[{index}] must contain tensor-product paths")
    paths = []
    weight_offset = 0
    multiplicity: int | None = None
    for path_index, instruction in enumerate(instructions):
        if not isinstance(instruction, Mapping):
            raise ValueError(
                f"interactions[{index}].instructions[{path_index}] is invalid"
            )
        block_indices = {}
        selected_blocks = {}
        for field, irrep_key in (
            ("i_in1", "irreps_in1"),
            ("i_in2", "irreps_in2"),
            ("i_out", "irreps_out"),
        ):
            block_index = _require_int(
                instruction.get(field),
                f"interactions[{index}].instructions[{path_index}].{field}",
            )
            if block_index >= len(blocks[irrep_key]):
                raise ValueError(f"interactions[{index}] path irrep index is invalid")
            block_indices[field] = block_index
            selected_blocks[field] = blocks[irrep_key][block_index]
        source = selected_blocks["i_in1"]
        edge = selected_blocks["i_in2"]
        output = selected_blocks["i_out"]
        path_shape = _require_shape(
            instruction.get("path_shape"),
            f"interactions[{index}].instructions[{path_index}].path_shape",
            2,
        )
        if (
            instruction.get("connection_mode") != "uvu"
            or instruction.get("has_weight") is not True
            or int(edge["multiplicity"]) != 1
            or int(output["multiplicity"]) != int(source["multiplicity"])
            or path_shape != [int(source["multiplicity"]), 1]
        ):
            raise ValueError(
                f"interactions[{index}] is not a tied-channel UVU tensor product"
            )
        if multiplicity is None:
            multiplicity = int(source["multiplicity"])
        elif multiplicity != int(source["multiplicity"]):
            raise ValueError(f"interactions[{index}] UVU multiplicity is not uniform")

        path_weight = instruction.get("path_weight")
        if isinstance(path_weight, bool) or not isinstance(path_weight, (int, float)):
            raise ValueError(f"interactions[{index}] path_weight must be numeric")
        path_weight = float(path_weight)
        if not math.isfinite(path_weight):
            raise ValueError(f"interactions[{index}] path_weight must be finite")
        wigner = instruction.get("wigner_3j")
        if not isinstance(wigner, Mapping):
            raise ValueError(f"interactions[{index}] Wigner tensor is missing")
        wigner_shape = _require_shape(
            wigner.get("shape"),
            f"interactions[{index}].instructions[{path_index}].wigner_3j.shape",
            3,
        )
        expected_shape = [
            int(source["components"]),
            int(edge["components"]),
            int(output["components"]),
        ]
        if wigner_shape != expected_shape:
            raise ValueError(
                f"interactions[{index}] Wigner tensor shape is inconsistent"
            )
        values = wigner.get("values")
        if not isinstance(values, list) or len(values) != math.prod(wigner_shape):
            raise ValueError(
                f"interactions[{index}] Wigner tensor values are inconsistent"
            )
        sparse_terms = []
        for flat, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"interactions[{index}] Wigner coefficients must be numeric"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"interactions[{index}] Wigner coefficients must be finite"
                )
            if value == 0.0:
                continue
            b_dimension, c_dimension = wigner_shape[1:]
            a = flat // (b_dimension * c_dimension)
            remainder = flat % (b_dimension * c_dimension)
            b = remainder // c_dimension
            c = remainder % c_dimension
            sparse_terms.append({"a": a, "b": b, "c": c, "coefficient": value})
        if not sparse_terms:
            raise ValueError(f"interactions[{index}] Wigner tensor must not be zero")

        paths.append(
            {
                "index": path_index,
                "input_1": deepcopy(source),
                "input_2": deepcopy(edge),
                "output": deepcopy(output),
                "connection_mode": "uvu",
                "has_weight": True,
                "path_weight": path_weight,
                "path_shape": path_shape,
                "weight_offset": weight_offset,
                "sparse_wigner": {
                    "shape": wigner_shape,
                    "term_count": len(sparse_terms),
                    "ordering": "input_1_component,input_2_component,output_component",
                    "terms": sparse_terms,
                },
            }
        )
        weight_offset += math.prod(path_shape)

    expected_output_mask = [0.0] * dimensions["output"]
    for path in paths:
        output = path["output"]
        for component in range(int(output["components"])):
            for channel in range(int(output["multiplicity"])):
                expected_output_mask[
                    int(output["offset"])
                    + component * int(output["multiplicity"])
                    + channel
                ] = 1.0
    actual_output_mask = [
        _require_finite_number(
            value, f"interactions[{index}].conv_tp.output_mask[{component}]"
        )
        for component, value in enumerate(output_mask_values)
    ]
    if actual_output_mask != expected_output_mask:
        raise ValueError(
            f"interactions[{index}] output mask does not match tensor-product paths"
        )

    source_condition_dimension = _linear_output_dimension(
        interaction.get("source_embedding"),
        f"interactions[{index}].source_embedding",
    )
    target_condition_dimension = _linear_output_dimension(
        interaction.get("target_embedding"),
        f"interactions[{index}].target_embedding",
    )
    conditioned_mlp = _mlp_descriptor(
        interaction.get("conv_tp_weights"),
        radial_dimension=radial_dimension,
        source_condition_dimension=source_condition_dimension,
        target_condition_dimension=target_condition_dimension,
        expected_output_dimension=weight_offset,
        name=f"interactions[{index}].conv_tp_weights",
    )
    density_mlp = _mlp_descriptor(
        interaction.get("density_fn"),
        radial_dimension=radial_dimension,
        source_condition_dimension=source_condition_dimension,
        target_condition_dimension=target_condition_dimension,
        expected_output_dimension=1,
        name=f"interactions[{index}].density_fn",
    )
    descriptor = {
        "index": index,
        "irreps": irreps,
        "dimensions": {
            **dimensions,
            "channels": multiplicity,
            "weight": weight_offset,
            "radial_embedding": radial_dimension,
            "prefix": conditioned_mlp["factorization"]["prefix_output_dimension"],
            "phi": conditioned_mlp["factorization"]["prefix_output_dimension"],
            "density_prefix": density_mlp["factorization"]["prefix_output_dimension"],
            "density_output": density_mlp["factorization"][
                "final_affine_output_dimension"
            ],
            "input_1_angular": dimensions["input_1"] // int(multiplicity),
        },
        "paths": paths,
        "conditioned_mlp": conditioned_mlp,
        "density_mlp": density_mlp,
    }
    descriptor["generation_fingerprint"] = _fingerprint(descriptor)
    descriptor["structure_fingerprint"] = _fingerprint(
        _interaction_structure_payload(descriptor)
    )
    return descriptor


def _radial_descriptor(model_data: Mapping[str, Any]) -> dict[str, Any]:
    radial = model_data.get("radial_embedding")
    if not isinstance(radial, Mapping):
        raise ValueError("radial_embedding must be an object")
    basis = radial.get("basis")
    cutoff = radial.get("cutoff")
    transform = radial.get("distance_transform")
    if not isinstance(basis, Mapping) or basis.get("type") != "bessel":
        raise ValueError("MH1 UVU generation requires a Bessel radial basis")
    radial_shape = _tensor_shape(basis, "weights", 1)
    if radial_shape[0] <= 0:
        raise ValueError("MH1 UVU radial dimension must be positive")
    if not isinstance(cutoff, Mapping) or cutoff.get("type") != "polynomial":
        raise ValueError("MH1 UVU generation requires a polynomial cutoff")
    power = _require_int(cutoff.get("p"), "radial cutoff power", 1)
    radius = cutoff.get("r_max")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValueError("radial cutoff radius must be numeric")
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radial cutoff radius must be finite and positive")
    if not isinstance(transform, Mapping) or transform.get("type") not in (
        "none",
        "agnesi",
    ):
        raise ValueError("MH1 UVU distance transform is unsupported")
    return {
        "embedding_dimension": radial_shape[0],
        "basis": {"type": "bessel", "layout": "edge,radial_embedding"},
        "cutoff": {
            "type": "polynomial",
            "apply_to_tensor_product": bool(radial.get("apply_cutoff", True)),
            "r_max": radius,
            "power": power,
        },
        "distance_transform": {"type": transform["type"]},
    }


def _contract_structure_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "tag": contract["tag"],
        "interaction_structures": [
            descriptor["structure_fingerprint"]
            for descriptor in contract["interactions"]
        ],
        "radial_basis": contract["radial"]["basis"]["type"],
        "cutoff_type": contract["radial"]["cutoff"]["type"],
        "distance_transform": contract["radial"]["distance_transform"]["type"],
        "layouts": contract["layouts"],
        "derivatives": contract["derivatives"],
    }
    if "node_program" in contract:
        payload["node_program_structure"] = contract["node_program"][
            "structure_fingerprint"
        ]
        payload["runtime_layout_fingerprint"] = contract["runtime_layout_fingerprint"]
    return payload


def _finish_contract_fingerprints(contract: dict[str, Any]) -> dict[str, Any]:
    contract["semantic_fingerprint"] = _fingerprint(_semantic_payload(contract))
    contract["generation_fingerprint"] = _fingerprint(contract)
    contract["structure_fingerprint"] = _fingerprint(
        _contract_structure_payload(contract)
    )
    return contract


def make_execution_mh1_contract(model_data: Mapping[str, Any]) -> dict[str, Any]:
    """Build the parameter-free generated-kernel identity for an MH-1 model."""

    if model_data.get("model_type") != "MACE_Nonlinear":
        raise ValueError("MH1 UVU contracts require model_type MACE_Nonlinear")
    interactions = model_data.get("interactions")
    if not isinstance(interactions, list) or len(interactions) != 2:
        raise ValueError("MH1 UVU contracts require exactly two interactions")
    radial = _radial_descriptor(model_data)
    descriptors = [
        _interaction_descriptor(index, interaction, radial["embedding_dimension"])
        for index, interaction in enumerate(interactions)
    ]
    contract = {
        "tag": MH1_CONTRACT_TAG,
        "schema": MH1_SCHEMA,
        "version": MH1_SCHEMA_VERSION,
        "interaction_count": 2,
        "interactions": descriptors,
        "radial": radial,
        "layouts": deepcopy(_LAYOUTS),
        "derivatives": deepcopy(_DERIVATIVES),
        "accumulator": {
            "mode": "runtime_precision",
            "native_generated": ["float32"],
            "fallback": ["float32", "float64"],
        },
    }
    return _finish_contract_fingerprints(contract)


def make_execution_mh1_v4_contract(model_data: Mapping[str, Any]) -> dict[str, Any]:
    """Build the staged complete-node identity without changing active v3 JIT."""

    contract = make_execution_mh1_contract(model_data)
    contract.pop("semantic_fingerprint")
    contract.pop("generation_fingerprint")
    contract.pop("structure_fingerprint")
    contract["tag"] = MH1_V4_CONTRACT_TAG
    contract["version"] = MH1_V4_SCHEMA_VERSION
    contract["node_program"] = _node_program_descriptor(
        model_data, contract["interactions"]
    )
    contract["runtime_layout_fingerprint"] = contract["node_program"]["runtime_layout"][
        "fingerprint"
    ]
    return _finish_contract_fingerprints(contract)


def _v3_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Project a version-4 node contract onto the active version-3 identity."""

    v3 = deepcopy(dict(contract))
    for name in (
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
        "runtime_layout_fingerprint",
        "node_program",
    ):
        v3.pop(name, None)
    v3["tag"] = _MH1_V3_CONTRACT_TAG
    v3["version"] = _MH1_V3_SCHEMA_VERSION
    return _finish_contract_fingerprints(v3)


def _pre_retention_v4_contract_projection(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a current v4 identity onto its exact pre-retention encoding."""

    legacy = deepcopy(dict(contract))
    for name in (
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
    ):
        legacy.pop(name, None)
    node_program = legacy["node_program"]
    for layer in node_program["layers"]:
        layer["persistent_state"].pop("node_state_policy")
    node_program["structure_fingerprint"] = _fingerprint(
        _pre_retention_node_program_structure_payload(node_program)
    )
    return _finish_contract_fingerprints(legacy)


def project_execution_mh1_v3_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact ABI-v3 identity embedded in a validated v4 model."""

    return _v3_contract_projection(normalize_execution_mh1_v4_contract(contract))


def _v2_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Project a version-3 model contract onto the exact version-2 identity."""

    v2 = deepcopy(dict(contract))
    v2.pop("semantic_fingerprint", None)
    v2.pop("generation_fingerprint", None)
    v2.pop("structure_fingerprint", None)
    v2["tag"] = _MH1_V2_CONTRACT_TAG
    v2["version"] = _MH1_V2_SCHEMA_VERSION
    v2["layouts"] = deepcopy(_LAYOUTS_V2)
    for descriptor in v2["interactions"]:
        descriptor.pop("generation_fingerprint", None)
        descriptor.pop("structure_fingerprint", None)
        descriptor["generation_fingerprint"] = _fingerprint(descriptor)
        descriptor["structure_fingerprint"] = _fingerprint(
            _interaction_structure_payload(descriptor, include_density=True)
        )
    return _finish_contract_fingerprints(v2)


def _legacy_contract_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Project a version-3 model contract onto the exact version-1 identity."""

    legacy = _v2_contract_projection(contract)
    legacy.pop("semantic_fingerprint", None)
    legacy.pop("generation_fingerprint", None)
    legacy.pop("structure_fingerprint", None)
    legacy["tag"] = _MH1_LEGACY_CONTRACT_TAG
    legacy["version"] = _MH1_LEGACY_SCHEMA_VERSION
    for descriptor in legacy["interactions"]:
        descriptor.pop("generation_fingerprint", None)
        descriptor.pop("structure_fingerprint", None)
        descriptor.pop("density_mlp")
        descriptor["dimensions"].pop("density_prefix")
        descriptor["dimensions"].pop("density_output")
        descriptor["generation_fingerprint"] = _fingerprint(descriptor)
        descriptor["structure_fingerprint"] = _fingerprint(
            _interaction_structure_payload(descriptor, include_density=False)
        )
    return _finish_contract_fingerprints(legacy)


def _normalize_execution_mh1_contract_version(
    contract: Mapping[str, Any],
    *,
    tag: str,
    version: int,
    include_density: bool,
    layouts: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = deepcopy(dict(contract))
    supplied_semantic = normalized.pop("semantic_fingerprint", None)
    supplied_generation = normalized.pop("generation_fingerprint", None)
    supplied_structure = normalized.pop("structure_fingerprint", None)
    if normalized.get("tag") != tag:
        raise ValueError(f"MH1 UVU contract tag must be {tag!r}")
    if normalized.get("schema") != MH1_SCHEMA:
        raise ValueError(f"MH1 UVU contract schema must be {MH1_SCHEMA!r}")
    if normalized.get("version") != version:
        raise ValueError(f"MH1 UVU contract version must be {version}")
    if normalized.get("interaction_count") != 2:
        raise ValueError("MH1 UVU contracts require exactly two interactions")
    interactions = normalized.get("interactions")
    if not isinstance(interactions, list) or len(interactions) != 2:
        raise ValueError(
            "MH1 UVU contracts require exactly two interaction descriptors"
        )
    for index, descriptor in enumerate(interactions):
        if not isinstance(descriptor, dict) or descriptor.get("index") != index:
            raise ValueError("MH1 UVU interactions must use extraction order")
        supplied = descriptor.pop("generation_fingerprint", None)
        structure = descriptor.pop("structure_fingerprint", None)
        expected = _fingerprint(descriptor)
        try:
            expected_structure = _fingerprint(
                _interaction_structure_payload(
                    descriptor, include_density=include_density
                )
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"MH1 UVU interaction {index} descriptor is incomplete"
            ) from error
        if supplied != expected or structure != expected_structure:
            raise ValueError(f"MH1 UVU interaction {index} fingerprint is stale")
        descriptor["generation_fingerprint"] = expected
        descriptor["structure_fingerprint"] = expected_structure
    if normalized.get("layouts") != layouts:
        raise ValueError("MH1 UVU layouts do not match the generated ABI")
    if normalized.get("derivatives") != _DERIVATIVES:
        raise ValueError("MH1 UVU derivative capabilities are unsupported")
    if normalized.get("accumulator") != {
        "mode": "runtime_precision",
        "native_generated": ["float32"],
        "fallback": ["float32", "float64"],
    }:
        raise ValueError("MH1 UVU accumulator capabilities are unsupported")
    expected_semantic = _fingerprint(_semantic_payload(normalized))
    normalized["semantic_fingerprint"] = expected_semantic
    expected_generation = _fingerprint(normalized)
    try:
        expected_structure = _fingerprint(_contract_structure_payload(normalized))
    except (KeyError, TypeError) as error:
        raise ValueError("MH1 UVU contract structure is incomplete") from error
    if supplied_semantic != expected_semantic:
        raise ValueError("MH1 UVU semantic fingerprint is stale")
    if supplied_generation != expected_generation:
        raise ValueError("MH1 UVU generation fingerprint is stale")
    if supplied_structure != expected_structure:
        raise ValueError("MH1 UVU structure fingerprint is stale")
    normalized["generation_fingerprint"] = expected_generation
    normalized["structure_fingerprint"] = expected_structure
    return normalized


def normalize_execution_mh1_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a stored contract and return its canonical representation."""

    return _normalize_execution_mh1_contract_version(
        contract,
        tag=MH1_CONTRACT_TAG,
        version=MH1_SCHEMA_VERSION,
        include_density=True,
        layouts=_LAYOUTS,
    )


def _normalize_runtime_parameter_pack(pack: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(pack))
    supplied = normalized.pop("fingerprint", None)
    if normalized.get("tag") != MH1_NODE_RUNTIME_LAYOUT_TAG:
        raise ValueError("MH1 node runtime pack tag is unsupported")
    if normalized.get("scalar_type") != "float32":
        raise ValueError("MH1 node runtime packs require float32")
    if not isinstance(normalized.get("role"), str) or not normalized["role"]:
        raise ValueError("MH1 node runtime pack role is invalid")
    segments = normalized.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("MH1 node runtime pack segments are invalid")
    offset = 0
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("offset") != offset:
            raise ValueError("MH1 node runtime pack offsets are not contiguous")
        count = _require_int(segment.get("count"), "runtime segment count", 1)
        offset += count
    if normalized.get("parameter_count") != offset:
        raise ValueError("MH1 node runtime pack parameter count is inconsistent")
    expected = _fingerprint(normalized)
    if supplied != expected:
        raise ValueError("MH1 node runtime pack fingerprint is stale")
    normalized["fingerprint"] = expected
    return normalized


def _normalize_node_program(program: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(program))
    supplied_structure = normalized.pop("structure_fingerprint", None)
    if normalized.get("tag") != MH1_NODE_PROGRAM_TAG:
        raise ValueError("MH1 node-program tag is unsupported")
    if normalized.get("layer_count") != 2:
        raise ValueError("MH1 node generation requires exactly two layers")
    if normalized.get("feature_layout") != "ir_mul":
        raise ValueError("MH1 node generation requires persistent ir_mul state")
    if normalized.get("execution") != _NODE_EXECUTION:
        raise ValueError("MH1 node execution schedule is unsupported")
    if normalized.get("derivatives") != _NODE_DERIVATIVES:
        raise ValueError("MH1 node derivative capabilities are unsupported")
    layers = normalized.get("layers")
    if not isinstance(layers, list) or len(layers) != 2:
        raise ValueError("MH1 node-program layers are invalid")
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("index") != index:
            raise ValueError("MH1 node-program layers must use extraction order")
        derivatives = layer.get("derivatives")
        if not isinstance(derivatives, dict) or derivatives.get(
            "requires_tp_source_state_adjoint"
        ) is not (index > 0):
            raise ValueError("MH1 node TP source-adjoint ownership is invalid")
        state = layer.get("persistent_state")
        policy = state.get("node_state_policy") if isinstance(state, dict) else None
        expected_policy = {
            "default": "full-retention-v1",
            "supported": [
                "full-retention-v1",
                "recompute-v1",
                "reuse-adjoints-v1",
                "retain-interaction-v1",
            ],
            "retained_tensors": {
                "pre_gate": int(layer["linears"]["linear_res"]["dimensions"]["output"]),
                "interaction_output": int(
                    layer["linears"]["linear_2"]["dimensions"]["output"]
                ),
            },
        }
        if policy != expected_policy:
            raise ValueError("MH1 node-state retention policy is unsupported")
        packs = layer.get("runtime_parameter_packs")
        if not isinstance(packs, dict) or set(packs) != {
            "linear",
            "product",
            "readout",
        }:
            raise ValueError("MH1 node runtime parameter packs are incomplete")
        layer["runtime_parameter_packs"] = {
            name: _normalize_runtime_parameter_pack(pack)
            for name, pack in packs.items()
        }

    runtime_layout = normalized.get("runtime_layout")
    if not isinstance(runtime_layout, dict):
        raise ValueError("MH1 node runtime layout is missing")
    supplied_runtime = runtime_layout.pop("fingerprint", None)
    if runtime_layout.get("tag") != MH1_NODE_RUNTIME_LAYOUT_TAG:
        raise ValueError("MH1 node runtime layout tag is unsupported")
    if runtime_layout.get("scalar_type") != "float32":
        raise ValueError("MH1 node runtime layout requires float32")
    layout_layers = runtime_layout.get("layers")
    if not isinstance(layout_layers, list) or len(layout_layers) != 2:
        raise ValueError("MH1 node runtime layout layers are invalid")
    for index, layout_layer in enumerate(layout_layers):
        if not isinstance(layout_layer, dict) or layout_layer.get("index") != index:
            raise ValueError("MH1 node runtime layout order is invalid")
        packs = layout_layer.get("packs")
        if not isinstance(packs, dict) or set(packs) != {
            "linear",
            "product",
            "readout",
        }:
            raise ValueError("MH1 node runtime layout packs are incomplete")
        layout_layer["packs"] = {
            name: _normalize_runtime_parameter_pack(pack)
            for name, pack in packs.items()
        }
        if layout_layer["packs"] != layers[index]["runtime_parameter_packs"]:
            raise ValueError("MH1 node runtime layout disagrees with layer packs")
    expected_runtime = _fingerprint(runtime_layout)
    if supplied_runtime != expected_runtime:
        raise ValueError("MH1 node runtime layout fingerprint is stale")
    runtime_layout["fingerprint"] = expected_runtime

    expected_structure = _fingerprint(_node_program_structure_payload(normalized))
    if supplied_structure != expected_structure:
        raise ValueError("MH1 node-program structure fingerprint is stale")
    normalized["structure_fingerprint"] = expected_structure
    return normalized


def normalize_execution_mh1_v4_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the staged complete-node contract and its runtime pack layout."""

    normalized = deepcopy(dict(contract))
    runtime_layout_fingerprint = normalized.get("runtime_layout_fingerprint")
    node_program = normalized.get("node_program")
    if not isinstance(node_program, Mapping):
        raise ValueError("MH1 v4 contract is missing its node program")
    normalized["node_program"] = _normalize_node_program(node_program)
    expected_runtime = normalized["node_program"]["runtime_layout"]["fingerprint"]
    if runtime_layout_fingerprint != expected_runtime:
        raise ValueError("MH1 contract runtime layout fingerprint is stale")
    return _normalize_execution_mh1_contract_version(
        normalized,
        tag=MH1_V4_CONTRACT_TAG,
        version=MH1_V4_SCHEMA_VERSION,
        include_density=True,
        layouts=_LAYOUTS,
    )


def validate_execution_mh1_contract_for_model(
    model_data: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate model binding, upgrading exact version-1/2 contracts to version 3."""

    expected = make_execution_mh1_contract(model_data)
    if contract.get("tag") == _MH1_LEGACY_CONTRACT_TAG:
        normalized_legacy = _normalize_execution_mh1_contract_version(
            contract,
            tag=_MH1_LEGACY_CONTRACT_TAG,
            version=_MH1_LEGACY_SCHEMA_VERSION,
            include_density=False,
            layouts=_LAYOUTS_V2,
        )
        if normalized_legacy != _legacy_contract_projection(expected):
            raise ValueError(
                "the embedded Execution MH1_UVU contract does not match the model "
                "definition"
            )
        return expected

    if contract.get("tag") == _MH1_V2_CONTRACT_TAG:
        normalized_v2 = _normalize_execution_mh1_contract_version(
            contract,
            tag=_MH1_V2_CONTRACT_TAG,
            version=_MH1_V2_SCHEMA_VERSION,
            include_density=True,
            layouts=_LAYOUTS_V2,
        )
        if normalized_v2 != _v2_contract_projection(expected):
            raise ValueError(
                "the embedded Execution MH1_UVU contract does not match the model "
                "definition"
            )
        return expected

    normalized = normalize_execution_mh1_contract(contract)
    if normalized != expected:
        raise ValueError(
            "the embedded Execution MH1_UVU contract does not match the model "
            "definition"
        )
    return normalized


def validate_execution_mh1_v4_contract_for_model(
    model_data: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate v4 or upgrade an exact model-bound legacy identity to current v4."""

    expected = make_execution_mh1_v4_contract(model_data)
    if contract.get("tag") in {
        _MH1_LEGACY_CONTRACT_TAG,
        _MH1_V2_CONTRACT_TAG,
        _MH1_V3_CONTRACT_TAG,
    }:
        normalized_v3 = validate_execution_mh1_contract_for_model(model_data, contract)
        if normalized_v3 != _v3_contract_projection(expected):
            raise ValueError(
                "the embedded Execution MH1_UVU contract does not match the model "
                "definition"
            )
        return expected
    if contract == _pre_retention_v4_contract_projection(expected):
        return expected
    normalized = normalize_execution_mh1_v4_contract(contract)
    if normalized != expected:
        raise ValueError(
            "the embedded Execution MH1_UVU contract does not match the model "
            "definition"
        )
    return normalized


def maybe_make_execution_mh1_contract(
    model_data: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a generated contract when the existing MH-1 model is compatible."""

    try:
        return make_execution_mh1_contract(model_data)
    except (KeyError, TypeError, ValueError):
        return None


def maybe_make_execution_mh1_v4_contract(
    model_data: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the staged node contract for the existing supported MH-1 family."""

    try:
        return make_execution_mh1_v4_contract(model_data)
    except (KeyError, TypeError, ValueError):
        return None
