"""PolarMACE family execution-contract construction and validation.

The ordinary MH-1 module owns the reusable backbone descriptors.  This module
owns the Polar delta descriptors, unrolled schedule, parameter packs, and
fingerprints layered on top of that neutral backbone contract.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from .mace_nonlinear_execution_contract import (
    BACKBONE_MODEL_TYPES,
    _affine_mlp_layer_descriptors,
    _block_identity,
    _fingerprint,
    _gate_descriptor,
    _gate_structure_payload,
    _irrep_blocks,
    _irrep_dimension,
    _linear_descriptor,
    _linear_structure_payload,
    _normalize_runtime_parameter_pack,
    _require_finite_number,
    _require_int,
    _require_shape,
    _runtime_parameter_pack,
    _tensor_shape,
    _tensor_values,
    make_execution_mh1_v4_contract,
    validate_execution_mh1_v4_contract_for_model,
)

POLAR_SCHEMA = "symmetrix.execution.polar_backbone"
# Version 2 adds the unrolled global/local execution schedule as contract data.
POLAR_SCHEMA_VERSION = 2
POLAR_CONTRACT_TAG = f"{POLAR_SCHEMA}/{POLAR_SCHEMA_VERSION}"
POLAR_CONTRACT_KEY = "POLAR_BACKBONE"

_POLAR_SPIN_SOURCE_CLASS = "EnvironmentDependentSpinSourceBlock"
_POLAR_MIXER_CLASS = "MultiLayerFeatureMixer"
_POLAR_UPDATE_CLASS = "AgnosticEmbeddedOneBodyVariableUpdate"
_POLAR_POTENTIAL_EMBEDDING_CLASS = "AgnosticChargeBiasedLinearPotentialEmbedding"
_POLAR_UPDATE_READOUT_CLASS = "GeneralNonLinearBiasReadoutBlock"


def _require_keep_last_layer_irreps(model_data: Mapping[str, Any]) -> None:
    polar = model_data.get("polar")
    if not isinstance(polar, Mapping):
        raise ValueError("Polar backbone contracts require polar model data")
    if "keep_last_layer_irreps" not in polar:
        raise ValueError("Polar backbone contracts require keep-last-layer readouts")
    keep_last = polar.get("keep_last_layer_irreps")
    if not isinstance(keep_last, bool):
        raise ValueError("polar.keep_last_layer_irreps must be boolean")
    if not keep_last:
        raise ValueError("Polar backbone contracts require keep-last-layer readouts")


def make_execution_polar_mh1_v4_contract(
    model_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the retained-feature MH-1 contract used by the Polar family."""

    _require_keep_last_layer_irreps(model_data)
    return make_execution_mh1_v4_contract(
        model_data,
        retain_last_product_irreps=True,
    )


def validate_execution_polar_mh1_v4_contract_for_model(
    model_data: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the Polar family's retained-feature MH-1 contract."""

    _require_keep_last_layer_irreps(model_data)
    return validate_execution_mh1_v4_contract_for_model(
        model_data,
        contract,
        retain_last_product_irreps=True,
    )


def _require_polar_spherical_harmonic_pattern(
    blocks: list[dict[str, int | str]],
    name: str,
    max_l: int,
    multiplicity: int,
    *,
    repeats: int = 1,
) -> None:
    """Validate ``repeats`` consecutive copies of ``k * spherical_harmonics(l)``.

    e3nn ``Irreps`` scalar multiplication repeats the block sequence, so the
    upstream charge and potential signatures carry one copy per spin channel
    (channel-major, l-minor inside each copy) instead of doubled multiplicities.
    """

    pattern_length = max_l + 1
    if len(blocks) != repeats * pattern_length:
        raise ValueError(f"{name} must span l = 0..{max_l} per spin channel")
    for repeat in range(repeats):
        for l_value in range(pattern_length):
            block = blocks[repeat * pattern_length + l_value]
            if (
                int(block["multiplicity"]) != multiplicity
                or int(block["l"]) != l_value
                or block["parity"] != ("e" if l_value % 2 == 0 else "o")
            ):
                raise ValueError(f"{name} is outside the PolarMACE support matrix")


def _affine_mlp_runtime_parameters(layers: list[dict[str, Any]]) -> dict[str, Any]:
    segments = []
    offset = 0
    for layer in layers:
        if layer["type"] == "linear":
            weight_count = int(layer["output_dimension"]) * int(
                layer["input_dimension"]
            )
            bias_count = int(layer["output_dimension"])
        elif layer["type"] == "layer_norm":
            weight_count = bias_count = int(layer["dimension"])
        else:
            continue
        segments.append(
            {
                "name": f"layers[{layer['index']}].weight",
                "offset": offset,
                "count": weight_count,
            }
        )
        offset += weight_count
        segments.append(
            {
                "name": f"layers[{layer['index']}].bias",
                "offset": offset,
                "count": bias_count,
            }
        )
        offset += bias_count
    return {
        "scalar_type": "runtime_precision",
        "order": ["layer_weights_row_major", "layer_biases"],
        "segments": segments,
        "parameter_count": offset,
    }


def _plain_affine_mlp_descriptor(
    definition: Any,
    name: str,
    *,
    input_dimension: int,
    output_dimension: int,
) -> dict[str, Any]:
    layers = _affine_mlp_layer_descriptors(definition, name)
    if layers[0]["type"] != "linear" or layers[-1]["type"] != "linear":
        raise ValueError(f"{name} must begin and end with linear layers")
    if layers[0]["input_dimension"] != input_dimension:
        raise ValueError(f"{name} input dimension is inconsistent")
    if layers[-1]["output_dimension"] != output_dimension:
        raise ValueError(f"{name} output dimension is inconsistent")
    return {
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "layers": layers,
        "runtime_parameters": _affine_mlp_runtime_parameters(layers),
    }


def _sparse_uvu_tensor_product_descriptor(definition: Any, name: str) -> dict[str, Any]:
    """Describe a node-local sparse uvu tensor product, values runtime-only."""

    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be a serialized uvu tensor product")
    irreps = {}
    blocks = {}
    for key, field in (
        ("input_1", "irreps_in1"),
        ("input_2", "irreps_in2"),
        ("output", "irreps_out"),
    ):
        canonical, parsed = _irrep_blocks(definition.get(field), f"{name}.{field}")
        irreps[field] = canonical
        blocks[key] = parsed
    dimensions = {key: _irrep_dimension(value) for key, value in blocks.items()}
    output_mask_shape, output_mask_values = _tensor_values(definition, "output_mask", 1)
    if output_mask_shape != [dimensions["output"]]:
        raise ValueError(f"{name} output mask shape is inconsistent")

    instructions = definition.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"{name} must contain tensor-product paths")
    paths = []
    weight_offset = 0
    for path_index, instruction in enumerate(instructions):
        if not isinstance(instruction, Mapping):
            raise ValueError(f"{name}.instructions[{path_index}] is invalid")
        selected_blocks = {}
        for field, key in (
            ("i_in1", "input_1"),
            ("i_in2", "input_2"),
            ("i_out", "output"),
        ):
            block_index = _require_int(
                instruction.get(field),
                f"{name}.instructions[{path_index}].{field}",
            )
            if block_index >= len(blocks[key]):
                raise ValueError(f"{name} path irrep index is invalid")
            selected_blocks[key] = blocks[key][block_index]
        source = selected_blocks["input_1"]
        edge = selected_blocks["input_2"]
        output = selected_blocks["output"]
        path_shape = _require_shape(
            instruction.get("path_shape"),
            f"{name}.instructions[{path_index}].path_shape",
            2,
        )
        expected_shape = [int(source["multiplicity"]), int(edge["multiplicity"])]
        if (
            instruction.get("connection_mode") != "uvu"
            or instruction.get("has_weight") is not True
            or int(output["multiplicity"]) != int(source["multiplicity"])
            or path_shape != expected_shape
        ):
            raise ValueError(f"{name} is not a tied-channel UVU tensor product")
        source_dim = int(source["components"])
        edge_dim = int(edge["components"])
        output_dim = int(output["components"])
        if output_dim == 1 and source_dim == edge_dim:
            mode = "contraction"
        elif edge_dim == 1 and output_dim == source_dim:
            mode = "modulation"
        else:
            raise ValueError(
                f"{name} supports only (l x l -> 0) and (l x 0 -> l) sparse paths"
            )

        path_weight = instruction.get("path_weight")
        if isinstance(path_weight, bool) or not isinstance(path_weight, (int, float)):
            raise ValueError(f"{name} path_weight must be numeric")
        path_weight = float(path_weight)
        if not math.isfinite(path_weight):
            raise ValueError(f"{name} path_weight must be finite")
        wigner = instruction.get("wigner_3j")
        if not isinstance(wigner, Mapping):
            raise ValueError(f"{name} Wigner tensor is missing")
        wigner_shape = _require_shape(
            wigner.get("shape"),
            f"{name}.instructions[{path_index}].wigner_3j.shape",
            3,
        )
        expected_wigner = [source_dim, edge_dim, output_dim]
        if wigner_shape != expected_wigner:
            raise ValueError(f"{name} Wigner tensor shape is inconsistent")
        values = wigner.get("values")
        if not isinstance(values, list) or len(values) != math.prod(wigner_shape):
            raise ValueError(f"{name} Wigner tensor values are inconsistent")
        sparse_terms = []
        for flat, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} Wigner coefficients must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} Wigner coefficients must be finite")
            if value == 0.0:
                continue
            b_dimension, c_dimension = wigner_shape[1:]
            a = flat // (b_dimension * c_dimension)
            remainder = flat % (b_dimension * c_dimension)
            b = remainder // c_dimension
            c = remainder % c_dimension
            sparse_terms.append({"a": a, "b": b, "c": c, "coefficient": value})
        if not sparse_terms:
            raise ValueError(f"{name} Wigner tensor must not be zero")

        paths.append(
            {
                "index": path_index,
                "input_1": deepcopy(source),
                "input_2": deepcopy(edge),
                "output": deepcopy(output),
                "connection_mode": "uvu",
                "has_weight": True,
                "mode": mode,
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

    weight_shape, _ = _tensor_values(definition, "weight", 1)
    if weight_shape != [weight_offset]:
        raise ValueError(f"{name}.weight shape is inconsistent")
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
        _require_finite_number(value, f"{name}.output_mask[{component}]")
        for component, value in enumerate(output_mask_values)
    ]
    if actual_output_mask != expected_output_mask:
        raise ValueError(f"{name} output mask does not match tensor-product paths")
    return {
        "irreps": irreps,
        "blocks": deepcopy(blocks),
        "dimensions": dimensions,
        "paths": paths,
        "runtime_parameters": {
            "scalar_type": "runtime_precision",
            "order": ["path_weights_row_major"],
            "segments": [{"name": "path_weights", "offset": 0, "count": weight_offset}],
            "parameter_count": weight_offset,
        },
    }


def _sparse_uvu_structure_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paths": [
            {
                "input_1_l": path["input_1"]["l"],
                "input_2_l": path["input_2"]["l"],
                "output_l": path["output"]["l"],
                "mode": path["mode"],
                "path_weight": path["path_weight"],
                "sparse_wigner": path["sparse_wigner"],
            }
            for path in descriptor["paths"]
        ],
        "weight_count": descriptor["runtime_parameters"]["parameter_count"],
    }


def _polar_gated_readout_descriptor(
    definition: Any,
    name: str,
    *,
    node_feats_irreps: str,
    charges_blocks: list[dict[str, int | str]],
) -> dict[str, Any]:
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be an object")
    if definition.get("class") != _POLAR_UPDATE_READOUT_CLASS:
        raise ValueError(f"{name} is not a general nonlinear bias readout block")
    gate = _gate_descriptor(definition.get("gate"), f"{name}.gate")
    linear_1 = _linear_descriptor(definition.get("linear_1"), f"{name}.linear_1")
    linear_mid = _linear_descriptor(definition.get("linear_mid"), f"{name}.linear_mid")
    linear_2 = _linear_descriptor(definition.get("linear_2"), f"{name}.linear_2")
    if linear_1["irreps"]["input"] != node_feats_irreps:
        raise ValueError(f"{name}.linear_1 input is inconsistent")
    if (
        linear_1["irreps"]["output"] != gate["irreps"]["irreps_in"]
        or linear_mid["irreps"]["output"] != gate["irreps"]["irreps_in"]
    ):
        raise ValueError(f"{name} gate inputs are inconsistent")
    if (
        linear_mid["irreps"]["input"] != gate["irreps"]["irreps_out"]
        or linear_2["irreps"]["input"] != gate["irreps"]["irreps_out"]
    ):
        raise ValueError(f"{name} hidden irreps are inconsistent")
    expected_output = [_block_identity(block) for block in charges_blocks] + [
        (2, 0, "e")
    ]
    if [
        _block_identity(block) for block in linear_2["blocks"]["output"]
    ] != expected_output:
        raise ValueError(
            f"{name} must produce the charge spin channels followed by " "fukui sources"
        )
    return {
        "linear_1": linear_1,
        "gate": gate,
        "linear_mid": linear_mid,
        "linear_2": linear_2,
    }


def _polar_gated_readout_structure_payload(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "linear_1": _linear_structure_payload(descriptor["linear_1"]),
        "gate": _gate_structure_payload(descriptor["gate"]),
        "linear_mid": _linear_structure_payload(descriptor["linear_mid"]),
        "linear_2": _linear_structure_payload(descriptor["linear_2"]),
    }


def _polar_field_update_descriptor(
    index: int,
    definition: Any,
    *,
    node_feats_irreps: str,
    node_attrs_irreps: str,
    potential_irreps: str,
    charges_irreps: str,
    charges_blocks: list[dict[str, int | str]],
) -> dict[str, Any]:
    name = f"polar.field_update_blocks[{index}]"
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be an object")
    if definition.get("class") != _POLAR_UPDATE_CLASS:
        raise ValueError(f"{name} is not an agnostic embedded one-body variable update")
    embedding_definition = definition.get("potential_embedding")
    if not isinstance(embedding_definition, Mapping):
        raise ValueError(f"{name}.potential_embedding must be an object")
    if embedding_definition.get("class") != _POLAR_POTENTIAL_EMBEDDING_CLASS:
        raise ValueError(
            f"{name}.potential_embedding is not a charge-biased linear "
            "potential embedding"
        )
    potential_linear = _linear_descriptor(
        embedding_definition.get("potential_linear"),
        f"{name}.potential_embedding.potential_linear",
    )
    node_feats_linear = _linear_descriptor(
        embedding_definition.get("node_feats_linear"),
        f"{name}.potential_embedding.node_feats_linear",
    )
    charge_embedding = _linear_descriptor(
        embedding_definition.get("charge_embedding"),
        f"{name}.potential_embedding.charge_embedding",
    )
    if (
        potential_linear["irreps"]["input"] != potential_irreps
        or potential_linear["irreps"]["output"] != node_feats_irreps
    ):
        raise ValueError(f"{name}.potential_linear irreps are inconsistent")
    if (
        node_feats_linear["irreps"]["input"] != node_feats_irreps
        or node_feats_linear["irreps"]["output"] != node_feats_irreps
    ):
        raise ValueError(f"{name}.node_feats_linear irreps are inconsistent")
    if (
        charge_embedding["irreps"]["input"] != charges_irreps
        or charge_embedding["irreps"]["output"] != node_feats_irreps
    ):
        raise ValueError(
            f"{name}.charge_embedding must embed the Q/S charge spin channels"
        )
    source_embedding = _linear_descriptor(
        definition.get("source_embedding"), f"{name}.source_embedding"
    )
    if source_embedding["irreps"]["input"] != node_attrs_irreps:
        raise ValueError(f"{name}.source_embedding input is inconsistent")
    invar_irreps = source_embedding["irreps"]["output"]
    invar_blocks = source_embedding["blocks"]["output"]
    if (
        len(invar_blocks) != 1
        or int(invar_blocks[0]["l"]) != 0
        or invar_blocks[0]["parity"] != "e"
    ):
        raise ValueError(f"{name}.source_embedding must produce invariant channels")
    invar_dimension = _irrep_dimension(invar_blocks)
    dot_products = _sparse_uvu_tensor_product_descriptor(
        definition.get("dot_products"), f"{name}.dot_products"
    )
    if (
        dot_products["irreps"]["irreps_in1"] != node_feats_irreps
        or dot_products["irreps"]["irreps_in2"] != node_feats_irreps
        or dot_products["irreps"]["irreps_out"] != invar_irreps
    ):
        raise ValueError(f"{name}.dot_products irreps are inconsistent")
    nonlinearity = _plain_affine_mlp_descriptor(
        definition.get("nonlinearity"),
        f"{name}.nonlinearity",
        input_dimension=2 * invar_dimension,
        output_dimension=invar_dimension,
    )
    tp_out = _sparse_uvu_tensor_product_descriptor(
        definition.get("tp_out"), f"{name}.tp_out"
    )
    if (
        tp_out["irreps"]["irreps_in1"] != node_feats_irreps
        or tp_out["irreps"]["irreps_in2"] != invar_irreps
        or tp_out["irreps"]["irreps_out"] != node_feats_irreps
    ):
        raise ValueError(f"{name}.tp_out irreps are inconsistent")
    readout = _polar_gated_readout_descriptor(
        definition.get("readout"),
        f"{name}.readout",
        node_feats_irreps=node_feats_irreps,
        charges_blocks=charges_blocks,
    )
    field_norm_factor = _require_finite_number(
        definition.get("field_norm_factor"), f"{name}.field_norm_factor"
    )
    return {
        "index": index,
        "edge_set": "empty",
        "potential_embedding": {
            "potential_linear": potential_linear,
            "node_feats_linear": node_feats_linear,
            "charge_embedding": charge_embedding,
        },
        "source_embedding": source_embedding,
        "dot_products": dot_products,
        "nonlinearity": nonlinearity,
        "tp_out": tp_out,
        "readout": readout,
        "field_norm_factor": field_norm_factor,
    }


def _polar_update_structure_payload(update: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "potential_embedding": {
            name: _linear_structure_payload(descriptor)
            for name, descriptor in update["potential_embedding"].items()
        },
        "source_embedding": _linear_structure_payload(update["source_embedding"]),
        "dot_products": _sparse_uvu_structure_payload(update["dot_products"]),
        "nonlinearity": [layer["type"] for layer in update["nonlinearity"]["layers"]],
        "tp_out": _sparse_uvu_structure_payload(update["tp_out"]),
        "readout": _polar_gated_readout_structure_payload(update["readout"]),
    }


def _polar_spin_source_head_descriptor(
    index: int,
    definition: Any,
    *,
    input_irreps: str,
    charges_irreps: str,
) -> dict[str, Any]:
    name = f"polar.spin_source_heads[{index}]"
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be an object")
    if definition.get("class") != _POLAR_SPIN_SOURCE_CLASS:
        raise ValueError(f"{name} is not a spin-source multipole head")
    zero_charges = definition.get("zero_charges")
    if not isinstance(zero_charges, bool):
        raise ValueError(f"{name}.zero_charges must be boolean")
    linear = _linear_descriptor(definition.get("linear"), f"{name}.linear")
    if linear["irreps"]["input"] != input_irreps:
        raise ValueError(f"{name}.linear input is inconsistent")
    if linear["irreps"]["output"] != charges_irreps:
        raise ValueError(f"{name} must produce the spin-resolved multipoles")
    return {"index": index, "zero_charges": zero_charges, "linear": linear}


def _polar_source_map_descriptor(
    definition: Any,
    *,
    node_feats_irreps: str,
) -> dict[str, Any]:
    """Describe the pre-loop Fukui source map.

    The block is the oracle's ``NonLinearBiasReadoutBlock``: a dense linear into
    its hidden irreps, a plain normalized activation, a second dense linear, the
    activation again, and the output linear that emits the two source channels.
    Only its linear stack is model data; the activations are the oracle's own
    named pair with their normalization constants.
    """

    name = "polar.fukui_source_map"
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be an object")
    linear_1 = _linear_descriptor(definition.get("linear_1"), f"{name}.linear_1")
    linear_mid = _linear_descriptor(definition.get("linear_mid"), f"{name}.linear_mid")
    linear_2 = _linear_descriptor(definition.get("linear_2"), f"{name}.linear_2")
    if linear_1["irreps"]["input"] != node_feats_irreps:
        raise ValueError(f"{name}.linear_1 input is inconsistent")
    hidden_irreps = linear_1["irreps"]["output"]
    if (
        linear_mid["irreps"]["input"] != hidden_irreps
        or linear_mid["irreps"]["output"] != hidden_irreps
        or linear_2["irreps"]["input"] != hidden_irreps
    ):
        raise ValueError(f"{name} hidden irreps are inconsistent")
    hidden_blocks = linear_1["blocks"]["output"]
    if any(int(block["l"]) != 0 for block in hidden_blocks):
        raise ValueError(
            f"{name} hidden irreps must be scalars (the oracle activates them "
            "elementwise)"
        )
    output_blocks = linear_2["blocks"]["output"]
    if [_block_identity(block) for block in output_blocks] != [(2, 0, "e")]:
        raise ValueError(f"{name} must emit the two Fukui source channels")
    activation = definition.get("activation")
    if activation not in (None, "silu", "sigmoid"):
        raise ValueError(f"{name}.activation is not in the activation vocabulary")
    constants = definition.get("activation_constants")
    if not isinstance(constants, list) or len(constants) != 1:
        raise ValueError(f"{name}.activation_constants must carry one constant")
    constant = _require_finite_number(constants[0], f"{name}.activation_constants[0]")
    return {
        "linear_1": linear_1,
        "linear_mid": linear_mid,
        "linear_2": linear_2,
        "activation": activation or "silu",
        "activation_constant": constant,
    }


def _polar_source_map_structure_payload(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "linear_1": _linear_structure_payload(descriptor["linear_1"]),
        "linear_mid": _linear_structure_payload(descriptor["linear_mid"]),
        "linear_2": _linear_structure_payload(descriptor["linear_2"]),
        "activation": descriptor["activation"],
    }


def _polar_local_electron_descriptor(
    definition: Any,
    *,
    node_feats_irreps: str,
    potential_irreps: str,
    charges_irreps: str,
) -> dict[str, Any]:
    """Describe the post-loop local-electron-energy readout.

    The block is the oracle's ``OneBodyMLPFieldReadout``: two biased dense
    equivariant embeddings (of the summed charge density and of the potential
    features), two node-local tied-channel tensor products against the node
    features, one affine MLP over the concatenated invariants, and the per-node
    energy it emits.
    """

    name = "polar.local_electron_energy"
    if not isinstance(definition, Mapping):
        raise ValueError(f"{name} must be an object")
    linear_up_q = _linear_descriptor(
        definition.get("linear_up_q"), f"{name}.linear_up_q"
    )
    linear_up_v = _linear_descriptor(
        definition.get("linear_up_v"), f"{name}.linear_up_v"
    )
    if (
        linear_up_q["irreps"]["input"] != charges_irreps
        or linear_up_q["irreps"]["output"] != node_feats_irreps
    ):
        raise ValueError(f"{name}.linear_up_q irreps are inconsistent")
    if (
        linear_up_v["irreps"]["input"] != potential_irreps
        or linear_up_v["irreps"]["output"] != node_feats_irreps
    ):
        raise ValueError(f"{name}.linear_up_v irreps are inconsistent")
    dot_products_q = _sparse_uvu_tensor_product_descriptor(
        definition.get("dot_products_q"), f"{name}.dot_products_q"
    )
    dot_products_v = _sparse_uvu_tensor_product_descriptor(
        definition.get("dot_products_v"), f"{name}.dot_products_v"
    )
    for label, descriptor in (
        ("dot_products_q", dot_products_q),
        ("dot_products_v", dot_products_v),
    ):
        if (
            descriptor["irreps"]["irreps_in1"] != node_feats_irreps
            or descriptor["irreps"]["irreps_in2"] != node_feats_irreps
        ):
            raise ValueError(f"{name}.{label} irreps are inconsistent")
    invariants = dot_products_q["irreps"]["irreps_out"]
    if dot_products_v["irreps"]["irreps_out"] != invariants:
        raise ValueError(f"{name} tensor products must emit one invariant width")
    mlp = _plain_affine_mlp_descriptor(
        definition.get("mlp"),
        f"{name}.mlp",
        input_dimension=2 * _irrep_dimension(dot_products_q["blocks"]["output"]),
        output_dimension=1,
    )
    return {
        "linear_up_q": linear_up_q,
        "linear_up_v": linear_up_v,
        "dot_products_q": dot_products_q,
        "dot_products_v": dot_products_v,
        "mlp": mlp,
    }


def _polar_local_electron_structure_payload(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "linear_up_q": _linear_structure_payload(descriptor["linear_up_q"]),
        "linear_up_v": _linear_structure_payload(descriptor["linear_up_v"]),
        "dot_products_q": _sparse_uvu_structure_payload(descriptor["dot_products_q"]),
        "dot_products_v": _sparse_uvu_structure_payload(descriptor["dot_products_v"]),
        "mlp": [layer["type"] for layer in descriptor["mlp"]["layers"]],
    }


def _polar_backbone_descriptors(model_data: Mapping[str, Any]) -> dict[str, Any]:
    polar = model_data["polar"]
    interactions = model_data.get("interactions")
    products = model_data.get("products")
    if not isinstance(interactions, list) or not interactions:
        raise ValueError("Polar backbone contracts require interactions")
    if not isinstance(products, list) or len(products) != len(interactions):
        raise ValueError("Polar backbone contracts require one product per interaction")
    interaction_count = len(interactions)

    charges_irreps, charges_blocks = _irrep_blocks(
        polar.get("charges_irreps"), "polar.charges_irreps"
    )
    multipoles_max_l = _require_int(
        polar.get("atomic_multipoles_max_l"), "polar.atomic_multipoles_max_l"
    )
    _require_polar_spherical_harmonic_pattern(
        charges_blocks, "polar.charges_irreps", multipoles_max_l, 1, repeats=2
    )
    field_max_l = _require_int(
        polar.get("field_feature_max_l"), "polar.field_feature_max_l"
    )
    raw_widths = polar.get("field_feature_widths")
    if not isinstance(raw_widths, list) or not raw_widths:
        raise ValueError("polar.field_feature_widths must be a non-empty list")
    widths = [
        _require_finite_number(value, f"polar.field_feature_widths[{index}]")
        for index, value in enumerate(raw_widths)
    ]
    if any(width <= 0.0 for width in widths):
        raise ValueError("polar.field_feature_widths must be positive")
    field_irreps, field_blocks = _irrep_blocks(
        polar.get("field_irreps"), "polar.field_irreps"
    )
    _require_polar_spherical_harmonic_pattern(
        field_blocks, "polar.field_irreps", field_max_l, len(widths)
    )
    potential_irreps, potential_blocks = _irrep_blocks(
        polar.get("potential_irreps"), "polar.potential_irreps"
    )
    _require_polar_spherical_harmonic_pattern(
        potential_blocks, "polar.potential_irreps", field_max_l, len(widths), repeats=2
    )
    norm_shape = _tensor_shape(polar, "field_feature_norms", 1)
    field_dimension = _irrep_dimension(field_blocks)
    if norm_shape != [field_dimension]:
        raise ValueError("polar.field_feature_norms shape is inconsistent")
    smearing_width = _require_finite_number(
        polar.get("atomic_multipoles_smearing_width"),
        "polar.atomic_multipoles_smearing_width",
    )
    if smearing_width <= 0.0:
        raise ValueError("polar.atomic_multipoles_smearing_width must be positive")
    kspace_cutoff = _require_finite_number(
        polar.get("kspace_cutoff"), "polar.kspace_cutoff"
    )
    if kspace_cutoff <= 0.0:
        raise ValueError("polar.kspace_cutoff must be positive")
    field_norm_factor = _require_finite_number(
        polar.get("field_norm_factor"), "polar.field_norm_factor"
    )

    product_targets = []
    for index, product in enumerate(products):
        if not isinstance(product, Mapping):
            raise ValueError(f"products[{index}] must be an object")
        declared, _ = _irrep_blocks(
            product.get("target_irreps"), f"products[{index}].target_irreps"
        )
        product_targets.append(declared)
    node_feats_irreps = product_targets[0]
    if any(target != node_feats_irreps for target in product_targets):
        raise ValueError(
            "Polar backbone contracts require uniform keep-last-layer product targets"
        )

    heads = polar.get("spin_source_heads")
    if not isinstance(heads, list) or len(heads) != interaction_count:
        raise ValueError("polar.spin_source_heads must match the interaction count")
    head_descriptors = [
        _polar_spin_source_head_descriptor(
            index,
            head,
            input_irreps=product_targets[index],
            charges_irreps=charges_irreps,
        )
        for index, head in enumerate(heads)
    ]

    mixer = polar.get("feature_mixer")
    if not isinstance(mixer, Mapping):
        raise ValueError("polar.feature_mixer must be an object")
    if mixer.get("class") != _POLAR_MIXER_CLASS:
        raise ValueError("polar.feature_mixer is not a multi-layer feature mixer")
    mixer_linears = mixer.get("linears")
    if not isinstance(mixer_linears, list) or len(mixer_linears) != interaction_count:
        raise ValueError("polar.feature_mixer.linears must match the interaction count")
    mixer_descriptors = []
    for index, linear in enumerate(mixer_linears):
        descriptor = _linear_descriptor(linear, f"polar.feature_mixer.linears[{index}]")
        if (
            descriptor["irreps"]["input"] != node_feats_irreps
            or descriptor["irreps"]["output"] != node_feats_irreps
        ):
            raise ValueError(
                f"polar.feature_mixer.linears[{index}] irreps are inconsistent"
            )
        mixer_descriptors.append(descriptor)

    recursion_steps = _require_int(
        polar.get("num_recursion_steps"), "polar.num_recursion_steps", 1
    )
    updates = polar.get("field_update_blocks")
    if not isinstance(updates, list) or len(updates) != recursion_steps:
        raise ValueError("polar.field_update_blocks must match num_recursion_steps")
    num_elements = _require_int(model_data.get("num_elements"), "num_elements", 1)
    node_attrs_irreps, _ = _irrep_blocks(f"{num_elements}x0e", "node_attrs_irreps")
    update_descriptors = [
        _polar_field_update_descriptor(
            index,
            update,
            node_feats_irreps=node_feats_irreps,
            node_attrs_irreps=node_attrs_irreps,
            potential_irreps=potential_irreps,
            charges_irreps=charges_irreps,
            charges_blocks=charges_blocks,
        )
        for index, update in enumerate(updates)
    ]
    update_structures = [
        _polar_update_structure_payload(update) for update in update_descriptors
    ]
    if any(structure != update_structures[0] for structure in update_structures[1:]):
        raise ValueError("polar.field_update_blocks must share one structure")

    # The chain's two ends: the pre-loop Fukui source map and the post-loop
    # local-electron readout.  Both are model data the kernel parse consumes,
    # so the contract describes them exactly as it describes the update chain.
    source_map = _polar_source_map_descriptor(
        polar.get("fukui_source_map"), node_feats_irreps=node_feats_irreps
    )
    local_electron = _polar_local_electron_descriptor(
        polar.get("local_electron_energy"),
        node_feats_irreps=node_feats_irreps,
        potential_irreps=potential_irreps,
        charges_irreps=charges_irreps,
    )
    flags = {
        "add_local_electron_energy": bool(
            polar.get("add_local_electron_energy", False)
        ),
        "include_electrostatic_self_interaction": bool(
            polar.get("include_electrostatic_self_interaction", False)
        ),
        "quadrupole_feature_corrections": bool(
            polar.get("quadrupole_feature_corrections", False)
        ),
        "field_si": bool(polar.get("field_si", False)),
    }

    return {
        "interaction_count": interaction_count,
        "recursion_steps": recursion_steps,
        "flags": flags,
        "fukui_source_map": source_map,
        "local_electron_energy": local_electron,
        "charges": {
            "irreps": charges_irreps,
            "blocks": deepcopy(charges_blocks),
            "dimension": _irrep_dimension(charges_blocks),
            "max_l": multipoles_max_l,
            "smearing_width": smearing_width,
        },
        "field": {
            "irreps": field_irreps,
            "blocks": deepcopy(field_blocks),
            "dimension": field_dimension,
            "max_l": field_max_l,
            "smearing_widths": widths,
            "feature_norm_count": norm_shape[0],
            "kspace_cutoff": kspace_cutoff,
        },
        "potential": {
            "irreps": potential_irreps,
            "blocks": deepcopy(potential_blocks),
            "dimension": _irrep_dimension(potential_blocks),
        },
        "field_norm_factor": field_norm_factor,
        "node_feats_irreps": node_feats_irreps,
        "spin_source_heads": head_descriptors,
        "feature_mixer": {"linears": mixer_descriptors},
        "field_update_blocks": update_descriptors,
    }


def _polar_runtime_parameter_packs(descriptors: Mapping[str, Any]) -> dict[str, Any]:
    head_entries = [
        (f"layer_{head['index']}.linear", "e3_linear", head["linear"])
        for head in descriptors["spin_source_heads"]
    ]
    mixer_entries = [
        (f"layer_{index}.linear", "e3_linear", linear)
        for index, linear in enumerate(descriptors["feature_mixer"]["linears"])
    ]
    update_packs = []
    for update in descriptors["field_update_blocks"]:
        index = update["index"]
        embedding = update["potential_embedding"]
        readout = update["readout"]
        update_packs.append(
            {
                "potential_embedding": _runtime_parameter_pack(
                    f"polar.update_{index}.potential_embedding",
                    [
                        (
                            "potential_linear",
                            "e3_linear",
                            embedding["potential_linear"],
                        ),
                        (
                            "node_feats_linear",
                            "e3_linear",
                            embedding["node_feats_linear"],
                        ),
                        (
                            "charge_embedding",
                            "e3_linear",
                            embedding["charge_embedding"],
                        ),
                    ],
                ),
                "update_chain": _runtime_parameter_pack(
                    f"polar.update_{index}.update_chain",
                    [
                        ("source_embedding", "e3_linear", update["source_embedding"]),
                        (
                            "dot_products_weights",
                            "sparse_uvu_weights",
                            int(
                                update["dot_products"]["runtime_parameters"][
                                    "parameter_count"
                                ]
                            ),
                        ),
                        ("nonlinearity", "affine_mlp", update["nonlinearity"]),
                        (
                            "tp_out_weights",
                            "sparse_uvu_weights",
                            int(
                                update["tp_out"]["runtime_parameters"][
                                    "parameter_count"
                                ]
                            ),
                        ),
                    ],
                ),
                "readout": _runtime_parameter_pack(
                    f"polar.update_{index}.readout",
                    [
                        ("linear_1", "e3_linear", readout["linear_1"]),
                        ("linear_mid", "e3_linear", readout["linear_mid"]),
                        ("linear_2", "e3_linear", readout["linear_2"]),
                    ],
                ),
            }
        )
    source_map = descriptors["fukui_source_map"]
    local_electron = descriptors["local_electron_energy"]
    return {
        "spin_source_heads": _runtime_parameter_pack(
            "polar.spin_source_heads", head_entries
        ),
        "feature_mixer": _runtime_parameter_pack("polar.feature_mixer", mixer_entries),
        "field_update_blocks": update_packs,
        "fukui_source_map": _runtime_parameter_pack(
            "polar.fukui_source_map",
            [
                ("linear_1", "e3_linear", source_map["linear_1"]),
                ("linear_mid", "e3_linear", source_map["linear_mid"]),
                ("linear_2", "e3_linear", source_map["linear_2"]),
            ],
        ),
        "local_electron_energy": _runtime_parameter_pack(
            "polar.local_electron_energy",
            [
                ("linear_up_q", "e3_linear", local_electron["linear_up_q"]),
                ("linear_up_v", "e3_linear", local_electron["linear_up_v"]),
                (
                    "dot_products_q_weights",
                    "sparse_uvu_weights",
                    int(
                        local_electron["dot_products_q"]["runtime_parameters"][
                            "parameter_count"
                        ]
                    ),
                ),
                (
                    "dot_products_v_weights",
                    "sparse_uvu_weights",
                    int(
                        local_electron["dot_products_v"]["runtime_parameters"][
                            "parameter_count"
                        ]
                    ),
                ),
                ("mlp", "affine_mlp", local_electron["mlp"]),
            ],
        ),
    }


def _polar_contract_structure_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tag": contract["tag"],
        "interaction_count": contract["interaction_count"],
        "recursion_step_count": contract["recursion_step_count"],
        "charges": contract["charges"]["irreps"],
        "field": contract["field"]["irreps"],
        "potential": contract["potential"]["irreps"],
        "node_feats_irreps": contract["node_feats_irreps"],
        "spin_source_heads": [
            {
                "zero_charges": head["zero_charges"],
                "linear": _linear_structure_payload(head["linear"]),
            }
            for head in contract["spin_source_heads"]
        ],
        "feature_mixer": [
            _linear_structure_payload(linear)
            for linear in contract["feature_mixer"]["linears"]
        ],
        "field_update_blocks": [
            _polar_update_structure_payload(update)
            for update in contract["field_update_blocks"]
        ],
        "fukui_source_map": _polar_source_map_structure_payload(
            contract["fukui_source_map"]
        ),
        "local_electron_energy": _polar_local_electron_structure_payload(
            contract["local_electron_energy"]
        ),
        "flags": contract["flags"],
        # The unrolled chain is program structure, not learned data: a
        # topology with another depth or another stage order is a different
        # program, so the schedule participates in the structure fingerprint
        # exactly as the MH-1 node program's own schedule does.
        "execution": contract["execution"],
    }


def _strip_polar_pack_fingerprints(node: Any) -> None:
    if isinstance(node, Mapping):
        node.pop("fingerprint", None)
        for value in node.values():
            _strip_polar_pack_fingerprints(value)
    elif isinstance(node, list):
        for value in node:
            _strip_polar_pack_fingerprints(value)


def _polar_semantic_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(contract))
    for name in (
        "semantic_fingerprint",
        "generation_fingerprint",
        "structure_fingerprint",
    ):
        payload.pop(name, None)
    _strip_polar_pack_fingerprints(payload.get("runtime_parameter_packs"))
    return payload


def _finish_polar_contract_fingerprints(contract: dict[str, Any]) -> dict[str, Any]:
    contract["semantic_fingerprint"] = _fingerprint(_polar_semantic_payload(contract))
    contract["generation_fingerprint"] = _fingerprint(contract)
    contract["structure_fingerprint"] = _fingerprint(
        _polar_contract_structure_payload(contract)
    )
    return contract


def polar_execution_schedule(recursion_step_count: int) -> dict[str, Any]:
    """Return the unrolled PolarMACE global<->local chain for a recursion depth.

    The scheme is non-self-consistent by design (``num_update`` fixed-depth
    iterations of long-range electrostatic features, external-field coupling, the local charge
    update, and the closed-form Fukui equilibration -- no convergence
    criterion, no data-dependent control flow), so the chain is a fixed linear
    sequence whose only parameter is the depth.  It is contract data the way
    the MH-1 node program's own schedule is: a native orchestration reads it
    instead of hardcoding the order, and a topology with another depth is
    described by the same builder.

    Entry kinds: ``backbone`` names a region of the local work (the interaction
    layers, the feature mixer, the per-iteration field-dependent charge update,
    the post-loop local-electron-energy readout) and ``global_operator`` names
    one long-range operator kind.  ``iteration`` is the unrolled index -- 0 for
    the stages before the loop, 1..U for the unrolled updates, U+1 for the
    post-loop readout.  The C++ mirror is ``polar_execution_schedule`` in
    ``libsymmetrix/source/mace_polar_program.hpp``; the polar program probe
    pins the two sides equal.
    """

    if (
        isinstance(recursion_step_count, bool)
        or not isinstance(recursion_step_count, int)
        or recursion_step_count < 1
    ):
        raise ValueError(
            "the polar execution schedule requires at least one recursion step"
        )
    entries: list[dict[str, Any]] = [
        {"kind": "backbone", "name": "interactions", "iteration": 0},
        {"kind": "global_operator", "name": "kspace_geometry_cache", "iteration": 0},
        {"kind": "backbone", "name": "feature_mixer", "iteration": 0},
        {"kind": "global_operator", "name": "fukui_normalize", "iteration": 0},
    ]
    for step in range(1, recursion_step_count + 1):
        entries.extend(
            (
                {
                    "kind": "global_operator",
                    "name": "kspace_coulomb_features",
                    "iteration": step,
                },
                {
                    "kind": "global_operator",
                    "name": "external_field_projection",
                    "iteration": step,
                },
                {"kind": "backbone", "name": "field_update", "iteration": step},
                {
                    "kind": "global_operator",
                    "name": "fukui_normalize",
                    "iteration": step,
                },
            )
        )
    readout = recursion_step_count + 1
    entries.extend(
        (
            {
                "kind": "backbone",
                "name": "local_electron_energy",
                "iteration": readout,
            },
            {"kind": "global_operator", "name": "dipole_readout", "iteration": readout},
            {
                "kind": "global_operator",
                "name": "kspace_coulomb_energy",
                "iteration": readout,
            },
            {
                "kind": "global_operator",
                "name": "self_interaction_linear",
                "iteration": readout,
            },
            {
                "kind": "global_operator",
                "name": "analytic_correction_features",
                "iteration": readout,
            },
            {
                "kind": "global_operator",
                "name": "analytic_correction_energy",
                "iteration": readout,
            },
        )
    )
    return {
        "version": 1,
        "recursion_step_count": recursion_step_count,
        "entries": entries,
    }


def make_execution_polar_backbone_contract(
    model_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the parameter-free PolarMACE backbone-delta program identity."""

    if model_data.get("model_type") not in BACKBONE_MODEL_TYPES:
        raise ValueError(
            "Polar backbone contracts require a backbone model type "
            f"({', '.join(BACKBONE_MODEL_TYPES)})"
        )
    _require_keep_last_layer_irreps(model_data)
    descriptors = _polar_backbone_descriptors(model_data)
    contract = {
        "tag": POLAR_CONTRACT_TAG,
        "schema": POLAR_SCHEMA,
        "version": POLAR_SCHEMA_VERSION,
        "interaction_count": descriptors["interaction_count"],
        "recursion_step_count": descriptors["recursion_steps"],
        "keep_last_layer_irreps": True,
        "charges": descriptors["charges"],
        "field": descriptors["field"],
        "potential": descriptors["potential"],
        "field_norm_factor": descriptors["field_norm_factor"],
        "node_feats_irreps": descriptors["node_feats_irreps"],
        "spin_source_heads": descriptors["spin_source_heads"],
        "feature_mixer": descriptors["feature_mixer"],
        "field_update_blocks": descriptors["field_update_blocks"],
        "fukui_source_map": descriptors["fukui_source_map"],
        "local_electron_energy": descriptors["local_electron_energy"],
        "flags": descriptors["flags"],
        "execution": polar_execution_schedule(descriptors["recursion_steps"]),
        "layouts": {"feature_layout": "mul_ir"},
        "runtime_parameter_packs": _polar_runtime_parameter_packs(descriptors),
    }
    return _finish_polar_contract_fingerprints(contract)


def normalize_execution_polar_backbone_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a stored polar backbone contract and canonicalize it."""

    normalized = deepcopy(dict(contract))
    supplied_semantic = normalized.pop("semantic_fingerprint", None)
    supplied_generation = normalized.pop("generation_fingerprint", None)
    supplied_structure = normalized.pop("structure_fingerprint", None)
    if normalized.get("tag") != POLAR_CONTRACT_TAG:
        raise ValueError(f"Polar backbone contract tag must be {POLAR_CONTRACT_TAG!r}")
    if normalized.get("schema") != POLAR_SCHEMA:
        raise ValueError(f"Polar backbone contract schema must be {POLAR_SCHEMA!r}")
    if normalized.get("version") != POLAR_SCHEMA_VERSION:
        raise ValueError(
            f"Polar backbone contract version must be {POLAR_SCHEMA_VERSION}"
        )
    if normalized.get("keep_last_layer_irreps") is not True:
        raise ValueError("Polar backbone contracts require keep-last-layer readouts")
    if normalized.get("layouts") != {"feature_layout": "mul_ir"}:
        raise ValueError("Polar backbone layouts do not match the generated ABI")
    interaction_count = _require_int(
        normalized.get("interaction_count"), "polar interaction_count", 1
    )
    recursion_steps = _require_int(
        normalized.get("recursion_step_count"), "polar recursion_step_count", 1
    )
    heads = normalized.get("spin_source_heads")
    if not isinstance(heads, list) or len(heads) != interaction_count:
        raise ValueError("Polar backbone spin-source heads are invalid")
    for index, head in enumerate(heads):
        if not isinstance(head, dict) or head.get("index") != index:
            raise ValueError(
                "Polar backbone spin-source heads must use extraction order"
            )
    mixer = normalized.get("feature_mixer")
    mixer_linears = mixer.get("linears") if isinstance(mixer, dict) else None
    if not isinstance(mixer_linears, list) or len(mixer_linears) != interaction_count:
        raise ValueError("Polar backbone feature-mixer linears are invalid")
    updates = normalized.get("field_update_blocks")
    if not isinstance(updates, list) or len(updates) != recursion_steps:
        raise ValueError("Polar backbone field-update blocks are invalid")
    for index, update in enumerate(updates):
        if not isinstance(update, dict) or update.get("index") != index:
            raise ValueError(
                "Polar backbone field-update blocks must use extraction order"
            )
    # The unrolled chain is contract data: the stored schedule must be exactly
    # the one this recursion depth generates, the way the MH-1 node program's
    # own execution schedule is validated.
    if normalized.get("execution") != polar_execution_schedule(recursion_steps):
        raise ValueError("Polar backbone execution schedule is unsupported")

    for name in ("fukui_source_map", "local_electron_energy"):
        if not isinstance(normalized.get(name), dict):
            raise ValueError(f"Polar backbone {name} is invalid")
    flags = normalized.get("flags")
    if not isinstance(flags, dict) or set(flags) != {
        "add_local_electron_energy",
        "include_electrostatic_self_interaction",
        "quadrupole_feature_corrections",
        "field_si",
    }:
        raise ValueError("Polar backbone flags are incomplete")
    for name, value in flags.items():
        if not isinstance(value, bool):
            raise ValueError(f"Polar backbone flag {name} must be boolean")

    packs = normalized.get("runtime_parameter_packs")
    if not isinstance(packs, dict) or set(packs) != {
        "spin_source_heads",
        "feature_mixer",
        "field_update_blocks",
        "fukui_source_map",
        "local_electron_energy",
    }:
        raise ValueError("Polar backbone runtime parameter packs are incomplete")
    update_pack_list = packs["field_update_blocks"]
    if (
        not isinstance(update_pack_list, list)
        or len(update_pack_list) != recursion_steps
    ):
        raise ValueError("Polar backbone runtime parameter packs are incomplete")
    for block in update_pack_list:
        if not isinstance(block, dict) or set(block) != {
            "potential_embedding",
            "update_chain",
            "readout",
        }:
            raise ValueError("Polar backbone runtime parameter packs are incomplete")
    normalized["runtime_parameter_packs"] = {
        "spin_source_heads": _normalize_runtime_parameter_pack(
            packs["spin_source_heads"]
        ),
        "feature_mixer": _normalize_runtime_parameter_pack(packs["feature_mixer"]),
        "fukui_source_map": _normalize_runtime_parameter_pack(
            packs["fukui_source_map"]
        ),
        "local_electron_energy": _normalize_runtime_parameter_pack(
            packs["local_electron_energy"]
        ),
        "field_update_blocks": [
            {
                name: _normalize_runtime_parameter_pack(pack)
                for name, pack in block.items()
            }
            for block in update_pack_list
        ],
    }

    expected_semantic = _fingerprint(_polar_semantic_payload(normalized))
    normalized["semantic_fingerprint"] = expected_semantic
    expected_generation = _fingerprint(normalized)
    try:
        expected_structure = _fingerprint(_polar_contract_structure_payload(normalized))
    except (KeyError, TypeError) as error:
        raise ValueError("Polar backbone contract structure is incomplete") from error
    if supplied_semantic != expected_semantic:
        raise ValueError("Polar backbone semantic fingerprint is stale")
    if supplied_generation != expected_generation:
        raise ValueError("Polar backbone generation fingerprint is stale")
    if supplied_structure != expected_structure:
        raise ValueError("Polar backbone structure fingerprint is stale")
    normalized["generation_fingerprint"] = expected_generation
    normalized["structure_fingerprint"] = expected_structure
    return normalized


def validate_execution_polar_backbone_contract_for_model(
    model_data: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the stored polar backbone identity against the model data."""

    expected = make_execution_polar_backbone_contract(model_data)
    normalized = normalize_execution_polar_backbone_contract(contract)
    if normalized != expected:
        raise ValueError(
            "the embedded Execution POLAR_BACKBONE contract does not match the "
            "model definition"
        )
    return normalized
