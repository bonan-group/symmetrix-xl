"""Versioned, dependency-free JSON contract for exported NequIP models."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import sys
from typing import Any


NEQUIP_OAM_FORMAT_VERSION = 2
_MAX_TENSOR_ELEMENTS = sys.maxsize
_IRREP_TOKEN = re.compile(
    r"^(?P<multiplicity>[1-9][0-9]*)x(?P<l>[0-9]+)(?P<parity>[eo])$"
)
# Tensor placement tags never change what the exported model computes, so they
# are dropped at every depth.  ``precision`` is a runtime tag only at the
# contract root: a nested ``precision`` (for example
# ``layers[i].radial_basis.kokkos_radial.precision``) declares the precision the
# exported spline table is certified for and must participate in the
# fingerprint, so it is stripped at the root only.
_RUNTIME_KEYS = {"dtype", "device", "runtime_dtype"}
_ROOT_RUNTIME_KEYS = _RUNTIME_KEYS | {"precision"}
_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path} {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be a JSON object")
    return value


def _sequence(value: Any, path: str, *, allow_empty: bool = True) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _fail(path, "must be an array")
    if not allow_empty and not value:
        _fail(path, "must not be empty")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be finite") from error
    if not math.isfinite(result):
        _fail(path, "must be finite")
    if positive and result <= 0.0:
        _fail(path, "must be positive")
    return result


def _shape(value: Any, path: str) -> list[int]:
    values = _sequence(value, path)
    result = [
        _integer(item, f"{path}[{index}]", minimum=0)
        for index, item in enumerate(values)
    ]
    product = 1
    for index, extent in enumerate(result):
        if extent and product > _MAX_TENSOR_ELEMENTS // extent:
            _fail(
                path,
                f"product overflows the supported tensor bound at dimension {index}",
            )
        product *= extent
    return result


def _tensor(
    value: Any, path: str, *, rank: int | None = None
) -> tuple[list[int], list[Any]]:
    tensor = _mapping(value, path)
    if "shape" not in tensor:
        _fail(path, "must contain shape")
    if "values" not in tensor:
        _fail(path, "must contain values")
    shape = _shape(tensor["shape"], f"{path}.shape")
    if rank is not None and len(shape) != rank:
        _fail(f"{path}.shape", f"must have rank {rank}")
    values = _sequence(tensor["values"], f"{path}.values")
    expected = 1
    for extent in shape:
        expected *= extent
    if len(values) != expected:
        _fail(
            f"{path}.values",
            f"length {len(values)} does not agree with shape product {expected}",
        )
    for index, item in enumerate(values):
        _number(item, f"{path}.values[{index}]")
    return shape, list(values)


def _parse_irreps(value: Any, path: str) -> list[dict[str, int | str]]:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "must be a non-empty irreps string")
    blocks: list[dict[str, int | str]] = []
    for index, raw_token in enumerate(value.split("+")):
        token = raw_token.replace(" ", "")
        match = _IRREP_TOKEN.fullmatch(token)
        if match is None:
            _fail(
                path, f"contains invalid irrep token at position {index}: {raw_token!r}"
            )
        multiplicity = int(match.group("multiplicity"))
        degree = int(match.group("l"))
        blocks.append(
            {
                "multiplicity": multiplicity,
                "l": degree,
                "parity": match.group("parity"),
            }
        )
    return blocks


def _irreps_dimension(blocks: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(block["multiplicity"]) * (2 * int(block["l"]) + 1) for block in blocks
    )


def _irreps(value: Any, path: str) -> tuple[str, list[dict[str, int | str]], int]:
    blocks = _parse_irreps(value, path)
    return str(value), blocks, _irreps_dimension(blocks)


def _validate_linear(
    value: Any,
    path: str,
    *,
    require_output: bool = True,
    allow_matrix_weight: bool = False,
) -> tuple[int, int]:
    linear = _mapping(value, path)
    if "irreps_in" not in linear:
        _fail(path, "must contain irreps_in")
    _, input_blocks, input_dimension = _irreps(linear["irreps_in"], f"{path}.irreps_in")
    if require_output and "irreps_out" not in linear:
        _fail(path, "must contain irreps_out")
    if "irreps_out" in linear:
        _, output_blocks, output_dimension = _irreps(
            linear["irreps_out"], f"{path}.irreps_out"
        )
    else:
        output_blocks, output_dimension = [], 0
    for key in ("instructions", "bias", "output_mask"):
        if key not in linear:
            _fail(path, f"must contain {key}")
    instructions = _sequence(linear["instructions"], f"{path}.instructions")
    expected_weight_count = 0
    for index, raw_instruction in enumerate(instructions):
        instruction_path = f"{path}.instructions[{index}]"
        instruction = _mapping(raw_instruction, instruction_path)
        input_index = _integer(
            instruction.get("i_in", -1), f"{instruction_path}.i_in", minimum=0
        )
        output_index = _integer(
            instruction.get("i_out", -1), f"{instruction_path}.i_out", minimum=0
        )
        if input_index >= len(input_blocks):
            _fail(f"{instruction_path}.i_in", "is outside irreps_in block range")
        if output_index >= len(output_blocks):
            _fail(f"{instruction_path}.i_out", "is outside irreps_out block range")
        if input_blocks[input_index]["l"] != output_blocks[output_index]["l"]:
            _fail(
                f"{instruction_path}.i_out",
                "must have the same angular degree l as its input block",
            )
        path_shape = _shape(
            instruction.get("path_shape"), f"{instruction_path}.path_shape"
        )
        if len(path_shape) != 2:
            _fail(f"{instruction_path}.path_shape", "must have rank 2")
        expected_shape = [
            int(input_blocks[input_index]["multiplicity"]),
            int(output_blocks[output_index]["multiplicity"]),
        ]
        if path_shape != expected_shape:
            _fail(f"{instruction_path}.path_shape", f"must equal {expected_shape}")
        expected_weight_count += path_shape[0] * path_shape[1]
    weight_shape, _ = _tensor(linear.get("weight"), f"{path}.weight")
    if allow_matrix_weight and len(weight_shape) == 2:
        if weight_shape[1] != input_dimension:
            _fail(
                f"{path}.weight.shape",
                f"second dimension must equal input dimension {input_dimension}",
            )
        output_dimension = weight_shape[0]
    elif len(weight_shape) == 1:
        if expected_weight_count and weight_shape[0] != expected_weight_count:
            _fail(
                f"{path}.weight.shape",
                f"must contain {expected_weight_count} instruction weights",
            )
    else:
        _fail(f"{path}.weight.shape", "must have rank 1")
    bias_shape, _ = _tensor(linear["bias"], f"{path}.bias", rank=1)
    if output_dimension and bias_shape != [output_dimension]:
        _fail(f"{path}.bias.shape", f"must equal [{output_dimension}]")
    mask_shape, _ = _tensor(linear["output_mask"], f"{path}.output_mask", rank=1)
    if output_dimension and mask_shape != [output_dimension]:
        _fail(f"{path}.output_mask.shape", f"must equal [{output_dimension}]")
    return input_dimension, output_dimension


def _validate_tensor_product(
    value: Any, path: str, *, node_in: str, edge: str, node_out: str
) -> int:
    product = _mapping(value, path)
    for key in ("irreps_in1", "irreps_in2", "irreps_out"):
        if key not in product:
            _fail(path, f"must contain {key}")
    in1_text, in1_blocks, _ = _irreps(product["irreps_in1"], f"{path}.irreps_in1")
    in2_text, in2_blocks, _ = _irreps(product["irreps_in2"], f"{path}.irreps_in2")
    out_text, out_blocks, output_dimension = _irreps(
        product["irreps_out"], f"{path}.irreps_out"
    )
    if in1_text.replace(" ", "") != node_in.replace(" ", ""):
        _fail(f"{path}.irreps_in1", "must equal the layer node_irreps_in")
    if in2_text.replace(" ", "") != edge.replace(" ", ""):
        _fail(f"{path}.irreps_in2", "must equal the layer edge_irreps")
    if out_text.replace(" ", "") != node_out.replace(" ", ""):
        _fail(f"{path}.irreps_out", "must equal the layer node_irreps_out")
    instructions = _sequence(
        product.get("instructions"), f"{path}.instructions", allow_empty=False
    )
    expected_weight_count = 0
    for index, raw_instruction in enumerate(instructions):
        instruction_path = f"{path}.instructions[{index}]"
        instruction = _mapping(raw_instruction, instruction_path)
        i_in1 = _integer(
            instruction.get("i_in1", -1), f"{instruction_path}.i_in1", minimum=0
        )
        i_in2 = _integer(
            instruction.get("i_in2", -1), f"{instruction_path}.i_in2", minimum=0
        )
        i_out = _integer(
            instruction.get("i_out", -1), f"{instruction_path}.i_out", minimum=0
        )
        for name, block_index, blocks in (
            ("i_in1", i_in1, in1_blocks),
            ("i_in2", i_in2, in2_blocks),
            ("i_out", i_out, out_blocks),
        ):
            if block_index >= len(blocks):
                _fail(
                    f"{instruction_path}.{name}",
                    "is outside the corresponding irreps block range",
                )
        mode = instruction.get("connection_mode")
        if mode not in {"uvu", "uuu", "uvw"}:
            _fail(
                f"{instruction_path}.connection_mode", "must be 'uvu', 'uuu', or 'uvw'"
            )
        has_weight = instruction.get("has_weight")
        if not isinstance(has_weight, bool):
            _fail(f"{instruction_path}.has_weight", "must be a boolean")
        shape = _shape(instruction.get("path_shape"), f"{instruction_path}.path_shape")
        m1 = int(in1_blocks[i_in1]["multiplicity"])
        m2 = int(in2_blocks[i_in2]["multiplicity"])
        m3 = int(out_blocks[i_out]["multiplicity"])
        expected_shape = (
            [m1, m2, m3] if mode == "uvw" else ([m1, m2] if mode == "uvu" else [m1])
        )
        if mode != "uvw" and m3 != m1:
            _fail(
                f"{instruction_path}.i_out",
                "must preserve the first input multiplicity for this connection mode",
            )
        if shape != expected_shape:
            _fail(f"{instruction_path}.path_shape", f"must equal {expected_shape}")
        if has_weight:
            expected_weight_count += math.prod(shape)
        if "path_weight" not in instruction:
            _fail(instruction_path, "must contain path_weight")
        _number(instruction["path_weight"], f"{instruction_path}.path_weight")
        wigner_shape, _ = _tensor(
            instruction.get("wigner_3j"), f"{instruction_path}.wigner_3j", rank=3
        )
        expected_wigner_shape = [
            2 * int(in1_blocks[i_in1]["l"]) + 1,
            2 * int(in2_blocks[i_in2]["l"]) + 1,
            2 * int(out_blocks[i_out]["l"]) + 1,
        ]
        if wigner_shape != expected_wigner_shape:
            _fail(
                f"{instruction_path}.wigner_3j.shape",
                f"must equal {expected_wigner_shape}",
            )
    weight_shape, _ = _tensor(product.get("weight"), f"{path}.weight")
    if len(weight_shape) != 1:
        _fail(f"{path}.weight.shape", "must have rank 1")
    if weight_shape[0] != expected_weight_count:
        _fail(
            f"{path}.weight.shape",
            f"must contain {expected_weight_count} instruction weights",
        )
    if "output_mask" not in product:
        _fail(path, "must contain output_mask")
    output_shape, _ = _tensor(product["output_mask"], f"{path}.output_mask", rank=1)
    if output_shape != [output_dimension]:
        _fail(f"{path}.output_mask.shape", f"must equal [{output_dimension}]")
    return expected_weight_count


def _validate_radial_basis(value: Any, path: str, *, expected_output_size: int) -> None:
    radial = _mapping(value, path)
    if not isinstance(radial.get("type"), str) or not radial["type"]:
        _fail(f"{path}.type", "must be a non-empty string")
    num_basis = _integer(radial.get("num_basis"), f"{path}.num_basis", minimum=1)
    cutoff = _mapping(radial.get("cutoff"), f"{path}.cutoff")
    if not isinstance(cutoff.get("type"), str) or not cutoff["type"]:
        _fail(f"{path}.cutoff.type", "must be a non-empty string")
    if "p" in cutoff:
        _number(cutoff["p"], f"{path}.cutoff.p", positive=True)
    # The native evaluator uses one fixed, orthonormal Bessel basis and rejects
    # metadata that could describe a different normalization or prefactor.
    for key in (
        "normalization",
        "radial_normalization",
        "normalize",
        "prefactor",
        "basis_normalization",
    ):
        if key not in radial:
            continue
        _fail(f"{path}.{key}", "is unsupported by the fixed radial basis")
    mlp = radial.get("mlp")
    if isinstance(mlp, Mapping):
        layers = mlp.get("layers")
        if layers is None and "weights" in mlp:
            weights = _sequence(mlp["weights"], f"{path}.mlp.weights")
            biases = _sequence(mlp.get("biases", []), f"{path}.mlp.biases")
            if len(weights) != len(biases):
                _fail(f"{path}.mlp.biases", "length must match mlp.weights")
            layers = [
                {"weight": weight, "bias": bias}
                for weight, bias in zip(weights, biases)
            ]
        if layers is None:
            _fail(f"{path}.mlp", "must contain layers")
    else:
        layers = mlp
    layers = _sequence(layers, f"{path}.mlp.layers", allow_empty=False)
    previous_output: int | None = None
    for index, raw_layer in enumerate(layers):
        layer_path = f"{path}.mlp.layers[{index}]"
        layer = _mapping(raw_layer, layer_path)
        if layer.get("type") == "silu":
            if previous_output is None:
                _fail(layer_path, "SiLU cannot be the first radial MLP layer")
            continue
        weight_shape, _ = _tensor(layer.get("weight"), f"{layer_path}.weight", rank=2)
        bias_shape, _ = _tensor(layer.get("bias"), f"{layer_path}.bias", rank=1)
        if bias_shape != [weight_shape[0]]:
            _fail(f"{layer_path}.bias.shape", f"must equal [{weight_shape[0]}]")
        if index == 0 and weight_shape[1] != num_basis:
            _fail(
                f"{layer_path}.weight.shape",
                f"input width must equal num_basis {num_basis}",
            )
        if previous_output is not None and weight_shape[1] != previous_output:
            _fail(
                f"{layer_path}.weight.shape",
                f"input width must equal previous output width {previous_output}",
            )
        previous_output = weight_shape[0]
    if previous_output != expected_output_size:
        _fail(
            f"{path}.mlp.layers[-1].weight.shape",
            f"output width must equal tensor-product dynamic weight count {expected_output_size}",
        )


def _validate_atomic_numbers(data: Mapping[str, Any]) -> None:
    atomic_numbers = _sequence(
        data.get("atomic_numbers"), "$.atomic_numbers", allow_empty=False
    )
    model_atomic_numbers = _sequence(
        data.get("model_atomic_numbers"), "$.model_atomic_numbers", allow_empty=False
    )
    atomic = [
        _integer(value, f"$.atomic_numbers[{index}]", minimum=1)
        for index, value in enumerate(atomic_numbers)
    ]
    model = [
        _integer(value, f"$.model_atomic_numbers[{index}]", minimum=1)
        for index, value in enumerate(model_atomic_numbers)
    ]
    if len(set(atomic)) != len(atomic):
        _fail("$.atomic_numbers", "must not contain duplicates")
    if len(set(model)) != len(model):
        _fail("$.model_atomic_numbers", "must not contain duplicates")
    if set(atomic) != set(model):
        _fail(
            "$.model_atomic_numbers",
            "must contain the same universal element set as atomic_numbers",
        )
    mapping = data.get("atomic_number_to_model_type")
    if mapping is None:
        _fail("$.atomic_number_to_model_type", "is required")
    if isinstance(mapping, Mapping):
        normalized_mapping = {str(key): value for key, value in mapping.items()}
        if set(normalized_mapping) != {str(value) for value in model}:
            _fail(
                "$.atomic_number_to_model_type", "keys must cover model_atomic_numbers"
            )
        indices = []
        for number in model:
            indices.append(
                _integer(
                    normalized_mapping[str(number)],
                    f"$.atomic_number_to_model_type.{number}",
                    minimum=0,
                )
            )
    else:
        values = _sequence(mapping, "$.atomic_number_to_model_type", allow_empty=False)
        indices = [
            _integer(value, f"$.atomic_number_to_model_type[{index}]", minimum=0)
            for index, value in enumerate(values)
        ]
    if sorted(indices) != list(range(len(model))):
        _fail(
            "$.atomic_number_to_model_type",
            "must be a permutation of model type indices",
        )


def _validate_oam_linear(
    value: Any, path: str, *, expected_in: str, expected_out: str
) -> None:
    input_dim, output_dim = _validate_linear(value, path)
    _, _, expected_input_dim = _irreps(expected_in, f"{path}.expected_irreps_in")
    _, _, expected_output_dim = _irreps(expected_out, f"{path}.expected_irreps_out")
    if (input_dim, output_dim) != (expected_input_dim, expected_output_dim):
        _fail(path, "dimensions do not match the layer irreps")


def _validate_oam_layer(value: Any, path: str, *, previous_out: str | None) -> None:
    layer = _mapping(value, path)
    required = (
        "node_irreps_in",
        "edge_irreps",
        "tp_output_irreps",
        "node_irreps_pre_gate",
        "node_irreps_out",
        "pre_linear",
        "radial_basis",
        "edge_tensor_product",
        "post_linear",
        "self_connection",
        "normalization",
        "gate",
    )
    for key in required:
        if key not in layer:
            _fail(path, f"must contain {key}")
    node_in = str(layer["node_irreps_in"])
    edge = str(layer["edge_irreps"])
    tp_output = str(layer["tp_output_irreps"])
    pre_gate = str(layer["node_irreps_pre_gate"])
    node_out = str(layer["node_irreps_out"])
    _irreps(node_in, f"{path}.node_irreps_in")
    _irreps(edge, f"{path}.edge_irreps")
    _irreps(tp_output, f"{path}.tp_output_irreps")
    _irreps(pre_gate, f"{path}.node_irreps_pre_gate")
    _irreps(node_out, f"{path}.node_irreps_out")
    if previous_out is not None and node_in.replace(" ", "") != previous_out.replace(
        " ", ""
    ):
        _fail(f"{path}.node_irreps_in", "must match the previous layer node_irreps_out")
    _validate_oam_linear(
        layer["pre_linear"],
        f"{path}.pre_linear",
        expected_in=node_in,
        expected_out=node_in,
    )
    dynamic = _validate_tensor_product(
        layer["edge_tensor_product"],
        f"{path}.edge_tensor_product",
        node_in=node_in,
        edge=edge,
        node_out=tp_output,
    )
    product = _mapping(layer["edge_tensor_product"], f"{path}.edge_tensor_product")
    weight_numel = _integer(
        product.get("weight_numel"),
        f"{path}.edge_tensor_product.weight_numel",
        minimum=0,
    )
    if weight_numel != dynamic:
        _fail(f"{path}.edge_tensor_product.weight_numel", f"must equal {dynamic}")
    _validate_radial_basis(
        layer["radial_basis"], f"{path}.radial_basis", expected_output_size=dynamic
    )
    radial = _mapping(layer["radial_basis"], f"{path}.radial_basis")
    if "bessel_weights" not in radial:
        _fail(f"{path}.radial_basis", "must contain bessel_weights")
    bessel_shape, _ = _tensor(
        radial["bessel_weights"], f"{path}.radial_basis.bessel_weights", rank=1
    )
    if bessel_shape != [int(radial["num_basis"])]:
        _fail(f"{path}.radial_basis.bessel_weights.shape", "must equal [num_basis]")
    _number(radial.get("input_factor"), f"{path}.radial_basis.input_factor")
    _validate_oam_linear(
        layer["post_linear"],
        f"{path}.post_linear",
        expected_in=tp_output,
        expected_out=pre_gate,
    )
    if layer["self_connection"] is not None:
        self_path = f"{path}.self_connection"
        self_connection = _mapping(layer["self_connection"], self_path)
        if "irreps_in1" in self_connection:
            self_in2 = str(self_connection.get("irreps_in2", ""))
            _validate_tensor_product(
                self_connection,
                self_path,
                node_in=node_in,
                edge=self_in2,
                node_out=pre_gate,
            )
        else:
            _validate_oam_linear(
                self_connection, self_path, expected_in=node_in, expected_out=pre_gate
            )
    normalization = _mapping(layer["normalization"], f"{path}.normalization")
    if normalization.get("type") != "global_scalar":
        _fail(f"{path}.normalization.type", "must be 'global_scalar'")
    _number(normalization.get("factor"), f"{path}.normalization.factor", positive=True)
    gate = _mapping(layer["gate"], f"{path}.gate")
    if gate.get("type") != "scalar_gate":
        _fail(f"{path}.gate.type", "must be 'scalar_gate'")

    def gate_irreps(name: str) -> tuple[list[dict[str, int | str]], int]:
        raw = gate.get(name)
        if raw in (None, "", "0x0e"):
            return [], 0
        _, blocks, dimension = _irreps(raw, f"{path}.gate.{name}")
        return blocks, dimension

    scalar, scalar_dim = gate_irreps("scalar_irreps")
    gates, gate_dim = gate_irreps("gate_irreps")
    gated, gated_dim = gate_irreps("gated_irreps")
    if any(int(block["l"]) != 0 for block in scalar + gates) or any(
        block["parity"] != "e" for block in gates
    ):
        _fail(f"{path}.gate", "scalar and gate irreps must be even l=0")
    _, pre_blocks, pre_dim = _irreps(pre_gate, f"{path}.node_irreps_pre_gate")
    _, out_blocks, out_dim = _irreps(node_out, f"{path}.node_irreps_out")

    def aggregate(blocks: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], int]:
        result: dict[tuple[int, str], int] = {}
        for block in blocks:
            key = (int(block["l"]), str(block["parity"]))
            result[key] = result.get(key, 0) + int(block["multiplicity"])
        return result

    if aggregate(pre_blocks) != aggregate(scalar + gates + gated) or aggregate(
        out_blocks
    ) != aggregate(scalar + gated):
        _fail(f"{path}.gate", "irreps do not match scalar_gate concatenation")
    if (
        pre_dim != scalar_dim + gate_dim + gated_dim
        or out_dim != scalar_dim + gated_dim
    ):
        _fail(f"{path}.gate", "dimensions do not match scalar_gate concatenation")


def _validate_kokkos_radial(
    value: Any,
    path: str,
    *,
    expected_function_count: int,
    expected_cutoff: float,
) -> None:
    """Validate one type-independent compact spline table certificate."""
    radial = _mapping(value, path)
    for key in ("pair_tables", "element_pair_tables", "edge_tables", "tables"):
        if key in radial:
            _fail(f"{path}.{key}", "element-pair radial tables are unsupported")
    required = (
        "grid_min",
        "cutoff",
        "num_spline_points",
        "spacing",
        "function_count",
        "edge_type_count",
        "model_fingerprint",
        "table_fingerprint",
        "precision",
        "exporter_version",
        "certification",
    )
    for key in required:
        if key not in radial:
            _fail(path, f"must contain {key}")
    grid_min = _number(radial["grid_min"], f"{path}.grid_min")
    if grid_min < 0.0:
        _fail(f"{path}.grid_min", "must be non-negative")
    cutoff = _number(radial["cutoff"], f"{path}.cutoff", positive=True)
    if not math.isclose(cutoff, expected_cutoff, rel_tol=1e-12, abs_tol=1e-12):
        _fail(f"{path}.cutoff", "must match r_cut")
    points = _integer(
        radial["num_spline_points"], f"{path}.num_spline_points", minimum=4
    )
    spacing = _number(radial["spacing"], f"{path}.spacing", positive=True)
    if cutoff <= grid_min:
        _fail(f"{path}.cutoff", "must exceed grid_min")
    expected_spacing = (cutoff - grid_min) / float(points - 1)
    if not math.isclose(spacing, expected_spacing, rel_tol=1e-12, abs_tol=1e-12):
        _fail(f"{path}.spacing", "must describe a uniform grid from grid_min to cutoff")
    function_count = _integer(
        radial["function_count"], f"{path}.function_count", minimum=1
    )
    if function_count != expected_function_count:
        _fail(
            f"{path}.function_count",
            f"must equal radial output size {expected_function_count}",
        )
    if _integer(radial["edge_type_count"], f"{path}.edge_type_count", minimum=1) != 1:
        _fail(
            f"{path}.edge_type_count",
            "must equal one for type-independent OAM radial tables",
        )
    for key in ("model_fingerprint", "table_fingerprint"):
        fingerprint = radial[key]
        if (
            not isinstance(fingerprint, str)
            or _FINGERPRINT.fullmatch(fingerprint) is None
        ):
            _fail(f"{path}.{key}", "must be a sha256 fingerprint")
    if radial["precision"] not in {"float32", "float64"}:
        _fail(f"{path}.precision", "must be float32 or float64")
    if (
        not isinstance(radial["exporter_version"], str)
        or not radial["exporter_version"]
    ):
        _fail(f"{path}.exporter_version", "must be a non-empty string")
    certification = _mapping(radial["certification"], f"{path}.certification")
    for key in (
        "grid_points",
        "clamped_points",
        "max_abs_value_error",
        "max_abs_derivative_error",
    ):
        if key not in certification:
            _fail(f"{path}.certification", f"must contain {key}")
    if (
        _integer(
            certification["grid_points"], f"{path}.certification.grid_points", minimum=1
        )
        != points
    ):
        _fail(f"{path}.certification.grid_points", "must equal num_spline_points")
    clamped = _integer(
        certification["clamped_points"],
        f"{path}.certification.clamped_points",
        minimum=0,
    )
    if clamped > points:
        _fail(f"{path}.certification.clamped_points", "cannot exceed grid_points")
    for key in ("max_abs_value_error", "max_abs_derivative_error"):
        _number(certification[key], f"{path}.certification.{key}")


def validate_nequip_oam_data(data: Mapping[str, Any]) -> None:
    """Validate the explicit NequIP OAM v2 export contract."""
    root = _mapping(data, "$")
    if root.get("model_type") != "NequIP":
        _fail("$.model_type", "must be 'NequIP'")
    if root.get("symmetrix_nequip_format_version") != NEQUIP_OAM_FORMAT_VERSION:
        _fail(
            "$.symmetrix_nequip_format_version", f"must be {NEQUIP_OAM_FORMAT_VERSION}"
        )
    if root.get("architecture") != "oam":
        _fail("$.architecture", "must be 'oam'")
    metadata = _mapping(root.get("model_metadata"), "$.model_metadata")
    for key in (
        "source_package",
        "source_version",
        "exporter_version",
        "exporter_fingerprint",
    ):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            _fail(f"$.model_metadata.{key}", "must be a non-empty string")
    _validate_atomic_numbers(root)
    _number(root.get("r_cut"), "$.r_cut", positive=True)
    embedding = _mapping(root.get("node_attributes"), "$.node_attributes")
    embedding_shape, _ = _tensor(embedding, "$.node_attributes", rank=2)
    if embedding_shape[0] != len(root["model_atomic_numbers"]):
        _fail("$.node_attributes.shape", "must contain one row per model atomic number")
    edge_embedding = _mapping(root.get("edge_embedding"), "$.edge_embedding")
    bessel_shape, _ = _tensor(
        edge_embedding.get("bessel_weights"), "$.edge_embedding.bessel_weights", rank=1
    )
    if not bessel_shape or bessel_shape[0] < 1:
        _fail(
            "$.edge_embedding.bessel_weights.shape",
            "must contain at least one frequency",
        )
    cutoff = _mapping(edge_embedding.get("cutoff"), "$.edge_embedding.cutoff")
    if cutoff.get("type") != "polynomial":
        _fail("$.edge_embedding.cutoff.type", "must be 'polynomial'")
    _number(cutoff.get("p"), "$.edge_embedding.cutoff.p", positive=True)
    _number(edge_embedding.get("factor"), "$.edge_embedding.factor")
    layers = _sequence(root.get("layers"), "$.layers", allow_empty=False)
    if _integer(root.get("num_layers"), "$.num_layers", minimum=1) != len(layers):
        _fail("$.num_layers", "must equal layers length")
    previous = None
    for index, layer in enumerate(layers):
        _validate_oam_layer(layer, f"$.layers[{index}]", previous_out=previous)
        radial = _mapping(layer, f"$.layers[{index}]")
        if "kokkos_radial" in radial:
            product = _mapping(
                radial["edge_tensor_product"], f"$.layers[{index}].edge_tensor_product"
            )
            dynamic = _integer(
                product.get("weight_numel"),
                f"$.layers[{index}].edge_tensor_product.weight_numel",
                minimum=1,
            )
            _validate_kokkos_radial(
                radial["kokkos_radial"],
                f"$.layers[{index}].kokkos_radial",
                expected_function_count=dynamic,
                expected_cutoff=float(root["r_cut"]),
            )
        previous = str(_mapping(layer, f"$.layers[{index}]")["node_irreps_out"])
    readout = _mapping(root.get("readout"), "$.readout")
    final = _mapping(layers[-1], "$.layers[-1]")["node_irreps_out"]
    _validate_oam_linear(
        readout, "$.readout", expected_in=str(final), expected_out="1x0e"
    )
    for key in ("per_type_shift", "per_type_scale"):
        shape, _ = _tensor(root.get(key), f"$.{key}", rank=1)
        if shape != [len(root["model_atomic_numbers"])]:
            _fail(f"$.{key}.shape", "must contain one value per model atomic number")
    zbl = _mapping(root.get("zbl"), "$.zbl")
    if zbl.get("type") != "zbl":
        _fail("$.zbl.type", "must be 'zbl'")
    _number(zbl.get("cutoff"), "$.zbl.cutoff", positive=True)
    if "cutoff_p" in zbl:
        _number(zbl.get("cutoff_p"), "$.zbl.cutoff_p", positive=True)
    _number(
        zbl.get("half_energy_prefactor"), "$.zbl.half_energy_prefactor", positive=True
    )
    z_atomic = _sequence(
        zbl.get("atomic_numbers"), "$.zbl.atomic_numbers", allow_empty=False
    )
    z_exp = _sequence(
        zbl.get("screening_exponents"), "$.zbl.screening_exponents", allow_empty=False
    )
    z_coeff = _sequence(
        zbl.get("screening_coefficients"),
        "$.zbl.screening_coefficients",
        allow_empty=False,
    )
    if len(z_exp) != len(z_coeff):
        _fail(
            "$.zbl",
            "screening_exponents and screening_coefficients must have equal lengths",
        )
    for i, value in enumerate(z_atomic):
        _integer(value, f"$.zbl.atomic_numbers[{i}]", minimum=1)
    for name, values in (
        ("screening_exponents", z_exp),
        ("screening_coefficients", z_coeff),
    ):
        for i, value in enumerate(values):
            _number(value, f"$.zbl.{name}[{i}]")
    if len(set(z_atomic)) != len(z_atomic):
        _fail("$.zbl.atomic_numbers", "must not contain duplicates")
    if sorted(int(value) for value in z_atomic) != sorted(
        int(value) for value in root["model_atomic_numbers"]
    ):
        _fail("$.zbl.atomic_numbers", "must match the model element table")


def load_nequip_oam_data(path: str | Path) -> dict[str, Any]:
    """Load and validate an exported NequIP OAM v2 contract."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"{source} JSON root must be a JSON object")
    validate_nequip_oam_data(data)
    return dict(data)


def _canonical(value: Any, *, root: bool = False, model_metadata: bool = False) -> Any:
    if isinstance(value, Mapping):
        skipped = _ROOT_RUNTIME_KEYS if root else _RUNTIME_KEYS
        return {
            str(key): _canonical(
                item,
                model_metadata=root and str(key) == "model_metadata",
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in skipped
            and not (model_metadata and str(key) == "archive")
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract contains a non-finite number")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(f"contract contains unsupported value {type(value).__name__}")


def nequip_contract_fingerprint(data: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 identity for a validated contract."""

    validate_nequip_oam_data(data)
    encoded = json.dumps(
        _canonical(data, root=True),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
