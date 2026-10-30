"""Resolve the generated-code architecture of a validated OAM contract."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .jit_codegen_common import canonical_digest, irreps_dimension
from .nequip_schema import _canonical, _irreps, validate_nequip_oam_data

OAM_ARCHITECTURE_SCHEMA = "symmetrix.oam.generated-architecture/1"


def _blocks(text: str) -> list[dict[str, Any]]:
    return _irreps(text, "irreps")[1]


def _l_max(text: str) -> int:
    return max(int(block["l"]) for block in _blocks(text))


def resolve_oam_architecture(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return every contract field that changes generated OAM source code.

    Learned tensors, element tables, archive provenance, and filesystem paths
    remain runtime model data.  The returned descriptor contains the actual
    layer/irrep/channel/tensor-product layout used to select generated code.
    """

    validate_nequip_oam_data(contract)
    layers = []
    for index, layer in enumerate(contract["layers"]):
        product = layer["edge_tensor_product"]
        radial = layer["kokkos_radial"]
        source_irreps = str(layer["pre_linear"]["irreps_out"])
        edge_irreps = str(layer["edge_irreps"])
        output_irreps = str(layer["tp_output_irreps"])
        # Preserve the exact split/coalesced post-linear irrep normalization so
        # model-free architecture descriptors can reconstruct a valid contract.
        # This stage is runtime model math, not generated full-message code.
        from .oam_receiver_codegen import post_linear_structure

        post_linear = post_linear_structure(layer)
        layers.append(
            {
                "index": index,
                "source_irreps": source_irreps,
                "source_dimension": irreps_dimension(_blocks(source_irreps)),
                "edge_irreps": edge_irreps,
                "edge_dimension": irreps_dimension(_blocks(edge_irreps)),
                "edge_l_max": _l_max(edge_irreps),
                "output_irreps": output_irreps,
                "output_dimension": irreps_dimension(_blocks(output_irreps)),
                "tensor_product": _canonical(
                    {
                        key: product[key]
                        for key in (
                            "irreps_in1",
                            "irreps_in2",
                            "irreps_out",
                            "instructions",
                            "output_mask",
                            "weight_numel",
                        )
                    }
                ),
                "post_linear": post_linear,
                "spline_function_count": int(radial["function_count"]),
                "spline_interval_count": int(radial["num_spline_points"]) - 1,
                "normalization_factor": float(
                    layer["normalization"].get("factor", 1.0)
                ),
            }
        )
    return validate_oam_architecture(
        {
            "schema": OAM_ARCHITECTURE_SCHEMA,
            "layer_count": len(layers),
            "layers": layers,
        }
    )


def validate_oam_architecture(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an explicit generated-code architecture."""

    if not isinstance(value, Mapping):
        raise ValueError("OAM architecture must be an object")
    if value.get("schema") != OAM_ARCHITECTURE_SCHEMA:
        raise ValueError("OAM architecture schema is unsupported")
    layers = value.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("OAM architecture layers must be a non-empty list")
    if value.get("layer_count") != len(layers):
        raise ValueError("OAM architecture layer_count does not match layers")
    normalized = []
    for index, row in enumerate(layers):
        if not isinstance(row, Mapping) or row.get("index") != index:
            raise ValueError("OAM architecture layer indices must be contiguous")
        source_irreps = str(row.get("source_irreps", ""))
        edge_irreps = str(row.get("edge_irreps", ""))
        output_irreps = str(row.get("output_irreps", ""))
        expected = {
            "source_dimension": irreps_dimension(_blocks(source_irreps)),
            "edge_dimension": irreps_dimension(_blocks(edge_irreps)),
            "edge_l_max": _l_max(edge_irreps),
            "output_dimension": irreps_dimension(_blocks(output_irreps)),
        }
        for field, expected_value in expected.items():
            if row.get(field) != expected_value:
                raise ValueError(f"OAM architecture {field} is inconsistent")
        product = row.get("tensor_product")
        required_product = {
            "irreps_in1",
            "irreps_in2",
            "irreps_out",
            "instructions",
            "output_mask",
            "weight_numel",
        }
        if not isinstance(product, Mapping) or set(product) != required_product:
            raise ValueError("OAM architecture tensor_product is incomplete")
        post_linear = row.get("post_linear")
        required_post_linear = {
            "irreps_in",
            "irreps_out",
            "instructions",
            "output_mask",
            "weight_shape",
            "bias_present",
            "bias_shape",
        }
        if (
            not isinstance(post_linear, Mapping)
            or set(post_linear) != required_post_linear
            or not isinstance(post_linear["instructions"], list)
            or not post_linear["instructions"]
            or not isinstance(post_linear["bias_present"], bool)
        ):
            raise ValueError("OAM architecture post_linear topology is incomplete")
        function_count = row.get("spline_function_count")
        intervals = row.get("spline_interval_count")
        if (
            isinstance(function_count, bool)
            or not isinstance(function_count, int)
            or function_count <= 0
            or function_count != product["weight_numel"]
            or isinstance(intervals, bool)
            or not isinstance(intervals, int)
            or intervals <= 0
        ):
            raise ValueError("OAM architecture spline dimensions are invalid")
        normalization = row.get("normalization_factor")
        if (
            isinstance(normalization, bool)
            or not isinstance(normalization, (int, float))
            or not math.isfinite(float(normalization))
        ):
            raise ValueError("OAM architecture normalization is invalid")
        normalized.append(_canonical(dict(row)))
    return {
        "schema": OAM_ARCHITECTURE_SCHEMA,
        "layer_count": len(normalized),
        "layers": normalized,
    }


def oam_codegen_contract(architecture: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the model-free layer input consumed by OAM codegen."""

    architecture = validate_oam_architecture(architecture)
    return {
        "layers": [
            {
                "pre_linear": {"irreps_out": row["source_irreps"]},
                "edge_irreps": row["edge_irreps"],
                "tp_output_irreps": row["output_irreps"],
                "node_irreps_pre_gate": row["post_linear"]["irreps_out"],
                "post_linear": {
                    "irreps_in": row["post_linear"]["irreps_in"],
                    "irreps_out": row["post_linear"]["irreps_out"],
                    "instructions": row["post_linear"]["instructions"],
                    "output_mask": row["post_linear"]["output_mask"],
                    "weight": {
                        "shape": row["post_linear"]["weight_shape"],
                        "values": [0.0] * math.prod(row["post_linear"]["weight_shape"]),
                    },
                    "bias": {
                        "shape": row["post_linear"]["bias_shape"],
                        "values": (
                            [0.0] * math.prod(row["post_linear"]["bias_shape"])
                            if row["post_linear"]["bias_present"]
                            else []
                        ),
                    },
                },
                "edge_tensor_product": row["tensor_product"],
                "kokkos_radial": {
                    "function_count": row["spline_function_count"],
                    "num_spline_points": row["spline_interval_count"] + 1,
                },
                "normalization": {"factor": row["normalization_factor"]},
            }
            for row in architecture["layers"]
        ]
    }


def oam_architecture_fingerprint(contract_or_architecture: Mapping[str, Any]) -> str:
    """Return the loader identity of an explicit generated architecture."""

    if contract_or_architecture.get("schema") == OAM_ARCHITECTURE_SCHEMA:
        architecture = validate_oam_architecture(contract_or_architecture)
    else:
        architecture = resolve_oam_architecture(contract_or_architecture)
    return canonical_digest(architecture)


def oam_tensor_product_architecture_fingerprint(
    contract_or_architecture: Mapping[str, Any],
) -> str:
    """Return the native tensor-product fingerprint without post-linear topology."""

    if contract_or_architecture.get("schema") == OAM_ARCHITECTURE_SCHEMA:
        architecture = validate_oam_architecture(contract_or_architecture)
    else:
        architecture = resolve_oam_architecture(contract_or_architecture)
    base = {
        **architecture,
        "layers": [
            {key: value for key, value in layer.items() if key != "post_linear"}
            for layer in architecture["layers"]
        ],
    }
    return canonical_digest(base)


__all__ = [
    "OAM_ARCHITECTURE_SCHEMA",
    "oam_codegen_contract",
    "oam_architecture_fingerprint",
    "oam_tensor_product_architecture_fingerprint",
    "resolve_oam_architecture",
    "validate_oam_architecture",
]
