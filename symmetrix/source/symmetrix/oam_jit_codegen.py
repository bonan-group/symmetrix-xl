"""Deterministic host code generation for NequIP-OAM tensor products."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .jit import JIT_GENERATION_VERSION, ArtifactEnvelope
from .jit_codegen_common import (
    block_offsets,
    canonical_digest,
    cpp_string_literal,
    irreps_dimension,
    lane_column_permutation,
    scalar_literal,
)
from .nequip_schema import (
    _irreps,
    validate_nequip_oam_data,
)
from .oam_architecture import (
    oam_tensor_product_architecture_fingerprint,
    resolve_oam_architecture,
    validate_oam_architecture,
)

ABI_TAG = "symmetrix.jit.oam.host-plugin/1"
ABI_VERSION = 1
FORWARD_CAPABILITY = 1
REVERSE_CAPABILITY = 2
# The identity envelope every OAM host metadata builder publishes: the ABI tag
# and version above, the codegen artifact id, the generation/semantic/structure
# fingerprint triple, and the capability word its callbacks implement.
OAM_HOST_ARTIFACT_ENVELOPE = ArtifactEnvelope(
    "OAM host",
    identity_fingerprint="generation_fingerprint",
    capabilities_key="required_capabilities",
)
_ABI_HEADER = (
    Path(__file__)
    .with_name("oam_jit_host_plugin_abi.h")
    .read_text(encoding="ascii")
    .removeprefix("#pragma once\n\n")
)


def _irrep_blocks(text: str) -> list[dict[str, Any]]:
    return _irreps(text, "irreps")[1]


def oam_jit_metadata(
    contract: Mapping[str, Any],
    precision: str = "float64",
    *,
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ABI and model metadata used by both source and cache keys."""

    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    if architecture is None:
        validate_nequip_oam_data(contract)
        architecture = resolve_oam_architecture(contract)
    else:
        architecture = validate_oam_architecture(architecture)
    layers = []
    for index, layer in enumerate(contract["layers"]):
        product = layer["edge_tensor_product"]
        layers.append(
            {
                "layer": index,
                "source_dimension": irreps_dimension(
                    _irrep_blocks(layer["node_irreps_in"])
                ),
                "edge_dimension": irreps_dimension(_irrep_blocks(layer["edge_irreps"])),
                "weight_dimension": int(product["weight_numel"]),
                "output_dimension": irreps_dimension(
                    _irrep_blocks(layer["tp_output_irreps"])
                ),
                "instruction_count": len(product["instructions"]),
            }
        )
    semantic = oam_tensor_product_architecture_fingerprint(architecture)
    structure = canonical_digest(
        {
            "layers": [
                {
                    "source_dimension": row["source_dimension"],
                    "edge_dimension": row["edge_dimension"],
                    "weight_dimension": row["weight_dimension"],
                    "output_dimension": row["output_dimension"],
                    "instruction_count": row["instruction_count"],
                }
                for row in layers
            ]
        }
    )
    generation = canonical_digest(
        {
            "jit_generation_version": JIT_GENERATION_VERSION,
            "abi": ABI_TAG,
            "precision": precision,
        }
    )
    artifact_id = "oam-host-v1-{}-{}".format(
        precision, semantic.removeprefix("sha256:")[:16]
    )
    return {
        "abi": ABI_TAG,
        "abi_version": ABI_VERSION,
        "artifact_id": artifact_id,
        "architecture": architecture,
        "generation_fingerprint": generation,
        "semantic_fingerprint": semantic,
        "structure_fingerprint": structure,
        "precision": precision,
        "scalar_size": 4 if precision == "float32" else 8,
        "scalar_type": "float" if precision == "float32" else "double",
        "layer_count": len(layers),
        "layers": layers,
        "required_capabilities": FORWARD_CAPABILITY | REVERSE_CAPABILITY,
    }


def _render_instruction(
    layer: Mapping[str, Any],
    instruction: Mapping[str, Any],
    weight_offset: int,
    symbol_id: int,
    scalar: str = "double",
) -> str:
    product = layer["edge_tensor_product"]
    in1 = _irrep_blocks(product["irreps_in1"])[int(instruction["i_in1"])]
    in2 = _irrep_blocks(product["irreps_in2"])[int(instruction["i_in2"])]
    out = _irrep_blocks(product["irreps_out"])[int(instruction["i_out"])]
    o1 = block_offsets(_irrep_blocks(product["irreps_in1"]))[int(instruction["i_in1"])]
    o2 = block_offsets(_irrep_blocks(product["irreps_in2"]))[int(instruction["i_in2"])]
    oo = block_offsets(_irrep_blocks(product["irreps_out"]))[int(instruction["i_out"])]
    m1, m2, mo = (
        int(in1["multiplicity"]),
        int(in2["multiplicity"]),
        int(out["multiplicity"]),
    )
    d1, d2, do = 2 * int(in1["l"]) + 1, 2 * int(in2["l"]) + 1, 2 * int(out["l"]) + 1
    mode = instruction["connection_mode"]
    path = [float(v) for v in instruction["wigner_3j"]["values"]]
    factor = float(instruction["path_weight"])
    c_extent = mo if mode == "uvw" else 1
    out_m = "c" if mode == "uvw" else "a"
    if not instruction["has_weight"]:
        weight_expr = f"{scalar}(1)"
    elif mode == "uvu":
        weight_expr = f"weights[{weight_offset} + a*{m2} + b]"
    elif mode == "uuu":
        weight_expr = f"weights[{weight_offset} + a]"
    else:
        weight_expr = f"weights[{weight_offset} + (a*{m2}+b)*{mo}+c]"
    lines = ["{ int a = 0; int b = 0; int c = 0;"]
    lines.append(
        f"for (a=0; a<{m1}; ++a) for (b=0; b<{m2}; ++b) for (c=0; c<{c_extent}; ++c) {{"
    )
    if mode == "uuu":
        lines.append("if (a != b) continue;")
    # Only nonzero Wigner entries contribute; bake them with literal
    # component indices so each output component stays a register reduction.
    for z in range(do):
        terms = [
            (x, y, path[(x * d2 + y) * do + z])
            for x in range(d1)
            for y in range(d2)
            if path[(x * d2 + y) * do + z] != 0.0
        ]
        if not terms:
            continue
        lines.append(f"{scalar} sum{symbol_id}_{z} = {scalar}(0);")
        for x, y, value in terms:
            literal = scalar_literal(factor * value, scalar)
            lines.append(
                f"sum{symbol_id}_{z} += src[{o1} + {x}*{m1} + a]"
                f" * edge[{o2} + b*{d2} + {y}] * {literal};"
            )
        lines.append(
            f"out[{oo} + {z}*{mo} + {out_m}] += sum{symbol_id}_{z} * {weight_expr};"
        )
    lines.append("}}")
    return "\n".join(lines)


def _render_instruction_reverse(
    layer: Mapping[str, Any],
    instruction: Mapping[str, Any],
    weight_offset: int,
    symbol_id: int,
    scalar: str = "double",
    names: Mapping[str, str] | None = None,
) -> str:
    """Render one instruction's lane-major adjoint block.

    ``names`` overrides the accumulator and operand identifiers (``out_adj``,
    ``src``, ``ds``, ``de``, ``dw``) so the receiver-owned reverse can reuse
    this body against its own buffers.  The edge operand, Wigner path, and
    weight tables keep their definition (node-major where per-element) order.
    """
    out_adj_name = "out_adj" if names is None else names["out_adj"]
    src_name = "src" if names is None else names["src"]
    ds_name = "ds" if names is None else names["ds"]
    de_name = "de" if names is None else names["de"]
    dw_name = "dw" if names is None else names["dw"]
    product = layer["edge_tensor_product"]
    in1 = _irrep_blocks(product["irreps_in1"])[int(instruction["i_in1"])]
    in2 = _irrep_blocks(product["irreps_in2"])[int(instruction["i_in2"])]
    out = _irrep_blocks(product["irreps_out"])[int(instruction["i_out"])]
    o1 = block_offsets(_irrep_blocks(product["irreps_in1"]))[int(instruction["i_in1"])]
    o2 = block_offsets(_irrep_blocks(product["irreps_in2"]))[int(instruction["i_in2"])]
    oo = block_offsets(_irrep_blocks(product["irreps_out"]))[int(instruction["i_out"])]
    m1, m2, mo = (
        int(in1["multiplicity"]),
        int(in2["multiplicity"]),
        int(out["multiplicity"]),
    )
    d1, d2, do = 2 * int(in1["l"]) + 1, 2 * int(in2["l"]) + 1, 2 * int(out["l"]) + 1
    mode = instruction["connection_mode"]
    path = [float(v) for v in instruction["wigner_3j"]["values"]]
    factor = float(instruction["path_weight"])
    c_extent = mo if mode == "uvw" else 1
    out_m = "c" if mode == "uvw" else "a"
    if not instruction["has_weight"]:
        weight_expr = f"{scalar}(1)"
    elif mode == "uvu":
        weight_expr = f"weights[{weight_offset} + a*{m2} + b]"
    elif mode == "uuu":
        weight_expr = f"weights[{weight_offset} + a]"
    else:
        weight_expr = f"weights[{weight_offset} + (a*{m2}+b)*{mo}+c]"
    # Only nonzero Wigner entries contribute.  Bake each with literal
    # component indices so the ds/de row accumulators stay in registers and
    # no zero terms are evaluated (OAM edge Wigner tensors are ~20% dense).
    terms = [
        (x, y, z, factor * path[(x * d2 + y) * do + z])
        for x in range(d1)
        for y in range(d2)
        for z in range(do)
        if path[(x * d2 + y) * do + z] != 0.0
    ]
    has_weight = bool(instruction["has_weight"])
    lines = [
        f"for (int a=0; a<{m1}; ++a) {{",
        f"    {scalar} ds_row{symbol_id}[{d1}];",
        f"    for (int rx=0; rx<{d1}; ++rx) ds_row{symbol_id}[rx] = {scalar}(0);",
        f"    for (int b=0; b<{m2}; ++b) {{",
    ]
    if mode == "uuu":
        lines.append("        if (a != b) continue;")
    lines.extend(
        [
            f"        {scalar} de_row{symbol_id}[{d2}];",
            f"        for (int ry=0; ry<{d2}; ++ry)"
            f" de_row{symbol_id}[ry] = {scalar}(0);",
            f"        for (int c=0; c<{c_extent}; ++c) {{",
            f"            const {scalar} w_v{symbol_id} = {weight_expr};",
        ]
    )
    if has_weight:
        lines.append(f"            {scalar} dw_acc{symbol_id} = {scalar}(0);")
    for x, y, z, value in terms:
        literal = scalar_literal(value, scalar)
        lines.append(
            f"            {{ const {scalar} coeff{symbol_id} ="
            f" {out_adj_name}[{oo} + {z}*{mo} + {out_m}] * {literal};"
        )
        lines.append(
            f"              ds_row{symbol_id}[{x}] += coeff{symbol_id}"
            f" * edge[{o2} + b*{d2} + {y}] * w_v{symbol_id};"
        )
        lines.append(
            f"              de_row{symbol_id}[{y}] += coeff{symbol_id}"
            f" * {src_name}[{o1} + {x}*{m1} + a] * w_v{symbol_id};"
        )
        if has_weight:
            lines.append(
                f"              dw_acc{symbol_id} += coeff{symbol_id}"
                f" * {src_name}[{o1} + {x}*{m1} + a]"
                f" * edge[{o2} + b*{d2} + {y}]; }}"
            )
        else:
            lines.append("            }")
    if mode == "uvw":
        dw_index = f"{weight_offset} + (a*{m2}+b)*{mo}+c"
    elif mode == "uvu":
        dw_index = f"{weight_offset} + a*{m2} + b"
    else:
        dw_index = f"{weight_offset} + a"
    if has_weight:
        lines.append(f"            {dw_name}[{dw_index}] += dw_acc{symbol_id};")
    lines.extend(
        [
            "        }",
            f"        for (int y=0; y<{d2}; ++y)"
            f" {de_name}[{o2} + b*{d2} + y] += de_row{symbol_id}[y];",
            "    }",
            f"    for (int x=0; x<{d1}; ++x)"
            f" {ds_name}[{o1} + x*{m1} + a] += ds_row{symbol_id}[x];",
            "}",
        ]
    )
    return "\n".join(lines)


def render_oam_host_plugin(
    contract: Mapping[str, Any], precision: str = "float64"
) -> str:
    """Render a self-contained C++ plugin with one callback per OAM layer."""

    metadata = oam_jit_metadata(contract, precision)
    scalar = metadata["scalar_type"]
    forward_functions: list[str] = []
    reverse_functions: list[str] = []
    layer_rows: list[str] = []
    for index, layer in enumerate(contract["layers"]):
        product = layer["edge_tensor_product"]
        source_dim = metadata["layers"][index]["source_dimension"]
        edge_dim = metadata["layers"][index]["edge_dimension"]
        weight_dim = metadata["layers"][index]["weight_dimension"]
        output_dim = metadata["layers"][index]["output_dimension"]
        output_mask = layer["edge_tensor_product"]["output_mask"]["values"]
        lane_index = lane_column_permutation(_irrep_blocks(product["irreps_out"]))
        weight_offset = 0
        body = []
        for instruction_index, instruction in enumerate(product["instructions"]):
            body.append(
                _render_instruction(
                    layer, instruction, weight_offset, instruction_index, scalar
                )
            )
            if instruction["has_weight"]:
                weight_offset += math.prod(
                    int(extent) for extent in instruction["path_shape"]
                )
        forward_functions.append(
            f"void forward_{index}(const SymmetrixJitOAMHostForwardArgsV1* a, int32_t e) {{\n"
            f"const auto* src = static_cast<const {scalar}*>(a->source_features) + static_cast<size_t>(a->source_indices[e])*{source_dim};\n"
            f"const auto* edge = static_cast<const {scalar}*>(a->edge_features) + static_cast<size_t>(e)*{edge_dim};\n"
            f"const auto* weights = static_cast<const {scalar}*>(a->edge_weights) + static_cast<size_t>(e)*{weight_dim};\n"
            f"auto* out = static_cast<{scalar}*>(a->edge_messages) + static_cast<size_t>(e)*{output_dim};\n"
            f"static constexpr double output_mask[] = {{{','.join(str(float(v)) for v in output_mask)}}};\n"
            f"static constexpr int output_lane_index[] = {{{','.join(str(v) for v in lane_index)}}};\n"
            f"for (int i=0;i<{output_dim};++i) out[i]=0;\n"
            + "\n".join(body)
            + "\n"
            + f"for (int i=0;i<{output_dim};++i) out[output_lane_index[i]] *= static_cast<{scalar}>(output_mask[i]);\n"
            + "\n}\n"
        )
        reverse_body = []
        reverse_weight_offset = 0
        for instruction_index, instruction in enumerate(product["instructions"]):
            reverse_body.append(
                _render_instruction_reverse(
                    layer, instruction, reverse_weight_offset, instruction_index, scalar
                )
            )
            if instruction["has_weight"]:
                reverse_weight_offset += math.prod(
                    int(extent) for extent in instruction["path_shape"]
                )
        reverse_functions.append(
            f"void reverse_{index}(const SymmetrixJitOAMHostReverseArgsV1* a, int32_t e) {{\n"
            f"const auto* src = static_cast<const {scalar}*>(a->source_features) + static_cast<size_t>(a->source_indices[e])*{source_dim};\n"
            f"const auto* edge = static_cast<const {scalar}*>(a->edge_features) + static_cast<size_t>(e)*{edge_dim};\n"
            f"auto* ds = static_cast<{scalar}*>(a->source_adjoint) + static_cast<size_t>(e)*{source_dim};\n"
            f"auto* de = static_cast<{scalar}*>(a->edge_feature_adjoint) + static_cast<size_t>(e)*{edge_dim};\n"
            f"auto* dw = static_cast<{scalar}*>(a->edge_weight_adjoint) + static_cast<size_t>(e)*{weight_dim};\n"
            f"for (int i=0;i<{source_dim};++i) ds[i]=0; for (int i=0;i<{edge_dim};++i) de[i]=0; for (int i=0;i<{weight_dim};++i) dw[i]=0;\n"
            f"const auto* weights = static_cast<const {scalar}*>(a->edge_weights) + static_cast<size_t>(e)*{weight_dim};\n"
            f"const auto* out_adj = static_cast<const {scalar}*>(a->edge_output_adjoint) + static_cast<size_t>(e)*{output_dim};\n"
            + "\n".join(reverse_body)
            + "\n}\n"
        )
        layer_rows.append(
            f"{{sizeof(SymmetrixJitOAMHostLayerV1), {index}u, 0u, 0u, {source_dim}, {edge_dim}, {weight_dim}, {output_dim}, {len(product['instructions'])}}}"
        )
    forward_array = ",".join(f"&forward_{i}" for i in range(len(contract["layers"])))
    reverse_array = ",".join(f"&reverse_{i}" for i in range(len(contract["layers"])))
    return f"""// Generated OAM host tensor-product plugin.\n{_ABI_HEADER}\n#include <cstddef>\n#include <cstdint>\nnamespace {{\n{"".join(forward_functions)}\n{"".join(reverse_functions)}\nstatic const SymmetrixJitOAMHostLayerV1 layers[] = {{{",".join(layer_rows)}}};\nstatic const SymmetrixJitOAMHostForwardOwnerV1 forward_owners[] = {{{forward_array}}};\nstatic const SymmetrixJitOAMHostReverseOwnerV1 reverse_owners[] = {{{reverse_array}}};\nstatic const SymmetrixJitOAMHostPluginV1 plugin = {{\nSYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION, sizeof(SymmetrixJitOAMHostPluginV1), sizeof(void*), SYMMETRIX_JIT_OAM_HOST_PLUGIN_BYTE_ORDER,\nSYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1 | SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1, {len(contract["layers"])}, {metadata["scalar_size"]}, 0u,\nSYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG, {cpp_string_literal(metadata["artifact_id"])}, {cpp_string_literal(metadata["generation_fingerprint"])}, {cpp_string_literal(metadata["semantic_fingerprint"])}, {cpp_string_literal(metadata["structure_fingerprint"])}, layers, forward_owners, reverse_owners}};\n}}\nextern \"C\" SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT const SymmetrixJitOAMHostPluginV1* symmetrix_jit_oam_host_plugin_query_v1() {{ return &plugin; }}\n"""


__all__ = [
    "OAM_HOST_ARTIFACT_ENVELOPE",
    "oam_jit_metadata",
    "render_oam_host_plugin",
]
