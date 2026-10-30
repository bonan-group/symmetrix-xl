"""Full-message receiver-forward/source-reverse OAM host generation (ABI v7).

The generated plugin owns one receiver or receiver/channel-tile work item per
forward callback and one source or source/channel-tile per reverse callback.
Neither callback materializes per-edge model intermediates. Cubic spline values
and analytic derivatives are evaluated inline, so no radial MLP matrix
multiplication participates in inference.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .jit import JIT_GENERATION_VERSION, ArtifactEnvelope
from .jit_codegen_common import (
    block_offsets,
    canonical_digest,
    cpp_string_literal,
    harmonic_indices,
    irreps_dimension,
    lane_column_permutation,
    render_host_spline_block,
    scalar_literal,
)
from .nequip_schema import (
    _canonical,
    validate_nequip_oam_data,
)
from .oam_architecture import (
    oam_tensor_product_architecture_fingerprint,
    resolve_oam_architecture,
    validate_oam_architecture,
)
from .oam_jit_codegen import (
    _irrep_blocks,
    _render_instruction,
    _render_instruction_reverse,
)

ABI_TAG = "symmetrix.jit.oam.host-plugin/7"
ABI_VERSION = 7
RECEIVER_FORWARD_CAPABILITY = 1
SOURCE_REVERSE_CAPABILITY = 2
# This receiver generation presents lane-major node tensors to the callbacks and
# applies the layer output mask and normalization inside the generated forward
# callback.  The reverse callback receives the layer adjoint after the runtime
# has applied those same factors to it, so the runtime must not apply them again
# to the forward output.
LANE_MAJOR_CAPABILITY = 1 << 3
GENERATOR = "symmetrix.jit.oam-receiver-v7"
OAM_HOST_FLOATING_POINT_POLICIES = ("strict", "fast")
# Full-message-only generation. Dual-variant v7 artifacts have a different
# generation fingerprint and are rejected by the manifest admission contract.
RECEIVER_CODEGEN_VERSION = 13
# The identity envelope the receiver generation publishes.  It is the same
# shape as the edge-owned host envelope with the receiver capability word.
OAM_RECEIVER_ARTIFACT_ENVELOPE = ArtifactEnvelope(
    "OAM receiver",
    identity_fingerprint="generation_fingerprint",
    capabilities_key="required_capabilities",
)
_ABI_HEADER = (
    Path(__file__)
    .with_name("oam_jit_host_plugin_abi.h")
    .read_text(encoding="ascii")
    .removeprefix("#pragma once\n\n")
)


@dataclass(frozen=True)
class OAMHostGenerationPolicy:
    """Identity-bound host owner and channel schedule.

    ``whole_owner`` is the retained control. ``channel_tiled`` is an explicit
    candidate for full-message forward/reverse: each receiver/source channel
    tile is a separate work item with disjoint lane ownership.
    """

    owner_schedule: str = "whole_owner"
    channel_tile: int = 1
    scalar_width: int | None = None
    vector_policy: str = "host-simd-lane-major-v1"
    required_isa: str = "host-target-bound-v1"
    full_reverse_reduction_policy: str = "owner-local-deterministic-v1"
    resource_policy: str = "compiler-managed-v1"

    def __post_init__(self) -> None:
        if self.owner_schedule not in ("whole_owner", "channel_tiled"):
            raise ValueError("host owner schedule is unsupported")
        if self.channel_tile <= 0 or self.channel_tile > 2**31 - 1:
            raise ValueError(
                "host channel tile must fit a positive signed 32-bit value"
            )
        if self.owner_schedule == "whole_owner" and self.channel_tile != 1:
            raise ValueError("whole-owner host schedule requires channel tile 1")
        if self.scalar_width is not None and self.scalar_width not in (4, 8):
            raise ValueError("host scalar width must be 4 or 8 bytes")
        if self.owner_schedule == "channel_tiled":
            if self.vector_policy == "host-simd-lane-major-v1":
                object.__setattr__(self, "vector_policy", "channel-tiled-lane-major-v1")
            if self.resource_policy == "compiler-managed-v1":
                object.__setattr__(self, "resource_policy", "bounded-owner-stack-v1")
        expected_vector = (
            "channel-tiled-lane-major-v1"
            if self.owner_schedule == "channel_tiled"
            else "host-simd-lane-major-v1"
        )
        if self.vector_policy != expected_vector:
            raise ValueError("host vector policy does not implement the owner schedule")
        expected_resource = (
            "bounded-owner-stack-v1"
            if self.owner_schedule == "channel_tiled"
            else "compiler-managed-v1"
        )
        if self.resource_policy != expected_resource:
            raise ValueError(
                "host resource policy does not implement the owner schedule"
            )
        if self.required_isa != "host-target-bound-v1":
            raise ValueError("host required ISA must be bound to the artifact target")
        if self.full_reverse_reduction_policy != "owner-local-deterministic-v1":
            raise ValueError("host full reverse reduction policy is unsupported")


def _normalized_host_policy(
    policy: OAMHostGenerationPolicy | None, precision: str
) -> OAMHostGenerationPolicy:
    if policy is None:
        policy = OAMHostGenerationPolicy()
    if not isinstance(policy, OAMHostGenerationPolicy):
        raise TypeError("generation policy must be OAMHostGenerationPolicy or None")
    scalar_size = 4 if precision == "float32" else 8
    if policy.scalar_width is None:
        return replace(policy, scalar_width=scalar_size)
    if policy.scalar_width != scalar_size:
        raise ValueError("host scalar width must exactly match the scalar type")
    return policy


def canonical_oam_host_floating_point_policy(value: str) -> str:
    """Return the identity spelling for host floating-point semantics."""

    if value not in OAM_HOST_FLOATING_POINT_POLICIES:
        raise ValueError("OAM host floating-point policy must be 'strict' or 'fast'")
    return value


def _pre_linear_width(layer: Mapping[str, Any]) -> int:
    return irreps_dimension(_irrep_blocks(layer["pre_linear"]["irreps_out"]))


def _normalized_post_linear(layer: Mapping[str, Any]) -> dict[str, Any]:
    """Split coalesced linear blocks to the evaluator's exact layer layouts."""

    linear = layer["post_linear"]
    current_in = _irrep_blocks(linear["irreps_in"])
    current_out = _irrep_blocks(linear["irreps_out"])
    target_in = _irrep_blocks(layer["tp_output_irreps"])
    target_out = _irrep_blocks(layer["node_irreps_pre_gate"])
    if current_in == target_in and current_out == target_out:
        return copy.deepcopy(linear)

    def cover(coarse, fine):
        result = []
        cursor = 0
        for block in coarse:
            selected = []
            multiplicity = 0
            while multiplicity < int(block["multiplicity"]):
                if cursor >= len(fine):
                    raise ValueError(
                        "post-linear irrep layout does not cover the layer"
                    )
                candidate = fine[cursor]
                if (candidate["l"], candidate["parity"]) != (
                    block["l"],
                    block["parity"],
                ):
                    raise ValueError(
                        "post-linear irrep layout does not cover the layer"
                    )
                selected.append(cursor)
                multiplicity += int(candidate["multiplicity"])
                cursor += 1
            if multiplicity != int(block["multiplicity"]):
                raise ValueError("post-linear irrep multiplicity is inconsistent")
            result.append(selected)
        if cursor != len(fine):
            raise ValueError("post-linear irrep layout does not cover the layer")
        return result

    cover_in = cover(current_in, target_in)
    cover_out = cover(current_out, target_out)
    source_weights = linear["weight"]["values"]
    instructions = []
    weights = []
    cursor = 0
    for instruction in linear["instructions"]:
        source_block = int(instruction["i_in"])
        output_block = int(instruction["i_out"])
        rows, columns = (int(v) for v in instruction["path_shape"])
        row = 0
        for fine_in in cover_in[source_block]:
            fine_rows = int(target_in[fine_in]["multiplicity"])
            column = 0
            for fine_out in cover_out[output_block]:
                fine_columns = int(target_out[fine_out]["multiplicity"])
                for r in range(fine_rows):
                    for c in range(fine_columns):
                        weights.append(
                            source_weights[cursor + (row + r) * columns + column + c]
                        )
                instructions.append(
                    {
                        "i_in": fine_in,
                        "i_out": fine_out,
                        "path_shape": [fine_rows, fine_columns],
                        "path_weight": instruction.get("path_weight", 1.0),
                    }
                )
                column += fine_columns
            row += fine_rows
        cursor += rows * columns
    result = copy.deepcopy(linear)
    result["irreps_in"] = layer["tp_output_irreps"]
    result["irreps_out"] = layer["node_irreps_pre_gate"]
    result["instructions"] = instructions
    result["weight"] = {"shape": [len(weights)], "values": weights}
    return result


def post_linear_structure(layer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the post-linear topology needed to reconstruct a model contract."""

    post = _normalized_post_linear(layer)
    return _canonical(
        {
            "irreps_in": post["irreps_in"],
            "irreps_out": post["irreps_out"],
            "instructions": [
                {
                    key: instruction[key]
                    for key in ("i_in", "i_out", "path_shape", "path_weight")
                }
                for instruction in post["instructions"]
            ],
            "output_mask": post["output_mask"],
            "weight_shape": post["weight"]["shape"],
            "bias_present": bool(post["bias"]["values"]),
            "bias_shape": post["bias"]["shape"],
        }
    )


def oam_receiver_jit_metadata(
    contract: Mapping[str, Any],
    precision: str = "float64",
    *,
    architecture: Mapping[str, Any] | None = None,
    host_target: str | None = None,
    generation_policy: OAMHostGenerationPolicy | None = None,
    floating_point_policy: str = "strict",
) -> dict[str, Any]:
    """Return ABI v7 model metadata used by both source and cache keys."""

    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    policy = _normalized_host_policy(generation_policy, precision)
    floating_point_policy = canonical_oam_host_floating_point_policy(
        floating_point_policy
    )
    policy_data = asdict(policy)
    if not isinstance(host_target, str) or not host_target.startswith(
        ("host-isa-", "host-native-sha256-", "host-sha256-")
    ):
        raise ValueError("OAM ABI v7 generation requires a canonical host target")
    if architecture is None:
        validate_nequip_oam_data(contract)
        architecture = resolve_oam_architecture(contract)
    else:
        architecture = validate_oam_architecture(architecture)
    layers = []
    for index, layer in enumerate(contract["layers"]):
        product = layer["edge_tensor_product"]
        output_dimension = irreps_dimension(_irrep_blocks(layer["tp_output_irreps"]))
        layers.append(
            {
                "layer": index,
                "source_dimension": _pre_linear_width(layer),
                "edge_dimension": irreps_dimension(_irrep_blocks(layer["edge_irreps"])),
                "weight_dimension": int(product["weight_numel"]),
                "output_dimension": output_dimension,
                "instruction_count": len(product["instructions"]),
                "target_dimension": output_dimension,
                "channel_tile": policy.channel_tile,
            }
        )
    contract_fingerprint = oam_tensor_product_architecture_fingerprint(architecture)
    semantic = canonical_digest({"architecture": contract_fingerprint})
    structure = canonical_digest(
        {
            "layers": [
                {
                    "source_dimension": row["source_dimension"],
                    "edge_dimension": row["edge_dimension"],
                    "weight_dimension": row["weight_dimension"],
                    "output_dimension": row["output_dimension"],
                    "instruction_count": row["instruction_count"],
                    "target_dimension": row["target_dimension"],
                }
                for row in layers
            ]
        }
    )
    generation = canonical_digest(
        {
            "jit_generation_version": JIT_GENERATION_VERSION,
            "receiver_codegen_version": RECEIVER_CODEGEN_VERSION,
            "abi": ABI_TAG,
            "precision": precision,
            "execution_variants": ["full_message"],
            "backend": "host",
            "ownership": {
                "forward": "receiver",
                "reverse": "source",
            },
            "full_reverse_owner_schedule": (
                "source-channel-tiled-owner-v1"
                if policy.owner_schedule == "channel_tiled"
                else "source-owner-v1"
            ),
            "generation_policy": policy_data,
            "tiling_policy": (
                "channel-tiled-owner-v1"
                if policy.owner_schedule == "channel_tiled"
                else "whole-owner-v1"
            ),
            "vector_policy": policy.vector_policy,
            "shared_memory_policy": "none",
            "register_budget_policy": policy.resource_policy,
            "required_isa": policy.required_isa,
            "scalar_width": policy.scalar_width,
            "full_reverse_reduction_policy": policy.full_reverse_reduction_policy,
            "host_target": host_target,
            "floating_point_policy": floating_point_policy,
        }
    )
    schedule_identity = canonical_digest(
        {
            "ownership": {
                "forward": "receiver",
                "reverse": "source",
            },
            "full_reverse_owner_schedule": (
                "source-channel-tiled-owner-v1"
                if policy.owner_schedule == "channel_tiled"
                else "source-owner-v1"
            ),
            "generation_policy": policy_data,
            "tiling_policy": (
                "channel-tiled-owner-v1"
                if policy.owner_schedule == "channel_tiled"
                else "whole-owner-v1"
            ),
            "vector_policy": policy.vector_policy,
        }
    )
    artifact_digest = canonical_digest(
        {"semantic": semantic, "structure": structure, "generation": generation}
    )
    artifact_id = "oam-receiver-v7-{}-{}".format(
        precision, artifact_digest.removeprefix("sha256:")[:16]
    )
    return {
        "abi": ABI_TAG,
        "abi_version": ABI_VERSION,
        "generator": GENERATOR,
        "artifact_id": artifact_id,
        "architecture": architecture,
        "contract_fingerprint": contract_fingerprint,
        "generation_fingerprint": generation,
        "semantic_fingerprint": semantic,
        "structure_fingerprint": structure,
        "precision": precision,
        "scalar_size": 4 if precision == "float32" else 8,
        "scalar_type": "float" if precision == "float32" else "double",
        "scalar_width": policy.scalar_width,
        "channel_tile": policy.channel_tile,
        "layer_count": len(layers),
        "layers": layers,
        "execution_variants": ["full_message"],
        "backend": "host",
        "ownership": {
            "forward": "receiver",
            "reverse": "source",
        },
        "generation_policy": policy_data,
        "tiling_policy": (
            "channel-tiled-owner-v1"
            if policy.owner_schedule == "channel_tiled"
            else "whole-owner-v1"
        ),
        "vector_policy": policy.vector_policy,
        "shared_memory_policy": "none",
        "register_budget_policy": policy.resource_policy,
        "required_isa": host_target,
        "full_reverse_reduction_policy": policy.full_reverse_reduction_policy,
        "schedule_qualification": (
            "candidate-unqualified-explicit"
            if policy.owner_schedule == "channel_tiled"
            else "conservative-control"
        ),
        "direct_geometry_policy": "evaluator-harmonic-gradient-reduction-v1",
        "forward_owner_schedule": (
            "receiver-channel-tiled-owner-v1"
            if policy.owner_schedule == "channel_tiled"
            else "receiver-owner-v1"
        ),
        "full_reverse_owner_schedule": (
            "source-channel-tiled-owner-v1"
            if policy.owner_schedule == "channel_tiled"
            else "source-owner-v1"
        ),
        "host_target": host_target,
        "floating_point_policy": floating_point_policy,
        "schedule_identity": schedule_identity,
        "required_capabilities": RECEIVER_FORWARD_CAPABILITY
        | SOURCE_REVERSE_CAPABILITY
        | LANE_MAJOR_CAPABILITY,
    }


def _render_receiver_forward(
    layer: Mapping[str, Any],
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
) -> str:
    scalar = metadata["scalar_type"]
    channel_tile = int(metadata["channel_tile"])
    channel_tiled = metadata["tiling_policy"] == "channel-tiled-owner-v1"
    product = layer["edge_tensor_product"]
    source_dimension = int(row["source_dimension"])
    edge_dimension = int(row["edge_dimension"])
    weight_dimension = int(row["weight_dimension"])
    output_dimension = int(row["output_dimension"])
    output_mask = product["output_mask"]["values"]
    lane_index = lane_column_permutation(_irrep_blocks(layer["tp_output_irreps"]))
    if len(lane_index) != output_dimension:
        raise ValueError("tensor-product output irrep layout is inconsistent")
    harmonic_map = harmonic_indices(_irrep_blocks(layer["edge_irreps"]))
    if len(harmonic_map) != edge_dimension:
        raise ValueError("edge harmonic layout is inconsistent with the edge irreps")
    weight_offset = 0
    body: list[str] = []
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
    edge_loads = [
        f"            harmonics[static_cast<int64_t>(directed_edge)"
        f" * a->edge_harmonic_count + {harmonic}],"
        for harmonic in harmonic_map
    ]
    tile_count = (
        (output_dimension + channel_tile - 1) // channel_tile if channel_tiled else 1
    )
    lines = [
        f"void oam_receiver_forward_{index}("
        f"const SymmetrixJitOAMHostReceiverForwardArgsV3* a, int32_t work_item)",
        "{",
        f"    static constexpr int channel_tile = {channel_tile};",
        f"    static constexpr int channel_tile_count = {tile_count};",
        "    const int receiver = work_item / channel_tile_count;",
        "    const int channel_tile_index = work_item % channel_tile_count;",
        (
            "    const int channel_begin = channel_tile_index * channel_tile;"
            if channel_tiled
            else "    const int channel_begin = 0;"
        ),
        (
            f"    const int bounded_channel_end = channel_begin + channel_tile < {output_dimension} ? channel_begin + channel_tile : {output_dimension};"
            if channel_tiled
            else f"    const int bounded_channel_end = {output_dimension};"
        ),
        f"    const auto* source_values = static_cast<const {scalar}*>(a->source_node_values);",
        f"    const auto* harmonics = static_cast<const {scalar}*>(a->edge_harmonics);",
        f"    const auto* coefficients = static_cast<const {scalar}*>(a->spline_coefficients);",
        f"    auto* messages = static_cast<{scalar}*>(a->node_messages);",
        "    const auto* sources = a->source_indices;",
        "    const int64_t first_receiver_edge = a->receiver_offsets[receiver];",
        "    const int64_t next_receiver_edge = a->receiver_offsets[receiver + 1];",
        "    const int64_t first = first_receiver_edge > a->first_edge",
        "        ? first_receiver_edge : a->first_edge;",
        "    const int64_t last_input = a->first_edge + a->samples;",
        "    const int64_t last = next_receiver_edge < last_input",
        "        ? next_receiver_edge : last_input;",
        f"    static constexpr double output_mask[] = "
        f"{{{','.join(str(float(v)) for v in output_mask)}}};",
        f"    static constexpr int output_lane_index[] = "
        f"{{{','.join(str(v) for v in lane_index)}}};",
        f"    {scalar} tile[{output_dimension}];",
        f"    for (int index = 0; index < {output_dimension}; ++index)"
        f" tile[index] = {scalar}(0);",
        "    for (int64_t directed_edge = first; directed_edge < last; ++directed_edge) {",
        render_host_spline_block(scalar, weight_dimension, derivatives=False),
        "        const auto* src = source_values",
        f"            + static_cast<int64_t>(sources[directed_edge]) * {source_dimension};",
        f"        const {scalar} edge[{edge_dimension}] = {{",
        *edge_loads,
        "        };",
        f"        {scalar}* out = tile;",
        "\n".join(body),
        "    }",
        f"    const {scalar} normalization ="
        f" static_cast<{scalar}>(a->normalization_factor);",
    ]
    lines.extend(
        [
            f"    auto* destination = messages + static_cast<int64_t>(receiver) * {output_dimension};",
            "    for (int index = channel_begin; index < bounded_channel_end; ++index) {",
            "        const int lane = output_lane_index[index];",
            "        destination[lane] = tile[lane] * normalization",
            f"            * static_cast<{scalar}>(output_mask[index]);",
            "    }",
        ]
    )
    lines.extend(["}", ""])
    return "\n".join(lines)


def _render_source_reverse(
    layer: Mapping[str, Any],
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
) -> str:
    """Emit the lane-major source reverse callback.

    The callback reads the lane-major source values and target adjoint from
    the regular args pointers and writes its own source adjoint row directly
    (each directed edge appears exactly once in the source CSR).  Every
    instruction takes the wide register-vector form when its angular width is
    at least five and the scalar register-row form otherwise; both forms read
    and write lane-major node tensors.
    """

    scalar = metadata["scalar_type"]
    channel_tile = int(metadata["channel_tile"])
    channel_tiled = metadata["tiling_policy"] == "channel-tiled-owner-v1"
    product = layer["edge_tensor_product"]
    source_dimension = int(row["source_dimension"])
    target_dimension = int(row["target_dimension"])
    weight_dimension = int(row["weight_dimension"])
    edge_dimension = int(row["edge_dimension"])
    in1_blocks = _irrep_blocks(product["irreps_in1"])
    in1_offsets = block_offsets(_irrep_blocks(product["irreps_in1"]))
    in2_blocks = _irrep_blocks(product["irreps_in2"])
    in2_offsets = block_offsets(_irrep_blocks(product["irreps_in2"]))
    out_blocks = _irrep_blocks(product["irreps_out"])
    out_offsets = block_offsets(_irrep_blocks(product["irreps_out"]))
    harmonic_map = harmonic_indices(_irrep_blocks(layer["edge_irreps"]))
    if len(harmonic_map) != edge_dimension:
        raise ValueError("edge harmonic layout is inconsistent with the edge irreps")
    bodies: list[str] = []
    weight_offset = 0
    for instruction_index, instruction in enumerate(product["instructions"]):
        in1 = in1_blocks[int(instruction["i_in1"])]
        in2 = in2_blocks[int(instruction["i_in2"])]
        out = out_blocks[int(instruction["i_out"])]
        o1 = in1_offsets[int(instruction["i_in1"])]
        o2 = in2_offsets[int(instruction["i_in2"])]
        oo = out_offsets[int(instruction["i_out"])]
        m1 = int(in1["multiplicity"])
        m2 = int(in2["multiplicity"])
        d1, d2 = 2 * int(in1["l"]) + 1, 2 * int(in2["l"]) + 1
        do = 2 * int(out["l"]) + 1
        mo = int(out["multiplicity"])
        mode = instruction["connection_mode"]
        has_weight = bool(instruction["has_weight"])
        factor = float(instruction["path_weight"])
        path = [float(v) for v in instruction["wigner_3j"]["values"]]
        terms = [
            (x, y, z, factor * path[(x * d2 + y) * do + z])
            for x in range(d1)
            for y in range(d2)
            for z in range(do)
            if path[(x * d2 + y) * do + z] != 0.0
        ]
        symbol = instruction_index
        if has_weight and mode == "uvu" and m2 == 1 and max(d1, d2, do) < 5:
            # Narrow coupling: the scalar register-row form stays unit-stride
            # over the lane-major source multiplicity axis.
            narrow = [
                f"        for (int a = 0; a < {m1}; ++a) {{",
                f"            {scalar} ds_row_{symbol}[{d1}];",
                f"            for (int i = 0; i < {d1}; ++i)"
                f" ds_row_{symbol}[i] = {scalar}(0);",
                f"            {scalar} de_row_{symbol}[{d2}];",
                f"            for (int i = 0; i < {d2}; ++i)"
                f" de_row_{symbol}[i] = {scalar}(0);",
                f"            {scalar} dw_acc_{symbol} = {scalar}(0);",
            ]
            for x, y, z, value in terms:
                literal = scalar_literal(value, scalar)
                narrow.extend(
                    [
                        f"            {{ const {scalar} coeff_{symbol} ="
                        f" adjoint_lane[{oo} + {z} * {mo} + a] * {literal};",
                        f"              ds_row_{symbol}[{x}] += coeff_{symbol}"
                        f" * edge[{o2} + {y}] * weights[{weight_offset} + a];",
                        f"              de_row_{symbol}[{y}] += coeff_{symbol}"
                        f" * source_values_lane[{o1} + {x} * {m1} + a]"
                        f" * weights[{weight_offset} + a];",
                        f"              dw_acc_{symbol} += coeff_{symbol}"
                        f" * source_values_lane[{o1} + {x} * {m1} + a]"
                        f" * edge[{o2} + {y}]; }}",
                    ]
                )
            narrow.extend(
                [
                    f"            weights_adjoint[{weight_offset} + a]"
                    f" += dw_acc_{symbol};",
                    f"            for (int y = 0; y < {d2}; ++y)"
                    f" edge_adjoint[{o2} + y] += de_row_{symbol}[y];",
                    f"            for (int x = 0; x < {d1}; ++x)"
                    f" source_adjoint_lane[{o1} + x * {m1} + a]"
                    f" += ds_row_{symbol}[x];",
                    "        }",
                ]
            )
            bodies.append("\n".join(narrow))
        elif has_weight and mode == "uvu" and m2 == 1:
            # Wide coupling: the register row becomes a vector over the source
            # multiplicity axis so the innermost loop vectorizes.
            lines = [
                f"        {scalar} dw_acc_{symbol}[{m1}];",
                f"        for (int a = 0; a < {m1}; ++a)"
                f" dw_acc_{symbol}[a] = {scalar}(0);",
                f"        {scalar} de_acc_{symbol}[{d2 * m1}];",
                f"        for (int i = 0; i < {d2 * m1}; ++i)"
                f" de_acc_{symbol}[i] = {scalar}(0);",
            ]
            for x, y, z, value in terms:
                literal = scalar_literal(value, scalar)
                lines.extend(
                    [
                        f"        {{ const {scalar} ey_{symbol} = edge[{o2} + {y}];",
                        f"          for (int a = 0; a < {m1}; ++a) {{",
                        f"            const {scalar} coeff_{symbol} ="
                        f" adjoint_lane[{oo} + {z} * {mo} + a] * {literal};",
                        f"            source_adjoint_lane[{o1} + {x} * {m1} + a]"
                        f" += coeff_{symbol} * ey_{symbol}"
                        f" * weights[{weight_offset} + a];",
                        f"            de_acc_{symbol}[{y} * {m1} + a]"
                        f" += coeff_{symbol} * source_values_lane"
                        f"[{o1} + {x} * {m1} + a]"
                        f" * weights[{weight_offset} + a];",
                        f"            dw_acc_{symbol}[a] += coeff_{symbol}"
                        f" * source_values_lane[{o1} + {x} * {m1} + a]"
                        f" * ey_{symbol};",
                        "          } }",
                    ]
                )
            lines.extend(
                [
                    f"        for (int a = 0; a < {m1}; ++a)"
                    f" weights_adjoint[{weight_offset} + a] += dw_acc_{symbol}[a];",
                    f"        for (int y = 0; y < {d2}; ++y) {{",
                    f"            {scalar} reduced_{symbol} = {scalar}(0);",
                    f"            for (int a = 0; a < {m1}; ++a)"
                    f" reduced_{symbol} += de_acc_{symbol}[y * {m1} + a];",
                    f"            edge_adjoint[{o2} + y] += reduced_{symbol};",
                    "        }",
                ]
            )
            bodies.append("\n".join(lines))
        else:
            # General lane-major body: covers uvw couplings, weightless paths,
            # and couplings whose second operand has multiplicity above one.
            bodies.append(
                _render_instruction_reverse(
                    layer,
                    instruction,
                    weight_offset,
                    instruction_index,
                    scalar,
                    names={
                        "out_adj": "adjoint_lane",
                        "src": "source_values_lane",
                        "ds": "source_adjoint_lane",
                        "de": "edge_adjoint",
                        "dw": "weights_adjoint",
                    },
                )
            )
        if instruction["has_weight"]:
            weight_offset += math.prod(
                int(extent) for extent in instruction["path_shape"]
            )
    adjoint_binding = [
        f"        const {scalar}* adjoint_lane = adjoint_lane_base"
        f" + target * {target_dimension};"
    ]
    edge_loads = [
        f"            harmonics[static_cast<int64_t>(directed_edge)"
        f" * a->edge_harmonic_count + {harmonic}],"
        for harmonic in harmonic_map
    ]
    tile_count = (
        (source_dimension + channel_tile - 1) // channel_tile if channel_tiled else 1
    )
    lines = [
        f"void oam_source_reverse_{index}("
        f"const SymmetrixJitOAMHostSourceReverseArgsV3* a, int32_t work_item)",
        "{",
        f"    static constexpr int channel_tile = {channel_tile};",
        f"    static constexpr int channel_tile_count = {tile_count};",
        "    const int source = work_item / channel_tile_count;",
        "    const int channel_tile_index = work_item % channel_tile_count;",
        (
            "    const int channel_begin = channel_tile_index * channel_tile;"
            if channel_tiled
            else "    const int channel_begin = 0;"
        ),
        (
            f"    const int bounded_channel_end = channel_begin + channel_tile < {source_dimension} ? channel_begin + channel_tile : {source_dimension};"
            if channel_tiled
            else f"    const int bounded_channel_end = {source_dimension};"
        ),
        f"    const auto* source_values_lane_base ="
        f" static_cast<const {scalar}*>(a->source_node_values);",
        f"    const auto* adjoint_lane_base ="
        f" static_cast<const {scalar}*>(a->target_node_output_adjoint);",
        f"    auto* source_adjoint_out ="
        f" static_cast<{scalar}*>(a->source_node_input_adjoint);",
        f"    const auto* harmonics = static_cast<const {scalar}*>(a->edge_harmonics);",
        f"    const auto* coefficients ="
        f" static_cast<const {scalar}*>(a->spline_coefficients);",
        f"    auto* harmonic_adjoint ="
        f" static_cast<{scalar}*>(a->edge_harmonic_adjoint);",
        "    auto* distance_adjoint = static_cast<double*>(a->distance_adjoint);"
        "  // runtime buffer is double-precision",
        "    const auto* source_indices = a->source_indices;",
        f"    static constexpr int output_lane_index[] = {{{','.join(str(v) for v in lane_column_permutation(out_blocks))}}};",
        "    const auto begin = a->source_edge_offsets[source];",
        "    const auto end = a->source_edge_offsets[source + 1];",
        f"    auto* source_row = source_adjoint_out"
        f" + static_cast<int64_t>(source) * {source_dimension};",
        f"    for (int index = channel_begin; index < bounded_channel_end; ++index) source_row[index] = {scalar}(0);",
        "    if (begin >= end) return;",
        "    for (int64_t slot = begin; slot < end; ++slot) {",
        "        const int64_t directed_edge = a->source_edge_indices[slot];",
        "        const int64_t target = a->target_indices[directed_edge];",
        "        const int64_t edge_source = source_indices[directed_edge];",
        f"        const {scalar}* source_values_lane = source_values_lane_base"
        f" + edge_source * {source_dimension};",
        f"        {scalar} source_edge_adjoint[{source_dimension}];",
        f"        for (int lane = 0; lane < {source_dimension}; ++lane)"
        f" source_edge_adjoint[lane] = {scalar}(0);",
        f"        {scalar}* source_adjoint_lane = source_edge_adjoint;",
        *adjoint_binding,
        render_host_spline_block(scalar, weight_dimension, derivatives=True),
        f"        const {scalar} edge[{edge_dimension}] = {{",
        *edge_loads,
        "        };",
        f"        {scalar} edge_adjoint[{edge_dimension}];",
        f"        for (int index = 0; index < {edge_dimension}; ++index)"
        f" edge_adjoint[index] = {scalar}(0);",
        f"        {scalar} weights_adjoint[{weight_dimension}];",
        f"        for (int index = 0; index < {weight_dimension}; ++index)"
        f" weights_adjoint[index] = {scalar}(0);",
        "\n".join(bodies),
        "        for (int lane = channel_begin; lane < bounded_channel_end; ++lane)",
        "            source_row[lane] += source_edge_adjoint[lane];",
        "        double radial_adjoint = 0.0;",
        "#if defined(_OPENMP)",
        "#pragma omp simd reduction(+:radial_adjoint)",
        "#endif",
        "        for (int function = 0; function < a->spline_function_count;"
        " ++function)",
        "            radial_adjoint += static_cast<double>(weights_adjoint[function])",
        "                * static_cast<double>(weight_derivatives[function]);",
        "        if (channel_tile_index == 0) {",
        "            distance_adjoint[directed_edge] = radial_adjoint;",
        f"            for (int index = 0; index < {edge_dimension}; ++index)",
        f"                harmonic_adjoint[directed_edge * {edge_dimension}"
        " + index] = edge_adjoint[index];",
        "        }",
        "    }",
        "}",
        "",
    ]
    return "\n".join(lines)


def render_oam_receiver_host_plugin(
    contract: Mapping[str, Any],
    precision: str = "float64",
    *,
    architecture: Mapping[str, Any] | None = None,
    host_target: str | None = None,
    generation_policy: OAMHostGenerationPolicy | None = None,
    floating_point_policy: str = "strict",
) -> str:
    """Render a full-message receiver-forward/source-reverse ABI v7 plugin."""

    metadata = oam_receiver_jit_metadata(
        contract,
        precision,
        architecture=architecture,
        host_target=host_target,
        generation_policy=generation_policy,
        floating_point_policy=floating_point_policy,
    )
    forward_functions: list[str] = []
    reverse_functions: list[str] = []
    layer_rows: list[str] = []
    for index, layer in enumerate(contract["layers"]):
        row = metadata["layers"][index]
        forward_functions.append(_render_receiver_forward(layer, metadata, row, index))
        reverse_functions.append(_render_source_reverse(layer, metadata, row, index))
        layer_rows.append(
            "{{sizeof(SymmetrixJitOAMHostLayerV7), {index}u, 1u, 0u, {source}, {edge}, "
            "{weight}, {output}, {instructions}, {tile}, {target}, 0, 0, 0u}}".format(
                index=index,
                source=row["source_dimension"],
                edge=row["edge_dimension"],
                weight=row["weight_dimension"],
                output=row["output_dimension"],
                instructions=row["instruction_count"],
                tile=row["channel_tile"],
                target=row["target_dimension"],
            )
        )
    forward_array = ",".join(
        f"&oam_receiver_forward_{index}" for index in range(len(contract["layers"]))
    )
    reverse_array = ",".join(
        f"&oam_source_reverse_{index}" for index in range(len(contract["layers"]))
    )
    return (
        "// Generated receiver-forward/source-reverse OAM host plugin (ABI v7).\n"
        f"{_ABI_HEADER}\n"
        "#include <cmath>\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n"
        "namespace {\n"
        f"{''.join(forward_functions)}\n"
        f"{''.join(reverse_functions)}\n"
        f"static const SymmetrixJitOAMHostLayerV7 layers[] = "
        f"{{{','.join(layer_rows)}}};\n"
        "static const SymmetrixJitOAMHostReceiverForwardOwnerV4 "
        f"full_forward_owners[] = {{{forward_array}}};\n"
        "static const SymmetrixJitOAMHostSourceReverseOwnerV4 "
        f"full_reverse_owners[] = {{{reverse_array}}};\n"
        "static const SymmetrixJitOAMHostPluginV7 plugin = {\n"
        "SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V7, "
        "sizeof(SymmetrixJitOAMHostPluginV7), sizeof(void*), "
        "SYMMETRIX_JIT_OAM_HOST_PLUGIN_BYTE_ORDER,\n"
        "SYMMETRIX_JIT_OAM_HOST_FULL_FORWARD_V3 | "
        "SYMMETRIX_JIT_OAM_HOST_FULL_REVERSE_V3 | "
        "SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V3" + f", {len(contract['layers'])}, "
        f"{metadata['scalar_size']}, 0u,\n"
        "SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V7, "
        f"{cpp_string_literal(metadata['artifact_id'])}, "
        f"{cpp_string_literal(metadata['generation_fingerprint'])}, "
        f"{cpp_string_literal(metadata['semantic_fingerprint'])}, "
        f"{cpp_string_literal(metadata['structure_fingerprint'])}, "
        f"{cpp_string_literal(metadata['schedule_identity'])}, "
        "nullptr, layers, full_forward_owners, full_reverse_owners, nullptr, "
        "nullptr, "
        f"{cpp_string_literal(metadata['host_target'])}, "
        f"{cpp_string_literal(metadata['forward_owner_schedule'])}, "
        f"{cpp_string_literal(metadata['full_reverse_owner_schedule'])}, nullptr, "
        f"{cpp_string_literal(metadata['tiling_policy'])}, "
        f"{cpp_string_literal(metadata['vector_policy'])}, "
        f"{cpp_string_literal(metadata['shared_memory_policy'])}, "
        f"{cpp_string_literal(metadata['register_budget_policy'])}, "
        f"{cpp_string_literal(metadata['direct_geometry_policy'])}, "
        f"{metadata['scalar_width']}u, 0u, "
        f"{cpp_string_literal(metadata['required_isa'])}, "
        f"{cpp_string_literal(metadata['full_reverse_reduction_policy'])}, "
        "nullptr, nullptr, "
        f"{cpp_string_literal(metadata['schedule_qualification'])}}};\n"
        "}\n"
        'extern "C" SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT '
        "const SymmetrixJitOAMHostPluginV7* "
        "symmetrix_jit_oam_host_plugin_query_v7() { return &plugin; }\n"
    )


__all__ = [
    "OAM_RECEIVER_ARTIFACT_ENVELOPE",
    "OAMHostGenerationPolicy",
    "oam_receiver_jit_metadata",
    "render_oam_receiver_host_plugin",
]
