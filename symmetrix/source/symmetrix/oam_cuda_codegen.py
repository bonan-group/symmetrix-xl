"""Deterministic CUDA code generation for NequIP-OAM v2 R1 layers.

The renderer emits one self-contained CUDA translation unit per OAM contract.
Everything structural is baked into the generated source:

* the certified cubic radial spline evaluated inline from the runtime's
  coefficient table (values in forward; values and analytic derivatives in
  reverse), with the same interval clamping as the host ABI v7 callbacks,
* the weighted tensor-product instructions including every nonzero Wigner
  coefficient, the output mask, and the lane permutation,
* the layer's global scatter normalization factor, and
* the per-layer forward/reverse kernel and launch schedule.

Only the spline coefficient table stays a runtime input (it is certified and
fingerprinted by the native builder, and the kernels read the exact
``[interval][4][function]`` table the host callbacks read).  Pre/post-linear,
gate, self-connection, readout, and ZBL stages are deliberately absent: they
run in the evaluator's existing Kokkos primitives on the active device.

The emitted unit inlines ``oam_cuda_plugin_abi.h`` so it has no include path
dependency, and it exports ``symmetrix_oam_cuda_plugin_query_v3`` returning a
descriptor with the contract, semantic, and structure fingerprints plus exact
layer dimensions and the per-layer ``forward_launch``/``reverse_launch``
entry points.  Host-only pieces are guarded with ``__CUDACC_RTC__`` so the
same text also compiles as a device-only NVRTC module.

Node tensors are lane-major exactly as the host ABI v7 contract defines; the
forward kernel applies the layer output mask and normalization on write, and
the evaluator applies the same product to the node output adjoint before the
source-owned reverse kernel reads it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cuda_codegen_common import (
    DEFAULT_THREADS_PER_BLOCK,
    _compute_capability,
    _normalize_precision,
    _threads_per_block,
)
from .jit import JIT_GENERATION_VERSION, ArtifactEnvelope
from .jit_codegen_common import (
    block_offsets,
    canonical_digest,
    cpp_string_literal,
    harmonic_indices,
    irreps_dimension,
    lane_column_permutation,
    render_cooperative_spline_block,
    render_cooperative_spline_helpers,
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
)

ABI_TAG = "symmetrix.jit.oam.cuda-plugin/3"
ABI_VERSION = 3
ABI_HEADER_NAME = "oam_cuda_plugin_abi.h"
SCHEMA = "symmetrix.oam.cuda-r1"
SCHEMA_VERSION = 3
GENERATOR = "symmetrix.oam.cuda-r1-v21"
# Bumping this version changes the generation fingerprint and therefore the
# artifact identity; older artifacts must never be mixed with the direct module.
OAM_CUDA_CODEGEN_VERSION = 21

# Full-message reverse is a source-owner CTA: every CTA owns one source at a
# time and advances through sources by grid stride. These strings are part of
# the identity so a cache cannot silently reuse a kernel with another schedule.
EDGE_OWNER_SCHEDULE = "source-cta-grid-stride-v1"
EDGE_CHANNEL_POLICY = "input-multiplicity-a-lane-owner-v1"
FORWARD_OWNER_SCHEDULE = "receiver-cta-grid-stride-v1"
FORWARD_CHANNEL_POLICY = "output-multiplicity-lane-owner-v1"
FORWARD_WARP_TASK_OWNER_SCHEDULE = "receiver-source-group-channel-warp-grid-stride-v1"
FORWARD_WARP_TASK_CHANNEL_POLICY = "source-multiplicity-channel-tile-lane-owner-v1"
REVERSE_GROUP_OWNER_SCHEDULE = "source-irrep-specialized-kernel-v1"
REVERSE_GROUP_CHANNEL_POLICY = "input-multiplicity-a-lane-owner-geometry-partial-v1"

CAPABILITY_FORWARD_LAUNCH = 1 << 0
CAPABILITY_REVERSE_LAUNCH = 1 << 1
CAPABILITY_SPLINE_RADIAL = 1 << 2
CAPABILITY_WEIGHTED_TENSOR_PRODUCT = 1 << 3
CAPABILITY_RECEIVER_ACCUMULATION = 1 << 4
CAPABILITY_SOURCE_ADJOINT = 1 << 5
CAPABILITY_GEOMETRY_ADJOINT = 1 << 6
CAPABILITY_RUNTIME_SPLINE_COEFFICIENTS = 1 << 7
CAPABILITY_DIRECT_GEOMETRY = 1 << 8
CAPABILITY_STAGED_FORWARD_INPUTS = 1 << 9
CAPABILITY_GROUPED_REVERSE_PARTIALS = 1 << 10
REQUIRED_CAPABILITIES = (
    CAPABILITY_FORWARD_LAUNCH
    | CAPABILITY_REVERSE_LAUNCH
    | CAPABILITY_SPLINE_RADIAL
    | CAPABILITY_WEIGHTED_TENSOR_PRODUCT
    | CAPABILITY_RECEIVER_ACCUMULATION
    | CAPABILITY_SOURCE_ADJOINT
    | CAPABILITY_GEOMETRY_ADJOINT
    | CAPABILITY_RUNTIME_SPLINE_COEFFICIENTS
    | CAPABILITY_DIRECT_GEOMETRY
)
SCALAR_KINDS = {"float32": 1, "float64": 2}

FORWARD_THREADS_PER_BLOCK = 128
FORWARD_WARPS_PER_BLOCK = FORWARD_THREADS_PER_BLOCK // 32
FORWARD_LIVE_OUTPUT_BUDGETS = (8, 16, 24, 32)


@dataclass(frozen=True)
class OAMCudaGenerationPolicy:
    """Identity-bound choices for generated OAM CUDA execution."""

    forward_live_output_budget: int | None = None
    forward_grouping_mode: str = "output_order"
    forward_group_execution: str = "serial"
    forward_input_mode: str = "inline"
    radial_adjoint_mode: str = "fp64"
    path_reuse_mode: str = "none"
    reverse_group_execution: str = "serial"

    def __post_init__(self) -> None:
        if (
            self.forward_live_output_budget is not None
            and self.forward_live_output_budget not in FORWARD_LIVE_OUTPUT_BUDGETS
        ):
            raise ValueError(
                "forward live output budget must be one of "
                f"{FORWARD_LIVE_OUTPUT_BUDGETS} or None"
            )
        if self.forward_input_mode not in ("inline", "staged"):
            raise ValueError("forward input mode must be 'inline' or 'staged'")
        if self.forward_grouping_mode not in ("output_order", "source_first_fit"):
            raise ValueError(
                "forward grouping mode must be 'output_order' or 'source_first_fit'"
            )
        if self.forward_group_execution not in ("serial", "warp_tasks"):
            raise ValueError("forward group execution must be 'serial' or 'warp_tasks'")
        if (
            self.forward_grouping_mode == "source_first_fit"
            and self.forward_live_output_budget is None
        ):
            raise ValueError("source-first-fit grouping requires a live output budget")
        if (
            self.forward_group_execution == "warp_tasks"
            and self.forward_grouping_mode != "source_first_fit"
        ):
            raise ValueError(
                "warp-task forward execution requires source-first-fit grouping"
            )
        if self.radial_adjoint_mode not in (
            "fp64",
            "fp32",
            "compensated_fp32",
            "hierarchical_fp32",
        ):
            raise ValueError("radial adjoint mode is unsupported")
        if self.path_reuse_mode not in (
            "none",
            "products",
            "products_and_coefficients",
        ):
            raise ValueError("path reuse mode is unsupported")
        if self.reverse_group_execution not in ("serial", "source_groups"):
            raise ValueError(
                "reverse group execution must be 'serial' or 'source_groups'"
            )


def recommended_oam_cuda_generation_policy(
    precision: str,
    target: Any,
    contract: Mapping[str, Any] | None = None,
) -> OAMCudaGenerationPolicy:
    """Return the measured production policy for one precision and target.

    Only the qualified FP32 SM120 retained-message result is selected
    automatically. Every other cell remains the conservative control.
    """

    normalized_precision = _normalize_precision(precision)
    target_code, _ = _compute_capability(target)
    if normalized_precision == "float32" and target_code == 120:
        retained = OAMCudaGenerationPolicy(
            forward_live_output_budget=32,
            forward_grouping_mode="source_first_fit",
            forward_group_execution="warp_tasks",
            forward_input_mode="staged",
            radial_adjoint_mode="fp32",
        )
        if contract is not None:
            try:
                for layer in contract["layers"]:
                    _forward_path_groups(
                        layer,
                        retained.forward_live_output_budget,
                        grouping_mode=retained.forward_grouping_mode,
                    )
            except (KeyError, TypeError, ValueError):
                # Source-first-fit owns each output block exactly once.  A
                # structurally valid custom OAM can instead accumulate several
                # tensor-product instructions into one output block; use the
                # conservative generated schedule for that architecture.
                return OAMCudaGenerationPolicy()
        return retained
    return OAMCudaGenerationPolicy()


def _normalized_generation_policy(
    policy: OAMCudaGenerationPolicy | None,
    *,
    precision: str,
    target: Any,
    contract: Mapping[str, Any],
) -> OAMCudaGenerationPolicy:
    if policy is None:
        return recommended_oam_cuda_generation_policy(
            precision, target, contract=contract
        )
    if not isinstance(policy, OAMCudaGenerationPolicy):
        raise TypeError("generation policy must be OAMCudaGenerationPolicy or None")
    return policy


# The generated descriptor advertises the ``oam-cuda-r1-`` identity family
# shared with the JIT artifact identity; the native ABI loader compares the
# descriptor against the manifest expectation byte for byte.
ARTIFACT_ID_FAMILY = "oam-cuda-r1"
ARTIFACT_ID_PREFIX = ARTIFACT_ID_FAMILY + "-"
# The identity envelope every OAM CUDA metadata builder publishes: the ABI tag
# and version above, the codegen artifact id, the generation/semantic/structure
# fingerprint triple, and the capability word the module exports.
OAM_CUDA_ARTIFACT_ENVELOPE = ArtifactEnvelope(
    "OAM CUDA",
    identity_fingerprint="generation_fingerprint",
    capabilities_key="required_capabilities",
)
# Artifact base names used by the JIT cache for the two compiler paths.
OAM_CUDA_NVRTC_ARTIFACT_NAME = "oam_cuda_r1_module"
OAM_CUDA_NVCC_ARTIFACT_NAME = "oam_cuda_r1_plugin"

_ABI_HEADER = (
    Path(__file__)
    .with_name(ABI_HEADER_NAME)
    .read_text(encoding="ascii")
    .removeprefix("#pragma once\n\n")
)
_RTC_ABI_HEADER = _ABI_HEADER.replace(
    "#include <stdint.h>",
    """typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
namespace std {
using int32_t = ::int32_t;
using int64_t = ::int64_t;
using uint64_t = ::uint64_t;
}""",
)


def _layer_layouts(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the per-layer ABI schedule rows for one validated contract."""

    max_edge_l = 0
    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(contract["layers"]):
        product = layer["edge_tensor_product"]
        spline = layer["kokkos_radial"]
        edge_irreps = str(layer["edge_irreps"])
        output_dimension = irreps_dimension(_irrep_blocks(layer["tp_output_irreps"]))
        weight_dimension = int(product["weight_numel"])
        function_count = int(spline["function_count"])
        if weight_dimension != function_count:
            raise ValueError(
                "OAM tensor-product weight count must equal the radial spline "
                f"function count (layer {index}: {weight_dimension} vs "
                f"{function_count})"
            )
        num_spline_points = int(spline["num_spline_points"])
        if num_spline_points < 2:
            raise ValueError("OAM radial spline table has fewer than two nodes")
        edge_dimension = irreps_dimension(_irrep_blocks(edge_irreps))
        for block in _irrep_blocks(edge_irreps):
            max_edge_l = max(max_edge_l, int(block["l"]))
        rows.append(
            {
                "index": index,
                "source_dimension": irreps_dimension(
                    _irrep_blocks(layer["pre_linear"]["irreps_out"])
                ),
                "edge_dimension": edge_dimension,
                "weight_dimension": weight_dimension,
                "output_dimension": output_dimension,
                "target_dimension": output_dimension,
                "instruction_count": len(product["instructions"]),
                "spline_interval_count": num_spline_points - 1,
                "spline_function_count": function_count,
                "normalization_factor": float(
                    layer["normalization"].get("factor", 1.0)
                ),
            }
        )
    edge_harmonic_count = (max_edge_l + 1) * (max_edge_l + 1)
    for row in rows:
        row["edge_harmonic_count"] = edge_harmonic_count
    return rows


def _forward_path_groups(
    layer: Mapping[str, Any],
    live_output_budget: int | None,
    grouping_mode: str = "output_order",
) -> list[dict[str, Any]]:
    """Partition instructions under an identity-bound accumulator policy."""

    if (
        live_output_budget is not None
        and live_output_budget not in FORWARD_LIVE_OUTPUT_BUDGETS
    ):
        raise ValueError(
            "forward live output budget must be one of "
            f"{FORWARD_LIVE_OUTPUT_BUDGETS} or None"
        )
    if grouping_mode not in ("output_order", "source_first_fit"):
        raise ValueError("unsupported forward grouping mode")
    if grouping_mode == "source_first_fit" and live_output_budget is None:
        raise ValueError("source-first-fit grouping requires a live output budget")
    product = layer["edge_tensor_product"]
    instructions = product["instructions"]
    output_blocks = _irrep_blocks(product["irreps_out"])
    output_order: list[int] = []
    for instruction in instructions:
        output = int(instruction["i_out"])
        if output < 0 or output >= len(output_blocks):
            raise ValueError("tensor-product instruction output block is out of range")
        if output not in output_order:
            output_order.append(output)

    weight_offsets: list[int] = []
    weight_offset = 0
    for instruction in instructions:
        weight_offsets.append(weight_offset)
        if instruction["has_weight"]:
            weight_offset += math.prod(
                int(value) for value in instruction["path_shape"]
            )

    if grouping_mode == "source_first_fit":
        input_blocks = _irrep_blocks(product["irreps_in1"])
        output_counts: dict[int, int] = {}
        source_instructions: dict[int, list[int]] = {}
        for index, instruction in enumerate(instructions):
            if instruction["connection_mode"] != "uvu":
                raise ValueError(
                    "source-first-fit forward grouping currently requires uvu paths"
                )
            source = int(instruction["i_in1"])
            output = int(instruction["i_out"])
            if source < 0 or source >= len(input_blocks):
                raise ValueError(
                    "tensor-product instruction input block is out of range"
                )
            output_counts[output] = output_counts.get(output, 0) + 1
            source_instructions.setdefault(source, []).append(index)
            source_multiplicity = int(input_blocks[source]["multiplicity"])
            output_multiplicity = int(output_blocks[output]["multiplicity"])
            if source_multiplicity != output_multiplicity:
                raise ValueError(
                    "source-first-fit forward grouping requires matching uvu "
                    "input/output multiplicities"
                )
        if any(count != 1 for count in output_counts.values()):
            raise ValueError(
                "source-first-fit forward grouping requires one instruction per "
                "output block"
            )

        groups: list[dict[str, Any]] = []
        for source, indices in source_instructions.items():
            bins: list[tuple[int, list[int]]] = []
            ordered = sorted(
                indices,
                key=lambda index: (
                    -(
                        2 * int(output_blocks[int(instructions[index]["i_out"])]["l"])
                        + 1
                    ),
                    -sum(
                        float(value) != 0.0
                        for value in instructions[index]["wigner_3j"]["values"]
                    ),
                    index,
                ),
            )
            for index in ordered:
                output = int(instructions[index]["i_out"])
                width = 2 * int(output_blocks[output]["l"]) + 1
                for bin_index, (live_outputs, bin_indices) in enumerate(bins):
                    if live_outputs + width <= live_output_budget:
                        bin_indices.append(index)
                        bins[bin_index] = (live_outputs + width, bin_indices)
                        break
                else:
                    bins.append((width, [index]))
            for live_outputs, bin_indices in bins:
                groups.append(
                    {
                        "source_block": source,
                        "source_multiplicity": int(
                            input_blocks[source]["multiplicity"]
                        ),
                        "output_blocks": tuple(
                            int(instructions[index]["i_out"]) for index in bin_indices
                        ),
                        "instruction_indices": tuple(bin_indices),
                        "weight_offsets": tuple(
                            weight_offsets[index] for index in bin_indices
                        ),
                        "live_output_components": live_outputs,
                        "term_count": sum(
                            sum(
                                float(value) != 0.0
                                for value in instructions[index]["wigner_3j"]["values"]
                            )
                            for index in bin_indices
                        ),
                    }
                )
        groups.sort(key=lambda group: group["term_count"], reverse=True)
        return groups

    partitions: list[list[int]] = []
    current: list[int] = []
    current_width = 0
    for output in output_order:
        width = 2 * int(output_blocks[output]["l"]) + 1
        if (
            live_output_budget is not None
            and current
            and current_width + width > live_output_budget
        ):
            partitions.append(current)
            current = []
            current_width = 0
        current.append(output)
        current_width += width
    if current:
        partitions.append(current)

    groups: list[dict[str, Any]] = []
    for outputs in partitions:
        selected = set(outputs)
        indices = tuple(
            index
            for index, instruction in enumerate(instructions)
            if int(instruction["i_out"]) in selected
        )
        groups.append(
            {
                "output_blocks": tuple(outputs),
                "instruction_indices": indices,
                "weight_offsets": tuple(weight_offsets[index] for index in indices),
                "live_output_components": sum(
                    2 * int(output_blocks[output]["l"]) + 1 for output in outputs
                ),
            }
        )
    return groups


def _reverse_source_groups(layer: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Group reverse paths by their disjoint source-input irrep block."""

    product = layer["edge_tensor_product"]
    input_blocks = _irrep_blocks(product["irreps_in1"])
    instructions = product["instructions"]
    weight_offsets: list[int] = []
    weight_offset = 0
    for instruction in instructions:
        weight_offsets.append(weight_offset)
        if instruction["has_weight"]:
            weight_offset += math.prod(
                int(value) for value in instruction["path_shape"]
            )
    grouped: dict[int, list[int]] = {}
    for instruction_index, instruction in enumerate(instructions):
        source_block = int(instruction["i_in1"])
        if source_block < 0 or source_block >= len(input_blocks):
            raise ValueError("tensor-product instruction input block is out of range")
        grouped.setdefault(source_block, []).append(instruction_index)
    offsets = block_offsets(input_blocks)
    groups = []
    for source_block, indices in grouped.items():
        block = input_blocks[source_block]
        groups.append(
            {
                "source_block": source_block,
                "source_offset": offsets[source_block],
                "source_multiplicity": int(block["multiplicity"]),
                "source_width": 2 * int(block["l"]) + 1,
                "instruction_indices": tuple(indices),
                "weight_offsets": tuple(weight_offsets[index] for index in indices),
                "term_count": sum(
                    sum(
                        float(value) != 0.0
                        for value in instructions[index]["wigner_3j"]["values"]
                    )
                    for index in indices
                ),
            }
        )
    groups.sort(key=lambda group: group["term_count"], reverse=True)
    return groups


def oam_cuda_artifact_metadata(
    contract: Mapping[str, Any],
    precision: str = "float32",
    target: Any = None,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    generation_policy: OAMCudaGenerationPolicy | None = None,
    *,
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return four deterministic, independently scoped identities.

    ``contract_fingerprint`` is the ABI field carrying the explicit generated
    architecture fingerprint.  It binds layer counts, irreps, tensor-product
    instructions, spline dimensions, and normalization, but deliberately
    excludes file provenance and learned parameter values.
    ``semantic_fingerprint`` binds the generated forward/reverse mathematics:
    tensor-product instructions, Wigner/path coefficients, output mask,
    normalization, harmonic convention, and runtime cubic-spline semantics.
    Runtime spline values and non-generated model stages are contract inputs,
    not baked kernel semantics. ``structure_fingerprint`` binds lane/irrep
    layouts and ABI extents, excluding numerical factors and launch scratch.
    ``generation_fingerprint`` binds codegen/JIT versions, ABI/schema,
    precision, target, and cooperative schedules/scratch requirements. Compiler
    versions/options are additionally bound by the enclosing JIT cache key.
    """

    precision = _normalize_precision(precision)
    threads = _threads_per_block(threads_per_block)
    policy = _normalized_generation_policy(
        generation_policy, precision=precision, target=target, contract=contract
    )
    if precision == "float64" and policy.radial_adjoint_mode != "fp64":
        raise ValueError("non-FP64 radial adjoints require float32 execution")
    policy_data = asdict(policy)
    spline_coordinate_type = (
        "double" if policy.radial_adjoint_mode == "fp64" else "float"
    )
    target_code, target_name = _compute_capability(target)
    if architecture is None:
        validate_nequip_oam_data(contract)
        architecture = resolve_oam_architecture(contract)
    else:
        architecture = validate_oam_architecture(architecture)
    layouts = _layer_layouts(contract)
    # Forward shares the immutable source row, harmonics, and spline interval.
    # Reverse retains its source-adjoint row, harmonic gradients, and partials.
    scalar_size = 4 if precision == "float32" else 8
    warps = threads // 32
    shared_limit = (
        int(target.get("max_shared_memory_per_block", 49152))
        if isinstance(target, Mapping)
        else 49152
    )
    for row in layouts:
        source_bytes = scalar_size * int(row["source_dimension"])
        harmonic_offset = ((source_bytes + 15) // 16) * 16
        if policy.forward_group_execution == "warp_tasks":
            warp_scalar_bytes = scalar_size * int(row["edge_harmonic_count"])
            warp_bytes = ((warp_scalar_bytes + 7) // 8) * 8 + 12
            warp_bytes = ((warp_bytes + 15) // 16) * 16
            row["forward_static_shared_bytes"] = FORWARD_WARPS_PER_BLOCK * warp_bytes
        else:
            forward_scalar_bytes = harmonic_offset + scalar_size * int(
                row["edge_harmonic_count"]
            )
            forward_bytes = ((forward_scalar_bytes + 7) // 8) * 8 + 12
            row["forward_static_shared_bytes"] = ((forward_bytes + 15) // 16) * 16
        scalar_bytes = harmonic_offset + scalar_size * (
            4 * int(row["edge_harmonic_count"]) + 3 * warps
        )
        if policy.radial_adjoint_mode in ("fp64", "hierarchical_fp32"):
            shared_bytes = ((scalar_bytes + 7) // 8) * 8 + 8 * (warps + 2)
        else:
            scalar_bytes += scalar_size * warps
            shared_bytes = ((scalar_bytes + 7) // 8) * 8 + 12
        row["reverse_static_shared_bytes"] = ((shared_bytes + 15) // 16) * 16
        if (
            max(
                row["forward_static_shared_bytes"],
                row["reverse_static_shared_bytes"],
            )
            > shared_limit
        ):
            raise ValueError(
                "OAM cooperative kernels exceed the target shared-memory limit"
            )
    layer_identity = []
    for layer, row in zip(contract["layers"], layouts):
        forward_groups = _forward_path_groups(
            layer,
            policy.forward_live_output_budget,
            grouping_mode=policy.forward_grouping_mode,
        )
        forward_execution_count = (
            sum(
                (int(group["source_multiplicity"]) + 32 - 1) // 32
                for group in forward_groups
            )
            if policy.forward_group_execution == "warp_tasks"
            else len(forward_groups)
        )
        reverse_groups = _reverse_source_groups(layer)
        grouped_reverse = (
            policy.reverse_group_execution == "source_groups"
            and int(row["index"]) < len(layouts) - 1
        )
        layer_identity.append(
            {
                "layer": row["index"],
                "source_dimension": row["source_dimension"],
                "edge_dimension": row["edge_dimension"],
                "weight_dimension": row["weight_dimension"],
                "output_dimension": row["output_dimension"],
                "target_dimension": row["target_dimension"],
                "instruction_count": row["instruction_count"],
                "spline_interval_count": row["spline_interval_count"],
                "spline_function_count": row["spline_function_count"],
                "edge_harmonic_count": row["edge_harmonic_count"],
                "normalization_factor": row["normalization_factor"],
                "forward_static_shared_bytes": row["forward_static_shared_bytes"],
                "reverse_static_shared_bytes": row["reverse_static_shared_bytes"],
                "forward_path_group_count": len(forward_groups),
                "forward_execution_count": forward_execution_count,
                "forward_path_group_live_components": [
                    group["live_output_components"] for group in forward_groups
                ],
                "reverse_group_count": len(reverse_groups) if grouped_reverse else 0,
                "reverse_group_flags": 0,
                "reverse_source_blocks": [
                    group["source_block"] for group in reverse_groups
                ]
                if grouped_reverse
                else [],
            }
        )
    forward_owner_schedule = (
        FORWARD_WARP_TASK_OWNER_SCHEDULE
        if policy.forward_group_execution == "warp_tasks"
        else FORWARD_OWNER_SCHEDULE
    )
    forward_channel_policy = (
        FORWARD_WARP_TASK_CHANNEL_POLICY
        if policy.forward_group_execution == "warp_tasks"
        else FORWARD_CHANNEL_POLICY
    )
    reverse_owner_schedule = (
        REVERSE_GROUP_OWNER_SCHEDULE
        if policy.reverse_group_execution == "source_groups"
        else EDGE_OWNER_SCHEDULE
    )
    reverse_channel_policy = (
        REVERSE_GROUP_CHANNEL_POLICY
        if policy.reverse_group_execution == "source_groups"
        else EDGE_CHANNEL_POLICY
    )
    contract_fingerprint = oam_tensor_product_architecture_fingerprint(architecture)
    semantic = canonical_digest(
        _canonical(
            {
                "geometry": "normalized-real-harmonics-zxy-v1",
                "radial": "runtime-clamped-cubic-interval-polynomial-function-v1",
                "reverse": "source-geometry-adjoint-preprocessed-output-v1",
                "layers": [
                    {
                        "tensor_product": {
                            key: layer["edge_tensor_product"][key]
                            for key in (
                                "irreps_in1",
                                "irreps_in2",
                                "irreps_out",
                                "instructions",
                                "output_mask",
                            )
                        },
                        "edge_harmonic_indices": harmonic_indices(
                            _irrep_blocks(layer["edge_irreps"])
                        ),
                        "output_lane_permutation": lane_column_permutation(
                            _irrep_blocks(layer["tp_output_irreps"])
                        ),
                        "normalization_factor": row["normalization_factor"],
                    }
                    for layer, row in zip(contract["layers"], layer_identity)
                ],
            }
        )
    )
    structure = canonical_digest(
        {
            "direct_geometry": True,
            "edge_harmonic_count": layouts[0]["edge_harmonic_count"],
            "layers": [
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key
                        not in (
                            "normalization_factor",
                            "forward_static_shared_bytes",
                            "reverse_static_shared_bytes",
                            "forward_path_group_count",
                            "forward_execution_count",
                            "forward_path_group_live_components",
                        )
                    },
                    "source_irreps": layer["edge_tensor_product"]["irreps_in1"],
                    "edge_irreps": layer["edge_irreps"],
                    "output_irreps": layer["tp_output_irreps"],
                }
                for layer, row in zip(contract["layers"], layer_identity)
            ],
        }
    )
    generation_inputs = {
        "oam_cuda_codegen_version": OAM_CUDA_CODEGEN_VERSION,
        "jit_generation_version": JIT_GENERATION_VERSION,
        "abi": ABI_TAG,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "precision": precision,
        "target": target_name,
        "threads_per_block": threads,
        "forward_threads_per_block": FORWARD_THREADS_PER_BLOCK,
        "forward_owner_schedule": forward_owner_schedule,
        "forward_channel_policy": forward_channel_policy,
        "forward_static_shared_bytes": [
            row["forward_static_shared_bytes"] for row in layer_identity
        ],
        "reverse_owner_schedule": reverse_owner_schedule,
        "reverse_channel_policy": reverse_channel_policy,
        "reverse_static_shared_bytes": [
            row["reverse_static_shared_bytes"] for row in layer_identity
        ],
        "direct_geometry": True,
        "execution_variants": ["full_message"],
        "spline_coordinate_type": spline_coordinate_type,
        "generation_policy": policy_data,
    }
    generation = canonical_digest(generation_inputs)
    # The descriptor identity is architecture-bound.  Models with identical
    # generated layouts share an artifact even when their learned values or
    # source archives differ.
    artifact_digest = canonical_digest(
        {
            "contract": contract_fingerprint,
            "generation": generation_inputs,
            "structure": structure,
            "semantic": semantic,
        }
    )
    artifact_id = "{}-{}-{}".format(
        ARTIFACT_ID_FAMILY, precision, artifact_digest.removeprefix("sha256:")[:32]
    )
    required_capabilities = REQUIRED_CAPABILITIES
    if policy.forward_input_mode == "staged":
        required_capabilities |= CAPABILITY_STAGED_FORWARD_INPUTS
    if policy.reverse_group_execution == "source_groups":
        required_capabilities |= CAPABILITY_GROUPED_REVERSE_PARTIALS
    return {
        "abi": ABI_TAG,
        "abi_version": ABI_VERSION,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "artifact_id": artifact_id,
        "architecture": architecture,
        "generation_fingerprint": generation,
        "generation_policy": policy_data,
        "contract_fingerprint": contract_fingerprint,
        "semantic_fingerprint": semantic,
        "structure_fingerprint": structure,
        "direct_geometry": True,
        "execution_variants": ["full_message"],
        "schedule_qualification": (
            "conservative-control"
            if policy == OAMCudaGenerationPolicy()
            else (
                "retained-qualified-exact-cell"
                if policy
                == recommended_oam_cuda_generation_policy(
                    precision, target, contract=contract
                )
                else "candidate-unqualified-explicit"
            )
        ),
        "target_policy": "exact-compute-capability-v1",
        "precision_policy": f"exact-{precision}-v1",
        "spline_coordinate_type": spline_coordinate_type,
        "precision": precision,
        "scalar_kind": SCALAR_KINDS[precision],
        "scalar_size": scalar_size,
        "scalar_type": "float" if precision == "float32" else "double",
        "target_architecture": target_name,
        "target_compute_capability": target_code,
        "threads_per_block": threads,
        "forward_threads_per_block": FORWARD_THREADS_PER_BLOCK,
        "max_threads_per_block": max(
            threads,
            FORWARD_THREADS_PER_BLOCK,
        ),
        "forward_owner_schedule": forward_owner_schedule,
        "forward_channel_policy": forward_channel_policy,
        "forward_execution_count": [
            row["forward_execution_count"] for row in layer_identity
        ],
        "forward_static_shared_bytes": [
            row["forward_static_shared_bytes"] for row in layouts
        ],
        "reverse_owner_schedule": reverse_owner_schedule,
        "reverse_channel_policy": reverse_channel_policy,
        "reverse_static_shared_bytes": [
            row["reverse_static_shared_bytes"] for row in layouts
        ],
        "layer_count": len(layouts),
        "layers": layer_identity,
        "capabilities": required_capabilities,
        "required_capabilities": required_capabilities,
    }


def _render_cooperative_instruction_forward(
    layer: Mapping[str, Any],
    instruction: Mapping[str, Any],
    weight_offset: int,
    scalar: str,
    threads: int,
) -> str:
    """Assign output channels to disjoint lanes, retaining Scalar reductions.

    Repeated instructions for one output block use exactly the same owners.
    Each lane's compact accumulator survives across edges; the Wigner sum is
    rounded before multiplication by its spline weight, as in the host emitter.
    """

    product = layer["edge_tensor_product"]
    operands = (
        ("irreps_in1", "i_in1"),
        ("irreps_in2", "i_in2"),
        ("irreps_out", "i_out"),
    )
    blocks = [
        _irrep_blocks(product[key])[int(instruction[index])] for key, index in operands
    ]
    o1, o2, _ = [
        block_offsets(_irrep_blocks(product[key]))[int(instruction[index])]
        for key, index in operands
    ]
    m1, m2, mo = (int(block["multiplicity"]) for block in blocks)
    d1, d2, do = (2 * int(block["l"]) + 1 for block in blocks)
    output_block = int(instruction["i_out"])
    slots = (mo + threads - 1) // threads
    mode = instruction["connection_mode"]
    channel = "c" if mode == "uvw" else "a"
    weight_index = {
        "uvu": f"{weight_offset} + a*{m2} + b",
        "uuu": f"{weight_offset} + a",
        "uvw": f"{weight_offset} + (a*{m2}+b)*{mo}+c",
    }[mode]
    weight = (
        f"oam_spline_value(coefficient_base, args.spline_function_count, {weight_index}, coordinate)"
        if instruction["has_weight"]
        else f"{scalar}(1)"
    )
    lines = [
        "{",
        "#pragma unroll",
        f"for (int slot = 0; slot < {slots}; ++slot) {{",
        f"    const int {channel} = static_cast<int>(threadIdx.x) + slot * {threads};",
        f"    if ({channel} < {mo}) {{",
    ]
    if mode == "uvw":
        lines.append(f"        for (int a = 0; a < {m1}; ++a) {{")
    lines.append(f"        for (int b = 0; b < {m2}; ++b) {{")
    if mode == "uuu":
        lines.append("            if (a != b) continue;")
    lines.append(f"            const {scalar} weight = {weight};")
    harmonic_map = harmonic_indices(_irrep_blocks(layer["edge_irreps"]))
    path = instruction["wigner_3j"]["values"]
    factor = float(instruction["path_weight"])
    for z in range(do):
        terms = [
            (x, y, float(path[(x * d2 + y) * do + z]))
            for x in range(d1)
            for y in range(d2)
            if float(path[(x * d2 + y) * do + z]) != 0.0
        ]
        if not terms:
            continue
        lines.append(f"            {{ {scalar} sum = {scalar}(0);")
        for x, y, value in terms:
            harmonic = harmonic_map[o2 + y]
            # All multiplicity copies of one edge irrep share its harmonics.
            edge_value = f"harmonics[{harmonic}]"
            literal = scalar_literal(factor * value, scalar)
            lines.append(
                f"              sum += src[{o1} + {x}*{m1} + a] * {edge_value} * {literal};"
            )
        lines.append(
            f"              accumulator_{output_block}[slot*{do} + {z}] += sum * weight; }}"
        )
    lines.append("        }")
    if mode == "uvw":
        lines.append("        }")
    lines.extend(["    }", "}", "}"])
    return "\n".join(lines)


def _render_warp_instruction_forward(
    layer: Mapping[str, Any],
    instruction: Mapping[str, Any],
    weight_offset: int,
    scalar: str,
) -> str:
    """Render one UVU path for a warp-owned 32-channel task."""

    if instruction["connection_mode"] != "uvu":
        raise ValueError("warp-task forward execution currently requires uvu paths")
    product = layer["edge_tensor_product"]
    operands = (
        ("irreps_in1", "i_in1"),
        ("irreps_in2", "i_in2"),
        ("irreps_out", "i_out"),
    )
    blocks = [
        _irrep_blocks(product[key])[int(instruction[index])] for key, index in operands
    ]
    o1, o2, _ = [
        block_offsets(_irrep_blocks(product[key]))[int(instruction[index])]
        for key, index in operands
    ]
    m1, m2, mo = (int(block["multiplicity"]) for block in blocks)
    d1, d2, do = (2 * int(block["l"]) + 1 for block in blocks)
    if m1 != mo:
        raise ValueError(
            "warp-task forward execution requires matching uvu multiplicities"
        )
    output_block = int(instruction["i_out"])
    weight = (
        "oam_spline_value(coefficient_base, args.spline_function_count, "
        f"{weight_offset} + a*{m2} + b, coordinate)"
        if instruction["has_weight"]
        else f"{scalar}(1)"
    )
    harmonic_map = harmonic_indices(_irrep_blocks(layer["edge_irreps"]))
    path = instruction["wigner_3j"]["values"]
    factor = float(instruction["path_weight"])
    lines = ["{", f"    if (a < {m1}) {{", f"        for (int b = 0; b < {m2}; ++b) {{"]
    lines.append(f"            const {scalar} weight = {weight};")
    for z in range(do):
        terms = [
            (x, y, float(path[(x * d2 + y) * do + z]))
            for x in range(d1)
            for y in range(d2)
            if float(path[(x * d2 + y) * do + z]) != 0.0
        ]
        if not terms:
            continue
        component_sum = f"sum_{output_block}_{z}"
        lines.append(f"            {scalar} {component_sum} = {scalar}(0);")
        for x, y, value in terms:
            harmonic = harmonic_map[o2 + y]
            literal = scalar_literal(factor * value, scalar)
            lines.append(
                f"            {component_sum} += src[{o1} + {x}*{m1} + a] "
                f"* harmonics[{harmonic}] * {literal};"
            )
        lines.append(
            f"            accumulator_{output_block}_{z} += {component_sum} * weight;"
        )
    lines.extend(["        }", "    }", "}"])
    return "\n".join(lines)


def _render_harmonic_helpers(metadata: Mapping[str, Any], scalar: str) -> str:
    """Render the fixed-degree normalized real harmonics used by OAM CUDA."""

    harmonic_count = int(metadata["layers"][0]["edge_harmonic_count"])
    max_l = math.isqrt(harmonic_count) - 1
    if (max_l + 1) * (max_l + 1) != harmonic_count:
        raise ValueError("OAM edge harmonic count is not a square")
    if max_l > 4:
        raise ValueError(
            "direct OAM CUDA harmonic streaming currently supports edge l_max <= 4"
        )
    normalization = scalar_literal(2.0 * math.sqrt(math.pi), scalar)
    c0 = scalar_literal(0.282094791773878, scalar)
    c1 = scalar_literal(0.48860251190292, scalar)
    lines = [
        "__device__ inline void oam_compute_harmonics(",
        "    Scalar ox, Scalar oy, Scalar oz, Scalar* harmonics)",
        "{",
        f"    const Scalar normalization = static_cast<Scalar>({normalization});",
        "    const Scalar radius_squared = ox * ox + oy * oy + oz * oz;",
        "    const Scalar inverse_radius = Scalar(1) / ::sqrt(radius_squared);",
        "    // SpheriCart receives the shuffled (z, x, y) vector.",
        "    const Scalar x = oz * inverse_radius;",
        "    const Scalar y = ox * inverse_radius;",
        "    const Scalar z = oy * inverse_radius;",
        f"    Scalar raw[{harmonic_count}];",
        f"    raw[0] = static_cast<Scalar>({c0});",
    ]
    if max_l >= 1:
        lines += [
            f"    raw[1] = static_cast<Scalar>({c1}) * y;",
            f"    raw[2] = static_cast<Scalar>({c1}) * z;",
            f"    raw[3] = static_cast<Scalar>({c1}) * x;",
        ]
    if max_l >= 2:
        lines += [
            "    raw[4] = static_cast<Scalar>(2.23606797749979) * x * raw[1];",
            "    raw[5] = static_cast<Scalar>(2.23606797749979) * z * raw[1];",
            (
                "    raw[6] = -static_cast<Scalar>(0.315391565252520)"
                " * (x * x + y * y - Scalar(2) * z * z);"
            ),
            "    raw[7] = static_cast<Scalar>(2.23606797749979) * x * raw[2];",
            ("    raw[8] = static_cast<Scalar>(0.54627421529604) * (x * x - y * y);"),
        ]
    if max_l >= 3:
        lines += [
            "    raw[9] = -static_cast<Scalar>(0.59004358992664) * y * (y * y - Scalar(3) * x * x);",
            "    raw[10] = static_cast<Scalar>(2.64575131106459) * z * raw[4];",
            "    const Scalar degree_three_tmp = -static_cast<Scalar>(0.457045799464466) * (x * x + y * y - Scalar(4) * z * z);",
            "    raw[11] = y * degree_three_tmp;",
            "    raw[13] = x * degree_three_tmp;",
            "    raw[12] = -static_cast<Scalar>(1.49270533036046) * z * (z * z - static_cast<Scalar>(2.37799637856361) * raw[6]);",
            "    raw[14] = static_cast<Scalar>(1.44530572132028) * z * (x * x - y * y);",
            "    raw[15] = static_cast<Scalar>(0.59004358992664) * x * (x * x - Scalar(3) * y * y);",
        ]
    if max_l >= 4:
        lines += [
            "    raw[16] = static_cast<Scalar>(4.194391357527674) * raw[4] * raw[8];",
            "    raw[17] = Scalar(3) * z * raw[9];",
            "    const Scalar degree_four_tmp_1 = -static_cast<Scalar>(0.866025403784439) * (x * x + y * y - Scalar(6) * z * z);",
            "    raw[18] = degree_four_tmp_1 * raw[4];",
            "    raw[22] = degree_four_tmp_1 * raw[8];",
            "    raw[20] = -static_cast<Scalar>(0.69436507482941) * (y * raw[11] - static_cast<Scalar>(1.6329931618554521) * z * raw[12] + x * raw[13]);",
            "    const Scalar degree_four_tmp_2 = -static_cast<Scalar>(1.224744871391589) * (z * z - static_cast<Scalar>(4.755992757127213) * raw[6]);",
            "    raw[19] = raw[5] * degree_four_tmp_2;",
            "    raw[21] = raw[7] * degree_four_tmp_2;",
            "    raw[23] = Scalar(3) * z * raw[15];",
            "    raw[24] = -static_cast<Scalar>(1.060660171779821) * (y * raw[9] - x * raw[15]);",
        ]
    lines += [
        f"    for (int harmonic = 0; harmonic < {harmonic_count}; ++harmonic)",
        "        harmonics[harmonic] = normalization * raw[harmonic];",
        "}",
        "",
        "__device__ inline void oam_compute_harmonic_gradients(",
        (
            "    Scalar ox, Scalar oy, Scalar oz,"
            " Scalar* gradient_x, Scalar* gradient_y, Scalar* gradient_z)"
        ),
        "{",
        f"    const Scalar normalization = static_cast<Scalar>({normalization});",
        "    const Scalar radius_squared = ox * ox + oy * oy + oz * oz;",
        "    const Scalar inverse_radius = Scalar(1) / ::sqrt(radius_squared);",
        "    const Scalar x = oz * inverse_radius;",
        "    const Scalar y = ox * inverse_radius;",
        "    const Scalar z = oy * inverse_radius;",
    ]
    # Degree-one gradients are constant in normalized coordinates, so their
    # derivative path does not consume the harmonic values themselves. Avoid a
    # write-only array for OAM-S while retaining it for higher-degree formulas.
    if max_l >= 2:
        lines += [
            f"    Scalar raw[{harmonic_count}];",
            f"    raw[0] = static_cast<Scalar>({c0});",
        ]
    lines += [
        f"    Scalar raw_dx[{harmonic_count}] = {{}};",
        f"    Scalar raw_dy[{harmonic_count}] = {{}};",
        f"    Scalar raw_dz[{harmonic_count}] = {{}};",
        "    raw_dx[0] = raw_dy[0] = raw_dz[0] = Scalar(0);",
    ]
    if max_l >= 1:
        if max_l >= 2:
            lines += [
                f"    raw[1] = static_cast<Scalar>({c1}) * y;",
                f"    raw[2] = static_cast<Scalar>({c1}) * z;",
                f"    raw[3] = static_cast<Scalar>({c1}) * x;",
            ]
        lines += [
            "    raw_dx[1] = raw_dx[2] = static_cast<Scalar>(0);",
            f"    raw_dx[3] = static_cast<Scalar>({c1});",
            f"    raw_dy[1] = static_cast<Scalar>({c1});",
            "    raw_dy[2] = raw_dy[3] = static_cast<Scalar>(0);",
            "    raw_dz[1] = raw_dz[3] = static_cast<Scalar>(0);",
            f"    raw_dz[2] = static_cast<Scalar>({c1});",
        ]
    if max_l >= 2:
        lines += [
            "    raw[4] = static_cast<Scalar>(2.23606797749979) * x * raw[1];",
            "    raw[5] = static_cast<Scalar>(2.23606797749979) * z * raw[1];",
            (
                "    raw[6] = -static_cast<Scalar>(0.315391565252520)"
                " * (x * x + y * y - Scalar(2) * z * z);"
            ),
            "    raw[7] = static_cast<Scalar>(2.23606797749979) * x * raw[2];",
            ("    raw[8] = static_cast<Scalar>(0.54627421529604) * (x * x - y * y);"),
            "    raw_dx[4] = static_cast<Scalar>(2.23606797749979) * raw[1];",
            "    raw_dx[5] = Scalar(0);",
            "    raw_dx[6] = -static_cast<Scalar>(1.29099444873581) * raw[3];",
            "    raw_dx[7] = static_cast<Scalar>(2.23606797749979) * raw[2];",
            "    raw_dx[8] = static_cast<Scalar>(2.23606797749979) * raw[3];",
            "    raw_dy[4] = static_cast<Scalar>(2.23606797749979) * raw[3];",
            "    raw_dy[5] = raw_dx[7];",
            "    raw_dy[6] = -static_cast<Scalar>(0.577350269189626) * raw_dx[4];",
            "    raw_dy[7] = Scalar(0);",
            "    raw_dy[8] = -raw_dx[4];",
            "    raw_dz[4] = Scalar(0);",
            "    raw_dz[5] = raw_dx[4];",
            "    raw_dz[6] = static_cast<Scalar>(1.15470053837925) * raw_dx[7];",
            "    raw_dz[7] = raw_dy[4];",
            "    raw_dz[8] = Scalar(0);",
        ]
    if max_l >= 3:
        lines += [
            "    raw[9] = -static_cast<Scalar>(0.59004358992664) * y * (y * y - Scalar(3) * x * x);",
            "    raw[10] = static_cast<Scalar>(2.64575131106459) * z * raw[4];",
            "    const Scalar degree_three_tmp = -static_cast<Scalar>(0.457045799464466) * (x * x + y * y - Scalar(4) * z * z);",
            "    raw[11] = y * degree_three_tmp;",
            "    raw[13] = x * degree_three_tmp;",
            "    raw[12] = -static_cast<Scalar>(1.49270533036046) * z * (z * z - static_cast<Scalar>(2.37799637856361) * raw[6]);",
            "    raw[14] = static_cast<Scalar>(1.44530572132028) * z * (x * x - y * y);",
            "    raw[15] = static_cast<Scalar>(0.59004358992664) * x * (x * x - Scalar(3) * y * y);",
            "    raw_dx[9] = static_cast<Scalar>(3.24037034920393) * raw[4];",
            "    raw_dx[10] = static_cast<Scalar>(2.64575131106459) * raw[5];",
            "    raw_dx[11] = -static_cast<Scalar>(0.83666002653408) * raw[4];",
            "    raw_dx[12] = -static_cast<Scalar>(2.04939015319192) * raw[7];",
            "    raw_dx[13] = static_cast<Scalar>(0.91409159892893) * (y * y - z * z + static_cast<Scalar>(4.75599275712721) * raw[6]);",
            "    raw_dx[14] = static_cast<Scalar>(2.64575131106459) * raw[7];",
            "    raw_dx[15] = static_cast<Scalar>(3.24037034920393) * raw[8];",
            "    raw_dy[9] = raw_dx[15];",
            "    raw_dy[10] = raw_dx[14];",
            "    raw_dy[11] = -static_cast<Scalar>(0.91409159892893) * (y * y - z * z - static_cast<Scalar>(1.58533091904240) * raw[6]);",
            "    raw_dy[12] = -static_cast<Scalar>(2.04939015319192) * raw[5];",
            "    raw_dy[13] = -static_cast<Scalar>(0.83666002653408) * raw[4];",
            "    raw_dy[14] = -raw_dx[10];",
            "    raw_dy[15] = -raw_dx[9];",
            "    raw_dz[9] = Scalar(0);",
            "    raw_dz[10] = static_cast<Scalar>(2.64575131106459) * raw[4];",
            "    raw_dz[11] = static_cast<Scalar>(3.34664010613630) * raw[5];",
            "    raw_dz[12] = static_cast<Scalar>(3.54964786985977) * raw[6];",
            "    raw_dz[13] = static_cast<Scalar>(3.34664010613630) * raw[7];",
            "    raw_dz[14] = static_cast<Scalar>(2.64575131106459) * raw[8];",
            "    raw_dz[15] = Scalar(0);",
        ]
    if max_l >= 4:
        lines += [
            "    raw[16] = static_cast<Scalar>(4.194391357527674) * raw[4] * raw[8];",
            "    raw[17] = Scalar(3) * z * raw[9];",
            "    const Scalar degree_four_tmp_1 = -static_cast<Scalar>(0.866025403784439) * (x * x + y * y - Scalar(6) * z * z);",
            "    raw[18] = degree_four_tmp_1 * raw[4];",
            "    raw[22] = degree_four_tmp_1 * raw[8];",
            "    raw[20] = -static_cast<Scalar>(0.69436507482941) * (y * raw[11] - static_cast<Scalar>(1.6329931618554521) * z * raw[12] + x * raw[13]);",
            "    const Scalar degree_four_tmp_2 = -static_cast<Scalar>(1.224744871391589) * (z * z - static_cast<Scalar>(4.755992757127213) * raw[6]);",
            "    raw[19] = raw[5] * degree_four_tmp_2;",
            "    raw[21] = raw[7] * degree_four_tmp_2;",
            "    raw[23] = Scalar(3) * z * raw[15];",
            "    raw[24] = -static_cast<Scalar>(1.060660171779821) * (y * raw[9] - x * raw[15]);",
            "    raw_dx[16] = static_cast<Scalar>(4.242640687119285) * raw[9];",
            "    raw_dx[17] = static_cast<Scalar>(3.674234614174767) * raw[10];",
            "    raw_dx[18] = static_cast<Scalar>(1.892349391515120) * y * (y * y + static_cast<Scalar>(4.755992757127213) * raw[6]);",
            "    raw_dx[19] = -static_cast<Scalar>(1.388730149658827) * raw[10];",
            "    raw_dx[20] = -static_cast<Scalar>(2.777460299317654) * raw[13];",
            "    raw_dx[21] = -static_cast<Scalar>(1.338093087114578) * (z * z * z - static_cast<Scalar>(2.745873698591307) * y * raw[5] - static_cast<Scalar>(4.019547514144073) * raw[12]);",
            "    raw_dx[22] = -static_cast<Scalar>(1.892349391515120) * x * (x * x - Scalar(3) * z * z);",
            "    raw_dx[23] = static_cast<Scalar>(3.674234614174767) * raw[14];",
            "    raw_dx[24] = static_cast<Scalar>(4.242640687119285) * raw[15];",
            "    raw_dy[16] = raw_dx[24];",
            "    raw_dy[17] = raw_dx[23];",
            "    raw_dy[18] = -static_cast<Scalar>(1.892349391515120) * x * (y * y - Scalar(2) * z * z - static_cast<Scalar>(1.585330919042404) * raw[6]);",
            "    raw_dy[19] = -static_cast<Scalar>(1.338093087114578) * (z * (Scalar(3) * y * y - z * z) - static_cast<Scalar>(1.339849171381358) * raw[12]);",
            "    raw_dy[20] = -static_cast<Scalar>(2.777460299317654) * raw[11];",
            "    raw_dy[21] = raw_dx[19];",
            "    raw_dy[22] = static_cast<Scalar>(1.892349391515120) * y * (y * y - Scalar(3) * z * z);",
            "    raw_dy[23] = -raw_dx[17];",
            "    raw_dy[24] = -raw_dx[16];",
            "    raw_dz[16] = Scalar(0);",
            "    raw_dz[17] = Scalar(3) * raw[9];",
            "    raw_dz[18] = static_cast<Scalar>(3.927922024247863) * raw[10];",
            "    raw_dz[19] = static_cast<Scalar>(4.391550328268399) * raw[11];",
            "    raw_dz[20] = static_cast<Scalar>(4.535573676110727) * raw[12];",
            "    raw_dz[21] = static_cast<Scalar>(4.391550328268399) * raw[13];",
            "    raw_dz[22] = static_cast<Scalar>(3.927922024247863) * raw[14];",
            "    raw_dz[23] = Scalar(3) * raw[15];",
            "    raw_dz[24] = Scalar(0);",
        ]
    lines += [
        f"    for (int harmonic = 0; harmonic < {harmonic_count}; ++harmonic) {{",
        (
            "        const Scalar dot = raw_dx[harmonic] * x"
            " + raw_dy[harmonic] * y + raw_dz[harmonic] * z;"
        ),
        (
            "        const Scalar dx = normalization * (raw_dx[harmonic] - x * dot)"
            " * inverse_radius;"
        ),
        (
            "        const Scalar dy = normalization * (raw_dy[harmonic] - y * dot)"
            " * inverse_radius;"
        ),
        (
            "        const Scalar dz = normalization * (raw_dz[harmonic] - z * dot)"
            " * inverse_radius;"
        ),
        "        // Convert derivatives from shuffled (z, x, y) to original xyz.",
        "        gradient_x[harmonic] = dy;",
        "        gradient_y[harmonic] = dz;",
        "        gradient_z[harmonic] = dx;",
        "    }",
        "}",
    ]
    return "\n".join(lines)


def _render_warp_task_forward_kernel(
    layer: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
    scalar: str,
    *,
    live_output_budget: int,
    input_mode: str,
    spline_coordinate_type: str,
) -> str:
    """Render LunaTP-style receiver/source-group/channel warp tasks."""

    product = layer["edge_tensor_product"]
    source_dimension = int(row["source_dimension"])
    output_dimension = int(row["output_dimension"])
    output_blocks = _irrep_blocks(product["irreps_out"])
    output_offsets = block_offsets(output_blocks)
    normalization = scalar_literal(row["normalization_factor"], scalar)
    groups = _forward_path_groups(
        layer,
        live_output_budget,
        grouping_mode="source_first_fit",
    )
    executions = [
        (group_index, channel_base)
        for group_index, group in enumerate(groups)
        for channel_base in range(0, int(group["source_multiplicity"]), 32)
    ]
    cases: list[str] = []
    for execution_index, (group_index, channel_base) in enumerate(executions):
        group = groups[group_index]
        declarations: list[str] = []
        stores: list[str] = []
        for output_block in group["output_blocks"]:
            block = output_blocks[output_block]
            offset = output_offsets[output_block]
            multiplicity = int(block["multiplicity"])
            width = 2 * int(block["l"]) + 1
            declarations.extend(
                f"            Scalar accumulator_{output_block}_{component} = Scalar(0);"
                for component in range(width)
            )
            stores.append(f"            if (a < {multiplicity}) {{")
            for component in range(width):
                stores.append(
                    f"                destination[{offset} + {component}*{multiplicity} + a] = "
                    f"accumulator_{output_block}_{component} * static_cast<Scalar>({normalization}) "
                    f"* output_mask[{offset} + a*{width} + {component}];"
                )
            stores.append("            }")
        bodies = [
            _render_warp_instruction_forward(
                layer,
                product["instructions"][instruction_index],
                weight_offset,
                scalar,
            )
            for instruction_index, weight_offset in zip(
                group["instruction_indices"], group["weight_offsets"]
            )
        ]
        if input_mode == "staged":
            harmonic_setup = (
                "const Scalar* harmonics = staged_harmonics "
                f"+ directed_edge * {int(row['edge_harmonic_count'])};"
            )
        else:
            harmonic_setup = """if (lane == 0) {
                    const Scalar* vector = edge_vectors + directed_edge * 3;
                    oam_compute_harmonics(
                        vector[0], vector[1], vector[2], scratch.harmonics);
                }
                const Scalar* harmonics = scratch.harmonics;"""
        coordinate = (
            "const double coordinate = scratch.coordinate;"
            if spline_coordinate_type == "double"
            else "const Scalar coordinate = static_cast<Scalar>(scratch.coordinate);"
        )
        spline_setup = f"""if (lane == 0) {{
                    const double radius = args.distances[directed_edge];
                    int interval = static_cast<int>(::floor(
                        (radius - args.spline_x0) / args.spline_h));
                    double local_coordinate = radius - args.spline_x0
                        - args.spline_h * interval;
                    if (interval < 0) {{
                        interval = 0;
                        local_coordinate = 0.0;
                    }} else if (interval >= args.spline_interval_count) {{
                        interval = args.spline_interval_count - 1;
                        local_coordinate = args.spline_h;
                    }}
                    scratch.interval = interval;
                    scratch.coordinate = local_coordinate;
                }}
                __syncwarp();
                {coordinate}
                const Scalar* coefficient_base = coefficients
                    + static_cast<std::int64_t>(scratch.interval) * 4
                        * static_cast<std::int64_t>(args.spline_function_count);"""
        cases.append(
            f"""        case {execution_index}: {{
            const int a = {channel_base} + lane;
{chr(10).join(declarations)}
            const std::int64_t first = args.receiver_offsets[receiver];
            const std::int64_t last = args.receiver_offsets[receiver + 1];
            for (std::int64_t directed_edge = first; directed_edge < last;
                 ++directed_edge)
            {{
                const Scalar* src = source_values
                    + static_cast<std::int64_t>(args.source_indices[directed_edge])
                        * {source_dimension};
                {harmonic_setup}
                {spline_setup}
{chr(10).join("                " + line for body in bodies for line in body.splitlines())}
                __syncwarp();
            }}
            Scalar* destination = messages
                + static_cast<std::uint64_t>(receiver)
                    * static_cast<std::uint64_t>({output_dimension});
{chr(10).join(stores)}
            break;
        }}"""
        )
    harmonic_count = int(row["edge_harmonic_count"])
    warp_scratch_bytes = (
        int(row["forward_static_shared_bytes"]) // FORWARD_WARPS_PER_BLOCK
    )
    edge_vectors_declaration = (
        "const Scalar* edge_vectors = static_cast<const Scalar*>(args.edge_vectors);"
        if input_mode == "inline"
        else ""
    )
    staged_harmonics_declaration = (
        "const Scalar* staged_harmonics = "
        "static_cast<const Scalar*>(args.edge_harmonics);"
        if input_mode == "staged"
        else ""
    )
    return f"""// Forward: each warp owns one receiver/source-group/channel tile task.
extern "C" __global__ void __launch_bounds__({FORWARD_THREADS_PER_BLOCK})
symmetrix_oam_forward_kernel_{index}(SymmetrixOAMCudaLayerArgsV2 args)
{{
    constexpr int forward_warps = {FORWARD_WARPS_PER_BLOCK};
    constexpr int forward_execution_count = {len(executions)};
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const Scalar* source_values = static_cast<const Scalar*>(args.source_node_values);
    {edge_vectors_declaration}
    {staged_harmonics_declaration}
    const Scalar* coefficients = static_cast<const Scalar*>(args.spline_coefficients);
    Scalar* messages = static_cast<Scalar*>(args.node_messages);
    const Scalar* output_mask = oam_output_mask_{index};
    struct alignas(16) WarpScratch {{
        Scalar harmonics[{harmonic_count}];
        double coordinate;
        int interval;
    }};
    static_assert(sizeof(WarpScratch) == {warp_scratch_bytes});
    __shared__ WarpScratch warp_scratch[forward_warps];
    WarpScratch& scratch = warp_scratch[warp];
    const std::int64_t task_count = args.num_nodes
        * static_cast<std::int64_t>(forward_execution_count);
    for (std::int64_t task = static_cast<std::int64_t>(blockIdx.x) * forward_warps + warp;
         task < task_count;
         task += static_cast<std::int64_t>(gridDim.x) * forward_warps)
    {{
        const int execution = static_cast<int>(task % forward_execution_count);
        const std::int64_t receiver = task / forward_execution_count;
        switch (execution) {{
{chr(10).join(cases)}
        default:
            break;
        }}
    }}
}}"""


def _render_forward_kernel(
    layer: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
    scalar: str,
    threads: int,
    live_output_budget: int | None = None,
    grouping_mode: str = "output_order",
    group_execution: str = "serial",
    input_mode: str = "inline",
    spline_coordinate_type: str = "double",
) -> str:
    if group_execution == "warp_tasks":
        if grouping_mode != "source_first_fit" or live_output_budget is None:
            raise ValueError(
                "warp-task forward execution requires budgeted source-first-fit grouping"
            )
        return _render_warp_task_forward_kernel(
            layer,
            row,
            index,
            scalar,
            live_output_budget=live_output_budget,
            input_mode=input_mode,
            spline_coordinate_type=spline_coordinate_type,
        )
    if group_execution != "serial":
        raise ValueError("unsupported forward group execution")
    if grouping_mode != "output_order":
        raise ValueError("serial forward execution requires output-order grouping")
    if input_mode not in ("inline", "staged"):
        raise ValueError("forward input mode must be 'inline' or 'staged'")
    product = layer["edge_tensor_product"]
    source_dimension = int(row["source_dimension"])
    output_dimension = int(row["output_dimension"])
    lane_index = lane_column_permutation(_irrep_blocks(layer["tp_output_irreps"]))
    if len(lane_index) != output_dimension:
        raise ValueError("tensor-product output irrep layout is inconsistent")
    normalization = scalar_literal(row["normalization_factor"], scalar)
    output_blocks = _irrep_blocks(product["irreps_out"])
    output_offsets = block_offsets(output_blocks)
    groups = _forward_path_groups(
        layer, live_output_budget, grouping_mode=grouping_mode
    )
    rendered_groups: list[str] = []
    for group_index, group in enumerate(groups):
        declarations: list[str] = []
        stores: list[str] = []
        for block_index in group["output_blocks"]:
            block = output_blocks[block_index]
            offset = output_offsets[block_index]
            multiplicity = int(block["multiplicity"])
            width = 2 * int(block["l"]) + 1
            slots = (multiplicity + threads - 1) // threads
            declarations.append(
                f"            Scalar accumulator_{block_index}[{slots * width}] = {{}};"
            )
            stores.extend(
                [
                    "            #pragma unroll",
                    f"            for (int slot = 0; slot < {slots}; ++slot) {{",
                    f"                const int channel = static_cast<int>(threadIdx.x) + slot * {threads};",
                    f"                if (channel < {multiplicity}) {{",
                ]
            )
            for z in range(width):
                stores.append(
                    f"                    destination[{offset} + {z}*{multiplicity} + channel] = "
                    f"accumulator_{block_index}[slot*{width} + {z}] * static_cast<Scalar>({normalization}) "
                    f"* output_mask[{offset} + channel*{width} + {z}];"
                )
            stores.extend(["                }", "            }"])
        bodies = [
            _render_cooperative_instruction_forward(
                layer,
                product["instructions"][instruction_index],
                weight_offset,
                scalar,
                threads,
            )
            for instruction_index, weight_offset in zip(
                group["instruction_indices"], group["weight_offsets"]
            )
        ]
        if input_mode == "staged":
            harmonic_setup = (
                "const Scalar* harmonics = staged_harmonics "
                f"+ directed_edge * {int(row['edge_harmonic_count'])};"
            )
        else:
            harmonic_setup = """if (threadIdx.x == 0) {
                    const Scalar* vector = edge_vectors + directed_edge * 3;
                    oam_compute_harmonics(vector[0], vector[1], vector[2], scratch.harmonics);
                }"""
        rendered_groups.append(
            f"""        {{ // Forward path group {group_index}: {group["live_output_components"]} live components.
{chr(10).join(declarations)}
            for (std::int64_t directed_edge = first; directed_edge < last; ++directed_edge)
            {{
                const Scalar* source = source_values
                    + static_cast<std::int64_t>(args.source_indices[directed_edge]) * {source_dimension};
                for (std::int64_t position = static_cast<std::int64_t>(threadIdx.x);
                     position < {source_dimension}; position += blockDim.x)
                    scratch.source[position] = source[position];
                {harmonic_setup}
                {render_cooperative_spline_block(spline_coordinate_type)}
                const Scalar* src = scratch.source;
                {"" if input_mode == "staged" else "const Scalar* harmonics = scratch.harmonics;"}
{chr(10).join("                " + line for body in bodies for line in body.splitlines())}
                __syncthreads();
            }}
            Scalar* destination = messages
                + static_cast<std::uint64_t>(receiver)
                    * static_cast<std::uint64_t>({output_dimension});
{chr(10).join(stores)}
            __syncthreads();
        }}"""
        )
    harmonic_count = int(row["edge_harmonic_count"])
    edge_vectors_declaration = (
        "const Scalar* edge_vectors = static_cast<const Scalar*>(args.edge_vectors);"
        if input_mode == "inline"
        else ""
    )
    staged_harmonics_declaration = (
        "const Scalar* staged_harmonics = "
        "static_cast<const Scalar*>(args.edge_harmonics);"
        if input_mode == "staged"
        else ""
    )
    return f"""// Forward: one CTA owns a receiver; output multiplicity lanes own
// compact register accumulators, including repeated output-block instructions.
extern "C" __global__ void __launch_bounds__({threads})
symmetrix_oam_forward_kernel_{index}(SymmetrixOAMCudaLayerArgsV2 args)
{{
    const Scalar* source_values = static_cast<const Scalar*>(args.source_node_values);
    {edge_vectors_declaration}
    {staged_harmonics_declaration}
    const Scalar* coefficients = static_cast<const Scalar*>(args.spline_coefficients);
    Scalar* messages = static_cast<Scalar*>(args.node_messages);
    const Scalar* output_mask = oam_output_mask_{index};
    struct ForwardScratch {{
        Scalar source[{source_dimension}];
        alignas(16) Scalar harmonics[{harmonic_count}];
        double coordinate;
        int interval;
    }};
    static_assert(sizeof(ForwardScratch) == {row["forward_static_shared_bytes"]});
    __shared__ ForwardScratch scratch;
    for (std::int64_t receiver = static_cast<std::int64_t>(blockIdx.x);
         receiver < args.num_nodes; receiver += static_cast<std::int64_t>(gridDim.x))
    {{
        const std::int64_t first = args.receiver_offsets[receiver];
        const std::int64_t last = args.receiver_offsets[receiver + 1];
{chr(10).join(rendered_groups)}
    }}
}}"""


def _render_cooperative_instruction_reverse(
    layer: Mapping[str, Any],
    instruction: Mapping[str, Any],
    weight_offset: int,
    scalar: str,
    radial_adjoint_mode: str,
) -> str:
    """Lower disjoint channel owners without changing host reverse semantics.

    Each weight adjoint rounds in Scalar before its immediate double radial
    contraction. All uvw output channels remain under the same input owner.
    """
    product = layer["edge_tensor_product"]
    operands = (
        ("irreps_in1", "i_in1"),
        ("irreps_in2", "i_in2"),
        ("irreps_out", "i_out"),
    )
    blocks = [
        _irrep_blocks(product[key])[int(instruction[index])] for key, index in operands
    ]
    o1, o2, oo = [
        block_offsets(_irrep_blocks(product[key]))[int(instruction[index])]
        for key, index in operands
    ]
    m1, m2, mo = (int(block["multiplicity"]) for block in blocks)
    d1, d2, do = (2 * int(block["l"]) + 1 for block in blocks)
    mode = instruction["connection_mode"]
    out_m = "c" if mode == "uvw" else "a"
    c_extent = mo if mode == "uvw" else 1
    weight_index = {
        "uvu": f"{weight_offset} + a*{m2} + b",
        "uuu": f"{weight_offset} + a",
        "uvw": f"{weight_offset} + (a*{m2}+b)*{mo}+c",
    }[mode]
    has_weight = bool(instruction["has_weight"])
    weight = (
        f"oam_spline_value(coefficient_base, args.spline_function_count, {weight_index}, coordinate)"
        if has_weight
        else f"{scalar}(1)"
    )
    lines = [
        f"for (std::int64_t a=threadIdx.x; a<{m1}; a+=blockDim.x) {{",
        f"    {scalar} ds_row[{d1}] = {{}};",
        f"    for (std::int64_t b=0; b<{m2}; ++b) {{",
    ]
    if mode == "uuu":
        lines.append("        if (a != b) continue;")
    lines.extend(
        [
            f"        {scalar} de_row[{d2}] = {{}};",
            f"        for (std::int64_t c=0; c<{c_extent}; ++c) {{",
            f"            const {scalar} weight = {weight};",
        ]
    )
    if has_weight:
        lines.append(f"            {scalar} dw_acc = {scalar}(0);")
    path = instruction["wigner_3j"]["values"]
    factor = float(instruction["path_weight"])
    for x in range(d1):
        for y in range(d2):
            for z in range(do):
                value = float(path[(x * d2 + y) * do + z])
                if value == 0.0:
                    continue
                literal = scalar_literal(factor * value, scalar)
                lines.extend(
                    [
                        f"            {{ const {scalar} coefficient = adjoint_lane[{oo} + {z}*{mo} + {out_m}] * {literal};",
                        f"              ds_row[{x}] += coefficient * edge[{o2} + b*{d2} + {y}] * weight;",
                        f"              de_row[{y}] += coefficient * src[{o1} + {x}*{m1} + a] * weight;",
                    ]
                )
                if has_weight:
                    lines.append(
                        f"              dw_acc += coefficient * src[{o1} + {x}*{m1} + a] * edge[{o2} + b*{d2} + {y}];"
                    )
                lines.append("            }")
    if has_weight:
        derivative = (
            "oam_spline_derivative(coefficient_base, "
            f"args.spline_function_count, {weight_index}, coordinate)"
        )
        if radial_adjoint_mode == "fp64":
            lines.append(
                "            radial_adjoint += static_cast<double>(dw_acc) "
                f"* static_cast<double>({derivative});"
            )
        elif radial_adjoint_mode == "compensated_fp32":
            lines.extend(
                [
                    f"            const {scalar} radial_term = dw_acc * {derivative};",
                    f"            const {scalar} radial_delta = radial_term - radial_compensation;",
                    f"            const {scalar} radial_next = radial_adjoint + radial_delta;",
                    (
                        "            radial_compensation = "
                        "(radial_next - radial_adjoint) - radial_delta;"
                    ),
                    "            radial_adjoint = radial_next;",
                ]
            )
        else:
            lines.append(f"            radial_adjoint += dw_acc * {derivative};")
    lines.extend(
        [
            "        }",
            f"        for (int y=0; y<{d2}; ++y) edge_adjoint[{o2} + b*{d2} + y] += de_row[y];",
            "    }",
            f"    for (int x=0; x<{d1}; ++x) source_row[{o1} + x*{m1} + a] = source_row[{o1} + x*{m1} + a] + ds_row[x];",
            "}",
        ]
    )
    return "\n".join(lines)


def _render_grouped_reverse_kernels(
    layer: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
    scalar: str,
    threads: int,
    radial_adjoint_mode: str,
) -> str:
    """Render one source-owner reverse kernel per source-input irrep group."""

    product = layer["edge_tensor_product"]
    source_dimension = int(row["source_dimension"])
    target_dimension = int(row["target_dimension"])
    edge_dimension = int(row["edge_dimension"])
    harmonic_map = harmonic_indices(_irrep_blocks(layer["edge_irreps"]))
    harmonic_count = int(row["edge_harmonic_count"])
    groups = _reverse_source_groups(layer)
    edge_loads = [
        f"                harmonics[{harmonic}]," for harmonic in harmonic_map
    ]
    geometry_loads = [
        f"                angular_force_{axis} += edge_adjoint[{position}]"
        f" * gradient_{axis}[{harmonic}];"
        for position, harmonic in enumerate(harmonic_map)
        for axis in ("x", "y", "z")
    ]
    radial_scratch_type = (
        "double" if radial_adjoint_mode in ("fp64", "hierarchical_fp32") else "Scalar"
    )
    radial_lane_declaration = (
        "double radial_adjoint = 0.0;"
        if radial_adjoint_mode == "fp64"
        else "Scalar radial_adjoint = Scalar(0);"
    )
    compensation_declaration = (
        "\n                Scalar radial_compensation = Scalar(0);"
        if radial_adjoint_mode == "compensated_fp32"
        else ""
    )
    radial_warp_store = (
        "static_cast<double>(radial_adjoint)"
        if radial_adjoint_mode == "hierarchical_fp32"
        else "radial_adjoint"
    )
    radial_total_declaration = (
        f"double radial_adjoint_total = threadIdx.x < {threads // 32} "
        "? scratch.radial[threadIdx.x] : 0.0;"
        if radial_adjoint_mode in ("fp64", "hierarchical_fp32")
        else f"Scalar radial_adjoint_total = threadIdx.x < {threads // 32} "
        "? scratch.radial[threadIdx.x] : Scalar(0);"
    )
    if radial_adjoint_mode in ("fp64", "hierarchical_fp32"):
        radial_geometry = """const double radial_scale = radial_adjoint_total / args.distances[directed_edge];
                        geometry_partial[0] = static_cast<Scalar>(radial_scale * static_cast<double>(vector[0])) + angular_x;
                        geometry_partial[1] = static_cast<Scalar>(radial_scale * static_cast<double>(vector[1])) + angular_y;
                        geometry_partial[2] = static_cast<Scalar>(radial_scale * static_cast<double>(vector[2])) + angular_z;"""
    else:
        radial_geometry = """const Scalar radial_scale = radial_adjoint_total
                            / static_cast<Scalar>(args.distances[directed_edge]);
                        geometry_partial[0] = radial_scale * vector[0] + angular_x;
                        geometry_partial[1] = radial_scale * vector[1] + angular_y;
                        geometry_partial[2] = radial_scale * vector[2] + angular_z;"""
    kernels: list[str] = []
    for group_index, group in enumerate(groups):
        source_offset = int(group["source_offset"])
        source_extent = int(group["source_multiplicity"]) * int(group["source_width"])
        bodies = [
            _render_cooperative_instruction_reverse(
                layer,
                product["instructions"][instruction_index],
                weight_offset,
                scalar,
                radial_adjoint_mode,
            )
            for instruction_index, weight_offset in zip(
                group["instruction_indices"], group["weight_offsets"]
            )
        ]
        kernels.append(
            f"""// Grouped reverse experiment: this kernel owns one disjoint
// source-input irrep block. Source adjoints are written directly; geometry is
// retained per group for one following stream-ordered reduction.
extern "C" __global__ void __launch_bounds__({threads})
symmetrix_oam_reverse_kernel_{index}_group_{group_index}(SymmetrixOAMCudaLayerArgsV2 args)
{{
    const Scalar* source_values = static_cast<const Scalar*>(args.source_node_values);
    const Scalar* adjoint_values = static_cast<const Scalar*>(args.target_node_output_adjoint);
    const Scalar* edge_vectors = static_cast<const Scalar*>(args.edge_vectors);
    const Scalar* staged_harmonics = static_cast<const Scalar*>(args.edge_harmonics);
    const Scalar* staged_gradients = static_cast<const Scalar*>(args.reverse_harmonic_gradients);
    const Scalar* coefficients = static_cast<const Scalar*>(args.spline_coefficients);
    Scalar* source_adjoint = static_cast<Scalar*>(args.source_node_input_adjoint);
    Scalar* geometry_partials = static_cast<Scalar*>(args.reverse_geometry_partials);
    struct ReverseScratch {{
        Scalar source[{source_dimension}];
        Scalar angular[{threads // 32}][3];
        {radial_scratch_type} radial[{threads // 32}];
        double coordinate;
        int interval;
    }};
    __shared__ ReverseScratch scratch;
    volatile Scalar* source_row = scratch.source;
    for (std::int64_t source = static_cast<std::int64_t>(blockIdx.x);
         source < static_cast<std::int64_t>(args.num_nodes);
         source += static_cast<std::int64_t>(gridDim.x))
    {{
            for (int position = static_cast<int>(threadIdx.x);
                 position < {source_extent}; position += blockDim.x)
                source_row[{source_offset} + position] = Scalar(0);
            __syncthreads();
            const std::int64_t begin = args.source_edge_offsets[source];
            const std::int64_t end = args.source_edge_offsets[source + 1];
            const Scalar* src = source_values + source * {source_dimension};
            for (std::int64_t slot = begin; slot < end; ++slot)
            {{
                __syncthreads();
                const std::int64_t directed_edge = args.source_edge_indices[slot];
                const std::int64_t target = args.receiver_indices[directed_edge];
                const Scalar* vector = edge_vectors + directed_edge * 3;
                const Scalar* harmonics = staged_harmonics
                    + directed_edge * {harmonic_count};
                const Scalar* gradient_x = staged_gradients
                    + directed_edge * {3 * harmonic_count};
                const Scalar* gradient_y = gradient_x + {harmonic_count};
                const Scalar* gradient_z = gradient_y + {harmonic_count};
                const Scalar* adjoint_lane = adjoint_values
                    + target * {target_dimension};
                {render_cooperative_spline_block("double" if radial_adjoint_mode == "fp64" else "Scalar")}
                Scalar edge[{edge_dimension}] = {{
{chr(10).join(edge_loads)}
                }};
                Scalar edge_adjoint[{edge_dimension}] = {{}};
                {radial_lane_declaration}{compensation_declaration}
{chr(10).join("                " + line for body in bodies for line in body.splitlines())}
                Scalar angular_force_x = Scalar(0);
                Scalar angular_force_y = Scalar(0);
                Scalar angular_force_z = Scalar(0);
{chr(10).join(geometry_loads)}
                for (int offset = 16; offset > 0; offset /= 2) {{
                    angular_force_x += __shfl_down_sync(0xffffffffu, angular_force_x, offset);
                    angular_force_y += __shfl_down_sync(0xffffffffu, angular_force_y, offset);
                    angular_force_z += __shfl_down_sync(0xffffffffu, angular_force_z, offset);
                    radial_adjoint += __shfl_down_sync(0xffffffffu, radial_adjoint, offset);
                }}
                const int warp = threadIdx.x / 32;
                if (threadIdx.x % 32 == 0) {{
                    scratch.angular[warp][0] = angular_force_x;
                    scratch.angular[warp][1] = angular_force_y;
                    scratch.angular[warp][2] = angular_force_z;
                    scratch.radial[warp] = {radial_warp_store};
                }}
                __syncthreads();
                if (threadIdx.x < 32) {{
                    Scalar angular_x = threadIdx.x < {threads // 32} ? scratch.angular[threadIdx.x][0] : Scalar(0);
                    Scalar angular_y = threadIdx.x < {threads // 32} ? scratch.angular[threadIdx.x][1] : Scalar(0);
                    Scalar angular_z = threadIdx.x < {threads // 32} ? scratch.angular[threadIdx.x][2] : Scalar(0);
                    {radial_total_declaration}
                    for (int offset = 16; offset > 0; offset /= 2) {{
                        angular_x += __shfl_down_sync(0xffffffffu, angular_x, offset);
                        angular_y += __shfl_down_sync(0xffffffffu, angular_y, offset);
                        angular_z += __shfl_down_sync(0xffffffffu, angular_z, offset);
                        radial_adjoint_total += __shfl_down_sync(0xffffffffu, radial_adjoint_total, offset);
                    }}
                    if (threadIdx.x == 0) {{
                        Scalar* geometry_partial = geometry_partials
                            + (static_cast<std::int64_t>({group_index}) * args.num_edges
                                + directed_edge) * 3;
                        {radial_geometry}
                    }}
                }}
                __syncthreads();
            }}
            for (int position = static_cast<int>(threadIdx.x);
                 position < {source_extent}; position += blockDim.x)
                source_adjoint[source * {source_dimension} + {source_offset} + position]
                    = source_row[{source_offset} + position];
            __syncthreads();
    }}
}}"""
        )
    return "\n\n".join(kernels)


def _render_reverse_kernel(
    layer: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
    scalar: str,
    threads: int,
    radial_adjoint_mode: str = "fp64",
    group_execution: str = "serial",
) -> str:
    if group_execution == "source_groups":
        return _render_grouped_reverse_kernels(
            layer,
            row,
            index,
            scalar,
            threads,
            radial_adjoint_mode=radial_adjoint_mode,
        )
    if group_execution != "serial":
        raise ValueError("unsupported reverse group execution")
    product = layer["edge_tensor_product"]
    source_dimension = int(row["source_dimension"])
    target_dimension = int(row["target_dimension"])
    edge_dimension = int(row["edge_dimension"])
    harmonic_map = harmonic_indices(_irrep_blocks(layer["edge_irreps"]))
    harmonic_count = int(row["edge_harmonic_count"])
    bodies: list[str] = []
    weight_offset = 0
    for instruction in product["instructions"]:
        bodies.append(
            _render_cooperative_instruction_reverse(
                layer, instruction, weight_offset, scalar, radial_adjoint_mode
            )
        )
        if instruction["has_weight"]:
            weight_offset += math.prod(
                int(extent) for extent in instruction["path_shape"]
            )
    edge_loads = [f"            harmonics[{harmonic}]," for harmonic in harmonic_map]
    geometry_loads = [
        f"            angular_force_{axis} += edge_adjoint[{position}]"
        f" * gradient_{axis}[{harmonic}];"
        for position, harmonic in enumerate(harmonic_map)
        for axis in ("x", "y", "z")
    ]
    radial_scratch_type = (
        "double" if radial_adjoint_mode in ("fp64", "hierarchical_fp32") else "Scalar"
    )
    radial_lane_declaration = (
        "double radial_adjoint = 0.0;"
        if radial_adjoint_mode == "fp64"
        else "Scalar radial_adjoint = Scalar(0);"
    )
    compensation_declaration = (
        "\n            Scalar radial_compensation = Scalar(0);"
        if radial_adjoint_mode == "compensated_fp32"
        else ""
    )
    radial_warp_store = (
        "static_cast<double>(radial_adjoint)"
        if radial_adjoint_mode == "hierarchical_fp32"
        else "radial_adjoint"
    )
    radial_total_declaration = (
        f"double radial_adjoint_total = threadIdx.x < {threads // 32} "
        "? scratch.radial[threadIdx.x] : 0.0;"
        if radial_adjoint_mode in ("fp64", "hierarchical_fp32")
        else f"Scalar radial_adjoint_total = threadIdx.x < {threads // 32} "
        "? scratch.radial[threadIdx.x] : Scalar(0);"
    )
    if radial_adjoint_mode in ("fp64", "hierarchical_fp32"):
        radial_geometry = """const double radial_scale = radial_adjoint_total / args.distances[directed_edge];
                    geometry_adjoint[static_cast<std::int64_t>(3) * directed_edge + 0] +=
                        static_cast<Scalar>(radial_scale * static_cast<double>(vector[0])) + angular_x;
                    geometry_adjoint[static_cast<std::int64_t>(3) * directed_edge + 1] +=
                        static_cast<Scalar>(radial_scale * static_cast<double>(vector[1])) + angular_y;
                    geometry_adjoint[static_cast<std::int64_t>(3) * directed_edge + 2] +=
                        static_cast<Scalar>(radial_scale * static_cast<double>(vector[2])) + angular_z;"""
    else:
        radial_geometry = """const Scalar radial_scale = radial_adjoint_total
                        / static_cast<Scalar>(args.distances[directed_edge]);
                    geometry_adjoint[static_cast<std::int64_t>(3) * directed_edge + 0] +=
                        radial_scale * vector[0] + angular_x;
                    geometry_adjoint[static_cast<std::int64_t>(3) * directed_edge + 1] +=
                        radial_scale * vector[1] + angular_y;
                    geometry_adjoint[static_cast<std::int64_t>(3) * directed_edge + 2] +=
                        radial_scale * vector[2] + angular_z;"""
    return f"""// Reverse: one cooperative CTA owns each source at a time and
// advances by a grid stride. Input multiplicity lanes are disjoint owners;
// geometry is reduced once and written by CTA thread zero.
extern "C" __global__ void __launch_bounds__({threads})
symmetrix_oam_reverse_kernel_{index}(SymmetrixOAMCudaLayerArgsV2 args)
{{
    const Scalar* source_values = static_cast<const Scalar*>(args.source_node_values);
    const Scalar* adjoint_values =
        static_cast<const Scalar*>(args.target_node_output_adjoint);
    const Scalar* edge_vectors = static_cast<const Scalar*>(args.edge_vectors);
    const Scalar* coefficients = static_cast<const Scalar*>(args.spline_coefficients);
    Scalar* source_adjoint = static_cast<Scalar*>(args.source_node_input_adjoint);
    Scalar* geometry_adjoint = static_cast<Scalar*>(args.geometry_adjoint);
    struct ReverseScratch {{
        Scalar source[{source_dimension}];
        // Wide harmonic loads must not include the mutable source row or partials.
        alignas(16) Scalar harmonics[{harmonic_count}];
        Scalar gradient_x[{harmonic_count}];
        Scalar gradient_y[{harmonic_count}];
        Scalar gradient_z[{harmonic_count}];
        Scalar angular[{threads // 32}][3];
        {radial_scratch_type} radial[{threads // 32}];
        double coordinate;
        int interval;
    }};
    static_assert(sizeof(ReverseScratch) == {row["reverse_static_shared_bytes"]});
    __shared__ ReverseScratch scratch;
    Scalar* harmonics_scratch = scratch.harmonics;
    Scalar* gradient_x_scratch = scratch.gradient_x;
    Scalar* gradient_y_scratch = scratch.gradient_y;
    Scalar* gradient_z_scratch = scratch.gradient_z;
    for (std::int64_t source = static_cast<std::int64_t>(blockIdx.x);
         source < static_cast<std::int64_t>(args.num_nodes);
         source += static_cast<std::int64_t>(gridDim.x))
    {{
        // Keep loads Scalar-width and owner-predicated: speculative wide loads
        // can otherwise include a neighbor's concurrently updated shared value.
        volatile Scalar* source_row = scratch.source;
        for (std::int64_t position = static_cast<std::int64_t>(threadIdx.x);
             position < {source_dimension}; position += blockDim.x)
            source_row[position] = Scalar(0);
        __syncthreads();
        const std::int64_t begin = args.source_edge_offsets[source];
        const std::int64_t end = args.source_edge_offsets[source + 1];
        const Scalar* src = source_values
            + static_cast<std::int64_t>(source) * {source_dimension};
        for (std::int64_t slot = begin; slot < end; ++slot)
        {{
            __syncthreads();
            const std::int64_t directed_edge = args.source_edge_indices[slot];
            const std::int64_t target = args.receiver_indices[directed_edge];
            const Scalar* vector = edge_vectors
                + static_cast<std::int64_t>(directed_edge) * 3;
            if (threadIdx.x == 0) {{
                oam_compute_harmonics(
                    vector[0], vector[1], vector[2], harmonics_scratch);
                oam_compute_harmonic_gradients(
                    vector[0], vector[1], vector[2], gradient_x_scratch,
                    gradient_y_scratch, gradient_z_scratch);
            }}
            __syncthreads();
            const Scalar* harmonics = harmonics_scratch;
            const Scalar* gradient_x = gradient_x_scratch;
            const Scalar* gradient_y = gradient_y_scratch;
            const Scalar* gradient_z = gradient_z_scratch;
            const Scalar* adjoint_lane = adjoint_values
                + static_cast<std::int64_t>(target) * {target_dimension};
            {render_cooperative_spline_block("double" if radial_adjoint_mode == "fp64" else "Scalar")}
            Scalar edge[{edge_dimension}] = {{
{chr(10).join(edge_loads)}
            }};
            Scalar edge_adjoint[{edge_dimension}];
            for (int position = 0; position < {edge_dimension}; ++position)
                edge_adjoint[position] = Scalar(0);
            {radial_lane_declaration}{compensation_declaration}
{chr(10).join("            " + line for body in bodies for line in body.splitlines())}
            Scalar angular_force_x = Scalar(0);
            Scalar angular_force_y = Scalar(0);
            Scalar angular_force_z = Scalar(0);
{chr(10).join(geometry_loads)}
            for (int offset = 16; offset > 0; offset /= 2) {{
                angular_force_x += __shfl_down_sync(0xffffffffu, angular_force_x, offset);
                angular_force_y += __shfl_down_sync(0xffffffffu, angular_force_y, offset);
                angular_force_z += __shfl_down_sync(0xffffffffu, angular_force_z, offset);
                radial_adjoint += __shfl_down_sync(0xffffffffu, radial_adjoint, offset);
            }}
            const int warp = threadIdx.x / 32;
            if (threadIdx.x % 32 == 0) {{
                scratch.angular[warp][0] = angular_force_x;
                scratch.angular[warp][1] = angular_force_y;
                scratch.angular[warp][2] = angular_force_z;
                scratch.radial[warp] = {radial_warp_store};
            }}
            __syncthreads();
            if (threadIdx.x < 32) {{
                Scalar angular_x = threadIdx.x < {threads // 32} ? scratch.angular[threadIdx.x][0] : Scalar(0);
                Scalar angular_y = threadIdx.x < {threads // 32} ? scratch.angular[threadIdx.x][1] : Scalar(0);
                Scalar angular_z = threadIdx.x < {threads // 32} ? scratch.angular[threadIdx.x][2] : Scalar(0);
                {radial_total_declaration}
                for (int offset = 16; offset > 0; offset /= 2) {{
                    angular_x += __shfl_down_sync(0xffffffffu, angular_x, offset);
                    angular_y += __shfl_down_sync(0xffffffffu, angular_y, offset);
                    angular_z += __shfl_down_sync(0xffffffffu, angular_z, offset);
                    radial_adjoint_total += __shfl_down_sync(0xffffffffu, radial_adjoint_total, offset);
                }}
                if (threadIdx.x == 0) {{
                    {radial_geometry}
                }}
            }}
            __syncthreads();
        }}
        Scalar* destination = source_adjoint + source * {source_dimension};
        for (std::int64_t position = static_cast<std::int64_t>(threadIdx.x);
             position < {source_dimension}; position += blockDim.x)
            destination[position] = source_row[position];
        __syncthreads();
    }}
}}"""


def _render_launchers(
    index: int,
    threads: int,
    staged_forward: bool = False,
    reverse_group_count: int = 0,
) -> str:
    staged_pointer_check = (
        "\n                || args->edge_harmonics == nullptr" if staged_forward else ""
    )
    grouped = reverse_group_count > 0
    grouped_pointer_check = (
        """
        || (args->num_edges > 0 && (args->reverse_geometry_partials == nullptr
            || args->edge_harmonics == nullptr
            || args->reverse_harmonic_gradients == nullptr))"""
        if grouped
        else ""
    )
    reverse_comment = (
        "Each specialized source-irrep kernel uses the same persistent source grid."
        if grouped
        else "One CTA owns one source; persistent blocks are the owner count cap."
    )
    reverse_launches = (
        "\n".join(
            f"""    symmetrix_oam_reverse_kernel_{index}_group_{group}<<<
        static_cast<std::uint32_t>(blocks), threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);"""
            for group in range(reverse_group_count)
        )
        if grouped
        else f"""    symmetrix_oam_reverse_kernel_{index}<<<
        static_cast<std::uint32_t>(blocks), threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);"""
    )
    return f"""std::int32_t oam_forward_launch_{index}(
    const SymmetrixOAMCudaLayerArgsV2* args,
    void* stream,
    std::int32_t persistent_blocks)
{{
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixOAMCudaLayerArgsV2)
        || args->layer_index != {index}u
        || args->flags != SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1
        || args->reserved != 0u
        || args->num_nodes < 0 || args->num_edges < 0
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_nodes == 0)
        return static_cast<std::int32_t>(cudaSuccess);
    if (args->receiver_offsets == nullptr
        || args->source_node_values == nullptr
        || args->node_messages == nullptr
        || (args->num_edges > 0
            && (args->source_indices == nullptr
                || args->distances == nullptr
                || args->spline_coefficients == nullptr
                || args->edge_vectors == nullptr{staged_pointer_check})))
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = {FORWARD_THREADS_PER_BLOCK};
    // One CTA owns one receiver; persistent blocks cap the receiver owners.
    const std::int64_t blocks =
        args->num_nodes < persistent_blocks ? args->num_nodes : persistent_blocks;
    const auto launch_args = *args;
    symmetrix_oam_forward_kernel_{index}<<<
        static_cast<std::uint32_t>(blocks), threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(launch_args);
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}

std::int32_t oam_reverse_launch_{index}(
    const SymmetrixOAMCudaLayerArgsV2* args,
    void* stream,
    std::int32_t persistent_blocks)
{{
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixOAMCudaLayerArgsV2)
        || args->layer_index != {index}u
        || args->flags != SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1
        || args->reserved != 0u
        || args->num_nodes < 0 || args->num_edges < 0
        || persistent_blocks <= 0)
        return static_cast<std::int32_t>(cudaErrorInvalidValue);
    if (args->num_nodes == 0)
        return static_cast<std::int32_t>(cudaSuccess);
    if (args->receiver_offsets == nullptr
        || args->source_edge_offsets == nullptr
        || args->source_node_values == nullptr
        || args->target_node_output_adjoint == nullptr
        || args->source_node_input_adjoint == nullptr
        || (args->num_edges > 0
            && (args->source_indices == nullptr
                || args->receiver_indices == nullptr
                || args->source_edge_indices == nullptr
                || args->distances == nullptr
                || args->spline_coefficients == nullptr
                || args->edge_vectors == nullptr
                || args->geometry_adjoint == nullptr)){grouped_pointer_check})
        return static_cast<std::int32_t>(cudaErrorInvalidDevicePointer);
    constexpr std::int32_t threads = {threads};
    // {reverse_comment}
    const std::int64_t blocks =
        args->num_nodes < persistent_blocks ? args->num_nodes : persistent_blocks;
    const auto launch_args = *args;
{reverse_launches}
    return static_cast<std::int32_t>(cudaPeekAtLastError());
}}

"""


def _render_layer_constants(
    layer: Mapping[str, Any], row: Mapping[str, Any], index: int, scalar: str
) -> str:
    product = layer["edge_tensor_product"]
    output_dimension = int(row["output_dimension"])
    output_mask = product["output_mask"]["values"]
    lane_index = lane_column_permutation(_irrep_blocks(layer["tp_output_irreps"]))
    if len(lane_index) != output_dimension or len(output_mask) != output_dimension:
        raise ValueError("tensor-product output irrep layout is inconsistent")
    mask = ",".join(scalar_literal(float(value), scalar) for value in output_mask)
    return f"__device__ static const {scalar} oam_output_mask_{index}[] = {{{mask}}};\n"


def _render_descriptor(metadata: Mapping[str, Any], scalar: str) -> str:
    rows = []
    for row in metadata["layers"]:
        rows.append(
            "{{sizeof(SymmetrixOAMCudaLayerV2), 0u, {index}, {source}, {edge}, "
            "{weight}, {output}, {target}, {instructions}, {intervals}, "
            "{functions}, {harmonics}, {forward_threads}, {reverse_threads}, "
            "{reverse_groups}, {segment_edges}, {normalization}, 0, 0, 0ull, 0, 0}}".format(
                index=row["layer"],
                source=row["source_dimension"],
                edge=row["edge_dimension"],
                weight=row["weight_dimension"],
                output=row["output_dimension"],
                target=row["target_dimension"],
                instructions=row["instruction_count"],
                intervals=row["spline_interval_count"],
                functions=row["spline_function_count"],
                harmonics=row["edge_harmonic_count"],
                forward_threads=metadata["forward_threads_per_block"],
                reverse_threads=metadata["threads_per_block"],
                reverse_groups=row["reverse_group_count"],
                segment_edges=row["reverse_group_flags"],
                normalization=scalar_literal(row["normalization_factor"], scalar),
            )
        )
    layer_array = ",\n".join(f"    {text}" for text in rows)
    launches = ",\n".join(
        f"    {{sizeof(SymmetrixOAMCudaLayerLaunchesV2), {index}u, 0u, 0u,"
        f" &oam_forward_launch_{index}, &oam_reverse_launch_{index},"
        " nullptr, nullptr}"
        for index in range(int(metadata["layer_count"]))
    )
    return f"""static const SymmetrixOAMCudaLayerV2 oam_cuda_layers[] = {{
{layer_array}
}};
static const SymmetrixOAMCudaLayerLaunchesV2 oam_cuda_launches[] = {{
{launches}
}};
static const SymmetrixOAMCudaPluginV3 oam_cuda_plugin = {{
    SYMMETRIX_OAM_CUDA_PLUGIN_ABI_VERSION,
    sizeof(SymmetrixOAMCudaPluginV3),
    sizeof(void*),
    SYMMETRIX_OAM_CUDA_PLUGIN_BYTE_ORDER,
    {int(metadata["required_capabilities"])}u,
    {int(metadata["scalar_kind"])}u,
    {int(metadata["scalar_size"])}u,
    {int(metadata["layer_count"])}u,
    0u,
    0u,
    SYMMETRIX_OAM_CUDA_PLUGIN_ABI_TAG,
    {cpp_string_literal(metadata["artifact_id"])},
    {cpp_string_literal(metadata["contract_fingerprint"])},
    {cpp_string_literal(metadata["semantic_fingerprint"])},
    {cpp_string_literal(metadata["structure_fingerprint"])},
    {cpp_string_literal(metadata["target_architecture"])},
    {int(metadata["target_compute_capability"])},
    {int(metadata["max_threads_per_block"])},
    "full_message",
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    {cpp_string_literal(metadata["forward_owner_schedule"])},
    {cpp_string_literal(metadata["reverse_owner_schedule"])},
    {cpp_string_literal(metadata["forward_channel_policy"])},
    "generated-direct-harmonic-gradient-v1",
    "descriptor-layer-threads-v1",
    "descriptor-static-bytes-v1",
    "compiler-managed-v1",
    oam_cuda_layers,
    oam_cuda_launches,
    0,
    0,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    {cpp_string_literal(metadata["target_policy"])},
    {cpp_string_literal(metadata["precision_policy"])},
    {cpp_string_literal(metadata["schedule_qualification"])},
}};

extern "C" SYMMETRIX_OAM_CUDA_PLUGIN_EXPORT
const SymmetrixOAMCudaPluginV3* symmetrix_oam_cuda_plugin_query_v3()
{{
    return &oam_cuda_plugin;
}}"""


def _render_module_identity(metadata: Mapping[str, Any], scalar: str) -> str:
    rows = []
    for row in metadata["layers"]:
        rows.append(
            "{{sizeof(SymmetrixOAMCudaLayerV2), 0u, {index}, {source}, {edge}, "
            "{weight}, {output}, {target}, {instructions}, {intervals}, "
            "{functions}, {harmonics}, {forward_threads}, {reverse_threads}, "
            "{reverse_groups}, {segment_edges}, {normalization}, 0, 0, 0ull, 0, 0}}".format(
                index=row["layer"],
                source=row["source_dimension"],
                edge=row["edge_dimension"],
                weight=row["weight_dimension"],
                output=row["output_dimension"],
                target=row["target_dimension"],
                instructions=row["instruction_count"],
                intervals=row["spline_interval_count"],
                functions=row["spline_function_count"],
                harmonics=row["edge_harmonic_count"],
                forward_threads=metadata["forward_threads_per_block"],
                reverse_threads=metadata["threads_per_block"],
                reverse_groups=row["reverse_group_count"],
                segment_edges=row["reverse_group_flags"],
                normalization=scalar_literal(row["normalization_factor"], scalar),
            )
        )
    layer_array = ",\n".join(f"    {text}" for text in rows)
    return f"""extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_abi_version = SYMMETRIX_OAM_CUDA_PLUGIN_ABI_VERSION;
extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_struct_size = sizeof(SymmetrixOAMCudaPluginV3);
extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_pointer_size = sizeof(void*);
extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_byte_order = SYMMETRIX_OAM_CUDA_PLUGIN_BYTE_ORDER;
extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_capabilities = {int(metadata["required_capabilities"])}u;
extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_scalar_kind = {int(metadata["scalar_kind"])}u;
extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_scalar_size = {int(metadata["scalar_size"])}u;
extern "C" __device__ __constant__ unsigned int
symmetrix_oam_cuda_module_layer_count = {int(metadata["layer_count"])}u;
extern "C" __device__ __constant__ int
symmetrix_oam_cuda_module_target_compute_capability =
    {int(metadata["target_compute_capability"])};
extern "C" __device__ __constant__ int
symmetrix_oam_cuda_module_max_threads_per_block =
    {int(metadata["max_threads_per_block"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_abi_tag[] = {cpp_string_literal(metadata["abi"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_artifact_id[] = {cpp_string_literal(metadata["artifact_id"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_contract_fingerprint[] =
    {cpp_string_literal(metadata["contract_fingerprint"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_semantic_fingerprint[] =
    {cpp_string_literal(metadata["semantic_fingerprint"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_structure_fingerprint[] =
    {cpp_string_literal(metadata["structure_fingerprint"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_target_architecture[] =
    {cpp_string_literal(metadata["target_architecture"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_execution_variants[] = "full_message";
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_full_forward_schedule[] = {cpp_string_literal(metadata["forward_owner_schedule"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_full_reverse_schedule[] = {cpp_string_literal(metadata["reverse_owner_schedule"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_full_vector_policy[] = {cpp_string_literal(metadata["forward_channel_policy"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_direct_geometry_policy[] = "generated-direct-harmonic-gradient-v1";
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_full_tiling_policy[] = "descriptor-layer-threads-v1";
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_full_shared_memory_policy[] = "descriptor-static-bytes-v1";
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_full_register_budget_policy[] = "compiler-managed-v1";
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_target_policy[] = {cpp_string_literal(metadata["target_policy"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_precision_policy[] = {cpp_string_literal(metadata["precision_policy"])};
extern "C" __device__ __constant__ char
symmetrix_oam_cuda_module_schedule_qualification[] = {cpp_string_literal(metadata["schedule_qualification"])};
extern "C" __device__ __constant__ SymmetrixOAMCudaLayerV2
symmetrix_oam_cuda_module_layers[] = {{
{layer_array}
}};"""


def render_oam_cuda_plugin(
    contract: Mapping[str, Any],
    target: Any,
    precision: str = "float32",
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    generation_policy: OAMCudaGenerationPolicy | None = None,
    *,
    architecture: Mapping[str, Any] | None = None,
) -> str:
    """Render one self-contained CUDA translation unit for an OAM contract."""

    policy = _normalized_generation_policy(
        generation_policy, precision=precision, target=target, contract=contract
    )
    metadata = oam_cuda_artifact_metadata(
        contract,
        precision=precision,
        target=target,
        threads_per_block=threads_per_block,
        generation_policy=policy,
        architecture=architecture,
    )
    threads = int(metadata["threads_per_block"])
    scalar = str(metadata["scalar_type"])

    harmonic_helpers = _render_harmonic_helpers(metadata, scalar)
    kernels: list[str] = []
    constants: list[str] = []
    launchers: list[str] = []
    for index, layer in enumerate(contract["layers"]):
        row = metadata["layers"][index]
        constants.append(_render_layer_constants(layer, row, index, scalar))
        kernels.append(
            _render_forward_kernel(
                layer,
                row,
                index,
                scalar,
                FORWARD_THREADS_PER_BLOCK,
                live_output_budget=policy.forward_live_output_budget,
                grouping_mode=policy.forward_grouping_mode,
                group_execution=policy.forward_group_execution,
                input_mode=policy.forward_input_mode,
                spline_coordinate_type=(
                    "double" if policy.radial_adjoint_mode == "fp64" else "Scalar"
                ),
            )
        )
        kernels.append(
            _render_reverse_kernel(
                layer,
                row,
                index,
                scalar,
                threads,
                radial_adjoint_mode=policy.radial_adjoint_mode,
                group_execution=(
                    policy.reverse_group_execution
                    if int(row["reverse_group_count"]) > 0
                    else "serial"
                ),
            )
        )
        launchers.append(
            _render_launchers(
                index,
                threads,
                staged_forward=policy.forward_input_mode == "staged",
                reverse_group_count=int(row["reverse_group_count"]),
            )
        )
    return (
        "// Generated OAM CUDA R1 plugin (ABI version 3; module ABI v3).\n"
        "#if defined(__CUDACC_RTC__)\n"
        f"{_RTC_ABI_HEADER}\n"
        "#else\n"
        f"{_ABI_HEADER}\n"
        "#endif\n"
        "#if !defined(__CUDACC_RTC__)\n"
        "#include <cuda_runtime.h>\n"
        "#include <cmath>\n"
        "#include <cstddef>\n"
        "#include <cstdint>\n"
        "#endif\n"
        "\n"
        "\n"
        f"using Scalar = {scalar};\n"
        "\n"
        + render_cooperative_spline_helpers(
            value_symbol="oam_spline_value",
            derivative_symbol="oam_spline_derivative",
            coordinate_type=(
                "double" if policy.radial_adjoint_mode == "fp64" else "Scalar"
            ),
        )
        + "\n\n"
        + harmonic_helpers
        + "\n\n"
        + "\n".join(constants)
        + "\n\n"
        + "\n\n".join(kernels)
        + "\n"
        "#if defined(__CUDACC_RTC__) || defined(SYMMETRIX_OAM_CUDA_MODULE)\n"
        + _render_module_identity(metadata, scalar)
        + "\n#else\n"
        + "\n\n".join(launchers)
        + "\n\n"
        + _render_descriptor(metadata, scalar)
        + "\n#endif\n"
    )


__all__ = [
    "OAM_CUDA_ARTIFACT_ENVELOPE",
    "OAMCudaGenerationPolicy",
    "oam_cuda_artifact_metadata",
    "recommended_oam_cuda_generation_policy",
    "render_oam_cuda_plugin",
]
