"""Standard-library-only source generation for M0, M1, and R0 operators."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .generated_operator_core import (
    R0_DEVICE_ABI_VERSION,
    R0_HOST_ABI_VERSION,
    generated_m0_identity,
    generated_m1_identity,
    generated_r0_identity,
)
from .generated_operator_core import (
    render_direct_harmonic_gradient_helper as _render_direct_harmonic_gradient_helper,
)
from .generated_operator_version import (
    R0_DEVICE_SCHEDULE_REVISION,
    R0_HOST_SCHEDULE_REVISION,
)
from .jit_codegen_common import cpp_text_literal
from .mace_execution_contract import (
    normalize_generated_m1_contract,
    normalize_standard_m0_contract,
    normalize_standard_r0_contract,
)

M0_CAPABILITIES_V1 = 0x1B
PRODUCT_ALIAS_SAFE_CAPABILITY_V1 = 1 << 2
M0_THREADS_V1 = 256
M1_CAPABILITIES_V1 = 0x0B
M1_HOST_CAPABILITIES_V1 = 0x1B
M1_SCHEDULE_REVISION = 2
M0_SCHEDULE_REVISION = 2
M1_DEVICE_CHUNK32_SCHEDULE_REVISION = 3
M1_DEVICE_TABLE_SCHEDULE_REVISION = 2
PRODUCT_TILED128_SCHEDULE = "tiled128x4-v1"
PRODUCT_TILED128_SCHEDULE_REVISION = 3
M1_DEVICE_TILED128_SCHEDULE_REVISION = 4
R0_CAPABILITIES_V1 = 0x7F00
R0_THREADS_V1 = 256
R0_COMPATIBILITY_VERSION = 2
R0_HOST_CAPABILITIES_V1 = 0x1FF
R0_HOST_FORWARD_CHANNEL_TILE_V1 = 8
R0_HOST_COORDINATE_REVERSE_CHANNEL_TILE_V1 = 128
_SOURCE_CONTENT_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _m0_device_capabilities(schedule: str) -> int:
    del schedule
    return M0_CAPABILITIES_V1


def _m0_device_schedule_revision(schedule: str) -> int:
    if schedule == PRODUCT_TILED128_SCHEDULE:
        return PRODUCT_TILED128_SCHEDULE_REVISION
    return M0_SCHEDULE_REVISION if schedule == "chunk32" else 1


def _m1_device_schedule_revision(schedule: str) -> int:
    if schedule == PRODUCT_TILED128_SCHEDULE:
        return M1_DEVICE_TILED128_SCHEDULE_REVISION
    if schedule == "chunk32":
        return M1_DEVICE_CHUNK32_SCHEDULE_REVISION
    return M1_DEVICE_TABLE_SCHEDULE_REVISION


def select_product_device_schedule(
    contract: Mapping[str, Any], *, precision: str, backend: str, target: str
) -> str:
    """Select a product schedule from canonical contract and target facts only."""

    channels = contract.get("channels")
    correlation = contract.get("correlation")
    if precision not in ("float32", "float64"):
        raise ValueError("product device precision must be 'float32' or 'float64'")
    if backend not in ("cuda", "hip"):
        raise ValueError("product device backend must be 'cuda' or 'hip'")
    if not isinstance(target, str) or not target:
        raise ValueError("product device target must not be empty")
    # tiled128x4 remains an explicit qualification control until an actual
    # default-selected 128-channel MH-1 topology clears the end-to-end gate.
    # Keep this selector topology-only and deterministic while it is demoted.
    del channels, correlation
    return "chunk32"


_M0_HOST_PLUGIN_ABI_HEADER = (
    Path(__file__)
    .with_name("mace_jit_m0_host_plugin_abi.h")
    .read_text(encoding="ascii")
    .replace("#pragma once\n", "", 1)
)
_M1_HOST_PLUGIN_ABI_HEADER = (
    _M0_HOST_PLUGIN_ABI_HEADER
    + "\n"
    + Path(__file__)
    .with_name("mace_jit_m1_host_plugin_abi.h")
    .read_text(encoding="ascii")
    .replace("#pragma once\n", "", 1)
    .replace('#include "mace_jit_m0_host_plugin_abi.h"\n', "", 1)
)
_R0_HOST_PLUGIN_ABI_HEADER = (
    Path(__file__)
    .with_name("mace_jit_r0_host_plugin_abi.h")
    .read_text(encoding="ascii")
    .replace("#pragma once\n", "", 1)
)


def _contraction_terms(contract: Mapping[str, Any]):
    return [
        (int(group["output_component"]), tuple(int(value) for value in term))
        for group in contract["monomial_groups"]
        for term in group["terms"]
    ]


def jit_m0_device_metadata(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> dict[str, Any]:
    normalized = normalize_standard_m0_contract(contract)
    if precision not in ("float32", "float64"):
        raise ValueError("M0 device precision must be 'float32' or 'float64'")
    if schedule not in ("chunk32", "table", PRODUCT_TILED128_SCHEDULE):
        raise ValueError(
            "M0 device schedule must be 'chunk32', 'table', or 'tiled128x4-v1'"
        )
    if normalized["correlation"] > 4:
        raise ValueError(
            "M0 generated modules currently support correlation no greater than four"
        )
    if not target:
        raise ValueError("M0 device target must not be empty")
    suffix = "f32" if precision == "float32" else "f64"
    structure = normalized["structure_fingerprint"]
    term_count = sum(group["term_count"] for group in normalized["monomial_groups"])
    backend = "cuda" if str(target).startswith("sm_") else "hip"
    generated_identity = generated_m0_identity(
        generation_revision=generation_version,
        structure_fingerprint=structure,
        precision=precision,
        backend=backend,
        target=str(target),
        channels=normalized["channels"],
        input_components=normalized["input_components"],
        output_components=normalized["output_components"],
        correlation=normalized["correlation"],
        term_count=term_count,
        schedule=schedule,
        schedule_revision=_m0_device_schedule_revision(schedule),
    )
    legacy_artifact_id = (
        f"jit-m0-gen{generation_version}-{suffix}-"
        f"{structure.removeprefix('sha256:')[:16]}-c{normalized['channels']}-"
        f"{schedule}-r{generated_identity['schedule_revision']}-{target}"
    )
    return {
        "schema": "symmetrix.jit.operator-module/1",
        "operator": "M0",
        "abi_version": 3,
        "artifact_id": legacy_artifact_id,
        "generated_identity": generated_identity,
        "structure_fingerprint": structure,
        "precision": precision,
        "scalar_kind": 1 if precision == "float32" else 2,
        "scalar_size": 4 if precision == "float32" else 8,
        "target": str(target),
        "schedule": schedule,
        "input_components": normalized["input_components"],
        "output_components": normalized["output_components"],
        "correlation": normalized["correlation"],
        "term_count": term_count,
        "threads_per_block": (
            128 if schedule == PRODUCT_TILED128_SCHEDULE else M0_THREADS_V1
        ),
        "schedule_revision": _m0_device_schedule_revision(schedule),
        "launch_geometry": (
            {
                "channel_tile": 128,
                "forward_output_component_tile": 1,
                "forward_node_tile": 1,
                "reverse_input_component_tile": 1,
                "reverse_node_tile": 4,
                "grid": "persistent-1d-logical-tiles",
            }
            if schedule == PRODUCT_TILED128_SCHEDULE
            else None
        ),
        "capabilities": _m0_device_capabilities(schedule),
        "contract": normalized,
    }


def jit_m1_device_metadata(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> dict[str, Any]:
    """Return the identity and launch metadata for a generated M1 module."""

    normalized = normalize_generated_m1_contract(contract)
    if precision not in ("float32", "float64"):
        raise ValueError("M1 device precision must be 'float32' or 'float64'")
    if schedule not in ("chunk32", "table", PRODUCT_TILED128_SCHEDULE):
        raise ValueError(
            "M1 device schedule must be 'chunk32', 'table', or 'tiled128x4-v1'"
        )
    if normalized["correlation"] > 4:
        raise ValueError(
            "M1 generated modules currently support correlation no greater than four"
        )
    if not target:
        raise ValueError("M1 device target must not be empty")
    term_count = sum(group["term_count"] for group in normalized["monomial_groups"])
    backend = "cuda" if str(target).startswith("sm_") else "hip"
    generated_identity = generated_m1_identity(
        generation_revision=generation_version,
        structure_fingerprint=normalized["structure_fingerprint"],
        precision=precision,
        backend=backend,
        target=str(target),
        channels=normalized["channels"],
        input_components=normalized["input_components"],
        correlation=normalized["correlation"],
        term_count=term_count,
        schedule=schedule,
        schedule_revision=_m1_device_schedule_revision(schedule),
    )
    suffix = "f32" if precision == "float32" else "f64"
    structure = normalized["structure_fingerprint"]
    return {
        "schema": "symmetrix.jit.operator-module/1",
        "operator": "M1",
        "abi_version": 3,
        "artifact_id": (
            f"jit-m1-gen{generation_version}-{suffix}-"
            f"{structure.removeprefix('sha256:')[:16]}-c{normalized['channels']}-"
            f"{schedule}-r{generated_identity['schedule_revision']}-{target}"
        ),
        "generated_identity": generated_identity,
        "structure_fingerprint": normalized["structure_fingerprint"],
        "precision": precision,
        "scalar_kind": 1 if precision == "float32" else 2,
        "scalar_size": 4 if precision == "float32" else 8,
        "target": str(target),
        "schedule": schedule,
        "schedule_revision": _m1_device_schedule_revision(schedule),
        "input_components": normalized["input_components"],
        "output_components": 1,
        "correlation": normalized["correlation"],
        "term_count": term_count,
        "threads_per_block": (
            128 if schedule == PRODUCT_TILED128_SCHEDULE else M0_THREADS_V1
        ),
        "launch_geometry": (
            {
                "channel_tile": 128,
                "forward_output_component_tile": 1,
                "forward_node_tile": 1,
                "reverse_input_component_tile": 1,
                "reverse_node_tile": 4,
                "grid": "persistent-1d-logical-tiles",
            }
            if schedule == PRODUCT_TILED128_SCHEDULE
            else None
        ),
        "capabilities": M1_CAPABILITIES_V1,
        "contract": normalized,
    }


def _contraction_device_prefix(metadata: Mapping[str, Any], *, operator: str) -> str:
    scalar = "float" if metadata["precision"] == "float32" else "double"
    operator_kind = {"M0": 1, "M1": 3}[operator]
    return f"""// Generated Symmetrix state-free {operator} operator module.
namespace std {{ using size_t = decltype(sizeof(0)); }}
using Scalar = {scalar};
constexpr int input_components = {metadata["input_components"]};
constexpr int output_components = {metadata["output_components"]};
constexpr int correlation = {metadata["correlation"]};
constexpr int term_count = {metadata["term_count"]};
constexpr int channel_count = {metadata["contract"]["channels"]};
struct {operator}Args {{
    unsigned int struct_size;
    unsigned int reserved;
    long long num_nodes;
    int channels;
    int capture_input_scale_adjoint;
    const int* node_types;
    const void* input_raw;
    const void* weights_raw;
    const void* output_adjoint_raw;
    void* output_raw;
    void* input_adjoint_raw;
    double* input_scale_adjoint;
    long long input_node_stride;
    long long output_node_stride;
    long long output_adjoint_node_stride;
    long long input_adjoint_node_stride;
}};
extern "C" __device__ __constant__ unsigned int symmetrix_operator_abi_version_v1 = 3u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_kind_v1 = {operator_kind}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_kind_v1 = {metadata["scalar_kind"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_size_v1 = {metadata["scalar_size"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_capabilities_v1 = {metadata["capabilities"]}u;
extern "C" __device__ __constant__ int symmetrix_operator_input_components_v1 = input_components;
extern "C" __device__ __constant__ int symmetrix_operator_output_components_v1 = output_components;
extern "C" __device__ __constant__ int symmetrix_operator_correlation_v1 = correlation;
extern "C" __device__ __constant__ int symmetrix_operator_term_count_v1 = term_count;
extern "C" __device__ __constant__ int symmetrix_operator_threads_v1 = {metadata["threads_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_schedule_revision_v1 = {metadata["schedule_revision"]};
extern "C" __device__ __constant__ int symmetrix_operator_channel_tile_v1 = {128 if metadata["schedule"] == PRODUCT_TILED128_SCHEDULE else 1};
extern "C" __device__ __constant__ int symmetrix_operator_forward_node_tile_v1 = 1;
extern "C" __device__ __constant__ int symmetrix_operator_forward_component_tile_v1 = {1 if metadata["schedule"] == PRODUCT_TILED128_SCHEDULE else metadata["output_components"]};
extern "C" __device__ __constant__ int symmetrix_operator_reverse_node_tile_v1 = {4 if metadata["schedule"] == PRODUCT_TILED128_SCHEDULE else 1};
extern "C" __device__ __constant__ int symmetrix_operator_reverse_component_tile_v1 = {1 if metadata["schedule"] == PRODUCT_TILED128_SCHEDULE else metadata["input_components"]};
extern "C" __device__ __constant__ int symmetrix_operator_logical_tiles_per_block_v1 = {1 if metadata["schedule"] == PRODUCT_TILED128_SCHEDULE else metadata["threads_per_block"]};
extern "C" __device__ __constant__ char symmetrix_operator_artifact_id_v1[] = {cpp_text_literal(metadata["artifact_id"])};
extern "C" __device__ __constant__ char symmetrix_operator_structure_fingerprint_v1[] = {cpp_text_literal(metadata["structure_fingerprint"])};
extern "C" __device__ __constant__ char symmetrix_operator_target_v1[] = {cpp_text_literal(metadata["target"])};
"""


def _product(components: tuple[int, ...]) -> str:
    return "(" + "*".join(f"x[{component}]" for component in components) + ")"


def _render_contraction_chunked(metadata: Mapping[str, Any], *, operator: str) -> str:
    lower = operator.lower()
    terms = _contraction_terms(metadata["contract"])
    chunks = [terms[start : start + 32] for start in range(0, len(terms), 32)]
    forward_helpers = []
    reverse_helpers = []
    term_offset = 0
    for chunk_index, chunk in enumerate(chunks):
        forward = []
        reverse = []
        for local_term, (output, components) in enumerate(chunk):
            term = term_offset + local_term
            forward.append(
                f"    out[{output}] += weights[{term}*channels]*{_product(components)};"
            )
            reverse.append(
                f"    const Scalar scale_{term} = adj[{output}]*weights[{term}*channels];"
            )
            for component, multiplicity in Counter(components).items():
                remaining = list(components)
                remaining.remove(component)
                factor = _product(tuple(remaining)) if remaining else "Scalar(1)"
                if multiplicity != 1:
                    factor = f"(Scalar({multiplicity})*{factor})"
                reverse.append(f"    grad[{component}] += scale_{term}*{factor};")
        forward_helpers.append(
            f"__device__ __forceinline__ void {lower}_fwd_{chunk_index}("
            "const Scalar* x, const Scalar* weights, int channels, Scalar* out) {\n"
            + "\n".join(forward)
            + "\n}"
        )
        reverse_helpers.append(
            f"__device__ __forceinline__ void {lower}_rev_{chunk_index}("
            "const Scalar* x, const Scalar* weights, int channels, "
            "const Scalar* adj, Scalar* grad) {\n" + "\n".join(reverse) + "\n}"
        )
        term_offset += len(chunk)
    forward_calls = "\n".join(
        f"        {lower}_fwd_{index}(x, weights, channel_count, out);"
        for index in range(len(chunks))
    )
    reverse_calls = "\n".join(
        f"        {lower}_rev_{index}(x, weights, channel_count, adj, grad);"
        for index in range(len(chunks))
    )
    kernels = (
        _contraction_tiled128_kernels(metadata, reverse_calls, operator=operator)
        if metadata["schedule"] == PRODUCT_TILED128_SCHEDULE
        else _contraction_kernels(forward_calls, reverse_calls, operator=operator)
    )
    return "\n".join((*forward_helpers, *reverse_helpers)) + kernels


def _contraction_kernels(forward_body: str, reverse_body: str, *, operator: str) -> str:
    lower = operator.lower()
    return f"""
extern "C" __global__ __launch_bounds__(256)
void symmetrix_{lower}_forward_v1({operator}Args args) {{
    const Scalar* input = static_cast<const Scalar*>(args.input_raw);
    const Scalar* packed_weights = static_cast<const Scalar*>(args.weights_raw);
    Scalar* output_values = static_cast<Scalar*>(args.output_raw);
    const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
    const long long owners = args.num_nodes*channel_count;
    for (long long owner = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
         owner < owners; owner += stride) {{
        const long long node = owner/channel_count;
        const int channel = static_cast<int>(owner%channel_count);
        const int type = args.node_types[node];
        Scalar x[input_components];
        Scalar out[output_components] = {{}};
        for (int component=0; component<input_components; ++component)
            x[component] = input[node*args.input_node_stride+component*channel_count+channel];
        const Scalar* weights = packed_weights
            +(static_cast<long long>(type)*term_count)*channel_count+channel;
{forward_body}
        for (int component=0; component<output_components; ++component)
            output_values[node*args.output_node_stride+component*channel_count+channel]
                = out[component];
    }}
}}
extern "C" __global__ __launch_bounds__(256)
void symmetrix_{lower}_reverse_v1({operator}Args args) {{
    const Scalar* input = static_cast<const Scalar*>(args.input_raw);
    const Scalar* packed_weights = static_cast<const Scalar*>(args.weights_raw);
    const Scalar* output_adjoint =
        static_cast<const Scalar*>(args.output_adjoint_raw);
    Scalar* input_adjoint = static_cast<Scalar*>(args.input_adjoint_raw);
    const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
    const long long owners = args.num_nodes*channel_count;
    for (long long owner = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
         owner < owners; owner += stride) {{
        const long long node = owner/channel_count;
        const int channel = static_cast<int>(owner%channel_count);
        const int type = args.node_types[node];
        Scalar x[input_components];
        Scalar adj[output_components];
        Scalar grad[input_components] = {{}};
        for (int component=0; component<input_components; ++component)
            x[component] = input[node*args.input_node_stride+component*channel_count+channel];
        for (int component=0; component<output_components; ++component)
            adj[component] = output_adjoint[
                node*args.output_adjoint_node_stride+component*channel_count+channel];
        const Scalar* weights = packed_weights
            +(static_cast<long long>(type)*term_count)*channel_count+channel;
{reverse_body}
        double scale = 0.0;
        for (int component=0; component<input_components; ++component) {{
            const long long input_adjoint_index =
                node*args.input_adjoint_node_stride+component*channel_count+channel;
            if (args.reserved&1u)
                input_adjoint[input_adjoint_index] += grad[component];
            else
                input_adjoint[input_adjoint_index] = grad[component];
            scale += static_cast<double>(x[component])
                *static_cast<double>(grad[component]);
        }}
        if (args.capture_input_scale_adjoint)
            atomicAdd(args.input_scale_adjoint+node, scale);
    }}
}}
"""


def _contraction_tiled128_kernels(
    metadata: Mapping[str, Any], reverse_body: str, *, operator: str
) -> str:
    """Render 128-channel, multi-node work through the ABI-v2 1-D grid."""

    lower = operator.lower()
    terms = _contraction_terms(metadata["contract"])
    forward_branches = []
    for output in range(int(metadata["output_components"])):
        body = []
        for term, (term_output, components) in enumerate(terms):
            if term_output == output:
                body.append(
                    f"value += weights[{term}*channel_count]*{_product(components)};"
                )
        body.append(
            "output_values[node*args.output_node_stride+output_component"
            "*channel_count+channel] = value;"
        )
        prefix = "if" if not forward_branches else "else if"
        forward_branches.append(
            f"{prefix} (output_component == {output}) {{\n"
            + "\n".join(f"                {line}" for line in body)
            + "\n            }"
        )

    reverse_entries: dict[int, list[tuple[int, int, tuple[int, ...], int]]] = {
        component: [] for component in range(int(metadata["input_components"]))
    }
    for term, (output, components) in enumerate(terms):
        for component, multiplicity in Counter(components).items():
            remaining = list(components)
            remaining.remove(component)
            reverse_entries[component].append(
                (term, output, tuple(remaining), multiplicity)
            )
    reverse_branches = []
    for component, entries in reverse_entries.items():
        body = []
        for entry, (term, output, remaining, multiplicity) in enumerate(entries):
            for node_lane in range(4):
                factor = (
                    "("
                    + "*".join(f"x_{node_lane}[{value}]" for value in remaining)
                    + ")"
                    if remaining
                    else "Scalar(1)"
                )
                if multiplicity != 1:
                    factor = f"(Scalar({multiplicity})*{factor})"
                body.extend(
                    (
                        (
                            f"const Scalar scale_{entry}_{node_lane} = "
                            f"adj_{node_lane}[{output}]"
                            f"*weights_{node_lane}[{term}*channel_count];"
                        ),
                        f"value_{node_lane} += scale_{entry}_{node_lane}*{factor};",
                    )
                )
        prefix = "if" if not reverse_branches else "else if"
        reverse_branches.append(
            f"{prefix} (input_component == {component}) {{\n"
            + "\n".join(f"                {line}" for line in body)
            + "\n            }"
        )
    return f"""
extern "C" __global__ __launch_bounds__(128)
void symmetrix_{lower}_forward_v1({operator}Args args) {{
    constexpr int channel_tile = 128;
    constexpr int node_tile = 1;
    const Scalar* input = static_cast<const Scalar*>(args.input_raw);
    const Scalar* packed_weights = static_cast<const Scalar*>(args.weights_raw);
    Scalar* output_values = static_cast<Scalar*>(args.output_raw);
    const long long channel_tiles =
        (static_cast<long long>(channel_count)+channel_tile-1)/channel_tile;
    const long long logical_tiles =
        channel_tiles*args.num_nodes*output_components;
    for (long long logical_tile = blockIdx.x; logical_tile < logical_tiles;
         logical_tile += gridDim.x) {{
        const int output_component =
            static_cast<int>(logical_tile%output_components);
        const long long channel_owner = logical_tile/output_components;
        const int channel = static_cast<int>(channel_owner%channel_tiles)
            *channel_tile+threadIdx.x;
        const long long node_base = channel_owner/channel_tiles;
        if (channel >= channel_count) continue;
        for (int node_lane=0; node_lane<node_tile; ++node_lane) {{
            const long long node = node_base+node_lane;
            if (node >= args.num_nodes) continue;
            const int type = args.node_types[node];
            Scalar x[input_components];
            Scalar value = Scalar(0);
            for (int component=0; component<input_components; ++component)
                x[component] = input[
                    node*args.input_node_stride+component*channel_count+channel];
            const Scalar* weights = packed_weights
                +(static_cast<long long>(type)*term_count)*channel_count+channel;
            {chr(10).join(forward_branches)}
        }}
    }}
}}
extern "C" __global__ __launch_bounds__(128)
void symmetrix_{lower}_reverse_v1({operator}Args args) {{
    constexpr int channel_tile = 128;
    constexpr int node_tile = 4;
    const Scalar* input = static_cast<const Scalar*>(args.input_raw);
    const Scalar* packed_weights = static_cast<const Scalar*>(args.weights_raw);
    const Scalar* output_adjoint =
        static_cast<const Scalar*>(args.output_adjoint_raw);
    Scalar* input_adjoint = static_cast<Scalar*>(args.input_adjoint_raw);
    const long long channel_tiles =
        (static_cast<long long>(channel_count)+channel_tile-1)/channel_tile;
    const long long node_tiles = (args.num_nodes+node_tile-1)/node_tile;
    const long long logical_tiles =
        channel_tiles*node_tiles*input_components;
    for (long long logical_tile = blockIdx.x; logical_tile < logical_tiles;
         logical_tile += gridDim.x) {{
        const int input_component =
            static_cast<int>(logical_tile%input_components);
        const long long channel_owner = logical_tile/input_components;
        const int channel = static_cast<int>(channel_owner%channel_tiles)
            *channel_tile+threadIdx.x;
        const long long node_base = (channel_owner/channel_tiles)*node_tile;
        if (channel >= channel_count) continue;
        const Scalar* weights_0 = packed_weights+channel;
        const Scalar* weights_1 = packed_weights+channel;
        const Scalar* weights_2 = packed_weights+channel;
        const Scalar* weights_3 = packed_weights+channel;
        Scalar x_0[input_components] = {{}};
        Scalar x_1[input_components] = {{}};
        Scalar x_2[input_components] = {{}};
        Scalar x_3[input_components] = {{}};
        Scalar adj_0[output_components] = {{}};
        Scalar adj_1[output_components] = {{}};
        Scalar adj_2[output_components] = {{}};
        Scalar adj_3[output_components] = {{}};
        Scalar value_0 = Scalar(0);
        Scalar value_1 = Scalar(0);
        Scalar value_2 = Scalar(0);
        Scalar value_3 = Scalar(0);
        for (int node_lane=0; node_lane<node_tile; ++node_lane) {{
            const long long node=node_base+node_lane;
            if(node>=args.num_nodes) continue;
            const int type=args.node_types[node];
            const Scalar* lane_weights=packed_weights
                +(static_cast<long long>(type)*term_count)*channel_count+channel;
            if(node_lane==0) weights_0=lane_weights;
            else if(node_lane==1) weights_1=lane_weights;
            else if(node_lane==2) weights_2=lane_weights;
            else weights_3=lane_weights;
            Scalar* lane_x=node_lane==0?x_0:node_lane==1?x_1:node_lane==2?x_2:x_3;
            Scalar* lane_adj=node_lane==0?adj_0:node_lane==1?adj_1:node_lane==2?adj_2:adj_3;
            for (int component=0; component<input_components; ++component)
                lane_x[component] = input[
                    node*args.input_node_stride+component*channel_count+channel];
            for (int component=0; component<output_components; ++component)
                lane_adj[component] = output_adjoint[
                    node*args.output_adjoint_node_stride+component*channel_count+channel];
        }}
        {chr(10).join(reverse_branches)}
        const Scalar lane_values[node_tile] = {{value_0,value_1,value_2,value_3}};
        for (int node_lane=0; node_lane<node_tile; ++node_lane) {{
            const long long node=node_base+node_lane;
            if(node>=args.num_nodes) continue;
            const long long input_adjoint_index =
                node*args.input_adjoint_node_stride+input_component*channel_count+channel;
            if (args.reserved&1u)
                input_adjoint[input_adjoint_index] += lane_values[node_lane];
            else
                input_adjoint[input_adjoint_index] = lane_values[node_lane];
            if (args.capture_input_scale_adjoint && input_component == 0) {{
                const Scalar* x=node_lane==0?x_0:node_lane==1?x_1:node_lane==2?x_2:x_3;
                const Scalar* adj=node_lane==0?adj_0:node_lane==1?adj_1:node_lane==2?adj_2:adj_3;
                Scalar grad[input_components] = {{}};
                const Scalar* weights=node_lane==0?weights_0:node_lane==1?weights_1:node_lane==2?weights_2:weights_3;
{reverse_body}
                double scale = 0.0;
                for (int component=0; component<input_components; ++component)
                    scale += static_cast<double>(x[component])
                        *static_cast<double>(grad[component]);
                atomicAdd(args.input_scale_adjoint+node, scale);
            }}
        }}
    }}
}}
"""


def _render_contraction_table(metadata: Mapping[str, Any], *, operator: str) -> str:
    lower = operator.lower()
    terms = _contraction_terms(metadata["contract"])
    offsets = [0]
    for group in metadata["contract"]["monomial_groups"]:
        offsets.append(offsets[-1] + int(group["term_count"]))
    outputs = ",".join(str(output) for output, _ in terms)
    degrees = ",".join(str(len(components)) for _, components in terms)
    components = ",".join(
        str(value)
        for _, values in terms
        for value in (*values, *([-1] * (metadata["correlation"] - len(values))))
    )
    offset_values = ",".join(str(value) for value in offsets)
    tables = f"""
__device__ __constant__ int {lower}_term_outputs[term_count] = {{{outputs}}};
__device__ __constant__ int {lower}_term_degrees[term_count] = {{{degrees}}};
__device__ __constant__ int {lower}_term_components[term_count*correlation] = {{{components}}};
__device__ __constant__ int {lower}_output_offsets[output_components+1] = {{{offset_values}}};
"""
    forward = f"""
        for (int output=0; output<output_components; ++output) {{
            for (int term={lower}_output_offsets[output];
                 term<{lower}_output_offsets[output+1]; ++term) {{
                Scalar product = Scalar(1);
                for (int degree=0; degree<{lower}_term_degrees[term]; ++degree)
                    product *= x[{lower}_term_components[term*correlation+degree]];
                out[output] += weights[term*args.channels]*product;
            }}
        }}"""
    reverse = f"""
        for (int term=0; term<term_count; ++term) {{
            const Scalar scale = adj[{lower}_term_outputs[term]]
                *weights[term*args.channels];
            for (int factor=0; factor<{lower}_term_degrees[term]; ++factor) {{
                Scalar product = scale;
                for (int other=0; other<{lower}_term_degrees[term]; ++other)
                    if (other != factor)
                        product *= x[{lower}_term_components[term*correlation+other]];
                grad[{lower}_term_components[term*correlation+factor]] += product;
            }}
        }}"""
    return tables + _contraction_kernels(forward, reverse, operator=operator)


def render_jit_m0_device_module(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> tuple[str, dict[str, Any]]:
    """Render a self-contained CUDA/HIP RTC M0 module and its identity."""

    metadata = jit_m0_device_metadata(
        contract,
        generation_version=generation_version,
        precision=precision,
        target=target,
        schedule=schedule,
    )
    body = (
        _render_contraction_chunked(metadata, operator="M0")
        if schedule in ("chunk32", PRODUCT_TILED128_SCHEDULE)
        else _render_contraction_table(metadata, operator="M0")
    )
    return _contraction_device_prefix(metadata, operator="M0") + body, metadata


def render_jit_m1_device_module(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> tuple[str, dict[str, Any]]:
    """Render a scalar M1 module through the shared contraction lowering."""

    metadata = jit_m1_device_metadata(
        contract,
        generation_version=generation_version,
        precision=precision,
        target=target,
        schedule=schedule,
    )
    body = (
        _render_contraction_chunked(metadata, operator="M1")
        if schedule in ("chunk32", PRODUCT_TILED128_SCHEDULE)
        else _render_contraction_table(metadata, operator="M1")
    )
    return _contraction_device_prefix(metadata, operator="M1") + body, metadata


def jit_m0_host_metadata(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
) -> dict[str, Any]:
    """Return the exact identity for a generated state-free host M0 plugin."""

    metadata = jit_m0_device_metadata(
        contract,
        generation_version=generation_version,
        precision=precision,
        target=target,
        schedule=schedule,
    )
    structure = metadata["structure_fingerprint"]
    suffix = "f32" if precision == "float32" else "f64"
    generated_identity = generated_m0_identity(
        generation_revision=generation_version,
        structure_fingerprint=structure,
        precision=precision,
        backend="host",
        target=target,
        channels=metadata["contract"]["channels"],
        input_components=metadata["input_components"],
        output_components=metadata["output_components"],
        correlation=metadata["correlation"],
        term_count=metadata["term_count"],
        schedule=schedule,
        schedule_revision=metadata["schedule_revision"],
    )
    return {
        **metadata,
        "schema": "symmetrix.jit.m0-host-plugin/2",
        "abi": "symmetrix.jit.m0-host-plugin/2",
        "abi_version": 2,
        "artifact_id": (
            f"jit-m0-host-gen{generation_version}-{suffix}-"
            f"{structure.removeprefix('sha256:')[:16]}-"
            f"c{metadata['contract']['channels']}-{schedule}-r{M0_SCHEDULE_REVISION}"
        ),
        "generated_identity": generated_identity,
        "capabilities": M0_CAPABILITIES_V1,
    }


def jit_m1_host_metadata(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
) -> dict[str, Any]:
    """Return the exact identity for a generated host M1 plugin."""

    metadata = jit_m1_device_metadata(
        contract,
        generation_version=generation_version,
        precision=precision,
        target=target,
        schedule=schedule,
    )
    generated_identity = generated_m1_identity(
        generation_revision=generation_version,
        structure_fingerprint=metadata["structure_fingerprint"],
        precision=precision,
        backend="host",
        target=target,
        channels=metadata["contract"]["channels"],
        input_components=metadata["input_components"],
        correlation=metadata["correlation"],
        term_count=metadata["term_count"],
        schedule=schedule,
        schedule_revision=M1_SCHEDULE_REVISION,
    )
    return {
        **metadata,
        "schema": "symmetrix.jit.m1-host-plugin/2",
        "abi": "symmetrix.jit.m1-host-plugin/2",
        "abi_version": 2,
        "schedule_revision": M1_SCHEDULE_REVISION,
        "artifact_id": (
            f"jit-m1-host-gen{generation_version}-"
            f"{'f32' if precision == 'float32' else 'f64'}-"
            f"{metadata['structure_fingerprint'].removeprefix('sha256:')[:16]}-"
            f"c{metadata['contract']['channels']}-{schedule}-r{M1_SCHEDULE_REVISION}"
        ),
        "generated_identity": generated_identity,
        "capabilities": M1_HOST_CAPABILITIES_V1,
    }


def render_jit_m0_host_plugin(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
    _metadata: Mapping[str, Any] | None = None,
    _operator: str = "M0",
) -> tuple[str, dict[str, Any]]:
    """Render a standalone host contraction plugin with a tiled channel owner."""

    if schedule != "chunk32":
        raise ValueError(
            f"{_operator} host plugins currently require the chunk32 schedule"
        )
    lower = _operator.lower()
    abi_header = {
        "M0": _M0_HOST_PLUGIN_ABI_HEADER,
        "M1": _M1_HOST_PLUGIN_ABI_HEADER,
    }[_operator]
    metadata = (
        dict(_metadata)
        if _metadata is not None
        else jit_m0_host_metadata(
            contract,
            generation_version=generation_version,
            precision=precision,
            schedule=schedule,
            target=target,
        )
    )
    terms = _contraction_terms(metadata["contract"])
    chunks = [terms[start : start + 32] for start in range(0, len(terms), 32)]
    forward_helpers = []
    reverse_helpers = []
    term_offset = 0
    for chunk_index, chunk in enumerate(chunks):
        forward = []
        reverse = []
        for local_term, (output, components) in enumerate(chunk):
            term = term_offset + local_term
            vector_product = "*".join(
                f"x[{component}][lane]" for component in components
            )
            forward.append(
                "#if defined(__clang__)\n"
                "#pragma clang loop vectorize(enable)\n"
                "#elif defined(__GNUC__)\n"
                "#pragma GCC ivdep\n"
                "#endif\n"
                "    for (int lane=0; lane<active_channels; ++lane)\n"
                f"        out[{output}][lane] += weights[{term}*channels+lane]"
                f"*{vector_product};"
            )
            for component, multiplicity in Counter(components).items():
                remaining = list(components)
                remaining.remove(component)
                factor = (
                    "*".join(f"x[{value}][lane]" for value in remaining)
                    if remaining
                    else "Scalar(1)"
                )
                if multiplicity != 1:
                    factor = f"(Scalar({multiplicity})*{factor})"
                reverse.append(
                    "#if defined(__clang__)\n"
                    "#pragma clang loop vectorize(enable)\n"
                    "#elif defined(__GNUC__)\n"
                    "#pragma GCC ivdep\n"
                    "#endif\n"
                    "    for (int lane=0; lane<active_channels; ++lane) {\n"
                    f"        const Scalar scale = adj[{output}][lane]"
                    f"*weights[{term}*channels+lane];\n"
                    f"        grad[{component}][lane] += scale*{factor};\n"
                    "    }"
                )
        forward_helpers.append(
            f"SYMMETRIX_{_operator}_ALWAYS_INLINE void {lower}_fwd_{chunk_index}("
            "const Scalar (&x)[input_components][host_channel_tile], "
            "const Scalar* weights, int channels, int active_channels, "
            "Scalar (&out)[output_components][host_channel_tile]) {\n"
            + "\n".join(forward)
            + "\n}"
        )
        reverse_helpers.append(
            f"SYMMETRIX_{_operator}_ALWAYS_INLINE void {lower}_rev_{chunk_index}("
            "const Scalar (&x)[input_components][host_channel_tile], "
            "const Scalar* weights, int channels, int active_channels, "
            "const Scalar (&adj)[output_components][host_channel_tile], "
            "Scalar (&grad)[input_components][host_channel_tile]) {\n"
            + "\n".join(reverse)
            + "\n}"
        )
        term_offset += len(chunk)
    forward_calls = "\n".join(
        f"    {lower}_fwd_{index}(x, weights, channels, active_channels, out);"
        for index in range(len(chunks))
    )
    reverse_calls = "\n".join(
        f"    {lower}_rev_{index}(x, weights, channels, active_channels, adj, grad);"
        for index in range(len(chunks))
    )
    scalar = "float" if precision == "float32" else "double"
    helpers = "\n".join((*forward_helpers, *reverse_helpers))
    source = f"""// Generated Symmetrix state-free host {_operator} plugin.
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <type_traits>

{abi_header}

using Scalar = {scalar};
constexpr int input_components = {metadata["input_components"]};
constexpr int output_components = {metadata["output_components"]};
constexpr int channels = {metadata["contract"]["channels"]};
constexpr int term_count = {metadata["term_count"]};
#if defined(__AVX512F__)
constexpr int host_vector_width = std::is_same_v<Scalar,float> ? 16 : 8;
#elif defined(__AVX2__)
constexpr int host_vector_width = std::is_same_v<Scalar,float> ? 8 : 4;
#else
constexpr int host_vector_width = 1;
#endif
constexpr int host_channel_tile = channels >= host_vector_width ? host_vector_width
    : channels >= 8 ? 8 : channels >= 4 ? 4 : channels >= 2 ? 2 : 1;
constexpr std::uint32_t host_capabilities = host_channel_tile > 1
    ? {metadata["capabilities"]}u
    : (SYMMETRIX_JIT_{_operator}_HOST_FORWARD_OWNER_V1
        |SYMMETRIX_JIT_{_operator}_HOST_REVERSE_OWNER_V1
        |SYMMETRIX_JIT_{_operator}_HOST_SCALE_ADJOINT_V1);
constexpr std::uint32_t descriptor_channel_tile = host_channel_tile > 1
    ? static_cast<std::uint32_t>(host_channel_tile) : 0u;
#if defined(__clang__) || defined(__GNUC__)
#define SYMMETRIX_{_operator}_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SYMMETRIX_{_operator}_ALWAYS_INLINE inline
#endif

{helpers}

extern "C" void {lower}_forward_owner(
    const SymmetrixJit{_operator}HostArgsV1* args, std::int64_t owner) {{
    const auto* input = static_cast<const Scalar*>(args->input);
    const auto* packed_weights = static_cast<const Scalar*>(args->weights);
    auto* output = static_cast<Scalar*>(args->output);
    const std::int64_t node = owner/((channels+host_channel_tile-1)/host_channel_tile);
    const int channel = static_cast<int>(
        owner%((channels+host_channel_tile-1)/host_channel_tile))*host_channel_tile;
    const int active_channels = channel+host_channel_tile <= channels
        ? host_channel_tile : channels-channel;
    const int type = args->node_types[node];
    Scalar x[input_components][host_channel_tile];
    Scalar out[output_components][host_channel_tile] = {{}};
    for (int component=0; component<input_components; ++component)
        for (int lane=0; lane<active_channels; ++lane)
            x[component][lane] = input[
                (node*input_components+component)*channels+channel+lane];
    const Scalar* weights = packed_weights
        +(static_cast<std::int64_t>(type)*term_count)*channels+channel;
{forward_calls}
    for (int component=0; component<output_components; ++component)
        for (int lane=0; lane<active_channels; ++lane)
            output[(node*output_components+component)*channels+channel+lane] =
                out[component][lane];
}}

extern "C" void {lower}_reverse_owner(
    const SymmetrixJit{_operator}HostArgsV1* args, std::int64_t owner) {{
    const auto* input = static_cast<const Scalar*>(args->input);
    const auto* packed_weights = static_cast<const Scalar*>(args->weights);
    const auto* output_adjoint =
        static_cast<const Scalar*>(args->output_adjoint);
    auto* input_adjoint = static_cast<Scalar*>(args->input_adjoint);
    const std::int64_t node = owner/((channels+host_channel_tile-1)/host_channel_tile);
    const int channel = static_cast<int>(
        owner%((channels+host_channel_tile-1)/host_channel_tile))*host_channel_tile;
    const int active_channels = channel+host_channel_tile <= channels
        ? host_channel_tile : channels-channel;
    const int type = args->node_types[node];
    Scalar x[input_components][host_channel_tile];
    Scalar adj[output_components][host_channel_tile];
    Scalar grad[input_components][host_channel_tile] = {{}};
    for (int component=0; component<input_components; ++component)
        for (int lane=0; lane<active_channels; ++lane)
            x[component][lane] = input[
                (node*input_components+component)*channels+channel+lane];
    for (int component=0; component<output_components; ++component)
        for (int lane=0; lane<active_channels; ++lane)
            adj[component][lane] = output_adjoint[
                (node*output_components+component)*channels+channel+lane];
    const Scalar* weights = packed_weights
        +(static_cast<std::int64_t>(type)*term_count)*channels+channel;
{reverse_calls}
    double scale = 0.0;
    for (int component=0; component<input_components; ++component)
        for (int lane=0; lane<active_channels; ++lane) {{
            const std::int64_t input_adjoint_index =
                (node*input_components+component)*channels+channel+lane;
            if (args->reserved&1u)
                input_adjoint[input_adjoint_index] += grad[component][lane];
            else
                input_adjoint[input_adjoint_index] = grad[component][lane];
            scale += static_cast<double>(x[component][lane])
                *static_cast<double>(grad[component][lane]);
        }}
    if (args->capture_input_scale_adjoint)
        std::atomic_ref<double>(args->input_scale_adjoint[node])
            .fetch_add(scale, std::memory_order_relaxed);
}}

extern "C" const SymmetrixJit{_operator}HostPluginV2*
symmetrix_jit_{lower}_host_plugin_query_v2() {{
    static const SymmetrixJit{_operator}HostPluginV2 descriptor = {{{{
        SYMMETRIX_JIT_{_operator}_HOST_PLUGIN_ABI_VERSION_V2,
        sizeof(SymmetrixJit{_operator}HostPluginV2), sizeof(void*),
        SYMMETRIX_JIT_{_operator}_HOST_PLUGIN_BYTE_ORDER_V1,
        {metadata["scalar_kind"]}u,
        {metadata["scalar_size"]}u, host_capabilities,
        channels, input_components, output_components,
        {metadata["correlation"]}, term_count, descriptor_channel_tile,
        SYMMETRIX_JIT_{_operator}_HOST_PLUGIN_ABI_TAG_V2,
        {cpp_text_literal(metadata["artifact_id"])},
        {cpp_text_literal(metadata["structure_fingerprint"])},
        &{lower}_forward_owner, &{lower}_reverse_owner
    }},
        {cpp_text_literal(metadata["generated_identity"]["artifact_id"])},
        {cpp_text_literal(metadata["target"])}
    }};
    return &descriptor;
}}
"""
    return source, metadata


def render_jit_m1_host_plugin(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    precision: str,
    schedule: str = "chunk32",
    target: str = "host",
) -> tuple[str, dict[str, Any]]:
    """Render a host M1 plugin through the shared contraction lowering."""

    metadata = jit_m1_host_metadata(
        contract,
        generation_version=generation_version,
        precision=precision,
        schedule=schedule,
        target=target,
    )
    source, _ = render_jit_m0_host_plugin(
        contract,
        generation_version=generation_version,
        precision=precision,
        schedule=schedule,
        target=target,
        _metadata=metadata,
        _operator="M1",
    )
    return source, metadata


def normalize_generated_r0_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize and validate the scalar-source R0 contract used by codegen."""

    normalized = normalize_standard_r0_contract(contract)
    edge_l_max = normalized["edge_harmonics"]["l_max"]
    channels = normalized["channels"]
    if normalized["source_harmonics"]["l_max"] != 0:
        raise ValueError("R0 device modules require scalar source harmonics")
    expected_paths = [
        {
            "index": l_value,
            "output_l": l_value,
            "edge_l": l_value,
            "source_l": 0,
            "connection_mode": "uvu",
            "has_weight": True,
            "path_shape": [channels, 1],
        }
        for l_value in range(edge_l_max + 1)
    ]
    if normalized["paths"] != expected_paths:
        raise ValueError(
            "R0 device modules require one scalar-source uvu path per angular order"
        )
    expected_groups = [
        {"l": l_value, "components": 2 * l_value + 1, "path_indices": [l_value]}
        for l_value in range(edge_l_max + 1)
    ]
    if normalized["groups"] != expected_groups:
        raise ValueError(
            "R0 device module output groups are not dense by angular order"
        )
    harmonic_count = (edge_l_max + 1) ** 2
    terms = normalized["sparse_coupling"]["terms"]
    if len(terms) != harmonic_count or any(
        term
        != {
            "row": component,
            "lme": component,
            "lm1": component,
            "lm2": 0,
            "coefficient": 1.0,
        }
        for component, term in enumerate(terms)
    ):
        raise ValueError("R0 device module sparse coupling is not scalar identity")
    return normalized


def choose_r0_schedule(*, backend: str, schedule: str | None) -> str:
    """Resolve a generated R0 schedule for an explicit backend."""

    if backend not in ("host", "cuda", "hip"):
        raise ValueError("R0 backend must be 'host', 'cuda', or 'hip'")
    if backend == "host":
        if schedule != "host-owner":
            raise ValueError("R0 host generation requires the 'host-owner' schedule")
        return schedule
    if schedule is None:
        return "receiver-edge16"
    if schedule not in ("receiver-edge16", "receiver-edge32"):
        raise ValueError(
            f"R0 {backend} generation requires 'receiver-edge16' or 'receiver-edge32'"
        )
    return schedule


def _r0_device_schedule_metadata(
    *,
    backend: str,
    target: str,
    precision: str,
    schedule: str,
    channels: int,
    l_max: int,
) -> dict[str, Any]:
    edge_vector_length = 16 if schedule == "receiver-edge16" else 32
    edge_team_size = R0_THREADS_V1 // edge_vector_length
    hip_all_angular = backend == "hip" and precision == "float32"
    scalar_size = 4 if precision == "float32" else 8
    forward_scratch_bytes_per_team = channels * (2 * l_max + 1) * scalar_size
    forward_scratch_limit = (48 if backend == "cuda" else 64) * 1024
    device_wide_fallback = forward_scratch_bytes_per_team > forward_scratch_limit
    device_forward_teams = max(
        1,
        min(
            R0_THREADS_V1 // 32,
            forward_scratch_limit // forward_scratch_bytes_per_team,
        ),
    )
    forward_threads = (
        32 if backend == "cuda" and not device_wide_fallback else R0_THREADS_V1
    )
    if backend == "cuda" and not device_wide_fallback:
        device_forward_teams = 1
    if backend == "cuda" and precision == "float32":
        blocks, candidates = 3, [2, 3, 6]
    elif (
        backend == "hip"
        and precision == "float32"
        and target.split(":", 1)[0] == "gfx1151"
    ):
        blocks, candidates = 8, [4, 8, 16]
    else:
        blocks, candidates = 4, [2, 4, 8]
    return {
        "schedule_revision": R0_DEVICE_SCHEDULE_REVISION,
        "schedule_capabilities": [
            "density-receiver",
            "forward-receiver",
            "reverse-prepare-receiver",
            f"coordinate-reverse-edge{edge_vector_length}",
            "persistent-grid",
        ],
        "schedule_policy": {
            "density_prepare_owner": "receiver-warp",
            "forward_owner": (
                "receiver-channel-all-angular"
                if hip_all_angular
                else (
                    "receiver-channel-all-angular-fallback"
                    if device_wide_fallback
                    else "receiver-angular-order-team"
                )
            ),
            "coordinate_reverse_owner": f"edge{edge_vector_length}",
            "edge_team_size": edge_team_size,
            "edge_vector_length": edge_vector_length,
            "threads_per_block": R0_THREADS_V1,
        },
        "persistent_profile": {
            "blocks_per_compute_unit": blocks,
            "calibration_candidates": candidates,
        },
        "launch_geometry": {
            "density_prepare": {
                "extent": "nodes",
                "extent_multiplier": 1,
                "owners_per_block": R0_THREADS_V1 // 32,
                "threads_per_block": R0_THREADS_V1,
                "persistent": backend != "cuda",
            },
            "forward": {
                "extent": "nodes",
                "extent_multiplier": (
                    channels if hip_all_angular or device_wide_fallback else l_max + 1
                ),
                "owners_per_block": (
                    R0_THREADS_V1
                    if hip_all_angular or device_wide_fallback
                    else device_forward_teams
                ),
                "threads_per_block": forward_threads,
                "persistent": backend != "cuda",
            },
            "reverse_prepare": {
                "extent": "nodes",
                "extent_multiplier": 1,
                "owners_per_block": 1,
                "threads_per_block": R0_THREADS_V1,
                "persistent": backend != "cuda",
            },
            "coordinate_reverse": {
                "extent": "edges",
                "extent_multiplier": 1,
                "owners_per_block": edge_team_size,
                "threads_per_block": R0_THREADS_V1,
                "persistent": True,
            },
        },
    }


def _validated_source_content_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SOURCE_CONTENT_SHA256.fullmatch(value) is None:
        raise ValueError("source_content_sha256 must be a SHA-256 digest")
    return value


def jit_r0_device_metadata(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    source_content_sha256: str | None,
    backend: str,
    precision: str,
    target: str,
    schedule: str | None,
    floating_point_policy: str | None = None,
    apply_edge_weights: bool = False,
) -> dict[str, Any]:
    normalized = normalize_generated_r0_contract(contract)
    if backend not in ("cuda", "hip"):
        raise ValueError("R0 device backend must be 'cuda' or 'hip'")
    schedule = choose_r0_schedule(backend=backend, schedule=schedule)
    source_content_sha256 = _validated_source_content_sha256(source_content_sha256)
    if precision not in ("float32", "float64"):
        raise ValueError("R0 device precision must be 'float32' or 'float64'")
    if not target:
        raise ValueError("R0 device target must not be empty")
    if floating_point_policy is None:
        floating_point_policy = "cuda-fast-math" if backend == "cuda" else "strict"
    if backend != "cuda" and floating_point_policy != "strict":
        raise ValueError("non-CUDA R0 device generation requires strict floating point")
    if backend == "cuda" and floating_point_policy not in ("strict", "cuda-fast-math"):
        raise ValueError(
            "CUDA R0 floating_point_policy must be strict or cuda-fast-math"
        )
    suffix = "f32" if precision == "float32" else "f64"
    structure = normalized["structure_fingerprint"]
    l_max = normalized["edge_harmonics"]["l_max"]
    harmonic_count = (normalized["edge_harmonics"]["l_max"] + 1) ** 2
    weight_mode = "weighted" if apply_edge_weights else "unity"
    generated_identity = generated_r0_identity(
        generation_revision=generation_version,
        compatibility_version=R0_COMPATIBILITY_VERSION,
        contract_fingerprint=normalized["generation_fingerprint"],
        structure_fingerprint=structure,
        precision=precision,
        backend=backend,
        target=str(target),
        channels=normalized["channels"],
        l_max=l_max,
        edge_weight_mode=weight_mode,
        floating_point_policy=floating_point_policy,
        schedule=schedule,
        schedule_revision=R0_DEVICE_SCHEDULE_REVISION,
    )
    schedule_metadata = _r0_device_schedule_metadata(
        backend=backend,
        target=str(target),
        precision=precision,
        schedule=schedule,
        channels=normalized["channels"],
        l_max=l_max,
    )
    metadata = {
        "schema": "symmetrix.jit.operator-module/1",
        "stage": "R0",
        "operator": "R0",
        "abi_version": R0_DEVICE_ABI_VERSION,
        "compatibility_version": R0_COMPATIBILITY_VERSION,
        "artifact_id": generated_identity["artifact_id"],
        "legacy_artifact_id": (
            f"jit-r0-gen{generation_version}-{suffix}-"
            f"{structure.removeprefix('sha256:')[:16]}-{schedule}-{target}"
        ),
        "structure_fingerprint": structure,
        "precision": precision,
        "scalar_kind": 1 if precision == "float32" else 2,
        "scalar_size": 4 if precision == "float32" else 8,
        "target": str(target),
        "schedule": schedule,
        "floating_point_policy": floating_point_policy,
        **schedule_metadata,
        "edge_weight_mode": weight_mode,
        "input_components": harmonic_count,
        "output_components": harmonic_count,
        "correlation": 1,
        "term_count": len(normalized["sparse_coupling"]["terms"]),
        "threads_per_block": R0_THREADS_V1,
        "capabilities": R0_CAPABILITIES_V1,
        "architecture_bounds": {
            "channels": normalized["channels"],
            "l_max": l_max,
            "harmonic_count": harmonic_count,
        },
        "requires_retained_harmonic_gradients": harmonic_count > 16,
        "generated_identity": generated_identity,
        "contract": normalized,
    }
    if source_content_sha256 is not None:
        metadata["source_content_sha256"] = source_content_sha256
    return metadata


def jit_r0_host_metadata(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    source_content_sha256: str | None,
    target: str,
    precision: str,
    schedule: str,
    apply_edge_weights: bool = False,
) -> dict[str, Any]:
    normalized = normalize_generated_r0_contract(contract)
    choose_r0_schedule(backend="host", schedule=schedule)
    source_content_sha256 = _validated_source_content_sha256(source_content_sha256)
    if precision not in ("float32", "float64"):
        raise ValueError("R0 host precision must be 'float32' or 'float64'")
    suffix = "f32" if precision == "float32" else "f64"
    l_max = normalized["edge_harmonics"]["l_max"]
    channels = normalized["channels"]
    harmonic_count = (l_max + 1) ** 2
    weight_mode = "weighted" if apply_edge_weights else "unity"
    structure = normalized["structure_fingerprint"]
    generated_identity = generated_r0_identity(
        generation_revision=generation_version,
        compatibility_version=R0_COMPATIBILITY_VERSION,
        contract_fingerprint=normalized["generation_fingerprint"],
        structure_fingerprint=structure,
        precision=precision,
        backend="host",
        target=target,
        channels=channels,
        l_max=l_max,
        edge_weight_mode=weight_mode,
        floating_point_policy="strict",
        schedule=schedule,
        schedule_revision=R0_HOST_SCHEDULE_REVISION,
    )
    metadata = {
        "schema": "symmetrix.jit.r0-host-plugin/2",
        "stage": "R0",
        "operator": "R0",
        "abi_version": R0_HOST_ABI_VERSION,
        "compatibility_version": R0_COMPATIBILITY_VERSION,
        "artifact_id": generated_identity["artifact_id"],
        "legacy_artifact_id": (
            f"jit-r0-host-gen{generation_version}-{suffix}-"
            f"{structure.removeprefix('sha256:')[:16]}-l{l_max}-c{channels}-"
            f"{weight_mode}"
        ),
        "structure_fingerprint": structure,
        "target": target,
        "precision": precision,
        "scalar_kind": 1 if precision == "float32" else 2,
        "scalar_size": 4 if precision == "float32" else 8,
        "edge_weight_mode": weight_mode,
        "schedule": schedule,
        "schedule_revision": R0_HOST_SCHEDULE_REVISION,
        "schedule_policy": {
            "density_prepare_owner": "receiver",
            "forward_owner": "receiver-channel-tile",
            "forward_channel_tile": R0_HOST_FORWARD_CHANNEL_TILE_V1,
            "reverse_prepare_owner": "receiver",
            "coordinate_reverse_owner": "receiver",
            "coordinate_reverse_channel_tile": (
                R0_HOST_COORDINATE_REVERSE_CHANNEL_TILE_V1
            ),
        },
        "input_components": harmonic_count,
        "output_components": harmonic_count,
        "channels": channels,
        "l_max": l_max,
        "term_count": harmonic_count,
        "capabilities": R0_HOST_CAPABILITIES_V1,
        "generated_identity": generated_identity,
        "contract": normalized,
    }
    if source_content_sha256 is not None:
        metadata["source_content_sha256"] = source_content_sha256
    return metadata


def render_jit_r0_host_plugin(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    source_content_sha256: str | None,
    target: str,
    precision: str,
    schedule: str,
    apply_edge_weights: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Render a contract-specialized host R0 shared-library plugin."""

    metadata = jit_r0_host_metadata(
        contract,
        generation_version=generation_version,
        source_content_sha256=source_content_sha256,
        target=target,
        precision=precision,
        schedule=schedule,
        apply_edge_weights=apply_edge_weights,
    )
    scalar = "float" if precision == "float32" else "double"
    channels = metadata["channels"]
    l_max = metadata["l_max"]
    harmonic_count = metadata["input_components"]
    weight = (
        "static_cast<Scalar>(args->edge_weights[edge])"
        if apply_edge_weights
        else "Scalar(1)"
    )
    density_weight = "args->edge_weights[edge]" if apply_edge_weights else "1.0"
    harmonic_helpers = _render_direct_harmonic_gradient_helper(
        "inline", harmonic_count, index_type="std::int64_t"
    )
    source = f"""// Generated Symmetrix contract-specialized R0 host plugin.
{_R0_HOST_PLUGIN_ABI_HEADER}
#include <cmath>
#include <cstddef>
#include <cstdint>

using Scalar = {scalar};
constexpr int channel_count = {channels};
constexpr int r0_l_max = {l_max};
constexpr int harmonic_count = {harmonic_count};
constexpr int forward_channel_tile = {R0_HOST_FORWARD_CHANNEL_TILE_V1};
constexpr int coordinate_reverse_channel_tile = {R0_HOST_COORDINATE_REVERSE_CHANNEL_TILE_V1};
constexpr std::uint32_t r0_compatibility_version = {R0_COMPATIBILITY_VERSION}u;
static_assert(harmonic_count == (r0_l_max+1)*(r0_l_max+1));

inline Scalar spline_value(
    const SymmetrixJitR0HostSplineV1& spline, int edge_type,
    int function, double radius, Scalar* derivative) {{
    const auto* coefficients = static_cast<const Scalar*>(spline.coefficients);
    int interval = static_cast<int>(std::floor((radius-spline.x0)/spline.h));
    Scalar x = static_cast<Scalar>(radius-spline.x0-spline.h*interval);
    if (interval < 0) {{ interval = 0; x = Scalar(0); }}
    else if (interval >= static_cast<int>(spline.intervals)) {{
        interval = static_cast<int>(spline.intervals)-1;
        x = static_cast<Scalar>(spline.h);
    }}
    const std::size_t base =
        ((static_cast<std::size_t>(edge_type)*spline.intervals+interval)*4)
        *spline.functions+function;
    const Scalar c0 = coefficients[base];
    const Scalar c1 = coefficients[base+spline.functions];
    const Scalar c2 = coefficients[base+2*spline.functions];
    const Scalar c3 = coefficients[base+3*spline.functions];
    *derivative = c1+Scalar(2)*c2*x+Scalar(3)*c3*x*x;
    return c0+c1*x+c2*x*x+c3*x*x*x;
}}

inline double density_value(
    const SymmetrixJitR0HostSplineV1& spline, int edge_type,
    double radius, double* derivative) {{
    const auto* coefficients = static_cast<const double*>(spline.coefficients);
    int interval = static_cast<int>(std::floor((radius-spline.x0)/spline.h));
    double x = radius-spline.x0-spline.h*interval;
    if (interval < 0) {{ interval = 0; x = 0.0; }}
    else if (interval >= static_cast<int>(spline.intervals)) {{
        interval = static_cast<int>(spline.intervals)-1; x = spline.h;
    }}
    const std::size_t base =
        ((static_cast<std::size_t>(edge_type)*spline.intervals+interval)*4)
        *spline.functions;
    const double c0 = coefficients[base];
    const double c1 = coefficients[base+spline.functions];
    const double c2 = coefficients[base+2*spline.functions];
    const double c3 = coefficients[base+3*spline.functions];
    *derivative = c1+2.0*c2*x+3.0*c3*x*x;
    return c0+c1*x+c2*x*x+c3*x*x*x;
}}

{harmonic_helpers}

extern "C" void density_prepare_owner(
    const SymmetrixJitR0HostArgsV1* args, std::int64_t node) {{
    const int receiver_type = args->type_to_active[args->node_types[node]];
    double density = 0.0;
    const int begin = args->first_neigh[node];
    for (int local=0; local<args->num_neigh[node]; ++local) {{
        const int edge = begin+local;
        if (!(args->radius[edge] < args->cutoff)) continue;
        const int neighbor_type = args->type_to_active[args->neigh_types[edge]];
        const int lo = receiver_type < neighbor_type ? receiver_type : neighbor_type;
        const int hi = receiver_type < neighbor_type ? neighbor_type : receiver_type;
        const int edge_type = lo*(2*args->active_type_count-lo-1)/2+hi;
        double unused;
        density += {density_weight}*density_value(
            args->density, edge_type, args->radius[edge], &unused);
    }}
    args->density_state[node] = 1.0/(1.0+density);
}}

extern "C" void forward_owner(
    const SymmetrixJitR0HostArgsV1* args, std::int64_t owner) {{
    constexpr int channel_tile_count =
        (channel_count+forward_channel_tile-1)/forward_channel_tile;
    const std::int64_t node = owner/channel_tile_count;
    const int channel_begin = static_cast<int>(owner%channel_tile_count)
        *forward_channel_tile;
    const int tile_channels = channel_begin+forward_channel_tile <= channel_count
        ? forward_channel_tile : channel_count-channel_begin;
    const int receiver_type = args->type_to_active[args->node_types[node]];
    const int begin = args->first_neigh[node];
    const auto* harmonics = static_cast<const Scalar*>(args->harmonics);
    const auto* coefficients = static_cast<const Scalar*>(args->radial.coefficients);
    const Scalar inverse_scale = args->apply_density_scale
        ? static_cast<Scalar>(args->density_state[node]) : Scalar(1);
    Scalar accumulator[harmonic_count*forward_channel_tile] = {{}};
    for (int local=0; local<args->num_neigh[node]; ++local) {{
        const int edge = begin+local;
        if (!(args->radius[edge] < args->cutoff)) continue;
        const int neighbor_type = args->type_to_active[args->neigh_types[edge]];
        const int edge_type = receiver_type*args->active_type_count+neighbor_type;
        int interval = static_cast<int>(std::floor(
            (args->radius[edge]-args->radial.x0)/args->radial.h));
        Scalar x = static_cast<Scalar>(
            args->radius[edge]-args->radial.x0-args->radial.h*interval);
        if (interval < 0) {{ interval = 0; x = Scalar(0); }}
        else if (interval >= static_cast<int>(args->radial.intervals)) {{
            interval = static_cast<int>(args->radial.intervals)-1;
            x = static_cast<Scalar>(args->radial.h);
        }}
        const Scalar xx = x*x;
        const Scalar xxx = xx*x;
        const Scalar edge_weight = {weight};
        const std::int64_t harmonic_offset =
            static_cast<std::int64_t>(edge)*harmonic_count;
        for (int l=0; l<=r0_l_max; ++l) {{
            const int function_begin = l*channel_count+channel_begin;
            for (int lane=0; lane<tile_channels; ++lane) {{
                const int function = function_begin+lane;
                const std::size_t base =
                    ((static_cast<std::size_t>(edge_type)*args->radial.intervals
                      +interval)*4)*args->radial.functions+function;
                const Scalar value = coefficients[base]
                    +coefficients[base+args->radial.functions]*x
                    +coefficients[base+2*args->radial.functions]*xx
                    +coefficients[base+3*args->radial.functions]*xxx;
                for (int lm=l*l; lm<(l+1)*(l+1); ++lm)
                    accumulator[lm*forward_channel_tile+lane] +=
                        edge_weight*value*harmonics[harmonic_offset+lm];
            }}
        }}
    }}
    auto* output = static_cast<Scalar*>(args->output);
    for (int lm=0; lm<harmonic_count; ++lm)
        for (int lane=0; lane<tile_channels; ++lane)
            output[(node*harmonic_count+lm)*channel_count+channel_begin+lane] =
                accumulator[lm*forward_channel_tile+lane]*inverse_scale;
}}

extern "C" void reverse_prepare_owner(
    const SymmetrixJitR0HostArgsV1* args, std::int64_t node) {{
    auto* output = static_cast<Scalar*>(args->output);
    auto* adjoint = static_cast<Scalar*>(args->output_adjoint);
    constexpr std::int64_t values = harmonic_count*channel_count;
    const std::int64_t base = node*values;
    double dot = args->use_precomputed_scale_adjoint
        ? args->precomputed_scale_adjoint[node] : 0.0;
    if (!args->use_precomputed_scale_adjoint)
        for (std::int64_t value=0; value<values; ++value)
            dot += static_cast<double>(output[base+value])
                *static_cast<double>(adjoint[base+value]);
    const double inverse = args->density_state[node];
    for (std::int64_t value=0; value<values; ++value)
        adjoint[base+value] *= static_cast<Scalar>(inverse);
    args->density_state[node] = dot*inverse;
}}

extern "C" void coordinate_reverse_owner(
    const SymmetrixJitR0HostArgsV1* args, std::int64_t receiver) {{
    const auto* harmonics = static_cast<const Scalar*>(args->harmonics);
    const auto* adjoint = static_cast<const Scalar*>(args->output_adjoint);
    const auto* coefficients = static_cast<const Scalar*>(args->radial.coefficients);
    const int receiver_type = args->type_to_active[args->node_types[receiver]];
    const std::int64_t active_edge_begin = args->edge_begin;
    const std::int64_t active_edge_end = active_edge_begin+args->num_edges;
    const std::int64_t receiver_edge_begin = args->first_neigh[receiver];
    const std::int64_t receiver_edge_end =
        receiver_edge_begin+args->num_neigh[receiver];
    const std::int64_t clipped_edge_begin = receiver_edge_begin > active_edge_begin
        ? receiver_edge_begin : active_edge_begin;
    const std::int64_t clipped_edge_end = receiver_edge_end < active_edge_end
        ? receiver_edge_end : active_edge_end;
    const double density_factor = args->apply_density_scale
        ? args->density_state[receiver] : 0.0;
    const Scalar* gradients = static_cast<const Scalar*>(args->harmonic_gradients);
    for (std::int64_t global_edge=clipped_edge_begin;
         global_edge<clipped_edge_end; ++global_edge) {{
        const std::int64_t edge = global_edge-active_edge_begin;
        if (!(args->radius[edge] < args->cutoff)) continue;
        const int neighbor_type = args->type_to_active[args->neigh_types[edge]];
        const int edge_type = receiver_type*args->active_type_count+neighbor_type;
        int interval = static_cast<int>(std::floor(
            (args->radius[edge]-args->radial.x0)/args->radial.h));
        Scalar x = static_cast<Scalar>(
            args->radius[edge]-args->radial.x0-args->radial.h*interval);
        if (interval < 0) {{ interval = 0; x = Scalar(0); }}
        else if (interval >= static_cast<int>(args->radial.intervals)) {{
            interval = static_cast<int>(args->radial.intervals)-1;
            x = static_cast<Scalar>(args->radial.h);
        }}
        const Scalar xx = x*x;
        const Scalar xxx = xx*x;
        const std::size_t coordinate_offset = 3*static_cast<std::size_t>(edge);
        const std::size_t harmonic_offset =
            static_cast<std::size_t>(edge)*harmonic_count;
        const Scalar x_over_r = args->coordinates_are_unit
            ? static_cast<Scalar>(
                static_cast<const float*>(args->coordinates)[3*edge])
            : static_cast<Scalar>(
                static_cast<const double*>(args->coordinates)[3*edge]
                /args->radius[edge]);
        const Scalar y_over_r = args->coordinates_are_unit
            ? static_cast<Scalar>(
                static_cast<const float*>(args->coordinates)[3*edge+1])
            : static_cast<Scalar>(
                static_cast<const double*>(args->coordinates)[3*edge+1]
                /args->radius[edge]);
        const Scalar z_over_r = args->coordinates_are_unit
            ? static_cast<Scalar>(
                static_cast<const float*>(args->coordinates)[3*edge+2])
            : static_cast<Scalar>(
                static_cast<const double*>(args->coordinates)[3*edge+2]
                /args->radius[edge]);
        Scalar direct_gradients[3*harmonic_count];
        const Scalar* edge_gradients = gradients;
        if (edge_gradients == nullptr) {{
            direct_harmonic_gradients(
                args->coordinates,
                args->coordinates_are_unit ? sizeof(float) : sizeof(double),
                args->coordinates_are_unit, coordinate_offset,
                args->radius[edge], direct_gradients);
            edge_gradients = direct_gradients;
        }} else edge_gradients += coordinate_offset*harmonic_count;
        const Scalar edge_weight = {weight};
        Scalar force_x = Scalar(0);
        Scalar force_y = Scalar(0);
        Scalar force_z = Scalar(0);
        for (int channel_begin=0; channel_begin<channel_count;
             channel_begin+=coordinate_reverse_channel_tile) {{
            const int tile_channels =
                channel_begin+coordinate_reverse_channel_tile <= channel_count
                ? coordinate_reverse_channel_tile : channel_count-channel_begin;
            Scalar lane_force_x[coordinate_reverse_channel_tile] = {{}};
            Scalar lane_force_y[coordinate_reverse_channel_tile] = {{}};
            Scalar lane_force_z[coordinate_reverse_channel_tile] = {{}};
            for (int l=0; l<=r0_l_max; ++l) {{
                Scalar values[coordinate_reverse_channel_tile];
                Scalar derivatives[coordinate_reverse_channel_tile];
                const int function_begin = l*channel_count+channel_begin;
                for (int lane=0; lane<tile_channels; ++lane) {{
                    const int function = function_begin+lane;
                    const std::size_t base =
                        ((static_cast<std::size_t>(edge_type)*args->radial.intervals
                          +interval)*4)*args->radial.functions+function;
                    const Scalar c1 = coefficients[base+args->radial.functions];
                    const Scalar c2 = coefficients[base+2*args->radial.functions];
                    const Scalar c3 = coefficients[base+3*args->radial.functions];
                    values[lane] = coefficients[base]+c1*x+c2*xx+c3*xxx;
                    derivatives[lane] = c1+Scalar(2)*c2*x+Scalar(3)*c3*xx;
                }}
                for (int lm=l*l; lm<(l+1)*(l+1); ++lm) {{
                    const Scalar harmonic = harmonics[harmonic_offset+lm];
                    const Scalar gradient_x = edge_gradients[lm];
                    const Scalar gradient_y = edge_gradients[harmonic_count+lm];
                    const Scalar gradient_z = edge_gradients[2*harmonic_count+lm];
                    for (int lane=0; lane<tile_channels; ++lane) {{
                        const int channel = channel_begin+lane;
                        const Scalar value = edge_weight*adjoint[
                            (receiver*harmonic_count+lm)*channel_count+channel];
                        const Scalar radial = derivatives[lane]*harmonic*value;
                        const Scalar angular = values[lane]*value;
                        lane_force_x[lane] += radial*x_over_r+angular*gradient_x;
                        lane_force_y[lane] += radial*y_over_r+angular*gradient_y;
                        lane_force_z[lane] += radial*z_over_r+angular*gradient_z;
                    }}
                }}
            }}
            for (int lane=0; lane<tile_channels; ++lane) {{
                force_x += lane_force_x[lane];
                force_y += lane_force_y[lane];
                force_z += lane_force_z[lane];
            }}
        }}
        args->directed_forces[coordinate_offset] -= force_x;
        args->directed_forces[coordinate_offset+1] -= force_y;
        args->directed_forces[coordinate_offset+2] -= force_z;
        if (args->apply_density_scale) {{
            const int lo = receiver_type < neighbor_type
                ? receiver_type : neighbor_type;
            const int hi = receiver_type < neighbor_type
                ? neighbor_type : receiver_type;
            const int density_type =
                lo*(2*args->active_type_count-lo-1)/2+hi;
            double derivative;
            (void)density_value(
                args->density, density_type, args->radius[edge], &derivative);
            const double contribution = static_cast<double>(edge_weight)
                *density_factor*derivative;
            args->directed_forces[coordinate_offset] += contribution*x_over_r;
            args->directed_forces[coordinate_offset+1] += contribution*y_over_r;
            args->directed_forces[coordinate_offset+2] += contribution*z_over_r;
        }}
    }}
}}

extern "C" void density_prepare_range(
    const SymmetrixJitR0HostArgsV1* args,
    std::int64_t begin, std::int64_t end) {{
    for (std::int64_t owner=begin; owner<end; ++owner)
        density_prepare_owner(args, owner);
}}

extern "C" void forward_range(
    const SymmetrixJitR0HostArgsV1* args,
    std::int64_t begin, std::int64_t end) {{
    for (std::int64_t owner=begin; owner<end; ++owner)
        forward_owner(args, owner);
}}

extern "C" void reverse_prepare_range(
    const SymmetrixJitR0HostArgsV1* args,
    std::int64_t begin, std::int64_t end) {{
    for (std::int64_t owner=begin; owner<end; ++owner)
        reverse_prepare_owner(args, owner);
}}

extern "C" void coordinate_reverse_range(
    const SymmetrixJitR0HostArgsV1* args,
    std::int64_t begin, std::int64_t end) {{
    for (std::int64_t owner=begin; owner<end; ++owner)
        coordinate_reverse_owner(args, owner);
}}

extern "C" const SymmetrixJitR0HostPluginV2*
symmetrix_jit_r0_host_plugin_query_v2() {{
    static const SymmetrixJitR0HostPluginV2 descriptor = {{
        SYMMETRIX_JIT_R0_HOST_PLUGIN_ABI_VERSION_V2,
        sizeof(SymmetrixJitR0HostPluginV2), sizeof(void*),
        SYMMETRIX_JIT_R0_HOST_PLUGIN_BYTE_ORDER_V1,
        {metadata["scalar_kind"]}u, {metadata["scalar_size"]}u,
        {metadata["capabilities"]}u, channel_count, r0_l_max, harmonic_count,
        {1 if apply_edge_weights else 0},
        {cpp_text_literal(metadata["schema"])},
        {cpp_text_literal(metadata["artifact_id"])},
        {cpp_text_literal(metadata["structure_fingerprint"])},
        &density_prepare_owner, &forward_owner, &reverse_prepare_owner,
        &coordinate_reverse_owner,
        {cpp_text_literal(metadata["target"])},
        {cpp_text_literal(metadata["schedule"])},
        {metadata["schedule_revision"]}, r0_compatibility_version,
        &density_prepare_range, &forward_range, &reverse_prepare_range,
        &coordinate_reverse_range
    }};
    return &descriptor;
}}
"""
    return source, metadata


def _r0_prefix(metadata: Mapping[str, Any]) -> str:
    scalar = "float" if metadata["precision"] == "float32" else "double"
    bounds = metadata["architecture_bounds"]
    policy = metadata["schedule_policy"]
    geometry = metadata["launch_geometry"]
    return f"""// Generated Symmetrix state-free R0 operator module.
namespace std {{ using size_t = decltype(sizeof(0)); }}
using Scalar = {scalar};
constexpr int channel_count = {bounds["channels"]};
constexpr int r0_l_max = {bounds["l_max"]};
constexpr int harmonic_count = {bounds["harmonic_count"]};
constexpr int r0_threads_per_block = {metadata["threads_per_block"]};
constexpr int density_threads_per_block = {geometry["density_prepare"]["threads_per_block"]};
constexpr int forward_threads_per_block = {geometry["forward"]["threads_per_block"]};
constexpr int reverse_prepare_threads_per_block = {geometry["reverse_prepare"]["threads_per_block"]};
constexpr int coordinate_reverse_threads_per_block = {geometry["coordinate_reverse"]["threads_per_block"]};
constexpr int edge_team_size = {policy["edge_team_size"]};
constexpr int edge_vector_length = {policy["edge_vector_length"]};
constexpr int forward_teams_per_block = {geometry["forward"]["owners_per_block"]};
static_assert(harmonic_count == (r0_l_max+1)*(r0_l_max+1));
static_assert(edge_team_size*edge_vector_length == r0_threads_per_block);
struct Spline {{
    unsigned int struct_size;
    unsigned int edge_types;
    unsigned int intervals;
    unsigned int functions;
    double h;
    double x0;
    const void* coefficients_raw;
}};
struct R0Args {{
    unsigned int struct_size;
    unsigned int receiver_base;
    long long num_nodes;
    long long num_edges;
    int active_type_count;
    int channels;
    int l_max;
    int coordinates_are_unit;
    int apply_density_scale;
    int use_precomputed_scale_adjoint;
    const int* node_types;
    const int* num_neigh;
    const int* first_neigh;
    const int* neigh_types;
    const int* edge_receivers;
    const int* type_to_active;
    const void* coordinates;
    const double* radius;
    Spline radial;
    Spline density;
    const void* harmonics_raw;
    const void* harmonic_gradients_raw;
    void* output_raw;
    void* output_adjoint_raw;
    double* density_state;
    const double* precomputed_scale_adjoint;
    double* directed_forces;
    double cutoff;
    const double* edge_weights;
}};
extern "C" __device__ __constant__ unsigned int symmetrix_operator_abi_version_v1 = {R0_DEVICE_ABI_VERSION}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_kind_v1 = 2u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_kind_v1 = {metadata["scalar_kind"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_size_v1 = {metadata["scalar_size"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_capabilities_v1 = {metadata["capabilities"]}u;
extern "C" __device__ __constant__ int symmetrix_operator_input_components_v1 = {metadata["input_components"]};
extern "C" __device__ __constant__ int symmetrix_operator_output_components_v1 = {metadata["output_components"]};
extern "C" __device__ __constant__ int symmetrix_operator_correlation_v1 = 1;
extern "C" __device__ __constant__ int symmetrix_operator_term_count_v1 = {metadata["term_count"]};
extern "C" __device__ __constant__ int symmetrix_operator_threads_v1 = {metadata["threads_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_schedule_revision_v1 = {metadata["schedule_revision"]};
extern "C" __device__ __constant__ int symmetrix_operator_persistent_blocks_per_compute_unit_v1 = {metadata["persistent_profile"]["blocks_per_compute_unit"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_channels_v1 = channel_count;
extern "C" __device__ __constant__ int symmetrix_operator_r0_l_max_v1 = r0_l_max;
extern "C" __device__ __constant__ int symmetrix_operator_r0_harmonic_count_v1 = harmonic_count;
extern "C" __device__ __constant__ int symmetrix_operator_r0_edge_weight_mode_v1 = {1 if metadata["edge_weight_mode"] == "weighted" else 0};
extern "C" __device__ __constant__ unsigned int symmetrix_operator_r0_compatibility_version_v1 = {R0_COMPATIBILITY_VERSION}u;
extern "C" __device__ __constant__ int symmetrix_operator_r0_density_extent_multiplier_v1 = {geometry["density_prepare"]["extent_multiplier"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_density_owners_per_block_v1 = {geometry["density_prepare"]["owners_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_density_threads_per_block_v1 = {geometry["density_prepare"]["threads_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_density_persistent_v1 = {1 if geometry["density_prepare"]["persistent"] else 0};
extern "C" __device__ __constant__ int symmetrix_operator_r0_forward_extent_multiplier_v1 = {geometry["forward"]["extent_multiplier"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_forward_owners_per_block_v1 = {geometry["forward"]["owners_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_forward_threads_per_block_v1 = {geometry["forward"]["threads_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_forward_persistent_v1 = {1 if geometry["forward"]["persistent"] else 0};
extern "C" __device__ __constant__ int symmetrix_operator_r0_reverse_prepare_extent_multiplier_v1 = {geometry["reverse_prepare"]["extent_multiplier"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_reverse_prepare_owners_per_block_v1 = {geometry["reverse_prepare"]["owners_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_reverse_prepare_threads_per_block_v1 = {geometry["reverse_prepare"]["threads_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_reverse_prepare_persistent_v1 = {1 if geometry["reverse_prepare"]["persistent"] else 0};
extern "C" __device__ __constant__ int symmetrix_operator_r0_coordinate_reverse_extent_multiplier_v1 = {geometry["coordinate_reverse"]["extent_multiplier"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_coordinate_reverse_owners_per_block_v1 = {geometry["coordinate_reverse"]["owners_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_coordinate_reverse_threads_per_block_v1 = {geometry["coordinate_reverse"]["threads_per_block"]};
extern "C" __device__ __constant__ int symmetrix_operator_r0_coordinate_reverse_persistent_v1 = {1 if geometry["coordinate_reverse"]["persistent"] else 0};
extern "C" __device__ __constant__ char symmetrix_operator_artifact_id_v1[] = {cpp_text_literal(metadata["artifact_id"])};
extern "C" __device__ __constant__ char symmetrix_operator_structure_fingerprint_v1[] = {cpp_text_literal(metadata["structure_fingerprint"])};
extern "C" __device__ __constant__ char symmetrix_operator_target_v1[] = {cpp_text_literal(metadata["target"])};
extern "C" __device__ __constant__ char symmetrix_operator_r0_schedule_v1[] = {cpp_text_literal(metadata["schedule"])};

struct RadialSplineCoordinates {{
    int interval;
    Scalar x;
    Scalar x2;
    Scalar x3;
}};

__device__ __forceinline__ RadialSplineCoordinates radial_spline_coordinates(
    const Spline spline, double radius) {{
    int interval = static_cast<int>(floor((radius-spline.x0)/spline.h));
    Scalar x = static_cast<Scalar>(radius-spline.x0-spline.h*interval);
    if (interval < 0) {{ interval = 0; x = Scalar(0); }}
    else if (interval >= static_cast<int>(spline.intervals)) {{
        interval = static_cast<int>(spline.intervals)-1;
        x = static_cast<Scalar>(spline.h);
    }}
    return {{interval, x, x*x, x*x*x}};
}}

__device__ __forceinline__ Scalar radial_spline_from_coordinates(
    const Spline spline, int edge_type, int function,
    const RadialSplineCoordinates coordinates, Scalar* derivative) {{
    const Scalar* coefficients = static_cast<const Scalar*>(spline.coefficients_raw);
    const std::size_t base =
        ((static_cast<std::size_t>(edge_type)*spline.intervals+coordinates.interval)*4)
        *spline.functions+function;
    const Scalar c0 = coefficients[base];
    const Scalar c1 = coefficients[base+spline.functions];
    const Scalar c2 = coefficients[base+2*spline.functions];
    const Scalar c3 = coefficients[base+3*spline.functions];
    *derivative = c1+Scalar(2)*c2*coordinates.x
        +Scalar(3)*c3*coordinates.x2;
    return c0+c1*coordinates.x+c2*coordinates.x2+c3*coordinates.x3;
}}

__device__ __forceinline__ Scalar radial_spline(
    const Spline spline, int edge_type, int function, double radius,
    Scalar* derivative) {{
    return radial_spline_from_coordinates(
        spline, edge_type, function,
        radial_spline_coordinates(spline, radius), derivative);
}}

template<class Value>
__device__ __forceinline__ Value subgroup_sum(Value value, int width) {{
#if defined(__HIP_DEVICE_COMPILE__)
    for (int offset=width/2; offset>0; offset/=2)
        value += __shfl_down(value, offset, width);
#else
    for (int offset=width/2; offset>0; offset/=2)
        value += __shfl_down_sync(0xffffffffu, value, offset, width);
#endif
    return value;
}}

__device__ __forceinline__ void subgroup_sync(int width) {{
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_wave_barrier();
#else
    const unsigned int lane = threadIdx.x%32;
    const unsigned int first_lane = (lane/width)*width;
    const unsigned int mask = width == 32
        ? 0xffffffffu : ((1u << width)-1u) << first_lane;
    __syncwarp(mask);
#endif
}}

__device__ __forceinline__ bool edge_is_active(
    double cutoff, double radius) {{
    return radius < cutoff;
}}

__device__ __forceinline__ double density_spline(
    const Spline spline, int edge_type, double radius, double* derivative) {{
    const double* coefficients = static_cast<const double*>(spline.coefficients_raw);
    int interval = static_cast<int>(floor((radius-spline.x0)/spline.h));
    double x = radius-spline.x0-spline.h*interval;
    if (interval < 0) {{ interval = 0; x = 0.0; }}
    else if (interval >= static_cast<int>(spline.intervals)) {{
        interval = static_cast<int>(spline.intervals)-1;
        x = spline.h;
    }}
    const std::size_t base =
        ((static_cast<std::size_t>(edge_type)*spline.intervals+interval)*4)
        *spline.functions;
    const double c0 = coefficients[base];
    const double c1 = coefficients[base+spline.functions];
    const double c2 = coefficients[base+2*spline.functions];
    const double c3 = coefficients[base+3*spline.functions];
    *derivative = c1+2.0*c2*x+3.0*c3*x*x;
    return c0+c1*x+c2*x*x+c3*x*x*x;
}}

__device__ __forceinline__ double direction(
    const R0Args args, long long edge, int axis) {{
    if (args.coordinates_are_unit)
        return static_cast<double>(
            static_cast<const float*>(args.coordinates)[3*edge+axis]);
    return static_cast<const double*>(args.coordinates)[3*edge+axis]
        /args.radius[edge];
}}
"""


def _r0_kernels(metadata: Mapping[str, Any], *, apply_edge_weights: bool) -> str:
    forward_owner = metadata["schedule_policy"]["forward_owner"]
    if forward_owner in (
        "receiver-channel-all-angular",
        "receiver-channel-all-angular-fallback",
    ):
        forward = r"""
extern "C" __global__ __launch_bounds__(forward_threads_per_block)
void symmetrix_r0_forward_v1(R0Args args) {
    const Scalar* harmonics = static_cast<const Scalar*>(args.harmonics_raw);
    Scalar* output = static_cast<Scalar*>(args.output_raw);
    const long long owners = args.num_nodes*channel_count;
    const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
    for (long long owner = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
         owner < owners; owner += stride) {
        const long long receiver = owner/channel_count;
        const int channel = static_cast<int>(owner%channel_count);
        const int receiver_type = args.type_to_active[args.node_types[receiver]];
        const int edge_begin = args.first_neigh[receiver];
        Scalar accumulator[harmonic_count] = {};
        for (int local_edge=0; local_edge<args.num_neigh[receiver]; ++local_edge) {
            const int edge = edge_begin+local_edge;
            if (!edge_is_active(args.cutoff, args.radius[edge])) continue;
            const int neighbor_type = args.type_to_active[args.neigh_types[edge]];
            const int edge_type = receiver_type*args.active_type_count+neighbor_type;
            const Scalar weight = R0_EDGE_WEIGHT_SCALAR;
            const long long harmonic_offset =
                static_cast<long long>(edge)*harmonic_count;
            for (int l=0; l<=r0_l_max; ++l) {
                Scalar unused;
                const Scalar radial = radial_spline(
                    args.radial, edge_type, l*channel_count+channel,
                    args.radius[edge], &unused);
                for (int lm=l*l; lm<(l+1)*(l+1); ++lm)
                    accumulator[lm] += weight*radial
                        *harmonics[harmonic_offset+lm];
            }
        }
        const Scalar inverse_scale = args.apply_density_scale
            ? static_cast<Scalar>(args.density_state[receiver]) : Scalar(1);
        for (int lm=0; lm<harmonic_count; ++lm)
            output[(receiver*harmonic_count+lm)*channel_count+channel] =
                accumulator[lm]*inverse_scale;
    }
}
"""
    else:
        forward = r"""
extern "C" __global__ __launch_bounds__(forward_threads_per_block)
void symmetrix_r0_forward_v1(R0Args args) {
    const Scalar* harmonics = static_cast<const Scalar*>(args.harmonics_raw);
    Scalar* output = static_cast<Scalar*>(args.output_raw);
    __shared__ Scalar accumulator_storage[
        forward_teams_per_block*channel_count*(2*r0_l_max+1)];
    const int forward_team = threadIdx.x/32;
    if (forward_team >= forward_teams_per_block) return;
    const int lane = threadIdx.x%32;
    const long long receiver_l_count = args.num_nodes*(r0_l_max+1);
    for (long long receiver_l =
             static_cast<long long>(blockIdx.x)*forward_teams_per_block+forward_team;
         receiver_l < receiver_l_count;
         receiver_l += static_cast<long long>(gridDim.x)*forward_teams_per_block) {
        const long long receiver = receiver_l/(r0_l_max+1);
        const int angular_order = receiver_l%(r0_l_max+1);
        const int lm_begin = angular_order*angular_order;
        const int component_count = 2*angular_order+1;
        const int receiver_type = args.type_to_active[args.node_types[receiver]];
        const int edge_begin = args.first_neigh[receiver];
        const Scalar inverse_scale = args.apply_density_scale
            ? static_cast<Scalar>(args.density_state[receiver]) : Scalar(1);
        Scalar* accumulator = accumulator_storage
            +forward_team*channel_count*(2*r0_l_max+1);
        for (int channel=lane; channel<channel_count; channel+=32)
            for (int component=0; component<component_count; ++component)
                accumulator[component*channel_count+channel] = Scalar(0);
        for (int local_edge=0;
             local_edge<args.num_neigh[receiver]; ++local_edge) {
            const int edge = edge_begin+local_edge;
            if (!edge_is_active(args.cutoff, args.radius[edge])) continue;
            const int neighbor_type =
                args.type_to_active[args.neigh_types[edge]];
            const int edge_type =
                receiver_type*args.active_type_count+neighbor_type;
            const RadialSplineCoordinates spline_coordinates =
                radial_spline_coordinates(args.radial, args.radius[edge]);
            const long long harmonic_offset =
                static_cast<long long>(edge)*harmonic_count+lm_begin;
            for (int channel=lane; channel<channel_count; channel+=32) {
                Scalar unused;
                const Scalar radial = radial_spline_from_coordinates(
                    args.radial, edge_type,
                    angular_order*channel_count+channel,
                    spline_coordinates, &unused);
                const Scalar weighted_radial = R0_EDGE_WEIGHT_SCALAR*radial;
                switch (component_count) {
                default:
                    for (int component=0; component<component_count; ++component)
                        accumulator[component*channel_count+channel] += weighted_radial
                            *harmonics[harmonic_offset+component];
                    break;
                case 7:
                    accumulator[5*channel_count+channel] += weighted_radial
                        *harmonics[harmonic_offset+5];
                    accumulator[6*channel_count+channel] += weighted_radial
                        *harmonics[harmonic_offset+6];
                    [[fallthrough]];
                case 5:
                    accumulator[3*channel_count+channel] += weighted_radial
                        *harmonics[harmonic_offset+3];
                    accumulator[4*channel_count+channel] += weighted_radial
                        *harmonics[harmonic_offset+4];
                    [[fallthrough]];
                case 3:
                    accumulator[channel_count+channel] += weighted_radial
                        *harmonics[harmonic_offset+1];
                    accumulator[2*channel_count+channel] += weighted_radial
                        *harmonics[harmonic_offset+2];
                    [[fallthrough]];
                case 1:
                    accumulator[channel] += weighted_radial
                        *harmonics[harmonic_offset];
                }
            }
        }
        for (int channel=lane; channel<channel_count; channel+=32)
            for (int component=0; component<component_count; ++component)
                output[(receiver*harmonic_count+lm_begin+component)
                    *channel_count+channel] =
                    accumulator[component*channel_count+channel]*inverse_scale;
    }
}
"""
    source = r"""
extern "C" __global__ __launch_bounds__(density_threads_per_block)
void symmetrix_r0_density_prepare_v1(R0Args args) {
    const int density_team = threadIdx.x/32;
    const int lane = threadIdx.x%32;
    constexpr int density_teams_per_block = density_threads_per_block/32;
    for (long long node =
             static_cast<long long>(blockIdx.x)*density_teams_per_block+density_team;
         node < args.num_nodes;
         node += static_cast<long long>(gridDim.x)*density_teams_per_block) {
        const int receiver_type = args.type_to_active[args.node_types[node]];
        const int begin = args.first_neigh[node];
        double density = 0.0;
        for (int local=lane; local<args.num_neigh[node]; local+=32) {
            const int edge = begin+local;
            if (!edge_is_active(args.cutoff, args.radius[edge])) continue;
            const int neighbor_type = args.type_to_active[args.neigh_types[edge]];
            const int lo = receiver_type <= neighbor_type
                ? receiver_type : neighbor_type;
            const int hi = receiver_type <= neighbor_type
                ? neighbor_type : receiver_type;
            const int edge_type =
                lo*(2*args.active_type_count-lo-1)/2+hi;
            double unused;
            const double weight = R0_EDGE_WEIGHT_DOUBLE;
            density += weight*density_spline(
                args.density, edge_type, args.radius[edge], &unused);
        }
        density = subgroup_sum(density, 32);
        if (lane == 0)
            args.density_state[node] = 1.0/(1.0+density);
    }
}

R0_FORWARD_KERNEL

extern "C" __global__ __launch_bounds__(reverse_prepare_threads_per_block)
void symmetrix_r0_reverse_prepare_v1(R0Args args) {
    Scalar* output = static_cast<Scalar*>(args.output_raw);
    Scalar* output_adjoint = static_cast<Scalar*>(args.output_adjoint_raw);
    constexpr int value_count = harmonic_count*channel_count;
    __shared__ double partial[reverse_prepare_threads_per_block];
    for (long long receiver=blockIdx.x; receiver<args.num_nodes;
         receiver+=gridDim.x) {
        const long long base = receiver*value_count;
        double dot = 0.0;
        if (!args.use_precomputed_scale_adjoint)
            for (int value=threadIdx.x; value<value_count; value+=blockDim.x)
                dot += static_cast<double>(output[base+value])
                    *static_cast<double>(output_adjoint[base+value]);
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (int offset=reverse_prepare_threads_per_block/2; offset>0; offset/=2) {
            if (threadIdx.x < offset)
                partial[threadIdx.x] += partial[threadIdx.x+offset];
            __syncthreads();
        }
        const double inverse = args.density_state[receiver];
        for (int value=threadIdx.x; value<value_count; value+=blockDim.x)
            output_adjoint[base+value] *= static_cast<Scalar>(inverse);
        if (threadIdx.x == 0) {
            const double result = args.use_precomputed_scale_adjoint
                ? args.precomputed_scale_adjoint[receiver] : partial[0];
            args.density_state[receiver] = result*inverse;
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(coordinate_reverse_threads_per_block)
void symmetrix_r0_coordinate_reverse_v1(R0Args args) {
    const Scalar* harmonics = static_cast<const Scalar*>(args.harmonics_raw);
    const Scalar* gradients =
        static_cast<const Scalar*>(args.harmonic_gradients_raw);
    const Scalar* output_adjoint =
        static_cast<const Scalar*>(args.output_adjoint_raw);
    extern __shared__ Scalar direct_gradient_storage[];
    const int edge_team = threadIdx.x/edge_vector_length;
    const int lane = threadIdx.x%edge_vector_length;
    for (long long edge_base=
             static_cast<long long>(blockIdx.x)*edge_team_size;
         edge_base<args.num_edges;
         edge_base+=static_cast<long long>(gridDim.x)*edge_team_size) {
        const long long edge = edge_base+edge_team;
        const bool edge_active = edge < args.num_edges
            && edge_is_active(args.cutoff, args.radius[edge]);
        long long receiver = 0;
        int receiver_type = 0;
        int neighbor_type = 0;
        int edge_type = 0;
        long long coordinate_offset = 0;
        Scalar weight = Scalar(1);
        Scalar direction_x = Scalar(0);
        Scalar direction_y = Scalar(0);
        Scalar direction_z = Scalar(0);
        RadialSplineCoordinates spline_coordinates {};
        if (edge_active) {
            weight = R0_EDGE_WEIGHT_SCALAR;
            receiver = static_cast<long long>(args.edge_receivers[edge])
                -args.receiver_base;
            receiver_type = args.type_to_active[args.node_types[receiver]];
            neighbor_type = args.type_to_active[args.neigh_types[edge]];
            edge_type = receiver_type*args.active_type_count+neighbor_type;
            coordinate_offset = 3*edge;
            direction_x = static_cast<Scalar>(direction(args, edge, 0));
            direction_y = static_cast<Scalar>(direction(args, edge, 1));
            direction_z = static_cast<Scalar>(direction(args, edge, 2));
            spline_coordinates =
                radial_spline_coordinates(args.radial, args.radius[edge]);
        }
        Scalar* direct_gradients = direct_gradient_storage
            +edge_team*3*harmonic_count;
        if (lane == 0 && edge_active && gradients == nullptr)
            direct_harmonic_gradients(
                args.coordinates,
                args.coordinates_are_unit ? sizeof(float) : sizeof(double),
                args.coordinates_are_unit, coordinate_offset,
                args.radius[edge], direct_gradients);
        subgroup_sync(edge_vector_length);
        const Scalar* edge_gradients = gradients == nullptr
            ? direct_gradients : gradients+coordinate_offset*harmonic_count;
        Scalar force_x = Scalar(0);
        Scalar force_y = Scalar(0);
        Scalar force_z = Scalar(0);
        if (edge_active)
            for (int channel=lane; channel<channel_count;
                 channel+=edge_vector_length) {
                for (int l=0; l<=r0_l_max; ++l) {
                    const int lm_begin = l*l;
                    const int lm_end = (l+1)*(l+1);
                    Scalar radial_derivative;
                    const Scalar radial = radial_spline_from_coordinates(
                        args.radial, edge_type, l*channel_count+channel,
                        spline_coordinates, &radial_derivative);
                    for (int lm=lm_begin; lm<lm_end; ++lm) {
                        const Scalar adjoint = weight*output_adjoint[
                            (receiver*harmonic_count+lm)*channel_count+channel];
                        const Scalar radial_scale = adjoint*radial_derivative
                            *harmonics[edge*harmonic_count+lm];
                        const Scalar angular_scale = adjoint*radial;
                        force_x += radial_scale*direction_x
                            +angular_scale*edge_gradients[lm];
                        force_y += radial_scale*direction_y
                            +angular_scale*edge_gradients[harmonic_count+lm];
                        force_z += radial_scale*direction_z
                            +angular_scale*edge_gradients[2*harmonic_count+lm];
                    }
                }
            }
        force_x = subgroup_sum(force_x, edge_vector_length);
        force_y = subgroup_sum(force_y, edge_vector_length);
        force_z = subgroup_sum(force_z, edge_vector_length);
        if (lane == 0 && edge_active) {
            args.directed_forces[coordinate_offset] -=
                static_cast<double>(force_x);
            args.directed_forces[coordinate_offset+1] -=
                static_cast<double>(force_y);
            args.directed_forces[coordinate_offset+2] -=
                static_cast<double>(force_z);
        }
        if (lane == 0 && edge_active && args.apply_density_scale) {
            const int lo = receiver_type <= neighbor_type
                ? receiver_type : neighbor_type;
            const int hi = receiver_type <= neighbor_type
                ? neighbor_type : receiver_type;
            const int density_type =
                lo*(2*args.active_type_count-lo-1)/2+hi;
            double density_derivative;
            (void)density_spline(
                args.density, density_type, args.radius[edge],
                &density_derivative);
            const double contribution = static_cast<double>(weight)
                *args.density_state[receiver]*density_derivative;
            args.directed_forces[coordinate_offset] += contribution*direction_x;
            args.directed_forces[coordinate_offset+1] += contribution*direction_y;
            args.directed_forces[coordinate_offset+2] += contribution*direction_z;
        }
        subgroup_sync(edge_vector_length);
    }
}
"""
    source = source.replace("R0_FORWARD_KERNEL", forward)
    source = source.replace(
        "R0_EDGE_WEIGHT_DOUBLE",
        "args.edge_weights[edge]" if apply_edge_weights else "1.0",
    )
    return source.replace(
        "R0_EDGE_WEIGHT_SCALAR",
        (
            "static_cast<Scalar>(args.edge_weights[edge])"
            if apply_edge_weights
            else "Scalar(1)"
        ),
    )


def render_jit_r0_device_module(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    source_content_sha256: str | None,
    backend: str,
    precision: str,
    target: str,
    schedule: str | None,
    floating_point_policy: str | None = None,
    apply_edge_weights: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Render a self-contained CUDA/HIP RTC scalar-source R0 module."""

    metadata = jit_r0_device_metadata(
        contract,
        generation_version=generation_version,
        source_content_sha256=source_content_sha256,
        backend=backend,
        precision=precision,
        target=target,
        schedule=schedule,
        floating_point_policy=floating_point_policy,
        apply_edge_weights=apply_edge_weights,
    )
    harmonic_count = metadata["input_components"]
    if harmonic_count <= 16:
        harmonic_helpers = _render_direct_harmonic_gradient_helper(
            "__device__ __forceinline__", harmonic_count, index_type="int"
        )
    else:
        harmonic_helpers = """__device__ __forceinline__ void direct_harmonic_gradients(
    const void*, unsigned int, unsigned int, std::size_t, double, Scalar*) {}
"""
    return (
        _r0_prefix(metadata)
        + harmonic_helpers
        + "\n\n"
        + _r0_kernels(metadata, apply_edge_weights=apply_edge_weights),
        metadata,
    )


def render_generated_r0_operator(
    contract: Mapping[str, Any],
    *,
    generation_version: int,
    source_content_sha256: str,
    backend: str,
    target: str,
    precision: str,
    schedule: str,
    floating_point_policy: str | None = None,
    apply_edge_weights: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Render R0 solely from explicit build/runtime inputs."""

    choose_r0_schedule(backend=backend, schedule=schedule)
    if backend == "host":
        return render_jit_r0_host_plugin(
            contract,
            generation_version=generation_version,
            source_content_sha256=source_content_sha256,
            target=target,
            precision=precision,
            schedule=schedule,
            apply_edge_weights=apply_edge_weights,
        )
    return render_jit_r0_device_module(
        contract,
        generation_version=generation_version,
        source_content_sha256=source_content_sha256,
        backend=backend,
        precision=precision,
        target=target,
        schedule=schedule,
        floating_point_policy=floating_point_policy,
        apply_edge_weights=apply_edge_weights,
    )


__all__ = [
    "M0_CAPABILITIES_V1",
    "R0_CAPABILITIES_V1",
    "choose_r0_schedule",
    "generated_r0_identity",
    "jit_m0_device_metadata",
    "jit_r0_device_metadata",
    "jit_r0_host_metadata",
    "normalize_generated_r0_contract",
    "render_generated_r0_operator",
    "render_jit_m0_device_module",
    "render_jit_r0_device_module",
    "render_jit_r0_host_plugin",
]
