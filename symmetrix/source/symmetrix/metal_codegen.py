"""Render Metal Shading Language kernels for the FP32 R1 interaction.

The kernels reuse the host R1 owner renderers with single-channel tiles, so one
GPU thread owns one (node, channel) output in the forward and source passes and
one edge in the force pass. Metal has no FP64, so the Metal packets carry FP32
radii, cutoff, unit directions, and directed forces; the host converts the
double-precision evaluator views at the packet boundary.

The packet layouts must match ``libsymmetrix/source/metal/metal_r1_abi.hpp``.
"""

from __future__ import annotations

import re
from collections import Counter

from . import jit_codegen as _host
from .execution_contract import normalize_jit_r1_contract

METAL_R1_ABI_VERSION = 1

_PRELUDE = """#include <metal_stdlib>
using namespace metal;

typedef float Scalar;

struct MetalR1RadialSpline {
    uint edge_types;
    uint intervals;
    uint functions;
    float h;
    float x0;
    uint reserved;
    device const float4* coefficients;
};

struct MetalR1ForwardArgs {
    uint active_type_count;
    uint num_nodes;
    float cutoff;
    uint reserved;
    device const int* node_types;
    device const int* num_neigh;
    device const int* first_neigh;
    device const int* neigh_indices;
    device const int* neigh_types;
    device const int* type_to_active;
    device const float* radius;
    MetalR1RadialSpline radial;
    device const float* harmonics_values;
    device const float* neighbor_features;
    device float* output;
};

struct MetalR1SourceArgs {
    uint active_type_count;
    uint num_nodes;
    float cutoff;
    uint reserved;
    device const int* node_types;
    device const int* neigh_types;
    device const int* source_offsets;
    device const int* source_edges;
    device const int* edge_receivers;
    device const int* type_to_active;
    device const float* radius;
    MetalR1RadialSpline radial;
    device const float* harmonics_values;
    device const float* output_adjoint;
    device float* source_adjoint;
};

struct MetalR1EdgeArgs {
    uint active_type_count;
    uint num_edges;
    float cutoff;
    uint has_gradients;
    device const int* node_types;
    device const int* neigh_indices;
    device const int* neigh_types;
    device const int* edge_receivers;
    device const int* type_to_active;
    device const float* unit_xyz;
    device const float* radius;
    MetalR1RadialSpline radial;
    device const float* harmonics_values;
    device const float* harmonics_gradients;
    device const float* neighbor_features;
    device const float* output_adjoint;
    device float* directed_forces;
};

struct EvaluationPoint {
    int interval;
    float x;
    float xx;
    float xxx;
};

inline bool edge_is_active(float cutoff, float radius)
{
    return radius < cutoff;
}

inline EvaluationPoint evaluation_point(
    constant MetalR1RadialSpline& radial, float radius)
{
    int interval = int(floor((radius - radial.x0) / radial.h));
    float x = radius - radial.x0 - radial.h * float(interval);
    if (interval < 0) {
        interval = 0;
        x = 0.0f;
    } else if (interval >= int(radial.intervals)) {
        interval = int(radial.intervals) - 1;
        x = radial.h;
    }
    const float xx = x * x;
    return {interval, x, xx, xx * x};
}

// Coefficients are interleaved [edge_type, interval, function] float4 so one
// load fetches the cubic of one radial function. This speeds the edge pass by
// about 30% and slows the value-only source pass by about 5% on an M1 Max; a
// second, non-interleaved table for the source pass was not worth its memory.
inline ulong radial_index(
    constant MetalR1RadialSpline& radial,
    int edge_type,
    int interval,
    int function)
{
    return (ulong(edge_type) * radial.intervals + ulong(interval))
        * radial.functions + ulong(function);
}

inline Scalar evaluate_radial(
    constant MetalR1RadialSpline& radial,
    int edge_type,
    thread const EvaluationPoint& point,
    int function)
{
    const float4 c =
        radial.coefficients[radial_index(radial, edge_type, point.interval, function)];
    return c.x + c.y * point.x + c.z * point.xx + c.w * point.xxx;
}

inline void evaluate_radial(
    constant MetalR1RadialSpline& radial,
    int edge_type,
    thread const EvaluationPoint& point,
    int function,
    thread Scalar& value,
    thread Scalar& derivative)
{
    const float4 c =
        radial.coefficients[radial_index(radial, edge_type, point.interval, function)];
    value = c.x + c.y * point.x + c.z * point.xx + c.w * point.xxx;
    derivative = c.y + c.z * (2.0f * point.x) + c.w * (3.0f * point.xx);
}

inline int pair_type(int left, int right, int type_count)
{
    return left <= right
        ? left * (2 * type_count - left - 1) / 2 + right
        : right * (2 * type_count - right - 1) / 2 + left;
}

template <typename T>
inline void prefetch_read(T) {}
"""

_SIMD_WIDTH = 32

# Channel block of the blocked edge kernel: one SIMD lane per channel, so a
# receiver's adjoint tile is output_components * 32 floats.
METAL_R1_EDGE_BLOCK_CHANNELS = _SIMD_WIDTH


def _msl_float(value: float) -> str:
    return f"float({value!r})"


def _msl_row_loaders(path_rows, *, channels: int, output_components: int) -> str:
    """Receiver adjoint rows of each coupling path row.

    ``load_row_k`` combines rows of the receiver's adjoint in device memory for
    the forward-adjoint and edge owners; ``block_row_k`` reads the same rows
    from the threadgroup tile of one 32-channel block.
    """

    block = METAL_R1_EDGE_BLOCK_CHANNELS
    rendered = []
    for rows in path_rows:
        for row in rows:
            device_terms = "\n        + ".join(
                f"{_msl_float(term['coefficient'])} * args->output_adjoint["
                f"(static_cast<ulong>(receiver) * {output_components} + {term['lme']})"
                f" * {channels} + channel]"
                for term in row["terms"]
            )
            tile_terms = "\n        + ".join(
                f"{_msl_float(term['coefficient'])} * tile[{term['lme']} * {block} + local]"
                for term in row["terms"]
            )
            for args_type in ("MetalR1SourceArgs", "MetalR1EdgeArgs"):
                rendered.append(
                    f"inline float load_row_{row['row']}(\n"
                    f"    constant {args_type}* args,\n"
                    "    int receiver,\n"
                    "    int channel)\n{\n"
                    f"    return {device_terms};\n}}"
                )
            rendered.append(
                f"inline float block_row_{row['row']}(\n"
                "    threadgroup const float* tile,\n"
                "    int local)\n{\n"
                f"    return {tile_terms};\n}}"
            )
    if not rendered:
        raise ValueError("R1 contract has no coupling rows")
    return "\n\n".join(rendered)


def _msl_edge_owner(
    path_rows,
    *,
    channels: int,
    edge_harmonics: int,
    source_harmonics: int,
    blocked: bool,
) -> str:
    """Coordinate adjoint of one edge, mirroring the host edge owner's terms.

    In the per-edge variant the 32 lanes of a SIMD group split the channels
    and write the edge's force. In the blocked variant lane l owns channel
    block_base + l, reads receiver rows from the block's tile, and writes the
    block's partial force plane. ``simd_sum`` reduces the force over lanes.
    """

    count = 3 * edge_harmonics
    blocks = []
    for path_index, rows in enumerate(path_rows):
        statements = [
            "        {",
            f"        float radial_value_{path_index} = float(0);",
            f"        float radial_derivative_{path_index} = float(0);",
            "        evaluate_radial(",
            f"            args->radial, edge_type, point, {path_index} * {channels} + channel,",
            f"            radial_value_{path_index}, radial_derivative_{path_index});",
        ]
        for row in rows:
            r, lm1, lm2 = row["row"], row["lm1"], row["lm2"]
            adjoint = (
                f"block_row_{r}(tile, int(lane))"
                if blocked
                else f"load_row_{r}(args, receiver, channel)"
            )
            statements.extend(
                [
                    "        {",
                    f"        const float weighted_{r} =",
                    f"            args->neighbor_features[(source_offset + {lm2}) * {channels} + channel]",
                    f"            * {adjoint};",
                    f"        const float radial_force_{r} = radial_derivative_{path_index}",
                    f"            * weighted_{r} * args->harmonics_values[edge_offset + {lm1}];",
                    f"        const float angular_force_{r} = radial_value_{path_index} * weighted_{r};",
                    f"        local_force_x += radial_force_{r} * x_over_r",
                    f"            + angular_force_{r} * gradients[{lm1}];",
                    f"        local_force_y += radial_force_{r} * y_over_r",
                    f"            + angular_force_{r} * gradients[{edge_harmonics + lm1}];",
                    f"        local_force_z += radial_force_{r} * z_over_r",
                    f"            + angular_force_{r} * gradients[{2 * edge_harmonics + lm1}];",
                    "        }",
                ]
            )
        statements.append("        }")
        blocks.append("\n".join(statements))
    if blocked:
        name = "r1_edge_owner_blocked"
        parameters = (
            "    constant MetalR1EdgeArgs* args,\n    int edge,\n    uint lane,\n"
            "    int block_base,\n    threadgroup const float* tile,\n"
            "    device float* forces)"
        )
        loop = "    {\n        const int channel = block_base + int(lane);"
        target = "forces"
    else:
        name = "r1_edge_owner"
        parameters = (
            "    constant MetalR1EdgeArgs* args,\n    int edge,\n    uint lane)"
        )
        loop = (
            f"    for (int channel = int(lane); channel < {channels}; "
            f"channel += {_SIMD_WIDTH}) {{"
        )
        target = "args->directed_forces"
    body = "\n".join(blocks)
    return f"""void {name}(
{parameters}
{{
    if (!edge_is_active(args->cutoff, args->radius[edge]))
        return;
    const int receiver = args->edge_receivers[edge];
    const int source = args->neigh_indices[edge];
    const ulong coordinate_offset =
        static_cast<ulong>(3) * edge;
    const ulong edge_offset =
        static_cast<ulong>(edge) * {edge_harmonics};
    const ulong gradient_offset =
        static_cast<ulong>(3) * edge_offset;
    const ulong source_offset =
        static_cast<ulong>(source) * {source_harmonics};
    const int receiver_type =
        args->type_to_active[args->node_types[receiver]];
    const int source_type =
        args->type_to_active[args->neigh_types[edge]];
    const int edge_type = pair_type(
        receiver_type, source_type,
        static_cast<int>(args->active_type_count));
    const EvaluationPoint point = evaluation_point(
        args->radial, args->radius[edge]);
    const float x_over_r = args->unit_xyz[coordinate_offset];
    const float y_over_r = args->unit_xyz[coordinate_offset + 1];
    const float z_over_r = args->unit_xyz[coordinate_offset + 2];
    float gradients[{count}];
    if (args->has_gradients != 0) {{
        for (int g = 0; g < {count}; ++g)
            gradients[g] = args->harmonics_gradients[gradient_offset + g];
    }} else {{
        direct_harmonic_gradients(
            x_over_r, y_over_r, z_over_r, args->radius[edge], gradients);
    }}
    float local_force_x = float(0);
    float local_force_y = float(0);
    float local_force_z = float(0);
{loop}
{body}
    }}
    local_force_x = simd_sum(local_force_x);
    local_force_y = simd_sum(local_force_y);
    local_force_z = simd_sum(local_force_z);
    if (lane == 0) {{
        {target}[coordinate_offset] -= local_force_x;
        {target}[coordinate_offset + 1] -= local_force_y;
        {target}[coordinate_offset + 2] -= local_force_z;
    }}
}}"""


def _to_msl(source: str) -> str:
    replacements = (
        (
            "const SymmetrixJitHostR1ForwardArgsV1* args",
            "constant MetalR1ForwardArgs* args",
        ),
        (
            "const SymmetrixJitHostR1SourceArgsV1* args",
            "constant MetalR1SourceArgs* args",
        ),
        ("const SymmetrixJitHostR1EdgeArgsV1* args", "constant MetalR1EdgeArgs* args"),
        ("static_cast<double>(", "static_cast<float>("),
        # MSL 3.0 is C++14-based; the condition is a template constant.
        ("if constexpr (", "if ("),
        ("std::int32_t", "int"),
        ("std::uint32_t", "uint"),
        ("std::size_t", "ulong"),
    )
    for old, new in replacements:
        source = source.replace(old, new)
    source = re.sub(r"\bstd::(sqrt|floor|fabs|abs|fma|min|max)\b", r"\1", source)
    leftovers = sorted(set(re.findall(r"\bstd::\w+|\bdouble\b|\bnullptr\b", source)))
    if leftovers:
        raise ValueError(
            f"R1 source has constructs without an MSL mapping: {leftovers}"
        )
    return source


def _direct_gradient_helper(edge_harmonics: int) -> str:
    helper = _host._render_direct_harmonic_gradient_helper("inline", edge_harmonics)
    marker = "    const Scalar inverse_norm"
    body = helper[helper.index(marker) :]
    header = (
        "inline void direct_harmonic_gradients(\n"
        "    Scalar ux, Scalar uy, Scalar uz, float radius,\n"
        "    thread Scalar* gradients)\n{\n"
    )
    return _to_msl(header + body)


def metal_r1_metadata(contract: dict) -> dict:
    normalized = normalize_jit_r1_contract(contract)
    host = _host.jit_r1_host_plugin_metadata(normalized, None, precision="float32")
    return {
        "abi_version": METAL_R1_ABI_VERSION,
        "channels": host["channels"],
        "edge_harmonics": (host["edge_l_max"] + 1) ** 2,
        "source_harmonics": (host["source_l_max"] + 1) ** 2,
        "output_components": host["output_components"],
        "contract_fingerprint": host["contract_fingerprint"],
    }


def render_jit_r1_metal_source(contract: dict) -> str:
    """Render the FP32 Metal R1 forward, source, and edge kernels."""

    normalized = normalize_jit_r1_contract(contract)
    metadata = metal_r1_metadata(normalized)
    path_rows = _host._fixed_weight_path_rows(normalized)
    channels = metadata["channels"]
    edge_harmonics = metadata["edge_harmonics"]
    source_harmonics = metadata["source_harmonics"]
    output_components = metadata["output_components"]

    forward = _host._render_tiled_host_forward_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        output_components=output_components,
        channel_tile=1,
    )
    source = _host._render_tiled_host_source_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        output_components=output_components,
        channel_tile=1,
    )
    loaders_msl = _msl_row_loaders(
        path_rows, channels=channels, output_components=output_components
    )
    edge_msl = _msl_edge_owner(
        path_rows,
        channels=channels,
        edge_harmonics=edge_harmonics,
        source_harmonics=source_harmonics,
        blocked=False,
    )
    tile_floats = output_components * channels

    kernels = f"""
kernel void symmetrix_r1_forward(
    constant MetalR1ForwardArgs& args [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{{
    const uint receiver = flat / {channels}u;
    if (receiver >= args.num_nodes)
        return;
    r1_forward_owner(&args, int(receiver), int(flat % {channels}u));
}}

kernel void symmetrix_r1_source(
    constant MetalR1SourceArgs& args [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{{
    const uint source = flat / {channels}u;
    if (source >= args.num_nodes)
        return;
    r1_source_owner_impl<true>(&args, int(source), int(flat % {channels}u));
}}

kernel void symmetrix_r1_edge(
    constant MetalR1EdgeArgs& args [[buffer(0)]],
    uint flat [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint edge = flat / {_SIMD_WIDTH}u;
    if (edge >= args.num_edges)
        return;
    r1_edge_owner(&args, int(edge), lane);
}}
"""
    block = METAL_R1_EDGE_BLOCK_CHANNELS
    blocked = channels % block == 0
    if blocked:
        block_floats = output_components * block
        kernels += f"""
// One threadgroup per (receiver, 32-channel block) of a receiver-ordered edge
// list, flattened receiver-major. The block's adjoint rows are staged in threadgroup memory, small
// enough for several threadgroups per core, and each block writes a partial
// force plane that the host sums in a fixed order.
kernel void symmetrix_r1_edge_blocked(
    constant MetalR1EdgeArgs& args [[buffer(0)]],
    device const int* receiver_offsets [[buffer(1)]],
    device float* block_forces [[buffer(2)]],
    uint position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint groups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    threadgroup float tile[{block_floats}];
    const uint receiver = position / {channels // block}u;
    const uint block_index = position % {channels // block}u;
    const uint block_base = block_index * {block}u;
    const int begin = receiver_offsets[receiver];
    const int end = receiver_offsets[receiver + 1];
    if (begin == end)
        return;
    device const float* rows =
        args.output_adjoint + ulong(receiver) * {tile_floats}u + block_base;
    for (uint index = thread_index; index < {block_floats}u; index += threads)
        tile[index] = rows[(index / {block}u) * {channels}u + index % {block}u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* forces = block_forces + ulong(block_index) * 3u * args.num_edges;
    for (int edge = begin + int(group); edge < end; edge += int(groups))
        r1_edge_owner_blocked(&args, edge, lane, int(block_base), tile, forces);
}}
"""
    return "\n".join(
        (
            f"// Generated by symmetrix.metal_codegen (Metal R1 ABI "
            f"{METAL_R1_ABI_VERSION}). Do not edit.",
            f"// contract: {metadata['contract_fingerprint']}",
            _PRELUDE,
            _direct_gradient_helper(edge_harmonics),
            loaders_msl,
            _to_msl(forward),
            _to_msl(source),
            edge_msl,
            *(
                (
                    _msl_edge_owner(
                        path_rows,
                        channels=channels,
                        edge_harmonics=edge_harmonics,
                        source_harmonics=source_harmonics,
                        blocked=True,
                    ),
                )
                if blocked
                else ()
            ),
            kernels,
        )
    )


METAL_M0_ABI_VERSION = 1


def _product_terms(contract: dict, output_components: int | None):
    from .jit_operator_codegen import _m0_terms, jit_m0_host_metadata

    host = jit_m0_host_metadata(contract, precision="float32")
    terms = _m0_terms(host["contract"])
    if len(terms) != int(host["term_count"]):
        raise ValueError("M0 term list does not match the contract metadata")
    outputs = int(host["output_components"])
    if output_components is not None:
        if not 0 < output_components <= outputs:
            raise ValueError("requested output components exceed the contract")
        # Terms are grouped by output component, so a prefix keeps the
        # weight index of every retained term.
        kept = [term for term in terms if term[0] < output_components]
        if kept != terms[: len(kept)]:
            raise ValueError("M0 terms are not grouped by output component")
        terms, outputs = kept, output_components
    return host, terms, outputs


def metal_m0_metadata(contract: dict, output_components: int | None = None) -> dict:
    host, terms, outputs = _product_terms(contract, output_components)
    return {
        "abi_version": METAL_M0_ABI_VERSION,
        "channels": int(host["contract"]["channels"]),
        "input_components": int(host["input_components"]),
        "output_components": outputs,
        "term_count": len(terms),
        "structure_fingerprint": host["structure_fingerprint"],
    }


def metal_m1_metadata(m0_contract: dict) -> dict:
    """The standard M1 contraction: the scalar output block of M0's terms."""

    return metal_m0_metadata(m0_contract, output_components=1)


def render_jit_m1_metal_source(m0_contract: dict) -> str:
    return render_jit_m0_metal_source(m0_contract, output_components=1)


def render_jit_m0_metal_source(
    contract: dict, output_components: int | None = None
) -> str:
    """Render FP32 Metal product-basis kernels from the M0 term list.

    One thread owns one (node, channel) pair. Channels are padded to whole
    SIMD groups so a group never spans two nodes; the reverse pass reduces the
    input-scale adjoint over each group into one FP32 partial for the host to
    accumulate in FP64. ``output_components`` keeps a leading block of output
    components, which is how the standard M1 contraction is rendered.
    """

    metadata = metal_m0_metadata(contract, output_components)
    _, terms, _ = _product_terms(contract, output_components)
    forward = []
    reverse = []
    for term, (output, components) in enumerate(terms):
        product = "*".join(f"x[{component}]" for component in components)
        forward.append(f"    out[{output}] += weights[{term}*C]*{product};")
        for component, multiplicity in Counter(components).items():
            remaining = list(components)
            remaining.remove(component)
            factor = (
                "*".join(f"x[{value}]" for value in remaining) if remaining else "1.0f"
            )
            if multiplicity != 1:
                factor = f"({multiplicity}.0f*{factor})"
            reverse.append(
                f"        grad[{component}] += adj[{output}]*weights[{term}*C]*{factor};"
            )
    return f"""// Generated by symmetrix.metal_codegen (Metal M0 ABI {METAL_M0_ABI_VERSION}). Do not edit.
// structure: {metadata["structure_fingerprint"]}
#include <metal_stdlib>
using namespace metal;

constant constexpr uint C = {metadata["channels"]}u;
constant constexpr uint IN = {metadata["input_components"]}u;
constant constexpr uint OUT = {metadata["output_components"]}u;
constant constexpr uint TERMS = {metadata["term_count"]}u;

struct MetalM0Args {{
    uint num_nodes;
    uint capture_input_scale_adjoint;
    uint padded_channels;
    uint reserved;
    device const int* node_types;
    device const float* input;
    device const float* weights;
    device const float* output_adjoint;
    device float* output;
    device float* input_adjoint;
    device float* scale_partials;
}};

kernel void symmetrix_m0_forward(
    constant MetalM0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]])
{{
    const uint node = flat / a.padded_channels;
    const uint channel = flat % a.padded_channels;
    if (node >= a.num_nodes || channel >= C)
        return;
    float x[IN];
    for (uint component = 0; component < IN; ++component)
        x[component] = a.input[(ulong(node) * IN + component) * C + channel];
    device const float* weights =
        a.weights + ulong(a.node_types[node]) * TERMS * C + channel;
    float out[OUT];
    for (uint component = 0; component < OUT; ++component)
        out[component] = 0.0f;
{chr(10).join(forward)}
    for (uint component = 0; component < OUT; ++component)
        a.output[(ulong(node) * OUT + component) * C + channel] = out[component];
}}

kernel void symmetrix_m0_reverse(
    constant MetalM0Args& a [[buffer(0)]],
    uint flat [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint node = flat / a.padded_channels;
    if (node >= a.num_nodes)
        return;
    const uint channel = flat % a.padded_channels;
    float scale = 0.0f;
    if (channel < C) {{
        float x[IN];
        float grad[IN];
        float adj[OUT];
        for (uint component = 0; component < IN; ++component) {{
            x[component] = a.input[(ulong(node) * IN + component) * C + channel];
            grad[component] = 0.0f;
        }}
        for (uint component = 0; component < OUT; ++component)
            adj[component] =
                a.output_adjoint[(ulong(node) * OUT + component) * C + channel];
        device const float* weights =
            a.weights + ulong(a.node_types[node]) * TERMS * C + channel;
{chr(10).join(reverse)}
        for (uint component = 0; component < IN; ++component) {{
            a.input_adjoint[(ulong(node) * IN + component) * C + channel] =
                grad[component];
            scale += x[component] * grad[component];
        }}
    }}
    if (a.capture_input_scale_adjoint != 0) {{
        scale = simd_sum(scale);
        if (lane == 0)
            a.scale_partials[ulong(node) * (a.padded_channels / 32u)
                + channel / 32u] = scale;
    }}
}}
"""
