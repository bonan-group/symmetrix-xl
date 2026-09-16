"""Runtime-generated state-free M0 operator modules."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .execution_contract import (
    normalize_standard_m0_contract,
    normalize_standard_r0_contract,
)
from .jit import JIT_GENERATION_VERSION
from .jit_codegen import _render_direct_harmonic_gradient_helper

M0_CAPABILITIES_V1 = 0x1F
M0_THREADS_V1 = 256
R0_CAPABILITIES_V1 = 0x7F00
R0_THREADS_V1 = 256

_M0_HOST_PLUGIN_ABI_HEADER = (
    Path(__file__).with_name("jit_m0_host_plugin_abi.h").read_text(encoding="ascii")
)


def _string_literal(value: str) -> str:
    return json.dumps(str(value))


def _m0_terms(contract: Mapping[str, Any]):
    return [
        (int(group["output_component"]), tuple(int(value) for value in term))
        for group in contract["monomial_groups"]
        for term in group["terms"]
    ]


def jit_m0_device_metadata(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> dict[str, Any]:
    normalized = normalize_standard_m0_contract(contract)
    if precision not in ("float32", "float64"):
        raise ValueError("M0 device precision must be 'float32' or 'float64'")
    if schedule not in ("chunk32", "table"):
        raise ValueError("M0 device schedule must be 'chunk32' or 'table'")
    if normalized["correlation"] > 4:
        raise ValueError(
            "M0 generated modules currently support correlation no greater than four"
        )
    if not target:
        raise ValueError("M0 device target must not be empty")
    suffix = "f32" if precision == "float32" else "f64"
    structure = normalized["structure_fingerprint"]
    term_count = sum(group["term_count"] for group in normalized["monomial_groups"])
    return {
        "schema": "symmetrix.jit.operator-module/1",
        "operator": "M0",
        "abi_version": 1,
        "artifact_id": (
            f"jit-m0-gen{JIT_GENERATION_VERSION}-{suffix}-"
            f"{structure.removeprefix('sha256:')[:16]}-c{normalized['channels']}-"
            f"{schedule}-{target}"
        ),
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
        "threads_per_block": M0_THREADS_V1,
        "capabilities": M0_CAPABILITIES_V1,
        "contract": normalized,
    }


def _m0_prefix(metadata: Mapping[str, Any]) -> str:
    scalar = "float" if metadata["precision"] == "float32" else "double"
    return f"""// Generated Symmetrix state-free M0 operator module.
namespace std {{ using size_t = decltype(sizeof(0)); }}
using Scalar = {scalar};
constexpr int input_components = {metadata["input_components"]};
constexpr int output_components = {metadata["output_components"]};
constexpr int correlation = {metadata["correlation"]};
constexpr int term_count = {metadata["term_count"]};
constexpr int channel_count = {metadata["contract"]["channels"]};
struct M0Args {{
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
}};
extern "C" __device__ __constant__ unsigned int symmetrix_operator_abi_version_v1 = 1u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_kind_v1 = 1u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_kind_v1 = {metadata["scalar_kind"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_size_v1 = {metadata["scalar_size"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_capabilities_v1 = {metadata["capabilities"]}u;
extern "C" __device__ __constant__ int symmetrix_operator_input_components_v1 = input_components;
extern "C" __device__ __constant__ int symmetrix_operator_output_components_v1 = output_components;
extern "C" __device__ __constant__ int symmetrix_operator_correlation_v1 = correlation;
extern "C" __device__ __constant__ int symmetrix_operator_term_count_v1 = term_count;
extern "C" __device__ __constant__ int symmetrix_operator_threads_v1 = {metadata["threads_per_block"]};
extern "C" __device__ __constant__ char symmetrix_operator_artifact_id_v1[] = {_string_literal(metadata["artifact_id"])};
extern "C" __device__ __constant__ char symmetrix_operator_structure_fingerprint_v1[] = {_string_literal(metadata["structure_fingerprint"])};
extern "C" __device__ __constant__ char symmetrix_operator_target_v1[] = {_string_literal(metadata["target"])};
"""


def _product(components: tuple[int, ...]) -> str:
    return "(" + "*".join(f"x[{component}]" for component in components) + ")"


def _render_m0_chunked(metadata: Mapping[str, Any]) -> str:
    terms = _m0_terms(metadata["contract"])
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
            f"__device__ __forceinline__ void m0_fwd_{chunk_index}("
            "const Scalar* x, const Scalar* weights, int channels, Scalar* out) {\n"
            + "\n".join(forward)
            + "\n}"
        )
        reverse_helpers.append(
            f"__device__ __forceinline__ void m0_rev_{chunk_index}("
            "const Scalar* x, const Scalar* weights, int channels, "
            "const Scalar* adj, Scalar* grad) {\n" + "\n".join(reverse) + "\n}"
        )
        term_offset += len(chunk)
    forward_calls = "\n".join(
        f"        m0_fwd_{index}(x, weights, channel_count, out);"
        for index in range(len(chunks))
    )
    reverse_calls = "\n".join(
        f"        m0_rev_{index}(x, weights, channel_count, adj, grad);"
        for index in range(len(chunks))
    )
    return "\n".join((*forward_helpers, *reverse_helpers)) + _m0_kernels(
        forward_calls, reverse_calls
    )


def _m0_kernels(forward_body: str, reverse_body: str) -> str:
    return f"""
extern "C" __global__ __launch_bounds__(256)
void symmetrix_m0_forward_v1(M0Args args) {{
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
            x[component] = input[(node*input_components+component)*channel_count+channel];
        const Scalar* weights = packed_weights
            +(static_cast<long long>(type)*term_count)*channel_count+channel;
{forward_body}
        for (int component=0; component<output_components; ++component)
            output_values[(node*output_components+component)*channel_count+channel]
                = out[component];
    }}
}}
extern "C" __global__ __launch_bounds__(256)
void symmetrix_m0_reverse_v1(M0Args args) {{
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
            x[component] = input[(node*input_components+component)*channel_count+channel];
        for (int component=0; component<output_components; ++component)
            adj[component] = output_adjoint[
                (node*output_components+component)*channel_count+channel];
        const Scalar* weights = packed_weights
            +(static_cast<long long>(type)*term_count)*channel_count+channel;
{reverse_body}
        double scale = 0.0;
        for (int component=0; component<input_components; ++component) {{
            input_adjoint[(node*input_components+component)*channel_count+channel]
                = grad[component];
            scale += static_cast<double>(x[component])
                *static_cast<double>(grad[component]);
        }}
        if (args.capture_input_scale_adjoint)
            atomicAdd(args.input_scale_adjoint+node, scale);
    }}
}}
"""


def _render_m0_table(metadata: Mapping[str, Any]) -> str:
    terms = _m0_terms(metadata["contract"])
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
__device__ __constant__ int m0_term_outputs[term_count] = {{{outputs}}};
__device__ __constant__ int m0_term_degrees[term_count] = {{{degrees}}};
__device__ __constant__ int m0_term_components[term_count*correlation] = {{{components}}};
__device__ __constant__ int m0_output_offsets[output_components+1] = {{{offset_values}}};
"""
    forward = """
        for (int output=0; output<output_components; ++output) {
            for (int term=m0_output_offsets[output];
                 term<m0_output_offsets[output+1]; ++term) {
                Scalar product = Scalar(1);
                for (int degree=0; degree<m0_term_degrees[term]; ++degree)
                    product *= x[m0_term_components[term*correlation+degree]];
                out[output] += weights[term*args.channels]*product;
            }
        }"""
    reverse = """
        for (int term=0; term<term_count; ++term) {
            const Scalar scale = adj[m0_term_outputs[term]]
                *weights[term*args.channels];
            for (int factor=0; factor<m0_term_degrees[term]; ++factor) {
                Scalar product = scale;
                for (int other=0; other<m0_term_degrees[term]; ++other)
                    if (other != factor)
                        product *= x[m0_term_components[term*correlation+other]];
                grad[m0_term_components[term*correlation+factor]] += product;
            }
        }"""
    return tables + _m0_kernels(forward, reverse)


def render_jit_m0_device_module(
    contract: Mapping[str, Any],
    *,
    precision: str,
    target: str,
    schedule: str = "chunk32",
) -> tuple[str, dict[str, Any]]:
    """Render a self-contained CUDA/HIP RTC M0 module and its identity."""

    metadata = jit_m0_device_metadata(
        contract, precision=precision, target=target, schedule=schedule
    )
    body = (
        _render_m0_chunked(metadata)
        if schedule == "chunk32"
        else _render_m0_table(metadata)
    )
    return _m0_prefix(metadata) + body, metadata


def jit_m0_host_metadata(
    contract: Mapping[str, Any], *, precision: str, schedule: str = "chunk32"
) -> dict[str, Any]:
    """Return the exact identity for a generated state-free host M0 plugin."""

    metadata = jit_m0_device_metadata(
        contract, precision=precision, target="host", schedule=schedule
    )
    structure = metadata["structure_fingerprint"]
    suffix = "f32" if precision == "float32" else "f64"
    return {
        **metadata,
        "schema": "symmetrix.jit.m0-host-plugin/1",
        "abi": "symmetrix.jit.m0-host-plugin/1",
        "artifact_id": (
            f"jit-m0-host-gen{JIT_GENERATION_VERSION}-{suffix}-"
            f"{structure.removeprefix('sha256:')[:16]}-"
            f"c{metadata['contract']['channels']}-{schedule}"
        ),
        "capabilities": 0x1F,
    }


def render_jit_m0_host_plugin(
    contract: Mapping[str, Any], *, precision: str, schedule: str = "chunk32"
) -> tuple[str, dict[str, Any]]:
    """Render a standalone host M0 plugin with a vector-width channel owner."""

    if schedule != "chunk32":
        raise ValueError("M0 host plugins currently require the chunk32 schedule")
    metadata = jit_m0_host_metadata(contract, precision=precision, schedule=schedule)
    terms = _m0_terms(metadata["contract"])
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
            f"SYMMETRIX_M0_ALWAYS_INLINE void m0_fwd_{chunk_index}("
            "const Scalar (&x)[input_components][host_channel_tile], "
            "const Scalar* weights, int channels, int active_channels, "
            "Scalar (&out)[output_components][host_channel_tile]) {\n"
            + "\n".join(forward)
            + "\n}"
        )
        reverse_helpers.append(
            f"SYMMETRIX_M0_ALWAYS_INLINE void m0_rev_{chunk_index}("
            "const Scalar (&x)[input_components][host_channel_tile], "
            "const Scalar* weights, int channels, int active_channels, "
            "const Scalar (&adj)[output_components][host_channel_tile], "
            "Scalar (&grad)[input_components][host_channel_tile]) {\n"
            + "\n".join(reverse)
            + "\n}"
        )
        term_offset += len(chunk)
    forward_calls = "\n".join(
        f"    m0_fwd_{index}(x, weights, channels, active_channels, out);"
        for index in range(len(chunks))
    )
    reverse_calls = "\n".join(
        f"    m0_rev_{index}(x, weights, channels, active_channels, adj, grad);"
        for index in range(len(chunks))
    )
    scalar = "float" if precision == "float32" else "double"
    helpers = "\n".join((*forward_helpers, *reverse_helpers))
    source = f"""// Generated Symmetrix state-free host M0 plugin.
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <type_traits>

{_M0_HOST_PLUGIN_ABI_HEADER}

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
    : (SYMMETRIX_JIT_M0_HOST_FORWARD_OWNER_V1
        |SYMMETRIX_JIT_M0_HOST_REVERSE_OWNER_V1
        |SYMMETRIX_JIT_M0_HOST_ALIAS_SAFE_V1
        |SYMMETRIX_JIT_M0_HOST_SCALE_ADJOINT_V1);
constexpr std::uint32_t descriptor_channel_tile = host_channel_tile > 1
    ? static_cast<std::uint32_t>(host_channel_tile) : 0u;
#if defined(__clang__) || defined(__GNUC__)
#define SYMMETRIX_M0_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SYMMETRIX_M0_ALWAYS_INLINE inline
#endif

{helpers}

extern "C" void m0_forward_owner(
    const SymmetrixJitM0HostArgsV1* args, std::int64_t owner) {{
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

extern "C" void m0_reverse_owner(
    const SymmetrixJitM0HostArgsV1* args, std::int64_t owner) {{
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
            input_adjoint[(node*input_components+component)*channels+channel+lane]
                = grad[component][lane];
            scale += static_cast<double>(x[component][lane])
                *static_cast<double>(grad[component][lane]);
        }}
    if (args->capture_input_scale_adjoint)
        std::atomic_ref<double>(args->input_scale_adjoint[node])
            .fetch_add(scale, std::memory_order_relaxed);
}}

extern "C" const SymmetrixJitM0HostPluginV1*
symmetrix_jit_m0_host_plugin_query_v1() {{
    static const SymmetrixJitM0HostPluginV1 descriptor = {{
        SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_VERSION_V1,
        sizeof(SymmetrixJitM0HostPluginV1), sizeof(void*),
        SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1,
        {metadata["scalar_kind"]}u,
        {metadata["scalar_size"]}u, host_capabilities,
        channels, input_components, output_components,
        {metadata["correlation"]}, term_count, descriptor_channel_tile,
        {_string_literal(metadata["abi"])},
        {_string_literal(metadata["artifact_id"])},
        {_string_literal(metadata["structure_fingerprint"])},
        &m0_forward_owner, &m0_reverse_owner
    }};
    return &descriptor;
}}
"""
    return source, metadata


def _validate_scalar_source_r0(contract: Mapping[str, Any]) -> dict[str, Any]:
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


def jit_r0_device_metadata(
    contract: Mapping[str, Any], *, precision: str, target: str
) -> dict[str, Any]:
    normalized = _validate_scalar_source_r0(contract)
    if precision not in ("float32", "float64"):
        raise ValueError("R0 device precision must be 'float32' or 'float64'")
    if not target:
        raise ValueError("R0 device target must not be empty")
    suffix = "f32" if precision == "float32" else "f64"
    structure = normalized["structure_fingerprint"]
    harmonic_count = (normalized["edge_harmonics"]["l_max"] + 1) ** 2
    return {
        "schema": "symmetrix.jit.operator-module/1",
        "operator": "R0",
        "abi_version": 1,
        "artifact_id": (
            f"jit-r0-gen{JIT_GENERATION_VERSION}-{suffix}-"
            f"{structure.removeprefix('sha256:')[:16]}-edge-batch1-{target}"
        ),
        "structure_fingerprint": structure,
        "precision": precision,
        "scalar_kind": 1 if precision == "float32" else 2,
        "scalar_size": 4 if precision == "float32" else 8,
        "target": str(target),
        "input_components": harmonic_count,
        "output_components": harmonic_count,
        "correlation": 1,
        "term_count": len(normalized["sparse_coupling"]["terms"]),
        "threads_per_block": R0_THREADS_V1,
        "capabilities": R0_CAPABILITIES_V1,
        "contract": normalized,
    }


def _r0_prefix(metadata: Mapping[str, Any]) -> str:
    scalar = "float" if metadata["precision"] == "float32" else "double"
    return f"""// Generated Symmetrix state-free R0 operator module.
namespace std {{ using size_t = decltype(sizeof(0)); }}
using Scalar = {scalar};
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
}};
extern "C" __device__ __constant__ unsigned int symmetrix_operator_abi_version_v1 = 1u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_kind_v1 = 2u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_kind_v1 = {metadata["scalar_kind"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_scalar_size_v1 = {metadata["scalar_size"]}u;
extern "C" __device__ __constant__ unsigned int symmetrix_operator_capabilities_v1 = {metadata["capabilities"]}u;
extern "C" __device__ __constant__ int symmetrix_operator_input_components_v1 = {metadata["input_components"]};
extern "C" __device__ __constant__ int symmetrix_operator_output_components_v1 = {metadata["output_components"]};
extern "C" __device__ __constant__ int symmetrix_operator_correlation_v1 = 1;
extern "C" __device__ __constant__ int symmetrix_operator_term_count_v1 = {metadata["term_count"]};
extern "C" __device__ __constant__ int symmetrix_operator_threads_v1 = {metadata["threads_per_block"]};
extern "C" __device__ __constant__ char symmetrix_operator_artifact_id_v1[] = {_string_literal(metadata["artifact_id"])};
extern "C" __device__ __constant__ char symmetrix_operator_structure_fingerprint_v1[] = {_string_literal(metadata["structure_fingerprint"])};
extern "C" __device__ __constant__ char symmetrix_operator_target_v1[] = {_string_literal(metadata["target"])};

__device__ __forceinline__ Scalar radial_spline(
    const Spline spline, int edge_type, int function, double radius,
    Scalar* derivative) {{
    const Scalar* coefficients = static_cast<const Scalar*>(spline.coefficients_raw);
    int interval = static_cast<int>(floor((radius-spline.x0)/spline.h));
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


def _r0_kernels(harmonic_count: int) -> str:
    source = r"""
extern "C" __global__ __launch_bounds__(256)
void symmetrix_r0_density_prepare_v1(R0Args args) {
    const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
    for (long long node = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
         node < args.num_nodes; node += stride) {
        const int receiver_type = args.type_to_active[args.node_types[node]];
        const int begin = args.first_neigh[node];
        double density = 0.0;
        for (int local=0; local<args.num_neigh[node]; ++local) {
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
            density += density_spline(
                args.density, edge_type, args.radius[edge], &unused);
        }
        args.density_state[node] = 1.0/(1.0+density);
    }
}

extern "C" __global__ __launch_bounds__(256)
void symmetrix_r0_forward_v1(R0Args args) {
    const Scalar* harmonics = static_cast<const Scalar*>(args.harmonics_raw);
    Scalar* output = static_cast<Scalar*>(args.output_raw);
    const int harmonic_count = (args.l_max+1)*(args.l_max+1);
    const long long owners = args.num_nodes*harmonic_count*args.channels;
    const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
    for (long long owner = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
         owner < owners; owner += stride) {
        const int channel = static_cast<int>(owner%args.channels);
        const long long node_lm = owner/args.channels;
        const int lm = static_cast<int>(node_lm%harmonic_count);
        const long long node = node_lm/harmonic_count;
        int l = 0;
        while ((l+1)*(l+1) <= lm) ++l;
        const int receiver_type = args.type_to_active[args.node_types[node]];
        const int begin = args.first_neigh[node];
        Scalar sum = Scalar(0);
        for (int local=0; local<args.num_neigh[node]; ++local) {
            const int edge = begin+local;
            if (!edge_is_active(args.cutoff, args.radius[edge])) continue;
            const int neighbor_type = args.type_to_active[args.neigh_types[edge]];
            const int edge_type = receiver_type*args.active_type_count+neighbor_type;
            Scalar unused;
            const Scalar radial = radial_spline(
                args.radial, edge_type, l*args.channels+channel,
                args.radius[edge], &unused);
            sum += radial*harmonics[static_cast<long long>(edge)*harmonic_count+lm];
        }
        if (args.apply_density_scale)
            sum *= static_cast<Scalar>(args.density_state[node]);
        output[owner] = sum;
    }
}

extern "C" __global__ __launch_bounds__(256)
void symmetrix_r0_reverse_prepare_v1(R0Args args) {
    Scalar* output = static_cast<Scalar*>(args.output_raw);
    Scalar* output_adjoint = static_cast<Scalar*>(args.output_adjoint_raw);
    const int harmonic_count = (args.l_max+1)*(args.l_max+1);
    const long long values = static_cast<long long>(harmonic_count)*args.channels;
    const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
    for (long long node = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
         node < args.num_nodes; node += stride) {
        double dot = args.use_precomputed_scale_adjoint
            ? args.precomputed_scale_adjoint[node] : 0.0;
        const long long base = node*values;
        if (!args.use_precomputed_scale_adjoint)
            for (long long value=0; value<values; ++value)
                dot += static_cast<double>(output[base+value])
                    *static_cast<double>(output_adjoint[base+value]);
        const double inverse = args.density_state[node];
        for (long long value=0; value<values; ++value)
            output_adjoint[base+value] *= static_cast<Scalar>(inverse);
        args.density_state[node] = dot*inverse;
    }
}

extern "C" __global__ __launch_bounds__(256)
void symmetrix_r0_coordinate_reverse_v1(R0Args args) {
    const Scalar* harmonics = static_cast<const Scalar*>(args.harmonics_raw);
    const Scalar* gradients =
        static_cast<const Scalar*>(args.harmonic_gradients_raw);
    const Scalar* output_adjoint =
        static_cast<const Scalar*>(args.output_adjoint_raw);
    const int harmonic_count = (args.l_max+1)*(args.l_max+1);
    const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
    for (long long edge = static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;
         edge < args.num_edges; edge += stride) {
        if (!edge_is_active(args.cutoff, args.radius[edge])) continue;
        const long long node =
            static_cast<long long>(args.edge_receivers[edge])-args.receiver_base;
        const int receiver_type = args.type_to_active[args.node_types[node]];
        const int neighbor_type = args.type_to_active[args.neigh_types[edge]];
        const int edge_type = receiver_type*args.active_type_count+neighbor_type;
        const long long coordinate_offset = 3*edge;
        const Scalar* edge_gradients = gradients;
        Scalar direct_gradients[3*SYMMETRIX_R0_HARMONIC_COUNT];
        if (edge_gradients == nullptr) {
            direct_harmonic_gradients(
                args.coordinates,
                args.coordinates_are_unit ? sizeof(float) : sizeof(double),
                args.coordinates_are_unit, coordinate_offset,
                args.radius[edge], direct_gradients);
            edge_gradients = direct_gradients;
        } else {
            edge_gradients += coordinate_offset*harmonic_count;
        }
        double force[3] = {0.0, 0.0, 0.0};
        for (int l=0; l<=args.l_max; ++l) {
            const int lm_begin = l*l;
            const int lm_end = (l+1)*(l+1);
            for (int channel=0; channel<args.channels; ++channel) {
                Scalar radial_derivative;
                const Scalar radial = radial_spline(
                    args.radial, edge_type, l*args.channels+channel,
                    args.radius[edge], &radial_derivative);
                for (int lm=lm_begin; lm<lm_end; ++lm) {
                    const Scalar adjoint = output_adjoint[
                        (node*harmonic_count+lm)*args.channels+channel];
                    const double radial_scale = static_cast<double>(
                        adjoint*radial_derivative
                        *harmonics[edge*harmonic_count+lm]);
                    const double angular_scale =
                        static_cast<double>(adjoint*radial);
                    for (int axis=0; axis<3; ++axis)
                        force[axis] += radial_scale*direction(args,edge,axis)
                            +angular_scale*static_cast<double>(
                                edge_gradients[axis*harmonic_count+lm]);
                }
            }
        }
        for (int axis=0; axis<3; ++axis)
            args.directed_forces[3*edge+axis] -= force[axis];
        if (args.apply_density_scale) {
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
            const double contribution =
                args.density_state[node]*density_derivative;
            for (int axis=0; axis<3; ++axis)
                args.directed_forces[3*edge+axis] +=
                    contribution*direction(args,edge,axis);
        }
    }
}
"""
    return source.replace("SYMMETRIX_R0_HARMONIC_COUNT", str(harmonic_count))


def render_jit_r0_device_module(
    contract: Mapping[str, Any], *, precision: str, target: str
) -> tuple[str, dict[str, Any]]:
    """Render a self-contained CUDA/HIP RTC scalar-source R0 module."""

    metadata = jit_r0_device_metadata(contract, precision=precision, target=target)
    harmonic_count = metadata["input_components"]
    harmonic_helpers = _render_direct_harmonic_gradient_helper(
        "__device__ __forceinline__", harmonic_count, index_type="int"
    )
    return (
        _r0_prefix(metadata) + harmonic_helpers + "\n\n" + _r0_kernels(harmonic_count),
        metadata,
    )


__all__ = [
    "M0_CAPABILITIES_V1",
    "R0_CAPABILITIES_V1",
    "jit_m0_device_metadata",
    "jit_r0_device_metadata",
    "render_jit_m0_device_module",
    "render_jit_r0_device_module",
]
