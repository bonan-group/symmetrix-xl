#include "oam_cuda_plugin.hpp"
#include "jit_device_plugin_common.hpp"

#if __has_include(<Kokkos_Macros.hpp>)
#include <Kokkos_Macros.hpp>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda.h>
#endif

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
static_assert(std::is_standard_layout_v<SymmetrixOAMCudaLayerV2>);
static_assert(std::is_standard_layout_v<SymmetrixOAMCudaLayerArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixOAMCudaLayerLaunchesV2>);
static_assert(std::is_standard_layout_v<SymmetrixOAMCudaPluginV3>);
static_assert(alignof(SymmetrixOAMCudaLayerV2) >= alignof(void*));
static_assert(alignof(SymmetrixOAMCudaLayerArgsV2) >= alignof(void*));
#if defined(KOKKOS_ENABLE_CUDA)
struct symmetrix::execution::OAMCudaModuleState {
    CUmodule module = nullptr;
    CUcontext context = nullptr;
    std::vector<CUfunction> forward;
    std::vector<CUfunction> reverse;
    std::vector<std::vector<CUfunction>> reverse_groups;
    std::vector<SymmetrixOAMCudaLayerV2> layers;
    std::vector<SymmetrixOAMCudaLayerLaunchesV2> launches;
    std::string abi_tag;
    std::string artifact_id;
    std::string contract_fingerprint;
    std::string semantic_fingerprint;
    std::string structure_fingerprint;
    std::string target_architecture;
    std::string execution_variants;
    std::string full_forward_schedule;
    std::string full_reverse_schedule;
    std::string full_vector_policy;
    std::string direct_geometry_policy;
    std::string full_tiling_policy;
    std::string full_shared_memory_policy;
    std::string full_register_budget_policy;
    std::string target_policy, precision_policy, schedule_qualification;
    SymmetrixOAMCudaPluginV3 descriptor{};
};
#else
struct symmetrix::execution::OAMCudaModuleState {
    SymmetrixOAMCudaPluginV3 descriptor{};
};
#endif

#if defined(__unix__) || defined(__APPLE__)
namespace symmetrix::execution {

OAMCudaPlugin::OAMCudaPlugin() = default;

namespace {

constexpr std::uint32_t known_capabilities =
    SYMMETRIX_OAM_CUDA_FORWARD_LAUNCH_V1
    | SYMMETRIX_OAM_CUDA_REVERSE_LAUNCH_V1
    | SYMMETRIX_OAM_CUDA_SPLINE_RADIAL_V1
    | SYMMETRIX_OAM_CUDA_WEIGHTED_TENSOR_PRODUCT_V1
    | SYMMETRIX_OAM_CUDA_RECEIVER_ACCUMULATION_V1
    | SYMMETRIX_OAM_CUDA_SOURCE_ADJOINT_V1
    | SYMMETRIX_OAM_CUDA_GEOMETRY_ADJOINT_V1
    | SYMMETRIX_OAM_CUDA_RUNTIME_SPLINE_COEFFICIENTS_V1
    | SYMMETRIX_OAM_CUDA_DIRECT_GEOMETRY_V1
    | SYMMETRIX_OAM_CUDA_STAGED_FORWARD_INPUTS_V1
    | SYMMETRIX_OAM_CUDA_GROUPED_REVERSE_PARTIALS_V1;

constexpr std::uint32_t required_launch_capabilities =
    SYMMETRIX_OAM_CUDA_FORWARD_LAUNCH_V1
    | SYMMETRIX_OAM_CUDA_REVERSE_LAUNCH_V1;

detail::PluginLoaderContext context(const std::string& path)
{
    return detail::PluginLoaderContext("OAM CUDA", path);
}

void validate_expectation(const OAMCudaPluginExpectation& expectation)
{
    if (expectation.artifact_id.empty())
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires an artifact id.");
    if (expectation.contract_fingerprint.empty())
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires an architecture fingerprint.");
    if (expectation.semantic_fingerprint.empty())
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires a semantic fingerprint.");
    if (expectation.structure_fingerprint.empty())
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires a structure fingerprint.");
    if (expectation.target_architecture.empty())
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires a target architecture.");
    if (expectation.layers.empty()
        || expectation.layers.size() > SYMMETRIX_OAM_CUDA_PLUGIN_MAX_LAYERS)
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires a bounded layer schedule.");
    if (expectation.target_compute_capability <= 0
        || expectation.max_threads_per_block <= 0
        || expectation.persistent_blocks_per_compute_unit <= 0
        || expectation.persistent_blocks_per_compute_unit > 32)
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires exact target extents.");
    if ((expectation.required_capabilities&~known_capabilities) != 0
        || (expectation.required_capabilities&required_launch_capabilities)
            != required_launch_capabilities)
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation requires both launch capabilities.");
    if ((expectation.scalar_kind
                == SYMMETRIX_OAM_CUDA_SCALAR_FLOAT32_V1
            && expectation.scalar_size != sizeof(float))
        || (expectation.scalar_kind
                == SYMMETRIX_OAM_CUDA_SCALAR_FLOAT64_V1
            && expectation.scalar_size != sizeof(double))
        || (expectation.scalar_kind
                != SYMMETRIX_OAM_CUDA_SCALAR_FLOAT32_V1
            && expectation.scalar_kind
                != SYMMETRIX_OAM_CUDA_SCALAR_FLOAT64_V1))
        throw std::invalid_argument(
            "Execution OAM CUDA plugin expectation has an invalid scalar kind or width.");
}

void validate_layer_schedule(
    const detail::PluginLoaderContext& loader,
    const SymmetrixOAMCudaLayerV2& layer,
    const OAMCudaLayerExpectation& expectation,
    const bool declares_forward,
    const bool declares_reverse,
    const SymmetrixOAMCudaLayerLaunchesV2& launches,
    const std::size_t index,
    const std::int32_t max_threads_per_block,
    const bool device_module)
{
    const auto layer_index = static_cast<std::int32_t>(index);
    if (layer.struct_size < sizeof(SymmetrixOAMCudaLayerV2))
        throw loader.error("layer schedule entry is smaller than ABI version 2; regenerate the artifact");
    if (layer.reserved != 0)
        throw loader.error("reserved layer field is nonzero");
    if (layer.reverse_group_count < 0 || layer.reverse_group_flags != 0)
        throw loader.error("grouped reverse layer schedule is inconsistent");
    if (layer.layer_index != layer_index)
        throw loader.error("layer schedule order does not match its indices");
    if (layer.spline_interval_count <= 0)
        throw loader.error("layer spline interval count is not positive");
    if (layer.spline_function_count <= 0)
        throw loader.error("layer spline function count is not positive");
    if (layer.edge_harmonic_count <= 0)
        throw loader.error("layer edge harmonic count is not positive");
    if (layer.forward_threads_per_block <= 0
        || layer.reverse_threads_per_block <= 0
        || layer.forward_threads_per_block > max_threads_per_block
        || layer.reverse_threads_per_block > max_threads_per_block)
        throw loader.error("layer thread-block size is invalid for this device");
    loader.require_extent(
        expectation.source_dimension,
        layer.source_dimension, "source dimension");
    loader.require_extent(
        expectation.edge_dimension, layer.edge_dimension, "edge dimension");
    loader.require_extent(
        expectation.weight_dimension,
        layer.weight_dimension, "weight dimension");
    loader.require_extent(
        expectation.output_dimension,
        layer.output_dimension, "output dimension");
    loader.require_extent(
        expectation.target_dimension,
        layer.target_dimension, "target dimension");
    loader.require_extent(
        expectation.instruction_count,
        layer.instruction_count, "tensor-product instruction count");
    loader.require_extent(
        expectation.spline_interval_count,
        layer.spline_interval_count, "spline interval count");
    loader.require_extent(
        expectation.spline_function_count,
        layer.spline_function_count, "spline function count");
    loader.require_extent(
        expectation.edge_harmonic_count,
        layer.edge_harmonic_count, "edge harmonic count");
    if (layer.projected_dimension != 0 || layer.projection_weight_count != 0
        || layer.projected_partial_workspace_bytes != 0
        || layer.projected_channel_tile != 0 || layer.reserved_2 != 0)
        throw loader.error("legacy projected layer fields are nonzero");

    if (launches.struct_size < sizeof(SymmetrixOAMCudaLayerLaunchesV2))
        throw loader.error("layer launch entry is smaller than ABI version 2; regenerate the artifact");
    if (launches.reserved != 0 || launches.reserved_2 != 0)
        throw loader.error("reserved launch field is nonzero");
    if (launches.layer_index != static_cast<std::uint32_t>(layer_index))
        throw loader.error("layer launch order does not match its indices");
    if (!device_module
        && ((launches.full_forward_launch != nullptr) != declares_forward
            || (launches.full_reverse_launch != nullptr) != declares_reverse))
        throw loader.error(
            "launcher capability and function pointers disagree");
    if (launches.projected_forward_launch != nullptr
        || launches.projected_reverse_launch != nullptr)
        throw loader.error("legacy projected launch pointers are non-null");
}

void validate_descriptor(
    const std::string& path,
    const SymmetrixOAMCudaPluginV3& descriptor,
    const OAMCudaPluginExpectation& expectation,
    const bool device_module = false)
{
    const auto loader = context(path);
    if (descriptor.abi_version != SYMMETRIX_OAM_CUDA_PLUGIN_ABI_VERSION)
        throw loader.error(
            "unsupported ABI version; regenerate the CUDA artifact for ABI v3");
    if (descriptor.struct_size < sizeof(SymmetrixOAMCudaPluginV3))
        throw loader.error("descriptor is smaller than ABI version 3; regenerate the artifact");
    if (descriptor.pointer_size != sizeof(void*))
        throw loader.error("pointer width does not match this process");
    if (descriptor.byte_order != SYMMETRIX_OAM_CUDA_PLUGIN_BYTE_ORDER)
        throw loader.error("byte order does not match this process");
    if (descriptor.reserved != 0 || descriptor.reserved_2 != 0)
        throw loader.error("reserved descriptor field is nonzero");
    if (descriptor.scalar_kind != expectation.scalar_kind
        || descriptor.scalar_size != expectation.scalar_size)
        throw loader.error("scalar kind or width does not match");
    if ((descriptor.capabilities&~known_capabilities) != 0
        || (descriptor.capabilities&expectation.required_capabilities)
            != expectation.required_capabilities)
        throw loader.error("required launcher capabilities are missing");
    if (descriptor.layer_count == 0
        || descriptor.layer_count > SYMMETRIX_OAM_CUDA_PLUGIN_MAX_LAYERS
        || descriptor.layer_count != expectation.layers.size())
        throw loader.error("layer schedule size does not match");
    if (descriptor.layers == nullptr || descriptor.launches == nullptr)
        throw loader.error("layer schedule storage is missing");
    loader.require_extent(
        expectation.target_compute_capability,
        descriptor.target_compute_capability, "target compute capability");
    loader.require_extent(
        expectation.max_threads_per_block,
        descriptor.max_threads_per_block, "device thread-block limit");

    if (loader.require_text(descriptor.abi_tag, "ABI tag")
        != SYMMETRIX_OAM_CUDA_PLUGIN_ABI_TAG)
        throw loader.error("ABI tag does not match");
    loader.require_match(
        expectation.target_architecture,
        loader.require_text(
            descriptor.target_architecture, "target architecture").c_str(),
        "target architecture");
    loader.require_match(
        expectation.artifact_id,
        loader.require_text(descriptor.artifact_id, "artifact id").c_str(),
        "artifact id");
    loader.require_match(
        expectation.contract_fingerprint,
        loader.require_text(
            descriptor.contract_fingerprint, "architecture fingerprint").c_str(),
        "architecture fingerprint");
    loader.require_match(
        expectation.semantic_fingerprint,
        loader.require_text(
            descriptor.semantic_fingerprint, "semantic fingerprint").c_str(),
        "semantic fingerprint");
    loader.require_match(
        expectation.structure_fingerprint,
        loader.require_text(
            descriptor.structure_fingerprint, "structure fingerprint").c_str(),
        "structure fingerprint");
    loader.require_text(descriptor.full_forward_schedule,"full forward schedule");
    loader.require_text(descriptor.full_reverse_schedule,"full reverse schedule");
    if (loader.require_text(descriptor.execution_variants, "execution variants")
            != "full_message")
        throw loader.error(
            "execution variants must contain only full_message; regenerate the artifact");
    if (descriptor.projection_layout!=nullptr
        ||descriptor.projected_forward_schedule!=nullptr
        ||descriptor.projected_reverse_schedule!=nullptr
        ||descriptor.vector_policy!=nullptr
        ||descriptor.shared_memory_policy!=nullptr
        ||descriptor.register_budget_policy!=nullptr
        ||descriptor.projected_grid_policy!=nullptr
        ||descriptor.projected_reduction_workspace!=nullptr
        ||descriptor.projected_staged_input_policy!=nullptr
        ||descriptor.projected_reverse_grouping!=nullptr
        ||descriptor.projected_threads_per_block!=0)
        throw loader.error(
            "legacy projected descriptor fields are populated; regenerate the artifact");
    loader.require_text(descriptor.full_vector_policy, "full vector policy");
    if (loader.require_text(descriptor.direct_geometry_policy,
            "direct geometry policy") != "generated-direct-harmonic-gradient-v1")
        throw loader.error(
            "direct geometry policy is ambiguous; regenerate the artifact");
    if (loader.require_text(descriptor.full_tiling_policy,
            "full tiling policy") != "descriptor-layer-threads-v1"
        || loader.require_text(descriptor.full_shared_memory_policy,
            "full shared-memory policy") != "descriptor-static-bytes-v1"
        || loader.require_text(descriptor.full_register_budget_policy,
            "full register-budget policy") != "compiler-managed-v1")
        throw loader.error(
            "full-message resource policy is ambiguous; regenerate the artifact");
    if (descriptor.reserved_3 != 0)
        throw loader.error("resource descriptor reserved field is nonzero");
#define SYMMETRIX_REQUIRE_CUDA_RESOURCE(field, label) \
    if (loader.require_text(descriptor.field, label) != expectation.field) \
        throw loader.error(std::string(label) + " does not match the manifest")
    SYMMETRIX_REQUIRE_CUDA_RESOURCE(target_policy, "target policy");
    SYMMETRIX_REQUIRE_CUDA_RESOURCE(precision_policy, "precision policy");
    SYMMETRIX_REQUIRE_CUDA_RESOURCE(schedule_qualification, "schedule qualification");
#undef SYMMETRIX_REQUIRE_CUDA_RESOURCE
    const auto expected_precision_policy = descriptor.scalar_kind
            == SYMMETRIX_OAM_CUDA_SCALAR_FLOAT32_V1
        ? std::string_view("exact-float32-v1")
        : std::string_view("exact-float64-v1");
    if (expectation.target_policy != "exact-compute-capability-v1"
        || expectation.precision_policy != expected_precision_policy)
        throw loader.error("target or precision resource policy is unsupported");

    const bool declares_forward =
        (descriptor.capabilities&SYMMETRIX_OAM_CUDA_FORWARD_LAUNCH_V1) != 0;
    const bool declares_reverse =
        (descriptor.capabilities&SYMMETRIX_OAM_CUDA_REVERSE_LAUNCH_V1) != 0;
    for (std::size_t index=0; index<descriptor.layer_count; ++index) {
        validate_layer_schedule(
            loader, descriptor.layers[index], expectation.layers[index],
            declares_forward, declares_reverse, descriptor.launches[index],
            index, descriptor.max_threads_per_block, device_module);
    }
}
void validate_forward_packet(
    const std::string& path,
    const SymmetrixOAMCudaLayerArgsV2& args,
    const std::size_t layer_index)
{
    const bool direct =
        (args.flags&SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1) != 0;
    if (args.struct_size < sizeof(SymmetrixOAMCudaLayerArgsV2)
        || args.layer_index != layer_index
        || args.reserved != 0
        || (args.flags&~SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1) != 0
        || args.num_nodes < 0 || args.num_edges < 0
        || (args.num_nodes > 0
            && (args.receiver_offsets == nullptr
                || args.source_node_values == nullptr
                || args.node_messages == nullptr))
        || (args.num_edges > 0
            && (args.source_indices == nullptr
                || args.distances == nullptr
                || args.spline_coefficients == nullptr
                || (direct ? args.edge_vectors == nullptr
                    : args.edge_harmonics == nullptr))))
        throw context(path).error("forward packet is invalid");
}

void validate_reverse_packet(
    const std::string& path,
    const SymmetrixOAMCudaLayerArgsV2& args,
    const std::size_t layer_index)
{
    const bool direct =
        (args.flags&SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1) != 0;
    if (args.struct_size < sizeof(SymmetrixOAMCudaLayerArgsV2)
        || args.layer_index != layer_index
        || args.reserved != 0
        || (args.flags&~SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1) != 0
        || args.num_nodes < 0 || args.num_edges < 0
        || (args.num_nodes > 0
            && (args.receiver_offsets == nullptr
                || args.source_edge_offsets == nullptr
                || args.source_node_values == nullptr
                || args.target_node_output_adjoint == nullptr
                || args.source_node_input_adjoint == nullptr))
        || (args.num_edges > 0
            && (args.source_indices == nullptr
                || args.receiver_indices == nullptr
                || args.source_edge_indices == nullptr
                || args.distances == nullptr
                || args.spline_coefficients == nullptr
                || (direct
                    ? (args.edge_vectors == nullptr
                        || args.geometry_adjoint == nullptr)
                    : (args.edge_harmonics == nullptr
                        || args.edge_harmonic_adjoint == nullptr
                        || args.distance_adjoint == nullptr)))))
        throw context(path).error("reverse packet is invalid");
}

#if defined(KOKKOS_ENABLE_CUDA)
std::runtime_error driver_error(
    const CUresult status, const std::string& operation)
{
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    return std::runtime_error(
        operation + " failed: "
        + (name == nullptr ? std::string("CUDA_ERROR_UNKNOWN") : name)
        + (message == nullptr ? std::string()
            : std::string(" (") + message + ")"));
}

void check_driver(const CUresult status, const char* operation)
{
    if (status != CUDA_SUCCESS)
        throw driver_error(status, operation);
}

std::vector<char> read_cubin(const std::string& path)
{
    return detail::PluginLoaderContext("OAM CUDA", path).read_binary("cubin");
}

template<typename Value>
Value read_module_value(
    const std::string& path, const CUmodule module, const char* symbol)
{
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto lookup = cuModuleGetGlobal(&address, &size, module, symbol);
    if (lookup != CUDA_SUCCESS)
        throw context(path).error(
            std::string("could not resolve module identity ") + symbol + ": "
            + driver_error(lookup, "cuModuleGetGlobal").what());
    if (size != sizeof(Value))
        throw context(path).error(
            std::string("module identity has an invalid size for ") + symbol);
    Value value{};
    check_driver(cuMemcpyDtoH(&value, address, size),
        "cuMemcpyDtoH(module value)");
    return value;
}

std::string read_module_text(
    const std::string& path, const CUmodule module, const char* symbol,
    const std::size_t maximum_size = 512)
{
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto lookup = cuModuleGetGlobal(&address, &size, module, symbol);
    if (lookup != CUDA_SUCCESS)
        throw context(path).error(
            std::string("could not resolve module identity ") + symbol + ": "
            + driver_error(lookup, "cuModuleGetGlobal").what());
    if (size < 2 || size > maximum_size)
        throw context(path).error(
            std::string("module identity has an invalid size for ") + symbol);
    std::string value(size, '\0');
    check_driver(cuMemcpyDtoH(value.data(), address, size),
        "cuMemcpyDtoH(module identity)");
    if (value.back() != '\0')
        throw context(path).error(
            std::string("module identity is not null terminated: ") + symbol);
    value.pop_back();
    if (value.empty())
        throw context(path).error(
            std::string("module identity is empty: ") + symbol);
    return value;
}

template<typename Value>
std::vector<Value> read_module_array(
    const std::string& path, const CUmodule module, const char* symbol,
    const std::size_t count)
{
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto lookup = cuModuleGetGlobal(&address, &size, module, symbol);
    if (lookup != CUDA_SUCCESS)
        throw context(path).error(
            std::string("could not resolve module schedule ") + symbol + ": "
            + driver_error(lookup, "cuModuleGetGlobal").what());
    if (count == 0 || size != count * sizeof(Value))
        throw context(path).error(
            std::string("module schedule has an invalid size for ") + symbol);
    std::vector<Value> values(count);
    check_driver(cuMemcpyDtoH(values.data(), address, size),
        "cuMemcpyDtoH(module schedule)");
    return values;
}

void validate_module_context(const OAMCudaModuleState& state)
{
    CUcontext current = nullptr;
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current == nullptr || current != state.context)
        throw std::runtime_error(
            "The active CUDA context does not own the loaded OAM module");
}

void validate_function_block_size(
    const std::string& path, const CUfunction function,
    const std::int32_t threads)
{
    int maximum = 0;
    check_driver(
        cuFuncGetAttribute(
            &maximum, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
        "cuFuncGetAttribute");
    if (threads <= 0 || threads > maximum)
        throw context(path).error(
            "kernel thread-block size exceeds function limit");
}
void validate_function_shared_memory(
    const std::string& path, const CUfunction function)
{
    int shared_bytes = 0;
    check_driver(
        cuFuncGetAttribute(
            &shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
        "cuFuncGetAttribute(shared memory)");
    CUdevice device = 0;
    check_driver(cuCtxGetDevice(&device), "cuCtxGetDevice");
    int shared_limit = 0;
    check_driver(
        cuDeviceGetAttribute(
            &shared_limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            device),
        "cuDeviceGetAttribute(shared memory)");
    if (shared_bytes < 0 || shared_limit <= 0 || shared_bytes > shared_limit)
        throw context(path).error(
            "kernel static shared memory exceeds the active device limit");
}

#endif

}  // namespace

OAMCudaPlugin OAMCudaPlugin::load(
    std::string path,
    const OAMCudaPluginExpectation& expectation)
{
    if (path.empty())
        throw std::invalid_argument("Execution OAM CUDA plugin path is empty.");
    validate_expectation(expectation);
    if (std::string_view(path).ends_with(".cubin"))
        return load_cubin(std::move(path), expectation);

    void* handle = dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw context(path).error(
            error == nullptr ? "dlopen failed" : error);
    }
    try {
        dlerror();
        void* symbol = dlsym(handle, SYMMETRIX_OAM_CUDA_PLUGIN_QUERY_SYMBOL);
        const char* symbol_error = dlerror();
        if (symbol_error != nullptr) {
            const std::string query_error(symbol_error);
            dlerror();
            const bool stale_v2 = dlsym(
                handle, "symmetrix_oam_cuda_plugin_query_v2") != nullptr;
            dlerror();
            throw context(path).error(stale_v2
                ? "stale CUDA ABI v2 artifact; regenerate for exact resource ABI v3"
                : std::string("ABI v3 query is missing; regenerate the CUDA artifact: ")
                    + query_error);
        }
        if (symbol == nullptr)
            throw context(path).error("query symbol is null");
        const auto query =
            reinterpret_cast<SymmetrixOAMCudaPluginQueryV3>(symbol);
        const auto* descriptor = query();
        if (descriptor == nullptr)
            throw context(path).error("query returned a null descriptor");
        validate_descriptor(path, *descriptor, expectation);
        return OAMCudaPlugin(
            handle, descriptor, std::move(path),
            expectation.persistent_blocks_per_compute_unit);
    } catch (...) {
        dlclose(handle);
        throw;
    }
}

#if defined(KOKKOS_ENABLE_CUDA)
OAMCudaPlugin OAMCudaPlugin::load_cubin(
    std::string path,
    const OAMCudaPluginExpectation& expectation)
{
    if (path.empty())
        throw std::invalid_argument("Execution OAM CUDA cubin path is empty.");
    validate_expectation(expectation);
    const auto bytes = read_cubin(path);
    CUcontext cuda_context = nullptr;
    check_driver(cuCtxGetCurrent(&cuda_context), "cuCtxGetCurrent");
    if (cuda_context == nullptr)
        throw context(path).error("no CUDA context is active");

    CUmodule module = nullptr;
    check_driver(cuModuleLoadData(&module, bytes.data()), "cuModuleLoadData");
    try {
        const auto abi_version = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_abi_version");
        const auto struct_size = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_struct_size");
        const auto pointer_size = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_pointer_size");
        const auto byte_order = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_byte_order");
        const auto capabilities = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_capabilities");
        const auto scalar_kind = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_scalar_kind");
        const auto scalar_size = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_scalar_size");
        const auto layer_count = read_module_value<std::uint32_t>(
            path, module, "symmetrix_oam_cuda_module_layer_count");
        if (layer_count == 0
            || layer_count > SYMMETRIX_OAM_CUDA_PLUGIN_MAX_LAYERS)
            throw context(path).error("module layer count is invalid");

        auto state = std::make_unique<OAMCudaModuleState>();
        state->module = module;
        state->context = cuda_context;
        state->abi_tag = read_module_text(
            path, module, "symmetrix_oam_cuda_module_abi_tag");
        state->artifact_id = read_module_text(
            path, module, "symmetrix_oam_cuda_module_artifact_id");
        state->contract_fingerprint = read_module_text(
            path, module, "symmetrix_oam_cuda_module_contract_fingerprint");
        state->semantic_fingerprint = read_module_text(
            path, module, "symmetrix_oam_cuda_module_semantic_fingerprint");
        state->structure_fingerprint = read_module_text(
            path, module, "symmetrix_oam_cuda_module_structure_fingerprint");
        state->target_architecture = read_module_text(
            path, module, "symmetrix_oam_cuda_module_target_architecture");
        state->execution_variants = read_module_text(
            path, module, "symmetrix_oam_cuda_module_execution_variants");
        state->full_forward_schedule = read_module_text(
            path, module, "symmetrix_oam_cuda_module_full_forward_schedule");
        state->full_reverse_schedule = read_module_text(
            path, module, "symmetrix_oam_cuda_module_full_reverse_schedule");
        state->full_vector_policy = read_module_text(
            path, module, "symmetrix_oam_cuda_module_full_vector_policy");
        state->direct_geometry_policy = read_module_text(
            path, module, "symmetrix_oam_cuda_module_direct_geometry_policy");
        state->full_tiling_policy = read_module_text(
            path, module, "symmetrix_oam_cuda_module_full_tiling_policy");
        state->full_shared_memory_policy = read_module_text(
            path, module, "symmetrix_oam_cuda_module_full_shared_memory_policy");
        state->full_register_budget_policy = read_module_text(
            path, module, "symmetrix_oam_cuda_module_full_register_budget_policy");
        state->target_policy = read_module_text(
            path, module, "symmetrix_oam_cuda_module_target_policy");
        state->precision_policy = read_module_text(
            path, module, "symmetrix_oam_cuda_module_precision_policy");
        state->schedule_qualification = read_module_text(
            path, module, "symmetrix_oam_cuda_module_schedule_qualification");
        const auto target_compute_capability = read_module_value<std::int32_t>(
            path, module, "symmetrix_oam_cuda_module_target_compute_capability");
        const auto max_threads_per_block = read_module_value<std::int32_t>(
            path, module, "symmetrix_oam_cuda_module_max_threads_per_block");
        state->layers = read_module_array<SymmetrixOAMCudaLayerV2>(
            path, module, "symmetrix_oam_cuda_module_layers", layer_count);
        state->launches.resize(layer_count);
        for (std::size_t index = 0; index < layer_count; ++index) {
            state->launches[index] = {
                sizeof(SymmetrixOAMCudaLayerLaunchesV2),
                static_cast<std::uint32_t>(index), 0u, 0u,
                nullptr, nullptr};
        }
        state->descriptor = {
            abi_version,
            struct_size,
            pointer_size,
            byte_order,
            capabilities,
            scalar_kind,
            scalar_size,
            layer_count,
            0u,
            0u,
            state->abi_tag.c_str(),
            state->artifact_id.c_str(),
            state->contract_fingerprint.c_str(),
            state->semantic_fingerprint.c_str(),
            state->structure_fingerprint.c_str(),
            state->target_architecture.c_str(),
            target_compute_capability,
            max_threads_per_block,
            state->execution_variants.c_str(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            state->full_forward_schedule.c_str(),
            state->full_reverse_schedule.c_str(),
            state->full_vector_policy.c_str(),
            state->direct_geometry_policy.c_str(),
            state->full_tiling_policy.c_str(),
            state->full_shared_memory_policy.c_str(),
            state->full_register_budget_policy.c_str(),
            state->layers.data(),
            state->launches.data(),
            0, 0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            state->target_policy.c_str(),
            state->precision_policy.c_str(),
            state->schedule_qualification.c_str()};
        validate_descriptor(path, state->descriptor, expectation, true);

        const bool grouped_reverse =
            (capabilities&SYMMETRIX_OAM_CUDA_GROUPED_REVERSE_PARTIALS_V1) != 0;
        state->forward.reserve(layer_count);
        state->reverse.reserve(layer_count);
        state->reverse_groups.resize(layer_count);
        for (std::size_t index = 0; index < layer_count; ++index) {
            const auto forward_name =
                std::string("symmetrix_oam_forward_kernel_")
                + std::to_string(index);
            CUfunction forward = nullptr;
            CUfunction reverse = nullptr;
            check_driver(
                cuModuleGetFunction(&forward, module, forward_name.c_str()),
                "cuModuleGetFunction(OAM forward)");
            validate_function_block_size(
                path, forward, state->layers[index].forward_threads_per_block);
            validate_function_shared_memory(path, forward);
            if (grouped_reverse && state->layers[index].reverse_group_count > 0) {
                const auto count = state->layers[index].reverse_group_count;
                if (count <= 0)
                    throw context(path).error(
                        "grouped reverse layer has no source-irrep kernels");
                auto& functions = state->reverse_groups[index];
                functions.reserve(static_cast<std::size_t>(count));
                for (std::int32_t group = 0; group < count; ++group) {
                    const auto reverse_name =
                        std::string("symmetrix_oam_reverse_kernel_")
                        + std::to_string(index) + "_group_"
                        + std::to_string(group);
                    CUfunction function = nullptr;
                    check_driver(
                        cuModuleGetFunction(
                            &function, module, reverse_name.c_str()),
                        "cuModuleGetFunction(OAM grouped reverse)");
                    validate_function_block_size(path, function,
                        state->layers[index].reverse_threads_per_block);
                    validate_function_shared_memory(path, function);
                    functions.push_back(function);
                }
            } else {
                const auto reverse_name =
                    std::string("symmetrix_oam_reverse_kernel_")
                    + std::to_string(index);
                check_driver(
                    cuModuleGetFunction(&reverse, module, reverse_name.c_str()),
                    "cuModuleGetFunction(OAM reverse)");
                validate_function_block_size(path, reverse,
                    state->layers[index].reverse_threads_per_block);
                validate_function_shared_memory(path, reverse);
            }
            state->forward.push_back(forward);
            state->reverse.push_back(reverse);
        }
        OAMCudaPlugin result;
        result.path_ = std::move(path);
        result.module_state_ = std::move(state);
        result.persistent_blocks_per_compute_unit_ =
            expectation.persistent_blocks_per_compute_unit;
        return result;
    } catch (...) {
        cuModuleUnload(module);
        throw;
    }
}
#else
OAMCudaPlugin OAMCudaPlugin::load_cubin(
    std::string, const OAMCudaPluginExpectation&)
{
    throw std::runtime_error(
        "Execution OAM CUDA cubins require a CUDA-enabled Kokkos build.");
}
#endif

OAMCudaPlugin::OAMCudaPlugin(
    void* handle,
    const SymmetrixOAMCudaPluginV3* descriptor,
    std::string path,
    const std::int32_t persistent_blocks_per_compute_unit) noexcept
    : handle_(handle),
      descriptor_(descriptor),
      path_(std::move(path)),
      persistent_blocks_per_compute_unit_(persistent_blocks_per_compute_unit)
{}

OAMCudaPlugin::~OAMCudaPlugin()
{
    reset();
}

OAMCudaPlugin::OAMCudaPlugin(OAMCudaPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_(std::exchange(other.descriptor_, nullptr)),
      module_state_(std::move(other.module_state_)),
      path_(std::move(other.path_)),
      persistent_blocks_per_compute_unit_(
          std::exchange(other.persistent_blocks_per_compute_unit_, 4))
{}

OAMCudaPlugin& OAMCudaPlugin::operator=(OAMCudaPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_ = std::exchange(other.descriptor_, nullptr);
        module_state_ = std::move(other.module_state_);
        path_ = std::move(other.path_);
        persistent_blocks_per_compute_unit_ =
            std::exchange(other.persistent_blocks_per_compute_unit_, 4);
    }
    return *this;
}
const SymmetrixOAMCudaPluginV3& OAMCudaPlugin::descriptor() const
{
    if (module_state_ != nullptr)
        return module_state_->descriptor;
    if (descriptor_ == nullptr)
        throw std::logic_error("Execution OAM CUDA plugin handle is empty.");
    return *descriptor_;
}
std::size_t OAMCudaPlugin::layer_count() const noexcept
{
    if (module_state_ != nullptr)
        return module_state_->descriptor.layer_count;
    return descriptor_ == nullptr ? 0u : descriptor_->layer_count;
}
const SymmetrixOAMCudaLayerV2& OAMCudaPlugin::layer(
    const std::size_t layer_index) const
{
    const auto* descriptor = module_state_ != nullptr
        ? &module_state_->descriptor : descriptor_;
    if (descriptor == nullptr)
        throw std::logic_error("Execution OAM CUDA plugin handle is empty.");
    if (layer_index >= descriptor->layer_count)
        throw std::out_of_range(
            "Execution OAM CUDA layer index is outside the schedule.");
    return descriptor->layers[layer_index];
}


std::string_view OAMCudaPlugin::artifact_id() const
{
    return descriptor().artifact_id;
}

std::string_view OAMCudaPlugin::contract_fingerprint() const
{
    return descriptor().contract_fingerprint;
}

std::string_view OAMCudaPlugin::semantic_fingerprint() const
{
    return descriptor().semantic_fingerprint;
}

std::string_view OAMCudaPlugin::structure_fingerprint() const
{
    return descriptor().structure_fingerprint;
}

std::string_view OAMCudaPlugin::target_architecture() const
{
    return descriptor().target_architecture;
}

std::int32_t OAMCudaPlugin::launch_forward(
    const std::size_t layer_index,
    const SymmetrixOAMCudaLayerArgsV2* args,
    void* stream,
    const std::int32_t persistent_blocks) const
{
    const auto* descriptor = module_state_ != nullptr
        ? &module_state_->descriptor : descriptor_;
    if (descriptor == nullptr)
        throw std::logic_error("Execution OAM CUDA plugin handle is empty.");
    if (layer_index >= descriptor->layer_count)
        throw std::out_of_range(
            "Execution OAM CUDA layer index is outside the schedule.");
    if (args == nullptr)
        throw std::invalid_argument("Execution OAM CUDA forward packet is null.");
    validate_forward_packet(path_, *args, layer_index);
    if ((descriptor->capabilities
            & SYMMETRIX_OAM_CUDA_STAGED_FORWARD_INPUTS_V1) != 0
        && args->num_edges > 0 && args->edge_harmonics == nullptr)
        throw std::invalid_argument(
            "Execution OAM CUDA staged forward packet has no harmonic tape.");
    if (persistent_blocks <= 0)
        throw std::invalid_argument(
            "Execution OAM CUDA forward launch requires positive persistent blocks.");
#if defined(KOKKOS_ENABLE_CUDA)
    if (args->num_nodes == 0)
        return 0;
    if (module_state_ != nullptr) {
        validate_module_context(*module_state_);
        const auto& layer = module_state_->descriptor.layers[layer_index];
        const auto blocks = static_cast<std::int32_t>(
            args->num_nodes < persistent_blocks
                ? args->num_nodes : persistent_blocks);
        void* parameters[] = {const_cast<SymmetrixOAMCudaLayerArgsV2*>(args)};
        check_driver(
            cuLaunchKernel(
                module_state_->forward[layer_index],
                static_cast<unsigned int>(blocks), 1, 1,
                static_cast<unsigned int>(layer.forward_threads_per_block),
                1, 1, 0, reinterpret_cast<CUstream>(stream), parameters, nullptr),
            "cuLaunchKernel(OAM forward)");
        return 0;
    }
    return descriptor_->launches[layer_index].full_forward_launch(
        args, stream, persistent_blocks);
#else
    (void)stream;
    throw std::runtime_error(
        "Execution OAM CUDA plugins require a CUDA-enabled Kokkos build.");
#endif
}
std::int32_t OAMCudaPlugin::launch_reverse(
    const std::size_t layer_index,
    const SymmetrixOAMCudaLayerArgsV2* args,
    void* stream,
    const std::int32_t persistent_blocks) const
{
    const auto* descriptor = module_state_ != nullptr
        ? &module_state_->descriptor : descriptor_;
    if (descriptor == nullptr)
        throw std::logic_error("Execution OAM CUDA plugin handle is empty.");
    if (layer_index >= descriptor->layer_count)
        throw std::out_of_range(
            "Execution OAM CUDA layer index is outside the schedule.");
    if (args == nullptr)
        throw std::invalid_argument("Execution OAM CUDA reverse packet is null.");
    validate_reverse_packet(path_, *args, layer_index);
    const auto& scheduled_layer = descriptor->layers[layer_index];
    const bool grouped_reverse =
        (descriptor->capabilities
            & SYMMETRIX_OAM_CUDA_GROUPED_REVERSE_PARTIALS_V1) != 0
        && scheduled_layer.reverse_group_count > 0;
    if (grouped_reverse
        && (scheduled_layer.reverse_group_count <= 0
            || (args->num_edges > 0
                && (args->reverse_geometry_partials == nullptr
                    || args->edge_harmonics == nullptr
                    || args->reverse_harmonic_gradients == nullptr))))
        throw std::invalid_argument(
            "Execution OAM CUDA grouped reverse packet is incomplete.");
    if (persistent_blocks <= 0)
        throw std::invalid_argument(
            "Execution OAM CUDA reverse launch requires positive persistent blocks.");
#if defined(KOKKOS_ENABLE_CUDA)
    if (args->num_nodes == 0)
        return 0;
    if (module_state_ != nullptr) {
        validate_module_context(*module_state_);
        const auto& layer = module_state_->descriptor.layers[layer_index];
        const auto blocks = static_cast<std::int32_t>(
            args->num_nodes < persistent_blocks
                ? args->num_nodes : persistent_blocks);
        void* parameters[] = {const_cast<SymmetrixOAMCudaLayerArgsV2*>(args)};
        if (grouped_reverse) {
            for (const auto function : module_state_->reverse_groups[layer_index])
                check_driver(
                    cuLaunchKernel(
                        function,
                        static_cast<unsigned int>(blocks), 1, 1,
                        static_cast<unsigned int>(layer.reverse_threads_per_block),
                        1, 1, 0, reinterpret_cast<CUstream>(stream), parameters,
                        nullptr),
                    "cuLaunchKernel(OAM grouped reverse)");
        } else {
            check_driver(
                cuLaunchKernel(
                    module_state_->reverse[layer_index],
                    static_cast<unsigned int>(blocks), 1, 1,
                    static_cast<unsigned int>(layer.reverse_threads_per_block),
                    1, 1, 0, reinterpret_cast<CUstream>(stream), parameters,
                    nullptr),
                "cuLaunchKernel(OAM reverse)");
        }
        return 0;
    }
    const auto launch = descriptor_->launches[layer_index].full_reverse_launch;
    return launch(
        args, stream, persistent_blocks);
#else
    (void)stream;
    throw std::runtime_error(
        "Execution OAM CUDA plugins require a CUDA-enabled Kokkos build.");
#endif
}

void OAMCudaPlugin::reset() noexcept
{
    descriptor_ = nullptr;
    if (handle_ != nullptr) {
        dlclose(handle_);
        handle_ = nullptr;
    }
#if defined(KOKKOS_ENABLE_CUDA)
    if (module_state_ != nullptr && module_state_->module != nullptr)
        cuModuleUnload(module_state_->module);
#endif
    module_state_.reset();
    path_.clear();
    persistent_blocks_per_compute_unit_ = 4;
}

}  // namespace symmetrix::execution
#else
namespace symmetrix::execution {

OAMCudaPlugin OAMCudaPlugin::load(
    std::string,
    const OAMCudaPluginExpectation&)
{
    throw std::runtime_error(
        "Execution OAM CUDA plugins require POSIX dlopen/dlsym support.");
}
OAMCudaPlugin OAMCudaPlugin::load_cubin(
    std::string, const OAMCudaPluginExpectation&)
{
    throw std::runtime_error(
        "Execution OAM CUDA cubins require POSIX CUDA driver support.");
}
OAMCudaPlugin::OAMCudaPlugin() = default;

OAMCudaPlugin::OAMCudaPlugin(
    void* handle,
    const SymmetrixOAMCudaPluginV3* descriptor,
    std::string path,
    const std::int32_t persistent_blocks_per_compute_unit) noexcept
    : handle_(handle),
      descriptor_(descriptor),
      path_(std::move(path)),
      persistent_blocks_per_compute_unit_(persistent_blocks_per_compute_unit)
{}

OAMCudaPlugin::~OAMCudaPlugin()
{
    reset();
}

OAMCudaPlugin::OAMCudaPlugin(OAMCudaPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_(std::exchange(other.descriptor_, nullptr)),
      module_state_(std::move(other.module_state_)),
      path_(std::move(other.path_)),
      persistent_blocks_per_compute_unit_(
          std::exchange(other.persistent_blocks_per_compute_unit_, 4))
{}

OAMCudaPlugin& OAMCudaPlugin::operator=(OAMCudaPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_ = std::exchange(other.descriptor_, nullptr);
        module_state_ = std::move(other.module_state_);
        path_ = std::move(other.path_);
        persistent_blocks_per_compute_unit_ =
            std::exchange(other.persistent_blocks_per_compute_unit_, 4);
    }
    return *this;
}

const SymmetrixOAMCudaPluginV3& OAMCudaPlugin::descriptor() const
{
    throw std::logic_error("Execution OAM CUDA plugin support is disabled.");
}

std::size_t OAMCudaPlugin::layer_count() const noexcept
{
    return 0u;
}

const SymmetrixOAMCudaLayerV2& OAMCudaPlugin::layer(std::size_t) const
{
    throw std::logic_error("Execution OAM CUDA plugin support is disabled.");
}

std::string_view OAMCudaPlugin::artifact_id() const
{
    throw std::logic_error("Execution OAM CUDA plugin support is disabled.");
}

std::string_view OAMCudaPlugin::contract_fingerprint() const
{
    throw std::logic_error("Execution OAM CUDA plugin support is disabled.");
}

std::string_view OAMCudaPlugin::semantic_fingerprint() const
{
    throw std::logic_error("Execution OAM CUDA plugin support is disabled.");
}

std::string_view OAMCudaPlugin::structure_fingerprint() const
{
    throw std::logic_error("Execution OAM CUDA plugin support is disabled.");
}

std::string_view OAMCudaPlugin::target_architecture() const
{
    throw std::logic_error("Execution OAM CUDA plugin support is disabled.");
}

std::int32_t OAMCudaPlugin::launch_forward(
    std::size_t,
    const SymmetrixOAMCudaLayerArgsV2*,
    void*,
    std::int32_t) const
{
    throw std::logic_error(
        "Execution OAM CUDA plugin support requires POSIX dynamic loading.");
}

std::int32_t OAMCudaPlugin::launch_reverse(
    std::size_t,
    const SymmetrixOAMCudaLayerArgsV2*,
    void*,
    std::int32_t) const
{
    throw std::logic_error(
        "Execution OAM CUDA plugin support requires POSIX dynamic loading.");
}

void OAMCudaPlugin::reset() noexcept
{
    handle_ = nullptr;
    descriptor_ = nullptr;
    module_state_.reset();
    path_.clear();
    persistent_blocks_per_compute_unit_ = 4;
}

}  // namespace symmetrix::execution
#endif
