#include "jit_hip_plugin.hpp"
#include "jit_device_plugin_common.hpp"

#if __has_include(<Kokkos_Macros.hpp>)
#include <Kokkos_Macros.hpp>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include <fstream>
#include <cstddef>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<SymmetrixJitHipRadialSplineV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHipR1ForwardArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHipR1SourceArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHipR1EdgeArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHipR1ProjectedForwardArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHipR1ProjectedReverseArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHipPluginV1>);
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(SymmetrixJitHipRadialSplineV1) == 40);
static_assert(sizeof(SymmetrixJitHipR1ForwardArgsV1) == 152);
static_assert(sizeof(SymmetrixJitHipR1SourceArgsV1) == 152);
static_assert(sizeof(SymmetrixJitHipR1EdgeArgsV1) == 176);
static_assert(sizeof(SymmetrixJitHipR1ProjectedForwardArgsV1) == 160);
static_assert(sizeof(SymmetrixJitHipR1ProjectedReverseArgsV1) == 200);
static_assert(sizeof(SymmetrixJitHipPluginV1) == 144);
static_assert(alignof(SymmetrixJitHipRadialSplineV1) == 8);
static_assert(alignof(SymmetrixJitHipR1ForwardArgsV1) == 8);
static_assert(alignof(SymmetrixJitHipR1SourceArgsV1) == 8);
static_assert(alignof(SymmetrixJitHipR1EdgeArgsV1) == 8);
static_assert(alignof(SymmetrixJitHipPluginV1) == 8);
static_assert(offsetof(SymmetrixJitHipRadialSplineV1, coefficients) == 32);
static_assert(offsetof(SymmetrixJitHipR1ForwardArgsV1, radial) == 80);
static_assert(offsetof(SymmetrixJitHipR1SourceArgsV1, radial) == 80);
static_assert(offsetof(SymmetrixJitHipR1EdgeArgsV1, radial) == 88);
static_assert(offsetof(SymmetrixJitHipR1ForwardArgsV1, cutoff) == 144);
static_assert(offsetof(SymmetrixJitHipR1SourceArgsV1, cutoff) == 144);
static_assert(offsetof(SymmetrixJitHipR1EdgeArgsV1, cutoff) == 168);
static_assert(offsetof(SymmetrixJitHipR1ProjectedForwardArgsV1, cutoff) == 152);
static_assert(offsetof(SymmetrixJitHipR1ProjectedReverseArgsV1, cutoff) == 192);
static_assert(offsetof(SymmetrixJitHipPluginV1, r1_forward_launch) == 128);
#endif

namespace symmetrix::execution {

struct HipModuleState {
    void* module = nullptr;
    int device = -1;
    void* forward = nullptr;
    void* source = nullptr;
    void* edge = nullptr;
    void* reverse_fused = nullptr;
    void* tiled_forward = nullptr;
    void* tiled_reverse = nullptr;
    void* projected_forward = nullptr;
    void* projected_reverse = nullptr;
    int reverse_threads_per_block = 0;
    int tiled_forward_threads_per_block = 0;
    int tiled_reverse_threads_per_block = 0;
    int projected_threads_per_block = 0;
    int projected_shared_memory_bytes = 0;
    int edge_logical_subgroup_width = 1;
    std::string abi_tag;
    std::string artifact_id;
    std::string contract_fingerprint;
    std::string semantic_fingerprint;
    std::string structure_fingerprint;
    std::string target_architecture;
    std::string target_features;
    SymmetrixJitHipPluginV1 descriptor {};
};

namespace {

using detail::launch_blocks;

std::runtime_error plugin_error(
    const std::string& path, const std::string& message)
{
    return detail::PluginLoaderContext("HIP", path).error(message);
}

std::string require_text(
    const std::string& path, const char* value, const char* field)
{
    return detail::PluginLoaderContext("HIP", path).require_text(value, field);
}

void require_match(
    const std::string& path, const std::string_view expected,
    const char* actual, const char* field)
{
    detail::PluginLoaderContext("HIP", path).require_match(
        expected, actual, field);
}

void require_extent(
    const std::string& path, const std::int32_t expected,
    const std::int32_t actual, const char* field)
{
    detail::PluginLoaderContext("HIP", path).require_extent(
        expected, actual, field);
}

constexpr std::uint32_t known_capabilities =
    SYMMETRIX_JIT_HIP_R1_FORWARD_LAUNCH_V1
    |SYMMETRIX_JIT_HIP_R1_COORDINATE_REVERSE_LAUNCH_V1
    |SYMMETRIX_JIT_HIP_R1_FUSED_REVERSE_LAUNCH_V1;

#if defined(KOKKOS_ENABLE_HIP)
void check_hip(hipError_t status, const std::string& operation)
{
    if (status != hipSuccess)
        throw std::runtime_error(
            operation + " failed: " + hipGetErrorName(status) + " ("
            + hipGetErrorString(status) + ")");
}

std::vector<char> read_code_object(const std::string& path)
{
    return detail::PluginLoaderContext("HIP", path).read_binary(
        "HIP code object");
}

void validate_module_device(const HipModuleState& state)
{
    int current = -1;
    check_hip(hipGetDevice(&current), "hipGetDevice");
    if (current != state.device)
        throw std::runtime_error(
            "The active HIP device does not own the loaded Execution module");
}

void validate_function_block_size(
    const std::string& path, hipFunction_t function, std::int32_t threads)
{
    int maximum = 0;
    check_hip(
        hipFuncGetAttribute(
            &maximum, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
        "hipFuncGetAttribute");
    if (threads <= 0 || threads > maximum)
        throw plugin_error(path, "kernel thread-block size exceeds function limit");
}

template<typename Value>
Value read_module_value(
    const std::string& path, hipModule_t module, const char* symbol)
{
    hipDeviceptr_t address {};
    std::size_t size = 0;
    const auto lookup = hipModuleGetGlobal(&address, &size, module, symbol);
    if (lookup != hipSuccess)
        throw plugin_error(
            path, std::string("could not resolve module identity ") + symbol);
    if (size != sizeof(Value))
        throw plugin_error(path, "module identity has an invalid size");
    Value value {};
    check_hip(hipMemcpyDtoH(&value, address, size), "hipMemcpyDtoH(module value)");
    return value;
}

std::string read_module_text(
    const std::string& path, hipModule_t module, const char* symbol)
{
    hipDeviceptr_t address {};
    std::size_t size = 0;
    const auto lookup = hipModuleGetGlobal(&address, &size, module, symbol);
    if (lookup != hipSuccess)
        throw plugin_error(
            path, std::string("could not resolve module identity ") + symbol);
    if (size < 2 || size > 512)
        throw plugin_error(path, "module identity has an invalid size");
    std::string value(size, '\0');
    check_hip(
        hipMemcpyDtoH(value.data(), address, size),
        "hipMemcpyDtoH(module identity)");
    if (value.back() != '\0')
        throw plugin_error(path, "module identity is not null terminated");
    value.pop_back();
    if (value.empty())
        throw plugin_error(path, "module identity is empty");
    return value;
}
#endif

void validate_expectation(const HipPluginExpectation& expectation)
{
    if (expectation.artifact_id.empty()
        || expectation.contract_fingerprint.empty()
        || expectation.target_architecture.empty())
        throw std::invalid_argument(
            "Execution HIP plugin expectation requires artifact, contract, and target identity.");
    if (expectation.native_subgroup_width <= 0 || expectation.channels <= 0
        || expectation.embedding <= 0 || expectation.edge_l_max < 0
        || expectation.source_l_max < 0
        || expectation.max_threads_per_block <= 0
        || expectation.max_persistent_blocks_per_compute_unit <= 0)
        throw std::invalid_argument(
            "Execution HIP plugin expectation requires exact model and device extents.");
    if (expectation.required_capabilities == 0
        || (expectation.required_capabilities&~known_capabilities) != 0)
        throw std::invalid_argument(
            "Execution HIP plugin expectation requires exact known capabilities.");
    const bool valid_scalar =
        (expectation.scalar_kind == SYMMETRIX_JIT_HIP_SCALAR_FLOAT32_V1
            && expectation.scalar_size == sizeof(float))
        || (expectation.scalar_kind == SYMMETRIX_JIT_HIP_SCALAR_FLOAT64_V1
            && expectation.scalar_size == sizeof(double));
    if (!valid_scalar)
        throw std::invalid_argument(
            "Execution HIP plugin expectation has an invalid scalar kind or width.");
}

void validate_descriptor(
    const std::string& path, const SymmetrixJitHipPluginV1& descriptor,
    const HipPluginExpectation& expectation)
{
    if ((descriptor.capabilities&~known_capabilities) != 0)
        throw plugin_error(path, "launcher capabilities contain unknown bits");
    const detail::PluginLoaderContext context("HIP", path);
    detail::validate_descriptor_common(
        context, descriptor, expectation,
        SYMMETRIX_JIT_HIP_PLUGIN_ABI_VERSION_V1,
        sizeof(SymmetrixJitHipPluginV1),
        SYMMETRIX_JIT_HIP_PLUGIN_BYTE_ORDER,
        SYMMETRIX_JIT_HIP_PLUGIN_ABI_TAG_V1,
        SYMMETRIX_JIT_HIP_R1_FORWARD_LAUNCH_V1,
        SYMMETRIX_JIT_HIP_R1_COORDINATE_REVERSE_LAUNCH_V1);
    if (descriptor.persistent_blocks_per_compute_unit <= 0
        || descriptor.persistent_blocks_per_compute_unit
            > expectation.max_persistent_blocks_per_compute_unit)
        throw plugin_error(
            path, "persistent-block launch policy is invalid for this device");

    require_match(path, expectation.target_architecture,
        require_text(path, descriptor.target_architecture,
            "target architecture").c_str(), "target architecture");
    require_match(path, expectation.target_features,
        descriptor.target_features == nullptr ? "" : descriptor.target_features,
        "target features");
    require_extent(path, expectation.native_subgroup_width,
        descriptor.native_subgroup_width, "native subgroup width");
    require_extent(path, expectation.channels, descriptor.channels, "channel count");
    require_extent(path, expectation.embedding, descriptor.embedding,
        "embedding width");
    require_extent(path, expectation.edge_l_max, descriptor.edge_l_max,
        "edge l_max");
    require_extent(path, expectation.source_l_max, descriptor.source_l_max,
        "source l_max");
}

} // namespace

HipPlugin HipPlugin::load(
    std::string path, const HipPluginExpectation& expectation)
{
#if defined(__unix__) || defined(__APPLE__)
    if (path.empty())
        throw std::invalid_argument("Execution HIP plugin path is empty.");
    validate_expectation(expectation);
    void* handle = dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw plugin_error(path, error == nullptr ? "dlopen failed" : error);
    }
    try {
        dlerror();
        void* symbol = dlsym(handle, SYMMETRIX_JIT_HIP_PLUGIN_QUERY_SYMBOL_V1);
        const char* error = dlerror();
        if (error != nullptr)
            throw plugin_error(path, error);
        if (symbol == nullptr)
            throw plugin_error(path, "query symbol is null");
        const auto query = reinterpret_cast<SymmetrixJitHipPluginQueryV1>(symbol);
        const auto* descriptor = query();
        if (descriptor == nullptr)
            throw plugin_error(path, "query returned a null descriptor");
        validate_descriptor(path, *descriptor, expectation);
        return HipPlugin(handle, descriptor, std::move(path));
    } catch (...) {
        dlclose(handle);
        throw;
    }
#else
    (void)path;
    (void)expectation;
    throw std::runtime_error(
        "Execution HIP plugins require POSIX dlopen/dlsym support.");
#endif
}

HipPlugin HipPlugin::load_module(
    std::string path, const HipPluginExpectation& expectation)
{
#if defined(KOKKOS_ENABLE_HIP)
    if (path.empty())
        throw std::invalid_argument("Execution HIP module path is empty.");
    validate_expectation(expectation);
    const auto bytes = read_code_object(path);
    int device = -1;
    check_hip(hipGetDevice(&device), "hipGetDevice");
    hipModule_t module = nullptr;
    check_hip(hipModuleLoadData(&module, bytes.data()), "hipModuleLoadData");
    try {
        const auto contract_fingerprint = read_module_text(
            path, module, "symmetrix_factorized_contract_fingerprint");
        if (contract_fingerprint != expectation.contract_fingerprint)
            throw plugin_error(path, "module contract fingerprint does not match");
        const int edge_subgroup = read_module_value<int>(
            path, module, "symmetrix_factorized_edge_logical_subgroup_width");
        const int edge_threads = read_module_value<int>(
            path, module, "symmetrix_factorized_edge_threads_per_block");
        const int reverse_threads = read_module_value<int>(
            path, module, "symmetrix_factorized_reverse_threads_per_block");
        const int blocks_per_cu = read_module_value<int>(
            path, module,
            "symmetrix_factorized_persistent_blocks_per_compute_unit");
        if (edge_subgroup <= 0 || edge_subgroup > expectation.native_subgroup_width
            || edge_threads <= 0 || reverse_threads <= 0
            || edge_threads > expectation.max_threads_per_block
            || reverse_threads > expectation.max_threads_per_block
            || blocks_per_cu <= 0
            || blocks_per_cu
                > expectation.max_persistent_blocks_per_compute_unit)
            throw plugin_error(path, "module launch policy is invalid");

        hipFunction_t forward = nullptr;
        hipFunction_t source = nullptr;
        hipFunction_t edge = nullptr;
        hipFunction_t reverse_fused = nullptr;
        hipFunction_t tiled_forward = nullptr;
        hipFunction_t tiled_reverse = nullptr;
        hipFunction_t projected_forward = nullptr;
        hipFunction_t projected_reverse = nullptr;
        int projected_threads = 0;
        int projected_shared_memory = 0;
        check_hip(
            hipModuleGetFunction(
                &forward, module, "symmetrix_factorized_forward_v1"),
            "hipModuleGetFunction(forward)");
        check_hip(
            hipModuleGetFunction(
                &source, module, "symmetrix_factorized_source_v1"),
            "hipModuleGetFunction(source)");
        check_hip(
            hipModuleGetFunction(&edge, module, "symmetrix_factorized_edge_v1"),
            "hipModuleGetFunction(edge)");
        check_hip(
            hipModuleGetFunction(
                &reverse_fused, module,
                "symmetrix_factorized_reverse_fused_v1"),
            "hipModuleGetFunction(reverse fused)");
        int tiled_forward_threads = 0;
        int tiled_reverse_threads = 0;
        const auto tiled_lookup = hipModuleGetFunction(
            &tiled_forward, module,
            "symmetrix_factorized_tiled_forward_v1");
        if (tiled_lookup == hipSuccess) {
            check_hip(
                hipModuleGetFunction(
                    &tiled_reverse, module,
                    "symmetrix_factorized_tiled_reverse_v1"),
                "hipModuleGetFunction(tiled reverse)");
            tiled_forward_threads = read_module_value<int>(
                path, module,
                "symmetrix_factorized_tiled_forward_threads_per_block");
            tiled_reverse_threads = read_module_value<int>(
                path, module,
                "symmetrix_factorized_tiled_reverse_threads_per_block");
            if (tiled_forward_threads <= 0 || tiled_reverse_threads <= 0
                || tiled_forward_threads > expectation.max_threads_per_block
                || tiled_reverse_threads > expectation.max_threads_per_block)
                throw plugin_error(path, "tiled module launch policy is invalid");
        } else if (tiled_lookup != hipErrorNotFound) {
            check_hip(tiled_lookup, "hipModuleGetFunction(tiled forward)");
        }
        const auto projected_lookup = hipModuleGetFunction(
            &projected_forward, module,
            "symmetrix_factorized_projected_forward_v1");
        if (projected_lookup == hipSuccess) {
            check_hip(
                hipModuleGetFunction(
                    &projected_reverse, module,
                    "symmetrix_factorized_projected_reverse_v1"),
                "hipModuleGetFunction(projected reverse)");
            projected_threads = read_module_value<int>(
                path, module,
                "symmetrix_factorized_projected_threads_per_block");
            projected_shared_memory = read_module_value<int>(
                path, module,
                "symmetrix_factorized_projected_shared_memory_bytes");
            if (projected_threads <= 0 || projected_shared_memory <= 0
                || projected_threads > expectation.max_threads_per_block)
                throw plugin_error(path, "projected module launch policy is invalid");
        } else if (projected_lookup != hipErrorNotFound) {
            check_hip(projected_lookup, "hipModuleGetFunction(projected forward)");
        }
        validate_function_block_size(path, forward, 256);
        validate_function_block_size(path, source, 256);
        validate_function_block_size(path, edge, edge_threads);
        validate_function_block_size(path, reverse_fused, reverse_threads);
        if (tiled_forward != nullptr) {
            validate_function_block_size(
                path, tiled_forward, tiled_forward_threads);
            validate_function_block_size(
                path, tiled_reverse, tiled_reverse_threads);
        }
        if (projected_forward != nullptr) {
            validate_function_block_size(path, projected_forward, projected_threads);
            validate_function_block_size(path, projected_reverse, projected_threads);
        }

        HipPlugin result;
        result.path_ = std::move(path);
        result.module_state_ = std::make_unique<HipModuleState>();
        auto& state = *result.module_state_;
        state.module = reinterpret_cast<void*>(module);
        state.device = device;
        state.forward = reinterpret_cast<void*>(forward);
        state.source = reinterpret_cast<void*>(source);
        state.edge = reinterpret_cast<void*>(edge);
        state.reverse_fused = reinterpret_cast<void*>(reverse_fused);
        state.tiled_forward = reinterpret_cast<void*>(tiled_forward);
        state.tiled_reverse = reinterpret_cast<void*>(tiled_reverse);
        state.projected_forward = reinterpret_cast<void*>(projected_forward);
        state.projected_reverse = reinterpret_cast<void*>(projected_reverse);
        state.reverse_threads_per_block = reverse_threads;
        state.tiled_forward_threads_per_block = tiled_forward_threads;
        state.tiled_reverse_threads_per_block = tiled_reverse_threads;
        state.projected_threads_per_block = projected_threads;
        state.projected_shared_memory_bytes = projected_shared_memory;
        state.edge_logical_subgroup_width = edge_subgroup;
        state.abi_tag = SYMMETRIX_JIT_HIP_PLUGIN_ABI_TAG_V1;
        state.artifact_id = expectation.artifact_id;
        state.contract_fingerprint = expectation.contract_fingerprint;
        state.semantic_fingerprint = expectation.semantic_fingerprint;
        state.structure_fingerprint = expectation.structure_fingerprint;
        state.target_architecture = expectation.target_architecture;
        state.target_features = expectation.target_features;
        state.descriptor = {
            SYMMETRIX_JIT_HIP_PLUGIN_ABI_VERSION_V1,
            sizeof(SymmetrixJitHipPluginV1),
            sizeof(void*),
            SYMMETRIX_JIT_HIP_PLUGIN_BYTE_ORDER,
            expectation.required_capabilities
                | SYMMETRIX_JIT_HIP_R1_FUSED_REVERSE_LAUNCH_V1,
            expectation.scalar_kind,
            expectation.scalar_size,
            0u,
            state.abi_tag.c_str(),
            state.artifact_id.c_str(),
            state.contract_fingerprint.c_str(),
            state.semantic_fingerprint.c_str(),
            state.structure_fingerprint.c_str(),
            expectation.channels,
            expectation.embedding,
            expectation.edge_l_max,
            expectation.source_l_max,
            state.target_architecture.c_str(),
            state.target_features.c_str(),
            expectation.native_subgroup_width,
            256,
            256,
            edge_threads,
            blocks_per_cu,
            nullptr,
            nullptr,
        };
        return result;
    } catch (...) {
        (void)hipModuleUnload(module);
        throw;
    }
#else
    (void)path;
    (void)expectation;
    throw std::runtime_error(
        "Execution HIP modules require a HIP-enabled Kokkos build.");
#endif
}

HipPlugin::HipPlugin(
    void* handle, const SymmetrixJitHipPluginV1* descriptor,
    std::string path) noexcept
    : handle_(handle), descriptor_(descriptor), path_(std::move(path))
{}

HipPlugin::~HipPlugin() { reset(); }

HipPlugin::HipPlugin(HipPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_(std::exchange(other.descriptor_, nullptr)),
      module_state_(std::move(other.module_state_)),
      path_(std::move(other.path_))
{}

HipPlugin& HipPlugin::operator=(HipPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_ = std::exchange(other.descriptor_, nullptr);
        module_state_ = std::move(other.module_state_);
        path_ = std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitHipPluginV1& HipPlugin::descriptor() const
{
    if (module_state_ != nullptr)
        return module_state_->descriptor;
    if (descriptor_ == nullptr)
        throw std::logic_error("Execution HIP plugin handle is empty.");
    return *descriptor_;
}

std::string_view HipPlugin::artifact_id() const
{
    return descriptor().artifact_id;
}

std::string_view HipPlugin::contract_fingerprint() const
{
    return descriptor().contract_fingerprint;
}

std::int32_t HipPlugin::persistent_blocks_per_compute_unit() const
{
    return descriptor().persistent_blocks_per_compute_unit;
}

std::int32_t HipPlugin::launch_r1_forward(
    const SymmetrixJitHipR1ForwardArgsV1* args,
    void* stream, const std::int32_t persistent_blocks) const
{
    if (module_state_ == nullptr) {
        if (descriptor_ == nullptr || descriptor_->r1_forward_launch == nullptr)
            throw std::logic_error("Execution HIP R1 forward launcher is unavailable.");
        return descriptor_->r1_forward_launch(args, stream, persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_HIP)
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitHipR1ForwardArgsV1)
        || args->radial.struct_size < sizeof(SymmetrixJitHipRadialSplineV1)
        || args->num_nodes < 0 || args->num_edges < 0
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || persistent_blocks <= 0)
        throw std::invalid_argument("Execution HIP R1 forward packet is invalid");
    if (args->num_nodes == 0)
        return 0;
    validate_module_device(*module_state_);
    const auto& descriptor = module_state_->descriptor;
    const std::int64_t owners =
        static_cast<std::int64_t>(args->num_nodes) * descriptor.channels;
    const auto blocks = launch_blocks(
        owners, descriptor.forward_threads_per_block, persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitHipR1ForwardArgsV1*>(args)};
    check_hip(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(module_state_->forward),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(descriptor.forward_threads_per_block), 1, 1,
            0, reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution R1 forward)");
    return 0;
#else
    throw std::runtime_error("Execution HIP modules require a HIP-enabled build.");
#endif
}

std::int32_t HipPlugin::r1_reverse_physical_launch_count() const noexcept
{
    if (module_state_ != nullptr)
        return module_state_->reverse_fused == nullptr ? 2 : 1;
    const auto capabilities = descriptor_ == nullptr ? 0u : descriptor_->capabilities;
    return (capabilities&SYMMETRIX_JIT_HIP_R1_FUSED_REVERSE_LAUNCH_V1)
        == 0 ? 2 : 1;
}

bool HipPlugin::supports_tiled_r1() const noexcept
{
    return module_state_ != nullptr
        && module_state_->tiled_forward != nullptr
        && module_state_->tiled_reverse != nullptr;
}

std::int32_t HipPlugin::launch_r1_tiled_forward(
    const SymmetrixJitHipR1TiledForwardArgsV1* args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_HIP)
    if (!supports_tiled_r1())
        throw std::logic_error("Execution HIP tiled R1 module is unavailable.");
    const auto channels = module_state_->descriptor.channels;
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitHipR1TiledForwardArgsV1)
        || args->radial.struct_size < sizeof(SymmetrixJitHipRadialSplineV1)
        || args->num_nodes < 0 || args->num_edges < 0
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || args->channel_count != 64
        || args->channel_begin%64 != 0
        || args->channel_begin + args->channel_count
            > static_cast<std::uint32_t>(channels)
        || persistent_blocks <= 0)
        throw std::invalid_argument("Execution HIP tiled R1 forward packet is invalid");
    if (args->num_nodes == 0)
        return 0;
    validate_module_device(*module_state_);
    const std::int64_t owners =
        args->num_nodes * static_cast<std::int64_t>(args->channel_count);
    const auto blocks = launch_blocks(
        owners, module_state_->tiled_forward_threads_per_block,
        persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitHipR1TiledForwardArgsV1*>(args)};
    check_hip(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(module_state_->tiled_forward),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(
                module_state_->tiled_forward_threads_per_block),
            1, 1, 0, reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution tiled R1 forward)");
    return 0;
#else
    (void)args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution HIP modules require a HIP-enabled build.");
#endif
}

std::int32_t HipPlugin::launch_r1_tiled_reverse(
    const SymmetrixJitHipR1TiledSourceArgsV1* source_args,
    const SymmetrixJitHipR1TiledEdgeArgsV1* edge_args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_HIP)
    if (!supports_tiled_r1())
        throw std::logic_error("Execution HIP tiled R1 module is unavailable.");
    const auto channels = module_state_->descriptor.channels;
    if (source_args == nullptr || edge_args == nullptr
        || source_args->struct_size
            < sizeof(SymmetrixJitHipR1TiledSourceArgsV1)
        || edge_args->struct_size < sizeof(SymmetrixJitHipR1TiledEdgeArgsV1)
        || source_args->radial.struct_size
            < sizeof(SymmetrixJitHipRadialSplineV1)
        || edge_args->radial.struct_size
            < sizeof(SymmetrixJitHipRadialSplineV1)
        || source_args->num_nodes < 0 || source_args->num_edges < 0
        || source_args->num_edges != edge_args->num_edges
        || source_args->active_type_count != edge_args->active_type_count
        || source_args->cutoff != edge_args->cutoff
        || !std::isfinite(source_args->cutoff) || source_args->cutoff <= 0.0
        || source_args->channel_count != 64
        || source_args->channel_begin%64 != 0
        || source_args->channel_begin + source_args->channel_count
            > static_cast<std::uint32_t>(channels)
        || source_args->source_owner_count < 0
        || source_args->source_owner_count > source_args->num_edges
        || (source_args->source_owner_count > 0
            && source_args->source_ids == nullptr)
        || edge_args->coordinates_are_unit > 1
        || source_args->active_type_count == 0 || persistent_blocks <= 0)
        throw std::invalid_argument("Execution HIP tiled R1 reverse packet is invalid");
    if (source_args->num_nodes == 0 || source_args->num_edges == 0)
        return 0;
    validate_module_device(*module_state_);
    const auto source_owners = source_args->source_owner_count > 0
        ? source_args->source_owner_count : source_args->num_nodes;
    const auto blocks = static_cast<std::int32_t>(
        source_owners < persistent_blocks ? source_owners : persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitHipR1TiledSourceArgsV1*>(source_args),
        const_cast<SymmetrixJitHipR1TiledEdgeArgsV1*>(edge_args)};
    check_hip(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(module_state_->tiled_reverse),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(
                module_state_->tiled_reverse_threads_per_block),
            1, 1, 0, reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution tiled R1 reverse)");
    return 0;
#else
    (void)source_args;
    (void)edge_args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution HIP modules require a HIP-enabled build.");
#endif
}

bool HipPlugin::supports_projected_r1() const noexcept
{
    return module_state_ != nullptr
        && module_state_->projected_forward != nullptr
        && module_state_->projected_reverse != nullptr;
}

std::int32_t HipPlugin::launch_r1_projected_forward(
    const SymmetrixJitHipR1ProjectedForwardArgsV1* args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_HIP)
    if (!supports_projected_r1())
        throw std::logic_error("Execution HIP projected R1 module is unavailable.");
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitHipR1ProjectedForwardArgsV1)
        || args->radial.struct_size < sizeof(SymmetrixJitHipRadialSplineV1)
        || args->num_nodes < 0 || args->num_edges < 0
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || args->projection_weights == nullptr
        || persistent_blocks <= 0)
        throw std::invalid_argument("Execution HIP projected R1 forward packet is invalid");
    if (args->num_nodes == 0)
        return 0;
    validate_module_device(*module_state_);
    const auto blocks = launch_blocks(
        args->num_nodes, module_state_->projected_threads_per_block,
        persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitHipR1ProjectedForwardArgsV1*>(args)};
    check_hip(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(module_state_->projected_forward),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(module_state_->projected_threads_per_block),
            1, 1,
            static_cast<unsigned int>(module_state_->projected_shared_memory_bytes),
            reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution projected R1 forward)");
    return 0;
#else
    (void)args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution HIP modules require a HIP-enabled build.");
#endif
}

std::int32_t HipPlugin::launch_r1_projected_reverse(
    const SymmetrixJitHipR1ProjectedReverseArgsV1* args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_HIP)
    if (!supports_projected_r1())
        throw std::logic_error("Execution HIP projected R1 module is unavailable.");
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitHipR1ProjectedReverseArgsV1)
        || args->radial.struct_size < sizeof(SymmetrixJitHipRadialSplineV1)
        || args->num_nodes < 0 || args->num_edges < 0
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || args->coordinates_are_unit > 1
        || args->projection_weights == nullptr || persistent_blocks <= 0)
        throw std::invalid_argument("Execution HIP projected R1 reverse packet is invalid");
    if (args->num_nodes == 0 || args->num_edges == 0)
        return 0;
    validate_module_device(*module_state_);
    const auto blocks = launch_blocks(
        args->num_nodes, module_state_->projected_threads_per_block,
        persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitHipR1ProjectedReverseArgsV1*>(args)};
    check_hip(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(module_state_->projected_reverse),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(module_state_->projected_threads_per_block),
            1, 1,
            static_cast<unsigned int>(module_state_->projected_shared_memory_bytes),
            reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution projected R1 reverse)");
    return 0;
#else
    (void)args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution HIP modules require a HIP-enabled build.");
#endif
}

std::int32_t HipPlugin::launch_r1_coordinate_reverse(
    const SymmetrixJitHipR1SourceArgsV1* source_args,
    const SymmetrixJitHipR1EdgeArgsV1* edge_args,
    void* stream, const std::int32_t persistent_blocks) const
{
    if (module_state_ == nullptr) {
        if (descriptor_ == nullptr
            || descriptor_->r1_coordinate_reverse_launch == nullptr)
            throw std::logic_error("Execution HIP R1 reverse launcher is unavailable.");
        return descriptor_->r1_coordinate_reverse_launch(
            source_args, edge_args, stream, persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_HIP)
    if (source_args == nullptr || edge_args == nullptr
        || source_args->struct_size < sizeof(SymmetrixJitHipR1SourceArgsV1)
        || edge_args->struct_size < sizeof(SymmetrixJitHipR1EdgeArgsV1)
        || source_args->radial.struct_size
            < sizeof(SymmetrixJitHipRadialSplineV1)
        || edge_args->radial.struct_size
            < sizeof(SymmetrixJitHipRadialSplineV1)
        || source_args->num_nodes < 0 || source_args->num_edges < 0
        || source_args->num_edges != edge_args->num_edges
        || source_args->active_type_count != edge_args->active_type_count
        || source_args->cutoff != edge_args->cutoff
        || !std::isfinite(source_args->cutoff) || source_args->cutoff <= 0.0
        || edge_args->coordinates_are_unit > 1
        || (edge_args->coordinates_are_unit != 0
            ? edge_args->coordinate_scalar_size != sizeof(float)
            : edge_args->coordinate_scalar_size != sizeof(double))
        || source_args->active_type_count == 0 || persistent_blocks <= 0)
        throw std::invalid_argument("Execution HIP R1 reverse packet is invalid");
    if (source_args->num_nodes == 0 || source_args->num_edges == 0)
        return 0;
    validate_module_device(*module_state_);
    const auto& descriptor = module_state_->descriptor;
    if (module_state_->reverse_fused != nullptr) {
        const auto reverse_blocks = static_cast<std::int32_t>(
            source_args->num_nodes < persistent_blocks
                ? source_args->num_nodes : persistent_blocks);
        void* parameters[] = {
            const_cast<SymmetrixJitHipR1SourceArgsV1*>(source_args),
            const_cast<SymmetrixJitHipR1EdgeArgsV1*>(edge_args)};
        check_hip(
            hipModuleLaunchKernel(
                reinterpret_cast<hipFunction_t>(module_state_->reverse_fused),
                static_cast<unsigned int>(reverse_blocks), 1, 1,
                static_cast<unsigned int>(
                    module_state_->reverse_threads_per_block), 1, 1,
                0, reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
            "hipModuleLaunchKernel(Execution R1 fused reverse)");
        return 0;
    }
    const std::int64_t source_owners =
        static_cast<std::int64_t>(source_args->num_nodes) * descriptor.channels;
    const auto source_blocks = launch_blocks(
        source_owners, descriptor.source_threads_per_block, persistent_blocks);
    void* source_parameters[] = {
        const_cast<SymmetrixJitHipR1SourceArgsV1*>(source_args)};
    check_hip(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(module_state_->source),
            static_cast<unsigned int>(source_blocks), 1, 1,
            static_cast<unsigned int>(descriptor.source_threads_per_block), 1, 1,
            0, reinterpret_cast<hipStream_t>(stream), source_parameters, nullptr),
        "hipModuleLaunchKernel(Execution R1 source reverse)");
    const std::int64_t edge_work =
        static_cast<std::int64_t>(edge_args->num_edges)
        * module_state_->edge_logical_subgroup_width;
    const auto edge_blocks = launch_blocks(
        edge_work, descriptor.edge_threads_per_block, persistent_blocks);
    void* edge_parameters[] = {
        const_cast<SymmetrixJitHipR1EdgeArgsV1*>(edge_args)};
    check_hip(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(module_state_->edge),
            static_cast<unsigned int>(edge_blocks), 1, 1,
            static_cast<unsigned int>(descriptor.edge_threads_per_block), 1, 1,
            0, reinterpret_cast<hipStream_t>(stream), edge_parameters, nullptr),
        "hipModuleLaunchKernel(Execution R1 edge reverse)");
    return 0;
#else
    throw std::runtime_error("Execution HIP modules require a HIP-enabled build.");
#endif
}

void HipPlugin::reset() noexcept
{
    descriptor_ = nullptr;
#if defined(__unix__) || defined(__APPLE__)
    if (handle_ != nullptr)
        dlclose(handle_);
#endif
    handle_ = nullptr;
#if defined(KOKKOS_ENABLE_HIP)
    if (module_state_ != nullptr && module_state_->module != nullptr)
        (void)hipModuleUnload(
            reinterpret_cast<hipModule_t>(module_state_->module));
#endif
    module_state_.reset();
    path_.clear();
}

} // namespace symmetrix::execution
