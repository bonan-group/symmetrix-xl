#include "jit_cuda_plugin.hpp"
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

#include <fstream>
#include <cstddef>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(std::is_standard_layout_v<SymmetrixJitCudaRadialSplineV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1ForwardArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1SourceArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1EdgeArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaPluginV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaRadialSplineV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1ForwardArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1SourceArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1EdgeArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1ProjectedForwardArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaR1ProjectedReverseArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitCudaPluginV2>);
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(SymmetrixJitCudaRadialSplineV1) == 40);
static_assert(sizeof(SymmetrixJitCudaR1ForwardArgsV1) == 152);
static_assert(sizeof(SymmetrixJitCudaR1SourceArgsV1) == 152);
static_assert(sizeof(SymmetrixJitCudaR1EdgeArgsV1) == 168);
static_assert(sizeof(SymmetrixJitCudaPluginV1) == 112);
static_assert(sizeof(SymmetrixJitCudaRadialSplineV2) == 40);
static_assert(sizeof(SymmetrixJitCudaR1ForwardArgsV2) == 152);
static_assert(sizeof(SymmetrixJitCudaR1SourceArgsV2) == 152);
static_assert(sizeof(SymmetrixJitCudaR1EdgeArgsV2) == 176);
static_assert(sizeof(SymmetrixJitCudaR1ProjectedForwardArgsV2) == 160);
static_assert(sizeof(SymmetrixJitCudaR1ProjectedReverseArgsV2) == 200);
static_assert(sizeof(SymmetrixJitCudaPluginV2) == 120);
static_assert(alignof(SymmetrixJitCudaRadialSplineV1) == 8);
static_assert(alignof(SymmetrixJitCudaR1ForwardArgsV1) == 8);
static_assert(alignof(SymmetrixJitCudaR1SourceArgsV1) == 8);
static_assert(alignof(SymmetrixJitCudaR1EdgeArgsV1) == 8);
static_assert(alignof(SymmetrixJitCudaPluginV1) == 8);
static_assert(alignof(SymmetrixJitCudaRadialSplineV2) == 8);
static_assert(alignof(SymmetrixJitCudaR1ForwardArgsV2) == 8);
static_assert(alignof(SymmetrixJitCudaR1SourceArgsV2) == 8);
static_assert(alignof(SymmetrixJitCudaR1EdgeArgsV2) == 8);
static_assert(alignof(SymmetrixJitCudaPluginV2) == 8);
static_assert(offsetof(SymmetrixJitCudaRadialSplineV1, coefficients) == 32);
static_assert(offsetof(SymmetrixJitCudaR1ForwardArgsV1, radial) == 80);
static_assert(offsetof(SymmetrixJitCudaR1SourceArgsV1, radial) == 80);
static_assert(offsetof(SymmetrixJitCudaR1EdgeArgsV1, radial) == 80);
static_assert(offsetof(SymmetrixJitCudaR1ForwardArgsV1, cutoff) == 144);
static_assert(offsetof(SymmetrixJitCudaR1SourceArgsV1, cutoff) == 144);
static_assert(offsetof(SymmetrixJitCudaR1EdgeArgsV1, cutoff) == 160);
static_assert(offsetof(SymmetrixJitCudaPluginV1, r1_forward_launch) == 96);
static_assert(offsetof(SymmetrixJitCudaRadialSplineV2, coefficients) == 32);
static_assert(offsetof(SymmetrixJitCudaR1ForwardArgsV2, radial) == 80);
static_assert(offsetof(SymmetrixJitCudaR1SourceArgsV2, radial) == 80);
static_assert(offsetof(SymmetrixJitCudaR1EdgeArgsV2, radial) == 88);
static_assert(offsetof(SymmetrixJitCudaR1ForwardArgsV2, cutoff) == 144);
static_assert(offsetof(SymmetrixJitCudaR1SourceArgsV2, cutoff) == 144);
static_assert(offsetof(SymmetrixJitCudaR1EdgeArgsV2, cutoff) == 168);
static_assert(offsetof(SymmetrixJitCudaR1ProjectedForwardArgsV2, cutoff) == 152);
static_assert(offsetof(SymmetrixJitCudaR1ProjectedReverseArgsV2, cutoff) == 192);
static_assert(offsetof(SymmetrixJitCudaPluginV2, r1_forward_launch) == 104);
#endif

#if defined(__unix__) || defined(__APPLE__)
namespace symmetrix::execution {

struct CudaModuleState {
    void* module = nullptr;
    void* context = nullptr;
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
    std::string abi_tag;
    std::string artifact_id;
    std::string contract_fingerprint;
    std::string semantic_fingerprint;
    std::string structure_fingerprint;
    SymmetrixJitCudaPluginV2 descriptor {};
};

namespace {

using detail::launch_blocks;

std::runtime_error plugin_error(
    const std::string& path,
    const std::string& message)
{
    return detail::PluginLoaderContext("CUDA", path).error(message);
}

std::string require_text(
    const std::string& path,
    const char* value,
    const char* field)
{
    return detail::PluginLoaderContext("CUDA", path).require_text(value, field);
}

void require_match(
    const std::string& path,
    const std::string_view expected,
    const char* actual,
    const char* field)
{
    detail::PluginLoaderContext("CUDA", path).require_match(
        expected, actual, field);
}

void require_extent(
    const std::string& path,
    const std::int32_t expected,
    const std::int32_t actual,
    const char* field)
{
    detail::PluginLoaderContext("CUDA", path).require_extent(
        expected, actual, field);
}

constexpr std::uint32_t known_capabilities =
    SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V1
    | SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V1
    | SYMMETRIX_JIT_CUDA_R1_FUSED_REVERSE_LAUNCH_V2;

void validate_expectation(const CudaPluginExpectation& expectation)
{
    if (expectation.artifact_id.empty())
        throw std::invalid_argument(
            "Execution CUDA plugin expectation requires an artifact id.");
    if (expectation.contract_fingerprint.empty())
        throw std::invalid_argument(
            "Execution CUDA plugin expectation requires a contract fingerprint.");
    if (expectation.channels <= 0 || expectation.embedding <= 0
        || expectation.edge_l_max < 0 || expectation.source_l_max < 0
        || expectation.target_compute_capability <= 0
        || expectation.max_threads_per_block <= 0
        || expectation.persistent_blocks_per_compute_unit <= 0
        || expectation.persistent_blocks_per_compute_unit > 32)
        throw std::invalid_argument(
            "Execution CUDA plugin expectation requires exact model and device extents.");
    if (expectation.required_capabilities == 0
        || (expectation.required_capabilities&~known_capabilities) != 0)
        throw std::invalid_argument(
            "Execution CUDA plugin expectation requires exact known capabilities.");
    if ((expectation.scalar_kind
                == SYMMETRIX_JIT_CUDA_SCALAR_FLOAT32_V2
            && expectation.scalar_size != sizeof(float))
        || (expectation.scalar_kind
                == SYMMETRIX_JIT_CUDA_SCALAR_FLOAT64_V2
            && expectation.scalar_size != sizeof(double))
        || (expectation.scalar_kind
                != SYMMETRIX_JIT_CUDA_SCALAR_FLOAT32_V2
            && expectation.scalar_kind
                != SYMMETRIX_JIT_CUDA_SCALAR_FLOAT64_V2))
        throw std::invalid_argument(
            "Execution CUDA plugin expectation has an invalid scalar kind or width.");
}

#if defined(KOKKOS_ENABLE_CUDA)
std::runtime_error driver_error(CUresult status, const std::string& operation)
{
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    return std::runtime_error(
        operation + " failed: "
        + (name == nullptr ? std::string("CUDA_ERROR_UNKNOWN") : name)
        + (message == nullptr ? std::string() : std::string(" (") + message + ")"));
}

void check_driver(CUresult status, const std::string& operation)
{
    if (status != CUDA_SUCCESS)
        throw driver_error(status, operation);
}

std::vector<char> read_cubin(const std::string& path)
{
    return detail::PluginLoaderContext("CUDA", path).read_binary("cubin");
}

void validate_module_context(const CudaModuleState& state)
{
    CUcontext current = nullptr;
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current == nullptr)
        throw std::runtime_error("No CUDA context is active for the Execution module");
    if (reinterpret_cast<void*>(current) != state.context)
        throw std::runtime_error(
            "The active CUDA context does not own the loaded Execution module");
}

void validate_function_block_size(
    const std::string& path, CUfunction function, std::int32_t threads)
{
    int maximum = 0;
    check_driver(
        cuFuncGetAttribute(
            &maximum, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
        "cuFuncGetAttribute");
    if (threads <= 0 || threads > maximum)
        throw plugin_error(path, "kernel thread-block size exceeds function limit");
}

template<typename Value>
Value read_module_value(
    const std::string& path, CUmodule module, const char* symbol)
{
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto lookup = cuModuleGetGlobal(&address, &size, module, symbol);
    if (lookup != CUDA_SUCCESS)
        throw plugin_error(
            path,
            std::string("could not resolve module identity ") + symbol + ": "
                + driver_error(lookup, "cuModuleGetGlobal").what());
    if (size != sizeof(Value))
        throw plugin_error(path, "module identity has an invalid size");
    Value value {};
    check_driver(
        cuMemcpyDtoH(&value, address, size), "cuMemcpyDtoH(module value)");
    return value;
}

std::string read_module_text(
    const std::string& path,
    CUmodule module,
    const char* symbol,
    std::size_t maximum_size = 512)
{
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto lookup = cuModuleGetGlobal(&address, &size, module, symbol);
    if (lookup != CUDA_SUCCESS)
        throw plugin_error(
            path,
            std::string("could not resolve module identity ") + symbol + ": "
                + driver_error(lookup, "cuModuleGetGlobal").what());
    if (size < 2 || size > maximum_size)
        throw plugin_error(path, "module identity has an invalid size");
    std::string value(size, '\0');
    check_driver(
        cuMemcpyDtoH(value.data(), address, size),
        "cuMemcpyDtoH(module identity)");
    if (value.back() != '\0')
        throw plugin_error(path, "module identity is not null terminated");
    value.pop_back();
    if (value.empty())
        throw plugin_error(path, "module identity is empty");
    return value;
}
#endif

void validate_descriptor_v1(
    const std::string& path,
    const SymmetrixJitCudaPluginV1& descriptor,
    const CudaPluginExpectation& expectation)
{
    if (expectation.scalar_kind != SYMMETRIX_JIT_CUDA_SCALAR_FLOAT32_V2
        || expectation.scalar_size != sizeof(float))
        throw plugin_error(path, "ABI version 1 only supports float32");
    if (descriptor.abi_version != SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION)
        throw plugin_error(path, "unsupported ABI version");
    if (descriptor.struct_size < sizeof(SymmetrixJitCudaPluginV1))
        throw plugin_error(path, "descriptor is smaller than ABI version 1");
    if (descriptor.pointer_size != sizeof(void*))
        throw plugin_error(path, "pointer width does not match this process");
    if (descriptor.byte_order != SYMMETRIX_JIT_CUDA_PLUGIN_BYTE_ORDER)
        throw plugin_error(path, "byte order does not match this process");
    if (descriptor.reserved != 0)
        throw plugin_error(path, "reserved descriptor field is nonzero");

    const auto abi_tag = require_text(path, descriptor.abi_tag, "ABI tag");
    if (abi_tag != SYMMETRIX_JIT_CUDA_PLUGIN_ABI_TAG)
        throw plugin_error(path, "ABI tag does not match");
    const auto artifact_id =
        require_text(path, descriptor.artifact_id, "artifact id");
    const auto contract_fingerprint = require_text(
        path, descriptor.contract_fingerprint, "contract fingerprint");
    const auto semantic_fingerprint = require_text(
        path, descriptor.semantic_fingerprint, "semantic fingerprint");
    const auto structure_fingerprint = require_text(
        path, descriptor.structure_fingerprint, "structure fingerprint");

    if (descriptor.channels <= 0 || descriptor.embedding <= 0
        || descriptor.edge_l_max < 0 || descriptor.source_l_max < 0)
        throw plugin_error(path, "model extents are invalid");
    if (descriptor.capabilities != expectation.required_capabilities)
        throw plugin_error(path, "launcher capabilities do not match");
    if ((descriptor.capabilities
            &SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V1)
            != (descriptor.r1_forward_launch != nullptr
                ? SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V1 : 0u)
        || (descriptor.capabilities
            &SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V1)
            != (descriptor.r1_coordinate_reverse_launch != nullptr
                ? SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V1 : 0u))
        throw plugin_error(
            path, "launcher capability and function pointers disagree");
    if (descriptor.forward_threads_per_block <= 0
        || descriptor.source_threads_per_block <= 0
        || descriptor.edge_threads_per_block <= 0
        || descriptor.forward_threads_per_block
            > expectation.max_threads_per_block
        || descriptor.source_threads_per_block
            > expectation.max_threads_per_block
        || descriptor.edge_threads_per_block
            > expectation.max_threads_per_block)
        throw plugin_error(path, "thread-block size is invalid for this device");

    require_match(path, expectation.artifact_id, artifact_id.c_str(), "artifact id");
    require_match(
        path, expectation.contract_fingerprint,
        contract_fingerprint.c_str(), "contract fingerprint");
    require_match(
        path, expectation.semantic_fingerprint,
        semantic_fingerprint.c_str(), "semantic fingerprint");
    require_match(
        path, expectation.structure_fingerprint,
        structure_fingerprint.c_str(), "structure fingerprint");
    require_extent(path, expectation.channels, descriptor.channels, "channel count");
    require_extent(path, expectation.embedding, descriptor.embedding, "embedding width");
    require_extent(path, expectation.edge_l_max, descriptor.edge_l_max, "edge l_max");
    require_extent(
        path, expectation.source_l_max,
        descriptor.source_l_max, "source l_max");
    require_extent(
        path, expectation.target_compute_capability,
        descriptor.target_compute_capability, "target compute capability");
}

void validate_descriptor_v2(
    const std::string& path,
    const SymmetrixJitCudaPluginV2& descriptor,
    const CudaPluginExpectation& expectation)
{
    if ((descriptor.capabilities&~known_capabilities) != 0)
        throw plugin_error(path, "launcher capabilities contain unknown bits");
    const detail::PluginLoaderContext context("CUDA", path);
    detail::validate_descriptor_common(
        context, descriptor, expectation,
        SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION_V2,
        sizeof(SymmetrixJitCudaPluginV2),
        SYMMETRIX_JIT_CUDA_PLUGIN_BYTE_ORDER,
        SYMMETRIX_JIT_CUDA_PLUGIN_ABI_TAG_V2,
        SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V2,
        SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V2);
    context.require_extent(
        expectation.target_compute_capability,
        descriptor.target_compute_capability, "target compute capability");
}

}  // namespace

CudaPlugin CudaPlugin::load(
    std::string path,
    const CudaPluginExpectation& expectation)
{
    if (path.empty())
        throw std::invalid_argument("Execution CUDA plugin path is empty.");
    validate_expectation(expectation);

    void* handle = dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw plugin_error(path, error == nullptr ? "dlopen failed" : error);
    }
    try {
        dlerror();
        void* symbol_v2 = dlsym(
            handle, SYMMETRIX_JIT_CUDA_PLUGIN_QUERY_SYMBOL_V2);
        const char* symbol_v2_error = dlerror();
        if (symbol_v2_error == nullptr && symbol_v2 != nullptr) {
            const auto query =
                reinterpret_cast<SymmetrixJitCudaPluginQueryV2>(symbol_v2);
            const auto* descriptor = query();
            if (descriptor == nullptr)
                throw plugin_error(path, "query returned a null descriptor");
            validate_descriptor_v2(path, *descriptor, expectation);
            CudaPlugin result(handle, nullptr, descriptor, std::move(path));
            result.persistent_blocks_per_compute_unit_ =
                expectation.persistent_blocks_per_compute_unit;
            return result;
        }

        dlerror();
        void* symbol_v1 = dlsym(
            handle, SYMMETRIX_JIT_CUDA_PLUGIN_QUERY_SYMBOL);
        const char* symbol_v1_error = dlerror();
        if (symbol_v1_error != nullptr)
            throw plugin_error(path, symbol_v1_error);
        if (symbol_v1 == nullptr)
            throw plugin_error(path, "query symbol is null");
        const auto query =
            reinterpret_cast<SymmetrixJitCudaPluginQueryV1>(symbol_v1);
        const auto* descriptor = query();
        if (descriptor == nullptr)
            throw plugin_error(path, "query returned a null descriptor");
        validate_descriptor_v1(path, *descriptor, expectation);
        CudaPlugin result(handle, descriptor, nullptr, std::move(path));
        result.persistent_blocks_per_compute_unit_ =
            expectation.persistent_blocks_per_compute_unit;
        return result;
    } catch (...) {
        dlclose(handle);
        throw;
    }
}

CudaPlugin CudaPlugin::load_cubin(
    std::string path,
    const CudaPluginExpectation& expectation)
{
#if defined(KOKKOS_ENABLE_CUDA)
    if (path.empty())
        throw std::invalid_argument("Execution CUDA cubin path is empty.");
    validate_expectation(expectation);
    const auto bytes = read_cubin(path);
    CUcontext context = nullptr;
    check_driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (context == nullptr)
        throw plugin_error(path, "no CUDA context is active");

    CUmodule module = nullptr;
    check_driver(cuModuleLoadData(&module, bytes.data()), "cuModuleLoadData");
    try {
        const auto contract_fingerprint = read_module_text(
            path, module, "symmetrix_factorized_contract_fingerprint");
        if (contract_fingerprint != expectation.contract_fingerprint)
            throw plugin_error(path, "module contract fingerprint does not match");
        const int forward_threads = read_module_value<int>(
            path, module, "symmetrix_factorized_forward_threads_per_block");
        const int source_threads = read_module_value<int>(
            path, module, "symmetrix_factorized_source_threads_per_block");
        const int edge_threads = read_module_value<int>(
            path, module, "symmetrix_factorized_edge_threads_per_block");
        const int reverse_threads = read_module_value<int>(
            path, module, "symmetrix_factorized_reverse_threads_per_block");
        const int persistent_blocks = read_module_value<int>(
            path, module,
            "symmetrix_factorized_persistent_blocks_per_compute_unit");
        if (forward_threads <= 0 || source_threads <= 0 || edge_threads <= 0
            || reverse_threads <= 0
            || forward_threads > expectation.max_threads_per_block
            || source_threads > expectation.max_threads_per_block
            || edge_threads > expectation.max_threads_per_block
            || reverse_threads > expectation.max_threads_per_block)
            throw plugin_error(path, "module thread-block size is invalid");
        if (persistent_blocks != expectation.persistent_blocks_per_compute_unit)
            throw plugin_error(
                path, "module persistent-block policy does not match");
        CUfunction forward = nullptr;
        CUfunction source = nullptr;
        CUfunction edge = nullptr;
        CUfunction reverse_fused = nullptr;
        CUfunction tiled_forward = nullptr;
        CUfunction tiled_reverse = nullptr;
        CUfunction projected_forward = nullptr;
        CUfunction projected_reverse = nullptr;
        int projected_threads = 0;
        int projected_shared_memory = 0;
        check_driver(
            cuModuleGetFunction(
                &forward, module, "symmetrix_factorized_forward_v2"),
            "cuModuleGetFunction(forward)");
        check_driver(
            cuModuleGetFunction(
                &source, module, "symmetrix_factorized_source_v2"),
            "cuModuleGetFunction(source)");
        check_driver(
            cuModuleGetFunction(
                &edge, module, "symmetrix_factorized_edge_v2"),
            "cuModuleGetFunction(edge)");
        check_driver(
            cuModuleGetFunction(
                &reverse_fused, module,
                "symmetrix_factorized_reverse_fused_v2"),
            "cuModuleGetFunction(reverse fused)");
        int tiled_forward_threads = 0;
        int tiled_reverse_threads = 0;
        const auto tiled_lookup = cuModuleGetFunction(
            &tiled_forward, module,
            "symmetrix_factorized_tiled_forward_v2");
        if (tiled_lookup == CUDA_SUCCESS) {
            check_driver(
                cuModuleGetFunction(
                    &tiled_reverse, module,
                    "symmetrix_factorized_tiled_reverse_v2"),
                "cuModuleGetFunction(tiled reverse)");
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
        } else if (tiled_lookup != CUDA_ERROR_NOT_FOUND) {
            check_driver(tiled_lookup, "cuModuleGetFunction(tiled forward)");
        }
        const auto projected_lookup = cuModuleGetFunction(
            &projected_forward, module,
            "symmetrix_factorized_projected_forward_v2");
        if (projected_lookup == CUDA_SUCCESS) {
            check_driver(
                cuModuleGetFunction(
                    &projected_reverse, module,
                    "symmetrix_factorized_projected_reverse_v2"),
                "cuModuleGetFunction(projected reverse)");
            projected_threads = read_module_value<int>(
                path, module,
                "symmetrix_factorized_projected_threads_per_block");
            projected_shared_memory = read_module_value<int>(
                path, module,
                "symmetrix_factorized_projected_shared_memory_bytes");
            if (projected_threads <= 0 || projected_shared_memory <= 0
                || projected_threads > expectation.max_threads_per_block)
                throw plugin_error(path, "projected module launch policy is invalid");
        } else if (projected_lookup != CUDA_ERROR_NOT_FOUND) {
            check_driver(projected_lookup, "cuModuleGetFunction(projected forward)");
        }
        validate_function_block_size(path, forward, forward_threads);
        validate_function_block_size(path, source, source_threads);
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

        CudaPlugin result;
        result.path_ = std::move(path);
        result.module_state_ = std::make_unique<CudaModuleState>();
        auto& state = *result.module_state_;
        state.module = reinterpret_cast<void*>(module);
        state.context = reinterpret_cast<void*>(context);
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
        state.abi_tag = SYMMETRIX_JIT_CUDA_PLUGIN_ABI_TAG_V2;
        state.artifact_id = expectation.artifact_id;
        state.contract_fingerprint = expectation.contract_fingerprint;
        state.semantic_fingerprint = expectation.semantic_fingerprint;
        state.structure_fingerprint = expectation.structure_fingerprint;
        state.descriptor = {
            SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION_V2,
            sizeof(SymmetrixJitCudaPluginV2),
            sizeof(void*),
            SYMMETRIX_JIT_CUDA_PLUGIN_BYTE_ORDER,
            expectation.required_capabilities
                | SYMMETRIX_JIT_CUDA_R1_FUSED_REVERSE_LAUNCH_V2,
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
            expectation.target_compute_capability,
            forward_threads,
            source_threads,
            edge_threads,
            nullptr,
            nullptr,
        };
        result.persistent_blocks_per_compute_unit_ = persistent_blocks;
        return result;
    } catch (...) {
        cuModuleUnload(module);
        throw;
    }
#else
    (void)path;
    (void)expectation;
    throw std::runtime_error(
        "Execution CUDA cubins require a CUDA-enabled Kokkos build.");
#endif
}

CudaPlugin::CudaPlugin(
    void* handle,
    const SymmetrixJitCudaPluginV1* descriptor_v1,
    const SymmetrixJitCudaPluginV2* descriptor_v2,
    std::string path) noexcept
    : handle_(handle), descriptor_v1_(descriptor_v1),
      descriptor_v2_(descriptor_v2), path_(std::move(path))
{}

CudaPlugin::CudaPlugin() = default;

CudaPlugin::~CudaPlugin()
{
    reset();
}

CudaPlugin::CudaPlugin(CudaPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_v1_(std::exchange(other.descriptor_v1_, nullptr)),
      descriptor_v2_(std::exchange(other.descriptor_v2_, nullptr)),
      module_state_(std::move(other.module_state_)),
      path_(std::move(other.path_)),
      persistent_blocks_per_compute_unit_(
          other.persistent_blocks_per_compute_unit_)
{}

CudaPlugin& CudaPlugin::operator=(CudaPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_v1_ = std::exchange(other.descriptor_v1_, nullptr);
        descriptor_v2_ = std::exchange(other.descriptor_v2_, nullptr);
        module_state_ = std::move(other.module_state_);
        path_ = std::move(other.path_);
        persistent_blocks_per_compute_unit_ =
            other.persistent_blocks_per_compute_unit_;
    }
    return *this;
}

const SymmetrixJitCudaPluginV1& CudaPlugin::descriptor() const
{
    return descriptor_v1();
}

std::uint32_t CudaPlugin::abi_version() const noexcept
{
    if (descriptor_v2_ != nullptr || module_state_ != nullptr)
        return SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION_V2;
    return descriptor_v1_ == nullptr ? 0u : SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION;
}

const SymmetrixJitCudaPluginV1& CudaPlugin::descriptor_v1() const
{
    if (descriptor_v1_ == nullptr)
        throw std::logic_error("Execution CUDA plugin handle is empty.");
    return *descriptor_v1_;
}

const SymmetrixJitCudaPluginV2& CudaPlugin::descriptor_v2() const
{
    if (module_state_ != nullptr)
        return module_state_->descriptor;
    if (descriptor_v2_ == nullptr)
        throw std::logic_error("Execution CUDA plugin is not ABI version 2.");
    return *descriptor_v2_;
}

std::string_view CudaPlugin::artifact_id() const
{
    return descriptor_v2().artifact_id;
}

std::string_view CudaPlugin::contract_fingerprint() const
{
    return descriptor_v2().contract_fingerprint;
}

std::int32_t CudaPlugin::launch_r1_forward(
    const SymmetrixJitCudaR1ForwardArgsV2* args,
    void* stream,
    std::int32_t persistent_blocks) const
{
    if (module_state_ == nullptr) {
        if (descriptor_v2_ == nullptr || descriptor_v2_->r1_forward_launch == nullptr)
            throw std::logic_error("Execution CUDA R1 forward launcher is unavailable.");
        return descriptor_v2_->r1_forward_launch(args, stream, persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA)
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitCudaR1ForwardArgsV2)
        || args->radial.struct_size < sizeof(SymmetrixJitCudaRadialSplineV2)
        || args->num_nodes < 0 || args->num_nodes > INT32_MAX
        || args->num_edges < 0 || args->num_edges > INT32_MAX
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || persistent_blocks <= 0)
        throw std::invalid_argument("Execution CUDA R1 forward packet is invalid");
    if (args->num_nodes == 0)
        return 0;
    validate_module_context(*module_state_);
    const auto& descriptor = module_state_->descriptor;
    const std::int64_t owners = args->num_nodes * descriptor.channels;
    const std::int32_t blocks = launch_blocks(
        owners, descriptor.forward_threads_per_block, persistent_blocks);
    void* parameters[] = {const_cast<SymmetrixJitCudaR1ForwardArgsV2*>(args)};
    check_driver(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(module_state_->forward),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(descriptor.forward_threads_per_block), 1, 1,
            0, reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution R1 forward)");
    return 0;
#else
    (void)args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution CUDA modules require a CUDA-enabled build.");
#endif
}

std::int32_t CudaPlugin::launch_r1_coordinate_reverse(
    const SymmetrixJitCudaR1SourceArgsV2* source_args,
    const SymmetrixJitCudaR1EdgeArgsV2* edge_args,
    void* stream,
    std::int32_t persistent_blocks) const
{
    if (module_state_ == nullptr) {
        if (descriptor_v2_ == nullptr
            || descriptor_v2_->r1_coordinate_reverse_launch == nullptr)
            throw std::logic_error("Execution CUDA R1 reverse launcher is unavailable.");
        return descriptor_v2_->r1_coordinate_reverse_launch(
            source_args, edge_args, stream, persistent_blocks);
    }
#if defined(KOKKOS_ENABLE_CUDA)
    if (source_args == nullptr || edge_args == nullptr
        || source_args->struct_size < sizeof(SymmetrixJitCudaR1SourceArgsV2)
        || edge_args->struct_size < sizeof(SymmetrixJitCudaR1EdgeArgsV2)
        || source_args->radial.struct_size
            < sizeof(SymmetrixJitCudaRadialSplineV2)
        || edge_args->radial.struct_size
            < sizeof(SymmetrixJitCudaRadialSplineV2)
        || source_args->num_nodes < 0 || source_args->num_nodes > INT32_MAX
        || source_args->num_edges < 0 || source_args->num_edges > INT32_MAX
        || source_args->num_edges != edge_args->num_edges
        || source_args->active_type_count != edge_args->active_type_count
        || source_args->cutoff != edge_args->cutoff
        || !std::isfinite(source_args->cutoff) || source_args->cutoff <= 0.0
        || edge_args->coordinates_are_unit > 1
        || (edge_args->coordinates_are_unit != 0
            ? edge_args->coordinate_scalar_size != sizeof(float)
            : edge_args->coordinate_scalar_size != sizeof(double))
        || source_args->active_type_count == 0 || persistent_blocks <= 0)
        throw std::invalid_argument("Execution CUDA R1 reverse packet is invalid");
    if (source_args->num_nodes == 0 || source_args->num_edges == 0)
        return 0;
    validate_module_context(*module_state_);
    const auto& descriptor = module_state_->descriptor;
    if (module_state_->reverse_fused != nullptr) {
        const std::int32_t reverse_blocks = static_cast<std::int32_t>(
            source_args->num_nodes < persistent_blocks
                ? source_args->num_nodes : persistent_blocks);
        void* parameters[] = {
            const_cast<SymmetrixJitCudaR1SourceArgsV2*>(source_args),
            const_cast<SymmetrixJitCudaR1EdgeArgsV2*>(edge_args)};
        check_driver(
            cuLaunchKernel(
                reinterpret_cast<CUfunction>(module_state_->reverse_fused),
                static_cast<unsigned int>(reverse_blocks), 1, 1,
                static_cast<unsigned int>(
                    module_state_->reverse_threads_per_block), 1, 1,
                0, reinterpret_cast<CUstream>(stream), parameters, nullptr),
            "cuLaunchKernel(Execution R1 fused reverse)");
        return 0;
    }
    const std::int64_t source_owners =
        source_args->num_nodes * descriptor.channels;
    const std::int32_t source_blocks = launch_blocks(
        source_owners, descriptor.source_threads_per_block, persistent_blocks);
    void* source_parameters[] = {
        const_cast<SymmetrixJitCudaR1SourceArgsV2*>(source_args)};
    check_driver(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(module_state_->source),
            static_cast<unsigned int>(source_blocks), 1, 1,
            static_cast<unsigned int>(descriptor.source_threads_per_block), 1, 1,
            0, reinterpret_cast<CUstream>(stream), source_parameters, nullptr),
        "cuLaunchKernel(Execution R1 source reverse)");

    const std::int32_t edge_blocks = launch_blocks(
        edge_args->num_edges, descriptor.edge_threads_per_block, persistent_blocks);
    void* edge_parameters[] = {
        const_cast<SymmetrixJitCudaR1EdgeArgsV2*>(edge_args)};
    check_driver(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(module_state_->edge),
            static_cast<unsigned int>(edge_blocks), 1, 1,
            static_cast<unsigned int>(descriptor.edge_threads_per_block), 1, 1,
            0, reinterpret_cast<CUstream>(stream), edge_parameters, nullptr),
        "cuLaunchKernel(Execution R1 edge reverse)");
    return 0;
#else
    (void)source_args;
    (void)edge_args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution CUDA modules require a CUDA-enabled build.");
#endif
}

bool CudaPlugin::supports_tiled_r1() const noexcept
{
    return module_state_ != nullptr
        && module_state_->tiled_forward != nullptr
        && module_state_->tiled_reverse != nullptr;
}

std::int32_t CudaPlugin::launch_r1_tiled_forward(
    const SymmetrixJitCudaR1TiledForwardArgsV2* args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_CUDA)
    if (!supports_tiled_r1())
        throw std::logic_error("Execution CUDA tiled R1 module is unavailable.");
    const auto channels = module_state_->descriptor.channels;
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitCudaR1TiledForwardArgsV2)
        || args->radial.struct_size < sizeof(SymmetrixJitCudaRadialSplineV2)
        || args->num_nodes < 0 || args->num_edges < 0
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || args->channel_count != 64
        || args->channel_begin%64 != 0
        || args->channel_begin + args->channel_count
            > static_cast<std::uint32_t>(channels)
        || persistent_blocks <= 0)
        throw std::invalid_argument("Execution CUDA tiled R1 forward packet is invalid");
    if (args->num_nodes == 0)
        return 0;
    validate_module_context(*module_state_);
    const std::int64_t owners =
        args->num_nodes * static_cast<std::int64_t>(args->channel_count);
    const auto blocks = launch_blocks(
        owners, module_state_->tiled_forward_threads_per_block,
        persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitCudaR1TiledForwardArgsV2*>(args)};
    check_driver(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(module_state_->tiled_forward),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(
                module_state_->tiled_forward_threads_per_block),
            1, 1, 0, reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution tiled R1 forward)");
    return 0;
#else
    (void)args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution CUDA modules require a CUDA-enabled build.");
#endif
}

std::int32_t CudaPlugin::launch_r1_tiled_reverse(
    const SymmetrixJitCudaR1TiledSourceArgsV2* source_args,
    const SymmetrixJitCudaR1TiledEdgeArgsV2* edge_args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_CUDA)
    if (!supports_tiled_r1())
        throw std::logic_error("Execution CUDA tiled R1 module is unavailable.");
    const auto channels = module_state_->descriptor.channels;
    if (source_args == nullptr || edge_args == nullptr
        || source_args->struct_size
            < sizeof(SymmetrixJitCudaR1TiledSourceArgsV2)
        || edge_args->struct_size < sizeof(SymmetrixJitCudaR1TiledEdgeArgsV2)
        || source_args->radial.struct_size
            < sizeof(SymmetrixJitCudaRadialSplineV2)
        || edge_args->radial.struct_size
            < sizeof(SymmetrixJitCudaRadialSplineV2)
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
        throw std::invalid_argument("Execution CUDA tiled R1 reverse packet is invalid");
    if (source_args->num_nodes == 0 || source_args->num_edges == 0)
        return 0;
    validate_module_context(*module_state_);
    const auto source_owners = source_args->source_owner_count > 0
        ? source_args->source_owner_count : source_args->num_nodes;
    const auto blocks = static_cast<std::int32_t>(
        source_owners < persistent_blocks ? source_owners : persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitCudaR1TiledSourceArgsV2*>(source_args),
        const_cast<SymmetrixJitCudaR1TiledEdgeArgsV2*>(edge_args)};
    check_driver(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(module_state_->tiled_reverse),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(
                module_state_->tiled_reverse_threads_per_block),
            1, 1, 0, reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution tiled R1 reverse)");
    return 0;
#else
    (void)source_args;
    (void)edge_args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution CUDA modules require a CUDA-enabled build.");
#endif
}

std::int32_t CudaPlugin::launch_r1_projected_forward(
    const SymmetrixJitCudaR1ProjectedForwardArgsV2* args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_CUDA)
    if (!supports_projected_r1())
        throw std::logic_error("Execution CUDA projected R1 module is unavailable.");
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitCudaR1ProjectedForwardArgsV2)
        || args->radial.struct_size < sizeof(SymmetrixJitCudaRadialSplineV2)
        || args->num_nodes < 0 || args->num_edges < 0
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || args->projection_weights == nullptr
        || persistent_blocks <= 0)
        throw std::invalid_argument("Execution CUDA projected R1 forward packet is invalid");
    if (args->num_nodes == 0)
        return 0;
    validate_module_context(*module_state_);
    const auto blocks = launch_blocks(
        args->num_nodes, module_state_->projected_threads_per_block,
        persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitCudaR1ProjectedForwardArgsV2*>(args)};
    check_driver(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(module_state_->projected_forward),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(module_state_->projected_threads_per_block),
            1, 1,
            static_cast<unsigned int>(module_state_->projected_shared_memory_bytes),
            reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution projected R1 forward)");
    return 0;
#else
    (void)args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution CUDA modules require a CUDA-enabled build.");
#endif
}

std::int32_t CudaPlugin::launch_r1_projected_reverse(
    const SymmetrixJitCudaR1ProjectedReverseArgsV2* args,
    void* stream, const std::int32_t persistent_blocks) const
{
#if defined(KOKKOS_ENABLE_CUDA)
    if (!supports_projected_r1())
        throw std::logic_error("Execution CUDA projected R1 module is unavailable.");
    if (args == nullptr
        || args->struct_size < sizeof(SymmetrixJitCudaR1ProjectedReverseArgsV2)
        || args->radial.struct_size < sizeof(SymmetrixJitCudaRadialSplineV2)
        || args->num_nodes < 0 || args->num_edges < 0
        || !std::isfinite(args->cutoff) || args->cutoff <= 0.0
        || args->active_type_count == 0 || args->coordinates_are_unit > 1
        || args->projection_weights == nullptr || persistent_blocks <= 0)
        throw std::invalid_argument("Execution CUDA projected R1 reverse packet is invalid");
    if (args->num_nodes == 0 || args->num_edges == 0)
        return 0;
    validate_module_context(*module_state_);
    const auto blocks = launch_blocks(
        args->num_nodes, module_state_->projected_threads_per_block,
        persistent_blocks);
    void* parameters[] = {
        const_cast<SymmetrixJitCudaR1ProjectedReverseArgsV2*>(args)};
    check_driver(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(module_state_->projected_reverse),
            static_cast<unsigned int>(blocks), 1, 1,
            static_cast<unsigned int>(module_state_->projected_threads_per_block),
            1, 1,
            static_cast<unsigned int>(module_state_->projected_shared_memory_bytes),
            reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution projected R1 reverse)");
    return 0;
#else
    (void)args;
    (void)stream;
    (void)persistent_blocks;
    throw std::runtime_error("Execution CUDA modules require a CUDA-enabled build.");
#endif
}

std::int32_t CudaPlugin::r1_reverse_physical_launch_count() const noexcept
{
    if (module_state_ != nullptr)
        return module_state_->reverse_fused == nullptr ? 2 : 1;
    const auto capabilities = descriptor_v2_ == nullptr
        ? 0u : descriptor_v2_->capabilities;
    return (capabilities&SYMMETRIX_JIT_CUDA_R1_FUSED_REVERSE_LAUNCH_V2)
        == 0 ? 2 : 1;
}

bool CudaPlugin::supports_projected_r1() const noexcept
{
    return module_state_ != nullptr
        && module_state_->projected_forward != nullptr
        && module_state_->projected_reverse != nullptr;
}

void CudaPlugin::reset() noexcept
{
    descriptor_v1_ = nullptr;
    descriptor_v2_ = nullptr;
    if (handle_ != nullptr) {
        dlclose(handle_);
        handle_ = nullptr;
    }
#if defined(KOKKOS_ENABLE_CUDA)
    if (module_state_ != nullptr && module_state_->module != nullptr)
        cuModuleUnload(reinterpret_cast<CUmodule>(module_state_->module));
#endif
    module_state_.reset();
    path_.clear();
}

}  // namespace symmetrix::execution
#else
namespace symmetrix::execution {

struct CudaModuleState {};

CudaPlugin CudaPlugin::load(std::string, const CudaPluginExpectation&)
{
    throw std::runtime_error(
        "Execution CUDA plugins require POSIX dlopen/dlsym support.");
}

CudaPlugin CudaPlugin::load_cubin(std::string, const CudaPluginExpectation&)
{
    throw std::runtime_error(
        "Execution CUDA cubins require POSIX CUDA Driver API support.");
}

CudaPlugin::CudaPlugin(
    void* handle,
    const SymmetrixJitCudaPluginV1* descriptor_v1,
    const SymmetrixJitCudaPluginV2* descriptor_v2,
    std::string path) noexcept
    : handle_(handle), descriptor_v1_(descriptor_v1),
      descriptor_v2_(descriptor_v2), path_(std::move(path))
{}

CudaPlugin::CudaPlugin() = default;

CudaPlugin::~CudaPlugin() { reset(); }

CudaPlugin::CudaPlugin(CudaPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_v1_(std::exchange(other.descriptor_v1_, nullptr)),
      descriptor_v2_(std::exchange(other.descriptor_v2_, nullptr)),
      module_state_(std::move(other.module_state_)),
      path_(std::move(other.path_))
{}

CudaPlugin& CudaPlugin::operator=(CudaPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_v1_ = std::exchange(other.descriptor_v1_, nullptr);
        descriptor_v2_ = std::exchange(other.descriptor_v2_, nullptr);
        module_state_ = std::move(other.module_state_);
        path_ = std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitCudaPluginV1& CudaPlugin::descriptor() const
{
    throw std::logic_error("Execution CUDA plugin support is disabled.");
}

bool CudaPlugin::supports_projected_r1() const noexcept { return false; }

bool CudaPlugin::supports_tiled_r1() const noexcept { return false; }

std::int32_t CudaPlugin::launch_r1_tiled_forward(
    const SymmetrixJitCudaR1TiledForwardArgsV2*, void*, std::int32_t) const
{
    throw std::logic_error("Execution CUDA tiled R1 support is disabled.");
}

std::int32_t CudaPlugin::launch_r1_tiled_reverse(
    const SymmetrixJitCudaR1TiledSourceArgsV2*,
    const SymmetrixJitCudaR1TiledEdgeArgsV2*,
    void*, std::int32_t) const
{
    throw std::logic_error("Execution CUDA tiled R1 support is disabled.");
}

std::int32_t CudaPlugin::launch_r1_projected_forward(
    const SymmetrixJitCudaR1ProjectedForwardArgsV2*, void*, std::int32_t) const
{
    throw std::logic_error("Execution CUDA projected R1 support is disabled.");
}

std::int32_t CudaPlugin::launch_r1_projected_reverse(
    const SymmetrixJitCudaR1ProjectedReverseArgsV2*, void*, std::int32_t) const
{
    throw std::logic_error("Execution CUDA projected R1 support is disabled.");
}

std::uint32_t CudaPlugin::abi_version() const noexcept { return 0u; }

const SymmetrixJitCudaPluginV1& CudaPlugin::descriptor_v1() const
{
    throw std::logic_error("Execution CUDA plugin support is disabled.");
}

const SymmetrixJitCudaPluginV2& CudaPlugin::descriptor_v2() const
{
    throw std::logic_error("Execution CUDA plugin support is disabled.");
}

std::int32_t CudaPlugin::launch_r1_forward(
    const SymmetrixJitCudaR1ForwardArgsV2*, void*, std::int32_t) const
{
    throw std::logic_error("Execution CUDA plugin support is disabled.");
}

std::int32_t CudaPlugin::launch_r1_coordinate_reverse(
    const SymmetrixJitCudaR1SourceArgsV2*,
    const SymmetrixJitCudaR1EdgeArgsV2*,
    void*,
    std::int32_t) const
{
    throw std::logic_error("Execution CUDA plugin support is disabled.");
}

void CudaPlugin::reset() noexcept
{
    handle_ = nullptr;
    descriptor_v1_ = nullptr;
    descriptor_v2_ = nullptr;
    module_state_.reset();
    path_.clear();
}

}  // namespace symmetrix::execution
#endif
