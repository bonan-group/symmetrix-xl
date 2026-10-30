#include "mace_jit_operator_module.hpp"
#include "jit_device_plugin_common.hpp"

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <utility>

static_assert(sizeof(SymmetrixJitM0ArgsV1) == 80);
static_assert(sizeof(SymmetrixJitM0ArgsV2) == 112);
static_assert(sizeof(SymmetrixJitM1ArgsV1) == sizeof(SymmetrixJitM0ArgsV1));
static_assert(sizeof(SymmetrixJitR0SplineV1) == 40);
static_assert(sizeof(SymmetrixJitR0ArgsV1) == 264);

namespace symmetrix::execution {

struct OperatorModuleState {
    void* module = nullptr;
    void* owner = nullptr;
    void* forward = nullptr;
    void* reverse = nullptr;
    void* density_prepare = nullptr;
    void* reverse_prepare = nullptr;
    void* coordinate_reverse = nullptr;
    std::string artifact_id;
    std::string structure_fingerprint;
    std::string target;
    std::string r0_schedule;
    std::uint32_t kind = 0;
    std::uint32_t scalar_size = 0;
    std::uint32_t capabilities = 0;
    std::int32_t threads_per_block = 0;
    std::int32_t schedule_revision = 0;
    std::int32_t persistent_blocks_per_compute_unit = 0;
    std::int32_t channel_tile = 0;
    std::int32_t forward_node_tile = 0;
    std::int32_t forward_component_tile = 0;
    std::int32_t reverse_node_tile = 0;
    std::int32_t reverse_component_tile = 0;
    std::int32_t logical_tiles_per_block = 0;
    std::int32_t input_components = 0;
    std::int32_t output_components = 0;
    std::int32_t r0_channels = 0;
    std::int32_t r0_l_max = 0;
    std::int32_t r0_harmonic_count = 0;
    std::int32_t r0_edge_weight_mode = 0;
    std::uint32_t r0_compatibility_version = 0;
    std::int32_t r0_density_extent_multiplier = 0;
    std::int32_t r0_density_owners_per_block = 0;
    std::int32_t r0_density_threads_per_block = 0;
    std::int32_t r0_density_persistent = 0;
    std::int32_t r0_forward_extent_multiplier = 0;
    std::int32_t r0_forward_owners_per_block = 0;
    std::int32_t r0_forward_threads_per_block = 0;
    std::int32_t r0_forward_persistent = 0;
    std::int32_t r0_reverse_prepare_extent_multiplier = 0;
    std::int32_t r0_reverse_prepare_owners_per_block = 0;
    std::int32_t r0_reverse_prepare_threads_per_block = 0;
    std::int32_t r0_reverse_prepare_persistent = 0;
    std::int32_t r0_coordinate_reverse_extent_multiplier = 0;
    std::int32_t r0_coordinate_reverse_owners_per_block = 0;
    std::int32_t r0_coordinate_reverse_threads_per_block = 0;
    std::int32_t r0_coordinate_reverse_persistent = 0;
};

namespace {

std::runtime_error module_error(
    const std::string& path, const std::string& message)
{
    return detail::PluginLoaderContext("operator module", path).error(message);
}

void validate_expectation(const OperatorModuleExpectation& expectation)
{
    if (expectation.kind != SYMMETRIX_JIT_OPERATOR_KIND_M0_V1
        && expectation.kind != SYMMETRIX_JIT_OPERATOR_KIND_M1_V1
        && expectation.kind != SYMMETRIX_JIT_OPERATOR_KIND_R0_V1)
        throw std::invalid_argument("Execution operator module kind is invalid.");
    if (expectation.artifact_id.empty()
        || expectation.structure_fingerprint.empty()
        || expectation.target.empty())
        throw std::invalid_argument(
            "Execution operator module requires artifact, structure, and target identity.");
    if (expectation.scalar_size != sizeof(float)
        && expectation.scalar_size != sizeof(double))
        throw std::invalid_argument(
            "Execution operator module scalar width is invalid.");
    if (expectation.required_capabilities == 0
        || expectation.input_components <= 0
        || expectation.output_components <= 0
        || expectation.correlation <= 0
        || expectation.term_count <= 0
        || expectation.max_threads_per_block <= 0
        || expectation.expected_threads_per_block <= 0
        || expectation.expected_threads_per_block
            > expectation.max_threads_per_block)
        throw std::invalid_argument(
            "Execution operator module expectation has invalid extents.");
    if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_R0_V1
        && (expectation.schedule_revision <= 0
            || expectation.persistent_blocks_per_compute_unit <= 0
            || expectation.persistent_blocks_per_compute_unit > 32
            || expectation.r0_channels <= 0 || expectation.r0_l_max < 0
            || expectation.r0_density_extent_multiplier <= 0
            || expectation.r0_density_owners_per_block <= 0
            || expectation.r0_density_threads_per_block <= 0
            || expectation.r0_density_threads_per_block > expectation.max_threads_per_block
            || expectation.r0_density_persistent < 0
            || expectation.r0_density_persistent > 1
            || expectation.r0_forward_extent_multiplier <= 0
            || expectation.r0_forward_owners_per_block <= 0
            || expectation.r0_forward_threads_per_block <= 0
            || expectation.r0_forward_threads_per_block > expectation.max_threads_per_block
            || expectation.r0_forward_persistent < 0
            || expectation.r0_forward_persistent > 1
            || expectation.r0_reverse_prepare_extent_multiplier <= 0
            || expectation.r0_reverse_prepare_owners_per_block <= 0
            || expectation.r0_reverse_prepare_threads_per_block <= 0
            || expectation.r0_reverse_prepare_threads_per_block > expectation.max_threads_per_block
            || expectation.r0_reverse_prepare_persistent < 0
            || expectation.r0_reverse_prepare_persistent > 1
            || expectation.r0_coordinate_reverse_extent_multiplier <= 0
            || expectation.r0_coordinate_reverse_owners_per_block <= 0
            || expectation.r0_coordinate_reverse_threads_per_block <= 0
            || expectation.r0_coordinate_reverse_threads_per_block > expectation.max_threads_per_block
            || expectation.r0_coordinate_reverse_persistent < 0
            || expectation.r0_coordinate_reverse_persistent > 1
            || expectation.r0_descriptor.artifact_id.empty()
            || expectation.r0_descriptor.structure_fingerprint.empty()
            || expectation.r0_descriptor.target.empty()
            || expectation.r0_descriptor.schedule.empty()
            || expectation.r0_descriptor.harmonic_count <= 0
            || expectation.r0_descriptor.compatibility_version == 0))
        throw std::invalid_argument(
            "Execution R0 module expectation has invalid schedule geometry.");
    if (expectation.kind != SYMMETRIX_JIT_OPERATOR_KIND_R0_V1
        && (expectation.schedule_revision <= 0 || expectation.channel_tile <= 0
            || expectation.forward_node_tile <= 0
            || expectation.forward_component_tile <= 0
            || expectation.reverse_node_tile <= 0
            || expectation.reverse_component_tile <= 0
            || expectation.logical_tiles_per_block <= 0))
        throw std::invalid_argument(
            "Execution symmetric module expectation has invalid launch geometry.");
}

#if defined(KOKKOS_ENABLE_CUDA)

void check_backend(CUresult status, const std::string& operation)
{
    if (status == CUDA_SUCCESS)
        return;
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    throw std::runtime_error(
        operation+" failed: "+(name == nullptr ? "CUDA_ERROR_UNKNOWN" : name)
        +(message == nullptr ? std::string() : " ("+std::string(message)+")"));
}

template<class Value>
Value read_value(const std::string& path, CUmodule module, const char* symbol)
{
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto status = cuModuleGetGlobal(&address, &size, module, symbol);
    if (status != CUDA_SUCCESS)
        throw module_error(path, std::string("missing identity symbol ")+symbol);
    if (size != sizeof(Value))
        throw module_error(path, std::string("invalid identity width for ")+symbol);
    Value value {};
    check_backend(cuMemcpyDtoH(&value, address, size), "cuMemcpyDtoH(identity)");
    return value;
}

std::string read_text(const std::string& path, CUmodule module, const char* symbol)
{
    CUdeviceptr address = 0;
    std::size_t size = 0;
    const auto status = cuModuleGetGlobal(&address, &size, module, symbol);
    if (status != CUDA_SUCCESS)
        throw module_error(path, std::string("missing identity symbol ")+symbol);
    if (size < 2 || size > 512)
        throw module_error(path, std::string("invalid identity text for ")+symbol);
    std::string value(size, '\0');
    check_backend(
        cuMemcpyDtoH(value.data(), address, size), "cuMemcpyDtoH(identity text)");
    if (value.back() != '\0')
        throw module_error(path, std::string("unterminated identity text for ")+symbol);
    value.pop_back();
    return value;
}

void validate_owner(const OperatorModuleState& state)
{
    CUcontext current = nullptr;
    check_backend(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current == nullptr || reinterpret_cast<void*>(current) != state.owner)
        throw std::runtime_error(
            "The active CUDA context does not own the Execution operator module");
}

#elif defined(KOKKOS_ENABLE_HIP)

void check_backend(hipError_t status, const std::string& operation)
{
    if (status != hipSuccess)
        throw std::runtime_error(
            operation+" failed: "+hipGetErrorName(status)+" ("
            +hipGetErrorString(status)+")");
}

template<class Value>
Value read_value(const std::string& path, hipModule_t module, const char* symbol)
{
    hipDeviceptr_t address {};
    std::size_t size = 0;
    const auto status = hipModuleGetGlobal(&address, &size, module, symbol);
    if (status != hipSuccess)
        throw module_error(path, std::string("missing identity symbol ")+symbol);
    if (size != sizeof(Value))
        throw module_error(path, std::string("invalid identity width for ")+symbol);
    Value value {};
    check_backend(hipMemcpyDtoH(&value, address, size), "hipMemcpyDtoH(identity)");
    return value;
}

std::string read_text(
    const std::string& path, hipModule_t module, const char* symbol)
{
    hipDeviceptr_t address {};
    std::size_t size = 0;
    const auto status = hipModuleGetGlobal(&address, &size, module, symbol);
    if (status != hipSuccess)
        throw module_error(path, std::string("missing identity symbol ")+symbol);
    if (size < 2 || size > 512)
        throw module_error(path, std::string("invalid identity text for ")+symbol);
    std::string value(size, '\0');
    check_backend(
        hipMemcpyDtoH(value.data(), address, size), "hipMemcpyDtoH(identity text)");
    if (value.back() != '\0')
        throw module_error(path, std::string("unterminated identity text for ")+symbol);
    value.pop_back();
    return value;
}

void validate_owner(const OperatorModuleState& state)
{
    int current = -1;
    check_backend(hipGetDevice(&current), "hipGetDevice");
    if (reinterpret_cast<void*>(static_cast<std::intptr_t>(current)) != state.owner)
        throw std::runtime_error(
            "The active HIP device does not own the Execution operator module");
}

#else

void validate_owner(const OperatorModuleState&)
{
    throw std::runtime_error(
        "Execution operator modules require a CUDA- or HIP-enabled build.");
}

#endif

void validate_identity(
    const std::string& path,
    OperatorModuleState& state,
    const OperatorModuleExpectation& expectation,
    const std::uint32_t abi_version,
    const std::uint32_t scalar_kind,
    const std::uint32_t scalar_size,
    const std::int32_t input_components,
    const std::int32_t output_components,
    const std::int32_t correlation,
    const std::int32_t term_count)
{
    const std::uint32_t expected_abi_version =
        expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_R0_V1
            ? SYMMETRIX_JIT_OPERATOR_R0_ABI_VERSION_V4
            : SYMMETRIX_JIT_OPERATOR_ABI_VERSION_V1;
    if (abi_version != expected_abi_version
        || state.kind != expectation.kind
        || scalar_kind != expectation.scalar_kind
        || scalar_size != expectation.scalar_size
        || input_components != expectation.input_components
        || output_components != expectation.output_components
        || correlation != expectation.correlation
        || term_count != expectation.term_count)
        throw module_error(path, "operator ABI or topology identity does not match");
    if (state.artifact_id != expectation.artifact_id
        || state.structure_fingerprint != expectation.structure_fingerprint
        || state.target != expectation.target)
        throw module_error(path, "artifact, structure, or target identity does not match");
    if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_R0_V1) {
        if ((state.capabilities&expectation.required_capabilities)
                != expectation.required_capabilities)
            throw module_error(path, "required semantic capabilities are missing");
    } else if (state.capabilities != expectation.required_capabilities) {
        throw module_error(path, "symmetric semantic capabilities do not match");
    }
    if (state.threads_per_block != expectation.expected_threads_per_block
        || state.threads_per_block > expectation.max_threads_per_block)
        throw module_error(path, "thread-block size is invalid for this device");
    if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_R0_V1
        && (state.schedule_revision != expectation.schedule_revision
            || state.persistent_blocks_per_compute_unit
                != expectation.persistent_blocks_per_compute_unit
            || state.r0_channels != expectation.r0_channels
            || state.r0_l_max != expectation.r0_l_max
            || state.r0_density_extent_multiplier
                != expectation.r0_density_extent_multiplier
            || state.r0_density_owners_per_block
                != expectation.r0_density_owners_per_block
            || state.r0_density_threads_per_block
                != expectation.r0_density_threads_per_block
            || state.r0_density_persistent
                != expectation.r0_density_persistent
            || state.r0_forward_extent_multiplier
                != expectation.r0_forward_extent_multiplier
            || state.r0_forward_owners_per_block
                != expectation.r0_forward_owners_per_block
            || state.r0_forward_threads_per_block
                != expectation.r0_forward_threads_per_block
            || state.r0_forward_persistent
                != expectation.r0_forward_persistent
            || state.r0_reverse_prepare_extent_multiplier
                != expectation.r0_reverse_prepare_extent_multiplier
            || state.r0_reverse_prepare_owners_per_block
                != expectation.r0_reverse_prepare_owners_per_block
            || state.r0_reverse_prepare_threads_per_block
                != expectation.r0_reverse_prepare_threads_per_block
            || state.r0_reverse_prepare_persistent
                != expectation.r0_reverse_prepare_persistent
            || state.r0_coordinate_reverse_extent_multiplier
                != expectation.r0_coordinate_reverse_extent_multiplier
            || state.r0_coordinate_reverse_owners_per_block
                != expectation.r0_coordinate_reverse_owners_per_block
            || state.r0_coordinate_reverse_threads_per_block
                != expectation.r0_coordinate_reverse_threads_per_block
            || state.r0_coordinate_reverse_persistent
                != expectation.r0_coordinate_reverse_persistent))
        throw module_error(path, "R0 schedule profile or launch geometry does not match");
    if (expectation.kind != SYMMETRIX_JIT_OPERATOR_KIND_R0_V1
        && (state.schedule_revision != expectation.schedule_revision
            || state.channel_tile != expectation.channel_tile
            || state.forward_node_tile != expectation.forward_node_tile
            || state.forward_component_tile != expectation.forward_component_tile
            || state.reverse_node_tile != expectation.reverse_node_tile
            || state.reverse_component_tile != expectation.reverse_component_tile
            || state.logical_tiles_per_block != expectation.logical_tiles_per_block))
        throw module_error(path, "symmetric schedule or launch geometry does not match");
    if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_R0_V1) {
        std::uint32_t logical_capabilities = 0;
        if (state.capabilities&SYMMETRIX_JIT_OPERATOR_R0_DENSITY_PREPARE_V1)
            logical_capabilities |= r0_density_prepare_capability;
        if (state.capabilities&SYMMETRIX_JIT_OPERATOR_R0_FORWARD_V1)
            logical_capabilities |= r0_forward_capability;
        if (state.capabilities&SYMMETRIX_JIT_OPERATOR_R0_REVERSE_PREPARE_V1)
            logical_capabilities |= r0_reverse_prepare_capability;
        if (state.capabilities&SYMMETRIX_JIT_OPERATOR_R0_COORDINATE_REVERSE_V1)
            logical_capabilities |= r0_coordinate_reverse_capability;
        if (state.capabilities&SYMMETRIX_JIT_OPERATOR_R0_COMPACT_GEOMETRY_V1)
            logical_capabilities |= r0_compact_geometry_capability;
        if (state.capabilities&SYMMETRIX_JIT_OPERATOR_R0_PRECOMPUTED_SCALE_V1)
            logical_capabilities |= r0_precomputed_scale_capability;
        if (state.capabilities&SYMMETRIX_JIT_OPERATOR_R0_RECEIVER_BATCH_V1)
            logical_capabilities |= r0_receiver_batch_capability;
        const R0LogicalDescriptor actual{
            state.artifact_id, state.structure_fingerprint, state.target,
            state.r0_schedule, scalar_kind, scalar_size, logical_capabilities,
            state.r0_channels, state.r0_l_max, state.r0_harmonic_count,
            state.r0_edge_weight_mode, state.schedule_revision,
            state.r0_compatibility_version};
        if (const auto field = r0_descriptor_mismatch(
                actual, expectation.r0_descriptor); !field.empty())
            throw module_error(
                path, "logical R0 descriptor field "+std::string(field)
                    +" does not match");
    }
}

void validate_kind_identity(
    const std::string& path,
    const OperatorModuleState& state,
    const OperatorModuleExpectation& expectation)
{
    if (state.kind != expectation.kind)
        throw module_error(path, "operator kind identity does not match");
}

} // namespace

OperatorModule OperatorModule::load(
    std::string path, const OperatorModuleExpectation& expectation)
{
    if (path.empty())
        throw std::invalid_argument("Execution operator module path is empty.");
    validate_expectation(expectation);
#if defined(KOKKOS_ENABLE_CUDA)
    auto bytes =
        detail::PluginLoaderContext("CUDA", path).read_elf_binary("cubin");
    CUcontext context = nullptr;
    check_backend(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (context == nullptr)
        throw module_error(path, "no CUDA context is active");
    CUmodule module = nullptr;
    check_backend(cuModuleLoadData(&module, bytes.data()), "cuModuleLoadData");
    try {
        auto state = std::make_unique<OperatorModuleState>();
        state->module = reinterpret_cast<void*>(module);
        state->owner = reinterpret_cast<void*>(context);
        state->scalar_size = expectation.scalar_size;
        state->input_components = expectation.input_components;
        state->output_components = expectation.output_components;
        state->kind = read_value<std::uint32_t>(
            path, module, "symmetrix_operator_kind_v1");
        state->capabilities = read_value<std::uint32_t>(
            path, module, "symmetrix_operator_capabilities_v1");
        state->threads_per_block = read_value<std::int32_t>(
            path, module, "symmetrix_operator_threads_v1");
        state->artifact_id = read_text(
            path, module, "symmetrix_operator_artifact_id_v1");
        state->structure_fingerprint = read_text(
            path, module, "symmetrix_operator_structure_fingerprint_v1");
        state->target = read_text(
            path, module, "symmetrix_operator_target_v1");
        // Reject a module of the wrong operator family before reading
        // family-specific identity symbols that it is not required to export.
        validate_kind_identity(path, *state, expectation);
        if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_R0_V1) {
            state->schedule_revision = read_value<std::int32_t>(
                path, module, "symmetrix_operator_schedule_revision_v1");
            state->persistent_blocks_per_compute_unit = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_persistent_blocks_per_compute_unit_v1");
            state->r0_channels = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_channels_v1");
            state->r0_l_max = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_l_max_v1");
            state->r0_harmonic_count = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_harmonic_count_v1");
            state->r0_edge_weight_mode = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_edge_weight_mode_v1");
            state->r0_compatibility_version = read_value<std::uint32_t>(
                path, module,
                "symmetrix_operator_r0_compatibility_version_v1");
            state->r0_schedule = read_text(
                path, module, "symmetrix_operator_r0_schedule_v1");
            state->r0_density_extent_multiplier = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_density_extent_multiplier_v1");
            state->r0_density_owners_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_density_owners_per_block_v1");
            state->r0_density_threads_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_density_threads_per_block_v1");
            state->r0_density_persistent = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_density_persistent_v1");
            state->r0_forward_extent_multiplier = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_forward_extent_multiplier_v1");
            state->r0_forward_owners_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_forward_owners_per_block_v1");
            state->r0_forward_threads_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_forward_threads_per_block_v1");
            state->r0_forward_persistent = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_forward_persistent_v1");
            state->r0_reverse_prepare_extent_multiplier =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_reverse_prepare_extent_multiplier_v1");
            state->r0_reverse_prepare_owners_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_reverse_prepare_owners_per_block_v1");
            state->r0_reverse_prepare_threads_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_reverse_prepare_threads_per_block_v1");
            state->r0_reverse_prepare_persistent = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_reverse_prepare_persistent_v1");
            state->r0_coordinate_reverse_extent_multiplier =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_coordinate_reverse_extent_multiplier_v1");
            state->r0_coordinate_reverse_owners_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_coordinate_reverse_owners_per_block_v1");
            state->r0_coordinate_reverse_threads_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_coordinate_reverse_threads_per_block_v1");
            state->r0_coordinate_reverse_persistent = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_coordinate_reverse_persistent_v1");
        } else {
            state->schedule_revision = read_value<std::int32_t>(
                path, module, "symmetrix_operator_schedule_revision_v1");
            state->channel_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_channel_tile_v1");
            state->forward_node_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_forward_node_tile_v1");
            state->forward_component_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_forward_component_tile_v1");
            state->reverse_node_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_reverse_node_tile_v1");
            state->reverse_component_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_reverse_component_tile_v1");
            state->logical_tiles_per_block = read_value<std::int32_t>(
                path, module, "symmetrix_operator_logical_tiles_per_block_v1");
        }
        validate_identity(
            path, *state, expectation,
            read_value<std::uint32_t>(
                path, module, "symmetrix_operator_abi_version_v1"),
            read_value<std::uint32_t>(
                path, module, "symmetrix_operator_scalar_kind_v1"),
            read_value<std::uint32_t>(
                path, module, "symmetrix_operator_scalar_size_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_input_components_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_output_components_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_correlation_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_term_count_v1"));
        CUfunction forward = nullptr;
        if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_M0_V1
            || expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_M1_V1) {
            CUfunction reverse = nullptr;
            const bool is_m1 =
                expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_M1_V1;
            check_backend(
                cuModuleGetFunction(
                    &forward, module,
                    is_m1 ? "symmetrix_m1_forward_v1"
                          : "symmetrix_m0_forward_v1"),
                is_m1 ? "cuModuleGetFunction(M1 forward)"
                      : "cuModuleGetFunction(M0 forward)");
            check_backend(
                cuModuleGetFunction(
                    &reverse, module,
                    is_m1 ? "symmetrix_m1_reverse_v1"
                          : "symmetrix_m0_reverse_v1"),
                is_m1 ? "cuModuleGetFunction(M1 reverse)"
                      : "cuModuleGetFunction(M0 reverse)");
            state->reverse = reinterpret_cast<void*>(reverse);
        } else {
            CUfunction density_prepare = nullptr;
            CUfunction reverse_prepare = nullptr;
            CUfunction coordinate_reverse = nullptr;
            check_backend(cuModuleGetFunction(
                &density_prepare, module, "symmetrix_r0_density_prepare_v1"),
                "cuModuleGetFunction(R0 density prepare)");
            check_backend(cuModuleGetFunction(
                &forward, module, "symmetrix_r0_forward_v1"),
                "cuModuleGetFunction(R0 forward)");
            check_backend(cuModuleGetFunction(
                &reverse_prepare, module, "symmetrix_r0_reverse_prepare_v1"),
                "cuModuleGetFunction(R0 reverse prepare)");
            check_backend(cuModuleGetFunction(
                &coordinate_reverse, module,
                "symmetrix_r0_coordinate_reverse_v1"),
                "cuModuleGetFunction(R0 coordinate reverse)");
            state->density_prepare = reinterpret_cast<void*>(density_prepare);
            state->reverse_prepare = reinterpret_cast<void*>(reverse_prepare);
            state->coordinate_reverse = reinterpret_cast<void*>(coordinate_reverse);
        }
        state->forward = reinterpret_cast<void*>(forward);
        return OperatorModule(std::move(state), std::move(path));
    } catch (...) {
        cuModuleUnload(module);
        throw;
    }
#elif defined(KOKKOS_ENABLE_HIP)
    auto bytes = detail::PluginLoaderContext("HIP", path).read_elf_binary(
        "code object");
    int device = -1;
    check_backend(hipGetDevice(&device), "hipGetDevice");
    hipModule_t module = nullptr;
    check_backend(hipModuleLoadData(&module, bytes.data()), "hipModuleLoadData");
    try {
        auto state = std::make_unique<OperatorModuleState>();
        state->module = reinterpret_cast<void*>(module);
        state->owner = reinterpret_cast<void*>(static_cast<std::intptr_t>(device));
        state->scalar_size = expectation.scalar_size;
        state->input_components = expectation.input_components;
        state->output_components = expectation.output_components;
        state->kind = read_value<std::uint32_t>(
            path, module, "symmetrix_operator_kind_v1");
        state->capabilities = read_value<std::uint32_t>(
            path, module, "symmetrix_operator_capabilities_v1");
        state->threads_per_block = read_value<std::int32_t>(
            path, module, "symmetrix_operator_threads_v1");
        state->artifact_id = read_text(
            path, module, "symmetrix_operator_artifact_id_v1");
        state->structure_fingerprint = read_text(
            path, module, "symmetrix_operator_structure_fingerprint_v1");
        state->target = read_text(
            path, module, "symmetrix_operator_target_v1");
        // Reject a module of the wrong operator family before reading
        // family-specific identity symbols that it is not required to export.
        validate_kind_identity(path, *state, expectation);
        if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_R0_V1) {
            state->schedule_revision = read_value<std::int32_t>(
                path, module, "symmetrix_operator_schedule_revision_v1");
            state->persistent_blocks_per_compute_unit = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_persistent_blocks_per_compute_unit_v1");
            state->r0_channels = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_channels_v1");
            state->r0_l_max = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_l_max_v1");
            state->r0_harmonic_count = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_harmonic_count_v1");
            state->r0_edge_weight_mode = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_edge_weight_mode_v1");
            state->r0_compatibility_version = read_value<std::uint32_t>(
                path, module,
                "symmetrix_operator_r0_compatibility_version_v1");
            state->r0_schedule = read_text(
                path, module, "symmetrix_operator_r0_schedule_v1");
            state->r0_density_extent_multiplier = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_density_extent_multiplier_v1");
            state->r0_density_owners_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_density_owners_per_block_v1");
            state->r0_density_threads_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_density_threads_per_block_v1");
            state->r0_density_persistent = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_density_persistent_v1");
            state->r0_forward_extent_multiplier = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_forward_extent_multiplier_v1");
            state->r0_forward_owners_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_forward_owners_per_block_v1");
            state->r0_forward_threads_per_block = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_forward_threads_per_block_v1");
            state->r0_forward_persistent = read_value<std::int32_t>(
                path, module, "symmetrix_operator_r0_forward_persistent_v1");
            state->r0_reverse_prepare_extent_multiplier =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_reverse_prepare_extent_multiplier_v1");
            state->r0_reverse_prepare_owners_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_reverse_prepare_owners_per_block_v1");
            state->r0_reverse_prepare_threads_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_reverse_prepare_threads_per_block_v1");
            state->r0_reverse_prepare_persistent = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_reverse_prepare_persistent_v1");
            state->r0_coordinate_reverse_extent_multiplier =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_coordinate_reverse_extent_multiplier_v1");
            state->r0_coordinate_reverse_owners_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_coordinate_reverse_owners_per_block_v1");
            state->r0_coordinate_reverse_threads_per_block =
                read_value<std::int32_t>(
                    path, module,
                    "symmetrix_operator_r0_coordinate_reverse_threads_per_block_v1");
            state->r0_coordinate_reverse_persistent = read_value<std::int32_t>(
                path, module,
                "symmetrix_operator_r0_coordinate_reverse_persistent_v1");
        } else {
            state->schedule_revision = read_value<std::int32_t>(
                path, module, "symmetrix_operator_schedule_revision_v1");
            state->channel_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_channel_tile_v1");
            state->forward_node_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_forward_node_tile_v1");
            state->forward_component_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_forward_component_tile_v1");
            state->reverse_node_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_reverse_node_tile_v1");
            state->reverse_component_tile = read_value<std::int32_t>(
                path, module, "symmetrix_operator_reverse_component_tile_v1");
            state->logical_tiles_per_block = read_value<std::int32_t>(
                path, module, "symmetrix_operator_logical_tiles_per_block_v1");
        }
        validate_identity(
            path, *state, expectation,
            read_value<std::uint32_t>(
                path, module, "symmetrix_operator_abi_version_v1"),
            read_value<std::uint32_t>(
                path, module, "symmetrix_operator_scalar_kind_v1"),
            read_value<std::uint32_t>(
                path, module, "symmetrix_operator_scalar_size_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_input_components_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_output_components_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_correlation_v1"),
            read_value<std::int32_t>(
                path, module, "symmetrix_operator_term_count_v1"));
        hipFunction_t forward = nullptr;
        if (expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_M0_V1
            || expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_M1_V1) {
            hipFunction_t reverse = nullptr;
            const bool is_m1 =
                expectation.kind == SYMMETRIX_JIT_OPERATOR_KIND_M1_V1;
            check_backend(
                hipModuleGetFunction(
                    &forward, module,
                    is_m1 ? "symmetrix_m1_forward_v1"
                          : "symmetrix_m0_forward_v1"),
                is_m1 ? "hipModuleGetFunction(M1 forward)"
                      : "hipModuleGetFunction(M0 forward)");
            check_backend(
                hipModuleGetFunction(
                    &reverse, module,
                    is_m1 ? "symmetrix_m1_reverse_v1"
                          : "symmetrix_m0_reverse_v1"),
                is_m1 ? "hipModuleGetFunction(M1 reverse)"
                      : "hipModuleGetFunction(M0 reverse)");
            state->reverse = reinterpret_cast<void*>(reverse);
        } else {
            hipFunction_t density_prepare = nullptr;
            hipFunction_t reverse_prepare = nullptr;
            hipFunction_t coordinate_reverse = nullptr;
            check_backend(hipModuleGetFunction(
                &density_prepare, module, "symmetrix_r0_density_prepare_v1"),
                "hipModuleGetFunction(R0 density prepare)");
            check_backend(hipModuleGetFunction(
                &forward, module, "symmetrix_r0_forward_v1"),
                "hipModuleGetFunction(R0 forward)");
            check_backend(hipModuleGetFunction(
                &reverse_prepare, module, "symmetrix_r0_reverse_prepare_v1"),
                "hipModuleGetFunction(R0 reverse prepare)");
            check_backend(hipModuleGetFunction(
                &coordinate_reverse, module,
                "symmetrix_r0_coordinate_reverse_v1"),
                "hipModuleGetFunction(R0 coordinate reverse)");
            state->density_prepare = reinterpret_cast<void*>(density_prepare);
            state->reverse_prepare = reinterpret_cast<void*>(reverse_prepare);
            state->coordinate_reverse = reinterpret_cast<void*>(coordinate_reverse);
        }
        state->forward = reinterpret_cast<void*>(forward);
        return OperatorModule(std::move(state), std::move(path));
    } catch (...) {
        hipModuleUnload(module);
        throw;
    }
#else
    (void)path;
    throw std::invalid_argument(
        "Execution operator modules require a CUDA- or HIP-enabled build.");
#endif
}

OperatorModule::OperatorModule() = default;
OperatorModule::OperatorModule(
    std::unique_ptr<OperatorModuleState> state, std::string path) noexcept
    : state_(std::move(state)), path_(std::move(path))
{}
OperatorModule::~OperatorModule() { reset(); }
OperatorModule::OperatorModule(OperatorModule&& other) noexcept
    : state_(std::move(other.state_)), path_(std::move(other.path_))
{}
OperatorModule& OperatorModule::operator=(OperatorModule&& other) noexcept
{
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        path_ = std::move(other.path_);
    }
    return *this;
}

void OperatorModule::reset() noexcept
{
    if (state_ != nullptr && state_->module != nullptr) {
#if defined(KOKKOS_ENABLE_CUDA)
        cuModuleUnload(reinterpret_cast<CUmodule>(state_->module));
#elif defined(KOKKOS_ENABLE_HIP)
        hipModuleUnload(reinterpret_cast<hipModule_t>(state_->module));
#endif
    }
    state_.reset();
    path_.clear();
}

std::string_view OperatorModule::artifact_id() const noexcept
{
    return state_ == nullptr ? std::string_view() : state_->artifact_id;
}
std::string_view OperatorModule::structure_fingerprint() const noexcept
{
    return state_ == nullptr ? std::string_view() : state_->structure_fingerprint;
}
std::string_view OperatorModule::target() const noexcept
{
    return state_ == nullptr ? std::string_view() : state_->target;
}
std::uint32_t OperatorModule::kind() const noexcept
{
    return state_ == nullptr ? 0u : state_->kind;
}
std::uint32_t OperatorModule::capabilities() const noexcept
{
    return state_ == nullptr ? 0u : state_->capabilities;
}
std::int32_t OperatorModule::threads_per_block() const noexcept
{
    return state_ == nullptr ? 0 : state_->threads_per_block;
}

namespace {

void validate_symmetric_launch(
    const OperatorModuleState* state,
    const SymmetrixJitM0ArgsV1& args,
    const std::int32_t persistent_blocks,
    const std::uint32_t expected_kind)
{
    if (state == nullptr || state->kind != expected_kind)
        throw std::logic_error("Execution symmetric module is unavailable.");
    const bool valid_reserved = expected_kind == SYMMETRIX_JIT_OPERATOR_KIND_M1_V1
        ? (args.reserved
            &~SYMMETRIX_JIT_OPERATOR_M1_ACCUMULATE_INPUT_ADJOINT_V1) == 0
        : args.reserved == 0;
    if (args.struct_size < sizeof(SymmetrixJitM0ArgsV1) || !valid_reserved
        || args.num_nodes < 0 || args.num_nodes > INT32_MAX
        || args.channels <= 0 || args.capture_input_scale_adjoint < 0
        || args.capture_input_scale_adjoint > 1 || persistent_blocks <= 0)
        throw std::invalid_argument("Execution symmetric launch packet is invalid.");
}

SymmetrixJitM0ArgsV2 packed_symmetric_args(
    const SymmetrixJitM0ArgsV1& args, const OperatorModuleState& state)
{
    return {args,
        static_cast<std::int64_t>(state.input_components)*args.channels,
        static_cast<std::int64_t>(state.output_components)*args.channels,
        static_cast<std::int64_t>(state.output_components)*args.channels,
        static_cast<std::int64_t>(state.input_components)*args.channels};
}

void validate_strided_launch(
    const OperatorModuleState& state, const SymmetrixJitM0ArgsV2& args)
{
    if(args.packed.struct_size<sizeof(SymmetrixJitM0ArgsV1)
        ||args.input_node_stride
            <static_cast<std::int64_t>(state.input_components)*args.packed.channels
        ||args.output_node_stride
            <static_cast<std::int64_t>(state.output_components)*args.packed.channels
        ||args.output_adjoint_node_stride
            <static_cast<std::int64_t>(state.output_components)*args.packed.channels
        ||args.input_adjoint_node_stride
            <static_cast<std::int64_t>(state.input_components)*args.packed.channels)
        throw std::invalid_argument("Execution symmetric row strides are invalid.");
}

void validate_symmetric_aliasing(
    const SymmetrixJitM0ArgsV1& args, const bool reverse)
{
    const bool aliases = reverse
        ? args.input_adjoint == args.input : args.output == args.input;
    if(aliases)
        throw std::invalid_argument(
            "Execution symmetric module does not support aliased input/output buffers.");
}

bool validate_symmetric_product_launch(
    const OperatorModuleState* state, const SymmetrixJitM0ArgsV2& strided_args,
    const std::int32_t persistent_blocks, const std::uint32_t kind,
    const bool reverse)
{
    const auto& args = strided_args.packed;
    validate_symmetric_launch(state, args, persistent_blocks, kind);
    if (args.num_nodes == 0)
        return false;
    validate_symmetric_aliasing(args, reverse);
    validate_strided_launch(*state, strided_args);
    validate_owner(*state);
    return true;
}

std::int32_t symmetric_launch_blocks(
    const OperatorModuleState& state, const SymmetrixJitM0ArgsV1& args,
    const bool reverse, const std::int32_t persistent_blocks)
{
    const std::int64_t node_tile = reverse
        ? state.reverse_node_tile : state.forward_node_tile;
    const std::int64_t component_tile = reverse
        ? state.reverse_component_tile : state.forward_component_tile;
    const std::int64_t components = reverse
        ? state.input_components : state.output_components;
    const std::int64_t channel_tiles =
        (static_cast<std::int64_t>(args.channels)+state.channel_tile-1)
        /state.channel_tile;
    const std::int64_t node_tiles = (args.num_nodes+node_tile-1)/node_tile;
    const std::int64_t component_tiles =
        (components+component_tile-1)/component_tile;
    const std::int64_t logical_tiles =
        channel_tiles*node_tiles*component_tiles;
    return detail::launch_blocks(
        logical_tiles, state.logical_tiles_per_block, persistent_blocks);
}

} // namespace

void OperatorModule::validate_m0_launch(
    const SymmetrixJitM0ArgsV1& args, const bool reverse,
    const std::int32_t persistent_blocks) const
{
    if (state_ == nullptr)
        throw std::logic_error("Execution symmetric module is unavailable.");
    (void)validate_symmetric_product_launch(
        state_.get(), packed_symmetric_args(args, *state_), persistent_blocks,
        SYMMETRIX_JIT_OPERATOR_KIND_M0_V1, reverse);
}

void OperatorModule::validate_m1_launch(
    const SymmetrixJitM1ArgsV1& args, const bool reverse,
    const std::int32_t persistent_blocks) const
{
    if (state_ == nullptr)
        throw std::logic_error("Execution symmetric module is unavailable.");
    (void)validate_symmetric_product_launch(
        state_.get(), packed_symmetric_args(args, *state_), persistent_blocks,
        SYMMETRIX_JIT_OPERATOR_KIND_M1_V1, reverse);
}

void OperatorModule::launch_m0_forward(
    const SymmetrixJitM0ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    if(state_==nullptr)throw std::logic_error("Execution symmetric module is unavailable.");
    launch_m0_forward_strided(
        packed_symmetric_args(args,*state_),stream,persistent_blocks);
}

void OperatorModule::launch_m0_forward_strided(
    const SymmetrixJitM0ArgsV2& strided_args, void* stream,
    const std::int32_t persistent_blocks) const
{
    const auto& args=strided_args.packed;
    if (!validate_symmetric_product_launch(
            state_.get(), strided_args, persistent_blocks,
            SYMMETRIX_JIT_OPERATOR_KIND_M0_V1, false))
        return;
    const auto blocks = symmetric_launch_blocks(
        *state_, args, false, persistent_blocks);
    void* parameters[] = {const_cast<SymmetrixJitM0ArgsV2*>(&strided_args)};
#if defined(KOKKOS_ENABLE_CUDA)
    check_backend(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(state_->forward), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution M0 forward)");
#elif defined(KOKKOS_ENABLE_HIP)
    check_backend(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(state_->forward), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution M0 forward)");
#else
    (void)stream;
#endif
}

void OperatorModule::launch_m0_reverse(
    const SymmetrixJitM0ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    if(state_==nullptr)throw std::logic_error("Execution symmetric module is unavailable.");
    launch_m0_reverse_strided(
        packed_symmetric_args(args,*state_),stream,persistent_blocks);
}

void OperatorModule::launch_m0_reverse_strided(
    const SymmetrixJitM0ArgsV2& strided_args, void* stream,
    const std::int32_t persistent_blocks) const
{
    const auto& args=strided_args.packed;
    if (!validate_symmetric_product_launch(
            state_.get(), strided_args, persistent_blocks,
            SYMMETRIX_JIT_OPERATOR_KIND_M0_V1, true))
        return;
    const auto blocks = symmetric_launch_blocks(
        *state_, args, true, persistent_blocks);
    void* parameters[] = {const_cast<SymmetrixJitM0ArgsV2*>(&strided_args)};
#if defined(KOKKOS_ENABLE_CUDA)
    check_backend(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(state_->reverse), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution M0 reverse)");
#elif defined(KOKKOS_ENABLE_HIP)
    check_backend(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(state_->reverse), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution M0 reverse)");
#else
    (void)stream;
#endif
}

void OperatorModule::launch_m1_forward(
    const SymmetrixJitM1ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    if(state_==nullptr)throw std::logic_error("Execution symmetric module is unavailable.");
    launch_m1_forward_strided(
        packed_symmetric_args(args,*state_),stream,persistent_blocks);
}

void OperatorModule::launch_m1_forward_strided(
    const SymmetrixJitM1ArgsV2& strided_args, void* stream,
    const std::int32_t persistent_blocks) const
{
    const auto& args=strided_args.packed;
    if (!validate_symmetric_product_launch(
            state_.get(), strided_args, persistent_blocks,
            SYMMETRIX_JIT_OPERATOR_KIND_M1_V1, false))
        return;
    const auto blocks = symmetric_launch_blocks(
        *state_, args, false, persistent_blocks);
    void* parameters[] = {const_cast<SymmetrixJitM1ArgsV2*>(&strided_args)};
#if defined(KOKKOS_ENABLE_CUDA)
    check_backend(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(state_->forward), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution M1 forward)");
#elif defined(KOKKOS_ENABLE_HIP)
    check_backend(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(state_->forward), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution M1 forward)");
#else
    (void)stream;
#endif
}

void OperatorModule::launch_m1_reverse(
    const SymmetrixJitM1ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    if(state_==nullptr)throw std::logic_error("Execution symmetric module is unavailable.");
    launch_m1_reverse_strided(
        packed_symmetric_args(args,*state_),stream,persistent_blocks);
}

void OperatorModule::launch_m1_reverse_strided(
    const SymmetrixJitM1ArgsV2& strided_args, void* stream,
    const std::int32_t persistent_blocks) const
{
    const auto& args=strided_args.packed;
    if (!validate_symmetric_product_launch(
            state_.get(), strided_args, persistent_blocks,
            SYMMETRIX_JIT_OPERATOR_KIND_M1_V1, true))
        return;
    const auto blocks = symmetric_launch_blocks(
        *state_, args, true, persistent_blocks);
    void* parameters[] = {const_cast<SymmetrixJitM1ArgsV2*>(&strided_args)};
#if defined(KOKKOS_ENABLE_CUDA)
    check_backend(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(state_->reverse), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<CUstream>(stream), parameters, nullptr),
        "cuLaunchKernel(Execution M1 reverse)");
#elif defined(KOKKOS_ENABLE_HIP)
    check_backend(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(state_->reverse), blocks, 1, 1,
            state_->threads_per_block, 1, 1, 0,
            reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        "hipModuleLaunchKernel(Execution M1 reverse)");
#else
    (void)stream;
#endif
}

namespace {

void validate_r0_launch(
    const OperatorModuleState* state,
    const SymmetrixJitR0ArgsV1& args,
    const std::int32_t persistent_blocks)
{
    if (state == nullptr || state->kind != SYMMETRIX_JIT_OPERATOR_KIND_R0_V1)
        throw std::logic_error("Execution R0 module is unavailable.");
    if (args.struct_size < sizeof(SymmetrixJitR0ArgsV1)
        || args.radial.struct_size < sizeof(SymmetrixJitR0SplineV1)
        || args.density.struct_size < sizeof(SymmetrixJitR0SplineV1)
        || args.num_nodes < 0 || args.num_nodes > INT32_MAX
        || args.num_edges < 0 || args.num_edges > INT32_MAX
        || args.active_type_count <= 0 || args.channels <= 0
        || args.channels != state->r0_channels
        || args.l_max != state->r0_l_max || args.coordinates_are_unit < 0
        || args.coordinates_are_unit > 1 || args.apply_density_scale < 0
        || args.apply_density_scale > 1
        || args.use_precomputed_scale_adjoint < 0
        || args.use_precomputed_scale_adjoint > 1
        || !std::isfinite(args.cutoff) || args.cutoff <= 0.0
        || persistent_blocks <= 0)
        throw std::invalid_argument("Execution R0 launch packet is invalid.");
}

void launch_r0_kernel(
    const OperatorModuleState& state,
    void* function,
    const SymmetrixJitR0ArgsV1& args,
    void* stream,
    const std::int64_t work_items,
    const std::int32_t owners_per_block,
    const std::int32_t threads_per_block,
    const std::int32_t persistent_blocks,
    const bool persistent,
    const std::uint32_t dynamic_shared_bytes,
    const char* operation)
{
    if (work_items == 0)
        return;
    validate_owner(state);
    const auto blocks = detail::launch_blocks(
        work_items, owners_per_block,
        persistent ? persistent_blocks : INT32_MAX);
    void* parameters[] = {const_cast<SymmetrixJitR0ArgsV1*>(&args)};
#if defined(KOKKOS_ENABLE_CUDA)
    check_backend(
        cuLaunchKernel(
            reinterpret_cast<CUfunction>(function), blocks, 1, 1,
            threads_per_block, 1, 1, dynamic_shared_bytes,
            reinterpret_cast<CUstream>(stream), parameters, nullptr),
        operation);
#elif defined(KOKKOS_ENABLE_HIP)
    check_backend(
        hipModuleLaunchKernel(
            reinterpret_cast<hipFunction_t>(function), blocks, 1, 1,
            threads_per_block, 1, 1, dynamic_shared_bytes,
            reinterpret_cast<hipStream_t>(stream), parameters, nullptr),
        operation);
#else
    (void)function;
    (void)stream;
    (void)dynamic_shared_bytes;
    (void)operation;
#endif
}

} // namespace

void OperatorModule::launch_r0_density_prepare(
    const SymmetrixJitR0ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    validate_r0_launch(state_.get(), args, persistent_blocks);
    launch_r0_kernel(
        *state_, state_->density_prepare, args, stream,
        static_cast<std::int64_t>(args.num_nodes)
            *state_->r0_density_extent_multiplier,
        state_->r0_density_owners_per_block,
        state_->r0_density_threads_per_block,
        persistent_blocks, state_->r0_density_persistent != 0,
        0u,
        "Execution R0 density prepare");
}

void OperatorModule::launch_r0_forward(
    const SymmetrixJitR0ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    validate_r0_launch(state_.get(), args, persistent_blocks);
    launch_r0_kernel(
        *state_, state_->forward, args, stream,
        static_cast<std::int64_t>(args.num_nodes)
            *state_->r0_forward_extent_multiplier,
        state_->r0_forward_owners_per_block,
        state_->r0_forward_threads_per_block,
        persistent_blocks, state_->r0_forward_persistent != 0,
        0u,
        "Execution R0 forward");
}

void OperatorModule::launch_r0_reverse_prepare(
    const SymmetrixJitR0ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    validate_r0_launch(state_.get(), args, persistent_blocks);
    launch_r0_kernel(
        *state_, state_->reverse_prepare, args, stream,
        static_cast<std::int64_t>(args.num_nodes)
            *state_->r0_reverse_prepare_extent_multiplier,
        state_->r0_reverse_prepare_owners_per_block,
        state_->r0_reverse_prepare_threads_per_block,
        persistent_blocks, state_->r0_reverse_prepare_persistent != 0,
        0u,
        "Execution R0 reverse prepare");
}

void OperatorModule::launch_r0_coordinate_reverse(
    const SymmetrixJitR0ArgsV1& args, void* stream,
    const std::int32_t persistent_blocks) const
{
    validate_r0_launch(state_.get(), args, persistent_blocks);
    if (state_->r0_l_max > 3 && args.harmonic_gradients == nullptr)
        throw std::runtime_error(
            "Generated R0 coordinate reverse above l_max=3 requires retained "
            "harmonic gradients.");
    const auto gradient_scratch_bytes = args.harmonic_gradients == nullptr
        ? static_cast<std::uint32_t>(
            state_->r0_coordinate_reverse_owners_per_block*3
            *state_->r0_harmonic_count*state_->scalar_size)
        : 0u;
    launch_r0_kernel(
        *state_, state_->coordinate_reverse, args, stream,
        static_cast<std::int64_t>(args.num_edges)
            *state_->r0_coordinate_reverse_extent_multiplier,
        state_->r0_coordinate_reverse_owners_per_block,
        state_->r0_coordinate_reverse_threads_per_block,
        persistent_blocks, state_->r0_coordinate_reverse_persistent != 0,
        gradient_scratch_bytes,
        "Execution R0 coordinate reverse");
}

} // namespace symmetrix::execution
