#include "mace_jit_r0_host_plugin.hpp"
#include "jit_device_plugin_common.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<SymmetrixJitR0HostArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitR0HostPluginV2>);

namespace symmetrix::execution {

#if defined(__unix__) || defined(__APPLE__)
namespace {

std::runtime_error plugin_error(
    const std::string& path, const std::string& message)
{
    return std::runtime_error(
        "Could not load Execution R0 host plugin '"+path+"': "+message);
}

void validate(
    const std::string& path, const SymmetrixJitR0HostPluginV2& value,
    const R0HostPluginExpectation& expected)
{
    constexpr std::uint32_t capabilities =
        SYMMETRIX_JIT_R0_HOST_DENSITY_PREPARE_V1
        |SYMMETRIX_JIT_R0_HOST_FORWARD_V1
        |SYMMETRIX_JIT_R0_HOST_REVERSE_PREPARE_V1
        |SYMMETRIX_JIT_R0_HOST_COORDINATE_REVERSE_V1
        |SYMMETRIX_JIT_R0_HOST_FORWARD_CHANNEL_TILED_V1
        |SYMMETRIX_JIT_R0_HOST_COORDINATE_REVERSE_RECEIVER_V1
        |SYMMETRIX_JIT_R0_HOST_COMPACT_GEOMETRY_V1
        |SYMMETRIX_JIT_R0_HOST_PRECOMPUTED_SCALE_V1
        |SYMMETRIX_JIT_R0_HOST_RECEIVER_BATCH_V1;
    const detail::PluginLoaderContext context("R0 host", path);
    const detail::ArtifactEnvelope envelope{
        .abi_version = value.abi_version,
        .struct_size = value.struct_size,
        .pointer_size = value.pointer_size,
        .byte_order = value.byte_order,
        .checks_reserved = false,
        .abi_tag = value.abi_tag,
        .artifact_id = value.artifact_id,
        .identity_fingerprint = nullptr,
        .semantic_fingerprint = nullptr,
        .structure_fingerprint = value.structure_fingerprint,
    };
    detail::ArtifactEnvelopeExpectation identity;
    identity.abi_version = SYMMETRIX_JIT_R0_HOST_PLUGIN_ABI_VERSION_V2;
    identity.struct_size = sizeof(SymmetrixJitR0HostPluginV2);
    identity.byte_order = SYMMETRIX_JIT_R0_HOST_PLUGIN_BYTE_ORDER_V1;
    identity.abi_tag = SYMMETRIX_JIT_R0_HOST_PLUGIN_ABI_TAG_V2;
    identity.artifact_id = expected.artifact_id;
    identity.structure_fingerprint = expected.structure_fingerprint;
    identity.publishes_identity_fingerprint = false;
    identity.publishes_semantic_fingerprint = false;
    detail::validate_artifact_envelope(context, envelope, identity);

    if (value.capabilities != capabilities
        || value.density_prepare_owner == nullptr
        || value.forward_owner == nullptr
        || value.reverse_prepare_owner == nullptr
        || value.coordinate_reverse_owner == nullptr
        || value.density_prepare_range == nullptr
        || value.forward_range == nullptr
        || value.reverse_prepare_range == nullptr
        || value.coordinate_reverse_range == nullptr)
        throw context.error("owner capabilities do not match");
    const R0LogicalDescriptor actual{
        value.artifact_id == nullptr ? "" : value.artifact_id,
        value.structure_fingerprint == nullptr ? "" : value.structure_fingerprint,
        value.target == nullptr ? "" : value.target,
        value.schedule == nullptr ? "" : value.schedule,
        value.scalar_kind, value.scalar_size, r0_required_capabilities,
        value.channels, value.l_max, value.harmonic_count,
        value.edge_weight_mode, value.schedule_revision,
        value.compatibility_version};
    if (const auto field = r0_descriptor_mismatch(actual, expected); !field.empty())
        throw context.error(
            "logical R0 descriptor field "+std::string(field)+" does not match");
}

}  // namespace

R0HostPlugin R0HostPlugin::load(
    std::string path, const R0HostPluginExpectation& expectation)
{
    if (path.empty() || expectation.artifact_id.empty()
        || expectation.structure_fingerprint.empty()
        || expectation.target.empty() || expectation.schedule.empty()
        || expectation.compatibility_version == 0)
        throw std::invalid_argument("Execution R0 host plugin identity is incomplete.");
    void* handle = dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw plugin_error(path, error == nullptr ? "dlopen failed" : error);
    }
    try {
        dlerror();
        void* symbol = dlsym(
            handle, SYMMETRIX_JIT_R0_HOST_PLUGIN_QUERY_SYMBOL_V2);
        const char* error = dlerror();
        if (error != nullptr || symbol == nullptr)
            throw plugin_error(path, error == nullptr ? "query symbol is null" : error);
        const auto query = reinterpret_cast<SymmetrixJitR0HostPluginQueryV2>(symbol);
        const auto* descriptor = query();
        if (descriptor == nullptr)
            throw plugin_error(path, "query returned a null descriptor");
        validate(path, *descriptor, expectation);
        return R0HostPlugin(handle, descriptor, std::move(path));
    } catch (...) {
        dlclose(handle);
        throw;
    }
}

R0HostPlugin::R0HostPlugin(
    void* handle, const SymmetrixJitR0HostPluginV2* descriptor,
    std::string path) noexcept
    : handle_(handle), descriptor_(descriptor), path_(std::move(path)) {}

R0HostPlugin::~R0HostPlugin() { reset(); }
R0HostPlugin::R0HostPlugin(R0HostPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_(std::exchange(other.descriptor_, nullptr)),
      path_(std::move(other.path_)) {}
R0HostPlugin& R0HostPlugin::operator=(R0HostPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_ = std::exchange(other.descriptor_, nullptr);
        path_ = std::move(other.path_);
    }
    return *this;
}
const SymmetrixJitR0HostPluginV2& R0HostPlugin::descriptor() const
{
    if (descriptor_ == nullptr)
        throw std::logic_error("Execution R0 host plugin is not loaded.");
    return *descriptor_;
}
void R0HostPlugin::reset() noexcept
{
    descriptor_ = nullptr;
    if (handle_ != nullptr)
        dlclose(handle_);
    handle_ = nullptr;
    path_.clear();
}

#else
R0HostPlugin R0HostPlugin::load(std::string, const R0HostPluginExpectation&)
{
    throw std::runtime_error("Execution R0 host plugins require POSIX dynamic loading.");
}
R0HostPlugin::~R0HostPlugin() = default;
R0HostPlugin::R0HostPlugin(R0HostPlugin&&) noexcept = default;
R0HostPlugin& R0HostPlugin::operator=(R0HostPlugin&&) noexcept = default;
const SymmetrixJitR0HostPluginV2& R0HostPlugin::descriptor() const
{
    throw std::logic_error("Execution R0 host plugin is not loaded.");
}
void R0HostPlugin::reset() noexcept {}
#endif

}  // namespace symmetrix::execution
