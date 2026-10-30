#include "mace_jit_m0_host_plugin.hpp"
#include "jit_device_plugin_common.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<SymmetrixJitM0HostArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitM0HostPluginV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitM0HostPluginV2>);

namespace symmetrix::execution {

#if defined(__unix__) || defined(__APPLE__)
namespace {

std::runtime_error plugin_error(
    const std::string& path, const std::string& operator_name,
    const std::string& message)
{
    return std::runtime_error(
        "Could not load Execution "+operator_name+" host plugin '"+path+"': "+message);
}

void validate(
    const std::string& path, const SymmetrixJitM0HostPluginV1& value,
    const M0HostPluginExpectation& expected,
    const char* context_name,
    const char* abi_tag,
    const std::uint32_t abi_version,
    const std::uint32_t struct_size)
{
    constexpr std::uint32_t capabilities =
        SYMMETRIX_JIT_M0_HOST_FORWARD_OWNER_V1
        |SYMMETRIX_JIT_M0_HOST_REVERSE_OWNER_V1
        |SYMMETRIX_JIT_M0_HOST_SCALE_ADJOINT_V1;
    constexpr std::uint32_t tiled_capabilities = capabilities
        |SYMMETRIX_JIT_M0_HOST_CHANNEL_TILED_OWNER_V1;
    const detail::PluginLoaderContext context(context_name, path);
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
    identity.abi_version = abi_version;
    identity.struct_size = struct_size;
    identity.byte_order = SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1;
    identity.abi_tag = abi_tag;
    identity.artifact_id = expected.artifact_id;
    identity.structure_fingerprint = expected.structure_fingerprint;
    // The M0 host ABI identifies its artifact by the structure fingerprint; it
    // publishes neither a contract nor a semantic fingerprint.
    identity.publishes_identity_fingerprint = false;
    identity.publishes_semantic_fingerprint = false;
    detail::validate_artifact_envelope(context, envelope, identity);

    if (value.scalar_kind != expected.scalar_kind
        || value.scalar_size != expected.scalar_size)
        throw context.error("scalar kind or width does not match");
    if (value.channels != expected.channels
        || value.input_components != expected.input_components
        || value.output_components != expected.output_components
        || value.correlation != expected.correlation
        || value.term_count != expected.term_count)
        throw context.error("model extents do not match");
    if ((value.capabilities != capabilities
            && value.capabilities != tiled_capabilities)
        || value.forward_owner == nullptr
        || value.reverse_owner == nullptr)
        throw context.error("owner capabilities do not match");
    if (value.capabilities == capabilities) {
        if (value.owner_channel_tile != 0)
            throw context.error("scalar owner tile must be zero");
    } else if (value.owner_channel_tile <= 1
        || value.owner_channel_tile > static_cast<std::uint32_t>(value.channels)
        || (value.owner_channel_tile&(value.owner_channel_tile-1)) != 0) {
        throw context.error("channel owner tile is invalid");
    }
}

void validate_v2(
    const std::string& path, const SymmetrixJitM0HostPluginV2& value,
    const M0HostPluginExpectation& expected,
    const char* context_name,
    const char* abi_tag)
{
    validate(
        path, value.v1, expected, context_name, abi_tag,
        SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_VERSION_V2,
        sizeof(SymmetrixJitM0HostPluginV2));
    const detail::PluginLoaderContext context(context_name, path);
    if (!expected.generated_artifact_id.empty()
        && context.require_text(
            value.generated_artifact_id, "generated_artifact_id")
            != expected.generated_artifact_id)
        throw context.error("generated artifact identity does not match");
    if (!expected.target.empty()
        && context.require_text(value.target, "target")
            != expected.target)
        throw context.error("target does not match");
}

const SymmetrixJitM0HostPluginV1* load_descriptor(
    void* handle, const std::string& path,
    const M0HostPluginExpectation& expectation,
    const char* context_name,
    const char* query_v2_name,
    const char* query_v1_name,
    const char* abi_tag_v2,
    const char* abi_tag_v1)
{
    dlerror();
    void* symbol_v2 = dlsym(handle, query_v2_name);
    const char* error_v2 = dlerror();
    if (error_v2 == nullptr && symbol_v2 != nullptr) {
        const auto query =
            reinterpret_cast<SymmetrixJitM0HostPluginQueryV2>(symbol_v2);
        const auto* descriptor = query();
        if (descriptor == nullptr)
            throw plugin_error(path, context_name, "query returned a null descriptor");
        validate_v2(path, *descriptor, expectation, context_name, abi_tag_v2);
        return &descriptor->v1;
    }
    if (!expectation.generated_artifact_id.empty() || !expectation.target.empty())
        throw plugin_error(
            path, context_name,
            "target-bound ABI-v2 query symbol is required");
    dlerror();
    void* symbol_v1 = dlsym(handle, query_v1_name);
    const char* error_v1 = dlerror();
    if (error_v1 != nullptr || symbol_v1 == nullptr)
        throw plugin_error(
            path, context_name,
            error_v1 == nullptr ? "query symbol is null" : error_v1);
    const auto query =
        reinterpret_cast<SymmetrixJitM0HostPluginQueryV1>(symbol_v1);
    const auto* descriptor = query();
    if (descriptor == nullptr)
        throw plugin_error(path, context_name, "query returned a null descriptor");
    validate(
        path, *descriptor, expectation, context_name, abi_tag_v1,
        SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_VERSION_V1,
        sizeof(SymmetrixJitM0HostPluginV1));
    return descriptor;
}

}  // namespace

M0HostPlugin M0HostPlugin::load(
    std::string path, const M0HostPluginExpectation& expectation)
{
    if (path.empty() || expectation.artifact_id.empty()
        || expectation.structure_fingerprint.empty())
        throw std::invalid_argument("Execution M0 host plugin identity is incomplete.");
    void* handle = dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw plugin_error(
            path, "M0", error == nullptr ? "dlopen failed" : error);
    }
    try {
        const auto* descriptor = load_descriptor(
            handle, path, expectation, "M0 host",
            SYMMETRIX_JIT_M0_HOST_PLUGIN_QUERY_SYMBOL_V2,
            SYMMETRIX_JIT_M0_HOST_PLUGIN_QUERY_SYMBOL_V1,
            SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_TAG_V2,
            SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_TAG_V1);
        return M0HostPlugin(handle, descriptor, std::move(path));
    } catch (...) {
        dlclose(handle);
        throw;
    }
}

M0HostPlugin M0HostPlugin::load_m1(
    std::string path, const M1HostPluginExpectation& expectation)
{
    if (path.empty() || expectation.artifact_id.empty()
        || expectation.structure_fingerprint.empty())
        throw std::invalid_argument("Execution M1 host plugin identity is incomplete.");
    void* handle = dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw plugin_error(
            path, "M1", error == nullptr ? "dlopen failed" : error);
    }
    try {
        const auto* descriptor = load_descriptor(
            handle, path, expectation, "M1 host",
            SYMMETRIX_JIT_M1_HOST_PLUGIN_QUERY_SYMBOL_V2,
            SYMMETRIX_JIT_M1_HOST_PLUGIN_QUERY_SYMBOL_V1,
            SYMMETRIX_JIT_M1_HOST_PLUGIN_ABI_TAG_V2,
            SYMMETRIX_JIT_M1_HOST_PLUGIN_ABI_TAG_V1);
        return M0HostPlugin(handle, descriptor, std::move(path));
    } catch (...) {
        dlclose(handle);
        throw;
    }
}

M0HostPlugin::M0HostPlugin(
    void* handle, const SymmetrixJitM0HostPluginV1* descriptor,
    std::string path) noexcept
    : handle_(handle), descriptor_(descriptor), path_(std::move(path)) {}

M0HostPlugin::~M0HostPlugin() { reset(); }

M0HostPlugin::M0HostPlugin(M0HostPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_(std::exchange(other.descriptor_, nullptr)),
      path_(std::move(other.path_)) {}

M0HostPlugin& M0HostPlugin::operator=(M0HostPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_ = std::exchange(other.descriptor_, nullptr);
        path_ = std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitM0HostPluginV1& M0HostPlugin::descriptor() const
{
    if (descriptor_ == nullptr)
        throw std::logic_error("Execution M0 host plugin is not loaded.");
    return *descriptor_;
}

void M0HostPlugin::reset() noexcept
{
    descriptor_ = nullptr;
    if (handle_ != nullptr)
        dlclose(handle_);
    handle_ = nullptr;
    path_.clear();
}

#else

M0HostPlugin M0HostPlugin::load(
    std::string, const M0HostPluginExpectation&)
{
    throw std::runtime_error(
        "Execution M0 host plugins require POSIX dynamic loading.");
}
M0HostPlugin M0HostPlugin::load_m1(
    std::string, const M1HostPluginExpectation&)
{
    throw std::runtime_error(
        "Execution M1 host plugins require POSIX dynamic loading.");
}
M0HostPlugin::~M0HostPlugin() = default;
M0HostPlugin::M0HostPlugin(M0HostPlugin&&) noexcept = default;
M0HostPlugin& M0HostPlugin::operator=(M0HostPlugin&&) noexcept = default;
const SymmetrixJitM0HostPluginV1& M0HostPlugin::descriptor() const
{
    throw std::logic_error("Execution M0 host plugin is not loaded.");
}
void M0HostPlugin::reset() noexcept {}

#endif

}  // namespace symmetrix::execution
