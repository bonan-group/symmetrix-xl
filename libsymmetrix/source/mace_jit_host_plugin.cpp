#include "mace_jit_host_plugin.hpp"
#include "jit_device_plugin_common.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<SymmetrixJitHostRadialSplineV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostR1ForwardArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostR1SourceArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostR1EdgeArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostPluginV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostRadialSplineV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostR1ForwardArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostR1SourceArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostR1EdgeArgsV2>);
static_assert(std::is_standard_layout_v<SymmetrixJitHostPluginV2>);
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(SymmetrixJitHostR1ForwardArgsV1) == 160);
static_assert(sizeof(SymmetrixJitHostR1SourceArgsV1) == 160);
static_assert(sizeof(SymmetrixJitHostR1EdgeArgsV1) == 176);
static_assert(sizeof(SymmetrixJitHostR1ForwardArgsV2) == 160);
static_assert(sizeof(SymmetrixJitHostR1SourceArgsV2) == 160);
static_assert(sizeof(SymmetrixJitHostR1EdgeArgsV2) == 184);
#endif

#if defined(__unix__) || defined(__APPLE__)
namespace symmetrix::execution {
namespace {

std::runtime_error plugin_error(
    const std::string& path,
    const std::string& message)
{
    return std::runtime_error(
        "Could not load Execution host plugin '"+path+"': "+message);
}

constexpr std::uint32_t known_capabilities =
    SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V1
    | SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V1
    | SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V1
    | SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V1
    | SYMMETRIX_JIT_HOST_R1_SOURCE_CHANNEL_TILE_16_V2
    | SYMMETRIX_JIT_HOST_R1_FORWARD_CHANNEL_TILE_16_V2
    | SYMMETRIX_JIT_HOST_R1_SOURCE_CHANNEL_TILE_32_V2;

void validate_expectation(const HostPluginExpectation& expectation)
{
    if (expectation.artifact_id.empty())
        throw std::invalid_argument(
            "Execution host plugin expectation requires an artifact id.");
    if (expectation.contract_fingerprint.empty())
        throw std::invalid_argument(
            "Execution host plugin expectation requires a contract fingerprint.");
    if (expectation.channels <= 0 || expectation.embedding <= 0
        || expectation.edge_l_max < 0 || expectation.source_l_max < 0)
        throw std::invalid_argument(
            "Execution host plugin expectation requires exact model extents.");
    if (expectation.required_capabilities == 0
        || (expectation.required_capabilities&~known_capabilities) != 0)
        throw std::invalid_argument(
            "Execution host plugin expectation requires exact known capabilities.");
    if ((expectation.scalar_kind
                == SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2
            && expectation.scalar_size != sizeof(float))
        || (expectation.scalar_kind
                == SYMMETRIX_JIT_HOST_SCALAR_FLOAT64_V2
            && expectation.scalar_size != sizeof(double))
        || (expectation.scalar_kind
                != SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2
            && expectation.scalar_kind
                != SYMMETRIX_JIT_HOST_SCALAR_FLOAT64_V2))
        throw std::invalid_argument(
            "Execution host plugin expectation has an invalid scalar kind or width.");
}

// The identity envelope both host ABI generations publish.  Only the ABI tag,
// version, and descriptor size differ between them.
detail::ArtifactEnvelopeExpectation envelope_expectation(
    const HostPluginExpectation& expectation,
    const std::uint32_t abi_version,
    const std::size_t struct_size,
    const std::string_view abi_tag)
{
    detail::ArtifactEnvelopeExpectation identity;
    identity.abi_version = abi_version;
    identity.struct_size = struct_size;
    identity.byte_order = SYMMETRIX_JIT_HOST_PLUGIN_BYTE_ORDER;
    identity.abi_tag = abi_tag;
    identity.artifact_id = expectation.artifact_id;
    identity.identity_fingerprint = expectation.contract_fingerprint;
    identity.semantic_fingerprint = expectation.semantic_fingerprint;
    identity.structure_fingerprint = expectation.structure_fingerprint;
    return identity;
}

void validate_descriptor_v1(
    const std::string& path,
    const SymmetrixJitHostPluginV1& descriptor,
    const HostPluginExpectation& expectation)
{
    const detail::PluginLoaderContext context("host", path);
    if (expectation.scalar_kind != SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2
        || expectation.scalar_size != sizeof(float))
        throw context.error("ABI version 1 only supports float32");
    const detail::ArtifactEnvelope envelope{
        .abi_version = descriptor.abi_version,
        .struct_size = descriptor.struct_size,
        .pointer_size = descriptor.pointer_size,
        .byte_order = descriptor.byte_order,
        .checks_reserved = true,
        .reserved = descriptor.reserved,
        .abi_tag = descriptor.abi_tag,
        .artifact_id = descriptor.artifact_id,
        .identity_fingerprint = descriptor.contract_fingerprint,
        .semantic_fingerprint = descriptor.semantic_fingerprint,
        .structure_fingerprint = descriptor.structure_fingerprint,
    };
    detail::validate_artifact_envelope(
        context,
        envelope,
        envelope_expectation(
            expectation,
            SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION,
            sizeof(SymmetrixJitHostPluginV1),
            SYMMETRIX_JIT_HOST_PLUGIN_ABI_TAG));

    if (descriptor.channels <= 0 || descriptor.embedding <= 0
        || descriptor.edge_l_max < 0 || descriptor.source_l_max < 0)
        throw context.error("model extents are invalid");
    if (descriptor.capabilities != expectation.required_capabilities)
        throw context.error("owner function capabilities do not match");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V1,
        descriptor.r1_forward_owner != nullptr, "R1 forward owner");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V1,
        descriptor.r1_source_owner != nullptr, "R1 source owner");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V1,
        descriptor.r1_compensated_source_owner != nullptr,
        "R1 compensated source owner");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V1,
        descriptor.r1_edge_owner != nullptr, "R1 edge owner");

    context.require_extent(
        expectation.channels, descriptor.channels, "channel count");
    context.require_extent(
        expectation.embedding, descriptor.embedding, "embedding width");
    context.require_extent(
        expectation.edge_l_max, descriptor.edge_l_max, "edge l_max");
    context.require_extent(
        expectation.source_l_max, descriptor.source_l_max, "source l_max");
}

void validate_descriptor_v2(
    const std::string& path,
    const SymmetrixJitHostPluginV2& descriptor,
    const HostPluginExpectation& expectation)
{
    const detail::PluginLoaderContext context("host", path);
    const detail::ArtifactEnvelope envelope{
        .abi_version = descriptor.abi_version,
        .struct_size = descriptor.struct_size,
        .pointer_size = descriptor.pointer_size,
        .byte_order = descriptor.byte_order,
        .checks_reserved = true,
        .reserved = descriptor.reserved,
        .abi_tag = descriptor.abi_tag,
        .artifact_id = descriptor.artifact_id,
        .identity_fingerprint = descriptor.contract_fingerprint,
        .semantic_fingerprint = descriptor.semantic_fingerprint,
        .structure_fingerprint = descriptor.structure_fingerprint,
    };
    detail::validate_artifact_envelope(
        context,
        envelope,
        envelope_expectation(
            expectation,
            SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION_V2,
            sizeof(SymmetrixJitHostPluginV2),
            SYMMETRIX_JIT_HOST_PLUGIN_ABI_TAG_V2));

    if (descriptor.scalar_kind != expectation.scalar_kind
        || descriptor.scalar_size != expectation.scalar_size)
        throw context.error("scalar kind or width does not match");
    if (descriptor.channels <= 0 || descriptor.embedding <= 0
        || descriptor.edge_l_max < 0 || descriptor.source_l_max < 0)
        throw context.error("model extents are invalid");
    if (descriptor.capabilities != expectation.required_capabilities)
        throw context.error("owner function capabilities do not match");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V2,
        descriptor.r1_forward_owner != nullptr, "R1 forward owner");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V2,
        descriptor.r1_source_owner != nullptr, "R1 source owner");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V2,
        descriptor.r1_compensated_source_owner != nullptr,
        "R1 compensated source owner");
    detail::require_capability_agreement(
        context, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V2,
        descriptor.r1_edge_owner != nullptr, "R1 edge owner");

    context.require_extent(
        expectation.channels, descriptor.channels, "channel count");
    context.require_extent(
        expectation.embedding, descriptor.embedding, "embedding width");
    context.require_extent(
        expectation.edge_l_max, descriptor.edge_l_max, "edge l_max");
    context.require_extent(
        expectation.source_l_max, descriptor.source_l_max, "source l_max");
}

}  // namespace

HostPlugin HostPlugin::load(
    std::string path,
    const HostPluginExpectation& expectation)
{
    if (path.empty())
        throw std::invalid_argument("Execution host plugin path is empty.");
    validate_expectation(expectation);

    void* handle = dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw plugin_error(path, error == nullptr ? "dlopen failed" : error);
    }

    try {
        dlerror();
        void* symbol_v2 = dlsym(
            handle, SYMMETRIX_JIT_HOST_PLUGIN_QUERY_SYMBOL_V2);
        const char* symbol_v2_error = dlerror();
        if (symbol_v2_error == nullptr && symbol_v2 != nullptr) {
            const auto query =
                reinterpret_cast<SymmetrixJitHostPluginQueryV2>(symbol_v2);
            const auto* descriptor = query();
            if (descriptor == nullptr)
                throw plugin_error(path, "query returned a null descriptor");
            validate_descriptor_v2(path, *descriptor, expectation);
            return HostPlugin(handle, nullptr, descriptor, std::move(path));
        }

        dlerror();
        void* symbol_v1 = dlsym(
            handle, SYMMETRIX_JIT_HOST_PLUGIN_QUERY_SYMBOL);
        const char* symbol_v1_error = dlerror();
        if (symbol_v1_error != nullptr)
            throw plugin_error(path, symbol_v1_error);
        if (symbol_v1 == nullptr)
            throw plugin_error(path, "query symbol is null");
        const auto query =
            reinterpret_cast<SymmetrixJitHostPluginQueryV1>(symbol_v1);
        const auto* descriptor = query();
        if (descriptor == nullptr)
            throw plugin_error(path, "query returned a null descriptor");
        validate_descriptor_v1(path, *descriptor, expectation);
        return HostPlugin(handle, descriptor, nullptr, std::move(path));
    } catch (...) {
        dlclose(handle);
        throw;
    }
}

HostPlugin::HostPlugin(
    void* handle,
    const SymmetrixJitHostPluginV1* descriptor_v1,
    const SymmetrixJitHostPluginV2* descriptor_v2,
    std::string path) noexcept
    : handle_(handle), descriptor_v1_(descriptor_v1),
      descriptor_v2_(descriptor_v2), path_(std::move(path))
{}

HostPlugin::~HostPlugin()
{
    reset();
}

HostPlugin::HostPlugin(HostPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_v1_(std::exchange(other.descriptor_v1_, nullptr)),
      descriptor_v2_(std::exchange(other.descriptor_v2_, nullptr)),
      path_(std::move(other.path_))
{}

HostPlugin& HostPlugin::operator=(HostPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_v1_ = std::exchange(other.descriptor_v1_, nullptr);
        descriptor_v2_ = std::exchange(other.descriptor_v2_, nullptr);
        path_ = std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitHostPluginV1& HostPlugin::descriptor() const
{
    return descriptor_v1();
}

std::uint32_t HostPlugin::abi_version() const noexcept
{
    if (descriptor_v2_ != nullptr)
        return SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION_V2;
    return descriptor_v1_ == nullptr ? 0u : SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION;
}

const SymmetrixJitHostPluginV1& HostPlugin::descriptor_v1() const
{
    if (descriptor_v1_ == nullptr)
        throw std::logic_error("Execution host plugin handle is empty.");
    return *descriptor_v1_;
}

const SymmetrixJitHostPluginV2& HostPlugin::descriptor_v2() const
{
    if (descriptor_v2_ == nullptr)
        throw std::logic_error("Execution host plugin is not ABI version 2.");
    return *descriptor_v2_;
}

void HostPlugin::reset() noexcept
{
    descriptor_v1_ = nullptr;
    descriptor_v2_ = nullptr;
    if (handle_ != nullptr) {
        dlclose(handle_);
        handle_ = nullptr;
    }
    path_.clear();
}

}  // namespace symmetrix::execution
#else
namespace symmetrix::execution {

HostPlugin HostPlugin::load(
    std::string,
    const HostPluginExpectation&)
{
    throw std::runtime_error(
        "Execution host plugins require POSIX dlopen/dlsym support.");
}

HostPlugin::HostPlugin(
    void* handle,
    const SymmetrixJitHostPluginV1* descriptor_v1,
    const SymmetrixJitHostPluginV2* descriptor_v2,
    std::string path) noexcept
    : handle_(handle), descriptor_v1_(descriptor_v1),
      descriptor_v2_(descriptor_v2), path_(std::move(path))
{}

HostPlugin::~HostPlugin()
{
    reset();
}

HostPlugin::HostPlugin(HostPlugin&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      descriptor_v1_(std::exchange(other.descriptor_v1_, nullptr)),
      descriptor_v2_(std::exchange(other.descriptor_v2_, nullptr)),
      path_(std::move(other.path_))
{}

HostPlugin& HostPlugin::operator=(HostPlugin&& other) noexcept
{
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        descriptor_v1_ = std::exchange(other.descriptor_v1_, nullptr);
        descriptor_v2_ = std::exchange(other.descriptor_v2_, nullptr);
        path_ = std::move(other.path_);
    }
    return *this;
}

const SymmetrixJitHostPluginV1& HostPlugin::descriptor() const
{
    throw std::logic_error("Execution host plugin support is disabled.");
}

std::uint32_t HostPlugin::abi_version() const noexcept { return 0u; }

const SymmetrixJitHostPluginV1& HostPlugin::descriptor_v1() const
{
    throw std::logic_error("Execution host plugin support is disabled.");
}

const SymmetrixJitHostPluginV2& HostPlugin::descriptor_v2() const
{
    throw std::logic_error("Execution host plugin support is disabled.");
}

void HostPlugin::reset() noexcept
{
    handle_ = nullptr;
    descriptor_v1_ = nullptr;
    descriptor_v2_ = nullptr;
    path_.clear();
}

}  // namespace symmetrix::execution
#endif
