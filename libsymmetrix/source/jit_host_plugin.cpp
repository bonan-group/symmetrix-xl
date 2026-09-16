#include "jit_host_plugin.hpp"

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

std::string require_text(
    const std::string& path,
    const char* value,
    const char* field)
{
    if (value == nullptr || value[0] == '\0')
        throw plugin_error(path, std::string(field)+" is empty");
    return value;
}

void require_match(
    const std::string& path,
    const std::string_view expected,
    const char* actual,
    const char* field)
{
    if (!expected.empty() && expected != actual)
        throw plugin_error(path, std::string(field)+" does not match");
}

void require_extent(
    const std::string& path,
    const std::int32_t expected,
    const std::int32_t actual,
    const char* field)
{
    if (expected >= 0 && expected != actual)
        throw plugin_error(path, std::string(field)+" does not match");
}

void validate_capability(
    const std::string& path,
    const std::uint32_t capabilities,
    const std::uint32_t capability,
    const bool has_function,
    const char* name)
{
    const bool declared = (capabilities&capability) != 0;
    if (declared != has_function)
        throw plugin_error(
            path,
            std::string(name)+" capability and function pointer disagree");
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

void validate_descriptor_v1(
    const std::string& path,
    const SymmetrixJitHostPluginV1& descriptor,
    const HostPluginExpectation& expectation)
{
    if (expectation.scalar_kind != SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2
        || expectation.scalar_size != sizeof(float))
        throw plugin_error(path, "ABI version 1 only supports float32");
    if (descriptor.abi_version != SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION)
        throw plugin_error(path, "unsupported ABI version");
    if (descriptor.struct_size < sizeof(SymmetrixJitHostPluginV1))
        throw plugin_error(path, "descriptor is smaller than ABI version 1");
    if (descriptor.pointer_size != sizeof(void*))
        throw plugin_error(path, "pointer width does not match this process");
    if (descriptor.byte_order != SYMMETRIX_JIT_HOST_PLUGIN_BYTE_ORDER)
        throw plugin_error(path, "byte order does not match this process");
    if (descriptor.reserved != 0)
        throw plugin_error(path, "reserved descriptor field is nonzero");

    const auto abi_tag = require_text(path, descriptor.abi_tag, "ABI tag");
    if (abi_tag != SYMMETRIX_JIT_HOST_PLUGIN_ABI_TAG)
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
        throw plugin_error(path, "owner function capabilities do not match");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V1,
        descriptor.r1_forward_owner != nullptr, "R1 forward owner");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V1,
        descriptor.r1_source_owner != nullptr, "R1 source owner");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V1,
        descriptor.r1_compensated_source_owner != nullptr,
        "R1 compensated source owner");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V1,
        descriptor.r1_edge_owner != nullptr, "R1 edge owner");

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
    require_extent(
        path, expectation.channels, descriptor.channels, "channel count");
    require_extent(
        path, expectation.embedding, descriptor.embedding, "embedding width");
    require_extent(
        path, expectation.edge_l_max, descriptor.edge_l_max, "edge l_max");
    require_extent(
        path, expectation.source_l_max,
        descriptor.source_l_max, "source l_max");
}

void validate_descriptor_v2(
    const std::string& path,
    const SymmetrixJitHostPluginV2& descriptor,
    const HostPluginExpectation& expectation)
{
    if (descriptor.abi_version != SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION_V2)
        throw plugin_error(path, "unsupported ABI version");
    if (descriptor.struct_size < sizeof(SymmetrixJitHostPluginV2))
        throw plugin_error(path, "descriptor is smaller than ABI version 2");
    if (descriptor.pointer_size != sizeof(void*))
        throw plugin_error(path, "pointer width does not match this process");
    if (descriptor.byte_order != SYMMETRIX_JIT_HOST_PLUGIN_BYTE_ORDER)
        throw plugin_error(path, "byte order does not match this process");
    if (descriptor.reserved != 0)
        throw plugin_error(path, "reserved descriptor field is nonzero");
    if (descriptor.scalar_kind != expectation.scalar_kind
        || descriptor.scalar_size != expectation.scalar_size)
        throw plugin_error(path, "scalar kind or width does not match");

    const auto abi_tag = require_text(path, descriptor.abi_tag, "ABI tag");
    if (abi_tag != SYMMETRIX_JIT_HOST_PLUGIN_ABI_TAG_V2)
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
        throw plugin_error(path, "owner function capabilities do not match");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V2,
        descriptor.r1_forward_owner != nullptr, "R1 forward owner");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V2,
        descriptor.r1_source_owner != nullptr, "R1 source owner");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V2,
        descriptor.r1_compensated_source_owner != nullptr,
        "R1 compensated source owner");
    validate_capability(
        path, descriptor.capabilities,
        SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V2,
        descriptor.r1_edge_owner != nullptr, "R1 edge owner");

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
