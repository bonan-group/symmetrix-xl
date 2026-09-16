#include "jit_m0_host_plugin.hpp"

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<SymmetrixJitM0HostArgsV1>);
static_assert(std::is_standard_layout_v<SymmetrixJitM0HostPluginV1>);

namespace symmetrix::execution {

#if defined(__unix__) || defined(__APPLE__)
namespace {

std::runtime_error plugin_error(
    const std::string& path, const std::string& message)
{
    return std::runtime_error(
        "Could not load Execution M0 host plugin '"+path+"': "+message);
}

void validate(
    const std::string& path, const SymmetrixJitM0HostPluginV1& value,
    const M0HostPluginExpectation& expected)
{
    constexpr std::uint32_t capabilities =
        SYMMETRIX_JIT_M0_HOST_FORWARD_OWNER_V1
        |SYMMETRIX_JIT_M0_HOST_REVERSE_OWNER_V1
        |SYMMETRIX_JIT_M0_HOST_ALIAS_SAFE_V1
        |SYMMETRIX_JIT_M0_HOST_SCALE_ADJOINT_V1;
    constexpr std::uint32_t tiled_capabilities = capabilities
        |SYMMETRIX_JIT_M0_HOST_CHANNEL_TILED_OWNER_V1;
    if (value.abi_version != SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_VERSION_V1
        || value.struct_size < sizeof(SymmetrixJitM0HostPluginV1)
        || value.pointer_size != sizeof(void*)
        || value.byte_order != SYMMETRIX_JIT_M0_HOST_PLUGIN_BYTE_ORDER_V1)
        throw plugin_error(path, "ABI layout does not match");
    if (value.abi_tag == nullptr
        || std::string_view(value.abi_tag) != SYMMETRIX_JIT_M0_HOST_PLUGIN_ABI_TAG_V1)
        throw plugin_error(path, "ABI tag does not match");
    if (value.artifact_id == nullptr
        || std::string_view(value.artifact_id) != expected.artifact_id)
        throw plugin_error(path, "artifact id does not match");
    if (value.structure_fingerprint == nullptr
        || std::string_view(value.structure_fingerprint)
            != expected.structure_fingerprint)
        throw plugin_error(path, "structure fingerprint does not match");
    if (value.scalar_kind != expected.scalar_kind
        || value.scalar_size != expected.scalar_size)
        throw plugin_error(path, "scalar kind or width does not match");
    if (value.channels != expected.channels
        || value.input_components != expected.input_components
        || value.output_components != expected.output_components
        || value.correlation != expected.correlation
        || value.term_count != expected.term_count)
        throw plugin_error(path, "model extents do not match");
    if ((value.capabilities != capabilities
            && value.capabilities != tiled_capabilities)
        || value.forward_owner == nullptr
        || value.reverse_owner == nullptr)
        throw plugin_error(path, "owner capabilities do not match");
    if (value.capabilities == capabilities) {
        if (value.owner_channel_tile != 0)
            throw plugin_error(path, "scalar owner tile must be zero");
    } else if (value.owner_channel_tile <= 1
        || value.owner_channel_tile > static_cast<std::uint32_t>(value.channels)
        || (value.owner_channel_tile&(value.owner_channel_tile-1)) != 0) {
        throw plugin_error(path, "channel owner tile is invalid");
    }
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
        throw plugin_error(path, error == nullptr ? "dlopen failed" : error);
    }
    try {
        dlerror();
        void* symbol = dlsym(
            handle, SYMMETRIX_JIT_M0_HOST_PLUGIN_QUERY_SYMBOL_V1);
        const char* error = dlerror();
        if (error != nullptr || symbol == nullptr)
            throw plugin_error(path, error == nullptr ? "query symbol is null" : error);
        const auto query = reinterpret_cast<SymmetrixJitM0HostPluginQueryV1>(symbol);
        const auto* descriptor = query();
        if (descriptor == nullptr)
            throw plugin_error(path, "query returned a null descriptor");
        validate(path, *descriptor, expectation);
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
