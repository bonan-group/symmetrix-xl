#pragma once

#include "jit_host_plugin_abi.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace symmetrix::execution {

struct HostPluginExpectation {
    std::string_view artifact_id;
    std::string_view contract_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::int32_t channels = -1;
    std::int32_t embedding = -1;
    std::int32_t edge_l_max = -1;
    std::int32_t source_l_max = -1;
    std::uint32_t required_capabilities = 0;
    std::uint32_t scalar_kind = SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2;
    std::uint32_t scalar_size = sizeof(float);
};

class HostPlugin {
public:
    static HostPlugin load(
        std::string path,
        const HostPluginExpectation& expectation);

    HostPlugin() = default;
    // The caller must fence its scheduler before destruction or move-assignment;
    // no owner function may remain in flight when its library is unloaded.
    ~HostPlugin();
    HostPlugin(const HostPlugin&) = delete;
    HostPlugin& operator=(const HostPlugin&) = delete;
    HostPlugin(HostPlugin&& other) noexcept;
    HostPlugin& operator=(HostPlugin&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    std::uint32_t abi_version() const noexcept;
    bool has_v2_descriptor() const noexcept { return descriptor_v2_ != nullptr; }
    const SymmetrixJitHostPluginV1& descriptor() const;
    const SymmetrixJitHostPluginV1& descriptor_v1() const;
    const SymmetrixJitHostPluginV2& descriptor_v2() const;

private:
    HostPlugin(
        void* handle,
        const SymmetrixJitHostPluginV1* descriptor_v1,
        const SymmetrixJitHostPluginV2* descriptor_v2,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitHostPluginV1* descriptor_v1_ = nullptr;
    const SymmetrixJitHostPluginV2* descriptor_v2_ = nullptr;
    std::string path_;
};

}  // namespace symmetrix::execution
