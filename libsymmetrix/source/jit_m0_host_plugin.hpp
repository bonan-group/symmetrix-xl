#pragma once

#include "jit_m0_host_plugin_abi.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace symmetrix::execution {

struct M0HostPluginExpectation {
    std::string_view artifact_id;
    std::string_view structure_fingerprint;
    std::uint32_t scalar_kind = 0;
    std::uint32_t scalar_size = 0;
    std::int32_t channels = 0;
    std::int32_t input_components = 0;
    std::int32_t output_components = 0;
    std::int32_t correlation = 0;
    std::int32_t term_count = 0;
};

class M0HostPlugin {
public:
    static M0HostPlugin load(
        std::string path, const M0HostPluginExpectation& expectation);

    M0HostPlugin() = default;
    ~M0HostPlugin();
    M0HostPlugin(const M0HostPlugin&) = delete;
    M0HostPlugin& operator=(const M0HostPlugin&) = delete;
    M0HostPlugin(M0HostPlugin&& other) noexcept;
    M0HostPlugin& operator=(M0HostPlugin&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitM0HostPluginV1& descriptor() const;

private:
    M0HostPlugin(
        void* handle, const SymmetrixJitM0HostPluginV1* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitM0HostPluginV1* descriptor_ = nullptr;
    std::string path_;
};

}  // namespace symmetrix::execution
