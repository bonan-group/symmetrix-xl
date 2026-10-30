#pragma once

#include "mace_jit_r0_host_plugin_abi.h"
#include "mace_jit_r0_contract.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace symmetrix::execution {

using R0HostPluginExpectation = R0LogicalDescriptor;

class R0HostPlugin {
public:
    static R0HostPlugin load(
        std::string path, const R0HostPluginExpectation& expectation);

    R0HostPlugin() = default;
    ~R0HostPlugin();
    R0HostPlugin(const R0HostPlugin&) = delete;
    R0HostPlugin& operator=(const R0HostPlugin&) = delete;
    R0HostPlugin(R0HostPlugin&& other) noexcept;
    R0HostPlugin& operator=(R0HostPlugin&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitR0HostPluginV2& descriptor() const;

private:
    R0HostPlugin(
        void* handle, const SymmetrixJitR0HostPluginV2* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitR0HostPluginV2* descriptor_ = nullptr;
    std::string path_;
};

}  // namespace symmetrix::execution
