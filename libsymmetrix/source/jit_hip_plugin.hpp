#pragma once

#include "jit_hip_plugin_abi.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace symmetrix::execution {

struct HipModuleState;

struct HipPluginExpectation {
    std::string_view artifact_id;
    std::string_view contract_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view target_architecture;
    std::string_view target_features;
    std::int32_t native_subgroup_width = -1;
    std::int32_t channels = -1;
    std::int32_t embedding = -1;
    std::int32_t edge_l_max = -1;
    std::int32_t source_l_max = -1;
    std::int32_t max_threads_per_block = -1;
    std::int32_t max_persistent_blocks_per_compute_unit = -1;
    std::uint32_t required_capabilities = 0;
    std::uint32_t scalar_kind = SYMMETRIX_JIT_HIP_SCALAR_FLOAT32_V1;
    std::uint32_t scalar_size = sizeof(float);
};

class HipPlugin {
public:
    static HipPlugin load(
        std::string path, const HipPluginExpectation& expectation);
    static HipPlugin load_module(
        std::string path, const HipPluginExpectation& expectation);

    HipPlugin() = default;
    ~HipPlugin();
    HipPlugin(const HipPlugin&) = delete;
    HipPlugin& operator=(const HipPlugin&) = delete;
    HipPlugin(HipPlugin&& other) noexcept;
    HipPlugin& operator=(HipPlugin&& other) noexcept;

    explicit operator bool() const noexcept
    {
        return handle_ != nullptr || module_state_ != nullptr;
    }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitHipPluginV1& descriptor() const;
    std::string_view artifact_id() const;
    std::string_view contract_fingerprint() const;
    std::int32_t persistent_blocks_per_compute_unit() const;
    std::int32_t r1_reverse_physical_launch_count() const noexcept;
    bool supports_tiled_r1() const noexcept;
    bool supports_projected_r1() const noexcept;
    std::int32_t launch_r1_forward(
        const SymmetrixJitHipR1ForwardArgsV1* args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_coordinate_reverse(
        const SymmetrixJitHipR1SourceArgsV1* source_args,
        const SymmetrixJitHipR1EdgeArgsV1* edge_args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_tiled_forward(
        const SymmetrixJitHipR1TiledForwardArgsV1* args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_tiled_reverse(
        const SymmetrixJitHipR1TiledSourceArgsV1* source_args,
        const SymmetrixJitHipR1TiledEdgeArgsV1* edge_args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_projected_forward(
        const SymmetrixJitHipR1ProjectedForwardArgsV1* args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_projected_reverse(
        const SymmetrixJitHipR1ProjectedReverseArgsV1* args,
        void* stream, std::int32_t persistent_blocks) const;

private:
    HipPlugin(
        void* handle, const SymmetrixJitHipPluginV1* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitHipPluginV1* descriptor_ = nullptr;
    std::unique_ptr<HipModuleState> module_state_;
    std::string path_;
};

} // namespace symmetrix::execution
