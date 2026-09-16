#pragma once

#include "jit_cuda_plugin_abi.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace symmetrix::execution {

struct CudaModuleState;

struct CudaPluginExpectation {
    std::string_view artifact_id;
    std::string_view contract_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::int32_t channels = -1;
    std::int32_t embedding = -1;
    std::int32_t edge_l_max = -1;
    std::int32_t source_l_max = -1;
    std::int32_t target_compute_capability = -1;
    std::int32_t max_threads_per_block = -1;
    std::int32_t persistent_blocks_per_compute_unit = 4;
    std::uint32_t required_capabilities = 0;
    std::uint32_t scalar_kind = SYMMETRIX_JIT_CUDA_SCALAR_FLOAT32_V2;
    std::uint32_t scalar_size = sizeof(float);
};

class CudaPlugin {
public:
    static CudaPlugin load(
        std::string path,
        const CudaPluginExpectation& expectation);
    static CudaPlugin load_cubin(
        std::string path,
        const CudaPluginExpectation& expectation);

    CudaPlugin();
    // The owning CUDA stream must be fenced before destruction or replacement.
    ~CudaPlugin();
    CudaPlugin(const CudaPlugin&) = delete;
    CudaPlugin& operator=(const CudaPlugin&) = delete;
    CudaPlugin(CudaPlugin&& other) noexcept;
    CudaPlugin& operator=(CudaPlugin&& other) noexcept;

    explicit operator bool() const noexcept
    {
        return handle_ != nullptr || module_state_ != nullptr;
    }
    const std::string& path() const noexcept { return path_; }
    std::uint32_t abi_version() const noexcept;
    bool has_v2_descriptor() const noexcept
    {
        return descriptor_v2_ != nullptr || module_state_ != nullptr;
    }
    const SymmetrixJitCudaPluginV1& descriptor() const;
    const SymmetrixJitCudaPluginV1& descriptor_v1() const;
    const SymmetrixJitCudaPluginV2& descriptor_v2() const;
    std::string_view artifact_id() const;
    std::string_view contract_fingerprint() const;
    std::int32_t persistent_blocks_per_compute_unit() const noexcept
    {
        return persistent_blocks_per_compute_unit_;
    }
    std::int32_t r1_reverse_physical_launch_count() const noexcept;
    bool supports_tiled_r1() const noexcept;
    bool supports_projected_r1() const noexcept;
    std::int32_t launch_r1_forward(
        const SymmetrixJitCudaR1ForwardArgsV2* args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_coordinate_reverse(
        const SymmetrixJitCudaR1SourceArgsV2* source_args,
        const SymmetrixJitCudaR1EdgeArgsV2* edge_args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_tiled_forward(
        const SymmetrixJitCudaR1TiledForwardArgsV2* args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_tiled_reverse(
        const SymmetrixJitCudaR1TiledSourceArgsV2* source_args,
        const SymmetrixJitCudaR1TiledEdgeArgsV2* edge_args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_projected_forward(
        const SymmetrixJitCudaR1ProjectedForwardArgsV2* args,
        void* stream, std::int32_t persistent_blocks) const;
    std::int32_t launch_r1_projected_reverse(
        const SymmetrixJitCudaR1ProjectedReverseArgsV2* args,
        void* stream, std::int32_t persistent_blocks) const;

private:
    CudaPlugin(
        void* handle,
        const SymmetrixJitCudaPluginV1* descriptor_v1,
        const SymmetrixJitCudaPluginV2* descriptor_v2,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitCudaPluginV1* descriptor_v1_ = nullptr;
    const SymmetrixJitCudaPluginV2* descriptor_v2_ = nullptr;
    std::unique_ptr<CudaModuleState> module_state_;
    std::string path_;
    std::int32_t persistent_blocks_per_compute_unit_ = 4;
};

}  // namespace symmetrix::execution
