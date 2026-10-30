#pragma once

#include "oam_cuda_plugin_abi.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace symmetrix::execution {

struct OAMCudaModuleState;

// Exact static schedule of one generated interaction layer. Every value is
// validated against the loaded descriptor before the handle is retained.
struct OAMCudaLayerExpectation {
    std::int32_t source_dimension = -1;
    std::int32_t edge_dimension = -1;
    std::int32_t weight_dimension = -1;
    std::int32_t output_dimension = -1;
    std::int32_t target_dimension = -1;
    std::int32_t instruction_count = -1;
    std::int32_t spline_interval_count = -1;
    std::int32_t spline_function_count = -1;
    std::int32_t edge_harmonic_count = -1;
};

// Identity and device extents a generated OAM CUDA artifact must match.
// `layers` is the full layer schedule in launch order; its size must equal the
// descriptor layer count. `required_capabilities` must contain both
// SYMMETRIX_OAM_CUDA_FORWARD_LAUNCH_V1 and
// SYMMETRIX_OAM_CUDA_REVERSE_LAUNCH_V1 and may not name unknown bits.
struct OAMCudaPluginExpectation {
    std::string_view artifact_id;
    std::string_view contract_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view target_architecture;
    std::vector<OAMCudaLayerExpectation> layers;
    std::int32_t target_compute_capability = -1;
    std::int32_t max_threads_per_block = -1;
    std::int32_t persistent_blocks_per_compute_unit = 4;
    std::uint32_t required_capabilities = 0;
    std::uint32_t scalar_kind = SYMMETRIX_OAM_CUDA_SCALAR_FLOAT32_V1;
    std::uint32_t scalar_size = sizeof(float);
    std::string_view target_policy;
    std::string_view precision_policy;
    std::string_view schedule_qualification;
};

// Move-only owner of one generated OAM CUDA shared library or cubin module.
// The handle is only retained after the descriptor, target, scalar kind, and
// architecture/semantic/structure fingerprints all match the expectation.
// The ABI retains the ``contract_fingerprint`` field name, but OAM generation
// stores the explicit generated-architecture fingerprint in that field.
class OAMCudaPlugin {
public:
    static OAMCudaPlugin load(
        std::string path,
        const OAMCudaPluginExpectation& expectation);
    static OAMCudaPlugin load_cubin(
        std::string path,
        const OAMCudaPluginExpectation& expectation);

    OAMCudaPlugin();
    // The owning CUDA stream must be fenced before destruction or replacement.
    ~OAMCudaPlugin();
    OAMCudaPlugin(const OAMCudaPlugin&) = delete;
    OAMCudaPlugin& operator=(const OAMCudaPlugin&) = delete;
    OAMCudaPlugin(OAMCudaPlugin&& other) noexcept;
    OAMCudaPlugin& operator=(OAMCudaPlugin&& other) noexcept;

    explicit operator bool() const noexcept
    {
        return handle_ != nullptr || module_state_ != nullptr;
    }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixOAMCudaPluginV3& descriptor() const;
    std::size_t layer_count() const noexcept;
    const SymmetrixOAMCudaLayerV2& layer(std::size_t layer_index) const;
    std::string_view artifact_id() const;
    std::string_view contract_fingerprint() const;
    std::string_view semantic_fingerprint() const;
    std::string_view structure_fingerprint() const;
    std::string_view target_architecture() const;
    std::int32_t persistent_blocks_per_compute_unit() const noexcept
    {
        return persistent_blocks_per_compute_unit_;
    }
    std::int32_t launch_forward(
        std::size_t layer_index,
        const SymmetrixOAMCudaLayerArgsV2* args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_reverse(
        std::size_t layer_index,
        const SymmetrixOAMCudaLayerArgsV2* args,
        void* stream,
        std::int32_t persistent_blocks) const;

private:
    OAMCudaPlugin(
        void* handle,
        const SymmetrixOAMCudaPluginV3* descriptor,
        std::string path,
        std::int32_t persistent_blocks_per_compute_unit) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixOAMCudaPluginV3* descriptor_ = nullptr;
    std::unique_ptr<OAMCudaModuleState> module_state_;
    std::string path_;
    std::int32_t persistent_blocks_per_compute_unit_ = 4;
};

}  // namespace symmetrix::execution
