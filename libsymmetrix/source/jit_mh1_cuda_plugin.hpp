#pragma once

#include "jit_mh1_cuda_plugin_abi.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace symmetrix::execution {

struct MH1CudaModuleState;

struct MH1CudaInteractionExpectation {
    std::int32_t input_1_dimension = -1;
    std::int32_t input_2_dimension = -1;
    std::int32_t output_dimension = -1;
    std::int32_t weight_size = -1;
    std::int32_t phi_dimension = -1;
    std::int32_t multiplicity = -1;
    std::int32_t input_1_angular_dimension = -1;
    std::int32_t instruction_count = -1;
};

struct MH1CudaPluginExpectation {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::array<MH1CudaInteractionExpectation,
        SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS> interactions;
    std::int32_t target_compute_capability = -1;
    std::int32_t max_threads_per_block = -1;
    std::uint32_t required_capabilities = 0;
};

struct MH1CudaNodeProgramExpectation {
    std::int32_t element_count = -1;
    std::int32_t input_dimension = -1;
    std::int32_t up_dimension = -1;
    std::int32_t residual_dimension = -1;
    std::int32_t skip_dimension = -1;
    std::int32_t message_dimension = -1;
    std::int32_t interaction_output_dimension = -1;
    std::int32_t output_dimension = -1;
    std::int32_t product_term_count = -1;
    std::int32_t node_arena_dimension = -1;
    bool requires_tp_source_state_adjoint = false;
    std::int64_t linear_parameter_count = -1;
    std::int64_t product_parameter_count = -1;
    std::int64_t readout_parameter_count = -1;
    std::array<std::int32_t,
        SYMMETRIX_JIT_MH1_CUDA_NODE_FORWARD_PHASES_V4>
        forward_threads_per_block{};
    std::array<std::int32_t,
        SYMMETRIX_JIT_MH1_CUDA_NODE_REVERSE_PHASES_V4>
        reverse_threads_per_block{};
};

struct MH1CudaPluginV4Expectation {
    std::string_view artifact_id;
    std::string_view generation_fingerprint;
    std::string_view semantic_fingerprint;
    std::string_view structure_fingerprint;
    std::string_view runtime_layout_fingerprint;
    std::array<MH1CudaInteractionExpectation,
        SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS> interactions;
    std::array<MH1CudaNodeProgramExpectation,
        SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS> node_programs;
    std::string_view target_backend = "cuda";
    std::string_view target_architecture;
    std::int32_t target_compute_capability = -1;
    std::int32_t max_threads_per_block = -1;
    std::uint32_t scalar_size = sizeof(float);
    std::uint32_t required_capabilities = 0;
};

class MH1CudaPlugin {
public:
    static MH1CudaPlugin load(
        std::string path,
        const MH1CudaPluginExpectation& expectation);

    MH1CudaPlugin() = default;
    // The owning CUDA stream must be fenced before destruction or replacement.
    ~MH1CudaPlugin();
    MH1CudaPlugin(const MH1CudaPlugin&) = delete;
    MH1CudaPlugin& operator=(const MH1CudaPlugin&) = delete;
    MH1CudaPlugin(MH1CudaPlugin&& other) noexcept;
    MH1CudaPlugin& operator=(MH1CudaPlugin&& other) noexcept;

    explicit operator bool() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitMH1CudaPluginV3& descriptor() const;

private:
    MH1CudaPlugin(
        void* handle,
        const SymmetrixJitMH1CudaPluginV3* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitMH1CudaPluginV3* descriptor_ = nullptr;
    std::string path_;
};

// Opt-in ABI-v4 handle. Existing evaluator code intentionally continues to
// load ABI v3 until generated node-program dispatch is available.
class MH1CudaPluginV4 {
public:
    static MH1CudaPluginV4 load(
        std::string path,
        const MH1CudaPluginV4Expectation& expectation);
    static MH1CudaPluginV4 load_cubin(
        std::string path,
        const MH1CudaPluginV4Expectation& expectation,
        std::string launch_plan_json);
    static MH1CudaPluginV4 load_module(
        std::string path,
        const MH1CudaPluginV4Expectation& expectation,
        std::string launch_plan_json);

    MH1CudaPluginV4();
    ~MH1CudaPluginV4();
    MH1CudaPluginV4(const MH1CudaPluginV4&) = delete;
    MH1CudaPluginV4& operator=(const MH1CudaPluginV4&) = delete;
    MH1CudaPluginV4(MH1CudaPluginV4&& other) noexcept;
    MH1CudaPluginV4& operator=(MH1CudaPluginV4&& other) noexcept;

    explicit operator bool() const noexcept
    {
        return handle_ != nullptr || module_state_ != nullptr;
    }
    const std::string& path() const noexcept { return path_; }
    const SymmetrixJitMH1CudaPluginV4& descriptor() const;
    std::int32_t launch_forward(
        const SymmetrixJitMH1CudaForwardArgsV4* args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_reverse(
        const SymmetrixJitMH1CudaReverseArgsV4* args,
        void* stream,
        std::int32_t source_persistent_blocks,
        std::int32_t edge_persistent_blocks) const;
    bool has_spline_r_program() const noexcept;
    std::int32_t launch_spline_r_forward(
        const SymmetrixJitMH1CudaSplineRForwardArgsV5* args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_spline_r_reverse(
        const SymmetrixJitMH1CudaSplineRSourceArgsV5* source_args,
        const SymmetrixJitMH1CudaSplineREdgeArgsV5* edge_args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_conditioning_forward(
        const SymmetrixJitMH1CudaConditioningForwardArgsV4* args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_conditioning_reverse(
        const SymmetrixJitMH1CudaConditioningReverseArgsV4* args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_node_forward(
        const SymmetrixJitMH1CudaNodeForwardArgsV4* args,
        void* stream,
        std::int32_t persistent_blocks) const;
    std::int32_t launch_node_reverse(
        const SymmetrixJitMH1CudaNodeReverseArgsV4* args,
        void* stream,
        std::int32_t persistent_blocks) const;

private:
    MH1CudaPluginV4(
        void* handle,
        const SymmetrixJitMH1CudaPluginV4* descriptor,
        std::string path) noexcept;
    void reset() noexcept;

    void* handle_ = nullptr;
    const SymmetrixJitMH1CudaPluginV4* descriptor_ = nullptr;
    std::unique_ptr<MH1CudaModuleState> module_state_;
    std::string path_;
};

}  // namespace symmetrix::execution
