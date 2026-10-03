#pragma once

// FP32 Metal execution of the generated state-free M0 product basis. Kernels
// are rendered by symmetrix.metal_codegen.render_jit_m0_metal_source.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace symmetrix::execution::metal {

class HostMemoryMap;

struct MetalM0Shape {
    std::int32_t channels = 0;
    std::int32_t input_components = 0;
    std::int32_t output_components = 0;
    std::int32_t term_count = 0;
};

// A per-degree channel mixing applied to the M0 output in the same
// submission: output[node, lm, :] = m0[node, lm, :] weights[l(lm)], with
// weights [l_max+1, channels, channels] row-major.
struct MetalM0LinearRequest {
    std::int32_t l_max = -1;
    const float* weights = nullptr;
    float* output = nullptr;
};

struct MetalM0Statistics {
    std::uint64_t forward_launches = 0;
    std::uint64_t reverse_launches = 0;
    double gpu_seconds = 0.0;
    double staging_seconds = 0.0;
    std::uint64_t fused_linear_launches = 0;
};

class MetalM0Module {
public:
    MetalM0Module(std::string_view msl_source, const MetalM0Shape& shape);
    ~MetalM0Module();
    MetalM0Module(const MetalM0Module&) = delete;
    MetalM0Module& operator=(const MetalM0Module&) = delete;

    const MetalM0Shape& shape() const;
    const MetalM0Statistics& statistics() const;

    // Host ranges in this map are read and written in place.
    void set_host_memory(std::shared_ptr<const HostMemoryMap> host_memory);
    // Forgets device copies of weight and spline tables; call after the
    // evaluator rebuilds any of them.
    void release_persistent_tables();
    // Forgets results kept for a later stage of the same evaluation; the
    // evaluator calls this when an evaluation begins.
    void discard_pending_results();

    // input [nodes, input_components, channels], weights
    // [weight_types, term_count, channels], output [nodes, output_components,
    // channels], all row-major FP32.
    void forward(
        std::int64_t num_nodes,
        const std::int32_t* node_types,
        const float* input,
        const float* weights,
        std::size_t weight_count,
        float* output,
        const MetalM0LinearRequest* linear = nullptr);

    // True once, when the previous forward fused the linear request that
    // maps this output to linear_output for num_nodes nodes.
    bool take_fused_linear(
        const float* output, const float* linear_output, std::int64_t num_nodes);

    // Writes input_adjoint and, when requested, adds sum_c x*grad per node to
    // the FP64 input_scale_adjoint.
    void reverse(
        std::int64_t num_nodes,
        const std::int32_t* node_types,
        const float* input,
        const float* weights,
        std::size_t weight_count,
        const float* output_adjoint,
        float* input_adjoint,
        double* input_scale_adjoint,
        bool capture_input_scale_adjoint);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace symmetrix::execution::metal
