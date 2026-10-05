#pragma once

// FP32 Metal execution of the R1 interaction behind the host-plugin packets.
// Evaluator views stay in Kokkos host memory; each launch stages its inputs
// into shared Metal buffers, converts double radii and geometry to FP32, and
// copies the outputs back. Spline coefficients are uploaded once per view.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "jit_host_plugin_abi.h"

namespace symmetrix::execution::metal {

class HostMemoryMap;

struct MetalR1Shape {
    std::int32_t channels = 0;
    std::int32_t edge_harmonics = 0;
    std::int32_t source_harmonics = 0;
    std::int32_t output_components = 0;
};

// Element counts of the views behind a packet. Pointer fields of the packet
// carry no extents, and the evaluator may pass views longer than the graph.
struct MetalR1ForwardExtents {
    std::size_t node_types = 0;
    std::size_t type_to_active = 0;
    std::size_t neighbor_feature_nodes = 0;
};

struct MetalR1ReverseExtents {
    std::size_t node_types = 0;
    std::size_t type_to_active = 0;
    std::size_t receivers = 0;
    std::size_t neighbor_feature_nodes = 0;
};

// Per-degree blocks of the A1 channel mixing. For degree l, rows
// [lme[l], lme[l] + (2l+1)*eta[l]) of each node's Phi1 form a
// (2l+1) x (eta[l]*channels) matrix multiplied by weights[l]
// [eta[l]*channels, channels]; weights_trans[l] is its transpose.
struct MetalA1Layout {
    std::int32_t l_max = -1;
    std::int32_t num_lme = 0;
    std::int32_t num_lm = 0;
    std::int32_t lme[4] = {};
    std::int32_t eta[4] = {};
    const float* weights[4] = {};
    const float* weights_trans[4] = {};
};

struct MetalA1Request {
    MetalA1Layout layout;
    float* a1 = nullptr;
};

struct MetalR1Statistics {
    std::uint64_t forward_launches = 0;
    std::uint64_t reverse_launches = 0;
    double gpu_seconds = 0.0;
    double staging_seconds = 0.0;
    // Per-kernel GPU time, recorded only when SYMMETRIX_METAL_PROFILE=1
    // submits each kernel in its own command buffer.
    double forward_seconds = 0.0;
    double source_seconds = 0.0;
    double edge_seconds = 0.0;
    std::uint64_t a1_forward_launches = 0;
    std::uint64_t a1_reverse_launches = 0;
    std::uint64_t row_scale_launches = 0;
    std::uint64_t fused_a1_launches = 0;
    std::uint64_t resident_uploads_skipped = 0;
    double a1_seconds = 0.0;
    std::uint64_t blocked_edge_launches = 0;
};

class MetalR1Module {
public:
    MetalR1Module(std::string_view msl_source, const MetalR1Shape& shape);
    ~MetalR1Module();
    MetalR1Module(const MetalR1Module&) = delete;
    MetalR1Module& operator=(const MetalR1Module&) = delete;

    const MetalR1Shape& shape() const;
    const std::string& device_name() const;
    const MetalR1Statistics& statistics() const;

    // Host ranges in this map are read and written in place.
    void set_host_memory(std::shared_ptr<const HostMemoryMap> host_memory);
    // Forgets device copies of weight and spline tables; call after the
    // evaluator rebuilds any of them.
    void release_persistent_tables();
    // Forgets results kept for a later stage of the same evaluation; the
    // evaluator calls this when an evaluation begins.
    void discard_pending_results();

    // With a1_request, the A1 GEMMs run in the same submission and the next
    // a1_forward for this Phi1 and A1 returns without a launch.
    void forward(
        const SymmetrixJitHostR1ForwardArgsV2& args,
        const MetalR1ForwardExtents& extents,
        const MetalA1Request* a1_request = nullptr);
    void reverse(
        const SymmetrixJitHostR1SourceArgsV2& source,
        const SymmetrixJitHostR1EdgeArgsV2& edge,
        const MetalR1ReverseExtents& extents);

    // A1 = Phi1 W from the Phi1 left on the GPU by the preceding forward.
    // Returns false, without side effects, when phi1 is not that output.
    bool a1_forward(
        const float* phi1, std::size_t num_nodes,
        const MetalA1Layout& layout, float* a1);

    // dPhi1 = dA1 W^T on the GPU. dPhi1 is copied to the host and kept
    // resident so the next reverse does not upload it again.
    // capacity_nodes is the leading extent of the dPhi1 view, which the
    // following reverse stages as its output adjoint.
    void a1_reverse(
        std::size_t num_nodes, std::size_t capacity_nodes,
        const MetalA1Layout& layout,
        const float* a1_adjoint, float* phi1_adjoint);

    // rows[node, :] *= factors[node] in place for host memory mapped for GPU
    // access, as the A1 density scaling does. Returns false, without side
    // effects, when rows are not mapped; a staged copy would cost more than
    // the host pass it replaces.
    bool scale_node_rows(
        float* rows, std::size_t num_nodes, std::size_t row_length,
        const float* factors);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace symmetrix::execution::metal
