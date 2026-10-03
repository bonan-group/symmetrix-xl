#pragma once

// Host mirror of the Metal R1 packets rendered by symmetrix.metal_codegen.
// Device pointers are 64-bit GPU addresses; padding is explicit so the host
// and MSL layouts agree member for member.

#include <cstddef>
#include <cstdint>

namespace symmetrix::execution::metal {

inline constexpr std::uint32_t metal_r1_abi_version = 1;

struct MetalR1RadialSpline {
    std::uint32_t edge_types;
    std::uint32_t intervals;
    std::uint32_t functions;
    float h;
    float x0;
    std::uint32_t reserved;
    std::uint64_t coefficients;
};
static_assert(sizeof(MetalR1RadialSpline) == 32);
static_assert(offsetof(MetalR1RadialSpline, coefficients) == 24);

struct MetalR1ForwardArgs {
    std::uint32_t active_type_count;
    std::uint32_t num_nodes;
    float cutoff;
    std::uint32_t reserved;
    std::uint64_t node_types;
    std::uint64_t num_neigh;
    std::uint64_t first_neigh;
    std::uint64_t neigh_indices;
    std::uint64_t neigh_types;
    std::uint64_t type_to_active;
    std::uint64_t radius;
    MetalR1RadialSpline radial;
    std::uint64_t harmonics_values;
    std::uint64_t neighbor_features;
    std::uint64_t output;
};
static_assert(sizeof(MetalR1ForwardArgs) == 128);
static_assert(offsetof(MetalR1ForwardArgs, radial) == 72);

struct MetalR1SourceArgs {
    std::uint32_t active_type_count;
    std::uint32_t num_nodes;
    float cutoff;
    std::uint32_t reserved;
    std::uint64_t node_types;
    std::uint64_t neigh_types;
    std::uint64_t source_offsets;
    std::uint64_t source_edges;
    std::uint64_t edge_receivers;
    std::uint64_t type_to_active;
    std::uint64_t radius;
    MetalR1RadialSpline radial;
    std::uint64_t harmonics_values;
    std::uint64_t output_adjoint;
    std::uint64_t source_adjoint;
};
static_assert(sizeof(MetalR1SourceArgs) == 128);
static_assert(offsetof(MetalR1SourceArgs, radial) == 72);

struct MetalR1EdgeArgs {
    std::uint32_t active_type_count;
    std::uint32_t num_edges;
    float cutoff;
    std::uint32_t has_gradients;
    std::uint64_t node_types;
    std::uint64_t neigh_indices;
    std::uint64_t neigh_types;
    std::uint64_t edge_receivers;
    std::uint64_t type_to_active;
    std::uint64_t unit_xyz;
    std::uint64_t radius;
    MetalR1RadialSpline radial;
    std::uint64_t harmonics_values;
    std::uint64_t harmonics_gradients;
    std::uint64_t neighbor_features;
    std::uint64_t output_adjoint;
    std::uint64_t directed_forces;
};
static_assert(sizeof(MetalR1EdgeArgs) == 144);
static_assert(offsetof(MetalR1EdgeArgs, radial) == 72);

}  // namespace symmetrix::execution::metal
