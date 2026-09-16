#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_HIP_PLUGIN_ABI_VERSION_V1 1u
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_HIP_PLUGIN_BYTE_ORDER 0x04030201u
#else
#define SYMMETRIX_JIT_HIP_PLUGIN_BYTE_ORDER 0x01020304u
#endif
#define SYMMETRIX_JIT_HIP_PLUGIN_ABI_TAG_V1 \
    "symmetrix.jit.hip-plugin/1"
#define SYMMETRIX_JIT_HIP_PLUGIN_QUERY_SYMBOL_V1 \
    "symmetrix_jit_hip_plugin_query_v1"
#if defined(__unix__) || defined(__APPLE__)
#define SYMMETRIX_JIT_HIP_PLUGIN_EXPORT \
    __attribute__((visibility("default")))
#else
#define SYMMETRIX_JIT_HIP_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SymmetrixJitHipScalarKindV1 {
    SYMMETRIX_JIT_HIP_SCALAR_FLOAT32_V1 = 1u,
    SYMMETRIX_JIT_HIP_SCALAR_FLOAT64_V1 = 2u,
};

enum SymmetrixJitHipPluginCapabilityV1 {
    SYMMETRIX_JIT_HIP_R1_FORWARD_LAUNCH_V1 = 1u << 0,
    SYMMETRIX_JIT_HIP_R1_COORDINATE_REVERSE_LAUNCH_V1 = 1u << 1,
    SYMMETRIX_JIT_HIP_R1_FUSED_REVERSE_LAUNCH_V1 = 1u << 2,
};

typedef struct SymmetrixJitHipRadialSplineV1 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const void* coefficients;
} SymmetrixJitHipRadialSplineV1;

// Tensor pointers address contiguous row-major HIP device storage. Argument
// structures reside in host memory and are copied into asynchronous launches.
typedef struct SymmetrixJitHipR1ForwardArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* num_neigh;
    const int32_t* first_neigh;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* type_to_active;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    void* output;
    double cutoff;
} SymmetrixJitHipR1ForwardArgsV1;

typedef struct SymmetrixJitHipR1SourceArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* neigh_types;
    const int32_t* source_offsets;
    const int32_t* source_edges;
    const int32_t* edge_receivers;
    const int32_t* type_to_active;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* output_adjoint;
    void* source_adjoint;
    double cutoff;
} SymmetrixJitHipR1SourceArgsV1;

// Module-only experimental packets. They do not alter the shared-plugin ABI.
typedef struct SymmetrixJitHipR1TiledForwardArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    uint32_t channel_begin;
    uint32_t channel_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* num_neigh;
    const int32_t* first_neigh;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* type_to_active;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    void* output;
    double cutoff;
} SymmetrixJitHipR1TiledForwardArgsV1;

typedef struct SymmetrixJitHipR1TiledSourceArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    uint32_t channel_begin;
    uint32_t channel_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* neigh_types;
    const int32_t* source_offsets;
    const int32_t* source_edges;
    const int32_t* edge_receivers;
    const int32_t* type_to_active;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* output_adjoint;
    void* source_adjoint;
    double cutoff;
    int64_t source_owner_count;
    const int32_t* source_ids;
} SymmetrixJitHipR1TiledSourceArgsV1;

typedef struct SymmetrixJitHipR1TiledEdgeArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    uint32_t coordinate_scalar_size;
    uint32_t coordinates_are_unit;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* edge_receivers;
    const int32_t* type_to_active;
    const void* xyz;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
    double cutoff;
} SymmetrixJitHipR1TiledEdgeArgsV1;

typedef struct SymmetrixJitHipR1ProjectedForwardArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* num_neigh;
    const int32_t* first_neigh;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* type_to_active;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    const void* projection_weights;
    void* output;
    double cutoff;
} SymmetrixJitHipR1ProjectedForwardArgsV1;

typedef struct SymmetrixJitHipR1ProjectedReverseArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    uint32_t coordinate_scalar_size;
    uint32_t coordinates_are_unit;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* num_neigh;
    const int32_t* first_neigh;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* type_to_active;
    const void* xyz;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    void* source_adjoint;
    double* directed_forces;
    const void* projection_weights;
    double cutoff;
} SymmetrixJitHipR1ProjectedReverseArgsV1;

typedef struct SymmetrixJitHipR1EdgeArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    uint32_t coordinate_scalar_size;
    uint32_t coordinates_are_unit;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* edge_receivers;
    const int32_t* type_to_active;
    const void* xyz;
    const double* radius;
    SymmetrixJitHipRadialSplineV1 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
    double cutoff;
} SymmetrixJitHipR1EdgeArgsV1;

// `stream` is the opaque hipStream_t owned by the evaluator. Launch functions
// return the immediate numeric hipError_t and retain no host argument pointer.
typedef int32_t (*SymmetrixJitHipR1ForwardLaunchV1)(
    const SymmetrixJitHipR1ForwardArgsV1*, void*, int32_t);
typedef int32_t (*SymmetrixJitHipR1CoordinateReverseLaunchV1)(
    const SymmetrixJitHipR1SourceArgsV1*,
    const SymmetrixJitHipR1EdgeArgsV1*,
    void*,
    int32_t);

typedef struct SymmetrixJitHipPluginV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t scalar_kind;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* contract_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    int32_t channels;
    int32_t embedding;
    int32_t edge_l_max;
    int32_t source_l_max;
    const char* target_architecture;
    const char* target_features;
    int32_t native_subgroup_width;
    int32_t forward_threads_per_block;
    int32_t source_threads_per_block;
    int32_t edge_threads_per_block;
    // Maximum resident-style grid capacity requested per compute unit.
    int32_t persistent_blocks_per_compute_unit;
    SymmetrixJitHipR1ForwardLaunchV1 r1_forward_launch;
    SymmetrixJitHipR1CoordinateReverseLaunchV1
        r1_coordinate_reverse_launch;
} SymmetrixJitHipPluginV1;

typedef const SymmetrixJitHipPluginV1*
    (*SymmetrixJitHipPluginQueryV1)(void);

SYMMETRIX_JIT_HIP_PLUGIN_EXPORT
const SymmetrixJitHipPluginV1* symmetrix_jit_hip_plugin_query_v1(void);

#ifdef __cplusplus
}
#endif
