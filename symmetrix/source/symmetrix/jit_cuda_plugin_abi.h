#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION 1u
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_CUDA_PLUGIN_BYTE_ORDER 0x04030201u
#else
#define SYMMETRIX_JIT_CUDA_PLUGIN_BYTE_ORDER 0x01020304u
#endif
#define SYMMETRIX_JIT_CUDA_PLUGIN_ABI_TAG \
    "symmetrix.jit.cuda-plugin/1"
#define SYMMETRIX_JIT_CUDA_PLUGIN_QUERY_SYMBOL \
    "symmetrix_jit_cuda_plugin_query_v1"
#define SYMMETRIX_JIT_CUDA_PLUGIN_ABI_VERSION_V2 2u
#define SYMMETRIX_JIT_CUDA_PLUGIN_ABI_TAG_V2 \
    "symmetrix.jit.cuda-plugin/2"
#define SYMMETRIX_JIT_CUDA_PLUGIN_QUERY_SYMBOL_V2 \
    "symmetrix_jit_cuda_plugin_query_v2"
#if defined(__unix__) || defined(__APPLE__)
#define SYMMETRIX_JIT_CUDA_PLUGIN_EXPORT \
    __attribute__((visibility("default")))
#else
#define SYMMETRIX_JIT_CUDA_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SymmetrixJitCudaPluginCapabilityV1 {
    SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V1 = 1u << 0,
    SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V1 = 1u << 1,
};

typedef struct SymmetrixJitCudaRadialSplineV1 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const float* coefficients;
} SymmetrixJitCudaRadialSplineV1;

// Tensor pointers address contiguous row-major CUDA device storage. The
// argument structures themselves reside in host memory and are copied by value
// into each asynchronous kernel launch.
typedef struct SymmetrixJitCudaR1ForwardArgsV1 {
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
    SymmetrixJitCudaRadialSplineV1 radial;
    const float* harmonics_values;
    const float* neighbor_features;
    float* output;
    double cutoff;
} SymmetrixJitCudaR1ForwardArgsV1;

typedef struct SymmetrixJitCudaR1SourceArgsV1 {
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
    SymmetrixJitCudaRadialSplineV1 radial;
    const float* harmonics_values;
    const float* output_adjoint;
    float* source_adjoint;
    double cutoff;
} SymmetrixJitCudaR1SourceArgsV1;

typedef struct SymmetrixJitCudaR1EdgeArgsV1 {
    uint32_t struct_size;
    uint32_t active_type_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* edge_receivers;
    const int32_t* type_to_active;
    const double* xyz;
    const double* radius;
    SymmetrixJitCudaRadialSplineV1 radial;
    const float* harmonics_values;
    const float* harmonics_gradients;
    const float* neighbor_features;
    const float* output_adjoint;
    double* directed_forces;
    double cutoff;
} SymmetrixJitCudaR1EdgeArgsV1;

// `stream` is the opaque cudaStream_t used by the owning evaluator.
// Launch functions enqueue work, return the immediate numeric cudaError_t, and
// must not retain any argument or stream pointer.
typedef int32_t (*SymmetrixJitCudaR1ForwardLaunchV1)(
    const SymmetrixJitCudaR1ForwardArgsV1*, void*, int32_t);
typedef int32_t (*SymmetrixJitCudaR1CoordinateReverseLaunchV1)(
    const SymmetrixJitCudaR1SourceArgsV1*,
    const SymmetrixJitCudaR1EdgeArgsV1*,
    void*,
    int32_t);

typedef struct SymmetrixJitCudaPluginV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
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
    int32_t target_compute_capability;
    int32_t forward_threads_per_block;
    int32_t source_threads_per_block;
    int32_t edge_threads_per_block;
    SymmetrixJitCudaR1ForwardLaunchV1 r1_forward_launch;
    SymmetrixJitCudaR1CoordinateReverseLaunchV1
        r1_coordinate_reverse_launch;
} SymmetrixJitCudaPluginV1;

typedef const SymmetrixJitCudaPluginV1*
    (*SymmetrixJitCudaPluginQueryV1)(void);

SYMMETRIX_JIT_CUDA_PLUGIN_EXPORT
const SymmetrixJitCudaPluginV1* symmetrix_jit_cuda_plugin_query_v1(void);

enum SymmetrixJitCudaScalarKindV2 {
    SYMMETRIX_JIT_CUDA_SCALAR_FLOAT32_V2 = 1u,
    SYMMETRIX_JIT_CUDA_SCALAR_FLOAT64_V2 = 2u,
};

enum SymmetrixJitCudaPluginCapabilityV2 {
    SYMMETRIX_JIT_CUDA_R1_FORWARD_LAUNCH_V2 = 1u << 0,
    SYMMETRIX_JIT_CUDA_R1_COORDINATE_REVERSE_LAUNCH_V2 = 1u << 1,
    SYMMETRIX_JIT_CUDA_R1_FUSED_REVERSE_LAUNCH_V2 = 1u << 2,
};

typedef struct SymmetrixJitCudaRadialSplineV2 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const void* coefficients;
} SymmetrixJitCudaRadialSplineV2;

typedef struct SymmetrixJitCudaR1ForwardArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    void* output;
    double cutoff;
} SymmetrixJitCudaR1ForwardArgsV2;

typedef struct SymmetrixJitCudaR1SourceArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* output_adjoint;
    void* source_adjoint;
    double cutoff;
} SymmetrixJitCudaR1SourceArgsV2;

// Module-only experimental packets. They do not alter the shared-plugin ABI.
typedef struct SymmetrixJitCudaR1TiledForwardArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    void* output;
    double cutoff;
} SymmetrixJitCudaR1TiledForwardArgsV2;

typedef struct SymmetrixJitCudaR1TiledSourceArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* output_adjoint;
    void* source_adjoint;
    double cutoff;
    int64_t source_owner_count;
    const int32_t* source_ids;
} SymmetrixJitCudaR1TiledSourceArgsV2;

typedef struct SymmetrixJitCudaR1TiledEdgeArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
    double cutoff;
} SymmetrixJitCudaR1TiledEdgeArgsV2;

typedef struct SymmetrixJitCudaR1ProjectedForwardArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    const void* projection_weights;
    void* output;
    double cutoff;
} SymmetrixJitCudaR1ProjectedForwardArgsV2;

typedef struct SymmetrixJitCudaR1ProjectedReverseArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    void* source_adjoint;
    double* directed_forces;
    const void* projection_weights;
    double cutoff;
} SymmetrixJitCudaR1ProjectedReverseArgsV2;

typedef struct SymmetrixJitCudaR1EdgeArgsV2 {
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
    SymmetrixJitCudaRadialSplineV2 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
    double cutoff;
} SymmetrixJitCudaR1EdgeArgsV2;

typedef int32_t (*SymmetrixJitCudaR1ForwardLaunchV2)(
    const SymmetrixJitCudaR1ForwardArgsV2*, void*, int32_t);
typedef int32_t (*SymmetrixJitCudaR1CoordinateReverseLaunchV2)(
    const SymmetrixJitCudaR1SourceArgsV2*,
    const SymmetrixJitCudaR1EdgeArgsV2*,
    void*,
    int32_t);

typedef struct SymmetrixJitCudaPluginV2 {
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
    int32_t target_compute_capability;
    int32_t forward_threads_per_block;
    int32_t source_threads_per_block;
    int32_t edge_threads_per_block;
    SymmetrixJitCudaR1ForwardLaunchV2 r1_forward_launch;
    SymmetrixJitCudaR1CoordinateReverseLaunchV2
        r1_coordinate_reverse_launch;
} SymmetrixJitCudaPluginV2;

typedef const SymmetrixJitCudaPluginV2*
    (*SymmetrixJitCudaPluginQueryV2)(void);

SYMMETRIX_JIT_CUDA_PLUGIN_EXPORT
const SymmetrixJitCudaPluginV2* symmetrix_jit_cuda_plugin_query_v2(void);

#ifdef __cplusplus
}
#endif
