#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION 1u
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_HOST_PLUGIN_BYTE_ORDER 0x04030201u
#else
#define SYMMETRIX_JIT_HOST_PLUGIN_BYTE_ORDER 0x01020304u
#endif
#define SYMMETRIX_JIT_HOST_PLUGIN_ABI_TAG \
    "symmetrix.jit.host-plugin/1"
#define SYMMETRIX_JIT_HOST_PLUGIN_QUERY_SYMBOL \
    "symmetrix_jit_host_plugin_query_v1"
#define SYMMETRIX_JIT_HOST_PLUGIN_ABI_VERSION_V2 2u
#define SYMMETRIX_JIT_HOST_PLUGIN_ABI_TAG_V2 \
    "symmetrix.jit.host-plugin/2"
#define SYMMETRIX_JIT_HOST_PLUGIN_QUERY_SYMBOL_V2 \
    "symmetrix_jit_host_plugin_query_v2"
#if defined(__unix__) || defined(__APPLE__)
#define SYMMETRIX_JIT_HOST_PLUGIN_EXPORT \
    __attribute__((visibility("default")))
#else
#define SYMMETRIX_JIT_HOST_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SymmetrixJitHostPluginCapabilityV1 {
    SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V1 = 1u << 0,
    SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V1 = 1u << 1,
    SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V1 = 1u << 2,
    SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V1 = 1u << 3,
};

typedef struct SymmetrixJitHostRadialSplineV1 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const float* coefficients;
} SymmetrixJitHostRadialSplineV1;

// Tensor pointers use contiguous row-major (Kokkos LayoutRight) storage.
// Radial coefficients are [edge_types, intervals, 4, functions].
typedef struct SymmetrixJitHostR1ForwardArgsV1 {
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
    SymmetrixJitHostRadialSplineV1 radial;
    const float* harmonics_values;
    const float* neighbor_features;
    float* output;
    double cutoff;
} SymmetrixJitHostR1ForwardArgsV1;

typedef struct SymmetrixJitHostR1SourceArgsV1 {
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
    SymmetrixJitHostRadialSplineV1 radial;
    const float* harmonics_values;
    const float* output_adjoint;
    float* source_adjoint;
    double cutoff;
} SymmetrixJitHostR1SourceArgsV1;

typedef struct SymmetrixJitHostR1EdgeArgsV1 {
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
    SymmetrixJitHostRadialSplineV1 radial;
    const float* harmonics_values;
    const float* harmonics_gradients;
    const float* neighbor_features;
    const float* output_adjoint;
    double* directed_forces;
    double cutoff;
} SymmetrixJitHostR1EdgeArgsV1;

// Owner functions may run concurrently. They must not throw across this ABI,
// retain an argument pointer, or write outside the owner's unique output.
typedef void (*SymmetrixJitHostR1ForwardOwnerV1)(
    const SymmetrixJitHostR1ForwardArgsV1*, int32_t, int32_t);
typedef void (*SymmetrixJitHostR1SourceOwnerV1)(
    const SymmetrixJitHostR1SourceArgsV1*, int32_t, int32_t);
typedef void (*SymmetrixJitHostR1EdgeOwnerV1)(
    const SymmetrixJitHostR1EdgeArgsV1*, int32_t);

typedef struct SymmetrixJitHostPluginV1 {
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
    SymmetrixJitHostR1ForwardOwnerV1 r1_forward_owner;
    SymmetrixJitHostR1SourceOwnerV1 r1_source_owner;
    SymmetrixJitHostR1SourceOwnerV1 r1_compensated_source_owner;
    SymmetrixJitHostR1EdgeOwnerV1 r1_edge_owner;
} SymmetrixJitHostPluginV1;

typedef const SymmetrixJitHostPluginV1*
    (*SymmetrixJitHostPluginQueryV1)(void);

SYMMETRIX_JIT_HOST_PLUGIN_EXPORT
const SymmetrixJitHostPluginV1* symmetrix_jit_host_plugin_query_v1(void);

enum SymmetrixJitHostScalarKindV2 {
    SYMMETRIX_JIT_HOST_SCALAR_FLOAT32_V2 = 1u,
    SYMMETRIX_JIT_HOST_SCALAR_FLOAT64_V2 = 2u,
};

enum SymmetrixJitHostPluginCapabilityV2 {
    SYMMETRIX_JIT_HOST_R1_FORWARD_OWNER_V2 = 1u << 0,
    SYMMETRIX_JIT_HOST_R1_SOURCE_OWNER_V2 = 1u << 1,
    SYMMETRIX_JIT_HOST_R1_COMPENSATED_SOURCE_OWNER_V2 = 1u << 2,
    SYMMETRIX_JIT_HOST_R1_EDGE_OWNER_V2 = 1u << 3,
    SYMMETRIX_JIT_HOST_R1_SOURCE_CHANNEL_TILE_16_V2 = 1u << 4,
    SYMMETRIX_JIT_HOST_R1_FORWARD_CHANNEL_TILE_16_V2 = 1u << 5,
    SYMMETRIX_JIT_HOST_R1_SOURCE_CHANNEL_TILE_32_V2 = 1u << 6,
};

typedef struct SymmetrixJitHostRadialSplineV2 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const void* coefficients;
} SymmetrixJitHostRadialSplineV2;

typedef struct SymmetrixJitHostR1ForwardArgsV2 {
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
    SymmetrixJitHostRadialSplineV2 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    void* output;
    double cutoff;
} SymmetrixJitHostR1ForwardArgsV2;

typedef struct SymmetrixJitHostR1SourceArgsV2 {
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
    SymmetrixJitHostRadialSplineV2 radial;
    const void* harmonics_values;
    const void* output_adjoint;
    void* source_adjoint;
    double cutoff;
} SymmetrixJitHostR1SourceArgsV2;

typedef struct SymmetrixJitHostR1EdgeArgsV2 {
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
    SymmetrixJitHostRadialSplineV2 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
    double cutoff;
} SymmetrixJitHostR1EdgeArgsV2;

typedef void (*SymmetrixJitHostR1ForwardOwnerV2)(
    const SymmetrixJitHostR1ForwardArgsV2*, int32_t, int32_t);
typedef void (*SymmetrixJitHostR1SourceOwnerV2)(
    const SymmetrixJitHostR1SourceArgsV2*, int32_t, int32_t);
typedef void (*SymmetrixJitHostR1EdgeOwnerV2)(
    const SymmetrixJitHostR1EdgeArgsV2*, int32_t);

typedef struct SymmetrixJitHostPluginV2 {
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
    SymmetrixJitHostR1ForwardOwnerV2 r1_forward_owner;
    SymmetrixJitHostR1SourceOwnerV2 r1_source_owner;
    SymmetrixJitHostR1SourceOwnerV2 r1_compensated_source_owner;
    SymmetrixJitHostR1EdgeOwnerV2 r1_edge_owner;
} SymmetrixJitHostPluginV2;

typedef const SymmetrixJitHostPluginV2*
    (*SymmetrixJitHostPluginQueryV2)(void);

SYMMETRIX_JIT_HOST_PLUGIN_EXPORT
const SymmetrixJitHostPluginV2* symmetrix_jit_host_plugin_query_v2(void);

#ifdef __cplusplus
}
#endif
