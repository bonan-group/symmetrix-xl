#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_R0_HOST_PLUGIN_ABI_VERSION_V2 2u
#define SYMMETRIX_JIT_R0_HOST_PLUGIN_ABI_TAG_V2 \
    "symmetrix.jit.r0-host-plugin/2"
#define SYMMETRIX_JIT_R0_HOST_PLUGIN_QUERY_SYMBOL_V2 \
    "symmetrix_jit_r0_host_plugin_query_v2"
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_R0_HOST_PLUGIN_BYTE_ORDER_V1 0x04030201u
#else
#define SYMMETRIX_JIT_R0_HOST_PLUGIN_BYTE_ORDER_V1 0x01020304u
#endif

#define SYMMETRIX_JIT_R0_HOST_SCALAR_FLOAT32_V1 1u
#define SYMMETRIX_JIT_R0_HOST_SCALAR_FLOAT64_V1 2u
#define SYMMETRIX_JIT_R0_HOST_DENSITY_PREPARE_V1 (1u << 0)
#define SYMMETRIX_JIT_R0_HOST_FORWARD_V1 (1u << 1)
#define SYMMETRIX_JIT_R0_HOST_REVERSE_PREPARE_V1 (1u << 2)
#define SYMMETRIX_JIT_R0_HOST_COORDINATE_REVERSE_V1 (1u << 3)
#define SYMMETRIX_JIT_R0_HOST_FORWARD_CHANNEL_TILED_V1 (1u << 4)
#define SYMMETRIX_JIT_R0_HOST_COORDINATE_REVERSE_RECEIVER_V1 (1u << 5)
#define SYMMETRIX_JIT_R0_HOST_COMPACT_GEOMETRY_V1 (1u << 6)
#define SYMMETRIX_JIT_R0_HOST_PRECOMPUTED_SCALE_V1 (1u << 7)
#define SYMMETRIX_JIT_R0_HOST_RECEIVER_BATCH_V1 (1u << 8)
#define SYMMETRIX_JIT_R0_HOST_FORWARD_CHANNEL_TILE_V1 8
#define SYMMETRIX_JIT_R0_HOST_COORDINATE_REVERSE_CHANNEL_TILE_V1 128
#define SYMMETRIX_JIT_R0_HOST_OWNER_BATCH_V2 8

typedef struct SymmetrixJitR0HostSplineV1 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const void* coefficients;
} SymmetrixJitR0HostSplineV1;

typedef struct SymmetrixJitR0HostArgsV1 {
    uint32_t struct_size;
    uint32_t receiver_base;
    int64_t num_nodes;
    int64_t num_edges;
    int32_t active_type_count;
    int32_t channels;
    int32_t l_max;
    int32_t coordinates_are_unit;
    int32_t apply_density_scale;
    int32_t use_precomputed_scale_adjoint;
    const int32_t* node_types;
    const int32_t* num_neigh;
    const int32_t* first_neigh;
    const int32_t* neigh_types;
    const int32_t* edge_receivers;
    const int32_t* type_to_active;
    const void* coordinates;
    const double* radius;
    SymmetrixJitR0HostSplineV1 radial;
    SymmetrixJitR0HostSplineV1 density;
    const void* harmonics;
    const void* harmonic_gradients;
    void* output;
    void* output_adjoint;
    double* density_state;
    const double* precomputed_scale_adjoint;
    double* directed_forces;
    double cutoff;
    const double* edge_weights;
    int64_t edge_begin;
} SymmetrixJitR0HostArgsV1;

typedef void (*SymmetrixJitR0HostOwnerV1)(
    const SymmetrixJitR0HostArgsV1*, int64_t);
typedef void (*SymmetrixJitR0HostRangeV2)(
    const SymmetrixJitR0HostArgsV1*, int64_t, int64_t);

typedef struct SymmetrixJitR0HostPluginV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t scalar_kind;
    uint32_t scalar_size;
    uint32_t capabilities;
    int32_t channels;
    int32_t l_max;
    int32_t harmonic_count;
    int32_t edge_weight_mode;
    const char* abi_tag;
    const char* artifact_id;
    const char* structure_fingerprint;
    SymmetrixJitR0HostOwnerV1 density_prepare_owner;
    SymmetrixJitR0HostOwnerV1 forward_owner;
    SymmetrixJitR0HostOwnerV1 reverse_prepare_owner;
    SymmetrixJitR0HostOwnerV1 coordinate_reverse_owner;
    const char* target;
    const char* schedule;
    int32_t schedule_revision;
    uint32_t compatibility_version;
    SymmetrixJitR0HostRangeV2 density_prepare_range;
    SymmetrixJitR0HostRangeV2 forward_range;
    SymmetrixJitR0HostRangeV2 reverse_prepare_range;
    SymmetrixJitR0HostRangeV2 coordinate_reverse_range;
} SymmetrixJitR0HostPluginV2;

typedef const SymmetrixJitR0HostPluginV2*
    (*SymmetrixJitR0HostPluginQueryV2)(void);
