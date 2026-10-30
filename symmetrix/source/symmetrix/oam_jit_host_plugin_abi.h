#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION 1u
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_BYTE_ORDER 0x04030201u
#else
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_BYTE_ORDER 0x01020304u
#endif
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG \
    "symmetrix.jit.oam.host-plugin/1"
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL \
    "symmetrix_jit_oam_host_plugin_query_v1"

#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V2 2u
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V2 \
    "symmetrix.jit.oam.host-plugin/2"
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V2 \
    "symmetrix_jit_oam_host_plugin_query_v2"

#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V3 3u
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V3 \
    "symmetrix.jit.oam.host-plugin/3"
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V3 \
    "symmetrix_jit_oam_host_plugin_query_v3"

#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V4 4u
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V4 \
    "symmetrix.jit.oam.host-plugin/4"
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V4 \
    "symmetrix_jit_oam_host_plugin_query_v4"

#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V5 5u
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V5 \
    "symmetrix.jit.oam.host-plugin/5"
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V5 \
    "symmetrix_jit_oam_host_plugin_query_v5"

#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V6 6u
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V6 \
    "symmetrix.jit.oam.host-plugin/6"
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V6 \
    "symmetrix_jit_oam_host_plugin_query_v6"

#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_VERSION_V7 7u
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_ABI_TAG_V7 \
    "symmetrix.jit.oam.host-plugin/7"
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_QUERY_SYMBOL_V7 \
    "symmetrix_jit_oam_host_plugin_query_v7"

#if defined(__unix__) || defined(__APPLE__)
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT \
    __attribute__((visibility("default")))
#else
#define SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SymmetrixJitOAMHostPluginCapabilityV1 {
    SYMMETRIX_JIT_OAM_HOST_FORWARD_OWNER_V1 = 1u << 0,
    SYMMETRIX_JIT_OAM_HOST_REVERSE_OWNER_V1 = 1u << 1,
};

// All tensor pointers use contiguous row-major storage. The callback receives
// one edge index in [0, samples) and owns only the corresponding output rows.
// first_edge allows streamed chunks to retain global edge indexing while the
// callback index remains local to the packet.
typedef struct SymmetrixJitOAMHostForwardArgsV1 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t scalar_size;
    uint32_t reserved;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    const int32_t* source_indices;
    const void* source_features;
    const void* edge_features;
    const void* edge_weights;
    void* edge_messages;
} SymmetrixJitOAMHostForwardArgsV1;

typedef struct SymmetrixJitOAMHostReverseArgsV1 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t scalar_size;
    uint32_t reserved;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    const int32_t* source_indices;
    const void* source_features;
    const void* edge_features;
    const void* edge_weights;
    const void* edge_output_adjoint;
    void* source_adjoint;
    void* edge_feature_adjoint;
    void* edge_weight_adjoint;
} SymmetrixJitOAMHostReverseArgsV1;

typedef void (*SymmetrixJitOAMHostForwardOwnerV1)(
    const SymmetrixJitOAMHostForwardArgsV1*, int32_t edge);
typedef void (*SymmetrixJitOAMHostReverseOwnerV1)(
    const SymmetrixJitOAMHostReverseArgsV1*, int32_t edge);

typedef struct SymmetrixJitOAMHostLayerV1 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t reserved;
    uint32_t reserved_2;
    int32_t source_dimension;
    int32_t edge_dimension;
    int32_t weight_dimension;
    int32_t output_dimension;
    int32_t instruction_count;
} SymmetrixJitOAMHostLayerV1;

typedef struct SymmetrixJitOAMHostPluginV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t layer_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const SymmetrixJitOAMHostLayerV1* layers;
    const SymmetrixJitOAMHostForwardOwnerV1* forward_owners;
    const SymmetrixJitOAMHostReverseOwnerV1* reverse_owners;
} SymmetrixJitOAMHostPluginV1;

typedef const SymmetrixJitOAMHostPluginV1*
    (*SymmetrixJitOAMHostPluginQueryV1)(void);

SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV1*
symmetrix_jit_oam_host_plugin_query_v1(void);

enum SymmetrixJitOAMHostPluginCapabilityV2 {
    SYMMETRIX_JIT_OAM_HOST_RECEIVER_FORWARD_V2 = 1u << 0,
    SYMMETRIX_JIT_OAM_HOST_SOURCE_REVERSE_V2 = 1u << 1,
    /* Bit 1u<<2 is unassigned: it was reserved when the version-2 capability
       word was defined and must not be set by a producer.  The loader rejects
       any capability bit outside this enum, so leaving the hole undocumented
       would make an accidental use indistinguishable from a producer bug. */
    /* Every node tensor the receiver-forward and source-reverse callbacks
       consume or produce is lane-major: for an irrep block at offset o with
       multiplicity m and angular width d = 2l+1, element (channel a,
       component x) is stored at o + x*m + a instead of the node-major
       o + a*d + x.  The runtime presents and consumes lane-major node tensors;
       the generated receiver-forward callback applies the layer output mask
       and the normalization factor to the node messages it writes, and the
       runtime reverse applies the same mask-and-normalization product to the
       node output adjoint before the source-reverse callback reads it. */
    SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V2 = 1u << 3,
};

typedef struct SymmetrixJitOAMHostReceiverForwardArgsV2 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t flags;
    uint32_t scalar_size;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    int32_t edge_harmonic_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    double spline_x0;
    double spline_h;
    const int32_t* source_indices;
    const int32_t* receiver_offsets;
    const double* distances;
    const void* spline_coefficients;
    const void* edge_harmonics;
    const void* source_node_values;
    void* node_messages;
    double normalization_factor;
} SymmetrixJitOAMHostReceiverForwardArgsV2;

typedef struct SymmetrixJitOAMHostSourceReverseArgsV2 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t flags;
    uint32_t scalar_size;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t source_owner_count;
    int32_t edge_harmonic_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    double spline_x0;
    double spline_h;
    const int32_t* source_indices;
    const int32_t* target_indices;
    const int32_t* source_edge_offsets;
    const int32_t* source_edge_indices;
    const double* distances;
    const void* spline_coefficients;
    const void* edge_harmonics;
    const void* target_node_output_adjoint;
    const void* source_node_values;
    void* source_node_input_adjoint;
    void* edge_harmonic_adjoint;
    void* distance_adjoint;  /* one double per directed edge (always double) */
} SymmetrixJitOAMHostSourceReverseArgsV2;

typedef void (*SymmetrixJitOAMHostReceiverForwardOwnerV2)(
    const SymmetrixJitOAMHostReceiverForwardArgsV2*, int32_t receiver);
typedef void (*SymmetrixJitOAMHostSourceReverseOwnerV2)(
    const SymmetrixJitOAMHostSourceReverseArgsV2*, int32_t source);

typedef struct SymmetrixJitOAMHostLayerV2 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t reserved;
    uint32_t reserved_2;
    int32_t source_dimension;        /* node feature dimension consumed by the layer */
    int32_t edge_dimension;          /* layer edge-harmonic component count */
    int32_t weight_dimension;        /* radial spline function count */
    int32_t output_dimension;        /* receiver node-message dimension */
    int32_t instruction_count;
    int32_t channel_tile;            /* generated SIMD tile width, >= 1 */
    int32_t target_dimension;        /* node output-adjoint dimension consumed by reverse */
} SymmetrixJitOAMHostLayerV2;

typedef struct SymmetrixJitOAMHostPluginV2 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t layer_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const SymmetrixJitOAMHostLayerV2* layers;
    const SymmetrixJitOAMHostReceiverForwardOwnerV2* receiver_forward_owners;
    const SymmetrixJitOAMHostSourceReverseOwnerV2* source_reverse_owners;
} SymmetrixJitOAMHostPluginV2;

typedef const SymmetrixJitOAMHostPluginV2*
    (*SymmetrixJitOAMHostPluginQueryV2)(void);

SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV2*
symmetrix_jit_oam_host_plugin_query_v2(void);

enum SymmetrixJitOAMHostPluginCapabilityV3 {
    SYMMETRIX_JIT_OAM_HOST_FULL_FORWARD_V3 = 1u << 0,
    SYMMETRIX_JIT_OAM_HOST_FULL_REVERSE_V3 = 1u << 1,
    SYMMETRIX_JIT_OAM_HOST_LANE_MAJOR_V3 = 1u << 3,
    SYMMETRIX_JIT_OAM_HOST_PROJECTED_FORWARD_V3 = 1u << 4,
    SYMMETRIX_JIT_OAM_HOST_PROJECTED_REVERSE_V3 = 1u << 5,
    SYMMETRIX_JIT_OAM_HOST_RUNTIME_POST_LINEAR_V3 = 1u << 6,
};

typedef struct SymmetrixJitOAMHostReceiverForwardArgsV3 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t flags;
    uint32_t scalar_size;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    int32_t edge_harmonic_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    double spline_x0;
    double spline_h;
    const int32_t* source_indices;
    const int32_t* receiver_offsets;
    const double* distances;
    const void* spline_coefficients;
    const void* edge_harmonics;
    const void* source_node_values;
    void* node_messages;
    double normalization_factor;
    const void* runtime_post_linear_weights;
    const void* runtime_post_linear_bias;
} SymmetrixJitOAMHostReceiverForwardArgsV3;

typedef struct SymmetrixJitOAMHostSourceReverseArgsV3 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t flags;
    uint32_t scalar_size;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t source_owner_count;
    int32_t edge_harmonic_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    double spline_x0;
    double spline_h;
    const int32_t* source_indices;
    const int32_t* target_indices;
    const int32_t* source_edge_offsets;
    const int32_t* source_edge_indices;
    const double* distances;
    const void* spline_coefficients;
    const void* edge_harmonics;
    const void* target_node_output_adjoint;
    const void* source_node_values;
    void* source_node_input_adjoint;
    void* edge_harmonic_adjoint;
    void* distance_adjoint;
    double normalization_factor;
    const void* runtime_post_linear_weights;
} SymmetrixJitOAMHostSourceReverseArgsV3;

// The second argument is an identity-bound work item. Whole-owner schedules
// use the node index directly. Channel-tiled schedules encode
// node * channel_tile_count + channel_tile_index.
typedef void (*SymmetrixJitOAMHostReceiverForwardOwnerV4)(
    const SymmetrixJitOAMHostReceiverForwardArgsV3*, int32_t work_item);
typedef void (*SymmetrixJitOAMHostSourceReverseOwnerV4)(
    const SymmetrixJitOAMHostSourceReverseArgsV3*, int32_t work_item);

typedef struct SymmetrixJitOAMHostLayerV4 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t variant_mask;
    uint32_t reserved;
    int32_t source_dimension;
    int32_t edge_dimension;
    int32_t weight_dimension;
    int32_t output_dimension;
    int32_t instruction_count;
    int32_t channel_tile;
    int32_t target_dimension;
    int32_t projected_dimension;
    int32_t projection_weight_count;
    uint64_t projected_partial_workspace_bytes;
} SymmetrixJitOAMHostLayerV4;

typedef struct SymmetrixJitOAMHostPluginV4 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t layer_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* schedule_identity;
    const char* projection_layout;
    const SymmetrixJitOAMHostLayerV4* layers;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* full_forward_owners;
    const SymmetrixJitOAMHostSourceReverseOwnerV4* full_reverse_owners;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* projected_forward_owners;
    const SymmetrixJitOAMHostSourceReverseOwnerV4* projected_reverse_owners;
    const char* host_target;
    const char* forward_owner_schedule;
    const char* reverse_owner_schedule;
    const char* tiling_policy;
    const char* vector_policy;
    const char* shared_memory_policy;
    const char* register_budget_policy;
    const char* direct_geometry_policy;
    uint32_t scalar_width;
    uint32_t reserved_2;
    const char* required_isa;
    const char* partial_reduction_policy;
    const char* projection_mode;
    const char* schedule_qualification;
} SymmetrixJitOAMHostPluginV4;

typedef const SymmetrixJitOAMHostPluginV4*
    (*SymmetrixJitOAMHostPluginQueryV4)(void);

SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV4*
symmetrix_jit_oam_host_plugin_query_v4(void);

typedef struct SymmetrixJitOAMHostReceiverReverseArgsV4 {
    uint32_t struct_size;
    uint32_t layer;
    uint32_t flags;
    uint32_t scalar_size;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t receiver_owner_count;
    int32_t edge_harmonic_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    double spline_x0;
    double spline_h;
    const int32_t* source_indices;
    const int32_t* receiver_offsets;
    const double* distances;
    const void* spline_coefficients;
    const void* edge_harmonics;
    const void* target_node_output_adjoint;
    const void* source_node_values;
    void* source_node_input_adjoint;
    void* edge_harmonic_adjoint;
    void* distance_adjoint;
    double normalization_factor;
    const void* runtime_post_linear_weights;
} SymmetrixJitOAMHostReceiverReverseArgsV4;

typedef void (*SymmetrixJitOAMHostReceiverReverseOwnerV5)(
    const SymmetrixJitOAMHostReceiverReverseArgsV4*, int32_t receiver);

typedef SymmetrixJitOAMHostLayerV4 SymmetrixJitOAMHostLayerV5;

typedef struct SymmetrixJitOAMHostPluginV5 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t layer_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* schedule_identity;
    const char* projection_layout;
    const SymmetrixJitOAMHostLayerV5* layers;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* full_forward_owners;
    const SymmetrixJitOAMHostSourceReverseOwnerV4* full_reverse_owners;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* projected_forward_owners;
    const SymmetrixJitOAMHostReceiverReverseOwnerV5* projected_reverse_owners;
    const char* host_target;
    const char* forward_owner_schedule;
    const char* reverse_owner_schedule;
    const char* tiling_policy;
    const char* vector_policy;
    const char* shared_memory_policy;
    const char* register_budget_policy;
    const char* direct_geometry_policy;
    uint32_t scalar_width;
    uint32_t reserved_2;
    const char* required_isa;
    const char* partial_reduction_policy;
    const char* projection_mode;
    const char* schedule_qualification;
} SymmetrixJitOAMHostPluginV5;

typedef const SymmetrixJitOAMHostPluginV5*
    (*SymmetrixJitOAMHostPluginQueryV5)(void);

SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV5*
symmetrix_jit_oam_host_plugin_query_v5(void);

typedef SymmetrixJitOAMHostLayerV5 SymmetrixJitOAMHostLayerV6;

typedef struct SymmetrixJitOAMHostPluginV6 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t layer_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* schedule_identity;
    const char* projection_layout;
    const SymmetrixJitOAMHostLayerV6* layers;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* full_forward_owners;
    const SymmetrixJitOAMHostSourceReverseOwnerV4* full_reverse_owners;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* projected_forward_owners;
    const SymmetrixJitOAMHostReceiverReverseOwnerV5* projected_reverse_owners;
    const char* host_target;
    const char* forward_owner_schedule;
    const char* full_reverse_owner_schedule;
    const char* projected_reverse_owner_schedule;
    const char* tiling_policy;
    const char* vector_policy;
    const char* shared_memory_policy;
    const char* register_budget_policy;
    const char* direct_geometry_policy;
    uint32_t scalar_width;
    uint32_t reserved_2;
    const char* required_isa;
    const char* full_reverse_reduction_policy;
    const char* projected_reverse_reduction_policy;
    const char* projection_mode;
    const char* schedule_qualification;
} SymmetrixJitOAMHostPluginV6;

typedef const SymmetrixJitOAMHostPluginV6*
    (*SymmetrixJitOAMHostPluginQueryV6)(void);

SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV6*
symmetrix_jit_oam_host_plugin_query_v6(void);

typedef SymmetrixJitOAMHostLayerV6 SymmetrixJitOAMHostLayerV7;

/* ABI v7 retains the former projected fields only to preserve the descriptor
   layout. Full-message is the sole supported OAM variant: generators leave
   these fields zero/null, and loaders reject stale artifacts that populate
   them. */
typedef struct SymmetrixJitOAMHostPluginV7 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t layer_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* schedule_identity;
    const char* projection_layout;
    const SymmetrixJitOAMHostLayerV7* layers;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* full_forward_owners;
    const SymmetrixJitOAMHostSourceReverseOwnerV4* full_reverse_owners;
    const SymmetrixJitOAMHostReceiverForwardOwnerV4* projected_forward_owners;
    const SymmetrixJitOAMHostSourceReverseOwnerV4* projected_reverse_owners;
    const char* host_target;
    const char* forward_owner_schedule;
    const char* full_reverse_owner_schedule;
    const char* projected_reverse_owner_schedule;
    const char* tiling_policy;
    const char* vector_policy;
    const char* shared_memory_policy;
    const char* register_budget_policy;
    const char* direct_geometry_policy;
    uint32_t scalar_width;
    uint32_t reserved_2;
    const char* required_isa;
    const char* full_reverse_reduction_policy;
    const char* projected_reverse_reduction_policy;
    const char* projection_mode;
    const char* schedule_qualification;
} SymmetrixJitOAMHostPluginV7;

typedef const SymmetrixJitOAMHostPluginV7*
    (*SymmetrixJitOAMHostPluginQueryV7)(void);

SYMMETRIX_JIT_OAM_HOST_PLUGIN_EXPORT
const SymmetrixJitOAMHostPluginV7*
symmetrix_jit_oam_host_plugin_query_v7(void);

#ifdef __cplusplus
}
#endif
