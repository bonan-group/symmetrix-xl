#pragma once

#include <stdint.h>

// NequIP-OAM generated-R1 CUDA packet ABI.
//
// One generated shared library owns every interaction layer of exactly one
// NequIP-OAM v2 contract. The plugin descriptor below is a static object in
// that library: it publishes the contract identity, the per-layer schedule,
// and one specialized forward/reverse launch pair per layer. Layers are
// specialized by array index, so every launch must be called with a packet
// whose `layer_index` equals that index.
//
// The generated kernels evaluate the certified cubic radial spline and the
// weighted tensor-product contraction with baked Wigner constants. Reverse
// kernels own direct, single-owner per-edge geometry-adjoint rows. Projected
// reverse aliases the full source-owner launch after the evaluator stages the
// wide adjoint once per node. Projected forward fuses the post-linear projection,
// output mask, and normalization in generated code. Pre-linear, gate,
// self-connection, readout, and ZBL remain in the evaluator's Kokkos runtime.
//
// Every tensor pointer in the layer arguments addresses contiguous CUDA
// device storage. Node tensors are lane-major exactly as the host ABI v7
// contract defines: for an irrep block at offset o with multiplicity m and
// angular width d = 2l+1, element (channel a, component x) is stored at
// o + x*m + a. The edge-harmonic operand keeps its definition order and is
// gathered through the layer's baked harmonic index map. The argument
// structure itself resides in host memory and is copied by value into each
// asynchronous launch. Generated code contains only the exact layer topology,
// dimensions, Wigner constants, lane permutation, and output mask of the
// contract; no approximation or host staging is admitted.

#define SYMMETRIX_OAM_CUDA_PLUGIN_SCHEMA "symmetrix.oam.cuda-r1"
#define SYMMETRIX_OAM_CUDA_PLUGIN_SCHEMA_VERSION 3u
#define SYMMETRIX_OAM_CUDA_PLUGIN_ABI_VERSION 3u
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_OAM_CUDA_PLUGIN_BYTE_ORDER 0x04030201u
#else
#define SYMMETRIX_OAM_CUDA_PLUGIN_BYTE_ORDER 0x01020304u
#endif
#define SYMMETRIX_OAM_CUDA_PLUGIN_ABI_TAG \
    "symmetrix.jit.oam.cuda-plugin/3"
#define SYMMETRIX_OAM_CUDA_PLUGIN_QUERY_SYMBOL \
    "symmetrix_oam_cuda_plugin_query_v3"
// Schema bound on the interaction-layer count accepted by ABI version 1.
#define SYMMETRIX_OAM_CUDA_PLUGIN_MAX_LAYERS 64u
#if defined(__unix__) || defined(__APPLE__)
#define SYMMETRIX_OAM_CUDA_PLUGIN_EXPORT \
    __attribute__((visibility("default")))
#else
#define SYMMETRIX_OAM_CUDA_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SymmetrixOAMCudaScalarKindV1 {
    SYMMETRIX_OAM_CUDA_SCALAR_FLOAT32_V1 = 1u,
    SYMMETRIX_OAM_CUDA_SCALAR_FLOAT64_V1 = 2u,
};

// Capability bits describe what the generated artifact implements.
// FORWARD_LAUNCH/REVERSE_LAUNCH must agree with the function pointers in
// every layer launch entry. The remaining bits are descriptive: SPLINE_RADIAL
// records that the radial stage evaluates the certified cubic spline;
// RUNTIME_SPLINE_COEFFICIENTS means the launches read the packet's
// spline_coefficients device buffer (always set by this generator: the
// certified coefficient table stays a runtime input, never baked).
enum SymmetrixOAMCudaPluginCapabilityV1 {
    SYMMETRIX_OAM_CUDA_FORWARD_LAUNCH_V1 = 1u << 0,
    SYMMETRIX_OAM_CUDA_REVERSE_LAUNCH_V1 = 1u << 1,
    SYMMETRIX_OAM_CUDA_SPLINE_RADIAL_V1 = 1u << 2,
    SYMMETRIX_OAM_CUDA_WEIGHTED_TENSOR_PRODUCT_V1 = 1u << 3,
    SYMMETRIX_OAM_CUDA_RECEIVER_ACCUMULATION_V1 = 1u << 4,
    SYMMETRIX_OAM_CUDA_SOURCE_ADJOINT_V1 = 1u << 5,
    SYMMETRIX_OAM_CUDA_GEOMETRY_ADJOINT_V1 = 1u << 6,
    SYMMETRIX_OAM_CUDA_RUNTIME_SPLINE_COEFFICIENTS_V1 = 1u << 7,
    // Direct packets compute harmonics and geometry adjoints in the generated
    // edge-owner kernels; their harmonic tape pointers are not required.
    SYMMETRIX_OAM_CUDA_DIRECT_GEOMETRY_V1 = 1u << 8,
    // Forward launches consume one evaluator-owned harmonic row per edge.
    // Reverse remains direct and computes its geometry gradients inline.
    SYMMETRIX_OAM_CUDA_STAGED_FORWARD_INPUTS_V1 = 1u << 9,
    // Reverse launches use one independently specialized source-owner kernel
    // per source irrep. Source channels are disjoint; only geometry partials
    // require a following stream-ordered reduction.
    SYMMETRIX_OAM_CUDA_GROUPED_REVERSE_PARTIALS_V1 = 1u << 10,
};

enum SymmetrixOAMCudaLayerFlagV1 {
    SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1 = 1u << 0,
};

// Static schedule entry for one interaction layer. The dimensions are exactly
// the serialized contract dimensions of that layer; the evaluator validates
// them before any launch, and the generated launch is specialized for them.
// `weight_dimension` is the tensor-product weight vector length and equals
// the spline function count. `normalization_factor` is the layer's global
// scatter normalization exactly as the runtime's LayerState carries it.
typedef struct SymmetrixOAMCudaLayerV1 {
    uint32_t struct_size;
    uint32_t reserved;
    int32_t layer_index;
    int32_t source_dimension;
    int32_t edge_dimension;
    int32_t weight_dimension;
    int32_t output_dimension;
    int32_t target_dimension;
    int32_t instruction_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    int32_t edge_harmonic_count;
    int32_t forward_threads_per_block;
    int32_t reverse_threads_per_block;
    int32_t reverse_group_count;
    int32_t reverse_group_flags;
    double normalization_factor;
} SymmetrixOAMCudaLayerV1;

// One packet serves both launches of a layer. Legacy launches consume the
// graphs, distances, spline coefficients, harmonics, source_node_values, and
// write node_messages (stored, including empty receiver rows); the adjoint
// pointers are ignored. Direct launches additionally consume edge_vectors and
// write geometry_adjoint directly. A staged-forward artifact additionally
// consumes edge_harmonics during forward; edge_harmonic_adjoint and
// distance_adjoint remain ignored in direct mode.
//
// Graph layout follows the evaluator: edges are receiver-major,
// source_indices[e] is the source node of edge e, receiver_offsets is the
// receiver CSR [num_nodes + 1], receiver_indices[e] is the receiving node,
// and source_edge_offsets/source_edge_indices are the compact source CSR.
// distances is double-precision [num_edges]; distance_adjoint is likewise
// double per directed edge, matching the host ABI v7 contract. Packets
// address the full edge range; there is no chunking in ABI version 1. Direct
// kernels still stream edge-local geometry and harmonics through their owner
// loop and retain no full-graph harmonic tape.
typedef struct SymmetrixOAMCudaLayerArgsV1 {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t layer_index;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int32_t edge_harmonic_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    double spline_x0;
    double spline_h;
    const int32_t* source_indices;
    const int32_t* receiver_offsets;
    const int32_t* receiver_indices;
    const int32_t* source_edge_offsets;
    const int32_t* source_edge_indices;
    const double* distances;
    const void* spline_coefficients;
    const void* edge_harmonics;
    const void* source_node_values;
    void* node_messages;
    const void* target_node_output_adjoint;
    void* source_node_input_adjoint;
    void* edge_harmonic_adjoint;
    double* distance_adjoint;
    const void* edge_vectors;
    void* geometry_adjoint;
    int64_t reverse_segment_count;
    const int32_t* reverse_segment_sources;
    const int32_t* reverse_segment_begin;
    const int32_t* reverse_segment_end;
    const int32_t* reverse_segment_order;
    void* reverse_source_partials;
    void* reverse_geometry_partials;
    const void* reverse_harmonic_gradients;
} SymmetrixOAMCudaLayerArgsV1;

// `stream` is the opaque cudaStream_t owned by the evaluator. Launch functions
// enqueue work, return the immediate numeric cudaError_t, and must not retain
// any argument or stream pointer after returning.
typedef int32_t (*SymmetrixOAMCudaForwardLaunchV1)(
    const SymmetrixOAMCudaLayerArgsV1*, void*, int32_t);
typedef int32_t (*SymmetrixOAMCudaReverseLaunchV1)(
    const SymmetrixOAMCudaLayerArgsV1*, void*, int32_t);

typedef struct SymmetrixOAMCudaLayerLaunchesV1 {
    uint32_t struct_size;
    uint32_t layer_index;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixOAMCudaForwardLaunchV1 forward_launch;
    SymmetrixOAMCudaReverseLaunchV1 reverse_launch;
} SymmetrixOAMCudaLayerLaunchesV1;

// The descriptor, its layer schedule, and its launch table are static objects
// of the generated library; the query returns a pointer to them and the caller
// must not free or modify anything reachable from it. `target_architecture`
// is the canonical artifact architecture ("sm120") and
// `target_compute_capability` is its numeric form (120).
typedef struct SymmetrixOAMCudaPluginV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t scalar_kind;
    uint32_t scalar_size;
    uint32_t layer_count;
    uint32_t reserved;
    uint32_t reserved_2;
    const char* abi_tag;
    const char* artifact_id;
    const char* contract_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* target_architecture;
    int32_t target_compute_capability;
    int32_t max_threads_per_block;
    const SymmetrixOAMCudaLayerV1* layers;
    const SymmetrixOAMCudaLayerLaunchesV1* launches;
} SymmetrixOAMCudaPluginV1;

typedef const SymmetrixOAMCudaPluginV1*
    (*SymmetrixOAMCudaPluginQueryV1)(void);

#if !defined(__CUDACC_RTC__)
SYMMETRIX_OAM_CUDA_PLUGIN_EXPORT
const SymmetrixOAMCudaPluginV1* symmetrix_oam_cuda_plugin_query_v1(void);
#endif

// ABI v3 retains the former projected fields only to preserve the descriptor
// layout. Full-message is the sole supported OAM variant: generators must leave
// these fields zero/null, and loaders reject stale artifacts that populate them.
enum SymmetrixOAMCudaPluginCapabilityV2 {
    SYMMETRIX_OAM_CUDA_FORWARD_LAUNCH_V2 = 1u << 0,
    SYMMETRIX_OAM_CUDA_REVERSE_LAUNCH_V2 = 1u << 1,
    SYMMETRIX_OAM_CUDA_SPLINE_RADIAL_V2 = 1u << 2,
    SYMMETRIX_OAM_CUDA_WEIGHTED_TENSOR_PRODUCT_V2 = 1u << 3,
    SYMMETRIX_OAM_CUDA_RECEIVER_ACCUMULATION_V2 = 1u << 4,
    SYMMETRIX_OAM_CUDA_SOURCE_ADJOINT_V2 = 1u << 5,
    SYMMETRIX_OAM_CUDA_GEOMETRY_ADJOINT_V2 = 1u << 6,
    SYMMETRIX_OAM_CUDA_RUNTIME_SPLINE_COEFFICIENTS_V2 = 1u << 7,
    SYMMETRIX_OAM_CUDA_DIRECT_GEOMETRY_V2 = 1u << 8,
    SYMMETRIX_OAM_CUDA_STAGED_FORWARD_INPUTS_V2 = 1u << 9,
    SYMMETRIX_OAM_CUDA_GROUPED_REVERSE_PARTIALS_V2 = 1u << 10,
    SYMMETRIX_OAM_CUDA_PROJECTED_FORWARD_V2 = 1u << 11,
    SYMMETRIX_OAM_CUDA_PROJECTED_REVERSE_V2 = 1u << 12,
    SYMMETRIX_OAM_CUDA_RUNTIME_POST_LINEAR_V2 = 1u << 13,
};

typedef struct SymmetrixOAMCudaLayerV2 {
    uint32_t struct_size;
    uint32_t reserved;
    int32_t layer_index;
    int32_t source_dimension;
    int32_t edge_dimension;
    int32_t weight_dimension;
    int32_t output_dimension;
    int32_t target_dimension;
    int32_t instruction_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    int32_t edge_harmonic_count;
    int32_t forward_threads_per_block;
    int32_t reverse_threads_per_block;
    int32_t reverse_group_count;
    int32_t reverse_group_flags;
    double normalization_factor;
    int32_t projected_dimension;
    int32_t projection_weight_count;
    uint64_t projected_partial_workspace_bytes;
    int32_t projected_channel_tile;
    int32_t reserved_2;
} SymmetrixOAMCudaLayerV2;

typedef struct SymmetrixOAMCudaLayerArgsV2 {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t layer_index;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int32_t edge_harmonic_count;
    int32_t spline_interval_count;
    int32_t spline_function_count;
    double spline_x0;
    double spline_h;
    const int32_t* source_indices;
    const int32_t* receiver_offsets;
    const int32_t* receiver_indices;
    const int32_t* source_edge_offsets;
    const int32_t* source_edge_indices;
    const double* distances;
    const void* spline_coefficients;
    const void* edge_harmonics;
    const void* source_node_values;
    void* node_messages;
    const void* target_node_output_adjoint;
    void* source_node_input_adjoint;
    void* edge_harmonic_adjoint;
    double* distance_adjoint;
    const void* edge_vectors;
    void* geometry_adjoint;
    int64_t reverse_segment_count;
    const int32_t* reverse_segment_sources;
    const int32_t* reverse_segment_begin;
    const int32_t* reverse_segment_end;
    const int32_t* reverse_segment_order;
    void* reverse_source_partials;
    void* reverse_geometry_partials;
    const void* reverse_harmonic_gradients;
    const void* runtime_post_linear_weights;
    const void* runtime_post_linear_bias;
    double normalization_factor;
} SymmetrixOAMCudaLayerArgsV2;

typedef int32_t (*SymmetrixOAMCudaLaunchV2)(
    const SymmetrixOAMCudaLayerArgsV2*, void*, int32_t);

typedef struct SymmetrixOAMCudaLayerLaunchesV2 {
    uint32_t struct_size;
    uint32_t layer_index;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixOAMCudaLaunchV2 full_forward_launch;
    SymmetrixOAMCudaLaunchV2 full_reverse_launch;
    SymmetrixOAMCudaLaunchV2 projected_forward_launch;
    SymmetrixOAMCudaLaunchV2 projected_reverse_launch;
} SymmetrixOAMCudaLayerLaunchesV2;

typedef struct SymmetrixOAMCudaPluginV3 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t scalar_kind;
    uint32_t scalar_size;
    uint32_t layer_count;
    uint32_t reserved;
    uint32_t reserved_2;
    const char* abi_tag;
    const char* artifact_id;
    const char* contract_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* target_architecture;
    int32_t target_compute_capability;
    int32_t max_threads_per_block;
    const char* execution_variants;
    const char* projection_layout;
    const char* projected_forward_schedule;
    const char* projected_reverse_schedule;
    const char* vector_policy;
    const char* shared_memory_policy;
    const char* register_budget_policy;
    const char* full_forward_schedule;
    const char* full_reverse_schedule;
    const char* full_vector_policy;
    const char* direct_geometry_policy;
    const char* full_tiling_policy;
    const char* full_shared_memory_policy;
    const char* full_register_budget_policy;
    const SymmetrixOAMCudaLayerV2* layers;
    const SymmetrixOAMCudaLayerLaunchesV2* launches;
    int32_t projected_threads_per_block;
    int32_t reserved_3;
    const char* projected_grid_policy;
    const char* projected_reduction_workspace;
    const char* projected_staged_input_policy;
    const char* projected_reverse_grouping;
    const char* target_policy;
    const char* precision_policy;
    const char* schedule_qualification;
} SymmetrixOAMCudaPluginV3;

typedef const SymmetrixOAMCudaPluginV3*
    (*SymmetrixOAMCudaPluginQueryV3)(void);

#if !defined(__CUDACC_RTC__)
SYMMETRIX_OAM_CUDA_PLUGIN_EXPORT
const SymmetrixOAMCudaPluginV3* symmetrix_oam_cuda_plugin_query_v3(void);
#endif

#ifdef __cplusplus
}
#endif
