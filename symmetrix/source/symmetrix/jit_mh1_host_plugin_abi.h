#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_ABI_VERSION 3u
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS 2u
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER 0x04030201u
#else
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_BYTE_ORDER 0x01020304u
#endif
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_ABI_TAG \
    "symmetrix.jit.mh1.host-plugin/3"
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_QUERY_SYMBOL \
    "symmetrix_jit_mh1_host_plugin_query_v3"
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_VERSION 4u
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_ABI_TAG \
    "symmetrix.jit.mh1.host-plugin/4"
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_V4_QUERY_SYMBOL \
    "symmetrix_jit_mh1_host_plugin_query_v4"
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_ABI_VERSION 5u
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_ABI_TAG \
    "symmetrix.jit.mh1.host-plugin/5"
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_V5_QUERY_SYMBOL \
    "symmetrix_jit_mh1_host_plugin_query_v5"
#define SYMMETRIX_JIT_MH1_HOST_NODE_FORWARD_PHASES_V4 2u
#define SYMMETRIX_JIT_MH1_HOST_NODE_REVERSE_PHASES_V4 2u
#if defined(__unix__) || defined(__APPLE__)
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT \
    __attribute__((visibility("default")))
#else
#define SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SymmetrixJitMH1HostPluginCapabilityV3 {
    SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V3 = 1u << 0,
    SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V3 = 1u << 1,
    SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V3 = 1u << 2,
    SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V3 = 1u << 3,
    SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V3 = 1u << 4,
    SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V3 = 1u << 5,
};

enum SymmetrixJitMH1HostArgumentFlagV3 {
    SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V3 = 1u << 0,
    SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V3 = 1u << 1,
    SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V3 = 1u << 2,
};

// Each interaction is a generated UVU tensor program. All tensor pointers in
// the argument structures use contiguous row-major (Kokkos LayoutRight)
// storage, with trailing extents specified here.
typedef struct SymmetrixJitMH1HostInteractionV3 {
    uint32_t struct_size;
    uint32_t reserved;
    int32_t input_1_dimension;
    int32_t input_2_dimension;
    int32_t output_dimension;
    int32_t weight_size;
    int32_t phi_dimension;
    int32_t multiplicity;
    int32_t input_1_angular_dimension;
    int32_t instruction_count;
} SymmetrixJitMH1HostInteractionV3;

// Version 3 requires first_edge=0 and samples=num_edges. The callback owns one
// complete receiver. It increments the receiver's full
// node_messages row and does not write any other receiver. source_indices and
// target_indices are global [num_edges], receiver_offsets is [num_nodes+1],
// and edge_cutoff_scale is global [num_edges]. All edge tensors have num_edges
// rows. Node feature tensors and output_mask use ir_mul layout. The
// affine is linear_weight [weight_size,phi_dimension] plus optional
// linear_bias [weight_size]. source_node_values and node_messages are
// [num_nodes,input_1_dimension] and [num_nodes,output_dimension].
typedef struct SymmetrixJitMH1HostForwardArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    const int32_t* source_indices;
    const int32_t* target_indices;
    const int32_t* receiver_offsets;
    const float* edge_phi;
    const float* linear_weight;
    const float* linear_bias;
    const float* edge_linear_contribution;
    const float* edge_input_2;
    const float* edge_cutoff_scale;
    const float* output_mask;
    const float* source_node_values;
    float* node_messages;
} SymmetrixJitMH1HostForwardArgsV3;

// source_edge_offsets/source_edge_indices form one graph-wide compact
// source-owned CSR. source_owner_count gives its segment count; the offsets
// index the global source_edge_indices array. Every segment is nonempty and
// contains global edge IDs for one source. A source callback owns one segment
// and increments the complete source adjoint row. An edge callback overwrites
// edge_phi_adjoint
// [samples,phi_dimension], edge_input_2_adjoint
// [samples,input_2_dimension], and edge_cutoff_scale_adjoint [samples].
typedef struct SymmetrixJitMH1HostReverseArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    int64_t source_owner_count;
    const int32_t* source_indices;
    const int32_t* target_indices;
    const int32_t* source_edge_offsets;
    const int32_t* source_edge_indices;
    const float* edge_phi;
    const float* linear_weight;
    const float* linear_bias;
    const float* edge_linear_contribution;
    const float* edge_input_2;
    const float* edge_cutoff_scale;
    const float* output_mask;
    const float* source_node_values;
    const float* target_node_output_adjoint;
    float* source_node_input_adjoint;
    float* edge_phi_adjoint;
    float* edge_input_2_adjoint;
    float* edge_cutoff_scale_adjoint;
} SymmetrixJitMH1HostReverseArgsV3;

// These packets extend the generated TPConv boundary to the exact
// pair-conditioned prefix and density programs. The owners receive learned
// arrays at runtime and expose only geometry derivatives required for forces
// and stress; parameter and species derivatives are deliberately absent.
typedef struct SymmetrixJitMH1HostConditioningForwardArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t active_receiver_count;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* source_types;
    const int32_t* target_indices;
    const int32_t* node_types;
    const int32_t* active_receivers;
    const int32_t* receiver_offsets;
    const float* radial;
    const float* edge_cutoff_scale;
    const float* convolution_source_contributions;
    const float* convolution_target_contributions;
    const float* density_source_contributions;
    const float* density_target_contributions;
    const float* convolution_parameters;
    const float* density_parameters;
    float* edge_phi;
    float* node_density;
} SymmetrixJitMH1HostConditioningForwardArgsV3;

typedef struct SymmetrixJitMH1HostConditioningReverseArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* source_types;
    const int32_t* target_indices;
    const int32_t* node_types;
    const float* radial;
    const float* edge_cutoff_scale;
    const float* convolution_source_contributions;
    const float* convolution_target_contributions;
    const float* density_source_contributions;
    const float* density_target_contributions;
    const float* convolution_parameters;
    const float* density_parameters;
    const float* edge_phi_adjoint;
    const float* node_density_adjoint;
    float* radial_adjoint;
    float* edge_cutoff_scale_adjoint;
} SymmetrixJitMH1HostConditioningReverseArgsV3;

// Owner callbacks may run concurrently. They must not throw across this ABI,
// retain an argument pointer, or write outside the owner's unique output.
typedef void (*SymmetrixJitMH1HostForwardOwnerV3)(
    const SymmetrixJitMH1HostForwardArgsV3*, int32_t);
typedef void (*SymmetrixJitMH1HostSourceReverseOwnerV3)(
    const SymmetrixJitMH1HostReverseArgsV3*, int32_t);
typedef void (*SymmetrixJitMH1HostEdgeReverseOwnerV3)(
    const SymmetrixJitMH1HostReverseArgsV3*, int32_t);
typedef void (*SymmetrixJitMH1HostConditioningForwardOwnerV3)(
    const SymmetrixJitMH1HostConditioningForwardArgsV3*, int32_t);
typedef void (*SymmetrixJitMH1HostConditioningReverseOwnerV3)(
    const SymmetrixJitMH1HostConditioningReverseArgsV3*, int32_t);

typedef struct SymmetrixJitMH1HostConditioningOwnersV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixJitMH1HostConditioningForwardOwnerV3 forward_owner;
    SymmetrixJitMH1HostConditioningReverseOwnerV3 reverse_owner;
} SymmetrixJitMH1HostConditioningOwnersV3;

typedef struct SymmetrixJitMH1HostPluginV3 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t interaction_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    SymmetrixJitMH1HostInteractionV3
        interactions[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1HostForwardOwnerV3 forward_owner;
    SymmetrixJitMH1HostSourceReverseOwnerV3 source_reverse_owner;
    SymmetrixJitMH1HostEdgeReverseOwnerV3 edge_reverse_owner;
    SymmetrixJitMH1HostConditioningOwnersV3
        conditioning[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
} SymmetrixJitMH1HostPluginV3;

typedef const SymmetrixJitMH1HostPluginV3*
    (*SymmetrixJitMH1HostPluginQueryV3)(void);

SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
const SymmetrixJitMH1HostPluginV3*
symmetrix_jit_mh1_host_plugin_query_v3(void);

// ABI v4 is an additive, opt-in interface. ABI v3 remains the active evaluator
// interface until a generated node program and its runtime integration are
// admitted together. The inherited capability bits retain their v3 values.
enum SymmetrixJitMH1HostPluginCapabilityV4 {
    SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V4 = 1u << 0,
    SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V4 = 1u << 1,
    SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V4 = 1u << 2,
    SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V4 = 1u << 3,
    SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V4 = 1u << 4,
    SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V4 = 1u << 5,
    SYMMETRIX_JIT_MH1_HOST_NODE_PROGRAM_PHASES_V4 = 1u << 6,
    SYMMETRIX_JIT_MH1_HOST_PERSISTENT_IR_MUL_NODE_STATE_V4 = 1u << 7,
    SYMMETRIX_JIT_MH1_HOST_FIXED_WEIGHT_COORDINATES_V4 = 1u << 8,
    SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_V4 = 1u << 9,
    SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_V4 = 1u << 10,
    SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V4 = 1u << 11,
    // V3 forward weights are [weight_size, phi_dimension]. V4/V5 descriptors
    // with this capability use [phi_dimension, weight_size] for forward_owner.
    SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V4 = 1u << 14,
};

enum SymmetrixJitMH1HostArgumentFlagV4 {
    SYMMETRIX_JIT_MH1_HOST_APPLY_EDGE_CUTOFF_V4 = 1u << 0,
    SYMMETRIX_JIT_MH1_HOST_HAS_LINEAR_BIAS_V4 = 1u << 1,
    SYMMETRIX_JIT_MH1_HOST_HAS_FIXED_CONTRIBUTION_V4 = 1u << 2,
    SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_ARGUMENT_V4 = 1u << 8,
    SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_ARGUMENT_V4 = 1u << 9,
};

enum SymmetrixJitMH1HostNodePhaseV4 {
    SYMMETRIX_JIT_MH1_HOST_NODE_PRE_FORWARD_V4 = 0u,
    SYMMETRIX_JIT_MH1_HOST_NODE_POST_FORWARD_V4 = 1u,
    SYMMETRIX_JIT_MH1_HOST_NODE_POST_REVERSE_V4 = 2u,
    SYMMETRIX_JIT_MH1_HOST_NODE_PRE_REVERSE_V4 = 3u,
};

enum SymmetrixJitMH1HostNodeProgramFlagV4 {
    SYMMETRIX_JIT_MH1_HOST_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4 =
        1u << 0,
    // The runtime retains linear_2 output across forward and reverse so the
    // host tile scheduler can omit its forward replay during backpropagation.
    SYMMETRIX_JIT_MH1_HOST_NODE_RETAIN_INTERACTION_OUTPUT_V4 = 1u << 1,
};

typedef SymmetrixJitMH1HostInteractionV3
    SymmetrixJitMH1HostInteractionV4;
typedef SymmetrixJitMH1HostForwardArgsV3
    SymmetrixJitMH1HostForwardArgsV4;
typedef SymmetrixJitMH1HostReverseArgsV3
    SymmetrixJitMH1HostReverseArgsV4;
typedef SymmetrixJitMH1HostConditioningForwardArgsV3
    SymmetrixJitMH1HostConditioningForwardArgsV4;
typedef SymmetrixJitMH1HostConditioningReverseArgsV3
    SymmetrixJitMH1HostConditioningReverseArgsV4;

// Every node tensor is contiguous float32 [num_nodes,dimension] ir_mul state.
// element_indices contains the model embedding-table row for each node, not a
// caller-local species index. Layer 0 may pass a null layer_input: its embedded
// feature is fused from element_indices in pre-forward and recomputed for the
// skip path in post-forward.
// Runtime parameter arrays follow generated, fixed coordinate order. Their
// exact counts are recorded in the node-program descriptor; learned values are
// never embedded in the artifact. Pre-forward maps layer_input to up_output.
// Post-forward consumes messages, up, and layer_input; it recomputes residual
// and skip values and writes layer_output. readout_contribution is float32
// scratch only. The evaluator accumulates it into final double node energies;
// generated code must never cast or write the evaluator's double energy view.
// Unused phase pointers may be null. node_arena is optional phase-local scratch
// and does not imply retained residual, gate, product, or readout tapes.
typedef struct SymmetrixJitMH1HostNodeForwardArgsV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t flags;
    int64_t num_nodes;
    const int32_t* element_indices;
    const float* node_density;
    const float* linear_parameters;
    const float* product_parameters;
    const float* readout_parameters;
    const float* layer_input;
    const float* up;
    const float* messages;
    float* up_output;
    float* layer_output;
    float* readout_contribution;
    float* node_arena;
} SymmetrixJitMH1HostNodeForwardArgsV4;

// Post-reverse adds the generated readout derivative, scaled by energy_scale,
// to the mutable persistent layer output adjoint, then recomputes all forward
// node locals from messages, up, layer_input, density, and runtime parameters.
// It writes message, up, layer-input, and density adjoints.
// Pre-reverse consumes the completed up adjoint after TP reverse and increments
// layer_input_adjoint. Parameter/type adjoints are deliberately absent.
typedef struct SymmetrixJitMH1HostNodeReverseArgsV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t flags;
    float energy_scale;
    uint32_t reserved;
    int64_t num_nodes;
    const int32_t* element_indices;
    const float* node_density;
    const float* linear_parameters;
    const float* product_parameters;
    const float* readout_parameters;
    const float* layer_input;
    const float* up;
    const float* messages;
    const float* layer_output;
    float* node_arena;
    float* layer_output_adjoint;
    float* message_adjoint;
    float* up_adjoint;
    float* layer_input_adjoint;
    float* node_density_adjoint;
} SymmetrixJitMH1HostNodeReverseArgsV4;

typedef void (*SymmetrixJitMH1HostForwardOwnerV4)(
    const SymmetrixJitMH1HostForwardArgsV4*, int32_t);
typedef void (*SymmetrixJitMH1HostSourceReverseOwnerV4)(
    const SymmetrixJitMH1HostReverseArgsV4*, int32_t);
typedef void (*SymmetrixJitMH1HostEdgeReverseOwnerV4)(
    const SymmetrixJitMH1HostReverseArgsV4*, int32_t);
typedef void (*SymmetrixJitMH1HostConditioningForwardOwnerV4)(
    const SymmetrixJitMH1HostConditioningForwardArgsV4*, int32_t);
typedef void (*SymmetrixJitMH1HostConditioningReverseOwnerV4)(
    const SymmetrixJitMH1HostConditioningReverseArgsV4*, int32_t);
typedef void (*SymmetrixJitMH1HostNodeForwardOwnerV4)(
    const SymmetrixJitMH1HostNodeForwardArgsV4*, int32_t);
typedef void (*SymmetrixJitMH1HostNodeReverseOwnerV4)(
    const SymmetrixJitMH1HostNodeReverseArgsV4*, int32_t);

typedef struct SymmetrixJitMH1HostConditioningOwnersV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixJitMH1HostConditioningForwardOwnerV4 forward_owner;
    SymmetrixJitMH1HostConditioningReverseOwnerV4 reverse_owner;
} SymmetrixJitMH1HostConditioningOwnersV4;

typedef struct SymmetrixJitMH1HostNodeForwardPhaseV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t reserved;
    int32_t owners_per_node;
    int32_t reserved_2;
    int32_t reserved_3;
    int32_t reserved_4;
    SymmetrixJitMH1HostNodeForwardOwnerV4 owner;
} SymmetrixJitMH1HostNodeForwardPhaseV4;

typedef struct SymmetrixJitMH1HostNodeReversePhaseV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t reserved;
    int32_t owners_per_node;
    int32_t reserved_2;
    int32_t reserved_3;
    int32_t reserved_4;
    SymmetrixJitMH1HostNodeReverseOwnerV4 owner;
} SymmetrixJitMH1HostNodeReversePhaseV4;

typedef struct SymmetrixJitMH1HostNodeProgramV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t forward_phase_count;
    uint32_t reverse_phase_count;
    int32_t element_count;
    int32_t input_dimension;
    int32_t up_dimension;
    int32_t residual_dimension;
    int32_t skip_dimension;
    int32_t message_dimension;
    int32_t interaction_output_dimension;
    int32_t output_dimension;
    int32_t product_term_count;
    int32_t node_arena_dimension;
    uint32_t flags;
    int32_t reserved_2;
    int64_t linear_parameter_count;
    int64_t product_parameter_count;
    int64_t readout_parameter_count;
    int64_t reserved_3;
    SymmetrixJitMH1HostNodeForwardPhaseV4
        forward_phases[SYMMETRIX_JIT_MH1_HOST_NODE_FORWARD_PHASES_V4];
    SymmetrixJitMH1HostNodeReversePhaseV4
        reverse_phases[SYMMETRIX_JIT_MH1_HOST_NODE_REVERSE_PHASES_V4];
} SymmetrixJitMH1HostNodeProgramV4;

typedef struct SymmetrixJitMH1HostPluginV4 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t interaction_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* runtime_layout_fingerprint;
    SymmetrixJitMH1HostInteractionV4
        interactions[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1HostForwardOwnerV4 forward_owner;
    SymmetrixJitMH1HostSourceReverseOwnerV4 source_reverse_owner;
    SymmetrixJitMH1HostEdgeReverseOwnerV4 edge_reverse_owner;
    SymmetrixJitMH1HostConditioningOwnersV4
        conditioning[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1HostNodeProgramV4
        node_programs[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
} SymmetrixJitMH1HostPluginV4;

typedef const SymmetrixJitMH1HostPluginV4*
    (*SymmetrixJitMH1HostPluginQueryV4)(void);

SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
const SymmetrixJitMH1HostPluginV4*
symmetrix_jit_mh1_host_plugin_query_v4(void);

// ABI v5 adds the pair-spline R-stage boundary without changing the ABI-v3 or
// ABI-v4 descriptors. The first implementation owns one complete receiver and
// writes directly to persistent ir_mul messages and receiver density. Its
// source reverse owner recomputes the same spline values and derivatives while
// retaining one source-adjoint row. The spline already contains the cutoff
// behavior and complete TP weights, so these packets have no conditioner,
// edge-phi, or separate cutoff inputs.
enum SymmetrixJitMH1HostPluginCapabilityV5 {
    SYMMETRIX_JIT_MH1_HOST_FORWARD_OWNER_V5 = 1u << 0,
    SYMMETRIX_JIT_MH1_HOST_SOURCE_REVERSE_OWNER_V5 = 1u << 1,
    SYMMETRIX_JIT_MH1_HOST_EDGE_REVERSE_OWNER_V5 = 1u << 2,
    SYMMETRIX_JIT_MH1_HOST_GRAPH_WIDE_V5 = 1u << 3,
    SYMMETRIX_JIT_MH1_HOST_IR_MUL_FEATURES_V5 = 1u << 4,
    SYMMETRIX_JIT_MH1_HOST_CONDITIONING_OWNER_V5 = 1u << 5,
    SYMMETRIX_JIT_MH1_HOST_NODE_PROGRAM_PHASES_V5 = 1u << 6,
    SYMMETRIX_JIT_MH1_HOST_PERSISTENT_IR_MUL_NODE_STATE_V5 = 1u << 7,
    SYMMETRIX_JIT_MH1_HOST_FIXED_WEIGHT_COORDINATES_V5 = 1u << 8,
    SYMMETRIX_JIT_MH1_HOST_NODE_TILE_GATE_V5 = 1u << 9,
    SYMMETRIX_JIT_MH1_HOST_NODE_TILE_PRODUCT_V5 = 1u << 10,
    SYMMETRIX_JIT_MH1_HOST_SOURCE_OWNS_EDGE_REVERSE_V5 = 1u << 11,
    SYMMETRIX_JIT_MH1_HOST_SPLINE_R_FORWARD_V5 = 1u << 12,
    SYMMETRIX_JIT_MH1_HOST_SPLINE_R_SOURCE_REVERSE_V5 = 1u << 13,
    SYMMETRIX_JIT_MH1_HOST_PHI_MAJOR_FORWARD_WEIGHT_V5 = 1u << 14,
};

typedef SymmetrixJitMH1HostInteractionV4
    SymmetrixJitMH1HostInteractionV5;
typedef SymmetrixJitMH1HostConditioningOwnersV4
    SymmetrixJitMH1HostConditioningOwnersV5;
typedef SymmetrixJitMH1HostNodeProgramV4
    SymmetrixJitMH1HostNodeProgramV5;

enum SymmetrixJitMH1HostSplineRProgramFlagV5 {
    // Forward message values use the blockwise [multiplicity][node][component]
    // layout consumed by packed host linears. Reverse retains receiver-major
    // adjoints so source-owned edge traversal remains cache-local.
    SYMMETRIX_JIT_MH1_HOST_SPLINE_R_PACKED_FORWARD_MESSAGES_V5 = 1u << 0,
};

typedef struct SymmetrixJitMH1HostSplineRForwardArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int32_t type_count;
    int32_t pair_count;
    int32_t interval_count;
    int32_t function_count;
    double spline_h;
    double spline_x0;
    const int32_t* source_indices;
    const int32_t* target_indices;
    const int32_t* receiver_offsets;
    const int32_t* source_types;
    const int32_t* node_types;
    const double* distances;
    const void* coefficients;
    const void* edge_input_2;
    const void* output_mask;
    const void* source_node_values;
    void* node_messages;
    void* node_density;
} SymmetrixJitMH1HostSplineRForwardArgsV5;

typedef void (*SymmetrixJitMH1HostSplineRForwardOwnerV5)(
    const SymmetrixJitMH1HostSplineRForwardArgsV5*, int32_t);

typedef struct SymmetrixJitMH1HostSplineRReverseArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int32_t source_owner_count;
    int32_t type_count;
    int32_t pair_count;
    int32_t interval_count;
    int32_t function_count;
    int32_t reserved_2;
    double spline_h;
    double spline_x0;
    const int32_t* source_indices;
    const int32_t* target_indices;
    const int32_t* source_edge_offsets;
    const int32_t* source_edge_indices;
    const int32_t* source_types;
    const int32_t* node_types;
    const double* distances;
    const void* coefficients;
    const void* edge_input_2;
    const void* output_mask;
    const void* source_node_values;
    const void* target_node_output_adjoint;
    const void* node_density_adjoint;
    void* source_node_input_adjoint;
    void* edge_input_2_adjoint;
    void* distance_adjoint;
} SymmetrixJitMH1HostSplineRReverseArgsV5;

typedef void (*SymmetrixJitMH1HostSplineRReverseOwnerV5)(
    const SymmetrixJitMH1HostSplineRReverseArgsV5*, int32_t);

typedef struct SymmetrixJitMH1HostSplineRProgramV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t reserved;
    int32_t weight_function_count;
    int32_t density_function_index;
    int32_t coefficient_count;
    int32_t reserved_2;
    SymmetrixJitMH1HostSplineRForwardOwnerV5 forward_owner;
    SymmetrixJitMH1HostSplineRReverseOwnerV5 source_reverse_owner;
} SymmetrixJitMH1HostSplineRProgramV5;

typedef struct SymmetrixJitMH1HostPluginV5 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t pointer_size;
    uint32_t byte_order;
    uint32_t capabilities;
    uint32_t interaction_count;
    uint32_t scalar_size;
    uint32_t reserved;
    const char* abi_tag;
    const char* artifact_id;
    const char* generation_fingerprint;
    const char* semantic_fingerprint;
    const char* structure_fingerprint;
    const char* runtime_layout_fingerprint;
    SymmetrixJitMH1HostInteractionV5
        interactions[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1HostForwardOwnerV4 forward_owner;
    SymmetrixJitMH1HostSourceReverseOwnerV4 source_reverse_owner;
    SymmetrixJitMH1HostEdgeReverseOwnerV4 edge_reverse_owner;
    SymmetrixJitMH1HostConditioningOwnersV5
        conditioning[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1HostNodeProgramV5
        node_programs[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1HostSplineRProgramV5
        spline_r_programs[SYMMETRIX_JIT_MH1_HOST_PLUGIN_INTERACTIONS];
} SymmetrixJitMH1HostPluginV5;

typedef const SymmetrixJitMH1HostPluginV5*
    (*SymmetrixJitMH1HostPluginQueryV5)(void);

SYMMETRIX_JIT_MH1_HOST_PLUGIN_EXPORT
const SymmetrixJitMH1HostPluginV5*
symmetrix_jit_mh1_host_plugin_query_v5(void);

#ifdef __cplusplus
}
#endif
