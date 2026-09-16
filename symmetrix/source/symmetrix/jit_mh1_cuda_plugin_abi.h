#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_VERSION 3u
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS 2u
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER 0x04030201u
#else
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_BYTE_ORDER 0x01020304u
#endif
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_ABI_TAG \
    "symmetrix.jit.mh1.cuda-plugin/3"
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_QUERY_SYMBOL \
    "symmetrix_jit_mh1_cuda_plugin_query_v3"
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_VERSION 4u
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_ABI_TAG \
    "symmetrix.jit.mh1.cuda-plugin/4"
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_V4_QUERY_SYMBOL \
    "symmetrix_jit_mh1_cuda_plugin_query_v4"
#define SYMMETRIX_JIT_MH1_CUDA_NODE_FORWARD_PHASES_V4 2u
#define SYMMETRIX_JIT_MH1_CUDA_NODE_REVERSE_PHASES_V4 2u
#if defined(__unix__) || defined(__APPLE__)
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT \
    __attribute__((visibility("default")))
#else
#define SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum SymmetrixJitMH1CudaPluginCapabilityV3 {
    SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V3 = 1u << 0,
    SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V3 = 1u << 1,
    SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V3 = 1u << 2,
    SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V3 = 1u << 3,
    SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V3 = 1u << 4,
    SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V3 = 1u << 5,
    SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V3 = 1u << 6,
    SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V3 = 1u << 7,
    SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V3 = 1u << 8,
};

enum SymmetrixJitMH1CudaArgumentFlagV3 {
    SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V3 = 1u << 0,
    SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V3 = 1u << 1,
    SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V3 = 1u << 2,
};

// Each interaction is a generated UVU tensor program. Its leading metadata is
// intentionally identical to the host plugin interaction metadata so one
// contract-derived expectation can validate both backends. CUDA launch tuning
// is platform-specific and follows those common extents.
typedef struct SymmetrixJitMH1CudaInteractionV3 {
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
    int32_t forward_threads_per_block;
    int32_t source_threads_per_block;
    int32_t edge_threads_per_block;
    int32_t edge_reverse_physical_launch_count;
} SymmetrixJitMH1CudaInteractionV3;

// All tensor pointers address contiguous row-major float32 CUDA device
// storage. Graph offsets and indices address int32 device storage. Argument
// structures reside in host memory and are copied by value into asynchronous
// launches. Version 3 requires first_edge=0 and samples=num_edges.
// source_indices and receiver_offsets are global [num_edges] and
// [num_nodes+1] arrays. active_receivers is a graph-wide compact list with
// active_receiver_count entries. All edge tensors have num_edges rows.
// Node feature tensors and output_mask use ir_mul layout. linear_weight is
// phi-major with shape
// [phi_dimension][weight_size], so adjacent generated channel owners read
// adjacent values. linear_bias, edge_linear_contribution, and
// edge_cutoff_scale are read only when their corresponding flags are set and
// may be null otherwise. output_mask has output_dimension entries. The
// destination node_messages is incremented.
typedef struct SymmetrixJitMH1CudaForwardArgsV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t flags;
    uint32_t active_receiver_count;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t first_edge;
    int64_t samples;
    const int32_t* source_indices;
    const int32_t* active_receivers;
    const int32_t* receiver_offsets;
    const float* edge_phi;
    const float* linear_weight;
    const float* linear_bias;
    const float* edge_linear_contribution;
    const float* edge_input_2;
    const float* edge_cutoff_scale;
    const float* source_node_values;
    const float* output_mask;
    float* node_messages;
} SymmetrixJitMH1CudaForwardArgsV3;

// source_edge_offsets/source_edge_indices form one graph-wide compact
// source-owned CSR. source_owner_count gives its segment count; the
// offsets index the global source_edge_indices array. Every segment is
// nonempty and contains global edge IDs for one source. The source-node
// adjoint is incremented. The edge phi, input-2 (harmonic), and cutoff
// adjoints are overwritten. Learned parameters are inference inputs and
// deliberately have no adjoints in this ABI.
typedef struct SymmetrixJitMH1CudaReverseArgsV3 {
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
    const float* source_node_values;
    const float* output_mask;
    const float* target_node_output_adjoint;
    float* source_node_input_adjoint;
    float* edge_phi_adjoint;
    float* edge_input_2_adjoint;
    float* edge_cutoff_scale_adjoint;
} SymmetrixJitMH1CudaReverseArgsV3;

// The generated conditioning program owns the exact pair-conditioned prefix
// and density networks surrounding the graph-wide UVU program. Learned
// parameters and pre-folded species contributions remain runtime values;
// generated code contains only layer topology, dimensions, and derivatives.
// Forward owns one active receiver and traverses its CSR in stable edge order,
// writing edge_phi and incrementing node_density. Reverse owns one edge,
// recomputes its network activations, and increments only radial/cutoff
// adjoints. Parameter, species, and type adjoints are intentionally absent.
typedef struct SymmetrixJitMH1CudaConditioningForwardArgsV3 {
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
} SymmetrixJitMH1CudaConditioningForwardArgsV3;

typedef struct SymmetrixJitMH1CudaConditioningReverseArgsV3 {
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
} SymmetrixJitMH1CudaConditioningReverseArgsV3;

// `stream` is the opaque cudaStream_t owned by the evaluator. Launchers enqueue
// work, return the immediate numeric cudaError_t, and must not retain argument
// or stream pointers after returning. Each descriptor entry is specialized for
// its array index, so the argument interaction field must equal that index.
typedef int32_t (*SymmetrixJitMH1CudaForwardLaunchV3)(
    const SymmetrixJitMH1CudaForwardArgsV3*, void*, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaReverseLaunchV3)(
    const SymmetrixJitMH1CudaReverseArgsV3*, void*, int32_t, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaConditioningForwardLaunchV3)(
    const SymmetrixJitMH1CudaConditioningForwardArgsV3*, void*, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaConditioningReverseLaunchV3)(
    const SymmetrixJitMH1CudaConditioningReverseArgsV3*, void*, int32_t);

typedef struct SymmetrixJitMH1CudaInteractionLaunchesV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixJitMH1CudaForwardLaunchV3 forward_launch;
    SymmetrixJitMH1CudaReverseLaunchV3 reverse_launch;
} SymmetrixJitMH1CudaInteractionLaunchesV3;

typedef struct SymmetrixJitMH1CudaConditioningLaunchesV3 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixJitMH1CudaConditioningForwardLaunchV3 forward_launch;
    SymmetrixJitMH1CudaConditioningReverseLaunchV3 reverse_launch;
} SymmetrixJitMH1CudaConditioningLaunchesV3;

typedef struct SymmetrixJitMH1CudaPluginV3 {
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
    int32_t target_compute_capability;
    int32_t reserved_2;
    SymmetrixJitMH1CudaInteractionV3
        interactions[SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1CudaInteractionLaunchesV3
        launches[SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1CudaConditioningLaunchesV3
        conditioning_launches[SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS];
} SymmetrixJitMH1CudaPluginV3;

typedef const SymmetrixJitMH1CudaPluginV3*
    (*SymmetrixJitMH1CudaPluginQueryV3)(void);

SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT
const SymmetrixJitMH1CudaPluginV3*
symmetrix_jit_mh1_cuda_plugin_query_v3(void);

// ABI v4 is an additive, opt-in interface. The ABI-v3 query and loader remain
// active until generated node programs and evaluator dispatch land together.
// Inherited capability bits intentionally retain their v3 values.
enum SymmetrixJitMH1CudaPluginCapabilityV4 {
    SYMMETRIX_JIT_MH1_CUDA_FORWARD_LAUNCH_V4 = 1u << 0,
    SYMMETRIX_JIT_MH1_CUDA_REVERSE_LAUNCH_V4 = 1u << 1,
    SYMMETRIX_JIT_MH1_CUDA_PHI_MAJOR_LINEAR_WEIGHT_V4 = 1u << 2,
    SYMMETRIX_JIT_MH1_CUDA_ACTIVE_RECEIVERS_V4 = 1u << 3,
    SYMMETRIX_JIT_MH1_CUDA_GRAPH_WIDE_V4 = 1u << 4,
    SYMMETRIX_JIT_MH1_CUDA_IR_MUL_FEATURES_V4 = 1u << 5,
    SYMMETRIX_JIT_MH1_CUDA_GLOBAL_SOURCE_REVERSE_V4 = 1u << 6,
    SYMMETRIX_JIT_MH1_CUDA_EDGE_LAUNCH_COUNT_V4 = 1u << 7,
    SYMMETRIX_JIT_MH1_CUDA_CONDITIONING_LAUNCH_V4 = 1u << 8,
    SYMMETRIX_JIT_MH1_CUDA_NODE_PROGRAM_PHASES_V4 = 1u << 9,
    SYMMETRIX_JIT_MH1_CUDA_PERSISTENT_IR_MUL_NODE_STATE_V4 = 1u << 10,
    SYMMETRIX_JIT_MH1_CUDA_FIXED_WEIGHT_COORDINATES_V4 = 1u << 11,
};

enum SymmetrixJitMH1CudaArgumentFlagV4 {
    SYMMETRIX_JIT_MH1_CUDA_APPLY_EDGE_CUTOFF_V4 = 1u << 0,
    SYMMETRIX_JIT_MH1_CUDA_HAS_LINEAR_BIAS_V4 = 1u << 1,
    SYMMETRIX_JIT_MH1_CUDA_HAS_FIXED_CONTRIBUTION_V4 = 1u << 2,
};

enum SymmetrixJitMH1CudaNodePhaseV4 {
    SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_FORWARD_V4 = 0u,
    SYMMETRIX_JIT_MH1_CUDA_NODE_POST_FORWARD_V4 = 1u,
    SYMMETRIX_JIT_MH1_CUDA_NODE_POST_REVERSE_V4 = 2u,
    SYMMETRIX_JIT_MH1_CUDA_NODE_PRE_REVERSE_V4 = 3u,
};

enum SymmetrixJitMH1CudaNodeProgramFlagV4 {
    SYMMETRIX_JIT_MH1_CUDA_NODE_REQUIRES_TP_SOURCE_STATE_ADJOINT_V4 =
        1u << 0,
    SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_PRE_GATE_V4 = 1u << 1,
    SYMMETRIX_JIT_MH1_CUDA_NODE_RETAIN_INTERACTION_OUTPUT_V4 = 1u << 2,
    SYMMETRIX_JIT_MH1_CUDA_NODE_REUSE_MESSAGE_ADJOINT_V4 = 1u << 3,
};

typedef SymmetrixJitMH1CudaInteractionV3
    SymmetrixJitMH1CudaInteractionV4;
typedef SymmetrixJitMH1CudaForwardArgsV3
    SymmetrixJitMH1CudaForwardArgsV4;
typedef SymmetrixJitMH1CudaReverseArgsV3
    SymmetrixJitMH1CudaReverseArgsV4;
typedef SymmetrixJitMH1CudaConditioningForwardArgsV3
    SymmetrixJitMH1CudaConditioningForwardArgsV4;
typedef SymmetrixJitMH1CudaConditioningReverseArgsV3
    SymmetrixJitMH1CudaConditioningReverseArgsV4;

// Device-module-only spline packets share the generated R1 ownership model
// with ordinary MACE. Pair indices are ordered source-major, and the final
// spline function is the MH-1 density contribution.
typedef struct SymmetrixJitMH1CudaSplineV5 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const void* coefficients;
} SymmetrixJitMH1CudaSplineV5;

typedef struct SymmetrixJitMH1CudaSplineRForwardArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t active_type_count;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    const int32_t* node_types;
    const int32_t* num_neigh;
    const int32_t* first_neigh;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const double* radius;
    SymmetrixJitMH1CudaSplineV5 radial;
    const void* harmonics_values;
    const void* neighbor_features;
    const void* output_mask;
    void* node_density;
    void* output;
} SymmetrixJitMH1CudaSplineRForwardArgsV5;

typedef struct SymmetrixJitMH1CudaSplineRSourceArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t active_type_count;
    uint32_t reserved;
    int64_t num_nodes;
    int64_t num_edges;
    int64_t source_owner_count;
    const int32_t* node_types;
    const int32_t* neigh_indices;
    const int32_t* neigh_types;
    const int32_t* source_offsets;
    const int32_t* source_edges;
    const int32_t* edge_receivers;
    const double* radius;
    SymmetrixJitMH1CudaSplineV5 radial;
    const void* output_adjoint;
    const void* output_mask;
    const void* node_density_adjoint;
    void* source_adjoint;
} SymmetrixJitMH1CudaSplineRSourceArgsV5;

typedef struct SymmetrixJitMH1CudaSplineREdgeArgsV5 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t coordinate_scalar_size;
    uint32_t coordinates_are_unit;
    int64_t num_nodes;
    int64_t num_edges;
    const void* xyz;
    const double* radius;
    SymmetrixJitMH1CudaSplineV5 radial;
    const void* harmonics_values;
    const void* harmonics_gradients;
    const void* neighbor_features;
    const void* output_adjoint;
    double* directed_forces;
} SymmetrixJitMH1CudaSplineREdgeArgsV5;

// All pointers address contiguous float32 CUDA device storage. Node tensors
// are [num_nodes,dimension] persistent ir_mul state. element_indices contains
// the model embedding-table row for each node, not a caller-local species
// index. Layer 0 may pass a null layer_input: its embedded feature is fused in
// pre-forward and recomputed for the skip path in post-forward. Runtime arrays
// use the generated fixed coordinate order and exact counts in the associated
// node-program descriptor. Pre-forward maps layer_input to up_output.
// Post-forward consumes messages, up, and layer_input, recomputes residual and
// skip values, and writes layer_output. readout_contribution is float32 scratch
// only; final accumulation into the evaluator's double node energies remains
// outside the plugin. node_arena is optional phase-local scratch, not a tape.
// Unused phase pointers may be null.
typedef struct SymmetrixJitMH1CudaNodeForwardArgsV4 {
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
    float* retained_pre_gate;
    float* retained_interaction_output;
} SymmetrixJitMH1CudaNodeForwardArgsV4;

// Post-reverse adds the generated readout derivative, scaled by energy_scale,
// to the mutable layer output adjoint, recomputes forward node locals, and
// writes message, up, layer-input, and density adjoints. Pre-reverse consumes
// the completed up adjoint after TP reverse and increments layer_input_adjoint.
// Learned parameter, type, and species adjoints are intentionally absent.
typedef struct SymmetrixJitMH1CudaNodeReverseArgsV4 {
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
    const float* retained_pre_gate;
    const float* retained_interaction_output;
} SymmetrixJitMH1CudaNodeReverseArgsV4;

typedef int32_t (*SymmetrixJitMH1CudaForwardLaunchV4)(
    const SymmetrixJitMH1CudaForwardArgsV4*, void*, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaReverseLaunchV4)(
    const SymmetrixJitMH1CudaReverseArgsV4*, void*, int32_t, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaConditioningForwardLaunchV4)(
    const SymmetrixJitMH1CudaConditioningForwardArgsV4*, void*, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaConditioningReverseLaunchV4)(
    const SymmetrixJitMH1CudaConditioningReverseArgsV4*, void*, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaNodeForwardLaunchV4)(
    const SymmetrixJitMH1CudaNodeForwardArgsV4*, void*, int32_t);
typedef int32_t (*SymmetrixJitMH1CudaNodeReverseLaunchV4)(
    const SymmetrixJitMH1CudaNodeReverseArgsV4*, void*, int32_t);

typedef struct SymmetrixJitMH1CudaInteractionLaunchesV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixJitMH1CudaForwardLaunchV4 forward_launch;
    SymmetrixJitMH1CudaReverseLaunchV4 reverse_launch;
} SymmetrixJitMH1CudaInteractionLaunchesV4;

typedef struct SymmetrixJitMH1CudaConditioningLaunchesV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t reserved;
    uint32_t reserved_2;
    SymmetrixJitMH1CudaConditioningForwardLaunchV4 forward_launch;
    SymmetrixJitMH1CudaConditioningReverseLaunchV4 reverse_launch;
} SymmetrixJitMH1CudaConditioningLaunchesV4;

typedef struct SymmetrixJitMH1CudaNodeForwardPhaseV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t reserved;
    int32_t threads_per_block;
    int32_t reserved_2;
    int32_t reserved_3;
    int32_t reserved_4;
    SymmetrixJitMH1CudaNodeForwardLaunchV4 launch;
} SymmetrixJitMH1CudaNodeForwardPhaseV4;

typedef struct SymmetrixJitMH1CudaNodeReversePhaseV4 {
    uint32_t struct_size;
    uint32_t interaction;
    uint32_t phase;
    uint32_t reserved;
    int32_t threads_per_block;
    int32_t reserved_2;
    int32_t reserved_3;
    int32_t reserved_4;
    SymmetrixJitMH1CudaNodeReverseLaunchV4 launch;
} SymmetrixJitMH1CudaNodeReversePhaseV4;

typedef struct SymmetrixJitMH1CudaNodeProgramV4 {
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
    int32_t retained_pre_gate_dimension;
    int64_t linear_parameter_count;
    int64_t product_parameter_count;
    int64_t readout_parameter_count;
    int64_t retained_interaction_output_dimension;
    SymmetrixJitMH1CudaNodeForwardPhaseV4
        forward_phases[SYMMETRIX_JIT_MH1_CUDA_NODE_FORWARD_PHASES_V4];
    SymmetrixJitMH1CudaNodeReversePhaseV4
        reverse_phases[SYMMETRIX_JIT_MH1_CUDA_NODE_REVERSE_PHASES_V4];
} SymmetrixJitMH1CudaNodeProgramV4;

typedef struct SymmetrixJitMH1CudaPluginV4 {
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
    int32_t target_compute_capability;
    int32_t reserved_2;
    SymmetrixJitMH1CudaInteractionV4
        interactions[SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1CudaInteractionLaunchesV4
        launches[SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1CudaConditioningLaunchesV4
        conditioning_launches[SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS];
    SymmetrixJitMH1CudaNodeProgramV4
        node_programs[SYMMETRIX_JIT_MH1_CUDA_PLUGIN_INTERACTIONS];
} SymmetrixJitMH1CudaPluginV4;

typedef const SymmetrixJitMH1CudaPluginV4*
    (*SymmetrixJitMH1CudaPluginQueryV4)(void);

SYMMETRIX_JIT_MH1_CUDA_PLUGIN_EXPORT
const SymmetrixJitMH1CudaPluginV4*
symmetrix_jit_mh1_cuda_plugin_query_v4(void);

#ifdef __cplusplus
}
#endif
