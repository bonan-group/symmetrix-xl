#pragma once

#include <stdint.h>

#define SYMMETRIX_JIT_OPERATOR_ABI_VERSION_V1 3u
#define SYMMETRIX_JIT_OPERATOR_R0_ABI_VERSION_V2 3u
#define SYMMETRIX_JIT_OPERATOR_R0_ABI_VERSION_V3 4u
#define SYMMETRIX_JIT_OPERATOR_R0_ABI_VERSION_V4 5u
#define SYMMETRIX_JIT_OPERATOR_KIND_M0_V1 1u
#define SYMMETRIX_JIT_OPERATOR_KIND_R0_V1 2u
#define SYMMETRIX_JIT_OPERATOR_KIND_M1_V1 3u
#define SYMMETRIX_JIT_OPERATOR_SCALAR_FLOAT32_V1 1u
#define SYMMETRIX_JIT_OPERATOR_SCALAR_FLOAT64_V1 2u

#define SYMMETRIX_JIT_OPERATOR_M0_FORWARD_V1 (1u << 0)
#define SYMMETRIX_JIT_OPERATOR_M0_REVERSE_V1 (1u << 1)
#define SYMMETRIX_JIT_OPERATOR_M0_ALIAS_SAFE_V1 (1u << 2)
#define SYMMETRIX_JIT_OPERATOR_M0_SCALE_ADJOINT_V1 (1u << 3)
#define SYMMETRIX_JIT_OPERATOR_M0_RESPONSE_REVERSE_V1 (1u << 4)

#define SYMMETRIX_JIT_OPERATOR_M1_FORWARD_V1 SYMMETRIX_JIT_OPERATOR_M0_FORWARD_V1
#define SYMMETRIX_JIT_OPERATOR_M1_REVERSE_V1 SYMMETRIX_JIT_OPERATOR_M0_REVERSE_V1
#define SYMMETRIX_JIT_OPERATOR_M1_ALIAS_SAFE_V1 SYMMETRIX_JIT_OPERATOR_M0_ALIAS_SAFE_V1
#define SYMMETRIX_JIT_OPERATOR_M1_SCALE_ADJOINT_V1 SYMMETRIX_JIT_OPERATOR_M0_SCALE_ADJOINT_V1
#define SYMMETRIX_JIT_OPERATOR_M1_ACCUMULATE_INPUT_ADJOINT_V1 (1u << 0)

#define SYMMETRIX_JIT_OPERATOR_R0_DENSITY_PREPARE_V1 (1u << 8)
#define SYMMETRIX_JIT_OPERATOR_R0_FORWARD_V1 (1u << 9)
#define SYMMETRIX_JIT_OPERATOR_R0_REVERSE_PREPARE_V1 (1u << 10)
#define SYMMETRIX_JIT_OPERATOR_R0_COORDINATE_REVERSE_V1 (1u << 11)
#define SYMMETRIX_JIT_OPERATOR_R0_COMPACT_GEOMETRY_V1 (1u << 12)
#define SYMMETRIX_JIT_OPERATOR_R0_PRECOMPUTED_SCALE_V1 (1u << 13)
#define SYMMETRIX_JIT_OPERATOR_R0_RECEIVER_BATCH_V1 (1u << 14)

typedef struct SymmetrixJitM0ArgsV1 {
    uint32_t struct_size;
    uint32_t reserved;
    int64_t num_nodes;
    int32_t channels;
    int32_t capture_input_scale_adjoint;
    const int32_t* node_types;
    const void* input;
    const void* weights;
    const void* output_adjoint;
    void* output;
    void* input_adjoint;
    double* input_scale_adjoint;
} SymmetrixJitM0ArgsV1;

typedef SymmetrixJitM0ArgsV1 SymmetrixJitM1ArgsV1;

typedef struct SymmetrixJitM0ArgsV2 {
    SymmetrixJitM0ArgsV1 packed;
    int64_t input_node_stride;
    int64_t output_node_stride;
    int64_t output_adjoint_node_stride;
    int64_t input_adjoint_node_stride;
} SymmetrixJitM0ArgsV2;

typedef SymmetrixJitM0ArgsV2 SymmetrixJitM1ArgsV2;

typedef struct SymmetrixJitR0SplineV1 {
    uint32_t struct_size;
    uint32_t edge_types;
    uint32_t intervals;
    uint32_t functions;
    double h;
    double x0;
    const void* coefficients;
} SymmetrixJitR0SplineV1;

typedef struct SymmetrixJitR0ArgsV1 {
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
    SymmetrixJitR0SplineV1 radial;
    SymmetrixJitR0SplineV1 density;
    const void* harmonics;
    const void* harmonic_gradients;
    void* output;
    void* output_adjoint;
    double* density_state;
    const double* precomputed_scale_adjoint;
    double* directed_forces;
    double cutoff;
    const double* edge_weights;
} SymmetrixJitR0ArgsV1;
