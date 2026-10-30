#pragma once

#include <cstdint>
#include <type_traits>

enum class FactorizedBlasTranspose : std::uint8_t {
    none,
    transpose,
};

enum class FactorizedBlasScalar : std::uint8_t {
    float32,
    float64,
};

struct FactorizedStridedBatchedGemmDescriptor {
    FactorizedBlasTranspose transpose_a = FactorizedBlasTranspose::none;
    FactorizedBlasTranspose transpose_b = FactorizedBlasTranspose::none;
    FactorizedBlasScalar scalar = FactorizedBlasScalar::float32;
    std::int32_t m = 0;
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::int32_t leading_a = 0;
    std::int32_t leading_b = 0;
    std::int32_t leading_c = 0;
    std::int64_t stride_a = 0;
    std::int64_t stride_b = 0;
    std::int64_t stride_c = 0;
    std::int32_t batch_count = 0;
    double alpha = 1.0;
    double beta = 0.0;
};

template<typename Precision>
constexpr FactorizedBlasScalar factorized_blas_scalar =
    std::is_same_v<Precision,float>
        ? FactorizedBlasScalar::float32 : FactorizedBlasScalar::float64;

// The MH-0 second-interaction contraction, expressed for the column-major BLAS
// API from the row-major operands the M1/A1 views carry.  The row-major problem
//     C(m x n) = A(m x k) * B(k x n)
// is computed as its transpose
//     C^T(n x m) = B^T(n x k) * A^T(k x m),
// so the caller passes the operands in the order (B, A, C) and each leading
// dimension is the row stride of the row-major operand it describes.  A stride
// of zero on the first operand is the shared-weight broadcast: one weight
// matrix serves every node in the batch.
template<typename Precision>
constexpr FactorizedStridedBatchedGemmDescriptor
make_mh0_second_interaction_gemm_descriptor(
    const std::int32_t batch,
    const std::int32_t rows,
    const std::int32_t inner,
    const std::int32_t columns,
    const std::int64_t stride_a,
    const std::int64_t stride_b,
    const std::int64_t stride_c)
{
    static_assert(
        std::is_same_v<Precision,float> || std::is_same_v<Precision,double>);
    return {
        FactorizedBlasTranspose::none,
        FactorizedBlasTranspose::none,
        factorized_blas_scalar<Precision>,
        columns,
        rows,
        inner,
        columns,
        inner,
        columns,
        stride_b,
        stride_a,
        stride_c,
        batch,
        1.0,
        0.0,
    };
}

template<typename Precision>
constexpr FactorizedStridedBatchedGemmDescriptor
make_execution_reverse_gemm_descriptor(
    const std::int32_t batch,
    const std::int32_t degree,
    const std::int32_t embedding,
    const std::int32_t columns,
    const std::int32_t state_columns,
    const std::int32_t coupling_columns)
{
    static_assert(
        std::is_same_v<Precision,float> || std::is_same_v<Precision,double>);
    return {
        FactorizedBlasTranspose::none,
        FactorizedBlasTranspose::none,
        factorized_blas_scalar<Precision>,
        columns,
        degree,
        embedding,
        state_columns,
        embedding,
        coupling_columns,
        static_cast<std::int64_t>(embedding)*state_columns,
        static_cast<std::int64_t>(degree)*embedding,
        static_cast<std::int64_t>(degree)*coupling_columns,
        batch,
        1.0,
        0.0,
    };
}

template<typename Precision>
constexpr FactorizedStridedBatchedGemmDescriptor
make_execution_forward_gemm_descriptor(
    const std::int32_t batch,
    const std::int32_t degree,
    const std::int32_t embedding,
    const std::int32_t columns,
    const std::int32_t maximum_columns)
{
    static_assert(
        std::is_same_v<Precision,float> || std::is_same_v<Precision,double>);
    return {
        FactorizedBlasTranspose::none,
        FactorizedBlasTranspose::transpose,
        factorized_blas_scalar<Precision>,
        columns,
        embedding,
        degree,
        maximum_columns,
        embedding,
        maximum_columns,
        static_cast<std::int64_t>(degree)*maximum_columns,
        static_cast<std::int64_t>(degree)*embedding,
        static_cast<std::int64_t>(embedding)*maximum_columns,
        batch,
        1.0,
        0.0,
    };
}

static_assert(std::is_standard_layout_v<FactorizedStridedBatchedGemmDescriptor>);
static_assert(std::is_trivially_copyable_v<FactorizedStridedBatchedGemmDescriptor>);
