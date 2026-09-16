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
