#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include "host_worker_blas.hpp"

namespace symmetrix {

inline int host_dense_backend_policy_code()
{
    static const int policy = [] {
        const char* raw = std::getenv("SYMMETRIX_HOST_DENSE_BACKEND");
        const std::string_view value = raw == nullptr
            ? std::string_view("automatic") : std::string_view(raw);
        if (value.empty() || value == "automatic")
            return 0;
        if (value == "flat")
            return 1;
        if (value == "team")
            return 2;
        throw std::invalid_argument(
            "SYMMETRIX_HOST_DENSE_BACKEND must be 'automatic', 'flat', or "
            "'team'.");
    }();
    return policy;
}

inline std::string_view host_dense_backend_requested_policy()
{
    switch (host_dense_backend_policy_code()) {
    case 0:
        return "automatic";
    case 1:
        return "flat";
    default:
        return "team";
    }
}

inline bool host_flat_dense_enabled()
{
    const int policy = host_dense_backend_policy_code();
    if (policy == 1)
        return true;
    if (policy == 2)
        return false;
    return !host_worker_cblas_enabled();
}

inline bool host_dense_backend_overrides_cblas()
{
    return host_dense_backend_policy_code() != 0;
}

inline std::string_view host_dense_backend_selected_backend()
{
    if (host_flat_dense_enabled())
        return "flat";
    if (host_dense_backend_policy_code() == 2)
        return "team";
    return "cblas";
}

template <typename Precision>
inline void host_dense_gemm_nn(
    const Precision* input,
    const Precision* weights,
    Precision* output,
    const int rows,
    const int inner,
    const int columns,
    const bool accumulate = false)
{
    for (int row=0; row<rows; ++row) {
        Precision* output_row = output+row*columns;
        if (!accumulate) {
#if defined(KOKKOS_ENABLE_OPENMP)
#pragma omp simd
#endif
            for (int column=0; column<columns; ++column)
                output_row[column] = Precision(0);
        }
        for (int index=0; index<inner; ++index) {
            const Precision value = input[row*inner+index];
            const Precision* weight_row = weights+index*columns;
#if defined(KOKKOS_ENABLE_OPENMP)
#pragma omp simd
#endif
            for (int column=0; column<columns; ++column)
                output_row[column] += value*weight_row[column];
        }
    }
}

template <typename Precision>
inline void host_dense_gemm_nt(
    const Precision* input,
    const Precision* weights,
    Precision* output,
    const int rows,
    const int inner,
    const int columns,
    const bool accumulate = false)
{
    for (int row=0; row<rows; ++row) {
        const Precision* input_row = input+row*inner;
        Precision* output_row = output+row*columns;
        for (int column=0; column<columns; ++column) {
            const Precision* weight_row = weights+column*inner;
            Precision value = Precision(0);
#if defined(KOKKOS_ENABLE_OPENMP)
#pragma omp simd reduction(+:value)
#endif
            for (int index=0; index<inner; ++index)
                value += input_row[index]*weight_row[index];
            if (accumulate)
                output_row[column] += value;
            else
                output_row[column] = value;
        }
    }
}

}
