#pragma once

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "cblas.hpp"

#if defined(KOKKOS_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace symmetrix {

inline int host_worker_blas_policy_code()
{
    static const int policy = [] {
        const char* raw = std::getenv("SYMMETRIX_HOST_WORKER_BLAS");
        const std::string_view value = raw == nullptr
            ? std::string_view("automatic") : std::string_view(raw);
        if (value.empty() || value == "automatic")
            return 0;
        if (value == "off")
            return 1;
        if (value == "unsafe")
            return 2;
        throw std::invalid_argument(
            "SYMMETRIX_HOST_WORKER_BLAS must be 'automatic', 'off', or "
            "'unsafe'.");
    }();
    return policy;
}

inline std::string_view host_worker_blas_requested_policy()
{
    switch (host_worker_blas_policy_code()) {
    case 0:
        return "automatic";
    case 1:
        return "off";
    default:
        return "unsafe";
    }
}

inline bool host_worker_cblas_enabled()
{
    const int policy = host_worker_blas_policy_code();
    if (policy == 1)
        return false;
    if (policy == 2)
        return true;
#if defined(KOKKOS_ENABLE_OPENMP)
    const auto provider = symmetrix_blas_provider();
    const auto threading_layer = symmetrix_blas_threading_layer();
    if (provider == "mkl" && threading_layer == "sequential")
        return true;
    if ((provider == "mkl" && threading_layer == "gnu_openmp")
            || (provider == "openblas" && threading_layer == "openmp"))
        return symmetrix_blas_allows_concurrent_cblas()
            && omp_get_max_active_levels() <= 1;
    if (provider == "openblas" && threading_layer == "pthread") {
        if (!Kokkos::is_initialized())
            return omp_get_max_threads() <= 1;
        return Kokkos::DefaultExecutionSpace().concurrency() <= 1;
    }
    return false;
#else
    return true;
#endif
}

inline std::string_view host_worker_blas_selected_backend()
{
    return host_worker_cblas_enabled() ? "cblas" : "kokkos";
}

}
