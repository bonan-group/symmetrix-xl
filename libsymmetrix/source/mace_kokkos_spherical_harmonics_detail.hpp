#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBlas_tpl_spec.hpp"
#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <rocblas/rocblas.h>
#endif
#include "nlohmann/json.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"
#include "spherical_harmonic_device.hpp"

#include "tools_kokkos.hpp"
#include "mace_kokkos.hpp"
#include "device_backend.hpp"
#include "factorized_blas.hpp"

template <typename Precision>
struct MACEKokkos<Precision>::SphericalHarmonicsState {
#ifdef SYMMETRIX_SPHERICART_CUDA
    sphericart::cuda::SphericalHarmonics<Precision> calculator;
#else
    sphericart::SphericalHarmonics<Precision> calculator;
#endif

    explicit SphericalHarmonicsState(const int l_max)
        : calculator(l_max)
    {}

#ifdef SYMMETRIX_SPHERICART_CUDA
    ~SphericalHarmonicsState() noexcept
    {
        try {
            calculator.release_device_resources();
        } catch (...) {
            // Destruction must remain noexcept during evaluator teardown.
        }
    }
#endif
};
