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

namespace {

void require_execution_host_packet(
    const bool condition,
    const std::string_view message)
{
    if (!condition)
        throw std::runtime_error(
            "Execution host plugin launch packet is invalid: "+std::string(message));
}

template <typename Precision>
SymmetrixJitHostRadialSplineV2 make_execution_host_radial_spline(
    const RadialFunctionSetKokkos<Precision>& radial)
{
    return {
        sizeof(SymmetrixJitHostRadialSplineV2),
        static_cast<std::uint32_t>(radial.edge_type_count()),
        static_cast<std::uint32_t>(radial.interval_count()),
        static_cast<std::uint32_t>(radial.function_count()),
        radial.spline_h(),
        radial.spline_x0(),
        radial.coefficient_data()};
}

void require_execution_device_packet(
    const bool condition,
    const std::string_view message)
{
    if (!condition)
        throw std::runtime_error(
            "Execution device plugin launch packet is invalid: "+std::string(message));
}

#ifdef KOKKOS_ENABLE_CUDA
using ExecutionDeviceBackend = DeviceBackendTraits<Kokkos::Cuda>;
#elif defined(KOKKOS_ENABLE_HIP)
using ExecutionDeviceBackend = DeviceBackendTraits<Kokkos::HIP>;
#endif

template <typename Precision>
ExecutionDeviceRadialSpline make_execution_device_radial_spline(
    const RadialFunctionSetKokkos<Precision>& radial)
{
    return {
        sizeof(ExecutionDeviceRadialSpline),
        static_cast<std::uint32_t>(radial.edge_type_count()),
        static_cast<std::uint32_t>(radial.interval_count()),
        static_cast<std::uint32_t>(radial.function_count()),
        radial.spline_h(),
        radial.spline_x0(),
        radial.coefficient_data()};
}

}
