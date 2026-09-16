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
#include "kernel_launch_profile.hpp"
#include "factorized_blas.hpp"

namespace {

struct KernelLaunchEnvironment {
    std::string backend;
    std::string architecture;
    int warp_width = 0;
    int compute_capability = 0;
    int multiprocessor_count = 1;
    int max_threads_per_block = 1;
    std::size_t max_shared_memory_per_block = 0;

    bool supported() const { return !backend.empty(); }
};

template <typename ExecutionSpace>
KernelLaunchEnvironment kernel_launch_environment(
    const ExecutionSpace& execution_space)
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<ExecutionSpace,Kokkos::Cuda>) {
        const auto& device = execution_space.cuda_device_prop();
        return {
            "cuda",
            "sm_"+std::to_string(device.major)+std::to_string(device.minor),
            device.warpSize,
            10*device.major+device.minor,
            device.multiProcessorCount,
            device.maxThreadsPerBlock,
            device.sharedMemPerBlock};
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<ExecutionSpace,Kokkos::HIP>) {
        const auto& device = execution_space.hip_device_prop();
        const auto target = normalize_hip_agent_target(device.gcnArchName);
        return {
            "hip",
            target.base_isa,
            device.warpSize,
            0,
            device.multiProcessorCount,
            device.maxThreadsPerBlock,
            device.sharedMemPerBlock};
    }
#endif
    if constexpr (std::is_same_v<
            typename ExecutionSpace::memory_space,Kokkos::HostSpace>)
        return {"host", "host", 0, 0, 1, 1, 0};
    return {};
}

template <typename Precision>
const symmetrix::execution::LaunchProfile* kernel_launch_profile(
    const std::string_view implementation_id,
    const KernelLaunchEnvironment& environment)
{
    return symmetrix::execution::select_launch_profile(
        implementation_id,
        environment.backend,
        std::is_same_v<Precision,float> ? "float32" : "float64",
        environment.warp_width,
        environment.compute_capability,
        environment.architecture);
}

}
