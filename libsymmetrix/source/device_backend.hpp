#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "jit_cuda_plugin.hpp"
#include "jit_hip_plugin.hpp"

namespace symmetrix::execution {

#ifdef KOKKOS_ENABLE_CUDA
class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int requested_device);
    ~CudaDeviceGuard();
    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

private:
    int previous_device_ = -1;
    bool restore_ = false;
};
#endif

#ifdef KOKKOS_ENABLE_HIP
class HipDeviceGuard {
public:
    explicit HipDeviceGuard(int requested_device);
    ~HipDeviceGuard();
    HipDeviceGuard(const HipDeviceGuard&) = delete;
    HipDeviceGuard& operator=(const HipDeviceGuard&) = delete;

private:
    int previous_device_ = -1;
    bool restore_ = false;
};
#endif

} // namespace symmetrix::execution

struct DeviceExecutionEnvironment {
    bool available = false;
    std::string backend = "host";
    std::string execution_space;
    std::string memory_space;
    int device_ordinal = -1;
    std::string device_name;
    std::string raw_agent_target;
    std::string architecture;
    std::string target_features;
    std::string compiler_offload_target;
    int native_subgroup_width = 1;
    int compute_unit_count = 1;
    int max_team_size = 1;
    std::size_t max_shared_memory_per_block = 0;
    int compute_capability = 0;
    std::string runtime_version;
    std::string driver_version;
    std::string compiler_version;
    bool device_memory = false;
    bool aot_available = false;
    bool jit_available = false;
    bool batched_blas_available = false;
};

struct HipAgentTarget {
    std::string raw;
    std::string base_isa;
    std::string features;
};

HipAgentTarget normalize_hip_agent_target(const std::string& target);
DeviceExecutionEnvironment execution_device_execution_environment();
bool kokkos_device_sentinel();
std::pair<std::vector<double>, std::vector<double>>
kokkos_spherical_harmonics_reference(
    const std::vector<double>& coordinates, int l_max);

template<class ExecutionSpace>
struct DeviceBackendTraits {
    static constexpr const char* backend_name = "host";
    static constexpr bool is_device = false;
    static constexpr int default_logical_tile_width = 1;
};

#ifdef KOKKOS_ENABLE_CUDA
template<>
struct DeviceBackendTraits<Kokkos::Cuda> {
    using Plugin = symmetrix::execution::CudaPlugin;
    using DeviceGuard = symmetrix::execution::CudaDeviceGuard;
    using RadialSpline = SymmetrixJitCudaRadialSplineV2;
    using R1ForwardArgs = SymmetrixJitCudaR1ForwardArgsV2;
    using R1SourceArgs = SymmetrixJitCudaR1SourceArgsV2;
    using R1EdgeArgs = SymmetrixJitCudaR1EdgeArgsV2;
    using R1TiledForwardArgs = SymmetrixJitCudaR1TiledForwardArgsV2;
    using R1TiledSourceArgs = SymmetrixJitCudaR1TiledSourceArgsV2;
    using R1TiledEdgeArgs = SymmetrixJitCudaR1TiledEdgeArgsV2;
    using R1ProjectedForwardArgs = SymmetrixJitCudaR1ProjectedForwardArgsV2;
    using R1ProjectedReverseArgs = SymmetrixJitCudaR1ProjectedReverseArgsV2;
    static constexpr const char* backend_name = "cuda";
    static constexpr bool is_device = true;
    static constexpr int default_logical_tile_width = 32;
    static auto native_stream(const Kokkos::Cuda& execution_space) {
        return execution_space.cuda_stream();
    }
    static int device_ordinal(const Kokkos::Cuda& execution_space) {
        return execution_space.cuda_device();
    }
    static void check_status(std::int32_t status, const char* operation);
    static bool plugin_ready(const Plugin& plugin) noexcept {
        return static_cast<bool>(plugin) && plugin.has_v2_descriptor();
    }
    static int persistent_blocks(
        int compute_unit_count, const Plugin& plugin)
    {
        return std::max(
            1, plugin.persistent_blocks_per_compute_unit()*compute_unit_count);
    }
};
#endif

#ifdef KOKKOS_ENABLE_HIP
template<>
struct DeviceBackendTraits<Kokkos::HIP> {
    using Plugin = symmetrix::execution::HipPlugin;
    using DeviceGuard = symmetrix::execution::HipDeviceGuard;
    using RadialSpline = SymmetrixJitHipRadialSplineV1;
    using R1ForwardArgs = SymmetrixJitHipR1ForwardArgsV1;
    using R1SourceArgs = SymmetrixJitHipR1SourceArgsV1;
    using R1EdgeArgs = SymmetrixJitHipR1EdgeArgsV1;
    using R1TiledForwardArgs = SymmetrixJitHipR1TiledForwardArgsV1;
    using R1TiledSourceArgs = SymmetrixJitHipR1TiledSourceArgsV1;
    using R1TiledEdgeArgs = SymmetrixJitHipR1TiledEdgeArgsV1;
    using R1ProjectedForwardArgs = SymmetrixJitHipR1ProjectedForwardArgsV1;
    using R1ProjectedReverseArgs = SymmetrixJitHipR1ProjectedReverseArgsV1;
    static constexpr const char* backend_name = "hip";
    static constexpr bool is_device = true;
    static constexpr int default_logical_tile_width = 32;
    static auto native_stream(const Kokkos::HIP& execution_space) {
        return execution_space.hip_stream();
    }
    static int device_ordinal(const Kokkos::HIP& execution_space) {
        return execution_space.hip_device();
    }
    static void check_status(std::int32_t status, const char* operation);
    static bool plugin_ready(const Plugin& plugin) noexcept {
        return static_cast<bool>(plugin);
    }
    static int persistent_blocks(
        int compute_unit_count, const Plugin& plugin)
    {
        return std::max(
            1, plugin.persistent_blocks_per_compute_unit()*compute_unit_count);
    }
};
#endif

template<class ExecutionSpace>
inline constexpr bool device_execution_space =
    DeviceBackendTraits<ExecutionSpace>::is_device;

#ifdef KOKKOS_ENABLE_CUDA
using JitDevicePlugin = symmetrix::execution::CudaPlugin;
using ExecutionDeviceRadialSpline = DeviceBackendTraits<Kokkos::Cuda>::RadialSpline;
using ExecutionDeviceR1ForwardArgs = DeviceBackendTraits<Kokkos::Cuda>::R1ForwardArgs;
using ExecutionDeviceR1SourceArgs = DeviceBackendTraits<Kokkos::Cuda>::R1SourceArgs;
using ExecutionDeviceR1EdgeArgs = DeviceBackendTraits<Kokkos::Cuda>::R1EdgeArgs;
using ExecutionDeviceR1TiledForwardArgs =
    DeviceBackendTraits<Kokkos::Cuda>::R1TiledForwardArgs;
using ExecutionDeviceR1TiledSourceArgs =
    DeviceBackendTraits<Kokkos::Cuda>::R1TiledSourceArgs;
using ExecutionDeviceR1TiledEdgeArgs =
    DeviceBackendTraits<Kokkos::Cuda>::R1TiledEdgeArgs;
using ExecutionDeviceR1ProjectedForwardArgs =
    DeviceBackendTraits<Kokkos::Cuda>::R1ProjectedForwardArgs;
using ExecutionDeviceR1ProjectedReverseArgs =
    DeviceBackendTraits<Kokkos::Cuda>::R1ProjectedReverseArgs;
#elif defined(KOKKOS_ENABLE_HIP)
using JitDevicePlugin = symmetrix::execution::HipPlugin;
using ExecutionDeviceRadialSpline = DeviceBackendTraits<Kokkos::HIP>::RadialSpline;
using ExecutionDeviceR1ForwardArgs = DeviceBackendTraits<Kokkos::HIP>::R1ForwardArgs;
using ExecutionDeviceR1SourceArgs = DeviceBackendTraits<Kokkos::HIP>::R1SourceArgs;
using ExecutionDeviceR1EdgeArgs = DeviceBackendTraits<Kokkos::HIP>::R1EdgeArgs;
using ExecutionDeviceR1TiledForwardArgs =
    DeviceBackendTraits<Kokkos::HIP>::R1TiledForwardArgs;
using ExecutionDeviceR1TiledSourceArgs =
    DeviceBackendTraits<Kokkos::HIP>::R1TiledSourceArgs;
using ExecutionDeviceR1TiledEdgeArgs =
    DeviceBackendTraits<Kokkos::HIP>::R1TiledEdgeArgs;
using ExecutionDeviceR1ProjectedForwardArgs =
    DeviceBackendTraits<Kokkos::HIP>::R1ProjectedForwardArgs;
using ExecutionDeviceR1ProjectedReverseArgs =
    DeviceBackendTraits<Kokkos::HIP>::R1ProjectedReverseArgs;
#else
// Keep packet helper templates parseable in host-only builds. They are never
// admitted or launched without an active device backend.
using ExecutionDeviceRadialSpline = SymmetrixJitCudaRadialSplineV2;
using ExecutionDeviceR1ForwardArgs = SymmetrixJitCudaR1ForwardArgsV2;
using ExecutionDeviceR1SourceArgs = SymmetrixJitCudaR1SourceArgsV2;
using ExecutionDeviceR1EdgeArgs = SymmetrixJitCudaR1EdgeArgsV2;
using ExecutionDeviceR1TiledForwardArgs = SymmetrixJitCudaR1TiledForwardArgsV2;
using ExecutionDeviceR1TiledSourceArgs = SymmetrixJitCudaR1TiledSourceArgsV2;
using ExecutionDeviceR1TiledEdgeArgs = SymmetrixJitCudaR1TiledEdgeArgsV2;
using ExecutionDeviceR1ProjectedForwardArgs =
    SymmetrixJitCudaR1ProjectedForwardArgsV2;
using ExecutionDeviceR1ProjectedReverseArgs =
    SymmetrixJitCudaR1ProjectedReverseArgsV2;
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
static_assert(!std::is_polymorphic_v<JitDevicePlugin>);
#endif
