#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

namespace symmetrix::execution {

template <typename ExecutionSpace>
inline constexpr bool host_execution_space = std::is_same_v<
    typename ExecutionSpace::memory_space, Kokkos::HostSpace>;

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

template<class ExecutionSpace>
inline constexpr bool device_execution_space =
    DeviceBackendTraits<ExecutionSpace>::is_device;
