#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "mace_streamed_edges.hpp"

namespace symmetrix::execution {

enum class ExecutionProfile {
    capacity,
    speed,
};

inline ExecutionProfile parse_execution_profile(const std::string& value)
{
    if (value == "capacity")
        return ExecutionProfile::capacity;
    if (value == "speed")
        return ExecutionProfile::speed;
    throw std::invalid_argument(
        "execution_profile must be 'capacity' or 'speed'.");
}

inline const char* execution_profile_name(const ExecutionProfile profile)
{
    return profile == ExecutionProfile::capacity ? "capacity" : "speed";
}

struct ExecutionPlanCandidate {
    std::string id;
    bool qualified = false;
    std::size_t estimated_bytes = 0;
    std::string reason;
};

struct ExecutionPlanReport {
    MACEStreamedEdgesMode requested_algorithm = MACEStreamedEdgesMode::direct;
    ExecutionProfile requested_profile = ExecutionProfile::capacity;
    std::string selection_source = "request";
    std::string state = "pending";
    std::string selected_id;
    std::string selection_reason;
    std::size_t available_bytes = 0;
    std::size_t reserve_bytes = 0;
    bool boundary_attempt = false;
    std::vector<ExecutionPlanCandidate> candidates;
};

} // namespace symmetrix::execution
