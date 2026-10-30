#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "execution_mode.hpp"
#include "implementation_registry.hpp"

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
    std::optional<std::size_t> estimated_bytes;
    std::string reason;
    bool fixed_workspace = false;
};

struct ExecutionPlanReport {
    ExecutionMode requested_algorithm = ExecutionMode::direct;
    ExecutionProfile requested_profile = ExecutionProfile::capacity;
    std::string selection_source = "request";
    std::string state = "pending";
    std::string selected_id;
    std::string selection_reason;
    std::size_t available_bytes = 0;
    std::size_t reserve_bytes = 0;
    bool boundary_attempt = false;
    std::vector<ExecutionPlanCandidate> candidates;

    // Per-operator implementation binding, additive and versioned.  The
    // whole-plan selection above says which plan runs; this says which tier of
    // each operator it runs, so a consumer can see a portable node next to a
    // generated one, and each binding's `selected_implementation_id` names the
    // implementation the execution path dispatches for that node.  Version 0
    // means the report predates tier binding: the field is reported, and the
    // binding list is empty rather than meaning "no implementation qualified".
    std::uint32_t binding_report_version = 0;
    std::vector<ImplementationBinding> bindings;
};

} // namespace symmetrix::execution
