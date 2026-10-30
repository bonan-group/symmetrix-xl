#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

// Family-neutral execution mode.  ``materialized`` retains the materialized
// streamed edge representation, ``generic`` streams edges through the
// interpreter, and ``direct`` runs the prepared factorized execution.  The
// mode strings are the cross-language contract with the Python calculators, so
// every accepted spelling, error text, and canonical name below is fixed by
// that interface rather than by this header's naming.
enum class ExecutionMode {
    materialized,
    generic,
    direct,
};

inline ExecutionMode parse_execution_mode(std::string_view mode)
{
    if (mode == "materialized")
        return ExecutionMode::materialized;
    if (mode == "generic" || mode == "non-compiled" || mode == "all_interactions")
        return ExecutionMode::generic;
    if (mode == "direct" || mode == "factorized" || mode == "direct_streamed")
        return ExecutionMode::direct;
    if (mode == "receiver_factorized")
        throw std::invalid_argument(
            "streamed_edges='receiver_factorized' was removed; use "
            "streamed_edges='direct'.");
    throw std::invalid_argument(
        "streamed_edges must be one of 'materialized', 'non-compiled', or "
        "'direct' (compatibility aliases: 'generic', 'all_interactions', "
        "'factorized', 'direct_streamed').");
}

inline std::string execution_mode_name(ExecutionMode mode)
{
    switch (mode) {
    case ExecutionMode::materialized:
        return "materialized";
    case ExecutionMode::generic:
        return "generic";
    case ExecutionMode::direct:
        return "direct";
    }
    throw std::logic_error("Invalid streamed-edge mode.");
}

constexpr bool uses_prepared_execution(const ExecutionMode mode)
{
    return mode == ExecutionMode::direct;
}

constexpr bool uses_direct_execution(const ExecutionMode mode)
{
    return mode == ExecutionMode::direct;
}

constexpr bool admits_low_memory(const ExecutionMode mode)
{
    return uses_direct_execution(mode);
}
