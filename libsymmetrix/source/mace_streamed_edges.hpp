#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

enum class MACEStreamedEdgesMode {
    materialized,
    generic,
    direct,
};

inline MACEStreamedEdgesMode parse_mace_streamed_edges_mode(std::string_view mode)
{
    if (mode == "materialized")
        return MACEStreamedEdgesMode::materialized;
    if (mode == "generic" || mode == "non-compiled" || mode == "all_interactions")
        return MACEStreamedEdgesMode::generic;
    if (mode == "direct" || mode == "factorized" || mode == "direct_streamed")
        return MACEStreamedEdgesMode::direct;
    if (mode == "receiver_factorized")
        throw std::invalid_argument(
            "streamed_edges='receiver_factorized' was removed; use "
            "streamed_edges='direct'.");
    throw std::invalid_argument(
        "streamed_edges must be one of 'materialized', 'non-compiled', or "
        "'direct' (compatibility aliases: 'generic', 'all_interactions', "
        "'factorized', 'direct_streamed').");
}

inline std::string mace_streamed_edges_mode_name(MACEStreamedEdgesMode mode)
{
    switch (mode) {
    case MACEStreamedEdgesMode::materialized:
        return "materialized";
    case MACEStreamedEdgesMode::generic:
        return "generic";
    case MACEStreamedEdgesMode::direct:
        return "direct";
    }
    throw std::logic_error("Invalid streamed-edge mode.");
}

constexpr bool mace_uses_prepared_execution(const MACEStreamedEdgesMode mode)
{
    return mode == MACEStreamedEdgesMode::direct;
}

constexpr bool mace_uses_direct_execution(const MACEStreamedEdgesMode mode)
{
    return mode == MACEStreamedEdgesMode::direct;
}

constexpr bool mace_uses_receiver_factorization(const MACEStreamedEdgesMode mode)
{
    return false;
}

constexpr bool mace_admits_low_memory(const MACEStreamedEdgesMode mode)
{
    return mace_uses_direct_execution(mode);
}
