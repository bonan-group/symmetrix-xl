#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace symmetrix::execution::long_range {

enum class LongRangeSolverRequest {
    automatic,
    realspace,
    periodic,
    direct_reciprocal,
    pme,
};

enum class LongRangeSolver {
    unresolved,
    realspace,
    direct_reciprocal,
    pme,
};

struct LongRangeSolverDecision {
    LongRangeSolverRequest requested = LongRangeSolverRequest::automatic;
    LongRangeSolver resolved = LongRangeSolver::unresolved;
    bool accepted = false;
    std::string reason;
    std::string error;
};

struct LongRangeSolverCapabilities {
    bool pme_fft_provider_available = false;
    bool pme_backend_qualified = false;
    std::size_t automatic_nodes = 0;
    std::size_t automatic_reciprocal_modes = 0;
};

inline constexpr std::size_t automatic_pme_atom_mode_pair_threshold =
    40'000'000;

inline std::size_t automatic_pme_required_modes(const std::size_t nodes)
{
    if (nodes == 0)
        return 0;
    const std::size_t quotient = automatic_pme_atom_mode_pair_threshold / nodes;
    const std::size_t remainder = automatic_pme_atom_mode_pair_threshold % nodes;
    return quotient + (remainder != 0 ? 1 : 0);
}

inline std::size_t automatic_pme_required_nodes(const std::size_t modes)
{
    if (modes == 0)
        return 0;
    const std::size_t quotient = automatic_pme_atom_mode_pair_threshold / modes;
    const std::size_t remainder = automatic_pme_atom_mode_pair_threshold % modes;
    return quotient + (remainder != 0 ? 1 : 0);
}

inline bool automatic_pme_size_admitted(
    const LongRangeSolverCapabilities& capabilities)
{
    const std::size_t modes = capabilities.automatic_reciprocal_modes;
    if (modes == 0)
        return false;
    return capabilities.automatic_nodes
        >= automatic_pme_required_nodes(modes);
}

inline std::string_view to_string(const LongRangeSolverRequest request)
{
    switch (request) {
    case LongRangeSolverRequest::automatic: return "auto";
    case LongRangeSolverRequest::realspace: return "realspace";
    case LongRangeSolverRequest::periodic: return "periodic";
    case LongRangeSolverRequest::direct_reciprocal:
        return "direct_reciprocal";
    case LongRangeSolverRequest::pme: return "pme";
    }
    throw std::logic_error("unknown long-range solver request.");
}

inline std::string_view to_string(const LongRangeSolver solver)
{
    switch (solver) {
    case LongRangeSolver::unresolved: return "unresolved";
    case LongRangeSolver::realspace: return "realspace";
    case LongRangeSolver::direct_reciprocal: return "direct_reciprocal";
    case LongRangeSolver::pme: return "pme";
    }
    throw std::logic_error("unknown resolved long-range solver.");
}

inline LongRangeSolverRequest parse_long_range_solver_request(
    const std::string_view token)
{
    if (token == "auto")
        return LongRangeSolverRequest::automatic;
    if (token == "realspace")
        return LongRangeSolverRequest::realspace;
    if (token == "periodic")
        return LongRangeSolverRequest::periodic;
    if (token == "direct_reciprocal")
        return LongRangeSolverRequest::direct_reciprocal;
    if (token == "pme")
        return LongRangeSolverRequest::pme;
    throw std::invalid_argument(
        "polar electrostatics solver must be one of 'auto', 'realspace', "
        "'periodic', 'direct_reciprocal', or 'pme'.");
}

inline LongRangeSolverDecision resolve_long_range_solver(
    const LongRangeSolverRequest request,
    const std::array<int, 3>& pbc,
    const int source_max_l,
    const int receiver_max_l,
    const LongRangeSolverCapabilities capabilities = {})
{
    LongRangeSolverDecision decision;
    decision.requested = request;
    const bool any_periodic = pbc[0] != 0 || pbc[1] != 0 || pbc[2] != 0;

    switch (request) {
    case LongRangeSolverRequest::automatic:
        if (any_periodic) {
            const bool pme_admitted =
                pbc == std::array<int, 3>{1, 1, 1}
                && source_max_l <= 1 && receiver_max_l <= 1
                && capabilities.pme_fft_provider_available
                && capabilities.pme_backend_qualified
                && automatic_pme_size_admitted(capabilities);
            decision.resolved = pme_admitted
                ? LongRangeSolver::pme
                : LongRangeSolver::direct_reciprocal;
            decision.reason = pme_admitted
                ? "automatic_pme_atom_mode_pairs"
                : "periodic_input_delivery_1";
        } else {
            decision.resolved = LongRangeSolver::realspace;
            decision.reason = "fully_nonperiodic_input";
        }
        break;
    case LongRangeSolverRequest::realspace:
        if (any_periodic) {
            decision.reason = "realspace_requires_fully_nonperiodic_input";
            decision.error =
                "polar_electrostatics='realspace' requires fully "
                "non-periodic input (PBC FFF).";
            return decision;
        }
        decision.resolved = LongRangeSolver::realspace;
        decision.reason = "explicit_realspace";
        break;
    case LongRangeSolverRequest::periodic:
        decision.resolved = LongRangeSolver::direct_reciprocal;
        decision.reason = "periodic_compatibility_override";
        break;
    case LongRangeSolverRequest::direct_reciprocal:
        decision.resolved = LongRangeSolver::direct_reciprocal;
        decision.reason = "explicit_direct_reciprocal";
        break;
    case LongRangeSolverRequest::pme:
        if (pbc != std::array<int, 3>{1, 1, 1}) {
            decision.reason = "pme_requires_fully_periodic_input";
            decision.error =
                "polar_electrostatics='pme' requires fully periodic input "
                "(PBC TTT).";
            return decision;
        }
        if (source_max_l > 1 || receiver_max_l > 1) {
            decision.reason = "pme_supports_l_at_most_1";
            decision.error =
                "POLAR particle-mesh electrostatics supports source and "
                "receiver angular momentum l<=1.";
            return decision;
        }
        if (!capabilities.pme_fft_provider_available) {
            decision.reason = "pme_fft_provider_unavailable";
            decision.error =
                "polar_electrostatics='pme' is unavailable because this "
                "build has no particle-mesh FFT provider.";
            return decision;
        }
        if (!capabilities.pme_backend_qualified) {
            decision.reason = "pme_backend_not_qualified";
            decision.error =
                "polar_electrostatics='pme' is currently qualified only on "
                "CUDA backends.";
            return decision;
        }
        decision.resolved = LongRangeSolver::pme;
        decision.reason = "explicit_pme";
        break;
    }

    if (decision.resolved == LongRangeSolver::realspace
        && (source_max_l > 1 || receiver_max_l > 1)) {
        decision.reason = "realspace_supports_l_at_most_1";
        decision.error =
            "real-space POLAR electrostatics supports source and receiver "
            "angular momentum l<=1.";
        return decision;
    }
    decision.accepted = true;
    return decision;
}

} // namespace symmetrix::execution::long_range
